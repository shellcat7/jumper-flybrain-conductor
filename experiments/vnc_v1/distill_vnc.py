"""Distil an orchestrating cord: the official gait when nothing is wrong, a trouble gait when it is.

The student is the VNC actor (`vnc_model.VNCModel`, the official MLP frozen). Which gait is
right for which trouble comes from the 10/23 evaluations (speed ratio, official / mlp-b / mlp-d,
both MLPs fine-tuned from the official under the v1 troubles):

- a dead front leg: mlp-b (dead LF in a corner 0.44 / 0.63 / 0.59, dead RF running 0.28 / 0.40 / 0.41);
- a dead rear leg: mlp-b (dead RR running 0.49 / 0.57 / 0.57, in a corner 0.46 / 0.58 / 0.46);
- foot friction under --grip-switch: mlp-d (mu 0.06 in a corner 0.66 / 0.75 / 0.76, mu 0.03
  running 0.95 / 0.92 / 1.02);
- otherwise the official MLP: on flat ground (walking 0.92 / 0.70 / 0.79) and with a dead middle
  leg (dead RM walking 0.53 / 0.44 / 0.46).

The cord sees only the official MLP's observations, never the trouble flags, so it has to
recognise a dead leg or a slip from its leg sensors. Two ways to use that:

- `--mode gate` (the director): the experts run beside the official MLP, frozen, and a readout
  of the cord's ascending and descending neurons picks among them (`VNCModel(experts=...)`). The
  readout is fitted to the right program by cross-entropy.
- `--mode residual`: the cord's own residual is fitted to (right gait's action - official
  action) by squared error; the experts serve only as teachers.

DAgger in both: the student drives (its own action plus a little noise), the right program is
labelled at every state it visits, and the loss is backpropagated through the cord over the
window. This replaced PPO fine-tuning after rounds 1-6 on 10/24-25, where the cord either drifted
from the official gait or, held by a pull toward zero, learnt nothing.

    python experiments/vnc_v1/distill_vnc.py --mode gate --packet runs/vnc-v1/packet-1/vnc_ge5.npz \
        --layout runs/vnc-v1/packet-1/layout.json \
        --experts runs/reflex-v1/resume/mlp-d-75100-weights.pt runs/reflex-v1/resume/mlp-b-75200-weights.pt \
        --front 2 --rear 2 --grip 1 --num-envs 4096 --iterations 1000 --learn-synapses --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
import troubles  # noqa: E402

TASK = "jumper.posture"
FRONT, MIDDLE, REAR = (0, 1), (2, 3), (4, 5)  # troubles.LEGS = LF, RF, LM, RM, LR, RR


def actor_kwargs(agent_cfg) -> dict:
    """The official actor's constructor arguments, as mjlab's runner passes them."""
    kw = asdict(agent_cfg)["actor"]
    kw.pop("class_name", None)
    for opt in ("cnn_cfg", "distribution_cfg"):
        if kw.get(opt) is None:
            kw.pop(opt, None)
    if kw.get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            kw.pop(opt, None)
    return kw


def labels(env, foot_gids, args):
    """The right program per robot (0 = official, k = the k-th expert), from the trouble flags.

    With `--slow-official V`, low grip and a dead rear leg keep the official gait while the
    command's effective speed (|v| + 0.2 m x |wz|, the cadence law's) is under V: walking, the
    experts are no better there (grip 0.06 walk 0.96 / 0.97, dead RR walk 0.56 / 0.48), and on
    slippery ground at a walk nothing slips, so the cord cannot see the trouble and hedged.
    dgate-s1 (every low-grip step labelled mlp-d) walked at 0.90 on flat ground at iteration 150
    against the official's 0.93.
    """
    import torch

    dead = troubles._dead_flags(env)
    grip = env.sim.model.geom_friction[:, foot_gids, 0].min(dim=-1).values
    out = torch.zeros_like(dead)
    fast = torch.ones_like(dead, dtype=torch.bool)
    if args.slow_official:
        cmd = env.command_manager.get_command("twist")
        fast = cmd[:, :2].norm(dim=-1) + 0.2 * cmd[:, 2].abs() >= args.slow_official
    out = torch.where((grip < args.grip_switch) & fast, torch.full_like(out, args.grip), out)
    def pick(side, default):
        return default if side is None else side

    # Per leg: (leg, program, gated). A gated program runs only at speed (`fast`); slower, the
    # official gait. The middle legs' programs are gated (mlpLM-s1 walks a dead LM worse than the
    # official and corners it better); the rear legs' unless --rear-always.
    for leg, k, gated in ((FRONT[0], pick(args.front_left, args.front), False),
                          (FRONT[1], pick(args.front_right, args.front), False),
                          (REAR[0], pick(args.rear_left, args.rear), not args.rear_always),
                          (REAR[1], pick(args.rear_right, args.rear), not args.rear_always),
                          (MIDDLE[0], pick(args.middle_left, 0), True),
                          (MIDDLE[1], pick(args.middle_right, 0), True)):
        hit = dead == leg
        if gated:
            out = torch.where(hit, torch.where(fast, torch.full_like(out, k), torch.zeros_like(out)), out)
        else:
            out = torch.where(hit, torch.full_like(out, k), out)
    # A dead middle leg below the speed gate: its slow-walk program, where there is one.
    for leg, k in ((MIDDLE[0], args.middle_left_slow), (MIDDLE[1], args.middle_right_slow)):
        if k:
            out = torch.where((dead == leg) & ~fast, torch.full_like(out, k), out)
    # ... and turning at speed, its cornering program (10/26: the slow-walk program mlpLMslow-s1
    # walks a dead LM at 0.63 and runs it at 0.72, against mlpLM-s1's 0.42 / 0.62, but corners it
    # at 0.88 against 0.94).
    if args.middle_left_turn or args.middle_right_turn:
        cmd = env.command_manager.get_command("twist")
        turning = fast & (cmd[:, 2].abs() >= args.turn_rate)
        for leg, k in ((MIDDLE[0], args.middle_left_turn), (MIDDLE[1], args.middle_right_turn)):
            if k:
                out = torch.where((dead == leg) & turning, torch.full_like(out, k), out)
    # Two legs that died together (`--kill-legs` pairs): the pair's own program, at any speed.
    dead2 = troubles._second_flags(env)
    for spec in args.pairs:
        legs, k = spec.split(":")
        first, second = (troubles.LEGS.index(name) for name in legs.split("+"))
        out = torch.where((dead == first) & (dead2 == second), torch.full_like(out, int(k)), out)
    if args.sprint:  # a fast command with nothing wrong: the sprint gait
        cmd = env.command_manager.get_command("twist")
        sprinting = (cmd[:, :2].norm(dim=-1) >= args.sprint_speed) & (dead < 0) & (grip >= args.grip_switch)
        out = torch.where(sprinting, torch.full_like(out, args.sprint), out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", default="gate", choices=["gate", "residual"])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--steps", type=int, default=48, help="rollout window per iteration (BPTT length)")
    p.add_argument("--chunk", type=int, default=1024, help="robots per backward pass")
    p.add_argument("--epochs", type=int, default=1, help="passes over each rollout")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--packet", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--experts", nargs="+", required=True, help="the trouble gaits' checkpoints (actor_state_dict)")
    p.add_argument("--front", type=int, default=2, help="program for a dead front leg (0 = official)")
    p.add_argument("--front-left", type=int, default=None, help="program for a dead left-front leg (default --front)")
    p.add_argument("--front-right", type=int, default=None, help="program for a dead right-front leg (default --front)")
    p.add_argument("--adapters", nargs="*", default=[], choices=["five_foot", "five_foot_mirror"],
                   help="programs from other official tasks, after the experts (gate mode)")
    p.add_argument("--adapter-vmax", type=float, default=0.5,
                   help="the adapted programs' linear command bound (five_foot's training ceiling is 0.5 m/s)")
    p.add_argument("--rear", type=int, default=2, help="program for a dead rear leg")
    p.add_argument("--rear-left", type=int, default=None, help="program for a dead left-rear leg (default --rear)")
    p.add_argument("--rear-right", type=int, default=None, help="program for a dead right-rear leg (default --rear)")
    p.add_argument("--rear-always", action="store_true",
                   help="the rear legs' programs at any speed (otherwise only above --slow-official)")
    p.add_argument("--middle-left", type=int, default=None, help="program for a dead left-middle leg (default 0)")
    p.add_argument("--middle-right", type=int, default=None, help="program for a dead right-middle leg (default 0)")
    p.add_argument("--middle-left-slow", type=int, default=0,
                   help="program for a dead left-middle leg below --slow-official (0: the official)")
    p.add_argument("--middle-right-slow", type=int, default=0,
                   help="program for a dead right-middle leg below --slow-official (0: the official)")
    p.add_argument("--middle-left-turn", type=int, default=0,
                   help="program for a dead left-middle leg turning at speed (0: as --middle-left)")
    p.add_argument("--middle-right-turn", type=int, default=0,
                   help="program for a dead right-middle leg turning at speed (0: as --middle-right)")
    p.add_argument("--turn-rate", type=float, default=0.5, help="rad/s of commanded turn that counts as turning")
    p.add_argument("--kill-legs", default="",
                   help="comma list of the legs that may die, pairs joined with + (default: any one leg)")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="programs for pairs that died together, e.g. LM+LR:8 (order as in --kill-legs)")
    p.add_argument("--obs-mirror", default="", help="mirror_matrix.py's npz, for experts named <file>#mirror")
    p.add_argument("--grip", type=int, default=1, help="program for foot friction under --grip-switch")
    p.add_argument("--sprint", type=int, default=0, help="program for a command of --sprint-speed or more (0: none)")
    p.add_argument("--sprint-speed", type=float, default=1.0)
    p.add_argument("--ladder-top", type=float, default=None,
                   help="raise the command ladder's top rung to this speed (as train_mlp.py --sprint)")
    p.add_argument("--grip-switch", type=float, default=0.3)
    p.add_argument("--slow-official", type=float, default=0.0,
                   help="below this effective command speed (m/s), low grip and a dead rear leg keep the official")
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--residual-scale", type=float, default=2.0)
    p.add_argument("--huber", type=float, default=0.0,
                   help="residual mode: Huber loss with this delta instead of squared error (dres-s1's "
                        "squared error spiked 40-fold on outlier targets around iteration 300)")
    p.add_argument("--learn-synapses", action="store_true")
    p.add_argument("--brain-hidden", type=int, default=64)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--grip-p-slip", type=float, default=0.3)
    p.add_argument("--kill-prob", type=float, default=0.10)
    p.add_argument("--noise", type=float, default=0.05, help="action noise while the student drives")
    p.add_argument("--init", default="", help="start the student from this checkpoint (same design)")
    p.add_argument("--init-map", default="",
                   help="comma list: where each of --init's gate rows goes (default: the same index)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert args.num_envs % args.chunk == 0, "--num-envs must be a multiple of --chunk"
    n_prog = len(args.experts) + len(args.adapters)
    assert all(k is None or 0 <= k <= n_prog
               for k in (args.front, args.front_left, args.front_right, args.rear, args.rear_left, args.rear_right,
                         args.middle_left, args.middle_right, args.middle_left_slow, args.middle_right_slow,
                         args.middle_left_turn, args.middle_right_turn,
                         args.grip, args.sprint, *(int(spec.split(":")[1]) for spec in args.pairs)))
    # VNCModel builds a missing expert empty (for evaluation, where the checkpoint brings the
    # weights); here it would teach a random gait.
    missing = [path for path in args.experts if not Path(path.removesuffix("#mirror")).exists()]
    assert not missing, f"expert checkpoints not found: {missing}"

    import torch
    import torch.nn.functional as F

    import tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend=args.backend, device=args.device, num_envs=args.num_envs)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.rl import RslRlVecEnvWrapper

    from tasks.jumper.common.constants import FOOT_GEOMS
    from tasks.jumper.common.runner import restore_curriculum
    from vnc_model import VNCModel

    torch.manual_seed(args.seed)
    cfg = tasks.load_env_cfg(TASK)
    cfg.seed = args.seed
    cfg.scene.num_envs = res.num_envs
    added = troubles.add_troubles(cfg, grip_low=0.02, shove=None, kill_prob=args.kill_prob,
                                  grip_p_slip=args.grip_p_slip,
                                  kill_legs=tuple(args.kill_legs.split(",")) if args.kill_legs else None)
    added["guard"] = troubles.add_stability_guard(cfg)
    if args.ladder_top:
        levels = list(cfg.curriculum["command"].params["levels"])
        levels[-1] = args.ladder_top
        cfg.curriculum["command"].params["levels"] = tuple(levels)
        added["ladder_top"] = args.ladder_top
    agent_cfg = tasks.load_agent_cfg(TASK)
    gate = args.mode == "gate"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vnc-config.json").write_text(json.dumps({
        "task": TASK, "method": f"distill-{args.mode}", "base_checkpoint": args.checkpoint,
        "experts": args.experts if gate else [], "teachers": args.experts,
        "adapters": args.adapters if gate else [], "adapter_vmax": args.adapter_vmax,
        "programs": {"front": args.front, "front_left": args.front_left, "front_right": args.front_right,
                     "rear": args.rear, "rear_left": args.rear_left, "rear_right": args.rear_right,
                     "rear_always": args.rear_always, "grip": args.grip, "middle": 0,
                     "middle_left": args.middle_left, "middle_right": args.middle_right,
                     "middle_left_slow": args.middle_left_slow, "middle_right_slow": args.middle_right_slow,
                     "middle_left_turn": args.middle_left_turn, "middle_right_turn": args.middle_right_turn,
                     "turn_rate": args.turn_rate,
                     "pairs": args.pairs,
                     "sprint": args.sprint, "sprint_speed": args.sprint_speed},
        "obs_mirror": args.obs_mirror or None,
        "grip_switch": args.grip_switch, "slow_official": args.slow_official,
        "packet": args.packet, "layout": args.layout, "width": 1,
        "residual_scale": args.residual_scale, "shuffle_seed": args.shuffle_seed,
        "brain_hidden": args.brain_hidden, "learn_synapses": args.learn_synapses, "command_readout": False,
        "escape": False, "num_envs": res.num_envs, "iterations": args.iterations, "steps": args.steps,
        "chunk": args.chunk, "epochs": args.epochs, "lr": args.lr, "noise": args.noise, "seed": args.seed,
        "init": args.init or None, "huber": args.huber, "troubles": added, "started_unix": time.time(),
    }, indent=2) + "\n")

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    troubles.clip_rewards(env, added["guard"]["reward_clip"])
    try:
        dev = env.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        official = torch.load(args.checkpoint, weights_only=False, map_location=dev)
        for line in restore_curriculum(env, (official.get("infos") or {}).get("curriculum")):
            print(f"[distill] curriculum {line}")
        # Staggered episodes, as PPO's first rollout has them (init_at_random_ep_len).
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
        obs = wrapped.get_observations()
        groups = asdict(agent_cfg)["obs_groups"]
        n_act = wrapped.num_actions
        # In residual mode the student carries no experts; a separate model holds them.
        student = VNCModel(obs, groups, "actor", n_act, **actor_kwargs(agent_cfg), packet=args.packet,
                           layout=args.layout, base_checkpoint=args.checkpoint,
                           residual_scale=args.residual_scale, shuffle_seed=args.shuffle_seed,
                           brain_hidden=args.brain_hidden, learn_synapses=args.learn_synapses,
                           experts=args.experts if gate else (),
                           adapters=args.adapters if gate else (), adapter_vmax=args.adapter_vmax,
                           obs_mirror=args.obs_mirror or None).to(dev)
        if gate:
            programs = student.programs
        else:
            holder = VNCModel(obs, groups, "actor", n_act, **actor_kwargs(agent_cfg), packet=args.packet,
                              layout=args.layout, base_checkpoint=args.checkpoint,
                              experts=args.experts).to(dev)
            programs = holder.programs
        if args.init:
            # Everything whose shape matches: with more programs than the checkpoint had, the
            # cord and its readout's rows for the shared programs carry over (to the rows
            # --init-map names, by default the same ones); the new rows start at zero weight
            # and the checkpoint's bias for the official minus 3.
            src = torch.load(args.init, weights_only=False, map_location=dev)["actor_state_dict"]
            src = {k: v for k, v in src.items() if not k.startswith("experts.")}  # this run's own files
            own = student.state_dict()
            skipped = [k for k in src if k in own and own[k].shape != src[k].shape]
            for k in skipped:
                if k.startswith("gate_out."):
                    n = src[k].shape[0]
                    dst = [int(i) for i in args.init_map.split(",")] if args.init_map else list(range(n))
                    assert len(dst) == n and len(set(dst)) == n, f"--init-map needs {n} distinct rows"
                    new = [i for i in range(own[k].shape[0]) if i not in dst]
                    own[k][dst] = src[k]
                    if k.endswith("bias"):
                        own[k][new] = src[k][0] - 3.0
                    else:
                        own[k][new] = 0.0
                    src[k] = own[k]
            student.load_state_dict({k: v for k, v in src.items() if k in own and own[k].shape == v.shape},
                                    strict=False)
            print(f"[distill] student from {args.init}; resized {skipped}", flush=True)
        params = [q for q in student.parameters() if q.requires_grad]
        opt = torch.optim.Adam(params, lr=args.lr)
        print(f"[distill] mode {args.mode}; cord {student.n} nodes; trainable {sum(q.numel() for q in params)}; "
              f"experts {args.experts}", flush=True)

        robot = env.scene["robot"]
        feet = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
        feet.resolve(env.scene)
        foot_gids = robot.indexing.geom_ids[feet.geom_ids]
        b = res.num_envs
        h = torch.zeros(b, student.n * student.width, device=dev)
        t0 = time.time()
        for it in range(args.iterations):
            xs, targets, dones = [], [], []
            h0 = h.detach().clone()
            stats = torch.zeros(4, device=dev)  # trouble share, right on nominal, right on trouble, |res|
            with torch.no_grad():
                for _ in range(args.steps):
                    x = torch.cat([obs[g] for g in student.obs_groups], dim=-1)
                    r, h, _ = student._step(student.obs_normalizer(x), h, None)
                    acts = programs(x)  # programs x robots x actions
                    lab = labels(env, foot_gids, args)
                    trouble = lab > 0
                    if gate:
                        logits = student.gate(h)
                        action = torch.einsum("kba,bk->ba", acts, torch.softmax(logits, dim=-1))
                        target = lab
                        right = logits.argmax(-1) == lab
                    else:
                        action = acts[0] + r
                        target = acts.gather(0, lab.view(1, -1, 1).expand(1, -1, acts.shape[-1]))[0] - acts[0]
                        right = (r - target).abs().mean(-1) < target.abs().mean(-1).clamp_min(1e-3)
                    stats += torch.stack([trouble.float().mean(),
                                          (right & ~trouble).float().sum() / (~trouble).sum().clamp_min(1),
                                          (right & trouble).float().sum() / trouble.sum().clamp_min(1),
                                          r.abs().mean()])
                    obs, _, done, _ = wrapped.step(action + args.noise * torch.randn_like(action))
                    student.reset(done)  # the adapted programs' per-robot state
                    xs.append(x)
                    targets.append(target)
                    dones.append(done.float())
                    h = h * (1.0 - done.float())[:, None]
            losses = []
            for _ in range(args.epochs):
                for start in torch.randperm(b // args.chunk).tolist():
                    sl = slice(start * args.chunk, (start + 1) * args.chunk)
                    hc = h0[sl]
                    loss = 0.0
                    for t in range(args.steps):
                        rc, hc, _ = student._step(student.obs_normalizer(xs[t][sl]), hc, None)
                        if gate:
                            loss = loss + F.cross_entropy(student.gate(hc), targets[t][sl])
                        elif args.huber:
                            loss = loss + F.huber_loss(rc, targets[t][sl], delta=args.huber)
                        else:
                            loss = loss + ((rc - targets[t][sl]) ** 2).mean()
                        hc = hc * (1.0 - dones[t][sl])[:, None]
                    loss = loss / args.steps
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    losses.append(float(loss))
            s = (stats / args.steps).tolist()
            print(f"[distill] it {it} loss {sum(losses) / len(losses):.5f} trouble {s[0]:.3f} "
                  f"right: nominal {s[1]:.3f} trouble {s[2]:.3f} |res| {s[3]:.4f} {time.time() - t0:.1f}s",
                  flush=True)
            if it % args.save_every == 0 or it == args.iterations - 1:
                torch.save({"actor_state_dict": student.state_dict(), "iter": it, "infos": None},
                           out / f"model_{it}.pt")
        print(f"VNC_DISTILL_DONE {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
