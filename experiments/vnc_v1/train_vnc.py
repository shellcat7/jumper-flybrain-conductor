"""Train the VNC residual under the reflex troubles (same troubles as the MLP controls).

The actor is `vnc_model.VNCModel`: the official posture MLP, frozen, plus the fly
cord's correction, read out per leg and starting at zero. The critic is the
official critic, warm-started. PPO settings are the official task's, except that
rsl_rl refuses symmetry augmentation with a recurrent actor, so it is off here
(the MLP controls had it on). The command-curriculum levels are restored from the
official checkpoint, as the MLP controls' resume does.

    python experiments/vnc_v1/train_vnc.py --packet runs/vnc-v1/packet-1/vnc_ge5.npz \
        --layout runs/vnc-v1/packet-1/layout.json --num-envs 4096 --iterations 2000 --out ...
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


def load_padded(module, state: dict) -> None:
    """Load `state` into `module` where an input grew (the critic with the escape task's
    looming signal): the official weights fill the leading slice, the new input columns
    start at zero, so the critic starts as the official one."""
    own = module.state_dict()
    for key, value in state.items():
        if key not in own:
            continue
        if own[key].shape == value.shape:
            own[key] = value
            continue
        own[key][tuple(slice(0, s) for s in value.shape)] = value
        if own[key].dim() == 2 and own[key].shape[1] > value.shape[1]:
            own[key][:, value.shape[1]:] = 0.0
    module.load_state_dict(own)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--packet", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--residual-scale", type=float, default=0.5)
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="the shuffled-wiring control: permute every synapse's source with this seed")
    p.add_argument("--grip-low", type=float, default=0.02)
    p.add_argument("--shove", type=float, default=1.5)
    p.add_argument("--kill-prob", type=float, default=0.15)
    p.add_argument("--grip-p-slip", type=float, default=None,
                   help="troubles v2: grip nominal except with this probability (see troubles.py)")
    p.add_argument("--official-shove", action="store_true", help="troubles v2: keep the official shoves")
    p.add_argument("--residual-l2", type=float, default=0.0,
                   help="pull the residual toward zero: adds this times mean(sum of squares) to the loss")
    p.add_argument("--brain-hidden", type=int, default=0,
                   help="brain stub: ascending neurons -> this many units -> descending neurons (0: off)")
    p.add_argument("--desired-kl", type=float, default=None,
                   help="PPO's adaptive-step target (official 0.01); smaller drifts less per update")
    p.add_argument("--learn-synapses", action="store_true",
                   help="learn every synapse's magnitude from the fly's weight, keeping its sign")
    p.add_argument("--command-readout", action="store_true",
                   help="descending neurons -> offsets on the official MLP's commands")
    p.add_argument("--no-guard", action="store_true",
                   help="leave out the blow-up terminations and the per-step reward clip")
    p.add_argument("--command-l2", type=float, default=0.0,
                   help="pull the descending commands toward zero, in units of their scale")
    p.add_argument("--mlp-anchor", type=float, default=None,
                   help="the hybrid: the official MLP learns too, pulled toward its weights by this x (w - w0)^2")
    p.add_argument("--escape", action="store_true",
                   help="the escape task: balls thrown at the robot, a looming signal for the cord, a hit penalty")
    p.add_argument("--sprint", type=float, default=None,
                   help="raise the forward-speed command ceiling to this (m/s) after the curriculum restore")
    p.add_argument("--critic-warmup", type=int, default=0,
                   help="first iterations with the actor frozen, so the critic (warm-started from the "
                        "official, which never met a trouble) learns the troubles before the cord follows "
                        "its advantages")
    p.add_argument("--mini-batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch

    import tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend=args.backend, device=args.device, num_envs=args.num_envs)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    import rsl_rl.algorithms.ppo as ppo_module
    from tasks.jumper.common.runner import restore_curriculum
    from vnc_model import FirstStepHiddenStorage, VNCModel

    ppo_module.RolloutStorage = FirstStepHiddenStorage  # keep one copy of the cord, not 48

    cfg = tasks.load_env_cfg(TASK)
    cfg.seed = args.seed
    cfg.scene.num_envs = res.num_envs
    added = troubles.add_troubles(cfg, grip_low=args.grip_low, shove=None if args.official_shove else args.shove,
                                  kill_prob=args.kill_prob, grip_p_slip=args.grip_p_slip)
    if not args.no_guard:
        added["guard"] = troubles.add_stability_guard(cfg)
    if args.sprint:
        # The command ladder's top rung (where the official checkpoint sits) is the speed
        # ceiling, 0.8 m/s; raising it lets the commands ask for more than the official
        # gait was trained to give.
        levels = list(cfg.curriculum["command"].params["levels"])
        levels[-1] = args.sprint
        cfg.curriculum["command"].params["levels"] = tuple(levels)
        added["sprint"] = args.sprint
    if args.escape:
        import escape
        added["escape"] = escape.add_escape(cfg)
    agent_cfg = tasks.load_agent_cfg(TASK)
    agent_cfg.logger = "tensorboard"
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    kwargs = asdict(agent_cfg)
    kwargs["actor"].update(class_name=VNCModel, packet=args.packet, layout=args.layout,
                           base_checkpoint=args.checkpoint, width=args.width,
                           residual_scale=args.residual_scale, shuffle_seed=args.shuffle_seed,
                           residual_l2=args.residual_l2, brain_hidden=args.brain_hidden,
                           learn_synapses=args.learn_synapses, command_readout=args.command_readout,
                           mlp_anchor=args.mlp_anchor, command_l2=args.command_l2)
    if args.escape:
        kwargs["obs_groups"] = {"actor": ["actor", "threat"], "critic": ["critic", "threat"]}
    kwargs["algorithm"]["symmetry_cfg"] = None
    kwargs["algorithm"]["num_mini_batches"] = args.mini_batches
    if args.desired_kl is not None:
        kwargs["algorithm"]["desired_kl"] = args.desired_kl
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vnc-config.json").write_text(json.dumps({
        "task": TASK, "base_checkpoint": args.checkpoint, "packet": args.packet, "layout": args.layout,
        "width": args.width, "residual_scale": args.residual_scale, "shuffle_seed": args.shuffle_seed,
        "residual_l2": args.residual_l2, "brain_hidden": args.brain_hidden,
        "desired_kl": kwargs["algorithm"].get("desired_kl"), "critic_warmup": args.critic_warmup,
        "learn_synapses": args.learn_synapses, "command_readout": args.command_readout,
        "escape": args.escape, "sprint": args.sprint, "mlp_anchor": args.mlp_anchor,
        "command_l2": args.command_l2,
        "num_envs": res.num_envs,
        "iterations": args.iterations, "mini_batches": args.mini_batches, "seed": args.seed,
        "troubles": added, "symmetry": False, "backend": res.backend, "device": str(res.device),
        "started_unix": time.time(),
    }, indent=2) + "\n")

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    if not args.no_guard:
        troubles.clip_rewards(env, added["guard"]["reward_clip"])
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner_cls = tasks.load_runner_cls(TASK)
        runner = runner_cls(wrapped, kwargs, str(out), res.device)
        official = torch.load(args.checkpoint, weights_only=False, map_location=res.device)
        load_padded(runner.alg._raw_critic, official["critic_state_dict"])
        for line in restore_curriculum(env, (official.get("infos") or {}).get("curriculum")):
            print(f"[vnc] curriculum {line}")
        actor = runner.alg._raw_actor
        trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        print(f"[vnc] cord {actor.n} nodes x width {actor.width}; trainable actor parameters {trainable}")
        # On 10/24 every run that updated the cord from the first iteration drifted from the
        # official gait (flyv2b-s1 at iteration 100: nominal walk 0.81 of the command against
        # 0.93), and a strong residual pull held it at the official without gains. The critic
        # starts from the official one, which never met a trouble, so the first advantages are
        # largely noise; freezing the actor while the critic catches up is the usual remedy.
        # PPO's adaptive step does not grow on a KL of exactly zero, so the step size is where
        # it was when the actor thaws.
        warm = min(args.critic_warmup, args.iterations)
        if warm:
            frozen = [q for q in actor.parameters() if q.requires_grad]
            for q in frozen:
                q.requires_grad_(False)
            print(f"[vnc] critic warm-up: {warm} iterations with the actor frozen")
            runner.learn(num_learning_iterations=warm, init_at_random_ep_len=True)
            for q in frozen:
                q.requires_grad_(True)
        runner.learn(num_learning_iterations=args.iterations - warm, init_at_random_ep_len=not warm)
        print(f"VNC_V1_TRAIN_DONE {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
