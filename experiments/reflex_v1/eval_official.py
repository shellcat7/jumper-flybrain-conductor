"""The official posture policy under sudden trouble: low grip, a dead leg, a shove.

Each block of environments is pinned to one command and one disturbance that
strikes at `--event-s`. Nothing resets after a fall: the run keeps every robot
on the ground and records who stays up and how well the command is kept. This
is the baseline the reflex scenes have to beat, and it shows which troubles the
official brain really has (the film may only show those).

    python experiments/reflex_v1/eval_official.py --backend native --device cpu \
        --envs-per-case 1 --duration-s 4 --out runs/reflex-v1/smoke
    python experiments/reflex_v1/eval_official.py --backend warp --device cuda:0 \
        --envs-per-case 256 --out /content/results/reflex-v1-official
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

TASK = "jumper.posture"

#: Body-frame (vx, vy, wz). "corner" is the chase's turn at speed.
COMMANDS = {
    "walk": (0.3, 0.0, 0.0),
    "run": (0.6, 0.0, 0.0),
    "corner": (0.5, 0.0, 1.5),
}

#: Sprint commands, chosen with --commands: the official ceiling and beyond it.
SPRINT_COMMANDS = {
    "fast": (0.8, 0.0, 0.0),
    "sprint": (1.0, 0.0, 0.0),
    "sprint_corner": (0.8, 0.0, 2.0),
    "sprint12": (1.2, 0.0, 0.0),
    "sprint15": (1.5, 0.0, 0.0),
    "sprint18": (1.8, 0.0, 0.0),
    "sprint20": (2.0, 0.0, 0.0),
    # sideways (10/26: the lateral specialist; a bolt from ahead is dodged only by a sideways step)
    "side": (0.0, 0.4, 0.0),
    "side_fast": (0.0, 0.7, 0.0),
    "side_sprint": (0.0, 1.0, 0.0),
}

#: Leg prefixes of the joint names. LF/RF are the arms that also walk.
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")

#: Foot friction outside the trouble. Training drew it from 0.5-1.5.
NOMINAL_GRIP = 1.0


def conditions() -> list[dict]:
    rows = [{"name": "nominal", "kind": "none", "value": 0.0}]
    grips = (0.3, 0.2, 0.12, 0.06, 0.03)
    rows += [{"name": f"grip_{mu:g}", "kind": "grip", "value": mu} for mu in grips]
    # One leg, then two: a smoke run on 2026-10-23 (CPU, one robot per case) kept
    # every robot up on five legs, slower and drifting.
    dead = [*LEGS, "LM+LR", "LM+RR"]
    rows += [{"name": f"dead_{legs}", "kind": "dead", "value": legs} for legs in dead]
    rows += [{"name": f"shove_{v:g}", "kind": "shove", "value": v} for v in (0.4, 0.7, 1.0, 1.5)]
    return rows


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs-per-case", type=int, default=256)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--event-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--obs-noise", type=int, default=1, help="1: training's sensor noise")
    p.add_argument("--only", default="", help="comma list of condition names to keep")
    p.add_argument("--five-foot", action="store_true",
                   help="prototype: hand robots with a dead left-front leg to the official five-legged gait")
    p.add_argument("--five-foot-vmax", type=float, default=0.5,
                   help="the five-legged gait's linear command bound (its training ceiling is 0.5 m/s)")
    p.add_argument("--commands", default="walk,run,corner",
                   help="comma list from walk, run, corner, fast, sprint, sprint_corner")
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt",
                   help="relative to the upstream checkout (the cwd)")
    p.add_argument("--vnc-config", default="",
                   help="a VNC run's vnc-config.json: evaluate --checkpoint (a training checkpoint, or a "
                        "receipt from colab/vnc_v1/watch.py) as that run's VNC actor")
    p.add_argument("--vnc-packet", default="", help="override the config's packet path")
    p.add_argument("--vnc-layout", default="", help="override the config's layout path")
    p.add_argument("--out", required=True)
    return p.parse_args()


def local_mirror(path: str | None, packet: str) -> str | None:
    """`path` if it exists here, else mirror_actor.npz beside the packet (None without a path)."""
    if not path or Path(path).exists():
        return path
    return str(Path(packet).with_name("mirror_actor.npz"))


def vnc_actor(args) -> dict:
    """The actor entries a VNC run was trained with, from its vnc-config.json."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vnc_v1"))
    from vnc_model import VNCModel

    cfg = json.loads(Path(args.vnc_config).read_text())
    return {
        "class_name": VNCModel,
        "packet": args.vnc_packet or cfg["packet"],
        "layout": args.vnc_layout or cfg["layout"],
        "base_checkpoint": cfg["base_checkpoint"],
        "width": cfg["width"],
        "residual_scale": cfg["residual_scale"],
        "shuffle_seed": cfg.get("shuffle_seed"),
        "brain_hidden": cfg.get("brain_hidden", 0),
        "learn_synapses": cfg.get("learn_synapses", False),
        "command_readout": cfg.get("command_readout", False),
        "mlp_anchor": cfg.get("mlp_anchor"),
        # The director's motor programs; their weights come with the checkpoint (the adapted
        # ones, `five_foot.py`, from the official task's own file).
        "experts": cfg.get("experts", []),
        "adapters": cfg.get("adapters", []),
        "adapter_vmax": cfg.get("adapter_vmax", 0.5),
        # Mirrored experts read mirror_matrix.py's npz; a Colab path falls back to the one beside the packet.
        "obs_mirror": local_mirror(cfg.get("obs_mirror"), args.vnc_packet or cfg["packet"]),
        # The hard-wired giant-fibre escape (`vnc_model.VNCModel.reflex`), and its jump.
        "gf_reflex": cfg.get("gf_reflex"),
        "threat_cord": cfg.get("threat_cord", True),
        "gf_jump": cfg.get("gf_jump"),
    }


def apply_jump_gains(env, policy) -> None:
    """Servo stiffness for the giant-fibre jump: the jump task's kp 20 on the robots `VNCModel`
    flags as mid-jump, the posture task's default on the rest. Call after the policy, before the
    step it acts in."""
    active = getattr(policy, "jump_active", None)
    guard_t = getattr(policy, "_guard_t", None)
    guard_kp = ((getattr(policy, "gf_reflex", None) or {}).get("guard") or {}).get("kp")
    if active is None and not (guard_kp and guard_t is not None):
        return
    import torch

    act = env.scene["robot"].actuators[0]
    kp = act.default_stiffness.clone()
    if active is not None:
        kp = torch.where(active[:, None], torch.full_like(kp, 20.0), kp)
    if guard_kp and guard_t is not None:
        # (10/26) The guarding arm stiffer, so the claw is in place sooner and is not knocked back:
        # on the posture task's kp 10 it met a bolt from 60 cm up in time only 0.43 of the time.
        right = policy._guard_side if policy._guard_side is not None else torch.zeros_like(guard_t, dtype=torch.bool)
        on = guard_t >= 0
        for side, mask in (("LF", on & ~right), ("RF", on & right)):
            cols = [i for i, n in enumerate(act.target_names) if n.startswith(f"{side}_J") and not n.startswith(f"{side}_J4")]
            kp[:, cols] = torch.where(mask[:, None], torch.full_like(kp[:, cols], float(guard_kp)), kp[:, cols])
    act.set_gains(slice(None), kp=kp)


def vnc_escape(args) -> bool:
    """Whether the VNC run trained with the escape task (its actor then reads a "threat" group)."""
    return bool(args.vnc_config) and bool(json.loads(Path(args.vnc_config).read_text()).get("escape"))


def load_vnc(runner, path: str, device) -> None:
    import torch

    ck = torch.load(path, weights_only=False, map_location=device)
    actor = runner.alg._raw_actor
    if "actor_state_dict" in ck:  # a training checkpoint
        actor.load_state_dict(ck["actor_state_dict"], strict=True)
        return
    # A receipt carries no frozen MLP; the model loaded the official one itself.
    missing, unexpected = actor.load_state_dict(ck["actor"], strict=False)
    assert not unexpected and all(k.startswith("mlp.") for k in missing), (missing, unexpected)


def match_training_noise(env_cfg) -> None:
    """Copy each observation term's training noise into the replay config.

    The same move as `scripts/play.py --obs-noise`: replay builds clean sensors, and
    the policy is meant to read noisy ones.
    """
    import tasks

    train = tasks.load_env_cfg(TASK)
    for name, group in env_cfg.observations.items():
        ref = train.observations.get(name)
        if ref is None:
            continue
        group.enable_corruption = ref.enable_corruption
        for term_name, term in group.terms.items():
            src = ref.terms.get(term_name)
            if src is not None:
                term.noise = src.noise


def build(args, num_envs: int):
    import tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend=args.backend, device=args.device, num_envs=num_envs, training=False)
    use_backend(res)  # before the env is built, or it has no effect

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    cfg = tasks.load_env_cfg(TASK, play=True)
    if args.obs_noise:
        match_training_noise(cfg)
    cfg.seed = args.seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1e9
    # No resets: a fallen robot stays where it fell, and the fall is ours to count.
    cfg.terminations = {}
    twist = cfg.commands["twist"]
    twist.heading_command = False
    twist.ranges.heading = None  # the command class refuses a heading range without it
    twist.rel_standing_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0
    for term in cfg.commands.values():
        term.resampling_time_range = (1e9, 1e9)
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = (NOMINAL_GRIP, NOMINAL_GRIP)
    escape_on = vnc_escape(args) or getattr(args, "escape", False)
    if escape_on and "ball" not in cfg.scene.entities:
        # The ball and the looming signal, never thrown by the event (the caller throws).
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vnc_v1"))
        import escape

        escape.add_escape(cfg, interval=(1e9, 1e9), prob=0.0, balls=getattr(args, "balls", 1),
                          block=getattr(args, "block", False))
        if getattr(args, "balls", 1) >= 4:  # each resting ball's contacts take constraint rows: 4 overflow 512
            cfg.sim.njmax = max(cfg.sim.njmax or 0, 768)

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    from mjrl.replay import skip_rewards

    skip_rewards(env)
    agent_cfg = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner
    kwargs = asdict(agent_cfg)
    if args.vnc_config:
        kwargs["actor"].update(vnc_actor(args))
        kwargs["algorithm"]["symmetry_cfg"] = None  # rsl_rl refuses it with a recurrent actor
        if vnc_escape(args):
            kwargs["obs_groups"] = {"actor": ["actor", "threat"], "critic": ["critic", "threat"]}
    runner = runner_cls(wrapped, kwargs, device=res.device)
    if args.vnc_config:
        load_vnc(runner, args.checkpoint, res.device)
    else:
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=res.device)
    # A VNC actor keeps its cord state between calls; nothing resets here, so it runs on.
    policy = runner.get_inference_policy(device=res.device)
    return env, wrapped, policy, res


def foot_asset_cfg(env):
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if "foot_friction" in env.event_manager.active_terms.get("startup", []):
        return env.event_manager.get_term_cfg("foot_friction").params["asset_cfg"]
    from tasks.jumper.common.constants import FOOT_GEOMS

    cfg = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
    cfg.resolve(env.scene)
    return cfg


def set_grip(env, ids, mu: float) -> None:
    from mjlab.envs.mdp import dr

    if len(ids):
        dr.geom_friction(env, ids, ranges=(mu, mu), asset_cfg=foot_asset_cfg(env), operation="abs")


def kill_leg(env, ids, legs: str) -> int:
    """Zero the torque limit of every joint of `legs` ("LM" or "LM+LR"): the brain
    still commands them, the legs no longer answer. Returns how many joints died."""
    robot = env.scene["robot"]
    prefixes = tuple(f"{leg}_" for leg in legs.split("+"))
    dead = 0
    for act in robot.actuators:
        cols = [i for i, n in enumerate(act.target_names) if n.startswith(prefixes)]
        if not cols or act.force_limit is None:
            continue
        limit = act.force_limit[ids].clone()
        limit[:, cols] = 0.0
        act.set_effort_limit(ids, effort_limit=limit)
        dead += len(cols)
    return dead


def shove(env, ids, speed: float) -> None:
    """Add `speed` m/s to the base, sideways (body +y), at once."""
    import torch

    robot = env.scene["robot"]
    vel = robot.data.root_link_vel_w[ids].clone()
    heading = robot.data.heading_w[ids]
    vel[:, 0] += -torch.sin(heading) * speed
    vel[:, 1] += torch.cos(heading) * speed
    robot.write_root_link_velocity_to_sim(vel, env_ids=ids)


def main() -> None:
    args = parse()
    import torch

    conds = conditions()
    if args.only:
        keep = set(args.only.split(","))
        conds = [c for c in conds if c["name"] in keep]
    table = {**COMMANDS, **SPRINT_COMMANDS}
    cases = [(cmd, c) for cmd in args.commands.split(",") for c in conds]
    n = args.envs_per_case
    num_envs = n * len(cases)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env, wrapped, policy, res = build(args, num_envs)
    build_s = time.time() - t0
    dev = env.device
    robot = env.scene["robot"]
    if args.five_foot:
        # A prototype (10/25): robots whose left- or right-front leg has no torque left are handed
        # to the official five-legged gait (mirrored for the right) at once -- the brain's own
        # detection is not modelled here.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vnc_v1"))
        from five_foot import FiveFootProgram

        act0 = robot.actuators[0]
        programs = {}
        for leg, mirror in (("LF", False), ("RF", True)):
            cols = [i for i, name in enumerate(act0.target_names) if name.startswith(f"{leg}_")]
            programs[leg] = (FiveFootProgram(num_envs, dev, mirror=mirror, vmax=args.five_foot_vmax), cols,
                             torch.zeros(num_envs, dtype=torch.bool, device=dev))
        base_policy = policy

        def policy(obs):  # noqa: F811
            action = base_policy(obs)
            for program, cols, on in programs.values():
                dead = (act0.force_limit[:, cols] == 0).all(dim=-1)
                program.reset(dead & ~on)
                on |= dead
                if bool(on.any()):
                    action = torch.where(on[:, None], program(obs["actor"], on), action)
            return action
    case_of = torch.arange(num_envs, device=dev) // n
    cmd = torch.tensor([table[c] for c, _ in cases], device=dev)[case_of]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")

    def pin_commands() -> None:
        twist.vel_command_b[:] = cmd
        posture.posture_command[:, :3] = 0.0
        posture.posture_command[:, 3] = posture.cfg.neutral_height

    set_grip(env, torch.arange(num_envs, device=dev), NOMINAL_GRIP)
    pin_commands()
    obs = wrapped.get_observations()

    dt = env.step_dt
    steps = int(round(args.duration_s / dt))
    event_step = int(round(args.event_s / dt))
    fall_tilt = math.radians(50.0)  # the task's own fell_over bound
    collapse_z = 0.045  # base this low is lying on the ground (stands at 0.106)

    first_fall = torch.full((num_envs,), -1, device=dev, dtype=torch.long)
    first_collapse = torch.full_like(first_fall, -1)
    max_tilt_post = torch.zeros(num_envs, device=dev)
    err_sum = torch.zeros(num_envs, 2, device=dev)  # [pre, post] planar + yaw error
    err_cnt = torch.zeros(num_envs, 2, device=dev)
    fwd_sum = torch.zeros(num_envs, device=dev)  # post-event speed along the command
    cmd_dir = cmd[:, :2] / torch.norm(cmd[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
    pos_event = torch.zeros(num_envs, 3, device=dev)
    yaw_event = torch.zeros(num_envs, device=dev)
    dead_joints = {}

    t1 = time.time()
    with torch.inference_mode():
        for step in range(steps):
            if step == event_step:
                pos_event[:] = robot.data.root_link_pos_w
                yaw_event[:] = robot.data.heading_w
                for k, (_, c) in enumerate(cases):
                    ids = torch.arange(k * n, (k + 1) * n, device=dev)
                    if c["kind"] == "grip":
                        set_grip(env, ids, c["value"])
                    elif c["kind"] == "dead":
                        dead_joints[c["name"]] = kill_leg(env, ids, c["value"])
                    elif c["kind"] == "shove":
                        shove(env, ids, c["value"])
            pin_commands()
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)

            g = robot.data.projected_gravity_b
            tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
            z = robot.data.root_link_pos_w[:, 2]
            fell = tilt > fall_tilt
            first_fall = torch.where(fell & (first_fall < 0), step, first_fall)
            first_collapse = torch.where((z < collapse_z) & (first_collapse < 0), step, first_collapse)
            up = first_fall < 0
            v = robot.data.root_link_lin_vel_b
            w = robot.data.root_link_ang_vel_b[:, 2]
            err = torch.norm(cmd[:, :2] - v[:, :2], dim=-1) + 0.1 * torch.abs(cmd[:, 2] - w)
            phase = 0 if step < event_step else 1
            if step >= event_step + int(0.5 / dt) or phase == 0:  # skip the first 0.5 s after
                err_sum[:, phase] += torch.where(up, err, 0.0)
                err_cnt[:, phase] += up.float()
            if phase == 1:
                max_tilt_post = torch.maximum(max_tilt_post, tilt)
                fwd_sum += torch.where(up, (v[:, :2] * cmd_dir).sum(-1), 0.0)
    loop_s = time.time() - t1

    post_steps = steps - event_step
    disp = robot.data.root_link_pos_w - pos_event
    rows = []
    for k, (cname, c) in enumerate(cases):
        s = slice(k * n, (k + 1) * n)
        ff = first_fall[s]
        fell_post = (ff >= event_step).float()
        fell_pre = ((ff >= 0) & (ff < event_step)).float()
        fall_t = ((ff[ff >= event_step] - event_step).float() * dt)
        coll = (first_collapse[s] >= event_step).float()
        e = err_sum[s] / err_cnt[s].clamp(min=1)
        v_cmd = math.hypot(*table[cname][:2])  # the side commands have no forward part
        rows.append({
            "command": cname,
            "condition": c["name"],
            "n": n,
            "fell_before_event": round(float(fell_pre.mean()), 4),
            "fell_after_event": round(float(fell_post.mean()), 4),
            "collapsed_after_event": round(float(coll.mean()), 4),
            "fall_time_median_s": round(float(fall_t.median()), 3) if fall_t.numel() else None,
            "track_err_pre": round(float(e[:, 0].mean()), 4),
            "track_err_post": round(float(e[:, 1].mean()), 4),
            "speed_ratio_post": round(float(fwd_sum[s].mean() / post_steps / v_cmd), 4),
            "max_tilt_post_deg": round(float(torch.rad2deg(max_tilt_post[s]).mean()), 2),
            "displacement_m": round(float(torch.norm(disp[s, :2], dim=-1).mean()), 3),
        })

    meta = {
        "task": TASK,
        "checkpoint": args.checkpoint,
        "vnc_config": args.vnc_config or None,
        "backend": res.backend,
        "device": str(res.device),
        "num_envs": num_envs,
        "envs_per_case": n,
        "control_dt": dt,
        "duration_s": args.duration_s,
        "event_s": args.event_s,
        "seed": args.seed,
        "obs_noise": bool(args.obs_noise),
        "nominal_grip": NOMINAL_GRIP,
        "dead_joints": dead_joints,
        "build_s": round(build_s, 1),
        "loop_s": round(loop_s, 1),
        "env_steps_per_s": round(num_envs * steps / max(loop_s, 1e-9)),
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({
        "first_fall": first_fall.cpu(), "first_collapse": first_collapse.cpu(),
        "max_tilt_post": max_tilt_post.cpu(), "err": (err_sum / err_cnt.clamp(min=1)).cpu(),
        "case_of": case_of.cpu(), "cases": [(a, b["name"]) for a, b in cases],
    }, out / "per_env.pt")
    print(json.dumps(meta))
    for r in rows:
        print(f"{r['command']:7s} {r['condition']:10s} fell {r['fell_after_event']:.3f} "
              f"(pre {r['fell_before_event']:.3f}) err {r['track_err_pre']:.3f}->{r['track_err_post']:.3f} "
              f"speed {r['speed_ratio_post']:.2f}")
    print(f"REFLEX_V1_OFFICIAL_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
