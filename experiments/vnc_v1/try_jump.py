"""Prototype: the official reference-free jump as a motor program inside the posture gait.

The robots walk with the posture policy; at `--at` s the jump program takes over for
`--duration` s and the servo stiffness goes from the posture task's kp 10 to the jump task's kp 20
(the official torque curve and thermal model stay as they are), then the posture policy and kp 10
return. The jump policy (tasks/jumper/ref_free_jump, model_5999) reads the posture observation's
shared terms -- the same `actuator_force` term, the same joint order and default pose -- with its
own action units (scale 0.5 against the posture task's 0.25). Reports the base's and the lowest
foot's rise, falls, and whether the walk resumes.

    python experiments/vnc_v1/try_jump.py --robots 16 --out runs/vnc-v1/eval/try-jump
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
sys.path.insert(0, str(HERE))
import eval_official as ev  # noqa: E402

JUMP_CKPT = "tasks/jumper/ref_free_jump/out/example/model_5999.pt"
POSTURE_SCALE, JUMP_SCALE = 0.25, 0.5
KP_POSTURE, KP_JUMP = 10.0, 20.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--robots", type=int, default=16)
    p.add_argument("--command", default="0.3,0,0", help="walking command before and after")
    p.add_argument("--at", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--after", type=float, default=3.0, help="seconds of walking after the jump")
    p.add_argument("--home-s", type=float, default=0.3, help="seconds at HOME (zero action, kp 20) before the jump")
    p.add_argument("--go-steps", type=int, default=1, help="control steps of the go pulse (0: none)")
    p.add_argument("--go-file", default="../../../runs/vnc-v1/packet-1/jump_go_force.npy",
                   help="the torque history the official jump's episodes began with (406-input order, 100 values)")
    p.add_argument("--own-history", type=int, default=1,
                   help="1: the jump program keeps its own action history from zero; 0: converted posture history")
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--vnc-config", default="")
    p.add_argument("--vnc-packet", default="")
    p.add_argument("--vnc-layout", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from tensordict import TensorDict

    import tasks
    from distill_vnc import actor_kwargs
    from rsl_rl.models.mlp_model import MLPModel

    n = args.robots
    env, wrapped, policy, res = ev.build(args, n)
    dev = env.device
    robot = env.scene["robot"]
    act = robot.actuators[0]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    cmd = torch.tensor([float(v) for v in args.command.split(",")], device=dev).repeat(n, 1)
    jump_cfg = tasks.load_agent_cfg("jumper.ref_free_jump")
    probe = TensorDict({"actor": torch.zeros(n, 406, device=dev)}, batch_size=[n])
    jump = MLPModel(probe, {"actor": ["actor"]}, "actor", 20, **actor_kwargs(jump_cfg)).to(dev)
    jump.load_state_dict(torch.load(JUMP_CKPT, weights_only=False, map_location=dev)["actor_state_dict"])
    jump.eval()
    # What touches the ground: the four legs' foot tips and the two arms' palm pads.
    feet = [i for i, name in enumerate(robot.body_names) if "foot_tip" in name or "palm_pad" in name]

    def pin() -> None:
        twist.vel_command_b[:] = cmd
        posture.posture_command[:, :3] = 0.0
        posture.posture_command[:, 3] = posture.cfg.neutral_height

    pin()
    obs = wrapped.get_observations()
    dt = env.step_dt
    start, stop = int(round(args.at / dt)), int(round((args.at + args.duration) / dt))
    home = int(round(args.home_s / dt))
    stop += home
    import numpy as np

    go = torch.tensor(np.load(args.go_file), device=dev)[None] if args.go_steps else None
    fring = torch.zeros(17, n, 20, device=dev)
    steps = stop + int(round(args.after / dt))
    all_ids = torch.arange(n, device=dev)
    base_z, foot_z, tilt = [], [], []
    # The jump program's own action history (5 frames, 4 steps apart, oldest first), starting at
    # zero when it takes over, as every jump of its training started: with the posture gait's
    # history in its place it read "already landed" and stood (base +3-7 mm).
    ring = torch.zeros(17, n, 20, device=dev)
    with torch.inference_mode():
        for step in range(steps):
            jumping = start <= step < stop
            if step == start:
                act.set_gains(all_ids, kp=torch.full((n,), KP_JUMP, device=dev))
                ring.zero_()
            if step == stop:
                act.set_gains(all_ids, kp=torch.full((n,), KP_POSTURE, device=dev))
            pin()
            if jumping and step < start + home:
                # Settle at HOME first: the official task gives the go only to a robot at HOME and
                # still, and its policy stood when handed the posture gait's stance (arms 0.2 rad
                # off HOME), its actions near zero.
                action = torch.zeros(n, 20, device=dev)
                policy(obs)
            elif jumping:
                x = obs["actor"]
                hist = ring[[0, 4, 8, 12, 16]].permute(1, 0, 2).reshape(n, 100) if args.own_history \
                    else x[:, 206:306] * (POSTURE_SCALE / JUMP_SCALE)
                force = x[:, 309:409]
                if go is not None:
                    # The go: every training jump began with the torque history saturated (a
                    # reset artefact the policy took as its cue: mean |action| 0.74 on that input
                    # against 0.07 on a settled stance). The giant fibre's pulse fills the jump
                    # program's own torque history with it, and real torques then push it out
                    # frame by frame, over the 16 steps the native history took.
                    if step == start + home:
                        fring[:] = go.view(5, 20)[-1]
                    fring = torch.cat([fring[1:], x[None, :, 389:409]], dim=0) if step > start + home else fring
                    force = fring[[0, 4, 8, 12, 16]].permute(1, 0, 2).reshape(n, 100)
                xj = torch.cat([x[:, :206], hist, force], dim=-1)
                a_jump = jump(TensorDict({"actor": xj}, batch_size=[n]))
                ring = torch.cat([ring[1:], a_jump[None]], dim=0)
                action = a_jump * (JUMP_SCALE / POSTURE_SCALE)
                policy(obs)  # the posture brain keeps its own state running through the jump
            else:
                action = policy(obs)
            obs, _, _, _ = wrapped.step(action)
            pos = robot.data.body_link_pos_w
            base_z.append(float(pos[:, 0, 2].mean()))
            foot_z.append(float(pos[:, feet, 2].min(dim=-1).values.mean()) if feet else float("nan"))
            g = robot.data.projected_gravity_b
            tilt.append(torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))))
    tilt = torch.stack(tilt)  # [steps, robots]
    pre = slice(max(0, start - int(0.5 / dt)), start)
    fell = (tilt > 50).any(dim=0)
    resumed = (tilt[-int(1.0 / dt):] < 30).all(dim=0)
    report = {
        "robots": n, "command": args.command, "jump_at_s": args.at, "jump_s": args.duration,
        "base_rise_m": round(max(base_z[start:stop]) - sum(base_z[pre]) / len(base_z[pre]), 4),
        "lowest_foot_rise_m": round(max(foot_z[start:stop]) - sum(foot_z[pre]) / len(foot_z[pre]), 4),
        "apex_after_s": round((max(range(start, stop), key=lambda s: base_z[s]) - start) * dt, 3),
        "fell": round(float(fell.float().mean()), 3), "walking_again": round(float(resumed.float().mean()), 3),
        "feet_bodies": [robot.body_names[i] for i in feet],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    print(f"VNC_TRY_JUMP_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
