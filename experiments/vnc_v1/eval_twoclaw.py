"""The two-claw block, held still: both claws up as a shield, standing on the four middle and rear legs.

The user's idea (10/26): block with both claws when one is not enough. No four-legged program exists
yet, so this tests the stance itself, without a policy: the legs hold the standing pose (posture
action 0), both arms the guard pose (the left one as given, the right one mirrored: J0, J2, J3 change
sign), the arms at `--arm-kp`. Straight bolts from ahead (and `--spread` degrees either side) are
fired at the standing robot; each is deflected, grazed, hit or missed as in `eval_guard.py`, and a
robot whose trunk tilts past 50 degrees has fallen. Whether a robot can stand this way at all, and
whether two claws side by side stop what one does not, is what it answers.

    python experiments/vnc_v1/eval_twoclaw.py --out runs/vnc-v1/eval/twoclaw-g100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
import escape  # noqa: E402
import eval_official as ev  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--robots-per-case", type=int, default=32)
    p.add_argument("--throws", type=int, default=4)
    p.add_argument("--every", type=float, default=2.0)
    p.add_argument("--flight", type=float, default=0.45)
    p.add_argument("--dist", type=float, default=3.0)
    p.add_argument("--start-height", type=float, default=0.25)
    p.add_argument("--spread", type=float, default=10.0)
    p.add_argument("--pose", default="-1.638,-2.507,-0.604,2.293", help="the left claw's guard (posture action)")
    p.add_argument("--arm-kp", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.escape, args.block, args.balls, args.vnc_config = True, True, 1, ""

    import torch

    bearings = [-args.spread, 0.0, args.spread]
    n = args.robots_per_case
    num = n * len(bearings)
    env, wrapped, policy, res = ev.build(args, num)
    dev = env.device
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    act = env.scene["robot"].actuators[0]
    left = torch.tensor([float(v) for v in args.pose.split(",")], device=dev)
    action = torch.zeros(num, 20, device=dev)
    action[:, :4] = left
    action[:, 4:8] = left * torch.tensor([-1.0, 1.0, -1.0, -1.0], device=dev)
    kp = act.default_stiffness.clone()
    cols = [i for i, nm in enumerate(act.target_names) if nm[:4] in ("LF_J", "RF_J") and not nm.endswith("J4_joint")]
    kp[:, cols] = args.arm_kp
    act.set_gains(slice(None), kp=kp)
    hit_s = env.scene[escape.hit_sensor_name(0)]
    claws = [env.scene[escape.block_sensor_name(0, side)] for side in ("LF", "RF")]
    st = escape._state(env, 0)
    ball = env.scene[escape.ball_name(0)]
    dt = env.step_dt
    throw_steps = [int(round((2.0 + k * args.every) / dt)) for k in range(args.throws)]
    steps = throw_steps[-1] + int(round(1.5 / dt))
    claw_at = torch.full((num,), -1, dtype=torch.long, device=dev)
    body_hit = torch.zeros(num, dtype=torch.bool, device=dev)
    turned = torch.zeros(num, dtype=torch.bool, device=dev)
    counts = {key: torch.zeros(num, device=dev) for key in ("deflected", "grazed", "hit", "missed")}
    fell = torch.zeros(num, dtype=torch.bool, device=dev)
    robot = env.scene["robot"]
    look = int(round(0.04 / dt))

    def tally() -> None:
        touched = claw_at >= 0
        counts["hit"] += body_hit.float()
        counts["deflected"] += (touched & ~body_hit & turned).float()
        counts["grazed"] += (touched & ~body_hit & ~turned).float()
        counts["missed"] += (~touched & ~body_hit).float()

    with torch.inference_mode():
        for step in range(steps):
            if step in throw_steps:
                if step != throw_steps[0]:
                    tally()
                claw_at[:], body_hit[:], turned[:] = -1, False, False
                for k, b in enumerate(bearings):
                    ids = torch.arange(k * n, (k + 1) * n, device=dev)
                    escape.throw_ball(env, ids, prob=1.0, dist=(args.dist, args.dist), flight=(args.flight, args.flight),
                                      start_height=(args.start_height, args.start_height), aim_height=(0.09, 0.09),
                                      lead=1.0, bearing=math.radians(b), which=0, straight=True)
            twist.vel_command_b[:] = 0.0
            posture.posture_command[:, :3] = 0.0
            posture.posture_command[:, 3] = posture.cfg.neutral_height
            escape.hold_bolts(env, 0, stop=(claw_at >= 0) | body_hit)
            wrapped.step(action)
            flying = st["flight_end"] > escape._now(env)
            claw = torch.stack([(c.data.found.reshape(num, -1) > 0).any(-1) for c in claws]).any(0)
            body = (hit_s.data.found.reshape(num, -1) > 0).any(-1) & ~claw
            claw_at = torch.where((claw_at < 0) & flying & claw, torch.full_like(claw_at, step), claw_at)
            body_hit |= flying & body
            check = (claw_at >= 0) & (claw_at == step - look)
            if bool(check.any()):
                v0 = st["bolt_v"]
                along = (ball.data.root_link_lin_vel_w * v0).sum(-1) / (v0 * v0).sum(-1).clamp_min(1e-6)
                turned |= check & (along < 0.5)
            g = robot.data.projected_gravity_b
            fell |= torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) > math.radians(50)
    tally()
    total = num * args.throws
    meta = {"flight_s": args.flight, "start_height_m": args.start_height, "pose": args.pose, "arm_kp": args.arm_kp,
            **{key: round(float(v.sum()) / total, 4) for key, v in counts.items()},
            "fell": round(float(fell.float().mean()), 4),
            "base_height_end_m": round(float(robot.data.root_link_pos_w[:, 2].mean()), 4)}
    for k, b in enumerate(bearings):
        sl = slice(k * n, (k + 1) * n)
        print(f"{b:+5.0f} deg  " + "  ".join(f"{key} {float(v[sl].sum()) / (n * args.throws):.3f}" for key, v in counts.items())
              + f"  fell {float(fell[sl].float().mean()):.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta}, indent=2) + "\n")
    print(json.dumps(meta))
    print(f"TWOCLAW_EVAL_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
