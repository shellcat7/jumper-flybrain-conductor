"""Evaluate the claw guard: straight bolts from ahead at walking robots; count blocks and hits.

Each bolt is one of: hit (it touched anything but a claw while in flight -- even after grazing a
claw), deflected (it touched a claw, `escape.block_sensor_name`, kept under half its speed along
its line 40 ms later, and touched nothing else), grazed (a claw touched it but it flew on, missing
the rest), missed. (10/26: the first version scored any first contact with a claw as a block; the
"blocks" were grazes -- the bolts kept 52-87% of their speed and passed under the robot.) Same schedule as `eval_escape.py` (a bolt every --every s from 2 s), one bearing (ahead,
and --spread degrees either side), walking and running.

    python experiments/vnc_v1/eval_guard.py --checkpoint <model.pt> --vnc-config <config with guard> \\
        --vnc-packet ... --vnc-layout ... --out runs/vnc-v1/eval/<name>
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

COMMANDS = {"walk": (0.3, 0.0, 0.0), "run": (0.6, 0.0, 0.0)}


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
    p.add_argument("--spread", type=float, default=10.0, help="degrees either side of dead ahead")
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vnc-config", required=True)
    p.add_argument("--vnc-packet", default="")
    p.add_argument("--vnc-layout", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.escape, args.block, args.balls = True, True, 1

    import torch

    bearings = [-args.spread, 0.0, args.spread]
    cases = [(c, b) for c in COMMANDS for b in bearings]
    n = args.robots_per_case
    num = n * len(cases)
    env, wrapped, policy, res = ev.build(args, num)
    dev = env.device
    case_of = torch.arange(num, device=dev) // n
    cmd = torch.tensor([COMMANDS[c] for c, _ in cases], device=dev)[case_of]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    hit_s = env.scene[escape.hit_sensor_name(0)]
    claws = [env.scene[escape.block_sensor_name(0, side)] for side in ("LF", "RF")]
    st = escape._state(env, 0)
    dt = env.step_dt
    throw_steps = [int(round((2.0 + k * args.every) / dt)) for k in range(args.throws)]
    steps = throw_steps[-1] + int(round(1.5 / dt))
    # Per throw: the step the bolt first touched a claw (-1: not yet), whether it touched anything
    # else while in flight (a hit, even after a claw), and whether the claw turned it back.
    claw_at = torch.full((num,), -1, dtype=torch.long, device=dev)
    body_hit = torch.zeros(num, dtype=torch.bool, device=dev)
    turned = torch.zeros(num, dtype=torch.bool, device=dev)
    counts = {key: torch.zeros(num, device=dev) for key in ("deflected", "grazed", "hit", "missed")}
    fell = torch.zeros(num, dtype=torch.bool, device=dev)
    robot = env.scene["robot"]
    ball = env.scene[escape.ball_name(0)]
    obs = wrapped.get_observations()

    def tally() -> None:
        touched = claw_at >= 0
        counts["hit"] += body_hit.float()
        counts["deflected"] += (touched & ~body_hit & turned).float()
        counts["grazed"] += (touched & ~body_hit & ~turned).float()
        counts["missed"] += (~touched & ~body_hit).float()

    look = int(round(0.04 / dt))  # how long after the claw the bolt's speed along its line is read
    with torch.inference_mode():
        for step in range(steps):
            if step in throw_steps:
                if step != throw_steps[0]:
                    tally()
                claw_at[:], body_hit[:], turned[:] = -1, False, False
                for k, (_, b) in enumerate(cases):
                    ids = torch.arange(k * n, (k + 1) * n, device=dev)
                    escape.throw_ball(env, ids, prob=1.0, dist=(args.dist, args.dist), flight=(args.flight, args.flight),
                                      start_height=(args.start_height, args.start_height), aim_height=(0.09, 0.09),
                                      lead=1.0, bearing=math.radians(b), which=0, straight=True)
            twist.vel_command_b[:] = cmd
            posture.posture_command[:, :3] = 0.0
            posture.posture_command[:, 3] = posture.cfg.neutral_height
            escape.hold_bolts(env, 0, stop=(claw_at >= 0) | body_hit)
            action = policy(obs)
            ev.apply_jump_gains(env, policy)
            obs, _, _, _ = wrapped.step(action)
            flying = st["flight_end"] > escape._now(env)
            claw = torch.stack([(c.data.found.reshape(num, -1) > 0).any(-1) for c in claws]).any(0)
            body = (hit_s.data.found.reshape(num, -1) > 0).any(-1) & ~claw
            claw_at = torch.where((claw_at < 0) & flying & claw, torch.full_like(claw_at, step), claw_at)
            body_hit |= flying & body
            # turned back: 40 ms after the claw, under half its speed left along its line
            check = (claw_at >= 0) & (claw_at == step - look)
            if bool(check.any()):
                v0 = st["bolt_v"]
                along = (ball.data.root_link_lin_vel_w * v0).sum(-1) / (v0 * v0).sum(-1).clamp_min(1e-6)
                turned |= check & (along < 0.5)
            g = robot.data.projected_gravity_b
            fell |= torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) > math.radians(50)
    tally()
    rows = []
    for k, (c, b) in enumerate(cases):
        sl = slice(k * n, (k + 1) * n)
        row = {"command": c, "bearing_deg": b, "n": n, "throws": args.throws,
               "fell": round(float(fell[sl].float().mean()), 4)}
        row.update({key: round(float(v[sl].sum()) / (n * args.throws), 4) for key, v in counts.items()})
        rows.append(row)
        print(f"{c:5s} {b:+5.0f} deg  deflected {row['deflected']:.3f}  grazed {row['grazed']:.3f}  "
              f"hit {row['hit']:.3f}  missed {row['missed']:.3f}  fell {row['fell']:.3f}")
    total = num * args.throws
    meta = {"flight_s": args.flight, "dist_m": args.dist, "start_height_m": args.start_height,
            **{key: round(float(v.sum()) / total, 4) for key, v in counts.items()},
            "checkpoint": args.checkpoint, "vnc_config": args.vnc_config}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(json.dumps(meta))
    print(f"VNC_GUARD_EVAL_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
