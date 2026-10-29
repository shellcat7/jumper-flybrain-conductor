"""Evaluate the escape: balls thrown at walking robots on a fixed schedule; count the hits.

Each block of robots walks one command and receives throws from one bearing (body frame)
at fixed times, every throw on a ballistic arc that reaches the body after `--flight` s.
A hit is a contact between the ball and any robot geom during that throw's flight, read
from the contact sensor (the evaluation skips rewards, so the training's hit counter does
not run). The official brain never sees the looming signal: it is the control.

    python experiments/vnc_v1/eval_escape.py --robots-per-case 32 --out runs/vnc-v1/eval/escape-official
    python experiments/vnc_v1/eval_escape.py --checkpoint <model.pt> --vnc-config <cfg> ... --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
sys.path.insert(0, str(HERE))
import escape  # noqa: E402
import eval_official as ev  # noqa: E402

COMMANDS = {"walk": (0.3, 0.0, 0.0), "run": (0.6, 0.0, 0.0)}
BEARINGS = {"front": 0.0, "front_left": math.pi / 4, "left": math.pi / 2, "back_left": 3 * math.pi / 4,
            "back": math.pi, "right": -math.pi / 2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--robots-per-case", type=int, default=32)
    p.add_argument("--throws", type=int, default=4, help="throws per robot, one every --every s from 2 s")
    p.add_argument("--every", type=float, default=2.0)
    p.add_argument("--flight", type=float, default=0.4)
    p.add_argument("--dist", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--vnc-config", default="")
    p.add_argument("--vnc-packet", default="")
    p.add_argument("--vnc-layout", default="")
    p.add_argument("--balls", type=int, default=1,
                   help="balls per volley, thrown together from bearings --spread apart (the case's bearing first)")
    p.add_argument("--spread", type=float, default=None, help="degrees between a volley's bearings (default 360/balls)")
    p.add_argument("--stagger", type=float, default=0.0, help="s added to each further ball's flight")
    p.add_argument("--straight", action="store_true",
                   help="powered bolts (the film's energy shots): a straight line at constant speed, no arc")
    p.add_argument("--start-height", type=float, default=0.1, help="m above the ground a throw starts")
    p.add_argument("--roll", type=float, default=0.0,
                   help="roll the ball along the ground at this speed (m/s) instead of throwing it")
    p.add_argument("--scripted-dodge", type=float, default=0.0,
                   help="a hand-written reflex for the test: once the looming rate passes 1/tau > 1.5/s, "
                        "command this speed (m/s) across the ball's path and a 3 cm duck")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.escape = True

    import torch

    cases = [(c, b) for c in COMMANDS for b in BEARINGS]
    n = args.robots_per_case
    num = n * len(cases)
    env, wrapped, policy, res = ev.build(args, num)
    dev = env.device
    case_of = torch.arange(num, device=dev) // n
    cmd = torch.tensor([COMMANDS[c] for c, _ in cases], device=dev)[case_of]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    nb = args.balls
    spread = math.radians(360.0 / nb if args.spread is None else args.spread)
    sensors = [env.scene[escape.hit_sensor_name(j)] for j in range(nb)]
    states = [escape._state(env, j) for j in range(nb)]

    def pin(threat=None) -> None:
        twist.vel_command_b[:] = cmd
        posture.posture_command[:, :3] = 0.0
        posture.posture_command[:, 3] = posture.cfg.neutral_height
        if args.scripted_dodge and threat is not None:
            # Across the ball's horizontal path (the perpendicular with a forward component),
            # plus a duck, while the looming rate says contact is under about 0.7 s away.
            live = threat[:, 3] * 10.0 > 1.5
            dx, dy = threat[:, 0], threat[:, 1]
            norm = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-3)
            px, py = -dy / norm, dx / norm
            flip = px < 0
            px, py = torch.where(flip, -px, px), torch.where(flip, -py, py)
            dodge = torch.stack([cmd[:, 0] + args.scripted_dodge * px, args.scripted_dodge * py,
                                 torch.zeros_like(px)], dim=-1)
            twist.vel_command_b[:] = torch.where(live[:, None], dodge, cmd)
            posture.posture_command[:, 3] = torch.where(live, posture.cfg.neutral_height - 0.03,
                                                        posture.posture_command[:, 3])

    pin()
    obs = wrapped.get_observations()
    dt = env.step_dt
    throw_steps = [int(round((2.0 + k * args.every) / dt)) for k in range(args.throws)]
    steps = throw_steps[-1] + int(round(1.5 / dt))
    hits = torch.zeros(num, device=dev)  # balls that hit
    volley_hits = torch.zeros(num, device=dev)  # volleys with at least one hit
    fell = torch.zeros(num, dtype=torch.bool, device=dev)
    hit_now = torch.zeros(num, nb, dtype=torch.bool, device=dev)
    robot = env.scene["robot"]
    with torch.inference_mode():
        for step in range(steps):
            if step in throw_steps:
                volley_hits += hit_now.any(-1).float() if step != throw_steps[0] else 0.0
                for k, (_, b) in enumerate(cases):
                    ids = torch.arange(k * n, (k + 1) * n, device=dev)
                    for j in range(nb):
                        bearing = BEARINGS[b] + j * spread
                        if args.roll:
                            escape.roll_ball(env, ids, dist=args.dist, speed=args.roll, bearing=bearing, which=j)
                        else:
                            f = args.flight + j * args.stagger
                            escape.throw_ball(env, ids, prob=1.0, dist=(args.dist, args.dist), flight=(f, f),
                                              start_height=(args.start_height, args.start_height),
                                              aim_height=(0.09, 0.09), lead=1.0, bearing=bearing, which=j,
                                              straight=args.straight)
                hit_now[:] = False
            pin(obs["threat"] if "threat" in obs.keys() else None)
            if args.straight:
                for j in range(nb):
                    escape.hold_bolts(env, j, stop=hit_now[:, j])
            action = policy(obs)
            ev.apply_jump_gains(env, policy)
            obs, _, _, _ = wrapped.step(action)
            for j in range(nb):
                touching = (sensors[j].data.found.reshape(num, -1) > 0).any(-1)
                flying = states[j]["flight_end"] > escape._now(env)
                new = touching & flying & ~hit_now[:, j]
                hits += new.float()
                hit_now[:, j] |= new
            g = robot.data.projected_gravity_b
            fell |= torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) > math.radians(50)
    volley_hits += hit_now.any(-1).float()
    rows = []
    for k, (c, b) in enumerate(cases):
        s = slice(k * n, (k + 1) * n)
        rows.append({"command": c, "bearing": b, "n": n, "throws": args.throws,
                     "hit_rate": round(float(hits[s].sum() / (n * args.throws * nb)), 4),
                     "volley_hit_rate": round(float(volley_hits[s].sum() / (n * args.throws)), 4),
                     "fell": round(float(fell[s].float().mean()), 4)})
    total = round(float(hits.sum() / (num * args.throws * nb)), 4)
    volley_total = round(float(volley_hits.sum() / (num * args.throws)), 4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": args.checkpoint, "vnc_config": args.vnc_config or None, "robots": num,
            "throws": args.throws, "flight_s": args.flight, "dist_m": args.dist, "seed": args.seed,
            "roll_mps": args.roll or None, "every_s": args.every, "balls": nb,
            "spread_deg": round(math.degrees(spread), 1), "stagger_s": args.stagger,
            "straight": args.straight, "start_height_m": args.start_height,
            "hit_rate": total, "volley_hit_rate": volley_total}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(json.dumps(meta))
    for r in rows:
        print(f"{r['command']:5s} {r['bearing']:10s} hit {r['hit_rate']:.3f} fell {r['fell']:.3f}")
    print(f"VNC_ESCAPE_EVAL_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
