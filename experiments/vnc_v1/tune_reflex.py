"""Tune the hard-wired giant-fibre escape on the official gait: a grid of reflexes in one run.

The reflex: when the looming rate 1/tau (what drives the giant fibre, DNp01) first passes the
threshold during a throw, the fibre fires once and a fixed escape command is latched -- a step
of `speed` m/s in a direction set at that moment from the ball's bearing, plus an optional duck
-- and held until the throw is over. The official MLP carries the command out; nothing is
learnt. `eval_escape.py --scripted-dodge` (10/24) recomputed the direction at every step, and on
a throw from behind the sign of the sidestep flipped as soon as the robot moved, so it never got
away (hit 1.00 at 0.6 s); the latch is the giant fibre's all-or-none spike.

Directions, from the ball's horizontal bearing d (body frame, robot to ball):
- perp: across the ball's path, the side with the larger forward component;
- away: straight away from the ball (-d);
- blend: halfway between the two.

Each block of robots gets one reflex, one throw flight time, one walking command and one
bearing; hits are counted as in `eval_escape.py`.

    python experiments/vnc_v1/tune_reflex.py --out runs/vnc-v1/eval/reflex-grid-1
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
sys.path.insert(0, str(HERE))
import escape  # noqa: E402
import eval_official as ev  # noqa: E402
from eval_escape import BEARINGS, COMMANDS  # noqa: E402

MODES = ("perp", "away", "blend")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--robots-per-case", type=int, default=4)
    p.add_argument("--throws", type=int, default=4)
    p.add_argument("--every", type=float, default=2.0)
    p.add_argument("--dist", type=float, default=1.0)
    p.add_argument("--flights", type=float, nargs="+", default=[0.6, 0.9])
    p.add_argument("--thr", type=float, nargs="+", default=[0.8, 1.5], help="1/tau thresholds (1/s)")
    p.add_argument("--speed", type=float, nargs="+", default=[0.8, 1.2])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--duck", type=float, nargs="+", default=[0.0, 0.03])
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--vnc-config", default="")
    p.add_argument("--vnc-packet", default="")
    p.add_argument("--vnc-layout", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.escape = True

    import torch

    variants = list(itertools.product(args.thr, args.speed, args.modes, args.duck))
    cases = list(itertools.product(range(len(variants)), args.flights, COMMANDS, BEARINGS))
    n = args.robots_per_case
    num = n * len(cases)
    print(f"[reflex] {len(variants)} reflexes x {len(cases) // len(variants)} throw cases x {n} robots = {num}")
    env, wrapped, policy, res = ev.build(args, num)
    dev = env.device
    case_of = torch.arange(num, device=dev) // n
    col = lambda vals: torch.tensor(vals, device=dev, dtype=torch.float32)[case_of]  # noqa: E731
    thr = col([variants[v][0] for v, *_ in cases])
    speed = col([variants[v][1] for v, *_ in cases])
    mode = col([MODES.index(variants[v][2]) for v, *_ in cases])
    duck = col([variants[v][3] for v, *_ in cases])
    cmd = torch.tensor([COMMANDS[c] for _, _, c, _ in cases], device=dev)[case_of]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    sensor = env.scene["ball_hit"]
    st = escape._state(env)
    fired = torch.zeros(num, dtype=torch.bool, device=dev)
    dodge = torch.zeros(num, 2, device=dev)

    def pin(threat) -> None:
        nonlocal fired
        if threat is not None:
            # The escape lasts while the ball closes in (the looming signal is zero otherwise),
            # the rule `VNCModel`'s reflex uses, with no knowledge of the throw.
            live = threat.abs().sum(-1) > 0
            fired &= live
            new = ~fired & live & (threat[:, 3] * 10.0 > thr)
            d = threat[:, :2] / threat[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-3)
            perp = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
            perp = torch.where((perp[:, :1] < 0) | ((perp[:, :1] == 0) & (perp[:, 1:] < 0)), -perp, perp)
            away = -d
            blend = perp + away
            blend = blend / blend.norm(dim=-1, keepdim=True).clamp_min(1e-3)
            u = torch.where((mode == 0)[:, None], perp, torch.where((mode == 1)[:, None], away, blend))
            dodge[new] = u[new] * speed[new, None]
            fired |= new
        twist.vel_command_b[:] = cmd
        twist.vel_command_b[:, :2] += torch.where(fired[:, None], dodge, torch.zeros_like(dodge))
        posture.posture_command[:, :3] = 0.0
        posture.posture_command[:, 3] = posture.cfg.neutral_height - torch.where(fired, duck, torch.zeros_like(duck))

    pin(None)
    obs = wrapped.get_observations()
    dt = env.step_dt
    throw_steps = [int(round((2.0 + k * args.every) / dt)) for k in range(args.throws)]
    steps = throw_steps[-1] + int(round(1.5 / dt))
    hits = torch.zeros(num, device=dev)
    fell = torch.zeros(num, dtype=torch.bool, device=dev)
    hit_now = torch.zeros(num, dtype=torch.bool, device=dev)
    robot = env.scene["robot"]
    with torch.inference_mode():
        for step in range(steps):
            if step in throw_steps:
                for k, (_, flight, _, b) in enumerate(cases):
                    ids = torch.arange(k * n, (k + 1) * n, device=dev)
                    escape.throw_ball(env, ids, prob=1.0, dist=(args.dist, args.dist), flight=(flight, flight),
                                      start_height=(0.1, 0.1), aim_height=(0.09, 0.09), lead=1.0,
                                      bearing=BEARINGS[b])
                hit_now[:] = False
            pin(obs["threat"])
            obs, _, _, _ = wrapped.step(policy(obs))
            touching = (sensor.data.found.reshape(num, -1) > 0).any(-1)
            flying = st["flight_end"] > escape._now(env)
            new = touching & flying & ~hit_now
            hits += new.float()
            hit_now |= new
            g = robot.data.projected_gravity_b
            fell |= torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) > math.radians(50)
    per_case = (hits.view(len(cases), n).sum(-1) / (n * args.throws)).tolist()
    fell_case = fell.view(len(cases), n).float().mean(-1).tolist()
    rows = []
    for v, (t, s, m, dk) in enumerate(variants):
        ks = [k for k, c in enumerate(cases) if c[0] == v]
        row = {"thr": t, "speed": s, "mode": m, "duck": dk,
               "hit_rate": round(sum(per_case[k] for k in ks) / len(ks), 4),
               "fell": round(sum(fell_case[k] for k in ks) / len(ks), 4)}
        for flight in args.flights:
            kf = [k for k in ks if cases[k][1] == flight]
            row[f"hit_f{flight}"] = round(sum(per_case[k] for k in kf) / len(kf), 4)
        for b in BEARINGS:
            kb = [k for k in ks if cases[k][3] == b]
            row[f"hit_{b}"] = round(sum(per_case[k] for k in kb) / len(kb), 4)
        rows.append(row)
    rows.sort(key=lambda r: r["hit_rate"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": args.checkpoint, "vnc_config": args.vnc_config or None, "robots": num,
            "robots_per_case": n, "throws": args.throws, "flights": args.flights, "dist_m": args.dist,
            "seed": args.seed}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(json.dumps(meta))
    for r in rows:
        print(f"thr {r['thr']:.1f} speed {r['speed']:.1f} {r['mode']:5s} duck {r['duck']:.2f}: hit {r['hit_rate']:.3f} "
              + " ".join(f"{k[4:]} {v:.2f}" for k, v in r.items() if k.startswith("hit_")) + f" fell {r['fell']:.2f}")
    print(f"VNC_REFLEX_GRID_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
