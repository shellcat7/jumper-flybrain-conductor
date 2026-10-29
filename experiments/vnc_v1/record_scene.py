"""Record a scene take for the film: the robot's motion, its motors, and the cord's activity.

Builds the evaluation environment of `experiments/reflex_v1/eval_official.py` (no resets,
pinned commands, the training's sensor noise), runs one scripted scenario on `--takes`
robots side by side (different noise, same script) and saves per control step (200 Hz):

- every body's world position and orientation, for Blender to animate the meshes directly;
- joint positions, actuator torque and each joint's normalised winding temperature
  (the official servo model: 1.0 is where it derates from the 1.75 N*m peak to 1.2 N*m);
- the command and the trouble schedule;
- for a VNC actor, the cord state (every neuron) every `--cord-every` steps as float16; for
  the director, its choice among the motor programs (softmax, every step); with the
  giant-fibre reflex, whether it is firing;
- for a scenario with throws, the ball's position.

The film shows only what a take really did; the numbers it quotes come from batch
evaluation, not from the take.

    python experiments/vnc_v1/record_scene.py --scenario slip_corner --takes 8 \
        --checkpoint <model.pt> --vnc-config <vnc-config.json> --out runs/vnc-v1/scenes/slip_vnc.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reflex_v1"))
import eval_official as ev  # noqa: E402
from troubles import LEGS  # noqa: E402

#: Body-frame command (vx, vy, wz) and the troubles, each (time s, kind, value).
SCENARIOS = {
    "slip_corner": {"command": (0.5, 0.0, 1.5), "duration": 6.0, "events": [(2.0, "grip", 0.03)]},
    "slip_straight": {"command": (0.6, 0.0, 0.0), "duration": 6.0, "events": [(2.0, "grip", 0.03)]},
    "dead_front_escape": {"command": (0.3, 0.0, 0.0), "duration": 8.0, "events": [(2.0, "dead", "LF")]},
    "dead_middle_run": {"command": (0.6, 0.0, 0.0), "duration": 8.0, "events": [(2.0, "dead", "LM")]},
    "shove": {"command": (0.3, 0.0, 0.0), "duration": 6.0, "events": [(2.0, "shove", 1.0)]},
    "nominal_run": {"command": (0.6, 0.0, 0.0), "duration": 6.0, "events": []},
    # Throws (bearing as in eval_escape.BEARINGS, flight time s) from 1 m, aimed 9 cm up.
    "dodge_sides": {"command": (0.3, 0.0, 0.0), "duration": 7.0,
                    "events": [(2.0, "throw", ("left", 0.5)), (4.5, "throw", ("right", 0.5))]},
    "dodge_front_back": {"command": (0.3, 0.0, 0.0), "duration": 7.0,
                         "events": [(2.0, "throw", ("front", 0.6)), (4.5, "throw", ("back", 0.6))]},
    "dodge_run": {"command": (0.6, 0.0, 0.0), "duration": 7.0,
                  "events": [(2.0, "throw", ("left", 0.4)), (4.5, "throw", ("front_left", 0.6))]},
    "slip_then_dodge": {"command": (0.5, 0.0, 1.5), "duration": 8.0,
                        "events": [(2.0, "grip", 0.03), (5.0, "throw", ("right", 0.5))]},
    # The director's other programs: a front leg dying in a corner (mlp-b), a dash (the sprint gait).
    "dead_front_corner": {"command": (0.5, 0.0, 1.5), "duration": 8.0, "events": [(2.0, "dead", "LF")]},
    "sprint_dash": {"command": (2.0, 0.0, 0.0), "duration": 6.0, "events": []},
    # A ball rolled along the ground (bearing, speed m/s) from 1.5 m: low, from the front, so the
    # giant fibre picks the jump (with `reflex_actor.py --jump`).
    "jump_over_roll": {"command": (0.3, 0.0, 0.0), "duration": 6.0, "events": [(2.0, "roll", ("front", 1.5))]},
    # Several shots (10/26): a volley of three from three sides at once, then a second, and one
    # ball re-thrown every 0.8 s from changing sides (the reflex re-arms between shots).
    "volley_three": {"command": (0.3, 0.0, 0.0), "duration": 7.0,
                     "events": [(2.0, "volley", [("front_left", 0.6), ("back", 0.6), ("right", 0.6)]),
                                (4.5, "volley", [("left", 0.5), ("front", 0.65), ("back_left", 0.8)])]},
    # Energy shots (10/26): straight bolts from drones 2.5-3 m out, 0.4-0.5 s of warning (6-7.5 m/s).
    "bolt_volley": {"command": (0.3, 0.0, 0.0), "duration": 7.0,
                    "events": [(2.0, "bolt_volley", [("front_left", 0.4, 3.0), ("back", 0.45, 3.0), ("right", 0.5, 3.0)]),
                               (4.5, "bolt_volley", [("left", 0.4, 2.5), ("front", 0.45, 3.0), ("back_left", 0.5, 3.0)])]},
    "bolt_rapid": {"command": (0.3, 0.0, 0.0), "duration": 7.0,
                   "events": [(2.0, "bolt", ("left", 0.4, 3.0)), (2.75, "bolt", ("right", 0.4, 3.0)),
                              (3.5, "bolt", ("front_left", 0.4, 3.0)), (4.25, "bolt", ("back", 0.4, 3.0)),
                              (5.0, "bolt", ("left", 0.4, 3.0))]},
    # An assault: four waves from all round, staggered, every bolt with 0.4 s or more of warning.
    "assault": {"command": (0.3, 0.0, 0.0), "duration": 7.5,
                "events": [(1.5, "bolt_volley", [("front_left", 0.45, 3.0), ("right", 0.55, 3.0)]),
                           (2.6, "bolt_volley", [("back", 0.4, 3.0), ("left", 0.5, 2.8), ("front", 0.6, 3.2)]),
                           (3.8, "bolt_volley", [("back_left", 0.45, 3.0), ("right", 0.55, 3.0),
                                                 ("front_left", 0.65, 3.0)]),
                           (5.0, "bolt_volley", [("left", 0.4, 2.5), ("back", 0.5, 3.0), ("front", 0.6, 3.0),
                                                 ("right", 0.7, 3.0)])]},
    # A ball is busy until 0.3 s past its arrival and a throw at a busy ball is skipped, so a
    # single bolt (ball 0) waits for the last one on ball 0.
    # The film's assault (10/26), after the storyboard: the drones fire from 0.6 m up; waves the
    # reflex clears (0.2 s or more between arrivals), then three fired at once from three sides --
    # beyond what any step clears in 0.45 s (batch: 0.45-0.49 of such bolts hit) -- and the leg
    # nearest the hit dies ("wound"); the robot runs on, on five legs.
    "assault_wound": {"command": (0.3, 0.0, 0.0), "duration": 10.0, "bolt_height": 0.6, "wound": True,
                      "events": [(1.5, "bolt_volley", [("front_left", 0.45, 3.0), ("right", 0.65, 3.0)]),
                                 (2.7, "bolt_volley", [("back", 0.45, 3.0), ("left", 0.65, 3.0), ("front", 0.85, 3.0)]),
                                 (4.0, "bolt", ("right", 0.45, 3.0)),
                                 (4.9, "bolt", ("back_left", 0.45, 3.0)),
                                 (5.8, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 3.0),
                                                       ("right", 0.45, 3.0)]),
                                 (6.6, "command", (0.6, 0.0, 0.0))]},
    # With the jump (`reflex_actor.py --jump`, 10/26): a bolt from ahead is leapt over, then three at
    # once from three sides (no step clears them: the crowd jump), and last a wave too long to leap
    # (the fourth bolt arrives 0.3 s after the rest) and too dense to step out of -- a hit, a wound.
    "assault_jump": {"command": (0.3, 0.0, 0.0), "duration": 11.0, "bolt_height": 0.6, "wound": True,
                     "events": [(1.5, "bolt_volley", [("front_left", 0.45, 3.0), ("right", 0.65, 3.0)]),
                                (2.8, "bolt", ("front", 0.45, 3.0)),
                                (4.3, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 3.0),
                                                      ("right", 0.45, 3.0)]),
                                (5.8, "bolt", ("left", 0.45, 3.0)),
                                (6.6, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 3.0),
                                                      ("right", 0.45, 3.0), ("left", 0.75, 3.0)]),
                                (7.4, "command", (0.6, 0.0, 0.0))]},
    # The same assault in a back alley 4.4 m wide (10/26): the drones must stay between the walls, so
    # bolts from the side start 1.8 m out and diagonal ones 2.6 m (1.84 m to the side); the flight
    # times -- the warning, which is what sets a dodge -- are unchanged.
    "alley_jump": {"command": (0.3, 0.0, 0.0), "duration": 11.0, "bolt_height": 0.6, "wound": True,
                   "events": [(1.5, "bolt_volley", [("front_left", 0.45, 2.6), ("right", 0.65, 1.8)]),
                              (2.8, "bolt", ("front", 0.45, 3.0)),
                              (4.3, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6),
                                                    ("right", 0.45, 1.8)]),
                              (5.8, "bolt", ("left", 0.45, 1.8)),
                              (6.6, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6),
                                                    ("right", 0.45, 1.8), ("left", 0.75, 1.8)]),
                              (7.4, "command", (0.6, 0.0, 0.0))]},
    # The round-1 short (10/28): the alley take with the real parry. The drones fire from 25 cm up,
    # where the centred claw at kp 40 deflects 0.90 of lone bolts from ahead (`gfmgc` reflex config):
    # a side bolt stepped from, a lone bolt from ahead parried, three at once leapt over, a side bolt
    # again, then the wave too long to leap that wounds on landing; the run on five legs after.
    "alley_parry": {"command": (0.3, 0.0, 0.0), "duration": 13.0, "bolt_height": 0.25, "wound": True,
                    "events": [(1.5, "bolt", ("left", 0.45, 1.8)),
                               (2.8, "bolt", ("front", 0.45, 3.0)),
                               (4.3, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6),
                                                     ("right", 0.45, 1.8)]),
                               (5.8, "bolt", ("right", 0.45, 1.8)),
                               (6.6, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6),
                                                     ("right", 0.45, 1.8), ("left", 0.75, 1.8)]),
                               (7.4, "command", (0.6, 0.0, 0.0))]},
    "rapid_fire": {"command": (0.3, 0.0, 0.0), "duration": 7.5,
                   "events": [(2.0, "throw", ("left", 0.45)), (2.8, "throw", ("right", 0.45)),
                              (3.6, "throw", ("front_left", 0.45)), (4.4, "throw", ("back", 0.45)),
                              (5.2, "throw", ("left", 0.45))]},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--takes", type=int, default=8)
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--obs-noise", type=int, default=1)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--vnc-config", default="")
    p.add_argument("--vnc-packet", default="")
    p.add_argument("--vnc-layout", default="")
    p.add_argument("--cord-every", type=int, default=4, help="cord state every N control steps (4: 50 Hz)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import numpy as np
    import torch

    sc = SCENARIOS[args.scenario]
    n = args.takes
    throws = any(kind in ("throw", "roll", "volley", "bolt", "bolt_volley") for _, kind, _ in sc["events"])
    # A volley throws several balls at once, one per (bearing, flight); the scene gets that many.
    args.balls = max([len(value) for _, kind, value in sc["events"] if kind in ("volley", "bolt_volley")] + [1])
    # Bolts (the energy shots): straight powered shots from 25 cm up, held on their line until they hit.
    bolts = any(kind in ("bolt", "bolt_volley") for _, kind, _ in sc["events"])
    if throws:
        args.escape = True  # the ball and the looming signal (ev.build adds them)
        sys.path.insert(0, str(HERE))
        import escape
        from eval_escape import BEARINGS
    env, wrapped, policy, res = ev.build(args, n)
    dev = env.device
    robot = env.scene["robot"]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term("posture")
    cmd = torch.tensor(sc["command"], device=dev).repeat(n, 1)
    every = torch.arange(n, device=dev)

    def pin_commands() -> None:
        twist.vel_command_b[:] = cmd
        posture.posture_command[:, :3] = 0.0
        posture.posture_command[:, 3] = posture.cfg.neutral_height

    ev.set_grip(env, every, ev.NOMINAL_GRIP)
    pin_commands()
    obs = wrapped.get_observations()
    dt = env.step_dt
    steps = int(round(sc["duration"] / dt))
    events = {int(round(t / dt)): (kind, value) for t, kind, value in sc["events"]}
    act = robot.actuators[0]
    is_vnc = hasattr(policy, "hidden_state") and getattr(policy, "n", None) is not None

    rec = {k: [] for k in ("body_pos", "body_quat", "joint_pos", "torque", "heat", "gate", "gf_fired", "ball_pos",
                           "balls_pos", "jumping", "struck", "wounded")}
    cord = []
    director = is_vnc and getattr(policy, "has_gate", False)
    struck = [torch.zeros(n, dtype=torch.bool, device=dev) for _ in range(args.balls)]  # a bolt that has hit falls
    bolt_hits = torch.zeros(n, device=dev)  # bolts that hit, each counted once
    bolt_blocks = torch.zeros(n, device=dev)  # bolts the claw guard took
    wounded = torch.full((n,), -1, dtype=torch.long, device=dev)  # the leg a hit killed ("wound")
    leg_bodies = [[i for i, name in enumerate(robot.body_names) if name.startswith(f"{leg}_")] for leg in LEGS]
    with torch.inference_mode():
        for step in range(steps):
            if step in events:
                kind, value = events[step]
                if kind == "grip":
                    ev.set_grip(env, every, value)
                elif kind == "dead":
                    ev.kill_leg(env, every, value)
                elif kind == "shove":
                    ev.shove(env, every, value)
                elif kind == "throw":
                    bearing, flight = value
                    escape.throw_ball(env, every, prob=1.0, dist=(1.0, 1.0), flight=(flight, flight),
                                      start_height=(0.1, 0.1), aim_height=(0.09, 0.09), lead=1.0,
                                      bearing=BEARINGS[bearing])
                elif kind == "volley":
                    for j, (bearing, flight) in enumerate(value):
                        escape.throw_ball(env, every, prob=1.0, dist=(1.0, 1.0), flight=(flight, flight),
                                          start_height=(0.1, 0.1), aim_height=(0.09, 0.09), lead=1.0,
                                          bearing=BEARINGS[bearing], which=j)
                elif kind in ("bolt", "bolt_volley"):
                    for j, (bearing, flight, dist) in enumerate([value] if kind == "bolt" else value):
                        h0 = sc.get("bolt_height", 0.25)
                        escape.throw_ball(env, every, prob=1.0, dist=(dist, dist), flight=(flight, flight),
                                          start_height=(h0, h0), aim_height=(0.09, 0.09), lead=1.0,
                                          bearing=BEARINGS[bearing], which=j, straight=True)
                        struck[j][:] = False
                elif kind == "roll":
                    bearing, speed = value
                    escape.roll_ball(env, every, dist=1.5, speed=speed, bearing=BEARINGS[bearing])
                elif kind == "command":
                    cmd[:] = torch.tensor(value, device=dev)
            pin_commands()
            if bolts:
                for j in range(args.balls):
                    touch = (env.scene[escape.hit_sensor_name(j)].data.found.reshape(n, -1) > 0).any(-1)
                    # only in flight, as eval_escape counts: a spent bolt lying on the ground that a
                    # foot kicks later is no hit (10/26: "hits" at 8.5 s, a second after the last
                    # bolt landed)
                    touch = touch & (escape._state(env, j)["flight_end"] > escape._now(env))
                    # a bolt met while the claw guard is up is a block (the claw takes it), not a hit
                    gt = getattr(policy, "_guard_t", None)
                    guarding = gt >= 0 if gt is not None else torch.zeros_like(touch)
                    bolt_blocks += (touch & ~struck[j] & guarding).float()
                    bolt_hits += (touch & ~struck[j] & ~guarding).float()
                    if sc.get("wound"):  # the first hit kills the leg nearest the bolt, for good
                        new = torch.nonzero(touch & ~struck[j] & (wounded < 0) & ~guarding).flatten()
                        if len(new):
                            at = env.scene[escape.ball_name(j)].data.root_link_pos_w[new]
                            parts = robot.data.body_link_pos_w[new]
                            near = torch.stack([(parts[:, ids] - at[:, None]).norm(dim=-1).min(dim=1).values
                                                for ids in leg_bodies], dim=1).argmin(dim=1)
                            for i, k in zip(new.tolist(), near.tolist()):
                                ev.kill_leg(env, torch.tensor([i], device=dev), LEGS[k])
                                wounded[i] = k
                    struck[j] |= touch
                    escape.hold_bolts(env, j, stop=struck[j])
                rec["struck"].append(torch.stack(struck, dim=1).cpu().numpy())  # [takes, balls]: hit since fired
                rec["wounded"].append(wounded.cpu().numpy())
            action = policy(obs)
            ev.apply_jump_gains(env, policy)
            obs, _, _, _ = wrapped.step(action)
            if getattr(policy, "jump_active", None) is not None:
                rec["jumping"].append(policy.jump_active.cpu().numpy())
            if director:
                rec["gate"].append(torch.softmax(policy.gate(policy.hidden_state[0]), dim=-1).cpu().numpy().astype(np.float16))
            if getattr(policy, "gf_reflex", None) and policy._gf_fired is not None:
                rec["gf_fired"].append(policy._gf_fired.cpu().numpy())
            if throws:
                rec["ball_pos"].append(env.scene["ball"].data.root_link_pos_w.cpu().numpy().astype(np.float32))
            if args.balls > 1:  # every ball, [takes, balls, 3] per step
                rec["balls_pos"].append(np.stack([env.scene[escape.ball_name(j)].data.root_link_pos_w.cpu().numpy()
                                                  for j in range(args.balls)], axis=1).astype(np.float32))
            rec["body_pos"].append(robot.data.body_link_pos_w.cpu().numpy().astype(np.float32))
            rec["body_quat"].append(robot.data.body_link_quat_w.cpu().numpy().astype(np.float32))
            rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
            rec["torque"].append(robot.data.actuator_force.cpu().numpy().astype(np.float32))
            heat = getattr(act, "_thermal", None)
            if heat is not None:
                rec["heat"].append(heat.cpu().numpy().astype(np.float16))
            if is_vnc and step % args.cord_every == 0 and policy.hidden_state is not None:
                cord.append(policy.hidden_state[0].cpu().numpy().astype(np.float16))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack(v, axis=1) for k, v in rec.items() if v}  # [takes, steps, ...]
    if cord:
        arrays["cord"] = np.stack(cord, axis=1)  # [takes, steps / cord_every, nodes * width]
    meta = {
        "scenario": args.scenario, "spec": sc, "takes": n, "control_dt": dt, "seed": args.seed,
        "checkpoint": args.checkpoint, "vnc_config": args.vnc_config or None,
        "cord_every": args.cord_every if cord else None,
        "programs": (["official"] + [Path(p).stem for p in policy.experts_paths] + list(policy.adapter_names))
                    if director else None,
        "body_names": list(robot.body_names), "joint_names": list(robot.joint_names),
        "actuator_names": list(act.target_names) if hasattr(act, "target_names") else None,
    }
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    sizes = {k: list(v.shape) for k, v in arrays.items()}
    print(json.dumps({"out": str(out), "shapes": sizes}))
    if bolts:
        print(json.dumps({"bolt_hits_per_take": bolt_hits.int().tolist(),
                          "bolt_blocks_per_take": bolt_blocks.int().tolist(),
                          "wounded_leg_per_take": [LEGS[k] if k >= 0 else None for k in wounded.tolist()]}))
    print(f"VNC_SCENE_DONE {out}")
    env.close()


if __name__ == "__main__":
    main()
