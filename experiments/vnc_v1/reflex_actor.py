"""Give a VNC actor the hard-wired giant-fibre escape (`VNCModel.reflex`).

Writes a vnc-config.json that `eval_escape.py` and `eval_official.py` read. From a trained run's
config (`--base-config`) the checkpoint stays that run's; with none, it also writes model_0.pt,
a cord at its initial state, whose residual is zero -- the official gait with the reflex and
nothing else.

    python experiments/vnc_v1/reflex_actor.py --out runs/vnc-v1/reflex/official-gf
    python experiments/vnc_v1/reflex_actor.py --base-config runs/vnc-v1/colab/dgate-s1/vnc-config.json \
        --out runs/vnc-v1/reflex/dgate-gf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-config", default="")
    p.add_argument("--thr", type=float, default=0.8, help="1/tau at which the giant fibre fires (1/s)")
    p.add_argument("--speed", type=float, default=0.6, help="the escape step (m/s)")
    p.add_argument("--mode", default="perp", choices=["perp", "away", "blend"])
    p.add_argument("--duck", type=float, default=0.0, help="m")
    p.add_argument("--lean", type=float, default=0.0,
                   help="rad the trunk tilts away from the ball while the escape runs (walking band: 0.26)")
    p.add_argument("--lines", action="store_true",
                   help="with several balls closing in, step clear of every ball's line (`VNCModel._clear_lines`)")
    p.add_argument("--lines-cap", type=float, default=0.28, help="m across a ball's line counted as clear")
    p.add_argument("--guard", default="",
                   help="J0,J1,J2,J3: the left claw's guard pose in posture-action units; a lone low bolt "
                        "from ahead is met with the claw on the five-legged gait (`VNCModel.guard_override`)")
    p.add_argument("--guard-steps", type=int, default=120, help="control steps the guard is held")
    p.add_argument("--guard-kp", type=float, default=0.0, help="the guarding arm's stiffness (0: the task's 10)")
    p.add_argument("--jump", action="store_true",
                   help="balls from the front or behind get the official jump instead of a sidestep")
    p.add_argument("--jump-tau", type=float, default=0.45, help="seconds to contact at which the jump starts")
    p.add_argument("--jump-ahead-only", action="store_true",
                   help="the single-ball jump only for a ball from ahead, not from behind")
    p.add_argument("--jump-max-elev", type=float, default=0.1,
                   help="sine of the ball's elevation under which it counts as low enough to jump over")
    p.add_argument("--jump-steps", type=int, default=240, help="control steps the jump program runs")
    p.add_argument("--jump-checkpoint", default="tasks/jumper/ref_free_jump/out/example/model_5999.pt")
    p.add_argument("--jump-go", default="../../../runs/vnc-v1/packet-1/jump_go_force.npy")
    p.add_argument("--packet", default="runs/vnc-v1/packet-1/vnc_ge5.npz")
    p.add_argument("--layout", default="runs/vnc-v1/packet-1/layout.json")
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import tasks

    # The gait clock's tempo law, read from the task (VNCModel keeps its own phase offset).
    law = tasks.load_env_cfg("jumper.posture").observations["actor"].terms["gait_phase"].params
    cadence = {k: float(law[k]) for k in ("stride", "freq_min", "freq_max", "turn_radius")}
    reflex = {"thr": args.thr, "speed": args.speed, "mode": args.mode, "duck": args.duck, "cadence": cadence}
    if args.lean:
        reflex["lean"] = args.lean
    if args.lines:
        reflex["lines"] = {"n": 64, "t0": 0.07, "gain": 0.85, "cap": args.lines_cap}
    if args.guard:
        reflex["guard"] = {"pose": [float(v) for v in args.guard.split(",")], "steps": args.guard_steps,
                           "cone": 0.8, "max_elev": 0.3}
        if args.guard_kp:
            reflex["guard"]["kp"] = args.guard_kp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.base_config:
        cfg = json.loads(Path(args.base_config).read_text())
    else:
        import torch
        from tensordict import TensorDict

        from distill_vnc import actor_kwargs
        from vnc_model import VNCModel

        cfg = {"task": "jumper.posture", "method": "official gait + giant-fibre reflex",
               "base_checkpoint": args.checkpoint, "packet": args.packet, "layout": args.layout, "width": 1,
               "residual_scale": 0.5, "shuffle_seed": None, "brain_hidden": 0, "learn_synapses": False,
               "command_readout": False, "mlp_anchor": None, "experts": []}
        agent_cfg = tasks.load_agent_cfg("jumper.posture")
        obs = TensorDict({"actor": torch.zeros(1, 415), "threat": torch.zeros(1, 5)}, batch_size=[1])
        model = VNCModel(obs, {"actor": ["actor", "threat"]}, "actor", 20, **actor_kwargs(agent_cfg),
                         packet=args.packet, layout=args.layout, base_checkpoint=args.checkpoint,
                         residual_scale=0.5, gf_reflex=reflex, threat_cord=False)
        torch.save({"actor_state_dict": model.state_dict(), "iter": 0, "infos": None}, out / "model_0.pt")
    cfg.update({"escape": True, "threat_cord": False, "gf_reflex": reflex})
    if args.jump:
        cfg["gf_jump"] = {"checkpoint": args.jump_checkpoint, "go": args.jump_go, "tau": args.jump_tau,
                          "steps": args.jump_steps, "unit": 2.0,  # jump action scale 0.5 / posture 0.25
                          "max_elev": args.jump_max_elev, "ahead_only": args.jump_ahead_only}
    (out / "vnc-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(reflex), "->", out)


if __name__ == "__main__":
    main()
