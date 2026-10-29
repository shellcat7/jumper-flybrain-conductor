"""The conventional control: the official posture MLP, fine-tuned under the troubles.

Same network, rewards and PPO as the official run, resumed from its published
checkpoint (actor, critic, optimizer and curriculum levels); only the troubles in
`troubles.py` are added. Whatever the seven-graph brain gains has to be measured
against this, not only against the official policy as published.

    python experiments/reflex_v1/train_mlp.py --iterations 3000 --out /content/results/mlp-a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import troubles  # noqa: E402

TASK = "jumper.posture"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="warp", choices=["warp", "native"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--grip-low", type=float, default=0.03)
    p.add_argument("--shove", type=float, default=1.0)
    p.add_argument("--kill-prob", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--no-optimizer", action="store_true",
                   help="the checkpoint carries weights only: Adam starts fresh")
    p.add_argument("--no-troubles", action="store_true", help="the official training conditions only")
    p.add_argument("--grip-p-slip", type=float, default=None, help="v2 troubles (troubles.add_troubles)")
    p.add_argument("--official-shove", action="store_true", help="keep the official shoves (v2)")
    p.add_argument("--kill-legs", default="", help="only these legs die, e.g. LF,RF (a specialist)")
    p.add_argument("--lateral", type=float, default=None,
                   help="scale every sampled forward command by this (e.g. 0.2): a sideways specialist")
    p.add_argument("--sprint", type=float, default=None,
                   help="raise the command ladder's top rung to this speed (m/s; the official's is 0.8)")
    p.add_argument("--no-guard", action="store_true", help="without troubles.add_stability_guard")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend=args.backend, device=args.device, num_envs=args.num_envs)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    cfg = tasks.load_env_cfg(TASK)
    cfg.seed = args.seed
    cfg.scene.num_envs = res.num_envs
    added = {}
    if not args.no_troubles:
        added = troubles.add_troubles(cfg, grip_low=args.grip_low,
                                      shove=None if args.official_shove else args.shove,
                                      kill_prob=args.kill_prob, grip_p_slip=args.grip_p_slip,
                                      kill_legs=tuple(args.kill_legs.split(",")) if args.kill_legs else None)
    if not args.no_guard:
        added["guard"] = troubles.add_stability_guard(cfg)
    if args.sprint:
        # As train_vnc.py --sprint: the top rung (where the official checkpoint sits) is the
        # speed ceiling; raising it asks for more than the official gait was trained to give.
        levels = list(cfg.curriculum["command"].params["levels"])
        levels[-1] = args.sprint
        cfg.curriculum["command"].params["levels"] = tuple(levels)
        added["sprint"] = args.sprint
    agent_cfg = tasks.load_agent_cfg(TASK)
    agent_cfg.logger = "tensorboard"
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reflex-config.json").write_text(json.dumps({
        "task": TASK, "checkpoint": args.checkpoint, "num_envs": res.num_envs,
        "iterations": args.iterations, "seed": args.seed, "troubles": added, "lateral": args.lateral,
        "backend": res.backend, "device": str(res.device),
        "started_unix": time.time(),
    }, indent=2) + "\n")

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    if args.lateral is not None:
        # (10/26) The command ladder rewrites both linear ranges at every level, so the sideways
        # bias goes on the sample instead: each resample keeps vy and wz and shrinks vx.
        twist = env.command_manager.get_term("twist")
        resample = twist._resample_command

        def sideways(env_ids, _resample=resample, _twist=twist, _k=float(args.lateral)):
            _resample(env_ids)
            _twist.vel_command_b[env_ids, 0] *= _k

        twist._resample_command = sideways
    if not args.no_guard:
        troubles.clip_rewards(env, added["guard"]["reward_clip"])
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner_cls = tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), str(out), res.device)
        load_cfg = None
        if args.no_optimizer:
            load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True}
        runner.load(args.checkpoint, load_cfg=load_cfg, map_location=res.device)
        start = runner.current_learning_iteration
        print(f"[reflex] resumed the official run at iteration {start}; {args.iterations} more")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        print(f"REFLEX_V1_MLP_DONE {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
