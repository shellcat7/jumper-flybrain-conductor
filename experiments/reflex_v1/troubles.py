"""The reflex troubles as training events: grip that drops mid-stride, a leg that
dies, shoves. Shared by every learner in reflex v1 (the conventional MLP and the
seven-graph brain), so a difference between them is not a difference in what
they trained against.

The evaluation (`eval_official.py`) strikes once at a fixed time; training draws
the same kinds of trouble at random times, never the evaluation's schedule.
"""

from __future__ import annotations

LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")


def _dead_flags(env):
    import torch

    flags = getattr(env, "_reflex_dead_leg", None)
    if flags is None:
        flags = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._reflex_dead_leg = flags
    return flags


def _second_flags(env):
    """The second leg of a pair that died together (-1: none), beside `_dead_flags`' first."""
    import torch

    flags = getattr(env, "_reflex_dead_leg2", None)
    if flags is None:
        flags = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._reflex_dead_leg2 = flags
    return flags


def restore_legs(env, env_ids) -> None:
    """Reset event: every leg answers again."""
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    for act in robot.actuators:
        if getattr(act, "default_force_limit", None) is not None:
            act.set_effort_limit(env_ids, effort_limit=act.default_force_limit[env_ids])
    _dead_flags(env)[env_ids] = -1
    _second_flags(env)[env_ids] = -1


def kill_random_leg(env, env_ids, prob: float, legs: tuple[str, ...] | None = None) -> None:
    """Interval event: with probability `prob`, a robot with all legs working loses
    one at random, from `legs` if given (zero torque limit; the brain still commands it).
    An entry joined with "+" (e.g. "LM+LR") is a pair of legs that die together; the flags
    record its first and second (`_second_flags`)."""
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    flags = _dead_flags(env)
    alive = env_ids[flags[env_ids] < 0]
    hit = alive[torch.rand(len(alive), device=env.device) < prob]
    if len(hit) == 0:
        return
    combos = [[LEGS.index(name) for name in entry.split("+")] for entry in (legs or LEGS)]
    pick = torch.randint(0, len(combos), (len(hit),), device=env.device)
    flags[hit] = torch.tensor([c[0] for c in combos], device=env.device)[pick]
    _second_flags(env)[hit] = torch.tensor([c[1] if len(c) > 1 else -1 for c in combos], device=env.device)[pick]
    robot = env.scene["robot"]
    for act in robot.actuators:
        if act.force_limit is None:
            continue
        for ci, combo in enumerate(combos):
            ids = hit[pick == ci]
            cols = [i for i, n in enumerate(act.target_names) if any(n.startswith(f"{LEGS[k]}_") for k in combo)]
            if len(ids) == 0 or not cols:
                continue
            limit = act.force_limit[ids].clone()
            limit[:, cols] = 0.0
            act.set_effort_limit(ids, effort_limit=limit)


def unstable_state(env, max_lin: float = 10.0, max_ang: float = 100.0) -> "torch.Tensor":
    """Termination: a robot moving at a speed the hardware cannot (base above `max_lin`
    m/s or `max_ang` rad/s), or with a non-finite state -- a solver blow-up, not behaviour."""
    import torch

    robot = env.scene["robot"]
    lin = robot.data.root_link_lin_vel_w.norm(dim=-1)
    ang = robot.data.root_link_ang_vel_w.norm(dim=-1)
    finite = torch.isfinite(robot.data.joint_vel).all(-1) & torch.isfinite(lin) & torch.isfinite(ang)
    return ~finite | (lin > max_lin) | (ang > max_ang)


def add_stability_guard(cfg, reward_clip: float = 1.0) -> dict:
    """Terminate solver blow-ups and clip each step's reward, in place.

    On 10/24 every run logged rare value-loss spikes (flyv4a 1.4e7 at iteration 71, during
    the critic warm-up with the official actor) from steps whose reward was off by many orders
    (mean episode rewards of -1.6e10 in hybv2-s1); hybv2-s1 then diverged to an infinite value
    loss once its actor learned. A normal step pays about 0.03 here, so a clip at +-1 changes
    nothing else.
    """
    from mjlab.envs.mdp import terminations as mdp_terms
    from mjlab.managers.termination_manager import TerminationTermCfg

    cfg.terminations["nan"] = TerminationTermCfg(func=mdp_terms.nan_detection)
    cfg.terminations["unstable"] = TerminationTermCfg(func=unstable_state)
    return {"terminations": ["nan", "unstable"], "reward_clip": reward_clip}


def clip_rewards(env, limit: float) -> None:
    """Wrap the reward manager so each step's total reward is finite and within +-limit."""
    import torch

    manager = env.reward_manager
    compute = manager.compute

    def clipped(dt):
        return torch.clamp(torch.nan_to_num(compute(dt), nan=0.0, posinf=limit, neginf=-limit), -limit, limit)

    manager.compute = clipped


def grip_mixture(env, env_ids, slip: tuple, nominal: tuple, p_slip: float, asset_cfg) -> None:
    """Startup/interval event: each robot's foot friction comes from the slippery range
    (log-uniform) with probability `p_slip`, else from the nominal range (uniform)."""
    import torch
    from mjlab.envs.mdp import dr

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    slipping = torch.rand(len(env_ids), device=env.device) < p_slip
    for mask, rng, dist in ((slipping, slip, "log_uniform"), (~slipping, nominal, "uniform")):
        ids = env_ids[mask]
        if len(ids):
            dr.geom_friction(env, ids, ranges=rng, distribution=dist, asset_cfg=asset_cfg, operation="abs")


def add_troubles(cfg, grip_low: float = 0.03, grip_high: float = 1.5, shove: float | None = 1.0,
                 kill_prob: float = 0.08, grip_p_slip: float | None = None,
                 grip_slip_high: float = 0.3, kill_legs: tuple[str, ...] | None = None) -> dict:
    """Add the troubles to a `jumper.posture` training config, in place.

    v1 (the defaults, 10/23-24): grip log-uniform over [grip_low, grip_high], shoves
    widened to +-`shove` m/s. v2 (`grip_p_slip` set, `shove` None): grip is nominal
    (0.5-1.5) except with probability `grip_p_slip`, when it is log-uniform over
    [grip_low, grip_slip_high]; shoves stay the official +-0.2 m/s. On 10/24 the v1
    runs gave up nominal speed (fly-s1 at iteration 100 walked at 0.69 of the command
    against the official 0.93): v1 has grip under 0.3 63% of the time, a shove up to
    1.5 m/s every 1-3 s and a dead leg in most episodes, so caution pays everywhere.
    """
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from tasks.jumper.common.constants import FOOT_GEOMS

    # The event manager gives a model field per-robot memory only when some event's
    # function declares it. Without this, every robot shares one friction (the native
    # backend refuses the write; warp takes it silently -- the first v2 runs of 10/24
    # trained that way and were restarted).
    requires_model_fields("geom_friction")(grip_mixture)
    feet = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
    if grip_p_slip is None:
        # Training's own startup draw was 0.5-1.5. Log-uniform so the slippery end,
        # which is the point, is not a sliver of the range.
        grip = {"ranges": (grip_low, grip_high), "distribution": "log_uniform",
                "asset_cfg": feet, "operation": "abs"}
        cfg.events["foot_friction"].params.update(grip)
        func = dr.geom_friction
    else:
        grip = {"slip": (grip_low, grip_slip_high), "nominal": (0.5, 1.5), "p_slip": grip_p_slip,
                "asset_cfg": feet}
        cfg.events["foot_friction"].func = grip_mixture
        cfg.events["foot_friction"].params = dict(grip)
        func = grip_mixture
    cfg.events["grip_change"] = EventTermCfg(
        func=func, mode="interval", interval_range_s=(2.0, 6.0), params=dict(grip))
    if shove is not None and "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"].update(
            {"x": (-shove, shove), "y": (-shove, shove)})
    cfg.events["leg_restore"] = EventTermCfg(func=restore_legs, mode="reset")
    cfg.events["leg_kill"] = EventTermCfg(
        func=kill_random_leg, mode="interval", interval_range_s=(1.0, 4.0),
        params={"prob": kill_prob, "legs": tuple(kill_legs) if kill_legs else None})
    return {"grip": [grip_low, grip_high], "grip_p_slip": grip_p_slip, "grip_slip_high": grip_slip_high,
            "grip_change_s": [2.0, 6.0], "shove": shove, "kill_prob": kill_prob,
            "kill_interval_s": [1.0, 4.0], "kill_legs": list(kill_legs) if kill_legs else None}
