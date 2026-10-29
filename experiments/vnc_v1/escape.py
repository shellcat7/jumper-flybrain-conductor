"""The escape task: a ball thrown at the robot, a looming signal for the giant fiber, a hit penalty.

A fly escapes a looming object through the giant fiber: looming detectors drive DNp01, which
drives the jump motor neuron (TTMn) and the middle legs push off. Here a ball is thrown at
the robot every few seconds, a looming signal (the ball's direction in the body frame and its
expansion rate, 1 / time-to-contact) goes into its own observation group, which only the VNC
reads (`VNCModel` feeds it to the descending neurons, the giant fiber among them), and a hit
costs reward. The official MLP never sees the signal, so the official brain is the control:
same throws, no warning.

The ball is a 25 mm, 30 g sphere under gravity, thrown on a ballistic arc that reaches the
robot 0.6-1.2 s later from 0.8-1.2 m away, aimed 6-16 cm up (body to shell top, so a duck
can work as well as a sidestep). On 10/24 the official brain was hit by every throw at 0.4 and
0.6 s, and flyesc-s1, trained on 0.4-0.8 s throws, had learnt no dodge by iteration 200: a
sidestep wider than half the body takes a few tenths of a second to start. Between throws it rests 30 m away, beyond
anything an episode walks. A contact between the ball and any robot geom during a throw's
flight counts as one hit.
"""

from __future__ import annotations

import math

BALL_RADIUS = 0.025
PARK = (30.0, 0.0, BALL_RADIUS)

BALL_XML = f"""
<mujoco model="ball">
  <worldbody>
    <body name="ball">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" mass="0.03" rgba="0.95 0.85 0.1 1"
            friction="0.5 0.005 0.0001" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def ball_name(k: int = 0) -> str:
    """The k-th ball's scene entity; the first is "ball", as with one ball."""
    return "ball" if k == 0 else f"ball{k}"


def hit_sensor_name(k: int = 0) -> str:
    """The k-th ball's contact sensor."""
    return "ball_hit" if k == 0 else f"ball_hit{k}"


def block_sensor_name(k: int = 0, side: str = "LF") -> str:
    """The k-th ball's contact sensor on a claw (`side` "LF" or "RF": forearm, palm and fingers): a block."""
    return ("ball_block" if k == 0 else f"ball_block{k}") + ("" if side == "LF" else "_rf")


def _state(env, k: int = 0):
    """Per-robot throw state of the k-th ball: when its current flight ends, and whether it
    already hit."""
    import torch

    key = "_escape" if k == 0 else f"_escape{k}"
    st = getattr(env, key, None)
    if st is None:
        st = {"flight_end": torch.zeros(env.num_envs, device=env.device),
              "hit": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
              "throws": torch.zeros(env.num_envs, device=env.device),
              "hits": torch.zeros(env.num_envs, device=env.device)}
        setattr(env, key, st)
    return st


def _now(env):
    import torch

    return env.episode_length_buf.to(torch.float32) * env.step_dt


def park_pos(k: int = 0) -> tuple[float, float, float]:
    """Where the k-th ball rests between throws (1 m apart, so several never touch)."""
    return (PARK[0] + k, PARK[1], PARK[2])


def park_ball(env, env_ids, which: int = 0) -> None:
    """Reset event: ball number `which` rests far away and no throw is in flight."""
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ball = env.scene[ball_name(which)]
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids] + torch.tensor(park_pos(which), device=env.device)
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    st = _state(env, which)
    st["flight_end"][env_ids] = 0.0
    st["hit"][env_ids] = False


def throw_ball(env, env_ids, prob: float = 0.8, dist=(0.8, 1.2), flight=(0.6, 1.2),
               start_height=(0.06, 0.2), aim_height=(0.06, 0.16), lead: float = 1.0,
               bearing=None, which: int = 0, straight: bool = False) -> None:
    """Interval event: with probability `prob`, throw ball number `which` at the robot from a random
    bearing (body frame, radians; None draws all round), aimed `lead` x its current velocity ahead.
    `straight`: a powered bolt (the film's energy shot) -- a straight line at constant speed, which
    `hold_bolts` keeps up every step until it hits; otherwise a ballistic throw."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    st = _state(env, which)
    busy = st["flight_end"][env_ids] > _now(env)[env_ids]
    go = env_ids[(~busy) & (torch.rand(len(env_ids), device=env.device) < prob)]
    if len(go) == 0:
        return
    k = len(go)
    dev = env.device
    u = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(k, device=dev)  # noqa: E731
    robot = env.scene["robot"]
    base = robot.data.root_link_pos_w[go]
    vel = robot.data.root_link_lin_vel_w[go]
    quat = robot.data.root_link_quat_w[go]
    if bearing is None:
        theta = 2 * math.pi * torch.rand(k, device=dev)
    else:
        theta = torch.full((k,), float(bearing), device=dev)
    d = u(dist)
    offset_b = torch.stack([d * torch.cos(theta), d * torch.sin(theta), torch.zeros(k, device=dev)], dim=-1)
    offset_w = quat_apply(quat, offset_b)
    offset_w[:, 2] = 0.0
    start = base + offset_w
    start[:, 2] = u(start_height)
    t = u(flight)
    target = base + lead * vel * t[:, None]
    target[:, 2] = u(aim_height)
    g = torch.tensor([0.0, 0.0, 0.0 if straight else -9.81], device=dev)
    v0 = (target - start - 0.5 * g * (t ** 2)[:, None]) / t[:, None]
    pose = torch.zeros(k, 7, device=dev)
    pose[:, :3] = start
    pose[:, 3] = 1.0
    ball = env.scene[ball_name(which)]
    ball.write_root_link_pose_to_sim(pose, env_ids=go)
    twist = torch.zeros(k, 6, device=dev)
    twist[:, :3] = v0
    ball.write_root_link_velocity_to_sim(twist, env_ids=go)
    st["flight_end"][go] = _now(env)[go] + t + 0.3
    st["hit"][go] = False
    st["throws"][go] += 1
    if "bolt_v" not in st:
        st["bolt_v"] = torch.zeros(env.num_envs, 3, device=dev)
        st["bolt"] = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    st["bolt_v"][go] = v0
    st["bolt"][go] = straight


def hold_bolts(env, which: int = 0, stop=None) -> None:
    """Keep each bolt in flight on its line: rewrite its velocity to the launch velocity (gravity
    and drag undone), except where `stop` (bool, num_envs; e.g. it has hit) says to let it fall."""
    import torch

    st = _state(env, which)
    if "bolt" not in st:
        return
    live = st["bolt"] & (st["flight_end"] > _now(env))
    if stop is not None:
        live = live & ~stop
    ids = torch.nonzero(live).flatten()
    if len(ids) == 0:
        return
    twist = torch.zeros(len(ids), 6, device=env.device)
    twist[:, :3] = st["bolt_v"][ids]
    env.scene[ball_name(which)].write_root_link_velocity_to_sim(twist, env_ids=ids)


def roll_ball(env, env_ids, dist: float = 1.5, speed: float = 1.5, bearing: float = 0.0, lead: float = 1.0,
              which: int = 0) -> None:
    """Roll the ball along the ground at the robot from `dist` m on a body-frame `bearing`, aimed
    `lead` x its current velocity ahead: something low that a sidestep must clear by the robot's
    whole width and a jump clears by going over it."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    k = len(env_ids)
    dev = env.device
    robot = env.scene["robot"]
    base = robot.data.root_link_pos_w[env_ids]
    vel = robot.data.root_link_lin_vel_w[env_ids]
    quat = robot.data.root_link_quat_w[env_ids]
    theta = torch.full((k,), float(bearing), device=dev)
    offset_b = torch.stack([dist * torch.cos(theta), dist * torch.sin(theta), torch.zeros(k, device=dev)], dim=-1)
    offset_w = quat_apply(quat, offset_b)
    offset_w[:, 2] = 0.0
    start = base + offset_w
    start[:, 2] = BALL_RADIUS
    t = torch.full((k,), dist / speed, device=dev)
    target = base + lead * vel * t[:, None]
    target[:, 2] = BALL_RADIUS
    v = (target - start) / t[:, None]
    v[:, 2] = 0.0
    pose = torch.zeros(k, 7, device=dev)
    pose[:, :3] = start
    pose[:, 3] = 1.0
    ball = env.scene[ball_name(which)]
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    twist = torch.zeros(k, 6, device=dev)
    twist[:, :3] = v
    twist[:, 3:5] = torch.stack([-v[:, 1], v[:, 0]], dim=-1) / BALL_RADIUS  # rolling, not sliding
    ball.write_root_link_velocity_to_sim(twist, env_ids=env_ids)
    st = _state(env, which)
    st["flight_end"][env_ids] = _now(env)[env_ids] + t + 0.3
    st["hit"][env_ids] = False
    st["throws"][env_ids] += 1


def looming(env) -> "torch.Tensor":
    """Observation, [N, 5]: the ball's direction in the body frame (3), its expansion rate
    1/tau scaled by 1/10 (closing speed over distance; what a fly's looming detectors report),
    and exp(-distance / 0.3). Zero when no throw is in flight."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot = env.scene["robot"]
    ball = env.scene["ball"]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    dvel = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    dist = rel.norm(dim=-1).clamp_min(1e-3)
    closing = -(rel * dvel).sum(-1) / dist
    inv_tau = (closing / dist).clamp(0.0, 30.0)
    direction = quat_apply_inverse(robot.data.root_link_quat_w, rel / dist[:, None])
    live = (_state(env)["flight_end"] > _now(env)) & (closing > 0)
    out = torch.cat([direction, (inv_tau / 10.0)[:, None], torch.exp(-dist / 0.3)[:, None]], dim=-1)
    return torch.where(live[:, None], out, torch.zeros_like(out))


def looming_multi(env, balls: int) -> "torch.Tensor":
    """Observation, [N, 9 + 3 x balls], with several balls: `looming`'s five channels for the most
    urgent ball in flight (the largest 1/tau), that ball's number plus one (0 when none is in
    flight), the other balls in flight's body-frame horizontal directions summed with weights
    1/tau / 10 -- where the rest are coming from, and how soon -- and how many balls are closing
    in; then each ball's own body-frame horizontal direction and 1/tau / 10 (zero for one not
    closing in), as a fly's looming detectors report each object in its own part of the visual
    field. Zero when no throw is in flight."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot = env.scene["robot"]
    quat = robot.data.root_link_quat_w
    chans, rates, lives = [], [], []
    for k in range(balls):
        ball = env.scene[ball_name(k)]
        rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
        dvel = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
        dist = rel.norm(dim=-1).clamp_min(1e-3)
        closing = -(rel * dvel).sum(-1) / dist
        inv_tau = (closing / dist).clamp(0.0, 30.0)
        direction = quat_apply_inverse(quat, rel / dist[:, None])
        chans.append(torch.cat([direction, (inv_tau / 10.0)[:, None], torch.exp(-dist / 0.3)[:, None]], dim=-1))
        rates.append(inv_tau / 10.0)
        lives.append((_state(env, k)["flight_end"] > _now(env)) & (closing > 0))
    chans, rates, lives = torch.stack(chans, 1), torch.stack(rates, 1), torch.stack(lives, 1)  # N x B (x 5)
    urgency = torch.where(lives, rates, torch.full_like(rates, -1.0))
    best = urgency.argmax(dim=1)
    any_live = lives.any(dim=1)
    first = chans[torch.arange(len(best), device=best.device), best]
    others = torch.where(lives & (torch.arange(balls, device=best.device)[None, :] != best[:, None]), rates, 0.0)
    rest = (others[..., None] * chans[..., :2]).sum(dim=1)
    each = torch.where(lives[..., None], torch.cat([chans[..., :2], rates[..., None]], dim=-1), 0.0)
    out = torch.cat([first, (best + 1).to(first.dtype)[:, None], rest,
                     lives.sum(dim=1).to(first.dtype)[:, None], each.reshape(len(best), -1)], dim=-1)
    return torch.where(any_live[:, None], out, torch.zeros_like(out))


def ball_hit(env, sensor_name: str = "ball_hit") -> "torch.Tensor":
    """Reward term: 1 on the step a throw first touches the robot, else 0."""
    import torch

    st = _state(env)
    found = env.scene[sensor_name].data.found
    touching = (found.reshape(env.num_envs, -1) > 0).any(-1)
    in_flight = st["flight_end"] > _now(env)
    new = touching & in_flight & ~st["hit"]
    st["hit"] |= new
    st["hits"] += new.float()
    return new.to(torch.float32)


def add_escape(cfg, weight: float = -1000.0, interval=(2.5, 5.0), prob: float = 0.8, balls: int = 1,
               block: bool = False) -> dict:
    """Add the ball, the throws, the looming observation group and the hit penalty to a
    `jumper.posture` training config, in place. With `balls` > 1 the evaluation's extra balls
    (`ball1`, ...; each with its own hit sensor, parked 1 m apart) and the several-ball looming
    signal (`looming_multi`); the training throws and the penalty stay with the first ball."""
    import mujoco
    from mjlab.entity import EntityCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    for k in range(balls):
        cfg.scene.entities[ball_name(k)] = EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_string(BALL_XML),
            init_state=EntityCfg.InitialStateCfg(pos=park_pos(k)),
        )
        hit = ContactSensorCfg(
            name=hit_sensor_name(k),
            primary=ContactMatch(mode="geom", pattern="ball_geom", entity=ball_name(k)),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (hit,)
        if block:  # each claw's own sensor, to score a block apart from a hit
            for side in ("LF", "RF"):
                claw = ContactSensorCfg(
                    name=block_sensor_name(k, side),
                    primary=ContactMatch(mode="geom", pattern="ball_geom", entity=ball_name(k)),
                    secondary=ContactMatch(mode="subtree", pattern=f"{side}_forearm_link", entity="robot"),
                    fields=("found",),
                    reduce="none",
                    num_slots=1,
                )
                cfg.scene.sensors = cfg.scene.sensors + (claw,)
        cfg.events["ball_park" if k == 0 else f"ball_park{k}"] = EventTermCfg(
            func=park_ball, mode="reset", params={"which": k})
    cfg.events["ball_throw"] = EventTermCfg(func=throw_ball, mode="interval", interval_range_s=interval,
                                            params={"prob": prob})
    term = (ObservationTermCfg(func=looming) if balls == 1
            else ObservationTermCfg(func=looming_multi, params={"balls": balls}))
    cfg.observations["threat"] = ObservationGroupCfg(
        terms={"looming": term}, concatenate_terms=True, enable_corruption=False)
    cfg.rewards["ball_hit"] = RewardTermCfg(func=ball_hit, weight=weight)
    return {"ball_radius": BALL_RADIUS, "interval_s": list(interval), "prob": prob, "hit_weight": weight,
            "balls": balls}
