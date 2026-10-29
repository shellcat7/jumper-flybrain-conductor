"""Looming from the robot's own dToF: can the giant fibre's input come from the sensor, not the simulator?

The escape reads the simulator's true ball state (all round, no delay). The real Jumper has a
forward dToF (VL53L9CX: 54 x 42 zones over 55 x 42 degrees, 50 frames a second here) on
`tof_sensor_link` -- the model's `tof` camera -- and an RGB camera. This trains a small network to
read, from the last three ToF frames alone, what the looming signal says: whether a ball is closing
in (threat), how soon (1/tau) and from where (azimuth, elevation in the sensor's frame).

Data: plain MuJoCo, the robot standing (jittered as a walk would jitter the sensor: a few mm, a
degree), a 25 mm ball flown in straight lines from 3 m at 5-8 m/s from inside the field of view,
aimed at or near the robot, random boxes behind as clutter, and the depth rendered from the `tof`
camera, made radial, noised (a few mm plus 1% of range, 1% dropouts, no return past 4 m). Half the
flights pass wide (no threat). Scored on held-out flights: when the detector's 1/tau first passes
2.2 / s (0.45 s to contact) against when the truth does, and false alarms on wide passes; beside it
an analytic baseline (the fastest-closing zone's range rate).

    MUJOCO_GL=egl python experiments/vnc_v1/tof_looming.py --mjcf assets/jumper/jumper.xml --out results/tof
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

H, W = 42, 54
FOVX, FOVY = 54.97, 42.0
FPS = 50.0
FIRE = 2.2  # 1/tau at which the reflex should fire: 0.45 s to contact


def build(mjcf: str, n_boxes: int = 6):
    import mujoco

    spec = mujoco.MjSpec.from_file(mjcf)
    world = spec.worldbody
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.3, 0.3, 0.3, 1])
    b = world.add_body(name="ball", mocap=True)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[1, 0.2, 0.1, 1], contype=0, conaffinity=0)
    for k in range(n_boxes):
        c = world.add_body(name=f"box{k}", mocap=True)
        c.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.5], rgba=[0.5, 0.5, 0.5, 1], contype=0, conaffinity=0)
    model = spec.compile()
    return model


def pose_robot(model, data, rng, home):
    """The robot standing at the origin facing +x, jittered as a walk jitters the sensor."""
    import mujoco

    names = [model.joint(i).name for i in range(model.njnt)]
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
    a = model.jnt_qposadr[free]
    data.qpos[:] = 0
    for n, v in home.items():
        if n in names:
            data.qpos[model.jnt_qposadr[names.index(n)]] = v
    data.qpos[a:a + 3] = [0.0, 0.0, 0.107 + rng.normal(0, 0.004)]
    roll, pitch, yaw = rng.normal(0, math.radians(1.0), 3)
    cy, sy, cp, sp, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(roll / 2), math.sin(roll / 2)
    data.qpos[a + 3:a + 7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                              cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def radial_scale():
    """Per-zone factor from the depth buffer's z-distance to the range along the zone's ray."""
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = (np.arange(H) + 0.5) / H * 2 - 1
    tx, ty = math.tan(math.radians(FOVX / 2)), math.tan(math.radians(FOVY / 2))
    gx, gy = np.meshgrid(xs * tx, ys * ty)
    return np.sqrt(1 + gx ** 2 + gy ** 2).astype(np.float32)


def flight(rng, threat: bool):
    """A straight flight from 3 m inside the field of view: start, velocity, duration."""
    az = math.radians(rng.uniform(-24, 24))
    d0 = rng.uniform(2.5, 3.5)
    start = np.array([d0 * math.cos(az), d0 * math.sin(az), rng.uniform(0.12, 0.5)])
    if threat:
        aim = np.array([0.05 + rng.uniform(-0.05, 0.1), rng.uniform(-0.12, 0.12), rng.uniform(0.05, 0.14)])
    else:  # wide: passes 0.45-1.2 m to one side
        side = rng.choice([-1.0, 1.0])
        aim = np.array([rng.uniform(-0.3, 0.3), side * rng.uniform(0.45, 1.2), rng.uniform(0.05, 0.3)])
    speed = rng.uniform(5.0, 8.0)
    u = (aim - start) / np.linalg.norm(aim - start)
    return start, u * speed, np.linalg.norm(aim - start) / speed


def generate(model, n_flights, rng, home, scale, renderer, tof_body):
    """Frames [n, 3, H, W] (three consecutive radial depth maps, the last the current one) and
    labels [n, 5]: threat, 1/tau / 10, sin az, sin el, in-view; plus each flight's frame count."""
    import mujoco

    data = mujoco.MjData(model)
    ball_mocap = model.body("ball").mocapid[0]
    box_ids = [model.body(f"box{k}").mocapid[0] for k in range(6)]
    xs, ys, lens = [], [], []
    for i in range(n_flights):
        threat = i % 2 == 0
        start, vel, t_end = flight(rng, threat)
        for k, m in enumerate(box_ids):
            r, a = rng.uniform(1.5, 6.0), math.radians(rng.uniform(-40, 40))
            data.mocap_pos[m] = [r * math.cos(a), r * math.sin(a), 0.5 if rng.random() < 0.7 else 30.0]
        n = int(t_end * FPS) + 6
        frames = []
        for f in range(n):
            t = f / FPS
            pose_robot(model, data, rng, home)
            p = start + vel * t
            data.mocap_pos[ball_mocap] = p
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="tof")
            z = renderer.render().astype(np.float32)
            rng_m = z * scale
            rng_m = rng_m + rng.normal(0, 1, rng_m.shape).astype(np.float32) * (0.004 + 0.01 * rng_m)
            rng_m[(rng.random(rng_m.shape) < 0.01) | (rng_m > 4.0)] = -1.0
            frames.append(rng_m)
            # labels in the sensor's frame
            cam = data.xpos[tof_body]
            rel = p - cam
            dist = float(np.linalg.norm(rel))
            closing = -float(rel @ vel) / max(dist, 1e-3)
            inv_tau = max(closing, 0.0) / max(dist, 1e-3)
            R = data.xmat[tof_body].reshape(3, 3)
            rb = R.T @ rel / max(dist, 1e-3)  # body axes: x ahead, y left, z up
            az = math.atan2(rb[1], rb[0])
            el = math.atan2(rb[2], math.hypot(rb[0], rb[1]))
            in_view = abs(math.degrees(az)) < FOVX / 2 and abs(math.degrees(el)) < FOVY / 2 and closing > 0
            if f >= 2:
                xs.append(np.stack(frames[-3:]))
                ys.append([1.0 if (threat and closing > 0) else 0.0, min(inv_tau, 30.0) / 10.0,
                           math.sin(az), math.sin(el), 1.0 if in_view else 0.0])
        lens.append(n - 2)
    return np.stack(xs).astype(np.float16), np.array(ys, np.float32), lens


def net():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * 6 * 7, 128), nn.ReLU(), nn.Linear(128, 4))


def prep(x):
    """Radial ranges to inverse range (near is large), no return as 0."""
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(x > 0, 1.0 / x.clamp_min(0.05), torch.zeros_like(x)) / 5.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mjcf", default="assets/jumper/jumper.xml")
    p.add_argument("--train-flights", type=int, default=3000)
    p.add_argument("--test-flights", type=int, default=400)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import mujoco
    import torch
    import torch.nn.functional as F

    from tasks.jumper.common.constants import HOME

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build(args.mjcf)
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()
    tof_body = model.body("tof_sensor_link").id
    scale = radial_scale()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    xtr, ytr, _ = generate(model, args.train_flights, rng, HOME, scale, renderer, tof_body)
    xte, yte, lens = generate(model, args.test_flights, np.random.default_rng(args.seed + 1), HOME, scale, renderer, tof_body)
    print(f"[tof] data: train {len(xtr)} frames, test {len(xte)}, {time.time() - t0:.0f} s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = net().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xt, yt = prep(xtr), torch.as_tensor(ytr)
    for ep in range(args.epochs):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 512):
            idx = perm[i:i + 512]
            xb, yb = xt[idx].to(dev), yt[idx].to(dev)
            o = m(xb)
            w = yb[:, 4:5]  # bearing and 1/tau only where the ball is in view
            loss = (F.binary_cross_entropy_with_logits(o[:, 0], yb[:, 0])
                    + (w * (o[:, 1:4] - yb[:, 1:4]) ** 2).mean() * 10)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"[tof] epoch {ep} loss {tot / len(xt):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        o = torch.cat([m(prep(xte[i:i + 1024]).to(dev)).cpu() for i in range(0, len(xte), 1024)])
    prob, inv_tau = torch.sigmoid(o[:, 0]).numpy(), o[:, 1].numpy() * 10
    # per flight: when the detector fires (p > 0.5 and 1/tau > FIRE) against when the truth does
    lat, alarms, misses, k = [], 0, 0, 0
    base_lat, base_alarms = [], 0
    for fi, n in enumerate(lens):
        y = yte[k:k + n]
        pr, it = prob[k:k + n], inv_tau[k:k + n]
        x = xte[k:k + n].astype(np.float32)
        threat = fi % 2 == 0
        fire = np.nonzero((pr > 0.5) & (it > FIRE))[0]
        truth = np.nonzero(y[:, 1] * 10 > FIRE)[0] if threat else []
        # baseline: the fastest-closing zone (valid in both frames, under 3.5 m), its range rate
        both = (x[:, 1] > 0) & (x[:, 2] > 0) & (x[:, 2] < 3.5)
        rate = np.where(both, (x[:, 1] - x[:, 2]) * FPS / np.maximum(x[:, 2], 0.05), 0.0)
        b_inv = np.clip(rate.reshape(len(x), -1).max(axis=1), 0, 30)
        b_fire = np.nonzero(b_inv > FIRE)[0]
        if threat:
            if len(truth):
                if len(fire):
                    lat.append((fire[0] - truth[0]) / FPS)
                else:
                    misses += 1
                if len(b_fire):
                    base_lat.append((b_fire[0] - truth[0]) / FPS)
        else:
            alarms += int(len(fire) > 0)
            base_alarms += int(len(b_fire) > 0)
        k += n
    in_view = yte[:, 4] > 0.5
    res = {
        "frames_train": int(len(xtr)), "frames_test": int(len(xte)),
        "threat_flights": int(sum(1 for i in range(len(lens)) if i % 2 == 0)),
        "detected": len(lat), "undetected": misses,
        "fire_latency_s_median": float(np.median(lat)) if lat else None,
        "fire_latency_s_p90": float(np.percentile(lat, 90)) if lat else None,
        "false_alarms_on_wide_passes": alarms / max(1, len(lens) // 2),
        "inv_tau_mae_in_view": float(np.mean(np.abs(inv_tau[in_view] - yte[in_view, 1] * 10))),
        "azimuth_err_deg_in_view": float(np.degrees(np.mean(np.abs(np.arcsin(np.clip(o[in_view, 2].numpy(), -1, 1))
                                                                    - np.arcsin(yte[in_view, 2]))))),
        "baseline_detected": len(base_lat), "baseline_fire_latency_s_median": float(np.median(base_lat)) if base_lat else None,
        "baseline_false_alarms": base_alarms / max(1, len(lens) // 2),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    torch.save(m.state_dict(), out / "tof_looming_net.pt")
    print(json.dumps(res))
    print("TOF_LOOMING_DONE")


if __name__ == "__main__":
    main()
