"""The robot's-eye view of a frontal shot: the onboard RGB camera and the dToF, as the film would cut them.

The user's idea (10/26): during the frontal attack, cut to what the crab sees -- the RGB camera
(`onboard`, 1920 x 1440, 95.7 degrees) and the ToF depth map (`tof`, 54 x 42 zones) in false colour
-- switching between them as the bolt closes in. This renders one bolt flown straight at the sensor
in the plain scene of `tof_looming.py` (boxes standing in for alley walls), with the ToF made radial
and noised as there, and writes two previews: a cut (RGB, then ToF, then RGB at contact) and the
two side by side. Time runs 6.7 times slower than real (200 samples a second shown at 30 fps); the
ToF holds each of its 50 frames a second. The captions say what is simulated: the film's escape
reads the simulator's ball state, not these sensors.

    python experiments/vnc_v1/pov_frames.py --mjcf assets/jumper/jumper.xml --out film/previz/pov
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tof_looming as tl  # noqa: E402

RATE = 200.0  # samples a second
SHOW_FPS = 30
PW, PH = 640, 480  # one panel (4:3, as the onboard camera)


def false_colour(rng_m, cmap):
    """Radial ranges (m, -1 for no return) to RGB: near red, far blue, no return black."""
    v = np.clip(rng_m / 4.0, 0.0, 1.0)
    rgb = (cmap(1.0 - v)[..., :3] * 255).astype(np.uint8)
    rgb[rng_m < 0] = 0
    return rgb


def tof_panel(rgb_small):
    """The 54 x 42 zones as visible blocks, centred on a dark panel."""
    k = min(PW // tl.W, PH // tl.H)
    big = np.repeat(np.repeat(rgb_small, k, axis=0), k, axis=1)
    panel = np.full((PH, PW, 3), 12, np.uint8)
    y0, x0 = (PH - big.shape[0]) // 2, (PW - big.shape[1]) // 2
    panel[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
    return panel


def label(img, lines, font, corner="tl", colour=(235, 235, 235)):
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    y = 10 if corner[0] == "t" else img.height - 10 - 22 * len(lines)
    for line in lines:
        w = d.textlength(line, font=font)
        x = 12 if corner[1] == "l" else img.width - 12 - w
        d.rectangle([x - 4, y - 2, x + w + 4, y + 19], fill=(0, 0, 0))
        d.text((x, y), line, font=font, fill=colour)
        y += 22


CAPTION = "SIMULATION. The film's escape reads the simulator's ball state, not these sensors."


def captioned(img, font):
    """The sim-only caption in a bar under the picture (it must be in every frame of the film)."""
    from PIL import Image, ImageDraw

    out = Image.new("RGB", (img.width, img.height + 30), (0, 0, 0))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    text = CAPTION if d.textlength(CAPTION, font=font) < img.width - 20 else "SIMULATION: the escape reads simulator state."
    d.text(((img.width - d.textlength(text, font=font)) / 2, img.height + 6), text, font=font, fill=(235, 235, 235))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mjcf", default="assets/jumper/jumper.xml")
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--start", default="3.0,0.25,0.30", help="the bolt's start (m, robot frame)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--ffmpeg", default="", help="default: the one imageio-ffmpeg ships")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import matplotlib
    import mujoco
    from PIL import Image, ImageFont

    from tasks.jumper.common.constants import HOME

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = tl.build(args.mjcf)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
    rgb_r = mujoco.Renderer(model, height=PH, width=PW)
    tof_r = mujoco.Renderer(model, height=tl.H, width=tl.W)
    tof_r.enable_depth_rendering()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    scale = tl.radial_scale()
    cmap = matplotlib.colormaps["turbo"]
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    ball = model.body("ball").mocapid[0]
    boxes = [model.body(f"box{k}").mocapid[0] for k in range(6)]
    for k, m in enumerate(boxes):  # two rows of boxes for alley walls
        data.mocap_pos[m] = [1.4 + 1.3 * (k // 2), 1.0 if k % 2 == 0 else -1.0, 0.5]
    tof_body = model.body("tof_sensor_link").id
    tl.pose_robot(model, data, np.random.default_rng(0), HOME)
    mujoco.mj_forward(model, data)
    aim = data.xpos[tof_body].copy()
    start = np.array([float(v) for v in args.start.split(",")])
    vel = (aim - start) / np.linalg.norm(aim - start) * args.speed
    t_hit = np.linalg.norm(aim - start) / args.speed - 0.03  # the ball's surface reaches the sensor
    lead, tail = 0.2, 0.06
    n = int((lead + t_hit + tail) * RATE)

    side, cut = [], []
    tof_img = None
    for i in range(n):
        t = i / RATE - lead
        tl.pose_robot(model, data, rng, HOME)
        p_ball = start + vel * max(t, 0.0)
        data.mocap_pos[ball] = p_ball
        mujoco.mj_forward(model, data)
        rgb_r.update_scene(data, camera="onboard")
        rgb = rgb_r.render().copy()
        if tof_img is None or i % int(RATE / tl.FPS) == 0:  # the ToF runs at 50 Hz
            tof_r.update_scene(data, camera="tof")
            z = tof_r.render().astype(np.float32)
            r = z * scale + rng.normal(0, 1, z.shape).astype(np.float32) * (0.004 + 0.01 * z * scale)
            r[(rng.random(r.shape) < 0.01) | (r > 4.0)] = -1.0
            tof_img = tof_panel(false_colour(r, cmap))
        rel = p_ball - data.xpos[tof_body]
        dist = float(np.linalg.norm(rel))
        closing = -float(rel @ vel) / max(dist, 1e-3) if t >= 0 else 0.0
        ttc = (dist - 0.03) / closing if closing > 0 else None
        fire = closing > 0 and closing / dist > tl.FIRE
        tag = [f"t {max(t, 0.0):.3f} s (6.7x slow)", f"time to contact {ttc:.2f} s" if ttc is not None else "time to contact --"]
        if fire:
            tag.append("LOOMING (1/tau > 2.2): REFLEX FIRES")
        a = Image.fromarray(rgb)
        label(a, ["ROBOT POV  RGB  (onboard camera, simulated)"], font)
        label(a, tag, font, "bl", (255, 210, 120) if fire else (235, 235, 235))
        b = Image.fromarray(tof_img)
        label(b, ["ROBOT POV  ToF  54 x 42 zones (simulated)", "near red  far blue  no return black"], font)
        label(b, tag, font, "bl", (255, 210, 120) if fire else (235, 235, 235))
        both = Image.new("RGB", (2 * PW, PH))
        both.paste(a, (0, 0))
        both.paste(b, (PW, 0))
        side.append(captioned(both, font))
        use_tof = ttc is not None and 0.12 < ttc <= 0.35
        cut.append(captioned(b if use_tof else a, font))

    ff = args.ffmpeg
    if not ff:
        try:
            import imageio_ffmpeg

            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ff = "ffmpeg"
    for name, frames in (("pov_side", side), ("pov_cut", cut)):
        tmp = out / f"{name}_frames"
        tmp.mkdir(exist_ok=True)
        for k, im in enumerate(frames):
            im.save(tmp / f"{k:04d}.png")
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(SHOW_FPS), "-i", str(tmp / "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / f"{name}.mp4")], check=True)
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()
    picks = [int(n * q) for q in (0.3, 0.55, 0.75, 0.9)]
    h = side[0].height
    sheet = Image.new("RGB", (2 * PW, h * len(picks)))
    for k, i in enumerate(picks):
        sheet.paste(side[i], (0, k * h))
    sheet.save(out / "pov_sheet.png")
    print(f"POV_DONE {out} frames {n} hit at {t_hit:.3f} s")


if __name__ == "__main__":
    main()
