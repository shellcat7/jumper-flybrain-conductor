"""The conductor network's activity as a small inset (the user's idea, 10/28): every one of the
director's 24,154 nodes, drawn where its neuron sits in the MaleCNS nerve cord, lit by its own
recorded state -- the take's "cord" log, 50 Hz -- not animated by hand.

Each node's brightness is how far its state is above its own mean over the take, in its own
standard deviations; colours by superclass (descending amber, motor green, ascending violet,
intrinsic cyan, sensory pale blue). When the giant-fibre reflex fires, a red ring and a label;
below, the motor program the director has chosen and the take time. A dark translucent panel with
a hairline frame, so it reads as an instrument laid over the picture.

The network is the controller built on the fly's nerve-cord wiring, not a fly brain: the label
says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# MaleCNS superclasses as `node_positions.npz` names them (10/28: the old keys matched none, so every node was grey)
COLORS = {"descending_neuron": (255, 176, 60), "sensory_descending": (255, 176, 60), "vnc_motor": (90, 235, 120),
          "vnc_efferent": (90, 235, 120), "ascending_neuron": (190, 120, 255), "sensory_ascending": (190, 120, 255),
          "efferent_ascending": (190, 120, 255), "vnc_intrinsic": (60, 220, 255), "vnc_sensory": (150, 185, 255),
          "vnc_sensory_tbc": (150, 185, 255)}
NAMES = {"official": "六本脚 歩行", "five_foot": "五本脚 歩行（左前脚なし）", "five_foot_mirror": "五本脚 歩行（右前脚なし）",
         "sprint16-75399": "六本脚 疾走"}


#: (10/28 evening, the user: make the firing's strength readable) activity as a heat scale -- faint
#: blue for a node just above its usual level, cyan, yellow, white for the strongest -- over a dim,
#: neutral map of the cord, so how strongly a region fires reads at a glance
HEAT_STOPS = (0.0, 0.35, 0.7, 1.0)
HEAT = np.array([[0.10, 0.22, 0.60], [0.0, 0.72, 1.0], [1.0, 0.85, 0.15], [1.0, 1.0, 1.0]], np.float32)


def heat(a):
    """Activity in [0, 1] to the heat colours (n x 3)."""
    return np.stack([np.interp(a, HEAT_STOPS, HEAT[:, c]) for c in range(3)], axis=1).astype(np.float32)


def shot_steps(every: int, first: int, last: int, bullet: str = "") -> list:
    """The take's control step behind each output frame, as `import_take.py --movie` orders them
    (a bullet-time hold repeats its frame)."""
    holds = sorted((tuple(float(v) for v in s.split(",")) for s in bullet.split(";") if s), key=lambda x: x[0])
    frames, f = [], first
    for bf, hold, _deg in holds:
        frames += list(range(f, int(bf)))
        frames += [int(bf)] * int(hold)
        f = int(bf) + 1
    frames += list(range(f, last + 1))
    return [(fr - 1) * every for fr in frames]


def render(cord_npz: Path, positions: Path, steps: list, out: Path, size=(520, 430), font_path: str = "", text: float = 1.0,
           scene: Path | None = None, minimal: bool = False) -> None:
    """`minimal` (10/28, the user: explain the screen once, early, then show only the figure): the network
    alone, no lettering, with the giant fibre's red ring when it fires; the legend card says what it is."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    z = np.load(cord_npz)
    cord = z["cord"].astype(np.float32)  # samples x nodes, every 4 control steps
    # (10/28 night, the user: it kept twinkling, the strength hard to read) a quarter-second moving average
    # (13 samples at 50 Hz), so a region that fires stays lit and one that falls quiet fades
    k = 13
    cs = np.cumsum(np.pad(cord, ((k // 2, k // 2), (0, 0)), mode="edge"), axis=0)
    cord = ((cs[k - 1:] - np.r_[np.zeros((1, cord.shape[1]), np.float32), cs[:-k]]) / k).astype(np.float32)
    gate = z["gate"].astype(np.float32) if "gate" in z.files else None
    fired = z["gf_fired"] if "gf_fired" in z.files else None
    meta = json.loads(str(z["meta"]))
    # a five-legged program before the wound is the claw guard (`VNCModel.guard_override`), after it the lost leg
    wounded = None
    if scene is not None:
        sz = np.load(scene)
        wounded = sz["wounded"]
    names = meta.get("programs") or ["official"]
    p = np.load(positions, allow_pickle=True)
    pos = p["pos_um"].astype(np.float64)
    sup = p["superclass"]
    # project on the two main axes
    c = pos - pos.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T
    ks = size[1] / 430  # layout drawn for 430 px tall, scaled for larger frames
    tk = ks * text  # (10/28) a small panel keeps readable lettering: text scales on its own
    y_sub = 8 * ks + 21 * tk
    y_time = size[1] - 8 * ks - 17 * tk
    y_label = y_time - 24 * tk
    top, bottom = y_sub + 20 * tk, y_label - 6 * ks
    if minimal:
        top, bottom = 8 * ks, size[1] - 8 * ks
    pw, ph = size[0] - int(40 * ks), int(bottom - top)
    lo, hi = np.percentile(xy, 1.0, axis=0), np.percentile(xy, 99.5, axis=0)  # a few far nodes no longer shrink the rest
    span = hi - lo
    # (10/28) the long axis across the panel, so the cord fills its width and the lit nodes keep their colours
    scale = min(pw / span[0], ph / span[1])
    px = (xy[:, 0] - lo[0]) * scale + 20 * ks + (pw - span[0] * scale) / 2
    py = (xy[:, 1] - lo[1]) * scale + top + (ph - span[1] * scale) / 2
    ix, iy = np.clip(px.astype(int), 0, size[0] - 1), np.clip(py.astype(int), 0, size[1] - 1)
    col = np.array([COLORS.get(s, (200, 200, 200)) for s in sup], dtype=np.float32) / 255.0
    mean, std = cord.mean(0), cord.std(0) + 1e-3
    base = np.zeros((size[1], size[0], 3), np.float32)
    # (10/28) a large figure spreads the nodes over more pixels: brighter splats and wider dots keep it as full
    gain = max(1.0, (pw * ph) / 120000.0) ** 0.9
    # the anatomy, plainly visible but dim and neutral, so the heat colours alone carry the activity
    np.add.at(base, (iy, ix), (col * 0.35 + np.array([0.30, 0.40, 0.50], np.float32) * 0.65) * 0.06 * gain)
    base = np.minimum(base, 0.16)  # a dim silhouette: the heat colours must stand out against it
    f_big = ImageFont.truetype(font_path, int(17 * tk)) if font_path else ImageFont.load_default()
    f_small = ImageFont.truetype(font_path, int(13 * tk)) if font_path else ImageFont.load_default()
    out.mkdir(parents=True, exist_ok=True)
    for k, step in enumerate(steps):
        s = min(step // 4, cord.shape[0] - 1)
        # graded from 0.8 sigma (faint blue) to 2.4 sigma (white): each lit node a dot, larger and hotter the
        # stronger it fires, the weak drawn first so the strong stay on top (summing overlaps washed every
        # dense region to the same bright cyan), over a dim silhouette, with a soft glow
        a = np.clip(((cord[s] - mean) / std - 0.8) / 1.6, 0.0, 1.0)
        dots = Image.new("RGB", size, (0, 0, 0))
        dd = ImageDraw.Draw(dots)
        order = np.argsort(a)
        order = order[a[order] > 0.12]  # the faint flicker left out: what shows is clearly firing
        hc = (heat(a[order]) * (0.35 + 0.65 * a[order, None]) * 255).astype(int)
        rr = (0.6 + 2.6 * a[order] ** 1.5) * max(ks, 0.8)  # the strongest clearly the largest
        for i, c, r in zip(order, hc, rr):
            dd.ellipse((px[i] - r, py[i] - r, px[i] + r, py[i] + r), fill=(int(c[0]), int(c[1]), int(c[2])))
        sil = Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8))
        glow = dots.filter(ImageFilter.GaussianBlur(3.0 * ks))
        # the glow under the dots, not added on top: summing them washed dense regions to white
        layer = np.clip(np.asarray(sil, np.float32) + np.maximum(np.asarray(dots, np.float32),
                                                                 np.asarray(glow, np.float32) * 0.7), 0, 255).astype(np.uint8)
        panel = Image.new("RGBA", size, (8, 12, 16, 178))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=int(10 * ks), outline=(120, 220, 255, 140), width=max(1, int(ks)))
        rgb = Image.fromarray(layer).convert("RGBA")
        alpha = Image.fromarray(np.clip(layer.max(axis=2).astype(np.float32) * 1.5, 0, 255).astype(np.uint8))
        rgb.putalpha(alpha)
        panel = Image.alpha_composite(panel, rgb)
        d = ImageDraw.Draw(panel)
        if minimal:
            if fired is not None and bool(fired[min(step, len(fired) - 1)]):
                r = 14 * ks
                cx, cy = size[0] - r - 10 * ks, r + 10 * ks
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 70, 50, 255), width=max(2, int(3 * ks)))
            panel.save(out / f"brain_{k + 1:04d}.png")
            continue
        d.text((12 * ks, 8 * ks), "指揮者ネットワーク  24,154 ノード", font=f_big, fill=(225, 240, 248, 235))
        d.text((12 * ks, y_sub), "ショウジョウバエ神経索の配線にもとづく（脳の再現ではない）", font=f_small, fill=(160, 185, 200, 210))
        prog = "official"
        if gate is not None:
            g = int(np.argmax(gate[min(step, gate.shape[0] - 1)]))
            prog = names[g] if g < len(names) else "official"
        label = NAMES.get(prog, prog)
        if prog.startswith("five_foot") and wounded is not None and wounded.ndim == 1 and wounded[min(step, len(wounded) - 1)] < 0:
            label = "五本脚 歩行（" + ("右" if prog.endswith("mirror") else "左") + "爪で防御）"
        d.text((12 * ks, y_label), "» " + label, font=f_big, fill=(0, 225, 255, 240))
        d.text((12 * ks, y_time), f"t = {step * 0.005:5.2f} s   50 Hz 記録", font=f_small, fill=(160, 185, 200, 210))
        if fired is not None and bool(fired[min(step, len(fired) - 1)]):
            cx, cy, r = size[0] - 22 * tk, y_label + 10 * tk, 11 * tk
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 70, 50, 255), width=int(3 * ks))
            d.text((cx - r - 6 * ks, cy), "巨大線維 発火", font=f_big, fill=(255, 110, 80, 250), anchor="rm")
        panel.save(out / f"brain_{k + 1:04d}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cord", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--every", type=int, required=True)
    ap.add_argument("--range", required=True)
    ap.add_argument("--bullet-time", default="")
    ap.add_argument("--font", default="")
    ap.add_argument("--scene", default="", help="the slim take npz (wounded per step), for the guard label")
    ap.add_argument("--take", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    first, last = (int(v) for v in a.range.split(","))
    st = shot_steps(a.every, first, last, a.bullet_time)
    sc = None
    if a.scene:
        import tempfile

        w = np.load(a.scene)["wounded"][a.take]
        sc = Path(tempfile.mkdtemp()) / "w.npz"
        np.savez(sc, wounded=w)
    render(Path(a.cord), Path(a.positions), st, Path(a.out), font_path=a.font, scene=sc)
    print(f"BRAIN_INSET {a.out} {len(st)} frames")
