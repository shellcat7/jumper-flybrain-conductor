"""Cut the round-1 short (10/28): about 30 s from the alley take and the PLATEAU city.

The order: high over Shinjuku at night; the robot walking in the back alley and stepping from a
side bolt; a bolt parried off the claw at 1/8 speed; three at once, crouched under; the jump over the
wave of four, bullet time at its apex, the bolt that catches the left claw, the landing on five legs;
the run; the shutter it gets under and the drone that does not; black, and a card with what is
simulated and what is not, and the credits.

Two stages, so the light work runs on the PC and only the encode needs ffmpeg (Colab):

    prep: per shot, the conductor inset (`brain_inset.py`, the take's recorded cord activity), the HUD
          (six leg lamps from the feet's heights, the playback speed) and the sound (`sound.py`), the
          music under the whole cut, the end card. All on the shot's own output frames.
    cut:  each shot graded (bloom, fringing, grade, grain) with its inset and HUD laid over, the shots
          joined, letterboxed to 2.39:1 with the standing caption in the lower bar (the sim-only
          caveat, `film/previz/caption_*.txt`, and the MaleCNS credit), faded, the end card, the
          sound; H.264 at 2560x1440 and 1920x1080.

    python edit_short.py prep --scene alley_parry_slim.npz --cord alley_parry_t21_cord.npz \
        --positions node_positions.npz --font NotoSansJP-VF.ttf --out <post dir>
    python edit_short.py cut --renders <film-2026-10-28> --post <post dir> --out short.mp4

The edit's on-screen text is factual only (no title, the user's call 10/26).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brain_inset import render as render_inset, shot_steps  # noqa: E402

FPS = 25
W, H = 2560, 1440
BAR = 184  # 2.39:1 inside 16:9
TAKE = 21
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")

#: the shots in order. `use`: the shot's output frames kept (1-based, inclusive); `offset`: where
#: the camera sits from the robot (for the sound's panning); the city has its own frame numbers.
SHUTTER_T = 11.75  # when the robot's base passes under the shutter (set from the re-recorded take)
SHOTS = [
    # (10/28 afternoon, the user: slower and smooth, the whole city first) city.py --aerial 300
    {"name": "aerial", "kind": "city", "files": (1, 300)},
    # (10/28, the user: explain the HUD once, early; afterwards the brain figure alone, small, no text)
    {"name": "legend", "kind": "legend", "still": ("establish", 10), "step": 72, "gf_step": 301, "dur": 5.0},
    # (10/28 afternoon, the user: the enemy never seen, the layout unreadable) every shot frames the drones
    {"name": "establish", "every": 8, "range": (1, 78), "use": (10, 78), "offset": (-3.2, -0.35, 1.55)},
    # (10/28 evening, the user: every action shot played as slow motion, heavy and hard to follow) speed
    # ramps over the rendered frames: `ramp` (first, stop, stride) in rendered frame numbers -- real time
    # (a stride of 8 / every), slowed only round the moment that matters
    {"name": "parry", "every": 1, "range": (573, 681), "use": (1, 109), "offset": (-0.95, 0.42, 0.24),
     "ramp": [(1, 49, 8), (49, 65, 2), (65, 77, 1), (77, 89, 2), (89, 110, 8)]},  # the claw meets it at frame 69
    {"name": "volley", "every": 2, "range": (411, 491), "use": (1, 81), "offset": (-1.2, -0.7, 3.1),
     "ramp": [(1, 37, 4), (37, 65, 2), (65, 82, 4)]},  # the three pass over it at 4.5-4.7 s
    {"name": "wound", "every": 4, "range": (345, 390), "bullet": "356,50,-100", "use": (1, 95),
     "offset": (-1.25, 0.9, 0.42), "ramp": [(1, 12, 2), (12, 62, 2), (62, 78, 1), (78, 96, 2)]},  # the hold in 1 s
    # (10/28 evening, the user: the straight run left unclear why it gets away; weave round bins and
    # dumpsters; duck low under the shutter) the re-recorded take ("alt": --scene2/--cord2, the same first
    # 7.4 s): the weave seen from ahead and above, then the duck under the shutter
    # (10/28 night, the user: the escape still felt slow -- it runs at 0.6 m/s, weaving) played at x2, shown so
    {"name": "runweave", "every": 8, "range": (185, 285), "use": (1, 101), "offset": (2.3, 0.25, 1.75), "alt": True,
     "ramp": [(1, 102, 2)]},
    {"name": "shutterweave", "every": 4, "range": (569, 649), "use": (1, 81), "offset": (0.9, 0.3, 0.075),
     "shutter": SHUTTER_T, "tail": 0.6, "alt": True,
     "ramp": [(1, 17, 4), (17, 41, 2), (41, 53, 1), (53, 82, 2)]},  # in at x2, under it at x1, the crash at x0.5
    # (10/28, the user: from the ground up to the sky too) the aerial's second half played backwards: up
    # from above the street, the gaze rising to the towers and the whole night city -- then the card
    {"name": "rise", "kind": "rise", "files": (300, 200)},
]
END_CARD_S = 5.0
SHUTTER_SHOT = next(x for x in SHOTS if x.get("shutter"))
SHUTTER_TAIL = SHUTTER_SHOT["tail"]  # (the rise comes after it now)
INSET = (470, 292)  # the figure-only inset, top right (10/28 afternoon, the user: a little bigger)
INSET_XY = (W - INSET[0] - 32, BAR + 22)
HUD_XY = (56, H - BAR - 150)  # (the user: the leg lamps much bigger, they were hard to see)
END_JA = ["すべて物理シミュレーション（MuJoCo）の記録です。ロボットの動きは手付けではありません。",
          "指揮役のネットワークはショウジョウバエ神経索の配線（MaleCNS）にもとづきます。脳の再現ではありません。",
          "弾の位置はシミュレーターから全方向・遅延なしで取得しています（実機は前方カメラのみ・要追加センサー）。",
          "弾は直径25 mmの球。蛇行は走行の指令、ゴミ箱は後から置いた美術、ドローンの飛行・シャッター・墜落は手付けです。"]
END_EN = ["Recorded from a physics simulation (MuJoCo); the robot's motion is not hand-animated.",
          "The conductor network follows the fruit fly's nerve-cord wiring (MaleCNS). It is not a fly brain.",
          "Shots are sensed from the simulator's true state, all round and without delay; the real robot would need added sensors.",
          "The 'bolts' are 25 mm balls. The weave is a commanded path, the bins are set dressing; the drones, shutter and crash are keyframed."]
CREDITS = ["Jumper robot & toolkit — KingKong Robotics (Apache-2.0)",
           "Nerve-cord wiring: MaleCNS (CC BY 4.0), modified",
           "3D都市モデル：Project PLATEAU（国土交通省）を加工して作成",
           "Props, textures & HDRI: Poly Haven (CC0)"]


def font(path: str, size: int, weight: int = 500):
    """Noto Sans JP (variable) at a given weight; its default instance is Thin."""
    from PIL import ImageFont

    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_axes([weight])
    except (OSError, AttributeError):
        pass
    return f


def picked(shot):
    """The rendered frame numbers (1-based) the cut plays, in order: `use`, or the speed `ramp`."""
    if shot.get("ramp"):
        return [f for a, b, k in shot["ramp"] for f in range(a, b, k)]
    a, b = shot["use"]
    return list(range(a, b + 1))


def shot_frames(shot):
    """The take step behind each played frame."""
    steps = shot_steps(shot["every"], *shot["range"], shot.get("bullet", ""))
    return [steps[f - 1] for f in picked(shot)]


def hud(scene_npz, steps, rates, out: Path, font: str):
    """Transparent full-frame PNGs: six leg lamps lit while each foot is down, the playback speed."""
    from PIL import Image, ImageDraw, ImageFont

    z = np.load(scene_npz)
    meta = json.loads(str(z["meta"]))
    bodies = meta["body_names"]
    pos = z["body_pos"][TAKE]
    tips = np.stack([pos[:, [i for i, b in enumerate(bodies) if b.startswith(leg + "_")], 2].min(axis=1)
                     for leg in LEGS], axis=1)
    ground = np.percentile(tips, 2, axis=0)
    # (10/28, the user: the HUD was too big and covered too much -- then, in the afternoon: the leg lamps
    # much bigger, they could not be read) one row of six lamps with labels and the speed, on a dark plate
    f_mid = globals()["font"](font, 44, 700)
    f_small = globals()["font"](font, 26, 700)
    out.mkdir(parents=True, exist_ok=True)
    x0, y0 = HUD_XY
    for k, (s, r) in enumerate(zip(steps, rates)):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x0 - 18, y0 - 18, x0 + 6 * 66 + 150, y0 + 104), radius=14, fill=(2, 6, 10, 120))
        for i, leg in enumerate(LEGS):
            on = tips[min(s, len(tips) - 1), i] - ground[i] < 0.012
            cx, cy, rad = x0 + 66 * i + 22, y0 + 24, 17
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(0, 225, 255, 235) if on else (70, 84, 96, 170))
            d.text((cx, cy + 48), leg, font=f_small, fill=(215, 225, 232, 220), anchor="mm")
        d.text((x0 + 6 * 66 + 18, y0 + 26), "x0" if r == 0 else f"x{r:g}", font=f_mid, fill=(235, 242, 246, 235),
               anchor="lm")
        img.save(out / f"hud_{k + 1:04d}.png")


def legend(args, shot, out: Path):
    """The screen explained once, on a freeze (10/28, the user: the pause is for explaining, so
    explain clearly -- big figure, readable colours -- rather than keep the picture). Frames
    `legend/legend_0001.png...`: the picture dimmed, the network figure large with what it is and
    what it is not, the leg lamps drawn large, the speed; in the last 0.6 s the labels fade and the
    figure shrinks into the corner where it stays for the rest of the film. English."""
    from PIL import Image, ImageDraw

    n = int(round(shot["dur"] * FPS))
    big_xy, big_wh = (1200, 250), (1280, 640)
    tmp = out / "_legend_inset"
    render_inset(Path(args.cord), Path(args.positions), [shot["gf_step"]], tmp, size=big_wh, font_path=args.font,
                 minimal=True)
    fig = Image.open(tmp / "brain_0001.png")
    f_t, f_h, f_b, f_s = (font(args.font, 64, 700), font(args.font, 46, 700), font(args.font, 34, 500),
                          font(args.font, 30, 500))
    cyan, white, soft, red = (40, 230, 255, 255), (255, 255, 255, 255), (205, 218, 228, 255), (255, 95, 75, 255)
    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(text)
    d.text((110, BAR + 70), "HOW TO READ THE SCREEN", font=f_t, fill=white)
    d.line((112, BAR + 152, 700, BAR + 152), fill=cyan, width=4)
    y = BAR + 205
    d.text((110, y), "CONDUCTOR NETWORK", font=f_h, fill=cyan)
    for i, line in enumerate(("24,154 nodes, drawn where their neurons sit",
                              "in the fruit fly's nerve cord.",
                              "Lit dots: nodes active right now,",
                              "replayed from the simulation's log.")):
        d.text((110, y + 70 + i * 48), line, font=f_b, fill=white)
    d.text((110, y + 70 + 4 * 48 + 14), "Built on the fly's wiring -- not a fly brain.", font=f_s, fill=soft)
    # (10/28 evening) the heat scale the dots are drawn in: how strongly each node fires
    from brain_inset import heat
    bx, by, bw = 110, y + 70 + 4 * 48 + 72, 420
    for i in range(bw):
        c = heat(np.array([i / (bw - 1)]))[0] * 255
        d.line((bx + i, by, bx + i, by + 18), fill=(int(c[0]), int(c[1]), int(c[2]), 255))
    d.text((bx, by + 26), "weak", font=f_s, fill=soft)
    d.text((bx + bw, by + 26), "strong", font=f_s, fill=soft, anchor="ra")
    d.text((bx + bw + 24, by - 6), "FIRING", font=f_s, fill=white)
    d.line((900, y + 120, big_xy[0] - 10, y + 120), fill=cyan, width=4)
    # the red ring, top right of the figure
    rx, ry = big_xy[0] + big_wh[0] - 34, big_xy[1] + 34
    d.line((rx, ry + 40, rx, big_xy[1] + big_wh[1] + 26), fill=red, width=4)
    d.text((rx, big_xy[1] + big_wh[1] + 34), "RED RING: the giant fibre fires -- the escape reflex", font=f_b, fill=white,
           anchor="ra")
    # the leg lamps, large
    lx, ly = 140, H - BAR - 250
    d.text((110, ly - 76), "LEG LAMPS", font=f_h, fill=cyan)
    for i, (leg, on) in enumerate(zip(LEGS, (True, False, True, True, False, True))):
        cx = lx + i * 92
        d.ellipse((cx - 22, ly - 22, cx + 22, ly + 22), fill=(0, 225, 255, 255) if on else (70, 84, 96, 255))
        d.text((cx, ly + 46), leg, font=f_s, fill=white, anchor="mm")
    d.text((lx + 6 * 92, ly - 20), "lit while that foot is on the ground", font=f_b, fill=white)
    d.text((110, ly + 100), "x1", font=f_h, fill=cyan)
    d.text((200, ly + 108), "PLAYBACK SPEED   (x1 real time · x0.125 eight times slower)", font=f_b, fill=white)
    out_dir = out / "legend"
    out_dir.mkdir(parents=True, exist_ok=True)
    shrink = int(0.6 * FPS)
    for k in range(n):
        u = max(0.0, (k - (n - shrink)) / shrink)  # 0 until the last 0.6 s, then to 1
        e = u * u * (3 - 2 * u)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(img)
        dd.rectangle((0, BAR, W, H - BAR), fill=(2, 6, 10, int(185 * (1 - e))))
        x = big_xy[0] + (INSET_XY[0] - big_xy[0]) * e
        yy = big_xy[1] + (INSET_XY[1] - big_xy[1]) * e
        w = big_wh[0] + (INSET[0] - big_wh[0]) * e
        h = big_wh[1] + (INSET[1] - big_wh[1]) * e
        dd.rounded_rectangle((int(x), int(yy), int(x + w), int(yy + h)), radius=12, fill=(4, 8, 12, int(215 * (1 - e))))
        f = fig.resize((max(2, int(w)), max(2, int(h))), Image.LANCZOS)
        img.alpha_composite(f, (int(x), int(yy)))
        if e < 1:
            t = text.copy()
            t.putalpha(t.getchannel("A").point(lambda v: int(v * (1 - e))))
            img.alpha_composite(t)
        img.save(out_dir / f"legend_{k + 1:04d}.png")


def bars(out: Path, font_path: str):
    """The 2.39:1 bars with the standing caption in the lower one: the sim-only caveat and the credit."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, BAR), fill=(0, 0, 0, 255))
    d.rectangle((0, H - BAR, W, H), fill=(0, 0, 0, 255))
    ja = (HERE.parent / "previz" / "caption_ja.txt").read_text(encoding="utf-8").strip()
    en = (HERE.parent / "previz" / "caption_en.txt").read_text(encoding="utf-8").strip()
    cr = (HERE.parent / "previz" / "credit.txt").read_text(encoding="utf-8").strip()
    for y, txt, f, c in ((H - BAR + 46, ja, font(font_path, 28, 500), (235, 240, 244, 225)),
                         (H - BAR + 92, en, font(font_path, 23, 400), (190, 200, 208, 200)),
                         (H - BAR + 134, cr, font(font_path, 20, 400), (150, 160, 168, 190))):
        d.text((W // 2, y), txt, font=f, fill=c, anchor="mm")
    img.save(out / "bars.png")


def end_card(out: Path, font_path: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    f_ja, f_en, f_cr = font(font_path, 34, 500), font(font_path, 25, 400), font(font_path, 24, 400)
    y = 330
    for ja, en in zip(END_JA, END_EN):
        d.text((W // 2, y), ja, font=f_ja, fill=(235, 240, 244), anchor="mm")
        d.text((W // 2, y + 46), en, font=f_en, fill=(150, 162, 172), anchor="mm")
        y += 128
    y += 40
    for line in CREDITS:
        d.text((W // 2, y), line, font=f_cr, fill=(120, 132, 142), anchor="mm")
        y += 40
    img.save(out)


def prep(args):
    import sound

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.scene)
    z2 = np.load(args.scene2) if args.scene2 else z
    timeline, t = [], 0.0
    parts = []
    rng = np.random.default_rng(3)
    for shot in SHOTS:
        if shot.get("kind") == "city":
            n = shot["files"][1] - shot["files"][0] + 1
            wav = out / f"sound_{shot['name']}.wav"
            sound.write_wav(wav, sound.aerial_calm(n / FPS, rng) * 0.9)
            timeline.append({"name": shot["name"], "start": t, "frames": n})
            parts.append(str(wav))
            t += n / FPS
            continue
        if shot.get("kind") == "rise":
            n = shot["files"][0] - shot["files"][1] + 1
            fl = sound.aerial_calm(SHOTS[0]["files"][1] / FPS, rng)
            k0 = int((shot["files"][1] - 1) / FPS * sound.SR)
            k1 = int(shot["files"][0] / FPS * sound.SR)
            audio = np.ascontiguousarray(fl[:, k0:k1][:, ::-1])
            m = int(1.2 * sound.SR)
            clank = np.zeros(m)
            land = sound.landing(rng)[:m]
            clank[: len(land)] += land * 0.8
            audio[:, int(0.1 * sound.SR): int(0.1 * sound.SR) + m] += clank * 0.7
            audio[:, : int(0.5 * sound.SR)] *= np.linspace(0.3, 1, int(0.5 * sound.SR))
            sound.write_wav(out / "sound_rise.wav", audio * 0.9)
            parts.append(str(out / "sound_rise.wav"))
            timeline.append({"name": "rise", "start": t, "frames": n})
            t += n / FPS
            continue
        if shot.get("kind") == "legend":
            legend(args, shot, out)
            dur = shot["dur"]
            amb = sound.ambience(int(dur * sound.SR), rng) * 0.8
            chime = sound.neuron_blip(rng) * 1.6
            amb[:, int(0.2 * sound.SR): int(0.2 * sound.SR) + len(chime)] += chime
            sound.write_wav(out / "sound_legend.wav", amb * 0.9)
            parts.append(str(out / "sound_legend.wav"))
            timeline.append({"name": "legend", "start": t, "frames": int(round(dur * FPS))})
            t += dur
            continue
        steps = shot_frames(shot)
        rates = [round((steps[i + 1] - steps[i]) / (200 / FPS), 4) for i in range(len(steps) - 1)]
        rates.append(rates[-1] if rates else round(shot["every"] / 8, 4))
        name = shot["name"]
        alt = bool(shot.get("alt")) and bool(args.scene2)
        zs, scene_path, cord_path = (z2, args.scene2, args.cord2) if alt else (z, args.scene, args.cord)
        if len(list((out / f"inset_{name}").glob("brain_*.png"))) != len(steps):  # (a rerun keeps finished frames)
            render_inset(Path(cord_path), Path(args.positions), steps, out / f"inset_{name}", size=INSET,
                         font_path=args.font, scene=_wound_npz(zs, out, "_alt" if alt else ""), minimal=True)
        if len(list((out / f"hud_{name}").glob("hud_*.png"))) != len(steps):
            hud(scene_path, steps, rates, out / f"hud_{name}", args.font)
        audio = sound.shot(zs, TAKE, shot["every"], *shot["range"], shot.get("bullet", ""), shot["offset"],
                           shutter_T=shot.get("shutter"), steps=steps)
        if shot.get("tail"):  # the held last frame going to black: the sound fades under it
            k = int(0.4 * sound.SR)
            audio[:, -k:] *= np.linspace(1, 0.25, k)
            pad = np.zeros((2, int(shot["tail"] * sound.SR)))
            pad[:, : int(0.6 * sound.SR)] = audio[:, -1:] * np.linspace(1, 0, int(0.6 * sound.SR))
            audio = np.concatenate([audio, pad], axis=1)
        wav = out / f"sound_{name}.wav"
        sound.write_wav(wav, audio * 0.9)
        parts.append(str(wav))
        n = len(steps)
        timeline.append({"name": name, "start": t, "frames": n, "picked": picked(shot), "steps": [int(v) for v in steps]})
        t += n / FPS + shot.get("tail", 0.0)
        print(f"[prep] {name}: {n} frames from {t - n / FPS:.2f} s", flush=True)
    # the end: the curtain's last clank on black, then the card
    tail = np.zeros((2, int(END_CARD_S * sound.SR)))  # the card: the music's tail only
    sound.write_wav(out / "sound_end.wav", tail * 0.9)
    parts.append(str(out / "sound_end.wav"))
    total = t + END_CARD_S
    starts = {e["name"]: e["start"] for e in timeline}
    by_name = {e["name"]: e for e in timeline}

    def at_step(name, step):  # when, in the cut, a shot plays the take step nearest `step`
        st = by_name[name]["steps"]
        return starts[name] + int(np.argmin([abs(v - step) for v in st])) / FPS

    # the parry (3.2 s), the jump's hold, the hit on the claw (7.31 s), the drone into the shutter (12.25 s)
    cues = [(0.0, "start"), (starts["establish"], "pulse"), (at_step("parry", 640), "hit"),
            (starts["volley"], "drive"), (starts["wound"] + 6 / FPS, "hit"), (at_step("wound", 1462) + 0.02, "hit"),
            (at_step(SHUTTER_SHOT["name"], int((SHUTTER_SHOT["shutter"] + 0.5) * 200)), "hit"), (starts["rise"] + 0.2, "hit"),
            (t + END_CARD_S - 0.3, "end")]  # the music rings out under the card
    sound.write_wav(out / "music.wav", sound.music(total, cues) * 0.9)
    sound.mix(parts, str(out / "music.wav"), str(out / "short.wav"))
    end_card(out / "end_card.png", args.font)
    bars(out, args.font)
    (out / "timeline.json").write_text(json.dumps({"shots": timeline, "total": total, "cues": cues}, indent=1))
    print(f"PREP_DONE {out}: {total:.2f} s", flush=True)


def _wound_npz(z, out: Path, tag: str = ""):
    p = out / f"_wounded{tag}.npz"
    if not p.exists():
        np.savez(p, wounded=z["wounded"][TAKE])
    return p


def _grade(src_label, sigma):
    return (f"[{src_label}]split[a][b];[b]curves=all='0/0 0.7/0 1/1',gblur=sigma={sigma}[g];"
            "[a][g]blend=all_mode=screen:all_opacity=0.55[bl];"
            "[bl]rgbashift=rh=1:bh=-1,vignette=PI/5,colorbalance=rs=-0.03:bs=0.04:rh=0.04:bh=-0.03,"
            "eq=contrast=1.04:saturation=1.05,noise=alls=4:allf=t[gr]")


def cut(args):
    ffmpeg = args.ffmpeg
    renders, post, work = Path(args.renders), Path(args.post), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tl = json.loads((post / "timeline.json").read_text())
    clips = []
    for e in [e for e in tl["shots"] if not args.only or e["name"] in args.only.split(",")]:
        name = e["name"]
        clip = work / f"{name}.mp4"
        clips.append(clip)
        if clip.exists() and not args.force:
            continue
        if name == "legend":
            sh = next(x for x in SHOTS if x["name"] == "legend")
            src, fr = sh["still"]
            dur = sh["dur"]
            inp = ["-loop", "1", "-framerate", str(FPS), "-i", str(renders / src / f"frame_{fr:04d}.png"),
                   "-framerate", str(FPS), "-i", str(post / "legend" / "legend_%04d.png")]
            chain = (f"[0:v]scale={W}:{H},format=rgb24[s0];" + _grade("s0", 22)
                     + ";[1:v]format=rgba,fade=t=in:st=0:d=0.3:alpha=1[lg];[gr][lg]overlay=0:0:eof_action=repeat[out]")
            cmd = [ffmpeg, "-y", "-loglevel", "error", *inp, "-filter_complex", chain, "-map", "[out]", "-t", str(dur),
                   "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv444p", "-r", str(FPS), str(clip)]
            subprocess.run(cmd, check=True)
            print(f"[cut] {name}", flush=True)
            continue
        if name == "rise":
            sh = next(x for x in SHOTS if x["name"] == "rise")
            rev = work / "rise_frames"
            shutil.rmtree(rev, ignore_errors=True)
            rev.mkdir(parents=True)
            for i, f in enumerate(range(sh["files"][0], sh["files"][1] - 1, -1)):
                (rev / f"frame_{i + 1:04d}.png").symlink_to(renders / SHOTS[0]["name"] / f"frame_{f:04d}.png")
            n = e["frames"]
            inp = ["-framerate", str(FPS), "-i", str(rev / "frame_%04d.png")]
            chain = (f"[0:v]scale={W}:{H}:flags=lanczos,format=rgb24[s0];" + _grade("s0", 16)
                     + f";[gr]fade=t=in:st=0:d=0.8,fade=t=out:st={n / FPS - 0.9}:d=0.9[out]")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inp, "-filter_complex", chain, "-map", "[out]",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv444p", "-r", str(FPS), str(clip)],
                           check=True)
            print(f"[cut] {name}", flush=True)
            continue
        if name == SHOTS[0]["name"]:
            first = SHOTS[0]["files"][0]
            inp = ["-framerate", str(FPS), "-start_number", str(first), "-i", str(renders / name / "frame_%04d.png")]
            chain = (f"[0:v]scale={W}:{H}:flags=lanczos,format=rgb24[s0];" + _grade("s0", 16)
                     + f";[gr]fade=t=in:st=0:d=1.2[out]")
        else:
            seq = work / f"{name}_frames"  # the played frames, linked in order (speed ramps skip and repeat)
            shutil.rmtree(seq, ignore_errors=True)
            seq.mkdir(parents=True)
            for i, f in enumerate(e["picked"]):
                (seq / f"frame_{i + 1:04d}.png").symlink_to(renders / name / f"frame_{f:04d}.png")
            inp = ["-framerate", str(FPS), "-start_number", "1", "-i", str(seq / "frame_%04d.png"),
                   "-framerate", str(FPS), "-i", str(post / f"inset_{name}" / "brain_%04d.png"),
                   "-framerate", str(FPS), "-i", str(post / f"hud_{name}" / "hud_%04d.png")]
            chain = (f"[0:v]scale={W}:{H},format=rgb24[s0];" + _grade("s0", 22)
                     + f";[1:v]colorchannelmixer=aa=0.9[ins]"
                     + f";[gr][ins]overlay={INSET_XY[0]}:{INSET_XY[1]}[i1];[i1][2:v]overlay=0:0[out]")  # small, in the corner
            if name == SHUTTER_SHOT["name"]:  # hold the last frame, then down to black
                tail = SHUTTER_TAIL
                chain = chain.replace("[out]", f"[o1];[o1]tpad=stop_mode=clone:stop_duration={tail},"
                                               f"fade=t=out:st={e['frames'] / FPS + tail - 0.7}:d=0.7[out]")
        cmd = [ffmpeg, "-y", "-loglevel", "error", *inp, "-filter_complex", chain, "-map", "[out]",
               "-frames:v", str(e["frames"] + (int(SHUTTER_TAIL * FPS) if name == SHUTTER_SHOT["name"] else 0)),
               "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv444p", "-r", str(FPS), str(clip)]
        subprocess.run(cmd, check=True)
        print(f"[cut] {name}", flush=True)
    card = work / "end.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS), "-i", str(post / "end_card.png"),
                    "-vf", f"fade=t=in:st=0.4:d=0.8,format=yuv444p", "-t", str(END_CARD_S), "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv444p", "-r", str(FPS), str(card)], check=True)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips + [card]))
    joined = work / "joined.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)],
                   check=True)
    # the bars and the standing caption (bars.png), over everything but the end card
    body = tl["total"] - END_CARD_S
    for res, crf, name in (((W, H), 16, args.out), ((1920, 1080), 17, args.out.replace(".mp4", "_1080p.mp4"))):
        fc = (f"[0:v][2:v]overlay=0:0:enable='lt(t,{body})'"
              + (f",scale={res[0]}:{res[1]}:flags=lanczos" if res != (W, H) else "") + ",format=yuv420p[v]")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(joined), "-i", str(post / "short.wav"),
                        "-loop", "1", "-i", str(post / "bars.png"), "-filter_complex", fc,
                        "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high",
                        "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", name], check=True)
        print(f"SHORT_DONE {name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--scene", required=True)
    p.add_argument("--cord", required=True)
    p.add_argument("--scene2", default="", help="the re-recorded take for the shots marked alt (the weave)")
    p.add_argument("--cord2", default="")
    p.add_argument("--positions", required=True)
    p.add_argument("--font", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--renders", required=True)
    c.add_argument("--post", required=True)
    c.add_argument("--work", default="/content/cut")
    c.add_argument("--font", default="/content/up/NotoSansJP-VF.ttf")
    c.add_argument("--out", required=True)
    c.add_argument("--force", action="store_true")
    c.add_argument("--only", default="", help="a test cut of these shots only (comma-separated)")
    c.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else cut(a)


if __name__ == "__main__":
    main()
