"""Sound for the short, synthesised from the take (10/28): no library samples.

Every effect is generated with numpy from what the simulator recorded in the take -- a footfall
where a foot tip comes down to the floor, the servos' whine from how fast the 22 joints turn, a
drone's shot where a bolt appears, its hiss in flight (louder and panned by where it is from the
camera), the clang where a bolt meets the claw, the crack where one strikes and wounds, the jump's
servo burst and its landing, a soft blip when the giant-fibre reflex fires -- and placed on the
shot's own frames (the frame-to-step order of `import_take.py --movie`), so it stays in step through
slow motion (pitched down, the alley muffled, more reverb) and bullet time (a held swell and a
heartbeat). Under it, the alley at night: drizzle and drips, the 100 Hz buzz of neon on Tokyo's 50 Hz
mains, an AC unit's fan, the drones' rotors, the city's rumble. The music is a slow dark synth
pulse with risers into its hits.

    python sound.py shot --scene alley_parry_slim.npz --take 21 --every 8 --range 1,45 \
        --cam-offset 0.15,-0.65,0.07 --out walk.wav
    python sound.py music --length 28 --cues 0:start,3:pulse,9.5:hit,20:hit,26.5:end --out music.wav
    python sound.py mix --parts walk.wav,parry.wav --music music.wav --out short.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_inset import shot_steps  # noqa: E402

SR = 48000
FPS = 25
STEP_HZ = 200


# ---------------------------------------------------------------- building blocks

def band(x, lo=None, hi=None):
    """Zero-phase band limit by an FFT mask (about 24 dB per octave either side)."""
    if len(x) < 4:
        return x
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo:
        m /= 1 + (lo / np.maximum(f, 1e-3)) ** 4
    if hi:
        m /= 1 + (f / hi) ** 4
    return np.fft.irfft(np.fft.rfft(x) * m, len(x))


def norm(x, rms=0.3):
    return x * (rms / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def tone(freq, n):
    """A sine from a per-sample (or constant) frequency, by phase accumulation."""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saw(freq, n):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return 2 * (np.cumsum(f) / SR % 1.0) - 1


def decay(n, tau, attack=0.0):
    t = np.arange(n) / SR
    e = np.exp(-t / tau)
    if attack > 0:
        e *= np.clip(t / attack, 0, 1)
    return e


def repitch(x, p):
    """Play `x` at rate p: p < 1 is lower and longer (slow motion)."""
    if abs(p - 1) < 1e-3:
        return x
    return np.interp(np.arange(0, len(x) - 1, p), np.arange(len(x)), x)


def reverb_ir(rng, rt60=1.0, n_early=14, width=0.0):
    """A narrow concrete alley: dense early slaps off the walls, then a decaying diffuse tail."""
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = band(ir, 180, 7000) * 0.35
    for k in range(n_early):
        d = int((0.004 + 0.0045 * k + rng.uniform(0, 0.002)) * SR)
        ir[d] += (0.8 ** k) * rng.choice([-1, 1]) * 0.6
    return ir / np.sqrt(np.sum(ir ** 2))


def convolve(x, ir):
    n = len(x) + len(ir) - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]


def pan2(x, pan):
    """Constant-power pan; `pan` scalar or per-sample in [-1, 1]."""
    a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


def place(buf, t, sig, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    s = pan2(sig * gain, pan)
    a, b = max(i, 0), min(i + len(sig), buf.shape[1])
    buf[:, a:b] += s[:, a - i: b - i]


# ---------------------------------------------------------------- the effects

def footstep(rng, hard=1.0):
    n = int(0.14 * SR)
    t = np.arange(n) / SR
    tick = norm(band(rng.standard_normal(n), 1800, 7500), 0.5) * decay(n, 0.005)
    clack = np.sin(2 * np.pi * rng.uniform(1450, 1900) * t + rng.uniform(0, 6)) * decay(n, 0.010) * 0.45
    thump = np.sin(2 * np.pi * (65 + 45 * np.exp(-t / 0.02)) * t) * decay(n, 0.032) * 0.8
    grit = norm(band(rng.standard_normal(n), 3000, 12000), 0.1) * decay(n, 0.03)  # wet grit on concrete
    return (tick + clack + thump + grit) * hard


def comb(x, delay_s, g):
    """A feedback comb (y[i] = x[i] + g y[i - d]), block by block: a springy metallic flutter."""
    d = max(int(delay_s * SR), 1)
    y = np.array(x, np.float64)
    for a in range(d, len(y), d):
        b = min(a + d, len(y))
        y[a:b] += g * y[a - d: a - d + (b - a)]
    return y


def laser_shot(rng, pitch=1.0):
    """A drone's shot, a space-opera blaster: a bright dispersive 'pew' -- the highs first, falling
    fast, as a struck steel guy-wire sounds -- through a little saturation, a springy flutter in its
    tail, over a click and a short thump. Synthesised; no samples."""
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, (f_hi, f_lo, tau, amp) in enumerate(((5200.0, 380.0, 0.045, 1.0), (3600.0, 260.0, 0.06, 0.6),
                                                 (7400.0, 900.0, 0.03, 0.4))):
        f = (f_lo + (f_hi - f_lo) * np.exp(-t / tau)) * pitch * rng.uniform(0.97, 1.03)
        out += amp * np.tanh(2.2 * tone(f, n)) * decay(n, 0.10 + 0.03 * k, attack=0.0015)
    out = norm(band(comb(out, 0.023, 0.42), 150, 12000), 0.25)
    click = norm(band(rng.standard_normal(n), 3000, 14000), 0.8) * decay(n, 0.003)
    thump = np.sin(2 * np.pi * (70 + 120 * np.exp(-t / 0.015)) * t) * decay(n, 0.05) * 0.6
    return (out + click + thump) * 1.1


def clang(rng, f0=None):
    """A bolt turned back off the claw: a hard electric crack, a buzzing clash of energy on the shell,
    a fizz of sparks and the bolt whining away -- dry, nothing that rings on (the old metal partials
    read as a temple bell once slowed down)."""
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    crack = norm(band(rng.standard_normal(n), 1500, 15000), 1.0) * decay(n, 0.008)
    buzz = band(saw(118 * (1 + 0.04 * np.sin(2 * np.pi * 31 * t)), n) + 0.7 * saw(177.5, n) + 0.5 * saw(236.8, n), 200, 6000)
    buzz = np.tanh(3 * buzz) * decay(n, 0.09, attack=0.001) * 0.5
    crackle = band((rng.random(n) < 0.02) * rng.standard_normal(n), 2500, 14000) * decay(n, 0.12) * 2.0
    away = np.r_[np.zeros(int(0.02 * SR)), laser_shot(rng, pitch=1.3)][:n]  # the ricochet, faster and higher
    away = np.r_[away, np.zeros(n - len(away))]
    tick = np.sin(2 * np.pi * 2300 * t) * decay(n, 0.012) * 0.3  # the claw's shell, dry
    return crack + buzz + crackle + 0.7 * away + tick + sparks(rng, n, 0.25) * 0.5


def sparks(rng, n, dur):
    out = np.zeros(n)
    for _ in range(int(60 * dur)):
        i = int(rng.exponential(dur * 0.4) * SR)
        if i >= n - 400:
            continue
        m = 400
        g = norm(band(rng.standard_normal(m), 4000, 16000), 0.6) * decay(m, 0.0015)
        out[i: i + m] += g * np.exp(-i / SR / dur) * rng.uniform(0.3, 1.0)
    return out


def strike(rng):
    """A bolt that strikes and wounds: crack, electric zap, sparks, a servo fault, then the warning beeps."""
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    crack = norm(band(rng.standard_normal(n), 400, 12000), 1.2) * decay(n, 0.012)
    buzz = norm(band(rng.standard_normal(n), 500, 8000), 0.6) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 120 * t)))
    zap = buzz * decay(n, 0.18) * (t < 0.45)
    thud = np.sin(2 * np.pi * (55 + 80 * np.exp(-t / 0.02)) * t) * decay(n, 0.09)
    fault = tone(900 + 500 * np.sin(2 * np.pi * 9 * t) * np.exp(-t / 0.3), n) * decay(n, 0.25) * 0.25 * (t > 0.05)
    out = crack + zap + thud + fault + sparks(rng, n, 0.6) * 0.8
    for k in range(3):  # the warning, two tones three times
        for j, f in enumerate((1320, 990)):
            a = 0.35 + k * 0.32 + j * 0.13
            seg = (t >= a) & (t < a + 0.11)
            out[seg] += 0.22 * np.sin(2 * np.pi * f * t[seg]) * np.clip((t[seg] - a) / 0.004, 0, 1)
    return out * 0.8


def jump_burst(rng):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    whine = (tone(600 + 1500 * (1 - np.exp(-t / 0.08)), n) * 0.5 + tone(150 + 300 * (t / 0.45), n) * 0.3)
    whoosh = norm(band(rng.standard_normal(n), 300, 3000), 0.5) * np.sin(np.pi * np.clip(t / 0.4, 0, 1))
    return (whine * decay(n, 0.15, attack=0.01) + whoosh) * 0.7


def landing(rng):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (42 + 70 * np.exp(-t / 0.025)) * t) * decay(n, 0.11) * 1.2
    rattle = np.zeros(n)
    for _ in range(7):
        i = int(rng.uniform(0, 0.09) * SR)
        fs = footstep(rng, rng.uniform(0.4, 0.9))
        m = min(len(fs), n - i)
        rattle[i: i + m] += fs[:m]
    return thump + rattle * 0.6


def neuron_blip(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2600 * t) + 0.6 * np.sin(2 * np.pi * 3900 * t + 1)) * decay(n, 0.07, attack=0.002) * 0.18


# ---------------------------------------------------------------- the take

def _yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def take_events(z, take):
    """Footfalls, bolt spawns, contacts, the jump and the wound, as take steps; plus per-step tracks."""
    meta = json.loads(str(z["meta"]))
    names = meta["body_names"]
    bp = z["body_pos"][take].astype(np.float64)
    ev = []
    for i, nm in enumerate(names):
        if not (nm.endswith("foot_tip_link") or nm.endswith("finger_tip_link")):
            continue
        zf = bp[:, i, 2]
        level = np.percentile(zf, 5) + 0.006
        last = -99
        for k in np.flatnonzero((zf[1:] < level) & (zf[:-1] >= level)) + 1:
            if k - last < 16:
                continue
            last = k
            v = (zf[max(k - 3, 0)] - zf[k]) * STEP_HZ / 3
            ev.append((int(k), "foot", {"pos": bp[k, i], "hard": float(np.clip(v / 0.25, 0.35, 1.4))}))
    balls = z["balls_pos"][take].astype(np.float64)
    struck = z["struck"][take]
    wounded = z["wounded"][take]
    wound_steps = set((np.flatnonzero(np.diff(wounded) != 0) + 1).tolist())
    for j in range(balls.shape[1]):
        p = balls[:, j]
        live = p[:, 0] < 20
        jump = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
        for k in np.flatnonzero(live & ((~np.r_[False, live[:-1]]) | (jump > 0.25))):
            ev.append((int(k), "shot", {"pos": p[k]}))
        for k in np.flatnonzero(np.diff(struck[:, j].astype(int)) == 1) + 1:
            kind = "strike" if any(abs(k - w) <= 4 for w in wound_steps) else "clang"
            ev.append((int(k), kind, {"pos": p[k]}))
    jumping = z["jumping"][take].astype(int)
    for k in np.flatnonzero(np.diff(jumping) == 1) + 1:
        ev.append((int(k), "jump", {"pos": bp[k, 0]}))
    # landing: the body's lowest point back to standing height after a jump
    for k in np.flatnonzero(np.diff(jumping) == -1) + 1:
        zb = bp[k:, 0, 2]
        land = k + int(np.argmax(zb < np.percentile(bp[:, 0, 2], 50) + 0.01))
        ev.append((land, "land", {"pos": bp[land, 0]}))
    for k in np.flatnonzero(np.diff(z["gf_fired"][take].astype(int)) == 1) + 1:
        ev.append((int(k), "blip", {"pos": bp[k, 0]}))
    jp = z["joint_pos"][take].astype(np.float64)
    speed = np.r_[0, np.abs(np.diff(jp, axis=0)).sum(1) * STEP_HZ]
    torque = np.abs(z["torque"][take].astype(np.float64)).sum(1)
    yaw = _yaw(z["body_quat"][take][:, 0].astype(np.float64))
    yaw = np.unwrap(yaw)
    kern = np.ones(101) / 101
    yaw_s = np.convolve(np.pad(yaw, 50, mode="edge"), kern, "valid")
    return sorted(ev, key=lambda e: e[0]), {"speed": speed, "torque": torque, "base": bp[:, 0], "yaw": yaw_s,
                                            "balls": balls}


class Listener:
    """The camera: robot position plus the follow offset in the robot's (smoothed) heading."""

    def __init__(self, tracks, offset):
        self.base, self.yaw, self.off = tracks["base"], tracks["yaw"], np.asarray(offset, np.float64)

    def at(self, step):
        k = int(np.clip(step, 0, len(self.yaw) - 1))
        c, s = np.cos(self.yaw[k]), np.sin(self.yaw[k])
        cam = self.base[k] + np.array([c * self.off[0] - s * self.off[1], s * self.off[0] + c * self.off[1], self.off[2]])
        view = self.base[k][:2] - cam[:2]
        view /= np.linalg.norm(view) + 1e-9
        return cam, np.array([view[1], -view[0]])

    def gain_pan(self, step, pos, ref=0.4):
        cam, right = self.at(step)
        d = pos - cam
        dist = np.linalg.norm(d) + 1e-6
        return ref / (dist + ref * 0.6), float(np.clip(np.dot(d[:2], right) / dist * 1.3, -0.95, 0.95))


# ---------------------------------------------------------------- the alley

def ambience(n, rng, drones=True):
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for c in range(2):
        rain = norm(band(rng.standard_normal(n), 2000, 10000), 0.020)
        swell = 0.7 + 0.3 * band(rng.standard_normal(n), None, 0.4) / 0.02
        out[c] += rain * np.clip(swell, 0.3, 1.3)
        out[c] += norm(band(rng.standard_normal(n), 25, 180), 0.045)  # the city's rumble
        out[c] += norm(band(rng.standard_normal(n), 100, 600), 0.010)  # the AC unit's fan
    for _ in range(int(22 * n / SR)):  # drizzle drops
        i = int(rng.uniform(0, n))
        m = int(0.02 * SR)
        place(out, i / SR, np.sin(2 * np.pi * rng.uniform(2000, 5500) * np.arange(m) / SR) * decay(m, rng.uniform(0.002, 0.006)),
              rng.uniform(0.004, 0.015), rng.uniform(-1, 1))
    for _ in range(int(0.7 * n / SR)):  # drips off the pipes, into puddles and onto metal
        m = int(0.15 * SR)
        tt = np.arange(m) / SR
        if rng.random() < 0.5:
            s = tone(700 + 1200 * (1 - np.exp(-tt / 0.02)), m) * decay(m, 0.03, attack=0.001)
        else:
            s = np.sin(2 * np.pi * rng.uniform(1200, 2600) * tt) * decay(m, 0.06)
        place(out, rng.uniform(0, n / SR), s, rng.uniform(0.02, 0.05), rng.uniform(-0.9, 0.9))
    hum = sum(np.sin(2 * np.pi * 100 * h * t + h) / h ** 1.3 for h in range(1, 9))  # neon on 50 Hz mains
    sizzle = norm(band(rng.standard_normal(n), 3000, 6000), 0.004) * (1 + np.sin(2 * np.pi * 100 * t))
    out += pan2(hum * 0.006 + sizzle, 0.35)
    if drones:
        for k, (f, pan) in enumerate(((118.0, -0.5), (124.5, 0.4), (111.0, 0.1))):
            jit = f * (1 + 0.012 * band(rng.standard_normal(n), None, 0.8) / 0.03)
            rotor = saw(jit, n)
            rotor = band(rotor, 60, 1400) * (0.75 + 0.25 * np.sin(2 * np.pi * (0.21 + 0.07 * k) * t + k))
            whine = tone(jit * 11.7, n) * 0.15
            out += pan2((rotor + whine) * 0.010, pan)
    return out


def shutter_track(rng, t0, t1, T):
    """The ending's sounds in take time from t0 to t1 s (mono): the shutter rolling down (motor hum,
    slats rattling in the guides), the chasing drone's dive, its crash into the curtain (a boom, the
    corrugated steel ringing and shuddering, plastic cracking, sparks, the rotors dying) and its fall.
    T: when the robot's base passes under (shutter.py keys the dive and the hit from it)."""
    n = int((t1 - t0) * SR)
    t = t0 + np.arange(n) / SR
    out = np.zeros(n)
    down = np.clip((T + 2.0 - t) / 0.3, 0, 1)  # the curtain moving until it shuts
    hum = sum(np.sin(2 * np.pi * 100 * h * t) / h for h in (1, 2, 3, 5)) * 0.03
    rattle = norm(band(rng.standard_normal(n), 300, 3500), 0.05) * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * t) ** 2)
    out += (hum + rattle) * down
    for _ in range(int(22 * (t1 - t0))):  # slats knocking in the guides
        i = int(rng.uniform(0, n))
        m = int(0.03 * SR)
        place_mono(out, i, np.sin(2 * np.pi * rng.uniform(700, 1600) * np.arange(m) / SR) * decay(m, 0.006) * down[i] * rng.uniform(0.04, 0.1))
    a, h = T - 0.75, T + 0.5
    dive = np.clip((t - a) / (h - a), 0, 1) * (t < h)
    f = 190 * (1 + 0.25 * dive)
    rotor = band(saw(f, n), 80, 3000) * 0.5 + tone(f * 11.5, n) * 0.25
    out += rotor * (dive ** 1.5) * 0.35
    i = int((h - t0) * SR)
    if 0 <= i < n:
        m = min(int(2.5 * SR), n - i)
        tt = np.arange(m) / SR
        boom = np.sin(2 * np.pi * (40 + 70 * np.exp(-tt / 0.03)) * tt) * decay(m, 0.3) * 1.0
        sheet = np.zeros(m)
        for _ in range(18):  # corrugated steel ringing, shuddering at the swing's 6 Hz
            sheet += np.sin(2 * np.pi * rng.uniform(90, 2400) * tt + rng.uniform(0, 6)) * decay(m, rng.uniform(0.3, 1.5)) * rng.uniform(0.1, 0.35)
        sheet *= 0.7 + 0.3 * np.cos(2 * np.pi * 6 * tt) * np.exp(-tt / 0.4)
        crack = norm(band(rng.standard_normal(m), 2000, 12000), 0.9) * decay(m, 0.02)
        sputter = tone(f[i] * 11.5 * np.exp(-tt / 0.15), m) * decay(m, 0.2) * (rng.random(m) < 0.6) * 0.2
        out[i: i + m] += boom + sheet * 0.6 + crack + sparks(rng, m, 0.5) * 0.7 + sputter
        for at, g in ((0.34, 0.8), (0.42, 0.5)):  # the fall onto its back and a bounce
            j = int((h + at - t0) * SR)
            if j < n:
                s = landing(rng)[: n - j] * g * 0.5
                out[j: j + len(s)] += s
        j = int((h + 0.45 - t0) * SR)
        if j < n:
            m2 = min(int(0.5 * SR), n - j)
            out[j: j + m2] += norm(band(rng.standard_normal(m2), 1000, 5000), 0.06) * np.linspace(1, 0, m2)  # the skid
    return out


def place_mono(buf, i, sig):
    b = min(i + len(sig), len(buf))
    if 0 <= i < b:
        buf[i:b] += sig[: b - i]


def drone_flight(dur, rng):
    """(10/28) The drone move over Shinjuku: wind at altitude, a rushing swell as it drops past the towers'
    flanks, then the street at night coming up -- traffic, a crowd murmur, neon buzz, a far horn."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    out = np.zeros((2, n))
    for c in range(2):
        wind = norm(band(rng.standard_normal(n), 120, 2400), 0.06) * (0.6 + 0.4 * np.clip(1 - u, 0, 1))
        rush = norm(band(rng.standard_normal(n), 300, 5000), 0.09) * np.exp(-((u - 0.5) / 0.12) ** 2)
        street = (norm(band(rng.standard_normal(n), 40, 400), 0.07) + norm(band(rng.standard_normal(n), 300, 3000), 0.025))             * np.clip((u - 0.62) / 0.25, 0, 1)
        out[c] += wind + rush + street
    for k in range(3):  # the towers going past: low whooshes
        at = 0.38 + 0.08 * k
        m = int(0.9 * SR)
        sw = norm(band(rng.standard_normal(m), 80, 1500), 0.25) * np.sin(np.pi * np.linspace(0, 1, m)) ** 2
        place(out, at * dur, sw, 0.8, (-0.6, 0.5, -0.2)[k])
    hum = sum(np.sin(2 * np.pi * 100 * h * t) / h ** 1.2 for h in range(1, 6)) * 0.008 * np.clip((u - 0.7) / 0.2, 0, 1)
    out += pan2(hum, 0.2)
    horn = tone(np.where(t < dur * 0.86 + 0.25, 392.0, 349.0), n) * 0.02 * ((u > 0.86) & (u < 0.86 + 0.6 / dur))
    out += pan2(band(horn, 200, 3000), -0.5)
    return out


def aerial_calm(dur, rng):
    """(10/28 afternoon) The slow overview of Shinjuku: soft wind at altitude, the city's low roar far
    below, a siren a long way off, and as the view tilts down into a street at the end, its traffic and
    neon hum rising faintly. No whooshes -- nothing rushes past in this move."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    out = np.zeros((2, n))
    for c in range(2):
        gust = 0.7 + 0.3 * band(rng.standard_normal(n), None, 0.3) / 0.02
        out[c] += norm(band(rng.standard_normal(n), 150, 1800), 0.035) * np.clip(gust, 0.3, 1.3)
        out[c] += norm(band(rng.standard_normal(n), 30, 260), 0.05)
        out[c] += (norm(band(rng.standard_normal(n), 60, 500), 0.03) + norm(band(rng.standard_normal(n), 400, 2500), 0.01))             * np.clip((u - 0.6) / 0.35, 0, 1)
    siren = tone(720 + 160 * np.sin(2 * np.pi * 0.4 * t), n) * 0.008 * np.clip(t / 2.0, 0, 1) * np.clip((dur - t) / 2.0, 0, 1)
    out += pan2(band(siren, 300, 2200), -0.5)
    hum = sum(np.sin(2 * np.pi * 100 * h * t) / h ** 1.2 for h in range(1, 6)) * 0.004 * np.clip((u - 0.75) / 0.2, 0, 1)
    out += pan2(hum, 0.15)
    return out


def city_ambience(dur, rng):
    """High over the city at night: wind, the city's roar, a siren far off."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for c in range(2):
        gust = 0.6 + 0.4 * band(rng.standard_normal(n), None, 0.5) / 0.02
        out[c] += norm(band(rng.standard_normal(n), 80, 900), 0.05) * np.clip(gust, 0.2, 1.4)
        out[c] += norm(band(rng.standard_normal(n), 30, 300), 0.06)
    siren = tone(760 + 180 * np.sin(2 * np.pi * 0.45 * t), n) * 0.012 * np.clip(t / 1.5, 0, 1)
    out += pan2(band(siren, 300, 2500), -0.6)
    return out


# ---------------------------------------------------------------- one shot

def shot(z, take, every, first, last, bullet, offset, seed=7, ambient=True, shutter_T=None, steps=None):
    """`steps`: the take step of each played frame, when the cut ramps the speed (default: every rendered frame)."""
    rng = np.random.default_rng(seed)
    steps = np.array(shot_steps(every, first, last, bullet) if steps is None else steps, np.float64)
    frames = len(steps)
    n = int(frames / FPS * SR)
    t_frames = np.arange(frames) / FPS
    rate_f = np.r_[np.diff(steps), np.diff(steps)[-1:] if frames > 1 else [every]] / (STEP_HZ / FPS)
    t = np.arange(n) / SR
    tau = np.interp(t, t_frames, steps)  # the take step heard at each sample
    rate = np.interp(t, t_frames, rate_f)
    events, tracks = take_events(z, take)
    ear = Listener(tracks, offset)
    dry, wet = np.zeros((2, n)), np.zeros((2, n))
    for step, kind, info in events:
        if step < steps[0] or step > steps[-1]:
            continue
        k = int(np.searchsorted(steps, step, side="right") - 1)
        k = min(k, frames - 1)
        span = steps[k + 1] - steps[k] if k + 1 < frames else every
        if span <= 0:
            continue
        at = (k + (step - steps[k]) / span) / FPS
        p = float(np.clip(rate_f[k], 0.2, 1.0) ** 0.55)
        if kind in ("shot", "clang", "strike"):  # stylised, not stretched: slowed blasters turn to mush
            p = max(p, 0.72)
        g, pan = ear.gain_pan(step, info["pos"])
        sig = {"foot": lambda: footstep(rng, info["hard"]) * 0.5, "shot": lambda: laser_shot(rng) * 0.9,
               "clang": lambda: clang(rng) * 2.2, "strike": lambda: strike(rng) * 2.0, "jump": lambda: jump_burst(rng),
               "land": lambda: landing(rng) * 1.5, "blip": lambda: neuron_blip(rng) / max(g, 0.2)}[kind]()
        sig = repitch(sig, p)
        send = 0.18 + 0.45 * (1 - p)
        place(dry, at, sig, g * (1 - send * 0.5), pan)
        place(wet, at, sig, g * send, pan)
    # continuous: the servos and each bolt in flight
    k_s = np.clip(tau.astype(int), 0, len(tracks["speed"]) - 1)
    sp = np.convolve(tracks["speed"], np.ones(20) / 20, "same")[k_s]
    tq = np.convolve(tracks["torque"], np.ones(20) / 20, "same")[k_s]
    s = np.clip(sp / np.percentile(tracks["speed"], 90), 0, 1.6) ** 1.5  # quiet when standing, a whine when it hurries
    p_c = np.clip(rate, 0.2, 1.0) ** 0.55
    f = (480 + 720 * s) * p_c
    gear = band(np.sign(tone(f * 0.125, n)), 80, 2500) * 0.25
    whine = (tone(f, n) + 0.35 * tone(f * 2.01, n) + 0.15 * tone(f * 3.02, n) + gear)
    g_rob, _ = ear.gain_pan(steps[0], tracks["base"][int(steps[0])])
    amp = (0.05 * s ** 0.8 + 0.012 * np.clip(tq / max(np.percentile(tracks["torque"], 95), 1e-6), 0, 1.5)) * g_rob
    dry += pan2(whine * amp, 0.0)
    balls = tracks["balls"]
    for j in range(balls.shape[1]):
        bp = balls[k_s, j]
        live = bp[:, 0] < 20
        if not live.any():
            continue
        vel = np.gradient(balls[:, j], axis=0)[k_s] * STEP_HZ
        cam_steps = np.unique(k_s[:: SR // 100])
        cams = {int(c): ear.at(c) for c in cam_steps}
        idx = np.searchsorted(cam_steps, k_s).clip(0, len(cam_steps) - 1)
        cam = np.stack([cams[int(cam_steps[i])][0] for i in range(len(cam_steps))])[idx]
        right = np.stack([cams[int(cam_steps[i])][1] for i in range(len(cam_steps))])[idx]
        d = bp - cam
        dist = np.linalg.norm(d, axis=1) + 1e-6
        radial = np.sum(vel * d, axis=1) / dist
        speed = np.linalg.norm(vel, axis=1)
        air = np.where(bp[:, 2] > 0.04, 1.0, 0.25) * (speed > 0.8) * live * (dist < 6)
        loud = air * 0.5 / (dist + 0.25)
        hiss = norm(band(rng.standard_normal(n), 900, 7000), 0.25)
        crackle = (rng.random(n) < 0.004) * rng.standard_normal(n) * 1.5
        dop = 40 / (40 + radial)
        hum = band(np.tanh(2.0 * saw(150 * dop * p_c, n)) + 0.5 * np.tanh(2.0 * saw(301 * dop * p_c, n)), 120, 5000) * 0.22
        sing = tone(980 * dop * p_c, n) * 0.08
        sig = (hiss * 0.6 + band(crackle, 2000, 12000) + hum + sing) * band(loud, None, 30)
        pan = np.clip(np.sum(d[:, :2] * right, axis=1) / dist * 1.3, -0.95, 0.95)
        dry += pan2(sig * 0.8, pan)
        wet += pan2(sig * 0.3, pan)
    # bullet time: a swell into each hold, a low drone and a heartbeat through it, a whoomp out of it
    hold = rate < 0.02
    if hold.any():
        edges = np.flatnonzero(np.diff(hold.astype(int)))
        starts = [e + 1 for e in edges if hold[e + 1]]
        ends = [e + 1 for e in edges if not hold[e + 1]]
        for a in starts:
            m = int(0.6 * SR)
            sw = norm(band(rng.standard_normal(m), 200, 4000), 0.2) * np.linspace(0, 1, m) ** 2
            place(dry, a / SR - 0.6, sw, 1.0, 0.0)
            place(dry, a / SR, landing(rng)[: int(0.5 * SR)] * 0.8, 1.0, 0.0)
        drone = (np.sin(2 * np.pi * 41 * t) + 0.5 * np.sin(2 * np.pi * 61.5 * t)) * 0.08
        hb = np.zeros(n)
        for a in starts:
            end = next((e for e in ends if e > a), n)
            for b in np.arange(a + int(0.3 * SR), end - int(0.2 * SR), int(0.85 * SR)):
                for off, g in ((0, 1.0), (int(0.22 * SR), 0.6)):
                    m = int(0.25 * SR)
                    beat = np.sin(2 * np.pi * (48 + 30 * np.exp(-np.arange(m) / SR / 0.02)) * np.arange(m) / SR) * decay(m, 0.06)
                    hb[b + off: b + off + m] += beat[: max(0, min(m, n - b - off))] * g * 0.5
        hold_s = band(hold.astype(float), None, 8)
        dry += pan2((drone * hold_s + hb), 0.0)
        for e in ends:
            m = int(0.5 * SR)
            tt = np.arange(m) / SR
            place(dry, e / SR, norm(band(rng.standard_normal(m), 150, 2500), 0.4) * decay(m, 0.15, attack=0.01)
                  + np.sin(2 * np.pi * (50 + 40 * np.exp(-tt / 0.05)) * tt) * decay(m, 0.2) * 0.6, 1.0, 0.0)
    if shutter_T is not None:  # the ending: made in take time, heard through the shot's own clock (slowed, lowered)
        secs = tau / STEP_HZ
        t0, t1 = float(secs.min()), float(secs.max()) + 0.05
        track = shutter_track(rng, t0, t1, shutter_T)
        heard = np.interp(secs, t0 + np.arange(len(track)) / SR, track)
        dry += pan2(heard * 0.9, 0.0)
        wet += pan2(heard * 0.35, 0.0)
    wet = np.stack([convolve(wet[0], reverb_ir(rng, 1.0)), convolve(wet[1], reverb_ir(rng, 1.0))])  # two tails, wide
    mix = dry + wet * 0.8
    if ambient:
        amb = ambience(n, rng)
        m = band(1 - np.clip(rate, 0, 1) ** 0.5, None, 6)  # slow motion muffles the world
        muffled = np.stack([band(a, None, 450) for a in amb]) * 1.3
        mix += amb * (1 - m) + muffled * m
    return mix


# ---------------------------------------------------------------- music

def music(length, cues, seed=11):
    """A slow dark pulse in D minor at 100 BPM: sub drone and pad from the start, the 16th-note bass
    from 'pulse', kicks from 'drive', a riser into each 'hit' and a boom on it, all out at 'end'."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    when = {k: [c for c, kk in cues if kk == k] for k in ("start", "pulse", "drive", "hit", "end")}
    end = min(when["end"] or [length])
    t0 = min(when["start"] or [0])
    fade_in = np.clip((t - t0) / 2.5, 0, 1)
    master = fade_in * np.clip((end - t) / 2.5, 0, 1)  # (10/28) a slow fade out under the end card
    d1 = 36.71
    for c, det in enumerate((-0.12, 0.13)):
        sub = np.sin(2 * np.pi * d1 * t) * 0.35 + band(saw(d1 * 2 + det, n), None, 300) * 0.25
        pad = sum(band(saw(f * (1 + det * 0.004), n), None, 900 + 500 * c) for f in (146.83, 174.61, 220.0)) * 0.05
        pad *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + c)
        out[c] += (sub + pad)
    beat = 60 / 100
    if when["pulse"]:
        p0 = min(when["pulse"])
        notes = [73.42, 73.42, 73.42, 87.31, 73.42, 73.42, 65.41, 73.42]
        k = 0
        for a in np.arange(p0, end, beat / 4):
            m = int(beat / 4 * SR)
            tt = np.arange(m) / SR
            f = notes[k % len(notes)]
            acc = 1.0 if k % 4 == 0 else 0.6
            blip = band(saw(f, m) + 0.5 * saw(f * 1.005, m), None, 700 + 900 * acc) * decay(m, 0.06) * 0.22 * acc
            place(out, a, blip, 1.0, 0.0)
            k += 1
    if when["drive"]:
        for a in np.arange(min(when["drive"]), end, beat):
            m = int(0.4 * SR)
            tt = np.arange(m) / SR
            kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-tt / 0.03)) * tt) * decay(m, 0.13) * 0.55
            place(out, a, kick, 1.0, 0.0)
            hat = norm(band(rng.standard_normal(int(0.05 * SR)), 7000, 15000), 0.05) * decay(int(0.05 * SR), 0.01)
            place(out, a + beat / 2, hat, 1.0, 0.3)
    for h in when["hit"]:
        m = int(2.0 * SR)
        tt = np.arange(m) / SR
        riser = norm(band(rng.standard_normal(m), 400, 9000), 0.12) * (tt / 2.0) ** 3 + tone(200 + 900 * (tt / 2.0) ** 2, m) * 0.05 * (tt / 2.0) ** 2
        place(out, h - 2.0, riser, 1.0, 0.0)
        m = int(3.0 * SR)
        tt = np.arange(m) / SR
        boom = np.sin(2 * np.pi * (32 + 60 * np.exp(-tt / 0.06)) * tt) * decay(m, 0.8) * 0.8
        stab = band(saw(146.83, m) + saw(220.0, m) + saw(293.66 * 1.003, m), None, 2400) * decay(m, 0.5) * 0.12
        place(out, h, boom + stab, 1.0, 0.0)
    return out * master


# ---------------------------------------------------------------- I/O and the mix

def limit(x, ceiling=0.89, look_s=0.004, release_s=0.12):
    """A look-ahead peak limiter: the gain comes down just before a peak and recovers slowly, so the
    transients stay clean (a tanh over the whole mix crunched every hit)."""
    x = np.asarray(x, np.float64)
    a = np.max(np.abs(x), axis=0)
    w = max(int(look_s * SR), 1)
    pk = np.copy(a)
    for k in range(1, w):  # the peak over the look-ahead window
        pk[:-k] = np.maximum(pk[:-k], a[k:])
    need = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    r = np.exp(-1.0 / (release_s * SR))
    g = np.empty_like(need)
    cur = 1.0
    for i in range(len(need)):  # instant down, slow up
        cur = need[i] if need[i] < cur else cur * r + (1 - r) * need[i]
        g[i] = cur
    g = np.convolve(g, np.ones(w) / w, "same")
    return x * np.minimum(g, need)


def write_wav(path, x):
    x = np.asarray(x)
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.98:
        x = limit(x, 0.97)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32767
        return a.reshape(-1, w.getnchannels()).T


def mix(parts, music_path, out, music_gain=0.4, xfade=0.01):
    clips = [read_wav(p) for p in parts]
    total = sum(c.shape[1] for c in clips)
    buf = np.zeros((2, total))
    i = 0
    x = int(xfade * SR)
    for c in clips:
        c = c.copy()
        c[:, :x] *= np.linspace(0, 1, x)
        c[:, -x:] *= np.linspace(1, 0, x)
        buf[:, i: i + c.shape[1]] += c
        i += c.shape[1]
    if music_path:
        m = read_wav(music_path)
        k = min(m.shape[1], total)
        buf[:, :k] += m[:, :k] * music_gain
    buf *= 0.16 / (np.sqrt(np.mean(buf ** 2)) + 1e-9)  # about -16 dBFS RMS, then the limiter at -1 dBFS
    write_wav(out, limit(buf, 0.89))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shot")
    s.add_argument("--scene", required=True)
    s.add_argument("--take", type=int, default=21)
    s.add_argument("--every", type=int, required=True)
    s.add_argument("--range", required=True)
    s.add_argument("--bullet-time", default="")
    s.add_argument("--cam-offset", default="0.5,-0.5,0.08")
    s.add_argument("--no-ambience", action="store_true")
    s.add_argument("--shutter", type=float, default=None, help="the ending: the robot passes under at this take time")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)
    m = sub.add_parser("music")
    m.add_argument("--length", type=float, required=True)
    m.add_argument("--cues", required=True, help="time:kind,... with kinds start, pulse, drive, hit, end")
    m.add_argument("--out", required=True)
    x = sub.add_parser("mix")
    x.add_argument("--parts", required=True)
    x.add_argument("--music", default="")
    x.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "shot":
        z = np.load(a.scene)
        first, last = (int(v) for v in a.range.split(","))
        audio = shot(z, a.take, a.every, first, last, a.bullet_time, [float(v) for v in a.cam_offset.split(",")],
                     seed=a.seed, ambient=not a.no_ambience, shutter_T=a.shutter)
        write_wav(a.out, audio * 0.9)
        print(f"SOUND_SHOT {a.out} {audio.shape[1] / SR:.2f} s")
    elif a.cmd == "music":
        cues = [(float(c.split(":")[0]), c.split(":")[1]) for c in a.cues.split(",")]
        write_wav(a.out, music(a.length, cues) * 0.9)
        print(f"SOUND_MUSIC {a.out}")
    else:
        mix(a.parts.split(","), a.music, a.out)
        print(f"SOUND_MIX {a.out}")


if __name__ == "__main__":
    main()
