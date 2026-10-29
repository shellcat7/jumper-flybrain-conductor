"""The scoreboard: a brain against the official on troubles, top speed and thrown balls.

Reads evaluations already written by `experiments/reflex_v1/eval_official.py` (summary.csv) and
`experiments/vnc_v1/eval_escape.py` (summary.json) and prints one markdown report. Pure standard
library.

    python -I experiments/vnc_v1/scoreboard.py --name "dgate2 + reflex" \
        --troubles runs/vnc-v1/eval/official-f64 runs/vnc-v1/eval/dgate2-s1-300-f64 \
        --sprint runs/vnc-v1/eval/official-sprint20-f64 <brain sprint dir> \
        --escape runs/vnc-v1/eval/escape-official-f09:runs/vnc-v1/eval/escape-dgate300-gf-f0.9 ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def speeds(path: str) -> dict:
    with open(Path(path) / "summary.csv", encoding="utf-8") as f:
        return {(r["command"], r["condition"]): float(r["speed_ratio_post"]) for r in csv.DictReader(f)}


def hit_rate(path: str) -> tuple[float, float]:
    s = json.loads((Path(path) / "summary.json").read_text())
    return float(s["meta"]["flight_s"]), float(s["meta"]["hit_rate"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default="brain")
    p.add_argument("--troubles", nargs=2, metavar=("OFFICIAL", "BRAIN"))
    p.add_argument("--sprint", nargs=2, metavar=("OFFICIAL", "BRAIN"))
    p.add_argument("--escape", nargs="*", default=[], help="official_dir:brain_dir per flight time")
    p.add_argument("--margin", type=float, default=0.02, help="a cell counts as better/worse past this")
    args = p.parse_args()
    lines = [f"## {args.name} against the official", ""]
    if args.troubles:
        o, b = speeds(args.troubles[0]), speeds(args.troubles[1])
        keys = sorted(set(o) & set(b), key=lambda k: (k[1] != "nominal", k[1], k[0]))
        nom = [k for k in keys if k[1] == "nominal"]
        tro = [k for k in keys if k[1] != "nominal"]
        mean = lambda m, ks: sum(m[k] for k in ks) / len(ks)  # noqa: E731
        better = [k for k in tro if b[k] > o[k] + args.margin]
        worse = [k for k in tro if b[k] < o[k] - args.margin]
        lines += [
            "**Troubles** (speed ratio after the trouble strikes; higher is better)", "",
            "| | official | " + args.name + " |", "|---|---|---|",
            f"| flat ground, mean of {len(nom)} | {mean(o, nom):.3f} | {mean(b, nom):.3f} |",
            f"| troubles, mean of {len(tro)} | {mean(o, tro):.3f} | {mean(b, tro):.3f} "
            f"({100 * (mean(b, tro) / mean(o, tro) - 1):+.1f}%) |",
            f"| trouble cells better / worse by > {args.margin} | | {len(better)} / {len(worse)} |", "",
            "| condition | command | official | " + args.name + " |", "|---|---|---|---|",
        ]
        lines += [f"| {k[1]} | {k[0]} | {o[k]:.2f} | {b[k]:.2f} |" for k in keys]
        lines.append("")
    if args.sprint:
        o, b = speeds(args.sprint[0]), speeds(args.sprint[1])
        from_cmd = {"walk": 0.3, "run": 0.6, "fast": 0.8, "sprint": 1.0, "sprint12": 1.2, "sprint15": 1.5,
                    "sprint18": 1.8, "sprint20": 2.0}
        cmds = sorted({k[0] for k in o if k[1] == "nominal"} & {k[0] for k in b if k[1] == "nominal"},
                      key=lambda c: from_cmd.get(c, 0))
        lines += ["**Speed on flat ground** (m/s reached: command x speed ratio)", "",
                  "| command (m/s) | official | " + args.name + " |", "|---|---|---|"]
        for c in cmds:
            v = from_cmd.get(c, 0.0)
            lines.append(f"| {v:.1f} | {v * o[(c, 'nominal')]:.2f} | {v * b[(c, 'nominal')]:.2f} |")
        top_o = max(from_cmd.get(c, 0) * o[(c, "nominal")] for c in cmds)
        top_b = max(from_cmd.get(c, 0) * b[(c, "nominal")] for c in cmds)
        lines += [f"| top speed | {top_o:.2f} | {top_b:.2f} ({100 * (top_b / top_o - 1):+.0f}%) |", ""]
    if args.escape:
        lines += ["**Thrown balls** (share of throws that hit; lower is better)", "",
                  "| ball flight (s) | official | " + args.name + " |", "|---|---|---|"]
        for pair in args.escape:
            od, bd = pair.split(":", 1) if ":" in pair[2:] else (None, pair)
            fo, ho = hit_rate(od) if od else (None, float("nan"))
            fb, hb = hit_rate(bd)
            lines.append(f"| {fb:.1f} | {ho:.2f} | {hb:.2f} |")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
