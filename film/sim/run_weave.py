"""Re-record the alley take with the run after the wound weaving, and a last shot at the shutter (10/28
evening, the user: the run to the shutter is a straight line, so why it gets away is unclear -- weave
left and right round bins and dumpsters; duck low under the shutter).

The same script, seed and 24 robots as `alley_parry` (take 21 is the film's), so the first 7.4 s are
the same take; from 7.4 s the command weaves (turning left and right at 0.6 m/s), and three variants
add a last shot from behind at 11.45 s for the giant-fibre reflex to duck: a volley from three sides
behind (a), a single bolt (b), none (c). The obstacles are placed in Blender where it swerved; they
were not in the simulation, and the weave is commanded, not chosen (the caption says so).

    python run_weave.py --variant a --takes 24 --checkpoint dgate6-s1-299.pt --vnc-config cfg.json --out x.npz
"""
import sys

sys.path.insert(0, "/content/repo/experiments/vnc_v1")
variant = sys.argv[sys.argv.index("--variant") + 1]
argv = [a for i, a in enumerate(sys.argv[1:]) if a != "--variant" and sys.argv[i] != "--variant"]
sys.argv = ["record_scene.py", "--scenario", f"alley_weave_{variant}"] + argv
import record_scene as rs  # noqa: E402

BASE = [(1.5, "bolt", ("left", 0.45, 1.8)), (2.8, "bolt", ("front", 0.45, 3.0)),
        (4.3, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6), ("right", 0.45, 1.8)]),
        (5.8, "bolt", ("right", 0.45, 1.8)),
        (6.6, "bolt_volley", [("front", 0.45, 3.0), ("back_left", 0.45, 2.6), ("right", 0.45, 1.8), ("left", 0.75, 1.8)]),
        (7.4, "command", (0.6, 0.0, 0.0))]
WEAVE = [(8.0, "command", (0.6, 0.0, 1.5)), (8.7, "command", (0.6, 0.0, -1.5)), (9.5, "command", (0.6, 0.0, 1.5)),
         (10.2, "command", (0.6, 0.0, -1.5)), (10.9, "command", (0.6, 0.0, 0.6)), (11.3, "command", (0.6, 0.0, 0.0))]
# (second round) the left turns overshot on five legs and the path drifted left: harder right turns
WEAVE2 = [(8.0, "command", (0.6, 0.0, 1.2)), (8.6, "command", (0.6, 0.0, -2.2)), (9.3, "command", (0.6, 0.0, 1.4)),
          (9.9, "command", (0.6, 0.0, -2.2)), (10.6, "command", (0.6, 0.0, 1.0)), (11.1, "command", (0.6, 0.0, 0.0))]
# the volley that made it duck at 4.3 s, now at the shutter: one from beyond it, two from behind
DUCK = [("front", 0.45, 3.0), ("back_left", 0.45, 2.6), ("right", 0.45, 1.8)]
LAST = {"a": (WEAVE, [(11.45, "bolt_volley", [("back", 0.45, 2.2), ("back_left", 0.45, 2.0), ("right", 0.45, 1.8)])]),
        "b": (WEAVE, [(11.45, "bolt", ("back", 0.45, 2.2))]),
        "c": (WEAVE, []),
        "d": (WEAVE2, [(11.45, "bolt_volley", DUCK)]),
        "e": (WEAVE2, [(11.3, "bolt_volley", DUCK)]),
        "f": (WEAVE, [(11.45, "bolt_volley", DUCK)]),
        # (10/28 night, the user: the drones' attack and chase looked half-hearted) the chasers keep firing
        # during the run, from behind and the sides, every 0.8-0.9 s
        "g": (WEAVE, [(8.3, "bolt", ("back", 0.45, 2.2)), (9.1, "bolt", ("back_left", 0.45, 2.0)),
                      (9.9, "bolt", ("right", 0.45, 1.8)), (10.7, "bolt", ("back", 0.45, 2.2))]),
        "h": (WEAVE, [(8.3, "bolt_volley", [("back", 0.45, 2.2), ("right", 0.45, 1.8)]),
                      (9.4, "bolt_volley", [("back_left", 0.45, 2.0), ("back", 0.45, 2.4)]),
                      (10.5, "bolt_volley", [("back", 0.45, 2.2), ("right", 0.45, 1.8)])])}
for k, (weave, last) in LAST.items():
    rs.SCENARIOS[f"alley_weave_{k}"] = {"command": (0.3, 0.0, 0.0), "duration": 13.0, "bolt_height": 0.25,
                                         "wound": True, "events": BASE + weave + last}
rs.main()
