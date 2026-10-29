"""Does the fly's wiring keep each leg's reflex local? Two checks on the VNC packet.

1. Hops: the shortest path, in synapses, from each Jumper leg's sensory pool to
   each leg's motor pool (breadth-first over pre -> post, contacts >= 5).
2. Response: inject a step into one leg's sensory pool and run the untrained rate
   model (signed, incoming-normalised, x <- 0.9x + 0.1 tanh(Wx + s)) for 40 steps;
   report the mean |activity| of every leg's motor pool.

A diagonal that stands out means the anatomy already separates the legs, which
is what the leg-to-leg wiring of the design relies on. CPU only, one robot.

    python experiments/vnc_v1/leg_locality.py runs/vnc-v1/packet-1/vnc_ge5.npz evidence/vnc-v1-20261024/leg-locality.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")


def main() -> None:
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    n = len(z["indptr"]) - 1
    cols = z["indices"].astype(np.int64) & 0xFFFF
    counts = z["counts"].astype(np.float64)
    rows = np.repeat(np.arange(n), np.diff(z["indptr"]))
    post_pre = sparse.csr_matrix((counts, (rows, cols)), shape=(n, n))  # matrix[post, pre]
    # Hops: a path runs pre -> post, i.e. along the transpose.
    pre_post = (post_pre.T > 0).astype(np.float64).tocsr()
    sens = {leg: z[f"sensory_{leg}"] for leg in LEGS}
    motor = {leg: z[f"motor_{leg}"] for leg in LEGS}
    hops = {}
    for a in LEGS:
        d = shortest_path(pre_post, method="D", unweighted=True, indices=sens[a])
        best = d.min(axis=0)
        hops[a] = {b: float(np.median(best[motor[b]])) for b in LEGS}
    # Response of the untrained, signed rate model.
    insum = np.asarray(post_pre.sum(axis=1)).ravel()
    sign = z["sign"].astype(np.float64)
    w = sparse.csr_matrix((sign[cols] * counts / np.maximum(insum[rows], 1.0), (rows, cols)), shape=(n, n))
    response = {}
    for a in LEGS:
        x = np.zeros(n)
        s = np.zeros(n)
        s[sens[a]] = 1.0
        trace = []
        for _ in range(40):
            x = 0.9 * x + 0.1 * np.tanh(w @ x + s)
            trace.append({b: float(np.abs(x[motor[b]]).mean()) for b in LEGS})
        response[a] = {b: float(np.mean([t[b] for t in trace[-10:]])) for b in LEGS}
    ratio = {}
    for a in LEGS:
        others = [response[a][b] for b in LEGS if b != a]
        ratio[a] = response[a][a] / max(np.mean(others), 1e-12)
    report = {"packet": Path(sys.argv[1]).name, "nodes": n, "edges": int(post_pre.nnz),
              "median_hops_sensory_to_motor": hops, "motor_response": response,
              "own_leg_over_mean_of_others": ratio,
              "model": "untrained, signed, incoming-normalised; x <- 0.9x + 0.1 tanh(Wx + s), 40 steps, last 10 averaged"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("hops (rows: stimulated leg's sensors, cols: motor pools)")
    for a in LEGS:
        print(a, " ".join(f"{hops[a][b]:4.1f}" for b in LEGS))
    print("response x1000")
    for a in LEGS:
        print(a, " ".join(f"{1000 * response[a][b]:6.2f}" for b in LEGS), f" own/others {ratio[a]:.1f}")


if __name__ == "__main__":
    main()
