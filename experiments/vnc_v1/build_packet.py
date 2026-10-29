"""The VNC model's input packet: the thoracic nerve cord with its ascending and
descending links, signed, reordered by segment and side, with each Jumper leg's
sensory and motor pools.

Writes `vnc_packet.npz` (the whole subgraph, every contact) and `vnc_ge5.npz`
(contacts >= 5 only, small enough for the browser upload to Colab), plus a
manifest with hashes. Indices fit int16 (24,154 nodes); counts are uint16.
matrix[post, pre] = contact count, as in Codex's classified graph.

    python experiments/vnc_v1/build_packet.py runs/vnc-v1/packet-1
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/toolchain/connectome-reader"))  # Codex's pyarrow

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
from scipy import sparse  # noqa: E402

VNC = {"vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_tbc", "vnc_sensory_tbc", "vnc_endocrine"}
LINK = {"ascending_neuron", "descending_neuron", "sensory_ascending", "sensory_descending",
        "efferent_ascending", "efferent_descending", "descending_neuron_tbc", "sensory_ascending_tbc"}
LEG_NERVES = {"LF": ("ProLN", "L"), "RF": ("ProLN", "R"), "LM": ("MesoLN", "L"),
              "RM": ("MesoLN", "R"), "LR": ("MetaLN", "L"), "RR": ("MetaLN", "R")}
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}  # the rest 0: learned later
SEGMENT = {"T1": 0, "T2": 1, "T3": 2}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path):
    with pa.memory_map(str(path), "r") as f:
        return ipc.open_file(f).read_all()


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    graph = ROOT / "runs/connectome-inventory-v1-20261016/graph-1"
    a = sparse.load_npz(graph / "classified-all-contact-counts.npz").tocsr()
    ids = np.load(graph / "body-ids.npy")
    ann_path = ROOT / "work/data/malecns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path = ROOT / "work/data/malecns/v1.0/body-neurotransmitters-male-cns-v1.0.feather"
    ann, nt = table(ann_path), table(nt_path)
    keys = ["bodyId", "superclass", "type", "subclass", "somaNeuromere", "somaSide", "rootSide",
            "entryNerve", "exitNerve"]
    rows = {r["bodyId"]: r for r in ann.select(keys).to_pylist()}
    consensus = dict(zip(nt["body"].to_pylist(), nt["consensus_nt"].to_pylist()))
    info = [rows[b] for b in ids]
    keep = np.array([r["superclass"] in VNC or r["superclass"] in LINK for r in info])
    sel = np.flatnonzero(keep)

    def side_of(r):
        return r["somaSide"] or r["rootSide"] or "?"

    def segment_of(r):
        if r["somaNeuromere"] in SEGMENT:
            return SEGMENT[r["somaNeuromere"]]
        for seg, nerve in enumerate(("ProLN", "MesoLN", "MetaLN")):
            if nerve in (r["entryNerve"], r["exitNerve"]):
                return seg
        return 3 if r["superclass"] in VNC else 4  # other VNC, then the links

    # Neighbours in memory where they are neighbours in the cord: segment, side, class.
    order = sorted(sel, key=lambda i: (segment_of(info[i]), side_of(info[i]), info[i]["superclass"] or "", ids[i]))
    order = np.array(order)
    nodes = [info[i] for i in order]
    sub = a[order][:, order].tocsr()
    sub.sort_indices()
    n = len(order)
    assert n < 32768, "indices must fit int16"
    counts = np.clip(np.rint(sub.data), 0, 65535).astype(np.uint16)
    sign = np.array([SIGN.get(consensus.get(int(ids[i])), 0) for i in order], dtype=np.int8)
    ports, labels = {}, {}
    for leg, (nerve, s) in LEG_NERVES.items():
        motor = [k for k, r in enumerate(nodes) if r["superclass"] == "vnc_motor" and r["exitNerve"] == nerve and side_of(r) == s]
        sense = [k for k, r in enumerate(nodes) if r["superclass"] == "vnc_sensory" and r["entryNerve"] == nerve and r["rootSide"] == s]
        ports[f"motor_{leg}"] = np.array(motor, dtype=np.int32)
        ports[f"sensory_{leg}"] = np.array(sense, dtype=np.int32)
        labels[f"motor_{leg}"] = [nodes[k]["type"] for k in motor]
        labels[f"sensory_{leg}"] = [nodes[k]["subclass"] for k in sense]
    for s in ("L", "R"):
        ports[f"haltere_{s}"] = np.array([k for k, r in enumerate(nodes) if r["superclass"] == "vnc_sensory"
                                          and r["entryNerve"] == "DMetaN" and r["rootSide"] == s], dtype=np.int32)
    ports["descending"] = np.array([k for k, r in enumerate(nodes) if r["superclass"] == "descending_neuron"], dtype=np.int32)
    ports["ascending"] = np.array([k for k, r in enumerate(nodes) if r["superclass"] == "ascending_neuron"], dtype=np.int32)
    for t in ("DNa01", "DNa02", "DNp01", "TTMn"):
        ports[f"type_{t}"] = np.array([k for k, r in enumerate(nodes) if r["type"] == t], dtype=np.int32)

    common = dict(body_ids=ids[order].astype(np.int64), sign=sign, **ports)
    full = out / "vnc_packet.npz"
    np.savez_compressed(full, indptr=sub.indptr.astype(np.int32), indices=sub.indices.astype(np.int16),
                        counts=counts, **common)
    strong = counts >= 5
    rows_of = np.repeat(np.arange(n), np.diff(sub.indptr))
    ge5_indptr = np.concatenate([[0], np.cumsum(np.bincount(rows_of[strong], minlength=n))]).astype(np.int32)
    small = out / "vnc_ge5.npz"
    np.savez_compressed(small, indptr=ge5_indptr, indices=sub.indices[strong].astype(np.int16),
                        counts=counts[strong], **common)
    manifest = {
        "nodes": n, "edges": int(sub.nnz), "edges_ge5": int(strong.sum()),
        "inhibitory": int((sign < 0).sum()), "excitatory": int((sign > 0).sum()), "unsigned": int((sign == 0).sum()),
        "order": "segment (T1, T2, T3, other VNC, links), side, superclass, bodyId",
        "ports": {k: int(len(v)) for k, v in ports.items()},
        "motor_types": {k: sorted({str(x) for x in v}) for k, v in labels.items() if k.startswith("motor_")},
        "sources": {"graph": sha(graph / "classified-all-contact-counts.npz"), "annotations": sha(ann_path),
                    "neurotransmitters": sha(nt_path)},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in (full, small)},
        "sign_rule": "acetylcholine +1, gaba -1, glutamate -1, other 0 (consensus_nt)",
        "license": "MaleCNS v1.0 (Janelia FlyEM), CC BY 4.0",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("nodes", "edges", "edges_ge5", "inhibitory", "files")}))


if __name__ == "__main__":
    main()
