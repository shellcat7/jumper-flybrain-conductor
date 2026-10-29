"""Census of the MaleCNS graph for a lighter fly brain: where the cost is, and what a
leg-reflex model needs.

Reads Codex's classified graph (166,700 annotated neurons, matrix[post, pre] =
contact counts) and the MaleCNS v1.0 annotation and neurotransmitter tables, and
writes one JSON: edge-weight thresholds, the optic / central / VNC split, the
VNC + ascending/descending subgraph, the six leg nerves' motor and sensory pools
by side, the inhibitory share, and the escape-circuit cell types.

    python experiments/reflex_v1/vnc_census.py evidence/reflex-v1-20261023/vnc-census.json
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/toolchain/connectome-reader"))  # Codex's pyarrow

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
from scipy import sparse  # noqa: E402

OPTIC = {"ol_intrinsic", "visual_projection", "ol_sensory", "visual_centrifugal", "visual_projection_tbc"}
VNC = {"vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_tbc", "vnc_sensory_tbc", "vnc_endocrine"}
LINK = {"ascending_neuron", "descending_neuron", "sensory_ascending", "sensory_descending",
        "efferent_ascending", "efferent_descending", "descending_neuron_tbc", "sensory_ascending_tbc"}
#: Jumper leg <- fly leg nerve and side (T1 front, T2 middle, T3 hind).
LEG_NERVES = {"LF": ("ProLN", "L"), "RF": ("ProLN", "R"), "LM": ("MesoLN", "L"),
              "RM": ("MesoLN", "R"), "LR": ("MetaLN", "L"), "RR": ("MetaLN", "R")}
INHIBITORY = {"gaba", "glutamate"}  # glutamate acts mostly through GluCl in the fly CNS


def table(path: Path):
    with pa.memory_map(str(path), "r") as f:
        return ipc.open_file(f).read_all()


def main() -> None:
    out = Path(sys.argv[1])
    graph = ROOT / "runs/connectome-inventory-v1-20261016/graph-1"
    a = sparse.load_npz(graph / "classified-all-contact-counts.npz").tocsr()
    ids = np.load(graph / "body-ids.npy")
    ann = table(ROOT / "work/data/malecns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = table(ROOT / "work/data/malecns/v1.0/body-neurotransmitters-male-cns-v1.0.feather")
    rows = {r["bodyId"]: r for r in ann.select(
        ["bodyId", "superclass", "type", "subclass", "somaSide", "rootSide", "entryNerve", "exitNerve"]).to_pylist()}
    consensus = dict(zip(nt["body"].to_pylist(), nt["consensus_nt"].to_pylist()))
    col = {k: np.array([rows[b][k] for b in ids], dtype=object)
           for k in ("superclass", "type", "subclass", "somaSide", "rootSide", "entryNerve", "exitNerve")}
    sc = col["superclass"]
    side = np.where(col["somaSide"] != None, col["somaSide"], col["rootSide"])  # noqa: E711
    report = {"graph": {"nodes": int(a.shape[0]), "edges": int(a.nnz)}, "thresholds": {}}
    for th in (1, 2, 3, 5, 10):
        keep = a.data >= th
        report["thresholds"][th] = {"edges": int(keep.sum()), "edge_share": round(float(keep.mean()), 4),
                                    "contact_share": round(float(a.data[keep].sum() / a.data.sum()), 4)}
    optic, vnc, link = (np.isin(sc, list(s)) for s in (OPTIC, VNC, LINK))
    report["nodes"] = {"optic": int(optic.sum()), "vnc": int(vnc.sum()), "an_dn": int(link.sum()),
                       "central": int((~optic & ~vnc & ~link).sum())}

    def sub(mask):
        idx = np.flatnonzero(mask)
        s = a[idx][:, idx]
        return {"nodes": len(idx), "edges": int(s.nnz), "edges_ge5": int((s.data >= 5).sum())}

    report["subgraphs"] = {"non_optic": sub(~optic), "vnc_an_dn": sub(vnc | link), "vnc": sub(vnc)}
    legs = {}
    for leg, (nerve, s) in LEG_NERVES.items():
        motor = (sc == "vnc_motor") & (col["exitNerve"] == nerve) & (side == s)
        sense = (sc == "vnc_sensory") & (col["entryNerve"] == nerve) & (col["rootSide"] == s)
        legs[leg] = {"nerve": nerve, "side": s, "motor": int(motor.sum()), "sensory": int(sense.sum()),
                     "sensory_by_subclass": dict(collections.Counter(col["subclass"][sense]).most_common(8))}
    report["legs"] = legs
    haltere = (sc == "vnc_sensory") & (col["entryNerve"] == "DMetaN")
    report["haltere_afferents"] = dict(collections.Counter(col["rootSide"][haltere]))
    sel = vnc | link
    nts = collections.Counter(consensus.get(b) for b in ids[sel])
    report["neurotransmitters_vnc_an_dn"] = {str(k): v for k, v in nts.most_common()}
    report["inhibitory_share_vnc_an_dn"] = round(sum(v for k, v in nts.items() if k in INHIBITORY) / sel.sum(), 4)
    report["cell_types"] = {t: {"count": int((col["type"] == t).sum()),
                                "superclass": sorted({str(x) for x in sc[col["type"] == t]})}
                            for t in ("DNp01", "TTMn", "LPLC2", "LC4", "DNa01", "DNa02")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("nodes", "subgraphs", "inhibitory_share_vnc_an_dn")}))


if __name__ == "__main__":
    main()
