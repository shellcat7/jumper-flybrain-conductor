"""Speed and memory of one VNC-model step on a GPU, before any learning is wired in.

One step is the rate update x <- leak*x + (1-leak)*tanh(gain*(W @ x) + stim) on the
signed, incoming-normalised subgraph from `build_packet.py`, for every robot and
state channel at once (x is nodes x robots*width). The backward probe runs 8 such
steps through a fixed-matrix autograd function (W^T on the way back), as the
learner will. Inputs are random: this measures time and memory, not behaviour.

    python bench_vnc.py --packet /content/up/vnc_packet.npz --packet /content/up/vnc_ge5.npz \
        --out /content/results/bench-vnc.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


class FixedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, wt):
        ctx.wt = wt
        return w @ x

    @staticmethod
    def backward(ctx, grad):
        return ctx.wt @ grad, None, None


def load(path: str, dtype: torch.dtype):
    z = np.load(path)
    n = len(z["indptr"]) - 1
    counts = z["counts"].astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(z["indptr"]))
    cols = z["indices"].astype(np.int64) & 0xFFFF  # int16 storage, nodes < 32768
    insum = np.bincount(rows, weights=counts, minlength=n)
    values = z["sign"][cols].astype(np.float32) * counts / np.maximum(insum[rows], 1.0)
    crow = torch.tensor(z["indptr"].astype(np.int64))
    w = torch.sparse_csr_tensor(crow, torch.tensor(cols), torch.tensor(values), (n, n)).to("cuda", dtype)
    wt = w.to_sparse_coo().t().coalesce().to_sparse_csr()
    return w, wt, n, int(len(values))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--packet", action="append", required=True)
    p.add_argument("--widths", default="1,2,4")
    p.add_argument("--batches", default="1024,4096,8192,16384")
    p.add_argument("--dtypes", default="float32,bfloat16")
    p.add_argument("--backward-batches", default="1024,4096,8192")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "runs": []}
    out = Path(args.out)

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    for packet in args.packet:
        for dname in args.dtypes.split(","):
            dtype = getattr(torch, dname)
            try:
                w, wt, n, e = load(packet, dtype)
            except Exception as error:  # noqa: BLE001 - an unsupported dtype is a result
                report["runs"].append({"packet": packet, "dtype": dname, "error": repr(error)[:300]})
                save()
                continue
            for width in [int(v) for v in args.widths.split(",")]:
                for batch in [int(v) for v in args.batches.split(",")]:
                    row = {"packet": Path(packet).name, "nodes": n, "edges": e, "dtype": dname,
                           "width": width, "robots": batch}
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        x = torch.zeros(n, batch * width, device="cuda", dtype=dtype)
                        stim = 0.1 * torch.randn_like(x)
                        leak, gain = 0.9, 1.0
                        times = []
                        with torch.inference_mode():
                            for k in range(13):
                                torch.cuda.synchronize()
                                s = time.perf_counter()
                                x = leak * x + (1 - leak) * torch.tanh(gain * (w @ x) + stim)
                                torch.cuda.synchronize()
                                if k >= 3:
                                    times.append((time.perf_counter() - s) * 1000)
                        ms = float(np.median(times))
                        row.update(step_ms=round(ms, 3), us_per_robot_step=round(1000 * ms / batch, 3),
                                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                   finite=bool(torch.isfinite(x).all()))
                        del x, stim
                    except torch.cuda.OutOfMemoryError:
                        row["error"] = "OOM"
                    except Exception as error:  # noqa: BLE001
                        row["error"] = repr(error)[:300]
                    report["runs"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
            for batch in [int(v) for v in args.backward_batches.split(",")]:
                row = {"packet": Path(packet).name, "dtype": dname, "width": 2, "robots": batch, "backward_frames": 8}
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # Every operand in the matrix's dtype: on 10/24 a bf16 matrix met an fp32
                    # state here and cuSPARSE refused the mix, which is not a bf16 limit.
                    gain = torch.ones(n, 1, device="cuda", dtype=dtype, requires_grad=True)
                    bias = torch.zeros(n, 1, device="cuda", dtype=dtype, requires_grad=True)
                    x = torch.zeros(n, batch * 2, device="cuda", dtype=dtype)
                    torch.cuda.synchronize()
                    s = time.perf_counter()
                    loss = 0.0
                    for _ in range(8):
                        stim = 0.1 * torch.randn(n, batch * 2, device="cuda", dtype=dtype)
                        x = 0.9 * x + 0.1 * torch.tanh(gain * FixedMatmul.apply(x, w, wt) + bias + stim)
                        loss = loss + x.square().mean()
                    loss.backward()
                    torch.cuda.synchronize()
                    row.update(wall_s=round(time.perf_counter() - s, 3),
                               peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                               finite_grads=bool(torch.isfinite(gain.grad).all()))
                    del x, loss, gain, bias
                except torch.cuda.OutOfMemoryError:
                    row["error"] = "OOM"
                except Exception as error:  # noqa: BLE001
                    row["error"] = repr(error)[:300]
                report["runs"].append(row)
                save()
                print(json.dumps(row), flush=True)
            del w, wt
    save()
    print("VNC_BENCH_DONE " + str(out))


if __name__ == "__main__":
    main()
