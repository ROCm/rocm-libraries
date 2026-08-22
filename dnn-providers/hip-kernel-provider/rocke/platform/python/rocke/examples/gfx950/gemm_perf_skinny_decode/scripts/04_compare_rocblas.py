#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 4: side-by-side DSL winner vs rocBLAS on the same shape, same harness.

Runbook anchors:
- §2.2 Performance Baselines (compare on the same hardware, same warmup)
- §12.1.P Benchmark hygiene
- §17 case-study form: report the gap honestly with the per-iter signal,
  not just the latency number

rocBLAS goes through torch.matmul(bf16). We use the same warmup / iters
budget the DSL sweep used, time with torch HIP events (cuda.Event), and
report median of N attempts plus spread, matching 02_sweep_bench.py.
"""

from __future__ import annotations
import json
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]

assert torch.cuda.is_available(), "ROCm-torch required"
DEV = "cuda"

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
PEAK_BF16_TF = 2500.0
WARMUP = 20
TIMED = 200
ATTEMPTS = 5

torch.manual_seed(0)
A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
W = torch.randn(
    N, K, dtype=torch.bfloat16, device=DEV
)  # weight matrix (RowMajor [N,K])

# C = A @ W^T  (linear layer)
out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)


def time_once():
    # Match the manifest harness: warmup, then timed loop, return per-call ms.
    for _ in range(WARMUP):
        torch.matmul(A, W.t(), out=out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED):
        torch.matmul(A, W.t(), out=out)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / TIMED


samples_ms = []
for i in range(ATTEMPTS + 1):
    ms = time_once()
    if i == 0:
        continue  # cold-cache discard (§12.1.P)
    samples_ms.append(ms)

ms_med = median(samples_ms)
ms_best = min(samples_ms)
spread = (max(samples_ms) - min(samples_ms)) / ms_med * 100.0

flop = 2.0 * M * N * K
bytes_xfer = 2 * (M * K + N * K + M * N)
tflops_best = flop / (ms_best * 1e-3) / 1e12
gbs_best = bytes_xfer / (ms_best * 1e-3) / 1e9

print(f"rocBLAS bf16 (torch.matmul)  shape M={M} N={N} K={K}")
print(f"  median={ms_med:.4f}ms  spread={spread:.1f}%  best={ms_best:.4f}ms")
print(f"  → {tflops_best:.2f} TF  ({tflops_best / PEAK_BF16_TF * 100:.1f}% peak)")
print(f"  → {gbs_best:.0f} GB/s  ({gbs_best / HBM_PEAK_GBS * 100:.1f}% HBM)")

# Pull DSL winner
sweep = json.loads((ROOT / "data" / "02_sweep_bench.json").read_text())
winner = min(
    (r for r in sweep["results"] if "ms_best" in r), key=lambda r: r["ms_best"]
)

print(f"\nDSL winner: {winner['label']}")
print(
    f"  best={winner['ms_best']:.4f}ms  {winner['tflops_best']:.2f} TF  "
    f"{winner['gbs_best']:.0f} GB/s  ({winner['pct_hbm']:.1f}% HBM)"
)

ratio = winner["ms_best"] / ms_best
print(
    f"\nDSL / rocBLAS latency ratio: {ratio:.2f}×  "
    f"({'DSL faster' if ratio < 1 else 'rocBLAS faster'})"
)

out_json = ROOT / "data" / "04_compare_rocblas.json"
with out_json.open("w") as f:
    json.dump(
        {
            "shape": {"M": M, "N": N, "K": K},
            "warmup": WARMUP,
            "timed": TIMED,
            "attempts": ATTEMPTS,
            "rocblas": {
                "ms_median": ms_med,
                "ms_best": ms_best,
                "ms_spread_pct": spread,
                "tflops_best": tflops_best,
                "gbs_best": gbs_best,
                "pct_hbm": gbs_best / HBM_PEAK_GBS * 100.0,
            },
            "dsl_winner": winner,
            "dsl_over_rocblas_latency_ratio": ratio,
        },
        f,
        indent=2,
    )
print(f"\nWrote {out_json}")
