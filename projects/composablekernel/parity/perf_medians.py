#!/usr/bin/env python3
"""Task #20 — performance medians (>=10 runs) for the bridge default kernel.

Runs the force-included default fp16/rcr kernel over the parity shapes and
reports median/min/max kernel-exec time and TFLOPS. Median over >=N_RUNS calls;
time_ms is the dispatcher's measured kernel-exec time (host<->device copies are
excluded from the reported figure by the C++ ABI).
"""
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "dispatcher" / "python"))

import numpy as np  # noqa: E402
from gemm_utils import GpuGemmRunner, GemmProblem  # noqa: E402

SO = (_ROOT / "dispatcher" / "build" / "examples"
      / "libgemm_fp16_rcr_compv4_cshuffle_intrawave_True_True_True_False_128x128x32_2x2x1_32x32x16.so")

SHAPES = [
    ("square baseline", 1024, 1024, 1024),
    ("large square", 2048, 2048, 2048),
    ("non-square", 1536, 2048, 512),
    ("awkward M", 257, 1024, 512),
]
N_RUNS = 12
WARMUP = 3


def main():
    runner = GpuGemmRunner(lib_path=str(SO))
    print(f"Kernel: {SO.name}")
    print(f"Runs/shape: {N_RUNS} (+{WARMUP} warmup), reporting median\n")
    hdr = f"{'shape':<18}{'M':>6}{'N':>6}{'K':>6}{'med_ms':>10}{'min_ms':>10}{'max_ms':>10}{'med_TFLOPS':>12}{'cv%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, M, N, K in SHAPES:
        prob = GemmProblem(M, N, K)
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(K, N) * 0.1).astype(np.float16)
        for _ in range(WARMUP):
            runner.run(A, B, prob)
        times = []
        ok = True
        for _ in range(N_RUNS):
            r = runner.run(A, B, prob)
            if not r.success or r.time_ms <= 0:
                ok = False
                break
            times.append(r.time_ms)
        if not ok or not times:
            print(f"{label:<18}{M:>6}{N:>6}{K:>6}   FAILED (status nonzero)")
            continue
        med = statistics.median(times)
        cv = (statistics.pstdev(times) / med * 100) if med > 0 else 0.0
        tflops = (prob.flops / (med * 1e-3)) / 1e12
        print(f"{label:<18}{M:>6}{N:>6}{K:>6}{med:>10.3f}{min(times):>10.3f}"
              f"{max(times):>10.3f}{tflops:>12.1f}{cv:>7.1f}")
    print("\nPERF MEDIANS DONE")


if __name__ == "__main__":
    main()
