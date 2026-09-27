#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Skinny BF16 GEMM — QKV and O projections (M=2, decode batch).

Problem
-------
ATOM's LinearBase.forward() calls tgemm.mm → hipBLASLt for all
unquantized (BF16) GEMMs.  hipBLASLt delegates M=2 decode shapes to
Tensile's skinny-gemm path (``wv_splitk_small_fp16_bf16``), which
achieves 10–12% of peak HBM bandwidth.

The rocke universal_gemm kernel reaches 19–22% HBM bandwidth on the
same shapes by using:

  1. Direct-To-LDS (DTLA) for matrix A — bypasses L2 and writes the
     tile straight to LDS, saving one memory hop.
  2. Tile-K = 512 or 1024 — the K-loop unrolls deeply enough to hide
     the HBM fetch latency for both the A and B tiles.
  3. Chiplet swizzle — distributes CTAs across the 8 XCDs so every
     XCD's L2 sees a different tile; prevents all CTAs from hammering
     the same cache set on a single XCD.

Optimization walk-through
-------------------------
The winning configuration for A3B decode shapes was found by the
``gemm_perf_skinny_decode`` runbook (scripts 01 → 22 in
``examples/gemm_perf_skinny_decode/``).  Key steps:

  Step 01  Occupancy probe: count VGPR/AGPR/LDS for candidate tiles.
           Confirms tile_m=16, tile_n=16, tile_k=512 fits without spill.
  Step 07  tile_k sweep (64 → 512 → 1024): tile_k=512 wins for N≤2560,
           tile_k=1024 wins for the square N=K=2048 shape.
  Step 12  DTLA flag: add direct_to_lds=True — shaves 0.4µs from QKV.
  Step 17  Chiplet swizzle: chiplet_wgm=4, chunk_size=64 — shaves another
           0.3µs by balancing XCD load.
  Step 22  Final confirmation vs rocBLAS with apples-to-apples timer.

Results (MI355X / gfx950, batch=2, bf16)
-----------------------------------------
  Shape               hipBLASLt   DSL      speedup
  QKV M=2 N=2560 K=2048  11.1µs   6.8µs   1.63×
  O   M=2 N=2048 K=2048  11.1µs   6.6µs   1.68×

Run
---
  PYTHONPATH=<rocke_python_root> python3 01_gemm_skinny.py
"""

from __future__ import annotations


import torch

from _common import (
    BATCH,
    DTYPE,
    HIDDEN,
    ISA,
    NHEAD_K,
    NHEAD_Q,
    HEAD_DIM,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
    build_gemm_kernel,
)


def _bench_shape(
    M: int,
    N: int,
    K: int,
    tile_k: int,
    chiplet_wgm: int,
    chiplet_chunk: int,
    label: str,
) -> None:
    print(f"\n  [{label}]  M={M} N={N} K={K}  tile_k={tile_k}")

    A = torch.randn(M, K, dtype=DTYPE, device="cuda") * 0.1
    B_nt = torch.randn(K, N, dtype=DTYPE, device="cuda") * 0.1  # col-major for matmul
    B_t = torch.randn(N, K, dtype=DTYPE, device="cuda") * 0.1  # row-major for DSL RCR

    # Baseline: hipBLASLt via torch.matmul (same path as ATOM LinearBase)
    bl_ms = ms(
        lambda: torch.matmul(A, B_nt), warmup=WARMUP, iters=ITERS, repeats=REPEATS
    )
    print(f"    hipBLASLt (torch.matmul): {bl_ms * 1000:.2f}µs")

    # DSL: universal_gemm RCR layout (A row-major, B row-major, C row-major)
    run, rt, C = build_gemm_kernel(
        M,
        N,
        K,
        tile_k=tile_k,
        chiplet_wgm=chiplet_wgm,
        chiplet_chunk_size=chiplet_chunk,
    )
    Ap, Bp, Cp = int(A.data_ptr()), int(B_t.data_ptr()), int(C.data_ptr())
    dsl_ms = ms(lambda: run(Ap, Bp, Cp), warmup=WARMUP, iters=ITERS, repeats=REPEATS)
    spd = speedup(bl_ms, dsl_ms)
    print(f"    DSL universal_gemm:        {dsl_ms * 1000:.2f}µs  speedup={spd:.3f}×")

    # Quick correctness check (RCR: C = A @ B_t.T = A @ B_nt if B_nt = B_t.T)
    ref = torch.matmul(A.float(), B_t.float().T).to(DTYPE)
    err = (C - ref).abs().max().item()
    ok = err < 0.05
    print(f"    Correctness: max|err|={err:.2e}  {'PASS' if ok else 'FAIL'}")


def main() -> None:
    print("=" * 60)
    print("01  Skinny BF16 GEMM — QKV and O projections")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ISA: {ISA}")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()
    print("Optimization: DTLA + tile_k=512/1024 + chiplet_swizzle")
    print("Wins by hiding HBM latency and balancing XCD load.")

    # QKV: M=2, N=(32+2×4)×64=2560, K=2048
    _bench_shape(
        BATCH,
        (NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM,
        HIDDEN,
        tile_k=512,
        chiplet_wgm=4,
        chiplet_chunk=64,
        label="QKV proj",
    )

    # O proj: M=2, N=2048, K=2048
    _bench_shape(
        BATCH,
        HIDDEN,
        HIDDEN,
        tile_k=1024,
        chiplet_wgm=8,
        chiplet_chunk=64,
        label="O proj",
    )


if __name__ == "__main__":
    main()
