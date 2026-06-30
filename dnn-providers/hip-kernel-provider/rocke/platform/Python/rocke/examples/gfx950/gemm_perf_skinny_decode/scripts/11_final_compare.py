#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 11: final side-by-side under one harness.

After steps 6–10 we have a much stronger DSL kernel:
  baseline (sweep winner, step 02):   48.3 µs    8.7% HBM
  ground-truth-inspired (step 07):    13.5 µs   31.0% HBM    (3.6x speedup)
  + preshuffle_b (step 09):           13.3 µs   31.5% HBM
  + lever combos (step 08):           13.5 µs   31.2% HBM    (noise)
  + hipcc backend (step 10):          13.4 µs   31.4% HBM    (noise)

rocBLAS reference:                    11.0 µs   38.1% HBM

This step re-runs the new DSL winner *and* rocBLAS under the same HIP-
event timing the original step 04 used, so the ratio is honest. The
old step 04 ratio was 4.38× slower. We expect roughly 1.2× now.
"""

from __future__ import annotations
import ctypes
import json
import struct
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5])
)  # .../rocke/.. = python root

import torch  # noqa: E402

from rocke.instances.common.gemm_universal import (  # noqa: E402
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402

assert torch.cuda.is_available(), "ROCm-torch required"
DEV = "cuda"

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
PEAK_BF16_TF = 2500.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5

# Winner: t16x16x512 mem interwave cshuffle (from steps 07+08)
WINNER_TILE = dict(
    tile_m=16,
    tile_n=16,
    tile_k=512,
    warp_m=1,
    warp_n=1,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)


def build_winner():
    tile = TileSpec(**WINNER_TILE)
    trait = TraitSpec(
        pipeline="mem",
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    salted = f"final_winner__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_final" / salted
    sub.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=tile.tile_m,
        block_n=tile.tile_n,
        block_k=tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(M, N, K),
        warmup_iters=WARMUP,
        timed_iters=TIMED,
        atoms=["mfma_f32_16x16x32_bf16"],
    )
    write_artifact(art, sub, manifest)
    return art, sub, spec, manifest


def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def time_dsl(art, spec, manifest, A_torch, B_natural_torch):
    """Time DSL kernel using torch.cuda.Event for fair comparison with rocBLAS."""
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    # Use torch tensors directly — same memory rocBLAS sees
    A_ptr = int(A_torch.data_ptr())
    B_ptr = int(B_natural_torch.data_ptr())
    C = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    C_ptr = int(C.data_ptr())
    args = struct.pack("<QQQiii", A_ptr, B_ptr, C_ptr, M, N, K)
    bm = int(manifest["block_m"])
    bn = int(manifest["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)

    samples_ms = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            rt.launch(fn, grid, block, args, stream=0)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            rt.launch(fn, grid, block, args, stream=0)
        end.record()
        torch.cuda.synchronize()
        ms_per = start.elapsed_time(end) / TIMED
        if attempt == 0:
            continue
        samples_ms.append(ms_per)
    return samples_ms, C


def time_rocblas(A_torch, W_torch, out):
    samples_ms = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            torch.matmul(A_torch, W_torch.t(), out=out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            torch.matmul(A_torch, W_torch.t(), out=out)
        end.record()
        torch.cuda.synchronize()
        ms_per = start.elapsed_time(end) / TIMED
        if attempt == 0:
            continue
        samples_ms.append(ms_per)
    return samples_ms


def main():
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    W = torch.randn(
        N, K, dtype=torch.bfloat16, device=DEV
    )  # rocBLAS: W [N,K]; DSL sees same B
    out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)

    print("Building DSL winner: t16x16x512 mem interwave cshuffle")
    art, sub, spec, manifest = build_winner()
    print(f"  kernel: {art.kernel_name}")
    print(f"  block_size: {spec.block_size}")

    print("\n--- DSL t16x16x512 (HIP events, same harness as 04) ---")
    dsl_samples, C_out = time_dsl(art, spec, manifest, A, W)
    dsl_med = median(dsl_samples)
    dsl_best = min(dsl_samples)
    dsl_spread = (max(dsl_samples) - min(dsl_samples)) / dsl_med * 100.0
    flop = 2.0 * M * N * K
    bytes_xfer = 2 * (M * K + N * K + M * N)
    tflops_dsl = flop / (dsl_best * 1e-3) / 1e12
    gbs_dsl = bytes_xfer / (dsl_best * 1e-3) / 1e9
    print(f"  median={dsl_med:.4f}ms  spread={dsl_spread:.1f}%  best={dsl_best:.4f}ms")
    print(f"  → {tflops_dsl:.2f} TF  ({tflops_dsl / PEAK_BF16_TF * 100:.1f}% peak)")
    print(f"  → {gbs_dsl:.0f} GB/s  ({gbs_dsl / HBM_PEAK_GBS * 100:.1f}% HBM)")

    print("\n--- rocBLAS bf16 (torch.matmul, same harness) ---")
    rb_samples = time_rocblas(A, W, out)
    rb_med = median(rb_samples)
    rb_best = min(rb_samples)
    rb_spread = (max(rb_samples) - min(rb_samples)) / rb_med * 100.0
    tflops_rb = flop / (rb_best * 1e-3) / 1e12
    gbs_rb = bytes_xfer / (rb_best * 1e-3) / 1e9
    print(f"  median={rb_med:.4f}ms  spread={rb_spread:.1f}%  best={rb_best:.4f}ms")
    print(f"  → {tflops_rb:.2f} TF  ({tflops_rb / PEAK_BF16_TF * 100:.1f}% peak)")
    print(f"  → {gbs_rb:.0f} GB/s  ({gbs_rb / HBM_PEAK_GBS * 100:.1f}% HBM)")

    # Correctness vs rocBLAS itself (looser tolerance: both kernels round bf16)
    diff = (C_out.float() - out.float()).abs()
    print(
        f"\n  DSL vs rocBLAS max abs diff = {float(diff.max()):.4f}  (both bf16-rounded outputs)"
    )

    ratio = dsl_best / rb_best
    print(
        f"\nDSL / rocBLAS latency ratio: {ratio:.2f}×  "
        f"({'DSL faster' if ratio < 1 else 'rocBLAS faster'})"
    )
    print(
        f"Speedup vs original step-02 winner (48.3µs): {48.3 / (dsl_best * 1000):.2f}×"
    )

    out_json = ROOT / "data" / "11_final_compare.json"
    with out_json.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "warmup": WARMUP,
                "timed": TIMED,
                "attempts": ATTEMPTS,
                "dsl_winner": {
                    "tile": WINNER_TILE,
                    "trait": {
                        "pipeline": "mem",
                        "scheduler": "interwave",
                        "epilogue": "cshuffle",
                    },
                    "ms_median": dsl_med,
                    "ms_best": dsl_best,
                    "ms_spread_pct": dsl_spread,
                    "tflops_best": tflops_dsl,
                    "gbs_best": gbs_dsl,
                    "pct_hbm": gbs_dsl / HBM_PEAK_GBS * 100.0,
                },
                "rocblas": {
                    "ms_median": rb_med,
                    "ms_best": rb_best,
                    "ms_spread_pct": rb_spread,
                    "tflops_best": tflops_rb,
                    "gbs_best": gbs_rb,
                    "pct_hbm": gbs_rb / HBM_PEAK_GBS * 100.0,
                },
                "dsl_over_rocblas_latency_ratio": ratio,
                "speedup_vs_step02_winner": 48.3 / (dsl_best * 1000),
                "max_abs_diff_dsl_vs_rocblas": float(diff.max()),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
