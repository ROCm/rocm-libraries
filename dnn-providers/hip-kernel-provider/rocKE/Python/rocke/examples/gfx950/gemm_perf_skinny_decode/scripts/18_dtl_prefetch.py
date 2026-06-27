#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 18: ping-pong DTLA prefetch.

Hypothesis: the simple kloop drains vmcnt to 0 between K-tiles. With
two LDS halves, we can issue next-tile DTLA loads before the current
tile's MFMAs and only s_waitcnt vmcnt(loads_per_tile) — overlapping
HBM latency with MFMA.

Cost: 2x LDS (drops 2 WGs/CU to 1 WG/CU at t16x16x1024 — already 1
WG/CU under DTLA anyway, so no extra occupancy loss).

Compare against the 10.51µs DTLA-only baseline and rocBLAS 10.4µs.
"""

from __future__ import annotations
import ctypes
import json
import struct
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5])
)  # .../rocke/.. = python root

import numpy as np  # noqa: E402
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
from rocke.runtime.hip_module import Runtime  # noqa: E402

DEV = "cuda"
M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 30, 500, 10
ROCBLAS_MS = 0.0104


def build(label, *, prefetch, waves_per_eu=None):
    tile = TileSpec(
        tile_m=16,
        tile_n=16,
        tile_k=1024,
        warp_m=1,
        warp_n=1,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline="mem",
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        direct_to_lds=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        dtl_prefetch=prefetch,
        waves_per_eu=waves_per_eu,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_dtl_prefetch" / label
    sub.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=16,
        block_n=16,
        block_k=1024,
        threads_per_block=spec.block_size,
        default_shape=(M, N, K),
        warmup_iters=WARMUP,
        timed_iters=TIMED,
        atoms=["mfma_f32_16x16x32_bf16"],
    )
    write_artifact(art, sub, manifest)
    return art, manifest


def correct_and_bench(art, manifest, label):
    rng = np.random.default_rng(0xC0FFEE)
    A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
    B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)
    A_u16 = (
        torch.from_numpy(A_f32)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    B_u16 = (
        torch.from_numpy(B_f32)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    A_bf32 = (
        torch.from_numpy(A_u16.view(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )
    B_bf32 = (
        torch.from_numpy(B_u16.view(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )
    ref = (
        torch.from_numpy(A_bf32 @ B_bf32.T).to(torch.bfloat16).to(torch.float32).numpy()
    )

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A_dev = rt.alloc(A_u16.nbytes)
    B_dev = rt.alloc(B_u16.nbytes)
    C_dev = rt.alloc(M * N * 2)
    rt.memcpy_h2d(
        A_dev, (ctypes.c_uint8 * A_u16.nbytes).from_buffer(A_u16), A_u16.nbytes
    )
    rt.memcpy_h2d(
        B_dev, (ctypes.c_uint8 * B_u16.nbytes).from_buffer(B_u16), B_u16.nbytes
    )
    rt.memset(C_dev, 0, M * N * 2)
    args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
    bm = int(manifest["block_m"])
    bn = int(manifest["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)
    rt.launch(fn, grid, block, args, stream=0)
    rt.stream_sync(0)
    out_buf = (ctypes.c_uint8 * (M * N * 2))()
    rt.memcpy_d2h(out_buf, C_dev, M * N * 2)
    C_out = (
        torch.from_numpy(np.frombuffer(bytes(out_buf), dtype=np.int16).copy())
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
        .reshape(M, N)
    )
    max_abs = float(np.abs(C_out - ref).max())
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(C_dev)
    print(f"  correctness max_abs={max_abs}")
    if max_abs > 1.0:
        return {"error": f"max_abs={max_abs}", "label": label}

    A_t = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    W_t = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
    C_t = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    args = struct.pack(
        "<QQQiii",
        int(A_t.data_ptr()),
        int(W_t.data_ptr()),
        int(C_t.data_ptr()),
        M,
        N,
        K,
    )
    samples = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            rt.launch(fn, grid, block, args, stream=0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(TIMED):
            rt.launch(fn, grid, block, args, stream=0)
        e.record()
        torch.cuda.synchronize()
        ms_per = s.elapsed_time(e) / TIMED
        if attempt == 0:
            continue
        samples.append(ms_per)
    samples.sort()
    return {
        "label": label,
        "best": samples[0],
        "median": samples[len(samples) // 2],
        "mean": mean(samples),
        "stdev": stdev(samples),
        "max_abs": max_abs,
        "vs_rocblas": samples[0] / ROCBLAS_MS,
    }


def main():
    results = []
    for prefetch in [False, True]:
        for wpe in [None, 4]:
            label = f"pref{int(prefetch)}_wpe{wpe}"
            print(f"\n=== {label} ===")
            try:
                art, mf = build(label, prefetch=prefetch, waves_per_eu=wpe)
            except Exception as e:
                print(f"  COMPILE-FAIL: {type(e).__name__}: {str(e)[:300]}")
                results.append({"label": label, "error": f"compile: {e}"})
                continue
            try:
                r = correct_and_bench(art, mf, label)
                results.append(r)
                if "best" in r:
                    print(
                        f"  best={r['best'] * 1000:.3f}µs  med={r['median'] * 1000:.3f}  "
                        f"std={r['stdev'] * 1000:.4f}  vs_rocBLAS={r['vs_rocblas']:.3f}x"
                    )
            except Exception as e:
                print(f"  BENCH-FAIL: {type(e).__name__}: {str(e)[:300]}")
                results.append({"label": label, "error": f"bench: {e}"})

    out = ROOT / "data" / "18_dtl_prefetch.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    ok = [r for r in results if "best" in r]
    ok.sort(key=lambda r: r["best"])
    print("\nRanked:")
    for r in ok:
        print(
            f"  {r['label']:<25} best={r['best'] * 1000:.3f}µs  med={r['median'] * 1000:.3f}µs  "
            f"vs_rocBLAS={r['vs_rocblas']:.3f}x"
        )


if __name__ == "__main__":
    main()
