#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 10: re-compile the t16x16x512 winner through the alternate
HIP-C++ -> hipcc backend instead of the default LLVM-IR -> libamd_comgr.

`rocke.helpers.compile_kernel_via_hipcc` lowers the same IR to HIP C++
and pushes it through `hipcc --genco -O3`. The runtime ABI is identical
(same HSACO arg-pack), but the codegen path is different:

  default (compile_kernel):       IR -> LLVM IR -> libamd_comgr      ~90 ms
  alternate (via_hipcc):          IR -> HIP C++ -> hipcc -O3         ~450 ms

Per the helpers/compile.py docstring, this has historically given
~5% speedup on long-running attention kernels because the clang
frontend does richer scheduling. For a tiny memory-bound GEMM the
expected win is smaller, but worth checking on the winning geometry
since we're already at 1.21x rocBLAS — every percent matters.

Also tries `-O3 -ffast-math -fgpu-flush-denormals-to-zero` plus the
amdclang-specific `-mllvm -amdgpu-early-inline-all=true`.
"""

from __future__ import annotations
import ctypes
import json
import struct
import sys
import time
from pathlib import Path
from statistics import median

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
from rocke.helpers import compile_kernel, make_gemm_manifest  # noqa: E402
from rocke.helpers.compile import compile_kernel_via_hipcc  # noqa: E402
from rocke.runtime.hip_module import Runtime  # noqa: E402

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5
ROCBLAS_MS = 0.0110


# Each entry: (label, build_fn, extra_flags or None)
def _build_llvm(spec):
    return compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")


def _build_hipcc(spec, flags=None):
    return compile_kernel_via_hipcc(
        build_universal_gemm(spec), arch="gfx950", extra_flags=flags or []
    )


VARIANTS = [
    ("llvm_default", lambda s: _build_llvm(s)),
    ("hipcc_O3", lambda s: _build_hipcc(s, [])),
    ("hipcc_O3_ffast", lambda s: _build_hipcc(s, ["-ffast-math"])),
    (
        "hipcc_O3_flushdenorm",
        lambda s: _build_hipcc(s, ["-ffast-math", "-fgpu-flush-denormals-to-zero"]),
    ),
    (
        "hipcc_O3_inline_all",
        lambda s: _build_hipcc(
            s,
            [
                "-ffast-math",
                "-fgpu-flush-denormals-to-zero",
                "-mllvm",
                "-amdgpu-early-inline-all=true",
            ],
        ),
    ),
]

BASE_TILE = dict(
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


def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def bench(art, manifest_dict, label):
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

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A_dev = rt.alloc(A_u16.nbytes)
    B_dev = rt.alloc(B_u16.nbytes)
    C_dev = rt.alloc(M * N * 2)
    rt.memcpy_h2d(A_dev, _as_u8(A_u16), A_u16.nbytes)
    rt.memcpy_h2d(B_dev, _as_u8(B_u16), B_u16.nbytes)
    rt.memset(C_dev, 0, M * N * 2)

    args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
    bm = int(manifest_dict["block_m"])
    bn = int(manifest_dict["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest_dict["threads_per_block"]), 1, 1)

    # correctness check first
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
    max_abs = float(np.abs(C_out - ref).max())
    if max_abs > 1.0:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(C_dev)
        return {"label": label, "error": f"correctness FAIL max_abs={max_abs}"}

    # bench
    samples_ms = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            rt.launch(fn, grid, block, args, stream=0)
        rt.stream_sync(0)
        t0 = time.perf_counter()
        for _ in range(TIMED):
            rt.launch(fn, grid, block, args, stream=0)
        rt.stream_sync(0)
        ms = (time.perf_counter() - t0) * 1e3 / TIMED
        if attempt == 0:
            continue
        samples_ms.append(ms)
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(C_dev)

    ms_med = median(samples_ms)
    ms_best = min(samples_ms)
    flop = 2.0 * M * N * K
    bytes_xfer = 2 * (M * K + N * K + M * N)
    tflops = flop / (ms_best * 1e-3) / 1e12
    gbs = bytes_xfer / (ms_best * 1e-3) / 1e9
    return {
        "label": label,
        "ms_median": ms_med,
        "ms_best": ms_best,
        "tflops_best": tflops,
        "gbs_best": gbs,
        "pct_hbm": gbs / HBM_PEAK_GBS * 100.0,
        "vs_rocblas_ratio": ms_best / ROCBLAS_MS,
        "max_abs_diff": max_abs,
    }


def main():
    tile = TileSpec(**BASE_TILE)
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

    results = []
    for label, builder in VARIANTS:
        print(f"\n=== {label} ===")
        spec = UniversalGemmSpec(
            name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
        )
        try:
            t0 = time.perf_counter()
            art = builder(spec)
            compile_ms = (time.perf_counter() - t0) * 1e3
            print(f"  compile: {compile_ms:.0f}ms")
        except Exception as e:
            print(f"  COMPILE-FAIL: {type(e).__name__}: {str(e)[:200]}")
            results.append({"label": label, "error": f"compile fail: {e}"})
            continue
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
        try:
            r = bench(art, manifest, label)
            if "ms_best" in r:
                print(
                    f"  bench: best={r['ms_best'] * 1000:.3f}µs  ({r['pct_hbm']:.1f}% HBM)  "
                    f"vs_rocBLAS={r['vs_rocblas_ratio']:.3f}x"
                )
            else:
                print(f"  ERROR: {r.get('error')}")
            r["compile_ms"] = compile_ms
            results.append(r)
        except Exception as e:
            print(f"  BENCH-FAIL: {e}")
            results.append({"label": label, "error": f"bench fail: {e}"})

    out = ROOT / "data" / "10_hipcc_backend.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "base_tile": BASE_TILE,
                "rocblas_ms": ROCBLAS_MS,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")

    ok = [r for r in results if "ms_best" in r]
    if ok:
        w = min(ok, key=lambda r: r["ms_best"])
        print(
            f"\n*** Best: {w['label']}  {w['ms_best'] * 1000:.3f}µs  "
            f"({w['pct_hbm']:.1f}% HBM)  {w['vs_rocblas_ratio']:.3f}x vs rocBLAS ***"
        )


if __name__ == "__main__":
    main()
