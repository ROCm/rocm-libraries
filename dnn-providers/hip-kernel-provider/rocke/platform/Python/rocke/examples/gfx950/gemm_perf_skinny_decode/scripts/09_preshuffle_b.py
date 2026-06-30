#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 9: build + correctness + bench the preshuffle_b path.

`run_manifest` doesn't know how to permute B before launch, so we use
the standalone HIP Runtime harness (like 03_correctness.py) with the
host-side permutation from `rocke.helpers.host_preshuffle_layout`.

Layout (per helpers/preshuffle.py):
    canonical    B [N, K]    row-major
    preshuffled  B [k_tiles, n_tiles, block_n, block_k]  contiguous

Loop:
  - build kernel with trait.preshuffle_b=True
  - permute B on host  (one-shot; reused across timed iters)
  - correctness: compare against fp32 reference
  - bench: warmup + N timed launches with HIP events
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
from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402
from rocke.runtime.hip_module import Runtime  # noqa: E402

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5
ROCBLAS_MS = 0.0110

# Try the t16x16 family at multiple tile_k values. preshuffle requires
# N % block_n == 0 and K % block_k == 0.
TARGETS = [
    ("t16x16x256_preB", 16, 16, 256, 1, 1, "mem"),
    ("t16x16x512_preB", 16, 16, 512, 1, 1, "mem"),
    ("t16x16x256_compv4_preB", 16, 16, 256, 1, 1, "compv4"),
    ("t16x16x512_compv4_preB", 16, 16, 512, 1, 1, "compv4"),
]

BUILD = ROOT / "build_preB"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)


def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def build_kernel(label, tm, tn, tk, wm, wn, pipe):
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline=pipe,
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        preshuffle_b=True,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    salted = f"{label}__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = BUILD / salted
    sub.mkdir(exist_ok=True)
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
    return sub, art.kernel_name, spec.block_size, tile.tile_n, tile.tile_m, tile.tile_k


def permute_b(B_natural_u16: np.ndarray, block_n: int, block_k: int) -> np.ndarray:
    """Permute [N, K] (row-major) B into [k_tiles, n_tiles, block_n, block_k] packed."""
    n_tiles = N // block_n
    k_tiles = K // block_k
    # B is shape (N, K). Reshape into (n_tiles, block_n, k_tiles, block_k),
    # then transpose to (k_tiles, n_tiles, block_n, block_k), then contiguous.
    B = B_natural_u16.reshape(n_tiles, block_n, k_tiles, block_k)
    B = np.transpose(B, (2, 0, 1, 3))
    return np.ascontiguousarray(B)


def run_one(label, tm, tn, tk, wm, wn, pipe):
    print(f"\n=== {label} ===")
    sub, kname, block_size, bn, bm, bk = build_kernel(label, tm, tn, tk, wm, wn, pipe)
    manifest = json.loads((sub / "manifest.json").read_text())
    hsaco_path = sub / manifest["hsaco"]

    # --- correctness ---
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
    ref_f32 = A_bf32 @ B_bf32.T
    ref_bf32 = torch.from_numpy(ref_f32).to(torch.bfloat16).to(torch.float32).numpy()

    B_pre = permute_b(B_u16, bn, bk)
    assert B_pre.nbytes == B_u16.nbytes

    rt = Runtime()
    mod = rt.load_module(hsaco_path.read_bytes())
    fn = mod.get_function(manifest["kernel_name"])
    A_dev = rt.alloc(A_u16.nbytes)
    B_dev = rt.alloc(B_pre.nbytes)
    C_dev = rt.alloc(M * N * 2)
    rt.memcpy_h2d(A_dev, _as_u8(A_u16), A_u16.nbytes)
    rt.memcpy_h2d(B_dev, _as_u8(B_pre), B_pre.nbytes)
    rt.memset(C_dev, 0, M * N * 2)

    args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (block_size, 1, 1)
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
    diff = np.abs(C_out - ref_bf32)
    max_abs = float(diff.max())
    print(
        f"  correctness: max|out-ref|={max_abs:.4f}  ref_max={float(np.abs(ref_bf32).max()):.0f}"
    )
    if max_abs > 1.0:
        print("  → FAIL, skipping bench")
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(C_dev)
        return {"label": label, "error": f"correctness FAIL max_abs={max_abs}"}

    # --- bench (HIP events via subprocess to torch, or hand-rolled) ---
    # Simplest: re-launch many times and use Python wall clock; for small kernels
    # this is noisy, so we do warmup + per-attempt timed loop, take min.
    samples_ms = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            rt.launch(fn, grid, block, args, stream=0)
        rt.stream_sync(0)
        t0 = time.perf_counter()
        for _ in range(TIMED):
            rt.launch(fn, grid, block, args, stream=0)
        rt.stream_sync(0)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3 / TIMED
        if attempt == 0:
            continue  # cold
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
    gap = ms_best / ROCBLAS_MS
    print(
        f"  bench: median={ms_med:.4f}ms  best={ms_best:.4f}ms  "
        f"{tflops:.2f}TF  {gbs:.0f}GB/s  ({gbs / HBM_PEAK_GBS * 100:.1f}% HBM)  vs_rocBLAS={gap:.2f}x"
    )
    return {
        "label": label,
        "ms_best": ms_best,
        "ms_median": ms_med,
        "tflops_best": tflops,
        "gbs_best": gbs,
        "pct_hbm": gbs / HBM_PEAK_GBS * 100.0,
        "vs_rocblas_ratio": gap,
        "max_abs_diff": max_abs,
    }


def main():
    results = []
    for exp in TARGETS:
        try:
            r = run_one(*exp)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            r = {"label": exp[0], "error": f"{type(e).__name__}: {e}"}
        results.append(r)

    out = DATA / "09_preshuffle_b.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
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
            f"\n*** Best preB: {w['label']}  {w['ms_best'] * 1000:.2f}µs  "
            f"({w['pct_hbm']:.1f}% HBM)  {w['vs_rocblas_ratio']:.2f}x vs rocBLAS ***"
        )


if __name__ == "__main__":
    main()
