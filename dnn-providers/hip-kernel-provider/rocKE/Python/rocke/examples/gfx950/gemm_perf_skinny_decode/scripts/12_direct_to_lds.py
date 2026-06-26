#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 12: DirectToLDS (DTLA/DTLB) — the missing lever.

rocprof showed hipBLASLt's MT16x16x512 kernel uses ``DTLA1_DTLB1``:
the global -> LDS load issues one ``buffer_load_dwordx4 ... lds`` per
chunk, writing the dword payload straight into LDS. Our DSL kernel
was emitting the round-trip pattern instead (``global_load_dwordx4 ->
VGPR -> ds_write_b128``), costing 32 extra instructions and 32 extra
VGPRs of pressure per K-tile.

The DSL has the primitive (``async_buffer_load_lds_addr``, used by
``attention_tiled_2d.py``); ``gemm_universal.py`` just didn't wire it.
This step exercises the new ``TraitSpec.direct_to_lds`` flag.
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

assert torch.cuda.is_available()
DEV = "cuda"

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5

TILE = dict(
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


def build(label, *, direct_to_lds):
    tile = TileSpec(**TILE)
    trait = TraitSpec(
        pipeline="mem",
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        direct_to_lds=direct_to_lds,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_dtl" / label
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
    return art, manifest


def correctness(art, manifest):
    """Run kernel against a fp32 reference via HIP Runtime harness."""
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
    rt.memcpy_h2d(A_dev, _as_u8(A_u16), A_u16.nbytes)
    rt.memcpy_h2d(B_dev, _as_u8(B_u16), B_u16.nbytes)
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
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(C_dev)
    return float(np.abs(C_out - ref).max()), float(np.abs(ref).max())


def bench(art, manifest):
    """Time with torch.cuda.Event (same harness as step 11)."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)

    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    args = struct.pack(
        "<QQQiii", int(A.data_ptr()), int(W.data_ptr()), int(C.data_ptr()), M, N, K
    )
    bm = int(manifest["block_m"])
    bn = int(manifest["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)

    samples = []
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
        samples.append(ms_per)
    return samples


def main():
    results = {}
    for label, dtl in [("baseline_no_dtl", False), ("direct_to_lds", True)]:
        print(f"\n=== {label} (direct_to_lds={dtl}) ===")
        try:
            art, manifest = build(label, direct_to_lds=dtl)
            print(f"  kernel: {art.kernel_name}")
        except Exception as e:
            print(f"  COMPILE-FAIL: {type(e).__name__}: {e}")
            results[label] = {"error": f"compile: {e}"}
            continue
        try:
            max_abs, ref_max = correctness(art, manifest)
            print(f"  correctness: max|out-ref|={max_abs:.4f}  ref_max={ref_max:.0f}")
            if max_abs > 1.0:
                results[label] = {"error": f"correctness FAIL max_abs={max_abs}"}
                continue
        except Exception as e:
            print(f"  CORRECTNESS-FAIL: {type(e).__name__}: {e}")
            results[label] = {"error": f"correctness: {e}"}
            continue
        try:
            samples = bench(art, manifest)
            ms_med = median(samples)
            ms_best = min(samples)
            spread = (max(samples) - min(samples)) / ms_med * 100.0
            flop = 2.0 * M * N * K
            bytes_xfer = 2 * (M * K + N * K + M * N)
            tflops = flop / (ms_best * 1e-3) / 1e12
            gbs = bytes_xfer / (ms_best * 1e-3) / 1e9
            print(
                f"  bench: med={ms_med * 1000:.2f}µs  best={ms_best * 1000:.2f}µs  "
                f"spread={spread:.1f}%  {gbs:.0f}GB/s ({gbs / HBM_PEAK_GBS * 100:.1f}% HBM)  "
                f"vs_rocBLAS={ms_best / 0.0104:.2f}x"
            )
            results[label] = {
                "ms_median": ms_med,
                "ms_best": ms_best,
                "ms_spread_pct": spread,
                "tflops_best": tflops,
                "gbs_best": gbs,
                "pct_hbm": gbs / HBM_PEAK_GBS * 100.0,
                "max_abs_diff": max_abs,
            }
        except Exception as e:
            print(f"  BENCH-FAIL: {type(e).__name__}: {e}")
            results[label] = {"error": f"bench: {e}"}

    out = ROOT / "data" / "12_direct_to_lds.json"
    with out.open("w") as f:
        json.dump(
            {"shape": {"M": M, "N": N, "K": K}, "tile": TILE, "results": results},
            f,
            indent=2,
        )
    print(f"\nWrote {out}")

    base = results.get("baseline_no_dtl", {})
    dtl = results.get("direct_to_lds", {})
    if "ms_best" in base and "ms_best" in dtl:
        speedup = base["ms_best"] / dtl["ms_best"]
        print(
            f"\n*** DTL speedup: {speedup:.2f}x  "
            f"({base['ms_best'] * 1000:.2f}µs -> {dtl['ms_best'] * 1000:.2f}µs)  "
            f"HBM: {base['pct_hbm']:.1f}% -> {dtl['pct_hbm']:.1f}% ***"
        )


if __name__ == "__main__":
    main()
