#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 15: warp geometry + waves_per_eu on top of the DTLA winner.

Step 13/14 settled on t16x16x1024 + DTLA + CACHE_ALL/CACHE_ALL @ 10.51µs.
That config uses warp_m=warp_n=warp_k=1 -> block_size=64 -> 1 wave/WG.
The kernel hits 40% HBM, which means it's latency-bound on global loads,
not bandwidth-bound. More waves per WG would let the scheduler hide
latency by switching between waves while one is waiting for memory.

Sweep:
  warp_n ∈ {1, 2, 4}    (more waves = more parallel MFMA columns)
  warp_k ∈ {1, 2}       (split K within WG)
  waves_per_eu ∈ {None, 2, 4}

Cap: total waves = warp_m*warp_n*warp_k, block_size = waves * 64.
Hardware caps: block_size <= 1024. Tile geometry must stay 16x16x1024.
Note: changing warps reshapes the MFMA tile space, so we keep warp_tile
fixed and the warps just multiply how many MFMAs each wave issues.
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

DEV = "cuda"
M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5
ROCBLAS_MS = 0.0104


def build(label, *, warp_m, warp_n, warp_k, waves_per_eu, pipeline="mem"):
    tile = TileSpec(
        tile_m=16,
        tile_n=16,
        tile_k=1024,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline=pipeline,
        scheduler="interwave" if pipeline == "mem" else "intrawave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        direct_to_lds=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        waves_per_eu=waves_per_eu,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_warp_geom" / label
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


def correct_and_bench(art, manifest):
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
    if max_abs > 1.0:
        return {"error": f"max_abs={max_abs}"}

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
    ms_best = min(samples)
    ms_med = median(samples)
    bytes_xfer = 2 * (M * K + N * K + M * N)
    gbs = bytes_xfer / (ms_best * 1e-3) / 1e9
    return {
        "ms_best": ms_best,
        "ms_median": ms_med,
        "pct_hbm": gbs / HBM_PEAK_GBS * 100.0,
        "vs_rocblas": ms_best / ROCBLAS_MS,
        "max_abs": max_abs,
    }


def main():
    results = []
    configs = []
    # warp shapes; for tile_m/tile_n = 16 = warp_tile_m/n, warp_m and warp_n
    # *should* stay at 1 (each warp covers the full output tile). But we can
    # still split K across warps via warp_k.
    for warp_k in [1, 2, 4]:
        for wpe in [None, 2, 4, 8]:
            configs.append(dict(warp_m=1, warp_n=1, warp_k=warp_k, waves_per_eu=wpe))

    for cfg in configs:
        bs = cfg["warp_m"] * cfg["warp_n"] * cfg["warp_k"] * 64
        wpe = cfg["waves_per_eu"]
        label = f"w{cfg['warp_m']}_{cfg['warp_n']}_{cfg['warp_k']}_bs{bs}_wpe{wpe}"
        print(f"\n=== {label} ===")
        try:
            art, mf = build(label, **cfg)
        except Exception as e:
            print(f"  COMPILE-FAIL: {type(e).__name__}: {str(e)[:150]}")
            results.append({"label": label, "error": f"compile: {e}", **cfg})
            continue
        try:
            r = correct_and_bench(art, mf)
            r.update({"label": label, **cfg, "block_size": bs})
            results.append(r)
            if "ms_best" in r:
                print(
                    f"  {r['ms_best'] * 1000:.2f}µs  ({r['pct_hbm']:.1f}% HBM)  "
                    f"{r['vs_rocblas']:.2f}x rocBLAS"
                )
            else:
                print(f"  ERR {r.get('error')}")
        except Exception as e:
            print(f"  BENCH-FAIL: {type(e).__name__}: {str(e)[:150]}")
            results.append({"label": label, "error": f"bench: {e}", **cfg})

    out = ROOT / "data" / "15_warp_geom.json"
    with out.open("w") as f:
        json.dump({"shape": {"M": M, "N": N, "K": K}, "results": results}, f, indent=2)
    print(f"\nWrote {out}")

    ok = [r for r in results if "ms_best" in r]
    ok.sort(key=lambda r: r["ms_best"])
    print(f"\nTop 5 of {len(ok)}:")
    for r in ok[:5]:
        print(
            f"  {r['label']:<40} {r['ms_best'] * 1000:.2f}µs  ({r['pct_hbm']:.1f}% HBM)  {r['vs_rocblas']:.2f}x"
        )


if __name__ == "__main__":
    main()
