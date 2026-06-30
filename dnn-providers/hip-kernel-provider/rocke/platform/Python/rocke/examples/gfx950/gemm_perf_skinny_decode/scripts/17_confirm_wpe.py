#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 17: high-confidence confirmation of waves_per_eu=4 win.

Step 15 showed waves_per_eu=4 nudged the kernel from 10.51µs to 10.44µs
(1.00x rocBLAS instead of 1.01x). At this gap (0.4% below the noise
floor of normal runs) we need more attempts to know whether the
difference is real.

Run 20 attempts × 200 iters each for: baseline (no wpe hint), wpe=2,
wpe=4, wpe=8. Report best, median, p25, p75, and a paired comparison.
"""

from __future__ import annotations
import json
import struct
import sys
from pathlib import Path
from statistics import mean, stdev

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
from rocke.runtime.hip_module import Runtime  # noqa: E402

DEV = "cuda"
M, N, K = 2, 4096, 4096
WARMUP, TIMED, ATTEMPTS = 30, 500, 20
ROCBLAS_MS = 0.0104


def build(label, *, waves_per_eu):
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
        waves_per_eu=waves_per_eu,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_confirm_wpe" / label
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


def bench(art, manifest, label):
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
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
    bm = int(manifest["block_m"])
    bn = int(manifest["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)

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
        "p25": samples[len(samples) // 4],
        "median": samples[len(samples) // 2],
        "p75": samples[3 * len(samples) // 4],
        "mean": mean(samples),
        "stdev": stdev(samples),
    }


def main():
    results = []
    for wpe in [None, 2, 4, 8]:
        label = f"wpe{wpe}"
        print(f"\n=== {label} ===")
        art, mf = build(label, waves_per_eu=wpe)
        r = bench(art, mf, label)
        results.append(r)
        print(
            f"  best={r['best'] * 1000:.3f}µs  p25={r['p25'] * 1000:.3f}  med={r['median'] * 1000:.3f}  "
            f"p75={r['p75'] * 1000:.3f}  stdev={r['stdev'] * 1000:.4f}  "
            f"vs_rocBLAS={r['best'] / ROCBLAS_MS:.3f}x"
        )

    out = ROOT / "data" / "17_confirm_wpe.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    results.sort(key=lambda r: r["best"])
    print("\nRanked by best:")
    for r in results:
        print(
            f"  {r['label']:<8} best={r['best'] * 1000:.3f}µs  med={r['median'] * 1000:.3f}µs  "
            f"vs_rocBLAS={r['best'] / ROCBLAS_MS:.3f}x"
        )


if __name__ == "__main__":
    main()
