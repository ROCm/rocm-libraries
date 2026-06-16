#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 22: high-confidence confirmation that chiplet_swizzle wgm=8
chunk=16 actually exceeds rocBLAS.

20 attempts x 1000 iters each, paired against baseline (no chiplet) and
against a direct rocBLAS torch.matmul.
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
WARMUP, TIMED, ATTEMPTS = 50, 1000, 20


def build(label, *, chiplet):
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
        chiplet_swizzle=chiplet,
        chiplet_wgm=8,
        chiplet_num_xcds=8,
        chiplet_chunk_size=16,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_confirm_winner" / label
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


def bench_kernel(art, manifest, label):
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
    return samples


def bench_rocblas():
    A_t = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    W_t = torch.randn(N, K, dtype=torch.bfloat16, device=DEV).t()  # K,N
    samples = []
    for attempt in range(ATTEMPTS + 1):
        for _ in range(WARMUP):
            torch.matmul(A_t, W_t)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(TIMED):
            torch.matmul(A_t, W_t)
        e.record()
        torch.cuda.synchronize()
        ms_per = s.elapsed_time(e) / TIMED
        if attempt == 0:
            continue
        samples.append(ms_per)
    samples.sort()
    return samples


def summary(label, s):
    return {
        "label": label,
        "best": s[0],
        "p10": s[len(s) // 10],
        "median": s[len(s) // 2],
        "p90": s[len(s) * 9 // 10],
        "worst": s[-1],
        "mean": mean(s),
        "stdev": stdev(s),
        "n": len(s),
    }


def main():
    print("Building chiplet winner...")
    art_chip, mf_chip = build("chip_w8_ck16", chiplet=True)
    print("Building baseline (no chiplet)...")
    art_base, mf_base = build("baseline", chiplet=False)

    # Interleave to share cache state between runs.
    out = {}
    for round in range(3):
        print(f"\n--- round {round + 1}/3 ---")
        out.setdefault("chiplet", []).extend(bench_kernel(art_chip, mf_chip, "chiplet"))
        out.setdefault("baseline", []).extend(
            bench_kernel(art_base, mf_base, "baseline")
        )
        out.setdefault("rocblas", []).extend(bench_rocblas())

    results = [summary(k, sorted(v)) for k, v in out.items()]
    for r in results:
        print(
            f"\n{r['label']:<10}  best={r['best'] * 1000:.3f}µs  p10={r['p10'] * 1000:.3f}  "
            f"med={r['median'] * 1000:.3f}  p90={r['p90'] * 1000:.3f}  std={r['stdev'] * 1000:.4f}  "
            f"n={r['n']}"
        )

    chip = next(r for r in results if r["label"] == "chiplet")
    base = next(r for r in results if r["label"] == "baseline")
    roc = next(r for r in results if r["label"] == "rocblas")
    print(
        f"\nchiplet/rocblas  best={chip['best'] / roc['best']:.4f}x  med={chip['median'] / roc['median']:.4f}x"
    )
    print(
        f"baseline/rocblas best={base['best'] / roc['best']:.4f}x  med={base['median'] / roc['median']:.4f}x"
    )
    print(
        f"chiplet/baseline best={chip['best'] / base['best']:.4f}x  med={chip['median'] / base['median']:.4f}x"
    )

    out_path = ROOT / "data" / "22_confirm_winner.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
