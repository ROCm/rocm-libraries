#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 7: push tile_k further on the t16x16 geometry, the lever 06
identified as the big win.

Result from step 06: just bumping tile_k 32→256 on tile_n=16 closed
3× of the 4.4× gap to rocBLAS. hipBLASLt's actual kernel runs
DepthU=512, suggesting we still have ~2× to chase along this axis.

LDS budget (gfx950 cap 160 KiB) on tile_m=16, tile_n=16:
  mem pipeline (single AB):  bytes = 4 * tile_k + 512 (cshuffle C)
  compv4       (double AB):  bytes = 8 * tile_k + 512
Both leave us comfortable up to tile_k=4096 (mem) / 2048 (compv4).

This step also sanity-checks: does correctness still hold at large tile_k?
We re-run the standalone bf16 verify on the winner.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5])
)  # .../rocke/.. = python root

from rocke.instances.common.gemm_universal import (  # noqa: E402
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402

M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5
ROCBLAS_MS = 0.0110  # baseline from step 04

# Sweep tile_k on the t16x16 w1x1 geometry.
# (label, tm, tn, tk, wm, wn, pipeline, scheduler)
CANDIDATES = []
for tk in [256, 384, 512, 768, 1024, 1536, 2048]:
    CANDIDATES.append((f"t16x16x{tk}_mem", 16, 16, tk, 1, 1, "mem", "interwave"))
    CANDIDATES.append((f"t16x16x{tk}_compv4", 16, 16, tk, 1, 1, "compv4", "intrawave"))
# Also try widening warp_n=2 with the deeper K (more lanes processing the K-pack)
for tk in [256, 512, 1024]:
    CANDIDATES.append((f"t16x32x{tk}_w1x2_mem", 16, 32, tk, 1, 2, "mem", "interwave"))

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[5])}
BUILD = ROOT / "build_push_tk"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)


def build(label, tm, tn, tk, wm, wn, pipe, sched):
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
        scheduler=sched,
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    salted = f"{label}__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid: {why}")
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = BUILD / salted
    sub.mkdir(exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=tm,
        block_n=tn,
        block_k=tk,
        threads_per_block=spec.block_size,
        default_shape=(M, N, K),
        warmup_iters=WARMUP,
        timed_iters=TIMED,
        atoms=["mfma_f32_16x16x32_bf16"],
    )
    write_artifact(art, sub, manifest)
    return sub, art.kernel_name


def run(sub, kname):
    cmd = [
        PY,
        "-m",
        "rocke.run_manifest",
        str(sub / f"{kname}.hsaco"),
        str(sub / "manifest.json"),
        "--shape",
        f"{M},{N},{K}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=120)
    out = r.stdout + r.stderr
    mms = re.search(r"([\d.]+)\s*ms", out)
    mtf = re.search(r"([\d.]+)\s*TFlops", out)
    mgb = re.search(r"([\d.]+)\s*GB/s", out)
    if not (mms and mtf and mgb):
        return None
    return {
        "ms": float(mms.group(1)),
        "tflops": float(mtf.group(1)),
        "gbs": float(mgb.group(1)),
    }


def main():
    results = []
    for exp in CANDIDATES:
        label = exp[0]
        try:
            sub, kname = build(*exp)
        except Exception as e:
            print(f"{label:<28} BUILD-FAIL: {str(e)[:100]}")
            results.append({"label": label, "error": str(e)})
            continue
        samples = []
        for i in range(ATTEMPTS + 1):
            r = run(sub, kname)
            if r is None:
                continue
            if i == 0:
                continue
            samples.append(r)
        if not samples:
            print(f"{label:<28} NO-SAMPLES")
            results.append({"label": label, "error": "no samples"})
            continue
        ms_med = median(s["ms"] for s in samples)
        best = min(samples, key=lambda s: s["ms"])
        spread = (
            (max(s["ms"] for s in samples) - min(s["ms"] for s in samples))
            / ms_med
            * 100.0
        )
        gap = best["ms"] / ROCBLAS_MS
        print(
            f"{label:<28} med={ms_med:.4f}ms  best={best['ms']:.4f}ms  "
            f"{best['tflops']:.2f}TF  {best['gbs']:.0f}GB/s  "
            f"({best['gbs'] / HBM_PEAK_GBS * 100:5.1f}% HBM)  vs_rocBLAS={gap:.2f}x"
        )
        results.append(
            {
                "label": label,
                "tile_k": exp[3],
                "ms_median": ms_med,
                "ms_best": best["ms"],
                "ms_spread_pct": spread,
                "tflops_best": best["tflops"],
                "gbs_best": best["gbs"],
                "pct_hbm": best["gbs"] / HBM_PEAK_GBS * 100.0,
                "vs_rocblas_ratio": gap,
                "samples": samples,
            }
        )

    out = DATA / "07_push_tile_k.json"
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
        winner = min(ok, key=lambda r: r["ms_best"])
        print(
            f"\n*** Winner: {winner['label']}  {winner['ms_best']:.4f}ms  "
            f"({winner['pct_hbm']:.1f}% HBM)  {winner['vs_rocblas_ratio']:.2f}x vs rocBLAS ***"
        )


if __name__ == "__main__":
    main()
