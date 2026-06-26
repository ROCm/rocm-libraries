#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 2: benchmark every candidate that survived 01_probe_occupancy.

Runbook anchors:
- §2.2 Performance Baselines, §2.3 Benchmark Hygiene
- §12.1.P Benchmark hygiene (warmup, attempts, salt symbol)
- §12.2 Sweep Discipline (one family at a time, cache builds, salt with
  spec hash, keep best per shape)

What we measure:
    Per kernel: best of N attempts, each with W warmup + I timed launches.
    Time is HIP-event wall clock from rocke.run_manifest.
    Memory traffic uses the runbook's bytes_xfer = 2*(M*K + N*K + M*N).
    Roofline HBM = 8 TB/s for MI355X HBM3e.

What we DON'T do:
    No --verify (run_manifest verify path allocates fp16 buffers regardless
    of dtype, so it cannot validate bf16 results — see §14.3 "Verification
    included in timing" and a separate correctness pass in 03_correctness.py).
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
)
from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402

# Re-declare the CANDIDATES list inline (kept in sync with 01_probe_occupancy.py)
CANDIDATES = [
    (
        "t16x64x32_w1x2_a16x16x32_mem",
        16,
        64,
        32,
        1,
        2,
        1,
        16,
        16,
        32,
        "mem",
        "interwave",
    ),
    (
        "t16x128x32_w1x2_a16x16x32_mem",
        16,
        128,
        32,
        1,
        2,
        1,
        16,
        16,
        32,
        "mem",
        "interwave",
    ),
    (
        "t16x128x64_w1x2_a16x16x32_mem",
        16,
        128,
        64,
        1,
        2,
        1,
        16,
        16,
        32,
        "mem",
        "interwave",
    ),
    (
        "t16x256x32_w1x4_a16x16x32_mem",
        16,
        256,
        32,
        1,
        4,
        1,
        16,
        16,
        32,
        "mem",
        "interwave",
    ),
    (
        "t32x128x32_w2x2_a16x16x32_mem",
        32,
        128,
        32,
        2,
        2,
        1,
        16,
        16,
        32,
        "mem",
        "interwave",
    ),
    (
        "t16x128x32_w1x2_a16x16x32_compv3",
        16,
        128,
        32,
        1,
        2,
        1,
        16,
        16,
        32,
        "compv3",
        "intrawave",
    ),
    (
        "t16x128x32_w1x2_a16x16x32_compv4",
        16,
        128,
        32,
        1,
        2,
        1,
        16,
        16,
        32,
        "compv4",
        "intrawave",
    ),
    (
        "t16x128x64_w1x2_a16x16x32_compv4",
        16,
        128,
        64,
        1,
        2,
        1,
        16,
        16,
        32,
        "compv4",
        "intrawave",
    ),
]

# Single shape: o_proj decode (target identified by Addendum B / decode trace)
M, N, K = 2, 4096, 4096

# Hygiene knobs from §12.1.P
WARMUP_ITERS = 20
TIMED_ITERS = 200
ATTEMPTS = 5

HBM_PEAK_GBS = 8000.0  # MI355X HBM3e
PEAK_BF16_TF = 2500.0

BUILD = ROOT / "build"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[5])}


def compile_one(cand):
    label, tm, tn, tk, wm, wn, wk, wtm, wtn, wtk, pipe, sched = cand
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=wk,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
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
    # §12.1.P "Salt kernel symbol with shape hash" — append M_N_K to the name
    # so HSACO cache cannot alias two specializations.
    salted = f"{label}__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
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
        warmup_iters=WARMUP_ITERS,
        timed_iters=TIMED_ITERS,
        atoms=[f"mfma_f32_{wtm}x{wtn}x{wtk}_bf16"],
        notes=f"o_proj decode (Qwen3-8B) — runbook lever sweep, {pipe}/{sched}",
    )
    write_artifact(art, sub, manifest)
    return label, sub, art.kernel_name


def run_one(sub, kernel_name):
    cmd = [
        PY,
        "-m",
        "rocke.run_manifest",
        str(sub / f"{kernel_name}.hsaco"),
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
    for cand in CANDIDATES:
        label = cand[0]
        try:
            _, sub, kname = compile_one(cand)
        except Exception as e:
            print(f"{label:<40} COMPILE-FAIL: {e}")
            results.append({"label": label, "error": str(e)})
            continue

        # §12.1.P "Discard first run" (cold cache) and "--attempts ≥ 5"
        samples = []
        for i in range(ATTEMPTS + 1):
            r = run_one(sub, kname)
            if r is None:
                continue
            if i == 0:
                continue  # cold attempt
            samples.append(r)

        if not samples:
            print(f"{label:<40} NO-SAMPLES")
            results.append({"label": label, "error": "no samples"})
            continue

        ms_list = [s["ms"] for s in samples]
        ms_med = median(ms_list)
        spread = (max(ms_list) - min(ms_list)) / ms_med * 100.0
        best = min(samples, key=lambda s: s["ms"])
        print(
            f"{label:<40} med={ms_med:.4f}ms  spread={spread:4.1f}%  "
            f"best={best['ms']:.4f}ms  {best['tflops']:.2f} TF  "
            f"{best['gbs']:.0f} GB/s  ({best['gbs'] / HBM_PEAK_GBS * 100:.1f}% HBM)"
        )
        results.append(
            {
                "label": label,
                "ms_median": ms_med,
                "ms_spread_pct": spread,
                "ms_best": best["ms"],
                "tflops_best": best["tflops"],
                "gbs_best": best["gbs"],
                "pct_hbm": best["gbs"] / HBM_PEAK_GBS * 100.0,
                "pct_peak_bf16": best["tflops"] / PEAK_BF16_TF * 100.0,
                "samples": samples,
            }
        )

    out = DATA / "02_sweep_bench.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "hbm_peak_gbs": HBM_PEAK_GBS,
                "peak_bf16_tflops": PEAK_BF16_TF,
                "warmup_iters": WARMUP_ITERS,
                "timed_iters": TIMED_ITERS,
                "attempts": ATTEMPTS,
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
            f"\nBest variant: {winner['label']}  {winner['ms_best']:.4f}ms  "
            f"{winner['tflops_best']:.2f} TF  {winner['pct_hbm']:.1f}% HBM"
        )


if __name__ == "__main__":
    main()
