#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 6: try the geometries hipBLASLt actually picks.

Ground truth (from `rocprof --hsa-trace` on `torch.matmul(bf16)` for
M=2 N=4096 K=4096 on gfx950 / MI355X, ROCm 7.0.2):

    Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1
        ..._GRVWA8_GRVWB8_GSU0_SK3_SVW1_PGR2_PLR1_SIA3_DTLA1_DTLB1...
        WG=16,4,4 (256 threads, 4 warps)

Decoded:
  MT16x16x512     macro tile: 16 x 16, DepthU (tile_k) = 512
  MI16x16x1       MFMA atom 16x16x32 bf16 (the "x1" is MIBlock packing)
  GRVWA8/GRVWB8   8-element vector loads for both A and B
  GSU0 / SK3      no GlobalSplitU; StreamK enabled in "3" mode (atomic+two-tile)
  PGR2 PLR1 SIA3  PrefetchGlobalRead=2, PrefetchLocalRead=1, fixed schedule
  DTLA1 DTLB1     DirectToLDS on both A and B (HBM -> LDS, skipping VGPRs)
  WG 16,4,4       4 waves of 64 lanes

The DSL example sweep's winner was `t16x64x32` (tile_k=32). hipBLASLt
runs `tile_k=512` — sixteen times more K per CTA pass. That changes
the geometry/occupancy trade in a way none of the runbook scheduling
knobs reach.

What this script tests, using DSL knobs only:

  A. tile_n=16 (vs 64): match hipBLASLt's N tile so we get more CTAs
     covering N=4096 (256 CTAs vs 64).
  B. tile_k bumped to 64/128/256: closer to depthU=512 (we can't go
     all the way without making the LDS budget explode given the DSL
     emitter doesn't fold AB+C LDS yet).
  C. preshuffle_b=True on the surviving geometry: contiguous B loads.
     This is the DSL equivalent of hipBLASLt's DirectToLDS+wide-vector
     B-side.

Goal: close as much of the 4.38× gap as the DSL surface can reach,
report honestly what's left.
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

# Candidates inspired by the ground-truth hipBLASLt kernel.
# (label, tm, tn, tk, wm, wn, wtm, wtn, wtk, pipeline, scheduler, preshuffle_b)
CANDIDATES = [
    # Baseline (the sweep winner)
    ("baseline_t16x64x32", 16, 64, 32, 1, 2, 16, 16, 32, "mem", "interwave", False),
    # Lever A: smaller N tile (more CTAs)
    ("t16x16x32_w1x1", 16, 16, 32, 1, 1, 16, 16, 32, "mem", "interwave", False),
    ("t16x16x64_w1x1", 16, 16, 64, 1, 1, 16, 16, 32, "mem", "interwave", False),
    ("t16x16x128_w1x1", 16, 16, 128, 1, 1, 16, 16, 32, "mem", "interwave", False),
    ("t16x16x256_w1x1", 16, 16, 256, 1, 1, 16, 16, 32, "mem", "interwave", False),
    # Lever B: bump tile_k on the original N tile (deeper K, same CTA count)
    ("t16x64x64_w1x2", 16, 64, 64, 1, 2, 16, 16, 32, "mem", "interwave", False),
    ("t16x64x128_w1x2", 16, 64, 128, 1, 2, 16, 16, 32, "mem", "interwave", False),
    ("t16x64x256_w1x2", 16, 64, 256, 1, 2, 16, 16, 32, "mem", "interwave", False),
    # Lever C: preshuffle_b on the most promising-looking geometries.
    # Per docs/preshuffle.md: B is laid out (k_tiles, n_tiles, block_n, block_k)
    # so each tile's load is one contiguous buffer_load_dwordxN burst.
    ("t16x16x128_w1x1_preB", 16, 16, 128, 1, 1, 16, 16, 32, "mem", "interwave", True),
    ("t16x16x256_w1x1_preB", 16, 16, 256, 1, 1, 16, 16, 32, "mem", "interwave", True),
    ("t16x64x128_w1x2_preB", 16, 64, 128, 1, 2, 16, 16, 32, "mem", "interwave", True),
    ("t16x64x256_w1x2_preB", 16, 64, 256, 1, 2, 16, 16, 32, "mem", "interwave", True),
    # compv4 / cshuffle stack on the deep-K geometry (close composition test)
    (
        "t16x16x128_w1x1_compv4",
        16,
        16,
        128,
        1,
        1,
        16,
        16,
        32,
        "compv4",
        "intrawave",
        False,
    ),
    (
        "t16x16x256_w1x1_compv4_preB",
        16,
        16,
        256,
        1,
        1,
        16,
        16,
        32,
        "compv4",
        "intrawave",
        True,
    ),
]

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[5])}

BUILD = ROOT / "build_ground_truth"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)


def build(label, tm, tn, tk, wm, wn, wtm, wtn, wtk, pipe, sched, preB):
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
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
        preshuffle_b=preB,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    salted = f"{label}__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid spec: {why}")
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
        atoms=[f"mfma_f32_{wtm}x{wtn}x{wtk}_bf16"],
    )
    write_artifact(art, sub, manifest)
    return sub, art.kernel_name, preB


def run(sub, kname, preB):
    cmd = [
        PY,
        "-m",
        "rocke.run_manifest",
        str(sub / f"{kname}.hsaco"),
        str(sub / "manifest.json"),
        "--shape",
        f"{M},{N},{K}",
    ]
    if preB:
        cmd.append("--preshuffle-b")
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=120)
    out = r.stdout + r.stderr
    mms = re.search(r"([\d.]+)\s*ms", out)
    mtf = re.search(r"([\d.]+)\s*TFlops", out)
    mgb = re.search(r"([\d.]+)\s*GB/s", out)
    if not (mms and mtf and mgb):
        return None, out
    return {
        "ms": float(mms.group(1)),
        "tflops": float(mtf.group(1)),
        "gbs": float(mgb.group(1)),
    }, out


def main():
    results = []
    for exp in CANDIDATES:
        label = exp[0]
        try:
            sub, kname, preB = build(*exp)
        except Exception as e:
            print(f"{label:<36} BUILD-FAIL: {type(e).__name__}: {str(e)[:100]}")
            results.append({"label": label, "error": str(e)})
            continue
        samples = []
        last_out = ""
        for i in range(ATTEMPTS + 1):
            r, out = run(sub, kname, preB)
            last_out = out
            if r is None:
                continue
            if i == 0:
                continue
            samples.append(r)
        if not samples:
            print(f"{label:<36} NO-SAMPLES")
            if last_out:
                print(f"    last stderr: {last_out.strip().splitlines()[-1][:120]}")
            results.append({"label": label, "error": "no samples"})
            continue
        ms_med = median(s["ms"] for s in samples)
        best = min(samples, key=lambda s: s["ms"])
        spread = (
            (max(s["ms"] for s in samples) - min(s["ms"] for s in samples))
            / ms_med
            * 100.0
        )
        print(
            f"{label:<36} med={ms_med:.4f}ms spread={spread:4.1f}%  "
            f"best={best['ms']:.4f}ms  {best['tflops']:.2f}TF  "
            f"{best['gbs']:.0f}GB/s  ({best['gbs'] / HBM_PEAK_GBS * 100:.1f}% HBM)"
        )
        results.append(
            {
                "label": label,
                "ms_median": ms_med,
                "ms_best": best["ms"],
                "ms_spread_pct": spread,
                "tflops_best": best["tflops"],
                "gbs_best": best["gbs"],
                "pct_hbm": best["gbs"] / HBM_PEAK_GBS * 100.0,
                "samples": samples,
            }
        )

    out = DATA / "06_ground_truth_geometries.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "results": results,
                "note": "Candidates inspired by rocprof of hipBLASLt's actual kernel "
                "MT16x16x512 PGR2 SK3 DTLA1 DTLB1 GRVWA8/B8 for this shape.",
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")

    ok = [r for r in results if "ms_best" in r]
    if ok:
        baseline = next((r for r in ok if r["label"] == "baseline_t16x64x32"), None)
        winner = min(ok, key=lambda r: r["ms_best"])
        print(
            f"\nWinner: {winner['label']}  {winner['ms_best']:.4f}ms  "
            f"({winner['pct_hbm']:.1f}% HBM)"
        )
        if baseline:
            speedup = baseline["ms_best"] / winner["ms_best"]
            print(f"Speedup over baseline: {speedup:.2f}x")
            print(f"Gap to rocBLAS (11.0 µs): {winner['ms_best'] / 0.0110:.2f}x slower")


if __name__ == "__main__":
    main()
