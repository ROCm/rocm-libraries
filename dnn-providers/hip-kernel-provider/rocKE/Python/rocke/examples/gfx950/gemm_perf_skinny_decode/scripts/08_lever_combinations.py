#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 8: combine the t16x16x512 winner with every earlier-runbook
lever and report which compositions move the needle.

After 07 we have:
  baseline (step 02 winner):  48.3 µs  (8.7% HBM)
  t16x16x512_mem (step 07):   13.5 µs  (31.0% HBM)  vs rocBLAS 1.23x
  rocBLAS target:             11.0 µs  (38.1% HBM)

The runbook says: stacking can either compose (rare) or noise out
(common). This step is the explicit composition matrix.

We test on the t16x16x512 base (winner from step 07):
- pipeline: {mem, compv3, compv4}
- scheduler: {intrawave, interwave}
- epilogue: {default, cshuffle}
- persistent: {False, True}
- chiplet_swizzle: {False, True}
- waves_per_eu: {None, 2, 4, 8}

Also tested separately (different code path):
- preshuffle_b on t16x16x512 (needs custom launcher; see 09)

Total: 3*2*2*2*2*4 = 192 combos. We prune via is_valid_spec and skip
obvious noise duplicates.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import itertools
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
WARMUP, TIMED, ATTEMPTS = 20, 200, 3  # 3 attempts (192 combos)
ROCBLAS_MS = 0.0110

# Lever axes — base geometry is t16x16x512 w1x1
TILE_BASE = dict(
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

PIPELINES = ["mem", "compv3", "compv4"]
SCHEDS = ["intrawave", "interwave"]
EPILOGUES = ["default", "cshuffle"]
PERSISTENT = [False, True]
CHIPLET = [False, True]
WAVES_PER_EU = [None, 2, 4, 8]

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[5])}
BUILD = ROOT / "build_combo"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)


def label_for(pipe, sched, ep, pers, chip, wpe):
    parts = [pipe, sched, ep]
    if pers:
        parts.append("pers")
    if chip:
        parts.append("chip")
    if wpe is not None:
        parts.append(f"wpe{wpe}")
    return "_".join(parts)


def build(label, pipe, sched, ep, pers, chip, wpe):
    tile = TileSpec(**TILE_BASE)
    trait = TraitSpec(
        pipeline=pipe,
        scheduler=sched,
        epilogue=ep,
        pad_m=True,
        pad_n=True,
        pad_k=True,
        persistent=pers,
        chiplet_swizzle=chip,
        waves_per_eu=wpe,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    salted = f"{label}__m{M}n{N}k{K}"
    spec = UniversalGemmSpec(name=salted, tile=tile, trait=trait, data=data)
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(why)
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
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=60)
    out = r.stdout + r.stderr
    mms = re.search(r"([\d.]+)\s*ms", out)
    if not mms:
        return None
    mtf = re.search(r"([\d.]+)\s*TFlops", out)
    mgb = re.search(r"([\d.]+)\s*GB/s", out)
    return {
        "ms": float(mms.group(1)),
        "tflops": float(mtf.group(1)) if mtf else 0,
        "gbs": float(mgb.group(1)) if mgb else 0,
    }


def main():
    combos = list(
        itertools.product(
            PIPELINES, SCHEDS, EPILOGUES, PERSISTENT, CHIPLET, WAVES_PER_EU
        )
    )
    print(f"Sweeping {len(combos)} combinations on t16x16x512 base...")
    results = []
    for i, (pipe, sched, ep, pers, chip, wpe) in enumerate(combos):
        # Skip mem+intrawave (mem implies interwave hint pattern; not a strict rule but reduces noise)
        label = label_for(pipe, sched, ep, pers, chip, wpe)
        try:
            sub, kname = build(label, pipe, sched, ep, pers, chip, wpe)
        except Exception as e:
            results.append({"label": label, "error": str(e)[:80]})
            continue
        samples = []
        for j in range(ATTEMPTS + 1):
            r = run(sub, kname)
            if r is None:
                continue
            if j == 0:
                continue
            samples.append(r)
        if not samples:
            results.append({"label": label, "error": "no samples"})
            continue
        best = min(samples, key=lambda s: s["ms"])
        ms_med = median(s["ms"] for s in samples)
        gap = best["ms"] / ROCBLAS_MS
        results.append(
            {
                "label": label,
                "ms_best": best["ms"],
                "ms_median": ms_med,
                "tflops_best": best["tflops"],
                "gbs_best": best["gbs"],
                "pct_hbm": best["gbs"] / HBM_PEAK_GBS * 100.0,
                "vs_rocblas_ratio": gap,
                "config": {
                    "pipeline": pipe,
                    "scheduler": sched,
                    "epilogue": ep,
                    "persistent": pers,
                    "chiplet": chip,
                    "waves_per_eu": wpe,
                },
            }
        )

    out = DATA / "08_lever_combinations.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "base_tile": TILE_BASE,
                "rocblas_ms": ROCBLAS_MS,
                "n_combos": len(combos),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")

    ok = [r for r in results if "ms_best" in r]
    ok.sort(key=lambda r: r["ms_best"])
    print(f"\nTop 10 of {len(ok)} successful combinations:")
    print(f"  {'label':<50} {'ms':>8} {'%HBM':>6} {'vs rocBLAS':>10}")
    for r in ok[:10]:
        print(
            f"  {r['label']:<50} {r['ms_best'] * 1000:>7.2f}µs {r['pct_hbm']:>5.1f}% {r['vs_rocblas_ratio']:>9.2f}x"
        )

    if ok:
        winner = ok[0]
        print(
            f"\n*** Best: {winner['label']}  {winner['ms_best'] * 1000:.2f} µs  "
            f"({winner['pct_hbm']:.1f}% HBM)  {winner['vs_rocblas_ratio']:.2f}x vs rocBLAS ***"
        )


if __name__ == "__main__":
    main()
