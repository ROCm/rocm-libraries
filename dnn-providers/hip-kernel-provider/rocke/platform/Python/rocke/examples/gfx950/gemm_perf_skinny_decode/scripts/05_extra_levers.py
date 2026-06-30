#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 5: try the remaining `UniversalGemmSpec.TraitSpec` levers on the
sweep winner.

Runbook anchors:
- §12.1.G `waves_per_eu` occupancy hint
- §12.1.I `persistent` (CK Tile uses persistent CTAs in 03_gemm,
  18_flatmm, 40_streamk_gemm)
- §12.1.L Multi-XCD / chiplet grid swizzle (CK Tile's
  `chiplet_aware_super_tile` from the multi-XCD MI300X/MI355X path)
- §17.4 "compiler hints rarely close large gaps"
- §17.4 "a structural change often needs multiple co-evolved levers"

Inspired by the CK Tile examples under
`example/ck_tile/03_gemm/universal_gemm.cpp` (persistent / k_batch /
chiplet) and `example/ck_tile/18_flatmm/flatmm_basic.hpp`
(FlatmmConfig32 reference defaults).

The hypothesis going in: with only 64 N-tiles at tile_n=64, a single
XCD (38 CUs on MI355X) is comfortably enough — chiplet swizzle should
be neutral. `waves_per_eu=2` may help shake more waves out of the 56-VGPR
kernel (probe step showed `MAX_WAVES_PER_CU` is already the limiter at
32, so this probably doesn't move). `persistent=True` keeps the same
work item count; only the dispatch loop changes — also unlikely to
move a 48 µs kernel.

This step is the *honest* runbook check: try the remaining knobs, then
report which ones moved the needle and which didn't. (§17.4 pattern:
record what didn't help, with the why.)
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

# Lock the geometry to the sweep winner (t16x64x32_w1x2_a16x16x32_mem)
BASE_TILE = dict(
    tile_m=16,
    tile_n=64,
    tile_k=32,
    warp_m=1,
    warp_n=2,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)
M, N, K = 2, 4096, 4096
HBM_PEAK_GBS = 8000.0
PEAK_BF16_TF = 2500.0
WARMUP, TIMED, ATTEMPTS = 20, 200, 5

# Each row is one extra-lever experiment layered on the winner geometry.
# (label, pipeline, scheduler, persistent, chiplet_swizzle, waves_per_eu, hypothesis)
EXPERIMENTS = [
    (
        "baseline_winner",
        "mem",
        "interwave",
        False,
        False,
        None,
        "sweep winner reference",
    ),
    (
        "waves_per_eu_2",
        "mem",
        "interwave",
        False,
        False,
        2,
        "lower VGPR target → more waves/CU",
    ),
    ("waves_per_eu_3", "mem", "interwave", False, False, 3, "even-lower VGPR target"),
    (
        "persistent",
        "mem",
        "interwave",
        True,
        False,
        None,
        "persistent CTAs: amortise launch",
    ),
    (
        "chiplet_swizzle",
        "mem",
        "interwave",
        False,
        True,
        None,
        "XCD-aware grid swizzle (MI355X 8 XCDs)",
    ),
    (
        "persistent_chiplet",
        "mem",
        "interwave",
        True,
        True,
        None,
        "stack: persistent + chiplet",
    ),
    (
        "compv4_persistent",
        "compv4",
        "intrawave",
        True,
        False,
        None,
        "co-evolve: pipeline + persistent (§17.4)",
    ),
    (
        "compv4_persistent_chiplet",
        "compv4",
        "intrawave",
        True,
        True,
        None,
        "full stack",
    ),
]

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[5])}

BUILD = ROOT / "build_extra"
DATA = ROOT / "data"
BUILD.mkdir(exist_ok=True)


def build(label, pipe, sched, persist, chiplet, wpu, _hypo):
    tile = TileSpec(**BASE_TILE)
    trait = TraitSpec(
        pipeline=pipe,
        scheduler=sched,
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        persistent=persist,
        chiplet_swizzle=chiplet,
        waves_per_eu=wpu,
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
        atoms=[
            f"mfma_f32_{tile.warp_tile_m}x{tile.warp_tile_n}x{tile.warp_tile_k}_bf16"
        ],
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
        return None, out
    return {
        "ms": float(mms.group(1)),
        "tflops": float(mtf.group(1)),
        "gbs": float(mgb.group(1)),
    }, out


def main():
    results = []
    for exp in EXPERIMENTS:
        label = exp[0]
        hypo = exp[-1]
        try:
            sub, kname = build(*exp)
        except Exception as e:
            print(f"{label:<32} BUILD-FAIL: {type(e).__name__}: {str(e)[:120]}")
            results.append({"label": label, "hypothesis": hypo, "error": str(e)})
            continue
        samples = []
        for i in range(ATTEMPTS + 1):
            r, _ = run(sub, kname)
            if r is None:
                continue
            if i == 0:
                continue
            samples.append(r)
        if not samples:
            print(f"{label:<32} NO-SAMPLES")
            results.append({"label": label, "hypothesis": hypo, "error": "no samples"})
            continue
        ms_med = median(s["ms"] for s in samples)
        best = min(samples, key=lambda s: s["ms"])
        spread = (
            (max(s["ms"] for s in samples) - min(s["ms"] for s in samples))
            / ms_med
            * 100.0
        )
        print(
            f"{label:<32} med={ms_med:.4f}ms spread={spread:4.1f}%  "
            f"best={best['ms']:.4f}ms  {best['tflops']:.2f}TF  "
            f"{best['gbs']:.0f}GB/s  ({best['gbs'] / HBM_PEAK_GBS * 100:.1f}% HBM)"
        )
        results.append(
            {
                "label": label,
                "hypothesis": hypo,
                "ms_median": ms_med,
                "ms_best": best["ms"],
                "ms_spread_pct": spread,
                "tflops_best": best["tflops"],
                "gbs_best": best["gbs"],
                "pct_hbm": best["gbs"] / HBM_PEAK_GBS * 100.0,
                "samples": samples,
            }
        )

    out = DATA / "05_extra_levers.json"
    with out.open("w") as f:
        json.dump(
            {
                "shape": {"M": M, "N": N, "K": K},
                "base_tile": BASE_TILE,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")

    ok = [r for r in results if "ms_best" in r]
    if not ok:
        return
    baseline = next((r for r in ok if r["label"] == "baseline_winner"), None)
    if baseline is None:
        return
    print("\n--- Per-lever delta vs baseline_winner ---")
    print(f"{'label':<32} {'Δms':>8} {'Δ%':>7}  verdict")
    for r in ok:
        dms = r["ms_best"] - baseline["ms_best"]
        dpct = dms / baseline["ms_best"] * 100.0
        v = "win" if dpct < -0.5 else ("loss" if dpct > 0.5 else "noise")
        print(f"{r['label']:<32} {dms:>+8.4f} {dpct:>+6.1f}%  {v}")


if __name__ == "__main__":
    main()
