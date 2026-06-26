#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 1 of the runbook loop: STATIC probe before any benchmark.

Runbook anchors:
- §3.1b Static Inspection First — The DSL Probe Tier
- §12.1.Q Static probes
- §13.1 GEMM Checklist (occupancy / VGPR / LDS questions)

For each candidate o_proj decode tile we compile the kernel and read
VGPR / AGPR / SGPR / LDS / spill / waves-per-CU straight from the
HSACO via `probe_occupancy`. Anything that spills or drops waves/CU
below 4 is filtered out BEFORE we run a single launch.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSL = str(Path(__file__).resolve().parents[5])
PROBES = (
    str(Path(__file__).resolve().parents[6]) + "/"
    "dsl_docs/optimization/utilities/tools/dsl_probes"
)
sys.path.insert(0, DSL)
sys.path.insert(0, PROBES)

from rocke.instances.common.gemm_universal import (  # noqa: E402
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from probe_occupancy import probe_occupancy, ARCH_GFX950  # noqa: E402

# o_proj decode shape: A[M=2, K=4096] @ B[N=4096, K=4096]^T  -> C[M=2, N=4096]
# bf16, RCR layout.
#
# Tile-geometry candidates (runbook §12.1.B):
#   - tile_m must be a multiple of warp_m × warp_tile_m; pad up from M=2 -> 16.
#   - tile_n covers a slab of weight rows; bigger ⇒ more reuse, more LDS / VGPR.
#   - tile_k drives K-loop trips; pair with kpack via warp_tile_k atom (§12.1.C).
#
# Pipeline / scheduler (runbook §12.1.D):
#   - "mem" = single-buffered; baseline for memory-bound regimes.
#   - "compv3" / "compv4" = double-buffered async DMA + MFMA overlap.
#     compv4 is the runbook's "skinny needs latency hiding" pick.
#   - intrawave keeps producer/consumer in one wave (use with compv4);
#     interwave splits them (pairs with mem).
#
# We DO sweep across these; each row carries a hypothesis tag.
CANDIDATES = [
    # label, tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, wtm, wtn, wtk, pipeline, scheduler, hypothesis
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
        "smallest N slab, lowest LDS",
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
        "default skinny tile",
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
        "wider K window per loop trip",
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
        "wider N slab, 4 warps",
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
        "more M rows (pad waste, but more warps)",
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
        "double-buffer pipeline",
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
        "async-DMA + MFMA overlap",
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
        "compv4 + wider K",
    ),
]


def build(label, tm, tn, tk, wm, wn, wk, wtm, wtn, wtk, pipe, sched, _hypo):
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
    spec = UniversalGemmSpec(name=label, tile=tile, trait=trait, data=data)
    return build_universal_gemm(spec), spec.block_size, wm * wn * wk


def main():
    entries = []
    for cand in CANDIDATES:
        label = cand[0]
        try:
            kdef, _block_size, waves_per_wg = build(*cand)
        except Exception as e:
            print(f"{label:<40}  BUILD-CONSTRUCT-FAIL: {e}")
            continue
        entries.append((label, kdef, waves_per_wg))

    rows = probe_occupancy(entries, arch=ARCH_GFX950)

    # Annotate with hypothesis + apply the "≥4 waves/CU AND zero spill" filter
    # from runbook §14.2 (low occupancy + register spills are both failure modes).
    hyp = {c[0]: c[-1] for c in CANDIDATES}
    decisions = []
    for r in rows:
        lbl = r["label"]
        keep = (r["vgpr_spill_count"] == 0) and (r["waves_per_cu"] >= 4)
        decisions.append(
            {
                "label": lbl,
                "hypothesis": hyp.get(lbl, ""),
                "vgpr": r["vgpr_count"],
                "agpr": r["agpr_count"],
                "sgpr": r["sgpr_count"],
                "spill": r["vgpr_spill_count"],
                "lds_bytes": r["lds_size"],
                "waves_per_cu": r["waves_per_cu"],
                "wgs_per_cu": r["wgs_per_cu"],
                "limited_by": r["limited_by"],
                "passes_filter": keep,
            }
        )

    print()
    print(f"{'label':<40} {'keep?':<6} hypothesis")
    print("-" * 110)
    for d in decisions:
        flag = "KEEP" if d["passes_filter"] else "drop"
        print(f"{d['label']:<40} {flag:<6} {d['hypothesis']}")

    out = ROOT / "data" / "01_occupancy.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump({"rows": decisions}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
