#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split a workload's GPU cycles into issue cost vs stall, using rocprofv3.

Answers "am I issue-bound or stalled?" with numbers instead of intuition. The
probe runs a workload under rocprofv3 in several small counter groups, then
prices the retired instruction mix against per-instruction issue costs (in
cycles per instruction per SIMD) to produce a budget:

    WMMA issue + transcendental issue + other VALU issue = total issue
    measured cycles - total issue                        = stall / unaccounted

Get the issue-cost anchors from ``probe_roofline_peaks.py`` on the same part;
they are clock-independent, which matters on APUs that power-throttle mid-run.

    python probe_cycle_budget.py --simds 80 \\
        --anchor-mma 36.15 --anchor-valu 1.31 --anchor-transcendental 4.00 \\
        --mma-count 1.18e9 --transcendental-count 3.15e7 -- \\
        python -m builders.gfx1151.attention.wmma_fmha_swapqk_verify \\
            --seqlen-q 16384 --seqlen-k 16384 --no-verify --iters 3

``--mma-count`` / ``--transcendental-count`` are the dynamic counts you can
derive exactly from the problem shape (tiles per iteration x iterations x
waves). They are needed because ``SQ_INSTS_VALU`` on RDNA3 lumps matrix and
transcendental instructions in with plain VALU, so the split cannot be read off
a counter. Omit them to get a plain total-VALU budget with no breakdown.

Several groups are used because many parts cannot collect more than a handful of
counters per pass; asking for too many makes rocprofv3 replay the dispatch until
it times out. A group that fails degrades gracefully instead of aborting the run.

``run()`` is the programmatic entry point.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

# Small groups, each repeating GRBM_GUI_ACTIVE so every pass is self-normalizing.
COUNTER_GROUPS = [
    ("core", ["GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_TEX_LOAD"]),
    ("occupancy", ["GRBM_GUI_ACTIVE", "MeanOccupancyPerCU", "TA_TA_BUSY", "VALUBusy"]),
    ("memory", ["GRBM_GUI_ACTIVE", "GL2C_HIT_sum", "GL2C_MISS_sum"]),
]


def _read_mean_per_dispatch(outdir: Path, names: list[str]) -> dict[str, float]:
    """Average each counter over the dispatches rocprofv3 recorded."""
    agg: dict[str, float] = collections.defaultdict(float)
    dispatches = set()
    files = sorted(outdir.glob("**/*counter_collection.csv"))
    if not files:
        return {}
    for path in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    agg[row["Counter_Name"]] += float(row["Counter_Value"])
                except (KeyError, ValueError):
                    continue
                dispatches.add(row.get("Dispatch_Id"))
    if not dispatches:
        return {}
    return {n: agg.get(n, 0.0) / len(dispatches) for n in names}


def run(
    workload: list[str],
    *,
    simds: int,
    anchor_mma: float,
    anchor_valu: float,
    anchor_transcendental: float,
    mma_count: float | None = None,
    transcendental_count: float | None = None,
    timeout: int = 600,
) -> dict:
    """Profile ``workload`` and return the cycle budget as a dict."""
    if shutil.which("rocprofv3") is None:
        raise SystemExit("rocprofv3 not found on PATH")

    got: dict[str, float] = {}
    failed = []
    with tempfile.TemporaryDirectory(prefix="cycbud_") as tmp:
        for tag, counters in COUNTER_GROUPS:
            outdir = Path(tmp) / tag
            try:
                subprocess.run(
                    [
                        "rocprofv3",
                        "--pmc",
                        *counters,
                        "-d",
                        str(outdir),
                        "-f",
                        "csv",
                        "--",
                        *workload,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                failed.append(tag)
                continue
            got.update(_read_mean_per_dispatch(outdir, counters))

    cyc = got.get("GRBM_GUI_ACTIVE", 0.0)
    valu = got.get("SQ_INSTS_VALU", 0.0)
    if cyc <= 0 or valu <= 0:
        return {"ok": False, "failed_groups": failed, "counters": got}

    mma = mma_count or 0.0
    trans = transcendental_count or 0.0
    other_valu = max(valu - mma - trans, 0.0)
    c_mma = mma * anchor_mma / simds
    c_trans = trans * anchor_transcendental / simds
    c_valu = other_valu * anchor_valu / simds
    c_issue = c_mma + c_trans + c_valu

    hit, miss = got.get("GL2C_HIT_sum", 0.0), got.get("GL2C_MISS_sum", 0.0)
    return {
        "ok": True,
        "failed_groups": failed,
        "cycles": cyc,
        "waves": got.get("SQ_WAVES", 0.0),
        "occupancy_per_cu": got.get("MeanOccupancyPerCU", 0.0),
        "insts_valu_total": valu,
        "mma_count": mma,
        "transcendental_count": trans,
        "other_valu_count": other_valu,
        "issue_mma": c_mma,
        "issue_transcendental": c_trans,
        "issue_other_valu": c_valu,
        "issue_total": c_issue,
        "stall": cyc - c_issue,
        "l2_hit_rate": 100.0 * hit / (hit + miss) if hit + miss else None,
        "ta_busy": got.get("TA_TA_BUSY", 0.0),
        "valu_busy": got.get("VALUBusy", 0.0),
    }


def _report(r: dict) -> None:
    cyc = r["cycles"]
    print(
        f"cycles (GRBM_GUI_ACTIVE) {cyc / 1e6:.2f} M   "
        f"waves {r['waves']:.0f}   occupancy {r['occupancy_per_cu']:.1f} waves/CU"
    )
    print("\ndynamic instructions (per dispatch, millions)")
    print(
        f"  SQ_INSTS_VALU (incl MMA + transcendental) {r['insts_valu_total'] / 1e6:>9.1f}"
    )
    print(f"  MMA (supplied)                            {r['mma_count'] / 1e6:>9.1f}")
    print(
        f"  transcendental (supplied)                 {r['transcendental_count'] / 1e6:>9.1f}"
    )
    print(
        f"  other VALU (derived)                      {r['other_valu_count'] / 1e6:>9.1f}"
    )
    print("\ncycle budget (per SIMD, Mcyc)")
    for label, key in (
        ("MMA issue", "issue_mma"),
        ("transcendental issue", "issue_transcendental"),
        ("other VALU issue", "issue_other_valu"),
        ("= total issue", "issue_total"),
        ("stall / unaccounted", "stall"),
    ):
        v = r[key]
        print(f"  {label:<22}{v / 1e6:>8.2f}  {100 * v / cyc:>5.1f}%")
    if r["l2_hit_rate"] is not None:
        print(
            f"\nmemory: L2 hit {r['l2_hit_rate']:.1f}%   "
            f"TA_TA_BUSY {r['ta_busy'] / 1e6:.1f} M   VALUBusy {r['valu_busy']:.1f}%"
        )
    if r["failed_groups"]:
        print(f"\n(counter groups that timed out: {', '.join(r['failed_groups'])})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--simds", type=int, required=True, help="SIMDs on the part")
    p.add_argument("--anchor-mma", type=float, required=True, help="cyc/instr/SIMD")
    p.add_argument("--anchor-valu", type=float, required=True)
    p.add_argument("--anchor-transcendental", type=float, required=True)
    p.add_argument("--mma-count", type=float, default=None)
    p.add_argument("--transcendental-count", type=float, default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--json", action="store_true")
    p.add_argument("workload", nargs=argparse.REMAINDER, help="command after a bare --")
    args = p.parse_args(argv)

    workload = args.workload[1:] if args.workload[:1] == ["--"] else args.workload
    if not workload:
        raise SystemExit("no workload given; pass it after a bare --")

    r = run(
        workload,
        simds=args.simds,
        anchor_mma=args.anchor_mma,
        anchor_valu=args.anchor_valu,
        anchor_transcendental=args.anchor_transcendental,
        mma_count=args.mma_count,
        transcendental_count=args.transcendental_count,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    if not r["ok"]:
        print("FAIL: rocprofv3 produced no usable counters")
        print(f"  groups that timed out: {r['failed_groups']}")
        return 1
    _report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
