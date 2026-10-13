#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Gate a workload on its L2 hit rate, measured with a single rocprofv3 pass.

Collects ``GL2C_HIT_sum`` and ``GL2C_MISS_sum`` around an arbitrary workload
command and reports the hit rate against a threshold, exiting non-zero when the
target is missed. That makes it usable as a CI/bisect gate rather than only as a
diagnostic.

    python probe_l2_residency.py --target 85 -- \\
        python -m builders.gfx1151.attention.wmma_fmha_swapqk_verify \\
            --seqlen-q 16384 --seqlen-k 16384 --no-verify --iters 3

Everything after ``--`` is the workload, run verbatim. Keep the iteration count
low: rocprofv3 serializes dispatches, so a long timing loop only makes the pass
slower without changing the hit rate.

Why this counter pair: when a reused working set (a K/V tile, a weight panel)
outgrows L2/MALL, the hit rate collapses and the kernel becomes residency-bound.
That is invisible in wall clock on parts that also power-throttle, and it is the
signal that distinguishes "needs fewer requests" from "needs better locality".

``measure()`` is the programmatic entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HIT = "GL2C_HIT_sum"
MISS = "GL2C_MISS_sum"


def _read_counters(outdir: Path, names: list[str]) -> dict[str, float]:
    """Sum each named counter over every dispatch in the rocprofv3 output."""
    totals: dict[str, float] = {n: 0.0 for n in names}
    for path in outdir.glob("**/*counter_collection.csv"):
        with open(path) as f:
            for row in csv.DictReader(f):
                name = row.get("Counter_Name", "")
                if name in totals:
                    try:
                        totals[name] += float(row.get("Counter_Value", 0.0))
                    except ValueError:
                        pass
    return totals


def measure(workload: list[str], *, timeout: int = 600) -> dict:
    """Run ``workload`` under rocprofv3 and return hit/miss/hit_rate."""
    if shutil.which("rocprofv3") is None:
        raise SystemExit("rocprofv3 not found on PATH")
    with tempfile.TemporaryDirectory(prefix="l2res_") as tmp:
        outdir = Path(tmp) / "out"
        proc = subprocess.run(
            [
                "rocprofv3",
                "--pmc",
                HIT,
                MISS,
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
        counters = _read_counters(outdir, [HIT, MISS])

    hit, miss = counters[HIT], counters[MISS]
    total = hit + miss
    if total <= 0:
        return {
            "hit": hit,
            "miss": miss,
            "hit_rate": None,
            "stderr": proc.stderr.strip()[-400:],
        }
    return {"hit": hit, "miss": miss, "hit_rate": 100.0 * hit / total, "stderr": ""}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", type=float, default=85.0, help="minimum hit rate %%")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "workload",
        nargs=argparse.REMAINDER,
        help="command to profile, after a bare --",
    )
    args = p.parse_args(argv)

    workload = args.workload[1:] if args.workload[:1] == ["--"] else args.workload
    if not workload:
        raise SystemExit("no workload given; pass it after a bare --")

    r = measure(workload, timeout=args.timeout)
    if args.json:
        print(json.dumps(r, indent=2))
    if r["hit_rate"] is None:
        print("FAIL: rocprofv3 collected no GL2C rows")
        if r["stderr"]:
            print(r["stderr"])
        return 1

    ok = r["hit_rate"] >= args.target
    if not args.json:
        print(
            f"L2 hit {r['hit_rate']:.1f}% "
            f"(hit {r['hit'] / 1e6:.1f}M / miss {r['miss'] / 1e6:.1f}M) "
            f"target {args.target:.1f}% -> {'PASS' if ok else 'FAIL'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
