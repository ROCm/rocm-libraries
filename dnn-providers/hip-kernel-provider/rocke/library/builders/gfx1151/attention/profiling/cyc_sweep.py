#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Clock-independent config sweep: FLOP/cycle for the swapqk kernel.

Wall-clock TF is not comparable across runs on this part -- it power-throttles
from ~2400 MHz / 87 W to ~1600 MHz / 43 W under sustained load, which is larger
than the config differences being compared. This driver runs each config under
rocprofv3, reads GRBM_GUI_ACTIVE (GPU-active cycles) and the dispatch wall time,
and reports FLOP/cycle plus the TFLOPS that implies at a chosen fixed clock.

Usage:
    python3 cyc_sweep.py --seqlen 16384 --reps 2 --clock-mhz 2400
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path

# (label, extra vt_prof args)
CONFIGS = [
    ("mq1 of32 bn32 w2", ["--mq", "1", "--of16", "0", "--block-n", "32"]),
    (
        "mq1 of32 bn32 w2 vt",
        ["--mq", "1", "--of16", "0", "--block-n", "32", "--vt", "1"],
    ),
    ("mq1 of32 bn64 w2", ["--mq", "1", "--of16", "0", "--block-n", "64"]),
    (
        "mq1 of32 bn64 w2 vt",
        ["--mq", "1", "--of16", "0", "--block-n", "64", "--vt", "1"],
    ),
    (
        "mq1 of32 bn64 w2 ilp1",
        ["--mq", "1", "--of16", "0", "--block-n", "64", "--ilp", "1"],
    ),
    (
        "mq1 of32 bn64 w1",
        ["--mq", "1", "--of16", "0", "--block-n", "64", "--waves", "1"],
    ),
    (
        "mq1 of32 bn64 w4",
        ["--mq", "1", "--of16", "0", "--block-n", "64", "--waves", "4"],
    ),
    ("mq1 of16 bn64 w2", ["--mq", "1", "--of16", "1", "--block-n", "64"]),
    (
        "mq1 of16 bn64 w2 vt",
        ["--mq", "1", "--of16", "1", "--block-n", "64", "--vt", "1"],
    ),
    ("mq2 of16 bn32 w2", ["--mq", "2", "--of16", "1", "--block-n", "32"]),
    ("mq2 of32 bn32 w2", ["--mq", "2", "--of16", "0", "--block-n", "32"]),
]


def read_counters(outdir: str):
    files = glob.glob(f"{outdir}/*/*counter_collection.csv")
    if not files:
        return None
    agg = collections.defaultdict(float)
    dur = {}
    for row in csv.DictReader(open(files[0])):
        agg[row["Counter_Name"]] += float(row["Counter_Value"])
        dur[row["Dispatch_Id"]] = (
            int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        ) / 1e6
    if not dur:
        return None
    n = len(dur)
    return agg["GRBM_GUI_ACTIVE"] / n, sum(dur.values()) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--clock-mhz", type=float, default=2400.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--filter", default="", help="only configs containing this")
    args = ap.parse_args()

    L, D = args.seqlen, args.head_size
    # the profiled dispatch covers ONE head (chunk=1)
    flops = 4.0 * L * L * D
    here = Path(__file__).resolve().parent

    print(
        f"L={L} D={D}  one-head dispatch = {flops / 1e9:.1f} GFLOP  "
        f"reference clock = {args.clock_mhz:.0f} MHz\n"
    )
    print(
        f"{'config':<24}{'Mcyc mean':>10}{'min':>9}{'spread':>8}"
        f"{'FLOP/cyc':>10}{'TF@clk':>8}{'TF@clk*':>9}"
    )
    print("-" * 74)

    results = []
    for label, extra in CONFIGS:
        if args.filter and args.filter not in label:
            continue
        cyc, ms = [], []
        for r in range(args.reps):
            outdir = f"/tmp/cyc_{abs(hash(label)) % 10**8}_{r}"
            shutil.rmtree(outdir, ignore_errors=True)
            cmd = [
                "rocprofv3",
                "--pmc",
                "GRBM_GUI_ACTIVE",
                "-d",
                outdir,
                "-f",
                "csv",
                "--",
                sys.executable,
                str(here / "vt_prof.py"),
                "--seqlen",
                str(L),
                "--head-size",
                str(D),
                "--heads",
                str(args.heads),
                "--iters",
                str(args.iters),
            ] + extra
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            got = read_counters(outdir)
            if got is None:
                print(
                    f"{label:<24}{'FAIL':>10}  {p.stderr.strip().splitlines()[-1][:30]}"
                )
                break
            cyc.append(got[0])
            ms.append(got[1])
        if len(cyc) < args.reps:
            continue
        cmin, cmean = min(cyc), sum(cyc) / len(cyc)
        fpc = flops / cmean
        tf_at = fpc * args.clock_mhz * 1e6 / 1e12
        tf_best = flops / cmin * args.clock_mhz * 1e6 / 1e12
        spread = 100 * (max(cyc) - cmin) / cmin
        print(
            f"{label:<24}{cmean / 1e6:>10.2f}{cmin / 1e6:>9.2f}"
            f"{spread:>7.1f}%{fpc:>10.0f}{tf_at:>8.2f}{tf_best:>9.2f}"
        )
        results.append((tf_at, label, fpc))

    if results:
        results.sort(reverse=True)
        tf, label, fpc = results[0]
        print(
            f"\nbest: {label}  {fpc:.0f} FLOP/cycle "
            f"-> {tf:.2f} TF at {args.clock_mhz:.0f} MHz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
