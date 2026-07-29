#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cycle-exact roofline anchors for gfx1151, via rocprofv3 GRBM_GUI_ACTIVE.

peaks.hip runs NACC=8 independent chains of one instruction class. Dividing the
dispatch's GPU-active cycles by the ops each SIMD retired gives the per-SIMD
issue cost in clocks, which is clock-independent and therefore immune to the
power throttle on this part.

Run at 1 wave/SIMD (issue cost with the SIMD to itself) and saturated (the real
ceiling once many waves compete).
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import re
import shutil
import subprocess

CUS = 40
SIMDS = CUS * 2
NACC = 8
FLOP = {"wmma": 8192.0, "valu": 64.0, "exp": 32.0}


def cycles(outdir):
    files = glob.glob(f"{outdir}/*/*counter_collection.csv")
    if not files:
        return None
    agg = collections.defaultdict(float)
    disp = set()
    for row in csv.DictReader(open(files[0])):
        if "peak" not in row["Kernel_Name"]:
            continue
        agg[row["Counter_Name"]] += float(row["Counter_Value"])
        disp.add(row["Dispatch_Id"])
    if not disp:
        return None
    # warmup + timed dispatch both run; take the larger (timed) one
    per = {}
    for row in csv.DictReader(open(files[0])):
        if "peak" in row["Kernel_Name"] and row["Counter_Name"] == "GRBM_GUI_ACTIVE":
            per[row["Dispatch_Id"]] = float(row["Counter_Value"])
    return max(per.values()) if per else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default="/tmp/peaks")
    ap.add_argument("--clock-mhz", type=float, default=2400.0)
    args = ap.parse_args()

    # (label, kernel, iters, blocks, threads)
    RUNS = [
        ("1 wave/SIMD", "wmma", 20000, SIMDS, 32),
        ("1 wave/SIMD", "valu", 200000, SIMDS, 32),
        ("1 wave/SIMD", "exp", 200000, SIMDS, 32),
        ("saturated", "wmma", 4000, CUS * 8, 256),
        ("saturated", "valu", 40000, CUS * 8, 256),
        ("saturated", "exp", 40000, CUS * 8, 256),
    ]

    print(
        f"gfx1151, {CUS} CUs / {SIMDS} SIMDs, reference clock "
        f"{args.clock_mhz:.0f} MHz\n"
    )
    print(
        f"{'mode':<13}{'op':<6}{'waves/SIMD':>11}{'cyc/op/SIMD':>13}"
        f"{'ops/clk/CU':>12}{'TFLOP/s@ref':>13}"
    )
    print("-" * 68)

    for label, which, iters, blocks, thr in RUNS:
        outdir = f"/tmp/pk_{which}_{blocks}"
        shutil.rmtree(outdir, ignore_errors=True)
        p = subprocess.run(
            [
                "rocprofv3",
                "--pmc",
                "GRBM_GUI_ACTIVE",
                "-d",
                outdir,
                "-f",
                "csv",
                "--",
                args.exe,
                which,
                str(iters),
                str(blocks),
                str(thr),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        cyc = cycles(outdir)
        if cyc is None:
            print(f"{label:<13}{which:<6}   FAIL  {p.stderr.strip()[-80:]}")
            continue
        waves = blocks * (thr // 32)
        wps = waves / SIMDS
        ops_per_simd = wps * iters * NACC
        cpo = cyc / ops_per_simd
        ops_clk_cu = 2.0 / cpo  # 2 SIMDs per CU
        tf = ops_clk_cu * FLOP[which] * args.clock_mhz * 1e6 * CUS / 1e12
        print(
            f"{label:<13}{which:<6}{wps:>11.1f}{cpo:>13.2f}"
            f"{ops_clk_cu:>12.3f}{tf:>13.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
