#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sum rocprofv3 counter CSVs for the attention kernel in one output dir.

prof_full.py profiles a reduced single-head dispatch, which does NOT reproduce the
long-sequence cliffs seen in the real chunked-head benchmark (bcast_group and
o_f16 both look merely a few percent slow there while collapsing ~10x in
chunk_sweep). This lets rocprofv3 be pointed straight at chunk_sweep so counters
come from the shape that actually fails.

Usage: rocprofv3 --pmc ... -d DIR -f csv -- python3 chunk_sweep.py ...
       sum_counters.py DIR [kernel-substring]
"""

from __future__ import annotations

import collections
import csv
import glob
import sys


def main() -> int:
    outdir = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "fmha"

    tot: collections.Counter = collections.Counter()
    ndisp = set()
    for path in glob.glob(f"{outdir}/**/*counter_collection.csv", recursive=True):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if want not in row.get("Kernel_Name", ""):
                    continue
                tot[row["Counter_Name"]] += float(row["Counter_Value"])
                ndisp.add(row.get("Dispatch_Id", ""))
    if not tot:
        print(f"  no counters matching {want!r} in {outdir}")
        return 1

    hit = tot.get("GL2C_HIT_sum", 0.0)
    miss = tot.get("GL2C_MISS_sum", 0.0)
    rd = sum(tot.get(f"GL2C_EA_RDREQ_{n}B_sum", 0.0) * n for n in (32, 128))
    cyc = tot.get("GRBM_GUI_ACTIVE", 0.0)
    print(f"  dispatches {len(ndisp):4d}   cycles {cyc / 1e6:9.2f} M")
    if hit or miss:
        print(
            f"  L2 hit {hit / 1e6:8.2f} M   miss {miss / 1e6:8.2f} M   "
            f"hit {100 * hit / max(hit + miss, 1):5.1f}%"
        )
    if rd:
        print(f"  DRAM read {rd / 2 ** 20:10.1f} MiB")
    for name in sorted(tot):
        if name not in (
            "GRBM_GUI_ACTIVE",
            "GL2C_HIT_sum",
            "GL2C_MISS_sum",
            "GL2C_EA_RDREQ_128B_sum",
            "GL2C_EA_RDREQ_32B_sum",
        ):
            print(f"  {name:32s} {tot[name] / 1e6:12.2f} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
