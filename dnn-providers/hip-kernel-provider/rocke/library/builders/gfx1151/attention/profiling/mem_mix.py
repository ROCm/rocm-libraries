#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split the vector-memory instruction stream by class, per wave per K-iteration.

SQ_INSTS_TEX_LOAD counts MUBUF (the buffer_load V gathers) but NOT the FLAT-class
global_load that K rides on, so the 8.4M figure used so far is V-only. Before
deciding what to stage in LDS we need to know which operand actually dominates
the vector-memory path now that v_transposed cut V by 8x.

Normalises to loads per wave per K-loop iteration, which is the number that
matters for a staging decision (it is what an LDS tile would replace).
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

COUNTERS = (
    "GRBM_GUI_ACTIVE",
    "SQ_INSTS_TEX_LOAD",
    "SQ_INSTS_FLAT",
    "SQ_INSTS_LDS",
    "SQ_WAVES",
)


def read(outdir):
    files = glob.glob(f"{outdir}/*/*counter_collection.csv")
    if not files:
        return None
    agg = collections.defaultdict(float)
    disp = set()
    for row in csv.DictReader(open(files[0])):
        agg[row["Counter_Name"]] += float(row["Counter_Value"])
        disp.add(row["Dispatch_Id"])
    if not disp:
        return None
    return {k: agg[k] / len(disp) for k in COUNTERS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--of16", type=int, default=0)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--vt", default="1")
    ap.add_argument("--vpf", default="0")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    iters = args.seqlen // args.block_n
    print(
        f"L={args.seqlen} bn={args.block_n} of16={args.of16}  "
        f"{iters} K-iterations/wave"
    )
    print(
        f"{'vt':>3}{'vpf':>5}{'Mcyc':>9}{'waves':>8}"
        f"{'MUBUF/it':>10}{'FLAT/it':>9}{'LDS/it':>8}{'tot/it':>8}"
    )
    print("-" * 60)

    for vt in (int(x) for x in args.vt.split(",")):
        for vpf in (int(x) for x in args.vpf.split(",")):
            outdir = f"/tmp/mm_{vt}_{vpf}"
            shutil.rmtree(outdir, ignore_errors=True)
            subprocess.run(
                [
                    "rocprofv3",
                    "--pmc",
                    *COUNTERS,
                    "-d",
                    outdir,
                    "-f",
                    "csv",
                    "--",
                    sys.executable,
                    str(here / "vt_prof.py"),
                    "--seqlen",
                    str(args.seqlen),
                    "--heads",
                    "24",
                    "--iters",
                    "3",
                    "--mq",
                    "1",
                    "--of16",
                    str(args.of16),
                    "--block-n",
                    str(args.block_n),
                    "--vt",
                    str(vt),
                    "--vpf",
                    str(vpf),
                ],
                capture_output=True,
                text=True,
                timeout=900,
            )
            got = read(outdir)
            if got is None:
                print(f"{vt:>3}{vpf:>5}     FAIL")
                continue
            w = got["SQ_WAVES"] or 1
            per = lambda k: got[k] / w / iters
            mubuf, flat, lds = (
                per("SQ_INSTS_TEX_LOAD"),
                per("SQ_INSTS_FLAT"),
                per("SQ_INSTS_LDS"),
            )
            print(
                f"{vt:>3}{vpf:>5}{got['GRBM_GUI_ACTIVE'] / 1e6:>9.2f}"
                f"{w:>8.0f}{mubuf:>10.1f}{flat:>9.1f}{lds:>8.1f}"
                f"{mubuf + flat + lds:>8.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
