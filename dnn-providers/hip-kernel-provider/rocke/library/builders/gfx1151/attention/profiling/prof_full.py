#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Comprehensive rocprof pass + cycle budget for one swapqk config at L16K.

Collects counters in several passes (gfx1151 can't take them all at once) and
prices the result against the issue costs measured by peaks.py:

    v_wmma 36.15 cyc/instr/SIMD, VALU 1.31, v_exp_f32 4.00

so the kernel's cycles can be split into WMMA issue, other-VALU issue, and
everything left over (memory stall + waitcnt + scheduling).

The dual_gather comparison is the interesting one: dual halves V REQUESTS at the
cost of ~3 broadcast VALU ops per dword. Comparing bytes (GL2C_EA_RDREQ_*) with
requests (SQ_INSTS_TEX_LOAD) across dual=0/1 says whether the V path is limited
by cache-line bytes or by request count -- which decides what to optimise.

Usage:
    python3 prof_full.py --block-n 64 --vt 1 --dual 1,0
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

SIMDS = 80
CUS = 40
# measured by peaks.py
CYC_WMMA, CYC_VALU, CYC_EXP = 36.15, 1.31, 4.00

# rocprofv3 on gfx1151 needs one hardware pass per counter group; >5 counters
# per invocation makes it replay the dispatch until it times out. Keep groups
# small and let a failed group degrade gracefully.
PASSES = [
    (
        "core",
        [
            "GRBM_GUI_ACTIVE",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "SQ_INSTS_TEX_LOAD",
            "SQ_INSTS_FLAT",
        ],
    ),
    (
        "occ",
        [
            "GRBM_GUI_ACTIVE",
            "MeanOccupancyPerCU",
            "MemUnitBusy",
            "TA_TA_BUSY",
            "VALUBusy",
        ],
    ),
    (
        "mem",
        [
            "GRBM_GUI_ACTIVE",
            "GL2C_HIT_sum",
            "GL2C_MISS_sum",
            "GL2C_EA_RDREQ_128B_sum",
            "GL2C_EA_RDREQ_32B_sum",
        ],
    ),
]


def read(outdir, names):
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
    return {k: agg.get(k, 0.0) / len(disp) for k in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--vt", type=int, default=1)
    ap.add_argument("--of16", type=int, default=0)
    ap.add_argument("--dual", default="1")
    ap.add_argument("--qkdo", default="0", help="comma list of qk_douter values")
    ap.add_argument("--bg", default="0", help="comma list of bcast_group values")
    ap.add_argument(
        "--vkb", default="0", help="key-blocked V: comma list of KB (0=full transpose)"
    )
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--clock-mhz", type=float, default=2400.0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    L, D = args.seqlen, args.head_size
    flops = 4.0 * L * L * D
    iters = L // args.block_n

    combos = [
        (d, k, q, bg)
        for d in (int(x) for x in args.dual.split(","))
        for k in (int(x) for x in args.vkb.split(","))
        for q in (int(x) for x in args.qkdo.split(","))
        for bg in (int(x) for x in args.bg.split(","))
    ]
    for dual, vkb, qkdo, bg in combos:
        got = {}
        for tag, ctrs in PASSES:
            outdir = f"/tmp/pf_{dual}_{vkb}_{qkdo}_{bg}_{tag}"
            shutil.rmtree(outdir, ignore_errors=True)
            try:
                subprocess.run(
                    [
                        "rocprofv3",
                        "--pmc",
                        *ctrs,
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
                        "24",
                        "--iters",
                        str(args.iters),
                        "--mq",
                        "1",
                        "--of16",
                        str(args.of16),
                        "--block-n",
                        str(args.block_n),
                        "--vt",
                        str(args.vt),
                        "--dual",
                        str(dual),
                        "--vkb",
                        str(vkb),
                        "--qkdo",
                        str(qkdo),
                        "--bg",
                        str(bg),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
            except subprocess.TimeoutExpired:
                print(f"  (pass {tag} timed out, skipped)")
                continue
            r = read(outdir, ctrs)
            if r:
                got.update(r)

        if "SQ_INSTS_VALU" not in got:
            print(f"dual={dual} vkb={vkb} qkdo={qkdo} bg={bg}: FAIL")
            continue

        cyc = got["GRBM_GUI_ACTIVE"]
        waves = got["SQ_WAVES"]
        valu = got["SQ_INSTS_VALU"]
        tex, flat = got["SQ_INSTS_TEX_LOAD"], got["SQ_INSTS_FLAT"]
        # WMMA count is exact from the shape: (QK + PV) tiles per iteration.
        n_dk, n_kv = D // 16, args.block_n // 16
        wmma = waves * iters * (2 * n_dk * n_kv)
        exp = waves * iters * (n_kv * 8)

        # SQ_INSTS_VALU on RDNA3 includes WMMA and transcendentals.
        other_valu = max(valu - wmma - exp, 0)
        c_wmma = wmma * CYC_WMMA / SIMDS
        c_exp = exp * CYC_EXP / SIMDS
        c_valu = other_valu * CYC_VALU / SIMDS
        c_issue = c_wmma + c_exp + c_valu
        rd = sum(got.get(f"GL2C_EA_RDREQ_{n}B_sum", 0.0) * n for n in (32, 128))
        h, m = got.get("GL2C_HIT_sum", 0), got.get("GL2C_MISS_sum", 0)

        print(
            f"\n{'=' * 72}\ndual={dual} vkb={vkb} qkdo={qkdo} bg={bg}  L={L} bn={args.block_n} "
            f"vt={args.vt} of16={args.of16}   {iters} iters, "
            f"{waves:.0f} waves ({waves / SIMDS:.1f}/SIMD)\n{'=' * 72}"
        )
        print(
            f"  cycles (GRBM_GUI_ACTIVE) {cyc / 1e6:>10.2f} M   "
            f"-> {flops / cyc * args.clock_mhz * 1e6 / 1e12:.2f} TF @ "
            f"{args.clock_mhz:.0f} MHz"
        )
        print(
            f"  occupancy                {got.get('MeanOccupancyPerCU', 0):>10.1f} waves/CU"
        )
        print(f"\n  dynamic instructions (per dispatch, millions)")
        print(f"    SQ_INSTS_VALU (incl WMMA+trans) {valu / 1e6:>9.1f}")
        print(f"    v_wmma (derived from shape)     {wmma / 1e6:>9.1f}")
        print(f"    v_exp  (derived from shape)     {exp / 1e6:>9.1f}")
        print(f"    other VALU                      {other_valu / 1e6:>9.1f}")
        print(
            f"    MUBUF loads (V gather)          {tex / 1e6:>9.1f}"
            f"   ({tex / waves / iters:.0f}/wave/iter)"
        )
        print(
            f"    FLAT loads (K + Q)              {flat / 1e6:>9.1f}"
            f"   ({flat / waves / iters:.0f}/wave/iter)"
        )
        print(f"\n  cycle budget (per SIMD, Mcyc)")
        for lbl, c in (
            ("WMMA issue", c_wmma),
            ("v_exp issue", c_exp),
            ("other VALU issue", c_valu),
        ):
            print(f"    {lbl:<22}{c / 1e6:>8.2f}  {100 * c / cyc:>5.1f}%")
        print(
            f"    {'= total issue':<22}{c_issue / 1e6:>8.2f}  "
            f"{100 * c_issue / cyc:>5.1f}%"
        )
        print(
            f"    {'unaccounted (stall)':<22}{(cyc - c_issue) / 1e6:>8.2f}  "
            f"{100 * (cyc - c_issue) / cyc:>5.1f}%"
        )
        print(f"\n  memory")
        print(
            f"    L2 hit {h / 1e6:>8.1f} M   miss {m / 1e6:>8.1f} M   "
            f"hit {100 * h / max(h + m, 1):.1f}%"
        )
        print(f"    DRAM read (EA_RDREQ)  {rd / 2**20:>8.1f} MiB")
        print(
            f"    MemUnitBusy {got.get('MemUnitBusy', 0):>6.1f}%   "
            f"TA_TA_BUSY {got.get('TA_TA_BUSY', 0) / 1e6:>8.1f} M   "
            f"VALUBusy {got.get('VALUBusy', 0):>6.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
