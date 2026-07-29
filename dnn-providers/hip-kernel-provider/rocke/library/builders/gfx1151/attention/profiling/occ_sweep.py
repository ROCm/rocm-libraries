#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Occupancy / register-pressure sweep for the swapqk kernel at long Sq.

L16K is memory-LATENCY bound (see v_kblock in the kernel: 8x fewer V loads and
8x fewer cache lines both leave TA_TA_BUSY untouched). The lever for a
latency-bound kernel is more memory parallelism in flight, i.e. more resident
waves. This sweeps the ``waves_per_eu`` hint, which forces the compiler to fit
VGPRs for N waves/EU, and reports what each setting actually buys:

  * VGPR / spill from the compiled object (no GPU needed)
  * MeanOccupancyPerCU  (waves actually resident)
  * GRBM_GUI_ACTIVE     (clock-independent runtime)
  * issue rate          (instructions per wave-cycle -- how much of a resident
                         wave's life is spent issuing rather than waiting)

Usage:
    python3 occ_sweep.py --seqlen 16384 --reps 3
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

from rocke.helpers import compile_kernel

from kernels.gfx1151.wmma_fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk

# gfx1151 (RDNA3.5) wave32: VGPRs are allocated in blocks of 16 out of a
# 1536-register file per SIMD, and a SIMD holds at most 16 waves.
VGPR_FILE = 1536
VGPR_GRAN = 16
MAX_WAVES_PER_SIMD = 16


def _resource_counts(hsaco: bytes):
    raw = bytes(hsaco)

    def after(key):
        i = raw.find(key.encode())
        if i < 0:
            return None
        j = i + len(key)
        b0 = raw[j]
        if b0 < 0x80:
            return b0
        if b0 == 0xCC:
            return raw[j + 1]
        if b0 == 0xCD:
            return int.from_bytes(raw[j + 1 : j + 3], "big")
        if b0 == 0xCE:
            return int.from_bytes(raw[j + 1 : j + 5], "big")
        return None

    return {
        "vgpr": after(".vgpr_count"),
        "spill": after(".vgpr_spill_count"),
    }


def vgpr_waves(vgpr: int) -> int:
    alloc = ((vgpr + VGPR_GRAN - 1) // VGPR_GRAN) * VGPR_GRAN
    return min(MAX_WAVES_PER_SIMD, VGPR_FILE // max(alloc, 1))


def read_counters(outdir: str, names):
    files = glob.glob(f"{outdir}/*/*counter_collection.csv")
    if not files:
        return None
    agg = collections.defaultdict(float)
    dur = {}
    for row in csv.DictReader(open(files[0])):
        agg[row["Counter_Name"]] += float(row["Counter_Value"])
        dur[row["Dispatch_Id"]] = 1
    if not dur:
        return None
    n = len(dur)
    return {k: agg[k] / n for k in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--vt", type=int, default=1)
    ap.add_argument("--of16", type=int, default=0)
    ap.add_argument("--dual", default="1", help="comma list of dual_gather values")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--clock-mhz", type=float, default=2400.0)
    ap.add_argument("--wpe", default="0")
    ap.add_argument("--vpf", default="0")
    args = ap.parse_args()

    L, D = args.seqlen, args.head_size
    flops = 4.0 * L * L * D
    here = Path(__file__).resolve().parent
    grid = [
        (w, v, d)
        for w in (int(x) for x in args.wpe.split(","))
        for v in (int(x) for x in args.vpf.split(","))
        for d in (int(x) for x in args.dual.split(","))
    ]

    print(
        f"L={L} D={D} bn={args.block_n} vt={args.vt} of16={args.of16}  "
        f"one-head dispatch = {flops / 1e9:.1f} GFLOP\n"
    )
    print(
        f"{'dual':>5}{'vpf':>5}{'vgpr':>6}{'spill':>6}{'w/SIMD':>8}{'occ/CU':>8}"
        f"{'Mcyc':>9}{'TF@2.4':>8}{'Mvalu':>9}{'Mtex':>8}"
    )
    print("-" * 65)

    for wpe, vpf, dual in grid:
        cfg = SwapQKCfg(
            head_size=D,
            num_query_heads=args.heads,
            num_kv_heads=0,
            mask_mode="none",
            n_waves=2,
            q_block=1,
            o_f16=bool(args.of16),
            block_n=args.block_n,
            qk_ilp=2,
            sched_mode="pingpong",
            buffer_gather=True,
            dual_gather=bool(dual),
            fast_exp2=True,
            v_transposed=bool(args.vt),
            waves_per_eu=(wpe or None),
            v_prefetch=vpf,
        )
        art = compile_kernel(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
        )
        res = _resource_counts(art.hsaco)
        vgpr = res["vgpr"] or 0

        best = None
        for r in range(args.reps):
            outdir = f"/tmp/occ_{wpe}_{vpf}_{dual}_{r}"
            shutil.rmtree(outdir, ignore_errors=True)
            cmd = [
                "rocprofv3",
                "--pmc",
                "GRBM_GUI_ACTIVE",
                "MeanOccupancyPerCU",
                "SQ_WAVE_CYCLES",
                "SQ_INSTS_VALU",
                "SQ_INSTS_TEX_LOAD",
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
                "--mq",
                "1",
                "--of16",
                str(args.of16),
                "--block-n",
                str(args.block_n),
                "--vt",
                str(args.vt),
                "--wpe",
                str(wpe),
                "--vpf",
                str(vpf),
                "--dual",
                str(dual),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            got = read_counters(
                outdir,
                (
                    "GRBM_GUI_ACTIVE",
                    "MeanOccupancyPerCU",
                    "SQ_WAVE_CYCLES",
                    "SQ_INSTS_VALU",
                    "SQ_INSTS_TEX_LOAD",
                ),
            )
            if got and (
                best is None or got["GRBM_GUI_ACTIVE"] < best["GRBM_GUI_ACTIVE"]
            ):
                best = got
        if best is None:
            print(f"{dual:>5}{vpf:>5}{vgpr:>6}{res['spill'] or 0:>6}   FAIL")
            continue

        cyc = best["GRBM_GUI_ACTIVE"]
        tf = flops / cyc * args.clock_mhz * 1e6 / 1e12
        # SQ_WAVE_CYCLES is in quad-cycles; instructions per wave-cycle shows how
        # much of a resident wave's life is issue vs wait.
        wcyc = best["SQ_WAVE_CYCLES"] * 4
        inst = best["SQ_INSTS_VALU"] + best["SQ_INSTS_TEX_LOAD"]
        ipw = inst / wcyc if wcyc else 0
        print(
            f"{dual:>5}{vpf:>5}{vgpr:>6}{res['spill'] or 0:>6}{vgpr_waves(vgpr):>8}"
            f"{best['MeanOccupancyPerCU']:>8.1f}{cyc / 1e6:>9.2f}{tf:>8.2f}"
            f"{best['SQ_INSTS_VALU'] / 1e6:>9.1f}"
            f"{best['SQ_INSTS_TEX_LOAD'] / 1e6:>8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
