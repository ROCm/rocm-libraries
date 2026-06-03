#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""R1 analysis: join the gemm_decode and FlyDSL small_m M-sweeps and locate the
warp-per-scalar <-> MFMA dispatch boundary.

Reads:
  - gemm_decode CSV from bench_msweep.cpp (impl in {gemm_decode_base,
    gemm_decode_best}; columns
    impl,M,N,K,time_us,tflops,gbytes_s,mp,np,swizzle,chunk)
  - FlyDSL CSV from flydsl_msweep.py (impl=flydsl_small_m; columns
    impl,M,N,K,time_us,tflops,gbytes_s,config)

Emits a per-M comparison table (markdown) and the crossover M (first M where
FlyDSL small_m becomes faster than gemm_decode_best), if any within range.

  ./r1_compare.py --gemm-decode-csv /tmp/gemm_decode_msweep_8192x7168.csv \
                  --flydsl-csv      /tmp/flydsl_msweep_8192x7168.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-decode-csv", required=True)
    ap.add_argument("--flydsl-csv", required=True)
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    gd = _read(args.gemm_decode_csv)
    fd = _read(args.flydsl_csv)

    # index by M
    gd_best = {int(r["M"]): r for r in gd if r["impl"] == "gemm_decode_best"}
    gd_base = {int(r["M"]): r for r in gd if r["impl"] == "gemm_decode_base"}
    fd_best = {int(r["M"]): r for r in fd if r["impl"] == "flydsl_small_m"}

    Ms = sorted(set(gd_best) & set(fd_best))
    if not Ms:
        print("No overlapping M between the two CSVs.")
        return 1

    N = gd_best[Ms[0]]["N"]
    K = gd_best[Ms[0]]["K"]

    lines = []
    lines.append(f"### gemm_decode vs FlyDSL small_m  (N={N}, K={K}, BF16, gfx950)")
    lines.append("")
    lines.append("| M | gemm_decode best (µs) | cfg | TF/s | FlyDSL small_m best (µs) | cfg | TF/s | "
                 "speedup gd/fly | winner |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    crossover = None
    for m in Ms:
        g = gd_best[m]
        f = fd_best[m]
        gt = float(g["time_us"])
        ft = float(f["time_us"])
        gcfg = f"mp{g.get('mp', 1)}/np{g['np']}/{g['swizzle']}/c{g['chunk']}"
        spd = ft / gt  # >1 => gemm_decode faster
        winner = "gemm_decode" if gt <= ft else "flydsl"
        if winner == "flydsl" and crossover is None:
            crossover = m
        lines.append(
            f"| {m} | {gt:.2f} | {gcfg} | {float(g['tflops']):.2f} | "
            f"{ft:.2f} | {f['config']} | {float(f['tflops']):.2f} | "
            f"{spd:.2f}× | {winner} |"
        )

    lines.append("")
    if crossover is None:
        lines.append(f"**Dispatch boundary:** gemm_decode (warp-per-scalar) is fastest "
                     f"across the entire swept range M=1..{Ms[-1]} at (N={N}, K={K}). "
                     f"FlyDSL small_m never overtakes within range.")
    else:
        lines.append(f"**Dispatch boundary:** FlyDSL small_m (MFMA) first overtakes "
                     f"gemm_decode at **M={crossover}** (N={N}, K={K}); below that, "
                     f"warp-per-scalar wins.")

    out = "\n".join(lines)
    print(out)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(out + "\n")
        print(f"\n# wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
