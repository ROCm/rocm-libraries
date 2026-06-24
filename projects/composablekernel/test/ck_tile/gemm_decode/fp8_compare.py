#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
r"""FP8 per-tensor analysis: join gemm_decode against its three FP8 competitors
and locate the dispatch boundaries that bracket it.

  - the VALU peer it claims to subsume at small M:
        AITER wvSplitKQ            (impl=aiter_wvsplitkq)
  - the two MFMA per-tensor ceilings it should beat until M grows large:
        AITER gemm_a8w8_CK         (impl=aiter_gemm_a8w8_ck; classic-CK, not
                                    locked to a 16-row tile)
        CKTile gemm_quant tensor   (impl=ck_gemm_quant_tensor; M_Warp_Tile=16,
                                    the hard 16-row MFMA-tile ceiling)

Reads (gemm_decode required; others optional):
  --gemm-decode-csv  bench_msweep_fp8.cpp  (impl gemm_decode_fp8_best/_base;
                     cols impl,M,N,K,time_us,tflops,gbytes_s,mp,np,kv,kb,swizzle,chunk)
  --aiter-csv        wvsplitk_msweep.py --fp8 (impls aiter_wvsplitkq,
                     aiter_gemm_a8w8_ck, aiter_fp8; cols ...,config)
  --ckquant-csv      gemm_quant_tensor_msweep.py (impl ck_gemm_quant_tensor)

Emits a per-M markdown table (speedup ratios + winner), the M<=4 subsume
verdict vs the VALU peer, and the crossover M against each MFMA ceiling.

  ./fp8_compare.py --gemm-decode-csv /tmp/fp8/gd_8192_7168.csv \
                   --aiter-csv       /tmp/fp8/aiter_8192_7168.csv \
                   --ckquant-csv     /tmp/fp8/ckq_8192_7168.csv
"""

from __future__ import annotations

import argparse
import csv


def _read(path):
    if not path:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _by_m(rows, impl):
    return {int(r["M"]): r for r in rows if r["impl"] == impl}


def _us(row):
    return float(row["time_us"]) if row else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-decode-csv", required=True)
    ap.add_argument("--aiter-csv", default="")
    ap.add_argument("--ckquant-csv", default="")
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="Relative margin treated as a tie (default 2%).")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    gd = _by_m(_read(args.gemm_decode_csv), "gemm_decode_fp8_best")
    aiter_rows = _read(args.aiter_csv)
    wvq = _by_m(aiter_rows, "aiter_wvsplitkq")
    a8w8 = _by_m(aiter_rows, "aiter_gemm_a8w8_ck")
    ckq = _by_m(_read(args.ckquant_csv), "ck_gemm_quant_tensor")

    if not gd:
        print("No gemm_decode_fp8_best rows found.")
        return 1

    any_m = next(iter(gd.values()))
    N, K = any_m["N"], any_m["K"]
    Ms = [m for m in sorted(gd) if m <= args.mmax]

    def gcfg(g):
        return (f"mp{g.get('mp', 1)}/np{g['np']}/v{g.get('kv', 16)}/"
                f"kb{g.get('kb', 1)}/{g['swizzle']}/c{g['chunk']}")

    lines = []
    lines.append(f"### gemm_decode (FP8 per-tensor) vs VALU peer & MFMA ceilings  "
                 f"(N={N}, K={K}, FP8 e4m3, gfx950)")
    lines.append("")
    lines.append("| M | gemm_decode (µs) | gd cfg | wvSplitKQ (µs) | gemm_a8w8_CK (µs) | "
                 "ck_quant_tensor (µs) | gd/wvq | gd/a8w8 | gd/ckq | winner |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    n_win = n_tie = n_loss = 0   # gemm_decode vs wvSplitKQ over M<=4 (the charter)
    wvq_cross = a8w8_cross = ckq_cross = None  # first M where gd loses (> tol)

    def ratio(other_t, gt):
        return (other_t / gt) if (other_t and gt) else None

    for m in Ms:
        g = gd[m]
        gt = float(g["time_us"])
        wt = _us(wvq.get(m))
        at = _us(a8w8.get(m))
        ct = _us(ckq.get(m))

        cand = [("gemm_decode", gt)]
        if wt:
            cand.append(("wvSplitKQ", wt))
        if at:
            cand.append(("gemm_a8w8_CK", at))
        if ct:
            cand.append(("ck_quant_tensor", ct))
        winner = min(cand, key=lambda c: c[1])[0]

        if wt and m <= 4:
            if gt < wt * (1 - args.tol):
                n_win += 1
            elif gt > wt * (1 + args.tol):
                n_loss += 1
            else:
                n_tie += 1
        if wt and gt > wt * (1 + args.tol) and wvq_cross is None:
            wvq_cross = m
        if at and gt > at * (1 + args.tol) and a8w8_cross is None:
            a8w8_cross = m
        if ct and gt > ct * (1 + args.tol) and ckq_cross is None:
            ckq_cross = m

        def cell(x):
            return f"{x:.2f}" if x else "—"

        def rcell(x):
            r = ratio(x, gt)
            return f"{r:.2f}×" if r else "—"

        lines.append(
            f"| {m} | {gt:.2f} | {gcfg(g)} | {cell(wt)} | {cell(at)} | {cell(ct)} | "
            f"{rcell(wt)} | {rcell(at)} | {rcell(ct)} | {winner} |"
        )

    lines.append("")
    if wvq:
        verdict = (f"**Subsume-wvSplitKQ (M≤4):** gemm_decode wins {n_win}, ties "
                   f"{n_tie}, loses {n_loss} of the M≤4 cells vs the AITER VALU peer "
                   f"at (N={N}, K={K}).")
        if wvq_cross is not None and wvq_cross <= 4:
            verdict += f" wvSplitKQ pulls ahead at M={wvq_cross} (>{args.tol:.0%})."
        elif n_loss == 0:
            verdict += f" gemm_decode is never >{args.tol:.0%} slower across M≤4."
        lines.append(verdict)

    ceil_notes = []
    for name, cross, present in (("gemm_a8w8_CK", a8w8_cross, a8w8),
                                 ("ck_quant_tensor", ckq_cross, ckq)):
        if not present:
            continue
        if cross is None:
            ceil_notes.append(f"gemm_decode stays ahead of **{name}** across "
                              f"M=1..{Ms[-1]}")
        else:
            ceil_notes.append(f"**{name}** overtakes gemm_decode at **M={cross}**")
    if ceil_notes:
        lines.append("")
        lines.append("**MFMA ceilings:** " + "; ".join(ceil_notes) +
                     f" (N={N}, K={K}).")
        if ckq:
            lines.append("")
            lines.append("> `ck_quant_tensor` is locked to a 16-row MMA tile "
                         "(M_Warp_Tile=16), so its µs is ~flat across M=1..8 — the "
                         "fixed-cost MFMA ceiling gemm_decode undercuts at small M.")

    out = "\n".join(lines)
    print(out)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(out + "\n")
        print(f"\n# wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
