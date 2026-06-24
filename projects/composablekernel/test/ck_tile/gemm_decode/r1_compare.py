#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""R1 analysis: join the gemm_decode, AITER wvSplitK*, and FlyDSL small_m
M-sweeps and locate the two dispatch boundaries that bracket gemm_decode:

  - the VALU peer (AITER wvSplitK* warp-per-scalar) it claims to *subsume* at
    small M (design doc section 16), and
  - the MFMA ceiling (FlyDSL small_m_hgemm) it eventually loses to as M grows
    (section 15.C).

Reads (all optional except gemm_decode):
  - gemm_decode CSV from bench_msweep.cpp (impl gemm_decode_base/gemm_decode_best;
    columns impl,M,N,K,time_us,tflops,gbytes_s,mp,np,[kv,]swizzle,chunk)
  - AITER wvSplitK CSV from wvsplitk_msweep.py (impl=aiter_wvsplitk = per-M best
    of the wvSplitK family; columns impl,M,N,K,time_us,tflops,gbytes_s,config)
  - FlyDSL CSV from flydsl_msweep.py (impl=flydsl_small_m; same columns)

Emits a per-M comparison table (markdown) with both speedup ratios and the
winner, plus the small-M subsume verdict and the MFMA crossover M.

  ./r1_compare.py --gemm-decode-csv /tmp/r1_a4/msweep_8192_7168.csv \
                  --wvsplitk-csv    /tmp/r1_a4/wvsplitk_msweep_8192x7168.csv \
                  --flydsl-csv      /tmp/r1_a4/flydsl_msweep_8192x7168.csv
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


def _fmt(x):
    return f"{float(x):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-decode-csv", required=True)
    ap.add_argument("--wvsplitk-csv", default="")
    ap.add_argument("--flydsl-csv", default="")
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    gd_best = _by_m(_read(args.gemm_decode_csv), "gemm_decode_best")
    wv = _by_m(_read(args.wvsplitk_csv), "aiter_wvsplitk")
    fly = _by_m(_read(args.flydsl_csv), "flydsl_small_m")

    if not gd_best:
        print("No gemm_decode_best rows found.")
        return 1

    any_m = next(iter(gd_best.values()))
    N, K = any_m["N"], any_m["K"]
    Ms = [m for m in sorted(gd_best) if m <= args.mmax]

    def gcfg(g):
        kv = f"/kv{g['kv']}" if g.get("kv") else ""
        return f"mp{g.get('mp', 1)}/np{g['np']}{kv}/{g['swizzle']}/c{g['chunk']}"

    lines = []
    lines.append(f"### gemm_decode vs AITER wvSplitK* (VALU) vs FlyDSL small_m (MFMA)  "
                 f"(N={N}, K={K}, BF16, gfx950)")
    lines.append("")
    lines.append("| M | gemm_decode best (µs) | gd cfg | wvSplitK* (µs) | kern | "
                 "FlyDSL MFMA (µs) | gd/wv | gd/fly | winner |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    n_win = n_tie = n_loss = 0  # gemm_decode vs wvSplitK over M<=4
    wv_crossover = None         # first M where gd loses to wvSplitK
    fly_crossover = None        # first M where gd loses to FlyDSL

    for m in Ms:
        g = gd_best[m]
        gt = float(g["time_us"])
        w = wv.get(m)
        fl = fly.get(m)
        wt = float(w["time_us"]) if w else None
        ft = float(fl["time_us"]) if fl else None
        wkern = (w["config"].split("/")[0] if w else "—")
        gw = (wt / gt) if wt else None  # >1 => gemm_decode faster than wvSplitK
        gf = (ft / gt) if ft else None  # >1 => gemm_decode faster than FlyDSL

        # Winner across the implementations that exist at this M.
        cand = [("gemm_decode", gt)]
        if wt:
            cand.append(("wvSplitK", wt))
        if ft:
            cand.append(("FlyDSL", ft))
        winner = min(cand, key=lambda c: c[1])[0]

        if wt and m <= 4:
            if gt < wt * 0.98:
                n_win += 1
            elif gt > wt * 1.02:
                n_loss += 1
            else:
                n_tie += 1
        if wt and gt > wt * 1.02 and wv_crossover is None:
            wv_crossover = m
        if ft and gt > ft * 1.02 and fly_crossover is None:
            fly_crossover = m

        lines.append(
            f"| {m} | {gt:.2f} | {gcfg(g)} | "
            f"{_fmt(wt) if wt else '—'} | {wkern} | "
            f"{_fmt(ft) if ft else '—'} | "
            f"{(f'{gw:.2f}×') if gw else '—'} | "
            f"{(f'{gf:.2f}×') if gf else '—'} | {winner} |"
        )

    lines.append("")
    # Subsume verdict (M<=4 vs the best VALU baseline).
    if wv:
        verdict = (f"**Subsume-wvSplitK (M<=4):** gemm_decode wins {n_win}, ties "
                   f"{n_tie}, loses {n_loss} of the M<=4 cells vs the best wvSplitK* "
                   f"kernel at (N={N}, K={K}).")
        if wv_crossover is not None and wv_crossover <= 4:
            verdict += (f" wvSplitK* is faster starting at M={wv_crossover} "
                        f"(>2% margin).")
        elif n_loss == 0:
            verdict += " gemm_decode is never >2% slower across M<=4."
        lines.append(verdict)
        # wvSplitK family has a hard upper-M limit (wvSpltK is M<=4 only; the
        # M=5..16 fallback wv_splitk_small is ~5-10x slower than gemm_decode).
        if any(m >= 5 for m in wv):
            lines.append("")
            lines.append("> Note: the fast `wvSpltK` path is M<=4 only; at M>=5 the "
                         "wvSplitK family falls back to `wv_splitk_small_fp16_bf16`, "
                         "which is far slower than gemm_decode (see gd/wv at M>=5).")
    if fly:
        lines.append("")
        if fly_crossover is None:
            lines.append(f"**MFMA ceiling:** gemm_decode stays ahead of FlyDSL small_m "
                         f"across M=1..{Ms[-1]} at (N={N}, K={K}).")
        else:
            lines.append(f"**MFMA ceiling:** FlyDSL small_m (MFMA) overtakes gemm_decode "
                         f"at **M={fly_crossover}** (N={N}, K={K}); below that "
                         f"warp-per-scalar gemm_decode wins.")

    out = "\n".join(lines)
    print(out)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(out + "\n")
        print(f"\n# wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
