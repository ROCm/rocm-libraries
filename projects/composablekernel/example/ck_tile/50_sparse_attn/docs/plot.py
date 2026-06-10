#!/usr/bin/env python3
"""Plot the SpargeAttn + SageAttn comparison chart from a bench CSV.

Reads the CSV produced by run_bench.py and renders the 4-curve figure
(docs/pv_skip_mode_comparison.png). Aggregates across seeds with a median
line and interquartile band. Style follows the SpargeAttn paper figure.

Curves:
  - fmha_dense  (fp16 baseline, hline)
  - fmha_sage   (fp8 BLOCKSCALE, hline)
  - sparge_fp16 (sparse + fp16, sparsity sweep)
  - sparge_sage (sparse + int8 BLOCKSCALE Q/K, sparsity sweep, HERO)

Example:
  python3 docs/plot.py --csv docs/sparge_bench.csv --out docs/pv_skip_mode_comparison.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "sparge_fp16": dict(linestyle="--", marker="s", color="#ff7f0e",
                        label="SpargeAttn (sparse, FP16)"),
    "sparge_sage": dict(linestyle="-",  marker="o", color="#d62728",
                        label="SpargeAttn + SageAttn (sparse + INT8 Q/K)"),
    "fmha_dense":  dict(linestyle="-.", marker=None, color="#2ca02c",
                        label="Dense (FP16 baseline)"),
    "fmha_sage":   dict(linestyle="-.", marker=None, color="#9467bd",
                        label="Dense + SageAttn (FP8)"),
}
SWEEP_CURVES = ["sparge_fp16", "sparge_sage"]
HLINE_CURVES = ["fmha_dense", "fmha_sage"]
HERO = "sparge_sage"


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["curve_name"], r["topk"])].append(
            (float(r["measured_sparsity"]), float(r["tops"])))
    out = defaultdict(list)
    for (curve, _tk), pts in buckets.items():
        sps = [p[0] for p in pts]
        tps = sorted(p[1] for p in pts)
        sp_med = median(sps)
        tp_med = median(tps)
        if len(tps) >= 4:
            lo = tps[len(tps) // 4]
            hi = tps[(3 * len(tps)) // 4]
        else:
            lo, hi = min(tps), max(tps)
        out[curve].append((sp_med, tp_med, lo, hi))
    for c in out:
        out[c].sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("sparge_chart.png"))
    ap.add_argument("--title",
                    default="SpargeAttn + SageAttn on MI300X (b=2 h=16 s=8192 d=128)")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    agg = aggregate(rows)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    sparge_sps = [sp for c in SWEEP_CURVES for sp, *_ in agg.get(c, [])]
    x_lo, x_hi = (min(sparge_sps), max(sparge_sps)) if sparge_sps else (0.0, 1.0)

    # Sweep curves
    for curve in SWEEP_CURVES:
        if curve not in agg:
            continue
        pts = agg[curve]
        xs = [p[0] for p in pts]
        ymed = [p[1] for p in pts]
        ylo = [p[2] for p in pts]
        yhi = [p[3] for p in pts]
        ax.plot(xs, ymed, **STYLE[curve])
        if any(lo != hi for lo, hi in zip(ylo, yhi)):
            ax.fill_between(xs, ylo, yhi, color=STYLE[curve]["color"], alpha=0.15)

    # Horizontal baseline lines (dense, sage)
    hline_vals = {}
    for hcurve in HLINE_CURVES:
        if hcurve in agg:
            val = median(p[1] for p in agg[hcurve])
            hline_vals[hcurve] = val
            st = dict(STYLE[hcurve])
            st.pop("marker")
            ax.hlines(val, x_lo, x_hi, **st)
            short = "dense" if hcurve == "fmha_dense" else "sage"
            ax.annotate(f"{short} ~ {val:.0f} TOPS",
                        xy=(x_hi, val), xytext=(-5, 5),
                        textcoords="offset points", ha="right", fontsize=9,
                        color=STYLE[hcurve]["color"])

    # Annotate HERO speedup vs dense per sparsity point
    dense_tops = hline_vals.get("fmha_dense")
    if dense_tops and HERO in agg:
        for sp, tp, _lo, _hi in agg[HERO]:
            ax.annotate(f"{tp / dense_tops:.2f}x",
                        xy=(sp, tp), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=STYLE[HERO]["color"], fontweight="bold")

    ax.set_xlabel("Measured sparsity (1 - kept/total)")
    ax.set_ylabel("TOPS (dense FLOPs / wall)")
    ax.set_title(args.title + "\nlabels = speedup vs dense FP16", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
