#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
plot_profile.py — Plot fp64 emulation timing breakdown from a profiling CSV.

Usage:
    python3 plot_profile.py <profile.csv> [output.png]

The CSV is produced by fp64EmulatedGemm when HIPBLASLT_EMULATION_PROFILE is set.
Columns: m,n,k,num_moduli,chunk_size,
         t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,
         t_scale_ms,t_int8_gemm_ms,t_accum_ms,t_finalize_ms,t_total_ms

run_profile_sweep.sh runs with s=16 for N in {1024,2048,4096,8192,16384}.
Multiple rows appear per N (warmup + timed calls); the script averages them.
X-axis = N, stacked bars = phase breakdown, dashed line = total time.
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASES = [
    ("t_prelim_ms",      "prelim extract"),
    ("t_prelim_gemm_ms", "prelim GEMM"),
    ("t_refine_ms",      "shift refine"),
    ("t_scale_ms",       "scale A+B"),
    ("t_int8_gemm_ms",   "INT8 GEMM"),
    ("t_accum_ms",       "CRT accum"),
    ("t_finalize_ms",    "finalize"),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#937860", "#DA8BC3"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", ".png")

    df = pd.read_csv(csv_path)
    df["N"] = df["m"]

    phase_cols = [c for c, _ in PHASES]
    # Drop the first row per N (warmup call), average the rest.
    grouped = (df.groupby("N", group_keys=False)
                 .apply(lambda g: g.iloc[1:])
                 .groupby("N")[phase_cols + ["t_total_ms"]]
                 .mean()
                 .reset_index()
                 .sort_values("N"))

    sizes  = grouped["N"].values
    x      = np.arange(len(sizes))
    labels = [str(int(N)) for N in sizes]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(sizes)), 5))

    totals = grouped["t_total_ms"].values.copy()
    totals[totals == 0] = 1.0   # guard against division by zero

    bottom = np.zeros(len(sizes))
    for (col, label), color in zip(PHASES, COLORS):
        vals = grouped[col].values / totals * 100.0
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.6)
        bottom += vals

    # 100 % reference line
    ax.axhline(100, color="k", linestyle="--", lw=1.2, label="100 %")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N (square matrix size)")
    ax.set_ylabel("% of total time")
    ax.set_ylim(0, 110)
    ax.set_title("fp64 emulation timing breakdown  |  s=16 moduli, phi=0.5")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
