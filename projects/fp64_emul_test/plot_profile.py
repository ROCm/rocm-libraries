#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
plot_profile.py — Plot fp64 emulation timing breakdown from profiling CSV(s).

Usage (single transpose):
    python3 plot_profile.py <profile.csv> [output.png]

Usage (all four transposes — 2×2 grid):
    python3 plot_profile.py <NN.csv> <NT.csv> <TN.csv> <TT.csv> [output.png]

The CSV(s) are produced by fp64EmulatedGemm when HIPBLASLT_EMULATION_PROFILE is
set.  Columns: m,n,k,num_moduli,scale_chunk_size,gemm_chunk_size,workspace_bytes,
               t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,
               t_scale_ms,t_int8_gemm_ms,t_accum_ms,t_finalize_ms,t_total_ms

run_profile_sweep.sh runs s=16 for N in {1024,..,65536} across all four
transpose combinations, producing one CSV per transpose.  This script
accepts those CSVs and arranges them in a 2×2 subplot grid.

Multiple rows appear per N (warmup + timed calls); the script drops the
first row per N and averages the rest.
X-axis = N (with workspace size in GiB shown below each tick).
"""

import os
import re
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

_TRANS_RE = re.compile(r'_([NT]{2})(?:\.csv)?$', re.IGNORECASE)


def trans_label_from_path(path):
    """Extract 'transA=X  transB=Y' from a filename ending in _NN, _NT, etc."""
    m = _TRANS_RE.search(os.path.basename(path))
    if m:
        t = m.group(1).upper()
        return f"transA={t[0]}  transB={t[1]}"
    return os.path.splitext(os.path.basename(path))[0]


def load_and_aggregate(csv_path):
    """Load a profile CSV, drop warmup row per N, return mean-aggregated table."""
    df = pd.read_csv(csv_path)
    df["N"] = df["m"]

    phase_cols = [c for c, _ in PHASES]

    for col in ("scale_chunk_size", "gemm_chunk_size", "workspace_bytes"):
        if col not in df.columns:
            df[col] = 0

    extra_cols = ["scale_chunk_size", "gemm_chunk_size", "workspace_bytes"]

    grouped = (df.groupby("N", group_keys=False)
                 .apply(lambda g: g.iloc[1:])
                 .groupby("N")[phase_cols + ["t_total_ms"] + extra_cols]
                 .mean()
                 .reset_index()
                 .sort_values("N"))
    return grouped


# Bar geometry constants — adjust to taste.
_BAR_WIDTH   = 0.4    # width of each stacked bar
_BAR_SPACING = 0.65   # centre-to-centre distance between consecutive bars


def plot_panel(ax, grouped, title):
    """Draw a single stacked-bar timing breakdown on *ax*."""
    phase_cols = [c for c, _ in PHASES]

    sizes    = grouped["N"].values
    x        = np.arange(len(sizes)) * _BAR_SPACING   # compressed x positions
    ws_bytes = grouped["workspace_bytes"].values

    labels = []
    for N, wb in zip(sizes, ws_bytes):
        if wb > 0:
            ws_gib = wb / (1 << 30)
            labels.append(f"{int(N)}\n({ws_gib:.1f} GiB)")
        else:
            labels.append(str(int(N)))

    totals = grouped["t_total_ms"].values.copy()
    totals[totals == 0] = 1.0

    bottom = np.zeros(len(sizes))
    for (col, label), color in zip(PHASES, COLORS):
        vals = grouped[col].values / totals * 100.0
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=_BAR_WIDTH)
        bottom += vals

    ax.axhline(100, color="k", linestyle="--", lw=1.2, label="100 %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("N  (workspace in GiB)", fontsize=8)
    ax.set_ylabel("% of total time", fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title}  |  s=16, phi=0.5", fontsize=9)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]

    # Separate CSV paths from the optional output path.
    csv_paths = []
    out_path  = None
    for a in args:
        if a.lower().endswith(".png") or a.lower().endswith(".pdf"):
            out_path = a
        else:
            csv_paths.append(a)

    if not csv_paths:
        print("ERROR: no CSV file specified.", file=sys.stderr)
        sys.exit(1)

    # Default output path.
    if out_path is None:
        out_path = csv_paths[0].replace(".csv", ".png")

    n_csv = len(csv_paths)

    if n_csv == 1:
        # Single-panel mode — backward-compatible behaviour.
        grouped = load_and_aggregate(csv_paths[0])
        label   = trans_label_from_path(csv_paths[0])
        fig, ax = plt.subplots(figsize=(8, 6))   # roughly square panel
        plot_panel(ax, grouped, label)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
        fig.tight_layout()
    else:
        # Multi-panel mode — 2-column grid with fixed per-panel size.
        ncols = min(2, n_csv)
        nrows = (n_csv + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(8 * ncols, 6 * nrows),  # ~square panels
                                 squeeze=False)
        fig.suptitle("fp64 emulation timing breakdown  |  s=16 moduli, phi=0.5",
                     fontsize=11)

        for idx, csv_path in enumerate(csv_paths):
            ax    = axes[idx // ncols][idx % ncols]
            grouped = load_and_aggregate(csv_path)
            label   = trans_label_from_path(csv_path)
            plot_panel(ax, grouped, label)

            # Add legend only to the first panel to avoid clutter.
            if idx == 0:
                handles, labels_leg = ax.get_legend_handles_labels()
            ax.legend().remove() if ax.get_legend() else None

        # Place a single shared legend below the top-left panel.
        axes[0][0].legend(handles, labels_leg,
                          loc="upper left", fontsize=7, ncol=2, framealpha=0.9)

        # Hide any unused subplots.
        for idx in range(n_csv, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
