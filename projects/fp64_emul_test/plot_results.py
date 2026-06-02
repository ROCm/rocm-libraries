#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
plot_results.py — Parse and plot fp64_emul_accuracy CSV output.

Usage:
    python3 plot_results.py results.csv [output_prefix]

Generates three PDF files:
    <prefix>_accuracy_vs_N.pdf   — max relative error vs N (per phi)
    <prefix>_runtime_vs_N.pdf    — ms/run vs N (all algos)
    <prefix>_accuracy_vs_s.pdf   — max relative error vs num_moduli (per phi)

Requirements:
    pip install pandas matplotlib numpy
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ── Machine epsilon for FP64 ──────────────────────────────────────────────────
EPS_FP64 = 2.220446049250313e-16

# ── Representative s values to show in "accuracy vs N" and TFlop/s plots ────
HIGHLIGHT_S = [2, 4, 7, 10, 14, 16, 20]

# ── Colours for num_moduli (2..20) ───────────────────────────────────────────
S_MAX = 20
S_CMAP = cm.get_cmap("plasma", S_MAX + 1)

def s_color(s):
    return S_CMAP(s / S_MAX)

# ── Colours / markers for phi values ─────────────────────────────────────────
PHI_MARKERS = ["o", "s", "^", "D"]
PHI_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def phi_style(phi_idx):
    return dict(color=PHI_COLORS[phi_idx % len(PHI_COLORS)],
                marker=PHI_MARKERS[phi_idx % len(PHI_MARKERS)])

# ── Parse algo column ─────────────────────────────────────────────────────────
_RE_S = re.compile(r"OS2-accu-s(\d+)")

def parse_algo(algo):
    """Returns ('DGEMM', None), ('OS2-accu-adaptive', None), or ('OS2-accu', s_int)."""
    if algo == "DGEMM":
        return "DGEMM", None
    if algo == "OS2-accu-adaptive":
        return "OS2-accu-adaptive", None
    m = _RE_S.match(algo)
    if m:
        return "OS2-accu", int(m.group(1))
    return algo, None

# ── Load data ─────────────────────────────────────────────────────────────────
def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Parse algo
    parsed      = df["algo"].apply(parse_algo)
    df["method"] = parsed.apply(lambda x: x[0])
    df["s"]      = parsed.apply(lambda x: x[1]).astype("Int64")  # nullable int

    df["N"]       = df["N"].astype(int)
    df["phi"]     = df["phi"].astype(float)
    df["err_max"] = df["err_max"].astype(float)
    df["err_med"] = df["err_med"].astype(float)
    df["ms_per_run"] = df["ms_per_run"].astype(float)

    return df

# ── Figure 1: max relative error vs N, one subplot per phi ───────────────────
def plot_accuracy_vs_N(df, out_path):
    phi_vals = sorted(df["phi"].unique())
    n_phi    = len(phi_vals)
    ncols    = min(2, n_phi)
    nrows    = (n_phi + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle("Max relative error vs matrix size N\n"
                 "(reference: double-double GEMM, ~106-bit mantissa)",
                 fontsize=12)

    N_vals = sorted(df["N"].unique())

    for ax_idx, phi in enumerate(phi_vals):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[df["phi"] == phi]

        # Native DGEMM — single dashed line
        dgemm = sub[sub["method"] == "DGEMM"].sort_values("N")
        if not dgemm.empty:
            ax.semilogy(dgemm["N"], dgemm["err_max"],
                        color="black", linestyle="--", linewidth=1.8,
                        marker="x", markersize=6, label="DGEMM (native)")

        # Adaptive OS2-accu — default settings, s selected per call
        adaptive = sub[sub["method"] == "OS2-accu-adaptive"].sort_values("N")
        if not adaptive.empty:
            ax.semilogy(adaptive["N"], adaptive["err_max"],
                        color="crimson", linestyle="-.", linewidth=2.0,
                        marker="*", markersize=9,
                        label="OS2-accu-adaptive (s≤16, default)")

        # Emulation lines for highlighted s values
        for s in HIGHLIGHT_S:
            emul = sub[(sub["method"] == "OS2-accu") & (sub["s"] == s)].sort_values("N")
            if emul.empty:
                continue
            ax.semilogy(emul["N"], emul["err_max"],
                        color=s_color(s), linewidth=1.4,
                        marker="o", markersize=5,
                        label=f"OS2-accu s={s} ({emul['crt_bits'].iloc[0]:.0f} bits)")

        # Machine epsilon reference
        ax.axhline(EPS_FP64, color="gray", linestyle=":", linewidth=0.9,
                   label=r"$\varepsilon_{64}$ = 2.2×10⁻¹⁶")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N  (M = N = K = N)")
        ax.set_ylabel("max relative error")
        ax.set_title(f"phi = {phi}")
        ax.set_xticks(N_vals)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(loc="upper left", framealpha=0.85)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Hide unused subplots
    for ax_idx in range(n_phi, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Figure 2: runtime vs N ────────────────────────────────────────────────────
def plot_runtime_vs_N(df, out_path):
    """One line per algo, aggregated over phi (mean across phi values)."""
    N_vals = sorted(df["N"].unique())

    # Average runtime over phi (runtime shouldn't depend on phi, but average
    # anyway to smooth out any noise)
    agg = df.groupby(["method", "s", "N", "crt_bits"], dropna=False)["ms_per_run"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Runtime vs N  (mean over phi values)")

    # Native DGEMM
    dgemm = agg[agg["method"] == "DGEMM"].sort_values("N")
    if not dgemm.empty:
        ax.loglog(dgemm["N"], dgemm["ms_per_run"],
                  color="black", linestyle="--", linewidth=2,
                  marker="x", markersize=7, label="DGEMM (native)")

    # Adaptive OS2-accu — default settings
    adaptive = agg[agg["method"] == "OS2-accu-adaptive"].sort_values("N")
    if not adaptive.empty:
        ax.loglog(adaptive["N"], adaptive["ms_per_run"],
                  color="crimson", linestyle="-.", linewidth=2.2,
                  marker="*", markersize=8,
                  label="OS2-accu-adaptive (s≤16, default)")

    # All OS2-accu-s* lines, coloured by s
    s_vals = sorted(agg[agg["method"] == "OS2-accu"]["s"].dropna().unique())
    for s in s_vals:
        emul = agg[(agg["method"] == "OS2-accu") & (agg["s"] == s)].sort_values("N")
        if emul.empty:
            continue
        bits = emul["crt_bits"].iloc[0]
        ax.loglog(emul["N"], emul["ms_per_run"],
                  color=s_color(s), linewidth=1.2, alpha=0.85,
                  marker="o", markersize=4,
                  label=f"s={s}  ({bits:.0f} bits)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N  (M = N = K = N)")
    ax.set_ylabel("time per GEMM call  (ms)")
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Colourbar-style legend: split into two columns
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left",
              ncol=2, fontsize=7.5, framealpha=0.85)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Figure 3: max relative error vs num_moduli (accuracy/cost tradeoff) ───────
def plot_accuracy_vs_s(df, out_path):
    """One subplot per phi; one line per N; x-axis = num_moduli."""
    phi_vals = sorted(df["phi"].unique())
    n_phi    = len(phi_vals)
    ncols    = min(2, n_phi)
    nrows    = (n_phi + ncols - 1) // ncols

    N_vals = sorted(df["N"].unique())
    N_cmap = cm.get_cmap("viridis", len(N_vals) + 1)
    N_color = {N: N_cmap(i / len(N_vals)) for i, N in enumerate(N_vals)}

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle("Max relative error vs num_moduli (s)\n"
                 "(accuracy / cost tradeoff)",
                 fontsize=12)

    # Shared legend elements
    legend_n = [Line2D([0], [0], color=N_color[N], linewidth=1.5,
                       marker="o", markersize=5, label=f"N={N}")
                for N in N_vals]

    for ax_idx, phi in enumerate(phi_vals):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[(df["phi"] == phi) & (df["method"] == "OS2-accu")]

        for N in N_vals:
            row = sub[sub["N"] == N].sort_values("s")
            if row.empty:
                continue
            ax.semilogy(row["s"], row["err_max"],
                        color=N_color[N], linewidth=1.4,
                        marker="o", markersize=5)

        ax.axhline(EPS_FP64, color="gray", linestyle=":", linewidth=0.9,
                   label=r"$\varepsilon_{64}$")

        # Also mark the DGEMM error level for reference (use a representative N)
        dgemm_sub = df[(df["phi"] == phi) & (df["method"] == "DGEMM")]
        if not dgemm_sub.empty:
            # Use largest N as reference
            ref_N = dgemm_sub["N"].max()
            dgemm_err = dgemm_sub[dgemm_sub["N"] == ref_N]["err_max"].values
            if len(dgemm_err) > 0:
                ax.axhline(dgemm_err[0], color="black", linestyle="--",
                           linewidth=0.9, alpha=0.7,
                           label=f"DGEMM (N={ref_N})")

        # Horizontal reference lines: max accuracy achieved by the adaptive run,
        # one per N, in the same colour as the corresponding fixed-s curve.
        # The intersection of each coloured curve with its dashed peer marks
        # the minimum fixed s that matches the adaptive accuracy for that N.
        adaptive_sub = df[(df["phi"] == phi) & (df["method"] == "OS2-accu-adaptive")]
        for N in N_vals:
            arow = adaptive_sub[adaptive_sub["N"] == N]
            if arow.empty:
                continue
            ax.axhline(arow["err_max"].values[0],
                       color=N_color[N], linestyle="--", linewidth=1.2, alpha=0.75)

        ax.set_xlabel("num_moduli  (s)")
        ax.set_ylabel("max relative error")
        ax.set_title(f"phi = {phi}")
        ax.set_xticks(range(2, 21))
        ax.legend(handles=legend_n + [
            Line2D([0], [0], color="gray", linestyle=":", linewidth=0.9,
                   label=r"$\varepsilon_{64}$"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=0.9,
                   label="DGEMM level"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
                   label="OS2-accu-adaptive (dashed)"),
        ], loc="upper right", framealpha=0.85)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Hide unused subplots
    for ax_idx in range(n_phi, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Figure: TFlop/s vs N ─────────────────────────────────────────────────────
def plot_tflops_vs_N(df, out_path):
    """Effective throughput in TFlop/s vs N for DGEMM and selected s values.
    TFlop/s = 2*N^3 / (ms_per_run * 1e-3) / 1e12 = 2*N^3 / (ms_per_run * 1e9).
    Aggregated as mean over phi values.  Linear y-axis, log2 x-axis."""
    N_vals = sorted(df["N"].unique())

    # Compute TFlop/s per row
    df = df.copy()
    df["tflops"] = 2.0 * df["N"].astype(float)**3 / (df["ms_per_run"] * 1e9)

    # Average over phi
    agg = df.groupby(["method", "s", "N", "crt_bits"], dropna=False)["tflops"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Effective throughput vs N  (mean over phi values)")

    # Native DGEMM
    dgemm = agg[agg["method"] == "DGEMM"].sort_values("N")
    if not dgemm.empty:
        ax.plot(dgemm["N"], dgemm["tflops"],
                color="black", linestyle="--", linewidth=2,
                marker="x", markersize=7, label="DGEMM (native)")

    # Adaptive OS2-accu — default settings
    adaptive = agg[agg["method"] == "OS2-accu-adaptive"].sort_values("N")
    if not adaptive.empty:
        ax.plot(adaptive["N"], adaptive["tflops"],
                color="crimson", linestyle="-.", linewidth=2.2,
                marker="*", markersize=8,
                label="OS2-accu-adaptive (s≤16, default)")

    # Selected OS2-accu-s* lines
    for s in HIGHLIGHT_S:
        emul = agg[(agg["method"] == "OS2-accu") & (agg["s"] == s)].sort_values("N")
        if emul.empty:
            continue
        bits = emul["crt_bits"].iloc[0]
        ax.plot(emul["N"], emul["tflops"],
                color=s_color(s), linewidth=1.4,
                marker="o", markersize=5,
                label=f"s={s}  ({bits:.0f} bits)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N  (M = N = K = N)")
    ax.set_ylabel("TFlop/s")
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left",
              ncol=2, fontsize=7.5, framealpha=0.85)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Bonus: median error vs s ──────────────────────────────────────────────────
def plot_median_accuracy_vs_s(df, out_path):
    """Same as Figure 3 but using err_med instead of err_max."""
    phi_vals = sorted(df["phi"].unique())
    n_phi    = len(phi_vals)
    ncols    = min(2, n_phi)
    nrows    = (n_phi + ncols - 1) // ncols

    N_vals = sorted(df["N"].unique())
    N_cmap = cm.get_cmap("viridis", len(N_vals) + 1)
    N_color = {N: N_cmap(i / len(N_vals)) for i, N in enumerate(N_vals)}

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle("Median relative error vs num_moduli (s)", fontsize=12)

    legend_n = [Line2D([0], [0], color=N_color[N], linewidth=1.5,
                       marker="o", markersize=5, label=f"N={N}")
                for N in N_vals]

    for ax_idx, phi in enumerate(phi_vals):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[(df["phi"] == phi) & (df["method"] == "OS2-accu")]

        for N in N_vals:
            row = sub[sub["N"] == N].sort_values("s")
            if row.empty:
                continue
            ax.semilogy(row["s"], row["err_med"],
                        color=N_color[N], linewidth=1.4,
                        marker="o", markersize=5)

        ax.axhline(EPS_FP64, color="gray", linestyle=":", linewidth=0.9,
                   label=r"$\varepsilon_{64}$")

        # DGEMM median error reference
        dgemm_sub = df[(df["phi"] == phi) & (df["method"] == "DGEMM")]
        if not dgemm_sub.empty:
            ref_N = dgemm_sub["N"].max()
            dgemm_err_med = dgemm_sub[dgemm_sub["N"] == ref_N]["err_med"].values
            if len(dgemm_err_med) > 0:
                ax.axhline(dgemm_err_med[0], color="black", linestyle="--",
                           linewidth=0.9, alpha=0.7,
                           label=f"DGEMM median (N={ref_N})")

        # Horizontal reference lines: median accuracy achieved by the adaptive run.
        adaptive_sub = df[(df["phi"] == phi) & (df["method"] == "OS2-accu-adaptive")]
        for N in N_vals:
            arow = adaptive_sub[adaptive_sub["N"] == N]
            if arow.empty:
                continue
            ax.axhline(arow["err_med"].values[0],
                       color=N_color[N], linestyle="--", linewidth=1.2, alpha=0.75)

        ax.set_xlabel("num_moduli  (s)")
        ax.set_ylabel("median relative error")
        ax.set_title(f"phi = {phi}")
        ax.set_xticks(range(2, 21))
        ax.legend(handles=legend_n + [
            Line2D([0], [0], color="gray", linestyle=":", linewidth=0.9,
                   label=r"$\varepsilon_{64}$"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=0.9,
                   label="DGEMM median"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
                   label="OS2-accu-adaptive (dashed)"),
        ], loc="upper right", framealpha=0.85)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    for ax_idx in range(n_phi, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Output prefix: strip .csv extension, or use argument 2
    if len(sys.argv) >= 3:
        prefix = sys.argv[2]
    else:
        prefix = os.path.splitext(csv_path)[0]

    print(f"Reading: {csv_path}")
    df = load(csv_path)

    print(f"  Rows   : {len(df)}")
    print(f"  N vals : {sorted(df['N'].unique())}")
    print(f"  phi    : {sorted(df['phi'].unique())}")
    print(f"  algos  : {sorted(df['algo'].unique())}")
    print()

    print("Generating plots ...")
    plot_accuracy_vs_N(df,        f"{prefix}_accuracy_vs_N.pdf")
    plot_runtime_vs_N(df,         f"{prefix}_runtime_vs_N.pdf")
    plot_tflops_vs_N(df,          f"{prefix}_tflops_vs_N.pdf")
    plot_accuracy_vs_s(df,        f"{prefix}_accuracy_vs_s.pdf")
    plot_median_accuracy_vs_s(df, f"{prefix}_median_accuracy_vs_s.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
