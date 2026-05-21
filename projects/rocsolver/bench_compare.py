#!/usr/bin/env python3
"""Compare two rocsolver benchmark files (bench_latrd.sh / bench_sytrd.sh output).

Usage:
    python3 bench_compare.py <file1> <file2> [--output <prefix>]

Each input file is tab-separated with two columns: n and time (microseconds).
Produces two figures:
    <prefix>_abs.png   — absolute times for both files
    <prefix>_ratio.png — relative performance: time1 / time2
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    ns, ts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ns.append(int(parts[0]))
                    ts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(ns), np.array(ts)


def trim(name, maxlen=50):
    """Trim a filename to maxlen chars, keeping the tail (most informative part)."""
    return name if len(name) <= maxlen else "…" + name[-(maxlen - 1):]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file1", help="First benchmark file")
    parser.add_argument("file2", help="Second benchmark file")
    parser.add_argument("--output", default="bench_compare",
                        help="Output filename prefix (default: bench_compare)")
    args = parser.parse_args()

    label1 = os.path.basename(args.file1)
    label2 = os.path.basename(args.file2)

    ns1, ts1 = load(args.file1)
    ns2, ts2 = load(args.file2)

    # Align on common n values
    common = np.intersect1d(ns1, ns2)
    if len(common) == 0:
        print("No common n values found between the two files.", file=sys.stderr)
        sys.exit(1)

    mask1 = np.isin(ns1, common)
    mask2 = np.isin(ns2, common)
    ns  = ns1[mask1]
    t1  = ts1[mask1]
    t2  = ts2[mask2]

    # Figure 1: absolute times
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ns, t1, marker="o", markersize=3, linewidth=1.2, label=label1)
    ax.plot(ns, t2, marker="s", markersize=3, linewidth=1.2, label=label2)
    ax.set_xlabel("n")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Benchmark comparison — absolute time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    abs_path = args.output + "_abs.png"
    fig.savefig(abs_path, dpi=150)
    print(f"Saved {abs_path}")

    # Figure 2: relative performance (t1 / t2)
    ratio = t1 / t2
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(ns, ratio, marker="o", markersize=3, linewidth=1.2, color="tab:green")
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("n")
    ax2.set_ylabel(f"{label1} / {label2}")
    ax2.set_title(
        f"Relative performance  (ratio < 1 → {trim(label1)} is faster)\n"
        f"Reference (denominator): {trim(label2)}",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    ratio_path = args.output + "_ratio.png"
    fig2.savefig(ratio_path, dpi=150)
    print(f"Saved {ratio_path}")


if __name__ == "__main__":
    main()
