# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render layout slicing visually.

Three panels:
  1. The full 4x8 column-major layout L — each cell labeled with its
     offset L[r, c]; column 3 highlighted with a red border.
  2. The kept-mode sub-layout col3 = L[:, 3] — shows how the slice's
     own offsets (NOT including the fixed-mode base) enumerate.
  3. The recovered absolute offsets col3[i] + base_of_col_3 — proves
     the slice + base recovers L[:, 3].

Usage:
  python visualize.py            # writes figures/slice_demo.png
  python visualize.py --show
"""

from __future__ import annotations

import argparse
import pathlib

# IMPORTANT: import mosaic + trigger a JIT compile BEFORE matplotlib.
# See basic_tiling/visualize.py for the libstdc++ ABI explanation.
from pymosaic.layout import Layout

_ = Layout.row_major(2, 2)[0]  # JIT a trivial layout first

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def heatmap(ax, arr: np.ndarray, title: str, highlight_col: int | None = None) -> None:
    ax.imshow(arr, cmap="viridis", aspect="equal")
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            ax.text(
                c,
                r,
                f"{arr[r, c]:d}",
                ha="center",
                va="center",
                color="white" if arr[r, c] < arr.max() * 0.6 else "black",
                fontsize=10,
            )
    if highlight_col is not None:
        rect = mpatches.Rectangle(
            (highlight_col - 0.5, -0.5),
            1,
            h,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)


def column_strip(ax, vals: np.ndarray, title: str, color: str = "C0") -> None:
    """A single-column heatmap showing the slice's enumeration."""
    arr = vals.reshape(-1, 1)
    ax.imshow(arr, cmap="viridis", aspect="equal")
    for r, v in enumerate(vals):
        ax.text(
            0,
            r,
            f"{v:d}",
            ha="center",
            va="center",
            color="white" if v < arr.max() * 0.6 else "black",
            fontsize=10,
        )
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_yticks(range(len(vals)))
    ax.set_title(title, color=color)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "figures" / "slice_demo.png",
    )
    args = parser.parse_args()

    L = Layout.column_major(4, 8)
    col3 = L[:, 3]

    # Materialise the full L offset grid.
    full = np.array([[L[r, c] for c in range(8)] for r in range(4)])
    slice_vals = np.array([col3[i] for i in range(len(col3))])
    base = L[0, 3]  # base offset of column 3
    abs_vals = slice_vals + base

    fig = plt.figure(figsize=(13, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[5, 1, 1, 0.5])
    ax_full = fig.add_subplot(gs[0])
    ax_slice = fig.add_subplot(gs[1])
    ax_abs = fig.add_subplot(gs[2])
    fig.add_subplot(gs[3]).axis("off")  # spacer

    heatmap(ax_full, full, f"L = {L}\n(value = L[r, c])", highlight_col=3)
    column_strip(ax_slice, slice_vals, f"L[:, 3] = {col3}\n(slice-only)", color="C0")
    column_strip(
        ax_abs, abs_vals, f"L[:, 3] + base\nbase = L[0, 3] = {base}", color="C3"
    )

    fig.suptitle(
        "Slicing: kept-mode shape/stride + caller-supplied base offset",
        fontsize=12,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
