# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render the basic_tiling schedule as a PNG side-by-side comparison.

Default (row-major) schedule next to the tiled `blocked_product` one.
Cell color encodes the workgroup ID using a sequential colormap so that
"consecutive IDs stay near each other" is visually obvious.

Usage:
  python visualize.py            # writes figures/tiling_schedule.png
  python visualize.py --show     # also opens an interactive window
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# IMPORTANT: import mosaic + trigger a JIT compile BEFORE matplotlib.
# On systems with a conda Python whose libstdc++ predates the system
# g++'s GLIBCXX_*, importing matplotlib first locks the older libstdc++
# into the process, and the JIT .so then fails to dlopen with
# "GLIBCXX_3.4.32 not found". Forcing the system libstdc++ to load first
# (via the JIT .so's DT_NEEDED) sidesteps the issue.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_tiling import blocked_product  # noqa: E402
from pymosaic.layout import Layout  # noqa: E402

_ = Layout.row_major(2, 2)[0]  # JIT a trivial layout to load system libstdc++

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def build_schedule(grid_h: int, grid_w: int, l2_y: int, l2_x: int) -> np.ndarray:
    """Return an ``(grid_h, grid_w)`` int array where ``arr[m, n]`` is
    the wgid that lands at tile ``(m, n)`` under blocked_product.
    """
    inner = Layout.row_major(l2_y, l2_x)
    outer = Layout.column_major(grid_h // l2_y, grid_w // l2_x)
    L = blocked_product(inner, outer)

    arr = np.zeros((grid_h, grid_w), dtype=int)
    for wg_id in range(grid_h * grid_w):
        permuted = L[wg_id]
        m, n = divmod(permuted, grid_w)
        arr[m, n] = wg_id
    return arr


def default_schedule(grid_h: int, grid_w: int) -> np.ndarray:
    """The trivial wgid -> (m, n) = (wgid // W, wgid % W) row-major schedule."""
    return np.arange(grid_h * grid_w, dtype=int).reshape(grid_h, grid_w)


def render(arr: np.ndarray, ax, title: str) -> None:
    """Render one schedule as a heatmap with annotated cells."""
    h, w = arr.shape
    ax.imshow(arr, cmap="viridis", aspect="equal")
    for r in range(h):
        for c in range(w):
            ax.text(
                c,
                r,
                f"{arr[r, c]:d}",
                ha="center",
                va="center",
                color="white" if arr[r, c] < arr.max() * 0.6 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlabel("tile col (pid_n)")
    ax.set_ylabel("tile row (pid_m)")
    ax.set_title(title)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display interactively too")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent
        / "figures"
        / "tiling_schedule.png",
    )
    args = parser.parse_args()

    H, W = 8, 8
    L2_Y, L2_X = 2, 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    render(
        default_schedule(H, W),
        axes[0],
        f"Default row-major\nwgid -> (wgid // {W}, wgid % {W})",
    )
    render(
        build_schedule(H, W, L2_Y, L2_X),
        axes[1],
        f"blocked_product(row_major({L2_Y}, {L2_X}),"
        f"\n                column_major({H // L2_Y}, {W // L2_X}))",
    )
    fig.suptitle(
        f"Workgroup schedule comparison — {H}x{W} grid, {L2_Y}x{L2_X} inner tile",
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
