# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render MFMA TV layouts as "which lane owns which cell" heatmaps.

For the three operands of ``v_mfma_f32_32x32x8_f16`` (A 32x8, B 32x8,
D 32x32), each matrix cell is colored by the lane (0..63) that holds
that cell in a VGPR. This is the simplest visual way to "see" the
distribution the matrix core expects.

For D, also annotate one chosen lane's cells with their per-lane VGPR
id (vid), so you can see how the 16 values a lane owns are scattered
across the 32x32 D tile.

Usage:
  python visualize.py            # writes figures/mfma_tv_layouts.png
  python visualize.py --show
"""

from __future__ import annotations

import argparse
import pathlib

# IMPORTANT: import mosaic + trigger a JIT compile BEFORE matplotlib.
# See basic_tiling/visualize.py for the libstdc++ ABI explanation.
from pymosaic.layout import Layout

_ = Layout.row_major(2, 2)[0]  # JIT a trivial layout first

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WAVE = 64


def lane_grid(tv: Layout, rows: int, cols: int, vals: int) -> np.ndarray:
    """Return a (rows, cols) array where each cell = its owning lane id.

    Decodes the TV layout's col-major flat output back to (row, col).
    """
    out = np.full((rows, cols), -1, dtype=int)
    for tid in range(WAVE):
        for vid in range(vals):
            flat = tv[tid, vid]
            # col-major: flat = col * rows + row
            col, row = divmod(flat, rows)
            out[row, col] = tid
    assert (out >= 0).all(), "TV layout missed cells"
    return out


def vid_grid_for_lane(
    tv: Layout, lane: int, rows: int, cols: int, vals: int
) -> np.ndarray:
    """Return a (rows, cols) array where the chosen lane's cells hold
    their vid (0..vals-1), and every other cell is -1.
    """
    out = np.full((rows, cols), -1, dtype=int)
    for vid in range(vals):
        flat = tv[lane, vid]
        col, row = divmod(flat, rows)
        out[row, col] = vid
    return out


def plot_lane_heatmap(
    ax,
    arr: np.ndarray,
    title: str,
    *,
    annotate_lane: int | None = None,
    vid_overlay: np.ndarray | None = None,
) -> None:
    h, w = arr.shape
    # 64 lanes -> a tab20-derived categorical palette, but lifted to a
    # continuous cmap for readability at small cell sizes.
    cmap = plt.get_cmap("twilight", WAVE)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=WAVE - 1, aspect="equal")

    # Highlight chosen lane's cells with vid labels.
    if annotate_lane is not None and vid_overlay is not None:
        for r in range(h):
            for c in range(w):
                if vid_overlay[r, c] >= 0:
                    ax.text(
                        c,
                        r,
                        f"{vid_overlay[r, c]}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                        fontweight="bold",
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.75,
                            "pad": 0.6,
                            "edgecolor": "none",
                        },
                    )

    ax.set_xticks(range(0, w, max(1, w // 8)))
    ax.set_yticks(range(0, h, max(1, h // 8)))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if annotate_lane is not None:
        ax.set_title(
            f"{title}\n(cells of lane {annotate_lane} labelled with their vid)",
            fontsize=10,
        )
    else:
        ax.set_title(title, fontsize=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent
        / "figures"
        / "mfma_tv_layouts.png",
    )
    parser.add_argument(
        "--annotate-lane",
        type=int,
        default=5,
        help="lane whose vids to label on the D panel (default 5)",
    )
    args = parser.parse_args()

    A_tv = Layout(shape=((32, 2), 4), stride=((1, 128), 32))
    B_tv = Layout(shape=((32, 2), 4), stride=((1, 128), 32))
    D_tv = Layout(shape=((32, 2), (4, 4)), stride=((32, 4), (1, 8)))

    A_grid = lane_grid(A_tv, 32, 8, 4)
    B_grid = lane_grid(B_tv, 32, 8, 4)
    D_grid = lane_grid(D_tv, 32, 32, 16)
    D_vids = vid_grid_for_lane(D_tv, args.annotate_lane, 32, 32, 16)

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1.4, 5, 0.25])
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])
    ax_D = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    plot_lane_heatmap(ax_A, A_grid, "A (M=32 x K=8)")
    plot_lane_heatmap(ax_B, B_grid, "B (N=32 x K=8)")
    plot_lane_heatmap(
        ax_D,
        D_grid,
        "D (M=32 x N=32)",
        annotate_lane=args.annotate_lane,
        vid_overlay=D_vids,
    )

    # Shared colorbar — cell color = owning lane id.
    norm = plt.Normalize(vmin=0, vmax=WAVE - 1)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("twilight", WAVE), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ticks=[0, 16, 32, 48, 63])
    cbar.set_label("lane id (tid)")

    fig.suptitle(
        "v_mfma_f32_32x32x8_f16 — TV layouts (cell colour = owning lane)",
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
