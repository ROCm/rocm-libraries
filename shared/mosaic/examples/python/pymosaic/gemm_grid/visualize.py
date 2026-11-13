# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render the GEMM grid schedule as a side-by-side comparison plus an
XCD-assignment panel.

Panel 1: default row-major schedule on a 16x16 tile grid (heatmap of
         wgid landing at each (pid_m, pid_n)).
Panel 2: mosaic 3-level hierarchical schedule (after chiplet_transform_chunk
         + gemm_grid_transform). The L2 tiles (8x4 WGs) and L3
         super-tiles (2x4 L2 tiles) show up as block patterns.
Panel 3: which XCD each tile lands on under the mosaic schedule.
         Each of NUM_XCDS chiplets is a categorical color.

Usage:
  python visualize.py            # writes figures/gemm_grid_schedule.png
  python visualize.py --show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# IMPORTANT: import mosaic + trigger a JIT compile BEFORE matplotlib.
# See basic_tiling/visualize.py for the libstdc++ ABI explanation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grid_mapping import (  # noqa: E402
    build_gemm_layout,
    chiplet_transform_chunk,
    default_grid_mapping,
    mosaic_grid_mapping,
)
from pymosaic.layout import Layout  # noqa: E402

_ = Layout.row_major(2, 2)[0]  # JIT a trivial layout first

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_id_heatmap(
    ax, schedule: list[tuple[int, int]], grid_h: int, grid_w: int, title: str
) -> None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for wgid, (m, n) in enumerate(schedule):
        grid[m, n] = wgid
    ax.imshow(grid, cmap="viridis", aspect="equal")
    for r in range(grid_h):
        for c in range(grid_w):
            ax.text(
                c,
                r,
                f"{grid[r, c]:d}",
                ha="center",
                va="center",
                color="white" if grid[r, c] < grid.max() * 0.6 else "black",
                fontsize=6,
            )
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, grid_w, 4))
    ax.set_yticks(range(0, grid_h, 4))
    ax.set_xlabel("pid_n")
    ax.set_ylabel("pid_m")


def render_xcd_panel(ax, grid_h: int, grid_w: int, num_xcds: int) -> None:
    """For each tile (pid_m, pid_n), find the wgid that lands there
    under the mosaic schedule, then color the tile by that wgid's XCD.
    """
    total = grid_h * grid_w
    # Recover the schedule we computed elsewhere — needed because we
    # need both directions: tile -> wgid (to color), wgid -> tile (the
    # forward schedule).
    layout, qy, qx = build_gemm_layout(grid_h, grid_w)
    tile_to_xcd = np.zeros((grid_h, grid_w), dtype=int)
    for wgid in range(total):
        m, n = mosaic_grid_mapping(wgid, grid_h, grid_w, num_xcds, layout, qy, qx)
        # The chiplet remap maps original wgid -> remapped wgid; XCD is
        # remapped % num_xcds. We re-run the remap to derive XCD from
        # the wgid that lands here.
        remapped = chiplet_transform_chunk(wgid, total, 32, num_xcds)
        tile_to_xcd[m, n] = remapped % num_xcds

    cmap = plt.get_cmap("tab10", num_xcds)
    ax.imshow(tile_to_xcd, cmap=cmap, vmin=0, vmax=num_xcds - 1, aspect="equal")
    for r in range(grid_h):
        for c in range(grid_w):
            ax.text(
                c,
                r,
                f"{tile_to_xcd[r, c]:d}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    ax.set_title(f"XCD assignment (mosaic, {num_xcds} XCDs)", fontsize=10)
    ax.set_xticks(range(0, grid_w, 4))
    ax.set_yticks(range(0, grid_h, 4))
    ax.set_xlabel("pid_n")
    ax.set_ylabel("pid_m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent
        / "figures"
        / "gemm_grid_schedule.png",
    )
    args = parser.parse_args()

    GRID_H, GRID_W = 16, 16
    NUM_XCDS = 8

    layout, qy, qx = build_gemm_layout(GRID_H, GRID_W)
    total = GRID_H * GRID_W
    default = [default_grid_mapping(w, GRID_H, GRID_W) for w in range(total)]
    mosaic = [
        mosaic_grid_mapping(w, GRID_H, GRID_W, NUM_XCDS, layout, qy, qx)
        for w in range(total)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    render_id_heatmap(axes[0], default, GRID_H, GRID_W, "Default row-major")
    render_id_heatmap(
        axes[1],
        mosaic,
        GRID_H,
        GRID_W,
        "Mosaic: chiplet_transform_chunk\n+ 3-level blocked_product",
    )
    render_xcd_panel(axes[2], GRID_H, GRID_W, NUM_XCDS)

    fig.suptitle(
        f"GEMM grid schedule — {GRID_H}x{GRID_W} tile grid, NUM_XCDS={NUM_XCDS}",
        fontsize=13,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
