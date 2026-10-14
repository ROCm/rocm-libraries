# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side GEMM schedule visualisation (no GPU kernel).

This script lifts the **layout reasoning** — which workgroup ID executes
which (pid_m, pid_n) tile — into integer / layout arithmetic.

What this script computes:

  * The default row-major mapping ``wgid -> (pid_m, pid_n)``.
  * The mosaic 3-level hierarchical mapping, built from two nested
    ``blocked_product`` calls:
        block_l2 = column_major(L2_TILE_Y, L2_TILE_X)        # WGs within L2 tile
        tiler_l3 = column_major(L3_TILE_Y, L3_TILE_X)        # L2 tiles within L3
        tiler_out = row_major(GRID_H/(L3*L2), GRID_W/(L3*L2)) # L3 super-tiles
        l3_block        = blocked_product(block_l2, tiler_l3)
        gemm_grid_layout = blocked_product(l3_block, tiler_out)
  * The XCD-chiplet remap that runs first
    (``chiplet_transform_chunk`` — pure integer arithmetic, ported
    verbatim from include/mosaic/algorithms.hpp:46-72).
  * The boundary-strip handling for non-quantized grids
    (``gemm_grid_transform`` — port of include/mosaic/algorithms.hpp:133-169).

What this script prints:

  1. A side-by-side ASCII visualization of the first N workgroup IDs
     under the default vs. mosaic schedules — useful for seeing tile
     locality at a glance.
  2. A bijectivity assertion over the entire grid (every tile is
     executed exactly once).
  3. A summary of which XCD chiplet each workgroup ID would land on
     under the chiplet remap.

This script is the "what is the layout doing?" companion; it does not
run a GEMM kernel.
"""

from __future__ import annotations

from pymosaic.layout import Layout

# -------------------------------------------------------------------
# Helpers — Python ports of include/mosaic/algorithms.hpp + algebra.hpp
# -------------------------------------------------------------------


def blocked_product(inner: Layout, outer: Layout) -> Layout:
    """Port of ``mosaic::blocked_product`` (algebra.hpp:462-499).

    Per-mode shape ``(inner.shape[i], outer.shape[i])``; per-mode
    stride ``(inner.stride[i], cosize(inner) * outer.stride[i])``.
    Handles hierarchical inner/outer the same way mosaic does — the
    tuple components are passed through unchanged.
    """
    if inner.rank != outer.rank:
        raise ValueError(
            f"blocked_product requires same rank; "
            f"got inner.rank={inner.rank}, outer.rank={outer.rank}"
        )
    inner_cosize = inner.cosize
    new_shape = tuple((inner.shape[i], outer.shape[i]) for i in range(inner.rank))
    new_stride = tuple(
        (inner.stride[i], inner_cosize * outer.stride[i]) for i in range(inner.rank)
    )
    return Layout(shape=new_shape, stride=new_stride)


def chiplet_transform_chunk(
    wgid: int, num_wgs: int, chunk_size: int, num_xcds: int
) -> int:
    """Port of include/mosaic/algorithms.hpp:46-72.

    Groups ``chunk_size`` consecutive workgroup IDs onto the same XCD,
    so an L2 tile's worth of WGs all execute on the same chiplet. The
    tail (WGs that don't fit in a complete chunk-distributed cycle)
    keep their original ID.
    """
    chunk_size = min(chunk_size, num_wgs // num_xcds)
    xcd = wgid % num_xcds
    if wgid > (num_wgs // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return wgid
    local_idx = wgid // num_xcds
    chunk_idx = local_idx // chunk_size
    pos_in_chunk = local_idx % chunk_size
    return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk


def gemm_grid_transform(
    index: int, L: Layout, grid_height: int, grid_width: int, qy: int, qx: int
) -> int:
    """Port of include/mosaic/algorithms.hpp:133-169.

    ``L`` permutes IDs in the quantized ``qy x qx`` interior region.
    The non-quantized right strip (width ``grid_width - qx``) and
    bottom strip (height ``grid_height - qy``) are kept in default
    row-major order so non-uniform grids still cover every tile.
    """
    total_quantized = qx * qy
    non_quantized_x = grid_width - qx
    x_strip_size = non_quantized_x * qy
    y_strip_start = total_quantized + x_strip_size

    if index < total_quantized:
        local = int(L[index])
        y, x = divmod(local, qx)
        return y * grid_width + x

    if non_quantized_x and index < y_strip_start:
        rel = index - total_quantized
        y, x = divmod(rel, non_quantized_x)
        return y * grid_width + (qx + x)

    rel = index - y_strip_start
    y, x = divmod(rel, grid_width)
    return (qy + y) * grid_width + x


# -------------------------------------------------------------------
# Mapping strategies — the two functions the cpp example compares.
# -------------------------------------------------------------------


def default_grid_mapping(
    wgid: int, grid_height: int, grid_width: int
) -> tuple[int, int]:
    """Simple row-major mapping: ``pid_m = wgid / W;  pid_n = wgid % W``."""
    return wgid // grid_width, wgid % grid_width


def mosaic_grid_mapping(
    wgid: int,
    grid_height: int,
    grid_width: int,
    num_xcds: int,
    layout: Layout,
    qy: int,
    qx: int,
) -> tuple[int, int]:
    """Mosaic mapping: chiplet remap + hierarchical-layout permutation."""
    wgid = chiplet_transform_chunk(wgid, grid_height * grid_width, 32, num_xcds)
    idx = gemm_grid_transform(wgid, layout, grid_height, grid_width, qy, qx)
    return idx // grid_width, idx % grid_width


def build_gemm_layout(grid_height: int, grid_width: int) -> tuple[Layout, int, int]:
    """Build the 3-level hierarchical schedule and return the
    (layout, qy, qx) triple that ``gemm_grid_transform`` expects.

    Mirrors the cpp example's tile sizes exactly:
        L3 = 2 x 4 super-tiles of L2 tiles
        L2 = 8 x 4 workgroups per L2 tile
    """
    L3_TILE_Y, L3_TILE_X = 2, 4
    L2_TILE_Y, L2_TILE_X = 8, 4

    block_l2 = Layout.column_major(L2_TILE_Y, L2_TILE_X)
    tiler_l3 = Layout.column_major(L3_TILE_Y, L3_TILE_X)
    tiler_out = Layout.row_major(
        grid_height // (L3_TILE_Y * L2_TILE_Y),
        grid_width // (L3_TILE_X * L2_TILE_X),
    )
    l3_block = blocked_product(block_l2, tiler_l3)
    gemm_grid_layout = blocked_product(l3_block, tiler_out)

    # Quantized extents — the interior the layout cleanly covers.
    qy = (grid_height // (L3_TILE_Y * L2_TILE_Y)) * L3_TILE_Y * L2_TILE_Y
    qx = (grid_width // (L3_TILE_X * L2_TILE_X)) * L3_TILE_X * L2_TILE_X
    return gemm_grid_layout, qy, qx


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------


def print_schedule(
    label: str, schedule: list[tuple[int, int]], grid_h: int, grid_w: int
) -> None:
    """Print the wgid -> (pid_m, pid_n) schedule as a grid of "the
    wgid that lands at this tile". Lower-left is wgid 0 → top-left
    in a (row=0, col=0) interpretation.
    """
    grid = [["    " for _ in range(grid_w)] for _ in range(grid_h)]
    for wgid, (pm, pn) in enumerate(schedule):
        grid[pm][pn] = f"{wgid:4d}"
    print(f"\n{label}  ({grid_h}x{grid_w} tile grid; cell = wgid landing there)")
    for row in grid:
        print(" ".join(row))


# -------------------------------------------------------------------
# Main demo
# -------------------------------------------------------------------


def main() -> int:
    # Pick a small, quantized grid so the visualization fits. Real
    # workloads use 32x192 or similar; we mirror the cpp's L2/L3 ratio
    # at minimum size: L3_Y * L2_Y = 16, L3_X * L2_X = 16, so the
    # smallest non-trivial grid is 16 x 16.
    GRID_H = 16
    GRID_W = 16
    NUM_XCDS = 8  # MI300X / CDNA3 has 8 XCDs

    layout, qy, qx = build_gemm_layout(GRID_H, GRID_W)
    total = GRID_H * GRID_W

    print(f"Grid     : {GRID_H} x {GRID_W}  ({total} workgroups)")
    print(f"NUM_XCDS : {NUM_XCDS}")
    print(f"Quantized: qy={qy}, qx={qx}  (interior covered by the L3 layout)")
    print(f"Layout   : {layout}")
    print(f"           shape={layout.shape}")
    print(f"           stride={layout.stride}\n")

    # ----- compute both schedules -----
    default = [default_grid_mapping(w, GRID_H, GRID_W) for w in range(total)]
    mosaic = [
        mosaic_grid_mapping(w, GRID_H, GRID_W, NUM_XCDS, layout, qy, qx)
        for w in range(total)
    ]

    # ----- assert both are bijections over the tile grid -----
    assert sorted(default) == sorted(
        (m, n) for m in range(GRID_H) for n in range(GRID_W)
    )
    assert sorted(mosaic) == sorted(
        (m, n) for m in range(GRID_H) for n in range(GRID_W)
    )
    print("Both schedules are bijections over the tile grid. ✓")

    # ----- visualize -----
    print_schedule("DEFAULT  (row-major)", default, GRID_H, GRID_W)
    print_schedule("MOSAIC   (chiplet + 3-level hierarchical)", mosaic, GRID_H, GRID_W)

    # ----- chiplet-remap distribution -----
    # Group the first N wgids by which XCD they execute on. A good
    # schedule keeps consecutive IDs on the same XCD for L2 locality.
    print(f"\nXCD assignment for first 32 wgids ({NUM_XCDS} XCDs available):")
    for w in range(32):
        remapped = chiplet_transform_chunk(w, total, 32, NUM_XCDS)
        xcd = remapped % NUM_XCDS
        print(f"  wgid={w:3d} -> remapped={remapped:3d} -> xcd={xcd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
