# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example 4 — Tiling: composing layouts into a hierarchical schedule.

What this shows:
  * The single most useful thing layouts are for: *remapping* a flat
    index space so that neighbours in the new ordering stay close in
    some 2D structure (a tile / block / CTA).
  * `blocked_product(inner, outer)` — place a copy of the `inner`
    ordering at every position of the `outer` ordering. The result is a
    hierarchical Layout you can JIT-evaluate like any other.
  * Visualize the tiling with `pretty_print()`: the double-line block
    boundaries are exactly the tile edges, so you can *see* that
    consecutive IDs fill a tile before moving on.

The scenario:
  A 4x4 grid of "workgroups" (think GPU CTAs). The default scheduler
  hands out IDs row-major across the whole grid, so workgroup 0 and 1
  are horizontal neighbours but 0 and 4 are far apart. We instead want
  consecutive IDs to fill a 2x2 tile before moving to the next tile —
  better data locality.

`blocked_product` is not yet surfaced through the Python JIT (it lives
in C++ at include/mosaic/algebra.hpp). We inline the same ~10-line
pure-Python port used by the basic_tiling example; it returns an
ordinary Layout the JIT then compiles on demand.
"""

from __future__ import annotations

from pymosaic.layout import Layout


def blocked_product(inner: Layout, outer: Layout) -> Layout:
    """Place a copy of ``inner`` at every position of ``outer``.

    Each ``outer`` step advances by ``cosize(inner)`` so the block
    copies never collide. The result is a hierarchical Layout of shape
    ``((inner.shape[i], outer.shape[i]),)`` per mode — see
    include/mosaic/algebra.hpp.
    """
    inner_cosize = inner.cosize
    new_shape = tuple((inner.shape[i], outer.shape[i]) for i in range(inner.rank))
    new_stride = tuple(
        (inner.stride[i], inner_cosize * outer.stride[i]) for i in range(inner.rank)
    )
    return Layout(shape=new_shape, stride=new_stride)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("The default schedule — plain row-major 4x4")
    # Workgroup IDs handed out left-to-right, top-to-bottom. Notice how
    # the 2x2 corner {0,1,4,5} is split across two far-apart rows.
    grid = Layout.row_major(4, 4)
    grid.pretty_print()

    section("Build the tiling with blocked_product")
    # inner: how IDs are ordered *within* one 2x2 tile (row-major).
    # outer: how the four tiles are arranged across the grid (col-major).
    inner = Layout.row_major(2, 2)
    outer = Layout.column_major(2, 2)
    tiled = blocked_product(inner, outer)

    print(f"  inner       : {inner}      # ordering within a tile")
    print(f"  outer       : {outer}      # arrangement of tiles")
    print(f"  tiled       : {tiled}")
    print(f"  tiled.shape : {tiled.shape}   # nested: ((inner, outer) per axis)")
    print(f"  size, cosize: {len(tiled)}, {tiled.cosize}")

    section("See the tiles — tiled.pretty_print()")
    # The double-line boundaries ARE the tile edges. Read the cells in
    # increasing order: 0,1,2,3 fill the top-left tile, then 4,5,6,7 the
    # next tile, and so on — exactly the locality we wanted.
    tiled.pretty_print()

    section("Verify it's a bijection over [0, 16)")
    # A valid schedule is a permutation: every workgroup ID lands in
    # exactly one grid position and vice-versa.
    mapped = [tiled[wg_id] for wg_id in range(len(tiled))]
    ok = sorted(mapped) == list(range(len(tiled)))
    print(f"  permutation of [0, {len(tiled)}) : {ok}")

    section("Same idea, bigger: 8x8 grid, 2x4 tiles")
    big = blocked_product(Layout.row_major(2, 4), Layout.column_major(4, 2))
    print(f"  layout      : {big}")
    big.pretty_print()


if __name__ == "__main__":
    main()
