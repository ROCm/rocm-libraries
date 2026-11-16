# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of examples/cpp/basic_tiling/check_tiling.cpp.

Demonstrates a 2-level workgroup tiling using mosaic's
``blocked_product``. The cpp version runs the kernel on the GPU; this
port stays on the host (no HIP), which is fine because the schedule is
purely a function of the layout — there is no per-thread work.

Setup:
  An HxW grid of "workgroups" (think CTAs / blocks). The natural
  scheduler assigns workgroup IDs in row-major order across the grid.
  We remap so that consecutive IDs land within the same L2_Y x L2_X
  *tile* of workgroups before moving to the next tile.

The remap is expressed as::

    layout = blocked_product(inner_block, outer_tiler)

with ``inner_block`` the per-tile ordering and ``outer_tiler`` how
tiles are arranged across the full grid.

``blocked_product`` is not yet exposed through the Python JIT (it lives
in ``include/mosaic/algebra.hpp:462-499``). We inline a pure-Python
port that mirrors the C++ template — it returns a regular ``Layout``
the JIT can then compile and evaluate on demand.
"""

from __future__ import annotations

from pymosaic.layout import Layout


def blocked_product(inner: Layout, outer: Layout) -> Layout:
    """Python port of ``mosaic::blocked_product`` for rank-N tuple shapes.

    Mirrors ``include/mosaic/algebra.hpp:462-499``. Per the algebra
    docstring there: place a copy of ``inner`` at every position of
    ``outer``, where each ``outer`` step advances by ``cosize(inner)``
    so the block copies don't collide. The result is a hierarchical
    Layout of shape ``((inner.shape[i], outer.shape[i]),)`` with
    strides ``((inner.stride[i], inner_cosize * outer.stride[i]),)``.

    For inputs of matching rank and disjoint codomains, the result is
    bijective onto ``[0, size(inner) * size(outer))``.
    """
    if inner.rank != outer.rank:
        raise ValueError(
            f"blocked_product requires same rank for both layouts; "
            f"got inner.rank={inner.rank}, outer.rank={outer.rank}"
        )

    inner_cosize = inner.cosize
    new_shape = tuple((inner.shape[i], outer.shape[i]) for i in range(inner.rank))
    new_stride = tuple(
        (inner.stride[i], inner_cosize * outer.stride[i]) for i in range(inner.rank)
    )
    return Layout(shape=new_shape, stride=new_stride)


def is_bijective(arr: list[int], size: int) -> bool:
    """True if ``arr`` is a permutation of ``[0, size)``."""
    return len(arr) == size and sorted(arr) == list(range(size))


def print_grid(arr: list[int], height: int, width: int) -> None:
    """Print ``arr`` as an HxW grid, mirroring the cpp helper."""
    max_val = max(arr)
    cell_width = max(4, len(str(max_val)) + 2)
    for row in range(height):
        line = "".join(
            f"{arr[row * width + col]:>{cell_width}d}" for col in range(width)
        )
        print(line)


def main() -> int:
    HEIGHT = 8
    WIDTH = 8
    TOTAL = HEIGHT * WIDTH
    L2_Y = 2
    L2_X = 2

    inner = Layout.row_major(L2_Y, L2_X)
    outer = Layout.column_major(HEIGHT // L2_Y, WIDTH // L2_X)
    l2_layout = blocked_product(inner, outer)

    print(f"  inner       = {inner}      # per-tile WG ordering (row-major)")
    print(f"  outer       = {outer}      # tile arrangement on the grid (col-major)")
    print(f"  l2_layout   = {l2_layout}")
    print(f"  size, cosize: {len(l2_layout)}, {l2_layout.cosize}\n")

    # Evaluate the layout at every flat workgroup ID.
    # layout_map[wg_id] = "permuted id" (where this WG lands in the new schedule).
    layout_map = [l2_layout[wg_id] for wg_id in range(TOTAL)]
    # wg_map[permuted_id] = original wg_id (inverse table — what executes here).
    wg_map = [0] * TOTAL
    for wg_id, permuted in enumerate(layout_map):
        wg_map[permuted] = wg_id

    if not is_bijective(layout_map, TOTAL):
        print(f"ERROR: layout_map is not a bijection over [0, {TOTAL})")
        return 1

    print("map[transform(wg_id)] = wg_id")
    print_grid(wg_map, HEIGHT, WIDTH)
    print("\nmap[wg_id] = transform(wg_id)")
    print_grid(layout_map, HEIGHT, WIDTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
