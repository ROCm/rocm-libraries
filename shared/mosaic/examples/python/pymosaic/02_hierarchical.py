# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example 2 — Hierarchical (nested) layouts.

What this shows:
  * Build a hierarchical layout two ways:
      (a) Layout.from_modes(inner, outer) — assemble from sub-Layouts.
      (b) Layout(shape=((2, 2), 3), stride=((1, 2), 4)) — explicit.
  * The shape's nesting profile must match the stride's nesting profile.
  * Index a hierarchical layout with either:
      - a flat int  (L[i]),
      - top-level ints   (L[r, c] — scalars are idx2crd'd on tuple modes),
      - matching nested coord (L[(0, 1), 2] — strict shape match).
  * idx2crd / crd2idx as a bidirectional pair.
  * Visualize the nesting with `pretty_print()`: a hierarchical layout
    renders as a grid with double-line block boundaries separating the
    sub-modes, so you can literally see the tile-within-tile structure.
"""

from __future__ import annotations

from pymosaic.layout import Layout


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("Build with from_modes: inner column-major (2,2), outer row-major (3,3)")
    inner = Layout.column_major(2, 2)  # shape (2, 2), stride (1, 2)
    outer = Layout.row_major(3, 3)  # shape (3, 3), stride (3, 1)
    H = Layout.from_modes(inner, outer)

    print(f"  inner       : {inner}")
    print(f"  outer       : {outer}")
    print(f"  H           : {H}")
    print(f"  H.shape     : {H.shape}")  # ((2, 2), (3, 3))
    print(f"  H.stride    : {H.stride}")  # ((1, 2), (3, 1))
    print(f"  H.rank, depth: {H.rank}, {H.depth}")
    print(f"  len(H)      : {len(H)}")

    section("See the nesting — H.pretty_print()")
    # A flat (4, 4) grid would just be 16 numbers. Because H is nested,
    # pretty_print draws double-line boundaries between the outer (3, 3)
    # blocks; inside each block the inner (2, 2) column-major sub-layout
    # repeats. Axis labels show the nested coordinate for each row/col.
    H.pretty_print()

    section("Explicit construction with matching nesting profile")
    # shape ((2, 2), 3) means: mode 0 is a rank-2 tuple, mode 1 is a scalar.
    # The stride must mirror that nesting: ((1, 2), 4).
    L = Layout(shape=((2, 2), 3), stride=((1, 2), 4))
    print(f"  L           : {L}")
    print(f"  L.shape     : {L.shape}")
    print(f"  L.stride    : {L.stride}")
    # Mode 0 is a (2, 2) tuple, mode 1 is the scalar 3. The grid below has
    # 4 rows (the 2x2 inner mode, labelled with nested (i, j) coords) and
    # 3 columns. Reading a cell gives i*1 + j*2 + c*4.
    L.pretty_print()

    section("Three coord forms — flat, top-level ints, nested tuple")
    # Flat: walks the natural column-major enumeration. Index 5 in a
    # 12-element layout, decoded as ((1, 0), 1) → 1*1 + 0*2 + 1*4 = 5.
    print(f"  L[5]                  = {L[5]}     # flat int")
    # Top-level ints: scalars on tuple modes are idx2crd'd internally.
    # L[1, 2] on a ((2,2), 3) shape → mode-0 scalar 1 unpacks to (1, 0),
    # giving 1*1 + 0*2 + 2*4 = 9.
    print(f"  L[1, 2]               = {L[1, 2]}     # scalar on tuple mode")
    # Nested coord: exact match of the shape's nesting profile.
    print(f"  L[(1, 1), 2]          = {L[(1, 1), 2]}    # nested tuple coord")

    section("idx2crd / crd2idx — bidirectional identity")
    # The canonical correctness check: convert flat -> coord -> flat.
    # The composition must reproduce L[i] for every i in the domain.
    for i in [0, 1, 2, 5, 7, 11]:
        coord = L.idx2crd(i)
        back = L.crd2idx(coord)
        print(
            f"  i={i:2d}  idx2crd -> {coord!s:20s}  crd2idx -> {back:2d}   (L[{i}]={L[i]})"
        )
        assert back == L[i]

    section("Full round-trip identity over the whole domain")
    failures = 0
    for i in range(len(L)):
        if L.crd2idx(L.idx2crd(i)) != L[i]:
            failures += 1
    print(f"  {len(L) - failures}/{len(L)} indices round-trip cleanly")


if __name__ == "__main__":
    main()
