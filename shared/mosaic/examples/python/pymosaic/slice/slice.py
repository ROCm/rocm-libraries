# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of examples/cpp/slice/slice.cpp.

Demonstrates mosaic's CuTe-style "underscore" slicing exposed through
the Python ``Layout`` API.

  * ``L[:, k]``  -> sub-layout for column ``k`` along mode 1.
  * ``L[i, :]``  -> sub-layout for row ``i`` along mode 0.
  * Slicing returns a Layout (with its own ``shape``, ``stride``, etc.) —
    the absolute base offset contributed by the fixed mode is NOT folded
    in, the caller adds it back via the original layout's coord evaluation.
  * Hierarchical layouts can have inner modes kept and outer modes fixed
    (mosaic accepts nested-tuple ints as a "fix this whole mode" coord).

What this port DOES that the cpp doesn't:
  Python uses ``L[:, 3]`` instead of ``L(_, 3)``. The semantics match;
  the difference is just that Python's ``:`` plays the role of cpp's ``_``.

Both hierarchical slicing (``H[:, (1, 1)]`` — fix a nested mode with a
nested coord) and ``dice`` (the "drop free, keep fixed" complement of
``slice``) are now exposed through the Python layer.
"""

from __future__ import annotations

from pymosaic.layout import Layout


def enumerate_layout(label: str, layout: Layout) -> None:
    """Print a short one-line summary mirroring the cpp ``enumerate`` helper."""
    vals = " ".join(str(v) for v in layout)
    print(f"  {label} = {layout}  (size={len(layout)}):  {vals}")


def main() -> None:
    print("Mosaic slice demo (Python port)")
    print("================================\n")

    # ------------------------------------------------------------------
    # Example 1: Column slice of a flat 2D layout
    # ------------------------------------------------------------------
    print("Example 1: 4x8 column-major layout")
    L = Layout.column_major(4, 8)  # default in cpp is col-major
    print(f"  L = {L}\n")

    print("  L[:, 3] — sub-layout for column 3 (rank-1, size 4):")
    col3 = L[:, 3]
    enumerate_layout("col3", col3)
    # Recover the base offset of column 3: it's the contribution of the
    # fixed mode at coord (0, 3) under L's strides — for col-major (4, 8)
    # with stride (1, 4): 0*1 + 3*4 = 12.
    col3_base = L[0, 3]
    print(f"  base offset of column 3 in L: {col3_base}")
    full = " ".join(str(col3_base + col3[i]) for i in range(len(col3)))
    print(f"  full column-3 offsets in L:   {full}\n")

    print("  L[2, :] — sub-layout for row 2 (rank-1, size 8):")
    row2 = L[2, :]
    enumerate_layout("row2", row2)
    row2_base = L[2, 0]
    print(f"  base offset of row 2 in L: {row2_base}")
    full = " ".join(str(row2_base + row2[j]) for j in range(len(row2)))
    print(f"  full row-2 offsets in L:   {full}\n\n")

    # ------------------------------------------------------------------
    # Example 2: same slice via Ellipsis spelling
    # ------------------------------------------------------------------
    print("Example 2: L[..., 3] expands the leading ':' for you")
    col3_alt = L[..., 3]
    enumerate_layout("col3_alt", col3_alt)
    assert col3_alt == col3
    print()

    # ------------------------------------------------------------------
    # Example 3: Hierarchical layout, fix the outer mode, keep the inner
    # ------------------------------------------------------------------
    print("\nExample 3: hierarchical layout — fix outer mode, keep inner")
    inner = Layout.row_major(2, 2)  # shape (2,2), stride (2,1)
    outer = Layout.column_major(3, 3)  # shape (3,3), stride (1,3)
    H = Layout.from_modes(inner, outer)
    print(f"  inner   = {inner}")
    print(f"  outer   = {outer}")
    print(f"  H       = {H}")
    print(f"  H.shape = {H.shape}")
    print(f"  H.stride= {H.stride}\n")

    # The cpp example spells "fix outer mode at coord (1, 1), keep
    # inner" as ``H(_, make_coord(1, 1))``. In Python this is hierarchical
    # slicing: pass the nested coord (1, 1) at the fixed position.
    inner_keep = H[:, (1, 1)]
    enumerate_layout("inner_keep = H[:, (1, 1)]", inner_keep)
    print("  (in the cpp example this is H(_, make_coord(1, 1)).)")
    print()

    # ------------------------------------------------------------------
    # Example 4: dice — the complement of slice
    # ------------------------------------------------------------------
    print("\nExample 4: dice — keep the FIXED modes (complement of slice)")
    # The cpp example calls ``dice(make_coord(_, 3), L)`` which keeps the
    # FIXED modes — here mode 1 (size 8) — and drops the ':' modes.
    coord = (slice(None), 3)
    sliced = L[coord]  # keeps the ':' modes
    diced = L.dice(coord)  # keeps the fixed modes
    print(f"  coord       = (':', 3)")
    print(f"  L           = {L}")
    print(f"  L[':', 3]   = {sliced}   # KEPT by slice (the ':' modes)")
    print(f"  L.dice(...) = {diced}   # KEPT by dice  (the fixed modes)")


if __name__ == "__main__":
    main()
