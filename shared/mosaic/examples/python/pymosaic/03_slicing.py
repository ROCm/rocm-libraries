# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example 3 — Slicing semantics.

What this shows:
  * L[:, k] / L[i, :] / L[..., j] return a fresh, lower-rank Layout
    (NOT an offset).
  * Slicing carries the kept-mode shape/stride; the base offset
    contributed by the fixed modes is NOT folded in. The caller adds
    it back if they need the full-tensor offset.
  * Range slicing (L[0:2]) is intentionally rejected — slicing is
    "drop modes", not "shrink modes".
  * The all-axes-kept identity (L[:, :]) returns self by object
    identity, so it doesn't burn a new JIT cache entry.

Why this matters:
  This mirrors mosaic's C++ ``slice(_, L)`` semantics. The intuition:
  "the sub-Layout describes movement within the kept modes; the base
  offset belongs to the data pointer, not the layout."
"""

from __future__ import annotations

from pymosaic.layout import Layout


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("Setup: row-major (4, 8)")
    L = Layout.row_major(4, 8)
    print(f"  L           : {L}")  # (4,8):(8,1)
    print(f"  full layout offsets:")
    # The offset grid we slice into below. A column slice picks one
    # column of this grid; a row slice picks one row.
    L.pretty_print()

    section("Column slice — L[:, 3]")
    col = L[:, 3]
    print(f"  col         : {col}")  # (4,):(8,)
    print(f"  col.shape   : {col.shape}")
    print(f"  col.stride  : {col.stride}")
    print(f"  col[0..{len(col)-1}] : {list(col)}")
    # The base offset of column 3 in the original layout is 3*1 = 3.
    # That offset is NOT carried into `col`; the caller adds it back.
    base = 3 * 1
    print(f"  recovered absolute offsets (col[i] + base):")
    print("    " + " ".join(f"{col[i] + base:3d}" for i in range(len(col))))
    print(f"  matches L[:, 3]:")
    print("    " + " ".join(f"{L[r, 3]:3d}" for r in range(4)))

    section("Row slice — L[1, :]")
    row = L[1, :]
    print(f"  row         : {row}")  # (8,):(1,)
    print(f"  row[0..{len(row)-1}] : {list(row)}")
    base = 1 * 8
    print(f"  L[1, :] absolute offsets:")
    print("    " + " ".join(f"{row[j] + base:3d}" for j in range(len(row))))

    section("Ellipsis — L[..., 3] expands to L[:, 3] on rank-2")
    col_via_ellipsis = L[..., 3]
    assert col_via_ellipsis == col
    print(f"  L[..., 3]   : {col_via_ellipsis}   (same as L[:, 3])")

    section("Rank-3 — keep middle axis only: L[1, :, 3]")
    L3 = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
    sub = L3[1, :, 3]
    print(f"  L3          : {L3}")
    print(f"  sub         : {sub}")  # (5,):(7,)
    base = 1 * 35 + 3 * 1
    for i in range(len(sub)):
        absolute = sub[i] + base
        full = L3[1, i, 3]
        marker = "✓" if absolute == full else "✗"
        print(
            f"  sub[{i}] + base = {sub[i]:2d} + {base:2d} = {absolute:2d}  ==  L3[1,{i},3] = {full:2d}  {marker}"
        )

    section("All-axes-kept slicing returns self (no new JIT entry)")
    same = L[:, :]
    print(f"  L[:, :] is L : {same is L}")
    same_ell = L[...]
    print(f"  L[...]  is L : {same_ell is L}")

    section(
        "Range slicing is not supported — strided sub-layouts aren't part of this API yet"
    )
    try:
        _ = L[0:2, :]
    except NotImplementedError as exc:
        print(f"  L[0:2, :]   -> NotImplementedError: {exc}")


if __name__ == "__main__":
    main()
