# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example 1 — Construction, evaluation, iteration.

What this shows:
  * Build a row-major and a column-major Layout.
  * Inspect (shape, stride, size, cosize, rank, depth).
  * Evaluate at a flat index L[i] and at a coord L[r, c].
  * Iterate to materialize the full offset sequence.
  * Confirm that the first eval triggers a JIT compile (~1–2 s cold,
    sub-millisecond hot), and the second eval hits the on-disk cache.

How to read the output:
  Row-major (4, 8) has stride (8, 1):  offset = r*8 + c*1.
  Column-major (4, 8) has stride (1, 4): offset = r*1 + c*4.

The clearest way to *see* what a layout does is `L.pretty_print()`,
which draws the grid of offsets — cell (r, c) is the value the layout
maps that coordinate to. This example ends with that view.
"""

from __future__ import annotations

import os
import tempfile
import time

# Point the JIT cache at a fresh temp directory so the cold-path timing
# below is honest — otherwise the first call may hit your real
# ~/.cache/mosaic/jit/ and look misleadingly fast.
os.environ["MOSAIC_JIT_CACHE"] = tempfile.mkdtemp(prefix="mosaic-jit-ex01-")

from pymosaic.layout import Layout  # noqa: E402  (env var must be set first)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("Row-major Layout(4, 8) — pure Python inspection (no JIT yet)")
    # Construction is cheap: just canonicalises shape/stride. The C++
    # module is JIT-compiled lazily, on the first call that needs it.
    # repr() / .shape / .stride / .rank / .depth all stay pure Python.
    L = Layout.row_major(4, 8)
    print(f"  repr(L)     : {L!r}")
    print(f"  shape       : {L.shape}")
    print(f"  stride      : {L.stride}")
    print(f"  rank, depth : {L.rank}, {L.depth}")

    section("First evaluation triggers JIT compile (cold path)")
    # Anything that needs the C++ side fires the compile. The first
    # such call here is L[2, 3] — it routes through eval_coord in the
    # generated .so.  str(L), L[i], len(L), L.cosize would all do it too.
    t0 = time.perf_counter()
    val = L[2, 3]  # 2*8 + 3*1 = 19
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"  L[2, 3]     = {val}   (cold compile + first eval took ~{cold_ms:.0f} ms)")

    section("Second evaluation is a hot in-process call")
    t0 = time.perf_counter()
    val = L[1, 5]  # 1*8 + 5*1 = 13
    hot_us = (time.perf_counter() - t0) * 1_000_000
    print(f"  L[1, 5]     = {val}   (hot call took ~{hot_us:.1f} µs)")

    section("str(L) — matches mosaic's C++ operator<< format")
    print(f"  str(L)      : {L}")  # "(4,8):(8,1)"

    section("Size, cosize, and full enumeration")
    print(f"  len(L)      : {len(L)}        # number of valid flat indices")
    print(f"  L.cosize    : {L.cosize}        # max offset + 1")
    print(f"  enumerate() : {L.enumerate()}")
    # `enumerate()` calls into a single C++ loop. Equivalent to list(L)
    # but avoids the per-element Python round-trip.

    section("Visualize the offset grid — L.pretty_print()")
    # The intuition shortcut: print the grid of offsets. For row-major
    # the numbers count left-to-right across each row (stride 1 on the
    # last axis), jumping by 8 between rows.
    L.pretty_print()

    section("Column-major Layout(4, 8) — same shape, different stride")
    C = Layout.column_major(4, 8)
    print(f"  layout      : {C}")  # "(4,8):(1,4)"
    print(f"  stride      : {C.stride}")
    print(f"  C[1, 0]     : {C[1, 0]}   # row varies fastest")
    print(f"  C[0, 1]     : {C[0, 1]}   # column-step = 4")

    # For compact column-major, flat eval is the identity L[i] == i —
    # the canonical "enumeration walks elements in natural order" check.
    assert C.enumerate() == list(range(len(C)))
    print(f"  C.enumerate() == [0..{len(C) - 1}]  ✓ identity")

    # Same shape, transposed numbering: the grid now counts top-to-bottom
    # down each column (stride 1 on the first axis). Compare with L above.
    C.pretty_print()

    section("Equality and hashing — Layouts are value-typed")
    A = Layout.row_major(4, 8)
    B = Layout.row_major(4, 8)
    print(f"  A == B      : {A == B}")  # canonical (shape, stride) match
    print(f"  hash(A) == hash(B): {hash(A) == hash(B)}")
    d = {A: "v"}
    print(f"  used as dict key: dict[{{A: 'v'}}][B] -> {d[B]!r}")


if __name__ == "__main__":
    main()
