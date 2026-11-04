# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keeping a paged KV tile's V loads inside its sequence.

A sequence's last KV tile spans rows past its length. Their scores are masked, so P = 0, but
0 * NaN is NaN in P @ V: those rows must not bring in the stale slots after the sequence's last key
or blocks past its block-table row. Buffer copies load zero there, as the strided layout's rows
past the sequence do (``strided_kv_offset``): the offset is made out of range, the address math is
left as it is, and each copy adds a compare and a select. Plain loads have no range check, so their
source row is clamped to the sequence's last row; the LDS row stays the same. Only a tile reaching
past the sequence needs either, so kernels can skip both on the tiles before it.
"""

from __future__ import annotations

from rocke.core.ir import IRBuilder, Value

# Past any buffer resource the kernels build (their num_records stay at or below 0x7FFF0000).
OUT_OF_RANGE_OFFSET = 0x7FFFFFFF


def rows_in_sequence(
    b: IRBuilder, seq_len: Value, tile: Value, tile_size: int
) -> Value:
    """How many of ``tile``'s rows lie inside the sequence (``tile_size`` or more before its last
    tile)."""
    return b.sub(seq_len, b.mul(tile, b.const_i32(tile_size)))


def buffer_offset_in_sequence(
    b: IRBuilder, voff: Value, row: Value, rows: Value
) -> Value:
    """``voff`` for a tile-local ``row`` inside the sequence; an out-of-range offset (the load
    writes zero) past it."""
    return b.select(b.cmp_lt(row, rows), voff, b.const_i32(OUT_OF_RANGE_OFFSET))


def buffer_offset_in_rows_left(
    b: IRBuilder, voff: Value, rows_left: Value, row: int, *, uniform: bool = False
) -> Value:
    """``voff`` while a copy's tile-local ``row`` (an immediate) is below ``rows_left``, the
    lane's rows left in the sequence; out of range past it. ``uniform``: ``rows_left`` is
    wave-uniform, so the bound is a scalar 2**31 added to the offset (offsets stay below 2**31,
    so the sum neither wraps nor stays below any resource's num_records). An ``or`` of bit 31
    computes the same, but made the compiler spill scalar registers in kernels that copy one
    row per call."""
    inside = b.cmp_lt(b.const_i32(row), rows_left)
    if uniform:
        return b.add_nuw(
            voff, b.select(inside, b.const_i32(0), b.const_i32(-(1 << 31)))
        )
    return b.select(inside, voff, b.const_i32(OUT_OF_RANGE_OFFSET))


def last_row_in_sequence(
    b: IRBuilder, seq_len: Value, tile: Value, tile_size: int
) -> Value:
    """``tile``'s last row inside the sequence (non-negative, so divisions of a row clamped to it
    stay shifts)."""
    rows = rows_in_sequence(b, seq_len, tile, tile_size)
    return b.smax(b.sub(rows, b.const_i32(1)), b.const_i32(0))
