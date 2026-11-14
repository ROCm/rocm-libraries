# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keeping a paged KV tile's V loads inside its sequence.

A sequence's last KV tile spans rows past its length. Their scores are masked, so P = 0, but
0 * NaN is NaN in P @ V: those rows must not bring in the stale slots after the sequence's last key
or blocks past its block-table row. Buffer copies load zero there, as the strided layout's rows
past the sequence do (``strided_kv_offset``): the offset is made out of range, the address math is
left as it is, and each copy adds a compare and a select. Plain loads have no range check, so their
source row is clamped to the sequence's last row; the LDS row stays the same. Where the clamp costs
the load schedule and the tile lies in one page, the rows can instead be loaded as they are and
zeroed in LDS on the last tile. Only a tile reaching past the sequence needs any of these, so kernels
can skip them on the tiles before it.

Block-table reads stay inside the sequence's own entries: past them the table holds the next
sequence's blocks, padding, or nothing (the table's end), so a tile's page lookups past the
sequence read block 0, a valid page whose rows the masks and the V bounds discard.
"""

from __future__ import annotations

from rocke.core.ir import I32, IRBuilder, Value

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


def block_table_max_idx(
    b: IRBuilder,
    seq_base: Value,
    seq_len: Value,
    block_size: int,
    num_seqs: Value,
    bt_stride: Value,
) -> Value:
    """The end of the sequence's own block-table entries (``seq_base`` + ceil(``seq_len`` /
    ``block_size``)), and of the table; wave-uniform."""
    seq_end = b.add(
        seq_base,
        b.div(b.add(seq_len, b.const_i32(block_size - 1)), b.const_i32(block_size)),
    )
    return b.to_sgpr_u32(b.smin(seq_end, b.mul(num_seqs, bt_stride)))


def tile_page_id(
    b: IRBuilder,
    block_tables: Value,
    seq_base: Value,
    max_idx: Value,
    tile: Value,
    blk,
    blocks_per_tile: int,
) -> Value:
    """The page id of block ``blk`` (an int, or a wave-uniform value) of ``tile``: one
    block-table read, 0 past ``max_idx``; wave-uniform."""
    idx = b.add(
        seq_base,
        (
            tile
            if blocks_per_tile == 1
            else b.add(
                b.mul(tile, b.const_i32(blocks_per_tile)),
                blk if isinstance(blk, Value) else b.const_i32(blk),
            )
        ),
    )
    page = b.masked_global_load(
        block_tables,
        idx,
        b.cmp_lt(idx, max_idx),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )
    return b.to_sgpr_u32(page)


class VTileBound:
    """The bound of each bf16/fp16 V copy of ``tile``: past the sequence its offset is out of
    range. Construct it before the tile's copies (it emits the per-tile part), call it per copy.

    A copy of whole rows takes its row as the lane's first row plus an immediate: one
    subtraction per tile, one compare with an immediate per copy. With one row per copy every
    lane's row is 0 and the bound is wave-uniform (unless ``uniform`` is off, where the per-lane
    form measured faster). With ``in_row``, a copy inside one row (``head_size`` a multiple of
    the copy) is bounded by the call's row, wave-uniform. Otherwise each lane bounds its own
    row.
    """

    def __init__(
        self,
        b: IRBuilder,
        seq_len: Value,
        tile: Value,
        tile_size: int,
        head_size: int,
        halves_per_call: int,
        lane_half_base: Value,
        *,
        lane_row: Value | None = None,
        uniform: bool = True,
        in_row: bool = False,
    ) -> None:
        """``lane_row``: the lane's first row in a copy when the kernel has it already."""
        self.b = b
        self.head_size = head_size
        self.halves_per_call = halves_per_call
        self.lane_half_base = lane_half_base
        self.whole_rows = halves_per_call % head_size == 0
        self.in_row = (
            in_row and not self.whole_rows and head_size % halves_per_call == 0
        )
        self.rows = rows_in_sequence(b, seq_len, tile, tile_size)
        if self.whole_rows:
            self.uniform = uniform and lane_row is None and halves_per_call == head_size
            if lane_row is None:
                lane_row = (
                    b.const_i32(0)
                    if self.uniform
                    else b.div(lane_half_base, b.const_i32(head_size))
                )
            self.rows_left = b.sub(self.rows, lane_row)

    def __call__(
        self,
        voff: Value,
        call: int,
        *,
        row: Value | None = None,
        linear_half: Value | None = None,
    ) -> Value:
        """``voff`` of copy ``call`` bounded to the sequence. A copy that ends mid-row takes the
        lane's ``row`` if the kernel has it, else its row from ``linear_half`` (the lane's first
        half in the tile, built here if absent)."""
        b = self.b
        if self.whole_rows:
            return buffer_offset_in_rows_left(
                b,
                voff,
                self.rows_left,
                call * (self.halves_per_call // self.head_size),
                uniform=self.uniform,
            )
        if self.in_row:
            return buffer_offset_in_rows_left(
                b,
                voff,
                self.rows,
                call // (self.head_size // self.halves_per_call),
                uniform=True,
            )
        if row is None:
            if linear_half is None:
                linear_half = b.add(
                    b.const_i32(call * self.halves_per_call), self.lane_half_base
                )
            row = b.div(linear_half, b.const_i32(self.head_size))
        return buffer_offset_in_sequence(b, voff, row, self.rows)
