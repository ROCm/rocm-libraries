# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prefill-private emitters of the gfx942 MLA prefill kernels.

:mod:`kernels.mla.mla_prefill_gfx942` holds the spec, admission and the
builders; this module holds their single-use phases and the gfx942-only
pieces (the AGPR accumulator's inline asm, the LDS DMA, the permuted PV
gather). The arch-neutral emitters they share with other MLA kernels live in
:mod:`kernels.mla._mla_body`.

Every helper here emits its ops in exactly the order the builders used to emit
them inline, so moving a builder onto a helper never moves the golden IR. Keep
it that way: the IR builder is side-effecting, so reordering two calls inside
a helper is a codegen change.

The forward kernel's PV GEMM permutes latent rows: row ``m`` of latent tile
``4 w + e`` is ``r = 64 w + 4 m + e``, so in the C fragment register ``reg``
of tile ``4 w + e`` holds ``r = 64 w + 16 g + 4 reg + e`` in lane group
``g``. :func:`read_pv_group` produces that order and :func:`store_latent_out`
and :func:`project_wuv_heads` consume it; they are only correct together.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rocke.core.ir import F32, I32, I64, IRBuilder, Type, Value, VectorType
from rocke.helpers.attention_fwd_ext import natural_lse_from_log2_stats
from rocke.helpers.grid import chiplet_transform_chunked_dynamic
from rocke.helpers.schedule import DS_READ as SGB_DS_READ
from rocke.helpers.schedule import MFMA as SGB_MFMA

from kernels.common._read_ahead import read_ahead
from kernels.common._varlen import varlen_q_block
from kernels.mla._mla_body import (
    C_PER_LANE,
    MFMA_K,
    MFMA_M,
    MFMA_N,
    KernelArgs,
    Lanes,
    half4,
    mfma_16x16_c_row,
    mfma_16x16x16,
    on_wave,
    shift,
    wave_tile_row,
    zero_tiles,
)


def _chunks(what: str, count: int, threads: int, width: int) -> int:
    """Chunks per thread for staging ``count`` elements ``width`` at a time.

    The staging loops have no remainder handling, so a count that does not
    divide ``threads * width`` would leave part of the tile uncopied.
    Admission refuses those specs; this guards a caller that skips it.
    """
    if count % (threads * width):
        raise ValueError(
            f"{what}: {count} elements do not divide across {threads} threads "
            f"x {width}-wide chunks"
        )
    return count // (threads * width)


def wt_swizzle(b: IRBuilder, row: Value, col: Value, r_tile: int) -> Value:
    """Permute ``wt_lds`` 4-column groups by ``(row >> 4) & mask``.

    ``wt_lds`` is the epilogue's transposed ``W_UV`` slice, ``[d_v, r_tile]``,
    written one bf16 at a time and read as the MFMA B operand. Within a 16-lane group
    the store's column ``rl`` is uniform and its row is ``lane * 8 + e``, so at
    the ``WT_PAD`` stride (2 mod 32 dwords) the row term ``lane * 16`` reaches
    only two banks. XORing the column group with row bits 4 and up -- lane bits
    1-3, disjoint from the stride's lane bit 0 -- spreads it across 16.

    On the read side ``row >> 4`` is the MFMA n-tile, constant across the 16
    lanes of one read, so the twist only relabels which groups a tile reads and
    the read keeps its conflict-free bank pattern. Groups of four keep the
    ``n=4`` read contiguous and 8 B-aligned. The mask is the widest power of two
    (at most 8) dividing the group count, so the XOR never leaves the row.
    """
    groups = r_tile // 4
    mask = 1
    while mask < 8 and groups % (mask * 2) == 0:
        mask *= 2
    twist = b.land(b.lshr(row, b.const_i32(4)), b.const_i32(mask - 1))
    grp = b.xor(b.lshr(col, b.const_i32(2)), twist)
    return b.lor(b.shl(grp, b.const_i32(2)), b.land(col, b.const_i32(3)))


def reversed_q_block(
    b: IRBuilder,
    args: KernelArgs,
    *,
    block_q: int,
    num_groups: int = 0,
    num_xcds: int = 0,
    xcd_chunk: int = 0,
) -> tuple[Value, Value]:
    """``(q_block_global_idx, group)`` for this workgroup, heaviest first.

    Block ids are walked in reverse. With the causal loop bound a query tile's
    work grows with its position in the sequence, and workgroups launch in
    block-id order, so reversing puts the heaviest tiles first and leaves the
    light ones to fill the tail. The grid's x extent is the AITER count
    ``total_q // block_q + num_seqs``, and ``total_q == cu_seqlens_q[num_seqs]``.
    ``group`` is ``blockIdx.y`` -- the head, or head group.

    With ``num_xcds > 1`` the workgroups are remapped chiplet-aware: the
    dispatcher deals them round-robin across XCDs, so in launch order every
    XCD would see every one of the ``num_groups`` groups and its L2 would hold
    all of their weights. Remapped, each XCD walks ``xcd_chunk``-long runs of a
    group-major order.
    """
    total_q = b.global_load_i32(args.cu_q, args.num_seqs)
    num_q_blocks = b.add(b.div(total_q, b.const_i32(block_q)), args.num_seqs)
    # A grid wider than num_q_blocks would walk the reversed index below 0
    # (seq_idx -1, negative token offsets), or past the last group through the
    # chiplet remap. Uniform, so no partial workgroup.
    with b.scf_if(b.cmp_ge(b.block_id_x(), num_q_blocks)):
        b.ret()
    if num_xcds > 1:
        n_grp = b.const_i32(num_groups)
        hw_id = b.add(b.block_id_x(), b.mul(b.block_id_y(), num_q_blocks))
        logical = chiplet_transform_chunked_dynamic(
            b,
            hw_id,
            num_wgs=b.mul(num_q_blocks, n_grp),
            num_xcds=num_xcds,
            chunk_size=xcd_chunk,
        )
        group = b.div(logical, num_q_blocks)
        idx = b.sub(b.sub(num_q_blocks, b.const_i32(1)), b.mod(logical, num_q_blocks))
        return idx, group
    idx = b.sub(b.sub(num_q_blocks, b.const_i32(1)), b.block_id_x())
    return idx, b.block_id_y()


@dataclass(frozen=True)
class QTile:
    """The workgroup's query tile inside its sequence."""

    seq_idx: Value
    cu_q_start: Value
    s_q: Value
    # First query row of the tile, sequence-local.
    qb_start: Value


def varlen_tile(
    b: IRBuilder,
    args: KernelArgs,
    q_block_global_idx: Value,
    *,
    block_q: int,
    iters: int,
) -> QTile:
    """Decode the packed q-block and exit if it is a padding block.

    The exit is uniform across the workgroup (block ids only), so no barrier
    below is reached by a partial workgroup.
    """
    vq = varlen_q_block(
        b, args.cu_q, q_block_global_idx, args.num_seqs, block_q=block_q, iters=iters
    )
    qb_start = b.mul(vq.q_block_local_idx, b.const_i32(block_q))
    with b.scf_if(b.cmp_ge(qb_start, vq.q_len)):
        b.ret()
    return QTile(vq.seq_idx, vq.cu_q_start, vq.q_len, qb_start)


def seq_len_k(b: IRBuilder, args: KernelArgs, seq_idx: Value) -> Value:
    """The sequence's key length, ``cu_k[seq + 1] - cu_k[seq]``."""
    cu_k_start = b.global_load_i32(args.cu_k, seq_idx)
    cu_k_stop = b.global_load_i32(args.cu_k, b.add(seq_idx, b.const_i32(1)))
    return b.sub(cu_k_stop, cu_k_start)


@dataclass(frozen=True)
class Keys:
    """The sequence's key range and its bottom-right causal offset."""

    s_k: Value
    # Bottom-right causal alignment: query row ``i`` of the sequence attends to
    # keys ``0 .. i + context_off``. Negative when S_k < S_q, which is exactly
    # what produces fully masked leading rows (exact zeros / exact -inf).
    context_off: Value


def causal_keys(b: IRBuilder, args: KernelArgs, tile: QTile) -> Keys:
    """Key length and causal offset of the tile's sequence."""
    s_k = seq_len_k(b, args, tile.seq_idx)
    return Keys(s_k, b.sub(s_k, tile.s_q))


def causal_k_tiles(
    b: IRBuilder, q_start: Value, tile: QTile, keys: Keys, *, block_q: int, block_k: int
) -> Value:
    """Key tiles the causal window of rows ``[q_start, q_start + block_q)`` reaches.

    No row of the tile sees a key at or past ``q_end + context_off``, where
    ``q_end`` is the tile's last real row + 1, so the tiles beyond it are fully
    masked and are skipped rather than computed and discarded. ``q_end <= s_q``
    keeps the bound within ``s_k``; the clamp at 0 covers tiles whose rows are
    all fully masked, which then run zero iterations and finish with ``l == 0``
    exactly as masking would leave them. The per-element masks still handle
    the ragged edge tiles.
    """
    q_end = b.select(
        b.cmp_lt(b.add(q_start, b.const_i32(block_q)), tile.s_q),
        b.add(q_start, b.const_i32(block_q)),
        tile.s_q,
    )
    k_end = b.add(q_end, keys.context_off)
    k_end = b.select(b.cmp_gt(k_end, b.const_i32(0)), k_end, b.const_i32(0))
    return b.div(b.add(k_end, b.const_i32(block_k - 1)), b.const_i32(block_k))


def full_k_tiles(
    b: IRBuilder, q_start: Value, keys: Keys, n_k_tiles: Value, *, block_k: int
) -> Value:
    """Leading key tiles every row of the tile can fully see.

    A key tile is fully visible when its last key is below ``s_k`` and at or
    below the causal bound of the tile's first query row. Only the last tile
    or two can be partially masked, so those alone need per-element masking.
    """
    n_full = b.div(
        b.add(q_start, b.add(keys.context_off, b.const_i32(1))), b.const_i32(block_k)
    )
    k_whole = b.div(keys.s_k, b.const_i32(block_k))
    n_full = b.select(b.cmp_lt(n_full, k_whole), n_full, k_whole)
    n_full = b.select(b.cmp_gt(n_full, b.const_i32(0)), n_full, b.const_i32(0))
    return b.select(b.cmp_lt(n_full, n_k_tiles), n_full, n_k_tiles)


@dataclass(frozen=True)
class PageTable:
    """The sequence's row of the paged block table."""

    b: IRBuilder
    block_table: Value
    seq_idx: Value
    stride: Value

    def load(self, tile: Value, *, guard: Value | None = None) -> Value:
        """The block-table entry for ``tile``, raw (per lane, not yet uniform).

        With ``guard`` (a zero-trip loop) the block-table slot may be
        uninitialised: page 0 is always mapped, so the load stays in bounds,
        and a zero-trip loop never consumes the value.
        """
        b = self.b
        page = b.global_load_i32(
            self.block_table, b.add(b.mul(self.seq_idx, self.stride), tile)
        )
        if guard is not None:
            page = b.select(guard, page, b.const_i32(0))
        return page


def clamp_tile(b: IRBuilder, t: int, last_tile: Value) -> Value:
    """Constant tile ``t``, clamped to ``[0, last_tile]``."""
    idx = b.select(b.cmp_lt(b.const_i32(t), last_tile), b.const_i32(t), last_tile)
    return b.select(b.cmp_gt(idx, b.const_i32(0)), idx, b.const_i32(0))


def tile_ahead(b: IRBuilder, k_tile: Value, ahead: int, n_k_tiles: Value) -> Value:
    """Tile ``k_tile + ahead``, clamped to the last tile.

    Past the end the index re-reads the last tile's page, which is known to
    be mapped; the value is never consumed.
    """
    nxt = b.add(k_tile, b.const_i32(ahead))
    return b.select(b.cmp_lt(nxt, n_k_tiles), nxt, b.sub(n_k_tiles, b.const_i32(1)))


def page_offsets(
    b: IRBuilder, page: Value, *, page_size: int, r_kv: int, d_rope: int
) -> tuple[Value, Value]:
    """Byte offsets of ``page`` in the ``c_kv`` and ``k_rope`` buffers.

    ``page`` is made wave-uniform first, so both offsets are scalar and can
    ride in a buffer load's SGPR ``soffset``.
    """
    page = b.readfirstlane(page)
    return (
        b.mul(page, b.const_i32(page_size * r_kv * 2)),
        b.mul(page, b.const_i32(page_size * d_rope * 2)),
    )


def kv_buffer_rsrcs(
    b: IRBuilder, args: KernelArgs, num_records: int
) -> tuple[Value, Value]:
    """Buffer resources over the paged ``c_kv`` and ``k_rope`` tensors."""
    big_bytes = b.const_i32(num_records)
    return b.buffer_rsrc(args.c_kv, big_bytes), b.buffer_rsrc(args.k_rope, big_bytes)


# ---------------------------------------------------------------------------
# Query rows and staging copies
# ---------------------------------------------------------------------------


def row_of(b: IRBuilder, base: Value, lane_row: Value, mt: int) -> Value:
    """Query row of M tile ``mt`` held by this lane: ``base + 16 mt + lane_row``."""
    return b.add(base, shift(b, mt * MFMA_M, lane_row))


@dataclass(frozen=True)
class QRows:
    """Global addressing of the query-indexed tensors (``q``, ``out``, ``lse``).

    Rows past ``S_q`` are clamped onto the last valid token when loading
    (:meth:`clamped_token`) rather than masked: they still compute, every
    result for them is dropped by the store predicates, and clamping keeps the
    load in-bounds without a masked vector load.
    """

    b: IRBuilder
    tile: QTile
    head: Value
    num_heads: int
    last: Value

    def clamped_token(self, q_local: Value) -> Value:
        """Packed token of row ``q_local``, clamped onto the last real row."""
        b = self.b
        q_local = b.select(b.cmp_lt(q_local, self.tile.s_q), q_local, self.last)
        return b.add(self.tile.cu_q_start, q_local)

    def head_row(self, token: Value, cols: int | Value) -> Value:
        """Element offset of ``[token, head, 0]`` in a ``[*, H, cols]`` tensor."""
        b = self.b
        row = b.add(b.mul(token, b.const_i32(self.num_heads)), self.head)
        return b.mul(row, b.const_i32(cols) if isinstance(cols, int) else cols)

    def offset(
        self,
        token: Value,
        cols: int | Value,
        col: Value | Callable[[], Value],
    ) -> Value:
        """Element offset of ``[token, head, col]``; a callable ``col`` is
        emitted after the row offset."""
        row = self.head_row(token, cols)
        return self.b.add(row, col() if callable(col) else col)

    def lane_q_offset(
        self,
        q_start: Value,
        lane_row: Value,
        mt: int,
        cols: int,
        col: Value | Callable[[], Value],
    ) -> Value:
        """Offset of column ``col`` of this lane's (clamped) row of M tile ``mt``."""
        token = self.clamped_token(row_of(self.b, q_start, lane_row, mt))
        return self.offset(token, cols, col)


def query_rows(b: IRBuilder, tile: QTile, head: Value, num_heads: int) -> QRows:
    """:class:`QRows` for the tile; emits the clamp's last-row index."""
    return QRows(b, tile, head, num_heads, b.sub(tile.s_q, b.const_i32(1)))


def split_coords(
    b: IRBuilder, c: Value, per_row: int, width: int
) -> tuple[Value, Value]:
    """``(c // per_row, (c % per_row) * width)`` -- row and first column of
    chunk ``c`` of a row-major tile ``per_row`` chunks of ``width`` wide."""
    m = b.div(c, b.const_i32(per_row))
    return m, b.mul(b.mod(c, b.const_i32(per_row)), b.const_i32(width))


def chunk_coords(
    b: IRBuilder, tid: Value, j: int, threads: int, per_row: int, width: int
) -> tuple[Value, Value]:
    """:func:`split_coords` of thread ``tid``'s ``j``-th chunk."""
    return split_coords(b, b.add(tid, b.const_i32(j * threads)), per_row, width)


def byte_offset(b: IRBuilder, m: Value, row_elems: int, col: Value) -> Value:
    """Byte offset of bf16 element ``[m, col]`` in rows of ``row_elems``."""
    return b.mul(b.add(b.mul(m, b.const_i32(row_elems)), col), b.const_i32(2))


def copy_rows_to_lds(
    b: IRBuilder,
    src: Value,
    dst: Value,
    *,
    tid: Value,
    threads: int,
    base: Value,
    rows: int,
    cols: int,
    width: int,
    dtype: Type,
    dst_col: int | None = None,
) -> None:
    """``dst[m, dst_col + c] = src[base + m * cols + c]`` for a ``[rows, cols]`` tile.

    Each thread copies ``width``-wide chunks, row-major across the workgroup.
    """
    per_row = cols // width
    for j in range(_chunks("copy_rows_to_lds", rows * cols, threads, width)):
        m, c0 = chunk_coords(b, tid, j, threads, per_row, width)
        idx = b.add(b.add(base, b.mul(m, b.const_i32(cols))), c0)
        col = c0 if dst_col is None else b.add(c0, b.const_i32(dst_col))
        b.smem_store_vN(dst, [m, col], b.global_load_vN(src, idx, dtype, width), width)


def stage_q_cols(
    b: IRBuilder,
    q_ptr: Value,
    dst: Value,
    qrows: QRows,
    *,
    tid: Value,
    threads: int,
    rows: int,
    cols: int,
    row_cols: int,
    src_col: int,
    dst_col: int,
    dtype: Type,
) -> None:
    """Stage ``q[token, head, src_col : src_col + cols]`` of the tile's ``rows``
    query rows into ``dst[:, dst_col :]``, four columns per chunk."""
    per_row = cols // 4
    for j in range(_chunks("stage_q_cols", rows * cols, threads, 4)):
        m, d0 = chunk_coords(b, tid, j, threads, per_row, 4)
        token = qrows.clamped_token(b.add(qrows.tile.qb_start, m))
        idx = qrows.offset(token, row_cols, lambda d0=d0: shift(b, src_col, d0))
        v = b.global_load_vN(q_ptr, idx, dtype, 4)
        b.smem_store_vN(dst, [m, shift(b, dst_col, d0)], v, 4)


def store_c_frag(
    b: IRBuilder, lds: Value, frag: Value, lane: Value, *, col: Value, row_off: int = 0
) -> None:
    """Scatter one f32x4 C fragment into ``lds`` as bf16, element-wise.

    The C decode puts element ``reg`` at ``[row_off + c_row(lane, reg)][col]``.
    """
    packed = b.vec_trunc_f32_to_bf16(frag)
    for reg in range(C_PER_LANE):
        row = shift(b, row_off, mfma_16x16_c_row(b, lane, reg))
        b.smem_store_vN(lds, [row, col], b.vec_extract(packed, reg), 1)


# ---------------------------------------------------------------------------
# GEMM pieces
# ---------------------------------------------------------------------------


def w_head_base(b: IRBuilder, head: Value, rows: int, w_cols: int) -> Value:
    """Element offset of ``W_UK[head]`` -- ``rows`` rows of ``w_cols``."""
    return b.mul(b.mul(head, b.const_i32(rows)), b.const_i32(w_cols))


def w_slice_index(
    b: IRBuilder,
    head_base: Value,
    rl: Value,
    r_base: int,
    w_cols: int,
    n0: Value,
    col_offset: int | None = None,
) -> Value:
    """Element offset of ``W_UK[head, r_base + rl, col_offset + n0]``.

    A ``col_offset`` of ``None`` emits no offset add at all.
    """
    row = b.add(head_base, b.mul(b.add(rl, b.const_i32(r_base)), b.const_i32(w_cols)))
    return b.add(row, n0 if col_offset is None else b.add(n0, b.const_i32(col_offset)))


def expand_latent(
    b: IRBuilder,
    *,
    w_uk: Value,
    c_lds: Value,
    wt_lds: Value,
    tid: Value,
    wave: Value,
    lane_row: Value,
    lane_kq: Value,
    w_head_base: Value,
    col_offset: int | None,
    dtype: Type,
    threads: int,
    slices: range,
    c_col_base: int,
    r_tile: int,
    d_out: int,
    w_cols: int,
    exp_k_iters: int,
    n_tiles_per_wave: int,
    m_tiles: int = 1,
    out_transposed: bool = False,
    swizzle: bool = True,
    acc: list[list[Value]] | None = None,
) -> list[list[Value]]:
    """``C_lds @ W_UK[head, :, col_offset : col_offset + d_out]``, in registers.

    Only the r-slices in ``slices`` are walked; ``C_lds`` holds latent column
    ``r`` at column ``r - c_col_base``, and the products accumulate onto
    ``acc`` (zeros when ``None``) -- so a caller can stage the latent through
    ``C_lds`` one column range at a time.

    Walks the latent dimension in ``r_tile`` slices, staging each slice of the
    weight through ``wt_lds`` transposed on the store path (MFMA reads the B
    operand as ``[n][k]``, and gfx942 has no ``ds_read_tr``). The caller owns the
    C-layout writeback, and the barrier that opens each slice lives here so a
    caller that shares the ``wt_lds`` slot with something else needs no extra
    fencing of its own.

    ``m_tiles`` register-blocks the M axis: the weight slice in ``wt_lds`` is
    read once and fed to every M tile, so a taller ``C_lds`` costs extra MFMAs
    and accumulators but no extra LDS traffic on the B side.

    ``swizzle`` applies :func:`wt_swizzle` to the ``wt_lds`` columns; the
    score probe stages without it. A ``col_offset`` of ``None`` is column 0
    with no offset add emitted.

    Returns ``acc[m_tile][n_tile]``, one f32x4 per (M tile, n-tile owned by the
    calling wave).
    """
    if acc is None:
        acc = zero_tiles(b, m_tiles, n_tiles_per_wave)
    wt_chunks = _chunks("expand_latent W slice", r_tile * d_out, threads, 8)
    wt_chunks_per_row = d_out // 8
    for s in slices:
        r_base = s * r_tile
        b.sync()  # WAR: previous slice's (or previous GEMM's) readers must retire
        for j in range(wt_chunks):
            rl, n0 = chunk_coords(b, tid, j, threads, wt_chunks_per_row, 8)
            idx = w_slice_index(b, w_head_base, rl, r_base, w_cols, n0, col_offset)
            v8 = b.global_load_vN(w_uk, idx, dtype, 8)
            for e in range(8):
                row = b.add(n0, b.const_i32(e))
                b.smem_store_vN(
                    wt_lds,
                    [row, wt_swizzle(b, row, rl, r_tile) if swizzle else rl],
                    b.vec_extract(v8, e),
                    1,
                )
        b.sync()
        for kk in range(exp_k_iters):
            a_col = b.add(b.const_i32(r_base - c_col_base + kk * MFMA_K), lane_kq)
            a_vs = [
                b.smem_load_vN(
                    c_lds, shift(b, mt * MFMA_M, lane_row), a_col, dtype=dtype, n=4
                )
                for mt in range(m_tiles)
            ]
            b_col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
            for t in range(n_tiles_per_wave):
                b_row = wave_tile_row(b, wave, n_tiles_per_wave, t, lane_row)
                b_v = b.smem_load_vN(
                    wt_lds,
                    b_row,
                    wt_swizzle(b, b_row, b_col, r_tile) if swizzle else b_col,
                    dtype=dtype,
                    n=4,
                )
                for mt in range(m_tiles):
                    if out_transposed:
                        # outᵀ = W_UVᵀ · accᵀ: the same two fragments, swapped,
                        # so the C fragment is (m = v, n = query).
                        acc[mt][t] = mfma_16x16x16(b, dtype, b_v, a_vs[mt], acc[mt][t])
                    else:
                        acc[mt][t] = mfma_16x16x16(b, dtype, a_vs[mt], b_v, acc[mt][t])
    return acc


def absorb_query(
    b: IRBuilder,
    *,
    w_uk: Value,
    q_frags: list[list[Value]],
    wq_lds: Value,
    qa_lds: Value,
    tid: Value,
    lane: Value,
    wave: Value,
    lane_row: Value,
    lane_kq: Value,
    w_head_base: Value,
    dtype: Type,
    threads: int,
    slices: range,
    qa_col_base: int,
    r_tile: int,
    d_nope: int,
    w_cols: int,
    absorb_k_iters: int,
    n_tiles_per_wave: int,
    m_tiles: int,
    active_waves: int | None = None,
) -> None:
    """``Qa[q, r] = sum_n Q_nope[q, n] * W_UK[head, r, n]``, written to ``qa_lds``.

    ``q_frags[mt][kk]`` is the ``Q_nope`` A operand, already in registers.
    Only the r-slices in ``slices`` are computed, and latent column ``r`` lands
    in ``qa_lds`` column ``r - qa_col_base``. Each weight slice is staged once
    and fed to every M tile.

    The W_UK absorption of the query side. Because
    ``S[q,k] = Q_nope[q] . K_nope[k] = Q_nope[q] . (C[k] @ W_UK_nope)``
    equals ``Qa[q] . C[k]``, folding ``W_UK`` into ``Q`` once per workgroup lets
    the whole key loop run in the 512-wide latent space.

    Unlike :func:`expand_latent` this needs **no** store-path transpose: MFMA
    reads the B operand as ``[n][k]``, here ``[r][nope]``, which is exactly how
    ``w_uk[head, r, :d_nope]`` is already laid out. The weight slice therefore
    goes to LDS with contiguous 8-wide stores.

    Walks ``r`` in ``r_tile`` slices. Each slice's accumulator is complete after
    the full ``K = d_nope`` reduction, so it is written straight through to
    ``qa_lds`` and the registers are reused by the next slice.
    """
    wq_chunks_per_row = d_nope // 8
    for s in slices:
        r_base = s * r_tile
        b.sync()  # WAR on wq_lds; also publishes the Q_rope staging on the first
        for j in range(_chunks("absorb_query W slice", r_tile * d_nope, threads, 8)):
            rl, n0 = chunk_coords(b, tid, j, threads, wq_chunks_per_row, 8)
            idx = w_slice_index(b, w_head_base, rl, r_base, w_cols, n0)
            b.smem_store_vN(wq_lds, [rl, n0], b.global_load_vN(w_uk, idx, dtype, 8), 8)
        b.sync()

        # With fewer n-tiles than waves (``active_waves``), only the first
        # ``active_waves`` waves compute; the rest skip to the next slice's
        # barrier. The gated region holds no barrier and its only outputs are
        # qa_lds stores, so carrying no results out of the ``scf.if`` is fine.
        def _slice_compute(r_base: int):
            acc = zero_tiles(b, m_tiles, n_tiles_per_wave)
            for kk in range(absorb_k_iters):
                col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
                a_vs = [q_frags[mt][kk] for mt in range(m_tiles)]
                for t in range(n_tiles_per_wave):
                    b_row = wave_tile_row(b, wave, n_tiles_per_wave, t, lane_row)
                    b_v = b.smem_load_vN(wq_lds, b_row, col, dtype=dtype, n=4)
                    for mt in range(m_tiles):
                        acc[mt][t] = mfma_16x16x16(b, dtype, a_vs[mt], b_v, acc[mt][t])
            # C decodes as [m = q][n = r - r_base]; scatter it into the latent query.
            for t in range(n_tiles_per_wave):
                col = shift(
                    b,
                    r_base - qa_col_base,
                    wave_tile_row(b, wave, n_tiles_per_wave, t, lane_row),
                )
                for mt in range(m_tiles):
                    store_c_frag(
                        b, qa_lds, acc[mt][t], lane, col=col, row_off=mt * MFMA_M
                    )

        if active_waves is None:
            _slice_compute(r_base)
        else:
            with b.scf_if(b.cmp_lt(wave, b.const_i32(active_waves))):
                _slice_compute(r_base)


def gather_keys_t(
    b: IRBuilder, kv_lds: Value, a_col: Value, b_row: Value, dtype: Type
) -> Value:
    """``Cᵀ[r][key]`` A fragment gathered from ``kv_lds`` (``[key][r]``).

    Four keys at a time, one bf16 per key: the transpose happens on the read
    path, because gfx942 has no ``ds_read_tr``.
    """
    return b.vec_pack(
        [
            b.vec_extract(
                b.smem_load_vN(kv_lds, shift(b, i, a_col), b_row, dtype=dtype, n=1), 0
            )
            for i in range(4)
        ],
        dtype,
    )


def _agpr_clobbers(base: int, n: int) -> str:
    return ",".join(f"~{{a{base + i}}}" for i in range(n))


class AgprAcc:
    """Latent accumulator held in fixed AGPRs that LLVM never sees as values.

    With 32 rows per wave the accumulator is 256 registers. Carried as IR
    values -- loop iteration arguments -- the conditional lazy rescale makes
    LLVM hoist AGPR-to-VGPR copies of it to the top of the loop body and spill
    the absorbed query; no register-class hint avoids that. So the
    accumulator lives in ``a[0:N)`` and is only touched by inline asm: the PV
    MFMA reads and writes it in place, the rescale scales it in place under a
    wave-uniform branch, and the epilogue reads it out once. The kernel runs
    with ``agpr_alloc = (0, 0)`` so the backend allocates no AGPRs of its own
    (on llvm22 and later; llvm20 ignores it, which is why admission refuses
    llvm20). The asm is opaque to the hazard recognizer, so the helper pads
    its own hazards: one ``s_nop`` before each MFMA (VALU-written operands)
    and a fence before the first read after the MFMAs.
    """

    def __init__(self, b: IRBuilder, n_tiles: int):
        self.b = b
        self.n = 4 * n_tiles
        b.inline_asm(
            "\n\t".join(f"v_accvgpr_write_b32 a{i}, 0" for i in range(self.n)),
            _agpr_clobbers(0, self.n),
            [],
            sideeffect=True,
        )

    def mfma(self, tile: int, a_v: Value, b_v: Value) -> None:
        """``acc[tile] += a_v · b_v`` (``v_mfma_f32_16x16x16_bf16``)."""
        b = self.b
        ta = 4 * tile
        v2 = VectorType(I32, 2)
        b.inline_asm(
            f"s_nop 1\n\tv_mfma_f32_16x16x16_bf16 a[{ta}:{ta + 3}], $0, $1, "
            f"a[{ta}:{ta + 3}]",
            "v,v," + _agpr_clobbers(ta, 4),
            [b.vec_bitcast(a_v, v2), b.vec_bitcast(b_v, v2)],
            sideeffect=True,
        )

    def scale(self, tile: int, alpha: Value) -> None:
        """``acc[tile] *= alpha`` in place."""
        ta = 4 * tile
        lines = []
        for r in range(4):
            lines += [
                f"v_accvgpr_read_b32 $0, a{ta + r}",
                "v_mul_f32 $0, $0, $1",
                f"v_accvgpr_write_b32 a{ta + r}, $0",
            ]
        self.b.inline_asm(
            "\n\t".join(lines),
            "=&v,v," + _agpr_clobbers(ta, 4),
            [alpha],
            result_type=F32,
            sideeffect=True,
        )

    def scale_if(self, flag: Value, tiles: range, alpha: Value) -> None:
        """``acc[t] *= alpha`` for ``t`` in ``tiles``, under the wave-uniform
        branch ``flag != 0``: in place, so nothing is carried."""
        b = self.b
        with b.scf_if(b.cmp_ne(flag, b.const_i32(0))):
            for t in tiles:
                self.scale(t, alpha)

    def fence(self) -> None:
        """Wait states between the last MFMA write and a VALU read of it."""
        self.b.inline_asm("s_nop 7\n\ts_nop 7", "", [], sideeffect=True)

    def read(self, tile: int) -> Value:
        """Tile ``tile`` read out of the AGPR file as an f32x4."""
        b = self.b
        ta = 4 * tile
        return b.vec_pack(
            [
                b.inline_asm(
                    f"v_accvgpr_read_b32 $0, a{ta + r}",
                    "=v",
                    [],
                    result_type=F32,
                    sideeffect=True,
                )
                for r in range(4)
            ],
            F32,
        )

    def view(self, m_tiles: int, n_tiles: int) -> list[_AgprRow]:
        """``acc[mt][t]`` that reads each tile out of the file on first use."""
        return [_AgprRow(self, mt * n_tiles) for mt in range(m_tiles)]


@dataclass
class _AgprRow:
    acc: AgprAcc
    base: int
    cache: dict = field(default_factory=dict)

    def __getitem__(self, t: int) -> Value:
        if t not in self.cache:
            self.cache[t] = self.acc.read(self.base + t)
        return self.cache[t]


# ---------------------------------------------------------------------------
# Epilogue stores
# ---------------------------------------------------------------------------


def scaled_pack(b: IRBuilder, elems: Sequence[tuple[Value, int]], inv: Value) -> Value:
    """``vec_pack([v[i] * inv for (v, i) in elems])`` as f32."""
    return b.vec_pack([b.fmul(b.vec_extract(v, i), inv) for v, i in elems], F32)


def store_out_frag(
    b: IRBuilder,
    out: Value,
    frag: Value,
    inv: Value,
    qrows: QRows,
    *,
    q_local: Value,
    col: Value,
    d_v: int,
) -> None:
    """Normalize one ``outᵀ`` fragment by ``1/l`` and store it as bf16.

    The lane holds query ``q_local`` and four consecutive ``v`` at ``col``, so
    the store is one 8 B write, predicated on the row being real.
    """
    vals = scaled_pack(b, [(frag, r) for r in range(C_PER_LANE)], inv)
    with b.scf_if(b.cmp_lt(q_local, qrows.tile.s_q)):
        token = b.add(qrows.tile.cu_q_start, q_local)
        idx = b.add(b.mul(token, b.const_i32(qrows.num_heads * d_v)), col)
        idx = b.add(idx, b.mul(qrows.head, b.const_i32(d_v)))
        b.global_store_vN(out, idx, b.vec_trunc_f32_to_bf16(vals), C_PER_LANE, align=8)


def store_lse(
    b: IRBuilder,
    lse: Value,
    ms: Sequence[Value],
    ls: Sequence[Value],
    qrows: QRows,
    *,
    q_start: Value,
    lane_row: Value,
    lane: Value,
    wave: Value | None = None,
) -> None:
    """Natural-log LSE of every M tile's rows, one writer per query.

    The writer is the first 16-lane group -- of wave 0 when ``wave`` is given
    (every wave holds the same state), else of each wave.
    """
    if wave is not None:
        with on_wave(b, wave, 0):
            store_lse(
                b, lse, ms, ls, qrows, q_start=q_start, lane_row=lane_row, lane=lane
            )
        return
    with b.scf_if(b.cmp_lt(lane, b.const_i32(16))):
        for mt, (m, l) in enumerate(zip(ms, ls, strict=True)):
            q_local = row_of(b, q_start, lane_row, mt)
            value = natural_lse_from_log2_stats(b, m, l)
            with b.scf_if(b.cmp_lt(q_local, qrows.tile.s_q)):
                token = b.add(qrows.tile.cu_q_start, q_local)
                idx = b.add(b.mul(token, b.const_i32(qrows.num_heads)), qrows.head)
                b.global_store(lse, idx, value, align=4)


def load_q_frags(
    b: IRBuilder, q_ptr: Value, base: Value, *, n: int, dtype: Type, col0: int = 0
) -> list[Value]:
    """``n`` four-wide fragments at columns ``col0 + 16 kk`` past ``base``."""
    return [
        b.global_load_vN(q_ptr, shift(b, col0 + kk * MFMA_K, base), dtype, 4)
        for kk in range(n)
    ]


def lds_gemm_tile(
    b: IRBuilder, a_lds: Value, b_lds: Value, lanes: Lanes, *, k_iters: int, dtype: Type
) -> Value:
    """One ``16x16`` C tile, ``A[:, k] · B[:, k]`` over ``k_iters`` LDS k-steps."""
    acc = b.zero_vec_f32(4)
    for kk in range(k_iters):
        col = b.add(b.const_i32(kk * MFMA_K), lanes.lane_kq)
        a_v = b.smem_load_vN(a_lds, lanes.lane_row, col, dtype=dtype, n=4)
        b_v = b.smem_load_vN(b_lds, lanes.lane_row, col, dtype=dtype, n=4)
        acc = mfma_16x16x16(b, dtype, a_v, b_v, acc)
    return acc


def store_probe_scores(
    b: IRBuilder,
    scores: Value,
    s_acc: Value,
    qrows: QRows,
    lanes: Lanes,
    *,
    kb_start: Value,
    s_k: Value,
    scale: Value,
    row_stride: Value,
) -> None:
    """``scores[token, head, k] = scale * S`` for the in-range (query, key)
    pairs of one ``16x16`` tile (``m = query``, ``n = key``)."""
    k_abs = b.add(kb_start, lanes.lane_row)
    for reg in range(C_PER_LANE):
        row = mfma_16x16_c_row(b, lanes.lane, reg)
        q_local = b.add(qrows.tile.qb_start, row)
        value = b.fmul(b.vec_extract(s_acc, reg), scale)
        # Both bounds as two nested ifs (the second opens inside the first),
        # the predicate shape the probe's golden IR pins.
        with (
            b.scf_if(b.cmp_lt(q_local, qrows.tile.s_q)),
            b.scf_if(b.cmp_lt(k_abs, s_k)),
        ):
            token = b.add(qrows.tile.cu_q_start, q_local)
            idx = qrows.offset(token, row_stride, k_abs)
            b.global_store(scores, idx, value, align=4)


# ---------------------------------------------------------------------------
# One-head layout phases (``build_mla_prefill_fwd``)
# ---------------------------------------------------------------------------


def wave_k_col(b: IRBuilder, rank: Value, k_steps: int, lane_kq: Value) -> Value:
    """First column of the ``rank``-th run of ``k_steps`` 16-wide k-steps."""
    return b.add(b.mul(rank, b.const_i32(k_steps * MFMA_K)), lane_kq)


def window_rank(b: IRBuilder, wave: Value, waves_per_win: int, parts: int) -> Value:
    """The wave's rank inside its ``qa_lds`` window."""
    return wave if parts == 1 else b.mod(wave, b.const_i32(waves_per_win))


def in_window(
    b: IRBuilder, wave: Value, waves_per_win: int, parts: int, p: int
) -> Value | None:
    """Whether window ``p`` is the wave's own; ``None`` with a single window."""
    if parts == 1:
        return None
    return b.cmp_eq(b.div(wave, b.const_i32(waves_per_win)), b.const_i32(p))


def take_window_frags(
    b: IRBuilder,
    qa_lds: Value,
    qa_frags: list[list[Value]],
    lane_row: Value,
    win_col: Value,
    *,
    in_win: Value | None,
    first: bool,
    dtype: Type,
) -> None:
    """Read the published ``qa_lds`` window into ``qa_frags[kk][mt]``.

    Every wave reads it -- ``scf.if`` carries no results -- and keeps the value
    only where ``in_win`` (the window is its own); the first window seeds the
    fragments unconditionally.
    """
    for kk, frags in enumerate(qa_frags):
        for mt in range(len(frags)):
            v = b.smem_load_vN(
                qa_lds,
                shift(b, mt * MFMA_M, lane_row),
                shift(b, kk * MFMA_K, win_col),
                dtype=dtype,
                n=4,
            )
            frags[mt] = v if in_win is None or first else b.select(in_win, v, frags[mt])


def split_k_scores(
    b: IRBuilder,
    kv_lds: Value,
    qa_frags: list[list[Value]],
    lane_row: Value,
    score_col: Value,
    *,
    k_steps: int,
    m_tiles: int,
    k_tiles: int,
    sgb_group: int,
    dtype: Type,
) -> list[list[Value]]:
    """This wave's ``k_steps``-step partial of ``Sᵀ`` (``m = key``, ``n = query``).

    ``kv_lds`` is the A operand and the query fragment the B operand -- the
    same two fragments as ``S``, swapped. Every ``sgb_group`` steps the next
    group's LDS reads are hoisted above the MFMAs they feed; the hint is
    skipped when the group does not divide the trip count -- a partial
    trailing block measured worse than no hint.
    """
    s_acc = zero_tiles(b, m_tiles, k_tiles)
    for kk in range(k_steps):
        col = shift(b, kk * MFMA_K, score_col)
        a_vs = qa_frags[kk]
        for kt in range(k_tiles):
            b_v = b.smem_load_vN(
                kv_lds, shift(b, kt * MFMA_N, lane_row), col, dtype=dtype, n=4
            )
            for mt in range(m_tiles):
                s_acc[mt][kt] = mfma_16x16x16(b, dtype, b_v, a_vs[mt], s_acc[mt][kt])
        if k_steps % sgb_group == 0 and (kk + 1) % sgb_group == 0:
            b.sched_group_barrier(SGB_DS_READ, sgb_group * k_tiles, 0)
            b.sched_group_barrier(SGB_MFMA, sgb_group * m_tiles * k_tiles, 0)
    return s_acc


class ScorePartials:
    """Cross-wave sum of split-K score partials through ``s_part``.

    One f32x4 C fragment per lane per score tile, lane-major: every wave
    stores and loads the same layout, so the exchange needs no index remap.
    Every wave sums the partials in the same order, so the full S stays
    bit-identical across waves. The partial rows' previous readers retire
    before the k-loop's opening barrier, so only the RAW side is fenced.
    """

    def __init__(
        self, b: IRBuilder, s_part: Value, lanes: Lanes, *, num_warps: int, n_tiles: int
    ):
        self.b, self.s_part = b, s_part
        self.num_warps, self.n_tiles = num_warps, n_tiles
        self.part_row = b.mul(lanes.wave, b.const_i32(n_tiles))
        self.lane_part = b.mul(lanes.lane, b.const_i32(C_PER_LANE))

    def combine(self, s_acc: list[list[Value]]) -> list[list[Value]]:
        """Every wave's full score tiles: the sum of all waves' partials."""
        b = self.b
        k_tiles = len(s_acc[0])
        for mt, row_tiles in enumerate(s_acc):
            for kt, frag in enumerate(row_tiles):
                row = shift(b, mt * k_tiles + kt, self.part_row)
                b.smem_store_vN(self.s_part, [row, self.lane_part], frag, C_PER_LANE)
        b.sync_lds_only()
        out = []
        for mt in range(len(s_acc)):
            out.append([])
            for kt in range(k_tiles):
                total = None
                for w in range(self.num_warps):
                    part = b.smem_load_vN(
                        self.s_part,
                        b.const_i32(w * self.n_tiles + mt * k_tiles + kt),
                        self.lane_part,
                        dtype=F32,
                        n=C_PER_LANE,
                    )
                    total = part if total is None else b.fadd(total, part)
                out[mt].append(total)
        return out


class KvRegStager:
    """Register-staged key tile: buffer loads now, LDS stores next iteration.

    Each thread's byte offset inside a page is loop-invariant and computed
    once; the page base rides in the SGPR ``soffset``.
    """

    def __init__(
        self,
        b: IRBuilder,
        args: KernelArgs,
        kv_lds: Value,
        lanes: Lanes,
        *,
        threads: int,
        block_k: int,
        r_kv: int,
        d_rope: int,
        page_size: int,
        c_width: int,
        num_records: int,
        dtype: Type,
    ):
        self.b, self.kv_lds, self.tid, self.dtype = b, kv_lds, lanes.tid, dtype
        self.threads, self.r_kv, self.d_rope, self.page_size = (
            threads,
            r_kv,
            d_rope,
            page_size,
        )
        self.c_width = c_width
        self.c_chunks = _chunks("c_kv tile", block_k * r_kv, threads, c_width)
        self.c_per_row = r_kv // c_width
        self.kr_chunks = _chunks("k_rope tile", block_k * d_rope, threads, 4)
        self.kr_per_row = d_rope // 4
        self.c_rsrc, self.kr_rsrc = kv_buffer_rsrcs(b, args, num_records)
        self.c_voffs = []
        for j in range(self.c_chunks):
            m, r0 = chunk_coords(b, self.tid, j, threads, self.c_per_row, c_width)
            self.c_voffs.append(byte_offset(b, m, r_kv, r0))
        self.kr_voffs = []
        for j in range(self.kr_chunks):
            m, d0 = chunk_coords(b, self.tid, j, threads, self.kr_per_row, 4)
            self.kr_voffs.append(byte_offset(b, m, d_rope, d0))

    def loads(self, page: Value) -> list[Value]:
        """Issue one tile's global loads into registers. Touches no LDS.

        ``page`` is the tile's raw block-table entry from :meth:`PageTable.load`.
        """
        b = self.b
        c_soff, kr_soff = page_offsets(
            b, page, page_size=self.page_size, r_kv=self.r_kv, d_rope=self.d_rope
        )
        staged = [
            b.buffer_load_vN(self.c_rsrc, voff, c_soff, self.dtype, self.c_width)
            for voff in self.c_voffs
        ]
        staged += [
            b.buffer_load_vN(self.kr_rsrc, voff, kr_soff, self.dtype, 4)
            for voff in self.kr_voffs
        ]
        return staged

    def store(self, staged: Sequence[Value]) -> None:
        """Drain a staged tile into ``kv_lds``, ``[key][r]``.

        That is the natural layout for the score GEMM's B operand. The latent
        PV reads the same buffer transposed, one bf16 per key.
        """
        b = self.b
        for j in range(self.c_chunks):
            m, r0 = chunk_coords(
                b, self.tid, j, self.threads, self.c_per_row, self.c_width
            )
            b.smem_store_vN(self.kv_lds, [m, r0], staged[j], self.c_width)
        for j in range(self.kr_chunks):
            m, d0 = chunk_coords(b, self.tid, j, self.threads, self.kr_per_row, 4)
            b.smem_store_vN(
                self.kv_lds,
                [m, b.add(d0, b.const_i32(self.r_kv))],
                staged[self.c_chunks + j],
                4,
            )


def store_acc_t(
    b: IRBuilder,
    lds: Value,
    accs: Sequence[Sequence[Value]],
    lanes: Lanes,
    *,
    owner: int,
) -> None:
    """Wave ``owner`` writes its ``accᵀ`` back to ``lds`` as ``[query][r]`` bf16.

    ``accᵀ`` puts one query and four consecutive latent columns in each lane,
    so each fragment is a single 8 B store. Stores only, no barrier: safe in
    the wave-gated region.
    """
    with on_wave(b, lanes.wave, owner):
        _store_acc_t(b, lds, accs, lanes.lane_row, lanes.lane_kq)


def _store_acc_t(b, lds, accs, lane_row, lane_kq):
    for t in range(len(accs[0])):
        col = shift(b, t * MFMA_N, lane_kq)
        for mt, row_tiles in enumerate(accs):
            row = shift(b, mt * MFMA_M, lane_row)
            b.smem_store_vN(
                lds, [row, col], b.vec_trunc_f32_to_bf16(row_tiles[t]), C_PER_LANE
            )


# ---------------------------------------------------------------------------
# One-wave-per-head layout phases (``_build_mla_prefill_fwd_heads``)
# ---------------------------------------------------------------------------


def epilogue_lane_row(b: IRBuilder, lanes: Lanes, *, rederive: bool) -> Value:
    """``lane % 16`` for the epilogue.

    ``rederive`` recomputes it from the lane id (``v_mbcnt``), which the
    backend cannot merge with the prologue's thread-id form; holding the
    prologue value across the loop spills at 32 rows a wave.
    """
    return b.mod(b.lane_id(), b.const_i32(16)) if rederive else lanes.lane_row


def wave_head(
    b: IRBuilder, head_grp: Value, wave: Value, *, heads_per_wg: int, q_tiles: int
) -> tuple[Value, Value]:
    """``(head_local, head)``: wave ``w`` serves head ``w // q_tiles`` of the
    group, made wave-uniform so every per-head address is scalar math."""
    head_local = wave if q_tiles == 1 else b.div(wave, b.const_i32(q_tiles))
    head = b.readfirstlane(
        b.add(b.mul(head_grp, b.const_i32(heads_per_wg)), head_local)
    )
    return head_local, head


def wave_q_start(
    b: IRBuilder, tile: QTile, wave: Value, *, q_tiles: int, rows: int
) -> Value:
    """First query row of the wave's own ``rows``-row tile."""
    if q_tiles == 1:
        return tile.qb_start
    q_sub = b.readfirstlane(b.mod(wave, b.const_i32(q_tiles)))
    return b.add(tile.qb_start, b.mul(q_sub, b.const_i32(rows)))


def load_q_nope_rope(
    b: IRBuilder,
    q_ptr: Value,
    qrows: QRows,
    q_start: Value,
    lanes: Lanes,
    mt: int,
    *,
    d_nope: int,
    d_rope: int,
    dtype: Type,
) -> tuple[list[Value], list[Value]]:
    """``(Q_nope fragments, Q_rope fragments)`` of M tile ``mt``, B layout.

    Q_nope is read 16 B per lane and split into k-step halves (see
    :func:`absorb_query_t`); Q_rope four columns at ``lane_kq``.
    """
    hdqk = d_nope + d_rope
    token = qrows.clamped_token(row_of(b, q_start, lanes.lane_row, mt))
    q_base = qrows.offset(token, hdqk, lanes.lane_kq)
    q_base8 = qrows.offset(token, hdqk, lambda: b.add(lanes.lane_kq, lanes.lane_kq))
    nope = []
    for pp in range(d_nope // MFMA_K // 2):
        v8 = b.global_load_vN(q_ptr, shift(b, pp * 2 * MFMA_K, q_base8), dtype, 8)
        nope += [half4(b, v8, h, dtype) for h in range(2)]
    rope = load_q_frags(b, q_ptr, q_base, n=d_rope // MFMA_K, dtype=dtype, col0=d_nope)
    return nope, rope


def absorb_query_t(
    b: IRBuilder,
    w_uk: Value,
    q_frags: list[list[Value]],
    qa_frags: list[list[Value]],
    lanes: Lanes,
    head: Value,
    *,
    r_kv: int,
    d_nope: int,
    w_cols: int,
    prefetch: int,
    dtype: Type,
) -> None:
    """``Qaᵀ[r, q] = sum_n W_UK[head, r, n] * Q_nope[q, n]`` into ``qa_frags[t][mt]``.

    The A operand (``m = r = lane % 16``) is a plain 16 B read of a W_UK row,
    split into k-step halves with the same permutation of ``d_nope`` as
    ``Q_nope``. The C fragment is the score GEMM's B layout for k-step ``t``,
    so it is used as is. The W_UK loads run ``prefetch`` r-tiles ahead.
    """
    nope_k = d_nope // MFMA_K
    m_tiles = len(q_frags)
    w_lane = b.add(
        b.add(
            w_head_base(b, head, r_kv, w_cols),
            b.mul(lanes.lane_row, b.const_i32(w_cols)),
        ),
        b.add(lanes.lane_kq, lanes.lane_kq),
    )

    def _loads(t):
        w_row = shift(b, t * MFMA_N * w_cols, w_lane)
        return [
            b.global_load_vN(w_uk, shift(b, pp * 2 * MFMA_K, w_row), dtype, 8)
            for pp in range(nope_k // 2)
        ]

    n_tiles = r_kv // MFMA_N
    take = read_ahead(n_tiles, prefetch, _loads)
    for t in range(n_tiles):
        acc = zero_tiles(b, m_tiles)
        a_vs = []
        for v8 in take(t):
            a_vs += [half4(b, v8, h, dtype) for h in range(2)]
        for kk in range(nope_k):
            for mt in range(m_tiles):
                acc[mt] = mfma_16x16x16(b, dtype, a_vs[kk], q_frags[mt][kk], acc[mt])
        for mt in range(m_tiles):
            qa_frags[t][mt] = b.vec_trunc_f32_to_bf16(acc[mt])


def park_q_rope(
    b: IRBuilder,
    qr_lds: Value,
    rope_frags: list[list[Value]],
    lanes: Lanes,
    *,
    rows: int,
) -> Value:
    """Write the wave's Q_rope fragments to its rows of ``qr_lds``; returns
    the lane's row there."""
    qr_row = b.add(b.mul(lanes.wave, b.const_i32(rows)), lanes.lane_row)
    for j in range(len(rope_frags[0])):
        for mt, frags in enumerate(rope_frags):
            b.smem_store_vN(
                qr_lds,
                [shift(b, mt * MFMA_M, qr_row), shift(b, j * MFMA_K, lanes.lane_kq)],
                frags[j],
                4,
            )
    return qr_row


def read_q_rope(
    b: IRBuilder,
    qr_lds: Value,
    qr_row: Value,
    lane_kq: Value,
    j: int,
    m_tiles: int,
    dtype: Type,
) -> list[Value]:
    """Q_rope k-step ``j`` of every M tile, re-read from ``qr_lds``."""
    return [
        b.smem_load_vN(
            qr_lds,
            shift(b, mt * MFMA_M, qr_row),
            shift(b, j * MFMA_K, lane_kq),
            dtype=dtype,
            n=4,
        )
        for mt in range(m_tiles)
    ]


def read_kv_rows(
    b: IRBuilder,
    kv_lds: Value,
    lane_row: Value,
    col: Value,
    *,
    k_tiles: int,
    dtype: Type,
) -> list[Value]:
    """Score A fragments (``m = key``) at column ``col`` of each key tile."""
    return [
        b.smem_load_vN(kv_lds, shift(b, kt * MFMA_N, lane_row), col, dtype=dtype, n=4)
        for kt in range(k_tiles)
    ]


def read_k_rope_swizzled(
    b: IRBuilder,
    kr_lds: Value,
    lanes: Lanes,
    j: int,
    *,
    k_tiles: int,
    d_rope: int,
    dtype: Type,
) -> list[Value]:
    """K_rope k-step ``j`` from the unpadded DMA'd RoPE buffer.

    Unit ``4 j + g`` of row ``a`` sits at unit ``(4 j + g) ^ (a & 15)``: the
    DMA XOR-swizzles 8 B units by row so these reads stay conflict-free.
    """
    out = []
    for kt in range(k_tiles):
        row = shift(b, kt * MFMA_N, lanes.lane_row)
        unit = b.xor(
            shift(b, 4 * j, b.div(lanes.lane_kq, b.const_i32(4))),
            b.land(row, b.const_i32((d_rope * 2) // 8 - 1)),
        )
        idx = b.add(b.mul(row, b.const_i32(d_rope)), b.mul(unit, b.const_i32(4)))
        out.append(b.smem_load_vN(kr_lds, idx, dtype=dtype, n=4))
    return out


def read_pv_group(
    b: IRBuilder,
    kv_lds: Value,
    lane_kq: Value,
    lane_row4: Value,
    kt: int,
    w: int,
    *,
    dtype: Type,
) -> list[Value]:
    """The four 8 B key-row reads of PV group ``(kt, w)``.

    Row ``m`` of latent tile ``4 w + e`` is ``r = 64 w + 4 m + e``, so one read
    of a key row hands a lane its element of four tiles.
    """
    a_col = shift(b, kt * MFMA_N, lane_kq)
    return [
        b.smem_load_vN(
            kv_lds, shift(b, i, a_col), shift(b, 64 * w, lane_row4), dtype=dtype, n=4
        )
        for i in range(4)
    ]


def tile_pairs(b: IRBuilder, n_full: Value, n_k_tiles: Value) -> tuple[Value, Value]:
    """``(n_full, n_pad)`` for a loop unrolled by two key tiles.

    The unmasked run is rounded down to even; the masked tail is rounded up to
    even, and its padding tile (index ``n_k_tiles``, past every key) masks to
    nothing.
    """
    n_full = b.land(n_full, b.const_i32(-2))
    n_pad = b.add(
        n_full,
        b.land(b.add(b.sub(n_k_tiles, n_full), b.const_i32(1)), b.const_i32(-2)),
    )
    return n_full, n_pad


class KvSharedStager:
    """Register-staged key tile with one per-thread offset for every chunk.

    Chunk ``j`` of a thread is chunk 0 shifted by a whole number of rows
    (``threads`` is a multiple of the chunks per row), so every chunk shares
    one per-thread offset and the shift is a constant: in the SGPR ``soffset``
    for the loads, in the immediate for the LDS stores. That keeps the
    per-chunk offsets out of the register budget.
    """

    def __init__(
        self,
        b: IRBuilder,
        args: KernelArgs,
        kv_lds: Value,
        lanes: Lanes,
        *,
        threads: int,
        block_k: int,
        r_kv: int,
        d_rope: int,
        page_size: int,
        c_width: int,
        kr_width: int,
        num_records: int,
        dtype: Type,
    ):
        self.b, self.kv_lds, self.dtype = b, kv_lds, dtype
        self.r_kv, self.d_rope, self.page_size = r_kv, d_rope, page_size
        self.c_width, self.kr_width = c_width, kr_width
        self.c_chunks = _chunks("c_kv tile", block_k * r_kv, threads, c_width)
        c_per_row = r_kv // c_width
        self.kr_chunks = _chunks("k_rope tile", block_k * d_rope, threads, kr_width)
        kr_per_row = d_rope // kr_width
        assert threads % c_per_row == 0 and threads % kr_per_row == 0
        self.c_rows_per_chunk = threads // c_per_row
        self.kr_rows_per_chunk = threads // kr_per_row
        self.c_rsrc, self.kr_rsrc = kv_buffer_rsrcs(b, args, num_records)
        self.c_m0, self.c_r0 = split_coords(b, lanes.tid, c_per_row, c_width)
        self.c_voff = byte_offset(b, self.c_m0, r_kv, self.c_r0)
        self.kr_m0, self.kr_d0 = split_coords(b, lanes.tid, kr_per_row, kr_width)
        self.kr_voff = byte_offset(b, self.kr_m0, d_rope, self.kr_d0)

    def loads(self, page: Value) -> list[Value]:
        """Issue one tile's buffer loads into registers. Touches no LDS."""
        b = self.b
        c_soff, kr_soff = page_offsets(
            b, page, page_size=self.page_size, r_kv=self.r_kv, d_rope=self.d_rope
        )
        staged = [
            b.buffer_load_vN(
                self.c_rsrc,
                self.c_voff,
                shift(b, j * self.c_rows_per_chunk * self.r_kv * 2, c_soff),
                self.dtype,
                self.c_width,
            )
            for j in range(self.c_chunks)
        ]
        staged += [
            b.buffer_load_vN(
                self.kr_rsrc,
                self.kr_voff,
                shift(b, j * self.kr_rows_per_chunk * self.d_rope * 2, kr_soff),
                self.dtype,
                self.kr_width,
            )
            for j in range(self.kr_chunks)
        ]
        return staged

    def store(self, staged: Sequence[Value]) -> None:
        """Drain a staged tile into ``kv_lds``, ``[key][r]``."""
        b = self.b
        for j in range(self.c_chunks):
            m = shift(b, j * self.c_rows_per_chunk, self.c_m0)
            b.smem_store_vN(self.kv_lds, [m, self.c_r0], staged[j], self.c_width)
        kr_col = b.add(self.kr_d0, b.const_i32(self.r_kv))
        for j in range(self.kr_chunks):
            m = shift(b, j * self.kr_rows_per_chunk, self.kr_m0)
            b.smem_store_vN(
                self.kv_lds, [m, kr_col], staged[self.c_chunks + j], self.kr_width
            )


class KvTileDma:
    """The key tile DMA'd straight into LDS (``buffer_load ... lds``).

    gfx942 ``buffer_load ... lds`` writes one dword per lane to
    ``M0 + 4 * lane``: 256 contiguous bytes per wave instruction. Latent row
    ``r`` is ``R_KV * 2 / 256`` such chunks at ``r * row_bytes + 256 j`` (the
    row pad lies outside them). RoPE rows are 128 B, so one chunk covers two
    rows of the unpadded RoPE buffer; its 8 B units are XOR-swizzled by the
    row (unit ``u`` of row ``a`` sits at ``u ^ (a & 15)``) through the
    per-lane source address, so the score's RoPE reads stay conflict-free.
    """

    def __init__(
        self,
        b: IRBuilder,
        stager: KvSharedStager,
        lanes: Lanes,
        kv_slots: Sequence[Value],
        kr_slots: Sequence[Value | None],
        *,
        num_warps: int,
        block_k: int,
        r_kv: int,
        d_rope: int,
        page_size: int,
        row_pad: int,
    ):
        self.b, self.lanes = b, lanes
        self.c_rsrc, self.kr_rsrc = stager.c_rsrc, stager.kr_rsrc
        self.r_kv, self.d_rope, self.page_size = r_kv, d_rope, page_size
        self.rows_w = block_k // num_warps
        if (r_kv * 2) % 256 or d_rope * 2 != 128:
            raise ValueError(
                f"KvTileDma: r_kv={r_kv} (expected a multiple of 128) and "
                f"d_rope={d_rope} (expected 64) must fit 256 B DMA chunks"
            )
        self.c_chunks_row = (r_kv * 2) // 256
        self.kr_rows_chunk = 256 // (d_rope * 2)
        self.row_bytes = (r_kv + row_pad) * 2
        self.kr_units = (d_rope * 2) // 8
        self.wave_u = b.readfirstlane(lanes.wave)
        self.kvc_addr = [b.smem_addr_of(x) for x in kv_slots]
        self.kr_addr = [b.smem_addr_of(x) for x in kr_slots]

    def _kr_voff(self, cp: int) -> Value:
        # Recomputed at issue time rather than held: with the DMA spread
        # through the score loop, loop-invariant offsets held in VGPRs spill.
        b, lane = self.b, self.lanes.lane
        row = b.add(
            b.mul(self.lanes.wave, b.const_i32(self.rows_w)),
            b.add(b.const_i32(cp * self.kr_rows_chunk), b.div(lane, b.const_i32(32))),
        )
        d = b.mod(lane, b.const_i32(32))
        u = b.xor(b.div(d, b.const_i32(2)), b.land(row, b.const_i32(self.kr_units - 1)))
        return b.add(
            b.mul(row, b.const_i32(self.d_rope * 2)),
            b.add(
                b.mul(u, b.const_i32(8)),
                b.mul(b.mod(d, b.const_i32(2)), b.const_i32(4)),
            ),
        )

    def ops(self, page: Value, slot: int) -> list[Callable[[], None]]:
        """The tile's DMA as one thunk per instruction, so the caller can
        spread them between MFMAs instead of issuing one burst. The page
        offsets are emitted now; each thunk emits its own instruction."""
        b, lane = self.b, self.lanes.lane
        c_base, r_base = page_offsets(
            b, page, page_size=self.page_size, r_kv=self.r_kv, d_rope=self.d_rope
        )
        row0 = b.mul(self.wave_u, b.const_i32(self.rows_w))
        ops = []
        for jr in range(self.rows_w):
            for j in range(self.c_chunks_row):

                def _c(jr=jr, j=j):
                    row = shift(b, jr, row0)
                    b.async_buffer_load_lds_addr(
                        self.c_rsrc,
                        b.smem_ptr_add(
                            self.kvc_addr[slot],
                            b.zext(
                                shift(
                                    b, j * 256, b.mul(row, b.const_i32(self.row_bytes))
                                ),
                                I64,
                            ),
                        ),
                        b.mul(lane, b.const_i32(4)),
                        b.add(
                            c_base,
                            shift(b, j * 256, b.mul(row, b.const_i32(self.r_kv * 2))),
                        ),
                        1,
                    )

                ops.append(_c)
        for cp in range(self.rows_w // self.kr_rows_chunk):

            def _r(cp=cp):
                voff = self._kr_voff(cp)
                b.async_buffer_load_lds_addr(
                    self.kr_rsrc,
                    b.smem_ptr_add(
                        self.kr_addr[slot],
                        b.zext(
                            b.mul(
                                shift(b, cp * self.kr_rows_chunk, row0),
                                b.const_i32(self.d_rope * 2),
                            ),
                            I64,
                        ),
                    ),
                    voff,
                    r_base,
                    1,
                )

            ops.append(_r)
        return ops


def store_latent_out(
    b: IRBuilder,
    out: Value,
    accs: Sequence[Sequence[Value]],
    inv: Sequence[Value],
    qrows: QRows,
    lanes: Lanes,
    *,
    q_start: Value,
    lane_row: Value,
    r_kv: int,
) -> None:
    """Normalized latent output ``[token, head, r_kv]`` straight from ``accᵀ``.

    With the PV's row permutation, register ``reg`` of tile ``4 w + e`` holds
    ``r = 64 w + 16 g + 4 reg + e`` in lane group ``g``, so the four tiles of
    one ``w`` hand a lane four consecutive ``r``: one 8 B store per
    ``(w, reg)``.
    """
    r_lane = b.mul(lanes.lane_kq, b.const_i32(4))
    for mt in range(len(inv)):
        q_local = row_of(b, q_start, lane_row, mt)
        with b.scf_if(b.cmp_lt(q_local, qrows.tile.s_q)):
            token = b.add(qrows.tile.cu_q_start, q_local)
            row = qrows.offset(token, r_kv, r_lane)
            for w in range(r_kv // MFMA_N // 4):
                for reg in range(C_PER_LANE):
                    # Lazy: an AGPR-resident tile is read out on first use.
                    vals = scaled_pack(
                        b, ((accs[mt][4 * w + e], reg) for e in range(4)), inv[mt]
                    )
                    b.global_store_vN(
                        out,
                        shift(b, 64 * w + 4 * reg, row),
                        b.vec_trunc_f32_to_bf16(vals),
                        4,
                        align=8,
                    )


def project_wuv_heads(
    b: IRBuilder,
    w_uk: Value,
    wt_lds: Value,
    accs: Sequence[Sequence[Value]],
    lanes: Lanes,
    head_grp: Value,
    head_local: Value,
    *,
    threads: int,
    heads_per_wg: int,
    vq: int,
    m_tiles: int,
    r_kv: int,
    d_nope: int,
    d_v: int,
    w_cols: int,
    dtype: Type,
) -> list[list[Value]]:
    """``outᵀ = W_UVᵀ · accᵀ`` for every head of the group; returns
    ``out_acc[mt][vt]``.

    With the PV's row permutation, tile ``4 w + e`` holds ``r = 64 w + 16 g +
    4 reg + e`` in lane group ``g``, so a lane's ``W_UVᵀ`` fragment for that
    k-step is the four ``r`` of one 16-row block that share ``r % 4 == e``.
    Each round stages a 64-row latent block ``w`` and a ``vq``-wide ``v``
    window for every head (row = head-in-group * vq + v), with block column
    ``p = 16 g + 4 e + i`` holding ``r = 64 w + 16 g + 4 i + e``, so the
    fragment is one contiguous 8 B read. A staging thread loads those four
    ``W_UV`` rows (stride 4) for eight ``v`` and transposes the 4 x 8 block in
    registers into eight 8 B stores.
    """
    tid, lane_row, lane_kq = lanes.tid, lanes.lane_row, lanes.lane_kq
    n_r_tiles = r_kv // MFMA_N
    out_acc = zero_tiles(b, m_tiles, d_v // MFMA_N)
    acc_bf = [[None] * n_r_tiles for _ in range(m_tiles)]
    grp_base = w_head_base(b, head_grp, heads_per_wg * r_kv, w_cols)
    per_head = 16 * (vq // 8)
    stagers = []
    for j in range(_chunks("W_UV epilogue block", heads_per_wg * 64 * vq, threads, 32)):
        c = b.add(tid, b.const_i32(j * threads))
        hh = b.div(c, b.const_i32(per_head))
        rem = b.mod(c, b.const_i32(per_head))
        g = b.div(b.mod(rem, b.const_i32(16)), b.const_i32(4))
        e = b.mod(rem, b.const_i32(4))
        n0 = b.mul(b.div(rem, b.const_i32(16)), b.const_i32(8))
        # W_UK[head_grp * HPW + hh, 16 g + 4 i + e (+ 64 w), D_NOPE + n0 (+ vq)]
        src = b.add(
            grp_base,
            b.add(
                b.mul(
                    b.add(
                        b.mul(hh, b.const_i32(r_kv)),
                        b.add(b.mul(g, b.const_i32(16)), e),
                    ),
                    b.const_i32(w_cols),
                ),
                shift(b, d_nope, n0),
            ),
        )
        row0 = b.add(b.mul(hh, b.const_i32(vq)), n0)
        p_col = b.add(b.mul(g, b.const_i32(16)), b.mul(e, b.const_i32(4)))
        stagers.append((src, row0, p_col))
    a_rows = [
        b.add(b.mul(head_local, b.const_i32(vq)), shift(b, vl * MFMA_N, lane_row))
        for vl in range(vq // MFMA_N)
    ]
    a_cols = [shift(b, 4 * e, b.mul(lane_kq, b.const_i32(4))) for e in range(4)]
    for w in range(n_r_tiles // 4):
        for vqi in range(d_v // vq):
            b.sync()
            for src, row0, p_col in stagers:
                rows8 = [
                    b.global_load_vN(
                        w_uk,
                        shift(b, (64 * w + 4 * i) * w_cols + vqi * vq, src),
                        dtype,
                        8,
                    )
                    for i in range(4)
                ]
                for e8 in range(8):
                    b.smem_store_vN(
                        wt_lds,
                        [shift(b, e8, row0), p_col],
                        b.vec_pack([b.vec_extract(r8, e8) for r8 in rows8], dtype),
                        4,
                    )
            b.sync()
            for e in range(4):
                t = 4 * w + e
                for mt in range(m_tiles):
                    if acc_bf[mt][t] is None:
                        acc_bf[mt][t] = b.vec_trunc_f32_to_bf16(accs[mt][t])
                for vl in range(vq // MFMA_N):
                    a_v = b.smem_load_vN(
                        wt_lds, a_rows[vl], a_cols[e], dtype=dtype, n=4
                    )
                    vt = vqi * (vq // MFMA_N) + vl
                    for mt in range(m_tiles):
                        out_acc[mt][vt] = mfma_16x16x16(
                            b, dtype, a_v, acc_bf[mt][t], out_acc[mt][vt]
                        )
    return out_acc
