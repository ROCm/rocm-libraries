# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged address generator of the unified attention backward main kernel.

Family glue for the backward kernels (Python-only, no C++ twin). Like
:mod:`._attention_bwd_addr`, everything here emits integer / pointer / load /
store IR through the builder it is handed.

A CTA of the unified main kernel owns the logical kv tile ``[k0, k0 + kN0)``
of one sequence. :class:`PagedKvTile` maps that tile onto the physical pages
of the sequence's block table ``block_table[s][j]`` (int32, row stride
``bt_stride``):

* page class ``"multi"`` (``sub_page = None``, runtime ``page``, ``page % kN0
  == 0``): one page holds the whole tile: ``j = k0 / page``, first in-page
  slot ``k0 % page``;
* page class ``"sub"`` (compile-time ``sub_page = p``, ``kN0 % p == 0``):
  ``kN0 / p`` pages, page ``i`` holds tile rows ``[i * p, (i + 1) * p)``.

Every block-table entry the tile uses is read once (CTA-uniform) inside a
0/1-trip loop whose trip count is ``j < ceil(len_kv / page)``: an entry past
the sequence's live blocks is never read (its page id stays 0 and no access
uses it, because every access row is either clamped into the sequence or
predicated on ``k < ceil(len_kv / page) * page``).

Physical rebase: ``page_id * blk_stride`` (and ``head * h_stride``) is formed
in i64 and folded into the pointer with ``global_ptr_add``; the in-page index
``slot * tok_stride + col`` is i32 (the host plan bounds ``page *
tok_stride``). The fp32 page workspace ``[h][page_rows][D]`` (atomic page
mode) is addressed through :class:`~._attention_bwd_addr.WorkspaceAddr` at
physical slot ``page_id * page + in-page slot`` (i64), so a slot at or past
``page_rows`` is never written.

Block-id guard (:class:`PagedKv` given ``page_rows``, the cache's physical
slots ``num_blocks * page``): a block-table entry is device data, so an entry
outside ``[0, num_blocks)`` (negative, or past the cache) must never reach an
address. Each entry the tile reads is checked once (``0 <= id <
page_rows / page``, two compares, CTA-uniform); an out-of-range entry is
replaced by block 0 for every address (so no address leaves the cache), its
K / V loads read as exact zeros (no dQ contribution: ``dS K = 0`` and ``dP =
dO V^T = 0``), its direct-mode dK / dV stores are predicated off (the store
rule) and its page-workspace slot is moved to ``page_rows`` (past the
workspace, so the atomic adds are dropped by the workspace row bound). The
host check (``attention_bwd_unified_run.check_attn_bwd_unified_inputs``)
rejects such a table before launch; the guard is the defined device
behaviour when one reaches the kernel anyway. Without ``page_rows`` no guard
op is emitted.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

from rocke.core.ir import I64, IRBuilder, Type, Value

from ._attention_bwd_addr import (
    SeqInfo,
    WorkspaceAddr,
    WsSlab,
    _elem_bytes,
    _scaled32,
    _scaled64,
    as_i32,
    as_i64,
)

__all__ = [
    "PagedCacheRows",
    "PagedKv",
    "PagedKvTile",
    "PagedSlab",
    "live_blocks",
]

IntOrValue = int | Value


def live_blocks(b: IRBuilder, length: Value, page: IntOrValue) -> Value:
    """``ceil(length / page)`` (i32): the sequence's live logical blocks."""
    p = as_i32(b, page)
    return b.div(b.add(length, b.sub(p, b.const_i32(1))), p)


def _guarded_entry(
    b: IRBuilder, row_ptr: Value, j: Value, n_blocks: Value, n: int
) -> Value:
    """``row_ptr[j]`` when ``j < n_blocks``, else 0 without a read.

    The read sits in a loop of trip ``j < n_blocks`` (0 or 1), so the entry of
    a block past the sequence's live blocks is never loaded. ``n`` numbers the
    loop (its names must be unique in the kernel; :class:`PagedKv` counts the
    reads of its kernel).
    """
    zero = b.const_i32(0)
    trip = b.select(b.cmp_lt(j, n_blocks), b.const_i32(1), zero)
    loop = b.scf_for_iter(
        zero, trip, b.const_i32(1), [(f"page_id{n}", zero)], iv_name=f"bt_i{n}"
    )
    with loop as (_iv, (_cur,)):
        b.scf_yield(b.global_load_i32(row_ptr, j))
    return loop.results[0]


class PagedKvTile:
    """Physical placement of the logical kv tile ``[k0, k0 + kN0)``.

    ``n_blocks = ceil(len_kv / page)``; ``cover = n_blocks * page`` is the
    number of slots of the sequence's live pages (logical rows below it have
    a physical slot). ``page_ids`` holds one i32 page id per piece (one piece
    in the ``"multi"`` class).

    ``num_blocks`` (i32, optional) enables the block-id guard of the module
    docstring: ``page_ids`` then hold the checked ids, ``-1`` for an entry
    outside ``[0, num_blocks)``, so the check costs no value of its own across
    the q loops (the ids are live anyway); every use derives ``ok = id >= 0``
    and the address id ``max(id, 0)`` (:meth:`checked`). Without it
    ``guarded`` is False and no guard op is emitted. ``reads`` numbers the
    block-table reads (unique loop names per kernel; a fresh count when the
    tile is built on its own).
    """

    def __init__(
        self,
        b: IRBuilder,
        *,
        bt_ptr: Value,
        bt_stride: IntOrValue,
        seq_idx: Value,
        seq: SeqInfo,
        k0: Value,
        k_n0: int,
        page: IntOrValue,
        sub_page: int | None,
        num_blocks: Value | None = None,
        reads: Iterator[int] | None = None,
    ) -> None:
        if sub_page is not None:
            if sub_page % 16 or k_n0 % sub_page or sub_page >= k_n0:
                raise ValueError(
                    f"sub_page={sub_page} must be a multiple of 16 dividing "
                    f"block_n={k_n0} and below it"
                )
            if not isinstance(page, int) or page != sub_page:
                raise ValueError("the sub page class uses the compile-time page")
        self.b = b
        self.k0 = k0
        self.k_n0 = k_n0
        self.sub_page = sub_page
        self.page = as_i32(b, page)
        self.seq = seq
        self.n_blocks = live_blocks(b, seq.length, page)
        self.cover = b.mul(self.n_blocks, self.page)
        off = _scaled64(b, as_i64(b, seq_idx), bt_stride)
        row = b.global_ptr_add(bt_ptr, b.mul(off, b.const_i64(4)))
        reads = itertools.count() if reads is None else reads
        if sub_page is None:
            j = b.div(k0, self.page)
            self.in0 = b.sub(k0, b.mul(j, self.page))
            raw = [_guarded_entry(b, row, j, self.n_blocks, next(reads))]
        else:
            j0 = b.div(k0, b.const_i32(sub_page))
            self.in0 = None
            raw = [
                _guarded_entry(
                    b, row, j0 if i == 0 else b.add(j0, b.const_i32(i)),
                    self.n_blocks, next(reads),
                )
                for i in range(k_n0 // sub_page)
            ]  # fmt: skip
        self.guarded = num_blocks is not None
        self.page_ids = raw
        if self.guarded:
            zero, bad = b.const_i32(0), b.const_i32(-1)
            nb = as_i32(b, num_blocks)
            self.page_ids = [
                b.select(b.land(b.cmp_ge(pid, zero), b.cmp_lt(pid, nb)), pid, bad)
                for pid in raw
            ]

    @property
    def pieces(self) -> int:
        return len(self.page_ids)

    def locate(self, lr: Value) -> tuple[Value, Value]:
        """``(page id, in-page slot)`` (i32) of tile row ``lr`` in ``[0, kN0)``
        (under the block-id guard the checked id: ``-1`` for a bad entry; see
        :meth:`checked`)."""
        b = self.b
        if self.sub_page is None:
            return self.page_ids[0], b.add(self.in0, lr)
        p = self.sub_page
        piece = b.div(lr, b.const_i32(p))
        slot = b.sub(lr, b.mul(piece, b.const_i32(p)))
        pid = self.page_ids[-1]
        for i in range(self.pieces - 2, -1, -1):
            pid = b.select(b.cmp_eq(piece, b.const_i32(i)), self.page_ids[i], pid)
        return pid, slot

    def checked(self, pid: Value) -> tuple[Value, Value | None]:
        """``(address id, ok)`` of a :meth:`locate` id: under the guard
        ``(max(id, 0), id >= 0)`` (a bad entry addresses block 0 and is not
        ok), else ``(id, None)`` without an op."""
        if not self.guarded:
            return pid, None
        b = self.b
        zero = b.const_i32(0)
        return b.smax(pid, zero), b.cmp_ge(pid, zero)

    def store_rule(self, k: Value) -> tuple[Value, Value]:
        """``(store, row_live)`` of logical key row ``k`` (direct scatter).

        A row is stored iff it has a physical slot in a live page (``k <
        cover``) whose block id passed the guard (when enabled); its value is
        the gradient iff ``k < len_kv``, else an exact zero (slots of the last
        live page past the sequence end).
        """
        b = self.b
        store = b.cmp_lt(k, self.cover)
        if self.guarded:
            pid, _slot = self.locate(b.sub(k, self.k0))
            store = b.land(store, b.cmp_ge(pid, b.const_i32(0)))
        return store, b.cmp_lt(k, self.seq.length)

    def workspace_row(
        self, ws_ptr: Value, *, page_rows: IntOrValue, width: int, head: IntOrValue
    ):
        """``row_slab(lr) -> (WsSlab, local row)`` of the fp32 page workspace.

        Head-major ``[h][page_rows][width]`` indexed by physical slot ``page_id
        * page + in-page slot``. The ``"multi"`` class rebases once per tile
        (rows contiguous inside one page); the ``"sub"`` class rebases per row.
        Under the block-id guard a slot of an out-of-range block is moved to
        ``page_rows`` (past the workspace: no row of it is written).
        """
        b = self.b
        page64 = as_i64(b, self.page)

        def slab_at(pid: Value, slot: Value) -> WsSlab:
            pid, ok = self.checked(pid)
            phys = b.add(b.mul(b.zext(pid, I64), page64), b.zext(slot, I64))
            if ok is not None:
                phys = b.select(ok, phys, as_i64(b, page_rows))
            return WorkspaceAddr(
                b, ws_ptr, ws_rows=page_rows, width=width, row0=phys
            ).slab(head)

        if self.sub_page is None:
            tile_slab = slab_at(self.page_ids[0], self.in0)

            def row_slab(lr):
                return tile_slab, lr

            return row_slab

        zero = b.const_i32(0)

        def row_slab(lr):
            pid, slot = self.locate(lr)
            return slab_at(pid, slot), zero

        return row_slab


class PagedSlab:
    """Logical key rows of one head of a paged cache, for one kv tile.

    Rows are logical sequence rows ``k0 + lr``. With ``last`` (loads) a row
    is clamped to ``min(row, last)`` first (MMA operands always come from a
    row of the sequence); without it (stores) the caller predicates the
    access. ``stage_vec`` is the global access width (8: 16-byte vectors).
    """

    def __init__(
        self,
        b: IRBuilder,
        tile: PagedKvTile,
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[IntOrValue, IntOrValue, IntOrValue],
        head: IntOrValue,
        last: Value | None,
        stage_vec: int,
    ) -> None:
        if stage_vec not in (8, 1):
            raise ValueError(f"stage_vec must be 8 or 1 (got {stage_vec})")
        self.b = b
        self.tile = tile
        self.dtype = dtype
        self.last = last
        self.stage_vec = stage_vec
        self.esize = _elem_bytes(dtype)
        s_blk, s_tok, s_h = strides
        self.s_blk, self.s_tok = s_blk, s_tok
        head_off = None
        if not (isinstance(head, int) and head == 0):
            head_off = _scaled64(b, as_i64(b, head), s_h)
        # Head rebase folded into the base once; the page rebase per piece.
        self.base = ptr
        if head_off is not None:
            self.base = b.global_ptr_add(ptr, b.mul(head_off, b.const_i64(self.esize)))
        self.tile_ok = None
        if tile.sub_page is None:
            pid, self.tile_ok = tile.checked(tile.page_ids[0])
            self.tile_ptr = self._page_ptr(pid)
        else:
            self.tile_ptr = None

    def _page_ptr(self, pid: Value) -> Value:
        b = self.b
        off = _scaled64(b, b.zext(pid, I64), self.s_blk)
        return b.global_ptr_add(self.base, b.mul(off, b.const_i64(self.esize)))

    def _address(
        self, row: Value, col: IntOrValue
    ) -> tuple[Value, Value, Value | None]:
        """``(page pointer, in-page index, page check or None)`` of a row."""
        b = self.b
        r = row if self.last is None else b.smin(row, self.last)
        lr = b.sub(r, self.tile.k0)
        pid, slot = self.tile.locate(lr)
        if self.tile_ptr is not None:
            ptr, ok = self.tile_ptr, self.tile_ok
        else:
            pid, ok = self.tile.checked(pid)
            ptr = self._page_ptr(pid)
        idx = _scaled32(b, slot, self.s_tok)
        if not (isinstance(col, int) and col == 0):
            idx = b.add(idx, as_i32(b, col))
        return ptr, idx, ok

    def load_vec(self, row: Value, col0: IntOrValue, n: int = 8) -> Value:
        """``n`` consecutive elements as ``<n x dtype>`` (``stage_vec`` width);
        exact zeros for a row of a block that failed the block-id guard."""
        b = self.b
        ptr, idx, ok = self._address(row, col0)
        if self.stage_vec == 8 and n % 8 == 0 and self.esize == 2:
            parts = [
                b.global_load_vN(
                    ptr, idx if i == 0 else b.add(idx, b.const_i32(8 * i)),
                    self.dtype, 8, align=16,
                )
                for i in range(n // 8)
            ]  # fmt: skip
            out = parts[0]
            for p in parts[1:]:
                out = b.vec_concat(out, p)
        else:
            elems = [
                b.global_load(
                    ptr, idx if i == 0 else b.add(idx, b.const_i32(i)), self.dtype,
                    align=self.esize,
                )
                for i in range(n)
            ]  # fmt: skip
            out = b.vec_pack(elems, self.dtype)
        if ok is not None:
            out = b.select(ok, out, b.zero_vec(self.dtype, n))
        return out

    def store(self, row: Value, col: IntOrValue, value: Value) -> None:
        """Store at ``(row, col)``; the caller predicates on the store rule."""
        ptr, idx, _ok = self._address(row, col)
        self.b.global_store(ptr, idx, value, align=self.esize)

    def store_vec(self, row: Value, col0: IntOrValue, value: Value) -> None:
        """Store ``<n x dtype>`` at ``(row, col0 .. col0 + n)``.

        With ``stage_vec = 8`` an ``n`` of 8, 4 or 2 16-bit elements at a
        column that is a multiple of ``n`` is one vector store; otherwise the
        elements are stored one by one.
        """
        b = self.b
        n = value.type.count
        ptr, idx, _ok = self._address(row, col0)
        if self.stage_vec == 8 and n in (8, 4, 2) and self.esize == 2:
            b.global_store_vN(ptr, idx, value, n, align=2 * n)
            return
        for i in range(n):
            b.global_store(
                ptr, idx if i == 0 else b.add(idx, b.const_i32(i)),
                b.vec_extract(value, i), align=self.esize,
            )  # fmt: skip


class PagedCacheRows:
    """A paged ``[num_blocks, page, h, D]`` cache viewed by logical kv tile.

    ``slab(h, k0)`` returns the :class:`PagedSlab` of head ``h`` of the tile
    starting at ``k0`` (the shared :class:`PagedKvTile` of ``paged``).
    """

    def __init__(
        self,
        paged: "PagedKv",
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[IntOrValue, IntOrValue, IntOrValue],
        stage_vec: int,
    ) -> None:
        self.paged = paged
        self.ptr = ptr
        self.dtype = dtype
        self.strides = strides
        self.stage_vec = stage_vec

    def slab(self, h: IntOrValue, k0: Value, *, clamp: bool = True) -> PagedSlab:
        tile = self.paged.tile(k0)
        return PagedSlab(
            self.paged.b, tile, self.ptr, dtype=self.dtype, strides=self.strides,
            head=h, last=self.paged.seq.last if clamp else None,
            stage_vec=self.stage_vec,
        )  # fmt: skip


class PagedKv:
    """The paged kv side of one CTA: block table, page geometry, tiles.

    :meth:`tile` builds the :class:`PagedKvTile` of a tile start once (keyed
    by the ``k0`` value), so K, V and the dK / dV epilogue share its
    block-table reads. One ``PagedKv`` per kernel: it numbers the kernel's
    block-table reads (unique loop names).

    ``page_rows`` (i32 kernarg, ``num_blocks * page``) enables the block-id
    guard of the module docstring: ``num_blocks = page_rows / page`` is formed
    once per CTA and every tile checks its entries against it. Without it
    (``None``) the kernel trusts the block table and no guard op is emitted.
    """

    def __init__(
        self,
        b: IRBuilder,
        *,
        bt_ptr: Value,
        bt_stride: IntOrValue,
        seq_idx: Value,
        seq: SeqInfo,
        k_n0: int,
        page: IntOrValue,
        sub_page: int | None,
        page_rows: IntOrValue | None = None,
    ) -> None:
        self.b = b
        self.bt_ptr = bt_ptr
        self.bt_stride = bt_stride
        self.seq_idx = seq_idx
        self.seq = seq
        self.k_n0 = k_n0
        self.page = page
        self.sub_page = sub_page
        self.num_blocks: Value | None = None
        if page_rows is not None:
            self.num_blocks = b.div(as_i32(b, page_rows), as_i32(b, page))
        self._reads = itertools.count()
        self._tiles: dict[int, PagedKvTile] = {}

    @property
    def guarded(self) -> bool:
        """Whether the block-id guard is emitted."""
        return self.num_blocks is not None

    def tile(self, k0: Value) -> PagedKvTile:
        key = id(k0)
        if key not in self._tiles:
            self._tiles[key] = PagedKvTile(
                self.b, bt_ptr=self.bt_ptr, bt_stride=self.bt_stride,
                seq_idx=self.seq_idx, seq=self.seq, k0=k0, k_n0=self.k_n0,
                page=self.page, sub_page=self.sub_page,
                num_blocks=self.num_blocks, reads=self._reads,
            )  # fmt: skip
        return self._tiles[key]

    def cache(
        self,
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[IntOrValue, IntOrValue, IntOrValue],
        stage_vec: int,
    ) -> PagedCacheRows:
        return PagedCacheRows(
            self, ptr, dtype=dtype, strides=strides, stage_vec=stage_vec
        )
