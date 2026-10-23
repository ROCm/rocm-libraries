# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Address generators for the attention backward kernels.

Family glue for the backward kernels (prep, main, convert and the paged
roles): Python-only, no C++ twin.  Everything here emits integer IR through
the builder it is handed and only calls integer / pointer / load / store
methods of ``IRBuilder``, so the same code can be evaluated with a
duck-typed integer builder on the host (that is how the unit tests check the
formulas against numpy strides).

Sequence decode
---------------
``decode_batched`` / ``decode_thd`` turn the kernel arguments of one batch
(or THD sequence) into a :class:`SeqInfo`:

* batched: ``len = has_len ? clamp(SEQ[b * len_stride], 0, S_max) : S_max``;
  workspace row base ``b * S_max``.
* THD: ``o0 = OFF[b]``, ``o1 = OFF[b + 1]`` (int32 or int64 table, selected by
  ``off64``), ``tok0 = o0 * mult / div`` and ``n = (o1 - o0) * mult / div``
  in i64; ``len = clamp(n, 0, S_max)``, further ``min``-ed with the clamped
  ``SEQ`` value when ``has_len``.  Workspace row base: ``tok0`` (rows by
  storage token, ``ws_seg = 0``: the caller bounds the storage tokens) or
  ``b * S_max`` (rows by sequence slot, ``ws_seg != 0``: ``B * S_max`` rows
  bound every request without knowing the offsets).

``SEQ`` is never read when ``has_len`` is 0 (the pointer may be null): the
read sits in a loop whose trip count is ``has_len != 0``.  An int64 offset
table is read as two i32 words (little endian); for an int32 table the second
read repeats the first index, so the table is never read past ``B + 1``
entries.

Strided tensors
---------------
:class:`StridedTensorAddr` addresses a user tensor with per-tensor element
strides ``(batch, head, token)`` and unit stride along the head dimension.
``slab(h, row0)`` rebases the pointer once with a 64-bit byte offset through
``global_ptr_add`` (batch or token base, head and first row of the tile), so
the per-element index that remains is a small i32 (``< tile_rows * t_stride``
plus the head-dimension column).  With ``clamp=True`` rows are clamped to
``max(len - 1, 0)`` (MMA operands are always read from a valid row); with
``clamp=False`` the raw row is addressed (stores of padding rows).
``stage_vec`` selects the global access width of user tensors only: 8 issues
one 16-byte vector per 8 elements, 1 issues 2-byte scalar accesses.

Workspace
---------
:class:`WorkspaceAddr` addresses an fp32 workspace array
``[h][rows][X]`` (``head_major``) or ``[rows][h][X]`` (``token_major``).  The
per-(head, base row) offset is formed in i64 and folded into the pointer; the
remaining in-slab index is i32.  Rows at or past ``ws_rows`` are clamped to a
valid row and their values are replaced (reads) or zeroed (writes, atomics),
so no access leaves the workspace; ``WsSlab.atomic_add_row`` instead skips
such a row with a branch (epilogue atomics, one branch per row).

Packed rows
-----------
:class:`PackedRows` maps the M rows of one q step to ``(head, row)`` pairs
when several query heads share one tile (head packing for short query spans):
row ``r`` belongs to head ``head0 + r / rows_per_head`` at row
``row0 + r % rows_per_head``; rows of a head index at or past ``n_heads`` are
dead (their head is clamped to the last valid one for the address).

Paged caches
------------
:class:`PagedCacheAddr` addresses a ``[num_blocks, page, h, D]`` cache with
element strides ``(blk, tok, h)`` after a 64-bit rebase to the physical page;
``block_table_entry`` reads one block-table entry with an i64 sequence rebase.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Iterator
from dataclasses import dataclass

from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, Type, Value
from rocke.helpers.attention_band import I32_FAR
from rocke.helpers.spec import ceil_div

__all__ = [
    "PackedRows",
    "PagedCacheAddr",
    "RowSlab",
    "SeqInfo",
    "StridedBatchedAddr",
    "StridedTensorAddr",
    "ThdOffsetAddr",
    "WorkspaceAddr",
    "WsSlab",
    "as_i32",
    "as_i64",
    "block_table_entry",
    "chunk_coords",
    "deal_chunks",
    "decode_batched",
    "decode_thd",
    "last_row",
    "load_offset",
]

IntOrValue = int | Value

_ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}

# Per-builder counter of the guarded SEQ reads (unique loop names per kernel).
_SEQ_GUARDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def as_i32(b: IRBuilder, x: IntOrValue) -> Value:
    """An int as an i32 constant; an SSA value unchanged."""
    return b.const_i32(x) if isinstance(x, int) else x


def as_i64(b: IRBuilder, x: IntOrValue, *, signed: bool = False) -> Value:
    """Widen an int or i32 value to i64 (``zext`` unless ``signed``)."""
    if isinstance(x, int):
        return b.const_i64(x)
    if x.type == I64:
        return x
    return b.sext(x, I64) if signed else b.zext(x, I64)


def _min64(b: IRBuilder, x: Value, y: Value) -> Value:
    # smin/smax lower as i32 intrinsics only; i64 uses compare + select.
    return b.select(b.cmp_lt(x, y), x, y)


def _max64(b: IRBuilder, x: Value, y: Value) -> Value:
    return b.select(b.cmp_lt(x, y), y, x)


def _clamp64_to_i32(b: IRBuilder, x: Value, hi: Value) -> Value:
    """``clamp(x, 0, hi)`` of an i64 value, returned as i32 (``hi`` is i32)."""
    lo = _max64(b, x, b.const_i64(0))
    return b.trunc(_min64(b, lo, as_i64(b, hi)), I32)


def deal_chunks(
    b: IRBuilder, tid: Value, total: int, threads: int
) -> Iterator[tuple[Value, bool]]:
    """Thread ``tid``'s chunks of a ``total``-chunk tile dealt round-robin over
    ``threads`` threads: one ``(chunk, partial)`` per round, ``chunk = tid +
    round * threads`` (i32) and ``partial`` (a host bool) true for a round that
    reaches past ``total``, where the caller guards ``chunk < total``.

    Lazy: a round's index is emitted when the caller asks for that round.
    """
    for it in range(ceil_div(total, threads)):
        chunk = tid if it == 0 else b.add(tid, b.const_i32(it * threads))
        yield chunk, (it + 1) * threads > total


def chunk_coords(b: IRBuilder, chunk: Value, per_row: int, vec: int) -> tuple:
    """``(row, column)`` of a ``vec``-element chunk of rows of ``per_row``
    chunks: ``(chunk / per_row, (chunk % per_row) * vec)``; ``(chunk, 0)``
    with no division for one chunk per row."""
    if per_row == 1:
        return chunk, b.const_i32(0)
    r = b.div(chunk, b.const_i32(per_row))
    c = b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(vec))
    return r, c


def last_row(b: IRBuilder, length: Value) -> Value:
    """``max(length - 1, 0)``: the clamp bound of a row index into ``length`` rows."""
    return b.smax(b.sub(length, b.const_i32(1)), b.const_i32(0))


def _scaled32(b: IRBuilder, idx: Value, stride: IntOrValue) -> Value:
    if isinstance(stride, int):
        if stride == 1:
            return idx
        if stride == 0:
            return b.const_i32(0)
    return b.mul(idx, as_i32(b, stride))


def _scaled64(b: IRBuilder, idx64: Value, stride: IntOrValue) -> Value:
    if isinstance(stride, int):
        if stride == 1:
            return idx64
        return b.mul(idx64, b.const_i64(stride))
    return b.mul(idx64, as_i64(b, stride))


# ---------------------------------------------------------------------------
# Sequence decode
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SeqInfo:
    """Decoded valid length and bases of one batch / THD sequence.

    ``length`` is i32 in ``[0, s_max]``; ``last`` is ``max(length - 1, 0)``;
    ``tok0`` is the i64 first token (THD only, ``None`` in batched mode);
    ``ws_row0`` is the i64 first workspace row (``batch * s_max``, or in THD
    mode ``tok0`` / ``batch * s_max`` by the ``ws_seg`` flag).
    """

    mode: str  # "batched" | "thd"
    batch: Value
    length: Value
    last: Value
    s_max: Value
    ws_row0: Value
    tok0: Value | None = None


def load_offset(b: IRBuilder, off_ptr: Value, idx: Value, off64: IntOrValue) -> Value:
    """Entry ``idx`` of an int32 or int64 offset table, as i64.

    ``off64`` is a compile-time int (0 / 1) or an i32 runtime flag.  The int64
    form reads words ``2*idx`` (low) and ``2*idx + 1`` (high); the int32 form
    reads word ``idx`` twice and sign-extends it.
    """
    if isinstance(off64, int):
        if off64:
            two_i = b.mul(idx, b.const_i32(2))
            lo = b.global_load_i32(off_ptr, two_i)
            hi = b.global_load_i32(off_ptr, b.add(two_i, b.const_i32(1)))
            return b.lor(b.zext(lo, I64), b.shl(b.sext(hi, I64), b.const_i64(32)))
        return b.sext(b.global_load_i32(off_ptr, idx), I64)
    is64 = b.cmp_ne(off64, b.const_i32(0))
    two_i = b.mul(idx, b.const_i32(2))
    i_lo = b.select(is64, two_i, idx)
    i_hi = b.select(is64, b.add(two_i, b.const_i32(1)), idx)
    lo = b.global_load_i32(off_ptr, i_lo)
    hi = b.global_load_i32(off_ptr, i_hi)
    wide = b.lor(b.zext(lo, I64), b.shl(b.sext(hi, I64), b.const_i64(32)))
    return b.select(is64, wide, b.sext(lo, I64))


def _seq_len_value(
    b: IRBuilder,
    seq_ptr: Value | None,
    batch: Value,
    *,
    has_len: IntOrValue,
    len_stride: IntOrValue,
    s_max: Value,
) -> Value | None:
    """Clamped ``SEQ[batch * len_stride]`` or ``None`` when statically absent.

    With a runtime ``has_len`` the read sits in a 0/1-trip loop, so a null
    ``SEQ`` pointer is never dereferenced; the result is ``s_max`` when the
    flag is 0.
    """
    if isinstance(has_len, int) and not has_len:
        return None
    if seq_ptr is None:
        raise ValueError("has_len needs a SEQ pointer")
    zero = b.const_i32(0)

    def read() -> Value:
        raw = b.global_load_i32(seq_ptr, _scaled32(b, batch, len_stride))
        return b.smin(b.smax(raw, zero), s_max)

    if isinstance(has_len, int):
        return read()
    trip = b.select(b.cmp_ne(has_len, zero), b.const_i32(1), zero)
    # Loop names must be unique per kernel; a per-builder counter (owned by
    # this module, keyed weakly on the builder) keeps the emitted text
    # deterministic.
    n = next(_SEQ_GUARDS.setdefault(b, itertools.count()))
    loop = b.scf_for_iter(
        zero, trip, b.const_i32(1), [(f"seqlen{n}", s_max)], iv_name=f"seq_i{n}"
    )
    with loop as (_iv, (_cur,)):
        b.scf_yield(read())
    return loop.results[0]


def decode_batched(
    b: IRBuilder,
    batch: Value,
    *,
    s_max: IntOrValue,
    seq_ptr: Value | None = None,
    has_len: IntOrValue = 0,
    len_stride: IntOrValue = 1,
) -> SeqInfo:
    """Valid length and bases of batch ``batch`` (fixed or padded lengths)."""
    smax = as_i32(b, s_max)
    length = _seq_len_value(
        b, seq_ptr, batch, has_len=has_len, len_stride=len_stride, s_max=smax
    )
    if length is None:
        length = smax
    ws_row0 = b.mul(as_i64(b, batch), as_i64(b, smax))
    return SeqInfo(
        mode="batched",
        batch=batch,
        length=length,
        last=last_row(b, length),
        s_max=smax,
        ws_row0=ws_row0,
    )


def decode_thd(
    b: IRBuilder,
    batch: Value,
    *,
    s_max: IntOrValue,
    off_ptr: Value,
    off64: IntOrValue = 0,
    mult: IntOrValue = 1,
    div: IntOrValue = 1,
    seq_ptr: Value | None = None,
    has_len: IntOrValue = 0,
    len_stride: IntOrValue = 1,
    ws_seg: IntOrValue = 0,
) -> SeqInfo:
    """Valid length and token base of THD sequence ``batch``.

    Offsets are converted to tokens as ``off * mult / div`` in i64 (``div`` is
    the sequence stride of the tensor in elements when the offsets are element
    offsets; 1 for token offsets).

    ``ws_seg`` selects the workspace row base: 0 (a literal 0 emits nothing)
    gives ``tok0``, the storage token; nonzero gives ``batch * s_max``, the
    sequence slot, so ``B * S_max`` rows bound every sequence whatever the
    storage holds before, between or after the sequences.
    """
    smax = as_i32(b, s_max)
    o0 = load_offset(b, off_ptr, batch, off64)
    o1 = load_offset(b, off_ptr, b.add(batch, b.const_i32(1)), off64)
    m64 = as_i64(b, mult, signed=True)
    d64 = as_i64(b, div, signed=True)
    one_mult = isinstance(mult, int) and mult == 1
    one_div = isinstance(div, int) and div == 1

    def to_tok(x: Value) -> Value:
        if not one_mult:
            x = b.mul(x, m64)
        if not one_div:
            x = b.div(x, d64)
        return x

    tok0 = to_tok(o0)
    n = to_tok(b.sub(o1, o0))
    length = _clamp64_to_i32(b, n, smax)
    seq_len = _seq_len_value(
        b, seq_ptr, batch, has_len=has_len, len_stride=len_stride, s_max=smax
    )
    if seq_len is not None:
        length = b.smin(length, seq_len)
    ws_row0 = tok0
    if not (isinstance(ws_seg, int) and ws_seg == 0):
        slot0 = b.mul(as_i64(b, batch), as_i64(b, smax))
        if isinstance(ws_seg, int):
            ws_row0 = slot0
        else:
            # CTA-uniform; readfirstlane (convergent) keeps the selected base
            # from being sunk past the q loops, where its operands would stay
            # live in SGPRs instead of the one base
            ws_row0 = b.readfirstlane(
                b.select(b.cmp_ne(ws_seg, b.const_i32(0)), slot0, tok0)
            )
    return SeqInfo(
        mode="thd",
        batch=batch,
        length=length,
        last=last_row(b, length),
        s_max=smax,
        ws_row0=ws_row0,
        tok0=tok0,
    )


# ---------------------------------------------------------------------------
# Strided user tensors
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: Type) -> int:
    try:
        return _ELEM_BYTES[dtype.name]
    except KeyError:
        raise ValueError(f"unsupported element type {dtype.name}") from None


def sgpr_ptr(b: IRBuilder, ptr: Value, parts: tuple | None) -> Value:
    """``ptr``, a wave-uniform slab pointer, for an operand that must be in
    SGPRs.

    ``parts`` is ``(base, byte_off, ptr)``: ``ptr`` was formed as ``base +
    byte_off`` (``byte_off`` None: ``ptr`` is ``base``). The pointer is formed
    again as ``base + readfirstlane(byte_off)``. An inline-asm ``"s"`` operand
    (the LDS DMA's 64-bit global base) takes the registers the compiler
    materialised the value in, and the production compiler can place a
    uniform pointer in a VGPR pair, which the assembler rejects; the
    readfirstlane puts the value in SGPRs.
    """
    if parts is None or parts[2] is not ptr:
        raise ValueError("the slab pointer was not formed from (base, offset)")
    base, off, _ptr = parts
    if off is None:
        return base
    return b.global_ptr_add(base, b.readfirstlane(off))


@dataclass(frozen=True, eq=False)
class RowSlab:
    """A tensor rebased to ``(batch, head, base_row)``; i32 in-slab indexing."""

    b: IRBuilder
    ptr: Value
    dtype: Type
    t_stride: IntOrValue
    base_row: Value  # i32 row the pointer points at
    last: Value | None  # i32 clamp bound (None: rows are not clamped)
    stage_vec: int
    # elements per access when stage_vec = 1 (2 or 4: every stride and the
    # base allow that width; 1: one element at a time)
    narrow: int = 1
    # how ``ptr`` was formed: (base, i64 byte offset or None, ptr); see sgpr_ptr
    parts: tuple | None = None

    def sgpr_ptr(self) -> Value:
        """``ptr`` for an operand that must be in SGPRs (:func:`sgpr_ptr`)."""
        return sgpr_ptr(self.b, self.ptr, self.parts)

    def local_row(self, row: Value) -> Value:
        """i32 row offset from the slab base (clamped when the slab clamps)."""
        b = self.b
        r = row if self.last is None else b.smin(row, self.last)
        return b.sub(r, self.base_row)

    def index(self, row: Value, col: IntOrValue = 0) -> Value:
        """i32 element index of ``(row, col)`` relative to the rebased pointer."""
        b = self.b
        idx = _scaled32(b, self.local_row(row), self.t_stride)
        if isinstance(col, int) and col == 0:
            return idx
        return b.add(idx, as_i32(b, col))

    def load(self, row: Value, col: IntOrValue = 0) -> Value:
        """One element (the tensor dtype)."""
        esize = _elem_bytes(self.dtype)
        return self.b.global_load(
            self.ptr, self.index(row, col), self.dtype, align=esize
        )

    def store(self, row: Value, col: IntOrValue, value: Value) -> None:
        esize = _elem_bytes(self.dtype)
        self.b.global_store(self.ptr, self.index(row, col), value, align=esize)

    def load_vec(self, row: Value, col0: IntOrValue, n: int = 8) -> Value:
        """``n`` consecutive elements as ``<n x dtype>`` (``stage_vec`` width)."""
        b = self.b
        idx = self.index(row, col0)
        esize = _elem_bytes(self.dtype)
        if self.stage_vec == 8 and n % 8 == 0 and esize == 2:
            if n == 8:
                return b.global_load_vN(self.ptr, idx, self.dtype, 8, align=16)
            parts = [
                b.global_load_vN(
                    self.ptr, b.add(idx, b.const_i32(8 * i)), self.dtype, 8, align=16
                )
                for i in range(n // 8)
            ]
            out = parts[0]
            for p in parts[1:]:
                out = b.vec_concat(out, p)
            return out
        w = self.narrow
        if self.stage_vec != 8 and w > 1 and n % w == 0 and esize == 2:
            parts = [
                b.global_load_vN(
                    self.ptr,
                    idx if k == 0 else b.add(idx, b.const_i32(k)),
                    self.dtype,
                    w,
                    align=2 * w,
                )
                for k in range(0, n, w)
            ]
            out = parts[0]
            for p in parts[1:]:
                out = b.vec_concat(out, p)
            return out
        elems = [
            b.global_load(
                self.ptr,
                idx if i == 0 else b.add(idx, b.const_i32(i)),
                self.dtype,
                align=esize,
            )
            for i in range(n)
        ]
        return b.vec_pack(elems, self.dtype)

    def store_vec(self, row: Value, col0: IntOrValue, value: Value) -> None:
        """Store ``<n x dtype>`` at ``(row, col0 .. col0 + n)`` (``stage_vec`` width).

        With ``stage_vec = 8`` (16-byte aligned rows) an ``n`` of 8, 4 or 2
        16-bit elements at a column that is a multiple of ``n`` is one vector
        store of ``2 n`` bytes; otherwise the elements are stored one by one.
        """
        b = self.b
        n = value.type.count
        idx = self.index(row, col0)
        esize = _elem_bytes(self.dtype)
        if self.stage_vec == 8 and n in (8, 4, 2) and esize == 2:
            b.global_store_vN(self.ptr, idx, value, n, align=2 * n)
            return
        w = self.narrow
        if self.stage_vec != 8 and w > 1 and n % w == 0 and esize == 2:
            for k in range(0, n, w):
                part = value
                if n != w:
                    part = b.vec_pack(
                        [b.vec_extract(value, k + i) for i in range(w)], self.dtype
                    )
                b.global_store_vN(
                    self.ptr,
                    idx if k == 0 else b.add(idx, b.const_i32(k)),
                    part,
                    w,
                    align=2 * w,
                )
            return
        for i in range(n):
            b.global_store(
                self.ptr,
                idx if i == 0 else b.add(idx, b.const_i32(i)),
                b.vec_extract(value, i),
                align=esize,
            )


class StridedTensorAddr:
    """Element-strided ``(batch | token, head, row, d)`` tensor of one sequence.

    ``strides`` are ``(batch, head, token)`` element strides (ints or i32
    values); the head dimension has unit stride.  In THD mode the batch stride
    is ignored and the sequence base is ``tok0 * token_stride``.  Packed views
    (for example Q, K and V inside one QKV tensor) are expressed by their
    strides and a pointer to their first element.
    """

    def __init__(
        self,
        b: IRBuilder,
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[IntOrValue, IntOrValue, IntOrValue],
        seq: SeqInfo,
        stage_vec: int = 8,
        narrow_vec: int = 1,
    ) -> None:
        if stage_vec not in (8, 1):
            raise ValueError(f"stage_vec must be 8 or 1 (got {stage_vec})")
        if narrow_vec not in (1, 2, 4) or (narrow_vec != 1 and stage_vec != 1):
            raise ValueError(f"narrow_vec={narrow_vec} needs stage_vec = 1")
        self.narrow_vec = narrow_vec
        if dtype not in (F16, BF16, F32):
            raise ValueError(f"unsupported tensor dtype {dtype.name}")
        if len(strides) != 3:
            raise ValueError("strides must be (batch, head, token)")
        self.b = b
        self.ptr = ptr
        self.dtype = dtype
        self.strides = strides
        self.seq = seq
        self.stage_vec = stage_vec
        self._esize = _elem_bytes(dtype)
        s_b, _s_h, s_t = strides
        # Sequence base in elements (i64), formed once. In THD mode it is folded
        # into the base pointer here, so one 64-bit pointer (not a pointer and a
        # 64-bit offset) stays live across the q loops: the THD kernels sit at
        # the SGPR ceiling on gfx950 and spill without it.
        self._base = ptr
        if seq.mode == "thd":
            if seq.tok0 is None:
                raise ValueError("THD sequence info without tok0")
            seq_base = _scaled64(b, seq.tok0, s_t)
            self._base = b.global_ptr_add(
                ptr, b.mul(seq_base, b.const_i64(self._esize))
            )
            self._seq_base = None
        else:
            self._seq_base = _scaled64(b, as_i64(b, seq.batch), s_b)

    def slab(
        self, h: IntOrValue, row0: IntOrValue = 0, *, clamp: bool = True
    ) -> RowSlab:
        """Rebase to head ``h`` and tile row ``row0`` (i64 byte offset).

        ``clamp=True``: the base row is ``min(row0, last)`` and every row is
        clamped to ``last``; ``clamp=False``: the base row is ``row0`` and rows
        are addressed as given (the caller predicates the access).
        """
        b = self.b
        _s_b, s_h, s_t = self.strides
        r0 = as_i32(b, row0)
        static_zero = isinstance(row0, int) and row0 == 0
        # min(0, last) == 0 because last >= 0, so a literal 0 needs no clamp.
        base_row = b.smin(r0, self.seq.last) if clamp and not static_zero else r0
        off = self._seq_base
        if not (isinstance(h, int) and h == 0):
            term = _scaled64(b, as_i64(b, h), s_h)
            off = term if off is None else b.add(off, term)
        if not static_zero:
            term = _scaled64(b, as_i64(b, base_row), s_t)
            off = term if off is None else b.add(off, term)
        if off is None:
            ptr = self._base
        else:
            off = b.mul(off, b.const_i64(self._esize))
            ptr = b.global_ptr_add(self._base, off)
        return RowSlab(
            b=b,
            ptr=ptr,
            dtype=self.dtype,
            t_stride=s_t,
            base_row=base_row,
            last=self.seq.last if clamp else None,
            stage_vec=self.stage_vec,
            narrow=self.narrow_vec,
            parts=(self._base, off, ptr),
        )

    def offset_bytes(self, h: IntOrValue, row0: IntOrValue = 0) -> Value:
        """The i64 byte offset :meth:`slab` adds to the base pointer for head
        ``h`` and (unclamped) row ``row0``."""
        b = self.b
        _s_b, s_h, s_t = self.strides
        off = self._seq_base
        if not (isinstance(h, int) and h == 0):
            term = _scaled64(b, as_i64(b, h), s_h)
            off = term if off is None else b.add(off, term)
        if not (isinstance(row0, int) and row0 == 0):
            term = _scaled64(b, as_i64(b, as_i32(b, row0)), s_t)
            off = term if off is None else b.add(off, term)
        if off is None:
            return b.const_i64(0)
        return b.mul(off, b.const_i64(self._esize))

    def stride_bytes(self, which: str) -> Value:
        """The i64 byte stride of one head (``"h"``) or one row (``"t"``)."""
        _s_b, s_h, s_t = self.strides
        st = {"h": s_h, "t": s_t}[which]
        return _scaled64(self.b, as_i64(self.b, st), self._esize)

    def slab_at(self, off: Value, row0: Value, *, clamp: bool = True) -> RowSlab:
        """The slab whose pointer is the base plus the i64 byte offset ``off``
        (:meth:`offset_bytes` of a head and of ``row0``, possibly carried
        across steps) at base row ``row0``. Equal to :meth:`slab` ``(h, row0,
        clamp=clamp)`` when ``row0 <= last`` (the clamp of the base row is then
        an identity), which holds for every q tile a q loop visits."""
        ptr = self.b.global_ptr_add(self._base, off)
        return RowSlab(
            b=self.b,
            ptr=ptr,
            dtype=self.dtype,
            t_stride=self.strides[2],
            base_row=as_i32(self.b, row0),
            last=self.seq.last if clamp else None,
            stage_vec=self.stage_vec,
            narrow=self.narrow_vec,
            parts=(self._base, off, ptr),
        )


class StridedBatchedAddr(StridedTensorAddr):
    """:class:`StridedTensorAddr` of a batched (fixed or padded) sequence."""

    def __init__(self, b: IRBuilder, ptr: Value, *, seq: SeqInfo, **kw) -> None:
        if seq.mode != "batched":
            raise ValueError("StridedBatchedAddr needs a batched SeqInfo")
        super().__init__(b, ptr, seq=seq, **kw)


class ThdOffsetAddr(StridedTensorAddr):
    """:class:`StridedTensorAddr` of a THD sequence (token rebase in i64)."""

    def __init__(self, b: IRBuilder, ptr: Value, *, seq: SeqInfo, **kw) -> None:
        if seq.mode != "thd":
            raise ValueError("ThdOffsetAddr needs a THD SeqInfo")
        super().__init__(b, ptr, seq=seq, **kw)


# ---------------------------------------------------------------------------
# fp32 workspace
# ---------------------------------------------------------------------------

WS_LAYOUTS = ("head_major", "token_major")


@dataclass(frozen=True, eq=False)
class WsSlab:
    """Workspace rows of one head from a base row on; i32 in-slab indexing.

    ``rows_left`` is ``clamp(ws_rows - base, 0, 2**30)``: row offsets ``r``
    with ``r < rows_left`` are inside the workspace.  Other rows are clamped
    to a valid row for the address and masked for the value.
    """

    b: IRBuilder
    ptr: Value
    rows_left: Value
    row_stride: IntOrValue  # X (head-major) or H * X (token-major), elements
    # how ``ptr`` was formed: (base, i64 byte offset, ptr); see sgpr_ptr
    parts: tuple | None = None

    def sgpr_ptr(self) -> Value:
        """``ptr`` for an operand that must be in SGPRs (:func:`sgpr_ptr`)."""
        return sgpr_ptr(self.b, self.ptr, self.parts)

    def row_ok(self, r: Value) -> Value:
        return self.b.cmp_lt(r, self.rows_left)

    def local_row(self, r: Value) -> Value:
        b = self.b
        hi = b.sub(self.rows_left, b.const_i32(1))
        return b.smax(b.smin(r, hi), b.const_i32(0))

    def index(self, r: Value, x: IntOrValue = 0) -> Value:
        b = self.b
        idx = _scaled32(b, self.local_row(r), self.row_stride)
        if isinstance(x, int) and x == 0:
            return idx
        return b.add(idx, as_i32(b, x))

    def load(self, r: Value, x: IntOrValue, neutral: float) -> Value:
        """f32 at ``(r, x)``; ``neutral`` for rows outside the workspace."""
        b = self.b
        v = b.global_load_f32(self.ptr, self.index(r, x))
        return b.select(self.row_ok(r), v, b.const_f32(neutral))

    def load_raw(self, r: Value, x: IntOrValue) -> Value:
        """f32 at the clamped address of ``(r, x)``, without the neutral value
        for rows outside the workspace (:meth:`neutral` applies it later)."""
        return self.b.global_load_f32(self.ptr, self.index(r, x))

    def neutral(self, r: Value, v: Value, neutral: float) -> Value:
        """``v`` (a :meth:`load_raw` of row ``r``) or ``neutral`` outside the
        workspace: :meth:`load` in two parts."""
        b = self.b
        return b.select(self.row_ok(r), v, b.const_f32(neutral))

    def atomic_add(self, r: Value, x: IntOrValue, v: Value) -> None:
        """Branch-free fp32 atomic add; rows outside the workspace add 0."""
        b = self.b
        val = b.select(self.row_ok(r), v, b.const_f32(0.0))
        b.global_atomic_add_f32(self.ptr, self.index(r, x), val)

    def atomic_add_row(
        self, r: Value, items: list[tuple[IntOrValue, Value]], ok: Value | None = None
    ) -> None:
        """fp32 atomic adds of ``(x, v)`` pairs into row ``r``, predicated.

        One branch per row: the atomics run only when ``r`` is inside the
        workspace (and ``ok``, when given). Inside the branch the index is
        ``r * row_stride + x`` without a clamp, so the atomics of one row share
        their row base and differ by their column only (constant column
        offsets fold into the address).
        """
        b = self.b
        cond = self.row_ok(r) if ok is None else b.land(self.row_ok(r), ok)
        with b.scf_if(cond):
            base = _scaled32(b, r, self.row_stride)
            for x, v in items:
                idx = (
                    base if isinstance(x, int) and x == 0 else b.add(base, as_i32(b, x))
                )
                b.global_atomic_add_f32(self.ptr, idx, v)

    def atomic_add_row_select(
        self, r: Value, items: list[tuple[IntOrValue, Value]], ok: Value | None = None
    ) -> None:
        """Branch-free :meth:`atomic_add_row`: every ``(x, v)`` pair is added.

        A row outside the workspace (or with ``ok`` false) adds ``+0.0`` at
        the clamped row instead of skipping its atomics. An fp32 sum that
        starts at ``+0.0`` is never ``-0.0`` under round-to-nearest, so adding
        ``+0.0`` leaves every element bit for bit unchanged.
        """
        b = self.b
        cond = self.row_ok(r) if ok is None else b.land(self.row_ok(r), ok)
        zero = b.const_f32(0.0)
        base = _scaled32(b, self.local_row(r), self.row_stride)
        for x, v in items:
            idx = base if isinstance(x, int) and x == 0 else b.add(base, as_i32(b, x))
            b.global_atomic_add_f32(self.ptr, idx, b.select(cond, v, zero))

    def store_row(
        self, r: Value, items: list[tuple[IntOrValue, Value]], ok: Value | None = None
    ) -> None:
        """fp32 stores of ``(x, v)`` pairs into row ``r``, predicated.

        The plain-store twin of :meth:`atomic_add_row`: one branch per row
        (inside the workspace, and ``ok`` when given), an unclamped row base
        shared by the row's stores.
        """
        b = self.b
        cond = self.row_ok(r) if ok is None else b.land(self.row_ok(r), ok)
        with b.scf_if(cond):
            base = _scaled32(b, r, self.row_stride)
            for x, v in items:
                idx = (
                    base if isinstance(x, int) and x == 0 else b.add(base, as_i32(b, x))
                )
                b.global_store(self.ptr, idx, v, align=4)

    def store_vec_in_row(self, r: Value, x: IntOrValue, v: Value) -> None:
        """Store ``<n x f32>`` (``n`` in {1, 2, 4, 8}) at ``(r, x)`` with an
        unclamped index: the caller predicates on :meth:`row_ok`. ``x`` is a
        multiple of ``n``; every workspace row starts 16-byte aligned."""
        b = self.b
        n = v.type.count
        base = _scaled32(b, r, self.row_stride)
        idx = base if isinstance(x, int) and x == 0 else b.add(base, as_i32(b, x))
        if n == 1:
            b.global_store(self.ptr, idx, b.vec_extract(v, 0), align=4)
            return
        if n not in (2, 4, 8):
            raise ValueError(f"workspace row stores take 1, 2, 4 or 8 floats, not {n}")
        b.global_store_vN(self.ptr, idx, v, n, align=min(4 * n, 16))

    def store(self, r: Value, x: IntOrValue, v: Value) -> None:
        """Store at a clamped address; the caller predicates on ``row_ok``."""
        self.b.global_store(self.ptr, self.index(r, x), v, align=4)

    def store_vec(self, r: Value, x: IntOrValue, v: Value) -> None:
        """Store ``<n x f32>`` (``n`` in {4, 8}) as 16-byte pieces."""
        b = self.b
        n = v.type.count
        if n == 4:
            b.global_store_vN(self.ptr, self.index(r, x), v, 4, align=16)
            return
        if n % 4:
            raise ValueError("workspace vector stores take multiples of 4 floats")
        for i in range(n // 4):
            part = b.vec_pack([b.vec_extract(v, 4 * i + j) for j in range(4)], F32)
            xi = x + 4 * i if isinstance(x, int) else b.add(x, b.const_i32(4 * i))
            b.global_store_vN(self.ptr, self.index(r, xi), part, 4, align=16)

    def load_vec4(self, r: Value, x: IntOrValue) -> Value:
        """``<4 x f32>`` at ``(r, x)`` (clamped address, no masking)."""
        return self.b.global_load_vN(self.ptr, self.index(r, x), F32, 4, align=16)


class WorkspaceAddr:
    """fp32 workspace ``[h][rows][X]`` / ``[rows][h][X]`` of one sequence.

    ``row0`` is the sequence's first workspace row (``SeqInfo.ws_row0``, i64).
    ``n_heads`` is required for ``token_major``.
    """

    def __init__(
        self,
        b: IRBuilder,
        ptr: Value,
        *,
        ws_rows: IntOrValue,
        width: int,
        row0: Value,
        layout: str = "head_major",
        n_heads: IntOrValue | None = None,
    ) -> None:
        if layout not in WS_LAYOUTS:
            raise ValueError(f"ws_layout must be one of {WS_LAYOUTS} (got {layout!r})")
        if layout == "token_major" and n_heads is None:
            raise ValueError("token_major workspace needs n_heads")
        if width < 1:
            raise ValueError("workspace width must be >= 1")
        self.b = b
        self.ptr = ptr
        self.ws_rows = as_i32(b, ws_rows)
        self.width = width
        self.row0 = row0 if row0.type == I64 else as_i64(b, row0)
        self.layout = layout
        self.n_heads = n_heads

    def slab(self, h: IntOrValue, tile_row0: IntOrValue = 0) -> WsSlab:
        """Rebase to head ``h`` and row ``row0 + tile_row0`` (i64 offsets)."""
        b = self.b
        base = self.row0
        if not (isinstance(tile_row0, int) and tile_row0 == 0):
            base = b.add(base, as_i64(b, tile_row0, signed=True))
        rows64 = as_i64(b, self.ws_rows)
        one64 = b.const_i64(1)
        base_ws = _max64(b, _min64(b, base, b.sub(rows64, one64)), b.const_i64(0))
        rows_left = _clamp64_to_i32(b, b.sub(rows64, base), b.const_i32(I32_FAR))
        h64 = as_i64(b, h)
        if self.layout == "head_major":
            elem = b.add(b.mul(h64, rows64), base_ws)
            elem = _scaled64(b, elem, self.width)
            row_stride: IntOrValue = self.width
        else:
            elem = b.add(_scaled64(b, base_ws, self.n_heads), h64)
            elem = _scaled64(b, elem, self.width)
            nh = self.n_heads
            if isinstance(nh, int):
                row_stride = nh * self.width
            else:
                row_stride = _scaled32(b, nh, self.width)
        off = b.mul(elem, b.const_i64(4))
        ptr = b.global_ptr_add(self.ptr, off)
        return WsSlab(
            b=b, ptr=ptr, rows_left=rows_left, row_stride=row_stride,
            parts=(self.ptr, off, ptr),
        )  # fmt: skip


# ---------------------------------------------------------------------------
# Packed rows (several query heads on the M rows of one q step)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PackedRows:
    """``(head, row, ok)`` of each M row of a q step that packs query heads.

    ``head0`` is the first query head of the packed group, ``n_heads`` (>= 1)
    the number of valid heads in it, ``rows_per_head`` (>= 1) the rows each
    head occupies and ``row0`` the first sequence row of every head.  With
    ``rows_per_head = k_m0`` and ``n_heads = 1`` the map is the identity of an
    unpacked step (``row = row0 + r``).  Every value is i32.
    """

    head0: Value
    n_heads: Value
    rows_per_head: Value
    row0: Value

    def locate(self, b: IRBuilder, r: IntOrValue) -> tuple[Value, Value, Value]:
        """``(head, row, ok)`` of M row ``r``; dead rows get the last valid head."""
        rr = as_i32(b, r)
        hh = b.div(rr, self.rows_per_head)
        row = b.add(self.row0, b.sub(rr, b.mul(hh, self.rows_per_head)))
        ok = b.cmp_lt(hh, self.n_heads)
        last = b.sub(self.n_heads, b.const_i32(1))
        return b.add(self.head0, b.smin(hh, last)), row, ok


# ---------------------------------------------------------------------------
# Paged caches
# ---------------------------------------------------------------------------


def block_table_entry(
    b: IRBuilder, bt_ptr: Value, seq_idx: Value, j: Value, *, bt_stride: IntOrValue
) -> Value:
    """``block_table[seq_idx][j]`` (int32 table), with an i64 sequence rebase."""
    off = _scaled64(b, as_i64(b, seq_idx), bt_stride)
    row = b.global_ptr_add(bt_ptr, b.mul(off, b.const_i64(4)))
    return b.global_load_i32(row, j)


class PagedCacheAddr:
    """``[num_blocks, page, h, D]`` cache with element strides ``(blk, tok, h)``."""

    def __init__(
        self,
        b: IRBuilder,
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[IntOrValue, IntOrValue, IntOrValue],
        stage_vec: int = 8,
    ) -> None:
        if stage_vec not in (8, 1):
            raise ValueError(f"stage_vec must be 8 or 1 (got {stage_vec})")
        self.b = b
        self.ptr = ptr
        self.dtype = dtype
        self.strides = strides
        self.stage_vec = stage_vec
        self._esize = _elem_bytes(dtype)

    def slab(self, page_id: Value, h: IntOrValue, tok0: IntOrValue = 0) -> RowSlab:
        """Rebase to physical page ``page_id``, head ``h`` and in-page token ``tok0``."""
        b = self.b
        s_blk, s_tok, s_h = self.strides
        off = _scaled64(b, as_i64(b, page_id), s_blk)
        if not (isinstance(h, int) and h == 0):
            off = b.add(off, _scaled64(b, as_i64(b, h), s_h))
        r0 = as_i32(b, tok0)
        if not (isinstance(tok0, int) and tok0 == 0):
            off = b.add(off, _scaled64(b, as_i64(b, r0), s_tok))
        ptr = b.global_ptr_add(self.ptr, b.mul(off, b.const_i64(self._esize)))
        return RowSlab(
            b=b,
            ptr=ptr,
            dtype=self.dtype,
            t_stride=s_tok,
            base_row=r0,
            last=None,
            stage_vec=self.stage_vec,
        )
