# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Log-sum-exp (LSE) store for the MFMA / WMMA attention forward inner bodies.

The functions here produce *epilogue callables* for
:class:`rocke.helpers.attention_fwd_ext.AttnFwdExt.epilogue_hook`.  The inner
body calls the hook once per row slot after the K loop; the hook stores one
FP32 value per valid query row and returns the optional extra row-validity
predicate it was given (so the same predicate also zeroes the O row).

Convention
----------
* natural-log LSE: ``(m_log2 + log2(l)) * ln(2)`` where ``m_log2`` is the
  running max of the log2-domain scores and ``l`` the running sum of
  ``exp2(s - m_log2)``;
* stored as FP32;
* rows with no kept key store ``-inf``.  Whether a row is dead comes from the
  body's row validity (running-max sentinel and ``q_row < seqlen_q``) and from
  ``extra_valid`` when given -- never from ``l``;
* rows ``>= seqlen_q`` are not written at all.

Layout
------
The element index of row ``q_row`` of head ``h`` is
``(row_base + q_row) * row_stride + h * head_stride``.  Strides are in
elements and may be Python ints or i32 values.  Typical use:

* ``[B, H_q, S_q, 1]``: ``row_stride=1``, ``head_stride=S_q``, batch through
  ``batch_idx`` / ``batch_stride`` (a 64-bit pointer rebase);
* packed ``[T_q, H_q, 1]``: ``row_stride=H_q``, ``head_stride=1`` and
  ``row_base`` the first token of the sequence.

Body awareness
--------------
The MFMA accumulator replicates each row across 16 lanes and a lane owns 4
row slots; the WMMA accumulator gives a lane 8 row slots.  The body hands the
hook an ``is_row_leader`` predicate that is true on exactly one lane per row,
so every row is stored once; the two factories differ only in the slot count
they accept.  Nothing is allocated (no LDS, no scratch).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from rocke.core.ir import I64, IRBuilder, Value
from rocke.helpers.attention_fwd_ext import RowEpilogue, natural_lse_from_log2_stats

__all__ = [
    "LSE_ROWS_PER_LANE",
    "lse_row_value",
    "make_lse_epilogue",
    "make_mfma_lse_epilogue",
    "make_wmma_lse_epilogue",
]

# Row slots a lane holds per accumulator tile.
LSE_ROWS_PER_LANE = {"mfma": 4, "wmma": 8}

Stride = Union[int, Value]
ExtraValid = Callable[[IRBuilder, RowEpilogue], Optional[Value]]


def _i32(b: IRBuilder, x: Stride) -> Value:
    return b.const_i32(x) if isinstance(x, int) else x


def _scaled(b: IRBuilder, idx: Value, stride: Stride) -> Value:
    # A literal stride of 1 is the common contiguous case; skip the multiply.
    if isinstance(stride, int) and stride == 1:
        return idx
    return b.mul(idx, _i32(b, stride))


def lse_row_value(b: IRBuilder, e: RowEpilogue, valid: Value) -> Value:
    """FP32 natural-log LSE of the row, ``-inf`` where ``valid`` is false."""
    return b.select(
        valid,
        natural_lse_from_log2_stats(b, e.m_log2, e.l),
        b.const_f32(float("-inf")),
    )


def make_lse_epilogue(
    b: IRBuilder,
    lse: Value,
    *,
    head_idx: Value,
    row_stride: Stride,
    head_stride: Stride,
    max_slots: int,
    row_base: Optional[Value] = None,
    batch_idx: Optional[Value] = None,
    batch_stride: Optional[Stride] = None,
    extra_valid: Optional[ExtraValid] = None,
) -> Callable[[IRBuilder, RowEpilogue], Optional[Value]]:
    """Return an epilogue hook that stores the natural-log LSE.

    ``lse`` is the FP32 output pointer.  With ``batch_idx`` and
    ``batch_stride`` (elements) the pointer is rebased once, here, with a
    64-bit byte offset, so the per-row index stays 32-bit.  ``extra_valid``
    runs once per row slot and may return an i1 that is ANDed into the row
    validity; the same i1 is returned to the body so the O row is zeroed
    consistently.  Slots at or beyond ``max_slots`` are rejected at build time.
    """
    if (batch_idx is None) != (batch_stride is None):
        raise ValueError("batch_idx and batch_stride must be given together")
    base = lse
    if batch_idx is not None:
        off = b.mul(b.zext(batch_idx, I64), b.zext(_i32(b, batch_stride), I64))
        base = b.global_ptr_add(lse, b.mul(off, b.const_i64(4)))

    def hook(bb: IRBuilder, e: RowEpilogue) -> Optional[Value]:
        if not 0 <= e.slot < max_slots:
            raise ValueError(
                f"row slot {e.slot} outside 0..{max_slots - 1} for this body"
            )
        extra = extra_valid(bb, e) if extra_valid is not None else None
        valid = e.row_valid if extra is None else bb.land(e.row_valid, extra)
        row = e.q_row if row_base is None else bb.add(row_base, e.q_row)
        idx = bb.add(_scaled(bb, row, row_stride), _scaled(bb, head_idx, head_stride))
        store_ok = e.is_row_leader
        if e.in_range is not None:
            store_ok = bb.land(store_ok, e.in_range)
        value = lse_row_value(bb, e, valid)
        with bb.scf_if(store_ok):
            bb.global_store(base, idx, value, align=4)
        return extra

    return hook


def make_mfma_lse_epilogue(b: IRBuilder, lse: Value, **layout):
    """LSE epilogue for the MFMA body (4 row slots per lane)."""
    return make_lse_epilogue(b, lse, max_slots=LSE_ROWS_PER_LANE["mfma"], **layout)


def make_wmma_lse_epilogue(b: IRBuilder, lse: Value, **layout):
    """LSE epilogue for the WMMA body (8 row slots per lane)."""
    return make_lse_epilogue(b, lse, max_slots=LSE_ROWS_PER_LANE["wmma"], **layout)
