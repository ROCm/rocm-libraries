# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The SDPA dropout mask of hipDNN SEM-015, drawn in IR.

A forward and a backward kernel regenerate the same mask from the seed, the
offset and the element's position, so they need not pass it between them. The
helpers are bit-exact against the host oracle ``rocke.numeric.philox_dropout``.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import I32, I64, IRBuilder, Value
from .rng import PhiloxCall, add_u32_wrap

__all__ = [
    "dropout_keep",
    "dropout_row_counter",
    "dropout_words",
]


def dropout_row_counter(
    b: IRBuilder, *, offset_lo: Value, offset_hi: Value, row: Value, max_sk: Value
) -> Tuple[Value, Value]:
    """Counter words of key 0 of a row: ``offset + row * ceil(max_sk / 4)`` mod ``2**64``.

    ``row`` is the i32 ``(z * Hq + h) * max_sq + q`` and ``max_sk`` the declared
    key extent; their product is taken in i64, as it outgrows i32 on long
    sequences. ``offset`` is the u64 sum of hipDNN's offsets as two i32 words.
    Hoist it per row: a key ``kv`` adds ``kv // 4``
    (:func:`rocke.helpers.rng.philox_counter_add`).
    """
    per_row = b.lshr(b.add(max_sk, b.const_i32(3)), b.const_i32(2))
    row_base = b.mul(b.zext(row, I64), b.zext(per_row, I64))
    rb_lo = b.trunc(row_base, I32)
    rb_hi = b.trunc(b.lshr(row_base, b.const_i64(32)), I32)
    lo = add_u32_wrap(b, offset_lo, rb_lo)
    carry = b.zext(b.cmp_ult(lo, rb_lo), I32)
    hi = add_u32_wrap(b, add_u32_wrap(b, offset_hi, rb_hi), carry)
    return lo, hi


def dropout_words(
    b: IRBuilder,
    *,
    seed_lo: Value,
    seed_hi: Value,
    ctr_lo: Value,
    ctr_hi: Value,
    xor3: bool = False,
    mad64: bool = False,
) -> Tuple[Value, Value, Value, Value]:
    """The four SEM-015 words of counter ``(ctr_lo, ctr_hi)``: word ``i`` is key ``4 * (kv // 4) + i``."""
    call = PhiloxCall(
        b,
        seed_lo=seed_lo,
        seed_hi=seed_hi,
        ctr_lo=ctr_lo,
        ctr_hi=ctr_hi,
        xor3=xor3,
        mad64=mad64,
    )
    return call.finish()


def dropout_keep(b: IRBuilder, word: Value, threshold: Value) -> Value:
    """``i1`` keep flag of one SEM-015 word: ``int32(word) > threshold``.

    ``threshold`` is the host's ``rocke.numeric.philox_dropout.dropout_threshold(p)``:
    one signed compare, no float conversion.
    """
    return b.cmp_gt(word, threshold)
