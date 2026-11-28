# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack canonical encoded patterns into carriers, without numerical conversion."""

from __future__ import annotations

from collections.abc import Callable

from ..core.dtypes import dtype_info
from ..core.ir import I8, I16, I32, I64, IRBuilder, Value
from ..core.storage import FragmentPacking

__all__ = ["pack_fragment_bits"]


def pack_fragment_bits(
    b: IRBuilder,
    load_bits: Callable[[int], Value],
    fragment: FragmentPacking,
) -> list[Value]:
    """Pack unsigned encoded patterns into integer carriers, including split fields.

    Encoded fields are at most 32 bits; i64 carriers support eight-byte scale
    words. The loader must return canonical patterns with zero high bits. Tensor
    decoding and numeric quantization are separate operations.
    """
    if fragment.carrier_bits not in (32, 64):
        raise ValueError("IR pattern packing currently requires i32 or i64 carriers")
    if fragment.packing.element_bits > 32:
        raise ValueError(
            "IR pattern packing supports encoded fields of at most 32 bits"
        )
    word_type = I64 if fragment.carrier_bits == 64 else I32
    constant = b.const_i64 if fragment.carrier_bits == 64 else b.const_i32
    words = [constant(0) for _ in range(fragment.carrier_count)]
    for j in range(fragment.count):
        pattern = load_bits(j)
        if pattern.type not in (I8, I16, I32, I64):
            raise ValueError(
                "pattern packing requires unsigned patterns in integer carriers"
            )
        pattern_bits = dtype_info(pattern.type.name).encoded_bits
        if pattern_bits < fragment.packing.element_bits:
            raise ValueError("pattern carrier is smaller than encoded width")
        if pattern_bits < fragment.carrier_bits:
            pattern = b.zext(pattern, word_type)
        elif pattern_bits > fragment.carrier_bits:
            raise ValueError("pattern carrier is wider than output carrier")
        start = fragment.packing.bit_offset(j)
        remaining = fragment.packing.element_bits
        consumed = 0
        while remaining:
            word, shift = divmod(start, fragment.carrier_bits)
            take = min(remaining, fragment.carrier_bits - shift)
            part = b.lshr(pattern, constant(consumed)) if consumed else pattern
            if take < remaining:
                part = b.land(part, constant((1 << take) - 1))
            words[word] = b.lor(words[word], b.shl(part, constant(shift)))
            start += take
            consumed += take
            remaining -= take
    return words
