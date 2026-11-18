# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Internal global storage-load policy shared by fragment verbs and compatibility I/O.

The descriptor layer owns coordinates, bounds, and alignment guarantees. This
module only splits one complete storage run into supported vector loads. It is
not re-exported as a client-facing tiling verb.
"""

from __future__ import annotations

from math import gcd

from ...core.dtypes import dtype_info
from ...core.ir import I32, I64, IRBuilder, Type, Value

__all__ = ["load_storage_chunks"]


def load_storage_chunks(
    b: IRBuilder,
    ptr: Value,
    offset: Value,
    element: Type,
    count: int,
    *,
    alignment_bytes: int,
    byte_addressed: bool = False,
    word_vectors: bool = False,
) -> list[Value]:
    """Load a complete storage run, returning pieces in address order.

    ``count`` counts elements of ``element``. ``offset`` counts those units,
    except ``byte_addressed=True`` uses a signed byte offset for i32 word loads
    from a byte pointer. The guarantee is at the run's starting address and is
    reduced for each piece. No clipping or numerical conversion happens here.
    Concatenation remains with the caller so independent runs can be collected
    before register assembly.
    """
    if count <= 0:
        raise ValueError("storage load count must be positive")
    if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError("alignment_bytes must be a positive power of two")
    if byte_addressed and element != I32:
        raise ValueError("byte-addressed storage loads require i32 elements")
    element_bytes = dtype_info(element.name).encoded_bits // 8
    if byte_addressed and (count - 1) * element_bytes > 0x7FFFFFFF:
        raise ValueError("storage load displacement exceeds i32 range")
    max_width = 4 if byte_addressed else 8 if element_bytes == 4 else 16
    remaining, consumed = count, 0
    pieces = []
    while remaining:
        width = (
            min(4, remaining)
            if byte_addressed or word_vectors
            else min(max_width, 1 << (remaining.bit_length() - 1))
        )
        displacement = consumed * element_bytes if byte_addressed else consumed
        at = b.add(offset, b.const_i32(displacement)) if consumed else offset
        align = gcd(alignment_bytes, consumed * element_bytes)
        load_ptr = b.global_ptr_add(ptr, b.sext(at, I64)) if byte_addressed else ptr
        load_at = b.const_i32(0) if byte_addressed else at
        if width == 1:
            value = b.vector_splat(
                b.global_load(load_ptr, load_at, element, align=align), 1
            )
        else:
            value = b.global_load_vN(load_ptr, load_at, element, width, align=align)
        pieces.append(value)
        consumed += width
        remaining -= width
    return pieces
