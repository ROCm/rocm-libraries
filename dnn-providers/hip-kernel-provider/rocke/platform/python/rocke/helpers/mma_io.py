# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor-driven packed transport, expanded before IR serialization."""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from ..core.dtypes import dtype_info
from ..core.ir import (
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    Type,
    Value,
    VectorType,
    dtype_to_ir_type,
)
from ..core.storage import FragmentPacking, MatrixFragmentLayout, TensorStorage


def storage_ir_type(dtype: str) -> Type:
    """Addressable storage unit, distinct from the logical dtype resolver."""
    info = dtype_info(dtype)
    if info.encoded_bits % 8 or info.name in ("e8m0", "e4m3", "e5m3"):
        return I8
    return dtype_to_ir_type(info.name)


def load_matrix_fragment(
    b: IRBuilder,
    ptr: Value,
    row_base: Value,
    lane_group: Value,
    k0: int,
    *,
    storage: TensorStorage,
    layout: MatrixFragmentLayout,
    carrier_type: Type = I32,
) -> Value:
    """Load whole-byte chunks directly into the atom's carrier vector.

    row_base counts pointer storage units. lane_group is derived from the atom's
    lane mapping by the caller. Partial chunks and nonzero bit origins reject.
    """
    packing = layout.fragment
    if not (0 < packing.count <= 0x7FFFFFFF and packing.carrier_count <= 0x7FFFFFFF):
        raise ValueError("invalid matrix fragment chunk layout")
    unit_type = storage_ir_type(storage.dtype)
    unit_bytes = dtype_info(unit_type.name).encoded_bits // 8
    if ptr.type.pointee != unit_type:
        raise ValueError("matrix pointer storage type mismatch")
    if storage.packing != packing.packing or storage.base_bit_offset:
        raise ValueError(
            "matrix fragment requires matching packing and a zero bit origin"
        )
    span = layout.chunk_elements * layout.lane_groups * layout.chunks_per_lane
    if k0 < 0 or k0 + span > storage.shape[1]:
        raise ValueError("matrix fragment exceeds the packed row")
    origin_bits = storage.packing.bit_offset(k0)
    if origin_bits % (8 * unit_bytes) or layout.chunk_bytes % unit_bytes:
        raise ValueError("matrix fragment is not aligned to pointer storage units")
    if dtype_info(carrier_type.name).encoded_bits != packing.carrier_bits:
        raise ValueError("matrix carrier type width mismatch")
    if packing.payload_bits % packing.carrier_bits:
        raise ValueError("matrix fragment payload must occupy whole carriers")
    chunk_units = layout.chunk_bytes // unit_bytes
    origin_bytes = origin_bits // 8
    if (
        origin_bytes // unit_bytes > 0x7FFFFFFF
        or chunk_units * layout.lane_groups > 0x7FFFFFFF
    ):
        raise ValueError("matrix fragment offset exceeds i32 range")
    alignment = gcd(
        storage.alignment_bytes,
        storage.row_stride_bytes,
        layout.chunk_bytes,
        origin_bytes,
    )
    lane_chunk = b.mul(lane_group, b.const_i32(chunk_units))
    step_base = b.add(row_base, b.const_i32(origin_bytes // unit_bytes))
    chunks = []
    for j in range(layout.chunks_per_lane):
        offset = b.add(
            b.add(step_base, b.const_i32(j * layout.lane_groups * chunk_units)),
            lane_chunk,
        )
        remaining = chunk_units
        consumed = 0
        max_width = 8 if unit_bytes == 4 else 16
        while remaining:
            width = min(max_width, 1 << (remaining.bit_length() - 1))
            at = b.add(offset, b.const_i32(consumed)) if consumed else offset
            load_align = gcd(alignment, consumed * unit_bytes)
            if width == 1:
                value = b.vector_splat(
                    b.global_load(ptr, at, unit_type, align=load_align), 1
                )
            else:
                value = b.global_load_vN(ptr, at, unit_type, width, align=load_align)
            chunks.append(value)
            consumed += width
            remaining -= width
    payload = chunks[0]
    for chunk in chunks[1:]:
        payload = b.vec_concat(payload, chunk)
    payload_type = VectorType(carrier_type, packing.live_carriers)
    if payload.type != payload_type:
        payload = b.bitcast(payload, payload_type)
    padding = packing.carrier_count - packing.live_carriers
    if padding:
        if carrier_type != I32:
            raise ValueError("padded matrix fragments currently require i32 carriers")
        payload = b.vec_concat(payload, b.vector_splat(b.const_i32(0), padding))
    return payload


def pack_fragment_bits(
    b: IRBuilder,
    load_bits: Callable[[int], Value],
    fragment: FragmentPacking,
) -> list[Value]:
    """Pack unsigned encoded patterns into integer carriers, including split fields.

    The loader must return canonical patterns with zero high bits. Tensor decoding
    and numeric quantization are separate operations.
    """
    if fragment.carrier_bits not in (32, 64):
        raise ValueError("IR pattern packing currently requires i32 or i64 carriers")
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
