# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compatibility adapter for caller-addressed raw matrix chunks.

Production consumers use the descriptor-driven fragment verbs. This adapter
preserves existing raw slot/carrier and signed-row contracts while delegating
all actual loads to the shared internal storage transport policy.
"""

from __future__ import annotations

from math import gcd

from ...core.dtypes import dtype_info
from ...core.ir import I32, IRBuilder, Type, Value, VectorType
from ...core.ir_storage import storage_ir_type
from ...core.storage import MatrixFragmentLayout
from .transport import load_storage_chunks

__all__ = ["load_matrix_fragment"]


def load_matrix_fragment(
    b: IRBuilder,
    ptr: Value,
    row_base: Value,
    lane_group: Value,
    k0: int,
    *,
    dtype: str,
    layout: MatrixFragmentLayout,
    carrier_type: Type = I32,
    alignment_bytes: int = 1,
) -> Value:
    """Load complete chunks from a caller-selected row into carrier registers.

    row_base counts pointer storage units; alignment_bytes is the guaranteed
    alignment at that row address, before k0 and lane/chunk offsets. The caller
    owns tensor indexing and allocation bounds, including empty/partial rows.
    lane_group comes from the atom's lane mapping. k0 must address a whole
    pointer unit. New authoring uses load_fragment with a TensorDesc/Window;
    this compatibility entry point is not re-exported by the tiling package.
    """
    if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError("alignment_bytes must be a positive power of two")
    packing = layout.fragment
    if not (0 < packing.count <= 0x7FFFFFFF and packing.carrier_count <= 0x7FFFFFFF):
        raise ValueError("invalid matrix fragment chunk layout")
    unit_type = storage_ir_type(dtype)
    unit_bytes = dtype_info(unit_type.name).encoded_bits // 8
    if ptr.type.pointee != unit_type:
        raise ValueError("matrix pointer storage type mismatch")
    if dtype_info(dtype).encoded_bits != packing.packing.element_bits:
        raise ValueError("matrix dtype and packing width mismatch")
    origin_bits = packing.packing.bit_offset(k0)
    if origin_bits % (8 * unit_bytes) or layout.chunk_bytes % unit_bytes:
        raise ValueError("matrix fragment is not aligned to pointer storage units")
    if dtype_info(carrier_type.name).encoded_bits != packing.carrier_bits:
        raise ValueError("matrix carrier type width mismatch")
    if packing.payload_bits % packing.carrier_bits:
        raise ValueError("matrix fragment payload must occupy whole carriers")
    padding = packing.carrier_count - packing.live_carriers
    if padding and carrier_type != I32:
        raise ValueError("padded matrix fragments currently require i32 carriers")
    chunk_units = layout.chunk_bytes // unit_bytes
    origin_bytes = origin_bits // 8
    # Bound the last loaded unit across all chunks and lane groups, including k0.
    if (
        origin_bytes // unit_bytes > 0x7FFFFFFF
        or chunk_units * layout.lane_groups > 0x7FFFFFFF
        or layout.chunks_per_lane
        > (0x80000000 - origin_bytes // unit_bytes) // chunk_units // layout.lane_groups
    ):
        raise ValueError("matrix fragment offset exceeds i32 range")
    alignment = gcd(alignment_bytes, layout.chunk_bytes, origin_bytes)
    # Use 2--4-word vectors; retain storage-element loads for a one-word tail.
    word_loads = (
        unit_bytes == 1
        and carrier_type == I32
        and layout.chunk_bytes % 4 == 0
        and layout.chunk_bytes % 16 != 4
    )
    load_type = I32 if word_loads else unit_type
    load_step = 4 if word_loads else 1
    lane_chunk = b.mul(lane_group, b.const_i32(chunk_units))
    step_base = b.add(row_base, b.const_i32(origin_bytes // unit_bytes))
    chunks = []
    for j in range(layout.chunks_per_lane):
        offset = b.add(
            b.add(step_base, b.const_i32(j * layout.lane_groups * chunk_units)),
            lane_chunk,
        )
        chunks.extend(
            load_storage_chunks(
                b,
                ptr,
                offset,
                load_type,
                chunk_units // load_step,
                alignment_bytes=alignment,
                byte_addressed=word_loads,
            )
        )
    payload = chunks[0]
    for chunk in chunks[1:]:
        payload = b.vec_concat(payload, chunk)
    payload_type = VectorType(carrier_type, packing.live_carriers)
    if payload.type != payload_type:
        payload = b.bitcast(payload, payload_type)
    if padding:
        payload = b.vec_concat(payload, b.vector_splat(b.const_i32(0), padding))
    return payload
