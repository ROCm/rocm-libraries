# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Internal descriptor-derived byte bases and constant storage displacements.

Only complete, statically bounded global views with an explicit alignment
contract use this path. Literal window origins are separated from runtime
coordinates so repeated transport runs share a byte base. Default, clipped,
negative-stride and unbounded views retain their existing addressing path.
"""

from __future__ import annotations

import math

from ...core.dtypes import dtype_info
from ...core.ir import I32, IRBuilder, PtrType, Value
from ...core.storage import BitPacking, elements_to_storage, is_register_packed
from .descriptors import TensorWindow
from .fragments import TileDesc

__all__ = ["can_rebase_global", "split_window", "byte_address", "storage_displacement"]


def can_rebase_global(
    ptr: Value,
    window: TensorWindow,
    tile: TileDesc,
    alignment_bytes: int | None,
    in_bounds: bool,
) -> bool:
    """Whether a descriptor proves a bounded byte base without narrowing any index."""
    if alignment_bytes is None or not in_bounds or not isinstance(ptr.type, PtrType):
        return False
    if ptr.type.space != "global":
        return False
    tensor = window.tensor
    if any(type(n) is not int for n in (*tensor.lengths, *tensor.strides)):
        return False
    if any(n < 0 for n in tensor.strides):
        return False
    if any(
        isinstance(origin, Value) and origin.type != I32 for origin in window.origin
    ):
        return False
    if any(
        isinstance(origin, int) and not 0 <= origin < length
        for origin, length in zip(window.origin, tensor.lengths)
    ):
        return False
    if len(tile.layout.hierarchical_lengths) != len(tile.shape):
        return False
    if any(
        math.prod(levels) != extent
        for levels, extent in zip(tile.layout.hierarchical_lengths, tile.shape)
    ):
        return False
    span = tensor.element_span
    return (
        span <= 0x7FFFFFFF
        and BitPacking(dtype_info(tensor.dtype.name).encoded_bits).byte_size(span)
        <= 0x7FFFFFFF
    )


def split_window(window: TensorWindow) -> tuple[TensorWindow, int]:
    """Separate literal origin contributions from the runtime-positioned view."""
    constant = sum(
        origin * stride
        for origin, stride in zip(window.origin, window.tensor.strides)
        if isinstance(origin, int)
    )
    origin = tuple(0 if isinstance(value, int) else value for value in window.origin)
    return TensorWindow(window.tensor, origin, window.bounds), constant


def byte_address(b: IRBuilder, window: TensorWindow, positions: list[Value]) -> Value:
    """Byte index directly from descriptor coordinates, with exact packed unit conversion.

    Packed admission already proves whole-word starts. Dividing the coordinate
    or stride before multiplying avoids an overflowing aggregate logical-bit
    index and exposes the same byte-row arithmetic as a direct packed load.
    """
    tensor = window.tensor
    group, group_bytes = BitPacking(dtype_info(tensor.dtype.name).encoded_bits).group(8)
    lead = tensor.rank - len(positions)
    leading = [
        b.const_i32(value) if isinstance(value, int) else value
        for value in window.origin[:lead]
    ]
    ordered = [
        (position, tensor.strides[lead + axis])
        for axis, position in enumerate(positions)
    ]
    ordered += list(zip(leading, tensor.strides[:lead]))
    address = None
    for position, stride in ordered:
        if stride % group == 0:
            factor = (stride // group) * group_bytes
            term = position if factor == 1 else b.mul(position, b.const_i32(factor))
        else:
            if stride != 1:
                raise ValueError("packed byte addressing requires whole-byte strides")
            term = b.div(position, b.const_i32(group))
            if group_bytes != 1:
                term = b.mul(term, b.const_i32(group_bytes))
        address = term if address is None else b.add(address, term)
    return address if address is not None else b.const_i32(0)


def storage_displacement(
    window: TensorWindow,
    tile: TileDesc,
    coordinates: tuple[int, ...],
    constant_origin: int = 0,
) -> int:
    """Constant storage offset from a lane base, with optional literal origin.

    Non-packed displacements are signed, unlike element counts for allocation.
    Packed admission still requires whole-word runs and nonnegative offsets.
    """
    lead = window.tensor.rank - len(tile.shape)
    elements = constant_origin + sum(
        delta * window.tensor.strides[lead + axis]
        for axis, delta in enumerate(coordinates)
    )
    dtype = window.tensor.dtype.name
    return (
        elements_to_storage(dtype, elements) if is_register_packed(dtype) else elements
    )
