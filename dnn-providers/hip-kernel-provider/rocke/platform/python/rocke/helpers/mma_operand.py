# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adapt live logical fragments to atom-owned register ABIs, without loading data."""

from __future__ import annotations

from typing import Literal

from ..core.arch.target import MmaOp
from ..core.arch.mfma_scale import gfx950_scaled_mfma, matrix_layout
from ..core.arch.wmma_scale import gfx1250_scaled_wmma
from ..core.ir import I32, IRBuilder, Value, VectorType
from .tiling.fragments import Fragment

__all__ = ["prepare_scaled_mma_operand"]


def prepare_scaled_mma_operand(
    b: IRBuilder,
    atom: MmaOp,
    role: Literal["a", "b"],
    fragment: Fragment,
) -> Value:
    """Return the selected scaled atom's A/B carrier vector.

    The fragment retains only live storage words. This bridge validates its
    logical format and count, reinterprets byte-sized formats, and appends the
    atom's required zero tail. It emits no memory operations or conversions.
    LLVM may still materialize zero registers; inspect ISA before claiming the
    padding is instruction-free.
    """
    if role not in ("a", "b"):
        raise ValueError(f"scaled MMA operand role={role!r}, expected 'a' or 'b'")
    if not isinstance(fragment, Fragment):
        raise TypeError("scaled MMA operand requires a Fragment")
    if not isinstance(atom, MmaOp):
        raise TypeError("scaled MMA operand requires an MmaOp")
    index = 0 if role == "a" else 1
    if (
        not isinstance(atom.family, str)
        or not isinstance(atom.op_id, str)
        or not isinstance(atom.srcs[index].dtype, str)
    ):
        raise TypeError("scaled MMA operand requires complete atom metadata")
    if atom.family == "mfma_scaled" and gfx950_scaled_mfma(atom.op_id) is not None:
        layout = matrix_layout(atom.srcs[index].dtype)
    elif atom.family == "wmma_scaled":
        operation = gfx1250_scaled_wmma(atom.op_id)
        if operation is None:
            raise ValueError(f"unsupported scaled MMA operand ABI: {atom.op_id}")
        layout = operation.matrix_layout(role)
    else:
        raise ValueError(f"unsupported scaled MMA operand ABI: {atom.op_id}")
    expected_dtype = atom.srcs[index].dtype
    if fragment.dtype.name != expected_dtype:
        raise ValueError(
            f"scaled MMA operand {role} dtype={fragment.dtype.name!r}, "
            f"expected {expected_dtype!r}"
        )
    if fragment.tile_desc.register_count != layout.fragment.count:
        raise ValueError(
            f"scaled MMA operand {role} count={fragment.tile_desc.register_count}, "
            f"expected {layout.fragment.count} logical elements"
        )
    value = fragment.value
    packing = layout.fragment
    payload_type = VectorType(I32, packing.live_carriers)
    if value.type != payload_type:
        value = b.bitcast(value, payload_type)
    padding = packing.carrier_count - packing.live_carriers
    if padding:
        value = b.vec_concat(value, b.vector_splat(b.const_i32(0), padding))
    return value
