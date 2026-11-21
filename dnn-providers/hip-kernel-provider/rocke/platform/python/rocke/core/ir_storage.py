# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""IR types for logical tensor storage, independent of transport and instructions.

Memory pointers address bytes for sub-byte formats. Registers and LDS hold
those formats densely in whole i32 words. Neither rule adds instruction-ABI
padding; that is the selected instruction's operand contract.
"""

from __future__ import annotations

from .dtypes import dtype_info
from .ir import I8, I32, Type, VectorType, dtype_to_ir_type
from .storage import register_packing, storage_dtype

__all__ = ["storage_ir_type", "storage_element_type", "storage_vector_type"]


def storage_ir_type(dtype: str) -> Type:
    """Addressable pointer unit: bytes for sub-byte formats, storage elements otherwise."""
    info = dtype_info(dtype)
    if info.encoded_bits % 8:
        return I8
    return dtype_to_ir_type(storage_dtype(info.name))


def storage_element_type(dtype: Type) -> Type:
    """Scalar IR type holding one storage unit in registers and LDS."""
    return dtype_to_ir_type(storage_dtype(dtype.name))


def storage_vector_type(dtype: Type, count: int) -> VectorType:
    """Live register storage for ``count`` logical elements, with no ABI padding.

    Packed elements must fill whole i32 words; a partly filled word raises.
    Other dtypes are held one-to-one in their storage element type.
    """
    packing = register_packing(dtype.name, count)
    if packing is not None:
        return VectorType(I32, packing.carrier_count)
    return VectorType(storage_element_type(dtype), count)
