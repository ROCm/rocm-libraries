# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How an MMA operand reaches the instruction's register type.

An MMA instruction takes its A and B operands as one fixed register type, such
as ``f32`` for the f32 MFMAs or ``vec<i32x4>`` for the gfx11 iu8 WMMA. A kernel
may hold the same bits in another shape: a one-element ``vec<f32x1>``, or the
instruction's packed values as ``vec<i8x16>``. Reading such an operand as the
register type is a reinterpretation of the same bits, never a conversion.

:func:`mma_operand_needs_reinterpret` is the rule; it has no output syntax.
Each lowerer renders its answer in its own language (LLVM ``bitcast``, HIP
``__builtin_bit_cast``), so both lowerers of both engines accept and refuse the
same operands with the same text. The C++ engine's twin is
``rocke_mma_operand_needs_reinterpret`` in ``cpp/core/mma_operand.cpp``.
"""

from __future__ import annotations

from typing import Literal

from .arch import ArchTarget
from .ir import Op, Type, Value, VectorType, dtype_to_ir_type, is_packed_only, type_bits

MmaOperandRole = Literal["a", "b"]

# Position of each multiplicand among an MMA op's operands (A, B, C).
_OPERAND_INDEX = {"a": 0, "b": 1}


def mma_operand_value(op: Op, role: MmaOperandRole) -> Value:
    """MMA operand ``role`` ("a" or "b") of ``op``."""
    return op.operands[_OPERAND_INDEX[role]]


def _element_type(t: Type) -> Type:
    """The element type of a vector type, or the type itself for a scalar."""
    return t.elem if isinstance(t, VectorType) else t


def _admitted_elements(
    op_id: str, role: MmaOperandRole, register_type: Type, arch: ArchTarget
) -> frozenset[Type]:
    """The element types MMA operand ``role`` may hold in another shape.

    They are the register's own element type (a reshape, such as ``vec<f32x1>``
    for ``f32``) and the element type ``arch``'s MMA catalog records for that
    operand (the instruction's packed values, such as ``vec<i8x16>`` for
    ``vec<i32x4>``). An op_id the catalog does not list admits only the first,
    and so does a catalog element that is packed-only (``i4`` for the iu4
    instructions): no value has that element type, so its operands arrive in
    the register type.
    """
    elements = {_element_type(register_type)}
    catalog_op = arch.mma.by_op_id(op_id)
    if catalog_op is not None:
        dtype = catalog_op.a_dtype if role == "a" else catalog_op.b_dtype
        catalog_element = dtype_to_ir_type(dtype)
        if not is_packed_only(catalog_element):
            elements.add(catalog_element)
    return frozenset(elements)


def mma_operand_needs_reinterpret(
    op_id: str,
    role: MmaOperandRole,
    operand_type: Type,
    register_type: Type,
    arch: ArchTarget,
) -> bool:
    """Whether MMA operand ``role`` of ``op_id`` must be reinterpreted to be passed.

    False when ``operand_type`` already is ``register_type``; the catalog is not
    consulted then, so a target without an MMA catalog still lowers operands of
    the register type. True when one reinterpretation of the same bits gives the
    register type: the operand holds an admitted element type (see
    :func:`_admitted_elements`) and has the register's bit width.

    Raises ``ValueError`` for an operand holding another number format (its
    values would have to be converted, which is the kernel's decision, not the
    lowerer's) and for an operand of another bit width.
    """
    if operand_type == register_type:
        return False
    label = f"MMA operand {role} of {op_id!r}"
    element = _element_type(operand_type)
    admitted = _admitted_elements(op_id, role, register_type, arch)
    if element not in admitted:
        names = " or ".join(sorted(t.name for t in admitted))
        raise ValueError(
            f"{label} holds {element.name} values; the instruction takes {names} "
            "-- convert the values first"
        )
    operand_bits, register_bits = type_bits(operand_type), type_bits(register_type)
    if operand_bits != register_bits:
        raise ValueError(
            f"{label} is {operand_type.name} ({operand_bits} bits); the instruction "
            f"takes {register_type.name} ({register_bits} bits) -- pass a "
            f"{register_bits}-bit value"
        )
    return True
