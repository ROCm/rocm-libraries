# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How an MMA operand reaches the instruction's register type.

An MMA instruction takes each operand -- A, B and the accumulator C -- as one
fixed register type, such as ``f32`` for the A and B of the f32 MFMAs or
``vec<i32x4>`` for those of the gfx11 iu8 WMMA. A kernel
may hold the same bits in another shape: a one-element ``vec<f32x1>``, or the
instruction's packed values as ``vec<i8x16>``. Reading such an operand as the
register type is a reinterpretation of the same bits, never a conversion.

:func:`mma_operand_needs_reinterpret` is the rule; it has no output syntax.
Each lowerer renders its answer in its own language (LLVM ``bitcast``, HIP
``__builtin_bit_cast``), so both lowerers of both engines accept and refuse the
same operands with the same text. :func:`mma_result_needs_reinterpret` is the
same rule for the value an instruction returns; the LLVM lowering of the calls
in :data:`~rocke.core.isa.matrix_calls.MATRIX_CALLS` applies it, in both
engines. The C++ engine's twins are ``rocke_mma_operand_needs_reinterpret``
and ``rocke_mma_result_needs_reinterpret`` in ``cpp/core/mma_operand.cpp``.
"""

from __future__ import annotations

from typing import Literal

from .arch import ArchTarget
from .ir import (
    Op,
    Type,
    Value,
    VectorType,
    dtype_to_ir_type,
    element_type,
    is_packed_only,
    type_bits,
)

MmaOperandRole = Literal["a", "b", "c"]

# Position of each operand among an MMA op's operands (A, B, C).
_OPERAND_INDEX = {"a": 0, "b": 1, "c": 2}


def mma_operand_value(op: Op, role: MmaOperandRole) -> Value:
    """MMA operand ``role`` ("a", "b" or the accumulator "c") of ``op``."""
    return op.operands[_OPERAND_INDEX[role]]


def _admitted_elements(
    register_type: Type, catalog_dtype: str | None
) -> frozenset[Type]:
    """The element types a value may hold in another shape than ``register_type``.

    They are the register's own element type (a reshape, such as ``vec<f32x1>``
    for ``f32``) and the element type of ``catalog_dtype``, the dtype the MMA
    catalog row records for the value (the instruction's packed values, such
    as ``vec<i8x16>`` for ``vec<i32x4>``). ``None`` (an op_id the catalog does
    not list) admits only the first, and so does a catalog element that is
    packed-only (``i4`` for the iu4 instructions): no value has that element
    type, so its values arrive in the register type.
    """
    elements = {element_type(register_type)}
    if catalog_dtype is not None:
        catalog_element = dtype_to_ir_type(catalog_dtype)
        if not is_packed_only(catalog_element):
            elements.add(catalog_element)
    return frozenset(elements)


def _require_integer_vector_count(label: str, kind: str, ty: Type) -> None:
    """Check counts before equality can admit a numerically equal float or bool."""
    if isinstance(ty, VectorType) and type(ty.count) is not int:
        raise ValueError(
            f"{label} requires an integer {kind} vector count, got {ty.count!r}"
        )


def _require_reshape(
    label: str,
    verb: str,
    value_type: Type,
    register_type: Type,
    admitted: frozenset[Type],
    *,
    format_hint: str,
    width_hint: str,
) -> None:
    """Refuse ``value_type`` unless it is ``register_type``'s bits in another shape.

    It must hold an ``admitted`` element type at the register's bit width.
    ``label`` names the value, ``verb`` says what the instruction does with the
    register ("takes" or "returns"), and each hint ends its refusal.
    """
    element = element_type(value_type)
    if element not in admitted:
        names = " or ".join(sorted(t.name for t in admitted))
        raise ValueError(
            f"{label} holds {element.name} values; the instruction {verb} {names}"
            f"{format_hint}"
        )
    value_bits, register_bits = type_bits(value_type), type_bits(register_type)
    if value_bits != register_bits:
        raise ValueError(
            f"{label} is {value_type.name} ({value_bits} bits); the instruction "
            f"{verb} {register_type.name} ({register_bits} bits){width_hint}"
        )


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
    :func:`_admitted_elements`, with the catalog's dtype for ``role``) and has
    the register's bit width.

    Raises ``ValueError`` for an operand holding another number format (its
    values would have to be converted, which is the kernel's decision, not the
    lowerer's), an operand of another bit width, or a vector type whose count
    is not an integer.
    """
    label = f"MMA operand {role} of {op_id!r}"
    for kind, ty in (("operand", operand_type), ("register", register_type)):
        _require_integer_vector_count(label, kind, ty)
    if operand_type == register_type:
        return False
    catalog_op = arch.mma.by_op_id(op_id)
    catalog_dtype = (
        None if catalog_op is None else catalog_op.srcs[_OPERAND_INDEX[role]].dtype
    )
    register_bits = type_bits(register_type)
    _require_reshape(
        label,
        "takes",
        operand_type,
        register_type,
        _admitted_elements(register_type, catalog_dtype),
        format_hint=" -- convert the values first",
        width_hint=f" -- pass a {register_bits}-bit value",
    )
    return True


def mma_result_needs_reinterpret(
    op_id: str, result_type: Type, register_type: Type, arch: ArchTarget
) -> bool:
    """Whether the result of MMA ``op_id`` needs a register reinterpretation.

    Result admission uses ``dst`` metadata independently of the accumulator
    source. The register's or catalog result's element type must occupy the
    register's bit width; reinterpretation never converts number formats.
    """
    label = f"MMA result of {op_id!r}"
    for kind, ty in (("result", result_type), ("register", register_type)):
        _require_integer_vector_count(label, kind, ty)
    if result_type == register_type:
        return False
    catalog_op = arch.mma.by_op_id(op_id)
    catalog_dtype = None if catalog_op is None else catalog_op.dst.dtype
    register_bits = type_bits(register_type)
    _require_reshape(
        label,
        "returns",
        result_type,
        register_type,
        _admitted_elements(register_type, catalog_dtype),
        format_hint=" -- declare the result in an element type it returns",
        width_hint=f" -- declare a {register_bits}-bit result",
    )
    return True
