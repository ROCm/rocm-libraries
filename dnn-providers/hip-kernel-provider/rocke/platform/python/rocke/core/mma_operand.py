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
same rule for the value the instruction returns. The C++ engine's twins are
``rocke_mma_operand_needs_reinterpret`` and ``rocke_mma_result_needs_reinterpret``
in ``cpp/core/mma_operand.cpp``.
"""

from __future__ import annotations

from typing import Literal

from .arch import ArchTarget
from .ir import Op, Type, Value, VectorType, dtype_to_ir_type, is_packed_only, type_bits

MmaOperandRole = Literal["a", "b", "c"]

# Position of each operand among an MMA op's operands (A, B, C).
_OPERAND_INDEX = {"a": 0, "b": 1, "c": 2}


def mma_operand_value(op: Op, role: MmaOperandRole) -> Value:
    """MMA operand ``role`` ("a", "b" or the accumulator "c") of ``op``."""
    return op.operands[_OPERAND_INDEX[role]]


def _element_type(t: Type) -> Type:
    """The element type of a vector type, or the type itself for a scalar."""
    return t.elem if isinstance(t, VectorType) else t


# The MmaOp field holding each operand role's dtype, and the result's.
_CATALOG_DTYPE_FIELD: dict[MmaOperandRole, str] = {
    "a": "a_dtype",
    "b": "b_dtype",
    "c": "c_dtype",
}
_CATALOG_RESULT_DTYPE_FIELD = "dst_dtype"


def _admitted_elements(
    op_id: str, dtype_field: str, register_type: Type, arch: ArchTarget
) -> frozenset[Type]:
    """The element types a value may hold in another shape than ``register_type``.

    They are the register's own element type (a reshape, such as ``vec<f32x1>``
    for ``f32``) and the element type of field ``dtype_field`` of ``arch``'s
    MMA catalog row for ``op_id`` (the instruction's packed values,
    such as ``vec<i8x16>`` for ``vec<i32x4>``). An op_id the catalog does not
    list admits only the first, and so does a catalog element that is
    packed-only (``i4`` for the iu4 instructions): no value has that element
    type, so its values arrive in the register type.
    """
    elements = {_element_type(register_type)}
    catalog_op = arch.mma.by_op_id(op_id)
    if catalog_op is not None:
        catalog_element = dtype_to_ir_type(getattr(catalog_op, dtype_field))
        if not is_packed_only(catalog_element):
            elements.add(catalog_element)
    return frozenset(elements)


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
    """Refuse ``value_type`` unless it is ``register_type``'s bits in another
    shape: an ``admitted`` element type at the register's bit width. ``verb``
    says what the instruction does with the register ("takes" or "returns");
    each hint ends its refusal."""
    element = _element_type(value_type)
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
    lowerer's) and for an operand of another bit width.
    """
    if operand_type == register_type:
        return False
    _require_reshape(
        f"MMA operand {role} of {op_id!r}",
        "takes",
        operand_type,
        register_type,
        _admitted_elements(op_id, _CATALOG_DTYPE_FIELD[role], register_type, arch),
        format_hint=" -- convert the values first",
        width_hint=f" -- pass a {type_bits(register_type)}-bit value",
    )
    return True


def mma_result_needs_reinterpret(
    op_id: str, result_type: Type, register_type: Type, arch: ArchTarget
) -> bool:
    """Whether the result of MMA ``op_id`` must be reinterpreted from ``register_type``.

    The operand rule (:func:`mma_operand_needs_reinterpret`) applied to the
    value the instruction returns, with the catalog's result dtype
    (``dst_dtype``): False when ``result_type`` is ``register_type``; True when
    it holds the register's element type or the catalog's result element type
    in the register's bit width, such as ``vec<bf16x8>`` for an instruction
    that returns its bf16 results as ``<8 x i16>``.

    Raises ``ValueError`` for a result of another number format, such as
    ``vec<f16x8>`` for an instruction whose catalog row returns bf16 (one
    ``bitcast`` would read bf16 bits as f16 values), and for a result of
    another bit width.
    """
    if result_type == register_type:
        return False
    _require_reshape(
        f"MMA result of {op_id!r}",
        "returns",
        result_type,
        register_type,
        _admitted_elements(op_id, _CATALOG_RESULT_DTYPE_FIELD, register_type, arch),
        format_hint=" -- declare the result with one of those element types",
        width_hint="",
    )
    return True
