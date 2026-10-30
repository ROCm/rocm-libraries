# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Matrix intrinsic calls described as data.

One :class:`MatrixCall` says everything the lowering of one MMA instruction
needs: the intrinsic, the order of its arguments (the A, B and C operands and
the literal immediates), the form each operand takes, the form of the result,
and the declaration-table key. :data:`MATRIX_CALLS` maps an op_id to its call.

The declaration is derived from the same forms the call is rendered from
(:meth:`MatrixCall.declaration`), so a call and its ``declare`` line cannot
disagree at any LLVM flavor. ``_decls_for_flavor`` adds each table entry's
declaration after the hand-written ones.

The backends look an op_id up here before their hand-written handlers; an op_id
has one or the other, never both. The C++ engine holds the same table and
derivation in ``cpp/core/isa/matrix_calls.cpp``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ..ir import Type, type_bits
from .matrix_operand_forms import (
    AsIs,
    EvenHalves,
    MatrixOperandForm,
    MatrixResultForm,
)

# The integer types an immediate may have.
_IMMEDIATE_TYPES = ("i1", "i16", "i32", "i64")


class MatrixOperand(Enum):
    """Which MMA operand a call argument passes."""

    A = "a"
    B = "b"
    C = "c"


@dataclass(frozen=True)
class Imm:
    """One literal call argument, such as a format code or an ``i1`` flag.

    ``immarg`` records whether the intrinsic declares the parameter ``immarg``
    (a compile-time constant); it changes only the ``declare`` text.
    """

    llvm_type: Type
    value: int
    immarg: bool = True

    def __post_init__(self) -> None:
        if self.llvm_type.name not in _IMMEDIATE_TYPES:
            raise TypeError(
                f"an immediate is one of {', '.join(_IMMEDIATE_TYPES)}, "
                f"got {self.llvm_type.name}"
            )
        bits = type_bits(self.llvm_type)
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError(f"an immediate value is an int, got {self.value!r}")
        if not -(1 << (bits - 1)) <= self.value < (1 << bits):
            raise ValueError(
                f"{self.value} does not fit an {self.llvm_type.name} immediate"
            )

    def argument(self) -> str:
        """The call-site text, such as ``i32 0``."""
        return f"{self.llvm_type.name} {self.value}"

    def parameter(self) -> str:
        """The declaration text, such as ``i32 immarg``."""
        return f"{self.llvm_type.name} immarg" if self.immarg else self.llvm_type.name


@dataclass(frozen=True)
class MatrixCall:
    """One MMA instruction's intrinsic call, described as data.

    ``intrinsic`` is the full LLVM name including overload suffixes
    (``llvm.amdgcn....``). ``args`` lists the call's arguments in order: each of
    :class:`MatrixOperand` A, B and C exactly once, and any :class:`Imm`.
    ``decl_key`` names the declaration-table entry; it defaults to the
    intrinsic. A ``decl_key`` that names a hand-written entry adds nothing to
    the table, and the hand-written text must equal :meth:`declaration`.
    """

    intrinsic: str
    args: tuple[MatrixOperand | Imm, ...]
    a: MatrixOperandForm
    b: MatrixOperandForm
    c: MatrixOperandForm
    result: MatrixResultForm
    decl_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        operands = [arg for arg in self.args if isinstance(arg, MatrixOperand)]
        if sorted(operand.value for operand in operands) != ["a", "b", "c"]:
            raise ValueError(
                f"MatrixCall {self.intrinsic!r} must pass each of A, B and C exactly "
                f"once, got {[operand.name for operand in operands]}"
            )
        if not isinstance(self.result, (AsIs, EvenHalves)):
            raise TypeError(
                f"MatrixCall {self.intrinsic!r} result form must be AsIs or "
                f"EvenHalves, got {type(self.result).__name__}"
            )

    @property
    def key(self) -> str:
        """The declaration-table key this call needs."""
        return self.decl_key if self.decl_key is not None else self.intrinsic

    def form(self, operand: MatrixOperand) -> MatrixOperandForm:
        """The form operand ``operand`` takes."""
        return {
            MatrixOperand.A: self.a,
            MatrixOperand.B: self.b,
            MatrixOperand.C: self.c,
        }[operand]

    def declaration(self, flavor: str) -> str:
        """The ``declare`` line at LLVM ``flavor``, derived from the forms."""
        from ..lower_llvm import _llvm_type

        parameters = [
            (
                arg.parameter()
                if isinstance(arg, Imm)
                else _llvm_type(self.form(arg).register_type_at(flavor))
            )
            for arg in self.args
        ]
        returned = _llvm_type(self.result.register_type_at(flavor))
        return f"declare {returned} @{self.intrinsic}({', '.join(parameters)})"


# op_id -> call, for every MMA instruction lowered from data rather than by a
# hand-written handler.
MATRIX_CALLS: Mapping[str, MatrixCall] = MappingProxyType({})
