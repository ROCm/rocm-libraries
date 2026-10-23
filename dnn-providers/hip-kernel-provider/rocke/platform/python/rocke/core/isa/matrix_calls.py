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

from ..ir import F32, I16, I32, I64, Type, VectorType, type_bits
from ..lower_llvm import LLVM_FLAVOR_LLVM22
from .matrix_operand_forms import (
    AsIs,
    EvenHalves,
    FlavorType,
    MatrixOperandForm,
    MatrixResultForm,
    RegisterType,
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


def _cdna_mfma(intrinsic: str, operand: RegisterType, accumulator: Type) -> MatrixCall:
    """A dense CDNA MFMA: ``(A, B, C, cbsz, abid, blgp)`` with cbsz, abid and
    blgp all 0; A and B share one register type and the result takes the
    accumulator's."""
    return MatrixCall(
        intrinsic=f"llvm.amdgcn.{intrinsic}",
        args=(
            MatrixOperand.A,
            MatrixOperand.B,
            MatrixOperand.C,
            Imm(I32, 0),
            Imm(I32, 0),
            Imm(I32, 0),
        ),
        a=AsIs(operand),
        b=AsIs(operand),
        c=AsIs(accumulator),
        result=AsIs(accumulator),
    )


# Eight fp8 / bf8 values: two words at LLVM 20, one i64 from LLVM 22 on (the
# same register types the hand-written fp8 x fp8 and bf8 x bf8 MFMAs use).
_FP8_X8 = FlavorType(VectorType(I32, 2), ((LLVM_FLAVOR_LLVM22, I64),))
_F32_X4, _F32_X16 = VectorType(F32, 4), VectorType(F32, 16)
_I32_X4, _I32_X16 = VectorType(I32, 4), VectorType(I32, 16)

# op_id -> call, for every MMA instruction lowered from data rather than by a
# hand-written handler.
MATRIX_CALLS: Mapping[str, MatrixCall] = MappingProxyType(
    {
        # CDNA i8 -> i32: four i8 per lane in one word (CDNA1-2), eight in an
        # i64 (CDNA3), sixteen in four words (CDNA4).
        "mfma_i32_16x16x16_i8": _cdna_mfma("mfma.i32.16x16x16i8", I32, _I32_X4),
        "mfma_i32_32x32x8_i8": _cdna_mfma("mfma.i32.32x32x8i8", I32, _I32_X16),
        "mfma_i32_16x16x32_i8": _cdna_mfma("mfma.i32.16x16x32.i8", I64, _I32_X4),
        "mfma_i32_32x32x16_i8": _cdna_mfma("mfma.i32.32x32x16.i8", I64, _I32_X16),
        "mfma_i32_16x16x64_i8": _cdna_mfma("mfma.i32.16x16x64.i8", _I32_X4, _I32_X4),
        "mfma_i32_32x32x32_i8": _cdna_mfma("mfma.i32.32x32x32.i8", _I32_X4, _I32_X16),
        # CDNA bf16 without the _1k suffix: two bf16 per lane as <2 x i16>.
        "mfma_f32_16x16x8_bf16": _cdna_mfma(
            "mfma.f32.16x16x8bf16", VectorType(I16, 2), _F32_X4
        ),
        "mfma_f32_32x32x4_bf16": _cdna_mfma(
            "mfma.f32.32x32x4bf16", VectorType(I16, 2), _F32_X16
        ),
        # CDNA3+ mixed fp8 x bf8 and bf8 x fp8.
        "mfma_f32_16x16x32_fp8_bf8": _cdna_mfma(
            "mfma.f32.16x16x32.fp8.bf8", _FP8_X8, _F32_X4
        ),
        "mfma_f32_16x16x32_bf8_fp8": _cdna_mfma(
            "mfma.f32.16x16x32.bf8.fp8", _FP8_X8, _F32_X4
        ),
        "mfma_f32_32x32x16_fp8_bf8": _cdna_mfma(
            "mfma.f32.32x32x16.fp8.bf8", _FP8_X8, _F32_X16
        ),
        "mfma_f32_32x32x16_bf8_fp8": _cdna_mfma(
            "mfma.f32.32x32x16.bf8.fp8", _FP8_X8, _F32_X16
        ),
    }
)
