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

from ..arch.target import integer_mma_op_ids
from ..dtypes import DTypeCategory, dtype_info, dtype_names
from ..ir import (
    BF16,
    F16,
    F32,
    I1,
    I16,
    I32,
    I64,
    Type,
    VectorType,
    _is_unsigned,
    dtype_to_ir_type,
    type_bits,
)
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
class Signedness:
    """The ``i1`` flag an integer WMMA takes before matrix operand ``operand``.

    The flag is 1 when that operand's dtype in the selected catalog row is a
    signed integer and 0 when it is an unsigned one (the IR's own unsigned
    test, so a new unsigned type needs no entry here). It is the
    only place an integer WMMA flag is produced: the flag follows the row, so an
    instruction has one row (and one op_id) per signedness combination and no
    table entry or handler writes a flag literal. ``immarg`` changes only the
    ``declare`` text, as for :class:`Imm`.
    """

    operand: MatrixOperand
    immarg: bool = True

    def __post_init__(self) -> None:
        if self.operand not in (MatrixOperand.A, MatrixOperand.B):
            raise ValueError(
                f"a signedness flag precedes the A or B operand, got {self.operand.name}"
            )

    def flag(self, catalog_op) -> int:
        """1 or 0 for this operand of ``catalog_op`` (an :class:`MmaOp`)."""
        dtype = (
            catalog_op.a_dtype
            if self.operand is MatrixOperand.A
            else catalog_op.b_dtype
        )
        if dtype_info(dtype).category is not DTypeCategory.INTEGER:
            raise ValueError(
                f"tile.{catalog_op.op_id} operand {self.operand.value} is {dtype}; "
                f"a signedness flag is derived from an integer operand "
                f"({', '.join(dtype_names(DTypeCategory.INTEGER))})"
            )
        return 0 if _is_unsigned(dtype_to_ir_type(dtype)) else 1

    def argument(self, catalog_op) -> str:
        """The call-site text for ``catalog_op``, such as ``i1 1``."""
        return f"i1 {self.flag(catalog_op)}"

    def parameter(self) -> str:
        """The declaration text, such as ``i1 immarg``."""
        return "i1 immarg" if self.immarg else "i1"


@dataclass(frozen=True)
class MatrixCall:
    """One MMA instruction's intrinsic call, described as data.

    ``intrinsic`` is the full LLVM name including overload suffixes
    (``llvm.amdgcn....``). ``args`` lists the call's arguments in order: each of
    :class:`MatrixOperand` A, B and C exactly once, and any :class:`Imm` or
    :class:`Signedness`.
    ``decl_key`` names the declaration-table entry; it defaults to the
    intrinsic. A ``decl_key`` that names a hand-written entry adds nothing to
    the table, and the hand-written text must equal :meth:`declaration`.
    """

    intrinsic: str
    args: tuple[MatrixOperand | Imm | Signedness, ...]
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
                if isinstance(arg, (Imm, Signedness))
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


def _wmma(
    intrinsic: str,
    operand: Type,
    accumulator: Type,
    args: tuple[MatrixOperand | Imm | Signedness, ...] = (
        MatrixOperand.A,
        MatrixOperand.B,
        MatrixOperand.C,
    ),
    decl_key: str | None = None,
) -> MatrixCall:
    """A gfx11 or gfx12 WMMA: A and B in one register type, the accumulator and
    the result in another, the arguments in ``args`` order."""
    return MatrixCall(
        intrinsic=f"llvm.amdgcn.{intrinsic}",
        args=args,
        a=AsIs(operand),
        b=AsIs(operand),
        c=AsIs(accumulator),
        result=AsIs(accumulator),
        decl_key=decl_key,
    )


def _integer_wmma(op_id: str, bits: int, call: MatrixCall) -> dict[str, MatrixCall]:
    """``call`` under each op_id of one integer WMMA, one per operand
    signedness (:func:`~rocke.core.arch.target.integer_mma_op_ids`). The rows
    share one call; its :class:`Signedness` arguments read each row's operand
    dtypes."""
    return {row: call for row in integer_mma_op_ids(op_id, bits)}


# Integer WMMA: (signed A, A, signed B, B, C, clamp). Each flag comes from its
# operand's dtype in the catalog row; the result is not clamped.
_INTEGER_WMMA_ARGS = (
    Signedness(MatrixOperand.A),
    MatrixOperand.A,
    Signedness(MatrixOperand.B),
    MatrixOperand.B,
    MatrixOperand.C,
    Imm(I1, 0),
)
# The same arguments for the gfx11 integer WMMAs, whose hand-written
# declarations spell the three i1 parameters without immarg.
_GFX11_INTEGER_WMMA_ARGS = (
    Signedness(MatrixOperand.A, immarg=False),
    MatrixOperand.A,
    Signedness(MatrixOperand.B, immarg=False),
    MatrixOperand.B,
    MatrixOperand.C,
    Imm(I1, 0, immarg=False),
)

# gfx12 16-bit-output WMMA: (A, B, C, op_sel). Eight results fill the
# <8 x half> / <8 x i16> register, so op_sel is 0, as the gfx12 builtin
# (which has no op_sel argument) always passes.
_GFX12_16BIT_OUT_ARGS = (
    MatrixOperand.A,
    MatrixOperand.B,
    MatrixOperand.C,
    Imm(I1, 0),
)


def _gfx11_16bit_out_wmma(intrinsic: str, register: VectorType) -> MatrixCall:
    """A gfx11 16-bit-output WMMA: ``(A, B, C, op_sel)`` with A, B, C and the
    result all in the ``<16 x 16-bit>`` ``register``.

    The instruction keeps its eight results per lane in the even 16-bit halves
    of that register when op_sel is 0, so the accumulator and the result are
    the eight logical values (:class:`EvenHalves`) and op_sel is always 0.
    """
    return MatrixCall(
        intrinsic=f"llvm.amdgcn.{intrinsic}",
        args=(MatrixOperand.A, MatrixOperand.B, MatrixOperand.C, Imm(I1, 0)),
        a=AsIs(register),
        b=AsIs(register),
        c=EvenHalves(register),
        result=EvenHalves(register),
    )


# gfx1250 bf16 x bf16 with an fp32 accumulator and a bf16 result:
# (neg A, A, neg B, B, C modifier, C, reuse A, reuse B), every modifier 0. The
# result type differs from the accumulator's, so the call returns <8 x bfloat>
# while C passes as <8 x float>.
_GFX1250_BF16F32 = MatrixCall(
    intrinsic="llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16.v8f32",
    args=(
        Imm(I1, 0),
        MatrixOperand.A,
        Imm(I1, 0),
        MatrixOperand.B,
        Imm(I16, 0),
        MatrixOperand.C,
        Imm(I1, 0),
        Imm(I1, 0),
    ),
    a=AsIs(VectorType(BF16, 16)),
    b=AsIs(VectorType(BF16, 16)),
    c=AsIs(VectorType(F32, 8)),
    result=AsIs(VectorType(BF16, 8)),
)

# Eight fp8 / bf8 values: two words at LLVM 20, one i64 from LLVM 22 on (the
# same register types the hand-written fp8 x fp8 and bf8 x bf8 MFMAs use).
_FP8_X8 = FlavorType(VectorType(I32, 2), ((LLVM_FLAVOR_LLVM22, I64),))
_F32_X4, _F32_X16 = VectorType(F32, 4), VectorType(F32, 16)
_I32_X4, _I32_X16 = VectorType(I32, 4), VectorType(I32, 16)
_I32_X2, _I32_X8, _F32_X8 = VectorType(I32, 2), VectorType(I32, 8), VectorType(F32, 8)

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
        # gfx11 i8 -> i32: sixteen i8 per lane in four words; i4 -> i32:
        # sixteen i4 per lane in two words.
        **_integer_wmma(
            "wmma_i32_16x16x16_iu8",
            8,
            _wmma(
                "wmma.i32.16x16x16.iu8.v8i32.v4i32",
                _I32_X4,
                _I32_X8,
                _GFX11_INTEGER_WMMA_ARGS,
                "wmma.i32.16x16x16.iu8",
            ),
        ),
        **_integer_wmma(
            "wmma_i32_16x16x16_iu4",
            4,
            _wmma(
                "wmma.i32.16x16x16.iu4.v8i32.v2i32",
                _I32_X2,
                _I32_X8,
                _GFX11_INTEGER_WMMA_ARGS,
                "wmma.i32.16x16x16.iu4",
            ),
        ),
        # gfx11 f16 -> f16 and bf16 -> bf16: sixteen 16-bit values of A and of
        # B per lane, eight logical results in the even halves of the 16-slot
        # register; the bf16 form takes and returns them as i16.
        "wmma_f16_16x16x16_f16": _gfx11_16bit_out_wmma(
            "wmma.f16.16x16x16.f16.v16f16.v16f16", VectorType(F16, 16)
        ),
        "wmma_bf16_16x16x16_bf16": _gfx11_16bit_out_wmma(
            "wmma.bf16.16x16x16.bf16.v16i16.v16i16", VectorType(I16, 16)
        ),
        # gfx12 fp8 / bf8 -> f32: eight 8-bit values per lane in two words.
        "wmma_gfx12_f32_16x16x16_fp8_fp8": _wmma(
            "wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32", _I32_X2, _F32_X8
        ),
        "wmma_gfx12_f32_16x16x16_fp8_bf8": _wmma(
            "wmma.f32.16x16x16.fp8.bf8.v8f32.v2i32", _I32_X2, _F32_X8
        ),
        "wmma_gfx12_f32_16x16x16_bf8_fp8": _wmma(
            "wmma.f32.16x16x16.bf8.fp8.v8f32.v2i32", _I32_X2, _F32_X8
        ),
        "wmma_gfx12_f32_16x16x16_bf8_bf8": _wmma(
            "wmma.f32.16x16x16.bf8.bf8.v8f32.v2i32", _I32_X2, _F32_X8
        ),
        # gfx12 i8 -> i32: eight i8 per lane in two words.
        **_integer_wmma(
            "wmma_gfx12_i32_16x16x16_iu8",
            8,
            _wmma(
                "wmma.i32.16x16x16.iu8.v8i32.v2i32",
                _I32_X2,
                _I32_X8,
                _INTEGER_WMMA_ARGS,
            ),
        ),
        # gfx12 i4 -> i32: eight i4 per lane in one word (K 16), sixteen in
        # two words (K 32).
        **_integer_wmma(
            "wmma_gfx12_i32_16x16x16_iu4",
            4,
            _wmma("wmma.i32.16x16x16.iu4.v8i32.i32", I32, _I32_X8, _INTEGER_WMMA_ARGS),
        ),
        **_integer_wmma(
            "wmma_gfx12_i32_16x16x32_iu4",
            4,
            _wmma(
                "wmma.i32.16x16x32.iu4.v8i32.v2i32",
                _I32_X2,
                _I32_X8,
                _INTEGER_WMMA_ARGS,
            ),
        ),
        # gfx12 f16 -> f16 and bf16 -> bf16: eight 16-bit values per lane; the
        # bf16 form takes and returns them as <8 x i16>.
        "wmma_gfx12_f16_16x16x16_f16": _wmma(
            "wmma.f16.16x16x16.f16.v8f16.v8f16",
            VectorType(F16, 8),
            VectorType(F16, 8),
            _GFX12_16BIT_OUT_ARGS,
        ),
        "wmma_gfx12_bf16_16x16x16_bf16": _wmma(
            "wmma.bf16.16x16x16.bf16.v8i16.v8i16",
            VectorType(I16, 8),
            VectorType(I16, 8),
            _GFX12_16BIT_OUT_ARGS,
        ),
        # gfx1250 bf16 -> bf16 through an fp32 accumulator: sixteen bf16 of A
        # and of B per lane, eight fp32 accumulator values, eight bf16 results.
        "wmma_gfx1250_bf16f32_16x16x32_bf16": _GFX1250_BF16F32,
    }
)
