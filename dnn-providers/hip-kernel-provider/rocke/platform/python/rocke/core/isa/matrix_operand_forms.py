# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How one MMA operand, or the result, meets the register type its intrinsic takes.

A matrix intrinsic takes each operand as one fixed LLVM register type. A kernel
holds the value in the shape the tile layer gave it, which may differ. A *form*
names the register type and the one rule that turns the kernel's value into it:

- :class:`AsIs` -- the value already has the register type, or holds the same
  bits in another shape and is reinterpreted with one ``bitcast``
  (:func:`~rocke.core.mma_operand.mma_operand_needs_reinterpret` is the rule)::

      vec<fp8e4m3x32>  --bitcast-->  <8 x i32>

- :class:`ZeroPadded` -- a packed payload of exactly ``n`` 32-bit words fills
  only part of an ``<m x i32>`` register; the high ``m - n`` words are zero (a
  value already widened to the whole register passes unchanged)::

      <4 x i32> [w0 w1 w2 w3]  -->  <8 x i32> [w0 w1 w2 w3 0 0 0 0]

- :class:`EvenHalves` -- the instruction keeps ``n`` 16-bit results in the even
  half-slots of a ``2n``-slot register; the kernel holds the ``n`` values::

      <8 x half> [c0 .. c7]  -->  <16 x half> [c0 0 c1 0 .. c7 0]   (operand)
      <16 x half> [d0 x d1 x .. d7 x]  -->  <8 x half> [d0 .. d7]   (result)

A register type that changed spelling at a newer LLVM flavor is a
:class:`FlavorType`. :func:`f8f6f4_operand_form` picks :class:`AsIs` or
:class:`ZeroPadded` for an f8f6f4 operand by its matrix format. Forms are data;
each renders itself through the lowerer it is handed. :mod:`rocke.core.isa.matrix_calls` composes them into one call.

The C++ engine's twin is ``cpp/core/lower_llvm/matrix_call.cpp``; both render
the same text and raise the same errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arch.mfma_scale import matrix_operand_packing, require_matrix_format
from ..dtypes import dtype_info
from ..ir import I32, Op, Type, VectorType, type_bits
from ..mma_operand import (
    MmaOperandRole,
    _require_integer_vector_count,
    mma_operand_value,
    mma_operand_needs_reinterpret,
    mma_result_needs_reinterpret,
)


def _llvm_type(t: Type) -> str:
    from ..lower_llvm import _llvm_type as render

    return render(t)


def _op_id(op: Op) -> str:
    from ..lower_llvm import _mma_op_id

    return _mma_op_id(op)


def _operand_type(op: Op, role: MmaOperandRole) -> Type:
    """Require a typed SSA operand before any form renders it."""
    value = mma_operand_value(op, role)
    if value is None or not isinstance(value.type, Type):
        raise ValueError(f"MMA operand {role} of {_op_id(op)!r} must define an IR type")
    return value.type


def _result_type(op: Op) -> Type:
    """Require a typed SSA result before a form emits its instruction."""
    ty = op.results[0].type
    if not isinstance(ty, Type):
        raise ValueError(f"MMA result of {_op_id(op)!r} must define an IR type")
    return ty


@dataclass(frozen=True)
class FlavorType:
    """A register type whose spelling changed at a newer LLVM flavor.

    ``base`` is the oldest flavor's type; ``overrides`` pairs a flavor with the
    type it changed to there, such as ``((LLVM_FLAVOR_LLVM22, I64),)`` for an
    operand LLVM 22 takes as ``i64`` instead of ``<2 x i32>``. A flavor uses the
    newest override at or below it, the same inheritance the declaration tables
    follow.
    """

    base: Type
    overrides: tuple[tuple[str, Type], ...] = ()

    def __post_init__(self) -> None:
        from ..lower_llvm import LLVM_FLAVORS

        object.__setattr__(self, "overrides", tuple(self.overrides))
        for flavor, _ in self.overrides:
            if flavor not in LLVM_FLAVORS:
                raise ValueError(
                    f"FlavorType names unknown LLVM flavor {flavor!r}; "
                    f"expected one of {LLVM_FLAVORS}"
                )

    def for_flavor(self, flavor: str) -> Type:
        """The type at ``flavor``: its newest override at or below it, else ``base``."""
        from ..lower_llvm import LLVM_FLAVORS

        if flavor not in LLVM_FLAVORS:
            raise ValueError(
                f"unknown LLVM flavor {flavor!r}; expected one of {LLVM_FLAVORS}"
            )
        own = dict(self.overrides)
        resolved = self.base
        for inherited in LLVM_FLAVORS[: LLVM_FLAVORS.index(flavor) + 1]:
            resolved = own.get(inherited, resolved)
        return resolved

    def every_flavor(self) -> tuple[Type, ...]:
        """The base type and every override, for checks that hold at all flavors."""
        return (self.base,) + tuple(t for _, t in self.overrides)


RegisterType = Type | FlavorType


@dataclass(frozen=True)
class _Form:
    """One register type, possibly per flavor; subclasses add how a value meets it."""

    register_type: RegisterType

    def register_type_at(self, flavor: str) -> Type:
        """The register type at LLVM ``flavor``."""
        if isinstance(self.register_type, FlavorType):
            return self.register_type.for_flavor(flavor)
        return self.register_type

    def _every_register_type(self) -> tuple[Type, ...]:
        if isinstance(self.register_type, FlavorType):
            return self.register_type.every_flavor()
        return (self.register_type,)


@dataclass(frozen=True)
class AsIs(_Form):
    """Pass the value as the register type: unchanged, or with one ``bitcast``.

    As a result form, the call returns the register type and the IR result is
    either that type or the same bits in another shape (one ``bitcast`` back),
    as :func:`~rocke.core.mma_operand.mma_result_needs_reinterpret` decides.
    """

    def check_operand(self, lowerer, op: Op, role: MmaOperandRole) -> None:
        """Validate the value without recording a declaration or emitting IR."""
        mma_operand_needs_reinterpret(
            _op_id(op),
            role,
            _operand_type(op, role),
            self.register_type_at(lowerer._flavor),
            lowerer._backend.arch,
        )

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` as the register type; returns its LLVM name."""
        self.check_operand(lowerer, op, role)
        return lowerer._mma_operand(op, role, self.register_type_at(lowerer._flavor))

    def check_result(self, lowerer, op: Op) -> bool:
        """Validate the result without declaring an intrinsic or emitting IR."""
        return mma_result_needs_reinterpret(
            _op_id(op),
            _result_type(op),
            self.register_type_at(lowerer._flavor),
            lowerer._backend.arch,
        )

    def emit_result(self, lowerer, op: Op, call: str) -> None:
        """Emit ``call`` and bind its value to ``op``'s result."""
        register = self.register_type_at(lowerer._flavor)
        result = op.results[0]
        if not self.check_result(lowerer, op):
            lowerer._current().emit(f"  {result.name} = {call}")
            return
        returned = lowerer._fresh("mma_d")
        lowerer._current().emit(f"  {returned} = {call}")
        lowerer._current().emit(
            f"  {result.name} = bitcast {_llvm_type(register)} {returned} "
            f"to {_llvm_type(result.type)}"
        )


@dataclass(frozen=True)
class ZeroPadded(_Form):
    """Widen the ``payload_words``-word packed payload to the ``<m x i32>`` register.

    The operand is exactly ``vec<i32 x payload_words>``, widened with
    ``m - payload_words`` zero words, or the whole ``vec<i32 x m>`` register,
    passed unchanged. Any other word count would be a different, incomplete
    matrix, so it is refused rather than padded. An operand form only: no
    instruction returns a padded result.
    """

    payload_words: int

    def __post_init__(self) -> None:
        if isinstance(self.payload_words, bool) or not isinstance(
            self.payload_words, int
        ):
            raise TypeError(
                "ZeroPadded payload_words is a word count (int), got "
                f"{self.payload_words!r}"
            )
        for t in self._every_register_type():
            if not (isinstance(t, VectorType) and t.elem == I32):
                raise TypeError(
                    f"ZeroPadded register type must be a vector of i32 words, got {t.name}"
                )
            if not 1 <= self.payload_words <= t.count:
                raise ValueError(
                    f"ZeroPadded payload of {self.payload_words!r} words does not fit "
                    f"{t.name}; it is 1 to {t.count} words"
                )

    def check_operand(self, lowerer, op: Op, role: MmaOperandRole) -> None:
        """Check format and complete payload width before any padding emits IR."""
        register = self.register_type_at(lowerer._flavor)
        payload = _operand_type(op, role)
        words = self.payload_words
        for kind, ty in (("operand", payload), ("register", register)):
            _require_integer_vector_count(
                f"MMA operand {role} of {_op_id(op)!r}", kind, ty
            )
        if payload not in (VectorType(I32, words), register):
            raise ValueError(
                f"MMA operand {role} of {_op_id(op)!r} is {payload.name}; the "
                f"instruction takes the operand's {words} packed 32-bit words as "
                f"vec<i32x{words}>, or the whole {_llvm_type(register)} register "
                "-- pass one of those"
            )

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` widened to the register; returns its LLVM name."""
        self.check_operand(lowerer, op, role)
        register = self.register_type_at(lowerer._flavor)
        value = mma_operand_value(op, role)
        payload = value.type
        words = self.payload_words
        if payload.count == register.count:
            return lowerer._operand(value)
        width = register.count
        mask = ", ".join(
            [f"i32 {i}" for i in range(words)] + [f"i32 {words}"] * (width - words)
        )
        padded = lowerer._fresh("mma_pad")
        lowerer._current().emit(
            f"  {padded} = shufflevector <{words} x i32> {lowerer._operand(value)}, "
            f"<{words} x i32> zeroinitializer, <{width} x i32> <{mask}>"
        )
        return padded


@dataclass(frozen=True)
class EvenHalves(_Form):
    """Keep ``n`` logical 16-bit values in the even half-slots of a ``2n``-slot register.

    As an operand form the value is ``n`` 16-bit elements, spread into the even
    slots with zero odd slots. As a result form the even slots are gathered back
    into the ``n``-element IR result; the odd slots are never read.
    """

    def __post_init__(self) -> None:
        for t in self._every_register_type():
            if not (
                isinstance(t, VectorType)
                and t.count % 2 == 0
                and type_bits(t.elem) == 16
            ):
                raise TypeError(
                    "EvenHalves register type must be a vector of an even number "
                    f"of 16-bit elements, got {t.name}"
                )

    @staticmethod
    def _logical(register: VectorType) -> VectorType:
        return VectorType(register.elem, register.count // 2)

    def check_operand(self, lowerer, op: Op, role: MmaOperandRole) -> None:
        """Validate the complete logical input before any spreading emits IR."""
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        ty = _operand_type(op, role)
        if not (
            isinstance(ty, VectorType)
            and ty.count == logical.count
            and type_bits(ty.elem) == 16
        ):
            raise ValueError(
                f"MMA operand {role} of {_op_id(op)!r} is {ty.name}; the "
                f"instruction keeps {logical.count} results in the even halves of "
                f"{_llvm_type(register)} -- pass the {logical.count} logical "
                "accumulator values"
            )
        mma_operand_needs_reinterpret(
            _op_id(op), role, ty, logical, lowerer._backend.arch
        )

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` spread into the even half-slots; returns its name."""
        self.check_operand(lowerer, op, role)
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        values = lowerer._mma_operand(op, role, logical)
        mask = ", ".join(f"i32 {i}, i32 {logical.count}" for i in range(logical.count))
        spread = lowerer._fresh("mma_spread")
        lowerer._current().emit(
            f"  {spread} = shufflevector {_llvm_type(logical)} {values}, "
            f"{_llvm_type(logical)} zeroinitializer, <{register.count} x i32> <{mask}>"
        )
        return spread

    def check_result(self, lowerer, op: Op) -> bool:
        """Validate the logical result before the call or a gather is emitted."""
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        ty = _result_type(op)
        if not (
            isinstance(ty, VectorType)
            and type_bits(ty) == type_bits(logical)
            and ty.count == logical.count
        ):
            raise ValueError(
                f"MMA result of {_op_id(op)!r} is {ty.name}; the instruction "
                f"returns {logical.count} results in the even halves of "
                f"{_llvm_type(register)}"
            )
        return mma_result_needs_reinterpret(
            _op_id(op), ty, logical, lowerer._backend.arch
        )

    def emit_result(self, lowerer, op: Op, call: str) -> None:
        """Emit ``call`` and gather its even half-slots into ``op``'s result."""
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        result = op.results[0]
        reinterpret = self.check_result(lowerer, op)
        returned = lowerer._fresh("mma_d")
        lowerer._current().emit(f"  {returned} = {call}")
        gathered = lowerer._fresh("mma_gather") if reinterpret else result.name
        mask = ", ".join(f"i32 {2 * i}" for i in range(logical.count))
        lowerer._current().emit(
            f"  {gathered} = shufflevector {_llvm_type(register)} {returned}, "
            f"{_llvm_type(register)} poison, <{logical.count} x i32> <{mask}>"
        )
        if gathered != result.name:
            lowerer._current().emit(
                f"  {result.name} = bitcast {_llvm_type(logical)} {gathered} "
                f"to {_llvm_type(result.type)}"
            )


MatrixOperandForm = AsIs | ZeroPadded | EvenHalves
MatrixResultForm = AsIs | EvenHalves


def f8f6f4_operand_form(dtype: str, register_type: VectorType) -> AsIs | ZeroPadded:
    """The form an f8f6f4 operand of matrix format ``dtype`` takes in ``register_type``.

    An 8-bit format (fp8e4m3, bf8e5m2) arrives as its logical values, or as
    words of the register's width, and passes with at most one ``bitcast``
    (:class:`AsIs`). A 6-bit or 4-bit format arrives as exactly its packed
    words (:func:`~rocke.core.arch.mfma_scale.matrix_operand_packing`) and is
    widened with zero words (:class:`ZeroPadded`)::

        fp8e4m3   vec<fp8e4m3x64>  --bitcast-->   <16 x i32>
        fp6e2m3   vec<i32x12>      --zero words-> <16 x i32>
        fp4e2m1   vec<i32x8>       --zero words-> <16 x i32>

    ``register_type`` is a plain vector: no f8f6f4 register changed spelling
    across LLVM flavors. Raises ``ValueError`` when ``dtype`` is not one of the
    five matrix formats.
    """
    if dtype_info(require_matrix_format(dtype)).encoded_bits == 8:
        return AsIs(register_type)
    packing = matrix_operand_packing(dtype, register_type.count)
    return ZeroPadded(register_type, packing.live_carriers)
