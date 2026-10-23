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

- :class:`ZeroPadded` -- a packed payload of ``n`` 32-bit words fills only part
  of an ``<m x i32>`` register; the high ``m - n`` words are zero::

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

from ..arch.mfma_scale import require_matrix_format
from ..dtypes import dtype_info
from ..ir import I32, Op, Type, VectorType, type_bits
from ..mma_operand import MmaOperandRole, mma_operand_value


def _llvm_type(t: Type) -> str:
    from ..lower_llvm import _llvm_type as render

    return render(t)


def _op_id(op: Op) -> str:
    from ..lower_llvm import _mma_op_id

    return _mma_op_id(op)


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
    either that type or the same bits in another shape (one ``bitcast`` back).
    """

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` as the register type; returns its LLVM name."""
        return lowerer._mma_operand(op, role, self.register_type_at(lowerer._flavor))

    def emit_result(self, lowerer, op: Op, call: str) -> None:
        """Emit ``call`` and bind its value to ``op``'s result."""
        register = self.register_type_at(lowerer._flavor)
        result = op.results[0]
        if result.type == register:
            lowerer._current().emit(f"  {result.name} = {call}")
            return
        result_bits, register_bits = type_bits(result.type), type_bits(register)
        if result_bits != register_bits:
            raise ValueError(
                f"MMA result of {_op_id(op)!r} is {result.type.name} ({result_bits} "
                f"bits); the instruction returns {register.name} ({register_bits} bits)"
            )
        returned = lowerer._fresh("mma_d")
        lowerer._current().emit(f"  {returned} = {call}")
        lowerer._current().emit(
            f"  {result.name} = bitcast {_llvm_type(register)} {returned} "
            f"to {_llvm_type(result.type)}"
        )


@dataclass(frozen=True)
class ZeroPadded(_Form):
    """Widen a packed ``<n x i32>`` payload to the ``<m x i32>`` register, n <= m.

    The high ``m - n`` words are zero; ``n == m`` passes the value unchanged. An
    operand form only: no instruction returns a padded result.
    """

    def __post_init__(self) -> None:
        for t in self._every_register_type():
            if not (isinstance(t, VectorType) and t.elem == I32):
                raise TypeError(
                    f"ZeroPadded register type must be a vector of i32 words, got {t.name}"
                )

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` widened to the register; returns its LLVM name."""
        register = self.register_type_at(lowerer._flavor)
        value = mma_operand_value(op, role)
        payload = value.type
        if not (
            isinstance(payload, VectorType)
            and payload.elem == I32
            and payload.count <= register.count
        ):
            raise ValueError(
                f"MMA operand {role} of {_op_id(op)!r} is {payload.name}; the "
                "instruction takes a packed payload of whole 32-bit words in "
                f"{_llvm_type(register)} -- pass the operand's packed words"
            )
        if payload.count == register.count:
            return lowerer._operand(value)
        words, width = payload.count, register.count
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

    def operand(self, lowerer, op: Op, role: MmaOperandRole) -> str:
        """Operand ``role`` of ``op`` spread into the even half-slots; returns its name."""
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        value = mma_operand_value(op, role)
        if not (
            isinstance(value.type, VectorType)
            and value.type.count == logical.count
            and type_bits(value.type.elem) == 16
        ):
            raise ValueError(
                f"MMA operand {role} of {_op_id(op)!r} is {value.type.name}; the "
                f"instruction keeps {logical.count} results in the even halves of "
                f"{_llvm_type(register)} -- pass the {logical.count} logical "
                "accumulator values"
            )
        values = lowerer._mma_operand(op, role, logical)
        mask = ", ".join(f"i32 {i}, i32 {logical.count}" for i in range(logical.count))
        spread = lowerer._fresh("mma_spread")
        lowerer._current().emit(
            f"  {spread} = shufflevector {_llvm_type(logical)} {values}, "
            f"{_llvm_type(logical)} zeroinitializer, <{register.count} x i32> <{mask}>"
        )
        return spread

    def emit_result(self, lowerer, op: Op, call: str) -> None:
        """Emit ``call`` and gather its even half-slots into ``op``'s result."""
        register = self.register_type_at(lowerer._flavor)
        logical = self._logical(register)
        result = op.results[0]
        if not (
            isinstance(result.type, VectorType)
            and type_bits(result.type) == type_bits(logical)
            and result.type.count == logical.count
        ):
            raise ValueError(
                f"MMA result of {_op_id(op)!r} is {result.type.name}; the instruction "
                f"returns {logical.count} results in the even halves of "
                f"{_llvm_type(register)}"
            )
        returned = lowerer._fresh("mma_d")
        lowerer._current().emit(f"  {returned} = {call}")
        gathered = (
            result.name if result.type == logical else lowerer._fresh("mma_gather")
        )
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


def f8f6f4_operand_form(dtype: str, register_type: RegisterType) -> AsIs | ZeroPadded:
    """The form an f8f6f4 operand of matrix format ``dtype`` takes in ``register_type``.

    An 8-bit format (fp8e4m3, bf8e5m2) arrives as its logical values, or as
    words of the register's width, and passes with at most one ``bitcast``
    (:class:`AsIs`). A 6-bit or 4-bit format arrives as its packed words and is
    widened with zero words (:class:`ZeroPadded`)::

        fp8e4m3   vec<fp8e4m3x64>  --bitcast-->   <16 x i32>
        fp4e2m1   vec<i32x8>       --zero words-> <16 x i32>

    Raises ``ValueError`` when ``dtype`` is not one of the five matrix formats.
    """
    if dtype_info(require_matrix_format(dtype)).encoded_bits == 8:
        return AsIs(register_type)
    return ZeroPadded(register_type)
