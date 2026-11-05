# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exact FP32 reachable-set primitives for the structured-logit (P4) check.

P4 predicts, for every exposed output coordinate, the set of stored values that
the pinned dense-kernel arithmetic can produce. The set is built by
propagating closed intervals of FP32 values through each step of the path, so
it encloses every reachable state rather than one chosen realization.

Everything here is exact: values are :class:`fractions.Fraction`, rounding to
FP32 is computed from the exact rational, and the only transcendental quantity,
``2**x``, is enclosed by a certified :mod:`decimal` bracket. There is no
attention implementation in this module; its inputs are closed-form scalars.

Soundness rests on monotonicity. On nonnegative arguments, round-to-nearest,
round-down, round-up, ``+``, ``*`` and the FP16/BF16 cast are nondecreasing in
each argument, and ``x -> 1/x`` is nonincreasing, so the images of interval
endpoints bound the image of every point inside.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache

import numpy as np

from ._p4_shared_analytical import target_bits

F32_FRACTION_BITS = 23
F32_MIN_EXPONENT = -126
F32_MAX_EXPONENT = 127
F32_UNIT_ROUNDOFF = Fraction(1, 1 << 24)
F32_SMALLEST_SUBNORMAL = Fraction(1, 1 << 149)
_F32_OVERFLOW = Fraction(1 << (F32_MAX_EXPONENT + 1))
# Strictly below every FP32 spacing, so ``x +- _NUDGE`` never crosses two grid
# points; used to step to a neighbor.
_NUDGE = Fraction(1, 1 << 300)

_ROUNDING_MODES = frozenset({"rn", "rd", "ru"})


def floor_log2(value: Fraction) -> int:
    """Return ``e`` with ``2**e <= value < 2**(e+1)`` for a positive rational."""
    if value <= 0:
        raise ValueError("floor_log2 requires a positive value")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    elif Fraction(2) ** (exponent + 1) <= value:
        exponent += 1
    return exponent


def f32_quantum(value: Fraction) -> Fraction:
    """Spacing of the FP32 grid at ``|value|``, measured upward.

    For a power of two this is the spacing above it (twice the spacing below),
    which is the wider of the two conventions for "one ULP of ``value``"."""
    magnitude = abs(value)
    if magnitude == 0:
        return F32_SMALLEST_SUBNORMAL
    exponent = max(floor_log2(magnitude), F32_MIN_EXPONENT)
    return Fraction(2) ** (exponent - F32_FRACTION_BITS)


def round_f32(value: Fraction, mode: str = "rn") -> Fraction:
    """Round an exact rational to FP32 and return the result's exact value.

    ``mode`` is ``"rn"`` (nearest, ties to even), ``"rd"`` (toward -inf) or
    ``"ru"`` (toward +inf). Subnormals are kept (gradual underflow). Overflow
    raises: no P4 quantity comes near the FP32 range limit."""
    if mode not in _ROUNDING_MODES:
        raise ValueError(f"mode must be one of {sorted(_ROUNDING_MODES)}")
    value = Fraction(value)
    if value == 0:
        return Fraction(0)
    quantum = f32_quantum(value)
    steps = value / quantum
    if mode == "rd":
        count = steps.numerator // steps.denominator
    elif mode == "ru":
        count = -((-steps.numerator) // steps.denominator)
    else:
        count = round(steps)  # Fraction.__round__ ties to even
    result = count * quantum
    if abs(result) >= _F32_OVERFLOW:
        raise OverflowError(f"{float(value)!r} overflows FP32")
    return result


def is_f32(value: Fraction) -> bool:
    """Return whether ``value`` is exactly representable in FP32."""
    value = Fraction(value)
    return abs(value) < _F32_OVERFLOW and round_f32(value) == value


def f32(value: float | np.floating) -> Fraction:
    """Exact rational value of ``value`` rounded once to FP32 by NumPy."""
    return Fraction(float(np.float32(value)))


def next_up(value: Fraction) -> Fraction:
    """Smallest FP32 value strictly greater than an FP32 ``value``."""
    return round_f32(Fraction(value) + _NUDGE, "ru")


def next_down(value: Fraction) -> Fraction:
    """Largest FP32 value strictly less than an FP32 ``value``."""
    return round_f32(Fraction(value) - _NUDGE, "rd")


@dataclass(frozen=True)
class F32Interval:
    """A closed interval whose endpoints are FP32 values, ``lo <= hi``.

    It stands for every FP32 value between the endpoints; a reachable-set
    operation returns an interval containing every result the modeled
    operation can produce from any members of its inputs."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        lo, hi = Fraction(self.lo), Fraction(self.hi)
        if lo > hi:
            raise ValueError(f"empty interval [{float(lo)!r}, {float(hi)!r}]")
        if not (is_f32(lo) and is_f32(hi)):
            raise ValueError("interval endpoints must be FP32 values")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def point(cls, value: Fraction) -> F32Interval:
        """The singleton interval of one FP32 value."""
        return cls(value, value)

    @classmethod
    def hull(cls, values: Iterable[Fraction]) -> F32Interval:
        """The smallest interval containing every listed FP32 value."""
        items = [Fraction(value) for value in values]
        if not items:
            raise ValueError("hull of no values")
        return cls(min(items), max(items))

    def __contains__(self, value: object) -> bool:
        return self.lo <= Fraction(value) <= self.hi  # type: ignore[arg-type]

    def _require_nonnegative(self) -> None:
        if self.lo < 0:
            raise ValueError("operation is defined on nonnegative intervals only")


def add_rn(left: F32Interval, right: F32Interval) -> F32Interval:
    """Image of one IEEE FP32 add, ``RN(a + b)``, over both intervals."""
    return F32Interval(
        round_f32(left.lo + right.lo, "rn"), round_f32(left.hi + right.hi, "rn")
    )


def mul_rn(left: F32Interval, right: F32Interval) -> F32Interval:
    """Image of one IEEE FP32 multiply, ``RN(a * b)``, on nonnegative intervals."""
    left._require_nonnegative()
    right._require_nonnegative()
    return F32Interval(
        round_f32(left.lo * right.lo, "rn"), round_f32(left.hi * right.hi, "rn")
    )


def rcp_rn(value: F32Interval) -> F32Interval:
    """Image of the correctly rounded reciprocal ``RN(1 / l)`` on ``l > 0``.

    This is the dense epilogue's ``fdiv 1.0, l`` (no ``!fpmath``), not the
    approximate ``v_rcp_f32``."""
    if value.lo <= 0:
        raise ValueError("reciprocal requires a positive interval")
    return F32Interval(round_f32(1 / value.hi, "rn"), round_f32(1 / value.lo, "rn"))


def add_faithful(left: F32Interval, right: F32Interval) -> F32Interval:
    """Image of an FP32 add whose rounding is only known to be faithful.

    Either neighbor of the exact sum is allowed (``RD`` or ``RU``). This models
    the MFMA accumulate where the instruction's rounding is not pinned."""
    return F32Interval(
        round_f32(left.lo + right.lo, "rd"), round_f32(left.hi + right.hi, "ru")
    )


def gamma(count: int) -> Fraction:
    """Higham's ``gamma_n = n u / (1 - n u)`` for FP32, ``u = 2**-24``."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    product = count * F32_UNIT_ROUNDOFF
    if product >= 1:
        raise ValueError("gamma is undefined for n u >= 1")
    return product / (1 - product)


def sum_any_order(terms: Iterable[F32Interval]) -> F32Interval:
    """Enclose an FP32 sum of nonnegative terms in any order or tree shape.

    For ``n`` nonnegative FP32 terms with exact sum ``S``, every recursive or
    tree summation returns ``S (1 + theta)`` with ``|theta| <= gamma_{n-1}``.
    Gradual underflow keeps FP32 addition exact when the result is subnormal,
    and a term flushed to zero is the member ``0`` of its interval when the
    interval starts at ``0``. The result is itself FP32, so the real bounds are
    snapped inward: up for the lower end, down for the upper end."""
    items = list(terms)
    if not items:
        return F32Interval.point(Fraction(0))
    for item in items:
        item._require_nonnegative()
    bound = gamma(len(items) - 1)
    lower = sum((item.lo for item in items), Fraction(0)) * (1 - bound)
    upper = sum((item.hi for item in items), Fraction(0)) * (1 + bound)
    return F32Interval(round_f32(lower, "ru"), round_f32(upper, "rd"))


def cast_bits_hull(value: F32Interval, dtype: str) -> tuple[int, int]:
    """FP16/BF16 RNE bits of both endpoints of a nonnegative interval.

    The cast is monotone, so every target value between the two returned bit
    patterns (inclusive, in bit order) is reachable from some member."""
    value._require_nonnegative()
    lo, hi = value.lo, value.hi
    return (
        target_bits(lo.numerator, lo.denominator, dtype),
        target_bits(hi.numerator, hi.denominator, dtype),
    )


# ---------------------------------------------------------------------------
# v_exp_f32 candidate sets (assumption A-E).
#
# The documented accuracy of the hardware base-2 exponential is "1 ULP" for
# FP32 (LLVM AMDGPU usage notes, ``llvm.exp2``). The convention for "one ULP"
# is not stated, so A-E takes the wider one: |E(x) - 2**x| <= f32_quantum(2**x),
# the spacing above 2**x. At a power of two this admits four values, because
# the spacing below a power of two is half the spacing above it.
# ---------------------------------------------------------------------------

EXP2_ASSUMPTION = "A-E v1: |E(x) - 2**x| <= f32_quantum(2**x)"

_DECIMAL_DIGITS = 110
_DECIMAL_RELATIVE_ERROR = Fraction(1, 10**100)
# Below this argument 2**x is far under the smallest subnormal, and the
# candidate set is decided by the subnormal spacing alone.
_EXP2_UNDERFLOW_ARGUMENT = Fraction(-300)
_EXP2_UNDERFLOW_ENCLOSURE = (Fraction(1, 1 << 1000), Fraction(1, 1 << 200))


def _exp2_enclosure(argument: Fraction) -> tuple[Fraction, Fraction]:
    """Rational bracket ``[lo, hi]`` containing ``2**argument``.

    Exact for an integer argument. Otherwise ``Decimal.ln`` and ``Decimal.exp``
    are correctly rounded at 110 digits, so the evaluated value is within a
    relative ``1e-100`` of ``2**argument`` for ``|argument| <= 300``."""
    # Range checks first: large FP32 magnitudes are integers, and 2**(-1e30)
    # must not be formed exactly.
    if argument < _EXP2_UNDERFLOW_ARGUMENT:
        return _EXP2_UNDERFLOW_ENCLOSURE
    if argument > F32_MAX_EXPONENT + 1:
        raise OverflowError("2**argument overflows FP32")
    if argument.denominator == 1:
        exact = Fraction(2) ** argument.numerator
        return exact, exact
    with localcontext() as context:
        context.prec = _DECIMAL_DIGITS
        value = Fraction(
            (
                Decimal(argument.numerator)
                / Decimal(argument.denominator)
                * Decimal(2).ln()
            ).exp()
        )
    return (
        value * (1 - _DECIMAL_RELATIVE_ERROR),
        value * (1 + _DECIMAL_RELATIVE_ERROR),
    )


@cache
def exp2_candidates(argument: Fraction) -> tuple[Fraction, ...]:
    """Every FP32 value ``z`` with ``|z - 2**x| <= f32_quantum(2**x)``.

    ``argument`` is the FP32 argument ``x`` the kernel passes to ``v_exp_f32``.
    Membership is certified for every point of the ``2**x`` bracket; an
    undecidable candidate raises rather than being guessed."""
    argument = Fraction(argument)
    if not is_f32(argument):
        raise ValueError("exp2 argument must be an FP32 value")
    low, high = _exp2_enclosure(argument)
    quantum_low, quantum_high = f32_quantum(low), f32_quantum(high)
    if quantum_low != quantum_high:
        raise ArithmeticError(f"2**{float(argument)!r} brackets a binade boundary")
    quantum = quantum_low

    candidates: list[Fraction] = []
    candidate = round_f32(low - quantum, "rd")
    while candidate <= high + quantum:
        within_everywhere = high - quantum <= candidate <= low + quantum
        outside_everywhere = candidate < low - quantum or candidate > high + quantum
        if within_everywhere:
            candidates.append(candidate)
        elif not outside_everywhere:
            raise ArithmeticError(
                f"cannot certify candidate {float(candidate)!r} for "
                f"exp2({float(argument)!r})"
            )
        candidate = next_up(candidate)
    if not candidates:
        raise ArithmeticError(f"no FP32 candidate for exp2({float(argument)!r})")
    return tuple(candidates)


def exp2_interval(argument: Fraction) -> F32Interval:
    """Hull of :func:`exp2_candidates`; the candidates are consecutive FP32 values."""
    return F32Interval.hull(exp2_candidates(argument))
