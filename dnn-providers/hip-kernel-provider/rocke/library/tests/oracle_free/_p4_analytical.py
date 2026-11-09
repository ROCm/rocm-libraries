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

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction
from functools import cache
from types import MappingProxyType

import numpy as np
from rocke.numeric.bf16 import f32_to_bf16

from ._p4_shared_analytical import (
    UNIFORM_BASIS_OFFSETS,
    UNIFORM_CALIBRATION,
    UNIFORM_QUICK_CASES,
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformSpecFingerprint,
    _basis_offset,
    _validate_observation,
    decode_bits,
    legal_interval,
    target_bits,
    uniform_inputs,
    uniform_inputs_folded,
)

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
    numerator, denominator = value.numerator, value.denominator
    # numerator / denominator lies in (2**(e-1), 2**(e+1)) for this e.
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        below = numerator < denominator << exponent
    else:
        below = numerator << -exponent < denominator
    return exponent - 1 if below else exponent


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
def exp2_candidates(argument: Fraction, ulps: int = 1) -> tuple[Fraction, ...]:
    """Every FP32 value ``z`` with ``|z - 2**x| <= ulps * f32_quantum(2**x)``.

    ``argument`` is the FP32 argument ``x`` the kernel passes to ``v_exp_f32``.
    ``ulps = 1`` is A-E; a larger bound only attributes failures (an output a
    looser exponential explains is labeled "assumption A-E"), never the verdict.
    Membership is certified for every point of the ``2**x`` bracket; an
    undecidable candidate raises rather than being guessed."""
    argument = Fraction(argument)
    if not is_f32(argument):
        raise ValueError("exp2 argument must be an FP32 value")
    if ulps < 1:
        raise ValueError("ulps must be at least 1")
    low, high = _exp2_enclosure(argument)
    quantum_low, quantum_high = f32_quantum(low), f32_quantum(high)
    if quantum_low != quantum_high:
        raise ArithmeticError(f"2**{float(argument)!r} brackets a binade boundary")
    quantum = ulps * quantum_low

    candidates: list[Fraction] = []
    # 2**x is positive, so a negative value is never a candidate (under the
    # 1-ULP bound none is within reach anyway).
    candidate = max(round_f32(low - quantum, "rd"), Fraction(0))
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


@cache
def exp2_interval(argument: Fraction, ulps: int = 1) -> F32Interval:
    """Hull of :func:`exp2_candidates`; the candidates are consecutive FP32 values."""
    return F32Interval.hull(exp2_candidates(argument, ulps))


@cache
def exp2_nearest(argument: Fraction) -> Fraction:
    """``RN(2**x)``: the candidate a correctly rounded exponential would return."""
    low, high = _exp2_enclosure(Fraction(argument))
    nearest = round_f32(low, "rn")
    if round_f32(high, "rn") != nearest:
        raise ArithmeticError(f"2**{float(argument)!r} is too close to a midpoint")
    return nearest


def target_value(bits: int, dtype: str) -> Fraction:
    """Exact value of nonnegative raw FP16/BF16 bits."""
    if dtype not in _TARGET_FORMAT:
        raise ValueError(dtype)
    fraction_bits, exponent_bits, bias = _TARGET_FORMAT[dtype]
    exponent = (int(bits) >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = int(bits) & ((1 << fraction_bits) - 1)
    if int(bits) >> (fraction_bits + exponent_bits):
        raise ValueError("target_value expects a nonnegative value")
    if exponent == 0:
        return Fraction(fraction) * Fraction(2) ** (1 - bias - fraction_bits)
    significand = (1 << fraction_bits) | fraction
    return Fraction(significand) * Fraction(2) ** (exponent - bias - fraction_bits)


_TARGET_FORMAT = {"fp16": (10, 5, 15), "bf16": (7, 8, 127)}


def ordered_distance(
    first: tuple[int, int], second: tuple[int, int]
) -> int:
    """Target-ULP gap between two bit ranges of nonnegative values; 0 if they meet."""
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0


# ---------------------------------------------------------------------------
# P4 fixtures. Every row has two logit classes: the distinguished keys, with
# logit ``c``, and the background keys, with logit 0. Each head carries its own
# ``c = a_h b_h / sqrt(D)`` (``8 a_h b_h`` times a row sign for the Walsh
# contraction), so a kernel that reads the wrong Q or K head changes a logit.
# ---------------------------------------------------------------------------

P4_PERIOD = 64
# The softmax scale ``1/sqrt(64)`` the subject adapter passes, and ``log2 e``
# rounded once to FP32 from a 50-digit value. Both come from the definition of
# attention, not from the kernel: a contract checks the kernel's constant
# against this one, so a wrong kernel constant fails instead of being copied.
_SCALE_ARGUMENT = 1.0 / 8.0
with localcontext() as _context:
    _context.prec = 50
    LOG2E_F32 = round_f32(Fraction(1 / Decimal(2).ln()), "rn")
SCORE_SCALE = round_f32(f32(_SCALE_ARGUMENT) * LOG2E_F32)
# The kernel's masked score and initial row max (pinned by the IR contract).
# Spec masking excludes the key; Tier B inside Tier A shows that -1e30 stores
# the same exact zeros (its weight is at most 2**-149, cast to +0).
MASKED_SCORE = f32(-1e30)


class P4Fixture(str, Enum):
    """Structured Q/K profiles; V uses the P3 basis and folded exposures."""

    IDENTITY = "identity"  # P4A: Q and K one-hot at the key's period residue
    CYCLIC = "cyclic"  # P4B: K one-hot shifted by r
    WALSH = "walsh"  # P4C: dense signed Walsh rows, one nonzero score per row
    LATE_MAX = "late_max"  # P4A with K rows below L zeroed: the max arrives late


_FIXTURE_SHIFT = {
    P4Fixture.IDENTITY: 0,
    P4Fixture.LATE_MAX: 0,
    P4Fixture.WALSH: 1,
}


def _is_signed_power_of_two(value: Fraction) -> bool:
    magnitude = abs(value)
    return magnitude > 0 and (
        (magnitude.denominator == 1 and magnitude.numerator & (magnitude.numerator - 1) == 0)
        or (magnitude.numerator == 1 and magnitude.denominator & (magnitude.denominator - 1) == 0)
    )


def _scale_token(value: Fraction) -> str:
    if value == 0:
        return "0"
    sign = "n" if value < 0 else "p"
    return f"{sign}{floor_log2(abs(value))}"


@dataclass(frozen=True)
class P4Case:
    """One P4 launch configuration: shape, fixture, per-head scales, KV tiling.

    ``q_scales[h]`` and ``k_scales[h]`` are ``a_h`` and ``b_h``: zero or signed
    powers of two, exact in FP16 and BF16. ``block_m``/``block_n`` are the
    subject's tile sizes, which fix the KV tiles each row visits."""

    base: UniformCase
    fixture: P4Fixture
    q_scales: tuple[Fraction, ...]
    k_scales: tuple[Fraction, ...]
    shift: int = 0
    late: int = 0
    block_m: int = 128
    block_n: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture", P4Fixture(self.fixture))
        object.__setattr__(self, "q_scales", tuple(Fraction(v) for v in self.q_scales))
        object.__setattr__(self, "k_scales", tuple(Fraction(v) for v in self.k_scales))
        base = self.base
        if base.head_dim != P4_PERIOD:
            raise ValueError(f"P4 requires head_dim == {P4_PERIOD}")
        if base.nhead_q != base.nhead_k:
            raise ValueError("P4 requires Hq == Hkv: each head carries its own logit")
        if len(self.q_scales) != base.nhead_q or len(self.k_scales) != base.nhead_k:
            raise ValueError("one Q and one K scale per head")
        for value in (*self.q_scales, *self.k_scales):
            if value != 0 and not _is_signed_power_of_two(value):
                raise ValueError(f"scale {value} is not zero or a signed power of two")
        if self.block_n != P4_PERIOD or self.block_m % self.block_n:
            raise ValueError("P4 tiles require block_n == 64 dividing block_m")
        if self.fixture is P4Fixture.CYCLIC:
            if not 0 < self.shift < P4_PERIOD:
                raise ValueError("the cyclic fixture requires 0 < shift < 64")
        elif self.shift:
            raise ValueError("shift applies to the cyclic fixture only")
        if self.fixture is P4Fixture.LATE_MAX:
            if not 0 < self.late < base.seqlen_k:
                raise ValueError("the late-max fixture requires 0 < late < seqlen_k")
        elif self.late:
            raise ValueError("late applies to the late-max fixture only")

    # The shape protocol the subject adapter reads.
    output_shape = property(lambda self: self.base.output_shape)
    key_shape = property(lambda self: self.base.key_shape)
    dtype = property(lambda self: self.base.dtype)
    mask = property(lambda self: self.base.mask)
    window = property(lambda self: self.base.window)
    head_scale_exponents = property(lambda self: self.base.head_scale_exponents)

    @property
    def residue_shift(self) -> int:
        """``r`` such that key ``k`` is distinguished for row ``q`` iff
        ``k + r = q (mod 64)``."""
        return self.shift if self.fixture is P4Fixture.CYCLIC else _FIXTURE_SHIFT[self.fixture]

    @property
    def case_id(self) -> str:
        detail = self.fixture.value
        if self.fixture is P4Fixture.CYCLIC:
            detail += f"-r{self.shift}"
        if self.fixture is P4Fixture.LATE_MAX:
            detail += f"-l{self.late}"
        scales = ",".join(
            f"{_scale_token(a)}x{_scale_token(b)}"
            for a, b in zip(self.q_scales, self.k_scales)
        )
        return f"{self.base.case_id}-p4{detail}-{scales}"


class P4Mutation(str, Enum):
    """Wrong kernel semantics whose Tier B must be disjoint from the correct one
    at some coordinate (T5). Each is a closed-form change of the score, the
    legal key set or the softmax recurrence; none needs a reference attention."""

    OMIT_SQRT_D = "omit_sqrt_d"  # scale 1 instead of 1/sqrt(D)
    EXP_ENCODING = "exp_encoding"  # exp2 of a natural-log score: LOG2E dropped
    K_ROW_PLUS_ONE = "k_row_plus_one"  # key k reads K row k + 1
    K_ROW_MINUS_ONE = "k_row_minus_one"  # key k reads K row k - 1
    INVERSE_SHIFT = "inverse_shift"  # transposed cyclic K: shift r becomes -r
    OMIT_WALSH_BLOCK_0 = "omit_walsh_block_0"  # QK skips dimensions 0..15
    OMIT_WALSH_BLOCK_1 = "omit_walsh_block_1"  # QK skips dimensions 16..31
    OMIT_WALSH_BLOCK_3 = "omit_walsh_block_3"  # QK skips dimensions 48..63
    WALSH_SIGN = "walsh_sign"  # the (-1)**k K row sign is lost
    Q_HEAD_ROTATE = "q_head_rotate"  # output head h reads Q head h + 1
    K_HEAD_ROTATE = "k_head_rotate"  # output head h reads K head h + 1
    INCLUDE_FUTURE_KEY = "include_future_key"
    EXCLUDE_LAST_KEY = "exclude_last_key"
    INCLUDE_OLD_WINDOW_KEY = "include_old_window_key"
    EXCLUDE_OLDEST_WINDOW_KEY = "exclude_oldest_window_key"
    OMIT_RESCALE = "omit_rescale"  # alpha = 1: l and o are never rescaled
    DENOMINATOR_PLUS_ONE = "denominator_plus_one"  # one phantom logit-0 key in l


_K_ROW_OFFSET = {P4Mutation.K_ROW_PLUS_ONE: 1, P4Mutation.K_ROW_MINUS_ONE: -1}
_OMITTED_WALSH_BLOCK = {
    P4Mutation.OMIT_WALSH_BLOCK_0: 0,
    P4Mutation.OMIT_WALSH_BLOCK_1: 1,
    P4Mutation.OMIT_WALSH_BLOCK_3: 3,
}
_WALSH_BLOCK = 16
# The one-key mask-boundary mutations share P3's legal-interval semantics.
_BOUNDARY_MUTATION = {
    mutation: UniformMutation(mutation.value)
    for mutation in (
        P4Mutation.INCLUDE_FUTURE_KEY,
        P4Mutation.EXCLUDE_LAST_KEY,
        P4Mutation.INCLUDE_OLD_WINDOW_KEY,
        P4Mutation.EXCLUDE_OLDEST_WINDOW_KEY,
    )
}


def is_distinguished(case: P4Case, query: int, key: int) -> bool:
    """Whether ``key`` carries the row's nonzero score (legality aside)."""
    if (key + case.residue_shift - query) % P4_PERIOD:
        return False
    return not (case.fixture is P4Fixture.LATE_MAX and key < case.late)


def score_numerator(
    case: P4Case,
    head: int,
    query: int,
    key: int,
    mutation: P4Mutation | None = None,
) -> Fraction:
    """The exact QK dot product ``acc`` (lemma B1: every partial sum is exact).

    Under ``mutation`` it is the dot product the mutated kernel forms. K rows
    outside ``[0, Sk)`` (an off-by-one read) follow the fixture's formula. With
    one 16-dimension block of the Walsh contraction omitted, the ``Q[q] K[k]``
    dot product loses ``sum_{d in block} H[z, d]``, ``z = q ^ (k + 1)``, which is
    ``16 (-1)**popcount(z & 16 b)`` when ``z % 16 == 0`` and zero otherwise."""
    heads = case.base.nhead_q
    q_head = (head + 1) % heads if mutation is P4Mutation.Q_HEAD_ROTATE else head
    k_head = (head + 1) % heads if mutation is P4Mutation.K_HEAD_ROTATE else head
    product = case.q_scales[q_head] * case.k_scales[k_head]
    row = key + _K_ROW_OFFSET.get(mutation, 0)
    if case.fixture is P4Fixture.WALSH:
        overlap = (query % P4_PERIOD) ^ ((row + 1) % P4_PERIOD)
        total = P4_PERIOD if overlap == 0 else 0
        block = _OMITTED_WALSH_BLOCK.get(mutation)
        if block is not None and overlap % _WALSH_BLOCK == 0:
            parity = bin(overlap & (_WALSH_BLOCK * block)).count("1") & 1
            total -= _WALSH_BLOCK * (-1 if parity else 1)
        negative = row & 1 and mutation is not P4Mutation.WALSH_SIGN
        return product * total * (-1 if negative else 1)
    shift = case.residue_shift
    if mutation is P4Mutation.INVERSE_SHIFT:
        shift = -shift
    if (row + shift - query) % P4_PERIOD:
        return Fraction(0)
    if case.fixture is P4Fixture.LATE_MAX and row < case.late:
        return Fraction(0)
    return product


def row_logit(case: P4Case, head: int, query: int) -> Fraction:
    """Natural-domain logit ``c`` of the row's distinguished keys."""
    return score_numerator(case, head, query, query - case.residue_shift) / 8


def walsh_matrix(size: int = P4_PERIOD) -> np.ndarray:
    """``H[x, d] = (-1)**popcount(x & d)``, the Sylvester Hadamard matrix."""
    index = np.arange(size)
    overlap = index[:, None] & index[None, :]
    parity = np.zeros_like(overlap)
    for bit in range(max(size - 1, 1).bit_length()):
        parity ^= (overlap >> bit) & 1
    return np.where(parity, -1.0, 1.0).astype(np.float32)


def p4_query_key(case: P4Case) -> tuple[np.ndarray, np.ndarray]:
    """Build exactly representable Q ``[B,Sq,H,D]`` and K ``[B,Sk,H,D]``."""
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = np.zeros(case.key_shape, dtype=np.float32)
    a = np.array([float(value) for value in case.q_scales], dtype=np.float32)
    b = np.array([float(value) for value in case.k_scales], dtype=np.float32)
    if case.fixture is P4Fixture.WALSH:
        hadamard = walsh_matrix()
        for row in range(case.base.seqlen_q):
            q[:, row, :, :] = a[:, None] * hadamard[row % P4_PERIOD]
        for key in range(case.base.seqlen_k):
            sign = -1.0 if key & 1 else 1.0
            k[:, key, :, :] = sign * b[:, None] * hadamard[(key + 1) % P4_PERIOD]
        return q, k
    for row in range(case.base.seqlen_q):
        q[:, row, :, row % P4_PERIOD] = a
    for key in range(case.base.seqlen_k):
        k[:, key, :, (key + case.residue_shift) % P4_PERIOD] = b
    if case.fixture is P4Fixture.LATE_MAX:
        k[:, : case.late] = 0.0
    return q, k


P4_EXPOSURES = (*(f"basis@{offset}" for offset in UNIFORM_BASIS_OFFSETS), "folded")


def p4_inputs(
    case: P4Case, exposure: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, K and the P3 exposure's V (scaled ``2**h`` per KV head)."""
    q, k = p4_query_key(case)
    if exposure == "folded":
        v = uniform_inputs_folded(case.base)[2]
    else:
        v = uniform_inputs(case.base, _basis_offset(exposure))[2]
    return q, k, v


# ---------------------------------------------------------------------------
# The subject's KV schedule and the per-row reachable state.
# ---------------------------------------------------------------------------


def visited_tiles(
    case: P4Case, query: int, mutation: P4Mutation | None = None
) -> tuple[int, ...]:
    """KV tiles the dense kernel visits for ``query``, in loop order.

    All rows of a ``block_m`` query block share one KV loop: top-left causal
    clamps it at the block's diagonal tile, and a sliding window starts it
    ``W / block_n`` tiles before the block's first diagonal tile. A row can
    therefore visit fully masked tiles before or after its legal keys. A
    mask-boundary mutation also visits the tile of a key it makes legal."""
    per_block = case.block_m // case.block_n
    block = query // case.block_m
    tiles = case.base.seqlen_k // case.block_n
    upper = min(block * per_block + per_block, tiles)
    lower = 0
    if case.mask == "window":
        lower = max(block * per_block - case.window // case.block_n, 0)
    visited = set(range(lower, upper))
    if mutation in _BOUNDARY_MUTATION:
        first, last = legal_interval(case.base, query, _BOUNDARY_MUTATION[mutation])
        if first <= last:
            visited.update(range(first // case.block_n, last // case.block_n + 1))
    return tuple(sorted(visited))


# ``scale`` is 1 instead of 1/8, or LOG2E is not applied.
_MUTATED_SCORE_SCALE = {
    P4Mutation.OMIT_SQRT_D: round_f32(f32(1.0) * LOG2E_F32),
    P4Mutation.EXP_ENCODING: f32(_SCALE_ARGUMENT),
}


@cache
def _score_of(numerator: Fraction, scale: Fraction = SCORE_SCALE) -> Fraction:
    return round_f32(numerator * scale, "rn")


def score(
    case: P4Case,
    head: int,
    query: int,
    key: int,
    mutation: P4Mutation | None = None,
) -> Fraction:
    """The FP32 score in the log2 domain, or the masked value."""
    first, last = legal_interval(case.base, query, _BOUNDARY_MUTATION.get(mutation))
    if not first <= key <= last:
        return MASKED_SCORE
    return _score_of(
        score_numerator(case, head, query, key, mutation),
        _MUTATED_SCORE_SCALE.get(mutation, SCORE_SCALE),
    )


@dataclass(frozen=True)
class TileStep:
    """One visited tile: the rescale factor and each key's weight ``p~``."""

    tile: int
    alpha: F32Interval
    weights: tuple[F32Interval, ...]


@dataclass(frozen=True)
class RowTrace:
    """Reachable FP32 state of one (row, head) after the KV loop."""

    steps: tuple[TileStep, ...]
    denominator: F32Interval
    reciprocal: F32Interval


@cache
def row_trace(
    case: P4Case,
    head: int,
    query: int,
    mutation: P4Mutation | None = None,
    e_ulps: int = 1,
) -> RowTrace:
    """Propagate the row's running max, weights and denominator.

    The running max is exact (``max`` does not round). Each ``v_exp_f32`` is its
    candidate interval; each tile's 64 weights are summed in any order
    (``gamma_63``), then ``l = RN(RN(l alpha) + l_tile)``. ``OMIT_RESCALE``
    fixes ``alpha = 1``; ``DENOMINATOR_PLUS_ONE`` adds a logit-0 key's weight to
    the last tile's sum, never to the accumulator. ``e_ulps`` loosens A-E for
    failure attribution only."""
    running_max = MASKED_SCORE
    denominator = F32Interval.point(Fraction(0))
    steps: list[TileStep] = []
    tiles = visited_tiles(case, query, mutation)
    for tile in tiles:
        start = tile * case.block_n
        scores = [
            score(case, head, query, key, mutation)
            for key in range(start, start + case.block_n)
        ]
        new_max = max(running_max, max(scores))
        if mutation is P4Mutation.OMIT_RESCALE:
            alpha = F32Interval.point(Fraction(1))
        else:
            alpha = exp2_interval(round_f32(running_max - new_max, "rn"), e_ulps)
        weights = tuple(
            exp2_interval(round_f32(value - new_max, "rn"), e_ulps) for value in scores
        )
        terms = weights
        if mutation is P4Mutation.DENOMINATOR_PLUS_ONE and tile == tiles[-1]:
            terms += (exp2_interval(round_f32(-new_max, "rn"), e_ulps),)
        denominator = add_rn(mul_rn(denominator, alpha), sum_any_order(terms))
        steps.append(TileStep(tile, alpha, weights))
        running_max = new_max
    return RowTrace(tuple(steps), denominator, rcp_rn(denominator))


def exposure_key(case: P4Case, exposure: str, dim: int, tile: int) -> int | None:
    """The key in ``tile`` whose V row is nonzero at ``dim``, if any."""
    start = tile * case.block_n
    if exposure == "folded":
        return start + dim
    key = _basis_offset(exposure) + dim
    return key if start <= key < start + case.block_n and key < case.base.seqlen_k else None


@cache
def _scaled_cast(dtype: str, weight: F32Interval, exponent: int) -> F32Interval:
    """PV product interval ``q_T(p~) * 2**h``: exact in FP32."""
    low, high = cast_bits_hull(weight, dtype)
    scale = Fraction(2) ** exponent
    return F32Interval(target_value(low, dtype) * scale, target_value(high, dtype) * scale)


MFMA_ROUNDING = ("faithful", "rn")


@cache
def _output_interval(
    case: P4Case,
    head: int,
    query: int,
    products: tuple[F32Interval | None, ...],
    mfma: str,
    mutation: P4Mutation | None,
    e_ulps: int,
) -> F32Interval:
    """``RN(o R(l))`` over the reachable accumulator ``o``.

    Per tile ``o = MFMA(RN(o alpha), product)``. With one nonzero product per
    output per tile, the MFMA performs at most one real addition; it is exact
    when the accumulator is zero, else faithful (or RN when pinned)."""
    if mfma not in MFMA_ROUNDING:
        raise ValueError(f"mfma must be one of {MFMA_ROUNDING}")
    trace = row_trace(case, head, query, mutation, e_ulps)
    accumulator = F32Interval.point(Fraction(0))
    for step, product in zip(trace.steps, products):
        accumulator = mul_rn(accumulator, step.alpha)
        if product is None or product.hi == 0:
            continue
        if accumulator.hi == 0:
            accumulator = product
        elif mfma == "faithful":
            accumulator = add_faithful(accumulator, product)
        else:
            accumulator = add_rn(accumulator, product)
    return mul_rn(accumulator, trace.reciprocal)


def tier_b_row(
    case: P4Case,
    exposure: str,
    query: int,
    head: int,
    *,
    mfma: str = "faithful",
    mutation: P4Mutation | None = None,
    e_ulps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest reachable output bits of one row and query head,
    under the correct or a ``mutation``'s semantics."""
    trace = row_trace(case, head, query, mutation, e_ulps)
    exponent = case.head_scale_exponents[head]
    low = np.zeros(case.base.head_dim, dtype=np.uint16)
    high = np.zeros(case.base.head_dim, dtype=np.uint16)
    for dim in range(case.base.head_dim):
        products = []
        for step in trace.steps:
            key = exposure_key(case, exposure, dim, step.tile)
            if key is None:
                products.append(None)
            else:
                weight = step.weights[key - step.tile * case.block_n]
                products.append(_scaled_cast(case.dtype, weight, exponent))
        output = _output_interval(
            case, head, query, tuple(products), mfma, mutation, e_ulps
        )
        low[dim], high[dim] = cast_bits_hull(output, case.dtype)
    return low, high


def tier_b_bounds(
    case: P4Case,
    exposure: str,
    *,
    rows: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    mfma: str = "faithful",
    mutation: P4Mutation | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tier B for one exposure: every reachable stored value of each coordinate
    lies, in bit order, between the two returned arrays (inclusive)."""
    low = np.zeros(case.output_shape, dtype=np.uint16)
    high = np.zeros(case.output_shape, dtype=np.uint16)
    for query in range(case.base.seqlen_q) if rows is None else rows:
        for head in range(case.base.nhead_q) if heads is None else heads:
            row_low, row_high = tier_b_row(
                case, exposure, query, head, mfma=mfma, mutation=mutation
            )
            low[:, query, head, :] = row_low
            high[:, query, head, :] = row_high
    return low, high


# ---------------------------------------------------------------------------
# Certified targets (T0) and Tier A (T3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateClass:
    """Key counts behind one exposed coordinate.

    The row has ``n_c`` legal distinguished keys (logit ``c``) and ``n_0``
    legal background keys; the coordinate sums ``m_c`` and ``m_0`` of them."""

    m_c: int
    m_0: int
    n_c: int
    n_0: int
    logit: Fraction

    @property
    def is_zero(self) -> bool:
        return self.m_c + self.m_0 == 0

    @property
    def is_row_max(self) -> bool:
        """All of the coordinate's keys carry the row's largest logit."""
        if self.is_zero:
            return False
        if self.m_c and not self.m_0:
            return self.logit >= 0 or self.n_0 == 0
        if self.m_0 and not self.m_c:
            return self.logit <= 0 or self.n_c == 0
        return False


@cache
def _row_counts(case: P4Case, head: int, query: int) -> tuple[int, int, int]:
    """``(first, last, n_c)``: the legal interval and its distinguished count."""
    first, last = legal_interval(case.base, query)
    distinguished = sum(
        1
        for key in range(first, last + 1)
        if score_numerator(case, head, query, key) != 0
    )
    return first, last, distinguished


def coordinate_class(
    case: P4Case, exposure: str, query: int, head: int, dim: int
) -> CoordinateClass:
    first, last, n_c = _row_counts(case, head, query)
    if exposure == "folded":
        keys = range(first + (dim - first) % P4_PERIOD, last + 1, P4_PERIOD)
    else:
        key = _basis_offset(exposure) + dim
        keys = range(key, key + 1) if first <= key <= last else range(0)
    m_c = sum(1 for key in keys if score_numerator(case, head, query, key) != 0)
    return CoordinateClass(
        m_c, len(keys) - m_c, n_c, last - first + 1 - n_c, row_logit(case, head, query)
    )


@cache
def certified_target_bits(
    dtype: str, exponent: int, item: CoordinateClass, digits: int = 50
) -> int:
    """``q_T(2**h (m_c e^c + m_0) / (n_c e^c + n_0))``, certified (T0).

    Rational when ``c = 0`` or the coordinate's class mix is proportional to the
    row's; otherwise irrational (Lindemann-Weierstrass), so never on a
    midpoint, and a ``digits``-digit value with a margin certifies the bits."""
    if item.is_zero:
        return 0
    scale = Fraction(2) ** exponent
    if item.logit == 0 or item.m_c * item.n_0 == item.m_0 * item.n_c:
        exact = scale * Fraction(item.m_c + item.m_0, item.n_c + item.n_0)
        return target_bits(exact.numerator, exact.denominator, dtype)
    with localcontext() as context:
        context.prec = digits
        growth = (Decimal(item.logit.numerator) / Decimal(item.logit.denominator)).exp()
        value = Fraction(
            (item.m_c * growth + item.m_0) / (item.n_c * growth + item.n_0)
        ) * scale
    bits = target_bits(value.numerator, value.denominator, dtype)
    center = target_value(bits, dtype)
    below = (target_value(bits - 1, dtype) + center) / 2 if bits else Fraction(0)
    above = (target_value(bits + 1, dtype) + center) / 2
    margin = value * Fraction(1, 10 ** (digits - 10))
    if not below + margin < value < above - margin:
        raise ArithmeticError(f"target {float(value)!r} is not certified at {digits} digits")
    return bits


def tier_a_bounds(
    case: P4Case,
    exposure: str,
    *,
    rows: Iterable[int] | None = None,
    classes: set[tuple[int, CoordinateClass]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Certified target bits and the T3 radius (1 at the row max, 2 elsewhere,
    0 for a structural zero) for every coordinate of one exposure. ``classes``,
    when given, collects each distinct ``(head exponent, class)``."""
    center = np.zeros(case.output_shape, dtype=np.uint16)
    radius = np.zeros(case.output_shape, dtype=np.uint16)
    for query in range(case.base.seqlen_q) if rows is None else rows:
        for head in range(case.base.nhead_q):
            exponent = case.head_scale_exponents[head]
            for dim in range(case.base.head_dim):
                item = coordinate_class(case, exposure, query, head, dim)
                if classes is not None:
                    classes.add((exponent, item))
                center[:, query, head, dim] = certified_target_bits(
                    case.dtype, exponent, item
                )
                radius[:, query, head, dim] = 0 if item.is_zero else (
                    1 if item.is_row_max else 2
                )
    return center, radius


# ---------------------------------------------------------------------------
# Path-exact emulation (diagnostic; never a verdict).
# ---------------------------------------------------------------------------

# Within a 64-key tile, lane half ``h`` sums keys ``8 kk + 4 h + j`` for
# ``kk = 0..7``, ``j = 0..3`` in that order (the PV B-operand relayout), then
# adds the other half's sum.
_LANE_ORDER = tuple(
    tuple(8 * step + 4 * half + j for step in range(8) for j in range(4))
    for half in range(2)
)


def _cast_bits(value: np.float32, dtype: str) -> int:
    array = np.array([value], dtype=np.float32)
    if dtype == "fp16":
        return int(array.astype(np.float16).view(np.uint16)[0])
    return int(f32_to_bf16(array).reshape(-1)[0])


def emulate_row(
    case: P4Case,
    exposure: str,
    query: int,
    head: int,
    *,
    rng: np.random.Generator | None = None,
    mutation: P4Mutation | None = None,
) -> np.ndarray:
    """Output bits of one row from an FP32 emulation of the dense path.

    Without ``rng`` this is the pinned realization: the documented lane order,
    ``E = RN(2**x)`` and an RN MFMA accumulate. With ``rng`` it draws a random
    summation tree per tile, an ``E`` candidate per call and a directed
    rounding per MFMA addition: a member of the reachable set, used to test
    that Tier B encloses it. ``mutation`` emulates the mutated kernel, whose
    realizations its own Tier B must enclose."""
    exponent = case.head_scale_exponents[head]
    scale = np.float32(2.0**exponent)

    def exp2(argument: np.float32) -> np.float32:
        exact = Fraction(float(argument))
        if rng is None:
            return np.float32(float(exp2_nearest(exact)))
        candidates = exp2_candidates(exact)
        return np.float32(float(candidates[int(rng.integers(len(candidates)))]))

    running_max = np.float32(float(MASKED_SCORE))
    denominator = np.float32(0.0)
    accumulator = np.zeros(case.base.head_dim, dtype=np.float32)
    tiles = visited_tiles(case, query, mutation)
    for tile in tiles:
        start = tile * case.block_n
        scores = np.array(
            [
                float(score(case, head, query, key, mutation))
                for key in range(start, start + case.block_n)
            ],
            dtype=np.float32,
        )
        new_max = np.float32(max(running_max, np.max(scores)))
        if mutation is P4Mutation.OMIT_RESCALE:
            alpha = np.float32(1.0)
        else:
            alpha = exp2(np.float32(running_max - new_max))
        weights = [exp2(np.float32(value - new_max)) for value in scores]
        phantom = []
        if mutation is P4Mutation.DENOMINATOR_PLUS_ONE and tile == tiles[-1]:
            phantom.append(exp2(np.float32(-new_max)))
        if rng is None:
            halves = []
            for order in _LANE_ORDER:
                total = np.float32(0.0)
                for index in order:
                    total = np.float32(total + weights[index])
                halves.append(total)
            tile_sum = np.float32(halves[0] + halves[1])
            for extra in phantom:
                tile_sum = np.float32(tile_sum + extra)
        else:
            items = list(weights) + phantom
            while len(items) > 1:
                left = items.pop(int(rng.integers(len(items))))
                right = items.pop(int(rng.integers(len(items))))
                items.append(np.float32(left + right))
            tile_sum = items[0]
        denominator = np.float32(np.float32(denominator * alpha) + tile_sum)
        accumulator = (accumulator * alpha).astype(np.float32)
        for dim in range(case.base.head_dim):
            key = exposure_key(case, exposure, dim, tile)
            if key is None:
                continue
            cast = target_value(_cast_bits(weights[key - start], case.dtype), case.dtype)
            product = np.float32(float(cast)) * scale
            if product == 0:
                continue
            if accumulator[dim] == 0:
                accumulator[dim] = product
            elif rng is None:
                accumulator[dim] = np.float32(accumulator[dim] + product)
            else:
                exact = Fraction(float(accumulator[dim])) + Fraction(float(product))
                mode = ("rd", "ru")[int(rng.integers(2))]
                accumulator[dim] = np.float32(float(round_f32(exact, mode)))
        running_max = new_max
    reciprocal = np.float32(np.float32(1.0) / denominator)
    return np.array(
        [_cast_bits(np.float32(value * reciprocal), case.dtype) for value in accumulator],
        dtype=np.uint16,
    )


# ---------------------------------------------------------------------------
# Per-head scales chosen by separation in output space.
# ---------------------------------------------------------------------------

# Head logits per fixture family. Periodic fixtures (identity, cyclic, late
# max) cover large and moderate logits of both signs; the Walsh row sign
# alternates, so its magnitudes cover both signs on their own.
P4_HEAD_LOGITS: Mapping[str, tuple[Fraction, ...]] = MappingProxyType(
    {
        "periodic": (Fraction(8), Fraction(-8), Fraction(2), Fraction(-1, 2)),
        "walsh": (Fraction(8), Fraction(2), Fraction(1, 2), Fraction(4)),
    }
)
# Rows sampled for head-separation gaps: block, tile and window edges.
_GAP_ROWS = (0, 6, 7, 31, 63, 64, 65, 127, 128, 191, 192, 255)


def fixture_family(fixture: P4Fixture) -> str:
    return "walsh" if P4Fixture(fixture) is P4Fixture.WALSH else "periodic"


def _logit_case(base: UniformCase, block_m: int, family: str, logits: Sequence[Fraction]) -> P4Case:
    """A case whose head ``h`` has logit ``logits[h]`` with ``b_h = 1``."""
    fixture = P4Fixture.WALSH if family == "walsh" else P4Fixture.IDENTITY
    factor = Fraction(1, 8) if family == "walsh" else Fraction(8)
    return P4Case(
        base,
        fixture,
        tuple(value * factor for value in logits),
        tuple(Fraction(1) for _ in logits),
        block_m=block_m,
    )


@cache
def logit_gap(
    base: UniformCase, block_m: int, family: str, head: int, logit: Fraction, other: Fraction
) -> int:
    """Largest Tier B separation, over the sampled rows and every exposure, of
    head ``head`` computed with logit ``logit`` versus ``other``."""
    logits = [Fraction(1)] * base.nhead_q
    logits[head] = logit
    correct = _logit_case(base, block_m, family, logits)
    logits[head] = other
    mutated = _logit_case(base, block_m, family, logits)
    gap = 0
    for exposure in P4_EXPOSURES:
        for query in _GAP_ROWS:
            first = tier_b_row(correct, exposure, query, head)
            second = tier_b_row(mutated, exposure, query, head)
            for dim in range(base.head_dim):
                gap = max(
                    gap,
                    ordered_distance(
                        (int(first[0][dim]), int(first[1][dim])),
                        (int(second[0][dim]), int(second[1][dim])),
                    ),
                )
    return gap


def head_scale_margin(
    bases: Sequence[tuple[UniformCase, int]],
    family: str,
    q_scales: Sequence[Fraction],
    k_scales: Sequence[Fraction],
) -> int:
    """Worst separation over every single wrong Q-head or K-head read.

    Output head ``h`` reading Q head ``g`` has logit ``c_h a_g / a_h``; reading
    K head ``g``, ``c_h b_g / b_h``. A map that changes several heads is
    separated at least as well as its best single change, so single changes
    bound every wrong map."""
    worst: int | None = None
    heads = len(q_scales)
    for base, block_m in bases:
        for head in range(heads):
            logit = P4_HEAD_LOGITS[family][head]
            for other_head in range(heads):
                if other_head == head:
                    continue
                for ratio in (
                    q_scales[other_head] / q_scales[head],
                    k_scales[other_head] / k_scales[head],
                ):
                    gap = logit_gap(base, block_m, family, head, logit, logit * ratio)
                    worst = gap if worst is None else min(worst, gap)
    return 0 if worst is None else worst


def search_head_scales(
    bases: Sequence[tuple[UniformCase, int]],
    family: str,
    exponents: Sequence[int] = (-2, -1, 0, 1, 2, 3),
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], int]:
    """Maximize :func:`head_scale_margin` over signed power-of-two ``a_h``;
    ``b_h`` follows from the head logit. Ties prefer scales nearest one."""
    logits = P4_HEAD_LOGITS[family]
    choices = [Fraction(sign) * Fraction(2) ** power for power in exponents for sign in (1, -1)]
    product_of = (lambda c: c / 8) if family == "walsh" else (lambda c: c * 8)

    def k_scale(head: int, a: Fraction) -> Fraction | None:
        b = product_of(logits[head]) / a
        return b if _is_signed_power_of_two(b) and Fraction(1, 64) <= abs(b) <= 64 else None

    best: tuple[int, int, tuple, tuple] | None = None
    for combination in itertools.product(choices, repeat=len(logits)):
        k_scales = [k_scale(head, a) for head, a in enumerate(combination)]
        if any(value is None for value in k_scales):
            continue
        if len(set(combination)) < len(logits) or len(set(k_scales)) < len(logits):
            continue
        margin = head_scale_margin(bases, family, combination, k_scales)
        size = sum(abs(floor_log2(abs(v))) for v in (*combination, *k_scales))
        key = (margin, -size)
        if best is None or key > best[:2]:
            best = (margin, -size, tuple(combination), tuple(k_scales))
    if best is None:
        raise ValueError("no admissible head scales")
    return best[2], best[3], best[0]


# ---------------------------------------------------------------------------
# The quick matrix.
# ---------------------------------------------------------------------------

# Chosen by ``search_head_scales(P4_SEARCH_BASES, family)`` with the default
# exponent range; ``P4_HEAD_SCALE_MARGIN`` is the attained worst-case gap in
# target ULPs, recomputed by a CPU contract.
P4_HEAD_SCALES: Mapping[str, tuple[tuple[Fraction, ...], tuple[Fraction, ...]]] = (
    MappingProxyType(
        {
            "periodic": (
                (Fraction(1), Fraction(2), Fraction(-4), Fraction(-1)),
                (Fraction(64), Fraction(-32), Fraction(-4), Fraction(4)),
            ),
            "walsh": (
                (Fraction(1), Fraction(-1, 4), Fraction(1, 4), Fraction(-1)),
                (Fraction(1), Fraction(-1), Fraction(1, 4), Fraction(-1, 2)),
            ),
        }
    )
)
P4_HEAD_SCALE_MARGIN: Mapping[str, int] = MappingProxyType({"periodic": 181, "walsh": 181})
P4_CYCLIC_SHIFT = 37
P4_LATE_KEYS = 64
# Tile sizes of the calibrated P3 quick specs (the same dispatch identities).
_QUICK_BLOCK_M = {
    dtype: entry.spec.block_m for (_, _, dtype), entry in UNIFORM_CALIBRATION.items()
}
P4_SEARCH_BASES = tuple(
    (base, _QUICK_BLOCK_M[base.dtype]) for base in UNIFORM_QUICK_CASES
)


def p4_quick_case(base: UniformCase, fixture: P4Fixture) -> P4Case:
    """The quick-lane case of one fixture on one P3 quick shape."""
    fixture = P4Fixture(fixture)
    q_scales, k_scales = P4_HEAD_SCALES[fixture_family(fixture)]
    return P4Case(
        base,
        fixture,
        q_scales,
        k_scales,
        shift=P4_CYCLIC_SHIFT if fixture is P4Fixture.CYCLIC else 0,
        late=P4_LATE_KEYS if fixture is P4Fixture.LATE_MAX else 0,
        block_m=_QUICK_BLOCK_M[base.dtype],
    )


P4_QUICK_CASES = tuple(
    p4_quick_case(base, fixture)
    for base in UNIFORM_QUICK_CASES
    for fixture in P4Fixture
)


@cache
def tier_b_exposure(
    case: P4Case, exposure: str, mutation: P4Mutation | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cached, read-only :func:`tier_b_bounds` for a whole exposure."""
    low, high = tier_b_bounds(case, exposure, mutation=mutation)
    low.setflags(write=False)
    high.setflags(write=False)
    return low, high


@cache
def tier_a_exposure(
    case: P4Case, exposure: str
) -> tuple[np.ndarray, np.ndarray, frozenset[tuple[int, CoordinateClass]]]:
    """Cached, read-only :func:`tier_a_bounds` for a whole exposure, with the
    distinct ``(head exponent, class)`` pairs behind its targets."""
    classes: set[tuple[int, CoordinateClass]] = set()
    center, radius = tier_a_bounds(case, exposure, classes=classes)
    center.setflags(write=False)
    radius.setflags(write=False)
    return center, radius, frozenset(classes)


# ---------------------------------------------------------------------------
# Mutation separation (T5).
# ---------------------------------------------------------------------------

_EVERY_FIXTURE = (
    P4Mutation.OMIT_SQRT_D,
    P4Mutation.EXP_ENCODING,
    P4Mutation.K_ROW_PLUS_ONE,
    P4Mutation.K_ROW_MINUS_ONE,
    P4Mutation.Q_HEAD_ROTATE,
    P4Mutation.K_HEAD_ROTATE,
    P4Mutation.OMIT_RESCALE,
    P4Mutation.DENOMINATOR_PLUS_ONE,
)
_FIXTURE_MUTATIONS = {
    P4Fixture.CYCLIC: (P4Mutation.INVERSE_SHIFT,),
    P4Fixture.WALSH: (*_OMITTED_WALSH_BLOCK, P4Mutation.WALSH_SIGN),
}
_MASK_MUTATIONS = {
    "causal": (P4Mutation.INCLUDE_FUTURE_KEY, P4Mutation.EXCLUDE_LAST_KEY),
    "window": (
        P4Mutation.INCLUDE_OLD_WINDOW_KEY,
        P4Mutation.EXCLUDE_OLDEST_WINDOW_KEY,
    ),
}


def applicable_mutations(case: P4Case) -> tuple[P4Mutation, ...]:
    """The mutations ``case`` can express: transposing K is a shift only for the
    cyclic fixture, Walsh blocks and signs exist only in the Walsh fixture, and
    each mask has its own one-key boundary."""
    return (
        *_EVERY_FIXTURE,
        *_FIXTURE_MUTATIONS.get(case.fixture, ()),
        *_MASK_MUTATIONS[case.mask],
    )


@dataclass(frozen=True)
class Separation:
    """``G(m)``: the largest target-ULP distance between the correct and the
    mutated Tier B over the evaluated coordinates, and where it is attained
    (``exposure`` and ``b:q:h:d``). ``gap > 0`` proves the mutation is caught."""

    mutation: P4Mutation
    gap: int
    exposure: str | None
    coordinate: tuple[int, int, int, int] | None


# Block, tile, window and causal-diagonal edges of both quick kernels. A gap on
# a subset of rows lower-bounds the full gap, which is enough to prove I5.
P4_SEPARATION_ROWS = (0, 1, 6, 7, 31, 63, 64, 65, 127, 128, 191, 192, 255)

SEPARATION_LABELS = (
    "owned by P3",
    "owned by another P4 fixture",
    "needs a stronger fixture",
    "out of scope",
)
# Quick cells whose correct and mutated Tier B overlap at every coordinate,
# keyed by ``(case_id, mutation)``. Tier B is never loosened to separate them.
# On the causal kernel the identity, cyclic and Walsh rows all reach their max
# in the first visited tile, so every later ``alpha = E(0)`` admits 1 and the
# omitted rescale is a member of the correct reachable set; late max, whose
# max arrives in a later tile, separates it on the same kernel.
P4_UNSEPARATED: Mapping[tuple[str, P4Mutation], str] = MappingProxyType(
    {
        (case.case_id, P4Mutation.OMIT_RESCALE): "owned by another P4 fixture"
        for case in P4_QUICK_CASES
        if case.mask == "causal" and case.fixture is not P4Fixture.LATE_MAX
    }
)


def mutation_separation(
    case: P4Case,
    mutation: P4Mutation,
    *,
    rows: Iterable[int] | None = P4_SEPARATION_ROWS,
) -> Separation:
    """Evaluate T5 for one mutation over every exposure; ``rows=None`` uses
    every row."""
    mutation = P4Mutation(mutation)
    if mutation not in applicable_mutations(case):
        raise ValueError(f"mutation {mutation.value} does not apply to {case.case_id}")
    best = Separation(mutation, 0, None, None)
    queries = range(case.base.seqlen_q) if rows is None else tuple(rows)
    for exposure in P4_EXPOSURES:
        for query in queries:
            for head in range(case.base.nhead_q):
                correct = tier_b_row(case, exposure, query, head)
                mutated = tier_b_row(case, exposure, query, head, mutation=mutation)
                low, high = (bits.astype(np.int64) for bits in correct)
                mutated_low, mutated_high = (bits.astype(np.int64) for bits in mutated)
                gaps = np.maximum(np.maximum(mutated_low - high, low - mutated_high), 0)
                dim = int(np.argmax(gaps))
                if gaps[dim] > best.gap:
                    best = Separation(
                        mutation, int(gaps[dim]), exposure, (0, query, head, dim)
                    )
    return best


def format_separation(case: P4Case, separation: Separation) -> str:
    where = (
        "n/a"
        if separation.coordinate is None
        else f"{separation.exposure} b:q:h:d="
        + ":".join(str(value) for value in separation.coordinate)
    )
    return (
        f"{case.dtype}-{case.mask}-{case.fixture.value} "
        f"{separation.mutation.value}: G={separation.gap} at {where}"
    )


# ---------------------------------------------------------------------------
# Analyzer: the Tier B verdict, the failure taxonomy and diagnostics.
# ---------------------------------------------------------------------------

MFMA_ASSUMPTION = "A-MFMA v1: a folded MFMA addition is RD or RU of the exact sum"
P4_ASSUMPTIONS = ("A-E v1", "A-MFMA v1")
# Looser v_exp_f32 bound used only to attribute a failure to A-E.
_RELAXED_EXP2_ULPS = 2


class P4Label(str, Enum):
    """Strongest contract an observed coordinate satisfies, checked in this order.

    A zero-gap mutation (the fixture lacks sensitivity) is a CPU-side label,
    ``P4_UNSEPARATED``, not an output label."""

    PASS = "pass"
    INVALID = "invalid"  # NaN (including the unwritten prefill) or Inf
    NEGATIVE = "negative"  # sign bit set, -0 included
    STRUCTURAL = "structural"  # a required +0 coordinate stores another pattern
    E2 = "E2: denorm/rounding mode"  # FP16-subnormal target: near it, or +0
    A_MFMA = "assumption A-MFMA"  # folded addition near target, basis clean
    A_E = "assumption A-E"  # explained only by a looser v_exp_f32
    PATH_MODEL = "path model"  # outside Tier B, inside Tier A: re-pin, never widen
    NUMERICAL = "numerical"  # outside Tier A, or outside a mutated Tier B


P4_FAILURE_LABELS = tuple(label for label in P4Label if label is not P4Label.PASS)
_FINITE_MASK = {"fp16": 0x7C00, "bf16": 0x7F80}
_SIGN_BIT = 0x8000
_FP16_SMALLEST_NORMAL_BITS = 0x0400


@dataclass(frozen=True)
class P4GroupStats:
    """One exposure's evidence for the rows of one legal count and query head.

    ``exact`` counts nonstructural coordinates equal to the certified target;
    ``ambiguous`` counts two-value Tier B sets and ``ambiguous_high`` how many
    of them stored the upper value. Row sums (folded only, probability units)
    are diagnostics, never part of the verdict."""

    coordinates: int
    structural: int
    compared: int
    exact: int
    ambiguous: int
    ambiguous_high: int
    labels: Mapping[str, int]
    max_row_sum_stored: float
    max_row_sum_real: float


@dataclass(frozen=True)
class P4Result:
    """Tier B verdict of one observation, with the taxonomy of every failure."""

    mutation: P4Mutation | None
    violations: tuple[str, ...]
    labels: Mapping[str, int]
    first: Mapping[str, tuple[str, tuple[int, int, int, int]]]
    compared: int
    exact: int
    ambiguous: int
    ambiguous_high: int
    groups: Mapping[str, Mapping[tuple[int, int], P4GroupStats]]

    @property
    def exact_rate(self) -> float:
        return self.exact / self.compared if self.compared else 0.0


def _genuine_addition(case: P4Case, query: int, head: int, dim: int) -> bool:
    """Whether a folded coordinate's MFMA adds two nonzero values (§2.1)."""
    trace = row_trace(case, head, query)
    exponent = case.head_scale_exponents[head]
    nonzero = 0
    for step in trace.steps:
        key = exposure_key(case, "folded", dim, step.tile)
        weight = step.weights[key - step.tile * case.block_n]
        if _scaled_cast(case.dtype, weight, exponent).hi > 0:
            nonzero += 1
    return nonzero >= 2


def _attribute(
    case: P4Case,
    exposure: str,
    query: int,
    head: int,
    dim: int,
    bits: int,
    center: int,
    radius: int,
    basis_clean: bool,
    relaxed: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> P4Label:
    """Label a nonnegative finite value outside the correct Tier B.

    A mode or assumption label needs a value its failure can produce: E2 a
    value inside Tier A or a flushed ``+0`` at an FP16-subnormal target, A-MFMA
    a value inside Tier A. A far value there is numerical, not a mode.
    ``relaxed()`` returns the row's Tier B under a 2-ULP exponential, computed
    at most once per row."""
    near = abs(bits - center) <= radius
    if (
        case.dtype == "fp16"
        and 0 < center < _FP16_SMALLEST_NORMAL_BITS
        and (near or bits == 0)
    ):
        return P4Label.E2
    if (
        near
        and exposure == "folded"
        and basis_clean
        and _genuine_addition(case, query, head, dim)
    ):
        return P4Label.A_MFMA
    low, high = relaxed()
    if low[dim] <= bits <= high[dim]:
        return P4Label.A_E
    if near:
        return P4Label.PATH_MODEL
    return P4Label.NUMERICAL


@dataclass
class _GroupBuilder:
    coordinates: int = 0
    structural: int = 0
    compared: int = 0
    exact: int = 0
    ambiguous: int = 0
    ambiguous_high: int = 0
    labels: dict[str, int] | None = None
    max_row_sum_stored: float = 0.0
    max_row_sum_real: float = 0.0

    def freeze(self) -> P4GroupStats:
        return P4GroupStats(
            self.coordinates,
            self.structural,
            self.compared,
            self.exact,
            self.ambiguous,
            self.ambiguous_high,
            MappingProxyType(dict(self.labels or {})),
            self.max_row_sum_stored,
            self.max_row_sum_real,
        )


def analyze_p4_policy(
    case: P4Case,
    observation: UniformObservation,
    *,
    mutation: P4Mutation | str | None = None,
    rows: Iterable[int] | None = None,
    require_complete: bool = True,
) -> P4Result:
    """Judge raw output bits by Tier B membership, coordinate by coordinate.

    A coordinate passes iff its bits lie in Tier B; a structural zero (masked
    key, Tier B ``{+0}``) must store exactly ``+0``. Every other outcome fails
    and carries the strongest contract it still satisfies (``P4Label``). With
    ``mutation`` the observation is judged against the mutated Tier B and must
    be rejected; failures are then only labeled invalid, negative, structural
    or numerical. ``rows`` restricts the judgment to those query rows.

    Observed values are only compared: nothing here can enlarge Tier B or the
    exponential candidate sets (I6)."""
    selected = None if mutation is None else P4Mutation(mutation)
    if selected is not None and selected not in applicable_mutations(case):
        raise ValueError(f"mutation {selected.value} does not apply to {case.case_id}")
    _validate_observation(case.base, observation, require_complete=require_complete)
    queries = tuple(range(case.base.seqlen_q)) if rows is None else tuple(rows)
    if not queries:
        raise ValueError("rows must select at least one query")
    exponents = np.array(case.head_scale_exponents)

    # Pass 1: membership masks per exposure.
    judged: dict[str, tuple] = {}
    for exposure, raw in observation.exposures:
        if rows is None:
            low, high = tier_b_exposure(case, exposure, selected)
        else:
            low, high = tier_b_bounds(case, exposure, rows=queries, mutation=selected)
        center, radius, _ = tier_a_exposure(case, exposure)
        index = np.asarray(queries)
        bits = raw[:, index].astype(np.int64)
        low, high = low[:, index].astype(np.int64), high[:, index].astype(np.int64)
        center = center[:, index].astype(np.int64)
        radius = radius[:, index].astype(np.int64)
        finite = (bits & _FINITE_MASK[case.dtype]) != _FINITE_MASK[case.dtype]
        sign = (bits & _SIGN_BIT) != 0
        if selected is None:
            required_zero = radius == 0
        else:
            required_zero = (low == 0) & (high == 0)
        valid = finite & ~sign
        inside = valid & ~required_zero & (bits >= low) & (bits <= high)
        labels = np.full(bits.shape, P4Label.PASS.value, dtype=object)
        labels[~finite] = P4Label.INVALID.value
        labels[finite & sign] = P4Label.NEGATIVE.value
        labels[valid & required_zero & (bits != 0)] = P4Label.STRUCTURAL.value
        outside = valid & ~required_zero & ~inside
        judged[exposure] = (
            bits, low, high, center, radius, required_zero, labels, outside
        )

    basis = [name for name in judged if name != "folded"]

    def basis_clean(batch: int, row: int, head: int) -> bool:
        if len(basis) != len(UNIFORM_BASIS_OFFSETS):
            return False
        return all(
            np.all(judged[name][6][batch, row, head] == P4Label.PASS.value)
            and not np.any(judged[name][7][batch, row, head])
            for name in basis
        )

    # Pass 2: attribute each failure outside Tier B.
    relaxed_rows: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def relaxed(exposure: str, query: int, head: int):
        key = (exposure, query, head)
        if key not in relaxed_rows:
            relaxed_rows[key] = tier_b_row(
                case, exposure, query, head, e_ulps=_RELAXED_EXP2_ULPS
            )
        return relaxed_rows[key]

    for exposure, judgment in judged.items():
        bits, _, _, center, radius, _, labels, outside = judgment
        for batch, row, head, dim in zip(*np.nonzero(outside)):
            query = queries[row]
            if selected is not None:
                label = P4Label.NUMERICAL
            else:
                label = _attribute(
                    case,
                    exposure,
                    query,
                    int(head),
                    int(dim),
                    int(bits[batch, row, head, dim]),
                    int(center[batch, row, head, dim]),
                    int(radius[batch, row, head, dim]),
                    basis_clean(batch, row, head),
                    lambda: relaxed(exposure, query, int(head)),
                )
            labels[batch, row, head, dim] = label.value

    # Pass 3: counts, first failures and per-group evidence.
    totals: dict[str, int] = {label.value: 0 for label in P4Label}
    first: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
    groups: dict[str, dict[tuple[int, int], _GroupBuilder]] = {}
    compared = exact = ambiguous = ambiguous_high = 0
    for exposure, (bits, low, high, center, radius, zero, labels, _) in judged.items():
        by_group = groups.setdefault(exposure, {})
        decoded = None
        if exposure == "folded" and selected is None:
            scale = np.ldexp(1.0, exponents)[None, None, :, None]
            valid = (bits & _FINITE_MASK[case.dtype]) != _FINITE_MASK[case.dtype]
            stored = decode_bits(bits.astype(np.uint16), case.dtype)
            decoded = np.where(valid, stored, 0.0) / scale
            targets = decode_bits(center.astype(np.uint16), case.dtype) / scale
        for row, query in enumerate(queries):
            first_key, last_key = legal_interval(case.base, query)
            legal = max(0, last_key - first_key + 1)
            for head in range(case.base.nhead_q):
                group = by_group.setdefault((legal, head), _GroupBuilder(labels={}))
                cell = (slice(None), row, head)
                row_labels = labels[cell]
                group.coordinates += row_labels.size
                group.structural += int(np.count_nonzero(zero[cell]))
                nonzero = ~zero[cell]
                group.compared += int(np.count_nonzero(nonzero))
                hits = nonzero & (bits[cell] == center[cell])
                group.exact += int(np.count_nonzero(hits))
                two = nonzero & (high[cell] > low[cell])
                group.ambiguous += int(np.count_nonzero(two))
                upper = two & (bits[cell] == high[cell])
                group.ambiguous_high += int(np.count_nonzero(upper))
                for label in row_labels.ravel():
                    if label != P4Label.PASS.value:
                        group.labels[label] = group.labels.get(label, 0) + 1
                if decoded is not None:
                    stored = float(np.max(np.abs(
                        decoded[cell].sum(axis=-1) - targets[cell].sum(axis=-1)
                    )))
                    real = float(np.max(np.abs(decoded[cell].sum(axis=-1) - 1.0)))
                    group.max_row_sum_stored = max(group.max_row_sum_stored, stored)
                    group.max_row_sum_real = max(group.max_row_sum_real, real)
        for label in P4Label:
            hits = labels == label.value
            count = int(np.count_nonzero(hits))
            totals[label.value] += count
            if count and label is not P4Label.PASS and label.value not in first:
                batch, row, head, dim = (int(value) for value in np.argwhere(hits)[0])
                first[label.value] = (exposure, (batch, queries[row], head, dim))
        nonzero = ~zero
        compared += int(np.count_nonzero(nonzero))
        exact += int(np.count_nonzero(nonzero & (bits == center)))
        two = nonzero & (high > low)
        ambiguous += int(np.count_nonzero(two))
        ambiguous_high += int(np.count_nonzero(two & (bits == high)))

    # Most frequent label first: it names the failure; rarer labels at the
    # same time are coordinates another explanation also fits.
    failing = sorted(
        (label for label in P4_FAILURE_LABELS if totals[label.value]),
        key=lambda label: -totals[label.value],
    )
    violations = [
        f"{label.value}={totals[label.value]} first at {first[label.value][0]} "
        f"b:q:h:d={':'.join(str(value) for value in first[label.value][1])}"
        for label in failing
    ]
    if compared == 0:
        violations.append("no_coordinate_compared")
    return P4Result(
        mutation=selected,
        violations=tuple(violations),
        labels=MappingProxyType(totals),
        first=MappingProxyType(first),
        compared=compared,
        exact=exact,
        ambiguous=ambiguous,
        ambiguous_high=ambiguous_high,
        groups=MappingProxyType(
            {
                exposure: MappingProxyType(
                    {key: item.freeze() for key, item in sorted(by_group.items())}
                )
                for exposure, by_group in groups.items()
            }
        ),
    )


def format_p4_result(result: P4Result) -> str:
    violations = "; ".join(result.violations) if result.violations else "none"
    return (
        f"violations={violations}; exact={result.exact}/{result.compared} "
        f"({result.exact_rate:.6f}); two_value_B={result.ambiguous} "
        f"(upper observed {result.ambiguous_high})"
    )


@dataclass(frozen=True)
class P4Diagnostics:
    """Records-only evidence (never a verdict) on sampled rows.

    ``path_exact`` counts coordinates equal to the pinned emulation (lane
    order, ``E = RN(2**x)``, RN MFMA): agreement everywhere is consistent with
    E1 = the nearest exponential. ``mfma_rn`` counts folded coordinates with a
    genuine MFMA addition that lie in the RN-accumulate Tier B (E6)."""

    path_exact: int
    path_compared: int
    mfma_additions: int
    mfma_rn: int
    widths: Mapping[int, int]


def p4_diagnostics(
    case: P4Case,
    observation: UniformObservation,
    *,
    rows: Iterable[int] = P4_SEPARATION_ROWS,
) -> P4Diagnostics:
    path_exact = path_compared = additions = rn = 0
    widths: dict[int, int] = {}
    queries = tuple(rows)
    for exposure, raw in observation.exposures:
        low, high = tier_b_exposure(case, exposure)
        width = (high.astype(np.int64) - low.astype(np.int64) + 1)[:, queries]
        for value, count in zip(*np.unique(width, return_counts=True)):
            widths[int(value)] = widths.get(int(value), 0) + int(count)
        for query in queries:
            for head in range(case.base.nhead_q):
                observed = raw[0, query, head]
                pinned = emulate_row(case, exposure, query, head)
                path_exact += int(np.count_nonzero(observed == pinned))
                path_compared += observed.size
                if exposure != "folded":
                    continue
                rn_low, rn_high = tier_b_row(case, exposure, query, head, mfma="rn")
                for dim in range(case.base.head_dim):
                    if _genuine_addition(case, query, head, dim):
                        additions += 1
                        rn += int(rn_low[dim] <= observed[dim] <= rn_high[dim])
    return P4Diagnostics(
        path_exact,
        path_compared,
        additions,
        rn,
        MappingProxyType(dict(sorted(widths.items()))),
    )


def p4_records(
    case: P4Case,
    candidate_name: str,
    results: Mapping[str, P4Result],
    *,
    deterministic: bool,
    diagnostics: P4Diagnostics,
    separations: Mapping[str, Separation],
) -> list[dict]:
    """JSON-ready calibration records: per-group evidence for every analysis,
    one summary per analysis, and one diagnostics record for the case."""
    identity = {
        "arch": "gfx942",
        "candidate_name": candidate_name,
        "case_id": case.case_id,
        "fixture": case.fixture.value,
        "dtype": case.dtype,
        "mask": case.mask,
    }
    records: list[dict] = []
    for name, result in results.items():
        for exposure, by_group in result.groups.items():
            for (legal, head), stats in by_group.items():
                records.append(
                    {
                        "record_type": "rocke_p4_group",
                        **identity,
                        "mutation": name,
                        "exposure": exposure,
                        "legal_count": legal,
                        "head": head,
                        "head_scale_exponent": case.head_scale_exponents[head],
                        "coordinates": stats.coordinates,
                        "structural": stats.structural,
                        "compared": stats.compared,
                        "exact": stats.exact,
                        "two_value_b": stats.ambiguous,
                        "two_value_b_upper": stats.ambiguous_high,
                        "labels": dict(stats.labels),
                        "max_row_sum_stored": stats.max_row_sum_stored,
                        "max_row_sum_real": stats.max_row_sum_real,
                    }
                )
        separation = separations.get(name)
        records.append(
            {
                "record_type": "rocke_p4_summary",
                **identity,
                "mutation": name,
                "rejected": bool(result.violations),
                "violations": list(result.violations),
                "labels": dict(result.labels),
                "compared": result.compared,
                "exact": result.exact,
                "two_value_b": result.ambiguous,
                "two_value_b_upper": result.ambiguous_high,
                "cpu_gap": None if separation is None else separation.gap,
                "cpu_gap_at": None
                if separation is None or separation.coordinate is None
                else [separation.exposure, list(separation.coordinate)],
            }
        )
    records.append(
        {
            "record_type": "rocke_p4_diagnostics",
            **identity,
            "deterministic": deterministic,
            "assumptions": list(P4_ASSUMPTIONS),
            "path_exact": diagnostics.path_exact,
            "path_compared": diagnostics.path_compared,
            "mfma_additions": diagnostics.mfma_additions,
            "mfma_rn": diagnostics.mfma_rn,
            "tier_b_widths": {
                str(key): value for key, value in diagnostics.widths.items()
            },
        }
    )
    return records


# ---------------------------------------------------------------------------
# Calibration: identity and scope only (D8, invariant I6).
# ---------------------------------------------------------------------------

P4_POLICY = "tier_b"


@dataclass(frozen=True)
class P4CalibrationEntry:
    """Ratifies that the Tier B contract applies to one kernel identity.

    It names the policy, the assumption versions, the case ids and the dispatch
    fingerprint, and has no width, ULP or tolerance field: no measurement can
    enlarge Tier B. Provenance belongs in the commit that edits the table."""

    policy: str
    assumptions: tuple[str, ...]
    case_ids: tuple[str, ...]
    spec: UniformSpecFingerprint

    def __post_init__(self) -> None:
        if self.policy != P4_POLICY:
            raise ValueError(f"P4 supports only policy={P4_POLICY!r}")
        if tuple(self.assumptions) != P4_ASSUMPTIONS:
            raise ValueError(f"assumptions must be {P4_ASSUMPTIONS}")
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case_ids must be nonempty and unique")

    def mismatch(self, case: P4Case, spec: UniformSpecFingerprint) -> tuple[str, ...]:
        problems: list[str] = []
        if case.case_id not in self.case_ids:
            problems.append(f"case {case.case_id} not in calibrated {self.case_ids}")
        for item in fields(UniformSpecFingerprint):
            calibrated = getattr(self.spec, item.name)
            resolved = getattr(spec, item.name)
            if calibrated != resolved:
                problems.append(
                    f"{item.name}: calibrated {calibrated!r}, resolved {resolved!r}"
                )
        return tuple(problems)


# Empty until gfx942 ratification (Phase 5): the GPU lane fails closed.
P4_CALIBRATION: Mapping[tuple[str, str, str], P4CalibrationEntry] = MappingProxyType({})

_P4_RECALIBRATION_REMEDY = (
    "re-run the case on gfx942 with ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>, "
    "review the records (zero coordinates outside Tier B, every mutation "
    "rejected, deterministic), and update P4_CALIBRATION (spec, case_ids) "
    "deliberately"
)


def require_p4_calibration(
    arch: str,
    candidate_name: str,
    case: P4Case,
    spec: UniformSpecFingerprint,
    *,
    table: Mapping[tuple[str, str, str], P4CalibrationEntry] = P4_CALIBRATION,
    detail: str = "",
) -> P4CalibrationEntry:
    """Return the entry that authorizes this resolved case, or fail closed."""
    key = (arch, candidate_name, case.dtype)
    suffix = f"; {detail}" if detail else ""
    entry = table.get(key)
    if entry is None:
        raise AssertionError(
            f"no P4 calibration for {key}, case={case.case_id}; "
            f"{_P4_RECALIBRATION_REMEDY}{suffix}"
        )
    problems = entry.mismatch(case, spec)
    if problems:
        raise AssertionError(
            f"P4 calibration {key} does not cover case={case.case_id}: "
            f"{'; '.join(problems)}; {_P4_RECALIBRATION_REMEDY}{suffix}"
        )
    return entry
