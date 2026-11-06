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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    _basis_offset,
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


@cache
def exp2_interval(argument: Fraction) -> F32Interval:
    """Hull of :func:`exp2_candidates`; the candidates are consecutive FP32 values."""
    return F32Interval.hull(exp2_candidates(argument))


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
# What the subject adapter passes as ``scale`` and the dense kernel's LOG2E.
_SCALE_ARGUMENT = 1.0 / 8.0
_KERNEL_LOG2E = 1.4426950408889634
SCORE_SCALE = round_f32(f32(_SCALE_ARGUMENT) * f32(_KERNEL_LOG2E))
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


def is_distinguished(case: P4Case, query: int, key: int) -> bool:
    """Whether ``key`` carries the row's nonzero score (legality aside)."""
    if (key + case.residue_shift - query) % P4_PERIOD:
        return False
    return not (case.fixture is P4Fixture.LATE_MAX and key < case.late)


def score_numerator(case: P4Case, head: int, query: int, key: int) -> Fraction:
    """The exact QK dot product ``acc`` (lemma B1: every partial sum is exact)."""
    if not is_distinguished(case, query, key):
        return Fraction(0)
    product = case.q_scales[head] * case.k_scales[head]
    if case.fixture is P4Fixture.WALSH:
        return P4_PERIOD * product * (-1 if key & 1 else 1)
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


def visited_tiles(case: P4Case, query: int) -> range:
    """KV tiles the dense kernel visits for ``query``, in loop order.

    All rows of a ``block_m`` query block share one KV loop: top-left causal
    clamps it at the block's diagonal tile, and a sliding window starts it
    ``W / block_n`` tiles before the block's first diagonal tile. A row can
    therefore visit fully masked tiles before or after its legal keys."""
    per_block = case.block_m // case.block_n
    block = query // case.block_m
    tiles = case.base.seqlen_k // case.block_n
    upper = min(block * per_block + per_block, tiles)
    lower = 0
    if case.mask == "window":
        lower = max(block * per_block - case.window // case.block_n, 0)
    return range(lower, upper)


@cache
def _score_of(numerator: Fraction) -> Fraction:
    return round_f32(numerator * SCORE_SCALE, "rn")


def score(case: P4Case, head: int, query: int, key: int) -> Fraction:
    """The FP32 score in the log2 domain, or the masked value."""
    first, last = legal_interval(case.base, query)
    if not first <= key <= last:
        return MASKED_SCORE
    return _score_of(score_numerator(case, head, query, key))


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
def row_trace(case: P4Case, head: int, query: int) -> RowTrace:
    """Propagate the row's running max, weights and denominator.

    The running max is exact (``max`` does not round). Each ``v_exp_f32`` is its
    candidate interval; each tile's 64 weights are summed in any order
    (``gamma_63``), then ``l = RN(RN(l alpha) + l_tile)``."""
    running_max = MASKED_SCORE
    denominator = F32Interval.point(Fraction(0))
    steps: list[TileStep] = []
    for tile in visited_tiles(case, query):
        start = tile * case.block_n
        scores = [score(case, head, query, key) for key in range(start, start + case.block_n)]
        new_max = max(running_max, max(scores))
        alpha = exp2_interval(round_f32(running_max - new_max, "rn"))
        weights = tuple(exp2_interval(round_f32(value - new_max, "rn")) for value in scores)
        denominator = add_rn(mul_rn(denominator, alpha), sum_any_order(weights))
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
) -> F32Interval:
    """``RN(o R(l))`` over the reachable accumulator ``o``.

    Per tile ``o = MFMA(RN(o alpha), product)``. With one nonzero product per
    output per tile, the MFMA performs at most one real addition; it is exact
    when the accumulator is zero, else faithful (or RN when pinned)."""
    if mfma not in MFMA_ROUNDING:
        raise ValueError(f"mfma must be one of {MFMA_ROUNDING}")
    trace = row_trace(case, head, query)
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
    case: P4Case, exposure: str, query: int, head: int, *, mfma: str = "faithful"
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest reachable output bits of one row and query head."""
    trace = row_trace(case, head, query)
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
        output = _output_interval(case, head, query, tuple(products), mfma)
        low[dim], high[dim] = cast_bits_hull(output, case.dtype)
    return low, high


def tier_b_bounds(
    case: P4Case,
    exposure: str,
    *,
    rows: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    mfma: str = "faithful",
) -> tuple[np.ndarray, np.ndarray]:
    """Tier B for one exposure: every reachable stored value of each coordinate
    lies, in bit order, between the two returned arrays (inclusive)."""
    low = np.zeros(case.output_shape, dtype=np.uint16)
    high = np.zeros(case.output_shape, dtype=np.uint16)
    for query in range(case.base.seqlen_q) if rows is None else rows:
        for head in range(case.base.nhead_q) if heads is None else heads:
            row_low, row_high = tier_b_row(case, exposure, query, head, mfma=mfma)
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
) -> np.ndarray:
    """Output bits of one row from an FP32 emulation of the dense path.

    Without ``rng`` this is the pinned realization: the documented lane order,
    ``E = RN(2**x)`` and an RN MFMA accumulate. With ``rng`` it draws a random
    summation tree per tile, an ``E`` candidate per call and a directed
    rounding per MFMA addition: a member of the reachable set, used to test
    that Tier B encloses it."""
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
    for tile in visited_tiles(case, query):
        start = tile * case.block_n
        scores = np.array(
            [float(score(case, head, query, key)) for key in range(start, start + case.block_n)],
            dtype=np.float32,
        )
        new_max = np.float32(max(running_max, np.max(scores)))
        alpha = exp2(np.float32(running_max - new_max))
        weights = [exp2(np.float32(value - new_max)) for value in scores]
        if rng is None:
            halves = []
            for order in _LANE_ORDER:
                total = np.float32(0.0)
                for index in order:
                    total = np.float32(total + weights[index])
                halves.append(total)
            tile_sum = np.float32(halves[0] + halves[1])
        else:
            items = list(weights)
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
def tier_b_exposure(case: P4Case, exposure: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached, read-only :func:`tier_b_bounds` for a whole exposure."""
    low, high = tier_b_bounds(case, exposure)
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
