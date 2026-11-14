# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exact FP32 reachable-set primitives for the structured-logit check.

The check predicts, for every exposed output coordinate, the set of stored values that
the pinned dense-kernel arithmetic can produce. The set is built by
propagating closed intervals of FP32 values through each step of the path, so
it encloses every reachable state rather than one chosen realization.

Everything here is exact: values are :class:`fractions.Fraction`, rounding to
FP32 is computed from the exact rational, and the transcendental quantities
(``2**x`` behind ``v_exp_f32``, ``e**c`` behind the targets) are enclosed by
series brackets with explicit tail bounds, in rational arithmetic. There is no
attention implementation in this module; its inputs are closed-form scalars.

Soundness rests on monotonicity. On nonnegative arguments, round-to-nearest,
round-down, round-up, ``+``, ``*`` and the FP16/BF16 cast are nondecreasing in
each argument, and ``x -> 1/x`` is nonincreasing, so the images of interval
endpoints bound the image of every point inside.

The path follows the kernel's LLVM IR, with one versioned exception (A-F v1):
affected AMDGPU compilers store some FP16 outputs as the exact product rounded
once, instead of the IR's FP32 product rounded again to FP16.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from functools import cache
from types import MappingProxyType

import numpy as np
from rocke.numeric.bf16 import f32_to_bf16

from ._structured_shapes import (
    BASIS_OFFSETS,
    QUICK_SHAPES,
    DenseShape,
    MaskMutation,
    Observation,
    SpecFingerprint,
    basis_offset,
    validate_observation,
    decode_bits,
    legal_interval,
    target_bits,
    exposure_values,
)

F32_FRACTION_BITS = 23
F32_MIN_EXPONENT = -126
F32_MAX_EXPONENT = 127
F32_UNIT_ROUNDOFF = Fraction(1, 1 << 24)
F32_SMALLEST_SUBNORMAL = Fraction(1, 1 << 149)
F32_SMALLEST_NORMAL = Fraction(1, 1 << 126)
_F32_OVERFLOW = Fraction(1 << (F32_MAX_EXPONENT + 1))
# Strictly below every FP32 spacing, so ``x +- _NUDGE`` never crosses two grid
# points; used to step to a neighbor.
_NUDGE = Fraction(1, 1 << 300)

_ROUNDING_MODES = frozenset({"rn", "rd", "ru", "rz"})


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

    ``value`` is any real (rational) number, not only an FP32 value: for
    ``r > 0`` the result is ``2**(max(floor(log2 r), -126) - 23)``, the
    distance between consecutive FP32 values in the cell ``[2**e, 2**(e+1))``
    that holds ``r``, with the subnormal range counted in the first normal
    cell; for ``r = 0`` it is ``2**-149``. Equivalently it is
    ``next_up(f) - f`` for ``f = round_f32(r, "rd")``.

    For a power of two this is the spacing above it (twice the spacing below),
    which is the wider of the two conventions for "one ULP of ``value``"."""
    magnitude = abs(value)
    if magnitude == 0:
        return F32_SMALLEST_SUBNORMAL
    exponent = max(floor_log2(magnitude), F32_MIN_EXPONENT)
    return Fraction(2) ** (exponent - F32_FRACTION_BITS)


def round_f32(value: Fraction, mode: str = "rn") -> Fraction:
    """Round an exact rational to FP32 and return the result's exact value.

    ``mode`` is ``"rn"`` (nearest, ties to even), ``"rd"`` (toward -inf),
    ``"ru"`` (toward +inf) or ``"rz"`` (toward zero; no part of the path model,
    used only by the assumption-violating emulations). Subnormals are kept
    (gradual underflow). Overflow raises: no quantity of the check comes near the FP32
    range limit."""
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
    elif mode == "rz":
        count = int(steps)  # Fraction.__trunc__ rounds toward zero
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
    """Image of the mathematical map ``l -> RN(1 / l)`` on ``l > 0``.

    These are the IR semantics of the dense epilogue's ``fdiv 1.0, l`` (no
    ``!fpmath``). That the executed division returns this value is assumption
    A-DIV, not a property of this function: see :func:`rcp_a_div`."""
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
    The one-addition error model behind the bound holds when the rounded sum
    is normal, and also for a subnormal sum under gradual underflow, where the
    addition is exact. It does not hold for a subnormal sum under a flushing
    mode, so the row recurrence applies this function only to a tile that
    holds a normal-scale term (:func:`tile_sum`). The result is itself FP32,
    so the real bounds are snapped inward: up for the lower end, down for the
    upper end."""
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
    return _exact_target_bits(value.lo, dtype), _exact_target_bits(value.hi, dtype)


def _exact_target_bits(value: Fraction, dtype: str) -> int:
    """FP16/BF16 RNE bits of an exact rational, rounded once."""
    return target_bits(value.numerator, value.denominator, dtype)


# ---------------------------------------------------------------------------
# The output epilogue (assumption A-F).
#
# The IR stores ``fptrunc(fmul(o, R))``: the product rounded to FP32, then to
# the subject dtype. Affected AMDGPU LLVM revisions select some of these FP16
# sites as ``v_fma_mix{lo,hi}_f16 R, o, 0``, which rounds the exact product
# once to FP16, without the ``contract`` flag that permits it; a compiler that
# requires ``contract`` for the fold selects none. gfx942 has no BF16 form, and
# no other operation is fused: the denominator combine stays a separate FP32
# multiply and add in every inspected code object (the IR contract and the ISA
# contract in ``test_structured_contracts.py``).
# ---------------------------------------------------------------------------

EPILOGUE_ASSUMPTION = (
    "A-F v1: each FP16 output store is q(RN32(o R)) or q(o R); every other "
    "operation, BF16 stores included, follows the IR"
)
_SINGLE_ROUNDING_DTYPES = frozenset({"fp16"})


# Cached: a row's dimensions share a few accumulator intervals, and Tier B
# evaluates this once per coordinate.
@cache
def cast_single_or_double_hull(
    accumulator: F32Interval, reciprocal: F32Interval, dtype: str
) -> tuple[int, int]:
    """Stored bits of the output epilogue ``q_T(o R)`` under A-F v1.

    For every ``o`` in ``accumulator`` and ``R`` in ``reciprocal`` the IR stores
    ``q_T(RN32(o R))`` (double rounding); for FP16 an affected compiler may
    store ``q_T(o R)`` instead (the exact product rounded once). Both maps are
    nondecreasing in ``o`` and ``R`` on nonnegative arguments, so the hull of
    their endpoint images holds every value either can store. BF16 keeps the
    double-rounded image only."""
    accumulator._require_nonnegative()
    reciprocal._require_nonnegative()
    low, high = cast_bits_hull(mul_rn(accumulator, reciprocal), dtype)
    if dtype not in _SINGLE_ROUNDING_DTYPES:
        return low, high
    single_low = _exact_target_bits(accumulator.lo * reciprocal.lo, dtype)
    single_high = _exact_target_bits(accumulator.hi * reciprocal.hi, dtype)
    return min(low, single_low), max(high, single_high)


@dataclass(frozen=True)
class EpilogueArtifact:
    """A compiled pair of quick kernels whose FP16 epilogue was read from ISA.

    ``single_lanes`` are the output dimensions ``d mod 4`` (the element of each
    four-half store) that the FP16 code object computes with
    ``v_fma_mix{lo,hi}_f16``, single rounding; the other elements are an FP32
    multiply then ``v_cvt_f16_f32``. Neither BF16 code object has a mixed
    instruction. Each code object is named by the SHA-256 of the HSACO that
    ``run_attention_dense_torch`` compiles, so a record can say which one ran.

    The digest covers the whole code object, so it names one kernel revision
    under one compiler: any change to the dense builder's IR gives a new
    digest, which matches no artifact until it is recorded again.
    ``llvm_major`` lets :func:`epilogue_artifacts_for_llvm` tell that case
    (a recorded compiler, an unrecorded digest) from an unrecorded compiler."""

    compiler: str
    llvm_major: int
    code_objects: tuple[tuple[str, str], ...]  # (dtype, HSACO sha256)
    single_lanes: frozenset[int]
    evidence: str


# Only an artifact named here is emulated exactly (``artifact_exact``); the
# verdict never depends on this table: Tier B allows either rounding at every
# FP16 site.
STRUCTURED_EPILOGUE_ARTIFACTS: Mapping[str, EpilogueArtifact] = MappingProxyType(
    {
        "rocm-7.13.0": EpilogueArtifact(
            compiler=(
                "ROCm 7.13.0 COMGR, AMD clang 23.0.0git ROCm/llvm-project "
                "43215c73116c407735c85a180d174f718798c328"
            ),
            llvm_major=23,
            code_objects=(
                (
                    "fp16",
                    "ba92c6283870841e1e8405ac07ea04f294a25c15e8704e9a493a46807f43ac3d",
                ),
                (
                    "bf16",
                    "83204d7a008414796a83d860422f2cd41064d509ad33dee8ad81f9536a72c661",
                ),
            ),
            single_lanes=frozenset({0, 1}),
            evidence="ISA, and every coordinate of the gfx942 captures",
        ),
        "rocm-7.1.0": EpilogueArtifact(
            compiler=(
                "ROCm 7.1.0 COMGR, AMD LLVM 20.0.0git ROCm/llvm-project "
                "1b0eada6b0ee93e2e694c8c146d23fca90bc11c5"
            ),
            llvm_major=20,
            code_objects=(
                (
                    "fp16",
                    "b6184c4035a00ca49af86f90215b70f06c9fbce01dcaa95ceeeca7fbc866dd06",
                ),
                (
                    "bf16",
                    "b8b5a1b2b5a9f632f7c947468b421affb6abf25a1646c2676f16a91e7401305a",
                ),
            ),
            single_lanes=frozenset({0, 3}),
            evidence="ISA only; never launched",
        ),
    }
)


def epilogue_artifact_for(dtype: str, code_object_sha256: str) -> str | None:
    """The named artifact whose ``dtype`` code object has this digest, if any."""
    for name, artifact in STRUCTURED_EPILOGUE_ARTIFACTS.items():
        if (dtype, code_object_sha256) in artifact.code_objects:
            return name
    return None


def epilogue_artifacts_for_llvm(llvm_major: int) -> tuple[str, ...]:
    """The named artifacts recorded under a compiler of this LLVM major. A
    code object of such a compiler that :func:`epilogue_artifact_for` does not
    name means the recorded digests are out of date for it."""
    return tuple(
        name
        for name, artifact in STRUCTURED_EPILOGUE_ARTIFACTS.items()
        if artifact.llvm_major == llvm_major
    )


# ---------------------------------------------------------------------------
# Certified enclosures of ``e**x`` and ``ln 2``.
#
# Both transcendental quantities of the check are enclosed by a partial sum
# of a series plus an explicit bound of its tail. The sums are evaluated in
# fixed point with directed rounding, so every rounding moves an endpoint
# outward and no floating-point or decimal library function is trusted. A
# number of correct decimal digits is not an error bound; an enclosure is.
# ---------------------------------------------------------------------------

# Fraction bits of the fixed-point series, and the relative width every
# enclosure must stay under.
_SERIES_BITS = 320
_ENCLOSURE_BITS = 192


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _exp_enclosure(exponent: Fraction) -> tuple[Fraction, Fraction]:
    """Rational ``(lo, hi)`` with ``lo <= e**x <= hi`` for a rational ``x``.

    For ``x >= 0`` and ``N + 1 > x`` the partial sum
    ``S_N = sum_{n < N} x**n / n!`` satisfies
    ``0 < e**x - S_N <= (x**N / N!) (N + 1) / (N + 1 - x)``: the tail is
    bounded by a geometric series of ratio ``x / (N + 1)``. Terms are rounded
    down for the lower end and up for the upper end. ``e**-x`` is the
    reciprocal interval. The relative width is at most
    ``2**-_ENCLOSURE_BITS``; that postcondition is checked on return."""
    exponent = Fraction(exponent)
    if exponent < 0:
        low, high = _exp_enclosure(-exponent)
        return 1 / high, 1 / low
    one = 1 << _SERIES_BITS
    scaled = exponent * one
    x_low = scaled.numerator // scaled.denominator
    x_high = _ceil_div(scaled.numerator, scaled.denominator)
    term_low = term_high = one  # x**0 / 0!
    total_low = total_high = 0
    count = 0  # terms summed so far; the term bounds are those of n = count
    while True:
        remaining = (count + 1) * one - x_high
        if remaining > 0:
            tail = _ceil_div(term_high * (count + 1) * one, remaining)
            if tail << _ENCLOSURE_BITS + 2 <= total_low:
                break
        total_low += term_low
        total_high += term_high
        count += 1
        term_low = term_low * x_low // (count * one)
        term_high = _ceil_div(term_high * x_high, count * one)
    low, high = Fraction(total_low, one), Fraction(total_high + tail, one)
    if (high - low) * (1 << _ENCLOSURE_BITS) > low:
        raise ArithmeticError(f"e**{float(exponent)!r} is not enclosed tightly enough")
    return low, high


def _ln2_enclosure() -> tuple[Fraction, Fraction]:
    """Rational ``(lo, hi)`` with ``lo <= ln 2 <= hi``, from
    ``ln 2 = sum_{n >= 1} 1 / (n 2**n)``, whose tail after ``N`` terms is
    below ``1 / ((N + 1) 2**N)``."""
    one = 1 << _SERIES_BITS
    low = high = 0
    for n in range(1, _SERIES_BITS + 1):
        low += one // (n << n)
        high += _ceil_div(one, n << n)
    high += _ceil_div(one, (_SERIES_BITS + 1) << _SERIES_BITS)
    return Fraction(low, one), Fraction(high, one)


_LN2 = _ln2_enclosure()


# ---------------------------------------------------------------------------
# v_exp_f32 candidate sets (assumption A-E).
#
# The documented accuracy of the hardware base-2 exponential is "1 ULP" for
# FP32 (LLVM AMDGPU usage notes, ``llvm.exp2``). The convention for "one ULP"
# is not stated, so A-E takes the wider one: |E(x) - 2**x| <= f32_quantum(2**x),
# the spacing of the FP32 cell that holds the real number 2**x, measured
# upward. At a power of two this admits four values, because the spacing
# below a power of two is half the spacing above it.
# ---------------------------------------------------------------------------

EXP2_ASSUMPTION = "A-E v1: |E(x) - 2**x| <= f32_quantum(2**x)"

# Below this argument 2**x is far under the smallest subnormal, and the
# candidate set is decided by the subnormal spacing alone.
_EXP2_UNDERFLOW_ARGUMENT = Fraction(-300)
_EXP2_UNDERFLOW_ENCLOSURE = (Fraction(1, 1 << 1000), Fraction(1, 1 << 200))


def _exp2_enclosure(argument: Fraction) -> tuple[Fraction, Fraction]:
    """Rational bracket ``[lo, hi]`` containing ``2**argument``.

    Exact for an integer argument. Otherwise ``2**x = 2**k e**(f ln 2)`` with
    ``k = floor(x)`` and ``0 < f < 1``, and both ``ln 2`` and the exponential
    are enclosed by their series, so the bracket is certified: its relative
    width is below ``2**-190`` and no library function is trusted."""
    # Range checks first: large FP32 magnitudes are integers, and 2**(-1e30)
    # must not be formed exactly.
    if argument < _EXP2_UNDERFLOW_ARGUMENT:
        return _EXP2_UNDERFLOW_ENCLOSURE
    if argument > F32_MAX_EXPONENT + 1:
        raise OverflowError("2**argument overflows FP32")
    if argument.denominator == 1:
        exact = Fraction(2) ** argument.numerator
        return exact, exact
    whole = argument.numerator // argument.denominator
    fraction = argument - whole
    scale = Fraction(2) ** whole
    return (
        _exp_enclosure(fraction * _LN2[0])[0] * scale,
        _exp_enclosure(fraction * _LN2[1])[1] * scale,
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


class StructuredScopeError(ArithmeticError):
    """A row leaves the domain on which the enclosure is derived.

    The enclosure is then undefined rather than widened: the correct quick
    matrix must never raise this, and a mutated row that does makes no claim."""


# ---------------------------------------------------------------------------
# The reciprocal (assumption A-DIV).
#
# The IR is ``fdiv float 1.0, l`` without fast-math flags or ``!fpmath``, whose
# IR semantics are ``RN(1 / l)``. gfx942 has no FP32 divide instruction: the
# backend expands the ``fdiv`` into a scaled ``v_rcp_f32`` refinement sequence
# ending in ``v_div_fixup_f32``, and no cited source states that the sequence
# is correctly rounded. A-DIV names that premise on the range the denominator
# is shown to reach on the quick matrix; outside it the enclosure is not
# derived.
#
# The stored bits cannot show a wrong reciprocal (a neighbor of ``RN(1 / l)``
# leaves Tier B nowhere: ``STRUCTURED_UNOBSERVED_VIOLATIONS``), so passing
# launches are no evidence for A-DIV. The assumption is a statement about one
# instruction sequence, and the ISA contract pins it: a compiled quick kernel
# must hold exactly that expansion, and start with FP32 denormals kept.
# ---------------------------------------------------------------------------

DIV_ASSUMPTION = "A-DIV v1: FDIV(1, l) = RN(1 / l) for every FP32 l in [1/2, 2**9)"
A_DIV_DOMAIN = (Fraction(1, 2), Fraction(1 << 9))  # closed below, open above
# A structural zero does not need A-DIV. A coordinate that selects no legal key
# has an accumulator of exactly +0, or, after a fully masked leading tile, a
# residue of at most 2**(h - 149). It stores +0 for any reciprocal below this
# bound, correctly rounded or not: 2**(3 - 149 + 11) = 2**-135 is under half
# the smallest positive BF16 value, and BF16 is the only quick kernel with such
# a tile (FP16 would allow 2**120).
STRUCTURAL_ZERO_RECIPROCAL_BOUND = Fraction(1 << 11)


def rcp_a_div(denominator: F32Interval) -> F32Interval:
    """Image of the executed reciprocal ``fdiv 1.0, l`` under A-DIV v1.

    Raises :class:`StructuredScopeError` when the denominator interval leaves the
    A-DIV domain, so no reciprocal is modeled where the assumption says
    nothing."""
    low, high = A_DIV_DOMAIN
    if not (low <= denominator.lo and denominator.hi < high):
        raise StructuredScopeError(
            f"denominator [{float(denominator.lo)!r}, {float(denominator.hi)!r}] "
            f"is outside the A-DIV domain [{float(low)!r}, {float(high)!r})"
        )
    return rcp_rn(denominator)


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
_FINITE_MASK = {"fp16": 0x7C00, "bf16": 0x7F80}
_SIGN_BIT = 0x8000


def target_rank(bits: int, dtype: str) -> int:
    """Rank of a nonnegative finite FP16/BF16 value: its 16-bit encoding.

    With the sign bit clear the encoding order is the numerical order, so
    ``+0`` has rank 0 and consecutive values have consecutive ranks, across a
    binade edge and across the subnormal/normal transition alike. A negative
    value (``-0`` included), an infinity and a NaN have no rank."""
    bits = int(bits)
    finite = _FINITE_MASK[dtype]
    if not 0 <= bits <= 0xFFFF or bits & _SIGN_BIT or bits & finite == finite:
        raise ValueError(f"{bits:#06x} is not a nonnegative finite {dtype} value")
    return bits


def d_ulp(first: int, second: int, dtype: str) -> int:
    """Target-ULP distance: the rank difference of two FP16/BF16 values.

    Every count of target ULPs in this check (the Tier A radius, the separation
    margins, the head-scale search) is a value of this distance; no real-valued
    ULP size is used."""
    return abs(target_rank(first, dtype) - target_rank(second, dtype))


def ordered_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Smallest :func:`d_ulp` between members of two rank intervals; 0 if they
    meet. Each argument is the ``(low, high)`` bits of nonnegative finite
    values, so the gap is the rank difference of the nearest endpoints."""
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0


# ---------------------------------------------------------------------------
# Fixtures. Every row has two logit classes: the distinguished keys, with
# logit ``c``, and the background keys, with logit 0. Each head carries its own
# ``c = a_h b_h / sqrt(D)`` (``8 a_h b_h`` times a row sign for the Walsh
# contraction), so a kernel that reads the wrong Q or K head changes a logit.
# ---------------------------------------------------------------------------

STRUCTURED_PERIOD = 64
# The softmax scale ``1/sqrt(64)`` the subject adapter passes, and ``log2 e``
# rounded once to FP32 from the ``ln 2`` enclosure (both ends of ``1 / ln 2``
# round to the same value). Both come from the definition of attention, not
# from the kernel: a contract checks the kernel's constant against this one,
# so a wrong kernel constant fails instead of being copied.
_SCALE_ARGUMENT = 1.0 / 8.0
LOG2E_F32 = round_f32(1 / _LN2[1], "rn")
if round_f32(1 / _LN2[0], "rn") != LOG2E_F32:
    raise ArithmeticError("log2(e) is too close to an FP32 rounding boundary")
SCORE_SCALE = round_f32(f32(_SCALE_ARGUMENT) * LOG2E_F32)
# The kernel's masked score and initial row max (pinned by the IR contract).
# Spec masking excludes the key; Tier B inside Tier A shows that -1e30 stores
# the same exact zeros (its weight is at most 2**-149, cast to +0).
MASKED_SCORE = f32(-1e30)


class StructuredFixture(str, Enum):
    """Structured Q/K profiles; V is one of the basis or folded exposures."""

    IDENTITY = "identity"  # Q and K one-hot at the key's period residue
    CYCLIC = "cyclic"  # K one-hot shifted by r
    WALSH = "walsh"  # dense signed Walsh rows, one nonzero score per row
    LATE_MAX = "late_max"  # identity with K rows below L zeroed: the max arrives late


_FIXTURE_SHIFT = {
    StructuredFixture.IDENTITY: 0,
    StructuredFixture.LATE_MAX: 0,
    StructuredFixture.WALSH: 1,
}


def _is_signed_power_of_two(value: Fraction) -> bool:
    magnitude = abs(value)
    return magnitude > 0 and (
        (
            magnitude.denominator == 1
            and magnitude.numerator & (magnitude.numerator - 1) == 0
        )
        or (
            magnitude.numerator == 1
            and magnitude.denominator & (magnitude.denominator - 1) == 0
        )
    )


def _scale_token(value: Fraction) -> str:
    if value == 0:
        return "0"
    sign = "n" if value < 0 else "p"
    return f"{sign}{floor_log2(abs(value))}"


@dataclass(frozen=True)
class StructuredCase:
    """One launch configuration: shape, fixture, per-head scales, KV tiling.

    ``q_scales[h]`` and ``k_scales[h]`` are ``a_h`` and ``b_h``: zero or signed
    powers of two, exact in FP16 and BF16. ``block_m``/``block_n`` are the
    subject's tile sizes, which fix the KV tiles each row visits."""

    base: DenseShape
    fixture: StructuredFixture
    q_scales: tuple[Fraction, ...]
    k_scales: tuple[Fraction, ...]
    shift: int = 0
    late: int = 0
    block_m: int = 128
    block_n: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture", StructuredFixture(self.fixture))
        object.__setattr__(self, "q_scales", tuple(Fraction(v) for v in self.q_scales))
        object.__setattr__(self, "k_scales", tuple(Fraction(v) for v in self.k_scales))
        base = self.base
        if base.head_dim != STRUCTURED_PERIOD:
            raise ValueError(
                f"structured logits require head_dim == {STRUCTURED_PERIOD}"
            )
        if base.nhead_q != base.nhead_k:
            raise ValueError(
                "structured logits require Hq == Hkv: each head carries its own logit"
            )
        if len(self.q_scales) != base.nhead_q or len(self.k_scales) != base.nhead_k:
            raise ValueError("one Q and one K scale per head")
        for value in (*self.q_scales, *self.k_scales):
            if value != 0 and not _is_signed_power_of_two(value):
                raise ValueError(f"scale {value} is not zero or a signed power of two")
        if self.block_n != STRUCTURED_PERIOD or self.block_m % self.block_n:
            raise ValueError("the tiles require block_n == 64 dividing block_m")
        if self.fixture is StructuredFixture.CYCLIC:
            if not 0 < self.shift < STRUCTURED_PERIOD:
                raise ValueError("the cyclic fixture requires 0 < shift < 64")
        elif self.shift:
            raise ValueError("shift applies to the cyclic fixture only")
        if self.fixture is StructuredFixture.LATE_MAX:
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
        return (
            self.shift
            if self.fixture is StructuredFixture.CYCLIC
            else _FIXTURE_SHIFT[self.fixture]
        )

    @property
    def case_id(self) -> str:
        detail = self.fixture.value
        if self.fixture is StructuredFixture.CYCLIC:
            detail += f"-r{self.shift}"
        if self.fixture is StructuredFixture.LATE_MAX:
            detail += f"-l{self.late}"
        scales = ",".join(
            f"{_scale_token(a)}x{_scale_token(b)}"
            for a, b in zip(self.q_scales, self.k_scales)
        )
        return f"{self.base.case_id}-{detail}-{scales}"


class StructuredMutation(str, Enum):
    """Wrong kernel semantics whose Tier B must be disjoint from the correct one
    at some coordinate. Each is a closed-form change of the score, the
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


_K_ROW_OFFSET = {
    StructuredMutation.K_ROW_PLUS_ONE: 1,
    StructuredMutation.K_ROW_MINUS_ONE: -1,
}
_OMITTED_WALSH_BLOCK = {
    StructuredMutation.OMIT_WALSH_BLOCK_0: 0,
    StructuredMutation.OMIT_WALSH_BLOCK_1: 1,
    StructuredMutation.OMIT_WALSH_BLOCK_3: 3,
}
_WALSH_BLOCK = 16
# The one-key mask-boundary mutations, as changes of the legal key interval.
_BOUNDARY_MUTATION = {
    mutation: MaskMutation(mutation.value)
    for mutation in (
        StructuredMutation.INCLUDE_FUTURE_KEY,
        StructuredMutation.EXCLUDE_LAST_KEY,
        StructuredMutation.INCLUDE_OLD_WINDOW_KEY,
        StructuredMutation.EXCLUDE_OLDEST_WINDOW_KEY,
    )
}


def is_distinguished(case: StructuredCase, query: int, key: int) -> bool:
    """Whether ``key`` carries the row's nonzero score (legality aside)."""
    if (key + case.residue_shift - query) % STRUCTURED_PERIOD:
        return False
    return not (case.fixture is StructuredFixture.LATE_MAX and key < case.late)


def score_numerator(
    case: StructuredCase,
    head: int,
    query: int,
    key: int,
    mutation: StructuredMutation | None = None,
) -> Fraction:
    """The exact QK dot product ``acc`` (every partial sum is exact in FP32).

    Under ``mutation`` it is the dot product the mutated kernel forms. K rows
    outside ``[0, Sk)`` (an off-by-one read) follow the fixture's formula. With
    one 16-dimension block of the Walsh contraction omitted, the ``Q[q] K[k]``
    dot product loses ``sum_{d in block} H[z, d]``, ``z = q ^ (k + 1)``, which is
    ``16 (-1)**popcount(z & 16 b)`` when ``z % 16 == 0`` and zero otherwise."""
    heads = case.base.nhead_q
    q_head = (
        (head + 1) % heads if mutation is StructuredMutation.Q_HEAD_ROTATE else head
    )
    k_head = (
        (head + 1) % heads if mutation is StructuredMutation.K_HEAD_ROTATE else head
    )
    product = case.q_scales[q_head] * case.k_scales[k_head]
    row = key + _K_ROW_OFFSET.get(mutation, 0)
    if case.fixture is StructuredFixture.WALSH:
        overlap = (query % STRUCTURED_PERIOD) ^ ((row + 1) % STRUCTURED_PERIOD)
        total = STRUCTURED_PERIOD if overlap == 0 else 0
        block = _OMITTED_WALSH_BLOCK.get(mutation)
        if block is not None and overlap % _WALSH_BLOCK == 0:
            parity = int(overlap & (_WALSH_BLOCK * block)).bit_count() & 1
            total -= _WALSH_BLOCK * (-1 if parity else 1)
        negative = row & 1 and mutation is not StructuredMutation.WALSH_SIGN
        return product * total * (-1 if negative else 1)
    shift = case.residue_shift
    if mutation is StructuredMutation.INVERSE_SHIFT:
        shift = -shift
    if (row + shift - query) % STRUCTURED_PERIOD:
        return Fraction(0)
    if case.fixture is StructuredFixture.LATE_MAX and row < case.late:
        return Fraction(0)
    return product


def row_logit(case: StructuredCase, head: int, query: int) -> Fraction:
    """Natural-domain logit ``c`` of the row's distinguished keys."""
    return score_numerator(case, head, query, query - case.residue_shift) / 8


def walsh_matrix(size: int = STRUCTURED_PERIOD) -> np.ndarray:
    """``H[x, d] = (-1)**popcount(x & d)``, the Sylvester Hadamard matrix."""
    index = np.arange(size)
    overlap = index[:, None] & index[None, :]
    parity = np.zeros_like(overlap)
    for bit in range(max(size - 1, 1).bit_length()):
        parity ^= (overlap >> bit) & 1
    return np.where(parity, -1.0, 1.0).astype(np.float32)


def structured_query_key(case: StructuredCase) -> tuple[np.ndarray, np.ndarray]:
    """Build exactly representable Q ``[B,Sq,H,D]`` and K ``[B,Sk,H,D]``."""
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = np.zeros(case.key_shape, dtype=np.float32)
    a = np.array([float(value) for value in case.q_scales], dtype=np.float32)
    b = np.array([float(value) for value in case.k_scales], dtype=np.float32)
    if case.fixture is StructuredFixture.WALSH:
        hadamard = walsh_matrix()
        for row in range(case.base.seqlen_q):
            q[:, row, :, :] = a[:, None] * hadamard[row % STRUCTURED_PERIOD]
        for key in range(case.base.seqlen_k):
            sign = -1.0 if key & 1 else 1.0
            k[:, key, :, :] = (
                sign * b[:, None] * hadamard[(key + 1) % STRUCTURED_PERIOD]
            )
        return q, k
    for row in range(case.base.seqlen_q):
        q[:, row, :, row % STRUCTURED_PERIOD] = a
    for key in range(case.base.seqlen_k):
        k[:, key, :, (key + case.residue_shift) % STRUCTURED_PERIOD] = b
    if case.fixture is StructuredFixture.LATE_MAX:
        k[:, : case.late] = 0.0
    return q, k


STRUCTURED_EXPOSURES = (
    *(f"basis@{offset}" for offset in BASIS_OFFSETS),
    "folded",
)


def structured_inputs(
    case: StructuredCase, exposure: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, K and the exposure's V (scaled ``2**h`` per head)."""
    q, k = structured_query_key(case)
    return q, k, exposure_values(case.base, exposure)


# ---------------------------------------------------------------------------
# The subject's KV schedule and the per-row reachable state.
# ---------------------------------------------------------------------------


def visited_tiles(
    case: StructuredCase, query: int, mutation: StructuredMutation | None = None
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


class TileClass(str, Enum):
    """What one visited KV tile holds for one row.

    ``LEGAL`` has a legal key: its weights are normal-scale and its masked keys
    weigh at most ``2**-149``. ``MASKED_PREFIX`` is fully masked with no legal
    tile before it: every weight is ``E(0)``, near one, and the first legal
    tile rescales that state by at most ``2**-149``.
    ``MASKED_AFTER_LEGAL`` is fully masked after a legal tile: every weight is
    at most ``2**-149``."""

    LEGAL = "L"
    MASKED_PREFIX = "M0"
    MASKED_AFTER_LEGAL = "M1"


@cache
def tile_classes(
    case: StructuredCase, query: int, mutation: StructuredMutation | None = None
) -> tuple[TileClass, ...]:
    """The class of every visited tile of ``query``, in loop order, computed
    from the legal key interval of the correct or mutated semantics."""
    first, last = legal_interval(case.base, query, _BOUNDARY_MUTATION.get(mutation))
    classes: list[TileClass] = []
    seen_legal = False
    for tile in visited_tiles(case, query, mutation):
        start = tile * case.block_n
        if first <= last and first < start + case.block_n and last >= start:
            seen_legal = True
            classes.append(TileClass.LEGAL)
        elif seen_legal:
            classes.append(TileClass.MASKED_AFTER_LEGAL)
        else:
            classes.append(TileClass.MASKED_PREFIX)
    return tuple(classes)


def masked_prefix(
    case: StructuredCase, query: int, mutation: StructuredMutation | None = None
) -> tuple[int, ...]:
    """The fully masked leading tiles of ``query`` (class M0), in loop order;
    the enclosure is derived for at most one."""
    tiles = visited_tiles(case, query, mutation)
    classes = tile_classes(case, query, mutation)
    return tuple(
        tile
        for tile, tile_class in zip(tiles, classes)
        if tile_class is TileClass.MASKED_PREFIX
    )


def prefix_selected_keys(
    case: StructuredCase,
    exposure: str,
    query: int,
    dim: int,
    mutation: StructuredMutation | None = None,
) -> int:
    """How many keys the exposed coordinate selects inside the masked prefix of
    ``query``."""
    return sum(
        exposure_key(case, exposure, dim, tile) is not None
        for tile in masked_prefix(case, query, mutation)
    )


def masked_prefix_scope(
    case: StructuredCase, query: int, mutation: StructuredMutation | None = None
) -> str | None:
    """Why the masked-prefix argument does not cover ``query``, or ``None`` when
    it does.

    The residue bound is derived for at most one fully masked leading tile followed by a
    legal tile. More leading masked tiles, or a row with no legal tile at all,
    need their own residue bound and are not enclosed."""
    classes = tile_classes(case, query, mutation)
    prefix = sum(item is TileClass.MASKED_PREFIX for item in classes)
    if prefix > 1:
        return f"row {query} has {prefix} fully masked leading tiles (derived for 1)"
    if TileClass.LEGAL not in classes:
        return f"row {query} visits no tile with a legal key"
    return None


def tile_sum(
    tile_class: TileClass,
    weights: Sequence[F32Interval],
    phantom: Sequence[F32Interval] = (),
) -> F32Interval:
    """Enclose the FP32 sum of one tile's weights, by tile class.

    ``LEGAL`` and ``MASKED_PREFIX`` tiles hold normal-scale terms, the regime
    of the ``gamma`` bound (:func:`sum_any_order`). A ``MASKED_AFTER_LEGAL``
    tile holds only masked terms, each ``[0, 2**-149]`` under A-E (a looser
    exponential used for attribution widens that to a few subnormal steps):
    the bound is not used there and the sum is enclosed explicitly by zero and
    the exact sum of the upper ends, ``[0, 2**-143]`` for 64 keys, which is
    what gradual underflow computes and holds ``+0`` for a flushing mode.
    ``phantom`` terms (a mutation's extra normal-scale key) put any tile back
    in the ``gamma`` regime."""
    if tile_class is TileClass.MASKED_AFTER_LEGAL and not phantom:
        total = Fraction(0)
        for weight in weights:
            if weight.lo != 0 or weight.hi >= F32_SMALLEST_NORMAL:
                raise StructuredScopeError(
                    "a fully masked tile holds a normal-scale weight: "
                    f"[{float(weight.lo)!r}, {float(weight.hi)!r}]"
                )
            total += weight.hi
        return F32Interval(Fraction(0), total)
    return sum_any_order((*weights, *phantom))


# ``scale`` is 1 instead of 1/8, or LOG2E is not applied.
_MUTATED_SCORE_SCALE = {
    StructuredMutation.OMIT_SQRT_D: round_f32(f32(1.0) * LOG2E_F32),
    StructuredMutation.EXP_ENCODING: f32(_SCALE_ARGUMENT),
}


@cache
def _score_of(numerator: Fraction, scale: Fraction = SCORE_SCALE) -> Fraction:
    return round_f32(numerator * scale, "rn")


def score(
    case: StructuredCase,
    head: int,
    query: int,
    key: int,
    mutation: StructuredMutation | None = None,
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
    """Reachable FP32 state of one (row, head) after the KV loop.

    ``denominators[i]`` is the reachable ``l`` after ``steps[i]``; the last one
    is ``denominator``."""

    steps: tuple[TileStep, ...]
    denominators: tuple[F32Interval, ...]
    denominator: F32Interval
    reciprocal: F32Interval


@cache
def row_trace(
    case: StructuredCase,
    head: int,
    query: int,
    mutation: StructuredMutation | None = None,
    e_ulps: int = 1,
) -> RowTrace:
    """Propagate the row's running max, weights and denominator.

    The running max is exact (``max`` does not round). Each ``v_exp_f32`` is its
    candidate interval; each tile's 64 weights are summed by tile class
    (:func:`tile_sum`: ``gamma_63`` in any order, or the explicit masked
    enclosure), then ``l = RN(RN(l alpha) + l_tile)``, and the reciprocal is
    taken under A-DIV. ``OMIT_RESCALE`` fixes ``alpha = 1``;
    ``DENOMINATOR_PLUS_ONE`` adds a logit-0 key's weight to the last tile's
    sum, never to the accumulator. ``e_ulps`` loosens A-E for failure
    attribution only.

    Raises :class:`StructuredScopeError` for a row the masked-prefix argument does not
    cover or whose
    denominator leaves the A-DIV domain."""
    outside = masked_prefix_scope(case, query, mutation)
    if outside is not None:
        raise StructuredScopeError(f"{case.case_id}: {outside}")
    running_max = MASKED_SCORE
    denominator = F32Interval.point(Fraction(0))
    steps: list[TileStep] = []
    denominators: list[F32Interval] = []
    tiles = visited_tiles(case, query, mutation)
    classes = tile_classes(case, query, mutation)
    for tile, tile_class in zip(tiles, classes):
        start = tile * case.block_n
        scores = [
            score(case, head, query, key, mutation)
            for key in range(start, start + case.block_n)
        ]
        new_max = max(running_max, max(scores))
        if mutation is StructuredMutation.OMIT_RESCALE:
            alpha = F32Interval.point(Fraction(1))
        else:
            alpha = exp2_interval(round_f32(running_max - new_max, "rn"), e_ulps)
        weights = tuple(
            exp2_interval(round_f32(value - new_max, "rn"), e_ulps) for value in scores
        )
        phantom: tuple[F32Interval, ...] = ()
        if mutation is StructuredMutation.DENOMINATOR_PLUS_ONE and tile == tiles[-1]:
            phantom = (exp2_interval(round_f32(-new_max, "rn"), e_ulps),)
        denominator = add_rn(
            mul_rn(denominator, alpha), tile_sum(tile_class, weights, phantom)
        )
        steps.append(TileStep(tile, alpha, weights))
        denominators.append(denominator)
        running_max = new_max
    return RowTrace(
        tuple(steps), tuple(denominators), denominator, rcp_a_div(denominator)
    )


def exposure_key(
    case: StructuredCase, exposure: str, dim: int, tile: int
) -> int | None:
    """The key in ``tile`` whose V row is nonzero at ``dim``, if any."""
    start = tile * case.block_n
    if exposure == "folded":
        return start + dim
    key = basis_offset(exposure) + dim
    return (
        key
        if start <= key < start + case.block_n and key < case.base.seqlen_k
        else None
    )


@cache
def _scaled_cast(dtype: str, weight: F32Interval, exponent: int) -> F32Interval:
    """PV product interval ``q_T(p~) * 2**h``: exact in FP32."""
    low, high = cast_bits_hull(weight, dtype)
    scale = Fraction(2) ** exponent
    return F32Interval(
        target_value(low, dtype) * scale, target_value(high, dtype) * scale
    )


MFMA_ROUNDING = ("faithful", "rn")


@cache
def _accumulator_steps(
    case: StructuredCase,
    head: int,
    query: int,
    products: tuple[F32Interval | None, ...],
    mfma: str,
    mutation: StructuredMutation | None,
    e_ulps: int,
) -> tuple[tuple[F32Interval, F32Interval], ...]:
    """Per visited tile, the reachable accumulator after the rescale
    ``RN(o alpha)`` and after the tile's PV atoms.

    Per tile ``o = MFMA(RN(o alpha), product)``. With one nonzero product per
    output per tile, the tile performs at most one real addition; it
    is exact when the accumulator is zero, else faithful (or RN when pinned).
    After a masked prefix the rescaled value is the prefix residue, at most
    ``2**(h-149)``."""
    if mfma not in MFMA_ROUNDING:
        raise ValueError(f"mfma must be one of {MFMA_ROUNDING}")
    trace = row_trace(case, head, query, mutation, e_ulps)
    accumulator = F32Interval.point(Fraction(0))
    steps = []
    for step, product in zip(trace.steps, products):
        rescaled = accumulator = mul_rn(accumulator, step.alpha)
        if product is None or product.hi == 0:
            pass
        elif accumulator.hi == 0:
            accumulator = product
        elif mfma == "faithful":
            accumulator = add_faithful(accumulator, product)
        else:
            accumulator = add_rn(accumulator, product)
        steps.append((rescaled, accumulator))
    return tuple(steps)


def _exposed_products(
    case: StructuredCase, exposure: str, dim: int, head: int, trace: RowTrace
) -> tuple[F32Interval | None, ...]:
    """Per visited tile, the PV product ``q_T(p~) 2**h`` of the key the exposed
    coordinate selects there, or ``None`` when it selects none."""
    exponent = case.head_scale_exponents[head]
    products: list[F32Interval | None] = []
    for step in trace.steps:
        key = exposure_key(case, exposure, dim, step.tile)
        if key is None:
            products.append(None)
        else:
            weight = step.weights[key - step.tile * case.block_n]
            products.append(_scaled_cast(case.dtype, weight, exponent))
    return tuple(products)


def accumulator_steps(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    dim: int,
    *,
    mfma: str = "faithful",
    mutation: StructuredMutation | None = None,
) -> tuple[tuple[F32Interval, F32Interval], ...]:
    """The numerator recurrence of one exposed coordinate, tile by tile: the
    reachable ``RN(o alpha)`` and the reachable ``o`` after the tile."""
    trace = row_trace(case, head, query, mutation)
    products = _exposed_products(case, exposure, dim, head, trace)
    return _accumulator_steps(case, head, query, products, mfma, mutation, 1)


def tier_b_row(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    *,
    mfma: str = "faithful",
    mutation: StructuredMutation | None = None,
    e_ulps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest reachable output bits of one row and query head,
    under the correct or a ``mutation``'s semantics. The epilogue is the A-F
    hull for both, so a mutant is judged with the same compiler latitude."""
    trace = row_trace(case, head, query, mutation, e_ulps)
    low = np.zeros(case.base.head_dim, dtype=np.uint16)
    high = np.zeros(case.base.head_dim, dtype=np.uint16)
    for dim in range(case.base.head_dim):
        products = _exposed_products(case, exposure, dim, head, trace)
        steps = _accumulator_steps(case, head, query, products, mfma, mutation, e_ulps)
        accumulator = steps[-1][1]
        low[dim], high[dim] = cast_single_or_double_hull(
            accumulator, trace.reciprocal, case.dtype
        )
    return low, high


def tier_b_bounds(
    case: StructuredCase,
    exposure: str,
    *,
    rows: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    mfma: str = "faithful",
    mutation: StructuredMutation | None = None,
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
# Certified targets and Tier A.
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
def _row_counts(case: StructuredCase, head: int, query: int) -> tuple[int, int, int]:
    """``(first, last, n_c)``: the legal interval and its distinguished count."""
    first, last = legal_interval(case.base, query)
    distinguished = sum(
        1
        for key in range(first, last + 1)
        if score_numerator(case, head, query, key) != 0
    )
    return first, last, distinguished


def coordinate_class(
    case: StructuredCase, exposure: str, query: int, head: int, dim: int
) -> CoordinateClass:
    first, last, n_c = _row_counts(case, head, query)
    if exposure == "folded":
        keys = range(
            first + (dim - first) % STRUCTURED_PERIOD, last + 1, STRUCTURED_PERIOD
        )
    else:
        key = basis_offset(exposure) + dim
        keys = range(key, key + 1) if first <= key <= last else range(0)
    m_c = sum(1 for key in keys if score_numerator(case, head, query, key) != 0)
    return CoordinateClass(
        m_c, len(keys) - m_c, n_c, last - first + 1 - n_c, row_logit(case, head, query)
    )


@dataclass(frozen=True)
class CertifiedTarget:
    """``q_T(y*)`` and what certifies it.

    A rational target is rounded by exact rational comparison with the
    rounding boundaries of the format (ties to even); it is in general not
    representable, and it has no interval record. An irrational target
    carries a rational interval ``[lower, upper]`` that contains ``y*``, the
    two rounding boundaries adjacent to its bits (``lower_boundary`` is
    ``None`` at ``+0``, where only the upper one applies) and ``margin``, the
    positive distance from the interval to the nearer boundary: the whole
    interval lies inside one rounding cell."""

    bits: int
    rational: bool
    lower: Fraction | None = None
    upper: Fraction | None = None
    lower_boundary: Fraction | None = None
    upper_boundary: Fraction | None = None
    margin: Fraction | None = None


def rounding_boundaries(bits: int, dtype: str) -> tuple[Fraction | None, Fraction]:
    """The two rounding boundaries adjacent to a nonnegative finite target
    value: the midpoints to its lower and upper neighbors. At a power of two
    above the smallest normal the upper one is twice as far as the lower one,
    which is why the two boundaries, and not one ULP size, are what a target
    is certified against. ``+0`` has no lower boundary among nonnegative
    values."""
    center = target_value(target_rank(bits, dtype), dtype)
    below = (target_value(bits - 1, dtype) + center) / 2 if bits else None
    above = (target_value(target_rank(bits + 1, dtype), dtype) + center) / 2
    return below, above


@cache
def certify_target(dtype: str, exponent: int, item: CoordinateClass) -> CertifiedTarget:
    """``q_T(2**h (m_c e^c + m_0) / (n_c e^c + n_0))``, certified.

    The value is rational when ``c = 0`` or the coordinate's class mix is
    proportional to the row's (``m_c n_0 = m_0 n_c``); it is then rounded
    exactly. Otherwise it is transcendental (Lindemann-Weierstrass: the
    counts are integers, ``2**h`` and ``c`` are rational, and the map from
    ``e^c`` is invertible), so it is on no rounding boundary. It is certified
    by interval containment, not by a number of digits: ``e^c`` is enclosed
    by :func:`_exp_enclosure`, the target is monotone in ``e^c``, and the
    resulting rational interval must lie strictly between the two rounding
    boundaries adjacent to its bits. A target whose interval reaches a
    boundary raises; it is never guessed."""
    if item.is_zero:
        return CertifiedTarget(0, True)
    scale = Fraction(2) ** exponent
    if item.logit == 0 or item.m_c * item.n_0 == item.m_0 * item.n_c:
        exact = scale * Fraction(item.m_c + item.m_0, item.n_c + item.n_0)
        return CertifiedTarget(
            target_bits(exact.numerator, exact.denominator, dtype), True
        )
    ends = [
        scale * (item.m_c * growth + item.m_0) / (item.n_c * growth + item.n_0)
        for growth in _exp_enclosure(item.logit)
    ]
    lower, upper = min(ends), max(ends)
    bits = target_bits(lower.numerator, lower.denominator, dtype)
    below, above = rounding_boundaries(bits, dtype)
    if (below is not None and lower <= below) or upper >= above:
        raise ArithmeticError(
            f"target in [{float(lower)!r}, {float(upper)!r}] is not certified: "
            "its interval reaches a rounding boundary"
        )
    margin = above - upper if below is None else min(lower - below, above - upper)
    return CertifiedTarget(bits, False, lower, upper, below, above, margin)


def certified_target_bits(dtype: str, exponent: int, item: CoordinateClass) -> int:
    """The bits of :func:`certify_target`."""
    return certify_target(dtype, exponent, item).bits


def tier_a_bounds(
    case: StructuredCase,
    exposure: str,
    *,
    rows: Iterable[int] | None = None,
    classes: set[tuple[int, CoordinateClass]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Certified target bits and the Tier A radius (1 at the row max, 2 elsewhere,
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
                radius[:, query, head, dim] = (
                    0 if item.is_zero else (1 if item.is_row_max else 2)
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


class EmulationMode(str, Enum):
    """Which realization of the dense path :func:`emulate_row` computes."""

    # The strict IR: lane order, E = RN(2**x), RN MFMA, q_T(RN32(o R)).
    IR_EXACT = "ir_exact"
    # The IR path with a named artifact's per-site FP16 epilogue rounding.
    ARTIFACT_EXACT = "artifact_exact"
    # Any member A-E, A-MFMA and A-F allow: random summation tree, E candidate,
    # MFMA rounding direction and, for FP16, single or double epilogue rounding.
    RANDOMIZED_SEMANTIC = "randomized_semantic"


class StructuredViolation(str, Enum):
    """The IR-exact path with exactly one primitive replaced by a value its
    named assumption excludes.

    These realizations are diagnostic. They are never enclosed: an output that
    leaves Tier B this way must fail and be attributed to the assumption, and
    one that stays inside Tier B shows that the check cannot observe the primitive,
    which is no evidence for the assumption."""

    # A-E: every v_exp_f32 result two FP32 steps outside its candidate hull.
    E_LOW = "a_e_low"
    E_HIGH = "a_e_high"
    # A-MFMA: a genuine folded addition two FP32 steps outside {RD, RU}.
    MFMA_LOW = "a_mfma_low"
    MFMA_HIGH = "a_mfma_high"
    # A-QK: a Walsh score accumulator one FP32 step from the exact dot product.
    QK_DOWN = "a_qk_down"
    QK_UP = "a_qk_up"
    # A-DIV: the FP32 neighbor of RN(1 / l).
    DIV_LOW = "a_div_low"
    DIV_HIGH = "a_div_high"
    # A-F: an FP16 store of the exact product truncated toward zero.
    F_TRUNCATED = "a_f_truncated"
    # A-MODE32: FP32 add, subtract and multiply rounded toward zero.
    MODE32_RZ_ARITHMETIC = "a_mode32_rz_arithmetic"
    # A-CVT-F16, A-CVT-BF16: the FP32-to-target conversion (probability cast
    # and epilogue) truncating, in the FP16 and in the BF16 kernel.
    CVT_F16_TRUNCATING = "a_cvt_f16_truncating"
    CVT_BF16_TRUNCATING = "a_cvt_bf16_truncating"


_EXP2_VIOLATION = {StructuredViolation.E_LOW: -2, StructuredViolation.E_HIGH: 2}
_MFMA_VIOLATION = {
    StructuredViolation.MFMA_LOW: ("rd", -2),
    StructuredViolation.MFMA_HIGH: ("ru", 2),
}
_QK_VIOLATION = {StructuredViolation.QK_DOWN: -1, StructuredViolation.QK_UP: 1}
_DIV_VIOLATION = {StructuredViolation.DIV_LOW: -1, StructuredViolation.DIV_HIGH: 1}
# The truncating conversion of each target format.
_CVT_VIOLATION = {
    StructuredViolation.CVT_F16_TRUNCATING: "fp16",
    StructuredViolation.CVT_BF16_TRUNCATING: "bf16",
}


def applicable_violations(case: StructuredCase) -> tuple[StructuredViolation, ...]:
    """The violations ``case`` can express: only the Walsh contraction has a
    dense score atom, only FP16 has the A-F epilogue, and each kernel has the
    conversion of its own target format."""
    return tuple(
        violation
        for violation in StructuredViolation
        if not (
            violation in _QK_VIOLATION and case.fixture is not StructuredFixture.WALSH
        )
        and not (
            violation is StructuredViolation.F_TRUNCATED
            and case.dtype not in _SINGLE_ROUNDING_DTYPES
        )
        and _CVT_VIOLATION.get(violation, case.dtype) == case.dtype
    )


def _grid_steps(value: Fraction, count: int) -> Fraction:
    """``value`` moved ``count`` FP32 grid steps, not below zero for a
    nonnegative value."""
    floor = Fraction(0) if value >= 0 else None
    for _ in range(abs(count)):
        value = next_up(value) if count > 0 else next_down(value)
    return value if floor is None else max(value, floor)


def _truncated_target_bits(value: Fraction, dtype: str) -> int:
    """FP16/BF16 bits of a nonnegative rational rounded toward zero."""
    bits = _exact_target_bits(value, dtype)
    return bits - 1 if target_value(bits, dtype) > value else bits


def emulate_row(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    *,
    mode: EmulationMode | str = EmulationMode.IR_EXACT,
    rng: np.random.Generator | None = None,
    artifact: str | None = None,
    mutation: StructuredMutation | None = None,
    violation: StructuredViolation | None = None,
) -> np.ndarray:
    """Output bits of one row from an FP32 emulation of the dense path.

    ``IR_EXACT`` follows the kernel's IR. ``ARTIFACT_EXACT`` differs only in
    the FP16 stores that the code object ``artifact`` (a key of
    ``STRUCTURED_EPILOGUE_ARTIFACTS``) rounds once. ``RANDOMIZED_SEMANTIC`` draws from
    ``rng`` a member of the reachable set, used to test that Tier B encloses
    it. ``mutation`` emulates the mutated kernel, whose realizations its own
    Tier B must enclose. ``violation`` breaks one named assumption on a
    deterministic mode. Records report the IR and artifact agreement
    separately, so an exponential diagnosis is not confounded by the
    compiler's epilogue."""
    mode = EmulationMode(mode)
    if (rng is None) == (mode is EmulationMode.RANDOMIZED_SEMANTIC):
        raise ValueError("rng is required by, and only by, randomized_semantic")
    if (artifact is None) == (mode is EmulationMode.ARTIFACT_EXACT):
        raise ValueError("artifact is required by, and only by, artifact_exact")
    if artifact is not None and artifact not in STRUCTURED_EPILOGUE_ARTIFACTS:
        raise ValueError(f"no epilogue artifact named {artifact!r}")
    if violation is not None and rng is not None:
        raise ValueError("a violation applies to a deterministic mode only")
    single_lanes = (
        STRUCTURED_EPILOGUE_ARTIFACTS[artifact].single_lanes
        if artifact
        else frozenset()
    )
    accumulator, reciprocal = _emulate_path(
        case, exposure, query, head, rng, mutation, violation
    )
    bits = []
    for dim, value in enumerate(accumulator):
        if mode is EmulationMode.RANDOMIZED_SEMANTIC:
            single = bool(rng.integers(2))
        else:
            single = dim % 4 in single_lanes
        bits.append(
            _store_bits(
                value, reciprocal, case.dtype, single=single, violation=violation
            )
        )
    return np.array(bits, dtype=np.uint16)


def epilogue_alternatives(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    violation: StructuredViolation | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The IR-exact row's stored bits under double and single epilogue rounding.

    Every A-F realization of the IR-exact path stores, at each coordinate, one
    of the two. They are equal for BF16, which A-F keeps double-rounded. With
    ``violation`` they are the two stores of that violating path."""
    accumulator, reciprocal = _emulate_path(
        case, exposure, query, head, None, None, violation
    )
    double, single = (
        np.array(
            [
                _store_bits(
                    value, reciprocal, case.dtype, single=s, violation=violation
                )
                for value in accumulator
            ],
            dtype=np.uint16,
        )
        for s in (False, True)
    )
    return double, single


def _store_bits(
    value: np.float32,
    reciprocal: np.float32,
    dtype: str,
    *,
    single: bool,
    violation: StructuredViolation | None = None,
) -> int:
    """``q_T(RN32(o R))``, or under A-F for FP16 ``q_T(o R)`` when ``single``."""
    exact = Fraction(float(value)) * Fraction(float(reciprocal))
    fused = dtype in _SINGLE_ROUNDING_DTYPES
    truncating = _CVT_VIOLATION.get(violation) == dtype
    if fused and (single or violation is StructuredViolation.F_TRUNCATED):
        if truncating or violation is StructuredViolation.F_TRUNCATED:
            return _truncated_target_bits(exact, dtype)
        return _exact_target_bits(exact, dtype)
    if violation is StructuredViolation.MODE32_RZ_ARITHMETIC:
        product = np.float32(float(round_f32(exact, "rz")))
    else:
        product = np.float32(value * reciprocal)
    if truncating:
        return _truncated_target_bits(Fraction(float(product)), dtype)
    return _cast_bits(product, dtype)


@dataclass(frozen=True)
class _EmulatedTile:
    """One visited tile of an emulated row: the rescale and the 64 weights."""

    tile: int
    alpha: np.float32
    weights: tuple[np.float32, ...]


def _emulate_tiles(
    case: StructuredCase,
    query: int,
    head: int,
    rng: np.random.Generator | None,
    mutation: StructuredMutation | None,
    violation: StructuredViolation | None,
) -> tuple[tuple[_EmulatedTile, ...], np.float32]:
    """The exposure-independent part of the path: each tile's rescale factor
    and weights, and the reciprocal of the final denominator.

    Without ``rng``: the documented lane order and ``E = RN(2**x)``. With
    ``rng``: a random summation tree per tile and an ``E`` candidate per call.
    ``violation`` replaces one primitive (:class:`StructuredViolation`)."""
    toward_zero = violation is StructuredViolation.MODE32_RZ_ARITHMETIC

    def rounded(exact: Fraction) -> np.float32:
        return np.float32(float(round_f32(exact, "rz")))

    def add(left: np.float32, right: np.float32) -> np.float32:
        if toward_zero:
            return rounded(Fraction(float(left)) + Fraction(float(right)))
        return np.float32(left + right)

    def sub(left: np.float32, right: np.float32) -> np.float32:
        if toward_zero:
            return rounded(Fraction(float(left)) - Fraction(float(right)))
        return np.float32(left - right)

    def mul(left: np.float32, right: np.float32) -> np.float32:
        if toward_zero:
            return rounded(Fraction(float(left)) * Fraction(float(right)))
        return np.float32(left * right)

    exp2_steps = _EXP2_VIOLATION.get(violation, 0)

    def exp2(argument: np.float32) -> np.float32:
        exact = Fraction(float(argument))
        if rng is not None:
            candidates = exp2_candidates(exact)
            return np.float32(float(candidates[int(rng.integers(len(candidates)))]))
        if not exp2_steps:
            return np.float32(float(exp2_nearest(exact)))
        candidates = exp2_candidates(exact)
        edge = candidates[0] if exp2_steps < 0 else candidates[-1]
        return np.float32(float(_grid_steps(edge, exp2_steps)))

    qk_steps = (
        _QK_VIOLATION.get(violation, 0)
        if case.fixture is StructuredFixture.WALSH
        else 0
    )

    def score_of(key: int) -> Fraction:
        value = score(case, head, query, key, mutation)
        if not qk_steps or value == MASKED_SCORE:
            return value
        numerator = score_numerator(case, head, query, key, mutation)
        return _score_of(
            _grid_steps(numerator, qk_steps),
            _MUTATED_SCORE_SCALE.get(mutation, SCORE_SCALE),
        )

    running_max = np.float32(float(MASKED_SCORE))
    denominator = np.float32(0.0)
    emulated: list[_EmulatedTile] = []
    tiles = visited_tiles(case, query, mutation)
    for tile in tiles:
        start = tile * case.block_n
        scores = np.array(
            [float(score_of(key)) for key in range(start, start + case.block_n)],
            dtype=np.float32,
        )
        new_max = np.float32(max(running_max, np.max(scores)))
        if mutation is StructuredMutation.OMIT_RESCALE:
            alpha = np.float32(1.0)
        else:
            alpha = exp2(sub(running_max, new_max))
        weights = [exp2(sub(value, new_max)) for value in scores]
        phantom = []
        if mutation is StructuredMutation.DENOMINATOR_PLUS_ONE and tile == tiles[-1]:
            phantom.append(exp2(sub(np.float32(0.0), new_max)))
        if rng is None:
            halves = []
            for order in _LANE_ORDER:
                total = np.float32(0.0)
                for index in order:
                    total = add(total, weights[index])
                halves.append(total)
            tile_total = add(halves[0], halves[1])
            for extra in phantom:
                tile_total = add(tile_total, extra)
        else:
            items = list(weights) + phantom
            while len(items) > 1:
                left = items.pop(int(rng.integers(len(items))))
                right = items.pop(int(rng.integers(len(items))))
                items.append(np.float32(left + right))
            tile_total = items[0]
        denominator = add(mul(denominator, alpha), tile_total)
        emulated.append(_EmulatedTile(tile, alpha, tuple(weights)))
        running_max = new_max
    reciprocal = np.float32(np.float32(1.0) / denominator)
    if violation in _DIV_VIOLATION:
        reciprocal = np.float32(
            float(_grid_steps(Fraction(float(reciprocal)), _DIV_VIOLATION[violation]))
        )
    return tuple(emulated), reciprocal


@cache
def _emulated_tiles(
    case: StructuredCase,
    query: int,
    head: int,
    mutation: StructuredMutation | None,
    violation: StructuredViolation | None,
) -> tuple[tuple[_EmulatedTile, ...], np.float32]:
    """The deterministic row state, shared by the five exposures of a row."""
    return _emulate_tiles(case, query, head, None, mutation, violation)


def _emulate_path(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    rng: np.random.Generator | None,
    mutation: StructuredMutation | None,
    violation: StructuredViolation | None = None,
) -> tuple[np.ndarray, np.float32]:
    """FP32 accumulator row and reciprocal before the output epilogue.

    Without ``rng``: the documented lane order, ``E = RN(2**x)`` and an RN MFMA
    accumulate. With ``rng``: a random summation tree per tile, an ``E``
    candidate per call and a directed rounding per MFMA addition."""
    if rng is None:
        tiles, reciprocal = _emulated_tiles(case, query, head, mutation, violation)
    else:
        tiles, reciprocal = _emulate_tiles(case, query, head, rng, mutation, None)
    toward_zero = violation is StructuredViolation.MODE32_RZ_ARITHMETIC
    truncating = _CVT_VIOLATION.get(violation) == case.dtype
    mfma_mode, mfma_steps = _MFMA_VIOLATION.get(violation, (None, 0))
    scale = np.float32(2.0 ** case.head_scale_exponents[head])
    accumulator = np.zeros(case.base.head_dim, dtype=np.float32)
    for item in tiles:
        start = item.tile * case.block_n
        if toward_zero:
            alpha = Fraction(float(item.alpha))
            accumulator = np.array(
                [
                    float(round_f32(Fraction(float(value)) * alpha, "rz"))
                    for value in accumulator
                ],
                dtype=np.float32,
            )
        else:
            accumulator = (accumulator * item.alpha).astype(np.float32)
        for dim in range(case.base.head_dim):
            key = exposure_key(case, exposure, dim, item.tile)
            if key is None:
                continue
            weight = item.weights[key - start]
            if truncating:
                bits = _truncated_target_bits(Fraction(float(weight)), case.dtype)
            else:
                bits = _cast_bits(weight, case.dtype)
            product = np.float32(float(target_value(bits, case.dtype))) * scale
            if product == 0:
                continue
            if accumulator[dim] == 0:
                accumulator[dim] = product
            elif rng is None and mfma_mode is None:
                accumulator[dim] = np.float32(accumulator[dim] + product)
            else:
                exact = Fraction(float(accumulator[dim])) + Fraction(float(product))
                if rng is None:
                    value = _grid_steps(round_f32(exact, mfma_mode), mfma_steps)
                else:
                    value = round_f32(exact, ("rd", "ru")[int(rng.integers(2))])
                accumulator[dim] = np.float32(float(value))
    return accumulator, reciprocal


# ---------------------------------------------------------------------------
# Per-head scales chosen by separation in output space.
# ---------------------------------------------------------------------------

# Head logits per fixture family. Periodic fixtures (identity, cyclic, late
# max) cover large and moderate logits of both signs; the Walsh row sign
# alternates, so its magnitudes cover both signs on their own.
STRUCTURED_HEAD_LOGITS: Mapping[str, tuple[Fraction, ...]] = MappingProxyType(
    {
        "periodic": (Fraction(8), Fraction(-8), Fraction(2), Fraction(-1, 2)),
        "walsh": (Fraction(8), Fraction(2), Fraction(1, 2), Fraction(4)),
    }
)
# Rows sampled for head-separation gaps: block, tile and window edges.
_GAP_ROWS = (0, 6, 7, 31, 63, 64, 65, 127, 128, 191, 192, 255)


def fixture_family(fixture: StructuredFixture) -> str:
    return (
        "walsh" if StructuredFixture(fixture) is StructuredFixture.WALSH else "periodic"
    )


def _logit_case(
    base: DenseShape, block_m: int, family: str, logits: Sequence[Fraction]
) -> StructuredCase:
    """A case whose head ``h`` has logit ``logits[h]`` with ``b_h = 1``."""
    fixture = (
        StructuredFixture.WALSH if family == "walsh" else StructuredFixture.IDENTITY
    )
    factor = Fraction(1, 8) if family == "walsh" else Fraction(8)
    return StructuredCase(
        base,
        fixture,
        tuple(value * factor for value in logits),
        tuple(Fraction(1) for _ in logits),
        block_m=block_m,
    )


@cache
def logit_gap(
    base: DenseShape,
    block_m: int,
    family: str,
    head: int,
    logit: Fraction,
    other: Fraction,
) -> int:
    """Largest Tier B separation, over the sampled rows and every exposure, of
    head ``head`` computed with logit ``logit`` versus ``other``."""
    logits = [Fraction(1)] * base.nhead_q
    logits[head] = logit
    correct = _logit_case(base, block_m, family, logits)
    logits[head] = other
    mutated = _logit_case(base, block_m, family, logits)
    gap = 0
    for exposure in STRUCTURED_EXPOSURES:
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
    bases: Sequence[tuple[DenseShape, int]],
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
            logit = STRUCTURED_HEAD_LOGITS[family][head]
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
    bases: Sequence[tuple[DenseShape, int]],
    family: str,
    exponents: Sequence[int] = (-2, -1, 0, 1, 2, 3),
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], int]:
    """Maximize :func:`head_scale_margin` over signed power-of-two ``a_h``;
    ``b_h`` follows from the head logit. Ties prefer scales nearest one."""
    logits = STRUCTURED_HEAD_LOGITS[family]
    choices = [
        Fraction(sign) * Fraction(2) ** power for power in exponents for sign in (1, -1)
    ]
    product_of = (lambda c: c / 8) if family == "walsh" else (lambda c: c * 8)

    def k_scale(head: int, a: Fraction) -> Fraction | None:
        b = product_of(logits[head]) / a
        return (
            b
            if _is_signed_power_of_two(b) and Fraction(1, 64) <= abs(b) <= 64
            else None
        )

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

# Chosen by ``search_head_scales(STRUCTURED_SEARCH_BASES, family)`` with the default
# exponent range; ``STRUCTURED_HEAD_SCALE_MARGIN`` is the attained worst-case gap in
# target ULPs, recomputed by a CPU contract.
STRUCTURED_HEAD_SCALES: Mapping[
    str, tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
] = MappingProxyType(
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
STRUCTURED_HEAD_SCALE_MARGIN: Mapping[str, int] = MappingProxyType(
    {"periodic": 181, "walsh": 181}
)
STRUCTURED_CYCLIC_SHIFT = 37
STRUCTURED_LATE_KEYS = 64
# What dispatch resolves each quick shape to: the non-persistent dense path with
# the fast base-2 exponential, no sinks and no LSE, on 64-key KV tiles. Only the
# query block differs: 128 rows for FP16 causal, 64 for the BF16 window. The
# calibration entries carry these fingerprints, and the tile sizes fix which KV
# tiles each row visits.
STRUCTURED_QUICK_SPECS: Mapping[str, SpecFingerprint] = MappingProxyType(
    {
        dtype: SpecFingerprint(
            path="dense",
            persistent=False,
            runtime_shape=True,
            head_dim=64,
            block_m=block_m,
            block_n=64,
            use_exp2_fast=True,
            use_sinks=False,
            emit_lse=False,
        )
        for dtype, block_m in (("fp16", 128), ("bf16", 64))
    }
)
_QUICK_BLOCK_M = {dtype: spec.block_m for dtype, spec in STRUCTURED_QUICK_SPECS.items()}
STRUCTURED_SEARCH_BASES = tuple(
    (base, _QUICK_BLOCK_M[base.dtype]) for base in QUICK_SHAPES
)


def structured_quick_case(
    base: DenseShape, fixture: StructuredFixture
) -> StructuredCase:
    """The quick case of one fixture on one quick shape."""
    fixture = StructuredFixture(fixture)
    q_scales, k_scales = STRUCTURED_HEAD_SCALES[fixture_family(fixture)]
    return StructuredCase(
        base,
        fixture,
        q_scales,
        k_scales,
        shift=STRUCTURED_CYCLIC_SHIFT if fixture is StructuredFixture.CYCLIC else 0,
        late=STRUCTURED_LATE_KEYS if fixture is StructuredFixture.LATE_MAX else 0,
        block_m=_QUICK_BLOCK_M[base.dtype],
    )


STRUCTURED_QUICK_CASES = tuple(
    structured_quick_case(base, fixture)
    for base in QUICK_SHAPES
    for fixture in StructuredFixture
)


@cache
def tier_b_exposure(
    case: StructuredCase, exposure: str, mutation: StructuredMutation | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cached, read-only :func:`tier_b_bounds` for a whole exposure."""
    low, high = tier_b_bounds(case, exposure, mutation=mutation)
    low.setflags(write=False)
    high.setflags(write=False)
    return low, high


@cache
def tier_a_exposure(
    case: StructuredCase, exposure: str
) -> tuple[np.ndarray, np.ndarray, frozenset[tuple[int, CoordinateClass]]]:
    """Cached, read-only :func:`tier_a_bounds` for a whole exposure, with the
    distinct ``(head exponent, class)`` pairs behind its targets."""
    classes: set[tuple[int, CoordinateClass]] = set()
    center, radius = tier_a_bounds(case, exposure, classes=classes)
    center.setflags(write=False)
    radius.setflags(write=False)
    return center, radius, frozenset(classes)


# ---------------------------------------------------------------------------
# Mutation separation.
# ---------------------------------------------------------------------------

_EVERY_FIXTURE = (
    StructuredMutation.OMIT_SQRT_D,
    StructuredMutation.EXP_ENCODING,
    StructuredMutation.K_ROW_PLUS_ONE,
    StructuredMutation.K_ROW_MINUS_ONE,
    StructuredMutation.Q_HEAD_ROTATE,
    StructuredMutation.K_HEAD_ROTATE,
    StructuredMutation.OMIT_RESCALE,
    StructuredMutation.DENOMINATOR_PLUS_ONE,
)
_FIXTURE_MUTATIONS = {
    StructuredFixture.CYCLIC: (StructuredMutation.INVERSE_SHIFT,),
    StructuredFixture.WALSH: (*_OMITTED_WALSH_BLOCK, StructuredMutation.WALSH_SIGN),
}
_MASK_MUTATIONS = {
    "causal": (
        StructuredMutation.INCLUDE_FUTURE_KEY,
        StructuredMutation.EXCLUDE_LAST_KEY,
    ),
    "window": (
        StructuredMutation.INCLUDE_OLD_WINDOW_KEY,
        StructuredMutation.EXCLUDE_OLDEST_WINDOW_KEY,
    ),
}


def applicable_mutations(case: StructuredCase) -> tuple[StructuredMutation, ...]:
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
    (``exposure`` and ``b:q:h:d``).

    ``gap > 0`` certifies sensitivity: the two enclosures are disjoint at that
    coordinate, so no stored value satisfies both semantics. The certificate
    is sufficient, not necessary: Tier B is an outer enclosure, and two outer
    enclosures can overlap where the reachable sets do not, so ``gap == 0``
    decides nothing about the mutation."""

    mutation: StructuredMutation
    gap: int
    exposure: str | None
    coordinate: tuple[int, int, int, int] | None


# Block, tile, window and causal-diagonal edges of both quick kernels. A gap on
# a subset of rows lower-bounds the full gap, which is enough to prove that the
# mutation is caught.
STRUCTURED_SEPARATION_ROWS = (0, 1, 6, 7, 31, 63, 64, 65, 127, 128, 191, 192, 255)

SEPARATION_LABELS = (
    "owned by another check",
    "owned by another fixture",
    "needs a stronger fixture",
    "out of scope",
)
# Quick cells whose correct and mutated Tier B overlap at every coordinate,
# keyed by ``(case_id, mutation)``. Tier B is never loosened to separate them.
# On the causal kernel the identity, cyclic and Walsh rows all reach their max
# in the first visited tile, so every later ``alpha = E(0)`` admits 1 and the
# omitted rescale is a member of the correct reachable set; late max, whose
# max arrives in a later tile, separates it on the same kernel.
STRUCTURED_UNSEPARATED: Mapping[tuple[str, StructuredMutation], str] = MappingProxyType(
    {
        (case.case_id, StructuredMutation.OMIT_RESCALE): "owned by another fixture"
        for case in STRUCTURED_QUICK_CASES
        if case.mask == "causal" and case.fixture is not StructuredFixture.LATE_MAX
    }
)


def mutation_separation(
    case: StructuredCase,
    mutation: StructuredMutation,
    *,
    rows: Iterable[int] | None = STRUCTURED_SEPARATION_ROWS,
) -> Separation:
    """Evaluate the separation of one mutation over every exposure; ``rows=None``
    uses every row. A row whose mutated enclosure is not derived proves
    nothing; separation needs one coordinate, so
    leaving it out lower-bounds the gap."""
    mutation = StructuredMutation(mutation)
    if mutation not in applicable_mutations(case):
        raise ValueError(f"mutation {mutation.value} does not apply to {case.case_id}")
    best = Separation(mutation, 0, None, None)
    queries = derived_rows(
        case, range(case.base.seqlen_q) if rows is None else rows, mutation
    )
    for exposure in STRUCTURED_EXPOSURES:
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


def row_scope(
    case: StructuredCase, query: int, mutation: StructuredMutation | None
) -> str | None:
    """Why the enclosure of ``query`` is not derived under ``mutation`` for some
    head (more than one fully masked leading tile, no legal tile, or a
    denominator outside the A-DIV domain), or
    ``None`` when every head is covered."""
    try:
        for head in range(case.base.nhead_q):
            row_trace(case, head, query, mutation)
    except StructuredScopeError as error:
        return str(error)
    return None


def derived_rows(
    case: StructuredCase, rows: Iterable[int], mutation: StructuredMutation | None
) -> tuple[int, ...]:
    """The rows of ``rows`` whose enclosure under ``mutation`` is derived for
    every head (:func:`row_scope`). The correct semantics are never filtered:
    a correct row outside the derivation must fail closed in
    :func:`row_trace`."""
    rows = tuple(rows)
    if mutation is None:
        return rows
    return tuple(query for query in rows if row_scope(case, query, mutation) is None)


def format_separation(case: StructuredCase, separation: Separation) -> str:
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

# One MFMA call on one accumulator element is an atom. Zero neutrality and the
# sign of a zero result are part of A-MFMA, not consequences of
# representability: a structural coordinate must store the pattern +0, and it
# does because every zero PV product is +0 (V and the cast weights are
# nonnegative) and an all-zero atom returns its +0 accumulator unchanged.
MFMA_ASSUMPTION = (
    "A-MFMA v1: an MFMA atom whose products are all zero returns its "
    "accumulator c unchanged (c = +0 stays +0); with one nonzero product p it "
    "returns RD or RU of c + p"
)
# The exponents and the total magnitude are those of the Walsh score atoms and
# no more: products +-2**r with -4 <= r <= 0, at most 64 of them in a score.
A_QK_EXPONENTS = (-4, 0)
A_QK_MAGNITUDE = 64
QK_ASSUMPTION = (
    "A-QK v1: an MFMA atom whose accumulator and eight products are multiples "
    "of 2**e, -4 <= e <= 0, with total magnitude at most 64 * 2**e returns "
    "their exact sum"
)
# Rounding and conversion: three assumptions, one per mechanism. They fail
# differently and a compiler change touches one of them at a time (the BF16
# conversion is an instruction sequence the compiler chooses).
MODE32_ASSUMPTION = (
    "A-MODE32 v1: each FP32 add, subtract and multiply rounds its exact result "
    "to nearest even"
)
CVT_F16_ASSUMPTION = (
    "A-CVT-F16 v1: the FP32-to-FP16 conversion, standalone or as the final "
    "rounding of a mixed-precision FMA, rounds to nearest even and keeps "
    "subnormals"
)
CVT_BF16_ASSUMPTION = (
    "A-CVT-BF16 v1: the instruction sequence selected for the FP32-to-BF16 "
    "conversion rounds to nearest even and keeps subnormals"
)
# Every named assumption Tier B is conditional on, in the order of the
# soundness theorem. A calibration entry lists exactly these versions.
STRUCTURED_ASSUMPTION_STATEMENTS = (
    EXP2_ASSUMPTION,
    MFMA_ASSUMPTION,
    QK_ASSUMPTION,
    DIV_ASSUMPTION,
    EPILOGUE_ASSUMPTION,
    MODE32_ASSUMPTION,
    CVT_F16_ASSUMPTION,
    CVT_BF16_ASSUMPTION,
)
STRUCTURED_ASSUMPTIONS = tuple(
    statement.split(":", 1)[0] for statement in STRUCTURED_ASSUMPTION_STATEMENTS
)
# Looser v_exp_f32 bound used only to attribute a failure to A-E.
_RELAXED_EXP2_ULPS = 2


class StructuredLabel(str, Enum):
    """What an observed coordinate is judged to be.

    A value outside Tier B that equals the store of a single violated
    assumption (:class:`StructuredViolation`) carries that assumption's label. The
    other assumption labels are heuristics, checked in the order below. No
    label is a pass, and none enlarges Tier B.

    A zero-gap mutation (the fixture lacks sensitivity) is a CPU-side label,
    ``STRUCTURED_UNSEPARATED``, not an output label."""

    PASS = "pass"
    INVALID = "invalid"  # NaN (including the unwritten prefill) or Inf
    NEGATIVE = "negative"  # sign bit set, -0 included
    STRUCTURAL = "structural"  # a required +0 coordinate stores another pattern
    A_MODE32 = "assumption A-MODE32"  # FP32 arithmetic rounded toward zero
    # A truncating FP16 conversion; or, at an FP16-subnormal target, a value
    # near it or a flushed +0.
    A_CVT_F16 = "assumption A-CVT-F16"
    A_CVT_BF16 = "assumption A-CVT-BF16"  # a truncating BF16 conversion
    A_F = "assumption A-F"  # an FP16 store that is neither epilogue rounding
    A_DIV = "assumption A-DIV"  # the reciprocal is a neighbor of RN(1 / l)
    A_QK = "assumption A-QK"  # a Walsh score accumulator is inexact
    A_MFMA = "assumption A-MFMA"  # folded addition near target, basis clean
    A_E = "assumption A-E"  # explained only by a looser v_exp_f32
    PATH_MODEL = "path model"  # outside Tier B, inside Tier A: re-pin, never widen
    NUMERICAL = "numerical"  # outside Tier A, or outside a mutated Tier B


STRUCTURED_FAILURE_LABELS = tuple(
    label for label in StructuredLabel if label is not StructuredLabel.PASS
)
_VIOLATION_LABEL: Mapping[StructuredViolation, StructuredLabel] = MappingProxyType(
    {
        StructuredViolation.E_LOW: StructuredLabel.A_E,
        StructuredViolation.E_HIGH: StructuredLabel.A_E,
        StructuredViolation.MFMA_LOW: StructuredLabel.A_MFMA,
        StructuredViolation.MFMA_HIGH: StructuredLabel.A_MFMA,
        StructuredViolation.QK_DOWN: StructuredLabel.A_QK,
        StructuredViolation.QK_UP: StructuredLabel.A_QK,
        StructuredViolation.DIV_LOW: StructuredLabel.A_DIV,
        StructuredViolation.DIV_HIGH: StructuredLabel.A_DIV,
        StructuredViolation.F_TRUNCATED: StructuredLabel.A_F,
        StructuredViolation.MODE32_RZ_ARITHMETIC: StructuredLabel.A_MODE32,
        StructuredViolation.CVT_F16_TRUNCATING: StructuredLabel.A_CVT_F16,
        StructuredViolation.CVT_BF16_TRUNCATING: StructuredLabel.A_CVT_BF16,
    }
)
# Violations whose stores stay inside Tier B at every coordinate of the quick
# matrix (all 1,310,720 coordinates of each case, both epilogue roundings):
# the stored bits cannot show them. This records the check's sensitivity to the
# primitive; it is no evidence for the assumption. The conversions are the
# observable ones: a truncating cast, and for FP16 a truncated store.
STRUCTURED_UNOBSERVED_VIOLATIONS = frozenset(
    {
        StructuredViolation.E_LOW,
        StructuredViolation.E_HIGH,
        StructuredViolation.MFMA_LOW,
        StructuredViolation.MFMA_HIGH,
        StructuredViolation.QK_DOWN,
        StructuredViolation.QK_UP,
        StructuredViolation.DIV_LOW,
        StructuredViolation.DIV_HIGH,
        StructuredViolation.MODE32_RZ_ARITHMETIC,
    }
)
# No violation stores a value farther than this many target ULPs outside
# Tier B (a contract checks it), so a farther value is not compared to them.
_VIOLATION_REACH = 3
_FP16_SMALLEST_NORMAL_BITS = 0x0400


@dataclass(frozen=True)
class StructuredGroupStats:
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
class StructuredResult:
    """Tier B verdict of one observation, with the taxonomy of every failure."""

    mutation: StructuredMutation | None
    violations: tuple[str, ...]
    labels: Mapping[str, int]
    first: Mapping[str, tuple[str, tuple[int, int, int, int]]]
    compared: int
    exact: int
    ambiguous: int
    ambiguous_high: int
    groups: Mapping[str, Mapping[tuple[int, int], StructuredGroupStats]]

    @property
    def exact_rate(self) -> float:
        return self.exact / self.compared if self.compared else 0.0


def _genuine_addition(case: StructuredCase, query: int, head: int, dim: int) -> bool:
    """Whether a folded coordinate's MFMA adds two nonzero values."""
    trace = row_trace(case, head, query)
    exponent = case.head_scale_exponents[head]
    nonzero = 0
    for step in trace.steps:
        key = exposure_key(case, "folded", dim, step.tile)
        weight = step.weights[key - step.tile * case.block_n]
        if _scaled_cast(case.dtype, weight, exponent).hi > 0:
            nonzero += 1
    return nonzero >= 2


@cache
def _violation_alternatives(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    violation: StructuredViolation,
) -> tuple[np.ndarray, np.ndarray]:
    return epilogue_alternatives(case, exposure, query, head, violation)


def matching_violation(
    case: StructuredCase, exposure: str, query: int, head: int, dim: int, bits: int
) -> StructuredViolation | None:
    """The first violation, in declaration order, whose IR-exact path stores
    ``bits`` at this coordinate under either epilogue rounding."""
    for violation in applicable_violations(case):
        double, single = _violation_alternatives(case, exposure, query, head, violation)
        if bits in (int(double[dim]), int(single[dim])):
            return violation
    return None


def _attribute(
    case: StructuredCase,
    exposure: str,
    query: int,
    head: int,
    dim: int,
    bits: int,
    outside: int,
    center: int,
    radius: int,
    basis_clean: bool,
    relaxed: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> StructuredLabel:
    """Label a nonnegative finite value ``outside`` target ULPs outside the
    correct Tier B.

    A value that a single violated assumption stores carries that assumption.
    Otherwise an assumption label needs a value its failure can produce:
    A-CVT-F16 a value inside Tier A or a flushed ``+0`` at an FP16-subnormal
    target (a conversion that does not keep subnormals), A-MFMA a value inside
    Tier A. A far value there is numerical, not a conversion. ``relaxed()``
    returns the row's Tier B under a 2-ULP exponential, computed at most once
    per row."""
    if outside <= _VIOLATION_REACH:
        violation = matching_violation(case, exposure, query, head, dim, bits)
        if violation is not None:
            return _VIOLATION_LABEL[violation]
    near = d_ulp(bits, center, case.dtype) <= radius
    if (
        case.dtype == "fp16"
        and 0 < center < _FP16_SMALLEST_NORMAL_BITS
        and (near or bits == 0)
    ):
        return StructuredLabel.A_CVT_F16
    if (
        near
        and exposure == "folded"
        and basis_clean
        and _genuine_addition(case, query, head, dim)
    ):
        return StructuredLabel.A_MFMA
    low, high = relaxed()
    if low[dim] <= bits <= high[dim]:
        return StructuredLabel.A_E
    if near:
        return StructuredLabel.PATH_MODEL
    return StructuredLabel.NUMERICAL


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

    def freeze(self) -> StructuredGroupStats:
        return StructuredGroupStats(
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


def analyze_structured_policy(
    case: StructuredCase,
    observation: Observation,
    *,
    mutation: StructuredMutation | str | None = None,
    rows: Iterable[int] | None = None,
    require_complete: bool = True,
) -> StructuredResult:
    """Judge raw output bits by Tier B membership, coordinate by coordinate.

    A coordinate passes iff its bits lie in Tier B; a structural zero (masked
    key, Tier B ``{+0}``) must store exactly ``+0``. Every other outcome fails
    and carries the strongest contract it still satisfies (``StructuredLabel``). With
    ``mutation`` the observation is judged against the mutated Tier B and must
    be rejected; failures are then only labeled invalid, negative, structural
    or numerical. ``rows`` restricts the judgment to those query rows; a
    mutated judgment also leaves out the rows whose mutated enclosure is not
    derived (:func:`derived_rows`).

    Observed values are only compared: nothing here can enlarge Tier B or the
    exponential candidate sets."""
    selected = None if mutation is None else StructuredMutation(mutation)
    if selected is not None and selected not in applicable_mutations(case):
        raise ValueError(f"mutation {selected.value} does not apply to {case.case_id}")
    validate_observation(case.base, observation, require_complete=require_complete)
    every_row = tuple(range(case.base.seqlen_q))
    queries = derived_rows(case, every_row if rows is None else rows, selected)
    if not queries:
        raise ValueError("rows must select at least one query")
    whole = queries == every_row
    exponents = np.array(case.head_scale_exponents)

    # Pass 1: membership masks per exposure.
    judged: dict[str, tuple] = {}
    for exposure, raw in observation.exposures:
        if whole:
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
        labels = np.full(bits.shape, StructuredLabel.PASS.value, dtype=object)
        labels[~finite] = StructuredLabel.INVALID.value
        labels[finite & sign] = StructuredLabel.NEGATIVE.value
        labels[valid & required_zero & (bits != 0)] = StructuredLabel.STRUCTURAL.value
        outside = valid & ~required_zero & ~inside
        judged[exposure] = (
            bits,
            low,
            high,
            center,
            radius,
            required_zero,
            labels,
            outside,
        )

    basis = [name for name in judged if name != "folded"]

    def basis_clean(batch: int, row: int, head: int) -> bool:
        if len(basis) != len(BASIS_OFFSETS):
            return False
        return all(
            np.all(judged[name][6][batch, row, head] == StructuredLabel.PASS.value)
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
        bits, low, high, center, radius, _, labels, outside = judgment
        for batch, row, head, dim in zip(*np.nonzero(outside)):
            query = queries[row]
            if selected is not None:
                label = StructuredLabel.NUMERICAL
            else:
                stored = int(bits[batch, row, head, dim])
                label = _attribute(
                    case,
                    exposure,
                    query,
                    int(head),
                    int(dim),
                    stored,
                    max(
                        int(low[batch, row, head, dim]) - stored,
                        stored - int(high[batch, row, head, dim]),
                    ),
                    int(center[batch, row, head, dim]),
                    int(radius[batch, row, head, dim]),
                    basis_clean(batch, row, head),
                    lambda e=exposure, q=query, h=int(head): relaxed(e, q, h),
                )
            labels[batch, row, head, dim] = label.value

    # Pass 3: counts, first failures and per-group evidence.
    totals: dict[str, int] = {label.value: 0 for label in StructuredLabel}
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
                    if label != StructuredLabel.PASS.value:
                        group.labels[label] = group.labels.get(label, 0) + 1
                if decoded is not None:
                    stored = float(
                        np.max(
                            np.abs(
                                decoded[cell].sum(axis=-1) - targets[cell].sum(axis=-1)
                            )
                        )
                    )
                    real = float(np.max(np.abs(decoded[cell].sum(axis=-1) - 1.0)))
                    group.max_row_sum_stored = max(group.max_row_sum_stored, stored)
                    group.max_row_sum_real = max(group.max_row_sum_real, real)
        for label in StructuredLabel:
            hits = labels == label.value
            count = int(np.count_nonzero(hits))
            totals[label.value] += count
            if count and label is not StructuredLabel.PASS and label.value not in first:
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
        (label for label in STRUCTURED_FAILURE_LABELS if totals[label.value]),
        key=lambda label: -totals[label.value],
    )
    violations = [
        f"{label.value}={totals[label.value]} first at {first[label.value][0]} "
        f"b:q:h:d={':'.join(str(value) for value in first[label.value][1])}"
        for label in failing
    ]
    if compared == 0:
        violations.append("no_coordinate_compared")
    return StructuredResult(
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


def format_structured_result(result: StructuredResult) -> str:
    violations = "; ".join(result.violations) if result.violations else "none"
    return (
        f"violations={violations}; exact={result.exact}/{result.compared} "
        f"({result.exact_rate:.6f}); two_value_B={result.ambiguous} "
        f"(upper observed {result.ambiguous_high})"
    )


@dataclass(frozen=True)
class StructuredDiagnostics:
    """Records-only evidence (never a verdict) on sampled rows.

    ``ir_exact`` counts coordinates equal to the IR-exact emulation (lane
    order, ``E = RN(2**x)``, RN MFMA, double-rounded epilogue). When the
    launched code object is a named epilogue artifact, ``artifact_exact``
    counts agreement with that artifact's per-site epilogue; agreement
    everywhere is then consistent with a correctly rounded exponential, which the
    IR-exact count alone cannot show on an affected compiler. ``mfma_rn``
    counts folded coordinates with a genuine MFMA addition that lie in the
    RN-accumulate Tier B."""

    ir_exact: int
    artifact: str | None
    artifact_exact: int | None
    path_compared: int
    mfma_additions: int
    mfma_rn: int
    widths: Mapping[int, int]


def structured_diagnostics(
    case: StructuredCase,
    observation: Observation,
    *,
    rows: Iterable[int] = STRUCTURED_SEPARATION_ROWS,
    artifact: str | None = None,
) -> StructuredDiagnostics:
    if artifact is not None and artifact not in STRUCTURED_EPILOGUE_ARTIFACTS:
        raise ValueError(f"no epilogue artifact named {artifact!r}")
    lanes = np.arange(case.base.head_dim) % 4
    single_lanes = (
        np.isin(lanes, sorted(STRUCTURED_EPILOGUE_ARTIFACTS[artifact].single_lanes))
        if artifact
        else None
    )
    ir_exact = artifact_exact = path_compared = additions = rn = 0
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
                double, single = epilogue_alternatives(case, exposure, query, head)
                ir_exact += int(np.count_nonzero(observed == double))
                if single_lanes is not None:
                    expected = np.where(single_lanes, single, double)
                    artifact_exact += int(np.count_nonzero(observed == expected))
                path_compared += observed.size
                if exposure != "folded":
                    continue
                rn_low, rn_high = tier_b_row(case, exposure, query, head, mfma="rn")
                for dim in range(case.base.head_dim):
                    if _genuine_addition(case, query, head, dim):
                        additions += 1
                        rn += int(rn_low[dim] <= observed[dim] <= rn_high[dim])
    return StructuredDiagnostics(
        ir_exact,
        artifact,
        artifact_exact if artifact else None,
        path_compared,
        additions,
        rn,
        MappingProxyType(dict(sorted(widths.items()))),
    )


def structured_records(
    case: StructuredCase,
    candidate_name: str,
    results: Mapping[str, StructuredResult],
    *,
    deterministic: bool,
    diagnostics: StructuredDiagnostics,
    separations: Mapping[str, Separation],
    code_object: Mapping[str, str] | None = None,
) -> list[dict]:
    """JSON-ready calibration records: per-group evidence for every analysis,
    one summary per analysis, and one diagnostics record for the case.
    ``code_object`` (compiler identity and HSACO digest) is evidence only."""
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
                        "record_type": "rocke_structured_group",
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
                "record_type": "rocke_structured_summary",
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
                "cpu_gap_at": (
                    None
                    if separation is None or separation.coordinate is None
                    else [separation.exposure, list(separation.coordinate)]
                ),
            }
        )
    records.append(
        {
            "record_type": "rocke_structured_diagnostics",
            **identity,
            "deterministic": deterministic,
            "assumptions": list(STRUCTURED_ASSUMPTIONS),
            "code_object": dict(code_object or {}),
            "ir_exact": diagnostics.ir_exact,
            "epilogue_artifact": diagnostics.artifact,
            "artifact_exact": diagnostics.artifact_exact,
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
# Calibration: identity and scope only.
# ---------------------------------------------------------------------------

STRUCTURED_POLICY = "tier_b"


@dataclass(frozen=True)
class StructuredCalibrationEntry:
    """Ratifies that the Tier B contract applies to one kernel identity.

    It names the policy, the assumption versions, the case ids and the dispatch
    fingerprint, and has no width, ULP or tolerance field: no measurement can
    enlarge Tier B. An entry that omits a named assumption, or lists another
    version of one, ratifies a different contract and is refused. Provenance
    belongs in the commit that edits the table."""

    policy: str
    assumptions: tuple[str, ...]
    case_ids: tuple[str, ...]
    spec: SpecFingerprint

    def __post_init__(self) -> None:
        if self.policy != STRUCTURED_POLICY:
            raise ValueError(f"only policy={STRUCTURED_POLICY!r} is supported")
        if tuple(self.assumptions) != STRUCTURED_ASSUMPTIONS:
            raise ValueError(f"assumptions must be {STRUCTURED_ASSUMPTIONS}")
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case_ids must be nonempty and unique")

    def mismatch(self, case: StructuredCase, spec: SpecFingerprint) -> tuple[str, ...]:
        problems: list[str] = []
        if case.case_id not in self.case_ids:
            problems.append(f"case {case.case_id} not in calibrated {self.case_ids}")
        for item in fields(SpecFingerprint):
            calibrated = getattr(self.spec, item.name)
            resolved = getattr(spec, item.name)
            if calibrated != resolved:
                problems.append(
                    f"{item.name}: calibrated {calibrated!r}, resolved {resolved!r}"
                )
        return tuple(problems)


# Ratified on gfx942. Case ids are literal, so changing a fixture, its head
# scales or the quick shapes fails closed instead of following silently. The
# dispatch fingerprint is the one of ``STRUCTURED_QUICK_SPECS``.
STRUCTURED_CALIBRATION: Mapping[tuple[str, str, str], StructuredCalibrationEntry] = (
    MappingProxyType(
        {
            ("gfx942", "attention_gfx942_dense", "fp16"): StructuredCalibrationEntry(
                policy=STRUCTURED_POLICY,
                assumptions=(
                    "A-E v1",
                    "A-MFMA v1",
                    "A-QK v1",
                    "A-DIV v1",
                    "A-F v1",
                    "A-MODE32 v1",
                    "A-CVT-F16 v1",
                    "A-CVT-BF16 v1",
                ),
                case_ids=(
                    "fp16-causal-b1-sq256-sk256-hq4-hk4-d64-identity-p0xp6,p1xn5,n2xn2,n0xp2",
                    "fp16-causal-b1-sq256-sk256-hq4-hk4-d64-cyclic-r37-p0xp6,p1xn5,n2xn2,n0xp2",
                    "fp16-causal-b1-sq256-sk256-hq4-hk4-d64-walsh-p0xp0,n-2xn0,p-2xp-2,n0xn-1",
                    "fp16-causal-b1-sq256-sk256-hq4-hk4-d64-late_max-l64-p0xp6,p1xn5,n2xn2,n0xp2",
                ),
                spec=STRUCTURED_QUICK_SPECS["fp16"],
            ),
            ("gfx942", "attention_gfx942_dense", "bf16"): StructuredCalibrationEntry(
                policy=STRUCTURED_POLICY,
                assumptions=(
                    "A-E v1",
                    "A-MFMA v1",
                    "A-QK v1",
                    "A-DIV v1",
                    "A-F v1",
                    "A-MODE32 v1",
                    "A-CVT-F16 v1",
                    "A-CVT-BF16 v1",
                ),
                case_ids=(
                    "bf16-window64-b1-sq256-sk256-hq4-hk4-d64-identity-p0xp6,p1xn5,n2xn2,n0xp2",
                    "bf16-window64-b1-sq256-sk256-hq4-hk4-d64-cyclic-r37-p0xp6,p1xn5,n2xn2,n0xp2",
                    "bf16-window64-b1-sq256-sk256-hq4-hk4-d64-walsh-p0xp0,n-2xn0,p-2xp-2,n0xn-1",
                    "bf16-window64-b1-sq256-sk256-hq4-hk4-d64-late_max-l64-p0xp6,p1xn5,n2xn2,n0xp2",
                ),
                spec=STRUCTURED_QUICK_SPECS["bf16"],
            ),
        }
    )
)

_STRUCTURED_RECALIBRATION_REMEDY = (
    "re-run the case on gfx942 with ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>, "
    "review the records (zero coordinates outside Tier B, every mutation "
    "rejected, deterministic), and update STRUCTURED_CALIBRATION (spec, case_ids) "
    "deliberately"
)


def require_structured_calibration(
    arch: str,
    candidate_name: str,
    case: StructuredCase,
    spec: SpecFingerprint,
    *,
    table: Mapping[
        tuple[str, str, str], StructuredCalibrationEntry
    ] = STRUCTURED_CALIBRATION,
    detail: str = "",
) -> StructuredCalibrationEntry:
    """Return the entry that authorizes this resolved case, or fail closed."""
    key = (arch, candidate_name, case.dtype)
    suffix = f"; {detail}" if detail else ""
    entry = table.get(key)
    if entry is None:
        raise AssertionError(
            f"no structured-logit calibration for {key}, case={case.case_id}; "
            f"{_STRUCTURED_RECALIBRATION_REMEDY}{suffix}"
        )
    problems = entry.mismatch(case, spec)
    if problems:
        raise AssertionError(
            f"structured-logit calibration {key} does not cover case={case.case_id}: "
            f"{'; '.join(problems)}; {_STRUCTURED_RECALIBRATION_REMEDY}{suffix}"
        )
    return entry
