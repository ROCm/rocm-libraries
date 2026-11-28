# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free fixture, shift, derived tolerance and analyzer for the value-shift
relation.

The attention weights of a row sum to one, so adding one vector ``c`` to every
value token adds ``c`` to every output row. The relation compares two launches
of the subject, on ``V`` and on ``V + c``, and judges
``E = O(V + c) - O(V) - c`` element by element. No expected output is computed
and there is no attention implementation in this module: the tolerance follows
from the kernel's arithmetic path and the launched values, and its last term,
the rounding of the store, from the stored bits themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy as np
from rocke.numeric.bf16 import f32_to_bf16

from ._analytical import _DTYPES, _FORMAT, DenseCaseShape, decode_bits, encode_bits

# One period of the shift: alternating signs and three magnitudes, every
# component nonzero and exact in both subject dtypes.
_SHIFT_PERIOD = (0.5, -0.25, 0.125, -0.5)

# Q and K entries are n/8 with |n| <= 24, so every score is finite and far from
# the kernel's initial running maximum, whatever the keys are.
_LATTICE_STEPS = 8
_LATTICE_LIMIT = 24
_LATTICE_DRAWS = 4
_LATTICE_DRAW_VALUES = 2 * _LATTICE_LIMIT // _LATTICE_DRAWS + 1
_SEED_Q = 0x71
_SEED_K = 0x6B
_SEED_V = 0x76
_PROFILES = ("moderate", "concentrated")
# The concentrated profile: every query is nonnegative and one key is this level
# in every dimension, so that key holds nearly all of every row's probability.
_DOMINANT_KEY = 70
_DOMINANT_LEVEL = 2.0

# One FP32 unit of faithful rounding: an operation returns one of the two FP32
# neighbours of its exact result.
_FP32_UNIT = Fraction(1, 2**23)
# Products one MFMA call adds to its accumulator.
_ATOM_PRODUCTS = 8
# Largest absolute error of an FP32 product whose exact value is below the
# normal range, and of an MFMA result that is subnormal.
_FP32_UNDERFLOW = Fraction(1, 2**126)
_ATOM_UNDERFLOW = Fraction(1, 2**150)
# The residual is two FP64 operations on decoded values; each rounds by at most
# this fraction of its operands.
_HOST_UNIT = 2.0**-52
# Covers the FP64 rounding of the tolerance's own evaluation, upward.
_HOST_MARGIN = 1.0 + 2.0**-48

# Relative output scales of the normalization mutation, largest first.
OUTPUT_SCALE_SWEEP = tuple(2.0**-power for power in range(2, 13))


class ShiftMutation(str, Enum):
    """Follow-up inputs whose applied shift is not the expected one."""

    OMIT_SECOND_HALF = "omit_second_half"
    RESCALE_SECOND_HALF = "rescale_second_half"


@dataclass(frozen=True)
class ValueShiftCase(DenseCaseShape):
    """Logical unmasked dense-attention shape and attention profile for one
    value-shift case."""

    profile: str = "moderate"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.profile not in _PROFILES:
            raise ValueError(f"profile must be one of {sorted(_PROFILES)}")

    @property
    def mask(self) -> str:
        """The tolerance is derived for the unmasked path only."""
        return "none"

    @property
    def window(self) -> int:
        """No sliding window: a window is a positional mask."""
        return 0

    @property
    def case_id(self) -> str:
        """Return a stable identifier for test ids and resolved-subject checks."""
        return self._shape_id(f"{self.mask}-{self.profile}")

    def require_scope(self) -> None:
        """Refuse a case the tolerance's derivation or the fixture does not cover.

        The accumulation term counts MFMA calls of eight products, the lattice
        keeps every score finite only while the contraction stays in the exact
        FP32 range, and the concentrated profile needs its dominant key."""
        if self.seqlen_k % _ATOM_PRODUCTS:
            raise ValueError(
                f"case {self.case_id}: seqlen_k must be a multiple of "
                f"{_ATOM_PRODUCTS}, the products of one MFMA call"
            )
        if self.head_dim * _LATTICE_LIMIT**2 > 1 << 24:
            raise ValueError(
                f"case {self.case_id}: the lattice contraction is exact in FP32 "
                f"only while head_dim * {_LATTICE_LIMIT}**2 <= 2**24"
            )
        if self.profile == "concentrated" and self.seqlen_k <= _DOMINANT_KEY:
            raise ValueError(
                f"case {self.case_id}: the concentrated profile needs more than "
                f"{_DOMINANT_KEY} key tokens"
            )


# Unmasked GQA (four query heads per KV head), both subject dtypes and both
# attention profiles. Square, because the gfx942 dense kernel serves unmasked
# attention only at seqlen_q == seqlen_k. The GPU lane runs exactly these; a CPU
# contract pins its parametrization to them, so a case cannot silently stop
# executing.
VALUE_SHIFT_QUICK_CASES = tuple(
    ValueShiftCase(1, 256, 256, 8, 2, 64, dtype, profile)
    for dtype in ("fp16", "bf16")
    for profile in _PROFILES
)


def _hash64(count: int, seed: int) -> np.ndarray:
    """splitmix64 of the flat element index offset by ``seed``, as ``uint64``.

    An explicit integer hash, so the fixture does not depend on a library's
    random stream."""
    state = np.arange(count, dtype=np.uint64) + (np.uint64(seed) << np.uint64(32))
    state = state + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state ^ (state >> np.uint64(31))


def _lattice(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Bell-shaped pseudo-random multiples of 1/8 in ``[-3, 3]``.

    The centred sum of four uniform draws from ``0..12``, each taken from one
    16-bit field of the element's hash."""
    bits = _hash64(int(np.prod(shape)), seed)
    total = np.zeros(bits.shape, dtype=np.int64)
    for field in range(_LATTICE_DRAWS):
        draw = (bits >> np.uint64(16 * field)) & np.uint64(0xFFFF)
        scaled = (draw * np.uint64(_LATTICE_DRAW_VALUES)) >> np.uint64(16)
        total += scaled.astype(np.int64)
    values = (total - _LATTICE_LIMIT).astype(np.float32) / np.float32(_LATTICE_STEPS)
    return values.reshape(shape)


def _signs(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Pseudo-random ``+1`` / ``-1`` values from the top bit of each element's hash."""
    bits = _hash64(int(np.prod(shape)), seed)
    negative = (bits >> np.uint64(63)).astype(np.float32)
    return (np.float32(1.0) - np.float32(2.0) * negative).reshape(shape)


def value_shift_inputs(
    case: ValueShiftCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Q and K on the lattice and sign-valued V, as FP32 values that are
    exact in FP16 and BF16.

    ``moderate`` gives every row nonuniform attention spread over many keys.
    ``concentrated`` makes every query nonnegative and one key large in every
    dimension, so that key holds nearly all of each row's probability and most
    other weights are far below it: in FP16 they reach the kernel's probability
    cast as subnormals. V is ``+1`` or ``-1`` per element, so ``V + c`` is exact
    for the lane's shift."""
    case.require_scope()
    q = _lattice(case.output_shape, _SEED_Q)
    k = _lattice(case.key_shape, _SEED_K)
    v = _signs(case.key_shape, _SEED_V)
    if case.profile == "concentrated":
        q = np.abs(q)
        k[:, _DOMINANT_KEY] = np.float32(_DOMINANT_LEVEL)
    return q, k, v


def require_finite_scores(q: np.ndarray, k: np.ndarray) -> None:
    """Refuse Q or K off the lattice ``n / 8`` with ``|n| <= 24``.

    The tolerance needs every score finite and above the kernel's initial
    running maximum: then each row's largest score has weight ``exp2(0)`` and
    the normalizer is bounded away from zero. On the lattice that holds by
    arithmetic; for other inputs it is not certified and the lane must not
    launch."""
    for name, values in (("Q", q), ("K", k)):
        scaled = np.asarray(values, dtype=np.float64) * _LATTICE_STEPS
        if scaled.size == 0 or not np.all(np.isfinite(scaled)):
            raise ValueError(f"cannot certify finite scores: {name} is not finite")
        if (
            not np.array_equal(scaled, np.rint(scaled))
            or np.max(np.abs(scaled)) > _LATTICE_LIMIT
        ):
            raise ValueError(
                f"cannot certify finite scores: {name} is not on the lattice "
                f"n/{_LATTICE_STEPS} with |n| <= {_LATTICE_LIMIT}"
            )


def shift_vector(head_dim: int) -> np.ndarray:
    """The expected shift ``c_d = (1/2, -1/4, 1/8, -1/2)[d mod 4]`` as FP64."""
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return np.array(
        [_SHIFT_PERIOD[dim % len(_SHIFT_PERIOD)] for dim in range(head_dim)],
        dtype=np.float64,
    )


def applied_shift(
    head_dim: int, mutation: ShiftMutation | str | None = None
) -> np.ndarray:
    """The shift a follow-up launch applies: the expected one, or a mutation of
    it that leaves the second half of the columns unshifted or shifted by half
    as much. The analyzer keeps expecting :func:`shift_vector`."""
    shift = shift_vector(head_dim)
    if mutation is None:
        return shift
    try:
        selected = (
            mutation if isinstance(mutation, ShiftMutation) else ShiftMutation(mutation)
        )
    except ValueError as error:
        raise ValueError(f"unknown shift mutation {mutation!r}") from error
    second_half = head_dim // 2
    if selected is ShiftMutation.OMIT_SECOND_HALF:
        shift[second_half:] = 0.0
    else:
        shift[second_half:] *= 0.5
    return shift


def _require_value_bits(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.uint16:
        raise ValueError(f"{name} must contain raw uint16 bits")
    if raw.ndim != 4:
        raise ValueError(f"{name} must be a [B, Sk, Hkv, D] array")
    return raw


def _require_shift(shift: np.ndarray, head_dim: int) -> np.ndarray:
    addend = np.asarray(shift, dtype=np.float64)
    if addend.shape != (head_dim,) or not np.all(np.isfinite(addend)):
        raise ValueError(f"the shift must be {head_dim} finite values")
    return addend


def _exact_sum(values: np.ndarray, addend: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The FP64 sum ``values + addend`` and, per element, whether it is the
    exact sum: the rounding error of an FP64 addition is itself an FP64 number,
    recovered here without rounding."""
    addend = np.broadcast_to(addend, values.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        total = values + addend
        virtual = total - values
        error = (values - (total - virtual)) + (addend - virtual)
    return total, np.isfinite(total) & (error == 0.0)


def _first(flags: np.ndarray) -> tuple[int, ...]:
    return tuple(int(index) for index in np.argwhere(flags)[0])


def shifted_value_bits(values: np.ndarray, shift: np.ndarray, dtype: str) -> np.ndarray:
    """Raw bits of ``V + c`` for stored V bits, refusing any rounding.

    The relation's tolerance has no term for a rounded input transformation, so
    the follow-up tensor must be the exact sum: it is formed in FP64, checked to
    be exact there and in FP32, and encoded by :func:`encode_bits`, which refuses
    a value the dtype cannot hold."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    raw = _require_value_bits(values, "values")
    addend = _require_shift(shift, raw.shape[-1])
    source = decode_bits(raw, dtype)
    if not np.all(np.isfinite(source)):
        raise ValueError("cannot shift V: it is not finite")
    total, exact = _exact_sum(source, addend)
    with np.errstate(over="ignore"):
        single = total.astype(np.float32)
    inexact = ~(exact & (single.astype(np.float64) == total))
    if np.any(inexact):
        first = _first(inexact)
        raise ValueError(
            f"{int(np.count_nonzero(inexact))} value(s) of V + c are not exactly "
            f"representable in {dtype}, first {float(total[first])!r} at {first}"
        )
    return encode_bits(single, dtype)


def _min_exponent(dtype: str) -> int:
    """Smallest normal exponent of a subject dtype."""
    return 1 - _FORMAT[dtype][2]


def _float_not_below(value: Fraction) -> float:
    """The float nearest ``value``, moved up one step if it lies below it."""
    nearest = float(value)
    if Fraction(nearest) >= value:
        return nearest
    return math.nextafter(nearest, math.inf)


def _floor_log2(value: Fraction) -> int:
    """``floor(log2(value))`` of a positive rational, exactly."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    return exponent


@dataclass(frozen=True)
class ValueShiftBound:
    """Path constants of the value-shift tolerance, as exact rationals.

    ``row_sum`` bounds ``|sum_k P_k - 1|`` for the coefficients the kernel
    applies to the value tokens. It is built from ``cast``, the relative error
    of a probability cast in the dtype's normal range; ``subnormal_cast``, the
    absolute errors of casts below it, relative to the normalizer; and
    ``normalizer``, the error of the FP32 normalizer. One launch's accumulation
    error is at most ``accumulation`` times the largest ``|V|`` of the column
    plus ``accumulation_floor``.

    ``cast_normalizer_row_sum`` is ``row_sum`` without the two cast terms: the
    bound of a kernel whose normalizer summed the cast weights, so that the
    weights it applies sum to one up to FP32 rounding. No verdict uses it."""

    dtype: str
    keys: int
    key_tiles: int
    cast: Fraction
    subnormal_cast: Fraction
    normalizer: Fraction
    row_sum: Fraction
    cast_normalizer_row_sum: Fraction
    accumulation: Fraction
    accumulation_floor: Fraction

    def accumulation_bound(self, value_max: Fraction) -> Fraction:
        """Bound one launch's accumulation error in a column whose largest
        ``|V|`` is ``value_max``."""
        return self.accumulation * value_max + self.accumulation_floor


def value_shift_bound(dtype: str, *, keys: int, key_tiles: int) -> ValueShiftBound:
    """Evaluate the path constants for one single-pass unmasked launch over
    ``keys`` keys in ``key_tiles`` key tiles.

    Both launches of the relation read the same Q and K, so they compute the
    same weights ``p_k = exp2(s_k - m)``, the same rescales ``alpha`` and the
    same normalizer ``l``. Key ``k`` reaches the output with the coefficient
    ``P_k = cast(p_k) A_k / l``, ``A_k`` the product of the rescales after its
    tile, while ``l`` sums the uncast ``p_k A_k``. With ``D = sum p_k A_k``,
    ``u = 2**-23`` and ``c = 2**-(p+1)`` for a dtype with ``p`` fraction bits:

    - cast: ``sum cast(p_k) A_k = D (1 + sigma)``. A weight in the dtype's
      normal range moves by at most ``c`` of itself (round-to-nearest-even), a
      weighted average over the row; a weight below it by at most half the
      smallest subnormal, in absolute terms. ``D >= 1 / F`` because each row's
      largest score has weight ``exp2(0)``, and ``A_k <= F``, with
      ``F = 1 / (1 - n_T u) >= (1 + u)**n_T``;
    - normalizer: ``l = D (1 + theta)``, ``|theta| <= gamma(keys + n_T)`` for
      FP32 additions in any order and one rescale per tile, plus the absolute
      errors of rescale products below the FP32 normal range;
    - reciprocal: ``1 / l`` within one FP32 unit.

    So ``|sum P_k - 1| <= (1 + sigma)(1 + u) / (1 - theta) - 1``. The errors of
    the exponential and of its argument do not enter: numerator and normalizer
    use the same computed weights. The accumulator passes through
    ``keys / 8`` MFMA calls, each within eight FP32 units of the summed
    magnitudes of its accumulator and products, and ``n_T`` rescales:
    ``gamma(keys + n_T)`` of ``sum P_k |V_k|``, plus the absolute errors of
    subnormal results.

    Every constant is an exact rational and an upper bound of the derived
    expression. The derivation needs ``(keys + n_T) u < 1`` and a normalizer
    error below one, which keeps the computed normalizer positive; anything
    else is refused."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if keys < _ATOM_PRODUCTS or keys % _ATOM_PRODUCTS:
        raise ValueError(
            f"keys must be a positive multiple of {_ATOM_PRODUCTS}, the products "
            "of one MFMA call"
        )
    if not 1 <= key_tiles <= keys:
        raise ValueError("key_tiles must lie in [1, keys]")

    fraction_bits = _FORMAT[dtype][0]
    unit = _FP32_UNIT
    steps = (keys + key_tiles) * unit
    if steps >= 1:
        raise ValueError("too many keys for the FP32 accumulation bound")
    gamma = steps / (1 - steps)
    # Not below (1 + u)**n_T, the largest product of rescales, and not below
    # 1 / D for the exact normalizer D.
    rescaled = 1 / (1 - key_tiles * unit)

    cast = Fraction(1, 2 ** (fraction_bits + 1))
    subnormal_half = Fraction(2) ** (_min_exponent(dtype) - fraction_bits - 1)
    subnormal_cast = (keys - 1) * subnormal_half * rescaled**2
    normalizer = gamma + key_tiles * rescaled**2 * (1 + gamma) * _FP32_UNDERFLOW
    if normalizer >= 1:
        raise ValueError("the normalizer error bound left its valid range")
    row_sum = (1 + cast + subnormal_cast) * (1 + unit) / (1 - normalizer) - 1
    reciprocal = (1 + unit) * rescaled / (1 - normalizer)
    accumulator_underflow = (
        rescaled
        * (1 + gamma)
        * (keys // _ATOM_PRODUCTS * _ATOM_UNDERFLOW + key_tiles * _FP32_UNDERFLOW)
    )
    return ValueShiftBound(
        dtype=dtype,
        keys=keys,
        key_tiles=key_tiles,
        cast=cast,
        subnormal_cast=subnormal_cast,
        normalizer=normalizer,
        row_sum=row_sum,
        cast_normalizer_row_sum=(1 + unit) / (1 - normalizer) - 1,
        accumulation=gamma * (1 + row_sum),
        accumulation_floor=reciprocal * accumulator_underflow,
    )


def rounding_radius(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Per stored value, half the gap to its neighbour away from zero.

    A value rounded to nearest lies within this distance of what it was
    rounded from. It is half a unit in the last place of the value's binade; at
    a power of two that is the larger of the two adjacent half-gaps, and at
    zero and at subnormals half the smallest subnormal. Only the exponent field
    is read."""
    fraction_bits, exponent_bits, bias = _FORMAT[dtype]
    raw = np.asarray(bits, dtype=np.uint16).astype(np.int64)
    field = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
    exponent = np.maximum(field, 1) - bias
    return np.ldexp(1.0, exponent - fraction_bits - 1)


def store_allowance(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Bound ``|stored - o * rcp|`` for each stored output, from its own bits.

    The kernel stores ``cast(o * rcp)`` or ``cast(fl32(o * rcp))``, both rounded
    to nearest even with subnormals kept. The cast moves its operand by at most
    :func:`rounding_radius` of the result; the FP32 product, where there is
    one, by one FP32 unit of a magnitude that is at most the stored one plus
    that radius, or by the FP32 underflow error. The bound of the second form
    is used for both. A non-finite stored value has no allowance: the analyzer
    counts it invalid and its entry here is the allowance of zero."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    fraction_bits, exponent_bits, _ = _FORMAT[dtype]
    raw = np.asarray(bits, dtype=np.uint16)
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    finite = (raw & exponent_mask) != exponent_mask
    raw = np.where(finite, raw, np.uint16(0))
    radius = rounding_radius(raw, dtype)
    reach = np.abs(decode_bits(raw, dtype)) + radius
    unit = float(_FP32_UNIT)
    return radius + unit * reach / (1.0 - unit) + float(_FP32_UNDERFLOW)


def _store_allowance_within(magnitude: Fraction, dtype: str) -> Fraction:
    """:func:`store_allowance` of the largest value of at most ``magnitude``,
    exactly. The allowance does not decrease with the stored magnitude, so this
    bounds it for every stored value inside ``magnitude``."""
    fraction_bits = _FORMAT[dtype][0]
    exponent = _min_exponent(dtype)
    if magnitude > 0:
        exponent = max(_floor_log2(magnitude), exponent)
    radius = Fraction(2) ** (exponent - fraction_bits - 1)
    unit = _FP32_UNIT
    return radius + unit * (magnitude + radius) / (1 - unit) + _FP32_UNDERFLOW


def _magnitude_guard(reach: Fraction, dtype: str) -> Fraction:
    """Largest stored magnitude when ``|o * rcp| <= reach``: the FP32 product
    and the cast each move it up by at most their rounding."""
    fraction_bits = _FORMAT[dtype][0]
    product = reach * (1 + _FP32_UNIT) + _FP32_UNDERFLOW
    exponent = max(_floor_log2(product), _min_exponent(dtype))
    return product + Fraction(2) ** (exponent - fraction_bits - 1)


@dataclass(frozen=True, eq=False)
class ValueShiftContract:
    """Everything the verdict of one pair of launches needs besides the stored
    outputs. All arrays are per output column ``d`` and read-only.

    ``column_tolerance`` is the part of the tolerance that follows from the
    path and the launched values alone: ``|c_d|`` times the row-sum bound plus
    the accumulation bounds of both launches. The analyzer adds the store
    allowance of the two stored values. ``base_guard`` and ``follow_up_guard``
    are the largest magnitudes a launch can store, and ``apriori_tolerance`` is
    the tolerance of two stored values at those guards: no pair inside the
    guards has a larger one.

    ``cast_normalizer_tolerance`` is ``column_tolerance`` with the row-sum bound
    of a kernel whose normalizer sums the cast weights. The shipped kernel sums
    the uncast ones, which the verdict accepts; residuals are also reported
    against this tighter value, so that what the verdict accepts stays a
    measured number. It takes part in no verdict."""

    case: ValueShiftCase
    bound: ValueShiftBound
    shift: np.ndarray
    base_value_max: np.ndarray
    follow_up_value_max: np.ndarray
    column_tolerance: np.ndarray
    base_guard: np.ndarray
    follow_up_guard: np.ndarray
    apriori_tolerance: np.ndarray
    output_scale_floor: np.ndarray
    cast_normalizer_tolerance: np.ndarray


def _column_maxima(values: np.ndarray) -> np.ndarray:
    """Largest ``|V|`` of each column over every batch element, key and KV head:
    independent of how query heads are routed to KV heads."""
    return np.max(np.abs(values), axis=(0, 1, 2))


def value_shift_contract(
    case: ValueShiftCase,
    base_values: np.ndarray,
    follow_up_values: np.ndarray,
    *,
    key_tiles: int,
    applied: np.ndarray | None = None,
) -> ValueShiftContract:
    """Build the contract of one pair of launches from the raw V bits they read.

    ``key_tiles`` is the number of key tiles one query row visits in the kernel
    under test. ``applied`` is the shift the follow-up tensor was built with,
    the expected one unless the launch is a shift mutation; the analyzer
    expects :func:`shift_vector` either way.

    Fails closed. The follow-up bits must decode to ``V + applied`` exactly,
    because the tolerance has no input-rounding term, and V must be finite. A
    column whose tolerance is not below ``|c_d|`` cannot show a missing shift
    and is refused."""
    case.require_scope()
    base = _require_value_bits(base_values, "base_values")
    follow_up = _require_value_bits(follow_up_values, "follow_up_values")
    for name, raw in (("base_values", base), ("follow_up_values", follow_up)):
        if raw.shape != case.key_shape:
            raise ValueError(f"{name} shape {raw.shape} != {case.key_shape}")
    shift = shift_vector(case.head_dim)
    applied_values = (
        shift if applied is None else _require_shift(applied, case.head_dim)
    )

    source = decode_bits(base, case.dtype)
    target = decode_bits(follow_up, case.dtype)
    if not np.all(np.isfinite(source)):
        raise ValueError("cannot evaluate the tolerance: V is not finite")
    total, exact = _exact_sum(source, applied_values)
    with np.errstate(invalid="ignore"):
        wrong = ~(exact & np.isfinite(target) & (target == total))
    if np.any(wrong):
        first = _first(wrong)
        raise ValueError(
            f"the follow-up values are not V + c exactly at "
            f"{int(np.count_nonzero(wrong))} element(s), first at {first}; the "
            "tolerance has no input-rounding term"
        )

    bound = value_shift_bound(case.dtype, keys=case.seqlen_k, key_tiles=key_tiles)
    base_max = _column_maxima(source)
    follow_up_max = _column_maxima(target)
    columns = range(case.head_dim)
    column_tolerance = np.empty(case.head_dim, dtype=np.float64)
    base_guard = np.empty(case.head_dim, dtype=np.float64)
    follow_up_guard = np.empty(case.head_dim, dtype=np.float64)
    apriori = np.empty(case.head_dim, dtype=np.float64)
    floor = np.empty(case.head_dim, dtype=np.float64)
    cast_normalizer = np.empty(case.head_dim, dtype=np.float64)
    for dim in columns:
        expected = abs(Fraction(float(shift[dim])))
        accumulation = []
        guards = []
        stores = []
        for value_max in (base_max[dim], follow_up_max[dim]):
            largest = Fraction(float(value_max))
            error = bound.accumulation_bound(largest)
            guard = _magnitude_guard((1 + bound.row_sum) * largest + error, case.dtype)
            accumulation.append(error)
            guards.append(guard)
            stores.append(_store_allowance_within(guard, case.dtype))
        column = expected * bound.row_sum + sum(accumulation)
        host = Fraction(_HOST_UNIT) * (sum(guards) + expected)
        # Not below the tolerance of any stored pair inside the guards, with
        # room for the FP64 rounding of both evaluations.
        widest = (column + sum(stores) + host) * Fraction(1 + 2.0**-46)
        if widest >= expected:
            raise ValueError(
                f"the tolerance of column {dim} ({float(widest):.6g}) is not below "
                f"the shift it must resolve ({float(expected):.6g})"
            )
        column_tolerance[dim] = _float_not_below(column)
        base_guard[dim] = _float_not_below(guards[0])
        follow_up_guard[dim] = _float_not_below(guards[1])
        apriori[dim] = _float_not_below(widest)
        floor[dim] = _float_not_below((2 * widest + sum(stores)) / (expected - widest))
        cast_normalizer[dim] = _float_not_below(
            column - expected * (bound.row_sum - bound.cast_normalizer_row_sum)
        )
    arrays = (
        shift,
        base_max,
        follow_up_max,
        column_tolerance,
        base_guard,
        follow_up_guard,
        apriori,
        floor,
        cast_normalizer,
    )
    for array in arrays:
        array.setflags(write=False)
    return ValueShiftContract(case, bound, *arrays)


def value_shift_tolerance(
    contract: ValueShiftContract,
    base: np.ndarray,
    follow_up: np.ndarray,
    *,
    cast_normalizer: bool = False,
) -> np.ndarray:
    """The tolerance of every element of a pair of raw-bit outputs:
    the contract's column part, the store allowance of each of the two stored
    values, and the FP64 rounding of the residual itself, rounded upward.

    ``cast_normalizer`` selects the diagnostic column part of a kernel whose
    normalizer sums the cast weights; the verdict uses the default."""
    dtype = contract.case.dtype
    fraction_bits, exponent_bits, _ = _FORMAT[dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    magnitudes = np.zeros(base.shape, dtype=np.float64)
    for raw in (base, follow_up):
        finite = (raw & exponent_mask) != exponent_mask
        magnitudes += np.abs(np.where(finite, decode_bits(raw, dtype), 0.0))
    host = _HOST_UNIT * (magnitudes + np.abs(contract.shift))
    column = (
        contract.cast_normalizer_tolerance
        if cast_normalizer
        else contract.column_tolerance
    )
    return (
        column + store_allowance(follow_up, dtype) + store_allowance(base, dtype) + host
    ) * _HOST_MARGIN


@dataclass(frozen=True)
class ValueShiftResult:
    """Verdict and diagnostics of one pair of raw-bit outputs.

    ``violations`` is the verdict. Everything else is a diagnostic and never
    enters it. ``column_exceed_count`` splits ``exceed_count`` per output
    column; :func:`shift_mutation_failures` reads it to judge a wrong-shift
    launch. ``max_ratio`` is the largest residual over its tolerance, at
    ``worst_coordinate``; ``head_max_ratio`` and ``class_max_ratio`` split it
    per ``(batch, query head)`` and per ``d mod 4``, the period of the shift
    and of the FP16 store forms. ``eta_at_worst`` is the residual over the
    shift there, the apparent row-sum error, and ``eta_allowance_at_worst`` its
    tolerance in the same units. ``max_row_eta`` is the largest per-row median
    of that quotient and ``max_row_spread`` the largest distance of a column
    from its row's median: a row-sum defect moves the median, a column defect
    the spread. ``apriori_max_ratio`` is the largest residual over the
    tolerance that does not read the outputs, and ``tolerance_fraction`` the
    mean of the applied tolerance over that one.
    ``cast_normalizer_max_ratio`` and ``cast_normalizer_exceed_count`` judge the
    same residuals against the tolerance of a kernel whose normalizer sums the
    cast weights: what the shipped kernel's uncast normalizer costs, measured."""

    violations: tuple[str, ...]
    invalid_count: int
    magnitude_count: int
    exceed_count: int
    column_exceed_count: tuple[int, ...]
    compared_count: int
    max_ratio: float
    worst_coordinate: tuple[int, int, int, int] | None
    residual_at_worst: float
    tolerance_at_worst: float
    max_abs: float
    rms: float
    mean_signed: float
    head_max_ratio: tuple[tuple[float, ...], ...]
    class_max_ratio: tuple[float, ...]
    class_max_abs: tuple[float, ...]
    eta_at_worst: float
    eta_allowance_at_worst: float
    max_row_eta: float
    max_row_spread: float
    apriori_max_ratio: float
    tolerance_fraction: float
    cast_normalizer_max_ratio: float
    cast_normalizer_exceed_count: int

    @property
    def rejected_by_residual(self) -> bool:
        """Whether some residual exceeds its tolerance."""
        return self.exceed_count > 0


def analyze_value_shift(
    contract: ValueShiftContract, base: np.ndarray, follow_up: np.ndarray
) -> ValueShiftResult:
    """Judge ``E = O(V + c) - O(V) - c`` on two raw-bit outputs.

    Violations: a non-finite stored value in either output (an unwritten
    coordinate keeps its NaN prefill), a stored magnitude beyond the launch's
    guard (an error both launches share would otherwise cancel), or a residual
    beyond its tolerance."""
    case = contract.case
    outputs = []
    for name, values in (("base", base), ("follow-up", follow_up)):
        raw = np.asarray(values)
        if raw.dtype != np.uint16:
            raise ValueError(f"{name} output must contain raw uint16 bits")
        if raw.shape != case.output_shape:
            raise ValueError(f"{name} output shape {raw.shape} != {case.output_shape}")
        outputs.append(raw)
    base_raw, follow_raw = outputs

    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    base_finite = (base_raw & exponent_mask) != exponent_mask
    follow_finite = (follow_raw & exponent_mask) != exponent_mask
    invalid_count = sum(
        int(np.count_nonzero(~finite)) for finite in (base_finite, follow_finite)
    )
    base_values = np.where(base_finite, decode_bits(base_raw, case.dtype), 0.0)
    follow_values = np.where(follow_finite, decode_bits(follow_raw, case.dtype), 0.0)
    magnitude_count = int(
        np.count_nonzero(np.abs(base_values) > contract.base_guard)
    ) + int(np.count_nonzero(np.abs(follow_values) > contract.follow_up_guard))

    compared = base_finite & follow_finite
    compared_count = int(np.count_nonzero(compared))
    residual = np.where(compared, (follow_values - base_values) - contract.shift, 0.0)
    magnitude = np.abs(residual)
    tolerance = value_shift_tolerance(contract, base_raw, follow_raw)
    ratio = magnitude / tolerance
    exceed = compared & (magnitude > tolerance)
    exceed_count = int(np.count_nonzero(exceed))
    reduced = value_shift_tolerance(
        contract, base_raw, follow_raw, cast_normalizer=True
    )

    worst_coordinate: tuple[int, int, int, int] | None = None
    residual_at_worst = tolerance_at_worst = 0.0
    eta_at_worst = eta_allowance_at_worst = 0.0
    if compared_count:
        index = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        worst_coordinate = tuple(int(item) for item in index)
        expected = float(contract.shift[index[3]])
        residual_at_worst = float(residual[index])
        tolerance_at_worst = float(tolerance[index])
        eta_at_worst = residual_at_worst / expected
        eta_allowance_at_worst = tolerance_at_worst / abs(expected)

    # Per row: the median over the columns of the residual over the shift, and
    # the largest distance of a column from it. Rows with an invalid value are
    # left out; they are violations already.
    eta = residual / contract.shift
    row_median = np.median(eta, axis=3)
    row_spread = np.max(np.abs(eta - row_median[..., None]), axis=3)
    complete = np.all(compared, axis=3)
    max_row_eta = float(np.max(np.abs(row_median), where=complete, initial=0.0))
    max_row_spread = float(np.max(row_spread, where=complete, initial=0.0))

    period = len(_SHIFT_PERIOD)
    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if magnitude_count:
        violations.append(f"magnitude>guard at {magnitude_count}")
    if exceed_count:
        violations.append(
            f"residual={abs(residual_at_worst):.9g}>{tolerance_at_worst:.9g} "
            f"at {exceed_count}"
        )
    if compared_count == 0:
        violations.append("no_coordinate_compared")

    return ValueShiftResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        magnitude_count=magnitude_count,
        exceed_count=exceed_count,
        column_exceed_count=tuple(
            int(item) for item in np.count_nonzero(exceed, axis=(0, 1, 2))
        ),
        compared_count=compared_count,
        max_ratio=float(np.max(ratio)),
        worst_coordinate=worst_coordinate,
        residual_at_worst=residual_at_worst,
        tolerance_at_worst=tolerance_at_worst,
        max_abs=float(np.max(magnitude)),
        rms=(
            math.sqrt(float(np.square(residual).sum()) / compared_count)
            if compared_count
            else 0.0
        ),
        mean_signed=float(residual.sum()) / compared_count if compared_count else 0.0,
        head_max_ratio=tuple(
            tuple(float(item) for item in row) for row in np.max(ratio, axis=(1, 3))
        ),
        class_max_ratio=tuple(
            float(np.max(ratio[..., residue::period], initial=0.0))
            for residue in range(period)
        ),
        class_max_abs=tuple(
            float(np.max(magnitude[..., residue::period], initial=0.0))
            for residue in range(period)
        ),
        eta_at_worst=eta_at_worst,
        eta_allowance_at_worst=eta_allowance_at_worst,
        max_row_eta=max_row_eta,
        max_row_spread=max_row_spread,
        apriori_max_ratio=float(np.max(magnitude / contract.apriori_tolerance)),
        tolerance_fraction=(
            float(np.sum(tolerance / contract.apriori_tolerance, where=compared))
            / compared_count
            if compared_count
            else 0.0
        ),
        cast_normalizer_max_ratio=float(np.max(magnitude / reduced)),
        cast_normalizer_exceed_count=int(
            np.count_nonzero(compared & (magnitude > reduced))
        ),
    )


def shift_mutation_failures(
    contract: ValueShiftContract, applied: np.ndarray, result: ValueShiftResult
) -> tuple[str, ...]:
    """Why a launch on ``V + applied`` does not count as a rejected wrong shift.
    Empty when it does.

    ``contract`` and ``result`` are those of the base launch and the launch on
    ``V + applied``, judged against the expected shift ``c``. The mutated
    tensor is a legitimate input, so the pair obeys the relation with the
    applied shift, and its residual against ``c`` is ``applied_d - c_d`` plus
    an error inside the tolerance. That fixes where the residuals must lie:

    - every stored value is finite and inside its guard, as for any launch; a
      faulty launch must not stand in for a rejection;
    - a column the mutation leaves at ``c_d`` has no residual beyond its
      tolerance;
    - a column whose applied shift differs from ``c_d`` by more than twice the
      a-priori tolerance, which no stored pair inside the guards exceeds, has
      every residual beyond its tolerance;
    - some residual is beyond its tolerance.

    The tolerance is evaluated for ``|c_d|``, so it covers an applied shift
    that is not larger; a larger one is refused."""
    case = contract.case
    applied_values = _require_shift(applied, case.head_dim)
    if np.any(np.abs(applied_values) > np.abs(contract.shift)):
        raise ValueError(
            "the applied shift exceeds the expected one; the tolerance does "
            "not cover the pair"
        )
    difference = np.abs(applied_values - contract.shift)
    failures = []
    if result.invalid_count:
        failures.append(f"invalid={result.invalid_count}")
    if result.magnitude_count:
        failures.append(f"magnitude>guard at {result.magnitude_count}")
    exceed = np.asarray(result.column_exceed_count)
    unchanged = difference == 0.0
    stray = int(exceed[unchanged].sum())
    if stray:
        first = int(np.flatnonzero(unchanged & (exceed > 0))[0])
        failures.append(
            f"{stray} residuals beyond the tolerance in columns that carry the "
            f"expected shift, first in column {first}"
        )
    rows = case.batch * case.seqlen_q * case.nhead_q
    certain = difference > 2.0 * contract.apriori_tolerance
    short = certain & (exceed < rows)
    if np.any(short):
        first = int(np.flatnonzero(short)[0])
        failures.append(
            f"{int(np.count_nonzero(short))} columns whose wrong shift is certain "
            f"to exceed the tolerance keep residuals inside it, first column "
            f"{first} with {int(exceed[first])} of {rows} beyond"
        )
    if not result.exceed_count:
        failures.append("no residual beyond its tolerance")
    return tuple(failures)


def scale_output_bits(bits: np.ndarray, eta: float, dtype: str) -> np.ndarray:
    """Raw bits of ``cast((1 + eta) O)`` for stored output bits ``O``: the
    normalization mutation, applied after the kernel's own conversion.

    ``eta`` is a power of two, so the product is exact in FP64 and, for the
    lane's scales, in FP32: the result is one round-to-nearest-even of the exact
    product. A product that FP32 cannot hold is refused. Non-finite stored
    values are kept as they are."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if not 0.0 < eta <= 1.0 or math.frexp(eta)[0] != 0.5:
        raise ValueError("eta must be a power of two in (0, 1]")
    raw = np.asarray(bits)
    if raw.dtype != np.uint16:
        raise ValueError("the output must contain raw uint16 bits")
    fraction_bits, exponent_bits, _ = _FORMAT[dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    finite = (raw & exponent_mask) != exponent_mask
    values = np.where(finite, decode_bits(raw, dtype), 0.0)
    scaled, exact = _exact_sum(values, values * eta)
    exact &= (values * eta) / eta == values
    single = scaled.astype(np.float32)
    if not np.all(exact & (single.astype(np.float64) == scaled)):
        raise ValueError("the scaled output is not exact before its conversion")
    if dtype == "fp16":
        with np.errstate(over="ignore"):
            rounded = single.astype(np.float16).view(np.uint16)
    else:
        rounded = f32_to_bf16(single).reshape(single.shape)
    return np.ascontiguousarray(np.where(finite, rounded, raw).astype(np.uint16))


def output_scale_threshold(contract: ValueShiftContract) -> float:
    """Smallest relative output scale every pair of passing outputs is certain
    to fail under :func:`scale_output_bits`, over the columns.

    Scaling both stored outputs by ``1 + eta`` moves the residual by
    ``eta c_d`` and leaves ``(1 + eta)`` times the unscaled residual plus the
    two conversions of the mutation. Either a scaled value leaves its guard, or
    every tolerance involved is at most the a-priori one ``t_d`` and each
    conversion at most the store allowance ``s_d`` at its guard: the mutation
    is then rejected in column ``d`` once
    ``eta > (2 t_d + s_d + s_d') / (|c_d| - t_d)``."""
    return float(np.min(contract.output_scale_floor)) * (1.0 + 2.0**-20)


def format_value_shift_result(result: ValueShiftResult) -> str:
    """Render the verdict and the diagnostics of one pair of outputs."""
    coordinate = (
        "n/a"
        if result.worst_coordinate is None
        else ":".join(str(value) for value in result.worst_coordinate)
    )
    violations = ", ".join(result.violations) if result.violations else "none"
    classes = " ".join(f"{value:.3g}" for value in result.class_max_ratio)
    return (
        f"violations={violations}; rho_max={result.max_ratio:.3g} at "
        f"b:q:h:d={coordinate} E={result.residual_at_worst:.9g} "
        f"tau={result.tolerance_at_worst:.9g}; eta_hat={result.eta_at_worst:.3g} "
        f"(allowance {result.eta_allowance_at_worst:.3g}); "
        f"eta_row={result.max_row_eta:.3g} s_row={result.max_row_spread:.3g}; "
        f"rms={result.rms:.9g} mean={result.mean_signed:.9g}; "
        f"rho_max by d mod 4: {classes}; apriori rho_max="
        f"{result.apriori_max_ratio:.3g}, tau/apriori={result.tolerance_fraction:.3g}; "
        f"against a normalizer over cast weights: rho_max="
        f"{result.cast_normalizer_max_ratio:.3g}, "
        f"{result.cast_normalizer_exceed_count} beyond"
    )


def format_head_diagnostics(result: ValueShiftResult) -> str:
    """Render the largest residual over its tolerance per ``(batch, query head)``."""
    return "\n".join(
        f"  b{batch}: "
        + " ".join(f"h{head}={value:.3g}" for head, value in enumerate(row))
        for batch, row in enumerate(result.head_max_ratio)
    )
