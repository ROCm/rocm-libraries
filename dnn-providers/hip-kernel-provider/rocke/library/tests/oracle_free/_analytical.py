# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free closed forms and raw-bit policy checks for the uniform-probability probe.

There is intentionally no general attention implementation in this module.
Zero Q forces uniform legal logits, so every expectation follows directly from
legal-key membership and exact rational rounding into the subject dtype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import MappingProxyType

import numpy as np
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

_DTYPES = frozenset({"fp16", "bf16"})
_MASKS = frozenset({"causal", "window"})
_FORMAT = {
    "fp16": (10, 5, 15),
    "bf16": (7, 8, 127),
}
UNIFORM_FOCUSED_ROWS = (0, 62, 63, 64, 127, 128, 255)
# KV head h scales its V profile by 2**h, so heads are distinguishable: a kernel
# that reads the wrong V head, uses a wrong head stride, or writes one output head
# everywhere moves every legal coordinate by at least a factor of two. A power of
# two is exact in both dtypes and commutes with every rounding the kernel performs
# (the P cast, FP32 accumulation, normalization, the output cast), so each head's
# correct output is the head-0 output shifted in exponent and the zero-ULP
# argument is unchanged. The cap keeps outputs well inside the FP16 range.
_MAX_KV_HEADS = 8


class UniformMutation(str, Enum):
    """Analyzer-side legal-set and denominator mutations."""

    INCLUDE_FUTURE_KEY = "include_future_key"
    EXCLUDE_LAST_KEY = "exclude_last_key"
    INCLUDE_OLD_WINDOW_KEY = "include_old_window_key"
    EXCLUDE_OLDEST_WINDOW_KEY = "exclude_oldest_window_key"
    DENOMINATOR_PLUS_ONE = "denominator_plus_one"


@dataclass(frozen=True)
class UniformCase:
    """Logical dense-attention shape and semantics for one uniform-probability case."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str
    mask: str
    window: int = 0

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "head_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nhead_q % self.nhead_k:
            raise ValueError("nhead_q must be a multiple of nhead_k")
        if self.nhead_k > _MAX_KV_HEADS:
            raise ValueError(
                f"nhead_k must be at most {_MAX_KV_HEADS} for distinct head scales"
            )
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if self.mask not in _MASKS:
            raise ValueError(f"mask must be one of {sorted(_MASKS)}")
        if self.mask == "window":
            if self.window <= 0:
                raise ValueError("window mask requires window > 0")
        elif self.window:
            raise ValueError("window must be zero unless mask='window'")

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sq, Hq, D]`` output shape."""
        return (self.batch, self.seqlen_q, self.nhead_q, self.head_dim)

    @property
    def key_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sk, Hkv, D]`` K/V shape."""
        return (self.batch, self.seqlen_k, self.nhead_k, self.head_dim)

    @property
    def head_scale_exponents(self) -> tuple[int, ...]:
        """Per query head, the exponent of its KV head's V scale ``2**h_kv``.

        Query head ``h_q`` reads KV head ``h_q // (nhead_q // nhead_k)``."""
        group = self.nhead_q // self.nhead_k
        return tuple(head // group for head in range(self.nhead_q))

    @property
    def case_id(self) -> str:
        """Return a stable identifier for test ids and resolved-subject checks."""
        mask = f"window{self.window}" if self.mask == "window" else self.mask
        return (
            f"{self.dtype}-{mask}-b{self.batch}-sq{self.seqlen_q}-"
            f"sk{self.seqlen_k}-hq{self.nhead_q}-hk{self.nhead_k}-d{self.head_dim}"
        )

    def require_scope(self) -> None:
        """Refuse a case outside the zero-ULP theorem's range."""
        require_proven_scope(self)


# GQA (four query heads per KV head), so a kernel that drops the group divisor
# reads the wrong V scale and fails. The dtype/mask pairing complements the
# pinned SDPA cohort, which already runs FP16 causal on gfx942.
UNIFORM_WINDOW_FP16 = UniformCase(1, 256, 256, 8, 2, 64, "fp16", "window", 64)
UNIFORM_CAUSAL_BF16 = UniformCase(1, 256, 256, 8, 2, 64, "bf16", "causal")
# The GPU lane's cases. A CPU contract pins the lane's parametrization to them, so
# a case cannot silently stop executing.
UNIFORM_QUICK_CASES = (UNIFORM_WINDOW_FP16, UNIFORM_CAUSAL_BF16)


def _coerce_mutation(mutation: UniformMutation | str | None) -> UniformMutation | None:
    if mutation is None:
        return None
    try:
        return (
            mutation
            if isinstance(mutation, UniformMutation)
            else UniformMutation(mutation)
        )
    except ValueError as error:
        raise ValueError(f"unknown mutation {mutation!r}") from error


def _validate_mutation(case: UniformCase, mutation: UniformMutation | None) -> None:
    if (
        mutation
        in {
            UniformMutation.INCLUDE_FUTURE_KEY,
            UniformMutation.EXCLUDE_LAST_KEY,
        }
        and case.mask != "causal"
    ):
        raise ValueError(f"mutation {mutation.value} requires a causal case")
    if (
        mutation
        in {
            UniformMutation.INCLUDE_OLD_WINDOW_KEY,
            UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
        }
        and case.mask != "window"
    ):
        raise ValueError(f"mutation {mutation.value} requires a window case")


def legal_interval(
    case: UniformCase,
    query: int,
    mutation: UniformMutation | str | None = None,
) -> tuple[int, int]:
    """Return inclusive analyzer legal bounds, or ``(0, -1)`` when empty."""
    if not 0 <= query < case.seqlen_q:
        raise IndexError(query)
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)
    upper_shift, window = _interval_rule(case, selected)
    upper = min(query + upper_shift, case.seqlen_k - 1)
    lower = max(0, query - window + 1) if case.mask == "window" else 0
    return (lower, upper) if lower <= upper else (0, -1)


def _interval_rule(
    case: UniformCase, selected: UniformMutation | None
) -> tuple[int, int]:
    """Return the (possibly mutated) rule's upper-bound shift and window."""
    upper_shift = 0
    if selected == UniformMutation.INCLUDE_FUTURE_KEY:
        upper_shift = 1
    elif selected == UniformMutation.EXCLUDE_LAST_KEY:
        upper_shift = -1
    window = case.window
    if selected == UniformMutation.INCLUDE_OLD_WINDOW_KEY:
        window += 1
    elif selected == UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY:
        window -= 1
    return upper_shift, window


def _legal_bounds(
    case: UniformCase, selected: UniformMutation | None
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`legal_interval` for every query at once, as int64 ``first``/``last``."""
    upper_shift, window = _interval_rule(case, selected)
    query = np.arange(case.seqlen_q, dtype=np.int64)
    upper = np.minimum(query + upper_shift, case.seqlen_k - 1)
    if case.mask == "window":
        lower = np.maximum(query - window + 1, 0)
    else:
        lower = np.zeros_like(query)
    empty = lower > upper
    return np.where(empty, 0, lower), np.where(empty, -1, upper)


def _legal_counts(
    case: UniformCase, selected: UniformMutation | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-query ``first``, ``last`` and ``N_q``.

    The correct rule never leaves a row without a legal key; a mutated one may."""
    first, last = _legal_bounds(case, selected)
    legal_count = np.maximum(last - first + 1, 0)
    if selected is None and np.any(legal_count == 0):
        query = int(np.argmax(legal_count == 0))
        raise ValueError(f"correct legal-key rule has an empty row at q={query}")
    return first, last, legal_count


def legal_keys(
    case: UniformCase,
    query: int,
    mutation: UniformMutation | str | None = None,
) -> tuple[int, ...]:
    """Enumerate the independently derived legal keys for one query."""
    lower, upper = legal_interval(case, query, mutation)
    return tuple(range(lower, upper + 1))


def max_legal_count(case: UniformCase) -> int:
    """Return the largest ``N_q = |A(q)|`` of the correct legal-key rule."""
    _, _, legal_count = _legal_counts(case, None)
    return int(np.max(legal_count))


def require_proven_scope(case: UniformCase) -> None:
    """Refuse a case whose legal counts leave the zero-ULP theorem's range.

    The theorem holds for every single-pass dense variant, whatever its tile,
    grid, scheduling or exp2 lowering: with zero Q, ``exp2(0) = 1`` exactly, the
    running-max rescale is 0 or 1, ``l = N_q`` and the PV sums are exact in FP32,
    and the stored ``acc * rcp(l)`` rounds (RNE) to the correctly rounded target
    for ``N_q < 2**(21-p)``, ``p`` the dtype's fraction bits (two thirds of that
    for the folded exposure, whose outputs sum several probabilities). Beyond the
    range the expected bits are still exact, but zero ULP is no longer proven, so
    a mismatch would not show a kernel defect."""
    limit = _proven_limit(case.dtype, "folded")
    largest = max_legal_count(case)
    if largest >= limit:
        raise ValueError(
            f"case {case.case_id} reaches N_q={largest}, outside the proven "
            f"zero-ULP range N_q < {limit} ({case.dtype}, folded exposure)"
        )


def residue_count(first: int, last: int, residue: int, period: int) -> int:
    """Count integers in ``[first,last]`` having one nonnegative residue."""
    if period <= 0:
        raise ValueError("period must be positive")
    if not 0 <= residue < period:
        raise ValueError("residue must lie in [0, period)")
    if last < first:
        return 0
    return (last - residue) // period - (first - 1 - residue) // period


def _round_ratio_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + int(
        twice > denominator or (twice == denominator and quotient & 1)
    )


@cache
def target_bits(numerator: int, denominator: int, dtype: str) -> int:
    """Round one nonnegative exact rational to FP16 or BF16 with RNE."""
    if dtype not in _FORMAT:
        raise ValueError(dtype)
    if numerator < 0 or denominator <= 0:
        raise ValueError("target ratio requires numerator >= 0 and denominator > 0")
    if numerator == 0:
        return 0

    fraction_bits, exponent_bits, bias = _FORMAT[dtype]
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    min_exponent = 1 - bias
    max_exponent = (1 << exponent_bits) - 2 - bias
    if exponent < min_exponent:
        shift = fraction_bits - min_exponent
        rounded = _round_ratio_to_even(numerator << shift, denominator)
        if rounded >= 1 << fraction_bits:
            return 1 << fraction_bits
        return rounded

    shift = exponent - fraction_bits
    if shift >= 0:
        rounded = _round_ratio_to_even(numerator, denominator << shift)
    else:
        rounded = _round_ratio_to_even(numerator << -shift, denominator)
    if rounded == 1 << (fraction_bits + 1):
        rounded >>= 1
        exponent += 1
    if exponent > max_exponent:
        return ((1 << exponent_bits) - 1) << fraction_bits
    encoded_exponent = exponent + bias
    return (encoded_exponent << fraction_bits) | (rounded - (1 << fraction_bits))


def ordered_bits(bits: int) -> int:
    """Map binary16 bits to a monotone integer, coalescing signed zero."""
    raw = int(bits) & 0xFFFF
    if raw & 0x7FFF == 0:
        return 0x8000
    if raw & 0x8000:
        return 0x8000 - (raw & 0x7FFF)
    return 0x8000 + raw


def ulp_distance(a_bits: int, b_bits: int) -> int:
    """Return integer target-format ULP distance between two raw values."""
    return abs(ordered_bits(a_bits) - ordered_bits(b_bits))


def _ordered_array(bits: np.ndarray) -> np.ndarray:
    raw = np.asarray(bits, dtype=np.uint16).astype(np.uint32)
    magnitude = raw & np.uint32(0x7FFF)
    return np.where(
        magnitude == 0,
        np.uint32(0x8000),
        np.where(
            raw & np.uint32(0x8000),
            np.uint32(0x8000) - magnitude,
            np.uint32(0x8000) + raw,
        ),
    )


def decode_bits(bits: np.ndarray | int, dtype: str) -> np.ndarray:
    """Decode raw FP16/BF16 bits to FP64 without losing signed-zero input bits."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    raw = np.asarray(bits, dtype=np.uint16)
    if dtype == "fp16":
        return raw.copy().view(np.float16).astype(np.float64)
    # The shared codec returns at least 1-D; keep a scalar input scalar.
    return bf16_to_f32(raw).reshape(raw.shape).astype(np.float64)


def encode_bits(values: np.ndarray, dtype: str) -> np.ndarray:
    """Encode FP32 values as raw FP16/BF16 bits, refusing any value that would round.

    Exactness keeps the subject's inputs independent of conversion rounding: the
    expectations assume the kernel reads exactly these values. Each decoded value
    must reproduce its FP32 source bit-for-bit, so signed zero survives and NaN or
    out-of-range inputs are rejected."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    source = np.ascontiguousarray(values, dtype=np.float32)
    # NaN and infinity survive the bit round trip below, so reject them first.
    nonfinite = ~np.isfinite(source)
    if np.any(nonfinite):
        first = tuple(int(index) for index in np.argwhere(nonfinite)[0])
        raise ValueError(
            f"{int(np.count_nonzero(nonfinite))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    if dtype == "fp16":
        with np.errstate(over="ignore"):
            bits = source.astype(np.float16).view(np.uint16)
    else:
        # Round-to-nearest-even, but an accepted value never rounds: the
        # round-trip check below rejects anything that is not exact.
        bits = f32_to_bf16(source).reshape(source.shape)
    round_trip = decode_bits(bits, dtype).astype(np.float32)
    inexact = round_trip.view(np.uint32) != source.view(np.uint32)
    if np.any(inexact):
        first = tuple(int(index) for index in np.argwhere(inexact)[0])
        raise ValueError(
            f"{int(np.count_nonzero(inexact))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    return np.ascontiguousarray(bits)


def _deterministic_k(case: UniformCase) -> np.ndarray:
    """Construct finite, nonconstant, exactly representable K values."""
    batch = np.arange(case.batch, dtype=np.int64)[:, None, None, None]
    key = np.arange(case.seqlen_k, dtype=np.int64)[None, :, None, None]
    head = np.arange(case.nhead_k, dtype=np.int64)[None, None, :, None]
    dim = np.arange(case.head_dim, dtype=np.int64)[None, None, None, :]
    values = ((11 * batch + 3 * key + 5 * head + 7 * dim) % 17) - 8
    return (values.astype(np.float32) / 8.0).copy()


def _kv_head_scales(case: UniformCase) -> np.ndarray:
    """Return the per-KV-head V scales ``2**h``."""
    return np.ldexp(np.float32(1.0), np.arange(case.nhead_k)).astype(np.float32)


def uniform_basis_offsets(case: UniformCase) -> tuple[int, ...]:
    """Return the basis offsets ``0, D, 2D, ...`` that expose every key exactly once."""
    return tuple(range(0, case.seqlen_k, case.head_dim))


def uniform_inputs(
    case: UniformCase, basis_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build zero-Q, nonconstant-K, shifted-identity-V inputs, V scaled per head."""
    if basis_offset < 0:
        raise ValueError("basis_offset must be non-negative")
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = _deterministic_k(case)
    v = np.zeros(case.key_shape, dtype=np.float32)
    scales = _kv_head_scales(case)
    for dim in range(case.head_dim):
        key = basis_offset + dim
        if key < case.seqlen_k:
            v[:, key, :, dim] = scales
    return q, k, v


def uniform_inputs_folded(
    case: UniformCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build folded-basis V, where key ``k`` maps to dimension ``k mod D``,
    scaled per head like the basis profiles."""
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = _deterministic_k(case)
    v = np.zeros(case.key_shape, dtype=np.float32)
    scales = _kv_head_scales(case)
    for key in range(case.seqlen_k):
        v[:, key, :, key % case.head_dim] = scales
    return q, k, v


@dataclass(frozen=True, eq=False)
class UniformObservation:
    """Named raw-bit subject outputs from basis and folded exposures."""

    exposures: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        if not self.exposures:
            raise ValueError("an observation requires at least one exposure")
        normalized: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for name, values in self.exposures:
            if name in names:
                raise ValueError(f"duplicate exposure {name!r}")
            if name != "folded":
                _basis_offset(name)
            raw = np.asarray(values)
            if raw.dtype != np.uint16:
                raise ValueError(f"exposure {name!r} must contain raw uint16 bits")
            copy = np.ascontiguousarray(raw).copy()
            copy.setflags(write=False)
            normalized.append((name, copy))
            names.add(name)
        object.__setattr__(self, "exposures", tuple(normalized))

    @classmethod
    def from_mapping(cls, exposures: Mapping[str, np.ndarray]) -> UniformObservation:
        """Copy a mapping into one immutable observation."""
        return cls(tuple(exposures.items()))

    @property
    def by_name(self) -> Mapping[str, np.ndarray]:
        """Return a read-only name-to-bits view."""
        return MappingProxyType(dict(self.exposures))


def _basis_offset(name: str) -> int:
    prefix = "basis@"
    if not name.startswith(prefix):
        raise ValueError(f"unknown exposure {name!r}")
    text = name[len(prefix) :]
    try:
        offset = int(text)
    except ValueError as error:
        raise ValueError(f"invalid basis exposure {name!r}") from error
    if offset < 0 or str(offset) != text:
        raise ValueError(f"invalid basis exposure {name!r}")
    return offset


def _basis_offsets(observation: UniformObservation) -> tuple[int, ...]:
    return tuple(
        sorted(
            _basis_offset(name) for name, _ in observation.exposures if name != "folded"
        )
    )


def _validate_observation(
    case: UniformCase, observation: UniformObservation, *, require_complete: bool
) -> None:
    for name, values in observation.exposures:
        if values.shape != case.output_shape:
            raise ValueError(
                f"exposure {name!r} shape {values.shape} != {case.output_shape}"
            )
    if not require_complete:
        return
    names = observation.by_name
    if "folded" not in names:
        raise ValueError("a gated observation requires a folded exposure")
    offsets = _basis_offsets(observation)
    exposed = [
        offset + dim
        for offset in offsets
        for dim in range(case.head_dim)
        if offset + dim < case.seqlen_k
    ]
    if exposed != list(range(case.seqlen_k)):
        raise ValueError("basis exposures must expose every key exactly once")


def _exposure_representation(
    case: UniformCase, exposure: str, first: np.ndarray, last: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``has_key[D]`` and ``represented[Sq, D]`` for one exposure.

    ``has_key[d]`` says whether any key carries V in dimension ``d``;
    ``represented[q, d]`` counts the legal keys of query ``q`` that do (0 or 1
    for a basis exposure, up to ``ceil(Sk / D)`` for the folded one)."""
    dim = np.arange(case.head_dim, dtype=np.int64)
    if exposure == "folded":
        has_key = dim < case.seqlen_k
        # residue_count over [first, last] for every (query, residue) pair; an
        # empty row (first=0, last=-1) counts zero.
        represented = (last[:, None] - dim[None, :]) // case.head_dim - (
            first[:, None] - 1 - dim[None, :]
        ) // case.head_dim
    else:
        key = _basis_offset(exposure) + dim
        has_key = key < case.seqlen_k
        represented = (
            (first[:, None] <= key[None, :]) & (key[None, :] <= last[:, None])
        ).astype(np.int64)
    return has_key, np.where(has_key[None, :], represented, 0)


def _target_table(
    case: UniformCase, represented: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """Correctly rounded ``represented * 2**h_kv / N_q`` bits as ``[Sq, Hq, D]``.

    Zero wherever nothing is represented. Each distinct ``(represented, N)``
    pair is rounded once, through the exact integer :func:`target_bits`."""
    table = np.zeros((case.seqlen_q, case.nhead_q, case.head_dim), dtype=np.uint16)
    rows, dims = np.nonzero(represented)
    if rows.size == 0:
        return table
    pairs = np.stack((represented[rows, dims], denominator[rows]), axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    bits = np.stack(
        [_head_target_bits(case, int(count), int(total)) for count, total in unique]
    )
    table[rows, :, dims] = bits[inverse.reshape(-1)]
    return table


def uniform_expected_bits(
    case: UniformCase,
    exposure: str,
    mutation: UniformMutation | str | None = None,
) -> np.ndarray:
    """Construct correctly rounded synthetic output bits for one expectation."""
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)
    first, last, legal_count = _legal_counts(case, selected)
    denominator = legal_count + int(selected == UniformMutation.DENOMINATOR_PLUS_ONE)
    _, represented = _exposure_representation(case, exposure, first, last)
    table = _target_table(case, represented, denominator)
    return np.broadcast_to(table, case.output_shape).copy()


def _head_target_bits(
    case: UniformCase, represented: int, denominator: int
) -> np.ndarray:
    """Per query head, the correctly rounded ``represented * 2**h_kv / N`` bits."""
    return np.array(
        [
            target_bits(represented << exponent, denominator, case.dtype)
            for exponent in case.head_scale_exponents
        ],
        dtype=np.uint16,
    )


def correct_uniform_observation(
    case: UniformCase,
    *,
    basis_offsets: Sequence[int] | None = None,
) -> UniformObservation:
    """Build a complete correctly rounded synthetic observation."""
    offsets = (
        uniform_basis_offsets(case) if basis_offsets is None else tuple(basis_offsets)
    )
    exposures = [
        (f"basis@{offset}", uniform_expected_bits(case, f"basis@{offset}"))
        for offset in offsets
    ]
    exposures.append(("folded", uniform_expected_bits(case, "folded")))
    return UniformObservation(tuple(exposures))


def broadcast_query_head(
    observation: UniformObservation, head: int = 0
) -> UniformObservation:
    """Copy one query head's output into every head of every exposure.

    The observation a kernel with a head-indexing bug would produce (every head
    reading one V head, or one output head written everywhere). The head-specific
    V scales must make the policy reject it."""
    exposures = []
    for name, values in observation.exposures:
        if not 0 <= head < values.shape[2]:
            raise IndexError(head)
        broadcast = np.repeat(values[:, :, head : head + 1, :], values.shape[2], 2)
        exposures.append((name, broadcast))
    return UniformObservation(tuple(exposures))


@dataclass(frozen=True)
class UniformNqStats:
    """One exposure's reduced evidence for a legal count."""

    rows: int
    compared: int
    exact: int
    max_ulp: int
    max_stored_sum_residual: float


@dataclass
class _MutableNqStats:
    rows: int = 0
    compared: int = 0
    exact: int = 0
    max_ulp: int = 0
    max_stored_sum_residual: float = 0.0


@dataclass(frozen=True)
class UniformResult:
    """Decomposed uniform-probability verdict plus per-``N_q`` diagnostics."""

    violations: tuple[str, ...]
    invalid_count: int
    max_negative: float
    max_masked_abs: float
    total_masked_mass: float
    max_row_masked_mass: float
    max_trailing_abs: float
    max_ulp: int
    worst_ulp_nq: int
    worst_coordinate: tuple[int, int, int, int] | None
    worst_exposure: str | None
    compared_count: int
    exact_count: int
    max_stored_sum_residual: float
    real_row_sum_error: float
    legacy_abs_residual: float
    focused_row_errors: Mapping[int, float]
    max_ulp_within_proven: int
    max_ulp_beyond_proven: int
    per_nq: Mapping[str, Mapping[int, UniformNqStats]]

    @property
    def exact_rate(self) -> float:
        """Return the target-bit exact-hit fraction."""
        return self.exact_count / self.compared_count if self.compared_count else 0.0


def _proven_limit(dtype: str, exposure: str) -> int:
    fraction_bits = _FORMAT[dtype][0]
    limit = 1 << (21 - fraction_bits)
    return limit * 2 // 3 if exposure == "folded" else limit


def _freeze_groups(
    groups: Mapping[str, Mapping[int, _MutableNqStats]],
) -> Mapping[str, Mapping[int, UniformNqStats]]:
    frozen: dict[str, Mapping[int, UniformNqStats]] = {}
    for exposure, by_nq in groups.items():
        frozen[exposure] = MappingProxyType(
            {
                nq: UniformNqStats(
                    rows=item.rows,
                    compared=item.compared,
                    exact=item.exact,
                    max_ulp=item.max_ulp,
                    max_stored_sum_residual=item.max_stored_sum_residual,
                )
                for nq, item in sorted(by_nq.items())
            }
        )
    return MappingProxyType(frozen)


def _update_groups(
    groups: dict[int, _MutableNqStats],
    legal_count: np.ndarray,
    *,
    rows: int = 0,
    compared: np.ndarray | None = None,
    exact: np.ndarray | None = None,
    max_ulp: np.ndarray | None = None,
    max_residual: np.ndarray | None = None,
) -> None:
    """Fold per-query evidence into one exposure's per-``N_q`` groups.

    ``rows`` is the output-row count each query adds (``B * Hq``); the arrays are
    indexed by query."""
    for count in np.unique(legal_count):
        members = legal_count == count
        group = groups.setdefault(int(count), _MutableNqStats())
        group.rows += rows * int(np.count_nonzero(members))
        if compared is not None:
            group.compared += int(compared[members].sum())
        if exact is not None:
            group.exact += int(exact[members].sum())
        if max_ulp is not None:
            group.max_ulp = max(group.max_ulp, int(max_ulp[members].max()))
        if max_residual is not None:
            group.max_stored_sum_residual = max(
                group.max_stored_sum_residual, float(max_residual[members].max())
            )


def _row_sum_errors(
    actual: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    represented: np.ndarray,
    legal_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the largest stored row-sum residual and real row-sum error.

    Only rows whose exposure covers the whole legal set (``represented ==
    N_q``) and whose summed outputs are all finite take part; every other row
    contributes 0. ``actual``/``valid`` are ``[B, Sq, Hq]``, ``target`` broadcasts
    to them, and ``represented``/``legal_count`` are per query."""
    complete = (represented == legal_count)[None, :, None] & valid
    residual = np.where(complete, np.abs(actual - target), 0.0).max(axis=(0, 2))
    real = np.where(complete, np.abs(actual - 1.0), 0.0).max(axis=(0, 2))
    return residual, real


def analyze_uniform_policy(
    case: UniformCase,
    observation: UniformObservation,
    *,
    budget: int,
    mutation: UniformMutation | str | None = None,
    focused_rows: Sequence[int] = UNIFORM_FOCUSED_ROWS,
    require_complete: bool = True,
) -> UniformResult:
    """Apply the decomposed zero-ULP uniform-probability policy to raw output bits."""
    if budget != 0:
        raise ValueError(
            "the uniform-probability check currently supports only budget=0"
        )
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)
    _validate_observation(case, observation, require_complete=require_complete)

    row_shape = (case.batch, case.seqlen_q, case.nhead_q)
    masked_row_mass = np.zeros(row_shape, dtype=np.float64)
    basis_actual = np.zeros(row_shape, dtype=np.float64)
    basis_target = np.zeros(row_shape, dtype=np.float64)
    basis_valid = np.ones(row_shape, dtype=bool)
    basis_represented = np.zeros(case.seqlen_q, dtype=np.int64)
    basis_present = False

    invalid_count = 0
    max_negative = 0.0
    max_masked_abs = 0.0
    total_masked_mass = 0.0
    max_trailing_abs = 0.0
    max_ulp = 0
    worst_ulp_nq = 0
    worst_coordinate: tuple[int, int, int, int] | None = None
    worst_exposure: str | None = None
    compared_count = 0
    exact_count = 0
    max_stored_sum_residual = 0.0
    real_row_sum_error = 0.0
    legacy_abs_residual = 0.0
    max_ulp_within_proven = 0
    max_ulp_beyond_proven = 0
    focus = {query: 0.0 for query in focused_rows if 0 <= query < case.seqlen_q}
    groups: dict[str, dict[int, _MutableNqStats]] = {}

    # Everything below is evaluated on whole [B, Sq, Hq, D] arrays; the per-query
    # results are reduced over (B, Hq, D) and folded into per-N_q groups.
    first, last, legal_count = _legal_counts(case, selected)
    denominator = legal_count + int(selected == UniformMutation.DENOMINATOR_PLUS_ONE)
    heads = case.batch * case.nhead_q
    # A mutated rule can leave a row with no legal key: every output of such a
    # row is counted invalid, once per exposure.
    empty_row_outputs = int(np.count_nonzero(legal_count == 0)) * heads
    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)

    # Magnitude diagnostics are in probability units: each head's values are
    # divided by its exact power-of-two V scale. Target-bit ULP checks use the
    # raw bits against the scaled targets, so they stay head-aware.
    head_scales = np.ldexp(1.0, np.array(case.head_scale_exponents))
    for exposure, raw in observation.exposures:
        decoded = decode_bits(raw, case.dtype) / head_scales[None, None, :, None]
        finite = (raw & exponent_mask) != exponent_mask
        invalid_count += int(np.count_nonzero(~finite)) + empty_row_outputs
        negative = (
            finite & ((raw & np.uint16(0x8000)) != 0) & ((raw & np.uint16(0x7FFF)) != 0)
        )
        if np.any(negative):
            max_negative = max(max_negative, float(np.max(np.abs(decoded[negative]))))

        has_key, represented = _exposure_representation(case, exposure, first, last)
        targets = _target_table(case, represented, denominator)
        target_values = decode_bits(targets, case.dtype) / head_scales[None, :, None]
        legal = (represented > 0)[None, :, None, :]
        considered = legal & finite
        magnitudes = np.where(finite, np.abs(decoded), 0.0)

        trailing = ~has_key[None, None, None, :]
        max_trailing_abs = max(
            max_trailing_abs, float(np.max(np.where(trailing, magnitudes, 0.0)))
        )
        masked = (has_key[None, :] & (represented == 0))[None, :, None, :]
        masked_magnitudes = np.where(masked, magnitudes, 0.0)
        max_masked_abs = max(max_masked_abs, float(np.max(masked_magnitudes)))
        total_masked_mass += float(np.sum(masked_magnitudes))
        masked_row_mass += masked_magnitudes.sum(axis=3)

        distances = np.abs(
            _ordered_array(raw).astype(np.int64)
            - _ordered_array(targets).astype(np.int64)[None]
        )
        ulps = np.where(considered, distances, 0)
        compared_rows = np.count_nonzero(considered, axis=(0, 2, 3))
        exact_rows = np.count_nonzero(considered & (distances == 0), axis=(0, 2, 3))
        ulp_rows = ulps.max(axis=(0, 2, 3))
        compared_count += int(compared_rows.sum())
        exact_count += int(exact_rows.sum())
        exposure_groups = groups.setdefault(exposure, {})
        _update_groups(
            exposure_groups,
            legal_count,
            rows=heads,
            compared=compared_rows,
            exact=exact_rows,
            max_ulp=ulp_rows,
        )
        within = legal_count < _proven_limit(case.dtype, exposure)
        if np.any(within):
            max_ulp_within_proven = max(
                max_ulp_within_proven, int(ulp_rows[within].max())
            )
        if not np.all(within):
            max_ulp_beyond_proven = max(
                max_ulp_beyond_proven, int(ulp_rows[~within].max())
            )

        # The worst coordinate is the first (query, dim) reaching this exposure's
        # maximum, then the first (batch, head) within it.
        ulp_cells = ulps.max(axis=(0, 2))
        exposure_max = int(ulp_cells.max())
        if exposure_max > max_ulp:
            query, dim = np.unravel_index(
                int(np.argmax(ulp_cells == exposure_max)), ulp_cells.shape
            )
            cell = np.where(finite[:, query, :, dim], distances[:, query, :, dim], -1)
            batch, head = np.unravel_index(int(np.argmax(cell)), cell.shape)
            max_ulp = exposure_max
            worst_ulp_nq = int(legal_count[query])
            worst_exposure = exposure
            worst_coordinate = (int(batch), int(query), int(head), int(dim))

        real_expected = np.divide(
            represented,
            denominator[:, None],
            out=np.zeros(represented.shape, dtype=np.float64),
            where=represented > 0,
        )
        errors = np.where(
            considered, np.abs(decoded - real_expected[None, :, None, :]), 0.0
        )
        error_rows = errors.max(axis=(0, 2, 3))
        legacy_abs_residual = max(legacy_abs_residual, float(error_rows.max()))
        for query in focus:
            focus[query] = max(focus[query], float(error_rows[query]))

        row_actual = np.where(considered, decoded, 0.0).sum(axis=3)
        row_target = np.where(legal[0], target_values, 0.0).sum(axis=2)[None]
        row_valid = np.all(np.where(legal, finite, True), axis=3)
        row_represented = represented.sum(axis=1)
        if exposure == "folded":
            residual, real = _row_sum_errors(
                row_actual, row_target, row_valid, row_represented, legal_count
            )
            max_stored_sum_residual = max(
                max_stored_sum_residual, float(residual.max())
            )
            real_row_sum_error = max(real_row_sum_error, float(real.max()))
            _update_groups(exposure_groups, legal_count, max_residual=residual)
        else:
            basis_present = basis_present or bool(np.any(represented))
            basis_actual += row_actual
            basis_target += row_target
            basis_valid &= row_valid
            basis_represented += row_represented

    if basis_present:
        basis_groups = groups.setdefault("basis_union", {})
        residual, real = _row_sum_errors(
            basis_actual, basis_target, basis_valid, basis_represented, legal_count
        )
        max_stored_sum_residual = max(max_stored_sum_residual, float(residual.max()))
        real_row_sum_error = max(real_row_sum_error, float(real.max()))
        _update_groups(basis_groups, legal_count, rows=heads, max_residual=residual)

    max_row_masked_mass = float(np.max(masked_row_mass))
    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if max_negative != 0.0:
        violations.append(f"negative_max={max_negative:.9g}")
    if max_masked_abs != 0.0:
        violations.append(f"masked_max={max_masked_abs:.9g}")
    if total_masked_mass != 0.0:
        violations.append(f"masked_mass={total_masked_mass:.9g}")
    if max_trailing_abs != 0.0:
        violations.append(f"trailing_max={max_trailing_abs:.9g}")
    if max_ulp > budget:
        violations.append(f"target_ulp={max_ulp}>{budget} at N_q={worst_ulp_nq}")
    if max_stored_sum_residual != 0.0:
        violations.append(f"stored_row_sum={max_stored_sum_residual:.9g}")
    if compared_count == 0:
        violations.append("no_legal_coordinate_observed")

    return UniformResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        max_negative=max_negative,
        max_masked_abs=max_masked_abs,
        total_masked_mass=total_masked_mass,
        max_row_masked_mass=max_row_masked_mass,
        max_trailing_abs=max_trailing_abs,
        max_ulp=max_ulp,
        worst_ulp_nq=worst_ulp_nq,
        worst_coordinate=worst_coordinate,
        worst_exposure=worst_exposure,
        compared_count=compared_count,
        exact_count=exact_count,
        max_stored_sum_residual=max_stored_sum_residual,
        real_row_sum_error=real_row_sum_error,
        legacy_abs_residual=legacy_abs_residual,
        focused_row_errors=MappingProxyType(focus),
        max_ulp_within_proven=max_ulp_within_proven,
        max_ulp_beyond_proven=max_ulp_beyond_proven,
        per_nq=_freeze_groups(groups),
    )


def format_uniform_result(result: UniformResult) -> str:
    """Render the policy verdict and worst target-bit diagnostics."""
    coordinate = (
        "n/a"
        if result.worst_coordinate is None
        else ":".join(str(value) for value in result.worst_coordinate)
    )
    violations = ", ".join(result.violations) if result.violations else "none"
    return (
        f"violations={violations}; max_ulp={result.max_ulp} "
        f"at {result.worst_exposure or 'n/a'} b:q:h:d={coordinate} "
        f"N_q={result.worst_ulp_nq}; exact={result.exact_count}/"
        f"{result.compared_count} ({result.exact_rate:.6f}); "
        f"stored_sum={result.max_stored_sum_residual:.9g}; "
        f"legacy_abs={result.legacy_abs_residual:.9g}"
    )
