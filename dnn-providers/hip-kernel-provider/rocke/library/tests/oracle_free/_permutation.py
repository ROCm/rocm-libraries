# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free fixtures, token maps, derived bound and analyzer for the
key-permutation relation.

Unmasked attention depends on its keys only as a set of (key, value) pairs, so
permuting the key tokens of K and V together leaves the output unchanged. The
relation compares two launches of the subject on related inputs. No expected
output is computed and there is no attention implementation in this module: the
tolerance follows from the kernel's arithmetic path and from input statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._analytical import _DTYPES, _FORMAT, decode_bits

# Token span of the cross-tile and partial-final-tile maps. A test-space
# boundary: it moves data between streaming steps without assuming the
# subject's private tile size.
CANONICAL_SPAN = 16

# Q and K entries are n/8 with |n| <= 24. Every product is then a multiple of
# 1/64 and every partial sum of a D-term contraction is exact in FP32 while
# D * 24**2 <= 2**24, so a key's score does not depend on where the key sits.
_LATTICE_LIMIT = 24
_LATTICE_DRAWS = 4
_LATTICE_DRAW_VALUES = 2 * _LATTICE_LIMIT // _LATTICE_DRAWS + 1
_SEED_Q = 0x51
_SEED_K = 0x4B
_SEED_V = 0x56

# Smallest normal exponent of each subject dtype.
_MIN_EXPONENT = {"fp16": -14, "bf16": -126}
# One FP32 unit of faithful rounding: an operation returns one of the two FP32
# neighbours of its exact result.
_FP32_UNIT = 2.0**-23
# Range of the largest |V| the bound is evaluated for. Inside it no FP32
# underflow term below is material and no output overflows FP16.
_VALUE_MAX_RANGE = (2.0**-10, 2.0**10)


class TokenMap(str, Enum):
    """Key-token permutations, as destination-to-source maps."""

    SHIFT_ONE = "shift_one"
    REVERSE = "reverse"
    CROSS_TILE = "cross_tile"
    PARTIAL_FINAL_TILE = "partial_final_tile"
    ADJACENT_SWAP = "adjacent_swap"


class PairingMutation(str, Enum):
    """Input transformations that break the key-value pairing."""

    KEY_ONLY = "key_only"
    VALUE_ONLY = "value_only"
    DIFFERENT_MAPS = "different_maps"


@dataclass(frozen=True)
class PermutationCase:
    """Logical unmasked dense-attention shape for one key-permutation case."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str

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
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")

    @property
    def mask(self) -> str:
        """The relation holds only without a positional mask."""
        return "none"

    @property
    def window(self) -> int:
        """No sliding window: a window is a positional mask."""
        return 0

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sq, Hq, D]`` output shape."""
        return (self.batch, self.seqlen_q, self.nhead_q, self.head_dim)

    @property
    def key_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sk, Hkv, D]`` K/V shape."""
        return (self.batch, self.seqlen_k, self.nhead_k, self.head_dim)

    @property
    def case_id(self) -> str:
        """Return a stable identifier for test ids and resolved-subject checks."""
        return (
            f"{self.dtype}-none-b{self.batch}-sq{self.seqlen_q}-"
            f"sk{self.seqlen_k}-hq{self.nhead_q}-hk{self.nhead_k}-d{self.head_dim}"
        )

    def require_scope(self) -> None:
        """Refuse a case the lane's maps or the bound's derivation do not cover.

        Every quick map must exist, and ``cross_tile`` needs more than
        ``CANONICAL_SPAN`` key tokens. That also keeps the case away from the
        short lengths where the maps coincide: with two tokens ``shift_one``,
        ``reverse`` and ``adjacent_swap`` are one map, and exchanging the two
        values commutes with it."""
        if self.seqlen_k <= CANONICAL_SPAN:
            raise ValueError(
                f"case {self.case_id} needs more than {CANONICAL_SPAN} key tokens "
                "for the lane's map set"
            )
        if self.head_dim * _LATTICE_LIMIT**2 > 1 << 24:
            raise ValueError(
                f"case {self.case_id}: the lattice contraction is exact in FP32 "
                f"only while head_dim * {_LATTICE_LIMIT}**2 <= 2**24"
            )


# Unmasked GQA (four query heads per KV head) on both subject dtypes. Square,
# because the gfx942 dense kernel serves unmasked attention only at
# seqlen_q == seqlen_k.
PERMUTATION_SQUARE_FP16 = PermutationCase(1, 256, 256, 8, 2, 64, "fp16")
PERMUTATION_SQUARE_BF16 = PermutationCase(1, 256, 256, 8, 2, 64, "bf16")
# The GPU lane's cases. A CPU contract pins the lane's parametrization to them, so
# a case cannot silently stop executing.
PERMUTATION_QUICK_CASES = (PERMUTATION_SQUARE_FP16, PERMUTATION_SQUARE_BF16)
# The maps the GPU lane applies to both K and V, and the map its pairing
# mutations start from. The mutations need a map that moves every token.
PERMUTATION_QUICK_MAPS = (
    TokenMap.SHIFT_ONE,
    TokenMap.REVERSE,
    TokenMap.CROSS_TILE,
    TokenMap.ADJACENT_SWAP,
)
PERMUTATION_MUTATION_MAP = TokenMap.REVERSE


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
    values = (total - _LATTICE_LIMIT).astype(np.float32) / np.float32(8.0)
    return values.reshape(shape)


def _signs(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Pseudo-random ``+1`` / ``-1`` values from the top bit of each element's hash."""
    bits = _hash64(int(np.prod(shape)), seed)
    negative = (bits >> np.uint64(63)).astype(np.float32)
    return (np.float32(1.0) - np.float32(2.0) * negative).reshape(shape)


def permutation_inputs(
    case: PermutationCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build nondegenerate Q, K on the exact-contraction lattice and sign-valued V.

    Every element is exact in FP16 and BF16. Q and K give each row moderately
    concentrated, nonuniform attention; V is ``+1`` or ``-1`` per element, so a
    value token paired with the wrong key changes the output."""
    case.require_scope()
    q = _lattice(case.output_shape, _SEED_Q)
    k = _lattice(case.key_shape, _SEED_K)
    v = _signs(case.key_shape, _SEED_V)
    return q, k, v


def _require_permutation(source: np.ndarray, seqlen_k: int) -> None:
    expected = np.arange(seqlen_k)
    if source.shape != expected.shape or not np.array_equal(np.sort(source), expected):
        raise ValueError("a token map must contain every key index exactly once")


def token_map(kind: TokenMap | str, seqlen_k: int) -> np.ndarray:
    """Return one destination-to-source map: destination ``k`` receives source
    token ``map[k]``.

    ``shift_one`` is ``k + 1``, ``reverse`` is ``Sk - 1 - k`` and ``cross_tile``
    is ``k + 16``, all modulo ``Sk``. ``partial_final_tile`` moves the short tail
    span ``Sk mod 16`` to the front. ``adjacent_swap`` exchanges tokens 0 and 1
    and is the one map outside the group the others generate."""
    try:
        selected = kind if isinstance(kind, TokenMap) else TokenMap(kind)
    except ValueError as error:
        raise ValueError(f"unknown token map {kind!r}") from error
    if seqlen_k < 2:
        raise ValueError("a key permutation needs at least two key tokens")
    index = np.arange(seqlen_k, dtype=np.int64)
    if selected is TokenMap.SHIFT_ONE:
        source = (index + 1) % seqlen_k
    elif selected is TokenMap.REVERSE:
        source = seqlen_k - 1 - index
    elif selected is TokenMap.ADJACENT_SWAP:
        source = index.copy()
        source[:2] = (1, 0)
    else:
        if seqlen_k <= CANONICAL_SPAN:
            raise ValueError(
                f"{selected.value} needs more than {CANONICAL_SPAN} key tokens"
            )
        shift = CANONICAL_SPAN
        if selected is TokenMap.PARTIAL_FINAL_TILE:
            tail = seqlen_k % CANONICAL_SPAN
            if tail == 0:
                raise ValueError(
                    f"{selected.value} needs a partial final span: seqlen_k "
                    f"{seqlen_k} is a multiple of {CANONICAL_SPAN}"
                )
            shift = seqlen_k - tail
        source = (index + shift) % seqlen_k
    _require_permutation(source, seqlen_k)
    return source


def mutated_token_maps(
    mutation: PairingMutation | str, kind: TokenMap | str, seqlen_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(key, value)`` maps of one pairing mutation of ``kind``.

    The two maps always differ, so the transformed K and V no longer hold the
    same (key, value) pairs. ``different_maps`` gives V the ``reverse`` map, or
    ``shift_one`` when K already uses ``reverse``; where those coincide with the
    K map, V keeps the identity."""
    try:
        selected = (
            mutation
            if isinstance(mutation, PairingMutation)
            else PairingMutation(mutation)
        )
    except ValueError as error:
        raise ValueError(f"unknown pairing mutation {mutation!r}") from error
    primary = token_map(kind, seqlen_k)
    identity = np.arange(seqlen_k, dtype=np.int64)
    if selected is PairingMutation.KEY_ONLY:
        return primary, identity
    if selected is PairingMutation.VALUE_ONLY:
        return identity, primary
    reverse = TokenMap(kind) is TokenMap.REVERSE
    alternate = token_map(TokenMap.SHIFT_ONE if reverse else TokenMap.REVERSE, seqlen_k)
    if np.array_equal(alternate, primary):
        alternate = identity
    return primary, alternate


def permute_tokens(values: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Move whole key-token vectors of a ``[B, Sk, Hkv, D]`` array: destination
    token ``k`` receives source token ``source[k]``, for every batch element and
    head. Elements are copied, never converted, so the transformation is exact."""
    if values.ndim != 4:
        raise ValueError("token permutation expects a [B, Sk, Hkv, D] array")
    _require_permutation(np.asarray(source), values.shape[1])
    return np.ascontiguousarray(values[:, np.asarray(source)])


def score_spread_bound(q: np.ndarray, k: np.ndarray, scale: float) -> float:
    """Bound the spread of one row's log2-domain scores from input norms alone.

    ``|score| <= |q| |k| scale log2(e)`` for every (query, key) pair, so any two
    scores of a row differ by at most twice that. No score is evaluated. The
    result is inflated to cover the FP32 roundings of the kernel's score scale."""
    if not 0.0 < scale < math.inf:
        raise ValueError("scale must be positive and finite")
    q_norm = math.sqrt(float(np.max(np.sum(np.square(q, dtype=np.float64), axis=-1))))
    k_norm = math.sqrt(float(np.max(np.sum(np.square(k, dtype=np.float64), axis=-1))))
    return 2.0 * q_norm * k_norm * scale * math.log2(math.e) * (1.0 + 2.0**-20)


@dataclass(frozen=True)
class PermutationBound:
    """Derived tolerance of the key-permutation relation for one configuration.

    ``residual`` bounds ``|O_A - O_B|`` at every stored coordinate; ``magnitude``
    bounds every stored ``|O|``. Each run's error splits into the probability
    cast, the output store and the FP32 remainder."""

    dtype: str
    keys: int
    key_tiles: int
    value_max: float
    score_spread: float
    cast_term: float
    output_term: float
    fp32_term: float
    per_run: float
    residual: float
    magnitude: float


def permutation_bound(
    dtype: str,
    *,
    keys: int,
    key_tiles: int,
    value_max: float,
    score_spread: float,
) -> PermutationBound:
    """Bound the stored residual of two single-pass launches on permuted keys.

    Per row, both launches see the same FP32 scores as a multiset, so the same
    row maximum and the same ideal weights ``p_k = exp2(s_k - max)``. Each
    launch's stored output is within ``per_run`` of the ideal
    ``sum(p_k v_k) / sum(p_k)``; the residual bound is twice that. With
    ``u = 2**-23`` and ``c = 2**-(p+1)`` for a dtype with ``p`` fraction bits:

    - weights: one ``exp2`` within one FP32 unit and one rounded argument per key
      tile, ``(1 + u)**n_T * 2**(n_T u spread) - 1``;
    - probability cast: ``c`` per weight (round-to-nearest-even), because the
      accumulator consumes cast weights and the normalizer uncast ones, plus half
      the smallest subnormal per key;
    - FP32 accumulation of the numerator and of the normalizer in any order
      across ``n_T`` rescaled tiles, ``gamma(keys + n_T)``, and the reciprocal;
    - the output store: one FP32 product and one round-to-nearest-even
      conversion.

    The bound depends on the inputs only through ``value_max`` and
    ``score_spread``, never on an observed output."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if keys < 2:
        raise ValueError("the relation needs at least two keys")
    if not 1 <= key_tiles <= keys:
        raise ValueError("key_tiles must lie in [1, keys]")
    low, high = _VALUE_MAX_RANGE
    if not low <= value_max <= high:
        raise ValueError(f"value_max must lie in [{low}, {high}]")
    if not 0.0 <= score_spread < math.inf:
        raise ValueError("score_spread must be non-negative and finite")

    fraction_bits = _FORMAT[dtype][0]
    min_exponent = _MIN_EXPONENT[dtype]
    unit = _FP32_UNIT
    cast = 2.0 ** -(fraction_bits + 1)
    subnormal_half = 2.0 ** (min_exponent - fraction_bits - 1)
    steps = (keys + key_tiles) * unit
    if steps >= 0.5:
        raise ValueError("too many keys for the FP32 accumulation bound")
    gamma = steps / (1.0 - steps)
    # FP32 underflow of an exponential, a rescale or a product, per key.
    underflow = keys * (key_tiles + 1) * 2.0**-115

    weight = math.expm1(
        key_tiles * math.log1p(unit) + key_tiles * unit * score_spread * math.log(2.0)
    )
    weights = weight + underflow
    cast_error = cast * (1.0 + weights) + keys * subnormal_half * (1.0 + weight)
    accumulation = gamma * (1.0 + weights + cast_error) + underflow
    numerator = weights + cast_error + accumulation
    normalizer = weights + gamma * (1.0 + weights) + underflow
    if normalizer >= 0.5:
        raise ValueError("the normalizer error bound left its valid range")
    ratio = (unit + normalizer + numerator * (1.0 + unit)) / (1.0 - normalizer)

    largest = value_max * (1.0 + ratio)
    # floor(log2(x)) is frexp's exponent minus one.
    exponent = max(math.frexp(largest * (1.0 + unit))[1] - 1, min_exponent)
    output_term = unit * largest + 2.0 ** (exponent - fraction_bits - 1)
    cast_term = value_max * cast
    per_run = value_max * ratio + output_term
    return PermutationBound(
        dtype=dtype,
        keys=keys,
        key_tiles=key_tiles,
        value_max=value_max,
        score_spread=score_spread,
        cast_term=cast_term,
        output_term=output_term,
        fp32_term=per_run - cast_term - output_term,
        per_run=per_run,
        residual=math.nextafter(2.0 * per_run, math.inf),
        magnitude=math.nextafter(
            largest * (1.0 + unit) + 2.0 ** (exponent - fraction_bits - 1), math.inf
        ),
    )


def permutation_case_bound(
    case: PermutationCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    key_tiles: int,
    scale: float,
) -> PermutationBound:
    """Evaluate the bound for one case's inputs at the subject's tile count."""
    return permutation_bound(
        case.dtype,
        keys=case.seqlen_k,
        key_tiles=key_tiles,
        value_max=float(np.max(np.abs(v))),
        score_spread=score_spread_bound(q, k, scale),
    )


@dataclass(frozen=True)
class PermutationResult:
    """Verdict and diagnostics of one pair of raw-bit outputs.

    ``violations`` is the verdict of the relation. ``head_exceed_counts`` holds,
    per ``(batch, query head)``, how many coordinates exceed the bound: a pairing
    mutation must be visible in every head."""

    violations: tuple[str, ...]
    invalid_count: int
    magnitude_count: int
    exceed_count: int
    head_exceed_counts: tuple[tuple[int, ...], ...]
    max_abs: float
    worst_coordinate: tuple[int, int, int, int] | None
    signed_at_worst: float
    mean_signed: float
    rms: float
    bound: float

    @property
    def bound_ratio(self) -> float:
        """Largest residual as a fraction of the bound; a drift diagnostic."""
        return self.max_abs / self.bound

    @property
    def unresponsive_heads(self) -> tuple[tuple[int, int], ...]:
        """``(batch, query head)`` pairs with no coordinate beyond the bound."""
        return tuple(
            (batch, head)
            for batch, counts in enumerate(self.head_exceed_counts)
            for head, count in enumerate(counts)
            if count == 0
        )


def analyze_permutation(
    case: PermutationCase,
    bound: PermutationBound,
    base: np.ndarray,
    other: np.ndarray,
) -> PermutationResult:
    """Compare two raw-bit outputs against the derived bound.

    Violations: a non-finite stored value in either output (an unwritten
    coordinate keeps its NaN prefill), a stored magnitude beyond
    ``bound.magnitude`` (a blow-up both launches share would otherwise cancel),
    or a residual beyond ``bound.residual``. Residuals are exact FP64 differences
    of the decoded values."""
    if bound.dtype != case.dtype:
        raise ValueError(f"bound dtype {bound.dtype} != case dtype {case.dtype}")
    outputs = []
    for name, values in (("base", base), ("other", other)):
        raw = np.asarray(values)
        if raw.dtype != np.uint16:
            raise ValueError(f"{name} output must contain raw uint16 bits")
        if raw.shape != case.output_shape:
            raise ValueError(f"{name} output shape {raw.shape} != {case.output_shape}")
        outputs.append(raw)
    base_raw, other_raw = outputs

    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    base_finite = (base_raw & exponent_mask) != exponent_mask
    other_finite = (other_raw & exponent_mask) != exponent_mask
    invalid_count = sum(
        int(np.count_nonzero(~finite)) for finite in (base_finite, other_finite)
    )
    base_values = np.where(base_finite, decode_bits(base_raw, case.dtype), 0.0)
    other_values = np.where(other_finite, decode_bits(other_raw, case.dtype), 0.0)
    magnitude_count = sum(
        int(np.count_nonzero(np.abs(values) > bound.magnitude))
        for values in (base_values, other_values)
    )

    compared = base_finite & other_finite
    residual = np.where(compared, base_values - other_values, 0.0)
    magnitude = np.abs(residual)
    exceed = magnitude > bound.residual
    exceed_count = int(np.count_nonzero(exceed))
    head_exceed = np.count_nonzero(exceed, axis=(1, 3))
    compared_count = int(np.count_nonzero(compared))

    max_abs = float(np.max(magnitude))
    worst_coordinate: tuple[int, int, int, int] | None = None
    signed_at_worst = 0.0
    if compared_count:
        flat = int(np.argmax(magnitude))
        index = np.unravel_index(flat, magnitude.shape)
        worst_coordinate = tuple(int(item) for item in index)
        signed_at_worst = float(residual[index])
    mean_signed = float(residual.sum() / compared_count) if compared_count else 0.0
    rms = (
        math.sqrt(float(np.square(residual).sum() / compared_count))
        if compared_count
        else 0.0
    )

    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if magnitude_count:
        violations.append(f"magnitude>{bound.magnitude:.9g} at {magnitude_count}")
    if exceed_count:
        violations.append(
            f"residual={max_abs:.9g}>{bound.residual:.9g} at {exceed_count}"
        )
    if compared_count == 0:
        violations.append("no_coordinate_compared")

    return PermutationResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        magnitude_count=magnitude_count,
        exceed_count=exceed_count,
        head_exceed_counts=tuple(
            tuple(int(item) for item in row) for row in head_exceed
        ),
        max_abs=max_abs,
        worst_coordinate=worst_coordinate,
        signed_at_worst=signed_at_worst,
        mean_signed=mean_signed,
        rms=rms,
        bound=bound.residual,
    )


def format_permutation_result(result: PermutationResult) -> str:
    """Render the relation verdict and the worst-residual diagnostics."""
    coordinate = (
        "n/a"
        if result.worst_coordinate is None
        else ":".join(str(value) for value in result.worst_coordinate)
    )
    violations = ", ".join(result.violations) if result.violations else "none"
    return (
        f"violations={violations}; max_abs={result.max_abs:.9g} "
        f"({result.bound_ratio:.3g} of the bound {result.bound:.9g}) "
        f"at b:q:h:d={coordinate} signed={result.signed_at_worst:.9g}; "
        f"rms={result.rms:.9g}; mean={result.mean_signed:.9g}"
    )
