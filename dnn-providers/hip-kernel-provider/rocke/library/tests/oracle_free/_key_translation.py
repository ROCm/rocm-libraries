# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free fixture, level structure and analyzers for the key-translation
relation.

Adding one common vector to every key token of a KV head adds one constant to
every logit of a query row, and softmax removes a row constant, so the output
must not change. The relation compares launches of the subject on related
inputs. No expected output is computed and there is no attention implementation
in this module: the fixture makes every raw score a known integer level, and
the verdict a row can be held to follows from its levels alone.

Levels are raw scores counted in eighths of a query head's score gap ``g_h``.
The Walsh baseline has the levels 0 and +-8, and every translation moves a row
by an integer. Key tensors are counted in halves of their KV head's scale
``beta_j``, so one integer array describes the keys of every head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy as np

from ._analytical import _DTYPES, _FORMAT, DenseCaseShape, _ordered_array, decode_bits

# The Walsh construction is proven for this head dimension only.
CONSTRUCTION_DIM = 64
# Translation magnitudes of the shift sweep, in eighths of the score gap.
SWEEP_FACTORS = (1, 2, 4, 8, 16, 32, 64, 128)
# Largest base-2 logit magnitude a launch may hold. The subject's online softmax
# starts from a finite sentinel; it stays inert only while every legal logit is
# far above it, and this guard is what the fixture promises.
LEVEL_GUARD = 2.0**13
# A score contraction is exact while the summed magnitudes of its products stay
# below this many units of their common quantum.
LATTICE_SUM_LIMIT = 1 << 24

# One FP32 unit of faithful rounding.
_FP32_UNIT = Fraction(1, 1 << 23)
# The constant the subject scales its launch scale by to form base-2 logits.
_LOG2E = 1.4426950408889634
# The extreme translation and the controls put 2**12 * beta on every component.
_EXTREME_NUMERATOR = 1 << 13
_SEED_UNIT_GAPS = 0x4E37


def _walsh_matrix() -> np.ndarray:
    index = np.arange(CONSTRUCTION_DIM, dtype=np.int64)
    overlap = index[:, None] & index[None, :]
    parity = np.zeros_like(overlap)
    for bit in range(6):
        parity ^= (overlap >> bit) & 1
    matrix = 1 - 2 * parity
    matrix.setflags(write=False)
    return matrix


def _flat_vector() -> np.ndarray:
    index = np.arange(CONSTRUCTION_DIM, dtype=np.int64)
    bits = [(index >> bit) & 1 for bit in range(6)]
    vector = 1 - 2 * ((bits[0] & bits[1]) ^ (bits[2] & bits[3]) ^ (bits[4] & bits[5]))
    vector.setflags(write=False)
    return vector


# ``_WALSH[x, d] = (-1)**popcount(x & d)``: 64 mutually orthogonal sign vectors.
_WALSH = _walsh_matrix()
# The flat-spectrum vector ``u``: ``<H_x, u> = 8 u_x`` for every ``x``.
_FLAT = _flat_vector()


def walsh() -> np.ndarray:
    """The 64 Walsh sign vectors, ``[x, d]``, read-only."""
    return _WALSH


def flat_spectrum() -> np.ndarray:
    """The sign vector ``u`` whose product with every Walsh vector is ``+-8``."""
    return _FLAT


class Baseline(str, Enum):
    """The key tensors a translation starts from."""

    ZERO = "zero"
    WALSH = "walsh"


class Control(str, Enum):
    """Token-dependent key translations: the relation must not hold for them."""

    ALTERNATING_SIGN = "alternating_sign"
    BIT_ONE_GATE = "bit_one_gate"

    @property
    def coordinate(self) -> int:
        """The value coordinate constructed to expose this control."""
        return 0 if self is Control.ALTERNATING_SIGN else 1


@dataclass(frozen=True)
class KeyTranslationCase(DenseCaseShape):
    """Logical unmasked dense-attention shape for one key-translation case."""

    @property
    def mask(self) -> str:
        """The first derivation of the relation covers unmasked attention."""
        return "none"

    @property
    def window(self) -> int:
        """No sliding window: a window is a positional mask."""
        return 0

    @property
    def group(self) -> int:
        """Query heads routed to one KV head."""
        return self.nhead_q // self.nhead_k

    @property
    def case_id(self) -> str:
        """Return a stable identifier for test ids and resolved-subject checks."""
        return self._shape_id(self.mask)

    def require_scope(self) -> None:
        """Refuse a case the construction or the sentinel guard does not cover.

        The Walsh identities are proven for ``head_dim == 64``. The extreme
        translation and the controls put ``2**9 g_h`` on a raw score, which
        must stay inside :data:`LEVEL_GUARD` at the launch scale
        ``1 / sqrt(head_dim)``."""
        if self.head_dim != CONSTRUCTION_DIM:
            raise ValueError(
                f"case {self.case_id}: the construction is proven for head_dim "
                f"{CONSTRUCTION_DIM} only"
            )
        largest = (
            score_scale_constant(1.0 / math.sqrt(self.head_dim))
            * (_EXTREME_NUMERATOR // 2)
            * max(score_gaps(self))
            / 8
            * (1 + _FP32_UNIT)
        )
        if largest > Fraction(LEVEL_GUARD):
            raise ValueError(
                f"case {self.case_id}: the extreme translation reaches a base-2 "
                f"logit of {float(largest):.6g}, beyond the level guard "
                f"{LEVEL_GUARD:g}"
            )


# GQA (four query heads per KV head) on both subject dtypes: the worked example
# of the construction. Square, because the gfx942 dense kernel serves unmasked
# attention only at seqlen_q == seqlen_k.
KEY_TRANSLATION_SQUARE_FP16 = KeyTranslationCase(1, 256, 256, 8, 2, 64, "fp16")
KEY_TRANSLATION_SQUARE_BF16 = KeyTranslationCase(1, 256, 256, 8, 2, 64, "bf16")
# The GPU lane's cases. A CPU contract pins the lane's parametrization to them, so
# a case cannot silently stop executing.
KEY_TRANSLATION_QUICK_CASES = (
    KEY_TRANSLATION_SQUARE_FP16,
    KEY_TRANSLATION_SQUARE_BF16,
)


def per_run_error(
    dtype: str,
    *,
    keys: int,
    key_tiles: int,
    value_max: float,
    score_spread: float,
) -> float:
    """The per-run bound ``T_run`` of the key-permutation derivation, evaluated
    for one format and one base-2 score spread.

    It bounds the distance of every stored element of one launch from the ideal
    output on its own computed scores, so it is ``eps_fwd`` of the general text.
    Reuses ``permutation_bound`` rather than re-implementing the formula."""
    from ._permutation import permutation_bound

    return permutation_bound(
        dtype,
        keys=keys,
        key_tiles=key_tiles,
        value_max=value_max,
        score_spread=score_spread,
    ).per_run


def sweep_ceiling(
    case: KeyTranslationCase,
    factor: int,
    *,
    key_tiles: int,
    scale: float,
) -> np.ndarray:
    """The derived ceiling ``tau_{lambda,h,d}`` per query head and coordinate,
    as a ``[Hq, D]`` array.

    For a row of the ``C_lambda`` sweep in query head ``h`` and coordinate
    ``d``, it is
    ``Delta_{V,d} tanh(eta_lambda ln2 / 2) + T_run(sigma g_h) +
     T_run(sigma g_h + 2 eta_lambda)``
    with ``eta_lambda = u sigma g_h (1 + lambda/8)``. Every term is a
    function of the format, of the fixture, of ``n_T`` read from the artifact,
    and of the value ranges, never of an output.

    ``scale`` is the launch scale (``1 / sqrt(D)``), ``key_tiles`` is the
    subject's tile count. The logit term in position ``[h, d]`` is multiplied
    by the value range ``Delta_{V,d}`` of head ``h``'s KV head. The result is
    upward-safe: the intermediate products are upward-rounded with
    ``_float_not_below``."""
    sigma = score_scale_constant(scale)
    u = _FP32_UNIT
    gaps = score_gaps(case)
    ceilings = np.empty((case.nhead_q, case.head_dim))
    for head in range(case.nhead_q):
        eta = _float_not_below(
            u * sigma * gaps[head] * (1 + Fraction(factor, 8))
        )
        spread_base = _float_not_below(sigma * gaps[head])
        spread_shifted = _float_not_below(
            Fraction(spread_base) + 2 * Fraction(float(eta))
        )
        t_base = per_run_error(
            case.dtype,
            keys=case.seqlen_k,
            key_tiles=key_tiles,
            value_max=1.0,
            score_spread=spread_base,
        )
        t_shifted = per_run_error(
            case.dtype,
            keys=case.seqlen_k,
            key_tiles=key_tiles,
            value_max=1.0,
            score_spread=float(spread_shifted),
        )
        logit_term = math.tanh(float(eta) * math.log(2.0) / 2.0)
        ceilings[head, :] = logit_term + t_base + t_shifted
    ceilings.setflags(write=False)
    return ceilings


def forward_error(
    case: KeyTranslationCase,
    *,
    key_tiles: int,
    score_spread: float,
) -> float:
    """One per-run bound at the given spread, used as ``eps_fwd`` by the
    control designation."""
    return per_run_error(
        case.dtype,
        keys=case.seqlen_k,
        key_tiles=key_tiles,
        value_max=1.0,
        score_spread=score_spread,
    )


def query_scale_exponents(case: KeyTranslationCase) -> tuple[int, ...]:
    """Per query head ``h``, the exponent of its scale ``alpha_h``:
    ``-1 - (h mod group)``, so the heads of one KV head have distinct gaps."""
    return tuple(-1 - head % case.group for head in range(case.nhead_q))


def key_scale_exponents(case: KeyTranslationCase) -> tuple[int, ...]:
    """Per KV head ``j``, the exponent of its scale ``beta_j = 2**j``."""
    return tuple(range(case.nhead_k))


def score_gaps(case: KeyTranslationCase) -> tuple[Fraction, ...]:
    """Per query head, the raw-score gap ``g_h = 64 alpha_h beta_r(h)``."""
    alpha = query_scale_exponents(case)
    beta = key_scale_exponents(case)
    return tuple(
        Fraction(2) ** (6 + alpha[head] + beta[head // case.group])
        for head in range(case.nhead_q)
    )


def score_scale_constant(scale: float) -> Fraction:
    """The FP32 constant the subject multiplies each completed contraction by,
    as an exact rational: the FP32 product of the launch scale and ``log2(e)``.

    That the subject forms it this way, once and after the contraction, is a
    property of its lowered path and is checked there, not here."""
    if not 0.0 < scale < math.inf:
        raise ValueError("scale must be positive and finite")
    return Fraction(float(np.float32(scale) * np.float32(_LOG2E)))


def aligned_positions(case: KeyTranslationCase) -> tuple[np.ndarray, np.ndarray]:
    """Absolute query and key positions of aligned prefill: ``[B, Sq]`` and
    ``[B, Sk]`` with position ``i`` at local index ``i``."""
    return (
        np.tile(np.arange(case.seqlen_q, dtype=np.int64), (case.batch, 1)),
        np.tile(np.arange(case.seqlen_k, dtype=np.int64), (case.batch, 1)),
    )


def _positions(values: np.ndarray, name: str) -> np.ndarray:
    positions = np.asarray(values)
    if positions.ndim != 2 or positions.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a [B, S] integer array")
    if positions.size == 0 or int(positions.min()) < 0:
        raise ValueError(f"{name} must hold non-negative absolute positions")
    return positions.astype(np.int64)


def _case_positions(
    case: KeyTranslationCase, position: np.ndarray, name: str, length: int
) -> np.ndarray:
    positions = _positions(position, name)
    if positions.shape != (case.batch, length):
        raise ValueError(
            f"{name} shape {positions.shape} != {(case.batch, length)} of case "
            f"{case.case_id}"
        )
    return positions


def queries(case: KeyTranslationCase, position_q: np.ndarray) -> np.ndarray:
    """Q ``[B, Sq, Hq, D]``: head ``h`` holds ``alpha_h`` times the Walsh
    vector of the query's absolute position modulo 64."""
    case.require_scope()
    positions = _case_positions(case, position_q, "position_q", case.seqlen_q)
    signs = _WALSH[positions % CONSTRUCTION_DIM].astype(np.float32)
    scales = np.ldexp(np.float32(1.0), np.array(query_scale_exponents(case)))
    return np.ascontiguousarray(
        signs[:, :, None, :] * scales.astype(np.float32)[None, None, :, None]
    )


def values(case: KeyTranslationCase, position_k: np.ndarray) -> np.ndarray:
    """V ``[B, Sk, Hkv, D]``, the same for every KV head.

    Coordinate 0 is 1 on even absolute key positions and coordinate 1 is 1
    where bit 1 of the position is set; both are 0 elsewhere. They make the
    controls observable. The other coordinates are the Walsh signs of the
    position modulo 64."""
    case.require_scope()
    positions = _case_positions(case, position_k, "position_k", case.seqlen_k)
    one_head = _WALSH[positions % CONSTRUCTION_DIM].astype(np.float32)
    one_head[:, :, 0] = positions % 2 == 0
    one_head[:, :, 1] = (positions >> 1) & 1
    return np.ascontiguousarray(
        np.broadcast_to(one_head[:, :, None, :], case.key_shape)
    )


def _hash64(count: int, seed: int) -> np.ndarray:
    """splitmix64 of the element index offset by ``seed``, as ``uint64``.

    An explicit integer hash, so the fixture does not depend on a library's
    random stream."""
    state = np.arange(count, dtype=np.uint64) + (np.uint64(seed) << np.uint64(32))
    state = state + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state ^ (state >> np.uint64(31))


def unit_gap_signs() -> np.ndarray:
    """Per residue ``x``, the sign ``e_x`` of the one-gap translation: the top
    bit of the residue's hash. Both signs occur on even and on odd residues."""
    negative = (_hash64(CONSTRUCTION_DIM, _SEED_UNIT_GAPS) >> np.uint64(63)).astype(
        np.int64
    )
    return 1 - 2 * negative


@dataclass(frozen=True)
class Translation:
    """One common-key translation.

    ``coefficients`` is the translation vector in halves of the KV head's
    scale: every key token of KV head ``j`` receives
    ``coefficients * beta_j / 2``. A row whose query sits at residue ``x``
    moves by ``row_shifts[x]`` eighths of its head's score gap."""

    name: str
    baseline: Baseline
    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != CONSTRUCTION_DIM:
            raise ValueError(
                f"a translation has {CONSTRUCTION_DIM} coefficients, got "
                f"{len(self.coefficients)}"
            )
        if any(int(item) != item for item in self.coefficients):
            raise ValueError("translation coefficients must be integers")
        products = _WALSH @ np.array(self.coefficients, dtype=np.int64)
        if np.any(products % 16):
            raise ValueError(
                f"translation {self.name}: a row shift is not a whole number of "
                "eighths of the score gap"
            )

    @property
    def row_shifts(self) -> tuple[int, ...]:
        """Per query residue, the raw-score shift in eighths of the gap."""
        products = _WALSH @ np.array(self.coefficients, dtype=np.int64)
        return tuple(int(item) for item in products // 16)


def _walsh_combination(weights: np.ndarray) -> tuple[int, ...]:
    """``sum_x weights[x] H_x``: the row at residue ``x`` then moves by
    ``4 weights[x]`` eighths of its gap."""
    return tuple(int(item) for item in np.asarray(weights, dtype=np.int64) @ _WALSH)


def extreme_translation() -> Translation:
    """``2**12 beta u`` on the zero baseline: every row keeps one level and
    moves by ``+-2**9 g_h``, far outside the range of a direct exponential."""
    return Translation(
        "extreme",
        Baseline.ZERO,
        tuple(int(item) for item in _EXTREME_NUMERATOR * _FLAT),
    )


def unit_gap_translation() -> Translation:
    """Rows of the Walsh baseline move by one gap, up or down by residue."""
    return Translation(
        "unit_gaps", Baseline.WALSH, _walsh_combination(2 * unit_gap_signs())
    )


def mixed_gap_translation() -> Translation:
    """Rows of the Walsh baseline move by two gaps where bit 1 of the residue
    is set and by half a gap elsewhere: down on even residues, up on odd."""
    residue = np.arange(CONSTRUCTION_DIM, dtype=np.int64)
    magnitude = np.where((residue >> 1) & 1, 4, 1)
    return Translation(
        "mixed_gaps",
        Baseline.WALSH,
        _walsh_combination(np.where(residue % 2 == 0, -magnitude, magnitude)),
    )


def sweep_translation(factor: int) -> Translation:
    """``factor beta u`` on the Walsh baseline: every row moves by ``factor``
    eighths of its gap, up or down by residue."""
    if factor not in SWEEP_FACTORS:
        raise ValueError(f"sweep factor must be one of {SWEEP_FACTORS}")
    return Translation(
        f"sweep_{factor}",
        Baseline.WALSH,
        tuple(int(item) for item in 2 * factor * _FLAT),
    )


def baseline_key_numerators(position_k: np.ndarray, baseline: Baseline) -> np.ndarray:
    """Baseline keys ``[B, Sk, D]`` in halves of the KV head's scale: zero, or
    the Walsh vector of the key's position modulo 64 with the sign of the
    position's parity."""
    positions = _positions(position_k, "position_k")
    if Baseline(baseline) is Baseline.ZERO:
        return np.zeros((*positions.shape, CONSTRUCTION_DIM), dtype=np.int64)
    sign = 1 - 2 * (positions & 1)
    return 2 * sign[:, :, None] * _WALSH[positions % CONSTRUCTION_DIM]


def translated_key_numerators(
    position_k: np.ndarray, translation: Translation
) -> np.ndarray:
    """The translation's baseline keys plus its vector on every token."""
    return baseline_key_numerators(position_k, translation.baseline) + np.array(
        translation.coefficients, dtype=np.int64
    )


def control_key_numerators(position_k: np.ndarray, control: Control) -> np.ndarray:
    """Control keys ``[B, Sk, D]`` in halves of the KV head's scale: the
    extreme vector with a sign or a gate that depends on the token."""
    positions = _positions(position_k, "position_k")
    if Control(control) is Control.ALTERNATING_SIGN:
        factor = 1 - 2 * (positions & 1)
    else:
        factor = (positions >> 1) & 1
    return factor[:, :, None] * (_EXTREME_NUMERATOR * _FLAT)


def keys_from_numerators(
    case: KeyTranslationCase, numerators: np.ndarray
) -> np.ndarray:
    """K ``[B, Sk, Hkv, D]``: KV head ``j`` holds ``numerators * beta_j / 2``."""
    case.require_scope()
    numerators = np.asarray(numerators)
    expected = (case.batch, case.seqlen_k, case.head_dim)
    if numerators.shape != expected or numerators.dtype.kind not in "iu":
        raise ValueError(f"key numerators must be an integer array of shape {expected}")
    if int(np.abs(numerators).max(initial=0)) >= 1 << 24:
        raise ValueError("a key numerator is not exact in FP32")
    scales = np.ldexp(np.float32(1.0), np.array(key_scale_exponents(case)) - 1)
    return np.ascontiguousarray(
        numerators.astype(np.float32)[:, :, None, :]
        * scales.astype(np.float32)[None, None, :, None]
    )


def baseline_levels(
    position_q: np.ndarray, position_k: np.ndarray, baseline: Baseline
) -> np.ndarray:
    """Raw scores ``[B, Sq, Sk]`` of a baseline, in eighths of the gap.

    Zero keys give level 0. Walsh keys give ``+-8`` where the key's position
    is congruent to the query's modulo 64, with the sign of the query
    position's parity, and 0 elsewhere."""
    query = _positions(position_q, "position_q")
    key = _positions(position_k, "position_k")
    if query.shape[0] != key.shape[0]:
        raise ValueError("position_q and position_k must share the batch dimension")
    levels = np.zeros((query.shape[0], query.shape[1], key.shape[1]), dtype=np.int64)
    if Baseline(baseline) is Baseline.ZERO:
        return levels
    congruent = (query % CONSTRUCTION_DIM)[:, :, None] == (key % CONSTRUCTION_DIM)[
        :, None, :
    ]
    sign = 8 * (1 - 2 * (query & 1))
    return np.where(congruent, sign[:, :, None], levels)


def translated_levels(
    position_q: np.ndarray, position_k: np.ndarray, translation: Translation
) -> np.ndarray:
    """Raw scores ``[B, Sq, Sk]`` after a translation: the baseline levels plus
    the shift of the query's residue, the same for every key of the row."""
    query = _positions(position_q, "position_q")
    shifts = np.array(translation.row_shifts, dtype=np.int64)
    return (
        baseline_levels(query, position_k, translation.baseline)
        + shifts[query % CONSTRUCTION_DIM][:, :, None]
    )


def control_levels(
    position_q: np.ndarray, position_k: np.ndarray, control: Control
) -> np.ndarray:
    """Raw scores ``[B, Sq, Sk]`` of a control, in eighths of the gap:
    ``+-4096`` by key parity, or ``+-4096`` on keys with bit 1 set and 0
    elsewhere, with the sign of the query residue's flat-spectrum entry."""
    query = _positions(position_q, "position_q")
    key = _positions(position_k, "position_k")
    if query.shape[0] != key.shape[0]:
        raise ValueError("position_q and position_k must share the batch dimension")
    if Control(control) is Control.ALTERNATING_SIGN:
        factor = 1 - 2 * (key & 1)
    else:
        factor = (key >> 1) & 1
    row = (_EXTREME_NUMERATOR // 2) * _FLAT[query % CONSTRUCTION_DIM]
    return row[:, :, None] * factor[:, None, :]


def contracted_levels(position_q: np.ndarray, key_numerators: np.ndarray) -> np.ndarray:
    """Raw scores ``[B, Sq, Sk]`` in eighths of the gap, contracted in exact
    integers from the key numerators a launch stores.

    The closed-form levels describe the fixture; this is the same quantity
    taken from the tensor itself, so a launch can require that the two agree
    before a verdict is drawn from the levels."""
    query = _positions(position_q, "position_q")
    numerators = np.asarray(key_numerators)
    if numerators.ndim != 3 or numerators.dtype.kind not in "iu":
        raise ValueError("key numerators must be a [B, Sk, D] integer array")
    if numerators.shape[0] != query.shape[0] or numerators.shape[2] != CONSTRUCTION_DIM:
        raise ValueError("key numerators do not match the query positions")
    products = np.einsum(
        "bqd,bkd->bqk",
        _WALSH[query % CONSTRUCTION_DIM],
        numerators.astype(np.int64),
    )
    if np.any(products % 16):
        raise ValueError("a raw score is not a whole number of eighths of the gap")
    return products // 16


def _legal(levels: np.ndarray, legal: np.ndarray | None) -> np.ndarray:
    if legal is None:
        return np.ones(levels.shape, dtype=bool)
    mask = np.asarray(legal)
    if mask.dtype != bool or mask.shape != levels.shape:
        raise ValueError(f"legal must be a boolean array of shape {levels.shape}")
    if not np.all(mask.any(axis=-1)):
        raise ValueError("a row has no legal key")
    return mask


def _row_extremes(
    levels: np.ndarray, legal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    big = np.iinfo(np.int64).max
    return (
        np.where(legal, levels, big).min(axis=-1),
        np.where(legal, levels, -big).max(axis=-1),
    )


class RowVerdict(str, Enum):
    """What a row of a baseline and its translation can be held to."""

    ONE_LEVEL = "one_level"
    DYADIC = "dyadic"
    CEILING = "ceiling"


@dataclass(frozen=True)
class RowClasses:
    """Per row ``[B, Sq]``, the verdict its levels prove.

    ``one_level``: all legal keys share one raw score in each launch, so the
    two logit rows differ by a constant whatever its rounding. ``dyadic``:
    not one-level, and every legal raw score of both launches is zero or a
    signed power of two, so every logit is scaled exactly. ``ceiling``: neither;
    only a derived bound applies. ``two_level`` marks rows whose baseline has
    more than one level: the rows that can see a per-score defect."""

    one_level: np.ndarray
    dyadic: np.ndarray
    ceiling: np.ndarray
    two_level: np.ndarray

    @property
    def exact(self) -> np.ndarray:
        """Rows held to bit equality."""
        return self.one_level | self.dyadic

    def of(self, verdict: RowVerdict) -> np.ndarray:
        """The rows of one verdict."""
        return getattr(self, RowVerdict(verdict).value)

    def counts(self) -> dict[str, int]:
        """Number of rows per verdict, and of two-level rows."""
        return {
            **{item.value: int(np.count_nonzero(self.of(item))) for item in RowVerdict},
            "two_level": int(np.count_nonzero(self.two_level)),
        }


def classify_rows(
    base_levels: np.ndarray,
    other_levels: np.ndarray,
    legal: np.ndarray | None = None,
) -> RowClasses:
    """Classify every row of a pair of launches from its raw-score levels.

    ``legal`` is the boolean ``[B, Sq, Sk]`` legal-key mask; ``None`` means
    every key. A row without a legal key has no softmax and is refused. The
    gap of a head is a power of two, so a level is zero or a signed power of
    two exactly when its count of eighths is."""
    base = np.asarray(base_levels)
    other = np.asarray(other_levels)
    if base.shape != other.shape or base.ndim != 3:
        raise ValueError("levels must be two [B, Sq, Sk] arrays of one shape")
    if base.dtype.kind not in "iu" or other.dtype.kind not in "iu":
        raise ValueError("levels must be integer counts of eighths of the gap")
    mask = _legal(base, legal)
    base_low, base_high = _row_extremes(base, mask)
    other_low, other_high = _row_extremes(other, mask)
    one_level = (base_low == base_high) & (other_low == other_high)
    dyadic_levels = np.ones(base.shape[:2], dtype=bool)
    for levels in (base, other):
        magnitude = np.abs(levels.astype(np.int64))
        power = (magnitude & (magnitude - 1)) == 0
        dyadic_levels &= np.all(power | ~mask, axis=-1)
    dyadic = dyadic_levels & ~one_level
    return RowClasses(
        one_level=one_level,
        dyadic=dyadic,
        ceiling=~one_level & ~dyadic,
        two_level=base_low != base_high,
    )


def certify_score_lattice(
    case: KeyTranslationCase, q: np.ndarray, k: np.ndarray
) -> int:
    """Certify that the contraction of every (query, key) pair of one launch
    is exact, and return the largest summed product magnitude in units of the
    products' common quantum ``alpha_h beta_j / 2``.

    Every Q element must be ``+-alpha_h`` and every K element an integer
    multiple of ``beta_j / 2``; the summed magnitudes of a key's 64 products
    must not exceed :data:`LATTICE_SUM_LIMIT` quanta. Then every product and
    every partial sum of a contraction is a multiple of one power of two with
    at most 24 significant bits. Fails closed: anything else raises
    ``ValueError`` and no exact score may be assumed."""
    q_values = np.asarray(q, dtype=np.float64)
    k_values = np.asarray(k, dtype=np.float64)
    if q_values.shape != case.output_shape or k_values.shape != case.key_shape:
        raise ValueError(
            f"Q and K shapes {q_values.shape}, {k_values.shape} != "
            f"{case.output_shape}, {case.key_shape}"
        )
    if not (np.all(np.isfinite(q_values)) and np.all(np.isfinite(k_values))):
        raise ValueError("cannot certify the score lattice: Q or K is not finite")
    alpha = np.ldexp(1.0, np.array(query_scale_exponents(case)))
    if not np.array_equal(
        np.abs(q_values), np.broadcast_to(alpha[:, None], q_values.shape)
    ):
        raise ValueError(
            "cannot certify the score lattice: a Q element is not +-alpha_h"
        )
    half_beta = np.ldexp(1.0, np.array(key_scale_exponents(case)) - 1)
    numerators = k_values / half_beta[:, None]
    if not np.array_equal(numerators, np.rint(numerators)):
        raise ValueError(
            "cannot certify the score lattice: a K element is not a multiple of "
            "beta_j / 2"
        )
    largest = int(np.abs(numerators).sum(axis=-1).max())
    if largest > LATTICE_SUM_LIMIT:
        raise ValueError(
            f"cannot certify the score lattice: a key sums to {largest} quanta, "
            f"beyond {LATTICE_SUM_LIMIT}"
        )
    return largest


def _float_not_below(value: Fraction) -> float:
    """The float nearest ``value``, moved up one step if it lies below it."""
    nearest = float(value)
    if Fraction(nearest) >= value:
        return nearest
    return math.nextafter(nearest, math.inf)


def largest_logit(case: KeyTranslationCase, levels: np.ndarray, scale: float) -> float:
    """Upper bound on the magnitude of every base-2 logit of one launch: the
    largest level times the largest gap and the subject's scale constant, with
    one unit of FP32 rounding. Evaluated in exact rationals and rounded up."""
    counts = np.asarray(levels)
    if counts.dtype.kind not in "iu" or counts.size == 0:
        raise ValueError("levels must be a non-empty integer array")
    largest = int(np.abs(counts.astype(np.int64)).max())
    return _float_not_below(
        score_scale_constant(scale)
        * largest
        * max(score_gaps(case))
        / 8
        * (1 + _FP32_UNIT)
    )


def require_level_guard(
    case: KeyTranslationCase, levels: np.ndarray, scale: float
) -> float:
    """Refuse a launch whose logits could leave :data:`LEVEL_GUARD`, and
    return the bound on their magnitude."""
    largest = largest_logit(case, levels, scale)
    if largest > LEVEL_GUARD:
        raise ValueError(
            f"case {case.case_id}: a base-2 logit may reach {largest:.6g}, beyond "
            f"the level guard {LEVEL_GUARD:g}"
        )
    return largest


def value_ranges(case: KeyTranslationCase, v: np.ndarray) -> np.ndarray:
    """Per query head and coordinate ``[Hq, D]``, the range of V over every
    batch element and key of the head's KV head. It is not smaller than the
    range over any row's legal keys."""
    data = np.asarray(v, dtype=np.float64)
    if data.shape != case.key_shape or not np.all(np.isfinite(data)):
        raise ValueError(f"V must be a finite array of shape {case.key_shape}")
    per_kv = data.max(axis=(0, 1)) - data.min(axis=(0, 1))
    return per_kv[np.arange(case.nhead_q) // case.group]


def _raw_outputs(
    case: KeyTranslationCase, base: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate two raw-bit outputs and return them with their finite masks."""
    outputs = []
    for name, data in (("base", base), ("other", other)):
        raw = np.asarray(data)
        if raw.dtype != np.uint16:
            raise ValueError(f"{name} output must contain raw uint16 bits")
        if raw.shape != case.output_shape:
            raise ValueError(f"{name} output shape {raw.shape} != {case.output_shape}")
        outputs.append(raw)
    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    finite = [(raw & exponent_mask) != exponent_mask for raw in outputs]
    return outputs[0], outputs[1], finite[0], finite[1]


def _row_selection(case: KeyTranslationCase, rows: np.ndarray | None) -> np.ndarray:
    if rows is None:
        return np.ones((case.batch, case.seqlen_q), dtype=bool)
    selected = np.asarray(rows)
    expected = (case.batch, case.seqlen_q)
    if selected.dtype != bool or selected.shape != expected:
        raise ValueError(f"rows must be a boolean array of shape {expected}")
    return selected


def _first(flags: np.ndarray) -> tuple[int, ...] | None:
    found = np.argwhere(flags)
    return tuple(int(item) for item in found[0]) if found.size else None


@dataclass(frozen=True)
class BitComparison:
    """Verdict of a bit-equality comparison of two raw-bit outputs.

    ``violations`` is the verdict. ``mismatch_count`` counts stored elements of
    the selected rows whose bits differ; ``zero_sign_count`` is the part of it
    where both elements are zeros of opposite sign."""

    violations: tuple[str, ...]
    invalid_count: int
    compared_rows: int
    mismatch_count: int
    zero_sign_count: int
    head_mismatch_counts: tuple[tuple[int, ...], ...]
    first_mismatch: tuple[int, int, int, int] | None
    first_bits: tuple[int, int] | None


def compare_bits(
    case: KeyTranslationCase,
    base: np.ndarray,
    other: np.ndarray,
    rows: np.ndarray | None = None,
) -> BitComparison:
    """Require two raw-bit outputs to be identical on the selected rows.

    Violations: a non-finite stored value anywhere in either output (an
    unwritten coordinate keeps its NaN prefill), any bit difference on a
    selected row, the sign of a zero included, and an empty selection."""
    base_raw, other_raw, base_finite, other_finite = _raw_outputs(case, base, other)
    selected = _row_selection(case, rows)
    invalid_count = int(np.count_nonzero(~base_finite)) + int(
        np.count_nonzero(~other_finite)
    )
    mismatch = (base_raw != other_raw) & selected[:, :, None, None]
    mismatch_count = int(np.count_nonzero(mismatch))
    zeros = ((base_raw | other_raw) & np.uint16(0x7FFF)) == 0
    first = _first(mismatch)
    compared_rows = int(np.count_nonzero(selected))

    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if mismatch_count:
        violations.append(f"bit_mismatch={mismatch_count}")
    if compared_rows == 0:
        violations.append("no_row_compared")
    return BitComparison(
        violations=tuple(violations),
        invalid_count=invalid_count,
        compared_rows=compared_rows,
        mismatch_count=mismatch_count,
        zero_sign_count=int(np.count_nonzero(mismatch & zeros)),
        head_mismatch_counts=tuple(
            tuple(int(item) for item in row)
            for row in np.count_nonzero(mismatch, axis=(1, 3))
        ),
        first_mismatch=first,
        first_bits=(
            None if first is None else (int(base_raw[first]), int(other_raw[first]))
        ),
    )


def format_bit_comparison(result: BitComparison) -> str:
    """Render the bit-equality verdict and its first mismatch."""
    violations = ", ".join(result.violations) if result.violations else "none"
    first = "n/a"
    if result.first_mismatch is not None and result.first_bits is not None:
        first = (
            ":".join(str(item) for item in result.first_mismatch)
            + f" 0x{result.first_bits[0]:04x}!=0x{result.first_bits[1]:04x}"
        )
    return (
        f"violations={violations}; rows={result.compared_rows}; "
        f"mismatches={result.mismatch_count} ({result.zero_sign_count} zero-sign "
        f"only); first at b:q:h:d={first}"
    )


@dataclass(frozen=True)
class SweepResult:
    """Verdict and diagnostics of one sweep launch against its baseline, on
    the rows held to the derived ceiling.

    ``violations`` is the verdict: a non-finite stored value or a residual
    beyond the ceiling. Everything else is a diagnostic. ``over_one_ulp_count``
    counts residuals of more than one unit in the last place of the subject
    dtype: a correct kernel is expected to stay at or below one, so a nonzero
    count is reported for investigation although it does not fail the run."""

    violations: tuple[str, ...]
    invalid_count: int
    compared_rows: int
    exceed_count: int
    max_abs: float
    max_ulp: int
    over_one_ulp_count: int
    exact_hit_rate: float
    max_normalized: float
    max_ceiling_ratio: float
    worst_coordinate: tuple[int, int, int, int] | None
    head_max_abs: tuple[tuple[float, ...], ...]
    head_max_ulp: tuple[tuple[int, ...], ...]


def analyze_sweep(
    case: KeyTranslationCase,
    base: np.ndarray,
    other: np.ndarray,
    ceiling: np.ndarray,
    value_range: np.ndarray,
    rows: np.ndarray | None = None,
) -> SweepResult:
    """Compare a sweep launch with its baseline against a derived ceiling.

    ``ceiling`` and ``value_range`` broadcast to the output shape
    ``[B, Sq, Hq, D]``; a ``[Hq, D]`` array is the usual form. Residuals are
    exact FP64 differences of the decoded values. The ceiling must be positive
    and finite: it is derived before the launch and never from an output."""
    base_raw, other_raw, base_finite, other_finite = _raw_outputs(case, base, other)
    selected = _row_selection(case, rows)
    limit = np.broadcast_to(np.asarray(ceiling, dtype=np.float64), case.output_shape)
    spread = np.broadcast_to(
        np.asarray(value_range, dtype=np.float64), case.output_shape
    )
    if not np.all(np.isfinite(limit)) or np.any(limit <= 0.0):
        raise ValueError("the ceiling must be positive and finite")
    if not np.all(np.isfinite(spread)) or np.any(spread < 0.0):
        raise ValueError("the value range must be non-negative and finite")

    invalid_count = int(np.count_nonzero(~base_finite)) + int(
        np.count_nonzero(~other_finite)
    )
    compared = base_finite & other_finite & selected[:, :, None, None]
    compared_count = int(np.count_nonzero(compared))
    residual = np.where(
        compared,
        np.where(base_finite, decode_bits(base_raw, case.dtype), 0.0)
        - np.where(other_finite, decode_bits(other_raw, case.dtype), 0.0),
        0.0,
    )
    magnitude = np.abs(residual)
    ulps = np.where(
        compared,
        np.abs(
            _ordered_array(base_raw).astype(np.int64)
            - _ordered_array(other_raw).astype(np.int64)
        ),
        0,
    )
    exceed_count = int(np.count_nonzero(magnitude > limit))
    hits = int(np.count_nonzero(compared & (base_raw == other_raw)))
    normalized = np.divide(
        magnitude, spread, out=np.zeros_like(magnitude), where=spread > 0.0
    )
    max_abs = float(magnitude.max())

    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if exceed_count:
        violations.append(
            f"residual={max_abs:.9g} beyond the ceiling at {exceed_count}"
        )
    if compared_count == 0:
        violations.append("no_coordinate_compared")
    return SweepResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        compared_rows=int(np.count_nonzero(selected)),
        exceed_count=exceed_count,
        max_abs=max_abs,
        max_ulp=int(ulps.max()),
        over_one_ulp_count=int(np.count_nonzero(ulps > 1)),
        exact_hit_rate=hits / compared_count if compared_count else 0.0,
        max_normalized=float(normalized.max()),
        max_ceiling_ratio=float((magnitude / limit).max()),
        worst_coordinate=_first(magnitude == max_abs) if compared_count else None,
        head_max_abs=tuple(
            tuple(float(item) for item in row) for row in magnitude.max(axis=(1, 3))
        ),
        head_max_ulp=tuple(
            tuple(int(item) for item in row) for row in ulps.max(axis=(1, 3))
        ),
    )


def format_sweep_result(result: SweepResult) -> str:
    """Render the sweep verdict and its residual diagnostics."""
    violations = ", ".join(result.violations) if result.violations else "none"
    coordinate = (
        "n/a"
        if result.worst_coordinate is None
        else ":".join(str(item) for item in result.worst_coordinate)
    )
    return (
        f"violations={violations}; rows={result.compared_rows}; "
        f"max_abs={result.max_abs:.9g} ({result.max_ceiling_ratio:.3g} of the "
        f"ceiling) at b:q:h:d={coordinate}; max_ulp={result.max_ulp}; "
        f"over_one_ulp={result.over_one_ulp_count}; "
        f"exact_hits={result.exact_hit_rate:.6f}; "
        f"max_over_value_range={result.max_normalized:.9g}"
    )


@dataclass(frozen=True)
class ControlDesignation:
    """Where a control must be visible, decided before any launch.

    Per row and query head ``[B, Sq, Hq]``: ``separation`` is the distance, on
    the control's coordinate, between the average of V over the keys at the
    row's highest logit and its average over all legal keys, rounded toward
    zero; ``designated`` marks the pairs whose separation exceeds four forward
    errors plus the weight the lower keys can still carry."""

    control: Control
    coordinate: int
    designated: np.ndarray
    separation: np.ndarray
    high_count: np.ndarray
    legal_count: np.ndarray
    forward_error: tuple[float, ...]

    @property
    def head_counts(self) -> tuple[int, ...]:
        """Designated rows per query head."""
        return tuple(
            int(item) for item in np.count_nonzero(self.designated, axis=(0, 1))
        )


def designate_control(
    case: KeyTranslationCase,
    control: Control,
    levels: np.ndarray,
    v: np.ndarray,
    forward_error: float | np.ndarray,
    legal: np.ndarray | None = None,
) -> ControlDesignation:
    """Decide which rows of a control must separate from the baseline.

    ``levels`` are the control's raw scores ``[B, Sq, Sk]`` in eighths of the
    gap and ``forward_error`` a bound, per query head or common to all, on the
    distance of a stored output from the ideal output of its own computed
    logits, derived from the subject's arithmetic path. A row with all or none
    of its legal keys at the highest level is never designated. The lower keys
    sit at least the natural-unit gap ``G`` below, so they hold at most
    ``(N - n) exp(-G) / n`` of the weight; that tail and the forward errors are
    rounded up and the separation down. Raises ``ValueError`` when a query head
    is left without a designated row."""
    selected = Control(control)
    counts = np.asarray(levels)
    if counts.ndim != 3 or counts.dtype.kind not in "iu":
        raise ValueError("levels must be a [B, Sq, Sk] integer array")
    if counts.shape != (case.batch, case.seqlen_q, case.seqlen_k):
        raise ValueError("levels do not match the case shape")
    data = np.asarray(v, dtype=np.float64)
    if data.shape != case.key_shape or not np.all(np.isfinite(data)):
        raise ValueError(f"V must be a finite array of shape {case.key_shape}")
    errors = np.broadcast_to(
        np.asarray(forward_error, dtype=np.float64), (case.nhead_q,)
    )
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
        raise ValueError("the forward error must be positive and finite")
    mask = _legal(counts, legal)

    _, top = _row_extremes(counts, mask)
    high = mask & (counts == top[:, :, None])
    below = mask & ~high
    high_count = high.sum(axis=-1)
    legal_count = mask.sum(axis=-1)
    eligible = (high_count > 0) & (high_count < legal_count)
    big = np.iinfo(np.int64).max
    runner_up = np.where(below, counts, -big).max(axis=-1)
    level_gap = top - np.where(eligible, runner_up, top)

    coordinate = selected.coordinate
    column = data[:, :, :, coordinate]
    kv_head = np.arange(case.nhead_q) // case.group
    high_sum = np.einsum("bqk,bkj->bqj", high.astype(np.float64), column)
    legal_sum = np.einsum("bqk,bkj->bqj", mask.astype(np.float64), column)
    separation = np.abs(
        high_sum / high_count[:, :, None] - legal_sum / legal_count[:, :, None]
    )
    separation = np.nextafter(separation, 0.0)[:, :, kv_head]

    gaps = np.array([float(gap) for gap in score_gaps(case)])
    natural_gap = level_gap[:, :, None] * gaps[None, None, :] / 64.0
    tail = np.nextafter(
        (legal_count - high_count)[:, :, None]
        * np.exp(-natural_gap)
        / high_count[:, :, None],
        np.inf,
    )
    spread = value_ranges(case, data)[:, coordinate]
    required = np.nextafter(
        4.0 * errors[None, None, :] + tail * spread[None, None, :], np.inf
    )
    designated = eligible[:, :, None] & (separation > required)
    result = ControlDesignation(
        control=selected,
        coordinate=coordinate,
        designated=designated,
        separation=separation,
        high_count=high_count,
        legal_count=legal_count,
        forward_error=tuple(float(item) for item in errors),
    )
    missing = [head for head, count in enumerate(result.head_counts) if count == 0]
    if missing:
        raise ValueError(
            f"control {selected.value}: query heads {missing} have no designated row"
        )
    return result


@dataclass(frozen=True)
class ControlResult:
    """Verdict of one control launch against its baseline.

    The control passes when every designated row and head separates from the
    baseline, on the control's coordinate, by more than two forward errors."""

    violations: tuple[str, ...]
    invalid_count: int
    designated_count: int
    unseparated_count: int
    head_unseparated_counts: tuple[int, ...]
    min_separation: float
    first_unseparated: tuple[int, int, int] | None


def analyze_control(
    case: KeyTranslationCase,
    base: np.ndarray,
    other: np.ndarray,
    designation: ControlDesignation,
) -> ControlResult:
    """Require a control's output to leave the baseline's on every designated
    row and head.

    A control that stays within two forward errors of the baseline somewhere
    means the subject ignored a token-dependent change of its keys there."""
    base_raw, other_raw, base_finite, other_finite = _raw_outputs(case, base, other)
    if designation.designated.shape != (case.batch, case.seqlen_q, case.nhead_q):
        raise ValueError("the designation does not match the case shape")
    invalid_count = int(np.count_nonzero(~base_finite)) + int(
        np.count_nonzero(~other_finite)
    )
    coordinate = designation.coordinate
    both = base_finite[..., coordinate] & other_finite[..., coordinate]
    difference = np.abs(
        np.where(both, decode_bits(base_raw[..., coordinate], case.dtype), 0.0)
        - np.where(both, decode_bits(other_raw[..., coordinate], case.dtype), 0.0)
    )
    threshold = 2.0 * np.array(designation.forward_error)[None, None, :]
    unseparated = designation.designated & ~(both & (difference > threshold))
    unseparated_count = int(np.count_nonzero(unseparated))
    designated_count = int(np.count_nonzero(designation.designated))
    head_counts = tuple(
        int(item) for item in np.count_nonzero(unseparated, axis=(0, 1))
    )

    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if unseparated_count:
        heads = [head for head, count in enumerate(head_counts) if count]
        violations.append(f"unseparated={unseparated_count} in query heads {heads}")
    if designated_count == 0:
        violations.append("no_pair_designated")
    return ControlResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        designated_count=designated_count,
        unseparated_count=unseparated_count,
        head_unseparated_counts=head_counts,
        min_separation=(
            float(difference[designation.designated].min()) if designated_count else 0.0
        ),
        first_unseparated=_first(unseparated),
    )


def format_control_result(result: ControlResult) -> str:
    """Render the control verdict."""
    violations = ", ".join(result.violations) if result.violations else "none"
    first = (
        "n/a"
        if result.first_unseparated is None
        else ":".join(str(item) for item in result.first_unseparated)
    )
    return (
        f"violations={violations}; designated={result.designated_count}; "
        f"unseparated={result.unseparated_count} (first at b:q:h={first}); "
        f"min_separation={result.min_separation:.9g}"
    )
