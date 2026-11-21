# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free fixture, legal-set arithmetic, derived limits and analyzer for the
masked-token poisoning relation.

A row of masked attention depends only on its legal keys and values, so changing
K and V at tokens that are masked for a row must leave that row unchanged. The
relation compares two launches of the subject: a baseline, and one whose masked
tokens carry a large key and a large value. Rows that cannot see a poisoned
token must come back bit for bit; rows that can see one must be taken over by
it, which shows the poison was delivered. No expected output is computed and
there is no attention implementation in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy as np

from ._analytical import (
    _FORMAT,
    DenseCaseShape,
    _legal_bounds,
    _ordered_array,
    decode_bits,
)
from ._permutation import _float_not_below, _signs

_MASKS = frozenset({"causal", "window"})

# Poisoned keys are POISON_KEY_SCALE times the anchor key and poisoned values
# POISON_VALUE_SCALE times a sign row. Both are powers of two, so every poisoned
# entry is exact in FP16 and BF16.
POISON_KEY_SCALE = 16
POISON_VALUE_SCALE = 16
# A row that sees a poisoned token is an activity row when its integer score
# against the anchor key is at least this.
ACTIVITY_SCORE = 8
# An activity row must store at least this, signed by the poison's sign row.
ACTIVITY_THRESHOLD = 8.0
# Anchor key positions: one whose target rows straddle index 128, one interior.
POISON_ANCHORS = (127, 200)

# Largest magnitude of a baseline value entry.
_VALUE_MAX = 1
_SEED_K = 0x6B
_SEED_V = 0x76
_SEED_W = 0x77
# Seeds tried for a key tensor without two equal key tokens.
_KEY_SEED_ATTEMPTS = 8

# The kernel's score scale is the FP32 product of 1/sqrt(D) and FP32 log2(e).
_LOG2E = 1.4426950408889634
# One FP32 unit of faithful rounding.
_FP32_UNIT = Fraction(1, 2**23)
# Covers every absolute FP32 underflow term of one row.
_UNDERFLOW = Fraction(1, 2**100)
# Sequence lengths and magnitudes the limits are derived for.
_MAX_SEQLEN = 1 << 16
# exp2 returns +0 at or below -ZERO_GAP and overflows at or above
# OVERFLOW_GAP, in score units (base-2 logarithm), for both lowerings.
ZERO_GAP = 150
OVERFLOW_GAP = 128
# sqrt(2) rounded up at the seventh decimal.
_ROOT2_UPPER = Fraction(14142136, 10**7)


@dataclass(frozen=True)
class MaskedPoisonCase(DenseCaseShape):
    """Logical masked dense-attention shape for one masked-token poisoning case."""

    mask: str
    window: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mask not in _MASKS:
            raise ValueError(f"mask must be one of {sorted(_MASKS)}")
        if self.mask == "window":
            if self.window <= 0:
                raise ValueError("window mask requires window > 0")
        elif self.window:
            raise ValueError("window must be zero unless mask='window'")

    @property
    def group(self) -> int:
        """Query heads per KV head."""
        return self.nhead_q // self.nhead_k

    @property
    def case_id(self) -> str:
        """Return a stable identifier for test ids and resolved-subject checks."""
        return self._shape_id(
            f"window{self.window}" if self.mask == "window" else self.mask
        )

    def require_scope(self) -> None:
        """Refuse a case the fixture or the derived limits do not cover.

        The fixture is self-attention with one planted key per query head of a
        group, so a window must hold a whole group. The score scale must be
        exact, which needs ``head_dim`` to be a power of four. Every anchor must
        place its poisoned tokens inside the key range, and the activity
        threshold must separate the baseline ceiling from the takeover floor."""
        if self.seqlen_q != self.seqlen_k:
            raise ValueError(f"case {self.case_id} is not self-attention")
        if self.seqlen_k > _MAX_SEQLEN:
            raise ValueError(
                f"case {self.case_id}: the limits are derived for at most "
                f"{_MAX_SEQLEN} keys"
            )
        if self.head_dim & (self.head_dim - 1) or self.head_dim.bit_length() % 2 == 0:
            raise ValueError(
                f"case {self.case_id}: head_dim must be a power of four for an "
                "exact score scale"
            )
        if self.mask == "window" and self.window < self.group:
            raise ValueError(
                f"case {self.case_id}: a window of {self.window} keys cannot hold "
                f"the {self.group} planted keys of one head group"
            )
        for anchor in POISON_ANCHORS:
            poison_positions(self, anchor)
        masked_poison_limits(self)


# GQA (four query heads per KV head) at the shape of the uniform-probability
# cases. The quick tier runs one mask per dtype; the standard tier runs the
# other pairing, whose BF16 window case resolves to a different tile and loop.
POISON_WINDOW_FP16 = MaskedPoisonCase(1, 256, 256, 8, 2, 64, "fp16", "window", 64)
POISON_CAUSAL_BF16 = MaskedPoisonCase(1, 256, 256, 8, 2, 64, "bf16", "causal")
POISON_CAUSAL_FP16 = MaskedPoisonCase(1, 256, 256, 8, 2, 64, "fp16", "causal")
POISON_WINDOW_BF16 = MaskedPoisonCase(1, 256, 256, 8, 2, 64, "bf16", "window", 64)
# The GPU lanes' cases. A CPU contract pins each lane's parametrization to them,
# so a case cannot silently stop executing.
MASKED_POISON_QUICK_CASES = (POISON_WINDOW_FP16, POISON_CAUSAL_BF16)
MASKED_POISON_STANDARD_CASES = (POISON_CAUSAL_FP16, POISON_WINDOW_BF16)


class BoundaryControl(str, Enum):
    """Poison placements one token inside the legal set of a target row."""

    FUTURE_KEY = "future_key"
    WINDOW_KEY = "window_key"


def poison_positions(case: MaskedPoisonCase, anchor: int) -> tuple[int, ...]:
    """Return the poisoned key tokens of one anchor.

    The query rows ``anchor .. anchor + G - 1`` are the target rows: head offset
    ``d`` of a group copies the anchor key at row ``anchor + d``. Causal: the
    first key past the last target row, and the last key. Window: the key just
    below the first target row's window, and the first key past the last target
    row."""
    group = case.group
    if not 0 <= anchor or anchor + group > case.seqlen_q:
        raise ValueError(f"anchor {anchor} leaves no room for {group} target rows")
    if case.mask == "causal":
        positions = (anchor + group, case.seqlen_k - 1)
    else:
        positions = (anchor - case.window, anchor + group)
    if min(positions) < 0 or max(positions) >= case.seqlen_k:
        raise ValueError(
            f"anchor {anchor}: poisoned tokens {positions} leave the key range"
        )
    if positions[0] >= positions[1]:
        raise ValueError(f"anchor {anchor}: poisoned tokens {positions} coincide")
    return positions


def control_positions(
    case: MaskedPoisonCase, anchor: int, control: BoundaryControl | str
) -> tuple[int, ...]:
    """Return one boundary control's poisoned tokens: the tokens of
    :func:`poison_positions` with one of them moved a single token into the
    legal set of a target row."""
    try:
        selected = BoundaryControl(control)
    except ValueError as error:
        raise ValueError(f"unknown boundary control {control!r}") from error
    first, second = poison_positions(case, anchor)
    if selected is BoundaryControl.FUTURE_KEY:
        future = anchor + case.group - 1
        return (future, second) if case.mask == "causal" else (first, future)
    if case.mask != "window":
        raise ValueError(f"control {selected.value} requires a window case")
    return (first + 1, second)


def control_witness(
    case: MaskedPoisonCase, anchor: int, control: BoundaryControl | str
) -> tuple[int, int]:
    """Return the ``(query row, head offset)`` that must respond to a control.

    That target row copies the anchor key, so the token the control moved into
    its legal set has the largest possible score there."""
    selected = BoundaryControl(control)
    control_positions(case, anchor, selected)
    if selected is BoundaryControl.FUTURE_KEY:
        return anchor + case.group - 1, case.group - 1
    return anchor, 0


def _legal_rows(case: MaskedPoisonCase) -> tuple[np.ndarray, np.ndarray]:
    """Per query row, the first and last legal key of the case's mask rule."""
    first, last = _legal_bounds(case, None)
    if np.any(last < first):
        raise ValueError(f"case {case.case_id} has a row without a legal key")
    return first, last


def sees_token(case: MaskedPoisonCase, positions: tuple[int, ...]) -> np.ndarray:
    """Per query row, whether any of ``positions`` is a legal key."""
    first, last = _legal_rows(case)
    seen = np.zeros(case.seqlen_q, dtype=bool)
    for position in positions:
        seen |= (first <= position) & (position <= last)
    return seen


@dataclass(frozen=True, eq=False)
class PoisonBaseline:
    """Baseline inputs of one case, in the runner layout, and the key seed that
    produced pairwise distinct key tokens."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    key_seed: int


def _distinct_key_tokens(k: np.ndarray) -> bool:
    """Whether every ``(batch, KV head)`` holds pairwise distinct key tokens."""
    batch, seqlen_k, heads, _ = k.shape
    for b in range(batch):
        for head in range(heads):
            packed = np.packbits(k[b, :, head, :] > 0, axis=1)
            if np.unique(packed, axis=0).shape[0] != seqlen_k:
                return False
    return True


def masked_poison_inputs(case: MaskedPoisonCase) -> PoisonBaseline:
    """Build sign-valued K and V and queries that each copy one legal key.

    Query head ``h`` of row ``q`` copies key ``max(q - h mod G, 0)`` of its KV
    head. Key tokens are pairwise distinct, so that copy scores ``D`` and every
    other key at most ``D - 2``: each row has a known strict maximum and
    nonuniform logits. Every element is ``+1`` or ``-1``, exact in both dtypes."""
    case.require_scope()
    for attempt in range(_KEY_SEED_ATTEMPTS):
        key_seed = _SEED_K + attempt
        k = _signs(case.key_shape, key_seed)
        if _distinct_key_tokens(k):
            break
    else:
        raise ValueError(
            f"case {case.case_id}: no key tensor with pairwise distinct tokens "
            f"in {_KEY_SEED_ATTEMPTS} seeds"
        )
    v = _signs(case.key_shape, _SEED_V)

    first, last = _legal_rows(case)
    group = case.group
    q = np.empty(case.output_shape, dtype=np.float32)
    rows = np.arange(case.seqlen_q)
    for head in range(case.nhead_q):
        planted = np.maximum(rows - head % group, 0)
        if np.any(planted < first) or np.any(planted > last):
            raise ValueError(
                f"case {case.case_id}: a planted key of head {head} is not legal"
            )
        q[:, :, head, :] = k[:, planted, head // group, :]
    return PoisonBaseline(q=q, k=k, v=v, key_seed=key_seed)


def poison_sign_row(case: MaskedPoisonCase) -> np.ndarray:
    """The ``[B, Hkv, D]`` sign row every poisoned value is a multiple of."""
    return _signs((case.batch, case.nhead_k, case.head_dim), _SEED_W)


def poisoned_tensors(
    case: MaskedPoisonCase,
    baseline: PoisonBaseline,
    anchor: int,
    positions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return K and V with the tokens at ``positions`` replaced: the key by
    ``POISON_KEY_SCALE`` times the anchor key, the value by
    ``POISON_VALUE_SCALE`` times the sign row. Every other element is copied."""
    if not 0 <= anchor < case.seqlen_k:
        raise IndexError(anchor)
    k = baseline.k.copy()
    v = baseline.v.copy()
    sign_row = poison_sign_row(case)
    for position in positions:
        if not 0 <= position < case.seqlen_k:
            raise IndexError(position)
        k[:, position] = np.float32(POISON_KEY_SCALE) * baseline.k[:, anchor]
        v[:, position] = np.float32(POISON_VALUE_SCALE) * sign_row
    return k, v


@dataclass(frozen=True, eq=False)
class PoisonFixture:
    """One anchor's poisoned inputs and the row sets its verdicts are taken over.

    ``unaffected`` marks the query rows that see no poisoned token. ``targets``
    holds, per query head, the unaffected row that copies the anchor key.
    ``activity`` marks the ``(batch, row, head)`` triples that see a poisoned
    token and score at least ``ACTIVITY_SCORE`` against the anchor key;
    ``anchor_scores`` are those exact integer scores."""

    case: MaskedPoisonCase
    anchor: int
    positions: tuple[int, ...]
    k: np.ndarray
    v: np.ndarray
    sign_row: np.ndarray
    unaffected: np.ndarray
    targets: np.ndarray
    anchor_scores: np.ndarray
    activity: np.ndarray


def poison_fixture(
    case: MaskedPoisonCase, baseline: PoisonBaseline, anchor: int
) -> PoisonFixture:
    """Build one anchor's poisoned inputs and row sets, by integer arithmetic.

    The row sets follow from the mask rule and from the exact integer scores of
    the launched signs against the anchor key. Fails closed: a target row that
    sees a poisoned token, a target row that does not copy the anchor key, or a
    ``(batch, head)`` without an activity row rejects the fixture."""
    positions = poison_positions(case, anchor)
    k, v = poisoned_tensors(case, baseline, anchor, positions)
    unaffected = ~sees_token(case, positions)

    group = case.group
    heads = np.arange(case.nhead_q)
    routed = heads // group
    targets = anchor + heads % group
    if not np.all(unaffected[targets]):
        raise ValueError(f"anchor {anchor}: a target row sees a poisoned token")

    anchor_key = baseline.k[:, anchor][:, routed].astype(np.int64)
    scores = np.einsum("bqhd,bhd->bqh", baseline.q.astype(np.int64), anchor_key)
    if np.any(scores[:, targets, heads] != case.head_dim):
        raise ValueError(f"anchor {anchor}: a target row does not copy the anchor key")
    activity = ~unaffected[None, :, None] & (scores >= ACTIVITY_SCORE)
    if not np.all(activity.any(axis=1)):
        raise ValueError(
            f"anchor {anchor}: a (batch, head) has no activity row; "
            f"counts {activity.sum(axis=1).tolist()}"
        )
    return PoisonFixture(
        case=case,
        anchor=anchor,
        positions=positions,
        k=k,
        v=v,
        sign_row=poison_sign_row(case),
        unaffected=unaffected,
        targets=targets,
        anchor_scores=scores,
        activity=activity,
    )


class MaskMechanism(str, Enum):
    """How the kernel keeps one masked token out of one query row."""

    UNVISITED_TILE = "unvisited_tile"
    TILE_BEFORE_FIRST_LEGAL_KEY = "tile_before_first_legal_key"
    MASKED_LANE = "masked_lane"


def poison_mechanisms(
    fixture: PoisonFixture,
    *,
    block_n: int,
    tile_low: np.ndarray,
    tile_high: np.ndarray,
) -> dict[tuple[int, int], MaskMechanism]:
    """Per ``(target row, poisoned token)``, the mechanism that isolates it.

    ``tile_low``/``tile_high`` are, per query row, the first and one past the
    last key tile the launch visits, read from the kernel under test. A token in
    a tile the row never visits is not read at all. A token in a visited tile
    that lies wholly before the row's first legal key is accumulated with the
    row's other masked keys and removed by the first rescale. Any other token
    sits in a lane whose score the mask replaces under a legal maximum.

    A record of what a launch exercises, not a verdict."""
    first, _ = _legal_rows(fixture.case)
    found: dict[tuple[int, int], MaskMechanism] = {}
    for row in sorted({int(item) for item in fixture.targets}):
        for position in fixture.positions:
            tile = position // block_n
            if not tile_low[row] <= tile < tile_high[row]:
                mechanism = MaskMechanism.UNVISITED_TILE
            elif tile < int(first[row]) // block_n:
                mechanism = MaskMechanism.TILE_BEFORE_FIRST_LEGAL_KEY
            else:
                mechanism = MaskMechanism.MASKED_LANE
            found[(row, position)] = mechanism
    return found


def poisoned_bit_violations(
    fixture: PoisonFixture,
    k_bits: np.ndarray,
    v_bits: np.ndarray,
    poisoned_k_bits: np.ndarray,
    poisoned_v_bits: np.ndarray,
) -> tuple[str, ...]:
    """Check the encoded inputs of a launch pair, on the bits the kernel reads.

    The poisoned K and V must equal the baseline bit for bit at every token
    outside ``fixture.positions``, and at those tokens hold exactly the intended
    finite values. Anything else invalidates the pair before a launch."""
    case = fixture.case
    violations: list[str] = []
    kept = np.ones(case.seqlen_k, dtype=bool)
    kept[list(fixture.positions)] = False
    for name, base, poisoned in (
        ("K", k_bits, poisoned_k_bits),
        ("V", v_bits, poisoned_v_bits),
    ):
        if base.shape != case.key_shape or poisoned.shape != case.key_shape:
            violations.append(f"{name} shape differs from {case.key_shape}")
            continue
        changed = int(np.count_nonzero(base[:, kept] != poisoned[:, kept]))
        if changed:
            violations.append(f"{changed} {name} element(s) changed outside the poison")
    if violations:
        return tuple(violations)

    anchor_key = decode_bits(k_bits[:, fixture.anchor], case.dtype)
    for position in fixture.positions:
        key = decode_bits(poisoned_k_bits[:, position], case.dtype)
        value = decode_bits(poisoned_v_bits[:, position], case.dtype)
        if not np.array_equal(key, POISON_KEY_SCALE * anchor_key):
            violations.append(f"poisoned key {position} is not the scaled anchor key")
        if not np.array_equal(
            value, POISON_VALUE_SCALE * fixture.sign_row.astype(np.float64)
        ):
            violations.append(f"poisoned value {position} is not the scaled sign row")
    return tuple(violations)


def _float_not_above(value: Fraction) -> float:
    """The float nearest ``value``, moved down one step if it lies above it."""
    nearest = float(value)
    if Fraction(nearest) <= value:
        return nearest
    return math.nextafter(nearest, -math.inf)


def _relative(steps: int) -> Fraction:
    """``gamma_n = n u / (1 - n u)`` for ``n`` faithful FP32 operations."""
    scaled = steps * _FP32_UNIT
    if scaled >= 1:
        raise ValueError("too many FP32 operations for a relative bound")
    return scaled / (1 - scaled)


def _computed_gap(case: MaskedPoisonCase, poison_score: int) -> Fraction:
    """Lower bound on a poisoned key's computed score minus the largest computed
    score of an unpoisoned legal key, in score units.

    ``poison_score`` is the poisoned key's exact integer score; an unpoisoned
    key scores at most ``D``. A score is ``D / 8`` chained MFMA calls, each
    within ``gamma_8`` of the summed magnitudes of its operands, then one FP32
    product with the exact scale ``log2(e) / sqrt(D)``."""
    unit = _FP32_UNIT
    dim = case.head_dim
    chain = _relative(dim)
    scale = Fraction(float(np.float32(_LOG2E))) / math.isqrt(dim)
    poison = (poison_score - chain * POISON_KEY_SCALE * dim) * (1 - unit)
    legal = dim * (1 + chain) * (1 + unit)
    return scale * (poison - legal)


def _power_of_two_upper(half_steps: int) -> Fraction:
    """A rational not below ``2 ** (-half_steps / 2)``."""
    whole, odd = divmod(half_steps, 2)
    if odd:
        return _ROOT2_UPPER / 2 ** (whole + 1)
    return Fraction(1, 2**whole)


@dataclass(frozen=True)
class PoisonLimits:
    """Derived limits of the relation's sanity and activity verdicts.

    ``baseline_ceiling`` is not below any stored baseline magnitude of a correct
    kernel; ``activity_floor`` is not above the stored signed output of a
    correct kernel on an activity row. ``threshold`` lies strictly between
    them. ``target_gap`` bounds from below the computed score gap between a
    poisoned key and the legal maximum of a target row, ``takeover_gap`` the
    same on an activity row; both in score units."""

    dtype: str
    legal_keys: int
    baseline_ceiling: float
    activity_floor: float
    threshold: float
    target_gap: float
    takeover_gap: float


def masked_poison_limits(case: MaskedPoisonCase) -> PoisonLimits:
    """Evaluate the limits for one case, in exact rational arithmetic.

    With ``u = 2**-23``, ``c = 2**-(p+1)`` and ``z`` half the smallest subnormal
    of a dtype with ``p`` fraction bits, ``N`` the largest legal count of a row,
    ``g = gamma(2 Sk)`` and ``w = 2**-100``:

    - baseline: a stored output is a rounded, nearly convex combination of the
      legal values, ``|O| <= v_max H + z + 8 w`` with
      ``H = (1+u)**2 (1+c) (1+g) (1 + c + 2 N z) / (1 - g - 2 w)``;
    - activity: a legal poisoned key has weight exactly one, every other legal
      key at most ``pbar = 2**-gap (1+u)**(Sk+1)``, so the signed stored output
      is at least ``(1-c)(1-u)**2 (mu (1-g) - v_max b (1+g) - w) /
      ((1+g)(1 + (N-1) pbar) + w) - z`` with ``b = (N-1)((1+c) pbar + 2 z)``.

    Neither depends on an observed output. The ceiling is rounded up and the
    floor down. Refuses a case whose activity threshold does not separate them,
    or whose target gap does not reach the exponent range of the weight path."""
    fraction_bits, _, bias = _FORMAT[case.dtype]
    first, last = _legal_rows(case)
    legal_keys = int(np.max(last - first + 1))
    unit = _FP32_UNIT
    cast = Fraction(1, 2 ** (fraction_bits + 1))
    subnormal_half = Fraction(1, 2 ** (bias - 1 + fraction_bits + 1))
    accumulation = _relative(2 * case.seqlen_k)
    underflow = _UNDERFLOW

    target_gap = _computed_gap(case, POISON_KEY_SCALE * case.head_dim)
    if target_gap < max(ZERO_GAP, OVERFLOW_GAP):
        raise ValueError(
            f"case {case.case_id}: target gap {float(target_gap):.6g} does not "
            f"reach {max(ZERO_GAP, OVERFLOW_GAP)} score units"
        )
    takeover = _computed_gap(case, POISON_KEY_SCALE * ACTIVITY_SCORE)
    half_steps = math.floor(2 * takeover)
    if not 0 < half_steps <= 2 * 126:
        raise ValueError(
            f"case {case.case_id}: takeover gap {float(takeover):.6g} is outside "
            "(0, 126] score units"
        )

    hull = (
        (1 + unit) ** 2
        * (1 + cast)
        * (1 + accumulation)
        * (1 + cast + 2 * legal_keys * subnormal_half)
        / (1 - accumulation - 2 * underflow)
    )
    ceiling = _VALUE_MAX * hull + subnormal_half + 8 * underflow

    weight = _power_of_two_upper(half_steps) * (1 + unit) ** (case.seqlen_k + 1)
    others = legal_keys - 1
    leak = others * ((1 + cast) * weight + 2 * subnormal_half)
    floor = (1 - cast) * (1 - unit) ** 2 * (
        POISON_VALUE_SCALE * (1 - accumulation)
        - _VALUE_MAX * leak * (1 + accumulation)
        - underflow
    ) / ((1 + accumulation) * (1 + others * weight) + underflow) - subnormal_half

    threshold = Fraction(ACTIVITY_THRESHOLD)
    if not ceiling < threshold < floor:
        raise ValueError(
            f"case {case.case_id}: activity threshold {ACTIVITY_THRESHOLD} does "
            f"not separate the baseline ceiling {float(ceiling):.6g} from the "
            f"takeover floor {float(floor):.6g}"
        )
    return PoisonLimits(
        dtype=case.dtype,
        legal_keys=legal_keys,
        baseline_ceiling=_float_not_below(ceiling),
        activity_floor=_float_not_above(floor),
        threshold=ACTIVITY_THRESHOLD,
        target_gap=_float_not_above(target_gap),
        takeover_gap=half_steps / 2,
    )


class LeakClass(str, Enum):
    """What a target row that changed looks like in the poisoned output.

    A diagnostic label read from the stored bits; it asserts no mechanism."""

    NONFINITE = "nonfinite"
    ZERO = "zero"
    POISON_DOMINATED = "poison_dominated"
    OTHER = "other"


@dataclass(frozen=True, eq=False)
class PoisonResult:
    """The four verdicts of one launch pair, kept apart, plus diagnostics.

    ``validity`` holds execution findings that invalidate the pair (a baseline
    that does not repeat); input findings are raised before a launch.
    ``sanity`` holds non-finite outputs and baseline magnitudes beyond the
    derived ceiling. ``relation`` holds unaffected rows that changed.
    ``activity`` holds activity rows the poison did not take over.

    ``mismatched`` marks the ``(batch, row, head)`` triples of unaffected rows
    with a changed coordinate; ``dominated`` marks the triples of any row whose
    poisoned output reaches the activity threshold in every coordinate."""

    validity: tuple[str, ...]
    sanity: tuple[str, ...]
    relation: tuple[str, ...]
    activity: tuple[str, ...]
    replay_mismatch_count: int
    invalid_count: int
    ceiling_count: int
    max_baseline_abs: float
    zero_target_count: int
    compared_count: int
    mismatch_count: int
    zero_sign_only_count: int
    max_ulp: int
    mismatched_rows: tuple[int, ...]
    activity_row_count: int
    activity_failure_count: int
    min_activity_signed: float
    leak_classes: tuple[tuple[tuple[int, int, int], LeakClass], ...]
    mismatched: np.ndarray
    dominated: np.ndarray

    @property
    def violations(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The non-empty verdict classes, in order, each under its own name."""
        named = (
            ("validity", self.validity),
            ("sanity", self.sanity),
            ("relation", self.relation),
            ("activity", self.activity),
        )
        return tuple((name, found) for name, found in named if found)


def _raw_output(case: MaskedPoisonCase, name: str, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.uint16:
        raise ValueError(f"{name} output must contain raw uint16 bits")
    if raw.shape != case.output_shape:
        raise ValueError(f"{name} output shape {raw.shape} != {case.output_shape}")
    return raw


def analyze_masked_poison(
    fixture: PoisonFixture,
    limits: PoisonLimits,
    base: np.ndarray,
    replay: np.ndarray,
    poisoned: np.ndarray,
) -> PoisonResult:
    """Judge one baseline, its replay and one poisoned output, on raw bits.

    Relation: every coordinate of every unaffected row must hold the same bits
    in ``base`` and ``poisoned``; ``+0`` and ``-0`` are different. Activity:
    every coordinate of every activity row, multiplied by the poison's sign,
    must reach the threshold. Sanity: every stored value is finite (an unwritten
    coordinate keeps its NaN prefill) and no baseline magnitude exceeds the
    derived ceiling. Validity: the baseline repeats bit for bit."""
    case = fixture.case
    if limits.dtype != case.dtype:
        raise ValueError(f"limits dtype {limits.dtype} != case dtype {case.dtype}")
    base_raw = _raw_output(case, "base", base)
    replay_raw = _raw_output(case, "replay", replay)
    poisoned_raw = _raw_output(case, "poisoned", poisoned)

    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_mask = np.uint16(((1 << exponent_bits) - 1) << fraction_bits)
    magnitude_mask = np.uint16(0x7FFF)
    base_finite = (base_raw & exponent_mask) != exponent_mask
    poisoned_finite = (poisoned_raw & exponent_mask) != exponent_mask
    invalid_count = int(np.count_nonzero(~base_finite)) + int(
        np.count_nonzero(~poisoned_finite)
    )
    base_values = np.where(base_finite, decode_bits(base_raw, case.dtype), 0.0)
    poisoned_values = np.where(
        poisoned_finite, decode_bits(poisoned_raw, case.dtype), 0.0
    )

    replay_mismatch_count = int(np.count_nonzero(base_raw != replay_raw))
    max_baseline_abs = float(np.max(np.abs(base_values)))
    ceiling_count = int(np.count_nonzero(np.abs(base_values) > limits.baseline_ceiling))

    heads = np.arange(case.nhead_q)
    target_rows = base_raw[:, fixture.targets, heads, :]
    zero_target_count = int(
        np.count_nonzero(~np.any(target_rows & magnitude_mask, axis=-1))
    )

    # Relation, on raw bits.
    unaffected = fixture.unaffected[None, :, None, None]
    differs = (base_raw != poisoned_raw) & unaffected
    compared_count = int(np.count_nonzero(fixture.unaffected)) * (
        case.batch * case.nhead_q * case.head_dim
    )
    mismatch_count = int(np.count_nonzero(differs))
    both_zero = ((base_raw | poisoned_raw) & magnitude_mask) == 0
    zero_sign_only_count = int(np.count_nonzero(differs & both_zero))
    distances = np.abs(
        _ordered_array(base_raw).astype(np.int64)
        - _ordered_array(poisoned_raw).astype(np.int64)
    )
    measurable = differs & base_finite & poisoned_finite
    max_ulp = int(np.max(np.where(measurable, distances, 0)))
    mismatched = differs.any(axis=3)
    mismatched_rows = tuple(int(row) for row in np.flatnonzero(mismatched.any((0, 2))))

    # Activity, signed by the sign row of each query head's KV head.
    signs = fixture.sign_row[:, heads // case.group, :].astype(np.float64)
    signed = poisoned_values * signs[:, None, :, :]
    reaches = poisoned_finite & (signed >= limits.threshold)
    dominated = reaches.all(axis=3)
    activity = fixture.activity
    activity_row_count = int(np.count_nonzero(activity))
    activity_failure_count = int(np.count_nonzero(activity & ~dominated))
    on_activity = activity[:, :, :, None] & poisoned_finite
    min_activity_signed = (
        float(np.min(np.where(on_activity, signed, np.inf)))
        if np.any(on_activity)
        else math.inf
    )

    leak_classes: list[tuple[tuple[int, int, int], LeakClass]] = []
    for head in range(case.nhead_q):
        row = int(fixture.targets[head])
        for batch in range(case.batch):
            if not mismatched[batch, row, head]:
                continue
            if not np.all(poisoned_finite[batch, row, head]):
                label = LeakClass.NONFINITE
            elif not np.any(poisoned_raw[batch, row, head] & magnitude_mask):
                label = LeakClass.ZERO
            elif dominated[batch, row, head]:
                label = LeakClass.POISON_DOMINATED
            else:
                label = LeakClass.OTHER
            leak_classes.append(((batch, row, head), label))

    validity: list[str] = []
    if replay_mismatch_count:
        validity.append(f"baseline_replay_differs={replay_mismatch_count}")
    sanity: list[str] = []
    if invalid_count:
        sanity.append(f"invalid={invalid_count}")
    if ceiling_count:
        sanity.append(
            f"baseline_abs={max_baseline_abs:.9g}>{limits.baseline_ceiling:.9g} "
            f"at {ceiling_count}"
        )
    relation: list[str] = []
    if mismatch_count:
        relation.append(
            f"unaffected_changed={mismatch_count} of {compared_count} "
            f"(zero sign only {zero_sign_only_count}, max {max_ulp} ULP) "
            f"in rows {list(mismatched_rows)}"
        )
    if compared_count == 0:
        relation.append("no_unaffected_coordinate")
    activity_violations: list[str] = []
    if activity_failure_count:
        activity_violations.append(
            f"not_taken_over={activity_failure_count} of {activity_row_count} "
            f"rows, min signed {min_activity_signed:.9g}<{limits.threshold:.9g}"
        )
    if activity_row_count == 0:
        activity_violations.append("no_activity_row")

    return PoisonResult(
        validity=tuple(validity),
        sanity=tuple(sanity),
        relation=tuple(relation),
        activity=tuple(activity_violations),
        replay_mismatch_count=replay_mismatch_count,
        invalid_count=invalid_count,
        ceiling_count=ceiling_count,
        max_baseline_abs=max_baseline_abs,
        zero_target_count=zero_target_count,
        compared_count=compared_count,
        mismatch_count=mismatch_count,
        zero_sign_only_count=zero_sign_only_count,
        max_ulp=max_ulp,
        mismatched_rows=mismatched_rows,
        activity_row_count=activity_row_count,
        activity_failure_count=activity_failure_count,
        min_activity_signed=min_activity_signed,
        leak_classes=tuple(leak_classes),
        mismatched=mismatched,
        dominated=dominated,
    )


def format_masked_poison_result(result: PoisonResult) -> str:
    """Render the four verdicts, one per class, and the diagnostics."""
    classes = "; ".join(
        f"{name}: {', '.join(found)}" for name, found in result.violations
    )
    leaks = ", ".join(
        f"b{batch}:q{row}:h{head}={label.value}"
        for (batch, row, head), label in result.leak_classes
    )
    return (
        f"violations={classes or 'none'}; "
        f"unaffected {result.compared_count - result.mismatch_count}/"
        f"{result.compared_count} equal; "
        f"activity {result.activity_row_count - result.activity_failure_count}/"
        f"{result.activity_row_count} rows, min signed "
        f"{result.min_activity_signed:.9g}; "
        f"baseline max {result.max_baseline_abs:.9g}; "
        f"targets without a nonzero baseline coordinate {result.zero_target_count}"
        + (f"; target rows: {leaks}" if leaks else "")
    )
