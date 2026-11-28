# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the masked-token poisoning relation: fixture, row
sets, derived limits, analyzer, subject scope, and the checks of the compiled
artifact.

The analyzer and the lane are exercised on outputs of ``_model_dense_run``, an
FP32 model of the dense kernel's masked single-pass online softmax, with and
without deliberate mask leaks. The model exists only in these contracts. No GPU
verdict uses it: the relation compares two subject outputs.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from enum import Enum
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

from ._analytical import _legal_bounds, decode_bits, encode_bits, legal_interval
from ._dense_subject import (
    DENSE_SUBJECTS,
    MASKED_POISON_QUICK_SUBJECT_CASES,
    MASKED_POISON_STANDARD_SUBJECT_CASES,
    launch_geometry,
    resolve_dense_subject,
)
from ._masked_poison import (
    ACTIVITY_SCORE,
    ACTIVITY_THRESHOLD,
    MASKED_POISON_QUICK_CASES,
    MASKED_POISON_STANDARD_CASES,
    OVERFLOW_GAP,
    POISON_ANCHORS,
    POISON_CAUSAL_BF16,
    POISON_CAUSAL_FP16,
    POISON_KEY_SCALE,
    POISON_VALUE_SCALE,
    POISON_WINDOW_BF16,
    POISON_WINDOW_FP16,
    ZERO_GAP,
    BoundaryControl,
    LeakClass,
    MaskedPoisonCase,
    MaskMechanism,
    PoisonFixture,
    PoisonLimits,
    PoisonResult,
    analyze_masked_poison,
    control_positions,
    control_witness,
    format_masked_poison_result,
    masked_poison_inputs,
    masked_poison_limits,
    poison_fixture,
    poison_mechanisms,
    poison_positions,
    poison_sign_row,
    poisoned_bit_violations,
    poisoned_tensors,
    sees_token,
)
from ._masked_poison_lane import _control_failure
from ._masked_poison_path import (
    MaskedPath,
    _IndexEvaluator,
    _parse_function,
    classify_masked_path,
    expected_epilogue_counts,
    expected_loop_counts,
    legal_set_violations,
    masked_path_violations,
    path_scope,
    steering_violations,
    verified_masked_path,
    visited_tiles,
)
from ._permutation_path import _F32_INITIAL_MAX, REDERIVE

_ALL_SUBJECT_CASES = (
    *MASKED_POISON_QUICK_SUBJECT_CASES,
    *MASKED_POISON_STANDARD_SUBJECT_CASES,
)
_ALL_CASES = (*MASKED_POISON_QUICK_CASES, *MASKED_POISON_STANDARD_CASES)
_LOG2E = 1.4426950408889634
# Key tile and query block of the shipped gfx942 dense specs: every case uses
# 64-key tiles; the BF16 window case uses 64-row query blocks, the others 128.
_MODEL_BLOCK_N = 64


def _case_id(case: MaskedPoisonCase) -> str:
    return case.case_id


def _subject_id(item: tuple[str, MaskedPoisonCase]) -> str:
    return f"{item[0]}-{item[1].case_id}"


def _model_block_m(case: MaskedPoisonCase) -> int:
    return 64 if (case.mask, case.dtype) == ("window", "bf16") else 128


# ---------------------------------------------------------------------------
# Cases and scope
# ---------------------------------------------------------------------------


def test_matrices_split_the_four_cases_between_the_tiers() -> None:
    """One mask per dtype in the quick tier, the other pairing in the standard
    tier, all on the gfx942 dense subject."""
    assert MASKED_POISON_QUICK_CASES == (POISON_WINDOW_FP16, POISON_CAUSAL_BF16)
    assert MASKED_POISON_STANDARD_CASES == (POISON_CAUSAL_FP16, POISON_WINDOW_BF16)
    assert MASKED_POISON_QUICK_SUBJECT_CASES == tuple(
        ("gfx942", case) for case in MASKED_POISON_QUICK_CASES
    )
    assert MASKED_POISON_STANDARD_SUBJECT_CASES == tuple(
        ("gfx942", case) for case in MASKED_POISON_STANDARD_CASES
    )
    assert {arch for arch, _ in _ALL_SUBJECT_CASES} <= set(DENSE_SUBJECTS)
    assert len({case.case_id for case in _ALL_CASES}) == 4


@pytest.mark.parametrize(
    "arguments,match",
    [
        ((1, 256, 128, 8, 2, 64, "fp16", "causal"), "not self-attention"),
        ((1, 256, 256, 8, 2, 32, "fp16", "causal"), "power of four"),
        ((1, 256, 256, 8, 2, 128, "fp16", "causal"), "power of four"),
        ((1, 256, 256, 8, 2, 64, "fp16", "window", 2), "cannot hold"),
        ((1, 128, 128, 8, 2, 64, "fp16", "causal"), "leaves no room"),
        ((1, 256, 256, 8, 2, 64, "fp16", "window", 128), "leave the key range"),
        ((1, 256, 256, 8, 2, 16, "fp16", "causal"), "target gap"),
    ],
)
def test_case_scope_refuses_what_the_derivation_does_not_cover(
    arguments: tuple, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MaskedPoisonCase(*arguments).require_scope()


def test_case_rejects_an_unknown_mask_and_a_stray_window() -> None:
    with pytest.raises(ValueError, match="mask must be one of"):
        MaskedPoisonCase(1, 256, 256, 8, 2, 64, "fp16", "none")
    with pytest.raises(ValueError, match="window mask requires"):
        MaskedPoisonCase(1, 256, 256, 8, 2, 64, "fp16", "window")
    with pytest.raises(ValueError, match="window must be zero"):
        MaskedPoisonCase(1, 256, 256, 8, 2, 64, "fp16", "causal", 64)


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_quick_and_standard_cases_are_in_scope(case: MaskedPoisonCase) -> None:
    case.require_scope()


# ---------------------------------------------------------------------------
# Poisoned tokens and row sets
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive runs of True."""
    found, start = [], None
    for index, flag in enumerate([*mask.tolist(), False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            found.append((start, index - 1))
            start = None
    return found


@pytest.mark.parametrize(
    "case,anchor,positions,unaffected",
    [
        (POISON_CAUSAL_FP16, 127, (131, 255), [(0, 130)]),
        (POISON_CAUSAL_FP16, 200, (204, 255), [(0, 203)]),
        (POISON_WINDOW_FP16, 127, (63, 131), [(0, 62), (127, 130), (195, 255)]),
        (POISON_WINDOW_FP16, 200, (136, 204), [(0, 135), (200, 203)]),
    ],
)
def test_positions_and_unaffected_rows_of_the_worked_example(
    case: MaskedPoisonCase, anchor: int, positions: tuple, unaffected: list
) -> None:
    """The four rows of the write-up's numerical example."""
    assert poison_positions(case, anchor) == positions
    assert _runs(~sees_token(case, positions)) == unaffected


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
@pytest.mark.parametrize("anchor", POISON_ANCHORS)
def test_unaffected_rows_follow_the_closed_forms(
    case: MaskedPoisonCase, anchor: int
) -> None:
    """The row set comes from the mask rule key by key; the write-up's closed
    forms must agree with it, and with the scalar rule of the uniform gate."""
    rows = np.arange(case.seqlen_q)
    group = case.group
    if case.mask == "causal":
        closed = rows <= anchor + group - 1
    else:
        low = (rows >= anchor - case.window) & (rows <= anchor - 1)
        high = (rows >= anchor + group) & (rows <= anchor + group + case.window - 1)
        closed = ~(low | high)
    positions = poison_positions(case, anchor)
    assert np.array_equal(~sees_token(case, positions), closed)
    for row in (0, anchor - 1, anchor, anchor + group - 1, anchor + group, 255):
        lower, upper = legal_interval(case, row)
        seen = any(lower <= position <= upper for position in positions)
        assert seen == (not closed[row])


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_boundary_controls_move_one_token_into_one_target_row(
    case: MaskedPoisonCase,
) -> None:
    """A control differs from the poisoned tokens in one position, by one key,
    and among the unaffected rows only its witness row sees the moved token."""
    for anchor in POISON_ANCHORS:
        base = poison_positions(case, anchor)
        unaffected = ~sees_token(case, base)
        for control in BoundaryControl:
            if control is BoundaryControl.WINDOW_KEY and case.mask != "window":
                with pytest.raises(ValueError, match="requires a window case"):
                    control_positions(case, anchor, control)
                continue
            moved = control_positions(case, anchor, control)
            changed = [(a, b) for a, b in zip(base, moved) if a != b]
            assert len(changed) == 1 and abs(changed[0][0] - changed[0][1]) == 1
            row, offset = control_witness(case, anchor, control)
            exposed = np.flatnonzero(unaffected & sees_token(case, moved))
            assert exposed.tolist() == [row]
            assert row - anchor == offset
    with pytest.raises(ValueError, match="unknown boundary control"):
        control_positions(case, POISON_ANCHORS[0], "nonsense")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@cache
def _baseline(case: MaskedPoisonCase):
    return masked_poison_inputs(case)


@cache
def _fixture(case: MaskedPoisonCase, anchor: int) -> PoisonFixture:
    return poison_fixture(case, _baseline(case), anchor)


_INPUT_DIGESTS = {
    "fp16": ("11ea01a8b5764fa4", "072fc27c9f0e1015", "e7b3a7b8457ed75b"),
    "bf16": ("c69ad68395529814", "1b0a81c80f96ab8f", "ca9824a5b3f2334d"),
}


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_inputs_are_signs_exact_in_the_dtype_and_stay_pinned(
    case: MaskedPoisonCase,
) -> None:
    baseline = _baseline(case)
    digests = []
    for values in (baseline.q, baseline.k, baseline.v):
        assert set(np.unique(values).tolist()) == {-1.0, 1.0}
        bits = encode_bits(values, case.dtype)
        assert np.array_equal(decode_bits(bits, case.dtype), values)
        digests.append(hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest()[:16])
    assert tuple(digests) == _INPUT_DIGESTS[case.dtype]
    assert baseline.key_seed == 0x6B


@pytest.mark.parametrize("case", (POISON_CAUSAL_FP16, POISON_WINDOW_FP16), ids=_case_id)
def test_every_row_has_a_known_strict_maximum(case: MaskedPoisonCase) -> None:
    """Key tokens are pairwise distinct sign vectors, so the planted key scores
    ``D`` and every other key at most ``D - 2``: integer scores, checked for
    every row and head against its legal keys."""
    baseline = _baseline(case)
    first, last = _legal_bounds(case, None)
    keys = np.arange(case.seqlen_k)
    legal = (first[:, None] <= keys) & (keys <= last[:, None])
    rows = np.arange(case.seqlen_q)
    for head in range(case.nhead_q):
        routed = head // case.group
        tokens = baseline.k[0, :, routed, :].astype(np.int64)
        assert len({row.tobytes() for row in tokens}) == case.seqlen_k
        scores = baseline.q[0, :, head, :].astype(np.int64) @ tokens.T
        planted = np.maximum(rows - head % case.group, 0)
        assert np.all(legal[rows, planted])
        assert np.all(scores[rows, planted] == case.head_dim)
        others = legal.copy()
        others[rows, planted] = False
        assert np.max(np.where(others, scores, -case.head_dim)) <= case.head_dim - 2


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
@pytest.mark.parametrize("anchor", POISON_ANCHORS)
def test_fixture_sets_are_integer_facts_of_the_launched_inputs(
    case: MaskedPoisonCase, anchor: int
) -> None:
    baseline = _baseline(case)
    fixture = _fixture(case, anchor)
    heads = np.arange(case.nhead_q)
    assert fixture.targets.tolist() == (anchor + heads % case.group).tolist()
    assert np.all(fixture.unaffected[fixture.targets])

    # Activity, recomputed one triple at a time from the definition.
    anchor_key = baseline.k[0, anchor].astype(np.int64)
    expected = np.zeros_like(fixture.activity)
    for head in heads:
        for row in range(case.seqlen_q):
            score = int(
                baseline.q[0, row, head].astype(np.int64)
                @ anchor_key[head // case.group]
            )
            assert score == fixture.anchor_scores[0, row, head]
            expected[0, row, head] = (
                not fixture.unaffected[row] and score >= ACTIVITY_SCORE
            )
    assert np.array_equal(fixture.activity, expected)
    assert np.all(fixture.activity.any(axis=1))
    assert np.all(fixture.anchor_scores[0, fixture.targets, heads] == case.head_dim)


@pytest.mark.parametrize("case", (POISON_WINDOW_FP16, POISON_CAUSAL_BF16), ids=_case_id)
def test_poisoned_tensors_change_exactly_the_poisoned_tokens(
    case: MaskedPoisonCase,
) -> None:
    baseline = _baseline(case)
    anchor = POISON_ANCHORS[0]
    fixture = _fixture(case, anchor)
    kept = np.ones(case.seqlen_k, dtype=bool)
    kept[list(fixture.positions)] = False
    assert np.array_equal(fixture.k[:, kept], baseline.k[:, kept])
    assert np.array_equal(fixture.v[:, kept], baseline.v[:, kept])
    sign_row = poison_sign_row(case)
    assert set(np.unique(sign_row).tolist()) == {-1.0, 1.0}
    for position in fixture.positions:
        assert np.array_equal(
            fixture.k[:, position], POISON_KEY_SCALE * baseline.k[:, anchor]
        )
        assert np.array_equal(fixture.v[:, position], POISON_VALUE_SCALE * sign_row)

    bits = {
        name: encode_bits(values, case.dtype)
        for name, values in (
            ("k", baseline.k),
            ("v", baseline.v),
            ("pk", fixture.k),
            ("pv", fixture.v),
        )
    }
    assert not poisoned_bit_violations(
        fixture, bits["k"], bits["v"], bits["pk"], bits["pv"]
    )
    legal_edit = bits["pk"].copy()
    legal_edit[0, 5, 0, 0] ^= 1
    assert poisoned_bit_violations(
        fixture, bits["k"], bits["v"], legal_edit, bits["pv"]
    ) == ("1 K element(s) changed outside the poison",)
    wrong_value = bits["pv"].copy()
    wrong_value[0, fixture.positions[0], 0, 0] ^= 0x8000
    found = poisoned_bit_violations(
        fixture, bits["k"], bits["v"], bits["pk"], wrong_value
    )
    assert found == (
        f"poisoned value {fixture.positions[0]} is not the scaled sign row",
    )


def test_fixture_rejects_a_poison_inside_a_target_row(monkeypatch) -> None:
    """A fixture whose target rows see a poisoned token is not a fixture of the
    relation; it is refused, not judged."""
    case = POISON_CAUSAL_FP16
    module = __import__(poison_fixture.__module__, fromlist=["poison_positions"])
    monkeypatch.setattr(
        module,
        "poison_positions",
        lambda case, anchor: (anchor + 1, case.seqlen_k - 1),
    )
    with pytest.raises(ValueError, match="a target row sees a poisoned token"):
        poison_fixture(case, _baseline(case), POISON_ANCHORS[0])


# ---------------------------------------------------------------------------
# Derived limits
# ---------------------------------------------------------------------------


_PINNED_LIMITS = {
    POISON_WINDOW_FP16: (1.0011030908967156, 15.62851616257891),
    POISON_CAUSAL_BF16: (1.0079510399723732, 14.565109827643044),
    POISON_CAUSAL_FP16: (1.0011145419770147, 14.615350603869315),
    POISON_WINDOW_BF16: (1.0079510399723732, 15.575003495368913),
}


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_limits_pinned_values(case: MaskedPoisonCase) -> None:
    limits = masked_poison_limits(case)
    assert (limits.baseline_ceiling, limits.activity_floor) == _PINNED_LIMITS[case]
    assert limits.target_gap == 173.1218822576433
    assert limits.takeover_gap == 11.5
    assert limits.threshold == ACTIVITY_THRESHOLD
    assert limits.legal_keys == (case.window or case.seqlen_k)
    assert limits.baseline_ceiling < limits.threshold < limits.activity_floor
    assert limits.target_gap >= max(ZERO_GAP, OVERFLOW_GAP)


def test_limits_bracket_the_exact_arithmetic_levels() -> None:
    """In exact arithmetic a baseline output is at most 1 and an activity row
    stores at least ``16 Pi - (1 - Pi)`` with ``Pi = 1 / (1 + 255 e**-8)``. The
    derived limits lie on the conservative side of both, within the allowances
    the write-up states."""
    share = 1.0 / (1.0 + 255.0 * math.exp(-8.0))
    level = POISON_VALUE_SCALE * share - (1.0 - share)
    assert 14.66 < level < 14.661
    for case, eta_base, eta_act in (
        (POISON_CAUSAL_FP16, 1.12e-3, 0.05),
        (POISON_CAUSAL_BF16, 7.96e-3, 0.10),
    ):
        limits = masked_poison_limits(case)
        assert 1.0 < limits.baseline_ceiling < 1.0 + eta_base
        assert level - eta_act < limits.activity_floor < level


def test_takeover_gap_is_a_certified_half_integer() -> None:
    """The takeover gap is the largest half-integer not above the computed gap
    of an activity row, ``(lambda rho - D) log2(e) / sqrt(D)`` less the score
    rounding: 11.5 below 8 log2(e) = 11.54."""
    exact = Fraction(POISON_KEY_SCALE * ACTIVITY_SCORE - 64, 8) * Fraction(_LOG2E)
    limits = masked_poison_limits(POISON_CAUSAL_FP16)
    assert Fraction(limits.takeover_gap) <= exact
    assert exact - Fraction(limits.takeover_gap) < Fraction(1, 2)
    assert float(np.float32(limits.takeover_gap)) == limits.takeover_gap


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _synthetic_pair(case: MaskedPoisonCase, fixture: PoisonFixture):
    """A baseline of 1/2 everywhere and a poisoned output that equals it on the
    unaffected rows and holds the poison value on every other row."""
    base = encode_bits(np.full(case.output_shape, 0.5, np.float32), case.dtype)
    ideal = np.full(case.output_shape, 0.5, np.float32)
    signs = fixture.sign_row[:, np.arange(case.nhead_q) // case.group, :]
    ideal[:, ~fixture.unaffected] = (POISON_VALUE_SCALE * signs)[:, None]
    return base, encode_bits(ideal, case.dtype)


def _bits(value: float, dtype: str) -> int:
    return int(encode_bits(np.array([value], np.float32), dtype)[0])


@pytest.mark.parametrize("case", MASKED_POISON_QUICK_CASES, ids=_case_id)
def test_analyzer_keeps_the_four_verdicts_apart(case: MaskedPoisonCase) -> None:
    fixture = _fixture(case, POISON_ANCHORS[0])
    limits = masked_poison_limits(case)
    base, poisoned = _synthetic_pair(case, fixture)
    row = int(np.flatnonzero(fixture.unaffected)[3])

    clean = analyze_masked_poison(fixture, limits, base, base, poisoned)
    assert clean.violations == ()
    assert clean.compared_count == int(fixture.unaffected.sum()) * 8 * 64
    assert clean.activity_row_count == int(fixture.activity.sum())
    assert clean.min_activity_signed == POISON_VALUE_SCALE

    undelivered = analyze_masked_poison(fixture, limits, base, base, base)
    assert [name for name, _ in undelivered.violations] == ["activity"]
    assert undelivered.activity_failure_count == undelivered.activity_row_count

    changed = poisoned.copy()
    changed[0, row, 2, 7] += 1
    relation = analyze_masked_poison(fixture, limits, base, base, changed)
    assert [name for name, _ in relation.violations] == ["relation"]
    assert (relation.mismatch_count, relation.max_ulp) == (1, 1)
    assert relation.mismatched_rows == (row,)
    assert relation.mismatched[0, row, 2] and relation.mismatched.sum() == 1

    replayed = base.copy()
    replayed[0, 0, 0, 0] += 1
    validity = analyze_masked_poison(fixture, limits, base, replayed, poisoned)
    assert [name for name, _ in validity.violations] == ["validity"]

    unwritten = poisoned.copy()
    unwritten[0, row, 0, 0] = 0x7E00 if case.dtype == "fp16" else 0x7FC0
    invalid = analyze_masked_poison(fixture, limits, base, base, unwritten)
    assert [name for name, _ in invalid.violations] == ["sanity", "relation"]
    assert invalid.invalid_count == 1

    too_large = base.copy()
    too_large[0, row, 0, 0] = _bits(1.25, case.dtype)
    grown = analyze_masked_poison(fixture, limits, too_large, too_large, poisoned)
    assert "sanity" in [name for name, _ in grown.violations]
    assert grown.ceiling_count == 1 and grown.max_baseline_abs == 1.25


@pytest.mark.parametrize("case", MASKED_POISON_QUICK_CASES, ids=_case_id)
def test_analyzer_compares_raw_bits_including_the_sign_of_zero(
    case: MaskedPoisonCase,
) -> None:
    """``+0`` and ``-0`` are different stored outputs; the count of such
    coordinates is reported next to the violation."""
    fixture = _fixture(case, POISON_ANCHORS[0])
    limits = masked_poison_limits(case)
    base, poisoned = _synthetic_pair(case, fixture)
    row = int(np.flatnonzero(fixture.unaffected)[0])
    base[0, row, 1, 5] = 0x0000
    poisoned[0, row, 1, 5] = 0x8000
    result = analyze_masked_poison(fixture, limits, base, base, poisoned)
    assert [name for name, _ in result.violations] == ["relation"]
    assert (result.mismatch_count, result.zero_sign_only_count) == (1, 1)
    assert result.max_ulp == 0


@pytest.mark.parametrize("case", MASKED_POISON_QUICK_CASES, ids=_case_id)
def test_analyzer_labels_a_changed_target_row(case: MaskedPoisonCase) -> None:
    fixture = _fixture(case, POISON_ANCHORS[0])
    limits = masked_poison_limits(case)
    base, poisoned = _synthetic_pair(case, fixture)
    signs = fixture.sign_row[:, np.arange(case.nhead_q) // case.group, :]
    rows = {head: int(fixture.targets[head]) for head in range(4)}

    leaked = poisoned.copy()
    leaked[:, rows[0], 0] = 0x7E00 if case.dtype == "fp16" else 0x7FC0
    leaked[:, rows[1], 1] = 0
    leaked[:, rows[2], 2] = encode_bits(POISON_VALUE_SCALE * signs[:, 2], case.dtype)
    leaked[0, rows[3], 3, 0] += 1
    result = analyze_masked_poison(fixture, limits, base, base, leaked)
    assert dict(result.leak_classes) == {
        (0, rows[0], 0): LeakClass.NONFINITE,
        (0, rows[1], 1): LeakClass.ZERO,
        (0, rows[2], 2): LeakClass.POISON_DOMINATED,
        (0, rows[3], 3): LeakClass.OTHER,
    }
    assert result.dominated[0, rows[2], 2]
    text = format_masked_poison_result(result)
    assert f"b0:q{rows[1]}:h1=zero" in text and "relation:" in text


def _scalar_counts(
    fixture: PoisonFixture,
    limits: PoisonLimits,
    base: np.ndarray,
    replay: np.ndarray,
    poisoned: np.ndarray,
) -> dict[str, int]:
    """The analyzer's counts, one coordinate at a time: the executable
    specification the vectorized analyzer must agree with."""
    case = fixture.case
    exponent = 0x7C00 if case.dtype == "fp16" else 0x7F80
    counts = dict.fromkeys(
        ("replay", "invalid", "ceiling", "mismatch", "zero_sign", "activity_rows"), 0
    )
    failed_rows = set()
    for batch in range(case.batch):
        for row in range(case.seqlen_q):
            for head in range(case.nhead_q):
                active = bool(fixture.activity[batch, row, head])
                counts["activity_rows"] += active
                for dim in range(case.head_dim):
                    b = int(base[batch, row, head, dim])
                    p = int(poisoned[batch, row, head, dim])
                    counts["replay"] += b != int(replay[batch, row, head, dim])
                    b_finite = (b & exponent) != exponent
                    p_finite = (p & exponent) != exponent
                    counts["invalid"] += (not b_finite) + (not p_finite)
                    if b_finite:
                        value = abs(float(decode_bits(np.uint16(b), case.dtype)))
                        counts["ceiling"] += value > limits.baseline_ceiling
                    if fixture.unaffected[row] and b != p:
                        counts["mismatch"] += 1
                        counts["zero_sign"] += (b | p) & 0x7FFF == 0
                    if active:
                        sign = float(fixture.sign_row[batch, head // case.group, dim])
                        value = float(decode_bits(np.uint16(p), case.dtype))
                        if not p_finite or sign * value < limits.threshold:
                            failed_rows.add((batch, row, head))
    counts["activity_failures"] = len(failed_rows)
    return counts


def test_vectorized_analyzer_agrees_with_the_scalar_specification() -> None:
    case = POISON_WINDOW_FP16
    fixture = _fixture(case, POISON_ANCHORS[1])
    limits = masked_poison_limits(case)
    base, poisoned = _synthetic_pair(case, fixture)
    generator = np.random.default_rng(9)
    for array in (base, poisoned):
        flat = array.reshape(-1)
        picks = generator.choice(flat.size, size=600, replace=False)
        flat[picks[:200]] ^= 1
        flat[picks[200:260]] = 0x7E00
        flat[picks[260:300]] = 0x8000
        flat[picks[300:340]] = 0x0000
        flat[picks[340:400]] = _bits(1.5, case.dtype)
    row = int(np.flatnonzero(fixture.unaffected)[1])
    base[0, row, 0, 0], poisoned[0, row, 0, 0] = 0x0000, 0x8000
    replay = base.copy()
    replay.reshape(-1)[generator.choice(base.size, size=17, replace=False)] ^= 2

    result = analyze_masked_poison(fixture, limits, base, replay, poisoned)
    assert _scalar_counts(fixture, limits, base, replay, poisoned) == {
        "replay": result.replay_mismatch_count,
        "invalid": result.invalid_count,
        "ceiling": result.ceiling_count,
        "mismatch": result.mismatch_count,
        "zero_sign": result.zero_sign_only_count,
        "activity_rows": result.activity_row_count,
        "activity_failures": result.activity_failure_count,
    }
    assert result.mismatch_count and result.zero_sign_only_count


def test_analyzer_validates_its_inputs() -> None:
    case = POISON_CAUSAL_BF16
    fixture = _fixture(case, POISON_ANCHORS[0])
    limits = masked_poison_limits(case)
    base, poisoned = _synthetic_pair(case, fixture)
    with pytest.raises(ValueError, match="raw uint16 bits"):
        analyze_masked_poison(fixture, limits, base.astype(np.float32), base, poisoned)
    with pytest.raises(ValueError, match="output shape"):
        analyze_masked_poison(fixture, limits, base, base[:, :8], poisoned)
    with pytest.raises(ValueError, match="limits dtype"):
        other = masked_poison_limits(POISON_CAUSAL_FP16)
        analyze_masked_poison(fixture, other, base, base, poisoned)


# ---------------------------------------------------------------------------
# Model of the masked dense path, with deliberate leaks
# ---------------------------------------------------------------------------


class _Leak(str, Enum):
    """Mask defects of the model. Each breaks one clause of the path the bit
    equality is derived for."""

    FUTURE_KEY = "future_key"
    OLD_WINDOW_KEY = "old_window_key"
    WRONG_TILE = "wrong_tile"
    MASK_AFTER_MAXIMUM = "mask_after_maximum"
    UNMASKED_NORMALIZER = "unmasked_normalizer"
    UNMASKED_VALUES = "unmasked_values"
    NO_FIRST_RESCALE = "no_first_rescale"


def _round_to(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round FP32 values to the subject dtype, to nearest even, as FP32."""
    with np.errstate(over="ignore"):
        if dtype == "fp16":
            return values.astype(np.float16).astype(np.float32)
        return bf16_to_f32(f32_to_bf16(values)).reshape(values.shape)


def _store(values: np.ndarray, dtype: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        if dtype == "fp16":
            return values.astype(np.float16).view(np.uint16)
        return f32_to_bf16(values).reshape(values.shape)


def _model_tiles(case: MaskedPoisonCase, block_m: int, block_n: int):
    """Per query row, the first and one past the last key tile of the model:
    up to the diagonal tile of the row's query block, and with a window from
    the tile ``window / block_n`` below the block's first diagonal tile."""
    block = np.arange(case.seqlen_q) // block_m
    per_block = block_m // block_n
    high = np.minimum((block + 1) * per_block, case.seqlen_k // block_n)
    low = np.maximum(block * per_block - case.window // block_n, 0)
    return (low if case.mask == "window" else np.zeros_like(low)), high


def _model_dense_run(
    case: MaskedPoisonCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    block_m: int | None = None,
    block_n: int = _MODEL_BLOCK_N,
    leak: _Leak | None = None,
    sentinel: float = -1e30,
) -> np.ndarray:
    """FP32 model of the dense kernel's masked online softmax; returns raw bits.

    Per row and key tile: scores in the base-2 logarithm domain, the mask
    sentinel on illegal lanes, the running maximum, the rescale, uncast weights
    into the normalizer and cast weights into the accumulator, then one
    reciprocal and the store. Bit-for-bit agreement with the hardware is not
    claimed; the model shares the path's structure, which is what the relation
    tests."""
    block_m = _model_block_m(case) if block_m is None else block_m
    sigma = np.float32(sentinel)
    scale = np.float32(np.float32(1.0 / math.sqrt(case.head_dim)) * np.float32(_LOG2E))
    first, last = _legal_bounds(case, None)
    keys = np.arange(case.seqlen_k)
    upper = last + 1 if leak is _Leak.FUTURE_KEY else last
    lower = first - 1 if leak is _Leak.OLD_WINDOW_KEY else first
    legal = (lower[:, None] <= keys) & (keys <= upper[:, None])
    if leak is _Leak.WRONG_TILE:
        # Every tile after the first is masked with the previous tile's keys.
        legal = np.concatenate([legal[:, :block_n], legal[:, :-block_n]], axis=1)
    low, high = _model_tiles(case, block_m, block_n)
    out = np.empty(case.output_shape, dtype=np.uint16)
    with np.errstate(all="ignore"):
        for batch in range(case.batch):
            for head in range(case.nhead_q):
                routed = head // case.group
                raw = (q[batch, :, head, :] @ k[batch, :, routed, :].T).astype(
                    np.float32
                ) * scale
                maximum = np.full(case.seqlen_q, sigma, dtype=np.float32)
                normalizer = np.zeros(case.seqlen_q, dtype=np.float32)
                accumulator = np.zeros((case.seqlen_q, case.head_dim), np.float32)
                for tile in range(case.seqlen_k // block_n):
                    active = (low <= tile) & (tile < high)
                    columns = slice(tile * block_n, (tile + 1) * block_n)
                    scores = raw[:, columns]
                    masked = np.where(legal[:, columns], scores, sigma)
                    seen = scores if leak is _Leak.MASK_AFTER_MAXIMUM else masked
                    new = np.maximum(maximum, seen.max(axis=1))
                    alpha = np.exp2(maximum - new)
                    if leak is _Leak.NO_FIRST_RESCALE:
                        alpha = np.where(maximum == sigma, np.float32(1.0), alpha)
                    weights = np.exp2(masked - new[:, None])
                    loose = np.exp2(scores - new[:, None])
                    summed = loose if leak is _Leak.UNMASKED_NORMALIZER else weights
                    valued = loose if leak is _Leak.UNMASKED_VALUES else weights
                    grown = normalizer * alpha + summed.sum(axis=1, dtype=np.float32)
                    products = (
                        _round_to(valued, case.dtype)[:, :, None]
                        * v[batch, columns, routed, :][None, :, :]
                    )
                    filled = accumulator * alpha[:, None] + products.sum(
                        axis=1, dtype=np.float32
                    )
                    maximum = np.where(active, new, maximum)
                    normalizer = np.where(active, grown, normalizer)
                    accumulator = np.where(active[:, None], filled, accumulator)
                stored = accumulator * (np.float32(1.0) / normalizer)[:, None]
                out[batch, :, head, :] = _store(stored.astype(np.float32), case.dtype)
    return out


@cache
def _model_base(case: MaskedPoisonCase, leak: _Leak | None, sentinel: float):
    baseline = _baseline(case)
    return _model_dense_run(
        case, baseline.q, baseline.k, baseline.v, leak=leak, sentinel=sentinel
    )


def _model_result(
    case: MaskedPoisonCase,
    anchor: int,
    *,
    leak: _Leak | None = None,
    sentinel: float = -1e30,
    positions: tuple[int, ...] | None = None,
) -> PoisonResult:
    baseline = _baseline(case)
    fixture = _fixture(case, anchor)
    k, v = (
        (fixture.k, fixture.v)
        if positions is None
        else poisoned_tensors(case, baseline, anchor, positions)
    )
    base = _model_base(case, leak, sentinel)
    poisoned = _model_dense_run(case, baseline.q, k, v, leak=leak, sentinel=sentinel)
    return analyze_masked_poison(
        fixture, masked_poison_limits(case), base, base, poisoned
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
@pytest.mark.parametrize("anchor", POISON_ANCHORS)
def test_modeled_path_passes_all_four_verdicts(
    case: MaskedPoisonCase, anchor: int
) -> None:
    result = _model_result(case, anchor)
    assert result.violations == (), format_masked_poison_result(result)
    assert result.zero_target_count == 0
    assert result.min_activity_signed >= masked_poison_limits(case).activity_floor
    assert result.max_baseline_abs <= 1.0 + 1e-2


@pytest.mark.parametrize("case", MASKED_POISON_QUICK_CASES, ids=_case_id)
def test_modeled_boundary_controls_are_seen_where_they_must_be(
    case: MaskedPoisonCase,
) -> None:
    """One token moved one key into a target row's legal set changes that row,
    takes it over in the head that copies the anchor key, and changes no other
    unaffected row."""
    for anchor in POISON_ANCHORS:
        fixture = _fixture(case, anchor)
        for control in BoundaryControl:
            if control is BoundaryControl.WINDOW_KEY and case.mask != "window":
                continue
            moved = _model_result(
                case, anchor, positions=control_positions(case, anchor, control)
            )
            assert _control_failure("model", fixture, control, moved) is None
            row, _ = control_witness(case, anchor, control)
            assert moved.mismatched_rows == (row,)
        clean = _model_result(case, anchor)
        failure = _control_failure("model", fixture, BoundaryControl.FUTURE_KEY, clean)
        assert failure is not None and "did not respond" in failure


@pytest.mark.parametrize(
    "case,anchor,leak,label",
    [
        (POISON_CAUSAL_BF16, 127, _Leak.FUTURE_KEY, LeakClass.POISON_DOMINATED),
        (POISON_WINDOW_FP16, 127, _Leak.FUTURE_KEY, LeakClass.POISON_DOMINATED),
        (POISON_WINDOW_FP16, 200, _Leak.OLD_WINDOW_KEY, LeakClass.POISON_DOMINATED),
        (POISON_CAUSAL_BF16, 127, _Leak.WRONG_TILE, LeakClass.POISON_DOMINATED),
        (POISON_CAUSAL_BF16, 127, _Leak.MASK_AFTER_MAXIMUM, LeakClass.NONFINITE),
        (POISON_WINDOW_FP16, 200, _Leak.MASK_AFTER_MAXIMUM, LeakClass.NONFINITE),
        (POISON_CAUSAL_BF16, 127, _Leak.UNMASKED_NORMALIZER, LeakClass.ZERO),
        (POISON_CAUSAL_BF16, 127, _Leak.UNMASKED_VALUES, LeakClass.NONFINITE),
        (POISON_WINDOW_FP16, 127, _Leak.NO_FIRST_RESCALE, LeakClass.OTHER),
        (POISON_WINDOW_BF16, 127, _Leak.NO_FIRST_RESCALE, LeakClass.OTHER),
    ],
)
def test_modeled_leak_fails_the_relation_with_its_signature(
    case: MaskedPoisonCase, anchor: int, leak: _Leak, label: LeakClass
) -> None:
    """Each mask defect changes a target row, and the stored bits of that row
    carry the signature the write-up's taxonomy gives the defect."""
    result = _model_result(case, anchor, leak=leak)
    assert result.relation, format_masked_poison_result(result)
    targets = set(_fixture(case, anchor).targets.tolist())
    assert set(result.mismatched_rows) <= set(
        np.flatnonzero(_fixture(case, anchor).unaffected).tolist()
    )
    assert targets & set(result.mismatched_rows)
    assert label in {found for _, found in result.leak_classes}


def test_leak_before_the_first_legal_key_needs_a_window_case() -> None:
    """A causal row has key 0 in its first tile, so there is no state before
    its first legal key and nothing for that defect to leak."""
    for anchor in POISON_ANCHORS:
        result = _model_result(POISON_CAUSAL_BF16, anchor, leak=_Leak.NO_FIRST_RESCALE)
        assert result.violations == ()


@pytest.mark.parametrize("case", MASKED_POISON_QUICK_CASES, ids=_case_id)
def test_modeled_sentinel_decides_whether_a_masked_weight_matters(
    case: MaskedPoisonCase,
) -> None:
    """A sentinel inside the score range gives masked lanes a visible weight and
    fails the relation. A sentinel far below it but still finite gives them a
    weight that is not zero and rounds away: the relation passes although the
    exact-zero premise is false. Such a kernel is outside what the lane's path
    check admits; this is why the check pins the sentinel."""
    anchor = POISON_ANCHORS[0]
    visible = _model_result(case, anchor, sentinel=0.0)
    assert visible.relation
    assert _model_result(case, anchor, sentinel=-100.0).violations == ()


def test_model_tiles_are_the_kernels() -> None:
    """The model's tile ranges are the ones the lowered kernels' own loop
    bounds give, for every case."""
    for arch, case in _ALL_SUBJECT_CASES:
        ir, spec, scope = _lowered_subject(arch, case)
        assert (int(spec.block_m), int(spec.block_n)) == (
            _model_block_m(case),
            _MODEL_BLOCK_N,
        )
        path = classify_masked_path(
            ir, case.dtype, block_n=spec.block_n, head_size=spec.head_size
        )
        low, high = visited_tiles(ir, path, **_legal_scope(scope))
        expected = _model_tiles(case, int(spec.block_m), int(spec.block_n))
        assert np.array_equal(low, expected[0]) and np.array_equal(high, expected[1])


# ---------------------------------------------------------------------------
# Lowered masked path of the cases
# ---------------------------------------------------------------------------


@cache
def _lowered_subject(
    arch: str, case: MaskedPoisonCase, llvm_flavor: str | None = None
) -> tuple[str, object, dict]:
    """Lower one case's dispatch-resolved spec to LLVM IR text on the CPU, for
    the host's compiler lane or for a named one, with the scope its checks
    take."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.gfx942.attention_dense import build_attention_dense

    assert arch == "gfx942", "the path check models the gfx942 dense body"
    subject = resolve_dense_subject(arch, case)
    spec = subject.tuning_spec.kernel_spec
    grid, threads = launch_geometry(arch, subject)
    scope = path_scope(
        case,
        block_n=int(spec.block_n),
        head_size=int(spec.head_size),
        grid=grid,
        threads=threads,
    )
    kernel = build_attention_dense(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=llvm_flavor), spec, scope


def _legal_scope(scope: dict) -> dict:
    return {
        name: value
        for name, value in scope.items()
        if name not in ("block_n", "head_size")
    }


_EXPECTED_LOOPS = {
    POISON_WINDOW_FP16: ("both",),
    POISON_CAUSAL_BF16: ("upper",),
    POISON_CAUSAL_FP16: ("upper",),
    POISON_WINDOW_BF16: ("lower", "none", "upper"),
}


@pytest.mark.parametrize("subject_case", _ALL_SUBJECT_CASES, ids=_subject_id)
def test_lowered_masked_path_is_the_one_the_bit_equality_models(
    subject_case: tuple[str, MaskedPoisonCase],
) -> None:
    """The bit equality is derived for one masked path. Any floating-point
    instruction outside it fails here: the derivation is redone, the verdict is
    never loosened."""
    arch, case = subject_case
    ir, spec, scope = _lowered_subject(arch, case)
    path = classify_masked_path(
        ir, case.dtype, block_n=spec.block_n, head_size=spec.head_size
    )
    assert not path.violations, f"{REDERIVE}: " + "; ".join(path.violations)
    assert tuple(loop.mask for loop in path.loops) == _EXPECTED_LOOPS[case]
    for counts in path.loop_counts:
        assert counts == expected_loop_counts(spec.block_n, spec.head_size)
    assert path.epilogue_counts == expected_epilogue_counts(spec.head_size)
    assert not steering_violations(ir)
    assert not legal_set_violations(ir, path, **_legal_scope(scope))
    assert isinstance(verified_masked_path(ir, case.dtype, **scope), MaskedPath)


def test_modeled_counts_at_the_shipped_tile() -> None:
    """A 64-key tile and 64 output columns: two lane halves of 32 keys, 32
    accumulator elements per lane, and eight products per MFMA call."""
    assert expected_loop_counts(64, 64) == {
        "score": 32,
        "max_chain": 32,
        "max_partner": 1,
        "max_running": 1,
        "exp2_weight": 32,
        "exp2_alpha": 1,
        "rescale_normalizer": 1,
        "rescale_accumulator": 32,
        "lane_chain": 32,
        "partner_sum": 1,
        "combine": 1,
        "cast_weight": 32,
        "mfma": 32,
    }
    assert expected_epilogue_counts(64) == {
        "reciprocal": 1,
        "output_product": 32,
        "cast_output": 32,
    }


@pytest.mark.parametrize("llvm_flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("subject_case", _ALL_SUBJECT_CASES, ids=_subject_id)
def test_path_checks_read_every_compiler_lane(
    subject_case: tuple[str, MaskedPoisonCase], llvm_flavor: str
) -> None:
    """The GPU lanes apply the checks to whatever their compiler lane lowers,
    so the checks must read all of them."""
    arch, case = subject_case
    ir, _, scope = _lowered_subject(arch, case, llvm_flavor)
    assert not masked_path_violations(ir, case.dtype, **scope)


@pytest.mark.parametrize("subject_case", _ALL_SUBJECT_CASES, ids=_subject_id)
def test_legal_set_check_executes_the_kernels_own_positions(
    subject_case: tuple[str, MaskedPoisonCase],
) -> None:
    """The check compares the keys the kernel's position arithmetic leaves
    unmasked with the case's legal keys. Given another mask rule's legal keys
    it must disagree: one key more per row, one key fewer, or another
    window."""
    arch, case = subject_case
    ir, spec, scope = _lowered_subject(arch, case)
    path = classify_masked_path(
        ir, case.dtype, block_n=spec.block_n, head_size=spec.head_size
    )
    legal = _legal_scope(scope)
    last = legal["last"]
    shorter = {**legal, "last": np.maximum(last - 1, legal["first"])}
    assert legal_set_violations(ir, path, **shorter)
    longer = {**legal, "last": np.minimum(last + 1, case.seqlen_k - 1)}
    assert legal_set_violations(ir, path, **longer)
    if case.mask == "window":
        assert legal_set_violations(ir, path, **{**legal, "window": case.window // 2})
        wider = {**legal, "first": np.maximum(legal["first"] - 1, 0)}
        assert legal_set_violations(ir, path, **wider)


def _first(pattern: str, ir: str) -> re.Match:
    match = re.search(pattern, ir, flags=re.MULTILINE)
    assert match, f"the lowered IR has no {pattern!r}"
    return match


def _substitute_once(pattern: str, replacement: str) -> Callable[[str], str]:
    def edit(ir: str) -> str:
        edited, count = re.subn(pattern, replacement, ir, count=1, flags=re.MULTILINE)
        assert count == 1, f"the lowered IR has no {pattern!r}"
        return edited

    return edit


_SELECT = rf"(%[\w.]+) = select i1 (%[\w.]+), float (%[\w.]+), float {_F32_INITIAL_MAX}"


def _maximum_reads_a_raw_score(ir: str) -> str:
    """Hand the first mask select's score, not the select, to the tile maximum."""
    select, _, score = _first(_SELECT, ir).groups()
    return _substitute_once(
        rf"(@llvm\.maxnum\.f32\(float [^,]+, float ){re.escape(select)}\)",
        rf"\g<1>{score})",
    )(ir)


def _weight_reads_a_raw_score(ir: str) -> str:
    select, _, score = _first(_SELECT, ir).groups()
    return _substitute_once(rf"(= fsub float ){re.escape(select)},", rf"\g<1>{score},")(
        ir
    )


def _mask_compare_reads_a_score(ir: str) -> str:
    """Make the first mask compare read a value computed from a score."""
    select, compare, score = _first(_SELECT, ir).groups()
    del select
    line = _first(rf"^  {re.escape(compare)} = icmp \w+ i32 (%[\w.]+), ", ir)
    edited = ir.replace(
        line.group(0),
        f"  %steered = fptosi float {score} to i32\n"
        + line.group(0).replace(line.group(1), "%steered"),
        1,
    )
    assert edited != ir
    return edited


def _mask_free_loop_takes_a_tile(ir: str) -> str:
    """Empty the first loop of a three-loop kernel so the mask-free loop after
    it runs the tile."""
    start = _first(r"^  %kl = phi i32 \[ (%[\w.]+), %entry \]", ir).group(1)
    middle = _first(r"^  %km = phi i32 \[ (%[\w.]+), ", ir).group(1)
    return _substitute_once(
        rf"^(  {re.escape(middle)} = select i1 %[\w.]+, i32) %[\w.]+, i32 %[\w.]+$",
        rf"\g<1> {start}, i32 {start}",
    )(ir)


_CAUSAL = ("gfx942", POISON_CAUSAL_FP16)
_WINDOW = ("gfx942", POISON_WINDOW_FP16)
_THREE_LOOPS = ("gfx942", POISON_WINDOW_BF16)


_ONE = "0x3FF0000000000000"
# Edits of the lowered IR, each leaving the modeled path in one way, with the
# violation it must raise.
_EDITS = [
    (
        "sentinel is zero in one lane",
        _CAUSAL,
        _substitute_once(rf"(float %[\w.]+, float) {_F32_INITIAL_MAX}$", r"\1 0.0"),
        "unmodeled float select",
    ),
    (
        "causal bound excludes the diagonal key in one lane",
        _CAUSAL,
        _substitute_once(
            r"icmp sle (i32 %[\w.]+, %[\w.]+\n  %[\w.]+ = select i1 %[\w.]+, float)",
            r"icmp slt \1",
        ),
        "unmodeled float select",
    ),
    (
        "maximum reads a raw score",
        _CAUSAL,
        _maximum_reads_a_raw_score,
        "a score is read before its mask",
    ),
    (
        "weight reads a raw score",
        _CAUSAL,
        _weight_reads_a_raw_score,
        "a score is read before its mask",
    ),
    (
        "mask compare reads a score",
        _CAUSAL,
        _mask_compare_reads_a_score,
        "data-dependent icmp",
    ),
    (
        "one accumulator element is not rescaled",
        _CAUSAL,
        _substitute_once(r"(= fmul float %e\d+), %exp2f\d+$", rf"\1, {_ONE}"),
        "unmodeled fmul",
    ),
    (
        "normalizer is not rescaled",
        _CAUSAL,
        _substitute_once(r"(= fmul float %l), %exp2f\d+$", rf"\1, {_ONE}"),
        "unmodeled fmul",
    ),
    (
        "running maximum starts at zero",
        _CAUSAL,
        _substitute_once(
            rf"(%m = phi float \[) {_F32_INITIAL_MAX}", r"\1 0x0000000000000000"
        ),
        "unmodeled float phi",
    ),
    (
        "fast-math flag on a score",
        _CAUSAL,
        _substitute_once(r"= fmul float (%e\d+, %fmul\d+)$", r"= fmul fast float \1"),
        "fast-math flags",
    ),
    (
        "key loop stops one tile early",
        _CAUSAL,
        _substitute_once(
            r"(add nsw i32 %mul\d+), 2(\n  %lt\d+ = icmp slt i32 %add\d+, %div\d+)",
            r"\1, 1\2",
        ),
        "not left unmasked exactly once",
    ),
    (
        "window one key short",
        _WINDOW,
        _substitute_once(r"(= sub nsw i32 %add\d+), 64$", r"\1, 63"),
        "window bounds use",
    ),
    (
        "mask-free loop takes a tile",
        _THREE_LOOPS,
        _mask_free_loop_takes_a_tile,
        "mask-free key loop %km runs",
    ),
    (
        "maximum is not handed to the next loop",
        _THREE_LOOPS,
        _substitute_once(r"(%m_kl = phi float \[) %[\w.]+", rf"\1 {_F32_INITIAL_MAX}"),
        "unmodeled float phi",
    ),
]


@pytest.mark.parametrize(
    "subject_case,edit,match",
    [item[1:] for item in _EDITS],
    ids=[item[0] for item in _EDITS],
)
def test_path_check_rejects_each_unmodeled_change(
    subject_case: tuple[str, MaskedPoisonCase], edit, match: str
) -> None:
    arch, case = subject_case
    ir, _, scope = _lowered_subject(arch, case)
    edited = edit(ir)
    assert edited != ir
    violations = masked_path_violations(edited, case.dtype, **scope)
    assert any(match in item for item in violations), violations[:6]
    with pytest.raises(AssertionError, match=REDERIVE):
        verified_masked_path(edited, case.dtype, **scope)


def test_a_loop_that_only_adds_masked_tiles_is_not_a_violation() -> None:
    """Starting the window loop at tile 0 visits tiles in which every key is
    masked. The unmasked keys are unchanged, so the legal-set check passes; the
    relation, not this check, is what covers such tiles."""
    arch, case = _WINDOW
    ir, _, scope = _lowered_subject(arch, case)
    start = _first(r"^  %kt = phi i32 \[ (%[\w.]+), %entry \]", ir).group(1)
    edited = _substitute_once(
        rf"^(  {re.escape(start)} = select i1 %[\w.]+, i32) %[\w.]+, i32 0$",
        r"\1 0, i32 0",
    )(ir)
    assert not masked_path_violations(edited, case.dtype, **scope)


def test_index_evaluator_follows_llvm_integer_semantics() -> None:
    """Signed division truncates toward zero and the remainder takes the sign
    of the dividend; a value that is not position arithmetic is refused."""
    ir = """
define void @f(i32 %n, float %x) {
entry:
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %g = call i32 @llvm.amdgcn.workgroup.id.y()
  %a = sub nsw i32 %t, 3
  %q = sdiv i32 %a, 2
  %r = srem i32 %a, 2
  %c = icmp sle i32 %q, %g
  %s = select i1 %c, i32 %n, i32 -1
  %low = call i32 @llvm.smin.i32(i32 %s, i32 %r)
  %bad = fptosi float %x to i32
  ret void
}
"""
    _, named, _ = _parse_function(ir)
    evaluator = _IndexEvaluator(named, {"%n": 7}, (1, 2, 1), 6)
    assert evaluator.full("%q").tolist() == [[-1, -1, 0, 0, 0, 1]] * 2
    assert evaluator.full("%r").tolist() == [[-1, 0, -1, 0, 1, 0]] * 2
    assert evaluator.full("%s").tolist() == [[7, 7, 7, 7, 7, -1], [7] * 6]
    assert evaluator.full("%low").tolist() == [
        [-1, 0, -1, 0, 1, -1],
        [-1, 0, -1, 0, 1, 0],
    ]
    with pytest.raises(ValueError, match="does not model"):
        evaluator.value("%bad")
    steered = ir.replace("icmp sle i32 %q, %g", "icmp sle i32 %bad, %g")
    assert steering_violations(steered) == [
        "data-dependent icmp: icmp sle i32 %bad, %g",
        "data-dependent select: select i1 %c, i32 %n, i32 -1",
    ]


# ---------------------------------------------------------------------------
# What each launch exercises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,anchor,expected",
    [
        (
            POISON_CAUSAL_BF16,
            127,
            {
                (127, 131): "unvisited_tile",
                (127, 255): "unvisited_tile",
                (128, 131): "masked_lane",
                (128, 255): "masked_lane",
            },
        ),
        (
            POISON_WINDOW_FP16,
            127,
            {
                (127, 63): "tile_before_first_legal_key",
                (127, 131): "unvisited_tile",
                (128, 63): "unvisited_tile",
                (128, 131): "masked_lane",
            },
        ),
        (
            POISON_WINDOW_FP16,
            200,
            {(200, 136): "masked_lane", (200, 204): "masked_lane"},
        ),
        (
            POISON_WINDOW_BF16,
            127,
            {
                (127, 63): "tile_before_first_legal_key",
                (127, 131): "unvisited_tile",
                (128, 63): "unvisited_tile",
                (128, 131): "masked_lane",
            },
        ),
    ],
)
def test_anchors_exercise_every_isolation_mechanism(
    case: MaskedPoisonCase, anchor: int, expected: dict
) -> None:
    """At the shipped tiles the two anchors put a poisoned token in a tile a
    target row never visits, in a masked lane beside legal keys, and, with a
    window, in a tile that lies wholly before the row's first legal key."""
    ir, spec, scope = _lowered_subject("gfx942", case)
    path = classify_masked_path(
        ir, case.dtype, block_n=spec.block_n, head_size=spec.head_size
    )
    low, high = visited_tiles(ir, path, **_legal_scope(scope))
    found = poison_mechanisms(
        _fixture(case, anchor), block_n=int(spec.block_n), tile_low=low, tile_high=high
    )
    for key, mechanism in expected.items():
        assert found[key] is MaskMechanism(mechanism), key
    covered = {
        mechanism
        for each in POISON_ANCHORS
        for mechanism in poison_mechanisms(
            _fixture(case, each),
            block_n=int(spec.block_n),
            tile_low=low,
            tile_high=high,
        ).values()
    }
    wanted = {MaskMechanism.UNVISITED_TILE, MaskMechanism.MASKED_LANE}
    if case.mask == "window":
        wanted.add(MaskMechanism.TILE_BEFORE_FIRST_LEGAL_KEY)
    assert covered == wanted


def test_launch_geometry_is_the_launchers() -> None:
    """The position arithmetic is executed over the grid and block the dense
    launcher launches, read from the runner's own helpers."""
    from kernels.gfx942.attention_dense import (
        attention_dense_block,
        attention_dense_grid,
    )

    for arch, case in _ALL_SUBJECT_CASES:
        subject = resolve_dense_subject(arch, case)
        spec = subject.tuning_spec.kernel_spec
        grid, threads = launch_geometry(arch, subject)
        assert grid == tuple(attention_dense_grid(spec))
        assert threads == attention_dense_block(spec)[0] == 64 * (spec.block_m // 32)
        blocks = case.seqlen_q // spec.block_m
        assert math.prod(grid) == blocks * case.nhead_q * case.batch
