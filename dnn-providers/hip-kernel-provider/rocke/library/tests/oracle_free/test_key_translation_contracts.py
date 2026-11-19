# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the key-translation fixture, level structure and
analyzers."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from ._analytical import _ordered_array, decode_bits, encode_bits
from ._key_translation import (
    _FP32_UNIT,
    CONSTRUCTION_DIM,
    KEY_TRANSLATION_QUICK_CASES,
    KEY_TRANSLATION_SQUARE_BF16,
    KEY_TRANSLATION_SQUARE_FP16,
    LATTICE_SUM_LIMIT,
    LEVEL_GUARD,
    SWEEP_FACTORS,
    Baseline,
    Control,
    KeyTranslationCase,
    RowVerdict,
    Translation,
    _hash64,
    aligned_positions,
    analyze_control,
    analyze_sweep,
    baseline_key_numerators,
    baseline_levels,
    certify_score_lattice,
    classify_rows,
    compare_bits,
    contracted_levels,
    control_key_numerators,
    control_levels,
    designate_control,
    extreme_translation,
    flat_spectrum,
    format_bit_comparison,
    format_control_result,
    format_sweep_result,
    key_scale_exponents,
    keys_from_numerators,
    largest_logit,
    mixed_gap_translation,
    queries,
    query_scale_exponents,
    require_level_guard,
    score_gaps,
    score_scale_constant,
    sweep_translation,
    translated_key_numerators,
    translated_levels,
    unit_gap_signs,
    unit_gap_translation,
    value_ranges,
    values,
    walsh,
)

_SCALE = 0.125
# Residues ``x`` whose one-gap translation moves the row down: bit ``x`` set.
_UNIT_GAP_NEGATIVE = 0x0397E401D0588DE2
# A small shape for the scalar specifications of the analyzers.
_SMALL = KeyTranslationCase(1, 4, 4, 4, 2, 64, "fp16")


def _case_id(case: KeyTranslationCase) -> str:
    return case.case_id


def _translations() -> tuple[Translation, ...]:
    return (
        extreme_translation(),
        unit_gap_translation(),
        mixed_gap_translation(),
        *(sweep_translation(factor) for factor in SWEEP_FACTORS),
    )


def _launches(
    position_q: np.ndarray, position_k: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Key numerators and closed-form levels of every launch of the lane."""
    launches = {
        baseline.value: (
            baseline_key_numerators(position_k, baseline),
            baseline_levels(position_q, position_k, baseline),
        )
        for baseline in Baseline
    }
    for translation in _translations():
        launches[translation.name] = (
            translated_key_numerators(position_k, translation),
            translated_levels(position_q, position_k, translation),
        )
    for control in Control:
        launches[control.value] = (
            control_key_numerators(position_k, control),
            control_levels(position_q, position_k, control),
        )
    return launches


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_walsh_vectors_are_orthogonal_and_the_flat_vector_has_a_flat_spectrum() -> None:
    matrix, flat = walsh(), flat_spectrum()
    assert matrix.shape == (CONSTRUCTION_DIM, CONSTRUCTION_DIM)
    assert set(np.unique(matrix)) == {-1, 1}
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(matrix @ matrix.T, 64 * np.eye(64, dtype=np.int64))
    assert np.array_equal(matrix @ flat, 8 * flat)
    assert int(np.count_nonzero(flat > 0)) == 36
    assert int(np.count_nonzero(flat < 0)) == 28
    assert not matrix.flags.writeable and not flat.flags.writeable


def test_quick_case_head_scales_and_gaps() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    assert query_scale_exponents(case) == (-1, -2, -3, -4, -1, -2, -3, -4)
    assert key_scale_exponents(case) == (0, 1)
    assert score_gaps(case) == tuple(Fraction(g) for g in (32, 16, 8, 4, 64, 32, 16, 8))


def test_score_scale_constant_is_the_fp32_product_the_subject_forms() -> None:
    sigma = score_scale_constant(_SCALE)
    stored = np.float32(float(sigma))
    assert Fraction(float(stored)) == sigma
    assert int(stored.view(np.uint32)) == 0x3E38AA3B
    # A power-of-two launch scale leaves the significand of RN(log2 e).
    assert sigma * 8 == Fraction(float(np.float32(1.4426950408889634)))
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale must be positive and finite"):
            score_scale_constant(scale)


def test_unit_gap_signs_are_pinned_and_cover_both_parities() -> None:
    signs = unit_gap_signs()
    assert set(np.unique(signs)) == {-1, 1}
    mask = sum(1 << residue for residue in range(64) if signs[residue] < 0)
    assert mask == _UNIT_GAP_NEGATIVE
    classes = {(residue & 1, int(sign)) for residue, sign in enumerate(signs)}
    assert classes == {(0, -1), (0, 1), (1, -1), (1, 1)}
    # With the flat-spectrum signs the one-gap translation would be sweep_8.
    flat = flat_spectrum()
    assert not np.array_equal(signs, flat) and not np.array_equal(signs, -flat)
    assert unit_gap_translation().coefficients != sweep_translation(8).coefficients


def test_translations_shift_each_row_by_the_documented_amount() -> None:
    flat = flat_spectrum()
    residue = np.arange(64)
    assert extreme_translation().baseline is Baseline.ZERO
    assert np.array_equal(extreme_translation().row_shifts, 4096 * flat)
    assert unit_gap_translation().baseline is Baseline.WALSH
    assert np.array_equal(unit_gap_translation().row_shifts, 8 * unit_gap_signs())
    mixed = np.array(mixed_gap_translation().row_shifts)
    assert np.array_equal(np.abs(mixed), np.where((residue >> 1) & 1, 16, 4))
    assert np.all(mixed[residue % 2 == 0] < 0) and np.all(mixed[residue % 2 == 1] > 0)
    for factor in SWEEP_FACTORS:
        translation = sweep_translation(factor)
        assert translation.name == f"sweep_{factor}"
        assert translation.baseline is Baseline.WALSH
        assert np.array_equal(translation.row_shifts, factor * flat)


def test_translation_refuses_what_is_not_a_common_lattice_shift() -> None:
    with pytest.raises(ValueError, match="64 coefficients"):
        Translation("short", Baseline.WALSH, (0,) * 63)
    with pytest.raises(ValueError, match="must be integers"):
        Translation("fractional", Baseline.WALSH, (0.5,) + (0,) * 63)
    with pytest.raises(ValueError, match="not a whole number of eighths"):
        Translation("off_grid", Baseline.WALSH, (1,) + (0,) * 63)
    with pytest.raises(ValueError, match="sweep factor must be one of"):
        sweep_translation(3)


def test_key_numerators_stay_inside_the_derived_bounds() -> None:
    """In halves of the KV scale: at most 2 * 65 for one gap, 162 for the mixed
    gaps and 2 * 129 across the sweep."""
    _, position_k = aligned_positions(KEY_TRANSLATION_SQUARE_FP16)

    def largest(translation: Translation) -> int:
        return int(np.abs(translated_key_numerators(position_k, translation)).max())

    assert largest(unit_gap_translation()) == 42 <= 2 * 65
    assert largest(mixed_gap_translation()) == 158 <= 162
    assert max(largest(sweep_translation(f)) for f in SWEEP_FACTORS) == 258 == 2 * 129
    assert largest(extreme_translation()) == 1 << 13


def _shifted_positions() -> tuple[np.ndarray, np.ndarray]:
    """Two batch rows with unequal, nonzero sequence offsets."""
    return (
        np.stack([np.arange(96) + 37, np.arange(96) + 130]),
        np.stack([np.arange(80) + 5, np.arange(80) + 191]),
    )


@pytest.mark.parametrize(
    "positions",
    [aligned_positions(KEY_TRANSLATION_SQUARE_FP16), _shifted_positions()],
    ids=["aligned", "shifted"],
)
def test_closed_form_levels_equal_the_integer_contraction(positions) -> None:
    """The level a verdict is drawn from is the raw score of the stored keys,
    for every launch and for positions that are not local indices."""
    position_q, position_k = positions
    for name, (numerators, levels) in _launches(position_q, position_k).items():
        assert np.array_equal(contracted_levels(position_q, numerators), levels), name


def test_levels_follow_absolute_positions_not_local_indices() -> None:
    position_k = np.array([[0, 1, 64, 65]])
    walsh_levels = baseline_levels(np.array([[64, 65]]), position_k, Baseline.WALSH)
    assert walsh_levels.tolist() == [[[8, 0, 8, 0], [0, -8, 0, -8]]]
    assert not np.any(baseline_levels(np.array([[64]]), position_k, Baseline.ZERO))
    flat = flat_spectrum()
    alternating = control_levels(np.array([[3]]), position_k, Control.ALTERNATING_SIGN)
    assert alternating.tolist() == [[[4096 * s * int(flat[3]) for s in (1, -1, 1, -1)]]]
    gated = control_levels(
        np.array([[3]]), np.array([[0, 2, 3, 4]]), Control.BIT_ONE_GATE
    )
    assert gated.tolist() == [[[4096 * b * int(flat[3]) for b in (0, 1, 1, 0)]]]


def test_position_arguments_are_validated() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    with pytest.raises(ValueError, match=r"must be a \[B, S\] integer array"):
        baseline_levels(position_q.astype(np.float64), position_k, Baseline.WALSH)
    with pytest.raises(ValueError, match="non-negative absolute positions"):
        baseline_levels(position_q - 1, position_k, Baseline.WALSH)
    with pytest.raises(ValueError, match="share the batch dimension"):
        baseline_levels(position_q, np.tile(position_k, (2, 1)), Baseline.WALSH)
    with pytest.raises(ValueError, match="share the batch dimension"):
        control_levels(position_q, np.tile(position_k, (2, 1)), Control.BIT_ONE_GATE)
    with pytest.raises(ValueError, match="position_q shape"):
        queries(case, position_q[:, :-1])
    with pytest.raises(ValueError, match="position_k shape"):
        values(case, position_k[:, :-1])
    with pytest.raises(ValueError, match="do not match the query positions"):
        contracted_levels(position_q, np.zeros((2, 4, 64), dtype=np.int64))
    with pytest.raises(ValueError, match="not a whole number of eighths"):
        contracted_levels(position_q, np.eye(64, dtype=np.int64)[None, :4])


# ---------------------------------------------------------------------------
# Row classes
# ---------------------------------------------------------------------------


def _classes(
    translation: Translation, legal: np.ndarray | None = None, size: int = 256
):
    position = np.arange(size, dtype=np.int64)[None, :]
    return classify_rows(
        baseline_levels(position, position, translation.baseline),
        translated_levels(position, position, translation),
        legal,
    )


def test_exact_translations_hold_every_row_to_bit_equality() -> None:
    extreme = _classes(extreme_translation())
    assert extreme.counts() == {
        "one_level": 256,
        "dyadic": 0,
        "ceiling": 0,
        "two_level": 0,
    }
    for translation in (unit_gap_translation(), mixed_gap_translation()):
        classes = _classes(translation)
        assert classes.counts() == {
            "one_level": 0,
            "dyadic": 256,
            "ceiling": 0,
            "two_level": 256,
        }
        assert np.all(classes.exact) and np.all(classes.of(RowVerdict.DYADIC))


def test_translated_level_pairs_are_the_five_dyadic_pairs() -> None:
    position = np.arange(256, dtype=np.int64)[None, :]

    def pairs(translation: Translation) -> set[tuple[int, ...]]:
        levels = translated_levels(position, position, translation)[0]
        return {tuple(sorted(set(row.tolist()))) for row in levels}

    assert pairs(unit_gap_translation()) == {(8, 16), (-8, 0), (0, 8), (-16, -8)}
    assert pairs(mixed_gap_translation()) == {(-16, -8), (-4, 4), (8, 16)}
    assert pairs(extreme_translation()) == {(4096,), (-4096,)}
    assert pairs(sweep_translation(128)) == {
        (120, 128),
        (128, 136),
        (-136, -128),
        (-128, -120),
    }


def test_sweep_rows_are_exact_only_where_their_levels_are_dyadic() -> None:
    """All of sweep_8 is the one-gap structure. In sweep_4 and sweep_16 the rows
    with ``(-1)**q u_x == -1`` land on ``(-g/2, g/2)``, ``(-2g, -g)`` or
    ``(g, 2g)``. Every other sweep row has a level that is not a power of two."""
    query = np.arange(256)
    dyadic_rows = (1 - 2 * (query & 1)) * flat_spectrum()[query % 64] == -1
    assert int(np.count_nonzero(dyadic_rows)) == 112
    for factor in SWEEP_FACTORS:
        classes = _classes(sweep_translation(factor))
        assert np.all(classes.two_level) and not np.any(classes.one_level)
        if factor == 8:
            assert np.all(classes.dyadic)
        elif factor in (4, 16):
            assert np.array_equal(classes.dyadic[0], dyadic_rows)
            assert np.array_equal(classes.ceiling[0], ~dyadic_rows)
        else:
            assert np.all(classes.ceiling)


def test_row_classes_follow_the_legal_keys() -> None:
    """A row is two-level only when its legal set holds the congruent key and
    another residue; the classifier must say so instead of assuming it."""
    size = 128
    key = np.arange(size)[None, None, :]
    query = np.arange(size)[None, :, None]
    inclusive = key <= query
    classes = _classes(unit_gap_translation(), inclusive, size)
    assert classes.one_level[0, 0] and not classes.two_level[0, 0]
    assert np.all(classes.dyadic[0, 1:]) and np.all(classes.two_level[0, 1:])

    # Keys strictly before the query (row 0 keeps key 0): the congruent key is
    # legal only from row 64 on.
    before = (key < query) | ((query == 0) & (key == 0))
    classes = _classes(unit_gap_translation(), before, size)
    assert np.all(classes.one_level[0, 1:64]) and not np.any(classes.two_level[0, 1:64])
    assert np.all(classes.two_level[0, 64:]) and np.all(classes.dyadic[0, 64:])

    # A level that is not dyadic at an illegal key does not count.
    base = np.zeros((1, 1, 3), dtype=np.int64)
    other = np.array([[[8, 16, 24]]])
    assert classify_rows(base, other).ceiling[0, 0]
    legal = np.array([[[True, True, False]]])
    assert classify_rows(base, other, legal).dyadic[0, 0]


def test_classify_rows_refuses_what_it_cannot_classify() -> None:
    levels = np.zeros((1, 2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="a row has no legal key"):
        classify_rows(levels, levels, np.zeros((1, 2, 4), dtype=bool))
    with pytest.raises(ValueError, match="legal must be a boolean array"):
        classify_rows(levels, levels, np.ones((1, 2, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="legal must be a boolean array"):
        classify_rows(levels, levels, np.ones((1, 2, 3), dtype=bool))
    with pytest.raises(ValueError, match="two \\[B, Sq, Sk\\] arrays of one shape"):
        classify_rows(levels, levels[:, :1])
    with pytest.raises(ValueError, match="integer counts of eighths"):
        classify_rows(levels, levels.astype(np.float64))


# ---------------------------------------------------------------------------
# Stored tensors
# ---------------------------------------------------------------------------

_KEY_DIGESTS = {
    "zero": "fa43239bcee7b97c",
    "walsh": "f969d0b8750c358d",
    "extreme": "06b50cfdcb03ee51",
    "unit_gaps": "c3742a8c1eff8230",
    "mixed_gaps": "ddd8334ffcd7d980",
    "sweep_1": "c81af8163da68e15",
    "sweep_2": "2e63fe661ad1336b",
    "sweep_4": "38c0373352746503",
    "sweep_8": "29cb58b8850fa82c",
    "sweep_16": "909f49c9e46164ac",
    "sweep_32": "53b407584708e085",
    "sweep_64": "d8420667f9977b7b",
    "sweep_128": "5b4ed6a9181acf71",
    "alternating_sign": "221c90549caa2d37",
    "bit_one_gate": "5490985bbdfe6dc4",
}
# Largest summed product magnitude of one key, in quanta, per launch.
_LATTICE_SUMS = {
    "zero": 0,
    "walsh": 128,
    "extreme": 1 << 19,
    "unit_gaps": 864,
    "mixed_gaps": 380,
    "sweep_1": 144,
    "sweep_2": 272,
    "sweep_4": 528,
    "sweep_8": 1040,
    "sweep_16": 2064,
    "sweep_32": 4112,
    "sweep_64": 8208,
    "sweep_128": 16400,
    "alternating_sign": 1 << 19,
    "bit_one_gate": 1 << 19,
}


@pytest.mark.parametrize("case", KEY_TRANSLATION_QUICK_CASES, ids=_case_id)
def test_fixture_is_stored_exactly_and_stays_pinned(case: KeyTranslationCase) -> None:
    """Every tensor of every launch encodes without rounding in the subject
    dtype, does not depend on it, and is on the exact-score lattice."""
    position_q, position_k = aligned_positions(case)
    q = queries(case, position_q)
    v = values(case, position_k)
    assert (_digest(q), _digest(v)) == ("2beaf3eb81d72685", "30c664e5e0ba780d")
    for tensor in (q, v):
        bits = encode_bits(tensor, case.dtype)
        assert np.array_equal(decode_bits(bits, case.dtype), tensor.astype(np.float64))
    launches = _launches(position_q, position_k)
    assert set(launches) == set(_KEY_DIGESTS) == set(_LATTICE_SUMS)
    for name, (numerators, _levels) in launches.items():
        k = keys_from_numerators(case, numerators)
        assert _digest(k) == _KEY_DIGESTS[name], name
        bits = encode_bits(k, case.dtype)
        assert np.array_equal(decode_bits(bits, case.dtype), k.astype(np.float64))
        assert certify_score_lattice(case, q, k) == _LATTICE_SUMS[name], name
    assert max(_LATTICE_SUMS.values()) * 32 == LATTICE_SUM_LIMIT


def test_queries_and_values_have_the_constructed_structure() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    q = queries(case, position_q)
    alpha = np.ldexp(1.0, np.array(query_scale_exponents(case)))
    assert np.array_equal(np.abs(q), np.broadcast_to(alpha[:, None], q.shape))
    assert np.array_equal(np.sign(q[0, :, 3, :]), walsh()[position_q[0] % 64])
    v = values(case, position_k)
    assert np.array_equal(v[:, :, 0, :], v[:, :, 1, :])
    assert np.array_equal(v[0, :, 0, 0], position_k[0] % 2 == 0)
    assert np.array_equal(v[0, :, 0, 1], (position_k[0] >> 1) & 1)
    assert np.array_equal(v[0, :, 0, 2:], walsh()[position_k[0] % 64][:, 2:])
    ranges = value_ranges(case, v)
    assert ranges.shape == (case.nhead_q, case.head_dim)
    assert np.all(ranges[:, :2] == 1.0) and np.all(ranges[:, 2:] == 2.0)
    with pytest.raises(ValueError, match="V must be a finite array"):
        value_ranges(case, v[:, :-1])


def test_a_translation_the_dtype_cannot_store_is_refused_by_the_encoder() -> None:
    """Numerators of 4 * 511 need nine significant bits and 4 * 4095 twelve."""
    case = KEY_TRANSLATION_SQUARE_FP16
    _, position_k = aligned_positions(case)
    flat = flat_spectrum()

    def keys(scale: int) -> np.ndarray:
        translation = Translation(
            "too_fine", Baseline.WALSH, tuple(int(item) for item in scale * flat)
        )
        return keys_from_numerators(
            case, translated_key_numerators(position_k, translation)
        )

    encode_bits(keys(2046), "fp16")
    with pytest.raises(ValueError, match="not exactly representable in bf16"):
        encode_bits(keys(2046), "bf16")
    with pytest.raises(ValueError, match="not exactly representable in fp16"):
        encode_bits(keys(16382), "fp16")


def test_keys_from_numerators_validates_its_input() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    with pytest.raises(ValueError, match="integer array of shape"):
        keys_from_numerators(case, np.zeros((1, 256, 64)))
    with pytest.raises(ValueError, match="integer array of shape"):
        keys_from_numerators(case, np.zeros((1, 255, 64), dtype=np.int64))
    with pytest.raises(ValueError, match="not exact in FP32"):
        keys_from_numerators(case, np.full((1, 256, 64), 1 << 24, dtype=np.int64))


def test_score_lattice_refuses_what_it_cannot_certify() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    q = queries(case, position_q)
    k = keys_from_numerators(
        case, baseline_key_numerators(position_k, Baseline.WALSH)
    )

    doubled = q.copy()
    doubled[0, 5, 2, 7] *= 2
    with pytest.raises(ValueError, match="a Q element is not \\+-alpha_h"):
        certify_score_lattice(case, doubled, k)
    off = k.copy()
    off[0, 9, 0, 3] += 0.25
    with pytest.raises(ValueError, match="not a multiple of beta_j / 2"):
        certify_score_lattice(case, q, off)
    # Half of beta_1 is a lattice point of KV head 1 but not of KV head 0.
    shifted = k.copy()
    shifted[0, 9, 1, 3] += 1.0
    certify_score_lattice(case, q, shifted)
    for bad in (np.nan, np.inf):
        broken = k.copy()
        broken[0, 0, 0, 0] = bad
        with pytest.raises(ValueError, match="Q or K is not finite"):
            certify_score_lattice(case, q, broken)
    with pytest.raises(ValueError, match="Q and K shapes"):
        certify_score_lattice(case, q, k[:, :-1])
    large = np.full(case.key_shape, 2.0**18, dtype=np.float32)
    with pytest.raises(ValueError, match="beyond 16777216"):
        certify_score_lattice(case, q, large)
    at_limit = np.full(case.key_shape, 2.0**17, dtype=np.float32)
    assert certify_score_lattice(case, q, at_limit) == LATTICE_SUM_LIMIT


# ---------------------------------------------------------------------------
# Level guard and scope
# ---------------------------------------------------------------------------


def test_every_launch_stays_inside_the_level_guard() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    largest = {
        name: require_level_guard(case, levels, _SCALE)
        for name, (_numerators, levels) in _launches(position_q, position_k).items()
    }
    assert largest["zero"] == 0.0
    assert max(largest.values()) == largest["extreme"] == largest["alternating_sign"]
    assert 5909.27 < largest["extreme"] < 5909.29 < LEVEL_GUARD
    assert 196.20 < largest["sweep_128"] < 196.21


def test_largest_logit_is_not_below_its_rational_value() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    for count in (1, 7, 136, 4096):
        levels = np.array([[[0, -count]]])
        exact = score_scale_constant(_SCALE) * count * 64 / 8 * (1 + _FP32_UNIT)
        found = largest_logit(case, levels, _SCALE)
        assert exact <= Fraction(found) < exact * (1 + Fraction(1, 1 << 50))
    with pytest.raises(ValueError, match="non-empty integer array"):
        largest_logit(case, np.zeros((1, 1, 1)), _SCALE)
    with pytest.raises(ValueError, match="non-empty integer array"):
        largest_logit(case, np.zeros((1, 0, 1), dtype=np.int64), _SCALE)


def test_level_guard_refuses_a_launch_near_the_sentinel() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    with pytest.raises(ValueError, match="beyond the level guard 8192"):
        require_level_guard(case, np.array([[[8192]]]), _SCALE)
    # The same levels pass at a smaller launch scale.
    assert require_level_guard(case, np.array([[[8192]]]), _SCALE / 2) < LEVEL_GUARD


def test_case_scope_refuses_what_the_construction_does_not_cover() -> None:
    for case in KEY_TRANSLATION_QUICK_CASES:
        case.require_scope()
    with pytest.raises(ValueError, match="proven for head_dim 64 only"):
        replace(KEY_TRANSLATION_SQUARE_FP16, head_dim=128).require_scope()
    # Four KV heads: beta reaches 8 and the extreme translation 2**9 * 256.
    wide = KeyTranslationCase(1, 256, 256, 16, 4, 64, "fp16")
    assert max(score_gaps(wide)) == 256
    with pytest.raises(ValueError, match="beyond the level guard"):
        wide.require_scope()
    with pytest.raises(ValueError, match="beyond the level guard"):
        queries(wide, aligned_positions(wide)[0])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"batch": 0}, "batch must be positive"),
        ({"seqlen_q": -1}, "seqlen_q must be positive"),
        ({"seqlen_k": 0}, "seqlen_k must be positive"),
        ({"head_dim": 0}, "head_dim must be positive"),
        ({"nhead_k": 3}, "nhead_q must be a multiple of nhead_k"),
        ({"dtype": "fp32"}, "dtype must be one of"),
    ],
)
def test_case_validates_its_shape(override, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(KEY_TRANSLATION_SQUARE_FP16, **override)


def test_quick_cases_are_the_unmasked_square_gqa_shape() -> None:
    assert KEY_TRANSLATION_QUICK_CASES == (
        KeyTranslationCase(1, 256, 256, 8, 2, 64, "fp16"),
        KeyTranslationCase(1, 256, 256, 8, 2, 64, "bf16"),
    )
    assert KEY_TRANSLATION_SQUARE_BF16.case_id == "bf16-none-b1-sq256-sk256-hq8-hk2-d64"
    for case in KEY_TRANSLATION_QUICK_CASES:
        assert (case.mask, case.window, case.group) == ("none", 0, 4)
        assert case.output_shape == (1, 256, 8, 64)
        assert case.key_shape == (1, 256, 2, 64)


# ---------------------------------------------------------------------------
# Bit-equality analyzer
# ---------------------------------------------------------------------------


def _constant_output(case: KeyTranslationCase, value: float = 0.5) -> np.ndarray:
    return encode_bits(np.full(case.output_shape, value, dtype=np.float32), case.dtype)


def _planted_outputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A finite FP16 output of the small shape with hashed values and some
    zeros; a copy with low-bit changes at every seventh element, the sign of
    every other zero flipped, and a few quiet NaNs; and a row selection that
    leaves one row out."""
    case = _SMALL
    count = int(np.prod(case.output_shape))
    index = np.arange(count)
    draw = _hash64(count, seed)
    base = ((draw & np.uint64(0x3FFF)) + np.uint64(0x0400)).astype(np.uint16)
    base[index % 11 == 0] = 0
    other = base.copy()
    change = index % 7 == 3
    other[change] ^= (index[change] % 3 + 1).astype(np.uint16)
    other[index % 22 == 0] ^= np.uint16(0x8000)
    other[index % 97 == 5] = np.uint16(0x7E00)
    rows = np.array([[True, True, False, True]])
    return base.reshape(case.output_shape), other.reshape(case.output_shape), rows


def _is_finite(bits: int) -> bool:
    return bits & 0x7C00 != 0x7C00


@pytest.mark.parametrize("case", KEY_TRANSLATION_QUICK_CASES, ids=_case_id)
def test_bit_comparison_accepts_only_identical_bits(case: KeyTranslationCase) -> None:
    base = _constant_output(case)
    same = compare_bits(case, base, base.copy())
    assert same.violations == () and same.compared_rows == 256
    assert same.first_mismatch is None and same.first_bits is None

    other = base.copy()
    other[0, 3, 5, 7] ^= 1
    other[0, 9, 5, 0] ^= 1
    other[0, 200, 1, 63] ^= 0x10
    found = compare_bits(case, base, other)
    assert found.violations == ("bit_mismatch=3",)
    assert found.head_mismatch_counts == ((0, 1, 0, 0, 0, 2, 0, 0),)
    assert found.first_mismatch == (0, 3, 5, 7)
    assert found.first_bits == (int(base[0, 3, 5, 7]), int(base[0, 3, 5, 7]) ^ 1)
    assert "0:3:5:7" in format_bit_comparison(found)


def test_bit_comparison_does_not_identify_signed_zeros() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    base = np.zeros(case.output_shape, dtype=np.uint16)
    other = base.copy()
    other[0, 1, 2, 3] = 0x8000
    found = compare_bits(case, base, other)
    assert found.violations == ("bit_mismatch=1",)
    assert (found.mismatch_count, found.zero_sign_count) == (1, 1)


@pytest.mark.parametrize("case", KEY_TRANSLATION_QUICK_CASES, ids=_case_id)
def test_bit_comparison_rejects_unwritten_and_non_finite_outputs(
    case: KeyTranslationCase,
) -> None:
    base = _constant_output(case)
    nan = {"fp16": 0x7E00, "bf16": 0x7FC0}[case.dtype]
    infinity = {"fp16": 0x7C00, "bf16": 0x7F80}[case.dtype]
    both = base.copy()
    both[0, 0, 0, 0] = nan
    # The same NaN in both outputs is equal bit for bit and still invalid.
    assert compare_bits(case, both, both.copy()).violations == ("invalid=2",)
    other = base.copy()
    other[0, 4, 4, 4] = infinity
    found = compare_bits(case, base, other)
    assert found.violations == ("invalid=1", "bit_mismatch=1")


def test_bit_comparison_judges_only_the_selected_rows() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    other = base.copy()
    other[0, 10, 0, 0] ^= 1
    rows = np.ones((1, 256), dtype=bool)
    rows[0, 10] = False
    assert compare_bits(case, base, other, rows).violations == ()
    assert compare_bits(case, base, other, ~rows).violations == ("bit_mismatch=1",)
    # A non-finite element is a violation on any row.
    other[0, 10, 0, 1] = 0x7E00
    assert compare_bits(case, base, other, rows).violations == ("invalid=1",)
    nothing = compare_bits(case, base, base, np.zeros((1, 256), dtype=bool))
    assert nothing.violations == ("no_row_compared",)


def test_analyzers_validate_their_outputs_and_rows() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    with pytest.raises(ValueError, match="base output must contain raw uint16 bits"):
        compare_bits(case, base.astype(np.float32), base)
    with pytest.raises(ValueError, match="other output shape"):
        compare_bits(case, base, base[:, :-1])
    with pytest.raises(ValueError, match="rows must be a boolean array"):
        compare_bits(case, base, base, np.ones((1, 256), dtype=np.int64))
    with pytest.raises(ValueError, match="rows must be a boolean array"):
        analyze_sweep(case, base, base, 1.0, 1.0, np.ones((1, 255), dtype=bool))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bit_comparison_matches_its_scalar_specification(seed: int) -> None:
    case = _SMALL
    base, other, rows = _planted_outputs(seed)
    found = compare_bits(case, base, other, rows)

    invalid = mismatch = zero_sign = 0
    heads = [0] * case.nhead_q
    first = None
    for index in np.ndindex(case.output_shape):
        left, right = int(base[index]), int(other[index])
        invalid += (not _is_finite(left)) + (not _is_finite(right))
        if not rows[index[0], index[1]] or left == right:
            continue
        mismatch += 1
        heads[index[2]] += 1
        zero_sign += (left | right) & 0x7FFF == 0
        first = first or tuple(int(item) for item in index)
    assert mismatch and zero_sign and invalid
    assert found.invalid_count == invalid
    assert found.mismatch_count == mismatch
    assert found.zero_sign_count == zero_sign
    assert found.head_mismatch_counts == (tuple(heads),)
    assert found.first_mismatch == first
    assert found.compared_rows == int(np.count_nonzero(rows))


# ---------------------------------------------------------------------------
# Sweep analyzer
# ---------------------------------------------------------------------------


def test_sweep_analyzer_separates_residuals_at_the_ceiling() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    other = base.copy()
    other[0, 3, 5, 7] += 1
    step = 2.0**-11
    at = analyze_sweep(case, base, other, step, 1.0)
    assert at.violations == () and at.max_abs == step and at.exceed_count == 0
    assert (at.max_ulp, at.over_one_ulp_count) == (1, 0)
    assert at.exact_hit_rate == 1 - 1 / base.size
    assert at.worst_coordinate == (0, 3, 5, 7) and at.max_ceiling_ratio == 1.0
    below = analyze_sweep(case, base, other, np.nextafter(step, 0.0), 1.0)
    assert below.violations == (f"residual={step:.9g} beyond the ceiling at 1",)
    assert "beyond the ceiling" in format_sweep_result(below)


def test_sweep_analyzer_reports_what_passes_but_needs_explaining() -> None:
    """Two units in the last place are inside a four-unit ceiling and are still
    counted: a correct kernel is expected at one or below."""
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    other = base.copy()
    other[0, 0, 2, 1] += 2
    other[0, 1, 6, 9] -= 1
    ceiling = np.full((case.nhead_q, case.head_dim), 2.0e-3)
    ranges = np.ones((case.nhead_q, case.head_dim))
    ranges[:, 9] = 0.0
    found = analyze_sweep(case, base, other, ceiling, ranges)
    assert found.violations == ()
    assert (found.max_ulp, found.over_one_ulp_count) == (2, 1)
    assert found.head_max_ulp == ((0, 0, 2, 0, 0, 0, 1, 0),)
    assert found.head_max_abs[0][2] == 2.0**-10
    # Below 0.5 the spacing halves, and a zero value range is left out.
    assert found.head_max_abs[0][6] == 2.0**-12
    assert found.max_normalized == 2.0**-10


def test_sweep_analyzer_rejects_invalid_outputs_and_ceilings() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    other = base.copy()
    other[0, 0, 0, 0] = 0x7E00
    assert analyze_sweep(case, base, other, 1.0, 1.0).violations == ("invalid=1",)
    rows = np.zeros((1, 256), dtype=bool)
    empty = analyze_sweep(case, base, base, 1.0, 1.0, rows)
    assert empty.violations == ("no_coordinate_compared",)
    assert empty.worst_coordinate is None and empty.exact_hit_rate == 0.0
    for ceiling in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ceiling must be positive and finite"):
            analyze_sweep(case, base, base, ceiling, 1.0)
    for spread in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="value range must be non-negative"):
            analyze_sweep(case, base, base, 1.0, spread)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_sweep_analyzer_matches_its_scalar_specification(seed: int) -> None:
    case = _SMALL
    base, other, rows = _planted_outputs(seed)
    ceiling = 3.0e-4 * (1 + np.arange(case.nhead_q))[:, None] * np.ones(case.head_dim)
    ranges = np.ones((case.nhead_q, case.head_dim))
    ranges[:, ::5] = 0.0
    found = analyze_sweep(case, base, other, ceiling, ranges, rows)

    invalid = compared = exceed = hits = over = 0
    max_abs = max_normalized = 0.0
    max_ulp = 0
    for index in np.ndindex(case.output_shape):
        left, right = int(base[index]), int(other[index])
        invalid += (not _is_finite(left)) + (not _is_finite(right))
        if not (rows[index[0], index[1]] and _is_finite(left) and _is_finite(right)):
            continue
        compared += 1
        hits += left == right
        residual = abs(
            float(decode_bits(left, case.dtype)) - float(decode_bits(right, case.dtype))
        )
        distance = abs(
            int(_ordered_array(np.uint16(left))) - int(_ordered_array(np.uint16(right)))
        )
        exceed += residual > ceiling[index[2], index[3]]
        over += distance > 1
        max_abs = max(max_abs, residual)
        max_ulp = max(max_ulp, distance)
        if ranges[index[2], index[3]] > 0.0:
            max_normalized = max(max_normalized, residual / ranges[index[2], index[3]])
    assert exceed and over and hits and invalid
    assert found.invalid_count == invalid
    assert found.exceed_count == exceed
    assert found.over_one_ulp_count == over
    assert found.max_abs == max_abs
    assert found.max_ulp == max_ulp
    assert found.max_normalized == max_normalized
    assert found.exact_hit_rate == hits / compared


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _quick_designation(control: Control, forward_error=1.0e-2, legal=None):
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    return designate_control(
        case,
        control,
        control_levels(position_q, position_k, control),
        values(case, position_k),
        forward_error,
        legal,
    )


@pytest.mark.parametrize("control", list(Control))
def test_every_row_and_head_of_the_quick_fixture_is_designated(
    control: Control,
) -> None:
    designation = _quick_designation(control)
    assert designation.coordinate == control.coordinate
    assert designation.head_counts == (256,) * 8
    assert np.all(designation.high_count == 128)
    assert np.all(designation.legal_count == 256)
    # Half the keys carry the indicator; the separation is rounded toward zero.
    assert np.all(designation.separation == np.nextafter(0.5, 0.0))
    assert designation.forward_error == (1.0e-2,) * 8


def test_designation_needs_four_forward_errors_of_separation() -> None:
    control = Control.ALTERNATING_SIGN
    assert _quick_designation(control, 0.1249).head_counts == (256,) * 8
    with pytest.raises(ValueError, match=r"query heads \[0, 1, 2, 3, 4, 5, 6, 7\]"):
        _quick_designation(control, 0.125)
    errors = np.full(8, 1.0e-2)
    errors[3] = 0.2
    with pytest.raises(ValueError, match=r"query heads \[3\] have no designated row"):
        _quick_designation(control, errors)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="forward error must be positive"):
            _quick_designation(control, bad)


def test_one_key_rows_are_never_control_witnesses() -> None:
    legal = np.ones((1, 256, 256), dtype=bool)
    legal[0, 7, :] = False
    legal[0, 7, 7] = True
    # Row 9 keeps only even keys: all of them at one level under the sign control.
    legal[0, 9, 1::2] = False
    designation = _quick_designation(Control.ALTERNATING_SIGN, legal=legal)
    assert not np.any(designation.designated[0, 7])
    assert designation.high_count[0, 7] == 1
    assert not np.any(designation.designated[0, 9])
    assert designation.head_counts == (254,) * 8


def test_designation_counts_the_weight_the_lower_keys_keep() -> None:
    """With the two key groups one eighth of a score gap apart, the lower
    group keeps ``exp(-g_h / 64)`` of the weight per key. Only the head with
    the largest gap, ``g_h = 64``, still separates enough."""
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    levels = control_levels(position_q, position_k, Control.BIT_ONE_GATE)
    # Levels 1 on the keys with bit 1 set and 0 elsewhere, in every row.
    shallow = np.sign(levels) * np.sign(levels[:, :, 2:3])
    assert set(np.unique(shallow)) == {0, 1}
    with pytest.raises(ValueError, match=r"query heads \[0, 1, 2, 3, 5, 6, 7\]"):
        designate_control(
            case, Control.BIT_ONE_GATE, shallow, values(case, position_k), 1.0e-3
        )


def test_designation_validates_its_inputs() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    position_q, position_k = aligned_positions(case)
    levels = control_levels(position_q, position_k, Control.BIT_ONE_GATE)
    v = values(case, position_k)
    control = Control.BIT_ONE_GATE
    with pytest.raises(ValueError, match=r"\[B, Sq, Sk\] integer array"):
        designate_control(case, control, levels.astype(np.float64), v, 1.0e-2)
    with pytest.raises(ValueError, match="levels do not match the case shape"):
        designate_control(case, control, levels[:, :-1], v, 1.0e-2)
    with pytest.raises(ValueError, match="V must be a finite array"):
        designate_control(case, control, levels, v[:, :-1], 1.0e-2)
    with pytest.raises(ValueError, match="a row has no legal key"):
        designate_control(
            case, control, levels, v, 1.0e-2, np.zeros(levels.shape, dtype=bool)
        )


def _control_outputs(control: Control) -> tuple[np.ndarray, np.ndarray]:
    """A baseline at 1/2 and a control that reaches 1 on its coordinate."""
    case = KEY_TRANSLATION_SQUARE_FP16
    base = _constant_output(case)
    other = base.copy()
    other[..., control.coordinate] = _constant_output(case, 1.0)[..., 0]
    return base, other


@pytest.mark.parametrize("control", list(Control))
def test_control_must_leave_the_baseline_on_every_designated_pair(
    control: Control,
) -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    designation = _quick_designation(control)
    base, other = _control_outputs(control)
    passed = analyze_control(case, base, other, designation)
    assert passed.violations == () and passed.designated_count == 256 * 8
    assert passed.min_separation == 0.5 and passed.first_unseparated is None

    # The subject ignores the token-dependent change in query heads 2 and 6.
    other[0, 40:, 2, :] = base[0, 40:, 2, :]
    other[0, 5, 6, :] = base[0, 5, 6, :]
    failed = analyze_control(case, base, other, designation)
    assert failed.violations == ("unseparated=217 in query heads [2, 6]",)
    assert failed.head_unseparated_counts == (0, 0, 216, 0, 0, 0, 1, 0)
    assert failed.first_unseparated == (0, 5, 6)
    assert "b:q:h=0:5:6" in format_control_result(failed)


def test_control_separation_is_strict_and_reads_its_own_coordinate() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    control = Control.BIT_ONE_GATE
    # 2 eps = 2**-11: one FP16 step at 1/2 is not more than two forward errors.
    designation = _quick_designation(control, 2.0**-12)
    base = _constant_output(case)
    other = base.copy()
    other[..., control.coordinate] += 1
    assert analyze_control(case, base, other, designation).unseparated_count == 256 * 8
    other[..., control.coordinate] += 1
    assert analyze_control(case, base, other, designation).violations == ()
    # A change on another coordinate is not a separation.
    elsewhere = base.copy()
    elsewhere[..., 0] += 100
    assert analyze_control(case, base, elsewhere, designation).unseparated_count == 2048


def test_control_analyzer_rejects_invalid_outputs_and_foreign_designations() -> None:
    case = KEY_TRANSLATION_SQUARE_FP16
    control = Control.ALTERNATING_SIGN
    designation = _quick_designation(control)
    base, other = _control_outputs(control)
    other[0, 3, 4, control.coordinate] = 0x7E00
    found = analyze_control(case, base, other, designation)
    assert found.violations == ("invalid=1", "unseparated=1 in query heads [4]")
    with pytest.raises(ValueError, match="designation does not match the case shape"):
        analyze_control(
            _SMALL, _constant_output(_SMALL), _constant_output(_SMALL), designation
        )
    nothing = replace(designation, designated=np.zeros_like(designation.designated))
    found = analyze_control(case, base, base, nothing)
    assert found.violations == ("no_pair_designated",)
