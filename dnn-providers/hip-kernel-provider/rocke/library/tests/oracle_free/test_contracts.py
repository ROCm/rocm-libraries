# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the uniform-probability checker, the subject adapter and
the suite's source policy."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    _FORMAT,
    UNIFORM_CAUSAL_BF16,
    UNIFORM_FOCUSED_ROWS,
    UNIFORM_QUICK_CASES,
    UNIFORM_WINDOW_FP16,
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformResult,
    _basis_offset,
    _freeze_groups,
    _legal_bounds,
    _MutableNqStats,
    _ordered_array,
    _proven_limit,
    analyze_uniform_policy,
    broadcast_query_head,
    correct_uniform_observation,
    decode_bits,
    encode_bits,
    format_uniform_result,
    legal_interval,
    legal_keys,
    max_legal_count,
    require_proven_scope,
    residue_count,
    target_bits,
    ulp_distance,
    uniform_basis_offsets,
    uniform_expected_bits,
    uniform_inputs,
    uniform_inputs_folded,
)
from ._dense_subject import (
    DENSE_SUBJECTS,
    QUICK_SUBJECT_CASES,
    _resolve_tuning_spec,
    _validate_device_buffers,
    _validate_host_bits,
    _validate_supported_case,
    _verify_resolved_spec,
    resolve_dense_subject,
)
from .test_analytical_dense import _not_rejected


def _small_case(mask: str, *, window: int = 0, seqlen_k: int = 8) -> UniformCase:
    return UniformCase(1, 8, seqlen_k, 2, 2, 4, "fp16", mask, window)


def _replace_bits(
    observation: UniformObservation,
    exposure: str,
    coordinate: tuple[int, int, int, int],
    bits: int,
) -> UniformObservation:
    outputs: list[tuple[str, np.ndarray]] = []
    for name, values in observation.exposures:
        copy = values.copy()
        if name == exposure:
            copy[coordinate] = np.uint16(bits)
        outputs.append((name, copy))
    return UniformObservation(tuple(outputs))


def _fraction_from_bits(bits: int, dtype: str) -> Fraction:
    fraction_bits, exponent_bits, bias = {
        "fp16": (10, 5, 15),
        "bf16": (7, 8, 127),
    }[dtype]
    exponent_mask = (1 << exponent_bits) - 1
    exponent = (bits >> fraction_bits) & exponent_mask
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == 0:
        significand = fraction
        power = 1 - bias - fraction_bits
    else:
        significand = (1 << fraction_bits) | fraction
        power = exponent - bias - fraction_bits
    return (
        Fraction(significand << power, 1)
        if power >= 0
        else Fraction(significand, 1 << -power)
    )


def test_legal_key_sets_pin_causal_window_and_mutation_semantics() -> None:
    causal = _small_case("causal")
    window = _small_case("window", window=4)

    assert legal_keys(causal, 0) == (0,)
    assert legal_keys(causal, 7) == tuple(range(8))
    assert legal_keys(causal, 3, UniformMutation.INCLUDE_FUTURE_KEY) == (0, 1, 2, 3, 4)
    assert legal_keys(causal, 3, UniformMutation.EXCLUDE_LAST_KEY) == (0, 1, 2)
    assert legal_keys(window, 4) == (1, 2, 3, 4)
    assert legal_keys(window, 4, UniformMutation.INCLUDE_OLD_WINDOW_KEY) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert legal_keys(window, 4, UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY) == (2, 3, 4)


def _mutations_for(case: UniformCase) -> tuple[UniformMutation | None, ...]:
    if case.mask == "causal":
        return (
            None,
            UniformMutation.INCLUDE_FUTURE_KEY,
            UniformMutation.EXCLUDE_LAST_KEY,
            UniformMutation.DENOMINATOR_PLUS_ONE,
        )
    return (
        None,
        UniformMutation.INCLUDE_OLD_WINDOW_KEY,
        UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
        UniformMutation.DENOMINATOR_PLUS_ONE,
    )


@pytest.mark.parametrize(
    "case",
    [
        *UNIFORM_QUICK_CASES,
        _small_case("causal"),
        _small_case("causal", seqlen_k=5),
        _small_case("window", window=1),
        _small_case("window", window=3),
        UniformCase(1, 12, 6, 2, 2, 4, "fp16", "window", 4),
    ],
    ids=lambda case: case.case_id,
)
def test_vectorized_legal_bounds_match_the_scalar_rule(case: UniformCase) -> None:
    for mutation in _mutations_for(case):
        first, last = _legal_bounds(case, mutation)
        scalar = [
            legal_interval(case, query, mutation) for query in range(case.seqlen_q)
        ]
        assert list(zip(first.tolist(), last.tolist())) == scalar


def test_proven_scope_admits_the_quick_cases_and_refuses_beyond_the_theorem() -> None:
    for case in UNIFORM_QUICK_CASES:
        require_proven_scope(case)
        assert max_legal_count(case) < _proven_limit(case.dtype, "folded")
    assert max_legal_count(UNIFORM_WINDOW_FP16) == 64
    assert max_legal_count(UNIFORM_CAUSAL_BF16) == 256
    # FP16 folded limit: 2**11 * 2 // 3 = 1365 legal keys.
    require_proven_scope(UniformCase(1, 1364, 1364, 1, 1, 64, "fp16", "causal"))
    with pytest.raises(ValueError, match="outside the proven zero-ULP range"):
        require_proven_scope(UniformCase(1, 1365, 1365, 1, 1, 64, "fp16", "causal"))
    # A window bounds N_q, so a long sequence stays inside the range.
    require_proven_scope(UniformCase(1, 4096, 4096, 1, 1, 64, "fp16", "window", 64))


@pytest.mark.parametrize(
    "dtype,numerator,denominator,expected",
    [
        ("fp16", 1, 3, 0x3555),
        ("bf16", 1, 3, 0x3EAB),
        ("fp16", 1, 128, 0x2000),
        ("bf16", 1, 128, 0x3C00),
        ("fp16", 1, 129, 0x1FF0),
        ("bf16", 1, 129, 0x3BFE),
        ("fp16", 1, 32768, 0x0200),
    ],
)
def test_target_bits_pinned_values(
    dtype: str, numerator: int, denominator: int, expected: int
) -> None:
    assert target_bits(numerator, denominator, dtype) == expected


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_target_bits_are_nearest_with_ties_to_even(dtype: str) -> None:
    for denominator in range(1, 4097):
        bits = target_bits(1, denominator, dtype)
        exact = Fraction(1, denominator)
        error = abs(_fraction_from_bits(bits, dtype) - exact)
        neighbors = [candidate for candidate in (bits - 1, bits + 1) if candidate >= 0]
        neighbor_errors = [
            abs(_fraction_from_bits(candidate, dtype) - exact)
            for candidate in neighbors
        ]
        assert all(error <= neighbor for neighbor in neighbor_errors)
        if any(error == neighbor for neighbor in neighbor_errors):
            assert bits & 1 == 0


def test_ulp_distance_coalesces_signed_zero() -> None:
    assert ulp_distance(0x0000, 0x8000) == 0
    assert ulp_distance(0x3C00, 0x3C01) == 1
    assert ulp_distance(0xBC00, 0xBC01) == 1


def test_uniform_generators_cover_basis_and_folded_profiles() -> None:
    case = UNIFORM_CAUSAL_BF16
    assert uniform_basis_offsets(case) == (0, 64, 128, 192)
    exposed: list[int] = []
    for offset in uniform_basis_offsets(case):
        q, k, v = uniform_inputs(case, offset)
        assert not np.any(q)
        assert np.isfinite(k).all()
        assert np.unique(k).size > 1
        locations = np.argwhere(v[0, :, 0, :] == 1.0)
        exposed.extend(int(key) for key, _ in locations)
        assert all(key == offset + dim for key, dim in locations)
    assert exposed == list(range(256))

    q, k, folded = uniform_inputs_folded(case)
    assert not np.any(q)
    assert np.isfinite(k).all()
    locations = np.argwhere(folded[0, :, 0, :] == 1.0)
    assert len(locations) == case.seqlen_k
    assert all(dim == key % case.head_dim for key, dim in locations)


@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_v_profiles_scale_each_kv_head_by_a_distinct_power_of_two(
    case: UniformCase,
) -> None:
    profiles = [
        uniform_inputs(case, offset)[2] for offset in uniform_basis_offsets(case)
    ]
    profiles.append(uniform_inputs_folded(case)[2])
    for v in profiles:
        head_zero = v[:, :, 0, :]
        for head in range(case.nhead_k):
            np.testing.assert_array_equal(v[:, :, head, :], head_zero * 2.0**head)
    # Hq=8 over Hkv=2: query heads 0-3 read KV head 0, heads 4-7 KV head 1.
    assert case.head_scale_exponents == (0, 0, 0, 0, 1, 1, 1, 1)


def _swap_first_and_last(bits: np.ndarray, case: UniformCase) -> np.ndarray:
    order = [case.nhead_q - 1, *range(1, case.nhead_q - 1), 0]
    return bits[:, :, order, :]


def _interleaved_groups(bits: np.ndarray, case: UniformCase) -> np.ndarray:
    # A GQA mapping bug: h_kv = h_q mod Hkv instead of h_q // (Hq / Hkv). Query
    # head h_q then carries the output of a head that reads KV head h_q mod Hkv.
    group = case.nhead_q // case.nhead_k
    order = [(head % case.nhead_k) * group for head in range(case.nhead_q)]
    return bits[:, :, order, :]


_HEAD_MIXUPS: dict[str, Callable[[np.ndarray, UniformCase], np.ndarray]] = {
    "head-0-broadcast": lambda bits, case: np.repeat(
        bits[:, :, :1, :], bits.shape[2], 2
    ),
    "last-head-broadcast": lambda bits, case: np.repeat(
        bits[:, :, -1:, :], bits.shape[2], 2
    ),
    "heads-reversed": lambda bits, case: bits[:, :, ::-1, :].copy(),
    # Heads 0 and Hq-1 sit in different KV groups. A swap inside one group is
    # invisible by design: those heads share a V scale.
    "first-last-heads-swapped": _swap_first_and_last,
    "gqa-groups-interleaved": _interleaved_groups,
}


@pytest.mark.parametrize("mixup", sorted(_HEAD_MIXUPS))
@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_head_indexing_mixups_are_rejected(case: UniformCase, mixup: str) -> None:
    """A wrong V head, a wrong head stride, a wrong GQA mapping, or one output head
    written everywhere must not pass: per-head V scales move every legal
    coordinate of a head that reads the wrong KV head."""
    correct = correct_uniform_observation(case)
    mixed = UniformObservation(
        tuple(
            (name, _HEAD_MIXUPS[mixup](values, case))
            for name, values in correct.exposures
        )
    )
    result = analyze_uniform_policy(case, mixed, budget=0)
    assert any(item.startswith("target_ulp=") for item in result.violations)
    assert any(item.startswith("stored_row_sum=") for item in result.violations)


def test_broadcast_query_head_matches_the_head_zero_mixup() -> None:
    case = UNIFORM_WINDOW_FP16
    correct = correct_uniform_observation(case)
    broadcast = broadcast_query_head(correct, 0)
    for (_, actual), (_, source) in zip(broadcast.exposures, correct.exposures):
        np.testing.assert_array_equal(
            actual, _HEAD_MIXUPS["head-0-broadcast"](source, case)
        )
    assert analyze_uniform_policy(case, broadcast, budget=0).violations
    with pytest.raises(IndexError):
        broadcast_query_head(correct, case.nhead_q)


def test_gqa_query_heads_take_their_kv_heads_scale() -> None:
    case = UniformCase(1, 8, 8, 4, 2, 4, "fp16", "causal")
    assert case.head_scale_exponents == (0, 0, 1, 1)
    correct = correct_uniform_observation(case)
    assert analyze_uniform_policy(case, correct, budget=0).violations == ()
    assert analyze_uniform_policy(
        case, broadcast_query_head(correct, 0), budget=0
    ).violations


def test_kv_head_count_is_capped_for_distinct_scales() -> None:
    with pytest.raises(ValueError, match="nhead_k must be at most 8"):
        UniformCase(1, 8, 8, 16, 16, 4, "fp16", "causal")


@pytest.mark.parametrize(
    "case",
    [
        UNIFORM_WINDOW_FP16,
        UNIFORM_CAUSAL_BF16,
        _small_case("causal"),
        _small_case("window", window=1),
        _small_case("window", window=3),
        _small_case("window", window=4),
    ],
)
def test_correctly_rounded_synthetic_observations_pass(case: UniformCase) -> None:
    result = analyze_uniform_policy(
        case,
        correct_uniform_observation(case),
        budget=0,
        focused_rows=range(case.seqlen_q),
    )
    assert result.violations == ()
    assert result.max_ulp == 0
    assert result.exact_count == result.compared_count
    assert result.max_stored_sum_residual == 0.0


@pytest.mark.parametrize(
    "case,mutation,required",
    [
        (UNIFORM_CAUSAL_BF16, UniformMutation.INCLUDE_FUTURE_KEY, ("target_ulp=",)),
        (
            UNIFORM_CAUSAL_BF16,
            UniformMutation.EXCLUDE_LAST_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            UNIFORM_CAUSAL_BF16,
            UniformMutation.DENOMINATOR_PLUS_ONE,
            ("target_ulp=",),
        ),
        (
            UNIFORM_WINDOW_FP16,
            UniformMutation.INCLUDE_OLD_WINDOW_KEY,
            ("target_ulp=",),
        ),
        (
            UNIFORM_WINDOW_FP16,
            UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            UNIFORM_WINDOW_FP16,
            UniformMutation.DENOMINATOR_PLUS_ONE,
            ("target_ulp=",),
        ),
    ],
)
def test_mutations_are_detected_on_the_shipped_matrix(
    case: UniformCase, mutation: UniformMutation, required: tuple[str, ...]
) -> None:
    result = analyze_uniform_policy(
        case,
        correct_uniform_observation(case),
        budget=0,
        mutation=mutation,
    )
    for prefix in required:
        assert any(item.startswith(prefix) for item in result.violations)


def test_one_bf16_ulp_passes_old_ceiling_but_fails_exact_policy() -> None:
    # 4e-3 is the superseded BF16 elementwise absolute ceiling.
    case = UNIFORM_CAUSAL_BF16
    observation = correct_uniform_observation(case)
    name = "basis@192"
    coordinate = (0, 255, 0, 0)
    original = int(observation.by_name[name][coordinate])
    mutated = _replace_bits(observation, name, coordinate, original + 1)
    delta = abs(
        float(decode_bits(original + 1, "bf16")) - float(decode_bits(original, "bf16"))
    )
    assert delta < 4e-3
    result = analyze_uniform_policy(case, mutated, budget=0)
    assert any(item.startswith("target_ulp=") for item in result.violations)


def test_basis_and_folded_blind_spots_are_explicit() -> None:
    assert target_bits(1, 190, "bf16") == target_bits(1, 191, "bf16")
    assert target_bits(2, 190, "bf16") != target_bits(3, 191, "bf16")
    assert [residue_count(0, 7, d, 4) for d in range(4)] == [2, 2, 2, 2]
    assert [residue_count(4, 11, d, 4) for d in range(4)] == [2, 2, 2, 2]


def test_nonzero_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="only budget=0"):
        analyze_uniform_policy(
            _small_case("causal"),
            correct_uniform_observation(_small_case("causal")),
            budget=1,
        )


def test_vacuous_partial_observation_is_a_policy_violation() -> None:
    case = _small_case("causal")
    observation = UniformObservation(
        (("basis@8", np.zeros(case.output_shape, dtype=np.uint16)),)
    )
    result = analyze_uniform_policy(
        case, observation, budget=0, require_complete=False, focused_rows=()
    )
    assert "no_legal_coordinate_observed" in result.violations


def test_observation_validation_rejects_missing_duplicate_and_bad_shape() -> None:
    case = _small_case("causal")
    basis = uniform_expected_bits(case, "basis@0")
    with pytest.raises(ValueError, match="duplicate"):
        UniformObservation((("basis@0", basis), ("basis@0", basis)))
    with pytest.raises(ValueError, match="shape"):
        analyze_uniform_policy(
            case,
            UniformObservation((("basis@0", basis[:, :-1]), ("folded", basis))),
            budget=0,
        )
    with pytest.raises(ValueError, match="folded"):
        analyze_uniform_policy(
            case, UniformObservation((("basis@0", basis),)), budget=0
        )


def test_exact_zero_invalid_negative_and_masked_contracts() -> None:
    case = _small_case("causal")
    correct = correct_uniform_observation(case)
    masked = (0, 0, 0, 0)
    leak = _replace_bits(correct, "basis@4", masked, 0x0001)
    leak_result = analyze_uniform_policy(case, leak, budget=0)
    assert any(item.startswith("masked_max=") for item in leak_result.violations)
    assert any(item.startswith("masked_mass=") for item in leak_result.violations)

    negative_zero = _replace_bits(correct, "basis@4", masked, 0x8000)
    assert analyze_uniform_policy(case, negative_zero, budget=0).violations == ()

    legal = (0, 0, 0, 0)
    negative = _replace_bits(correct, "basis@0", legal, 0xBC00)
    negative_result = analyze_uniform_policy(case, negative, budget=0)
    assert any(item.startswith("negative_max=") for item in negative_result.violations)

    nan = _replace_bits(correct, "basis@0", legal, 0x7E00)
    nan_result = analyze_uniform_policy(case, nan, budget=0)
    assert any(item.startswith("invalid=") for item in nan_result.violations)
    assert nan_result.invalid_count == 1


def test_nonzero_trailing_coordinate_is_rejected() -> None:
    case = _small_case("causal", seqlen_k=6)
    correct = correct_uniform_observation(case)
    trailing = _replace_bits(correct, "basis@4", (0, 0, 0, 2), 0x0001)
    result = analyze_uniform_policy(case, trailing, budget=0)
    assert any(item.startswith("trailing_max=") for item in result.violations)


# The scalar analyzer the vectorized one replaced, kept as an executable
# specification: one query, one dimension, one rule at a time. The contracts
# below require the two to agree on every field of the result.


def _scalar_representation(
    case: UniformCase, exposure: str, dim: int, first: int, last: int
) -> tuple[bool, int]:
    if exposure == "folded":
        has_key = dim < case.seqlen_k
        represented = residue_count(first, last, dim, case.head_dim) if has_key else 0
        return has_key, represented
    key = _basis_offset(exposure) + dim
    has_key = key < case.seqlen_k
    return has_key, int(has_key and first <= key <= last)


def _scalar_head_targets(
    case: UniformCase, represented: int, denominator: int
) -> np.ndarray:
    return np.array(
        [
            target_bits(represented << exponent, denominator, case.dtype)
            for exponent in case.head_scale_exponents
        ],
        dtype=np.uint16,
    )


def _scalar_expected_bits(
    case: UniformCase, exposure: str, mutation: UniformMutation | None
) -> np.ndarray:
    output = np.zeros(case.output_shape, dtype=np.uint16)
    for query in range(case.seqlen_q):
        first, last = legal_interval(case, query, mutation)
        legal_count = max(0, last - first + 1)
        if legal_count == 0:
            continue
        denominator = legal_count + int(
            mutation == UniformMutation.DENOMINATOR_PLUS_ONE
        )
        for dim in range(case.head_dim):
            _, represented = _scalar_representation(case, exposure, dim, first, last)
            if represented:
                output[:, query, :, dim] = _scalar_head_targets(
                    case, represented, denominator
                )
    return output


def _scalar_analyze(
    case: UniformCase,
    observation: UniformObservation,
    mutation: UniformMutation | None,
    focused_rows: Sequence[int] = UNIFORM_FOCUSED_ROWS,
) -> UniformResult:
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
    worst_coordinate = None
    worst_exposure = None
    compared_count = 0
    exact_count = 0
    max_stored_sum_residual = 0.0
    real_row_sum_error = 0.0
    legacy_abs_residual = 0.0
    max_ulp_within_proven = 0
    max_ulp_beyond_proven = 0
    focus = {query: 0.0 for query in focused_rows if 0 <= query < case.seqlen_q}
    groups: dict[str, dict[int, _MutableNqStats]] = {}

    head_scales = np.ldexp(1.0, np.array(case.head_scale_exponents))
    for exposure, raw in observation.exposures:
        decoded = decode_bits(raw, case.dtype) / head_scales[None, None, :, None]
        fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
        exponent_mask = ((1 << exponent_bits) - 1) << fraction_bits
        finite = (raw & np.uint16(exponent_mask)) != np.uint16(exponent_mask)
        invalid_count += int(np.count_nonzero(~finite))
        negative = (
            finite & ((raw & np.uint16(0x8000)) != 0) & ((raw & np.uint16(0x7FFF)) != 0)
        )
        if np.any(negative):
            max_negative = max(max_negative, float(np.max(np.abs(decoded[negative]))))

        exposure_groups = groups.setdefault(exposure, {})
        folded_actual = np.zeros(row_shape, dtype=np.float64)
        folded_target = np.zeros(row_shape, dtype=np.float64)
        folded_valid = np.ones(row_shape, dtype=bool)
        folded_represented = np.zeros(case.seqlen_q, dtype=np.int64)

        for query in range(case.seqlen_q):
            first, last = legal_interval(case, query, mutation)
            legal_count = max(0, last - first + 1)
            if legal_count == 0:
                invalid_count += case.batch * case.nhead_q
            denominator = legal_count + int(
                mutation == UniformMutation.DENOMINATOR_PLUS_ONE
            )
            group = exposure_groups.setdefault(legal_count, _MutableNqStats())
            group.rows += case.batch * case.nhead_q

            for dim in range(case.head_dim):
                has_key, represented = _scalar_representation(
                    case, exposure, dim, first, last
                )
                values = decoded[:, query, :, dim]
                value_bits = raw[:, query, :, dim]
                valid = finite[:, query, :, dim]
                coordinate_magnitudes = np.where(valid, np.abs(values), 0.0)

                if not has_key:
                    local = float(np.max(coordinate_magnitudes))
                    max_trailing_abs = max(max_trailing_abs, local)
                    continue
                if represented == 0:
                    local = float(np.max(coordinate_magnitudes))
                    max_masked_abs = max(max_masked_abs, local)
                    total_masked_mass += float(np.sum(coordinate_magnitudes))
                    masked_row_mass[:, query, :] += coordinate_magnitudes
                    continue

                targets = _scalar_head_targets(case, represented, denominator)
                target_value = decode_bits(targets, case.dtype) / head_scales
                distances = np.abs(
                    _ordered_array(value_bits).astype(np.int64)
                    - _ordered_array(targets).astype(np.int64)[None, :]
                )
                valid_distances = distances[valid]
                compared = int(valid_distances.size)
                exact = int(np.count_nonzero(valid_distances == 0))
                compared_count += compared
                exact_count += exact
                group.compared += compared
                group.exact += exact
                local_ulp = int(np.max(valid_distances)) if compared else 0
                group.max_ulp = max(group.max_ulp, local_ulp)
                if legal_count < _proven_limit(case.dtype, exposure):
                    max_ulp_within_proven = max(max_ulp_within_proven, local_ulp)
                else:
                    max_ulp_beyond_proven = max(max_ulp_beyond_proven, local_ulp)
                if local_ulp > max_ulp:
                    max_ulp = local_ulp
                    worst_ulp_nq = legal_count
                    worst_exposure = exposure
                    masked_distances = np.where(valid, distances, -1)
                    batch, head = np.unravel_index(
                        int(np.argmax(masked_distances)), masked_distances.shape
                    )
                    worst_coordinate = (int(batch), query, int(head), dim)

                real_expected = represented / denominator
                if np.any(valid):
                    errors = np.abs(values[valid] - real_expected)
                    local_error = float(np.max(errors))
                    legacy_abs_residual = max(legacy_abs_residual, local_error)
                    if query in focus:
                        focus[query] = max(focus[query], local_error)

                if exposure == "folded":
                    folded_actual[:, query, :] += np.where(valid, values, 0.0)
                    folded_target[:, query, :] += target_value
                    folded_valid[:, query, :] &= valid
                    folded_represented[query] += represented
                else:
                    basis_present = True
                    basis_actual[:, query, :] += np.where(valid, values, 0.0)
                    basis_target[:, query, :] += target_value
                    basis_valid[:, query, :] &= valid
                    basis_represented[query] += represented

            if exposure == "folded" and folded_represented[query] == legal_count:
                valid_rows = folded_valid[:, query, :]
                if np.any(valid_rows):
                    residuals = np.abs(
                        folded_actual[:, query, :] - folded_target[:, query, :]
                    )
                    local = float(np.max(residuals[valid_rows]))
                    max_stored_sum_residual = max(max_stored_sum_residual, local)
                    group.max_stored_sum_residual = max(
                        group.max_stored_sum_residual, local
                    )
                    real_errors = np.abs(folded_actual[:, query, :] - 1.0)
                    real_row_sum_error = max(
                        real_row_sum_error, float(np.max(real_errors[valid_rows]))
                    )

    if basis_present:
        basis_groups = groups.setdefault("basis_union", {})
        for query in range(case.seqlen_q):
            first, last = legal_interval(case, query, mutation)
            legal_count = max(0, last - first + 1)
            group = basis_groups.setdefault(legal_count, _MutableNqStats())
            group.rows += case.batch * case.nhead_q
            if basis_represented[query] != legal_count:
                continue
            valid_rows = basis_valid[:, query, :]
            if not np.any(valid_rows):
                continue
            residuals = np.abs(basis_actual[:, query, :] - basis_target[:, query, :])
            local = float(np.max(residuals[valid_rows]))
            max_stored_sum_residual = max(max_stored_sum_residual, local)
            group.max_stored_sum_residual = max(group.max_stored_sum_residual, local)
            real_errors = np.abs(basis_actual[:, query, :] - 1.0)
            real_row_sum_error = max(
                real_row_sum_error, float(np.max(real_errors[valid_rows]))
            )

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
    if max_ulp > 0:
        violations.append(f"target_ulp={max_ulp}>0 at N_q={worst_ulp_nq}")
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
        max_row_masked_mass=float(np.max(masked_row_mass)),
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


# Small shapes, so the scalar reference stays fast: batch 2 and GQA 4/2 for the
# indexing, a key length that is not a multiple of D for trailing lanes, windows
# narrower than, equal to and wider than D, and a one-key window whose
# exclude-oldest mutation empties every row.
_EQUIVALENCE_CASES = (
    UniformCase(2, 12, 12, 4, 2, 4, "fp16", "causal"),
    UniformCase(2, 12, 10, 4, 2, 4, "bf16", "causal"),
    UniformCase(1, 12, 12, 4, 2, 4, "bf16", "window", 3),
    UniformCase(2, 12, 10, 4, 2, 4, "fp16", "window", 4),
    UniformCase(1, 16, 16, 2, 1, 4, "fp16", "window", 6),
    UniformCase(1, 8, 8, 2, 1, 4, "bf16", "window", 1),
)


def _equivalence_observations(
    case: UniformCase,
) -> list[tuple[str, UniformObservation]]:
    """Correct, mutated-kernel, head-mixed and single-coordinate-damaged outputs."""
    correct = correct_uniform_observation(case)
    exposures = [name for name, _ in correct.exposures]
    observations = [("correct", correct)]
    for mutation in _mutations_for(case)[1:]:
        observations.append(
            (
                f"kernel-{mutation.value}",
                UniformObservation(
                    tuple(
                        (name, uniform_expected_bits(case, name, mutation))
                        for name in exposures
                    )
                ),
            )
        )
    observations.append(("broadcast", broadcast_query_head(correct, 0)))
    last = case.seqlen_q - 1
    folded_bits = int(correct.by_name["folded"][0, last, 0, 0])
    infinity = 0x7C00 if case.dtype == "fp16" else 0x7F80
    damage = {
        "one-ulp": ("basis@0", (0, last, case.nhead_q - 1, 0), None),
        "two-ulp-folded": ("folded", (0, last, 0, 0), folded_bits + 2),
        "nan": ("basis@0", (case.batch - 1, last, 1, 0), 0x7E00),
        "infinity": ("folded", (0, last, 0, 1), infinity),
        "negative": ("basis@0", (0, last, 0, 0), 0xBC00),
        "masked-leak": (f"basis@{case.head_dim}", (0, 0, 0, 0), 0x0001),
    }
    for label, (name, coordinate, bits) in damage.items():
        if bits is None:
            bits = int(correct.by_name[name][coordinate]) + 1
        observations.append((label, _replace_bits(correct, name, coordinate, bits)))
    return observations


@pytest.mark.parametrize("case", _EQUIVALENCE_CASES, ids=lambda case: case.case_id)
def test_vectorized_expected_bits_match_the_scalar_construction(
    case: UniformCase,
) -> None:
    names = [f"basis@{offset}" for offset in uniform_basis_offsets(case)]
    names += ["basis@64", "folded"]
    for mutation in _mutations_for(case):
        for name in names:
            np.testing.assert_array_equal(
                uniform_expected_bits(case, name, mutation),
                _scalar_expected_bits(case, name, mutation),
                err_msg=f"{name} {mutation}",
            )


@pytest.mark.parametrize("case", _EQUIVALENCE_CASES, ids=lambda case: case.case_id)
def test_vectorized_analyzer_matches_the_scalar_reference(case: UniformCase) -> None:
    focused = range(case.seqlen_q)
    for label, observation in _equivalence_observations(case):
        for mutation in _mutations_for(case):
            vectorized = analyze_uniform_policy(
                case, observation, budget=0, mutation=mutation, focused_rows=focused
            )
            scalar = _scalar_analyze(case, observation, mutation, focused)
            assert vectorized == scalar, f"{label} under {mutation}"


def test_vectorized_analyzer_matches_on_a_quick_case_spot_check() -> None:
    """One full-size quick case, so the shipped shape is covered too."""
    case = UNIFORM_WINDOW_FP16
    observation = _replace_bits(
        correct_uniform_observation(case), "basis@128", (0, 200, 5, 10), 0x0001
    )
    vectorized = analyze_uniform_policy(case, observation, budget=0)
    assert vectorized == _scalar_analyze(case, observation, None)
    assert vectorized.violations


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_uniform_inputs_encode_without_rounding(case: UniformCase, dtype: str) -> None:
    profiles = [uniform_inputs(case, offset) for offset in uniform_basis_offsets(case)]
    profiles.append(uniform_inputs_folded(case))
    for arrays in profiles:
        for values in arrays:
            bits = encode_bits(values, dtype)
            assert bits.dtype == np.uint16
            assert bits.shape == values.shape
            assert bits.flags.c_contiguous
            np.testing.assert_array_equal(
                decode_bits(bits, dtype), values.astype(np.float64)
            )


@pytest.mark.parametrize(
    "dtype,value,expected",
    [
        ("fp16", 1.0, 0x3C00),
        ("bf16", 1.0, 0x3F80),
        ("fp16", -0.875, 0xBB00),
        ("bf16", -0.875, 0xBF60),
        ("fp16", -0.0, 0x8000),
        ("bf16", -0.0, 0x8000),
    ],
)
def test_encode_bits_pins_values_and_signed_zero(
    dtype: str, value: float, expected: int
) -> None:
    bits = encode_bits(np.array([value], dtype=np.float32), dtype)
    assert int(bits[0]) == expected


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("fp16", 1.0 / 3.0),
        ("bf16", 1.0 / 3.0),
        ("fp16", 1.0 + 2.0**-11),
        ("bf16", 1.0 + 2.0**-9),
        ("fp16", 65536.0),
        ("fp16", float("nan")),
        ("bf16", float("nan")),
        ("fp16", float("inf")),
        ("bf16", float("inf")),
    ],
)
def test_encode_bits_rejects_values_that_would_round(dtype: str, value: float) -> None:
    values = np.array([0.0, value], dtype=np.float32)
    with pytest.raises(ValueError, match="not exactly representable"):
        encode_bits(values, dtype)


def _host_bits(case: UniformCase) -> dict[str, np.ndarray]:
    return {
        "q": np.zeros(case.output_shape, dtype=np.uint16),
        "k": np.zeros(case.key_shape, dtype=np.uint16),
        "v": np.zeros(case.key_shape, dtype=np.uint16),
    }


def test_subject_host_validation_accepts_raw_bit_inputs() -> None:
    _validate_host_bits(UNIFORM_WINDOW_FP16, **_host_bits(UNIFORM_WINDOW_FP16))


@pytest.mark.parametrize(
    "name,replace,error,match",
    [
        ("q", lambda array: array[:, :-1], ValueError, "q shape"),
        ("k", lambda array: array.astype(np.float16), ValueError, "k dtype"),
        ("v", np.asfortranarray, ValueError, "v must be C-contiguous"),
        ("q", lambda array: array.tolist(), TypeError, "q must be a numpy array"),
    ],
)
def test_subject_host_validation_rejects_bad_abi(name, replace, error, match) -> None:
    arrays = _host_bits(UNIFORM_WINDOW_FP16)
    arrays[name] = replace(arrays[name])
    with pytest.raises(error, match=match):
        _validate_host_bits(UNIFORM_WINDOW_FP16, **arrays)


class _FakeDeviceBuffer:
    def __init__(self, pointer: int, size: int) -> None:
        self.pointer = pointer
        self.size = size

    def ptr(self) -> int:
        return self.pointer

    def nbytes(self) -> int:
        return self.size


def _fake_device_buffers(
    case: UniformCase,
) -> tuple[dict[str, _FakeDeviceBuffer], dict[str, int]]:
    stride = 1 << 24
    sizes = {
        "q": 2 * int(np.prod(case.output_shape)),
        "k": 2 * int(np.prod(case.key_shape)),
        "v": 2 * int(np.prod(case.key_shape)),
        "out": 2 * int(np.prod(case.output_shape)),
    }
    buffers = {
        name: _FakeDeviceBuffer((index + 1) * stride, size)
        for index, (name, size) in enumerate(sizes.items())
    }
    return buffers, sizes


def test_subject_device_validation_accepts_separate_aligned_buffers() -> None:
    _validate_device_buffers(*_fake_device_buffers(UNIFORM_WINDOW_FP16))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda buffers: setattr(buffers["k"], "size", 2), "k device size"),
        (lambda buffers: setattr(buffers["v"], "pointer", 0), "v device pointer"),
        (lambda buffers: setattr(buffers["out"], "pointer", (4 << 24) + 2), "aligned"),
        (
            lambda buffers: setattr(buffers["out"], "pointer", buffers["q"].ptr()),
            "q overlaps out",
        ),
    ],
)
def test_subject_device_validation_rejects_bad_buffers(mutation, match: str) -> None:
    buffers, sizes = _fake_device_buffers(UNIFORM_WINDOW_FP16)
    mutation(buffers)
    with pytest.raises(ValueError, match=match):
        _validate_device_buffers(buffers, sizes)


def test_subject_requires_a_known_arch_and_the_proven_scope() -> None:
    for arch in DENSE_SUBJECTS:
        for case in UNIFORM_QUICK_CASES:
            _validate_supported_case(arch, case)
    with pytest.raises(ValueError, match="no dense oracle-free subject"):
        _validate_supported_case("gfx90a", UNIFORM_WINDOW_FP16)
    with pytest.raises(ValueError, match="outside the proven zero-ULP range"):
        _validate_supported_case(
            "gfx942", UniformCase(1, 2048, 2048, 8, 2, 64, "fp16", "causal")
        )


def _fake_tuning_spec(**overrides) -> SimpleNamespace:
    """A dispatch result shaped like the FP16 GQA window quick case."""
    kernel_spec = SimpleNamespace(
        batch=1,
        seqlen_q=256,
        seqlen_kv=256,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        dtype="fp16",
        causal=True,
        sliding_window=64,
        persistent=False,
        runtime_shape=True,
        block_m=128,
        block_n=64,
        use_sinks=False,
        emit_lse=False,
        kv_segments=1,
        ragged=False,
        causal_bottom_right=False,
        paged=False,
        varlen=False,
    )
    for name, value in overrides.items():
        setattr(kernel_spec, name, value)
    return SimpleNamespace(
        path="dense",
        arch="gfx942",
        candidate_name="attention_gfx942_dense",
        kernel_spec=kernel_spec,
    )


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("kv_segments", 2, "single-pass"),
        ("ragged", True, "ragged"),
        ("causal_bottom_right", True, "top-left"),
        ("use_sinks", True, "sinks"),
        ("emit_lse", True, "LSE"),
        ("paged", True, "non-paged"),
        ("varlen", True, "non-varlen"),
        ("num_kv_heads", 8, "num_kv_heads=8"),
        ("sliding_window", 0, "sliding_window=0"),
    ],
)
def test_subject_refuses_resolved_specs_outside_the_proof(
    name: str, value, match: str
) -> None:
    _verify_resolved_spec("gfx942", UNIFORM_WINDOW_FP16, _fake_tuning_spec())
    with pytest.raises(AssertionError, match=match):
        _verify_resolved_spec(
            "gfx942", UNIFORM_WINDOW_FP16, _fake_tuning_spec(**{name: value})
        )


def test_subject_refuses_another_arch_or_candidate() -> None:
    with pytest.raises(AssertionError, match="arch='gfx942'"):
        _verify_resolved_spec("gfx950", UNIFORM_WINDOW_FP16, _fake_tuning_spec())
    other = _fake_tuning_spec()
    other.candidate_name = "attention_gfx942_dense_qbminor"
    with pytest.raises(AssertionError, match="candidate="):
        _verify_resolved_spec("gfx942", UNIFORM_WINDOW_FP16, other)
    other = _fake_tuning_spec()
    other.path = "2d"
    with pytest.raises(AssertionError, match="path="):
        _verify_resolved_spec("gfx942", UNIFORM_WINDOW_FP16, other)


@pytest.mark.parametrize(
    "name,value",
    [
        ("persistent", True),
        ("runtime_shape", False),
        ("block_m", 64),
        ("block_n", 32),
        ("use_exp2_fast", False),
        ("waves_per_eu", 1),
    ],
)
def test_tuning_knobs_inside_the_proof_stay_free(name: str, value) -> None:
    """The proof covers every single-pass dense variant, so a retune that keeps
    the scope keeps running instead of failing a re-ratification gate."""
    _verify_resolved_spec(
        "gfx942", UNIFORM_WINDOW_FP16, _fake_tuning_spec(**{name: value})
    )


@pytest.mark.parametrize(
    "subject_case",
    QUICK_SUBJECT_CASES,
    ids=lambda item: f"{item[0]}-{item[1].case_id}",
)
def test_subject_resolves_real_shipped_spec_without_torch(
    subject_case: tuple[str, UniformCase],
) -> None:
    arch, case = subject_case
    tuning_spec = _resolve_tuning_spec(arch, case)
    assert tuning_spec.path == "dense"
    assert tuning_spec.arch == arch
    assert tuning_spec.candidate_name == DENSE_SUBJECTS[arch].candidate_name
    subject = resolve_dense_subject(arch, case)
    assert (subject.arch, subject.case_id) == (arch, case.case_id)
    assert subject.candidate_name == DENSE_SUBJECTS[arch].candidate_name


@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_gfx950_subject_entry_resolves_without_torch(case: UniformCase) -> None:
    """gfx950 has no cases in the GPU lane yet; its table entry must still name a
    candidate that dispatch resolves inside the proof's scope."""
    subject = resolve_dense_subject("gfx950", case)
    assert subject.candidate_name == DENSE_SUBJECTS["gfx950"].candidate_name


def test_quick_matrix_runs_every_quick_case_on_a_known_subject() -> None:
    assert len(set(QUICK_SUBJECT_CASES)) == len(QUICK_SUBJECT_CASES)
    assert {arch for arch, _ in QUICK_SUBJECT_CASES} <= set(DENSE_SUBJECTS)
    on_gfx942 = [case for arch, case in QUICK_SUBJECT_CASES if arch == "gfx942"]
    assert on_gfx942 == list(UNIFORM_QUICK_CASES)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _oracle_dependency_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    violations: list[str] = []
    forbidden = (
        "torch.nn.functional.scaled_dot_product_attention",
        "rocke.numeric.references",
        "sdpa_reference",
        "reference_common",
        "pinned_reference",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                aliases[imported.asname or imported.name.split(".")[0]] = imported.name
                if any(item in imported.name for item in forbidden):
                    violations.append(f"line {node.lineno}: import {imported.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for imported in node.names:
                full_name = f"{module}.{imported.name}".strip(".")
                aliases[imported.asname or imported.name] = full_name
                if any(item in full_name for item in forbidden):
                    violations.append(f"line {node.lineno}: from {full_name}")
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            root, separator, rest = name.partition(".")
            expanded = aliases.get(root, root) + (separator + rest if separator else "")
            if any(item in expanded for item in forbidden):
                violations.append(f"line {node.lineno}: call {expanded}")
    return violations


def test_oracle_free_sources_have_no_complete_attention_dependency() -> None:
    root = Path(__file__).parent
    violations = {
        path.name: found
        for path in sorted(root.glob("*.py"))
        if (found := _oracle_dependency_violations(path))
    }
    assert not violations


# TheRock's provider test environment has NumPy and ml_dtypes but no torch, so a
# torch dependency anywhere in this suite would turn the GPU gate into a skip.
# The module name is held in one constant and compared by equality, so the
# checker below does not flag its own source.
_TORCH = "torch"
_IMPORT_CALLS = frozenset({"importorskip", "import_module", "__import__", "find_spec"})


def _is_torch_module(name: str) -> bool:
    return name.split(".")[0] == _TORCH


def _torch_dependency_violations(source: str, filename: str) -> list[str]:
    """Flag torch imports at any nesting, torch-loading calls, and torch names."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                f"line {node.lineno}: import {imported.name}"
                for imported in node.names
                if _is_torch_module(imported.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_torch_module(node.module or ""):
                violations.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            argument = node.args[0] if node.args else None
            if (
                name.split(".")[-1] in _IMPORT_CALLS
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and _is_torch_module(argument.value)
            ):
                violations.append(f"line {node.lineno}: {name}({argument.value!r})")
        elif isinstance(node, ast.Name) and node.id == _TORCH:
            violations.append(f"line {node.lineno}: name {node.id}")
    return violations


def test_oracle_free_sources_have_no_torch_dependency() -> None:
    root = Path(__file__).parent
    sources = sorted(root.glob("*.py"))
    assert {"__init__.py", Path(__file__).name} <= {path.name for path in sources}
    violations: dict[str, list[str]] = {}
    for path in sources:
        source = path.read_text(encoding="utf-8")
        if found := _torch_dependency_violations(source, str(path)):
            violations[path.name] = found
    assert not violations


@pytest.mark.parametrize(
    "source",
    [
        "import torch\n",
        "import torch.cuda as cuda\n",
        "def load():\n    from torch import nn\n    return nn\n",
        "import pytest\npytest.importorskip('torch')\n",
        "import importlib\nimportlib.import_module('torch.cuda')\n",
        "import importlib.util\nimportlib.util.find_spec('torch')\n",
        "__import__('torch')\n",
        "torch.zeros(1)\n",
    ],
)
def test_torch_checker_flags_every_dependency_form(source: str) -> None:
    assert _torch_dependency_violations(source, "<synthetic>")


def test_torch_checker_accepts_torch_free_source() -> None:
    clean = (
        "import numpy as np\n"
        "FORBIDDEN = ('torch.nn.functional.scaled_dot_product_attention',)\n"
        "def run(module):\n"
        "    return np.zeros(1), module.torch_free\n"
    )
    assert _torch_dependency_violations(clean, "<synthetic>") == []


_SKIP_CALLS = frozenset({"skip", "skipif", "importorskip", "xfail"})


# Each GPU lane module and the subject matrix its launches must run.
_GPU_LANES = (
    ("test_analytical_dense.py", "QUICK_SUBJECT_CASES"),
    ("test_permutation_dense.py", "PERMUTATION_SUBJECT_CASES"),
)
_GPU_LANE_MODULES = tuple(module for module, _ in _GPU_LANES)


def _gpu_lane_tree(module: str) -> ast.Module:
    path = Path(__file__).with_name(module)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("module,matrix", _GPU_LANES)
def test_gpu_lane_parametrizes_over_its_subject_matrix(
    module: str, matrix: str
) -> None:
    argvalues: list[ast.AST] = []
    for node in ast.walk(_gpu_lane_tree(module)):
        if not isinstance(node, ast.Call):
            continue
        if not _dotted_name(node.func).endswith("mark.parametrize"):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        argvalues.append(node.args[1] if len(node.args) > 1 else keywords["argvalues"])
    assert argvalues, "the GPU lane has no parametrized test"
    for value in argvalues:
        assert isinstance(value, ast.Name), ast.dump(value)
        assert value.id == matrix


@pytest.mark.parametrize("module", _GPU_LANE_MODULES)
def test_gpu_lane_never_skips_after_the_architecture_gate(module: str) -> None:
    """The shared architecture gate is the only permitted skip; on the case's
    architecture an out-of-scope dispatch outcome must fail, never skip."""
    escapes = [
        f"line {node.lineno}: {_dotted_name(node.func)}"
        for node in ast.walk(_gpu_lane_tree(module))
        if isinstance(node, ast.Call)
        and _dotted_name(node.func).split(".")[-1] in _SKIP_CALLS
    ]
    assert not escapes


@pytest.mark.parametrize("module", _GPU_LANE_MODULES)
def test_gpu_lane_gates_through_the_require_aware_arch_helper(module: str) -> None:
    """The architecture gate is ``current_device_arch_in((arch,), ...)`` on the
    case's own architecture, so ``ROCKE_REQUIRE_<ARCH>=1`` fails a lane that lost
    its GPU instead of letting it pass with the GPU cases skipped."""
    gates = [
        node
        for node in ast.walk(_gpu_lane_tree(module))
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "current_device_arch_in"
    ]
    assert len(gates) == 1, [ast.dump(gate) for gate in gates]
    arches = gates[0].args[0]
    assert isinstance(arches, ast.Tuple), ast.dump(arches)
    assert [_dotted_name(item) for item in arches.elts] == ["arch"]


def test_unrejected_mutation_blames_a_failing_control() -> None:
    """A kernel with a mask bug can write exactly what that mutation predicts, so
    the mutation is not rejected. The message must then point at the control's
    violations rather than at the checker."""
    case = UNIFORM_CAUSAL_BF16
    mutation = UniformMutation.INCLUDE_FUTURE_KEY
    exposures = [f"basis@{offset}" for offset in uniform_basis_offsets(case)]
    buggy = UniformObservation(
        tuple(
            (exposure, uniform_expected_bits(case, exposure, mutation))
            for exposure in [*exposures, "folded"]
        )
    )
    control = analyze_uniform_policy(case, buggy, budget=0)
    mutated = analyze_uniform_policy(case, buggy, budget=0, mutation=mutation)
    assert control.violations

    blamed = _not_rejected("label", mutation.value, mutated, control)
    assert "checker failed" not in blamed
    assert "kernel output itself fails" in blamed
    assert format_uniform_result(control) in blamed

    clean = analyze_uniform_policy(case, correct_uniform_observation(case), budget=0)
    assert not clean.violations
    assert "checker failed to reject" in _not_rejected(
        "label", mutation.value, mutated, clean
    )


# Run in a fresh interpreter whose import system refuses torch, so the check holds
# whether or not torch happens to be installed where the suite runs.
_TORCH_BLOCKED_PROBE = """
import importlib
import importlib.abc
import json
import sys


class _RefuseTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "torch":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, _RefuseTorch())
config = json.loads(sys.argv[1])
sys.path[:0] = config["path"]
package = config["package"]
importlib.import_module(package + "._analytical")
importlib.import_module(package + "._permutation")
importlib.import_module(package + "._permutation_path")
subject = importlib.import_module(package + "._dense_subject")
importlib.import_module(package + ".test_analytical_dense")
importlib.import_module(package + ".test_permutation_dense")
for arch, case in (*subject.QUICK_SUBJECT_CASES, *subject.PERMUTATION_SUBJECT_CASES):
    subject.resolve_dense_subject(arch, case)
    importlib.import_module(subject.DENSE_SUBJECTS[arch].runner_module)
loaded = sorted(name for name in sys.modules if name.split(".")[0] == "torch")
assert not loaded, loaded
print("torch-free import and resolution ok")
"""


def _package_root() -> tuple[str, Path]:
    """Return this suite's package name and the directory that makes it importable."""
    package = __package__ or ""
    assert package, "test_contracts must be imported as part of its package"
    root = Path(__file__).resolve().parent
    for _ in package.split("."):
        root = root.parent
    return package, root


def test_suite_imports_and_resolves_with_torch_blocked() -> None:
    package, root = _package_root()
    config = {
        "package": package,
        "path": [str(root), *(entry for entry in sys.path if entry)],
    }
    completed = subprocess.run(
        [sys.executable, "-c", _TORCH_BLOCKED_PROBE, json.dumps(config)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "torch-free import and resolution ok" in completed.stdout
