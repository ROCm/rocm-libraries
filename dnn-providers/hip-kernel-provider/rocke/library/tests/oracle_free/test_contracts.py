# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the uniform-probability checker and subject adapter."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    UNIFORM_BASIS_OFFSETS,
    UNIFORM_CALIBRATION,
    UNIFORM_CAUSAL_FP16,
    UNIFORM_QUICK_CASES,
    UNIFORM_WINDOW_BF16,
    CalibrationEntry,
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformSpecFingerprint,
    analyze_uniform_policy,
    correct_uniform_observation,
    decode_bits,
    encode_bits,
    legal_keys,
    require_calibration,
    residue_count,
    target_bits,
    ulp_distance,
    uniform_expected_bits,
    uniform_inputs,
    uniform_inputs_folded,
)
from ._gfx942_subject import (
    _resolve_tuning_spec,
    _validate_device_buffers,
    _validate_host_bits,
    _validate_supported_case,
    _verify_resolved_spec,
    resolve_dense_subject,
)


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
    exposed: list[int] = []
    for offset in UNIFORM_BASIS_OFFSETS:
        q, k, v = uniform_inputs(UNIFORM_CAUSAL_FP16, offset)
        assert not np.any(q)
        assert np.isfinite(k).all()
        assert np.unique(k).size > 1
        locations = np.argwhere(v[0, :, 0, :] == 1.0)
        exposed.extend(int(key) for key, _ in locations)
        assert all(key == offset + dim for key, dim in locations)
    assert exposed == list(range(256))

    q, k, folded = uniform_inputs_folded(UNIFORM_CAUSAL_FP16)
    assert not np.any(q)
    assert np.isfinite(k).all()
    locations = np.argwhere(folded[0, :, 0, :] == 1.0)
    assert len(locations) == UNIFORM_CAUSAL_FP16.seqlen_k
    assert all(dim == key % UNIFORM_CAUSAL_FP16.head_dim for key, dim in locations)


@pytest.mark.parametrize(
    "case",
    [
        UNIFORM_CAUSAL_FP16,
        UNIFORM_WINDOW_BF16,
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
        (UNIFORM_CAUSAL_FP16, UniformMutation.INCLUDE_FUTURE_KEY, ("target_ulp=",)),
        (
            UNIFORM_CAUSAL_FP16,
            UniformMutation.EXCLUDE_LAST_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            UNIFORM_CAUSAL_FP16,
            UniformMutation.DENOMINATOR_PLUS_ONE,
            ("target_ulp=",),
        ),
        (
            UNIFORM_WINDOW_BF16,
            UniformMutation.INCLUDE_OLD_WINDOW_KEY,
            ("target_ulp=",),
        ),
        (
            UNIFORM_WINDOW_BF16,
            UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            UNIFORM_WINDOW_BF16,
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
    case = UNIFORM_WINDOW_BF16
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


def test_nonzero_budget_is_rejected_until_calibrated() -> None:
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


def _fingerprint(block_m: int = 128) -> UniformSpecFingerprint:
    return UniformSpecFingerprint(
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


def test_calibration_entries_are_zero_budget_ratified_and_scoped() -> None:
    fp16_fp = _fingerprint(block_m=128)
    bf16_fp = _fingerprint(block_m=64)
    entry = CalibrationEntry(
        ulp_budget=0,
        source_revision="0123456789abcdef",
        case_ids=(UNIFORM_CAUSAL_FP16.case_id,),
        spec=fp16_fp,
        ratified=True,
    )
    assert entry.covers(UNIFORM_CAUSAL_FP16, fp16_fp)
    assert not entry.covers(UNIFORM_WINDOW_BF16, fp16_fp)
    with pytest.raises(ValueError, match="ulp_budget=0"):
        CalibrationEntry(1, "revision", (UNIFORM_CAUSAL_FP16.case_id,), fp16_fp, True)
    with pytest.raises(ValueError, match="ratified"):
        CalibrationEntry(0, "revision", (UNIFORM_CAUSAL_FP16.case_id,), fp16_fp, False)
    assert set(UNIFORM_CALIBRATION) == {
        ("gfx942", "attention_gfx942_dense", "fp16"),
        ("gfx942", "attention_gfx942_dense", "bf16"),
    }
    for key, installed in UNIFORM_CALIBRATION.items():
        assert len(key) == 3
        assert installed.ulp_budget == 0
        assert installed.ratified
        assert installed.source_revision == "b64ae3c925b9e789fe6f62663e4c4d282ffd0f05"
    assert UNIFORM_CALIBRATION[("gfx942", "attention_gfx942_dense", "fp16")].covers(
        UNIFORM_CAUSAL_FP16, fp16_fp
    )
    assert UNIFORM_CALIBRATION[("gfx942", "attention_gfx942_dense", "bf16")].covers(
        UNIFORM_WINDOW_BF16, bf16_fp
    )


_CALIBRATED_FP16_KEY = ("gfx942", "attention_gfx942_dense", "fp16")


def test_calibration_mismatch_names_the_case_and_every_differing_field() -> None:
    entry = CalibrationEntry(
        ulp_budget=0,
        source_revision="0123456789abcdef",
        case_ids=(UNIFORM_CAUSAL_FP16.case_id,),
        spec=_fingerprint(block_m=128),
        ratified=True,
    )
    assert entry.mismatch(UNIFORM_CAUSAL_FP16, _fingerprint(block_m=128)) == ()

    case_problem, field_problem = entry.mismatch(
        UNIFORM_WINDOW_BF16, _fingerprint(block_m=64)
    )
    assert UNIFORM_WINDOW_BF16.case_id in case_problem
    assert "not in calibrated" in case_problem
    assert field_problem == "block_m: calibrated 128, resolved 64"

    drifted = dataclasses.replace(
        _fingerprint(block_m=128), persistent=True, use_sinks=True
    )
    assert entry.mismatch(UNIFORM_CAUSAL_FP16, drifted) == (
        "persistent: calibrated False, resolved True",
        "use_sinks: calibrated False, resolved True",
    )
    assert not entry.covers(UNIFORM_CAUSAL_FP16, drifted)


def test_require_calibration_returns_the_covering_entry() -> None:
    entry = require_calibration(
        "gfx942",
        "attention_gfx942_dense",
        UNIFORM_CAUSAL_FP16,
        _fingerprint(block_m=128),
    )
    assert entry is UNIFORM_CALIBRATION[_CALIBRATED_FP16_KEY]


@pytest.mark.parametrize(
    "arch,table,case,fingerprint,match",
    [
        pytest.param(
            "gfx942",
            {},
            UNIFORM_CAUSAL_FP16,
            _fingerprint(block_m=128),
            "no uniform-probability calibration",
            id="empty-table",
        ),
        pytest.param(
            "gfx950",
            UNIFORM_CALIBRATION,
            UNIFORM_CAUSAL_FP16,
            _fingerprint(block_m=128),
            "no uniform-probability calibration",
            id="other-arch",
        ),
        pytest.param(
            "gfx942",
            UNIFORM_CALIBRATION,
            UNIFORM_CAUSAL_FP16,
            _fingerprint(block_m=256),
            "block_m: calibrated 128, resolved 256",
            id="fingerprint-mismatch",
        ),
        pytest.param(
            "gfx942",
            UNIFORM_CALIBRATION,
            UniformCase(1, 256, 256, 4, 4, 64, "fp16", "window", 64),
            _fingerprint(block_m=128),
            "not in calibrated",
            id="case-not-listed",
        ),
    ],
)
def test_require_calibration_fails_closed_with_the_remedy(
    arch: str,
    table,
    case: UniformCase,
    fingerprint: UniformSpecFingerprint,
    match: str,
) -> None:
    with pytest.raises(AssertionError) as raised:
        require_calibration(
            arch,
            "attention_gfx942_dense",
            case,
            fingerprint,
            table=table,
            detail="control=ok",
        )
    message = str(raised.value)
    assert match in message
    assert case.case_id in message
    assert "ROCKE_ORACLE_FREE_CALIBRATION_OUT" in message
    assert message.endswith("; control=ok")


def test_gpu_lane_cases_are_exactly_the_calibrated_cases() -> None:
    calibrated = [
        case_id
        for (arch, _, _), entry in UNIFORM_CALIBRATION.items()
        if arch == "gfx942"
        for case_id in entry.case_ids
    ]
    executed = [case.case_id for case in UNIFORM_QUICK_CASES]
    assert len(set(executed)) == len(executed)
    assert set(calibrated) == set(executed)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_uniform_inputs_encode_without_rounding(case: UniformCase, dtype: str) -> None:
    profiles = [uniform_inputs(case, offset) for offset in UNIFORM_BASIS_OFFSETS]
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
    _validate_host_bits(UNIFORM_CAUSAL_FP16, **_host_bits(UNIFORM_CAUSAL_FP16))


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
    arrays = _host_bits(UNIFORM_CAUSAL_FP16)
    arrays[name] = replace(arrays[name])
    with pytest.raises(error, match=match):
        _validate_host_bits(UNIFORM_CAUSAL_FP16, **arrays)


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
    _validate_device_buffers(*_fake_device_buffers(UNIFORM_CAUSAL_FP16))


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
    buffers, sizes = _fake_device_buffers(UNIFORM_CAUSAL_FP16)
    mutation(buffers)
    with pytest.raises(ValueError, match=match):
        _validate_device_buffers(buffers, sizes)


def test_subject_accepts_only_the_shipped_uniform_cases() -> None:
    _validate_supported_case(UNIFORM_CAUSAL_FP16)
    _validate_supported_case(UNIFORM_WINDOW_BF16)
    with pytest.raises(ValueError, match="D64"):
        _validate_supported_case(UniformCase(1, 256, 256, 4, 4, 128, "fp16", "causal"))


def test_subject_requires_the_proof_pinned_resolved_spec() -> None:
    kernel_spec = SimpleNamespace(
        batch=1,
        seqlen_q=256,
        seqlen_kv=256,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        dtype="fp16",
        causal=True,
        sliding_window=0,
        persistent=False,
        runtime_shape=True,
        block_m=128,
        block_n=64,
        use_sinks=False,
        emit_lse=False,
        kv_segments=1,
        ragged=False,
        causal_bottom_right=False,
        resolved_use_exp2_fast=lambda: True,
    )
    tuning_spec = SimpleNamespace(
        path="dense",
        arch="gfx942",
        candidate_name="attention_gfx942_dense",
        kernel_spec=kernel_spec,
    )
    expected = _fingerprint(block_m=128)
    assert _verify_resolved_spec(UNIFORM_CAUSAL_FP16, tuning_spec) == expected
    for name, value, match in (
        ("kv_segments", 2, "single-pass"),
        ("ragged", True, "ragged"),
        ("causal_bottom_right", True, "top-left"),
    ):
        original = getattr(kernel_spec, name)
        setattr(kernel_spec, name, value)
        with pytest.raises(AssertionError, match=match):
            _verify_resolved_spec(UNIFORM_CAUSAL_FP16, tuning_spec)
        setattr(kernel_spec, name, original)
    assert _verify_resolved_spec(UNIFORM_CAUSAL_FP16, tuning_spec) == expected
    kernel_spec.resolved_use_exp2_fast = lambda: False
    with pytest.raises(AssertionError, match="fast exp2"):
        _verify_resolved_spec(UNIFORM_CAUSAL_FP16, tuning_spec)


@pytest.mark.parametrize("case", [UNIFORM_CAUSAL_FP16, UNIFORM_WINDOW_BF16])
def test_subject_resolves_real_shipped_spec_without_torch(case: UniformCase) -> None:
    tuning_spec = _resolve_tuning_spec(case)
    assert tuning_spec.path == "dense"
    assert tuning_spec.arch == "gfx942"
    assert tuning_spec.candidate_name == "attention_gfx942_dense"
    subject = resolve_dense_subject(case)
    assert subject.case_id == case.case_id
    assert subject.fingerprint == _verify_resolved_spec(case, tuning_spec)


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
# torch dependency anywhere in this suite would turn the gfx942 gate into a skip.
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


_SKIP_CALLS = frozenset({"skip", "importorskip", "xfail"})


def _gpu_lane_tree() -> ast.Module:
    path = Path(__file__).with_name("test_analytical_gfx942.py")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_gpu_lane_parametrizes_over_the_calibrated_case_list() -> None:
    argvalues: list[ast.AST] = []
    for node in ast.walk(_gpu_lane_tree()):
        if not isinstance(node, ast.Call):
            continue
        if not _dotted_name(node.func).endswith("mark.parametrize"):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        argvalues.append(node.args[1] if len(node.args) > 1 else keywords["argvalues"])
    assert argvalues, "the GPU lane has no parametrized test"
    for value in argvalues:
        assert isinstance(value, ast.Name), ast.dump(value)
        assert value.id == "UNIFORM_QUICK_CASES"


def test_gpu_lane_never_skips_after_the_architecture_gate() -> None:
    """The module-level gfx942 ``skipif`` is the only permitted skip; on gfx942 a
    missing or mismatched calibration must fail, never skip."""
    escapes = [
        f"line {node.lineno}: {_dotted_name(node.func)}"
        for node in ast.walk(_gpu_lane_tree())
        if isinstance(node, ast.Call)
        and _dotted_name(node.func).split(".")[-1] in _SKIP_CALLS
    ]
    assert not escapes


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
analytical = importlib.import_module(package + "._analytical")
subject = importlib.import_module(package + "._gfx942_subject")
importlib.import_module(package + ".test_analytical_gfx942")
for case in analytical.UNIFORM_QUICK_CASES:
    subject.resolve_dense_subject(case)
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
