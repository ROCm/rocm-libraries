# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the P3 checker and dense subject adapter."""

from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    P3_BASIS_OFFSETS,
    P3_CALIBRATION,
    P3_CAUSAL_FP16,
    P3_WINDOW_BF16,
    CalibrationEntry,
    P3Case,
    P3Mutation,
    P3Observation,
    P3SpecFingerprint,
    analyze_p3_policy,
    correct_p3_observation,
    decode_bits,
    legal_keys,
    p3_expected_bits,
    p3_inputs,
    p3_inputs_folded,
    residue_count,
    target_bits,
    ulp_distance,
)
from ._gfx942_subject import (
    _resolve_tuning_spec,
    _validate_supported_case,
    _validate_tensors,
    _verify_resolved_spec,
    resolve_dense_subject,
)


def _small_case(mask: str, *, window: int = 0, seqlen_k: int = 8) -> P3Case:
    return P3Case(1, 8, seqlen_k, 2, 2, 4, "fp16", mask, window)


def _replace_bits(
    observation: P3Observation,
    exposure: str,
    coordinate: tuple[int, int, int, int],
    bits: int,
) -> P3Observation:
    outputs: list[tuple[str, np.ndarray]] = []
    for name, values in observation.exposures:
        copy = values.copy()
        if name == exposure:
            copy[coordinate] = np.uint16(bits)
        outputs.append((name, copy))
    return P3Observation(tuple(outputs))


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
    assert legal_keys(causal, 3, P3Mutation.INCLUDE_FUTURE_KEY) == (0, 1, 2, 3, 4)
    assert legal_keys(causal, 3, P3Mutation.EXCLUDE_LAST_KEY) == (0, 1, 2)
    assert legal_keys(window, 4) == (1, 2, 3, 4)
    assert legal_keys(window, 4, P3Mutation.INCLUDE_OLD_WINDOW_KEY) == (0, 1, 2, 3, 4)
    assert legal_keys(window, 4, P3Mutation.EXCLUDE_OLDEST_WINDOW_KEY) == (2, 3, 4)


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


def test_p3_generators_cover_basis_and_folded_profiles() -> None:
    exposed: list[int] = []
    for offset in P3_BASIS_OFFSETS:
        q, k, v = p3_inputs(P3_CAUSAL_FP16, offset)
        assert not np.any(q)
        assert np.isfinite(k).all()
        assert np.unique(k).size > 1
        locations = np.argwhere(v[0, :, 0, :] == 1.0)
        exposed.extend(int(key) for key, _ in locations)
        assert all(key == offset + dim for key, dim in locations)
    assert exposed == list(range(256))

    q, k, folded = p3_inputs_folded(P3_CAUSAL_FP16)
    assert not np.any(q)
    assert np.isfinite(k).all()
    locations = np.argwhere(folded[0, :, 0, :] == 1.0)
    assert len(locations) == P3_CAUSAL_FP16.seqlen_k
    assert all(dim == key % P3_CAUSAL_FP16.head_dim for key, dim in locations)


@pytest.mark.parametrize(
    "case",
    [
        P3_CAUSAL_FP16,
        P3_WINDOW_BF16,
        _small_case("causal"),
        _small_case("window", window=1),
        _small_case("window", window=3),
        _small_case("window", window=4),
    ],
)
def test_correctly_rounded_synthetic_observations_pass(case: P3Case) -> None:
    result = analyze_p3_policy(
        case,
        correct_p3_observation(case),
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
        (P3_CAUSAL_FP16, P3Mutation.INCLUDE_FUTURE_KEY, ("target_ulp=",)),
        (
            P3_CAUSAL_FP16,
            P3Mutation.EXCLUDE_LAST_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            P3_CAUSAL_FP16,
            P3Mutation.DENOMINATOR_PLUS_ONE,
            ("target_ulp=",),
        ),
        (
            P3_WINDOW_BF16,
            P3Mutation.INCLUDE_OLD_WINDOW_KEY,
            ("target_ulp=",),
        ),
        (
            P3_WINDOW_BF16,
            P3Mutation.EXCLUDE_OLDEST_WINDOW_KEY,
            ("masked_max=", "masked_mass="),
        ),
        (
            P3_WINDOW_BF16,
            P3Mutation.DENOMINATOR_PLUS_ONE,
            ("target_ulp=",),
        ),
    ],
)
def test_mutations_are_detected_on_the_shipped_matrix(
    case: P3Case, mutation: P3Mutation, required: tuple[str, ...]
) -> None:
    result = analyze_p3_policy(
        case,
        correct_p3_observation(case),
        budget=0,
        mutation=mutation,
    )
    for prefix in required:
        assert any(item.startswith(prefix) for item in result.violations)


def test_one_bf16_ulp_passes_old_ceiling_but_fails_exact_policy() -> None:
    case = P3_WINDOW_BF16
    observation = correct_p3_observation(case)
    name = "basis@192"
    coordinate = (0, 255, 0, 0)
    original = int(observation.by_name[name][coordinate])
    mutated = _replace_bits(observation, name, coordinate, original + 1)
    delta = abs(
        float(decode_bits(original + 1, "bf16")) - float(decode_bits(original, "bf16"))
    )
    assert delta < 4e-3
    result = analyze_p3_policy(case, mutated, budget=0)
    assert any(item.startswith("target_ulp=") for item in result.violations)


def test_basis_and_folded_blind_spots_are_explicit() -> None:
    assert target_bits(1, 190, "bf16") == target_bits(1, 191, "bf16")
    assert target_bits(2, 190, "bf16") != target_bits(3, 191, "bf16")
    assert [residue_count(0, 7, d, 4) for d in range(4)] == [2, 2, 2, 2]
    assert [residue_count(4, 11, d, 4) for d in range(4)] == [2, 2, 2, 2]


def test_nonzero_budget_is_rejected_until_calibrated() -> None:
    with pytest.raises(ValueError, match="only budget=0"):
        analyze_p3_policy(
            _small_case("causal"),
            correct_p3_observation(_small_case("causal")),
            budget=1,
        )


def test_vacuous_partial_observation_is_a_policy_violation() -> None:
    case = _small_case("causal")
    observation = P3Observation(
        (("basis@8", np.zeros(case.output_shape, dtype=np.uint16)),)
    )
    result = analyze_p3_policy(
        case, observation, budget=0, require_complete=False, focused_rows=()
    )
    assert "no_legal_coordinate_observed" in result.violations


def test_observation_validation_rejects_missing_duplicate_and_bad_shape() -> None:
    case = _small_case("causal")
    basis = p3_expected_bits(case, "basis@0")
    with pytest.raises(ValueError, match="duplicate"):
        P3Observation((("basis@0", basis), ("basis@0", basis)))
    with pytest.raises(ValueError, match="shape"):
        analyze_p3_policy(
            case,
            P3Observation((("basis@0", basis[:, :-1]), ("folded", basis))),
            budget=0,
        )
    with pytest.raises(ValueError, match="folded"):
        analyze_p3_policy(case, P3Observation((("basis@0", basis),)), budget=0)


def test_exact_zero_invalid_negative_and_masked_contracts() -> None:
    case = _small_case("causal")
    correct = correct_p3_observation(case)
    masked = (0, 0, 0, 0)
    leak = _replace_bits(correct, "basis@4", masked, 0x0001)
    leak_result = analyze_p3_policy(case, leak, budget=0)
    assert any(item.startswith("masked_max=") for item in leak_result.violations)
    assert any(item.startswith("masked_mass=") for item in leak_result.violations)

    negative_zero = _replace_bits(correct, "basis@4", masked, 0x8000)
    assert analyze_p3_policy(case, negative_zero, budget=0).violations == ()

    legal = (0, 0, 0, 0)
    negative = _replace_bits(correct, "basis@0", legal, 0xBC00)
    negative_result = analyze_p3_policy(case, negative, budget=0)
    assert any(item.startswith("negative_max=") for item in negative_result.violations)

    nan = _replace_bits(correct, "basis@0", legal, 0x7E00)
    nan_result = analyze_p3_policy(case, nan, budget=0)
    assert any(item.startswith("invalid=") for item in nan_result.violations)
    assert nan_result.invalid_count == 1


def test_nonzero_trailing_coordinate_is_rejected() -> None:
    case = _small_case("causal", seqlen_k=6)
    correct = correct_p3_observation(case)
    trailing = _replace_bits(correct, "basis@4", (0, 0, 0, 2), 0x0001)
    result = analyze_p3_policy(case, trailing, budget=0)
    assert any(item.startswith("trailing_max=") for item in result.violations)


def _fingerprint() -> P3SpecFingerprint:
    return P3SpecFingerprint(
        path="dense",
        persistent=False,
        runtime_shape=True,
        head_dim=64,
        block_m=256,
        block_n=64,
        use_exp2_fast=True,
        use_sinks=False,
        emit_lse=False,
    )


def test_calibration_entries_are_zero_budget_ratified_and_scoped() -> None:
    entry = CalibrationEntry(
        ulp_budget=0,
        source_revision="0123456789abcdef",
        case_ids=(P3_CAUSAL_FP16.case_id,),
        spec=_fingerprint(),
        ratified=True,
    )
    assert entry.covers(P3_CAUSAL_FP16, _fingerprint())
    assert not entry.covers(P3_WINDOW_BF16, _fingerprint())
    with pytest.raises(ValueError, match="ulp_budget=0"):
        CalibrationEntry(1, "revision", (P3_CAUSAL_FP16.case_id,), _fingerprint(), True)
    with pytest.raises(ValueError, match="ratified"):
        CalibrationEntry(
            0, "revision", (P3_CAUSAL_FP16.case_id,), _fingerprint(), False
        )
    assert set(P3_CALIBRATION) == {
        ("gfx942", "attention_gfx942_dense", "fp16"),
        ("gfx942", "attention_gfx942_dense", "bf16"),
    }
    for key, installed in P3_CALIBRATION.items():
        assert len(key) == 3
        assert installed.ulp_budget == 0
        assert installed.ratified
        assert installed.source_revision == "bf8ab4fc093979d7d9ae54d0934bf8e0fc38baf2"
    assert P3_CALIBRATION[("gfx942", "attention_gfx942_dense", "fp16")].covers(
        P3_CAUSAL_FP16, _fingerprint()
    )
    assert P3_CALIBRATION[("gfx942", "attention_gfx942_dense", "bf16")].covers(
        P3_WINDOW_BF16, _fingerprint()
    )


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        pointer: int,
        *,
        dtype: str = "fp16",
        device: str = "cuda:0",
        contiguous: bool = True,
        is_cuda: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda
        self._pointer = pointer
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._pointer

    def numel(self) -> int:
        return int(np.prod(self.shape))

    def element_size(self) -> int:
        return 2


def _fake_tensors(case: P3Case) -> dict[str, _FakeTensor]:
    stride = 1 << 24
    return {
        "q": _FakeTensor(case.output_shape, stride),
        "k": _FakeTensor(case.key_shape, 2 * stride),
        "v": _FakeTensor(case.key_shape, 3 * stride),
        "out": _FakeTensor(case.output_shape, 4 * stride),
    }


def test_subject_tensor_validation_accepts_p3_contract() -> None:
    tensors = _fake_tensors(P3_CAUSAL_FP16)
    _validate_tensors(P3_CAUSAL_FP16, **tensors, expected_dtype="fp16")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda tensors: setattr(tensors["out"], "shape", (1,)), "out shape"),
        (lambda tensors: setattr(tensors["k"], "dtype", "bf16"), "k dtype"),
        (lambda tensors: setattr(tensors["v"], "is_cuda", False), "v must be"),
        (
            lambda tensors: setattr(tensors["out"], "_contiguous", False),
            "out must be contiguous",
        ),
        (lambda tensors: setattr(tensors["out"], "_pointer", (4 << 24) + 2), "aligned"),
        (
            lambda tensors: setattr(
                tensors["out"], "_pointer", tensors["q"].data_ptr()
            ),
            "q overlaps out",
        ),
    ],
)
def test_subject_tensor_validation_rejects_bad_abi(mutation, match: str) -> None:
    tensors = _fake_tensors(P3_CAUSAL_FP16)
    mutation(tensors)
    with pytest.raises(ValueError, match=match):
        _validate_tensors(P3_CAUSAL_FP16, **tensors, expected_dtype="fp16")


def test_subject_accepts_only_the_shipped_p3_cases() -> None:
    _validate_supported_case(P3_CAUSAL_FP16)
    _validate_supported_case(P3_WINDOW_BF16)
    with pytest.raises(ValueError, match="D64"):
        _validate_supported_case(P3Case(1, 256, 256, 4, 4, 128, "fp16", "causal"))


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
        block_m=256,
        block_n=64,
        use_sinks=False,
        emit_lse=False,
        resolved_use_exp2_fast=lambda: True,
    )
    tuning_spec = SimpleNamespace(
        path="dense",
        arch="gfx942",
        candidate_name="attention_gfx942_dense",
        kernel_spec=kernel_spec,
    )
    assert _verify_resolved_spec(P3_CAUSAL_FP16, tuning_spec) == _fingerprint()
    kernel_spec.resolved_use_exp2_fast = lambda: False
    with pytest.raises(AssertionError, match="fast exp2"):
        _verify_resolved_spec(P3_CAUSAL_FP16, tuning_spec)


@pytest.mark.parametrize("case", [P3_CAUSAL_FP16, P3_WINDOW_BF16])
def test_subject_resolves_real_shipped_spec_without_torch(case: P3Case) -> None:
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


def test_analytical_module_has_no_torch_import() -> None:
    path = Path(__file__).with_name("_analytical.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_roots = {
        imported.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for imported in node.names
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "torch" not in imported_roots
