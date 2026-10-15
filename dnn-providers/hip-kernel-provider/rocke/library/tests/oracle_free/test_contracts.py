# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the P3 generators, checker, and subject adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    P3_BASIS_OFFSETS,
    P3_CAUSAL_FP16,
    P3_TOLERANCES,
    P3_WINDOW_BF16,
    P3Case,
    analyze_p3,
    legal_keys,
    p3_expected,
    p3_inputs,
    p3_tolerance,
    round_to_storage,
)
from ._gfx942_subject import (
    _resolve_tuning_spec,
    _validate_supported_case,
    _validate_tensors,
    _verify_resolved_spec,
)


def _small_case(mask: str, *, window: int = 0) -> P3Case:
    return P3Case(1, 8, 8, 2, 2, 4, "fp16", mask, window)


def test_legal_key_sets_pin_causal_and_window_semantics() -> None:
    causal = _small_case("causal")
    window = _small_case("window", window=4)

    assert legal_keys(causal, 0) == (0,)
    assert legal_keys(causal, 7) == tuple(range(8))
    assert legal_keys(window, 2) == (0, 1, 2)
    assert legal_keys(window, 4) == (1, 2, 3, 4)


def test_wrong_membership_rules_change_p3_expected_values() -> None:
    causal = _small_case("causal")
    window = _small_case("window", window=4)

    assert not np.array_equal(
        p3_expected(causal, 0),
        p3_expected(causal, 0, rule="causal_exclusive"),
    )
    assert not np.array_equal(
        p3_expected(window, 0),
        p3_expected(window, 0, rule="window_inclusive_lower"),
    )


@pytest.mark.parametrize(
    "case,wrong_rule",
    [
        (_small_case("causal"), "causal_exclusive"),
        (_small_case("window", window=4), "window_inclusive_lower"),
    ],
)
def test_p3_checker_rejects_wrong_mask_model(case: P3Case, wrong_rule: str) -> None:
    offsets = (0, 4)
    correct = {offset: p3_expected(case, offset) for offset in offsets}
    positive = analyze_p3(case, correct, offsets=offsets, focused_rows=(0, 3, 4, 7))
    wrong = analyze_p3(
        case,
        correct,
        offsets=offsets,
        focused_rows=(0, 3, 4, 7),
        rule=wrong_rule,
    )

    assert positive.invalid_count == 0
    assert positive.zero_leakage_count == 0
    assert positive.row_mass_error < 1e-15
    assert max(item.maximum_error for item in positive.elements.values()) == 0.0
    assert max(item.maximum_error for item in wrong.elements.values()) > p3_tolerance(
        "element", case.dtype
    )


def test_p3_generators_expose_each_key_once_across_four_windows() -> None:
    exposed: list[int] = []
    for offset in P3_BASIS_OFFSETS:
        q, k, v = p3_inputs(P3_CAUSAL_FP16, offset)
        assert not np.any(q)
        assert np.isfinite(k).all()
        assert np.unique(k).size > 1
        assert set(np.unique(v)) == {0.0, 1.0}
        locations = np.argwhere(v[0, :, 0, :] == 1.0)
        exposed.extend(int(key) for key, _ in locations)
        assert all(key == offset + dim for key, dim in locations)
    assert exposed == list(range(256))


@pytest.mark.parametrize("case", [P3_CAUSAL_FP16, P3_WINDOW_BF16])
def test_fixed_tolerances_cover_unavoidable_storage_rounding(case: P3Case) -> None:
    rounded = {
        offset: round_to_storage(p3_expected(case, offset), case.dtype)
        for offset in P3_BASIS_OFFSETS
    }
    result = analyze_p3(case, rounded)

    assert result.invalid_count == 0
    assert result.zero_leakage_count == 0
    assert max(item.maximum_error for item in result.elements.values()) < p3_tolerance(
        "element", case.dtype
    )
    assert result.row_mass_error < p3_tolerance("row_mass", case.dtype)


def test_checker_detects_element_error_exact_leakage_and_nan_sentinel() -> None:
    case = _small_case("causal")
    offsets = (0, 4)
    outputs = {offset: p3_expected(case, offset) for offset in offsets}

    element_mutation = {offset: values.copy() for offset, values in outputs.items()}
    element_mutation[0][0, 7, 0, 0] += 2.0 * p3_tolerance("element", case.dtype)
    element_result = analyze_p3(case, element_mutation, offsets=offsets)
    assert element_result.elements[0].maximum_error > p3_tolerance(
        "element", case.dtype
    )

    leakage_mutation = {offset: values.copy() for offset, values in outputs.items()}
    leakage_mutation[4][0, 0, 0, 0] = np.nextafter(0.0, 1.0)
    leakage_result = analyze_p3(case, leakage_mutation, offsets=offsets)
    assert leakage_result.zero_leakage_count == 1

    sentinel_mutation = {offset: values.copy() for offset, values in outputs.items()}
    sentinel_mutation[0][0, 0, 0, 0] = np.nan
    sentinel_result = analyze_p3(case, sentinel_mutation, offsets=offsets)
    assert sentinel_result.invalid_count == 1
    assert sentinel_result.elements[0].maximum_error == float("inf")


def test_checker_detects_accumulated_row_mass_below_element_ceiling() -> None:
    case = P3_WINDOW_BF16
    outputs = {offset: p3_expected(case, offset).copy() for offset in P3_BASIS_OFFSETS}
    per_element = 5e-4
    assert per_element < p3_tolerance("element", case.dtype)
    query = 255
    for key in legal_keys(case, query):
        offset = P3_BASIS_OFFSETS[key // case.head_dim]
        dim = key - offset
        outputs[offset][:, query, :, dim] += per_element

    result = analyze_p3(case, outputs)
    assert max(item.maximum_error for item in result.elements.values()) < p3_tolerance(
        "element", case.dtype
    )
    assert result.row_mass_error > p3_tolerance("row_mass", case.dtype)


def test_basis_partition_must_cover_every_key_exactly_once() -> None:
    case = _small_case("causal")
    outputs = {0: p3_expected(case, 0), 3: p3_expected(case, 3)}
    with pytest.raises(ValueError, match="every key exactly once"):
        analyze_p3(case, outputs, offsets=(0, 3))


def test_tolerance_table_is_complete_fixed_and_structural_checks_are_exact() -> None:
    assert set(P3_TOLERANCES) == {"element", "row_mass"}
    assert P3_TOLERANCES["element"].fp16 == 5e-4
    assert P3_TOLERANCES["element"].bf16 == 4e-3
    assert P3_TOLERANCES["row_mass"].fp16 == 5e-3
    assert P3_TOLERANCES["row_mass"].bf16 == 2e-2


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
        _validate_tensors(
            P3_CAUSAL_FP16, **tensors, expected_dtype=P3_CAUSAL_FP16.dtype
        )


def test_subject_accepts_the_two_shipped_p3_cases() -> None:
    _validate_supported_case(P3_CAUSAL_FP16)
    _validate_supported_case(P3_WINDOW_BF16)


@pytest.mark.parametrize(
    "case,match",
    [
        (P3Case(2, 256, 256, 4, 4, 64, "fp16", "causal"), "batch 1"),
        (P3Case(1, 256, 256, 8, 4, 64, "fp16", "causal"), "Hq=Hkv=4"),
        (P3Case(1, 256, 256, 4, 4, 128, "fp16", "causal"), "D64"),
        (P3Case(1, 128, 256, 4, 4, 64, "fp16", "causal"), "query length"),
        (P3Case(1, 256, 128, 4, 4, 64, "fp16", "causal"), "key length"),
        (P3Case(1, 256, 256, 4, 4, 64, "bf16", "window", 63), "window 64"),
    ],
)
def test_subject_rejects_cases_outside_shipped_p3_matrix(
    case: P3Case, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_supported_case(case)


def test_subject_requires_dispatch_resolved_nonpersistent_runtime_spec() -> None:
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
    )
    tuning_spec = SimpleNamespace(
        path="dense",
        arch="gfx942",
        candidate_name="attention_gfx942_dense",
        kernel_spec=kernel_spec,
    )
    _verify_resolved_spec(P3_CAUSAL_FP16, tuning_spec)
    kernel_spec.persistent = True
    with pytest.raises(AssertionError, match="persistent"):
        _verify_resolved_spec(P3_CAUSAL_FP16, tuning_spec)


@pytest.mark.parametrize("case", [P3_CAUSAL_FP16, P3_WINDOW_BF16])
def test_subject_resolves_real_shipped_spec_without_torch(case: P3Case) -> None:
    tuning_spec = _resolve_tuning_spec(case)
    assert tuning_spec.path == "dense"
    assert tuning_spec.arch == "gfx942"
    assert tuning_spec.candidate_name == "attention_gfx942_dense"
    assert tuning_spec.kernel_spec.persistent is False
    assert tuning_spec.kernel_spec.runtime_shape is True


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
