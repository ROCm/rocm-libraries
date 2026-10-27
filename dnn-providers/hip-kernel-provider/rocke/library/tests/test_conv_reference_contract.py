# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent oracle, frozen metric, cohort, and fail-closed verification checks."""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from conv_reference import cli, contract
from conv_reference.architectures import ARCHITECTURES, get_architecture
from conv_reference.contract import (
    Case,
    INPUT_GENERATOR,
    independent_reference,
    make_inputs,
    normalized_distance,
    reference_scale,
)
from reference_common import cli as common_cli
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    decode,
    encode,
    file_digest,
    write_json,
)

# Geometries each architecture's published bundle qualifies, in FP16 and BF16.
# Adding a case needs no edit here; removing one drops qualified coverage. Either
# change requires requalifying that architecture's complete cohort.
_QUALIFIED_GEOMETRIES = dict.fromkeys(
    ("gfx942", "gfx950"),
    (
        "padded",
        "pointwise",
        "stride2",
        "dilation2",
        "groups2",
        "asymmetric",
        "multi-tile",
        "odd-channels",
        "odd-input-channels",
        "stem",
        "large",
        "long-k",
        "grouped-odd",
        "depthwise",
        "anisotropic",
        "out-1x1",
    ),
)
_REQUALIFY = (
    "Cohort changes require requalifying the complete cohort; see "
    "reference_common/docs/gpu-reference-workflow.md."
)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_grouped_pointwise_reference_does_not_mix_channels(dtype):
    case = Case("analytic", dtype, Hi=1, Wi=1, C=4, K=4, Y=1, X=1, pH=0, pW=0, groups=2)
    inputs = {
        "a": encode(np.array([[[[1, 2, 10, 20]]]]), dtype),
        "b": encode(np.array([[[[1, 0]]], [[[0, 1]]], [[[1, 0]]], [[[0, 1]]]]), dtype),
    }
    np.testing.assert_array_equal(
        independent_reference(case, inputs), [[[[1, 2, 10, 20]]]]
    )


@pytest.mark.parametrize("stride,dilation,padding", [(1, 1, 0), (2, 1, 1), (1, 2, 2)])
def test_spatial_reference_matches_explicit_patch_sums(stride, dilation, padding):
    case = Case(
        "spatial",
        "fp16",
        Hi=5,
        Wi=7,
        C=1,
        K=1,
        Y=2,
        X=3,
        sH=stride,
        sW=stride,
        pH=padding,
        pW=padding,
        dH=dilation,
        dW=dilation,
    )
    a = np.arange(35, dtype=np.float32).reshape(1, 5, 7, 1)
    inputs = {
        "a": encode(a, case.dtype),
        "b": encode(np.ones((1, 2, 3, 1)), case.dtype),
    }
    padded = np.pad(a[0, :, :, 0], padding)
    expected = np.empty(case.output_shape)
    for ho in range(expected.shape[1]):
        for wo in range(expected.shape[2]):
            patch = padded[
                ho * stride : ho * stride + 2 * dilation : dilation,
                wo * stride : wo * stride + 3 * dilation : dilation,
            ]
            expected[0, ho, wo, 0] = patch.sum()
    np.testing.assert_array_equal(independent_reference(case, inputs), expected)


@pytest.mark.parametrize(
    "case",
    # Architectures share most geometries; check each distinct case once.
    tuple(
        dict.fromkeys(
            case for name in ARCHITECTURES for case in get_architecture(name).CASES
        )
    ),
    ids=lambda c: c.id,
)
def test_corpus_and_oracle_are_finite_reproducible_and_authenticated(case):
    arrays = make_inputs(case)
    digests = {name: array_digest(value) for name, value in arrays.items()}
    regenerated = cli.OPERATION.checked_inputs(case, digests)
    reference = independent_reference(case, regenerated)
    assert reference.shape == case.output_shape
    assert np.isfinite(reference).all()
    assert reference_scale(reference) >= 1
    digests["a"] = "0" * 64
    with pytest.raises(ValueError, match="input digest mismatch"):
        cli.OPERATION.checked_inputs(case, digests)


@pytest.mark.parametrize(
    "change",
    [
        {"groups": 3},
        {"sH": 0},
        {"Hi": -1},
        {"pH": -1},
        {"name": "../escape"},
        {"dtype": "fp32"},
    ],
)
def test_invalid_case_rejected(change):
    with pytest.raises(ValueError):
        replace(Case("valid", "fp16"), **change)


def test_normalized_budget_preserves_original_metric_conservatively():
    reference = np.array([10.0, -2.0])
    old = np.array([10.1, -2.0])
    current = np.array([10.3, -2.1])
    scale = reference_scale(reference)
    bound = normalized_distance(old, reference, scale)
    budget = ErrorBudget(0.05, bound, 0.0025)
    distance = normalized_distance(current, old, scale)
    budget.check(distance)
    assert Fraction(bound) + Fraction(budget.comparison_limit) + Fraction(
        budget.margin
    ) <= Fraction(0.05)
    assert normalized_distance(current, reference, scale) < 0.05
    with pytest.raises(AssertionError, match="remaining limit"):
        budget.check(normalized_distance(np.array([20.0, -2.0]), old, scale))


@pytest.mark.parametrize("scale", [0, 0.5, float("inf"), float("nan")])
def test_invalid_frozen_scale_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        normalized_distance(np.ones(1), np.ones(1), scale)


def _manifest():
    entries = {}
    for case in get_architecture("gfx942").CASES:
        arrays = make_inputs(case)
        budget = ErrorBudget(case.tolerance, 0.001)
        output = encode(np.ones(case.output_shape), case.dtype)
        entries[case.id] = {
            "case": asdict(case),
            "device_target": "gfx942:test",
            "budget": asdict(budget),
            "comparison_limit": budget.comparison_limit,
            "reference_scale": 2.0,
            "input_digests": {
                name: array_digest(value) for name, value in arrays.items()
            },
            "output_digest": array_digest(output),
            "kernel": {},
        }
    return {
        "schema": 2,
        "budget_policy": "max-reference-bound-v1",
        "operation": "conv-fwd",
        "baseline_revision": "a" * 40,
        "input_generation": INPUT_GENERATOR,
        "files": {},
        "cases": entries,
    }


@pytest.mark.parametrize(
    "change",
    [
        "cohort",
        "geometry",
        "budget",
        "scale",
        "target",
        "operation",
        "undeclared",
        "payload",
        "manifest",
    ],
)
def test_locked_bundle_rejects_invalid_contracts(tmp_path, change):
    manifest = _manifest()
    case = get_architecture("gfx942").CASES[0]
    entry = manifest["cases"][case.id]
    if change == "cohort":
        del manifest["cases"][case.id]
    if change == "geometry":
        entry["case"]["sH"] = 2
    if change == "budget":
        entry["budget"]["tolerance"] = 1
    if change == "scale":
        entry["reference_scale"] = 0
    if change == "target":
        entry["device_target"] = "gfx950"
    if change == "operation":
        manifest["operation"] = "sdpa"
    if change == "undeclared":
        del manifest["operation"]
    (tmp_path / "payload").mkdir()
    write_json(tmp_path / "manifest.json", manifest)
    lock = tmp_path / "lock.json"
    write_json(
        lock,
        {
            "schema": 2,
            "baseline_revision": manifest["baseline_revision"],
            "manifest_sha256": file_digest(tmp_path / "manifest.json"),
        },
    )
    if change == "payload":
        (tmp_path / "payload/extra").write_text("unexpected")
    if change == "manifest":
        (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        cli.load_bundle(tmp_path, lock)


@pytest.mark.parametrize(
    "device_target,accepted",
    [
        ("gfx942-strict:sramecc+:xnack-", True),
        ("gfx950:sramecc+:xnack-", False),
    ],
)
def test_bundle_target_is_the_physical_processor(device_target, accepted):
    """Profiles and features do not change the processor that was qualified."""
    manifest = _manifest()
    for entry in manifest["cases"].values():
        entry["device_target"] = device_target
    if accepted:
        cli.OPERATION.validate_cases(manifest, "gfx942")
    else:
        with pytest.raises(ValueError, match="different architecture"):
            cli.OPERATION.validate_cases(manifest, "gfx942")


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_bundle_must_record_the_enrolled_cohort_one_to_one(change):
    manifest = _manifest()
    case = get_architecture("gfx942").CASES[0]
    entry = manifest["cases"][case.id]
    if change == "missing":
        del manifest["cases"][case.id]
        expected = rf"missing: {case.id} \(.*name={case.name},.*\); extra: none"
    elif change == "extra":
        manifest["cases"]["variant"] = dict(entry, case=asdict(replace(case, K=48)))
        expected = r"missing: none; extra: variant \(.*K=48,.*\)"
    else:
        manifest["cases"]["copy"] = entry
        expected = rf"records case {case.id} more than once: copy, {case.id}$"
    with pytest.raises(ValueError, match=expected):
        cli.OPERATION.validate_cases(manifest, "gfx942")


def test_colliding_case_ids_cannot_share_an_entry(monkeypatch):
    target = get_architecture("gfx942")
    # Adapters index ids on import, so a collision fails instead of dropping a case.
    assert list(target.CASE_BY_ID.values()) == list(target.CASES)
    case = target.CASES[0]
    monkeypatch.setattr(target, "CASES", (*target.CASES, replace(case, K=48)))
    with pytest.raises(ValueError, match=f"duplicate convolution case id: {case.id}"):
        cli.OPERATION.validate_cases(_manifest(), "gfx942")


@pytest.mark.parametrize("recorded", [None, "gfx942", "gfx9-4-generic"])
def test_adapter_compiler_target_must_match_the_bundle(monkeypatch, recorded):
    """An adapter that compiles a family code object pins its compiler target."""
    target = get_architecture("gfx942")
    monkeypatch.setattr(target, "COMPILER_TARGET", "gfx9-4-generic", raising=False)
    manifest = _manifest()
    if recorded is not None:
        for entry in manifest["cases"].values():
            entry["kernel"]["compiler_target"] = recorded
    if recorded == target.COMPILER_TARGET:
        cli.OPERATION.validate_cases(manifest, "gfx942")
    else:
        with pytest.raises(ValueError, match="different compiler target"):
            cli.OPERATION.validate_cases(manifest, "gfx942")


def test_unsupported_torch_device_fails_before_creating_output(tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="must be one of cpu"):
        cli.qualify(tmp_path / "missing", output, 3, torch_reference="cuda")
    assert not output.exists()


@pytest.mark.parametrize("failure", [None, "old", "current", "target"])
def test_verification_uses_gpu_outputs_and_frozen_scale_only(
    tmp_path, monkeypatch, failure
):
    manifest = _manifest()
    case = get_architecture("gfx942").CASES[0]
    calls = []

    def worker(request, **kwargs):
        calls.append(request["mode"])
        values = np.ones(case.output_shape)
        if (failure == "old" and request["mode"] == "replay") or (
            failure == "current" and request["mode"] == "source"
        ):
            values.flat[0] = 100
        return [encode(values, case.dtype)] * 2, {
            "launches": 2,
            "torch_imported": False,
            "device_target": "gfx942:other" if failure == "target" else "gfx942:test",
        }

    def forbidden(*args):
        raise AssertionError("CI must never compute an independent answer")

    monkeypatch.setattr(common_cli, "run_worker", worker)
    monkeypatch.setattr(contract, "independent_reference", forbidden)
    if failure:
        with pytest.raises((AssertionError, ValueError)):
            cli.verify_case(
                case, bundle=tmp_path, manifest=manifest, current_root=tmp_path
            )
    else:
        report = cli.verify_case(
            case, bundle=tmp_path, manifest=manifest, current_root=tmp_path
        )
        assert report["normalized_error_upper"] == 0
        assert calls == ["replay", "source"]


def _qualified_cases(architecture):
    """Cases with the qualified ids; the adapter owns their exact geometry."""
    return tuple(
        Case(name, dtype)
        for name in _QUALIFIED_GEOMETRIES[architecture]
        for dtype in ("fp16", "bf16")
    )


def _assert_keeps_qualified_cases(architecture):
    """Allow additions, but name any qualified case the cohort no longer has."""
    ids = [case.id for case in get_architecture(architecture).CASES]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    assert not duplicates, f"{architecture} conv case ids repeat: {duplicates}"
    missing = sorted({case.id for case in _qualified_cases(architecture)} - set(ids))
    assert not missing, (
        f"{architecture} conv cohort no longer contains qualified cases {missing}. "
        "Restore them, or remove them from _QUALIFIED_GEOMETRIES deliberately. "
        + _REQUALIFY
    )


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
@pytest.mark.parametrize("change", ["added", "dropped", "duplicate"])
def test_cohort_check_allows_additions_and_names_lost_cases(
    monkeypatch, architecture, change
):
    target = get_architecture(architecture)
    # Start from exactly the qualified ids so this checks the guard, not the cohort.
    cases = _qualified_cases(architecture)
    changed = {
        "added": (*cases, Case("added", "fp16", K=48)),
        "dropped": cases[1:],
        "duplicate": (*cases, replace(cases[0], K=48)),
    }
    monkeypatch.setattr(target, "CASES", changed[change])
    first_id = rf"\['{cases[0].id}'\]"
    if change == "added":
        _assert_keeps_qualified_cases(architecture)
    elif change == "dropped":
        with pytest.raises(AssertionError, match=first_id + ".*gpu-reference-workflow"):
            _assert_keeps_qualified_cases(architecture)
    else:
        with pytest.raises(AssertionError, match="repeat: " + first_id):
            _assert_keeps_qualified_cases(architecture)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_source_adapter_preserves_implicit_gemm_cohort(architecture):
    from conv_reference.architectures.common import convolution_request
    from dispatch.grouped_convolution import dispatch_conv_grouped, launch_plan_for
    from kernels.common.conv_implicit_gemm import implicit_gemm_conv_grid

    target = get_architecture(architecture)
    _assert_keeps_qualified_cases(architecture)
    for case in target.CASES:
        if architecture == "gfx942":
            spec = target.kernel_spec(case)
            grid = implicit_gemm_conv_grid(spec)
            assert (spec.tile_m, spec.tile_n, spec.tile_k) == (64, 64, 64)
            assert spec.pipeline == "mem"
        else:
            request = convolution_request(case, architecture)
            assert request.algorithm == "implicit_gemm"
            selected = dispatch_conv_grouped(request)
            spec = selected.spec.identity
            assert spec.algorithm == "implicit_gemm"
            assert spec.direction == "fwd"
            plan = launch_plan_for(request, selected.spec)
            assert len(plan.steps) == 1 and not plan.workspaces
            grid = plan.steps[0].grid
        assert all(v > 0 for v in grid)
        # Named cases cover tiling paths described in conv-test-reference.md. A
        # dispatch or instance-list change can move a case off its path.
        if case.name == "multi-tile":
            coverage = (
                f"{architecture} {case.id} must span two output-channel tiles "
                f"with a partial second tile and cshuffle; got K={case.K}, "
                f"tile_n={spec.tile_n}, grid={grid}, epilogue={spec.epilogue}. "
                "Keep this coverage by choosing a K that still needs two N tiles "
                "for the selected instance. " + _REQUALIFY
            )
            assert grid[0] == 2, coverage
            assert case.K % spec.tile_n != 0, coverage
            assert spec.epilogue == "cshuffle", coverage
        elif case.name == "odd-channels":
            assert case.K % 2 == 1
            assert spec.epilogue == "default"
        elif case.name == "large":
            coverage = (
                f"{architecture} {case.id} must span 2 output-channel and 7 "
                "output-pixel tiles with whole K tiles; got "
                f"tile_n={spec.tile_n}, tile_k={spec.tile_k}, grid={grid}. Keep "
                "this coverage by choosing geometry that does so for the selected "
                "instance. " + _REQUALIFY
            )
            assert grid == (2, 7, 1), coverage
            assert case.Y * case.X * case.C % spec.tile_k == 0, coverage
        elif case.name == "grouped-odd":
            assert case.C // case.groups == case.K // case.groups == 33
        elif case.name == "depthwise":
            assert grid[2] == 64
        elif case.name == "out-1x1":
            assert case.output_shape[1:3] == (1, 1)


@pytest.mark.parametrize(
    "architecture,case",
    [
        (architecture, case)
        for architecture in ("gfx942", "gfx950")
        for case in get_architecture(architecture).CASES
    ],
    ids=lambda value: value.id if isinstance(value, Case) else value,
)
def test_adapter_builds_real_kernel_and_freezes_complete_launch_abi(
    case, monkeypatch, architecture
):
    from types import SimpleNamespace
    import rocke
    import rocke.runtime
    import kernels

    captured = []

    def compile_kernel(kernel, **kwargs):
        assert kwargs == {"arch": architecture, "backend": "python"}
        captured.append(kernel)
        return SimpleNamespace(hsaco=b"host-abi-fixture", kernel_name="fixture")

    class Launcher:
        def __init__(self, hsaco, kernel_name, signature):
            self._hsaco, self.kernel_name, self.signature = (
                hsaco,
                kernel_name,
                signature,
            )

        def __call__(self, values, config):
            assert set(values) == {argument["name"] for argument in self.signature}
            assert (values["A"], values["B"], values["D"]) == (11, 22, 33)
            assert values["D_bytes"] == np.prod(case.output_shape) * 2

    monkeypatch.setattr(rocke, "compile_kernel", compile_kernel)
    monkeypatch.setattr(rocke.runtime, "KernelLauncher", Launcher)
    target = get_architecture(architecture)
    prepared = target.prepare(case, str(Path(kernels.__file__).parent.parent))
    target.launch(prepared, {"a": 11, "b": 22, "out": 33})
    code, metadata = target.exported_kernel(prepared)
    assert len(captured) == 1
    assert code == b"host-abi-fixture"
    if architecture == "gfx950":
        assert metadata["dispatch_spec"]["identity"]["arch"] == architecture
        assert metadata["dispatch_spec"]["identity"]["algorithm"] == "implicit_gemm"
    else:
        assert metadata["kernel_spec"]["tile_m"] == 64
    assert set(metadata["bindings"]) == {"A", "B", "D"}
    assert not set(metadata["scalars"]) & set(metadata["bindings"])
    json.dumps(metadata, allow_nan=False)


def _baseline_snapshot(tmp_path):
    """A committed-snapshot layout whose runtime qualification can copy."""
    from reference_common.numeric import payload_digests

    baseline = tmp_path / "baseline"
    runtime = baseline / "source/platform/python/rocke"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text('"""Host qualification fixture."""\n')
    write_json(
        baseline / "snapshot.json",
        {"revision": "a" * 40, "files": payload_digests(baseline / "source")},
    )
    return baseline


@pytest.mark.parametrize("with_torch", [False, True])
def test_qualification_freezes_self_contained_replay_support(
    tmp_path, monkeypatch, with_torch
):
    """Exercise qualification plumbing with a synthetic GPU-worker result."""
    import subprocess
    import sys

    baseline = _baseline_snapshot(tmp_path)
    target = get_architecture("gfx942")
    case = Case("tiny", "fp16", Hi=1, Wi=1, C=1, K=1, Y=1, X=1, pH=0, pW=0)
    monkeypatch.setattr(target, "CASES", (case,))

    def worker(request, **kwargs):
        Path(request["export"]).write_bytes(b"synthetic-code-object")
        with np.load(request["inputs"], allow_pickle=False) as arrays:
            reference = independent_reference(case, dict(arrays))
        return [encode(reference, case.dtype)] * request["repetitions"], {
            "kernel": {"sha256": file_digest(Path(request["export"]))},
            "device_target": "gfx942:fixture",
            "compiler": {"fixture": True},
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    bundle = tmp_path / "bundle"
    if with_torch:
        import types
        from conv_reference import torch_reference

        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setattr(
            torch_reference,
            "cross_check",
            lambda *args: {"results": {"float32": {"baseline_error_bound": 0.001}}},
        )
    cli.qualify(
        baseline, bundle, repetitions=3, torch_reference="cpu" if with_torch else None
    )
    manifest = cli.load_bundle(bundle, bundle / "qualification-lock.json")
    assert manifest["cases"][case.id]["budget"]["baseline_error_bound"] == (
        0.001 if with_torch else 0
    )
    assert ("torch_reference" in manifest["cases"][case.id]) == with_torch
    assert not list(bundle.rglob("*.npz"))
    assert (bundle / "payload/runner/reference_common/numeric.py").is_file()
    # -I removes inherited source paths; the frozen runner must import alone.
    script = """
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import conv_reference.worker
from conv_reference.architectures import get_architecture

get_architecture('gfx942')
get_architecture('gfx950')
for name, module in tuple(sys.modules.items()):
    if name.split(".")[0] in ("conv_reference", "reference_common"):
        assert Path(module.__file__).resolve().is_relative_to(root), name
assert "reference_common.worker" in sys.modules
assert 'kernels' not in sys.modules
assert 'dispatch' not in sys.modules
assert 'torch' not in sys.modules
assert 'pytest' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(bundle / "payload/runner")],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize("failure", ["nondeterministic", "worker"])
def test_qualification_failures_name_the_snapshot_case_and_cause(
    tmp_path, monkeypatch, failure
):
    baseline = _baseline_snapshot(tmp_path)
    target = get_architecture("gfx942")
    case = Case("tiny", "fp16", Hi=1, Wi=1, C=1, K=1, Y=1, X=1, pH=0, pW=0)
    monkeypatch.setattr(target, "CASES", (case,))

    def worker(request, **kwargs):
        if failure == "worker":
            raise RuntimeError(
                "GPU reference source worker failed with exit code 1:\nworker log"
            )
        values = (np.ones(case.output_shape), np.zeros(case.output_shape))
        return [encode(value, case.dtype) for value in values], {}

    monkeypatch.setattr(common_cli, "run_worker", worker)
    with pytest.raises(RuntimeError if failure == "worker" else ValueError) as error:
        cli.qualify(baseline, tmp_path / "bundle", repetitions=2)
    message = str(error.value)
    assert f"convolution gfx942 case {case.id}: the baseline snapshot " in message
    assert "likely cause: " in message and "next step: " in message
    if failure == "worker":
        assert "(qualification) worker failed" in message
        assert "conv_reference/architectures/gfx942/__init__.py" in message
        assert message.endswith("worker log")
    else:
        assert "produced 2 different outputs in 2 executions" in message


@pytest.mark.parametrize("package", ["kernels", "dispatch"])
def test_adapter_rejects_production_imports_outside_selected_tree(
    tmp_path, monkeypatch, package
):
    import importlib
    from conv_reference.architectures.common import validate_library_root

    for name in ("kernels", "dispatch"):
        module = importlib.import_module(name)
        root = tmp_path if name == package else tmp_path / "selected"
        monkeypatch.setattr(module, "__file__", str(root / name / "__init__.py"))
    with pytest.raises(RuntimeError, match="outside the selected library"):
        validate_library_root(str(tmp_path / "selected"))


@pytest.mark.parametrize("previous", [None, "llvm23"])
@pytest.mark.parametrize("fail", [False, True])
def test_instance_compilation_uses_identity_flavor_and_restores_environment(
    monkeypatch, previous, fail
):
    import os
    import rocke
    from conv_reference.architectures.common import compile_reference_kernel

    if previous is None:
        monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    else:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", previous)

    def compile_kernel(kernel, **kwargs):
        assert os.environ["ROCKE_LLVM_FLAVOR"] == "llvm22"
        assert kwargs == {"arch": "gfx950", "backend": "python"}
        if fail:
            raise RuntimeError("compiler failure")
        return kernel

    monkeypatch.setattr(rocke, "compile_kernel", compile_kernel)
    if fail:
        with pytest.raises(RuntimeError, match="compiler failure"):
            compile_reference_kernel("kernel", "gfx950", "llvm22")
    else:
        assert compile_reference_kernel("kernel", "gfx950", "llvm22") == "kernel"
    assert os.environ.get("ROCKE_LLVM_FLAVOR") == previous
