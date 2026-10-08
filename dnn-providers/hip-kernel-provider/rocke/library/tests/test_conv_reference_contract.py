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

from conv_reference import cli
from conv_reference.architectures import get_architecture
from conv_reference.contract import (
    Case,
    INPUT_GENERATOR,
    checked_inputs,
    independent_reference,
    make_inputs,
    normalized_distance,
    reference_scale,
)
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    decode,
    encode,
    file_digest,
    write_json,
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


@pytest.mark.parametrize("case", get_architecture("gfx942").CASES, ids=lambda c: c.id)
def test_corpus_and_oracle_are_finite_reproducible_and_authenticated(case):
    arrays = make_inputs(case)
    digests = {name: array_digest(value) for name, value in arrays.items()}
    regenerated = checked_inputs(case, digests)
    reference = independent_reference(case, regenerated)
    assert reference.shape == case.output_shape
    assert np.isfinite(reference).all()
    assert reference_scale(reference) >= 1
    digests["a"] = "0" * 64
    with pytest.raises(ValueError, match="input digest mismatch"):
        checked_inputs(case, digests)


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
        budget = ErrorBudget(case.tolerance, 0.001, case.margin)
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

    monkeypatch.setattr(cli, "_worker", worker)
    monkeypatch.setattr(cli, "independent_reference", forbidden)
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


def test_source_adapter_selects_supported_production_cohort_without_torch():
    from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped

    for case in get_architecture("gfx942").CASES:
        request = ConvGroupedRequest(
            N=case.N,
            C=case.C,
            K=case.K,
            Hi=case.Hi,
            Wi=case.Wi,
            Y=case.Y,
            X=case.X,
            G=case.groups,
            arch="gfx942",
            dtype=case.dtype,
            stride_h=case.sH,
            stride_w=case.sW,
            pad_h=case.pH,
            pad_w=case.pW,
            dilation_h=case.dH,
            dilation_w=case.dW,
        )
        selected = dispatch_conv_grouped(request)
        assert selected.spec.direction == "fwd"
        assert all(v > 0 for v in selected.grid)


@pytest.mark.parametrize("case", get_architecture("gfx942").CASES, ids=lambda c: c.id)
def test_adapter_builds_real_kernel_and_freezes_complete_launch_abi(case, monkeypatch):
    from types import SimpleNamespace
    import rocke
    import rocke.runtime
    import kernels

    captured = []

    def compile_kernel(kernel, **kwargs):
        assert kwargs == {"arch": "gfx942", "backend": "python"}
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
    target = get_architecture("gfx942")
    prepared = target.prepare(case, str(Path(kernels.__file__).parent.parent))
    target.launch(prepared, {"a": 11, "b": 22, "out": 33})
    code, metadata = target.exported_kernel(prepared)
    assert len(captured) == 1
    assert code == b"host-abi-fixture"
    assert set(metadata["bindings"]) == {"A", "B", "D"}
    assert not set(metadata["scalars"]) & set(metadata["bindings"])
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("with_torch", [False, True])
def test_qualification_freezes_self_contained_replay_support(
    tmp_path, monkeypatch, with_torch
):
    """Exercise qualification plumbing with a synthetic GPU-worker result."""
    import subprocess
    import sys
    from reference_common.numeric import payload_digests

    baseline = tmp_path / "baseline"
    runtime = baseline / "source/platform/python/rocke"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text('"""Host qualification fixture."""\n')
    write_json(
        baseline / "snapshot.json",
        {"revision": "a" * 40, "files": payload_digests(baseline / "source")},
    )
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

    monkeypatch.setattr(cli, "_worker", worker)
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
    cli.qualify(baseline, bundle, repetitions=3, torch_reference=with_torch)
    manifest = cli.load_bundle(bundle, bundle / "qualification-lock.json")
    assert manifest["cases"][case.id]["budget"]["baseline_error_bound"] == (
        0.001 if with_torch else 0
    )
    assert ("torch_reference" in manifest["cases"][case.id]) == with_torch
    assert not list(bundle.rglob("*.npz"))
    assert (bundle / "payload/runner/reference_common/numeric.py").is_file()
    # -I removes inherited source paths; the frozen runner must import alone.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import conv_reference.worker; "
            "assert 'torch' not in sys.modules",
            str(bundle / "payload/runner"),
        ],
        cwd=tmp_path,
        check=True,
    )
