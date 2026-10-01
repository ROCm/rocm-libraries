# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Required GPU comparisons when a qualified SDPA reference bundle is installed."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sdpa_reference.cli import load_bundle, verify_case
from sdpa_reference.contract import checked_inputs, decode, encode
from sdpa_reference.architectures import ARCHITECTURES, get_architecture
from sdpa_reference.session import reuse_workers


@pytest.fixture(scope="module")
def reference_bundle():
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch()
    declared = os.environ.get("AMDGPU_FAMILIES", "").lower()
    if arch not in ARCHITECTURES:
        expected = any(
            family in declared
            for name in ARCHITECTURES
            for family in get_architecture(name).FAMILIES
        )
        required = os.environ.get("ROCKE_REQUIRE_SDPA_GPU") == "1" or any(
            key.startswith("ROCKE_SDPA_REFERENCE_BUNDLE") and value
            for key, value in os.environ.items()
        )
        if expected or (not arch and required):
            pytest.fail("the required enrolled SDPA GPU is not available")
        pytest.skip(f"no SDPA reference cohort is enrolled for {arch}")
    target = get_architecture(arch)
    configured = os.environ.get(f"ROCKE_SDPA_REFERENCE_BUNDLE_{arch.upper()}")
    if arch == "gfx942" and not configured:
        configured = os.environ.get("ROCKE_SDPA_REFERENCE_BUNDLE")
    bundle = (
        Path(configured).resolve()
        if configured
        else Path(__file__).parent / "reference_bundles" / arch
    )
    if not bundle.is_dir():
        if os.environ.get("ROCKE_REQUIRE_SDPA_GPU") == "1" or configured:
            pytest.fail(f"required {arch} SDPA reference bundle is missing: {bundle}")
        pytest.skip("qualified SDPA bundle not installed; see TESTING.md")
    manifest = load_bundle(bundle, architecture=arch)
    with reuse_workers():
        yield target, bundle, manifest


@pytest.mark.gpu
@pytest.mark.parametrize(
    "architecture,case",
    [(name, case) for name in ARCHITECTURES for case in get_architecture(name).CASES],
    ids=[
        f"{name}-{case.id}"
        for name in ARCHITECTURES
        for case in get_architecture(name).CASES
    ],
)
def test_sdpa_correctness_against_qualified_rocke(architecture, case, reference_bundle):
    target, bundle, manifest = reference_bundle
    if architecture != target.NAME:
        pytest.skip(f"case requires {architecture}, found {target.NAME}")
    report = verify_case(
        case, bundle=bundle, manifest=manifest, architecture=architecture
    )
    assert report["old_launches"] == report["current_launches"] == 2
    assert not report["torch_imported"]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["source", "replay"])
def test_sdpa_rejects_perturbed_gpu_results(reference_bundle, monkeypatch, mode):
    from sdpa_reference import cli

    target, bundle, manifest = reference_bundle
    case = target.CASES[0]
    real_worker = cli._worker

    def perturbed_worker(request, **kwargs):
        outputs, report = real_worker(request, **kwargs)
        if request["mode"] == mode:
            changed = decode(outputs[-1], case.dtype).copy()
            changed.flat[0] += 1.0
            outputs[-1] = encode(changed.astype(np.float32), case.dtype)
        return outputs, report

    monkeypatch.setattr(cli, "_worker", perturbed_worker)
    expected = "remaining limit" if mode == "source" else "qualification failure"
    with pytest.raises(AssertionError, match=expected):
        verify_case(case, bundle=bundle, manifest=manifest, architecture=target.NAME)


@pytest.mark.gpu
def test_sdpa_rejects_missing_gpu_launch(reference_bundle, monkeypatch, tmp_path):
    import rocke
    from rocke.runtime import KernelLauncher
    from sdpa_reference.worker import run

    target, bundle, manifest = reference_bundle
    case = target.CASES[0]
    entry = manifest["cases"][case.id]
    case_dir = bundle / "payload/cases" / case.id
    input_file = tmp_path / "inputs.npz"
    np.savez(input_file, **checked_inputs(case, entry["input_digests"]))
    monkeypatch.setattr(
        KernelLauncher, "__call__", lambda *args, **kwargs: SimpleNamespace(launches=0)
    )
    with pytest.raises(ValueError, match="non-finite or unwritten"):
        run(
            {
                "mode": "replay",
                "architecture": target.NAME,
                "platform_root": str(Path(rocke.__file__).resolve().parent.parent),
                "case": asdict(case),
                "inputs": str(input_file),
                "input_digests": entry["input_digests"],
                "kernel": entry["kernel"],
                "hsaco": str(case_dir / "kernel.hsaco"),
                "repetitions": 1,
            },
            tmp_path,
        )
