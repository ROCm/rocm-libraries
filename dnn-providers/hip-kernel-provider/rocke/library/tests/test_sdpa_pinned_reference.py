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
from sdpa_reference.contract import CASES, decode, encode


@pytest.fixture(scope="module")
def reference_bundle():
    configured = os.environ.get("ROCKE_SDPA_REFERENCE_BUNDLE")
    bundle = (
        Path(configured).resolve()
        if configured
        else Path(__file__).with_name("sdpa_reference_bundle")
    )
    required = os.environ.get("ROCKE_REQUIRE_SDPA_GPU") == "1" or configured
    if not bundle.is_dir():
        if required:
            pytest.fail(f"required SDPA reference bundle is missing: {bundle}")
        pytest.skip("qualified SDPA bundle not installed; see TESTING.md")
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch()
    declared = os.environ.get("AMDGPU_FAMILIES", "").lower()
    if (
        arch
        and arch != "gfx942"
        and "gfx942" not in declared
        and "gfx94x" not in declared
    ):
        pytest.skip(f"the SDPA reference pilot is enrolled for gfx942, found {arch}")
    if arch != "gfx942":
        pytest.fail("the required gfx942 SDPA GPU is not available")
    return bundle, load_bundle(bundle)


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_sdpa_correctness_against_qualified_rocke(case, reference_bundle):
    bundle, manifest = reference_bundle
    report = verify_case(case, bundle=bundle, manifest=manifest)
    assert report["old_launches"] == report["current_launches"] == 2
    assert not report["torch_imported"]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["source", "replay"])
def test_sdpa_rejects_perturbed_gpu_results(reference_bundle, monkeypatch, mode):
    from sdpa_reference import cli

    bundle, manifest = reference_bundle
    case = CASES[0]
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
        verify_case(case, bundle=bundle, manifest=manifest)


@pytest.mark.gpu
def test_sdpa_rejects_missing_gpu_launch(reference_bundle, monkeypatch, tmp_path):
    import rocke
    from rocke.runtime import KernelLauncher
    from sdpa_reference.worker import run

    bundle, manifest = reference_bundle
    case = CASES[0]
    entry = manifest["cases"][case.id]
    case_dir = bundle / "payload/cases" / case.id
    monkeypatch.setattr(
        KernelLauncher, "__call__", lambda *args, **kwargs: SimpleNamespace(launches=0)
    )
    with pytest.raises(ValueError, match="non-finite or unwritten"):
        run(
            {
                "mode": "replay",
                "platform_root": str(Path(rocke.__file__).resolve().parent.parent),
                "case": asdict(case),
                "inputs": str(case_dir / "inputs.npz"),
                "input_digests": entry["input_digests"],
                "kernel": entry["kernel"],
                "hsaco": str(case_dir / "kernel.hsaco"),
                "repetitions": 1,
            },
            tmp_path,
        )
