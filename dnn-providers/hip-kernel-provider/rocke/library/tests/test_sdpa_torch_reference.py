# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free checks of the optional SDPA reference-transfer contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reference_common import cli as common_cli
from reference_common.numeric import (
    encode,
    file_digest,
    max_abs_upper,
    payload_digests,
    write_json,
)
from sdpa_reference import cli, torch_reference
from sdpa_reference.architectures import get_architecture
from sdpa_reference.contract import Case, independent_reference


@pytest.mark.parametrize(
    "with_torch,torch_bound", [(False, 0.001), (True, 0.001), (True, 0.03)]
)
def test_qualification_bounds_all_enabled_references(
    tmp_path, monkeypatch, with_torch, torch_bound
):
    baseline = tmp_path / "baseline"
    runtime = baseline / "source/platform/python/rocke"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text('"""Synthetic runtime fixture."""\n')
    write_json(
        baseline / "snapshot.json",
        {
            "revision": "a" * 40,
            "files": payload_digests(baseline / "source"),
        },
    )
    target = get_architecture("gfx942")
    case = Case("fp16", 2, 2, 1, False, True, sequence_length=2)
    monkeypatch.setattr(target, "CASES", (case,))
    measured = {}

    def worker(request, **kwargs):
        Path(request["export"]).write_bytes(b"synthetic-code-object")
        with np.load(request["inputs"], allow_pickle=False) as arrays:
            reference = independent_reference(case, dict(arrays))
        output = encode(reference, case.dtype)
        measured["bound"] = max_abs_upper(output.astype(np.float32), reference)
        return [output] * request["repetitions"], {
            "kernel": {"sha256": file_digest(Path(request["export"]))},
            "device_target": "gfx942:fixture",
            "compiler": {"fixture": True},
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    monkeypatch.setattr(torch_reference, "validate_device", lambda *args: None)
    report = {
        "device": "cpu",
        "results": {
            "float32": {
                "numpy_distance_upper": 0.0001,
                "baseline_error_bound": torch_bound,
            },
        },
    }

    def cross_check(*args, **kwargs):
        assert with_torch
        assert kwargs["device"] == "cpu"
        return report

    monkeypatch.setattr(torch_reference, "cross_check", cross_check)
    bundle = tmp_path / "bundle"
    if with_torch and torch_bound >= case.tolerance:
        with pytest.raises(ValueError, match="no comparison budget"):
            cli.qualify(baseline, bundle, 3, torch_reference="cpu")
        assert not (bundle / "qualification-lock.json").exists()
        return
    cli.qualify(baseline, bundle, 3, torch_reference="cpu" if with_torch else None)
    manifest = cli.load_bundle(bundle, bundle / "qualification-lock.json")
    entry = manifest["cases"][case.id]
    assert entry["budget"]["baseline_error_bound"] == (
        max(measured["bound"], torch_bound) if with_torch else measured["bound"]
    )
    assert entry["budget"]["margin"] == 0
    assert manifest["budget_policy"] == "max-reference-bound-v1"
    assert entry.get("torch_reference") == (report if with_torch else None)
    assert not list(bundle.rglob("*.npz"))
    assert not (bundle / "payload/runner/sdpa_reference/torch_reference.py").exists()
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import sdpa_reference.worker; assert 'torch' not in sys.modules",
            str(bundle / "payload/runner"),
        ],
        cwd=tmp_path,
        check=True,
    )
    if with_torch:
        entry["torch_reference"]["results"]["float32"]["numpy_distance_upper"] = 0
        write_json(bundle / "manifest.json", manifest)
        with pytest.raises(ValueError, match="pinned lock"):
            cli.load_bundle(bundle, bundle / "qualification-lock.json")


def test_ordinary_qualification_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import sdpa_reference.cli; assert 'torch' not in sys.modules",
            str(Path(__file__).resolve().parent),
        ],
        check=True,
    )


def test_requested_torch_fails_before_creating_output(tmp_path, monkeypatch):
    def unavailable(*args):
        raise ImportError("Torch is absent")

    monkeypatch.setattr(torch_reference, "validate_device", unavailable)
    output = tmp_path / "bundle"
    with pytest.raises(ImportError, match="Torch is absent"):
        cli.qualify(tmp_path / "missing", output, 3, torch_reference="cpu")
    assert not output.exists()
