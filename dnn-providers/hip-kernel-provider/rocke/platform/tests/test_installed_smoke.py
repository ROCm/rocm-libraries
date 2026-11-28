# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Installed artifact path discovery and independent engine smoke checks."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def smoke():
    root = Path(__file__).resolve().parent.parent
    script = root / "tools" / "rocke_installed_smoke.py"
    if not script.is_file():
        script = root / "rocke_installed_smoke.py"
    spec = importlib.util.spec_from_file_location("installed_smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bucket", ["rocke", "hip_kernel_provider"])
def test_relocated_native_artifact_path(smoke, tmp_path, monkeypatch, bucket):
    scripts = tmp_path / "bin" / bucket
    native = scripts / "tests" / "native"
    native.mkdir(parents=True)
    (native / "rocke_test_native_probe.py").write_text("value = 17\n")
    monkeypatch.setattr(smoke, "__file__", str(scripts / "rocke_installed_smoke.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    smoke._add_installed_python_paths()
    spec = importlib.util.find_spec("rocke_test_native_probe")
    assert Path(spec.origin).parent == native


def test_python_smoke_uses_independent_lowerer(smoke, monkeypatch):
    import rocke.core.lower_llvm as lowering

    def dispatched_backend(*args, **kwargs):
        raise AssertionError("Python smoke must exercise the Python engine directly")

    monkeypatch.setattr(lowering, "lower_kernel_to_llvm", dispatched_backend)
    serialized, llvm = smoke._smoke_python_engine()
    assert serialized and "rocke_install_smoke" in llvm
