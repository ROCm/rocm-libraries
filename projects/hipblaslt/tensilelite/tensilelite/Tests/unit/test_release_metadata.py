# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import runpy

import pytest


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def test_distribution_version_comes_from_selected_rocm_root(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("ROCM_VERSION", "7.3.0")
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    assert metadata["distribution_version"]("5.0.0") == "5.0.0+rocm7.2.4"


def test_compatibility_version_comes_from_selected_rocm_root(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("ROCM_VERSION", "7.3.0")
    metadata = runpy.run_path(str(_SOURCE_ROOT / "compat" / "release_metadata.py"))

    assert metadata["rocm_version"]() == "7.2.4"
