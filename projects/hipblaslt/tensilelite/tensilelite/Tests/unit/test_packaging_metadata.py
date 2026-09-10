# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path, monkeypatch):
    """rocisa is currently provisioned from source, not resolved by pip."""
    # ``pip wheel`` runs in a child process. Give it an explicit ROCm identity
    # so this metadata test does not depend on tox's bootstrap fallback.
    monkeypatch.setenv("TENSILELITE_ROCM_VERSION", "7.0.0")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
            ".",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")

    assert "Requires-Dist: rocisa" not in metadata

def test_cmake_device_generation_owns_raw_rocisa():
    """Device generation builds raw rocisa; rocisa-only builds opt in explicitly."""
    cmake = (_PROJECT_ROOT.parent / "CMakeLists.txt").read_text(encoding="utf-8")

    assert (
        'option(ROCISA_BUILD_PYTHON '
        '"Build the in-tree rocisa Python extension without device libraries." OFF)'
    ) in cmake
    assert "if(HIPBLASLT_ENABLE_DEVICE OR ROCISA_BUILD_PYTHON" in cmake
