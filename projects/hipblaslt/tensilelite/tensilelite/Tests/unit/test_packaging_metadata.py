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


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path):
    """rocisa is currently provisioned from source, not resolved by pip."""
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


def test_cmake_keeps_python_codegen_dependencies_enabled_by_default():
    """TheRock's raw rocisa artifact still relies on the generic CMake default."""
    cmake = (_PROJECT_ROOT.parent / "CMakeLists.txt").read_text(encoding="utf-8")

    assert (
        'option(HIPBLASLT_BUNDLE_PYTHON_DEPS '
        '"Build Python dependencies required for device code generation." ON)'
    ) in cmake
