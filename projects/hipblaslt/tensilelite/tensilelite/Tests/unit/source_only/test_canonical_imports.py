# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging checks for the canonical TensileLite package."""

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.unit
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def test_installed_artifacts_exclude_source_only_tests():
    cmake = (_PACKAGE_ROOT.parent.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    package_install = cmake.split('DIRECTORY "${_tensilelite_src}/tensilelite/"', 1)[1].split(
        ")", 1
    )[0]

    assert 'PATTERN "source_only" EXCLUDE' in package_install


def test_wheel_excludes_source_only_tests(tmp_path):
    """Build the wheel and check its contents independently of discovery settings."""
    project_root = _PACKAGE_ROOT.parent
    build_source = tmp_path / "wheel-source"
    build_source.mkdir()
    shutil.copytree(
        project_root / "tensilelite",
        build_source / "tensilelite",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in ("pyproject.toml", "MANIFEST.in", "requirements.txt"):
        shutil.copyfile(project_root / filename, build_source / filename)
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=build_source,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((build_source / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "tensilelite/resources.py" in names
    assert not any(name.startswith("tensilelite/Tests/unit/source_only/") for name in names)
