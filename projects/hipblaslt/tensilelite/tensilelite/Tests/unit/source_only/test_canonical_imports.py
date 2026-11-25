# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging checks for the canonical TensileLite package."""

import os
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


def test_tests_do_not_import_the_test_tree_as_a_package():
    """The wheel does not ship ``tensilelite.Tests``, so such imports fail against it."""
    offenders = [
        f"{path.relative_to(_PACKAGE_ROOT)}:{number}"
        for path in sorted((_PACKAGE_ROOT / "Tests").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.lstrip().startswith(("from tensilelite.Tests", "import tensilelite.Tests"))
    ]
    assert offenders == []


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
    # Copy every top-level file so build inputs added later (setup.py,
    # release_metadata.py, VERSION, ...) cannot drift out of this test's list.
    for path in project_root.iterdir():
        if path.is_file():
            shutil.copyfile(path, build_source / path.name)
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=build_source,
        env={k: v for k, v in os.environ.items() if k != "TENSILELITE_SOURCE_REVISION"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((build_source / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")
    # The copy is not tracked by any repository, so it has no revision and no label.
    assert "\nVersion: 5.0.0\n" in metadata
    assert "Requires-Dist: pandas" in metadata
    assert "tensilelite/resources.py" in names
    assert "tensilelite/_build_info.py" in names
    assert not any(name.startswith("tensilelite/Tests/") for name in names)
    assert not any(name.startswith("tensilelite/Tests/unit/source_only/") for name in names)
