# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source checks for imports within the canonical TensileLite package."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.unit
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _legacy_imports(paths):
    violations = []

    for path in sorted(paths):
        relative_path = path.relative_to(_PACKAGE_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and (node.module == "Tensile" or node.module.startswith("Tensile."))
            ):
                violations.append(f"{relative_path}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "Tensile" or alias.name.startswith("Tensile."):
                        violations.append(f"{relative_path}:{node.lineno}: import {alias.name}")
    return violations


def test_production_modules_do_not_import_the_legacy_tensile_package():
    """Production modules must not require the separately packaged compatibility alias."""
    production_modules = (
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "Tests" not in path.relative_to(_PACKAGE_ROOT).parts
    )
    assert _legacy_imports(production_modules) == []


def test_unit_modules_do_not_import_the_legacy_tensile_package():
    """Canonical unit tests must exercise the package name shipped by the wheel."""
    unit_root = _PACKAGE_ROOT / "Tests/unit"
    unit_modules = (
        path for path in unit_root.rglob("*.py") if path.name != "test_namespace_bridge.py"
    )
    assert _legacy_imports(unit_modules) == []


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
    for package in ("tensilelite", "Tensile"):
        shutil.copytree(
            project_root / package,
            build_source / package,
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
    assert "Tensile/__init__.py" in names
    assert "tensilelite/resources.py" in names
    assert not any(name.startswith("tensilelite/Tests/unit/source_only/") for name in names)
