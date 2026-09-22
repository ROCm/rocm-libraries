# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source checks for imports within the canonical TensileLite package."""

import ast
from pathlib import Path

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
        path
        for path in unit_root.rglob("*.py")
        if path.name != "test_namespace_bridge.py"
    )
    assert _legacy_imports(unit_modules) == []
