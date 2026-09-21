# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source checks for imports within the canonical TensileLite package."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def test_production_modules_do_not_import_the_legacy_tensile_package():
    """Production modules must not require the separately packaged compatibility alias."""
    violations = []

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_PACKAGE_ROOT)
        if "Tests" in relative_path.parts:
            continue

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

    assert violations == []
