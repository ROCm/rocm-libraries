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


def _legacy_internal_imports(paths, root=_PACKAGE_ROOT):
    legacy_prefixes = (
        "tensilelite.Tensile",
        "tensilelite.TensileCreateLibrary",
        "tensilelite.TensileLogic",
        "tensilelite._extops.AMaxGenerator",
        "tensilelite._extops.ExtOpCreateLibrary",
        "tensilelite._extops.LayerNormGenerator",
        "tensilelite._extops.SoftmaxGenerator",
        "tensilelite.tensilelite_logic.ValidCorpusConsistency",
    )
    legacy_imported_names = {
        "tensilelite": {"Tensile", "TensileCreateLibrary", "TensileLogic"},
        "tensilelite._extops": {
            "AMaxGenerator",
            "ExtOpCreateLibrary",
            "LayerNormGenerator",
            "SoftmaxGenerator",
        },
        "tensilelite.tensilelite_logic": {"ValidCorpusConsistency"},
    }
    legacy_relative_imported_names = {
        "_extops": legacy_imported_names["tensilelite._extops"],
        "tensilelite_logic": legacy_imported_names["tensilelite.tensilelite_logic"],
    }
    legacy_relative_modules = {
        "AMaxGenerator",
        "ExtOpCreateLibrary",
        "HandleCustomKernel",
        "KnownBugs",
        "LayerNormGenerator",
        "ParseArguments",
        "Run",
        "SoftmaxGenerator",
        "Tensile",
        "TensileBenchmarkCluster",
        "TensileBenchmarkClusterScripts",
        "TensileCreateLibrary",
        "TensileLibLogicToYaml",
        "TensileLogic",
        "TensileMergeLibrary",
        "TensileRetuneLibrary",
        "TensileUpdateLibrary",
        "ValidChipId",
        "ValidCorpusConsistency",
        "ValidMatrixInstruction",
        "ValidWorkGroup",
        "ValidWorkGroupMappingXCC",
    }
    violations = []

    for path in sorted(paths):
        relative_path = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith(legacy_prefixes)
            ):
                violations.append(f"{relative_path}:{node.lineno}: from {node.module}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module in legacy_imported_names
            ):
                for alias in node.names:
                    if alias.name in legacy_imported_names[node.module]:
                        violations.append(
                            f"{relative_path}:{node.lineno}: from {node.module} import {alias.name}"
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level > 0
                and node.module
                and node.module.split(".")[-1] in legacy_relative_modules
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: from {'.' * node.level}{node.module}"
                )
            elif isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
                imported_names = legacy_relative_imported_names.get(node.module.split(".")[-1], ())
                for alias in node.names:
                    if alias.name in imported_names:
                        violations.append(
                            f"{relative_path}:{node.lineno}: "
                            f"from {'.' * node.level}{node.module} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.level > 0 and node.module is None:
                for alias in node.names:
                    if alias.name in legacy_relative_modules:
                        violations.append(
                            f"{relative_path}:{node.lineno}: from {'.' * node.level} import {alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(legacy_prefixes):
                        violations.append(f"{relative_path}:{node.lineno}: import {alias.name}")
    return violations


def test_legacy_internal_import_scanner_covers_import_forms(tmp_path):
    module = tmp_path / "stale_imports.py"
    module.write_text(
        "\n".join(
            (
                "from tensilelite import TensileLogic",
                "from tensilelite._extops import AMaxGenerator",
                "from tensilelite.tensilelite_logic import ValidCorpusConsistency",
                "import tensilelite._extops.LayerNormGenerator",
                "from .ValidCorpusConsistency import check_corpus_invariants",
                "from . import TensileLogic",
                "from .._extops import SoftmaxGenerator",
            )
        ),
        encoding="utf-8",
    )

    assert _legacy_internal_imports([module], root=tmp_path) == [
        "stale_imports.py:1: from tensilelite import TensileLogic",
        "stale_imports.py:2: from tensilelite._extops import AMaxGenerator",
        "stale_imports.py:3: from tensilelite.tensilelite_logic import ValidCorpusConsistency",
        "stale_imports.py:4: import tensilelite._extops.LayerNormGenerator",
        "stale_imports.py:5: from .ValidCorpusConsistency",
        "stale_imports.py:6: from . import TensileLogic",
        "stale_imports.py:7: from .._extops import SoftmaxGenerator",
    ]


def test_production_modules_do_not_import_the_legacy_tensile_package():
    """Production modules must not require the separately packaged compatibility alias."""
    production_modules = (
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "Tests" not in path.relative_to(_PACKAGE_ROOT).parts
    )
    assert _legacy_imports(production_modules) == []


def test_production_modules_do_not_import_legacy_internal_packages():
    """Production modules must follow the lower-case internal module layout."""
    production_modules = (
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "Tests" not in path.relative_to(_PACKAGE_ROOT).parts
    )
    assert _legacy_internal_imports(production_modules) == []


def test_unit_modules_do_not_import_the_legacy_tensile_package():
    """Canonical unit tests must exercise the package name shipped by the wheel."""
    unit_root = _PACKAGE_ROOT / "Tests/unit"
    unit_modules = unit_root.rglob("*.py")
    assert _legacy_imports(unit_modules) == []


def test_test_modules_do_not_import_legacy_internal_packages():
    """All test modules must follow the lower-case internal package layout."""
    test_root = _PACKAGE_ROOT / "Tests"
    assert _legacy_internal_imports(test_root.rglob("*.py")) == []


def test_installed_artifacts_exclude_source_only_tests():
    cmake = (_PACKAGE_ROOT.parent.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    package_install = cmake.split('DIRECTORY "${_tensilelite_src}/tensilelite/"', 1)[1].split(
        ")", 1
    )[0]

    assert 'PATTERN "source_only" EXCLUDE' in package_install


def test_legacy_namespace_bridge_files_are_absent():
    project_root = _PACKAGE_ROOT.parent

    assert not (project_root / "Tensile/__init__.py").exists()
    assert not (_PACKAGE_ROOT / "_namespace_bridge.py").exists()
    assert not (_PACKAGE_ROOT / "Tests/unit/test_namespace_bridge.py").exists()
