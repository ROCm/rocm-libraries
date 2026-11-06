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


# Every module the lower-case migration renamed, as (old, new) paths relative to
# the ``tensilelite`` package. The ext-op generators were top-level scripts
# outside the package, so their old path is only the module name.
_RENAMED_MODULES = (
    ("Tensile", "tensilelite"),
    ("TensileBenchmarkCluster", "benchmark_cluster"),
    ("TensileBenchmarkClusterScripts", "benchmark_cluster_scripts"),
    ("TensileCreateLibrary", "create_library"),
    ("TensileCreateLibrary.ParseArguments", "create_library.parse_arguments"),
    ("TensileCreateLibrary.Run", "create_library.run"),
    ("TensileLibLogicToYaml", "lib_logic_to_yaml"),
    ("TensileLogic", "logic"),
    ("TensileLogic.HandleCustomKernel", "logic.handle_custom_kernel"),
    ("TensileLogic.KnownBugs", "logic.known_bugs"),
    ("TensileLogic.ParseArguments", "logic.parse_arguments"),
    ("TensileLogic.Run", "logic.run"),
    ("TensileLogic.ValidChipId", "logic.valid_chip_id"),
    ("TensileLogic.ValidCorpusConsistency", "logic.valid_corpus_consistency"),
    ("TensileLogic.ValidMatrixInstruction", "logic.valid_matrix_instruction"),
    ("TensileLogic.ValidWorkGroup", "logic.valid_work_group"),
    ("TensileLogic.ValidWorkGroupMappingXCC", "logic.valid_work_group_mapping_xcc"),
    ("TensileMergeLibrary", "merge_library"),
    ("TensileRetuneLibrary", "retune_library"),
    ("TensileUpdateLibrary", "update_library"),
    ("AMaxGenerator", "_extops.amax_generator"),
    ("ExtOpCreateLibrary", "_extops.extop_create_library"),
    ("LayerNormGenerator", "_extops.layer_norm_generator"),
    ("SoftmaxGenerator", "_extops.softmax_generator"),
)


def _legacy_module_names():
    """Absolute legacy module names: each old path, plus its old leaf under the new parent."""
    names = set()
    for old, new in _RENAMED_MODULES:
        names.add(f"tensilelite.{old}")
        new_parent = new.rpartition(".")[0]
        old_leaf = old.rpartition(".")[2]
        names.add(".".join(filter(None, ("tensilelite", new_parent, old_leaf))))
    return names


def _legacy_internal_imports(paths, root=_PACKAGE_ROOT):
    legacy_modules = _legacy_module_names()
    legacy_prefixes = tuple(sorted(legacy_modules))
    legacy_imported_names = {}
    for module in legacy_modules:
        parent, _, leaf = module.rpartition(".")
        legacy_imported_names.setdefault(parent, set()).add(leaf)
    legacy_relative_imported_names = {
        parent.rpartition(".")[2]: names
        for parent, names in legacy_imported_names.items()
        if parent != "tensilelite"
    }
    legacy_relative_modules = {module.rpartition(".")[2] for module in legacy_modules}

    def has_legacy_prefix(module):
        return any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in legacy_prefixes
        )

    violations = []

    for path in sorted(paths):
        relative_path = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and has_legacy_prefix(node.module)
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
                    if has_legacy_prefix(alias.name):
                        violations.append(f"{relative_path}:{node.lineno}: import {alias.name}")
    return violations

def test_legacy_internal_import_scanner_covers_import_forms(tmp_path):
    module = tmp_path / "stale_imports.py"
    module.write_text(
        "\n".join(
            (
                "from tensilelite import TensileLogic",
                "from tensilelite._extops import AMaxGenerator",
                "from tensilelite.logic import ValidCorpusConsistency",
                "import tensilelite._extops.LayerNormGenerator",
                "from .ValidCorpusConsistency import check_corpus_invariants",
                "from . import TensileLogic",
                "from .._extops import SoftmaxGenerator",
                "from tensilelite.TensileGenerateUID import main",
                "import tensilelite.TensileGenerateUID",
                "from tensilelite.TensileMergeLibrary import fixSizeInconsistencies",
                "from tensilelite import TensileLibLogicToYaml",
                "import tensilelite.TensileUpdateLibrary",
                "from tensilelite.logic import ValidChipId",
                "from tensilelite.create_library import ParseArguments",
                "from ..logic import KnownBugs",
                "from tensilelite.create_library import run",
                "from tensilelite.merge_library import fixSizeInconsistencies",
            )
        ),
        encoding="utf-8",
    )

    assert _legacy_internal_imports([module], root=tmp_path) == [
        "stale_imports.py:1: from tensilelite import TensileLogic",
        "stale_imports.py:2: from tensilelite._extops import AMaxGenerator",
        "stale_imports.py:3: from tensilelite.logic import ValidCorpusConsistency",
        "stale_imports.py:4: import tensilelite._extops.LayerNormGenerator",
        "stale_imports.py:5: from .ValidCorpusConsistency",
        "stale_imports.py:6: from . import TensileLogic",
        "stale_imports.py:7: from .._extops import SoftmaxGenerator",
        "stale_imports.py:10: from tensilelite.TensileMergeLibrary",
        "stale_imports.py:11: from tensilelite import TensileLibLogicToYaml",
        "stale_imports.py:12: import tensilelite.TensileUpdateLibrary",
        "stale_imports.py:13: from tensilelite.logic import ValidChipId",
        "stale_imports.py:14: from tensilelite.create_library import ParseArguments",
        "stale_imports.py:15: from ..logic import KnownBugs",
    ]
def test_production_modules_do_not_import_the_legacy_tensile_package():
    """Production modules must not import the removed legacy ``Tensile`` package."""
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


def test_non_unit_test_modules_do_not_import_the_legacy_tensile_package():
    """Shared test helpers and suites outside Tests/unit use the canonical package name too."""
    tests_root = _PACKAGE_ROOT / "Tests"
    unit_root = tests_root / "unit"
    non_unit_modules = (
        path for path in tests_root.rglob("*.py") if unit_root not in path.parents
    )
    assert _legacy_imports(non_unit_modules) == []


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
