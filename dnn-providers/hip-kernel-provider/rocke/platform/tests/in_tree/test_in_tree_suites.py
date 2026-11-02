# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI wiring for the test directories that live inside the ``rocke`` package.
#
# ``rocke_pytest`` collects ``platform/tests`` only. The directories below sit
# next to the code they test, so no other lane runs them. Each directory runs
# in its own pytest process: the directories use ``unittest`` and plain pytest
# styles and share module names, and a process per directory keeps them apart.
# ``portable_ir/tests`` is not listed: ``portable_ir/test_portable_ir.py::test_unit``
# runs it.

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_PLATFORM = Path(__file__).resolve().parents[2]
_LIBRARY_ROOTS = (
    _PLATFORM.parent / "library",  # source tree: rocke/library
    _PLATFORM / "tests" / "library",  # installed tree: <test-root>/tests/library
)

# Importable package name -> optional modules that the directory needs.
_SUITES = {
    "rocke.benchmark.perf.tests": (),
    "rocke.heuristics.tests": ("pandas", "sklearn", "lightgbm"),
    "rocke.tests.helpers.tiling": (),
}


# Files that need ``matplotlib`` skip without it, as the plotting helpers import it lazily.
_PLOTTING = ("visualization/", "test_lds_conflict.py")
# GPU demo kernels fail on a device other than the gfx90a they target, instead of
# skipping. A GPU lane owns them, so this host lane leaves them out.
_GPU_DEMO_DIRS = ("kernels/",)


def _package_dir(package: str) -> Path:
    # Resolve through the import system: TheRock relocates ``rocke`` beside the
    # installed CTest metadata, where no ``platform/python`` source layout exists.
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations, package
    return Path(next(iter(spec.submodule_search_locations)))


# One item per test file keeps each item under the per-test CTest timeout.
_CASES = sorted(
    (package, path.relative_to(_package_dir(package)).as_posix())
    for package in _SUITES
    for path in _package_dir(package).rglob("test_*.py")
    if not path.relative_to(_package_dir(package)).as_posix().startswith(_GPU_DEMO_DIRS)
)
if not _CASES:
    raise RuntimeError("in-tree test directories contain no test files")


@pytest.mark.parametrize(
    ("package", "relpath"), _CASES, ids=[f"{p}:{r}" for p, r in _CASES]
)
def test_in_tree_file(package, relpath):
    """Run one in-package test file in a child pytest process."""
    for module in _SUITES[package]:
        pytest.importorskip(module)
    if relpath.startswith(_PLOTTING):
        pytest.importorskip("matplotlib")
    library_roots = [root for root in _LIBRARY_ROOTS if (root / "kernels").is_dir()]
    env = rocke_subprocess_env(python_paths=library_roots)
    env["ROCKE_BACKEND"] = "python"
    target = _package_dir(package) / relpath
    # ``-c`` replaces this directory's ini file: the child runs in the default
    # import mode, which resolves the relative imports these files use.
    command = rocke_python_command(
        "-m",
        "pytest",
        str(target),
        "-m",
        "not integration",
        "-q",
        "-rs",
        "-p",
        "no:cacheprovider",
        "-c",
        os.devnull,
        "--rootdir",
        str(_package_dir(package)),
    )
    result = subprocess.run(
        command,
        cwd=str(_PLATFORM),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] or result.stderr[-4000:]
