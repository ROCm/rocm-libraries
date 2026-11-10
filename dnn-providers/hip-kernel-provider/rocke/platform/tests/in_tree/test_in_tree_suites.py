# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI wiring for the test directories that live inside the ``rocke`` package.
#
# The ``rocke_pytest_*`` entries collect ``platform/tests`` only. The
# directories below sit next to the code they test, so no other lane runs them.
# Each directory runs in its own pytest process: the directories use
# ``unittest`` and plain pytest styles and share module names, and a process per
# directory keeps them apart.
# ``portable_ir/tests`` is not listed: ``portable_ir/test_portable_ir.py::test_unit``
# runs it.

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_PLATFORM = Path(__file__).resolve().parents[2]

# Importable package name -> optional modules that the directory needs.
_SUITES = {
    "rocke.benchmark.perf.tests": (),
}
# ``rocke.heuristics.tests`` and ``test_wavescope_pmc.py`` run in no lane.
# ``TESTING.md`` gap G7 gives the reasons and the step that enrolls them.
_EXCLUDED_FILES = ("test_wavescope_pmc.py",)


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
    if path.name not in _EXCLUDED_FILES
)
if not _CASES:
    raise RuntimeError("in-tree test directories contain no test files")


def _report(returncode: int, stdout: str, stderr: str) -> None:
    """Fail on a non-zero exit code. Skip when the child passed no test."""
    assert returncode == 0, stdout[-4000:] or stderr[-4000:]
    # pytest exits with 0 when every test skips. Report that as a skip.
    if not re.search(r"\d+ passed", stdout):
        pytest.skip("child ran no test: " + stdout[-1500:])


@pytest.mark.parametrize(
    ("package", "relpath"), _CASES, ids=[f"{p}:{r}" for p, r in _CASES]
)
def test_in_tree_file(package, relpath):
    """Run one in-package test file in a child pytest process."""
    for module in _SUITES[package]:
        pytest.importorskip(module)
    env = rocke_subprocess_env()
    target = _package_dir(package) / relpath
    # ``-c`` replaces ``platform/tests/pytest.ini``: the child does not read the
    # ``rocke_pytest_*`` options.
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
    _report(result.returncode, result.stdout, result.stderr)


def test_report_fails_on_nonzero_exit():
    with pytest.raises(AssertionError, match="boom"):
        _report(1, "boom", "")


def test_report_uses_stderr_when_stdout_is_empty():
    with pytest.raises(AssertionError, match="trace"):
        _report(2, "", "trace")


def test_report_skips_when_no_test_passed():
    with pytest.raises(pytest.skip.Exception, match="SKIPPED"):
        _report(0, "1 skipped in 0.1s\nSKIPPED [1] reason", "")


def test_report_accepts_a_passed_test():
    _report(0, "3 passed, 1 skipped in 0.2s", "")
