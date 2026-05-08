################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Regression tests guarding against a previously-observed test-flake root cause.

Background:
    `test_MatrixInstructionConversion.py` used to call
    `makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)` at module import time. That
    runs the rocisa init for every supported ISA at pytest collection time
    and mutates singleton state, which is suspected to leak into other tests'
    state (see XYV_TEST_FLAKE_INVESTIGATION.md, commit f2d307cf3d).

These tests defend against that pattern coming back:
    1. ``test_matrixinstructionconversion_does_not_call_makeIsaInfoMap_on_import``
       monkey-patches ``Tensile.Common.Capabilities.makeIsaInfoMap`` to raise on
       call and then imports the test module. If the refactor regresses
       (someone re-introduces a top-level call), the import will fail loudly.
    2. ``test_unit_suite_results_are_cwd_independent`` invokes pytest twice
       from two distinct cwds and asserts the result counts match. Differences
       imply hidden cwd-dependent state (the original flake's signature).
"""

import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


SLOW_TEST_FILE = (
    "projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py"
)


def _find_repo_root() -> Path:
    """Locate the repo root by walking upward looking for the slow-file path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / SLOW_TEST_FILE).exists():
            return parent
    raise RuntimeError(f"Could not locate repo root containing {SLOW_TEST_FILE}")


def test_matrixinstructionconversion_does_not_call_makeIsaInfoMap_on_import(monkeypatch):
    """Importing the slow test module must not trigger makeIsaInfoMap.

    If anyone reintroduces a top-level ``makeIsaInfoMap(...)`` call, this
    monkeypatched stub will raise during import and fail the test. This is
    the no-collect-time-init invariant promised in the bead scope.
    """
    import Tensile.Common.Capabilities as caps_mod

    def _boom(*args, **kwargs):
        raise AssertionError(
            "makeIsaInfoMap was invoked at import/collection time; this leaks "
            "rocisa singleton state across tests. Build it inside a fixture."
        )

    monkeypatch.setattr(caps_mod, "makeIsaInfoMap", _boom)

    # Ensure a clean import (the module may already be loaded from earlier
    # collection in this very session).
    mod_name = "Tensile.Tests.unit.test_MatrixInstructionConversion"
    sys.modules.pop(mod_name, None)
    # Also drop a possible top-level import name.
    sys.modules.pop("test_MatrixInstructionConversion", None)

    # Load the module by file path so we don't depend on whether
    # Tensile.Tests.unit is a real package on sys.path.
    repo_root = _find_repo_root()
    target = (
        repo_root
        / "projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_MatrixInstructionConversion_probe", target
    )
    module = importlib.util.module_from_spec(spec)
    # If the refactor regressed, spec.loader.exec_module will raise the
    # AssertionError from _boom above.
    spec.loader.exec_module(module)


def _run_pytest_collect(cwd: Path, target_relpath: str) -> str:
    """Run pytest --collect-only from ``cwd`` and return its stdout.

    We deliberately ``--ignore`` the slow file so this regression test never
    triggers the multi-minute MatrixInstructionConversion run.
    """
    repo_root = _find_repo_root()
    slow_abs = str(repo_root / SLOW_TEST_FILE)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        f"--ignore={slow_abs}",
        target_relpath,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Collection-only must succeed; otherwise the test environment itself is
    # broken and we should fail loudly rather than silently pass.
    assert result.returncode == 0, (
        f"pytest collection failed in cwd={cwd}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected", re.IGNORECASE)


def _collected_count(stdout: str) -> int:
    matches = _COLLECTED_RE.findall(stdout)
    assert matches, f"Could not find collected-count in pytest output:\n{stdout}"
    # When pytest emits multiple summary lines (errors etc.), take the last.
    return int(matches[-1])


def test_unit_collection_count_is_cwd_independent_when_slow_file_ignored(tmp_path):
    """Sanity check: pytest --collect-only yields the same count from two cwds.

    Scope caveat — this is a weaker invariant than the bead originally asked
    for. The bead's hypothesized regression is "the slow file's import-time
    makeIsaInfoMap mutates singletons that other tests depend on, producing
    cwd-dependent flakes." Catching that requires (a) importing the slow
    file AND (b) actually running other tests after the import. This test
    does neither — it --ignore's the slow file from both invocations (per
    the standing memory not to run the slow suite in routine validation)
    and uses --collect-only (so runtime leakage symptoms cannot manifest).

    What this test DOES catch: gross cwd-dependent collection bugs in the
    rest of the suite (e.g., a future test file that conditionally collects
    based on `os.getcwd()`). That has some value as a smoke check but it is
    not the bead's intended invariant. The first regression test above —
    `test_matrixinstructionconversion_does_not_call_makeIsaInfoMap_on_import`
    — is the load-bearing guard for the actual hypothesis.
    """
    repo_root = _find_repo_root()
    unit_dir_abs = repo_root / "projects/hipblaslt/tensilelite/Tensile/Tests/unit"

    # Invocation A: run from the repo root, target with absolute path.
    out_a = _run_pytest_collect(repo_root, str(unit_dir_abs))

    # Invocation B: run from a fresh tmp dir, also targeting absolute path
    # (the only path that resolves from an unrelated cwd).
    out_b = _run_pytest_collect(tmp_path, str(unit_dir_abs))

    count_a = _collected_count(out_a)
    count_b = _collected_count(out_b)

    assert count_a == count_b, (
        f"Pytest collection count differs by cwd: {count_a} (from {repo_root}) "
        f"vs {count_b} (from {tmp_path}). This is the cwd-dependence signature "
        f"flagged in XYV_TEST_FLAKE_INVESTIGATION.md."
    )
