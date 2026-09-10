# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The census driver's own failure modes.

`hkp_census_check.py` exists for exactly one reason: a `--gtest_filter` that selects
nothing is a gtest PASS -- exit 0, "0 tests ran" -- so a census suite that was renamed,
dropped from the binary or misspelled in the registration table would report green
forever. The driver converts that into a failure.

Nothing tested it. A guard against false greens that is itself unverified is the same
bet one level up, so the cases below drive the real script against a stand-in binary
that mimics gtest's contract: `--gtest_list_tests` prints suites flush-left and cases
indented, and a filter matching nothing still exits 0.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hkp_census_check.py"

# Suite line flush-left, case lines indented: the shape the driver parses to tell
# "selected nothing" from "ran and passed".
_FAKE_BINARY = """#!/usr/bin/env python3
import sys

CASES = {"TestBinaryOpsPacks": ["LoadsPacks", "LoadsKernels"]}

selected = [s for s in CASES if any(
    a.startswith("--gtest_filter=") and a[len("--gtest_filter="):].split(".")[0] in (s, "*")
    for a in sys.argv[1:])]

if "--gtest_list_tests" in sys.argv:
    for suite in selected:
        print(suite + ".")
        for case in CASES[suite]:
            print("  " + case)
    # gtest exits 0 whether or not the filter matched anything.
    sys.exit(0)

sys.exit(RUN_EXIT)
"""


def _fake_binary(tmp_path: Path, run_exit: int = 0) -> Path:
    binary = tmp_path / "fake_tests.py"
    binary.write_text(_FAKE_BINARY.replace("RUN_EXIT", str(run_exit)))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


def _run(binary: Path, root: Path, gtest_filter: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--arch",
            "gfx942",
            "--descriptor-root",
            str(root),
            "--test-binary",
            str(binary),
            "--gtest-filter",
            gtest_filter,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.quick
def test_a_filter_selecting_cases_passes(tmp_path):
    """The control. Without it every red below could be red for the wrong reason."""
    result = _run(_fake_binary(tmp_path), tmp_path, "TestBinaryOpsPacks.*")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 case(s) selected" in result.stdout


@pytest.mark.quick
def test_a_filter_selecting_nothing_fails(tmp_path):
    """The whole reason this driver exists: gtest calls this a pass."""
    result = _run(_fake_binary(tmp_path), tmp_path, "TestRenamedAway.*")
    assert result.returncode != 0
    assert "selects no test case" in result.stdout


@pytest.mark.quick
def test_a_failing_run_propagates_its_exit_code(tmp_path):
    """Selection succeeding must not mask the run's own verdict."""
    result = _run(_fake_binary(tmp_path, run_exit=1), tmp_path, "TestBinaryOpsPacks.*")
    assert result.returncode == 1


@pytest.mark.quick
def test_a_missing_descriptor_shard_fails(tmp_path):
    """Absence is a packaging failure, not a reason to skip: the architecture came from
    the configured packaging list, so the build was asked to produce this shard."""
    result = _run(
        _fake_binary(tmp_path), tmp_path / "not-packed", "TestBinaryOpsPacks.*"
    )
    assert result.returncode != 0
    assert "no descriptor shard" in result.stdout


@pytest.mark.quick
def test_a_missing_test_binary_fails(tmp_path):
    result = _run(tmp_path / "never-built", tmp_path, "TestBinaryOpsPacks.*")
    assert result.returncode != 0
    assert "does not exist" in result.stdout


@pytest.mark.quick
def test_an_empty_filter_fails_before_running_anything(tmp_path):
    """An empty filter selects the WHOLE binary, which would report some other suite's
    result as this census's."""
    result = _run(_fake_binary(tmp_path), tmp_path, "   ")
    assert result.returncode != 0
    assert "would select every test" in result.stdout
