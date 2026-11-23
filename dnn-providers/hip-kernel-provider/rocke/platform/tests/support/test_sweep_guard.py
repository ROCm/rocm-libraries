# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.sweep_guard: a sweep that checked nothing fails instead of
passing, in a real pytest run, and a pytest too old to report subtests fails
the session rather than run without the guard."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import pytest
from rocke.tests.subprocess_launcher import rocke_subprocess_env
from rocke.tests.sweep_guard import assert_case_ran

_CHILD_TIMEOUT_SECONDS = 120

_SWEEPS = """
import unittest

import pytest


class TestSweeps(unittest.TestCase):
    def test_every_subtest_skips(self):
        for case in range(3):
            with self.subTest(case=case):
                self.skipTest("this case cannot run here")

    def test_some_subtests_skip(self):
        for case in range(3):
            with self.subTest(case=case):
                if case:
                    self.skipTest("this case cannot run here")

    def test_skips_before_its_sweep(self):
        self.skipTest("no case of this sweep can run here")

    def test_without_subtests(self):
        pass


def test_every_fixture_subtest_skips(subtests):
    for case in range(2):
        with subtests.test(case=case):
            pytest.skip("this case cannot run here")
"""

_CONFTEST = """
from rocke.tests.sweep_guard import register_sweep_guard


def pytest_configure(config):
    register_sweep_guard(config)
"""


# A pytest older than 9 has no SubtestReport. The child session removes it from
# its own pytest module before the guard registers, to stand in for one.
_CONFTEST_WITHOUT_SUBTEST_REPORTS = """
import pytest

del pytest.SubtestReport

from rocke.tests.sweep_guard import register_sweep_guard


def pytest_configure(config):
    register_sweep_guard(config)
"""


def _run_pytest_in_tree(tree: Path, conftest: str) -> subprocess.CompletedProcess:
    """Run pytest on the sweeps above in a fresh ``tree`` with ``conftest``."""
    tree.mkdir()
    (tree / "conftest.py").write_text(textwrap.dedent(conftest), encoding="utf-8")
    (tree / "test_sweeps.py").write_text(textwrap.dedent(_SWEEPS), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rfE", "-q"],
        cwd=tree,
        env=rocke_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )


def test_a_test_whose_subtests_all_skipped_fails(tmp_path: Path):
    result = _run_pytest_in_tree(tmp_path / "tree", _CONFTEST)
    failed = {
        line.split()[1].partition(" ")[0]
        for line in result.stdout.splitlines()
        if line.startswith("FAILED ")
    }
    assert result.returncode == 1, result.stdout[-3000:] + result.stderr[-3000:]
    assert failed == {
        "test_sweeps.py::TestSweeps::test_every_subtest_skips",
        "test_sweeps.py::test_every_fixture_subtest_skips",
    }, result.stdout[-3000:]
    assert "every subtest skipped (3 of them): the sweep checked nothing" in (
        result.stdout
    )


def test_a_pytest_without_subtest_reports_fails_the_session(tmp_path: Path):
    result = _run_pytest_in_tree(tmp_path / "tree", _CONFTEST_WITHOUT_SUBTEST_REPORTS)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout[-3000:] + result.stderr[-3000:]
    )
    assert (
        f"rocke sweep guard failed - pytest={pytest.__version__}, expected 9 or newer"
        in result.stderr
    ), result.stderr[-3000:]
    assert "passed" not in result.stdout, result.stdout[-3000:]


class TestAssertCaseRan(unittest.TestCase):
    def test_a_case_that_ran_and_passed_passes(self):
        assert_case_ran(self, True, "")

    def test_a_failed_case_fails_with_its_reason(self):
        with self.assertRaisesRegex(AssertionError, "dW mismatch"):
            assert_case_ran(self, False, "dW mismatch")

    def test_a_case_stepped_past_fails(self):
        with self.assertRaisesRegex(
            AssertionError, r"case was skipped rather than run: skip \(no atom\)"
        ):
            assert_case_ran(self, True, "skip (no atom)")
