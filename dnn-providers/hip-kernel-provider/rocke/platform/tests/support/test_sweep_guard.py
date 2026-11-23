# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.sweep_guard: a sweep that checked nothing fails instead of
passing, in a real pytest run."""

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


@pytest.mark.skipif(
    not hasattr(pytest, "SubtestReport"),
    reason="this pytest reports no subtests (pytest 9 does)",
)
def test_a_test_whose_subtests_all_skipped_fails(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "conftest.py").write_text(textwrap.dedent(_CONFTEST), encoding="utf-8")
    (tree / "test_sweeps.py").write_text(textwrap.dedent(_SWEEPS), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rfE", "-q"],
        cwd=tree,
        env=rocke_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
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
