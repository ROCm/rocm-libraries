# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A sweep that checked nothing fails, instead of passing.

A test that runs its cases as subtests (``self.subTest`` or the ``subtests``
fixture) passes when every subtest skips: pytest reports each skipped subtest,
and the test itself as passed, though it checked nothing. The plugin
:func:`register_sweep_guard` installs fails such a test: a test that ran
subtests, all of which skipped, fails with their count. Each pytest root
registers it from its ``pytest_configure``, through
``rocke.tests.pytest_roots.configure_pytest_root``.

A sweep that wants to skip as a whole (no GPU of the arch it needs) skips
before its first subtest, which leaves the test skipped, not failed. A test
whose combinations report as subtests only when they fail or skip must report
every combination (a ``with subtests.test(...)`` per case), or the guard sees
only its skips.

The guard reads the subtest reports pytest 9 makes (``pytest.SubtestReport``).
An older pytest makes none, so the guard could not see a sweep that checked
nothing: registering it there fails the session instead (``pytest.UsageError``).

A sweep whose runner reports each case as ``(ok, why)``, with a case it stepped
past (an arch that cannot build the spec) as ``(True, "skip (...)")``, checks
each outcome with :func:`assert_case_ran`, so a skipped case fails rather than
counting as a pass.
"""

from __future__ import annotations

import collections
import unittest

import pytest

__all__ = ["assert_case_ran", "register_sweep_guard"]

_PLUGIN_NAME = "rocke-sweep-guard"


def assert_case_ran(test: unittest.TestCase, ok: bool, why: str) -> None:
    """Fail ``test`` unless one sweep case ran and passed: ``ok`` is its
    verdict, and ``why`` its reason, which starts with ``"skip"`` when the
    runner stepped past the case instead of building, launching and comparing
    it."""
    test.assertTrue(ok, why)
    test.assertFalse(why.startswith("skip"), f"case was skipped rather than run: {why}")


class _EverySubtestSkippedFails:
    """Turns the passing report of a test whose subtests all skipped into a
    failure."""

    def __init__(self, subtest_report_type: type[pytest.TestReport]) -> None:
        self._subtest_report_type = subtest_report_type
        self._subtest_outcomes: dict[str, collections.Counter[str]] = {}

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if isinstance(report, self._subtest_report_type):
            outcomes = self._subtest_outcomes.setdefault(
                report.nodeid, collections.Counter()
            )
            outcomes[report.outcome] += 1
            return
        if report.when != "call":
            return
        outcomes = self._subtest_outcomes.pop(report.nodeid, None)
        if (
            report.passed
            and outcomes
            and outcomes["skipped"]
            and not outcomes["passed"]
            and not outcomes["failed"]
        ):
            report.outcome = "failed"
            report.longrepr = (
                f"every subtest skipped ({outcomes['skipped']} of them): the sweep "
                "checked nothing. Skip the test as a whole before its first subtest "
                "when none of its cases can run here."
            )


def register_sweep_guard(config: pytest.Config) -> None:
    """Install the guard in this pytest session, once (see the module
    docstring). Raises ``pytest.UsageError`` when this pytest makes no subtest
    reports (it is older than 9), since the guard could not see a sweep that
    checked nothing."""
    if config.pluginmanager.has_plugin(_PLUGIN_NAME):
        return
    subtest_report_type = getattr(pytest, "SubtestReport", None)
    if subtest_report_type is None:
        raise pytest.UsageError(
            "rocke sweep guard failed - "
            f"pytest={pytest.__version__}, expected 9 or newer, which reports "
            "subtests (without them a sweep whose every case skipped would "
            "pass); install the test requirements in "
            "rocke/platform/requirements.txt"
        )
    config.pluginmanager.register(
        _EverySubtestSkippedFails(subtest_report_type), _PLUGIN_NAME
    )
