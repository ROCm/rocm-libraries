# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fails the run on a skipped test when HIPDNN_REQUIRE_UHD_GEN_TESTS is set.

Every skip in uhd_gen's suites (tests/ and dataset/tests/), and in the two that load this
file with `-p uhd_gen.conftest` (the hipdnn-uhd skill scripts' and mine_shapes'), is a missing
prerequisite: a Python package (pytest.importorskip), the hipdnn_uhd_features binary (the
`evaluator` fixture), or the checkout's rocKE library. Locally that stays a skip. CMake's HIPDNN_REQUIRE_UHD_GEN_TESTS sets the variable
on the ctest entries, so in CI a gap fails the run rather than passing having checked
nothing. The skipped tests still report as skipped; the run's exit status is what changes.
"""
import os

import pytest

REQUIRE_VARIABLE = "HIPDNN_REQUIRE_UHD_GEN_TESTS"


def prerequisites_required() -> bool:
    """Whether the variable is set to anything but empty or "0"."""
    return os.environ.get(REQUIRE_VARIABLE, "") not in ("", "0")


class _RequirePrerequisites:
    """Records every skip, at collection or at run time, and fails the session on any."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def _record(self, report) -> None:
        if report.skipped:
            # A skip's longrepr is (path, line, reason).
            reason = (
                report.longrepr[2]
                if isinstance(report.longrepr, tuple)
                else report.longrepr
            )
            self.skipped.append(f"{report.nodeid}: {reason}")

    def pytest_collectreport(self, report) -> None:
        self._record(report)

    def pytest_runtest_logreport(self, report) -> None:
        self._record(report)

    def pytest_sessionfinish(self, session) -> None:
        if self.skipped and session.exitstatus in (
            pytest.ExitCode.OK,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if not self.skipped:
            return
        terminalreporter.section(
            f"{REQUIRE_VARIABLE} is set: skips fail the run", red=True
        )
        for line in self.skipped:
            terminalreporter.line(line)


def pytest_configure(config) -> None:
    if prerequisites_required():
        config.pluginmanager.register(
            _RequirePrerequisites(), "uhd_gen-require-prerequisites"
        )
