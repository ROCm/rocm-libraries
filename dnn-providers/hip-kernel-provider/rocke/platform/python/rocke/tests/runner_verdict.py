# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The one verdict and exit-code contract shared by every rocKE test runner.

``platform/tests/run_all.py``, ``library/tests/run_all.py``,
``rocke.examples.run_all`` and ``tools/run_checks.py`` each record what their
stages did here, then print and return the verdict this module derives. A
runner never computes its own exit status.

Verdicts, in precedence order (the first that applies wins):

``RED``        a stage failed, or a required stage selected nothing to run.
``INCOMPLETE`` nothing failed, but a stage could not run or executed no tests.
``PARTIAL``    nothing failed, but the caller narrowed the run (a filter,
               ``--no-*``, a subset of steps).
``GREEN``      every stage ran, executed work, and passed.

Exit codes: ``GREEN`` and ``PARTIAL`` 0, ``RED`` 1, ``INCOMPLETE`` 3, and
argparse's 2 for a usage error. ``--allow-incomplete`` turns ``INCOMPLETE``
into 0 for local iteration; the printed verdict stays ``INCOMPLETE``, so it
never reads as ``GREEN``.

Stdlib only, and importing nothing else from ``rocke``: every runner loads it
before anything heavier, and the library runner reaches it through the one-way
library -> platform dependency.
"""

from __future__ import annotations

import enum
import getpass
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

__all__ = [
    "EXIT_GREEN",
    "EXIT_INCOMPLETE",
    "EXIT_RED",
    "PYTEST_NO_TESTS_COLLECTED",
    "SKIP_RETURN_CODE",
    "Outcome",
    "RunRecord",
    "Stage",
    "Verdict",
    "junit_counts",
    "line_buffered_stdout",
    "user_scratch_dir",
]

EXIT_GREEN = 0
EXIT_RED = 1
EXIT_INCOMPLETE = 3

# A child stage reports "could not run here" with 77, the automake and CTest
# ``SKIP_RETURN_CODE`` convention.
SKIP_RETURN_CODE = 77

# pytest's own code for "no tests were collected".
PYTEST_NO_TESTS_COLLECTED = 5


class Outcome(enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    # Could not run here: a missing tool, device or build product.
    SKIPPED = "skipped"
    # Ran, but executed no tests (all collected tests skipped).
    EMPTY = "executed no tests"
    # Not started because an earlier stage failed (fail-fast).
    NOT_RUN = "not run"


class Verdict(enum.Enum):
    GREEN = "GREEN"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    RED = "RED"


@dataclass(frozen=True)
class Stage:
    name: str
    outcome: Outcome
    detail: str = ""


@dataclass
class RunRecord:
    """What one runner invocation did, and the verdict that follows from it."""

    allow_incomplete: bool = False
    stages: list[Stage] = field(default_factory=list)
    partial_reasons: list[str] = field(default_factory=list)

    def add(self, name: str, outcome: Outcome, detail: str = "") -> Outcome:
        self.stages.append(Stage(name, outcome, detail))
        return outcome

    def passed(self, name: str, detail: str = "") -> Outcome:
        return self.add(name, Outcome.PASSED, detail)

    def failed(self, name: str, detail: str = "") -> Outcome:
        return self.add(name, Outcome.FAILED, detail)

    def skipped(self, name: str, reason: str) -> Outcome:
        return self.add(name, Outcome.SKIPPED, reason)

    def not_run(self, name: str, reason: str) -> Outcome:
        return self.add(name, Outcome.NOT_RUN, reason)

    def returncode(
        self,
        name: str,
        code: int,
        *,
        skip_allowed: bool = False,
        delegated_runner: bool = False,
        executed: int | None = None,
    ) -> Outcome:
        """Record a child process's exit status as a stage outcome.

        ``skip_allowed``: the child may report "could not run here" with
        ``SKIP_RETURN_CODE``. ``delegated_runner``: the child is another rocKE
        runner, whose ``EXIT_INCOMPLETE`` is recorded as a skip so the parent
        cannot read GREEN. Only then is 3 special: for pytest it is an internal
        error. ``executed`` is the number of tests that actually ran, when the
        caller knows it (see :func:`junit_counts`); zero turns a pass into
        ``EMPTY``.
        """
        if code == 0:
            if executed == 0:
                return self.add(name, Outcome.EMPTY)
            return self.passed(name)
        if code == SKIP_RETURN_CODE and skip_allowed:
            return self.skipped(name, f"exit {SKIP_RETURN_CODE}: could not run here")
        if code == EXIT_INCOMPLETE and delegated_runner:
            return self.skipped(name, "the delegated runner reported INCOMPLETE")
        if code == PYTEST_NO_TESTS_COLLECTED:
            return self.failed(name, "selected no tests")
        return self.failed(name, f"exit {code}")

    def partial(self, reason: str) -> None:
        """Note that the caller asked for less than a complete run."""
        if reason not in self.partial_reasons:
            self.partial_reasons.append(reason)

    @property
    def failed_any(self) -> bool:
        return any(stage.outcome is Outcome.FAILED for stage in self.stages)

    def verdict(self) -> Verdict:
        outcomes = {stage.outcome for stage in self.stages}
        if Outcome.FAILED in outcomes:
            return Verdict.RED
        if outcomes & {Outcome.SKIPPED, Outcome.EMPTY, Outcome.NOT_RUN}:
            return Verdict.INCOMPLETE
        if self.partial_reasons:
            return Verdict.PARTIAL
        if not self.stages:
            # A run that recorded nothing proved nothing.
            return Verdict.INCOMPLETE
        return Verdict.GREEN

    def exit_code(self) -> int:
        verdict = self.verdict()
        if verdict is Verdict.RED:
            return EXIT_RED
        if verdict is Verdict.INCOMPLETE and not self.allow_incomplete:
            return EXIT_INCOMPLETE
        return EXIT_GREEN

    def report(self, runner: str, stream: TextIO | None = None) -> int:
        """Print the verdict and every stage that did not pass; return the exit code."""
        out = stream or sys.stdout
        verdict = self.verdict()
        code = self.exit_code()
        print(f"\n{runner} stages:", file=out)
        for stage in self.stages:
            detail = f" ({stage.detail})" if stage.detail else ""
            print(f"  {stage.outcome.value:<17} {stage.name}{detail}", file=out)
        if self.partial_reasons:
            print(f"  narrowed by: {', '.join(self.partial_reasons)}", file=out)
        suffix = ""
        if verdict is Verdict.INCOMPLETE and self.allow_incomplete:
            suffix = " (allowed by --allow-incomplete)"
        print(f"\nRESULT: {verdict.value}{suffix}", file=out, flush=True)
        return code


def junit_counts(path: os.PathLike[str] | str) -> tuple[int, int] | None:
    """``(executed, skipped)`` test cases in a pytest JUnit XML report.

    ``None`` when the report is missing or unreadable, so a caller can tell
    "unknown" from "zero executed".
    """
    try:
        cases = list(ET.parse(Path(path)).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return None
    skipped = sum(1 for case in cases if case.find("skipped") is not None)
    return len(cases) - skipped, skipped


def line_buffered_stdout() -> None:
    """Flush stdout at every newline, so stage banners precede child output in logs."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)


def user_scratch_dir(name: str) -> Path:
    """``<tempdir>/<name>-<user>``: a default build root no other user shares.

    A shared ``<tempdir>/<name>`` lets one user's stale or unwritable tree
    break another's run on the same host.
    """
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name: fall back to the uid
        user = str(os.getuid()) if hasattr(os, "getuid") else "user"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user) or "user"
    return Path(tempfile.gettempdir()) / f"{name}-{safe}"
