# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""pytest plugin: collect (or run) as CI's test environment would, on a host that has more than CI has.

- The modules RKMON_CI_ABSENT names (comma-separated top-level names, e.g. torch,rocke_engine) cannot be imported, even
  where this interpreter has them: an `import torch` raises ModuleNotFoundError, so `pytest.importorskip("torch")`
  skips and a plain module-level import is a collection error, as in CI.
- After collection, each test's skip marks are evaluated as pytest would before running it (skip, and skipif with its
  condition), and every test prints one line at the end: "rkmon-ci:<TAB>RUN<TAB><node id>" or
  "rkmon-ci:<TAB>SKIP<TAB><node id><TAB><reason>"; a module skipped while it is collected (a module-level
  importorskip) prints a SKIP line for the module.
- In a run, each test's outcome prints one line at the end: "rkmon-ci-result:<TAB>passed|failed|skipped<TAB><node
  id><TAB><skip reason>" (a module skipped while collected counts as one skipped test).

Used by tools/lane_of.py (`rk lane-of --collect`), the fix-line premise (rkmon_lib/fixline.py) and the checks'
CI-condition run (tools/ci_condition.py).
"""

import importlib.abc
import os
import sys

import pytest

ABSENT = frozenset(
    m.strip() for m in os.environ.get("RKMON_CI_ABSENT", "").split(",") if m.strip()
)
PREFIX = "rkmon-ci:"
RESULT = "rkmon-ci-result:"
_LINES: list = []
_RESULTS: list = []


class _Absent(importlib.abc.MetaPathFinder):
    """Refuses the absent modules, as an environment without them would."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ABSENT:
            raise ModuleNotFoundError(
                f"No module named '{name}' (rkmon: CI's test environment has none)",
                name=name,
            )
        return None


if ABSENT:
    for _name in list(sys.modules):
        if _name.split(".")[0] in ABSENT:
            del sys.modules[_name]
    sys.meta_path.insert(0, _Absent())


def _one_line(s) -> str:
    return " ".join(str(s).split())


def _skip_reason(item) -> str | None:
    """The reason a test would be skipped before it runs, by its marks; None when it would run."""
    try:
        from _pytest.skipping import evaluate_skip_marks

        hit = evaluate_skip_marks(item)
        return None if hit is None else (hit.reason or "skip mark")
    except (
        Exception
    ) as exc:  # a condition that cannot be evaluated here: said, never a silent run
        return f"skip condition not evaluated ({type(exc).__name__}: {_one_line(exc)[:80]})"


def pytest_collection_modifyitems(config, items):
    for item in items:
        why = _skip_reason(item)
        if why is None:
            _LINES.append(f"{PREFIX}\tRUN\t{item.nodeid}")
        else:
            _LINES.append(f"{PREFIX}\tSKIP\t{item.nodeid}\t{_one_line(why)}")


def pytest_collectreport(report):
    if report.skipped:
        rep = report.longrepr
        why = rep[2] if isinstance(rep, tuple) and len(rep) == 3 else str(rep or "")
        why = _one_line(why)
        if why.startswith("Skipped: "):
            why = why[len("Skipped: ") :]
        _LINES.append(
            f"{PREFIX}\tSKIP\t{report.nodeid or report.fspath}\t{why or 'skipped while collected'}"
        )
        _RESULTS.append(
            f"{RESULT}\tskipped\t{report.nodeid or report.fspath}\t{why or 'skipped while collected'}"
        )


def pytest_runtest_logreport(report):
    """One outcome a test: its call's, or its setup's when the setup skipped or failed (an error counts as failed)."""
    if report.when == "call" or (report.when == "setup" and not report.passed):
        if report.skipped and not hasattr(report, "wasxfail"):
            rep = report.longrepr
            why = _one_line(
                rep[2] if isinstance(rep, tuple) and len(rep) == 3 else rep or ""
            )
            why = why[len("Skipped: ") :] if why.startswith("Skipped: ") else why
            _RESULTS.append(f"{RESULT}\tskipped\t{report.nodeid}\t{why}")
        elif report.failed:
            _RESULTS.append(f"{RESULT}\tfailed\t{report.nodeid}\t")
        elif report.passed or hasattr(report, "wasxfail"):
            _RESULTS.append(f"{RESULT}\tpassed\t{report.nodeid}\t")


def pytest_terminal_summary(terminalreporter):
    for ln in _LINES:
        terminalreporter.write_line(ln)
    if not terminalreporter.config.option.collectonly:
        for ln in _RESULTS:
            terminalreporter.write_line(ln)
