# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render a per-test table from a ``ctest --output-junit`` report.

This lives in the rocKE tree (so it is exercised and tested by the component's
own CI) but is deliberately a standalone module, not part of the ``rocke``
package: the outer TheRock test runner imports it without paying for the heavy
``rocke`` package import, and it depends only on the standard library.

The table is written to a GitHub Actions step-summary file so per-test results
are visible on the run Summary page. It is strictly decorative — ctest's exit
code remains the sole pass/fail authority — and best-effort: a problem here is
logged and marked, never raised, so it can never fail the test job.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple, Optional


class TestResult(NamedTuple):
    name: str
    status: str  # "Passed" | "Failed" | "Skipped"
    seconds: float


def parse_junit(path: Path) -> list[TestResult]:
    """Parse a ``ctest --output-junit`` file into per-test results.

    Status comes from the child element ctest writes for each ``<testcase>``:
    a ``<failure>``/``<error>`` means the test failed, ``<skipped>`` means it
    was skipped, and otherwise it passed.
    """
    root = ET.parse(path).getroot()
    results: list[TestResult] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            status = "Failed"
        elif case.find("skipped") is not None:
            status = "Skipped"
        else:
            status = "Passed"
        try:
            seconds = float(case.get("time") or 0)
        except ValueError:
            seconds = 0.0
        results.append(TestResult(case.get("name", "?"), status, seconds))
    return results


_MARK = {"Passed": "✅", "Failed": "❌", "Skipped": "⚪"}


def render_markdown(results: list[TestResult], arch: str) -> str:
    passed = sum(1 for r in results if r.status == "Passed")
    lines = [
        f"### rocKE tests — {arch or 'unknown arch'}",
        "",
        f"{passed}/{len(results)} ctest entries passed — each row is one ctest "
        "entry (often a whole gtest binary, not a single case).",
        "",
        "| | test | time |",
        "| --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda t: -t.seconds):
        lines.append(
            f"| {_MARK.get(r.status, r.status)} | `{r.name}` | {r.seconds:.2f}s |"
        )
    return "\n".join(lines) + "\n"


def write_step_summary(
    junit_path: Path, arch: str, summary_path: Optional[str]
) -> None:
    """Best-effort: render the per-test table to ``summary_path``.

    A no-op when ``summary_path`` is falsy (local runs with no
    ``$GITHUB_STEP_SUMMARY``). Never raises — but never fails silently either:
    on error it logs and writes a visible marker so a missing table is
    diagnosable rather than mistaken for a clean pass.
    """
    if not summary_path:
        return
    try:
        markdown = render_markdown(parse_junit(Path(junit_path)), arch)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown)
    except Exception as exc:  # best-effort, but loud (see docstring)
        logging.warning("rocKE test-summary generation failed: %s", exc)
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(
                    "### rocKE tests\n\n"
                    f"⚠️ test-result summary generation failed: `{exc}` "
                    "— see job log.\n"
                )
        except Exception:
            logging.exception("could not write test-summary failure marker")
