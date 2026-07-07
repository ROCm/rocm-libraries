#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render a characterization-vs-unit coverage breakdown as Markdown.

The coverage-unit lane measures one combined coverage number (characterization
tests plus the pure unit tests, which all live under Tensile/Tests/unit and
carry the ``unit`` marker). Characterization tests are separated only by path
(the ``characterization/`` subtree), so to attribute coverage to each suite we
run coverage.py twice with different test selections and feed the two JSON
reports (plus the combined report) here.

This produces:

* a headline table of each suite's whole-project coverage percentage, and
* a line-level "who covers what" breakdown - lines reached only by
  characterization, only by the unit tests, or by both - which is the honest
  way to show each suite's unique contribution (the two percentages overlap
  heavily and do not add up to the combined number).

Output goes to stdout and, when running in GitHub Actions, is appended to the
job summary (``$GITHUB_STEP_SUMMARY``) so it renders as a card in the run UI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET


def _load(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def _junit_ran(path: str) -> int | None:
    """Count executed (non-skipped) tests in a JUnit XML report.

    Returns the number of ``<testcase>`` entries that were not skipped, which
    equals the passed count on a green run. Returns ``None`` if the file is
    missing or unparseable so the card can still render without the count.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    ran = 0
    for tc in root.iter("testcase"):
        if tc.find("skipped") is None:
            ran += 1
    return ran


def _pct(report: dict) -> float:
    return float(report["totals"]["percent_covered"])


def _executed_line_set(report: dict) -> set[tuple[str, int]]:
    """Set of (file, line) pairs executed anywhere in this report."""
    lines: set[tuple[str, int]] = set()
    for path, info in report.get("files", {}).items():
        for ln in info.get("executed_lines", []):
            lines.add((path, ln))
    return lines


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def build_markdown(
    char: dict, unit: dict, combined: dict | None,
    char_tests: int | None, unit_tests: int | None,
) -> str:
    char_lines = _executed_line_set(char)
    unit_lines = _executed_line_set(unit)
    both = char_lines & unit_lines
    char_only = char_lines - unit_lines
    unit_only = unit_lines - char_lines
    union = char_lines | unit_lines

    def tests_cell(n): return _fmt_int(n) if n is not None else "-"

    rows = [
        "## TensileLite coverage: characterization vs unit",
        "",
        "| Suite | Tests | Whole-project coverage |",
        "| --- | ---: | ---: |",
        f"| Characterization | {tests_cell(char_tests)} | {_pct(char):.2f}% |",
        f"| Unit (non-characterization) | {tests_cell(unit_tests)} | {_pct(unit):.2f}% |",
    ]
    if combined is not None:
        rows.append(f"| **Combined** | | **{_pct(combined):.2f}%** |")
    rows += [
        "",
        "### Line-level contribution (executed lines)",
        "",
        "Percentages overlap, so they do not sum to the combined number. This is "
        "who actually reaches each line:",
        "",
        "| Reached by | Executed lines | Share of union |",
        "| --- | ---: | ---: |",
        f"| Both suites | {_fmt_int(len(both))} | {len(both) / len(union) * 100:.1f}% |",
        f"| Characterization only | {_fmt_int(len(char_only))} | {len(char_only) / len(union) * 100:.1f}% |",
        f"| Unit only | {_fmt_int(len(unit_only))} | {len(unit_only) / len(union) * 100:.1f}% |",
        f"| Union (any suite) | {_fmt_int(len(union))} | 100.0% |",
        "",
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--characterization", required=True, help="char coverage.json")
    p.add_argument("--unit", required=True, help="unit-only coverage.json")
    p.add_argument("--combined", default=None, help="combined coverage.json (optional)")
    p.add_argument("--characterization-tests", type=int, default=None)
    p.add_argument("--unit-tests", type=int, default=None)
    p.add_argument("--characterization-junit", default=None,
                   help="char JUnit xml; test count derived when --characterization-tests omitted")
    p.add_argument("--unit-junit", default=None,
                   help="unit JUnit xml; test count derived when --unit-tests omitted")
    args = p.parse_args(argv)

    char_tests = args.characterization_tests
    if char_tests is None and args.characterization_junit:
        char_tests = _junit_ran(args.characterization_junit)
    unit_tests = args.unit_tests
    if unit_tests is None and args.unit_junit:
        unit_tests = _junit_ran(args.unit_junit)

    md = build_markdown(
        _load(args.characterization),
        _load(args.unit),
        _load(args.combined) if args.combined else None,
        char_tests,
        unit_tests,
    )
    print(md)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
