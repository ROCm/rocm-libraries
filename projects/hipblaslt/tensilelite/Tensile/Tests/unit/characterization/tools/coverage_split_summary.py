#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render a characterization-vs-unit coverage breakdown as Markdown.

The coverage-unit lane runs two disjoint test selections: the characterization
suite (the ``characterization/`` subtree) and the pure unit tests (the rest of
``Tensile/Tests/unit``, with the characterization subtree excluded). It feeds
their two JSON coverage reports here, plus the combined report.

Why this card exists: characterization tests are scaffolding, and the goal is to
replace them with real unit tests. This card is the migration dashboard. Its key
number is the *characterization-only* line count: the lines still reached only by
scaffolding, i.e. the migration debt that should fall toward zero as unit tests
take over.

This produces:

* a headline table of each suite's whole-project coverage percentage, and
* a line-level breakdown, as a share of every measurable statement, of what each
  suite reaches: both suites, characterization only, unit only, or no test at
  all (the untested surface). Those rows sum to 100% of the project, so the
  breakdown, not the two overlapping percentages, is what shows each suite's
  unique contribution and how much code no test touches yet. (When no combined
  report is passed the total-statement count is unknown, so this falls back to
  shares of the union of executed lines and omits the untested-surface row.)

The per-suite attribution is only meaningful because the two selections are
disjoint. If the pure-unit run re-ran the characterization tests, every
characterization line would also count as "unit" and the characterization-only
number would collapse to ~0. The ``--ignore`` in the coverage-unit tox env is
what keeps them disjoint.

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


def _aligned_table(
    headers: list[str], rows: list[list[str]], right: set[int]
) -> list[str]:
    """Render a Markdown table whose columns line up as raw text too.

    Cells are padded to a uniform per-column width so the pipes align even when
    the surface shows the raw markdown instead of rendering it (e.g. a CI step
    log). Columns whose index is in ``right`` are right-aligned (numbers); the
    rest are left-aligned. The output is still valid GitHub-flavored Markdown.
    """
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        out = [
            cells[i].rjust(widths[i]) if i in right else cells[i].ljust(widths[i])
            for i in range(len(cells))
        ]
        return "| " + " | ".join(out) + " |"

    sep = [
        ("-" * (widths[i] - 1) + ":") if i in right else ("-" * widths[i])
        for i in range(len(headers))
    ]
    return [fmt(headers), "| " + " | ".join(sep) + " |"] + [fmt(r) for r in rows]


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

    suite_rows = [
        ["Characterization", tests_cell(char_tests), f"{_pct(char):.2f}%"],
        ["Unit (non-characterization)", tests_cell(unit_tests), f"{_pct(unit):.2f}%"],
    ]
    if combined is not None:
        suite_rows.append(["**Combined**", "", f"**{_pct(combined):.2f}%**"])

    rows = [
        "## TensileLite coverage: characterization vs unit",
        "",
        *_aligned_table(
            ["Suite", "Tests", "Whole-project coverage"], suite_rows, right={1, 2}
        ),
        "",
    ]

    # Total measurable statements (covered + uncovered) comes from the combined
    # report. With it we can show every statement as one of: reached by both
    # suites, characterization only, unit only, or reached by no test at all
    # (the untested surface). Those shares sum to 100% of the project.
    total_stmts = combined.get("totals", {}).get("num_statements") if combined else None

    if total_stmts:
        covered = len(union)
        no_tests = max(0, total_stmts - covered)

        def share(n: int) -> str:
            return f"{n / total_stmts * 100:.2f}%"

        rows += [
            "### Line-level contribution (share of all measurable statements)",
            "",
            'Every statement falls into exactly one row below, so the shares sum '
            'to 100%. "No test coverage" is the untested surface; its share is '
            "100% minus the combined coverage above.",
            "",
            *_aligned_table(
                ["Reached by", "Statements", "Share of all"],
                [
                    ["Both suites", _fmt_int(len(both)), share(len(both))],
                    ["Characterization only", _fmt_int(len(char_only)), share(len(char_only))],
                    ["Unit only", _fmt_int(len(unit_only)), share(len(unit_only))],
                    ["Covered by any suite", _fmt_int(covered), share(covered)],
                    ["No test coverage", _fmt_int(no_tests), share(no_tests)],
                    ["**Total statements**", f"**{_fmt_int(total_stmts)}**", "**100.00%**"],
                ],
                right={1, 2},
            ),
            "",
        ]
    else:
        # No combined report, so the total-statement denominator is unknown; fall
        # back to shares of the union of executed lines (the untested surface
        # cannot be shown without the combined totals).
        rows += [
            "### Line-level contribution (executed lines)",
            "",
            "No combined report was provided, so the untested surface is unknown. "
            "These are shares of the union of executed lines:",
            "",
            *_aligned_table(
                ["Reached by", "Executed lines", "Share of union"],
                [
                    ["Both suites", _fmt_int(len(both)),
                     (f"{len(both) / len(union) * 100:.1f}%" if union else "-")],
                    ["Characterization only", _fmt_int(len(char_only)),
                     (f"{len(char_only) / len(union) * 100:.1f}%" if union else "-")],
                    ["Unit only", _fmt_int(len(unit_only)),
                     (f"{len(unit_only) / len(union) * 100:.1f}%" if union else "-")],
                    ["Union (any suite)", _fmt_int(len(union)), ("100.0%" if union else "-")],
                ],
                right={1, 2},
            ),
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
