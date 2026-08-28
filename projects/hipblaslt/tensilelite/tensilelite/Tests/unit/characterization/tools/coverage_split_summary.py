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

* a headline table of each suite's whole-project coverage percentage,
* a line-level breakdown, as a share of every measurable statement, of what each
  suite reaches: both suites, characterization only, unit only, or no test at
  all (the untested surface). Those rows sum to 100% of the project, so the
  breakdown, not the two overlapping percentages, is what shows each suite's
  unique contribution and how much code no test touches yet. (When no combined
  report is passed the total-statement count is unknown, so this falls back to
  shares of the union of executed lines and omits the untested-surface row.), and
* a leaderboard of the largest files by measurable statement count, with each
  file's unit-suite and characterization-suite coverage side by side plus its
  characterization-only statement count. The whole-project numbers above say how
  far the migration has come; this says *where* the work is. Big files dominate
  the untested surface, so ranking by size names the refactor targets, and the
  per-file characterization-only count is the debt to convert in each one.

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


def _per_file_lines(report: dict) -> dict[str, set[int]]:
    """``{file: set of executed line numbers}`` for one report."""
    return {
        path: set(info.get("executed_lines", []))
        for path, info in report.get("files", {}).items()
    }


def _file_stmts(*reports: dict | None) -> dict[str, int]:
    """``{file: measurable statement count}`` gathered from any report that has it.

    coverage.py records this per file as ``summary.num_statements``. It is a
    property of the source, not of the test selection, so every report that
    measured a file agrees on it; taking the largest value seen is only a guard
    for a report that omits the field for some file.
    """
    stmts: dict[str, int] = {}
    for report in reports:
        if not report:
            continue
        for path, info in report.get("files", {}).items():
            n = (info.get("summary") or {}).get("num_statements")
            if isinstance(n, (int, float)) and int(n) > stmts.get(path, 0):
                stmts[path] = int(n)
    return stmts


def top_files_rows(
    char: dict, unit: dict, combined: dict | None, limit: int
) -> list[list[str]]:
    """Leaderboard rows for the largest files, biggest first.

    "Largest" is measurable statements rather than worst percentage, because a
    500-statement file at 60% hides far more untested code than a 20-statement
    file at 30%, and the point of this table is to name the refactor targets.

    The percentages are statement-level (executed statements over that file's
    measurable statements), which is deliberately the same metric as the
    line-level breakdown above rather than coverage.py's per-file
    ``percent_covered``. That keeps the unit, characterization, and
    characterization-only figures on one shared denominator, so each row reads as
    a single set breakdown of the same file. It also means these percentages run
    a little higher than the branch-inclusive whole-project numbers in the
    headline table.

    Files whose statement count is unknown or zero are skipped: there is no
    honest denominator for them. Ties are broken by path so the table is stable
    across runs.
    """
    stmts = _file_stmts(combined, char, unit)
    char_lines = _per_file_lines(char)
    unit_lines = _per_file_lines(unit)

    ranked = sorted(
        ((path, n) for path, n in stmts.items() if n > 0),
        key=lambda row: (-row[1], row[0]),
    )[:limit]

    rows: list[list[str]] = []
    for path, n in ranked:
        reached_by_char = char_lines.get(path, set())
        reached_by_unit = unit_lines.get(path, set())
        rows.append(
            [
                path,
                _fmt_int(n),
                f"{len(reached_by_unit) / n * 100:.1f}%",
                f"{len(reached_by_char) / n * 100:.1f}%",
                _fmt_int(len(reached_by_char - reached_by_unit)),
            ]
        )
    return rows


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


def _top_files_section(
    char: dict, unit: dict, combined: dict | None, limit: int
) -> list[str]:
    """The largest-files leaderboard, or nothing when it cannot be built.

    Returns an empty list when disabled (``limit <= 0``) or when no file has a
    known statement count, so an older or partial report degrades to the rest of
    the card instead of failing the step.
    """
    if limit <= 0:
        return []
    table_rows = top_files_rows(char, unit, combined, limit)
    if not table_rows:
        return []
    return [
        f"### Largest {len(table_rows)} files: unit vs characterization coverage",
        "",
        "Ranked by measurable statements, biggest first, whether or not this PR "
        "touched them. These files hold most of the untested surface, so they are "
        "the refactor targets. Percentages are statement-level shares of each "
        'file\'s own statements: "Unit %" is what the pure unit tests reach, '
        '"Char %" what the characterization suite reaches. The two overlap and do '
        'not sum. "Char only" is the statements reached by characterization but by '
        "no unit test, which is that file's migration debt: the work left to "
        "convert scaffolding into real unit tests.",
        "",
        *_aligned_table(
            ["File", "Stmts", "Unit %", "Char %", "Char only"],
            table_rows,
            right={1, 2, 3, 4},
        ),
        "",
    ]


def build_markdown(
    char: dict, unit: dict, combined: dict | None,
    char_tests: int | None, unit_tests: int | None,
    top_files: int = 20,
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
            "These rows count statements (lines), not branches, so they sum to "
            '100% of the project. "Covered by any suite" is the share reached by '
            'at least one suite; "No test coverage" is the rest (100% minus '
            '"Covered by any suite"). This line-level number runs a little higher '
            "than the whole-project percentage at the top, which is lower because "
            "it also penalizes untaken branches.",
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

    rows += _top_files_section(char, unit, combined, top_files)
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
    p.add_argument("--top-files", type=int, default=20, metavar="N",
                   help="rows in the largest-files leaderboard, 0 to omit it "
                        "(default: %(default)s)")
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
        args.top_files,
    )
    print(md)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
