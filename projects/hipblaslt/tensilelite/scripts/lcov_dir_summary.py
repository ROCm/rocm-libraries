#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Print per-directory line coverage totals for one or more LCOV files.

Used by ``tox -e coverage-cpp`` to show, side by side, the report that is
uploaded to Codecov and an advisory report that also counts headers.

Usage:
    python scripts/lcov_dir_summary.py --root . coverage_cpp/combined.lcov \
        coverage_cpp/advisory_with_headers.lcov

A file's directory group is its path relative to ``--root`` up to and
including the first ``src`` or ``include`` component (``src``,
``include``, ``client/src``, ``rocisa/rocisa/include``, ...). Files with
no such component are grouped by their parent directory, and files outside
``--root`` keep their absolute parent directory.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

GROUP_ANCHORS = ("src", "include")


def group_for(source: Path, root: Path) -> str:
    """Return the directory group a source file is reported under."""
    try:
        parts = source.relative_to(root).parts
    except ValueError:
        return str(source.parent)
    for index, part in enumerate(parts[:-1]):
        if part in GROUP_ANCHORS:
            return "/".join(parts[: index + 1])
    return "/".join(parts[:-1]) or "."


def read_lcov(path: Path, root: Path) -> dict[str, list[int]]:
    """Sum LF (lines found) and LH (lines hit) per directory group."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    group = None
    for line in path.read_text().splitlines():
        if line.startswith("SF:"):
            group = group_for(Path(line[3:]).resolve(), root)
        elif line.startswith("LF:") and group is not None:
            totals[group][0] += int(line[3:])
        elif line.startswith("LH:") and group is not None:
            totals[group][1] += int(line[3:])
        elif line == "end_of_record":
            group = None
    return totals


def format_cell(found: int, hit: int) -> str:
    if found == 0:
        return "-"
    return f"{hit}/{found} ({100.0 * hit / found:.2f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print per-directory line coverage totals for LCOV files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory that source paths are reported relative to.",
    )
    parser.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="LCOV files to summarize, one column each.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    columns = []
    for report in args.reports:
        if not report.is_file():
            print(f"lcov_dir_summary: {report} not found", file=sys.stderr)
            return 1
        columns.append(read_lcov(report, root))

    groups = sorted(set().union(*columns))
    headers = ["directory"] + [report.name for report in args.reports]
    rows = [
        [group] + [format_cell(*column.get(group, [0, 0])) for column in columns]
        for group in groups
    ]
    rows.append(
        ["TOTAL"]
        + [
            format_cell(
                sum(found for found, _ in column.values()),
                sum(hit for _, hit in column.values()),
            )
            for column in columns
        ]
    )

    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    for row in [headers] + rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
