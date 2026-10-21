#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Aggregate the timing results of several runs into one overall report.

Every log file passed in is analyzed with analyze_timing.py, turned into its tiered dict
(``timings_to_tree``), and the dicts are summed node by node (count and total_ms). The summed tree
is then printed with the same report as a single run, so the output has exactly the same layout as
``analyze_timing.py <log>``.

Usage:
    python scripts/aggregate_timing.py LOG [LOG ...]

Logs may be plain text or ``.gz``. Which runs are included is decided entirely by the files passed
in; there are no defaults.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_timing import (  # noqa: E402
    analyze_timing_file,
    print_summary,
    timings_to_tree,
    tree_to_timings,
)


def _add_node(total: dict, node: dict) -> None:
    """Add ``node`` (count, total_ms and children) into ``total`` in place."""
    total["count"] += node["count"]
    total["total_ms"] += node["total_ms"]
    for name, child in node["children"].items():
        _add_node(total["children"].setdefault(
            name, {"count": 0, "total_ms": 0.0, "children": {}}), child)


def add_trees(total: dict, tree: dict) -> None:
    """Add the tiered dict ``tree`` (from ``timings_to_tree``) into ``total`` in place."""
    for name, node in tree["phases"].items():
        _add_node(total["phases"].setdefault(
            name, {"count": 0, "total_ms": 0.0, "children": {}}), node)
    for section in ("overlapping_totals", "unknown_categories"):
        for name, stats in tree[section].items():
            dst = total[section].setdefault(name, {"count": 0, "total_ms": 0.0})
            dst["count"] += stats["count"]
            dst["total_ms"] += stats["total_ms"]


def empty_tree() -> dict:
    return {"phases": {}, "overlapping_totals": {}, "unknown_categories": {}}


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate the timing results of several runs into one report',
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('logfiles', nargs='+',
                        help='Log files (plain or .gz) containing timing output, one per run')
    args = parser.parse_args()

    total = empty_tree()
    problem_timings = []
    for path in args.logfiles:
        timings, problems = analyze_timing_file(path)
        if not timings:
            print(f"warning: no timing data in {path}", file=sys.stderr)
            continue
        add_trees(total, timings_to_tree(timings))
        problem_timings.extend(problems)

    print_summary(tree_to_timings(total), problem_timings)


if __name__ == '__main__':
    main()
