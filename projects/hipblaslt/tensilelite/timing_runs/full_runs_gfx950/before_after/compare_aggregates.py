#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two per-config timing passes side by side.

Each argument is the OUT directory of a run_per_config.sh or run_per_config_build.sh pass: it holds
AGGREGATE_analysis.txt (written by scripts/aggregate_timing.py) and one <config>/times.txt per
config. The script prints the "TIMING BY PURPOSE" table of both aggregates side by side with the
change in percent, then the summed pytest wall time, the slowest configs and every config that
got more than 2 s slower.

Usage: compare_aggregates.py BEFORE_DIR AFTER_DIR [--max-depth N]
"""

import argparse
import glob
import os
import re


def purpose_rows(path):
    """(indent, category, milliseconds) for each row of the TIMING BY PURPOSE table."""
    rows = []
    inside = False
    with open(path) as f:
        for line in f:
            if line.startswith("TIMING BY PURPOSE"):
                inside = True
                continue
            if inside and line.startswith("TIMING BY TIMER NESTING"):
                break
            m = re.match(r"^(\s+)(\S.*?)\s{2,}(?:([\d,]+)\s+)?([\d,]+\.\d\d)\s", line)
            if inside and m:
                rows.append((len(m.group(1)), m.group(2).strip(), float(m.group(4).replace(",", ""))))
    return rows


def keyed(rows):
    """Rows keyed by their path in the category tree, so equal names at different places differ."""
    out = {}
    stack = []
    for indent, name, ms in rows:
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, name))
        out["/".join(n for _, n in stack)] = (indent, ms)
    return out


def pytest_seconds(directory):
    """Pytest wall seconds per config, from <config>/times.txt."""
    seconds = {}
    for path in glob.glob(os.path.join(directory, "*", "times.txt")):
        with open(path) as f:
            m = re.search(r"seconds=(\d+)", f.read())
        if m:
            seconds[os.path.basename(os.path.dirname(path))] = int(m.group(1))
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="OUT directory of the 'before' pass")
    parser.add_argument("after", help="OUT directory of the 'after' pass")
    parser.add_argument("--max-depth", type=int, default=8, help="deepest category level printed")
    args = parser.parse_args()

    a = keyed(purpose_rows(os.path.join(args.before, "AGGREGATE_analysis.txt")))
    b = keyed(purpose_rows(os.path.join(args.after, "AGGREGATE_analysis.txt")))
    print(f"{'Category':70s} {'before s':>10s} {'after s':>10s} {'delta':>8s}")
    for key in list(a) + [k for k in b if k not in a]:
        indent = (a.get(key) or b.get(key))[0]
        if (indent - 4) // 2 > args.max_depth:
            continue
        x = a.get(key, (0, 0))[1] / 1000
        y = b.get(key, (0, 0))[1] / 1000
        if max(x, y) < 1:
            continue
        delta = f"{(y - x) / x * 100:+.1f}%" if x else "new"
        width = 70 - indent + 4
        print(f"{' ' * (indent - 4)}{key.split('/')[-1][:width]:{width}s} {x:10.1f} {y:10.1f} {delta:>8s}")

    tb = pytest_seconds(args.before)
    ta = pytest_seconds(args.after)
    total_b = sum(tb.values())
    total_a = sum(ta.values())
    change = f" ({(total_a - total_b) / total_b * 100:+.1f}%)" if total_b else ""
    print(f"\npytest wall sum: before {total_b} s, after {total_a} s{change}")
    print("largest per-config (before -> after s):")
    for k in sorted(tb, key=lambda k: -tb[k])[:12]:
        print(f"  {k:60s} {tb[k]:5d} -> {ta.get(k, 0):5d}")
    slower = sorted(((k, tb[k], ta[k]) for k in tb if k in ta and ta[k] > tb[k] + 2), key=lambda t: t[1] - t[2])
    print("slower by more than 2 s:", slower if slower else "none")


if __name__ == "__main__":
    main()
