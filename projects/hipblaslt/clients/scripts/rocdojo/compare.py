#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare hipblaslt-bench --yaml runs of the same problem list, one log per library.

    compare.py stock.log merged.log [rocdojo.log ...] [--csv out.csv]

Each log is the stdout of one `hipblaslt-bench --yaml <list>` run with
print_kernel_info. Problems are matched by (transA, transB, batch, m, n, k) and
order. Prints, for every log after the first, the geometric mean of its TFLOPS
over the first log's, wins/losses beyond 2%, and how many of its picks are
rocDojo kernels, per layout and overall.
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict

_HEADER = re.compile(r"^\[\d+\]:(.*)$")
_KERNEL = re.compile(r"--kernel name:\s*(\S+)")
_NO_SOLUTION = "NO solution"


def parse(path):
    """[(key, tflops or None, us, kernel)] in problem order."""
    rows = []
    lines = open(path, errors="replace").read().splitlines()
    i = 0
    winner = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("Winner:"):
            winner = True  # the next header reprints the fastest of the problem above
        if _NO_SOLUTION in line:
            rows.append((None, None, None, "NO_SOLUTION"))
        m = _HEADER.match(line.strip())
        if m and i + 1 < len(lines):
            cols = m.group(1).split(",")
            vals = [v.strip() for v in lines[i + 1].split(",")]
            if len(vals) == len(cols):
                rec = dict(zip(cols, vals))
                key = tuple(rec[c] for c in ("transA", "transB", "batch_count", "m", "n", "k"))
                kernel = ""
                for j in range(i + 2, min(i + 8, len(lines))):
                    km = _KERNEL.search(lines[j])
                    if km:
                        kernel = km.group(1)
                        break
                gflops = float(rec["hipblaslt-Gflops"])
                if winner and rows:
                    rows[-1] = (key, gflops / 1000, float(rec["us"]), kernel)
                else:
                    rows.append((key, gflops / 1000, float(rec["us"]), kernel))
                winner = False
                i += 2
                continue
        i += 1
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--csv", help="write one row per problem")
    ap.add_argument("--threshold", type=float, default=0.02, help="win/loss margin (default 0.02)")
    args = ap.parse_args()
    runs = [parse(p) for p in args.logs]
    n = min(len(r) for r in runs)
    if any(len(r) != n for r in runs):
        print(
            f"warning: logs hold {[len(r) for r in runs]} problems; comparing the first {n}",
            file=sys.stderr,
        )
    base = runs[0]
    for r in runs[1:]:
        for a, b in zip(base[:n], r[:n]):
            if a[0] and b[0] and a[0] != b[0]:
                sys.exit(f"problem order differs: {a[0]} vs {b[0]}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["transA", "transB", "batch", "m", "n", "k"]
                + [f"{col}_{i}" for i in range(len(runs)) for col in ("tflops", "kernel")]
            )
            for p in range(n):
                key = next((r[p][0] for r in runs if r[p][0]), ("",) * 6)
                w.writerow([*key] + [x for r in runs for x in (r[p][1], r[p][3])])
    for idx, r in enumerate(runs[1:], 1):
        groups = defaultdict(
            lambda: {"logs": [], "win": 0, "loss": 0, "rocdojo": 0, "n": 0, "missing": 0}
        )
        for p in range(n):
            a, b = base[p], r[p]
            key = a[0] or b[0]
            layout = f"{key[0]}{key[1]}" + ("" if key[2] == "1" else "-batched") if key else "?"
            for g in (groups[layout], groups["all"]):
                g["n"] += 1
                if b[3].startswith("rocdojo_"):
                    g["rocdojo"] += 1
                if not a[1] or not b[1]:
                    g["missing"] += 1
                    continue
                ratio = b[1] / a[1]
                g["logs"].append(math.log(ratio))
                g["win"] += ratio > 1 + args.threshold
                g["loss"] += ratio < 1 - args.threshold
        print(f"{args.logs[idx]} vs {args.logs[0]}")
        print(
            f"  {'layout':<12} {'n':>5} {'geomean':>8} {'win':>5} {'loss':>5} {'rocdojo':>8} {'missing':>8}"
        )
        for name in sorted(groups, key=lambda s: (s == "all", s)):
            g = groups[name]
            gm = math.exp(sum(g["logs"]) / len(g["logs"])) if g["logs"] else float("nan")
            print(
                f"  {name:<12} {g['n']:>5} {gm:>8.3f} {g['win']:>5} {g['loss']:>5}"
                f" {g['rocdojo']:>8} {g['missing']:>8}"
            )


if __name__ == "__main__":
    main()
