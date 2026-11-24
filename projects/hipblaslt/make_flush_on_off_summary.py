#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two hipblaslt-bench runs that differ only in --flush, size by size.

Reads two ``hipblaslt-bench --yaml`` logs and writes a CSV with one row per problem
size: M, N, K, MT, the winning time with icache flush on, the same with it off, and
the difference (off - on). Times are microseconds, taken from the bench's own ``us``
column.

Under ``algo_method: 1`` the bench benchmarks every solution for a size rather than
just the one the heuristic picks, so a log holds a few hundred blocks per size. Each
block is ``[n]:<header>``, a row of values, then the solution and kernel names; a
``[0]:`` block starts the next size. The winner for a size is the block with the
lowest ``us``, so the comparison is best-achievable against best-achievable.

Which log is which is decided by the ``flush`` column the bench recorded, not by the
file name - the two have been swapped before. Pass --trust-names to override.

Usage:
    python3 make_flush_on_off_summary.py [--log-a LOG] [--log-b LOG] [-o NAME]
"""

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_A = os.path.join(HERE, "bbs_tn_mk_pairs_n_sweep_flush_on.log")
DEFAULT_B = os.path.join(HERE, "bbs_tn_mk_pairs_n_sweep_flush_off.log")
DEFAULT_OUTPUT = "bbs_tn_mk_pairs_n_sweep_bench_mc_flush_on_off.csv"

HEADER = re.compile(r"^\[\d+\]:(.*)$")
KERNEL = re.compile(r"^\s*--kernel name:\s*(\S+)\s*$")
MACRO_TILE = re.compile(r"_MT(\d+x\d+x\d+)_")


def parse_log(path, verbose=True):
    """Return ({(M, N, K): (us, macro tile, kernel)}, flush value, blocks seen).

    Only the fastest block per size is kept. The file is streamed rather than read
    whole: these logs run to hundreds of megabytes once every solution is timed.
    """
    best = {}
    flush_values = set()
    blocks = 0
    columns = None
    want_row = False
    pending = None

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            header = HEADER.match(line)
            if header:
                names = [n.strip() for n in header.group(1).split(",")]
                columns = {name: i for i, name in enumerate(names)}
                for needed in ("m", "n", "k", "us"):
                    if needed not in columns:
                        raise SystemExit(f"{path}: bench header has no {needed!r} column")
                want_row = True
                pending = None
                continue

            if want_row and line.strip():
                values = [c.strip() for c in line.split(",")]
                want_row = False
                if len(values) < len(columns):
                    continue  # a size the bench skipped prints no row
                try:
                    size = tuple(int(values[columns[d]]) for d in ("m", "n", "k"))
                    us = float(values[columns["us"]])
                except ValueError:
                    continue
                if "flush" in columns:
                    flush_values.add(values[columns["flush"]])
                if us > 0:
                    pending = (size, us)
                    blocks += 1
                continue

            if pending:
                found = KERNEL.match(line)
                if found:
                    size, us = pending
                    pending = None
                    if size not in best or us < best[size][0]:
                        tile = MACRO_TILE.search(found.group(1))
                        best[size] = (us, tile.group(1) if tile else "", found.group(1))

    if len(flush_values) > 1:
        raise SystemExit(f"{path}: mixed flush settings in one log: {sorted(flush_values)}")
    flush = flush_values.pop() if flush_values else None

    if verbose:
        print(f"{os.path.basename(path)}: {blocks} timed blocks, "
              f"{len(best)} sizes, flush={flush}")
    return best, flush


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-a", default=DEFAULT_A, help="first bench log")
    parser.add_argument("--log-b", default=DEFAULT_B, help="second bench log")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="output file name, written to the current directory")
    parser.add_argument("--trust-names", action="store_true",
                        help="take --log-a as flush on and --log-b as flush off instead of "
                             "reading the flush column")
    args = parser.parse_args(argv)

    for path in (args.log_a, args.log_b):
        if not os.path.exists(path):
            parser.error(f"log not found: {path}")

    a, flush_a = parse_log(args.log_a)
    b, flush_b = parse_log(args.log_b)

    # "1" is flush on: the bench's --flush turns icache flushing on.
    if args.trust_names or flush_a is None or flush_b is None or flush_a == flush_b:
        if not args.trust_names:
            print("warning: cannot tell the runs apart from the flush column "
                  f"(a={flush_a}, b={flush_b}); falling back to argument order")
        (on, on_path), (off, off_path) = (a, args.log_a), (b, args.log_b)
    else:
        (on, on_path), (off, off_path) = (((a, args.log_a), (b, args.log_b))
                                          if flush_a == "1" else
                                          ((b, args.log_b), (a, args.log_a)))
    print(f"flush on  <- {os.path.basename(on_path)}")
    print(f"flush off <- {os.path.basename(off_path)}")

    shared = sorted(set(on) & set(off))
    if not shared:
        parser.error("the two logs share no problem size")

    rows = [["M", "N", "K", "MT", "flush on", "flush off", "change (off-on)"]]
    differing = []
    for size in shared:
        us_on, mt_on, _ = on[size]
        us_off, mt_off, _ = off[size]
        if mt_on != mt_off:
            differing.append((size, mt_on, mt_off))
        rows.append([size[0], size[1], size[2], mt_on,
                     f"{us_on:.2f}", f"{us_off:.2f}", f"{us_off - us_on:+.2f}"])

    output = os.path.abspath(os.path.join(os.getcwd(), args.output))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)

    print(f"\nwrote {output}  ({len(rows) - 1} sizes)")
    only = set(on) ^ set(off)
    if only:
        print(f"skipped {len(only)} size(s) present in only one log")
    if differing:
        # The two columns then describe different kernels, so the delta is not a
        # like-for-like measurement of what flushing costs.
        print(f"note: {len(differing)} size(s) have a different winning tile in the two "
              f"runs; the MT column shows the flush-on winner, e.g.")
        for size, mt_on, mt_off in differing[:3]:
            print(f"    M={size[0]} N={size[1]} K={size[2]}: on {mt_on} vs off {mt_off}")

    deltas = [float(r[6]) for r in rows[1:]]
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    print(f"change (off-on): mean {mean:+.2f} us, median {deltas[len(deltas) // 2]:+.2f} us, "
          f"range {deltas[0]:+.2f} .. {deltas[-1]:+.2f}")
    print(f"off slower than on in {sum(1 for d in deltas if d > 0)} of {len(deltas)} sizes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
