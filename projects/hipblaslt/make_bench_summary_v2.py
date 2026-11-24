#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Append hipblaslt-bench results to the LWPARCH-565 rocKE / FlyDSL reference table.

Reads a ``hipblaslt-bench --yaml`` log. For every problem the bench prints a ``[0]:``
header naming its CSV columns, then one indented row of values, then ``--Solution index``,
``--Solution name`` and ``--kernel name``. Columns are read from the header rather than by
position, so a bench that gains or loses a column keeps working.

Writes ``bf16_mi450_mc_bench_summary_v2.csv`` into the current working directory: the
reference CSV with the bench columns appended. "bench" is in TFLOP/s to match the existing
rocKE / FlyDSL / hipBLASLt columns and reports what the rebuilt library actually selects,
"MT" is MacroTile0 x MacroTile1 x DepthU and "bench winner" is the kernel name it reported.

A second log can be passed with ``--log-b`` to compare two builds or two runtime settings;
three more columns are then added: "bench b", the ratio "a/b", and "MT b".

Usage:
    python3 make_bench_summary_v2.py [--reference CSV] [--log LOG] [--log-b LOG] [-o NAME]
"""

import argparse
import csv
import os
import re
import sys

# This script lives next to the logs it reads.
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOG = os.path.join(HERE, "bbs_tn_mk_pairs_n_sweep_v2.log")
DEFAULT_OUTPUT = "bf16_mi450_mc_bench_summary_v2.csv"

# The reference table is shared with the tuning work, which lives in the other worktree.
# Prefer a local copy if one is made, otherwise fall back to where it is kept today.
REFERENCE_NAME = "LWPARCH-565_GEMM_shapes_v2_FlyDSL_and_RockE.csv"
REFERENCE_CANDIDATES = [os.path.join(HERE, REFERENCE_NAME)]
DEFAULT_REFERENCE = next((p for p in REFERENCE_CANDIDATES if os.path.exists(p)),
                         REFERENCE_CANDIDATES[0])

# Values the reference table uses for "not measured".
MISSING = {"", "-", "—", "N/A", "n/a"}

HEADER_LINE = re.compile(r"^\[\d+\]:(.*)$")
KERNEL_LINE = re.compile(r"^\s*--kernel name:\s*(\S+)\s*$")
MACRO_TILE = re.compile(r"_MT(\d+x\d+x\d+)_")


def parse_log(path):
    """Return {(M, N, K): (tflops, macro tile, kernel name)} for one bench log."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    results = {}
    i = 0
    while i < len(lines):
        header = HEADER_LINE.match(lines[i])
        if not header:
            i += 1
            continue

        columns = [name.strip() for name in header.group(1).split(",")]
        index = {name: pos for pos, name in enumerate(columns)}
        for needed in ("m", "n", "k", "hipblaslt-Gflops"):
            if needed not in index:
                raise ValueError(f"{path}: line {i + 1}: bench header has no {needed!r} column")

        # The row of values is the next non-empty line.
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        values = [cell.strip() for cell in lines[i].split(",")]
        if len(values) < len(columns):
            # A problem the bench skipped prints no row; fall through to the next header.
            continue

        size = tuple(int(values[index[d]]) for d in ("m", "n", "k"))
        gflops = float(values[index["hipblaslt-Gflops"]])

        # The kernel name follows, before the next header.
        kernel = ""
        j = i + 1
        while j < len(lines) and not HEADER_LINE.match(lines[j]):
            found = KERNEL_LINE.match(lines[j])
            if found:
                kernel = found.group(1)
                break
            j += 1

        tile = MACRO_TILE.search(kernel)
        # A repeated size keeps the last measurement in the log.
        results[size] = (gflops / 1000.0, tile.group(1) if tile else "", kernel)
        i += 1
    return results


def to_number(text):
    """Parse a reference-table cell, tolerating thousands separators and em dashes."""
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if cleaned in MISSING:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="reference CSV with the rocKE / FlyDSL columns")
    parser.add_argument("--log", dest="log_a", default=DEFAULT_LOG,
                        help="hipblaslt-bench log to report")
    parser.add_argument("--log-b", dest="log_b", default=None,
                        help="second bench log to compare against; adds the comparison columns")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="output file name, written to the current directory")
    args = parser.parse_args(argv)

    if not os.path.exists(args.log_a):
        parser.error(f"log not found: {args.log_a}")
    measured = parse_log(args.log_a)
    print(f"{os.path.basename(args.log_a)}: {len(measured)} problems")

    baseline = {}
    if args.log_b:
        if not os.path.exists(args.log_b):
            parser.error(f"log not found: {args.log_b}")
        baseline = parse_log(args.log_b)
        print(f"{os.path.basename(args.log_b)}: {len(baseline)} problems  (comparison)")

    with open(args.reference, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        parser.error(f"empty reference table: {args.reference}")

    header, body = rows[0], rows[1:]
    columns = {name.strip(): i for i, name in enumerate(header)}
    for needed in ("M", "N", "K", "rocKE", "FlyDSL"):
        if needed not in columns:
            parser.error(f"{args.reference}: missing column {needed!r}")

    extra = ["bench", "vs rocKE", "vs FlyDSL", "MT", "bench winner"]
    if baseline:
        extra += ["bench b", "a/b", "MT b"]
    out_rows = [header + extra]
    matched = 0
    unmatched = []
    for row in body:
        if not any(cell.strip() for cell in row):
            continue
        m = to_number(row[columns["M"]])
        n = to_number(row[columns["N"]])
        k = to_number(row[columns["K"]])
        found = None
        if None not in (m, n, k):
            found = measured.get((int(m), int(n), int(k)))

        size = None if None in (m, n, k) else (int(m), int(n), int(k))
        off = baseline.get(size) if baseline else None

        if found is None:
            unmatched.append("/".join(row[columns[c]] for c in ("GEMM", "M") if c in columns))
            out_rows.append(row + [""] * len(extra))
            continue

        matched += 1
        bench, tile, kernel = found
        cells = [f"{bench:.2f}"]
        for name in ("rocKE", "FlyDSL"):
            ref = to_number(row[columns[name]])
            cells.append(f"{bench / ref:.2f}" if ref else "")
        cells += [tile, kernel]
        if baseline:
            cells += ([f"{off[0]:.2f}", f"{bench / off[0]:.3f}" if off[0] else "", off[1]]
                      if off else ["", "", ""])
        out_rows.append(row + cells)

    output = os.path.abspath(os.path.join(os.getcwd(), args.output))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(out_rows)

    print(f"matched {matched} of {len(out_rows) - 1} reference rows")
    if unmatched:
        print(f"no bench result for: {', '.join(unmatched)}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
