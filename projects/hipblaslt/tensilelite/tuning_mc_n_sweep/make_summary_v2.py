#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Append tuned hipBLASLt results to the LWPARCH-565 rocKE / FlyDSL reference table.

Reads the winners straight out of the Tensile tuning logs: each "# Exact Logic:" block maps
``(M, N, batch, K, ldd, ldc, lda, ldb)`` to ``[solution index, GFlops]``, and the index points into
the "( N) <solution name> : <kernel name>" listing printed just above it. The GFlops figure
is the one that also lands in ``2_BenchmarkData/*_CSVWinner.csv``, so the logs alone suffice.

Writes ``bf16_mi450_mc_summary_v2.csv`` into the current working directory: the
reference CSV with five extra columns appended. "tuned" is in TFLOP/s to match the existing
rocKE / FlyDSL / hipBLASLt columns; "MT" is MacroTile0 x MacroTile1 x DepthU and "winner"
is the full kernel name.

Usage:
    python3 make_summary_v2.py [--reference CSV] [--log LOG ...] [-o NAME]
"""

import argparse
import ast
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_REFERENCE = os.path.join(REPO, "LWPARCH-565_GEMM_shapes_v2_FlyDSL_and_RockE.csv")
DEFAULT_LOGS = [
    os.path.join(HERE, "bf16_mi450_mc_cat0_qkv_output_gate_v2.log"),
    os.path.join(HERE, "bf16_mi450_mc_cat1_o_projection_v2.log"),
]
DEFAULT_OUTPUT = "bf16_mi450_mc_summary_v2.csv"

# Values the reference table uses for "not measured".
MISSING = {"", "-", "—", "N/A", "n/a"}

# "# Exact Logic:" is followed by a blank line and then the size -> [index, GFlops] table.
# The index points into the "( N) <solution name> : <kernel name>" listing printed just above it.
EXACT_LOGIC = re.compile(r"^#\s*Exact Logic:\s*$", re.MULTILINE)
LOGIC_TABLE = re.compile(r"^\{.*\}\s*$", re.MULTILINE)
SOLUTION_LINE = re.compile(r"^\(\s*(\d+)\)\s+(\S.*)$", re.MULTILINE)
MACRO_TILE = re.compile(r"_MT(\d+x\d+x\d+)_")


def parse_log(path):
    """Return {(M, N, K): (tflops, macro tile, kernel name)} for one tuning log.

    A log can hold more than one "# Exact Logic:" block (one per problem type), and the
    solution indices restart in each block, so each block is paired with the listing that
    precedes it rather than with a single file-wide listing.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    blocks = list(EXACT_LOGIC.finditer(text))
    if not blocks:
        raise ValueError(f"{path}: no '# Exact Logic:' block found")

    winners = {}
    start = 0
    for block in blocks:
        # The listing for this block sits between the end of the previous block and this one.
        listing = {int(idx): line.split(" : ")[-1].strip()
                   for idx, line in SOLUTION_LINE.findall(text[start:block.start()])}

        table_match = LOGIC_TABLE.search(text, block.end())
        if table_match is None:
            raise ValueError(f"{path}: '# Exact Logic:' block has no table")
        start = table_match.end()

        try:
            table = ast.literal_eval(table_match.group(0))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{path}: cannot parse the Exact Logic table: {exc}") from exc

        for key, value in table.items():
            # key: (SizeI=M, SizeJ=N, SizeK=batch, SizeL=K, ldd, ldc, lda, ldb)
            if len(key) < 4:
                raise ValueError(f"{path}: unexpected Exact Logic key {key!r}")
            size = (int(key[0]), int(key[1]), int(key[3]))
            kernel = listing.get(int(value[0]), "")
            tile = MACRO_TILE.search(kernel)
            # A later benchmark group wins over an earlier one for the same size.
            winners[size] = (float(value[1]) / 1000.0,
                             tile.group(1) if tile else "",
                             kernel)
    return winners


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
    parser.add_argument("--log", dest="logs", action="append", default=None,
                        help="tuning log to read winners from (repeatable)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="output file name, written to the current directory")
    args = parser.parse_args(argv)

    logs = args.logs if args.logs else DEFAULT_LOGS

    winners = {}
    for log in logs:
        if not os.path.exists(log):
            parser.error(f"log not found: {log}")
        found = parse_log(log)
        print(f"{os.path.basename(log)}: {len(found)} winners")
        winners.update(found)

    with open(args.reference, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        parser.error(f"empty reference table: {args.reference}")

    header, body = rows[0], rows[1:]
    columns = {name.strip(): i for i, name in enumerate(header)}
    for needed in ("M", "N", "K", "rocKE", "FlyDSL"):
        if needed not in columns:
            parser.error(f"{args.reference}: missing column {needed!r}")

    out_rows = [header + ["tuned", "vs rocKE", "vs FlyDSL", "MT", "winner"]]
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
            found = winners.get((int(m), int(n), int(k)))

        if found is None:
            unmatched.append(tuple(row[columns[c]] for c in ("GEMM", "M") if c in columns))
            out_rows.append(row + ["", "", "", "", ""])
            continue

        matched += 1
        tuned, tile, kernel = found
        cells = [f"{tuned:.2f}"]
        for name in ("rocKE", "FlyDSL"):
            ref = to_number(row[columns[name]])
            cells.append(f"{tuned / ref:.2f}" if ref else "")
        out_rows.append(row + cells + [tile, kernel])

    output = os.path.abspath(os.path.join(os.getcwd(), args.output))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(out_rows)

    print(f"matched {matched} of {len(out_rows) - 1} reference rows")
    if unmatched:
        print(f"no tuning result for: {', '.join('/'.join(u) for u in unmatched)}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
