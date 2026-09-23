#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Split a liveness file into separate files for each kind of register.

The individual output files alternate between the liveness state and the
instruction. This is much easier to view, especially with word wrapping
disabled.

It's recommended to view the output files with line wrapping OFF.
"""

import argparse
import collections
import itertools
from pathlib import Path


def get_columns(line: str) -> dict[str, tuple[int, int]]:
    col_names = list([p.strip() for p in line.split("|")])
    pipe_cols = [-1] + list([idx for idx, ch in enumerate(line) if ch == "|"]) + [None]
    assert len(col_names) + 1 == len(pipe_cols)

    rv = {}
    for i in range(len(col_names)):
        rv[col_names[i]] = (pipe_cols[i] + 1, pipe_cols[i + 1])

    return rv


def new_name(fname, part):
    parts = fname.split(".")
    return parts[0] + "_" + part + "." + ".".join(parts[1:])


def get_part(line, all_bounds, name):
    bounds = all_bounds[name]
    return line[bounds[0] : bounds[1]]


def split_lines(lines: [str]) -> dict[str, [str]]:
    rv = collections.defaultdict(list)

    first_line = next(lines)
    cols = get_columns(first_line)
    f2 = itertools.chain([first_line], lines)

    for line in f2:
        line = line.rstrip()
        for k, bounds in cols.items():
            if k != "Instruction":
                rv[k].append(get_part(line, cols, k))
                rv[k].append(get_part(line, cols, "Instruction"))

    return {k: v for k, v in rv.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fname", help="The liveness file to split.")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="The directory to write the split files to.",
        default=".",
        type=Path,
    )
    args = parser.parse_args()

    with open(args.fname) as f:
        splits = split_lines(f)

    for k, v in splits.items():
        new_file = args.output_dir / new_name(args.fname, k)
        with open(new_file, "w") as f:
            for line in v:
                f.write(line + "\n")
        print(f"Wrote {new_file}")
