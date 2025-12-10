#!/usr/bin/env python3
import itertools
import sys


def get_columns(line: str):
    col_names = list([p.strip() for p in line.split("|")])
    pipe_cols = [-1] + list([idx for idx, ch in enumerate(line) if ch == "|"]) + [None]
    assert len(col_names) + 1 == len(pipe_cols)

    rv = {}
    for i in range(len(col_names)):
        rv[col_names[i]] = (pipe_cols[i]+1, pipe_cols[i+1])

    return rv


def new_name(fname, part):
    parts = fname.split('.')
    return parts[0] + "_" + part + '.' + '.'.join(parts[1:])


def get_part(line, all_bounds, name):
    bounds = all_bounds[name]
    return line[bounds[0]:bounds[1]]


if __name__ == "__main__":
    fname = sys.argv[1]

    with open(fname) as f:
        first_line = next(f)
        cols = get_columns(first_line)

        print(cols)

        f2 = itertools.chain([first_line], f)

        base = sys
        out_files = {}
        for k in cols:
            out_files[k] = open(new_name(fname, k), "w")

        for line in f2:
            line = line.rstrip()
            for k, bounds in cols.items():
                if k != "Instruction":
                    part = get_part(line, cols, k) + "\n" + get_part(line, cols, "Instruction")
                    out_files[k].write(part + '\n')