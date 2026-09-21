#!/usr/bin/env python3
"""Extract final tuning results (time-us, kernel) from Tensile tuning log folders.

Usage:
    python extract_tuning_results.py <tuning_dir1> <tuning_dir2> [tuning_dir3 ...] [-o output.csv]

Each tuning_dir should contain subdirectories with log.txt files whose last CSV
data line holds the benchmark result.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def parse_result_line(line):
    """Parse the final CSV result line from a Tensile log.

    The line is CSV but problem-sizes is a quoted field with commas inside,
    e.g. "(512,2048,1,7168)". We handle that with csv.reader.
    """
    # The header columns (for reference):
    # run,problem-progress,solution-progress,operation,problem-sizes,
    # bias-type,factor-dim,activation-type,solution,validation,time-us,...
    reader = csv.reader([line])
    fields = next(reader)
    if len(fields) < 11:
        return None
    return {
        "problem-sizes": fields[4],
        "solution": fields[8],
        "time-us": fields[10],
    }


def extract_from_log(log_path):
    """Read a log file and return the parsed last result line, or None."""
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Walk backwards to find the last line that starts with "0," (the result)
    for line in reversed(lines):
        stripped = line.strip()
        if re.match(r"^\d+,", stripped):
            return parse_result_line(stripped)
    return None


def gather_results(tuning_dirs):
    """Iterate over tuning directories and collect results."""
    rows = []
    for tdir in tuning_dirs:
        tdir = Path(tdir)
        run_name = tdir.name
        # Find all subdirectories that contain a log.txt
        subdirs = sorted(
            [d for d in tdir.iterdir() if d.is_dir() and (d / "log.txt").exists()]
        )
        if not subdirs:
            print(f"Warning: no log.txt found in subdirs of {tdir}", file=sys.stderr)
            continue
        for subdir in subdirs:
            log_path = subdir / "log.txt"
            result = extract_from_log(log_path)
            if result is None:
                print(f"Warning: no result line in {log_path}", file=sys.stderr)
                continue
            rows.append(
                {
                    "run": run_name,
                    "config": subdir.name,
                    "problem-sizes": result["problem-sizes"],
                    "time-us": result["time-us"],
                    "solution": result["solution"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract final tuning latency and kernel from Tensile log folders."
    )
    parser.add_argument(
        "tuning_dirs",
        nargs="+",
        help="Paths to tuning run directories (each containing subdirs with log.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tuning_results.csv",
        help="Output CSV path (default: tuning_results.csv)",
    )
    args = parser.parse_args()

    # Validate inputs
    for d in args.tuning_dirs:
        if not os.path.isdir(d):
            parser.error(f"Not a directory: {d}")

    rows = gather_results(args.tuning_dirs)
    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Pivot: rows = problem-sizes, columns = latency + kernel per run folder
    run_names = []
    seen = set()
    for d in args.tuning_dirs:
        name = Path(d).name
        if name not in seen:
            run_names.append(name)
            seen.add(name)

    # Build lookup: (run, config) -> result row
    lookup = {}
    for r in rows:
        lookup[(r["run"], r["config"])] = r

    # Collect all unique problem-sizes preserving config order
    size_rows = []
    seen_sizes = set()
    for r in rows:
        key = (r["config"], r["problem-sizes"])
        if key not in seen_sizes:
            size_rows.append({"config": r["config"], "problem-sizes": r["problem-sizes"]})
            seen_sizes.add(key)

    # Build header
    fieldnames = ["config", "problem-sizes"]
    for name in run_names:
        fieldnames.append(f"time-us_{name}")
        fieldnames.append(f"solution_{name}")

    # Build pivoted rows
    out_rows = []
    for sr in size_rows:
        out = {"config": sr["config"], "problem-sizes": sr["problem-sizes"]}
        for name in run_names:
            entry = lookup.get((name, sr["config"]))
            out[f"time-us_{name}"] = entry["time-us"] if entry else ""
            out[f"solution_{name}"] = entry["solution"] if entry else ""
        out_rows.append(out)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
