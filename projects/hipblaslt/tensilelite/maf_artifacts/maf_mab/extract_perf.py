#!/usr/bin/env python3
"""Extract latency and GFLOPS from Tensile benchmark logs.

Usage:
    python extract_perf.py <measurement_dir>
    python extract_perf.py ~/maf_mab_measurment_Sept14

Scans every sub-directory of <measurement_dir> that contains benchmark result
folders (each with a log.txt).  Produces a single CSV on stdout with rows =
(gemm_type, problem_size) and one pair of columns (time_us, gflops) per source
folder (e.g. bkc_26.9.5).

The CSV is also written to <measurement_dir>/perf_summary.csv.
"""
import csv
import io
import os
import re
import sys
from collections import defaultdict


def parse_log(log_path):
    """Yield (problem_sizes, solution_short, time_us, gflops) from a log."""
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("run,"):
                continue
            # Data lines start with "0," (run index)
            if not re.match(r"^\d+,", line):
                continue
            # The problem-sizes field is quoted and contains commas, e.g.
            #   "(4096,4096,1,65536)"
            # Use csv reader to handle that correctly.
            reader = csv.reader(io.StringIO(line))
            try:
                fields = next(reader)
            except StopIteration:
                continue
            if len(fields) < 12:
                continue
            problem_sizes = fields[4]       # "(M,N,batch,K)"
            solution_name = fields[8]       # full kernel name
            time_us = fields[10]            # latency in microseconds
            gflops = fields[11]             # reported GFLOPS
            yield problem_sizes, solution_name, time_us, gflops


def gemm_type_from_dir(dirname):
    """Return a human-friendly GEMM type label from the directory name."""
    return dirname  # e.g. bbs_tn_maf, f8_tn_maf, mxf4_tn_maf, ...


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <measurement_dir>", file=sys.stderr)
        sys.exit(1)

    base_dir = os.path.expanduser(sys.argv[1])
    if not os.path.isdir(base_dir):
        print(f"ERROR: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Discover source folders (e.g. bkc_26.9.5) -- any subdirectory that
    # itself contains subdirectories with log.txt files.
    source_folders = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d != "configs"
    )

    # Structure:  data[source_folder][(gemm_type, problem_sizes)] = [(time_us, gflops), ...]
    data = defaultdict(lambda: defaultdict(list))
    all_keys = set()  # (gemm_type, problem_sizes)

    for src in source_folders:
        src_path = os.path.join(base_dir, src)
        for gemm_dir in sorted(os.listdir(src_path)):
            log_path = os.path.join(src_path, gemm_dir, "log.txt")
            if not os.path.isfile(log_path):
                continue
            gemm_type = gemm_type_from_dir(gemm_dir)
            for problem_sizes, _, time_us, gflops in parse_log(log_path):
                key = (gemm_type, problem_sizes)
                all_keys.add(key)
                data[src][(gemm_type, problem_sizes)].append((time_us, gflops))

    if not all_keys:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    # Sort keys: by gemm_type then problem_sizes
    all_keys = sorted(all_keys)

    # Build CSV
    out = io.StringIO()
    writer = csv.writer(out)

    # Header
    header = ["gemm_type", "problem_sizes"]
    for src in source_folders:
        header.append(f"{src}_time_us")
        header.append(f"{src}_gflops")
    writer.writerow(header)

    # Rows
    for gemm_type, problem_sizes in all_keys:
        row = [gemm_type, problem_sizes]
        for src in source_folders:
            entries = data[src].get((gemm_type, problem_sizes), [])
            if entries:
                # If multiple results for same key, report all (semicolon-separated)
                times = ";".join(e[0] for e in entries)
                flops = ";".join(e[1] for e in entries)
            else:
                times = ""
                flops = ""
            row.append(times)
            row.append(flops)
        writer.writerow(row)

    csv_text = out.getvalue()

    # Write to file
    out_path = os.path.join(base_dir, "perf_summary.csv")
    with open(out_path, "w") as f:
        f.write(csv_text)

    # Also print to stdout
    print(csv_text, end="")
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
