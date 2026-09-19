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
import argparse
import csv
import io
import os
import re
import statistics
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
    parser = argparse.ArgumentParser(
        description="Extract latency and GFLOPS from Tensile benchmark logs."
    )
    parser.add_argument(
        "measurement_dir",
        help="Directory containing benchmark result folders (each with log.txt)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: <measurement_dir>/perf_summary.csv)",
    )
    parser.add_argument(
        "-s", "--sources", nargs="+", default=None,
        help="Only include these source folders (default: all)",
    )
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.measurement_dir)
    if not os.path.isdir(base_dir):
        parser.error(f"Not a directory: {base_dir}")

    # Discover source folders (e.g. bkc_26.9.5) -- any subdirectory that
    # itself contains subdirectories with log.txt files.
    source_folders = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d != "configs" and not d.startswith(".")
    )

    # Filter to requested sources if specified
    if args.sources:
        requested = set(args.sources)
        source_folders = [s for s in source_folders if s in requested]

    # Structure:  data[source_folder][(gemm_type, problem_sizes)] = [(time_us, gflops), ...]
    data = defaultdict(lambda: defaultdict(list))
    all_keys = set()  # (gemm_type, problem_sizes)

    for src in source_folders:
        src_path = os.path.join(base_dir, src)
        for gemm_dir in sorted(os.listdir(src_path)):
            gemm_path = os.path.join(src_path, gemm_dir)
            if not os.path.isdir(gemm_path):
                continue
            gemm_type = gemm_type_from_dir(gemm_dir)

            # Collect log files: flat layout (<config>/log.txt) or
            # multi-run layout (<config>/run_N/log.txt)
            log_files = []
            direct_log = os.path.join(gemm_path, "log.txt")
            if os.path.isfile(direct_log):
                log_files.append(direct_log)
            else:
                for sub in sorted(os.listdir(gemm_path)):
                    sub_log = os.path.join(gemm_path, sub, "log.txt")
                    if os.path.isfile(sub_log):
                        log_files.append(sub_log)

            for log_path in log_files:
                for problem_sizes, _, time_us, gflops in parse_log(log_path):
                    key = (gemm_type, problem_sizes)
                    all_keys.add(key)
                    data[src][(gemm_type, problem_sizes)].append((time_us, gflops))

    # Drop source folders that produced no data
    source_folders = [s for s in source_folders if data[s]]

    if not all_keys:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    # Sort keys: by gemm_type then problem_sizes
    all_keys = sorted(all_keys)

    def _stats(values):
        """Return (avg, min, max, median) for a list of floats, or empty strings."""
        if not values:
            return ("", "", "", "")
        avg = statistics.mean(values)
        mn = min(values)
        mx = max(values)
        med = statistics.median(values)
        return (f"{avg:.3f}", f"{mn:.3f}", f"{mx:.3f}", f"{med:.3f}")

    # Determine the max number of runs across all source/key combos
    max_runs = 0
    for src in source_folders:
        for entries in data[src].values():
            max_runs = max(max_runs, len(entries))

    # Build CSV
    out = io.StringIO()
    writer = csv.writer(out)

    # Header: gflops section first, then latency section
    header = ["gemm_type", "problem_sizes"]
    for src in source_folders:
        for r in range(1, max_runs + 1):
            header.append(f"{src}_run{r}_gflops")
        header.extend([
            f"{src}_avg_gflops", f"{src}_min_gflops",
            f"{src}_max_gflops", f"{src}_median_gflops",
        ])
        for r in range(1, max_runs + 1):
            header.append(f"{src}_run{r}_time_us")
        header.extend([
            f"{src}_avg_time_us", f"{src}_min_time_us",
            f"{src}_max_time_us", f"{src}_median_time_us",
        ])
    writer.writerow(header)

    # Rows
    for gemm_type, problem_sizes in all_keys:
        row = [gemm_type, problem_sizes]
        for src in source_folders:
            entries = data[src].get((gemm_type, problem_sizes), [])
            times = [float(e[0]) for e in entries if e[0]]
            flops = [float(e[1]) for e in entries if e[1]]
            t_avg, t_min, t_max, t_med = _stats(times)
            g_avg, g_min, g_max, g_med = _stats(flops)
            # GFLOPS section: per-run then stats
            for r in range(max_runs):
                row.append(entries[r][1] if r < len(entries) else "")
            row.extend([g_avg, g_min, g_max, g_med])
            # Latency section: per-run then stats
            for r in range(max_runs):
                row.append(entries[r][0] if r < len(entries) else "")
            row.extend([t_avg, t_min, t_max, t_med])
        writer.writerow(row)

    csv_text = out.getvalue()

    # Write to file
    out_path = args.output or os.path.join(base_dir, "perf_summary.csv")
    with open(out_path, "w") as f:
        f.write(csv_text)

    # Also print to stdout
    print(csv_text, end="")
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
