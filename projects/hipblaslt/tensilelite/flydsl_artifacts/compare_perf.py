#!/usr/bin/env python3
"""Compare FlyDSL vs hipBLASLt Tensile tuning performance.

Usage:
    python compare_perf.py --flydsl perf_flydsl.txt --hipblaslt tuning_run/ [-o comparison.csv]
    python compare_perf.py --flydsl perf_flydsl.txt --hipblaslt-csv tuning_results.csv [-o comparison.csv]

Reads:
  --flydsl       : FlyDSL benchmark output (the text file produced by
                   test_gemm_a8w8_blockscale.py, containing the summary table)
  --hipblaslt    : A Tensile tuning output directory (each sub-dir has log.txt).
                   The best (minimum) time-us per problem size is used.
  --hipblaslt-csv: Alternatively, a tuning_results.csv from extract_tuning_perf.py

Outputs a CSV with columns:
    M, N, K, flydsl_us, hipblaslt_us, ratio, note
where ratio = hipblaslt_us / flydsl_us (>1 means hipBLASLt is slower).
"""
import argparse
import csv
import io
import os
import re
import sys
from collections import defaultdict


def parse_flydsl_perf(path):
    """Parse flydsl benchmark text output.

    Looks for lines in the fixed-width summary table with columns:
        dtype  m  n  k  ck_preshuffle  use_flydsl  init  apre  ck us  ...
    Returns dict: (M, N, K) -> ck_us (float).
    """
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Match lines like: torch.bfloat16 512  6144  7168 ...
            # or pipe-delimited markdown table rows
            # Try fixed-width format first
            m = re.match(
                r"(?:\|\s*)?torch\.\w+\s+"
                r"(\d+)\s+(\d+)\s+(\d+)\s+"  # m, n, k
                r"\S+\s+\S+\s+\S+\s+\S+\s+"  # ck_preshuffle, use_flydsl, init, apre
                r"([\d.]+)",                   # ck us
                line,
            )
            if m:
                M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
                ck_us = float(m.group(4))
                results[(M, N, K)] = ck_us
    return results


def parse_hipblaslt_logs(tuning_dir):
    """Parse Tensile tuning logs from a run directory.

    Each sub-directory has a log.txt with CSV result lines.
    Returns dict: (M, N, K) -> best_time_us (float).
    """
    results = defaultdict(lambda: float("inf"))
    for name in sorted(os.listdir(tuning_dir)):
        log_path = os.path.join(tuning_dir, name, "log.txt")
        if not os.path.isfile(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not re.match(r"^\d+,", line):
                    continue
                reader = csv.reader(io.StringIO(line))
                try:
                    fields = next(reader)
                except StopIteration:
                    continue
                if len(fields) < 11:
                    continue
                # problem-sizes: "(M,N,batch,K)"
                ps = fields[4].strip().strip('"()')
                parts = [int(x) for x in ps.split(",")]
                if len(parts) == 4:
                    M, N, batch, K = parts
                elif len(parts) == 3:
                    M, N, K = parts
                else:
                    continue
                time_us = float(fields[10])
                results[(M, N, K)] = min(results[(M, N, K)], time_us)
    return {k: v for k, v in results.items() if v < float("inf")}


def parse_hipblaslt_csv(csv_path):
    """Parse a tuning_results.csv from extract_tuning_perf.py.

    Returns dict: (M, N, K) -> best_time_us (float).
    """
    results = defaultdict(lambda: float("inf"))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ps = row["problem-sizes"].strip().strip('"()')
            parts = [int(x) for x in ps.split(",")]
            if len(parts) == 4:
                M, N, batch, K = parts
            elif len(parts) == 3:
                M, N, K = parts
            else:
                continue
            # Find all time-us columns
            for col, val in row.items():
                if col.startswith("time-us") and val:
                    time_us = float(val)
                    results[(M, N, K)] = min(results[(M, N, K)], time_us)
    return {k: v for k, v in results.items() if v < float("inf")}


def main():
    parser = argparse.ArgumentParser(
        description="Compare FlyDSL vs hipBLASLt tuning performance."
    )
    parser.add_argument(
        "--flydsl",
        required=True,
        help="Path to FlyDSL benchmark output text file (from test_gemm_a8w8_blockscale.py)",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--hipblaslt",
        help="Path to hipBLASLt Tensile tuning output directory (contains sub-dirs with log.txt)",
    )
    grp.add_argument(
        "--hipblaslt-csv",
        help="Path to tuning_results.csv from extract_tuning_perf.py",
    )
    parser.add_argument(
        "-o", "--output",
        default="flydsl_vs_hipblaslt.csv",
        help="Output CSV path (default: flydsl_vs_hipblaslt.csv)",
    )
    args = parser.parse_args()

    flydsl = parse_flydsl_perf(args.flydsl)
    if not flydsl:
        print(f"ERROR: no FlyDSL results parsed from {args.flydsl}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(flydsl)} FlyDSL results", file=sys.stderr)

    if args.hipblaslt:
        hipblaslt = parse_hipblaslt_logs(args.hipblaslt)
    else:
        hipblaslt = parse_hipblaslt_csv(args.hipblaslt_csv)
    if not hipblaslt:
        print("ERROR: no hipBLASLt results parsed", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(hipblaslt)} hipBLASLt results", file=sys.stderr)

    all_keys = sorted(set(flydsl) | set(hipblaslt))

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["M", "N", "K", "flydsl_us", "hipblaslt_us", "ratio", "note"])

    for key in all_keys:
        M, N, K = key
        f_us = flydsl.get(key)
        h_us = hipblaslt.get(key)
        if f_us is not None and h_us is not None:
            ratio = h_us / f_us
            if ratio > 1.005:
                pct = int(round((ratio - 1) * 100))
                note = f"hipBLASLt {pct}% slower"
            elif ratio < 0.995:
                pct = int(round((1 - ratio) * 100))
                note = f"hipBLASLt {pct}% faster"
            else:
                note = "~equal"
            writer.writerow([M, N, K, f"{f_us:.4f}", f"{h_us:.4f}", f"{ratio:.2f}", note])
        elif f_us is not None:
            writer.writerow([M, N, K, f"{f_us:.4f}", "", "", "hipBLASLt: no data"])
        else:
            writer.writerow([M, N, K, "", f"{h_us:.4f}", "", "FlyDSL: no data"])

    csv_text = out.getvalue()

    with open(args.output, "w") as f:
        f.write(csv_text)

    print(csv_text, end="")
    print(f"\nWritten to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
