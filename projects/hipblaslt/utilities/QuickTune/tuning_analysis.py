# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import re
import csv
from collections import Counter

def parse_mnksol(line):
    """parsing -m, -n, -k, --solution_index"""
    m = re.search(r"-m\s+(\d+)", line)
    n = re.search(r"-n\s+(\d+)", line)
    k = re.search(r"-k\s+(\d+)", line)
    if m and n and k:
        return (int(m.group(1)), int(n.group(1)), int(k.group(1)))
    return None

def load_tuning_results(csv_file):
    """loading tuned_solution_idx info"""
    mapping = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = int(row["m"])
                n = int(row["n"])
                k = int(row["k"])
            except Exception:
                continue

            mapping[(m, n, k)] = {
                "baseline_latency(us)": row["baseline_latency(us)"],
                "baseline_solution_idx": row["baseline_solution_idx"],
                "tuned_latency(us)": row["tuned_latency(us)"],
                "tuned_solution_idx": row["tuned_solution_idx"],
                "baseline/tuned": row["baseline/tuned"],
            }
    return mapping

def count_mnksol(input_file, tuning_csv, output_csv):
    combos = []
    with open(input_file, "r") as f:
        for line in f:
            combo = parse_mnksol(line)
            if combo:
                combos.append(combo)

    total_combos = len(combos)
    counter = Counter(combos)

    tuning_map = load_tuning_results(tuning_csv)

    # compute total latency for each kernel
    total_baseline_all = 0.0
    total_tuned_all = 0.0
    per_kernel_times = {}

    for combo, cnt in counter.items():
        tuning_info = tuning_map.get(combo, None)
        if tuning_info:
            baseline_latency = float(tuning_info["baseline_latency(us)"])
            tuned_latency = float(tuning_info["tuned_latency(us)"])
            # Skip failed benchmarks - don't count in aggregation
            if baseline_latency < 0 or tuned_latency < 0:
                per_kernel_times[combo] = (0.0, 0.0)
                continue
        else:
            # Skip missing data - don't count in aggregation
            per_kernel_times[combo] = (0.0, 0.0)
            continue
        total_baseline = baseline_latency * cnt
        total_tuned = tuned_latency * cnt
        per_kernel_times[combo] = (total_baseline, total_tuned)
        total_baseline_all += total_baseline
        total_tuned_all += total_tuned

    # sort by total baseline latency 
    sorted_combos = sorted(counter.keys(), key=lambda c: per_kernel_times[c][0], reverse=True)

    # print statistic info
    header = f"{'(-m, -n, -k)':>30}  {'count':>6}  {'baseline/tuned':>12}  " \
             f"{'total_baseline(us)':>18}  {'total_tuned(us)':>16}  " \
             f"{'baseline%':>10}  {'tuned%':>8}"
    print(f"Total -m -n -k combos: {total_combos}\n")
    print(header)
    print("-" * len(header))

    fieldnames = [
        "m", "n", "k", "count",
        "baseline_latency(us)", "baseline_solution_idx",
        "tuned_latency(us)", "tuned_solution_idx", "baseline/tuned",
        "total_baseline_time(us)", "total_tuned_time(us)",
        "baseline_time_percent", "tuned_time_percent"
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for combo in sorted_combos:
            cnt = counter[combo]
            m, n, k = combo
            tuning_info = tuning_map.get(combo, None)
            baseline_tuned = tuning_info.get("baseline/tuned") if tuning_info else "no_match"
            total_baseline, total_tuned = per_kernel_times[combo]
            baseline_percent = total_baseline / total_baseline_all * 100 if total_baseline_all else 0
            tuned_percent = total_tuned / total_tuned_all * 100 if total_tuned_all else 0

            print(f"{str(combo[:3]):>30}  {cnt:6d}  {baseline_tuned:>12}  "
                  f"{total_baseline:18.3f}  {total_tuned:16.3f}  "
                  f"{baseline_percent:10.2f}%  {tuned_percent:8.2f}%")

            row = {
                "m": m, "n": n, "k": k, "count": cnt,
                "baseline_latency(us)": tuning_info.get("baseline_latency(us)") if tuning_info else "no_match",
                "baseline_solution_idx": tuning_info.get("baseline_solution_idx") if tuning_info else "no_match",
                "tuned_latency(us)": tuning_info.get("tuned_latency(us)") if tuning_info else "no_match",
                "tuned_solution_idx": tuning_info.get("tuned_solution_idx") if tuning_info else "no_match",
                "baseline/tuned": baseline_tuned,
                "total_baseline_time(us)": f"{total_baseline:.3f}",
                "total_tuned_time(us)": f"{total_tuned:.3f}",
                "baseline_time_percent": f"{baseline_percent:.2f}",
                "tuned_time_percent": f"{tuned_percent:.2f}"
            }
            writer.writerow(row)

        # print Gemm Speedup
        speedup = total_baseline_all / total_tuned_all if total_tuned_all else 0
        print("-" * len(header))
        print(f"Gemm Speedup: {speedup:.2f}x")
        print(f"Total baseline time: {total_baseline_all:.3f} us, Total tuned time: {total_tuned_all:.3f} us")

        # last line in CSV, TOTAL
        total_row = {
            "m": "-", "n": "-", "k": "-", "count": "-",
            "baseline_latency(us)": "-", "baseline_solution_idx": "-",
            "tuned_latency(us)": "-", "tuned_solution_idx": "-",
            "baseline/tuned": "-",
            "total_baseline_time(us)": f"{total_baseline_all:.3f}",
            "total_tuned_time(us)": f"{total_tuned_all:.3f}",
            "baseline_time_percent": "100.00",
            "tuned_time_percent": "100.00"
        }
        writer.writerow(total_row)

    print(f"\nOutput CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="calculate weighted improvement rate for each kernel"
    )
    parser.add_argument("--input_log", help="input hipblaslt log")
    parser.add_argument("--input_csv", help="tuning result csv file")
    parser.add_argument("--output_csv", help="tuning analysis csv file")

    args = parser.parse_args()
    count_mnksol(args.input_log, args.input_csv, args.output_csv)

