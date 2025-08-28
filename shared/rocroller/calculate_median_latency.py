#!/usr/bin/env python3
"""
Script to process CSV files in build/output/ and calculate median latency.
"""

import os
import csv
import statistics
import re
import argparse
from pathlib import Path


def parse_directory_name(path):
    """Parse directory name to extract operation, bit width, and stride for sorting."""
    dirname = os.path.basename(os.path.dirname(path))
    # Pattern: ds_(read|write)_b(32|64|128)_stride_(1|2|4|8|16|32)
    match = re.match(r"ds_(read|write)_b(\d+)_stride_(\d+)", dirname)
    if match:
        op_type = match.group(1)
        bit_width = int(match.group(2))
        stride = int(match.group(3))
        return (op_type, bit_width, stride)
    return (dirname, 0, 0)  # Fallback for non-matching names


def find_csv_files(directory):
    """Find all stats_ui_output CSV files in the given directory."""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("stats_ui_output_agent_") and file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # Sort by operation type, then bit width (numerically), then stride
    return sorted(csv_files, key=parse_directory_name)


def calculate_median_latency(csv_file):
    """Calculate median latency divided by median hitcount from a CSV file."""
    latencies = []
    hitcounts = []

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract latency and hitcount values
                latency = int(row["Latency"])
                hitcount = int(row["Hitcount"])
                # Only include rows with non-zero values
                if hitcount > 0 and latency > 0:
                    latencies.append(latency)
                    hitcounts.append(hitcount)

        if latencies and hitcounts:
            median_latency = statistics.mode(latencies)
            median_hitcount = statistics.mode(hitcounts)
            # Avoid division by zero
            if median_hitcount > 0:
                return median_latency / median_hitcount
            else:
                return None
        else:
            return None

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def main():
    """Main function to process all CSV files."""
    parser = argparse.ArgumentParser(
        description="Calculate median latency from CSV files in a directory"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for CSV files (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = args.directory

    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} not found!")
        return

    csv_files = find_csv_files(output_dir)

    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process\n")
    print("-" * 80)
    print(f"{'File Path':<60} {'Med Latency/Med Hitcount':>25}")
    print("-" * 80)

    for csv_file in csv_files:
        # Get relative path for cleaner output
        relative_path = os.path.relpath(csv_file, output_dir)
        median_latency = calculate_median_latency(csv_file)

        if median_latency is not None:
            print(f"{relative_path:<60} {median_latency:>25.1f}")
        else:
            print(f"{relative_path:<60} {'N/A':>25}")

    print("-" * 80)


if __name__ == "__main__":
    main()
