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


def extract_specific_lines(csv_file):
    """Extract data from lines 3, 4, 5, and 6 of a CSV file."""
    lines_data = []

    try:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Skip header (row 0) and get rows 2, 3, 4, 5 (lines 3, 4, 5, 6)
            for i in [2, 3, 4, 5]:
                if i < len(rows):
                    row = rows[i]
                    if len(row) >= 5:
                        # Extract Instruction (index 2) and Latency (index 4)
                        instruction = row[2]
                        latency = row[4]
                        lines_data.append((instruction, latency))
                    else:
                        lines_data.append(("N/A", "N/A"))
                else:
                    lines_data.append(("N/A", "N/A"))

        return lines_data
    except Exception as e:
        print(f"Error reading lines from {csv_file}: {e}")
        return [("N/A", "N/A"), ("N/A", "N/A"), ("N/A", "N/A"), ("N/A", "N/A")]


def calculate_median_latency(csv_file):
    """Calculate mode latency divided by mode hitcount from a CSV file."""
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
            mode_latency = statistics.mode(latencies)
            mode_hitcount = statistics.mode(hitcounts)
            # Avoid division by zero
            if mode_hitcount > 0:
                return mode_latency / mode_hitcount
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
        description="Calculate mode latency from CSV files in a directory"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="output",
        help="Directory to search for CSV files",
    )
    parser.add_argument(
        "--hide-path",
        "-H",
        action="store_true",
        help="Hide the full path column in the output (default: show path)",
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

    # Adjust formatting based on whether path is shown
    if args.hide_path:
        separator_width = 227
        print("-" * separator_width)
        print(
            f"{'Directory':<35} | {'Latency':>10} | {'Line 3 Instruction':>40} | {'Line 4 Instruction':>40} | {'Line 5 Instruction':>40} | {'Line 6 Instruction':>40}"
        )
        print("-" * separator_width)
    else:
        separator_width = 280
        print("-" * separator_width)
        print(
            f"{'Directory':<35} | {'Latency':>10} | {'Line 3 Instruction':>40} | {'Line 4 Instruction':>40} | {'Line 5 Instruction':>40} | {'Line 6 Instruction':>40} | {'Full Path':<50}"
        )
        print("-" * separator_width)

    for csv_file in csv_files:
        # Get absolute path
        absolute_path = os.path.abspath(csv_file)
        # Extract directory name (e.g., ds_read_b32_stride_1)
        dir_name = os.path.basename(os.path.dirname(csv_file))
        median_latency = calculate_median_latency(csv_file)

        # Extract data from lines 3, 4, 5, and 6
        lines_data = extract_specific_lines(csv_file)

        # Format line data for display (instruction only, no latency)
        line_3_str = lines_data[0][0][:40] if lines_data[0][0] != "N/A" else "N/A"
        line_4_str = lines_data[1][0][:40] if lines_data[1][0] != "N/A" else "N/A"
        line_5_str = lines_data[2][0][:40] if lines_data[2][0] != "N/A" else "N/A"
        line_6_str = lines_data[3][0][:40] if lines_data[3][0] != "N/A" else "N/A"

        if args.hide_path:
            if median_latency is not None:
                print(
                    f"{dir_name:<35} | {median_latency:>10.1f} | {line_3_str:>40} | {line_4_str:>40} | {line_5_str:>40} | {line_6_str:>40}"
                )
            else:
                print(
                    f"{dir_name:<35} | {'N/A':>10} | {line_3_str:>40} | {line_4_str:>40} | {line_5_str:>40} | {line_6_str:>40}"
                )
        else:
            if median_latency is not None:
                print(
                    f"{dir_name:<35} | {median_latency:>10.1f} | {line_3_str:>40} | {line_4_str:>40} | {line_5_str:>40} | {line_6_str:>40} | {absolute_path}"
                )
            else:
                print(
                    f"{dir_name:<35} | {'N/A':>10} | {line_3_str:>40} | {line_4_str:>40} | {line_5_str:>40} | {line_6_str:>40} | {absolute_path}"
                )

    print("-" * separator_width)


if __name__ == "__main__":
    main()
