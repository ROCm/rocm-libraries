#!/usr/bin/env python3

"""
export_database_csv.py - Export database results to CSV format

This script reads the database JSON files and exports them to CSV format
with test cases as rows and git commits as columns.
It also exports sia3cmp database for stinkytofu vs SIA3 comparisons.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_database(db_file: str) -> Dict:
    """Load database from JSON file."""
    try:
        with open(db_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading database: {e}", file=sys.stderr)
        return {}


def transform_database_to_csv_format(db: Dict, time_type: str = 'best') -> Dict[str, Dict[str, float]]:
    """
    Transform database from {git_hash: {testcase: {best_time, latest_time}}}
    to {testcase: {git_hash: time}} for CSV export.

    Args:
        db: Database dictionary
        time_type: 'best' or 'latest' to select which time to export

    Returns: {test_case: {git_hash: time}}
    """
    results = defaultdict(dict)

    # Database structure: {git_hash: {testcase: {best_time, latest_time}}} or old format {git_hash: {testcase: time}}
    for git_hash, test_results in db.items():
        for test_case, time_value in test_results.items():
            # Handle both new dict format and old scalar format
            if isinstance(time_value, dict):
                # New format with best_time and latest_time
                if time_type == 'best':
                    results[test_case][git_hash] = time_value.get('best_time', time_value.get('latest_time'))
                else:  # latest
                    results[test_case][git_hash] = time_value.get('latest_time', time_value.get('best_time'))
            else:
                # Old scalar format - use the same value for both
                results[test_case][git_hash] = time_value

    return dict(results)


def export_to_csv(results: Dict[str, Dict[str, float]], output_file: str, metric_name: str):
    """Export results to CSV format."""
    if not results:
        print(f"Warning: No data to export for {output_file}", file=sys.stderr)
        return

    # Get all unique git commits (sorted)
    all_git_commits = set()
    for test_case_data in results.values():
        all_git_commits.update(test_case_data.keys())

    git_commits = sorted(all_git_commits)

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header row
        header = ['testcase'] + git_commits
        writer.writerow(header)

        # Data rows (sorted by test case name)
        for test_case in sorted(results.keys()):
            row = [test_case]

            for git_commit in git_commits:
                time_value = results[test_case].get(git_commit)
                if time_value is not None and time_value != float('inf'):
                    if metric_name == 'time_us':
                        row.append(f"{time_value:.3f}")
                    else:  # time_sec
                        row.append(f"{time_value:.6f}")
                else:
                    row.append('')  # Empty cell if no data

            writer.writerow(row)

    print(f"Exported {len(results)} test cases to: {output_file}")


def format_speedup(stinky_time, sia3_time):
    """Calculate and format speedup as percentage difference."""
    if stinky_time is None or sia3_time is None:
        return "N/A", "N/A"
    if stinky_time <= 0 or sia3_time <= 0:
        return "N/A", "N/A"

    # Calculate percentage difference: (sia3 - stinky) / sia3 * 100
    # Positive means stinkytofu is faster, negative means slower
    pct_diff = ((sia3_time - stinky_time) / sia3_time) * 100

    # Also calculate speedup multiplier
    speedup = sia3_time / stinky_time
    if speedup > 1:
        speedup_str = f"{speedup:.2f}x faster"
    else:
        speedup_str = f"{1/speedup:.2f}x slower"

    return f"{pct_diff:+.2f}%", speedup_str


def export_sia3cmp_to_csv(db, output_file, hostname, gpu_arch):
    """Export sia3cmp database to CSV format."""
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Problem_Sizes',
                'Git_Hash',
                'Type',
                'Stinkytofu_Time_us',
                'Stinkytofu_GFLOPS',
                'Stinkytofu_Date',
                'Stinkytofu_Validation',
                'SIA3_Time_us',
                'SIA3_GFLOPS',
                'SIA3_Date',
                'SIA3_Validation',
                'Percent_Diff',
                'Speedup',
                'DataType',
                'DestDataType',
                'ComputeDataType',
                'TransposeA',
                'TransposeB',
                'RotatingBufferSize'
            ])

            # Write data
            for problem_key, problem_data in sorted(db.items()):
                config = problem_data['problem_config']
                problem_sizes = config['problem_sizes']

                for git_hash in sorted(problem_data['git_hashes'].keys()):
                    git_data = problem_data['git_hashes'][git_hash]

                    stinky_data = git_data.get('stinkytofu', {})
                    sia3_data = git_data.get('SIA3', {})

                    # Write fastest comparison
                    stinky_fastest = stinky_data.get('fastest')
                    sia3_fastest = sia3_data.get('fastest')

                    pct_diff_fastest = ''
                    speedup_fastest = ''
                    if stinky_fastest and sia3_fastest:
                        pct_diff_fastest, speedup_fastest = format_speedup(
                            stinky_fastest['time_us'],
                            sia3_fastest['time_us']
                        )

                    writer.writerow([
                        problem_sizes,
                        git_hash,
                        'Fastest',
                        f"{stinky_fastest['time_us']:.2f}" if stinky_fastest else '',
                        f"{stinky_fastest['gflops']:.2f}" if stinky_fastest else '',
                        stinky_fastest['test_date'] if stinky_fastest else '',
                        stinky_fastest.get('validation', '') if stinky_fastest else '',
                        f"{sia3_fastest['time_us']:.2f}" if sia3_fastest else '',
                        f"{sia3_fastest['gflops']:.2f}" if sia3_fastest else '',
                        sia3_fastest['test_date'] if sia3_fastest else '',
                        sia3_fastest.get('validation', '') if sia3_fastest else '',
                        pct_diff_fastest,
                        speedup_fastest,
                        config['DataType'],
                        config['DestDataType'],
                        config['ComputeDataType'],
                        config['TransposeA'],
                        config['TransposeB'],
                        config.get('RotatingBufferSize', 0)
                    ])

                    # Write latest comparison
                    stinky_latest = stinky_data.get('latest')
                    sia3_latest = sia3_data.get('latest')

                    pct_diff_latest = ''
                    speedup_latest = ''
                    if stinky_latest and sia3_latest:
                        pct_diff_latest, speedup_latest = format_speedup(
                            stinky_latest['time_us'],
                            sia3_latest['time_us']
                        )

                    writer.writerow([
                        problem_sizes,
                        git_hash,
                        'Latest',
                        f"{stinky_latest['time_us']:.2f}" if stinky_latest else '',
                        f"{stinky_latest['gflops']:.2f}" if stinky_latest else '',
                        stinky_latest['test_date'] if stinky_latest else '',
                        stinky_latest.get('validation', '') if stinky_latest else '',
                        f"{sia3_latest['time_us']:.2f}" if sia3_latest else '',
                        f"{sia3_latest['gflops']:.2f}" if sia3_latest else '',
                        sia3_latest['test_date'] if sia3_latest else '',
                        sia3_latest.get('validation', '') if sia3_latest else '',
                        pct_diff_latest,
                        speedup_latest,
                        config['DataType'],
                        config['DestDataType'],
                        config['ComputeDataType'],
                        config['TransposeA'],
                        config['TransposeB'],
                        config.get('RotatingBufferSize', 0)
                    ])

        return True
    except Exception as e:
        print(f"Error exporting sia3cmp to CSV: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Export database results to CSV format')
    parser.add_argument('--database-dir', required=True, help='Path to database directory')
    parser.add_argument('--hostname', required=True, help='Hostname')
    parser.add_argument('--gpu-arch', required=True, help='GPU architecture (e.g., gfx950)')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Validate database directory
    if not os.path.exists(args.database_dir):
        print(f"Error: Database directory not found: {args.database_dir}", file=sys.stderr)
        return 1

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load databases
    exe_time_db_file = os.path.join(args.database_dir, f"exe_time-{args.hostname}_{args.gpu_arch}.json")
    codegen_time_db_file = os.path.join(args.database_dir, f"codegen_time-{args.hostname}_{args.gpu_arch}.json")
    sia3cmp_db_file = os.path.join(args.database_dir, f"sia3cmp-{args.hostname}_{args.gpu_arch}.json")

    # Process exe_time
    if os.path.exists(exe_time_db_file):
        print(f"Loading exe_time database: {exe_time_db_file}")
        exe_db = load_database(exe_time_db_file)

        # Export best times
        exe_best_results = transform_database_to_csv_format(exe_db, time_type='best')
        exe_best_csv_file = os.path.join(args.output_dir, f"exe_time_best_{args.hostname}_{args.gpu_arch}.csv")
        export_to_csv(exe_best_results, exe_best_csv_file, 'time_us')

        # Export latest times
        exe_latest_results = transform_database_to_csv_format(exe_db, time_type='latest')
        exe_latest_csv_file = os.path.join(args.output_dir, f"exe_time_latest_{args.hostname}_{args.gpu_arch}.csv")
        export_to_csv(exe_latest_results, exe_latest_csv_file, 'time_us')
    else:
        print(f"Warning: exe_time database not found: {exe_time_db_file}", file=sys.stderr)

    # Process codegen_time
    if os.path.exists(codegen_time_db_file):
        print(f"Loading codegen_time database: {codegen_time_db_file}")
        codegen_db = load_database(codegen_time_db_file)

        # Export best times
        codegen_best_results = transform_database_to_csv_format(codegen_db, time_type='best')
        codegen_best_csv_file = os.path.join(args.output_dir, f"codegen_time_best_{args.hostname}_{args.gpu_arch}.csv")
        export_to_csv(codegen_best_results, codegen_best_csv_file, 'time_sec')

        # Export latest times
        codegen_latest_results = transform_database_to_csv_format(codegen_db, time_type='latest')
        codegen_latest_csv_file = os.path.join(args.output_dir, f"codegen_time_latest_{args.hostname}_{args.gpu_arch}.csv")
        export_to_csv(codegen_latest_results, codegen_latest_csv_file, 'time_sec')
    else:
        print(f"Warning: codegen_time database not found: {codegen_time_db_file}", file=sys.stderr)

    # Process sia3cmp database
    if os.path.exists(sia3cmp_db_file):
        if args.verbose:
            print(f"Loading sia3cmp database: {sia3cmp_db_file}")

        sia3cmp_db = load_database(sia3cmp_db_file)

        if sia3cmp_db:
            if args.verbose:
                print(f"Sia3cmp database has {len(sia3cmp_db)} problem(s)")

            # Export to CSV
            sia3cmp_csv_file = os.path.join(args.output_dir, f"sia3cmp-{args.hostname}_{args.gpu_arch}.csv")
            if args.verbose:
                print(f"Exporting sia3cmp to CSV: {sia3cmp_csv_file}")

            if export_sia3cmp_to_csv(sia3cmp_db, sia3cmp_csv_file, args.hostname, args.gpu_arch):
                print(f"Exported sia3cmp to: {sia3cmp_csv_file}")
    else:
        if args.verbose:
            print(f"Info: sia3cmp database not found: {sia3cmp_db_file} (skipping)", file=sys.stderr)

    print("\nCSV export completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
