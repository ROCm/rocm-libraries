#!/usr/bin/env python3

"""
parse_exe_time.py - Parse execution time test results

This script parses the log file from exe-time tests, extracts relevant information,
and stores it in a structured format for easy comparison across different runs.
"""

import argparse
import csv
import json
import os
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path


def parse_yaml_config(yaml_file):
    """Parse YAML configuration to extract data types and transpose settings."""
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        # Extract problem type information from BenchmarkProblems
        if 'BenchmarkProblems' not in config:
            return None

        for problem_set in config['BenchmarkProblems']:
            if isinstance(problem_set, list) and len(problem_set) > 0:
                problem_type = problem_set[0]
                if isinstance(problem_type, dict):
                    # Extract RotatingBufferSize from GlobalParameters if present
                    rotating_buffer_size = 0
                    if 'GlobalParameters' in config:
                        rotating_buffer_size = config['GlobalParameters'].get('RotatingBufferSize', 0)

                    return {
                        'DataType': problem_type.get('DataType', 'Unknown'),
                        'DestDataType': problem_type.get('DestDataType', 'Unknown'),
                        'ComputeDataType': problem_type.get('ComputeDataType', 'Unknown'),
                        'TransposeA': problem_type.get('TransposeA', 'Unknown'),
                        'TransposeB': problem_type.get('TransposeB', 'Unknown'),
                        'RotatingBufferSize': rotating_buffer_size,
                    }

        return None
    except Exception as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        return None


def extract_sia_version(solution_name):
    """Extract SIA version (SIA1 or SIA3) from solution name."""
    match = re.search(r'_SIA(\d+)_', solution_name)
    if match:
        return f"SIA{match.group(1)}"
    return "Unknown"


def format_sia_version_for_output(sia_version):
    """Format SIA version for output: SIA1 -> stinkytofu, others unchanged."""
    if sia_version == "SIA1":
        return "stinkytofu"
    return sia_version


def parse_log_file(log_file):
    """Parse the log file and extract test results."""
    results = []

    try:
        with open(log_file, 'r') as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                # Skip header (starts with 'run') and empty lines
                if not row or (len(row) > 0 and row[0] == 'run'):
                    continue

                # CSV should have at least 12 columns (indices 0-11)
                if len(row) < 12:
                    continue

                # Extract relevant fields
                # Column indices: 0=run, 1=problem-progress, 2=solution-progress, 3=operation,
                # 4=problem-sizes, 5=bias-type, 6=factor-dim, 7=activation-type,
                # 8=solution, 9=validation, 10=time-us, 11=gflops
                try:
                    result = {
                        'problem_sizes': row[4],
                        'solution': row[8],
                        'validation': row[9],
                        'time_us': row[10],
                        'gflops': row[11],
                        'sia_version': extract_sia_version(row[8]),
                    }
                    results.append(result)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Failed to parse row: {row}", file=sys.stderr)
                    continue

    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return []

    return results


def load_database(db_file):
    """Load existing database or create new one."""
    if os.path.exists(db_file):
        try:
            with open(db_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load database: {e}", file=sys.stderr)
            return {}
    return {}


def save_database(db_file, data):
    """Save database to file."""
    try:
        with open(db_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving database: {e}", file=sys.stderr)
        return False
    return True


def create_problem_key(problem_sizes, yaml_config):
    """Create a unique key for a problem configuration."""
    return json.dumps({
        'problem_sizes': problem_sizes,
        'DataType': yaml_config['DataType'],
        'DestDataType': yaml_config['DestDataType'],
        'ComputeDataType': yaml_config['ComputeDataType'],
        'TransposeA': yaml_config['TransposeA'],
        'TransposeB': yaml_config['TransposeB'],
        'RotatingBufferSize': yaml_config['RotatingBufferSize'],
    }, sort_keys=True)


def create_run_key(git_info, test_date, sia_version, run_idx):
    """Create a unique key for a test run."""
    return f"{git_info}_{test_date}_{sia_version}_run{run_idx}"


def update_database(db, results, yaml_config, git_info, test_date, hostname, gpu_arch):
    """Update database with new results."""
    # Group results by SIA version
    sia_groups = {}
    for result in results:
        sia = result['sia_version']
        if sia not in sia_groups:
            sia_groups[sia] = []
        sia_groups[sia].append(result)

    # Process each result
    for sia_version, sia_results in sia_groups.items():
        for idx, result in enumerate(sia_results):
            # Create problem key
            problem_key = create_problem_key(result['problem_sizes'], yaml_config)

            # Create run key
            run_key = create_run_key(git_info, test_date, sia_version, idx)

            # Initialize problem entry if not exists
            if problem_key not in db:
                db[problem_key] = {
                    'problem_config': {
                        'problem_sizes': result['problem_sizes'],
                        'DataType': yaml_config['DataType'],
                        'DestDataType': yaml_config['DestDataType'],
                        'ComputeDataType': yaml_config['ComputeDataType'],
                        'TransposeA': yaml_config['TransposeA'],
                        'TransposeB': yaml_config['TransposeB'],
                        'RotatingBufferSize': yaml_config['RotatingBufferSize'],
                    },
                    'runs': {}
                }

            # Add run data
            db[problem_key]['runs'][run_key] = {
                'git_info': git_info,
                'test_date': test_date,
                'sia_version': sia_version,
                'run_idx': idx,
                'hostname': hostname,
                'gpu_arch': gpu_arch,
                'solution': result['solution'],
                'validation': result['validation'],
                'time_us': result['time_us'],
                'gflops': result['gflops'],
            }


def create_comparison_report(db, output_file):
    """Create a human-readable comparison report."""
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("Execution Time Comparison Report\n")
        f.write("=" * 120 + "\n\n")

        for problem_key, problem_data in sorted(db.items()):
            config = problem_data['problem_config']

            f.write("-" * 120 + "\n")
            f.write(f"Problem: {config['problem_sizes']}\n")
            f.write(f"  DataType: {config['DataType']}, "
                   f"DestDataType: {config['DestDataType']}, "
                   f"ComputeDataType: {config['ComputeDataType']}\n")
            f.write(f"  TransposeA: {config['TransposeA']}, "
                   f"TransposeB: {config['TransposeB']}, "
                   f"RotatingBufferSize: {config.get('RotatingBufferSize', 0)}\n")
            f.write("-" * 120 + "\n\n")

            # Group runs by SIA version
            sia_runs = {}
            for run_key, run_data in problem_data['runs'].items():
                sia = run_data['sia_version']
                if sia not in sia_runs:
                    sia_runs[sia] = []
                sia_runs[sia].append((run_key, run_data))

            # Display runs grouped by SIA version
            for sia_version in sorted(sia_runs.keys()):
                f.write(f"  {sia_version}:\n")

                for run_key, run_data in sorted(sia_runs[sia_version],
                                                key=lambda x: x[1]['test_date'],
                                                reverse=True):
                    f.write(f"    {run_data['test_date']} | "
                           f"git:{run_data['git_info']} | "
                           f"{run_data['hostname']} | "
                           f"{run_data['gpu_arch']}\n")
                    f.write(f"      Validation: {run_data['validation']}, "
                           f"Time: {run_data['time_us']} us, "
                           f"GFLOPS: {run_data['gflops']}\n")

                f.write("\n")

            f.write("\n")


def save_local_results(output_dir, yaml_name, results, yaml_config, git_info, test_date, hostname, gpu_arch):
    """Save local results to JSON files in the log directory."""

    # ===== Part 1: Save detailed results for database updates (local-sia3cmp.json) =====
    local_detailed_json = os.path.join(output_dir, 'local-sia3cmp.json')

    # Create detailed local data structure
    local_data = {
        'test_name': yaml_name,
        'config': yaml_config,
        'git_info': git_info,
        'test_date': test_date,
        'hostname': hostname,
        'gpu_arch': gpu_arch,
        'results': []
    }

    # Add all results
    for result in results:
        local_data['results'].append({
            'problem_sizes': result['problem_sizes'],
            'solution': result['solution'],
            'sia_version': format_sia_version_for_output(result['sia_version']),
            'validation': result['validation'],
            'time_us': result['time_us'],
            'gflops': result['gflops']
        })

    # Load existing detailed results
    if os.path.exists(local_detailed_json):
        try:
            with open(local_detailed_json, 'r') as f:
                existing_detailed = json.load(f)
            if not isinstance(existing_detailed, list):
                existing_detailed = [existing_detailed]
        except:
            existing_detailed = []
    else:
        existing_detailed = []

    # Append new results
    existing_detailed.append(local_data)

    # Save detailed results
    try:
        with open(local_detailed_json, 'w') as f:
            json.dump(existing_detailed, f, indent=2)
    except Exception as e:
        print(f"Error saving detailed local results: {e}", file=sys.stderr)
        return False

    # Create detailed report from local-sia3cmp.json in table format
    # Combine results from all test runs (e.g., mfma and mfma-sia3) into single tables
    local_detailed_report = os.path.join(output_dir, 'local-sia3cmp_report.txt')
    try:
        with open(local_detailed_report, 'w') as f:
            f.write("Execution Time Detailed Results - Stinkytofu vs SIA3 Comparison\n")

            # First, collect all results grouped by problem configuration
            # Key: (problem_sizes, config_json), Value: {sia_version: result}
            all_problems = {}
            test_metadata = []  # Track test info for display

            for test_data in existing_detailed:
                config = test_data.get('config', {})
                test_metadata.append({
                    'test_name': test_data['test_name'],
                    'git_info': test_data['git_info'],
                    'test_date': test_data['test_date'],
                    'hostname': test_data['hostname'],
                    'gpu_arch': test_data['gpu_arch']
                })

                for result in test_data.get('results', []):
                    problem = result['problem_sizes']
                    sia = result['sia_version']

                    # Normalize sia version name
                    if sia.lower() in ['sia1', 'stinkytofu']:
                        sia = 'stinkytofu'

                    # Create unique key for this problem configuration
                    config_key = json.dumps({
                        'DataType': config.get('DataType', 'N/A'),
                        'DestDataType': config.get('DestDataType', 'N/A'),
                        'ComputeDataType': config.get('ComputeDataType', 'N/A'),
                        'TransposeA': config.get('TransposeA', 'N/A'),
                        'TransposeB': config.get('TransposeB', 'N/A'),
                        'RotatingBufferSize': config.get('RotatingBufferSize', 0),
                    }, sort_keys=True)

                    problem_key = (problem, config_key)

                    if problem_key not in all_problems:
                        all_problems[problem_key] = {
                            'config': config,
                            'problem_sizes': problem,
                            'sia_versions': {}
                        }

                    # Keep both latest and fastest results for each sia version
                    if sia not in all_problems[problem_key]['sia_versions']:
                        all_problems[problem_key]['sia_versions'][sia] = {
                            'latest': result,
                            'fastest': result
                        }
                    else:
                        # Update latest if newer
                        existing_date = all_problems[problem_key]['sia_versions'][sia]['latest'].get('test_date', '')
                        new_date = test_data.get('test_date', '')
                        if new_date >= existing_date:
                            all_problems[problem_key]['sia_versions'][sia]['latest'] = result

                        # Update fastest if faster
                        try:
                            existing_time = float(all_problems[problem_key]['sia_versions'][sia]['fastest'].get('time_us', float('inf')))
                            new_time = float(result.get('time_us', float('inf')))
                            if new_time < existing_time:
                                all_problems[problem_key]['sia_versions'][sia]['fastest'] = result
                        except (ValueError, TypeError):
                            pass

            # Write test metadata
            if test_metadata:
                for meta in test_metadata:
                    f.write(f"Test: {meta['test_name']}\n")
                    f.write(f"Git: {meta['git_info']}, Date: {meta['test_date']}\n")
                    f.write(f"Host: {meta['hostname']}, GPU: {meta['gpu_arch']}\n")
                f.write("-" * 180 + "\n\n")

            # Generate combined tables for each problem (latest and best)
            for problem_key in sorted(all_problems.keys()):
                problem_data = all_problems[problem_key]
                problem_sizes = problem_data['problem_sizes']
                config = problem_data['config']
                sia_versions = problem_data['sia_versions']

                # Get latest and fastest results
                stinky_data = sia_versions.get('stinkytofu', {})
                sia3_data = sia_versions.get('SIA3', {})

                stinky_latest = stinky_data.get('latest') if stinky_data else None
                stinky_fastest = stinky_data.get('fastest') if stinky_data else None
                sia3_latest = sia3_data.get('latest') if sia3_data else None
                sia3_fastest = sia3_data.get('fastest') if sia3_data else None

                # === LATEST RESULTS TABLE ===
                # Calculate percentage difference for latest
                pct_diff_latest = ''
                if stinky_latest and sia3_latest:
                    try:
                        stinky_time = float(stinky_latest['time_us'])
                        sia3_time = float(sia3_latest['time_us'])
                        if sia3_time > 0:
                            pct_diff_latest = ((sia3_time - stinky_time) / sia3_time) * 100
                            pct_diff_latest = f"{pct_diff_latest:+.2f}%"
                    except:
                        pct_diff_latest = 'N/A'

                # Latest table header
                f.write(f"{'':50} | {'Stinkytofu (Latest)':40} | {'SIA3 (Latest)':40} | {'% Diff':15}\n")
                f.write("-" * 50 + "+" + "-" * 42 + "+" + "-" * 42 + "+" + "-" * 17 + "\n")

                # Problem configuration
                f.write(f"{'Problem: ' + problem_sizes:50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  DataType: ' + str(config.get('DataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  DestDataType: ' + str(config.get('DestDataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  ComputeDataType: ' + str(config.get('ComputeDataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  TransposeA: ' + str(config.get('TransposeA', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  TransposeB: ' + str(config.get('TransposeB', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  RotatingBufferSize: ' + str(config.get('RotatingBufferSize', 0)):50} | {'':<40} | {'':<40} | {'':<15}\n")

                # Latest performance data
                stinky_val = stinky_latest.get('validation', 'N/A') if stinky_latest else 'N/A'
                stinky_time = f"{stinky_latest['time_us']}" if stinky_latest else 'N/A'
                stinky_gflops = f"{stinky_latest['gflops']}" if stinky_latest else 'N/A'

                sia3_val = sia3_latest.get('validation', 'N/A') if sia3_latest else 'N/A'
                sia3_time = f"{sia3_latest['time_us']}" if sia3_latest else 'N/A'
                sia3_gflops = f"{sia3_latest['gflops']}" if sia3_latest else 'N/A'

                f.write(f"{'':50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'Validation':50} | {str(stinky_val):<40} | {str(sia3_val):<40} | {'':<15}\n")
                f.write(f"{'Time':50} | {str(stinky_time) + ' us':<40} | {str(sia3_time) + ' us':<40} | {pct_diff_latest:<15}\n")
                f.write(f"{'GFLOPS':50} | {str(stinky_gflops):<40} | {str(sia3_gflops):<40} | {'':<15}\n")

                f.write("\n")

                # === BEST (FASTEST) RESULTS TABLE ===
                # Calculate percentage difference for best
                pct_diff_best = ''
                if stinky_fastest and sia3_fastest:
                    try:
                        stinky_time_best = float(stinky_fastest['time_us'])
                        sia3_time_best = float(sia3_fastest['time_us'])
                        if sia3_time_best > 0:
                            pct_diff_best = ((sia3_time_best - stinky_time_best) / sia3_time_best) * 100
                            pct_diff_best = f"{pct_diff_best:+.2f}%"
                    except:
                        pct_diff_best = 'N/A'

                # Best table header
                f.write(f"{'':50} | {'Stinkytofu (Best)':40} | {'SIA3 (Best)':40} | {'% Diff':15}\n")
                f.write("-" * 50 + "+" + "-" * 42 + "+" + "-" * 42 + "+" + "-" * 17 + "\n")

                # Problem configuration (same as above)
                f.write(f"{'Problem: ' + problem_sizes:50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  DataType: ' + str(config.get('DataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  DestDataType: ' + str(config.get('DestDataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  ComputeDataType: ' + str(config.get('ComputeDataType', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  TransposeA: ' + str(config.get('TransposeA', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  TransposeB: ' + str(config.get('TransposeB', 'N/A')):50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'  RotatingBufferSize: ' + str(config.get('RotatingBufferSize', 0)):50} | {'':<40} | {'':<40} | {'':<15}\n")

                # Best performance data
                stinky_val_best = stinky_fastest.get('validation', 'N/A') if stinky_fastest else 'N/A'
                stinky_time_best = f"{stinky_fastest['time_us']}" if stinky_fastest else 'N/A'
                stinky_gflops_best = f"{stinky_fastest['gflops']}" if stinky_fastest else 'N/A'

                sia3_val_best = sia3_fastest.get('validation', 'N/A') if sia3_fastest else 'N/A'
                sia3_time_best = f"{sia3_fastest['time_us']}" if sia3_fastest else 'N/A'
                sia3_gflops_best = f"{sia3_fastest['gflops']}" if sia3_fastest else 'N/A'

                f.write(f"{'':50} | {'':<40} | {'':<40} | {'':<15}\n")
                f.write(f"{'Validation':50} | {str(stinky_val_best):<40} | {str(sia3_val_best):<40} | {'':<15}\n")
                f.write(f"{'Time':50} | {str(stinky_time_best) + ' us':<40} | {str(sia3_time_best) + ' us':<40} | {pct_diff_best:<15}\n")
                f.write(f"{'GFLOPS':50} | {str(stinky_gflops_best):<40} | {str(sia3_gflops_best):<40} | {'':<15}\n")

                f.write("\n\n")
    except Exception as e:
        print(f"Warning: Failed to create detailed report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    # ===== Part 2: Generate summary for quick review (local.json) =====
    local_json = os.path.join(output_dir, 'local.json')

    # Load existing summary
    if os.path.exists(local_json):
        try:
            with open(local_json, 'r') as f:
                summary = json.load(f)
        except:
            summary = {}
    else:
        summary = {}

    # Add times to summary
    for result in results:
        if result.get('validation') != 'PASSED':
            continue

        try:
            time_us = float(result.get('time_us', 0))
        except (ValueError, TypeError):
            continue

        # Key format: "yaml_name.yaml"
        key = f"{yaml_name}.yaml"

        # Initialize key if not exists
        if key not in summary:
            summary[key] = []

        # Append time
        summary[key].append(time_us)

    # Save summary JSON
    try:
        with open(local_json, 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Error saving summary: {e}", file=sys.stderr)
        return False

    # ===== Part 3: Create local report from summary =====
    local_report = os.path.join(output_dir, 'local_report.txt')
    try:
        with open(local_report, 'w') as f:
            f.write("Execution Time Run Summary\n\n")

            if not summary:
                f.write("No data available.\n")
                return True

            # Find max number of runs
            max_runs = max(len(times) for times in summary.values()) if summary else 0

            # Create header
            header_parts = [f"{'Test Case':<40}"]
            for run_idx in range(max_runs):
                header_parts.append(f"run{run_idx:<5}")

            header = " ".join(header_parts)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Write data rows
            for test_case in sorted(summary.keys()):
                times = summary[test_case]

                row_parts = [f"{test_case:<40}"]

                # Add run times
                for time_val in times:
                    row_parts.append(f"{time_val:<8.2f}")

                f.write(" ".join(row_parts) + "\n")

            f.write("\n")
            f.write("Note: Times in microseconds (us)\n")
    except Exception as e:
        print(f"Warning: Failed to create local report: {e}", file=sys.stderr)

    return True


def main():
    parser = argparse.ArgumentParser(description='Parse exe-time test results and save locally')
    parser.add_argument('--log-file', required=True, help='Path to log file')
    parser.add_argument('--yaml-file', required=True, help='Path to YAML config file')
    parser.add_argument('--output-dir', required=True, help='Output directory (log folder)')
    parser.add_argument('--git-info', required=True, help='Git commit info')
    parser.add_argument('--hostname', required=True, help='Hostname')
    parser.add_argument('--gpu-arch', required=True, help='GPU architecture')
    parser.add_argument('--test-date', required=True, help='Test date')
    parser.add_argument('--yaml-name', required=True, help='YAML test name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Parse YAML configuration
    yaml_config = parse_yaml_config(args.yaml_file)
    if yaml_config is None:
        print("Error: Failed to parse YAML configuration", file=sys.stderr)
        return 1

    # Parse log file
    results = parse_log_file(args.log_file)
    if not results:
        print("Warning: No results found in log file", file=sys.stderr)
        return 0

    if args.verbose:
        print(f"Parsed {len(results)} result(s)")

    # Save local results to log directory
    if not save_local_results(args.output_dir, args.yaml_name, results, yaml_config,
                              args.git_info, args.test_date, args.hostname, args.gpu_arch):
        return 1

    if args.verbose:
        print(f"Detailed results saved to: {os.path.join(args.output_dir, 'local-sia3cmp.json')}")
        print(f"Detailed report saved to: {os.path.join(args.output_dir, 'local-sia3cmp_report.txt')}")
        print(f"Summary saved to: {os.path.join(args.output_dir, 'local.json')}")
        print(f"Summary report saved to: {os.path.join(args.output_dir, 'local_report.txt')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

