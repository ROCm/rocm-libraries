#!/usr/bin/env python3

"""
parse_codegen_time.py - Parse code generation time test results

This script parses the yappi profiling results from codegen-time tests,
extracts the _getKernelSource timing, and stores it in a structured format.
"""

import argparse
import json
import os
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path


def parse_yaml_config(yaml_file):
    """Parse YAML configuration to extract problem information."""
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
                    return {
                        'DataType': problem_type.get('DataType', 'Unknown'),
                        'DestDataType': problem_type.get('DestDataType', 'Unknown'),
                        'ComputeDataType': problem_type.get('ComputeDataType', 'Unknown'),
                        'TransposeA': problem_type.get('TransposeA', 'Unknown'),
                        'TransposeB': problem_type.get('TransposeB', 'Unknown'),
                        'OperationType': problem_type.get('OperationType', 'Unknown'),
                    }

        return None
    except Exception as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        return None


def parse_yappi_file(yappi_file):
    """Parse yappi profiling results and extract _getKernelSource timing."""
    try:
        with open(yappi_file, 'r') as f:
            for line in f:
                # Look for the line containing '_getKernelSource'
                if 'lWriterAssembly._getKernelSource' in line or \
                   'WriterAssembly._getKernelSource' in line:
                    # Parse the line to extract ttot
                    # Format: name  ncall  tsub  ttot  tavg
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            # ttot is typically the 4th column (index 3)
                            ttot = float(parts[3])
                            return ttot
                        except (ValueError, IndexError):
                            continue

        print("Warning: _getKernelSource timing not found in yappi results", file=sys.stderr)
        return None

    except Exception as e:
        print(f"Error reading yappi file: {e}", file=sys.stderr)
        return None


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


def update_database(db, yaml_name, yaml_config, codegen_time, git_info, test_date,
                   hostname, gpu_arch):
    """Update database with new codegen time result."""
    # Create test key
    test_key = f"{yaml_name}"

    # Initialize test entry if not exists
    if test_key not in db:
        db[test_key] = {
            'test_name': yaml_name,
            'config': yaml_config,
            'runs': []
        }

    # Add run data
    run_data = {
        'git_info': git_info,
        'test_date': test_date,
        'hostname': hostname,
        'gpu_arch': gpu_arch,
        'codegen_time_sec': codegen_time,
    }

    db[test_key]['runs'].append(run_data)

    # Keep only last 100 runs per test
    if len(db[test_key]['runs']) > 100:
        db[test_key]['runs'] = sorted(db[test_key]['runs'],
                                      key=lambda x: x['test_date'],
                                      reverse=True)[:100]


def create_comparison_report(db, output_file):
    """Create a human-readable comparison report."""
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Code Generation Time Comparison Report\n")
        f.write("=" * 100 + "\n\n")

        for test_key, test_data in sorted(db.items()):
            f.write("-" * 100 + "\n")
            f.write(f"Test: {test_data['test_name']}\n")

            config = test_data['config']
            f.write(f"  DataType: {config['DataType']}, "
                   f"DestDataType: {config['DestDataType']}, "
                   f"ComputeDataType: {config['ComputeDataType']}\n")
            f.write(f"  TransposeA: {config['TransposeA']}, "
                   f"TransposeB: {config['TransposeB']}, "
                   f"OperationType: {config['OperationType']}\n")
            f.write("-" * 100 + "\n\n")

            # Sort runs by date (most recent first)
            sorted_runs = sorted(test_data['runs'],
                               key=lambda x: x['test_date'],
                               reverse=True)

            f.write("  Recent runs:\n")
            for run in sorted_runs[:10]:  # Show last 10 runs
                f.write(f"    {run['test_date']} | "
                       f"git:{run['git_info']} | "
                       f"{run['hostname']} | "
                       f"{run['gpu_arch']}\n")
                f.write(f"      Code generation time: {run['codegen_time_sec']:.6f} sec\n")

            f.write("\n")

            # Calculate statistics if we have multiple runs
            if len(sorted_runs) > 1:
                times = [run['codegen_time_sec'] for run in sorted_runs]
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                f.write("  Statistics:\n")
                f.write(f"    Average: {avg_time:.6f} sec\n")
                f.write(f"    Min: {min_time:.6f} sec\n")
                f.write(f"    Max: {max_time:.6f} sec\n")
                f.write("\n")


def save_local_results(output_dir, yaml_name, codegen_time, yaml_config, git_info, test_date, hostname, gpu_arch):
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
        'codegen_time_sec': codegen_time
    }

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
    local_detailed_report = os.path.join(output_dir, 'local-sia3cmp_report.txt')
    try:
        with open(local_detailed_report, 'w') as f:
            f.write("Code Generation Time Detailed Results\n")
            f.write(f"Host: {hostname}, GPU: {gpu_arch}\n")

            # Aggregate results by test name to find latest and best
            test_aggregates = {}
            for test_data in existing_detailed:
                test_name = test_data['test_name']
                if test_name not in test_aggregates:
                    test_aggregates[test_name] = {
                        'latest': test_data,
                        'fastest': test_data,
                        'all_runs': []
                    }
                else:
                    # Update latest if newer
                    if test_data['test_date'] >= test_aggregates[test_name]['latest']['test_date']:
                        test_aggregates[test_name]['latest'] = test_data

                    # Update fastest if faster
                    if test_data['codegen_time_sec'] < test_aggregates[test_name]['fastest']['codegen_time_sec']:
                        test_aggregates[test_name]['fastest'] = test_data

                test_aggregates[test_name]['all_runs'].append(test_data)

            # === SUMMARY TABLE: Latest and Best ===
            f.write("SUMMARY: Latest and Best (Fastest) Code Generation Times\n")
            f.write("-" * 150 + "\n\n")

            f.write(f"{'Test Name':<40} | {'Latest Time (sec)':>18} | {'Latest Date':<20} | {'Best Time (sec)':>18} | {'Best Date':<20}\n")
            f.write("-" * 40 + "+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 20 + "+" + "-" * 22 + "\n")

            for test_name in sorted(test_aggregates.keys()):
                latest = test_aggregates[test_name]['latest']
                fastest = test_aggregates[test_name]['fastest']

                f.write(f"{test_name:<40} | "
                       f"{latest['codegen_time_sec']:>18.6f} | "
                       f"{latest['test_date']:<20} | "
                       f"{fastest['codegen_time_sec']:>18.6f} | "
                       f"{fastest['test_date']:<20}\n")

            f.write("\n\n")

            # === DETAILED TABLE: All Runs ===
            f.write("ALL RUNS: Complete History\n")
            f.write("-" * 150 + "\n\n")

            f.write(f"{'Test Name':<40} | {'Git Hash':<20} | {'Test Date':<20} | {'Codegen Time (sec)':>18}\n")
            f.write("-" * 40 + "+" + "-" * 22 + "+" + "-" * 22 + "+" + "-" * 20 + "\n")

            for test_data in existing_detailed:
                f.write(f"{test_data['test_name']:<40} | "
                       f"{test_data['git_info']:<20} | "
                       f"{test_data['test_date']:<20} | "
                       f"{test_data['codegen_time_sec']:>18.6f}\n")

            f.write("\n")
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

    # Key format: "yaml_name.yaml"
    key = f"{yaml_name}.yaml"

    # Initialize key if not exists
    if key not in summary:
        summary[key] = []

    # Append time
    summary[key].append(codegen_time)

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
            f.write("Code Generation Time Run Summary\n")

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
                    row_parts.append(f"{time_val:<8.4f}")

                f.write(" ".join(row_parts) + "\n")

            f.write("\n")
            f.write("Note: Times in seconds (sec)\n")
    except Exception as e:
        print(f"Warning: Failed to create local report: {e}", file=sys.stderr)

    return True


def main():
    parser = argparse.ArgumentParser(description='Parse codegen-time test results and save locally')
    parser.add_argument('--yappi-file', required=True, help='Path to yappi results file')
    parser.add_argument('--yaml-file', required=True, help='Path to YAML config file')
    parser.add_argument('--output-dir', required=True, help='Output directory (log folder)')
    parser.add_argument('--git-info', required=True, help='Git commit info')
    parser.add_argument('--hostname', required=True, help='Hostname')
    parser.add_argument('--gpu-arch', required=True, help='GPU architecture')
    parser.add_argument('--test-date', required=True, help='Test date')
    parser.add_argument('--yaml-name', required=True, help='YAML test name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Check if yappi file exists
    if not os.path.exists(args.yappi_file):
        print(f"Warning: Yappi file not found: {args.yappi_file}", file=sys.stderr)
        return 0

    # Parse YAML configuration
    yaml_config = parse_yaml_config(args.yaml_file)
    if yaml_config is None:
        print("Error: Failed to parse YAML configuration", file=sys.stderr)
        return 1

    # Parse yappi results
    codegen_time = parse_yappi_file(args.yappi_file)
    if codegen_time is None:
        print("Warning: Failed to extract codegen time from yappi results", file=sys.stderr)
        return 0

    if args.verbose:
        print(f"Code generation time: {codegen_time:.6f} seconds")

    # Save local results to log directory
    if not save_local_results(args.output_dir, args.yaml_name, codegen_time, yaml_config,
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

