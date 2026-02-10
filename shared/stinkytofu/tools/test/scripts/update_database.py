#!/usr/bin/env python3

"""
update_database.py - Update database from local results with best and latest scores

This script reads local.json files from log folders and updates the database
with both the best (minimum) and latest scores for each test case and git commit.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def load_json(file_path: str) -> any:
    """Load JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None


def save_json(file_path: str, data: any) -> bool:
    """Save JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}", file=sys.stderr)
        return False


def create_problem_key(problem_sizes: str, yaml_config: Dict) -> str:
    """Create a unique key for a problem configuration."""
    return json.dumps({
        'problem_sizes': problem_sizes,
        'DataType': yaml_config.get('DataType', 'Unknown'),
        'DestDataType': yaml_config.get('DestDataType', 'Unknown'),
        'ComputeDataType': yaml_config.get('ComputeDataType', 'Unknown'),
        'TransposeA': yaml_config.get('TransposeA', 'Unknown'),
        'TransposeB': yaml_config.get('TransposeB', 'Unknown'),
        'RotatingBufferSize': yaml_config.get('RotatingBufferSize', 0),
    }, sort_keys=True)


def update_exe_time_database(db: Dict, local_data_list: List[Dict]) -> int:
    """
    Update exe_time database from local results.
    Returns number of records updated.
    """
    updates = 0

    for local_data in local_data_list:
        yaml_config = local_data.get('config', {})
        git_info = local_data.get('git_info', 'unknown')
        test_date = local_data.get('test_date', '')
        hostname = local_data.get('hostname', 'unknown')
        gpu_arch = local_data.get('gpu_arch', 'unknown')

        # Group results by (problem_sizes, sia_version) and keep best time
        best_results = {}
        for result in local_data.get('results', []):
            if result.get('validation') != 'PASSED':
                continue

            try:
                time_us = float(result.get('time_us', float('inf')))
            except (ValueError, TypeError):
                continue

            key = (result['problem_sizes'], result['sia_version'])
            if key not in best_results or time_us < float(best_results[key]['time_us']):
                best_results[key] = result

        # Update database with best results
        for (problem_sizes, sia_version), result in best_results.items():
            problem_key = create_problem_key(problem_sizes, yaml_config)

            # Initialize problem entry if not exists
            if problem_key not in db:
                db[problem_key] = {
                    'problem_config': {
                        'problem_sizes': problem_sizes,
                        'DataType': yaml_config.get('DataType', 'Unknown'),
                        'DestDataType': yaml_config.get('DestDataType', 'Unknown'),
                        'ComputeDataType': yaml_config.get('ComputeDataType', 'Unknown'),
                        'TransposeA': yaml_config.get('TransposeA', 'Unknown'),
                        'TransposeB': yaml_config.get('TransposeB', 'Unknown'),
                        'RotatingBufferSize': yaml_config.get('RotatingBufferSize', 0),
                    },
                    'runs': {}
                }

            # Create run key
            run_key = f"{git_info}_{test_date}_{sia_version}"

            # Check if this run already exists
            new_time = float(result['time_us'])
            if run_key in db[problem_key]['runs']:
                existing_time = float(db[problem_key]['runs'][run_key]['best_time_us'])

                # Update with latest time and best time
                db[problem_key]['runs'][run_key]['latest_time_us'] = result['time_us']
                db[problem_key]['runs'][run_key]['latest_gflops'] = result['gflops']

                # Update best time if new time is better
                if new_time < existing_time:
                    db[problem_key]['runs'][run_key]['best_time_us'] = result['time_us']
                    db[problem_key]['runs'][run_key]['best_gflops'] = result['gflops']
            else:
                # Add new run data with both best and latest time (initially the same)
                db[problem_key]['runs'][run_key] = {
                    'git_info': git_info,
                    'test_date': test_date,
                    'sia_version': sia_version,
                    'hostname': hostname,
                    'gpu_arch': gpu_arch,
                    'solution': result['solution'],
                    'validation': result['validation'],
                    'best_time_us': result['time_us'],
                    'best_gflops': result['gflops'],
                    'latest_time_us': result['time_us'],
                    'latest_gflops': result['gflops'],
                }

            updates += 1

    return updates


def update_codegen_time_database(db: Dict, local_data_list: List[Dict]) -> int:
    """
    Update codegen_time database from local results.
    Returns number of records updated.
    """
    updates = 0

    # Group by test_name and git_info, keep best (minimum) time
    best_times = {}

    for local_data in local_data_list:
        test_name = local_data.get('test_name', 'unknown')
        git_info = local_data.get('git_info', 'unknown')
        test_date = local_data.get('test_date', '')
        hostname = local_data.get('hostname', 'unknown')
        gpu_arch = local_data.get('gpu_arch', 'unknown')
        yaml_config = local_data.get('config', {})

        try:
            codegen_time = float(local_data.get('codegen_time_sec', float('inf')))
        except (ValueError, TypeError):
            continue

        key = (test_name, git_info)
        if key not in best_times or codegen_time < best_times[key]['codegen_time_sec']:
            best_times[key] = {
                'test_name': test_name,
                'config': yaml_config,
                'git_info': git_info,
                'test_date': test_date,
                'hostname': hostname,
                'gpu_arch': gpu_arch,
                'codegen_time_sec': codegen_time
            }

    # Update database with best times
    for (test_name, git_info), data in best_times.items():
        # Initialize test entry if not exists
        if test_name not in db:
            db[test_name] = {
                'test_name': test_name,
                'config': data['config'],
                'runs': []
            }

        # Check if this git_info already exists in runs
        existing_run = None
        for i, run in enumerate(db[test_name]['runs']):
            if run['git_info'] == git_info:
                existing_run = i
                break

        if existing_run is not None:
            # Always update latest time
            db[test_name]['runs'][existing_run]['latest_codegen_time_sec'] = data['codegen_time_sec']
            db[test_name]['runs'][existing_run]['test_date'] = data['test_date']

            # Update best time only if new time is better
            if data['codegen_time_sec'] < db[test_name]['runs'][existing_run].get('best_codegen_time_sec', float('inf')):
                db[test_name]['runs'][existing_run]['best_codegen_time_sec'] = data['codegen_time_sec']

            # Ensure best_codegen_time_sec exists (for backward compatibility)
            if 'best_codegen_time_sec' not in db[test_name]['runs'][existing_run]:
                db[test_name]['runs'][existing_run]['best_codegen_time_sec'] = db[test_name]['runs'][existing_run].get('codegen_time_sec', data['codegen_time_sec'])

            updates += 1
        else:
            # Add new run with both best and latest time (initially the same)
            db[test_name]['runs'].append({
                'git_info': git_info,
                'test_date': data['test_date'],
                'hostname': data['hostname'],
                'gpu_arch': data['gpu_arch'],
                'best_codegen_time_sec': data['codegen_time_sec'],
                'latest_codegen_time_sec': data['codegen_time_sec'],
            })
            updates += 1

        # Keep only last 100 runs per test
        if len(db[test_name]['runs']) > 100:
            db[test_name]['runs'] = sorted(db[test_name]['runs'],
                                           key=lambda x: x['test_date'],
                                           reverse=True)[:100]

    return updates


def create_exe_time_report(db: Dict, output_file: str):
    """Create human-readable comparison report for exe_time."""
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("Execution Time Database Report\n")
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
                    f.write(f"      Validation: {run_data['validation']}\n")
                    f.write(f"      Best Time: {run_data.get('best_time_us', run_data.get('time_us', 'N/A'))} us, "
                           f"GFLOPS: {run_data.get('best_gflops', run_data.get('gflops', 'N/A'))}\n")
                    f.write(f"      Latest Time: {run_data.get('latest_time_us', run_data.get('time_us', 'N/A'))} us, "
                           f"GFLOPS: {run_data.get('latest_gflops', run_data.get('gflops', 'N/A'))}\n")

                f.write("\n")

            f.write("\n")


def create_codegen_time_report(db: Dict, output_file: str):
    """Create human-readable comparison report for codegen_time."""
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Code Generation Time Database Report\n")
        f.write("=" * 100 + "\n\n")

        for test_key, test_data in sorted(db.items()):
            f.write("-" * 100 + "\n")
            f.write(f"Test: {test_data['test_name']}\n")

            config = test_data['config']
            f.write(f"  DataType: {config.get('DataType', '?')}, "
                   f"DestDataType: {config.get('DestDataType', '?')}, "
                   f"ComputeDataType: {config.get('ComputeDataType', '?')}\n")
            f.write(f"  TransposeA: {config.get('TransposeA', '?')}, "
                   f"TransposeB: {config.get('TransposeB', '?')}\n")
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
                best_time = run.get('best_codegen_time_sec', run.get('codegen_time_sec', 0))
                latest_time = run.get('latest_codegen_time_sec', run.get('codegen_time_sec', 0))
                f.write(f"      Best code generation time: {best_time:.6f} sec\n")
                f.write(f"      Latest code generation time: {latest_time:.6f} sec\n")

            f.write("\n")

            # Calculate statistics if we have multiple runs
            if len(sorted_runs) > 1:
                best_times = [run.get('best_codegen_time_sec', run.get('codegen_time_sec', 0)) for run in sorted_runs]
                latest_times = [run.get('latest_codegen_time_sec', run.get('codegen_time_sec', 0)) for run in sorted_runs]

                f.write("  Statistics (Best Times):\n")
                f.write(f"    Average: {sum(best_times) / len(best_times):.6f} sec\n")
                f.write(f"    Min: {min(best_times):.6f} sec\n")
                f.write(f"    Max: {max(best_times):.6f} sec\n")

                f.write("  Statistics (Latest Times):\n")
                f.write(f"    Average: {sum(latest_times) / len(latest_times):.6f} sec\n")
                f.write(f"    Min: {min(latest_times):.6f} sec\n")
                f.write(f"    Max: {max(latest_times):.6f} sec\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Update database from local results with best and latest scores'
    )
    parser.add_argument('--logs-folder', required=True,
                       help='Path to log folder containing local.json files')
    parser.add_argument('--database-dir', required=True,
                       help='Path to database directory')
    parser.add_argument('--hostname', required=True, help='Hostname')
    parser.add_argument('--gpu-arch', required=True, help='GPU architecture')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Validate folders exist
    if not os.path.exists(args.logs_folder):
        print(f"Error: Logs folder not found: {args.logs_folder}", file=sys.stderr)
        return 1

    os.makedirs(args.database_dir, exist_ok=True)

    # Process exe_time
    exe_time_dir = Path(args.logs_folder) / "exe_time"
    if exe_time_dir.exists():
        local_json = exe_time_dir / "local-sia3cmp.json"
        if local_json.exists():
            print(f"Processing exe_time: {local_json}")

            # Load local results
            local_data = load_json(str(local_json))
            if local_data is None:
                print("Error: Failed to load exe_time local results", file=sys.stderr)
            else:
                # Ensure it's a list
                if not isinstance(local_data, list):
                    local_data = [local_data]

                # Load database with sia3cmp prefix
                db_file = os.path.join(args.database_dir,
                                      f"exe_time-sia3cmp-{args.hostname}_{args.gpu_arch}.json")
                db = load_json(db_file) if os.path.exists(db_file) else {}

                # Update database with best scores
                updates = update_exe_time_database(db, local_data)

                # Save database
                if save_json(db_file, db):
                    print(f"✓ Updated exe_time database: {updates} record(s)")

                    # Create report with sia3cmp prefix
                    report_file = os.path.join(args.database_dir,
                                              f"exe_time-sia3cmp-{args.hostname}_{args.gpu_arch}_report.txt")
                    create_exe_time_report(db, report_file)
                    if args.verbose:
                        print(f"✓ Created report: {report_file}")
                else:
                    print("Error: Failed to save exe_time database", file=sys.stderr)
        else:
            print(f"No exe_time local-sia3cmp.json found in {exe_time_dir}")

    # Process codegen_time
    codegen_time_dir = Path(args.logs_folder) / "codegen_time"
    if codegen_time_dir.exists():
        local_json = codegen_time_dir / "local-sia3cmp.json"
        if local_json.exists():
            print(f"Processing codegen_time: {local_json}")

            # Load local results
            local_data = load_json(str(local_json))
            if local_data is None:
                print("Error: Failed to load codegen_time local results", file=sys.stderr)
            else:
                # Ensure it's a list
                if not isinstance(local_data, list):
                    local_data = [local_data]

                # Load database with sia3cmp prefix
                db_file = os.path.join(args.database_dir,
                                      f"codegen_time-sia3cmp-{args.hostname}_{args.gpu_arch}.json")
                db = load_json(db_file) if os.path.exists(db_file) else {}

                # Update database with best scores
                updates = update_codegen_time_database(db, local_data)

                # Save database
                if save_json(db_file, db):
                    print(f"✓ Updated codegen_time database: {updates} record(s)")

                    # Create report with sia3cmp prefix
                    report_file = os.path.join(args.database_dir,
                                              f"codegen_time-sia3cmp-{args.hostname}_{args.gpu_arch}_report.txt")
                    create_codegen_time_report(db, report_file)
                    if args.verbose:
                        print(f"✓ Created report: {report_file}")
                else:
                    print("Error: Failed to save codegen_time database", file=sys.stderr)
        else:
            print(f"No codegen_time local-sia3cmp.json found in {codegen_time_dir}")

    print("\nDatabase update completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

