#!/usr/bin/env python3

"""
update_sia3cmp_database.py - Update sia3cmp database with latest and fastest results

This script reads local-sia3cmp.json and updates a persistent database that tracks
the best (fastest) and latest results for stinkytofu vs SIA3 comparisons per git hash.
"""

import argparse
import json
import os
import sys
from datetime import datetime


def load_json_file(json_file):
    """Load JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        return data
    except Exception as e:
        print(f"Error loading {json_file}: {e}", file=sys.stderr)
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
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        with open(db_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving database: {e}", file=sys.stderr)
        return False
    return True


def parse_time_us(time_str):
    """Parse time_us field which may be string or number."""
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
        cleaned = ''.join(c for c in str(time_str) if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else float('inf')
    except:
        return float('inf')


def parse_gflops(gflops_str):
    """Parse gflops field which may be string or number."""
    try:
        if isinstance(gflops_str, (int, float)):
            return float(gflops_str)
        cleaned = ''.join(c for c in str(gflops_str) if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0


def create_problem_key(problem_sizes, config):
    """Create a unique key for a problem configuration."""
    return json.dumps({
        'problem_sizes': problem_sizes,
        'DataType': config.get('DataType', 'Unknown'),
        'DestDataType': config.get('DestDataType', 'Unknown'),
        'ComputeDataType': config.get('ComputeDataType', 'Unknown'),
        'TransposeA': config.get('TransposeA', 'Unknown'),
        'TransposeB': config.get('TransposeB', 'Unknown'),
        'RotatingBufferSize': config.get('RotatingBufferSize', 0),
    }, sort_keys=True)


def update_database(db, local_data, git_hash, verbose=False):
    """Update database with new results from local-sia3cmp.json."""

    for test_run in local_data:
        test_name = test_run.get('test_name', '')
        test_date = test_run.get('test_date', '')
        hostname = test_run.get('hostname', '')
        gpu_arch = test_run.get('gpu_arch', '')
        config = test_run.get('config', {})

        # Group results by problem_sizes and sia_version
        problem_groups = {}
        for result in test_run.get('results', []):
            problem_sizes = result.get('problem_sizes', '')
            sia_version = result.get('sia_version', 'Unknown')

            # Normalize sia_version
            if sia_version.lower() in ['sia1', 'stinkytofu']:
                sia_version = 'stinkytofu'

            key = (problem_sizes, sia_version)
            if key not in problem_groups:
                problem_groups[key] = []

            problem_groups[key].append({
                'test_date': test_date,
                'solution': result.get('solution', ''),
                'validation': result.get('validation', ''),
                'time_us': parse_time_us(result.get('time_us', 'inf')),
                'gflops': parse_gflops(result.get('gflops', '0'))
            })

        # Update database for each problem
        for (problem_sizes, sia_version), results in problem_groups.items():
            problem_key = create_problem_key(problem_sizes, config)

            # Initialize problem entry if not exists
            if problem_key not in db:
                db[problem_key] = {
                    'problem_config': {
                        'problem_sizes': problem_sizes,
                        'DataType': config.get('DataType', 'Unknown'),
                        'DestDataType': config.get('DestDataType', 'Unknown'),
                        'ComputeDataType': config.get('ComputeDataType', 'Unknown'),
                        'TransposeA': config.get('TransposeA', 'Unknown'),
                        'TransposeB': config.get('TransposeB', 'Unknown'),
                        'RotatingBufferSize': config.get('RotatingBufferSize', 0),
                    },
                    'git_hashes': {}
                }

            # Initialize git hash entry if not exists
            if git_hash not in db[problem_key]['git_hashes']:
                db[problem_key]['git_hashes'][git_hash] = {}

            # Initialize sia_version entry if not exists
            if sia_version not in db[problem_key]['git_hashes'][git_hash]:
                db[problem_key]['git_hashes'][git_hash][sia_version] = {
                    'fastest': None,
                    'latest': None
                }

            sia_data = db[problem_key]['git_hashes'][git_hash][sia_version]

            # Find fastest and latest from current results
            fastest_result = min(results, key=lambda x: x['time_us'])
            latest_result = max(results, key=lambda x: x['test_date'])

            # Update fastest if this is faster or first time
            if sia_data['fastest'] is None or fastest_result['time_us'] < sia_data['fastest']['time_us']:
                sia_data['fastest'] = {
                    'test_name': test_name,
                    'test_date': fastest_result['test_date'],
                    'hostname': hostname,
                    'gpu_arch': gpu_arch,
                    'solution': fastest_result['solution'],
                    'validation': fastest_result['validation'],
                    'time_us': fastest_result['time_us'],
                    'gflops': fastest_result['gflops']
                }
                if verbose:
                    print(f"Updated fastest for {problem_sizes} ({sia_version}, {git_hash}): {fastest_result['time_us']:.2f} us")

            # Update latest if this is more recent or first time
            if sia_data['latest'] is None or latest_result['test_date'] > sia_data['latest']['test_date']:
                sia_data['latest'] = {
                    'test_name': test_name,
                    'test_date': latest_result['test_date'],
                    'hostname': hostname,
                    'gpu_arch': gpu_arch,
                    'solution': latest_result['solution'],
                    'validation': latest_result['validation'],
                    'time_us': latest_result['time_us'],
                    'gflops': latest_result['gflops']
                }
                if verbose:
                    print(f"Updated latest for {problem_sizes} ({sia_version}, {git_hash}): {latest_result['test_date']}")


def main():
    parser = argparse.ArgumentParser(
        description='Update sia3cmp database with latest and fastest results'
    )
    parser.add_argument(
        '--local-json',
        required=True,
        help='Input local-sia3cmp.json file'
    )
    parser.add_argument(
        '--database-file',
        required=True,
        help='Database file to update (e.g., sia3cmp-hostname_gfx950.json)'
    )
    parser.add_argument(
        '--git-hash',
        required=True,
        help='Git hash for this test run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Load local data
    if args.verbose:
        print(f"Loading local data from {args.local_json}...")

    local_data = load_json_file(args.local_json)
    if local_data is None:
        return 1

    if args.verbose:
        print(f"Loaded {len(local_data)} test run(s)")

    # Load existing database
    if args.verbose:
        print(f"Loading database from {args.database_file}...")

    db = load_database(args.database_file)

    if args.verbose:
        print(f"Database has {len(db)} problem(s)")

    # Update database
    if args.verbose:
        print(f"Updating database for git hash: {args.git_hash}...")

    update_database(db, local_data, args.git_hash, args.verbose)

    # Save database
    if args.verbose:
        print(f"Saving database to {args.database_file}...")

    if save_database(args.database_file, db):
        print(f"Successfully updated database: {args.database_file}")
        return 0
    else:
        print("Error saving database", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

