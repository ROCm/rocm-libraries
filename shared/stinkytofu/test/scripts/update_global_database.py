#!/usr/bin/env python3

"""
update_global_database.py - Update global database from local.json

This script updates the global database with results from a test run.
The database is keyed by git hash, and stores both best and latest times.
For best time, it only updates if the new score is faster.
For latest time, it always updates with the most recent run.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict


def load_json(file_path: str) -> Dict:
    """Load JSON file, return empty dict if not exists."""
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}


def save_json(file_path: str, data: Dict):
    """Save data to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {file_path}: {e}", file=sys.stderr)
        raise


def update_database(db: Dict, local_data: Dict, git_hash: str) -> int:
    """
    Update database with local results.
    Database structure: {git_hash: {testcase: {best_time, latest_time}}}
    Returns number of records updated.
    """
    updates = 0

    # Ensure git_hash entry exists
    if git_hash not in db:
        db[git_hash] = {}

    # Update each test case
    for test_case, times in local_data.items():
        if not times:
            continue

        # Get best (minimum) and latest time from this run
        best_time = min(times)
        latest_time = times[-1] if isinstance(times, list) else best_time

        # Update or create test case entry
        if test_case not in db[git_hash]:
            # New test case: initialize with both best and latest time
            db[git_hash][test_case] = {
                'best_time': best_time,
                'latest_time': latest_time
            }
            updates += 1
        else:
            # Existing test case: handle both dict and old scalar format
            if isinstance(db[git_hash][test_case], dict):
                # New format: update both best and latest
                existing_best = db[git_hash][test_case].get('best_time', float('inf'))
                db[git_hash][test_case]['latest_time'] = latest_time

                if best_time < existing_best:
                    db[git_hash][test_case]['best_time'] = best_time
            else:
                # Old scalar format: convert to dict and update
                existing_best = db[git_hash][test_case]
                db[git_hash][test_case] = {
                    'best_time': min(best_time, existing_best),
                    'latest_time': latest_time
                }

            updates += 1

    return updates


def main():
    parser = argparse.ArgumentParser(description='Update global database from local.json')
    parser.add_argument('--local-json', required=True, help='Path to local.json file')
    parser.add_argument('--database-file', required=True, help='Path to global database JSON file')
    parser.add_argument('--git-hash', required=True, help='Git commit hash for this run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load local results
    if args.verbose:
        print(f"Loading local results from: {args.local_json}")

    local_data = load_json(args.local_json)

    if not local_data:
        print(f"Warning: No data found in {args.local_json}", file=sys.stderr)
        return 0

    if args.verbose:
        print(f"Found {len(local_data)} test case(s) in local results")

    # Load global database
    if args.verbose:
        print(f"Loading global database from: {args.database_file}")

    db = load_json(args.database_file)

    # Update database
    updates = update_database(db, local_data, args.git_hash)

    # Save database
    if args.verbose:
        print(f"Saving updated database to: {args.database_file}")

    save_json(args.database_file, db)

    if args.verbose:
        print(f"✓ Updated {updates} record(s) in database for git hash: {args.git_hash}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

