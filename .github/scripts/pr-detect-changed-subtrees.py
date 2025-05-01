"""
PR Detect Changed Subtrees Script
------------------
This script analyzes a pull request's changed files and determines which subtrees
(defined in .github/repos-config.json by category/name) were affected.

Steps:
    1. Fetch the changed files in the PR using the GitHub API.
    2. Load the subtree mapping from repos-config.json.
    3. Match changed paths against known category/name prefixes.
    4. Emit a comma-separated list of changed subtrees to GITHUB_OUTPUT as 'subtrees'.

Arguments:
    --token   : GitHub token for authentication
    --repo    : Full repository name (e.g., org/repo)
    --pr      : Pull request number
    --config  : Path to the repos-config.json file
    --dry-run : Print outputs instead of writing to GITHUB_OUTPUT
    --debug   : Enable debug logging

Outputs:
    Writes 'subtrees' key to the GitHub Actions $GITHUB_OUTPUT file, which
    the workflow reads to call the subsequent python script to create/update PRs.

Usage Examples:
    # Normal run inside GitHub Actions
    python pr-detect-changed-subtrees.py --token ${{ secrets.GITHUB_TOKEN }} --repo ROCm/rocm-libraries --pr 123

    # Dry run locally to see what subtrees would be reported
    python pr-detect-changed-subtrees.py --token fake --repo ROCm/rocm-libraries --pr 123 --dry-run

    # Dry run with verbose logging
    python pr-detect-changed-subtrees.py --token fake --repo ROCm/rocm-libraries --pr 123 --dry-run --debug
"""

import argparse
import requests
import sys
import os
import json
from pathlib import Path

def get_paginated_results(url, headers, debug=False):
    results = []
    while url:
        try:
            if debug:
                print(f"Fetching: {url}")
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            results.extend(r.json())
            url = None
            if 'link' in r.headers:
                for part in r.headers['link'].split(','):
                    if 'rel="next"' in part:
                        url = part.split(';')[0].strip()[1:-1]
                        break
        except requests.exceptions.RequestException as e:
            print(f"Error while fetching data: {e}", file=sys.stderr)
            sys.exit(1)
    return results

parser = argparse.ArgumentParser(description="Detect changed subtrees in a PR.")
parser.add_argument("--token", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--pr", required=True)
parser.add_argument("--config", default=".github/repos-config.json")
parser.add_argument("--dry-run", action="store_true", help="Print outputs instead of writing to GITHUB_OUTPUT")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

headers = {
    "Authorization": f"token {args.token}",
    "Accept": "application/vnd.github.v3+json"
}

if args.debug:
    print(f"Fetching changed files from PR #{args.pr} in {args.repo}...")

files_url = f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}/files"
changed_files = get_paginated_results(files_url, headers, debug=args.debug)
file_paths = [file["filename"] for file in changed_files]

if args.debug:
    print("Changed files:")
    for path in file_paths:
        print(f"  {path}")

with open(args.config) as f:
    config = json.load(f)["repositories"]

valid_prefixes = {f"{entry['category']}/{entry['name']}" for entry in config}
if args.debug:
    print("Valid subtrees from config:")
    for prefix in sorted(valid_prefixes):
        print(f"  {prefix}")

# Identify changed subtrees
subtrees = {
    "/".join(path.split("/", 2)[:2])
    for path in file_paths
    if len(path.split("/")) >= 2
}

matched = sorted(prefix.split("/", 1)[1] for prefix in (subtrees & valid_prefixes))

if args.debug:
    print(f"Matched subtrees: {matched}")

# Output results
if args.dry_run:
    print(f"[Dry-run] Would output: subtrees={','.join(matched)}")
else:
    output_file = os.environ.get('GITHUB_OUTPUT')
    if output_file:
        with open(output_file, 'a') as f:
            print(f"subtrees={','.join(matched)}", file=f)
        if args.debug:
            print(f"Wrote to GITHUB_OUTPUT: subtrees={','.join(matched)}")
    else:
        print("GITHUB_OUTPUT environment variable not set. Outputs cannot be written.", file=sys.stderr)
        sys.exit(1)
