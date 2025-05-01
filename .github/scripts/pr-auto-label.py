"""
PR Auto Label Script
------------------
This script analyzes the file paths changed in a pull request and determines which
labels should be added or removed.

Arguments:
    --token     : GitHub token for authentication
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Outputs:
    Writes 'add' and 'remove' keys to the GitHub Actions $GITHUB_OUTPUT file, which
    the workflow reads to apply label changes using the GitHub CLI.

Example Usage:

    To run in debug mode and perform a dry-run (no changes made):
        python pr_auto_label.py --token <your-token> --repo <your-repo> --pr <pr-number> --dry-run --debug

    To run in debug mode and apply label changes:
        python pr_auto_label.py --token <your-token> --repo <your-repo> --pr <pr-number> --debug
"""

import argparse
import requests
import sys
import os
from pathlib import Path

def get_paginated_results(url, headers, debug=False):
    results = []
    while url:
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            results.extend(r.json())
            url = None
            if 'link' in r.headers:
                for part in r.headers['link'].split(','):
                    if 'rel="next"' in part:
                        url = part.split(';')[0].strip()[1:-1]
                        break
            if debug:
                print(f"Fetched {len(r.json())} items from {url}")
        except requests.exceptions.RequestException as e:
            print(f"Error while fetching data: {e}", file=sys.stderr)
            sys.exit(1)
    return results

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--pr", required=True)
parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
args = parser.parse_args()

headers = {
    "Authorization": f"token {args.token}",
    "Accept": "application/vnd.github.v3+json"
}

# Debug log start
if args.debug:
    print(f"Debug mode enabled. Fetching changed files for PR {args.pr}...")

# Get changed files
files_url = f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}/files"
changed_files = get_paginated_results(files_url, headers, debug=args.debug)
file_paths = [file["filename"] for file in changed_files]

if args.debug:
    print(f"Changed files: {file_paths}")

# Get existing labels on the PR
pr_url = f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}"
try:
    pr_data = requests.get(pr_url, headers=headers).json()
    existing_labels = set(label["name"] for label in pr_data["labels"])
    if args.debug:
        print(f"Existing labels on PR: {existing_labels}")
except requests.exceptions.RequestException as e:
    print(f"Error while fetching PR data: {e}", file=sys.stderr)
    sys.exit(1)

# Determine the desired labels based on changed files
desired_labels = set()
for path in file_paths:
    parts = Path(path).parts
    if len(parts) >= 2:
        if parts[0] == "projects":
            desired_labels.add(f"project: {parts[1]}")
        elif parts[0] == "shared":
            desired_labels.add(f"shared: {parts[1]}")

if args.debug:
    print(f"Desired labels based on changes: {desired_labels}")

# Filter out the existing auto labels (project and shared)
existing_auto_labels = {
    label for label in existing_labels
    if label.startswith("project: ") or label.startswith("shared: ")
}

# Determine which labels need to be added or removed
to_add = sorted(desired_labels - existing_labels)
to_remove = sorted(existing_auto_labels - desired_labels)

if args.debug:
    print(f"Labels to add: {to_add}")
    print(f"Labels to remove: {to_remove}")

# Output the results to GitHub Actions via GITHUB_OUTPUT
output_file = os.environ.get('GITHUB_OUTPUT')
if output_file:
    with open(output_file, 'a') as f:
        print(f"add={','.join(to_add)}", file=f)
        print(f"remove={','.join(to_remove)}", file=f)
else:
    print("GITHUB_OUTPUT environment variable not set. Outputs cannot be written.")
    sys.exit(1)

# If dry-run is enabled, prevent actual changes to labels
if args.dry_run:
    print("Dry run enabled. Labels will not be applied.")
else:
    # Here you can add code to apply the labels (e.g., using `gh` CLI) if not in dry-run mode
    if to_add:
        print(f"Would add labels: {', '.join(to_add)}")
    if to_remove:
        print(f"Would remove labels: {', '.join(to_remove)}")
