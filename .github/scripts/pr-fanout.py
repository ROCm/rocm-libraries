"""
PR Fanout Script
------------------
This script takes a list of changed subtrees and for each:
    - Splits the corresponding subtree directory from the monorepo using `git subtree split`.
    - Force pushes the split branch to the corresponding sub-repo.
    - Creates or updates a pull request in the sub-repo with a standardized branch and label.

Arguments:
    --token     : GitHub token for API and git authentication
    --repo      : Full name of the monorepo
    --pr        : The monorepo pull request number
    --config  : Path to the repos-config.json file
    --subtrees  : A comma-separated list of subtree folder names (from detect_changed_subtrees)
    --dry-run   : Print actions instead of executing them
    --debug     : Enable debug logging

Usage Examples:
    # Normal usage in GitHub Actions
    python pr-fanout.py --token ${{ secrets.GITHUB_TOKEN }} --repo ROCm/rocm-libraries --pr 123 --subtrees rocBLAS,hipBLASLt,rocSPARSE

    # Dry run to test logic without pushing or editing PRs
    python pr-fanout.py --token fake --repo ROCm/rocm-libraries --pr 123 --subtrees rocBLAS,hipBLASLt,rocSPARSE --dry-run --debug
"""

import argparse
import subprocess
import json

parser = argparse.ArgumentParser(description="Fanout monorepo PR to sub-repos.")
parser.add_argument("--token", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--pr", required=True)
parser.add_argument("--subtrees", required=True)
parser.add_argument("--config", default=".github/repos-config.json")
parser.add_argument("--dry-run", action="store_true", help="Print actions instead of executing them")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

with open(".github/repos-config.json") as f:
    config = json.load(f)["repositories"]

subtrees = args.subtrees.split(',')

for entry in config:
    if entry["name"] not in subtrees:
        continue

    branch = f"monorepo-pr-{args.pr}-{entry['name']}"
    prefix = f"{entry['category']}/{entry['name']}"
    remote = f"https://x-access-token:{args.token}@github.com/{entry['url']}.git"

    pr_title = f"[Fanout] Sync monorepo PR #{args.pr} to {entry['name']}"
    pr_body = f"This is an automated PR for subtree `{entry['name']}` from monorepo PR #{args.pr}."

    if args.debug:
        print(f"\nProcessing subtree: {entry['name']}")
        print(f"  Prefix: {prefix}")
        print(f"  Branch: {branch}")
        print(f"  Remote: {remote}")
        print(f"  PR title: {pr_title}")

    # Split the subtree
    split_cmd = ["git", "subtree", "split", "--prefix", prefix, "-b", branch]
    if args.dry_run or args.debug:
        print(f"Running: {' '.join(split_cmd)}")
    if not args.dry_run:
        subprocess.run(split_cmd, check=True)

    # Push the split branch
    push_cmd = ["git", "push", remote, f"{branch}:refs/heads/{branch}", "--force"]
    if args.dry_run or args.debug:
        print(f"Running: {' '.join(push_cmd)}")
    if not args.dry_run:
        subprocess.run(push_cmd, check=True)

    # Check if PR already exists
    view_cmd = [
        "gh", "pr", "view",
        "--json", "number",
        "--repo", entry["url"],
        "--head", branch
    ]
    if args.dry_run or args.debug:
        print(f"Checking for existing PR with: {' '.join(view_cmd)}")

    result = subprocess.run(view_cmd, capture_output=True, text=True)
    pr_exists = result.returncode == 0

    if not pr_exists:
        create_cmd = [
            "gh", "pr", "create",
            "--repo", entry["url"],
            "--base", entry["branch"],
            "--head", branch,
            "--title", pr_title,
            "--body", pr_body,
            "--label", "auto-fanout"
        ]
        if args.dry_run or args.debug:
            print(f"Creating PR with: {' '.join(create_cmd)}")
        if not args.dry_run:
            subprocess.run(create_cmd, check=True)
    else:
        edit_cmd = [
            "gh", "pr", "edit",
            "--repo", entry["url"],
            "--head", branch,
            "--title", pr_title,
            "--body", pr_body
        ]
        if args.dry_run or args.debug:
            print(f"Updating existing PR with: {' '.join(edit_cmd)}")
        if not args.dry_run:
            subprocess.run(edit_cmd, check=True)
