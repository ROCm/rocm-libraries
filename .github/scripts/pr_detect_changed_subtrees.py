#!/usr/bin/env python3

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
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Outputs:
    Writes 'subtrees' key to the GitHub Actions $GITHUB_OUTPUT file, which
    the workflow reads to call the subsequent python script to create/update PRs.
    The output is a new-line separated list of subtrees.

Example Usage:

    To run in debug mode and perform a dry-run (no changes made):
    python pr_detect_changed_subtrees.py --repo ROCm/rocm-libraries --pr 123 --dry-run

    To run in debug mode and get the changed subtrees output:
    python pr_detect_changed_subtrees.py --repo ROCm/rocm-libraries --pr 123 --debug
"""

import argparse
import sys
import os
import logging
from typing import List, Optional, Set
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect changed subtrees in a PR.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to GITHUB_OUTPUT.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)

def get_valid_prefixes(config: List[RepoEntry]) -> Set[str]:
    """Extract valid subtree prefixes from the configuration."""
    valid_prefixes = {f"{entry.category}/{entry.name}" for entry in config}
    logger.debug("Valid subtrees:\n" + "\n".join(sorted(valid_prefixes)))
    return valid_prefixes

def find_matched_subtrees(changed_files: List[str], valid_prefixes: Set[str]) -> List[str]:
    """Find subtrees that match the changed files."""
    changed_subtrees = {
        "/".join(path.split("/", 2)[:2])
        for path in changed_files
        if len(path.split("/")) >= 2
    }
    matched = sorted(prefix.split("/", 1)[1] for prefix in (changed_subtrees & valid_prefixes))
    logger.debug(f"Matched subtrees: {matched}")
    return matched

def output_subtrees(matched_subtrees: List[str], dry_run: bool) -> None:
    """Output the matched subtrees to GITHUB_OUTPUT or log them in dry-run mode."""
    newline_separated = "\n".join(matched_subtrees)
    if dry_run:
        logger.info(f"[Dry-run] Would output:\n{newline_separated}")
    else:
        output_file = os.environ.get('GITHUB_OUTPUT')
        if output_file:
            with open(output_file, 'a') as f:
                print(f"subtrees<<EOF\n{newline_separated}\nEOF", file=f)
            logger.info(f"Wrote to GITHUB_OUTPUT: subtrees=<newline-separated values>")
        else:
            logger.error("GITHUB_OUTPUT environment variable not set. Outputs cannot be written.")
            sys.exit(1)

def main(argv=None) -> None:
    """Main function to determine changed subtrees in PR."""
    args = parse_arguments(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    changed_files = [file for file in client.get_changed_files(args.repo, int(args.pr))]
    valid_prefixes = get_valid_prefixes(config)
    matched_subtrees = find_matched_subtrees(changed_files, valid_prefixes)
    output_subtrees(matched_subtrees, args.dry_run)

if __name__ == "__main__":
    main()