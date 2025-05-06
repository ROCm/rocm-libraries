#!/usr/bin/env python3

"""
PR Fanout Script
------------------
This script takes a list of changed subtrees and for each:
    - Splits the corresponding subtree directory from the monorepo using `git subtree split`.
    - Force pushes the split branch to the corresponding sub-repo.
    - Creates or updates a pull request in the sub-repo with a standardized branch and label.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --subtrees  : A newline-separated list of subtree folder names (from detect_changed_subtrees.py)
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:

    To run in debug mode and perform a dry-run (no changes made):
    python pr-fanout.py --repo ROCm/rocm-libraries --pr 123 --subtrees "$(printf 'rocBLAS\nhipBLASLt\nrocSPARSE')" --dry-run --debug
"""

import argparse
import subprocess
import logging
from typing import List, Optional
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fanout monorepo PR to sub-repos.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, help="Pull request number")
    parser.add_argument("--subtrees", required=True)
    parser.add_argument("--config", required=False, default=".github/repos-config.json")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def get_subtree_info(config: List[RepoEntry], subtrees: List[str]) -> List[RepoEntry]:
    """Filter and return relevant subtree info from the config."""
    return [entry for entry in config if entry.name in subtrees]

def split_and_push_subtree(entry: RepoEntry, branch: str, prefix: str, remote: str, dry_run: bool) -> None:
    """Split the subtree and push it to the corresponding sub-repo."""
    split_cmd = ["git", "subtree", "split", "--prefix", prefix, "-b", branch]
    logger.debug(f"Running: {' '.join(split_cmd)}")
    if not dry_run:
        subprocess.run(split_cmd, check=True)
    push_cmd = ["git", "push", remote, f"{branch}:refs/heads/{branch}", "--force"]
    logger.debug(f"Running: {' '.join(push_cmd)}")
    if not dry_run:
        subprocess.run(push_cmd, check=True)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to execute the PR fanout logic."""
    args = parse_arguments(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    subtrees = [line.strip() for line in args.subtrees.splitlines() if line.strip()]
    relevant_subtrees = get_subtree_info(config, subtrees)

    for entry in relevant_subtrees:
        branch = f"monorepo-pr-{args.pr}-{entry.name}"
        prefix = f"{entry.category}/{entry.name}"
        remote = f"https://github.com/{entry.url}.git"
        pr_title = f"[Fanout] Sync rocm-libraries PR #{args.pr} to {entry.name}"
        pr_body = f"This is an automated PR for subtree `{entry.name}` from monorepo PR #{args.pr}."
        logger.debug(f"\nProcessing subtree: {entry.name}")
        logger.debug(f"\tPrefix: {prefix}")
        logger.debug(f"\tBranch: {branch}")
        logger.debug(f"\tRemote: {remote}")
        logger.debug(f"\tPR title: {pr_title}")
        split_and_push_subtree(entry, branch, prefix, remote, args.dry_run)
        pr_exists: bool = client.pr_view(entry.url, branch)
        if not pr_exists:
            if not args.dry_run:
                client.pr_create(entry.url, entry.branch, branch, pr_title, pr_body)
        else:
            if not args.dry_run:
                client.pr_edit(entry.url, branch, pr_title, pr_body)

if __name__ == "__main__":
    main()
