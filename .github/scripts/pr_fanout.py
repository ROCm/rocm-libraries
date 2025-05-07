#!/usr/bin/env python3

"""
PR Fanout Script
------------------
This script takes a list of changed subtrees in `category/name` format and for each:
    - Pushes the corresponding subtree directory from the monorepo to the appropriate branch in the sub-repo using `git subtree push`.
    - Creates or updates a pull request in the sub-repo with a standardized branch and label.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --subtrees  : A newline-separated list of subtree paths in category/name format (e.g., projects/rocBLAS)
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:

    To run in debug mode and perform a dry-run (no changes made):
        python pr-fanout.py --repo ROCm/rocm-libraries --pr 123 --subtrees "$(printf 'projects/rocBLAS\nprojects/hipBLASLt\nshared/rocSPARSE')" --dry-run --debug
"""

import argparse
import subprocess
import logging
from typing import List, Optional
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fanout monorepo PR to sub-repos.")
    parser.add_argument("--repo", required=True, help="Full  repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--subtrees", required=True, help="Newline-separated list of changed subtrees (category/name)")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def get_subtree_info(config: List[RepoEntry], subtrees: List[str]) -> List[RepoEntry]:
    """Return config entries matching the given subtrees in category/name format."""
    requested = set(subtrees)
    matched = [
        entry for entry in config
        if f"{entry.category}/{entry.name}" in requested
    ]
    missing = requested - {f"{e.category}/{e.name}" for e in matched}
    if missing:
        logger.warning(f"Some subtrees not found in config: {', '.join(sorted(missing))}")
    return matched

def subtree_push(entry: RepoEntry, branch: str, prefix: str, subrepo_full_url: str, dry_run: bool) -> None:
    """Push the specified subtree to the sub-repo using `git subtree push`."""
    push_cmd = ["git", "subtree", "push", "--prefix", prefix, subrepo_full_url, branch]
    logger.debug(f"Running: {' '.join(push_cmd)}")
    if not dry_run:
        subprocess.run(push_cmd, check=True)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to execute the PR fanout logic."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    subtrees = [line.strip() for line in args.subtrees.splitlines() if line.strip()]
    relevant_subtrees = get_subtree_info(config, subtrees)
    for entry in relevant_subtrees:
        branch = f"monorepo-pr-{args.pr}-{entry.name}"
        prefix = f"{entry.category}/{entry.name}"
        subrepo_full_url = f"https://github.com/{entry.url}.git"
        pr_title = f"[Fanout] Sync {args.repo} PR #{args.pr} to {entry.name}"
        pr_body = f"This is an automated PR for subtree `{entry.category}/{entry.name}` from monorepo PR #{args.pr}."
        logger.debug(f"\nProcessing subtree: {entry.category}/{entry.name}")
        logger.debug(f"\tPrefix: {prefix}")
        logger.debug(f"\tBranch: {branch}")
        logger.debug(f"\tRemote: {subrepo_full_url}")
        logger.debug(f"\tPR title: {pr_title}")
        subtree_push(entry, branch, prefix, subrepo_full_url, args.dry_run)
        pr_exists: bool = client.pr_view(entry.url, branch)
        if not pr_exists:
            if not args.dry_run:
                client.pr_create(entry.url, entry.branch, branch, pr_title, pr_body)
                logger.info(f"Created PR in {entry.url} for branch {branch}")

if __name__ == "__main__":
    main()
