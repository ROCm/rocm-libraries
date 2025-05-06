#!/usr/bin/env python3

"""
PR Fanout Sync Label Script
-----------------------
This script reads labels from a monorepo pull request and ensures they exist
on all related fanned-out pull requests, skipping any label that does not
already exist in the sub-repos.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    To run in debug mode and perform a dry-run (no changes made):
    python pr_fanout_sync_labels.py --repo ROCm/rocm-libraries --pr 123 --debug --dry-run
"""

import argparse
import logging
from typing import List, Optional
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sync labels from monorepo PR to fanned-out PRs.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, help="Pull request number")
    parser.add_argument("--config", required=False, default=".github/repos-config.json")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def sync_labels(client: GitHubCLIClient, monorepo: str, pr_number: str, entries: List[RepoEntry], dry_run: bool) -> None:
    """Sync labels from the monorepo PR to the fanned-out PRs."""
    source_labels = client.get_existing_labels(monorepo, pr_number)
    for entry in entries:
        branch = f"monorepo-pr-{pr_number}-{entry.name}"
        subrepo = entry.url
        logger.debug(f"Processing labels for {subrepo} PR branch {branch}")
        existing_pr = client.pr_view(entry.url, branch)
        if not existing_pr:
            logger.debug(f"No PR found for branch {branch} in {subrepo}")
            continue
        else:
            subrepo_labels = client.get_existing_labels(subrepo, existing_pr)
            applicable_labels = [label for label in source_labels if label in subrepo_labels]
            logger.debug(f"Applying labels to {subrepo}#{existing_pr}: {applicable_labels}")
            if not dry_run:
                client.sync_labels(monorepo, subrepo, applicable_labels, dry_run)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to execute the PR fanout label sync logic."""
    args = parse_arguments(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    sync_labels(client, args.repo, args.pr, config, args.dry_run)

if __name__ == "__main__":
    main()
