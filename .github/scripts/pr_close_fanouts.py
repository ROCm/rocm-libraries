#!/usr/bin/env python3

"""
Close Fanned-Out PRs and Delete Branches
----------------------------------------

This script is used in the monorepo fanout automation system. It runs when a monorepo pull request
is closed (either merged or manually closed) and performs cleanup actions:

- Identifies all fanned-out pull requests that were created as part of the monorepo PR.
- Closes each corresponding pull request in the original sub-repositories.
- Deletes the temporary branch associated with each fanned-out PR.

It uses a consistent naming convention for identifying branches: monorepo-pr-<number>-<subtree>

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    To run in debug mode and perform a dry-run (no changes made):
        python pr_close_fanouts.py --repo ROCm/rocm-libraries --pr 123 --dry-run --debug
"""

import argparse
import logging
from typing import Optional, List
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Close fanned-out PRs and delete associated branches.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to GITHUB_OUTPUT.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to close fanned-out PRs and delete branches."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    for entry in config:
        branch = f"monorepo-pr-{args.pr}-{entry.name}"
        pr = client.get_pr_by_head_branch(entry.url, branch)
        if pr:
            number = pr["number"]
            client.close_pr_and_delete_branch(entry.url, number)
            logger.info(f"Closing PR #{number} in {entry.url} for branch {branch}")
        else:
            logger.info(f"No open PR found in {entry.url} for branch {branch}")

if __name__ == "__main__":
    main()
