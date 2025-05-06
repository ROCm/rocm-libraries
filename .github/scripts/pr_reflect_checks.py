#!/usr/bin/env python3

"""
PR Checks Reflection Script
-----------------------------
This script polls the status of checks on fanned-out pull requests (in sub-repositories)
and reflects them as synthetic checks on the original monorepo pull request.

Each fanned-out PR must use the naming convention:
    monorepo-pr-<pr_number>-<subtree>

Arguments:
    --repo      : Full     --repo      : Full repository name (e.g., org/repo)
 name (e.g., org/repo)
    --pr        : Pull request number
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    To run in debug mode and perform a dry-run (no changes made):
    python pr-reflect-checks.py --repo ROCm/rocm-libraries --pr 123 --debug --dry-run
"""

import argparse
import logging
from typing import List, Optional
from github_checks_api_client import GitHubChecksAPIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reflect fanned-out PR checks onto the monorepo PR.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, help="Pull request number")
    parser.add_argument("--config", required=False, default=".github/repos-config.json")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def reflect_checks(client: GitHubChecksAPIClient, monorepo: str, pr_number: str, config: List[RepoEntry], dry_run: bool) -> None:
    """Reflect checks from fanned-out PRs to the monorepo PR."""
    for entry in config:
        subrepo = entry.url
        logger.debug(f"Fetching checks for {subrepo} PR {pr_number}")
        checks = client.get_pr_checks(subrepo, pr_number)
        for check in checks:
            check_name = f"{entry['name']}: {check['name']}"
            conclusion = check['conclusion'] or "neutral"
            status = check['status']
            summary = check.get('output', {}).get('summary', "")
            logger.info(f"[{check_name}] Status: {status} | Conclusion: {conclusion}")
            if not dry_run:
                client.upsert_check_run(monorepo, check_name, pr_number, status, conclusion, summary)

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to execute the PR checks reflection logic."""
    args = parse_arguments(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    client = GitHubChecksAPIClient()
    config = load_repo_config(args.config)
    reflect_checks(client, args.repo, args.pr, config, args.dry_run)

if __name__ == "__main__":
    main()
