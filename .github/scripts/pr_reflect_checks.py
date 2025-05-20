#!/usr/bin/env python3

"""
PR Checks Reflection Script
-----------------------------
This script polls the status of checks on fanned-out pull requests (in sub-repositories)
and reflects them as synthetic checks on the original monorepo pull request.
The branch name convention for fanned-out PRs is computed using the FanoutNaming class.

Steps:
    1. Fetch the status of checks for the monorepo PR.
    2. Load the subtree mapping from repos-config.json.
    3. For each sub-repo, check if there is an open PR with the expected branch name.
    4. For each check in the sub-repo PR, reflect it as a synthetic check on the monorepo PR.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --config    : OPTIONAL, path to the repos-config.json file
    --subrepo   : OPTIONAL, only process this subrepo by its repo name (e.g., ROCm/hipBLASlt).
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    To run in debug mode and perform a dry-run (no changes made):
        python pr_reflect_checks.py --repo ROCm/rocm-libraries --pr 123 --debug --dry-run
    To run in debug mode and perform a dry-run for a specific subrepo:
        python pr_reflect_checks.py --repo ROCm/rocm-libraries --pr 123 --subrepo ROCm/hipBLASlt --debug --dry-run
"""

import argparse
import logging
from typing import List, Optional

from github_cli_client import GitHubCLIClient
from config_loader import load_repo_config
from utils_fanout_naming import FanoutNaming

logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reflect fanned-out PR checks onto the monorepo PR.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--subrepo", required=False, help="If set, only process this subrepo.")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def reflect_checks_from_subrepos(client: GitHubCLIClient, config: list, monorepo_repo: str,
                                    monorepo_pr_number: int, subrepo_filter: Optional[str] = None,
                                    dry_run: bool = False) -> None:
    """Reflect checks from subrepo PRs onto the corresponding monorepo PR."""
    monorepo_branch = client.get_branch_name_for_pr(monorepo_repo, monorepo_pr_number)
    monorepo_pr_sha = client.get_head_sha_for_pr(monorepo_repo, monorepo_pr_number)
    monorepo_checks = {
        check["name"]: check
        for check in client.get_check_runs_for_ref(monorepo_repo, monorepo_branch)
    }
    for entry in config:
        if subrepo_filter and entry.url != subrepo_filter:
            continue
        reflect_checks_for_subrepo(
            client, entry, monorepo_repo, monorepo_pr_sha,
            monorepo_checks, monorepo_pr_number, dry_run
        )

def reflect_checks_for_subrepo(client: GitHubCLIClient, entry, monorepo_repo: str, monorepo_pr_sha: str,
                                monorepo_checks: dict, monorepo_pr_number: int, dry_run: bool) -> None:
    """Reflect checks for a single subrepo entry onto the monorepo PR."""
    subrepo = entry.url
    branch = FanoutNaming.compute_branch_name(monorepo_pr_number, entry.name)
    pr = client.get_pr_by_head_branch(subrepo, branch)
    if not pr:
        logger.info(f"No open PR found in {subrepo} for branch {branch}")
        return
    checks = client.get_check_runs_for_ref(subrepo, branch)
    for check in checks:
        synthetic_name = f"{entry.name}: {check['name']}"
        status = check["status"]
        details_url = check.get("details_url", "")
        conclusion = check.get("conclusion")
        completed_at = check.get("completed_at")
        title = check.get("output", {}).get("title", synthetic_name)
        summary = check.get("output", {}).get("summary", "")
        existing = monorepo_checks.get(synthetic_name)
        needs_update = (
            not existing or
            existing["status"] != status or
            existing.get("conclusion") != conclusion or
            existing.get("output", {}).get("summary") != summary
        )
        if not needs_update:
            logger.debug(f"Skipped unchanged check: {synthetic_name}")
            continue
        logger.info(f"Reflecting check: {synthetic_name}")
        if not dry_run:
            client.upsert_check_run(
                monorepo_repo, synthetic_name, monorepo_pr_sha,
                status, details_url, conclusion, completed_at,
                title, summary
            )
        else:
            logger.info(f"Dry run: would reflect check: {check}")

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to execute the PR checks reflection logic."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    reflect_checks_from_subrepos(
        client,
        config,
        monorepo_repo = args.repo,
        monorepo_pr_number = args.pr,
        subrepo_filter = args.subrepo,
        dry_run = args.dry_run
    )

if __name__ == "__main__":
    main()
