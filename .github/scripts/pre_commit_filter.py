#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Pre-commit Filter Script
------------------------
This script filters changed files for pre-commit checks in the CI workflow.
It determines which files should undergo pre-commit checks based on opted-in projects.

Logic:
    1. Identify changed files (via git diff or provided list).
    2. Filter files:
        - Check if file exists (skip deleted files).
        - Include files belonging to opted-in projects (projects/<project>/... or shared/<project>/...).
        - Include files outside the 'projects/' and 'shared/' directories.
    3. Output results to GITHUB_OUTPUT:
        - should_run: 'true' if there are filtered files, else 'false'.
        - files: Newline-separated list of filtered files.
        - <project>_changed: 'true' for each opted-in project that has changes.

Usage:
    python pre_commit_filter.py --base-ref origin/develop --head-ref HEAD
    python pre_commit_filter.py --file-list "file1.py file2.cpp"
"""

import argparse
import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Set, Tuple

# Logs go to stderr to avoid interfering with other output
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter changed files for pre-commit checks."
    )
    parser.add_argument(
        "--base-ref", help="Git base reference for diff (e.g., origin/develop)"
    )
    parser.add_argument("--head-ref", help="Git head reference for diff (e.g., HEAD)")
    parser.add_argument(
        "--file-list",
        nargs="+",
        help="List of changed files (alternative to git diff for testing purposes)",
    )
    parser.add_argument("projects", nargs="*", help="List of opted-in projects")
    return parser.parse_args()


def get_changed_files(base_ref: str, head_ref: str) -> List[str]:
    """Get list of changed files using git diff."""
    try:
        cmd = ["git", "diff", "-z", "--name-only", f"{base_ref}...{head_ref}"]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [line for line in result.stdout.split("\0") if line.strip()]
    except subprocess.CalledProcessError as e:
        logger.error(f"Git diff failed: {e.stderr}")
        sys.exit(1)


def filter_files(
    changed_files: List[str], opted_in_projects: Set[str]
) -> Tuple[List[str], Set[str]]:
    """
    Filter changed files based on opted-in projects.
    Returns:
        - List of sorted filtered files to run pre-commit on.
        - Set of opted-in projects that have changes.
    """
    PROJECT_FOLDERS = {"projects", "shared"}

    filtered_files = set()
    changed_projects = set()

    for path_str in changed_files:
        path = Path(path_str)

        # skip deleted files
        if not path.exists():
            continue

        parts = path.parts

        # Check if file is in an opted-in project (either in projects/ or shared/)
        # Structure: projects/<project_name>/... or shared/<project_name>/...
        if len(parts) > 1 and parts[0] in PROJECT_FOLDERS:
            project = parts[1]
            if project in opted_in_projects:
                filtered_files.add(path_str)
                changed_projects.add(project)
        # Vacuous inclusion: files outside 'projects/' and 'shared/' are always included
        else:
            filtered_files.add(path_str)

    return sorted(filtered_files), changed_projects


def write_github_output(key: str, value: str):
    """Write a key-value pair to GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            if "\n" in value:
                # EOF to mark start and end of multiline value
                f.write(f"{key}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{key}={value}\n")
    else:
        # Fallback for local testing
        logger.info(f"[GITHUB_OUTPUT] {key}={value}")


def main():
    args = parse_arguments()

    if args.file_list:
        changed_files = args.file_list
    elif args.base_ref and args.head_ref:
        changed_files = get_changed_files(args.base_ref, args.head_ref)
    else:
        logger.error(
            "Error: Must provide either --file-list or both --base-ref and --head-ref"
        )
        sys.exit(1)

    logger.info(f"Changed files: {changed_files}")

    filtered_files, changed_projects = filter_files(changed_files, set(args.projects))

    if not filtered_files:
        logger.info("================================================")
        logger.info("Skipping pre-commit checks")
        logger.info("================================================")
        logger.info("No change found in projects using pre-commit")
        logger.info(f"Opted-in projects: {', '.join(args.projects)}")

        write_github_output("should_run", "false")
    else:
        logger.info("Running pre-commit on the following files:")
        for f in filtered_files:
            logger.info(f)

        write_github_output("should_run", "true")
        write_github_output("files", "\n".join(filtered_files))

        for project in changed_projects:
            write_github_output(f"{project}_changed", "true")


if __name__ == "__main__":
    main()
