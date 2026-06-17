#!/usr/bin/env python3
"""Get changed project paths based on git diff, validated against repos-config.json."""

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Optional

from ci_utils import get_modified_paths, matches_paths
from config_loader import load_repo_config
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees

SCRIPT_DIR = Path(__file__).resolve().parent

SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "docs/*",
    "projects/*/docs/*",
    "shared/*/docs/*",
]

THEROCK_CI_PATTERNS = [
    ".github/workflows/therock*",
    ".github/scripts/therock*",
]


def is_path_skippable(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return matches_paths(paths, THEROCK_CI_PATTERNS)


def get_changed_projects(base_ref: str) -> str:
    """Get comma-separated list of changed project paths validated against repos-config.json."""
    modified_paths = get_modified_paths(base_ref)
    if not modified_paths:
        return ""

    # TODO: Uncomment after testing
    # # If CI workflow files changed, run all tests
    # if check_for_workflow_file_related_to_ci(modified_paths):
    #     return ""

    # # If only skippable files changed, skip
    # if not check_for_non_skippable_path(modified_paths):
    #     return ""

    repo_config_path = SCRIPT_DIR / ".." / "repos-config.json"
    config = load_repo_config(str(repo_config_path))
    valid_prefixes = get_valid_prefixes(config)
    matched_subtrees = find_matched_subtrees(list(modified_paths), valid_prefixes)

    return ",".join(matched_subtrees)


if __name__ == "__main__":
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    result = get_changed_projects(base_ref)
    print(result)
