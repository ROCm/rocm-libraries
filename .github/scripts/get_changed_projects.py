#!/usr/bin/env python3
"""Get changed project paths based on git diff."""

import os
import subprocess


def get_changed_files(base_ref: str) -> list[str]:
    """Get list of changed files from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def get_changed_projects(base_ref: str) -> str:
    """Get comma-separated list of changed project paths."""
    changed_files = get_changed_files(base_ref)
    if not changed_files:
        return ""

    projects = set()
    for file_path in changed_files:
        parts = file_path.split("/")
        if len(parts) >= 2 and parts[0] in ("projects", "shared"):
            projects.add(f"{parts[0]}/{parts[1]}")

    return ",".join(sorted(projects))


if __name__ == "__main__":
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    result = get_changed_projects(base_ref)
    print(result)
