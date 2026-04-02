#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path

from therock_matrix import collect_projects_to_run, subtree_to_project_map
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees
from config_loader import load_repo_config
from therock_configure_ci import get_modified_paths  # reuse existing helper

logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent


def get_changed_subtrees_only():
    repo_config_path = SCRIPT_DIR / ".." / "repos-config.json"
    config = load_repo_config(str(repo_config_path))
    valid_prefixes = get_valid_prefixes(config)

    base_ref = os.environ.get("BASE_REF", "HEAD^")
    modified_paths = get_modified_paths(base_ref)

    matched_subtrees = find_matched_subtrees(modified_paths, valid_prefixes)
    return matched_subtrees


def main():
    subtrees = get_changed_subtrees_only()

    changed_project_keys = set()
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            changed_project_keys.add(subtree_to_project_map[subtree])

    projects = collect_projects_to_run(subtrees)

    for proj in projects:
        pts_list = [p for p in proj.get("projects_to_test", "").split(",") if p]

        # TODO: We don't want a default
        primary = pts_list[0] if pts_list else ""

        # TODO: There's a better way than iterating and comparing to projects_to_test
        for p in pts_list:
            if p in changed_project_keys:
                primary = p
                break

        proj["project_name"] = primary.upper()

    output = {
        "linux_projects_coverage": json.dumps(projects),
    }
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for k, v in output.items():
                f.write(f"{k}={v}\n")
    else:
        logging.warning("GITHUB_OUTPUT not set; printing to stdout instead")
        print(json.dumps(output))


if __name__ == "__main__":
    main()
