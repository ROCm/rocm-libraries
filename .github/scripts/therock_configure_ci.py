"""
This script determines which build flag and tests to run based on SUBTREES

Required environment variables:
  - SUBTREES
"""

import fnmatch
import json
import logging
import subprocess
from pathlib import Path
import sys
from therock_matrix import subtree_to_project_map, collect_projects_to_run
import time
from typing import Mapping, Optional, Iterable
import os
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees
from config_loader import load_repo_config

logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    logging.info(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        logging.warning(
            "Warning: GITHUB_OUTPUT env var not set, can't set github outputs"
        )
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())

def retry(max_attempts, delay_seconds, exceptions):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f'Exception {str(e)} thrown when attempting to run , attempt {attempt} of {max_attempts}')
                    attempt += 1
                    if attempt < max_attempts:
                        backoff = delay_seconds * (2 ** (attempt - 1))
                        time.sleep(backoff)
            return func(*args, **kwargs)
        return newfn
    return decorator

@retry(max_attempts=3, delay_seconds=2, exceptions=(TimeoutError))
def get_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of modified files relative to the base reference."""
    return subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    ).stdout.splitlines()


GITHUB_WORKFLOWS_CI_PATTERNS = [
    "therock*",
]


def is_path_workflow_file_related_to_ci(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    ) or any(
        fnmatch.fnmatch(path, ".github/scripts/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    )


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(is_path_workflow_file_related_to_ci(p) for p in paths)


def get_changed_path_projects(paths: Optional[Iterable[str]]) -> Iterable[str]:
    repo_config_path = Path(SCRIPT_DIR / ".." / "repos-config.json")
    config = load_repo_config(str(repo_config_path))
    valid_prefixes = get_valid_prefixes(config)
    matched_subtrees = find_matched_subtrees(paths, valid_prefixes)
    return matched_subtrees


def retrieve_projects(args):
    # For pushes and pull_requests, we only want to test changed projects
    base_ref = args.get("base_ref")
    modified_paths = get_modified_paths(base_ref)
    subtrees = get_changed_path_projects(modified_paths)
    
    # by default, we select full tests
    test_type = "full"

    if args.get("is_workflow_dispatch"):
        if args.get("input_projects") == "all":
            subtrees = list(subtree_to_project_map.keys())
        else:
            subtrees = args.get("input_projects").split()

    # If .github/*/therock* were changed for a push or pull request, run all subtrees
    if args.get("is_push") or args.get("is_pull_request"):
        related_to_therock_ci = check_for_workflow_file_related_to_ci(modified_paths)
        if related_to_therock_ci:
            subtrees = list(subtree_to_project_map.keys())
            test_type = "smoke"
            
    # for nightly runs, run everything with full tests
    if args.get("is_nightly"):
        subtrees = list(subtree_to_project_map.keys())

    project_to_run = collect_projects_to_run(subtrees)

    return project_to_run, test_type


def run(args):
    project_to_run, test_type = retrieve_projects(args)
    set_github_output({
        "projects": json.dumps(project_to_run),
        "test_type": test_type
    })


if __name__ == "__main__":
    args = {}
    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    args["is_pull_request"] = github_event_name == "pull_request"
    args["is_push"] = github_event_name == "push"
    args["is_workflow_dispatch"] = github_event_name == "workflow_dispatch"
    args["is_nightly"] = github_event_name == "schedule"

    input_projects = os.getenv("PROJECTS", "")
    args["input_projects"] = input_projects

    args["base_ref"] = os.environ.get("BASE_REF", "HEAD^")

    logging.info(f"Retrieved arguments {args}")

    run(args)
