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
from typing import Mapping, Optional, Iterable, List
import os
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees
from config_loader import load_repo_config

# Add TheRock's github_actions to path for shared utilities
THEROCK_ACTIONS_PATH = Path("TheRock") / "build_tools" / "github_actions"
sys.path.insert(0, str(THEROCK_ACTIONS_PATH))
from amdgpu_family_matrix import BUILD_RUNNER_LABELS, select_weighted_label

logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent

# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
SKIPPABLE_PATH_PATTERNS = [
    "*.clinerules",
    "*.cursorrules",
    "*.mdc",
    "*.md",
    "*.rst",
    "*.rtf",
    "*/.markdownlint-ci2.yaml",
    "*/.readthedocs.yaml",
    "*/.spellcheck.local.yaml",
    "*/.wordlist.txt",
    ".gitignore",
    "dnn-providers/*/.gitignore",
    "dnn-providers/*/docs/*",
    "docs/*",
    "projects/*/.gitignore",
    "projects/*/docs/*",
    # Tools are standalone scripts/utilities not part of the build or test pipeline.
    # Changes here should not trigger CI builds.
    "projects/hipdnn/tools/*",
    "shared/*/.gitignore",
    "shared/*/docs/*",
    "projects/composablekernel/Jenkinsfile",
    "projects/composablekernel/Docker*",
    "projects/composablekernel/client_example/*",
    "projects/composablekernel/codegen/*",
    "projects/composablekernel/dispatcher/*",
    "projects/composablekernel/example/*",
    "projects/composablekernel/experimental/*",
    "projects/composablekernel/profiler/*",
    "projects/composablekernel/python/*",
    "projects/composablekernel/rocm_ck/*",
    "projects/composablekernel/script/*",
    "projects/composablekernel/test/*",
    "projects/composablekernel/test_data/*",
    "projects/composablekernel/tile_engine/*",
    "projects/composablekernel/tutorial/*",
    "projects/composablekernel/vars/*",
    "projects/composablekernel/groovy/*",
]


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(args.get("pr_labels", "{}"))
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def parse_test_labels(pr_labels: List[str]) -> tuple[List[str], Optional[str]]:
    """
    Parse PR labels to extract test projects and test type.

    Examples:
        ['test:rocblas', 'test:miopen'] -> (['rocblas', 'miopen'], None)
        ['test:rocblas', 'test_type:comprehensive'] -> (['rocblas'], 'comprehensive')
    """
    projects_to_test = []
    test_type = None

    # Build label_to_project_map from subtree_to_project_map
    label_to_project_map = {}
    for subtree, project in subtree_to_project_map.items():
        # Extract the project name from the subtree path (e.g., "projects/rocblas" -> "rocblas")
        if subtree.startswith("projects/") or subtree.startswith("dnn-providers/"):
            label_name = subtree.split("/")[-1]
            label_to_project_map[label_name] = project

    # Valid test types in order of comprehensiveness (least to most)
    valid_test_types = ["quick", "standard", "comprehensive", "full"]
    test_type_priority = {t: i for i, t in enumerate(valid_test_types)}

    for label in pr_labels:
        # Parse test:* labels for project selection
        if label.startswith("test:"):
            project_name = label[5:]  # Remove 'test:' prefix
            if project_name in label_to_project_map:
                mapped_project = label_to_project_map[project_name]
                if mapped_project not in projects_to_test:
                    projects_to_test.append(mapped_project)
            else:
                logging.warning(f"Unknown project in label: {label}")
                continue

        # Parse test_type:* labels
        elif label.startswith("test_type:"):
            label_test_type = label.split("test_type:")[
                -1
            ]  # Remove 'test_type:' prefix
            if label_test_type in valid_test_types:
                # If multiple test_type labels, use the most comprehensive one
                if (
                    test_type is None
                    or test_type_priority[label_test_type]
                    > test_type_priority[test_type]
                ):
                    test_type = label_test_type
            else:
                logging.warning(f"Unknown test type in label: {label}")

    return projects_to_test, test_type


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


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
                    print(
                        f"Exception {str(e)} thrown when attempting to run , attempt {attempt} of {max_attempts}"
                    )
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


def select_build_runner(platform: str) -> str:
    """Select a build runner label based on platform and build variant."""
    if platform not in BUILD_RUNNER_LABELS:
        # Platform not configured for weighted selection, return default
        print(f"  No build runner config for platform {platform}, using default")
        return ""

    platform_config = BUILD_RUNNER_LABELS[platform]

    labels_config = platform_config["default"]
    context_name = f"build-runner ({platform})"

    return select_weighted_label(labels_config, context_name)


def retrieve_projects(args):
    # For pushes and pull_requests, we only want to test changed projects
    base_ref = args.get("base_ref")
    modified_paths = get_modified_paths(base_ref)

    # by default, we select standard tests
    test_type = "standard"

    # Variables to track if labels override defaults
    label_projects = []
    label_test_type = None

    # Check if CI should be skipped based on modified paths
    # (only for push and pull_request events, not workflow_dispatch or nightly)
    if args.get("is_push") or args.get("is_pull_request"):
        paths_set = set(modified_paths)
        contains_non_skippable_files = check_for_non_skippable_path(paths_set)
        pr_labels = get_pr_labels(args)

        # Parse PR labels for test: and test_type: labels
        label_projects, label_test_type = parse_test_labels(pr_labels)

        # If test_type label is present, override the default test type
        if label_test_type:
            test_type = label_test_type
            logging.info(f"Test type overridden by label: {test_type}")

        # If only skippable paths were modified and no test labels, skip CI
        if not contains_non_skippable_files and not label_projects:
            logging.info("Only skippable paths were modified, skipping CI")
            return [], test_type

        if "skip-therockci" in pr_labels:
            logging.info("`skip-therockci` label was added, skipping CI")
            return [], test_type

    subtrees = get_changed_path_projects(modified_paths)

    # If test: labels are present (only for pull requests), add those projects to the build/test list
    if label_projects and args.get("is_pull_request"):
        logging.info(f"Projects specified by labels: {label_projects}")
        # Find at least one subtree that maps to each labeled project
        # We need actual subtrees because collect_projects_to_run expects them
        label_subtrees = []
        for project in label_projects:
            # Find first subtree that maps to this project
            for subtree, mapped_project in subtree_to_project_map.items():
                if mapped_project == project:
                    label_subtrees.append(subtree)
                    break  # Only need one representative subtree per project

        # Combine file-based detection with label-based selection
        subtrees = list(set(subtrees + label_subtrees))

    if args.get("is_workflow_dispatch"):
        if args.get("input_projects") == "all":
            subtrees = list(subtree_to_project_map.keys())
        else:
            subtrees = args.get("input_projects").split()

    # If .github/*/therock* were changed for a push or pull request, run all subtrees
    if args.get("is_push") or args.get("is_pull_request"):
        related_to_therock_ci = check_for_workflow_file_related_to_ci(modified_paths)
        if related_to_therock_ci:
            logging.info(
                "Enabling all projects since a related workflow file was modified"
            )
            subtrees = list(subtree_to_project_map.keys())
            # Only override test_type if not already set by label (and it's a PR)
            if not (label_test_type and args.get("is_pull_request")):
                test_type = "quick"

    # for nightly runs, run everything with full tests
    if args.get("is_nightly"):
        subtrees = list(subtree_to_project_map.keys())
        test_type = "comprehensive"

    project_to_run = collect_projects_to_run(subtrees)

    return project_to_run, test_type


def run(args):
    platform = args.get("platform")
    project_to_run, test_type = retrieve_projects(args)
    build_runs_on = select_build_runner(platform)
    set_github_output(
        {
            f"{platform}_projects": json.dumps(project_to_run),
            "test_type": test_type,
            "build_runs_on": build_runs_on,
        }
    )


if __name__ == "__main__":
    args = {}
    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    platform = os.getenv("PLATFORM")
    args["platform"] = platform
    args["is_pull_request"] = github_event_name == "pull_request"
    args["is_push"] = github_event_name == "push"
    args["is_workflow_dispatch"] = github_event_name == "workflow_dispatch"
    args["is_nightly"] = github_event_name == "schedule"

    args["pr_labels"] = os.environ.get("PR_LABELS", '{"labels": []}')

    input_projects = os.getenv("PROJECTS", "")
    args["input_projects"] = input_projects

    args["base_ref"] = os.environ.get("BASE_REF", "HEAD^")

    logging.info(f"Retrieved arguments {args}")

    run(args)
