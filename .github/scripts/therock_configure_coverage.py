#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path

from therock_matrix import collect_projects_to_run
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees
from config_loader import load_repo_config
from ci_utils import get_modified_paths

logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent

# Coverage-enabled projects:
#   project key -> (cmake_target, build_subdir, cmake_options, coverage_config, artifact_stage)
# Only projects listed here will get coverage jobs. cmake_options pins the build
# to just this project so the coverage job does not inherit the (possibly merged)
# mega-group options that would otherwise build unrelated components.
# coverage_config is the per-project coverage metadata file, kept next to the
# project's existing test_categories.yaml (the repo's test_categories_*.yaml
# convention) instead of a single top-level file.
# artifact_stage is the stage name used in artifact naming (e.g., "rand" for hiprand/rocrand).
COVERAGE_PROJECT_METADATA = {
    "hiprand": (
        "hipRAND",
        "ml-libs/hipRAND",
        "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects/hiprand/test_categories_coverage.yaml",
        "rand",
    ),
}


def get_build_metadata(project_key: str, base_dir: str = "TheRock/build-coverage"):
    """Get CMake target and build directory for a coverage-enabled project.

    Returns:
        Tuple of (uppercase_name, cmake_target, build_dir, cmake_options,
        coverage_config, artifact_stage) or None if not coverage-enabled
    """
    if project_key not in COVERAGE_PROJECT_METADATA:
        return None

    cmake_target, build_subdir, cmake_options, coverage_config, artifact_stage = (
        COVERAGE_PROJECT_METADATA[project_key]
    )
    build_dir = f"{base_dir}/{build_subdir}/build"
    return project_key.upper(), cmake_target, build_dir, cmake_options, coverage_config, artifact_stage


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
    projects = collect_projects_to_run(subtrees)

    # Emit one INDEPENDENT coverage job per coverage-enabled project in each
    # group. Several coverage-enabled projects can share a group (e.g. the rand
    # group contains both rocrand and hiprand); each gets its own build -> test
    # -> report pipeline so the components are tracked separately.
    coverage_projects = []
    seen_projects = set()
    for proj in projects:
        pts_list = [p for p in proj.get("projects_to_test", "").split(",") if p]

        covered = [p for p in pts_list if p in COVERAGE_PROJECT_METADATA]
        if not covered:
            logging.info(
                "Skipping group with tests %s - no coverage-enabled project", pts_list
            )
            continue

        for project_key in covered:
            if project_key in seen_projects:
                continue  # avoid duplicate jobs if a project appears in multiple groups
            seen_projects.add(project_key)

            uppercase_name, cmake_target, build_dir, cmake_options, coverage_config, artifact_stage = (
                get_build_metadata(project_key)
            )
            # Copy the group entry so each coverage project gets its own job.
            entry = dict(proj)
            entry["project_name"] = uppercase_name
            entry["cmake_target"] = cmake_target
            entry["build_dir"] = build_dir
            # Pin to this project's own options so we don't build the merged
            # mega-group (which pulls in unrelated components like hipdnn/providers).
            entry["cmake_options"] = cmake_options
            # Per-project coverage metadata file (next to the project's tests).
            entry["coverage_config"] = coverage_config
            # Only run this project's own tests, so the test stage matches the
            # pinned (single-project) build.
            entry["projects_to_test"] = project_key
            # Ensure fetch_artifact_args is set so the test downloads the correct artifacts.
            # Use artifact_stage (e.g., "rand") not project_key (e.g., "hiprand") since
            # artifacts are named by stage: rand_lib_gfx950-dcgpu, not hiprand_lib_*.
            if "fetch_artifact_args" not in entry or not entry["fetch_artifact_args"]:
                entry["fetch_artifact_args"] = f"--{artifact_stage} --tests"
            coverage_projects.append(entry)

    # Output for GitHub Actions
    output = {
        "coverage_projects": json.dumps(coverage_projects),
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
