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

# Coverage-enabled projects: project key -> (cmake_target, build_subdir, cmake_options)
# Only projects listed here will get coverage jobs. cmake_options pins the build
# to just this project so the coverage job does not inherit the (possibly merged)
# mega-group options that would otherwise build unrelated components.
COVERAGE_PROJECT_METADATA = {
    "hiprand": ("hipRAND", "ml-libs/hipRAND", "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF"),
}


def get_build_metadata(project_key: str, base_dir: str = "TheRock/build-coverage"):
    """Get CMake target and build directory for a coverage-enabled project.

    Returns:
        Tuple of (uppercase_name, cmake_target, build_dir, cmake_options) or None if not
        coverage-enabled
    """
    if project_key not in COVERAGE_PROJECT_METADATA:
        return None

    cmake_target, build_subdir, cmake_options = COVERAGE_PROJECT_METADATA[project_key]
    build_dir = f"{base_dir}/{build_subdir}/build"
    return project_key.upper(), cmake_target, build_dir, cmake_options


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

    # Filter: only keep projects that have coverage-enabled tests
    coverage_projects = []
    for proj in projects:
        pts_list = [p for p in proj.get("projects_to_test", "").split(",") if p]

        # Find primary project: prefer the first coverage-enabled test project,
        # then a changed project, otherwise fall back to the first entry.
        primary = next(
            (p for p in pts_list if p in COVERAGE_PROJECT_METADATA),
            pts_list[0] if pts_list else "",
        )
        # Only let a changed project override the choice if it is itself
        # coverage-enabled (otherwise a changed-but-uncovered project such as
        # hipdnn would displace a covered one like hiprand in a merged group).
        for p in pts_list:
            if p in changed_project_keys and p in COVERAGE_PROJECT_METADATA:
                primary = p
                break

        # FILTER: Skip if not coverage-enabled
        metadata = get_build_metadata(primary)
        if metadata is None:
            logging.info(f"Skipping {primary} - not coverage-enabled")
            continue

        # Add coverage metadata
        uppercase_name, cmake_target, build_dir, cmake_options = metadata
        proj["project_name"] = uppercase_name
        proj["cmake_target"] = cmake_target
        proj["build_dir"] = build_dir
        # Pin to this project's own options so we don't build the merged
        # mega-group (which pulls in unrelated components like hipdnn/providers).
        proj["cmake_options"] = cmake_options
        # Only run the coverage project's own tests, not every test in the
        # (possibly merged) group, so the test stage matches the pinned build.
        proj["projects_to_test"] = primary

        coverage_projects.append(proj)

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
