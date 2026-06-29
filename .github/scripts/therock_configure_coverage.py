#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path

from therock_matrix import collect_projects_to_run
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
    "hiprand": (
        "hipRAND",
        "ml-libs/hipRAND",
        "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    "rocrand": (
        "rocRAND",
        "math-libs/rocRAND",
        "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    "rocfft": (
        "rocFFT",
        "math-libs/rocFFT",
        "-DTHEROCK_ENABLE_FFT=ON -DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    "rocblas": (
        "rocBLAS",
        "math-libs/rocBLAS",
        "-DTHEROCK_ENABLE_BLAS=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    # Header-only libraries: coverage comes from the instrumented test binaries
    # (there is no shared library to instrument). All three build together in
    # TheRock's PRIM group.
    "rocprim": (
        "rocPRIM",
        "math-libs/rocPRIM",
        "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    "hipcub": (
        "hipCUB",
        "math-libs/hipCUB",
        "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
    "rocthrust": (
        "rocThrust",
        "math-libs/rocThrust",
        "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
    ),
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

            uppercase_name, cmake_target, build_dir, cmake_options = get_build_metadata(
                project_key
            )
            # Copy the group entry so each coverage project gets its own job.
            entry = dict(proj)
            entry["project_name"] = uppercase_name
            entry["cmake_target"] = cmake_target
            entry["build_dir"] = build_dir
            # Pin to this project's own options so we don't build the merged
            # mega-group (which pulls in unrelated components like hipdnn/providers).
            entry["cmake_options"] = cmake_options
            # Only run this project's own tests, so the test stage matches the
            # pinned (single-project) build.
            entry["projects_to_test"] = project_key
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
