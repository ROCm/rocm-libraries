#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map changed monorepo subtrees to TheRock multi-arch test projects."""

import argparse

from ci_utils import set_github_output
from therock_matrix import collect_projects_to_run, subtree_to_project_map

# The legacy BLAS umbrella does not include these shared components' own
# relevant test jobs.
SHARED_ADJACENT_TEST_PROJECTS = {
    "shared/origami": ["origami"],
    "shared/mxdatagenerator": ["rocroller"],
}


def select_test_projects(changed_projects: str) -> str:
    """Return the projects TheRock should test for changed monorepo subtrees."""
    if not changed_projects:
        return ""

    selected: list[str] = []

    for changed_project in changed_projects.split(","):
        changed_project = changed_project.strip()
        if not changed_project:
            continue

        if not changed_project.startswith("shared/"):
            selected.append(changed_project)
            continue

        if changed_project not in subtree_to_project_map:
            raise ValueError(f"Shared project has no test mapping: {changed_project}")

        project_rows = collect_projects_to_run(
            [changed_project], run_rocjitsu_race_check=False
        )
        if not project_rows:
            raise ValueError(f"Shared project selects no tests: {changed_project}")

        shared_tests = {
            project
            for row in project_rows
            for project in row["projects_to_test"].split(",")
        }
        shared_tests.update(SHARED_ADJACENT_TEST_PROJECTS.get(changed_project, []))
        selected.extend(sorted(shared_tests))

    return ",".join(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select TheRock multi-arch tests for changed monorepo projects."
    )
    parser.add_argument(
        "--changed-projects",
        default="",
        help="Comma-separated monorepo subtree paths selected by CI.",
    )
    args = parser.parse_args()

    test_projects = select_test_projects(args.changed_projects)
    set_github_output({"test_projects": test_projects})


if __name__ == "__main__":
    main()
