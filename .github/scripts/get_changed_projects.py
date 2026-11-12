#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Select TheRock projects using the repository's existing skip-CI policy."""

import fnmatch
import logging
from pathlib import Path
import tomllib

import configure_external_repo_ci as therock
from configure_ci_path_filters import get_modified_paths_via_api


def get_changed_projects(
    modified_paths: set[str] | None, config_path: Path
) -> therock.ConfigureResult:
    run_all = therock.ConfigureResult("", True, "quick")
    if modified_paths is None:
        logging.warning("Changed files unavailable - running all tests")
        return run_all

    # The project list feeds both Linux and Windows. Platform-specific skips
    # remain in TheRock's per-platform setup; only common skips apply here.
    skip_patterns = []
    for path in (
        Path(therock.__file__).with_name("skip-ci-base.toml"),
        config_path.with_name("skip-ci-config.toml"),
    ):
        try:
            with path.open("rb") as f:
                patterns = tomllib.load(f).get("skip_ci", {}).get("common", [])
            if not isinstance(patterns, list) or any(
                not isinstance(p, str) for p in patterns
            ):
                raise ValueError("common skip patterns must be a list of strings")
            skip_patterns.extend(patterns)
        except (OSError, ValueError, AttributeError) as e:
            logging.warning(
                "Cannot load skip policy %s: %s - running all tests", path, e
            )
            return run_all

    relevant = {
        path
        for path in modified_paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in skip_patterns)
    }
    logging.info(
        "Ignoring paths excluded by skip-CI policy: %s",
        sorted(modified_paths - relevant),
    )
    if not relevant:
        # TheRock setup decides whether the whole run can skip.
        return therock.ConfigureResult("", False, "standard")
    if therock.matches_patterns(relevant, therock.FULL_TEST_TRIGGER_PATTERNS):
        logging.info("Shared CI or test-selection files changed - running all tests")
        return run_all

    config = therock.load_repo_config(str(config_path))
    if not config:
        return run_all
    configured_projects = therock.get_valid_prefixes(config)
    valid_prefixes = configured_projects | therock.CI_RELEVANT_NON_SUBTREE_PREFIXES
    projects = set()
    unknown = []
    for path in relevant:
        if therock.matches_patterns([path], therock.CALLER_WORKFLOW_TRIGGER_PATTERNS):
            projects.update(configured_projects)
        else:
            matched = therock.find_matched_subtrees([path], valid_prefixes)
            projects.update(matched)
            if not matched:
                unknown.append(path)
    if unknown:
        logging.info("Unclassified changes %s - running all tests", sorted(unknown))
        return run_all

    return therock.ConfigureResult(",".join(sorted(projects)), False, "standard")


def main(argv=None):
    args = therock.parse_args(argv)
    paths = None
    if args.event_name in ("pull_request", "push") and args.base_sha and args.head_sha:
        changed = get_modified_paths_via_api(
            args.github_repo, args.base_sha, args.head_sha
        )
        if changed is not None:
            paths = set(changed)
    result = get_changed_projects(paths, Path(args.config_path))
    therock.set_github_output(
        {
            "changed_projects": result.changed_projects,
            "run_all_tests": str(result.run_all_tests).lower(),
            "test_type": result.test_type,
        }
    )


if __name__ == "__main__":
    main()
