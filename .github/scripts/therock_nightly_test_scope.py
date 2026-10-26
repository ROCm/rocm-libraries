#!/usr/bin/env python3

"""Emits the subtrees the multi-arch nightly CI should test, as `category/name`.

TEMPORARY: a scheduled run has no diff to narrow testing against, so TheRock
tests every component it builds, including ROCm system components (hip-tests,
rocrtst, rocgdb, amdsmi, ...) that no change here can affect. Passing every
subtree in repos-config.json as `changed_projects` scopes it back to the code
this repo owns. Delete this script, its test, and the `configure` job in
therock-multi-arch-ci-nightly.yml once TheRock scopes schedule runs natively.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ci_utils import append_step_summary, set_github_output
from config_loader import load_repo_config
from pr_detect_changed_subtrees import get_valid_prefixes

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "repos-config.json"


def get_test_projects(config_path: Path) -> List[str]:
    """Return every subtree in repos-config.json as a sorted `category/name` list."""
    return sorted(get_valid_prefixes(load_repo_config(str(config_path))))


def format_step_summary(test_projects: List[str]) -> str:
    """Render the test scope as a markdown step summary."""
    subtrees = "\n".join(test_projects)
    return (
        "### Nightly test scope\n\n"
        "Testing rocm-libraries subtrees and their direct consumers only:\n\n"
        f"```\n{subtrees}\n```\n"
    )


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the repos-config.json file",
    )
    args = parser.parse_args(argv)

    test_projects = get_test_projects(args.config)
    if not test_projects:
        # TheRock reads an empty `changed_projects` as "test everything".
        raise SystemExit(f"No subtrees found in {args.config}")

    set_github_output({"test_projects": ",".join(test_projects)})
    append_step_summary(format_step_summary(test_projects))


if __name__ == "__main__":
    main()
