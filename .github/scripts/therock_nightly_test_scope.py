#!/usr/bin/env python3

"""
TheRock Nightly Test Scope
--------------------------
TEMPORARY: this script is a workaround and is expected to be deleted.

TheRock does not yet scope a schedule run to the components an external
repository actually owns, so rocm-libraries has to compute that scope itself
and hand it over as `changed_projects`. Once TheRock narrows schedule runs to
the calling repository's own components natively, delete this script, its test,
and the `configure` job in therock-multi-arch-ci-nightly.yml that consumes it.

Computes the set of projects the multi-arch nightly CI workflow should test.

A scheduled run has no diff to narrow testing against, so TheRock would
otherwise test every component it builds, including ROCm system components
(hip-tests, rocrtst, rocgdb, rocprofiler-*, amdsmi, rccl, ...) that no change
in this repository can affect. This script emits every subtree registered in
.github/repos-config.json in `category/name` form, which the workflow passes to
TheRock as `changed_projects` so testing stays scoped to the code this
repository owns (plus the consumers TheRock's dependency graph pulls in).

Arguments:
    --config : OPTIONAL, path to the repos-config.json file.

Outputs:
    Writes 'test_projects' to the GitHub Actions $GITHUB_OUTPUT file as a
    comma-separated list of subtrees in `category/name` format, and appends a
    human-readable scope listing to $GITHUB_STEP_SUMMARY. Both are printed to
    stdout when run outside of GitHub Actions.

Example Usage:
    python therock_nightly_test_scope.py
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ci_utils import append_step_summary, set_github_output
from config_loader import load_repo_config
from pr_detect_changed_subtrees import get_valid_prefixes

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "repos-config.json"


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the nightly multi-arch CI test scope."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the repos-config.json file",
    )
    return parser.parse_args(argv)


def get_test_projects(config_path: Path) -> List[str]:
    """Return every subtree in repos-config.json as a sorted `category/name` list."""
    config = load_repo_config(str(config_path))
    return sorted(get_valid_prefixes(config))


def format_step_summary(test_projects: List[str]) -> str:
    """Render the test scope as a markdown step summary."""
    subtree_list = "\n".join(test_projects)
    return (
        "### Nightly test scope\n"
        "\n"
        "Testing rocm-libraries subtrees and their direct consumers only:\n"
        "\n"
        f"```\n{subtree_list}\n```\n"
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Emit the nightly test scope for the multi-arch CI workflow."""
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments(argv)

    test_projects = get_test_projects(args.config)
    if not test_projects:
        # TheRock reads an empty `changed_projects` as "test everything", which
        # is the opposite of this job's purpose, so fail instead of emitting it.
        raise SystemExit(f"No subtrees found in {args.config}")

    set_github_output({"test_projects": ",".join(test_projects)})
    append_step_summary(format_step_summary(test_projects))


if __name__ == "__main__":
    main()
