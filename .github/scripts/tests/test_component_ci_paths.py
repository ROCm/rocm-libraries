# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that component CI paths still match files in the repository."""

import fnmatch
import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component_ci import COMPONENTS, INFRA_PATHS

REPO_ROOT = Path(__file__).resolve().parents[3]


def tracked_files():
    # The index includes the complete tree in a sparse checkout and newly staged files.
    return subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


class ComponentCIPathsTest(unittest.TestCase):
    def test_every_pattern_matches_a_tracked_file(self):
        patterns = {path for group in COMPONENTS.values() for path in group}
        patterns.update(INFRA_PATHS)

        tracked = tracked_files()

        # Same matcher as component_ci.matches_paths, so a pattern that matches
        # nothing here can never trigger its component.
        unmatched = sorted(
            p for p in patterns if not any(fnmatch.fnmatch(f, p) for f in tracked)
        )
        self.assertEqual(
            unmatched, [], f"Component CI patterns matching no file: {unmatched}"
        )

    def test_therock_policy_patterns_and_owners_are_current(self):
        config_path = REPO_ROOT / ".github/repos-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with (config_path.parent / "ci-paths.toml").open("rb") as f:
            policy = tomllib.load(f)

        registered = {
            f"{entry['category']}/{entry['name']}" for entry in config["repositories"]
        }
        self.assertEqual(set(policy["project_paths"]) - registered, set())
        patterns = set(policy["full_test_paths"])
        patterns.update(p for paths in policy["project_paths"].values() for p in paths)
        tracked = tracked_files()
        unmatched = sorted(
            p for p in patterns if not any(fnmatch.fnmatch(f, p) for f in tracked)
        )
        self.assertEqual(
            unmatched, [], f"TheRock CI patterns matching no file: {unmatched}"
        )


if __name__ == "__main__":
    unittest.main()
