# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that component CI paths still match files in the repository."""

import fnmatch
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component_ci import COMPONENTS, INFRA_PATHS


class ComponentCIPathsTest(unittest.TestCase):
    def test_every_pattern_matches_a_tracked_file(self):
        repo_root = Path(__file__).resolve().parents[3]
        patterns = {path for group in COMPONENTS.values() for path in group}
        patterns.update(INFRA_PATHS)

        # Git can list the complete tree even in component CI's sparse checkout.
        tracked = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

        # Same matcher as component_ci.matches_paths, so a pattern that matches
        # nothing here can never trigger its component.
        unmatched = sorted(
            p for p in patterns if not any(fnmatch.fnmatch(f, p) for f in tracked)
        )
        self.assertEqual(
            unmatched, [], f"Component CI patterns matching no file: {unmatched}"
        )


if __name__ == "__main__":
    unittest.main()
