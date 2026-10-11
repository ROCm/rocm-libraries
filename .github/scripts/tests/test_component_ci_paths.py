# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that literal component CI paths still name files in the repository."""

import subprocess
import sys
import unittest
from glob import has_magic
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component_ci import COMPONENTS, INFRA_PATHS


class ComponentCIPathsTest(unittest.TestCase):
    def test_literal_paths_exist_in_head_tree(self):
        repo_root = Path(__file__).resolve().parents[3]
        paths = {path for patterns in COMPONENTS.values() for path in patterns}
        paths.update(INFRA_PATHS)

        # Git can inspect the complete tree even in component CI's sparse checkout.
        missing = sorted(
            path
            for path in paths
            if not has_magic(path)
            and subprocess.run(
                ["git", "cat-file", "-e", f"HEAD:{path}"],
                cwd=repo_root,
                capture_output=True,
                check=False,
            ).returncode
        )
        self.assertEqual(missing, [], f"Missing literal component CI paths: {missing}")


if __name__ == "__main__":
    unittest.main()
