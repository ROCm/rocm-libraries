import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_nightly_test_scope as scope


def write_config(tmpdir: str, names: List[tuple]) -> Path:
    entries = [
        {
            "name": name,
            "url": f"ROCm/{name}",
            "branch": "develop",
            "category": category,
            "auto_subtree_pull": False,
            "auto_subtree_push": True,
            "monorepo_source_of_truth": True,
        }
        for category, name in names
    ]
    config_path = Path(tmpdir) / "repos-config.json"
    config_path.write_text(json.dumps({"repositories": entries}), encoding="utf-8")
    return config_path


class NightlyTestScopeTest(unittest.TestCase):
    def test_returns_sorted_category_name_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_config(
                tmpdir,
                [
                    ("projects", "rocsparse"),
                    ("shared", "hipblas-common"),
                    ("projects", "rocblas"),
                ],
            )
            self.assertEqual(
                scope.get_test_projects(config_path),
                ["projects/rocblas", "projects/rocsparse", "shared/hipblas-common"],
            )

    def test_empty_config_fails_instead_of_testing_everything(self):
        # An empty changed_projects means "test everything" to TheRock.
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_config(tmpdir, [])
            with self.assertRaises(SystemExit):
                scope.main(["--config", os.fspath(config_path)])

    def test_real_config_is_non_empty(self):
        self.assertTrue(scope.get_test_projects(scope.DEFAULT_CONFIG_PATH))


if __name__ == "__main__":
    unittest.main()
