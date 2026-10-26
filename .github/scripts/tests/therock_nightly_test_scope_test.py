import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_nightly_test_scope as scope


def make_entry(name: str, category: str) -> dict:
    return {
        "name": name,
        "url": f"ROCm/{name}",
        "branch": "develop",
        "category": category,
        "auto_subtree_pull": False,
        "auto_subtree_push": True,
        "monorepo_source_of_truth": True,
    }


def write_config(tmpdir: str, entries: List[dict]) -> Path:
    config_path = Path(tmpdir) / "repos-config.json"
    config_path.write_text(json.dumps({"repositories": entries}), encoding="utf-8")
    return config_path


class NightlyTestScopeTest(unittest.TestCase):
    def test_returns_sorted_category_name_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_config(
                tmpdir,
                [
                    make_entry("rocblas", "projects"),
                    make_entry("rocsparse", "projects"),
                    make_entry("hipblas-common", "shared"),
                ],
            )
            self.assertEqual(
                scope.get_test_projects(config_path),
                [
                    "projects/rocblas",
                    "projects/rocsparse",
                    "shared/hipblas-common",
                ],
            )

    def test_includes_entries_regardless_of_subtree_sync_flags(self):
        # The nightly scope is "everything this repo owns", so entries are not
        # filtered by the auto_subtree_pull/push flags that the sync tooling uses.
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = make_entry("rocprim", "projects")
            entry["auto_subtree_push"] = False
            entry["monorepo_source_of_truth"] = False
            config_path = write_config(tmpdir, [entry])
            self.assertEqual(scope.get_test_projects(config_path), ["projects/rocprim"])

    def test_empty_config_fails_instead_of_testing_everything(self):
        # An empty changed_projects means "test everything" to TheRock.
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_config(tmpdir, [])
            with self.assertRaises(SystemExit):
                scope.main(["--config", os.fspath(config_path)])

    def test_real_config_is_non_empty(self):
        projects = scope.get_test_projects(scope.DEFAULT_CONFIG_PATH)
        self.assertTrue(projects)
        for project in projects:
            self.assertIn("/", project)

    def test_step_summary_lists_each_subtree_on_its_own_line(self):
        summary = scope.format_step_summary(["projects/rocblas", "shared/rocroller"])
        self.assertIn("### Nightly test scope", summary)
        self.assertIn("projects/rocblas\nshared/rocroller", summary)


if __name__ == "__main__":
    unittest.main()
