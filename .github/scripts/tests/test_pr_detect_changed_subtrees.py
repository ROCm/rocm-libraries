#!/usr/bin/env python3
"""
Baseline unit tests for pr_detect_changed_subtrees.py.

These tests document the CURRENT behavior of every testable function,
including known-buggy behavior. Tests that capture a known bug are marked
with a BUG comment. When a bug is fixed:
  1. Update the test to assert the DESIRED behavior (it will fail).
  2. Fix the production code until the test passes.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pr_detect_changed_subtrees as sut
from repo_config_model import RepoEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    name: str,
    category: str = "projects",
    auto_subtree_pull: bool = False,
    auto_subtree_push: bool = False,
    monorepo_source_of_truth: bool = False,
) -> RepoEntry:
    return RepoEntry(
        name=name,
        url=f"ROCm/{name}",
        branch="develop",
        category=category,
        auto_subtree_pull=auto_subtree_pull,
        auto_subtree_push=auto_subtree_push,
        monorepo_source_of_truth=monorepo_source_of_truth,
    )


FULL_CONFIG = [
    make_entry(
        "rocblas",
        auto_subtree_pull=True,
        auto_subtree_push=True,
        monorepo_source_of_truth=True,
    ),
    make_entry(
        "hipcub",
        auto_subtree_pull=False,
        auto_subtree_push=True,
        monorepo_source_of_truth=True,
    ),
    make_entry(
        "tensile",
        category="shared",
        auto_subtree_pull=False,
        auto_subtree_push=False,
        monorepo_source_of_truth=False,
    ),
]


# ---------------------------------------------------------------------------
# get_valid_prefixes
# ---------------------------------------------------------------------------


class GetValidPrefixesTest(unittest.TestCase):

    def test_no_filters_returns_all_prefixes(self):
        result = sut.get_valid_prefixes(FULL_CONFIG)
        self.assertEqual(
            result, {"projects/rocblas", "projects/hipcub", "shared/tensile"}
        )

    def test_require_auto_pull_excludes_false_entries(self):
        result = sut.get_valid_prefixes(FULL_CONFIG, require_auto_pull=True)
        self.assertEqual(result, {"projects/rocblas"})

    def test_require_auto_push_excludes_false_entries(self):
        result = sut.get_valid_prefixes(FULL_CONFIG, require_auto_push=True)
        self.assertEqual(result, {"projects/rocblas", "projects/hipcub"})

    def test_require_monorepo_source_excludes_false_entries(self):
        result = sut.get_valid_prefixes(FULL_CONFIG, require_monorepo_source=True)
        self.assertEqual(result, {"projects/rocblas", "projects/hipcub"})

    def test_multiple_filters_are_and_combined(self):
        result = sut.get_valid_prefixes(
            FULL_CONFIG,
            require_auto_pull=True,
            require_auto_push=True,
            require_monorepo_source=True,
        )
        self.assertEqual(result, {"projects/rocblas"})

    def test_push_and_monorepo_source_combined(self):
        result = sut.get_valid_prefixes(
            FULL_CONFIG, require_auto_push=True, require_monorepo_source=True
        )
        self.assertEqual(result, {"projects/rocblas", "projects/hipcub"})

    def test_empty_config_returns_empty_set(self):
        result = sut.get_valid_prefixes([])
        self.assertEqual(result, set())

    def test_all_filters_false_is_equivalent_to_no_filter(self):
        result = sut.get_valid_prefixes(
            FULL_CONFIG,
            require_auto_pull=False,
            require_auto_push=False,
            require_monorepo_source=False,
        )
        self.assertEqual(result, sut.get_valid_prefixes(FULL_CONFIG))

    def test_shared_category_prefix_is_included(self):
        result = sut.get_valid_prefixes(FULL_CONFIG)
        self.assertIn("shared/tensile", result)


# ---------------------------------------------------------------------------
# find_matched_subtrees
# ---------------------------------------------------------------------------


class FindMatchedSubtreesTest(unittest.TestCase):

    PREFIXES = {"projects/rocblas", "projects/hipcub"}

    def test_file_in_matched_subtree_returns_that_subtree(self):
        result = sut.find_matched_subtrees(
            ["projects/rocblas/src/foo.cpp"], self.PREFIXES
        )
        self.assertEqual(result, ["projects/rocblas"])

    def test_multiple_files_same_subtree_deduplicated(self):
        result = sut.find_matched_subtrees(
            ["projects/rocblas/src/a.cpp", "projects/rocblas/include/b.h"],
            self.PREFIXES,
        )
        self.assertEqual(result, ["projects/rocblas"])

    def test_files_across_multiple_subtrees_returns_all_matched(self):
        result = sut.find_matched_subtrees(
            ["projects/rocblas/src/a.cpp", "projects/hipcub/src/b.cpp"],
            self.PREFIXES,
        )
        self.assertEqual(result, ["projects/hipcub", "projects/rocblas"])

    def test_result_is_sorted(self):
        result = sut.find_matched_subtrees(
            ["projects/rocblas/a.cpp", "projects/hipcub/b.cpp"],
            self.PREFIXES,
        )
        self.assertEqual(result, sorted(result))

    def test_top_level_file_with_no_slash_is_excluded(self):
        result = sut.find_matched_subtrees(["README.md"], self.PREFIXES)
        self.assertEqual(result, [])

    def test_file_with_exactly_two_components_is_matched(self):
        # A path like "projects/rocblas" (no trailing filename) still matches.
        result = sut.find_matched_subtrees(["projects/rocblas"], self.PREFIXES)
        self.assertEqual(result, ["projects/rocblas"])

    def test_file_not_in_valid_prefixes_is_excluded(self):
        result = sut.find_matched_subtrees(
            ["projects/rocsolver/src/foo.cpp"], self.PREFIXES
        )
        self.assertEqual(result, [])

    def test_empty_changed_files_returns_empty_list(self):
        result = sut.find_matched_subtrees([], self.PREFIXES)
        self.assertEqual(result, [])

    def test_empty_valid_prefixes_returns_empty_list(self):
        result = sut.find_matched_subtrees(["projects/rocblas/src/foo.cpp"], set())
        self.assertEqual(result, [])

    def test_mix_of_matched_unmatched_and_top_level_files(self):
        result = sut.find_matched_subtrees(
            ["README.md", "projects/rocblas/src/a.cpp", "projects/unknown/b.cpp"],
            self.PREFIXES,
        )
        self.assertEqual(result, ["projects/rocblas"])

    def test_deeply_nested_file_uses_first_two_path_components(self):
        result = sut.find_matched_subtrees(
            ["projects/rocblas/src/deep/nested/file.cpp"], self.PREFIXES
        )
        self.assertEqual(result, ["projects/rocblas"])


# ---------------------------------------------------------------------------
# output_subtrees
# ---------------------------------------------------------------------------


class OutputSubtreesTest(unittest.TestCase):

    def test_dry_run_does_not_write_to_any_file(self):
        # In dry-run mode the function should only log; it must not touch
        # GITHUB_OUTPUT or any other file.
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "/nonexistent/path"}):
            # Should not raise even though the file doesn't exist.
            sut.output_subtrees(["projects/rocblas"], dry_run=True)

    def test_dry_run_with_empty_list_does_not_raise(self):
        sut.output_subtrees([], dry_run=True)

    def test_non_dry_run_writes_to_github_output(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": tmp_path}):
                sut.output_subtrees(
                    ["projects/rocblas", "projects/hipcub"], dry_run=False
                )
            content = Path(tmp_path).read_text()
            self.assertIn("subtrees<<EOF", content)
            self.assertIn("projects/rocblas", content)
            self.assertIn("projects/hipcub", content)
            self.assertIn("EOF", content)
        finally:
            os.unlink(tmp_path)

    def test_non_dry_run_appends_to_existing_github_output(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("existing=value\n")
            tmp_path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": tmp_path}):
                sut.output_subtrees(["projects/rocblas"], dry_run=False)
            content = Path(tmp_path).read_text()
            self.assertIn("existing=value", content)
            self.assertIn("projects/rocblas", content)
        finally:
            os.unlink(tmp_path)

    def test_non_dry_run_without_github_output_exits_nonzero(self):
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                sut.output_subtrees(["projects/rocblas"], dry_run=False)
        self.assertEqual(ctx.exception.code, 1)

    def test_empty_subtrees_writes_empty_body_to_github_output(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": tmp_path}):
                sut.output_subtrees([], dry_run=False)
            content = Path(tmp_path).read_text()
            self.assertIn("subtrees<<EOF", content)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
