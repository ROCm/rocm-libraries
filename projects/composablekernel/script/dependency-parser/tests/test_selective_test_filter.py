#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for selective_test_filter output emission.

Covers the sibling ``tests_to_run.txt`` file that smart_test.sh consumes via
``ctest --tests-from-file``: it must be created next to the JSON and contain the
expected basename test names, one per line.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestExportSelection(unittest.TestCase):
    """Tests for export_selection()'s JSON + sibling .txt emission."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_json = os.path.join(self.temp_dir, "tests_to_run.json")
        self.list_file = os.path.join(self.temp_dir, "tests_to_run.txt")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_emits_sibling_txt_with_basenames(self):
        """The .txt is created beside the JSON, one basename per line."""
        from selective_test_filter import export_selection

        tests = ["bin/test_gemm", "bin/test_conv", "test_reduce"]
        returned = export_selection(tests, {"a.cpp"}, self.output_json)

        self.assertEqual(returned, self.list_file)
        self.assertTrue(os.path.exists(self.list_file))

        lines = Path(self.list_file).read_text().splitlines()
        self.assertEqual(lines, ["test_gemm", "test_conv", "test_reduce"])

    def test_txt_lines_match_json_executables_basenames(self):
        """The .txt list is exactly the basenames of the JSON executables."""
        from selective_test_filter import export_selection

        tests = ["a/b/test_gemm", "c/test_conv"]
        export_selection(tests, set(), self.output_json)

        with open(self.output_json) as f:
            payload = json.load(f)
        expected = [os.path.basename(t) for t in payload["executables"]]

        lines = Path(self.list_file).read_text().splitlines()
        self.assertEqual(lines, expected)

    def test_empty_selection_writes_empty_txt(self):
        """No tests -> an empty .txt is still created (no spurious lines)."""
        from selective_test_filter import export_selection

        export_selection([], {"only_header.hpp"}, self.output_json)

        self.assertTrue(os.path.exists(self.list_file))
        self.assertEqual(Path(self.list_file).read_text(), "")

    def test_txt_filename_follows_output_json_stem(self):
        """The sibling .txt derives its name from the JSON stem, not a literal."""
        from selective_test_filter import export_selection

        custom_json = os.path.join(self.temp_dir, "selection.json")
        returned = export_selection(["test_gemm"], set(), custom_json)

        self.assertEqual(returned, os.path.join(self.temp_dir, "selection.txt"))
        self.assertTrue(os.path.exists(returned))


if __name__ == "__main__":
    unittest.main()
