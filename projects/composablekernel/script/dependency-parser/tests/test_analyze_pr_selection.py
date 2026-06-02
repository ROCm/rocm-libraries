#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for analyze_pr_selection.py (pure functions, no I/O)."""

import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import analyze_pr_selection as aps  # noqa: E402


CTEST_N_OUTPUT = textwrap.dedent("""\
    Test project /build
      Test #1: test_gemm
      Test #2: test_conv
      Test #3: test_fmha
      Test #4: test_ck_tile_streamk_generate_test_files
""")


class TestIsCodeFile(unittest.TestCase):
    def test_headers_and_sources_are_code(self):
        for ext in [".hpp", ".h", ".cpp", ".cu", ".hip", ".inc", ".tpp"]:
            self.assertTrue(aps.is_code_file(f"path/file{ext}"), ext)

    def test_non_code(self):
        for f in ["CMakeLists.txt", "README.md", "script.py", "config.yaml"]:
            self.assertFalse(aps.is_code_file(f), f)

    def test_case_insensitive(self):
        self.assertTrue(aps.is_code_file("Kernel.HPP"))


class TestLoadCtestTests(unittest.TestCase):
    def test_parses_standard_format(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(CTEST_N_OUTPUT)
            name = f.name
        try:
            tests = aps.load_ctest_tests(name)
        finally:
            os.unlink(name)
        self.assertIn("test_gemm", tests)
        self.assertIn("test_fmha", tests)
        self.assertIn("test_ck_tile_streamk_generate_test_files", tests)
        self.assertEqual(len(tests), 4)

    def test_empty_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("No tests.\n")
            name = f.name
        try:
            tests = aps.load_ctest_tests(name)
        finally:
            os.unlink(name)
        self.assertEqual(tests, set())


class TestAnalyzePr(unittest.TestCase):
    def _f2e(self):
        return {
            "include/ck/gemm.hpp": ["bin/test_gemm", "bin/example_gemm"],
            "library/src/conv.cpp": ["bin/test_conv"],
            "include/dead_header.hpp": [],  # in depmap but no exe
        }

    def _ctest(self):
        return {"test_gemm", "test_conv", "test_fmha"}

    def _pr(self, files):
        return {"number": 42, "title": "Test PR", "files": files}

    def test_basic_selection(self):
        pr = self._pr([
            "projects/composablekernel/include/ck/gemm.hpp",
            "projects/composablekernel/library/src/conv.cpp",
        ])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        self.assertEqual(r["n_ck_files"], 2)
        self.assertIn("bin/test_gemm", r["selected"])
        self.assertIn("bin/test_conv", r["selected"])
        self.assertNotIn("bin/example_gemm", r["selected"])  # not in ctest

    def test_non_ck_files_excluded(self):
        pr = self._pr([
            "projects/rocsolver/include/foo.hpp",
            "projects/composablekernel/include/ck/gemm.hpp",
        ])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        self.assertEqual(r["n_ck_files"], 1)
        self.assertEqual(r["files_outside_composablekernel"], ["projects/rocsolver/include/foo.hpp"])

    def test_file_not_in_depmap_flagged(self):
        pr = self._pr(["projects/composablekernel/include/ck/new_kernel.hpp"])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        self.assertIn("include/ck/new_kernel.hpp", r["flags"]["code_files_not_in_depmap"])
        self.assertEqual(r["n_selected"], 0)

    def test_dead_header_flagged(self):
        pr = self._pr(["projects/composablekernel/include/dead_header.hpp"])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        self.assertIn("include/dead_header.hpp", r["flags"]["code_files_with_no_dependents"])

    def test_noncode_files_flagged(self):
        pr = self._pr([
            "projects/composablekernel/CMakeLists.txt",
            "projects/composablekernel/README.md",
        ])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        self.assertEqual(r["n_code_files"], 0)
        self.assertEqual(len(r["flags"]["noncode_files"]), 2)
        self.assertEqual(r["n_selected"], 0)

    def test_dropped_non_ctest_reported(self):
        pr = self._pr(["projects/composablekernel/include/ck/gemm.hpp"])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        # example_gemm is in depmap but not in ctest
        self.assertIn("bin/example_gemm", r["dropped_non_ctest"])

    def test_empty_pr(self):
        r = aps.analyze_pr(self._f2e(), self._ctest(), {"number": 0, "title": "", "files": []})
        self.assertEqual(r["n_selected"], 0)
        self.assertEqual(r["n_ck_files"], 0)

    def test_per_file_metadata(self):
        pr = self._pr(["projects/composablekernel/include/ck/gemm.hpp"])
        r = aps.analyze_pr(self._f2e(), self._ctest(), pr)
        pf = r["per_file"]["include/ck/gemm.hpp"]
        self.assertTrue(pf["is_code"])
        self.assertTrue(pf["in_depmap"])
        self.assertEqual(pf["n_deps"], 2)


class TestSummaryLine(unittest.TestCase):
    def test_no_exceptions_on_typical_result(self):
        result = {
            "pr": 123,
            "title": "A" * 60,  # longer than 48 -> truncated
            "n_selected": 5,
            "n_expected_dependents": 8,
            "n_code_files": 3,
            "dropped_non_ctest": ["a", "b"],
            "flags": {
                "code_files_with_no_dependents": [],
                "code_files_not_in_depmap": ["x.hpp"],
                "noncode_files": [],
            },
        }
        line = aps.summary_line(result)
        self.assertIn("PR #123", line)
        self.assertIn("sel=5", line)
        self.assertIn("not_in_map=1", line)


if __name__ == "__main__":
    unittest.main()
