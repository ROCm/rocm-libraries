#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Tests of filter_lcov_exclusions.py on small source files and reports."""

import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "filter_lcov_exclusions.py"
)


class FilterLcovExclusions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.output = os.path.join(self.dir.name, "out.info")

    def source(self, lines):
        path = os.path.join(self.dir.name, "src.cpp")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def run_filter(self, source, nlines, *options):
        """Filter a report with a hit line record for every line of source."""
        report = os.path.join(self.dir.name, "in.info")
        with open(report, "w", encoding="utf-8") as f:
            f.write(f"TN:\nSF:{source}\n")
            f.writelines(f"DA:{n},1\n" for n in range(1, nlines + 1))
            f.write("end_of_record\n")
        return subprocess.run(
            [sys.executable, SCRIPT, *options, report, self.output],
            capture_output=True,
            text=True,
        )

    def kept_lines(self):
        with open(self.output, encoding="utf-8") as f:
            return [int(r[3:].split(",")[0]) for r in f if r.startswith("DA:")]

    def assert_fails(self, result, message):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(os.path.exists(self.output))

    def test_balanced_markers_exclude_their_lines(self):
        lines = [
            "int a;",
            "int b; // LCOV_EXCL_START",
            "int c;",
            "int d; // LCOV_EXCL_STOP",
            "int e; // LCOV_EXCL_START LCOV_EXCL_STOP",
            "int f; // LCOV_EXCL_LINE",
            "int g;",
        ]
        result = self.run_filter(self.source(lines), len(lines))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kept_lines(), [1, 7])

    def test_no_mpi_blocks_are_tracked_apart_from_lcov_blocks(self):
        lines = [
            "int a; // LCOV_EXCL_START",
            "int b; // COVERAGE_EXCL_NO_MPI_START",
            "int c; // LCOV_EXCL_STOP",
            "int d; // COVERAGE_EXCL_NO_MPI_STOP",
            "int e;",
        ]
        source = self.source(lines)
        result = self.run_filter(source, len(lines))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kept_lines(), [4, 5])
        result = self.run_filter(source, len(lines), "--no-mpi")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kept_lines(), [5])

    def test_a_start_without_a_stop_fails(self):
        for marker in ("LCOV_EXCL", "LCOV_EXCL_BR", "COVERAGE_EXCL_NO_MPI"):
            with self.subTest(marker=marker):
                lines = ["int a;", f"int b; // {marker}_START", "int c;"]
                result = self.run_filter(self.source(lines), len(lines))
                self.assert_fails(result, f"src.cpp:2: {marker}_START without a STOP")

    def test_a_start_inside_an_open_block_fails(self):
        # A misspelled STOP lets the block run on to the next STOP
        lines = [
            "int a; // LCOV_EXCL_START",
            "int b; // LCOV_EXCL_STop",
            "int c; // LCOV_EXCL_START",
            "int d; // LCOV_EXCL_STOP",
        ]
        result = self.run_filter(self.source(lines), len(lines))
        self.assert_fails(
            result, "src.cpp:3: LCOV_EXCL_START inside the block opened at line 1"
        )

    def test_a_stop_without_a_start_fails(self):
        lines = ["int a;", "int b; // LCOV_EXCL_BR_STOP"]
        result = self.run_filter(self.source(lines), len(lines))
        self.assert_fails(result, "src.cpp:2: LCOV_EXCL_BR_STOP without a START")

    def test_a_misspelled_marker_fails(self):
        lines = ["int a; // LCOV_EXCL_Line"]
        result = self.run_filter(self.source(lines), len(lines))
        self.assert_fails(result, "src.cpp:1: unknown marker LCOV_EXCL_Line")

    def test_a_missing_source_fails_and_removes_an_old_report(self):
        with open(self.output, "w", encoding="utf-8") as f:
            f.write("old report\n")
        missing = os.path.join(self.dir.name, "missing.cpp")
        result = self.run_filter(missing, 1)
        self.assert_fails(result, "missing.cpp: source file not found")


if __name__ == "__main__":
    unittest.main()
