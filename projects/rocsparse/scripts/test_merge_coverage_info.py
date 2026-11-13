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

"""Tests for merge_coverage_info.py. Run: python3 test_merge_coverage_info.py"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("merge_coverage_info.py")
SOURCE = "projects/rocsparse/library/src/foo.cpp"


def _section(source: str, body: str) -> str:
    return f"SF:{source}\n{body.rstrip()}\nend_of_record\n"


class MergeCoverageInfoTest(unittest.TestCase):
    def run_merge(self, *texts: str, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            inputs = []
            for index, text in enumerate(texts):
                path = directory_path / f"in{index}.info"
                path.write_text(text, encoding="utf-8")
                inputs.append(str(path))
            output = directory_path / "out.info"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "-o", str(output), *extra, *inputs],
                capture_output=True,
                text=True,
                check=False,
            )
            produced = output.read_text(encoding="utf-8") if output.exists() else ""
            self.last_stdout = completed.stdout
            return completed.returncode, produced, completed.stderr

    def test_union_takes_the_max_hit_and_keeps_lines_from_either_side(self):
        left = _section(
            f"/jenkins/node-a/rocm-libraries/{SOURCE}",
            "FN:10,foo\nFNDA:1,foo\nFNF:9\nFNH:9\nDA:10,4\nDA:11,0\nLF:9\nLH:9",
        )
        right = _section(
            f"/tmp/other/rocm-libraries/{SOURCE}",
            "FN:10,foo\nFNDA:3,foo\nDA:10,1\nDA:11,2\nDA:12,5\nBRDA:11,0,0,-\nBRDA:11,0,1,2",
        )
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn(f"SF:{SOURCE}\n", produced)
        self.assertNotIn("/jenkins/", produced)
        self.assertNotIn("/tmp/other/", produced)
        self.assertIn("DA:10,4\n", produced)
        self.assertIn("DA:11,2\n", produced)
        self.assertIn("DA:12,5\n", produced)
        self.assertIn("FNDA:3,foo\n", produced)
        self.assertIn("FNF:1\n", produced)
        self.assertIn("FNH:1\n", produced)
        self.assertIn("LF:3\n", produced)
        self.assertIn("LH:3\n", produced)
        self.assertIn("BRDA:11,0,0,-\n", produced)
        self.assertIn("BRDA:11,0,1,2\n", produced)
        self.assertIn("BRF:2\n", produced)
        self.assertIn("BRH:1\n", produced)

    def test_branch_not_taken_on_one_side_keeps_the_taken_count(self):
        left = _section(SOURCE, "DA:1,0\nBRDA:4,0,0,-")
        right = _section(SOURCE, "DA:1,0\nBRDA:4,0,0,7")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("BRDA:4,0,0,7\n", produced)
        self.assertIn("BRH:1\n", produced)

    def test_zero_and_dash_branch_stays_unhit(self):
        left = _section(SOURCE, "DA:1,0\nBRDA:4,0,0,-")
        right = _section(SOURCE, "DA:1,0\nBRDA:4,0,0,0")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("BRDA:4,0,0,0\n", produced)
        self.assertIn("BRH:0\n", produced)

    def test_file_present_on_only_one_side_is_kept(self):
        only = _section(
            "projects/rocsparse/library/src/only.cpp",
            "DA:3,1",
        )
        other = _section(SOURCE, "DA:1,1")
        code, produced, stderr = self.run_merge(only, other)
        self.assertEqual(code, 0, stderr)
        self.assertIn("SF:projects/rocsparse/library/src/only.cpp\n", produced)
        self.assertIn("SF:projects/rocsparse/library/src/foo.cpp\n", produced)
        self.assertLess(
            produced.index("SF:projects/rocsparse/library/src/foo.cpp"),
            produced.index("SF:projects/rocsparse/library/src/only.cpp"),
        )

    def test_checksum_follows_the_higher_count(self):
        left = _section(SOURCE, "DA:8,2,aaa")
        right = _section(SOURCE, "DA:8,9,bbb")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("DA:8,9,bbb\n", produced)

    def test_equal_counts_keep_an_existing_checksum(self):
        left = _section(SOURCE, "DA:8,2,aaa")
        right = _section(SOURCE, "DA:8,2")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("DA:8,2,aaa\n", produced)

    def test_function_start_line_conflict_keeps_the_first_and_warns(self):
        left = _section(SOURCE, "FN:10,foo\nFNDA:1,foo\nDA:10,1")
        right = _section(SOURCE, "FN:12,foo\nFNDA:4,foo\nDA:12,1")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("FN:10,foo\n", produced)
        self.assertNotIn("FN:12,foo\n", produced)
        self.assertIn("FNDA:4,foo\n", produced)
        self.assertIn("start line 10 vs 12", stderr)

    def test_path_without_the_anchor_is_kept_and_warned(self):
        text = _section("/opt/rocm/include/header.h", "DA:1,1")
        code, produced, stderr = self.run_merge(text)
        self.assertEqual(code, 0, stderr)
        self.assertIn("SF:/opt/rocm/include/header.h\n", produced)
        self.assertIn("does not start with", stderr)

    def test_second_run_does_not_change_normalized_output(self):
        original = _section(f"/build/tree/{SOURCE}", "DA:2,1\nDA:3,0")
        code, once, stderr = self.run_merge(original)
        self.assertEqual(code, 0, stderr)
        code, twice, stderr = self.run_merge(once)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(once, twice)

    def test_missing_end_of_record_fails(self):
        code, _produced, stderr = self.run_merge(f"SF:{SOURCE}\nDA:1,1\n")
        self.assertNotEqual(code, 0)
        self.assertIn("missing end_of_record", stderr)

    def test_empty_file_fails(self):
        code, _produced, stderr = self.run_merge("")
        self.assertNotEqual(code, 0)
        self.assertIn("no coverage records", stderr)

    def test_unrecognized_record_is_passed_through(self):
        code, produced, stderr = self.run_merge(_section(SOURCE, "DA:1,1\nNOPE:1"))
        self.assertEqual(code, 0, stderr)
        self.assertIn("NOPE:1\n", produced)
        self.assertIn("passed through", stderr)
        self.assertIn("DA:1,1\n", produced)

    def test_mismatched_branch_ids_are_dropped(self):
        left = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,1\nBRDA:4,0,1,-")
        right = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,3")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("dropped these BRDA records", stderr)
        self.assertNotIn("BRDA:", produced)
        self.assertIn("DA:4,1\n", produced)

    def test_dropped_branch_line_stays_dropped_for_a_later_input(self):
        first = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,1\nBRDA:4,0,1,-")
        second = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,3")
        third = _section(SOURCE, "DA:4,2\nBRDA:4,0,0,1\nBRDA:4,0,1,4")
        code, produced, stderr = self.run_merge(first, second, third)
        self.assertEqual(code, 0, stderr)
        self.assertIn("dropped these BRDA records", stderr)
        self.assertNotIn("BRDA:", produced)
        self.assertNotIn("BRF:", produced)
        self.assertIn("DA:4,2\n", produced)

    def test_anchor_is_the_last_matching_component(self):
        nested = "/work/projects/rocsparse/checkout/" + SOURCE
        code, produced, stderr = self.run_merge(_section(nested, "DA:1,1"))
        self.assertEqual(code, 0, stderr)
        self.assertIn(f"SF:{SOURCE}\n", produced)
        self.assertNotIn("does not start with", stderr)

    def test_underscore_in_a_count_is_rejected(self):
        code, _produced, stderr = self.run_merge(_section(SOURCE, "DA:1,1_0"))
        self.assertNotEqual(code, 0)
        self.assertIn("bad count", stderr)

    def test_backslash_path_without_anchor_warns(self):
        code, produced, stderr = self.run_merge(_section(r"C:\src\x.cpp", "DA:1,1"))
        self.assertEqual(code, 0, stderr)
        self.assertIn("SF:C:/src/x.cpp\n", produced)
        self.assertIn("does not start with", stderr)

    def test_fn_end_line_form_is_rejected(self):
        code, _produced, stderr = self.run_merge(
            _section(SOURCE, "FN:10,20,foo\nFNDA:1,foo\nDA:10,1")
        )
        self.assertNotEqual(code, 0)
        self.assertIn("FN end-line form is not accepted", stderr)

    def test_stale_summary_is_recomputed(self):
        text = _section(SOURCE, "DA:1,0\nDA:2,3\nLF:100\nLH:100")
        code, produced, stderr = self.run_merge(text)
        self.assertEqual(code, 0, stderr)
        self.assertIn("LF:2\n", produced)
        self.assertIn("LH:1\n", produced)
        self.assertNotIn("LF:100", produced)
        self.assertIn("0 dropped branch lines", self.last_stdout)

    def test_union_is_independent_of_input_order(self):
        wave64 = _section(
            SOURCE,
            "FN:10,foo\nFN:20,bar\nFNDA:7,foo\nFNDA:0,bar\nDA:4,1\n"
            "BRDA:4,0,0,5\nBRDA:4,0,1,-\nBRDA:4,0,2,2",
        )
        wave32 = _section(
            SOURCE,
            "FN:10,foo\nFN:20,bar\nFNDA:0,foo\nFNDA:0,bar\nDA:4,1\n"
            "BRDA:4,0,0,-\nBRDA:4,0,1,3\nBRDA:4,0,2,6",
        )
        code, forward, stderr = self.run_merge(wave64, wave32)
        self.assertEqual(code, 0, stderr)
        code, backward, stderr = self.run_merge(wave32, wave64)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(forward, backward)
        for record in (
            "FNDA:7,foo\n",
            "FNH:1\n",
            "BRDA:4,0,0,5\n",
            "BRDA:4,0,1,3\n",
            "BRDA:4,0,2,6\n",
            "BRH:3\n",
        ):
            self.assertIn(record, forward)

    def test_same_size_branch_id_mismatch_is_dropped(self):
        left = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,1\nBRDA:4,0,1,-")
        right = _section(SOURCE, "DA:4,1\nBRDA:4,0,0,1\nBRDA:4,1,0,2")
        code, produced, stderr = self.run_merge(left, right)
        self.assertEqual(code, 0, stderr)
        self.assertIn("dropped these BRDA records", stderr)
        self.assertNotIn("BRDA:", produced)
        self.assertIn("DA:4,1\n", produced)
        self.assertIn("1 dropped branch line", self.last_stdout)

    def test_sf_before_end_of_record_is_rejected(self):
        text = f"SF:{SOURCE}\n" + _section(SOURCE, "DA:1,1")
        code, _produced, stderr = self.run_merge(text)
        self.assertNotEqual(code, 0)
        self.assertIn("SF: before end_of_record", stderr)

    def test_strict_rejects_a_branch_drop_and_still_writes_the_file(self):
        left = _section(SOURCE, "DA:4,5\nBRDA:4,0,0,5\nBRDA:4,1,0,0")
        right = _section(
            SOURCE,
            "DA:4,7\nBRDA:4,0,0,5\nBRDA:4,1,0,0\nBRDA:4,2,1,7\nBRDA:4,3,1,0",
        )
        code, produced, stderr = self.run_merge(left, right, extra=("--strict",))
        self.assertNotEqual(code, 0)
        self.assertIn("dropped these BRDA records", stderr)
        self.assertIn("DA:4,7\n", produced)
        self.assertNotIn("BRDA:", produced)
        self.assertIn("1 dropped branch line", self.last_stdout)

    def test_strict_accepts_a_clean_merge(self):
        code, produced, stderr = self.run_merge(
            _section(SOURCE, "DA:1,1"), extra=("--strict",)
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("DA:1,1\n", produced)
        self.assertIn("0 dropped branch lines", self.last_stdout)


if __name__ == "__main__":
    unittest.main()
