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

"""Tests for the testing.md metrics scripts. Run: python3 test_testing_md_metrics.py"""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_fix_prs as fixes  # noqa: E402
import sync_testing_md_metrics as sync  # noqa: E402

LIB = "projects/rocsparse/library/src/level2/rocsparse_csrmv.cpp"
UT = "projects/rocsparse/clients/unittests/unit_test_csrmv.cpp"
YAML = "projects/rocsparse/clients/tests/test_csrmv.yaml"
UT_CMAKE = "projects/rocsparse/clients/unittests/CMakeLists.txt"


def f(name, additions=5):
    return {"filename": name, "additions": additions}


class TitleRule(unittest.TestCase):
    def test_accepts_the_three_title_forms(self):
        for title in ("fix(rocsparse): guard a null info",
                      "[rocSPARSE] Fix ellmv integer overflow",
                      "Fix rocsparse_sptrsv_get_output singularity outputs"):
            self.assertTrue(fixes.TITLE.match(title), title)

    def test_rejects_other_titles(self):
        for title in ("test(rocsparse): add unit tests",
                      "fix(hipsparse): something else",
                      "perf(rocsparse): tune coomm"):
            self.assertFalse(fixes.TITLE.match(title), title)


class Classify(unittest.TestCase):
    def test_unit_test_counts_as_both(self):
        self.assertEqual(fixes.classify([f(LIB), f(UT)]), (True, True, True))

    def test_yaml_only_is_any_test(self):
        self.assertEqual(fixes.classify([f(LIB), f(YAML)]), (True, False, True))

    def test_cmake_in_unittests_is_not_a_test(self):
        self.assertEqual(fixes.classify([f(LIB), f(UT_CMAKE)]), (True, False, False))

    def test_removed_lines_do_not_count(self):
        self.assertEqual(fixes.classify([f(LIB), f(UT, 0)]), (True, False, False))

    def test_no_library_change(self):
        self.assertEqual(fixes.classify([f(UT)]), (False, True, True))


class Apply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        self.addCleanup(os.unlink, self.tmp.name)
        self._render = sync.render
        sync.render = lambda today: "GENERATED " + today
        self.addCleanup(setattr, sync, "render", self._render)

    def write(self, text):
        self.tmp.write(text)
        self.tmp.close()

    def test_replaces_only_between_markers(self):
        self.write("before\n<!-- metrics:start -->\nold\n<!-- metrics:end -->\nafter\n")
        sync.apply(self.tmp.name, "2026-01-01")
        with open(self.tmp.name) as fh:
            result = fh.read()
        self.assertEqual(
            result,
            "before\n<!-- metrics:start -->\nGENERATED 2026-01-01\n<!-- metrics:end -->\nafter\n")

    def test_missing_markers_is_an_error(self):
        self.write("no markers here\n")
        with self.assertRaises(SystemExit):
            sync.apply(self.tmp.name, "2026-01-01")


class CoverageSync(unittest.TestCase):
    """sync_coverage with Codecov faked out."""

    def upload(self, flag, cov, files=541, kind="uploaded", build=382):
        return {"flags": [flag], "upload_type": kind, "build_code": build,
                "totals": {"coverage": cov, "files": files}}

    def both(self, a, b, **k):
        return [self.upload("rocSPARSE", a, **k), self.upload("rocSPARSE-unit", b, **k)]

    def run_sync(self, commits, uploads, rows=(("2026-11-19", "85.81", "82.64", "381"),)):
        """`uploads` maps a commit to a list of pages (each a list of uploads)."""
        def fake_get(url):
            if "/commits?branch=develop" in url:
                return {"results": commits if url.endswith("page=1") else []}
            sha = url.split("/commits/")[1].split("/")[0]
            pages = uploads[sha]
            n = int(url.split("page=")[1]) if "page=" in url else 0
            return {"results": pages[n],
                    "next": f"{url.split('?')[0]}?page={n + 1}" if n + 1 < len(pages) else None}

        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        self.addCleanup(os.unlink, tmp.name)
        w = csv.writer(tmp)
        w.writerow(["date", "develop_combined", "unit_only", "build"])
        w.writerows(rows)
        tmp.close()
        old = (sync._get, sync.CSV_PATH)
        sync._get, sync.CSV_PATH = fake_get, tmp.name
        self.addCleanup(lambda: setattr(sync, "_get", old[0]))
        self.addCleanup(lambda: setattr(sync, "CSV_PATH", old[1]))
        sync.sync_coverage()
        with open(tmp.name, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_guards_and_per_day_rules(self):
        commits = [  # newest first, as Codecov returns them
            {"commitid": "c5", "timestamp": "2026-11-22T10:00:00Z"},
            {"commitid": "c4", "timestamp": "2026-11-21T20:00:00Z"},
            {"commitid": "c3", "timestamp": "2026-11-21T05:00:00Z"},
            {"commitid": "c2", "timestamp": "2026-11-20T23:00:00Z"},
            {"commitid": "c1", "timestamp": "2026-11-20T10:00:00Z"},
            {"commitid": "c0", "timestamp": "2026-11-18T10:00:00Z"},
        ]
        uploads = {
            "c5": [self.both(85.0, 82.0, kind="carriedforward")],
            "c4": [self.both(99.0, 99.0, files=99)],  # partial report, skipped
            "c3": [self.both(95.0, 91.0, files=100), []],  # exactly MIN_FILES counts
            # a duplicate with the higher value listed first: the higher one wins
            "c2": [self.both(85.81, 82.64, build=382) + self.both(85.0, 82.0, build=383)],
            # earlier the same day with higher values: the later commit wins
            "c1": [self.both(90.0, 88.0, build=381)],
            "c0": [self.both(1.0, 1.0)],  # before the last recorded day
        }
        rows = self.run_sync(commits, uploads)
        self.assertEqual([r["date"] for r in rows],
                         ["2026-11-19", "2026-11-20", "2026-11-21"])
        self.assertEqual((rows[1]["develop_combined"], rows[1]["unit_only"], rows[1]["build"]),
                         ("85.81", "82.64", "382"))
        self.assertEqual((rows[2]["develop_combined"], rows[2]["unit_only"]), ("95.0", "91.0"))

    def test_last_recorded_day_is_refreshed(self):
        commits = [{"commitid": "c1", "timestamp": "2026-11-19T22:00:00Z"}]
        uploads = {"c1": [self.both(86.5, 83.0, build=385)]}
        rows = self.run_sync(commits, uploads)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["develop_combined"], rows[0]["build"]), ("86.5", "385"))

    def test_follows_upload_pagination(self):
        commits = [{"commitid": "c1", "timestamp": "2026-11-20T10:00:00Z"}]
        uploads = {"c1": [[self.upload("rocSPARSE", 85.0)],
                          [self.upload("rocSPARSE-unit", 82.0)]]}
        rows = self.run_sync(commits, uploads)
        self.assertEqual([r["date"] for r in rows], ["2026-11-19", "2026-11-20"])

    def test_upload_without_totals_is_skipped(self):
        commits = [{"commitid": "c1", "timestamp": "2026-11-20T10:00:00Z"}]
        pending = {"flags": ["rocSPARSE"], "upload_type": "uploaded", "totals": None}
        uploads = {"c1": [[pending] + self.both(85.0, 82.0)]}
        rows = self.run_sync(commits, uploads)
        self.assertEqual(rows[-1]["develop_combined"], "85.0")


if __name__ == "__main__":
    unittest.main()
