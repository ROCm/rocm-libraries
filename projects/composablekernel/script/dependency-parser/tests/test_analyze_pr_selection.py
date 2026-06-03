#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for analyze_pr_selection.py."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))
import analyze_pr_selection as aps  # noqa: E402


CTEST_N_OUTPUT = textwrap.dedent("""\
    Test project /build
      Test #1: test_gemm
      Test #2: test_conv
      Test #3: test_fmha
      Test #4: test_ck_tile_streamk_generate_test_files
""")


class TestDepmapStripPrefix(unittest.TestCase):
    def test_no_metadata_defaults_to_project_root(self):
        # No repo metadata -> documented project-root convention.
        self.assertEqual(aps.depmap_strip_prefix({"file_to_executables": {}}),
                         "projects/composablekernel/")

    def test_project_root_workspace(self):
        d = {"repo": {"workspace_root": "/x/projects/composablekernel"}}
        self.assertEqual(aps.depmap_strip_prefix(d), "projects/composablekernel/")

    def test_repo_root_workspace_strips_nothing(self):
        # workspace_root is the repo root -> keys already repo-root-relative.
        d = {"repo": {"workspace_root": "/tmp/abc/ck"}}
        self.assertEqual(aps.depmap_strip_prefix(d), "")

    def test_monorepo_type(self):
        d = {"repo": {"type": "monorepo", "project": "composablekernel"}}
        self.assertEqual(aps.depmap_strip_prefix(d), "projects/composablekernel/")


class TestAnalyzeAdaptsToRoot(unittest.TestCase):
    PR = {"number": 7, "title": "t",
          "files": ["projects/composablekernel/include/ck/gemm.hpp"]}
    CTESTS = {"test_gemm"}

    def test_repo_root_depmap_matches(self):
        # repo-root keys carry the projects/ prefix; strip_prefix="" keeps the
        # full PR path so it matches.
        f2e = {"projects/composablekernel/include/ck/gemm.hpp": ["bin/test_gemm"]}
        r = aps.analyze_pr(f2e, self.CTESTS, self.PR, strip_prefix="")
        self.assertEqual(r["selected"], ["bin/test_gemm"])
        self.assertEqual(r["flags"]["code_files_not_in_depmap"], [])

    def test_project_root_depmap_matches(self):
        f2e = {"include/ck/gemm.hpp": ["bin/test_gemm"]}
        r = aps.analyze_pr(f2e, self.CTESTS, self.PR,
                           strip_prefix="projects/composablekernel/")
        self.assertEqual(r["selected"], ["bin/test_gemm"])

    def test_wrong_root_flags_unmapped(self):
        # repo-root keys but we (wrongly) strip the prefix -> no match, surfaced
        # as an unmapped code file rather than silently selecting nothing.
        f2e = {"projects/composablekernel/include/ck/gemm.hpp": ["bin/test_gemm"]}
        r = aps.analyze_pr(f2e, self.CTESTS, self.PR,
                           strip_prefix="projects/composablekernel/")
        self.assertEqual(r["selected"], [])
        self.assertEqual(r["flags"]["code_files_not_in_depmap"],
                         ["include/ck/gemm.hpp"])


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


class TestNormalizeAndLoad(unittest.TestCase):
    def test_normalize_raw_gh_shape(self):
        # gh pr view --json files -> files is a list of {path, ...} objects.
        data = {"number": 7, "title": "t", "files": [{"path": "a/b.cpp", "additions": 3}]}
        self.assertEqual(aps._normalize_pr(data),
                         {"number": 7, "title": "t", "files": ["a/b.cpp"]})

    def test_normalize_our_shape(self):
        data = {"number": 7, "title": "t", "files": ["a/b.cpp", "c/d.hpp"]}
        self.assertEqual(aps._normalize_pr(data)["files"], ["a/b.cpp", "c/d.hpp"])

    def test_load_pr_file_normalizes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"number": 9, "title": "x", "files": [{"path": "p.hpp"}]}, f)
            name = f.name
        try:
            pr = aps.load_pr_file(name)
        finally:
            os.unlink(name)
        self.assertEqual(pr["files"], ["p.hpp"])


class TestFetchPr(unittest.TestCase):
    def test_fetch_parses_gh_json(self):
        gh_out = json.dumps({
            "number": 7964, "title": "Smart build",
            "files": [{"path": "projects/composablekernel/x.hpp"}, {"path": "README.md"}],
        })
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=gh_out)
        with mock.patch("analyze_pr_selection.subprocess.run", return_value=completed):
            pr = aps.fetch_pr(7964)
        self.assertEqual(pr["number"], 7964)
        self.assertEqual(pr["files"],
                         ["projects/composablekernel/x.hpp", "README.md"])

    def test_fetch_missing_gh_raises_runtimeerror(self):
        with mock.patch("analyze_pr_selection.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaises(RuntimeError):
                aps.fetch_pr(1)

    def test_fetch_gh_failure_raises_runtimeerror(self):
        err = subprocess.CalledProcessError(1, "gh", stderr="no such PR")
        with mock.patch("analyze_pr_selection.subprocess.run", side_effect=err):
            with self.assertRaises(RuntimeError):
                aps.fetch_pr(1)


class TestMainOffline(unittest.TestCase):
    def _setup(self, tmp):
        depmap = Path(tmp) / "dep.json"
        depmap.write_text(json.dumps({"file_to_executables": {
            "include/ck/gemm.hpp": ["bin/test_gemm", "bin/example_gemm"],
        }}))
        ctest = Path(tmp) / "ctest.txt"
        ctest.write_text(CTEST_N_OUTPUT)
        prf = Path(tmp) / "pr.json"
        prf.write_text(json.dumps({
            "number": 100, "title": "demo",
            "files": ["projects/composablekernel/include/ck/gemm.hpp"],
        }))
        return depmap, ctest, prf

    def test_offline_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            depmap, ctest, prf = self._setup(tmp)
            outdir = Path(tmp) / "out"
            summary = Path(tmp) / "summary.json"
            rc = aps.main([
                "--pr-files", str(prf),
                "--depmap", str(depmap), "--ctest", str(ctest),
                "--output-dir", str(outdir), "--summary", str(summary),
            ])
            self.assertEqual(rc, 0)
            per_pr = json.loads((outdir / "pr_100.json").read_text())
            self.assertIn("bin/test_gemm", per_pr["selected"])
            agg = json.loads(summary.read_text())
            self.assertEqual(agg["n_prs"], 1)
            self.assertEqual(agg["prs"][0]["pr"], 100)

    def test_missing_depmap_exit_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ctest, prf = self._setup(tmp)
            rc = aps.main([
                "--pr-files", str(prf),
                "--depmap", str(Path(tmp) / "nope.json"), "--ctest", str(ctest),
            ])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
