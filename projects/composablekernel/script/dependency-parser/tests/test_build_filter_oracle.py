#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the build-filter completeness oracle (pure logic)."""

import sys
import unittest
from pathlib import Path

# filter_oracle.py lives in the dependency-parser dir (parent of tests/).
sys.path.insert(0, str(Path(__file__).parent.parent))
import filter_oracle as bfo  # noqa: E402


class TestParseFailedObjects(unittest.TestCase):
    def test_extracts_object_outputs_from_failed_lines(self):
        text = (
            "[1/9] Building CXX object a/b.cpp.o\n"
            "FAILED: lib/gemm/foo.cpp.o \n"
            "/opt/rocm/bin/amdclang++ ... -c foo.cpp\n"
            "<built-in>: error: ...\n"
            "FAILED: lib/gemm/bar.cpp.o lib/gemm/baz.cpp.o\n"
        )
        self.assertEqual(
            bfo.parse_failed_objects(text),
            {"lib/gemm/foo.cpp.o", "lib/gemm/bar.cpp.o", "lib/gemm/baz.cpp.o"},
        )

    def test_ignores_non_object_failed_tokens(self):
        text = "FAILED: bin/test_gemm\n"  # a link failure, not an object
        self.assertEqual(bfo.parse_failed_objects(text), set())

    def test_empty(self):
        self.assertEqual(bfo.parse_failed_objects(""), set())


class TestExesForObjects(unittest.TestCase):
    def setUp(self):
        self.exe_to_objects = {
            "bin/test_gemm": ["t/test_gemm.cpp.o", "lib/gemm.cpp.o"],
            "bin/test_conv": ["t/test_conv.cpp.o", "lib/conv.cpp.o"],
            "bin/example_gemm": ["e/example_gemm.cpp.o", "lib/gemm.cpp.o"],
        }

    def test_shared_object_hits_multiple_exes(self):
        hit = bfo.exes_for_objects(self.exe_to_objects, {"lib/gemm.cpp.o"})
        self.assertEqual(hit, {"bin/test_gemm", "bin/example_gemm"})

    def test_unique_object(self):
        hit = bfo.exes_for_objects(self.exe_to_objects, {"lib/conv.cpp.o"})
        self.assertEqual(hit, {"bin/test_conv"})

    def test_ctest_intersection_excludes_nontests(self):
        hit = bfo.exes_for_objects(
            self.exe_to_objects, {"lib/gemm.cpp.o"}, ctest_tests={"test_gemm"}
        )
        self.assertEqual(hit, {"bin/test_gemm"})  # example_gemm not in ctest


class TestSelForFile(unittest.TestCase):
    def test_sel_intersects_ctest(self):
        f2e = {"include/gemm.hpp": ["bin/test_gemm", "bin/example_gemm", "bin/ckProfiler"]}
        sel = bfo.sel_for_file(f2e, {"test_gemm", "example_gemm"}, "include/gemm.hpp")
        self.assertEqual(sel, {"bin/test_gemm", "bin/example_gemm"})  # ckProfiler dropped


class TestEvaluate(unittest.TestCase):
    def test_pass_when_true_subset_of_sel(self):
        r = bfo.evaluate("h.hpp", sel={"bin/a", "bin/b"}, true_set={"bin/a"})
        self.assertEqual(r["verdict"], "pass")
        self.assertEqual(r["n_fn"], 0)
        self.assertEqual(r["false_positives"], ["bin/b"])

    def test_fail_with_false_negative(self):
        r = bfo.evaluate("h.hpp", sel={"bin/a"}, true_set={"bin/a", "bin/rocm_ck_x"})
        self.assertEqual(r["verdict"], "fail")
        self.assertEqual(r["false_negatives"], ["bin/rocm_ck_x"])


class TestReachability(unittest.TestCase):
    def _depmap(self):
        return {
            "file_to_executables": {"h.hpp": ["bin/test_a", "bin/test_b"]},
            "executable_to_files": {"bin/test_a": ["h.hpp"], "bin/test_b": ["h.hpp"]},
        }

    def test_reachable_basenames(self):
        self.assertEqual(
            bfo.reachable_exe_basenames(self._depmap()), {"test_a", "test_b"}
        )


class TestClassifyUnreachable(unittest.TestCase):
    def _depmap(self):
        return {"executable_to_files": {"bin/test_a": ["h.hpp"]}}  # only test_a reachable

    def test_without_ninja_all_unreachable_are_fn(self):
        # test_b (compiled, unreachable) + test_py (non-compiled) both -> FN when
        # we have no build.ninja to classify.
        fn, nc = bfo.classify_unreachable(
            self._depmap(), {"test_a", "test_b", "test_py"}, compiled=None
        )
        self.assertEqual(fn, ["test_b", "test_py"])
        self.assertEqual(nc, [])

    def test_with_ninja_splits_fn_vs_noncompiled(self):
        # test_b has a bin/ target (compiled) -> real FN; test_py has none -> non-compiled.
        compiled = {"test_a", "test_b"}
        fn, nc = bfo.classify_unreachable(
            self._depmap(), {"test_a", "test_b", "test_py"}, compiled=compiled
        )
        self.assertEqual(fn, ["test_b"])       # compiled + unreachable
        self.assertEqual(nc, ["test_py"])      # no bin/ target -> always-run class

    def test_allowlist_applies_before_classification(self):
        compiled = {"test_a", "test_b"}
        fn, nc = bfo.classify_unreachable(
            self._depmap(), {"test_b"}, compiled=compiled, allow={"test_b"}
        )
        self.assertEqual(fn, [])
        self.assertEqual(nc, [])


class TestCodegenGlobs(unittest.TestCase):
    def _inventory(self, tmp):
        import json
        from pathlib import Path
        p = Path(tmp) / "codegen.json"
        p.write_text(json.dumps({"generators": [
            {"input": "a/generate.py", "test_globs": ["test_ck_tile_fmha_*"]},
            {"input": "b/generate.py", "test_globs": ["tile_example_sageattn_*", "test_x"]},
            {"input": "cmake/x.in", "test_globs": []},
        ]}))
        return str(p)

    def test_load_codegen_globs_flattens(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            globs = bfo.load_codegen_globs(self._inventory(tmp))
        self.assertEqual(
            globs, ["test_ck_tile_fmha_*", "tile_example_sageattn_*", "test_x"]
        )

    def test_expand_matches_globs_and_exact(self):
        ctest = {"test_ck_tile_fmha_fwd_fp16", "tile_example_sageattn_fwd",
                 "test_x", "test_gemm"}
        globs = ["test_ck_tile_fmha_*", "tile_example_sageattn_*", "test_x"]
        self.assertEqual(
            bfo.expand_test_globs(globs, ctest),
            ["test_ck_tile_fmha_fwd_fp16", "test_x", "tile_example_sageattn_fwd"],
        )

    def test_expand_no_match_is_empty(self):
        self.assertEqual(bfo.expand_test_globs(["nope_*"], {"test_gemm"}), [])

    def test_expand_empty_globs_is_empty(self):
        self.assertEqual(bfo.expand_test_globs([], {"test_gemm"}), [])


class TestCoverage(unittest.TestCase):
    def test_full_coverage_when_pre_superset(self):
        pre = {"a.hpp": ["bin/test_a", "bin/test_b"], "b.cpp": ["bin/test_b"]}
        post = {"a.hpp": ["bin/test_a"], "b.cpp": ["bin/test_b"]}
        r = bfo.compute_coverage(pre, post)
        self.assertEqual(r["coverage"], 1.0)
        self.assertEqual(r["n_false_negatives"], 0)
        self.assertEqual(r["verdict"], "pass")

    def test_missing_edge_is_false_negative(self):
        pre = {"gen.cpp": []}  # depmap saw the file but no exe (e.g. generated)
        post = {"gen.cpp": ["bin/test_fmha"]}  # real build proves the edge
        r = bfo.compute_coverage(pre, post)
        self.assertEqual(r["verdict"], "fail")
        self.assertEqual(r["false_negatives"], {"gen.cpp": ["bin/test_fmha"]})
        self.assertEqual(r["coverage"], 0.0)

    def test_ctest_intersection_excludes_nontests(self):
        pre = {"h.hpp": ["bin/test_x"]}
        post = {"h.hpp": ["bin/test_x", "bin/example_x"]}
        # example_x is not ctest-registered -> not counted as a missing edge
        r = bfo.compute_coverage(pre, post, ctest_tests={"test_x"})
        self.assertEqual(r["coverage"], 1.0)
        self.assertEqual(r["n_false_negatives"], 0)

    def test_partial_coverage_fraction(self):
        pre = {"a": ["bin/t1"], "b": []}
        post = {"a": ["bin/t1"], "b": ["bin/t2"]}  # 1 of 2 edges covered
        r = bfo.compute_coverage(pre, post)
        self.assertEqual(r["n_edges_post"], 2)
        self.assertEqual(r["n_edges_covered"], 1)
        self.assertEqual(r["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
