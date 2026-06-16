#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the build-filter completeness oracle (pure logic)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# filter_oracle.py lives in the dependency-parser dir (parent of tests/).
ORACLE_PY = Path(__file__).parent.parent / "filter_oracle.py"
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

    def test_edge_file_test_level_metrics_differ(self):
        # a.hpp resolves to both its tests; b.hpp misses one of two.
        pre = {"a.hpp": ["bin/t1", "bin/t2"], "b.hpp": ["bin/t1"]}
        post = {"a.hpp": ["bin/t1", "bin/t2"], "b.hpp": ["bin/t1", "bin/t2"]}
        r = bfo.compute_coverage(pre, post)
        # edge: 3/4 covered
        self.assertEqual((r["n_edges_covered"], r["n_edges_post"]), (3, 4))
        self.assertEqual(r["coverage"], 0.75)
        # file: a.hpp full, b.hpp has a miss -> 1/2
        self.assertEqual((r["n_files_covered"], r["n_files_with_edges"]), (1, 2))
        self.assertEqual(r["file_coverage"], 0.5)
        # test: t1 fully captured, t2 missing on b.hpp -> 1/2
        self.assertEqual((r["n_tests_covered"], r["n_tests"]), (1, 2))
        self.assertEqual(r["test_coverage"], 0.5)
        self.assertEqual(r["tests_with_fn"], ["t2"])  # basenames (ctest test names)


class TestCoverageCanon(unittest.TestCase):
    def test_canonical_key_strips_monorepo_prefix(self):
        self.assertEqual(
            bfo._canonical_key("projects/composablekernel/include/ck/x.hpp"),
            "include/ck/x.hpp",
        )
        # already project-root or build/ keys are untouched
        self.assertEqual(bfo._canonical_key("include/ck/x.hpp"), "include/ck/x.hpp")
        self.assertEqual(bfo._canonical_key("build/_deps/gtest/x.h"),
                         "build/_deps/gtest/x.h")

    def test_is_source_key(self):
        self.assertTrue(bfo._is_source_key("include/ck/x.hpp"))
        self.assertTrue(bfo._is_source_key("test/foo/bar.cpp"))
        self.assertFalse(bfo._is_source_key("build/_deps/gtest/x.h"))
        self.assertFalse(bfo._is_source_key("build/library/gen/inst.cpp"))
        self.assertFalse(bfo._is_source_key("/usr/include/c++/vector"))

    def test_canon_f2e_source_only_drops_build_and_system(self):
        f2e = {
            "projects/composablekernel/include/ck/x.hpp": ["bin/test_a"],
            "build/_deps/gtest/g.h": ["bin/test_a"],
            "/usr/include/vector": ["bin/test_a"],
        }
        out = bfo._canon_f2e(f2e, source_only=True)
        self.assertEqual(out, {"include/ck/x.hpp": ["bin/test_a"]})

    def test_compute_coverage_after_canon_aligns_mismatched_roots(self):
        # pre keyed at repo root, post at project root (the real-world mismatch).
        pre = {"projects/composablekernel/include/ck/x.hpp": ["bin/test_a"]}
        post = {"include/ck/x.hpp": ["bin/test_a"]}
        # Raw diff misses (different keys) ...
        self.assertEqual(bfo.compute_coverage(pre, post)["coverage"], 0.0)
        # ... canonicalized, they align.
        r = bfo.compute_coverage(bfo._canon_f2e(pre), bfo._canon_f2e(post))
        self.assertEqual(r["coverage"], 1.0)
        self.assertEqual(r["n_false_negatives"], 0)


class TestCoverageAggregate(unittest.TestCase):
    def _r(self, label, fn=None, tests_fn=None, edge=1.0, file=1.0, test=1.0, verdict="pass"):
        return {"label": label, "coverage": edge, "file_coverage": file,
                "test_coverage": test, "n_false_negatives": sum(len(v) for v in (fn or {}).values()),
                "false_negatives": fn or {}, "tests_with_fn": tests_fn or [], "verdict": verdict}

    def test_worst_case_and_union(self):
        r1 = self._r("gfx942", {"a.hpp": ["t1", "t2"]}, ["t1", "t2"],
                     edge=0.99, file=0.98, test=0.95, verdict="fail")
        r2 = self._r("gfx950")
        a = bfo.aggregate_coverage([r1, r2])
        self.assertEqual(a["n_arches"], 2)
        self.assertEqual(a["worst_test_coverage"], 0.95)
        self.assertEqual(a["worst_file_coverage"], 0.98)
        self.assertEqual(a["tests_with_fn"], ["t1", "t2"])
        self.assertEqual(a["false_negatives"], {"a.hpp": ["t1", "t2"]})
        self.assertEqual(a["verdict"], "fail")

    def test_union_merges_same_file_across_arches(self):
        r1 = self._r("a", {"x.hpp": ["t1"]}, ["t1"], verdict="fail")
        r2 = self._r("b", {"x.hpp": ["t2"]}, ["t2"], verdict="fail")
        a = bfo.aggregate_coverage([r1, r2])
        self.assertEqual(a["false_negatives"], {"x.hpp": ["t1", "t2"]})
        self.assertEqual(a["n_false_negatives"], 2)
        self.assertEqual(a["n_files_with_fn"], 1)

    def test_all_pass(self):
        a = bfo.aggregate_coverage([self._r("a"), self._r("b")])
        self.assertEqual(a["verdict"], "pass")
        self.assertEqual(a["n_false_negatives"], 0)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            bfo.aggregate_coverage([])


class TestOracleCli(unittest.TestCase):
    """End-to-end subprocess tests for filter_oracle.py CLI subcommands."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_json(self, name, obj):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    def _write_text(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def _run(self, *args):
        proc = subprocess.run(
            [sys.executable, str(ORACLE_PY)] + list(args),
            capture_output=True, text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_probe_missing_required_arg_exits_two(self):
        # probe needs --depmap/--ninja/--file/--failed-objects; argparse must
        # reject an incomplete invocation (exit 2) rather than the subcommand
        # being silently unregistered.
        rc, _, stderr = self._run("probe", "--depmap", "x.json")
        self.assertEqual(rc, 2)
        self.assertIn("--failed-objects", stderr)

    def test_reachability_pass_exit_zero(self):
        depmap = {"file_to_executables": {"a.hpp": ["bin/test_a"]}}
        depmap_path = self._write_json("dep.json", depmap)
        # ctest list: one compiled test that IS reachable
        ctest_path = self._write_text("ctest.txt", "Test #1: test_a\n")
        # ninja targets: bin/test_a exists
        ninja_path = self._write_text("ninja.txt", "bin/test_a: phony\n")
        out_path = os.path.join(self.tmp, "reach.json")
        rc, stdout, _ = self._run(
            "reachability",
            "--depmap", depmap_path,
            "--ctest", ctest_path,
            "--ninja", ninja_path,
            "--output", out_path,
        )
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            result = json.load(f)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["n_false_negatives"], 0)

    def test_reachability_fail_exit_one_on_fn(self):
        # test_b is not in the depmap; without --ninja, all unreachable tests are
        # classified as false negatives (unclassified mode).
        depmap = {"file_to_executables": {}}
        depmap_path = self._write_json("dep.json", depmap)
        ctest_path = self._write_text("ctest.txt", "Test #1: test_b\n")
        out_path = os.path.join(self.tmp, "reach.json")
        rc, _, _ = self._run(
            "reachability",
            "--depmap", depmap_path,
            "--ctest", ctest_path,
            "--output", out_path,
        )
        self.assertEqual(rc, 1)
        with open(out_path) as f:
            result = json.load(f)
        self.assertEqual(result["verdict"], "fail")
        self.assertIn("test_b", result["false_negatives"])

    def test_coverage_pass_when_pre_superset(self):
        pre = {"file_to_executables": {"a.hpp": ["bin/test_a"], "b.hpp": ["bin/test_a"]}}
        post = {"file_to_executables": {"a.hpp": ["bin/test_a"]}}
        pre_path = self._write_json("pre.json", pre)
        post_path = self._write_json("post.json", post)
        out_path = os.path.join(self.tmp, "cov.json")
        rc, _, _ = self._run(
            "coverage", "--pre", pre_path, "--post", post_path, "--output", out_path,
        )
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            result = json.load(f)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["coverage"], 1.0)

    def test_coverage_fail_when_post_has_uncovered_edge(self):
        pre = {"file_to_executables": {"a.hpp": ["bin/test_a"]}}
        post = {"file_to_executables": {"a.hpp": ["bin/test_a"], "b.hpp": ["bin/test_b"]}}
        pre_path = self._write_json("pre.json", pre)
        post_path = self._write_json("post.json", post)
        out_path = os.path.join(self.tmp, "cov.json")
        rc, _, _ = self._run(
            "coverage", "--pre", pre_path, "--post", post_path, "--output", out_path,
        )
        self.assertEqual(rc, 1)
        with open(out_path) as f:
            result = json.load(f)
        self.assertEqual(result["verdict"], "fail")

    def test_reachability_junit_written_when_flag_given(self):
        import xml.etree.ElementTree as ET

        depmap = {"file_to_executables": {"a.hpp": ["bin/test_a"]}}
        depmap_path = self._write_json("dep.json", depmap)
        ctest_path = self._write_text("ctest.txt", "Test #1: test_a\n")
        ninja_path = self._write_text("build.ninja", "bin/test_a: phony\n")
        junit_path = os.path.join(self.tmp, "reach.xml")
        rc, _, _ = self._run(
            "reachability",
            "--depmap", depmap_path,
            "--ctest", ctest_path,
            "--ninja", ninja_path,
            "--junit", junit_path,
        )
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(junit_path))
        root = ET.parse(junit_path).getroot()
        self.assertEqual(root.get("failures"), "0")

    def test_reachability_junit_fail_has_failure_per_fn(self):
        import xml.etree.ElementTree as ET

        depmap = {"file_to_executables": {}}
        depmap_path = self._write_json("dep.json", depmap)
        ctest_path = self._write_text("ctest.txt", "Test #1: test_b\nTest #2: test_c\n")
        junit_path = os.path.join(self.tmp, "reach.xml")
        rc, _, _ = self._run(
            "reachability",
            "--depmap", depmap_path,
            "--ctest", ctest_path,
            "--junit", junit_path,
        )
        self.assertEqual(rc, 1)
        root = ET.parse(junit_path).getroot()
        self.assertEqual(root.get("failures"), "2")
        failures = root.findall(".//failure")
        self.assertEqual(len(failures), 2)


class TestReachabilityJunit(unittest.TestCase):
    """Unit tests for render_junit_reachability."""

    def _pass_result(self, **kwargs):
        base = {
            "n_ctest": 3, "n_reachable": 3, "n_false_negatives": 0,
            "false_negatives": [], "n_non_compiled": 0, "non_compiled": [],
            "allowlisted": [], "n_codegen_allowlisted": 0, "codegen_allowlisted": [],
            "classified": True, "verdict": "pass",
        }
        base.update(kwargs)
        return base

    def _fail_result(self, fn_tests, **kwargs):
        base = self._pass_result(**kwargs)
        base.update({
            "n_false_negatives": len(fn_tests),
            "false_negatives": fn_tests,
            "verdict": "fail",
        })
        return base

    def test_pass_no_failures(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(bfo.render_junit_reachability(self._pass_result()))
        self.assertEqual(root.get("failures"), "0")
        self.assertEqual(len(root.findall(".//failure")), 0)

    def test_fail_has_failure_per_fn(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(bfo.render_junit_reachability(self._fail_result(["t1", "t2"])))
        self.assertEqual(root.get("failures"), "2")
        self.assertEqual(len(root.findall(".//failure")), 2)

    def test_xml_is_well_formed(self):
        import xml.etree.ElementTree as ET

        ET.fromstring(bfo.render_junit_reachability(self._pass_result()))
        ET.fromstring(bfo.render_junit_reachability(self._fail_result(["t1"])))

    def test_properties_present_with_counts(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(bfo.render_junit_reachability(self._pass_result(n_ctest=5, n_reachable=4)))
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["n_ctest"], "5")
        self.assertEqual(props["n_reachable"], "4")
        self.assertEqual(props["n_false_negatives"], "0")
        self.assertIn("classified", props)

    def test_mode_label_tagging(self):
        import xml.etree.ElementTree as ET

        result = self._pass_result(mode="full", label="gfx942")
        xml = bfo.render_junit_reachability(result)
        self.assertIn('classname="smart-build.reachability.full.gfx942"', xml)
        self.assertIn('name="smart-build-reachability-full-gfx942"', xml)
        root = ET.fromstring(xml)
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["advisory"], "true")

    def test_selective_mode_not_advisory(self):
        import xml.etree.ElementTree as ET

        result = self._pass_result(mode="selective")
        root = ET.fromstring(bfo.render_junit_reachability(result))
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["advisory"], "false")

    def test_no_mode_no_advisory_property(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(bfo.render_junit_reachability(self._pass_result()))
        props = {p.get("name") for p in root.findall("./properties/property")}
        self.assertNotIn("advisory", props)

    def test_pass_case_name_includes_label(self):
        result = self._pass_result(label="gfx950")
        xml = bfo.render_junit_reachability(result)
        self.assertIn("all-compiled-tests-reachable (gfx950)", xml)


if __name__ == "__main__":
    unittest.main()
