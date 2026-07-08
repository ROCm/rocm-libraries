# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the harness's pure logic (no GPU; profiler monkeypatched)."""
import tempfile
import unittest
from pathlib import Path

from rocke.benchmark.perf import harness


class TestPerfFromStdout(unittest.TestCase):
    def test_parses_perfjson(self):
        out = harness._perf_from_stdout('noise\nPerfJSON: {"ms": 1.5, "tflops": 12.0, "gbps": 500.0}\n')
        self.assertEqual(out["ms_median"], 1.5)
        self.assertEqual(out["tflops"], 12.0)
        self.assertEqual(out["gbs"], 500.0)     # gbps -> gbs

    def test_empty_when_no_perfjson(self):
        self.assertEqual(harness._perf_from_stdout("just logs\n"), {})


class TestPmcInput(unittest.TestCase):
    def test_one_line_per_group(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pmc.txt"
            harness._write_pmc_input([["A", "B", "C"], ["D"]], p)
            self.assertEqual(p.read_text(), "pmc: A B C\npmc: D\n")


class TestCountPasses(unittest.TestCase):
    def test_counts_pmc_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "pmc_1" / "host").mkdir(parents=True)
            (out / "pmc_2" / "host").mkdir(parents=True)
            (out / "notapass").mkdir()
            self.assertEqual(harness._count_passes(out), 2)

    def test_zero_when_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(harness._count_passes(Path(d)), 0)


class TestCounterMedians(unittest.TestCase):
    def _rows(self, did_val_pairs, counter="C1"):
        # one CSV row per (dispatch, counter)
        return [{"Dispatch_Id": str(d), "Counter_Name": counter,
                 "Counter_Value": str(v)} for d, v in did_val_pairs]

    def test_drops_leading_warmup_by_dispatch_order(self):
        # dispatches out of order in the list; warmup=2 drops the 2 lowest ids
        rows = self._rows([(3, 30), (1, 1000), (2, 900), (4, 40), (5, 50)])
        out = harness._counter_medians(rows, {"C1": "cyc"}, warmup=2)
        # kept dispatches 3,4,5 -> values 30,40,50 -> median 40 (warmup 1,2 dropped)
        self.assertEqual(out["cyc"], 40)

    def test_warmup_zero_keeps_all(self):
        rows = self._rows([(1, 10), (2, 20), (3, 30)])
        self.assertEqual(harness._counter_medians(rows, {"C1": "cyc"}, warmup=0)["cyc"], 20)

    def test_warmup_ge_dispatches_keeps_all(self):
        rows = self._rows([(1, 10), (2, 20)])
        # dropping 5 would leave nothing -> fall back to all
        self.assertEqual(harness._counter_medians(rows, {"C1": "cyc"}, warmup=5)["cyc"], 15)

    def test_ignores_unrequested_counters(self):
        rows = self._rows([(1, 10)], counter="OTHER")
        self.assertEqual(harness._counter_medians(rows, {"C1": "cyc"}, warmup=0), {})


class TestPickTarget(unittest.TestCase):
    def _rows(self, *names):
        return [{"Kernel_Name": n} for n in names]

    def test_busiest_non_helper(self):
        rows = self._rows("gemm", "gemm", "saxpy", "__amd_memset")
        self.assertEqual(harness._pick_target_kernel(rows, None), "gemm")

    def test_substring_match(self):
        rows = self._rows("mygemm_tile64_pad8", "mygemm_tile64_pad8", "other_k")
        self.assertEqual(
            harness._pick_target_kernel(rows, "mygemm"), "mygemm_tile64_pad8")

    def test_helpers_skipped(self):
        rows = self._rows("__amd_rocclr_fillBuffer", "__hip_x", "realk")
        self.assertEqual(harness._pick_target_kernel(rows, None), "realk")

    def test_no_match_returns_none(self):
        self.assertIsNone(harness._pick_target_kernel(self._rows("a", "b"), "zzz"))


class TestProfileDegradation(unittest.TestCase):
    """Wall-only paths, exercised by forcing the profiler branches off."""

    def setUp(self):
        self._orig_disc = harness._counters.discover
        self._orig_run = harness._run_rocprofv3
        self._orig_wall = harness._wall
        harness._wall = lambda cmd, env, timeout: {}   # no subprocess

    def tearDown(self):
        harness._counters.discover = self._orig_disc
        harness._run_rocprofv3 = self._orig_run
        harness._wall = self._orig_wall

    def test_no_counters_warns_and_wall_only(self):
        harness._counters.discover = lambda arch: {}
        warns = []
        rec = harness.profile(["x"], "gfx1201", label="mylabel", op="op",
                              shape={"M": 1}, warn=warns.append)
        self.assertEqual(rec["kernel"]["kernel_name"], "mylabel")  # label = identity
        self.assertEqual(rec["counters"], {})
        self.assertTrue(any("no PMU counters" in w for w in warns))

    def test_rocprofv3_failure_warns(self):
        harness._counters.discover = lambda arch: {"busy_cycles": "GRBM_GUI_ACTIVE"}
        harness._run_rocprofv3 = lambda *a, **k: (False, "")   # (ok, stdout)
        warns = []
        rec = harness.profile(["x"], "gfx950", op="op", shape={"M": 1},
                              warn=warns.append)
        self.assertEqual(rec["counters"], {})
        self.assertTrue(any("rocprofv3 failed" in w for w in warns))

    def test_label_overrides_identity_but_keeps_dispatch_symbol_absent(self):
        # With no profiler, there is no dispatched symbol, so no dispatch_symbol key.
        harness._counters.discover = lambda arch: {}
        rec = harness.profile(["x"], "gfx1201", label="lbl", op="o", shape={})
        self.assertEqual(rec["kernel"]["kernel_name"], "lbl")
        self.assertNotIn("dispatch_symbol", rec["kernel"])


if __name__ == "__main__":
    unittest.main()
