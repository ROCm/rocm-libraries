# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the harness's pure logic (no GPU; profiler monkeypatched)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rocke.benchmark.perf import harness


class TestPerfFromStdout(unittest.TestCase):
    def test_parses_perfjson(self):
        out = harness._perf_from_stdout(
            'noise\nPerfJSON: {"ms": 1.5, "tflops": 12.0, "gbps": 500.0}\n'
        )
        self.assertEqual(out["ms_median"], 1.5)
        self.assertEqual(out["tflops"], 12.0)
        self.assertEqual(out["gbs"], 500.0)  # gbps -> gbs

    def test_empty_when_no_perfjson(self):
        self.assertEqual(harness._perf_from_stdout("just logs\n"), {})

    def test_malformed_numeric_fields_are_ignored(self):
        out = harness._perf_from_stdout(
            'PerfJSON: {"ms": "bad", "tflops": 12.0, "gbps": null}\n'
        )
        self.assertEqual(out, {"tflops": 12.0})

    def test_parses_verification_fields(self):
        out = harness._verification_from_stdout(
            'PerfJSON: {"max_abs_diff": 0.25, "bad_count": 3, "total": 64}\n',
            verified=True,
        )
        self.assertEqual(
            out,
            {"max_abs_diff": 0.25, "bad_count": 3, "total": 64, "ok": False},
        )

    def test_unverified_payload_does_not_claim_correctness(self):
        out = harness._verification_from_stdout(
            'PerfJSON: {"max_abs_diff": 0.0, "bad_count": 0, "total": 64}\n'
        )
        self.assertEqual(out, {})


class TestPmcInput(unittest.TestCase):
    def test_one_line_per_group(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pmc.txt"
            harness._write_pmc_input([["A", "B", "C"], ["D"]], p)
            self.assertEqual(p.read_text(), "pmc: A B C\npmc: D\n")


class TestWall(unittest.TestCase):
    def test_nonzero_command_is_rejected(self):
        original = harness.subprocess.run
        harness.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout='PerfJSON: {"ms": 1.0, "bad_count": 1}\n',
            stderr="verification failed",
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "status 1"):
                harness._wall(["kernel"], {}, 1)
        finally:
            harness.subprocess.run = original

    def test_success_returns_timing_and_verification(self):
        original = harness.subprocess.run
        harness.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                'PerfJSON: {"ms": 1.0, "max_abs_diff": 0.0, '
                '"bad_count": 0, "total": 64}\n'
            ),
            stderr="",
        )
        try:
            wall, verify = harness._wall(["kernel", "--verify"], {}, 1)
        finally:
            harness.subprocess.run = original
        self.assertEqual(wall, {"ms_median": 1.0})
        self.assertEqual(
            verify,
            {"max_abs_diff": 0.0, "bad_count": 0, "total": 64, "ok": True},
        )

    def test_success_without_valid_timing_is_rejected(self):
        original = harness.subprocess.run
        outputs = (
            "completed without structured output\n",
            "PerfJSON: {not-json}\n",
            'PerfJSON: {"tflops": 12.0}\n',
            'PerfJSON: {"ms": "nan"}\n',
        )
        try:
            for stdout in outputs:
                with self.subTest(stdout=stdout):
                    harness.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
                        returncode=0, stdout=stdout, stderr=""
                    )
                    with self.assertRaisesRegex(RuntimeError, "valid PerfJSON timing"):
                        harness._wall(["kernel"], {}, 1)
        finally:
            harness.subprocess.run = original


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


class TestReadCounterCsvs(unittest.TestCase):
    def test_records_rocprof_counter_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = root / "pmc_1" / "first_counter_collection.csv"
            second = root / "pmc_2" / "second_counter_collection.csv"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("Dispatch_Id,Counter_Name,Counter_Value\n1,H,10\n")
            second.write_text("Dispatch_Id,Counter_Name,Counter_Value\n1,M,20\n")
            rows = harness._read_counter_csvs(root)
        self.assertCountEqual([row["counter_pass"] for row in rows], ["pmc_1", "pmc_2"])


class TestCounterMedians(unittest.TestCase):
    def _rows(self, did_val_pairs, counter="C1"):
        # one CSV row per (dispatch, counter)
        return [
            {"Dispatch_Id": str(d), "Counter_Name": counter, "Counter_Value": str(v)}
            for d, v in did_val_pairs
        ]

    def test_drops_leading_warmup_by_dispatch_order(self):
        # dispatches out of order in the list; warmup=2 drops the 2 lowest ids
        rows = self._rows([(3, 30), (1, 1000), (2, 900), (4, 40), (5, 50)])
        out = harness._counter_medians(rows, {"C1": "cyc"}, warmup=2)
        # kept dispatches 3,4,5 -> values 30,40,50 -> median 40 (warmup 1,2 dropped)
        self.assertEqual(out["cyc"], 40)

    def test_warmup_zero_keeps_all(self):
        rows = self._rows([(1, 10), (2, 20), (3, 30)])
        self.assertEqual(
            harness._counter_medians(rows, {"C1": "cyc"}, warmup=0)["cyc"], 20
        )

    def test_warmup_ge_dispatches_fails(self):
        rows = self._rows([(1, 10), (2, 20)])
        for warmup in (2, 5):
            with self.subTest(warmup=warmup), self.assertRaisesRegex(
                RuntimeError, "leaves no measured dispatches"
            ):
                harness._counter_medians(rows, {"C1": "cyc"}, warmup=warmup)

    def test_warmup_dropped_independently_per_counter_pass(self):
        rows = []
        for counter_pass, values in (
            ("pmc_1", (1000, 10, 20)),
            ("pmc_2", (900, 30, 40)),
        ):
            rows.extend(
                {
                    "Dispatch_Id": str(dispatch_id),
                    "Counter_Name": "C1",
                    "Counter_Value": str(value),
                    "counter_pass": counter_pass,
                }
                for dispatch_id, value in enumerate(values, start=1)
            )
        out = harness._counter_medians(rows, {"C1": "cyc"}, warmup=1)
        self.assertEqual(out["cyc"], 25)

    def test_ignores_unrequested_counters(self):
        rows = self._rows([(1, 10)], counter="OTHER")
        self.assertEqual(harness._counter_medians(rows, {"C1": "cyc"}, warmup=0), {})


class TestCounterSamples(unittest.TestCase):
    def test_duration_ns_parses_float_timestamps(self):
        # rocprofv3 CSVs can emit timestamps as floats/scientific notation, like the
        # counter/dispatch fields; duration_ns must survive that, not silently drop.
        rows = [
            {
                "Dispatch_Id": "1",
                "Counter_Name": "H",
                "Counter_Value": "10",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "100.0",
                "End_Timestamp": "1.1e2",
            }
        ]
        out = harness._counter_samples(rows, {"H": "l2_hit"})
        self.assertEqual(
            out,
            [
                {
                    "dispatch_id": 1,
                    "counter_pass": "pmc_1",
                    "l2_hit": 10,
                    "duration_ns": 10,
                }
            ],
        )

    def test_per_dispatch_sorted_all_dispatches(self):
        rows = [
            {
                "Dispatch_Id": "3",
                "Counter_Name": "H",
                "Counter_Value": "30",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "300",
                "End_Timestamp": "325",
            },
            {
                "Dispatch_Id": "3",
                "Counter_Name": "M",
                "Counter_Value": "3",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "300",
                "End_Timestamp": "325",
            },
            {
                "Dispatch_Id": "1",
                "Counter_Name": "H",
                "Counter_Value": "10",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "100",
                "End_Timestamp": "110",
            },
            {
                "Dispatch_Id": "1",
                "Counter_Name": "M",
                "Counter_Value": "1",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "100",
                "End_Timestamp": "110",
            },
        ]
        out = harness._counter_samples(rows, {"H": "l2_hit", "M": "l2_miss"})
        # sorted by dispatch_id, warmup NOT dropped, each dispatch has both counters
        self.assertEqual([s["dispatch_id"] for s in out], [1, 3])
        self.assertEqual(
            out[0],
            {
                "dispatch_id": 1,
                "counter_pass": "pmc_1",
                "l2_hit": 10,
                "l2_miss": 1,
                "duration_ns": 10,
            },
        )
        self.assertEqual(
            out[1],
            {
                "dispatch_id": 3,
                "counter_pass": "pmc_1",
                "l2_hit": 30,
                "l2_miss": 3,
                "duration_ns": 25,
            },
        )

    def test_per_dispatch_keeps_counter_passes_separate(self):
        rows = [
            {
                "Dispatch_Id": "1",
                "Counter_Name": "H",
                "Counter_Value": "10",
                "counter_pass": "pmc_1",
                "Start_Timestamp": "100",
                "End_Timestamp": "110",
            },
            {
                "Dispatch_Id": "1",
                "Counter_Name": "M",
                "Counter_Value": "3",
                "counter_pass": "pmc_2",
                "Start_Timestamp": "200",
                "End_Timestamp": "225",
            },
        ]
        out = harness._counter_samples(rows, {"H": "l2_hit", "M": "l2_miss"})
        self.assertEqual(
            out,
            [
                {
                    "dispatch_id": 1,
                    "counter_pass": "pmc_1",
                    "l2_hit": 10,
                    "duration_ns": 10,
                },
                {
                    "dispatch_id": 1,
                    "counter_pass": "pmc_2",
                    "l2_miss": 3,
                    "duration_ns": 25,
                },
            ],
        )

    def test_per_dispatch_omits_invalid_timestamps(self):
        rows = [
            {
                "Dispatch_Id": "1",
                "Counter_Name": "H",
                "Counter_Value": "10",
                "Start_Timestamp": "20",
                "End_Timestamp": "10",
            }
        ]
        out = harness._counter_samples(rows, {"H": "l2_hit"})
        self.assertEqual(
            out,
            [{"dispatch_id": 1, "counter_pass": "pmc_0", "l2_hit": 10}],
        )

    def test_ignores_unrequested_counters(self):
        rows = [{"Dispatch_Id": "1", "Counter_Name": "X", "Counter_Value": "9"}]
        self.assertEqual(harness._counter_samples(rows, {"H": "l2_hit"}), [])


class TestPickTarget(unittest.TestCase):
    def _rows(self, *names):
        return [{"Kernel_Name": n} for n in names]

    def test_busiest_non_helper(self):
        rows = self._rows("gemm", "gemm", "saxpy", "__amd_memset")
        self.assertEqual(harness._pick_target_kernel(rows, None), "gemm")

    def test_substring_match(self):
        rows = self._rows("mygemm_tile64_pad8", "mygemm_tile64_pad8", "other_k")
        self.assertEqual(
            harness._pick_target_kernel(rows, "mygemm"), "mygemm_tile64_pad8"
        )

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
        self._orig_read = harness._read_counter_csvs
        self._orig_passes = harness._count_passes
        harness._wall = lambda cmd, env, timeout: ({}, {})  # no subprocess

    def tearDown(self):
        harness._counters.discover = self._orig_disc
        harness._run_rocprofv3 = self._orig_run
        harness._wall = self._orig_wall
        harness._read_counter_csvs = self._orig_read
        harness._count_passes = self._orig_passes

    def test_no_counters_warns_and_wall_only(self):
        harness._counters.discover = lambda arch: {}
        warns = []
        rec = harness.profile(
            ["x"],
            "gfx1201",
            label="mylabel",
            op="op",
            shape={"M": 1},
            warn=warns.append,
        )
        self.assertEqual(rec["kernel"]["kernel_name"], "mylabel")  # label = identity
        self.assertEqual(rec["counters"], {})
        self.assertTrue(any("no PMU counters" in w for w in warns))

    def test_rocprofv3_failure_warns(self):
        harness._counters.discover = lambda arch: {"busy_cycles": "GRBM_GUI_ACTIVE"}
        harness._run_rocprofv3 = lambda *a, **k: (False, "")  # (ok, stdout)
        warns = []
        rec = harness.profile(
            ["x"], "gfx950", op="op", shape={"M": 1}, warn=warns.append
        )
        self.assertEqual(rec["counters"], {})
        self.assertTrue(any("rocprofv3 failed" in w for w in warns))

    def test_label_overrides_identity_but_keeps_dispatch_symbol_absent(self):
        # With no profiler, there is no dispatched symbol, so no dispatch_symbol key.
        harness._counters.discover = lambda arch: {}
        rec = harness.profile(["x"], "gfx1201", label="lbl", op="o", shape={})
        self.assertEqual(rec["kernel"]["kernel_name"], "lbl")
        self.assertNotIn("dispatch_symbol", rec["kernel"])

    def test_profiler_fallback_uses_a_stable_command_identity(self):
        harness._counters.discover = lambda arch: {}
        first = harness.profile(["first"], "gfx1201", op="o", shape={})
        second = harness.profile(["second"], "gfx1201", op="o", shape={})
        self.assertTrue(first["kernel"]["kernel_name"].startswith("command_"))
        self.assertNotEqual(
            first["kernel"]["kernel_name"], second["kernel"]["kernel_name"]
        )
        self.assertNotEqual(first["run"]["run_id"], second["run"]["run_id"])

    def test_negative_warmup_rejected(self):
        with self.assertRaises(ValueError):
            harness.profile(["x"], "gfx1201", warmup=-1)

    def test_l2_hit_without_miss_skips_ratio_no_keyerror(self):
        # Only the L2 hit counter populated (miss absent, e.g. a partial arch
        # capture): l2_hit_rate must be omitted, not raise KeyError.
        harness._counters.discover = lambda arch: {"l2_hit": "TCC_HIT"}
        harness._run_rocprofv3 = lambda *a, **k: (True, "")
        harness._count_passes = lambda outdir: 1
        harness._read_counter_csvs = lambda outdir: [
            {
                "Kernel_Name": "gemm",
                "Dispatch_Id": "1",
                "Counter_Name": "TCC_HIT",
                "Counter_Value": "500",
            }
        ]
        rec = harness.profile(["x"], "gfx950", op="op", shape={"M": 1})
        self.assertEqual(rec["counters"], {"l2_hit": 500})
        self.assertNotIn("l2_hit_rate", rec["derived"])


if __name__ == "__main__":
    unittest.main()
