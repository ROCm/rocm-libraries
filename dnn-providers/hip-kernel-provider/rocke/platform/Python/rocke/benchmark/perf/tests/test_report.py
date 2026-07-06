# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the reporting primitive (pure, no GPU)."""
import unittest

from rocke.benchmark.perf import report, schema


def _rec(arch="gfx950", kernel="k", shape=None, *, busy=None, total=None,
         ms=None, waves=None, wait=None, l2_hit=None, l2_miss=None,
         occupancy=None, lds_bytes=None, spread=None, n=None):
    counters = {}
    if busy is not None:
        counters["busy_cycles"] = busy
    if total is not None:
        counters["total_clocks"] = total
    if waves is not None:
        counters["waves"] = waves
    if wait is not None:
        counters["wait_cycles"] = wait
    if l2_hit is not None:
        counters["l2_hit"] = l2_hit
    if l2_miss is not None:
        counters["l2_miss"] = l2_miss
    derived = {}
    if total and busy is not None:
        derived["busy_fraction"] = busy / total
    if l2_hit is not None and l2_miss is not None and (l2_hit + l2_miss):
        derived["l2_hit_rate"] = l2_hit / (l2_hit + l2_miss)
    resources = {}
    if occupancy is not None:
        resources["occupancy"] = occupancy
    if lds_bytes is not None:
        resources["lds_bytes"] = lds_bytes
    wall = {}
    if ms is not None:
        wall["ms_median"] = ms
    rec = {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": kernel, "op": "gemm", "shape": shape or {"M": 8}},
        "wall": wall,
        "counters": counters,
        "resources": resources,
        "derived": derived,
        "captured_counters": sorted(counters),
        "verify": {},
    }
    if spread is not None:
        rec["spread"] = spread
    if n is not None:
        rec["n_samples"] = n
    return rec


class TestSerialize(unittest.TestCase):
    def test_round_trip_preserves_values(self):
        rec = _rec(busy=1000, total=1050, ms=1.23, waves=16384, l2_hit=9, l2_miss=1)
        back = report.from_json(report.to_json(rec))
        self.assertEqual(back, rec)


class TestPanel(unittest.TestCase):
    def test_panel_pulls_from_all_sections(self):
        rec = _rec(busy=980, total=1000, waves=16384, wait=50, l2_hit=9, l2_miss=1,
                   occupancy=12, lds_bytes=2048)
        p = report.panel(rec)
        # derived
        self.assertAlmostEqual(p["busy_fraction"], 0.98)
        self.assertAlmostEqual(p["l2_hit_rate"], 0.9)
        # counters
        self.assertEqual(p["waves"], 16384)
        self.assertEqual(p["wait_cycles"], 50)
        # resources
        self.assertEqual(p["occupancy"], 12)
        self.assertEqual(p["lds_bytes"], 2048)

    def test_partial_coverage_yields_smaller_panel(self):
        # Tier-W only: no counters/derived/resources -> empty panel, not zeros.
        self.assertEqual(report.panel(_rec(ms=1.0)), {})


class TestDiff(unittest.TestCase):
    def test_slower_regression(self):
        base = _rec(busy=1000, total=1000)
        cur = _rec(busy=1100, total=1000)
        d = report.diff(base, cur)
        self.assertEqual(d["metric"], "busy_cycles")
        self.assertTrue(d["slower"])
        self.assertAlmostEqual(d["pct_change"], 10.0)
        self.assertEqual(d["abs_delta"], 100)

    def test_faster_improvement(self):
        d = report.diff(_rec(busy=1000, total=1000), _rec(busy=900, total=1000))
        self.assertFalse(d["slower"])
        self.assertAlmostEqual(d["pct_change"], -10.0)

    def test_metric_mismatch_omits_pct(self):
        base = _rec(busy=1000, total=1000)     # primary = busy_cycles
        cur = _rec(ms=2.0)                       # only wall -> ms_median
        d = report.diff(base, cur)
        self.assertTrue(d["metric_mismatch"])
        self.assertNotIn("pct_change", d)

    def test_spread_surfaced_from_aggregate(self):
        cur = _rec(busy=1100, total=1000, spread={"busy_cycles_pct": 3.5})
        d = report.diff(_rec(busy=1000, total=1000), cur)
        self.assertAlmostEqual(d["spread_pct"], 3.5)

    def test_panel_deltas(self):
        base = _rec(busy=980, total=1000, waves=100)
        cur = _rec(busy=990, total=1000, waves=120)
        d = report.diff(base, cur)
        self.assertEqual(d["panel"]["waves"]["delta"], 20)
        self.assertAlmostEqual(d["panel"]["busy_fraction"]["delta"], 0.01)


class TestFormatting(unittest.TestCase):
    def test_format_record_smoke(self):
        rec = _rec(busy=1000, total=1050, waves=16384, spread={"busy_cycles_pct": 2.1}, n=5)
        txt = report.format_record(rec)
        self.assertIn("gfx950", txt)
        self.assertIn("busy_cycles", txt)
        self.assertIn("n=5", txt)
        self.assertIn("panel", txt)

    def test_format_diff_smoke(self):
        txt = report.format_diff(_rec(busy=1000, total=1000),
                                 _rec(busy=1100, total=1000, spread={"busy_cycles_pct": 2.0}))
        self.assertIn("SLOWER", txt)
        self.assertIn("+10.0%", txt)


if __name__ == "__main__":
    unittest.main()
