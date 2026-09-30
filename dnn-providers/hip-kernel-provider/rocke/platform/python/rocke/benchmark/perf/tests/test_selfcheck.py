# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the advisory self-check (pure, no GPU)."""
import unittest

from rocke.benchmark.perf import schema
from rocke.benchmark.perf.tool import selfcheck


def _rec(
    arch="gfx950",
    kernel="k",
    shape=None,
    *,
    busy=None,
    total=None,
    ms=None,
    l2_hit=None,
    l2_miss=None,
    spread=None
):
    counters = {}
    if busy is not None:
        counters["busy_cycles"] = busy
    if total is not None:
        counters["total_clocks"] = total
    if l2_hit is not None:
        counters["l2_hit"] = l2_hit
    if l2_miss is not None:
        counters["l2_miss"] = l2_miss
    derived = {}
    if total and busy is not None:
        derived["busy_fraction"] = busy / total
    if l2_hit is not None and l2_miss is not None and (l2_hit + l2_miss):
        derived["l2_hit_rate"] = l2_hit / (l2_hit + l2_miss)
    wall = {"ms_median": ms} if ms is not None else {}
    rec = {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": kernel, "op": "gemm", "shape": shape or {"M": 8}},
        "wall": wall,
        "counters": counters,
        "resources": {},
        "derived": derived,
        "captured_counters": sorted(counters),
        "verify": {},
    }
    if spread is not None:
        rec["spread"] = spread
    return rec


class TestCompare(unittest.TestCase):
    def test_real_regression_flagged(self):
        prev = _rec(busy=1000, total=1000)
        cur = _rec(busy=1100, total=1000)  # +10%, > 5% floor
        r = selfcheck.compare(prev, cur)
        self.assertEqual(r["verdict"], "regressed")
        self.assertAlmostEqual(r["pct_change"], 10.0)

    def test_small_change_within_noise(self):
        prev = _rec(busy=1000, total=1000)
        cur = _rec(busy=1020, total=1000)  # +2%, < 5% floor
        self.assertEqual(selfcheck.compare(prev, cur)["verdict"], "within_noise")

    def test_improvement_flagged(self):
        prev = _rec(busy=1000, total=1000)
        cur = _rec(busy=900, total=1000)  # -10%
        self.assertEqual(selfcheck.compare(prev, cur)["verdict"], "improved")

    def test_noise_floor_uses_spread(self):
        # +8% change but spread 4% -> floor = max(5, 3*4=12) = 12% -> within noise
        prev = _rec(busy=1000, total=1000)
        cur = _rec(busy=1080, total=1000, spread={"busy_cycles_pct": 4.0})
        r = selfcheck.compare(prev, cur)
        self.assertEqual(r["verdict"], "within_noise")
        self.assertAlmostEqual(r["floor_pct"], 12.0)

    def test_noise_floor_uses_baseline_spread(self):
        # The baseline has 4% observed variation, so a later +8% change remains
        # within the 12% noise floor even when the current run is perfectly stable.
        prev = _rec(busy=1000, total=1000, spread={"busy_cycles_pct": 4.0})
        cur = _rec(busy=1080, total=1000, spread={"busy_cycles_pct": 0.0})
        r = selfcheck.compare(prev, cur)
        self.assertEqual(r["verdict"], "within_noise")
        self.assertAlmostEqual(r["floor_pct"], 12.0)

    def test_reports_which_counter_moved(self):
        # slowdown accompanied by an L2 hit-rate collapse -> panel shows it
        prev = _rec(busy=1000, total=1000, l2_hit=90, l2_miss=10)
        cur = _rec(busy=1200, total=1000, l2_hit=50, l2_miss=50)
        r = selfcheck.compare(prev, cur)
        self.assertEqual(r["verdict"], "regressed")
        panel = r["diff"]["panel"]
        self.assertAlmostEqual(panel["l2_hit_rate"]["baseline"], 0.9)
        self.assertAlmostEqual(panel["l2_hit_rate"]["current"], 0.5)
        self.assertAlmostEqual(panel["l2_hit_rate"]["delta"], -0.4)

    def test_metric_mismatch_unknown(self):
        prev = _rec(busy=1000, total=1000)  # busy_cycles
        cur = _rec(ms=2.0)  # ms only
        self.assertEqual(selfcheck.compare(prev, cur)["verdict"], "unknown")

    def test_uses_shared_wall_when_only_one_record_has_counters(self):
        prev = _rec(busy=1000, total=1000, ms=1.0)
        cur = _rec(ms=1.2)
        result = selfcheck.compare(prev, cur)
        self.assertEqual(result["metric"], "ms_median")
        self.assertEqual(result["verdict"], "regressed")

    def test_fallback_to_wall_when_no_counters(self):
        prev = _rec(ms=1.0)
        cur = _rec(ms=1.2)  # +20% on ms
        r = selfcheck.compare(prev, cur)
        self.assertEqual(r["metric"], "ms_median")
        self.assertEqual(r["verdict"], "regressed")


class TestCheckHistory(unittest.TestCase):
    def test_no_baseline_with_single_run(self):
        recs = [_rec(busy=1000, total=1000)]
        ident = schema.identity(recs[0])
        self.assertEqual(selfcheck.check_history(recs, ident)["verdict"], "no_baseline")

    def test_compares_two_most_recent(self):
        recs = [
            _rec(busy=1000, total=1000),
            _rec(busy=1005, total=1000),  # prev (2nd most recent)
            _rec(busy=1200, total=1000),
        ]  # current -> regressed vs 1005
        ident = schema.identity(recs[0])
        r = selfcheck.check_history(recs, ident)
        self.assertEqual(r["verdict"], "regressed")
        self.assertAlmostEqual(r["diff"]["baseline"], 1005)
        self.assertAlmostEqual(r["diff"]["current"], 1200)

    def test_ignores_other_identities(self):
        recs = [
            _rec(kernel="a", busy=1000, total=1000),
            _rec(kernel="b", busy=9999, total=1000),
            _rec(kernel="a", busy=1010, total=1000),
        ]
        ident = schema.identity(_rec(kernel="a"))
        r = selfcheck.check_history(recs, ident)  # only the two "a" runs compared
        self.assertEqual(r["verdict"], "within_noise")


class TestFormat(unittest.TestCase):
    def test_format_regression_shows_tag_and_panel(self):
        prev = _rec(busy=1000, total=1000, l2_hit=90, l2_miss=10)
        cur = _rec(busy=1200, total=1000, l2_hit=50, l2_miss=50)
        txt = selfcheck.format_result(selfcheck.compare(prev, cur))
        self.assertIn("REGRESSED", txt)
        self.assertIn("l2_hit_rate", txt)

    def test_format_no_baseline(self):
        recs = [_rec(busy=1000, total=1000)]
        ident = schema.identity(recs[0])
        txt = selfcheck.format_result(selfcheck.check_history(recs, ident))
        self.assertIn("no baseline", txt)


if __name__ == "__main__":
    unittest.main()
