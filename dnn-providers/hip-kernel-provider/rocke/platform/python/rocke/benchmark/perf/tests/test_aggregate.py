# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the sampling primitive (pure, no GPU)."""
import unittest

from rocke.benchmark.perf import aggregate, schema


def _rec(
    arch="gfx950",
    kernel="k",
    shape=None,
    *,
    op="gemm",
    busy=None,
    total=None,
    ms=None,
    prof_ms=None,
    l2_hit=None,
    l2_miss=None,
    counter_samples=None,
):
    """Minimal schema-valid single-run record with the fields under test."""
    counters = {}
    if busy is not None:
        counters["busy_cycles"] = busy
    if total is not None:
        counters["total_clocks"] = total
    if l2_hit is not None:
        counters["l2_hit"] = l2_hit
    if l2_miss is not None:
        counters["l2_miss"] = l2_miss
    wall = {"ms_median": ms} if ms is not None else {}
    profiled = {"ms_median": prof_ms} if prof_ms is not None else {}
    record = {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": kernel, "op": op, "shape": shape or {"M": 8}},
        "wall": wall,
        "profiled": profiled,
        "counters": counters,
        "resources": {},
        "derived": {},
        "captured_counters": sorted(counters),
        "verify": {},
    }
    if counter_samples is not None:
        record["counter_samples"] = counter_samples
    return record


class TestProfiled(unittest.TestCase):
    def test_profiled_medianed_and_overhead(self):
        # wall (real) faster than profiled (profiler overhead)
        recs = [_rec(busy=b, total=1000, ms=1.0, prof_ms=1.2) for b in (100, 110, 120)]
        out = aggregate.aggregate(recs)
        self.assertEqual(out["profiled"]["ms_median"], 1.2)
        self.assertEqual(out["wall"]["ms_median"], 1.0)
        self.assertAlmostEqual(out["derived"]["profiler_overhead_pct"], 20.0)

    def test_no_overhead_when_profiled_absent(self):
        recs = [_rec(busy=100, total=1000, ms=1.0) for _ in range(2)]
        out = aggregate.aggregate(recs)
        self.assertEqual(out["profiled"], {})
        self.assertNotIn("profiler_overhead_pct", out["derived"])


class TestAggregate(unittest.TestCase):
    def test_median_and_spread(self):
        recs = [
            _rec(busy=b, total=1000, ms=m)
            for b, m in [(100, 1.0), (110, 2.0), (120, 3.0)]
        ]
        out = aggregate.aggregate(recs)
        self.assertEqual(out["counters"]["busy_cycles"], 110)
        self.assertEqual(out["wall"]["ms_median"], 2.0)
        self.assertEqual(out["n_samples"], 3)
        # peak-to-peak / |median| * 100
        self.assertAlmostEqual(out["spread"]["busy_cycles_pct"], 20 / 110 * 100)
        self.assertAlmostEqual(out["spread"]["ms_pct"], 100.0)
        self.assertAlmostEqual(out["wall"]["ms_spread_pct"], 100.0)

    def test_k_configurable(self):
        for k in (1, 2, 5, 7):
            recs = [_rec(busy=100 + i, total=1000, ms=1.0) for i in range(k)]
            out = aggregate.aggregate(recs)
            self.assertEqual(out["n_samples"], k)
            self.assertEqual(len(out["wall"]["samples"]), k)

    def test_single_record_zero_spread(self):
        out = aggregate.aggregate([_rec(busy=100, total=1000, ms=1.0)])
        self.assertEqual(out["spread"]["busy_cycles_pct"], 0.0)
        self.assertEqual(out["wall"]["ms_spread_pct"], 0.0)

    def test_derived_recomputed_from_medians(self):
        recs = [
            _rec(busy=b, total=1000, l2_hit=h, l2_miss=m)
            for b, h, m in [(500, 90, 10), (600, 80, 20), (700, 70, 30)]
        ]
        out = aggregate.aggregate(recs)
        # medians: busy 600, total 1000, l2_hit 80, l2_miss 20
        self.assertAlmostEqual(out["derived"]["busy_fraction"], 0.6)
        self.assertAlmostEqual(out["derived"]["l2_hit_rate"], 80 / 100)

    def test_partial_counter_capture_is_not_reported_as_stable(self):
        out = aggregate.aggregate(
            [_rec(busy=100, total=1000), _rec(ms=1.0), _rec(busy=120, total=1000)]
        )
        self.assertNotIn("busy_cycles", out["counters"])
        self.assertIsNone(out["spread"]["busy_cycles_pct"])

    def test_missing_l2_counter_omits_derived_ratio(self):
        out = aggregate.aggregate([_rec(l2_hit=90), _rec(l2_hit=80), _rec(l2_hit=70)])
        self.assertNotIn("l2_hit_rate", out["derived"])

    def test_verification_combines_every_repeat(self):
        recs = [_rec(busy=100), _rec(busy=110)]
        recs[0]["verify"] = {
            "ok": True,
            "max_abs_diff": 0.1,
            "bad_count": 0,
            "total": 8,
        }
        recs[1]["verify"] = {
            "ok": False,
            "max_abs_diff": 0.2,
            "bad_count": 1,
            "total": 8,
        }
        out = aggregate.aggregate(recs)
        self.assertEqual(
            out["verify"],
            {"ok": False, "max_abs_diff": 0.2, "bad_count": 1, "total": 16},
        )

    def test_missing_verification_does_not_inherit_first_repeat(self):
        recs = [_rec(busy=100), _rec(busy=110)]
        recs[0]["verify"] = {"ok": True}
        self.assertEqual(aggregate.aggregate(recs)["verify"], {})

    def test_tier_w_only(self):
        recs = [_rec(ms=m) for m in (1.0, 1.5, 2.0)]
        out = aggregate.aggregate(recs)
        self.assertEqual(out["counters"], {})
        self.assertEqual(out["captured_counters"], [])
        self.assertIsNone(out["spread"]["busy_cycles_pct"])
        self.assertEqual(out["wall"]["ms_median"], 1.5)
        self.assertEqual(out["derived"], {})

    def test_mixed_identity_raises(self):
        with self.assertRaises(ValueError):
            aggregate.aggregate([_rec(kernel="a"), _rec(kernel="b")])
        with self.assertRaises(ValueError):
            aggregate.aggregate([_rec(shape={"M": 8}), _rec(shape={"M": 16})])
        with self.assertRaises(ValueError):
            aggregate.aggregate([_rec(op="gemm"), _rec(op="conv")])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            aggregate.aggregate([])

    def test_output_is_schema_valid(self):
        out = aggregate.aggregate([_rec(busy=100, total=1000, ms=1.0)])
        schema.validate(out)  # raises on failure

    def test_counter_samples_preserved_across_repeats(self):
        recs = [
            _rec(
                busy=100,
                counter_samples=[
                    {"dispatch_id": 5, "duration_ns": 10, "busy_cycles": 100}
                ],
            ),
            _rec(
                busy=110,
                counter_samples=[
                    {"dispatch_id": 5, "duration_ns": 12, "busy_cycles": 110}
                ],
            ),
        ]
        out = aggregate.aggregate(recs)
        self.assertEqual(
            out["counter_samples"],
            [
                {
                    "dispatch_id": 5,
                    "duration_ns": 10,
                    "busy_cycles": 100,
                    "sample_index": 0,
                },
                {
                    "dispatch_id": 5,
                    "duration_ns": 12,
                    "busy_cycles": 110,
                    "sample_index": 1,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
