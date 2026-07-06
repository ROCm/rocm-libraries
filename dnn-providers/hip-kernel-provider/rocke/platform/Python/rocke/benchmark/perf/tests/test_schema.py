# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the measurement-record schema (the contract)."""

from __future__ import annotations

import unittest

from rocke.benchmark.perf import schema


def _rec(arch="gfx950", kernel="k", shape=None, *, busy=None, ms=None):
    counters = {"busy_cycles": busy} if busy is not None else {}
    wall = {"ms_median": ms} if ms is not None else {}
    return {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": kernel, "op": "gemm", "shape": shape or {"M": 8}},
        "wall": wall,
        "counters": counters,
        "resources": {},
        "derived": {},
        "captured_counters": sorted(counters),
        "verify": {},
    }


class TestValidate(unittest.TestCase):
    def test_valid_record_passes(self):
        schema.validate(_rec(busy=100))  # no raise

    def test_missing_top_level_key(self):
        rec = _rec(busy=100)
        del rec["wall"]
        with self.assertRaises(schema.SchemaError):
            schema.validate(rec)

    def test_schema_version_mismatch(self):
        rec = _rec(busy=100)
        rec["schema"] = "something/else"
        with self.assertRaises(schema.SchemaError):
            schema.validate(rec)

    def test_missing_run_field(self):
        rec = _rec(busy=100)
        del rec["run"]["arch"]
        with self.assertRaises(schema.SchemaError):
            schema.validate(rec)

    def test_missing_kernel_field(self):
        rec = _rec(busy=100)
        del rec["kernel"]["op"]
        with self.assertRaises(schema.SchemaError):
            schema.validate(rec)

    def test_shape_must_be_mapping(self):
        rec = _rec(busy=100)
        rec["kernel"]["shape"] = [8, 8]
        with self.assertRaises(schema.SchemaError):
            schema.validate(rec)


class TestShapeSignature(unittest.TestCase):
    def test_sorted_key_serialization(self):
        self.assertEqual(
            schema.shape_signature({"M": 512, "N": 256, "K": 128}),
            "K=128,M=512,N=256",
        )

    def test_empty_shape(self):
        self.assertEqual(schema.shape_signature({}), "")

    def test_op_agnostic(self):
        # non-GEMM dims serialize the same way, no op privileged
        self.assertEqual(
            schema.shape_signature({"batch": 2, "heads": 8, "seqlen": 1024}),
            "batch=2,heads=8,seqlen=1024",
        )


class TestIdentity(unittest.TestCase):
    def test_identity_tuple(self):
        rec = _rec(arch="gfx1201", kernel="mygemm", shape={"M": 8, "N": 8, "K": 8})
        self.assertEqual(
            schema.identity(rec),
            ("gfx1201", "mygemm", "K=8,M=8,N=8"),
        )

    def test_differing_shape_differs(self):
        a = schema.identity(_rec(shape={"M": 8}))
        b = schema.identity(_rec(shape={"M": 16}))
        self.assertNotEqual(a, b)


class TestMetric(unittest.TestCase):
    def test_primary_cycle_metric_when_present(self):
        val, which = schema.metric(_rec(busy=1234, ms=5.0))
        self.assertEqual(which, schema.PRIMARY_METRIC)
        self.assertEqual(val, 1234.0)

    def test_falls_back_to_wall(self):
        val, which = schema.metric(_rec(ms=5.0))  # no counters
        self.assertEqual(which, schema.FALLBACK_METRIC)
        self.assertEqual(val, 5.0)

    def test_none_when_neither(self):
        val, which = schema.metric(_rec())
        self.assertIsNone(val)
        self.assertEqual(which, "")


class TestCaptured(unittest.TestCase):
    def test_returns_captured_list(self):
        rec = _rec(busy=1)
        rec["captured_counters"] = ["busy_cycles", "waves"]
        self.assertEqual(schema.captured(rec), ["busy_cycles", "waves"])

    def test_empty_when_absent(self):
        rec = _rec()
        rec.pop("captured_counters", None)
        self.assertEqual(schema.captured(rec), [])


if __name__ == "__main__":
    unittest.main()
