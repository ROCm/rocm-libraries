# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the local record store (pure, no GPU)."""
import json
import tempfile
import unittest
from pathlib import Path

from rocke.benchmark.perf import schema
from rocke.benchmark.perf.tool import store


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


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_then_load_round_trip(self):
        r1 = _rec(busy=100, ms=1.0)
        r2 = _rec(busy=110, ms=1.1)
        store.append(r1, cache=self.cache)
        store.append(r2, cache=self.cache)
        got = store.load(cache=self.cache)
        self.assertEqual(got, [r1, r2])          # order + values preserved

    def test_load_missing_is_empty(self):
        self.assertEqual(store.load(cache=self.cache), [])

    def test_blank_and_corrupt_lines_skipped(self):
        store.append(_rec(busy=1), cache=self.cache)
        with store.history_path(self.cache).open("a") as f:
            f.write("\n")
            f.write("{not valid json}\n")
        store.append(_rec(busy=2), cache=self.cache)
        got = store.load(cache=self.cache)
        self.assertEqual([r["counters"]["busy_cycles"] for r in got], [1, 2])

    def test_env_var_resolution(self):
        import os
        old = os.environ.get("ROCKE_PERF_CACHE")
        try:
            os.environ["ROCKE_PERF_CACHE"] = str(self.cache)
            store.append(_rec(busy=7))           # no explicit cache -> env
            got = store.load()
            self.assertEqual(got[-1]["counters"]["busy_cycles"], 7)
        finally:
            if old is None:
                os.environ.pop("ROCKE_PERF_CACHE", None)
            else:
                os.environ["ROCKE_PERF_CACHE"] = old

    def test_append_validates(self):
        bad = {"schema": "wrong"}
        with self.assertRaises(schema.SchemaError):
            store.append(bad, cache=self.cache)

    def test_group_by_identity(self):
        recs = [_rec(kernel="a", shape={"M": 8}, busy=1),
                _rec(kernel="a", shape={"M": 8}, busy=2),
                _rec(kernel="b", shape={"M": 8}, busy=3),
                _rec(kernel="a", shape={"M": 16}, busy=4)]
        groups = store.group_by_identity(recs)
        self.assertEqual(len(groups), 3)         # (a,M=8), (b,M=8), (a,M=16)
        a8 = schema.identity(_rec(kernel="a", shape={"M": 8}))
        self.assertEqual([r["counters"]["busy_cycles"] for r in groups[a8]], [1, 2])

    def test_records_for_filters_identity(self):
        recs = [_rec(kernel="a", busy=1), _rec(kernel="b", busy=2),
                _rec(kernel="a", busy=3)]
        ident = schema.identity(_rec(kernel="a"))
        got = store.records_for(recs, ident)
        self.assertEqual([r["counters"]["busy_cycles"] for r in got], [1, 3])


if __name__ == "__main__":
    unittest.main()
