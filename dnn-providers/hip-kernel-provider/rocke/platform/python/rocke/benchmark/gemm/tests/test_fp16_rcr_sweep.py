# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the FP16 RCR GEMM benchmark sweep harness."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rocke.benchmark.gemm.fp16_rcr_sweep import (
    GemmRunRecord,
    GemmSweepConfig,
    GemmSweepShape,
    default_gemm_shapes,
    expand_sweep,
    _parse_shape,
    write_sweep_csv,
    write_sweep_json,
)


class TestGemmFp16RcrSweepPlan(unittest.TestCase):
    def test_default_shapes_include_correctness_and_large_examples(self):
        shapes = default_gemm_shapes()
        self.assertTrue(any(s.verify for s in shapes))
        self.assertIn((4096, 4096, 4096), {s.as_tuple() for s in shapes})

    def test_expand_sweep_uses_registered_candidates_and_kernel_ids(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                shapes=(GemmSweepShape(128, 128, 32, "small", verify=True),),
            )
        )
        self.assertGreaterEqual(len(plan.variants), 1)
        cache_keys = [v.cache_key for v in plan.variants]
        self.assertEqual(len(cache_keys), len(set(cache_keys)))
        for variant in plan.variants:
            self.assertEqual(variant.kernel_id["op"], "gemm")
            self.assertEqual(variant.kernel_id["arch"], "gfx950")
            self.assertEqual(variant.shape.label, "small")

    def test_explicit_spec_id_limits_sweep(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                spec_id="cdna_mem_64x128",
                shapes=(GemmSweepShape(128, 128, 32, "explicit"),),
            )
        )
        self.assertEqual({v.spec_id for v in plan.variants}, {"cdna_mem_64x128"})

    def test_non_granular_shape_is_filtered(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                shapes=(GemmSweepShape(130, 128, 32, "bad-m"),),
            )
        )
        self.assertEqual(plan.variants, ())
        self.assertTrue(any("not divisible" in f.reason for f in plan.filtered))

    def test_write_sweep_json_schema(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                shapes=(GemmSweepShape(128, 128, 32, "json"),),
            )
        )
        out = Path(tempfile.mkdtemp(prefix="rocke_gemm_sweep_test_")) / "sweep.json"
        write_sweep_json(out, plan)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema"], "ck.dsl.benchmark.gemm.fp16_rcr_sweep/v1")
        self.assertIn("variants", doc)
        self.assertIn("filtered", doc)
        self.assertIn("builds", doc)
        self.assertIn("runs", doc)

    def test_csv_maps_perf_to_kernel_key_and_config(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                shapes=(GemmSweepShape(128, 128, 32, "csv"),),
            )
        )
        self.assertGreaterEqual(len(plan.variants), 1)
        runs = tuple(
            GemmRunRecord(
                cache_key=v.cache_key,
                shape=v.shape,
                ok=True,
                verify=v.shape.verify,
                ms=1.0 + i,
                tflops=10.0 + i,
                gbps=5.0 + i,
            )
            for i, v in enumerate(plan.variants)
        )
        out = Path(tempfile.mkdtemp(prefix="rocke_gemm_csv_test_")) / "sweep.csv"
        write_sweep_csv(out, runs, plan=plan)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        self.assertEqual(len(rows), len(plan.variants))
        variant_by_key = {v.cache_key: v for v in plan.variants}
        for row in rows:
            # The kernel key must be present and resolvable back to a variant.
            self.assertIn(row["cache_key"], variant_by_key)
            variant = variant_by_key[row["cache_key"]]
            self.assertEqual(row["candidate"], variant.candidate)
            self.assertEqual(row["spec_id"], variant.spec_id)
            self.assertEqual(row["arch"], "gfx950")
            self.assertEqual(row["spec_hash"], variant.kernel_id["spec_hash"])
            # Config columns must reflect the variant's spec, not be blank.
            self.assertEqual(int(row["tile_m"]), variant.spec["tile"]["tile_m"])
            self.assertEqual(int(row["tile_n"]), variant.spec["tile"]["tile_n"])
            self.assertEqual(row["pipeline"], variant.spec["trait"]["pipeline"])
            self.assertEqual(row["dtype_a"], variant.spec["data"]["dtype_a"])
            self.assertEqual(int(row["block_size"]), variant.spec["block_size"])

    def test_csv_without_plan_leaves_config_blank(self):
        shape = GemmSweepShape(128, 128, 32, "noplan")
        runs = (
            GemmRunRecord(
                cache_key="k1",
                shape=shape,
                ok=True,
                verify=False,
                ms=1.0,
                tflops=2.0,
                gbps=3.0,
            ),
        )
        out = Path(tempfile.mkdtemp(prefix="rocke_gemm_csv_noplan_")) / "sweep.csv"
        write_sweep_csv(out, runs)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        self.assertEqual(rows[0]["cache_key"], "k1")
        self.assertEqual(rows[0]["tile_m"], "")
        self.assertEqual(rows[0]["candidate"], "")

    def test_shape_parser(self):
        shape = _parse_shape("128,256,64:small:true")
        self.assertEqual(shape.as_tuple(), (128, 256, 64))
        self.assertEqual(shape.label, "small")
        self.assertTrue(shape.verify)


if __name__ == "__main__":
    unittest.main()
