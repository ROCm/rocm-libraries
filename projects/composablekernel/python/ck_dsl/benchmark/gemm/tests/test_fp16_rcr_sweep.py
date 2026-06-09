# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the FP16 RCR GEMM benchmark sweep harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ck_dsl.benchmark.gemm.fp16_rcr_sweep import (
    GemmSweepConfig,
    GemmSweepShape,
    default_gemm_shapes,
    expand_sweep,
    _parse_shape,
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
        out = Path(tempfile.mkdtemp(prefix="ckdsl_gemm_sweep_test_")) / "sweep.json"
        write_sweep_json(out, plan)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema"], "ck.dsl.benchmark.gemm.fp16_rcr_sweep/v1")
        self.assertIn("variants", doc)
        self.assertIn("filtered", doc)
        self.assertIn("builds", doc)
        self.assertIn("runs", doc)

    def test_shape_parser(self):
        shape = _parse_shape("128,256,64:small:true")
        self.assertEqual(shape.as_tuple(), (128, 256, 64))
        self.assertEqual(shape.label, "small")
        self.assertTrue(shape.verify)


if __name__ == "__main__":
    unittest.main()
