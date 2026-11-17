#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import batchnorm_inference_plan as plan  # noqa: E402
import generate_batchnorm_inference_bundles as gen  # noqa: E402


class TestPlanIds(unittest.TestCase):
    def test_round_trip(self):
        for cid in (
            "4_3_1_fp32_ncl",
            "2_3_1_1_bfp16_nchw_upcast",
            "1_16_112_112_fp32_nhwc_rlo0p1_norslope_rhi0p5",
        ):
            spec = plan.parse_id(cid)
            self.assertEqual(plan.make_id(spec["dims"], spec["dt"], spec["layout"], spec["clip"], spec["upcast"]), cid)

    def test_rejects_unknown_layout(self):
        with self.assertRaises(ValueError):
            plan.parse_id("4_3_1_fp32_nxyz")

    def test_channel_last_strides(self):
        self.assertEqual(plan.strides([2, 3, 4, 5], "nhwc"), [60, 1, 15, 3])
        self.assertEqual(plan.strides([2, 3, 4, 5], "nchw"), [60, 20, 5, 1])


class TestTierRules(unittest.TestCase):
    def all_ids(self):
        for pointwise in (False, True):
            for tier in gen.TIERS:
                yield tier, pointwise, gen.tier_ids(tier, pointwise)

    def test_ids_unique_and_parseable(self):
        for tier, pointwise, ids in self.all_ids():
            self.assertEqual(len(ids), len(set(ids)), (tier, pointwise))
            for cid in ids:
                plan.parse_id(cid)

    def test_quick_is_subset_of_standard(self):
        for pointwise in (False, True):
            self.assertLessEqual(set(gen.tier_ids("quick", pointwise)), set(gen.tier_ids("standard", pointwise)))

    def test_standard_comprehensive_full_are_disjoint(self):
        for pointwise in (False, True):
            seen = set()
            for tier in ("standard", "comprehensive", "full"):
                ids = set(gen.tier_ids(tier, pointwise))
                self.assertFalse(seen & ids, (tier, pointwise))
                seen |= ids

    def test_quick_covers_every_layout_and_dtype(self):
        for pointwise in (False, True):
            specs = [plan.parse_id(c) for c in gen.tier_ids("quick", pointwise)]
            self.assertEqual({s["layout"] for s in specs}, set(plan.LAYOUTS))
            self.assertEqual({s["dt"] for s in specs}, set(plan.DTYPES))
            self.assertTrue(any(s["upcast"] for s in specs))

    def test_quick_stays_small(self):
        for pointwise in (False, True):
            for cid in gen.tier_ids("quick", pointwise):
                self.assertLess(plan.elements(plan.parse_id(cid)["dims"]), 10_000)

    def test_only_pointwise_has_clips(self):
        for tier in gen.TIERS:
            self.assertTrue(all(plan.parse_id(c)["clip"] == "" for c in gen.tier_ids(tier, False)))
        self.assertTrue(any(plan.parse_id(c)["clip"] for c in gen.tier_ids("standard", True)))

    def test_leaky_only_in_full(self):
        for tier in gen.TIERS:
            leaky = [c for c in gen.tier_ids(tier, True) if plan.parse_id(c)["clip"] == gen.LEAKY]
            self.assertEqual(bool(leaky), tier == "full", tier)

    def test_upcast_is_16_bit_only(self):
        for _, _, ids in self.all_ids():
            for cid in ids:
                spec = plan.parse_id(cid)
                if spec["upcast"]:
                    self.assertNotEqual(spec["dt"], "fp32", cid)

    def test_large_shapes_only_in_full(self):
        large = {tuple(d) for d in gen.LARGE}
        for tier in ("quick", "standard", "comprehensive"):
            for pointwise in (False, True):
                dims = {tuple(plan.parse_id(c)["dims"]) for c in gen.tier_ids(tier, pointwise)}
                self.assertFalse(dims & large, tier)


class TestClaims(unittest.TestCase):
    def test_leaky_unclaimed_by_both(self):
        spec = plan.parse_id("4_3_1_fp32_ncl_norlo_rslope0p01_norhi")
        self.assertFalse(any(gen.claimed_by(e, spec) for e in gen.ENGINES))

    def test_miopen_skips_upcast(self):
        spec = plan.parse_id("2_3_1_1_fp16_nchw_upcast")
        self.assertTrue(gen.claimed_by("HIP_MLOPS_ENGINE", spec))
        self.assertFalse(gen.claimed_by("MIOPEN_ENGINE", spec))

    def test_build_support_partitions_ids(self):
        ids = ["4_3_1_fp32_ncl", "2_3_1_1_fp16_nchw_upcast"]
        old = {"claims": {e: [{"cases": [], "support": {"gfx942": ["linux"]}}] for e in gen.ENGINES}}
        claims = gen.build_support(ids, old)["claims"]
        self.assertEqual(claims["HIP_MLOPS_ENGINE"][0]["cases"], sorted(ids))
        self.assertEqual(claims["MIOPEN_ENGINE"][0]["cases"], ["4_3_1_fp32_ncl"])


class TestCheckedInBundles(unittest.TestCase):
    def test_bundles_match_generator(self):
        self.assertEqual(gen.main(["--dry-run"]), 0)

    def test_every_case_has_generator_and_reference_source(self):
        for op in gen.OPS:
            for tier in gen.TIERS:
                sweep = json.loads((gen.DEFAULT_BUNDLE_DIR / tier / op / "Default/sweep.json").read_text())
                for case in sweep["cases"]:
                    for key in ("generator", "reference_source"):
                        self.assertTrue(case["metadata"].get(key), (tier, op, case["id"], key))

    def test_pointwise_values_shape(self):
        sweep = json.loads(
            (gen.DEFAULT_BUNDLE_DIR / "quick/BatchnormInferencePointwise/Default/sweep.json").read_text()
        )
        for case in sweep["cases"]:
            self.assertIn("attributes", case["values"])
            self.assertEqual(len(case["values"]["tensors"]), 7)


if __name__ == "__main__":
    unittest.main()
