# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU unit tests for the advisory gfx1250 VGPR budget estimate in
gemm_validation_utils (estimate_vgpr_usage / is_vgpr_budget_ok /
vgpr_budget_tags). No GPU is required."""

import logging
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gemm_validation_utils as vu  # noqa: E402


def _bf16(tile, warps, pipeline="compv3", warp_tile=(16, 16, 32), **kwargs):
    return dict(
        tile_m=tile[0],
        tile_n=tile[1],
        tile_k=tile[2],
        warp_m=warps[0],
        warp_n=warps[1],
        warp_k=warps[2],
        warp_tile_m=warp_tile[0],
        warp_tile_n=warp_tile[1],
        warp_tile_k=warp_tile[2],
        a_datatype="bf16",
        b_datatype="bf16",
        pipeline=pipeline,
        gpu_target="gfx1250",
        **kwargs,
    )


class TestVgprBudgetPerWave(unittest.TestCase):
    def test_block_size_table(self):
        # 128 / 256 / 512 / 1024 threads at wave32.
        self.assertEqual(vu.get_vgpr_budget_per_wave(4), 1024)
        self.assertEqual(vu.get_vgpr_budget_per_wave(8), 512)
        self.assertEqual(vu.get_vgpr_budget_per_wave(16), 256)
        self.assertEqual(vu.get_vgpr_budget_per_wave(32), 128)

    def test_small_blocks_and_k_block_per_cu(self):
        self.assertEqual(vu.get_vgpr_budget_per_wave(1), 1024)
        self.assertEqual(vu.get_vgpr_budget_per_wave(4, k_block_per_cu=2), 512)
        self.assertEqual(vu.get_vgpr_budget_per_wave(8, k_block_per_cu=2), 512)
        self.assertEqual(vu.get_vgpr_budget_per_wave(8, k_block_per_cu=3), 336)

    def test_target_suffix_and_unknown_arch(self):
        self.assertEqual(vu.get_vgpr_budget_per_wave(8, 1, "gfx1250:xnack-"), 512)
        self.assertEqual(vu.get_vgpr_budget_per_wave(8, 1, "gfx942"), 0)
        self.assertEqual(vu.get_vgpr_budget_per_wave(8, 1, "gfx950"), 0)


class TestVgprEstimateWorkedNumbers(unittest.TestCase):
    """Plan M2 B2 worked results (CompV3, bf16, 16x16x32 WMMA)."""

    def test_256x256x64_w2x4_over_budget(self):
        est = vu.estimate_vgpr_usage(**_bf16((256, 256, 64), (2, 4, 1)))
        self.assertEqual(
            (est["acc"], est["frag"], est["pref"], est["overhead"]),
            (256, 192, 64, 32),
        )
        self.assertEqual(est["estimate"], 544)
        self.assertEqual(est["budget"], 512)
        ok, reason = vu.is_vgpr_budget_ok(**_bf16((256, 256, 64), (2, 4, 1)))
        self.assertFalse(ok)
        self.assertIn("544", reason)
        self.assertIn("512", reason)

    def test_256x256x64_w8x4_over_budget(self):
        est = vu.estimate_vgpr_usage(**_bf16((256, 256, 64), (8, 4, 1)))
        self.assertEqual(
            (est["acc"], est["frag"], est["pref"], est["overhead"]),
            (64, 96, 16, 32),
        )
        self.assertEqual(est["estimate"], 208)
        self.assertEqual(est["budget"], 128)
        self.assertFalse(vu.is_vgpr_budget_ok(**_bf16((256, 256, 64), (8, 4, 1)))[0])

    def test_256x128x64_w4x2_within_budget(self):
        est = vu.estimate_vgpr_usage(**_bf16((256, 128, 64), (4, 2, 1)))
        self.assertEqual(
            (est["acc"], est["frag"], est["pref"], est["overhead"]),
            (128, 128, 48, 32),
        )
        self.assertEqual(est["estimate"], 336)
        self.assertEqual(est["budget"], 512)
        self.assertEqual(
            vu.is_vgpr_budget_ok(**_bf16((256, 128, 64), (4, 2, 1))), (True, "")
        )


class TestVgprEstimateTerms(unittest.TestCase):
    def test_async_and_tdm_have_no_prefetch_term(self):
        for pipeline in (
            "comp_async",
            "comp_async_eight_waves",
            "comp_tdm",
            "comp_tdm_v2",
        ):
            est = vu.estimate_vgpr_usage(
                **_bf16((256, 256, 64), (2, 4, 1), pipeline=pipeline)
            )
            self.assertEqual(est["pref"], 0, pipeline)
            self.assertEqual(est["estimate"], 480, pipeline)

    def test_mem_and_compv4_keep_prefetch_term(self):
        for pipeline in ("mem", "compv4"):
            est = vu.estimate_vgpr_usage(
                **_bf16((256, 256, 64), (2, 4, 1), pipeline=pipeline)
            )
            self.assertEqual(est["pref"], 64, pipeline)

    def test_scale_factors(self):
        est = vu.estimate_vgpr_usage(
            **_bf16((256, 256, 64), (2, 4, 1), frag_scale=0.5, prefetch_scale=0.5)
        )
        self.assertEqual((est["frag"], est["pref"]), (96, 32))
        self.assertEqual(est["estimate"], 256 + 96 + 32 + 32)

    def test_8bit_operands(self):
        args = _bf16((256, 128, 128), (4, 2, 1), warp_tile=(16, 16, 64))
        args.update(a_datatype="fp8", b_datatype="fp8")
        est = vu.estimate_vgpr_usage(**args)
        # 2 K steps x (4 + 4) atoms x 8 VGPRs; (256 + 128) * 128 B / 256 / 4.
        self.assertEqual((est["acc"], est["frag"], est["pref"]), (128, 128, 48))

    def test_k_block_per_cu_lowers_budget(self):
        ok, _ = vu.is_vgpr_budget_ok(
            **_bf16((256, 128, 64), (4, 2, 1), k_block_per_cu=4)
        )
        self.assertFalse(ok)  # 336 > 1024 / 4


class TestVgprBudgetTags(unittest.TestCase):
    def test_tags(self):
        self.assertEqual(
            vu.vgpr_budget_tags(**_bf16((256, 256, 64), (2, 4, 1))),
            [vu.VGPR_TAG_OVER_BUDGET, vu.VGPR_TAG_ABOVE_FAST],
        )
        self.assertEqual(
            vu.vgpr_budget_tags(**_bf16((256, 256, 64), (8, 4, 1))),
            [vu.VGPR_TAG_OVER_BUDGET],
        )
        self.assertEqual(
            vu.vgpr_budget_tags(**_bf16((256, 128, 64), (4, 2, 1))),
            [vu.VGPR_TAG_ABOVE_FAST],
        )
        self.assertEqual(vu.vgpr_budget_tags(**_bf16((128, 128, 32), (2, 2, 1))), [])

    def test_unmodelled_arch_is_untagged(self):
        args = _bf16((256, 256, 64), (2, 4, 1))
        args["gpu_target"] = "gfx942"
        self.assertEqual(vu.vgpr_budget_tags(**args), [])
        self.assertEqual(vu.is_vgpr_budget_ok(**args), (True, ""))


class TestVgprTagIsAdvisory(unittest.TestCase):
    """The tag must never change is_tile_config_valid's verdict."""

    def _valid(self, tile, warps):
        return vu.is_tile_config_valid(
            tile[0],
            tile[1],
            tile[2],
            warps[0],
            warps[1],
            warps[2],
            16,
            16,
            32,
            "bf16",
            "bf16",
            "bf16",
            "compv3",
            "rcr",
            "gfx1250",
        )

    def test_over_budget_config_still_valid(self):
        with self.assertLogs(level="DEBUG") as logs:
            self.assertTrue(self._valid((256, 256, 64), (2, 4, 1)))
        self.assertTrue(any(vu.VGPR_TAG_OVER_BUDGET in line for line in logs.output))

    def test_in_budget_config_unchanged(self):
        self.assertTrue(self._valid((256, 128, 64), (4, 2, 1)))

    def test_estimate_skipped_without_debug_logging(self):
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.WARNING)
        try:
            with mock.patch.object(vu, "vgpr_budget_tags") as tags:
                self.assertTrue(self._valid((256, 256, 64), (2, 4, 1)))
            tags.assert_not_called()
        finally:
            root.setLevel(old_level)


if __name__ == "__main__":
    unittest.main()
