# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batch prefill codegen: per-target kernel selection and the rendered dispatch (CPU only)."""

import itertools
import re
import sys
import unittest
from pathlib import Path

_CK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CK_ROOT / "example" / "ck_tile" / "01_fmha"))

from codegen.ops.fmha_batch_prefill import get_fwd_blobs  # noqa: E402

_PAGE_TILE = "qr_tdm_sched"


def _blobs(targets, receipt=600, kernel_filter="", mask_impl="simplified"):
    return get_fwd_blobs(kernel_filter, receipt, [-1], mask_impl, targets)


def _arms(api: str):
    """One entry per rendered dispatch arm: (condition, trait)."""
    return re.findall(
        r"if\((\(t\.is_group_mode.*?)\) \{\s*using trait_ = ([^;]+);", api, re.S
    )


class BatchPrefillCodegenTest(unittest.TestCase):
    def test_gfx9_targets_keep_only_the_async_pipeline(self):
        pool, kernels = _blobs(["gfx942", "gfx950"])
        self.assertTrue(kernels)
        self.assertFalse([k for k in kernels if k.F_pipeline.tag == _PAGE_TILE])
        # No arch guard is added when only gfx9 is generated.
        self.assertFalse([k for k in kernels if "__gfx9__" in k.template])
        self.assertNotIn("gfx125", pool.api)

    def test_other_targets_generate_nothing(self):
        for targets in (["gfx1100"], ["gfx1200"], ["gfx1250", "gfx1100"]):
            pool, kernels = _blobs(targets)
            self.assertEqual(kernels, [], targets)
            self.assertNotIn("fmha_batch_prefill_<", pool.api)

    def test_gfx125_generates_the_page_tile_product(self):
        pool, kernels = _blobs(["gfx1250"])
        self.assertTrue(all(k.F_pipeline.tag == _PAGE_TILE for k in kernels))
        product = {
            (
                k.F_dtype,
                k.F_hdim,
                k.F_page_size,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_logits,
                k.F_pipeline.F_bias,
                k.F_pipeline.F_kv_lookup_table,
                k.F_pipeline.F_kv_memory_layout,
            )
            for k in kernels
        }
        expected = {
            c
            for c in itertools.product(
                ["fp16", "bf16"],
                [64, 128],
                [16, 64, 128, 256, 1024],
                ["s_no", "s_mask"],
                ["t", "f"],
                ["t", "f"],
                ["t", "f"],
                ["no", "bias", "alibi"],
                ["vllm", "sglang"],
                ["linear", "vectorized"],
            )
            # sink instances are masked only; the soft cap is unbiased only
            if not (c[5] == "t" and c[3] == "s_no")
            and not (c[6] == "t" and c[7] != "no")
        }
        self.assertEqual(len(kernels), 1920)
        self.assertEqual(product, expected)
        for k in kernels:
            self.assertEqual(k.F_mode, "group")
            body = k.template
            self.assertIn("ck_tile::FmhaBatchPrefillQrTdmPageTileKernel<", body)
            self.assertIn("k_::MakeCheckedKernel(kargs, grids)", body)
            self.assertIn("(defined(__gfx125__))", body)

    def test_gfx125_dispatch_conditions(self):
        pool, kernels = _blobs(["gfx1250"])
        api = pool.api
        arms = _arms(api)
        self.assertEqual(len(arms), len(kernels))
        # Page-tile arms take no kv_load_mode selection.
        self.assertNotIn("kElementBytes", api)
        for cond, trait in arms:
            self.assertIn('device_name.compare(0, 6, "gfx125") == 0', cond)
            self.assertIn("BlockFmhaPipelineEnum::QRKSVS_TDM_SCHED", trait)
            hdim = int(trait.split("<", 1)[1].split(",", 1)[0])
            self.assertIn(f"(a.hdim_q == {hdim}) && (a.hdim_v == {hdim})", cond)
            if "SimplifiedGenericAttentionMask<true>" in trait:
                # Any window from either corner: no window clause in the arm.
                self.assertIn("t.mask_type != mask_enum::no_mask", cond)
                self.assertNotIn("window_size", cond)
            else:
                self.assertIn("t.mask_type == mask_enum::no_mask", cond)
        # Head-dim cases ascend, so a D64 problem is not captured by the D128 case.
        self.assertLess(api.index("t.hdim_q <= 64 "), api.index("t.hdim_q <= 128 "))

    def test_generic_mask_impl_still_uses_simplified_page_tile_masks(self):
        pool, kernels = _blobs(["gfx1250"], mask_impl="generic")
        self.assertEqual(len(kernels), 1920)
        self.assertFalse([k for k in kernels if "FmhaMasks::" in k.template])

    def test_mixed_build_guards_the_gfx9_kernels(self):
        _, gfx9 = _blobs(["gfx942"])
        pool, kernels = _blobs(["gfx942", "gfx1250"])
        page_tile = [k for k in kernels if k.F_pipeline.tag == _PAGE_TILE]
        old = [k for k in kernels if k.F_pipeline.tag != _PAGE_TILE]
        self.assertEqual(len(page_tile), 1920)
        self.assertEqual(sorted(k.name for k in old), sorted(k.name for k in gfx9))
        for k in old:
            self.assertRegex(k.template, r"defined\(__gfx9(42)?__\)")
        names = [k.filename for k in kernels]
        self.assertEqual(len(names), len(set(names)))
        for cond, trait in _arms(pool.api):
            if "QRKSVS_TDM_SCHED" in trait:
                self.assertIn('"gfx125"', cond)
            else:
                self.assertRegex(cond, r'device_name\.compare\(0, [46], "gfx9(42)?"\)')

    def test_aiter_receipts_and_filter_keep_the_page_tile_kernels(self):
        for receipt in (200, 600):
            _, kernels = _blobs(["gfx1250"], receipt, kernel_filter="*_ndropout_*")
            self.assertEqual(len(kernels), 1920, receipt)
        _, kernels = _blobs(["gfx1250"], 100)
        self.assertEqual(kernels, [])


if __name__ == "__main__":
    unittest.main()
