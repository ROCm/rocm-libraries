# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Public gfx1151 selection, unsupported requests, and cross-architecture isolation."""

import unittest

from dispatch.attention import AttentionMaskType, AttentionRequest, dispatch_attention


def _request(**changes):
    fields = dict(
        batch=2,
        nhead_q=16,
        nhead_k=4,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch="gfx1151",
        dtype="fp16",
        num_cus=64,
    )
    fields.update(changes)
    return AttentionRequest(**fields)


class TestGfx1151AttentionSelection(unittest.TestCase):
    def test_auto_and_explicit_selectors_reach_the_executable_kernel(self):
        for selectors in (
            {},
            {"algorithm": "wmma_fmha_fwd"},
            {"spec_id": "gfx1151_wmma_fmha_fwd"},
        ):
            with self.subTest(selectors=selectors):
                result = dispatch_attention(_request(layout="dense", **selectors))
                self.assertEqual(result.candidate.name, "attention_gfx1151_wmma")
        with self.assertRaises(ValueError):
            dispatch_attention(_request(layout="dense", algorithm="unified_2d"))

    def test_unsupported_requests_cannot_fall_back_to_a_different_operation(self):
        cases = (
            {"dtype": "fp32"},
            {"use_fp8": True, "fp8_fnuz": True},
            {"nhead_q": 15},
            {"hdim_q": 96, "hdim_v": 96},
            {"layout": "sparse"},
            {"layout": "paged", "kv_block_size": 48},
            {"mask_type": 99},
            {"mask_type": AttentionMaskType.SLIDING_WINDOW},
            {"sliding_window": 32},
            {"sliding_window": -1},
        )
        for changes in cases:
            fields = {"layout": "dense"}
            fields.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dispatch_attention(_request(**fields))

    def test_explicit_layout_and_score_requirements_are_not_ignored_elsewhere(self):
        for changes in (
            {"layout": "dense"},
            {"layout": "ragged"},
            {"layout": "paged"},
            {"use_softcap": True},
            {"use_alibi": True},
            {"use_qq_bias": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dispatch_attention(_request(arch="gfx950", **changes))

    def test_other_arch_defaults_remain_unchanged(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    dispatch_attention(_request(arch=arch)).candidate.name,
                    "attention_unified_2d",
                )


if __name__ == "__main__":
    unittest.main()
