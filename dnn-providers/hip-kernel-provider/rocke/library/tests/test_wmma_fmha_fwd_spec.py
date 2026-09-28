# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import combinations

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, wmma_fmha_fwd_grid


class TestWmmaFmhaFwdSpec(unittest.TestCase):
    def test_mha_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        self.assertEqual(spec.kv_heads, 4)

    def test_divisible_gqa_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(spec.kv_heads, 2)

    def test_non_divisible_gqa_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, num_kv_heads=3)

    def test_dtype_cache_keys(self):
        fp16 = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="fp16")
        alias = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="f16")
        bf16 = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="bf16")
        self.assertEqual(fp16.kernel_name(), alias.kernel_name())
        self.assertNotEqual(fp16.kernel_name(), bf16.kernel_name())

    def test_unsupported_dtype_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="fp32")

    def test_bottom_right_requires_causal_mask(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(
                head_size=64, num_query_heads=4, causal_bottom_right=True,
            )

    def test_causal_alignment_cache_keys(self):
        top_left = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, mask_mode="causal",
        )
        bottom_right = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, mask_mode="causal",
            causal_bottom_right=True,
        )
        self.assertNotEqual(top_left.kernel_name(), bottom_right.kernel_name())

    def test_partial_query_grid_requires_specialization(self):
        aligned = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        with self.assertRaises(ValueError):
            wmma_fmha_fwd_grid(aligned, seqlen_q=1, batch=2)
        tail = replace(aligned, query_tail=True)
        for length, tiles in ((1, 1), (16, 1), (17, 2)):
            with self.subTest(length=length):
                self.assertEqual(
                    wmma_fmha_fwd_grid(tail, seqlen_q=length, batch=2),
                    (tiles, 4, 2),
                )

    def test_tail_specializations_do_not_alias(self):
        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        variants = [
            replace(base, query_tail=q, kv_tail=k)
            for q, k in ((False, False), (True, False), (False, True), (True, True))
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())


if __name__ == "__main__":
    unittest.main()
