# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec


class TestWmmaFmhaFwdSpec(unittest.TestCase):
    def test_mha_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        self.assertEqual(spec.kv_heads, 4)

    def test_divisible_gqa_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(spec.kv_heads, 2)

    def test_non_divisible_gqa_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of num_kv_heads"):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, num_kv_heads=3)


if __name__ == "__main__":
    unittest.main()
