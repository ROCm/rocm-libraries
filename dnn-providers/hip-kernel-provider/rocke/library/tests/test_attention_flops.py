# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Theoretical attention FLOP helper."""

from __future__ import annotations

import unittest

from benchmarks.common.attention_flops import attention_flops, attended_pairs


class TestAttendedPairs(unittest.TestCase):
    def test_full_mask_is_the_rectangle(self):
        self.assertEqual(attended_pairs(1024, 1024, causal=False), 1024 * 1024)

    def test_causal_square_is_the_right_aligned_triangle(self):
        self.assertEqual(attended_pairs(4, 4, causal=True), 10)  # 1+2+3+4

    def test_causal_decode_attends_the_whole_cache(self):
        self.assertEqual(attended_pairs(1, 4096, causal=True), 4096)

    def test_sliding_window_caps_the_lookback(self):
        self.assertEqual(attended_pairs(4, 4, causal=True, sliding_window=1), 4)

    def test_flops_are_four_times_pairs(self):
        pairs = attended_pairs(1024, 1024, causal=True)
        self.assertEqual(
            attention_flops(2, 32, 128, 1024, 1024, causal=True),
            4.0 * 2 * 32 * 128 * pairs,
        )


if __name__ == "__main__":
    unittest.main()
