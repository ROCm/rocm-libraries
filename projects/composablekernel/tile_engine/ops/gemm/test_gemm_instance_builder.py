# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for gemm_instance_builder helpers."""

import os
import sys
import unittest

# Make the instance builder importable by inserting its directory into sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gemm_instance_builder import _cshuffle_store_ok  # noqa: E402


class TestCShuffleStoreGate(unittest.TestCase):
    """CShuffle-store correctness gate (issue #9684).

    Only an ODD per-wave repeat (>1) paired with a 32-wide warp tile in that
    dimension is numerically wrong; every other non-power-of-two repeat is
    correct. Expectations GPU-verified on gfx942 across the tile_m=192 cshuffle
    config space (26 broken / 90 correct).
    """

    def test_broken_signature_rejected(self):
        # tile_m=192 / warp_m=2 / warp_tile_m=32 -> MRepeat = 192/(2*32) = 3.
        self.assertFalse(_cshuffle_store_ok(3, 2, 32, 32))
        self.assertFalse(_cshuffle_store_ok(3, 4, 32, 16))  # M side triggers
        self.assertFalse(_cshuffle_store_ok(4, 3, 16, 32))  # N side triggers

    def test_odd_repeat_with_16_warp_tile_allowed(self):
        # MRepeat=3 via warp_m=4 / warp_tile_m=16 is numerically correct.
        self.assertTrue(_cshuffle_store_ok(3, 2, 16, 16))

    def test_even_nonpow2_repeat_allowed(self):
        # Repeats 6 and 12 are non-power-of-two but verified correct, incl. w/32.
        self.assertTrue(_cshuffle_store_ok(6, 4, 32, 16))
        self.assertTrue(_cshuffle_store_ok(12, 2, 16, 32))

    def test_power_of_two_repeats_allowed(self):
        for rep in (1, 2, 4, 8):
            self.assertTrue(_cshuffle_store_ok(rep, rep, 32, 32))
            self.assertTrue(_cshuffle_store_ok(rep, rep, 16, 16))


if __name__ == "__main__":
    unittest.main()
