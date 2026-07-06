# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the occupancy primitive's pure logic (no GPU)."""
import unittest

from rocke.benchmark.perf import occupancy


class TestResourcesSourceTag(unittest.TestCase):
    def setUp(self):
        self._orig = occupancy.parse_notes

    def tearDown(self):
        occupancy.parse_notes = self._orig

    def test_source_tagged_elf_notes(self):
        occupancy.parse_notes = lambda b: {"vgpr": 24, "sgpr": 16, "lds_bytes": 2048}
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["source"], "elf_notes")   # distinguishes from rocprofv3
        self.assertEqual(res["vgpr"], 24)
        self.assertIsNotNone(res["occupancy"])

    def test_empty_notes_returns_empty(self):
        occupancy.parse_notes = lambda b: {}
        self.assertEqual(occupancy.resources(b"x", "gfx950"), {})


class TestOccupancyEstimate(unittest.TestCase):
    def test_estimate_is_capped(self):
        # tiny VGPR -> capped at max_waves_per_simd, not unbounded
        est = occupancy._occupancy_estimate(4, "gfx950")   # maps to cdna caps
        self.assertEqual(est, 8)   # cdna max_waves_per_simd

    def test_zero_vgpr_none(self):
        self.assertIsNone(occupancy._occupancy_estimate(0, "gfx950"))


if __name__ == "__main__":
    unittest.main()
