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
        self.assertEqual(res["source"], "elf_notes")  # distinguishes from rocprofv3
        self.assertEqual(res["vgpr"], 24)
        self.assertIsNotNone(res["occupancy"])

    def test_empty_notes_returns_empty(self):
        occupancy.parse_notes = lambda b: {}
        self.assertEqual(occupancy.resources(b"x", "gfx950"), {})


class TestOccupancyEstimate(unittest.TestCase):
    def test_estimate_is_capped(self):
        # tiny VGPR -> capped at max_waves_per_simd, not unbounded
        est = occupancy._occupancy_estimate(4, "gfx950")  # maps to cdna caps
        self.assertEqual(est, 8)  # cdna max_waves_per_simd

    def test_zero_vgpr_none(self):
        self.assertIsNone(occupancy._occupancy_estimate(0, "gfx950"))

    def test_gfx90a_uses_eight_vgpr_allocation_granularity(self):
        self.assertEqual(occupancy._occupancy_estimate(65, "gfx90a"), 7)


class TestEstimateOccupancyDetail(unittest.TestCase):
    """Multi-limiter model: min over VGPR / AGPR / LDS / workgroup / wave cap."""

    def setUp(self):
        self._orig = occupancy.parse_notes

    def tearDown(self):
        occupancy.parse_notes = self._orig

    def _patch(self, **notes):
        occupancy.parse_notes = lambda b: notes

    def test_vgpr_limited(self):
        # 256 VGPR on gfx950 -> 512//256 = 2 waves/SIMD; LDS/AGPR slack.
        self._patch(vgpr=256, agpr=0, lds_bytes=2048, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "VGPR")

    def test_lds_limited(self):
        # low VGPR (8 waves) but a fat 80 KB workgroup -> LDS binds below regs.
        self._patch(vgpr=64, agpr=0, lds_bytes=80000, max_flat_workgroup_size=512)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["limited_by"], "LDS")
        self.assertEqual(det["waves_per_simd"], 4)  # (163840//80000=2 wg) * 8 / 4

    def test_agpr_limited_separate_pool(self):
        # gfx942 (CDNA3) keeps separate VGPR/AGPR pools -> heavy accumulator use
        # starves waves before VGPR does.
        self._patch(vgpr=64, agpr=200, lds_bytes=2048, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx942")
        self.assertEqual(det["limited_by"], "AGPR")

    def test_gfx950_combined_reg_pool(self):
        # gfx950 (CDNA4): VGPR+AGPR share one file, so the SUM sets the wave count.
        # Calibrated vs gfx950 hardware MeanOccupancyPerCU: an AGPR-heavy kernel (44 VGPR +
        # 132 AGPR) measured ~7 waves/CU. 512 // align_up(176,16)=176 -> 2/SIMD
        # (8/CU); a separate 256-AGPR pool would wrongly give 1/SIMD (4/CU).
        self._patch(vgpr=44, agpr=132, lds_bytes=24576, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "VGPR+AGPR")

    def test_waves_per_wg_override_beats_notes(self):
        self._patch(vgpr=64, agpr=0, lds_bytes=80000, max_flat_workgroup_size=64)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950", waves_per_wg=8)
        self.assertEqual(det["limited_by"], "LDS")  # override, not the note's 1 wave

    def test_unknown_arch_returns_empty(self):
        self._patch(vgpr=64, lds_bytes=2048)
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx1201"), {})

    def test_empty_notes_returns_empty(self):
        self._patch()  # {}
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx950"), {})


if __name__ == "__main__":
    unittest.main()
