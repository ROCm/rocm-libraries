# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the occupancy primitive's pure logic (no GPU)."""
import unittest
from types import SimpleNamespace

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

    def test_elf_target_overrides_a_stale_arch_flag(self):
        # Real gfx1201 binary, caller passed --arch gfx950: the RDNA occupancy model
        # must win, otherwise the wrong wave count is reported with no signal.
        occupancy.parse_notes = lambda b: {"vgpr": 111, "target": "gfx1201"}
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["target_arch"], "gfx1201")
        self.assertEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx1201")
        )
        self.assertNotEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx950")
        )

    def test_arch_argument_is_the_fallback_when_notes_lack_a_target(self):
        occupancy.parse_notes = lambda b: {"vgpr": 111}
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["target_arch"], "gfx950")
        self.assertEqual(res["occupancy"], occupancy._occupancy_estimate(111, "gfx950"))


class TestParseNotes(unittest.TestCase):
    """Parse a real `llvm-readelf --notes` excerpt (captured from a gfx1201 HSACO)."""

    _NOTES = (
        "Displaying notes found in: .note\n"
        "  amdhsa.kernels:\n"
        "    - .agpr_count:    0\n"
        "      .group_segment_fixed_size: 512\n"
        "      .sgpr_count:    23\n"
        "      .sgpr_spill_count: 0\n"
        "      .vgpr_count:    111\n"
        "      .vgpr_spill_count: 0\n"
        "  amdhsa.target:   amdgcn-amd-amdhsa--gfx1201\n"
        "  amdhsa.version:\n"
    )

    def setUp(self):
        self._orig_readelf = occupancy._readelf
        self._orig_run = occupancy.subprocess.run
        occupancy._readelf = lambda: "llvm-readelf"
        occupancy.subprocess.run = lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=self._NOTES, stderr=""
        )

    def tearDown(self):
        occupancy._readelf = self._orig_readelf
        occupancy.subprocess.run = self._orig_run

    def test_fields_and_target_parsed(self):
        fields = occupancy.parse_notes(b"fake")
        self.assertEqual(fields["vgpr"], 111)
        self.assertEqual(fields["lds_bytes"], 512)
        self.assertEqual(fields["target"], "gfx1201")

    def test_resources_uses_the_parsed_target(self):
        res = occupancy.resources(b"fake", "gfx950")  # deliberately wrong flag
        self.assertEqual(res["target_arch"], "gfx1201")
        self.assertEqual(res["occupancy"], 12)  # RDNA model, not the CDNA 4


class TestOccupancyEstimate(unittest.TestCase):
    def test_estimate_is_capped(self):
        # tiny VGPR -> capped at max_waves_per_simd, not unbounded
        est = occupancy._occupancy_estimate(4, "gfx950")  # maps to cdna caps
        self.assertEqual(est, 8)  # cdna max_waves_per_simd

    def test_zero_vgpr_none(self):
        self.assertIsNone(occupancy._occupancy_estimate(0, "gfx950"))

    def test_gfx90a_uses_eight_vgpr_allocation_granularity(self):
        self.assertEqual(occupancy._occupancy_estimate(65, "gfx90a"), 7)


if __name__ == "__main__":
    unittest.main()
