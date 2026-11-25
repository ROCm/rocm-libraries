# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sched_group_barrier mask names in helpers/schedule.py match the AMDGPU
backend's SchedGroupMask bits (llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp)."""

import unittest

from rocke.helpers import schedule


class TestScheduleMasks(unittest.TestCase):
    def test_masks_match_the_backend_bits(self):
        expected = {
            "VALU": 0x002,
            "SALU": 0x004,
            "MFMA": 0x008,
            "VMEM": 0x010,
            "VMEM_READ": 0x020,
            "VMEM_WRITE": 0x040,
            "DS": 0x080,
            "DS_READ": 0x100,
            "DS_WRITE": 0x200,
            "TRANS": 0x400,
        }
        for name, bit in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(schedule, name), bit)

    def test_combined_and_split_memory_masks_are_distinct_bits(self):
        # VMEM / DS are their own bits that match every op of the class; they are
        # not the OR of the read and write bits.
        self.assertNotEqual(schedule.VMEM, schedule.VMEM_READ | schedule.VMEM_WRITE)
        self.assertNotEqual(schedule.DS, schedule.DS_READ | schedule.DS_WRITE)


if __name__ == "__main__":
    unittest.main()
