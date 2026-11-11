# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rocjitsu_race_smoke as smoke

DISPATCH = "[rocjitsu] Kernel dispatch: Cijk_example\n"
MFMA = "[rocjitsu] mfma detected in dispatch 3\n"


class WorkloadFailuresTest(unittest.TestCase):
    def test_clean_run_passes(self):
        self.assertEqual(smoke.workload_failures(0, DISPATCH, MFMA), [])

    def test_reported_race_fails(self):
        race_log = DISPATCH + "RACE vgpr read-after-write\nRACE lds write-after-read\n"
        self.assertEqual(
            smoke.workload_failures(0, race_log, MFMA),
            ["race detector reported 2 race(s)"],
        )

    def test_run_without_detector_output_fails(self):
        # A missing race plugin leaves the race sink empty; that must not pass.
        self.assertEqual(
            smoke.workload_failures(0, "", MFMA),
            ["race detector recorded no kernel dispatch"],
        )

    def test_run_without_mfma_is_inconclusive(self):
        self.assertEqual(
            smoke.workload_failures(0, DISPATCH, ""),
            ["no MFMA dispatch reached; the race check is inconclusive"],
        )

    def test_failed_or_timed_out_workload_fails(self):
        self.assertEqual(
            smoke.workload_failures(3, DISPATCH, MFMA), ["exited with status 3"]
        )
        self.assertEqual(smoke.workload_failures(None, DISPATCH, MFMA), ["timed out"])


class SelectTargetTest(unittest.TestCase):
    def test_families_map_to_targets(self):
        self.assertEqual(smoke.select_target("gfx94X-dcgpu")[0], "gfx942")
        self.assertEqual(smoke.select_target("gfx950-dcgpu")[0], "gfx950")

    def test_unsupported_family_is_rejected(self):
        with self.assertRaises(SystemExit):
            smoke.select_target("gfx120X-all")


if __name__ == "__main__":
    unittest.main()
