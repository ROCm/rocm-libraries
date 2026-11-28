# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import select_cpu_build_runner as selector  # noqa: E402


class SelectCpuBuildRunnerTest(unittest.TestCase):
    def test_rocjitsu_uses_the_asan_scale_set(self):
        self.assertEqual(
            selector.select_linux_build_runner(),
            "az-linux-scale-rocm-asan-prod-2",
        )


if __name__ == "__main__":
    unittest.main()
