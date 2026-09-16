# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device functional coverage for the canonical device-print path."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from rocke.runtime.hip_module import get_device_arch, get_device_count


class TestDevicePrintGpu(unittest.TestCase):
    def test_device_print_gpu_functional(self) -> None:
        if get_device_count() < 1:
            self.skipTest("requires a HIP-visible AMD GPU")
        arch = get_device_arch(0)
        if not arch:
            self.skipTest("HIP device architecture is unavailable")

        runner = Path(__file__).with_name("device_print_gpu_runner.py")
        platform_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        source_python = platform_root / "python"
        if (source_python / "rocke").is_dir():
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(source_python), environment.get("PYTHONPATH", "")]
            )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        result = subprocess.run(
            [sys.executable, str(runner), "--arch", arch],
            cwd=platform_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
        )
        context = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        self.assertEqual(result.returncode, 0, context)
        self.assertNotIn("DEVICE_PRINT_FALSE_SENTINEL", result.stdout, context)

        output = re.search(
            r"^DEVICE_PRINT_GPU t f -5 4294967291 6\.5 (0x[0-9a-f]+) 7 8$",
            result.stdout,
            re.MULTILINE,
        )
        expected = re.search(
            r"^DEVICE_PRINT_EXPECTED_PTR=(0x[0-9a-f]+)$",
            result.stderr,
            re.MULTILINE,
        )
        self.assertIsNotNone(output, context)
        self.assertIsNotNone(expected, context)
        assert output is not None
        assert expected is not None
        self.assertEqual(int(output.group(1), 16), int(expected.group(1), 16), context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
