#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
rocSOLVER test runner using BaseTestRunner.

This demonstrates the direct GTest binary execution pattern.
"""

import sys
from pathlib import Path

# Import base class from shared location
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "test_runners"))

from base_test_runner import BaseTestRunner


class RocsolverTestRunner(BaseTestRunner):
    """Test runner for rocSOLVER component."""

    def __init__(self):
        super().__init__(component_name="rocsolver")
        self.binary_name = "rocsolver-test"

    def get_quick_test_filters(self):
        """
        Define quick test patterns for rocSOLVER.

        Test filter patterns retrieved from:
        https://github.com/ROCm/rocm-libraries/blob/a18b17eef6c24bcd4bcf8dd6a0e36325cbcd11a7/projects/rocsolver/rtest.xml
        """
        return [
            "checkin*BDSQR*",
            "checkin*STEBZ*",
            "checkin*STEIN*",
            "checkin*STERF*",
            "checkin*STEQR*",
            "checkin*SYEVJ*",
            "checkin*HEEVJ*",
            "checkin*LARFG*",
            "checkin*LARF*",
            "checkin*LARFT*",
            "checkin*GETF2*",
            "checkin*POTF2*",
            "checkin*GEQR2*",
            "checkin*GELQ2*",
            "checkin*SPLITLU*",
            "checkin*REFACTLU*",
            "checkin*REFACTCHOL*",
        ]

    def build_command(self):
        """Build direct GTest binary command with optional filter."""
        cmd = [f"{self.therock_bin_dir}/{self.binary_name}"]

        # Apply quick test filters if in quick mode
        if self.is_quick_test():
            quick_filters = self.get_quick_test_filters()
            if quick_filters:
                # Exclude LARFB and known_bug tests
                filter_str = ":".join(quick_filters) + "-*LARFB*:*known_bug*"
                cmd.append(f"--gtest_filter={filter_str}")

        return cmd


if __name__ == "__main__":
    runner = RocsolverTestRunner()
    sys.exit(runner.run())
