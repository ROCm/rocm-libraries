# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the per-outcome test lists against real pytest outcomes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).parents[1]


class TestRocjitsuTestSummary(unittest.TestCase):
    def run_suite(self, source, *extra):
        """Run real pytest through rocjitsu_pytest, then summarize its progress."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_example.py").write_text(source)
            env = {
                **os.environ,
                "PYTHONPATH": str(SCRIPTS),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "GITHUB_STEP_SUMMARY": str(root / "summary.md"),
            }
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "rocjitsu_pytest",
                    "--rocjitsu-report-dir",
                    str(root / "reports"),
                    "-p",
                    "no:cacheprovider",
                    *extra,
                    str(root / "test_example.py"),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "rocjitsu_test_summary.py"),
                    "--report-dir",
                    str(root / "reports"),
                    "--title",
                    "suite",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return (root / "summary.md").read_text()

    def test_lists_each_outcome(self):
        summary = self.run_suite(
            """
import pytest
def test_pass(): pass
def test_fail(): assert False
def test_skip(): pytest.skip('unsupported')
@pytest.mark.xfail(reason='known')
def test_xfail(): assert False
@pytest.mark.xfail(reason='known')
def test_xpass(): pass
@pytest.mark.xfail(reason='known', strict=True)
def test_xpass_strict(): pass
"""
        )
        self.assertIn(
            "### suite: 1 failed, 2 xpassed, 1 xfailed, 1 passed, 1 skipped", summary
        )
        self.assertIn("**Failed (1)**\n\n- test_fail (", summary)
        self.assertIn("- test_xpass (", summary)
        self.assertIn("- test_xpass_strict (", summary)
        self.assertIn("**XFAIL (1)**\n\n- test_xfail (", summary)
        self.assertIn("<details><summary><b>Passed (1)</b></summary>", summary)
        self.assertIn("**Skipped (1)**\n\n- test_skip (", summary)

    def test_counts_skips_instead_of_listing_many(self):
        summary = self.run_suite(
            "import pytest\n"
            "@pytest.mark.parametrize('n', range(11))\n"
            "def test_skip(n): pytest.skip('unsupported')\n"
        )
        self.assertIn("**Skipped:** 11 tests", summary)
        self.assertNotIn("- test_skip", summary)


if __name__ == "__main__":
    unittest.main()
