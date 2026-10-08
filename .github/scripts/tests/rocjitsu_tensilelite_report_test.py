# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the JUnit summary against pytest outcome semantics."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "rocjitsu_junit_summary.py"


class TestTensileliteReport(unittest.TestCase):
    def test_distinguishes_skips_and_failures_from_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            junit = Path(directory) / "junit"
            junit.mkdir()
            (junit / "tensilelite.xml").write_text(
                "<testsuites><testsuite>"
                '<testcase name="passed" time="1"/>'
                '<testcase name="failed" time="2"><failure/></testcase>'
                '<testcase name="error" time="3"><error/></testcase>'
                '<testcase name="skipped" time="4"><skipped/></testcase>'
                '<testcase name="xfail" time="5"><skipped type="pytest.xfail"/></testcase>'
                "</testsuite></testsuites>"
            )
            result = subprocess.run(
                [sys.executable, SCRIPT, "--report-dir", directory],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("5 tests, 1 passed, 2 failed, 2 skipped", result.stdout)
        self.assertIn("15s aggregate test time", result.stdout)

    def test_missing_report_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, SCRIPT, "--report-dir", directory],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("No JUnit report", result.stdout)


if __name__ == "__main__":
    unittest.main()
