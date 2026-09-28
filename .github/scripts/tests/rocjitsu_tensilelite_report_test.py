# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the shell runner's JUnit summary against pytest outcome semantics."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestTensileliteReport(unittest.TestCase):
    def test_distinguishes_skips_and_failures_from_passes(self):
        script = Path(__file__).parents[1] / "run_rocjitsu_tensilelite_test.sh"
        parser = script.read_text().split("<< 'JUNIT_PARSE'\n", 1)[1].split(
            "\nJUNIT_PARSE", 1
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            junit = Path(directory) / "junit"
            junit.mkdir()
            (junit / "tensilelite.xml").write_text(
                '<testsuites><testsuite>'
                '<testcase name="passed" time="1"/>'
                '<testcase name="failed" time="2"><failure/></testcase>'
                '<testcase name="error" time="3"><error/></testcase>'
                '<testcase name="skipped" time="4"><skipped/></testcase>'
                '<testcase name="xfail" time="5"><skipped type="pytest.xfail"/></testcase>'
                '</testsuite></testsuites>'
            )
            result = subprocess.run(
                [sys.executable, "-c", parser],
                env={**os.environ, "REPORT_DIR": directory},
                check=True, capture_output=True, text=True,
            )
        self.assertIn("5 tests, 1 passed, 2 failed, 2 skipped", result.stdout)
        self.assertIn("15s aggregate test time", result.stdout)


if __name__ == "__main__":
    unittest.main()
