# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check that the completeness gate fails killed suites and records memory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "rocjitsu_check_suite.py"


def run_check(report_dir, cgroup_root):
    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--report-dir",
            report_dir,
            "--cgroup-root",
            cgroup_root,
        ],
        capture_output=True,
        text=True,
    )


class TestCheckSuite(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.report = Path(temporary.name) / "report"
        self.report.mkdir()
        self.cgroup = Path(temporary.name) / "cgroup"
        self.cgroup.mkdir()

    def write(self, name, value):
        (self.report / name).write_text(json.dumps(value))

    def test_finished_suite_passes(self):
        self.write("progress.json", dict(selected=3, finished=3, incomplete=0))
        self.write("execution.json", dict(exit_code=1, complete=True))
        result = run_check(self.report, self.cgroup)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("::error::", result.stdout)

    def test_killed_suite_fails(self):
        self.write("progress.json", dict(selected=145, finished=58, incomplete=87))
        result = run_check(self.report, self.cgroup)
        self.assertEqual(result.returncode, 1)
        self.assertIn("87 of 145 tests did not finish", result.stdout)
        self.assertIn("execution.json is missing", result.stdout)

    def test_records_memory_and_oom_kills(self):
        self.write("progress.json", dict(selected=1, finished=1, incomplete=0))
        self.write("execution.json", dict(exit_code=0, complete=True))
        (self.cgroup / "memory.max").write_text("68719476736\n")
        (self.cgroup / "memory.events").write_text("oom 3\noom_kill 2\n")
        result = run_check(self.report, self.cgroup)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("recorded 2 OOM kill(s)", result.stdout)
        recorded = (self.report / "container-memory.txt").read_text()
        self.assertIn("memory.max: 68719476736", recorded)
        self.assertIn("memory.events: oom_kill 2", recorded)


if __name__ == "__main__":
    unittest.main()
