# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests: python -m unittest discover -s clients/scripts/sdc_hunt."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import fast_check_sdc_hunt as hunt


class ProcessTests(unittest.TestCase):
    def test_stop_kills_descendant_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            child = (
                "import os, signal, sys, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)"
            )
            parent = (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "time.sleep(120)"
            )
            leader = subprocess.Popen(
                [sys.executable, "-c", parent, child, str(ready)], start_new_session=True
            )
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "descendant did not start")
                pid = int(ready.read_text())
                hunt.stop(leader)

                def running():
                    try:
                        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
                    except FileNotFoundError:
                        return False

                deadline = time.monotonic() + 2
                while running() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(running(), "load descendant survived cleanup")
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait(timeout=5)


class ParserTests(unittest.TestCase):
    def test_failure_summary_and_skips(self):
        record = hunt.parse_run(
            "[  PASSED  ] 3 tests.\n"
            "[  SKIPPED ] 2 tests, listed below:\n"
            "[  FAILED  ] suite.bad\n"
            "  solution 3 (library index 42, kernel example): iterations 1, 2\n"
        )
        self.assertEqual(record["tests_passed"], 3)
        self.assertEqual(record["tests_skipped"], 2)
        self.assertEqual(record["tests_failed"], ["suite.bad"])
        self.assertEqual(len(record["failing_solutions"]), 1)
        self.assertTrue(hunt.is_failure(dict(record, exit_code=0)))


if __name__ == "__main__":
    unittest.main()
