# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the supervisor with actual pytest workers and child processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "linux", "Linux CI process supervision")
class TestRocjitsuPytest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reports = self.root / "reports"
        self.scripts = Path(__file__).parents[1]

    def run_pytest(self, source, *, timeout=20, workers=0, extra=()):
        test = self.root / "test_example.py"
        test.write_text(source)
        self.reports.mkdir(exist_ok=True)
        junit = self.reports / "junit/tensilelite.xml"
        command = [
            sys.executable,
            str(self.scripts / "rocjitsu_pytest.py"),
            "--report-dir",
            str(self.reports),
            "--timeout",
            str(timeout),
            "--grace",
            "0.3",
            "--",
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "rocjitsu_pytest",
            "--rocjitsu-report-dir",
            str(self.reports),
            "--junit-xml",
            str(junit),
            "-p",
            "no:cacheprovider",
            "-q",
            *extra,
            str(test),
        ]
        if workers:
            command += ["-p", "xdist.plugin", "-n", str(workers)]
        env = {
            **os.environ,
            "PYTHONPATH": str(self.scripts),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=30
        )
        execution = json.loads((self.reports / "execution.json").read_text())
        progress = json.loads((self.reports / "progress.json").read_text())
        return result, execution, progress

    def test_xdist_reports_failures_skips_and_teardown_errors(self):
        result, execution, progress = self.run_pytest(
            """
import pytest
def test_pass(): pass
def test_fail(): assert False
def test_skip(): pytest.skip('unsupported')
@pytest.mark.xfail(reason='known issue')
def test_xfail(): assert False
@pytest.fixture
def cleanup():
    yield
    pytest.fail('teardown failed')
def test_teardown(cleanup): pass
""",
            workers=2,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertTrue(execution["complete"])
        self.assertEqual(progress["selected"], 5)
        self.assertEqual(progress["finished"], 5)
        self.assertEqual(progress["passed"], 1)
        self.assertEqual(progress["failed"], 2)
        self.assertEqual(progress["skipped"], 2)

    def test_reaps_detached_client_after_success(self):
        result, execution, progress = self.run_pytest(
            """
import subprocess, sys
from pathlib import Path
def test_client():
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    Path(__file__).with_suffix('.pid').write_text(str(child.pid))
"""
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(execution["complete"])
        self.assertEqual(progress["passed"], 1)
        pid = int((self.root / "test_example.pid").read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists(), "orphan was not reaped")

    def test_timeout_preserves_partial_results_and_reaps_clients(self):
        self.reports.mkdir()
        (self.reports / "progress.json").write_text('{"passed": 999}')
        result, execution, progress = self.run_pytest(
            """
import signal, subprocess, sys, time
from pathlib import Path
def test_pass(): pass
def test_hang():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    Path(__file__).with_suffix('.pid').write_text(str(child.pid))
    time.sleep(60)
""",
            timeout=2,
        )
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertTrue(execution["timed_out"])
        self.assertFalse(execution["complete"])
        self.assertEqual(progress["selected"], 2)
        self.assertEqual(progress["passed"], 1)
        self.assertEqual(progress["incomplete"], 1)
        pid = int((self.root / "test_example.pid").read_text())
        self.assertFalse(
            Path(f"/proc/{pid}").exists(), "timed-out client was not reaped"
        )

    def test_durations_pair_longest_with_shortest(self):
        durations = self.root / "durations.json"
        durations.write_text(
            json.dumps({"test_run[a]": 1, "test_run[b]": 3, "test_run[c]": 5})
        )
        result, execution, progress = self.run_pytest(
            """
import pytest
from pathlib import Path
@pytest.mark.parametrize('name', 'abcdz')
def test_run(name):
    with Path(__file__).with_suffix('.order').open('a') as order:
        order.write(name)
""",
            extra=["--rocjitsu-durations", str(durations)],
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(progress["passed"], 5)
        # Unrecorded d and z tie with the longest, c; ties keep node id order.
        self.assertEqual((self.root / "test_example.order").read_text(), "cadbz")

    def test_all_skipped_is_not_success(self):
        result, execution, progress = self.run_pytest(
            "import pytest\ndef test_skip(): pytest.skip('unsupported')\n"
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(progress["passed"], 0)
        self.assertEqual(progress["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
