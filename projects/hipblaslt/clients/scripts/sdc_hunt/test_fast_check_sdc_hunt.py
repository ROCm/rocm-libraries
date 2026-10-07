# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests: python -m unittest discover -s clients/scripts/sdc_hunt."""

import os
import io
import json
import datetime
from unittest import mock
from pathlib import Path
import signal
import shlex
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
                [sys.executable, "-c", parent, child, str(ready)],
                start_new_session=True,
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


class CliTests(unittest.TestCase):
    def setUp(self):
        for sig in (signal.SIGTERM, signal.SIGHUP):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root / "fake-test"
        self.binary.write_text("#!/bin/sh\nprintf '[  PASSED  ] 1 test.\\n'\n")
        self.binary.chmod(0o755)
        self.args = [
            "--test-bin",
            str(self.binary),
            "--results",
            str(self.root / "results.jsonl"),
        ]
        self.environment = mock.patch.object(
            hunt, "environment_record", side_effect=lambda *_: {"cwsr_enable": None}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def invoke(self, args):
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            return hunt.main(args)

    def test_relative_binary_in_current_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            args = self.args.copy()
            args[1] = "./fake-test"
            self.assertEqual(self.invoke(args), 0)
        finally:
            os.chdir(previous)

    def test_empty_command_is_not_an_uncontended_pass(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                self.invoke(self.args + ["--load", "command:"])
        self.assertEqual(raised.exception.code, 2)

    def test_repeated_invocations_keep_distinct_failure_logs(self):
        moment = datetime.datetime.now()
        with mock.patch.object(hunt.datetime, "datetime") as clock:
            clock.now.return_value = moment
            for label in ("first", "second"):
                self.binary.write_text("#!/bin/sh\nprintf '" + label + "\\n'\n")
                self.assertEqual(self.invoke(self.args), 1)
        records = [
            json.loads(line)
            for line in (self.root / "results.jsonl").read_text().splitlines()
        ]
        self.assertNotEqual(records[0]["log"], records[1]["log"])
        self.assertEqual(Path(records[0]["log"]).read_text(), "first\n")
        self.assertEqual(Path(records[1]["log"]).read_text(), "second\n")

    def test_long_load_command_keeps_logs_and_full_metadata(self):
        load = "command:" + shlex.join(
            [sys.executable, "-c", "import time; time.sleep(120) # " + "x" * 300]
        )
        self.binary.write_text("#!/bin/sh\nprintf 'failure evidence\\n'\nexit 1\n")
        with mock.patch.object(
            hunt,
            "environment_record",
            side_effect=lambda _xnack, command: {"cwsr_enable": None, "load": command},
        ):
            self.assertEqual(
                self.invoke(self.args + ["--load", load, "--load-settle-seconds", "0"]),
                1,
            )
        record = json.loads((self.root / "results.jsonl").read_text())
        self.assertEqual(record["load"], load)
        self.assertTrue(record["load_ran_throughout"])
        self.assertEqual(Path(record["log"]).read_text(), "failure evidence\n")
        self.assertTrue(Path(record["load_log"]).is_file())

    def test_empty_and_skipped_runs_fail(self):
        for output, extra in [
            ("", []),
            ("[  PASSED  ] 1 test.\n[  SKIPPED ] 1 test.\n", ["--fail-on-skip"]),
        ]:
            self.binary.write_text(
                "#!/bin/sh\ncat <<'OUTPUT'\n" + output + "\nOUTPUT\n"
            )
            self.assertEqual(self.invoke(self.args + extra), 1)

    def foreground_tree(self, inherit_output=False, exit_early=False):
        ready = self.root / f"child-{time.monotonic_ns()}"
        child = (
            "import os, signal, sys, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)"
        )
        redirect = (
            "" if inherit_output
            else ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        )
        parent = (
            "import subprocess, sys, time\nfrom pathlib import Path\n"
            "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]"
            + redirect
            + ")\n"
            "while not Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
            + (
                "print('[  PASSED  ] 1 test.', flush=True)\n" if exit_early
                else "print('partial output', flush=True)\ntime.sleep(120)\n"
            )
        )
        self.binary.write_text(
            "#!/bin/sh\nexec "
            + shlex.join([sys.executable, "-c", parent, child, str(ready)])
            + "\n"
        )

        def cleanup():
            if ready.exists():
                try:
                    os.kill(int(ready.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.addCleanup(cleanup)
        return ready

    def wait_for_child(self, ready):
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "foreground descendant did not start")
        return int(ready.read_text())

    def assert_child_stopped(self, ready):
        pid = self.wait_for_child(ready)

        def running():
            try:
                return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
            except FileNotFoundError:
                return False

        deadline = time.monotonic() + 2
        while running() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(running(), "foreground descendant survived cleanup")

    def test_timeout_stops_foreground_descendants_and_keeps_partial_output(self):
        for inherit_output in (False, True):
            with self.subTest(inherit_output=inherit_output):
                ready = self.foreground_tree(inherit_output=inherit_output)
                self.assertEqual(self.invoke(self.args + ["--timeout", "1"]), 1)
                record = json.loads(
                    (self.root / "results.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(record["exit_code"], "timeout")
                self.assertIn("partial output", Path(record["log"]).read_text())
                self.assert_child_stopped(ready)

    def test_completed_foreground_stops_descendants(self):
        ready = self.foreground_tree(exit_early=True)
        self.assertEqual(self.invoke(self.args), 0)
        self.assert_child_stopped(ready)

    def test_interrupted_or_failed_foreground_stops_descendants(self):
        for exception in (KeyboardInterrupt, RuntimeError):
            with self.subTest(exception=exception):
                ready = self.foreground_tree()

                def interrupt(_process, *args, **kwargs):
                    self.wait_for_child(ready)
                    raise exception

                with mock.patch.object(hunt.subprocess.Popen, "communicate", interrupt):
                    with self.assertRaises(exception):
                        self.invoke(self.args)
                self.assert_child_stopped(ready)


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
