# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GitHub request cap (rkmon_lib/gh_cap.py) and its counter (docker/rk_gh.py), with a fake gh after the counter on
PATH: the Claude session's gh, the tools' (tools/ghr.py) and the writer's (gh.write, as the broker runs it) each count
once in the record and run the real gh with their arguments; a call over the cap waits until the last hour has room,
gh.py's before it starts gh, and says so; the wait is open in the record while it lasts, and a call stopped while it
waited is told as one; a count waits for the record's lock, which another process holds, and a count in a process of its
own leaves no bytecode in the code it loads; a call that pages through results counts once, and the cap key's
description says so; gh's own local commands, a PR runner's calls (no monitor home) and every call while there is no cap
count nothing; a record that cannot be read or written lets the call run uncounted, and the counter and each tick say
so. Each tick gives the counter the configuration's cap, the monitor on or off, and says when gh calls waited for it
while the tick ran; a full tick tells the day log the waits it does not have (rkmon.py).
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import gh, gh_cap  # noqa: E402

# A fake gh: each call's arguments, one line each, in $FAKE_GH_LOG.
FAKE_GH = """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
echo fake-gh
"""
MINUTE = 1000  # the minute the clocks below start in


class Clock:
    """A clock that only a sleep moves."""

    def __init__(self, now: float):
        self.now, self.slept = now, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Base(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.base = pathlib.Path(t.name)
        self.home = self.base / "home"
        self.state = self.home / "state"
        self.state.mkdir(parents=True)
        self.env = {"RKMON_HOME": str(self.home)}

    def record(self) -> dict:
        return json.loads((self.state / gh_cap.RECORD).read_text(encoding="utf-8"))

    def requests(self) -> int:
        return sum(self.record()["requests_by_minute"].values())

    def put_waits(self, *waits, cap=3, open_waits=None):
        """A record with the cap CAP, WAITS, each (at, waited_seconds), and OPEN_WAITS, {key: (since, looked)}."""
        record = {
            "cap_per_hour": cap,
            "requests_by_minute": {},
            "waits": [{"at": at, "waited_seconds": s} for at, s in waits],
            "open_waits": {
                k: {"since": since, "looked": looked}
                for k, (since, looked) in (open_waits or {}).items()
            },
        }
        (self.state / gh_cap.RECORD).write_text(json.dumps(record), encoding="utf-8")

    def count(self, args, clock, say=None):
        gh_cap.count(args, self.env, clock=clock, sleep=clock.sleep, say=say)


class TheCount(Base):
    def test_no_cap_counts_nothing_and_makes_no_file(self):
        self.count(["pr", "view", "1"], Clock(MINUTE * 60))
        self.assertEqual(list(self.state.iterdir()), [])

    def test_each_call_counts_once_and_a_call_that_pages_counts_once(self):
        """Each page of a paginated call is a request to GitHub, but the counter sees one gh call."""
        gh_cap.set_cap(self.state, 10)
        clock = Clock(MINUTE * 60 + 5)
        for args in (
            ["pr", "view", "1"],
            ["api", "--paginate", "repos/o/r/pulls?per_page=100"],
            ["pr", "list", "--limit", "500"],
        ):
            self.count(args, clock)
        self.assertEqual(self.record()["requests_by_minute"], {str(MINUTE): 3})
        self.assertEqual(clock.slept, [])

    def test_the_cap_keys_description_asks_for_headroom_for_a_call_that_pages(self):
        """What the operator reads of the cap (its comment in the settings file, its row in README.md's settings
        reference) says what the test above shows: a gh command that reads several pages counts once, though GitHub
        counts each page.
        """
        from rkmon_lib import deploy_tree

        self.assertIn(
            "A gh command that reads several pages of results counts once toward this cap, though GitHub counts "
            "each page: leave headroom.",
            deploy_tree.description(gh_cap.SETTING),
        )

    def test_a_call_over_the_cap_waits_until_the_last_hour_has_room(self):
        gh_cap.set_cap(self.state, 2)
        clock, said = Clock(MINUTE * 60 + 30), []
        self.count(["pr", "view", "1"], clock)
        self.count(["pr", "view", "2"], clock)
        self.count(["pr", "view", "3"], clock, say=said.append)
        # The minute that holds both requests leaves the last hour an hour after it began.
        self.assertEqual(clock.now, (MINUTE + 60) * 60)
        self.assertEqual(sum(clock.slept), 3570)
        self.assertLessEqual(max(clock.slept), gh_cap.LOOK_SECONDS)
        self.assertEqual(
            said,
            [
                "waiting until 17:40:00Z for the GitHub request cap "
                "(integrations.github_cli.max_requests_per_hour)"
            ],
        )
        record = self.record()
        self.assertEqual(record["requests_by_minute"], {str(MINUTE + 60): 1})
        self.assertEqual(
            record["waits"], [{"at": (MINUTE + 60) * 60, "waited_seconds": 3570.0}]
        )

    def test_a_lower_cap_waits_for_as_many_minutes_as_it_takes(self):
        gh_cap.set_cap(self.state, 10)
        clock = Clock(MINUTE * 60)
        for _ in range(2):
            self.count(["pr", "view", "1"], clock)
        clock.now += 10 * 60
        for _ in range(2):
            self.count(["pr", "view", "1"], clock)
        gh_cap.set_cap(self.state, 2, clock=clock)
        self.count(["pr", "view", "1"], clock)
        # Three of the four requests must leave before one more fits under 2: the first minute's two, and one of the
        # second minute's, whose two leave together.
        self.assertEqual(clock.now, (MINUTE + 10 + 60) * 60)
        self.assertEqual(self.record()["requests_by_minute"], {str(MINUTE + 70): 1})

    def test_waiting_for_room_counts_nothing(self):
        """gh.py's step waits as the counter does, and leaves the count to the counter."""
        gh_cap.set_cap(self.state, 1)
        clock = Clock(MINUTE * 60)
        self.count(["pr", "view", "1"], clock)
        waited = gh_cap.wait_for_room(
            ["pr", "view", "2"], self.env, clock=clock, sleep=clock.sleep, say=None
        )
        self.assertEqual(waited, 3600)
        self.assertEqual(self.requests(), 0)  # the first minute left the last hour
        self.assertEqual(len(self.record()["waits"]), 1)

    def test_waiting_for_room_says_so_on_standard_error_as_the_counter_does(self):
        """gh.py's wait says the line the counter says, which the tick filter knows."""
        from rkmon_lib import tickfilter

        gh_cap.set_cap(self.state, 1)
        clock, err = Clock(MINUTE * 60), io.StringIO()
        self.count(["pr", "view", "1"], clock)
        with contextlib.redirect_stderr(err):
            gh_cap.wait_for_room(
                ["pr", "view", "2"], self.env, clock=clock, sleep=clock.sleep
            )
        said = (
            "gh: waiting until 17:40:00Z for the GitHub request cap "
            "(integrations.github_cli.max_requests_per_hour)"
        )
        self.assertEqual(err.getvalue(), f"{said}\n")
        self.assertEqual(tickfilter.classify(said)[0], tickfilter.ROUTINE)

    def test_a_wait_is_open_in_the_record_while_it_lasts(self):
        """So that a tick's line and the tick loop's stop line can name the cap while a call still waits."""
        gh_cap.set_cap(self.state, 1)
        clock, seen = Clock(MINUTE * 60), []
        self.count(["pr", "view", "1"], clock)

        def sleep(seconds):
            if not seen:  # the first look found no room
                seen.append(
                    (
                        self.record()["open_waits"],
                        gh_cap.waits(self.state, 0, clock=clock),
                        gh_cap.waiting_text(self.state, clock()),
                    )
                )
            clock.sleep(seconds)

        gh_cap.wait_for_room(
            ["pr", "view", "2"], self.env, clock=clock, sleep=sleep, say=None
        )
        [(open_waits, waits, text)] = seen
        self.assertEqual(
            list(open_waits.values()), [{"since": MINUTE * 60, "looked": MINUTE * 60}]
        )
        self.assertEqual(
            waits, [{"at": MINUTE * 60, "waited_seconds": 0.0, "waiting": True}]
        )
        self.assertEqual(
            text,
            "while 1 gh call waited for the GitHub request cap "
            "(integrations.github_cli.max_requests_per_hour), the first since 16:40Z",
        )
        record = self.record()
        self.assertEqual(record["open_waits"], {})
        self.assertEqual(
            record["waits"], [{"at": (MINUTE + 60) * 60, "waited_seconds": 3600.0}]
        )
        self.assertEqual(gh_cap.waiting_text(self.state, clock()), "")

    def test_an_interrupted_wait_ends_as_the_wait_of_a_stopped_call(self):
        gh_cap.set_cap(self.state, 1)
        clock = Clock(MINUTE * 60)
        self.count(["pr", "view", "1"], clock)

        def interrupt(seconds):
            clock.sleep(seconds)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            gh_cap.wait_for_room(
                ["pr", "view", "2"], self.env, clock=clock, sleep=interrupt, say=None
            )
        record = self.record()
        self.assertEqual(record["open_waits"], {})
        self.assertEqual(
            record["waits"],
            [{"at": MINUTE * 60 + 60, "waited_seconds": 60.0, "stopped": True}],
        )

    def test_local_commands_and_a_pr_runner_count_nothing(self):
        gh_cap.set_cap(self.state, 10)
        clock = Clock(MINUTE * 60)
        for args in (
            [],
            ["--version"],
            ["version"],
            ["help", "pr"],
            ["pr", "view", "1", "--help"],
            ["completion", "-s", "bash"],
            ["config", "get", "editor"],
            ["alias", "list"],
            ["auth", "token"],
            ["auth", "setup-git"],
            ["auth", "git-credential", "get"],
        ):
            with self.subTest(args=args):
                self.assertFalse(gh_cap.asks_github(args))
                self.count(args, clock)
        gh_cap.count(["pr", "view", "1"], {}, clock=clock, sleep=clock.sleep)
        self.assertEqual(self.requests(), 0)
        for args in (["auth", "status"], ["api", "user"], ["search", "prs", "x"]):
            with self.subTest(args=args):
                self.assertTrue(gh_cap.asks_github(args))

    def test_a_record_that_cannot_be_read_lets_the_call_run_uncounted_and_says_so(
        self,
    ):
        (self.state / gh_cap.RECORD).mkdir()
        said = []
        self.count(["pr", "view", "1"], Clock(MINUTE * 60), say=said.append)
        self.assertEqual(
            said,
            [
                "the GitHub request cap (integrations.github_cli.max_requests_per_hour) counts no gh call: its record "
                f"{self.state / gh_cap.RECORD} cannot be read or written (Is a directory). A restart of the "
                "container gives the monitor's state folder its permissions again"
            ],
        )
        self.assertEqual(
            gh_cap.wait_for_room(["pr", "view", "1"], self.env, clock=Clock(0)), 0.0
        )

    @unittest.skipIf(os.geteuid() == 0, "root writes in a folder that is read-only")
    def test_a_record_that_reads_but_takes_no_writes_is_said_by_the_counter_and_the_tick(
        self,
    ):
        gh_cap.set_cap(self.state, 2)
        self.state.chmod(0o555)
        self.addCleanup(self.state.chmod, 0o755)
        said = []
        for n in range(3):
            self.count(["pr", "view", str(n)], Clock(MINUTE * 60), say=said.append)
        self.assertEqual(len(said), 3)
        self.assertIn("cannot be read or written (Permission denied)", said[0])
        self.assertEqual(gh_cap.set_cap(self.state, 2), [f"GH_CAP_OFF: {said[0]}"])

    def test_a_file_that_is_no_record_reads_as_one_with_no_cap(self):
        for text in ("[]", "{", '{"cap_per_hour": "10"}', "\udcff"):
            with self.subTest(text=text):
                (self.state / gh_cap.RECORD).write_text(
                    text, encoding="utf-8", errors="surrogateescape"
                )
                self.count(["pr", "view", "1"], Clock(MINUTE * 60))
                self.assertEqual(
                    (self.state / gh_cap.RECORD).read_text(
                        encoding="utf-8", errors="surrogateescape"
                    ),
                    text,
                )

    def test_the_cap_is_written_at_each_tick_while_there_is_one(self):
        """So that each tick finds a record it cannot write; with no cap, and none in the record, nothing is written."""
        self.assertEqual(gh_cap.set_cap(self.state, 0), [])
        self.assertEqual(list(self.state.iterdir()), [])
        with mock.patch.object(
            gh_cap.whole_file, "write", wraps=gh_cap.whole_file.write
        ) as write:
            gh_cap.set_cap(self.state, 5)
            gh_cap.set_cap(self.state, 5)
            self.assertEqual(self.record()["cap_per_hour"], 5)
            gh_cap.set_cap(self.state, 0)  # the cap goes: written once
            gh_cap.set_cap(self.state, 0)
        self.assertEqual(write.call_count, 3)
        self.assertEqual(self.record()["cap_per_hour"], 0)
        self.assertEqual(
            oct((self.state / gh_cap.RECORD).stat().st_mode & 0o777), oct(0o660)
        )
        [line] = gh_cap.set_cap(self.base / "none", 5)
        self.assertTrue(
            line.startswith(
                "GH_CAP_OFF: the GitHub request cap (integrations.github_cli.max_requests_per_hour) counts no gh "
                f"call: its record {self.base / 'none' / gh_cap.RECORD} cannot be read or written (No such file"
            ),
            line,
        )


def waits_for_a_lock(pid: int) -> bool:
    """Whether the process PID waits for a lock (flock): a "->" line of its own in /proc/locks."""
    lines = pathlib.Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    return any(f[1:2] == ["->"] and str(pid) in f for f in map(str.split, lines))


@unittest.skipUnless(os.path.exists("/proc/locks"), "the kernel's list of locks")
class TheLock(Base):
    """The read, the room check, the count and the write all happen under LOCK (perms.shared_lock), which the monitor
    and the writer share: two processes never count over each other."""

    # A count in a process of its own: the package's code from the folder its first argument names, the monitor home
    # its second names. -I drops the suite's PYTHONDONTWRITEBYTECODE with the environment's other PYTHON names, so -B
    # keeps the bytecode of the code it loads (rkmon_lib, and worker/perms.py for the lock) out of that folder.
    COUNT = (
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import gh_cap; "
        "gh_cap.count(['pr', 'view', '1'], {'RKMON_HOME': sys.argv[2]}, say=None)",
    )

    def test_the_counting_process_leaves_no_bytecode_in_the_code_it_loads(self):
        """Run on a copy of the package's code, the count leaves no __pycache__ there: the test below runs it on the
        package's folder, where the suite fails on any entry a test adds."""
        gh_cap.set_cap(self.state, 10)
        code = self.base / "code"
        for part in ("rkmon_lib", "worker"):
            shutil.copytree(
                ROOT / part, code / part, ignore=shutil.ignore_patterns("__pycache__")
            )
        p = subprocess.run(
            [*self.COUNT, str(code), str(self.home)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual((p.returncode, p.stderr), (0, ""))
        self.assertEqual(self.requests(), 1)
        self.assertEqual(sorted(code.rglob("__pycache__")), [])

    def test_a_count_writes_nothing_while_another_process_holds_the_lock(self):
        gh_cap.set_cap(self.state, 10)
        before = self.record()
        with gh_cap._perms().shared_lock(self.state / gh_cap.LOCK):
            child = subprocess.Popen([*self.COUNT, str(ROOT), str(self.home)])
            self.addCleanup(child.kill)
            deadline = time.monotonic() + 30
            while not waits_for_a_lock(child.pid):
                self.assertIsNone(child.poll(), "the count ended without the lock")
                self.assertLess(time.monotonic(), deadline, "the count never waited")
                time.sleep(0.02)
            self.assertEqual(self.record(), before)  # nothing written while it waits
        self.assertEqual(child.wait(timeout=30), 0)
        self.assertEqual(self.requests(), 1)


class TheCounter(Base):
    """docker/rk_gh.py as the first gh on PATH, with a fake gh after it."""

    def setUp(self):
        super().setUp()
        counter, real = self.base / "counter", self.base / "real"
        counter.mkdir()
        real.mkdir()
        os.symlink(ROOT / "docker" / "rk_gh.py", counter / "gh")
        (real / "gh").write_text(FAKE_GH, encoding="utf-8")
        (real / "gh").chmod(0o755)
        self.log = self.base / "gh.log"
        self.path = os.pathsep.join((str(counter), str(real), "/usr/bin", "/bin"))
        self.run_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("RKMON_GH", "GH_CONFIG_DIR", gh.BROKERED)
        }
        self.run_env.update(
            PATH=self.path, RKMON_HOME=str(self.home), FAKE_GH_LOG=str(self.log)
        )
        gh_cap.set_cap(self.state, 10)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def run_line(self, line: str, env=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", "-c", line],
            env=self.run_env if env is None else env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_the_sessions_gh_counts_and_runs_the_real_one(self):
        p = self.run_line("gh pr view 7 --json title")
        self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "fake-gh\n", ""))
        self.assertEqual(self.calls(), ["pr view 7 --json title"])
        self.assertEqual(self.requests(), 1)

    def test_the_tools_gh_counts_once(self):
        p = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ghr.py"), "pr", "view", "7"],
            env=self.run_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual((p.returncode, p.stdout), (0, "fake-gh\n"), p.stderr)
        self.assertEqual(self.calls(), ["pr view 7"])
        self.assertEqual(self.requests(), 1)

    def test_the_writers_gh_counts(self):
        """The broker's rk writes with gh on its PATH (rkmon_lib/writer_run.py BASE_ENV): the counter's."""
        env = dict(self.run_env, **{gh.BROKERED: "1"})
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.multiple(
            gh, BIN="gh", CONFIGURED=True, WRITES_OFF="", UNSPLIT="", WRITE_DIR=""
        ):
            p = gh.write(["api", "-X", "POST", "repos/o/r/issues/1/comments"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.calls(), ["api -X POST repos/o/r/issues/1/comments"])
        self.assertEqual(self.requests(), 1)

    def test_a_pr_runner_and_a_local_command_are_not_counted(self):
        runner = {k: v for k, v in self.run_env.items() if k != "RKMON_HOME"}
        self.assertEqual(self.run_line("gh pr view 7", env=runner).returncode, 0)
        self.assertEqual(self.run_line("gh --version").returncode, 0)
        self.assertEqual(self.calls(), ["pr view 7", "--version"])
        self.assertEqual(self.requests(), 0)

    def test_with_no_gh_after_it_the_counter_says_so(self):
        env = dict(self.run_env, PATH=self.path.split(os.pathsep)[0])
        p = subprocess.run(
            [str(self.base / "counter" / "gh"), "pr", "view", "7"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        from rkmon_lib import container_layout

        self.assertEqual(p.returncode, 127)
        self.assertEqual(
            p.stderr,
            "gh: the image's gh is missing (no gh after the GitHub request counter on PATH): "
            f"{container_layout.GH_MISSING_FIX}\n",
        )
        self.assertFalse(self.log.exists())

    def test_a_relative_folder_on_path_is_passed_over(self):
        """A gh in the folder a call runs in (a pull request's tree) never runs in the real one's place."""
        planted = self.base / "tree"
        planted.mkdir()
        shutil.copy(self.base / "real" / "gh", planted / "gh")
        (planted / "gh").write_text("#!/bin/sh\necho planted\n", encoding="utf-8")
        counter, real, *rest = self.path.split(os.pathsep)
        env = dict(self.run_env, PATH=os.pathsep.join((counter, ".", real, *rest)))
        p = subprocess.run(
            ["sh", "-c", "gh pr view 7"],
            cwd=planted,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(p.stdout, "fake-gh\n", p.stderr)


class GhPyWaitsFirst(Base):
    """gh.py waits for room before it starts gh, so its time limit and its retries run on gh's own time."""

    def test_a_read_and_a_write_wait_before_gh_starts(self):
        fake = self.base / "gh"
        fake.write_text(FAKE_GH, encoding="utf-8")
        fake.chmod(0o755)
        log = self.base / "gh.log"
        gh_cap.set_cap(self.state, 1)
        clock = Clock(MINUTE * 60)
        self.count(["pr", "view", "1"], clock)
        real = gh_cap.wait_for_room
        events = []

        def waits(args):
            events.append(("wait", args[:2]))
            return real(args, clock=clock, sleep=clock.sleep, say=None)

        env = dict(self.env, FAKE_GH_LOG=str(log))
        with mock.patch.dict(os.environ, env), mock.patch.object(
            gh_cap, "wait_for_room", waits
        ), mock.patch.multiple(
            gh, BIN=str(fake), CONFIGURED=True, WRITES_OFF="", UNSPLIT="", WRITE_DIR=""
        ):
            gh.run_read(["pr", "view", "2"])
            self.assertEqual(clock.slept and sum(clock.slept), 3600)
            gh.write(["api", "-X", "POST", "repos/o/r/issues/1/comments"])
            with mock.patch.object(
                gh.os, "execvpe", lambda *a: events.append(("exec", a[1][1:3]))
            ):
                gh.exec_read(["pr", "diff", "3"])
        self.assertEqual(
            events,
            [
                ("wait", ["pr", "view"]),
                ("wait", ["api", "-X"]),
                ("wait", ["pr", "diff"]),
                ("exec", ["pr", "diff"]),
            ],
        )
        self.assertEqual(
            log.read_text(encoding="utf-8").splitlines(),
            ["pr view 2", "api -X POST repos/o/r/issues/1/comments"],
        )


class TheWaits(Base):
    """What the record tells of the calls that waited: a tick's line (waits) and the day log's entries (report)."""

    def test_with_no_record_nothing_waited_and_no_file_is_made(self):
        self.assertEqual(gh_cap.waits(self.state, 0), [])
        self.assertEqual(gh_cap.report(self.state, time.time()), [])
        self.assertEqual(list(self.state.iterdir()), [])

    def test_the_day_log_has_each_wait_once(self):
        now = time.time()
        self.put_waits((now - 600, 30.0), (now - 60, 60.0))
        self.assertEqual(
            gh_cap.waits(self.state, now - 300),
            [{"at": now - 60, "waited_seconds": 60.0}],
        )
        self.assertEqual(len(gh_cap.report(self.state, now - 300)), 1)
        self.assertEqual(len(gh_cap.report(self.state, now)), 1)
        self.assertEqual(gh_cap.report(self.state, now), [])
        self.assertEqual(self.record()["reported_until"], now)
        # A tick's line reads them all the same.
        self.assertEqual(len(gh_cap.waits(self.state, 0)), 2)

    def test_a_call_stopped_while_it_waited_is_told_once(self):
        """A tick stopped at tick_timeout_min leaves its call's wait open: once the call has not looked at the record
        for STOPPED_SECONDS, the record takes it as the wait of a stopped call, which the day log tells once.
        """
        now = time.time()
        looked = now - gh_cap.STOPPED_SECONDS - 1
        self.put_waits(open_waits={"k": (looked - 1200, looked)})
        self.assertEqual(gh_cap.waits(self.state, 0), [])  # it waits no longer
        self.assertEqual(gh_cap.waiting_text(self.state, now), "")
        told = gh_cap.report(self.state, now)
        self.assertEqual(told, [{"at": now, "waited_seconds": 1200.0, "stopped": True}])
        self.assertEqual(gh_cap.report(self.state, now + 1), [])
        self.assertEqual(self.record()["open_waits"], {})
        self.assertEqual(
            gh_cap.wait_text(told, 3),
            "1 gh call waited 20 minutes in all, stopped while waiting, for the cap of 3 GitHub requests an hour "
            "(integrations.github_cli.max_requests_per_hour)",
        )

    def test_the_text_says_how_many_waited_how_long_and_for_which_cap(self):
        for waited, text in (
            ([30.0], "1 gh call waited 30 seconds in all"),
            ([600.0, 120.0], "2 gh calls waited 12 minutes in all"),
            (
                [30.0, (30.0, "waiting"), (60.0, "stopped")],
                "3 gh calls waited 2 minutes in all, 1 of them still waiting and 1 of them stopped while waiting,",
            ),
        ):
            with self.subTest(waited=waited):
                self.assertEqual(
                    gh_cap.wait_text(
                        [
                            (
                                {"at": 0, "waited_seconds": s[0], s[1]: True}
                                if isinstance(s, tuple)
                                else {"at": 0, "waited_seconds": s}
                            )
                            for s in waited
                        ],
                        100,
                    ),
                    f"{text} for the cap of 100 GitHub requests an hour "
                    "(integrations.github_cli.max_requests_per_hour)",
                )


class TheTick(Base):
    """`rk tick` (rkmon.py), with the monitor paused: it gives the counter the configuration's cap all the same."""

    def setUp(self):
        super().setUp()
        import rkmon
        from rkmon_lib import config
        from rkmon_lib.store import Store, empty_state

        self.rkmon = rkmon
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        c["paths"] = dict(c["paths"], working_dir=str(self.base / "work"))
        c["gh"] = dict(c["gh"], max_requests_per_hour=3)
        self.cfg = config.normalize(c)
        self.store = Store(self.state)
        st = empty_state()
        st["enabled"] = False  # paused: the tick does nothing else
        self.store.save(st)

    def tick(self, during=lambda: None) -> list[str]:
        """The lines `rk tick` printed; DURING runs inside the tick, as its gh calls would."""
        rkmon, printed = self.rkmon, []
        real = rkmon.run_tick

        def run(*args):
            during()
            return real(*args)

        env = dict(self.env, XDG_CONFIG_HOME=str(self.base / "xdg"))
        with mock.patch.dict(os.environ, env), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            rkmon, "run_tick", run
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["tick"]), 0)
        return printed

    def test_a_tick_gives_the_counter_the_cap_the_monitor_on_or_off(self):
        self.assertIn("monitor OFF", self.tick()[-1])
        self.assertEqual(self.record()["cap_per_hour"], 3)
        self.cfg["gh"]["max_requests_per_hour"] = 0
        self.tick()
        self.assertEqual(self.record()["cap_per_hour"], 0)

    def test_a_tick_says_when_the_counter_cannot_use_the_caps_record(self):
        from rkmon_lib import tickfilter

        (self.state / gh_cap.RECORD).mkdir()
        printed = self.tick()
        self.assertTrue(
            printed[0].startswith(
                "GH_CAP_OFF: the GitHub request cap (integrations.github_cli.max_requests_per_hour) counts no gh "
                "call: its record "
            ),
            printed,
        )
        self.assertEqual(tickfilter.classify(printed[0])[0], tickfilter.STANDING)
        self.assertIn("monitor OFF", printed[-1])

    def test_a_tick_says_when_gh_calls_waited_while_it_ran(self):
        self.put_waits((time.time() - 5, 30.0))  # before the tick: not the tick's
        self.assertIn("monitor OFF", self.tick()[-1])
        clock = Clock((int(time.time() // 60) + 1) * 60)

        def calls():  # four in a minute under a cap of 3: the fourth waits an hour
            for n in range(4):
                gh_cap.count(
                    ["pr", "view", str(n)],
                    self.env,
                    clock=clock,
                    sleep=clock.sleep,
                    say=None,
                )

        printed = self.tick(calls)
        self.assertIn("monitor OFF", printed[-2])
        self.assertEqual(
            printed[-1],
            "GH_CAP_WAIT while this tick ran, 1 gh call waited 60 minutes in all for the cap of 3 GitHub requests an "
            "hour (integrations.github_cli.max_requests_per_hour)",
        )

    def test_a_full_tick_tells_the_day_log_the_waits_it_does_not_have(self):
        from rkmon_lib import daylog

        now = time.time()
        self.put_waits((now - 600, 30.0), (now - 60, 60.0))
        with mock.patch.dict(os.environ, self.env):
            for _ in range(2):  # the second has nothing to tell
                self.rkmon.gh_cap_log(self.cfg, self.store)
        first, last = (
            time.strftime("%H:%MZ", time.gmtime(at)) for at in (now - 600, now - 60)
        )
        [entry] = daylog.entries(self.store)
        self.assertEqual((entry["kind"], entry["pr"]), ("wait", None))
        self.assertEqual(
            entry["text"],
            "2 gh calls waited 90 seconds in all for the cap of 3 GitHub requests an hour "
            f"(integrations.github_cli.max_requests_per_hour), between {first} and {last}",
        )


if __name__ == "__main__":
    unittest.main()
