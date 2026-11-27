# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stopping the monitor: pause (rk pause), stop (rk stop, --drain) and kill (rk kill), each its own level, and rk
status saying which one holds. With real stand-in processes: the monitor's own (a registered driver, a job and its
child) go, an unrelated process and a process that merely reuses a registered pid are never touched, and a GitHub write
in flight finishes before anything is killed. rk start runs nothing: the tests pass their own `run`, and every path is
in a temporary folder.
"""

import contextlib
import io
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (
    config,
    control,
    drain,
    jobs,
    post,
    procs,
    program,
    review,
)  # noqa: E402
from rkmon_lib import store as store_mod  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

OLD = "2000-01-01T00:00:00Z"
SLEEPER = [sys.executable, "-c", "import time; time.sleep(300)"]


def gone(pid: int) -> bool:
    s = procs.stat(pid)
    return s is None or s["state"] == "Z"


class FakeRun:
    """The commands rk start runs (the admin dashboard's server) as the tests see them: recorded, never run."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")


@contextlib.contextmanager
def nothing_runs():
    """No command runs and no process starts while entered: a stop or a kill signals the monitor's own processes and
    runs nothing, no service manager among them (the container's restart policy brings the monitor back).
    """
    with mock.patch.object(
        subprocess, "run", side_effect=AssertionError("a command ran")
    ), mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("a process started")
    ):
        yield


class _Monitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        c["paths"] = dict(c["paths"], working_dir=str(self.base / "work"))
        self.cfg = config.normalize(c)
        self.run_dir = config.paths.get(self.cfg, "run")
        self.store = Store(self.base / "state")
        st = empty_state()
        st["enabled"] = True
        self.store.save(st)
        self.env = mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": str(self.base / "xdg")}
        )
        self.env.start()
        self.children = []

    def tearDown(self):
        for p in self.children:
            if p.poll() is None:
                p.kill()
            p.wait()
        for meta in procs.job_dirs(
            self.store.dir
        ).values():  # a job's leader is this process's child
            try:
                os.killpg(meta["pid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError, KeyError):
                pass
            try:
                os.waitpid(meta["pid"], 0)
            except (ChildProcessError, KeyError):
                pass
        self.env.stop()
        self.tmp.cleanup()

    def spawn(self, register: str | None = None) -> subprocess.Popen:
        p = subprocess.Popen(SLEEPER)
        self.children.append(p)
        for _ in range(50):  # until /proc shows it
            if procs.start_time(p.pid) is not None:
                break
            time.sleep(0.02)
        if register:
            procs.register(self.run_dir, register, pid=p.pid)
        return p

    def a_job(self) -> dict:
        job = jobs.start(
            self.store, "worker", "pr1", "sleep 300 & wait", timeout_min=60
        )
        for _ in range(100):  # until its child, the sleep, runs
            if any(s["ppid"] == job["pid"] for s in procs.table().values()):
                break
            time.sleep(0.02)
        return job


class TestKill(_Monitor):
    def test_kill_takes_the_monitors_processes_and_never_another(self):
        unrelated = self.spawn()
        driver = self.spawn(register="driver")
        job = self.a_job()
        impostor = self.spawn()
        # A registry entry for the impostor's pid with another start time: a pid the system handed out again.
        (pathlib.Path(self.run_dir) / "procs" / f"{impostor.pid}.json").write_text(
            json.dumps(
                {
                    "pid": impostor.pid,
                    "start": procs.start_time(impostor.pid) - 1,
                    "kind": "driver",
                }
            ),
            encoding="utf-8",
        )
        st = self.store.load()
        st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "first-round",
                "head": "1" * 40,
                "worker": job,
                "checks": None,
            }
        ]
        st["jobs"]["heavy"] = job
        self.store.save(st)
        out = []
        with nothing_runs():
            self.assertEqual(
                control.kill_all(self.cfg, self.store, wait=1, out=out.append), 0
            )
        driver.wait(timeout=10)
        time.sleep(0.2)
        self.assertTrue(gone(job["pid"]), "the job's leader")
        self.assertEqual(jobs.status(self.store, job), "lost")
        self.assertIsNone(unrelated.poll(), "an unrelated process was touched")
        self.assertIsNone(impostor.poll(), "a reused pid was touched")
        text = "\n".join(out)
        self.assertIn(f"KILLED pid {driver.pid} (review driver", text)
        self.assertIn(f"KILLED pid {job['pid']} (job: {job['id']}", text)
        self.assertNotIn(f"pid {unrelated.pid} ", text)
        self.assertNotIn(f"pid {impostor.pid} ", text)
        self.assertIn("Nothing else was touched", text)
        st = self.store.load()
        self.assertEqual(store_mod.stopped(st)["how"], "kill")
        self.assertEqual((review.pipes(st), st["jobs"]["heavy"]), ([], None))

    def test_kill_never_cuts_a_write_in_half(self):
        marker = self.base / "written"
        writer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib, sys, time; sys.path.insert(0, %r)\n"
                    "from rkmon_lib import control\n"
                    "with control.write_lock(pathlib.Path(%r)):\n"
                    "    print('holding', flush=True); time.sleep(1.5)\n"
                    "    pathlib.Path(%r).write_text('the whole write')\n"
                )
                % (str(ROOT), str(self.store.dir), str(marker)),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.children.append(writer)
        self.assertEqual(writer.stdout.readline().strip(), "holding")
        procs.register(
            self.run_dir, "tick", pid=writer.pid
        )  # even a registered process: never killed mid-write
        out = []
        control.kill_all(self.cfg, self.store, wait=10, out=out.append)
        self.assertEqual(
            marker.read_text(), "the whole write"
        )  # the write finished before the kill
        writer.wait(
            timeout=10
        )  # (killed or not once its write was whole: either is fine)
        writer.stdout.close()
        self.assertNotIn("left alone", "\n".join(out))
        # A write still in flight after the wait: left alone to finish, and the lock keeps a new one out meanwhile.
        marker.unlink()
        slow = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib, sys, time; sys.path.insert(0, %r)\n"
                    "from rkmon_lib import control\n"
                    "with control.write_lock(pathlib.Path(%r)):\n"
                    "    print('holding', flush=True); time.sleep(2)\n"
                    "    pathlib.Path(%r).write_text('the whole write')\n"
                )
                % (str(ROOT), str(self.store.dir), str(marker)),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.children.append(slow)
        self.assertEqual(slow.stdout.readline().strip(), "holding")
        procs.register(self.run_dir, "tick", pid=slow.pid)
        out = []
        control.kill_all(self.cfg, self.store, wait=0.3, out=out.append)
        self.assertIn(f"pid {slow.pid}): it is left alone to finish", "\n".join(out))
        self.assertEqual(slow.wait(timeout=10), 0)
        slow.stdout.close()
        self.assertEqual(marker.read_text(), "the whole write")


class TestStop(_Monitor):
    def test_stop_ends_the_running_pieces_and_start_resumes(self):
        unrelated = self.spawn()
        driver = self.spawn(register="driver")
        server = self.spawn(register="serve")
        job = self.a_job()
        st = self.store.load()
        st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "first-round",
                "head": "1" * 40,
                "worker": job,
                "checks": None,
            }
        ]
        st["review_ready"] = [
            {"pr": 2, "kind": "follow-up", "head": "2" * 40, "at": OLD}
        ]
        st["queue_state"] = {
            "gate": {
                "pr": 3,
                "head": "3" * 40,
                "develop": "d",
                "job": {"id": "gate-pr3"},
            }
        }
        st["guard"] = {"running": {"sha": "a" * 40, "job": {"id": "sanity-x"}}}
        st["develop"]["pending_sanity"] = []
        self.store.save(st)
        out = []
        with nothing_runs():
            control.stop(self.cfg, self.store, out=out.append, grace=5)
        for p in (driver, server):
            p.wait(timeout=10)
        time.sleep(0.2)
        self.assertTrue(gone(job["pid"]))
        self.assertIsNone(unrelated.poll())
        st = self.store.load()
        self.assertEqual(store_mod.stopped(st)["how"], "stop")
        self.assertEqual(review.pipes(st), [])  # it starts over after rk start
        self.assertEqual(
            [r["pr"] for r in st["review_ready"]], [2]
        )  # a draft awaiting verification stays
        self.assertIsNone(st["queue_state"]["gate"])
        self.assertEqual(
            (st["guard"]["running"], st["develop"]["pending_sanity"]),
            (None, ["a" * 40]),
        )
        self.assertIn("The Claude side is stopped from Claude Code", "\n".join(out))
        # While stopped: writes and new work are refused, taking auto-merge off is not. The writer's sends for another
        # process are refused too (rkmon.py write_gate asks once it knows the command sends); its reads are not.
        self.assertTrue(control.refusal("post", st))
        self.assertTrue(control.refusal("gate", st))
        self.assertEqual(control.refusal("disarm", st), "")
        self.assertTrue(control.refusal("teams-send", st))
        self.assertEqual(control.refusal("web-requests", st), "")
        # rk start
        out, run = [], FakeRun()
        control.start(self.cfg, self.store, run=run, out=out.append)
        self.assertIsNone(store_mod.stopped(self.store.load()))
        said = "\n".join(out)
        self.assertIn(
            "/rocke-monitor resume: it starts the tick loop (tools/tick_loop.py) and the watchdog cron",
            said,
        )
        self.assertNotIn("/loop", said)
        out = []
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
            control.start(self.cfg, self.store, run=FakeRun(), out=out.append)
        self.assertIn(
            "the container's orchestrator (rk orchestrate) sends /rocke-monitor resume",
            "\n".join(out),
        )

    def test_start_saves_the_state_only_when_it_lifts_a_stop_or_a_drain(self):
        """rk start clears a stop or a drain; a home with neither keeps its state file as it was, and a home with
        none gets none."""
        path = self.store.state_path
        before = path.stat()
        control.start(self.cfg, self.store, run=FakeRun(), out=lambda *a: None)
        self.assertEqual(path.stat().st_ino, before.st_ino)  # not written again
        path.unlink()
        control.start(self.cfg, self.store, run=FakeRun(), out=lambda *a: None)
        self.assertFalse(path.exists())
        st = self.store.load()
        st["draining"] = {"at": OLD}
        self.store.save(st)
        control.start(self.cfg, self.store, run=FakeRun(), out=lambda *a: None)
        self.assertNotIn("draining", self.store.load())

    def test_start_starts_the_admin_dashboard(self):
        st = self.store.load()
        control.mark_stopped(st, "kill")
        self.store.save(st)
        run = FakeRun()
        control.start(self.cfg, self.store, run=run, out=lambda *a: None)
        self.assertTrue(
            run.calls[0][:2] == [sys.executable, "-B"]
            and run.calls[0][2].endswith("tools/rkweb.py"),
            run.calls,
        )

    def test_drain_lets_the_drafts_in_flight_finish(self):
        """rk stop --drain waits as rk drain does (rkmon_lib/drain.py), with its lines: the reviews advance under its
        flag until nothing is in flight, then the stop."""
        st = self.store.load()
        st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "first-round",
                "head": "1" * 40,
                "worker": {"id": "w1"},
                "checks": {"id": "c1"},
            }
        ]
        self.store.save(st)
        seen = []

        def advance(cfg, st, store, plan):
            seen.append(store_mod.halted(st))  # nothing new may start while it drains
            if len(seen) == 1:
                return ["REVIEW #1: worker done, checks running"]
            review.drop_pipe(st, 1)
            st.setdefault("review_ready", []).append(
                {"pr": 1, "kind": "first-round", "head": "1" * 40, "at": OLD}
            )
            return [
                "REVIEW_READY #1 (first-round): worker done, checks done. Inputs: ..."
            ]

        out = []
        with mock.patch.object(control.review, "advance", advance), nothing_runs():
            rc = control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=5,
                out=out.append,
                sleep=lambda s: None,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(seen), 2)
        self.assertTrue(
            all(
                h.startswith("the monitor is draining (rk stop --drain, since ")
                for h in seen
            ),
            seen,
        )
        self.assertEqual(
            out[:2],
            [
                "drain: nothing new starts; the work in flight finishes first (up to 5 minutes): review #1",
                "DRAINED: nothing is in flight",
            ],
        )
        self.assertTrue(out[-1].startswith("STOPPED: "), out)
        st = self.store.load()
        self.assertEqual(
            [r["pr"] for r in st["review_ready"]], [1]
        )  # the finished draft waits for verification
        self.assertEqual(store_mod.stopped(st)["how"], "stop")
        self.assertNotIn("draining", st)

    def test_a_drain_that_runs_out_of_time_stops_what_is_left(self):
        """Past its limit the drain holds until the stop records itself, so that nothing starts in between; what is
        still in flight is stopped."""
        st = self.store.load()
        st["review_pipes"] = [
            {
                "pr": 5,
                "kind": "follow-up",
                "head": "5" * 40,
                "worker": {"id": "w5"},
                "checks": None,
            }
        ]
        self.store.save(st)
        out, held = [], []
        record = control.mark_stopped

        def mark_stopped(st, how):
            held.append((st.get("draining") or {}).get("why"))
            record(st, how)

        with mock.patch.object(
            control.review,
            "advance",
            lambda *a: ["REVIEW #5: worker running, checks none"],
        ), mock.patch.object(control, "mark_stopped", mark_stopped):
            rc = control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=0,
                out=out.append,
                sleep=lambda s: None,
            )
        self.assertEqual(rc, 0)
        self.assertIn(
            "DRAIN TIMEOUT after 0 minutes: review #5 still in flight; what is left is stopped now, and the reviews "
            "start again at their heads after rk start",
            out,
        )
        self.assertEqual(held, [control.DRAIN_WHY])
        st = self.store.load()
        self.assertEqual(review.pipes(st), [])
        self.assertNotIn("draining", st)

    def test_a_stop_drain_refuses_while_another_drain_is_in_force(self):
        """Another drain in force (rk drain's, this process standing for its own) refuses rk stop --drain, as it
        refuses a second rk drain, in rk drain's words (tests/test_drain.py): REFUSED, exit 4, nothing stopped, and the
        other drain's flag stays."""
        held = store_mod.drain_record("rk drain", 60)
        st = self.store.load()
        st["draining"] = held
        self.store.save(st)
        out = []
        rc = control.stop(
            self.cfg,
            self.store,
            drain=True,
            timeout_minutes=5,
            out=out.append,
            sleep=lambda s: None,
        )
        self.assertEqual(rc, 4)
        self.assertEqual(out, [f"REFUSED: {drain.Busy(held)}; nothing stopped"])
        st = self.store.load()
        self.assertIsNone(store_mod.stopped(st))
        self.assertEqual(st["draining"], held)

    def test_a_stop_drain_that_rk_start_lifts_stops_nothing(self):
        st = self.store.load()
        st["review_pipes"] = [{"pr": 5, "kind": "follow-up", "head": "5" * 40}]
        self.store.save(st)

        def started(_seconds):  # rk start meanwhile
            control.start(self.cfg, self.store, run=FakeRun(), out=lambda *a: None)

        out = []
        with mock.patch.object(control.review, "advance", lambda *a: []):
            rc = control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=5,
                out=out.append,
                sleep=started,
            )
        self.assertEqual(rc, 1)
        self.assertEqual(
            out[-1], "UNDRAINED: rk start lifted the drain meanwhile; nothing stopped"
        )
        st = self.store.load()
        self.assertIsNone(store_mod.stopped(st))
        self.assertEqual([p["pr"] for p in review.pipes(st)], [5])

    def test_a_stop_meanwhile_leaves_nothing_to_drain_and_this_stop_goes_on(self):
        st = self.store.load()
        st["review_pipes"] = [{"pr": 5, "kind": "follow-up", "head": "5" * 40}]
        self.store.save(st)
        looks = []

        def killed(_seconds):  # rk kill meanwhile, once
            if not looks:
                with self.store.locked(wait=True):
                    st = self.store.load()
                    control.mark_stopped(st, "kill")
                    self.store.save(st)
            looks.append(_seconds)

        out = []
        with mock.patch.object(control.review, "advance", lambda *a: []):
            rc = control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=5,
                out=out.append,
                sleep=killed,
            )
        self.assertEqual(rc, 0)
        self.assertTrue(
            out[1].startswith(
                "DRAINED: nothing is left to drain: the monitor is stopped (rk kill, "
            ),
            out,
        )
        self.assertTrue(out[-1].startswith("STOPPED: "), out)
        st = self.store.load()
        self.assertEqual(store_mod.stopped(st)["how"], "stop")
        self.assertEqual(review.pipes(st), [])

    def test_a_stop_drain_cut_short_clears_its_flag_and_a_dead_one_holds_nothing(self):
        st = self.store.load()
        st["review_pipes"] = [{"pr": 5, "kind": "follow-up", "head": "5" * 40}]
        self.store.save(st)

        def interrupted(s):
            raise KeyboardInterrupt

        with mock.patch.object(
            control.review, "advance", lambda *a: []
        ), self.assertRaises(KeyboardInterrupt):
            control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=5,
                out=lambda x: None,
                sleep=interrupted,
            )
        self.assertNotIn("draining", self.store.load())
        # A drain whose process is gone (a stop killed in its wait) halts nothing, and the status says so.
        st = self.store.load()
        st["draining"] = dict(
            store_mod.drain_record("rk stop --drain", 60), pid=2**22 + 3, pstart=1
        )
        self.assertEqual(store_mod.halted(st), "")
        self.assertIn("DRAIN STALE", control.state_line(st))

    def test_in_the_container_a_stop_says_what_became_of_the_orchestrator(self):
        """In the container the Claude side is the orchestrator's session (rk orchestrate), and there is no Claude Code
        to stop it from: rk stop says whether it ended the session, or left it idle because the session ran the stop
        itself (a stop spares the processes it runs under), and that rk start, which the host's start runs too, brings
        it back."""
        back = "rk start, or ./rkmon-team start on the host, brings it back."
        container = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon-team"}

        def stop() -> str:
            out = []
            with mock.patch.dict(os.environ, container), nothing_runs():
                self.assertEqual(
                    control.stop(self.cfg, self.store, out=out.append, grace=5), 0
                )
            return out[-1]

        self.assertTrue(
            stop().endswith(". The Claude side: no orchestrator session ran."),
            "no session",
        )
        session = self.spawn(register=control.ORCHESTRATOR)
        said = stop()
        session.wait(timeout=10)
        self.assertTrue(
            said.endswith(
                ". The Claude side: this stop ended the orchestrator's session (rk orchestrate); "
                + back
            ),
            said,
        )
        # The session that runs the stop: this process, which the stop spares.
        procs.register(self.run_dir, control.ORCHESTRATOR, pid=os.getpid())
        said = stop()
        self.assertTrue(
            said.endswith(
                ". The Claude side: the orchestrator's session ran this stop itself and is left idle; "
                + back
            ),
            said,
        )
        self.assertNotIn("Claude Code", said)

    def test_a_drain_whose_stop_cannot_get_the_state_lock_says_the_drain_stays(self):
        """rk stop --drain keeps its drain until the stop records itself (drain.wait keep=True): a stop that cannot get
        the state lock says NOT STOPPED, and that the drain stays in force, so nothing new starts until rk start.
        """
        real = self.store.locked

        def locked(*a, **k):
            # the drain's waits get the lock; the stop's own, with its timeout, never does
            if "timeout" in k:
                raise TimeoutError("the state lock")
            return real(*a, **k)

        out = []
        with mock.patch.object(self.store, "locked", locked), mock.patch.object(
            control.review, "advance", lambda *a, **k: []
        ), nothing_runs():
            code = control.stop(
                self.cfg,
                self.store,
                drain=True,
                timeout_minutes=5,
                out=out.append,
                sleep=lambda s: None,
                grace=0,
            )
        self.assertEqual(code, 1)
        said = "\n".join(out)
        self.assertIn(
            "review advance and review driver was ended; the drain stays in force, so nothing new starts (rk start "
            "lifts it); run rk stop again, or rk kill",
            said,
        )
        self.assertIn("draining", self.store.load())

    def test_a_stop_that_cannot_get_the_state_lock_says_so(self):
        """The state lock held past both waits (a tick or the review driver this stop cannot end): a clear NOT STOPPED
        line and exit 1, never a traceback."""

        class Held:
            def __call__(self, *a, **k):
                raise TimeoutError("the state lock")

        out = []
        with mock.patch.object(self.store, "locked", Held()):
            code = control.stop(self.cfg, self.store, out=out.append, grace=0)
        self.assertEqual(code, 1)
        self.assertTrue(any(x.startswith("NOT STOPPED") for x in out), out)
        # It says what it ended, and that the monitor is not marked stopped.
        self.assertIn(
            "the monitor is not marked stopped, and nothing but its tick, review advance and review driver was ended",
            "\n".join(out),
        )
        self.assertNotIn("drain", "\n".join(out))  # no drain, so none stays


class TestLevels(_Monitor):
    def test_a_stop_ends_at_a_reboot(self):
        st = self.store.load()
        with mock.patch.object(store_mod, "boot_id", return_value="boot-1"):
            control.mark_stopped(st, "stop")
            self.assertTrue(store_mod.stopped(st))
        with mock.patch.object(store_mod, "boot_id", return_value="boot-2"):
            self.assertIsNone(store_mod.stopped(st))

    def test_rk_tick_and_the_writes_while_stopped(self):
        import rkmon

        st = self.store.load()
        control.mark_stopped(st, "stop")
        self.store.save(st)
        printed = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["tick"]), 0)
            self.assertIn("monitor STOPPED", printed[-1])
            self.assertEqual(rkmon.main(["post", "7", str(self.base / "p.md")]), 4)
            self.assertTrue(
                printed[-1].startswith("REFUSED: the monitor is stopped"), printed[-1]
            )
            self.assertEqual(rkmon.main(["review-advance"]), 0)
            self.assertTrue(printed[-1].startswith("STOPPED:"), printed[-1])

    def test_rk_orchestrate_running_says_whether_a_session_runs_here(self):
        """`rk orchestrate --running` exits 0, naming its pid, while a Claude session runs here (a process registered
        as control.ORCHESTRATOR), else 1. It and `--check` only ask: neither registers as a session while it runs, so
        neither finds itself."""
        import rkmon
        from rkmon_lib import orchestrator

        printed, during = [], []

        def check(cfg, check_only=False):
            during.append(control.session_pids(cfg))
            return 0

        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            orchestrator, "main", check
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["orchestrate", "--running"]), 1)
            self.assertEqual(printed[-1], "ORCHESTRATOR not running")
            self.assertEqual(rkmon.main(["orchestrate", "--check"]), 0)
            self.assertEqual(during, [[]])
            session = self.spawn(register=control.ORCHESTRATOR)
            self.assertEqual(rkmon.main(["orchestrate", "--running"]), 0)
            self.assertEqual(printed[-1], f"ORCHESTRATOR running (pid {session.pid})")
        session.kill()
        session.wait()
        self.assertEqual(control.session_pids(self.cfg), [])

    def test_rk_kill_kills_the_monitors_processes(self):
        import rkmon

        driver = self.spawn(register="driver")
        with mock.patch.object(
            rkmon.config, "folders", lambda path=None: self.cfg
        ), mock.patch.object(
            rkmon, "Store", lambda: self.store
        ), contextlib.redirect_stdout(
            io.StringIO()
        ) as out:
            self.assertEqual(rkmon.main(["kill", "--wait", "0"]), 0)
        driver.wait(timeout=10)
        self.assertIn(f"KILLED pid {driver.pid} (review driver", out.getvalue())
        self.assertEqual(driver.returncode, -signal.SIGKILL)
        self.assertEqual(store_mod.stopped(self.store.load())["how"], "kill")

    def test_rk_service_and_rk_kill_services_are_no_commands(self):
        """`rk service` and `rk kill --services` are usage errors, exit 2, that stop and kill nothing: the monitor has
        no services of its own, as the container's restart policy brings it back. rk's
        help names neither."""
        import rkmon

        driver = self.spawn(register="driver")
        for argv, said in (
            (["service", "status"], "invalid choice: 'service'"),
            (["service", "install", "--claude"], "invalid choice: 'service'"),
            (["service", "remove"], "invalid choice: 'service'"),
            (
                ["kill", "--services", "--wait", "0"],
                "unrecognized arguments: --services",
            ),
        ):
            # No command runs and no process starts.
            with self.subTest(argv=argv):
                with mock.patch.object(
                    rkmon.config, "folders", lambda path=None: self.cfg
                ), mock.patch.object(
                    rkmon, "Store", lambda: self.store
                ), mock.patch.object(
                    control, "kill_all"
                ) as kill_all, nothing_runs(), contextlib.redirect_stderr(
                    io.StringIO()
                ) as err, self.assertRaises(
                    SystemExit
                ) as exit_:
                    rkmon.main(argv)
                self.assertEqual(exit_.exception.code, 2)
                self.assertIn(said, err.getvalue())
                kill_all.assert_not_called()
        self.assertIsNone(driver.poll())  # still running: nothing was killed
        self.assertIsNone(store_mod.stopped(self.store.load()))
        for gone in ("service install", "rk service", "--services"):
            self.assertNotIn(gone, rkmon.__doc__)

    def test_no_file_of_the_package_names_wsl_exe(self):
        """The monitor runs in its container alone: no module, tool, worker script or
        container file says how a WSL distribution runs it, such as a job outliving a wsl.exe call.
        """
        found = [
            f"{p.relative_to(ROOT)}:{n}"
            for pattern in (
                "rkmon.py",
                "rkmon_lib/*.py",
                "tools/*.py",
                "worker/*.py",
                "docker/*",
            )
            for p in sorted(ROOT.glob(pattern))
            if p.is_file()
            for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
            if "wsl.exe" in line
        ]
        self.assertEqual(found, [])

    def test_rk_pause_holds_until_rk_resume(self):
        """`rk pause`: the tick does nothing and says how the pause ends, and a pause is no stop; `rk resume` ends
        it."""
        import rkmon

        printed = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["pause"]), 0)
            self.assertEqual(rkmon.main(["tick"]), 0)
            self.assertIn("monitor OFF", printed[-1])
            self.assertIn("paused until `rkmon rk resume`", printed[-1])
            st = self.store.load()
            self.assertFalse(st["enabled"])
            self.assertIn("paused", st)
            self.assertIsNone(store_mod.stopped(st))  # a pause is no stop
            self.assertEqual(rkmon.main(["resume"]), 0)
        st = self.store.load()
        self.assertTrue(st["enabled"])
        self.assertNotIn("paused", st)

    def test_the_pause_lines_name_the_program_as_it_was_run(self):
        """The tick's, the dry run's and rk init's lines on a pause name the command that ends it as the operator
        types it on the host, wherever this runs: through the wrapper as it was run (RKMON_WRAPPER, else as
        program.name() gives it), never rk alone."""
        import rkmon

        printed = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["pause"]), 0)
            box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
            with mock.patch.dict(os.environ, box):
                self.assertEqual(rkmon.main(["tick"]), 0)
                self.assertIn("(paused until `./rkmon rk resume`)", printed[-1])
                self.assertEqual(rkmon.main(["tick", "--dry-run"]), 0)
                self.assertIn("(paused until `./rkmon rk resume`)", printed[-1])
                self.assertEqual(rkmon.main(["init", "--if-new", "--on"]), 0)
                self.assertIn(
                    "`./rkmon rk resume` ends the pause (or `./rkmon start --on`)",
                    printed[-1],
                )
            with mock.patch.dict(os.environ, box), mock.patch.object(
                program, "name", return_value="/opt/pkg/docker/rkmon"
            ):
                os.environ.pop("RKMON_WRAPPER", None)
                self.assertEqual(rkmon.main(["tick"]), 0)
                self.assertIn(
                    "(paused until `/opt/pkg/docker/rkmon rk resume`)", printed[-1]
                )
            with mock.patch.dict(os.environ):
                os.environ.pop("RKMON_CONTAINER", None)
                self.assertEqual(rkmon.main(["tick"]), 0)
                self.assertIn("(paused until `rkmon rk resume`)", printed[-1])
                self.assertEqual(rkmon.main(["init", "--if-new", "--on"]), 0)
                self.assertTrue(
                    printed[-1].endswith(
                        "`rkmon rk resume` ends the pause (or `rkmon start --on`)"
                    ),
                    printed[-1],
                )

    def test_the_earlier_names_are_usage_errors_that_name_the_command(self):
        """`rk off`, `rk on` and `rk stop --wait` are no aliases: each is a usage error naming what to run instead,
        and changes nothing."""
        import rkmon

        for argv, want in (
            (["off"], 'rk: unknown command "off": rkmon rk pause pauses the monitor'),
            (["on"], 'rk: unknown command "on": rkmon rk resume ends a pause'),
            (
                ["stop", "--drain", "--wait", "5"],
                'rk: stop: unknown option "--wait": rkmon rk stop --drain --timeout-minutes N',
            ),
            (["stop", "--wait=5"], 'rk: stop: unknown option "--wait"'),
        ):
            with mock.patch.object(control, "stop") as stop, contextlib.redirect_stderr(
                io.StringIO()
            ) as err:
                self.assertEqual(rkmon.main(argv), 2, argv)
            self.assertIn(want, err.getvalue(), argv)
            stop.assert_not_called()
        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        ), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(rkmon.main(["off"]), 2)
        self.assertIn(
            'unknown command "off": ./rkmon rk pause pauses the monitor', err.getvalue()
        )  # in the container, as the operator types it there
        st = self.store.load()
        self.assertTrue(st["enabled"])
        self.assertNotIn("paused", st)

    def test_rk_settings_export_is_no_command(self):
        """`rk settings export` is gone, with no alias: its usage error names the command that took
        its place, rk config show, and it prints nothing else."""
        import rkmon

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(rkmon.main(["settings", "export"]), 2)
        self.assertIn(
            'rk: unknown command "settings export": rkmon rk config show prints the running configuration',
            err.getvalue(),
        )
        self.assertEqual(out.getvalue(), "")

    def test_rk_update_is_no_command(self):
        """`rk update` is gone, with no alias: its usage error names the host's commands that took its
        place, as the operator types them (releases.host_commands), with --rollback the way back, and nothing runs.
        """
        import rkmon

        install = "./rkmon package update, then ./rkmon confirm and ./rkmon apply install a newer package's code"
        for argv, said in (
            (["update"], install),
            (["update", "--from", "-", "--dry-run"], install),
            (["update", "--rollback"], "./rkmon rollback goes back to the code kept"),
        ):
            with self.subTest(argv=argv):
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.dict(
                    os.environ, {"RKMON_WRAPPER": "./rkmon"}
                ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(rkmon.main(argv), 2)
                self.assertEqual(
                    err.getvalue(),
                    f'rk: unknown command "update": on the host, {said}\n',
                )
                self.assertEqual(out.getvalue(), "")

    def test_rk_setup_is_no_command(self):
        """`rk setup` is gone, with any of its options, as rkmon's setup is: its usage error is
        setup's row of rkmon's GONE, on the host, as the operator types it there: the deploy settings file, login,
        secret, the deploy path's help and, with Claude Code, /rkmon-setup. Nothing runs.
        """
        import rkmon

        said = (
            'rk: unknown command "setup": on the host, set up a deployment with your deploy settings file, ./rkmon '
            "login and ./rkmon secret, as ./rkmon help deploy-path shows; with Claude Code, /rkmon-setup helps you "
            "write the file\n"
        )
        for argv in (["setup"], ["setup", "--section", "gh"]):
            with self.subTest(argv=argv):
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.dict(
                    os.environ, {"RKMON_WRAPPER": "./rkmon"}
                ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(rkmon.main(argv), 2)
                self.assertEqual((err.getvalue(), out.getvalue()), (said, ""))

    def test_the_earlier_names_are_in_the_one_list_of_old_commands(self):
        """Each earlier name INSTEAD refuses is in tests/doc_old_commands.yaml, the list the docs'
        lint reads, with the command INSTEAD names in its place; so are ON_THE_HOST and SETUP, which rk refuses
        too."""
        import rkmon
        from rkmon_lib import answers

        old = answers.load(ROOT / "tests" / "doc_old_commands.yaml")
        for key, (args, _does) in rkmon.INSTEAD.items():
            with self.subTest(key=key):
                self.assertEqual(old[f"rk {key}"]["instead"], " ".join(("rk", *args)))
        for command in (rkmon.ON_THE_HOST, rkmon.SETUP):
            self.assertIn(f"rk {command}", old)

    def test_rk_stop_takes_the_drain_limit_in_minutes(self):
        """`rk stop --drain` waits at most --timeout-minutes N minutes, and without it as long as drain.limits gives,
        as rk drain does (tests/test_drain.py); without --drain the option is rk stop's usage error: its own usage
        line, not rk's list of every command."""
        import rkmon

        for argv, minutes in (
            (["stop", "--drain"], None),
            (["stop", "--drain", "--timeout-minutes", "0"], 0),
            (["stop", "--drain", "--timeout-minutes", "2.5"], 2.5),
        ):
            with mock.patch.object(
                rkmon.config, "load", lambda path=None, check=True: self.cfg
            ), mock.patch.object(
                rkmon.config, "folders", lambda path=None: self.cfg
            ), mock.patch.object(
                rkmon, "Store", lambda: self.store
            ), mock.patch.object(
                control, "stop", return_value=0
            ) as stop:
                self.assertEqual(rkmon.main(argv), 0, argv)
            self.assertEqual(stop.call_args.kwargs["timeout_minutes"], minutes, argv)
            self.assertTrue(stop.call_args.kwargs["drain"], argv)
        with mock.patch.object(control, "stop") as stop, contextlib.redirect_stderr(
            io.StringIO()
        ) as err, self.assertRaises(SystemExit) as exit_:
            rkmon.main(["stop", "--timeout-minutes", "5"])
        self.assertEqual(exit_.exception.code, 2)
        self.assertEqual(
            err.getvalue().splitlines(),
            [
                "usage: rk stop [-h] [--drain] [--timeout-minutes N]",
                "rk stop: error: --timeout-minutes goes with --drain",
            ],
        )
        stop.assert_not_called()

    def test_status_says_which_level_holds(self):
        driver = self.spawn(register="driver")
        st = self.store.load()
        lines = control.status_lines(self.cfg, st, self.store, web_running=False)
        self.assertEqual(lines[0], "monitor: ON; running")
        self.assertIn(f"driver  pid {driver.pid}".replace("  ", " "), lines[1])
        # The monitor has no services of its own: no line names any.
        self.assertFalse([x for x in lines if x.startswith("services")], lines)
        text = "\n".join(lines)
        self.assertIn(f"  review driver: running (pid {driver.pid})", lines)
        self.assertIn(
            "  tick loop: not running; last tick never; last full tick never", lines
        )
        self.assertIn("  admin dashboard's service: not running", lines)
        self.assertIn("  gate: none", lines)
        self.assertIn("  develop sanity: none running", lines)
        # Each current setting against its range: before the scaler's first sample, the starting settings.
        self.assertIn(
            "  parallel reviews 3 of 1–5 — no sample yet (the review driver samples)",
            text,
        )
        self.assertIn("  parallel builds 1 of 1–8 — no sample yet", text)
        self.assertIn("  last changes: none", lines)
        self.assertIn("  review queue: empty", lines)
        self.assertIn("  work in flight: 0 of 3 parallel reviews: none", lines)
        self.assertIn("  builds running: 0 of 1 parallel builds: none", lines)
        self.assertEqual(lines[-1], "ramdisk: off (host.resources.ramdisk_gib 0)")
        st["draining"] = {"since": OLD}
        self.assertIn(
            "DRAINING since", control.status_lines(self.cfg, st, self.store)[0]
        )
        control.mark_stopped(st, "kill")
        st["enabled"] = False
        self.assertIn(
            "OFF (paused until rk resume); STOPPED since",
            control.status_lines(self.cfg, st, self.store)[0],
        )

    def test_status_names_the_reviews_that_wait_on_a_verifier(self):
        """Not one kept unposted (posting off: its review is finished), nor one of a skipped PR
        (review.waiting_on_verifier)."""
        st = self.store.load()
        st["review_ready"] = [
            {
                "pr": n,
                "kind": "follow-up",
                "head": "a" * 40,
                "at": "2026-11-19T07:07:00Z",
            }
            for n in (12, 13, 16)
        ]
        verified = self.base / "pr12_verified.md"
        verified.write_text("the verified review\n", encoding="utf-8")
        post.save_draft(self.cfg, st, self.store, 12, str(verified))  # kept unposted
        st["review_skip"] = [13]
        lines = control.status_lines(self.cfg, st, self.store, web_running=False)
        self.assertIn("  awaiting verification: #16 follow-up since 07:07Z", lines)


class FakeClock:
    """A clock that moves only when the code under test sleeps; each sleep is kept, and AT_SLEEP runs after it."""

    def __init__(self, at_sleep=None):
        self.t = 0.0
        self.slept = []
        self.at_sleep = at_sleep

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds
        if self.at_sleep:
            self.at_sleep(len(self.slept))


class TestTickNow(_Monitor):
    """rk tick-now: its request in the state folder, which the tick loop that runs here takes (control.take_tick), and
    its words and exit code for who takes it. The shell tick loop takes it through rk tick-wait (control.wait_tick).
    """

    def rk(self, *argv) -> tuple[int, str]:
        import rkmon

        printed = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(
            rkmon.config, "load", lambda check=True: self.cfg
        ), mock.patch.object(
            rkmon, "Store", lambda: self.store
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(list(argv))
        return code, "\n".join(printed)

    def a_loop(self) -> int:
        """A process standing in for the shell tick loop, marked as its rk tick-wait marks it: its pid."""
        pid = self.spawn().pid
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(control.wait_tick(self.cfg, self.store, 0, pid), 1)
        self.assertEqual(out.getvalue(), "")  # one look, nothing asked: nothing to say
        return pid

    def wait(self, seconds: float, loop: int, clock: FakeClock) -> int:
        return control.wait_tick(
            self.cfg, self.store, seconds, loop, sleep=clock.sleep, clock=clock
        )

    def test_who_takes_the_request(self):
        request = self.store.dir / control.TICK_NOW
        code, out = self.rk("tick-now")
        self.assertEqual(code, 3, out)
        m = re.fullmatch(
            r"WAITING: a tick at once \((\S+)\): none of the orchestrator \(rk orchestrate\), tools/tick_loop.py and "
            r"the tick loop that runs without the orchestrator \(docker/entrypoint.sh\) runs here now; the next of "
            r"them to start takes the request at once",
            out,
        )
        self.assertTrue(m, out)
        first = m.group(1)
        self.assertEqual(request.read_text(encoding="utf-8"), first + "\n")
        session = self.spawn(register=control.ORCHESTRATOR)
        code, out = self.rk("tick-now")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("ASKED: a tick at once ("), out)
        self.assertIn(f"; the request of {first} was not taken yet): ", out)
        self.assertTrue(
            out.endswith(
                f": the orchestrator (rk orchestrate, pid {session.pid}) takes the request within seconds and starts "
                "the tick (after the one in progress, if any)"
            ),
            out,
        )
        self.assertTrue(control.take_tick(self.store.dir))
        self.assertFalse(control.take_tick(self.store.dir))  # taken once
        self.assertFalse(request.exists())

    def test_the_tick_loop_takes_it_too(self):
        import fcntl

        from rkmon_lib import idle

        (self.store.dir / idle.LOOP_FILES["LOOP_PID"]).write_text("4321\n")
        with open(self.store.dir / idle.LOOP_FILES["LOOP_LOCK"], "wb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # a loop runs
            code, out = self.rk("tick-now")
        self.assertEqual(code, 0, out)
        self.assertIn(
            ": tools/tick_loop.py (pid 4321) takes the request within seconds", out
        )
        code, out = self.rk("loop-config")
        self.assertEqual(code, 0, out)
        self.assertIn(f"TICK_NOW={self.store.dir / control.TICK_NOW}", out.splitlines())

    def test_rk_status_names_the_tick_loop_that_runs_here(self):
        """rk status's tick loop line names the loops that run here as rk tick-now finds them (control.tick_loops): the
        container's shell tick loop while no orchestrator runs, the orchestrator, a host install's tools/tick_loop.py.
        The restart guide needs no word on the line: in the container too it names the loop that ticks.
        """
        self.assertNotIn(
            "tick loop: not running",
            (ROOT / "docs" / "restart.md").read_text(encoding="utf-8"),
        )
        import fcntl

        from rkmon_lib import idle

        def tick_loop() -> str:
            lines = control.status_lines(
                self.cfg, self.store.load(), self.store, web_running=False
            )
            return next(x for x in lines if x.startswith("  tick loop: "))

        last = "; last tick never; last full tick never"
        self.assertEqual(tick_loop(), "  tick loop: not running" + last)
        pid = self.a_loop()
        self.assertEqual(
            tick_loop(),
            "  tick loop: running: the tick loop that runs without the orchestrator (docker/entrypoint.sh, pid "
            f"{pid})" + last,
        )
        session = self.spawn(register=control.ORCHESTRATOR)
        self.assertEqual(
            tick_loop(),
            f"  tick loop: running: the orchestrator (rk orchestrate, pid {session.pid})"
            + last,
        )
        (self.store.dir / idle.LOOP_FILES["LOOP_PID"]).write_text("4321\n")
        with open(self.store.dir / idle.LOOP_FILES["LOOP_LOCK"], "wb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # a loop runs
            self.assertEqual(
                tick_loop(),
                f"  tick loop: running: the orchestrator (rk orchestrate, pid {session.pid}) and tools/tick_loop.py "
                "(pid 4321)" + last,
            )

    def test_refused_while_the_monitor_is_stopped(self):
        st = self.store.load()
        control.mark_stopped(st, "stop")
        self.store.save(st)
        code, out = self.rk("tick-now")
        self.assertEqual(code, 4, out)
        self.assertTrue(
            out.startswith("REFUSED: the monitor is stopped (rk stop, "), out
        )
        self.assertTrue(out.endswith("); rk start resumes it; no tick asked"), out)
        self.assertFalse((self.store.dir / control.TICK_NOW).exists())

    def test_a_request_that_cannot_be_written_is_an_error(self):
        request = self.store.dir / control.TICK_NOW
        request.mkdir()  # a folder at its path: the file cannot take its place
        code, out = self.rk("tick-now")
        self.assertEqual(code, 1, out)
        self.assertTrue(out.startswith(f"ERROR: {request} could not be written ("), out)
        self.assertTrue(out.endswith("); no tick asked"), out)
        self.assertTrue(request.is_dir())
        self.assertEqual(
            sorted(p.name for p in self.store.dir.iterdir() if "tick_now" in p.name),
            ["tick_now"],
        )  # nothing half-written is left beside it

    def test_the_shell_tick_loop_takes_it_while_no_session_runs(self):
        pid = self.a_loop()
        mark = self.store.dir / control.SHELL_LOOP_PID
        self.assertEqual(mark.read_text(encoding="utf-8"), f"{pid}\n")
        code, out = self.rk("tick-now")
        self.assertEqual(code, 0, out)
        self.assertIn(
            f": the tick loop that runs without the orchestrator (docker/entrypoint.sh, pid {pid}) takes the request "
            "within seconds and starts the tick",
            out,
        )
        at = re.match(r"ASKED: a tick at once \((\S+)\)", out).group(1)
        code, out = self.rk("tick-wait", "--loop", str(pid), "30")
        self.assertEqual(code, 0, out)
        self.assertRegex(
            out,
            rf"^TAKEN at \S+: rk tick-now's request of {re.escape(at)}; the loop starts a tick at once$",
        )
        self.assertFalse((self.store.dir / control.TICK_NOW).exists())
        code, out = self.rk("loop-config")
        self.assertEqual(code, 0, out)
        self.assertIn(f"SHELL_LOOP_PID={mark}", out.splitlines())

    def test_a_running_session_takes_it_and_the_shell_loop_leaves_it(self):
        pid = self.a_loop()
        session = self.spawn(register=control.ORCHESTRATOR)
        code, out = self.rk("tick-now")
        self.assertEqual(code, 0, out)
        self.assertIn(
            f": the orchestrator (rk orchestrate, pid {session.pid}) takes the request",
            out,
        )
        self.assertNotIn("docker/entrypoint.sh", out)
        clock = FakeClock()
        self.assertEqual(self.wait(5, pid, clock), 1)
        self.assertEqual(clock.slept, [2, 2, 1])
        self.assertTrue((self.store.dir / control.TICK_NOW).exists())

    def test_a_request_during_the_wait_ends_it(self):
        pid = self.a_loop()
        request = self.store.dir / control.TICK_NOW
        clock = FakeClock(
            at_sleep=lambda n: n == 2 and request.write_text("2026-11-23T10:00:00Z\n")
        )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.wait(600, pid, clock), 0)
        self.assertEqual(clock.slept, [2, 2])
        self.assertIn(
            ": rk tick-now's request of 2026-11-23T10:00:00Z; the loop starts a tick at once",
            out.getvalue(),
        )
        self.assertFalse(request.exists())

    def test_a_request_it_cannot_take_leaves_the_wait_as_it_is(self):
        pid = self.a_loop()
        request = self.store.dir / control.TICK_NOW
        request.mkdir()  # a folder at its path: never taken
        clock = FakeClock()
        self.assertEqual(self.wait(9, pid, clock), 1)
        self.assertEqual(clock.slept, [2, 2, 2, 2, 1])
        self.assertTrue(request.is_dir())

    def test_a_loop_that_is_gone_is_no_taker(self):
        gone = self.spawn()
        gone.kill()
        gone.wait()
        self.assertEqual(control.wait_tick(self.cfg, self.store, 0, gone.pid), 1)
        code, out = self.rk("tick-now")
        self.assertEqual(code, 3, out)
        self.assertIn(
            ": none of the orchestrator (rk orchestrate), tools/tick_loop.py and the tick loop that runs without the "
            "orchestrator (docker/entrypoint.sh) runs here now; ",
            out,
        )
        self.assertEqual(control.wait_tick(self.cfg, self.store, 0, gone.pid), 1)
        self.assertTrue((self.store.dir / control.TICK_NOW).exists())

    def test_a_configuration_with_a_problem_leaves_the_wait_to_run(self):
        """rk tick-wait loads the configuration without its check, as rk tick-min does: a problem the check finds must
        not end the shell tick loop's wait at once, or the loop would run its rounds back to back.
        """
        import rkmon

        pid = self.spawn().pid

        def load(path=None, check=True):
            if check:
                raise config.ConfigError("a problem the check finds")
            return self.cfg

        with mock.patch.object(rkmon, "load_config", lambda: load()), mock.patch.object(
            rkmon.config, "load", load
        ), mock.patch.object(
            rkmon, "Store", lambda: self.store
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            code = rkmon.main(["tick-wait", "--loop", str(pid), "0"])
        self.assertEqual(code, 1)  # it ran: one look, nothing asked
        self.assertEqual(
            (self.store.dir / control.SHELL_LOOP_PID).read_text(encoding="utf-8"),
            f"{pid}\n",
        )

    def test_a_mark_that_cannot_be_written_is_an_error_line(self):
        """The loop takes the request while the mark names it, as rk tick-now says: a folder there names no loop,
        and a mark written before still names it."""
        pid = self.spawn().pid
        mark = self.store.dir / control.SHELL_LOOP_PID
        request = self.store.dir / control.TICK_NOW
        mark.mkdir()  # a folder at its path: the mark cannot be written
        request.write_text("2026-11-23T10:00:00Z\n")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(control.wait_tick(self.cfg, self.store, 0, pid), 1)
        said = out.getvalue()
        self.assertTrue(
            said.startswith(f"ERROR: {mark} could not be written ([Errno 21] "), said
        )
        self.assertTrue(
            said.endswith(
                "); the loop takes rk tick-now's request only while the mark there names it\n"
            ),
            said,
        )
        self.assertTrue(request.exists())
        mark.rmdir()
        mark.write_text(f"{pid}\n")
        with mock.patch.object(
            control.whole_file, "write", side_effect=OSError("no room left")
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(control.wait_tick(self.cfg, self.store, 0, pid), 0)
        self.assertTrue(
            out.getvalue().startswith(
                f"ERROR: {mark} could not be written (no room left); "
            ),
            out.getvalue(),
        )
        self.assertIn("\nTAKEN at ", out.getvalue())
        self.assertFalse(request.exists())


if __name__ == "__main__":
    unittest.main()
