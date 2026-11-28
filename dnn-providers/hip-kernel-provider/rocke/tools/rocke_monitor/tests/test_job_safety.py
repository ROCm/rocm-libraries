# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Jobs, drains and state files that must hold up when something is cut short: a job past its time or lost (its partial
results are never a result, and nothing of it survives), a tick killed before it saved, a drain cut short, whose
process died or that ran past its time, and a ledger or incidents file a write left torn.
"""

import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import drain, guard, jobs, merge, procs, review  # noqa: E402
from rkmon_lib import store as store_mod  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

CFG = {
    "rocke_path": "rk",
    "repo": "o/r",
    "paths": {"repo": "/nonexistent"},
    "merge_authority": "arm-cleared",
    "deployment": {"phase": "queue"},
    "max_posts_per_hour": 6,
    "revert_offer_after_min": 120,
}


def wait_for(cond, s=10.0):
    end = time.monotonic() + s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def procs_with(word: str) -> list[int]:
    """Live (not zombie) processes whose command line holds `word`."""
    out = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (
                (p / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            state = (p / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if word in cmd and state != "Z":
            out.append(int(p.name))
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40
        self.cfg = dict(
            CFG,
            paths={
                "repo": "/nonexistent",
                "root": str(self.dir),
                "working_dir": str(self.dir),
                "daily_dir": str(self.dir / "daily"),
            },
        )


class TestKill(Base):
    def test_kill_reaches_the_whole_job_and_escalates(self):
        """A `timeout` inside a job makes a process group of its own, and a process may ignore SIGTERM: killing the
        job stops both (SIGTERM, then SIGKILL)."""
        tag = f"{os.getpid()}{time.monotonic_ns() % 100000}"
        job = jobs.start(
            self.store,
            "sanity",
            "kill",
            f"trap '' TERM\ntimeout 300 sleep 3{tag}1 &\nsleep 3{tag}2\n",
            timeout_min=5,
        )
        self.addCleanup(lambda: [os.kill(q, 9) for q in procs_with(f"sleep 3{tag}")])
        self.assertTrue(
            wait_for(lambda: len(procs_with(f"sleep 3{tag}")) >= 3)
        )  # timeout, its sleep, the sleep
        jobs.kill(self.store, job, grace_s=1)
        self.assertTrue(
            wait_for(lambda: not procs_with(f"sleep 3{tag}"), 5),
            procs_with(f"sleep 3{tag}"),
        )

    def test_survivors_of_a_lost_job_are_found_by_their_mark(self):
        tag = f"{os.getpid()}{time.monotonic_ns() % 100000}"
        job = jobs.start(
            self.store,
            "gate",
            "lost",
            f"setsid sleep 4{tag} &\nsleep 0.2\n",
            timeout_min=5,
        )
        self.addCleanup(lambda: [os.kill(q, 9) for q in procs_with(f"sleep 4{tag}")])
        self.assertTrue(wait_for(lambda: jobs.status(self.store, job) != "running"))
        self.assertTrue(
            procs_with(f"sleep 4{tag}")
        )  # its own session: the leader's session misses it
        jobs.kill(self.store, job, grace_s=1)
        self.assertTrue(wait_for(lambda: not procs_with(f"sleep 4{tag}"), 5))


class TestPartialResults(Base):
    """A gate or sanity job past its time or lost is never a result: nothing green, nothing closed."""

    def sanity(self, status_now: str, status_text: str):
        out = self.dir / "out"
        out.mkdir(exist_ok=True)
        (out / "status").write_text(status_text)
        job = {
            "id": "sanity-x-1",
            "pid": 2**22 + 1,
            "started": "2026-11-11T00:00:00Z",
            "timeout_min": 240,
        }
        self.st.setdefault("guard", {})["running"] = {
            "sha": "a" * 40,
            "job": job,
            "families": [],
            "full": False,
            "landed": [("c" * 12, 5)],
            "base": "b" * 40,
        }
        self.st["jobs"]["heavy"] = job
        killed = []
        with mock.patch.object(
            guard, "_out", lambda cfg, sha: str(out)
        ), mock.patch.object(
            jobs, "status", lambda store, j: status_now
        ), mock.patch.object(
            jobs, "kill", lambda store, j, *a, **k: killed.append(j["id"])
        ), mock.patch.object(
            guard, "worker_prefill", lambda st: True
        ):  # no new run starts in the test
            return guard.advance(self.cfg, self.st, self.store, []), killed

    def test_a_cut_short_sanity_run_is_not_green_and_closes_nothing(self):
        guard.save_incidents(
            self.store,
            [
                {
                    "id": "INC-1",
                    "status": "open",
                    "start": store_mod.now(),
                    "signals": ["build: red"],
                    "kinds": ["sanity"],
                }
            ],
        )
        for status_now in ("timeout", "lost", "failed(rc=1)"):
            with self.subTest(status=status_now):
                self.st["guard"] = {}
                self.st["develop"]["pending_sanity"] = []
                lines, killed = self.sanity(
                    status_now, "build pass ok\ngolden pass ok\n"
                )
                self.assertFalse(
                    any(x.startswith("SANITY_GREEN") for x in lines), lines
                )
                self.assertTrue(
                    any(
                        x.startswith(("SANITY_INCOMPLETE", "!! SANITY_INCOMPLETE"))
                        for x in lines
                    ),
                    lines,
                )
                self.assertIsNone(self.st["guard"].get("last_sanity_green"))
                self.assertEqual(guard.incidents(self.store)[0]["status"], "open")
                self.assertEqual(
                    killed, ["sanity-x-1"] if status_now in ("timeout", "lost") else []
                )

    def test_a_cut_short_run_runs_again_once(self):
        lines, _ = self.sanity("lost", "build pass ok\n")
        self.assertEqual(self.st["develop"]["pending_sanity"], ["a" * 40])
        self.assertIn("it runs again", lines[-1])
        lines, _ = self.sanity("lost", "")
        self.assertTrue(
            lines[-1].startswith(
                "!! SANITY_INCOMPLETE aaaaaaaaaa: job lost again (no result)"
            ),
            lines,
        )

    def test_a_finished_run_is_still_green(self):
        lines, killed = self.sanity("done", "build pass ok\ngolden pass ok\n")
        self.assertTrue(any(x.startswith("SANITY_GREEN") for x in lines), lines)
        self.assertEqual(killed, [])

    def test_a_cut_short_gate_records_nothing_and_is_killed(self):
        head = "1" * 40
        self.st["prs"] = {
            "1": {
                "n": 1,
                "head": head,
                "status": "open",
                "draft": False,
                "files_cache": {"hot": []},
            }
        }
        status = {k: ("pass", "") for k in ("merge", "build", "golden", "dg")}
        for status_now in ("timeout", "lost"):
            with self.subTest(status=status_now):
                job = {
                    "id": "gate-pr1-1",
                    "pid": 2**22 + 1,
                    "started": "t",
                    "timeout_min": 180,
                }
                self.st["queue_state"] = {
                    "gate": {"pr": 1, "head": head, "develop": "d" * 40, "job": job}
                }
                killed = []
                with mock.patch.object(
                    jobs, "status", lambda store, j: status_now
                ), mock.patch.object(
                    jobs, "kill", lambda store, j, *a, **k: killed.append(j["id"])
                ), mock.patch.object(
                    review, "read_checks_status", lambda p: (status, "d" * 40, head)
                ), mock.patch.object(
                    review, "worker_prefill", lambda st: True
                ):
                    lines = merge.advance(self.cfg, self.st, self.store, [], [])
                self.assertTrue(
                    any(
                        x.startswith(f"GATE_INCOMPLETE #1: job {status_now}")
                        for x in lines
                    ),
                    lines,
                )
                self.assertFalse(any(x.startswith("GATE_DONE") for x in lines), lines)
                self.assertIsNone(self.st["prs"]["1"].get("gates"))
                self.assertEqual(killed, ["gate-pr1-1"])

    def test_a_lost_batch_gate_validates_nothing(self):
        job = {"id": "batch-b-1", "pid": 2**22 + 1, "started": "t", "timeout_min": 360}
        bid = "bdddddddd-1-2"
        self.st["queue_state"] = {
            "batch_gate": {
                "id": bid,
                "members": [1, 2],
                "heads": {},
                "develop": "d" * 40,
                "job": job,
            }
        }
        (self.dir / "workers" / bid).mkdir(parents=True)
        (self.dir / "workers" / bid / "result").write_text("validated\n")
        with mock.patch.object(jobs, "status", lambda s, j: "lost"), mock.patch.object(
            jobs, "kill", lambda *a, **k: None
        ), mock.patch.object(review, "worker_prefill", lambda st: True):
            lines = merge.advance(self.cfg, self.st, self.store, [], [])
        self.assertFalse(any(x.startswith("BATCH_VALIDATED") for x in lines), lines)
        self.assertIsNone(self.st["queue_state"].get("validated"))


class TestOrphans(Base):
    """A tick killed before it saved leaves jobs the state does not name: the next tick kills them."""

    def test_the_next_tick_kills_jobs_started_after_the_last_save(self):
        tag = f"{os.getpid()}{time.monotonic_ns() % 100000}"
        let_go = jobs.start(
            self.store, "worker", "pr5", f"sleep 5{tag}1", timeout_min=5
        )  # named by no state
        self.addCleanup(lambda: [os.kill(q, 9) for q in procs_with(f"sleep 5{tag}")])
        self.st["enabled"] = True
        self.store.save(self.st)  # a save after it: let go on purpose, left alone
        time.sleep(0.05)
        orphan = jobs.start(
            self.store, "worker", "pr6", f"sleep 5{tag}2", timeout_min=5
        )  # the killed tick's
        self.assertTrue(wait_for(lambda: procs_with(f"sleep 5{tag}2")))
        self.assertEqual(
            [m["id"] for m in jobs.orphans(self.store, self.st)], [orphan["id"]]
        )
        lines = rkmon.tick_checks(self.st, self.store)
        self.assertEqual(
            lines,
            [
                f"JOB_ORPHAN {orphan['id']}: started by a tick or command that never saved; killed "
                "(its work starts again, tracked)"
            ],
        )
        self.assertTrue(wait_for(lambda: not procs_with(f"sleep 5{tag}2")))
        self.assertEqual(jobs.status(self.store, let_go), "running")
        jobs.kill(self.store, let_go, grace_s=1)


class TestDrain(Base):
    """A drain (rkmon_lib/drain.py) on real jobs and in the tick: a list of heavy holders, a wait cut short, a dead or
    late drainer."""

    def setUp(self):
        super().setUp()
        self.store.save(dict(self.st, enabled=True))
        p = mock.patch.object(review, "advance", lambda *a, **k: [])
        p.start()
        self.addCleanup(p.stop)

    def test_in_flight_takes_a_list_of_heavy_holders(self):
        a = jobs.start(self.store, "checks", "pr1", "sleep 30", timeout_min=5)
        b = jobs.start(self.store, "gate", "pr2", "sleep 30", timeout_min=5)
        self.addCleanup(lambda: [jobs.kill(self.store, j, grace_s=1) for j in (a, b)])
        self.assertEqual(
            drain.in_flight({"jobs": {"heavy": [a, b]}}, self.store),
            [f"build job {a['id']}", f"build job {b['id']}"],
        )

    def test_a_drain_cut_short_clears_its_flag(self):
        def interrupted(s):
            raise KeyboardInterrupt

        with mock.patch.object(
            drain, "in_flight", lambda st, store: ["review #7"]
        ), self.assertRaises(KeyboardInterrupt):
            drain.wait(
                {}, self.store, drain.WHY, 5, 0, say=lambda x: None, sleep=interrupted
            )
        self.assertNotIn("draining", self.store.load())
        with mock.patch.object(
            drain,
            "in_flight",
            lambda st, store: (_ for _ in ()).throw(AttributeError("x")),
        ), self.assertRaises(AttributeError):
            drain.wait(
                {},
                self.store,
                drain.WHY,
                5,
                0,
                say=lambda x: None,
                sleep=lambda s: None,
            )
        self.assertNotIn("draining", self.store.load())

    def test_a_drain_holds_while_its_process_lives_and_the_tick_says_so(self):
        with mock.patch.object(drain, "in_flight", lambda st, store: []):
            drain.wait({}, self.store, drain.WHY, 60, 0, say=lambda x: None)
        st = self.store.load()
        self.assertEqual(
            (st["draining"]["pid"], st["draining"]["pstart"]),
            (os.getpid(), store_mod.proc_start(os.getpid())),
        )
        self.assertIn("draining", store_mod.halted(st))
        self.assertEqual(
            [x.split(" (")[0] for x in rkmon.tick_checks(st, self.store)],
            [f"DRAINING since {st['draining']['since']}"],
        )
        self.assertIn("draining", self.store.load())

    def test_a_dead_or_late_drain_holds_nothing_and_the_tick_clears_it(self):
        for d, why in (
            (
                {
                    "since": store_mod.now(),
                    "why": drain.WHY,
                    "until_min": 60,
                    "pid": 2**22 + 3,
                    "pstart": "1",
                },
                "the process that set it (4194307) is gone",
            ),
            (
                {"since": "2026-01-01T00:00:00Z", "why": drain.WHY, "until_min": 60},
                "past its 60 min",
            ),
        ):
            with self.subTest(why=why):
                st = dict(self.store.load(), draining=d)
                self.store.save(st)
                self.assertEqual(store_mod.halted(st), "")
                lines = rkmon.tick_checks(st, self.store)
                self.assertTrue(
                    lines[-1].startswith(f"!! DRAIN_CLEARED ({drain.WHY}")
                    and why in lines[-1],
                    lines,
                )
                self.assertNotIn("draining", self.store.load())

    def test_a_drain_that_names_no_command_is_rk_stop_s(self):
        """The tick names a drain whose record has no why as rk stop --drain's, as the status and the store do."""
        held = {
            "since": store_mod.now(),
            "until_min": 60,
            "pid": os.getpid(),
            "pstart": store_mod.proc_start(os.getpid()),
        }
        late = {"since": "2026-01-01T00:00:00Z", "until_min": 60}
        for d, line in (
            (held, f"DRAINING since {held['since']} (rk stop --drain): "),
            (late, "!! DRAIN_CLEARED (rk stop --drain, since 2026-01-01T00:00:00Z): "),
        ):
            with self.subTest(line=line):
                st = dict(self.store.load(), draining=d)
                self.store.save(st)
                lines = rkmon.tick_checks(st, self.store)
                self.assertTrue(lines[-1].startswith(line), lines)

    def test_sigterm_ends_the_run_through_its_cleanup(self):
        import signal

        with self.assertRaises(SystemExit) as cm, procs.signals_exit():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)


class TestTornFiles(Base):
    """A torn ledger line or an empty incidents file never stops every later tick."""

    def test_a_torn_last_ledger_line_is_skipped_then_moved_aside(self):
        self.store.ledger_add("post", 1, "h1", "u1")
        with self.store.ledger_path.open("a") as f:
            f.write('{"ts": "2026-11-11T00:00:00Z", "kind": "po')
        self.assertEqual([r["pr"] for r in self.store.ledger()], [1])
        self.assertTrue(self.store.ledger_has("post", 1, "h1"))
        self.store.ledger_add("post", 2, "h2", "u2")
        self.assertEqual([r["pr"] for r in self.store.ledger()], [1, 2])
        self.assertIn('"kind": "po', (self.store.dir / "actions.torn").read_text())
        with self.store.ledger_path.open(
            "a"
        ) as f:  # a bad line in the middle is not a torn write: it still raises
            f.write("garbage\n")
        self.store.ledger_add("post", 3, "h3", "u3")
        with self.assertRaises(ValueError):
            self.store.ledger()

    def test_an_empty_or_corrupt_incidents_file_is_recovered(self):
        guard.save_incidents(self.store, [{"id": "INC-1", "status": "open"}])
        guard.save_incidents(
            self.store,
            [{"id": "INC-1", "status": "open"}, {"id": "INC-2", "status": "open"}],
        )
        path = self.store.dir / guard.INCIDENTS_FILE
        path.write_text('[{"id": "INC-1", "sta')  # cut short
        self.assertEqual(
            [i["id"] for i in guard.incidents(self.store)], ["INC-1"]
        )  # the copy the last save kept
        self.assertEqual(len(list(self.store.dir.glob("incidents.json.bad-*"))), 1)
        self.assertEqual(
            json.loads(path.read_text()), [{"id": "INC-1", "status": "open"}]
        )
        path.write_text("")
        (self.store.dir / "incidents.json.prev").unlink()
        self.assertEqual(guard.incidents(self.store), [])
        self.assertIn("incidents.json was empty", self.store.journal_path.read_text())

    def test_rk_doctor_says_a_drain_and_a_torn_file(self):
        from rkmon_lib import doctor

        st = self.store.load()
        st["draining"] = store_mod.drain_record(drain.WHY, 60)
        self.store.save(st)
        checks = {c.name: c for c in doctor.state_checks(self.store.dir)}
        self.assertEqual(checks["drain"].status, "WARN")
        self.assertIn(f"DRAINING: {drain.WHY} since", checks["drain"].found)
        st["draining"].update(pid=2**22 + 3, pstart=1)  # its process is gone
        self.store.save(st)
        self.store.ledger_add("post", 1, "h1", "u1")
        with self.store.ledger_path.open("a") as f:
            f.write('{"ts": "2026')
        (self.store.dir / guard.INCIDENTS_FILE).write_text("[{")
        checks = {c.name: c for c in doctor.state_checks(self.store.dir)}
        self.assertEqual(checks["drain"].status, "FAIL")
        self.assertIn("is gone; the next tick clears it", checks["drain"].found)
        self.assertEqual(
            (checks["ledger"].status, checks["incidents"].status), ("WARN", "WARN")
        )


class TestJobIds(Base):
    def test_two_starts_in_one_second_get_their_own_folders(self):
        """Two jobs of the same kind and name started in the same second never share a folder (log, pid, meta, rc)."""
        second = store_mod.dt.datetime(2026, 1, 1, tzinfo=store_mod.dt.timezone.utc)
        with mock.patch.object(jobs.dt, "datetime") as fake:
            fake.now.return_value = second
            a = jobs.start(self.store, "checks", "pr5", "echo A", timeout_min=5)
            b = jobs.start(self.store, "checks", "pr5", "echo B", timeout_min=5)
        self.assertEqual(b["id"], a["id"] + "-2")
        self.assertTrue(
            wait_for(
                lambda: jobs.status(self.store, a) == "done"
                and jobs.status(self.store, b) == "done"
            )
        )
        self.assertEqual(jobs.tail(self.store, a).strip(), "A")
        self.assertEqual(jobs.tail(self.store, b).strip(), "B")


if __name__ == "__main__":
    unittest.main()
