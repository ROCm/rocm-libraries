# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The state folder over time: logs rotate past retention.log_max_mb and their readers keep both parts, the journal's
tail is read from its end, job folders of jobs killed or lost are pruned, the state, config.json and the dashboards
are written whole, a weekly report missed while the monitor was down is written late, and nothing new starts while a
stop or a drain holds (the draft pool, shadow drafts, model roles, remote GPU jobs, `rk gate`, `rk sanity`, `rk bisect`,
`rk review-recheck`), which a drain waits for (rkmon_lib/drain.py)."""

import datetime as dt
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    cleanup,
    config,
    draft_stage,
    drain,
    gpu,
    jobs,
    retention,
    roles,
    stats,
    usage,
    whole_file,
)
from rkmon_lib.store import (  # noqa: E402
    Store,
    drain_record,
    empty_state,
    halted,
    now,
)

UTC = dt.timezone.utc


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")


class TestRetention(Base):
    def test_a_log_past_its_size_moves_aside_and_its_readers_read_both(self):
        cfg = {"retention": {"log_max_mb": 0.0001}}  # about 100 bytes
        self.store.append_jsonl(
            "events.jsonl", [{"ts": now(), "kind": "A", "i": i} for i in range(5)]
        )
        self.assertEqual(
            retention.rotate(cfg, self.store)[0].split(":")[0], "RETENTION events.jsonl"
        )
        self.store.append_jsonl("events.jsonl", [{"ts": now(), "kind": "B"}])
        self.assertEqual(
            [e["kind"] for e in self.store.read_jsonl("events.jsonl")],
            ["A"] * 5 + ["B"],
        )
        self.assertTrue((self.store.dir / "events.jsonl.1").is_file())
        self.assertEqual(
            retention.rotate({}, self.store), []
        )  # under 20 MB: nothing moves

    def test_the_journal_and_usage_records_keep_their_rotated_part(self):
        for i in range(30):
            self.store.journal("x", f"line {i}")
        retention.rotate({"retention": {"log_max_mb": 0.0001}}, self.store)
        self.store.journal("x", "after")
        lines = self.store.journal_lines()
        self.assertTrue(
            lines[0].endswith("line 0") and lines[-1].endswith("after"), lines
        )
        self.assertEqual(len(self.store.journal_tail(3)), 3)
        path = self.store.dir / "claude_usage.jsonl"
        path.write_text(json.dumps({"ts": "2026-01-02T00:00:00Z", "n": 1}) + "\n")
        os.replace(path, retention.older(path))
        path.write_text(json.dumps({"ts": "2026-01-03T00:00:00Z", "n": 2}) + "\n")
        self.assertEqual([r["n"] for r in usage._jsonl(path, "2026-01-01")], [1, 2])

    def test_the_journal_tail_is_read_from_the_end(self):
        for i in range(5000):
            self.store.journal("x", f"line {i}")
        tail = self.store.journal_tail(3)
        self.assertEqual([t.split()[-1] for t in tail], ["4997", "4998", "4999"])

    def test_the_service_log_is_copied_then_truncated(self):
        log = self.store.dir / "web.log"
        with open(log, "ab") as held:  # rk serve holds it open for appending
            held.write(b"x" * 500)
            held.flush()
            retention.rotate({"retention": {"log_max_mb": 0.0001}}, self.store)
            held.write(b"next\n")
        self.assertEqual(log.read_bytes(), b"next\n")
        self.assertEqual(retention.older(log).read_bytes(), b"x" * 500)

    def test_the_ledger_is_never_rotated(self):
        self.assertNotIn("actions.jsonl", retention.LOGS)


class TestJobFolders(Base):
    def test_a_lost_job_with_no_rc_is_pruned_once_old_and_named_by_nothing(self):
        lost = self.store.dir / "jobs" / "checks-pr5-1"
        lost.mkdir(parents=True)
        (lost / "meta.json").write_text(
            json.dumps(
                {
                    "id": lost.name,
                    "pid": 2**22 + 5,
                    "started": "2026-01-01T00:00:00Z",
                    "timeout_min": 5,
                }
            )
        )
        named = self.store.dir / "jobs" / "checks-pr6-1"
        named.mkdir()
        meta6 = {
            "id": named.name,
            "pid": 2**22 + 6,
            "started": "2026-01-01T00:00:00Z",
            "timeout_min": 5,
        }
        (named / "meta.json").write_text(json.dumps(meta6))
        alive = jobs.start(self.store, "worker", "pr7", "sleep 30", timeout_min=5)
        self.addCleanup(lambda: jobs.kill(self.store, alive, grace_s=1))
        st = empty_state()
        st["review_pipes"] = [{"pr": 6, "worker": meta6}]
        old = time.time() - 30 * 86400
        for d in (lost, named, self.store.dir / "jobs" / alive["id"]):
            os.utime(d, (old, old))
        known = jobs.referenced(st)
        self.assertTrue(cleanup.job_done(lost, known))
        self.assertFalse(cleanup.job_done(named, known))
        self.assertFalse(cleanup.job_done(self.store.dir / "jobs" / alive["id"], known))
        kept = cleanup._old(
            self.store.dir / "jobs", 7, keep=lambda p: not cleanup.job_done(p, known)
        )
        self.assertEqual([p.name for p in kept], [lost.name])


class TestWrittenWhole(Base):
    def test_the_state_and_the_pages(self):
        self.store.save(dict(empty_state(), x=1))
        self.assertEqual(self.store.load()["x"], 1)
        # No new file of the write is left beside state.json: the name whole_file gives it (beside).
        left = whole_file.NEW_FILE.format(name=self.store.state_path.name, unique="*")
        self.assertTrue(whole_file.beside(self.store.state_path).match(left))
        self.assertEqual(list(self.store.dir.glob(left)), [])
        page = self.dir / "p.html"
        whole_file.write(page, "<p>one</p>", 0o644)
        whole_file.write(page, "<p>two</p>", 0o600)
        self.assertEqual(page.read_text(), "<p>two</p>")
        self.assertEqual(
            page.stat().st_mode & 0o777, 0o600
        )  # the permissions it was given
        self.assertEqual([p.name for p in self.dir.glob(".p.html*")], [])


class TestWeeklyCatchUp(unittest.TestCase):
    def test_a_week_missed_while_down_is_written_late(self):
        # Monday of 2026-W46; the report is due on Fridays. W44 was the last written: W45 is owed now, W46 not yet.
        t = dt.datetime(2026, 11, 16, 8, tzinfo=UTC)
        self.assertEqual(stats.owed_weeks("2026-W44", t, due=False), ["2026-W45"])
        self.assertEqual(stats.owed_weeks("2026-W45", t, due=False), [])
        self.assertEqual(
            stats.owed_weeks("2026-W43", t, due=True),
            ["2026-W44", "2026-W45", "2026-W46"],
        )
        self.assertEqual(
            stats.owed_weeks(None, t, due=False), []
        )  # never written: no history backfill
        self.assertEqual(stats.owed_weeks(None, t, due=True), ["2026-W46"])
        self.assertEqual(
            len(stats.owed_weeks("2025-W01", t, due=True)), stats.CATCH_UP_WEEKS
        )

    def test_the_tick_writes_each_owed_week_oldest_first(self):
        cfg = {"weekly_report": {"weekday": 4, "hour_utc": 9}}
        st = {"weekly_done": "2026-W44"}
        written = []

        def write(cfg, st, store, label):
            written.append(label)
            return label, pathlib.Path(f"/w/{label}.md")

        t = dt.datetime(
            2026, 11, 20, 10, tzinfo=UTC
        )  # Friday of W47, past 09:00: W45 and W46 owed
        with mock.patch.object(rkmon, "write_weekly", write), mock.patch.object(
            rkmon.lessons, "collect", lambda *a: "/w/l.md"
        ):
            out = rkmon.maybe_weekly_report(cfg, st, mock.Mock(), t)
        self.assertEqual(written, ["2026-W45", "2026-W46"])
        self.assertEqual(st["weekly_done"], "2026-W46")
        self.assertIn("WEEKLY REPORT 2026-W45 ready", out)


class TestNothingNewWhileHalted(Base):
    def halted_st(self) -> dict:
        st = empty_state()
        st["draining"] = drain_record(drain.WHY, 60)
        return st

    def test_the_pool_shadow_roles_and_gpu_start_nothing(self):
        st = self.halted_st()
        cfg = {"reviewer": "x"}
        with mock.patch.object(
            draft_stage, "for_config", lambda cfg: mock.Mock(parallel=True, name="s")
        ), mock.patch.object(
            draft_stage, "candidates", side_effect=AssertionError("no pick is read")
        ):
            self.assertEqual(draft_stage.pool_advance(cfg, st, self.store, []), [])
        with mock.patch.object(
            draft_stage, "shadow_binding", lambda cfg: "b"
        ), mock.patch.object(
            draft_stage, "live_reviews", side_effect=AssertionError("nothing starts")
        ):
            self.assertEqual(draft_stage.shadow_start(cfg, st, self.store, []), [])
        with mock.patch.object(
            roles, "wanted", side_effect=AssertionError("nothing starts")
        ):
            self.assertEqual(roles.start(cfg, st, self.store, "tick", "fp", []), "")
        out = gpu.start(
            cfg, st, self.store, 7, "a" * 40, "d" * 40, {"gfx942": ["t"]}, "review"
        )
        self.assertTrue(out[0].startswith("GPU_WAIT #7: the monitor is draining"), out)
        self.assertEqual((st.get("gpu") or {}).get("jobs") or [], [])

    def test_hand_run_heavy_starts_refuse_while_halted_or_busy(self):
        st = self.halted_st()
        self.assertIn(
            "draining", rkmon.heavy_start_refusal({}, st, self.store, 7, now_=True)
        )
        st.pop("draining")
        with mock.patch.object(jobs, "heavy_busy", lambda *a: {"id": "gate-pr5-1"}):
            self.assertIn(
                "all parallel builds are busy (gate-pr5-1",
                rkmon.heavy_start_refusal({}, st, self.store, 7),
            )
            self.assertIsNone(
                rkmon.heavy_start_refusal({}, st, self.store, 7, now_=True)
            )
        with mock.patch.object(jobs, "heavy_busy", lambda *a: None), mock.patch.object(
            rkmon.scaler, "room", lambda *a: (False, "the RAM disk has 1.0 GiB free")
        ):
            self.assertIn(
                "the RAM disk has 1.0 GiB free",
                rkmon.heavy_start_refusal({}, st, self.store, 7),
            )
        with mock.patch.object(jobs, "heavy_busy", lambda *a: None), mock.patch.object(
            rkmon.scaler, "room", lambda *a: (True, "")
        ):
            self.assertIsNone(rkmon.heavy_start_refusal({}, st, self.store, 7))

    def test_a_hand_run_heavy_start_says_why_once(self):
        """A gate, sanity run or bisect started by hand while the monitor drains or is stopped is refused in
        store.halted's words alone: a drain's words say once that nothing new starts, a stop's that rk start resumes
        it.
        """
        st = self.halted_st()
        why = rkmon.heavy_start_refusal({}, st, self.store, 7, now_=True)
        self.assertEqual(why, halted(st))
        self.assertEqual(why.count("nothing new starts"), 1, why)
        st = empty_state()
        st["stopped"] = {"at": "2026-01-01T00:00:00Z", "how": "stop", "boot": ""}
        why = rkmon.heavy_start_refusal({}, st, self.store, 7)
        self.assertEqual(why, halted(st))
        self.assertTrue(why.endswith("; rk start resumes it"), why)

    def test_a_hand_run_recheck_says_why_once(self):
        """rk review-recheck while the monitor drains is refused in store.halted's words alone, as a heavy start is: a
        drain's words say once that nothing new starts."""
        st = self.halted_st()
        self.store.save(st)
        out = []
        with mock.patch.object(rkmon, "load_config", lambda: {}), mock.patch.object(
            rkmon, "Store", lambda: self.store
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["review-recheck", "7"]), 4)
        self.assertEqual(out, [f"REFUSED: {halted(st)}"])
        self.assertEqual(out[0].count("nothing new starts"), 1, out)

    def test_the_drain_waits_for_pool_shadow_role_and_gpu_jobs(self):
        job = jobs.start(self.store, "pool", "pr5", "sleep 30", timeout_min=5)
        self.addCleanup(lambda: jobs.kill(self.store, job, grace_s=1))
        st = {
            "draft_pool": {"5": {"job": job}},
            "shadow_pool": {"6": {"job": job}},
            "roles": {"tick": {"job": job}},
            "gpu": {"jobs": [{"id": "g1", "stage_job": job}]},
        }
        self.assertEqual(
            drain.in_flight(st, self.store),
            ["pool draft #5", "shadow draft #6", "role job tick", "GPU staging g1"],
        )


if __name__ == "__main__":
    unittest.main()
