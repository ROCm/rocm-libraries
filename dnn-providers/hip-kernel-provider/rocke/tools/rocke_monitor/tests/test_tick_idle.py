# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The idle short-circuit of `rk tick` (rkmon_lib/idle.py): the change fingerprint and its parts, the timers, idle and
full ticks, the dry run, the "checked" stamps and the tick counts on the board. No network: GitHub and the web host are
fakes."""

import contextlib
import datetime as dt
import io
import json
import pathlib
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rkmon  # noqa: E402  (registers every part, timer and stamp)
from rkmon_lib import (  # noqa: E402
    gh,
    idle,
    jobs,
    publish,
    registry,
    report,
    tickfilter,
)
from rkmon_lib.config import scope as scope_of  # noqa: E402  (config: the fixture)
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

import guard_labels  # noqa: E402

HEAD = "d" * 40


def ts(minutes: float) -> str:
    """UTC now plus `minutes` (negative: ago), in the state's format."""
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def config(d: pathlib.Path, **extra) -> dict:
    return dict(
        {
            "repo": "o/r",
            "operator": {"github_login": "me"},
            "deployment": {"phase": "review"},
            "merge_authority": "propose",
            "settle_min": 30,
            "scope": {"labels": ["scope: a", "scope: b"], "path_labels": ["area: x"]},
            "develop_watched_workflows": ["CI"],
            "develop_rocke_job_pattern": "unit",
            "incident_issue": guard_labels.INCIDENT_ISSUE,
            "codeowner_teams": {},
            "paths": {
                "dashboard": str(d / "queue.md"),
                "html": str(d / "page.html"),
                "repo": str(d / "repo"),
            },
        },
        **extra,
    )


def pr(
    n: int,
    head: str | None = None,
    updated: str = "2026-11-06T10:00:00Z",
    labels=("scope: a",),
) -> dict:
    return {
        "number": n,
        "isDraft": False,
        "updatedAt": updated,
        "headRefOid": head or f"{n:040d}",
        "mergeable": "MERGEABLE",
        "autoMergeRequest": None,
        "labels": {"nodes": [{"name": x} for x in labels]},
        "statusCheckRollup": {"state": "SUCCESS"},
    }


class FakeGitHub:
    """GitHub as the fingerprint's parts read it: the PR searches, the branch head, the develop issues, the CI runs."""

    def __init__(self):
        self.label_prs = [pr(1), pr(2)]
        self.path_prs = [pr(3, labels=("area: x",)), pr(4, labels=("area: x",))]
        self.issues = []
        self.head = HEAD
        self.runs: dict[str, list] = {}
        self.queries: list[str] = []
        self.rest_paths: list[str] = []

    @staticmethod
    def _page(nodes):
        return {
            "search": {
                "issueCount": len(nodes),
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": nodes,
            }
        }

    def graphql(self, query, timeout=120, attempts=3, **v):
        self.queries.append(v.get("q") or query)
        if "search(" in query:
            if "is:issue" in v["q"]:
                return self._page(self.issues)
            return self._page(self.path_prs if "-label:" in v["q"] else self.label_prs)
        ref = {"name": "develop", "target": {"oid": self.head}}
        if "defaultBranchRef" in query:
            return {"repository": {"defaultBranchRef": ref}}
        if "ref(qualifiedName" in query:
            return {"repository": {"ref": dict(ref, name=v["r"].rsplit("/", 1)[-1])}}
        raise AssertionError(query)

    def rest(self, path, jq=None, paginate=False, timeout=120):
        self.rest_paths.append(path)
        if "/actions/runs?" in path:
            return self.runs.get(path.split("head_sha=")[1].split("&")[0], [])
        raise AssertionError(path)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.store = Store(self.d / "state")
        self.cfg = config(self.d)
        self.st = empty_state()
        self.st["enabled"] = True
        self.st["develop"]["sha"] = HEAD
        self.gh = FakeGitHub()
        for p in (
            mock.patch.object(gh, "graphql", self.gh.graphql),
            mock.patch.object(gh, "rest", self.gh.rest),
            mock.patch.object(
                publish, "pull_requests", lambda cfg, dest, run=None: None
            ),
            mock.patch.object(publish, "html", lambda cfg, path, run=None: None),
        ):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()


def part(name, fn, **kw):
    return {name: idle.Part(name, fn, **kw)}


class TestEngine(Base):
    """decide, record_full and deadlines with registries of their own."""

    def record(self, fp, full_at=None, due=None):
        self.st[idle.KEY] = {"fp": fp, "full_at": full_at or ts(-5), "due": due or {}}

    def test_idle_only_when_every_part_reads_the_same_and_nothing_is_due(self):
        parts = part("a", lambda c, s, st: 1)
        self.record({"a": 1})
        d = idle.decide(self.cfg, self.st, self.store, parts=parts)
        self.assertTrue(d.idle, d.reasons)
        self.assertEqual(d.fp, {"a": 1})
        self.record({"a": 2})
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, parts=parts).reasons,
            ["changed: a"],
        )
        self.record({"a": 1}, due={"settle #5": ts(-1), "gap #6": ts(30)})
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, parts=parts).reasons,
            ["due: settle #5"],
        )

    def test_full_without_a_record_with_full_and_at_the_interval(self):
        parts = part("a", lambda c, s, st: 1)
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, parts=parts).reasons,
            ["no full tick on record"],
        )
        self.record({"a": 1})
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, full=True, parts=parts).reasons,
            ["--full"],
        )
        self.record({"a": 1}, full_at=ts(-61))
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, parts=parts).reasons,
            ["the last full tick is 60 min old"],
        )
        self.cfg["idle_full_tick_min"] = 90
        self.assertTrue(idle.decide(self.cfg, self.st, self.store, parts=parts).idle)
        self.cfg["idle_full_tick_min"] = 0
        self.assertEqual(
            idle.decide(self.cfg, self.st, self.store, parts=parts).reasons,
            ["monitor.full_tick_max_interval_minutes is 0"],
        )
        self.cfg["idle_full_tick_min"] = "often"
        with self.assertRaises(ValueError):
            idle.decide(self.cfg, self.st, self.store, parts=parts)

    def test_a_part_that_cannot_be_read_makes_the_tick_full_and_leaves_no_fingerprint(
        self,
    ):
        def boom(c, s, st):
            raise gh.GhError("gh api graphql: HTTP 502")

        parts = {**part("a", lambda c, s, st: 1), **part("b", boom, remote=True)}
        self.record({"a": 1, "b": 2})
        d = idle.decide(self.cfg, self.st, self.store, parts=parts)
        self.assertFalse(d.idle)
        self.assertIsNone(d.fp)
        self.assertTrue(
            d.reasons[0].startswith(
                "fingerprint not read (b: gh api graphql: HTTP 502"
            ),
            d.reasons,
        )
        self.store.save(self.st)
        rec = idle.record_full(self.cfg, self.store, d, 1.0, parts=parts, timers={})
        self.assertIsNone(rec["fp"])  # the next tick is full again
        st = self.store.load()
        self.assertFalse(
            idle.decide(
                self.cfg, st, self.store, parts=part("a", lambda c, s, st: 1)
            ).idle
        )

    def test_a_part_added_or_removed_by_a_code_change_makes_the_tick_full(self):
        self.record({"a": 1, "gone": 3})
        self.assertEqual(
            idle.decide(
                self.cfg, self.st, self.store, parts=part("a", lambda c, s, st: 1)
            ).reasons,
            ["changed: gone"],
        )

    def test_remote_parts_are_read_in_parallel(self):
        def slow(c, s, st):
            time.sleep(0.4)
            return 1

        parts = {
            **part("x", slow, remote=True),
            **part("y", slow, remote=True),
            **part("z", slow, remote=True),
        }
        t0 = time.monotonic()
        values, secs, errors = idle.probe(self.cfg, self.st, self.store, parts)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual((values, errors), ({"x": 1, "y": 1, "z": 1}, []))
        self.assertEqual(set(secs), {"x", "y", "z"})

    def test_record_full_keeps_the_start_values_and_reads_the_end_parts_again(self):
        self.st["x"] = 1
        parts = {
            **part("remote", lambda c, s, st: s["x"], remote=True),
            **part("local", lambda c, s, st: s["x"], end=True),
            **part(
                "jobs",
                lambda c, s, st: dict(s["jobs_now"]),
                end=True,
                merge=jobs._keep_start,
            ),
        }
        self.st["jobs_now"] = {"j1": "running"}
        d = idle.decide(self.cfg, self.st, self.store, parts=parts)
        self.st.update(
            x=2, jobs_now={"j1": "done", "j2": "running"}
        )  # what the full tick changed and saved
        self.store.save(self.st)
        rec = idle.record_full(self.cfg, self.store, d, 12.5, parts=parts, timers={})
        self.assertEqual(
            rec["fp"],
            {"remote": 1, "local": 2, "jobs": {"j1": "running", "j2": "running"}},
        )
        self.assertEqual(rec["full_at"], d.at)
        st = self.store.load()
        self.assertEqual(st[idle.KEY], rec)
        self.assertEqual([r[1:] for r in st[idle.STATS]["recent"]], [["full", 12.5]])

    def test_deadlines_one_shot_again_and_failing_timers(self):
        def failing(c, s, st):
            raise KeyError("prs")

        timers = {
            "one": idle.Timer(
                "one", lambda c, s, st: [("past", ts(-1)), ("ahead", ts(5))]
            ),
            "again": idle.Timer(
                "again", lambda c, s, st: [("retry #1", ts(-3))], again=True
            ),
            "bad": idle.Timer("bad", failing),
            "dup": idle.Timer("dup", lambda c, s, st: [("ahead", ts(2))]),
        }
        due = idle.deadlines(self.cfg, self.st, self.store, timers)
        self.assertEqual(set(due), {"ahead", "retry #1", "bad (not computed: 'prs')"})
        self.assertLessEqual(due["ahead"], ts(3))  # the earliest of two
        self.assertLessEqual(
            due["bad (not computed: 'prs')"], now()
        )  # due at once: fail closed

    def test_the_state_part_ignores_only_what_an_idle_tick_writes(self):
        read = idle.PARTS["state"].read
        a = read(self.cfg, self.st, self.store)
        self.st.update(
            last_tick=now(),
            tick_stats={"recent": [[now(), "idle", 1.0]]},
            idle={"fp": {}},
        )
        self.assertEqual(read(self.cfg, self.st, self.store), a)
        self.st["holds"] = {"5": {"status": "held"}}
        self.assertNotEqual(read(self.cfg, self.st, self.store), a)

    def test_settings_and_loop_config(self):
        # each setting at its default, which the registry keeps (rkmon_lib/registry.py)
        self.assertEqual(
            idle.settings({}), {k: registry.default(k) for k in idle.SETTINGS}
        )
        self.cfg.update(
            tick_interval_min=0.5,
            loop_heartbeat_min=2,
            tick_timeout_min=1,
            loop_log_max_mb=1,
        )
        conf = dict(
            line.split("=", 1) for line in idle.loop_config(self.cfg, self.store)
        )
        self.assertEqual(
            {
                k: conf[k]
                for k in (
                    "TICK_INTERVAL_S",
                    "LOOP_HEARTBEAT_S",
                    "TICK_TIMEOUT_S",
                    "LOOP_LOG_MAX_BYTES",
                )
            },
            {
                "TICK_INTERVAL_S": "30",
                "LOOP_HEARTBEAT_S": "120",
                "TICK_TIMEOUT_S": "60",
                "LOOP_LOG_MAX_BYTES": str(2**20),
            },
        )
        self.assertEqual(conf["LOOP_LOCK"], str(self.store.dir / "tick_loop.lock"))
        self.cfg["tick_timeout_min"] = -1
        with self.assertRaises(ValueError):
            idle.loop_config(self.cfg, self.store)

    def test_loop_pid_reads_the_lock(self):
        import fcntl
        import os

        self.assertIsNone(idle.loop_pid(self.store))
        lock = self.store.dir / "tick_loop.lock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR)
        try:
            self.assertIsNone(idle.loop_pid(self.store))  # the file without a holder
            fcntl.flock(fd, fcntl.LOCK_EX)
            (self.store.dir / "tick_loop.pid").write_text("4321\n")
            self.assertEqual(idle.loop_pid(self.store), 4321)
        finally:
            os.close(fd)
        self.assertIsNone(idle.loop_pid(self.store))


class TestParts(Base):
    """The registered parts against fake GitHub answers."""

    def read(self, name):
        return idle.PARTS[name].read(self.cfg, self.st, self.store)

    def test_pr_parts_read_the_scope_labels_and_the_shared_labels_without_them(self):
        # config.json's list of shared labels: each counts for the repository's folder
        folder = scope_of(self.cfg).shared["area: x"]
        self.st["scope_cache"] = {
            "3": {"head": f"{3:040d}", "folders": {folder: False}},
            "4": {"head": "e" * 40, "folders": {folder: False}},
        }
        self.assertEqual([r[0] for r in self.read("PRs by label")], [1, 2])
        self.assertEqual(
            [r[0] for r in self.read("PRs by shared label")], [4]
        )  # #3 changes no file in its label's folder at its head; #4's answer is for another head
        q = [x for x in self.gh.queries if "-label:" in x][0]
        self.assertEqual(
            q,
            'repo:o/r is:pr is:open label:"area: x" -label:"scope: a" -label:"scope: b"',
        )
        row = self.read("PRs by label")[0]
        self.assertEqual(
            row,
            [
                1,
                f"{1:040d}",
                "2026-11-06T10:00:00Z",
                False,
                "MERGEABLE",
                "SUCCESS",
                None,
                ["scope: a"],
            ],
        )
        # Two shared labels with two folders: a PR stays until the cache says no for each of its labels' folders.
        self.cfg["scope"]["path_labels"] = {"area: x": "p/x", "area: y": "p/y"}
        self.gh.path_prs = [
            pr(3, labels=("area: x", "area: y")),
            pr(5, labels=("area: y",)),
        ]
        self.st["scope_cache"] = {
            "3": {"head": f"{3:040d}", "folders": {"p/x": False}},
            "5": {"head": f"{5:040d}", "folders": {"p/y": False}},
        }
        self.assertEqual([r[0] for r in self.read("PRs by shared label")], [3])
        self.cfg["scope"]["path_labels"] = []
        self.assertEqual(self.read("PRs by shared label"), [])

    def test_the_develop_part_reads_the_head_and_the_ci_runs_not_settled(self):
        green = {
            "complete": True,
            "runs": {"CI": {"status": "completed", "conclusion": "success"}},
        }
        self.st["develop"]["ci"] = {
            "a" * 40: {"complete": False, "runs": {}},
            "b" * 40: green,
        }
        self.gh.runs["a" * 40] = [
            {
                "id": 7,
                "name": "CI",
                "status": "in_progress",
                "conclusion": None,
                "run_attempt": 1,
            },
            {
                "id": 8,
                "name": "Other",
                "status": "queued",
                "conclusion": None,
                "run_attempt": 1,
            },
        ]
        v = self.read("develop")
        self.assertEqual(
            v, {"head": HEAD, "ci": {"a" * 12: [["CI", 7, 1, False, None]]}}
        )
        self.assertEqual(len(self.gh.rest_paths), 1)  # the settled SHA is not read
        self.assertIn(
            "branch=develop&event=push&head_sha=" + "a" * 40, self.gh.rest_paths[0]
        )
        self.cfg["base_branch"] = "main"
        self.read("develop")
        self.assertIn("branch=main&", self.gh.rest_paths[-1])

    def test_the_develop_issues_part_keeps_the_issues_the_tick_reports(self):
        self.gh.issues = [
            {
                "number": 9,
                "updatedAt": "t1",
                "title": "x breaks",
                "labels": {"nodes": [{"name": guard_labels.RED}, {"name": "scope: a"}]},
            },
            {
                "number": 8,
                "updatedAt": "t2",
                "title": "unrelated",
                "labels": {"nodes": [{"name": guard_labels.STABILITY}]},
            },
        ]
        self.assertEqual(self.read("develop issues"), [[9, "t1"]])
        q = [x for x in self.gh.queries if "is:issue" in x][0]
        self.assertEqual(  # both kinds' labels, as the settings name them
            q,
            f'repo:o/r is:issue is:open label:"{guard_labels.RED}","{guard_labels.STABILITY}"',
        )

    def test_the_priority_requests_part(self):
        self.assertIsNone(
            self.read("priority requests")
        )  # nothing to read: no web host configured
        self.cfg["publish"] = {"html": {"requests": "site/requests.jsonl"}}

        def pull(cfg, dest, run=None):
            pathlib.Path(dest).write_text('{"id": "abcdef1", "pr": 5}\n')

        with mock.patch.object(publish, "pull_requests", pull):
            a = self.read("priority requests")
        self.assertEqual(len(a), 16)
        with mock.patch.object(
            publish, "pull_requests", lambda cfg, dest, run=None: "Broken pipe"
        ):
            self.assertEqual(self.read("priority requests"), "unread")

    def test_the_jobs_part_finds_every_job_the_state_refers_to(self):
        self.st["review_pipeline"] = {
            "pr": 5,
            "worker": {"id": "w1", "pid": 999999, "started": now(), "timeout_min": 60},
        }
        self.st["somewhere"] = {
            "deep": [
                {"job": {"id": "n1", "pid": 999998, "started": now(), "timeout_min": 5}}
            ]
        }
        # A pool draft in flight is one of the jobs: its end makes the next tick full.
        self.st["draft_pool"] = {
            "6": {
                "pr": 6,
                "head": "h",
                "job": {
                    "id": "d6",
                    "pid": 999997,
                    "started": now(),
                    "timeout_min": 110,
                },
            }
        }
        for job_id in ("w1", "n1", "d6"):
            (self.store.dir / "jobs" / job_id).mkdir(parents=True)
        (self.store.dir / "jobs" / "w1" / "rc").write_text("0\n")
        (self.store.dir / "jobs" / "d6" / "rc").write_text("1\n")
        self.assertEqual(
            self.read("jobs"), {"d6": "failed(rc=1)", "n1": "lost", "w1": "done"}
        )

    def test_the_review_files_part_follows_what_the_checks_wait_on(self):
        from rkmon_lib import review

        self.cfg["paths"]["workers"] = str(self.d / "workers")
        self.cfg["paths"]["root"] = str(
            self.d
        )  # the working folder: the worktrees are in it
        self.assertIsNone(self.read("review files"))  # no review in flight
        self.st["review_pipes"] = [
            {"pr": 5, "worker": {"started": now()}, "checks": None}
        ]
        self.assertIsNotNone(review)
        self.assertEqual(self.read("review files"), {"5": [False, False]})
        (self.d / "wt5m").mkdir()
        (self.d / "workers" / "pr5").mkdir(parents=True)
        (self.d / "workers" / "pr5" / "staged").write_text("")
        self.assertEqual(self.read("review files"), {"5": [True, True]})
        self.st["review_pipes"][0]["checks"] = {
            "id": "c5"
        }  # the checks run: nothing to wait on
        self.assertIsNone(self.read("review files"))

    def test_file_parts_change_with_their_file(self):
        before = self.read("ledger")
        self.store.ledger_add("post", 5, "h", "url")
        self.assertNotEqual(self.read("ledger"), before)
        self.assertIsNone(self.read("incidents"))

    def test_config_and_code_parts(self):
        a = self.read("config")
        self.cfg["deployment"] = {"phase": "queue"}
        self.assertNotEqual(self.read("config"), a)
        self.assertEqual(len(self.read("code")), 16)


class TestTimers(Base):
    """Each stage's registered timers on a small state."""

    def due(self):
        return idle.deadlines(self.cfg, self.st, self.store)

    def rec(self, n, **kw):
        r = {
            "n": n,
            "status": "open",
            "draft": False,
            "head": f"{n:040d}",
            "head_seen_at": ts(-5),
            "size": 10,
            "author": "a",
            "ours": None,
        }
        r.update(kw)
        self.st["prs"][str(n)] = r
        return r

    def test_an_empty_state_has_no_timer_but_the_weekly_report(self):
        self.assertEqual(self.due(), {})
        self.cfg["weekly_report"] = {
            "enabled": True,
            "weekday": 0,
            "hour_utc": 14,
            "dir": str(self.d),
        }
        self.assertEqual(set(self.due()), {"weekly report"})

    def test_settle_and_gap(self):
        self.rec(1)  # never reviewed, pushed 5 min ago: settles in 25 min
        self.rec(2, head_seen_at=ts(-90))  # settled
        self.rec(
            3, ours={"head": f"{3:040d}", "at": ts(-200)}
        )  # our review covers its head
        self.rec(
            4, head_seen_at=ts(-90), ours={"head": "f" * 40, "at": ts(-30)}
        )  # a follow-up inside the gap
        self.cfg["followup_min_gap_min"] = 60
        due = self.due()
        self.assertEqual(set(due), {"settle #1", "gap #4"})
        self.assertTrue(ts(24) <= due["settle #1"] <= ts(26), due)
        self.assertTrue(ts(29) <= due["gap #4"] <= ts(31), due)

    def test_merge_queue_guard_review_and_follow_up_timers(self):
        self.st["analysis_wait"] = {"5": ts(-10)}
        self.st["holds"] = {
            "6": {
                "head": f"{6:040d}",
                "status": "held",
                "since": ts(-60),
                "reason": "G5",
            },
            "7": {
                "head": f"{7:040d}",
                "status": "held",
                "since": ts(-60),
                "reason": "G6",
                "stale_told": True,
            },
        }
        self.rec(6, head_seen_at=ts(-90), ours={"head": f"{6:040d}", "at": ts(-100)})
        self.rec(7, head_seen_at=ts(-90), ours={"head": f"{7:040d}", "at": ts(-100)})
        (self.store.dir / "incidents.json").write_text(
            json.dumps([{"id": "INC-1", "status": "open", "start": ts(-30)}])
        )
        self.st["review_pipeline"] = {
            "pr": 8,
            "worker": {"started": ts(-2)},
            "checks": None,
        }
        self.st["jobs"]["heavy"] = {"id": "h", "started": ts(-1)}
        self.st["followup_scans"] = {
            "9": {"at": ts(-12), "tries": 1, "error": "x"},
            "10": {"file": "f"},
        }
        due = self.due()
        self.assertEqual(
            set(due),
            {
                "analysis wait #5",
                "hold stale #6",
                "incident INC-1",
                "worker prefill #8",
                "build's first minutes",
                "follow-up scan #9",
            },
        )
        self.assertTrue(ts(79) <= due["analysis wait #5"] <= ts(81))
        self.assertTrue(ts(89) <= due["incident INC-1"] <= ts(91))
        self.assertLess(
            due["follow-up scan #9"], now()
        )  # a retry: due at the next tick

    def test_a_draft_stage_without_prefill_and_the_scaler_hold(self):
        self.st["review_pipeline"] = {
            "pr": 8,
            "stage_name": "tool-loop",
            "worker": {"started": ts(-2)},
            "checks": None,
        }
        self.assertEqual(self.due(), {})  # a drafter elsewhere holds no heavy job back
        log = self.store.dir / "scaler.jsonl"

        def decision(allow, running, ceiling, at):
            with log.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "at": at,
                            "pr": 9,
                            "allow": allow,
                            "reason": "r",
                            "inputs": {"running": running, "ceiling": ceiling},
                        }
                    )
                    + "\n"
                )

        decision(
            False, 1, 3, ts(-2)
        )  # held for its budget: it may start 5 min after the decision
        self.assertEqual(set(self.due()), {"held draft"})
        self.assertTrue(ts(2) <= self.due()["held draft"] <= ts(4))
        decision(
            False, 3, 3, ts(-1)
        )  # held at the ceiling: a draft job's end frees it (the jobs part)
        self.assertEqual(self.due(), {})
        decision(True, 1, 3, ts(0))
        self.assertEqual(self.due(), {})

    def test_scheduled_reads_and_the_board(self):
        self.st.update(
            board={"fp": "x", "shown": ts(-10)},
            merges_daily={"at": ts(-30)},
            ci_times={"at": ts(-30)},
            codeowners={"at": ts(-60)},
            required_checks={"at": ts(-60)},
            conflicts={"live": {"11": {"logged": ts(-60)}}},
            overlaps={
                "at": ts(-1),
                "pending": {"12": "waits for the next tick"},
                "waiting": {"13": {"at": ts(-10)}},
            },
            merge_plan={"held_back": {"14": {"since": ts(-60)}}},
        )
        self.assertEqual(
            set(self.due()),
            {
                "board",
                "merges per day",
                "CI times",
                "code owners",
                "required checks",
                "conflict log #11",
                "overlap reads left",
                "overlap retry #13",
                "starvation cap #14",
            },
        )

    def test_the_weekly_report_falls_due_once_a_week(self):
        self.cfg["weekly_report"] = {
            "enabled": True,
            "weekday": 0,
            "hour_utc": 14,
            "dir": str(self.d),
        }
        t = dt.datetime.now(dt.timezone.utc)
        monday = dt.datetime.combine(
            t.date() - dt.timedelta(days=t.weekday()), dt.time(), dt.timezone.utc
        )
        this_week = (monday + dt.timedelta(hours=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertEqual(self.due()["weekly report"], this_week)
        self.st["weekly_done"] = rkmon.stats.week_bounds(None, t)[2]
        self.assertEqual(self.due()["weekly report"], idle.plus(this_week, 7 * 24 * 60))


@contextlib.contextmanager
def cli(cfg, store):
    out = io.StringIO()
    with mock.patch.object(rkmon, "load_config", lambda: cfg), mock.patch.object(
        rkmon, "Store", lambda: store
    ), mock.patch.object(
        rkmon.ghlogins, "tick_lines", lambda cfg, store: []
    ), contextlib.redirect_stdout(
        out
    ):  # the logins' line: tests/test_gh_split.py
        yield out


class TestTick(Base):
    """`rk tick`: the idle path, the full path and the dry run, through the CLI."""

    def setUp(self):
        super().setUp()
        self.st["prs"] = {
            "1": {"n": 1, "status": "open"},
            "2": {"n": 2, "status": "merged"},
        }
        self.store.save(self.st)
        self.md, self.html = self.d / "queue.md", self.d / "page.html"
        t = "2026-11-06T10:00:00Z"
        self.md.write_text(f"# rocKE merge queue — {t}\n\n{report.checked_line(t)}\n")
        page = "<span class='meta'>updated 10:00Z · checked <span id='checked'>10:00Z</span> · ticks</span>"
        self.html.write_text(page)
        (self.store.dir / "published.html").write_text(page)

    def tick(self, *args, decision=None, full_rc=0):
        pushed = []
        full = mock.Mock(return_value=full_rc)
        patches = [
            mock.patch.object(rkmon, "full_tick", full),
            mock.patch.object(
                publish, "html", lambda cfg, path, run=None: pushed.append(path)
            ),
        ]
        if decision:
            patches.append(mock.patch.object(idle, "decide", lambda *a, **kw: decision))
        with cli(self.cfg, self.store) as out, contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            rc = rkmon.main(["tick", *args])
        return rc, out.getvalue(), full, pushed

    def test_an_idle_tick_prints_one_line_stamps_the_dashboards_and_records_itself(
        self,
    ):
        d = idle.Decision(True, [], {"a": 1}, {}, now())
        self.st = self.store.load()
        self.st[idle.KEY] = {
            "fp": {"a": 1},
            "full_at": "2026-11-06T09:40:00Z",
            "due": {},
        }
        self.store.save(self.st)
        rc, out, full, pushed = self.tick(decision=d)
        self.assertEqual(rc, 0)
        self.assertEqual(
            out.splitlines(),
            [f"TICK idle: no change since 09:40Z (1 PRs, develop {HEAD[:10]})"],
        )
        full.assert_not_called()
        st = self.store.load()
        self.assertGreaterEqual(st["last_tick"], d.at)
        self.assertEqual([r[1] for r in st[idle.STATS]["recent"]], ["idle"])
        self.assertIn(f"Checked {st['last_tick']};", self.md.read_text())
        self.assertIn(
            "# rocKE merge queue — 2026-11-06T10:00:00Z", self.md.read_text()
        )  # the content's own time
        stamp = f"<span id='checked'>{st['last_tick'][11:16]}Z</span>"
        self.assertIn(stamp, self.html.read_text())
        self.assertIn(stamp, (self.store.dir / "published.html").read_text())
        self.assertEqual(pushed, [str(self.store.dir / "published.html")])

    def test_a_stamp_that_fails_is_one_advisory_line(self):
        d = idle.Decision(True, [], {"a": 1}, {}, now())
        with mock.patch.object(
            publish, "html", lambda cfg, path, run=None: "Broken pipe"
        ):
            with cli(self.cfg, self.store) as out, mock.patch.object(
                idle, "decide", lambda *a, **kw: d
            ), mock.patch.object(rkmon, "full_tick", mock.Mock(return_value=0)):
                rc = rkmon.main(["tick"])
        self.assertEqual(rc, 0)
        self.assertEqual(
            out.getvalue().splitlines()[1:],
            ["(html dashboard not published: Broken pipe)"],
        )

    def test_a_full_tick_runs_the_full_body_and_records_its_fingerprint(self):
        rc, out, full, _ = self.tick(full_rc=3)
        self.assertEqual(rc, 3)
        full.assert_called_once()
        rec = self.store.load()[idle.KEY]
        self.assertEqual(set(rec["fp"]), set(idle.PARTS))
        self.assertEqual(rec["why"], ["no full tick on record"])
        self.assertEqual(
            [r[1] for r in self.store.load()[idle.STATS]["recent"]], ["full"]
        )
        # Nothing changed since: the next tick is idle, and --full forces a full one.
        rc, out, full, _ = self.tick()
        self.assertEqual(out.splitlines()[0][:27], "TICK idle: no change since ")
        full.assert_not_called()
        rc, out, full, _ = self.tick("--full")
        full.assert_called_once()
        # A PR's update makes the next tick full, and says why.
        self.gh.label_prs[0]["updatedAt"] = "2026-11-06T11:00:00Z"
        with cli(self.cfg, self.store) as out:
            rkmon.main(["tick", "--dry-run"])
        self.assertIn("a full tick: changed: PRs by label", out.getvalue())

    def test_the_dry_run_writes_nothing(self):
        before = (self.store.dir / "state.json").read_bytes()
        with cli(self.cfg, self.store) as out:
            rc = rkmon.main(["tick", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertTrue(out.getvalue().startswith("DRY RUN ("), out.getvalue())
        self.assertIn("a full tick: no full tick on record", out.getvalue())
        self.assertEqual((self.store.dir / "state.json").read_bytes(), before)

    def test_off_is_one_line(self):
        self.st = self.store.load()
        self.st["enabled"] = False
        self.store.save(self.st)
        rc, out, full, _ = self.tick()
        self.assertEqual((rc, out.split()[2:4]), (0, ["monitor", "OFF:"]))
        full.assert_not_called()

    def test_a_tick_while_another_runs_is_one_line_the_filter_knows(self):
        """rk tick finds another tick running: one line, in the words the tick filter, the orchestrator and
        tools/tick_loop.py know a skipped tick by (tickfilter.TICK_RUNNING), and nothing else runs.
        """
        with self.store.locked():
            rc, out, full, pushed = self.tick()
        self.assertEqual(rc, 0)
        self.assertRegex(
            out, r"\ATICK \S+ " + re.escape(tickfilter.TICK_RUNNING) + r"; skipped\n\Z"
        )
        self.assertEqual(
            tickfilter.classify(out.strip()),
            (tickfilter.ROUTINE, "a tick was running already; the next one runs"),
        )
        full.assert_not_called()
        self.assertEqual(pushed, [])

    def test_loop_status(self):
        with cli(self.cfg, self.store) as out:
            rc = rkmon.main(["loop-status"])
        self.assertEqual(rc, 1)
        self.assertTrue(
            out.getvalue().startswith("no tick loop running; log "), out.getvalue()
        )


class TestBoard(Base):
    def test_the_monitor_row_shows_the_tick_counts_and_its_fingerprint_ignores_them(
        self,
    ):
        b0 = report.board(
            self.cfg | {"deployment": {"phase": "queue"}}, self.st, self.store, [], []
        )
        self.assertNotIn("ticks in 24 h", b0["md"])
        self.st[idle.STATS] = {
            "recent": [
                [ts(-30), "idle", 4.0],
                [ts(-20), "idle", 6.0],
                [ts(-10), "full", 120.0],
                [ts(-2000), "full", 99.0],
            ]
        }
        b1 = report.board(
            self.cfg | {"deployment": {"phase": "queue"}}, self.st, self.store, [], []
        )
        self.assertIn("ticks in 24 h: 2 idle (avg 5 s), 1 full (avg 120 s)", b1["md"])
        self.assertEqual(b0["fp"], b1["fp"])


if __name__ == "__main__":
    unittest.main()
