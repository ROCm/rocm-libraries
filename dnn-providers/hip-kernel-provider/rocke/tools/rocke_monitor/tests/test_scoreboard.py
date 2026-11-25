# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scoreboard: each counter from the monitor's records, conservative when the records do not single out one PR, the
window filter, no event twice (a tick run again, a backfill run again), the tick's git budget, and the admin-only gate:
the published page has no scoreboard, review spend or today's log, and none of their data.
"""

import datetime as dt
import json
import pathlib
import sys
import tempfile
import time
import unittest
from urllib.parse import unquote
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import report, scoreboard as sb, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

UTC = dt.timezone.utc
KEY = "PRJ"  # an invented ticket project
NOW = dt.datetime.now(UTC).replace(microsecond=0)


def ago(minutes: float) -> str:
    return (NOW - dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(minutes_ago: float) -> int:
    return int((NOW - dt.timedelta(minutes=minutes_ago)).timestamp())


def rec(n: int, author: str, status: str = "open") -> dict:
    return {
        "n": n,
        "author": author,
        "status": status,
        "base": "develop",
        "head": f"{n:040d}",
        "draft": False,
        "title": f"t{n}",
    }


class FakeGit:
    """git as the rules ask it: the develop window's commits with their files, a merge base, the commits since it."""

    def __init__(self, window=(), since_base=(), delay: float = 0.0):
        self.window, self.since_base, self.delay, self.calls = (
            list(window),
            list(since_base),
            delay,
            [],
        )

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        time.sleep(self.delay)
        if "merge-base" in args:
            return "b" * 40 + "\n"
        if "--name-only" in args:  # the window: (minutes ago, PR or None, files)
            return "".join(
                f"\x01{epoch(m)} subject"
                + (f" (#{n})" if n else "")
                + "\n"
                + "".join(f + "\n" for f in fs)
                + "\n"
                for m, n, fs in self.window
            )
        return "".join(
            f"\x01{epoch(m)} subject" + (f" (#{n})" if n else "") + "\n"
            for m, n in self.since_base
        )


class Base(unittest.TestCase):
    CFG = {
        "repo": "o/r",
        "operator": {"github_login": "me"},
        "deployment": {"phase": "queue"},
        "merge_authority": "arm-cleared",
        "paths": {"repo": "/nowhere"},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.tmp.name) / "state")
        self.st = empty_state()
        self.st["prs"] = {
            "1111": rec(1111, "ann", "merged"),
            "2222": rec(2222, "bob"),
            "2345": rec(2345, "ann"),
            "3333": rec(3333, "cid", "merged"),
            "4444": rec(4444, "dee"),
        }
        # #1111 (ann) landed 60 min ago; #3333 (cid) landed 2 days ago.
        self.store.append_jsonl(
            "events.jsonl",
            [
                {
                    "kind": "MERGED",
                    "pr": 1111,
                    "merged_at": ago(60),
                    "merge_sha": "a" * 10,
                    "by": "ann",
                    "ts": ago(58),
                },
                {
                    "kind": "MERGED",
                    "pr": 3333,
                    "merged_at": ago(2880),
                    "merge_sha": "c" * 10,
                    "by": "cid",
                    "ts": ago(2878),
                },
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def derive(self, git=None, days: float = 8, snaps=None) -> list[dict]:
        R = sb.Records(self.CFG, self.st, self.store, cache={}, git=git or FakeGit())
        return sb.derive(
            R,
            NOW - dt.timedelta(days=days),
            dt.datetime.now(UTC) + dt.timedelta(minutes=1),
            snaps or {},
        )

    def kinds(self, events: list[dict], kind: str) -> list[tuple]:
        return sorted(
            (e["author"], e["pr"], e.get("other_pr"))
            for e in events
            if e["kind"] == kind
        )

    def conflict(self, pr: int, minutes_ago: float, files=("a.py",)) -> None:
        self.store.append_jsonl(
            "events.jsonl", [{"kind": "CONFLICT", "pr": pr, "ts": ago(minutes_ago)}]
        )
        self.store.append_jsonl(
            "conflicts.jsonl",
            [
                {
                    "ts": ago(minutes_ago - 1),
                    "source": "live",
                    "pr": pr,
                    "head": f"{pr:012d}",
                    "develop": "d" * 12,
                    "files": list(files),
                }
            ],
        )


class TestCounters(Base):
    def test_merged_counts_each_landing_once_for_its_author(self):
        evs = self.derive()
        self.assertEqual(
            self.kinds(evs, "merged"), [("ann", 1111, None), ("cid", 3333, None)]
        )

    def test_a_pr_turning_conflicting_after_a_landing_on_its_files_is_a_rebase_forced(
        self,
    ):
        self.conflict(2222, 30)
        git = FakeGit(window=[(60, 1111, ["a.py", "b.py"])])
        self.assertEqual(
            self.kinds(self.derive(git), "rebase_forced"), [("ann", 1111, 2222)]
        )

    def test_the_limits_are_settings_with_the_old_defaults(self):
        self.assertEqual(
            sb.settings({}),
            {
                "lookback_days": 8,
                "conflict_window_h": 3,
                "ahead_h": 12,
                "git_budget_s": 0.4,
            },
        )
        self.conflict(2222, 30)
        git = FakeGit(window=[(60, 1111, ["a.py"])])
        cfg = self.CFG
        try:  # a conflict window shorter than the 30 minutes from the landing to the conflict: not counted
            self.CFG = dict(cfg, scoreboard={"conflict_window_h": 0.25})
            self.assertEqual(self.kinds(self.derive(git), "rebase_forced"), [])
            # a tick looking back less than the 2 days since #3333 landed records only #1111
            self.assertEqual(
                sb.tick(
                    dict(cfg, scoreboard={"lookback_days": 1}),
                    self.st,
                    self.store,
                    git=git,
                ),
                [],
            )
            self.assertEqual(
                sorted(e["pr"] for e in sb.load(self.store) if e["kind"] == "merged"),
                [1111],
            )
        finally:
            self.CFG = cfg

    def test_rebase_forced_is_not_counted_when_unsure(self):
        cases = {
            "two develop commits changed its files": FakeGit(
                window=[(60, 1111, ["a.py"]), (50, 9999, ["a.py"])]
            ),
            "a commit that names no PR changed them": FakeGit(
                window=[(60, 1111, ["a.py"]), (50, None, ["a.py"])]
            ),
            "no commit in the window changed them": FakeGit(
                window=[(60, 1111, ["other.py"])]
            ),
        }
        self.conflict(2222, 30)
        for why, git in cases.items():
            with self.subTest(why):
                self.assertEqual(self.kinds(self.derive(git), "rebase_forced"), [])

    def test_no_rebase_forced_on_the_same_author_nor_after_its_own_push(self):
        self.conflict(2345, 30)  # ann's own PR
        self.conflict(4444, 30)
        self.store.append_jsonl(
            "events.jsonl",
            [{"kind": "PUSH", "pr": 4444, "old": "x", "new": "y", "ts": ago(40)}],
        )
        git = FakeGit(window=[(60, 1111, ["a.py"])])
        self.assertEqual(self.kinds(self.derive(git), "rebase_forced"), [])

    def test_the_plan_forced_rebase_list_counts_without_a_file_record(self):
        self.store.append_jsonl(
            "events.jsonl", [{"kind": "CONFLICT", "pr": 2222, "ts": ago(30)}]
        )
        snaps = {"1111": {"at": ago(58), "forces": [2222], "ahead": ""}}
        self.assertEqual(
            self.kinds(self.derive(snaps=snaps), "rebase_forced"), [("ann", 1111, 2222)]
        )

    def test_a_develop_merge_resolving_a_conflict_one_landing_made(self):
        self.store.append_jsonl(
            "conflicts.jsonl",
            [
                {
                    "ts": ago(20),
                    "source": "resolved",
                    "pr": 2222,
                    "merge": "e" * 12,
                    "parent": "f" * 12,
                    "develop": "d" * 12,
                    "files": ["a.py"],
                }
            ],
        )
        self.assertEqual(
            self.kinds(self.derive(FakeGit(since_base=[(60, 1111)])), "rebase_forced"),
            [("ann", 1111, 2222)],
        )
        self.assertEqual(
            self.kinds(
                self.derive(FakeGit(since_base=[(60, 1111), (70, 3333)])),
                "rebase_forced",
            ),
            [],
        )

    def test_re_reviews_caused(self):
        def post(pr, minutes_ago, head, note="followup"):
            url = f"u-{pr}-{head}"
            self.store.append_jsonl(
                "actions.jsonl",
                [
                    {
                        "ts": ago(minutes_ago),
                        "kind": "post",
                        "pr": pr,
                        "head": head,
                        "ref": url,
                        "note": note,
                    }
                ],
            )
            return url

        def title(url, pr, minutes_ago, text):
            self.store.append_jsonl(
                "daylog.jsonl",
                [
                    {
                        "ts": ago(minutes_ago),
                        "kind": "review",
                        "pr": pr,
                        "url": url,
                        "text": text,
                        "verdict": "🟠",
                    }
                ],
            )

        # A round whose title names the landing as what to port.
        u = post(2222, 30, "h1" * 20)
        title(u, 2222, 30, "follow-up 🟠 Follow-up: 1111 landed; port it · N3 🟠")
        # A first round names it too: a first round is no re-review.
        u = post(4444, 30, "h2" * 20, note="first_round")
        title(u, 4444, 30, "first round 🟠 First-round: rebase over 1111 · N1 🟠")
        # A round naming two landings: neither.
        self.st["prs"]["3333"]["status"] = "merged"
        u = post(2345, 20, "h3" * 20)
        title(u, 2345, 20, "follow-up 🟠 Follow-up: ports for 1111 and 3333 · N2 🟠")
        evs = self.derive()
        self.assertEqual(self.kinds(evs, "rereview_caused"), [("ann", 1111, 2222)])

    def test_a_merge_only_round_resolving_the_landing_conflict_is_a_re_review(self):
        self.store.append_jsonl(
            "conflicts.jsonl",
            [
                {
                    "ts": ago(20),
                    "source": "resolved",
                    "pr": 4444,
                    "merge": "e" * 12,
                    "parent": "f" * 12,
                    "develop": "d" * 12,
                    "files": ["a.py"],
                }
            ],
        )
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(500),
                    "kind": "post",
                    "pr": 4444,
                    "head": "f" * 40,
                    "ref": "u1",
                    "note": "followup",
                },
                {
                    "ts": ago(10),
                    "kind": "post",
                    "pr": 4444,
                    "head": "e" * 40,
                    "ref": "u2",
                    "note": "followup",
                },
            ],
        )
        evs = self.derive(FakeGit(since_base=[(60, 1111)]))
        self.assertEqual(self.kinds(evs, "rereview_caused"), [("ann", 1111, 4444)])

    def test_the_first_follow_up_after_the_overlap_owner_landed(self):
        (self.store.dir / "overlaps_decided.json").write_text(
            json.dumps({"grp": {"owner": 1111, "members": [1111, 4444]}})
        )
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(900),
                    "kind": "notice-overlap",
                    "pr": 4444,
                    "head": "grp",
                    "ref": "n",
                    "note": "",
                },
                {
                    "ts": ago(30),
                    "kind": "post",
                    "pr": 4444,
                    "head": "1" * 40,
                    "ref": "u1",
                    "note": "followup",
                },
                {
                    "ts": ago(10),
                    "kind": "post",
                    "pr": 4444,
                    "head": "2" * 40,
                    "ref": "u2",
                    "note": "followup",
                },
            ],
        )
        self.store.append_jsonl(
            "daylog.jsonl",
            [
                {
                    "ts": ago(30),
                    "kind": "review",
                    "pr": 4444,
                    "url": "u1",
                    "text": "follow-up 🟡 Follow-up: land-order port",
                },
                {
                    "ts": ago(10),
                    "kind": "review",
                    "pr": 4444,
                    "url": "u2",
                    "text": "follow-up 🟡 Follow-up: port again",
                },
            ],
        )
        evs = self.derive()
        self.assertEqual(
            [
                (e["pr"], e["other_pr"], e["ref"])
                for e in evs
                if e["kind"] == "rereview_caused"
            ],
            [(1111, 4444, "u1")],
        )

    def test_develop_breaks_need_exactly_one_named_landing(self):
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(20),
                    "kind": "issue",
                    "pr": 5555,
                    "head": "-",
                    "ref": "x",
                    "note": "silent-break: [rocke] test_x fails on develop since #1111",
                },
                {
                    "ts": ago(20),
                    "kind": "issue",
                    "pr": 6666,
                    "head": "-",
                    "ref": "x",
                    "note": "red-develop: [rocke] shards time out on develop",
                },  # no culprit named
                {
                    "ts": ago(20),
                    "kind": "issue",
                    "pr": 7777,
                    "head": "-",
                    "ref": "x",
                    "note": "red-develop: [rocke] fails since #12345",
                },
            ],
        )  # not an in-scope landing
        (self.store.dir / "incidents.json").write_text(
            json.dumps([{"id": "I", "issue": 6666, "culprit": 3333}])
        )
        evs = self.derive()
        self.assertEqual(
            sorted((e["pr"], e["ref"]) for e in evs if e["kind"] == "develop_break"),
            [(1111, "#5555"), (3333, "#6666")],
        )

    def test_landed_ahead_of_plan(self):
        snaps = {"1111": {"at": ago(58), "forces": [], "ahead": "AUTOMERGE_JUMP"}}
        self.assertEqual(
            self.kinds(self.derive(snaps=snaps), "landed_ahead"), [("ann", 1111, None)]
        )
        R = sb.Records(self.CFG, self.st, self.store, cache={}, git=FakeGit())
        R.loop_alerts = [
            (ago(2900), 3333, "PRIORITY_READY_EARLY"),
            (ago(5000), 1111, "AUTOMERGE_JUMP"),
        ]
        evs = sb.derive(R, NOW - dt.timedelta(days=8), NOW, {})
        self.assertEqual(
            self.kinds(evs, "landed_ahead"), [("cid", 3333, None)]
        )  # #1111's alert was days before

    def test_the_landing_snapshot_reads_the_plan_the_tick_saw(self):
        mp = {
            "nodes": {
                "1111": {
                    "wave": 2,
                    "forces": [2222],
                    "conflicts": [2345],
                    "rebase_ahead": [],
                },
                "2345": {"wave": 1, "conflicts": [1111]},
                "202": {"wave": 3, "conflicts": [1111]},
            },
            "automerge": {},
            "ready_early": {},
        }
        st = {"merge_plan": mp}
        sb.note_landings(st, [{"kind": "MERGED", "pr": 1111}])
        self.assertEqual(st["scoreboard"]["landings"]["1111"]["forces"], [2222])
        self.assertEqual(
            st["scoreboard"]["landings"]["1111"]["ahead"], "against the plan's order"
        )
        mp["nodes"]["1111"]["conflicts"] = [
            202
        ]  # only a later wave rebases: the plan's order
        sb.note_landings(st, [{"kind": "MERGED", "pr": 1111}])
        self.assertEqual(st["scoreboard"]["landings"]["1111"]["ahead"], "")
        mp["ready_early"] = {"1111": {}}
        sb.note_landings(st, [{"kind": "MERGED", "pr": 1111}])
        self.assertEqual(
            st["scoreboard"]["landings"]["1111"]["ahead"], "PRIORITY_READY_EARLY"
        )

    def test_merged_with_open_review_items(self):
        self.st["merged_over"] = {
            "1111": {
                "kind": "MERGED_OVER_RED",
                "at": ago(50),
                "by": "x",
                "head": "h",
                "why": "w",
            }
        }
        R = sb.Records(self.CFG, self.st, self.store, cache={}, git=FakeGit())
        R.journal = [(ago(2800), 3333, "HELD_MERGED")]
        evs = sb.derive(R, NOW - dt.timedelta(days=8), NOW, {})
        self.assertEqual(
            sorted(
                (e["pr"], e["ref"]) for e in evs if e["kind"] == "merged_open_items"
            ),
            [(1111, "MERGED_OVER_RED"), (3333, "HELD_MERGED")],
        )

    def test_follow_up_tickets(self):
        d = self.store.dir / "followups"
        d.mkdir()
        (d / "tickets_1111.md").write_text(
            f"# Follow-up tickets: PR 1111\n\n- {KEY}-1: N3, a gap\n"
            f"- {KEY}-2 (existing, comment 5): only commented\n- N5: nit, no ticket\n"
        )
        (d / "tickets_2222.md").write_text(
            f"- {KEY}-3: an open PR's item\n"
        )  # not merged: not counted
        self.store.append_jsonl(
            "daylog.jsonl",
            [
                {
                    "ts": ago(30),
                    "kind": "ticket",
                    "pr": 3333,
                    "text": f"{KEY}-4 follow-up from merged 3333: N1, assigned cid",
                }
            ],
        )
        evs = self.derive()
        self.assertEqual(
            sorted((e["pr"], e["ref"]) for e in evs if e["kind"] == "followup"),
            [(1111, f"{KEY}-1"), (3333, f"{KEY}-4")],
        )

    def test_the_backfill_reads_the_journal_and_the_tick_log(self):
        self.store.journal_path.write_text(
            f"{ago(2800)} queue    #3333 merged by cid over our 🟠 review at 1234567890 is open\n"
            f"{ago(50)} queue    #1111 merged while held, by ann, at the held head 1234567890\n"
        )
        (self.store.dir / "tick_loop.log").write_text(
            f"TICK {ago(70)} monitor ON\n!! AUTOMERGE_JUMP #1111: auto-merge on (ann). It would land ahead\n"
        )
        since = NOW - dt.timedelta(days=8)
        self.assertEqual(
            sb.read_journal(self.store, since),
            [(ago(2800), 3333, "MERGED_OVER_RED"), (ago(50), 1111, "HELD_MERGED")],
        )
        self.assertEqual(
            sb.read_loop_log(self.store, since), [(ago(70), 1111, "AUTOMERGE_JUMP")]
        )


class TestNoDuplicates(Base):
    def test_a_tick_run_again_adds_nothing(self):
        self.conflict(2222, 30)
        git = FakeGit(window=[(60, 1111, ["a.py"])])
        self.assertEqual(sb.tick(self.CFG, self.st, self.store, git=git), [])
        first = sb.load(self.store)
        self.assertEqual(
            sorted(e["kind"] for e in first), ["merged", "merged", "rebase_forced"]
        )
        calls = len(git.calls)
        self.assertEqual(sb.tick(self.CFG, self.st, self.store, git=git), [])
        self.assertEqual(sb.load(self.store), first)
        self.assertEqual(
            len(git.calls), calls
        )  # its git answers are kept: the second tick asks git nothing

    def test_the_backfill_is_idempotent_and_shares_the_tick_events(self):
        self.conflict(2222, 30)
        with mock.patch.object(
            sb.Records,
            "_run_git",
            lambda self, *a: FakeGit(window=[(60, 1111, ["a.py"])])(*a),
        ):
            res = sb.backfill(self.CFG, self.st, self.store, 21, github=False)
            new = sb.record(self.store, self.st, res)
            self.assertEqual(len(new), 3)
            again = sb.record(
                self.store,
                self.st,
                sb.backfill(self.CFG, self.st, self.store, 21, github=False),
            )
            self.assertEqual(again, [])
            self.assertEqual(sb.tick(self.CFG, self.st, self.store), [])
        self.assertEqual(len(sb.load(self.store)), 3)
        lines = sb.backfill_lines(self.store, res, [], dry=True)
        self.assertIn("dry run: nothing written", lines[0])
        self.assertTrue(any("Rebases forced" in x and "1 found" in x for x in lines))

    def test_the_tick_spends_at_most_its_git_budget_and_the_next_tick_goes_on(self):
        for i, p in enumerate((2222, 4444)):
            self.store.append_jsonl(
                "events.jsonl", [{"kind": "CONFLICT", "pr": p, "ts": ago(30 + i)}]
            )
            self.store.append_jsonl(
                "conflicts.jsonl",
                [
                    {
                        "ts": ago(29 + i),
                        "source": "live",
                        "pr": p,
                        "head": "x" * 12,
                        "develop": f"{i}" * 12,
                        "files": ["a.py"],
                    }
                ],
            )
        git = FakeGit(
            window=[(60, 1111, ["a.py"])], delay=sb.DEFAULTS["git_budget_s"] + 0.05
        )
        sb.tick(self.CFG, self.st, self.store, git=git)
        self.assertEqual(
            len(git.calls), 1
        )  # past the budget: the second question waits
        self.assertEqual(self.st["scoreboard"]["last"]["deferred"], 1)
        sb.tick(self.CFG, self.st, self.store, git=git)
        self.assertEqual(
            sorted(
                e["other_pr"]
                for e in sb.load(self.store)
                if e["kind"] == "rebase_forced"
            ),
            [2222, 4444],
        )


class TestTable(unittest.TestCase):
    EVENTS = [
        {
            "kind": "merged",
            "ts": ago(60),
            "author": "ann",
            "pr": 1,
            "other_pr": None,
            "ref": "",
        },
        {
            "kind": "merged",
            "ts": ago(60 * 24 * 10),
            "author": "bob",
            "pr": 2,
            "other_pr": None,
            "ref": "",
        },
        {
            "kind": "rebase_forced",
            "ts": ago(60 * 24 * 10),
            "author": "bob",
            "pr": 2,
            "other_pr": 1,
            "ref": "",
        },
        {
            "kind": "rereview_caused",
            "ts": ago(60 * 24 * 40),
            "author": "bob",
            "pr": 2,
            "other_pr": 1,
            "ref": "u",
        },
        {
            "kind": "develop_break",
            "ts": ago(60 * 24 * 40),
            "author": "cid",
            "pr": 3,
            "other_pr": None,
            "ref": "#9",
        },
        {
            "kind": "followup",
            "ts": ago(60 * 24 * 100),
            "author": "dee",
            "pr": 4,
            "other_pr": None,
            "ref": f"{KEY}-1",
        },
    ]

    def test_window_filtering(self):
        for days, authors, rework in (
            (7, ["ann"], 0),
            (30, ["bob", "ann"], 1),
            (90, ["bob", "ann", "cid"], 2),
        ):
            with self.subTest(days=days):
                s = sb.summary(self.EVENTS, days, NOW)
                self.assertEqual(
                    [r["author"] for r in s["rows"]], authors
                )  # Rework for others first
                self.assertEqual(s["totals"].get("rework", 0), rework)
                self.assertEqual(
                    sum(b["merged"] for b in s["trend"]), s["totals"].get("merged", 0)
                )
        self.assertEqual(len(sb.summary(self.EVENTS, 7, NOW)["trend"]), 7)  # per day
        self.assertGreaterEqual(
            len(sb.summary(self.EVENTS, 90, NOW)["trend"]), 13
        )  # per ISO week

    def test_the_trend_starts_where_the_record_does(self):
        recent = [
            e for e in self.EVENTS if e["kind"] == "merged"
        ]  # the oldest is 10 days old
        s = sb.summary(recent, 90, NOW)
        first = (NOW - dt.timedelta(days=10)).date()
        self.assertEqual(s["from"][:10], first.isoformat())
        self.assertEqual(
            s["trend"][0]["key"],
            (first - dt.timedelta(days=first.weekday())).isoformat(),
        )
        self.assertIn("recorded from", sb.lines(recent, 90, NOW)[0])
        self.assertEqual(
            sb.summary(self.EVENTS, 7, NOW)["start"][11:], "00:00:00Z"
        )  # whole UTC days

    def test_the_terminal_table(self):
        out = sb.lines(self.EVENTS, 30, NOW)
        self.assertIn("sorted by Rework for others", out[0])
        self.assertTrue(out[3].startswith("bob") and "1 (1 + 0)" in out[3])
        self.assertTrue(out[-2].startswith("Team"))
        self.assertEqual(
            sb.lines([], 30, NOW)[1], "No activity recorded in the window."
        )

    def test_the_card_renders_fast(self):
        """A long record (12,000 events) renders in the time of a few reads of it, whatever the machine and its load:
        the card's CPU time is under 30 times that of reading every event's time once (it takes 10 to 14), each the
        best of three, timed in turn. CPU time leaves out the time the test waits for a processor, and the read slows
        as the card does on a processor that is shared or slower."""
        many = [
            dict(e, pr=i, ts=ago(i % (60 * 24 * 90)))
            for i, e in enumerate(self.EVENTS * 2000)
        ]
        card, read = [], []
        for _ in range(3):
            t0 = time.thread_time()
            html = webdash.scoreboard_card({"repo": "o/r"}, many, NOW)
            t1 = time.thread_time()
            for e in many:
                dt.datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ")
            card.append(t1 - t0)
            read.append(time.thread_time() - t1)
        self.assertLess(min(card), 30 * min(read))
        self.assertEqual(html.count("class='sb-panel'"), 3)
        self.assertIn("aria-pressed='true'>30 days", html)


WORDS = ("disruptive", "offender", "blame", " bad ")


class TestAdminOnly(unittest.TestCase):
    """The scoreboard, review spend and today's log are on the admin page (the LOCAL file) only, today's log last."""

    def page(self, variant: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {
                "deployment": {"phase": "queue"},
                "merge_authority": "arm-cleared",
                "operator": {"github_login": "me"},
                "repo": "o/r",
                "paths": {"html": f"{d}/x.html", "daily_dir": d},
            }
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            store.append_jsonl(
                sb.FILE,
                [
                    {
                        "kind": "merged",
                        "ts": now(),
                        "author": "zed-scoreboard-author",
                        "pr": 7,
                        "other_pr": None,
                        "ref": "",
                    },
                    {
                        "kind": "rebase_forced",
                        "ts": now(),
                        "author": "zed-scoreboard-author",
                        "pr": 7,
                        "other_pr": 8,
                        "ref": "conflict",
                    },
                ],
            )
            store.append_jsonl(
                "daylog.jsonl",
                [
                    {"ts": now(), "kind": "hold", "pr": None, "text": "LOGLEAK_4711"},
                    {
                        "ts": now(),
                        "kind": "save",
                        "save": "SAVELEAK",
                        "pr": None,
                        "text": "an issue avoided",
                    },
                ],
            )
            row = {
                "reviews": 3,
                "total": 12.5,
                "mean": 4.1,
                "median": 4.0,
                "p90": 5.0,
                "max": 6.0,
            }
            (store.dir / "spend_summary.json").write_text(
                json.dumps(
                    {
                        "today": dict(row, day=now()[:10]),
                        "week": dict(row, week="2026-W46"),
                        "days": [dict(row, day="x")],
                        "outliers": [
                            {"pr": 7, "round": "fr", "usd": 6.0, "driver": "SPENDLEAK"}
                        ],
                        "at": now(),
                    }
                )
            )
            board = report.board(cfg, st, store, [], [])
            return webdash.render(
                cfg,
                st,
                store,
                [],
                [],
                board,
                variant,
            )

    def test_the_published_page_has_none_of_the_admin_cards_nor_their_data(self):
        pub = self.page(webdash.PUBLISHED)
        for absent in (
            "id='scoreboard'",
            "id='spend'",
            "id='log'",
            "href='#scoreboard'",
            "href='#spend'",
            "href='#log'",
            "Scoreboard",
            "Review spend",
            "Today's log",
            "zed-scoreboard-author",
            "LOGLEAK_4711",
            "SPENDLEAK",
            "Rework for others",
            "rkmon-score",
            "d-notready",
            "class='sb-win'",
            "Issues avoided",
            "SAVELEAK",
        ):
            self.assertNotIn(absent, pub)

    def test_the_admin_page_has_them_with_todays_log_last(self):
        page = self.page(webdash.LOCAL)
        for present in (
            "zed-scoreboard-author",
            "LOGLEAK_4711",
            "SPENDLEAK",
            "href='#scoreboard'",
            "rkmon-score",
            "Issues avoided (7 d)",
            "SAVELEAK 1",
        ):
            self.assertIn(present, page)
        order = [page.index(f"<section id='{c}'") for c in webdash.ADMIN_CARDS]
        self.assertEqual(order, sorted(order))
        last = page.rindex("<section")
        self.assertTrue(page[last:].startswith("<section id='log' class='card'>"))
        self.assertGreater(
            last,
            max(
                page.index(f"<section id='{s}'")
                for s in ("needs", "develop", "insights")
            ),
        )

    def test_the_neutral_words(self):
        """The card and the module speak of rework for others, develop breaks, landed ahead of plan, merged with open
        review items and follow-ups; never of people causing trouble."""
        card = webdash.scoreboard_card({"repo": "o/r"}, TestTable.EVENTS, NOW).lower()
        module = pathlib.Path(sb.__file__).read_text(encoding="utf-8").lower()
        for w in WORDS:
            self.assertNotIn(w, card + module)
        for w in ("issue", "culprit"):
            self.assertNotIn(w, card)
        for term in (
            "rework for others",
            "develop breaks",
            "landed ahead of plan",
            "merged with open review items",
            "follow-ups",
        ):
            self.assertIn(term, card)


class TestGitHubMerges(unittest.TestCase):
    def test_a_shared_label_s_merge_counts_only_when_it_changes_files_in_its_folder(
        self,
    ):
        """The merges the records lack: a scope label's always; a shared label's when its merge commit changes files in
        that label's folder, two labels with two folders."""
        cfg = {
            "repo": "o/r",
            "scope": {
                "labels": ["s"],
                "path_labels": {"area: a": "projects/a", "area: b": "projects/b"},
            },
        }
        found = {"s": [1], "area: a": [7, 8], "area: b": [7, 9]}
        files = {
            1: [],
            7: ["projects/b/x.py"],
            8: ["projects/a/y.py"],
            9: ["projects/a/z.py"],
        }

        def rest(path, jq=None, paginate=False, timeout=120):
            if path.startswith("search/"):
                label = unquote(path).split('label:"', 1)[1].split('"', 1)[0]
                return [
                    {"n": n, "author": "dev", "merged_at": ago(60)}
                    for n in found[label]
                ]
            return {"s": f"{int(path.rsplit('/', 1)[1]):040d}"}

        def git(*args):
            sha, folder = args[3], args[5]
            return "".join(
                f + "\n" for f in files[int(sha)] if f.startswith(folder + "/")
            )

        class Records:
            _git = staticmethod(git)

            @staticmethod
            def landings():
                return {}

        with mock.patch.object(sb.gh, "rest", rest):
            out, notes = sb.github_merges(cfg, Records, NOW - dt.timedelta(days=7))
        self.assertEqual((sorted(out), notes), ([1, 7, 8], []))


if __name__ == "__main__":
    unittest.main()
