# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The user's status board: rows, the "Needs you" list, a fingerprint that ignores mere aging, and review throughput."""

import datetime as dt
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import report  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

CFG = {
    "features": {
        "merge_queue": {"auto_merge": "enabled"}
    },  # in prototype: the guard alerts
    "revert_offer_after_min": 120,
    "operator": {"github_login": "me"},
    "repo": "ROCm/r",
    "codeowner_teams": {"platform": ["plat"]},
}


def ago(minutes: float) -> str:
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def rec(n, seen_min=60):
    return {
        "n": n,
        "head": (str(n) * 40)[:40],
        "status": "open",
        "draft": False,
        "head_seen_at": ago(seen_min),
    }


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["enabled"] = True
        self.st["develop"]["sha"] = "d" * 40
        self.st["prs"] = {"1": rec(1, 300), "2": rec(2, 30), "3": rec(3)}
        self.plan = [
            {"prio": 2, "kind": "FOLLOW_UP", "pr": 1, "text": "", "critical": False},
            {"prio": 3, "kind": "FIRST_ROUND", "pr": 2, "text": "", "critical": False},
        ]

    def tearDown(self):
        self.d.cleanup()

    def test_rows_and_quiet_needs(self):
        b = report.board(CFG, self.st, self.store, self.plan, [])
        for part in (
            "Monitor",
            "Reviews",
            "Review throughput",
            "Merge queue",
            "Develop guard",
        ):
            self.assertIn(f"| {part} |", b["md"])
        self.assertIn("waiting: 1 follow-ups, 1 first rounds", b["md"])
        self.assertIn("oldest waiting 5.0 h", b["md"])
        self.assertIn("Needs you:** nothing right now.", b["md"])

    def test_needs_merge_stale_hold_and_critical(self):
        self.st["batches"] = [{"members": [3], "status": "validated"}]
        self.st["holds"] = {
            "2": {
                "head": rec(2)["head"],
                "reason": "G5 golden",
                "status": "held",
                "since": ago(1500),
                "stale_told": True,
            }
        }
        plan = self.plan + [
            {
                "prio": 0,
                "kind": "ALERT",
                "pr": None,
                "text": "develop red",
                "critical": True,
            }
        ]
        b = report.board(CFG, self.st, self.store, plan, [])
        self.assertTrue(b["needs"][0].startswith("Merge #3"), b["needs"])
        self.assertTrue(
            any(n.startswith("#2 has been held") for n in b["needs"]), b["needs"]
        )
        self.assertTrue(
            any(n.startswith("Critical: develop red") for n in b["needs"]), b["needs"]
        )
        self.st["cleared"] = [3]
        self.assertFalse(
            any(
                n.startswith("Merge #3")
                for n in report.board(CFG, self.st, self.store, plan, [])["needs"]
            )
        )

    def test_the_merge_line_names_the_clear_command_of_each_pr_not_cleared(self):
        """The Merge line names the command that clears a PR for auto-merge as the operator types it on the host
        (doctor.rk_command), for each PR of the batch not cleared yet."""
        self.st["prs"]["4"] = rec(4)
        self.st["batches"] = [{"members": [3, 4], "status": "validated"}]
        self.st["cleared"] = [4]
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "rkmon"}):
            needs = report.board(CFG, self.st, self.store, self.plan, [])["needs"]
        self.assertIn(
            "Merge #3 → #4: every gate is green on develop dddddddddd. Merge it, or clear it for auto-merge: "
            "`rkmon rk clear 3`.",
            needs,
        )
        self.st["cleared"] = []
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "rkmon"}):
            needs = report.board(CFG, self.st, self.store, self.plan, [])["needs"]
        self.assertIn(
            "Merge #3 → #4: every gate is green on develop dddddddddd. Merge it, or clear it for auto-merge: "
            "`rkmon rk clear 3`, `rkmon rk clear 4`.",
            needs,
        )

    def test_the_priority_and_ticket_lines_name_their_commands_as_typed_on_the_host(
        self,
    ):
        """The Needs-you lines of a pending priority request and of proposed tickets, and the merge plan's row of the
        request, name each command as the operator types it on the host (doctor.rk_command), never rk alone.
        """
        from rkmon_lib import priority, tickets

        self.st[priority.REQUESTS] = {
            "items": [
                {
                    "pr": 3,
                    "status": "pending",
                    "who": "Jane",
                    "why": "blocks the release",
                    "at": ago(5),
                }
            ]
        }
        self.st[tickets.BOOK] = {
            "items": [{"id": "0a1b2c3d", "status": "waiting", "at": ago(5)}]
        }
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "rkmon"}):
            needs = report.board(CFG, self.st, self.store, self.plan, [])["needs"]
            row = "\n".join(report.requests_md(CFG, self.st))
        self.assertIn(
            "Priority request for #3 by Jane: blocks the release. Approve it on the admin dashboard or with "
            "`rkmon rk prioritize 3`, or decline it there or with `rkmon rk priority-decline 3`.",
            needs,
        )
        self.assertIn(
            "1 proposed ticket waits for your OK: approve or decline on the admin dashboard's Tickets card (or "
            "`rkmon rk tickets`, `rkmon rk ticket-file ID`, `rkmon rk ticket-decline ID`).",
            needs,
        )
        self.assertIn(
            "`rkmon rk prioritize 3`; decline there or `rkmon rk priority-decline 3`",
            row,
        )

    def test_an_in_depth_review_waits_on_the_list_for_your_ok(self):
        """A first round over autopilot_max_lines is one Needs-you decision: its size, its agents and their cost, and
        the commands as typed on the host. The team sees its size and agents alone, sent to the admin dashboard: the
        cost stays on the admin side. Its cost moving with the spend card does not re-show the board; approved, the
        line goes."""
        from rkmon_lib import review, split

        # Approved, the review is started once its head settles (settle_min).
        cfg = dict(CFG, settle_min=10)
        self.st["prs"]["2"].update(size=80000, author="someone")

        def board(each: float) -> dict:
            with mock.patch.dict(
                os.environ, {"RKMON_WRAPPER": "rkmon"}
            ), mock.patch.object(split, "agent_cost", return_value=each):
                return report.board(cfg, self.st, self.store, self.plan, [])

        line = (
            "#2 (someone): an in-depth review of 80,000 changed lines needs your OK: about 11 agents, about $88. "
            "Approve it on the admin dashboard or with `rkmon rk review-approve 2`; decline it there or with "
            "`rkmon rk review-skip 2`."
        )
        a = board(8.0)
        self.assertIn(line, a["needs"])
        self.assertIn(
            {
                "kind": report.IN_DEPTH,
                "pr": 2,
                "key": "IN_DEPTH_REVIEW:2:",
                "text": line,
                "team": "#2: an in-depth review of 80,000 changed lines needs your OK: about 11 agents. Approve or "
                "decline it on the admin dashboard.",
            },
            a["asks"],
        )
        b = board(8.4)
        self.assertIn(line.replace("$88", "$92"), b["needs"])
        self.assertEqual(a["fp"], b["fp"])
        self.assertTrue(review.approve(cfg, self.st, self.store, 2)[0])
        c = board(8.0)
        self.assertFalse([t for t in c["needs"] if "in-depth" in t], c["needs"])
        self.assertFalse([x for x in c["asks"] if x["kind"] == report.IN_DEPTH])
        self.assertIn("too large for the autopilot (I start them by hand): #2", c["md"])

    def test_fingerprint_ignores_aging_but_not_stage(self):
        self.st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "follow-up",
                "stage": "worker",
                "since": ago(5),
                "checks": None,
            }
        ]
        a = report.board(CFG, self.st, self.store, self.plan, [])
        self.st["review_pipes"][0]["since"] = ago(25)
        self.assertEqual(
            a["fp"], report.board(CFG, self.st, self.store, self.plan, [])["fp"]
        )
        self.st["review_pipes"][0]["stage"] = "verify"
        self.assertNotEqual(
            a["fp"], report.board(CFG, self.st, self.store, self.plan, [])["fp"]
        )

    def test_fingerprint_ignores_the_age_in_a_needs_line(self):
        """An incident's "Develop has been red 2.3 h" does not re-show the whole board every tick."""
        from rkmon_lib import guard

        guard.open_incident(
            self.store, self.st, "a" * 40, ["develop CI: shard"], "b" * 40, []
        )
        incs = guard.incidents(self.store)
        incs[0]["start"] = ago(150)
        guard.save_incidents(self.store, incs)
        a = report.board(CFG, self.st, self.store, self.plan, [])
        self.assertTrue(any("has been red 2.5 h" in n for n in a["needs"]), a["needs"])
        incs[0]["start"] = ago(240)
        guard.save_incidents(self.store, incs)
        b = report.board(CFG, self.st, self.store, self.plan, [])
        self.assertTrue(any("has been red 4.0 h" in n for n in b["needs"]), b["needs"])
        self.assertEqual(a["fp"], b["fp"])
        incs[0]["issue"] = 10303  # a real change still shows
        guard.save_incidents(self.store, incs)
        self.assertNotEqual(
            a["fp"], report.board(CFG, self.st, self.store, self.plan, [])["fp"]
        )

    def test_an_incident_s_line_names_its_issue_by_the_kind_s_label(self):
        """The revert offer and the aged incident name the incident's develop issue by its kind's label, as the
        settings name it."""
        from rkmon_lib import breakage, guard

        cfg = dict(
            CFG,
            incident_issue={"labels": ["broke-ci"], "stability_label": "unseen-break"},
        )
        self.st["post_merge"] = {
            "at": now(),
            "open": [{"n": 11111, "kind": breakage.RED, "label": "broke-ci"}],
        }
        guard.open_incident(
            self.store, self.st, "a" * 40, ["develop CI: shard"], "b" * 40, []
        )
        for culprit in (None, 7):
            incs = guard.incidents(self.store)
            incs[0].update(start=ago(150), issue=11111, culprit=culprit)
            guard.save_incidents(self.store, incs)
            needs = report.board(cfg, self.st, self.store, self.plan, [])["needs"]
            line = [n for n in needs if n.startswith("Develop has been")]
            self.assertEqual(len(line), 1, needs)
            self.assertIn("; broke-ci issue #11111)", line[0])

    def test_fingerprint_ignores_the_rolling_24h_counts(self):
        """Posts and pushes leaving the 24 h window do not re-show the whole board."""
        from unittest import mock

        base = report.throughput(self.st, self.store, self.plan)
        a = report.board(CFG, self.st, self.store, self.plan, [])
        rolled = dict(
            base,
            posts=base["posts"] + 3,
            equiv=base["equiv"] + 1,
            pushes=base["pushes"] + 5,
            new=base["new"] + 1,
        )
        with mock.patch.object(report, "throughput", return_value=rolled):
            b = report.board(CFG, self.st, self.store, self.plan, [])
        self.assertNotEqual(a["md"], b["md"])  # the row still shows the new counts
        self.assertEqual(a["fp"], b["fp"])
        more = self.plan + [
            {"prio": 3, "kind": "FIRST_ROUND", "pr": 9, "text": "", "critical": False}
        ]
        self.assertNotEqual(
            a["fp"], report.board(CFG, self.st, self.store, more, [])["fp"]
        )  # a new review waiting

    def test_fingerprint_ignores_the_trend_unless_it_says_losing(self):
        """A trend from "flat over 6 h" to "+1 in 6 h" does not re-show the whole board."""
        from unittest import mock

        base = report.throughput(self.st, self.store, self.plan)
        fps = {}
        for trend in (
            "flat over 6 h",
            "+1 in 6 h",
            "-2 in 6 h: winning",
            "+3 in 6 h: LOSING",
        ):
            with mock.patch.object(
                report, "throughput", return_value=dict(base, trend=trend)
            ):
                fps[trend] = report.board(CFG, self.st, self.store, self.plan, [])["fp"]
        self.assertEqual(fps["flat over 6 h"], fps["+1 in 6 h"])
        self.assertEqual(fps["flat over 6 h"], fps["-2 in 6 h: winning"])
        self.assertNotEqual(fps["flat over 6 h"], fps["+3 in 6 h: LOSING"])

    def test_fingerprint_ignores_the_losing_trends_count_in_needs_you(self):
        """A new count in the Needs-you line's trend ("+20 in 6 h: LOSING" -> "+19 ...") does not re-show the board."""
        from unittest import mock

        base = report.throughput(self.st, self.store, self.plan)
        fps, needs = {}, {}
        for delta in (20, 19):
            tp = dict(base, trend=f"+{delta} in 6 h: LOSING", delta=delta, backlog=34)
            with mock.patch.object(report, "throughput", return_value=tp):
                b = report.board(CFG, self.st, self.store, self.plan, [])
            fps[delta] = b["fp"]
            needs[delta] = [n for n in b["needs"] if "backlog is growing" in n]
        self.assertTrue(
            needs[20] and needs[20] != needs[19], needs
        )  # the line still shows the new count
        self.assertEqual(fps[20], fps[19])

    def test_throughput_counts_and_losing_trend(self):
        self.store.ledger_add("post", 1, "h", "url")
        self.store.journal(
            "review", "#5 follow-up skipped: rebase-only (a -> b); our review stands"
        )
        self.store.append_jsonl(
            "events.jsonl",
            [
                {"kind": "NEW_PR", "pr": 2, "ts": now()},
                {"kind": "PUSH", "pr": 1, "ts": now()},
                {"kind": "PUSH", "pr": 1, "ts": ago(2000)},
            ],
        )
        self.store.append_jsonl(
            "metrics.jsonl", [{"ts": ago(360), "review_backlog": 0}]
        )
        tp = report.throughput(self.st, self.store, self.plan)
        self.assertEqual(
            (tp["posts"], tp["equiv"], tp["new"], tp["pushes"], tp["backlog"]),
            (1, 1, 1, 1, 2),
        )
        # Two waiting is no capacity problem (one push into an empty queue is not "LOSING").
        self.assertEqual(tp["trend"], "+2 in 6 h")
        many = [
            {"prio": 3, "kind": "FIRST_ROUND", "pr": 2, "text": "", "critical": False}
        ] * 25
        b = report.board(CFG, self.st, self.store, many, [])
        self.assertTrue(
            b["throughput"]["trend"].endswith(": LOSING"), b["throughput"]["trend"]
        )
        self.assertTrue(any("backlog is growing" in n for n in b["needs"]), b["needs"])

    def test_a_re_enabled_auto_merge_is_the_users_call(self):
        """Auto-merge turned back on by someone after our auto-merge guard turned it off is their call: the board does
        not say "I turn it off", as the auto-merge guard turns it off once per head."""
        cfg = dict(
            CFG, features={"merge_queue": {"mode": "live", "auto_merge": "enabled"}}
        )  # live: the auto-merge guard turns it off
        self.st["prs"]["3"]["author"] = "someone"
        head = self.st["prs"]["3"]["head"]
        self.st["automerge_risks"] = {
            "3": {
                "head": head,
                "by": "maintainer",
                "reasons": ["our review found problems"],
                "fixes": ["fix them"],
                "since": ago(30),
            }
        }
        needs = report.board(cfg, self.st, self.store, self.plan, [])["needs"]
        self.assertTrue(
            any(
                n.startswith("Auto-merge is on for #3") and "I turn it off" in n
                for n in needs
            ),
            needs,
        )
        self.store.ledger_add(
            "automerge-off", 3, "x" * 40, "an older head"
        )  # another head: still ours to turn off
        needs = report.board(cfg, self.st, self.store, self.plan, [])["needs"]
        self.assertTrue(
            any(n.startswith("Auto-merge is on for #3") for n in needs), needs
        )
        self.store.ledger_add("automerge-off", 3, head, "our review found problems")
        needs = report.board(cfg, self.st, self.store, self.plan, [])["needs"]
        line = next(n for n in needs if "#3" in n)
        self.assertTrue(
            line.startswith(
                "Auto-merge is on again for #3: maintainer turned it back on"
            ),
            line,
        )
        self.assertTrue(line.endswith("I don't turn it off twice: your call."), line)


class TestClip(unittest.TestCase):
    def test_a_cut_title_ends_in_an_ellipsis(self):
        title = (
            "test(rocke): add arch_specific helper for single-arch tests ("
            + TK
            + "-2819)"
        )
        line = report.approval_line(
            {
                "n": 10404,
                "verdict": "🟡",
                "author": "a",
                "title": title,
                "why": "w",
                "ci": "running",
                "also": [],
            }
        )
        self.assertIn("(" + TK + "-281… · w", line)
        self.assertEqual(len(report.clip(title)), 70)
        self.assertEqual(report.clip("short"), "short")


if __name__ == "__main__":
    unittest.main()
