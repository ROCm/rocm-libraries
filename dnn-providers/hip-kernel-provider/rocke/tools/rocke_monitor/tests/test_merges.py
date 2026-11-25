# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE PRs merged per UTC day: the hourly GitHub read, the daily log in between, each
PR once, develop only."""

import datetime as dt
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import daylog, merges, stats  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402


def day(offset: int) -> str:
    return (
        dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=offset)
    ).isoformat()


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class TestMerges(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.cfg = {"paths": {"daily_dir": self.d.name}}
        self.st = empty_state()
        self.st["prs"] = {"4": {"n": 4, "base": "users/x/stack"}}

    def tearDown(self):
        self.d.cleanup()

    def test_each_pr_counts_once_and_only_into_develop(self):
        self.st["merges_daily"] = {
            "at": now(),
            "days": 14,
            "prs": {"1": day(1), "2": day(0), "9": day(20)},
        }
        daylog.add(
            self.store, self.cfg, "merged", 2, "t2"
        )  # GitHub has it too: once, on GitHub's day
        daylog.add(
            self.store, self.cfg, "merged", 3, "t3"
        )  # merged since the last GitHub read: counts today
        daylog.add(
            self.store, self.cfg, "merged", 4, "t4"
        )  # into another PR's branch, not develop
        s = merges.series(self.st, self.store)
        self.assertEqual(len(s), 14)
        self.assertEqual(s[-1], (day(0), 2, [2, 3]))
        self.assertEqual(s[-2], (day(1), 1, [1]))
        self.assertEqual(sum(c for _, c, _ in s), 3)  # #9 merged before the window
        self.assertEqual(
            merges.board_note(self.st, self.store), "merged today 2 (last 7 days 3)"
        )

    def test_a_merge_into_another_prs_branch_is_logged_as_stacked(self):
        # A PR merged into another PR's branch (a stack collapsing into its bottom one) is logged as stacked
        self.st["prs"]["5"] = {"n": 5, "base": "develop", "title": "lands"}
        self.st["prs"]["4"]["title"] = "rides the stack"
        daylog.from_events(
            self.store,
            self.cfg,
            self.st,
            [
                {"kind": "MERGED", "pr": 5},
                {"kind": "MERGED", "pr": 4},
                {"kind": "CLOSED", "pr": 6},
            ],
        )
        rows = daylog.entries(self.store)
        self.assertEqual(
            [(r["kind"], r["pr"]) for r in rows],
            [("merged", 5), ("stacked", 4), ("closed", 6)],
        )
        self.assertEqual(rows[1]["text"], "into users/x/stack: rides the stack")
        self.assertIn("merged 1 (+1 stacked)", daylog.summary(rows))
        self.assertEqual(merges.series(self.st, self.store)[-1][2], [5])

    def test_github_is_read_at_most_hourly(self):
        calls = []

        def scoped(cfg, st, q):
            calls.append(q)
            return [
                {"number": 7, "mergedAt": f"{day(2)}T10:00:00Z"},
                {"number": 8, "mergedAt": None},
            ]

        with mock.patch.object(stats, "_scoped", scoped):
            self.assertTrue(merges.refresh(self.cfg, self.st))
            self.assertFalse(merges.refresh(self.cfg, self.st))
            self.st["merges_daily"]["at"] = ago(61)
            self.assertTrue(merges.refresh(self.cfg, self.st))
        self.assertEqual(len(calls), 2)
        self.assertIn("is:merged base:develop merged:>=", calls[0])
        self.assertEqual(self.st["merges_daily"]["prs"], {"7": day(2)})


if __name__ == "__main__":
    unittest.main()
