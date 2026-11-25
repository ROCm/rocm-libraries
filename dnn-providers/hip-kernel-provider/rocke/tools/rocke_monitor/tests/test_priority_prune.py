# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Priorities whose PR merged or closed leave every list: the dashboard's rows and st["user_priority"] alike, an
approved request included."""

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import priority  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402


def request(n: int, status: str) -> dict:
    return {
        "id": f"req{n:06d}",
        "pr": n,
        "who": "someone",
        "why": "release",
        "at": now(),
        "got": now(),
        "status": status,
        "decided_at": now() if status != "pending" else None,
        "decided_by": "me",
        "note": "",
    }


class TestPrune(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        st = empty_state()
        st["prs"] = {
            "1": {"n": 1, "status": "open"},
            "2": {"n": 2, "status": "merged"},
            "3": {"n": 3, "status": "closed"},
            "4": {"n": 4, "status": "out-of-scope"},
        }
        st["user_priority"] = {
            k: {"why": f"w{k}", "at": now()} for k in ("1", "2", "3", "4", "5")
        }
        st["priority_requests"] = {
            "items": [
                request(1, "approved"),
                request(2, "approved"),
                request(3, "answered"),
                request(4, "pending"),
            ],
            "seen": {},
        }
        self.st = st

    def tearDown(self):
        self.d.cleanup()

    def test_merged_and_closed_priorities_are_cleared_once_with_a_journal_line(self):
        lines = priority.prune(self.st, self.store)
        self.assertEqual(
            lines,
            [
                "PRIORITY_CLEARED #2: merged; it leaves the board and the dashboard",
                "PRIORITY_CLEARED #3: closed; it leaves the board and the dashboard",
            ],
        )
        # Open, out of scope (scope can come back) and untracked (no evidence) priorities stay.
        self.assertEqual(sorted(self.st["user_priority"]), ["1", "4", "5"])
        journal = (pathlib.Path(self.d.name) / "journal.md").read_text(encoding="utf-8")
        self.assertIn("priority #2 cleared: merged (it was: w2)", journal)
        self.assertIn("priority #3 cleared: closed (it was: w3)", journal)
        self.assertEqual(priority.prune(self.st, self.store), [])  # once

    def test_requests_of_merged_or_closed_prs_leave_the_lists(self):
        self.assertEqual([it["pr"] for it in priority.shown(self.st)], [4, 1])
        self.st["prs"]["1"][
            "status"
        ] = "merged"  # merged after its request was approved: the row goes too
        self.assertEqual([it["pr"] for it in priority.shown(self.st)], [4])


if __name__ == "__main__":
    unittest.main()
