# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The daily log (entries, saves, rendering) and the cleanup sweep (only merged/closed PRs, never shared trees)."""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import cleanup, daylog, gh  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

POST = """Automated review: Model - qwen, effort: max

## PR #5 — a fix (alice) — 🟠 Follow-up REJECT: golden not re-blessed

| Prior item | Status |
|---|---|
| Our 10-21 N1: rejection breaks a kernel | 🟢 FIXED `abcdef12` — capped in both engines (ran) |
| Bots / CI | 🟢 green |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟠 | `golden.json` | The differential golden mismatches on the merge → re-bless it (ran) |
| N2 | 🟡 | `x.py:3` | A nit |
"""


class TestDaylog(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.cfg = {"paths": {"daily_dir": str(pathlib.Path(self.d.name) / "daily")}}
        self.st = empty_state()

    def tearDown(self):
        self.d.cleanup()

    def test_post_entry_and_saves(self):
        daylog.from_review(
            self.store,
            self.cfg,
            5,
            "follow-up",
            POST,
            "https://github.com/ROCm/r/pull/5#c1",
        )
        rows = daylog.entries(self.store)
        review = [r for r in rows if r["kind"] == "review"][0]
        self.assertEqual(review["verdict"], "🟠")
        self.assertIn("N1 🟠, N2 🟡", review["text"])
        saves = sorted(r["save"] for r in rows if r["kind"] == "save")
        self.assertEqual(
            saves, ["finding", "fixed"]
        )  # N1 🟠 flagged; our 10-21 N1 confirmed fixed; the 🟡 is not a save
        day = rows[0]["ts"][:10]
        text = (
            pathlib.Path(self.cfg["paths"]["daily_dir"]) / f"rocke_{day}.md"
        ).read_text(encoding="utf-8")
        self.assertIn("reviews posted 1 (1 🟠)", text)
        self.assertIn("issues avoided 2 (finding 1, fixed 1)", text)
        self.assertIn("| review | #5 |", text)
        self.assertNotIn("kept unposted", text)

    def test_a_review_kept_unposted_is_its_own_kind_with_a_post_s_saves(self):
        """Posting off (url None): an `unposted` entry with no link, and the saves its post would make, so a
        prototype's log reads as a live monitor's; the summary counts it apart from the reviews posted.
        """
        daylog.from_review(self.store, self.cfg, 5, "follow-up", POST, None)
        rows = daylog.entries(self.store)
        self.assertEqual(
            [(r["kind"], r.get("save"), r.get("url")) for r in rows],
            [
                ("unposted", None, None),
                ("save", "finding", None),
                ("save", "fixed", None),
            ],
        )
        self.assertEqual(rows[0]["verdict"], "🟠")
        text = (
            pathlib.Path(self.cfg["paths"]["daily_dir"])
            / f"rocke_{rows[0]['ts'][:10]}.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "reviews posted 0 · reviews kept unposted 1 (1 🟠) · rebase-only kept 0",
            text,
        )
        self.assertIn("issues avoided 2 (finding 1, fixed 1)", text)
        self.assertIn("| unposted | #5 | follow-up 🟠 Follow-up REJECT", text)
        self.assertNotIn("[link]", text)

    def test_lines_proposal_dedupe_and_incident_save(self):
        lines = [
            "REVIEW_SKIP #7: rebase-only since our review; recorded abc as equivalent",
            "GATE_DONE #8: G5 pass; G6 pass; G7 pass",
            "PROPOSE #8: validated on develop d1; tell the user",
            "PROPOSE #8: validated on develop d2; tell the user",
            "!! CRITICAL INCIDENT_OPEN INC-2026-01-01-1: develop CI red at abc",
            "REVIEW #9: worker running, checks none",
        ]
        daylog.from_lines(self.store, self.cfg, self.st, lines)
        kinds = [r["kind"] for r in daylog.entries(self.store)]
        self.assertEqual(kinds, ["rebase", "gate", "proposed", "incident", "save"])
        self.assertEqual(daylog.saves_since(self.store)["develop"], 1)


class TestCleanup(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        for rel in (
            "wt100",
            "wt100m",
            "build/wt100m",
            "workers/pr100",
            "wt200",
            "workers/pr200",
            "workers/pr300",
            "wtbase",
            "wtsanity",
            "build/wtsanity",
            "repo",
        ):
            (self.root / rel).mkdir(parents=True)
        self.store = Store(self.root / "state")
        self.st = empty_state()
        self.st["prs"] = {
            "100": {"n": 100, "status": "merged"},
            "200": {"n": 200, "status": "open"},
        }

    def tearDown(self):
        self.d.cleanup()

    def test_dry_run_then_sweep(self):
        with mock.patch.object(
            gh, "rest", lambda path, jq=None: {"state": "closed", "merged": True}
        ):
            dry = cleanup.sweep(
                {"repo": "ROCm/r"}, self.st, self.store, self.root, limit=None, dry=True
            )
            self.assertTrue(
                any(line.startswith("would remove #100") for line in dry), dry
            )
            self.assertTrue((self.root / "wt100").exists())
            lines = cleanup.sweep(
                {"repo": "ROCm/r"}, self.st, self.store, self.root, limit=None
            )
        self.assertTrue(
            any(line.startswith("CLEANUP #100 (merged)") for line in lines), lines
        )
        self.assertTrue(
            any(line.startswith("CLEANUP #300 (merged)") for line in lines), lines
        )  # untracked: asked GitHub
        for gone in (
            "wt100",
            "wt100m",
            "build/wt100m",
            "workers/pr100",
            "workers/pr300",
        ):
            self.assertFalse((self.root / gone).exists(), gone)
        for kept in (
            "wt200",
            "workers/pr200",
            "wtbase",
            "wtsanity",
            "build/wtsanity",
            "repo",
        ):
            self.assertTrue((self.root / kept).exists(), kept)

    def test_in_use_is_kept(self):
        self.st["review_ready"] = [
            {"pr": 100, "head": "x", "kind": "follow-up", "at": "t"}
        ]
        with mock.patch.object(
            gh, "rest", lambda path, jq=None: {"state": "open", "merged": False}
        ):
            lines = cleanup.sweep(
                {"repo": "ROCm/r"}, self.st, self.store, self.root, limit=None
            )
        self.assertFalse(any("#100" in line for line in lines), lines)
        self.assertTrue((self.root / "wt100").exists())


if __name__ == "__main__":
    unittest.main()
