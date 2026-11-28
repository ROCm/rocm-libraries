# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The weekly lessons-learned review of the agentic process: its raw material covers
the week, and only the week."""

import datetime as dt
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import lessons, stats  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402

INSIDE, OUTSIDE = "2026-10-28T10:00:00Z", "2026-10-20T10:00:00Z"


class TestLessons(unittest.TestCase):
    def test_the_raw_material_covers_the_week_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            store = Store(root / "state")
            cfg = {"weekly_report": {"dir": str(root / "weekly")}}
            start, end, lab = stats.week_bounds("2026-W44")
            store.append_jsonl(
                "daylog.jsonl",
                [
                    {
                        "ts": INSIDE,
                        "kind": "save",
                        "save": "verify",
                        "pr": 1,
                        "text": "qwen's +17 boilerplate dropped",
                    },
                    {
                        "ts": OUTSIDE,
                        "kind": "save",
                        "save": "verify",
                        "pr": 2,
                        "text": "last week's correction",
                    },
                    {
                        "ts": INSIDE,
                        "kind": "skipped",
                        "pr": 3,
                        "text": "REVIEW_DROP #3: head moved",
                    },
                ],
            )
            store.ledger_path.write_text(
                json.dumps(
                    {
                        "ts": INSIDE,
                        "kind": "automerge-off",
                        "pr": 4,
                        "head": "h",
                        "ref": "u",
                        "note": "our 🔴 at head",
                    }
                )
                + "\n"
            )
            store.journal_path.write_text(
                f"{INSIDE} CRITICAL TOOL failure: gh non-JSON page\n"
                f"{OUTSIDE} CRITICAL TOOL failure: last week\n"
                "  a continuation line with no timestamp\n"
            )
            notes = root / "notes"
            notes.mkdir()
            note = notes / "2026-10-28_pr10101_fu.md"
            note.write_text("dropped: qwen's REJECT header\nadded: our N1 FIXED\n")
            t = dt.datetime(2026, 10, 28, 10, tzinfo=dt.timezone.utc).timestamp()
            os.utime(note, (t, t))
            path = lessons.collect(cfg, store, start, end, lab, notes_dir=notes)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, "rocke_lessons_raw_2026-W44.md")
        self.assertIn("qwen's +17 boilerplate dropped", text)
        self.assertNotIn("last week's correction", text)
        self.assertIn("REVIEW_DROP #3: head moved", text)
        self.assertIn("#4: our 🔴 at head", text)
        self.assertIn("gh non-JSON page", text)
        self.assertNotIn("TOOL failure: last week", text)
        self.assertIn("### 2026-10-28_pr10101_fu.md", text)
        self.assertIn("dropped: qwen's REJECT header", text)


if __name__ == "__main__":
    unittest.main()
