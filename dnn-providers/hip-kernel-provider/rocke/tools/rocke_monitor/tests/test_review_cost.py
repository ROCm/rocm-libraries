# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Each review's start and cost (rkmon_lib/review_cost.py): the mirror feed's line at each review start, the spec's
optional COST, and the line of state/review_cost.jsonl after a post through `rk post --premises`. GitHub and jobs are
faked."""

import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import premises, review, review_cost  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

H7, H8, OLD = "7" * 40, "8" * 40, "a" * 40


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class Case(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = pathlib.Path(self.d.name)
        self.cfg = {
            "review_first_line": "Automated review: Model - qwen, effort: max",
            "rocke_path": "rk",
            "repo": "ROCm/r",
            "operator": {"github_login": "me"},
            "settle_min": 30,
            "hot_spots": {},
            "worker_model": "qwen3.6:35b",
            "review": {"pipes": 2},
            "paths": {
                "working_dir": str(self.root),
                "root": str(self.root),
                "workers": str(self.root / "workers"),
            },
        }
        self.store = Store(self.root / "state")
        self.st = empty_state()
        common = {
            "status": "open",
            "draft": False,
            "size": 10,
            "head_seen_at": ago(300),
            "head_committed_at": ago(300),
        }
        self.st["prs"] = {
            "7": dict(
                common, n=7, head=H7, ours={"head": OLD, "at": ago(200), "count": 2}
            ),
            "8": dict(common, n=8, head=H8, ours=None),
        }


class TestFeed(Case):
    def test_each_review_start_writes_its_line(self):
        jobs = []

        def start(store, kind, name, script, timeout_min, env=None):
            jobs.append(name)
            return {"id": f"{kind}-{name}", "started": ago(0), "pid": 0}

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    review.gh,
                    "rest",
                    lambda path, jq=None, **k: {
                        "state": "open",
                        "head": H8 if path.endswith("/8") else H7,
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(
                    review,
                    "classify",
                    lambda cfg, rec, old: {
                        "kind": "substantive",
                        "old": old,
                        "new": rec["head"],
                        "changed_files": ["x.py"],
                        "n_changed": 1,
                        "new_commits": [],
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(review, "_changed_files", return_value=[])
            )
            stack.enter_context(mock.patch.object(review.jobs, "start", start))
            out = review.start_review(self.cfg, self.st, self.store, 8, "first-round")
            self.assertTrue(out[-1].startswith("REVIEW_START #8 (first-round)"), out)
            self.st["review_pipes"] = []
            out = review.start_review(
                self.cfg, self.st, self.store, 7, "follow-up", again=True
            )
            self.assertTrue(out[-1].startswith("REVIEW_START #7 (follow-up)"), out)
        lines = self.store.read_jsonl(review_cost.FEED)
        self.assertEqual(
            [{k: v for k, v in r.items() if k != "at"} for r in lines],
            [
                {
                    "pr": 8,
                    "head": H8,
                    "round": "fr",
                    "kind": "first-round",
                    "again": False,
                    "merge_only": False,
                },
                {
                    "pr": 7,
                    "head": H7,
                    "round": "fu2",
                    "kind": "follow-up",
                    "again": True,
                    "merge_only": False,
                },
            ],
        )
        self.assertEqual(review_cost.started(self.store, 7, H7), lines[1]["at"])

    def test_a_line_that_cannot_be_written_never_holds_the_start(self):
        review_cost.feed(self.st, self.store, 99, "first-round")  # an untracked PR
        self.assertEqual(self.store.read_jsonl(review_cost.FEED), [])
        self.assertIn(
            "#99 mirror feed line not written: KeyError",
            self.store.journal_path.read_text(),
        )


class TestCost(Case):
    def spec(self, **over) -> str:
        post = self.root / "pr7_fu2.md"
        post.write_text("post\n")
        spec = {
            "N": 7,
            "ROUND": "fu2",
            "SRC": str(post),
            "NOTES": str(self.root / "notes.md"),
            "HEAD10": H7[:10],
            "DEVBASE": "d" * 10,
            "THREADS": "0 0",
            "STOP": ago(1),
            "CONFLICTS": "",
            "MAXCANCEL": "0",
            "SAVE": "0 qwen claims corrected",
            **over,
        }
        path = self.root / "spec.json"
        path.write_text(json.dumps(spec))
        return str(path)

    def test_the_spec_s_cost_is_checked(self):
        post = str(self.root / "pr7_fu2.md")
        self.assertEqual(
            premises.load(
                self.spec(COST={"verifier": 180000, "quality": None}), 7, post
            )["COST"],
            {"verifier": 180000, "quality": None},
        )
        self.assertNotIn("COST", premises.load(self.spec(), 7, post))
        for cost, why in (
            ([1], 'COST: {"verifier": tokens, "quality": tokens}'),
            ({"verifer": 1}, "COST: unknown verifer (the keys are verifier, quality)"),
            ({"verifier": -1}, "COST verifier: a count of tokens, or null"),
            ({"quality": "230k"}, "COST quality: a count of tokens, or null"),
            ({"quality": True}, "COST quality: a count of tokens, or null"),
        ):
            with self.subTest(cost=cost), self.assertRaises(ValueError) as cm:
                premises.load(self.spec(COST=cost), 7, post)
            self.assertEqual(str(cm.exception), why)

    def test_the_record_after_a_post(self):
        ws = self.root / "workers" / "pr7"
        ws.mkdir(parents=True)
        (ws / "stats_agent.json").write_text(json.dumps({"steps": 3, "wall_s": 1096.4}))
        self.store.append_jsonl(
            review_cost.FEED,
            [
                {
                    "pr": 7,
                    "head": H7,
                    "round": "fu2",
                    "kind": "follow-up",
                    "again": False,
                    "merge_only": False,
                    "at": ago(50),
                }
            ],
        )
        self.st["drafts"] = [
            {
                "pr": 7,
                "head": H7,
                "binding": "",
                "status": "dropped",
                "tokens": 97795,
                "wall_s": 3096.1,
            },  # a pool draft dropped before the review: never the review's draft
            {
                "pr": 7,
                "head": H7,
                "role": "shadow_drafter",
                "binding": "glm",
                "method": "phased",
                "status": "ready",
                "tokens": 420000,
                "requests": 61,
                "wall_s": 1800.0,
            },
            {
                "pr": 7,
                "head": OLD,
                "role": "shadow_drafter",
                "binding": "glm",
                "tokens": 1,
            },
        ]
        self.store.save(self.st)
        spec = premises.load(
            self.spec(COST={"verifier": 180000, "quality": 230000}),
            7,
            str(self.root / "pr7_fu2.md"),
        )
        out = review_cost.record(
            self.cfg, self.store, 7, spec, {"sha": H7, "copy": self.root / "posted.md"}
        )
        rec = self.store.read_jsonl(review_cost.COSTS)[-1]
        self.assertEqual(
            (
                rec["pr"],
                rec["round"],
                rec["head"],
                rec["post"],
                rec["verifier"],
                rec["quality"],
            ),
            (7, "fu2", H7, str(self.root / "posted.md"), 180000, 230000),
        )
        self.assertEqual(
            rec["draft"],
            {
                "by": "qwen",
                "role": "drafter",
                "method": "local-worker",
                "status": "ready",
                "tokens": None,
                "requests": None,
                "wall_min": 18.3,
            },
        )
        self.assertEqual(
            rec["shadow"],
            {
                "by": "glm",
                "role": "shadow_drafter",
                "method": "phased",
                "status": "ready",
                "tokens": 420000,
                "requests": 61,
                "wall_min": 30.0,
            },
        )
        self.assertGreaterEqual(rec["review_min"], 49.0)
        self.assertTrue(
            out[0].startswith(
                "cost recorded: draft qwen uncounted, verifier 180000, quality 230000; "
            ),
            out,
        )
        # A binding's live draft, and no COST: the record says so.
        self.st["drafts"].append(
            {
                "pr": 7,
                "head": H7,
                "role": "drafter",
                "binding": "glm",
                "method": "phased",
                "status": "ready",
                "tokens": 300000,
                "requests": 40,
                "wall_s": 600.0,
            }
        )
        self.store.save(self.st)
        out = review_cost.record(
            self.cfg,
            self.store,
            7,
            premises.load(self.spec(), 7, str(self.root / "pr7_fu2.md")),
            {"sha": H7, "copy": "p"},
        )
        rec = self.store.read_jsonl(review_cost.COSTS)[-1]
        self.assertEqual(
            (
                rec["draft"]["by"],
                rec["draft"]["tokens"],
                rec["verifier"],
                rec["quality"],
            ),
            ("glm", 300000, None, None),
        )
        self.assertIn("verifier not given, quality not given", out[0])

    def test_the_live_draft_is_the_one_the_review_was_staged_with(self):
        """The line's live draft is the one recorded for the review at its head, not the configuration's now: a review
        staged with no drafter counts no draft after a drafter is back, and one the local worker drafted (its stage
        keeps no record) keeps its minutes after the passes change to no drafter."""
        ws = self.root / "workers" / "pr7"
        ws.mkdir(parents=True)
        (ws / "stats_agent.json").write_text(json.dumps({"steps": 3, "wall_s": 600.0}))
        self.st["drafts"] = [
            {"pr": 7, "head": H7, "role": "drafter", "status": "ready", "tokens": 9},
            {"pr": 7, "head": H7, "role": "drafter", "status": "none"},
        ]
        self.assertEqual(
            review_cost.drafts(self.cfg, self.st, 7, H7)[0],
            {
                "by": "none",
                "role": "drafter",
                "method": "none",
                "status": "none",
                "tokens": None,
                "requests": None,
                "wall_min": None,
            },
        )
        self.st["drafts"] = []
        live = review_cost.drafts(
            dict(self.cfg, worker_backend="none"), self.st, 7, H7
        )[0]
        self.assertEqual(
            (live["method"], live["status"], live["wall_min"]),
            ("local-worker", "ready", 10.0),
        )

    def test_finish_records_the_cost_and_a_failure_never_fails_it(self):
        spec = premises.load(
            self.spec(COST={"verifier": 1, "quality": 2}),
            7,
            str(self.root / "pr7_fu2.md"),
        )
        (self.root / "notes.md").write_text("notes\n")
        self.store.save(self.st)
        with mock.patch.object(
            premises.daylog, "save", lambda *a, **k: {}
        ), mock.patch.object(
            premises.daylog, "logged_line", lambda rec, n: "saved"
        ), mock.patch(
            "rkmon_lib.assessment.record", lambda *a, **k: []
        ):
            out, ok = premises.finish(
                self.cfg,
                self.store,
                7,
                spec,
                {"sha": H7, "dev": "d" * 40, "copy": self.root / "c.md"},
            )
            self.assertTrue(ok)
            self.assertTrue(any(x.startswith("cost recorded: ") for x in out), out)
            with mock.patch.object(
                review_cost, "record", side_effect=OSError("disk full")
            ):
                out, ok = premises.finish(
                    self.cfg,
                    self.store,
                    7,
                    spec,
                    {"sha": H7, "dev": "d" * 40, "copy": self.root / "c.md"},
                )
        self.assertIn("!! the cost was not recorded: OSError: disk full", out)


if __name__ == "__main__":
    unittest.main()
