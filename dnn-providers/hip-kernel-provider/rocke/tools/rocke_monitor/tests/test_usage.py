# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk usage` (rkmon_lib/usage.py): the models' tokens per day and per review, from the orchestrator's records (turns,
subagent briefs, task starts and notifications, transcript splits), the reviewer model's token log and the ledger's
posts. Tokens only: no price and no cost anywhere."""

import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import config, usage  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402


class TestUsage(unittest.TestCase):
    def at(self, hh_mm: str) -> str:
        return f"{self.today}T{hh_mm}:00Z"

    def setUp(self):
        # Taken in each test, as usage takes it: a run that crosses 00:00 UTC still compares the same day.
        self.today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        raw = config.load_defaults()
        raw.pop("_doc", None)
        raw.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        raw["paths"] = dict(raw["paths"], working_dir=str(d / "work"))
        self.cfg = config.normalize(raw)
        self.state = pathlib.Path(self.cfg["paths"]["toolkit"]) / "state"
        self.store = Store(self.state)
        run = pathlib.Path(self.cfg["paths"]["run"])
        run.mkdir(parents=True)

        def turn(ts, kind, tokens, prs=None, ms=60000):
            return dict(
                {
                    "ts": ts,
                    "kind": kind,
                    "num_turns": 3,
                    "duration_ms": ms,
                    "input": tokens[0],
                    "output": tokens[1],
                    "cache_read": tokens[2],
                    "cache_write": tokens[3],
                },
                **({"prs": prs} if prs else {}),
            )

        rows = [
            turn(self.at("09:00"), "resume", (10, 2, 100, 0)),
            turn(self.at("10:05"), "act", (100, 20, 1000, 80), prs=[7, 8]),
            {
                "ts": self.at("10:01"),
                "kind": "agent",
                "tool_use_id": "tu-1",
                "subagent_type": "rocke-review-verifier",
                "pr": 7,
                "round": "fu1",
                "by": "main",
            },
            {
                "ts": self.at("10:01"),
                "kind": "task_start",
                "task_id": "a1",
                "tool_use_id": "tu-1",
            },
            {
                "ts": self.at("10:30"),
                "kind": "subagent",
                "agent_id": "a1",
                "agent_type": "rocke-review-verifier",
                "tool_use_id": "",
                "input": 50,
                "output": 500,
                "cache_read": 5000,
                "cache_write": 450,
            },
            {
                "ts": self.at("10:31"),
                "kind": "task",
                "task_id": "a1",
                "tool_use_id": "tu-1",
                "total_tokens": 9999,
            },
            {
                "ts": self.at("10:02"),
                "kind": "agent",
                "tool_use_id": "tu-2",
                "subagent_type": "rocke-code-quality-reviewer",
                "pr": 7,
                "round": "fu1",
                "by": "main",
            },
            {
                "ts": self.at("10:20"),
                "kind": "task",
                "task_id": "a2",
                "tool_use_id": "tu-2",
                "total_tokens": 3000,
            },
            turn(self.at("10:40"), "task-notification", (5, 1, 400, 0), prs=[7]),
            {
                "ts": self.at("11:00"),
                "kind": "refused",
                "tool_use_id": "tu-9",
                "why": "refused: a GitHub write",
            },
        ]
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "claude_usage.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows) + "not json\n"
        )
        toks = [
            {
                "ts": self.at("08:00"),
                "who": "pr7",
                "model": "m-1",
                "prompt": 20000,
                "answer": 3000,
            },
            {
                "ts": self.at("08:30"),
                "who": "pr7",
                "model": "m-1",
                "prompt": 25000,
                "answer": 2000,
            },
            {
                "ts": self.at("09:10"),
                "who": "tick",
                "model": "m-1",
                "prompt": 4000,
                "answer": 500,
            },
            {
                "ts": self.at("09:20"),
                "who": "planner",
                "model": "m-2",
                "prompt": 1000,
                "answer": 100,
            },
            {
                "ts": self.at("09:30"),
                "who": "pr8",
                "model": "m-1",
                "prompt": 7000,
                "answer": 700,
            },
        ]
        (run / "tokens.jsonl").write_text("".join(json.dumps(t) + "\n" for t in toks))
        self.store.ledger_add(
            "post", 7, "aaaa", "https://github.com/ROCm/r/pull/7#c1", note="first_round"
        )
        ledger = [
            json.loads(x) for x in self.store.ledger_path.read_text().splitlines()
        ]
        ledger[0][
            "ts"
        ] = f"{(dt.date.fromisoformat(self.today) - dt.timedelta(days=30)).isoformat()}T09:00:00Z"
        ledger.append(
            {
                "ts": self.at("10:35"),
                "kind": "post",
                "pr": 7,
                "head": "bbbb",
                "ref": "u2",
                "note": "followup",
            }
        )
        self.store.ledger_path.write_text("".join(json.dumps(r) + "\n" for r in ledger))

    def test_per_day(self):
        (d,) = usage.daily(self.cfg, self.store, 7)
        self.assertEqual(
            (d["turns"], d["turn_kinds"]),
            (3, {"resume": 1, "act": 1, "task-notification": 1}),
        )
        self.assertEqual(
            (d["input"], d["output"], d["cache_read"], d["cache_write"]),
            (115, 23, 1500, 80),
        )
        self.assertEqual(
            d["subagents"], 6000 + 3000
        )  # the transcript's split wins over the notification's total
        self.assertEqual(
            (d["posted"], d["claude"], d["refused"]),
            (1, 115 + 23 + 1500 + 80 + 9000, 1),
        )
        self.assertEqual(
            (d["model_prompt"], d["model_answer"], d["model_requests"]),
            (57000, 6300, 5),
        )
        self.assertEqual(
            d["model_roles"], {"tick": [4000, 500], "planner": [1000, 100]}
        )
        text = "\n".join(usage.report(self.cfg, self.store, 7))
        self.assertIn(
            f"{self.today}: Claude 3 turns (1 act, 1 resume, 1 task-notification); 115 / 23 / 1500 / 80; "
            "subagents 9000; 1 review(s) posted, about 10718 Claude tokens a review",
            text,
        )
        self.assertIn(
            "reviewer model 57000 / 6300 (5 requests; planner 1000 / 100; the tick triage 4000 / 500)",
            text,
        )
        self.assertIn("1 command(s) refused by the session's guard", text)
        for word in ("cost", "price", "$"):
            self.assertNotIn(word, text.lower())

    def test_per_review(self):
        rows = {
            (r["pr"], r["posted"] is not None): r
            for r in usage.by_review(self.cfg, self.store, 7)
        }
        r7 = rows[(7, True)]
        self.assertEqual((r7["round"], r7["posted"]), ("follow-up 1", self.at("10:35")))
        self.assertEqual(
            r7["subagents"],
            {"rocke-review-verifier": 6000, "rocke-code-quality-reviewer": 3000},
        )
        # the act turn (started 10:04, before the post) audited #7 and #8: half each; the notification's turn (10:39)
        # comes after the post: #7's next review, pending
        self.assertEqual(r7["session"], (100 + 20 + 1000 + 80) // 2)
        self.assertEqual(
            (r7["model_prompt"], r7["model_answer"], r7["model_requests"]),
            (45000, 5000, 2),
        )
        self.assertEqual(rows[(7, False)]["session"], 406)
        r8 = rows[(8, False)]
        self.assertEqual((r8["session"], r8["model_prompt"]), (600, 7000))
        text = "\n".join(usage.by_review_report(self.cfg, self.store, 7))
        self.assertIn(
            f"#7 follow-up 1, posted {self.today} 10:35Z: Claude 9600 (session 600; subagents 9000: "
            "rocke-code-quality-reviewer 3000, rocke-review-verifier 6000); reviewer model 45000 / 5000 "
            "(2 requests)",
            text,
        )
        self.assertIn(
            "#8 not posted yet: Claude 600 (session 600; subagents 0); reviewer model 7000 / 700",
            text,
        )

    def test_the_dashboard_tile(self):
        t = usage.tile(self.cfg, self.store, 7)
        self.assertEqual(len(t["days"]), 7)
        self.assertEqual(t["days"][-1], (self.today, 10718, 1, 63300))
        self.assertEqual(
            (t["claude"], t["posted"], t["model"], t["per_review"]),
            (10718, 1, 63300, 10718),
        )

    def test_nothing_recorded(self):
        empty = dict(
            self.cfg,
            paths=dict(
                self.cfg["paths"],
                toolkit=str(pathlib.Path(self.tmp.name) / "none"),
                run=str(pathlib.Path(self.tmp.name) / "none" / "run"),
            ),
        )
        store = Store(pathlib.Path(self.tmp.name) / "none" / "state")
        self.assertIn("none recorded yet", usage.report(empty, store, 7)[1])
        self.assertEqual(usage.tile(empty, store)["per_review"], None)


if __name__ == "__main__":
    unittest.main()
