# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tick triage (rkmon_lib/triage.py): each tick line's class by the skill's tick procedure, the read-only rk
commands it may run, its pass with a fake model (the lines it places, escalates and runs; a model that does not answer;
one that keeps calling tools), and one pass against the fake gateway (tests/fakes/fake_models.py) on the reviewer
model's settings, its tokens in the run folder's log. No real model is called."""

import contextlib
import io
import json
import pathlib
import re
import sys
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "fakes"))
sys.dont_write_bytecode = True

import fake_models  # noqa: E402
from rkmon_lib import config, control, prep, secrets_registry, triage  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402

TICK = """(overlaps: #5's changed files are not local yet)
TICK 2026-11-05T10:20:00Z monitor ON reviews=live merge-queue=live auto-merge=enabled
develop 1234567890 CI green; last green 1234567890 (ci)
EVENTS (2): PUSH #7; MERGED #9
QUEUE: 1.#12*(d=0)
PLAN (2): P2 FOLLOW_UP #7 | P3 FIRST_ROUND #8
!! CRITICAL INCIDENT_OPEN i-3: develop CI red at abcdef1 (rocke-ci)
ANALYSIS_START #12 (G-ready, waited 3 min): early gate
PROPOSE #12 → #13: validated on develop 1234567890; tell the user to merge in this order
ARM_READY #12: batch #12 validated on develop 1234567890; run `rk arm 12`
!! DISARM_NEEDED #4: its head moved. Run `rk disarm 4`
!! PRIORITY_READY_EARLY #13 (wave 2, your priority): ready now, but wave 1 still waits on #12; your call
REVIEW_READY #7 (follow-up): worker done, checks pass. Inputs: /w/tmp/v7fu1
SOMETHING_NEW #8: a line no rule knows
BOARD (changed): show it to the user
| Part | State |
|---|---|
| Monitor | ON |

**Needs you**
1. approve #12
time: poll 3s queue 2s
"""


def cfg_for(d: pathlib.Path, **over) -> dict:
    raw = config.load_defaults()
    raw.pop("_doc", None)
    raw.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"]},
        features={
            "review_automation": {"mode": "live"},
            "merge_queue": {"mode": "live", "auto_merge": "enabled"},
        },
        reviewer="worker",
    )
    raw["paths"] = dict(raw["paths"], working_dir=str(d / "work"))
    raw.update(over)
    return config.normalize(raw)


class TestClassify(unittest.TestCase):
    def test_the_board_is_split_off_and_every_line_has_its_class(self):
        lines, board = triage.parse(TICK)
        self.assertEqual(board.splitlines()[0], "| Part | State |")
        self.assertTrue(board.endswith("1. approve #12"))
        self.assertNotIn("| Monitor | ON |", lines)
        self.assertIn("BOARD (changed): show it to the user", lines)
        got = dict(triage.classify_all(lines, True))
        want = {
            "(overlaps: #5's changed files are not local yet)": triage.QUIET,
            "TICK 2026-11-05T10:20:00Z monitor ON reviews=live merge-queue=live auto-merge=enabled": triage.QUIET,
            "EVENTS (2): PUSH #7; MERGED #9": triage.QUIET,
            "PLAN (2): P2 FOLLOW_UP #7 | P3 FIRST_ROUND #8": triage.QUIET,
            "!! CRITICAL INCIDENT_OPEN i-3: develop CI red at abcdef1 (rocke-ci)": triage.CLAUDE,
            "ANALYSIS_START #12 (G-ready, waited 3 min): early gate": triage.QUIET,
            "ARM_READY #12: batch #12 validated on develop 1234567890; run `rk arm 12`": triage.CLAUDE,
            "!! DISARM_NEEDED #4: its head moved. Run `rk disarm 4`": triage.CLAUDE,
            "SOMETHING_NEW #8: a line no rule knows": triage.UNSURE,
            "time: poll 3s queue 2s": triage.QUIET,
        }
        for line, cls in want.items():
            self.assertEqual(got[line], cls, line)
        self.assertEqual(
            got[next(x for x in lines if x.startswith("PROPOSE"))], triage.USER
        )
        self.assertEqual(
            got[next(x for x in lines if x.startswith("!! PRIORITY_READY_EARLY"))],
            triage.USER,
        )
        self.assertEqual(
            got[next(x for x in lines if x.startswith("REVIEW_READY"))], triage.CLAUDE
        )

    def test_incidents_are_claudes_only_while_incidents_are_filed(self):
        """`acting`: incidents are filed (the merge queue live with its develop guard: features.on(cfg,
        "incident-issues")); before that an incident is the user's to hear about."""
        line = "!! CRITICAL INCIDENT_OPEN i-3: develop CI red at abcdef1"
        self.assertEqual(
            [triage.classify(line, acting) for acting in (False, True)],
            [triage.USER, triage.CLAUDE],
        )
        self.assertEqual(
            triage.classify("BISECT_NEEDED i-3: 4 candidates", False), triage.USER
        )
        self.assertEqual(
            triage.classify("!! CRITICAL TOOL failure: gh timed out", False),
            triage.CLAUDE,
        )
        self.assertEqual(
            triage.classify("!! SOMETHING_URGENT #3: new", False), triage.CLAUDE
        )  # never left to a model
        for line in (
            "FIX_CANDIDATE #5 -> #9 (red-develop): names the test",
            "OVERLAP_NOTICE_DUE #5: k1 (owner #6)",
            "GPU_FAIL #5 gfx942 at abc: 2 GPU-only case(s)",
            "FOLLOWUP_SCAN #5: 3 candidates (1 deferred)",
            "WEEKLY REPORT 2026-W44 ready: /w/r.md (present it to the user)",
            "QUEUE_NOTICE #5: ready (x)",
            "TRIAGE #5: gate failed, held.",
            "!! HOLD_REVIEW_NEEDED #5: held",
            "DRIFT_FAIL #5 at develop d",
        ):
            self.assertEqual(triage.classify(line, True), triage.CLAUDE, line)
        for line in (
            "READY_FOR_APPROVAL #5 …",
            "PRIORITY_REQUEST #5 by someone: please",
            "HOLD_STALE #5: held 30h",
            "FEATURE_SKIPPED holds: on but needs me",
            "GPU_PENDING_LONG #5 gfx942 job 7: waiting 3.0 h",
        ):
            self.assertEqual(triage.classify(line, True), triage.USER, line)
        for line in (
            "GPU_QUEUED #5 gfx942 at abc: 2 case(s)",
            "REVIEW_START #5 (first-round): worker started",
            "LANDED #5 (armed by us); develop sanity follows",
            "DRIVER: nothing to start now",
            "(followup scan #5 failed, try 1 of 3: x)",
            "SANITY_GREEN abc: ok",
        ):
            self.assertEqual(triage.classify(line, True), triage.QUIET, line)


class TestReadOnly(unittest.TestCase):
    def test_only_the_read_only_commands_with_their_arguments(self):
        for args in (
            ["ci-fails", "12"],
            ["review-prep", "12"],
            ["review-prep", "12", "--round", "fu2"],
            ["board"],
            ["heads", "1", "2", "3"],
            ["ledger"],
            ["ledger", "5"],
            ["usage", "--days", "3", "--by-review"],
            ["overlaps", "12", "--section"],
            ["jobs", "j-12.3"],
        ):
            self.assertIsNone(triage.allowed(args), args)
        for args in (
            [],
            ["post", "12", "x.md"],
            ["arm", "12"],
            ["review-prep", "12", "--out", "/w/x"],
            ["status", "--queue"],
            ["classify", "12", "--record"],
            ["ci-fails", "12;", "rk", "arm"],
            ["heads"],
            ["review-start", "12"],
            ["tick"],
            ["ci-fails", "$(id)"],
        ):
            self.assertIsNotNone(triage.allowed(args), args)

    def test_review_prep_takes_the_rounds_rk_review_prep_takes(self):
        """One spelling of a round: prep.ROUND, by which rk review-prep checks its --round."""
        for rnd in "fr fr2 fu1 fu12 fu123 fu fr-1 x1 fu1; .".split() + [""]:
            self.assertEqual(
                triage.allowed(["review-prep", "12", "--round", rnd]) is None,
                bool(prep.ROUND.fullmatch(rnd)),
                rnd,
            )

    def test_none_writes_or_starts_work_and_each_is_an_rk_command(self):
        self.assertEqual(
            set(triage.READ_ONLY) & (set(control.WRITES) | set(control.STARTERS)), set()
        )
        import rkmon

        def exit_code(name: str) -> int:
            """rk's exit for `rk NAME --help`: 0 when rk's parser has the command, 2 when it does not."""
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit) as e:
                rkmon.main([name, "--help"])
            return e.exception.code

        for name in triage.READ_ONLY:
            self.assertEqual(exit_code(name), 0, name)
        self.assertEqual(exit_code("no-such-command"), 2)
        self.assertNotIn("review-driver", triage.READ_ONLY)


class FakeModel:
    """The tick-triage model as a script: each answer in turn, recording what it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def __call__(self, messages, tools):
        self.sent.append(([dict(m) for m in messages], tools))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def call(*args):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "rk", "arguments": {"args": list(args)}}}],
    }


VERDICT = {
    "summary": "rocke-monitor 10:20Z · develop 1234567 green · 2 events (PUSH, MERGED) · queue: #12 · next: #7",
    "unsure": [
        {
            "line": "SOMETHING_NEW #8: a line no rule knows",
            "class": "quiet",
            "why": "progress",
        }
    ],
    "claude": [
        {
            "line": "CI_AFTER_POST #9 abc1234: build-linux",
            "why": "a check failed after our post",
        }
    ],
    "user": ["#12 waits for the user's merge"],
    "done": ["rk ci-fails 9: build-linux failed at abc1234"],
}


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = cfg_for(pathlib.Path(self.tmp.name))
        lines, self.board = triage.parse(TICK)
        self.classes = triage.classify_all(lines, True)
        self.ran = []

    def rk(self, args):
        self.ran.append(args)
        return f"output of rk {' '.join(args)}\n(exit 0)"

    def test_it_runs_read_only_commands_places_lines_and_never_takes_one_from_claude(
        self,
    ):
        model = FakeModel(
            call("ci-fails", "9"),
            call("post", "9", "x.md"),
            {
                "role": "assistant",
                "content": "```json\n" + json.dumps(VERDICT) + "\n```",
            },
        )
        posts = [{"pr": 9, "head": "abc1234", "ts": "2026-11-05T09:00:00Z"}]
        v = triage.run(
            self.cfg, 0, self.classes, self.board, posts, chat=model, rk=self.rk
        )
        self.assertEqual(
            self.ran, [["ci-fails", "9"]]
        )  # the post was refused, never run
        self.assertEqual(v["ran"], ["rk ci-fails 9", "rk post 9 x.md (refused)"])
        self.assertEqual((v["summary"], v["requests"]), (VERDICT["summary"], 3))
        refused = model.sent[2][0][-1]
        self.assertEqual(refused["role"], "tool")
        self.assertIn("refused: rk post", refused["content"])
        brief = model.sent[0][0][1]["content"]
        self.assertIn("[claude] REVIEW_READY #7 (follow-up)", brief)
        self.assertIn("[unsure] SOMETHING_NEW #8", brief)
        self.assertIn("#9 at abc1234", brief)
        self.assertIn("| Monitor | ON |", brief)
        self.assertEqual(model.sent[0][1], [triage.TOOL])
        claude = triage.for_claude(self.classes, v)
        self.assertEqual(
            [x.split()[0] + " " + x.split()[1] for x in claude],
            [
                "!! CRITICAL",
                "ARM_READY #12:",
                "!! DISARM_NEEDED",
                "REVIEW_READY #7",
                "CI_AFTER_POST #9",
            ],
        )
        user = triage.for_user(self.classes, v)
        self.assertTrue(any(x.startswith("PROPOSE") for x in user))
        self.assertIn("#12 waits for the user's merge", user)
        # A verdict that says nothing about an unsure line leaves it with Claude; one that would drop a Claude line can't.
        v2 = dict(v, unsure={}, claude=[])
        self.assertIn(
            "SOMETHING_NEW #8: a line no rule knows",
            triage.for_claude(self.classes, v2),
        )
        self.assertTrue(
            any(
                x.startswith("ARM_READY")
                for x in triage.for_claude(
                    self.classes,
                    dict(
                        v2,
                        unsure={
                            "ARM_READY #12: batch #12 validated on develop 1234567890; run `rk arm 12`": "quiet"
                        },
                    ),
                )
            )
        )

    def test_a_model_that_does_not_answer_is_down(self):
        with self.assertRaisesRegex(triage.Down, "ConnectionError: refused"):
            triage.run(
                self.cfg,
                0,
                self.classes,
                "",
                [],
                chat=FakeModel(ConnectionError("refused")),
                rk=self.rk,
            )
        twice = FakeModel(
            {"content": "I think it is fine."}, {"content": "Still fine."}
        )
        with self.assertRaisesRegex(triage.Down, "no JSON"):
            triage.run(self.cfg, 0, self.classes, "", [], chat=twice, rk=self.rk)
        self.assertIsNone(twice.sent[1][1])  # the second ask offers no tools
        self.assertEqual(twice.sent[1][0][-1]["content"], triage.AGAIN)

    def test_a_model_that_keeps_calling_tools_must_answer_at_the_end(self):
        model = FakeModel(
            *[call("board") for _ in range(triage.STEPS)],
            {"content": json.dumps(VERDICT)},
        )
        v = triage.run(self.cfg, 0, self.classes, "", [], chat=model, rk=self.rk)
        self.assertEqual(len(self.ran), triage.STEPS)
        self.assertEqual(v["summary"], VERDICT["summary"])
        self.assertIsNone(model.sent[-1][1])

    def test_without_a_verdict_claude_gets_its_lines_and_every_unsure_one(self):
        out = triage.for_claude(self.classes, None)
        self.assertIn("SOMETHING_NEW #8: a line no rule knows", out)
        self.assertFalse(
            any(x.startswith(("PROPOSE", "ANALYSIS_START", "TICK")) for x in out)
        )

    def test_recent_posts_and_the_record(self):
        store = Store(pathlib.Path(self.tmp.name) / "state")
        store.ledger_add("post", 9, "abc1234def", "https://github.com/ROCm/r/pull/9#c1")
        self.assertEqual(
            [(p["pr"], p["head"]) for p in triage.recent_posts(store)],
            [(9, "abc1234def")],
        )
        triage.record(self.cfg, {"rc": 0, "claude": 2})
        rec = json.loads(
            (
                pathlib.Path(self.cfg["paths"]["toolkit"]) / "state" / triage.LOG
            ).read_text()
        )
        self.assertEqual((rec["rc"], rec["claude"]), (0, 2))

    def test_unavailable_says_why(self):
        self.assertIn(
            "the rule filter", triage.unavailable(self.cfg)
        )  # the default: no triage model
        self.assertEqual(triage.describe(self.cfg), triage.RULE_FILTER)
        want = dict(
            self.cfg, models=dict(self.cfg["models"], tick="worker")
        )  # the opt-in: the local worker's model, whose address and model the deploy settings file names
        self.assertEqual(
            triage.unavailable(want),
            "model_bindings.worker.server_url (the local review loop's endpoint); "
            "model_bindings.worker.model_id (the local review loop's model)",
        )
        on = dict(
            want,
            worker={
                "backend": "openai",
                "url": "https://models.example.invalid/v1",
                "model": "m-1",
            },
        )
        self.assertEqual(triage.unavailable(on), "")
        self.assertIn(
            "the rule filter", triage.unavailable(dict(on, models={"tick": ""}))
        )
        refused = dict(
            on,
            worker=dict(
                on["worker"],
                probe={
                    "url": "https://models.example.invalid/v1",
                    "model": "m-1",
                    "tools": "no tool calls",
                },
            ),
        )
        self.assertIn("failed the tool-call probe", triage.unavailable(refused))


class TestOnTheFakeGateway(unittest.TestCase):
    """The default model: the reviewer's settings on the fake gateway, its shared budget and the token log."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = pathlib.Path(self.tmp.name)
        key = base / "key"
        key.write_text("k-" + "fake-gateway-0099\n", encoding="utf-8")

        def answer_for(cls, req, tools):
            msgs = req.get("messages") or []
            if not any(m.get("role") == "tool" for m in msgs):
                return "", {"name": "rk", "arguments": {"args": ["board"]}}
            return json.dumps(VERDICT), None

        self.srv = fake_models.start(
            base / "fake",
            "gateway",
            key_file=str(key),
            answer_for=classmethod(answer_for),
        )
        self.addCleanup(self.srv.stop)
        self.cfg = cfg_for(
            base,
            worker={
                "backend": "openai",
                "url": self.srv.url,
                "model": fake_models.GOOD,
                "api_key_secret": secrets_registry.FIRST_PASS_KEY,
                "auth_header": "X-Example-Key",
                "auth_scheme": "",
                "rpm": 60,
                "stall_s": 30,
            },
            models=dict(config.load_defaults()["models"], tick="worker"),
        )  # the opt-in triage model
        # The key, in its secret's own file: the load derives worker.api_key_file and keeps no place config.json gives.
        mine = pathlib.Path(self.cfg["worker"]["api_key_file"])
        mine.parent.mkdir(parents=True, exist_ok=True)
        mine.write_text(key.read_text(encoding="utf-8"), encoding="utf-8")
        self.run_dir = pathlib.Path(self.cfg["paths"]["run"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base = base

    def test_one_triage_two_requests_and_their_tokens(self):
        lines, board = triage.parse(TICK)
        classes = triage.classify_all(lines, True)
        ran = []
        v = triage.run(
            self.cfg,
            0,
            classes,
            board,
            [],
            rk=lambda args: ran.append(args) or "| Monitor | ON |",
        )
        self.assertEqual(
            (ran, v["summary"], v["requests"]), ([["board"]], VERDICT["summary"], 2)
        )
        # The fake gateway logs a request after it answers it: wait for both lines.
        for _ in range(50):
            chats = [
                x
                for x in fake_models.read_log(self.base / "fake", "gateway")
                if x["method"] == "POST"
            ]
            if len(chats) >= 2:
                break
            time.sleep(0.1)
        self.assertEqual(
            [(x["status"], x["model"], x["tools"]) for x in chats],
            [(200, fake_models.GOOD, True)] * 2,
        )
        toks = [
            json.loads(x)
            for x in (self.run_dir / "tokens.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [(t["who"], t["model"], t["prompt"]) for t in toks],
            [("tick", fake_models.GOOD, 1000)] * 2,
        )
        self.assertEqual(v["tokens"], [2000, sum(t["answer"] for t in toks)])
        self.assertGreaterEqual(toks[0]["answer"], 300)


if __name__ == "__main__":
    unittest.main()
