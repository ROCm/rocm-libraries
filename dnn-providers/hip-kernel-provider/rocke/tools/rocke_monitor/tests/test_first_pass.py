# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The first passes (rkmon_lib/first_pass.py: models.ci_triage, models.quality_draft, models.notice_draft): at the
default nobody runs one and nothing changes; a bound model (a binding, or the local worker) runs it and hands its owner
leads whose quotes were found at their lines; a failing or slow model is a logged miss the owner works without; the
review driver starts the job only for a bound role; rk review-prep hands the leads to their owner; the notice lines
name the draft command only while notice_draft is bound; and every role's effort: its own, else claude.effort, a Claude
binding's request carrying it, rk roles showing where it comes from. No real model is called.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    ci_runs,
    config,
    doctor,
    drift,
    first_pass,
    models,
    overlaps,
    paths,
    prep,
    review,
    roles,
)
from rkmon_lib.store import Store  # noqa: E402

LOG = [
    "step 1",
    "compiling a.cpp",
    "a.cpp:3:1: error: no member named 'tile'",
    "make: *** [all] Error 1",
    "done",
]
LEADS = {
    "leads": [
        {
            "lead": "the build fails on a missing member",
            "evidence": [
                {
                    "source": "job 42",
                    "line": 3,
                    "quote": "a.cpp:3:1: error: no member named 'tile'",
                }
            ],
        },
        {
            "lead": "a claim with no line behind it",
            "evidence": [
                {
                    "source": "job 42",
                    "line": 1,
                    "quote": "nothing like this is in the log",
                }
            ],
        },
        {
            "lead": "a quote at the wrong line, found elsewhere",
            "evidence": [
                {"source": "job 42", "line": 1, "quote": "make: *** [all] Error 1"}
            ],
        },
    ],
    "draft": "**Drift**: the tests fail on develop.",
}


def cfg_for(d: pathlib.Path, endpoint: str = "http://127.0.0.1:9", **over) -> dict:
    """The configuration, with OVER over it; its bindings at ENDPOINT, which only a test that lets a binding's model be
    asked sets (a port it holds closed: every other test fakes the model)."""
    raw = config.load_defaults()
    raw.pop("_doc", None)
    raw.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"]},
        reviewer="default",
    )
    raw["paths"] = dict(raw["paths"], working_dir=str(d / "work"))
    raw["model_bindings"] = {
        "loc": {
            "provider": "openai",
            "url": endpoint,
            "model": "small-1",
            "max_resends": 0,
        },
        "cl": {
            "provider": "anthropic",
            "url": endpoint,
            "model": "claude-x",
        },
    }
    for k, v in over.items():
        raw[k] = dict(raw.get(k) or {}, **v) if isinstance(v, dict) else v
    return config.normalize(raw)


LLM = (
    first_pass.llm_client()
)  # the module itself, before any test puts a fake in its place


class FakeLLM:
    """worker/llm_client.py as first_pass uses it: records who was asked, with what effort; answers `answer`. What it
    does not fake is the module's own (the key's mark and the header reader a claude binding's headers are made with:
    rkmon_lib/bindings.py)."""

    def __init__(self, answer=LEADS, fail=None):
        self.calls, self.answer, self.fail = [], answer, fail

    def __getattr__(self, name):
        return getattr(LLM, name)

    def limiter_for(self, s, who="", run=True, **kw):
        test = self

        class Lim:
            def summary(self):
                return {"prompt_tokens": 10 * len(test.calls), "completion_tokens": 5}

        return Lim()

    def chat(self, s, messages, effort=None, limiter=None, max_tokens=None, **kw):
        self.calls.append(
            {"binding": s.get("binding"), "effort": effort, "messages": messages}
        )
        if self.fail:
            raise self.fail
        return {"content": json.dumps(self.answer)}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.store = Store(self.d / "state")
        self.out = self.d / "out" / "ci_triage.md"
        self.out.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def records(self, cfg) -> list[dict]:
        p = pathlib.Path(config.paths.get(cfg, "toolkit")) / "state" / first_pass.LOG
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def run_role(self, cfg, role="ci_triage", fake=None):
        fake = fake or FakeLLM()
        with mock.patch.object(first_pass, "llm_client", return_value=fake):
            rec = first_pass.run(
                cfg,
                role,
                [("job 42", "\n".join(LOG))],
                {"job 42": LOG},
                self.out,
                head="a" * 40,
                pr=7,
                store=self.store,
            )
        return rec, fake


class TestWhoRunsIt(Base):
    def test_the_default_runs_nothing_and_writes_nothing(self):
        cfg = cfg_for(self.d)
        for role in first_pass.ROLES:
            with self.subTest(role=role):
                self.assertEqual(config.model_for(cfg, role)["by"], "none")
                rec, fake = self.run_role(cfg, role)
                self.assertEqual((rec["status"], fake.calls), ("default", []))
                self.assertFalse(self.out.exists())
        self.assertEqual(self.records(cfg), [])  # nothing logged: today's behaviour
        self.assertEqual(first_pass.driver_roles(cfg), [])

    def test_a_bound_binding_runs_each_role_s_first_pass(self):
        for role in first_pass.ROLES:
            with self.subTest(role=role):
                cfg = cfg_for(self.d, models={role: "loc"})
                rec, fake = self.run_role(cfg, role)
                self.assertEqual([c["binding"] for c in fake.calls], ["loc"])
                self.assertEqual(
                    (rec["status"], rec["by"], rec["model"]),
                    ("ok", "binding", "small-1"),
                )
                text = self.out.read_text()
                self.assertIn("Leads only, from small-1 (binding", text)
                self.assertIn(first_pass.OWNER[role][1:], text)

    def test_the_local_worker_runs_it_through_the_reviewer_client(self):
        cfg = cfg_for(
            self.d,
            models={"quality_draft": "worker"},
            worker={"url": "http://127.0.0.1:9", "model": "w-1"},
        )
        asked = []

        class Client:
            class Budget:
                spent = [3, 4]

                def __init__(self, s, run_dir, log=None, who=""):
                    asked.append(("budget", who))

            @staticmethod
            def chat(s, messages, budget=None, rate_waits=3):
                asked.append(("chat", s["model"]))
                return {"content": json.dumps(LEADS)}

        with mock.patch.object(models, "client", return_value=Client):
            rec = first_pass.run(
                cfg,
                "quality_draft",
                [("x", "y")],
                {"job 42": LOG},
                self.out,
                store=self.store,
            )
        self.assertEqual(asked, [("budget", "quality_draft"), ("chat", "w-1")])
        self.assertEqual(
            (rec["status"], rec["by"], rec["tokens"]), ("ok", "worker", [3, 4])
        )

    def test_a_pass_on_the_local_worker_sends_its_own_effort(self):
        """A pass on the local worker sends its own effort as the worker takes it (its think), and none on "default":
        never worker.think, the first review pass's."""
        sent = []

        class Client:
            class Budget:
                spent = [0, 0]

                def __init__(self, s, run_dir, log=None, who=""):
                    pass

            @staticmethod
            def chat(s, messages, budget=None, rate_waits=3):
                sent.append(s.get("think"))
                return {"content": json.dumps(LEADS)}

        worker = {"url": "http://127.0.0.1:9", "model": "w-1", "think": "high"}
        for own, want in (("low", "low"), ("", None)):
            with self.subTest(own=own), mock.patch.object(
                models, "client", return_value=Client
            ):
                sent.clear()
                cfg = cfg_for(
                    self.d,
                    models={"quality_draft": "worker"},
                    worker=worker,
                    effort={"quality_draft": own},
                )
                rec = first_pass.run(
                    cfg,
                    "quality_draft",
                    [("x", "y")],
                    {"job 42": LOG},
                    self.out,
                    store=self.store,
                )
                self.assertEqual((sent, rec["effort"]), ([want], want or ""))

    def test_only_quotes_found_in_their_source_survive(self):
        rec, _ = self.run_role(cfg_for(self.d, models={"ci_triage": "loc"}))
        text = self.out.read_text()
        self.assertEqual((rec["leads"], rec["dropped"]), (2, 1))
        self.assertIn("`job 42:3`: `a.cpp:3:1: error: no member named 'tile'`", text)
        self.assertIn(
            "`job 42:4`: `make: *** [all] Error 1`", text
        )  # its line corrected
        self.assertNotIn("a claim with no line behind it", text)
        self.assertIn("1 lead(s) whose quotes were not were dropped", text)


class TestMisses(Base):
    def test_a_failing_endpoint_is_a_logged_miss_and_nothing_is_handed_on(self):
        closed = f"http://127.0.0.1:{sandbox.closed_port(self)}"
        cfg = cfg_for(self.d, endpoint=closed, models={"ci_triage": "loc"})
        self.out.write_text("an older run's leads")
        rec = first_pass.run(
            cfg,
            "ci_triage",
            [("job 42", "x")],
            {"job 42": LOG},
            self.out,
            pr=7,
            store=self.store,
        )
        self.assertEqual(rec["status"], "failed")
        self.assertFalse(self.out.exists())  # never a stale file for the owner
        self.assertEqual([r["status"] for r in self.records(cfg)], ["failed"])
        journal = self.store.journal_path.read_text()
        self.assertIn("ci_triage #7: no first pass (failed", journal)
        self.assertIn("the verifier works without it", journal)

    def test_a_model_past_its_cap_times_out(self):
        cfg = cfg_for(
            self.d,
            models={"notice_draft": "loc"},
            drafter={"first_pass_wall_min": 0.001},
        )

        def slow(messages):
            time.sleep(2)
            return json.dumps(LEADS)

        with mock.patch.object(first_pass, "GRACE_S", 0):
            t0 = time.monotonic()
            rec = first_pass.run(
                cfg,
                "notice_draft",
                [("f", "x")],
                {},
                self.out,
                chat=slow,
                store=self.store,
            )
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(rec["status"], "timeout")
        self.assertFalse(self.out.exists())

    def test_no_json_twice_and_a_model_not_at_hand_are_misses(self):
        cfg = cfg_for(self.d, models={"ci_triage": "loc"})
        rec, fake = self.run_role(cfg, fake=FakeLLM(answer=None))
        fake.answer = None
        with mock.patch.object(first_pass, "llm_client", return_value=fake):
            fake.chat = lambda *a, **k: {"content": "prose only"}
            rec = first_pass.run(
                cfg, "ci_triage", [("x", "y")], {}, self.out, store=self.store
            )
        self.assertEqual(
            (rec["status"], rec["why"]), ("failed", "it answered no JSON object, twice")
        )
        gone = cfg_for(self.d, models={"ci_triage": "worker"})  # no worker.url
        rec = first_pass.run(
            gone, "ci_triage", [("x", "y")], {}, self.out, store=self.store
        )
        self.assertEqual(rec["status"], "missed")
        self.assertIn(
            f"({doctor.worker_url_key(config.worker(gone))}, ", rec["why"]
        )  # the url's key, then the model's


class TestOwners(Base):
    def test_the_driver_starts_the_job_only_for_a_bound_role(self):
        st = {"prs": {"7": {"n": 7, "head": "7" * 40, "status": "open"}}}
        for models_, merge_only, want in (
            ({}, False, False),
            ({"ci_triage": "loc"}, False, True),
            ({"quality_draft": "loc"}, True, False),
        ):
            with self.subTest(models=models_, merge_only=merge_only):
                cfg = cfg_for(self.d, models=models_)
                ws = pathlib.Path(config.paths.get(cfg, "workers")) / "pr7" / "pr"
                ws.mkdir(parents=True, exist_ok=True)
                (ws / "meta.md").write_text("x")
                pipe = {"pr": 7, "head": "7" * 40, "merge_only": merge_only}
                with mock.patch.object(
                    review.jobs, "start", return_value={"id": "f1"}
                ) as start:
                    lines = review.first_pass_start(cfg, st, self.store, pipe)
                self.assertEqual(bool(lines), want)
                if want:
                    self.assertIn(
                        "first-pass 7 --head " + "7" * 40, start.call_args[0][3]
                    )
                    self.assertEqual(pipe[review.FIRST_PASS], {"id": "f1"})

    def test_every_pipe_s_advance_tries_the_first_pass(self):
        cfg = cfg_for(self.d, models={"ci_triage": "loc"})
        st = {"prs": {}, "review_pipes": [{"pr": 7, "head": "7" * 40}]}
        with mock.patch.object(
            review, "pipes", return_value=st["review_pipes"]
        ), mock.patch.object(
            review, "quality_start", return_value=[]
        ), mock.patch.object(
            review, "_advance_pipe", return_value=[]
        ), mock.patch.object(
            review, "_fill", return_value=[]
        ), mock.patch.object(
            review, "first_pass_start", return_value=["STARTED"]
        ) as fp:
            self.assertEqual(review._advance(cfg, st, self.store, []), ["STARTED"])
        self.assertEqual(fp.call_args[0][3]["pr"], 7)

    def test_rk_doctor_checks_each_bound_model(self):
        from rkmon_lib import doctor

        got = {
            c.name: c.status
            for c in doctor.first_pass_checks(
                cfg_for(self.d, models={"ci_triage": "loc", "notice_draft": "worker"})
            )
        }
        self.assertEqual(
            got,
            {
                "ci_triage first pass": "PASS",
                "quality_draft first pass": "OFF",
                "notice_draft first pass": "WARN",
            },
        )

    def test_the_job_s_ci_leads_reach_the_verifier_through_review_prep(self):
        cfg = cfg_for(self.d, models={"ci_triage": "loc"})
        head = "a" * 40
        fails = (
            [
                f"head {head}",
                "FAIL\tbuild\tfailure\thttps://github.com/acme/widgets/actions/runs/1/job/42",
                "runs 1",
            ],
            True,
        )

        def job_log(cfg_, job, **kw):
            ci_runs.job_log_path(job).write_text("\n".join(LOG))
            return ["5", "3:a.cpp:3:1: error: no member named 'tile'"], True

        fake = FakeLLM()
        # The job's log is kept in the temporary folder TMPDIR names (ci_runs.job_log_path): the test's own.
        with mock.patch.object(ci_runs, "fails", return_value=fails), mock.patch.object(
            ci_runs, "job_log", side_effect=job_log
        ), mock.patch.object(
            first_pass, "llm_client", return_value=fake
        ), mock.patch.dict(
            os.environ, {"TMPDIR": str(self.d)}
        ):
            lines = first_pass.job(cfg, {"prs": {}}, self.store, 7, head)
        self.assertEqual(lines[0][:20], "ci_triage: ok, 2 lea")
        review_dir = self.d / "v7fr"
        review_dir.mkdir()
        got = first_pass.copy(cfg, self.store, {"head": head}, 7, review_dir)
        self.assertIn("put it in the verifier's brief", got[0])
        self.assertIn(
            "no member named 'tile'", (review_dir / "ci_triage.md").read_text()
        )
        # another head: nothing copied, the owner works without it, and the miss is logged
        (review_dir / "ci_triage.md").unlink()
        got = first_pass.copy(cfg, self.store, {"head": "b" * 40}, 7, review_dir)
        self.assertIn(
            "ci_triage: no first pass (none at the head bbbbbbbbbb): the verifier works without it",
            got[0],
        )
        self.assertFalse((review_dir / "ci_triage.md").exists())
        self.assertEqual(
            first_pass.copy(cfg_for(self.d), self.store, {"head": head}, 7, review_dir),
            [],
        )

    def test_review_prep_calls_the_copy(self):
        cfg = cfg_for(self.d, models={"ci_triage": "loc"})
        with mock.patch.object(
            prep.first_pass, "copy", return_value=["COPIED"]
        ) as copy, mock.patch.object(
            prep.facts, "render", return_value=([], False)
        ), mock.patch.object(
            prep.review, "classify_pr", return_value=None
        ), mock.patch.object(
            prep.gitops, "fetch_pr", side_effect=prep.gitops.GitError("offline")
        ):
            lines, _ = prep.prepare(
                cfg, {"prs": {"7": {"head": "a" * 40}}}, self.store, 7, self.d / "v7"
            )
        self.assertIn("COPIED", lines)
        self.assertEqual(copy.call_args[0][3], 7)

    def test_quality_inputs_show_the_diff_and_check_quotes_against_the_head(self):
        repo = self.d / "repo"
        repo.mkdir()
        git = lambda *a: subprocess.run(
            ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
        )  # noqa
        git("init", "-q")
        git(
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@localhost",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        )
        (repo / "a.py").write_text("def f():\n    return 1\n")
        git("add", "a.py")
        git(
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@localhost",
            "commit",
            "-q",
            "-m",
            "head",
        )
        base, head = (
            git("rev-parse", "HEAD~1").stdout.strip(),
            git("rev-parse", "HEAD").stdout.strip(),
        )
        shown, sources = first_pass.quality_inputs(
            cfg_for(self.d), 7, head, base, str(repo)
        )
        self.assertIn("+    return 1", shown[1][1])
        self.assertEqual(sources["a.py"][1], "    return 1")

    def test_the_notice_lines_name_the_draft_only_while_notice_draft_is_bound(self):
        st = {
            "prs": {
                "7": {
                    "status": "open",
                    "drift": {
                        "status": "fail",
                        "failed": ["t::a"],
                        "develop": "d" * 40,
                    },
                }
            }
        }
        default, bound = cfg_for(self.d), cfg_for(
            self.d, models={"notice_draft": "loc"}
        )
        self.assertEqual(first_pass.notice_hint(default, 7, "drift"), "")
        self.assertIn(
            "`rk notice-draft 7 --kind drift`",
            first_pass.notice_hint(bound, 7, "drift"),
        )
        with mock.patch.object(
            overlaps, "decisions", return_value={"k": {"owner": 5, "members": [5, 7]}}
        ):
            line = overlaps.notices_due(
                st,
                self.store,
                hint=lambda n: first_pass.notice_hint(bound, n, "overlap"),
            )
        self.assertTrue(line[0].endswith("`rk notice-draft 7 --kind overlap`"), line)
        jdir = self.d / "drift"
        (jdir / "pr7").mkdir(parents=True)
        (jdir / "pr7" / "result").write_text("TESTS fail 1 failed")
        for cfg, want in ((default, False), (bound, True)):
            st["drift_job"] = {
                "job": {"id": "j"},
                "dir": str(jdir),
                "develop": "d" * 40,
                "items": [{"n": 7, "head": "7" * 40}],
            }
            with mock.patch.object(
                drift.jobs, "status", return_value="done"
            ), mock.patch.object(drift, "suspects", return_value=[]):
                line = drift.collect(cfg, st, self.store)[0]
            self.assertTrue(line.startswith("DRIFT_FAIL #7"), line)
            self.assertEqual(
                line.endswith("`rk notice-draft 7 --kind drift`"), want, line
            )
        self.assertIn(
            "write the drift notice yourself",
            first_pass.notice(default, st, self.store, 7, "drift")[0],
        )
        fake = FakeLLM(
            answer={
                "leads": [
                    {
                        "lead": "t::a fails",
                        "evidence": [{"source": "facts", "line": 1, "quote": "{"}],
                    }
                ],
                "draft": "**Drift**: t::a fails on develop.",
            }
        )
        with mock.patch.object(first_pass, "llm_client", return_value=fake):
            out = first_pass.notice(bound, st, self.store, 7, "drift")
        self.assertIn("Check every claim against the facts", out[0])
        path = pathlib.Path(out[0].split(": ")[1])
        self.assertIn("## Draft notice (unchecked)", path.read_text())
        self.assertIn(
            '"detail": "1 failed"', fake.calls[0]["messages"][1]["content"]
        )  # the facts it drafts from


class TestEfforts(Base):
    def test_a_role_s_own_effort_over_the_base_and_the_base_otherwise(self):
        cfg = cfg_for(
            self.d,
            claude={"effort": "xhigh"},
            effort={"verify": "low", "orchestrator": ""},
        )
        self.assertEqual(
            config.role_effort(cfg, "verify"),
            ("low", "features.review_automation.passes.verification_pass.effort"),
        )
        self.assertEqual(
            config.role_effort(cfg, "orchestrator"), ("xhigh", "claude.effort")
        )  # blank: the base
        self.assertEqual(
            config.role_effort(cfg, "quality"),
            ("medium", "features.review_automation.passes.code_quality_pass.effort"),
        )  # the shipped value
        # An optional pass at its default effort takes its owner's: the verifier's, the code quality pass's or the
        # orchestrator's, and follows it.
        self.assertEqual(
            config.role_effort(cfg, "ci_triage"),
            (
                "low",
                "features.review_automation.passes.verification_pass.effort (its owner's)",
            ),
        )
        owners = {
            "first_pass": "high",
            "ci_triage": "high",
            "quality_draft": "medium",
            "notice_draft": "high",
        }
        for role in config.FIRST_PASS_ROLES:
            self.assertEqual(
                config.role_effort(cfg_for(self.d), role)[0], owners[role], role
            )

    def test_no_effort_at_all_is_the_model_s_default(self):
        """Neither the role's effort nor claude.effort: no effort is passed (the SDK option, a Claude binding's
        request) and the agent files carry no effort line, so the model takes its own default.
        """
        from rkmon_lib import agentic, orchestrator

        cfg = cfg_for(
            self.d, claude={"effort": ""}, effort={"verify": "", "orchestrator": ""}
        )
        self.assertEqual(config.role_effort(cfg, "verify"), ("", config.MODEL_DEFAULT))
        self.assertIsNone(
            config.bound_effort(
                cfg, "verify", {"by": "binding", "provider": "anthropic"}
            )
        )
        seen = {}

        class Sdk:
            HookMatcher = staticmethod(lambda **kw: kw)

            @staticmethod
            def ClaudeAgentOptions(**kw):
                seen.clear()
                seen.update(kw)
                return kw

        orchestrator.options(cfg, Sdk, None)
        self.assertNotIn("effort", seen)
        orchestrator.options(cfg_for(self.d), Sdk, None)
        self.assertEqual(seen["effort"], "high")
        vals = {"<VERIFY_EFFORT>": agentic._effort(cfg, "verify")}
        text = agentic.fill(
            "---\nname: v\neffort: <VERIFY_EFFORT>\n---\nat effort <VERIFY_EFFORT>\n",
            vals,
        )
        self.assertEqual(text, f"---\nname: v\n---\nat effort {config.MODEL_DEFAULT}\n")

    def test_a_claude_binding_carries_the_role_s_effort_and_another_model_its_own(self):
        """A Claude binding's request carries the role's effort, its owner's while it has none; another model's, the
        role's own effort, and none on "default" or on an effort that model does not take.
        """
        from rkmon_lib import orchestrator

        cfg = cfg_for(self.d, models={"ci_triage": "cl"})
        login = orchestrator.key_path(
            cfg
        )  # a claude binding's key: the Claude session's, the claude login
        login.parent.mkdir(parents=True, exist_ok=True)
        login.write_text("not-a-real-key\n", encoding="utf-8")
        rec, fake = self.run_role(cfg)
        self.assertEqual(
            (fake.calls[0]["effort"], rec["effort"]), ("high", "high")
        )  # the verifier's, its owner's
        rec, fake = self.run_role(
            cfg_for(self.d, models={"ci_triage": "cl"}, effort={"ci_triage": "high"})
        )
        self.assertEqual(fake.calls[0]["effort"], "high")
        rec, fake = self.run_role(
            cfg_for(self.d, models={"ci_triage": "loc"}, effort={"ci_triage": "high"})
        )
        self.assertEqual((fake.calls[0]["effort"], rec["effort"]), ("high", "high"))
        for own in ("", "max"):
            with self.subTest(own=own):
                rec, fake = self.run_role(
                    cfg_for(
                        self.d, models={"ci_triage": "loc"}, effort={"ci_triage": own}
                    )
                )
                self.assertEqual((fake.calls[0]["effort"], rec["effort"]), (None, ""))
        maxed = cfg_for(self.d, models={"ci_triage": "cl"}, effort={"ci_triage": "max"})
        self.assertEqual(
            config.problems(maxed), []
        )  # a claude model takes every effort

    def test_the_tick_triage_s_claude_binding_carries_effort_tick(self):
        from rkmon_lib import triage

        fake = FakeLLM(answer={"summary": "quiet"})
        cfg = cfg_for(self.d, models={"tick": "cl"}, effort={"tick": "low"})
        with mock.patch.object(first_pass, "llm_client", return_value=fake):
            triage.run(cfg, 0, [], "", [])
        self.assertEqual([c["effort"] for c in fake.calls], ["low"])

    def test_the_tick_triage_on_another_model_sends_effort_tick_or_none(self):
        """The tick triage on a model that is not Claude, a binding or the local worker, sends effort.tick, and none
        while it is blank: never the base effort, nor worker.think."""
        from rkmon_lib import triage

        sent = []

        class Client:
            class Budget:
                spent = [0, 0]

                def __init__(self, s, run_dir, log=None, who=""):
                    pass

            @staticmethod
            def chat(s, messages, tools=None, budget=None, rate_waits=3):
                sent.append(s.get("think"))
                return {"content": json.dumps({"summary": "quiet"})}

        worker = {"url": "http://127.0.0.1:9", "model": "w-1", "think": "high"}
        for own, want in (("low", "low"), ("", None)):
            with self.subTest(own=own):
                fake = FakeLLM(answer={"summary": "quiet"})
                cfg = cfg_for(self.d, models={"tick": "loc"}, effort={"tick": own})
                with mock.patch.object(first_pass, "llm_client", return_value=fake):
                    triage.run(cfg, 0, [], "", [])
                self.assertEqual([c["effort"] for c in fake.calls], [want])
                sent.clear()
                cfg = cfg_for(
                    self.d,
                    models={"tick": "worker"},
                    worker=worker,
                    effort={"tick": own},
                )
                with mock.patch.object(models, "client", return_value=Client):
                    triage.run(cfg, 0, [], "", [])
                self.assertEqual(sent, [want])

    def test_rk_roles_shows_a_pass_on_the_local_worker_at_the_effort_it_sends(self):
        """rk roles' effort for each pass on the local worker: the first review pass's, worker.think as its drafter
        sends it (role_rows' draft row too); another pass's, its own effort, or "-" on "default" (none is sent).
        """
        cfg = cfg_for(
            self.d,
            reviewer="worker",
            models={
                "ci_triage": "worker",
                "quality_draft": "worker",
                "notice_draft": "worker",
            },
            worker={"url": "http://127.0.0.1:9", "model": "w-1", "think": "high"},
            effort={"ci_triage": "low", "notice_draft": "medium"},
        )
        rows = {r[0]: r[1:] for r in roles.pass_rows(cfg)}
        self.assertEqual(
            {
                name: rows[name][2]
                for name in rows
                if rows[name][0].startswith("worker ")
            },
            {
                roles.FIRST_REVIEW_PASS: "high",
                "ci_triage_pass": "low",
                "first_code_quality_pass": "-",
                "notice_draft_pass": "medium",
            },
        )
        by_role = {r[0]: r[2:] for r in roles.role_rows(cfg)}
        self.assertEqual(
            by_role["ci_triage"],
            ("low", "features.review_automation.passes.ci_triage_pass.effort"),
        )
        self.assertEqual(by_role["quality_draft"][0], "-")
        # the draft on the local review loop: the effort its drafter sends, and "-" while worker.think is unset
        self.assertEqual(
            by_role["draft"],
            ("high", "features.review_automation.passes.first_review_pass.effort"),
        )
        unset = cfg_for(
            self.d,
            reviewer="worker",
            worker={"url": "http://127.0.0.1:9", "model": "w-1"},
        )
        self.assertEqual(
            {r[0]: r[2:] for r in roles.role_rows(unset)}["draft"], roles.NO_EFFORT
        )

    def test_rk_roles_shows_every_role_s_model_effort_and_its_source(self):
        cfg = cfg_for(self.d, models={"notice_draft": "cl"}, effort={"verify": "low"})
        rows = {r[0]: r[1:] for r in roles.role_rows(cfg)}
        self.assertEqual(
            rows["verify"][1:],
            ("low", "features.review_automation.passes.verification_pass.effort"),
        )
        self.assertEqual(
            rows["orchestrator"][1:],
            ("high", "features.review_automation.orchestrator.effort"),
        )
        self.assertEqual(
            rows["notice_draft"][1:],
            ("high", "features.review_automation.orchestrator.effort (its owner's)"),
        )
        self.assertIn("cl (claude-x, claude)", rows["notice_draft"][0])
        self.assertEqual(rows["ci_triage"][1:], ("-", "no model"))
        for role in (
            "orchestrator",
            "verify",
            "quality",
            "post",
            "audit",
            "deep",
            "draft",
            "tick",
            *config.FIRST_PASS_ROLES,
        ):
            self.assertIn(role, rows)
        table = "\n".join(roles.table_lines(cfg))
        self.assertTrue(
            table.startswith("pass "), table
        )  # the deploy settings' passes first, resolved
        self.assertRegex(
            table,
            r"(?m)^ci_triage_pass +default → opus +claude +low \(from verification_pass\)$",
        )
        self.assertRegex(
            table,
            r"(?m)^notice_draft_pass +cl → claude-x +claude +high \(from orchestrator\)$",
        )
        self.assertIn("\nother roles:\nrole ", table)
        self.assertIn("effort from", table.split("other roles:")[1].splitlines()[1])

    def test_the_problems_name_a_bad_role_value_and_a_lead_with_no_reader(self):
        bad = cfg_for(self.d, models={"ci_triage": "nope"})
        self.assertTrue(
            any(
                "features.review_automation.passes.ci_triage_pass.model is 'nope'" in p
                for p in config.problems(bad)
            )
        )
        both = cfg_for(
            self.d, models={"quality_draft": "loc"}, roles={"quality": "loc"}
        )
        self.assertTrue(any("would have no reader" in p for p in config.problems(both)))
        self.assertFalse(
            [
                p
                for p in config.problems(
                    cfg_for(self.d, models={"quality_draft": "loc"})
                )
                if "quality_draft" in p
            ]
        )


class TestTheModelClient(unittest.TestCase):
    def test_its_folder_goes_on_the_import_path_once(self):
        """first_pass.llm_client, asked for twice: the same module, and worker/ on the import path once."""
        folder = str(paths.PACKAGE / "worker")
        with mock.patch.object(sys, "path", [p for p in sys.path if p != folder]):
            self.assertIs(first_pass.llm_client(), LLM)
            self.assertIs(first_pass.llm_client(), LLM)
            self.assertEqual(sys.path.count(folder), 1)

    def test_the_gateways_rows_the_scaler_and_the_tick_triage_take_it_from_here(self):
        """rk doctor's model gateway rows (reach.gateways), the scaler's budget and the tick triage's binding take the
        client from first_pass.llm_client, its one home: none puts worker/ on the import path itself. (The tick
        triage's pass takes it from here too: test_the_tick_triage_s_claude_binding_carries_effort_tick.)
        """
        from rkmon_lib import bindings, reach, scaler, triage

        fake = mock.Mock()
        fake.window_stats.return_value = {"requests": 1}
        fake.base_url.return_value = "https://gw.example.com/v1"
        binding = {"binding": "gw", "model": "m"}
        folder = str(paths.PACKAGE / "worker")
        with mock.patch.object(
            sys, "path", [p for p in sys.path if p != folder]
        ), mock.patch.object(first_pass, "llm_client", return_value=fake) as asked:
            self.assertEqual(reach.gateways({}), [])
            with mock.patch.object(bindings, "client_settings", return_value=binding):
                self.assertEqual(scaler.budget({}), {"requests": 1})
            self.assertEqual(triage.binding_gap(binding), "")
            self.assertEqual(sys.path.count(folder), 0)
        self.assertEqual(asked.call_count, 3)


if __name__ == "__main__":
    unittest.main()
