# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The draft stage's contract as the review driver uses it (rkmon_lib/draft_stage.py, review.py): the stage the
configuration picks, the job's script, adopting a pool draft, the scaler hook and its decision log, the checks once
staged, the one retry, REVIEW_READY with a draft or without, the per-draft records, the pool's pace, and the local
worker's path unchanged when no drafter binding is set. Jobs are fakes; the endpoint is the closed port 127.0.0.1:9.
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import bindings, draft_stage, jobs, review, scaler  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

OLD = "2000-01-01T00:00:00Z"
BINDING = {
    "provider": "openai",
    "url": "http://127.0.0.1:9/onprem",
    "model": "Cand-1",
    "tpm": 1000,
    "rpm": 10,
}


def a_cfg(work: pathlib.Path, **over) -> dict:
    cfg = {
        "review_first_line": "Automated review by rocke-monitor: Model - Cand-1, effort: high",
        "rocke_path": "proj",
        "repo": "o/r",
        "hot_spots": {},
        "settle_min": 0,
        "paths": {
            "root": str(work),
            "workers": str(work / "workers"),
            "repo": str(work / "repo"),
        },
        "drafter": {"limiter_dir": str(work / "limits")},
    }
    cfg.update(over)
    return cfg


def with_binding(work: pathlib.Path, **over) -> dict:
    return a_cfg(
        work, worker_backend="cand", model_bindings={"cand": dict(BINDING)}, **over
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = pathlib.Path(self.tmp.name)
        self.store = Store(self.work / "state")
        self.st = empty_state()
        self.st["last_tick"] = "2999-01-01T00:00:00Z"
        self.st["prs"] = {
            str(n): {
                "n": n,
                "head": str(n) * 40,
                "status": "open",
                "draft": False,
                "size": 10,
                "head_seen_at": OLD,
            }
            for n in (1, 2, 3)
        }
        self.status: dict[str, str] = {}
        self.started: list = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_jobs(self):
        def start(store, kind, name, script, timeout_min, env=None):
            jid = f"{kind}-{name}-{len(self.started)}"
            self.started.append(
                {
                    "kind": kind,
                    "name": name,
                    "script": script,
                    "timeout": timeout_min,
                    "env": env,
                }
            )
            self.status.setdefault(jid, "running")
            return {"id": jid, "started": OLD, "pid": 0}

        return [
            mock.patch.object(jobs, "start", start),
            mock.patch.object(
                jobs,
                "status",
                lambda store, job: (
                    self.status.get((job or {}).get("id"), "none") if job else "none"
                ),
            ),
            mock.patch.object(
                jobs,
                "kill",
                lambda store, job: self.status.__setitem__(job["id"], "failed(rc=143)"),
            ),
            mock.patch.object(jobs, "tail", lambda store, job, n=20: ""),
        ]

    def run_patched(self, fn, *patches):
        ps = self.fake_jobs() + list(patches)
        for p in ps:
            p.start()
        try:
            return fn()
        finally:
            for p in reversed(ps):
                p.stop()


class TestStageChoice(Base):
    def test_the_stage_the_configuration_picks(self):
        self.assertIs(draft_stage.for_config(a_cfg(self.work)), draft_stage.LOCAL)
        self.assertIs(
            draft_stage.for_config(with_binding(self.work)), draft_stage.TOOL_LOOP
        )
        self.assertIs(
            draft_stage.for_config(with_binding(self.work, draft_stage="none")),
            draft_stage.NONE,
        )
        self.assertIs(
            draft_stage.for_config(a_cfg(self.work, worker_backend="qwen")),
            draft_stage.LOCAL,
        )
        self.assertIs(draft_stage.of_pipe({"stage": "worker"}), draft_stage.LOCAL)
        self.assertIs(
            draft_stage.of_pipe({"stage_name": "tool-loop"}), draft_stage.TOOL_LOOP
        )

    def test_the_jobs_script(self):
        cfg = with_binding(
            self.work, effort={"first_pass": "high"}
        )  # the effort line 1 names
        s = draft_stage.script(
            cfg, draft_stage.TOOL_LOOP, 7, "Follow-up", ["projects/x"]
        )
        self.assertIn(f"rm -rf {self.work}/workers/pr7", s)
        self.assertIn(
            "REVIEW_ROUND=Follow-up REVIEW_LINE1='Automated review by rocke-monitor: Model - Cand-1, effort: high'",
            s,
        )
        self.assertIn("SPARSE_EXTRA=projects/x", s)
        self.assertTrue(s.rstrip().endswith("tools/draft_run.py 7"), s)
        self.assertIn("python3 -B ", s.splitlines()[-1])
        self.assertIn(
            f"export RKMON_WORKING_DIR={self.work}", s
        )  # the worker tools' settings
        self.assertNotIn("ollama stop", s)  # no local model to unload
        retry = draft_stage.script(
            cfg, draft_stage.TOOL_LOOP, 7, "Follow-up", [], retry=True
        )
        self.assertNotIn("rm -rf", retry)
        self.assertTrue(retry.rstrip().endswith("draft_run.py 7 --retry"))
        self.assertTrue(
            draft_stage.script(cfg, draft_stage.NONE, 7, "First-round", [])
            .rstrip()
            .endswith("draft_run.py 7 --stage-only")
        )
        # The local worker's own script is untouched.
        local = review.worker_script(a_cfg(self.work), 7, [], "first-round")
        self.assertIn("review_agent_run.py", local)
        self.assertIn("ollama", local)

    def test_the_local_worker_path_adds_nothing_to_the_state(self):
        cfg = a_cfg(self.work)
        self.st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "first-round",
                "stage": "worker",
                "head": "1" * 40,
                "since": None,
                "worker": {"id": "w", "started": OLD},
                "checks": {"id": "c"},
            }
        ]
        self.status.update(w="done", c="done")
        with mock.patch.object(
            review, "has_draft", lambda *a, **kw: True
        ), mock.patch.object(
            review, "_record_checks_gates", lambda cfg, rec, n: None
        ), mock.patch.object(
            review, "pick_next", lambda *a, **kw: None
        ):
            lines = self.run_patched(
                lambda: review.advance(cfg, self.st, self.store, [])
            )
        self.assertTrue(
            lines[0].startswith(
                "REVIEW_READY #1 (first-round): worker done, checks done."
            ),
            lines,
        )
        self.assertNotIn("draft_pool", self.st)
        self.assertNotIn("drafts", self.st)
        local = {"worker": {"url": "http://127.0.0.1:11434", "backend": "ollama"}}
        self.assertTrue(
            review.worker_prefill(
                local,
                {
                    "review_pipes": [
                        {"pr": 1, "worker": {"started": "2999-01-01T00:00:00Z"}}
                    ]
                },
            )
        )

    def test_the_local_worker_is_the_reviewer_model(self):
        """The local worker's model id and tag are the reviewer model's (worker.model); a drafter binding that names no
        model takes it too."""
        cfg = a_cfg(self.work, worker={"model": "qwen3.6:35b"})
        self.assertEqual(
            (bindings.worker_model(cfg), bindings.worker_tag(cfg)),
            ("qwen3.6:35b", "qwen"),
        )
        self.assertEqual(
            bindings.candidates(cfg),
            [{"role": "drafter", "tag": "qwen", "binding": "", "model": "qwen3.6:35b"}],
        )
        cfg = a_cfg(
            self.work,
            worker={"model": "M-1"},
            worker_backend="cand",
            model_bindings={"cand": {"url": "http://127.0.0.1:9"}},
        )
        self.assertEqual(bindings.binding(cfg, "drafter")["model"], "M-1")


class TestToolLoopPipeline(Base):
    def setUp(self):
        super().setUp()
        self.cfg = with_binding(self.work)
        self.ws = draft_stage.folder(self.cfg, 1)
        self.ws.mkdir(parents=True)
        self.patches = [
            mock.patch.object(review, "_record_checks_gates", lambda cfg, rec, n: None),
            mock.patch.object(review, "_changed_files", lambda cfg, rec: ["proj/x.py"]),
            mock.patch.object(review, "emitter_families", lambda tree, k: {}),
            mock.patch.object(review, "pick_next", lambda *a, **kw: None),
            mock.patch.object(
                review.gh,
                "rest",
                lambda path, **k: {
                    "state": "open",
                    "head": path.rsplit("/", 1)[1] * 40,
                },
            ),
        ]

    def advance(self):
        return self.run_patched(
            lambda: review.advance(self.cfg, self.st, self.store, []), *self.patches
        )

    def pipe(self, **over):
        p = {
            "pr": 1,
            "kind": "first-round",
            "stage": "worker",
            "head": "1" * 40,
            "since": None,
            "started": OLD,
            "worker": {"id": "d1", "started": OLD},
            "checks": None,
            "stage_name": "tool-loop",
        }
        p.update(over)
        self.st["review_pipes"] = [p]
        return p

    def test_start_adopts_the_pool_draft_or_asks_the_scaler(self):
        self.st["draft_pool"] = {
            "1": {
                "pr": 1,
                "kind": "first-round",
                "head": "1" * 40,
                "job": {"id": "pool1", "started": OLD},
                "started": OLD,
            }
        }
        lines = self.run_patched(
            lambda: review.start_review(
                self.cfg, self.st, self.store, 1, "first-round"
            ),
            *self.patches,
        )
        self.assertEqual(
            lines[-1],
            "REVIEW_START #1 (first-round): pool draft adopted at 1111111111 (parallel review 1 of 3)",
        )
        self.assertEqual(review.pipe_of(self.st, 1)["worker"]["id"], "pool1")
        self.assertEqual(review.pipe_of(self.st, 1)["stage_name"], "tool-loop")
        self.assertEqual(self.st["draft_pool"], {})
        self.assertEqual(self.started, [])
        # No pool draft at this head: a pool draft of another head stops; the scaler decides; a deny waits, an allow
        # starts the job.
        self.st["review_pipes"] = []
        self.st["draft_pool"]["2"] = {
            "pr": 2,
            "kind": "first-round",
            "head": "e" * 40,
            "job": {"id": "old2", "started": OLD},
            "started": OLD,
        }
        self.status["old2"] = "running"
        with mock.patch.object(scaler, "decide", lambda inp: (False, "the ceiling")):
            lines = self.run_patched(
                lambda: review.start_review(
                    self.cfg, self.st, self.store, 2, "first-round"
                ),
                *self.patches,
            )
        self.assertEqual(lines[-1], "REVIEW_WAIT #2: its draft waits: the ceiling")
        self.assertEqual(
            (
                self.status["old2"],
                self.st["draft_pool"],
                self.st["drafts"][-1]["status"],
            ),
            ("failed(rc=143)", {}, "dropped"),
        )
        lines = self.run_patched(
            lambda: review.start_review(
                self.cfg, self.st, self.store, 2, "first-round"
            ),
            *self.patches,
        )
        self.assertEqual(
            lines[-1],
            "REVIEW_START #2 (first-round): draft started at 2222222222 (parallel review 1 of 3)",
        )
        job = self.started[-1]
        self.assertEqual(
            (job["kind"], job["timeout"], job["env"]),
            ("draft", 110, {"RKMON_DRAFT_PRIO": "0"}),
        )
        log = [
            json.loads(x)
            for x in (self.store.dir / scaler.LOG).read_text().splitlines()
        ]
        self.assertEqual([(r["pr"], r["allow"]) for r in log], [(2, False), (2, True)])
        self.assertEqual(
            set(log[-1]["inputs"]), {"budget", "host", "queue", "running", "ceiling"}
        )
        self.assertFalse(
            review.worker_prefill(self.cfg, self.st)
        )  # the model runs elsewhere: no prefill hold

    def test_checks_start_once_staged(self):
        self.pipe()
        self.status["d1"] = "running"
        self.assertEqual(
            self.advance()[0], "REVIEW #1: worker running, checks none"
        )  # not staged yet
        (self.ws / "staged").write_text("now\n")
        (self.work / "wt1m").mkdir()
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith("REVIEW #1: worker running; checks started"), lines
        )

    def test_one_retry_then_ready_without_the_draft(self):
        self.pipe(checks={"id": "c1", "started": OLD})
        self.status.update(d1="failed(rc=1)", c1="running")
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith(
                "REVIEW #1: the draft ended with no draft (failed(rc=1)); retrying once"
            ),
            lines,
        )
        self.assertTrue(
            self.started[-1]["script"].rstrip().endswith("draft_run.py 1 --retry")
        )
        retry_id = review.pipe_of(self.st, 1)["worker"]["id"]
        self.status[retry_id] = "failed(rc=1)"
        self.status["c1"] = "done"
        (self.ws / "tokens_total.json").write_text(
            json.dumps(
                {
                    "total": {
                        "tokens": 5000,
                        "requests": 7,
                        "waits_s": 3.5,
                        "wall_s": 240,
                    }
                }
            )
        )
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith(
                "REVIEW_READY #1 (first-round): no draft (the drafter failed twice): "
                "verify without it, checks done."
            ),
            lines,
        )
        d = self.st["drafts"][-1]
        self.assertEqual(
            (
                d["status"],
                d["attempts"],
                d["tokens"],
                d["requests"],
                d["waits_s"],
                d["wall_s"],
            ),
            ("missing", 2, 5000, 7, 3.5, 240.0),
        )
        self.assertEqual(self.st["review_ready"][0]["pr"], 1)

    def test_ready_with_the_draft_its_evidence_and_lateness(self):
        self.pipe(checks={"id": "c1", "started": OLD})
        self.status.update(d1="done", c1="done")
        (self.ws / "review_agent.md").write_text("draft\n")
        (self.ws / "evidence_check.json").write_text(
            json.dumps({"summary": "PASS: 4 claims", "result": "PASS"})
        )
        (self.ws / "draft_findings.json").write_text(
            json.dumps(
                {
                    "findings": [
                        {"id": "N1", "severity": "🟠"},
                        {"id": "N2", "severity": "🟡", "dropped": True},
                    ],
                    "priors": [{}],
                }
            )
        )
        (self.ws / "tokens_total.json").write_text(
            json.dumps(
                {
                    "total": {
                        "tokens": 40000,
                        "requests": 21,
                        "waits_s": 0,
                        "wall_s": 600,
                    }
                }
            )
        )
        for jid, t in (
            ("d1", 2000.0),
            ("c1", 1000.0),
        ):  # the draft ended 1000 s after the checks
            d = self.store.dir / "jobs" / jid
            d.mkdir(parents=True)
            (d / "rc").write_text("0")
            import os

            os.utime(d / "rc", (t, t))
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith(
                "REVIEW_READY #1 (first-round): Cand-1 draft ready (40000 tokens, 21 "
                "requests, 10 min; evidence check: PASS: 4 claims), checks done."
            ),
            lines,
        )
        self.assertIn(f"Read first: {self.ws}/evidence_check.md", lines[0])
        self.assertIn("`cand ...` assessment lines", lines[0])
        d = self.st["drafts"][-1]
        self.assertEqual(
            (d["status"], d["late"], d["late_s"], d["findings"]),
            ("ready", True, 1000, [{"id": "N1", "severity": "🟠"}]),
        )
        self.assertEqual(draft_stage.draft_for(self.st, 1, "1" * 40)["status"], "ready")

    def test_a_moved_head_drops_and_records_the_draft(self):
        self.pipe()
        self.st["prs"]["1"]["head"] = "f" * 40
        self.status["d1"] = "running"
        lines = self.advance()
        self.assertTrue(lines[0].startswith("REVIEW_DROP #1"), lines)
        self.assertEqual(self.st["drafts"][-1]["status"], "dropped")


class TestDraftFor(unittest.TestCase):
    def test_the_last_record_of_the_statuses_and_the_role_asked_for(self):
        """draft_for: the last draft recorded for a PR at its head that came to verification, or of the statuses and
        the role asked for (STAGED: "none" too, a review staged with no drafter; a record that names no role is the
        drafter's)."""
        head = "1" * 40
        drafts = [
            {"pr": 1, "head": head, "role": "drafter", "status": "ready"},
            {"pr": 1, "head": head, "role": "shadow_drafter", "status": "missing"},
            {"pr": 1, "head": head, "status": "none"},
            {"pr": 1, "head": head, "role": "drafter", "status": "dropped"},
            {"pr": 1, "head": "2" * 40, "role": "drafter", "status": "ready"},
        ]
        st = {"drafts": drafts}
        self.assertIs(draft_stage.draft_for(st, 1, head), drafts[1])
        self.assertIs(draft_stage.draft_for(st, 1, head, role="drafter"), drafts[0])
        self.assertIs(
            draft_stage.draft_for(
                st, 1, head, statuses=draft_stage.STAGED, role="drafter"
            ),
            drafts[2],
        )
        self.assertIsNone(draft_stage.draft_for(st, 1, head, role="quality_drafter"))
        self.assertIsNone(draft_stage.draft_for(st, 2, head))


class TestPool(Base):
    def setUp(self):
        super().setUp()
        self.cfg = with_binding(self.work)
        self.plan = [
            {"kind": "FIRST_ROUND", "pr": n, "text": "x", "prio": 3} for n in (1, 2, 3)
        ]
        self.patches = [
            mock.patch.object(review, "_changed_files", lambda cfg, rec: ["proj/x.py"])
        ]

    def pool_advance(self):
        return self.run_patched(
            lambda: draft_stage.pool_advance(self.cfg, self.st, self.store, self.plan),
            *self.patches,
        )

    def test_one_draft_a_call_while_the_scaler_allows(self):
        self.st["develop_fixes"] = {"2": ["fixes develop"]}
        self.st["review_pipes"] = [
            {
                "pr": 1,
                "kind": "first-round",
                "stage": "worker",
                "head": "1" * 40,
                "worker": {"id": "w1"},
                "stage_name": "tool-loop",
            }
        ]
        self.status["w1"] = "running"
        self.assertEqual(
            self.pool_advance(),
            ["DRAFT_START #2 (first-round): pool draft at 2222222222"],
        )
        self.assertEqual(
            self.started[-1]["env"], {"RKMON_DRAFT_PRIO": "1"}
        )  # a develop fix ranks first
        self.assertEqual(
            self.pool_advance(),
            ["DRAFT_START #3 (first-round): pool draft at 3333333333"],
        )
        self.assertEqual(self.pool_advance(), [])  # nothing left to draft
        self.assertEqual(
            self.run_patched(lambda: draft_stage.running(self.st, self.store)), 3
        )
        self.assertEqual(sorted(self.st["draft_pool"]), ["2", "3"])

    def test_the_scaler_holds_and_stale_drafts_go(self):
        with mock.patch.object(
            scaler,
            "decide",
            lambda inp: (False, "the last 5 min used 85% of the budget"),
        ):
            self.assertEqual(
                self.pool_advance(),
                ["DRAFT_WAIT #1: the last 5 min used 85% of the budget"],
            )
        self.assertEqual(self.started, [])
        self.pool_advance()
        self.st["prs"]["1"]["head"] = "e" * 40
        lines = self.pool_advance()
        self.assertEqual(lines[0], "DRAFT_DROP #1: its head moved")
        self.assertEqual(self.st["drafts"][-1]["status"], "dropped")
        self.assertEqual(self.status["draft-pr1-0"], "failed(rc=143)")  # stopped

    def test_the_local_worker_runs_no_pool_and_the_ceiling_binds(self):
        self.cfg = a_cfg(self.work)
        self.assertEqual(self.pool_advance(), [])
        self.cfg = with_binding(self.work, draft_parallel_max=1)
        self.pool_advance()
        lines = self.pool_advance()
        self.assertEqual(lines, ["DRAFT_WAIT #2: 1 drafts running: the ceiling is 1"])

    def test_a_pool_draft_checks_out_what_a_review_s_worktrees_do(self):
        """A pool draft's worktrees check out the paths a review's do (review.sparse_extra): the tests' paths, then
        the top two levels of the PR's files outside the repository folder's top."""
        self.cfg["test_sparse"] = ["shared/ctest"]
        files = ["proj/x.py", "other/lib/a.cpp"]
        self.patches = [
            mock.patch.object(review, "_changed_files", lambda cfg, rec: files)
        ]
        self.assertEqual(
            self.pool_advance(),
            ["DRAFT_START #1 (first-round): pool draft at 1111111111"],
        )
        self.assertEqual(
            review.sparse_extra(self.cfg, files), ["shared/ctest", "other/lib"]
        )
        self.assertIn(
            "SPARSE_EXTRA='shared/ctest other/lib' ", self.started[-1]["script"]
        )


class TestScaler(Base):
    def test_the_policy_and_its_inputs(self):
        inp = {"budget": {"share": 0.5, "rate_limited": 0}, "running": 2, "ceiling": 6}
        self.assertEqual(scaler.decide(inp)[0], True)
        self.assertEqual(
            scaler.decide(dict(inp, running=6)),
            (False, "6 drafts running: the ceiling is 6"),
        )
        self.assertEqual(
            scaler.decide(dict(inp, budget={"share": 0.1, "rate_limited": 2}))[1],
            "2 HTTP 429 in the last 5 min",
        )
        self.assertIn(
            "used 80% of the budget", scaler.decide(dict(inp, budget={"share": 0.8}))[1]
        )
        mem = self.work / "meminfo"
        mem.write_text(
            "MemTotal:       1000 kB\nMemFree:  1 kB\nMemAvailable:    250 kB\n"
        )
        h = scaler.host_load({"max_host_usage_pct": 70}, str(mem))
        self.assertEqual((h["limit_pct"], h["mem_pct"]), (70, 75.0))
        self.st["user_priority"] = {"3": {}}
        q = scaler.queue_pressure(
            self.st, [{"kind": "FOLLOW_UP", "pr": 3}, {"kind": "MERGE", "pr": 1}]
        )
        self.assertEqual((q["backlog"], q["priorities"]), (1, 1))
        self.assertGreater(q["oldest_min"], 0)
        self.assertEqual(bindings.parallel_max(with_binding(self.work)), 6)
        self.assertEqual(
            bindings.parallel_max(with_binding(self.work, glm_parallel_max=2)), 2
        )


def with_shadow(work: pathlib.Path, **over) -> dict:
    """The local worker drafts live; the binding "cand" drafts every live review in the shadow."""
    return a_cfg(
        work,
        roles={"shadow_drafter": "cand"},
        model_bindings={"cand": dict(BINDING)},
        **over,
    )


class TestShadow(Base):
    """The shadow drafter: a binding that drafts every live review beside the live draft, never adopted, never waited
    for, recorded when it ends."""

    def setUp(self):
        super().setUp()
        self.cfg = with_shadow(self.work)
        self.patches = [
            mock.patch.object(review, "_record_checks_gates", lambda cfg, rec, n: None),
            mock.patch.object(review, "_changed_files", lambda cfg, rec: ["proj/x.py"]),
            mock.patch.object(review, "emitter_families", lambda tree, k: {}),
            mock.patch.object(review, "pick_next", lambda *a, **kw: None),
            mock.patch.object(review, "has_draft", lambda *a, **kw: True),
        ]

    def advance(self):
        return self.run_patched(
            lambda: review.advance(self.cfg, self.st, self.store, []), *self.patches
        )

    def live(self, n=1, **over):
        p = {
            "pr": n,
            "kind": "first-round",
            "stage": "worker",
            "head": str(n) * 40,
            "since": None,
            "started": OLD,
            "worker": {"id": f"w{n}", "started": OLD},
            "checks": {"id": f"c{n}", "started": OLD},
        }
        p.update(over)
        self.st["review_pipes"] = [p]
        self.status.update({f"w{n}": "running", f"c{n}": "running"})

    def shadow_done(self, n=1, method="phased"):
        ws = draft_stage.shadow_folder(self.cfg, n)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "review_agent.md").write_text("shadow draft\n")
        (ws / "draft_run.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "passes": {
                        "settle": {"status": "ok", "wall_s": 2.0},
                        "prior": {
                            "status": "ok",
                            "wall_s": 300.0,
                            "requests": 9,
                            "tokens": 40000,
                        },
                        "hunt": {
                            "status": "ok",
                            "wall_s": 1500.0,
                            "requests": 20,
                            "tokens": 150000,
                        },
                    },
                }
            )
        )
        (ws / "prior_settled.json").write_text(
            json.dumps(
                {
                    "head": str(n) * 40,
                    "counts": {
                        "rows": 5,
                        "script": 3,
                        "script_pass": 3,
                        "model": 2,
                        "model_pass": 1,
                        "look": 1,
                        "unsettled": 0,
                    },
                }
            )
        )
        self.status[self.st["shadow_pool"][str(n)]["job"]["id"]] = "done"
        return ws

    def test_it_drafts_every_live_review_and_is_never_adopted(self):
        self.live()
        lines = self.advance()
        self.assertIn("SHADOW_START #1 (first-round): cand at 1111111111", lines)
        job = self.started[-1]
        self.assertEqual(
            (job["kind"], job["timeout"], job["env"]),
            ("shadow", 110, {"RKMON_DRAFT_PRIO": "0"}),
        )
        self.assertIn("draft_run.py 1 --shadow cand " + "1" * 40, job["script"])
        self.assertIn(
            "REVIEW_LINE1='Automated review by rocke-monitor: Model - Cand-1, effort: high'",
            job["script"],
        )
        self.assertNotIn(
            "rm -rf", job["script"]
        )  # its own folder: the job wipes that itself
        self.assertEqual(
            self.advance(), ["REVIEW #1: worker running, checks running"]
        )  # one shadow a review
        self.assertEqual(self.st.get("draft_pool"), None)
        # The live review is ready before its shadow: REVIEW_READY says so, and waits for nothing.
        self.status.update(w1="done", c1="done")
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith(
                "REVIEW_READY #1 (first-round): worker done, checks done."
            ),
            lines,
        )
        self.assertIn("Shadow cand: not done (running since ", lines[0])
        self.assertEqual(self.st["review_ready"][0]["pr"], 1)
        # The shadow ends while its review awaits verification: SHADOW_READY names it.
        ws = self.shadow_done()
        lines = self.advance()
        self.assertTrue(
            any(x.startswith(f"SHADOW_READY #1: {ws}/review_agent.md") for x in lines),
            lines,
        )
        d = self.st["drafts"][-1]
        self.assertEqual(
            (d["role"], d["binding"], d["method"], d["status"], d["late"]),
            ("shadow_drafter", "cand", "phased", "ready", False),
        )
        self.assertEqual(
            (d["phases"]["hunt"]["tokens"], d["prior"]["model_pass"]), (150000, 1)
        )
        self.assertEqual(draft_stage.shadow_pool(self.st), {})
        self.assertEqual(
            draft_stage.shadow_note(self.cfg, self.st, 1, "1" * 40),
            f" Shadow cand: {ws}/review_agent.md (with evidence_check.md, prior_settled.md): once the post "
            "is final, compare it and write the `cand ...` lines.",
        )

    def test_one_that_ends_after_our_review_is_recorded_late(self):
        self.live()
        self.advance()
        self.st["review_pipes"] = []
        self.st["prs"]["1"]["ours"] = {"head": "1" * 40}  # our review covers its head
        self.shadow_done()
        lines = self.advance()
        self.assertIn("SHADOW_DONE #1: ready after our review (late)", lines)
        self.assertTrue(self.st["drafts"][-1]["late"])

    def test_one_whose_head_moved_is_dropped(self):
        self.live()
        self.advance()
        job = self.st["shadow_pool"]["1"]["job"]["id"]
        self.st["prs"]["1"]["head"] = "f" * 40
        lines = self.run_patched(
            lambda: draft_stage.shadow_collect(self.cfg, self.st, self.store)
        )
        self.assertEqual(lines, ["SHADOW_DROP #1: its head moved"])
        self.assertEqual(
            (self.status[job], self.st["drafts"][-1]["status"]),
            ("failed(rc=143)", "dropped"),
        )

    def test_the_scaler_holds_it(self):
        self.live()
        with mock.patch.object(
            scaler,
            "decide",
            lambda inp: (False, "the last 5 min used 85% of the budget"),
        ):
            lines = self.advance()
        self.assertIn("SHADOW_WAIT #1: the last 5 min used 85% of the budget", lines)
        self.assertEqual(self.started, [])
        rec = [
            json.loads(x)
            for x in (self.store.dir / scaler.LOG).read_text().splitlines()
        ][-1]
        self.assertEqual((rec["role"], rec["allow"]), ("shadow_drafter", False))

    def test_a_merge_only_head_gets_no_shadow_until_it_goes_to_verification(self):
        self.live(merge_only={"old": "0" * 40, "merges": 1})
        self.assertFalse(any(x.startswith("SHADOW_START") for x in self.advance()))

    def test_the_live_pool_goes_when_the_live_stage_runs_none(self):
        self.st["draft_pool"] = {
            "2": {
                "pr": 2,
                "kind": "first-round",
                "head": "2" * 40,
                "job": {"id": "pool2", "started": OLD},
                "started": OLD,
            }
        }
        self.status["pool2"] = "running"
        lines = self.run_patched(
            lambda: draft_stage.pool_advance(self.cfg, self.st, self.store, [])
        )
        self.assertEqual(
            lines,
            ["DRAFT_DROP #2: the live stage is local-worker: no pool draft feeds it"],
        )
        self.assertEqual(
            (
                self.status["pool2"],
                self.st["draft_pool"],
                self.st["drafts"][-1]["status"],
            ),
            ("failed(rc=143)", {}, "dropped"),
        )


class TestQualityLane(Base):
    """The code quality pass on a binding (roles.quality): the driver starts it once the PR is staged, REVIEW_READY
    waits for it and names its quality.md, or says the orchestrator's reviewer runs the lane.
    """

    def setUp(self):
        super().setUp()
        self.cfg = with_binding(self.work, roles={"quality": "cand"})
        self.ws = draft_stage.folder(self.cfg, 1)
        self.ws.mkdir(parents=True)
        self.patches = [
            mock.patch.object(review, "_record_checks_gates", lambda cfg, rec, n: None),
            mock.patch.object(review, "_changed_files", lambda cfg, rec: ["proj/x.py"]),
            mock.patch.object(review, "emitter_families", lambda tree, k: {}),
            mock.patch.object(review, "pick_next", lambda *a, **kw: None),
        ]

    def advance(self):
        return self.run_patched(
            lambda: review.advance(self.cfg, self.st, self.store, []), *self.patches
        )

    def pipe(self, **over):
        p = {
            "pr": 1,
            "kind": "first-round",
            "stage": "worker",
            "head": "1" * 40,
            "since": None,
            "started": OLD,
            "worker": {"id": "d1", "started": OLD},
            "checks": None,
            "stage_name": "tool-loop",
        }
        p.update(over)
        self.st["review_pipes"] = [p]
        return p

    def test_it_starts_once_staged_and_ready_waits_for_it(self):
        self.pipe()
        self.status["d1"] = "running"
        self.assertEqual(
            self.advance(), ["REVIEW #1: worker running, checks none"]
        )  # not staged: no lane yet
        self.assertNotIn("quality", review.pipe_of(self.st, 1))
        (self.ws / "staged").write_text("now\n")
        (self.work / "wt1m").mkdir()
        lines = self.advance()
        self.assertEqual(
            lines[0], "REVIEW #1: code quality pass started on cand (quality-pr1-0)"
        )
        self.assertTrue(
            lines[1].startswith("REVIEW #1: worker running; checks started"), lines
        )
        job = self.started[0]
        self.assertEqual((job["kind"], job["timeout"]), ("quality", 35))
        self.assertTrue(
            job["script"].rstrip().endswith("tools/quality_run.py 1"), job["script"]
        )
        self.assertIn(f"RKMON_WORKERS={self.work}/workers ", job["script"])
        # The draft and the checks end while the lane runs: REVIEW_READY waits for it.
        (self.ws / "review_agent.md").write_text("draft\n")
        self.status.update(
            {"d1": "done", review.pipe_of(self.st, 1)["checks"]["id"]: "done"}
        )
        self.assertEqual(
            self.advance(),
            [
                "REVIEW #1: worker done, checks done; the code quality pass runs (quality-pr1-0)"
            ],
        )
        # It ends with quality.md at this head: REVIEW_READY names it for review-prep and the verifier.
        q = review.quality_folder(self.cfg, 1)
        q.mkdir()
        (q / "quality.md").write_text(
            "| Code quality | 🟢 Clean<br>• one home per fact (read) |\n"
        )
        (q / "quality_run.json").write_text(
            json.dumps({"head": "1" * 40, "binding": "cand"})
        )
        self.status["quality-pr1-0"] = "done"
        lines = self.advance()
        self.assertTrue(lines[0].startswith("REVIEW_READY #1 (first-round): "), lines)
        self.assertTrue(
            lines[0].endswith(
                f" Code quality pass (cand): {q}/quality.md, done: `rk review-prep 1` puts "
                "it in the review directory; don't dispatch the code quality reviewer: the "
                "verifier folds it in."
            ),
            lines[0],
        )

    def test_ready_without_its_file_says_the_reviewer_runs_the_pass(self):
        self.pipe(
            checks={"id": "c1", "started": OLD}, quality={"id": "q1", "started": OLD}
        )
        self.status.update(d1="done", c1="done", q1="failed(rc=1)")
        (self.ws / "review_agent.md").write_text("draft\n")
        lines = self.advance()
        self.assertTrue(
            lines[0].endswith(
                " Code quality pass (cand): failed(rc=1), no quality.md at this head: "
                "dispatch the code quality reviewer as usual."
            ),
            lines,
        )

    def test_a_review_too_large_for_the_autopilot_names_no_pass(self):
        """Over autopilot_max_lines, REVIEW_READY's in-depth step says what becomes of the code quality pass (the split
        review, or a light later round): the note adds nothing, and never says to dispatch the code quality reviewer.
        """
        self.st["prs"]["1"]["size"] = 40000
        p = self.pipe(
            checks={"id": "c1", "started": OLD}, quality={"id": "q1", "started": OLD}
        )
        self.status.update(d1="done", c1="done", q1="failed(rc=1)")
        (self.ws / "review_agent.md").write_text("draft\n")
        line = self.advance()[0]
        self.assertTrue(line.startswith("REVIEW_READY #1 (first-round): "), line)
        self.assertIn(
            " 40,000 changed lines, too large for the autopilot: split the verification with "
            "`rk review-split 1`.",
            line,
        )
        self.assertNotIn("Code quality pass", line)
        self.assertNotIn("dispatch the code quality reviewer", line)
        p["kind"] = (
            "follow-up"  # a light later round: one verifier, no code quality pass
        )
        self.assertEqual(review.quality_note(self.cfg, self.st, self.store, p), "")

    def test_ready_keeps_whether_the_review_gets_the_pass(self):
        """REVIEW_READY keeps review.quality_runs's answer on its review_ready entry: what asks once the pipe is gone
        (`rk review-prep`, a mirror's `rk post`: quality_runs_for) reads that answer, whatever changed since.
        """
        for size, gets in ((10, True), (40000, False)):
            with self.subTest(size=size):
                self.st["prs"]["1"]["size"] = size
                self.st["review_ready"] = []
                self.pipe(
                    checks={"id": "c1", "started": OLD},
                    quality={"id": "q1", "started": OLD},
                )
                self.status.update(d1="done", c1="done", q1="done")
                (self.ws / "review_agent.md").write_text("draft\n")
                line = self.advance()[0]
                self.assertTrue(
                    line.startswith("REVIEW_READY #1 (first-round): "), line
                )
                self.assertIs(self.st["review_ready"][0]["quality_on_binding"], gets)
                self.st["prs"]["1"]["size"] = 50000 if gets else 10
                self.assertIs(review.quality_runs_for(self.cfg, self.st, 1), gets)

    def test_no_lane_for_a_review_too_large_for_the_autopilot(self):
        """Nothing uses the pass there (the split review, or a light later round): it does not start, and REVIEW_READY
        does not wait for it."""
        self.st["prs"]["1"]["size"] = 40000
        (self.ws / "staged").write_text("now\n")
        (self.work / "wt1m").mkdir()
        self.pipe()
        self.status["d1"] = "running"
        lines = self.advance()
        self.assertTrue(
            lines[0].startswith("REVIEW #1: worker running; checks started"), lines
        )
        self.assertNotIn("quality", review.pipe_of(self.st, 1))
        (self.ws / "review_agent.md").write_text("draft\n")
        self.status.update(
            {"d1": "done", review.pipe_of(self.st, 1)["checks"]["id"]: "done"}
        )
        line = self.advance()[0]
        self.assertTrue(line.startswith("REVIEW_READY #1 (first-round): "), line)
        self.assertNotIn("quality", [j["kind"] for j in self.started])

    def test_no_lane_without_a_binding_or_for_a_merge_only_head(self):
        (self.ws / "staged").write_text("now\n")
        (self.work / "wt1m").mkdir()
        for cfg, over in (
            (with_binding(self.work), {}),
            (self.cfg, {"merge_only": {"old": "0" * 40, "merges": 1}}),
        ):
            self.cfg = cfg
            p = self.pipe(**over)
            self.status["d1"] = "running"
            self.advance()
            self.assertNotIn("quality", review.pipe_of(self.st, 1))
            self.assertEqual(review.quality_note(cfg, self.st, self.store, p), "")
        self.assertNotIn("quality", [j["kind"] for j in self.started])

    def test_a_stopped_pipe_stops_its_lane(self):
        self.pipe(quality={"id": "q1", "started": OLD})
        self.status["q1"] = "running"
        self.run_patched(lambda: review.stop_pipe(self.st, self.store, 1))
        self.assertEqual(self.status["q1"], "failed(rc=143)")


class TestNoDraft(Base):
    """With no first-pass drafter (the verifier reviews alone): `worker_backend: "none"`. The driver stages the PR and
    runs its checks with no drafter, REVIEW_READY says so, line 1 and the footer name the verifier, and the assessment
    records no live drafter."""

    VERIFIER_LINE = (
        "Automated review by rocke-monitor: Model - Claude Opus 5.5, effort: high"
    )

    def cfg(self, **over):
        over = {"verifier_name": "Claude Opus 5.5", **over}
        return a_cfg(
            self.work, worker_backend="none", worker_model="qwen3.6:35b", **over
        )

    def test_the_verifier_is_the_audit_model_unless_named(self):
        from rkmon_lib import config

        cfg = a_cfg(self.work, worker_backend="none")
        self.assertEqual(bindings.verifier_name(cfg), config.claude_name(cfg, "audit"))
        self.assertEqual(
            bindings.first_line(cfg),
            f"Automated review by rocke-monitor: Model - {config.claude_name(cfg, 'audit')}, effort: high",
        )

    def test_the_mode_its_stage_and_line_1(self):
        for cfg in (
            self.cfg(),
            a_cfg(self.work, roles={"drafter": "none"}),
            a_cfg(self.work, draft_stage="none"),
        ):
            self.assertTrue(bindings.no_draft(cfg))
            self.assertIs(draft_stage.for_config(cfg), draft_stage.NONE)
        self.assertFalse(bindings.no_draft(a_cfg(self.work)))
        self.assertEqual(
            bindings.first_line(self.cfg()), self.VERIFIER_LINE
        )  # not review_first_line, the drafter's
        self.assertEqual(
            bindings.first_line(self.cfg(verifier_name="Claude Opus 6")),
            "Automated review by rocke-monitor: Model - Claude Opus 6, effort: high",
        )
        self.assertEqual(bindings.problems(self.cfg()), [])
        self.assertEqual(
            bindings.problems(a_cfg(self.work, roles={"drafter": "none"})), []
        )
        # A line 1 the configuration gives is not read, whatever it names: line 1 is the verifier's, with the name
        for given in (
            "Automated review: Model - Claude Opus 5.5, effort: max",
            "Automated review: Model - qwen3.6:35b, effort: high",
        ):
            with self.subTest(given=given):
                cfg = self.cfg(review_first_line_no_draft=given)
                self.assertEqual(
                    (bindings.first_line(cfg), bindings.problems(cfg)),
                    (self.VERIFIER_LINE, []),
                )
        self.assertEqual(
            bindings.describe(self.cfg())[:2],
            [
                f"first pass: default: the verifier (Claude Opus 5.5) reviews alone, with no first-pass drafter; "
                f"line 1: {self.VERIFIER_LINE}",
                f"drafter: default: no drafter, the verifier reviews alone; line 1: {self.VERIFIER_LINE}",
            ],
        )
        # The assessment scores no live drafter; a shadow drafter still drafts and is scored.
        self.assertEqual(bindings.candidates(self.cfg()), [])
        shadow = self.cfg(
            roles={"shadow_drafter": "cand"}, model_bindings={"cand": dict(BINDING)}
        )
        self.assertEqual(
            [c["role"] for c in bindings.candidates(shadow)], ["shadow_drafter"]
        )

    def test_the_driver_stages_checks_and_hands_over_with_no_draft(self):
        cfg = self.cfg()
        patches = [
            mock.patch.object(review, "_record_checks_gates", lambda cfg, rec, n: None),
            mock.patch.object(review, "_changed_files", lambda cfg, rec: ["proj/x.py"]),
            mock.patch.object(review, "emitter_families", lambda tree, k: {}),
            mock.patch.object(review, "pick_next", lambda *a, **kw: None),
            mock.patch.object(
                review.gh,
                "rest",
                lambda path, **k: {
                    "state": "open",
                    "head": path.rsplit("/", 1)[1] * 40,
                },
            ),
        ]
        lines = self.run_patched(
            lambda: review.start_review(cfg, self.st, self.store, 1, "first-round"),
            *patches,
        )
        self.assertEqual(
            lines[-1],
            "REVIEW_START #1 (first-round): staging started at 1111111111 (parallel review 1 of 3)",
        )
        job = self.started[-1]
        self.assertTrue(
            job["script"].rstrip().endswith("draft_run.py 1 --stage-only"),
            job["script"],
        )
        self.assertIn(f"REVIEW_LINE1='{self.VERIFIER_LINE}'", job["script"])
        self.assertNotIn("ollama stop", job["script"])  # no local model to unload
        ws = draft_stage.folder(cfg, 1)
        ws.mkdir(parents=True)
        (ws / "staged").write_text("now\n")
        (self.work / "wt1m").mkdir()
        self.status[job_id := review.pipe_of(self.st, 1)["worker"]["id"]] = "done"
        lines = self.run_patched(
            lambda: review.advance(cfg, self.st, self.store, []), *patches
        )
        self.assertTrue(
            lines[0].startswith("REVIEW #1: worker done; checks started"), lines
        )
        self.status[review.pipe_of(self.st, 1)["checks"]["id"]] = "done"
        lines = self.run_patched(
            lambda: review.advance(cfg, self.st, self.store, []), *patches
        )
        self.assertEqual(
            lines[0],
            f"REVIEW_READY #1 (first-round): no drafter: verify without a draft, checks done. "
            f"Inputs: {ws}/checks/checks.md, {ws}/pr/meta.md. Verify alone (line 1 and the "
            "footer name the verifier, Claude Opus 5.5; no first pass), then `rk post 1 FILE "
            "--premises SPEC`",
        )
        self.assertNotIn("lint", lines[0])
        self.assertEqual(
            (self.st["drafts"][-1]["status"], self.st["drafts"][-1]["binding"], job_id),
            ("none", "default", job_id),
        )
        self.assertEqual([r["pr"] for r in self.st["review_ready"]], [1])

    def test_the_cost_and_the_assessment_record_no_drafter(self):
        from rkmon_lib import assessment, review_cost

        cfg = self.cfg()
        # The review staged with no drafter: the record its stage keeps (review.advance: draft_stage.record, none).
        pipe = {"kind": "first-round", "head": "1" * 40}
        draft_stage.record(cfg, self.st, self.store, 1, pipe, "none")
        draft, shadow = review_cost.drafts(cfg, self.st, 1, "1" * 40)
        self.assertEqual(
            (draft["by"], draft["method"], draft["status"], shadow),
            ("none", "none", "none", None),
        )
        notes = self.work / "notes.md"
        notes.write_text("qwen none\nglm none\n")
        out = assessment.record(
            cfg, self.store, 1, {"NOTES": str(notes), "ROUND": "fr"}, {"sha": "1" * 40}
        )
        self.assertEqual(out, [])
        self.assertFalse((self.store.dir / assessment.FILE).exists())


class TestStageOurs(Base):
    def test_our_last_post_is_staged_for_the_prior_items(self):
        from rkmon_lib import gh, post

        cfg = a_cfg(self.work)
        out = self.work / "pr"
        self.assertEqual(
            draft_stage.stage_ours(cfg, self.st, 1, out),
            "no review of ours on #1: nothing staged",
        )
        self.st["prs"]["1"]["ours"] = {
            "head": "a" * 40,
            "at": "2026-11-02T09:00:00Z",
            "url": "https://github.com/ROCm/r/pull/1#issuecomment-5",
        }
        with mock.patch.object(
            post,
            "our_post_body",
            lambda cfg, ours: "## PR #1 — t\n\n| Prior item | Status |\n",
        ):
            line = draft_stage.stage_ours(cfg, self.st, 1, out)
        self.assertEqual(
            line,
            "#1: our last post (at aaaaaaaaaa, 2026-11-02T09:00:00Z) staged from GitHub",
        )
        text = (out / "ours.md").read_text()
        self.assertTrue(
            text.startswith(
                f"# Our last review post on PR #1\n- Head: {'a' * 40}\n- Posted: "
                "2026-11-02T09:00:00Z\n\n## Review\n## PR #1 — t"
            ),
            text,
        )

        def down(cfg, ours):
            raise gh.GhError("HTTP 502")

        with mock.patch.object(post, "our_post_body", down), mock.patch.object(
            post, "copies", lambda cfg, n: []
        ):
            self.assertEqual(
                draft_stage.stage_ours(cfg, self.st, 1, out),
                "#1: our last post's body was not read: nothing staged",
            )

    def test_without_github_s_body_a_post_is_read_from_its_newest_copy(self):
        """Our last post's body is GitHub's, the comment as it stands (post.our_last_review). When GitHub gives none
        (a deleted comment) or cannot be read, the newest copy of our posts on the PR stands in: never a review kept
        unposted since, nor another PR's copy."""
        from rkmon_lib import gh, post

        cfg = a_cfg(self.work)
        ours = {
            "head": "a" * 40,
            "at": "2026-11-02T09:00:00Z",
            "url": "https://github.com/ROCm/r/pull/1#issuecomment-5",
        }
        d = post.posted_dir(cfg)
        d.mkdir(parents=True, exist_ok=True)
        names = ("PR_FIRST_ROUND_1.md", "PR_FOLLOWUP_1.md", "PR_UNPOSTED_1.md")
        for t, name in enumerate(names + ("PR_FOLLOWUP_11.md",), 1):
            (d / name).write_text(f"## PR #1 — {name}\n", encoding="utf-8")
            os.utime(d / name, (t, t))

        def down(cfg, ours):
            raise gh.GhError("HTTP 502")

        copy = ("## PR #1 — PR_FOLLOWUP_1.md\n", "PR_FOLLOWUP_1.md")
        for body, got in (
            (lambda cfg, ours: "## PR #1 — t\n", ("## PR #1 — t\n", "GitHub")),
            (lambda cfg, ours: "", copy),
            (down, copy),
        ):
            with mock.patch.object(post, "our_post_body", body):
                self.assertEqual(post.our_last_review(cfg, 1, ours), got)
        self.st["prs"]["1"]["ours"] = ours
        with mock.patch.object(post, "our_post_body", down):
            self.assertEqual(
                draft_stage.stage_ours(cfg, self.st, 1, self.work / "pr"),
                "#1: our last post (at aaaaaaaaaa, 2026-11-02T09:00:00Z) staged from "
                "PR_FOLLOWUP_1.md",
            )

    def test_a_review_kept_unposted_is_read_from_its_file(self):
        """Posting off: our last review is the one post.save_draft kept, read from the file its record names, not the
        newest file on the PR (here one kept later by hand), and never from GitHub, which holds no post of ours. The
        draft's ours.md, the merge-only carry (review.our_last_review_body) and the facts read it there.
        """
        from rkmon_lib import facts, post

        cfg = a_cfg(self.work)
        self.st["review_ready"] = [
            {"pr": 1, "kind": "first-round", "head": "1" * 40, "at": OLD}
        ]
        verified = self.work / "pr1_verified.md"
        verified.write_text("## PR #1 — t\n\n| # | Finding |\n", encoding="utf-8")
        kept = pathlib.Path(post.save_draft(cfg, self.st, self.store, 1, str(verified)))
        verified.write_text("## PR #1 — kept by hand\n", encoding="utf-8")
        # No review awaits verification now: this one is not recorded as our last review.
        later = pathlib.Path(
            post.save_draft(cfg, self.st, self.store, 1, str(verified))
        )
        os.utime(later, (kept.stat().st_mtime + 60,) * 2)
        at = self.st["prs"]["1"]["ours"]["at"]
        out = self.work / "pr"
        github = mock.patch.object(
            post, "our_post_body", side_effect=AssertionError("read from GitHub")
        )
        with github:
            self.assertEqual(
                review.our_last_review_body(cfg, self.st["prs"]["1"], 1),
                "## PR #1 — t\n\n| # | Finding |\n",
            )
            self.assertEqual(
                facts._last_findings(cfg, self.st, self.store, 1, {}),
                [f"from {kept.name}", "## PR #1 — t"],
            )
            self.assertEqual(
                draft_stage.stage_ours(cfg, self.st, 1, out),
                f"#1: our last review (kept unposted at 1111111111, {at}) staged from "
                f"{kept.name}",
            )
        self.assertIn("\n## Review\n## PR #1 — t\n", (out / "ours.md").read_text())

    def test_a_review_kept_unposted_is_staged_as_our_last_review(self):
        """ours.md names a kept review for what it is, our last review, kept unposted at a time instead of posted (the
        drafter's prior items read that time, and that it was kept: test_prior_settle).
        """
        from rkmon_lib import post

        cfg = a_cfg(self.work)
        self.st["review_ready"] = [
            {"pr": 1, "kind": "first-round", "head": "1" * 40, "at": OLD}
        ]
        verified = self.work / "pr1_verified.md"
        verified.write_text("## PR #1 — t\n", encoding="utf-8")
        post.save_draft(cfg, self.st, self.store, 1, str(verified))
        at = self.st["prs"]["1"]["ours"]["at"]
        draft_stage.stage_ours(cfg, self.st, 1, self.work / "pr")
        text = (self.work / "pr" / "ours.md").read_text()
        self.assertEqual(
            text,
            f"# Our last review on PR #1\n- Head: {'1' * 40}\n- Kept unposted: {at}\n\n"
            "## Review\n## PR #1 — t\n",
        )


if __name__ == "__main__":
    unittest.main()
