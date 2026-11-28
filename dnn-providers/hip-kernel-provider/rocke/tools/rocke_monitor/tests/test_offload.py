# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review offload (the reviewer model does the reviews, first rounds included, and Claude audits them; the
trial is a shadow comparison): its settings, the passes through the fake gateway with
evidence for every claim, the sandbox, `rk evidence-check`, the pipes' finish stage and their REVIEW_READY lines, the
budget shares of the passes, the token records, `rk shadow-report` and rk doctor. Fakes only: the
gateway is tests/fakes/fake_models.py on 127.0.0.1, GitHub is mocked, rk is a stub script; no model is called.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(HERE / "fakes"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import fake_models  # noqa: E402
import home_repository  # noqa: E402
from sandbox import own_home  # noqa: E402
import reviewer_client  # noqa: E402
import offload_agent  # noqa: E402
import reviewer_tools as rt  # noqa: E402
from rkmon_lib import offload_evidence as evidence  # noqa: E402
import review_sandbox as sandbox  # noqa: E402
from rkmon_lib import (
    config,
    facts,
    gh,
    offload,
    review,
    shadow,
    verifier_file,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

KEY = "k-" + "fake-gateway-0451"  # a planted key, built at run time
OLD = "2000-01-01T00:00:00Z"
X_BASE = "def answer():\n    return 42\n"
X_HEAD = "def answer():\n    return 41\n\n\ndef helper(v):\n    return v + 1\n"
RK_STUB = """#!/bin/sh
# A stand-in for rk: the read-only commands the offload passes and the evidence check run.
case "$1" in
  review-prep)
    out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
    mkdir -p "$out"
    printf '== our last review\\n(none)\\n== rk abi\\n' > "$out/facts.txt"
    echo '{"kind": "never reviewed"}' > "$out/classify.txt"
    echo prepared ;;
  overlaps) echo "## Overlapping work in other open PRs"; echo "none" ;;
  ci-fails) printf 'head abc\\nFAIL\\tbuild (gfx942)\\tfailure\\thttps://example.com/job/12\\nruns 3 names 3\\n' ;;
  *) echo "stub rk: $*" ;;
esac
"""
DRAFT = """## Draft: PR #7 — t — 🟠 First-round: off by one

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟠 | `a/b/rocke/x.py:2` | **answer returns 41 instead of 42**<br>• impact: wrong answer<br>• fix: return 42 |
| N2 | 🟡 | `a/b/rocke/x.py:5` | **helper has no test**<br>• impact: untested<br>• fix: add one |

| # | Check | Result |
|---|---|---|
| D1 | grep for the constant | 🟢 found (ran) |

**Utility:** 🟡 Mixed — the change returns the wrong answer for now
"""
QUALITY = """| Code quality | 🟠 Should-fix<br>• one duplicate helper (read) |

| # | Sev | file:line | Finding |
|---|---|---|---|
| Q1 | 🟠 | `a/b/rocke/x.py:5` | **helper duplicates an existing increment**<br>• evidence: x<br>• fix: reuse it |
"""
REGRADE = """| # | Prior item | Status |
|---|---|---|
| P1 | a bot (11-05) rename the helper | 🟢 None yet |

probes: none
"""
POST = """Automated review: Model - m, effort: max

## PR #7 — t (a) — 🟠 First-round: off by one

| Prior item | Status |
|---|---|
| a bot (11-05) rename the helper | 🟢 None yet |
| Bots / CI | 🔴 FAIL<br>• fails: build (gfx942) (ran) |
| Our checks | 🟢 PASS<br>• merge clean (read) |
| Code quality | 🟠 Should-fix<br>• one duplicate helper |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟠 | `a/b/rocke/x.py:2` | **answer returns 41 instead of 42**<br>• impact: wrong answer<br>• fix: return 42 |
| N2 | 🟠 | `a/b/rocke/x.py:5` | **helper duplicates an existing increment**<br>• evidence: x<br>• fix: reuse it |

**Utility:** 🟡 Mixed — the change returns the wrong answer for now
=====NOTES=====
map: N1 = N1
map: N2 = Q1
premises: head x
"""
CHECKS = (
    "# Checks for #7\n\n## Merge with develop\n\nmerge abc = head def + develop 123\n"
)
SCRIPT = {
    "INDEPENDENT reviewer of one pull request": [
        {
            "call": "read_lines",
            "args": {"path": "a/b/rocke/x.py", "start": 1, "end": 6},
        },
        {"call": "grep_code", "args": {"pattern": "return 41"}},
        {
            "call": "record_evidence",
            "args": {
                "id": "N1",
                "severity": "🟠",
                "claim": "answer returns 41",
                "file": "a/b/rocke/x.py",
                "line": 2,
                "quote": "return 41",
            },
        },
        {
            "call": "record_evidence",
            "args": {
                "id": "N1",
                "severity": "🟠",
                "claim": "grep finds it",
                "run": "R1",
            },
        },
        {
            "call": "record_evidence",
            "args": {
                "id": "N2",
                "severity": "🟡",
                "claim": "wrong line",
                "file": "a/b/rocke/x.py",
                "line": 3,
                "quote": "def helper(v):",
            },
        },
        {"text": DRAFT},
    ],
    "code quality pass of the review": [
        {
            "call": "record_evidence",
            "args": {"id": "N1", "claim": "not a quality id", "run": "R1"},
        },
        {"text": QUALITY},
    ],
    "re-grade the prior items": [{"text": REGRADE}],
    "assemble the post of one rocKE PR review": [
        {
            "call": "read_lines",
            "args": {"path": "checks.md", "tree": "checks", "start": 1, "end": 5},
        },
        {"call": "ci_fails", "args": {}},
        {
            "call": "record_evidence",
            "args": {"id": "C1", "severity": "🔴", "claim": "build fails", "run": "R2"},
        },
        {
            "call": "record_evidence",
            "args": {
                "id": "C2",
                "severity": "🟢",
                "claim": "merge clean",
                "tree": "checks",
                "file": "checks.md",
                "line": 5,
                "quote": "merge abc = head def + develop 123",
            },
        },
        {"text": POST},
    ],
}


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make_trees(root: pathlib.Path) -> str:
    """wt7, a repository whose develop holds X_BASE and whose head (the PR) X_HEAD; wt7m, a clone at the head. The head's
    sha."""
    wt = root / "wt7"
    (wt / "a" / "b" / "rocke").mkdir(parents=True)
    git(root, "init", "-q", "-b", "develop", str(wt))
    for k, v in (
        ("user.name", "t"),
        ("user.email", "t@example.com"),
        ("commit.gpgsign", "false"),
    ):
        git(wt, "config", k, v)
    (wt / "a" / "b" / "rocke" / "x.py").write_text(X_BASE, encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "base")
    git(wt, "update-ref", "refs/remotes/origin/develop", "HEAD")
    git(wt, "checkout", "-q", "-b", "pr")
    (wt / "a" / "b" / "rocke" / "x.py").write_text(X_HEAD, encoding="utf-8")
    git(wt, "commit", "-q", "-am", "the change")
    git(root, "clone", "-q", str(wt), str(root / "wt7m"))
    return git(wt, "rev-parse", "HEAD")


def a_cfg(work: pathlib.Path, **over) -> dict:
    c = config.load_defaults()
    c.pop("_doc", None)
    c.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"]},
    )
    c["paths"] = dict(
        c["paths"],
        working_dir=str(work),
        repo=str(work / "wt7"),
        package=str(work / "toolkit"),
    )
    for k, v in over.items():
        c[k] = (
            dict(c.get(k) or {}, **v)
            if isinstance(v, dict) and isinstance(c.get(k), dict)
            else v
        )
    return config.normalize(c)


class _Work(unittest.TestCase):
    """A working folder with the PR's trees, a stub rk, rendered team members and the job's RKMON_* environment."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.head = make_trees(self.root)
        tk = self.root / "toolkit"
        tk.mkdir()
        (tk / "rk").write_text(RK_STUB, encoding="utf-8")
        (tk / "rk").chmod(0o755)
        tm = self.root / "agentic" / "team_members"
        tm.mkdir(parents=True)
        (tm / "rocke_code_quality_reviewer.md").write_text(
            "# The quality lane\n\n### Output Format\n\nquality.md\n", encoding="utf-8"
        )
        (tm / "rocke_review_verifier.md").write_text(
            "# The verifier\n\n3. Prior items.\n\n### Output Format\n\nposts\n",
            encoding="utf-8",
        )
        self.cfg = a_cfg(self.root, offload={"mode": "on"})
        self.env = mock.patch.dict(
            os.environ, {k: v for k, v in config.env(self.cfg).items() if v is not None}
        )
        self.env.start()
        self.ws = pathlib.Path(config.paths.get(self.cfg, "workers")) / "pr7"
        (self.ws / "pr").mkdir(parents=True)
        (self.ws / "pr" / "meta.md").write_text(
            "# PR #7: t\n- Round: First-round\n", encoding="utf-8"
        )
        (self.ws / "pr" / "diff.patch").write_text(
            git(self.root / "wt7", "diff", "develop", "HEAD") + "\n", encoding="utf-8"
        )
        (self.ws / "pr" / "files.txt").write_text(
            "     6 +     1 -  a/b/rocke/x.py\n", encoding="utf-8"
        )
        (self.ws / "review_book.md").write_text("# The review book\n", encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class TestSettings(unittest.TestCase):
    def test_the_defaults_are_the_packages_and_off(self):
        d = config.load_defaults()
        self.assertEqual(d["offload"], offload.DEFAULTS)
        self.assertEqual(
            (d["models"]["audit"], d["models"]["deep"], d["claude"]["effort"]),
            ("opus", "opus", "high"),
        )
        cfg = a_cfg(pathlib.Path("/w"))
        self.assertEqual(offload.mode(cfg), "off")
        self.assertEqual(offload.problems(cfg), [])
        self.assertEqual(config.problems(cfg), [])

    def test_bad_settings_are_named(self):
        bad = a_cfg(
            pathlib.Path("/w"),
            offload={
                "mode": "sometimes",
                "spot_check": 2,
                "budget_shares": {"draft": 0},
                "trial_prs": 0,
                "deep_paths": ["("],
                "colour": 1,
            },
            models={"audit": "worker"},
            claude={"effort": "huge"},
        )
        text = "\n".join(config.problems(bad))
        for words in (
            "offload.mode is 'sometimes'",
            "offload.spot_check is 2",
            "offload.budget_shares must give",
            "offload.trial_prs is 0",
            "offload.deep_paths: '(' is not a regular expression",
            "offload.colour is not a setting",
            "features.review_automation.passes.verification_pass.model must be a Claude model",
            "claude.effort is 'huge'",
        ):
            self.assertIn(words, text)

    def test_the_footer_may_name_any_of_claudes_models(self):
        cfg = a_cfg(pathlib.Path("/w"), orchestrator={"model": "sonnet"})
        self.assertEqual(
            config.model_for(cfg, "audit"), {"by": "claude", "model": "opus"}
        )  # Opus, whatever runs
        self.assertEqual(
            (config.claude_model(cfg, "deep"), config.claude_name(cfg, "audit")),
            ("opus", "Claude Opus"),
        )
        own = a_cfg(
            pathlib.Path("/w"),
            models={"audit": "claude-sonnet-5-5", "deep": "claude"},
            orchestrator={"model": "claude-opus-5-5"},
            effort={"orchestrator": "max"},
        )
        self.assertEqual(config.claude_model(own, "audit"), "claude-sonnet-5-5")
        self.assertEqual(config.claude_name(own, "audit"), "Claude Sonnet 5.5")
        self.assertEqual(
            config.claude_model(own, "deep"), "claude-opus-5-5"
        )  # "claude": the orchestrator's
        self.assertEqual(config.role_effort(own, "orchestrator")[0], "max")
        from rkmon_lib import premises

        self.assertEqual(
            premises.verifier_names(own), {"Claude Sonnet 5.5", "Claude Opus 5.5"}
        )

    def test_the_jobs_get_the_mode_and_the_passes_settings(self):
        cfg = a_cfg(
            pathlib.Path("/w"),
            offload={
                "mode": "shadow",
                "budget_shares": {"draft": 2, "quality": 1, "regrade": 1},
            },
            worker={"model": "m", "url": "http://127.0.0.1:9"},
        )
        env = config.env(cfg)
        self.assertEqual(env["RKMON_OFFLOAD_MODE"], "shadow")
        o = json.loads(env["RKMON_OFFLOAD_JSON"])
        self.assertEqual(
            (o["mode"], o["budget_shares"]["draft"], o["audit_name"]),
            ("shadow", 2, "Claude Opus"),
        )
        self.assertEqual(
            o["line1"],
            f"Automated review by rocke-monitor: Model - m, effort: {config.MODEL_DEFAULT}",
        )  # worker.think unset: the local review loop sends no effort
        self.assertEqual(
            (env["RKMON_AGENTIC"], env["RKMON_PROBES"]), ("/w/agentic", "/w/probes")
        )
        self.assertIn(
            "RKMON_OFFLOAD_ROUND=fu2",
            review.worker_script(cfg, 7, [], "follow-up", rnd="fu2"),
        )
        self.assertIn(
            "draft 50%, quality 25%, regrade 25%", offload.shares_line(cfg["offload"])
        )


class TestBudgetShares(unittest.TestCase):
    """The passes' shares of the reviewer endpoint's one budget: a role under its share goes before one that has used
    it; alone, a role may use the whole budget; the planner's and the analyst's capped shares still hold.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = pathlib.Path(self.tmp.name)
        self.t, self.said = [1000.0], []
        self.s = {"url": "https://models.example.com/route", "model": "m", "rpm": 2}
        self.shares = {"draft": 0.5, "quality": 0.25, "regrade": 0.25}

    def tearDown(self):
        self.tmp.cleanup()

    def budget(self, role, sleep=None, **limits):
        def tick(s):
            self.t[0] += s

        return reviewer_client.Budget(
            dict(self.s, **limits),
            str(self.run),
            log=self.said.append,
            clock=lambda: self.t[0],
            sleep=sleep or tick,
            who=role,
            role=role,
            shares=self.shares,
        )

    def state(self, **st) -> pathlib.Path:
        path = self.budget("draft").path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict({"window": [], "blocked_until": 0.0, "queue": []}, **st)),
            encoding="utf-8",
        )
        return path

    def test_a_role_under_its_share_goes_before_one_that_used_it(self):
        t0 = self.t[0]
        # draft has used its share (1 of 2 requests a minute) and waits since t0; quality asks later and goes first.
        self.state(
            window=[[t0 - 5, 10, "d0", "draft"]],
            queue=[["d-waiting", t0 - 1, os.getpid(), "draft"]],
        )
        self.assertEqual(self.budget("quality").acquire(10), 0)
        # Without shares, the same request waits behind the older one (first come, first served).
        self.state(
            window=[[t0 - 5, 10, "d0", "draft"]],
            queue=[["d-waiting", t0 - 1, os.getpid(), "draft"]],
        )
        plain = reviewer_client.Budget(
            self.s,
            str(self.run),
            log=self.said.append,
            clock=lambda: self.t[0],
            sleep=lambda s: (_ for _ in ()).throw(TimeoutError("waited")),
        )
        with self.assertRaises(TimeoutError):
            plain.acquire(10)

    def test_a_role_over_its_share_waits_for_the_others_and_says_why(self):
        t0 = self.t[0]
        path = self.state(
            window=[[t0 - 5, 10, "d0", "draft"]],
            queue=[["q-waiting", t0 + 1, os.getpid(), "quality"]],
        )

        def sleep(s):  # the quality request is served while the draft waits
            st = json.loads(path.read_text(encoding="utf-8"))
            st["queue"] = [q for q in st["queue"] if q[0] != "q-waiting"]
            st["window"].append([self.t[0], 10, "q1", "quality"])
            path.write_text(json.dumps(st), encoding="utf-8")
            self.t[0] += 61 + s

        self.assertGreater(self.budget("draft", sleep=sleep).acquire(10), 0)
        self.assertTrue(
            any("the draft share is used" in s for s in self.said), self.said
        )

    def test_alone_a_role_may_use_the_whole_budget(self):
        d = self.budget(
            "regrade"
        )  # a quarter of 2 requests a minute, but nobody else waits
        self.assertEqual([d.acquire(10), d.acquire(10)], [0, 0])
        self.assertEqual(
            reviewer_client.shares_of(
                {"draft": 2, "quality": 1, "regrade": 1, "x": 0, "y": "no"}
            ),
            {"draft": 0.5, "quality": 0.25, "regrade": 0.25},
        )

    def test_the_planners_capped_share_still_holds(self):
        def stop(_s):
            raise TimeoutError("waited")

        planner = reviewer_client.Budget(
            dict(self.s, rpm=4, budget_role="planner", budget_share=0.25),
            str(self.run),
            log=self.said.append,
            clock=lambda: self.t[0],
            sleep=stop,
            shares=self.shares,
        )
        self.assertEqual(planner.role, "planner")
        self.assertEqual(planner.acquire(10), 0)
        with self.assertRaises(
            TimeoutError
        ):  # its one request a minute (4 x 0.25) is spent
            planner.acquire(10)
        self.assertTrue(
            any(
                "the planner's share: 1 of 1 requests in the last minute" in s
                for s in self.said
            ),
            self.said,
        )


class TestTokens(unittest.TestCase):
    def test_each_request_and_each_review_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            keyfile = base / "key"
            keyfile.write_text(KEY, encoding="utf-8")
            srv = fake_models.start(
                base / "fake", "gateway", key_file=str(keyfile), info=False
            )
            try:
                s = {
                    "backend": "openai",
                    "url": srv.url,
                    "model": fake_models.GOOD,
                    "auth_header": "X-Example-Key",
                    "auth_scheme": "",
                    "api_key_file": str(keyfile),
                    "temperature": 0.3,
                    "max_output_tokens": 100,
                    "stall_s": 30,
                    "think": "max",
                }
                log = reviewer_client.TokenLog(base / "ws", 7, mode="on", round="fr")
                got = []
                budget = reviewer_client.Budget(
                    s,
                    str(base / "run"),
                    who="pr7",
                    role="quality",
                    shares={"quality": 1},
                )
                budget.tags = {"pass": "quality", "step": 3}
                msg = reviewer_client.chat(
                    s,
                    [{"role": "user", "content": "hi"}],
                    budget=budget,
                    on_usage=lambda r: (got.append(r), log.step(r, "quality", 3)),
                )
            finally:
                srv.stop()
            self.assertNotIn("usage", msg)
            r = got[0]
            self.assertEqual(
                (
                    r["reported"],
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["reasoning_tokens"],
                ),
                (True, 1000, 300, 250),
            )
            line = log.close(head="abc")
            self.assertEqual(line["passes"]["quality"]["requests"], 1)
            self.assertEqual(line["passes"]["all"]["prompt_tokens"], 1000)
            self.assertEqual(
                json.loads((base / "ws" / "tokens_total.json").read_text())["round"],
                "fr",
            )
            row = json.loads(
                (base / "run" / reviewer_client.TOKENS_LOG).read_text(encoding="utf-8")
            )
            self.assertEqual(
                (row["who"], row["role"], row["pass"], row["step"], row["prompt"]),
                ("pr7", "quality", "quality", 3, 1000),
            )
            self.assertNotIn("cost", json.dumps(line) + json.dumps(row))


class TestReviewTools(_Work):
    def test_quotes_lines_and_trees(self):
        t = rt.Tree(self.ws, 7)
        self.assertIn("     2|     return 41", rt.read_lines(t, "a/b/rocke/x.py", 1, 3))
        self.assertIn(
            "     2|     return 42",
            rt.read_lines(t, "a/b/rocke/x.py", 1, 3, tree="base"),
        )
        self.assertIn("no such file", rt.read_lines(t, "../etc/passwd"))
        lines = rt.file_lines(t, "a/b/rocke/x.py")
        self.assertEqual(rt.quote_at(lines, 2, "return 41"), (True, 2))
        self.assertEqual(rt.quote_at(lines, 3, "def helper(v):"), (False, 5))
        self.assertEqual(
            rt.quote_at(lines, 5, "def helper(v):\n    return v + 1"), (True, 5)
        )
        self.assertEqual(
            rt.quote_at(lines, 6, "v + 1"), (False, None)
        )  # a part under 12 characters is no quote
        self.assertEqual(t.sha("head"), self.head)
        self.assertEqual(t.sha("base"), git(self.root / "wt7", "rev-parse", "develop"))
        argv, _cwd, _t, _env = rt.git_log_cmd(t, "a/b/rocke/x.py")
        self.assertIn(f"{t.sha('base')}..HEAD", argv)
        self.assertIn("bad path", rt.git_diff_cmd(t, "/etc/passwd"))

    def test_what_a_rerun_must_keep(self):
        out = "..F.\nFAILED a/b/test_x.py::test_one - assert 1\n1 failed, 3 passed, 2 skipped in 3.21s\n"
        again = out.replace("3.21s", "9.99s")
        self.assertEqual(
            rt.normalize("run_pytest", out), rt.normalize("run_pytest", again)
        )
        self.assertEqual(
            json.loads(rt.normalize("run_pytest", out)),
            {
                "counts": {"failed": 1, "passed": 3, "skipped": 2},
                "failing": ["a/b/test_x.py::test_one"],
            },
        )
        t = rt.Tree(self.ws, 7)
        self.assertEqual(
            rt.normalize("grep_code", f"{t.wt}/b:2:y\n{t.wt}/a:1:x\n", t),
            "a:1:x\nb:2:y",
        )
        self.assertEqual(
            rt.normalize(
                "ci_fails",
                "head 1\nFAIL\tb\tfailure\tu\nRUN\ta\tqueued\tv\nruns 2 names 2\n",
            ),
            "FAIL\tb\nRUN\ta",
        )

    def test_the_models_own_code_runs_only_in_a_sandbox(self):
        t = rt.Tree(self.ws, 7)
        (self.ws / "probes").mkdir()
        (self.ws / "probes" / "mine.sh").write_text("exit 1\n", encoding="utf-8")
        (self.ws / "mutants").mkdir()
        (self.ws / "mutants" / "rkmut_m.py").write_text("x = 1\n", encoding="utf-8")
        with mock.patch.object(sandbox, "_KIND", ""):
            self.assertIn(
                "refused: the model's own probes run only in a sandbox",
                rt.probe_cmd(t, "mine"),
            )
            self.assertIn(
                "refused: mutants run only in a sandbox",
                rt.mutant_cmd(t, "m", "a/b/rocke/x.py"),
            )
        with mock.patch.object(sandbox, "_KIND", "bwrap"):
            argv, _cwd, _timeout, extra = rt.probe_cmd(t, "mine", "head", 30)
            self.assertEqual(
                argv[:6], ["timeout", "-k", "10", "30", "bwrap", "--ro-bind"]
            )
            self.assertIn("--unshare-net", argv)
            self.assertEqual(
                argv[-3:], ["bash", str(self.ws / "probes" / "mine.sh"), str(t.wt)]
            )
            self.assertEqual(extra, {"__sandbox__": "1"})
        self.assertIn("no probe 'gone'", rt.probe_cmd(t, "gone"))


class TestSandbox(unittest.TestCase):
    def test_nothing_outside_the_review_folder_is_writable_and_there_is_no_network(
        self,
    ):
        if not sandbox.kind():
            self.skipTest("no sandbox on this machine: " + sandbox.describe())
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            ws, outside = base / "ws", base / "elsewhere"
            ws.mkdir()
            outside.mkdir()
            script = (
                "echo in > inside.txt; (echo out > {o}/x.txt) 2>/dev/null && echo OUT-WRITTEN; "
                "python3 -c \"import socket; s=socket.socket(); s.settimeout(3); s.connect(('192.0.2.1', 9))\" "
                "2>/dev/null && echo NET-OK; echo done"
            ).format(o=outside)
            argv = sandbox.wrap(
                ["sh", "-c", script], ws, ws, readonly=[base], timeout_s=60
            )
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=90,
                env=sandbox.env(dict(os.environ, GH_TOKEN="x"), ws),
            )
            self.assertIn("done", p.stdout, p.stderr)
            self.assertNotIn("OUT-WRITTEN", p.stdout)
            self.assertNotIn("NET-OK", p.stdout)
            self.assertEqual((ws / "inside.txt").read_text().strip(), "in")
            self.assertFalse((outside / "x.txt").exists())
            self.assertNotIn(
                "GH_TOKEN", sandbox.env({"GH_TOKEN": "x", "PATH": "/bin"}, ws)
            )


class TestPasses(_Work):
    """The four passes through the fake gateway, each a scripted conversation: the evidence they record, what they
    refuse, the claims they leave without evidence, their outputs and tokens; then `rk evidence-check`.
    """

    def run_passes(self):
        keyfile = self.root / "key"
        keyfile.write_text(KEY, encoding="utf-8")
        srv = fake_models.start(
            self.root / "fake",
            "gateway",
            key_file=str(keyfile),
            info=False,
            script=SCRIPT,
        )
        s = {
            "backend": "openai",
            "url": srv.url,
            "model": fake_models.GOOD,
            "auth_header": "X-Example-Key",
            "auth_scheme": "",
            "api_key_file": str(keyfile),
            "temperature": 0.3,
            "max_output_tokens": 1000,
            "stall_s": 30,
            "think": "max",
            "max_resends": 1,
            "max_hangs": 1,
            "context": 262144,
        }
        o = dict(
            json.loads(os.environ["RKMON_OFFLOAD_JSON"]),
            round="fr",
            resend_wait_s=0,
            max_steps=10,
        )
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "RKMON_WORKER_JSON": json.dumps(s),
                    "RKMON_OFFLOAD_JSON": json.dumps(o),
                },
            ):
                self.assertEqual(
                    offload_agent.main(
                        [str(self.ws), "7", "draft", "quality", "regrade"]
                    ),
                    0,
                )
                (self.ws / "checks").mkdir()
                (self.ws / "checks" / "checks.md").write_text(CHECKS, encoding="utf-8")
                self.assertEqual(offload_agent.main([str(self.ws), "7", "assemble"]), 0)
        finally:
            srv.stop()
        return fake_models.read_log(self.root / "fake", "gateway")

    def records(self, name):
        return [
            json.loads(x)
            for x in (self.ws / name).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

    def test_the_passes_write_the_review_with_evidence_for_every_claim(self):
        log = self.run_passes()
        for name in (
            "draft.md",
            "quality.md",
            "regrade.md",
            "post_draft.md",
            "notes_draft.md",
        ):
            self.assertTrue((self.ws / name).is_file(), name)
        self.assertIn(
            "map: N2 = Q1", (self.ws / "notes_draft.md").read_text(encoding="utf-8")
        )
        self.assertNotIn(
            "=====NOTES=====", (self.ws / "post_draft.md").read_text(encoding="utf-8")
        )
        h = json.loads((self.ws / "offload_run.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {k: v["status"] for k, v in h["passes"].items()},
            dict.fromkeys(("draft", "quality", "regrade", "assemble"), "ok"),
        )
        self.assertEqual((h["head"], h["round"], h["mode"]), (self.head, "fr", "on"))
        runs = self.records("runs.jsonl")
        self.assertEqual(
            [(r["run"], r["tool"]) for r in runs],
            [("R1", "grep_code"), ("R2", "ci_fails")],
        )
        self.assertEqual(runs[0]["argv"][:4], ["grep", "-rnI", "-e", "return 41"])
        ev = {(e["id"], e["kind"]) for e in self.records("evidence.jsonl")}
        self.assertTrue(
            {("N1", "read"), ("N1", "ran"), ("C1", "ran"), ("C2", "read")} <= ev, ev
        )
        # The wrong quote and the quality pass's N id were refused, so N2, D1, Q1 and P1 have none: marked.
        self.assertTrue(
            {("N2", "none"), ("D1", "none"), ("Q1", "none"), ("P1", "none")} <= ev, ev
        )
        self.assertNotIn(("N2", "read"), ev)
        draft_log = (self.ws / "agent_log_draft.txt").read_text(encoding="utf-8")
        self.assertIn("REJECTED: the quote is not at a/b/rocke/x.py:3", draft_log)
        self.assertIn("the quote is at line 5", draft_log)
        self.assertIn(
            "REJECTED: the quality pass records Q ids",
            (self.ws / "agent_log_quality.txt").read_text(encoding="utf-8"),
        )
        self.assertTrue(
            (self.ws / "quality_member.md").is_file()
            and (self.ws / "prep" / "facts.txt").is_file()
        )
        # Every request offered the tools; each pass took the budget role its share is for; the tokens per request.
        self.assertTrue(all(r["tools"] for r in log if r["method"] == "POST"))
        tokens = [
            json.loads(x)
            for x in (self.root / "run" / reviewer_client.TOKENS_LOG)
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            {t["pass"] for t in tokens}, {"draft", "quality", "regrade", "assemble"}
        )
        self.assertEqual(
            {t["role"] for t in tokens if t["pass"] == "assemble"}, {"draft"}
        )
        self.assertEqual(
            [t["step"] for t in tokens if t["pass"] == "draft"], [0, 1, 2, 3, 4, 5]
        )
        total = json.loads(
            (self.ws / "tokens_total.json").read_text(encoding="utf-8")
        )  # both jobs' passes
        self.assertEqual(
            {k: v["requests"] for k, v in total["passes"].items()},
            {"draft": 6, "quality": 2, "regrade": 1, "assemble": 5, "all": 14},
        )
        self.assertEqual(total["passes"]["all"]["prompt_tokens"], 14000)

        report = evidence.check(self.cfg, 7, self.ws)
        by = {c["id"]: c for c in report["claims"]}
        self.assertEqual(
            (by["N1"]["result"], by["C1"]["result"], by["C2"]["result"]),
            ("PASS", "PASS", "PASS"),
        )
        self.assertEqual(
            (by["Q1"]["result"], by["N2"]["result"]), ("MISSING", "MISSING")
        )
        self.assertEqual(
            (report["result"], report["missing_serious"]), ("FAIL", ["Q1"])
        )
        self.assertEqual(len(report["sample"]), 2)
        self.assertTrue(set(report["sample"]) <= {"N2", "D1", "P1", "C2"})
        self.assertEqual(
            report["sample"], evidence.check(self.cfg, 7, self.ws)["sample"]
        )  # the same every time
        md = (self.ws / "evidence_check.md").read_text(encoding="utf-8")
        self.assertIn("— FAIL", md)
        self.assertIn("| Q1 | 🟠 | — | MISSING: no evidence |", md)
        lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 1)
        kept = offload.archive_dir(self.cfg, 7, "fr")
        self.assertIn(f"kept in {kept}", lines)
        for name in ("evidence_check.json", "post_draft.md", "checks.md", "runs.jsonl"):
            self.assertTrue((kept / name).is_file(), name)

        # Contradicted: a quote moved, a recorded output that differs, a command that is not the tool's.
        recs = self.records("evidence.jsonl")
        for e in recs:
            if e["id"] == "N1" and e["kind"] == "read":
                e["line"] = 1
        (self.ws / "evidence.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in recs), encoding="utf-8"
        )
        runs[0]["digest"] = "0" * 16
        runs[1]["argv"] = runs[1]["argv"] + ["--extra"]
        (self.ws / "runs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8"
        )
        by = {c["id"]: c for c in evidence.check(self.cfg, 7, self.ws)["claims"]}
        self.assertEqual(by["N1"]["result"], "FAIL")
        self.assertIn(
            "the quote is not at a/b/rocke/x.py:1 (it is at line 2)", by["N1"]["why"]
        )
        self.assertEqual(
            (by["C1"]["result"], by["C1"]["why"]),
            ("FAIL", "the recorded command is not the one the tool builds"),
        )

        # The head moved since: a re-run is not comparable, and says so; quotes are read at the recorded commit.
        (self.root / "wt7" / "a" / "b" / "rocke" / "x.py").write_text(
            X_HEAD + "# more\n", encoding="utf-8"
        )
        git(self.root / "wt7", "commit", "-q", "-am", "a push")
        runs[0]["digest"] = rt.digest("x")
        (self.ws / "runs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8"
        )
        recs = [e for e in recs if not (e["id"] == "N1" and e["kind"] == "read")]
        (self.ws / "evidence.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in recs), encoding="utf-8"
        )
        by = {c["id"]: c for c in evidence.check(self.cfg, 7, self.ws)["claims"]}
        self.assertEqual(by["N1"]["result"], "NOT RE-RUN")
        self.assertIn("the head worktree moved to", by["N1"]["why"])

    def test_a_kept_run_is_found_by_its_round_and_nothing_to_check_says_so(self):
        lines, code = evidence.cmd(self.cfg, 8)
        self.assertEqual(code, 2)
        self.assertIn("no offload run of #8", lines[0])
        self.run_passes()
        evidence.cmd(self.cfg, 7)
        for name in os.listdir(
            self.ws
        ):  # the live run is gone (a new review of the PR wiped it)
            p = self.ws / name
            if p.is_file():
                p.unlink()
        self.assertEqual(
            evidence.find(self.cfg, 7, "fr"), offload.archive_dir(self.cfg, 7, "fr")
        )
        report = evidence.check(self.cfg, 7, offload.archive_dir(self.cfg, 7, "fr"))
        by = {c["id"]: c for c in report["claims"]}
        self.assertEqual(
            by["N1"]["result"], "NOT RE-RUN"
        )  # its grep: a kept run's commands are not re-run
        self.assertEqual(
            by["C2"]["result"], "PASS"
        )  # its quote of the kept checks report


class TestQualityCard(_Work):
    def test_a_follow_up_reads_our_last_review_from_the_facts(self):
        """A follow-up's code quality pass gets our last review's section of the facts (its open quality findings,
        posted or kept unposted), and that section alone."""
        titles = [title.format(me="op") for title, _ in facts.SECTIONS]
        k = next(
            i
            for i, (_, read) in enumerate(facts.SECTIONS)
            if read is facts._last_findings
        )
        session = offload_agent.Session(
            self.ws, 7, settings={}, offload={"round": "fu2"}
        )
        (self.ws / "prep").mkdir()
        (self.ws / "prep" / "facts.txt").write_text(
            "".join(f"== {t}\nfact-{i}\n" for i, t in enumerate(titles)),
            encoding="utf-8",
        )
        user = session.prompt_quality()[1]
        self.assertIn(
            "===== OUR LAST REVIEW (its open quality findings, for a follow-up) =====\n"
            f"== {titles[k]}\nfact-{k}\n",
            user,
        )
        self.assertNotIn(f"fact-{k + 1}", user)


class TestClassicDraft(_Work):
    """review_agent.py on the shared tools (reviewer_tools.py), as the worker job runs it: its tools still answer, its
    draft is written, and its tokens are recorded like the passes'."""

    def test_the_first_pass_reviewer_runs_on_the_shared_tools(self):
        keyfile = self.root / "key"
        keyfile.write_text(KEY, encoding="utf-8")
        srv = fake_models.start(
            self.root / "fake", "gateway", key_file=str(keyfile), info=False
        )
        s = dict(
            config.worker(self.cfg),
            backend="openai",
            url=srv.url,
            model=fake_models.GOOD,
            auth_header="X-Example-Key",
            auth_scheme="",
            api_key_file=str(keyfile),
            stall_s=30,
        )
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "worker" / "review_agent.py"),
                    str(self.ws),
                    "7",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env=dict(os.environ, RKMON_WORKER_JSON=json.dumps(s)),
            )
        finally:
            srv.stop()
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn(
            "first-round review (from the fake reviewer model)",
            (self.ws / "review_agent.md").read_text(encoding="utf-8"),
        )
        log = (self.ws / "agent_log.txt").read_text(encoding="utf-8")
        self.assertIn('TOOL read_file({"path": "README.md"})', log)
        self.assertIn("no such file in worktree: 'README.md'", log)
        rows = [
            json.loads(x)
            for x in (self.root / "run" / reviewer_client.TOKENS_LOG)
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [(r["pass"], r["step"]) for r in rows], [("classic", 0), ("classic", 1)]
        )
        line = json.loads((self.ws / "tokens_total.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (line["mode"], line["steps"], line["passes"]["classic"]["requests"]),
            ("classic", 2, 2),
        )

    def test_a_run_with_no_claim_fails_its_check(self):
        (self.ws / "offload_run.json").write_text(
            json.dumps({"pr": 7, "round": "fr", "mode": "on", "head": self.head}),
            encoding="utf-8",
        )
        report = evidence.check(self.cfg, 7, self.ws)
        self.assertEqual(report["result"], "FAIL")
        self.assertIn("no claims recorded", report["summary"])


class TestPipes(unittest.TestCase):
    """The pipe's finish stage (the assembly and the evidence check, once the checks are done) and the REVIEW_READY
    lines: audit or deep with the offload on, the usual line with a note in shadow mode.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = pathlib.Path(self.tmp.name)
        self.store = Store(self.work / "state")
        self.st = empty_state()
        self.st["prs"] = {
            "7": {
                "n": 7,
                "head": "7" * 40,
                "status": "open",
                "draft": False,
                "size": 10,
            }
        }
        self.started, self.status = [], {"w7": "done", "c7": "done"}

    def tearDown(self):
        self.tmp.cleanup()

    def pipe(self, m):
        review.pipes(self.st).append(
            {
                "pr": 7,
                "kind": "first-round",
                "head": "7" * 40,
                "since": None,
                "started": OLD,
                "stage": "worker",
                "worker": {"id": "w7", "started": OLD},
                "checks": {"id": "c7", "started": OLD},
                "files": 1,
                "offload": m,
                "round": "fr",
                "finish": None,
            }
        )

    def advance(self, cfg):
        def start(store, kind, name, script, timeout_min):
            self.started.append((kind, name, script, timeout_min))
            return {"id": "f7", "started": OLD}

        with mock.patch.object(review.jobs, "start", start), mock.patch.object(
            review.jobs,
            "status",
            lambda store, job: (
                self.status.get((job or {}).get("id"), "none") if job else "none"
            ),
        ), mock.patch.object(
            review, "_record_checks_gates", lambda cfg, rec, n: None
        ), mock.patch.object(
            review, "_changed_files", lambda cfg, rec: ["a/b/rocke/x.py"]
        ), mock.patch.object(
            review, "_fill", lambda cfg, st, store, plan: []
        ), mock.patch(
            "rkmon_lib.gpu.after_checks", lambda *a: []
        ):
            return review.advance(cfg, self.st, self.store, [])

    def files(self, folder: pathlib.Path, result: str):
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("draft.md", "quality.md", "regrade.md", "post_draft.md"):
            (folder / name).write_text("x\n", encoding="utf-8")
        (folder / "evidence_check.json").write_text(
            json.dumps({"result": result, "summary": f"{result}: 3 claims"}),
            encoding="utf-8",
        )

    def test_on_the_finish_job_then_an_audit_or_a_deep_review(self):
        cfg = a_cfg(self.work, offload={"mode": "on"})
        ws = pathlib.Path(config.paths.get(cfg, "workers")) / "pr7"
        self.pipe("on")
        ws.mkdir(parents=True)
        (ws / "draft.md").write_text("x\n", encoding="utf-8")
        lines = self.advance(cfg)
        self.assertIn(
            "the offload's assembly and evidence check started (f7)", lines[0]
        )
        kind, name, script, timeout = self.started[0]
        self.assertEqual(
            (kind, name, timeout), ("offload", "pr7", offload.FINISH_TIMEOUT_MIN)
        )
        self.assertIn(f"offload_agent.py {ws} $N assemble", script)
        self.assertIn("RKMON_OFFLOAD_ROUND=fr", script)
        self.assertIn("$M/rk evidence-check $N", script)
        self.status["f7"] = "running"
        self.assertIn("run (f7)", self.advance(cfg)[0])
        self.files(ws, "PASS")
        self.status["f7"] = "done"
        ready = [x for x in self.advance(cfg) if x.startswith("REVIEW_READY")]
        self.assertTrue(
            ready[0].startswith("REVIEW_READY #7 (first-round, audit): offload review"),
            ready,
        )
        self.assertIn("PASS: 3 claims", ready[0])
        self.assertEqual(
            (self.st["review_ready"][0]["offload"], self.st["review_ready"][0]["why"]),
            ("audit", []),
        )
        self.st["review_ready"], self.started = [], []
        self.pipe("on")
        self.files(ws, "FAIL")
        self.status["f7"] = "done"
        self.advance(cfg)  # the finish job of this pipe is new: it starts
        ready = [x for x in self.advance(cfg) if x.startswith("REVIEW_READY")]
        self.assertTrue(
            ready[0].startswith(
                "REVIEW_READY #7 (first-round, deep: evidence check FAIL"
            ),
            ready,
        )

    def test_shadow_the_usual_line_after_the_finish_job(self):
        cfg = a_cfg(self.work, offload={"mode": "shadow"})
        ws = pathlib.Path(config.paths.get(cfg, "workers")) / "pr7"
        ws.mkdir(parents=True)
        (ws / "review_agent.md").write_text("draft\n", encoding="utf-8")
        self.pipe("shadow")
        self.advance(cfg)
        self.assertIn(
            f"offload_agent.py {ws}/offload $N assemble --pr-dir {ws}/pr",
            self.started[0][2],
        )
        self.status["f7"] = "done"
        ready = [x for x in self.advance(cfg) if x.startswith("REVIEW_READY")]
        self.assertTrue(
            ready[0].startswith(
                "REVIEW_READY #7 (first-round): worker done, checks done."
            ),
            ready,
        )
        self.assertIn(
            f"Shadow offload (posts nothing): {ws}/offload; after our post, `rk shadow-report 7`",
            ready[0],
        )
        self.assertNotIn("offload", self.st["review_ready"][0])

    def test_off_no_finish_job(self):
        cfg = a_cfg(self.work)
        ws = pathlib.Path(config.paths.get(cfg, "workers")) / "pr7"
        ws.mkdir(parents=True)
        (ws / "review_agent.md").write_text("draft\n", encoding="utf-8")
        review.pipes(self.st).append(
            {
                "pr": 7,
                "kind": "first-round",
                "head": "7" * 40,
                "since": None,
                "started": OLD,
                "stage": "worker",
                "worker": {"id": "w7", "started": OLD},
                "checks": {"id": "c7", "started": OLD},
                "files": 1,
            }
        )
        ready = [x for x in self.advance(cfg) if x.startswith("REVIEW_READY")]
        self.assertEqual(self.started, [])
        self.assertTrue(ready and "Shadow" not in ready[0])
        self.assertFalse(
            review.has_draft(a_cfg(self.work, offload={"mode": "on"}), 7)
        )  # on: draft.md

    def test_why_a_review_goes_deep(self):
        cfg = a_cfg(
            self.work,
            offload={"mode": "on"},
            hot_spots=home_repository.VALUES["hot_spots"],
        )
        folder = self.work / "run7"
        self.files(folder, "PASS")
        rec = {"size": 10}
        self.assertEqual(
            offload.deep_reasons(cfg, rec, "first-round", ["a/b/rocke/x.py"], folder),
            [],
        )
        abi = cfg["hot_spots"]["ABI"].replace("<ROCKE_PATH>", "a/b/rocke")
        why = offload.deep_reasons(
            cfg,
            {"size": 2500},
            "first-round",
            [f"f{i}.py" for i in range(41)] + [abi, "a/CMakeLists.txt"],
            folder,
        )
        text = "; ".join(why)
        for words in (
            "2500 changed lines",
            "43 files",
            "the hot spots ABI",
            "a/CMakeLists.txt",
        ):
            self.assertIn(words, text)
        self.assertEqual(
            offload.deep_reasons(cfg, {"size": 2500}, "follow-up", [abi], folder), []
        )  # first rounds
        (folder / "quality.md").unlink()
        self.assertEqual(
            offload.deep_reasons(cfg, rec, "follow-up", [], folder),
            ["the reviewer model wrote no " "quality.md"],
        )


class TestShadowReport(unittest.TestCase):
    POSTED = """Automated review: Model - m, effort: max

## PR #7 — t (a) — 🔴 First-round: wrong answer

| Prior item | Status |
|---|---|
| Bots / CI | 🔴 FAIL<br>• fails: build |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🔴 | `a/b/rocke/x.py:2` | **answer returns 41, not 42**<br>• impact: every caller gets a wrong answer<br>• fix: return 42 |
| N2 | 🟠 | `a/b/rocke/y.py:9` | **the cache key ignores the arch**<br>• impact: stale kernels<br>• fix: add it |
"""

    def test_matched_missed_and_extra_with_their_evidence_check(self):
        with tempfile.TemporaryDirectory() as d:
            work = pathlib.Path(d)
            cfg = a_cfg(work, offload={"mode": "shadow"})
            run = offload.archive_dir(cfg, 7, "fr")
            run.mkdir(parents=True)
            (run / "offload_run.json").write_text(
                json.dumps(
                    {
                        "pr": 7,
                        "round": "fr",
                        "mode": "shadow",
                        "head": "7" * 40,
                        "started": "2026-11-05T10:00:00Z",
                    }
                )
            )
            (run / "post_draft.md").write_text(POST, encoding="utf-8")
            for name, text in (
                ("draft.md", DRAFT),
                ("quality.md", QUALITY),
                ("regrade.md", REGRADE),
            ):
                (run / name).write_text(text, encoding="utf-8")
            (run / "notes_draft.md").write_text(
                "map: N1 = N1\nmap: N2 = Q1\n", encoding="utf-8"
            )
            (run / "evidence_check.json").write_text(
                json.dumps(
                    {
                        "result": "FAIL",
                        "summary": "FAIL: 2 claims",
                        "claims": [
                            {"id": "N1", "result": "PASS", "why": ""},
                            {"id": "Q1", "result": "MISSING", "why": "no evidence"},
                        ],
                    }
                )
            )
            (run / "tokens_total.json").write_text(
                json.dumps(
                    {
                        "passes": {
                            "draft": {"prompt_tokens": 100, "completion_tokens": 20},
                            "all": {
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "requests": 2,
                            },
                        }
                    }
                )
            )
            store = Store(work / "state")
            store.ledger_add(
                "post",
                7,
                "7" * 40,
                "https://github.com/ROCm/r/pull/7#issuecomment-555",
                "first_round",
            )
            st = empty_state()
            st["prs"] = {"7": {"n": 7, "head": "7" * 40, "size": 10}}
            asked = []

            def rest(path, jq=None, paginate=False, timeout=120):
                asked.append(path)
                return {
                    "body": self.POSTED,
                    "html_url": "https://github.com/ROCm/r/pull/7#issuecomment-555",
                    "created_at": "2026-11-05T11:00:00Z",
                }

            with mock.patch.object(gh, "rest", rest):
                lines, code = shadow.cmd(cfg, st, store, None)
            self.assertEqual(code, 0)
            self.assertEqual(asked, ["repos/acme/widgets/issues/comments/555"])
            page = (
                pathlib.Path(config.paths.get(cfg, "outputs_dir"))
                / "rocke_shadow"
                / "shadow_pr7_fr.md"
            ).read_text()
            self.assertIn("## Matched (1)", page)
            self.assertIn("| N1 | N1 | 🔴 / 🟠 | `a/b/rocke/x.py:2` | PASS |", page)
            self.assertIn("## Missed: in our post, not in the offload (1)", page)
            self.assertIn(
                "| N2 | 🟠 | `a/b/rocke/y.py:9` | the cache key ignores the arch |",
                page.split("## Missed")[1].split("## Extra")[0],
            )
            self.assertIn(
                "| N2 | 🟠 | `a/b/rocke/x.py:5` | MISSING: no evidence |", page
            )
            self.assertIn("reviewed in depth (evidence check FAIL", page)
            self.assertIn(
                "Reviewer model tokens: draft 120 (total 120 in 2 requests", page
            )
            summary_md = (
                pathlib.Path(config.paths.get(cfg, "outputs_dir"))
                / "rocke_shadow"
                / "summary.md"
            ).read_text()
            self.assertIn(
                "| #7 | fr | 1 of 2 | 1 (1) | 1 (1) | 🔴 / 🟠 | FAIL | deep | 120 |",
                summary_md,
            )
            self.assertIn(
                "- Findings found: 1 of 2 posted (50%); 🔴/🟠: 1 of 2 (50%)", lines
            )
            self.assertEqual(shadow.cmd(a_cfg(work / "none"), st, store, [9])[1], 2)

    def test_the_parsers(self):
        f = shadow.findings(self.POSTED)
        self.assertEqual(
            [(x["id"], x["severity"], x["file"], x["line"]) for x in f],
            [("N1", "🔴", "a/b/rocke/x.py", 2), ("N2", "🟠", "a/b/rocke/y.py", 9)],
        )
        self.assertEqual(shadow.header(self.POSTED), ("🔴", "wrong answer"))
        self.assertEqual(
            shadow.priors(REGRADE),
            [
                {
                    "item": "a bot (11-05) rename the helper",
                    "light": "🟢",
                    "status": "None yet",
                }
            ],
        )
        matched, missed, extra = shadow.match(f, shadow.findings(POST))
        self.assertEqual([(p["id"], o["id"]) for p, o, _s in matched], [("N1", "N1")])


class TestSummaryDoctor(unittest.TestCase):
    def setUp(self):
        # The monitor's home is one of the test's own.
        own_home(self)

    def test_rk_doctor_s_row_says_what_is_wrong_with_the_settings(self):
        """What is wrong with the offload settings: rk doctor's row of them, as checks.render prints it (with its
        fix), and no other offload row; PASS when they are fine."""
        from rkmon_lib import checks

        raw = a_cfg(pathlib.Path("/w"), offload={"mode": "shadow", "spot_check": 2})
        row = offload.settings_check(raw)
        self.assertEqual(offload.doctor_checks(raw), [row])
        out = checks.render([row])
        self.assertTrue(
            out.startswith("  FAIL  review offload: offload.spot_check is 2"), out
        )
        self.assertIn(
            "\n        fix: fix them in your deploy settings file, then rkmon confirm",
            out,
        )
        raw = a_cfg(pathlib.Path("/w"), offload={"mode": "shadow"})
        self.assertEqual(offload.settings_check(raw).status, "PASS")

    def test_rk_doctor_says_what_is_set(self):
        cfg = a_cfg(pathlib.Path("/w"), offload={"mode": "shadow"})
        checks = {c.name: c for c in offload.doctor_checks(cfg)}
        self.assertEqual(checks["review offload"].status, "PASS")
        # no sandbox row: the container makes no namespaces, so the passes' own code never has one there
        self.assertNotIn("offload sandbox", checks)
        self.assertEqual(
            checks["offload instructions"].status, "WARN"
        )  # nothing rendered under /w
        self.assertIn("shadow trial", checks)
        self.assertEqual(
            [c.status for c in offload.doctor_checks(a_cfg(pathlib.Path("/w")))],
            ["OFF"],
        )
        bad = offload.doctor_checks(
            a_cfg(pathlib.Path("/w"), offload={"mode": "never"})
        )
        self.assertEqual(bad[0].status, "FAIL")

    def test_the_instructions_row_names_each_file_the_passes_lack(self):
        """rk doctor's offload instructions: the quality reviewer's and the verifier's team member files in the
        agentic folder, the verifier's at verifier_file.FILE; WARN with each missing one's path there, else PASS.
        """
        quality = "team_members/rocke_code_quality_reviewer.md"
        with tempfile.TemporaryDirectory() as d:
            agentic = pathlib.Path(d) / "agentic"
            cfg = a_cfg(pathlib.Path(d), offload={"mode": "shadow"})

            def row():
                return {c.name: c for c in offload.doctor_checks(cfg)}[
                    "offload instructions"
                ]

            with mock.patch.object(verifier_file, "FILE", "team_members/v.md"):
                r = row()
                self.assertEqual(r.status, "WARN")
                self.assertEqual(
                    r.found,
                    f"missing {quality}, team_members/v.md in {agentic}; it blocks the quality, re-grade and "
                    "assembly passes",
                )
                for rel in (quality, "team_members/v.md"):
                    (agentic / rel).parent.mkdir(parents=True, exist_ok=True)
                    (agentic / rel).write_text("# instructions\n", encoding="utf-8")
                r = row()
                self.assertEqual(r.status, "PASS")
                self.assertEqual(
                    r.found,
                    f"{agentic}: the quality and verifier instructions the passes read",
                )
            r = row()  # the verifier's own file is not there
            self.assertEqual(r.status, "WARN")
            self.assertTrue(
                r.found.startswith(f"missing {verifier_file.FILE} in {agentic};"),
                r.found,
            )

    def test_rk_runs_both_commands(self):
        import rkmon

        with tempfile.TemporaryDirectory() as d:
            work = pathlib.Path(d)
            cfg = a_cfg(work, offload={"mode": "shadow"})
            store = Store(work / "state")
            st = empty_state()
            store.save(st)
            printed = []
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(rkmon, "Store", lambda: store), mock.patch(
                "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
            ):
                self.assertEqual(rkmon.main(["evidence-check", "7", "--offload"]), 2)
                self.assertIn("no offload run of #7", printed[-1])
                self.assertEqual(
                    rkmon.main(["evidence-check", "7"]), 2
                )  # a draft's run (the drafter's)
                self.assertIn("no draft run of #7", printed[-1])
                self.assertEqual(
                    rkmon.main(["evidence-check", "7", "--round", "x9"]), 4
                )
                self.assertEqual(rkmon.main(["shadow-report"]), 2)
                self.assertIn("no kept offload run", printed[-1])
            self.assertIn("evidence-check N --offload [--round R]", rkmon.__doc__)
            self.assertIn("shadow-report [N ...]", rkmon.__doc__)


if __name__ == "__main__":
    unittest.main()
