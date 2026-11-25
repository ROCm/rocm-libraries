# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GitHub read/write split: every gh the toolkit starts reads under the read login, and only post.py's guarded
writes run under the write login; in the container they run as the write user, through the broker.

- The code itself: no gh is started anywhere but rkmon_lib/gh.py (and the few named below, each naming its folder),
  no shell line runs gh, and gh.write is called from post.py alone.
- A fake gh on PATH logs each call's GH_CONFIG_DIR: every read (gh.py, tools/ghr.py, the tools) uses the read folder,
  every write command of post.py the write folder, and no token variable reaches gh.
- Jobs get the read folder and no other login folder, token or atlassian variable; a job that asks no model (every
  kind but jobs.MODEL_KINDS, a PR's code's among them) no model or Claude credential either, which a job that asks one
  keeps; PR code gets only what worker/pr_env.py lists, none of them, on a host too.
- The login check (ghlogins.py), the broker's routing and its own refusals, and tools/gh_split_apply.py and tools/gh_split_rollback.py, against fake logins.
"""

import ast
import contextlib
import io
import json
import os
import pathlib
import pty
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests" / "fakes"))

from rkmon_lib import (  # noqa: E402
    config,
    container_layout,
    gh,
    ghlogins,
    jobs,
    outward,
    paths,
    post,
    review,
    tickfilter,
    writer,
    writer_run,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402
import fake_net  # noqa: E402
import pr_env  # noqa: E402
import pr_user  # noqa: E402

# --- the code ---------------------------------------------------------------------------------------------------------

GH_VERBS = (
    r"(?:api|pr|issue|label|repo|run|release|search|auth|workflow|gist|secret|variable|cache|ruleset|project"
    r"|browse|codespace|extension|alias|config)"
)
# gh at a command's start in a shell line, or in a Python string that is one (a job script, a check).
SHELL_GH = re.compile(r"(?:^|[\s;&|(\"'{])gh\s+" + GH_VERBS + r"\b")
# A Python string that is a shell line: a variable, a pipe, a redirection or a command list.
SHELL_SYNTAX = re.compile(r"\$|\s\|\s|2>|&&|\|\|")
# A Python call that starts gh: "gh" as an argv element (the first or a later one, followed by another).
PY_ARGV = re.compile(r"""(?:\[\s*|,\s*|or\s+)["']gh["']\s*,""")
# The files that start gh themselves, each naming its folder: the doctor's version and status reads, the host's split
# helper, the container's wrapper (its `gh` and `gh-write` commands run in the container, under the read login and as
# the write user), and the worker tools (copies a job runs, whose environment holds the read login only: gh.job_env).
OWN_GH = {
    "rkmon_lib/doctor.py",
    "tools/gh_split_apply.py",
    "docker/rkmon",
    "worker/fetch_pr.py",
}


def tracked() -> list[pathlib.Path]:
    try:
        names = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        names = [
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]
    new = [
        "rkmon_lib/ghlogins.py",
        "tools/ghr.py",
        "tools/gh_split_apply.py",
        "docker/rk_write.py",
    ]
    return [
        ROOT / n
        for n in dict.fromkeys(names + new)
        if not n.startswith("tests/") and (ROOT / n).is_file()
    ]


def docstring_lines(src: str) -> set[int]:
    out = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.update(range(first.lineno, first.end_lineno + 1))
    return out


def python_files() -> list[pathlib.Path]:
    out = []
    for p in tracked():
        if p.suffix == ".py":
            out.append(p)
        elif not p.suffix and os.access(p, os.X_OK):
            with p.open("rb") as fh:
                if b"python" in fh.readline():
                    out.append(p)
    return out


class TestChokePoints(unittest.TestCase):
    def test_only_gh_py_starts_gh(self):
        bad = []
        for p in python_files():
            rel = str(p.relative_to(ROOT))
            if rel == "rkmon_lib/gh.py" or rel in OWN_GH:
                continue
            src = p.read_text(encoding="utf-8")
            docs = docstring_lines(src)
            for i, line in enumerate(src.splitlines(), 1):
                if (
                    i not in docs
                    and not line.lstrip().startswith("#")
                    and PY_ARGV.search(line)
                ):
                    bad.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(
            bad,
            [],
            "gh started outside rkmon_lib/gh.py: use gh.rest/raw/graphql/run_read or tools/ghr.py (reads), "
            "post._gh_write (writes)",
        )
        src = (ROOT / "rkmon_lib" / "gh.py").read_text(encoding="utf-8")
        self.assertEqual(
            (src.count("subprocess.run("), src.count("os.execvpe(")), (1, 1)
        )  # _exec, exec_read

    def test_no_shell_line_runs_gh(self):
        """A gh in a script the toolkit writes, or in a shell file, goes through tools/ghr.py (the read login)."""
        bad = []
        for p in tracked():
            if p.suffix not in (".py", ".sh") and p.name != "rk":
                continue
            src = p.read_text(encoding="utf-8")
            docs = docstring_lines(src) if p.suffix == ".py" else set()
            for i, line in enumerate(src.splitlines(), 1):
                code = line.lstrip()
                if i in docs or code.startswith("#"):
                    continue
                if p.suffix == ".py" and not SHELL_SYNTAX.search(line):
                    continue  # a message naming a gh command, not a shell line the toolkit writes
                if SHELL_GH.search(line) and "GH_CONFIG_DIR=" not in line:
                    bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
        # The container's entry point runs gh by hand for the operator (`rkmon gh`, `rkmon login github_write`), each
        # under the folder it names (with_settings: the read login; gh-write: the write user's own).
        bad = [b for b in bad if not b.startswith("docker/entrypoint.sh")]
        self.assertEqual(
            bad, [], "a gh call that does not name its login: use tools/ghr.py"
        )

    def test_writes_come_from_post_py_alone(self):
        callers = []
        for p in python_files():
            if p.name != "gh.py" and re.search(
                r"\bgh(?:_cli)?\.write\(|\b_exec\(", p.read_text(encoding="utf-8")
            ):
                callers.append(str(p.relative_to(ROOT)))
        self.assertEqual(callers, ["rkmon_lib/post.py"])
        src = (ROOT / "rkmon_lib" / "post.py").read_text(encoding="utf-8")
        self.assertNotIn("subprocess.run(", src)
        self.assertNotIn("subprocess.Popen(", src)

    def test_the_broker_takes_the_github_write_commands_only(self):
        from rkmon_lib import outward

        # Every outward write to GitHub is brokered, its file arguments checked (rkmon_lib/outward.py). spec-post: its
        # premises and AFTER lines run as the session user, its post through the broker.
        github = outward.names(dest=outward.GITHUB)
        self.assertIn("spec-post", github)
        for cmd in github:
            with self.subTest(cmd=cmd):
                brokered = cmd != "spec-post"
                self.assertEqual(writer.routes([cmd, "7"]), brokered)
                self.assertEqual(cmd in writer.FILE_ARGS, brokered)
        for word in ("approve", "dismiss", "merge-now"):
            self.assertFalse([c for c in writer.COMMANDS if word in c])


# --- a fake gh --------------------------------------------------------------------------------------------------------

FAKE_GH = r"""#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
d = os.environ.get("GH_CONFIG_DIR") or os.path.join(os.environ.get("XDG_CONFIG_HOME") or
                                                    os.path.join(os.environ["HOME"], ".config"), "gh")
tokens = sorted(k for k in ("GH_TOKEN", "GITHUB_TOKEN") if k in os.environ)
with open(os.environ["FAKE_GH_LOG"], "a") as f:
    f.write(json.dumps({"argv": a, "dir": d, "tokens": tokens}) + "\n")
joined = " ".join(a)
if a[:2] == ["api", "user"]:
    print(os.path.basename(d) + "-login"); sys.exit(0)
if a[:3] == ["api", "-X", "POST"] and joined.endswith("-f body="):  # the login probe's comment
    if os.path.basename(d).startswith("write") or os.path.exists(os.path.join(d, "CAN_WRITE")) or (
            os.environ.get("FAKE_CAN_WRITE_DIR") == d):
        print("gh: Validation Failed (HTTP 422)", file=sys.stderr)
    else:
        print("gh: Resource not accessible by personal access token (HTTP 403)", file=sys.stderr)
    sys.exit(1)
if a[:3] == ["api", "-X", "POST"] and "/git/refs" in joined:  # the login probe's push
    if not os.path.exists(os.path.join(d, "NO_PUSH")) and (
            os.path.basename(d).startswith("write") or os.path.exists(os.path.join(d, "CAN_PUSH")) or (
            os.environ.get("FAKE_CAN_WRITE_DIR") == d)):
        print("gh: Validation Failed (HTTP 422)", file=sys.stderr)
    else:
        print("gh: Resource not accessible by personal access token (HTTP 403)", file=sys.stderr)
    sys.exit(1)
for r in json.load(open(os.environ["FAKE_GH_RULES"])):
    if all(h in joined for h in r["has"]):
        sys.stdout.write(r.get("out", "")); sys.stderr.write(r.get("err", "")); sys.exit(r.get("rc", 0))
sys.exit(0)
"""
WRITES = {
    ("pr", "comment"),
    ("pr", "review"),
    ("pr", "merge"),
    ("issue", "create"),
    ("issue", "edit"),
    ("issue", "comment"),
    ("label", "create"),
}


def is_write(argv: list[str]) -> bool:
    if argv[:1] == ["api"]:
        return any(x in argv for x in ("-X", "--method")) and "-X GET" not in " ".join(
            argv
        )
    return tuple(argv[:2]) in WRITES


class FakeGh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.R, self.W = self.d / "read", self.d / "write"
        for f in (self.R, self.W):
            f.mkdir()
            (f / "hosts.yml").write_text("github.com: {}\n")
        b = self.d / "bin"
        b.mkdir()
        (b / "gh").write_text(FAKE_GH)
        (b / "gh").chmod(0o755)
        self.log, self.rules_file = self.d / "gh.log", self.d / "rules.json"
        self.rules()
        self.split = {"read_config_dir": str(self.R), "write_config_dir": str(self.W)}
        for p in (
            mock.patch.dict(
                os.environ,
                {
                    "PATH": f"{b}{os.pathsep}{os.environ['PATH']}",
                    "FAKE_GH_LOG": str(self.log),
                    "FAKE_GH_RULES": str(self.rules_file),
                    "GH_TOKEN": "must-not-reach-gh",
                    "GITHUB_TOKEN": "nor-this",
                    "GH_CONFIG_DIR": str(self.W),
                    "ATLASSIAN_CONFIG_DIR": str(self.d),
                },
            ),
            mock.patch.object(gh.time, "sleep", lambda s: None),
        ):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("RKMON_GH", None)
        gh.configure({"repo": "o/r", "gh": self.split})
        self.addCleanup(gh.configure, {})

    def rules(self, *rules):
        self.rules_file.write_text(json.dumps(list(rules)))

    def calls(self) -> list[dict]:
        return (
            [json.loads(x) for x in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def assert_split(self, want_writes: int | None = None):
        calls = self.calls()
        self.assertTrue(calls)
        for c in calls:
            self.assertEqual(
                c["dir"], str(self.W if is_write(c["argv"]) else self.R), c["argv"]
            )
            self.assertEqual(c["tokens"], [], c["argv"])
        if want_writes is not None:
            self.assertEqual(
                sum(is_write(c["argv"]) for c in calls), want_writes, calls
            )
        return calls


class TestReads(FakeGh):
    def test_every_read_helper_uses_the_read_login(self):
        self.rules(
            {"has": ["graphql"], "out": '{"data": {"x": 1}}'},
            {"has": ["api"], "out": '{"a": 1}'},
            {"has": ["pr", "view"], "out": "{}"},
        )
        self.assertEqual(gh.rest("repos/o/r/pulls/7"), {"a": 1})
        gh.rest("repos/o/r/pulls", paginate=True)
        gh.raw("repos/o/r/actions/jobs/5/logs")
        self.assertEqual(gh.graphql("query { x }"), {"x": 1})
        self.assertEqual(
            gh.run_read(["pr", "view", "7", "--repo", "o/r"]).returncode, 0
        )
        self.assertEqual(len(self.assert_split(want_writes=0)), 5)

    def test_ghr_reads_under_the_read_login_and_refuses_writes(self):
        """tools/ghr.py as a job runs it: not configured, so the job's GH_CONFIG_DIR (the read folder) names it."""
        self.rules({"has": ["api"], "out": "body\n"})
        ghr = [sys.executable, "-I", str(ROOT / "tools" / "ghr.py")]
        env = dict(os.environ, GH_CONFIG_DIR=str(self.R))
        r = subprocess.run(
            ghr + ["api", "repos/o/r/pulls/7"], capture_output=True, text=True, env=env
        )
        self.assertEqual((r.returncode, r.stdout), (0, "body\n"), r.stderr)
        c = self.calls()[-1]
        self.assertEqual((c["dir"], c["tokens"]), (str(self.R), []))
        n = len(self.calls())
        for bad in (
            ["api", "-X", "POST", "repos/o/r/issues/1/comments", "-f", "body=x"],
            ["pr", "comment", "7"],
            ["api", "graphql", "-f", "query=mutation { x }"],
            ["issue", "edit", "7"],
        ):
            r = subprocess.run(ghr + bad, capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("refusing", r.stderr)
        self.assertEqual(len(self.calls()), n)  # a refused call never reached gh

    def test_the_tools_read_under_the_read_login(self):
        self.rules(
            {
                "has": ["actions/runs?head_sha"],
                "out": json.dumps({"workflow_runs": []}),
            },
            {
                "has": ["pulls/7"],
                "out": json.dumps({"auto_merge": None, "title": "t", "body": "b"}),
            },
        )
        env = dict(os.environ, GH_CONFIG_DIR=str(self.R))
        for argv in (
            [str(ROOT / "tools" / "ci_attempts.py"), "a" * 40, "o/r"],
            [str(ROOT / "tools" / "automerge_msg.py"), "o/r", "7"],
        ):
            subprocess.run(
                [sys.executable, "-I", "-B", *argv],
                capture_output=True,
                text=True,
                env=env,
            )
        self.assert_split(want_writes=0)

    def test_a_token_variable_never_reaches_gh_and_the_callers_folder_is_ignored(self):
        self.rules({"has": ["api"], "out": "{}"})
        gh.rest("repos/o/r")  # os.environ names the write folder and two tokens (setUp)
        self.assertEqual(
            (self.calls()[0]["dir"], self.calls()[0]["tokens"]), (str(self.R), [])
        )


# --- every write command of post.py -----------------------------------------------------------------------------------

LINE1 = (
    f"Automated review by rocke-monitor: Model - qwen, effort: {config.MODEL_DEFAULT}"
)
HEAD = "a" * 40
URL = "https://github.com/ROCm/r/pull/7#issuecomment-99\n"
# Both features live: every write below acts.
FEATURES = {
    "review_automation": {"mode": "live"},
    "merge_queue": {"mode": "live", "auto_merge": "enabled", "auto_hold": True},
}


class TestPostWrites(FakeGh):
    def setUp(self):
        super().setUp()
        self.store = Store(self.d / "state")
        self.cfg = {
            "repo": "o/r",
            "operator": {"github_login": "me"},
            "max_posts_per_hour": 6,
            "review_first_line": LINE1,
            "worker": {
                "model": "qwen"
            },  # the reviewer model LINE1 gives (bindings.first_line)
            "rocke_path": "rk",
            "features": FEATURES,
            "gh": self.split,
            "paths": {
                "repo": "/nonexistent",
                "working_dir": str(self.d),
                "daily_dir": str(self.d / "daily"),
                "posted_dir": str(self.d),
            },
        }
        gh.configure(self.cfg)
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40
        self.st["prs"]["7"] = {
            "n": 7,
            "head": HEAD,
            "status": "open",
            "draft": False,
            "author": "dev",
            "ours": None,
        }

    def file(self, name: str, text: str) -> str:
        (self.d / name).write_text(text, encoding="utf-8")
        return str(self.d / name)

    def test_post(self):
        self.rules(
            {"has": ["pulls/7", "--jq .head.sha"], "out": HEAD + "\n"},
            {"has": ["pr", "comment"], "out": URL},
        )
        body = self.file(
            "p.md", f"{LINE1}\n\n## PR #7 — t — 🟡 First-round: fine — x\n\nbody\n"
        )
        self.assertIn(
            "issuecomment-99", post.post(self.cfg, self.st, self.store, 7, body)
        )
        self.assert_split(want_writes=1)

    def test_notice(self):
        self.rules({"has": ["pr", "comment"], "out": URL})
        body = self.file(
            "n.md", post.notice_line(self.cfg, "incident") + "\n\ndevelop broke\n"
        )
        post.notice(self.cfg, self.st, self.store, 7, body, "incident", "inc-1")
        self.assert_split(want_writes=1)

    def test_arm_and_disarm(self):
        self.st["cleared"] = [7]
        self.st["batches"] = [{"status": "validated", "members": [7]}]
        self.st["queue_state"] = {
            "validated": {"members": [7], "heads": {"7": HEAD}, "develop": "d" * 40}
        }
        self.rules(
            {
                "has": ["pulls/7"],
                "out": json.dumps(
                    {
                        "state": "open",
                        "head": HEAD,
                        "draft": False,
                        "mergeable": "MERGEABLE",
                    }
                ),
            }
        )
        try:
            post.arm(self.cfg, self.st, self.store, 7)
        except post.PostRefused:
            pass  # arm's own guards (the queue's validation): the disarm below is the write this test needs
        post.disarm(self.cfg, self.st, self.store, 7, "develop moved")
        calls = self.assert_split()
        self.assertTrue(any("--disable-auto" in c["argv"] for c in calls))

    def test_queue_notice_made_then_edited(self):
        self.rules({"has": ["pr", "comment"], "out": URL})
        text = post.notice_line(self.cfg, "queue") + "\n\nposition 1\n"
        post.queue_notice(self.cfg, self.st, self.store, 7, text, "k1")
        post.queue_notice(
            self.cfg, self.st, self.store, 7, text.replace("1", "2"), "k2"
        )
        calls = self.assert_split(want_writes=2)
        self.assertEqual(calls[-1]["argv"][:3], ["api", "-X", "PATCH"])

    def test_issue_assign_and_label(self):
        self.rules(
            {"has": ["issues/5", "assignees"], "out": "[]"},
            {"has": ["labels/rocke-x"], "rc": 1, "err": "gh: Not Found (HTTP 404)\n"},
        )
        for call in (
            lambda: post.issue_assign(self.cfg, self.store, 5, "someone"),
            lambda: post.label_create(
                self.cfg, self.store, "rocke-x", "a label", "aabbcc"
            ),
        ):
            try:
                call()
            except post.PostRefused:
                pass
        self.assert_split()

    def test_a_write_folder_that_is_the_read_folder_is_refused_before_gh(self):
        self.rules({"has": ["pr", "comment"], "out": URL})
        cfg = dict(
            self.cfg,
            gh={"read_config_dir": str(self.R), "write_config_dir": str(self.R) + "/"},
        )
        gh.configure(cfg)
        body = self.file("n.md", post.notice_line(cfg, "incident") + "\n\nx\n")
        with self.assertRaisesRegex(gh.GhError, "the read login must not write"):
            post.notice(cfg, self.st, self.store, 7, body, "incident", "inc-2")
        self.assertEqual(self.calls(), [])

    def test_a_dry_run_names_the_write_login(self):
        self.assertIn(f"GH_CONFIG_DIR {self.W}", post.write_login())


# --- PR code: jobs and checks -----------------------------------------------------------------------------------------


class TestPrCode(FakeGh):
    def test_a_job_gets_the_read_folder_and_no_other_login(self):
        store = Store(self.d / "state")
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_OAUTH_TOKEN": "t"}
        ):
            job = jobs.start(
                store,
                "checks",
                "pr7",
                'env | grep -E "^(GH_|GITHUB_|ATLASSIAN_|ANTHROPIC_|CLAUDE_CODE_)" | sort > '
                f"{self.d}/job_env; gh api repos/o/r/pulls/7 >/dev/null",
                timeout_min=1,
            )
        end = time.time() + 30
        # "failed(rc=)": the rc file made but not yet written, for a moment
        while (
            jobs.status(store, job) in ("running", "failed(rc=)") and time.time() < end
        ):
            time.sleep(0.1)
        self.assertEqual(jobs.status(store, job), "done")
        self.assertEqual(
            (self.d / "job_env").read_text().splitlines(), [f"GH_CONFIG_DIR={self.R}"]
        )
        self.assert_split(want_writes=0)

    def test_job_env_keeps_the_rest_and_the_model_credentials_for_a_model(self):
        base = {
            "PATH": "/bin",
            "GH_CONFIG_DIR": str(self.W),
            "GH_TOKEN": "t",
            "ATLASSIAN_CONFIG_DIR": "a",
            "TMPDIR": "/t",
            "ANTHROPIC_API_KEY": "k",
            "CLAUDE_CONFIG_DIR": "/c",
            "OPENAI_BASE_URL": "https://example.com",
            "MY_API_KEY": "x",
        }
        rest = {"PATH": "/bin", "TMPDIR": "/t", "GH_CONFIG_DIR": str(self.R)}
        self.assertEqual(gh.job_env(base), rest)
        self.assertEqual(
            gh.job_env(base, model=True),
            {
                **rest,
                "ANTHROPIC_API_KEY": "k",
                "CLAUDE_CONFIG_DIR": "/c",
                "OPENAI_BASE_URL": "https://example.com",
                "MY_API_KEY": "x",
            },
        )
        # PR code: only what worker/pr_env.py lists, no login folder, token or key
        self.assertEqual(pr_env.allowed(base), {"PATH": "/bin", **pr_env.FIXED})

    def test_only_a_job_that_asks_a_model_gets_the_model_credentials(self):
        planted = {
            "ANTHROPIC_API_KEY": "k",
            "ANTHROPIC_BASE_URL": "https://example.com",
            "CLAUDE_CONFIG_DIR": "/c",
            "CLAUDE_CODE_OAUTH_TOKEN": "t",
            "OPENAI_API_KEY": "k",
            "MY_GATEWAY_KEY": "k",
            "MY_SETTING": "s",
        }
        shown = re.compile(r"(ANTHROPIC_|CLAUDE_|OPENAI_|MY_)")
        others = {k: v for k, v in os.environ.items() if not shown.match(k)}
        store = Store(self.d / "state")
        seen = {}
        for kind in ("gpu", "draft"):
            with mock.patch.dict(os.environ, {**others, **planted}, clear=True):
                job = jobs.start(
                    store,
                    kind,
                    "pr7",
                    'env | grep -E "^(ANTHROPIC_|CLAUDE_|OPENAI_|MY_)" | sort > '
                    f"{self.d}/env_{kind}",
                    timeout_min=1,
                )
            end = time.time() + 30
            while (
                jobs.status(store, job) in ("running", "failed(rc=)")
                and time.time() < end
            ):
                time.sleep(0.1)
            self.assertEqual(jobs.status(store, job), "done")
            seen[kind] = (self.d / f"env_{kind}").read_text().splitlines()
        self.assertEqual(seen["gpu"], ["MY_SETTING=s"])
        self.assertEqual(seen["draft"], sorted(f"{k}={v}" for k, v in planted.items()))
        # The kinds review.py starts by name ask a model; a job that runs a PR's code asks none.
        self.assertLessEqual({review.FIRST_PASS, review.QUALITY}, set(jobs.MODEL_KINDS))
        self.assertFalse(set(jobs.MODEL_KINDS) & set(jobs.PR_CODE_KINDS))

    def test_a_probe_gets_no_login_or_key(self):
        """On a host too: the worker tools run a probe (and all PR code) through worker/pr_env.py, which runs it with
        that list of the environment; in the container the launcher does (tests/test_prcode.py).
        """
        argv, data = pr_user.wrap(["bash", "p1.sh"], {}, environ={})
        self.assertEqual(
            (argv[1:], data),
            (["-I", str(ROOT / "worker" / "pr_env.py"), "bash", "p1.sh"], None),
        )
        ran = []
        pr_env.main(
            argv[3:],
            execvpe=lambda f, args, env: ran.append((args, env)),
            environ={
                "PATH": "/bin",
                "GH_CONFIG_DIR": "w",
                "GITHUB_TOKEN": "t",
                "ANTHROPIC_API_KEY": "k",
            },
        )
        self.assertEqual(ran, [(["bash", "p1.sh"], {"PATH": "/bin", **pr_env.FIXED})])


# --- the login check --------------------------------------------------------------------------------------------------


class TestLogins(FakeGh):
    def setUp(self):
        super().setUp()
        self.store = Store(self.d / "state")
        p = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.d / "xdg")})
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("GH_CONFIG_DIR")
        (self.d / "xdg").mkdir()
        os.symlink(
            self.R, self.d / "xdg" / "gh"
        )  # the session's default folder is the read login
        self.cfg = {"repo": "o/r", "gh": self.split}
        # Something that is on writes to GitHub: review automation is live, so the write login is needed.
        self.live = dict(
            self.cfg, features={"review_automation": {"enabled": True, "mode": "live"}}
        )

    def test_the_split_holds_and_is_probed_once_an_hour(self):
        lines = ghlogins.status_lines(self.cfg, self.store)
        self.assertEqual(lines[1:], [])
        self.assertTrue(
            lines[0].startswith(
                f"GH_LOGINS read read-login read-only ({self.R}); write write-login can comment and push ({self.W}); "
                "session default = the read login; probed "
            ),
            lines,
        )
        n = len(self.calls())
        self.assertEqual(n, 6)  # the login and the probe's two writes, each side
        self.assertEqual(
            ghlogins.tick_lines(self.cfg, self.store), []
        )  # cached: nothing to say, nothing probed
        self.assertEqual(len(self.calls()), n)
        forced = ghlogins.status_lines(self.cfg, self.store, force=True)
        self.assertEqual(len(self.calls()), 2 * n)
        self.assertEqual(
            ghlogins.cached_lines(self.cfg, self.store)[0], forced[0]
        )  # the last probe's line

    def test_a_login_moved_in_is_probed_at_once(self):
        ghlogins.status_lines(self.cfg, self.store)
        n = len(self.calls())
        new = self.R / "hosts.new"
        new.write_text("github.com: {}\n")
        os.replace(new, self.R / "hosts.yml")
        self.assertTrue(
            ghlogins.tick_lines(self.cfg, self.store)[0].startswith("GH_LOGINS ")
        )
        self.assertEqual(len(self.calls()), 2 * n)

    def test_the_probe_result_is_written_whole_at_the_states_shared_mode(self):
        """The probe's result is written whole, at the state's shared mode whatever the umask, with nothing left
        beside it."""
        umask = os.umask(0o077)
        self.addCleanup(os.umask, umask)
        ghlogins.check(self.cfg, self.store, force=True)
        kept = self.store.dir / ghlogins.CACHE_FILE
        self.assertEqual(
            json.loads(kept.read_text())["sides"].keys(), {"read", "write"}
        )
        self.assertEqual(
            stat.S_IMODE(kept.stat().st_mode),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
        self.assertEqual(
            [p.name for p in self.store.dir.iterdir() if p.is_file()], [kept.name]
        )  # files only: the store makes folders of its own

    def test_a_read_login_that_can_write_or_a_write_login_that_cannot_stands_in_every_tick(
        self,
    ):
        (self.R / "CAN_WRITE").write_text("")
        (self.W / "hosts.yml").unlink()
        first = ghlogins.tick_lines(self.live, self.store)
        self.assertTrue(first[0].startswith("GH_LOGINS "), first)
        alarms = [x for x in first if x.startswith("!! ")]
        self.assertEqual(len(alarms), 2, first)
        self.assertIn("the read login can write", alarms[0])
        self.assertIn("the write login cannot comment", alarms[1])
        self.assertEqual(
            ghlogins.tick_lines(self.live, self.store), alarms
        )  # cached, and the same text every tick
        for a in alarms:
            self.assertEqual(tickfilter.classify(a)[0], tickfilter.STANDING)
        self.assertEqual(tickfilter.classify(first[0])[0], tickfilter.ROUTINE)

    def test_the_write_login_s_alarm_waits_for_something_that_writes(self):
        """A write login that cannot comment is an alarm only while something that is on writes to GitHub (the
        registry's need: in prototype nothing does); its fix is the login's own command, as the operator
        types it on the host wherever this runs. A folder that holds no login is "none", not a read-only login.
        """
        (self.W / "hosts.yml").unlink()
        lines = ghlogins.status_lines(self.cfg, self.store)  # everything in prototype
        self.assertIn(f"; write none ({self.W});", lines[0])
        self.assertEqual([x for x in lines if x.startswith("!! ")], [])
        c, _ = ghlogins.check(self.live, None)
        alarm = (
            f"!! GH_LOGIN the write login cannot comment: none in {self.W} (no gh login in {self.W}); rk post and "
            "every other write fail until it is fixed: "
        )
        from rkmon_lib import doctor

        for env, fix in (
            (
                {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"},
                "rkmon login github_write",
            ),
            ({"RKMON_CONTAINER": "", "RKMON_WRAPPER": ""}, "rkmon login github_write"),
        ):
            with self.subTest(fix=fix), mock.patch.dict(os.environ, env):
                self.assertEqual(ghlogins.alarms(self.live, c), [alarm + fix])
                self.assertEqual(ghlogins.alarms(self.cfg, c), [])
                (row,) = [
                    x
                    for x in doctor.gh_split_checks(
                        self.live, check=lambda cfg, store, force: (c, True)
                    )
                    if x.name == "GitHub write login"
                ]
                self.assertEqual(
                    (row.status, str(row.fix.parts[0])), ("FAIL", fix)
                )  # the doctor's fix is the alarm's

    def test_a_write_login_that_may_comment_but_not_push_is_judged_by_need(self):
        """The write login is judged by what is on: posting needs it to comment, and turning auto-merge on or off (the
        merge queue live) needs it to push as well. One that may comment but not push says so in plain words, and is an
        alarm and a doctor FAIL only while the merge queue would turn auto-merge on or off; the doctor's fix asks for a
        token that may do both."""
        from rkmon_lib import doctor

        (self.W / "NO_PUSH").write_text("")
        merging = dict(
            self.live,
            features=dict(
                self.live["features"],
                merge_queue={"enabled": True, "mode": "live", "auto_merge": "enabled"},
            ),
        )
        c, _ = ghlogins.check(self.live, None)
        why = "a comment: HTTP 422 (validation): allowed; a push: HTTP 403: refused"
        said = f"can comment but not push: {ghlogins.NO_PUSH}"
        self.assertEqual(c["sides"]["write"]["why"], why)
        self.assertIn(f"write write-login {said} ({self.W})", ghlogins.summary(c))
        for cfg, alarms, status in (
            (self.live, [], "PASS"),
            (
                merging,
                [
                    f"!! GH_LOGIN the write login cannot push: write-login in {self.W} ({why}); "
                    f"{ghlogins.NO_PUSH} until it is fixed: rkmon login github_write"
                ],
                "FAIL",
            ),
        ):
            with self.subTest(status=status), mock.patch.dict(
                os.environ, {"RKMON_CONTAINER": "", "RKMON_WRAPPER": ""}
            ):
                self.assertEqual(ghlogins.alarms(cfg, c), alarms)
                (row,) = [
                    x
                    for x in doctor.gh_split_checks(
                        cfg, check=lambda cfg, store, force: (c, True)
                    )
                    if x.name == "GitHub write login"
                ]
                self.assertEqual(
                    (row.status, row.found),
                    (status, f"write-login in {self.W}: {said} ({why})"),
                )
        self.assertIn("with a token that may comment and push", str(row.fix))

    def test_a_write_login_that_may_push_but_not_comment_posts_nothing(self):
        """A write login that may push but not comment can write, but posts nothing: an alarm while something that is
        on writes to GitHub, and a doctor FAIL that says what it blocks."""
        from rkmon_lib import doctor

        p = self.d / "pusher"
        p.mkdir()
        (p / "hosts.yml").write_text("github.com: {}\n")
        (p / "CAN_PUSH").write_text("")
        cfg = dict(self.live, gh=dict(self.split, write_config_dir=str(p)))
        c, _ = ghlogins.check(cfg, None)
        self.assertIs(c["sides"]["write"]["can_write"], True)
        why = "a comment: HTTP 403: refused; a push: HTTP 422 (validation): allowed"
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "", "RKMON_WRAPPER": ""}):
            self.assertEqual(
                ghlogins.alarms(cfg, c),
                [
                    f"!! GH_LOGIN the write login cannot comment: pusher-login in {p} ({why}); rk post and every "
                    "other write fail until it is fixed: rkmon login github_write"
                ],
            )
            (row,) = [
                x
                for x in doctor.gh_split_checks(
                    cfg, check=lambda cfg, store, force: (c, True)
                )
                if x.name == "GitHub write login"
            ]
        self.assertEqual(
            (row.status, row.found),
            (
                "FAIL",
                f"pusher-login in {p}: can push but not comment ({why}); it blocks every GitHub write (rk post and "
                "the rest)",
            ),
        )

    def test_the_notes_judge_the_write_side_by_what_it_is_judged_by(self):
        """A note says a side the probe could not settle: the write side while an answer it is judged by
        (write_lacks) is not known, and not for one it is not judged by, nor for one GitHub refused (the alarm's).
        """
        why = "a comment: HTTP 422 (validation): allowed; a push: timed out"
        c = {
            "sides": {
                "read": {"can_write": False, "why": "HTTP 403: refused"},
                "write": {
                    "can_comment": True,
                    "can_push": None,
                    "can_write": True,
                    "why": why,
                },
            }
        }
        merging = dict(
            self.live,
            features=dict(
                self.live["features"],
                merge_queue={"enabled": True, "mode": "live", "auto_merge": "enabled"},
            ),
        )
        self.assertEqual(ghlogins.notes(self.live, c), [])  # nothing on pushes
        self.assertEqual(
            ghlogins.notes(merging, c), [f"(gh login write not known: {why})"]
        )
        c["sides"]["write"].update(can_comment=False, can_write=None)
        self.assertEqual(ghlogins.notes(self.live, c), [])  # refused: the alarm says it

    def test_a_session_default_that_can_write_is_an_alarm(self):
        """A session whose environment names no read folder (gh.settings gives it none) runs gh with gh's own rule:
        that folder is the session's side."""
        s = self.d / "session"
        s.mkdir()
        (s / "hosts.yml").write_text("")
        (s / "CAN_WRITE").write_text("")
        with mock.patch.dict(os.environ, {"GH_CONFIG_DIR": str(s)}), mock.patch.object(
            ghlogins.gh, "settings", lambda cfg: ("gh", {})
        ):
            lines = ghlogins.status_lines(self.cfg, self.store)
        self.assertIn(f"session session-login can comment but not push ({s})", lines[0])
        self.assertTrue(
            any(
                x.startswith("!! GH_LOGIN the session's default gh login")
                for x in lines
            ),
            lines,
        )

    def test_the_session_runs_with_the_read_login_and_a_side_with_no_login_never_passes(
        self,
    ):
        """The session's gh runs with the read login's folder (gh.settings), whatever this process's own default is (in
        the container, the monitor user's empty ~/.config/gh): no session side, and the read row names the login the
        session runs with. A folder with no login is a FAIL, never `?` nor a 403 that was not asked.
        """
        from rkmon_lib import doctor

        os.unlink(self.d / "xdg" / "gh")  # this process's default folder: no login
        checks = {c.name: c for c in doctor.gh_split_checks(self.cfg)}
        self.assertEqual(set(checks), {"GitHub read login", "GitHub write login"})
        self.assertEqual(checks["GitHub read login"].status, "PASS")
        self.assertIn(
            f"read-login in {self.R}: read-only",
            checks["GitHub read login"].found,
        )
        self.assertIn(
            "the session's gh runs with it", checks["GitHub read login"].found
        )
        (self.R / "hosts.yml").unlink()
        for env, fix in (
            ({"RKMON_WRAPPER": ""}, "rkmon login github_read"),
            (
                {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"},
                "rkmon login github_read",
            ),
        ):
            with mock.patch.dict(os.environ, env):
                row = {c.name: c for c in doctor.gh_split_checks(self.cfg)}[
                    "GitHub read login"
                ]
            self.assertEqual(row.status, "FAIL", row.found)
            self.assertTrue(row.found.startswith(f"no login in {self.R}"), row.found)
            self.assertNotIn("?", row.found)
            self.assertNotIn("403", row.found)
            self.assertIn(fix, str(row.fix))

    def test_a_folder_with_no_login_is_not_probed(self):
        r = gh.probe(str(self.d / "none"), "o/r", 1)
        self.assertEqual((r["can_write"], r["held"], self.calls()), (False, False, []))
        self.assertIs(gh.probe(str(self.R), "o/r", 1)["held"], True)

    def test_a_read_login_that_may_push_can_write(self):
        """The probe asks GitHub for a push as well as a comment, each refused either way, so nothing is made (a branch
        with no name and no commit, a comment with no body): a read login that may push but not comment can write, an
        alarm as any other."""
        r, w = gh.probe(str(self.R), "o/r", 1), gh.probe(str(self.W), "o/r", 1)
        self.assertEqual(
            [(p["can_comment"], p["can_push"], p["can_write"]) for p in (r, w)],
            [(False, False, False), (True, True, True)],
        )
        self.assertEqual(r["why"], "a comment and a push: HTTP 403: refused")
        (self.R / "CAN_PUSH").write_text("")
        r = gh.probe(str(self.R), "o/r", 1)
        self.assertEqual(
            (r["can_comment"], r["can_push"], r["can_write"]), (False, True, True)
        )
        self.assertEqual(
            r["why"],
            "a comment: HTTP 403: refused; a push: HTTP 422 (validation): allowed",
        )
        pushes = {
            tuple(c["argv"]) for c in self.calls() if "/git/refs" in " ".join(c["argv"])
        }
        self.assertEqual(
            pushes,
            {("api", "-X", "POST", "repos/o/r/git/refs", "-f", "ref=", "-f", "sha=")},
        )
        alarms = [
            x for x in ghlogins.tick_lines(self.live, self.store) if x.startswith("!! ")
        ]
        self.assertEqual(len(alarms), 1, alarms)
        self.assertIn("the read login can write", alarms[0])
        self.assertIn("a push: HTTP 422", alarms[0])

    def test_the_image_test_s_stand_in_takes_the_probe_s_writes_as_probes(self):
        """The image test's stand-in GitHub (tests/fakes/fake_net.py) takes each write the probe asks for as one of the
        login check's probes, which the deploy path counts as no write of the deployment's, and answers a login that
        may make it as GitHub does (422)."""
        gh.probe(str(self.R), "o/r", 1)
        writes = [c["argv"] for c in self.calls() if is_write(c["argv"])]
        self.assertEqual(len(writes), 2, writes)
        for argv in writes:
            fields = dict(v.split("=", 1) for f, v in zip(argv, argv[1:]) if f == "-f")
            with self.subTest(path=argv[3]):
                self.assertEqual(argv[:3], ["api", "-X", "POST"])
                answer = fake_net.probe_answer(argv[3], fields)
                self.assertEqual((answer or {}).get("status"), "422", argv)

    def test_rk_doctor_checks_both_sides(self):
        from rkmon_lib import doctor

        checks = {c.name: c for c in doctor.gh_split_checks(self.cfg)}
        self.assertEqual(checks["GitHub read login"].status, "PASS")
        self.assertIn("read-only", checks["GitHub read login"].found)
        self.assertEqual(checks["GitHub write login"].status, "PASS")
        self.assertIn("can comment and push", checks["GitHub write login"].found)
        (self.R / "CAN_WRITE").write_text("")
        checks = {c.name: c for c in doctor.gh_split_checks(self.cfg)}
        self.assertEqual(checks["GitHub read login"].status, "FAIL")
        none = doctor.gh_split_checks({"repo": "o/r"})
        self.assertEqual(
            (none[0].status, none[0].name), ("WARN", "GitHub read/write split")
        )

    def test_the_write_side_through_the_broker_when_its_folder_is_not_ours(self):
        seen = []

        def run(argv, **kw):
            seen.append(argv)
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(
                    {
                        "dir": str(self.W),
                        "login": "w",
                        "http": 422,
                        "can_write": True,
                        "why": "x",
                    }
                ),
                "",
            )

        with mock.patch.dict(
            os.environ, {writer.BROKER_VAR: "/b/rk_write.py"}
        ), mock.patch.object(ghlogins.os, "access", lambda p, m: str(p) != str(self.W)):
            r = ghlogins.probe_side(self.cfg, "write", str(self.W), run=run)
        self.assertEqual(
            seen,
            [
                [
                    "sudo",
                    "-n",
                    "-u",
                    "rkwrite",
                    "/b/rk_write.py",
                    "gh-logins",
                    "--side",
                    "write",
                    "--json",
                ]
            ],
        )
        self.assertEqual((r["can_write"], r["via"]), (True, "the broker"))


# --- the broker ---------------------------------------------------------------------------------------------------------


class TestBroker(unittest.TestCase):
    def test_rk_routes_the_writers_commands_only_and_never_twice(self):
        env = {writer.BROKER_VAR: "/b/rk_write.py"}
        sudo = ["sudo", "-n", "-u", "rkwrite", "/b/rk_write.py"]
        self.assertEqual(
            writer.routed(["post", "7", "f.md", "--dry-run"], env),
            [*sudo, "post", "7", "f.md", "--dry-run"],
        )
        for argv in (
            ["teams-test"],
            ["ticket-file", "x"],
        ):  # the writer keeps the webhook and the atlassian login
            self.assertEqual(writer.routed(argv, env), [*sudo, *argv])
        for argv in (["status"], ["needs-you-teams"], [], ["tick"]):
            self.assertIsNone(writer.routed(argv, env), argv)
        self.assertIsNone(writer.routed(["post", "7", "f"], {}))  # a host: no broker
        self.assertIsNone(
            writer.routed(["post", "7", "f"], dict(env, RKMON_GH_BROKER="1"))
        )  # the broker's own rk

    def test_a_write_of_the_monitors_own_reaches_the_writer_with_its_mark(self):
        """A write of the monitor's own (its process's outward.OWN_VAR, or own=True from its code) goes to the writer
        with the word outward.OWN_WORD before the command's; the writer's run takes the word off and runs rk with the
        mark set, and its log line says whose the write is. A write you type carries no mark, and the writer's rk gets
        none (the gate keeps what it stops of the monitor's own, and refuses what you typed).
        """
        sudo = ["sudo", "-n", "-u", "rkwrite", "/b/rk_write.py"]
        typed = {writer.BROKER_VAR: "/b/rk_write.py"}
        mine = dict(typed, **{outward.OWN_VAR: "1"})
        argv = ["arm", "7"]
        self.assertEqual(writer.routed(argv, typed), [*sudo, *argv])
        self.assertEqual(writer.routed(argv, mine), [*sudo, outward.OWN_WORD, *argv])
        self.assertEqual(
            writer.routed(argv, typed, own=True), [*sudo, outward.OWN_WORD, *argv]
        )
        self.assertEqual(writer.routed(argv, mine, own=False), [*sudo, *argv])
        ran = []
        me = mock.Mock(pw_name="rkwrite", pw_dir="/var/lib/rkwrite")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            writer_run, "LOG", f"{d}/rk-write.log"
        ), mock.patch.object(
            writer_run.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(
            writer_run.os, "umask", lambda m: 0
        ), mock.patch.object(
            writer_run, "config_folders_outside", lambda protected: ""
        ):
            writer_run.main(
                [outward.OWN_WORD, *argv], environ={}, execve=lambda *a: ran.append(a)
            )
            writer_run.main(argv, environ={}, execve=lambda *a: ran.append(a))
            log = pathlib.Path(f"{d}/rk-write.log").read_text().splitlines()
        (_p1, own_argv, own_env), (_p2, typed_argv, typed_env) = ran
        self.assertEqual((own_argv[-2:], typed_argv[-2:]), (argv, argv))
        self.assertNotIn(outward.OWN_WORD, own_argv)
        self.assertEqual(own_env.get(outward.OWN_VAR), "1")
        self.assertNotIn(outward.OWN_VAR, typed_env)
        self.assertTrue(
            log[0].endswith("rk arm 7 (as rkwrite, the monitor's own)"), log
        )
        self.assertTrue(log[1].endswith("rk arm 7 (as rkwrite)"), log)

    def test_rkmon_main_sends_a_write_command_to_the_broker(self):
        import rkmon

        seen = []
        with mock.patch.dict(
            os.environ, {writer.BROKER_VAR: "/b/rk_write.py"}
        ), mock.patch.object(
            rkmon.subprocess,
            "run",
            lambda argv, **kw: seen.append(argv)
            or subprocess.CompletedProcess(argv, 3),
        ):
            self.assertEqual(
                rkmon.main(["notice", "7", "f.md", "--kind", "queue", "--key", "k"]), 3
            )
        self.assertEqual(seen[0][:5], ["sudo", "-n", "-u", "rkwrite", "/b/rk_write.py"])
        self.assertEqual(
            seen[0][5:], ["notice", "7", "f.md", "--kind", "queue", "--key", "k"]
        )

    def test_the_broker_refuses_all_but_rks_github_writes_and_runs_as_the_write_user_only(
        self,
    ):
        """The broker runs as the write user only; the writer's run (rkmon_lib/writer_run.py) refuses every command but
        the writer's, and runs those in its place as the writer: its own tree's rkmon.py, with the image's environment
        (in which the write login's folder is the layout's, whatever the configuration names, and the reads use it
        too), the monitor's home, that tree as the code every rk it starts runs, and none of the caller's
        environment."""
        import rk_write

        ran = []

        def execve(path, argv, env):
            ran.append((path, argv, env))

        me = mock.Mock(pw_name="rkwrite", pw_dir="/var/lib/rkwrite")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            writer_run, "LOG", f"{d}/rk-write.log"
        ), mock.patch.object(
            writer_run.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(
            writer_run.os, "umask", lambda m: 0
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            for argv in (
                ["status"],
                ["gh", "pr", "merge"],
                [],
                ["gh-logins"],
                ["shell"],
            ):
                with self.subTest(argv=argv):
                    self.assertEqual(writer_run.main(argv, execve=execve), 4)
            self.assertEqual(ran, [])
            writer_run.main(["post", "7", "f.md", "--dry-run"], execve=execve)
            writer_run.main(["gh-logins", "--side", "write", "--json"], execve=execve)
            log = pathlib.Path(f"{d}/rk-write.log").read_text()
        path, argv, env = ran[0]
        self.assertEqual(
            (path, argv[1:]),
            (
                sys.executable,
                ["-I", "-B", f"{ROOT}/rkmon.py", "post", "7", "f.md", "--dry-run"],
            ),
        )
        self.assertEqual(
            (env["RKMON_GH_BROKER"], env["HOME"], env["RKMON_CODE"]),
            ("1", "/var/lib/rkwrite", str(ROOT)),
        )
        cfg = {"repo": "o/r", "gh": {"write_config_dir": "/elsewhere"}}
        login = container_layout.WRITE_LOGIN_DIR
        self.assertEqual(gh.login_dirs(cfg, env), {"read": login, "write": login})
        self.assertFalse(
            [k for k in env if k.startswith(("GH_", "ANTHROPIC", "ATLASSIAN"))]
        )
        self.assertEqual(len(ran), 2)
        self.assertEqual(log.count("REFUSED"), 5)
        self.assertIn("rk post 7 f.md --dry-run (as rkwrite)", log)
        other = mock.Mock(pw_name="rkmon", pw_dir="/home/rkmon")
        with mock.patch.object(
            rk_write.pwd, "getpwuid", lambda uid: other
        ), mock.patch.object(
            rk_write, "log", lambda line: None
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(
                rk_write.main(["post", "7", "f"], run=lambda *a: ran.append(a)), 4
            )
        self.assertEqual(len(ran), 2)

    def test_what_the_session_writes_never_runs_as_the_write_user(self):
        """`rk post --premises` and `rk spec-post` run git and AFTER lines: never through the broker (their post alone
        goes through it), and refused inside it whatever the spelling; the broker's rk runs the image's gh.
        """
        env = {writer.BROKER_VAR: "/b/rk_write.py"}
        for argv in (
            ["post", "7", "f.md", "--premises", "s.json"],
            ["post", "7", "f.md", "--prem=s.json"],
            ["spec-post", "7", "--round", "fu1"],
        ):
            self.assertIsNone(writer.routed(argv, env), argv)
            self.assertTrue(writer.refused(argv), argv)
        self.assertEqual(writer.refused(["post", "7", "f.md", "--dry-run"]), "")
        import rkmon

        args = mock.Mock(premises="s.json", n=7, file="f.md", again=False)
        with mock.patch.dict(os.environ, {gh.BROKERED: "1"}), mock.patch(
            "builtins.print"
        ) as said:
            self.assertEqual(rkmon.cmd_post_checked(args, {}, None), 4)
        self.assertIn("runs as the session user", said.call_args.args[0])
        with mock.patch.dict(os.environ, {gh.BROKERED: "1"}):
            gh.configure({"repo": "o/r", "paths": {"gh": "/work/evil/gh"}})
        try:
            self.assertEqual(gh.BIN, "gh")
        finally:
            gh.configure({})

    def test_the_post_of_a_checked_post_goes_through_the_broker(self):
        """In the container `rk post --premises`'s checks and AFTER lines run here; its post is `rk post` through the
        broker, outside the state lock (the broker's rk takes it)."""
        from rkmon_lib import premises

        with tempfile.TemporaryDirectory() as d:
            copy = pathlib.Path(d) / "posted" / "p.md"
            ready = {"copy": copy, "text": "body\n", "sha": "s", "dev": "v"}
            ran, finished = [], []

            def run(argv, **kw):
                ran.append(argv)
                return subprocess.CompletedProcess(argv, 0, "posted: URL\n", "")

            def finish(*a, kept: bool):
                finished.append(kept)
                return ["saved"], True

            with mock.patch.dict(
                os.environ, {writer.BROKER_VAR: "/b/rk_write.py"}
            ), mock.patch.object(premises.subprocess, "run", run), mock.patch.object(
                premises, "finish", finish
            ):
                out = premises._post_ready(
                    {"repo": "o/r"}, None, 7, {}, ready, again=True, news=None
                )
        self.assertEqual(
            ran,
            [
                [
                    "sudo",
                    "-n",
                    "-u",
                    "rkwrite",
                    "/b/rk_write.py",
                    "post",
                    "7",
                    str(copy),
                    "--again",
                ]
            ],
        )
        self.assertEqual(out, (["posted: URL", "saved"], 0))
        self.assertEqual(finished, [False])  # posted, not kept

    def test_the_broker_passes_no_file_the_caller_could_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            mine, secret, protected = (
                d / "mine.md",
                d / "secret.md",
                d / "secrets-write",
            )
            mine.write_text("x")
            secret.write_text("x")
            protected.mkdir()
            (protected / "hosts.yml").write_text("x")
            link = d / "link.md"
            link.symlink_to(protected / "hosts.yml")
            me = {"SUDO_UID": str(os.getuid()), "SUDO_GID": str(os.getgid())}
            copies = d / "copies"
            copies.mkdir()
            out = writer_run.guarded_args(
                ["post", "7", str(d / "." / "mine.md"), "--dry-run"],
                [str(protected)],
                me,
                str(copies),
            )
            self.assertEqual(out[:2] + out[3:], ["post", "7", "--dry-run"])
            self.assertEqual(pathlib.Path(out[2]).parent, copies)
            self.assertEqual(pathlib.Path(out[2]).read_text(), "x")
            for bad in (
                str(link),
                str(protected / "hosts.yml"),
                f"--body={link}",
                str(d),  # a folder
                str(d / "not-yet.md"),  # an absolute name that names nothing yet
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    writer_run.guarded_args(
                        ["notice", "7", bad], [str(protected)], me, str(copies)
                    )
            # A relative name that names nothing passes as it is: rk opens it among the copies (its working folder).
            self.assertEqual(
                writer_run.guarded_args(["disarm", "7", "a/b"], [], me, str(copies)),
                ["disarm", "7", "a/b"],
            )
            secret.chmod(0o600)
            other = {"SUDO_UID": str(os.getuid() + 1), "SUDO_GID": str(os.getgid() + 1)}
            with mock.patch.object(
                writer_run.pwd, "getpwuid", lambda uid: mock.Mock(pw_name="nobody-here")
            ), mock.patch.object(
                writer_run.os, "getgrouplist", lambda name, gid: [gid]
            ):
                self.assertFalse(writer_run.caller_may(os.stat(secret), other))
                secret.chmod(0o644)
                self.assertTrue(writer_run.caller_may(os.stat(secret), other))
            # Not run by sudo: no caller to read for.
            self.assertFalse(writer_run.caller_may(os.stat(secret), {}))

    def test_a_link_swapped_in_after_the_check_is_never_what_rk_reads(self):
        """The writer's run checks a file argument on the descriptor it opened, and rk gets a copy of that: a symbolic
        link to the write login swapped in where the file was, after the check, changes nothing rk reads; one
        swapped in before the walk (after the path was resolved) fails the walk."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            protected, copies = d / "secrets-write", d / "copies"
            protected.mkdir()
            copies.mkdir()
            secret = protected / "hosts.yml"
            secret.write_text("oauth_token: FAKE-WRITE-TOKEN\n")
            body = d / "post.md"
            body.write_text("the review\n")
            me = {"SUDO_UID": str(os.getuid()), "SUDO_GID": str(os.getgid())}
            real_copy = writer_run._copy

            def swap_then_copy(fd, folder, name):
                body.unlink()
                body.symlink_to(
                    secret
                )  # the caller's swap, between the check and rk's read
                return real_copy(fd, folder, name)

            with mock.patch.object(writer_run, "_copy", swap_then_copy):
                out = writer_run.guarded_args(
                    ["post", "7", str(body)], [str(protected)], me, str(copies)
                )
            self.assertEqual(pathlib.Path(out[2]).read_text(), "the review\n")
            self.assertNotEqual(out[2], str(body))
            self.assertEqual(stat.S_IMODE(os.stat(out[2]).st_mode), 0o600)
            # Swapped between the resolution and the walk: the walk opens no link.
            body.unlink()
            body.write_text("the review\n")
            real_realpath = writer_run.os.path.realpath

            def resolve_then_swap(p):
                r = real_realpath(p)
                if p == str(body):
                    body.unlink()
                    body.symlink_to(secret)
                return r

            with mock.patch.object(
                writer_run.os.path, "realpath", resolve_then_swap
            ), self.assertRaises(ValueError) as caught:
                writer_run.guarded_args(
                    ["post", "7", str(body)], [str(protected)], me, str(copies)
                )
            self.assertNotIn("FAKE-WRITE-TOKEN", str(caught.exception))
            # A folder of the walk swapped for a link to the protected folder: refused the same way.
            sub = d / "sub"
            sub.mkdir()
            (sub / "hosts.yml").write_text("x")

            def swap_folder(p):
                r = real_realpath(p)
                if p.startswith(str(sub)):
                    (sub / "hosts.yml").unlink()
                    sub.rmdir()
                    sub.symlink_to(protected)
                return r

            with mock.patch.object(
                writer_run.os.path, "realpath", swap_folder
            ), self.assertRaises(ValueError):
                writer_run.guarded_args(
                    ["post", "7", str(sub / "hosts.yml")],
                    [str(protected)],
                    me,
                    str(copies),
                )

    def test_the_copies_are_the_write_users_and_go_when_rk_ends(self):
        """A run's copies are in a folder of their own, rk's working folder, and go when rk ends, whatever its code; a
        run whose broker was killed outright leaves its own, which a run ARGS_KEEP_S later removes.
        """
        import rk_write

        with tempfile.TemporaryDirectory() as home:
            first = rk_write.copy_dir(home, now=1000.0)
            base = pathlib.Path(home) / rk_write.ARGS_DIR
            self.assertEqual(stat.S_IMODE(base.stat().st_mode), 0o700)
            os.utime(first, (1000.0, 1000.0))
            second = rk_write.copy_dir(home, now=1000.0 + rk_write.ARGS_KEEP_S + 1)
            self.assertFalse(os.path.exists(first))
            self.assertTrue(os.path.isdir(second))
            ran = []

            def run(argv, env, cwd):
                ran.append((cwd, os.path.isdir(cwd)))
                return 3

            me = mock.Mock(pw_name="rkwrite", pw_dir=home)
            with mock.patch.object(
                rk_write.pwd, "getpwuid", lambda uid: me
            ), mock.patch.object(
                rk_write, "picked", lambda: str(ROOT)
            ), mock.patch.object(
                rk_write, "problems", lambda tree, lstat: []
            ), mock.patch.object(
                rk_write, "links", lambda lstat: []
            ):
                code = rk_write.main(
                    ["post", "7", "f.md", "--dry-run"], run=run, environ={}
                )
            [(cwd, there)] = ran
            self.assertEqual(
                (pathlib.Path(cwd).parent, there), (base, True)
            )  # rk's working folder: this run's copies
            self.assertEqual(code, 3)  # rk's code
            self.assertFalse(os.path.exists(cwd))
            self.assertTrue(os.path.isdir(second))  # another run's, still running

    # docker/rk_write.py's run_command in a process of its own, as the broker runs rk: argv is the code's folder, the
    # working folder, then the command.
    BROKER = (
        "import sys; sys.path.insert(0, sys.argv[1]); import rk_write;"
        " sys.exit(rk_write.run_command(sys.argv[3:], {'GIVEN': '1'}, sys.argv[2]))"
    )

    def broker(self, cwd: str, *argv: str) -> list[str]:
        """The command that runs ARGV in CWD through the broker's run_command."""
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            self.BROKER,
            str(ROOT / "docker"),
            cwd,
            *argv,
        ]

    @staticmethod
    def wait_for(path: str, timeout_s: float = 60) -> None:
        end = time.monotonic() + timeout_s
        while not os.path.exists(path):
            if time.monotonic() > end:
                raise AssertionError(f"{path} never came")
            time.sleep(0.02)

    def test_rk_runs_as_the_brokers_child_and_the_broker_exits_with_its_code(self):
        """run_command: the command in the folder given, with the environment given only; its exit code, 128 plus the
        number of the signal that ended it, or 127 when it could not start."""
        with tempfile.TemporaryDirectory() as d:
            d = os.path.realpath(d)
            seen = f"(os.getcwd(), os.environ.get('GIVEN'), 'NOT_GIVEN' in os.environ) == ({d!r}, '1', False)"
            for argv, code in (
                (
                    [
                        sys.executable,
                        "-c",
                        f"import os, sys; sys.exit(3 if {seen} else 1)",
                    ],
                    3,
                ),
                (
                    [
                        sys.executable,
                        "-c",
                        "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
                    ],
                    128 + signal.SIGKILL,
                ),
                ([f"{d}/no-such-program"], 127),
            ):
                with self.subTest(code=code):
                    p = subprocess.run(
                        self.broker(d, *argv),
                        env=dict(os.environ, NOT_GIVEN="1"),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(p.returncode, code, p.stderr)
            self.assertIn("no-such-program did not start", p.stderr)

    def test_a_signal_the_broker_gets_goes_on_to_rk(self):
        """Each signal of RELAYED sent to the broker reaches rk, which ends as it chooses, and the broker exits with its
        code: no signal ends the broker before rk."""
        import rk_write

        child = (
            "import signal, sys, time\n"
            "def stop(n, _frame):\n"
            "    with open('got', 'w') as f:\n"
            "        f.write(str(n))\n"
            "    sys.exit(7)\n"
            "signal.signal(int(sys.argv[1]), stop)\n"
            "open('ready', 'w').close()\n"
            "time.sleep(60)\n"
        )
        for sig in rk_write.RELAYED:
            with self.subTest(signal=sig.name), tempfile.TemporaryDirectory() as d:
                p = subprocess.Popen(
                    self.broker(d, sys.executable, "-c", child, str(int(sig)))
                )
                try:
                    self.wait_for(f"{d}/ready")
                    p.send_signal(sig)
                    self.assertEqual(p.wait(timeout=60), 7)
                finally:
                    p.kill()
                    p.wait()
                self.assertEqual(pathlib.Path(d, "got").read_text(), str(int(sig)))

    def test_a_keyboards_sigint_reaches_rk_once(self):
        """On a terminal, a typed ^C sends SIGINT to the broker and to rk, its child: the broker passes on no second
        one."""
        child = (
            "import signal, time\n"
            "got = []\n"
            "signal.signal(signal.SIGINT, lambda n, _frame: got.append(n))\n"
            "print('ready', flush=True)\n"
            "end = time.monotonic() + 60\n"
            "while not got and time.monotonic() < end:\n"
            "    time.sleep(0.02)\n"
            "time.sleep(1)  # a second SIGINT, were one passed on, comes in this time\n"
            "print('SIGINT', len(got), flush=True)\n"
        )
        with tempfile.TemporaryDirectory() as d:
            argv = self.broker(d, sys.executable, "-c", child)
            pid, fd = pty.fork()
            if pid == 0:  # the broker, on a terminal of its own
                try:
                    os.execv(argv[0], argv)
                finally:
                    os._exit(127)  # never back into the tests
            seen, typed = b"", False
            while True:
                try:
                    data = os.read(fd, 4096)
                except OSError:  # the terminal closed: the broker is done
                    break
                if not data:
                    break
                seen += data
                if not typed and b"ready" in seen:
                    os.write(fd, b"\x03")
                    typed = True
            os.close(fd)
            _pid, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, seen)
        self.assertIn(b"SIGINT 1", seen)

    def test_the_brokers_commands_run_with_umask_027(self):
        """The writer's rk runs with umask 027: rkmon_lib/writer_run.py sets it before rk runs in its place."""
        umasks, ran = [], []
        me = mock.Mock(pw_name="rkwrite", pw_dir="/var/lib/rkwrite")
        with mock.patch.object(
            writer_run.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(writer_run, "log", lambda line: None), mock.patch.object(
            writer_run, "config_folders_outside", lambda protected: ""
        ), mock.patch.object(
            writer_run.os, "umask", umasks.append
        ):
            writer_run.main(
                ["post", "7", "--dry-run"],
                environ={},
                execve=lambda *a: ran.append(umasks[:]),
            )
        self.assertEqual(ran, [[0o027]])

    def test_the_writer_passes_on_the_wrapper_s_name_checked_again(self):
        """The writer's rk gets the name the host's wrapper was run by, which its fixes show, while it is a path that
        ends in rkmon (container_layout.shown_wrapper), else as rkmon, whatever started the writer: unset, with
        another character, of another name, too long. It is shown only: the writer runs the same command with any.
        """
        me = mock.Mock(pw_name="rkwrite", pw_dir="/var/lib/rkwrite")
        for given, shown in (
            (None, "rkmon"),
            ("./rkmon", "./rkmon"),
            ("../docker/rkmon", "../docker/rkmon"),
            ("./rkmon\nfix: run this", "rkmon"),
            ("\x1b[2J./rkmon", "rkmon"),
            ("./rkmonx", "rkmon"),
            ("a" * writer_run.container_layout.WRAPPER_MAX + "/rkmon", "rkmon"),
        ):
            ran = []
            with self.subTest(given=given), mock.patch.object(
                writer_run.pwd, "getpwuid", lambda uid: me
            ), mock.patch.object(
                writer_run, "log", lambda line: None
            ), mock.patch.object(
                writer_run, "config_folders_outside", lambda protected: ""
            ), mock.patch.object(
                writer_run.os, "umask", lambda m: 0
            ):
                writer_run.main(
                    ["post", "7", "--dry-run"],
                    environ={} if given is None else {"RKMON_WRAPPER": given},
                    execve=lambda *a: ran.append(a),
                )
                [(program, argv, env)] = ran
                self.assertEqual(env["RKMON_WRAPPER"], shown)
                self.assertEqual(
                    [program, *argv],
                    [
                        sys.executable,
                        sys.executable,
                        "-I",
                        "-B",
                        f"{writer_run.TREE}/rkmon.py",
                        "post",
                        "7",
                        "--dry-run",
                    ],
                )

    def test_the_broker_refuses_a_configuration_that_writes_where_the_write_user_keeps_its_login(
        self,
    ):
        import rk_write

        ran = []
        me = mock.Mock(pw_name="rkwrite", pw_dir="/var/lib/rkwrite")

        def child(args, env, cwd):
            """The broker's child: rkmon_lib/writer_run.py in the broker's copies folder; RAN notes what it runs."""
            with contextlib.chdir(cwd):
                return writer_run.main(
                    list(args[6:]), environ=env, execve=lambda *a: ran.append(a)
                )

        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            rk_write, "LOG", f"{d}/rk-write.log"
        ), mock.patch.object(
            rk_write.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(
            writer_run, "LOG", f"{d}/rk-write.log"
        ), mock.patch.object(
            rk_write, "picked", lambda: str(ROOT)
        ), mock.patch.object(
            rk_write, "problems", lambda *a, **kw: []
        ), mock.patch.object(
            rk_write, "links", lambda *a, **kw: []
        ), mock.patch.object(
            writer_run,
            "config_folders_outside",
            lambda protected: "paths.posts (/secrets-write/gh)",
        ), mock.patch.object(
            rk_write, "copy_dir", lambda home: tempfile.mkdtemp(prefix="run-", dir=d)
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(
                rk_write.main(["post", "7", "f.md"], run=child, environ={}),
                4,
            )
            self.assertIn("paths.posts", pathlib.Path(f"{d}/rk-write.log").read_text())
            # This run's copies went with the refusal: the log alone is left.
            self.assertEqual(os.listdir(d), ["rk-write.log"])
        self.assertEqual(ran, [])
        # The configuration is read from the home the broker's rk runs with (sudo passes no RKMON_HOME).
        from rkmon_lib import config

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {}
        ), mock.patch.dict(writer_run.BASE_ENV, {"RKMON_HOME": d}):
            os.environ.pop("RKMON_HOME", None)
            raw = config.load_defaults()
            raw.pop("_doc", None)
            raw.update(
                repo="o/r", rocke_path="a/b", operator={"github_login": "op-login"}
            )
            raw["paths"] = dict(raw["paths"], working_dir=f"{d}/work")
            raw["paths"]["posts"] = f"{d}/protected/gh"
            pathlib.Path(d, "config.json").write_text(json.dumps(raw))
            self.assertEqual(
                writer_run.config_folders_outside([f"{d}/protected"]),
                f"paths.posts ({d}/protected/gh)",
            )
            self.assertEqual(writer_run.config_folders_outside([f"{d}/elsewhere"]), "")
            self.assertNotIn(
                "RKMON_HOME", os.environ
            )  # this process's environment as it was

    def test_inside_the_broker_reads_use_the_write_login(self):
        cfg = {"repo": "o/r", "gh": {"read_config_dir": "/r", "write_config_dir": "/w"}}
        self.assertEqual(gh.login_dirs(cfg, {}), {"read": "/r", "write": "/w"})
        self.assertEqual(
            gh.login_dirs(cfg, {"RKMON_GH_BROKER": "1"}), {"read": "/w", "write": "/w"}
        )

    def test_the_places_the_load_derives_are_the_secrets_folders(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.normalize(
                {"repo": "o/r", "features": {}, "paths": {"working_dir": d}}
            )
            got = gh.login_dirs(cfg, {})
            self.assertEqual(
                got, {"read": f"{d}/.config/gh-read", "write": f"{d}/.config/gh"}
            )
            self.assertEqual(
                gh.login_dirs({"repo": "o/r"}, {}), {"read": "", "write": ""}
            )  # before the package

    def test_with_no_read_login_yet_the_login_setup_made_reads_and_writes(self):
        """A deployment set up before the split keeps working: its one login reads and writes, gh.write takes it, and
        GH_LOGINS says so once, with no probe and no alarm, until setup adds the read login.
        """
        with tempfile.TemporaryDirectory() as d:
            cfg = config.normalize(
                {"repo": "o/r", "features": {}, "paths": {"working_dir": d}}
            )
            old = pathlib.Path(d) / ".config" / "gh"
            old.mkdir(parents=True)
            (old / "hosts.yml").write_text("github.com: {}\n")
            self.assertEqual(
                gh.login_dirs(cfg, {}),
                {"read": str(old), "write": str(old), "unsplit": str(old)},
            )
            gh.configure(cfg)
            try:
                with mock.patch.object(
                    gh, "_exec", lambda args, folder, timeout, errors="strict": folder
                ):
                    self.assertEqual(gh.write(["pr", "comment", "7"]), str(old))
                c, _ = ghlogins.check(cfg)
                self.assertEqual((c["sides"], ghlogins.alarms(cfg, c)), ({}, []))
                self.assertIn(
                    f"no read/write split yet: the login in {old} reads and writes",
                    ghlogins.summary(c),
                )
                from rkmon_lib import doctor

                (w,) = doctor.gh_split_checks(cfg)
                self.assertEqual(
                    (w.status, w.found),
                    (
                        "WARN",
                        f"none yet: the login in {old} reads and writes (no read login)",
                    ),
                )
                # once the read login is made, the split holds
                read = pathlib.Path(d) / ".config" / "gh-read"
                read.mkdir()
                (read / "hosts.yml").write_text("github.com: {}\n")
                gh.configure(cfg)
                self.assertEqual((gh.READ_DIR, gh.UNSPLIT), (str(read), ""))
            finally:
                gh.configure({})
            # a read folder the configuration names never falls back
            named = dict(cfg, gh={"read_config_dir": f"{d}/elsewhere"})
            self.assertNotIn("unsplit", gh.login_dirs(named, {}))


# --- the host's split helper ----------------------------------------------------------------------------------------------


def mode(p: pathlib.Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


class TestGhSplitTool(FakeGh):
    """In a scratch HOME and secrets folder, with fake logins: each login ends where the load looks (gh.login_dirs),
    and folders end 700 and files 600."""

    def setUp(self):
        super().setUp()
        self.home = self.d / "home"
        secrets = self.d / "secrets"  # the monitor's secrets folder (paths.secrets_dir)
        toolkit = self.d / "toolkit"  # the monitor's home, which holds config.json
        for folder in (self.home / ".config", secrets, toolkit, self.d / "work"):
            folder.mkdir(parents=True)
        secrets.chmod(0o700)
        (toolkit / "config.json").write_text(
            json.dumps(
                {
                    "repo": "o/r",
                    "paths": {
                        "working_dir": str(self.d / "work"),
                        "secrets_dir": str(secrets),
                    },
                }
            )
        )
        # gh's own folder, and each login's folder where the load looks (gh.login_dirs)
        self.default = self.home / ".config" / "gh"
        self.read, self.write = secrets / "gh-read", secrets / "gh"
        for folder, token in ((self.default, "WRITE-TOKEN"), (self.read, "READ-TOKEN")):
            folder.mkdir()
            (folder / "hosts.yml").write_text(f"github.com:\n  oauth_token: {token}\n")
            (folder / "config.yml").write_text("version: 1\n")
            (folder / "hosts.yml").chmod(0o600)
        self.default.chmod(0o751)
        # the fake gh's mark: this login may write
        (self.default / "CAN_WRITE").write_text("")
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GH_CONFIG_DIR", "XDG_CONFIG_HOME")
            and not k.startswith("RKMON_")
        }
        self.env.update(
            HOME=str(self.home),
            RKMON_HOME=str(toolkit),
            GH_SPLIT_REPO="o/r",
            GH_SPLIT_ISSUE="1",
        )

    def run_tool(self, action: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "tools" / f"gh_split_{action}.py"),
                *args,
            ],
            capture_output=True,
            text=True,
            env=self.env,
        )

    def token(self, folder: pathlib.Path) -> str:
        return (folder / "hosts.yml").read_text().split("oauth_token: ")[1].strip()

    def refused_and_nothing_moved(self, said: str = ""):
        """apply refuses (exit 2), saying SAID, and changes nothing: each login stays in its folder."""
        r = self.run_tool("apply", "--force")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn(said, r.stderr)
        self.assertIn("nothing changed", r.stderr)
        self.assertEqual(
            (self.token(self.default), self.token(self.read)),
            ("WRITE-TOKEN", "READ-TOKEN"),
        )
        self.assertFalse(self.write.exists())

    def test_apply_then_rollback(self):
        r = self.run_tool("apply", "--force")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # gh's own folder holds a copy of the read login, which keeps its own folder
        self.assertEqual(
            (self.token(self.default), self.token(self.write), self.token(self.read)),
            ("READ-TOKEN", "WRITE-TOKEN", "READ-TOKEN"),
        )
        for folder in (self.default, self.write):
            self.assertEqual(mode(folder), 0o700, folder)
            for f in folder.iterdir():
                self.assertEqual(mode(f), 0o600, f)
        probes = [c for c in self.calls() if c["argv"][:3] == ["api", "-X", "POST"]]
        # two logins before, two after, each a comment and a push; GitHub refuses each
        self.assertEqual(len(probes), 8)
        # applied already
        self.assertEqual(self.run_tool("apply", "--force").returncode, 2)
        r = self.run_tool("rollback")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(
            (self.token(self.default), self.token(self.read), self.write.exists()),
            ("WRITE-TOKEN", "READ-TOKEN", False),
        )

    def test_apply_changes_nothing_when_the_logins_are_not_what_it_expects(self):
        (self.default / "CAN_WRITE").unlink()
        self.refused_and_nothing_moved()

    def test_apply_refuses_a_write_login_that_may_push_but_not_comment(self):
        """The write login must be able to comment: one that may push but not comment is refused."""
        (self.default / "CAN_WRITE").unlink()
        # the fake gh's mark: this login may push
        (self.default / "CAN_PUSH").write_text("")
        self.refused_and_nothing_moved(
            f"the login in {self.default} should be the write login, which may comment: can push but not comment"
        )

    def test_apply_refuses_a_read_login_that_may_push_but_not_comment(self):
        """The read login must make neither: one that may push, though not comment, is refused."""
        (self.read / "CAN_PUSH").write_text("")
        self.refused_and_nothing_moved(
            f"the login in {self.read} should be the read login, which may neither comment nor push: can push "
            "but not comment"
        )

    def test_apply_refuses_without_a_read_login(self):
        """With no read login where the load looks, apply names the command that sets it, and changes nothing."""
        (self.read / "hosts.yml").unlink()
        r = self.run_tool("apply", "--force")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn(f"no read login in {self.read}: ", r.stderr)
        self.assertIn(" login github_read sets it; nothing changed", r.stderr)
        self.assertEqual(self.token(self.default), "WRITE-TOKEN")
        self.assertFalse(self.write.exists())

    def test_rollback_keeps_a_login_in_ghs_own_folder_that_may_write(self):
        """rollback removes the login in gh's own folder only when its probe says that it reads only, as the read
        login's copy does: one that may write stays, and nothing changes."""
        self.assertEqual(self.run_tool("apply", "--force").returncode, 0)
        # the fake gh's mark: this login may write
        (self.default / "CAN_WRITE").write_text("")
        r = self.run_tool("rollback")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("nothing changed", r.stderr)
        self.assertEqual(
            (self.token(self.default), self.token(self.write)),
            ("READ-TOKEN", "WRITE-TOKEN"),
        )

    def test_apply_rolls_back_when_the_check_after_fails(self):
        # after the move, the default login still writes
        self.env["FAKE_CAN_WRITE_DIR"] = str(self.default)
        r = self.run_tool("apply", "--force")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("rolling back", r.stderr)
        self.assertEqual(
            (self.token(self.default), self.token(self.read)),
            ("WRITE-TOKEN", "READ-TOKEN"),
        )
        self.assertFalse(self.write.exists())


if __name__ == "__main__":
    unittest.main()
