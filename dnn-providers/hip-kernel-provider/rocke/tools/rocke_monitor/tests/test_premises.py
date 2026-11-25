# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk post N FILE --premises SPEC` (lessons L-224, L-229, L-509): the verifier's
premises are re-checked live, then the post's own checks run, in the post script's order and with its messages; any
failed one stops the post. `--dry-run` stops before the post. The git side is a real repository; GitHub is faked.
"""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    features,
    gh,
    outward,
    post,
    premises,
    report,
    review_cost,
    spend,
    state_store,
    webdash,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

REAL_RUN = subprocess.run
LINE1 = (
    f"Automated review by rocke-monitor: Model - qwen, effort: {config.MODEL_DEFAULT}"
)
TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket
TAG = f"{TK}-1234"
T1, T2, STOP = "2026-11-05T10:00:00Z", "2026-11-05T11:00:00Z", "2026-11-05T15:00:00Z"
POST = f"""{LINE1}

## PR #7 — t (dev, {TAG}) — 🟡 Follow-up: fixes hold

| Prior item | Status |
|---|---|
| Ours (11-04) N1: x | 🟢 FIXED `abc1234`<br>• ran |
| Bots / CI | 🟡 running<br>• (sanity queued); 99 ✓ |
| Code quality | 🟢 None yet |

---
**Legend:** 🔴 blocking · 🟠 should-fix · 🟡 minor nit · 🟢 info/good · ⚪ retracted

_Follow-up review: first pass by a local qwen model, verified by the orchestrator (Claude Opus 5.5). Flags whether a \
closer look is needed; not an approval or rejection._
"""


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(repo, files, msg):
    for f, text in files.items():
        p = pathlib.Path(repo, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    g(repo, "add", "-A")
    g(repo, "commit", "-qm", msg)
    return g(repo, "rev-parse", "HEAD")


def run(name, status, conclusion, started):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started": started,
        "url": "u",
    }


HELD = "features.review_automation.mode is prototype: nothing goes out until it is live"  # what a post of review automation in prototype is held by


class TestPostPremises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_dir = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls.repo_dir.name)
        origin, cls.mirror = d / "origin", d / "repo"
        g(d, "init", "-q", "-b", "develop", "origin")
        base = commit(
            origin, {"rk/a.py": "1\n", "rk/b.py": "1\n", "other/c.txt": "1\n"}, "base"
        )
        g(origin, "checkout", "-q", "-b", "pr")
        cls.head = commit(origin, {"rk/a.py": "pr\n", "rk/new.py": "x\n"}, "the PR")
        g(origin, "update-ref", "refs/pull/7/head", cls.head)
        g(origin, "checkout", "-q", "develop")
        cls.devbase = commit(
            origin, {"rk/a.py": "dev\n"}, "develop changes a.py: the PR conflicts"
        )
        cls.dev = commit(
            origin, {"other/c.txt": "2\n"}, "develop moves on, away from the PR's files"
        )
        g(origin, "checkout", "-q", "-b", "dev3")
        cls.dev3 = commit(
            origin, {"rk/new.py": "y\n"}, "develop adds the PR's new file too"
        )
        g(origin, "checkout", "-q", "develop")
        g(d, "init", "-q", "repo")
        g(cls.mirror, "remote", "add", "origin", str(origin))
        g(
            cls.mirror,
            "fetch",
            "-q",
            "origin",
            "+develop:refs/remotes/origin/develop",
            "+dev3:refs/heads/dev3",
        )

    @classmethod
    def tearDownClass(cls):
        cls.repo_dir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = d = pathlib.Path(self.tmp.name)
        self.cfg = {
            "repo": "ROCm/r",
            "operator": {"github_login": "me"},
            "worker": {"model": "qwen"},  # the local worker drafts: line 1 is LINE1
            "review_first_line": LINE1,
            "max_posts_per_hour": 6,
            "features": {"review_automation": {"mode": "live"}},
            "paths": {
                "working_dir": str(d),
                "repo": str(self.mirror),
                "workers": str(d / "workers"),
                "daily_dir": str(d / "daily"),
                "posts": str(d / "posts"),
                "posted_dir": str(d / "posted"),
                "lessons": str(d),
            },
            "orchestrator": {"model": "claude-opus-5-5"},
        }
        # the logins' places, as the load derives them (config.normalize): rk post names the write login's
        self.cfg["gh"] = config.normalize(self.cfg)["gh"]
        self.store = Store(d / "state")
        self.st = empty_state()
        self.st["prs"]["7"] = {
            "n": 7,
            "head": self.head,
            "status": "open",
            "draft": False,
            "author": "dev",
            "title": "t",
            "ours": {"head": self.devbase, "count": 2, "verdict": "🟠"},
        }
        self.store.save(self.st)
        rev = d / "tmp/v7fu2"
        rev.mkdir(parents=True)
        self.src = rev / "pr7_fu2.md"
        self.src.write_text(POST, encoding="utf-8")
        (rev / "notes.md").write_text("premises: head x\n", encoding="utf-8")
        (rev / "files.txt").write_text(
            "rk/a.py\nrk/old.py\n", encoding="utf-8"
        )  # rk post --premises reads it too
        self.copy = d / "posts/pr7_fu2.md"
        self.spec = {
            "N": "7",
            "ROUND": "fu2",
            "HEAD10": self.head[:10],
            "LIGHT": "🟡: fixes hold",
            "DEVBASE": self.devbase[:10],
            "THREADS": "3 2",
            "STOP": STOP,
            "CONFLICTS": "rk/a.py ",
            "SRC": str(self.src),
            "NOTES": str(rev / "notes.md"),
            "TAG": TAG,
            "MAXCANCEL": "1",
            "FAILOK": ["Windows stage"],
            "RUNOK": ["sanity"],
            "fill": ["(sanity queued); 99 ✓", "(sanity queued); $ok ✓"],
            "FOOTER": "_Follow-up review",
            "SAVE": "fu2: N1 FIXED (ran)",
            "OUT": "/nonexistent/post7fu2.sh",
            "AFTER": [
                'echo "decided $RK"',
                f'printf "%s %s %s" "$sha" "$dev" "$P" > {d}/after.txt',
            ],
        }
        live = {
            "sha": self.head,
            "state": "open",
            "merged": False,
            "updated_at": "2026-11-05T15:30:00Z",
        }
        self.world = {
            "live": live,
            "dev": self.dev,
            "body": f"Fixes the bias.\n\nISSUE ID : {TAG}\n",
            "runs": [
                run("build", "completed", "success", T1),
                run("pre-commit", "completed", "success", T1),
                run(
                    "pre-commit", "completed", "success", T2
                ),  # a re-run: both count as ✓
                run("test (shard 1/2)", "completed", "failure", T1),
                run(
                    "test (shard 1/2)", "completed", "success", T2
                ),  # the re-run passed
                run("Windows stage", "completed", "failure", T1),
                run("pr-bot", "completed", "cancelled", T1),
                run("sanity", "queued", "-", ""),
            ],
            "resolved": [True, True, False],
            "reviews": ["2026-11-05T14:00:00Z", None],
            "review_comments": ["2026-11-05T14:30:00Z"],
            "comments": ["2026-11-05T14:59:59Z"],
            "fail": None,
        }
        self.comments_posted = []
        for target, name, value in (
            (rkmon, "load_config", lambda: self.cfg),
            (rkmon, "Store", lambda: self.store),
            (gh, "rest", self.rest),
            (gh, "graphql", self.graphql),
            (subprocess, "run", self.subprocess_run),
            (premises, "now", lambda: "2026-11-05T16:00:00Z"),
        ):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        (d / "posted").mkdir()

    # --- the fakes -----------------------------------------------------------------------------------------------
    def rest(self, path, jq=None, paginate=False, timeout=120):
        w = self.world
        if w["fail"] and w["fail"] in path:
            raise gh.GhError(f"gh api {path}: HTTP 502")
        if path == "repos/ROCm/r/pulls/7":
            return {
                premises.LIVE_JQ: lambda: dict(w["live"]),
                "{body}": lambda: {"body": w["body"]},
                ".head.sha": lambda: w["live"]["sha"],
                "{title, body}": lambda: {"title": "feat: t", "body": w["body"]},
                "{sha: .head.sha, base: .base.ref, body}": lambda: {
                    "sha": w["live"]["sha"],
                    "base": w["live"].get("base", "develop"),
                    "body": w["body"],
                },
            }[jq]()
        if path == "repos/ROCm/r/commits/develop":
            return w["dev"]
        if (
            path == f"repos/ROCm/r/commits/{w['live']['sha']}/check-runs?per_page=100"
            and paginate
        ):
            return [dict(r) for r in w["runs"]]
        if path == "repos/ROCm/r/pulls/7/reviews?per_page=100" and paginate:
            return list(w["reviews"])
        if path == "repos/ROCm/r/pulls/7/comments?per_page=100" and paginate:
            return list(w["review_comments"])
        if path == "repos/ROCm/r/issues/7/comments?per_page=100" and paginate:
            self.assertIn(
                'select(.user.type != "Bot")', jq
            )  # bots' comments never stop a post
            return list(w["comments"])
        if path == "repos/ROCm/r/pulls/7/commits?per_page=100":
            return ["feat: t"]
        raise AssertionError(f"unexpected GET {path} {jq}")

    def graphql(self, query, timeout=120, attempts=3, **v):
        self.assertEqual((v, "reviewThreads(first: 100)" in query), ({"n": 7}, True))
        nodes = [{"isResolved": x} for x in self.world["resolved"]]
        return {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {"totalCount": len(nodes), "nodes": nodes}
                }
            }
        }

    def subprocess_run(self, args, **kw):
        if args[:3] == ["gh", "pr", "comment"]:
            self.comments_posted.append(args)
            return subprocess.CompletedProcess(
                args, 0, "https://github.com/ROCm/r/pull/7#issuecomment-1\n", ""
            )
        if args[:1] == ["gh"]:
            raise AssertionError(f"unexpected gh call {args}")
        return REAL_RUN(args, **kw)

    # --- helpers -------------------------------------------------------------------------------------------------
    def own(self):
        """This process's writes as the monitor's own (outward.OWN_VAR), as the Claude session's are."""
        return mock.patch.dict(os.environ, {outward.OWN_VAR: "1"})

    def rk(self, *argv):
        """`rk ARGV` here: its exit code and the lines it printed. Its standard output is caught, not print itself, so
        a module the command imports first keeps the real print as its default (applied.refuse's `out=print`).
        """
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(list(argv))
        text = out.getvalue()
        return code, text.removesuffix("\n").split("\n")

    def write_login(self) -> str:
        """What a dry run of rk post names: this configuration's write login (<SECRETS_DIR>/gh)."""
        return f"it would post under the write login (GH_CONFIG_DIR {gh.login_dirs(self.cfg)['write']})"

    def post(self, *extra, **changes):
        """`rk post 7 SRC --premises SPEC *extra`, the spec changed by `changes` (None drops a key)."""
        spec = {k: v for k, v in dict(self.spec, **changes).items() if v is not None}
        path = self.d / "spec.json"
        path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        return self.rk("post", "7", str(self.src), "--premises", str(path), *extra)

    def state_files(self):
        return {
            p.name: p.read_bytes()
            for p in sorted(self.store.dir.iterdir())
            if p.is_file() and p.name not in ("lock", "write.lock")
        }  # the package takes a write lock for each write command

    def stopped(self, last, *extra, **changes):
        """The post stops with `last` as its last line: nothing posted, written or recorded."""
        before = self.state_files(), (
            self.copy.read_bytes() if self.copy.exists() else None
        )
        code, lines = self.post(*extra, **changes)
        self.assertEqual((code, lines[-1]), (4, last), lines)
        self.assertEqual(self.comments_posted, [])
        self.assertEqual(
            (
                self.state_files(),
                self.copy.read_bytes() if self.copy.exists() else None,
            ),
            before,
        )
        self.assertFalse((self.d / "notes").exists() or (self.d / "after.txt").exists())
        return lines

    def premise_lines(self):
        return [
            f"live: {self.head[:10]} open false 2026-11-05T15:30:00Z",
            f"develop: {self.dev[:10]}",
            "PR files: 3",
            "conflicts: [rk/a.py ]",
            "== premise (L-224): failures exactly as verified; cancels at most 1; nothing running beyond the "
            "verified set; the count is filled at post time",
            "latest-attempt failures 1; cancelled 1; raw success 4; running or queued 0",
            "   BAD Windows stage",
            "   RUNOK sanity",
            "threads total, resolved: 3 2",
            "new reviews 0; review comments 0; human comments 0",
        ]

    # --- the post ------------------------------------------------------------------------------------------------
    def test_it_posts_saves_runs_after_and_archives_the_notes(self):
        code, lines = self.post()
        self.assertEqual(code, 0, lines)
        self.assertEqual(
            lines,
            self.premise_lines()
            + [
                f"posted: https://github.com/ROCm/r/pull/7#issuecomment-1 (copy: {self.d}/posted/PR_FOLLOWUP_7.md)",
                "logged save (verify) for 7",
                f"decided {premises.rk_path()}",
                "notes archived",
                "cost recorded: draft qwen uncounted, verifier not given, quality not given",
            ],
        )
        # The copy posted carries the ✓ count of every successful run, attempts included (L-224): 4, of 3 latest.
        self.assertIn("• (sanity queued); 4 ✓ |", self.copy.read_text(encoding="utf-8"))
        self.assertEqual(
            self.comments_posted,
            [
                [
                    "gh",
                    "pr",
                    "comment",
                    "7",
                    "--repo",
                    "ROCm/r",
                    "--body-file",
                    str(self.copy),
                ]
            ],
        )
        self.assertTrue(self.store.ledger_has("post", 7, self.head))
        saves = [
            r
            for r in self.store.read_jsonl("daylog.jsonl")
            if r.get("save") == "verify"
        ]
        self.assertEqual([r["text"] for r in saves], ["fu2: N1 FIXED (ran)"])
        self.assertEqual(
            (self.d / "after.txt").read_text(), f"{self.head} {self.dev} {self.copy}"
        )
        self.assertEqual(
            (self.d / "notes/2026-11-05_pr7_fu2.md").read_text(), "premises: head x\n"
        )
        self.assertEqual(self.store.load()["prs"]["7"]["ours"]["count"], 3)
        # Once posted, the same spec stops at once: the copy exists.
        self.assertEqual(
            self.post()[1][-1],
            f"{self.copy} exists: round fu2 was posted or kept; a new review takes a new "
            "ROUND: stop",
        )

    def test_a_dry_run_stops_before_the_post(self):
        before = self.state_files()
        code, lines = self.post("--dry-run")
        self.assertEqual(
            (code, lines),
            (
                0,
                self.premise_lines()
                + [
                    f"dry run: the premises hold and rk post's guards pass; stopped before the post ({self.copy} not "
                    f"written; {self.write_login()})"
                ],
            ),
        )
        self.assertEqual(
            (self.comments_posted, self.copy.exists(), self.state_files()),
            ([], False, before),
        )
        self.assertFalse((self.d / "notes").exists() or (self.d / "after.txt").exists())
        # rk post's own guards run in a dry run too: we have not reviewed #7, so a Follow-up header is refused.
        del self.st["prs"]["7"]["ours"]
        self.store.save(self.st)
        before = self.state_files()
        code, lines = self.post("--dry-run")
        self.assertEqual(
            (code, lines[-2:]),
            (
                4,
                [
                    "REFUSED: the header says Follow-up, but we have not reviewed #7 yet: post the first round",
                    "post refused: read the message",
                ],
            ),
        )
        self.assertEqual(
            (self.comments_posted, self.copy.exists(), self.state_files()),
            ([], False, before),
        )
        # A plain post's dry run: its guards only.
        self.assertEqual(
            self.rk("post", "7", str(self.src), "--dry-run"),
            (
                4,
                [
                    "REFUSED: the header says Follow-up, but we have not reviewed #7 yet: post the first round"
                ],
            ),
        )
        self.st["prs"]["7"]["ours"] = {"head": self.devbase, "count": 2}
        self.store.save(self.st)
        self.assertEqual(
            self.rk("post", "7", str(self.src), "--dry-run"),
            (
                0,
                [
                    f"dry run: rk post's guards pass; stopped before the post ({self.write_login()})"
                ],
            ),
        )
        self.assertEqual(self.comments_posted, [])

    def test_a_refused_post_removes_the_copy(self):
        del self.st["prs"]["7"]["ours"]
        self.store.save(self.st)
        self.stopped("post refused: read the message")
        self.assertFalse(self.copy.exists())

    def test_after_the_post_a_failed_step_is_said_and_the_rest_runs(self):
        (self.d / "notes").mkdir()
        (self.d / "notes/2026-11-05_pr7_fu2.md").write_text("an earlier archive")
        code, lines = self.post(AFTER=["echo out; exit 3"])
        self.assertEqual(
            (code, lines[-5:]),
            (
                1,
                [
                    "logged save (verify) for 7",
                    "out",
                    "!! AFTER failed (exit 3): echo out; exit 3",
                    "notes archive exists: not overwritten",
                    "cost recorded: draft qwen uncounted, verifier not given, "
                    "quality not given",
                ],
            ),
        )
        self.assertEqual(len(self.comments_posted), 1)

    # --- each premise stops it ---------------------------------------------------------------------------------
    def test_the_head_and_the_pr_state(self):
        self.world["live"]["sha"] = "f" * 40
        self.assertEqual(
            self.stopped(
                f"head moved: live ffffffffff, the spec verified {self.head[:10]}: re-verify "
                "at the new head: stop"
            ),
            [
                "live: ffffffffff open false 2026-11-05T15:30:00Z",
                f"head moved: live ffffffffff, the spec verified {self.head[:10]}: re-verify at the new head: "
                "stop",
            ],
        )
        self.world["live"].update(sha=self.head, state="closed", merged=True)
        self.stopped("premise: the PR is closed, merged: nothing to post: stop")

    def test_develop_changed_the_prs_files(self):
        self.world["dev"] = self.dev3
        lines = self.stopped(
            f"premise: develop changed 1 of this PR's own files since DEVBASE {self.devbase[:10]} "
            f"(above): re-check them at {self.dev3[:10]} (review_procedure step 4), then set DEVBASE: "
            "stop"
        )
        self.assertEqual(lines[-2], "rk/new.py")

    def test_a_pr_on_a_feature_branch_is_checked_against_its_base(self):
        """For PRs that target feature branches, DEVBASE, the PR's files and the conflict set are about the
        base branch (`base.ref`), fetched into the mirror, not develop."""
        self.world["live"]["base"] = "dev3"
        lines = self.stopped(
            f"premise: the base dev3 changed 1 of this PR's own files since DEVBASE "
            f"{self.devbase[:10]} (above): re-check them at {self.dev3[:10]} (review_procedure step 4), "
            "then set DEVBASE: stop"
        )
        self.assertEqual(
            (lines[1], lines[-2]), (f"base dev3: {self.dev3[:10]}", "rk/new.py")
        )
        self.assertEqual(
            g(self.mirror, "rev-parse", "refs/remotes/origin/dev3"), self.dev3
        )

    def test_the_conflict_set(self):
        self.stopped(
            "premise: the conflict set changed: now [rk/a.py], the spec has []: re-check the merge: stop",
            CONFLICTS="",
        )

    def test_crlf_in_the_post(self):
        self.src.write_bytes(POST.replace("\n", "\r\n").encode("utf-8"))
        self.stopped(
            "the post has CRLF line ends (L-228): strip the carriage returns: stop"
        )

    def test_ci_at_the_latest_attempts(self):
        self.world["runs"].append(run("lint", "completed", "failure", T1))
        lines = self.stopped(
            "premise: the CI failure set changed: now failing lint (latest attempts): re-verify: "
            "stop"
        )
        self.assertIn("   BAD lint", lines)
        self.world["runs"].append(
            run("lint", "completed", "success", T2)
        )  # re-run green: as verified again
        self.assertEqual(self.post("--dry-run")[0], 0)
        self.stopped(
            "premise: 1 cancelled jobs, the spec allows 0: stop", MAXCANCEL="0"
        )
        lines = self.stopped(
            "premise: 1 run beyond the verified set (RUN above): update the Bots / CI row and RUNOK: "
            "stop",
            RUNOK=None,
        )
        self.stopped(
            "premise: the CI failure set changed: not failing now gone (latest attempts): re-verify: stop",
            FAILOK=["Windows stage", "gone"],
        )
        self.assertIn("   RUN sanity", lines)
        self.assertEqual(
            self.post("--dry-run", RUNOK=None, RUNANY=True)[0], 0
        )  # CI still running broadly

    def test_threads_reviews_comments_and_the_tag(self):
        self.world["resolved"] = [True, True, True]
        self.stopped(
            "premise: threads changed: now 3 3, the spec has 3 2 (total, resolved): stop"
        )
        self.world["resolved"] = [True, True, False]
        self.world["review_comments"].append("2026-11-05T15:00:01Z")
        lines = self.stopped(
            f"premise: newer than STOP {STOP}: 0 reviews, 1 review comments, 0 human comments: read "
            "them first: stop"
        )
        self.assertEqual(
            lines[-2], "new reviews 0; review comments 1; human comments 0"
        )
        self.world["review_comments"].pop()
        self.world["body"] = "Fixes the bias."
        self.stopped(f"premise: the tag {TAG} left the PR body: stop")

    def test_an_unread_premise_stops(self):
        self.world["fail"] = "check-runs"
        self.stopped(
            f"premise not read (gh api repos/ROCm/r/commits/{self.head}/check-runs?per_page=100: HTTP 502): stop"
        )

    # --- each check on the post stops it -----------------------------------------------------------------------
    def check_text(self, last, old, new, *hits):
        self.src.write_text(POST.replace(old, new), encoding="utf-8")
        lines = self.stopped(last)
        self.assertEqual(lines[-1 - len(hits) : -1], list(hits))

    def test_the_post_checks(self):
        self.copy.parent.mkdir()
        self.copy.write_text("x")
        self.stopped(
            f"{self.copy} exists: round fu2 was posted or kept; a new review takes a new ROUND: stop"
        )
        self.copy.unlink()
        self.stopped(
            "CI count not written: the fill anchor '(sanity running); 99 ✓' is not in the post: stop",
            fill=["(sanity running); 99 ✓", "$ok ✓"],
        )
        self.check_text(
            "a #N reference in the body would cross-link (lines above): stop",
            "N1: x",
            "N1: see #123",
            "7:| Ours (11-04) N1: see #123 | 🟢 FIXED `abc1234`<br>• ran |",
        )
        self.check_text(
            "hygiene hit (lines above): stop",
            "• ran",
            "• the user ran it",
            "7:| Ours (11-04) N1: x | 🟢 FIXED " "`abc1234`<br>• the user ran it |",
        )
        self.check_text(
            "perf or measured figure (lines above): stop",
            "• ran",
            "• 404s",
            "7:| Ours (11-04) N1: x | 🟢 FIXED " "`abc1234`<br>• 404s |",
        )
        self.check_text(
            "verdict word (lines above): stop",
            "fixes hold",
            "ACCEPT",
            f"3:## PR #7 — t (dev, {TAG}) — 🟡 Follow-up: ACCEPT",
        )
        self.check_text(
            f"line 1: the post's 'Automated review: Model - claude, effort: max' is not {LINE1!r}: stop",
            LINE1,
            "Automated review: Model - claude, effort: max",
        )
        # another deployment's name
        other = LINE1.replace("rocke-monitor", "other-team")
        self.check_text(
            f"line 1: the post's {other!r} is not {LINE1!r}: stop", LINE1, other
        )
        self.check_text(
            "header: no '## PR #7 — ' line, or line 3 is not the verified post's: stop",
            "## PR #7 —",
            "## PR #8 —",
        )
        self.check_text(
            "no line starts with the footer '_Follow-up review' (the spec's FOOTER, else the round's): stop",
            "_Follow-up review:",
            "_First-round review:",
        )
        self.check_text(
            f"the post never names the verifier ({premises.verifier_name(self.cfg)}): its footer must: "
            "stop",
            "Claude Opus 5.5",
            "the orchestrator",
        )
        self.check_text("no Code quality row: stop", "| Code quality |", "| Quality |")
        self.check_text(
            "the Code quality row is still pending: fold the lane in first: stop",
            "🟢 None yet",
            "pending (lane running)",
        )
        self.src.write_text(POST, encoding="utf-8")
        self.store.ledger_add("post", 7, self.head, "u")
        self.stopped(
            f"already posted at this head {self.head[:10]}: a re-review posts with --again and a new ROUND: "
            "stop"
        )

    def test_a_review_staged_before_a_change_of_its_pass_keeps_its_line_1(self):
        """The first pass changed after the review was staged (the local worker's model here): line 1 is the one
        the review's staging gives (the review folder's pr/meta.md, at the verified head), and rk post's own guard
        takes it too; with no staging of that head, the configuration's line now."""
        self.cfg["worker"] = {"model": "llama"}
        now = f"Automated review by rocke-monitor: Model - llama, effort: {config.MODEL_DEFAULT}"
        self.stopped(f"line 1: the post's {LINE1!r} is not {now!r}: stop")
        meta = self.d / "workers" / "pr7" / "pr" / "meta.md"
        meta.parent.mkdir(parents=True)
        meta.write_text(
            f"- Head: b7 @ {self.head[:10]}; base: develop @ {self.devbase[:10]}\n- Line 1: {LINE1}\n",
            encoding="utf-8",
        )
        code, lines = self.post("--dry-run")
        self.assertEqual(code, 0, lines)
        self.assertTrue(
            lines[-1].startswith(
                "dry run: the premises hold and rk post's guards pass"
            ),
            lines,
        )

    def test_no_draft_mode_takes_the_verifiers_line_1_and_a_footer_with_no_first_pass(
        self,
    ):
        """`worker_backend: "none"`: line 1 names the verifier, and the footer credits no first pass; rk post's own
        line-1 guard takes the verifier's line too."""
        self.cfg["worker_backend"] = "none"
        self.cfg["verifier_name"] = "Claude Opus 5.5"  # else the audit model in words
        self.stopped(
            f"line 1: the post's {LINE1!r} is not 'Automated review by rocke-monitor: Model - Claude Opus 5.5, "
            "effort: high': stop"
        )  # the drafter's line 1 is no longer the one
        verifier = (
            "Automated review by rocke-monitor: Model - Claude Opus 5.5, effort: high"
        )
        self.src.write_text(POST.replace(LINE1, verifier), encoding="utf-8")
        self.stopped(
            'the footer credits a first pass, but the review was staged with no drafter: write "verified by '
            'the orchestrator (Claude Opus 5.5)" alone: stop'
        )
        self.src.write_text(
            POST.replace(LINE1, verifier).replace(
                "first pass by a local qwen model, ", ""
            ),
            encoding="utf-8",
        )
        code, lines = self.post("--dry-run")
        self.assertEqual(code, 0, lines)
        self.assertTrue(
            lines[-1].startswith(
                "dry run: the premises hold and rk post's guards pass"
            ),
            lines,
        )

    def test_the_footer_check_takes_the_draft_the_review_was_staged_with(self):
        """The first pass a footer may credit is the one the review was staged with (the draft recorded for it at its
        head), not the configuration's now: a review a drafter drafted keeps its credit after the passes change to no
        drafter, and one staged with no drafter may not credit one after a drafter is back.
        """
        self.cfg["verifier_name"] = "Claude Opus 5.5"  # else the audit model in words
        verifier = (
            "Automated review by rocke-monitor: Model - Claude Opus 5.5, effort: high"
        )
        meta = self.d / "workers" / "pr7" / "pr" / "meta.md"
        meta.parent.mkdir(parents=True)

        def staged(status: str, line1: str) -> None:
            """The review staged at the head, line 1 as its staging gave it, its draft recorded with STATUS."""
            meta.write_text(
                f"- Head: b7 @ {self.head[:10]}; base: develop @ {self.devbase[:10]}\n- Line 1: {line1}\n",
                encoding="utf-8",
            )
            st = self.store.load()
            st["drafts"] = [
                {"pr": 7, "head": self.head, "role": "drafter", "status": status}
            ]
            self.store.save(st)

        self.cfg["worker_backend"] = "none"
        staged("ready", LINE1)
        code, lines = self.post("--dry-run")
        self.assertEqual(code, 0, lines)
        self.assertTrue(
            lines[-1].startswith(
                "dry run: the premises hold and rk post's guards pass"
            ),
            lines,
        )
        del self.cfg["worker_backend"]
        staged("none", verifier)
        self.src.write_text(POST.replace(LINE1, verifier), encoding="utf-8")
        self.stopped(
            'the footer credits a first pass, but the review was staged with no drafter: write "verified by '
            'the orchestrator (Claude Opus 5.5)" alone: stop'
        )

    def test_again_posts_a_second_time_at_a_head_we_posted_on(self):
        """A re-review at the head we posted on (`rk review-queue N --again`) posts with --again: every premise and
        check still runs, and every guard of rk post holds but the one post per head, the hourly cap among them.
        """
        self.store.ledger_add(
            "post", 7, self.head, "u"
        )  # our earlier post at this head
        self.stopped(
            f"already posted at this head {self.head[:10]}: a re-review posts with --again and a new ROUND: "
            "stop"
        )
        code, lines = self.post("--again", "--dry-run")
        self.assertEqual(
            (code, lines),
            (
                0,
                self.premise_lines()
                + [
                    f"dry run: the premises hold and rk post's guards pass; stopped before the post ({self.copy} not "
                    f"written; {self.write_login()})"
                ],
            ),
        )
        self.world["resolved"] = [True, True, True]  # a premise still stops it
        self.stopped(
            "premise: threads changed: now 3 3, the spec has 3 2 (total, resolved): stop",
            "--again",
        )
        self.world["resolved"] = [True, True, False]
        # rk post --again does not compare the live head with the state's, so the premises do.
        self.st["prs"]["7"]["head"] = "f" * 40
        self.store.save(self.st)
        self.stopped(
            "the state's head ffffffffff is not the live head: run rk tick, then post: stop",
            "--again",
        )
        self.st["prs"]["7"]["head"] = self.head
        self.store.save(self.st)
        for _ in range(5):  # six posts in the last hour with ours: the cap
            self.store.ledger_add("post", 9, "9" * 40, "u")
        lines = self.stopped("post refused: read the message", "--again")
        self.assertEqual(lines[-2], "REFUSED: rate limit: 6 posts in the last hour")
        self.store.ledger_path.write_text("")
        self.store.ledger_add("post", 7, self.head, "u")
        code, lines = self.post("--again")
        self.assertEqual(code, 0, lines)
        self.assertEqual(
            lines[len(self.premise_lines())],
            f"posted: https://github.com/ROCm/r/pull/7#issuecomment-1 (copy: {self.d}/posted/PR_FOLLOWUP_7.md)",
        )
        self.assertEqual(len(self.comments_posted), 1)
        self.assertEqual(
            [r["head"] for r in self.store.ledger() if r["kind"] == "post"],
            [self.head, self.head],
        )

    # --- rk spec: the spec written from the review directory and live reads -----------------------------------
    NOTES = (
        "# Verifier notes, 7 follow-up (fu2)\n"
        "settled: N1 FIXED (ran)\n"
        "premises: head {head10}; develop {devbase10}; CI <4 ✓ · 1 running · 1 ✗ · 1 cancelled>; threads 1/3; "
        "stop 2026-11-05T15:00:00Z; probes none\n"
        "FAILOK (as printed by rk ci-fails): {failok}\n"
        "CI pending: sanity (queued)\n"
        "conflict set (merge-tree --name-only with {devbase10}): rk/a.py\n"
        "decide: like/py.func/f+g 7 7,9\n"
        "decide: none (the lane wrote no decide: line)\n"
    )

    def spec_cmd(self, *extra, failok="Windows stage"):
        (self.d / "tmp/v7fu2/notes.md").write_text(
            self.NOTES.format(
                head10=self.head[:10], devbase10=self.devbase[:10], failok=failok
            ),
            encoding="utf-8",
        )
        return self.rk(
            "spec", "7", "--round", "fu2", "--dir", str(self.d / "tmp/v7fu2"), *extra
        )

    def test_rk_spec_writes_a_spec_the_premises_pass(self):
        code, lines = self.spec_cmd(
            "--save", "fu2: N1 FIXED (ran)", "--cost-verifier", "1000"
        )
        self.assertEqual(code, 0, lines)
        path = self.d / "tmp/spec7fu2.json"
        self.assertEqual(
            lines[-1],
            f"wrote {path}: rk post 7 {self.src} --premises {path} [--dry-run]",
        )
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            {
                k: spec[k]
                for k in (
                    "N",
                    "ROUND",
                    "HEAD10",
                    "DEVBASE",
                    "THREADS",
                    "STOP",
                    "CONFLICTS",
                    "TAG",
                    "MAXCANCEL",
                    "FAILOK",
                    "RUNOK",
                    "RUNANY",
                    "fill",
                    "SAVE",
                    "AFTER",
                    "COST",
                )
            },
            {
                "N": "7",
                "ROUND": "fu2",
                "HEAD10": self.head[:10],
                "DEVBASE": self.devbase[:10],
                "THREADS": "3 2",
                "STOP": "2026-11-05T15:00:00Z",
                "CONFLICTS": "rk/a.py",
                "TAG": TAG,
                "MAXCANCEL": "1",
                "FAILOK": ["Windows stage"],
                "RUNOK": ["sanity"],
                "RUNANY": False,
                "fill": ["99 ✓", "$ok ✓"],
                "SAVE": "fu2: N1 FIXED (ran)",
                "AFTER": [
                    [
                        "$RK",
                        "overlap-decide",
                        "like/py.func/f+g",
                        "7",
                        "--members",
                        "7,9",
                        "--note",
                        "the verifier's decide line, #7 fu2",
                    ]
                ],
                "COST": {"verifier": 1000, "quality": None},
            },
        )
        self.assertEqual(spec["LIGHT"], "🟡 Follow-up: fixes hold")
        self.assertEqual(
            self.rk("post", "7", str(self.src), "--premises", str(path), "--dry-run")[
                0
            ],
            0,
        )
        # A name the notes cut ("...") still names its job.
        self.assertEqual(self.spec_cmd(failok="Windows stag...")[0], 0)
        self.assertEqual(json.loads(path.read_text())["FAILOK"], ["Windows stage"])

    def test_a_decide_key_a_shell_would_read_stops_the_post(self):
        """The verifier's notes come from a model that read PR text: a decide: key that is not an overlap key never
        reaches a command; it is a CHECK line (the post stops)."""
        for key in ("$(id>/tmp/x)", "x;touch${IFS}/tmp/p;#", "a`id`", "k|sh"):
            with self.subTest(key=key):
                self.NOTES = TestPostPremises.NOTES.replace(  # the notes are a format string: braces doubled
                    "decide: like/py.func/f+g 7 7,9",
                    "decide: " + key.replace("{", "{{").replace("}", "}}") + " 7 7,9",
                )
                code, lines = self.spec_cmd()
                self.assertEqual(code, 3, lines)
                self.assertIn(
                    f"CHECK: the notes' decide: key {key!r} is not an overlap key",
                    "\n".join(lines),
                )
                self.assertEqual(
                    json.loads((self.d / "tmp/spec7fu2.json").read_text())["AFTER"], []
                )

    def test_spec_post_keeps_an_unposted_review_in_prototype(self):
        """`rk spec-post` (the orchestrator's post path) reaches the post's mode guard: in prototype the session's
        checked post is kept unposted and as an unsent item, and nothing goes outward; the verifier's decide line, an rk
        command that records the owner on this machine, runs as the monitor's own; typed, the post is refused, naming
        the mode."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        (self.d / "tmp/v7fu2/notes.md").write_text(
            self.NOTES.format(
                head10=self.head[:10],
                devbase10=self.devbase[:10],
                failok="Windows stage",
            ),
            encoding="utf-8",
        )
        argv = ("spec-post", "7", "--round", "fu2", "--dir", str(self.d / "tmp/v7fu2"))
        code, lines = self.rk(*argv)
        self.assertEqual(
            (code, lines[-1]), (4, f"REFUSED: a review post is not sent: {HELD}"), lines
        )
        self.assertEqual(post.copies(self.cfg, 7), [])
        rk, ran = self.fake_rk()
        with self.own(), rk:
            code, lines = self.rk(*argv)
        self.assertEqual(code, 0, lines)
        self.assertEqual(
            ran.read_text(),
            "1|overlap-decide like/py.func/f+g 7 --members 7,9 --note the verifier's decide line, #7 fu2",
        )
        self.assertIn(
            f"UNPOSTED: {HELD}, so the verified review stays unposted, kept as an unsent item: "
            f"{self.d}/posted/PR_UNPOSTED_7.md.",
            lines,
        )
        self.assertEqual(self.comments_posted, [])
        # Live, the kept ROUND stays used (its copy); a new review takes a new ROUND and posts.
        self.cfg["features"] = {"review_automation": {"mode": "live"}}
        self.assertEqual(
            self.post()[1][-1],
            f"{self.copy} exists: round fu2 was posted or kept; a new review takes a new ROUND: stop",
        )
        self.assertEqual(self.post(ROUND="fu3")[0], 0)

    def test_a_review_kept_unposted_gets_the_after_post_work_of_this_machine(self):
        """Review automation in prototype: the session's checked review is kept unposted (the count filled in) and as
        an unsent item, and gets the after-post work that stays on this machine: the verify note, the day log, the
        notes archive, the drafters' assessment and the review cost, its post the kept file. Nothing goes outward: no
        comment, no ledger entry, and its AFTER lines, shell lines here, do not run."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        with self.own():
            code, lines = self.post()
        kept = self.d / "posted/PR_UNPOSTED_7.md"
        self.assertEqual(
            (code, lines),
            (
                0,
                self.premise_lines()
                + [
                    f"UNPOSTED: {HELD}, so the verified review stays unposted, kept as an unsent item: "
                    f"{kept}.",
                    "logged save (verify) for 7",
                    *(
                        f"AFTER not run in prototype (not an rk command, so it could send past the one outward gate): "
                        f"{line}"
                        for line in self.spec["AFTER"]
                    ),
                    "notes archived",
                    "cost recorded: draft qwen uncounted, verifier not given, quality not given",
                ],
            ),
        )
        self.assertIn("• (sanity queued); 4 ✓ |", kept.read_text(encoding="utf-8"))
        self.assertEqual((self.comments_posted, self.store.ledger()), ([], []))
        # The round's copy is the kept post: the keep used its ROUND, as a post does.
        self.assertEqual(self.copy.read_bytes(), kept.read_bytes())
        self.assertEqual(
            [
                (i["type"], i["for"], i["file"], i["why"])
                for i in state_store.unsent_items(self.store.dir)
            ],
            [(state_store.REVIEW, "pull request 7", str(kept), state_store.PROTOTYPE)],
        )
        self.assertFalse((self.d / "after.txt").exists())
        self.assertEqual(
            [
                (r["kind"], r.get("save"), r["pr"])
                for r in self.store.read_jsonl("daylog.jsonl")
            ],
            [("unposted", None, 7), ("save", "verify", 7)],
        )
        self.assertEqual(
            (self.d / "notes/2026-11-05_pr7_fu2.md").read_text(), "premises: head x\n"
        )
        cost = self.store.read_jsonl("review_cost.jsonl")
        self.assertEqual(
            [(c["pr"], c["round"], c["post"], c.get("unposted")) for c in cost],
            [(7, "fu2", str(kept), True)],
        )

    def test_a_second_keep_at_the_head_of_our_last_review_stops(self):
        """Review automation in prototype: the session's keep at the head our last review (rec["ours"]) is at stops, as
        a second post at a head we posted on does, whether that review was kept unposted (the spec run again) or posted (by another monitor under
        our login: our ledger has no entry). Nothing is kept, logged or counted twice; a re-review keeps with --again
        and a new ROUND. The keep's round copy stops its ROUND being kept again, with --again too.
        """
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        self.st["review_ready"] = [
            {
                "pr": 7,
                "kind": "follow-up",
                "head": self.head,
                "at": "2026-11-05T15:00:00Z",
            }
        ]  # handed to verification at the live head
        self.store.save(self.st)
        mark = self.own()
        mark.start()  # the session's keeps
        self.addCleanup(mark.stop)
        self.assertEqual(self.post()[0], 0)
        ours = self.store.load()["prs"]["7"]["ours"]
        self.assertEqual((ours["head"], ours["file"]), (self.head, "PR_UNPOSTED_7.md"))
        archived = self.d / "notes/2026-11-05_pr7_fu2.md"
        archived.unlink()  # the keep archived its notes: a stop archives none
        archived.parent.rmdir()
        for extra in ((), ("--again",)):  # the same ROUND: its copy stops it first
            self.stopped(
                f"{self.copy} exists: round fu2 was posted or kept; a new review takes a new ROUND: stop",
                *extra,
            )
        self.stopped(
            f"our last review is at this head {self.head[:10]} already (kept unposted: PR_UNPOSTED_7.md): a re-review "
            "keeps with --again and a new ROUND: stop",
            ROUND="fu3",
        )
        st = self.store.load()
        st["prs"]["7"]["ours"] = dict(
            ours, kind="comment", url="https://example.com/review/1"
        )
        self.store.save(st)
        self.stopped(
            f"our last review is at this head {self.head[:10]} already (posted): a re-review keeps with --again and "
            "a new ROUND: stop",
            ROUND="fu3",
        )
        self.assertEqual(
            [p.name for p in post.copies(self.cfg, 7)], ["PR_UNPOSTED_7.md"]
        )
        code, lines = self.post("--again", ROUND="fu3")
        self.assertEqual(code, 0, lines)
        self.assertEqual(
            [p.name for p in post.copies(self.cfg, 7)],
            ["PR_UNPOSTED_7.md", "PR_UNPOSTED_7_2.md"],
        )
        self.assertEqual(len(self.store.read_jsonl("review_cost.jsonl")), 2)
        (self.d / "notes/2026-11-05_pr7_fu3.md").unlink()
        (self.d / "notes").rmdir()
        self.stopped(
            f"{self.copy.with_name('pr7_fu3.md')} exists: round fu3 was posted or kept; a new review takes a new "
            "ROUND: stop",
            "--again",
            ROUND="fu3",
        )

    def test_a_keep_by_hand_is_our_last_review_at_the_prs_head_and_a_paid_review(self):
        """Review automation in prototype, no review awaiting verification: the session's plain `rk post` keeps the
        review as our last review at the PR's head, as a post is recorded, so a second keep there is refused; and the
        keep writes its line of state/review_cost.jsonl (no spec: no tokens of the verifier or the lane), so rk spend
        counts it as a review."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        kept = self.d / "posted/PR_UNPOSTED_7.md"
        with self.own():
            code, lines = self.rk("post", "7", str(self.src))
            self.assertEqual(code, 0, lines)
            ours = self.store.load()["prs"]["7"]["ours"]
            self.assertEqual(
                (ours["kind"], ours["head"], ours["file"], ours["count"]),
                (post.UNPOSTED, self.head, kept.name, 3),
            )
            code, lines = self.rk("post", "7", str(self.src))
        self.assertEqual(
            (code, lines),
            (
                4,
                [
                    f"REFUSED: our last review of #7 is at head {self.head[:10]} already (kept unposted: "
                    f"{kept.name}) (use --again if intended)"
                ],
            ),
        )
        self.assertEqual(
            [
                (c["pr"], c["round"], c["head"], c["post"], c.get("unposted"))
                for c in self.store.read_jsonl("review_cost.jsonl")
            ],
            [(7, "fu2", self.head, str(kept), True)],
        )
        self.assertEqual(
            [
                (r["pr"], r["round"], r["unposted"])
                for r in spend.paid_reviews(self.store)
            ],
            [(7, "fu2", True)],
        )

    def test_a_kept_review_carries_the_gpu_tests_row(self):
        """Remote GPU tests ran for the head: the review kept unposted carries their GPU tests row, as its post would,
        whether the keep is a checked post's or one by hand."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        self.st["gpu"] = {
            "jobs": [
                {
                    "id": "j1",
                    "pr": 7,
                    "head": self.head,
                    "arch": "gfx942",
                    "state": "ended",
                    "result": {"status": "pass", "detail": "2 passed"},
                }
            ]
        }
        self.store.save(self.st)
        row = f"| GPU tests | 🟢 Passed<br>• gfx942: 2 passed at `{self.head[:8]}` (ran) |"
        with self.own():
            self.assertEqual(self.post()[0], 0)
            code, lines = self.rk("post", "7", str(self.src), "--again")
        self.assertEqual(code, 0, lines)
        self.assertEqual(
            [row in p.read_text(encoding="utf-8") for p in post.copies(self.cfg, 7)],
            [True, True],
        )

    def test_a_review_the_writer_kept_has_one_cost_line(self):
        """In the container the writer's `rk post` keeps a review the gate holds since the checks, and its keep writes
        the cost line (post.keep_as_draft); the monitor's `finish` writes no second one for the same kept file and
        head, so rk spend counts the review once."""
        kept = self.d / "posted/PR_UNPOSTED_7.md"
        review_cost.record_keep(
            self.cfg, self.store.load(), self.store, 7, "fu2", self.head, str(kept)
        )
        said = review_cost.record(
            self.cfg,
            self.store,
            7,
            self.spec,
            {"sha": self.head, "copy": kept},
            kept=True,
        )
        self.assertEqual(said, [f"cost recorded when the review was kept: {kept}"])
        self.assertEqual(len(self.store.read_jsonl("review_cost.jsonl")), 1)

    def test_a_kept_review_is_cleared_today_on_the_admin_dashboard(self):
        """The admin dashboard's "Cleared today" counts and lists a review kept unposted (the day log's kind unposted)
        as it does a posted one, and its cleared-vs-arrived chart counts it too."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        with self.own():
            self.assertEqual(self.post()[0], 0)
        st = self.store.load()
        board = report.board(self.cfg, st, self.store, [], [])
        page = webdash.render(self.cfg, st, self.store, [], [], board, webdash.LOCAL)
        self.assertIn("0 posted · 1 kept unposted · 0 rebase-only · 0 merge-only", page)
        cleared = page.split("<details id='d-cleared'>", 1)[1].split("</details>", 1)[0]
        self.assertIn("Cleared today (1)", cleared)
        self.assertIn("follow-up 🟡 Follow-up: fixes hold", cleared)
        self.assertEqual(webdash.daily_series(self.store)[-1][1], 1)

    def test_a_pr_in_review_skip_is_neither_posted_nor_kept(self):
        """A PR in review_skip (rk review-skip) gets no review from rk post, plain or with its premises, posted or kept,
        as its review does not start: the post is refused before any check, naming the fix, and nothing is posted,
        kept or recorded."""
        st = self.store.load()
        st["review_skip"] = [7]
        self.store.save(st)
        refused = "REFUSED: #7 is in review_skip: run rk review-unskip 7 first"
        self.stopped(refused)
        self.assertEqual(self.rk("post", "7", str(self.src)), (4, [refused]))
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        with self.own():
            self.stopped(refused)
            self.assertEqual(self.rk("post", "7", str(self.src)), (4, [refused]))
        self.assertEqual(post.copies(self.cfg, 7), [])

    def test_spec_post_is_an_outward_write(self):
        """`rk spec-post` holds the write lock and is refused while the monitor is stopped, as `rk post` is."""
        from rkmon_lib import control, mirror

        self.assertIn("spec-post", control.WRITES)
        self.assertIsNone(
            mirror.guard(dict(self.cfg, mirror={"enabled": True}), "spec-post")
        )
        st = self.store.load()
        control.mark_stopped(st, "stop")
        self.store.save(st)
        code, lines = self.rk(
            "spec-post", "7", "--round", "fu2", "--dir", str(self.d / "tmp/v7fu2")
        )
        self.assertEqual(code, 4, lines)
        self.assertTrue(lines[-1].startswith("REFUSED: the monitor is stopped"), lines)
        self.assertEqual(self.comments_posted, [])

    def fake_rk(self):
        """A fake rk for the AFTER lines' $RK (premises.rk_path), which writes "1|ARGS" when it runs as the monitor's
        own (outward.OWN_VAR), "|ARGS" else. The patch that puts it in place, and its file.
        """
        fake, ran = self.d / "fake_rk", self.d / "rk_ran.txt"
        fake.write_text(
            f'#!/bin/sh\nprintf "%s|%s" "${outward.OWN_VAR}" "$*" > {ran}\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return mock.patch.object(premises, "rk_path", lambda: str(fake)), ran

    def kept_with_after(self, active=None):
        """A review kept unposted (`rk post --premises` as the session's), its AFTER lines an rk command (fake_rk),
        another command and a shell line, each of the two touching a file; ACTIVE in place of features.active. The
        lines, the AFTER lines as the post prints them, and the fake rk's file."""
        other, shell = self.d / "other.txt", self.d / "shell.txt"
        after = [
            ["$RK", "overlap-decide", "k", "7"],
            ["touch", str(other)],
            f"touch {shell}",
        ]
        rk, ran = self.fake_rk()
        with self.own(), rk, mock.patch.object(
            features, "active", active or features.active
        ):
            code, lines = self.post(AFTER=after)
        self.assertEqual(code, 0, lines)
        self.assertFalse(other.exists() or shell.exists())
        self.assertEqual(self.comments_posted, [])
        return lines, [" ".join(x) if isinstance(x, list) else x for x in after], ran

    def rk_commands_run_as_the_monitors_own(self, held):
        """A review kept while review automation sends nothing (HELD: prototype or standby): its AFTER lists that run
        $RK run, as the monitor's own (outward.OWN_VAR), so what they would send is kept as an unsent item at the one
        outward gate; no other line runs, as it could send past the gate."""
        lines, after, ran = self.kept_with_after()
        self.assertEqual(ran.read_text(), "1|overlap-decide k 7")
        self.assertEqual(
            [x for x in lines if x.startswith("AFTER not run")],
            [
                f"AFTER not run in {held} (not an rk command, so it could send past the one outward gate): {x}"
                for x in after[1:]
            ],
        )

    def test_a_review_kept_in_prototype_runs_its_rk_commands_as_the_monitors_own(self):
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        self.rk_commands_run_as_the_monitors_own("prototype")

    def test_a_review_kept_in_standby_runs_its_rk_commands_as_the_monitors_own(self):
        with mock.patch.object(features, "standby", lambda home=None: True):
            self.rk_commands_run_as_the_monitors_own("standby")

    def test_a_review_kept_while_posting_is_not_set_up_runs_no_after_line(self):
        """Review automation live but its posting not set up: the review is kept, and none of its AFTER lines runs."""
        real = features.active
        lines, after, ran = self.kept_with_after(
            lambda cfg, name: name != "review-posting" and real(cfg, name)
        )
        self.assertFalse(ran.exists())
        self.assertEqual(
            [x for x in lines if x.startswith("AFTER not run")],
            [f"AFTER not run while review posting is off: {x}" for x in after],
        )

    def test_an_after_line_never_runs_a_command_the_session_may_not_run(self):
        """A post's AFTER lines never run an rk command the Claude session may not run (outward.not_the_sessions), as a
        list or in a shell line, by $RK, ${RK} or rk: each says so, the other lines run, and the post stands.
        """
        not_run = (
            ["$RK", "teams-test"],
            '"$RK" teams-test',
            "${RK} teams-test",
            "rk teams-test",
        )
        rk, ran = self.fake_rk()
        with rk:
            code, lines = self.post(
                AFTER=[["$RK", "overlap-decide", "k", "7"], *not_run]
            )
        self.assertEqual(code, 0, lines)
        self.assertEqual(ran.read_text(), "|overlap-decide k 7")
        self.assertEqual(len(self.comments_posted), 1)
        self.assertEqual(
            [x for x in lines if x.startswith("AFTER")],
            [
                "AFTER not run: rk teams-test is for me to run, or for rk itself, never for the session: "
                + (" ".join(x) if isinstance(x, list) else x)
                for x in not_run
            ],
        )

    def test_a_kept_review_never_runs_a_command_the_session_may_not_run(self):
        """Kept in prototype: the review's $RK lines run as the monitor's own, but for a command the session may not
        run, whose line says so; the kept review stands."""
        self.cfg["features"] = {"review_automation": {"mode": "prototype"}}
        rk, ran = self.fake_rk()
        with self.own(), rk:
            code, lines = self.post(
                AFTER=[["$RK", "overlap-decide", "k", "7"], ["$RK", "teams-test"]]
            )
        self.assertEqual(code, 0, lines)
        self.assertEqual(ran.read_text(), "1|overlap-decide k 7")
        self.assertEqual(
            [x for x in lines if x.startswith("AFTER")],
            [
                "AFTER not run: rk teams-test is for me to run, or for rk itself, never for the session: "
                "$RK teams-test"
            ],
        )
        self.assertEqual(self.comments_posted, [])

    def held_since_the_checks(self, writer_said=None):
        """The session's post (`rk post --premises`), review automation live, standby beginning after the checks,
        right before the post: on this host (WRITER_SAID None), or in the container, where the writer's `rk post`
        printed WRITER_SAID, having kept the review at the kept file. The lines, the kept file, and its line.
        """
        standby = mock.patch.object(features, "standby", lambda home=None: True)
        kept = self.d / "posted/PR_UNPOSTED_7.md"
        fake_writer = (
            [
                sys.executable,
                "-c",
                "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('kept'); print(sys.argv[2])",
                str(kept),
                writer_said,
            ]
            if writer_said
            else None
        )

        def routed(argv):
            standby.start()
            return fake_writer

        try:
            with self.own(), mock.patch.object(
                premises, "writer", types.SimpleNamespace(routed=routed)
            ):
                code, lines = self.post()
            said = writer_said or post.kept_line(self.cfg, str(kept))
            held = features.held_words(self.cfg, post.REVIEWS_SEND_FOR)
        finally:
            standby.stop()
        self.assertIn("the monitor is in standby", held)
        self.assertEqual(
            (code, lines),
            (
                0,
                self.premise_lines()
                + [
                    said,
                    "logged save (verify) for 7",
                    *(
                        "AFTER not run in standby (not an rk command, so it could send past the one outward gate): "
                        f"{line}"
                        for line in self.spec["AFTER"]
                    ),
                    "notes archived",
                    "cost recorded: draft qwen uncounted, verifier not given, quality not given",
                ],
            ),
        )
        self.assertEqual((self.comments_posted, self.store.ledger()), ([], []))
        # The round's copy stays, the filled post: the keep used its ROUND, as a post does.
        self.assertIn("• (sanity queued); 4 ✓ |", self.copy.read_text(encoding="utf-8"))
        self.assertFalse((self.d / "after.txt").exists())
        cost = self.store.read_jsonl("review_cost.jsonl")
        self.assertEqual(
            [(c["post"], c.get("unposted")) for c in cost], [(str(kept), True)]
        )
        return kept

    def test_a_post_the_gate_holds_since_the_checks_is_kept_on_a_host(self):
        """Standby begins after the post's checks: on a host `rk post` keeps the review, as the writer's does, and it
        counts as a review kept: no comment, no copy, the kept file its post, none of its shell AFTER lines run, and
        it is an unsent item."""
        kept = self.held_since_the_checks()
        self.assertEqual(
            [
                (i["type"], i["file"], i["why"])
                for i in state_store.unsent_items(self.store.dir)
            ],
            [(state_store.REVIEW, str(kept), state_store.STANDBY)],
        )

    def test_a_post_the_writer_kept_since_the_checks_counts_as_kept(self):
        """In the container, the writer's `rk post` keeps a review the gate holds since the checks, and says so: the
        monitor's `finish` counts it as kept."""
        self.held_since_the_checks(
            "UNPOSTED: the monitor is in standby, so the verified review stays unposted, kept as an unsent item: "
            "PR_UNPOSTED_7.md."
        )

    def test_an_after_list_runs_without_a_shell(self):
        """An AFTER list is one command's arguments: $RK and the other whole `$NAME`s are taken from the post's
        variables, and nothing in an argument is expanded."""
        out = self.d / "argv.txt"
        code, lines = self.post(
            AFTER=[
                [
                    "python3",
                    "-c",
                    "import sys; open(sys.argv[1], 'w').write(repr(sys.argv[2:]))",
                    str(out),
                    "$sha",
                    "$(touch pwned)",
                    "x;id",
                ]
            ]
        )
        self.assertEqual(code, 0, lines)
        self.assertEqual(out.read_text(), repr([self.head, "$(touch pwned)", "x;id"]))
        self.assertFalse(pathlib.Path("pwned").exists())

    def test_a_stop_later_than_now_refuses_the_post(self):
        self.stopped(
            "REFUSED: STOP 2026-11-05T17:00:00Z is later than now (2026-11-05T16:00:00Z): the verifier's stop "
            "time is when it stopped reading",
            STOP="2026-11-05T17:00:00Z",
        )

    def test_a_minute_stop_counts_a_comment_within_its_minute(self):
        """A STOP to the minute is its first second: a human comment 30 s into that minute stops the post."""
        self.world["comments"] = ["2026-11-05T15:00:30Z"]
        self.stopped(
            "premise: newer than STOP 2026-11-05T15:00:00Z: 0 reviews, 0 review comments, 1 human comments: "
            "read them first: stop",
            STOP="2026-11-05T15:00Z",
        )

    def test_rk_spec_says_where_the_notes_and_now_differ(self):
        self.world["runs"].append(run("lint", "completed", "failure", T1))
        self.world["resolved"] = [True, True, True]
        code, lines = self.spec_cmd()
        self.assertEqual(code, 3, lines)
        self.assertIn(
            "CHECK: lint fails now and the notes do not account for it: re-verify",
            lines,
        )
        self.assertIn(
            "CHECK: threads now 3 total, 3 resolved; the notes say 1/3", lines
        )
        spec = json.loads((self.d / "tmp/spec7fu2.json").read_text())
        self.assertEqual(
            (spec["FAILOK"], spec["THREADS"]), (["Windows stage"], "3 2")
        )  # the verified side
        code, lines = self.rk(
            "post",
            "7",
            str(self.src),
            "--premises",
            str(self.d / "tmp/spec7fu2.json"),
            "--dry-run",
        )
        self.assertEqual(
            (code, lines[-1]),
            (
                4,
                "premise: the CI failure set changed: now failing lint (latest "
                "attempts): re-verify: stop",
            ),
        )

    def test_rk_spec_adds_the_count_to_a_bots_row_without_one(self):
        self.src.write_text(
            POST.replace("• (sanity queued); 99 ✓ |", "• sanity queued |"),
            encoding="utf-8",
        )
        code, lines = self.spec_cmd()
        self.assertEqual(code, 0, lines)
        self.assertIn(
            "| Bots / CI | 🟡 running<br>• sanity queued; 4 ✓ |",
            self.src.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            json.loads((self.d / "tmp/spec7fu2.json").read_text())["fill"],
            ["4 ✓", "$ok ✓"],
        )
        self.assertEqual(
            self.rk(
                "post",
                "7",
                str(self.src),
                "--premises",
                str(self.d / "tmp/spec7fu2.json"),
                "--dry-run",
            )[0],
            0,
        )
        # An anchor that is not unique is extended to the left until it is.
        self.src.write_text(POST.replace("N1: x", "N1: 99 ✓ cases"), encoding="utf-8")
        self.spec_cmd()
        self.assertEqual(
            json.loads((self.d / "tmp/spec7fu2.json").read_text())["fill"],
            ["queued); 99 ✓", "queued); $ok ✓"],
        )
        # No notes, no spec.
        (self.d / "tmp/v7fu2/notes.md").unlink()
        self.assertEqual(
            self.rk("spec", "7", "--round", "fu2", "--dir", str(self.d / "tmp/v7fu2")),
            (4, [f"REFUSED: no {self.d / 'tmp/v7fu2/notes.md'}"]),
        )

    # --- the spec ----------------------------------------------------------------------------------------------
    def test_the_spec_is_checked_first(self):
        spec = str(self.d / "spec.json")
        for changes, why in (
            ({"STOP": None}, f"REFUSED: spec {spec}: missing STOP"),
            ({"FAIL_OK": ["x"]}, f"REFUSED: spec {spec}: unknown FAIL_OK"),
            ({"N": "8"}, "REFUSED: the spec is for #8, not #7"),
            (
                {"ROUND": "../x"},
                "REFUSED: ROUND '../x': fr, or fr or fu with a number (fr2, fu1)",
            ),
            (
                {"SRC": str(self.d / "other.md")},
                f"REFUSED: {self.src} is not the spec's SRC, "
                f"{self.d / 'other.md'}: wrong spec?",
            ),
            ({"MAXCANCEL": "one"}, "REFUSED: MAXCANCEL 'one': a count"),
        ):
            self.assertEqual(self.post(**changes), (4, [why]))

    def test_mkpost_fu_specs_are_taken_as_they_are(self):
        """The keys of a first round's spec, FOOTER, LIGHT and OUT included, with the ticket tag under the JIRA
        project's key, as the skill's docs name it; in the package that key is jira.project's, never written here.
        """
        keys = [
            "N",
            "ROUND",
            "HEAD10",
            "LIGHT",
            "DEVBASE",
            "THREADS",
            "STOP",
            "CONFLICTS",
            "SRC",
            "NOTES",
            "MAXCANCEL",
            "FAILOK",
            "RUNOK",
            "fill",
            "FOOTER",
            "SAVE",
            "OUT",
        ]
        path = self.d / "spec7fr.json"
        path.write_text(
            json.dumps({**{k: self.spec[k] for k in keys}, TK: TAG}), encoding="utf-8"
        )
        cfg = dict(self.cfg, jira={"project": TK})
        spec = premises.load(str(path), 7, str(self.src), cfg)
        self.assertEqual(
            (spec["FOOTER"], premises.tag(cfg, spec)), ("_Follow-up review", TAG)
        )
        with self.assertRaisesRegex(
            ValueError, f"unknown {TK}"
        ):  # with no JIRA project, that key reads as a typo
            premises.load(str(path), 7, str(self.src), self.cfg)
        path.write_text(json.dumps({k: self.spec[k] for k in keys}), encoding="utf-8")
        # A project and no tag loads: the premise reads the body (a PR whose body names no key carries none).
        self.assertIsNone(
            premises.tag(cfg, premises.load(str(path), 7, str(self.src), cfg))
        )

    def test_the_tag_is_optional_while_the_body_names_no_key(self):
        """With a JIRA project, a spec with no tag (absent or "", as `rk spec` writes it for a body with no key)
        posts on a PR whose body names no key; a body that names one stops it (the tag is optional only then).
        """
        self.cfg["jira"] = {"project": TK}
        self.world["body"] = "Fixes the bias, from an outside contributor."
        with_tag = self.src.read_text(encoding="utf-8")
        self.src.write_text(with_tag.replace(f", {TAG})", ")"), encoding="utf-8")
        for changes in ({"TAG": None}, {"TAG": ""}):
            code, lines = self.post("--dry-run", **changes)
            self.assertEqual(code, 0, (changes, lines))
        self.world["body"] = f"Fixes the bias.\n\nISSUE ID : {TAG}\n"
        self.stopped(
            f"premise: the PR body names {TAG}, the spec names no TAG: add TAG {TAG}: stop",
            TAG="",
        )
        self.src.write_text(with_tag, encoding="utf-8")
        self.assertEqual(self.post("--dry-run")[0], 0)  # the spec names it: as before

    def test_failok_and_runok_that_name_no_job_are_empty(self):
        path = self.d / "spec_none.json"
        path.write_text(
            json.dumps(dict(self.spec, FAILOK=["(none)"], RUNOK=["None", "-"])),
            encoding="utf-8",
        )
        spec = premises.load(str(path), 7, str(self.src))
        self.assertEqual((spec["FAILOK"], spec["RUNOK"]), ([], []))


class TestHygienePattern(unittest.TestCase):
    def test_the_traps_match_whole_words_only(self):
        from rkmon_lib.premises import HYGIENE

        # Split literals: the package's privacy check scans this source for the same traps.
        for text in (
            "runs on Na" "vi31",
            "a na" "vi GPU",
            "na" "vi4x",
            "Ins" "tinct M" "I300X",
            "an M" "I355",
            "Str" "ix Halo",
            "Rad" "eon cards",
            "a stale head",
            "stale-head",
            "the user ran it",
            "the user's call",
            "rk abi says",
            "see http" "s://x",
            "under /ho" "me/x",
            "ask @someone",
            "a Confluence page",
            "3 G" "B/s",
        ):
            self.assertTrue(HYGIENE.search(text), text)
        for text in (
            "the navigation bar",
            "instinctively",
            "a stalemate",
            "staleness check",
            "the users of the API",
            "work abi",
            "a confluent stream",
        ):
            self.assertIsNone(HYGIENE.search(text), text)


class TestFigurePattern(unittest.TestCase):
    def test_a_unit_after_a_word_is_no_figure(self):
        """A unit after a word ("both lists match their .dvc md5s") is no measured figure."""
        from rkmon_lib.premises import FIGURE

        for text in (
            "404s",
            "took 12 " + "ms",
            "3.5 " + "GB/s",
            "1.5" + "ms",
            "(30 s)",
        ):  # split: the privacy check
            self.assertTrue(FIGURE.search(text), text)
        for text in ("both lists match their md5s", "fp16s", "sha256s", "gfx942s"):
            self.assertIsNone(FIGURE.search(text), text)


class TestStopTime(unittest.TestCase):
    """STOP is read to the second (a minute is its first second, so a comment within it counts as newer), and a STOP
    later than now is refused: it would hide every new comment."""

    def test_normalised_and_checked(self):
        at = "2026-11-05T16:00:00Z"
        self.assertEqual(
            premises.stop_time("2026-11-05T10:00Z", at), "2026-11-05T10:00:00Z"
        )
        self.assertEqual(
            premises.stop_time("2026-11-05T10:00:30Z", at), "2026-11-05T10:00:30Z"
        )
        self.assertEqual(
            premises.stop_time("2026-11-05T10:00:30.5Z", at), "2026-11-05T10:00:30Z"
        )
        self.assertEqual(  # clock skew between the session and here
            premises.stop_time("2026-11-05T16:01:00Z", at), "2026-11-05T16:01:00Z"
        )
        self.assertGreater(
            "2026-11-05T10:00:30Z", premises.stop_time("2026-11-05T10:00Z", at)
        )
        for bad in (
            "2026-11-05T17:00:00Z",
            "yesterday",
            "2026-11-05 10:00Z",
            "2026-13-05T10:00:00Z",
        ):
            with self.subTest(stop=bad), self.assertRaises(ValueError):
                premises.stop_time(bad, at)


if __name__ == "__main__":
    unittest.main()
