# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Outward writes that must never go wrong quietly: a disarm that failed, a write while its feature is in prototype, a
write that timed out after GitHub took it, a body file left behind for gh, and the Teams webhook (a credential) in an
error.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

from rkmon_lib import config, gh, merge, post  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

LINE1 = (
    f"Automated review by rocke-monitor: Model - qwen, effort: {config.MODEL_DEFAULT}"
)
CFG = {
    "incident_issue": guard_labels.INCIDENT_ISSUE,
    "repo": "o/r",
    "operator": {"github_login": "me"},
    "max_posts_per_hour": 6,
    "review_first_line": LINE1,
    "worker": {"model": "qwen"},  # the reviewer model LINE1 gives (bindings.first_line)
    "rocke_path": "rk",
    "paths": {"repo": "/nonexistent"},
}
HEAD = "a" * 40
BODY = f"{LINE1}\n\n## PR #7 — t — 🟡 First-round: fine — x\n\nbody\n"
URL = "https://github.com/ROCm/r/pull/7#issuecomment-99"
LIVE = {
    "review_automation": {"enabled": True, "mode": "live"},
    "merge_queue": {
        "enabled": True,
        "mode": "live",
        "auto_merge": "enabled",
        "develop_guard": True,
    },
}


def modes(review: str, queue: str) -> dict:
    """The features with review automation's and the merge queue's modes."""
    return {
        "review_automation": dict(LIVE["review_automation"], mode=review),
        "merge_queue": dict(LIVE["merge_queue"], mode=queue),
    }


def done(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")
        # Every file a write makes goes under the test's folder (the day log above all), and the features the writes
        # act for are live (outward_gate refuses one in prototype).
        self.cfg = dict(
            CFG,
            paths={
                "repo": "/nonexistent",
                "working_dir": str(self.dir),
                "root": str(self.dir),
                "daily_dir": str(self.dir / "daily"),
                "posted_dir": str(self.dir),
            },
            features=LIVE,
        )
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
        self.body = self.dir / "p.md"
        self.body.write_text(BODY, encoding="utf-8")
        self.calls = []

    def run_gh(self, *results):
        """subprocess.run for gh: each call takes the next result (a CompletedProcess, or an exception to raise)."""
        todo = list(results)

        def run(args, **kw):
            self.calls.append(args)
            r = todo.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

        return mock.patch.object(post.subprocess, "run", run)

    def kinds(self):
        return [r["kind"] for r in self.store.ledger()]


class TestDisarm(Base):
    """A failed `rk disarm`: the PR stays armed, nothing is ledgered, and the tick asks again."""

    def setUp(self):
        super().setUp()
        self.st["queue_state"] = {
            "armed": {"pr": 7, "head": HEAD, "develop": "d" * 40, "at": "t"}
        }

    def test_a_failed_disarm_keeps_the_pr_armed_and_ledgers_nothing(self):
        for failure in (
            done(1, err="HTTP 502"),
            subprocess.TimeoutExpired(["gh"], 120),
        ):
            with self.subTest(failure=type(failure).__name__), self.run_gh(
                failure
            ), mock.patch.object(
                gh, "rest", return_value={"state": "open", "auto": True}
            ):
                with self.assertRaises(post.PostRefused) as cm:
                    post.disarm(self.cfg, self.st, self.store, 7, "develop moved")
            self.assertIn("stays armed", str(cm.exception))
            self.assertEqual(self.st["queue_state"]["armed"]["pr"], 7)
            self.assertEqual(self.kinds(), [])

    def test_auto_merge_already_off_counts_as_disarmed(self):
        with self.run_gh(done(1, err="not in auto-merge state")), mock.patch.object(
            gh, "rest", return_value={"state": "open", "auto": False}
        ):
            self.assertIn(
                "already off", post.disarm(self.cfg, self.st, self.store, 7, "frozen")
            )
        self.assertIsNone(self.st["queue_state"]["armed"])
        self.assertEqual(self.kinds(), ["disarm"])

    def test_a_disarm_that_took_clears_and_ledgers(self):
        with self.run_gh(done(0)):
            self.assertEqual(
                post.disarm(self.cfg, self.st, self.store, 7, "frozen"),
                "auto-merge disabled",
            )
        self.assertIsNone(self.st["queue_state"]["armed"])
        self.assertEqual(self.kinds(), ["disarm"])

    def test_the_tick_asks_again_while_a_moved_pr_stays_armed(self):
        self.st["prs"]["7"]["head"] = (
            "b" * 40
        )  # pushed since it was armed; the push event was an earlier tick's
        for _ in range(2):
            lines = merge.advance(self.cfg, self.st, self.store, [], [])
            self.assertTrue(
                any(x.startswith("DISARM_NEEDED #7: its head moved") for x in lines),
                lines,
            )


class TestModes(Base):
    """An outward write you type is refused while the feature it sends for is in prototype, naming the mode, or while
    the capability it acts for is off: nothing goes out."""

    def test_a_configuration_with_no_features_writes_nothing(self):
        """features left out takes the schema's defaults: both features in prototype, nothing outward."""
        bare = {k: v for k, v in self.cfg.items() if k != "features"}
        with self.assertRaisesRegex(
            post.PostRefused,
            r"review posting is off \(features.review_automation.mode is prototype: nothing goes out until it is live\)",
        ):
            post.outward_gate(bare, self.st, "a review post", post.POST_FEATURES)
        post.outward_gate(
            dict(bare, features=modes("live", "prototype")),
            self.st,
            "a review post",
            post.POST_FEATURES,
        )

    def test_a_review_post_needs_review_automation_live(self):
        with mock.patch.object(gh, "rest", return_value=HEAD), self.run_gh(
            done(0, out=URL + "\n")
        ):
            for features in (
                modes("prototype", "live"),
                dict(LIVE, review_automation={"enabled": False, "mode": "live"}),
            ):
                with self.assertRaisesRegex(post.PostRefused, "review posting is off"):
                    post.post(
                        dict(self.cfg, features=features),
                        self.st,
                        self.store,
                        7,
                        str(self.body),
                    )
                with self.assertRaisesRegex(post.PostRefused, "review posting is off"):
                    post.preflight(
                        dict(self.cfg, features=features),
                        self.st,
                        self.store,
                        7,
                        str(self.body),
                    )  # --dry-run
            self.assertEqual(self.calls, [])
            self.assertTrue(
                post.post(
                    dict(self.cfg, features=modes("live", "prototype")),
                    self.st,
                    self.store,
                    7,
                    str(self.body),
                ).startswith(URL)
            )
        self.assertEqual(self.kinds(), ["post"])

    def test_each_notice_kind_needs_its_feature_live(self):
        f = self.dir / "n.md"
        for kind, refused, allowed in (
            ("incident", modes("live", "prototype"), modes("prototype", "live")),
            ("fix", modes("live", "prototype"), modes("prototype", "live")),
            ("overlap", modes("prototype", "live"), modes("live", "prototype")),
            ("drift", modes("live", "prototype"), modes("prototype", "live")),
            ("queue", modes("live", "prototype"), modes("prototype", "live")),
        ):
            sends = "review_automation" if kind == "overlap" else "merge_queue"
            with self.subTest(kind=kind):
                f.write_text(
                    f"{post.notice_line(self.cfg, kind)}\n\ntext\n", encoding="utf-8"
                )
                with self.run_gh(), self.assertRaisesRegex(
                    post.PostRefused,
                    f"an? {kind} notice is not sent: features.{sends}.mode is prototype",
                ):
                    post.notice(
                        dict(self.cfg, features=refused),
                        self.st,
                        self.store,
                        7,
                        str(f),
                        kind,
                        f"k-{kind}",
                    )
                with self.run_gh(done(0, out=URL + "\n")):
                    self.assertEqual(
                        post.notice(
                            dict(self.cfg, features=allowed),
                            self.st,
                            self.store,
                            7,
                            str(f),
                            kind,
                            f"k-{kind}",
                        ),
                        URL,
                    )

    def test_the_develop_writes_need_the_develop_guard(self):
        no_guard = dict(
            LIVE, merge_queue=dict(LIVE["merge_queue"], develop_guard=False)
        )
        with self.run_gh(), self.assertRaisesRegex(
            post.PostRefused, "its develop_guard.enabled is true"
        ):
            post.outward_gate(
                dict(self.cfg, features=no_guard),
                self.st,
                "a develop issue",
                ("stability-issues",),
            )
        post.outward_gate(self.cfg, self.st, "a develop issue", ("stability-issues",))

    def test_a_protective_write_runs_in_prototype(self):
        self.st["queue_state"] = {
            "armed": {"pr": 7, "head": HEAD, "develop": "d" * 40, "at": "t"}
        }
        with self.run_gh(done(0)):
            post.disarm(
                dict(self.cfg, features=modes("prototype", "prototype")),
                self.st,
                self.store,
                7,
                "manual",
            )
        self.assertEqual(self.kinds(), ["disarm"])


class TestInDoubt(Base):
    """A write that timed out or failed after GitHub took it is found on GitHub, never made twice."""

    def ours(self, body=BODY, url=URL):
        return [
            {
                "html_url": "https://github.com/ROCm/r/pull/7#issuecomment-1",
                "body": "an older one",
            },
            {"html_url": url, "body": body.replace("\n", "\r\n")},
        ]

    def rest(self, comments):
        def rest(path, jq=None, paginate=False, timeout=120):
            if path.startswith("repos/o/r/issues/7/comments?per_page=100&since="):
                self.assertIn('select(.user.login == "me")', jq)
                if isinstance(comments, BaseException):
                    raise comments
                return comments
            return HEAD  # the live head

        return mock.patch.object(gh, "rest", rest)

    def test_a_timed_out_post_that_went_out_is_recorded_not_repeated(self):
        with self.run_gh(subprocess.TimeoutExpired(["gh"], 120)), self.rest(
            self.ours()
        ):
            self.assertTrue(
                post.post(self.cfg, self.st, self.store, 7, str(self.body)).startswith(
                    URL
                )
            )
        self.assertEqual(
            [(r["kind"], r["ref"]) for r in self.store.ledger()], [("post", URL)]
        )
        self.assertEqual(len(self.calls), 1)

    def test_a_failure_github_has_no_comment_for_is_a_failure(self):
        with self.run_gh(done(1, err="HTTP 502")), self.rest(
            self.ours(body="another body")
        ):
            with self.assertRaisesRegex(
                post.PostRefused, "gh pr comment failed: HTTP 502"
            ):
                post.post(self.cfg, self.st, self.store, 7, str(self.body))
        self.assertEqual(self.kinds(), [])
        self.assertEqual(post._doubts(self.store), {})

    def test_an_unsettled_write_is_read_back_before_the_next_try(self):
        with self.run_gh(subprocess.TimeoutExpired(["gh"], 120)), self.rest(
            gh.GhError("HTTP 502")
        ):
            with self.assertRaisesRegex(post.PostRefused, "in doubt"):
                post.post(self.cfg, self.st, self.store, 7, str(self.body))
        self.assertEqual(self.kinds(), [])
        self.assertEqual(list(post._doubts(self.store)), [f"post:7:{HEAD}"])
        # The retry reads GitHub first, finds the comment, and posts nothing.
        with self.run_gh(), self.rest(self.ours()):
            self.assertTrue(
                post.post(self.cfg, self.st, self.store, 7, str(self.body)).startswith(
                    URL
                )
            )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.kinds(), ["post"])
        self.assertEqual(post._doubts(self.store), {})

    def test_a_notice_whose_answer_was_lost_is_recorded_once(self):
        f = self.dir / "n.md"
        text = f"{post.notice_line(self.cfg, 'incident')}\n\ndevelop is red\n"
        f.write_text(text, encoding="utf-8")
        with self.run_gh(done(1, err="stream error: CANCEL")), self.rest(
            self.ours(body=text)
        ):
            self.assertEqual(
                post.notice(
                    self.cfg, self.st, self.store, 7, str(f), "incident", "INC-1"
                ),
                URL,
            )
        with self.assertRaisesRegex(post.PostRefused, "already on #7"):
            post.notice(self.cfg, self.st, self.store, 7, str(f), "incident", "INC-1")
        self.assertEqual(self.kinds(), ["notice-incident"])


class TestBodyFiles(Base):
    """A write hands gh its text in a temporary file, its path after `--body-file` or `-F body=@`: the file is there
    while gh reads it and gone after, whether the write went out, failed or timed out. Each test has a temporary folder
    of its own, and a write leaves it as it found it."""

    def setUp(self):
        super().setUp()
        self.tmp = self.dir / "tmp"
        self.tmp.mkdir()
        patcher = mock.patch.object(tempfile, "tempdir", str(self.tmp))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.read = []  # each body file's text, as gh read it

    def gh(self, *results):
        """run_gh's gh, which also reads each body file it is given, as gh does."""
        todo = list(results)

        def run(args, **kw):
            self.calls.append(args)
            files = [a[len("body=@") :] for a in args if a.startswith("body=@")]
            files += [args[i + 1] for i, a in enumerate(args) if a == "--body-file"]
            self.read += [pathlib.Path(f).read_text(encoding="utf-8") for f in files]
            r = todo.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

        return mock.patch.object(post.subprocess, "run", run)

    def github(self, labels=()):
        """GitHub's answers: PR 7 is open and ready, issue 9 is open with `labels`, and the read-back after a failed
        write finds nothing (the write did not go out)."""

        def rest(path, jq=None, paginate=False, timeout=120):
            if path == "repos/o/r/pulls/7":
                return {"state": "open", "draft": False}
            if path == "repos/o/r/issues/9":
                return {
                    "title": "test_x fails",
                    "state": "open",
                    "labels": list(labels),
                    "pr": False,
                }
            return []

        return mock.patch.object(gh, "rest", rest)

    def left(self):
        return sorted(p.name for p in self.tmp.iterdir())

    def test_a_queue_notice_takes_its_body_file_away(self):
        text = f"{post.notice_line(self.cfg, 'queue')}\n\nfirst in the queue\n"
        for what, result, refusal in (
            ("post fails", done(1, err="HTTP 502"), "gh pr comment failed: HTTP 502"),
            ("post", done(0, out=URL + "\n"), None),
            ("edit fails", done(1, err="HTTP 502"), "editing comment 99 failed"),
            (
                "edit times out",
                subprocess.TimeoutExpired(["gh"], 120),
                "editing comment 99 timed out",
            ),
            ("edit", done(0), None),
        ):
            with self.subTest(what), self.gh(result), self.github():
                if refusal:
                    with self.assertRaisesRegex(post.PostRefused, refusal):
                        post.queue_notice(self.cfg, self.st, self.store, 7, text, "k1")
                else:
                    self.assertEqual(
                        post.queue_notice(self.cfg, self.st, self.store, 7, text, "k1"),
                        URL,
                    )
                self.assertEqual(self.read[-1], text)
                self.assertEqual(self.left(), [])
        self.assertEqual(len(self.read), 5)
        self.assertEqual(self.kinds(), ["notice-queue-sticky", "notice-queue-edit"])

    def test_a_develop_issue_takes_its_body_file_away(self):
        from rkmon_lib import breakage

        body = f"{post.notice_line(self.cfg, 'report')}\n\ntest_x fails on develop\n"
        f = self.dir / "issue.md"
        f.write_text(
            "---\ntitle: [rocke] test_x fails on develop\nlabels: "
            f"{breakage.labels(self.cfg)[breakage.RED]}\n---\n{body}",
            encoding="utf-8",
        )
        url = "https://github.com/ROCm/r/issues/12"
        with mock.patch.object(
            post, "label_missing", lambda cfg, name: []
        ), self.github():
            with self.gh(done(1, err="HTTP 502")), self.assertRaisesRegex(
                post.PostRefused, "gh issue create failed: HTTP 502"
            ):
                post.issue_file(self.cfg, self.st, self.store, str(f), new=True)
            self.assertEqual(self.left(), [])
            with self.gh(done(0, out=url + "\n")):
                self.assertEqual(
                    post.issue_file(self.cfg, self.st, self.store, str(f), new=True),
                    url,
                )
            self.assertEqual(self.left(), [])
        self.assertEqual(self.read, [body, body])
        self.assertEqual(self.kinds(), ["issue"])

    def test_a_fix_link_takes_its_body_files_away(self):
        from rkmon_lib import breakage

        comment_url = "https://github.com/ROCm/r/issues/9#issuecomment-1"
        with self.github(labels=[breakage.labels(self.cfg)[breakage.STABILITY]]):
            # The issue comment fails; the next try comments and the PR notice fails; the last one sends the notice.
            for results, refusal in (
                ((done(1, err="HTTP 502"),), "gh issue comment failed: HTTP 502"),
                (
                    (done(0, out=comment_url + "\n"), done(1, err="HTTP 502")),
                    "gh pr comment failed: HTTP 502",
                ),
                ((done(0, out=URL + "\n"),), None),
            ):
                with self.subTest(refusal=refusal), self.gh(*results):
                    if refusal:
                        with self.assertRaisesRegex(post.PostRefused, refusal):
                            post.fix_link(self.cfg, self.st, self.store, 9, 7)
                    else:
                        self.assertIn(
                            f"PR notice {URL}",
                            post.fix_link(self.cfg, self.st, self.store, 9, 7),
                        )
                    self.assertEqual(self.left(), [])
        self.assertEqual(
            [r.splitlines()[0] for r in self.read],
            [post.notice_line(self.cfg, "report")] * 2
            + [post.notice_line(self.cfg, "fix")] * 2,
        )
        self.assertEqual(self.kinds(), ["issue-comment", "notice-fix"])


INTERNAL = "https://wiki.example.net/x"  # a link to a host off the allow list
PLACEHOLDER = post.LOGIN_PLACEHOLDER


class TestPublicRules(Base):
    """Every outward writer refuses a text that holds the operator's placeholder or a link to a non-public host, before
    it reads or writes GitHub (gh is faked: any call fails the test)."""

    def no_github(self):
        def rest(*a, **kw):
            raise AssertionError(f"GitHub was read: {a}")

        return mock.patch.object(gh, "rest", rest)

    def refused(self, call, *words):
        self.calls = []  # each writer on its own: one that wrote fails alone
        with self.run_gh(), self.no_github(), self.assertRaises(post.PostRefused) as cm:
            call()
        for w in words:
            self.assertIn(w, str(cm.exception))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.kinds(), [])

    def test_rk_post_refuses_the_placeholder_and_a_non_public_link(self):
        for extra, words in (
            (f"Thanks, {PLACEHOLDER} will re-approve.", f"placeholder {PLACEHOLDER}"),
            (f"see {INTERNAL}", "link to a non-public host"),
        ):
            self.body.write_text(BODY + extra + "\n", encoding="utf-8")
            for name, call in (
                (
                    "preflight",
                    lambda: post.preflight(
                        self.cfg, self.st, self.store, 7, str(self.body)
                    ),
                ),
                (
                    "post",
                    lambda: post.post(self.cfg, self.st, self.store, 7, str(self.body)),
                ),
            ):
                with self.subTest(name=name, extra=extra):
                    self.refused(call, words)

    def test_the_gpu_tests_row_is_checked_again(self):
        """The review is clean, the GPU tests row rk post adds for this head is not: refused."""
        self.st["gpu"] = {
            "jobs": [
                {
                    "pr": 7,
                    "head": HEAD,
                    "arch": "gfx942",
                    "state": "ended",
                    "result": {"status": "pass", "detail": f"2 passed, log {INTERNAL}"},
                }
            ]
        }
        self.body.write_text(
            BODY + "\n| Row | Status |\n|---|---|\n| Our checks | 🟢 Clean |\n",
            encoding="utf-8",
        )
        self.refused(
            lambda: post.preflight(self.cfg, self.st, self.store, 7, str(self.body)),
            "link to a non-public host",
        )
        self.st["gpu"]["jobs"][0]["result"]["detail"] = "2 passed"
        with self.no_github():
            with self.assertRaises(
                AssertionError
            ):  # clean: it goes on to read the head
                post.preflight(self.cfg, self.st, self.store, 7, str(self.body))

    def test_the_notices_issues_and_change_requests_refuse_the_placeholder(self):
        f = self.dir / "w.md"

        def notice():
            f.write_text(
                f"{post.notice_line(self.cfg, 'incident')}\n\nask {PLACEHOLDER}\n",
                encoding="utf-8",
            )
            return post.notice(
                self.cfg, self.st, self.store, 7, str(f), "incident", "INC-1"
            )

        def request_changes():
            f.write_text(
                f"{post.notice_line(self.cfg, 'hold')}\n\nask {PLACEHOLDER}\n",
                encoding="utf-8",
            )
            return post.request_changes(
                self.cfg, self.st, self.store, 7, str(f), user=True
            )

        def queue_notice():
            return post.queue_notice(
                self.cfg,
                self.st,
                self.store,
                7,
                f"{post.notice_line(self.cfg, 'queue')}\n\nask {PLACEHOLDER}\n",
                "k1",
            )

        def issue_file():
            from rkmon_lib import breakage

            f.write_text(
                "---\ntitle: [rocke] test_x fails on develop\nlabels: "
                f"{breakage.labels(self.cfg)[breakage.RED]}\n---\n"
                f"{post.notice_line(self.cfg, 'report')}\n\nask {PLACEHOLDER}\n",
                encoding="utf-8",
            )
            # the label lookup is the one read before the refusal: the label is there
            with mock.patch.object(post, "label_missing", lambda cfg, name: []):
                return post.issue_file(self.cfg, self.st, self.store, str(f), new=True)

        def issue_comment():
            f.write_text(
                f"{post.notice_line(self.cfg, 'report')}\n\nask {PLACEHOLDER}\n",
                encoding="utf-8",
            )
            return post.issue_comment(self.cfg, self.st, self.store, 9, str(f), "k1")

        for call in (notice, request_changes, queue_notice, issue_file, issue_comment):
            with self.subTest(writer=call.__name__):
                self.refused(call, f"placeholder {PLACEHOLDER}")


class TestTeamsWebhook(unittest.TestCase):
    """The webhook URL is a credential: a note on a second line is not part of it, and no error carries it."""

    HOOK = (
        "https://prod-1.hooks.example.invalid:443/workflows/abc/triggers/manual/paths/invoke"
        "?api-version=1&sig=SECRET"
    )

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.file = pathlib.Path(self.d.name) / "teams_webhook"
        self.file.write_text(f"\n{self.HOOK}\nthe team channel, set up 11-01\n")
        self.cfg = {
            "teams": {"webhook_file": str(self.file), "channel": "the team channel"}
        }

    def test_only_the_first_line_is_the_url(self):
        seen = []

        class Resp:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            return Resp()

        with mock.patch("urllib.request.urlopen", urlopen):
            self.assertEqual(post._teams_post(self.cfg, {"x": 1}), 202)
        self.assertEqual(seen, [self.HOOK])

    def test_no_error_carries_the_url(self):
        import http.client
        import urllib.error

        for exc in (
            http.client.InvalidURL(
                f"URL can't contain control characters. '{self.HOOK[30:]}'"
            ),
            ValueError(f"unknown url type: '{self.HOOK[8:]}'"),
            urllib.error.URLError(f"bad {self.HOOK}"),
            urllib.error.HTTPError(self.HOOK, 400, "Bad Request", {}, None),
        ):
            with self.subTest(exc=type(exc).__name__), mock.patch(
                "urllib.request.urlopen", side_effect=exc
            ):
                with self.assertRaises(post.PostRefused) as cm:
                    post._teams_post(self.cfg, {"x": 1})
                text = str(cm.exception)
                for part in ("SECRET", "sig=", "/workflows/", "hooks.example.invalid"):
                    self.assertNotIn(part, text)

    def test_the_tick_line_carries_no_url(self):
        from rkmon_lib import needs_teams

        st = empty_state()
        cfg = dict(
            self.cfg,
            features={
                "alerts": {"enabled": True, "develop_breaks": True, "needs_you": True}
            },
            repo="o/r",
        )
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            needs_teams, "message", lambda cfg, items: {}
        ), mock.patch(
            "urllib.request.urlopen",
            side_effect=ValueError(f"unknown url type: '{self.HOOK}'"),
        ):
            lines = needs_teams.send(
                cfg,
                st,
                Store(pathlib.Path(d)),
                [
                    {
                        "key": "k1",
                        "kind": "MERGE",
                        "pr": 5,
                        "ask": "merge it",
                        "url": None,
                    }
                ],
            )
        self.assertTrue(lines and "not sent" in lines[0], lines)
        self.assertNotIn("SECRET", json.dumps([lines, st]))


if __name__ == "__main__":
    unittest.main()
