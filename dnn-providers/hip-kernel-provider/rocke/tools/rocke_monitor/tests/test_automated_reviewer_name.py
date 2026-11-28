# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The automated reviewer name on what a deployment posts. Each notice's first line ends "by
<automated_reviewer_name>", from one home (post.notice_line), and every guarded writer refuses a notice whose first
line names another deployment, before it reads or writes GitHub. Each review post records the name it gives
(reviews.automated_reviewer_names_used), so the deployment's earlier reviews stay its own after a rename.
"""

import errno
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

from rkmon_lib import (  # noqa: E402
    bindings,
    breakage,
    gh,
    gpu,
    merge,
    post,
    registry,
    reviewer_name,
    settings_file,
    state_store,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

ALPHA, BETA = "team-alpha", "team-beta"
HEAD = "a" * 40
URL = "https://github.com/ROCm/r/pull/{n}#issuecomment-99"
LIVE = {
    "review_automation": {"enabled": True, "mode": "live"},
    "merge_queue": {
        "enabled": True,
        "mode": "live",
        "auto_merge": "enabled",
        "develop_guard": True,
    },
}


def named(name: str, base: pathlib.Path | None = None, **over) -> dict:
    """A deployment's configuration, its automated reviewer name NAME, with review automation, the merge queue and
    the remote GPU tests live; its files under BASE."""
    features = dict(LIVE, review_automation=dict(LIVE["review_automation"]))
    features["review_automation"]["automated_reviewer_name"] = name
    where = str(base or "/nonexistent")
    return {
        "incident_issue": guard_labels.INCIDENT_ISSUE,
        "repo": "o/r",
        "operator": {"github_login": "me"},
        "max_posts_per_hour": 6,
        "worker": {"model": "qwen"},
        "rocke_path": "rk",
        "features": features,
        "gpu": {"enabled": True},
        "paths": {
            "repo": "/nonexistent",
            "working_dir": where,
            "root": where,
            "daily_dir": f"{where}/daily",
            "posted_dir": where,
        },
        **over,
    }


class TestNoticeLines(unittest.TestCase):
    def test_each_notice_s_first_line_ends_by_its_deployment_s_name(self):
        for name in (ALPHA, BETA):
            cfg = named(name)
            for kind, what in post.NOTICES.items():
                with self.subTest(name=name, kind=kind):
                    self.assertEqual(post.notice_line(cfg, kind), f"{what} by {name}")
        self.assertEqual(
            post.notice_line(named(ALPHA), "gpu"),
            "Automated GPU test notice by team-alpha",
        )
        # With the name blank, the container's name.
        self.assertEqual(
            post.notice_line(named("", deployment={"project": "rkmon-b"}), "hold"),
            "Automated merge-queue hold by rkmon-b",
        )

    def test_each_notice_the_monitor_writes_starts_with_its_deployment_s_line(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        job = {
            "arch": "gfx942",
            "head": "a" * 40,
            "develop": "d" * 40,
            "result": {"pr_failed": ["t.py::x"], "dev_passed": ["t.py::x"]},
        }
        for name in (ALPHA, BETA):
            cfg = named(name)
            for kind, text in (
                ("queue", merge.notice_text(cfg, st, 3, "next", "abc")),
                ("gpu", gpu.notice_text(cfg, job)),
                ("fix", post.fix_notice_text(cfg, {"author": "dev"}, 11, "t fails")),
                (
                    "queue",
                    post.automerge_off_text(
                        cfg, {"author": "dev"}, {"reasons": ["r"], "fixes": ["f"]}
                    ),
                ),
            ):
                with self.subTest(name=name, kind=kind):
                    self.assertEqual(
                        text.splitlines()[0], f"{post.NOTICES[kind]} by {name}"
                    )
                    self.assertEqual(
                        post.check(
                            dict(cfg, review_first_line=post.notice_line(cfg, kind)),
                            text,
                        ),
                        [],
                    )


class TestAnotherDeploymentsLine(unittest.TestCase):
    """Deployment alpha's writers, each given a notice whose first line is deployment beta's: refused, naming alpha's
    line, before GitHub is read or written (any gh call fails the test)."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")
        self.cfg, self.other = named(ALPHA, self.dir), named(BETA, self.dir)
        self.st = empty_state()
        self.st["prs"]["7"] = {
            "n": 7,
            "head": "a" * 40,
            "status": "open",
            "draft": False,
            "author": "dev",
            "ours": None,
        }
        self.file = self.dir / "notice.md"

    def no_github(self):
        def called(*a, **kw):
            raise AssertionError(f"GitHub was called: {a}")

        return mock.patch.object(gh, "rest", called), mock.patch.object(
            post.subprocess, "run", called
        )

    def write(self, kind: str) -> str:
        """The notice file, its first line deployment beta's line of KIND."""
        self.file.write_text(
            f"{post.notice_line(self.other, kind)}\n\nwhat happened\n",
            encoding="utf-8",
        )
        return str(self.file)

    def test_each_writer_refuses_it(self):
        def notice(kind: str):
            return lambda: post.notice(
                self.cfg, self.st, self.store, 7, self.write(kind), kind, "k1"
            )

        def queue_notice():
            text = f"{post.notice_line(self.other, 'queue')}\n\nnext\n"
            return post.queue_notice(self.cfg, self.st, self.store, 7, text, "k1")

        def request_changes():
            return post.request_changes(
                self.cfg, self.st, self.store, 7, self.write("hold"), user=True
            )

        def issue_file():
            labels = breakage.labels(self.cfg)[breakage.RED]
            self.file.write_text(
                f"---\ntitle: [develop] test_x fails\nlabels: {labels}\n---\n"
                f"{post.notice_line(self.other, 'report')}\n\nwhat happened\n",
                encoding="utf-8",
            )
            with mock.patch.object(post, "label_missing", lambda cfg, name: []):
                return post.issue_file(
                    self.cfg, self.st, self.store, str(self.file), new=True
                )

        def issue_comment():
            return post.issue_comment(
                self.cfg, self.st, self.store, 9, self.write("report"), "k1"
            )

        writers = [
            (f"notice --kind {kind}", kind, notice(kind))
            for kind in post.NOTICE_CHOICES
        ] + [
            ("queue_notice", "queue", queue_notice),
            ("request_changes", "hold", request_changes),
            ("issue_file", "report", issue_file),
            ("issue_comment", "report", issue_comment),
        ]
        for writer, kind, call in writers:
            mine = post.notice_line(self.cfg, kind)
            rest, run = self.no_github()
            with self.subTest(writer=writer), rest, run:
                with self.assertRaisesRegex(
                    post.PostRefused, re.escape(f"line 1 must be exactly {mine!r}")
                ):
                    call()
        self.assertEqual(self.store.ledger(), [])


class TestNamesUsed(unittest.TestCase):
    """Each review a deployment posts adds the name it posts under to reviews.automated_reviewer_names_used in its
    internal state, before the review goes out; under a new name, its earlier reviews stay its own. An internal state
    that cannot be read or written stops the post before it goes out, refused with the fix.
    """

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")
        self.st = empty_state()

    def post(self, name: str, n: int, run=None) -> str:
        """A first-round review of PR N, posted under NAME; GitHub's answer RUN (by default, the comment's URL)."""
        cfg = named(name, self.dir)
        self.st["prs"][str(n)] = {
            "n": n,
            "head": HEAD,
            "status": "open",
            "draft": False,
            "author": "dev",
            "ours": None,
        }
        body = self.dir / f"p{n}.md"
        body.write_text(
            f"{bindings.first_line(cfg)}\n\n## PR #{n} — t — 🟡 First-round: fine — x\n\nbody\n",
            encoding="utf-8",
        )
        answer = run or (
            lambda *a, **kw: subprocess.CompletedProcess(
                [], 0, stdout=URL.format(n=n), stderr=""
            )
        )
        with mock.patch.object(gh, "rest", return_value=HEAD), mock.patch.object(
            post.subprocess, "run", answer
        ):
            return post.post(cfg, self.st, self.store, n, str(body))

    @staticmethod
    def going_out(*a, **kw):
        """GitHub's answer to a review that must not go out."""
        raise AssertionError(f"the review went out: {a}")

    def used(self):
        """The names used that the internal state records (registry.MISSING: none)."""
        return registry.get(state_store.read(self.dir), reviewer_name.NAMES_USED)

    def test_the_names_used_grow_with_each_post_under_a_new_name(self):
        self.assertIs(self.used(), registry.MISSING)
        self.post(ALPHA, 7)
        self.assertEqual(self.used(), [ALPHA])
        self.post(ALPHA, 8)
        self.assertEqual(self.used(), [ALPHA])
        self.post(BETA, 9)
        self.assertEqual(self.used(), [ALPHA, BETA])
        self.assertEqual(
            [r["kind"] for r in self.store.ledger()], ["post", "post", "post"]
        )
        # Renamed to beta, the deployment still owns the review alpha's name gives.
        cfg = dict(named(BETA, self.dir), **state_store.read(self.dir))
        self.assertEqual(reviewer_name.names(cfg), [BETA, ALPHA])
        alpha_review = (self.dir / "p7.md").read_text(encoding="utf-8")
        at = self.store.ledger()[0]["ts"]  # when alpha's review went out
        self.assertTrue(reviewer_name.own_review(cfg, alpha_review, at))
        self.assertFalse(
            reviewer_name.own_review(named(BETA, self.dir), alpha_review, at)
        )

    def test_a_state_that_cannot_be_read_stops_the_post_with_its_fix(self):
        """A broken internal state: the refusal says why, then the fix state_store names for the file (mend it, or move
        it aside), and nothing is written."""
        state = self.dir / "state" / state_store.STATE_FILE
        state.write_text("{", encoding="utf-8")
        with self.assertRaises(state_store.StateError) as broken:
            state_store.read(self.dir)
        with self.assertRaises(post.PostRefused) as refused:
            self.post(ALPHA, 7, self.going_out)
        said, exc = str(refused.exception), broken.exception
        self.assertEqual(
            said,
            f"the automated reviewer name {ALPHA!r} cannot be recorded: {exc}\nfix: {exc.fix('rkmon')}",
        )
        self.assertIn(
            "\nfix: mend it at the line and column above, or move it aside: mv ", said
        )
        self.assertEqual(self.store.ledger(), [])
        self.assertEqual(state.read_text(encoding="utf-8"), "{")

    @unittest.skipIf(os.geteuid() == 0, "root writes in a folder that is read-only")
    def test_a_state_that_cannot_be_written_stops_the_post_with_its_fix(self):
        """An internal state that reads but cannot be written, its folder or its lock file kept from the monitor by its
        mode: the post is refused (no error escapes), with the fix that makes that place writable at the mode the
        monitor and the writer share, and nothing is written."""
        self.post(ALPHA, 7)
        folder = self.dir / "state"
        lock = folder / state_store.LOCK_FILE
        self.assertTrue(lock.is_file())
        self.addCleanup(os.chmod, folder, 0o770)
        self.addCleanup(os.chmod, lock, 0o660)
        why = f"the internal state {state_store.path(self.dir)} cannot be written: {os.strerror(errno.EACCES)}"
        for place, mode, shared, back in (
            (folder, 0o555, "2770", 0o770),
            (lock, 0o444, "660", 0o660),
        ):
            with self.subTest(place=place.name):
                os.chmod(place, mode)
                try:
                    with self.assertRaises(post.PostRefused) as refused:
                        self.post(BETA, 8, self.going_out)
                finally:
                    os.chmod(place, back)
                self.assertEqual(
                    str(refused.exception),
                    f"the automated reviewer name {BETA!r} cannot be recorded: {why}\nfix: chmod {shared} "
                    f"{settings_file.shown(place)} as its owner: the monitor and the writer share it",
                )
                self.assertEqual(self.used(), [ALPHA])
        self.assertEqual([r["kind"] for r in self.store.ledger()], ["post"])


if __name__ == "__main__":
    unittest.main()
