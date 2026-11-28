# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk's GitHub writes (rkmon_lib/post.py) where this process reads GitHub with the write login's folder, as the
writer's rk does (gh.login_dirs), and that folder holds no login (github_write not set, as in prototype): a develop
issue, or another write whose checks read GitHub (a change request, auto-merge on or off, a relabel, an assignee, a
label, a fix link), that the merge queue holds is kept as its unsent item without reading GitHub, and one that would go
out is refused before any check, naming the command that sets the login. The record of the writes in doubt is written
whole.
"""

import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import guard_labels  # noqa: E402
from rkmon_lib import breakage, features, gh, outward, post, state_store  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402


HEAD = "a" * 40
NO_LOGIN = "the GitHub write login is not set (the login github_write: rkmon login github_write)"


def config(mode: str) -> dict:
    """A deployment whose develop guard files develop issues and whose merge queue holds PRs, in MODE."""
    return {
        "rocke_path": "rk",
        "repo": "o/r",
        "operator": {"github_login": "me"},
        "incident_issue": guard_labels.INCIDENT_ISSUE,
        "features": {
            "merge_queue": {
                "mode": mode,
                "auto_merge": "enabled",
                "auto_hold": True,
                "develop_guard": True,
            }
        },
        "max_posts_per_hour": 6,
        "review_first_line": "x",
    }


class TestNoWriteLogin(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)
        self.store = Store(self.d / "state")
        self.login = self.d / "gh"  # the write login's folder: no login in it
        self.login.mkdir()
        self.file = self.d / "issue.md"
        self.file.write_text(
            f"---\ntitle: [develop] test_x fails on develop since #1111\nlabels: {guard_labels.STABILITY}\n"
            f"assign: dev-k\n---\n{post.notice_line(config('live'), 'report')}\n\nculprit #1111\n",
            encoding="utf-8",
        )
        # Each GitHub read fails as gh does with no login; no write is made.
        self.reads = mock.Mock(
            side_effect=gh.GhError(
                "gh: To get started with GitHub CLI, please run:  gh auth login"
            )
        )
        self.writes = mock.Mock(side_effect=AssertionError("nothing is written"))
        for p in (
            mock.patch.object(gh, "rest", self.reads),
            mock.patch.object(post, "_gh_write", self.writes),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(gh.configure, {})

    def as_the_writer(self, cfg: dict, own: bool):
        """The writer's rk: its reads use the write login's folder (gh.login_dirs), which CFG names from here on; OWN:
        the monitor's own write."""
        cfg["gh"] = {"write_config_dir": str(self.login)}
        env = {gh.BROKERED: "1"}
        if own:
            env[outward.OWN_VAR] = "1"
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        gh.configure(cfg)
        self.assertEqual(gh.config_dir("read"), str(self.login))

    def test_a_held_develop_issue_is_kept_without_reading_github(self):
        cfg = config("prototype")
        self.as_the_writer(cfg, own=True)
        said = post.issue_file(cfg, empty_state(), self.store, str(self.file))
        self.assertTrue(
            said.startswith(
                f"{post.UNSENT}: a develop issue for a new issue in o/r, kept as an unsent item: "
            ),
            said,
        )
        self.reads.assert_not_called()
        self.writes.assert_not_called()
        kept = state_store.unsent_items(self.store.dir)
        self.assertEqual(
            [(i["type"], i["for"]) for i in kept],
            [(state_store.ISSUE, "a new issue in o/r")],
        )
        self.assertIn(
            f"labels: {guard_labels.STABILITY}; assigned to dev-k", kept[0]["content"]
        )

    def test_a_develop_issue_that_would_go_out_is_refused_first_naming_the_login(self):
        cfg = config("live")
        self.as_the_writer(cfg, own=True)
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "rkmon"}):
            with self.assertRaises(post.PostRefused) as refused:
                post.issue_file(cfg, empty_state(), self.store, str(self.file))
        self.assertEqual(
            str(refused.exception), f"a develop issue is not sent: {NO_LOGIN}"
        )
        self.reads.assert_not_called()
        self.assertEqual(state_store.unsent_items(self.store.dir), [])

    def state(self) -> dict:
        """PR 7: cleared, the next member of a validated batch, someone else's auto-merge on it at risk; develop issue
        9 as the tick last read it."""
        st = empty_state()
        st["prs"] = {
            "7": {
                "n": 7,
                "head": HEAD,
                "status": "open",
                "author": "dev-k",
                "closes": [],
                "auto_merge": {"by": "dev-k"},
            }
        }
        st["cleared"] = [7]
        st["batches"] = [{"status": "validated", "members": [7]}]
        st["automerge_risks"] = {
            "7": {
                "head": HEAD,
                "reasons": ["a hold label"],
                "fixes": ["remove the label"],
                "by": "dev-k",
            }
        }
        issue = {"n": 9, "title": "test_x fails on develop", "kind": breakage.STABILITY}
        st["post_merge"] = {
            "open": [dict(issue, labels=[guard_labels.STABILITY])],
            "at": None,
        }
        return st

    def commands(self, cfg: dict, st: dict) -> list:
        """The writes whose checks read GitHub, each as rk runs it on PR 7 or develop issue 9: what it is, what its
        unsent item is for, and the call."""
        hold = self.d / "hold.md"
        hold.write_text(
            f"{post.notice_line(cfg, 'hold')}\n\nThe widget test fails at this head: fix it, then re-request a "
            "review.\n",
            encoding="utf-8",
        )
        reason = self.d / "reason.md"
        reason.write_text("Other merges go first.\n", encoding="utf-8")
        store = self.store
        return [
            (
                "a change request",
                "pull request 7",
                lambda: post.request_changes(cfg, st, store, 7, str(hold), user=True),
            ),
            ("auto-merge", "pull request 7", lambda: post.arm(cfg, st, store, 7)),
            (
                "turning auto-merge off",
                "pull request 7",
                lambda: post.automerge_off_user(cfg, st, store, 7, str(reason)),
            ),
            (
                "relabelling a develop issue",
                "issue 9",
                lambda: post.issue_relabel(
                    cfg, st, store, 9, breakage.STABILITY, "dev-k"
                ),
            ),
            (
                "assigning a develop issue",
                "issue 9",
                lambda: post.issue_assign(cfg, store, 9, "dev-k"),
            ),
            (
                "a new label",
                f"the repository {breakage.issue_repository(cfg)}",
                lambda: post.label_create(
                    cfg, store, "widget-break", "A develop break", "d73a4a"
                ),
            ),
        ]

    def test_other_held_writes_are_kept_without_reading_github(self):
        cfg = config("prototype")
        self.as_the_writer(cfg, own=True)
        st = self.state()
        wanted = []
        for what, for_, write in self.commands(cfg, st):
            with self.subTest(what=what):
                said = write()
                self.assertTrue(
                    said.startswith(
                        f"{post.UNSENT}: {what} for {for_}, kept as an unsent item: "
                    ),
                    said,
                )
                wanted.append(for_)
        self.reads.assert_not_called()
        self.writes.assert_not_called()
        kept = state_store.unsent_items(self.store.dir)
        self.assertEqual([i["for"] for i in kept], wanted)
        # The relabel not read keeps the whole edit: the CI-red labels off, the stability label on, the assignee.
        self.assertEqual(
            kept[3]["content"],
            f"--add-label {guard_labels.STABILITY} --remove-label {guard_labels.RED} --remove-label "
            f"{guard_labels.EXTRA} --add-assignee dev-k",
        )

    def test_other_writes_that_would_go_out_are_refused_first_naming_the_login(self):
        cfg = config("live")
        self.as_the_writer(cfg, own=True)
        st = self.state()
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "rkmon"}):
            for what, _, write in self.commands(cfg, st):
                with self.subTest(what=what):
                    with self.assertRaises(post.PostRefused) as refused:
                        write()
                    self.assertEqual(
                        str(refused.exception), f"{what} is not sent: {NO_LOGIN}"
                    )
            # The guard's own turn-off too (it acts only while the merge queue sends).
            with mock.patch.object(features, "guard_mode", return_value="act"):
                with self.assertRaises(post.PostRefused) as refused:
                    post.automerge_off(cfg, st, self.store, 7)
            self.assertEqual(
                str(refused.exception),
                f"turning auto-merge off is not sent: {NO_LOGIN}",
            )
            with self.assertRaises(post.PostRefused) as refused:
                post.fix_link(cfg, st, self.store, 9, 7)
            self.assertEqual(
                str(refused.exception), f"a fix link is not sent: {NO_LOGIN}"
            )
        self.reads.assert_not_called()
        self.writes.assert_not_called()
        self.assertEqual(state_store.unsent_items(self.store.dir), [])
        self.assertFalse(st.get("fix_links"))

    def test_a_held_fix_link_is_recorded_and_its_writes_kept_from_the_last_read(self):
        """A fix link the merge queue holds is recorded, and its issue comment and PR notice are kept, with the issue's
        title and kind as the tick last read the open develop issues."""
        cfg = config("prototype")
        self.as_the_writer(cfg, own=True)
        st = self.state()
        said = post.fix_link(cfg, st, self.store, 9, 7).splitlines()
        self.assertEqual(said[0], f"#7 is a develop fix ({guard_labels.STABILITY} #9)")
        self.assertEqual(
            [ln.split(", kept as")[0] for ln in said[1:]],
            [
                f"{post.UNSENT}: an issue comment for issue 9",
                f"{post.UNSENT}: a fix notice for pull request 7",
            ],
        )
        self.assertEqual(st["fix_links"], {"9": [7]})
        self.reads.assert_not_called()
        self.writes.assert_not_called()
        kept = state_store.unsent_items(self.store.dir)
        self.assertIn(
            "this PR looks like the fix for #9 (test_x fails on develop).",
            kept[1]["content"],
        )

    def test_with_the_write_login_set_its_checks_read_github(self):
        (self.login / gh.HOSTS_FILE).write_text("github.com:\n", encoding="utf-8")
        self.reads.side_effect = gh.GhError("gh api: Not Found (HTTP 404)")
        cfg = config("live")
        self.as_the_writer(cfg, own=True)
        with self.assertRaisesRegex(
            post.PostRefused, f"o/r has no {guard_labels.STABILITY} label"
        ):
            post.issue_file(cfg, empty_state(), self.store, str(self.file))
        self.reads.assert_called()


class TestWritesInDoubt(unittest.TestCase):
    def test_the_record_is_written_whole_at_the_shared_mode(self):
        """The record of the writes in doubt is written whole (state_store.write_file): a new file of its own beside
        it, made new and never through a link, at the mode of the files the monitor and the writer share whatever the
        umask."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        store = Store(d / "state")
        elsewhere = d / "elsewhere.json"
        elsewhere.write_text("{}\n", encoding="utf-8")
        (store.dir / (post.DOUBT_FILE + ".tmp")).symlink_to(elsewhere)
        umask = os.umask(0o077)
        try:
            post._doubt(store, "post:7:abc", {"since": "s", "at": "t"})
        finally:
            os.umask(umask)
        record = store.dir / post.DOUBT_FILE
        self.assertFalse(record.is_symlink())
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(post._doubts(store), {"post:7:abc": {"since": "s", "at": "t"}})
        self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o660)
        post._doubt(store, "post:7:abc", None)
        self.assertFalse(record.exists())  # nothing in doubt: no file


if __name__ == "__main__":
    unittest.main()
