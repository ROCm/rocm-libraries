# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The follow-up policy: a push of merges only runs its checks and is posted on only when they flag something (clean, it
is carried as covered by our last review), a due follow-up waits until our last review of its PR is
`followup_min_gap_min` old (develop fixes, priorities and merge-only pushes exempt), CI updates start no gap, and the
board and the page count both."""

import datetime as dt
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import config, plan, post, queue, report, review, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, parse_ts  # noqa: E402

CFG = {
    "review_first_line": f"Automated review by rocke-monitor: Model - qwen, effort: {config.MODEL_DEFAULT}",
    "worker": {
        "model": "qwen"
    },  # the reviewer model line 1 gives (bindings.first_line)
    "rocke_path": "rk",
    "repo": "o/r",
    "operator": {"github_login": "me"},
    "features": {
        "review_automation": {"mode": "live"},
        "merge_queue": {"mode": "live"},
    },
    "settle_min": 30,
    "followup_min_gap_min": 120,  # a two-hour gap: the tests below count from it (the default is 60)
    "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    "codeowner_teams": {"platform": ["plat"]},
}
GREEN = "\U0001f7e2"


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def plus(ts: str, minutes: int) -> str:
    return (parse_ts(ts) + dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def commit(repo, files: dict, msg: str, empty: bool = False) -> str:
    for f, text in files.items():
        p = pathlib.Path(repo, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    g(repo, "add", "-A")
    g(repo, "commit", "-qm", msg, *(["--allow-empty"] if empty else []))
    return g(repo, "rev-parse", "HEAD")


class GitCase(unittest.TestCase):
    """A repo whose develop moves while a PR merges it in, one way per head:

        develop: base - d1 (other/c.txt) - d2 (other/d.txt) - d3 (rk/a.py: line 2, against the PR's)
        PR:      base - old (rk/a.py: line 2; the head we reviewed)
          m1     = old + merge d1                       one merge
          m2     = m1 + merge d2                        several merges
          m3     = old + merge d3, conflict resolved    a merge whose resolution changes the PR's own lines
          empty  = m1 + an empty commit                 a commit on none of the PR's files
          own_b  = m1 + rk/b.py changed                 a merge plus a commit on a PR file (one at the new head)
          own_a  = m1 + rk/a.py changed                 a merge plus a commit on a file the PR changed when reviewed
          plain  = old + rk/b.py changed                no merge at all
          rebased = d1 + old's change                   a rewrite: the reviewed head is gone
          retrigger = old + an empty commit             no merge, and a commit on none of the PR's files
    refs/remotes/origin/develop is d3, as the tick's fetch leaves it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        r = cls.repo = pathlib.Path(cls.tmp.name) / "repo"
        subprocess.run(["git", "init", "-q", "-b", "develop", str(r)], check=True)
        cls.base = commit(
            r, {"rk/a.py": "1\n2\n3\n", "rk/b.py": "1\n", "other/c.txt": "1\n"}, "base"
        )
        cls.d1 = commit(r, {"other/c.txt": "2\n"}, "develop: c")
        cls.d2 = commit(r, {"other/d.txt": "1\n"}, "develop: d")
        cls.d3 = commit(r, {"rk/a.py": "1\n2-dev\n3\n"}, "develop: a")
        g(r, "checkout", "-q", "--detach", cls.base)
        cls.old = commit(r, {"rk/a.py": "1\n2-pr\n3\n"}, "pr: a")
        g(r, "merge", "-q", "--no-edit", cls.d1)
        cls.m1 = g(r, "rev-parse", "HEAD")
        g(r, "merge", "-q", "--no-edit", cls.d2)
        cls.m2 = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", "--detach", cls.old)
        subprocess.run(
            [
                "git",
                "-C",
                str(r),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "merge",
                "-q",
                "--no-edit",
                cls.d3,
            ],
            capture_output=True,
        )  # conflicts in rk/a.py
        (r / "rk/a.py").write_text("1\n2-pr-and-dev\n3\n")
        g(r, "add", "rk/a.py")
        g(r, "commit", "-q", "--no-edit")
        cls.m3 = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", "--detach", cls.m1)
        cls.empty = commit(r, {}, "retrigger CI", empty=True)
        g(r, "checkout", "-q", "--detach", cls.m1)
        cls.own_b = commit(r, {"rk/b.py": "2\n"}, "pr: b")
        g(r, "checkout", "-q", "--detach", cls.m1)
        cls.own_a = commit(r, {"rk/a.py": "1\n2-pr2\n3\n"}, "pr: a again")
        g(r, "checkout", "-q", "--detach", cls.old)
        cls.plain = commit(r, {"rk/b.py": "2\n"}, "pr: b, no merge")
        g(r, "checkout", "-q", "--detach", cls.d1)
        cls.rebased = commit(r, {"rk/a.py": "1\n2-pr\n3\n"}, "pr: a, rebased")
        g(r, "checkout", "-q", "--detach", cls.old)
        cls.retrigger = commit(r, {}, "retrigger CI, no merge", empty=True)
        g(r, "update-ref", "refs/remotes/origin/develop", cls.d3)
        cls.cfg = dict(CFG, paths={"repo": str(r), "root": cls.tmp.name})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def at(self, n: int, head: str) -> None:
        """PR n's mirror ref at `head`, as the queue's fetch leaves it (no fetch in these tests)."""
        g(self.repo, "update-ref", f"refs/remotes/origin/pr{n}", head)


class TestMergeOnly(GitCase):
    def mo(self, new):
        return review.merge_only(str(self.repo), self.old, new, lazy=False)

    def test_one_merge_and_several_merges(self):
        self.assertEqual(self.mo(self.m1), {"merges": 1, "own": 0})
        self.assertEqual(self.mo(self.m2), {"merges": 2, "own": 0})

    def test_a_conflict_resolution_on_the_prs_own_lines_is_still_merge_only(self):
        self.assertEqual(self.mo(self.m3), {"merges": 1, "own": 0})
        # The classification says so too, although the PR's own lines changed.
        self.at(7, self.m3)
        c = review.classify(self.cfg, {"n": 7, "head": self.m3}, self.old)
        self.assertEqual(
            (c["kind"], c["merges"], c["changed_files"]), ("merge-only", 1, ["a.py"])
        )

    def test_a_merge_plus_a_commit_on_the_prs_files_is_not(self):
        self.assertIsNone(self.mo(self.own_b))  # a file the PR changes at the new head
        self.assertIsNone(self.mo(self.own_a))  # a file it changed at the reviewed head
        self.at(7, self.own_b)
        self.assertEqual(
            review.classify(self.cfg, {"n": 7, "head": self.own_b}, self.old)["kind"],
            "substantive",
        )

    def test_a_commit_on_none_of_the_prs_files_does_not_count(self):
        self.assertEqual(self.mo(self.empty), {"merges": 1, "own": 1})

    def test_no_merge_or_a_rewrite_is_not(self):
        self.assertIsNone(self.mo(self.plain))
        # no merge: never merge-only, even when the new commit changes none of the PR's files
        self.assertIsNone(self.mo(self.retrigger))
        self.assertIsNone(self.mo(self.rebased))
        self.assertIsNone(
            review.merge_only(str(self.repo), "f" * 40, self.m1, lazy=False)
        )  # an old head the clone lacks

    def test_a_develop_newer_than_the_mirrors_reads_as_the_prs_own(self):
        """A merge of develop commits the mirror has not fetched yet: those commits look like the PR's own, so the push
        gets its follow-up rather than a wrong carry."""
        g(self.repo, "update-ref", "refs/remotes/origin/develop", self.base)
        try:
            self.assertIsNone(self.mo(self.m3))
        finally:
            g(self.repo, "update-ref", "refs/remotes/origin/develop", self.d3)


class TestMarkMerges(GitCase):
    """The tick marks each moved head merge-only or not before the plan reads it; it carries nothing: a merge-only
    head's checks run first."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        self.store = Store(self.root / "state")
        self.st = empty_state()
        self.st["last_tick"] = ago(2)

        def pr(n, head, **kw):
            self.at(n, head)
            return dict(
                {
                    "n": n,
                    "head": head,
                    "status": "open",
                    "draft": False,
                    "size": 10,
                    "author": "a",
                    "head_seen_at": ago(300),
                    "head_committed_at": ago(300),
                    "reviews": [],
                    "labels": [],
                    "review_decision": None,
                    "ci_rollup": "SUCCESS",
                    "mergeable": "MERGEABLE",
                    "base": "develop",
                    "ours": {"head": self.old, "verdict": GREEN, "at": ago(500)},
                },
                **kw,
            )

        self.st["prs"] = {
            "7": pr(7, self.m1),
            "8": pr(8, self.own_b),
            "9": pr(9, self.m2),
            "10": pr(10, self.m2, draft=True),
            "12": pr(12, self.m3),
        }
        # #12's head is recorded as covered, but its merge_check has no merge count: no check has run on it.
        self.st["prs"]["12"]["ours"]["equivalent_heads"] = [self.m3]
        self.st["prs"]["12"]["merge_check"] = {
            "old": self.old,
            "new": self.m3,
            "merge_only": True,
        }
        # We posted on #7 ten minutes ago: a gap would hold an ordinary follow-up back.
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(10),
                    "kind": "post",
                    "pr": 7,
                    "head": self.old,
                    "ref": "u",
                    "note": "followup",
                }
            ],
        )

    def tearDown(self):
        self.d.cleanup()

    def test_marked_once_per_head_carried_never_and_not_held_by_the_gap(self):
        lines = review.mark_merges(self.cfg, self.st)
        self.assertEqual(
            lines,
            [
                f"MERGE_ONLY_REOPENED #12: {self.m3[:10]} was recorded as covered before its checks "
                "ran; they run now"
            ],
        )
        p = self.st["prs"]
        self.assertEqual(
            p["7"]["merge_check"],
            {
                "old": self.old,
                "new": self.m1,
                "merge_only": True,
                "merges": 1,
                "own": 0,
            },
        )
        self.assertEqual(
            p["8"]["merge_check"],
            {"old": self.old, "new": self.own_b, "merge_only": False},
        )
        self.assertNotIn("merge_check", p["10"])  # a draft
        self.assertEqual(
            (p["12"]["ours"]["equivalent_heads"], p["12"]["merge_check"]["merges"]),
            ([], 1),
        )
        for k in ("7", "9"):
            self.assertNotIn("equivalent_heads", p[k]["ours"])
        self.assertEqual(
            queue.gate_review(self.cfg, p["7"])[0], queue.FAIL
        )  # changed since our review: no check yet
        # The plan keeps the follow-up, says it is merge-only, and no gap holds it back; an ordinary one waits.
        last = plan.last_review_posts(self.store.ledger())
        p["8"]["ours"]["at"] = ago(
            10
        )  # #8: an ordinary follow-up, our post ten minutes old
        items = {
            i["pr"]: i
            for i in plan.make_plan(self.cfg, self.st, [], [], last)
            if i["kind"] == "FOLLOW_UP"
        }
        self.assertEqual(sorted(items), [7, 8, 9, 12])
        self.assertTrue(items[7]["merge_only"])
        self.assertNotIn("gap_until", items[7])
        self.assertIn(
            "merge-only: its checks decide, nothing is posted when they are clean",
            items[7]["text"],
        )
        self.assertIn("gap_until", items[8])
        self.assertNotIn("merge_only", items[8])
        self.assertIn(
            "P4 FOLLOW_UP #7 (merge-only)",
            report.tick_summary(self.cfg, self.st, [], list(items.values()), []),
        )
        self.assertEqual(
            review.pick_next(self.cfg, self.st, self.store, [items[8], items[7]])["pr"],
            7,
        )
        # Checked once per head: the next tick does no git work for these heads.
        with mock.patch.object(
            review, "merge_only", side_effect=AssertionError("checked again")
        ):
            self.assertEqual(review.mark_merges(self.cfg, self.st), [])

    def test_a_git_failure_is_a_line_and_is_tried_again(self):
        with mock.patch.object(
            review,
            "merge_only",
            side_effect=review.gitops.GitError("git rev-list: boom"),
        ):
            lines = review.mark_merges(self.cfg, self.st)
        self.assertEqual(
            lines[0], "(merge-only check of #7 failed: git rev-list: boom)"
        )
        self.assertNotIn("merge_check", self.st["prs"]["7"])
        self.assertEqual(review.mark_merges(self.cfg, self.st), [])
        self.assertTrue(self.st["prs"]["7"]["merge_check"]["merge_only"])

    def test_a_merge_only_review_starts_its_worker_and_checks(self):
        """`rk review-start` and the driver: no skip; the probes' results at our reviewed head are kept first."""
        chk = self.root / "workers/pr9/checks"
        chk.mkdir(parents=True)
        (chk / "head_sha").write_text(self.old + "\n")
        (chk / "checks.md").write_text(
            "## Our open findings, re-probed (probes/prN; REPRODUCES = still open)\n\n"
            "N1_x: REPRODUCES: still\nN2_y: FIXED: fixed\n\n## CI at head\n"
        )
        started = []
        cfg = dict(
            self.cfg, paths=dict(self.cfg["paths"], workers=str(self.root / "workers"))
        )
        with mock.patch.object(
            review.gh, "rest", return_value={"state": "open", "head": self.m2}
        ), mock.patch.object(
            review, "_changed_files", return_value=[]
        ), mock.patch.object(
            review.jobs,
            "start",
            lambda store, kind, name, script, timeout_min: started.append(kind)
            or {"id": f"{kind}-{name}", "started": ago(0)},
        ):
            out = review.start_review(cfg, self.st, self.store, 9, "follow-up")
        self.assertEqual(
            out,
            [
                "#9 is merge-only since our review (2 merges): nothing is posted if its checks come back "
                "clean",
                f"REVIEW_START #9 (follow-up, merge-only): reviewer model worker started at "
                f"{self.m2[:10]} (parallel review 1 of 3)",
            ],
        )
        self.assertEqual(started, ["worker"])
        self.assertEqual(
            review.pipe_of(self.st, 9)["merge_only"],
            {
                "old": self.old,
                "merges": 2,
                "probes": {"N1_x": "REPRODUCES", "N2_y": "FIXED"},
                "peer": {
                    "threads": None,
                    "title": None,
                    "tests": None,
                    "installed": None,
                    "prnumbers": None,
                },
            },
        )  # the checks there wrote no item lists
        self.assertNotIn("equivalent_heads", self.st["prs"]["9"]["ours"])


class TestMergeOnlyVerdict(unittest.TestCase):
    """A merge-only head's finished review: clean checks, and nothing is posted (the head is carried as covered); a
    failing or unrun check, an ABI note, a probe that changed state or a new CI failure sends it to verification, saying
    why."""

    STATUS = (
        "merge pass clean\nbuild pass 100% tests passed out of 57\nbyteid pass touched families\n"
        "golden pass PASS\ndg pass GOLDEN OK\ntests skip no test to run\ninstalled pass 4 run of 4\n"
        "compile pass spills 0\n"
    )
    OLD, NEW = "a" * 40, "b" * 40

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.chk = pathlib.Path(self.d.name) / "workers/pr7/checks"
        self.chk.mkdir(parents=True)
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.ci, self.infra, self.infra_calls = {}, None, []
        self.cfg = dict(CFG, paths={"working_dir": self.d.name})  # workers/ under it
        for p in (
            mock.patch.object(
                review.known_infra,
                "classify",
                lambda cfg, st, n, head, names: self.infra_calls.append(names)
                or self.infra,
            ),
            mock.patch.object(
                review.jobs, "status", lambda store, job: "done" if job else "none"
            ),
            mock.patch("rkmon_lib.queue.rocke_tree", lambda cfg, sha: "t" * 40),
            mock.patch.object(
                review, "ci_failing", lambda cfg, sha: self.ci.get(sha, set())
            ),
            mock.patch.object(review, "has_draft", lambda *a, **kw: True),
            # The reviews and comments since our last review (activity_since_ours): none read unless a test says.
            mock.patch.object(
                review.gh, "rest", side_effect=review.gh.GhError("no GitHub here")
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def run_review(
        self,
        status=None,
        abi="no struct layout change in include/rocke/ without a bump",
        probes="N1_x: REPRODUCES: still\nN2_y: FIXED: fixed",
        head=None,
        peer=None,
    ) -> list[str]:
        (self.chk / "status").write_text(self.STATUS if status is None else status)
        (self.chk / "head_sha").write_text((head or self.NEW) + "\n")
        (self.chk / "develop_sha").write_text("d" * 40 + "\n")
        (self.chk / "checks.md").write_text(
            "# Checks for #7\n\n## Merge with develop\n\nmerge ok\n\n"
            f"## Our open findings, re-probed (probes/prN; REPRODUCES = still open)\n\n{probes}\n\n"
            "## CI at head\n\nsuccess 3\n\n## ABI (abi.h counters, C struct layouts in include/rocke/)\n\n"
            f"ROCKE_ABI_VERSION: develop 10; no bump\n{abi}\n\n## rule_scan (tracker IDs, perf numbers)\n\nnone\n"
        )
        self.st = empty_state()
        self.st["prs"] = {
            "7": {
                "n": 7,
                "head": self.NEW,
                "status": "open",
                "draft": False,
                "size": 10,
                "ours": {"head": self.OLD, "verdict": GREEN, "at": ago(300)},
            }
        }
        self.st["review_pipes"] = [
            {
                "pr": 7,
                "kind": "follow-up",
                "head": self.NEW,
                "since": ago(300),
                "stage": "worker",
                "worker": {"id": "w", "started": ago(60)},
                "checks": {"id": "c", "started": ago(50)},
                "merge_only": {
                    "old": self.OLD,
                    "merges": 1,
                    "probes": {"N1_x": "REPRODUCES", "N2_y": "FIXED"},
                    **({"peer": peer} if peer is not None else {}),
                },
            }
        ]
        return review.advance(self.cfg, self.st, self.store, [])

    def test_clean_checks_post_nothing_and_carry_the_head(self):
        self.assertEqual(
            self.run_review(),
            [
                f"REVIEW_SKIP #7: merge-only since our review (1 merge; its checks are "
                f"clean, so nothing is posted); recorded {self.NEW[:10]} as equivalent"
            ],
        )
        rec = self.st["prs"]["7"]
        self.assertEqual(rec["ours"]["equivalent_heads"], [self.NEW])
        self.assertEqual(self.st["review_pipes"], [])
        self.assertEqual(self.st.get("review_ready") or [], [])
        self.assertEqual(
            (rec["gates"]["head"], rec["probes_seen"]),
            (
                self.NEW,
                {"head": self.NEW, "results": {"N1_x": "REPRODUCES", "N2_y": "FIXED"}},
            ),
        )
        self.assertEqual(queue.gate_review(CFG, rec)[0], queue.PASS)
        self.assertEqual(self.store.ledger(), [])  # nothing posted
        self.assertEqual(
            list(report.equivalent_clears(self.store).values())[0][1], "merge-only"
        )
        # A new probe that reproduces (saved for an open finding) and a CI failure our reviewed head had too are clean.
        self.ci = {self.OLD: {"Linux shard 1"}, self.NEW: {"Linux shard 1"}}
        self.assertTrue(
            self.run_review(
                probes="N1_x: REPRODUCES: s\nN2_y: FIXED: f\nN9_new: REPRODUCES: s"
            )[0].startswith("REVIEW_SKIP #7")
        )
        # A pass with skips CI's lanes share (no GPU here) is clean: the skips are listed in the section.
        skipped = self.STATUS.replace(
            "tests skip no test to run\n",
            "tests pass (2 skipped: 2 needs a gfx950 GPU) " "10 passed, 2 skipped\n",
        )
        self.assertTrue(self.run_review(status=skipped)[0].startswith("REVIEW_SKIP #7"))

    def test_a_review_left_since_our_last_review_makes_a_micro_round(self):
        """Clean checks, but a reviewer re-reviewed since our last review: the head is not carried silently; the
        verifier alone answers it (a micro round). Only our own login's items and bots' (the filter GitHub is asked
        with), or a GitHub that cannot be read, leave the carry as it was."""
        asked = []
        new = {
            "reviews": [ago(10)],
            "pulls/7/comments": [],
            "issues/7/comments": [ago(400)],
        }

        def rest(path, jq=None, paginate=False, timeout=120):
            asked.append(jq)
            return next(v for k, v in new.items() if k in path)

        with mock.patch.object(review.gh, "rest", rest):
            lines = self.run_review()
        self.assertTrue(lines[-1].startswith("REVIEW_READY #7 (follow-up)"), lines)
        self.assertIn("micro: since our last review", lines[-1])
        self.assertIn(
            "1 review, 0 review comments and 0 comments from others: answer them",
            lines[-1],
        )
        self.assertNotIn("equivalent_heads", self.st["prs"]["7"]["ours"])
        self.assertTrue(
            all(
                'select(.user.type != "Bot")' in q and "select(.user.login != " in q
                for q in asked
            ),
            asked,
        )
        new["reviews"] = [ago(400)]  # before our last review: nothing new
        with mock.patch.object(review.gh, "rest", rest):
            self.assertTrue(self.run_review()[0].startswith("REVIEW_SKIP #7"))
        with mock.patch.object(
            review.gh, "rest", side_effect=review.gh.GhError("HTTP 502")
        ):
            self.assertTrue(self.run_review()[0].startswith("REVIEW_SKIP #7"))

    def test_each_flag_sends_it_to_verification_saying_why(self):
        note = "portable_ir/src/abi.py BINARY_ABI = 10 but abi.h ROCKE_ABI_VERSION = 11: bump both together"
        build = "build pass 100% tests passed out of 57\n"
        cases = [
            (
                "build fails (native build failed)",
                {
                    "status": self.STATUS.replace(
                        build, "build fail native build failed\n"
                    )
                },
            ),
            (
                "build did not run (low memory)",
                {"status": self.STATUS.replace(build, "build skip low memory\n")},
            ),
            # A test a missing tool kept from running (cmake, PyYAML: CI's lanes have them) is never a pass.
            (
                "tests did not run (NOT RUN: cmake missing (2 tests); 10 passed, 2 skipped)",
                {
                    "status": self.STATUS.replace(
                        "tests skip no test to run\n",
                        "tests skip NOT RUN: cmake missing (2 tests); "
                        "10 passed, 2 skipped\n",
                    )
                },
            ),
            (
                "installed did not run (NOT RUN: PyYAML missing (1 test); 4 run of 4",
                {
                    "status": self.STATUS.replace(
                        "installed pass 4 run of 4\n",
                        "installed skip NOT RUN: PyYAML missing (1 "
                        "test); 4 run of 4 changed test files, 9 s: 40 passed, 1 skipped\n",
                    )
                },
            ),
            (
                "no merge with develop was checked",
                {"status": self.STATUS.replace("merge pass clean\n", "")},
            ),
            ("the checks wrote no status", {"status": ""}),
            (f"the checks ran at cccccccccc, not {self.NEW[:10]}", {"head": "c" * 40}),
            (
                f"ABI: {note}",
                {"abi": f"NOTE (G9 waits for `rk abi-ok N` after review): {note}"},
            ),
            (
                "probe N2_y: FIXED -> REPRODUCES",
                {"probes": "N1_x: REPRODUCES: s\nN2_y: REPRODUCES: back"},
            ),
            (
                "probe N1_x: REPRODUCES -> ERROR",
                {"probes": "N1_x: ERROR rc=1 (crashed): x\nN2_y: FIXED: f"},
            ),
            (
                "probe N3_z: no earlier result -> FIXED",
                {"probes": "N1_x: REPRODUCES: s\nN2_y: FIXED: f\nN3_z: FIXED: f"},
            ),
            # The peer checks are flags like the others (rkmon_lib/peer_checks.py).
            (
                "trackers fails (1 tracker ID in added lines: TKEY" "-2967)",
                {
                    "status": self.STATUS
                    + "trackers fail 1 tracker ID in added lines: TKEY"
                    "-2967\n"
                },
            ),
            (
                "threads fails (2 live threads with no owner reply: copilot-pull-request-reviewer 2)",
                {
                    "status": self.STATUS
                    + "threads fail 2 live threads with no owner reply: copilot-pull-request-reviewer 2\n"
                },
            ),
            (
                "threads did not run (NOT RUN: GitHub not read",
                {
                    "status": self.STATUS
                    + "threads skip NOT RUN: GitHub not read: timeout\n"
                },
            ),
            (
                "title fails (not a Conventional Commits title: it starts with a space)",
                {
                    "status": self.STATUS
                    + "title fail not a Conventional Commits title: it starts with a space\n"
                },
            ),
            (
                "testlists fails",
                {
                    "status": self.STATUS
                    + "testlists fail 1 new library test in no CMake or CTest list\n"
                },
            ),
            (
                "envdocs fails",
                {
                    "status": self.STATUS
                    + "envdocs fail 1 ROCKE_* variable read and not in env_flags.md\n"
                },
            ),
            (
                "cppenv fails",
                {
                    "status": self.STATUS
                    + "cppenv fail 1 getenv call in added C/C++ lines: a.cpp:3\n"
                },
            ),
            (
                "abi did not run (NOT RUN: x)",
                {"status": self.STATUS + "abi skip NOT RUN: x\n"},
            ),
        ]
        for why, kw in cases:
            with self.subTest(why=why):
                lines = self.run_review(**kw)
                self.assertEqual(len(lines), 1, lines)
                self.assertTrue(
                    lines[0].startswith(
                        "REVIEW_READY #7 (follow-up): worker done, checks done. Merge-only "
                        "since our review, but the checks flag: "
                    ),
                    lines[0],
                )
                self.assertIn(why, lines[0])
                self.assertIn(". Inputs: ", lines[0])
                self.assertNotIn("equivalent_heads", self.st["prs"]["7"]["ours"])
                self.assertEqual([r["pr"] for r in self.st["review_ready"]], [7])
                self.assertEqual(self.st["review_pipes"], [])

    def test_the_lead_sections_flag_nothing(self):
        """Their findings are leads for the reviewers: another PR's change as much as this one's, survivors a
        merge-only push did not change, and the head-against-merge-base leads (CTest tiers, moved-file references,
        the engine import probe) the verifier confirms."""
        status = (
            self.STATUS
            + "clash fail 1 PR clash: #9 (1 silent, 0 modify/delete, 0 conflicts)\n"
            "mutants fail 3 of 9 mutants survive (checks no test catches): m.py:4\n"
            "labels fail 1 lead: t loses quick in the build tree\n"
            "movedrefs fail 1 reference names moved or deleted files (a.md:3)\n"
            'imports fail 1 lead: b.f needs rocke_library_engine with backend="python"\n'
            "peerleads fail 2 leads: coercion 1, torch device query 1\n"
        )
        self.assertTrue(self.run_review(status=status)[0].startswith("REVIEW_SKIP #7"))
        skips = "".join(
            f"{k} skip no result\n"
            for k in ("clash", "mutants", "labels", "movedrefs", "imports", "peerleads")
        )
        self.assertTrue(
            self.run_review(status=self.STATUS + skips)[0].startswith("REVIEW_SKIP #7")
        )

    def test_an_abi_note_flags_once(self):
        """The ABI section's status fails on its notes; the flag lists each note once, not the status as well."""
        note = "generated-header struct layouts change without a binary bump: rocke_coalesced_tile_loader_t"
        lines = self.run_review(
            status=self.STATUS + f"abi fail 1 note: {note}\n",
            abi=f"NOTE (G9 waits for `rk abi-ok N` after review): {note}\nABI fail 1 note: {note}",
        )
        self.assertIn(f"ABI: {note}", lines[0])
        self.assertNotIn("abi fails", lines[0])
        self.assertTrue(
            self.run_review(
                status=self.STATUS + "abi pass no ABI change without a bump\n"
            )[0].startswith("REVIEW_SKIP #7")
        )

    def test_a_new_ci_failure_or_an_unread_ci_flags_it(self):
        self.ci = {
            self.NEW: {"Linux shard 1", "Windows shard 2"},
            self.OLD: {"Windows shard 2"},
        }
        self.assertIn(
            "CI fails at the head and not at our reviewed head: Linux shard 1",
            self.run_review()[0],
        )
        with mock.patch.object(
            review, "ci_failing", side_effect=review.gh.GhError("HTTP 502")
        ):
            self.assertIn(
                "the checks flag: CI not read (HTTP 502)", self.run_review()[0]
            )

    def test_a_known_infrastructure_ci_failure_is_carried_with_a_line(self):
        """The only flag is a new CI failure, and every failing job is known infrastructure: nothing is posted, the
        head is carried as covered, and CI_INFRA names the PR, each job and why."""
        self.ci = {self.NEW: {"Windows shard 1"}, self.OLD: set()}
        self.infra = {"Windows shard 1": "signature `pwsh: command not found`"}
        lines = self.run_review()
        self.assertEqual(
            lines,
            [
                f"REVIEW_SKIP #7: merge-only since our review (1 merge; its only flag is CI failures that are known "
                f"infrastructure, so nothing is posted); recorded {self.NEW[:10]} as equivalent",
                f"CI_INFRA #7 at {self.NEW[:10]}: merge-only, and its only flag is known CI infrastructure: "
                "Windows shard 1 (signature `pwsh: command not found`)",
            ],
        )
        self.assertEqual(self.infra_calls, [["Windows shard 1"]])
        self.assertEqual(self.st["prs"]["7"]["ours"]["equivalent_heads"], [self.NEW])
        self.assertEqual(self.st.get("review_ready") or [], [])
        self.assertEqual(self.store.ledger(), [])
        self.assertIn("CI_INFRA #7", self.store.journal_path.read_text())

    def test_known_infrastructure_carries_nothing_beside_another_flag(self):
        """Another flag beside the CI one, or a job that is not known infrastructure: verification as before."""
        self.ci = {self.NEW: {"Windows shard 1"}, self.OLD: set()}
        self.infra = {"Windows shard 1": "signature `pwsh: command not found`"}
        lines = self.run_review(probes="N1_x: REPRODUCES: s\nN2_y: REPRODUCES: back")
        self.assertTrue(lines[0].startswith("REVIEW_READY #7"), lines)
        self.assertEqual(
            self.infra_calls, []
        )  # not asked: the head goes to verification anyway
        self.infra = None
        lines = self.run_review()
        self.assertTrue(lines[0].startswith("REVIEW_READY #7"), lines)
        self.assertIn(
            "CI fails at the head and not at our reviewed head: Windows shard 1",
            lines[0],
        )
        self.assertNotIn("equivalent_heads", self.st["prs"]["7"]["ours"])

    def test_the_probe_baseline(self):
        rec = {
            "n": 7,
            "head": self.NEW,
            "ours": {"head": self.OLD, "equivalent_heads": ["e" * 40]},
        }
        self.assertIsNone(review.probe_baseline(self.cfg, rec, 7))  # nothing known
        (self.chk / "head_sha").write_text(self.OLD + "\n")
        (self.chk / "checks.md").write_text(
            "## Our open findings, re-probed (x)\n\nno saved probe\n"
        )
        self.assertEqual(
            review.probe_baseline(self.cfg, rec, 7), {}
        )  # our reviewed head's checks had none
        (self.chk / "head_sha").write_text(
            "f" * 40 + "\n"
        )  # checks of a head our review does not cover
        self.assertIsNone(review.probe_baseline(self.cfg, rec, 7))
        rec["probes_seen"] = {
            "head": "e" * 40,
            "results": {"N1_x": "FIXED"},
        }  # the last carried head's
        self.assertEqual(review.probe_baseline(self.cfg, rec, 7), {"N1_x": "FIXED"})


class TestManualClose(GitCase):
    """`rk classify N --record` on a merge-only head records it and closes a review of it without a post: a draft
    awaiting verification, or a review in flight."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.cfg2 = dict(self.cfg, paths=dict(self.cfg["paths"], daily_dir=self.d.name))
        self.at(7, self.m3)
        self.st = empty_state()
        self.st["prs"] = {
            "7": {
                "n": 7,
                "head": self.m3,
                "status": "open",
                "draft": False,
                "size": 10,
                "ours": {"head": self.old, "verdict": "\U0001f7e1", "at": ago(300)},
            }
        }
        self.st["review_ready"] = [
            {"pr": 7, "kind": "follow-up", "head": self.m3, "at": ago(20)},
            {"pr": 8, "kind": "first-round", "head": "8" * 40, "at": ago(10)},
        ]
        self.store.save(self.st)

    def tearDown(self):
        self.d.cleanup()

    def cli(self, *argv):
        import rkmon

        out = []
        live = getattr(self, "live", None)
        head = (
            (lambda cfg, n: live)
            if not isinstance(live, Exception) and live
            else (lambda cfg, n: self.store.load()["prs"][str(n)]["head"])
        )
        if isinstance(live, Exception):

            def head(cfg, n):
                raise live

        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg2
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            review, "live_head", head
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(list(argv))
        return code, out

    def test_classify_record_closes_the_draft_without_a_post(self):
        code, out = self.cli("classify", "7")  # without --record nothing changes
        self.assertEqual((code, json.loads(out[0])["kind"]), (0, "merge-only"))
        self.assertEqual([r["pr"] for r in self.store.load()["review_ready"]], [7, 8])
        code, out = self.cli("classify", "7", "--record")
        self.assertEqual(code, 0)
        self.assertEqual(
            out[1:],
            [
                "recorded as equivalent: our review stands for the new head",
                f"#7: the draft awaiting verification at {self.m3[:10]} is closed without a post "
                "(merge-only since our review)",
            ],
        )
        st = self.store.load()
        self.assertEqual(st["prs"]["7"]["ours"]["equivalent_heads"], [self.m3])
        self.assertEqual([r["pr"] for r in st["review_ready"]], [8])
        journal = self.store.journal_path.read_text()
        self.assertIn(
            f"#7 merge-only since our review: {self.m3[:10]} recorded as equivalent (manual classify)",
            journal,
        )
        self.assertIn(
            f"#7 draft closed without a post at {self.m3[:10]}: merge-only since our review",
            journal,
        )
        self.assertEqual(self.store.ledger(), [])  # nothing posted
        self.assertEqual(
            queue.gate_review(self.cfg2, st["prs"]["7"])[0], queue.PASS
        )  # current for the approval lists
        self.assertEqual(
            list(report.equivalent_clears(self.store).values())[0][1], "merge-only"
        )
        self.cli("classify", "7", "--record")  # again: nothing more to record or close
        self.assertEqual(
            self.store.load()["prs"]["7"]["ours"]["equivalent_heads"], [self.m3]
        )

    def test_record_refuses_a_live_head_the_tick_has_not_seen(self):
        """A push after the last tick: --record would cover the state's older head, so it refuses (L-594)."""
        self.live = "f" * 40
        code, out = self.cli("classify", "7", "--record")
        self.assertEqual(code, 4)
        self.assertEqual(
            out,
            [
                f"REFUSED: #7's live head is ffffffffff, but the state's head is {self.m3[:10]}: the "
                "tick has not seen ffffffffff yet. Run `rk tick`, then `rk classify 7 --record` "
                "again; nothing recorded"
            ],
        )
        st = self.store.load()
        self.assertNotIn("equivalent_heads", st["prs"]["7"]["ours"])
        self.assertEqual([r["pr"] for r in st["review_ready"]], [7, 8])
        self.live = review.gh.GhError("HTTP 502")
        code, out = self.cli("classify", "7", "--record")
        self.assertEqual(
            (code, out),
            (4, ["REFUSED: #7's live head was not read (HTTP 502); nothing recorded"]),
        )
        self.live = None
        code, out = self.cli(
            "classify", "7", "--record"
        )  # the live head is the one the tick saw
        self.assertEqual(
            (code, out[1]),
            (0, "recorded as equivalent: our review stands for the new head"),
        )

    def test_a_review_in_flight_at_that_head_is_stopped(self):
        killed = []
        self.st["review_ready"] = []
        self.st["review_pipes"] = [
            {
                "pr": 7,
                "head": self.m3,
                "kind": "follow-up",
                "stage": "worker",
                "worker": {"id": "w"},
                "checks": {"id": "c"},
            }
        ]
        self.st["jobs"] = {"heavy": {"id": "c"}}
        with mock.patch.object(
            review.jobs, "status", lambda store, job: "running"
        ), mock.patch.object(
            review.jobs, "kill", lambda store, job: killed.append(job["id"])
        ):
            lines = review.close_review(
                self.st, self.store, 7, self.m3, "merge-only since our review"
            )
        self.assertEqual(
            lines,
            [
                f"#7: the review in flight at {self.m3[:10]} is stopped and closed without a post "
                "(merge-only since our review)"
            ],
        )
        self.assertEqual(killed, ["w", "c"])
        self.assertEqual(self.st["review_pipes"], [])
        self.assertIsNone(self.st["jobs"]["heavy"])
        self.st["review_pipes"] = [
            {"pr": 7, "head": self.m1}
        ]  # a review of another head is left alone
        self.assertEqual(review.close_review(self.st, self.store, 7, self.m3, "x"), [])
        self.assertEqual(self.st["review_pipes"], [{"pr": 7, "head": self.m1}])

    def test_a_substantive_head_is_not_recorded_and_its_draft_stays(self):
        self.at(7, self.own_b)
        self.st["prs"]["7"]["head"] = self.own_b
        self.st["review_ready"][0]["head"] = self.own_b
        self.store.save(self.st)
        code, out = self.cli("classify", "7", "--record")
        self.assertEqual(
            (code, json.loads(out[0])["kind"], len(out)), (0, "substantive", 1)
        )
        st = self.store.load()
        self.assertNotIn("equivalent_heads", st["prs"]["7"]["ours"])
        self.assertEqual([r["pr"] for r in st["review_ready"]], [7, 8])


class TestGap(unittest.TestCase):
    """At most one follow-up review post per PR every `followup_min_gap_min` minutes."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.st = empty_state()
        self.st["last_tick"] = ago(2)
        common = {
            "status": "open",
            "draft": False,
            "size": 10,
            "author": "a",
            "head_seen_at": ago(300),
            "head_committed_at": ago(300),
            "reviews": [],
            "review_decision": None,
        }
        self.st["prs"] = {
            "7": dict(
                common,
                n=7,
                head="b" * 40,
                ours={"head": "a" * 40, "verdict": GREEN, "at": ago(400)},
            ),
            "8": dict(common, n=8, head="8" * 40, ours=None),
        }
        self.posted = ago(30)  # our follow-up on #7, half an hour ago
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": self.posted,
                    "kind": "post",
                    "pr": 7,
                    "head": "a" * 40,
                    "ref": "u",
                    "note": "followup",
                }
            ],
        )
        self.plan = [
            {"prio": 1, "kind": "FOLLOW_UP", "pr": 7, "text": "follow-up on #7"},
            {
                "prio": 3,
                "kind": "FIRST_ROUND",
                "pr": 8,
                "text": "first-round review of #8",
            },
        ]

    def tearDown(self):
        self.d.cleanup()

    def last(self):
        return plan.last_review_posts(self.store.ledger())

    def test_the_wait_and_its_release_time(self):
        until = plan.gap_until(CFG, self.st, self.st["prs"]["7"], self.last())
        self.assertEqual(until, plus(self.posted, 120))
        # The plan says until when, on its line in the tick summary too.
        items = plan.make_plan(CFG, self.st, [], [], self.last())
        fu = next(i for i in items if i["pr"] == 7)
        self.assertEqual(fu["gap_until"], until)
        self.assertTrue(
            fu["text"].endswith(f"(gap: after {until[11:16]}Z)"), fu["text"]
        )
        self.assertIn(
            f"P4 FOLLOW_UP #7 (gap: after {until[11:16]}Z)",
            report.tick_summary(CFG, self.st, [], items, []),
        )
        self.assertNotIn("gap_until", next(i for i in items if i["pr"] == 8))
        # The driver does not start it early: the first round behind it goes first.
        self.assertEqual(review.pick_next(CFG, self.st, self.store, self.plan)["pr"], 8)
        started = []
        with mock.patch.object(
            review,
            "start_review",
            lambda cfg, st, store, n, kind, again=False: started.append(n)
            or [f"REVIEW_START #{n} ({kind})"],
        ):
            self.assertEqual(
                review.advance(CFG, self.st, self.store, self.plan),
                ["REVIEW_START #8 (first-round)"],
            )
        self.assertEqual(started, [8])
        self.st["review_pipeline"] = None
        self.st["review_ready"] = [
            {"pr": 8, "head": "8" * 40}
        ]  # #8 is drafted: nothing else can start now
        self.assertIsNone(review.pick_next(CFG, self.st, self.store, self.plan))
        self.assertEqual(
            review.advance(CFG, self.st, self.store, self.plan), []
        )  # the driver exits: nothing to start
        # Released 120 minutes after the post.
        self.store.ledger_path.write_text("")
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(121),
                    "kind": "post",
                    "pr": 7,
                    "head": "a" * 40,
                    "ref": "u",
                    "note": "followup",
                }
            ],
        )
        self.assertIsNone(
            plan.gap_until(CFG, self.st, self.st["prs"]["7"], self.last())
        )
        self.assertEqual(
            review.pick_next(CFG, self.st, self.store, self.plan),
            {"pr": 7, "kind": "follow-up", "again": False},
        )
        # The gap is a setting: 0 turns it off, 180 waits longer.
        self.store.ledger_path.write_text("")
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": self.posted,
                    "kind": "post",
                    "pr": 7,
                    "head": "a" * 40,
                    "ref": "u",
                    "note": "followup",
                }
            ],
        )
        self.assertIsNone(
            plan.gap_until(
                dict(CFG, followup_min_gap_min=0),
                self.st,
                self.st["prs"]["7"],
                self.last(),
            )
        )
        self.assertEqual(
            plan.gap_until(
                dict(CFG, followup_min_gap_min=180),
                self.st,
                self.st["prs"]["7"],
                self.last(),
            ),
            plus(self.posted, 180),
        )
        self.assertEqual(
            (plan.gap_label(CFG), plan.gap_label({"followup_min_gap_min": 90})),
            ("2-h", "90-min"),
        )

    def test_develop_fixes_and_priorities_are_exempt_and_the_list_is_a_setting(self):
        rec = self.st["prs"]["7"]
        for why in (
            ["red-develop #12345"],
            ["silent-break #1111", "your priority: x"],
            ["your priority: x"],
        ):
            self.st["develop_fixes"] = {"7": why}
            self.assertIsNone(plan.gap_until(CFG, self.st, rec, self.last()), why)
            self.assertEqual(
                review.pick_next(CFG, self.st, self.store, self.plan)["pr"], 7
            )
        self.st["develop_fixes"] = {
            "7": ["your priority: x"]
        }  # priorities wait once the list leaves them out
        self.assertIsNotNone(
            plan.gap_until(
                dict(CFG, followup_gap_exempt=["develop_fix"]),
                self.st,
                rec,
                self.last(),
            )
        )
        self.st["develop_fixes"] = {"7": ["red-develop #12345"]}
        self.assertIsNotNone(
            plan.gap_until(dict(CFG, followup_gap_exempt=[]), self.st, rec, self.last())
        )
        self.st["develop_fixes"] = {}
        for exempt in ([7], ["7"], ["#7"]):  # a PR by number
            self.assertIsNone(
                plan.gap_until(
                    dict(CFG, followup_gap_exempt=exempt), self.st, rec, self.last()
                ),
                exempt,
            )
        self.assertIsNotNone(
            plan.gap_until(
                dict(CFG, followup_gap_exempt=[8]), self.st, rec, self.last()
            )
        )
        # A merge-only push never waits: its checks run at once, and one that flags something may be landing.
        rec["merge_check"] = {
            "old": "a" * 40,
            "new": "b" * 40,
            "merge_only": True,
            "merges": 1,
        }
        self.assertIsNone(
            plan.gap_until(dict(CFG, followup_gap_exempt=[]), self.st, rec, self.last())
        )
        rec["merge_check"]["new"] = (
            "c" * 40
        )  # a finding for another head says nothing about this one
        self.assertIsNotNone(plan.gap_until(CFG, self.st, rec, self.last()))

    def test_ci_updates_and_notices_start_no_gap(self):
        self.store.ledger_path.write_text("")
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(400),
                    "kind": "post",
                    "pr": 7,
                    "head": "1" * 40,
                    "ref": "u",
                    "note": "first_round",
                },
                {
                    "ts": ago(300),
                    "kind": "post",
                    "pr": 7,
                    "head": "a" * 40,
                    "ref": "u",
                    "note": "followup",
                },
                {
                    "ts": ago(20),
                    "kind": "post",
                    "pr": 7,
                    "head": "a" * 40,
                    "ref": "u",
                    "note": "followup",
                },  # same head
                {
                    "ts": ago(15),
                    "kind": "post",
                    "pr": 7,
                    "head": "c" * 40,
                    "ref": "u",
                    "note": "ci-update",
                },
                {
                    "ts": ago(10),
                    "kind": "notice-queue",
                    "pr": 7,
                    "head": "k",
                    "ref": "u",
                    "note": "",
                },
                {
                    "ts": ago(5),
                    "kind": "post",
                    "pr": 9,
                    "head": "9" * 40,
                    "ref": "u",
                    "note": "ci-update",
                },
            ],
        )
        self.assertEqual(self.last(), {7: self.store.ledger()[1]["ts"], 9: None})
        self.assertIsNone(
            plan.gap_until(CFG, self.st, self.st["prs"]["7"], self.last())
        )
        # A PR the ledger never saw counts from our review's own time; one with only CI updates does not.
        rec = dict(
            self.st["prs"]["7"], n=12, ours={"head": "a" * 40, "at": self.posted}
        )
        self.assertEqual(
            plan.gap_until(CFG, self.st, rec, self.last()), plus(self.posted, 120)
        )
        self.assertIsNone(plan.gap_until(CFG, self.st, dict(rec, n=9), self.last()))

    def test_the_post_notes_a_ci_update(self):
        """A post at a head our review covers, with no review of ours in flight on the PR, is a CI update."""
        st = {"prs": {"7": {"head": "b" * 40, "ours": {"head": "b" * 40}}}}
        self.assertEqual(post.post_note(st, 7), "ci-update")
        st["prs"]["7"]["ours"] = {
            "head": "a" * 40,
            "equivalent_heads": ["b" * 40],
        }  # a carried head
        self.assertEqual(post.post_note(st, 7), "ci-update")
        st["review_ready"] = [
            {"pr": 7, "head": "b" * 40}
        ]  # the review of that head, verified
        self.assertEqual(post.post_note(st, 7), "followup")
        st = {
            "prs": {"7": {"head": "b" * 40, "ours": {"head": "a" * 40}}},
            "review_pipeline": None,
        }
        self.assertEqual(post.post_note(st, 7), "followup")
        self.assertEqual(
            post.post_note({"prs": {"7": {"head": "b" * 40, "ours": None}}}, 7),
            "first_round",
        )
        # End to end: `rk post` ledgers it so, and the gap ignores it.
        d = pathlib.Path(self.d.name)
        cfg = dict(CFG, paths={"daily_dir": str(d / "daily"), "posted_dir": str(d)})
        st = {
            "prs": {
                "7": {
                    "n": 7,
                    "head": "b" * 40,
                    "ours": {"head": "b" * 40, "verdict": GREEN},
                }
            }
        }
        body = d / "p.md"
        body.write_text(
            f"{CFG['review_first_line']}\n\n## PR #7 — t — 🔴 Follow-up: CI fails — x\n",
            encoding="utf-8",
        )
        done = subprocess.CompletedProcess(
            [], 0, stdout="https://github.com/ROCm/r/pull/7#issuecomment-1\n", stderr=""
        )
        self.store.ledger_path.write_text("")
        with mock.patch("rkmon_lib.gh.rest", return_value="b" * 40), mock.patch.object(
            post.gh_cli, "write", return_value=done
        ):
            post.post(cfg, st, self.store, 7, str(body))
        self.assertEqual(self.store.ledger()[-1]["note"], "ci-update")
        self.assertEqual(self.last(), {7: None})

    def test_review_start_by_hand_waits_too_unless_now(self):
        import rkmon

        out, started = [], []
        self.store.save(self.st)

        def start(cfg, st, store, n, kind, force=False, again=False):
            started.append((n, kind))
            return [f"REVIEW_START #{n} ({kind}): qwen worker started"]

        with mock.patch.object(rkmon, "load_config", lambda: CFG), mock.patch.object(
            rkmon, "Store", lambda: self.store
        ), mock.patch.object(review, "start_review", start), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["review-start", "7"]), 4)
            until = plus(self.posted, 120)
            self.assertEqual(
                out[-1],
                f"REFUSED: #7's follow-up waits for the 2-h gap after our last review of it; "
                f"it may start after {until[11:16]}Z. Start it then, or now anyway with --now",
            )
            self.assertEqual(started, [])
            self.assertEqual(rkmon.main(["review-start", "7", "--now"]), 0)
            self.assertEqual(
                rkmon.main(["review-start", "8"]), 0
            )  # a first round has no gap
        self.assertEqual(started, [(7, "follow-up"), (8, "first-round")])

    def test_an_oversized_follow_up_inside_the_gap_is_not_named(self):
        cfg = dict(CFG, autopilot_max_lines=30000)
        self.st["prs"]["7"]["size"] = 40000
        self.assertEqual(review.oversized(cfg, self.st, self.plan, self.store), [])
        self.assertEqual(review.manual_lines(cfg, self.st, self.store, self.plan), [])
        self.store.ledger_path.write_text("")
        self.assertEqual(
            [b["pr"] for b in review.oversized(cfg, self.st, self.plan, self.store)],
            [7],
        )


class TestBoardAndPage(unittest.TestCase):
    """The board's Reviews row and the page count the follow-ups waiting for the gap and the merge-only carries."""

    def test_counts(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            cfg = dict(
                CFG, paths={"html": str(pathlib.Path(d) / "x.html"), "daily_dir": d}
            )
            st = empty_state()
            st["enabled"] = True
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {
                str(n): {
                    "n": n,
                    "head": str(n) * 40,
                    "status": "open",
                    "draft": False,
                    "head_seen_at": ago(60),
                    "title": f"t{n}",
                    "author": "a",
                }
                for n in (1, 2, 3)
            }
            soon = plus(ago(0), 45)
            items = [
                {
                    "prio": 1,
                    "kind": "FOLLOW_UP",
                    "pr": 1,
                    "text": "",
                    "critical": False,
                    "gap_until": soon,
                },
                {
                    "prio": 2,
                    "kind": "FOLLOW_UP",
                    "pr": 2,
                    "text": "",
                    "critical": False,
                    "gap_until": ago(1),
                },
                {
                    "prio": 3,
                    "kind": "FIRST_ROUND",
                    "pr": 3,
                    "text": "",
                    "critical": False,
                },
            ]
            store.journal(
                "review",
                "#5 follow-up skipped: merge-only (aaaaaaaa -> bbbbbbbb); our review stands",
            )
            store.journal(
                "review",
                "#6 follow-up skipped: rebase-only (cccccccc -> dddddddd); our review stands",
            )
            store.journal(
                "review",
                "#5 follow-up skipped: merge-only (aaaaaaaa -> bbbbbbbb); our review stands",
            )  # again
            self.assertEqual(
                [p["pr"] for p in report.gap_waiting(items)], [1]
            )  # #2's gap has ended
            b = report.board(cfg, st, store, items, [])
            self.assertIn(
                "waiting: 2 follow-ups, 1 first rounds; 1 follow-ups wait for the 2-h gap; 1 heads carried "
                "as merge-only (24 h)",
                b["md"],
            )
            self.assertIn("2 cleared (0 posted, 1 rebase-only, 1 merge-only)", b["md"])
            self.assertEqual(b["throughput"]["merge_only"], 1)
            # A gap ending is no change to show the board again for.
            ended = [dict(items[0], gap_until=ago(1))] + items[1:]
            self.assertEqual(b["fp"], report.board(cfg, st, store, ended, [])["fp"])
            page = pathlib.Path(webdash.write(cfg, st, store, items, [], b)).read_text(
                encoding="utf-8"
            )
        self.assertIn("2 follow-ups (1 wait for the 2-h gap) · 1 first rounds", page)
        self.assertIn("0 posted · 1 rebase-only · 1 merge-only", page)
        self.assertIn("1 heads carried as merge-only (clean checks, no post)", page)
        self.assertIn(f"(gap: after {soon[11:16]}Z)", page)
        self.assertIn("Cleared today (2)", page)


if __name__ == "__main__":
    unittest.main()
