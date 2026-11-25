# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review queue (state review_first, `rk review-queue`): queued reviews start before the plan's, in order; `again`
re-reviews the head of our last review, past our review there and the follow-up gap but not the settle; `rk
review-start` refuses while every pipe is busy, and names the queue. A queued review that a push drops before its post,
whose staging meets another fetch, or whose worker writes no draft goes back to the queue's front, as its entry was; one
started by hand while a stop holds it takes the entry its stopped record carries. A review kept unposted while posting is
off is finished as a posted
one is: nothing counts it in flight, and the picks and the rounds read it as our last review. GitHub is faked; one
pipe, as before the pipes.
"""

import datetime as dt
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    cleanup,
    daylog,
    dispatch,
    doctor,
    draft_stage,
    idle,
    paths,
    plan,
    post,
    prep,
    report,
    review,
    scaler,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

CFG = {
    "review_first_line": "Automated review: Model - qwen, effort: max",
    "rocke_path": "rk",
    "repo": "o/r",
    "operator": {"github_login": "me"},
    "deployment": {"phase": "review"},
    "merge_authority": "propose",
    "settle_min": 30,
    "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    "review": {"pipes": 1},
}
GREEN = "\U0001f7e2"
OLD, H7, H8, H9, H10 = "a" * 40, "7" * 40, "8" * 40, "9" * 40, "b" * 40


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def same_head(head: str) -> dict:
    return {
        "kind": "same-head",
        "old": head,
        "new": head,
        "changed_files": [],
        "new_commits": [],
    }


def started(n: int, head: str, what: str) -> str:
    """The REVIEW_START line of a reviewer-model review in the one pipe."""
    return f"REVIEW_START #{n} ({what}): reviewer model worker started at {head[:10]} (parallel review 1 of 1)"


class QueueCase(unittest.TestCase):
    """#7: our review was posted at its head half an hour ago, inside the 1-h gap. #8 and #10: never reviewed. #9:
    pushed since our review half an hour ago, a follow-up inside the gap. The plan: #9's follow-up, then #8 and #10.
    """

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.cfg = dict(CFG, paths={"working_dir": self.d.name})
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
                common, n=7, head=H7, ours={"head": H7, "verdict": GREEN, "at": ago(30)}
            ),
            "8": dict(common, n=8, head=H8, ours=None),
            "9": dict(
                common,
                n=9,
                head=H9,
                ours={"head": OLD, "verdict": GREEN, "at": ago(30)},
            ),
            "10": dict(common, n=10, head=H10, ours=None),
        }
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(30),
                    "kind": "post",
                    "pr": n,
                    "head": h,
                    "ref": "u",
                    "note": "followup",
                }
                for n, h in ((7, H7), (9, OLD))
            ],
        )
        self.plan = [
            {"prio": 1, "kind": "FOLLOW_UP", "pr": 9, "text": "follow-up on #9"},
            {
                "prio": 3,
                "kind": "FIRST_ROUND",
                "pr": 8,
                "text": "first-round review of #8",
            },
            {
                "prio": 3,
                "kind": "FIRST_ROUND",
                "pr": 10,
                "text": "first-round review of #10",
            },
        ]
        self.live = {
            n: {"state": "open", "head": self.st["prs"][str(n)]["head"]}
            for n in (7, 8, 9, 10)
        }
        self.classified = {7: same_head(H7), 9: same_head(H9)}
        self.jobs = []

    def queue(self, *entries):
        self.st["review_first"] = [{"pr": n, "again": again} for n, again in entries]

    def keep(self, n: int) -> str:
        """PR n's verified review kept unposted, as `rk post` keeps it while posting is off (post.save_draft); the
        kept file."""
        verified = pathlib.Path(self.d.name) / f"pr{n}_verified.md"
        verified.write_text(
            f"{CFG['review_first_line']}\n\n## PR #{n} — a title — {GREEN} First-round: fine\n",
            encoding="utf-8",
        )
        return post.save_draft(self.cfg, self.st, self.store, n, str(verified))

    def pick(self):
        return review.pick_next(
            self.cfg, self.st, self.store, review.first_items(self.st) + self.plan
        )

    def rest(self, path, jq=None, paginate=False, timeout=120):
        """GitHub: each PR's live state and head (self.live); a head pushed two minutes ago."""
        if path.startswith("repos/o/r/pulls/"):
            return dict(self.live[int(path.rsplit("/", 1)[1])])
        if path.startswith("repos/o/r/commits/"):
            return ago(2)
        raise AssertionError(f"unexpected GET {path}")

    def patched(self):
        def start(store, kind, name, script, timeout_min, env=None):
            self.jobs.append((kind, name))
            return {"id": f"{kind}-{name}", "started": ago(0), "pid": 0}

        stack = ExitStack()
        stack.enter_context(mock.patch.object(review.gh, "rest", self.rest))
        stack.enter_context(
            mock.patch.object(
                review,
                "classify",
                lambda cfg, rec, old: dict(self.classified[rec["n"]], old=old),
            )
        )
        stack.enter_context(
            mock.patch.object(review, "_changed_files", return_value=[])
        )
        stack.enter_context(mock.patch.object(review.jobs, "start", start))
        return stack


class TestPick(QueueCase):
    def test_the_queue_goes_before_the_plan_in_its_order(self):
        self.assertEqual(
            self.pick(), {"pr": 8, "kind": "first-round", "again": False}
        )  # #9 waits for the gap
        self.queue((10, False), (8, False))
        self.assertEqual(self.pick(), {"pr": 10, "kind": "first-round", "again": False})
        self.queue((7, True), (10, False))
        self.assertEqual(
            self.pick(), {"pr": 7, "kind": "follow-up", "again": True}
        )  # our review has a head
        # The pool's lookahead (draft_stage.candidates) reads the plan alone; an entry the driver tried is passed over.
        self.assertEqual(
            review.pick_next(self.cfg, self.st, self.store, self.plan)["pr"], 8
        )
        self.assertEqual(
            review.pick_next(
                self.cfg,
                self.st,
                self.store,
                review.first_items(self.st) + self.plan,
                exclude={7},
            )["pr"],
            10,
        )

    def test_an_entry_waits_while_skipped_in_flight_awaiting_verification_settling_or_closed(
        self,
    ):
        self.queue((10, False), (7, True))
        for key, value, why in (
            ("review_skip", [10], "it is in review_skip (rk review-unskip)"),
            ("review_pipes", [{"pr": 10, "head": H10}], "its review is in flight"),
            (
                "review_ready",
                [{"pr": 10, "head": H10}],
                "its review awaits verification",
            ),
        ):
            with self.subTest(key):
                self.st[key] = value
                self.assertEqual(
                    review.first_wait(self.cfg, self.st, self.store, {"pr": 10}), why
                )
                self.assertEqual(self.pick()["pr"], 7)  # the next entry goes
                del self.st[key]
        rec = self.st["prs"]["10"]
        rec.update(head_seen_at=ago(3), head_committed_at=ago(3))
        self.assertTrue(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 10}).startswith(
                "its head settles in "
            )
        )
        self.assertTrue(
            review.first_wait(
                self.cfg, self.st, self.store, {"pr": 10, "again": True}
            ).startswith("its head settles in ")
        )  # again or not
        self.assertEqual(self.pick()["pr"], 7)
        rec.update(head_seen_at=ago(300), head_committed_at=ago(300), status="closed")
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 10}),
            "its PR is closed",
        )
        self.assertEqual(self.pick()["pr"], 7)

    def test_again_passes_our_post_at_the_head_and_the_gap_for_queue_entries_only(self):
        self.queue((7, False))
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 7}),
            "we posted at its head (a re-review there: rk review-queue 7 --again)",
        )
        self.store.ledger_path.write_text(
            ""
        )  # a head recorded as covered, with no post of ours there
        self.st["prs"]["7"]["ours"] = {
            "head": OLD,
            "equivalent_heads": [H7],
            "verdict": GREEN,
            "at": ago(30),
        }
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 7}),
            "our review covers its head (a re-review there: rk review-queue 7 --again)",
        )
        self.st["prs"]["7"]["ours"] = {"head": H7, "verdict": GREEN, "at": ago(30)}
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(30),
                    "kind": "post",
                    "pr": n,
                    "head": h,
                    "ref": "u",
                    "note": "followup",
                }
                for n, h in ((7, H7), (9, OLD))
            ],
        )
        self.assertEqual(self.pick()["pr"], 8)  # it waits; the plan goes on
        self.queue((7, True))
        self.assertEqual(self.pick(), {"pr": 7, "kind": "follow-up", "again": True})
        until = review.gap_until(
            self.cfg,
            self.st,
            self.st["prs"]["9"],
            review.last_review_posts(self.store.ledger()),
        )
        self.queue((9, False))
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 9}),
            f"the gap after our last review, until {until[11:16]}Z",
        )
        self.assertEqual(self.pick()["pr"], 8)
        self.queue((9, True))
        self.assertEqual(self.pick(), {"pr": 9, "kind": "follow-up", "again": True})
        # The plan's items keep both checks: #9's follow-up waits for the gap, #7's head carries our post.
        self.assertEqual(
            review.pick_next(self.cfg, self.st, self.store, self.plan)["pr"], 8
        )
        self.assertIsNone(
            review.pick_next(
                self.cfg,
                self.st,
                self.store,
                [{"prio": 1, "kind": "FOLLOW_UP", "pr": 7, "text": "follow-up on #7"}],
            )
        )


class TestStart(QueueCase):
    def test_same_head_with_again_starts_and_without_again_carries(self):
        with self.patched():
            out = review.start_review(self.cfg, self.st, self.store, 7, "follow-up")
        self.assertEqual(
            out,
            [
                f"REVIEW_SKIP #7: same-head since our review; recorded {H7[:10]} as equivalent"
            ],
        )
        self.assertEqual(review.pipes(self.st), [])
        with self.patched():
            out = review.start_review(
                self.cfg, self.st, self.store, 7, "follow-up", again=True
            )
        self.assertEqual(
            out, [started(7, H7, "follow-up, a re-review at the same head")]
        )
        pipe = review.pipe_of(self.st, 7)
        self.assertEqual(
            (pipe["pr"], pipe["kind"], pipe["head"], pipe["since"]),
            (7, "follow-up", H7, self.st["prs"]["7"]["ours"]["at"]),
        )
        self.assertEqual(self.jobs, [("worker", "pr7")])
        self.assertIn(
            f"#7 follow-up, a re-review at the same head: worker started (worker-pr7) at {H7[:10]}",
            self.store.journal_path.read_text(),
        )

    def test_with_again_a_rebase_only_push_is_still_carried(self):
        self.classified[9] = {
            "kind": "rebase-only",
            "old": OLD,
            "new": H9,
            "changed_files": [],
            "new_commits": [],
        }
        with self.patched():
            out = review.start_review(
                self.cfg, self.st, self.store, 9, "follow-up", again=True
            )
        self.assertEqual(
            out,
            [
                f"REVIEW_SKIP #9: rebase-only since our review; recorded {H9[:10]} as equivalent"
            ],
        )
        self.assertEqual(self.st["prs"]["9"]["ours"]["equivalent_heads"], [H9])

    def test_a_reviewed_head_missing_from_the_mirror_is_a_substantive_follow_up(self):
        self.classified[9] = {
            "kind": "unknown-old-head",
            "old": OLD,
            "new": H9,
            "changed_files": [],
            "new_commits": [],
        }
        with self.patched():
            out = review.start_review(self.cfg, self.st, self.store, 9, "follow-up")
        self.assertEqual(
            out,
            [
                f"#9 changed substantively: our reviewed head {OLD[:10]} is not in the mirror "
                "(unknown-old-head)",
                started(9, H9, "follow-up"),
            ],
        )

    def test_the_driver_starts_the_queue_first_and_each_entry_leaves_once_started(self):
        self.queue((7, True), (9, True), (8, False))
        self.classified[9] = {
            "kind": "rebase-only",
            "old": OLD,
            "new": H9,
            "changed_files": [],
            "new_commits": [],
        }
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines, [started(7, H7, "follow-up, a re-review at the same head")]
        )
        self.assertEqual(
            self.st["review_first"],
            [{"pr": 9, "again": True}, {"pr": 8, "again": False}],
        )
        self.assertIn(
            "#7 left the review queue: REVIEW_START #7",
            self.store.journal_path.read_text(),
        )
        # Carried (REVIEW_SKIP): it leaves, and the next entry starts in the same advance.
        self.st["review_pipes"] = []
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines,
            [
                f"REVIEW_SKIP #9: rebase-only since our review; recorded {H9[:10]} as equivalent",
                started(8, H8, "first-round"),
            ],
        )
        self.assertEqual(self.st["review_first"], [])

    def test_an_entry_stays_while_its_live_head_settles_and_leaves_once_closed_or_merged(
        self,
    ):
        self.queue((10, False), (8, False))
        self.live[10] = {"state": "open", "head": "c" * 40}  # pushed two minutes ago
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertTrue(lines[1].startswith("REVIEW_SETTLING #10"), lines)
        self.assertEqual(lines[2], started(8, H8, "first-round"))
        self.assertEqual(self.st["review_first"], [{"pr": 10, "again": False}])
        self.st["review_pipes"] = []
        self.live[10] = {"state": "closed", "head": "c" * 40}
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, [])
        self.assertEqual(
            lines, [], lines
        )  # #10's new head still settles: nothing is asked of GitHub
        self.st["prs"]["10"].update(
            head_seen_at=ago(300), head_committed_at=ago(300), push_times=[]
        )
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, [])
        self.assertEqual(
            lines,
            [
                "REVIEW_CLOSED #10: it is closed on GitHub; nothing started (the next tick catches "
                "up)"
            ],
        )
        self.assertEqual(self.st.get("review_first"), [])
        self.queue((8, False))
        self.st["prs"]["8"]["status"] = "merged"  # the tick saw it merge
        with self.patched():
            self.assertEqual(review.advance(self.cfg, self.st, self.store, []), [])
        self.assertEqual(self.st["review_first"], [])
        self.assertIn(
            "#8 left the review queue: the PR is merged",
            self.store.journal_path.read_text(),
        )

    def ready(self, n: int) -> str:
        """The REVIEW_READY line of PR n's review in its pipe, its worker and checks done."""
        self.st["review_pipes"] = [
            {
                "pr": n,
                "kind": review.round_kind(self.st["prs"][str(n)]),
                "stage": "worker",
                "head": self.st["prs"][str(n)]["head"],
                "since": ago(30),
                "merge_only": None,
                "worker": {"id": "w", "started": ago(60)},
                "checks": {"id": "c", "started": ago(50)},
            }
        ]
        with self.patched(), mock.patch.object(
            review.jobs, "status", lambda store, job: "done" if job else "none"
        ), mock.patch.object(review, "has_draft", lambda *a, **kw: True):
            return review.advance(self.cfg, self.st, self.store, [])[0]

    def test_review_ready_at_a_head_we_posted_on_says_to_post_with_again(self):
        self.assertTrue(
            self.ready(7).startswith(
                "REVIEW_READY #7 (follow-up): worker done, checks done. We posted at "
                "this head already: post it with --again. Inputs: "
            )
        )
        self.assertTrue(
            self.ready(8).startswith(
                "REVIEW_READY #8 (first-round): worker done, checks done. Inputs: "
            )
        )

    def test_review_ready_at_the_head_of_our_kept_review_says_to_keep_with_again(self):
        """Posting off: #8's review was kept unposted at its head. A re-review there (`rk review-queue 8 --again`) is
        kept only with --again, as one at a head we posted on is posted only with it; once posting is on, the keep
        does not hold the post back."""
        head = self.st["prs"]["8"]["head"]
        self.st["review_ready"] = [
            {"pr": 8, "kind": "first-round", "head": head, "at": ago(20)}
        ]
        kept = pathlib.Path(self.keep(8)).name
        self.assertTrue(
            self.ready(8).startswith(
                "REVIEW_READY #8 (follow-up): worker done, checks done. Our last review is at this head already "
                f"(kept unposted: {kept}): keep it with --again. Inputs: "
            )
        )
        self.cfg["features"] = {"review_automation": {"mode": "live"}}
        self.assertTrue(
            self.ready(8).startswith(
                "REVIEW_READY #8 (follow-up): worker done, checks done. Inputs: "
            )
        )

    def test_a_head_whose_post_of_ours_is_gone_from_the_pr_is_re_reviewed(self):
        """#9's follow-up at its head was deleted from the PR: our review is the earlier one, at an older head, and the
        ledger keeps the deleted post at the head."""
        self.store.append_jsonl(
            "actions.jsonl",
            [
                {
                    "ts": ago(20),
                    "kind": "post",
                    "pr": 9,
                    "head": H9,
                    "ref": "u",
                    "note": "followup",
                }
            ],
        )
        self.classified[9] = {
            "kind": "substantive",
            "old": OLD,
            "new": H9,
            "changed_files": ["x.py"],
            "n_changed": 1,
            "new_commits": [],
        }
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 9}),
            "we posted at its head (a re-review there: rk review-queue 9 --again)",
        )
        self.queue((9, True))
        with self.patched():
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines,
            [
                "#9 changed substantively: 1 files, e.g. x.py",
                started(9, H9, "follow-up, a re-review at the same head"),
            ],
        )
        self.assertIn(
            "We posted at this head already: post it with --again.", self.ready(9)
        )


class TestRequeue(QueueCase):
    """A review the queue started and dropped with no post goes back to the queue's front, as its entry was, while its
    PR is open: its head moved during the review, while it awaited verification, or while a stop held it; its staging
    met another fetch into the shared clone; or its worker wrote no draft the first time. One whose PR closed, or one
    the plan started, does not."""

    NEW = "c" * 40

    def advance(self, items=(), drafted=True) -> list[str]:
        """One advance of the driver over `items`; every job is done, and each draft is written (none when not
        `drafted`)."""
        with self.patched(), mock.patch.object(
            review.jobs, "status", lambda store, job: "done" if job else "none"
        ), mock.patch.object(review, "has_draft", lambda *a, **kw: drafted):
            return review.advance(self.cfg, self.st, self.store, list(items))

    def past_its_first_minutes(self, n: int) -> dict:
        """PR n's worker started an hour ago, so its checks may start; its pipe."""
        pipe = review.pipe_of(self.st, n)
        pipe["worker"]["started"] = ago(60)
        return pipe

    def checked(self, n: int) -> None:
        """PR n's worker and checks ran: the next advance finds both done."""
        self.past_its_first_minutes(n)["checks"] = {"id": "c", "started": ago(50)}

    def push(self, n: int) -> None:
        """A push to PR n a minute ago, which the tick has read: the new head has not settled."""
        self.st["prs"][str(n)].update(
            head=self.NEW, head_seen_at=ago(1), head_committed_at=ago(1)
        )
        self.live[n] = {"state": "open", "head": self.NEW}

    def journal(self) -> str:
        return self.store.journal_path.read_text()

    def test_a_push_during_its_review_puts_it_back_at_the_front_with_its_options(self):
        self.queue((8, True), (7, False))  # #7 waits: we posted at its head
        self.assertEqual(self.advance(), [started(8, H8, "first-round")])
        self.assertEqual(self.st["review_first"], [{"pr": 7, "again": False}])
        self.push(8)
        self.assertEqual(
            self.advance(), [f"REVIEW_DROP #8: its head moved; {review.REQUEUED}"]
        )
        self.assertEqual(
            self.st["review_first"],
            [{"pr": 8, "again": True}, {"pr": 7, "again": False}],
        )
        self.assertIn(
            "#8 back in the review queue: its head moved during the review",
            self.journal(),
        )
        # Once the new head settles, the queue starts it again before the plan, with its options.
        self.st["prs"]["8"].update(head_seen_at=ago(300), head_committed_at=ago(300))
        self.assertEqual(self.advance(self.plan), [started(8, self.NEW, "first-round")])
        pipe = review.pipe_of(self.st, 8)
        self.assertEqual(
            (pipe["again"], pipe[review.QUEUE_ENTRY]), (True, {"pr": 8, "again": True})
        )
        self.assertEqual(self.st["review_first"], [{"pr": 7, "again": False}])

    def test_a_push_while_it_awaits_verification_puts_it_back(self):
        """Its verification included: `rk post` refuses a moved head and leaves the review awaiting verification, and
        the advance after the tick read the new head drops it."""
        self.queue((8, True))
        self.advance()
        self.checked(8)
        self.assertTrue(self.advance()[0].startswith("REVIEW_READY #8 (first-round): "))
        self.assertEqual(
            self.st["review_ready"][0][review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )
        self.push(8)
        self.assertEqual(
            self.advance(),
            [f"REVIEW_DROP #8: its head moved before verification; {review.REQUEUED}"],
        )
        self.assertEqual(
            (self.st["review_ready"], self.st["review_first"]),
            ([], [{"pr": 8, "again": True}]),
        )
        self.assertIn(
            "#8 back in the review queue: its head moved before its review was posted",
            self.journal(),
        )

    def test_a_push_while_a_stop_held_it_puts_it_back(self):
        self.queue((8, True))
        self.advance()
        # rk stop records the review as interrupted.
        review.interrupt_pipe(self.st, self.store, review.pipe_of(self.st, 8))
        self.assertEqual(
            review.interrupted(self.st, 8)[review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )
        # After rk start it starts again at its head, and keeps the entry.
        self.assertEqual(
            self.advance(),
            [
                f"REVIEW_RESUME #8: the first-round a stop interrupted starts again at {H8[:10]}",
                started(8, H8, "first-round"),
            ],
        )
        self.assertEqual(
            review.pipe_of(self.st, 8)[review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )
        # Stopped again, and pushed meanwhile: not started again, back in the queue.
        review.interrupt_pipe(self.st, self.store, review.pipe_of(self.st, 8))
        self.push(8)
        self.assertEqual(
            self.advance(),
            [
                "REVIEW_RESUME #8: not started again: its head moved since the stop; "
                f"{review.REQUEUED}"
            ],
        )
        self.assertEqual(self.st["review_first"], [{"pr": 8, "again": True}])
        self.assertIn(
            "#8 back in the review queue: its head moved while its review was stopped",
            self.journal(),
        )

    def test_one_whose_pr_closed_does_not_go_back(self):
        self.queue((8, False))
        self.advance()
        self.st["prs"]["8"]["status"] = "closed"
        self.assertEqual(
            self.advance(), ["REVIEW_DROP #8: closed or head moved; will re-pick"]
        )
        self.assertEqual(self.st["review_first"], [])
        self.assertNotIn(review.REQUEUED, self.journal())

    def test_one_the_plan_started_does_not_go_back(self):
        self.assertEqual(
            self.advance(self.plan), [started(8, H8, "first-round")]
        )  # #9 waits for the gap
        self.assertNotIn(review.QUEUE_ENTRY, review.pipe_of(self.st, 8))
        self.push(8)
        self.assertEqual(
            self.advance(), ["REVIEW_DROP #8: closed or head moved; will re-pick"]
        )
        self.assertEqual(self.st.get("review_first") or [], [])
        self.assertNotIn(review.REQUEUED, self.journal())

    def test_an_entry_queued_again_during_the_review_goes_first_with_its_options(self):
        self.queue((8, True), (7, False))
        self.advance()
        review.queue_add(self.st, self.store, [8], False)  # rk review-queue 8
        self.push(8)
        self.advance()
        self.assertEqual(
            self.st["review_first"],
            [{"pr": 8, "again": False}, {"pr": 7, "again": False}],
        )

    def test_a_staging_that_met_another_fetch_goes_back_and_starts_again(self):
        """Not the PR's fault: no skip. The queue starts it again in the same advance, with its options."""
        self.queue((8, True))
        self.advance()
        self.past_its_first_minutes(8)
        log = pathlib.Path(paths.get(self.cfg, "workers")) / "pr8" / "mkwt.log"
        log.parent.mkdir(parents=True)
        log.write_text(f"fatal: {review.CLONE_RACE}\nmkwt failed\n")
        self.assertEqual(
            self.advance(),
            [
                "REVIEW_FAIL #8: its staging met another fetch into the shared clone (shallow file changed); "
                + review.REQUEUED,
                started(8, H8, "first-round"),
            ],
        )
        self.assertEqual(
            review.pipe_of(self.st, 8)[review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )
        self.assertEqual(
            (self.st["review_first"], self.st.get("review_skip") or []), ([], [])
        )
        self.assertIn(
            "#8 back in the review queue: its staging met a change of the shared clone",
            self.journal(),
        )

    def test_a_worker_that_wrote_no_draft_goes_back_and_is_tried_once_more(self):
        """The queue starts it again in the same advance, with its options; a second worker with no draft skips the
        PR, as it skips one the plan started."""
        self.queue((8, True))
        self.advance()
        self.checked(8)
        self.assertEqual(
            self.advance(drafted=False),
            [
                f"REVIEW_FAIL #8: worker done, no draft; {review.REQUEUED}",
                started(8, H8, "first-round"),
            ],
        )
        self.assertEqual(
            review.pipe_of(self.st, 8)[review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )
        self.assertIn(
            "#8 back in the review queue: its worker ended done with no draft",
            self.journal(),
        )
        self.checked(8)
        self.assertEqual(
            self.advance(drafted=False),
            [
                "REVIEW_FAIL #8: worker done again, no draft; skipped until `rk review-unskip 8`"
            ],
        )
        self.assertEqual((self.st["review_first"], self.st["review_skip"]), ([], [8]))

    def test_the_procedure_says_which_dropped_reviews_go_back(self):
        """The session stops a dropped review's agents; the procedure tells it where the PR goes, in the driver's
        words."""
        proc = (
            pathlib.Path(__file__).resolve().parent.parent
            / "agentic/skills/rocke-monitor/review_procedure.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "a PR the review queue started goes back to the front of the queue (its line says "
            f"`{review.REQUEUED}`), any other comes back with the plan",
            " ".join(proc.split()),
        )


class TestCli(QueueCase):
    def rk(self, *argv):
        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(list(argv))
        return code, "\n".join(out).split("\n")

    def test_add_update_list_and_drop(self):
        self.store.save(self.st)
        self.assertEqual(self.rk("review-queue"), (0, ["review queue: empty"]))
        head = "review queue (2; the driver starts these before the plan, in order):"
        self.assertEqual(
            self.rk("review-queue", "8", "7", "--again"),
            (
                0,
                [
                    "queued #8, #7 with --again",
                    head,
                    "  1. #8 --again: next",
                    "  2. #7 --again: may start",
                ],
            ),
        )
        self.assertEqual(
            self.store.load()["review_first"],
            [{"pr": 8, "again": True}, {"pr": 7, "again": True}],
        )
        # Queued once more: the entry keeps its place and takes the new flag.
        self.assertEqual(
            self.rk("review-queue", "7")[1],
            [
                "queued #7",
                head,
                "  1. #8 --again: next",
                "  2. #7: waits: we posted at its head (a re-review there: rk review-queue 7 --again)",
            ],
        )
        self.assertEqual(
            self.store.load()["review_first"],
            [{"pr": 8, "again": True}, {"pr": 7, "again": False}],
        )
        self.assertEqual(
            self.rk("review-queue", "--drop", "8", "5")[1][:2],
            [
                "#8 dropped; #5 was not queued",
                "review queue (1; the driver starts these before the plan, in order):",
            ],
        )
        self.assertEqual(self.store.load()["review_first"], [{"pr": 7, "again": False}])
        self.assertIn(
            "#8 left the review queue: dropped by hand",
            self.store.journal_path.read_text(),
        )
        # Refused, and nothing changes.
        st = self.store.load()
        st["prs"]["10"]["status"] = "merged"
        self.store.save(st)
        before = self.store.state_path.read_bytes()
        self.assertEqual(
            self.rk("review-queue", "8", "10"),
            (4, ["REFUSED: #10 is merged: only an open PR is queued"]),
        )
        self.assertEqual(
            self.rk("review-queue", "12"),
            (4, ["REFUSED: #12 is not tracked: only an open PR is queued"]),
        )
        self.assertEqual(self.rk("review-queue", "--drop")[0], 4)
        self.assertEqual(self.rk("review-queue", "7", "--again", "--drop")[0], 4)
        self.assertEqual(self.store.state_path.read_bytes(), before)

    def test_review_status_shows_the_queue(self):
        self.queue((9, False), (10, False))
        self.store.save(self.st)
        until = review.gap_until(
            self.cfg,
            self.st,
            self.st["prs"]["9"],
            review.last_review_posts(self.store.ledger()),
        )
        code, out = self.rk("review-status")
        # The dashboards' In review count: #10 may start now (queued to start); #9 waits for the gap.
        self.assertEqual(
            out[0], "in review: 1 (1 queued to start); #10 queued to start"
        )
        i = out.index(
            "review queue (2; the driver starts these before the plan, in order):"
        )
        self.assertEqual(
            out[i + 1 : i + 3],
            [
                f"  1. #9: waits: the gap after our last review, until {until[11:16]}Z",
                "  2. #10: next",
            ],
        )

    def test_review_status_names_the_reviews_that_wait_on_a_verifier(self):
        """Not one kept unposted (posting off: its review is finished), nor one of a skipped PR
        (review.waiting_on_verifier)."""
        at = ago(20)
        self.st["review_ready"] = [
            {"pr": n, "kind": "follow-up", "head": h, "at": at}
            for n, h in ((7, H7), (8, H8), (9, H9))
        ]
        self.keep(7)  # posting off: kept unposted
        self.st["review_skip"] = [9]
        self.store.save(self.st)
        code, out = self.rk("review-status")
        self.assertEqual(
            [x for x in out if x.startswith("awaiting verification:")],
            [f"awaiting verification: #8 follow-up at {H8[:10]} since {at}"],
        )

    def test_a_skip_withdraws_the_pr_s_review_awaiting_verification(self):
        """`rk review-skip N` keeps PR N out of the picks and takes back what it has in flight: its review awaiting
        verification leaves the list, so nothing waits on a verifier for it any more."""
        self.st["review_ready"] = [
            {"pr": n, "kind": "follow-up", "head": h, "at": ago(20)}
            for n, h in ((8, H8), (9, H9))
        ]
        self.store.save(self.st)
        self.assertEqual(
            self.rk("review-skip", "9"),
            (0, ["#9's review awaiting verification is withdrawn", "review_skip: [9]"]),
        )
        self.assertEqual([r["pr"] for r in self.store.load()["review_ready"]], [8])
        # Skipped again, it has nothing left to withdraw.
        self.assertEqual(self.rk("review-skip", "9"), (0, ["review_skip: [9]"]))

    def test_review_start_refuses_while_every_pipe_is_busy(self):
        started_ = []

        def start(cfg, st, store, n, kind, force=False, again=False):
            started_.append((n, kind, again))
            return [
                f"REVIEW_START #{n} ({kind}{', a re-review at the same head' if again else ''}): reviewer model "
                "worker started"
            ]

        self.st["review_pipes"] = [{"pr": 9, "kind": "follow-up", "head": H9}]
        self.queue((7, True))
        self.store.save(self.st)
        with mock.patch.object(review, "start_review", start):
            code, out = self.rk("review-start", "7", "--again")
            self.assertEqual(code, 4)
            self.assertTrue(
                out[-1].startswith(
                    "REFUSED: all 1 parallel reviews are busy (#9); queue it: rk review-queue 7 "
                    "--again. Parallel reviews: "
                ),
                out,
            )
            code, out = self.rk("review-start", "8")
            self.assertEqual(code, 4)
            self.assertTrue(
                out[-1].startswith(
                    "REFUSED: all 1 parallel reviews are busy (#9); queue it: rk review-queue 8. "
                    "Parallel reviews: "
                ),
                out,
            )
            self.assertEqual(
                self.rk("review-start", "9"),
                (4, ["REFUSED: #9 is being reviewed already (rk review-status)"]),
            )
            self.assertEqual(started_, [])
            st = self.store.load()
            st["review_pipes"] = []
            self.store.save(st)
            code, out = self.rk(
                "review-start", "7"
            )  # a pipe is free; without --again the gap holds it
            self.assertEqual(code, 4)
            self.assertTrue(
                out[-1].startswith("REFUSED: #7's follow-up waits for the 1-h gap"), out
            )
            self.assertEqual(
                self.rk("review-start", "7", "--again"),
                (
                    0,
                    [
                        "REVIEW_START #7 (follow-up, a re-review at the same head): reviewer model worker "
                        "started"
                    ],
                ),
            )
        self.assertEqual(started_, [(7, "follow-up", True)])
        self.assertEqual(
            self.store.load()["review_first"], []
        )  # it started: it left the queue

    def test_review_start_of_a_queued_pr_keeps_its_entry_as_the_driver_does(self):
        """Its review goes back to the queue as one the driver started would, should a push drop it."""
        self.queue((8, True))
        self.store.save(self.st)
        with self.patched():
            self.assertEqual(
                self.rk("review-start", "8"), (0, [started(8, H8, "first-round")])
            )
        st = self.store.load()
        self.assertEqual(st["review_first"], [])
        self.assertEqual(
            review.pipe_of(st, 8)[review.QUEUE_ENTRY], {"pr": 8, "again": True}
        )

    def test_review_start_of_a_stopped_queued_review_takes_its_entry(self):
        """Started by hand before the driver starts it again, it replaces the review a stop held: that record ends, and
        the entry it carries passes to the review started, should a push drop it."""
        entry = {"pr": 8, "again": True}
        review.interrupt_pipe(
            self.st,
            self.store,
            {"pr": 8, "kind": "first-round", "head": H8, review.QUEUE_ENTRY: entry},
        )
        self.store.save(self.st)
        with self.patched():
            self.assertEqual(
                self.rk("review-start", "8"), (0, [started(8, H8, "first-round")])
            )
        st = self.store.load()
        self.assertEqual(review.pipe_of(st, 8)[review.QUEUE_ENTRY], entry)
        self.assertIsNone(review.interrupted(st, 8))


class TestApproval(QueueCase):
    """#8 is an in-depth review: a first round over autopilot_max_lines (30,000 by default). It waits for the operator's
    approval: the driver, the review queue and `rk review-start` leave it, and nothing names it to the session, while
    the reviews behind it go on. Approved, it starts as a review too large for the autopilot does; declined (`rk
    review-skip`), it is skipped. A later round of a giant PR is light and needs no approval, nor does a normal PR.
    """

    BIG = 40000
    rk = TestCli.rk
    ready = TestStart.ready
    advance, push, NEW = TestRequeue.advance, TestRequeue.push, TestRequeue.NEW

    def setUp(self):
        super().setUp()
        self.st["prs"]["8"]["size"] = self.BIG
        # #10 below #8: a review started by hand holds back the ones below it.
        self.plan[2]["prio"] = 4

    def approve(self, n: int) -> str:
        ok, said = review.approve(self.cfg, self.st, self.store, n)
        self.assertTrue(ok, said)
        return said

    def test_it_waits_and_names_nothing_to_the_session(self):
        self.assertTrue(review.needs_approval(self.cfg, self.st, 8))
        # #9 waits for the gap; #8 holds nothing back.
        self.assertEqual(self.pick()["pr"], 10)
        self.assertEqual(review.oversized(self.cfg, self.st, self.plan, self.store), [])
        self.assertEqual(review.awaiting_approval(self.cfg, self.st, self.plan), [8])
        self.st["review_ready"] = [{"pr": 10, "head": H10}]
        # No REVIEW_MANUAL: nothing wakes the session.
        with self.patched():
            self.assertEqual(
                review.advance(self.cfg, self.st, self.store, self.plan), []
            )
        self.assertEqual((review.pipes(self.st), self.jobs), ([], []))
        self.assertEqual(review.awaiting_approval(self.cfg, self.st, self.plan), [8])

    def test_approved_it_is_started_by_hand_and_declined_it_is_skipped(self):
        self.assertEqual(
            self.approve(8),
            "#8 approved: its in-depth review (40,000 changed lines: about 6 agents) starts at its turn.",
        )
        self.assertIn(
            "review-approve #8: its in-depth review (40000 changed lines) may start",
            self.store.journal_path.read_text(),
        )
        # A push before it starts: the approval stands.
        self.st["prs"]["8"]["head"] = "c" * 40
        self.assertFalse(review.needs_approval(self.cfg, self.st, 8))
        self.assertEqual(review.awaiting_approval(self.cfg, self.st, self.plan), [])
        self.assertIsNone(self.pick())  # it holds back #10, below it
        self.assertEqual(
            review.manual_lines(self.cfg, self.st, self.store, self.plan),
            [
                "REVIEW_MANUAL #8 (first-round, 40000 lines > 30000): too large for the autopilot; start it by hand "
                "now with `rk review-start 8`, and at its REVIEW_READY split the verification with `rk review-split 8`"
            ],
        )
        # Declined: skipped, and its approval withdrawn.
        self.assertEqual(
            review.decline(self.cfg, self.st, self.store, 8),
            (
                True,
                "#8 declined: its in-depth review does not start, and the PR is in review_skip "
                f"(`{doctor.rk_command('review-unskip', '8')}` puts it back).",
            ),
        )
        self.assertEqual(self.st["review_skip"], [8])
        self.assertEqual(review.oversized(self.cfg, self.st, self.plan, self.store), [])
        self.assertEqual(review.awaiting_approval(self.cfg, self.st, self.plan), [])
        self.assertEqual(self.pick()["pr"], 10)
        # Approved again, it leaves review_skip.
        self.assertTrue(self.approve(8).endswith(", and the PR is out of review_skip."))
        self.assertEqual(self.st["review_skip"], [])

    def test_a_later_round_and_a_normal_pr_need_no_approval(self):
        self.st["prs"]["9"]["size"] = self.BIG
        self.assertFalse(review.needs_approval(self.cfg, self.st, 9))
        self.assertFalse(review.needs_approval(self.cfg, self.st, 10))
        self.st["prs"]["7"]["status"] = "closed"
        # A Decline of #9 would skip its light later rounds. (An approval of #9 is for a first round forced on it:
        # test_a_first_round_forced_on_a_reviewed_pr_waits_for_an_approval.)
        self.assertEqual(
            review.decline(self.cfg, self.st, self.store, 9),
            (
                False,
                "#9's next review is a later round, a light one (one verifier): it needs no approval. Nothing "
                "changed.",
            ),
        )
        for n, said in (
            (
                10,
                "#10 is 10 changed lines, within the 30,000 the autopilot reviews: its review needs no approval. "
                "Nothing changed.",
            ),
            (7, "#7 is closed: nothing changed."),
            (12, "#12 is not tracked: nothing changed."),
        ):
            for decide in (review.approve, review.decline):
                with self.subTest(pr=n, decide=decide.__name__):
                    self.assertEqual(
                        decide(self.cfg, self.st, self.store, n), (False, said)
                    )
        self.assertNotIn(review.APPROVED, self.st)
        self.assertFalse(self.st.get("review_skip"))
        # The light round is left to the session as before, once outside the gap.
        self.store.ledger_path.write_text("")
        self.st["prs"]["9"]["ours"]["at"] = ago(300)
        self.assertEqual(
            review.oversized(self.cfg, self.st, self.plan, self.store),
            [{"pr": 9, "kind": "follow-up", "size": self.BIG, "cap": 30000}],
        )

    def test_a_hand_queued_one_waits_and_starts_from_the_queue_once_approved(self):
        self.queue((8, False), (10, False))
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 8, "again": True}),
            "your approval",
        )
        self.assertEqual(
            review.queue_lines(self.cfg, self.st, self.store)[1:],
            ["  1. #8: waits: your approval", "  2. #10: next"],
        )
        # A hand-queued normal PR starts as before.
        self.assertEqual(self.pick()["pr"], 10)
        self.approve(8)
        self.assertEqual(self.pick(), {"pr": 8, "kind": "first-round", "again": False})
        with self.patched():
            self.assertEqual(
                review.advance(self.cfg, self.st, self.store, self.plan),
                [started(8, H8, "first-round")],
            )
        self.assertEqual(self.st["review_first"], [{"pr": 10, "again": False}])

    def test_its_review_ready_says_how_to_verify_it(self):
        """A review the queue started had no REVIEW_MANUAL line: its REVIEW_READY says it is to be split, and a later
        round's that it is light."""
        self.approve(8)
        self.assertIn(
            "checks done. 40,000 changed lines, too large for the autopilot: split the verification with `rk "
            "review-split 8`. Inputs: ",
            self.ready(8),
        )
        self.st["prs"]["9"]["size"] = self.BIG
        self.assertIn(
            "checks done. 40,000 changed lines, too large for the autopilot: brief one light verifier on our open "
            "findings and the delta (no split, no code quality pass). Inputs: ",
            self.ready(9),
        )
        self.assertIn("checks done. Inputs: ", self.ready(10))

    def test_an_approval_ends_with_its_round(self):
        self.st["prs"]["10"]["size"] = self.BIG
        self.approve(8)
        self.approve(10)
        for a in self.st[review.APPROVED]:
            a["at"] = ago(5)  # approved five minutes ago
        self.st["prs"]["8"]["ours"] = {"head": H8, "verdict": GREEN, "at": ago(1)}
        with self.patched():
            review.advance(self.cfg, self.st, self.store, [])
        self.assertEqual([a["pr"] for a in self.st[review.APPROVED]], [10])
        self.st["prs"]["10"]["status"] = "merged"
        with self.patched():
            review.advance(self.cfg, self.st, self.store, [])
        self.assertEqual(self.st[review.APPROVED], [])

    def test_the_commands(self):
        self.store.save(self.st)
        wait = (
            "#8 is 40,000 changed lines: its in-depth review runs about 6 agents. It starts once you approve it on the "
            f"admin dashboard or with `{doctor.rk_command('review-approve', '8')}`."
        )
        self.assertEqual(
            self.rk("review-queue", "8", "10"),
            (
                0,
                [
                    "queued #8, #10",
                    wait,
                    "review queue (2; the driver starts these before the plan, in order):",
                    "  1. #8: waits: your approval",
                    "  2. #10: next",
                ],
            ),
        )
        for opts in ((), ("--now", "--again"), ("--kind", "follow-up")):
            with self.subTest(opts=opts):
                self.assertEqual(
                    self.rk("review-start", "8", *opts), (4, [f"REFUSED: {wait}"])
                )
        approved = "#8 approved: its in-depth review (40,000 changed lines: about 6 agents) starts at its turn"
        self.assertEqual(self.rk("review-approve", "8"), (0, [approved + "."]))
        self.assertEqual([a["pr"] for a in self.store.load()[review.APPROVED]], [8])
        self.assertEqual(
            self.rk("review-skip", "8"),
            (
                0,
                [
                    "#8's approval of its in-depth review is withdrawn",
                    "review_skip: [8]",
                ],
            ),
        )
        self.assertEqual(
            self.rk("review-approve", "8"),
            (0, [approved + ", and the PR is out of review_skip."]),
        )
        self.assertEqual(self.rk("review-approve", "10")[0], 4)
        with self.patched():
            self.assertEqual(
                self.rk("review-start", "8"), (0, [started(8, H8, "first-round")])
            )
        self.assertEqual(
            self.store.load()["review_first"], [{"pr": 10, "again": False}]
        )

    def test_a_first_round_forced_on_a_reviewed_pr_waits_for_an_approval(self):
        """A first round forced on a reviewed giant PR is an in-depth review too: `rk review-start N --kind
        first-round` refuses it, and any other start of it waits (start_review), until the operator approves it with
        the command, as the Needs you list does not ask for it. The approval stands until a review of ours is recorded
        after it."""
        self.st["prs"]["9"]["size"] = self.BIG
        self.store.save(self.st)
        wait = (
            "#9 is 40,000 changed lines: its in-depth review runs about 6 agents. It starts once you approve it with "
            f"`{doctor.rk_command('review-approve', '9')}`."
        )
        forced = ("review-start", "9", "--kind", "first-round")
        self.assertEqual(self.rk(*forced, "--now"), (4, [f"REFUSED: {wait}"]))
        with self.patched():
            self.assertEqual(
                review.start_review(
                    self.cfg, self.st, self.store, 9, "first-round", force=True
                ),
                [f"REVIEW_WAIT #9: {wait}"],
            )
        self.assertEqual((review.pipes(self.st), self.jobs), ([], []))
        # The Needs you list asks for #8's first review alone.
        self.assertEqual(review.awaiting_approval(self.cfg, self.st, self.plan), [8])
        self.assertEqual(
            self.rk("review-approve", "9"),
            (
                0,
                [
                    "#9 approved: an in-depth review of it (40,000 changed lines: about 6 agents) may start with "
                    f"`{doctor.rk_command(*forced)}`; its next review is otherwise a light one."
                ],
            ),
        )
        # Our review from before the approval leaves it standing.
        st = self.store.load()
        review.prune_approved(st)
        self.assertTrue(review.approved(st, 9))
        with self.patched():
            self.assertEqual(self.rk(*forced), (0, [started(9, H9, "first-round")]))
        st = self.store.load()
        st[review.APPROVED][0]["at"] = ago(5)  # approved five minutes ago
        st["prs"]["9"]["ours"] = {"head": H9, "verdict": GREEN, "at": ago(1)}
        review.prune_approved(st)
        self.assertFalse(review.approved(st, 9))

    def test_a_forced_first_round_dropped_from_the_queue_goes_back_as_one(self):
        """A first round forced on a reviewed giant PR, started by hand from its queue entry, goes back to the queue
        when a push drops it, its entry carrying the kind: the next start is the same in-depth round, its approval
        standing, with no follow-up gap to wait for."""
        self.st["prs"]["9"]["size"] = self.BIG
        self.queue((9, False))
        self.approve(9)
        self.store.save(self.st)
        with self.patched():
            self.assertEqual(
                self.rk("review-start", "9", "--kind", "first-round"),
                (0, [started(9, H9, "first-round")]),
            )
        self.st = self.store.load()
        self.push(9)
        self.assertEqual(
            self.advance(), [f"REVIEW_DROP #9: its head moved; {review.REQUEUED}"]
        )
        self.assertEqual(
            self.st["review_first"], [{"pr": 9, "again": False, "kind": "first-round"}]
        )
        self.assertTrue(review.approved(self.st, 9))
        self.assertTrue(
            review.queue_lines(self.cfg, self.st, self.store)[1].startswith(
                "  1. #9 (first-round): waits: its head settles in "
            )
        )
        # Once the new head settles, the queue starts the same round, inside the gap after our last review.
        self.st["prs"]["9"].update(head_seen_at=ago(300), head_committed_at=ago(300))
        self.assertEqual(self.pick(), {"pr": 9, "kind": "first-round", "again": False})
        self.assertEqual(self.advance(), [started(9, self.NEW, "first-round")])
        self.assertEqual(self.st["review_first"], [])


class TestKeptUnposted(QueueCase):
    """Posting off: a verified review kept unposted (post.save_draft) is finished work, as a posted one is
    (review.review_finished). Nothing counts it in flight any more, and it is our last review of its PR.
    """

    def hand_over(self, n: int) -> None:
        """PR n's review at its head, handed to verification (the driver's REVIEW_READY), a micro round's."""
        rec = self.st["prs"][str(n)]
        self.st["review_ready"] = [
            {
                "pr": n,
                "kind": review.round_kind(rec),
                "head": rec["head"],
                "at": ago(20),
                "micro": ["a new thread"],
            }
        ]

    def counting(self, n: int) -> list[str]:
        """The readers that count PR n's review at its head as in flight."""
        st, head = self.st, self.st["prs"][str(n)]["head"]
        return [
            name
            for name, counts in (
                ("verifier", n in [r["pr"] for r in review.waiting_on_verifier(st)]),
                (
                    "in review",
                    n in [r["pr"] for r in report.in_review(self.cfg, st, self.store)],
                ),
                ("scaler", n in scaler.active(st, self.store)["pipe"]),
                ("scaler awaiting", n in scaler.awaiting(st)),
                ("cleanup", n in cleanup.in_use(st)),
                (
                    "shadow drafter",
                    n in [r["pr"] for r in draft_stage.live_reviews(st)],
                ),
                (
                    "shadow drafter's wait",
                    bool(draft_stage.waiting_review(st, n, head)),
                ),
                ("micro items", bool(dispatch.micro_items(st, n, head))),
            )
            if counts
        ]

    def test_it_holds_nothing_in_flight(self):
        self.hand_over(8)
        self.assertEqual(
            len(self.counting(8)), 8
        )  # each one, while it awaits verification
        kept = pathlib.Path(self.keep(8))
        self.assertEqual(self.counting(8), [])
        self.assertEqual(self.st["review_ready"], [])
        self.assertEqual(kept.parent, post.posted_dir(self.cfg))
        self.assertIn(
            f"#8 kept unposted, posting is off: {kept}",
            self.store.journal_path.read_text(),
        )

    def test_it_is_our_last_review_of_the_pr(self):
        """Not reviewed again at its head, and a new head gets the follow-up it would get after a post."""
        self.hand_over(8)
        self.keep(8)
        rec = self.st["prs"]["8"]
        self.assertTrue(post.kept_unposted(rec["ours"]))
        self.assertEqual(
            {k: rec["ours"][k] for k in ("head", "url", "count", "verdict")},
            {"head": H8, "url": None, "count": 1, "verdict": GREEN},
        )
        self.assertIsNone(plan.review_kind(rec))
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 8}),
            "our review covers its head (a re-review there: rk review-queue 8 --again)",
        )
        rec["head"] = "c" * 40  # a push
        self.assertEqual(plan.review_kind(rec), "FOLLOW_UP")
        self.assertEqual(review.round_kind(rec), "follow-up")
        self.assertEqual(prep.default_round(self.st, 8), "fu1")

    def test_its_record_names_its_file(self):
        """The record names the file the review was kept in, the second keep's its own (post.kept_file); a posted
        review's record names none."""
        self.hand_over(8)
        first = pathlib.Path(self.keep(8))
        self.hand_over(8)
        second = pathlib.Path(self.keep(8))
        ours = self.st["prs"]["8"]["ours"]
        self.assertEqual(
            (first.name, second.name, ours["file"]),
            ("PR_UNPOSTED_8.md", "PR_UNPOSTED_8_2.md", "PR_UNPOSTED_8_2.md"),
        )
        self.assertEqual(post.kept_file(self.cfg, ours), second)
        posted = post.our_review(
            self.st["prs"]["8"], "x", H8, url="https://example.com/review/1"
        )
        self.assertNotIn("file", posted)
        self.assertIsNone(post.kept_file(self.cfg, posted))

    def test_its_follow_up_waits_for_the_gap_after_it(self):
        """As one after a post waits (plan.gap_until), though #9's last post in the ledger is three hours old; the
        queue says it waits for the gap after our last review."""
        self.hand_over(9)
        self.keep(9)
        rec = self.st["prs"]["9"]
        rec["head"] = "c" * 40  # a push
        until = idle.plus(rec["ours"]["at"], 60)
        self.assertEqual(plan.gap_until(self.cfg, self.st, rec, {9: ago(180)}), until)
        self.assertEqual(
            review.first_wait(self.cfg, self.st, self.store, {"pr": 9}),
            f"the gap after our last review, until {until[11:16]}Z",
        )

    def test_the_last_tick_s_plan_does_not_start_it_again(self):
        """The driver picks from the last tick's plan (review.pick_next), made while #8's review awaited
        verification."""
        self.hand_over(8)
        self.assertEqual(
            self.pick(), {"pr": 10, "kind": "first-round", "again": False}
        )  # #9 waits for the gap
        self.keep(8)
        self.assertEqual(self.pick(), {"pr": 10, "kind": "first-round", "again": False})

    def refresh(self, latest: dict, changed: set[str]) -> int:
        """The poll's refresh (rkmon.refresh_ours) of the PRs that `changed`, GitHub showing latest[n] as our last
        post on PR n (None: no post of ours); a new post's reviewed head is the PR's old head.
        """
        with mock.patch.object(
            rkmon.snapshot, "our_latest_review", lambda cfg, n: latest[n]
        ), mock.patch.object(
            rkmon.snapshot,
            "resolve_reviewed_head",
            lambda cfg, store, n, ours: dict(ours, head=OLD, head_src="timeline"),
        ):
            return rkmon.refresh_ours(self.cfg, self.store, self.st, changed)

    def test_it_stays_our_last_review_until_a_post_of_ours_follows_it(self):
        """The poll reads our last post on GitHub again whenever a PR changes, a push among them (rkmon.refresh_ours).
        #9's post at its old head came before the kept review."""
        self.hand_over(9)
        self.keep(9)
        kept = self.st["prs"]["9"]["ours"]
        older = {"kind": "comment", "id": 1, "at": ago(30), "head": None}
        self.assertEqual(self.refresh({9: older}, {"9"}), 0)
        # None of ours: one deleted, say.
        self.assertEqual(self.refresh({9: None}, {"9"}), 0)
        self.assertIs(self.st["prs"]["9"]["ours"], kept)
        # A post of ours since (by hand, say) is our last review from then on.
        self.assertEqual(self.refresh({9: dict(older, id=2, at=ago(-1))}, {"9"}), 1)
        self.assertEqual(self.st["prs"]["9"]["ours"]["id"], 2)

    def test_once_posting_is_on_our_last_review_is_the_one_github_shows(self):
        """A kept review is our last review only while posting is off. Once posting is on, the next poll reads every
        PR whose last review is a kept one again, changed or not: #8 has no post of ours on GitHub, so its next review
        is a first round; #9 has one from before the kept review (another monitor's, under the same login), so its
        next review is a follow-up to that post."""
        for n in (8, 9):
            self.hand_over(n)
            self.keep(n)
        self.cfg["features"] = {"review_automation": {"mode": "live"}}
        theirs = {"kind": "comment", "id": 1, "at": ago(30), "head": None}
        self.assertEqual(self.refresh({8: None, 9: theirs}, set()), 1)
        rec8, rec9 = self.st["prs"]["8"], self.st["prs"]["9"]
        self.assertIsNone(rec8["ours"])
        self.assertEqual(plan.review_kind(rec8), "FIRST_ROUND")
        self.assertEqual((rec9["ours"]["id"], rec9["ours"]["head"]), (1, OLD))
        self.assertEqual(plan.review_kind(rec9), "FOLLOW_UP")

    def test_with_no_review_awaiting_verification_it_is_not_our_last_review(self):
        """A keep by hand, or one late after the head moved: the file is kept and logged, but the head it covers is not
        known, and the review of the new head in flight stays."""
        self.st["review_pipes"] = [{"pr": 8, "kind": "first-round", "head": "c" * 40}]
        kept = pathlib.Path(self.keep(8))
        self.assertTrue(kept.is_file())
        self.assertIsNone(self.st["prs"]["8"]["ours"])
        self.assertEqual([p["pr"] for p in review.pipes(self.st)], [8])
        self.assertEqual(
            [(r["kind"], r["pr"]) for r in daylog.entries(self.store)],
            [("unposted", 8)],
        )

    def test_it_is_logged_in_the_day_log_as_its_post_would_be(self):
        """With the round a post of it would have (post.post_note, read before the review is finished), its verdict
        and its findings, as an `unposted` entry: the day's summary keeps it apart from the reviews posted.
        """
        self.hand_over(8)
        self.keep(8)
        self.hand_over(9)
        self.keep(9)
        rows = daylog.entries(self.store)
        fine = f"{GREEN} First-round: fine · no net-new findings"
        self.assertEqual(
            [(r["kind"], r["pr"], r["text"], r["verdict"]) for r in rows],
            [
                ("unposted", 8, f"first round {fine}", GREEN),
                ("unposted", 9, f"follow-up {fine}", GREEN),
            ],
        )
        self.assertNotIn("url", rows[0])
        self.assertTrue(
            daylog.summary(rows).startswith(
                f"reviews posted 0 · reviews kept unposted 2 (2 {GREEN}) · "
            ),
            daylog.summary(rows),
        )


if __name__ == "__main__":
    unittest.main()
