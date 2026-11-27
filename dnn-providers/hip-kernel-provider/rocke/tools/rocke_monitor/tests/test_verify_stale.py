# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The stuck-verification alarm (rkmon_lib/verify_stale.py): a review handed to verification that nothing posts gets
its VERIFY_STALE line past its limit, at most once an hour per PR and head; the line names the review's head and
`rk review-close N`, for a head our account posted a review at already that stands for ours, which a mirror's line
leaves out; a split review's limit grows with its waves; the idle tick's timer falls due with the next line; the tick's
review step adds the alarm after the advance; a review kept unposted, skipped, posted or dropped waits for nothing. The
states and times are made up.
"""

import json
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkmon  # noqa: E402
from rkmon_lib import idle, post, split, triage, verify_stale  # noqa: E402
from rkmon_lib.store import Store, empty_state, now, parse_ts  # noqa: E402

HEAD, NEW_HEAD = "a" * 40, "b" * 40
T0 = "2026-11-19T07:07:00Z"  # when PR 12's review was handed to verification


def at(minutes: float) -> str:
    """T0 plus `minutes`."""
    return idle.plus(T0, minutes)


def ready(pr: int = 12, head: str = HEAD, when: str = T0) -> dict:
    """A review_ready entry as the review pipeline writes it."""
    return {"pr": pr, "kind": "first-round", "head": head, "at": when, "jobs": {}}


def stale(pr: int, age: str, head: str = HEAD) -> str:
    return (
        f"!! VERIFY_STALE #{pr}: handed to verification {age} ago, nothing posted (head {head[:10]}); if our account "
        f"posted at that head already, run `rk review-close {pr}`"
    )


class Base(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._d.name)
        self.cfg = {"paths": {"working_dir": str(self.d)}}
        self.st = empty_state()
        self.st["review_ready"] = [ready()]

    def tearDown(self):
        self._d.cleanup()

    def lines(self, minutes: float) -> list[str]:
        return verify_stale.lines(self.cfg, self.st, at(minutes))


class TestLine(Base):
    def test_the_line_comes_once_the_limit_has_passed_and_says_how_long_ago(self):
        self.assertEqual(self.lines(179), [])
        self.assertEqual(self.lines(180), [stale(12, "3.0 h")])

    def test_the_line_names_the_review_s_head_and_the_command_that_closes_it(self):
        """When our account posted a review at that head already that stands for ours (an older monitor's, or one
        posted by hand; never another deployment's), `rk review-close N` closes the review without a post.
        """
        self.assertEqual(
            self.lines(180),
            [
                "!! VERIFY_STALE #12: handed to verification 3.0 h ago, nothing posted (head aaaaaaaaaa); if our "
                "account posted at that head already, run `rk review-close 12`"
            ],
        )

    def test_a_mirror_s_line_names_no_rk_review_close(self):
        """In a mirror `rk review-close` closes nothing (mirror.NO_CLOSE): the line names the review's head alone."""
        self.cfg["mirror"] = {"enabled": True}
        self.assertEqual(
            self.lines(180),
            [
                "!! VERIFY_STALE #12: handed to verification 3.0 h ago, nothing posted (head aaaaaaaaaa)"
            ],
        )

    def test_at_most_once_an_hour_for_each_pr(self):
        self.st["review_ready"].append(ready(13, "c" * 40, at(30)))
        said = [
            (m, int(re.match(r"!! VERIFY_STALE #(\d+): ", line).group(1)))
            for m in range(0, 8 * 60 + 1, 10)  # a tick every 10 minutes for 8 hours
            for line in self.lines(m)
        ]
        self.assertEqual(
            said,
            [
                (180, 12),
                (210, 13),
                (240, 12),
                (270, 13),
                (300, 12),
                (330, 13),
                (360, 12),
                (390, 13),
                (420, 12),
                (450, 13),
                (480, 12),
            ],
        )
        self.assertEqual(self.lines(540)[0], stale(12, "9.0 h"))

    def test_a_new_head_has_its_own_limit_and_its_own_hour(self):
        self.assertEqual(self.lines(180), [stale(12, "3.0 h")])
        # The head moved: the review was dropped, picked again and handed to verification at 200, all between two full
        # ticks. The next one drops the record of the old head.
        self.st["review_ready"] = [ready(head=NEW_HEAD, when=at(200))]
        self.assertEqual(self.lines(210), [])
        self.assertNotIn(verify_stale.KEY, self.st)
        self.assertEqual(self.lines(379), [])
        self.assertEqual(self.lines(380), [stale(12, "3.0 h", NEW_HEAD)])
        self.assertEqual(self.lines(430), [])
        self.assertEqual(
            self.st[verify_stale.KEY], {"12": {"head": NEW_HEAD, "said": at(380)}}
        )
        self.assertEqual(self.lines(440), [stale(12, "4.0 h", NEW_HEAD)])

    def test_a_review_posted_dropped_or_skipped_gets_none_and_its_record_goes(self):
        self.assertEqual(len(self.lines(180)), 1)
        self.st["review_ready"] = []  # posted (rk post), or dropped (REVIEW_DROP)
        self.assertEqual(self.lines(300), [])
        self.assertNotIn(verify_stale.KEY, self.st)
        self.st["review_ready"] = [ready()]
        self.st["review_skip"] = [12]  # rk review-skip 12
        self.assertEqual(self.lines(300), [])

    def test_a_review_kept_unposted_waits_for_no_post(self):
        self.st["prs"]["12"] = {"n": 12, "status": "open", "head": HEAD, "ours": None}
        verified = self.d / "pr12_fr.md"
        verified.write_text("the verified review\n", encoding="utf-8")
        post.save_draft(self.cfg, self.st, Store(self.d / "state"), 12, str(verified))
        self.assertEqual(self.st["review_ready"], [])  # finished, as a post finishes it
        # The state's word: the review is unposted, not a draft.
        self.assertEqual(self.st["prs"]["12"]["ours"]["kind"], "unposted")
        self.assertTrue((self.d / "PR_UNPOSTED_12.md").exists())
        self.assertEqual(self.lines(300), [])

    def test_the_session_gets_the_line_whatever_triages_it(self):
        self.assertEqual(
            triage.classify(stale(12, "3.0 h"), False), triage.CLAUDE
        )  # tests/test_tick_filter.py: the tick filter's class


class TestSplit(Base):
    def plan(self, rnd: str, parts: int, waves: list, minutes: float) -> None:
        """A split plan in PR 12's review directory of round `rnd`, written `minutes` after the hand-off."""
        d = self.d / "tmp" / f"v12{rnd}" / split.DIR
        d.mkdir(parents=True, exist_ok=True)
        f = d / split.PLAN
        f.write_text(
            json.dumps(
                {"parts": [{"k": k} for k in range(1, parts + 1)], "waves": waves}
            ),
            encoding="utf-8",
        )
        t = parse_ts(at(minutes)).timestamp()
        os.utime(f, (t, t))

    def test_a_split_review_gets_a_limit_per_wave_and_one_for_the_cross_pass(self):
        self.plan("fr", 3, [[1, 2], [3]], 10)
        self.assertEqual(verify_stale.limit_min(self.cfg, ready()), 3 * 180)
        self.assertEqual(self.lines(539), [])
        self.assertEqual(self.lines(540), [stale(12, "9.0 h")])

    def test_one_part_or_a_plan_from_before_the_hand_off_is_no_split(self):
        self.plan("fr", 1, [[1]], 10)  # one part: verified as usual
        self.plan("fu1", 4, [[1], [2], [3], [4]], -60)  # an earlier round's
        self.assertEqual(verify_stale.limit_min(self.cfg, ready()), 180)
        self.assertEqual(self.lines(180), [stale(12, "3.0 h")])


class TestTick(Base):
    def setUp(self):
        super().setUp()
        # The review pipeline on (review-drafts): the full tick's review step says the lines.
        self.cfg["features"] = {"review_automation": {"enabled": True}}

    def due(self, when: str) -> dict:
        timers = {"verify stale": idle.TIMERS["verify stale"]}
        return idle.deadlines(self.cfg, self.st, None, timers, at=when)

    def test_the_idle_tick_s_timer_falls_due_with_the_next_line(self):
        self.assertEqual(self.due(T0), {"verify stale #12": at(180)})
        self.lines(180)
        self.assertEqual(self.due(at(180)), {"verify stale #12": at(240)})
        self.st["review_ready"] = []
        self.assertEqual(self.due(at(180)), {})

    def test_a_line_due_after_the_review_step_keeps_the_next_tick_full(self):
        # The full tick's review step ran a minute before the limit, and the tick records its deadlines a minute after.
        self.assertEqual(self.lines(179), [])
        self.assertEqual(self.due(at(181)), {"verify stale #12": at(180)})

    def test_nothing_falls_due_while_the_review_pipeline_is_off(self):
        self.cfg["features"]["review_automation"]["enabled"] = False
        self.assertEqual(self.due(T0), {})
        self.assertEqual(self.due(at(181)), {})

    def test_the_tick_s_review_step_adds_the_alarm_after_the_advance(self):
        self.st["review_ready"] = [
            ready(12, when=idle.plus(now(), -200)),
            ready(13, when=idle.plus(now(), -200)),
        ]

        def advance(cfg, st, store, plan):  # #13's head moved: the advance drops it
            st["review_ready"] = [r for r in st["review_ready"] if r["pr"] != 13]
            return [
                "REVIEW_DROP #13: closed or head moved before verification; will re-pick"
            ]

        with mock.patch.object(rkmon.review, "advance", side_effect=advance):
            out = rkmon.review_step(self.cfg, self.st, None, [])
        self.assertEqual(len(out), 2, out)
        self.assertTrue(out[0].startswith("REVIEW_DROP #13: "))
        self.assertRegex(
            out[1],
            r"^!! VERIFY_STALE #12: handed to verification 3\.\d h ago, nothing posted \(head aaaaaaaaaa\); if our "
            r"account posted at that head already, run `rk review-close 12`$",
        )

    def test_a_failure_of_the_alarm_is_a_line_and_the_advance_stands(self):
        with mock.patch.object(
            rkmon.review, "advance", return_value=["REVIEW_START #14"]
        ), mock.patch.object(verify_stale, "lines", side_effect=ValueError("bad")):
            self.assertEqual(
                rkmon.review_step(self.cfg, self.st, None, []),
                ["REVIEW_START #14", "(stuck verifications not checked: bad)"],
            )


class TestCard(unittest.TestCase):
    """The session's card for the line, in the skill's tick procedure."""

    def card(self) -> str:
        skill = (ROOT / "agentic" / "skills" / "rocke-monitor" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        return skill.split("`!! VERIFY_STALE #N", 1)[1].split("\n   - **", 1)[0]

    def test_a_stuck_verification_starts_again_in_its_own_round_and_part(self):
        card = self.card()
        # Its own review directory: once the stuck verifier has written notes.md, a follow-up's default round is the
        # next one (prep.default_round).
        self.assertIn("`rk dispatch-pack N --round <round>`", card)
        self.assertIn("briefed to start from that directory's `notes.md`", card)
        # A split review starts again only the part verifier or the cross pass that is stuck; a part's notes are its
        # own file.
        self.assertIn("(`notes.md`; a part verifier's `part<k>_notes.md`)", card)
        self.assertIn(
            "a new one for that part only (`review_procedure.md`, Split review, step 3)",
            card,
        )
        self.assertIn("Its cross pass: a new one (step 4 there)", card)

    def test_a_review_our_account_posted_already_is_closed_without_a_post(self):
        card = self.card()
        self.assertIn("our account posted a review at that head already", card)
        self.assertIn("`rk review-close N`", card)
        self.assertIn("`rk facts N`", card)
        self.assertIn("A mirror's line names no `rk review-close`.", card)


if __name__ == "__main__":
    unittest.main()
