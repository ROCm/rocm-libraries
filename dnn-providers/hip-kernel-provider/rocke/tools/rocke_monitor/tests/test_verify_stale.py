# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The stuck-verification alarms (rkmon_lib/verify_stale.py): a review handed to verification that nothing posts gets
its VERIFY_STALE line past its limit, at most once an hour per PR and head; the line names the review's head and
`rk review-close N`, for a head our account posted a review at already that stands for ours, which a mirror's line
leaves out; a split review's limit grows with its waves; the idle tick's timer falls due with the next line; the tick's
review step adds the alarm after the advance; a review kept unposted, skipped, posted or dropped waits for nothing.
The stall: a review still waiting on verification an hour past its own limit (4 hours for a review that is not split)
gets the tick's critical line, once for each review that stalls, which the session gets whatever triages it; one line
on the Needs-you list, whose Teams copy is one message when reviews start to stall, then quiet while any review is
still stalled, kept unsent in standby; and a line of `rk status`. Nothing below 4 hours, nor while the monitor is
paused or stopped or its review automation is off; the idle tick's timer falls due with a stall until its line told
it. The skill's card for the line, and the command it records a refused tool with: the session's guard lets it
through, and it writes the day's log. The states and times are made up.
"""

import copy
import json
import os
import pathlib
import re
import shlex
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    control,
    daylog,
    idle,
    needs_teams,
    orchestrator,
    paths,
    post,
    report,
    split,
    state_store,
    tickfilter,
    triage,
    verify_stale,
)
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


def stalled(prs: str, words: str) -> str:
    """The stall's critical line for the reviews PRS, the stall in WORDS."""
    return f"!! CRITICAL VERIFY_STALLED {prs}: {words}"


ONE = "1 review has waited on verification for over 4 hours (#12, for {age})"


def ago(minutes: float) -> str:
    """`minutes` before now."""
    return idle.plus(now(), -minutes)


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

    def test_a_split_review_stalls_an_hour_past_its_own_limit(self):
        """Its VERIFY_STALE line comes at its limit, an hour before the stall's line."""
        self.cfg["features"] = {"review_automation": {"enabled": True}}
        self.st["enabled"] = True
        self.plan("fr", 3, [[1, 2], [3]], 10)
        self.assertEqual(self.lines(540), [stale(12, "9.0 h")])
        self.assertEqual(verify_stale.stalled_lines(self.cfg, self.st, at(540)), [])
        self.assertEqual(verify_stale.stalled_lines(self.cfg, self.st, at(599)), [])
        self.assertEqual(
            verify_stale.stalled_lines(self.cfg, self.st, at(600)),
            [stalled("#12", ONE.format(age="10.0 h"))],
        )

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


class TestStall(Base):
    """The stall's critical line, while the monitor is on and its review pipeline verifies."""

    def setUp(self):
        super().setUp()
        self.cfg["features"] = {"review_automation": {"enabled": True}}
        self.st["enabled"] = True

    def stall(self, minutes: float) -> list[str]:
        return verify_stale.stalled_lines(self.cfg, self.st, at(minutes))

    def due(self, when: str, cfg: dict | None = None, st: dict | None = None) -> dict:
        timers = {"verify stalled": idle.TIMERS["verify stalled"]}
        return idle.deadlines(
            cfg or self.cfg, self.st if st is None else st, None, timers, at=when
        )

    def test_a_review_waiting_over_4_hours_gets_the_critical_line_once(self):
        self.assertEqual(verify_stale.STALL_MIN, 4 * 60)
        self.assertEqual(self.stall(239), [])
        self.assertEqual(
            self.stall(240),
            [
                "!! CRITICAL VERIFY_STALLED #12: 1 review has waited on verification for over 4 hours (#12, for "
                "4.0 h)"
            ],
        )
        for m in (250, 300, 600, 24 * 60):  # a repeat: the same stall
            self.assertEqual(self.stall(m), [], m)
        self.assertEqual(self.st[verify_stale.STALLED_KEY], [12])

    def test_another_review_that_stalls_brings_the_line_again_naming_each_stalled_one(
        self,
    ):
        self.st["review_ready"].append(ready(13, "c" * 40, at(60)))
        self.assertEqual(self.stall(240), [stalled("#12", ONE.format(age="4.0 h"))])
        self.assertEqual(self.stall(299), [])
        self.assertEqual(
            self.stall(300),
            [
                stalled(
                    "#12 #13",
                    "2 reviews have waited on verification for over 4 hours (the oldest, #12, for 5.0 h)",
                )
            ],
        )
        self.assertEqual(self.stall(310), [])

    def test_a_review_stalling_again_at_a_new_head_is_told_again_and_a_posted_one_goes(
        self,
    ):
        self.assertEqual(len(self.stall(240)), 1)
        # The head moved: the review was dropped, picked again and handed to verification at 250, as an entry of its
        # own. The next full tick finds it not stalled: its PR leaves the record, and the idle tick's timer falls due
        # with its own stall.
        self.st["review_ready"] = [ready(head=NEW_HEAD, when=at(250))]
        self.assertEqual(self.stall(260), [])
        self.assertNotIn(verify_stale.STALLED_KEY, self.st)
        self.assertEqual(self.due(at(260)), {"verify stalled #12": at(490)})
        self.assertEqual(self.stall(489), [])
        self.assertEqual(self.stall(490), [stalled("#12", ONE.format(age="4.0 h"))])
        self.st["review_ready"] = []  # posted
        self.assertEqual(self.stall(500), [])
        self.assertNotIn(verify_stale.STALLED_KEY, self.st)

    def test_nothing_while_paused_or_stopped_or_with_review_automation_off(self):
        """A pause or a stop says why nothing is verified; with review automation off, nothing verifies."""
        for case in ("paused", "stopped", "review automation off"):
            with self.subTest(case):
                cfg, st = copy.deepcopy(self.cfg), copy.deepcopy(self.st)
                if case == "paused":
                    st["enabled"] = False  # rk pause
                elif case == "stopped":
                    st["stopped"] = {"at": T0, "how": "stop"}  # rk stop
                else:
                    cfg["features"]["review_automation"]["enabled"] = False
                self.assertEqual(verify_stale.stalled_lines(cfg, st, at(600)), [])
                self.assertEqual(verify_stale.stalled_text(cfg, st, at(600)), "")
                self.assertEqual(self.due(T0, cfg, st), {})
        # On again: the stall is told.
        self.assertEqual(len(self.stall(600)), 1)

    def test_the_idle_tick_s_timer_falls_due_with_a_stall_until_its_line_told_it(self):
        self.assertEqual(self.due(T0), {"verify stalled #12": at(240)})
        self.assertEqual(self.stall(239), [])
        # Due after the review step: kept, so that the next tick is full and tells it.
        self.assertEqual(self.due(at(241)), {"verify stalled #12": at(240)})
        self.stall(240)
        self.assertEqual(self.due(at(240)), {})

    def test_the_tick_s_review_step_adds_the_stall_after_the_stale_lines(self):
        self.st["review_ready"] = [ready(12, when=ago(250))]
        with mock.patch.object(rkmon.review, "advance", return_value=[]):
            out = rkmon.review_step(self.cfg, self.st, None, [])
        self.assertEqual(len(out), 2, out)
        self.assertTrue(out[0].startswith("!! VERIFY_STALE #12: "), out)
        self.assertRegex(
            out[1],
            r"^!! CRITICAL VERIFY_STALLED #12: 1 review has waited on verification for over 4 hours \(#12, for "
            r"4\.\d h\)$",
        )
        self.st.pop(verify_stale.STALLED_KEY)
        with mock.patch.object(
            rkmon.review, "advance", return_value=["REVIEW_START #14"]
        ), mock.patch.object(
            verify_stale, "stalled_lines", side_effect=ValueError("bad")
        ):
            self.assertEqual(
                rkmon.review_step(self.cfg, self.st, None, []),
                ["REVIEW_START #14", "(stalled verifications not checked: bad)"],
            )

    def test_the_session_gets_the_line_whatever_triages_it(self):
        line = self.stall(240)[0]
        # The triage's copy of how the line starts (it stays light: features imports it).
        self.assertIn(verify_stale.STALLED_LINE, triage.CRITICAL_CLAUDE)
        for acting in (False, True):  # incidents filed or not
            self.assertEqual(triage.classify(line, acting), triage.CLAUDE)
        # The tick filter's row for it: the session acts on it each time it comes, as the skill's card says; the row
        # points at the card.
        cls, what = tickfilter.classify(line)
        self.assertEqual(cls, tickfilter.ACT)
        self.assertIn("handle each as the tick procedure says", what)


class TestStallTold(unittest.TestCase):
    """The stall on the Needs-you list, its Teams copy and `rk status`, in the monitor's default configuration (review
    automation on)."""

    LINE = "2 reviews have waited on verification for over 4 hours (the oldest, #12, for 9.2 h)"

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._d.name)
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(repo="acme/widgets", operator={"github_login": "op-login"})
        c["paths"] = dict(c["paths"], working_dir=str(self.d / "work"))
        self.cfg = config.normalize(c)
        self.store = Store(self.d / "state")
        self.st = empty_state()
        self.st["enabled"] = True
        self.st["develop"]["sha"] = "d" * 40
        self.st["review_ready"] = [
            ready(12, when=ago(9 * 60 + 12)),
            ready(13, "c" * 40, ago(5 * 60)),
            ready(14, "e" * 40, ago(60)),
        ]
        home = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.d / "home")})
        home.start()
        self.addCleanup(home.stop)

    def tearDown(self):
        self._d.cleanup()

    def board(self) -> dict:
        return report.board(self.cfg, self.st, self.store, [], [])

    def told(self, b: dict) -> list:
        return [t for t in b["needs"] if "waited on verification" in t] + [
            a for a in b["asks"] if a["kind"] == verify_stale.STALLED
        ]

    def alerts(self) -> list[dict]:
        """Needs-you alerts on, live, with a webhook; the posts they make."""
        self.cfg["features"]["alerts"] = dict(
            self.cfg["features"]["alerts"], enabled=True, mode="live", needs_you=True
        )
        hook = self.d / "hook"
        hook.write_text("https://hooks.example.invalid/workflows/x\n", encoding="utf-8")
        self.cfg["teams"] = dict(
            self.cfg["teams"], channel="Chan", webhook_file=str(hook)
        )
        posted = []
        patch = mock.patch.object(
            post, "_teams_post", lambda cfg, payload: posted.append(payload) or 202
        )
        patch.start()
        self.addCleanup(patch.stop)
        return posted

    def test_needs_you_has_one_line_with_the_count_and_the_oldest_wait(self):
        b = self.board()
        self.assertIn(self.LINE, b["needs"])
        (a,) = [a for a in b["asks"] if a["kind"] == verify_stale.STALLED]
        self.assertEqual(
            (a["key"], a["text"], a["pr"]), ("VERIFY_STALLED:-:", self.LINE, None)
        )

    def test_the_board_shows_again_for_another_stalled_review_not_for_a_longer_wait(
        self,
    ):
        fp = self.board()["fp"]
        self.st["review_ready"][0]["at"] = ago(9 * 60 + 30)
        self.assertEqual(self.board()["fp"], fp)
        self.st["review_ready"][2]["at"] = ago(4 * 60 + 1)
        self.assertNotEqual(self.board()["fp"], fp)

    def test_none_below_4_hours_or_while_paused(self):
        self.st["review_ready"] = [ready(12, when=ago(239))]
        self.assertEqual(self.told(self.board()), [])
        self.st["review_ready"] = [ready(12, when=ago(600))]
        self.st["enabled"] = False  # rk pause
        self.assertEqual(self.told(self.board()), [])

    def test_the_teams_copy_is_one_message_when_reviews_start_to_stall(self):
        """Then quiet while any review is still stalled; once none is, a review that stalls brings one again."""
        posted = self.alerts()

        def tick() -> list[str]:
            return needs_teams.tick(self.cfg, self.st, self.store, self.board())

        self.assertEqual(
            tick(), ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"]
        )
        self.assertIn(
            f"**VERIFY_STALLED** · {self.LINE}",
            [b.get("text") for b in posted[0]["attachments"][0]["content"]["body"]],
        )
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(60))  # past the gap
        self.st["review_ready"][2]["at"] = ago(4 * 60 + 1)  # a third stalls: quiet
        self.assertEqual(tick(), [])
        self.st["review_ready"] = []  # verified: none is stalled
        self.assertEqual(tick(), [])
        self.st["review_ready"] = [ready(15, "f" * 40, ago(300))]  # one stalls
        self.assertEqual(
            tick(), ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"]
        )
        self.assertEqual(len(posted), 2)

    def test_in_standby_the_teams_copy_is_kept_unsent_and_the_stall_still_told(self):
        posted = self.alerts()
        state_store.set(self.d / "home", "standby.on", True)
        (line,) = needs_teams.tick(self.cfg, self.st, self.store, self.board())
        self.assertTrue(
            line.startswith("NEEDS_YOU_TEAMS UNSENT: the Needs-you message (1 item)"),
            line,
        )
        self.assertEqual(posted, [])
        (item,) = state_store.unsent_items(self.store.dir)
        self.assertIn(self.LINE, item["content"])
        # Standby verifies: the list and the session still have it.
        self.assertIn(self.LINE, self.board()["needs"])
        self.assertEqual(len(verify_stale.stalled_lines(self.cfg, self.st)), 1)

    def test_rk_status_says_it_under_the_reviews_awaiting_verification(self):
        lines = control.status_lines(self.cfg, self.st, self.store, web_running=False)
        i = next(
            k for k, x in enumerate(lines) if x.startswith("  awaiting verification: ")
        )
        self.assertEqual(lines[i + 1], f"  STALLED: {self.LINE}")
        self.st["enabled"] = False  # rk pause
        lines = control.status_lines(self.cfg, self.st, self.store, web_running=False)
        self.assertFalse([x for x in lines if "STALLED" in x], lines)


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


class TestStallCard(unittest.TestCase):
    """The session's card for the stall's line, in the skill's tick procedure, and the command it records a refused
    tool with."""

    def card(self, line: str = "!! CRITICAL VERIFY_STALLED") -> str:
        """The skill's card for the tick line that starts with LINE, its words on one line."""
        skill = (ROOT / "agentic" / "skills" / "rocke-monitor" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        card = skill.split(f"`{line}", 1)[1].split("\n   - **", 1)[0]
        return " ".join(card.split())

    def test_each_stalled_review_is_handled_as_its_verify_stale_card_says(self):
        """The stall starts no second verification: a verifier still running with its notes changing is left running,
        as the VERIFY_STALE card leaves it."""
        self.assertIn(
            "Handle each now, as the card for its VERIFY_STALE line above says: post it if its verifier handed back, "
            "start that verifier again where it is gone or stuck, and leave one that is still running with its notes "
            "changing",
            self.card(),
        )
        stale = self.card("!! VERIFY_STALE #N")
        self.assertIn("Still running, its notes still changing", stale)
        self.assertIn("nothing to do", stale)
        self.assertIn("Handed back with its post: check and post it", stale)
        self.assertIn(
            "Gone or stuck: stop it if it runs (`TaskStop`), start that one again",
            stale,
        )

    def test_a_refused_tool_holds_no_review_and_is_recorded(self):
        self.assertIn(
            "Never hold a review because a tool was refused: record the refusal with "
            '`rk log refused N "<tool> <path>"`, which writes the day\'s log that <user_login> reads',
            self.card(),
        )

    def test_the_session_may_run_the_record_and_rk_writes_it_in_the_day_s_log(self):
        """The card's command, filled in as the session runs it: the session's guard lets it through, a path in a
        shared build too, and rk takes its words as they are: an entry of kind refused for the PR, with the tool and
        the path, in the day's log."""
        # A shared build (workguard.shared_paths), which the agents only read.
        build = "/work/workers/pr12/checks/build"
        (command,) = re.findall(r"`(rk log [^`]+)`", self.card())
        command = command.replace(" N ", " 12 ").replace(
            "<tool> <path>", f"Write {build}/notes.md"
        )
        self.assertIsNone(orchestrator.guard(command, shared=[build]))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        c = config.load_defaults()
        c.pop("_doc", None)
        c["paths"] = dict(c["paths"], working_dir=str(d / "work"))
        cfg, store, printed = config.normalize(c), Store(d / "state"), []
        with mock.patch.dict(
            os.environ, {"RKMON_HOME": str(d / "home")}
        ), mock.patch.object(rkmon, "load_config", lambda: cfg), mock.patch.object(
            rkmon, "Store", lambda: store
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(shlex.split(command)[1:]), 0)
        self.assertEqual(printed, ["logged refused for 12"])
        (entry,) = daylog.entries(store)
        self.assertEqual(
            (entry["kind"], entry["pr"], entry["text"]),
            ("refused", 12, f"Write {build}/notes.md"),
        )
        day = pathlib.Path(paths.get(cfg, "daily_dir")) / f"rocke_{entry['ts'][:10]}.md"
        self.assertIn(
            f"| refused | #12 | Write {build}/notes.md |",
            day.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
