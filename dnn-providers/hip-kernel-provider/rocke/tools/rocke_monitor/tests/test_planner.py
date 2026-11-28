# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The optional model roles beside the reviewer. The planner: its proposal is the
merge plan when it keeps every rule of the toolkit's own plan as a hard constraint (no conflict inside a wave, each
wave's union clean on develop, the priorities' places, hubs last, the rebase-first bar, and the conflict evidence a PR
kept out of its wave needs); otherwise the toolkit's own plan stands and says why. The analyst: its reading of the
per-PR insights may cite only numbers the statistics hold. The settings, the background job and its answer, the shared
rate budget, rk doctor's rows, and the silent fallback when a role is off, down or wrong. Fake gateway only: no real
model is called; every name and value here is invented."""

import datetime as dt
import itertools
import json
import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "fakes"))
sys.dont_write_bytecode = True

import fake_models  # noqa: E402
import test_mergeplan as tm  # noqa: E402  (the plan's fixtures)
from rkmon_lib import (
    analyst,
    config,
    doctor,
    jobs,
    mergeplan,
    planner,
    registry,
    report,
    roles,
    webdash,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

PLAN_MODEL = "Plan.Model-1"
WORKER = {
    "backend": "openai",
    "url": "https://models.example.invalid/route",
    "model": "Big.Model-v5.3-Flash",
}
ON = dict(tm.CFG, worker=WORKER, models={"planner": {"model": PLAN_MODEL}})
AT = "2026-11-05T10:00:00Z"


def answer(st: dict, p: dict, waves, kept=None, ok=True, error=""):
    """The planner's last answer for the facts `p`, as roles.poll leaves it."""
    last = {"fp": p["fp"], "at": AT, "ok": ok, "model": PLAN_MODEL}
    last.update(
        {"answer": {"waves": waves, "kept_out": kept or {}, "why": "fewer rebases"}}
        if ok
        else {"error": error}
    )
    roles.state(st, "planner")["last"] = last


def aside(p: dict, why: str) -> bool:
    """The plan says the planner's proposal was set aside, and why."""
    return p["planner"].startswith(planner.SET_ASIDE + why)


def plan(cfg: dict, st: dict, rows: list, M: dict) -> tuple[dict, dict]:
    """The plan, and the planner's facts it computed."""
    p = mergeplan.build(cfg, st, rows, M)
    return p, mergeplan.PLANNER["payload"]


class TestPlannerProposal(unittest.TestCase):
    """Priorities #1; ready #3 and R1 #2 conflict: the toolkit's waves are [[1, 3], [2]]."""

    def setUp(self):
        self.st = tm.state(1, 2, 3)
        self.st["develop_fixes"] = {"1": tm.PRIO}
        self.M = tm.matrix([1, 2, 3], {(2, 3): ["rk/a.py"]})

    def rows(self):
        return [tm.row(1, g3=tm.FAIL), tm.row(2, g2=tm.PEND), tm.row(3)]

    def test_off_the_toolkit_plans_alone(self):
        p, facts = plan(tm.CFG, self.st, self.rows(), self.M)
        self.assertEqual(
            (p["waves"], p["made_by"], p["planner"], facts),
            ([[1, 3], [2]], "toolkit", "", None),
        )
        self.assertEqual(mergeplan.made_by_line(p), "")

    def test_a_proposal_that_keeps_every_rule_is_the_plan(self):
        p, facts = plan(ON, self.st, self.rows(), self.M)
        self.assertEqual(
            (p["made_by"], p["planner"]),
            ("toolkit", "the planner has no proposal for these facts yet"),
        )
        self.assertEqual(facts["baseline"], [[1, 3], [2]])
        self.assertEqual(facts["admitted"], {"1": 0, "2": 1, "3": 0})
        # #3 is kept out of wave 1, where the rules place it, by #2, which the proposal lands first: a conflict the
        # facts hold. #2 lands earlier than the rules place it: always allowed.
        answer(
            self.st,
            facts,
            [[1, 2], [3]],
            {"3": {"conflicts_with": 2, "evidence": "a.py"}},
        )
        rows = self.rows()
        p, _ = plan(ON, self.st, rows, self.M)
        self.assertEqual((p["waves"], p["made_by"]), ([[1, 2], [3]], "planner"))
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {1: (0, True), 2: (1, True), 3: (2, False)},
        )
        self.assertEqual(
            p["advice"]["3"], "wait: #2 lands first (conflicts in a.py), then rebase"
        )
        line = f"Made by the planner: its proposal of {AT} ({PLAN_MODEL}) keeps every rule of the plan."
        self.assertEqual(mergeplan.made_by_line(p), line)
        st = dict(self.st, merge_plan=p)
        self.assertIn(line, "\n".join(report.plan_md(ON, st)))
        self.assertIn(line, webdash.plan_card(ON, st))
        self.assertEqual(
            mergeplan.summary(p), "wave 1 #1 #2; 2 waves; made by the planner"
        )
        self.assertEqual(report.plan_row(ON, st, p)[1], "2 waves; made by the planner")
        # The same facts on the next tick (a size changed: no new answer due) keep it; a new conflict does not.
        rows = self.rows()
        rows[0]["size"] = 120
        self.assertEqual(plan(ON, self.st, rows, self.M)[0]["made_by"], "planner")
        p, _ = plan(
            ON,
            self.st,
            self.rows(),
            tm.matrix([1, 2, 3], {(2, 3): ["rk/a.py"], (1, 2): ["rk/b.py"]}),
        )
        self.assertEqual(
            (p["made_by"], p["planner"]),
            ("toolkit", "the planner has no proposal for these facts yet"),
        )

    def test_a_proposal_that_breaks_a_rule_is_set_aside_and_says_why(self):
        _, facts = plan(ON, self.st, self.rows(), self.M)
        cases = [
            (
                [[1, 2], [3]],
                {},
                "#3, which the rules place in wave 1, lands in wave 2 without a conflict the facts "
                "hold with a PR of wave 1",
            ),
            (
                [[1, 2], [3]],
                {"3": {"conflicts_with": 1}},
                "#3, which the rules place in wave 1, lands in wave 2",
            ),
            (
                [[1, 2, 3]],
                {},
                "its wave 1 has #2 and #3, which conflict (textual: a.py)",
            ),
            ([[1, 3]], {}, "it leaves #2 unplaced"),
            ([[1, 3], [2, 9]], {}, "it places #9, which are not to be placed"),
            ([[1, 3], [2, 3]], {}, "it places #3 twice"),
            ([[1, 3], [], [2]], {}, "it has an empty wave"),
            ([["one", 3], [2]], {}, "its waves are not lists of PR numbers"),
        ]
        for waves, kept, why in cases:
            answer(self.st, facts, waves, kept)
            p, _ = plan(ON, self.st, self.rows(), self.M)
            self.assertEqual(
                (p["waves"], p["made_by"]), ([[1, 3], [2]], "toolkit"), waves
            )
            self.assertTrue(aside(p, why), (waves, p["planner"]))
            self.assertTrue(
                mergeplan.made_by_line(p).startswith(
                    f"Made by the toolkit: the planner's proposal was "
                    f"set aside: {why}"
                ),
                mergeplan.made_by_line(p),
            )
        # "#3" for 3: a model may write a PR number either way.
        answer(self.st, facts, [["#1", "#2"], [3]], {"#3": {"conflicts_with": "#2"}})
        self.assertEqual(
            plan(ON, self.st, self.rows(), self.M)[0]["waves"], [[1, 2], [3]]
        )

    def test_no_usable_answer_leaves_the_toolkits_plan(self):
        _, facts = plan(ON, self.st, self.rows(), self.M)
        answer(self.st, facts, None, ok=False, error="no answer within 10 min")
        p, _ = plan(ON, self.st, self.rows(), self.M)
        self.assertEqual(
            (p["made_by"], p["planner"]),
            ("toolkit", "the planner failed: no answer within 10 min"),
        )
        r = roles.state(self.st, "planner")
        r.update(
            last={"fp": "elsewhere", "ok": True},
            job={"id": "role-planner-1"},
            fp=facts["fp"],
        )
        self.assertEqual(
            plan(ON, self.st, self.rows(), self.M)[0]["planner"],
            "the planner's proposal for these facts is on its way",
        )
        with mock.patch("rkmon_lib.planner.check", side_effect=RuntimeError("boom")):
            answer(self.st, facts, [[1, 3], [2]])
            p, _ = plan(ON, self.st, self.rows(), self.M)
        self.assertEqual(
            (p["waves"], p["made_by"], p["planner"]),
            ([[1, 3], [2]], "toolkit", "the planner's proposal was not used: boom"),
        )


class TestPlannerRules(unittest.TestCase):
    """Each rule of the toolkit's plan, as the planner's hard constraint."""

    def check(self, cfg, st, rows, M, waves, kept=None):
        _, facts = plan(cfg, st, rows, M)
        answer(st, facts, waves, kept)
        p, _ = plan(cfg, st, rows, M)
        return p

    def test_the_priorities_places(self):
        st = tm.state(1, 2, 3)
        st["develop_fixes"] = {str(n): tm.PRIO for n in (1, 2, 3)}
        rows = [tm.row(n, g3=tm.FAIL) for n in (1, 2, 3)]
        M = tm.matrix([1, 2, 3], {(1, 2): ["rk/a.py"], (2, 3): ["rk/b.py"]})
        self.assertEqual(plan(ON, st, rows, M)[0]["waves"], [[1, 3], [2]])
        p = self.check(ON, st, rows, M, [[1], [2], [3]])
        self.assertTrue(
            aside(
                p,
                "the priority #3 lands in wave 3, though wave 1 has no higher priority it "
                "conflicts with",
            ),
            p["planner"],
        )
        p = self.check(
            ON,
            st,
            rows,
            M,
            [[2], [1, 3]],
            {"1": {"conflicts_with": 2}, "3": {"conflicts_with": 2}},
        )
        self.assertTrue(
            aside(
                p,
                "the priority #1 lands in wave 2, though wave 1 has no higher priority it "
                "conflicts with",
            ),
            p["planner"],
        )
        # A lower priority clear of the higher ones may land ahead of one it conflicts with, as in the toolkit's own
        # waves; their order in a wave is the planner's.
        p = self.check(ON, st, rows, M, [[3, 1], [2]])
        self.assertEqual((p["waves"], p["made_by"]), ([[3, 1], [2]], "planner"))

    def test_no_other_pr_ahead_of_a_priority_it_conflicts_with(self):
        st = tm.state(1, 2, 3, 4)
        st["develop_fixes"] = {"1": tm.PRIO, "2": tm.PRIO}
        rows = [tm.row(1, g3=tm.FAIL), tm.row(2, g3=tm.FAIL), tm.row(3), tm.row(4)]
        M = tm.matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (2, 4): ["rk/b.py"]})
        self.assertEqual(plan(ON, st, rows, M)[0]["waves"], [[3, 1], [2], [4]])
        p = self.check(ON, st, rows, M, [[3, 1, 4], [2]])
        self.assertTrue(
            aside(p, "#4 lands ahead of the priority #2, which it conflicts with"),
            p["planner"],
        )

    def test_a_top_priority_that_must_rebase_first_ends_its_wave(self):
        """It is a priority like the others, placed by the files and ABI lanes it shares (test_mergeplan's scenario:
        #9 rebases first; #3 shares s.py with it, #6 its ABI lane)."""
        files = {
            1: {"rk/f1.py"},
            3: {"rk/s.py", "rk/f3.py"},
            4: {"rk/f4.py"},
            6: {"rk/f6.py"},
            9: {"rk/s.py"},
        }
        st = tm.state(1, 3, 4, 6, 9)
        st["develop_fixes"] = {"1": tm.PRIO, "9": tm.PRIO}
        tm.bumps(st, (6, "binary", 7, "h6"), (9, "binary", 7, "h9"))
        rows = [
            tm.row(1, g3=tm.FAIL),
            tm.row(3),
            tm.row(4, g2=tm.PEND),
            tm.row(6),
            tm.row(9, g3=tm.FAIL),
        ]
        M = tm.matrix([1, 3, 4, 6], dev=[9], files=files)
        p, facts = plan(ON, st, rows, M)
        self.assertEqual(
            (p["waves"], facts["baseline"], facts["rebase"], facts["prio"]),
            ([[1, 4], [3, 6]], [[1, 4, 9], [3, 6]], [9], [1, 9]),
        )
        self.assertIn(
            [3, 9, "shares s.py with a PR that must rebase first"], facts["edges"]
        )
        self.assertIn(
            [6, 9, "an ABI lane with a PR that must rebase first"], facts["edges"]
        )
        for waves, why in (
            (
                [[1, 9, 4], [3, 6]],
                "#4 lands after #9 in wave 1, which must rebase first and ends its "
                "wave",
            ),
            (
                [[1, 4, 3, 9], [6]],
                "its wave 1 has #3 and #9, which conflict (shares s.py with a PR "
                "that must rebase first)",
            ),
            (
                [[1, 4, 3], [9], [6]],
                "#3 lands ahead of the priority #9, which it conflicts with",
            ),
        ):
            p = self.check(ON, st, rows, M, waves)
            self.assertTrue(aside(p, why), (waves, p["planner"]))
        p = self.check(ON, st, rows, M, [[4, 1, 9], [6, 3]])
        self.assertEqual(
            (p["waves"], p["made_by"], p["dev_conflicts"]["9"]["wave"]),
            ([[4, 1], [6, 3]], "planner", 1),
        )
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {4: (0, True), 1: (1, True), 9: (2, False), 6: (3, False), 3: (4, False)},
        )
        # Two that share a file: #8 ends wave 1, #9 wave 2; the higher priority takes the earlier wave.
        files = {1: {"rk/f1.py"}, 8: {"rk/s.py", "rk/f8.py"}, 9: {"rk/s.py"}}
        st = tm.state(1, 8, 9)
        st["develop_fixes"] = {"8": tm.PRIO, "9": tm.PRIO}
        rows = [tm.row(1), tm.row(8, g3=tm.FAIL), tm.row(9, g3=tm.FAIL)]
        M = tm.matrix([1], dev=[8, 9], files=files)
        self.assertEqual(plan(ON, st, rows, M)[1]["baseline"], [[1, 8], [9]])
        p = self.check(ON, st, rows, M, [[1, 9], [8]], {"8": {"conflicts_with": 9}})
        self.assertTrue(
            aside(
                p,
                "the priority #8 lands in wave 2, though wave 1 has no higher priority it "
                "conflicts with",
            ),
            p["planner"],
        )

    def test_hubs_land_after_every_pr_but_the_priorities(self):
        ns = [1, 10, 11, 12, 13, 14, 15]
        st = tm.state(*ns)
        st["develop_fixes"] = {"1": tm.PRIO}
        rows = [tm.row(1, g3=tm.FAIL), tm.row(10)] + [
            tm.row(n, g2=tm.PEND) for n in range(11, 16)
        ]
        M = tm.matrix(ns, {(10, m): ["rk/core.py"] for m in range(11, 16)})
        self.assertEqual(
            plan(ON, st, rows, M)[0]["waves"], [[1, 11, 12, 13, 14, 15], [10]]
        )
        p = self.check(
            ON,
            st,
            rows,
            M,
            [[1, 10], [11, 12, 13, 14, 15]],
            {str(n): {"conflicts_with": 10} for n in range(11, 16)},
        )
        self.assertTrue(
            aside(
                p,
                "#11, #12, #13, #14, #15 land with or after a hub (wave 1); hubs land after every "
                "PR but the priorities",
            ),
            p["planner"],
        )

    def test_a_pinned_group_stays_together_in_its_order(self):
        st = tm.state(1, 2, 3, 4)
        st["groups"] = [[1, 2]]
        rows = [tm.row(n) for n in (1, 2, 3, 4)]
        M = tm.matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (3, 4): ["rk/b.py"]})
        self.assertEqual(plan(ON, st, rows, M)[0]["waves"], [[1, 2, 3], [4]])
        p = self.check(ON, st, rows, M, [[1, 3, 2], [4]])
        self.assertTrue(
            aside(
                p, "the pinned group #1, #2 is not together in one wave, in its order"
            ),
            p["planner"],
        )
        # The group's own conflict is the user's call, not a conflict inside the wave; #4 may take #3's place.
        p = self.check(ON, st, rows, M, [[3, 1, 2], [4]])
        self.assertEqual((p["waves"], p["made_by"]), ([[3, 1, 2], [4]], "planner"))
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows}, {1: True, 2: False, 3: True, 4: False}
        )
        p = self.check(
            ON,
            st,
            rows,
            M,
            [[1, 2, 4], [3]],
            {"3": {"conflicts_with": 4, "evidence": "b.py"}},
        )
        self.assertEqual((p["waves"], p["made_by"]), ([[1, 2, 4], [3]], "planner"))

    def test_the_toolkits_own_waves_always_keep_the_rules(self):
        """Random plans (as test_mergeplan's): the toolkit's waves pass the planner's check, and as the planner's
        answer they give the same plan: waves, ranks, what may land now, advice."""
        rng = random.Random(1105)
        gates = {
            "T0": {"g3": tm.FAIL},
            "R0": {},
            "R1": {"g2": tm.PEND},
            "R3": {"g1": ["fail", f"our verdict {tm.O}"]},
            "R4": {"g1": ["fail", f"our verdict {tm.R}"]},
        }
        for _ in range(120):
            ns = list(range(1, rng.randint(2, 11) + 1))
            cls = {
                n: rng.choice(("T0", "T0", "R0", "R1", "R1", "R3", "R4")) for n in ns
            }
            held = {n for n in ns if cls[n] == "R4" and rng.random() < 0.4}
            dev = {n for n in ns if cls[n] == "T0" and rng.random() < 0.3}
            clean = [n for n in ns if n not in dev]
            files = {
                n: {f"rk/f{n}.py"}
                | {f"rk/s{k}.py" for k in (1, 2) if rng.random() < 0.25}
                for n in ns
            }
            pairs = {
                (a, b): rng.choice((["rk/x.py"], ["rk/x.py"], None))
                for a, b in itertools.combinations(clean, 2)
                if rng.random() < 0.35
            }
            st = tm.state(*ns)
            st["develop_fixes"] = {str(n): tm.PRIO for n in ns if cls[n] == "T0"}
            if len(clean) > 3 and rng.random() < 0.3:
                st["groups"] = [rng.sample(clean, 2)]

            def rows():
                return [
                    tm.row(
                        n,
                        size=10 * n,
                        **gates[cls[n]],
                        **({"held": {"status": "held"}} if n in held else {}),
                    )
                    for n in ns
                ]

            M = tm.matrix(clean, pairs, dev=sorted(dev), files=files)
            r1 = rows()
            p, facts = plan(ON, st, r1, M)
            ctx = (cls, sorted(held), sorted(dev), pairs, st.get("groups"), p["waves"])
            self.assertEqual(
                planner.check(facts, {"waves": facts["baseline"]}), [], ctx
            )
            answer(st, facts, facts["baseline"])
            r2 = rows()
            q, _ = plan(ON, st, r2, M)
            self.assertEqual(q["made_by"], "planner", (ctx, q["planner"]))
            self.assertEqual(
                (q["waves"], q["advice"], q["nodes"], q["dev_conflicts"]),
                (p["waves"], p["advice"], p["nodes"], p["dev_conflicts"]),
                ctx,
            )
            self.assertEqual(
                [(r["plan_rank"], r["plan_go"]) for r in r2],
                [(r["plan_rank"], r["plan_go"]) for r in r1],
                ctx,
            )
            roles.state(st, "planner").clear()


class TestPlannerUnionCheck(unittest.TestCase):
    """The proposal's waves are union-checked as the toolkit's are (test_mergeplan's fake merges: #1, #2 and #3 conflict
    together in s.py), in the proposal's own landing order, on the tick's merge budget after the toolkit's own.
    """

    FILES = tm.TestUnionCheck.FILES
    setUp, merge_tree, plan = (
        tm.TestUnionCheck.setUp,
        tm.TestUnionCheck.merge_tree,
        tm.TestUnionCheck.plan,
    )

    def plan_on(self, st: dict, cfg: dict = ON) -> dict:
        p, _ = self.plan(st, cfg)
        return p

    def test_a_proposal_is_union_checked_in_its_own_order(self):
        st = tm.state(1, 2, 3, 4)
        p = self.plan_on(st)
        facts = mergeplan.PLANNER["payload"]
        self.assertEqual((p["waves"], facts["union"]), ([[4, 1, 2], [3]], True))
        self.assertIn(
            [1, 3, "together with the PRs before it in a wave: s.py"], facts["edges"]
        )
        answer(st, facts, [[4, 2, 1], [3]])
        self.calls.clear()
        p = self.plan_on(st)
        self.assertEqual(
            (p["waves"], p["made_by"], self.calls),
            ([[4, 2, 1], [3]], "planner", [("t2", "t1")]),
        )

    def test_a_proposal_not_checked_to_its_end_waits_for_the_next_tick(self):
        st = tm.state(1, 2, 3, 4)
        self.plan_on(st)
        answer(st, mergeplan.PLANNER["payload"], [[4, 2, 1], [3]])
        p = self.plan_on(
            st, dict(ON, merge_plan={"max_merges": 0})
        )  # the toolkit's steps are cached; no budget left
        self.assertEqual(
            (p["waves"], p["made_by"], p["partial"]),
            ([[4, 1, 2], [3]], "toolkit", None),
        )
        self.assertEqual(
            p["planner"],
            "the planner's proposal was set aside: its wave 1 does not merge clean on "
            "develop as one: "
            "not checked to the end yet: the merge budget ran out",
        )
        p = self.plan_on(st)  # the next tick has room
        self.assertEqual(
            (p["waves"], p["made_by"], p["partial"]),
            ([[4, 2, 1], [3]], "planner", None),
        )

    def test_a_wave_whose_union_conflicts_is_refused(self):
        facts = planner.payload(
            [{"pr": n, "cls": "R0", "weight": 8, "size": 1} for n in (1, 2, 3)],
            {},
            [],
            [],
            [],
            {},
            [[1, 2], [3]],
            union=True,
        )

        def union(wave):
            return (
                "#3 conflicts with #1, #2 together in s.py"
                if {1, 2, 3} <= set(wave)
                else ""
            )

        self.assertEqual(
            planner.check(facts, {"waves": [[1, 2, 3]]}, union),
            [
                "its wave 1 does not merge clean on develop as one: #3 conflicts with #1, #2 together in "
                "s.py"
            ],
        )
        self.assertEqual(planner.check(facts, {"waves": [[1, 2], [3]]}, union), [])


class TestPlannerFacts(unittest.TestCase):
    def test_the_request_states_the_rules_and_the_answers_form(self):
        st = tm.state(1, 2, 3)
        st["develop_fixes"] = {"1": tm.PRIO}
        _, facts = plan(
            ON,
            st,
            [tm.row(1, g3=tm.FAIL), tm.row(2, g2=tm.PEND), tm.row(3)],
            tm.matrix([1, 2, 3], {(2, 3): ["rk/a.py"]}),
        )
        msgs = planner.messages(facts)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        text = msgs[1]["content"]
        for part in (
            "#1 T0 weight 8 size 100; #2 R1 weight 4 size 100; #3 R0 weight 8 size 100",
            "Conflicts (two PRs listed together never land in one wave, but for two members of a pinned "
            'group): [[2, 3, "textual: a.py"]]',
            "Priorities, highest first: [1].",
            'The rules place each PR in a wave: {"1": 1, "2": 2, "3": 1}.',
            "The toolkit's own waves, which you may keep or improve: [[1, 3], [2]].",
            '"kept_out": {"PR": {"conflicts_with": PR',
        ):
            self.assertIn(part, text)
        self.assertNotIn("Hubs", text)
        self.assertEqual(len(facts["fp"]), 12)

    def test_the_planner_starts_once_per_set_of_facts(self):
        st = tm.state(1, 2)
        _, facts = plan(ON, st, [tm.row(1), tm.row(2)], tm.matrix([1, 2]))
        with mock.patch.object(
            roles, "start", return_value="ROLE_STARTED planner"
        ) as start:
            self.assertEqual(planner.start(ON, st, None, facts), "ROLE_STARTED planner")
            self.assertEqual(planner.start(tm.CFG, st, None, facts), "")  # off
            self.assertEqual(planner.start(ON, st, None, None), "")  # no plan this tick
            answer(st, facts, [[1, 2]])
            self.assertEqual(planner.start(ON, st, None, facts), "")  # answered already
        self.assertEqual(start.call_count, 1)
        self.assertEqual(start.call_args[0][3:5], ("planner", facts["fp"]))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.store = Store(self.base / "state")
        self.cfg = {
            "repo": "acme/widgets",
            "paths": {"working_dir": str(self.base / "work")},
            "worker": dict(WORKER),
            "models": {
                "planner": {"model": PLAN_MODEL, "share": 0.25, "timeout_min": 7}
            },
        }


class TestRoleSettings(Base):
    def test_off_worker_or_a_model_of_the_reviewers_endpoint(self):
        cfg = self.cfg
        self.assertEqual(
            roles.wanted(cfg, "planner"),
            {"model": PLAN_MODEL, "share": 0.25, "timeout_min": 7},
        )
        self.assertEqual(
            (roles.on(cfg, "planner"), roles.on(cfg, "analyst")), (True, False)
        )
        self.assertEqual(
            roles.describe(cfg, "planner"),
            f"{PLAN_MODEL} at the reviewer's endpoint, 25% of its rate " "limits",
        )
        self.assertEqual(
            (roles.describe(cfg, "analyst"), roles.describe({}, "planner")),
            ("off (no annotations)", "off (the toolkit's own plan)"),
        )
        s = roles.endpoint(cfg, "planner")
        w = config.worker(cfg)
        self.assertEqual(
            (s["url"], s["model"], s["budget_role"], s["budget_share"]),
            (w["url"], PLAN_MODEL, "planner", 0.25),
        )
        self.assertEqual(
            s["budget_label"], f"{w['url']} {w['model']}"
        )  # the reviewer's own budget
        cfg = dict(
            cfg, models={"planner": "worker", "analyst": {"model": " ", "share": 5}}
        )
        self.assertEqual(
            roles.wanted(cfg, "planner"),
            {"model": WORKER["model"], "share": 0.1, "timeout_min": 10},
        )
        self.assertIsNone(roles.wanted(cfg, "analyst"))  # a blank model is off
        self.assertEqual(
            roles.wanted(
                dict(cfg, models={"planner": {"model": "m", "share": 7}}), "planner"
            )["share"],
            1.0,
        )
        self.assertFalse(
            roles.on(dict(cfg, worker={}), "planner")
        )  # no reviewer endpoint: off
        self.assertIsNone(roles.endpoint(dict(cfg, worker={}), "planner"))

    def test_a_role_s_limits_are_the_code_s_where_its_map_leaves_them_out(self):
        """A role's share and time limit, where its map leaves them out, are roles.DEFAULTS', entry by entry: a
        constant of the code, which no setting names (no deployment runs these roles).
        """
        package = roles.DEFAULTS
        self.assertEqual(sorted(package), ["share", "timeout_min"])
        self.assertFalse(
            [k for k in registry.keys() if k.name.startswith("model_roles")]
        )
        self.assertNotIn("model_roles", config.load_defaults())
        cfg = dict(self.cfg, models={"planner": {"model": "m"}})
        want = roles.wanted(cfg, "planner")
        self.assertEqual(
            (want["share"], want["timeout_min"]),
            (package["share"], package["timeout_min"]),
        )
        cfg = dict(self.cfg, models={"planner": {"model": "m", "share": 0.5}})
        want = roles.wanted(cfg, "planner")
        self.assertEqual(
            (want["share"], want["timeout_min"]), (0.5, package["timeout_min"])
        )

    def test_the_analytics_model_serves_both_unless_a_role_has_its_own(self):
        """One analytics model by default; the analyst is selectable apart."""
        cfg = dict(
            self.cfg, models={"analytics": {"model": "Shared.Model-3", "share": 0.2}}
        )
        self.assertEqual(
            [roles.wanted(cfg, r)["model"] for r in roles.ROLES], ["Shared.Model-3"] * 2
        )
        self.assertEqual(
            roles.describe(cfg, "analyst"),
            "Shared.Model-3 (the analytics model) at the reviewer's "
            "endpoint, 20% of its rate limits",
        )
        cfg["models"].update(analyst={"model": "Own.Model-4"}, planner="off")
        self.assertEqual(
            (roles.wanted(cfg, "planner"), roles.wanted(cfg, "analyst")["model"]),
            (None, "Own.Model-4"),
        )
        self.assertEqual(
            roles.endpoint(cfg, "analyst")["budget_role"], "analyst"
        )  # its own share, as each role's
        self.assertEqual(
            roles.describe(cfg, roles.SHARED),
            "Shared.Model-3 at the reviewer's endpoint, 20% of its " "rate limits",
        )
        self.assertEqual(roles.describe(dict(cfg, models={}), roles.SHARED), "off")

    def test_rk_roles_shows_a_role_only_while_it_is_on(self):
        """No deploy setting turns the planner, the analyst or their analytics model on: rk roles (its table and its
        lines) leaves each out while it is off, as its help's example does; one that is on has its row and its line.
        """
        rows = [r[0] for r in roles.role_rows(self.cfg)]
        self.assertIn("planner", rows)
        self.assertFalse({"analyst", roles.SHARED} & set(rows), rows)
        lines = roles.status_lines(self.cfg, empty_state(), self.store)
        self.assertEqual([line.split(":")[0] for line in lines], ["planner"])
        off = dict(self.cfg, models={})
        self.assertFalse(
            {roles.SHARED, *roles.ROLES} & {r[0] for r in roles.role_rows(off)}
        )
        self.assertEqual(roles.status_lines(off, empty_state(), self.store), [])

    def test_the_configuration_refuses_what_a_role_cannot_be(self):
        def problems(v, role="planner"):
            return [
                x
                for x in config.problems(dict(tm.CFG, models={role: v}))
                if f"models.{role}" in x
            ]

        self.assertEqual(
            [
                problems(v)
                for v in (
                    "",
                    "off",
                    "worker",
                    {"model": "m", "share": 0.5, "timeout_min": 3},
                )
            ],
            [[], [], [], []],
        )
        self.assertIn(
            'use "" (the analytics model), "off", "worker" (the reviewer\'s own model) or '
            '{"model": ...}',
            problems("claude")[0],
        )
        self.assertIn('use "" (off), "worker"', problems("off", "analytics")[0])
        self.assertEqual(problems({"model": "m"}, "analytics"), [])
        self.assertIn("models.planner.model is blank", problems({"share": 0.2})[0])
        self.assertIn(
            "models.planner.share is 2", problems({"model": "m", "share": 2})[0]
        )
        self.assertIn(
            "models.planner.timeout_min is 0",
            problems({"model": "m", "timeout_min": 0})[0],
        )
        self.assertIn(
            "models.planner.act is reserved for a future switch and must stay false",
            problems({"model": "m", "act": True})[0],
        )

    def test_a_models_answer_holds_one_json_object(self):
        j = roles.json_answer
        self.assertEqual(j('{"waves": [[1]]}'), {"waves": [[1]]})
        self.assertEqual(
            j('Here it is:\n```json\n{"waves": [[2]]}\n```\nDone.'), {"waves": [[2]]}
        )
        self.assertEqual(j('I propose {"waves": [[3]]} as the plan.'), {"waves": [[3]]})
        for bad in ("OK", "[[1, 2]]", ""):
            with self.assertRaises(ValueError):
                j(bad)


class TestRoleChecks(Base):
    """rk doctor's row of each role (doctor.role_checks): OFF, or its model on the reviewer's endpoint, read from the
    endpoint's model list (the presence check, stubbed here), never while a review worker runs. A role blocks nothing:
    what it lacks is a WARN, with its fix."""

    BOTH = f"{PLAN_MODEL} at the reviewer's endpoint, 25% of its rate limits"
    FIX = (
        "edit model_bindings in your deploy settings file, then ./rkmon confirm and "
        "./rkmon apply"
    )

    def rows(self, cfg, listed=(), answers=True, running=False):
        """The rows, and each (address, model) the presence check was asked: the endpoint answers (HTTP 200, else
        503) with the models LISTED; RUNNING: a review worker runs."""
        asked = []

        def presence(_reviewer_client, w):
            asked.append((w["url"], w["model"]))
            return {
                "answers": answers,
                "model_found": answers and w["model"] in listed,
                "error": "" if answers else "HTTP 503",
            }

        with mock.patch.object(
            doctor, "worker_running", lambda cfg: running
        ), mock.patch.object(doctor, "_presence", presence), mock.patch.dict(
            "os.environ", {"RKMON_WRAPPER": "./rkmon"}
        ):
            rows = doctor.role_checks(cfg)
        return [(c.name, c.status, c.found, c.fix and str(c.fix)) for c in rows], asked

    def test_off_listed_or_not_and_never_while_a_review_worker_runs(self):
        off = ("analyst model", "OFF", "off (no annotations)", None)
        here = (WORKER["url"], PLAN_MODEL)  # the reviewer's endpoint, the role's model
        self.assertEqual(
            self.rows(self.cfg, listed=(PLAN_MODEL,)),
            ([("planner model", "PASS", self.BOTH, None), off], [here]),
        )
        self.assertEqual(
            self.rows(self.cfg, listed=(WORKER["model"],)),
            (
                [
                    (
                        "planner model",
                        "WARN",
                        f"{self.BOTH}; not listed there; the toolkit's own result stands meanwhile",
                        f"choose a model the endpoint lists: {self.FIX}",
                    ),
                    off,
                ],
                [here],
            ),
        )
        self.assertEqual(
            self.rows(self.cfg, answers=False)[0][0],
            (
                "planner model",
                "WARN",
                f"{self.BOTH}; no answer (HTTP 503); the toolkit's own result stands meanwhile",
                f"choose a model the endpoint lists: {self.FIX}",
            ),
        )
        self.assertEqual(
            self.rows(
                dict(self.cfg, worker=dict(WORKER, url="")), listed=(PLAN_MODEL,)
            ),
            (
                [
                    (
                        "planner model",
                        "WARN",
                        f"{PLAN_MODEL}, but the reviewer has no endpoint",
                        self.FIX,
                    ),
                    off,
                ],
                [],
            ),
        )
        self.assertEqual(
            self.rows(self.cfg, listed=(PLAN_MODEL,), running=True),
            (
                [
                    (
                        "planner model",
                        "SKIP",
                        f"{self.BOTH}: a review worker is running: not checked now",
                        None,
                    ),
                    off,
                ],
                [],
            ),
        )
        self.assertEqual(
            self.rows(dict(self.cfg, models={})),
            ([("planner model", "OFF", "off (the toolkit's own plan)", None), off], []),
        )


class TestRoleJobs(Base):
    def test_start_writes_the_request_and_runs_rk_role_run_in_the_background(self):
        st = empty_state()
        started = []

        def start(store, kind, name, script, timeout_min, env=None):
            started.append((kind, name, script, timeout_min))
            return {
                "id": f"{kind}-{name}-1",
                "pid": 1,
                "started": "2026-11-05T10:00:00Z",
                "timeout_min": timeout_min,
            }

        with mock.patch.object(jobs, "start", start), mock.patch.object(
            jobs, "status", return_value="running"
        ):
            line = roles.start(
                self.cfg,
                st,
                self.store,
                "planner",
                "f00d",
                [{"role": "user", "content": "plan"}],
            )
            self.assertEqual(
                line, f"ROLE_STARTED planner: {PLAN_MODEL} on the facts f00d"
            )
            self.assertEqual(
                roles.start(self.cfg, st, self.store, "planner", "beef", []), ""
            )  # one at a time
        ((kind, name, script, timeout),) = started
        self.assertEqual((kind, name, timeout), ("role", "planner", 7))
        src = self.base / "work" / "run" / "roles" / "planner" / "in-f00d.json"
        self.assertIn(
            f"role-run planner --input {src} --output {src.with_name('out-f00d.json')}",
            script,
        )
        self.assertEqual(
            json.loads(src.read_text(encoding="utf-8"))["messages"],
            [{"role": "user", "content": "plan"}],
        )
        self.assertEqual(roles.state(st, "planner")["fp"], "f00d")

    def test_poll_reads_the_answer_and_a_failure_is_journaled_once(self):
        st = empty_state()
        d = roles.folder(self.cfg, "planner")
        d.mkdir(parents=True)
        job = {
            "id": "role-planner-1",
            "pid": 1,
            "started": "2026-11-05T10:00:00Z",
            "timeout_min": 7,
        }
        roles.state(st, "planner").update(job=job, fp="f00d")
        (d / "out-f00d.json").write_text(
            json.dumps({"fp": "f00d", "at": AT, "ok": True, "answer": {"waves": []}}),
            encoding="utf-8",
        )
        with mock.patch.object(jobs, "status", return_value="done"):
            self.assertEqual(roles.poll(self.cfg, st, self.store), [])
        r = roles.state(st, "planner")
        self.assertEqual((r["last"]["ok"], "job" in r), (True, False))
        r["job"] = job
        with mock.patch.object(
            jobs, "status", return_value="timeout"
        ), mock.patch.object(jobs, "kill") as kill:
            lines = roles.poll(self.cfg, st, self.store)
        kill.assert_called_once()
        self.assertEqual(
            lines,
            [
                "(the planner failed: no answer within 7 min; the toolkit's own result stands)"
            ],
        )
        self.assertEqual(r["last"]["error"], "no answer within 7 min")
        self.assertEqual(roles.poll(self.cfg, st, self.store), [])  # once: no job left
        self.assertIn(
            "planner: no usable answer (no answer within 7 min); the toolkit's own result stands",
            "\n".join(self.store.journal_tail()),
        )
        lines = roles.status_lines(self.cfg, st, self.store)
        self.assertEqual(
            lines[0],
            f"planner: {PLAN_MODEL} at the reviewer's endpoint, 25% of its rate limits; last "
            f"answer {r['last']['at']}: failed: no answer within 7 min",
        )
        self.assertEqual(len(lines), 1, lines)  # the analyst, off, has no line


class TestRoleRunOnTheGateway(Base):
    """`rk role-run` on the fake gateway: the reviewer's address and key, the role's own model, one chat, the answer's
    JSON in the output file; anything else is a failure recorded there, never raised."""

    def setUp(self):
        super().setUp()
        key = self.base / "key"
        key.write_text("k-" + "fake-gateway-0042\n", encoding="utf-8")
        self.reply = {"waves": [[1, 2]], "kept_out": {}, "why": "one wave"}
        self.srv = fake_models.start(
            self.base / "fake",
            "gateway",
            key_file=str(key),
            answer_for=classmethod(lambda cls, req, tools: (self.text, None)),
        )
        self.addCleanup(self.srv.stop)
        self.text = "```json\n" + json.dumps(self.reply) + "\n```"
        self.cfg["worker"] = {
            "backend": "openai",
            "url": self.srv.url,
            "model": fake_models.GOOD,
            "api_key_file": str(key),
            "auth_header": "X-Example-Key",
            "auth_scheme": "",
            "rpm": 60,
        }
        self.cfg["models"] = {"planner": {"model": fake_models.TEXT_ONLY, "share": 0.5}}
        (self.base / "work" / "run").mkdir(parents=True)
        quiet = mock.patch("builtins.print")  # the job's one line goes to its log
        quiet.start()
        self.addCleanup(quiet.stop)
        self.src, self.out = self.base / "in.json", self.base / "out.json"
        self.src.write_text(
            json.dumps(
                {
                    "fp": "f00d",
                    "messages": planner.messages(
                        planner.payload(
                            [
                                {"pr": 1, "cls": "R0", "weight": 8, "size": 1},
                                {"pr": 2, "cls": "R0", "weight": 8, "size": 1},
                            ],
                            {},
                            [],
                            [],
                            [],
                            {},
                            [[1, 2]],
                        )
                    ),
                }
            ),
            encoding="utf-8",
        )

    def test_one_chat_with_the_roles_model_on_the_reviewers_budget(self):
        self.assertEqual(
            roles.run_job(self.cfg, "planner", str(self.src), str(self.out)), 0
        )
        rec = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(
            (rec["ok"], rec["fp"], rec["answer"], rec["model"]),
            (True, "f00d", self.reply, fake_models.TEXT_ONLY),
        )
        chats = [
            x
            for x in fake_models.read_log(self.base / "fake", "gateway")
            if x["method"] == "POST"
        ]
        self.assertEqual(
            [(x["status"], x["model"], x["tools"]) for x in chats],
            [(200, fake_models.TEXT_ONLY, False)],
        )
        (budget,) = (self.base / "work" / "run").glob("budget-*.json")
        w = config.worker(self.cfg)
        import reviewer_client

        self.assertEqual(
            budget, reviewer_client.Budget(w, str(self.base / "work" / "run")).path
        )  # the reviewer's file
        self.assertEqual(
            [x[3] for x in json.loads(budget.read_text(encoding="utf-8"))["window"]],
            ["planner"],
        )

    def test_an_answer_without_json_or_a_role_turned_off_is_a_recorded_failure(self):
        self.text = "OK"
        self.assertEqual(
            roles.run_job(self.cfg, "planner", str(self.src), str(self.out)), 0
        )
        rec = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(
            (rec["ok"], rec["error"]), (False, "the answer has no JSON object")
        )
        self.cfg["models"] = {}
        self.assertEqual(
            roles.run_job(self.cfg, "planner", str(self.src), str(self.out)), 0
        )
        self.assertEqual(
            json.loads(self.out.read_text(encoding="utf-8"))["error"],
            "the planner role is off, or the reviewer has no endpoint",
        )

    def test_the_answer_is_written_whole_and_yours_only(self):
        """The answer is a file made new beside the output, then renamed onto it (rkmon_lib/whole_file.py): a link at
        out.json.new, beside it, is never written through."""
        other = self.base / "other.json"
        other.write_text("other\n", encoding="utf-8")
        self.out.with_suffix(".json.new").symlink_to(other)
        self.assertEqual(
            roles.run_job(self.cfg, "planner", str(self.src), str(self.out)), 0
        )
        self.assertEqual(other.read_text(encoding="utf-8"), "other\n")
        self.assertFalse(self.out.is_symlink())
        self.assertTrue(json.loads(self.out.read_text(encoding="utf-8"))["ok"])
        self.assertEqual(self.out.stat().st_mode & 0o777, roles.ANSWER_MODE)

    def test_the_commands(self):
        import rkmon

        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(
                rkmon.main(
                    [
                        "role-run",
                        "planner",
                        "--input",
                        str(self.src),
                        "--output",
                        str(self.out),
                    ]
                ),
                0,
            )
            self.assertEqual(rkmon.main(["roles"]), 0)
        self.assertTrue(json.loads(self.out.read_text(encoding="utf-8"))["ok"])
        self.assertEqual(out[0], "planner: answered")
        self.assertTrue(
            out[1].startswith("pass "), out[1]
        )  # the passes' table first, then the other roles
        self.assertIn(
            f"planner: {fake_models.TEXT_ONLY} at the reviewer's endpoint, 50% of its rate limits",
            out[2],
        )


class TestTickStep(Base):
    def test_the_tick_asks_the_planner_and_journals_a_set_aside_proposal_once(self):
        import rkmon

        cfg = dict(ON, paths={"working_dir": str(self.base / "work")})
        st = tm.state(1, 2, 3)
        st["develop_fixes"] = {"1": tm.PRIO}
        rows = [tm.row(1, g3=tm.FAIL), tm.row(2, g2=tm.PEND), tm.row(3)]
        M = tm.matrix([1, 2, 3], {(2, 3): ["rk/a.py"]})
        st["merge_plan"], facts = plan(cfg, st, rows, M)
        with mock.patch.object(
            roles, "start", return_value="ROLE_STARTED planner: x on the facts y"
        ) as start:
            self.assertEqual(rkmon.role_step(cfg, st, self.store), [])
            start.assert_called_once()
            answer(st, facts, [[1, 2], [3]])  # no evidence for #3: set aside
            st["merge_plan"], _ = plan(cfg, st, rows, M)
            for _ in range(2):
                rkmon.role_step(cfg, st, self.store)
        journal = "\n".join(self.store.journal_tail())
        self.assertIn("ROLE_STARTED planner: x on the facts y", journal)
        self.assertEqual(
            journal.count(
                "the planner's proposal was set aside: #3, which the rules place in wave 1"
            ),
            1,
        )
        with mock.patch.object(
            mergeplan, "PLANNER", {"payload": None}
        ), mock.patch.object(planner, "start", side_effect=RuntimeError("boom")):
            self.assertEqual(
                rkmon.role_step(cfg, st, self.store), ["(the model roles: boom)"]
            )


STATS = {
    "days": 14,
    "start": "2026-10-22T00:00:00Z",
    "end": "2026-11-05T00:00:00Z",
    "fp": "abc",
    "categories": [
        {
            "key": "rebased",
            "title": "Rebased",
            "what": "force-pushes",
            "overall": {"rebases": 9, "prs": 3},
            "top": [
                {
                    "pr": 101,
                    "value": 5,
                    "detail": "4 force-push(es), 1 develop merge(s)",
                },
                {
                    "pr": 102,
                    "value": 3,
                    "detail": "3 force-push(es), 0 develop merge(s)",
                },
            ],
        },
        {
            "key": "runner_waits",
            "title": "CI runner waits",
            "what": "jobs",
            "overall": {"total_h": 12.5},
            "top": [
                {
                    "pr": 102,
                    "value": 340.5,
                    "detail": "12 job(s) on 2 head(s), the longest 61.2 min",
                }
            ],
            "labels": [
                {
                    "label": "linux-gpu-x86",
                    "jobs": 12,
                    "median_min": 4.5,
                    "p90_min": 9,
                    "total_min": 75,
                }
            ],
        },
    ],
}


class TestAnalyst(Base):
    def setUp(self):
        super().setUp()
        self.cfg["models"] = {"analyst": {"model": "Read.Model-2", "every_h": 12}}

    def reading(self, summary, ranking=()):
        return {"summary": summary, "ranking": list(ranking)}

    def test_a_reading_may_cite_only_what_the_statistics_hold(self):
        ok, why = analyst.check(
            STATS,
            self.reading(
                "#101 was rebased 5 times; runner waits reached 12.5 h, 340.5 min of them on #102.",
                [
                    {
                        "category": "runner_waits",
                        "pr": 102,
                        "note": "12 jobs waited, the longest 61.2 min on linux-gpu-x86",
                    },
                    {
                        "category": "rebased",
                        "pr": None,
                        "note": "9 rebases over 14 days",
                    },
                ],
            ),
        )
        self.assertEqual(why, "")
        self.assertEqual(
            ok["ranking"][0],
            {
                "category": "runner_waits",
                "pr": 102,
                "note": "12 jobs waited, the longest 61.2 min on linux-gpu-x86",
            },
        )
        refused = {
            "it cites 40, which the statistics do not hold": self.reading(
                "About 40% of the waits are on #102."
            ),
            "it cites 6, which the statistics do not hold": self.reading(
                "#101 was rebased 6 times."
            ),
            "it names PR 103, which the statistics do not": self.reading(
                "Rebases.", [{"category": "rebased", "pr": 103, "note": "x"}]
            ),
            "it ranks a category the statistics do not have (flaky)": self.reading(
                "Flaky.", [{"category": "flaky", "pr": None, "note": "x"}]
            ),
            "it is not the reading asked for (a summary and a ranking)": {
                "ranking": []
            },
        }
        for why, ans in refused.items():
            self.assertEqual(analyst.check(STATS, ans), (None, why))
        self.assertEqual(
            analyst.check(
                STATS, self.reading("5.0 rebases on #101, version 1.5.2 aside.")
            )[1],
            "",
        )

    def test_a_reading_shows_once_checked_and_a_refused_one_is_journaled_once(self):
        st = empty_state()
        r = roles.state(st, "analyst")
        r["data"] = dict(STATS)
        r["last"] = {
            "fp": "abc",
            "at": AT,
            "ok": True,
            "model": "Read.Model-2",
            "answer": self.reading(
                "Runner waits first: 12.5 h.",
                [{"category": "runner_waits", "pr": 102, "note": "340.5 min"}],
            ),
        }
        got = analyst.reading(st, self.store)
        self.assertEqual(
            (got["summary"], got["model"], got["at"]),
            ("Runner waits first: 12.5 h.", "Read.Model-2", AT),
        )
        r["last"] = dict(
            r["last"],
            at="2026-11-05T11:00:00Z",
            answer=self.reading("Waits cost 99 h."),
        )
        for _ in range(2):
            self.assertIsNone(analyst.reading(st, self.store))
        journal = "\n".join(self.store.journal_tail())
        self.assertEqual(
            journal.count(
                "analyst: its reading of 2026-11-05T11:00:00Z was refused: it cites 99, which "
                "the statistics do not hold"
            ),
            1,
        )
        r["last"] = dict(r["last"], fp="other")
        self.assertIsNone(
            analyst.reading(st)
        )  # an answer for other data is never shown

    def test_a_reading_is_due_when_the_statistics_changed_and_every_h_passed(self):
        st = empty_state()
        self.assertFalse(analyst.due({}, st, STATS))  # off
        self.assertTrue(analyst.due(self.cfg, st, STATS))
        with mock.patch.object(roles, "start", return_value="ROLE_STARTED analyst"):
            self.assertEqual(
                analyst.start(self.cfg, st, self.store, STATS), "ROLE_STARTED analyst"
            )
        r = roles.state(st, "analyst")
        self.assertEqual((r["data_fp"], r["data"]["fp"]), ("abc", "abc"))
        self.assertFalse(analyst.due(self.cfg, st, STATS))  # the same statistics
        r["started"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=11)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        changed = dict(STATS, fp="def")
        self.assertFalse(analyst.due(self.cfg, st, changed))  # 11 h < every_h 12
        r["started"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=13)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(analyst.due(self.cfg, st, changed))
        r["job"] = {"id": "role-analyst-1"}
        self.assertFalse(analyst.due(self.cfg, st, changed))  # one at a time
        msgs = analyst.messages(STATS)
        self.assertIn(
            "Every number you write must appear in the data above, exactly",
            msgs[1]["content"],
        )
        self.assertIn('"linux-gpu-x86"', msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
