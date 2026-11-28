# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rules around the merge queue: hold labels, candidate vs ready, rocKE-tree freshness, candidate
analysis scheduling, auto-merge risks, approval candidates, post-merge breakage issues, and the HTML page built on them.
"""

import datetime as dt
import pathlib
import sys
import tempfile
import unittest

TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402
import home_repository  # noqa: E402

from rkmon_lib import breakage, guard, merge, post, queue, report, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

G, O, R = "\U0001f7e2", "\U0001f7e0", "\U0001f534"
# daily_dir: the daily-log file of a guarded write goes to a folder of this module's own, never over the real one in the
# outputs folder.
_DAILY = tempfile.TemporaryDirectory()


def tearDownModule():
    _DAILY.cleanup()


CFG = {
    "incident_issue": guard_labels.INCIDENT_ISSUE,
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "paths": {
        "working_dir": "/nonexistent",
        "repo": "/nonexistent",
        "daily_dir": _DAILY.name,
    },
    "features": {
        "merge_queue": {
            "mode": "prototype",
            "auto_merge": "enabled",
            "auto_hold": True,
            "develop_guard": True,
        },
        "alerts": {
            "enabled": True,
            "mode": "live",
            "develop_breaks": True,
            "needs_you": True,
        },
    },
    "max_posts_per_hour": 6,
    "review_first_line": "x",
    "codeowner_teams": {"platform": ["plat"], "library": ["lib"]},
}
# The merge queue live: everything acts, the auto-merge guard too (it turns someone else's auto-merge off), not only
# alerts as it does in prototype.
GUARD_ACT = dict(
    CFG,
    features=dict(
        CFG["features"], merge_queue=dict(CFG["features"]["merge_queue"], mode="live")
    ),
)


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def head(n: int) -> str:
    return (str(n) * 40)[:40]


def rec(n: int, **kw) -> dict:
    r = {
        "n": n,
        "head": head(n),
        "size": 100,
        "status": "open",
        "draft": False,
        "author": "a",
        "title": f"t{n}",
        "labels": [],
        "base": "develop",
        "head_ref": f"b{n}",
        "mergeable": "MERGEABLE",
        "ci_rollup": "SUCCESS",
        "review_decision": "REVIEW_REQUIRED",
        "reviews": [],
        "files_cache": {"head": head(n), "hot": [], "areas": ["platform"]},
        "ours": {
            "verdict": G,
            "head": head(n),
            "at": now(),
            "url": "https://github.com/ROCm/r/pull/1#c",
        },
        "threads": {"updated_at": None, "open": []},
    }
    r.update(kw)
    return r


class TestHoldLabels(unittest.TestCase):
    def test_patterns(self):
        for lab in (
            "DO NOT MERGE: documentation",
            "on hold",
            "ON_HOLD",
            "WIP",
            "Not ready to Review",
            "do-not-merge",
            "DNM",
        ):
            self.assertEqual(queue.hold_labels(CFG, {"labels": [lab]}), [lab], lab)
        for lab in (
            "ready for review",
            "rocke: platform",
            "red-develop",
            "merge conflict",
            "wipe-cache",
        ):
            self.assertEqual(queue.hold_labels(CFG, {"labels": [lab]}), [], lab)

    def test_label_fails_g8_until_removed(self):
        g = queue.gate_deps(CFG, empty_state(), rec(1, labels=["WIP"]), {})
        self.assertEqual(g[0], queue.FAIL)
        self.assertIn("'WIP'", g[1])
        self.assertEqual(queue.gate_deps(CFG, empty_state(), rec(1), {})[0], queue.PASS)


class TestGithubBlocks(unittest.TestCase):
    def test_blocked_with_ci_green_keeps_g2_pending(self):
        r = rec(1, review_decision="APPROVED", merge_state="BLOCKED")
        self.assertEqual(queue.gate_humans(CFG, r)[0], queue.PENDING)
        self.assertEqual(
            queue.gate_humans(CFG, dict(r, ci_rollup="PENDING"))[0], queue.PASS
        )  # CI explains BLOCKED
        self.assertEqual(
            queue.gate_humans(CFG, dict(r, merge_state="CLEAN"))[0], queue.PASS
        )
        with_thread = dict(
            r, threads={"updated_at": None, "open": [{"login": "yr", "path": "g.json"}]}
        )
        self.assertIn(
            "1 unresolved review thread (yr)", queue.gate_humans(CFG, with_thread)[1]
        )
        no_threads = dict(r, threads={"updated_at": None, "open": []})
        self.assertIn("ruleset rule", queue.gate_humans(CFG, no_threads)[1])
        self.assertNotIn(
            "fresh approval", queue.gate_humans(CFG, no_threads)[1]
        )  # develop: no last-push approval
        # A required check that never reported explains BLOCKED; G3 reports it, G2 passes.
        missing = dict(
            no_threads,
            required={"head": r["head"], "checks": {"Math CI Summary": "missing"}},
        )
        self.assertEqual(queue.gate_humans(CFG, missing), (queue.PASS, "approved"))
        self.assertIn("Math CI Summary (missing)", queue.gate_ci(missing)[1])

    def test_open_threads_block_while_github_recomputes(self):
        """When develop moves, GitHub reads UNKNOWN with CI re-running: a PR with unresolved threads does not head the
        queue. A known open thread keeps G2 pending unless GitHub reports the merge unblocked.
        """
        t = {"updated_at": None, "open": [{"login": "ar", "path": "v.py"}] * 5}
        r = rec(1, review_decision="APPROVED", threads=t, ci_rollup="PENDING")
        for state in ("UNKNOWN", "BEHIND", None):
            g = queue.gate_humans(CFG, dict(r, merge_state=state))
            self.assertEqual(g[0], queue.PENDING, state)
            self.assertIn("5 unresolved review threads (ar)", g[1])
        for state in queue.NOT_BLOCKED:
            self.assertEqual(
                queue.gate_humans(CFG, dict(r, merge_state=state))[0], queue.PASS, state
            )

    def test_approved_and_blocked_names_the_thread_not_the_user(self):
        """Approved (the approval still counts) and blocked by an unresolved thread: the line names the thread."""
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me"]}}
        st["prs"] = {
            "1": rec(
                1,
                review_decision=None,
                merge_state="BLOCKED",
                reviews=[{"login": "me", "state": "APPROVED", "commit": "0" * 40}],
                threads={
                    "updated_at": None,
                    "open": [{"login": "yr", "path": "g.json"}],
                },
            )
        }
        self.assertEqual(report.approval_candidates(CFG, st), [])
        left = report.merge_blockers(CFG, st, st["prs"]["1"])
        self.assertEqual([b["kind"] for b in left], ["threads"])
        self.assertEqual(left[0]["who"], "author and yr")


class TestFreshness(unittest.TestCase):
    def test_rocke_tree_keeps_gates_fresh(self):
        r = rec(
            1,
            gates={
                "head": head(1),
                "develop": "a" * 40,
                "develop_tree": "t" * 40,
                "results": {k: ["pass", ""] for k in ("G5", "G6", "G7")},
            },
        )
        self.assertEqual(
            queue.gates_dynamic(r, "b" * 40, "t" * 40)["G5"][0], "pass"
        )  # develop moved outside rocKE
        self.assertEqual(
            queue.gates_dynamic(r, "b" * 40, "u" * 40)["G5"][0], "pending"
        )  # rocKE moved
        self.assertEqual(queue.gates_dynamic(r, "b" * 40, None)["G5"][0], "pending")
        self.assertEqual(queue.gates_dynamic(r, "a" * 40)["G5"][0], "pass")


class TestStages(unittest.TestCase):
    def test_candidate_vs_ready(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            "1": rec(1, review_decision="APPROVED"),
            "2": rec(2, review_decision="APPROVED", ci_rollup="PENDING"),
            "3": rec(3),
            "4": rec(4, review_decision="APPROVED", labels=["WIP"]),
        }
        with mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue, "disruption", lambda cfg, st, cands: {r["n"]: [] for r in cands}
        ), mock.patch.object(
            queue, "abi_lanes", lambda *a: {}
        ), mock.patch.object(
            queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
        ), mock.patch.object(
            queue.batch, "build_batches", lambda *a: []
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        self.assertEqual(
            {r["n"]: r["stage"] for r in q},
            {1: "ready", 2: "candidate", 3: None, 4: None},
        )
        self.assertEqual([r["n"] for r in q][:2], [1, 2])
        self.assertEqual(report.stage_counts(q)["label"], 1)


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40
        self.st["prs"] = {"2": rec(2)}
        self.gate = mock.patch.object(
            merge,
            "start_gate",
            lambda cfg, st, store, n: [f"GATE_START #{n} against develop d"],
        )
        self.gate.start()

    def tearDown(self):
        self.gate.stop()
        self.d.cleanup()

    @staticmethod
    def q():
        return [
            {
                "n": 2,
                "stage": "candidate",
                "size": 100,
                "gates": dict(
                    {k: ["pending", ""] for k in ("G5", "G6", "G7")}, G4=["pass", ""]
                ),
            }
        ]

    def test_starts_when_the_slot_is_free(self):
        lines = merge.analyse_candidates(CFG, self.st, self.store, self.q())
        self.assertTrue(
            lines and lines[0].startswith("ANALYSIS_START #2 (candidate"), lines
        )

    def test_review_checks_go_first_until_a_candidate_starves(self):
        self.st["review_pipeline"] = {
            "pr": 9,
            "worker": {"started": ago(30)},
            "checks": None,
        }
        self.assertEqual(
            merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
        )
        self.st["analysis_wait"]["2"] = ago(200)
        self.assertTrue(merge.analyse_candidates(CFG, self.st, self.store, self.q()))

    def test_never_in_a_workers_first_five_minutes(self):
        self.st["review_pipeline"] = {
            "pr": 9,
            "worker": {"started": ago(1)},
            "checks": None,
        }
        self.st["analysis_wait"] = {"2": ago(500)}
        local = dict(CFG, worker={"url": "http://127.0.0.1:11434", "model": "m:1"})
        self.assertEqual(
            merge.analyse_candidates(local, self.st, self.store, self.q()), []
        )
        # A reviewer on another machine loads this one lightly: the analysis goes ahead.
        remote = dict(
            CFG,
            worker={
                "url": "https://models.example.com",
                "backend": "openai",
                "model": "m:1",
            },
        )
        self.assertTrue(merge.analyse_candidates(remote, self.st, self.store, self.q()))

    def test_waits_for_a_worker_on_a_huge_pr(self):
        """A worker on a huge PR is waited for while builds would load the machine."""
        self.st["prs"]["9"] = rec(9, size=17764)
        self.st["review_pipeline"] = {
            "pr": 9,
            "worker": {"id": "w9", "started": ago(20)},
            "checks": {"id": "c9"},
        }
        # jobs.status takes the job record, not its id
        with mock.patch.object(
            merge.jobs,
            "status",
            lambda store, job: "running" if job["id"] == "w9" else "done",
        ):
            self.assertEqual(
                merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
            )
            self.st["analysis_wait"] = {
                "2": ago(200)
            }  # waited past analysis_max_wait_min: it goes anyway
            self.assertTrue(
                merge.analyse_candidates(CFG, self.st, self.store, self.q())
            )
            self.st["prs"]["9"][
                "size"
            ] = 500  # a normal worker does not hold analysis back
            self.st["analysis_wait"] = {}
            self.assertTrue(
                merge.analyse_candidates(CFG, self.st, self.store, self.q())
            )

    def test_a_result_recorded_this_tick_is_not_re_run(self):
        """A result recorded this tick is not re-run, though `q` predates the GATE_DONE."""
        self.st["prs"]["2"]["gates"] = {
            "head": head(2),
            "develop": "d" * 40,
            "results": {k: ["pass", ""] for k in ("G5", "G6", "G7")},
        }
        self.assertEqual(
            merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
        )  # q still says pending

    def test_a_validated_batch_member_needs_no_analysis(self):
        """A validated batch member needs no gate job of its own to hold up a review's worker."""
        self.st["queue_state"] = {
            "validated": {
                "members": [1, 2],
                "heads": {"1": head(1), "2": head(2)},
                "develop": "d" * 40,
                "at": now(),
            }
        }
        self.assertEqual(
            merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
        )
        self.st["prs"]["2"]["head"] = (
            "f" * 40
        )  # pushed since the batch passed: the batch no longer covers it
        self.assertTrue(merge.analyse_candidates(CFG, self.st, self.store, self.q()))
        self.st["prs"]["2"]["head"] = head(2)
        self.st["develop"]["sha"] = "e" * 40
        with mock.patch.object(
            merge, "_carry", lambda cfg, st, val: True
        ):  # develop moved only outside rocKE
            self.assertEqual(
                merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
            )
        with mock.patch.object(
            merge, "_carry", lambda cfg, st, val: False
        ):  # other rocKE work landed since
            self.assertTrue(
                merge.analyse_candidates(CFG, self.st, self.store, self.q())
            )

    def test_fresh_or_held_candidates_need_no_analysis(self):
        self.st["prs"]["2"]["gates"] = {
            "head": head(2),
            "develop": "e" * 40,
            "develop_tree": "t" * 40,
            "results": {k: ["pass", ""] for k in ("G5", "G6", "G7")},
        }
        self.st["develop"]["rocke_tree"] = (
            "t" * 40
        )  # develop moved outside rocKE: the result is still fresh
        self.assertEqual(
            merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
        )
        self.st["prs"]["2"]["gates"] = None
        self.st["holds"] = {
            "2": {"head": head(2), "status": "triage", "reason": "x", "since": now()}
        }
        self.assertEqual(
            merge.analyse_candidates(CFG, self.st, self.store, self.q()), []
        )


class TestAutomergeRisks(unittest.TestCase):
    @staticmethod
    def gh_run(result, bodies: list) -> mock.Mock:
        """subprocess.run for gh: `result` for each call, and the text of each body file gh is given appended to
        `bodies` while gh runs (the file goes once gh has read it)."""

        def run(args, **kw):
            if "--body-file" in args:
                path = pathlib.Path(args[args.index("--body-file") + 1])
                bodies.append(path.read_text(encoding="utf-8"))
            return result

        return mock.Mock(side_effect=run)

    def test_the_users_auto_merge_on_a_wave_1_pr_over_our_orange_is_kept(self):
        """The operator's own auto-merge on a wave-1 PR is kept over our 🟠 (its follow-up tickets come anyway). Only
        the operator's own arming, only the plan's wave 1 (its end after a rebase included), only our 🟠.
        """
        st = empty_state()
        mine = {"by": "me", "at": now(), "method": "SQUASH"}
        st["merge_plan"] = {"waves": [[1, 2, 3], [5]], "first_rebase": [4]}
        st["prs"] = {
            "1": rec(1, auto_merge=mine, ours={"verdict": O, "head": head(1)}),
            "2": rec(2, auto_merge=mine, ours={"verdict": R, "head": head(2)}),
            "3": rec(
                3,
                auto_merge={"by": "bob", "at": now()},
                ours={"verdict": O, "head": head(3)},
            ),
            "4": rec(
                4,
                auto_merge=mine,
                ours={"verdict": O, "head": head(4)},
                labels=["DO NOT MERGE: documentation"],
            ),
            "5": rec(5, auto_merge=mine, ours={"verdict": O, "head": head(5)}),
        }
        lines = merge.automerge_risks(CFG, st)
        self.assertEqual(sorted(st["automerge_kept"]), ["1"])
        self.assertEqual(
            sorted(st["automerge_risks"]), ["2", "3", "4", "5"]
        )  # 🔴, someone else, a label, wave 2
        self.assertFalse(any("#1 " in line for line in lines), lines)

    def test_the_users_own_arming_outside_the_first_group_is_an_alert_not_a_disarm(
        self,
    ):
        """The operator's own arming outside the first group, with a 🟠 from our later follow-up: the guard alerts
        them and never turns their auto-merge off."""
        st = empty_state()
        st["merge_plan"] = {"waves": [], "first_rebase": []}
        st["prs"] = {
            "1": rec(
                1,
                auto_merge={"by": "me", "at": now(), "method": "SQUASH"},
                ours={"verdict": O, "head": head(1)},
            ),
            "2": rec(
                2,
                auto_merge={"by": "bob", "at": now(), "method": "SQUASH"},
                ours={"verdict": O, "head": head(2)},
            ),
        }
        cfg = dict(GUARD_ACT, automerge_guard_own=True)
        lines = merge.automerge_risks(cfg, st)
        self.assertEqual(sorted(st["automerge_risks"]), ["1", "2"])
        self.assertTrue(
            [
                x
                for x in lines
                if x.startswith("AUTOMERGE_RISK #1 ") and x.endswith("alert the user")
            ],
            lines,
        )
        self.assertTrue(
            [
                x
                for x in lines
                if x.startswith("AUTOMERGE_RISK #2 ") and "rk automerge-off 2" in x
            ],
            lines,
        )

    def test_a_re_enable_at_the_head_we_turned_off_is_an_alert_not_a_second_turn_off(
        self,
    ):
        """Auto-merge turned back on at the head we turned it off at is an alert, not a second turn-off: the tick line
        and the board agree ("I don't turn it off twice: your call")."""
        st = empty_state()
        st["merge_plan"] = {"waves": [], "first_rebase": []}
        st["prs"] = {
            "2": rec(
                2,
                auto_merge={"by": "bob", "at": now(), "method": "SQUASH"},
                ours={"verdict": O, "head": head(2)},
            )
        }
        cfg = GUARD_ACT
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            store.ledger_add(
                "automerge-off", 2, "x" * 40, "an older head"
            )  # another head: still ours to act on
            lines = merge.automerge_risks(cfg, st, store)
            self.assertTrue(
                [
                    x
                    for x in lines
                    if x.startswith("AUTOMERGE_RISK #2 ") and "rk automerge-off 2" in x
                ],
                lines,
            )
            store.ledger_add("automerge-off", 2, head(2), "our review found problems")
            lines = merge.automerge_risks(cfg, dict(st, automerge_risks={}), store)
            line = next(x for x in lines if x.startswith("AUTOMERGE_RISK #2 "))
            self.assertNotIn("rk automerge-off", line)
            self.assertTrue(
                line.endswith(
                    "alert the user (the auto-merge guard turns auto-merge off once per head)"
                ),
                line,
            )

    def test_failing_gates_with_auto_merge_on(self):
        st = empty_state()
        am = {"by": "bob", "at": now(), "method": "SQUASH"}
        st["prs"] = {
            "1": rec(1, auto_merge=am, ours={"verdict": R, "head": head(1)}),
            "2": rec(2, auto_merge=am),
            "3": rec(3, auto_merge=am),
            "4": rec(4, auto_merge=am),
            "5": rec(5, auto_merge=am, ours={"verdict": R, "head": "f" * 40}),
            "6": rec(6, auto_merge=am, labels=["DO NOT MERGE: documentation"]),
        }
        st["holds"] = {
            "3": {
                "head": head(3),
                "status": "held",
                "reason": "G5 dg: 13 unblessed; G6 build failed",
                "since": now(),
            },
            "4": {
                "head": head(4),
                "status": "triage",
                "reason": "G5 dg fail",
                "since": now(),
            },
        }
        lines = merge.automerge_risks(CFG, st)
        self.assertEqual(sorted(st["automerge_risks"]), ["1", "3", "6"])
        self.assertTrue(
            lines and all("alert the user" in line for line in lines), lines
        )
        three = st["automerge_risks"]["3"]
        self.assertIn("golden files", three["reasons"][0])
        self.assertIn("the build or the tests fail", three["reasons"][1])
        self.assertIn("fail (build failed)", three["reasons"][1])
        st["holds"]["3"][
            "reason"
        ] = "G6: its new test t.py::test_x fails on the merge without a device; our N1"
        st["holds"]["3"]["reason"] = st["holds"]["3"]["reason"].replace(
            "device", "device " + "x" * 200
        )
        merge.automerge_risks(CFG, st)
        cut = st["automerge_risks"]["3"]["reasons"][0]
        self.assertIn(
            "fail (its new test t.py::test_x", cut
        )  # no ": " left over, and the cut is marked
        self.assertTrue(cut.endswith("…)"), cut)
        st["holds"]["3"]["reason"] = "G5 dg: 13 unblessed; G6 build failed"
        st["automerge_risks"] = {}
        merge.automerge_risks(CFG, st)
        self.assertIn("re-bless", three["fixes"][0])
        self.assertIn("should not merge yet", st["automerge_risks"]["6"]["reasons"][0])
        self.assertIn("blocking problems", st["automerge_risks"]["1"]["reasons"][0])
        self.assertEqual(merge.automerge_risks(CFG, st), [])  # unchanged: reported once
        st["queue_state"] = {"armed": {"pr": 1}}
        merge.automerge_risks(CFG, st)
        self.assertEqual(sorted(st["automerge_risks"]), ["3", "6"])
        lines = merge.automerge_risks(GUARD_ACT, dict(st, automerge_risks={}))
        self.assertIn("rk automerge-off 3", lines[0])
        text = post.automerge_off_text(CFG, st["prs"]["3"], st["automerge_risks"]["3"])
        line = post.notice_line(CFG, "queue")
        self.assertTrue(text.startswith(line + "\n\n"))
        self.assertIn("Hi @a, **I turned auto-merge off on this PR for now.**", text)
        self.assertEqual(post.check(dict(CFG, review_first_line=line), text), [])
        text = post.automerge_off_text(
            CFG,
            st["prs"]["3"],
            st["automerge_risks"]["3"],
            ["N1 (x.py:3): golden stale → re-bless with run_diff.py"],
        )
        self.assertIn("The open items from our review, each with its fix:", text)
        self.assertIn("- N1 (x.py:3): golden stale → re-bless with run_diff.py", text)

    def test_automerge_off_guards(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {"1": rec(1, auto_merge={"by": "me"}, author="me")}
            st["automerge_risks"] = {
                "1": {"head": head(1), "by": "me", "reasons": ["r"], "fixes": ["f"]}
            }
            with self.assertRaisesRegex(
                post.PostRefused, "the auto-merge guard only alerts while"
            ):
                post.automerge_off(CFG, st, store, 1)
            with self.assertRaisesRegex(post.PostRefused, "own PR"):
                post.automerge_off(GUARD_ACT, st, store, 1)
            st["prs"]["1"]["head"] = "e" * 40
            with self.assertRaisesRegex(
                post.PostRefused, "no recorded auto-merge risk"
            ):
                post.automerge_off(GUARD_ACT, st, store, 1)

    def test_automerge_off_reads_who_enabled_it_live(self):
        """The tick saw someone else's auto-merge; the user turned it on again since: theirs is left on, read live."""
        from rkmon_lib import gh

        cfg = dict(GUARD_ACT, operator={"github_login": "me"})
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {"1": rec(1, auto_merge={"by": "b"}, author="b")}
            st["automerge_risks"] = {
                "1": {"head": head(1), "by": "b", "reasons": ["r"], "fixes": ["f"]}
            }
            run = mock.Mock()
            with mock.patch.object(
                gh,
                "rest",
                return_value={
                    "state": "open",
                    "head": head(1),
                    "auto": True,
                    "by": "me",
                },
            ), mock.patch.object(post.subprocess, "run", run):
                with self.assertRaisesRegex(
                    post.PostRefused, "the user's own: it is left on"
                ):
                    post.automerge_off(cfg, st, store, 1)
            run.assert_not_called()
            self.assertFalse(store.ledger_has("automerge-off", 1, head(1)))

    def test_the_explanation_is_never_rate_limited_and_a_rerun_finishes_it(self):
        """The guard's explanation after it turns auto-merge off is never rate-limited, and a rerun finishes it."""
        from rkmon_lib import gh

        cfg = GUARD_ACT
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            for i in range(12):  # a busy hour: well past the rate limit
                store.ledger_add("post", 100 + i, "h", f"u{i}")
            st = empty_state()
            st["prs"] = {"1": rec(1, auto_merge={"by": "b"}, author="b")}
            st["automerge_risks"] = {
                "1": {"head": head(1), "by": "b", "reasons": ["r"], "fixes": ["f"]}
            }
            done = mock.Mock(
                returncode=0,
                stdout="https://github.com/ROCm/r/pull/1#issuecomment-7\n",
                stderr="",
            )
            with mock.patch.object(
                gh,
                "rest",
                return_value={"state": "open", "head": head(1), "auto": True},
            ), mock.patch.object(post.subprocess, "run", return_value=done):
                out = post.automerge_off(cfg, st, store, 1)
            self.assertIn(
                "comment https://github.com/ROCm/r/pull/1#issuecomment-7", out
            )
            self.assertTrue(
                store.ledger_has(
                    "notice-queue-sticky", 1, f"automerge-off-{head(1)[:10]}"
                )
            )
        with tempfile.TemporaryDirectory() as d:  # turned off, explanation missing, risk already dropped: re-run
            store = Store(pathlib.Path(d))
            store.ledger_add(
                "automerge-off",
                1,
                head(1),
                "our latest review of this exact version found problems",
            )
            st = empty_state()
            st["prs"] = {"1": rec(1, author="b")}
            bodies = []
            with mock.patch.object(
                post.subprocess, "run", self.gh_run(done, bodies)
            ), mock.patch.object(
                post, "open_findings", return_value=["N1 (x.py:3): y → z"]
            ):
                out = post.automerge_off(cfg, st, store, 1)
            self.assertTrue(out.startswith("explanation posted:"), out)
            self.assertIn("N1 (x.py:3): y → z", bodies[-1])
            with self.assertRaisesRegex(
                post.PostRefused, "no recorded auto-merge risk"
            ):
                post.automerge_off(
                    cfg, st, store, 1
                )  # explained now: nothing left to do

    def test_automerge_off_on_the_users_instruction(self):
        """The operator turns off an author's auto-merge to land other PRs first, with their own reason (the merge
        queue live: in prototype a write typed is refused before its guards)."""
        from rkmon_lib import gh

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "1": rec(1, auto_merge={"by": "b"}, author="b")
            }  # no recorded risk: the guard would refuse
            msg = pathlib.Path(d) / "why.md"
            msg.write_text(
                "Hi @b, auto-merge is off for now: other merges go first.\n",
                encoding="utf-8",
            )
            done = mock.Mock(
                returncode=0,
                stdout="https://github.com/ROCm/r/pull/1#issuecomment-9\n",
                stderr="",
            )
            bad = pathlib.Path(d) / "bad.md"
            bad.write_text(
                "See https://tracker.internal.example/browse/" + TK + "-1 first.\n",
                encoding="utf-8",
            )
            bodies = []
            with mock.patch.object(
                gh,
                "rest",
                return_value={"state": "open", "head": head(1), "auto": True},
            ), mock.patch.object(
                post.subprocess, "run", self.gh_run(done, bodies)
            ) as run:
                with self.assertRaises(
                    post.PostRefused
                ):  # a message the public repo may not get changes nothing
                    post.automerge_off_user(GUARD_ACT, st, store, 1, str(bad))
                self.assertFalse(run.called)
                out = post.automerge_off_user(GUARD_ACT, st, store, 1, str(msg))
                self.assertIn("auto-merge disabled on the user's instruction", out)
                self.assertEqual(
                    run.call_args_list[0][0][0][:4], ["gh", "pr", "merge", "1"]
                )
                (posted,) = bodies
                self.assertTrue(
                    posted.startswith(post.notice_line(GUARD_ACT, "queue") + "\n\n"),
                    posted,
                )
                self.assertIn("other merges go first", posted)
                self.assertTrue(store.ledger_has("automerge-off-user", 1, head(1)))
                self.assertFalse(
                    store.ledger_has("automerge-off", 1, head(1))
                )  # never read as the guard's own
                with self.assertRaisesRegex(
                    post.PostRefused, "already turned off at this head"
                ):
                    post.automerge_off_user(GUARD_ACT, st, store, 1, str(msg))
            with mock.patch.object(
                gh,
                "rest",
                return_value={"state": "open", "head": head(1), "auto": False},
            ):
                st["prs"]["2"] = rec(2, author="b")
                with self.assertRaisesRegex(post.PostRefused, "changed on GitHub"):
                    post.automerge_off_user(GUARD_ACT, st, store, 2, str(msg))


class TestApprovals(unittest.TestCase):
    def test_who_needs_the_user_and_in_what_order(self):
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me", "x"], "lib": ["y"]}}
        lib = {"hot": [], "areas": ["library"]}
        st["prs"] = {
            "1": rec(1),  # needs a platform code-owner approval: the user's
            "2": rec(
                2,
                reviews=[
                    {"login": "me", "state": "CHANGES_REQUESTED", "commit": "0" * 40}
                ],
            ),
            "3": rec(
                3, reviews=[{"login": "me", "state": "APPROVED", "commit": head(3)}]
            ),  # approved at head
            "4": rec(
                4, files_cache=dict(lib, head=head(4))
            ),  # library only: not the user's area
            "5": rec(5, author="me"),  # the user's own PR
            "6": rec(6, labels=["DO NOT MERGE: documentation"]),
            "7": rec(7, ours={"verdict": O, "head": head(7), "at": now()}),
            "8": rec(8, review_decision="APPROVED"),
            "9": rec(9, files_cache=dict(lib, head=head(9)), review_requests=["me"]),
            # The user approved an older head: that approval still counts (stale reviews are not dismissed).
            "10": rec(
                10,
                review_decision=None,
                reviews=[{"login": "me", "state": "APPROVED", "commit": "0" * 40}],
            ),
            "11": rec(
                11, ours={"verdict": G, "head": "f" * 40, "at": now()}
            ),  # pushed since our review: final review
            "12": rec(
                12,
                threads={"updated_at": None, "open": [{"login": "q", "path": "x.py"}]},
            ),  # a comment is open
            "13": {
                k: v for k, v in rec(13).items() if k != "threads"
            },  # threads not read yet: not proven addressed
            "14": rec(
                14,
                reviews=[
                    {"login": "z", "state": "CHANGES_REQUESTED", "commit": head(14)}
                ],
            ),
        }
        a = report.approval_candidates(CFG, st)
        self.assertEqual(
            a[0]["n"], 2
        )  # only the user can clear their own CHANGES_REQUESTED
        self.assertEqual(sorted(x["n"] for x in a), [1, 2, 9])
        one = next(x for x in a if x["n"] == 1)
        self.assertIn("platform code-owner approval: yours would do", one["why"])
        self.assertIn("review is requested", next(x for x in a if x["n"] == 9)["why"])
        self.assertIn("every thread resolved", report.approval_line(one))
        nine = next(x for x in a if x["n"] == 9)
        self.assertEqual(nine["also"], ["a library code-owner approval (lib)"])
        kinds = {
            n: [b["kind"] for b in report.merge_blockers(CFG, st, st["prs"][n])]
            for n in ("10", "11", "12", "14")
        }
        self.assertEqual(
            kinds,
            {
                "10": ["analysis"],
                "11": ["review", "approval"],
                "12": ["threads", "approval"],
                "14": ["changes", "approval"],
            },
        )
        close = report.close_to_merge(CFG, st)
        self.assertEqual(
            [len(r["blockers"]) for r in close],
            sorted(len(r["blockers"]) for r in close),
        )
        self.assertEqual(
            [b["kind"] for b in next(r for r in close if r["n"] == 8)["blockers"]],
            ["analysis"],
        )

    def test_one_box_per_code_owner_team(self):
        """One box per code-owner team, "Needs platform approval" (anyone from the platform team can approve), and
        "Needs rocke-lib approval" underneath, with the same rules; then "Needs any approval": a PR whose only missing
        approval is a general one is there and in neither team box.
        """
        cfg = dict(
            CFG,
            codeowner_teams={
                "platform": ["plat"],
                "library": ["lib"],
                "root": ["plat", "lib"],
            },
        )
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me", "x"], "lib": ["y"]}}
        both, lib = {"hot": [], "areas": ["library", "platform"]}, {
            "hot": [],
            "areas": ["library"],
        }
        mine_cr = [{"login": "me", "state": "CHANGES_REQUESTED", "commit": head(5)}]
        st["prs"] = {
            "1": rec(1, files_cache=dict(both, head=head(1))),  # needs both teams
            "2": rec(2, files_cache=dict(lib, head=head(2))),  # rocke-lib only
            "3": rec(
                3, author="me"
            ),  # the user's own PR: another platform member approves
            "4": rec(
                4, files_cache={"head": head(4), "hot": [], "areas": ["root"]}
            ),  # either team's approval does
            "5": rec(
                5, files_cache=dict(lib, head=head(5)), reviews=mine_cr
            ),  # only the user can clear theirs
            "6": rec(
                6, files_cache=dict(lib, head=head(6)), review_requests=["me"]
            ),  # a request is no requirement
            "7": rec(
                7,
                files_cache=dict(both, head=head(7)),
                threads={"updated_at": None, "open": [{"login": "q"}]},
            ),
            "8": rec(
                8,
                reviews=[{"login": "x", "state": "APPROVED", "commit": head(8)}],
                review_requests=["lib"],
            ),
            # the develop ruleset's extra approval: approved by the code owners, CI green, GitHub still blocks
            "9": rec(
                9,
                review_decision="APPROVED",
                merge_state="BLOCKED",
                required={"head": head(9), "checks": {}},
            ),
            # a required review no code-owner team gives (no rocKE area, nobody requested)
            "10": rec(10, files_cache={"head": head(10), "hot": [], "areas": []}),
        }
        teams = report.team_approvals(cfg, st)
        self.assertEqual(list(teams), ["plat", "lib"])  # config order
        rows = {t: {a["n"]: a for a in v} for t, v in teams.items()}
        self.assertEqual(
            (sorted(rows["plat"]), sorted(rows["lib"])),
            ([1, 3, 4, 5], [1, 2, 4, 5, 6, 8]),
        )
        self.assertEqual(
            (rows["plat"][1]["why"], rows["plat"][1]["also"]),
            (
                "needs a platform code-owner approval (plat)",
                ["a library code-owner approval (lib)"],
            ),
        )
        self.assertEqual(
            rows["lib"][1]["also"], ["a platform code-owner approval (plat)"]
        )
        self.assertEqual((rows["plat"][4]["also"], rows["lib"][4]["also"]), ([], []))
        stands = "me's change request stands (the only review blocking it)"
        self.assertEqual(
            (rows["plat"][5]["why"], rows["plat"][5]["also"]),
            (stands, ["a library code-owner approval " "(lib)"]),
        )
        self.assertEqual(rows["lib"][5]["also"], [stands])
        self.assertEqual(
            rows["lib"][6]["why"],
            "needs a library code-owner approval (lib); me's review is requested",
        )
        self.assertEqual(
            rows["lib"][8]["why"],
            "needs another required approval (GitHub: review required; requested: "
            "lib)",
        )
        self.assertEqual(
            [a["n"] for a in report.approval_candidates(cfg, st)], [5, 1, 4, 6]
        )  # the user's own list
        anyone = {a["n"]: a for a in report.any_approvals(cfg, st, teams)}
        self.assertEqual(sorted(anyone), [9, 10])  # in neither team box
        self.assertTrue(
            anyone[9]["why"].startswith(
                "needs GitHub still blocks the merge with CI green and the approvals in"
            ),
            anyone[9]["why"],
        )
        self.assertEqual(
            anyone[10]["why"],
            "needs another required approval (GitHub: review required): any approver will do",
        )
        with tempfile.TemporaryDirectory() as d:
            board = report.board(
                dict(
                    cfg,
                    paths={"working_dir": d, "repo": "/nonexistent", "daily_dir": d},
                ),
                st,
                Store(pathlib.Path(d)),
                [],
                [],
            )
        md = board["md"]
        self.assertIn(
            "4 PRs are ready for your approval: no more reviews needed, every review thread resolved (#5, #1, "
            "#4, #6; in the Needs approval lists below).",
            md,
        )
        heads = [
            "**Needs plat approval** (anyone in plat can approve):",
            "**Needs lib approval** (anyone in lib can approve):",
            "**Needs any approval** (any approver will do):",
        ]
        # platform, rocke-lib, any
        self.assertEqual(
            [md.index(h) for h in heads], sorted(md.index(h) for h in heads)
        )
        any_md = md[md.index(heads[2]) :]
        self.assertIn("#9 ", any_md)
        self.assertNotIn("#9 ", md[: md.index(heads[2])])
        for gone in ("Ready for your approval", "Ready for plat", "Ready for lib"):
            self.assertNotIn(gone, md)
        self.assertEqual(board["teams"], teams)
        self.assertEqual(
            [a["n"] for a in board["any"]],
            [a["n"] for a in report.any_approvals(cfg, st, teams)],
        )

    def test_a_green_rollup_with_required_checks_missing_is_not_called_green(self):
        # A fork whose workflows await a maintainer's approval is not "CI green" on its bot checks alone.
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me", "x"], "lib": ["y"]}}
        st["prs"] = {
            "1": rec(
                1,
                required={
                    "head": head(1),
                    "checks": {
                        "Multi-Arch CI Summary": "missing",
                        "Math CI Summary": "missing",
                    },
                },
            ),
            "2": rec(2),
        }
        a = {x["n"]: x for x in report.approval_candidates(CFG, st)}
        self.assertEqual(a[2]["ci"], "green")
        self.assertEqual(
            a[1]["ci"],
            "incomplete: Multi-Arch CI Summary (missing), Math CI Summary (missing)",
        )
        self.assertIn(
            "· CI incomplete: Multi-Arch CI Summary (missing)",
            report.approval_line(a[1]),
        )
        self.assertEqual(
            [x["n"] for x in report.approval_candidates(CFG, st)], [2, 1]
        )  # green first

    def test_a_required_review_names_the_requested_owners(self):
        # A PR whose rocKE areas are not computed yet (our 🟠 at an older head) names the requested owners, never
        # code owners outside the rocKE tree
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me", "x"], "lib": ["y"]}}
        approved = [{"login": "x", "state": "APPROVED", "commit": head(1)}]
        st["prs"] = {
            "1": rec(1, reviews=approved, files_cache=None, review_requests=["lib"]),
            "2": rec(2, reviews=approved, files_cache=None, review_requests=["plat"]),
            "3": rec(3, reviews=approved, files_cache=None, review_requests=[]),
        }
        who = {
            n: [
                b["who"]
                for b in report.merge_blockers(CFG, st, st["prs"][n])
                if b["kind"] == "approval"
            ]
            for n in ("1", "2", "3")
        }
        self.assertEqual(
            who,
            {
                "1": ["lib"],
                "2": ["you (or another plat member)"],
                "3": ["code owners outside the rocKE tree"],
            },
        )
        what = next(
            b["what"]
            for b in report.merge_blockers(CFG, st, st["prs"]["1"])
            if b["kind"] == "approval"
        )
        self.assertEqual(
            what, "another required approval (GitHub: review required; requested: lib)"
        )

    def test_the_users_own_pr_never_waits_on_the_users_approval(self):
        # The operator's own PR never waits on "you (or another rocke-platform member)": GitHub never counts an
        # author's approval of their own PR.
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me", "x"], "lib": ["y"]}}
        st["prs"] = {
            "1": rec(1, author="me"),
            "2": rec(2, author="me", files_cache=None, review_requests=["plat"]),
            "3": rec(3),
        }
        who = {
            n: [
                b["who"]
                for b in report.merge_blockers(CFG, st, st["prs"][n])
                if b["kind"] == "approval"
            ]
            for n in ("1", "2", "3")
        }
        self.assertEqual(
            who,
            {
                "1": ["another plat member"],
                "2": ["another plat member"],
                "3": ["you (or another plat member)"],
            },
        )

    def test_a_conflict_github_is_recomputing_still_blocks(self):
        # A PR that conflicts, read UNKNOWN by GitHub right after develop moves, is not ready for approval.
        st = empty_state()
        st["codeowners"] = {"teams": {"plat": ["me"], "lib": ["y"]}}
        st["prs"] = {
            "1": rec(1, mergeable="UNKNOWN", mergeable_known="CONFLICTING"),
            "2": rec(2, mergeable="UNKNOWN", mergeable_known="MERGEABLE"),
        }
        kinds = {
            n: [b["kind"] for b in report.merge_blockers(CFG, st, st["prs"][n])]
            for n in ("1", "2")
        }
        self.assertIn("conflict", kinds["1"])
        self.assertNotIn("conflict", kinds["2"])
        self.assertEqual(
            queue.gate_fresh_merge(CFG, st["prs"]["1"], local=False),
            (queue.FAIL, "conflicts with develop (GitHub)"),
        )
        self.assertEqual(
            queue.gate_fresh_merge(CFG, st["prs"]["2"], local=False)[0], queue.PENDING
        )

    def test_final_review_goes_first(self):
        from rkmon_lib import plan

        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["codeowners"] = {"teams": {"plat": ["me"]}}
        moved = {"verdict": G, "head": "f" * 40, "at": now()}
        st["prs"] = {
            "1": rec(1, ours=moved, review_decision="APPROVED", head_seen_at=ago(120)),
            "2": rec(
                2, ours=moved, head_seen_at=ago(120)
            ),  # needs the user's platform approval
            "3": rec(
                3,
                ours=moved,
                head_seen_at=ago(120),
                reviews=[
                    {"login": "z", "state": "CHANGES_REQUESTED", "commit": "f" * 40}
                ],
            ),
        }
        items = {
            i["pr"]: i
            for i in plan.make_plan(dict(CFG, settle_min=30), st, [], [])
            if i["kind"] == "FOLLOW_UP"
        }
        self.assertEqual(
            (items[1]["prio"], items[2]["prio"], items[3]["prio"]), (1, 1, 4)
        )
        self.assertTrue(items[1]["text"].startswith("final review (close to merge)"))


class TestBreakage(unittest.TestCase):
    def test_parse_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "i.md"
            p.write_text(
                "---\ntitle: [develop] test_attention_ir_cpp_parity fails\nlabels: red-develop, project: "
                "hip-kernel-provider, rocke: platform\nincident: i1\n---\n"
                + post.notice_line(CFG, "report")
                + "\n\nbody #3333\n",
                encoding="utf-8",
            )
            f = breakage.parse_file(str(p), CFG)
        self.assertEqual(
            f["labels"],
            ["red-develop", "project: hip-kernel-provider", "rocke: platform"],
        )
        self.assertEqual((f["kind"], f["assign"]), (breakage.RED, []))
        self.assertEqual(
            (f["title"], f["incident"]),
            ("[develop] test_attention_ir_cpp_parity fails", "i1"),
        )
        self.assertTrue(f["body"].startswith(post.notice_line(CFG, "report")))
        # the test names are the home repository's, which its repository defaults name
        home = home_repository.config(CFG)
        self.assertEqual(
            breakage.overlaps(
                home,
                "golden stale after #3333; rocke_differential_golden_ll red",
                "culprit #3333: rocke_differential_golden_ll",
            ),
            ["#3333", "rocke_differential_golden_ll"],
        )
        self.assertEqual(breakage.overlaps(home, "see #14", "nothing shared"), [])
        self.assertEqual(
            breakage.overlaps(
                home,
                "rocke_pytest: test_a_b fails",
                "rocke_pytest: test_c_d times out",
            ),
            [],
        )

    def test_poll_reports_rocke_issues_that_open_and_close(self):
        st = empty_state()
        cfg = dict(
            CFG,
            scope={
                "labels": ["rocke: platform"],
                "path_labels": ["project: hip-kernel-provider"],
            },
        )
        base = {
            "author": "s",
            "assignees": [],
            "created": now(),
            "url": "u",
            "body": "",
        }
        first = [
            dict(
                base,
                n=1,
                title="[develop] rocFFT smoketest times out",
                labels=["project: rocfft"],
            )
        ]
        second = first + [
            dict(
                base,
                n=2,
                title="[develop] golden stale",
                labels=["project: hip-kernel-provider"],
            )
        ]
        with mock.patch.object(
            breakage, "fetch_open", lambda cfg: [dict(i) for i in first]
        ):
            self.assertEqual(
                breakage.poll(cfg, st), []
            )  # the first read reports nothing
        with mock.patch.object(
            breakage, "fetch_open", lambda cfg: [dict(i) for i in second]
        ):
            lines = breakage.poll(cfg, st)
        self.assertTrue(lines and lines[0].startswith("BREAKAGE_OPEN #2"), lines)
        self.assertNotIn("body", st["post_merge"]["open"][0])
        with mock.patch.object(
            breakage, "fetch_open", lambda cfg: [dict(i) for i in first]
        ):
            self.assertTrue(breakage.poll(cfg, st)[0].startswith("BREAKAGE_CLOSED #2"))

    def test_issue_file_guards_and_links_the_incident(self):
        from rkmon_lib import gh

        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            gh, "rest", return_value="red-develop"
        ):
            store = Store(pathlib.Path(d))
            guard.save_incidents(
                store, [{"id": "i1", "status": "open", "start": now()}]
            )
            p = pathlib.Path(d) / "i.md"
            p.write_text(
                "---\ntitle: [develop] x fails\nlabels: red-develop, rocke: platform\nincident: i1\n---\n"
                "wrong line 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(post.PostRefused, "line 1"):
                post.issue_file(GUARD_ACT, empty_state(), store, str(p))
            p.write_text(
                "---\ntitle: [develop] x fails\nlabels: red-develop, rocke: platform\nincident: i1\n---\n"
                + post.notice_line(GUARD_ACT, "report")
                + "\n\nculprit #3333\n",
                encoding="utf-8",
            )
            dup = [(7777, "golden stale", ["#3333"], "red-develop")]
            with mock.patch.object(breakage, "duplicates", lambda cfg, t, b: dup):
                with self.assertRaisesRegex(
                    post.PostRefused, "possible duplicate of open red-develop #7777"
                ):
                    post.issue_file(GUARD_ACT, empty_state(), store, str(p))
            done = mock.Mock(
                returncode=0, stdout="https://github.com/ROCm/r/issues/99\n", stderr=""
            )
            with mock.patch.object(post.subprocess, "run", return_value=done) as run:
                url = post.issue_file(GUARD_ACT, empty_state(), store, str(p), new=True)
            self.assertEqual(url, "https://github.com/ROCm/r/issues/99")
            self.assertIn("red-develop", run.call_args[0][0])
            self.assertEqual(guard.incidents(store)[0]["issue"], 99)
            with self.assertRaisesRegex(post.PostRefused, "already has issue #99"):
                post.issue_file(GUARD_ACT, empty_state(), store, str(p), new=True)

    def test_label_create_is_guarded_and_ledgered(self):
        """Creating a label the repo does not have (silent-break) is guarded and ledgered (the merge queue live: in
        prototype a write typed is refused before its guards)."""
        from rkmon_lib import gh

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            with mock.patch.object(gh, "rest", return_value="silent-break"):
                with self.assertRaisesRegex(post.PostRefused, "already exists"):
                    post.label_create(GUARD_ACT, store, "silent-break", "d", "FBCA04")
            missing = mock.Mock(side_effect=gh.GhError("gh api: Not Found (HTTP 404)"))
            done = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(gh, "rest", missing), mock.patch.object(
                post.subprocess, "run", return_value=done
            ) as run:
                with self.assertRaisesRegex(post.PostRefused, "six hex digits"):
                    post.label_create(GUARD_ACT, store, "silent-break", "d", "#FBCA04")
                out = post.label_create(
                    GUARD_ACT,
                    store,
                    "silent-break",
                    "a develop break CI does not catch",
                    "FBCA04",
                )
            self.assertIn("created", out)
            self.assertEqual(
                run.call_args[0][0][:4], ["gh", "label", "create", "silent-break"]
            )
            self.assertTrue(store.ledger_has("label", 0, "silent-break"))


class TestPage(unittest.TestCase):
    def test_sections_flow_and_escaping(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                CFG,
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["enabled"] = True
            st["develop"]["sha"] = "d" * 40
            st["codeowners"] = {"teams": {"plat": ["me"]}}
            st["prs"] = {
                "1": rec(1, title="<img src=x onerror=alert(1)>"),
                "2": rec(2, review_decision="APPROVED", ci_rollup="PENDING"),
            }
            st["post_merge"] = {
                "at": now(),
                "open": [
                    {
                        "n": 7777,
                        "title": "[rocke] golden <b>stale</b>",
                        "rocke": True,
                        "author": "b",
                        "assignees": ["c"],
                        "created": now(),
                        "labels": [],
                    }
                ],
            }
            q = [
                {
                    "n": 2,
                    "stage": "candidate",
                    "blockers": ["G3: CI running"],
                    "gs": {"G1": "pass", "G3": "pending"},
                }
            ]
            board = report.board(cfg, st, store, [], q)
            page = pathlib.Path(webdash.write(cfg, st, store, [], q, board)).read_text(
                encoding="utf-8"
            )
        # One approval box per code-owner team (plat, then lib); no "Close to merge"
        for sid in (
            "needs",
            "flow",
            "approvals",
            "approvals-lib",
            "plan",
            "merge",
            "reviews",
            "develop",
            "log",
        ):
            self.assertIn(f"id='{sid}'", page)
        self.assertNotIn("id='close'", page)
        self.assertNotIn("<img src=x", page)
        self.assertNotIn("<b>stale</b>", page)
        self.assertIn("issues/7777", page)
        self.assertIn("pull/1", page)  # #1 is an approval candidate
        self.assertIn("1 PR is ready for your approval", board["md"])
        self.assertIn("st-candidate", page)

    def test_approval_boxes_then_mergeable_candidates_then_the_plan(self):
        """ "Needs platform approval", "Needs rocke-lib approval" underneath, then "Needs any approval", no "Close to
        merge", then the Mergeable Candidates (the PRs that passed CI and can merge now, and their gates), then the merge
        plan.
        """
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                CFG,
                codeowner_teams={
                    "platform": ["rocke-platform"],
                    "library": ["rocke-lib"],
                },
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["codeowners"] = {"teams": {"rocke-platform": ["me"], "rocke-lib": ["y"]}}
            st["prs"] = {
                "1": rec(
                    1,
                    files_cache={
                        "head": head(1),
                        "hot": [],
                        "areas": ["library", "platform"],
                    },
                ),
                "2": rec(2, review_decision="APPROVED"),
            }
            st["batches"] = [
                {
                    "members": [2],
                    "status": "validated",
                    "lines": 100,
                    "check_prefixes": False,
                }
            ]
            st["abi_lanes"] = {
                "binary": {
                    "develop": 6,
                    "head": 0,
                    "groups": [
                        {
                            "members": [2],
                            "root": 2,
                            "to": 7,
                            "reserve": 7,
                            "ready": True,
                        }
                    ],
                }
            }
            q = [
                {
                    "n": 2,
                    "position": 1,
                    "stage": "ready",
                    "blockers": [],
                    "gs": {k: "pass" for k in report.GATES},
                }
            ]
            board = report.board(cfg, st, store, [], q)
            page = webdash.render(cfg, st, store, [], q, board)
        sids = ("approvals", "approvals-rocke-lib", "approvals-any", "merge", "plan")
        order = [page.index(f"<section id='{sid}'") for sid in sids]
        self.assertEqual(order, sorted(order))
        self.assertIn("<h2>Needs platform approval <span class='c'>", page)
        self.assertIn("<h2>Needs rocke-lib approval <span class='c'>", page)
        self.assertIn("<h2>Needs any approval <span class='c'>", page)
        self.assertIn("<h2>Mergeable Candidates <span class='c'>", page)
        self.assertEqual(
            page.count("href='https://github.com/ROCm/r/pull/1'"), 2
        )  # #1 needs both teams: in both boxes
        self.assertIn(
            "<a href='#approvals'>Needs platform <span class='c'>1</span></a>", page
        )
        self.assertIn(
            "<a href='#approvals-rocke-lib'>Needs rocke-lib <span class='c'>1</span></a>",
            page,
        )
        self.assertIn(
            "<a href='#approvals-any'>Needs any <span class='c'>0</span></a>", page
        )
        self.assertIn(
            "<div class='label'>Needs approval</div><div class='value'>1 / 1 / 0</div>",
            page,
        )
        nav = page[page.index("<nav") : page.index("</nav>")]
        self.assertLess(nav.index("href='#merge'"), nav.index("href='#plan'"))
        for gone in (
            "Close to merge",
            "Ready for your approval",
            "id='close'",
            "Ready for platform",
            "Ready for rocke",
            "Merge queue <span",
        ):
            self.assertNotIn(gone, page)
        merge = page[
            page.index("<section id='merge'") : page.index(
                "</section>", page.index("<section id='merge'")
            )
        ]
        self.assertIn(
            "ready · batch 1: validated", merge
        )  # the batch shows in the stage
        self.assertNotIn("Merge batches", merge)
        self.assertNotIn("ABI lanes", merge)


class TestDevelopFixes(unittest.TestCase):
    def test_fixes_climb_to_the_top(self):
        """A PR that fixes a develop break goes first in review and merge."""
        from rkmon_lib import plan

        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["post_merge"] = {
            "at": now(),
            "open": [
                {"n": 11111, "title": "t", "rocke": True, "label": guard_labels.RED}
            ],
            "fixes": {"11111": [3], "999": [4]},
        }
        moved = {"verdict": G, "head": "f" * 40, "at": now()}
        st["prs"] = {
            "1": rec(1, ours=None, head_seen_at=ago(120), size=10),
            "2": rec(2, ours=moved, head_seen_at=ago(120), closes=[11111], size=99999),
            "3": rec(3, ours=None, head_seen_at=ago(15), size=500),
            "4": rec(4, ours=None, head_seen_at=ago(120)),
        }
        fixes = queue.refresh_develop_fixes(
            st,
            [
                {"id": "i1", "status": "open", "fixed_by": 1},
                {"id": "i0", "status": "closed", "fixed_by": 4},
            ],
        )
        self.assertEqual(
            fixes,
            {
                "1": ["incident i1"],
                "2": ["red-develop #11111"],
                "3": ["red-develop #11111"],
            },
        )
        items = [
            i
            for i in plan.make_plan(dict(CFG, settle_min=30), st, [], [])
            if i["kind"] in ("FOLLOW_UP", "FIRST_ROUND")
        ]
        self.assertEqual(
            [(i["pr"], i["prio"]) for i in items][:3], [(1, 0), (2, 0), (3, 0)]
        )
        self.assertTrue(
            items[1]["text"].startswith("develop fix (red-develop #11111): ")
        )
        self.assertNotIn(
            "(wait:", items[2]["text"]
        )  # a fix settles after 10 min, not 30
        self.assertEqual(report.close_to_merge(CFG, st)[0]["fix"], ["incident i1"])
        with breakage_issue() as text:
            self.assertEqual(sorted(breakage.named_fixes(text)), [4567, 9999])

    def test_a_user_priority_floats_like_a_fix(self):
        """An operator's priority (#12345) floats to the top of the review and merge queues like a fix."""
        from rkmon_lib import plan

        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            "1": rec(1, ours=None, head_seen_at=ago(120), size=10),
            "7": rec(7, ours=None, head_seen_at=ago(15), size=99999),
        }
        st["user_priority"] = {
            "7": {"why": "prevents golden conflicts", "at": now()},
            "8": {"why": "not tracked", "at": now()},
        }
        fixes = queue.refresh_develop_fixes(st, [])
        self.assertEqual(fixes, {"7": ["your priority: prevents golden conflicts"]})
        self.assertFalse(queue.fixes_develop(fixes["7"]))
        self.assertTrue(queue.fixes_develop(["red-develop #1", "your priority: x"]))
        items = [
            i
            for i in plan.make_plan(dict(CFG, settle_min=30), st, [], [])
            if i["kind"] in ("FOLLOW_UP", "FIRST_ROUND")
        ]
        self.assertEqual([(i["pr"], i["prio"]) for i in items], [(7, 0), (1, 3)])
        self.assertTrue(
            items[0]["text"].startswith(
                "top priority (your priority: prevents golden conflicts): "
            )
        )
        self.assertNotIn(
            "(wait:", items[0]["text"]
        )  # it settles after 10 min, like a develop fix


class TestTeamsBreakage(unittest.TestCase):
    def test_alert_only_without_a_planned_fix(self):
        """A proven develop break with no planned fix alerts the team channel."""
        import json
        from rkmon_lib import gh

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            hook = pathlib.Path(d) / "hook"
            cfg = dict(
                CFG, teams={"channel": "Team channel", "webhook_file": str(hook)}
            )
            st = empty_state()
            st["prs"] = {"1111": rec(1111, author="dev-a")}
            issue = {
                "number": 11111,
                "title": "[rocke] x fails on develop since #1111",
                "state": "open",
                "html_url": "https://github.com/ROCm/r/issues/11111",
                "body": "**Introduced by:** #1111\n",
                "labels": ["red-develop", "needs-revert"],
            }
            with mock.patch.object(gh, "rest", lambda path, jq=None: dict(issue)):
                with self.assertRaisesRegex(
                    post.PostRefused, "no Teams webhook configured"
                ):
                    post.teams_breakage(cfg, st, store, 11111)
                hook.write_text("https://example.invalid/webhook\n")
                sent = []
                resp = mock.MagicMock(status=202)
                resp.__enter__.return_value = resp

                def urlopen(req, timeout):
                    sent.append(json.loads(req.data))
                    return resp

                with mock.patch("urllib.request.urlopen", urlopen):
                    self.assertIn(
                        "HTTP 202", post.teams_breakage(cfg, st, store, 11111)
                    )
                    with self.assertRaisesRegex(post.PostRefused, "already alerted"):
                        post.teams_breakage(cfg, st, store, 11111)
            card = sent[0]["attachments"][0]["content"]
            text = json.dumps(card)
            from rkmon_lib import reviewer_name

            self.assertIn(f"Automated message by {reviewer_name.name(cfg)}", text)
            self.assertIn("issues/11111", text)
            self.assertIn("Who can take this?", text)
            self.assertIn("#1111 (@dev-a)", text)
            issue["body"] += "**Fix candidate:** #4567\n"
            with mock.patch.object(
                gh, "rest", lambda path, jq=None: dict(issue, number=14631)
            ):
                with self.assertRaisesRegex(
                    post.PostRefused, r"a fix is planned \(#4567\)"
                ):
                    post.teams_breakage(cfg, st, store, 14631)


class _Issue:
    def __enter__(self):
        return (
            "**Fix candidate:** #9999\nThe golden was re-blessed in #8888 earlier.\n"
            "Fixed by #4567 (" + TK + "-2670).\n"
        )

    def __exit__(self, *exc):
        return False


def breakage_issue():
    return _Issue()


class TestRequiredChecks(unittest.TestCase):
    def test_a_required_check_that_never_reported_is_not_green(self):
        """A green rollup whose required check (Math CI Summary) never reported is not green: GitHub blocks it."""
        from rkmon_lib import gh, snapshot

        runs = [
            {"name": "pre-commit", "status": "completed", "conclusion": "success"},
            {
                "name": "Multi-Arch CI Summary",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "name": "Component CI Summary",
                "status": "in_progress",
                "conclusion": None,
            },
        ]
        with mock.patch.object(
            gh,
            "rest",
            lambda path, jq=None, paginate=False: runs if "check-runs" in path else [],
        ):
            got = snapshot.required_status(
                CFG,
                "a" * 40,
                [
                    "Math CI Summary",
                    "pre-commit",
                    "Multi-Arch CI Summary",
                    "Component CI Summary",
                ],
            )
        self.assertEqual(
            got,
            {
                "Math CI Summary": "missing",
                "pre-commit": "success",
                "Multi-Arch CI Summary": "success",
                "Component CI Summary": "pending",
            },
        )
        r = rec(1, required={"head": head(1), "checks": got})
        self.assertEqual(queue.gate_ci(r)[0], queue.PENDING)
        self.assertIn("Math CI Summary (missing)", queue.gate_ci(r)[1])
        self.assertEqual(
            queue.gate_ci(
                dict(r, required={"head": head(1), "checks": {"pre-commit": "failure"}})
            )[0],
            queue.FAIL,
        )
        self.assertEqual(
            queue.gate_ci(dict(r, required={"head": "f" * 40, "checks": got}))[0],
            queue.PASS,
        )  # stale
        left = [
            b for b in report.merge_blockers(CFG, empty_state(), r) if b["kind"] == "ci"
        ]
        self.assertEqual(left[0]["who"], "author or a maintainer (re-run CI or push)")


class TestLibParity(unittest.TestCase):
    def test_library_kernel_changes_get_the_engine_parity_test_when_on(self):
        """Post-merge #11111: library kernels had no byte-identity coverage in our checks."""
        import subprocess
        from rkmon_lib import review

        files = ["rk/library/kernels/common/lightning_indexer.py"]
        self.assertFalse(
            review.plan_checks(CFG, rec(1), files, {})["libparity"]
        )  # off until the fix lands
        plan = review.plan_checks(dict(CFG, libparity_check=True), rec(1), files, {})
        self.assertTrue(plan["libparity"] and plan["build"])
        script = review.checks_script(
            dict(CFG, libparity_check=True), 1, head(1), "review", plan, None
        )
        self.assertIn("test_attention_ir_cpp_parity.py", script)
        r = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            review.gates_from_status({"libparity": ("fail", "not passing at llvm22")})[
                "G6"
            ][0],
            "fail",
        )


class TestReviewSkipLoop(unittest.TestCase):
    def test_a_stale_plan_never_loops(self):
        """A rebase-only follow-up is never re-picked from the tick's stale plan in a loop."""
        from rkmon_lib import review

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "1": rec(
                    1,
                    ours={
                        "verdict": G,
                        "head": "f" * 40,
                        "equivalent_heads": [head(1)],
                    },
                ),
                "2": rec(2, ours={"verdict": G, "head": "e" * 40}),
            }
            plan = [
                {"prio": 1, "kind": "FOLLOW_UP", "pr": n, "text": "", "critical": False}
                for n in (1, 2)
            ]
            self.assertEqual(
                review.pick_next(CFG, st, store, plan)["pr"], 2
            )  # #1's head is already covered
            calls = []

            def skip(cfg, st_, store_, n, kind, again=False):
                calls.append(n)
                return [f"REVIEW_SKIP #{n}: rebase-only"]

            with mock.patch.object(review, "start_review", skip):
                lines = review.advance(CFG, st, store, plan)
        # Bounded, even when nothing records the skip: each PR is tried once per advance (a PR found closed on GitHub, or
        # with a newer live head still settling, records nothing either).
        self.assertEqual(calls, [2])
        self.assertTrue(all(line.startswith("REVIEW_SKIP") for line in lines))


class TestHoldReviews(unittest.TestCase):
    """The hold rules: a hold requests changes from our account, so it cannot merge until fixed; the
    author fixes it and re-requests the operator's review; only the operator's explicit re-approval lifts it.
    We never approve or dismiss."""

    @staticmethod
    def held():
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["codeowners"] = {"teams": {"plat": ["me", "x"]}}
        st["prs"] = {
            "1": rec(
                1,
                review_decision="APPROVED",
                reviews=[{"login": "x", "state": "APPROVED", "commit": "0" * 40}],
            )
        }
        st["holds"] = {
            "1": {
                "head": head(1),
                "status": "held",
                "reason": "G5 dg: 13 unblessed",
                "since": now(),
            }
        }
        return st

    def test_a_confirmed_hold_needs_a_change_request(self):
        st = self.held()
        lines = merge.hold_reviews(CFG, st)
        self.assertTrue(
            lines and lines[0].startswith("!! HOLD_REVIEW_NEEDED #1"), lines
        )
        self.assertIn("rk request-changes 1 FILE", lines[0])
        st["prs"]["1"]["reviews"].append(
            {"login": "me", "state": "CHANGES_REQUESTED", "commit": head(1)}
        )
        self.assertEqual(
            merge.hold_reviews(CFG, st), []
        )  # the user's own change request counts (#2345)
        self.assertEqual(st["hold_reviews"]["1"]["by"], "you")
        for change in (
            {"status": "triage"},
            {"head": "f" * 40},
        ):  # untriaged, or ended by a push
            st = self.held()
            st["holds"]["1"].update(change)
            self.assertEqual(merge.hold_reviews(CFG, st), [], change)
        st = self.held()
        st["prs"]["1"]["author"] = "me"  # GitHub does not let an author request changes
        self.assertEqual(merge.hold_reviews(CFG, st), [])

    def test_request_changes_guards(self):
        """A change request's guards, and its ledger (the merge queue live: in prototype a write typed is refused
        before its guards)."""
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = self.held()
            body, bad = pathlib.Path(d) / "b.md", pathlib.Path(d) / "bad.md"
            body.write_text(
                post.notice_line(GUARD_ACT, "hold")
                + "\n\nWhat fails, how to fix it; re-request a review once fixed.\n"
            )
            bad.write_text("Hello\n")
            with self.assertRaisesRegex(post.PostRefused, "no confirmed hold"):
                post.request_changes(GUARD_ACT, dict(st, holds={}), store, 1, str(body))
            with self.assertRaisesRegex(post.PostRefused, "line 1"):
                post.request_changes(GUARD_ACT, st, store, 1, str(bad))
            with mock.patch(
                "rkmon_lib.gh.rest",
                return_value={"state": "closed", "merged": True, "head": head(1)},
            ):
                with self.assertRaisesRegex(
                    post.PostRefused, "merged: a change request cannot block it"
                ):
                    post.request_changes(GUARD_ACT, st, store, 1, str(body))
            ran = []

            def run(args, **kw):
                ran.append(args)
                return mock.Mock(returncode=0, stdout="", stderr="")

            seen = [
                [],
                [{"id": 77, "state": "CHANGES_REQUESTED", "submitted_at": now()}],
            ]
            with mock.patch(
                "rkmon_lib.gh.rest",
                return_value={"state": "open", "merged": False, "head": head(1)},
            ), mock.patch.object(
                post, "my_reviews", lambda cfg, n: seen.pop(0)
            ), mock.patch.object(
                post.subprocess, "run", run
            ), mock.patch.object(
                post.daylog, "add"
            ):
                url = post.request_changes(GUARD_ACT, st, store, 1, str(body))
            self.assertEqual(
                url, "https://github.com/ROCm/r/pull/1#pullrequestreview-77"
            )
            self.assertEqual(
                ran,
                [
                    [
                        "gh",
                        "pr",
                        "review",
                        "1",
                        "--repo",
                        "ROCm/r",
                        "--request-changes",
                        "--body-file",
                        str(body),
                    ]
                ],
            )
            self.assertEqual(
                (st["hold_reviews"]["1"]["by"], st["hold_reviews"]["1"]["head"]),
                ("us", head(1)),
            )
            self.assertTrue(store.ledger_has("request-changes", 1, head(1)))
            with self.assertRaisesRegex(post.PostRefused, "already requested"):
                post.request_changes(GUARD_ACT, st, store, 1, str(body))
            st["prs"]["1"]["author"] = "me"
            with self.assertRaisesRegex(post.PostRefused, "own PR"):
                post.request_changes(GUARD_ACT, st, store, 1, str(body), user=True)

    def test_the_fix_needs_a_re_request_a_re_gate_and_the_users_re_approval(self):
        st = self.held()
        pr = st["prs"]["1"]
        pr["reviews"].append(
            {"login": "me", "state": "CHANGES_REQUESTED", "commit": head(1)}
        )
        pr["review_decision"] = "CHANGES_REQUESTED"
        merge.hold_reviews(CFG, st)
        kinds = [b["kind"] for b in report.merge_blockers(CFG, st, pr)]
        self.assertIn("hold", kinds)  # while held, the hold item says what to fix
        self.assertNotIn("changes", kinds)
        new = (
            "e" * 40
        )  # the author pushes the fix; our follow-up review of it is current
        pr.update(head=new, ours=dict(pr["ours"], head=new))
        lines = merge.hold_reviews(CFG, st)
        self.assertTrue(lines and lines[0].startswith("HOLD_FIXED #1"), lines)
        self.assertEqual(merge.hold_reviews(CFG, st), [])  # once per head
        self.assertIn(
            "rereview", [b["kind"] for b in report.merge_blockers(CFG, st, pr)]
        )
        g2 = queue.gate_humans(CFG, pr, hold_cr=st["hold_reviews"]["1"])
        self.assertEqual(g2[0], queue.PENDING)
        self.assertIn("re-requests your review", g2[1])
        self.assertEqual(report.approval_candidates(CFG, st), [])
        pr["review_requests"] = ["me"]  # re-requested, but the fix is not re-gated yet
        self.assertIn("regate", [b["kind"] for b in report.merge_blockers(CFG, st, pr)])
        self.assertEqual(report.approval_candidates(CFG, st), [])
        self.assertIn(
            "explicit re-approval",
            queue.gate_humans(CFG, pr, hold_cr=st["hold_reviews"]["1"])[1],
        )
        pr["gates"] = {
            "head": new,
            "develop": "d" * 40,
            "results": {k: ["pass", ""] for k in ("G5", "G6", "G7")},
        }
        a = report.approval_candidates(CFG, st)
        self.assertEqual([x["n"] for x in a], [1])
        self.assertIn("it needs your explicit re-approval", a[0]["why"])
        pr["reviews"] = [r for r in pr["reviews"] if r["login"] != "me"] + [
            {"login": "me", "state": "APPROVED", "commit": new}
        ]
        lines = merge.hold_reviews(CFG, st)
        self.assertTrue(lines and lines[0].startswith("HOLD_REAPPROVED #1"), lines)
        self.assertNotIn("1", st["hold_reviews"])

    def test_a_dismissed_hold_change_request_still_needs_the_users_re_approval(self):
        """An author who dismisses the hold's change request ("already included in my follow up commits") does not
        lift it: the tick never reads it as HOLD_REAPPROVED."""
        st = self.held()
        pr = st["prs"]["1"]
        pr["reviews"].append(
            {"login": "me", "state": "CHANGES_REQUESTED", "commit": head(1)}
        )
        merge.hold_reviews(CFG, st)
        new = "e" * 40
        pr.update(head=new, ours=dict(pr["ours"], head=new))
        merge.hold_reviews(CFG, st)  # HOLD_FIXED
        pr["reviews"] = [
            r for r in pr["reviews"] if r["login"] != "me"
        ]  # the author dismissed it
        lines = merge.hold_reviews(CFG, st)
        self.assertTrue(lines and lines[0].startswith("!! HOLD_DISMISSED #1"), lines)
        self.assertIn("1", st["hold_reviews"])
        self.assertEqual(merge.hold_reviews(CFG, st), [])  # said once
        g2 = queue.gate_humans(CFG, pr, hold_cr=st["hold_reviews"]["1"])
        self.assertEqual(g2[0], queue.PENDING)
        self.assertIn("dismissed", g2[1])
        self.assertIn(
            "you (re-approve explicitly)",
            [b["who"] for b in report.merge_blockers(CFG, st, pr)],
        )
        pr["reviews"].append({"login": "me", "state": "APPROVED", "commit": new})
        lines = merge.hold_reviews(CFG, st)
        self.assertTrue(lines and lines[0].startswith("HOLD_REAPPROVED #1"), lines)
        self.assertNotIn("1", st["hold_reviews"])

    def test_a_held_prs_fix_is_analysed_and_reviewed_first(self):
        from rkmon_lib import plan

        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            "1": rec(
                1,
                review_decision="CHANGES_REQUESTED",
                head_seen_at=ago(120),
                reviews=[
                    {"login": "me", "state": "CHANGES_REQUESTED", "commit": "0" * 40}
                ],
            )
        }
        st["hold_reviews"] = {
            "1": {
                "head": "0" * 40,
                "reason": "G5 dg",
                "by": "you",
                "at": now(),
                "url": "",
            }
        }

        def stage():
            with mock.patch.object(
                queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
            ), mock.patch.object(
                queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
            ), mock.patch.object(
                queue, "rocke_tree", lambda cfg, sha: None
            ), mock.patch.object(
                queue, "disruption", lambda cfg, st, cands: {r["n"]: [] for r in cands}
            ), mock.patch.object(
                queue, "abi_lanes", lambda *a: {}
            ), mock.patch.object(
                queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
            ), mock.patch.object(
                queue.batch, "build_batches", lambda *a: []
            ):
                return queue.build_queue(CFG, st, "d" * 40)[0]["stage"]

        self.assertEqual(stage(), "candidate")  # its merge analysis re-gates the fix
        st["hold_reviews"] = {}
        self.assertIsNone(stage())  # anyone else's change request still keeps it out
        st["hold_reviews"] = {
            "1": {
                "head": "0" * 40,
                "reason": "G5 dg",
                "by": "you",
                "at": now(),
                "url": "",
            }
        }
        st["prs"]["1"]["ours"] = {
            "verdict": G,
            "head": "f" * 40,
            "at": now(),
        }  # our review predates the fix
        item = next(
            i
            for i in plan.make_plan(dict(CFG, settle_min=30), st, [], [])
            if i["kind"] == "FOLLOW_UP"
        )
        self.assertEqual(item["prio"], 1)
        self.assertTrue(item["text"].startswith("final review (held PR's fix)"))

    def test_a_pr_back_from_a_hold_stays_on_the_board_and_tops_the_close_list(self):
        """A PR back from a hold stays on the board and tops the close list in `rk approvals`."""
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                CFG,
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {
                "1": rec(
                    1,
                    review_decision="CHANGES_REQUESTED",
                    reviews=[
                        {
                            "login": "me",
                            "state": "CHANGES_REQUESTED",
                            "commit": "0" * 40,
                        }
                    ],
                ),
                "2": rec(2),
            }
            self.assertNotIn("| Holds |", report.board(cfg, st, store, [], [])["md"])
            st["hold_reviews"] = {
                "1": {
                    "head": "0" * 40,
                    "reason": "G5 dg",
                    "by": "you",
                    "at": now(),
                    "url": "",
                }
            }
            md = report.board(cfg, st, store, [], [])["md"]
            self.assertIn(
                "| Holds | 0 held; 1 back from a hold, it needs your explicit re-approval",
                md,
            )
            self.assertIn(
                "#1 (a, pushed 1111111111 since the hold): held earlier (G5 dg): re-request a review from me",
                md,
            )
            close = report.close_to_merge(cfg, st)
            self.assertGreater(len(close[0]["blockers"]), len(close[1]["blockers"]))
            self.assertEqual(close[0]["n"], 1)  # first despite more items left


if __name__ == "__main__":
    unittest.main()
