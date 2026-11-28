# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for Phase 3: gate/batch scripts, gates from status, queue advance decisions, arm guards, notices."""

import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import merge, paths, post, queue, review  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

CFG = {
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "paths": {"working_dir": "/nonexistent", "repo": "/nonexistent"},
    "features": {"merge_queue": {"mode": "live", "auto_merge": "enabled"}},
    "max_posts_per_hour": 6,
    "coordination_notices": True,
    "review_first_line": "x",
}
FAMS = {"gemm": "platform/tests/instances/parity"}


def rec(n, size=100):
    return {
        "n": n,
        "head": (str(n) * 40)[:40],
        "size": size,
        "status": "open",
        "draft": False,
        "files_cache": {"hot": []},
    }


def plan_patches():
    return (
        mock.patch.object(
            review,
            "_changed_files",
            lambda cfg, r: ["rk/platform/x/gemm.py", "rk/tests/test_x.py"],
        ),
        mock.patch.object(merge.guard, "emitter_families_at", lambda cfg, sha: FAMS),
    )


class TestScripts(unittest.TestCase):
    def test_gate_and_batch_scripts_parse(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {"1": rec(1), "2": rec(2)}
        a, b = plan_patches()
        with a, b:
            g = merge.gate_script(CFG, st["prs"]["1"], "d" * 40)
            bs = merge.batch_script(
                CFG,
                st,
                {
                    "members": [1, 2],
                    "heads": {},
                    "develop": "d" * 40,
                    "check_prefixes": True,
                },
            )
        for s in (g, bs):
            r = subprocess.run(["bash", "-n"], input=s, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("checks.md", g)
        self.assertIn("step1.sh", bs)
        self.assertIn("step2.sh", bs)
        self.assertIn("echo validated", bs)
        # The batch's tree is one PR code reads: its folder is made so before git fills it.
        self.assertIn(
            'if [ ! -d "$WT" ]; then prtree "$WT" && git -C "$R0" worktree add ', bs
        )

    def test_the_batch_s_folders_are_quoted_wherever_the_batch_names_them(self):
        """The batch's folder (OUT), its tree (WT, the working folder's wtbatch) and the clone (R0) are quoted
        wherever the batch script's own lines name them (the steps' scripts it writes are the review checks'): under a
        working folder with a space in its path, the batch's result and its tree are made there, git is given each
        folder whole, and nothing is made at the path's first word."""
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {"1": rec(1), "2": rec(2)}
        batch = {
            "members": [1, 2],
            "heads": {},
            "develop": "d" * 40,
            "check_prefixes": True,
        }
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(CFG, paths=dict(CFG["paths"], working_dir=f"{d}/a b"))
            a, b = plan_patches()
            with a, b:
                bs = merge.batch_script(cfg, st, batch)
            own, inside = [], False
            for line in bs.splitlines():
                if line == "RKMON_EOF":
                    inside = False
                elif not inside:
                    own.append(line)
                    inside = line.endswith("<<'RKMON_EOF'")
            self.assertEqual(
                [ln for ln in own if re.search(r'(?<!["/])\$(OUT|WT|R0|D)\b', ln)], []
            )
            first = next(line for line in own if line.startswith("WT="))
            made = next(line for line in own if line.startswith("if [ ! -d "))
            # The opening lines as they run, with prtree a plain folder and git a stand-in that writes its arguments.
            stand_ins = 'prtree() { mkdir -p "$1"; }; git() { printf \'%s\\n\' "$@" >> "$GIT_ARGS"; }'
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    "\n".join(
                        [own[0], f"GIT_ARGS={shlex.quote(d)}/git_args", stand_ins]
                        + [first, made]
                    ),
                ],
                cwd=d,
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            bid = merge.batch_id([1, 2], "d" * 40)
            self.assertTrue(
                (pathlib.Path(d) / "a b" / "workers" / bid / "result").is_file()
            )
            self.assertTrue((pathlib.Path(d) / "a b" / "wtbatch").is_dir())
            given = (pathlib.Path(d) / "git_args").read_text().splitlines()
            self.assertEqual(given[:2], ["-C", paths.get(cfg, "repo")])
            self.assertIn(f"{d}/a b/wtbatch", given)
            self.assertEqual(
                sorted(p.name for p in pathlib.Path(d).iterdir()), ["a b", "git_args"]
            )


class TestGates(unittest.TestCase):
    def test_gates_from_status(self):
        st = {
            "merge": ("pass", "clean"),
            "build": ("pass", "100%"),
            "byteid": ("pass", "x3"),
            "golden": ("pass", "x3"),
            "dg": ("fail", "re-bless needed"),
            "compile": ("pass", ""),
        }
        g = review.gates_from_status(st)
        self.assertEqual(g["G5"][0], "fail")
        self.assertEqual(g["G6"][0], "pass")
        self.assertEqual(g["G7"][0], "pass")
        self.assertEqual(review.gates_from_status({})["G6"][0], "pending")


class TestAdvance(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40
        self.st["prs"] = {"1": rec(1), "2": rec(2)}

    def tearDown(self):
        self.d.cleanup()

    def test_single_with_fresh_gates_proposes(self):
        self.st["batches"] = [
            {
                "members": [1],
                "heads": {"1": rec(1)["head"]},
                "develop": "d" * 40,
                "check_prefixes": False,
                "status": "candidate",
            }
        ]
        self.st["prs"]["1"]["gates"] = {
            "head": rec(1)["head"],
            "develop": "d" * 40,
            "results": {k: ["pass", ""] for k in ("G5", "G6", "G7")},
        }
        q = [{"n": 1, "gates": {k: ["pass", ""] for k in ("G5", "G6", "G7", "G9")}}]
        lines = merge.advance(CFG, self.st, self.store, q, [])
        self.assertTrue(any(line.startswith("PROPOSE #1") for line in lines), lines)
        self.st["cleared"] = [1]
        lines = merge.advance(CFG, self.st, self.store, q, [])
        self.assertTrue(any(line.startswith("ARM_READY #1") for line in lines), lines)

    def test_only_a_prefix_of_a_validated_batch_carries(self):
        # When the first member leaves a validated batch, the rest alone is not proposed as validated.
        heads = {"1": rec(1)["head"], "2": rec(2)["head"]}
        self.st["queue_state"] = {
            "validated": {"members": [1, 2], "heads": heads, "develop": "d" * 40}
        }
        self.st["jobs"]["heavy"] = {
            "id": "busy"
        }  # no gate starts here: only the validation decision is under test
        for members, proposed in (([2], False), ([1], True)):
            self.st["batches"] = [
                {
                    "members": members,
                    "heads": {str(n): heads[str(n)] for n in members},
                    "develop": "d" * 40,
                    "status": "candidate",
                }
            ]
            with mock.patch.object(merge.jobs, "status", return_value="running"):
                lines = merge.advance(CFG, self.st, self.store, [], [])
            self.assertEqual(
                any(line.startswith(f"PROPOSE #{members[0]}") for line in lines),
                proposed,
                lines,
            )
            self.assertEqual(self.st["batches"][0]["status"] == "validated", proposed)

    def test_gate_done_validates_in_the_same_advance(self):
        head = rec(1)["head"]
        self.st["batches"] = [
            {
                "members": [1],
                "heads": {"1": head},
                "develop": "d" * 40,
                "check_prefixes": False,
                "status": "candidate",
            }
        ]
        self.st["queue_state"] = {
            "gate": {"pr": 1, "head": head, "develop": "d" * 40, "job": "gate-pr1-t"}
        }
        status = {
            k: ("pass", "")
            for k in (
                "merge",
                "build",
                "byteid",
                "byteid_all",
                "tests",
                "golden",
                "dg",
                "compile",
            )
        }
        q = [
            {
                "n": 1,
                "gates": {
                    "G5": ["pending", ""],
                    "G6": ["pending", ""],
                    "G7": ["pending", ""],
                    "G9": ["pass", ""],
                },
            }
        ]
        with mock.patch.object(
            merge.jobs, "status", lambda store, job: "done"
        ), mock.patch.object(
            merge.review, "read_checks_status", lambda path: (status, "d" * 40, head)
        ):
            lines = merge.advance(CFG, self.st, self.store, q, [])
        self.assertTrue(any(line.startswith("GATE_DONE #1") for line in lines), lines)
        self.assertFalse(any(line.startswith("GATE_START") for line in lines), lines)
        self.assertTrue(any(line.startswith("PROPOSE #1") for line in lines), lines)

    def test_disarm_on_freeze_and_push(self):
        self.st["queue_state"] = {
            "armed": {"pr": 1, "head": rec(1)["head"], "develop": "d" * 40, "at": "t"}
        }
        self.st["frozen"] = {"reason": "develop red"}
        lines = merge.advance(CFG, self.st, self.store, [], [])
        self.assertTrue(
            any(line.startswith("DISARM_NEEDED #1") for line in lines), lines
        )
        self.st["frozen"] = None
        lines = merge.advance(CFG, self.st, self.store, [], [{"kind": "PUSH", "pr": 1}])
        self.assertIn("head moved", " ".join(lines))
        lines = merge.advance(
            CFG, self.st, self.store, [], [{"kind": "MERGED", "pr": 1}]
        )
        self.assertTrue(any(line.startswith("LANDED #1") for line in lines))
        self.assertIsNone(self.st["queue_state"]["armed"])


class TestHolds(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40
        self.st["prs"] = {"1": rec(1), "2": rec(2)}

    def tearDown(self):
        self.d.cleanup()

    def test_gate_failure_holds_instead_of_regating(self):
        head = rec(1)["head"]
        self.st["batches"] = [
            {
                "members": [1],
                "heads": {"1": head},
                "develop": "d" * 40,
                "check_prefixes": False,
                "status": "candidate",
            }
        ]
        self.st["queue_state"] = {
            "gate": {"pr": 1, "head": head, "develop": "d" * 40, "job": "gate-pr1-t"}
        }
        status = {
            "merge": ("pass", ""),
            "build": ("pass", ""),
            "golden": ("pass", ""),
            "dg": ("pass", ""),
            "compile": ("fail", "gemm: 1 newly rejected"),
        }
        with mock.patch.object(
            merge.jobs, "status", lambda store, job: "done"
        ), mock.patch.object(
            merge.review, "read_checks_status", lambda path: (status, "d" * 40, head)
        ):
            lines = merge.advance(CFG, self.st, self.store, [], [])
        self.assertTrue(any(line.startswith("TRIAGE #1") for line in lines), lines)
        self.assertFalse(any(line.startswith("GATE_START") for line in lines), lines)
        self.assertEqual(self.st["holds"]["1"]["status"], "triage")
        self.assertIn("newly rejected", self.st["holds"]["1"]["reason"])

    def test_push_clears_hold_and_stale_is_told_once(self):
        self.st["holds"] = {
            "1": {
                "head": "old" * 13 + "x",
                "reason": "r",
                "status": "held",
                "since": now(),
                "source": "gate",
            },
            "2": {
                "head": rec(2)["head"],
                "reason": "G5 golden: not PASS",
                "status": "held",
                "since": "2000-01-01T00:00:00Z",
                "source": "gate",
            },
        }
        lines = merge.advance(CFG, self.st, self.store, [], [])
        self.assertTrue(
            any(line.startswith("HOLD_CLEARED #1: pushed") for line in lines), lines
        )
        self.assertNotIn("1", self.st["holds"])
        self.assertTrue(any(line.startswith("HOLD_STALE #2") for line in lines), lines)
        lines = merge.advance(CFG, self.st, self.store, [], [])
        self.assertFalse(any(line.startswith("HOLD_STALE") for line in lines), lines)

    def test_a_held_pr_that_merges_anyway_is_loud(self):
        """A held PR merged anyway, at the held head or after a push, is a loud line."""
        self.st["holds"] = {
            "1": {
                "head": rec(1)["head"],
                "reason": "G6: its new test fails without a device",
                "status": "held",
                "since": now(),
                "source": "gate",
            },
            "2": {
                "head": "old" * 13 + "x",
                "reason": "r",
                "status": "held",
                "since": now(),
                "source": "gate",
            },
        }
        self.st["prs"]["1"]["status"] = "merged"
        self.st["prs"]["2"]["status"] = "merged"
        lines = merge.advance(
            CFG,
            self.st,
            self.store,
            [],
            [{"kind": "MERGED", "pr": 1, "by": "dev1"}, {"kind": "MERGED", "pr": 2}],
        )
        loud = [line for line in lines if line.startswith("!! HELD_MERGED")]
        self.assertEqual(len(loud), 2, lines)
        self.assertIn(
            "#1: merged while held (by dev1, at the held head 1111111111)", loud[0]
        )
        self.assertIn("G6: its new test fails without a device", loud[0])
        self.assertIn(
            "#2: merged while held (by someone, after a push since the hold)", loud[1]
        )
        self.assertTrue(
            any(line.startswith("HOLD_CLEARED #1: closed or merged") for line in lines),
            lines,
        )
        self.assertEqual(self.st["holds"], {})

    def test_a_merge_over_our_open_red_or_an_unapproved_hold_is_loud(self):
        """A merge at the head our red follow-up named, its hold's change request dismissed and no re-approval, is
        said at once, even after the PR left the holds, not only when develop's sanity fails.
        """
        for n in (1, 2, 3, 4):
            self.st["prs"][str(n)] = dict(rec(n), status="merged", reviews=[])
        self.st["prs"]["1"]["ours"] = {
            "head": rec(1)["head"],
            "verdict": "\U0001f534",
        }  # our red at the merged head
        self.st["prs"]["2"]["ours"] = {
            "head": rec(2)["head"],
            "verdict": "\U0001f7e1",
        }  # yellow: fine
        self.st["prs"]["3"]["ours"] = {
            "head": "old" * 13 + "x",
            "verdict": "\U0001f7e0",
        }  # orange on an older head
        self.st["hold_reviews"] = {"4": {"dismissed": True, "reason": "r", "head": "x"}}
        self.st["prs"]["4"]["reviews"] = [{"login": "me", "state": "DISMISSED"}]
        cfg = dict(CFG, operator={"github_login": "me"})
        lines = merge.advance(
            cfg,
            self.st,
            self.store,
            [],
            [{"kind": "MERGED", "pr": n, "by": "dev1"} for n in (1, 2, 3, 4)],
        )
        loud = [line for line in lines if line.startswith("!! MERGED_OVER_RED")]
        self.assertEqual(len(loud), 2, lines)
        self.assertTrue(
            loud[0].startswith(
                "!! MERGED_OVER_RED #1: merged by dev1 (our \U0001f534 review at 1111111111 "
                "is open)"
            ),
            loud[0],
        )
        self.assertIn(
            "#4: merged by dev1 (held earlier: its change request was dismissed without your re-approval)",
            loud[1],
        )
        self.st["prs"]["4"]["reviews"] = [
            {"login": "me", "state": "APPROVED"}
        ]  # re-approved: quiet
        lines = merge.advance(
            cfg, self.st, self.store, [], [{"kind": "MERGED", "pr": 4, "by": "dev1"}]
        )
        self.assertFalse(
            any(line.startswith("!! MERGED_OVER_RED") for line in lines), lines
        )

    def test_batch_failure_holds_the_failing_member(self):
        self.st["queue_state"] = {
            "batch_gate": {
                "id": "bdddddddd-1-2",
                "members": [1, 2],
                "heads": {"1": rec(1)["head"], "2": rec(2)["head"]},
                "develop": "d" * 40,
                "job": "batch-t",
            }
        }
        with mock.patch.object(
            merge.jobs, "status", lambda store, job: "done"
        ), mock.patch.object(
            merge, "_read", lambda path: "fail at step 2 (#2): compile "
        ):
            lines = merge.advance(CFG, self.st, self.store, [], [])
        self.assertTrue(any(line.startswith("TRIAGE #2") for line in lines), lines)
        self.assertEqual(self.st["holds"]["2"]["source"], "batch")
        self.assertNotIn("1", self.st["holds"])

    def test_blocked_notice_only_after_triage(self):
        self.st["holds"] = {
            "1": {
                "head": rec(1)["head"],
                "reason": "G5 golden: not PASS",
                "status": "triage",
                "since": now(),
                "source": "gate",
            }
        }
        self.assertNotIn(1, merge.desired_notices(CFG, self.st, []))
        self.st["holds"]["1"]["status"] = "held"
        self.assertEqual(merge.desired_notices(CFG, self.st, [])[1][0], "blocked")


class TestApplyHolds(unittest.TestCase):
    CFG = dict(CFG, hot_spots={"ABI": "rk/platform/cpp/include/rocke/abi.h"})

    def rows(self):
        return [{"n": n, "blockers": [], "static_ok": True} for n in (1, 2, 3, 4)]

    def test_held_waits_impacting_waits_lane_and_unrelated_proceed(self):
        st = empty_state()
        st["prs"] = {str(n): rec(n) for n in (1, 2, 3, 4)}
        st["holds"] = {
            "1": {
                "head": rec(1)["head"],
                "reason": "G5 golden",
                "status": "held",
                "since": now(),
            }
        }
        k = f"{rec(1)['head'][:12]}:{rec(3)['head'][:12]}"
        st["pair_cache"] = {
            k: {
                "conflicts": ["rk/platform/cpp/include/rocke/abi.h"],
                "undecided": False,
            }
        }
        score = {
            1: [(2, ["RG"]), (3, ["ABI", "files"])],
            2: [(1, ["RG"])],
            3: [(1, ["ABI", "files"])],
            4: [],
        }
        rows = self.rows()
        queue.apply_holds(self.CFG, st, rows, score)
        by = {r["n"]: r for r in rows}
        self.assertFalse(by[1]["static_ok"])
        self.assertTrue(by[1]["blockers"][0].startswith("HOLD (held)"))
        self.assertFalse(
            by[2]["static_ok"]
        )  # landing #2 would force held #1 to re-bless the golden
        self.assertIn("held #1", by[2]["blockers"][0])
        self.assertTrue(
            by[3]["static_ok"]
        )  # only the ABI counter: readiness order lets it take the number
        self.assertTrue(by[4]["static_ok"])

    def test_hold_ends_when_the_head_moves(self):
        st = empty_state()
        st["prs"] = {"1": rec(1)}
        st["holds"] = {
            "1": {"head": "f" * 40, "reason": "r", "status": "held", "since": now()}
        }
        self.assertEqual(queue.active_holds(st), {})


class TestCarry(unittest.TestCase):
    def test_develop_moves_outside_rocke_keep_a_validation(self):
        """A CK-only develop commit does not re-run a batch gate that just passed on the same rocKE tree."""
        st = empty_state()
        st["develop"]["sha"] = "n" * 40
        val = {"develop": "o" * 40, "members": [1, 2]}
        ck, one, two, other = (
            ("c" * 40, "ck: fmha (#900)"),
            ("a" * 40, "feat: one (#1)"),
            ("b" * 40, "feat: two (#2)"),
            ("e" * 40, "rocke: other (#901)"),
        )

        def carry(rows, inside):  # rows newest first, as git log prints them
            fps = lambda repo, old, new, path=None: (
                inside if path else rows
            )  # noqa: E731
            with mock.patch.object(merge.gitops, "first_parent_since", side_effect=fps):
                return merge._carry(CFG, st, val)

        self.assertTrue(carry([ck], []))
        self.assertTrue(carry([ck, one], [one]))
        self.assertTrue(carry([two, ck, one], [two, one]))
        self.assertFalse(carry([other], [other]))
        self.assertFalse(carry([two], [two]))  # out of order
        self.assertFalse(carry([one, other], [one, other]))

    def test_gate_cancel_stops_a_batch_gate(self):
        st = empty_state()
        job = {"id": "batch-x", "kind": "batch"}
        st["queue_state"] = {"batch_gate": {"id": "b", "members": [1, 2], "job": job}}
        st["jobs"]["heavy"] = job
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            merge.jobs, "kill"
        ) as kill:
            out = merge.cancel_gate(st, Store(pathlib.Path(d)), "still validated")
        kill.assert_called_once()
        self.assertIn("#1 → #2", out)
        self.assertIsNone(st["queue_state"]["batch_gate"])
        self.assertIsNone(st["jobs"]["heavy"])


class TestArmGuards(unittest.TestCase):
    def test_refusals(self):
        """Every reason an arm is refused, said at once: the gate's with the queue's own."""
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {"1": rec(1)}
            with self.assertRaises(post.PostRefused) as cm:
                post.arm(
                    dict(CFG, features={"merge_queue": {"auto_merge": "enabled"}}),
                    st,
                    store,
                    1,
                )  # the merge queue in prototype, and a person typed it
            msg = str(cm.exception)
            self.assertIn(
                "auto-merge is not sent: features.merge_queue.mode is prototype: nothing goes out until it is live",
                msg,
            )
            self.assertIn("not cleared", msg)
            self.assertIn("not the next member of a validated batch", msg)

    def test_the_guard_leaves_your_own_pull_request_by_the_deploy_setting(self):
        """While the guard acts, it turns auto-merge off on your own pull request only when the deploy settings treat
        yours as everyone else's: its refusal names that key and its value as the deploy settings file spells them.
        """
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            st["prs"] = {"1": rec(1)}
            st["automerge_risks"] = {"1": {"head": rec(1)["head"], "reasons": ["r"]}}
            with mock.patch(
                "rkmon_lib.features.guard_mode", return_value="act"
            ), mock.patch.object(
                merge, "automerge_guard_acts", return_value=False
            ), self.assertRaises(
                post.PostRefused
            ) as cm:
                post.automerge_off(CFG, st, store, 1)
        self.assertIn(
            "it is their own PR and features.merge_queue.auto_merge_guard.on_your_pull_requests is not "
            "same_as_others: alert them instead",
            str(cm.exception),
        )


class TestNotices(unittest.TestCase):
    def test_text_and_desired(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["batches"] = [{"members": [3, 4], "status": "validated"}]
        want = merge.desired_notices(CFG, st, [])
        self.assertEqual(want[3][0], "batch")
        text = merge.notice_text(CFG, st, 3, *want[3])
        line = post.notice_line(CFG, "queue")
        self.assertTrue(text.startswith(line + "\n\n"))
        self.assertIn("#3 → #4", text)
        self.assertIn("This monitor validates merges against today's develop", text)
        self.assertEqual(post.check(dict(CFG, review_first_line=line), text), [])


if __name__ == "__main__":
    unittest.main()


class TestHeavySlot(unittest.TestCase):
    """A finished gate never clears the slot a develop sanity has just taken, and no gate starts on a PR that
    has merged."""

    def test_a_finished_job_frees_only_its_own_slot(self):
        from rkmon_lib import jobs

        st = {"jobs": {"heavy": {"id": "sanity-1"}}}
        jobs.release_heavy(st, {"id": "gate-pr1-x"})
        self.assertEqual(st["jobs"]["heavy"], {"id": "sanity-1"})
        jobs.release_heavy(st, {"id": "sanity-1"})
        self.assertIsNone(st["jobs"]["heavy"])

    def test_no_gate_on_a_merged_pr(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {"1": rec(1)}
            st["develop"]["sha"] = "d" * 40
            with mock.patch.object(
                merge.gh, "rest", lambda path, jq=None: "closed"
            ), mock.patch.object(
                merge.jobs, "start", side_effect=AssertionError("started a gate")
            ):
                out = merge.start_gate(CFG, st, store, 1)
        self.assertEqual(
            out,
            ["GATE_SKIP #1: it is closed on GitHub; the PR list catches up next tick"],
        )
        self.assertFalse((st.get("queue_state") or {}).get("gate"))
