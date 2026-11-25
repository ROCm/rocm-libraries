# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for rkmon's pure logic: verdict parsing, the event diff, gates, ordering, the store and the GET guard."""

import os
import pathlib
import re
import shutil
import subprocess
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

from rkmon_lib import config, events, gh, gitops, plan, queue, snapshot  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

GREEN, ORANGE, RED, YELLOW = "\U0001f7e2", "\U0001f7e0", "\U0001f534", "\U0001f7e1"
CFG = {
    "incident_issue": guard_labels.INCIDENT_ISSUE,
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "scope_label": "x",
    "rocke_path": "rk",
    "features": {"merge_queue": {"auto_merge": "enabled"}},
    "settle_min": 30,
    "abi_chain": [3, 1, 2],
    "hot_spots": {"RG": "rk/golden.json", "ABI": "rk/abi.h"},
    "develop_rocke_job_pattern": "rocke",
    "paths": {
        "working_dir": "/nonexistent",
        "repo": "/nonexistent",
        "workers": "/nonexistent",
    },
}


def pr(n, head="a" * 40, **kw):
    base = {
        "n": n,
        "title": f"t{n}",
        "author": "dev",
        "draft": False,
        "labels": ["x"],
        "base": "develop",
        "head_ref": f"b{n}",
        "head": head,
        "created_at": "2026-10-20T00:00:00Z",
        "updated_at": "2026-10-20T00:00:00Z",
        "size": 10 * n,
        "files": 1,
        "mergeable": "MERGEABLE",
        "merge_state": "BLOCKED",
        "review_decision": "REVIEW_REQUIRED",
        "ci_rollup": "SUCCESS",
        "reviews": [],
    }
    base.update(kw)
    if "merge_state" not in kw and base["review_decision"] != "REVIEW_REQUIRED":
        base["merge_state"] = (
            "CLEAN"  # as GitHub reports it: BLOCKED only while something required is missing
        )
    return base


def dev_snap(sha="d" * 40, ci=None, new=(), rocke=()):
    return {
        "sha": sha,
        "old": None,
        "new_commits": list(new),
        "rocke_commits": list(rocke),
        "ci": ci or {},
    }


class TestVerdict(unittest.TestCase):
    def test_first_emoji_on_heading(self):
        body = f"Automated review: Model - q, effort: max\n\n## PR #1 — title — {GREEN} Follow-up: one {YELLOW} nit\n"
        self.assertEqual(snapshot.parse_verdict(body), GREEN)

    def test_orange_and_missing(self):
        self.assertEqual(
            snapshot.parse_verdict(f"## PR #2 — t — {ORANGE} rebase\n| {GREEN} |"),
            ORANGE,
        )
        self.assertEqual(snapshot.parse_verdict("no heading here"), "?")

    def test_same_commit(self):
        self.assertTrue(snapshot.same_commit("abcdef1234", "abcdef1234567890"))
        self.assertFalse(
            snapshot.same_commit("abcdef", "abcdef12")
        )  # too short to trust
        self.assertFalse(snapshot.same_commit(None, "abc"))


class TestEvents(unittest.TestCase):
    def setUp(self):
        self.st = empty_state()
        self.gone = mock.Mock(
            return_value={
                "state": "merged",
                "merged_at": "t",
                "merge_sha": "m",
                "by": "x",
            }
        )
        events.diff_and_apply(
            self.st,
            {"prs": [pr(1), pr(2, draft=True)], "develop": dev_snap()},
            self.gone,
            ts="T0",
        )

    def kinds(self, evs):
        return [(e["kind"], e.get("pr")) for e in evs]

    def test_push_ci_decision_review(self):
        snap = {
            "prs": [
                pr(
                    1,
                    head="b" * 40,
                    review_decision="APPROVED",
                    reviews=[
                        {
                            "login": "u",
                            "state": "APPROVED",
                            "at": "t",
                            "commit": "b" * 40,
                        }
                    ],
                ),
                pr(2, draft=True),
            ],
            "develop": dev_snap(),
        }
        evs = events.diff_and_apply(self.st, snap, self.gone, ts="T1")
        self.assertEqual(
            self.kinds(evs), [("PUSH", 1), ("DECISION", 1), ("HUMAN_REVIEW", 1)]
        )
        self.assertEqual(self.st["prs"]["1"]["head_seen_at"], "T1")
        snap["prs"][0]["ci_rollup"] = "FAILURE"
        self.assertEqual(
            self.kinds(events.diff_and_apply(self.st, snap, self.gone, ts="T2")),
            [("CI", 1)],
        )

    def test_ready_and_gone(self):
        evs = events.diff_and_apply(
            self.st, {"prs": [pr(2)], "develop": dev_snap()}, self.gone, ts="T1"
        )
        self.assertEqual(self.kinds(evs), [("READY", 2), ("MERGED", 1)])
        self.assertEqual(self.st["prs"]["1"]["status"], "merged")
        # A merged PR does not produce more events.
        self.assertEqual(
            events.diff_and_apply(
                self.st, {"prs": [pr(2)], "develop": dev_snap()}, self.gone
            ),
            [],
        )

    def test_a_pr_back_after_it_left_keeps_our_last_review(self):
        """A PR closed, or out of scope, that comes back is new to the open list, and keeps our last review of it: a
        review kept unposted is in its record alone, so the next review of it is a follow-up, not a first round.
        """
        kept = {"kind": "unposted", "head": "a" * 40, "count": 1, "file": "x.md"}
        self.st["prs"]["1"]["ours"] = kept
        for status in ("closed", "out-of-scope"):
            with self.subTest(status=status):
                self.st["prs"]["1"]["status"] = status
                evs = events.diff_and_apply(
                    self.st,
                    {"prs": [pr(1), pr(2, draft=True)], "develop": dev_snap()},
                    self.gone,
                    ts="T1",
                )
                self.assertEqual(self.kinds(evs), [("NEW_PR", 1)])
                self.assertEqual(self.st["prs"]["1"]["status"], "open")
                self.assertEqual(self.st["prs"]["1"]["ours"], kept)
        self.assertIsNone(self.st["prs"]["2"]["ours"])

    def test_conflict_is_reported_once_through_unknown(self):
        """A PR going CONFLICTING -> UNKNOWN -> CONFLICTING raises CONFLICT once, not each time."""
        seq = [
            "CONFLICTING",
            "UNKNOWN",
            "CONFLICTING",
            "UNKNOWN",
            "UNKNOWN",
            "CONFLICTING",
            "MERGEABLE",
            "CONFLICTING",
        ]
        got = []
        for i, m in enumerate(seq):
            snap = {
                "prs": [pr(1, mergeable=m), pr(2, draft=True)],
                "develop": dev_snap(),
            }
            got.append(
                self.kinds(
                    events.diff_and_apply(self.st, snap, self.gone, ts=f"T{i + 1}")
                )
            )
        self.assertEqual(
            got, [[("CONFLICT", 1)], [], [], [], [], [], [], [("CONFLICT", 1)]]
        )

    def test_a_red_newer_shas_already_cleared_is_history(self):
        """A re-run attempt that turns red after two newer SHAs passed its failed rocKE job is history: it opens no
        incident (CRITICAL, freeze) to close in the same tick. A newer red keeps it a break.
        """
        job = "Windows::release / Test gfx110X-all / Test hipdnn-integration-tests (shard 1/1)"
        red = {
            "complete": True,
            "runs": {
                "TheRock": {
                    "status": "completed",
                    "conclusion": "failure",
                    "failed_jobs": [job],
                    "rocke_failed": [job],
                }
            },
        }
        green = {
            "complete": True,
            "runs": {
                "TheRock": {
                    "status": "completed",
                    "conclusion": "success",
                    "rocke_passed": [job],
                }
            },
        }
        skipped = {
            "complete": True,
            "runs": {
                "TheRock": {
                    "status": "completed",
                    "conclusion": "success",
                    "rocke_passed": [],
                }
            },
        }
        a, b, c = "a" * 40, "b" * 40, "c" * 40  # newest first: c, b, a
        self.st["develop"]["ci"] = {c: green, b: green, a: {}}
        snap = {
            "prs": [pr(1), pr(2, draft=True)],
            "develop": dev_snap(c, ci={c: green, b: green, a: red}),
        }
        evs = events.diff_and_apply(self.st, snap, self.gone, ts="T1")
        cleared = [e for e in evs if e["kind"].startswith("DEVELOP_RED")]
        self.assertEqual(
            [(e["kind"], e["sha"], e["green"]) for e in cleared],
            [("DEVELOP_RED_CLEARED", "aaaaaaaaaa", "cccccccccc")],
        )
        self.assertEqual(
            snapshot.red_cleared_by({c: skipped, b: green, a: red}, a), b
        )  # c skipped the job
        self.assertIsNone(
            snapshot.red_cleared_by({c: red, b: green, a: red}, a)
        )  # red again at c
        self.assertIsNone(
            snapshot.red_cleared_by({c: skipped, a: red}, a)
        )  # no newer SHA ran the job

    def test_develop_moved_red_green(self):
        red = {
            "runs": {
                "TheRock": {
                    "status": "completed",
                    "conclusion": "failure",
                    "rocke_failed": ["rocke job"],
                }
            }
        }
        green = {"runs": {"TheRock": {"status": "completed", "conclusion": "success"}}}
        snap = {
            "prs": [pr(1), pr(2, draft=True)],
            "develop": dev_snap(
                "e" * 40,
                ci={"e" * 40: red, "d" * 40: green},
                new=[("e" * 40, "fix (#9)")],
                rocke=[("e" * 40, "fix (#9)")],
            ),
        }
        evs = events.diff_and_apply(self.st, snap, self.gone, ts="T1")
        kinds = [e["kind"] for e in evs]
        self.assertEqual(kinds, ["DEVELOP_MOVED", "DEVELOP_RED", "DEVELOP_GREEN"])
        self.assertEqual(evs[0]["rocke_prs"], [9])
        self.assertEqual(self.st["develop"]["last_green"]["sha"], "d" * 40)
        self.assertEqual(self.st["develop"]["pending_sanity"], ["e" * 40])
        # The red SHA turns green on a re-run: last_green moves forward.
        snap["develop"]["ci"]["e" * 40] = green
        evs = events.diff_and_apply(self.st, snap, self.gone, ts="T2")
        self.assertEqual([e["kind"] for e in evs], ["DEVELOP_GREEN"])
        self.assertEqual(self.st["develop"]["last_green"]["sha"], "e" * 40)


class TestGates(unittest.TestCase):
    def rec(self, n=1, **kw):
        r = pr(n, **kw)
        r.update(
            status="open", head_seen_at="2026-10-20T00:00:00Z", ours=kw.get("ours")
        )
        return r

    def test_review_gate_without_reviews_reads_the_human_approvals(self):
        mq = dict(
            CFG, features={"review_automation": {"enabled": False}}
        )  # the merge queue alone
        h, old = "a" * 40, "c" * 40

        def appr(login="lead", commit=h, state="APPROVED"):
            return {
                "login": login,
                "state": state,
                "at": "2026-10-20T01:00:00Z",
                "commit": commit,
            }

        def g(**kw):
            return queue.gate_review(mq, self.rec(**kw))

        self.assertEqual(g()[0], queue.FAIL)  # no approval by a person
        self.assertEqual(
            g(reviews=[appr()], review_decision="APPROVED"),
            (queue.PASS, "approved at head by lead"),
        )
        # The author's own approval and a bot's do not count.
        self.assertEqual(
            g(reviews=[appr("dev"), appr("ci[bot]")], review_decision="APPROVED")[0],
            queue.FAIL,
        )
        # A standing change request fails; required reviews not met yet are pending.
        self.assertEqual(
            g(
                reviews=[appr(), appr("other", state="CHANGES_REQUESTED")],
                review_decision="CHANGES_REQUESTED",
            )[0],
            queue.FAIL,
        )
        self.assertEqual(
            g(reviews=[appr()], review_decision="REVIEW_REQUIRED")[0], queue.PENDING
        )
        # An approval at an older head: not at head, unless the head was carried from it.
        stale = g(reviews=[appr(commit=old)], review_decision="APPROVED")
        self.assertEqual(stale[0], queue.FAIL)
        self.assertIn("changed since", stale[1])
        carried = {"head": h, "of": [old], "checked": [old]}
        self.assertEqual(
            g(reviews=[appr(commit=old)], review_decision="APPROVED", carried=carried)[
                0
            ],
            queue.PASS,
        )
        # Under the account, a review comment at the head (another deployment's, or one by hand) neither passes nor
        # fails G1: a person's approval passes over its 🟠, and its 🟢 is no approval.
        self.assertEqual(
            g(
                ours={"verdict": ORANGE, "head": h},
                reviews=[appr()],
                review_decision="APPROVED",
            ),
            (queue.PASS, "approved at head by lead"),
        )
        self.assertEqual(
            g(ours={"verdict": GREEN, "head": h}),
            (queue.FAIL, "no approval by a person yet"),
        )
        # The account's change request or approval counts like anyone's.
        self.assertEqual(
            g(
                reviews=[appr(), appr("me", state="CHANGES_REQUESTED")],
                review_decision="CHANGES_REQUESTED",
            ),
            (queue.FAIL, "changes requested by me"),
        )
        self.assertEqual(
            g(reviews=[appr("me")], review_decision="APPROVED"),
            (queue.PASS, "approved at head by me"),
        )
        # Where the monitor reviews, G1 is our review alone.
        self.assertEqual(
            queue.gate_review(
                CFG, self.rec(reviews=[appr()], review_decision="APPROVED")
            )[0],
            queue.FAIL,
        )

    def test_carry_heads_checks_each_moved_head_once(self):
        mq = dict(
            CFG, features={"review_automation": {"enabled": False}}
        )  # the merge queue alone
        h, old = "a" * 40, "c" * 40
        rec = self.rec(
            reviews=[{"login": "lead", "state": "APPROVED", "at": "x", "commit": old}],
            review_decision="APPROVED",
            ours={"verdict": GREEN, "head": "d" * 40},
        )
        st = {"prs": {"1": rec}}
        calls = []

        def classify(cfg, r, o):
            calls.append(o)
            return {"kind": "rebase-only"}

        with mock.patch("rkmon_lib.review.classify", side_effect=classify):
            lines = queue.carry_heads(mq, st)
            # The approval's commit alone is checked: a review under the account is no approval here, so it is
            # not carried.
            self.assertEqual(calls, [old])
            self.assertEqual(
                lines,
                [
                    f"CARRIED #1: rebase-only from {old[:8]} to {h[:8]}: the approval there stands"
                ],
            )
            self.assertEqual(queue.gate_review(mq, rec)[0], queue.PASS)
            self.assertNotIn("equivalent_heads", rec["ours"])
            self.assertEqual(queue.carry_heads(mq, st), [])  # checked once per head
            self.assertEqual(calls, [old])
        # A deployment that reviews carries through its own review step, not here.
        self.assertEqual(queue.carry_heads(CFG, st), [])

    def test_review_gate(self):
        h = "a" * 40
        self.assertEqual(queue.gate_review(CFG, self.rec())[0], queue.FAIL)
        ok = {"verdict": GREEN, "head": h}
        self.assertEqual(queue.gate_review(CFG, self.rec(ours=ok))[0], queue.PASS)
        self.assertIn(
            "changed",
            queue.gate_review(CFG, self.rec(ours={**ok, "head": "b" * 40}))[1],
        )
        self.assertEqual(
            queue.gate_review(CFG, self.rec(ours={**ok, "verdict": ORANGE}))[0],
            queue.FAIL,
        )
        self.assertEqual(
            queue.gate_review(CFG, self.rec(ours={**ok, "head": None}))[0],
            queue.PENDING,
        )

    def test_humans_gate_flags_our_account(self):
        r = self.rec(
            reviews=[
                {"login": "me", "state": "CHANGES_REQUESTED", "at": "t", "commit": "c"}
            ],
            review_decision="CHANGES_REQUESTED",
        )
        status, note = queue.gate_humans(CFG, r)
        self.assertEqual(status, queue.FAIL)
        self.assertIn("only you can clear it", note)

    def test_humans_gate_codeowner_areas(self):
        cfg = dict(
            CFG,
            codeowner_teams={
                "platform": ["tp"],
                "library": ["tl"],
                "root": ["tp", "tl"],
            },
        )
        owners = {"tp": ["p1"], "tl": ["l1"]}
        r = self.rec(
            review_decision=None,
            reviews=[{"login": "p1", "state": "APPROVED", "at": "t", "commit": "c"}],
        )
        self.assertEqual(queue.gate_humans(cfg, r, owners, ["platform"])[0], queue.PASS)
        status, note = queue.gate_humans(cfg, r, owners, ["library", "platform"])
        self.assertEqual(status, queue.PENDING)
        self.assertIn("library (tl)", note)
        self.assertEqual(queue.gate_humans(cfg, r, owners, ["root"])[0], queue.PASS)
        self.assertEqual(
            queue.areas_of(
                ["rk/platform/a.py", "rk/library/b.py", "rk/README.md", "x/y"], "rk"
            ),
            ["library", "platform", "root"],
        )

    def test_deps_gate(self):
        st = empty_state()
        self.assertEqual(
            queue.gate_deps(CFG, st, self.rec(), {})[0], queue.PASS
        )  # ABI order is G9 now
        stacked = self.rec(base="b7")
        st["prs"] = {}
        self.assertIn("stacked on #7", queue.gate_deps(CFG, st, stacked, {"b7": 7})[1])
        st["frozen"] = {"reason": "develop red"}
        self.assertEqual(queue.gate_deps(CFG, st, self.rec(n=5), {})[0], queue.FAIL)


class TestQueueOrder(unittest.TestCase):
    def test_least_disruptive_first(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        good = lambda h: {"verdict": GREEN, "head": h}
        for n, size in ((4, 500), (5, 100), (6, 50)):
            h = str(n) * 40
            r = pr(n, head=h, size=size, review_decision="APPROVED")
            r.update(
                status="open",
                head_seen_at="2026-10-20T00:00:00Z",
                ours=good(h),
                gates=None,
            )
            st["prs"][str(n)] = r
        files = {
            "refs/remotes/origin/pr4": ["rk/golden.json"],
            "refs/remotes/origin/pr5": ["rk/golden.json"],
            "refs/remotes/origin/pr6": ["rk/other.py"],
        }

        def no_claims(cfg, rec, ref, develop_ref=None):
            rec["abi_cache"] = {
                "head": rec["head"],
                "lanes": {},
                "families": [],
                "notes": [],
            }
            return rec["abi_cache"]

        with mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, rec: f"refs/remotes/origin/pr{rec['n']}"
        ), mock.patch.object(
            gitops, "changed_files", lambda repo, base, ref: files[ref]
        ), mock.patch.object(
            gitops, "merge_conflicts", lambda repo, a, b: []
        ), mock.patch.object(
            queue.abi, "develop_counters", lambda cfg: {"binary": 3, "recipe": 2}
        ), mock.patch.object(
            queue.abi, "pr_claims", no_claims
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        ready = [(r["n"], r["disruption"]) for r in q if r.get("position")]
        # #6 shares nothing (d=0); #5 and #4 share the golden (d=1 each): #5 first because it is smaller.
        self.assertEqual(ready, [(6, 0), (5, 1), (4, 1)])
        # No textual conflicts and no ABI lanes: all three form one batch, led by the least disruptive.
        self.assertEqual(st["batches"][0]["members"], [6, 5, 4])
        self.assertTrue(
            st["batches"][0]["check_prefixes"]
        )  # #5 and #4 touch the representative golden
        p = plan.make_plan(CFG, st, [], q)
        self.assertEqual((p[0]["kind"], p[0]["pr"]), ("BATCH_GATE", 6))


class TestStoreAndGuards(unittest.TestCase):
    def test_atomic_state_and_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            s = Store(pathlib.Path(d))
            st = empty_state()
            st["enabled"] = True
            s.save(st)
            self.assertTrue(s.load()["enabled"])
            self.assertFalse(s.ledger_has("post", 1, "h"))
            s.ledger_add("post", 1, "h", "url")
            self.assertTrue(s.ledger_has("post", 1, "h"))
            self.assertFalse(s.ledger_has("post", 1, "h2"))

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            s = Store(pathlib.Path(d))
            with s.locked():
                with self.assertRaises(BlockingIOError):
                    with Store(pathlib.Path(d)).locked():
                        pass

    def test_gh_refuses_writes(self):
        with self.assertRaises(gh.GhError):
            gh.graphql("mutation { addComment(input: {}) { clientMutationId } }")
        with mock.patch.object(gh, "_run", return_value="{}") as run:
            gh.rest("repos/ROCm/r/pulls/1")
            run.assert_called_once()
        # A write flag smuggled in as an argument is refused before gh runs.
        with mock.patch.object(gh, "_run", return_value="{}") as run:
            with self.assertRaises(gh.GhError):
                gh.rest("-X")
            with self.assertRaises(gh.GhError):
                gh.rest("--method=POST")
            run.assert_not_called()


class TestSearchPages(unittest.TestCase):
    def test_a_page_past_the_time_limit_is_asked_again_smaller(self):
        """GitHub answers the PR search with HTTP 502 (its GraphQL time limit) at 15 and at 5 PRs a page: the page is
        asked again smaller, and the tick does not fail."""
        nums, sizes = list(range(1, 8)), []

        def graphql(query, timeout=120, attempts=3, **v):
            first = int(re.search(r"first: (\d+), after", query).group(1))
            sizes.append(first)
            if first > 3:
                raise gh.GhError("gh api graphql: gh: HTTP 502")
            i = int(v.get("cursor") or 0)
            page = nums[i : i + first]
            return {
                "search": {
                    "issueCount": len(nums),
                    "nodes": [{"number": n} for n in page],
                    "pageInfo": {
                        "hasNextPage": i + first < len(nums),
                        "endCursor": str(i + first),
                    },
                }
            }

        with mock.patch.object(gh, "graphql", side_effect=graphql):
            got = [n["number"] for d in snapshot.search_pages("q") for n in d["nodes"]]
        self.assertEqual(got, nums)
        self.assertEqual(
            sizes, [15, 7, 3, 3, 3]
        )  # the smaller size holds for the rest of the walk
        # A JSON 4xx is GitHub's answer: no smaller retry.
        with mock.patch.object(
            gh,
            "graphql",
            side_effect=gh.GhError("gh api graphql: gh: Bad credentials (HTTP 401)"),
        ) as g, self.assertRaises(gh.GhError):
            list(snapshot.search_pages("q"))
        self.assertEqual(g.call_count, 1)
        # A DNS failure is asked again (lessons L-505), and does not fail the tick.
        fails = [
            gh.GhError(
                "gh api graphql: error connecting to api.github.com\ncheck your internet connection"
            )
        ]

        def flaky(query, timeout=120, attempts=3, **v):
            if fails:
                raise fails.pop()
            return graphql(query, timeout, attempts, **v)

        with mock.patch.object(gh, "graphql", side_effect=flaky):
            got = [
                n["number"]
                for d in snapshot.search_pages("q", first=3)
                for n in d["nodes"]
            ]
        self.assertEqual(got, nums)


def pr_node(n: int, updated: str, **over) -> dict:
    """One PR as the tick's search returns it."""
    node = {
        "number": n,
        "title": f"PR {n}",
        "isDraft": False,
        "createdAt": "2026-11-01T00:00:00Z",
        "updatedAt": updated,
        "additions": 1,
        "deletions": 1,
        "changedFiles": 1,
        "author": {"login": "dev-a"},
        "baseRefName": "develop",
        "headRefName": "b",
        "headRefOid": "a" * 40,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": None,
        "autoMergeRequest": None,
        "closingIssuesReferences": {"nodes": []},
        "labels": {"nodes": []},
        "commits": {"nodes": [{"commit": {"oid": "a" * 40, "committedDate": updated}}]},
        "latestOpinionatedReviews": {"nodes": []},
    }
    node.update(over)
    return node


class TestClosingReferences(unittest.TestCase):
    """A PR closes the develop issues among its closing references that are in the develop issues' repository; a
    reference that does not name its repository is in the watched one."""

    def test_only_the_issue_repositorys_issues_count(self):
        refs = {
            "nodes": [
                {"number": 11, "repository": {"nameWithOwner": "ROCm/r"}},
                {"number": 12, "repository": {"nameWithOwner": "ROCm/tracker"}},
                {"number": 13},
                None,
            ]
        }
        queries = []

        def graphql(query, timeout=120, attempts=3, **v):
            queries.append(query)
            node = pr_node(
                7,
                "2026-11-10T00:00:00Z",
                closingIssuesReferences=refs,
                reviewRequests={"nodes": []},
            )
            page = {"hasNextPage": False, "endCursor": None}
            return {"search": {"issueCount": 1, "nodes": [node], "pageInfo": page}}

        cfg = {"repo": "ROCm/r"}
        elsewhere = dict(cfg, incident_issue={"repo": "rocm/TRACKER"})
        with mock.patch.object(gh, "graphql", side_effect=graphql):
            here = snapshot._list_label(cfg, "q")[0]["closes"]
            there = snapshot._list_label(elsewhere, "q")[0]["closes"]
        self.assertEqual((here, there), ([11, 13], [12]))
        self.assertIn("nodes { number repository { nameWithOwner } }", queries[0])


class TestReadLoginWithoutOrgAccess(unittest.TestCase):
    """A read token on the repository alone cannot read organization data: the PR search's review requests (their
    teams) and the code-owner teams' members."""

    NO_ORG = "gh api graphql: Resource not accessible by personal access token"

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        self.cfg = {"repo": "ROCm/r", "paths": {"cache": str(self.d / "cache")}}

    def test_the_search_drops_the_review_requests_and_reads_them_per_pr(self):
        queries = []

        def graphql(query, timeout=120, attempts=3, **v):
            queries.append(query)
            if "reviewRequests" in query:
                raise gh.GhError(self.NO_ORG)
            return {
                "search": {
                    "issueCount": 1,
                    "nodes": [pr_node(7, "2026-11-10T00:00:00Z")],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }

        rest = mock.Mock(return_value=["dev-b", "plat-team"])
        with mock.patch.object(gh, "graphql", side_effect=graphql), mock.patch.object(
            gh, "rest", rest
        ):
            prs = snapshot._list_label(self.cfg, "q")
            self.assertEqual(prs[0]["review_requests"], ["dev-b", "plat-team"])
            self.assertEqual(len(queries), 2)  # once refused, then without them
            self.assertNotIn("reviewRequests", queries[1])
            self.assertIn("first: 15, after", queries[1])  # no smaller pages for it
            self.assertEqual(
                rest.call_args.args[0], "repos/ROCm/r/pulls/7/requested_reviewers"
            )
            # cached per PR until it is updated
            snapshot._list_label(self.cfg, "q")
            self.assertEqual(rest.call_count, 1)
        cache = self.d / "cache" / snapshot.REVIEW_REQUESTS_CACHE
        self.assertTrue(cache.is_file())
        # an update reads them again; a failed read gives the last list
        with mock.patch.object(
            gh, "rest", side_effect=gh.GhError("gh: HTTP 502")
        ) as failing:
            self.assertEqual(
                snapshot.rest_review_requests(self.cfg, 7, "2026-11-11T00:00:00Z"),
                ["dev-b", "plat-team"],
            )
        failing.assert_called_once()
        # a search that can read them never asks REST
        with mock.patch.object(
            gh,
            "graphql",
            return_value={
                "search": {
                    "issueCount": 1,
                    "nodes": [
                        pr_node(
                            8,
                            "2026-11-10T00:00:00Z",
                            reviewRequests={
                                "nodes": [{"requestedReviewer": {"login": "dev-c"}}]
                            },
                        )
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            },
        ), mock.patch.object(gh, "rest", side_effect=AssertionError("REST")):
            self.assertEqual(
                snapshot._list_label(self.cfg, "q")[0]["review_requests"], ["dev-c"]
            )

    def test_the_code_owner_list_stands_when_it_cannot_refresh(self):
        """The tick reads the code-owner teams through snapshot.refresh_codeowners, and a refused read never fails it:
        a list an earlier read made stands, said in a line, and the next try is a day later; with no list yet, the
        teams are none, one FEATURE_SKIPPED line says so, and a day later the read is tried again, said no more.
        """
        import rkmon

        class Stop(Exception):
            pass

        old = {"at": "2026-11-01T00:00:00Z", "teams": {"plat": ["dev-a"]}}
        st = {"codeowners": dict(old)}
        refused = gh.GhError("gh: Resource not accessible by personal access token")
        with mock.patch.object(
            snapshot, "codeowner_members", side_effect=refused
        ) as members, mock.patch.object(
            rkmon, "refresh_threads", side_effect=Stop
        ), mock.patch(
            "builtins.print"
        ) as said:
            with self.assertRaises(Stop):
                rkmon.do_queue(self.cfg, st)
            self.assertEqual(st["codeowners"]["teams"], old["teams"])
            self.assertIn(
                "the list from 2026-11-01T00:00:00Z stands", said.call_args.args[0]
            )
            # the next try is a day later
            with self.assertRaises(Stop):
                rkmon.do_queue(self.cfg, st)
            self.assertEqual(members.call_count, 1)
            # with no list yet, the tick goes on: the teams are none, said once
            said.reset_mock()
            fresh = {}
            with self.assertRaises(Stop):
                rkmon.do_queue(self.cfg, fresh)
            self.assertEqual(
                [c.args[0].split(": ")[0] for c in said.call_args_list],
                ["FEATURE_SKIPPED code-owner approvals"],
            )
            self.assertEqual(fresh["codeowners"]["teams"], {})
            self.assertTrue(fresh["codeowners"]["refused"])
            # a day later the read is tried again, and its refusal is said no more
            said.reset_mock()
            fresh["codeowners"]["at"] = "2026-11-01T00:00:00Z"
            with self.assertRaises(Stop):
                rkmon.do_queue(self.cfg, fresh)
            self.assertEqual(members.call_count, 3)
            said.assert_not_called()

    def test_a_refused_read_with_no_list_yet_is_said_once_and_the_queue_goes_on(self):
        """snapshot.refresh_codeowners with no code-owner list yet and GitHub refusing the read of the teams' members
        (no Members on the organization): the teams are none, the next try is a day later, and one FEATURE_SKIPPED
        line says what is missing and its fix, once; a read that lists them later takes their place, and a list a read
        made stands when a later read is refused."""
        from rkmon_lib import ghlogins

        cfg = dict(self.cfg, repo="example-org/r", codeowner_teams={"rk/a": ["team-a"]})
        refused = gh.GhError("gh: Not Found (HTTP 404)")
        st = {}
        with mock.patch.object(gh, "rest", side_effect=refused):
            lines = snapshot.refresh_codeowners(cfg, st, "2026-11-01T00:00:00Z")
            self.assertEqual(
                st["codeowners"],
                {"at": "2026-11-01T00:00:00Z", "teams": {}, "refused": True},
            )
            self.assertEqual(len(lines), 1, lines)
            self.assertTrue(
                lines[0].startswith(
                    "FEATURE_SKIPPED code-owner approvals: the read login cannot read example-org's team members "
                    "(gh: Not Found (HTTP 404)): it needs Members: read on example-org; until then the queue knows a "
                    "PR's code-owner approvals only from GitHub's review decision (fix: "
                ),
                lines[0],
            )
            fix = str(ghlogins.read_token_fix("example-org/r"))
            self.assertTrue(lines[0].endswith(f"(fix: {fix})"), lines[0])
            self.assertIn("example-org's team members", fix)
            self.assertIn("login github_read", fix)
            # once: the next refused read says nothing
            self.assertEqual(
                snapshot.refresh_codeowners(cfg, st, "2026-11-02T00:00:00Z"), []
            )
            self.assertEqual(st["codeowners"]["at"], "2026-11-02T00:00:00Z")
        with mock.patch.object(gh, "rest", return_value=["dev-a"]):
            self.assertEqual(
                snapshot.refresh_codeowners(cfg, st, "2026-11-03T00:00:00Z"), []
            )
        self.assertEqual(
            st["codeowners"],
            {"at": "2026-11-03T00:00:00Z", "teams": {"team-a": ["dev-a"]}},
        )
        with mock.patch.object(gh, "rest", side_effect=refused):
            lines = snapshot.refresh_codeowners(cfg, st, "2026-11-04T00:00:00Z")
        self.assertEqual(st["codeowners"]["teams"], {"team-a": ["dev-a"]})
        self.assertEqual(st["codeowners"]["tried"], "2026-11-04T00:00:00Z")
        self.assertEqual(
            lines,
            [
                "(code-owner teams not refreshed, the list from 2026-11-03T00:00:00Z stands: gh: Not Found "
                "(HTTP 404))"
            ],
        )


class TestDevelopCi(unittest.TestCase):
    def test_a_failed_attempt_is_read_from_its_own_jobs(self):
        """A develop SHA fails in a rocKE shard and the auto-retry starts attempt 2 between our two reads: the failed
        attempt is read from its own jobs, not the latest list's (the shard in progress, "red outside rocKE").
        """
        shard = (
            "Linux::release / Test gfx94X-dcgpu / Test hipkernelprovider (shard 1/2)"
        )

        def rest(path, jq=None, paginate=False, timeout=120):
            if "/actions/runs?" in path:
                return [
                    {
                        "id": 7,
                        "name": "TheRock Multi-Arch CI",
                        "status": "completed",
                        "conclusion": "failure",
                        "run_attempt": 1,
                    }
                ]
            return (
                [shard] if "/runs/7/attempts/1/jobs" in path else []
            )  # the latest attempt: re-run, in progress

        cfg = dict(
            CFG,
            develop_watched_workflows=["TheRock Multi-Arch CI"],
            develop_rocke_job_pattern="hipkernelprovider|rocke",
        )
        with mock.patch.object(gh, "rest", side_effect=rest):
            ci = snapshot.develop_ci(cfg, "c" * 40)
        self.assertEqual(ci["runs"]["TheRock Multi-Arch CI"]["rocke_failed"], [shard])
        self.assertEqual(snapshot.ci_verdict(ci), "red")

    def test_a_job_whose_tests_never_ran_is_infrastructure(self):
        """A Windows hipdnn-integration-tests runner lost in setup (Test step
        pending, no log): with the rocfft timeout beside it the SHA is red outside rocKE, and alone it is infra.
        """
        hip = "Windows::release / Test gfx110X-all / Test hipdnn-integration-tests (shard 1/1)"
        fft = "Windows::release / Test gfx110X-all / Test rocfft (shard 1/1)"
        summ = "Multi-Arch CI Summary"
        jobs = [
            {"name": hip, "test": "pending"},
            {"name": fft, "test": "failure"},
            {"name": summ, "test": None},
        ]

        def rest(path, jq=None, paginate=False, timeout=120):
            if "/actions/runs?" in path:
                return [
                    {
                        "id": 7,
                        "name": "TheRock Multi-Arch CI",
                        "status": "completed",
                        "conclusion": "failure",
                        "run_attempt": 2,
                    }
                ]
            return jobs if "/runs/7/attempts/2/jobs" in path else []

        cfg = dict(
            CFG,
            develop_watched_workflows=["TheRock Multi-Arch CI"],
            develop_rocke_job_pattern="hipkernelprovider|hipdnn|rocke",
        )
        with mock.patch.object(gh, "rest", side_effect=rest):
            ci = snapshot.develop_ci(cfg, "c" * 40)
        run = ci["runs"]["TheRock Multi-Arch CI"]
        self.assertEqual((run["rocke_failed"], run["infra_failed"]), ([], [hip]))
        self.assertEqual(snapshot.ci_verdict(ci), "red-other")
        jobs[:] = [{"name": hip, "test": "pending"}, {"name": summ, "test": None}]
        with mock.patch.object(gh, "rest", side_effect=rest):
            ci = snapshot.develop_ci(cfg, "c" * 40)
        self.assertEqual(snapshot.ci_verdict(ci), "infra")
        # A test step that ran and failed, or a job with no Test step, stays a failure.
        jobs[:] = [{"name": hip, "test": "failure"}]
        with mock.patch.object(gh, "rest", side_effect=rest):
            self.assertEqual(
                snapshot.ci_verdict(snapshot.develop_ci(cfg, "c" * 40)), "red"
            )
        jobs[:] = [{"name": hip, "test": None}]
        with mock.patch.object(gh, "rest", side_effect=rest):
            self.assertEqual(
                snapshot.ci_verdict(snapshot.develop_ci(cfg, "c" * 40)), "red"
            )


class TestScope(unittest.TestCase):
    """A PR with a shared label only, rebased onto a develop newer than the mirror's: its merge base with the stale
    develop takes in develop's own commits in the folder, and a PR that changes no file in its label's folder stays out
    of scope. Each shared label counts for its own folder, and the scope cache keeps one answer per PR, head and folder.
    """

    @staticmethod
    def g(repo, *a):
        return subprocess.run(
            ["git", "-C", repo, *a], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(self, files, msg):
        for f, text in files.items():
            p = pathlib.Path(self.origin, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.g(self.origin, "add", "-A")
        self.g(
            self.origin,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            msg,
        )
        return self.g(self.origin, "rev-parse", "HEAD")

    def pr_ref(self, n, files):
        self.g(self.origin, "checkout", "-q", "-b", f"pr{n}", "develop")
        sha = self.commit(files, f"pr {n}")
        self.g(self.origin, "update-ref", f"refs/pull/{n}/head", sha)
        self.g(self.origin, "checkout", "-q", "develop")
        return sha

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.origin, self.mirror = f"{self.d.name}/origin", f"{self.d.name}/mirror"
        self.g(self.d.name, "init", "-q", "-b", "develop", "origin")
        self.base = self.commit({"rk/a.txt": "1", "other/b.txt": "1"}, "base")
        self.g(self.d.name, "init", "-q", "mirror")
        self.g(self.mirror, "remote", "add", "origin", self.origin)
        self.g(
            self.mirror, "fetch", "-q", "origin", "+develop:refs/remotes/origin/develop"
        )  # the stale develop
        self.commit({"rk/a.txt": "2"}, "develop's own rocKE change")
        self.p7 = self.pr_ref(
            7, {"other/b.txt": "2"}
        )  # based on the newer develop; no rocKE file
        self.p9 = self.pr_ref(9, {"rk/c.txt": "1"})
        self.cfg = dict(
            CFG,
            paths={
                "working_dir": "/nonexistent",
                "repo": self.mirror,
                "workers": "/nonexistent",
            },
        )
        self.order = []
        real_pr, real_dev = gitops.fetch_pr, gitops.fetch_develop

        def fetch_pr(repo, n):
            self.order.append(f"pr{n}")
            return real_pr(repo, n)

        def fetch_develop(repo):
            self.order.append("develop")
            return real_dev(repo)

        for name, fn in (("fetch_pr", fetch_pr), ("fetch_develop", fetch_develop)):
            p = mock.patch.object(gitops, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def test_develop_is_fetched_after_the_pr(self):
        cache = {}
        self.assertEqual(
            snapshot.changes(self.cfg, 7, self.p7, ["rk"], cache), {"rk": False}
        )
        self.assertEqual(self.order, ["pr7", "develop"])
        self.assertEqual(cache["7"], {"head": self.p7, "folders": {"rk": False}})
        self.assertEqual(
            snapshot.changes(self.cfg, 9, self.p9, ["rk"], cache), {"rk": True}
        )

    def test_old_true_entries_are_checked_again_and_old_false_ones_stand(self):
        cache = {
            "7": {"head": self.p7, "rocke": True},
            "8": {"head": "8" * 40, "rocke": False},
        }
        self.assertFalse(snapshot.changes(self.cfg, 7, self.p7, ["rk"], cache)["rk"])
        self.assertFalse(
            snapshot.changes(self.cfg, 8, "8" * 40, ["rk"], cache)["rk"]
        )  # a stale develop only adds files
        self.assertFalse(snapshot.changes(self.cfg, 7, self.p7, ["rk"], cache)["rk"])
        self.assertEqual(self.order, ["pr7", "develop"])  # one check, cached since

    def test_an_entry_from_before_the_folders_answers_for_the_repository_s_folder(self):
        """An entry from before the answers by folder holds one, for the repository's folder (config rocke_path): it
        stands there, and a folder it has no answer for is checked, both kept in the entry's form by folder.
        """
        cache = {"9": {"head": self.p9, "rocke": True, "v": snapshot.OLD_V}}
        self.assertEqual(
            snapshot.changes(self.cfg, 9, self.p9, ["rk"], cache), {"rk": True}
        )
        self.assertEqual(self.order, [])
        self.assertEqual(
            snapshot.changes(self.cfg, 9, self.p9, ["other", "rk"], cache),
            {"other": False, "rk": True},
        )
        self.assertEqual(self.order, ["pr9", "develop"])
        self.assertEqual(
            cache["9"], {"head": self.p9, "folders": {"rk": True, "other": False}}
        )

    def test_scope_filter_fetches_the_prs_then_develop_once(self):
        cfg = dict(self.cfg, scope={"labels": ["x"], "path_labels": ["broad"]})
        prs = [
            {"n": 5, "head": "5" * 40, "labels": ["x"]},
            {"n": 7, "head": self.p7, "labels": ["broad"]},
            {"n": 9, "head": self.p9, "labels": ["broad"]},
        ]
        self.assertEqual([p["n"] for p in snapshot.scope_filter(cfg, prs, {})], [5, 9])
        self.assertEqual(self.order, ["pr7", "pr9", "develop"])

    def test_a_shared_label_counts_only_for_its_own_folder(self):
        """Two shared labels with two folders: a PR with one is in scope only when it changes files in that label's
        folder, and with both when it changes files in either; a scope label puts it in scope with no diff. The cache
        keeps one answer per PR, head and folder, and scope_filter fetches each PR to check once.
        """
        cfg = dict(
            self.cfg,
            scope={
                "labels": ["x"],
                "path_labels": {"area: rk": "rk", "area: other": "other"},
            },
        )
        cache = {}
        for n, head, labels, want in (
            (7, self.p7, ["area: rk"], False),  # #7 changes other/ only
            (7, self.p7, ["area: other"], True),
            (9, self.p9, ["area: rk"], True),  # #9 changes rk/ only
            (9, self.p9, ["area: other"], False),
            (9, self.p9, ["area: other", "area: rk"], True),
            (5, "5" * 40, ["x", "area: other"], True),
            (5, "5" * 40, ["unrelated"], False),
        ):
            with self.subTest(n=n, labels=labels):
                pr = {"n": n, "head": head, "labels": labels}
                self.assertEqual(snapshot.in_scope(cfg, pr, cache), want)
        self.assertEqual(
            cache,
            {
                "7": {"head": self.p7, "folders": {"rk": False, "other": True}},
                "9": {"head": self.p9, "folders": {"rk": True, "other": False}},
            },
        )
        self.order.clear()
        prs = [
            {"n": 7, "head": self.p7, "labels": ["area: rk", "area: other"]},
            {"n": 9, "head": self.p9, "labels": ["area: other"]},
        ]
        self.assertEqual([p["n"] for p in snapshot.scope_filter(cfg, prs, {})], [7])
        self.assertEqual(self.order, ["pr7", "pr9", "develop"])

    def test_a_pr_that_leaves_scope_by_its_diff_says_so(self):
        import rkmon

        cfg = dict(self.cfg, scope={"labels": ["x"], "path_labels": ["broad"]})
        p = {
            "state": "open",
            "merged_at": None,
            "merge_commit_sha": None,
            "by": None,
            "labels": ["broad"],
        }
        said = "shared label only, and its diff changes no file in rk"
        with mock.patch.object(gh, "rest", return_value=p):
            for cache in (
                {"7": {"head": "h", "folders": {"rk": False}}},
                {"7": {"head": "h", "rocke": False, "v": 2}},  # from before the folders
            ):
                self.assertEqual(rkmon.resolve_gone(cfg, cache)(7)["note"], said)
            self.assertIn("label removed", rkmon.resolve_gone(cfg, {})(7)["note"])
            # A second shared label whose folder has no answer yet: not known to leave by its diff.
            two = dict(
                cfg,
                scope={
                    "labels": ["x"],
                    "path_labels": {"broad": "rk", "more": "other"},
                },
            )
            p["labels"] = ["broad", "more"]
            cache = {"7": {"head": "h", "folders": {"rk": False}}}
            self.assertIn("label removed", rkmon.resolve_gone(two, cache)(7)["note"])
            cache["7"]["folders"]["other"] = False
            self.assertEqual(
                rkmon.resolve_gone(two, cache)(7)["note"],
                "shared label only, and its diff changes no file in other or rk",
            )


class TestTheScopeRule(unittest.TestCase):
    """The labels in scope have one home, config.scope: the labels, each shared label with its folder, and the one rule
    that decides by them (config.Scope.takes); every reader asks it."""

    def test_the_labels_and_each_shared_label_s_folder(self):
        sc = config.scope(
            {
                "scope": {"labels": ["a"], "path_labels": {"n1": "p/x", "n2": "p/y/"}},
                "rocke_path": "rk",
            }
        )
        self.assertEqual(sc, config.Scope(("a",), {"n1": "p/x", "n2": "p/y"}))
        self.assertEqual(sc.searched(), ["a", "n1", "n2"])
        self.assertEqual(sc.shared_text(), "n1 (p/x), n2 (p/y)")
        # config.json's list of shared labels: each counts for the repository's folder
        self.assertEqual(
            config.scope(
                {
                    "scope": {"labels": ["a"], "path_labels": ["n1"]},
                    "rocke_path": "/rk/",
                }
            ).shared,
            {"n1": "rk"},
        )
        self.assertEqual(
            config.scope({"scope_label": "old", "rocke_path": "rk"}),
            config.Scope(("old",), {}),
        )
        self.assertEqual(config.scope({}), config.Scope((), {}))
        self.assertEqual(config.Scope((), {}).shared_text(), "")

    def test_the_rule_asks_the_folders_in_order_until_one_answers_yes(self):
        sc = config.Scope(("a",), {"n1": "p/x", "n2": "p/y", "n3": "p/x"})
        self.assertIsNone(sc.folders(["n1", "a"]))
        self.assertEqual(sc.folders(["n3", "n2", "n1", "z"]), ["p/x", "p/y"])
        self.assertEqual(sc.folders(["z"]), [])
        asked = []

        def changes(folder):
            asked.append(folder)
            return folder == "p/x"

        self.assertTrue(sc.takes(["n2", "n1"], changes))
        self.assertEqual(asked, ["p/x"])
        self.assertFalse(sc.takes(["n2"], changes))
        self.assertTrue(sc.takes(["a"], None))  # a scope label: no folder asked
        self.assertFalse(sc.takes(["z"], None))

    def test_the_summary_shows_each_shared_label_with_its_folder(self):
        from rkmon_lib import summary

        cfg = config.load_defaults()
        cfg.pop("_doc", None)
        cfg.update(
            repo="acme/widgets",
            rocke_path="a/b",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"], "path_labels": {"needs-ci": "a/c"}},
        )
        cfg["paths"] = dict(cfg["paths"], working_dir="/w", repo="/w/repo")
        with mock.patch("rkmon_lib.doctor.rocm_image", return_value={}):
            text = "\n".join(summary.configuration(config.normalize(cfg)))
        self.assertIn("x; shared labels: needs-ci (a/c)", text)

    def test_every_reader_asks_the_one_function(self):
        """No module reads the shared labels from the configuration itself (`path_labels`): each asks config.scope,
        the readers named here among them. Only these name the key: config.py, the one function's home;
        from_layout1.py and tools/translate_production.py, migration code that reads earlier files by the names those
        files use."""
        from rkmon_lib import package_files

        root = pathlib.Path(__file__).resolve().parent.parent
        allowed = {
            "rkmon_lib/config.py",
            "rkmon_lib/from_layout1.py",
            "tools/translate_production.py",
        }
        named = set()
        for path in package_files.tracked(root):
            rel = (path.relative_to(root) if path.is_absolute() else path).as_posix()
            if rel.endswith(".py") and not rel.startswith("tests/"):
                if "path_labels" in (root / rel).read_text(encoding="utf-8"):
                    named.add(rel)
        self.assertEqual(named - allowed, set())
        for rel in (
            "rkmon.py",
            "rkmon_lib/snapshot.py",
            "rkmon_lib/stats.py",
            "rkmon_lib/breakage.py",
            "rkmon_lib/scoreboard.py",
            "rkmon_lib/summary.py",
        ):
            with self.subTest(reader=rel):
                self.assertIn("config.scope(", (root / rel).read_text(encoding="utf-8"))


class TestFacts(unittest.TestCase):
    """`rk facts N` (lessons L-536): a brief's facts come from GitHub and the state, never from memory."""

    HEAD, OLD, OLDER = "c" * 40, "b" * 40, "a" * 40
    ASK = "These two are still open before this can merge."

    def gh_fakes(self, threads_fail=False):
        from rkmon_lib import post

        long_body = (
            "@dev-a there are a couple of other unaddressed items:\n\n"
            + "\n".join(f"- item {i}: " + "x" * 90 for i in range(20))
            + "\n\n"
            + self.ASK
        )
        rest = {
            "repos/ROCm/r/pulls/7": {
                "sha": self.HEAD,
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "user": "dev-a",
                "base": "develop",
                "updated_at": "2026-11-02T10:19:53Z",
                "merged_at": None,
                "title": "perf: quad_perm",
            },
            "repos/ROCm/r/pulls/7/commits?per_page=100": [
                {
                    "sha": f"{i:02d}" * 20,
                    "at": f"2026-10-{20 + i:02d}T00:00:00Z",
                    "subject": f"commit {i}",
                }
                for i in range(8)
            ],
            "repos/ROCm/r/pulls/7/reviews?per_page=100": [
                {
                    "login": "dev-x",
                    "state": "CHANGES_REQUESTED",
                    "at": "2026-10-16T04:53:16Z",
                    "commit": self.OLDER,
                },
                {
                    "login": "me",
                    "state": "COMMENTED",
                    "at": "2026-10-16T17:13:56Z",
                    "commit": self.OLDER,
                },
                {
                    "login": "dev-x",
                    "state": "APPROVED",
                    "at": "2026-11-01T09:00:00Z",
                    "commit": self.HEAD,
                },
            ],
            # What the jq leaves: the account's comments only. Two of ours are automated, two hand-written; the
            # queue notice's line 1 names this deployment, as post.notice_line writes it.
            "repos/ROCm/r/issues/7/comments?per_page=100": [
                {
                    "at": "2026-10-21T15:23:33Z",
                    "url": "u1",
                    "body": "Automated review: Model - q, effort: max\n\n## PR #7",
                },
                {"at": "2026-10-30T14:41:04Z", "url": "u2", "body": long_body},
                {
                    "at": "2026-10-30T18:23:12Z",
                    "url": "u3",
                    "body": "@dev-a I rebased this branch onto develop.",
                },
                {
                    "at": "2026-10-29T04:06:43Z",
                    "url": "u4",
                    "body": "\n" + post.notice_line(CFG, "queue"),
                },
            ],
            # The account's review-thread replies ("will follow up" is one, and the brief keeps it).
            "repos/ROCm/r/pulls/7/comments?per_page=100": [
                {
                    "at": "2026-10-31T12:00:00Z",
                    "url": "u10",
                    "body": "will follow up",
                    "path": "rk/c.py",
                    "line": 12,
                }
            ],
        }
        issue = {
            "n": 10707,
            "title": "[develop] rocke native engine: 23 cases fail",
            "url": "x",
            "author": "me",
            "labels": [guard_labels.STABILITY, "rocke: platform"],
            "assignees": ["dev2"],
            "created": "2026-10-31T10:00:00Z",
            "body": "",
        }

        def fake_rest(path, jq=None, paginate=False, timeout=120):
            if path.startswith("repos/ROCm/r/issues?labels="):
                return [issue] if guard_labels.STABILITY in path else []
            if (
                path == "repos/ROCm/r/issues/7/comments?per_page=100"
                and jq
                and "!=" in jq
            ):
                return [
                    {
                        "login": "dev-s",
                        "at": "2026-10-29T10:00:00Z",
                        "url": "u7",
                        "body": "before our last review",
                    },
                    {
                        "login": "dev-s",
                        "at": "2026-10-31T16:01:00Z",
                        "url": "u8",
                        "body": "Blocking: x\ny",
                    },
                ]
            return rest[path]

        def fake_graphql(query, timeout=120, attempts=3, **v):
            self.assertEqual(v, {"n": 7})
            if "HEAD_REF_FORCE_PUSHED_EVENT" in query:
                nodes = [
                    {
                        "createdAt": "2026-10-29T16:25:08Z",
                        "actor": {"login": "dev-a"},
                        "beforeCommit": {"oid": self.OLDER},
                        "afterCommit": {"oid": self.OLD},
                    },
                    {
                        "createdAt": "2026-10-31T09:12:00Z",
                        "actor": {"login": "dev-a"},
                        "beforeCommit": {"oid": self.OLD},
                        "afterCommit": {"oid": self.HEAD},
                    },
                ]
                # As GitHub answers: with an item type filter, totalCount is the whole timeline's.
                return {
                    "repository": {
                        "pullRequest": {
                            "timelineItems": {
                                "totalCount": 51,
                                "filteredCount": 2,
                                "nodes": nodes,
                            }
                        }
                    }
                }
            if threads_fail:
                raise gh.GhError("gh api graphql: gh: Bad credentials (HTTP 401)")
            nodes = [
                {
                    "isResolved": True,
                    "path": "rk/a.py",
                    "line": 3,
                    "comments": {
                        "nodes": [
                            {
                                "author": {"login": "dev-x"},
                                "createdAt": "t",
                                "body": "resolved one",
                            }
                        ]
                    },
                },
                {
                    "isResolved": False,
                    "path": "rk/b.py",
                    "line": None,
                    "comments": {
                        "nodes": [
                            {
                                "author": {"login": "dev-x"},
                                "createdAt": "t",
                                "body": "use the new\nAPI",
                            },
                            {"author": None, "createdAt": "t", "body": "done"},
                        ]
                    },
                },
            ]
            return {
                "repository": {
                    "pullRequest": {"reviewThreads": {"totalCount": 2, "nodes": nodes}}
                }
            }

        return fake_rest, fake_graphql, long_body

    def test_a_later_comment_does_not_hide_a_standing_approval(self):
        """A reviewer who APPROVED and only COMMENTED after keeps the approval in `rk facts`, beside the comment."""
        from rkmon_lib import facts

        rows = [
            {
                "login": "dev-y",
                "state": "APPROVED",
                "at": "2026-10-17T10:00:00Z",
                "commit": self.OLDER,
            },
            {
                "login": "dev2",
                "state": "COMMENTED",
                "at": "2026-10-20T10:00:00Z",
                "commit": self.OLDER,
            },
            {
                "login": "dev-y",
                "state": "COMMENTED",
                "at": "2026-10-31T15:25:00Z",
                "commit": self.OLD,
            },
            {
                "login": "dev3",
                "state": "CHANGES_REQUESTED",
                "at": "2026-10-30T10:00:00Z",
                "commit": self.OLD,
            },
            {
                "login": "dev3",
                "state": "APPROVED",
                "at": "2026-11-01T10:00:00Z",
                "commit": self.HEAD,
            },
        ]
        with mock.patch.object(
            gh, "rest", lambda path, jq=None, paginate=False, timeout=120: rows
        ):
            lines = facts._reviews(
                dict(CFG, repo="ROCm/r"), {}, None, 7, {"head": self.HEAD}
            )
        self.assertEqual(
            lines,
            [
                "dev2 COMMENTED 2026-10-20T10:00:00Z aaaaaaaaaa",
                "dev-y APPROVED 2026-10-17T10:00:00Z aaaaaaaaaa; later COMMENTED 2026-10-31T15:25:00Z bbbbbbbbbb",
                "dev3 APPROVED 2026-11-01T10:00:00Z cccccccccc (at head)",
            ],
        )

    def test_an_approval_github_moved_onto_a_later_head_is_not_called_at_head(self):
        """An approval submitted before the head was committed, which GitHub lists at that later head (a develop
        merge), is not called "(at head)"."""
        from rkmon_lib import facts

        rows = [
            {
                "login": "me",
                "state": "APPROVED",
                "at": "2026-11-03T23:28:43Z",
                "commit": self.HEAD,
            },
            {
                "login": "dev2",
                "state": "APPROVED",
                "at": "2026-11-04T13:52:20Z",
                "commit": self.HEAD,
            },
        ]
        with mock.patch.object(
            gh, "rest", lambda path, jq=None, paginate=False, timeout=120: rows
        ):
            lines = facts._reviews(
                dict(CFG, repo="ROCm/r"),
                {},
                None,
                7,
                {"head": self.HEAD, "head_at": "2026-11-04T10:46:22Z"},
            )
        self.assertEqual(
            lines,
            [
                "me APPROVED 2026-11-03T23:28:43Z cccccccccc (GitHub moved it to the head; submitted before the head was "
                "committed)",
                "dev2 APPROVED 2026-11-04T13:52:20Z cccccccccc (at head)",
            ],
        )

    def test_our_last_posts_findings_come_from_its_copy(self):
        """Our last post's findings come from its latest copy (its header, flagged rows and finding rows), never from
        memory, which can mix in another PR's items."""
        from rkmon_lib import facts

        old = (
            "## PR #7 — t (a, "
            + TK
            + "-1) — 🔴 First-round: x\n\n| # | Sev | file:line | Finding |\n|---|---|---|---|\n"
            "| N9 | 🔴 | `a.py:1` | **an old finding**<br>• impact: y |\n"
        )
        new = (
            "Automated review: Model - q, effort: max\n\n## PR #7 — t (a, "
            + TK
            + "-1) — 🟠 Follow-up: fixes hold\n\n"
            "| Prior item | Status |\n|---|---|\n"
            "| Ours (11-01) N1: grouped view | 🟢 FIXED `2c2dc24c`<br>• ran |\n"
            "| Ours (11-01) N2: knob sets | 🟠 OPEN, half fixed `2c2dc24c`<br>• open: CI |\n"
            "| Bots / CI | 🟡 running<br>• 3 ✓ |\n"
            "| Code quality | 🟠 Should-fix<br>• N1 |\n\n| # | Sev | file:line | Finding |\n|---|---|---|---|\n"
            "| N3 | 🟠 | `b.py:2`, `c.cpp:3` | **Per-axis padding variants listed in two gates**<br>• impact: z |\n"
            "| N4 | 🟡 | `d.py:4` | **poly_view zeroes only 256 channels**<br>• fix: w |\n"
        )
        with tempfile.TemporaryDirectory() as d:
            dl = pathlib.Path(d)
            for name, text, t in (
                ("PR_FIRST_ROUND_7.md", old, 1),
                ("PR_FOLLOWUP_7.md", new, 2),
                ("PR_FOLLOWUP_77.md", old, 3),
            ):
                (dl / name).write_text(text, encoding="utf-8")
                os.utime(dl / name, (t, t))
            cfg = {"paths": {"posted_dir": str(dl)}}
            lines = facts._last_findings(cfg, {}, None, 7, {})
            self.assertEqual(facts._last_findings(cfg, {}, None, 8, {}), ["none"])
        self.assertEqual(
            lines,
            [
                "from PR_FOLLOWUP_7.md",
                "## PR #7 — t (a, " + TK + "-1) — 🟠 Follow-up: fixes hold",
                "Ours (11-01) N2: knob sets: 🟠 OPEN, half fixed `2c2dc24c`",
                "Bots / CI: 🟡 running",
                "Code quality: 🟠 Should-fix",
                "N3 🟠 Per-axis padding variants listed in two gates",
                "N4 🟡 poly_view zeroes only 256 channels",
            ],
        )

    def test_our_last_posts_findings_come_from_the_comment_on_github(self):
        """Our last post's findings come from the comment as GitHub holds it, an edit made on GitHub included, ahead of
        its copy (post.our_last_review)."""
        from rkmon_lib import facts, post

        edited = (
            "## PR #7 — t — 🟢 Follow-up: fixes hold\n\n| Bots / CI | 🟡 running |\n"
        )
        ours = {
            "head": "a" * 40,
            "at": "2026-11-02T09:00:00Z",
            "url": "https://github.com/ROCm/r/pull/7#issuecomment-5",
        }
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "PR_FOLLOWUP_7.md").write_text(
                "## PR #7 — t — 🟠 Follow-up: as posted\n", encoding="utf-8"
            )
            cfg = {"paths": {"posted_dir": d}}
            with mock.patch.object(post, "our_post_body", lambda cfg, ours: edited):
                lines = facts._last_findings(
                    cfg, {"prs": {"7": {"ours": ours}}}, None, 7, {}
                )
        self.assertEqual(
            lines,
            [
                "from GitHub",
                "## PR #7 — t — 🟢 Follow-up: fixes hold",
                "Bots / CI: 🟡 running",
            ],
        )

    def test_facts_lists_force_pushes_and_full_user_comments(self):
        """`rk facts` lists every force-push (the author's after our last review included) and the operator's comments
        in full, so no "still open before this can merge" ask is lost to a one-line summary.
        """
        import rkmon
        from rkmon_lib import abi, facts

        cfg = dict(
            CFG,
            operator={"github_login": "me"},
            scope={"labels": ["rocke: platform"], "path_labels": []},
        )
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            cfg["paths"] = {
                "working_dir": d,
                "repo": "/nonexistent",
                "workers": f"{d}/workers",
                "posted_dir": f"{d}/Downloads",
            }
            (pathlib.Path(d) / "workers/pr7/pr").mkdir(parents=True)
            (pathlib.Path(d) / "workers/pr7/pr/meta.md").write_text(
                f"- Head: x @ {self.OLD[:10]}; y\n"
            )
            dl = pathlib.Path(d) / "Downloads"
            dl.mkdir()
            for name in (
                "PR_FIRST_ROUND_7.md",
                "PR_FOLLOWUP_7.md",
                "PR_FOLLOWUP_77.md",
                "PR_UNPOSTED_7.md",
                "PR_DRAFT_7.md",  # an older monitor's name for a review kept unposted
            ):
                (dl / name).write_text("x")
            store.append_jsonl(
                "actions.jsonl",
                [
                    {
                        "ts": "2026-10-21T15:23:33Z",
                        "kind": "post",
                        "pr": 7,
                        "head": self.OLDER,
                        "ref": "u1",
                        "note": "first_round",
                    },
                    {
                        "ts": "2026-10-30T02:07:41Z",
                        "kind": "post",
                        "pr": 7,
                        "head": self.OLD,
                        "ref": "u5",
                        "note": "followup",
                    },
                    {
                        "ts": "2026-10-29T04:06:43Z",
                        "kind": "notice-queue",
                        "pr": 7,
                        "head": "gate-66c398a2e0",
                        "ref": "u4",
                        "note": "",
                    },
                    {
                        "ts": "2026-10-30T03:00:00Z",
                        "kind": "post",
                        "pr": 77,
                        "head": self.OLD,
                        "ref": "u9",
                        "note": "",
                    },
                ],
            )
            st = empty_state()
            st["last_tick"] = "2026-11-02T10:30:00Z"
            st["prs"] = {
                "7": {"n": 7, "head": self.OLD, "status": "open"},
                "8": {
                    "n": 8,
                    "head": "8" * 40,
                    "status": "open",
                    "abi_cache": {"notes": ["public header lines changed"]},
                },
            }
            st["abi_lanes"] = {
                "binary": {
                    "develop": 6,
                    "head": None,
                    "groups": [
                        {
                            "members": [5],
                            "root": 5,
                            "to": 7,
                            "reserve": 7,
                            "ready": False,
                        },
                        {
                            "members": [7, 8],
                            "root": 7,
                            "to": 7,
                            "reserve": 8,
                            "ready": False,
                        },
                    ],
                },
                "recipe": {
                    "develop": 3,
                    "head": None,
                    "groups": [
                        {
                            "members": [9],
                            "root": 9,
                            "to": 4,
                            "reserve": 4,
                            "ready": False,
                        }
                    ],
                },
            }
            store.save(st)
            files = sorted((pathlib.Path(d) / "state").iterdir())
            before = {p.name: p.read_bytes() for p in files if p.is_file()}
            fake_rest, fake_graphql, long_body = self.gh_fakes()
            out = []
            with mock.patch.object(gh, "rest", fake_rest), mock.patch.object(
                gh, "graphql", fake_graphql
            ), mock.patch.object(rkmon, "load_config", lambda: cfg), mock.patch.object(
                rkmon, "Store", lambda: store
            ), mock.patch(
                "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
            ):
                self.assertEqual(rkmon.main(["facts", "7"]), 0)
                text = "\n".join(out)
                lines, failed = facts.render(cfg, st, store, 7)
                with mock.patch.object(
                    gh, "graphql", self.gh_fakes(threads_fail=True)[1]
                ):
                    broken, n_failed = facts.render(cfg, st, store, 7)
            self.assertFalse(failed)
            after = {
                p.name: p.read_bytes()
                for p in sorted((pathlib.Path(d) / "state").iterdir())
                if p.is_file()
            }
            self.assertEqual(
                after, before
            )  # read-only: no state, ledger or journal write
        self.assertIn(
            f"== head and state\n{self.HEAD} open draft=false mergeable=true blocked by dev-a into develop "
            "updated 2026-11-02T10:19:53Z | perf: quad_perm\n(the last tick saw head bbbbbbbbbb: the PR moved "
            "since)",
            text,
        )
        # Every force-push, newest last; the one after our last review (posted 10-30T02:07Z) is marked.
        self.assertIn(
            "== force-pushes (every head the PR had, newest last)\n"
            "2026-10-29T16:25:08Z dev-a aaaaaaaaaa -> bbbbbbbbbb\n"
            "2026-10-31T09:12:00Z dev-a bbbbbbbbbb -> cccccccccc (after our last review)\n== commits",
            text,
        )
        self.assertIn(
            "== commits\n(the last 6 of 8)\n0202020202 2026-10-22T00:00:00Z commit 2\n",
            text,
        )
        self.assertIn(
            "0707070707 2026-10-27T00:00:00Z commit 7\n"
            "== our reviews (the posts in the ledger, then the files in the outputs folder)\n",
            text,
        )
        # Our posts with times (PR 77's are not #7's), then our files in the outputs folder: each post's copy, and each
        # review kept unposted named so, an older monitor's too (PR_FOLLOWUP_77.md is not #7's).
        self.assertIn("2026-10-30T02:07:41Z post at bbbbbbbbbb u5 (followup)", text)
        self.assertIn("2026-10-29T04:06:43Z notice-queue at gate-66c398a2e0 u4", text)
        self.assertNotIn(
            " u9", text
        )  # (with its space: a temporary folder's random name may hold "u9")
        self.assertIn("PR_FIRST_ROUND_7.md", text)
        self.assertIn("PR_FOLLOWUP_7.md", text)
        self.assertNotIn("PR_FOLLOWUP_77.md", text)
        self.assertIn(f"\ncopy: {dl}/PR_FOLLOWUP_7.md (", text)
        self.assertIn(f"\nkept unposted: {dl}/PR_UNPOSTED_7.md (", text)
        self.assertIn(f"\nkept unposted: {dl}/PR_DRAFT_7.md (", text)
        # The ABI lane with develop's number (a grep of `rk abi` for the PR lost the lane line).
        self.assertIn(
            "== rk abi\nbinary: develop 6, next to land: none ready\n  2. #7, #8 claims 7 -> tentative 8 ready=False\n"
            "(lanes as of the last tick, 2026-11-02T10:30:00Z)\n",
            text,
        )
        self.assertNotIn("recipe:", text)
        self.assertEqual(
            abi.lane_lines(st),
            [  # `rk abi` itself prints every lane, as before
                "binary: develop 6, next to land: none ready",
                "  1. #5 claims 7 -> tentative 7 ready=False",
                "  2. #7, #8 claims 7 -> tentative 8 ready=False",
                "recipe: develop 3, next to land: none ready",
                "  1. #9 claims 4 -> tentative 4 ready=False",
                "#8: notes=['public header lines changed'] families=None",
            ],
        )
        self.assertEqual(
            abi.lane_lines(st, 9)[0], "recipe: develop 3, next to land: none ready"
        )
        self.assertIn(
            "== rk breakages\n#10707 silent-break rocKE 2026-10-31 me -> dev2: [develop] rocke native "
            "engine: 23 cases fail\n",
            text,
        )
        # The latest review per reviewer, oldest first; the head's marked.
        self.assertIn(
            "== reviews (latest per reviewer)\nme COMMENTED 2026-10-16T17:13:56Z aaaaaaaaaa\n"
            "dev-x APPROVED 2026-11-01T09:00:00Z cccccccccc (at head)\n",
            text,
        )
        self.assertIn(
            "== unresolved threads\ntotal 2, unresolved 1\n* rk/b.py:0 dev-x: use the new API / ?: "
            "done\n",
            text,
        )
        self.assertNotIn("resolved one", text)
        # The user's hand-written comments, in full; the automated ones skipped.
        self.assertIn(long_body, text)
        self.assertIn(self.ASK, text)
        self.assertIn(
            "--- 2026-10-30T18:23:12Z u3\n@dev-a I rebased this branch onto develop.\n"
            "--- 2026-10-31T12:00:00Z u10 (review thread, rk/c.py:12)\nwill follow up\n",
            text,
        )
        self.assertNotIn("Automated review", text)
        self.assertNotIn("Automated merge-queue", text)
        # Other people's comments since our last review (posted 10-30T02:07Z), in full (a "Blocking" comment is never
        # missed); an older one is left out.
        self.assertIn(
            "== other people's comments since our last review, in full (bots left out)\n"
            "--- dev-s 2026-10-31T16:01:00Z u8\nBlocking: x\ny\n== worker inputs",
            text,
        )
        self.assertNotIn("before our last review", text)
        self.assertIn("== worker inputs\npr/meta.md: head bbbbbbbbbb", text)
        self.assertIn("(not the head)", text)
        self.assertEqual(
            "\n".join(lines).split("\n", 1)[1], text.split("\n", 1)[1]
        )  # the CLI prints render's lines
        # A section that cannot be read says so, and the others still print.
        self.assertEqual(n_failed, 1)
        i = broken.index("== unresolved threads")
        self.assertEqual(
            broken[i + 1], "(not read: gh api graphql: gh: Bad credentials (HTTP 401))"
        )
        self.assertIn(long_body, broken)

    def test_an_automated_post_with_a_quoted_or_bold_line_1_is_no_hand_written_comment(
        self,
    ):
        """`rk facts` reads line 1 of the account's posts as every reader does (reviewer_name.first_line): a review or a
        notice whose line 1 is quoted or bold is automated, and never listed among the operator's comments.
        """
        from rkmon_lib import facts, post, reviewer_name

        line1 = reviewer_name.review_line(CFG, "q", "max")
        bodies = [
            f"> {line1}\n\n## PR #7",
            f"**{line1}**\n\n## PR #7",
            f"\n> _{post.notice_line(CFG, 'queue')}_\n\nqueued",
            "@dev-a I rebased this branch onto develop.",
        ]
        rows = [
            {"at": f"2026-10-2{i}T10:00:00Z", "url": f"u{i}", "body": body}
            for i, body in enumerate(bodies)
        ]

        def fake_rest(path, jq=None, paginate=False, timeout=120):
            return rows if path.endswith("/issues/7/comments?per_page=100") else []

        with mock.patch.object(gh, "rest", fake_rest):
            lines = facts._user_comments(CFG, {}, None, 7, {})
        self.assertEqual(
            lines,
            [
                "--- 2026-10-23T10:00:00Z u3",
                "@dev-a I rebased this branch onto develop.",
            ],
        )

    def test_an_early_review_with_no_line_1_is_no_hand_written_comment(self):
        """An early review of the account's, from before line 1, starts with its header (## PR #N): `rk facts` knows
        it for an automated post as followups does (reviewer_name.automated), and never lists it among the operator's
        comments."""
        from rkmon_lib import facts

        bodies = [
            "## PR #7 — a/b: tidy the loader — 🟢 First-round: ready\n\nLegend: 🟢 fine",
            "> ## PR #7 — a/b: tidy the loader — 🟢 Follow-up: still ready",
            "@dev-a I rebased this branch onto develop.",
        ]
        rows = [
            {"at": f"2026-10-2{i}T10:00:00Z", "url": f"u{i}", "body": body}
            for i, body in enumerate(bodies)
        ]

        def fake_rest(path, jq=None, paginate=False, timeout=120):
            return rows if path.endswith("/issues/7/comments?per_page=100") else []

        with mock.patch.object(gh, "rest", fake_rest):
            lines = facts._user_comments(CFG, {}, None, 7, {})
        self.assertEqual(
            lines,
            [
                "--- 2026-10-22T10:00:00Z u2",
                "@dev-a I rebased this branch onto develop.",
            ],
        )

    def test_what_came_after_our_last_review_counts_from_its_time(self):
        """`rk facts` marks the force-pushes after our last review and lists other people's comments since it, by the
        follow-up gap's rule (plan.last_review_time): a review kept unposted while posting is off counts from its own
        time, which no ledger entry dates, whether the ledger holds no post of ours on the PR or older ones only; a
        posted one from its post, a CI update after it being no review."""
        from rkmon_lib import facts, post

        fake_rest, fake_graphql, _ = self.gh_fakes()
        cfg = dict(CFG, operator={"github_login": "me"})
        shown = [
            s
            for s in facts.SECTIONS
            if s[1] in (facts._force_pushes, facts._others_comments)
        ]

        def facts_with(ours: dict, ledger: list[dict]) -> str:
            with tempfile.TemporaryDirectory() as d:
                store = Store(pathlib.Path(d) / "state")
                store.append_jsonl("actions.jsonl", ledger)
                st = empty_state()
                st["prs"] = {"7": {"n": 7, "head": self.HEAD, "ours": ours}}
                with mock.patch.object(facts, "SECTIONS", shown), mock.patch.object(
                    gh, "rest", fake_rest
                ), mock.patch.object(gh, "graphql", fake_graphql):
                    lines, failed = facts.render(cfg, st, store, 7)
            self.assertEqual(failed, 0)
            return "\n".join(lines[1:])

        def posted(ts: str, head: str, note: str) -> dict:
            return {"ts": ts, "kind": "post", "pr": 7, "head": head, "note": note}

        def kept(at: str) -> dict:
            return {
                "kind": post.UNPOSTED,
                "at": at,
                "head": self.OLD,
                "file": "PR_UNPOSTED_7.md",
            }

        ours_posts = [
            posted("2026-10-21T15:23:33Z", self.OLDER, "first_round"),
            posted("2026-10-30T02:07:41Z", self.OLD, "followup"),
        ]
        pushes = (
            "== force-pushes (every head the PR had, newest last)\n"
            "2026-10-29T16:25:08Z dev-a aaaaaaaaaa -> bbbbbbbbbb\n"
            "2026-10-31T09:12:00Z dev-a bbbbbbbbbb -> cccccccccc"
        )
        others = "\n== other people's comments since our last review, in full (bots left out)\n"
        # Our last review before the second force-push: that push is marked, and only the later comment shows.
        before_push = f"{pushes} (after our last review){others}--- dev-s 2026-10-31T16:01:00Z u8\nBlocking: x\ny"
        with self.subTest("kept, and no post of ours in the ledger"):
            self.assertEqual(facts_with(kept("2026-10-30T12:00:00Z"), []), before_push)
        with self.subTest("kept, after our posts"):
            self.assertEqual(
                facts_with(kept("2026-11-01T00:00:00Z"), ours_posts),
                f"{pushes}{others}none since our last review (2026-11-01T00:00:00Z)",
            )
        with self.subTest("posted, then a CI update"):
            ci = posted("2026-10-31T12:00:00Z", self.HEAD, plan.CI_UPDATE)
            update = {"kind": "comment", "at": ci["ts"], "head": self.HEAD, "url": "u6"}
            self.assertEqual(facts_with(update, ours_posts + [ci]), before_push)


if __name__ == "__main__":
    unittest.main()
