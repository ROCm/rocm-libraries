# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The merge plan: the conflict matrix on develop and its caches, care, the exact
conflict-free waves and the ordering rule, ABI conflicts, rebase advice, auto-merge ahead of the plan, how the queue
uses it (order, batches, holds, ABI lanes), and the board, md and HTML renders."""

import datetime as dt
import html
import itertools
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

from rkmon_lib import mergeplan, queue, report, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

G, O, R = "\U0001f7e2", "\U0001f7e0", "\U0001f534"
CFG = {
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "deployment": {"phase": "queue"},
    "merge_authority": "arm-cleared",
    "paths": {"working_dir": "/nonexistent", "repo": "/nonexistent"},
    "codeowner_teams": {"platform": ["plat"]},
    "hot_spots": {
        "RG": ["rk/golden/rg/"],
        "DG": "rk/dg.json",
        "ABI": "rk/abi.h",
        "CMK": "rk/CMakeLists.txt",
    },
}
PASS, PEND, FAIL = ["pass", ""], ["pending", ""], ["fail", ""]


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
        "files_cache": {
            "head": head(n),
            "hot": [],
            "areas": ["platform"],
            "files": [f"rk/f{n}.py"],
        },
        "ours": {"verdict": G, "head": head(n), "at": now()},
        "threads": {"updated_at": None, "open": []},
    }
    r.update(kw)
    return r


def row(n: int, g1=PASS, g2=PASS, g3=PASS, size=100, **kw) -> dict:
    """A queue row as build_queue has it when the plan runs (gates before G9)."""
    return dict(
        {
            "n": n,
            "size": size,
            "tier": 2,
            "static_ok": g1 == g2 == g3 == PASS,
            "blockers": [],
            "gates": {
                "G1": list(g1),
                "G2": list(g2),
                "G3": list(g3),
                "G4": list(PASS),
                "G8": list(PASS),
            },
        },
        **kw,
    )


def matrix(clean, pairs=None, dev=(), files=None, unknown=()) -> dict:
    """A conflict matrix as mergeplan.matrix returns it, without git."""
    scope = list(clean) + list(dev) + list(unknown)
    return {
        "develop": "d" * 40,
        "scope": scope,
        "heads": {n: head(n) for n in scope},
        "files": files or {n: {f"rk/f{n}.py"} for n in list(clean) + list(dev)},
        "clean": {n: f"t{n}" for n in clean},
        "dev_conflicts": {n: ["rk/x.py"] for n in dev},
        "unknown": list(unknown),
        "pairs": {(min(a, b), max(a, b)): x for (a, b), x in (pairs or {}).items()},
        "blobs": {},
        "merges": 0,
        "secs": 0.0,
        "facts": {},
    }


def state(*ns, **prs) -> dict:
    st = empty_state()
    st["develop"]["sha"] = "d" * 40
    st["prs"] = {str(n): rec(n) for n in ns}
    for k, v in prs.items():
        st["prs"][k.lstrip("_")].update(v)
    return st


def bumps(st: dict, *specs) -> None:
    """ABI claims at the PRs' heads, as abi.pr_claims leaves them: (n, lane, to, hunk[, taken]) each; develop at binary
    6, recipe 2."""
    st["develop"]["abi"] = {"sha": "d" * 40, "binary": 6, "recipe": 2}
    for n, lane, to, hunk, *taken in specs:
        c = st["prs"][str(n)].setdefault(
            "abi_cache", {"head": head(n), "lanes": {}, "families": [], "notes": []}
        )
        c["lanes"][lane] = {
            "from": to - 1,
            "to": to,
            "history_line": True,
            "dropped_history": [],
            "hunk": hunk,
            "taken": list(taken[0]) if taken else [],
        }


class Git:
    """A throwaway repository: develop, PR branches as refs/remotes/origin/prN, as the review clone has them."""

    def __init__(self, d: str):
        self.d = d
        self("init", "-q", "-b", "develop")
        self("config", "user.email", "t@t")
        self("config", "user.name", "t")

    def __call__(self, *a: str) -> str:
        return subprocess.run(
            ["git", "-C", self.d, *a], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(self, files: dict, msg: str) -> str:
        for path, text in files.items():
            p = pathlib.Path(self.d, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self("add", "-A")
        self("commit", "-qm", msg)
        return self("rev-parse", "HEAD")

    def branch(self, name: str, at: str, files: dict) -> str:
        self("checkout", "-q", "-b", name, at)
        sha = self.commit(files, name)
        self("checkout", "-q", "develop")
        return sha


EMIT = "rk/library/tests/parity/fam_emit.c"
DG = (
    json.dumps(
        {
            "ir_canonical": {"famA": {"0": "a0", "1": "a1"}},
            "ll": {"llvm20": {"famA": {"0": "l0"}}},
        },
        indent=2,
    )
    + "\n"
)


class TestMatrix(unittest.TestCase):
    """The matrix on a real repository: one merge with develop per PR, a pair merge only for PRs sharing a file, the
    caches keyed by develop (trees) and by develop's blobs of the shared files (pairs), and the merge budget.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.g = g = Git(d)
        base = g.commit(
            {
                "rk/a.py": "1\n2\n3\n",
                "rk/b.py": "b\n",
                "rk/CMakeLists.txt": "x\ny\nz\n",
                EMIT: "switch(idx)\n{\n    case 0:\n    case 1:\n}\n",
                "rk/dg.json": DG,
            },
            "base",
        )
        heads = {
            1: g.branch(
                "p1",
                base,
                {
                    "rk/a.py": "1a\n2\n3\n",
                    EMIT: "switch(idx)\n{\n    case 0:\n    case 1:\n    case 2:\n}\n",
                },
            ),
            2: g.branch(
                "p2",
                base,
                {"rk/a.py": "1b\n2\n3\n", "rk/dg.json": DG.replace('"a1"', '"a1x"')},
            ),
            3: g.branch(
                "p3",
                base,
                {
                    "rk/b.py": "c\n",
                    "rk/dg.json": DG.replace('"l0"', '"l0x"'),
                    EMIT: "switch(idx)\n{\n    case 0:\n    case 1:\n    case 2: /* 3 */\n}\n",
                },
            ),
            4: g.branch("p4", base, {"rk/CMakeLists.txt": "x\ny\nz4\n"}),
        }
        self.dev = g.commit(
            {"rk/CMakeLists.txt": "x\ny\nzd\n"}, "develop takes the line #4 changes"
        )
        g("update-ref", "refs/remotes/origin/develop", self.dev)
        for n, h in heads.items():
            g("update-ref", f"refs/remotes/origin/pr{n}", h)
        self.cfg = dict(CFG, paths={"repo": d}, merge_plan={"budget_s": 120})
        self.st = empty_state()
        self.st["prs"] = {
            str(n): rec(n, head=h, files_cache=None) for n, h in heads.items()
        }
        self.rows = [row(n, g2=PEND) for n in heads]

    def tearDown(self):
        self.tmp.cleanup()

    def test_conflicts_on_develop_care_and_caches(self):
        M = mergeplan.matrix(self.cfg, self.st, self.rows, self.dev)
        self.assertEqual(sorted(M["clean"]), [1, 2, 3])
        self.assertEqual(M["dev_conflicts"], {4: ["rk/CMakeLists.txt"]})
        self.assertEqual(mergeplan.conflicts(M, 1, 2), ["rk/a.py"])
        self.assertEqual(
            mergeplan.conflicts(M, 1, 3), [EMIT]
        )  # both append a config after case 1
        self.assertEqual(
            mergeplan.conflicts(M, 2, 3), []
        )  # both edit the DG golden, in different lines
        self.assertEqual(
            M["merges"], 4 + 3
        )  # one merge with develop each; the three pairs share a file
        F = mergeplan.facts(self.cfg, self.st, M)
        self.assertEqual(F[1]["idx"], {"fam": [2]})
        self.assertEqual(F[2]["dg"], {"famA"})
        self.assertIn(
            "parity fam idx 2 (the first to land keeps it)",
            mergeplan.care_reasons(F[1], F[3], "rk"),
        )
        self.assertEqual(mergeplan.care_reasons(F[2], F[3], "rk"), ["DG famA"])
        # Warm: nothing to merge again on the same develop.
        self.assertEqual(
            mergeplan.matrix(self.cfg, self.st, self.rows, self.dev)["merges"], 0
        )
        # Develop moves elsewhere: each PR is merged with it again, the pairs come from the cache.
        dev2 = self.g.commit({"rk/elsewhere.py": "e\n"}, "unrelated")
        M2 = mergeplan.matrix(self.cfg, self.st, self.rows, dev2)
        self.assertEqual(
            (M2["merges"], mergeplan.conflicts(M2, 1, 2)), (4, ["rk/a.py"])
        )
        # Develop changes a shared file: that pair is merged again.
        dev3 = self.g.commit({"rk/a.py": "1\n2\n3d\n"}, "a shared file moves")
        M3 = mergeplan.matrix(self.cfg, self.st, self.rows, dev3)
        self.assertEqual(
            (M3["merges"], mergeplan.conflicts(M3, 1, 2)), (4 + 1, ["rk/a.py"])
        )

    def test_the_budget_leaves_a_partial_plan_and_git_failing_everywhere_gives_none(
        self,
    ):
        M = mergeplan.matrix(
            dict(self.cfg, merge_plan={"max_merges": 3}), self.st, self.rows, self.dev
        )
        self.assertEqual((sorted(M["clean"]), M["unknown"]), ([1, 2, 3], [4]))
        self.assertIsNone(
            mergeplan.conflicts(M, 1, 2)
        )  # not checked yet: counts as a conflict
        self.assertEqual(
            mergeplan.conflict_score(M)[1], [(2, ["files"]), (3, ["files"])]
        )
        self.st[mergeplan.CACHE] = {}
        self.assertIsNone(
            mergeplan.matrix(self.cfg, self.st, self.rows, "f" * 40)
        )  # no develop merge at all


class TestSolver(unittest.TestCase):
    def test_exact_against_brute_force(self):
        rng = random.Random(1103)
        for _ in range(60):
            n = rng.randint(1, 11)
            nodes = list(range(1, n + 1))
            adj = {v: set() for v in nodes}
            for a, b in itertools.combinations(nodes, 2):
                if rng.random() < 0.35:
                    adj[a].add(b)
                    adj[b].add(a)
            w = {v: rng.choice((1, 2, 4, 8)) for v in nodes}
            best = max(
                sum(w[v] for v in s)
                for k in range(n + 1)
                for s in itertools.combinations(nodes, k)
                if all(b not in adj[a] for a, b in itertools.combinations(s, 2))
            )
            got = mergeplan.mwis(nodes, w, adj)
            self.assertTrue(
                all(b not in adj[a] for a, b in itertools.combinations(got, 2))
            )
            self.assertEqual(sum(w[v] for v in got), best)
            self.assertEqual(
                got, mergeplan.mwis(list(reversed(nodes)), w, adj)
            )  # the same graph, the same set

    def test_waves_partition_conflict_free_and_hubs_last(self):
        adj = {1: {2, 3, 4, 5, 6}, 2: {1}, 3: {1}, 4: {1}, 5: {1}, 6: {1, 7}, 7: {6}}
        w = {v: 4 for v in adj} | {1: 100}
        ws = mergeplan.waves(sorted(adj), w, adj, hubs={1})
        self.assertEqual(sorted(v for x in ws for v in x), list(range(1, 8)))
        self.assertTrue(
            all(b not in adj[a] for x in ws for a, b in itertools.combinations(x, 2))
        )
        self.assertEqual(ws[-1], [1])  # the heaviest node waits: it is a hub


class TestPlan(unittest.TestCase):
    def test_top_priority_first_its_neighbours_wait_and_the_readiest_go(self):
        """A top priority goes in first; its R1 neighbours wait for it, the others may land."""
        st = state(1, 2, 3, 4)
        st["develop_fixes"] = {"1": ["your priority: blocking other work"]}
        M = matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (3, 4): ["rk/b.py"]})
        rows = [row(1, g3=FAIL), row(2, g2=PEND), row(3, g2=PEND), row(4)]
        p = mergeplan.build(CFG, st, rows, M)
        # Ready #4 has no conflict with #1: wave 1 with it. #2 and #3 wait for the PRs they conflict with: wave 2.
        self.assertEqual(p["waves"], [[1, 4], [2, 3]])
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {1: (0, True), 4: (1, True), 2: (2, False), 3: (3, False)},
        )
        self.assertEqual(
            p["advice"]["2"],
            "wait: #1 (your priority) lands first (conflicts in a.py), then rebase",
        )
        self.assertEqual(
            p["advice"]["3"], "wait: #4 lands first (conflicts in b.py), then rebase"
        )
        # #4 ends wave 1 and makes #3 rebase: its advice says so, as a priority's does.
        self.assertEqual(
            p["advice"]["4"],
            "next: no rebase needed (merges clean with develop and every PR ahead); its "
            "landing makes #3 rebase",
        )
        self.assertEqual(
            p["advice"]["1"],
            "your priority: no rebase needed; its landing makes #2 rebase",
        )
        self.assertEqual(p["nodes"]["1"]["cost"], 4)  # R1 #2 pays the rebase
        self.assertEqual(
            [mergeplan.place(dict(st, merge_plan=p), n) for n in (1, 3)],
            ["wave 1", "wave 2"],
        )

    def test_hubs_count_no_abi_lane_conflict_and_land_last(self):
        """Binary-lane members that all conflict with each other on abi.h are no hubs for it."""
        pairs = {(10, m): ["rk/core.py"] for m in range(11, 16)}
        pairs |= {(20, m): ["rk/abi.h"] for m in range(21, 26)}
        ns = [10] + list(range(11, 16)) + [20] + list(range(21, 26))
        p = mergeplan.build(
            CFG, state(*ns), [row(n, g2=PEND) for n in ns], matrix(ns, pairs)
        )
        self.assertEqual(p["hubs"], [10])
        self.assertEqual(p["waves"][-1], [10])
        self.assertIn("hub (5 conflicts outside the ABI lane)", p["advice"]["10"])

    def test_a_ready_pr_kept_waiting_a_day_lands_anyway(self):
        st = state(30, 31, 32, 33)
        M = matrix(
            [30, 31, 32, 33],
            {(30, 31): ["rk/a.py"], (30, 32): ["rk/b.py"], (30, 33): ["rk/c.py"]},
        )
        rows = [row(30)] + [row(n, g2=PEND) for n in (31, 32, 33)]
        p = mergeplan.build(CFG, st, rows, M)
        self.assertEqual(
            p["waves"], [[31, 32, 33], [30]]
        )  # three R1 (12) outweigh one ready PR (8)
        self.assertEqual(p["held_back"]["30"]["head"], head(30)[:12])
        st["merge_plan"] = dict(
            p, held_back={"30": {"head": head(30)[:12], "since": ago(25 * 60)}}
        )
        p = mergeplan.build(CFG, st, rows, M)
        self.assertEqual(
            (p["waves"], p["why"]["30"]),
            ([[30], [31, 32, 33]], "ready, kept waiting a day"),
        )
        st["merge_plan"] = dict(
            p, held_back={"30": {"head": "0" * 12, "since": ago(25 * 60)}}
        )  # pushed since
        self.assertEqual(
            mergeplan.build(CFG, st, rows, M)["waves"], [[31, 32, 33], [30]]
        )

    def test_a_pinned_group_is_one_node_in_the_users_order(self):
        st = state(40, 41, 42)
        st["groups"] = [[41, 40]]
        M = matrix([40, 41, 42], {(40, 42): ["rk/a.py"]})
        p = mergeplan.build(CFG, st, [row(n, g2=PEND) for n in (40, 41, 42)], M)
        self.assertEqual(
            (p["waves"], p["groups"]), ([[41, 40], [42]], {"41": [41, 40]})
        )
        self.assertTrue(
            p["advice"]["42"].startswith("wait: #40 (pinned group) lands first")
        )

    def test_held_prs_keep_their_place(self):
        p = mergeplan.build(
            CFG,
            state(50, 51),
            [row(50, held={"status": "held"}), row(51)],
            matrix([50, 51], {(50, 51): ["rk/a.py"]}),
        )
        self.assertEqual(p["waves"], [[50], [51]])
        self.assertEqual(
            p["advice"]["50"],
            "held: it keeps its place: no rebase needed; its landing makes #51 rebase",
        )
        self.assertTrue(p["advice"]["51"].startswith("wait: #50 (held) lands first"))

    def test_rebase_advice_for_a_pr_that_conflicts_with_develop(self):
        files = {
            60: {"rk/a.py", "rk/b.py"},
            61: {"rk/a.py"},
            62: {"rk/a.py", "rk/b.py", "rk/c.py"},
            63: {"rk/z.py"},
        }
        st = state(60, 61, 62, 63)
        st["develop_fixes"] = {"63": ["silent-break #1"]}
        p = mergeplan.build(
            CFG,
            st,
            [row(60, g2=PEND), row(61), row(62), row(63)],
            matrix([61, 62], dev=[60, 63], files=files),
        )
        self.assertEqual(
            p["advice"]["60"],
            "rebase now: you're next (conflicts with develop in x.py); after rebase it "
            "shares the most files with #62 (2), #61 (1)",
        )
        self.assertEqual(
            p["advice"]["63"],
            "rebase now: it ends wave 1 (conflicts with develop in x.py)",
        )
        self.assertEqual(
            (p["first_rebase"], mergeplan.place(dict(st, merge_plan=p), 63)),
            ([63], "wave 1, after its rebase"),
        )
        self.assertEqual(
            [mergeplan.wave_index(dict(st, merge_plan=p), n) for n in (63, 60)], [1, 2]
        )  # one wave

    def test_care_is_shown_never_ordered(self):
        info = lambda **kw: dict(
            {
                "rg": set(),
                "dg": set(),
                "abi": set(),
                "cmk": set(),
                "cat": set(),
                "idx": {},
            },
            **kw,
        )
        a = info(
            rg={"conv_wgrad"},
            dg={"gemm", "conv"},
            abi={"binary"},
            cmk={"rk/platform/CMakeLists.txt"},
            cat={"x/ROCKE_ENGINE_test_categories_external.yaml"},
            idx={"gfx942_attention_tiled_2d": [18, 19, 20]},
        )
        b = info(
            rg={"conv_wgrad"},
            dg={"gemm"},
            abi={"binary"},
            cmk={"rk/platform/CMakeLists.txt"},
            cat={"x/ROCKE_ENGINE_test_categories_external.yaml"},
            idx={"gfx942_attention_tiled_2d": [18]},
        )
        self.assertEqual(
            mergeplan.care_reasons(a, b, "rk"),
            [
                "golden shard conv_wgrad",
                "DG gemm",
                "ABI binary",
                "CMake platform/CMakeLists.txt",
                "test categories ROCKE_ENGINE_test_categories_external.yaml",
                "parity gfx942_attention_tiled_2d idx 18 (the first to land keeps it)",
            ],
        )
        M = dict(matrix([70, 71]), facts={70: a, 71: b})
        rows = [row(70), row(71)]
        p = mergeplan.build(CFG, state(70, 71), rows, M)
        self.assertEqual(p["waves"], [[70, 71]])  # care never splits a wave
        self.assertEqual(
            (p["nodes"]["70"]["care"], p["nodes"]["70"]["care_cost"]), ([71], 8)
        )
        self.assertEqual(rows[0]["disrupts"], ["#71◆"])

    def test_golden_and_emitter_reading(self):
        self.assertEqual(
            mergeplan.dg_families(DG, DG.replace('"l0"', '"l1"')), {"famA"}
        )
        self.assertEqual(
            mergeplan.dg_families(
                DG, json.dumps({"ir_canonical": {"famB": {"0": "b"}}})
            ),
            {"famA", "famB"},
        )
        self.assertEqual(
            mergeplan.new_cases(
                "case 0:\ncase 1:\n", "case 0:\n  case 1:\n  case 2: /* x */\ncase 3 :"
            ),
            [2, 3],
        )
        self.assertEqual(mergeplan.new_cases(None, "case 7:"), [7])


PRIO = ["your priority: blocking other work"]


class TestWaves(unittest.TestCase):
    """The waves: each wave is a group of PRs that can be merged together in the given order without causing
    conflicts, ideally ending with a PR that causes conflicts. The
    priorities split across waves when they conflict, the higher one first; each wave then takes in the best set of the
    other PRs clear of its members and of the priorities still waiting; in a wave, the PRs that make no later PR rebase
    first. Hubs last."""

    def test_conflicting_priorities_split_across_waves_in_priority_order(self):
        st = state(1, 2, 3)
        st["develop_fixes"] = {str(n): PRIO for n in (1, 2, 3)}
        rows = [row(n, g3=FAIL) for n in (1, 2, 3)]
        p = mergeplan.build(
            CFG, st, rows, matrix([1, 2, 3], {(1, 2): ["rk/a.py"], (2, 3): ["rk/b.py"]})
        )
        # #2 conflicts with #1, a higher priority: wave 2. #3, a lower priority clear of #1, stays in wave 1.
        self.assertEqual(p["waves"], [[1, 3], [2]])
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {1: (0, True), 3: (1, True), 2: (2, False)},
        )
        self.assertEqual(
            p["advice"]["2"],
            "your priority: wait: #1 (your priority), #3 (your priority) land first "
            "(conflicts in a.py, b.py), then rebase",
        )
        self.assertEqual(
            p["advice"]["1"],
            "your priority: no rebase needed; its landing makes #2 rebase",
        )
        self.assertEqual(
            [mergeplan.place(dict(st, merge_plan=p), n) for n in (1, 2, 3)],
            ["wave 1", "wave 2", "wave 1"],
        )
        # The priorities' order decides, not the PR numbers: a develop fix CI sees goes before the user's priorities.
        st["post_merge"] = {"open": [{"n": 9, "kind": "ci_red"}]}
        st["develop_fixes"] = {"1": PRIO, "2": ["red-develop #9"]}
        p = mergeplan.build(CFG, st, rows[:2], matrix([1, 2], {(1, 2): ["rk/a.py"]}))
        self.assertEqual(
            (p["waves"], p["why"]),
            ([[2], [1]], {"1": "your priority", "2": "fixes develop"}),
        )

    def test_a_pr_clear_of_wave_1_joins_it_unless_it_conflicts_with_a_waiting_priority(
        self,
    ):
        st = state(1, 2, 3, 4)
        st["develop_fixes"] = {"1": PRIO, "2": PRIO}
        rows = [row(1, g3=FAIL), row(2, g3=FAIL), row(3), row(4)]
        p = mergeplan.build(
            CFG,
            st,
            rows,
            matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (2, 4): ["rk/b.py"]}),
        )
        # Ready #3 conflicts with nothing: wave 1, ahead of #1, which makes #2 rebase. Ready #4 is clear of wave 1 too,
        # but it conflicts with #2, a priority waiting for wave 2: it lands after #2 and never makes it rebase.
        self.assertEqual(p["waves"], [[3, 1], [2], [4]])
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows}, {1: True, 2: False, 3: True, 4: False}
        )
        self.assertEqual(
            p["advice"]["4"],
            "wait: #2 (your priority) lands first (conflicts in b.py), then rebase",
        )
        self.assertEqual(
            p["advice"]["2"],
            "your priority: wait: #1 (your priority) lands first (conflicts in a.py), "
            "then rebase; its landing makes #4 rebase",
        )
        self.assertEqual(mergeplan.summary(p), "wave 1 #3 #1; 3 waves")

    def test_a_wave_ends_with_the_prs_that_cause_conflicts(self):
        """In a wave, the PRs whose landing makes no later PR rebase come first, the others after them, the costliest
        last; priorities first in each part."""
        ns = list(range(1, 8))
        st = state(*ns)
        st["develop_fixes"] = {"1": PRIO, "2": PRIO}
        M = matrix(
            ns,
            {
                (1, 6): ["rk/a.py"],
                (4, 6): ["rk/b.py"],
                (5, 6): ["rk/c.py"],
                (5, 7): ["rk/d.py"],
            },
        )
        rows = [
            row(1, g3=FAIL),
            row(2, g3=FAIL),
            row(3),
            row(4),
            row(5),
            row(6, g2=PEND),
            row(7, g2=PEND),
        ]
        p = mergeplan.build(CFG, st, rows, M)
        # #6 waits for #1, #4 and #5, #7 for #5 (ready #5 outweighs R1 #7). Wave 1: #2 (a priority) and #3 make no later
        # PR rebase; then #1 (a priority), #4 (R1 #6 rebases: cost 4) and #5 (#6 and #7: 8), the biggest causer, last.
        self.assertEqual(p["waves"], [[2, 3, 1, 4, 5], [6, 7]])
        self.assertEqual(
            [p["nodes"][str(n)]["cost"] for n in (2, 3, 1, 4, 5)], [0, 0, 4, 4, 8]
        )
        self.assertEqual(p["advice"]["2"], "your priority: no rebase needed")
        self.assertEqual(
            p["advice"]["4"],
            "next: no rebase needed (merges clean with develop and every PR ahead)",
        )
        self.assertEqual(
            p["advice"]["5"],
            "next: no rebase needed (merges clean with develop and every PR ahead); its "
            "landing makes #6, #7 rebase",
        )

    def test_hubs_land_last_even_clear_of_wave_1(self):
        ns = [1, 10, 11, 12, 13, 14, 15]
        st = state(*ns)
        st["develop_fixes"] = {"1": PRIO}
        pairs = {(10, m): ["rk/core.py"] for m in range(11, 16)}
        rows = [row(1, g3=FAIL), row(10)] + [row(n, g2=PEND) for n in range(11, 16)]
        p = mergeplan.build(CFG, st, rows, matrix(ns, pairs))
        # Ready #10 outweighs any one of the five, but it is a hub: they take wave 1 with #1, and it lands after them.
        self.assertEqual(
            (p["waves"], p["hubs"]), ([[1, 11, 12, 13, 14, 15], [10]], [10])
        )
        self.assertIn(
            "; hub (5 conflicts outside the ABI lane): it lands after the others",
            p["advice"]["10"],
        )
        self.assertEqual(mergeplan.wave_rows(p)[1][0], "wave 2 (hubs)")

    def test_the_priorities_in_order_pinned_groups_top_priorities_held_starving(self):
        """A pinned group first, then the top priorities, then held PRs (they keep their place), then a ready PR the
        plan kept waiting a day: four PRs that all conflict take four waves in that order.
        """
        ns = [10, 11, 20, 30, 40]
        st = state(*ns)
        st["groups"] = [[11, 10]]
        st["develop_fixes"] = {"20": PRIO}
        st["merge_plan"] = {
            "held_back": {"40": {"head": head(40)[:12], "since": ago(25 * 60)}}
        }
        M = matrix(
            ns,
            {
                (a, b): ["rk/a.py"]
                for a, b in itertools.combinations((10, 20, 30, 40), 2)
            },
        )
        rows = [
            row(10, g2=PEND),
            row(11, g2=PEND),
            row(20, g3=FAIL),
            row(30, held={"status": "held"}),
            row(40),
        ]
        p = mergeplan.build(CFG, st, rows, M)
        self.assertEqual(p["waves"], [[11, 10], [20], [30], [40]])
        self.assertEqual(
            p["why"],
            {
                "10": "pinned group",
                "11": "pinned group",
                "20": "your priority",
                "30": "held: it keeps its place",
                "40": "ready, kept waiting a day",
            },
        )

    def test_a_top_priority_that_rebases_first_ends_its_wave_and_its_rivals_land_after_it(
        self,
    ):
        """The matrix merges no pair with a PR that conflicts with develop: a file in common counts, and so does another
        claim in its ABI lane. Such a PR lands after it, as landing first could make the priority rebase again.
        """
        files = {
            1: {"rk/f1.py"},
            3: {"rk/s.py", "rk/f3.py"},
            4: {"rk/f4.py"},
            6: {"rk/f6.py"},
            9: {"rk/s.py"},
        }
        st = state(1, 3, 4, 6, 9)
        st["develop_fixes"] = {"1": PRIO, "9": PRIO}
        bumps(st, (6, "binary", 7, "h6"), (9, "binary", 7, "h9"))
        rows = [row(1, g3=FAIL), row(3), row(4, g2=PEND), row(6), row(9, g3=FAIL)]
        p = mergeplan.build(CFG, st, rows, matrix([1, 3, 4, 6], dev=[9], files=files))
        self.assertEqual((p["waves"], p["first_rebase"]), ([[1, 4], [3, 6]], [9]))
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {1: (0, True), 4: (1, True), 9: (2, False), 3: (3, False), 6: (4, False)},
        )
        self.assertTrue(
            p["advice"]["9"].startswith(
                "rebase now: it ends wave 1 (conflicts with develop in x.py)"
            )
        )
        self.assertEqual(
            p["advice"]["3"],
            "wait: #9 (your priority) lands first, after its rebase (shares s.py with "
            "it)",
        )
        st["merge_plan"] = p
        self.assertEqual(
            [mergeplan.place(st, n) for n in (1, 9, 3)],
            ["wave 1", "wave 1, after its rebase", "wave 2"],
        )
        self.assertEqual([mergeplan.wave_index(st, n) for n in (1, 9, 3)], [1, 1, 2])
        self.assertEqual(
            mergeplan.summary(p),
            "wave 1 #1 #4; after its rebase #9; 2 waves; ABI in turn: binary ×2; "
            "1 rebase first",
        )
        # Unless the hold rule keeps the priority waiting: then its ABI lane orders nothing (the hold rule).
        rows = [
            row(1, g3=FAIL),
            row(3),
            row(4, g2=PEND),
            row(6),
            row(9, g3=FAIL, held={"status": "held"}),
        ]
        p = mergeplan.build(CFG, st, rows, matrix([1, 3, 4, 6], dev=[9], files=files))
        self.assertEqual(p["waves"], [[1, 6, 4], [3]])

    def test_wave_1_holds_only_the_priority_that_rebases_first_when_every_other_pr_shares_its_files(
        self,
    ):
        files = {
            3: {"rk/s.py"},
            4: {"rk/s.py", "rk/t.py"},
            5: {"rk/u.py"},
            9: {"rk/s.py"},
        }
        st = state(3, 4, 5, 9)
        st["develop_fixes"] = {"9": PRIO}
        rows = [row(3), row(4, g2=PEND), row(5, g2=PEND), row(9, g3=FAIL)]
        p = mergeplan.build(CFG, st, rows, matrix([3, 4], dev=[5, 9], files=files))
        self.assertEqual((p["waves"], p["first_rebase"]), ([[], [3, 4]], [9]))
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {9: (0, False), 3: (1, False), 4: (2, False), 5: (3, False)},
        )
        st["merge_plan"] = p
        self.assertTrue(
            report.plan_row(CFG, st, p)[0].startswith("wave 1: #9 after its rebase")
        )
        self.assertEqual([mergeplan.wave_index(st, n) for n in (9, 3, 5)], [1, 2, 3])
        md = "\n".join(report.plan_md(CFG, st))
        self.assertIn("| wave 1 | #9 T0, after its rebase | — |", md)
        self.assertIn("| wave 2 | #3 R0, #4 R1 | none |", md)
        # No PR merges clean with develop: wave 1 still holds the priority, and the others rank after it.
        p = mergeplan.build(
            CFG,
            st,
            [row(5, g2=PEND), row(9, g3=FAIL)],
            matrix([], dev=[5, 9], files=files),
        )
        self.assertEqual((p["waves"], p["first_rebase"]), ([[]], [9]))
        st["merge_plan"] = p
        self.assertEqual([mergeplan.wave_index(st, n) for n in (9, 5)], [1, 2])
        self.assertEqual(
            mergeplan.summary(p), "after its rebase #9; 1 wave; 2 rebase first"
        )

    def test_two_rebasing_priorities_that_share_a_file_go_to_waves_1_and_2(self):
        files = {1: {"rk/f1.py"}, 8: {"rk/s.py", "rk/f8.py"}, 9: {"rk/s.py"}}
        st = state(1, 8, 9)
        st["develop_fixes"] = {"8": PRIO, "9": PRIO}
        rows = [row(1), row(8, g3=FAIL), row(9, g3=FAIL)]
        p = mergeplan.build(CFG, st, rows, matrix([1], dev=[8, 9], files=files))
        self.assertEqual((p["waves"], p["first_rebase"]), ([[1], []], [8, 9]))
        self.assertEqual(
            {r["n"]: (r["plan_rank"], r["plan_go"]) for r in rows},
            {1: (0, True), 8: (1, False), 9: (2, False)},
        )
        st["merge_plan"] = p
        self.assertEqual(
            [mergeplan.place(st, n) for n in (8, 9)],
            ["wave 1, after its rebase", "wave 2, after its rebase"],
        )
        self.assertTrue(
            p["advice"]["8"].startswith(
                "rebase now: it ends wave 1 (conflicts with develop in x.py)"
            )
        )
        # Rebased now, #9 would rebase again once #8 lands: it rebases after wave 1.
        self.assertTrue(
            p["advice"]["9"].startswith(
                "rebase after wave 1 lands: it ends wave 2 (conflicts with "
                "develop in x.py)"
            )
        )
        md = "\n".join(report.plan_md(CFG, st))
        self.assertIn("| wave 1 | #1 R0 ✓, #8 T0, after its rebase | none |", md)
        self.assertIn("| wave 2 | #9 T0, after its rebase | — |", md)
        self.assertEqual(
            mergeplan.summary(p),
            "wave 1 #1; after its rebase #8; after its rebase in wave 2 #9; "
            "2 waves; 2 rebase first",
        )

    def test_a_rebasing_priority_that_shares_a_file_with_a_higher_clean_priority_goes_to_the_next_wave(
        self,
    ):
        files = {1: {"rk/s.py"}, 2: {"rk/f2.py"}, 9: {"rk/s.py", "rk/f9.py"}}
        st = state(1, 2, 9)
        st["post_merge"] = {"open": [{"n": 5, "kind": "ci_red"}]}
        st["develop_fixes"] = {"1": ["red-develop #5"], "9": PRIO}
        rows = [row(1, g3=FAIL), row(2), row(9, g3=FAIL)]
        p = mergeplan.build(CFG, st, rows, matrix([1, 2], dev=[9], files=files))
        # #1, a develop fix CI sees, goes before the operator's #9, which shares s.py with it: #9 ends wave 2.
        self.assertEqual((p["waves"], p["first_rebase"]), ([[2, 1], []], [9]))
        st["merge_plan"] = p
        self.assertEqual(mergeplan.place(st, 9), "wave 2, after its rebase")
        self.assertTrue(
            p["advice"]["9"].startswith("rebase after wave 1 lands: it ends wave 2")
        )
        self.assertEqual(
            p["advice"]["1"],
            "fixes develop: no rebase needed; its landing makes #9 rebase",
        )
        # The dashboards name it among the PRs wave 1's landing makes rebase, though it is in no node.
        md = "\n".join(report.plan_md(CFG, st))
        self.assertIn("| wave 1 | #2 R0 ✓, #1 T0 ✓ | #9 (T0) |", md)
        self.assertIn("| wave 2 | #9 T0, after its rebase | — |", md)
        self.assertIn(">#9 T0</a></div></td><td>", webdash.plan_card(CFG, st))

    def test_a_pr_that_shares_its_files_waits_until_after_it(self):
        """Even when it is clear of wave 1: a PR that shares a file with a rebasing priority waiting for wave 2 lands
        after it."""
        files = {
            1: {"rk/a.py"},
            3: {"rk/s.py"},
            4: {"rk/f4.py"},
            9: {"rk/a.py", "rk/s.py"},
        }
        st = state(1, 3, 4, 9)
        st["post_merge"] = {"open": [{"n": 5, "kind": "ci_red"}]}
        st["develop_fixes"] = {"1": ["red-develop #5"], "9": PRIO}
        rows = [row(1, g3=FAIL), row(3), row(4), row(9, g3=FAIL)]
        p = mergeplan.build(CFG, st, rows, matrix([1, 3, 4], dev=[9], files=files))
        self.assertEqual(p["waves"], [[4, 1], [], [3]])
        st["merge_plan"] = p
        self.assertEqual(
            [mergeplan.place(st, n) for n in (9, 3)],
            ["wave 2, after its rebase", "wave 3"],
        )
        self.assertEqual(
            p["advice"]["3"],
            "wait: #9 (your priority) lands first, after its rebase (shares s.py with "
            "it)",
        )
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows}, {1: True, 3: False, 4: True, 9: False}
        )

    def test_todays_rebasing_priorities_split_across_waves(self):
        """Priorities that all conflict with develop in one file, and share files with each other and with a clean
        priority in wave 1, do not all end wave 1. Here #13, #59 and #8 rebase
        and share files the same way, and #22 shares with two of them: each takes the earliest wave its files allow,
        and wave 1 has no conflict inside. #30, which shares a file with #59 and #8, lands after both.
        """
        f = lambda *names: {f"rk/{x}.py" for x in names}  # noqa: E731
        files = {
            13: f("m2", "m3", "f13"),
            22: f("m4", "f22"),
            59: f("m1", "m2", "m4"),
            8: f("m1", "m3", "m4"),
            30: f("m1"),
            40: f("f40"),
        }
        ns = (13, 22, 59, 8, 30, 40)
        st = state(*ns)
        st["develop_fixes"] = {str(n): PRIO for n in (13, 22, 59, 8)}
        rows = [
            row(13, g3=FAIL, size=100),
            row(22, g3=FAIL, size=200),
            row(59, g3=FAIL, size=300),
            row(8, g3=FAIL, size=400),
            row(30),
            row(40),
        ]
        p = mergeplan.build(
            CFG, st, rows, matrix([22, 30, 40], dev=[13, 59, 8], files=files)
        )
        st["merge_plan"] = p
        self.assertEqual(
            (p["waves"], p["first_rebase"]), ([[40, 22], [], [], [30]], [13, 59, 8])
        )
        self.assertEqual([mergeplan.wave_index(st, n) for n in ns], [1, 1, 2, 3, 4, 1])
        wave1 = [n for n in ns if mergeplan.wave_index(st, n) == 1]
        self.assertFalse(
            any(files[a] & files[b] for a, b in itertools.combinations(wave1, 2))
        )
        self.assertEqual(
            p["advice"]["30"],
            "wait: #59 (your priority), #8 (your priority) land first, after their "
            "rebase (shares m1.py with them)",
        )

    def test_what_may_land_now(self):
        """plan_go (it feeds the batches): wave 1, with no PR ranked ahead of it conflicting. In a pinned group, a
        member that conflicts with an earlier one waits; a later wave never goes."""
        st = state(1, 2, 3, 4)
        st["groups"] = [[1, 2]]
        rows = [row(n) for n in (1, 2, 3, 4)]
        p = mergeplan.build(
            CFG,
            st,
            rows,
            matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (3, 4): ["rk/b.py"]}),
        )
        self.assertEqual(p["waves"], [[1, 2, 3], [4]])
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows}, {1: True, 2: False, 3: True, 4: False}
        )
        self.assertEqual(
            p["advice"]["2"],
            "pinned group: wait: #1 (pinned group) lands first (conflicts in a.py), "
            "then rebase",
        )

    def test_a_pair_not_checked_yet_counts_as_a_conflict(self):
        st = state(1, 2, 3, 4, 7)
        st["develop_fixes"] = {"1": PRIO}
        M = matrix([1, 2, 3, 4], {(1, 2): None, (3, 4): None}, unknown=[7])
        rows = [row(1, g3=FAIL)] + [row(n, g2=PEND) for n in (2, 3, 4, 7)]
        p = mergeplan.build(CFG, st, rows, M)
        # #2 may conflict with #1: wave 2. #3 and #4 may conflict: only one of them is in wave 1 (the older). #7, whose
        # merge with develop is not known yet, is not planned at all.
        self.assertEqual(
            (p["waves"], p["partial"]),
            ([[1, 3], [2, 4]], {"pairs": 2, "prs": 1, "waves": 0}),
        )
        self.assertEqual(
            p["advice"]["2"],
            "wait: #1 (your priority) lands first (not checked yet: counted as a "
            "conflict), then rebase",
        )
        self.assertEqual(
            p["advice"]["4"],
            "wait: #3 lands first (not checked yet: counted as a conflict), then rebase",
        )
        self.assertIsNone(rows[4]["plan_rank"])
        self.assertTrue(mergeplan.summary(p).endswith("; 2 waves; partial"))

    def test_random_plans_keep_the_rule(self):
        """Random plans, with top priorities that must rebase first (they conflict with a PR when they share a changed
        file with it): every wave without a conflict inside; every PR planned once; each priority in the earliest wave
        with no higher priority it conflicts with; no other PR in a wave before or with a priority it conflicts with; in
        each wave the PRs that make no later PR rebase first and the rebasing priorities last; wave 1 takes in a best
        set of the PRs clear of every priority (hubs aside); only wave 1's PRs that merge clean may land now.
        """
        rng = random.Random(1105)
        gates = {
            "T0": {"g3": FAIL},
            "R0": {},
            "R1": {"g2": PEND},
            "R3": {"g1": ["fail", f"our verdict {O}"]},
            "R4": {"g1": ["fail", f"our verdict {R}"]},
        }
        for _ in range(200):
            ns = list(range(1, rng.randint(2, 10) + 1))
            cls = {
                n: rng.choice(("T0", "T0", "R0", "R1", "R1", "R3", "R4")) for n in ns
            }
            held = {n for n in ns if cls[n] == "R4" and rng.random() < 0.4}
            dev = {n for n in ns if cls[n] == "T0" and rng.random() < 0.4}
            clean = [n for n in ns if n not in dev]
            size = {n: rng.choice((10, 100, 1000)) for n in ns}
            files = {
                n: {f"rk/f{n}.py"}
                | {f"rk/s{k}.py" for k in (1, 2, 3) if rng.random() < 0.25}
                for n in ns
            }
            pairs = {
                (a, b): rng.choice((["rk/x.py"], ["rk/x.py"], None))
                for a, b in itertools.combinations(clean, 2)
                if rng.random() < 0.35
            }
            st = state(*ns)
            st["develop_fixes"] = {str(n): PRIO for n in ns if cls[n] == "T0"}
            rows = [
                row(
                    n,
                    size=size[n],
                    **gates[cls[n]],
                    **({"held": {"status": "held"}} if n in held else {}),
                )
                for n in ns
            ]
            p = mergeplan.build(
                CFG, st, rows, matrix(clean, pairs, dev=sorted(dev), files=files)
            )
            st["merge_plan"] = p
            ctx = (cls, sorted(held), sorted(dev), files, pairs, p["waves"])
            prio = sorted(
                (n for n in ns if cls[n] == "T0"), key=lambda n: (size[n], n)
            ) + sorted(held)
            wave = {n: mergeplan.wave_index(st, n) for n in ns}
            rank = {r["n"]: r["plan_rank"] for r in rows}

            def clash(a: int, b: int) -> bool:
                return (
                    bool(files[a] & files[b])
                    if {a, b} & dev
                    else (min(a, b), max(a, b)) in pairs
                )

            def free(s) -> bool:
                return not any(clash(a, b) for a, b in itertools.combinations(s, 2))

            self.assertEqual(sorted(n for w in p["waves"] for n in w), clean, ctx)
            self.assertEqual(sorted(rank.values()), list(range(len(ns))), ctx)
            self.assertEqual(
                p["first_rebase"], sorted(dev, key=lambda n: (size[n], n)), ctx
            )
            for k in range(1, len(p["waves"]) + 1):
                members = sorted((n for n in ns if wave[n] == k), key=rank.get)
                self.assertTrue(
                    members and free(members), ctx
                )  # never empty, no conflict inside
                kinds = [
                    2 if n in dev else int(any(clash(n, m) and wave[m] > k for m in ns))
                    for n in members
                ]
                self.assertEqual(
                    kinds, sorted(kinds), ctx
                )  # the PRs that cause conflicts, then the rebasing ones
            for i, x in enumerate(prio):
                k = 1
                while any(wave[q] == k and clash(x, q) for q in prio[:i]):
                    k += 1
                self.assertEqual(wave[x], k, ctx)
            for x in set(clean) - set(prio):
                self.assertFalse(
                    any(clash(x, q) and wave[x] <= wave[q] for q in prio), ctx
                )
            for d in dev:
                k = wave[d]
                lead = (
                    "rebase now: it ends wave 1"
                    if k == 1
                    else f"rebase after wave {k - 1} lands: it ends wave {k}"
                )
                self.assertTrue(p["advice"][str(d)].startswith(lead), ctx)
            others = [n for n in clean if n not in prio]
            hub = {n for n in others if p["nodes"][str(n)]["hub"]}
            cand = [
                n
                for n in others
                if not any(clash(n, q) for q in prio)
                and (n not in hub or not set(others) - hub)
            ]
            best = max(
                sum(mergeplan.WEIGHT[cls[n]] for n in s)
                for k in range(len(cand) + 1)
                for s in itertools.combinations(cand, k)
                if free(s)
            )
            self.assertEqual(
                sum(mergeplan.WEIGHT[cls[n]] for n in p["waves"][0] if n not in prio),
                best,
                ctx,
            )
            self.assertEqual(
                {r["n"] for r in rows if r["plan_go"]}, set(p["waves"][0]), ctx
            )


class TestUnionCheck(unittest.TestCase):
    """The union check: a whole stack (PR A + B + C + D) must stay unconflicted, not each PR alone. Each wave's
    members are merged on develop one after another, in landing order; a member that conflicts with the union of those
    before it (each alone merges clean) conflicts with them from then on, and the waves are laid out again.
    """

    FILES = {1: {"rk/s.py"}, 2: {"rk/s.py", "rk/a.py"}, 3: {"rk/s.py"}, 4: {"rk/f4.py"}}

    def setUp(self):
        self.calls, self.bad = [], [{"t1", "t2", "t3"}]

    def merge_tree(self, repo, a, b, base=None, timeout=600):
        """A fake gitops.merge_tree on trees named by the PRs in them ("t1", "t1+t2", ...): a union that holds one of
        `bad` conflicts in rk/s.py (by default #1, #2 and #3 together; any two of them merge clean).
        """
        self.calls.append((a, b))
        tree = f"{a}+{b}"
        return tree, (
            ["rk/s.py"] if any(s <= set(tree.split("+")) for s in self.bad) else []
        )

    def plan(
        self, st: dict, cfg: dict = CFG, M: dict | None = None
    ) -> tuple[dict, list]:
        M = M or matrix(
            [1, 2, 3, 4], {(1, 2): [], (1, 3): [], (2, 3): []}, files=self.FILES
        )
        M["merge"], M["budget"] = mergeplan.merger(cfg)  # a new tick's budget
        rows = [row(n) for n in (1, 2, 3, 4)]
        with mock.patch.object(mergeplan.gitops, "merge_tree", self.merge_tree):
            return mergeplan.build(cfg, st, rows, M), rows

    def test_three_prs_that_merge_clean_in_pairs_but_conflict_together_split(self):
        st = state(1, 2, 3, 4)
        p, rows = self.plan(st)
        # #3's step conflicts with #1 + #2 in s.py: #3 conflicts with both from then on and waits for wave 2. #4 shares
        # no file with them: no merge. The second layout's chain (#1, #2) comes from the cache.
        self.assertEqual(p["waves"], [[4, 1, 2], [3]])
        self.assertEqual(self.calls, [("t1", "t2"), ("t1+t2", "t3")])
        self.assertEqual((p["merges"], p["partial"]), (2, None))
        self.assertEqual(p["unions"], [{"prs": [1, 2, 3], "files": ["rk/s.py"]}])
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows}, {1: True, 2: True, 3: False, 4: True}
        )
        self.assertEqual(
            p["advice"]["3"],
            "conflicts with #1 + #2 together in s.py (each alone merges clean): next "
            "wave",
        )
        self.assertEqual(
            p["advice"]["2"],
            "next: no rebase needed (merges clean with develop and every PR ahead); its "
            "landing makes #3 rebase",
        )
        self.assertEqual(p["edges"], {})  # each pair merges clean: no ✗ in the grid
        st["merge_plan"] = p
        self.assertIn(
            "1 union conflicts (PRs that conflict only together)",
            "\n".join(report.plan_md(CFG, st)),
        )

    def test_a_chain_not_checked_yet_keeps_the_plan_partial(self):
        st = state(1, 2, 3, 4)
        p, _ = self.plan(st, dict(CFG, merge_plan={"max_merges": 1}))
        # One merge in the budget: #1 + #2 merge clean, #3's step waits for the next tick. The wave stands.
        self.assertEqual(
            (p["waves"], p["unions"], p["partial"]),
            ([[1, 2, 3, 4]], [], {"pairs": 0, "prs": 0, "waves": 1}),
        )
        st["merge_plan"] = p
        self.assertTrue(
            report.plan_row(CFG, st, p)[1].endswith(
                "1 wave; partial: 0 pairs, 0 PRs not checked yet, 1 wave not union-checked"
            )
        )
        self.assertTrue(mergeplan.summary(p).endswith("; partial"))

    def test_a_cached_chain_costs_no_merge(self):
        st = state(1, 2, 3, 4)
        self.plan(st)
        self.calls.clear()
        p, _ = self.plan(st)  # the next tick: a new budget, the same develop and heads
        self.assertEqual(
            (self.calls, p["merges"], p["waves"]), ([], 0, [[4, 1, 2], [3]])
        )
        # A chain the budget cut short goes on from its cached steps on the next tick.
        st = state(1, 2, 3, 4)
        self.plan(st, dict(CFG, merge_plan={"max_merges": 1}))
        self.calls.clear()
        p, _ = self.plan(st, dict(CFG, merge_plan={"max_merges": 1}))
        self.assertEqual(
            (self.calls, p["waves"], p["partial"]),
            ([("t1+t2", "t3")], [[4, 1, 2], [3]], None),
        )

    def test_a_pinned_groups_own_conflict_moves_nothing(self):
        """A pinned group lands as the user pinned it: its members' conflict with each other is no union conflict."""
        self.bad = [{"t1", "t2"}]
        st = state(1, 2, 3, 4)
        st["groups"] = [[1, 2]]
        p, _ = self.plan(
            st,
            M=matrix(
                [1, 2, 3, 4],
                {(1, 2): ["rk/s.py"], (1, 3): [], (2, 3): []},
                files=self.FILES,
            ),
        )
        self.assertEqual(
            (p["waves"], p["unions"], p["partial"]), ([[1, 2, 3, 4]], [], None)
        )
        self.assertEqual(
            self.calls, [("t1", "t2"), ("t1", "t3")]
        )  # #3 checked against #1 alone


class TestReadyEarly(unittest.TestCase):
    """A heads-up when priority PRs in later waves are ready before earlier waves, so the operator can merge them
    and have the queues re-calculated. A top priority ready now in a later wave, while an earlier wave waits
    on a PR that is not: a tick line once per PR, head and wave, a Needs-you item while it holds, a chip marker.
    """

    LINE = (
        "PRIORITY_READY_EARLY #2 (wave 2, your priority): ready now, but wave 1 still waits on #1; landing it now "
        "makes #1 rebase; your call"
    )

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.cfg = dict(
            CFG,
            paths={
                "repo": "/nonexistent",
                "html": str(pathlib.Path(self.d.name) / "x.html"),
                "daily_dir": self.d.name,
            },
        )
        self.st = state(1, 2)
        self.st["develop_fixes"] = {"1": PRIO, "2": PRIO}

    def tearDown(self):
        self.d.cleanup()

    def tick(self, ready: set) -> list:
        """One tick's plan, with `ready` the PRs at the Ready stage, and its lines."""
        rows, M = [row(1, g3=FAIL), row(2)], matrix([1, 2], {(1, 2): ["rk/a.py"]})
        mergeplan.finish(
            self.cfg, self.st, mergeplan.build(self.cfg, self.st, rows, M), ready
        )
        return mergeplan.ready_lines(self.st)

    def test_a_ready_priority_behind_a_wave_that_waits_is_told_once_per_head_and_shows_until_it_lands(
        self,
    ):
        # #2 conflicts with #1, a higher priority: wave 2. #1's CI is red; #2 passes every readiness gate.
        self.assertEqual(self.tick({2}), [self.LINE])
        mp = self.st["merge_plan"]
        self.assertEqual(
            mp["ready_early"],
            {
                "2": {
                    "wave": 2,
                    "why": "your priority",
                    "waits": [1],
                    "rebase": [1],
                    "renumber": [],
                    "key": f"{head(2)[:12]}:2",
                }
            },
        )
        self.assertEqual(
            mergeplan.ready_lines(self.st), []
        )  # once per PR, head and wave
        self.assertEqual(self.tick({2}), [])  # the next tick's plan keeps what was told
        needs = report.board(
            self.cfg, self.st, Store(pathlib.Path(self.d.name)), [], []
        )["needs"]
        self.assertIn(
            "#2 (your priority) is ready to merge now but waits in wave 2 behind #1. Landing it now makes #1 "
            "rebase. Your call: merge it now, or let the waves run.",
            needs,
        )
        card = webdash.plan_card(self.cfg, self.st)
        self.assertIn(
            "'>#2 T0<span class='chip-ready-early'>ready</span></a>", card
        )  # its chip in the wave 2 row
        self.assertEqual(card.count("chip-ready-early"), 1)
        # A new head: told again. Not ready for a tick, then ready again: told again.
        self.st["prs"]["2"]["head"] = "f" * 40
        self.assertEqual(self.tick({2}), [self.LINE])
        self.assertEqual(
            (self.tick(set()), self.st["merge_plan"]["ready_early"]), ([], {})
        )
        self.assertEqual(self.tick({2}), [self.LINE])
        # The plan not computed for a tick: nothing printed, and what was told stays told.
        mergeplan.unavailable(
            self.st, "no near-merge PR could be merged with develop here"
        )
        self.assertEqual(mergeplan.ready_lines(self.st), [])
        self.assertEqual(self.tick({2}), [])

    def test_the_queue_reads_the_ready_stage_not_the_class(self):
        """build_queue: the Ready stage (every readiness gate, G9 included) decides; T0 hides it in the class."""
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["develop_fixes"] = {"1": PRIO, "2": PRIO}
        st["prs"] = {
            "1": rec(1, review_decision="APPROVED", ci_rollup="FAILURE"),
            "2": rec(2, review_decision="APPROVED"),
        }
        M = matrix([1, 2], {(1, 2): ["rk/a.py"]})
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: M
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue, "abi_lanes", lambda *a: {}
        ), mock.patch.object(
            queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        self.assertEqual({r["n"]: r["stage"] for r in q}, {1: "candidate", 2: "ready"})
        self.assertEqual(
            (st["merge_plan"]["waves"], list(st["merge_plan"]["ready_early"])),
            ([[1], [2]], ["2"]),
        )
        self.assertEqual(mergeplan.ready_lines(st), [self.LINE])

    def test_no_line_while_every_earlier_wave_is_ready(self):
        self.assertEqual(
            (self.tick({1, 2}), self.st["merge_plan"]["ready_early"]), ([], {})
        )
        needs = report.board(
            self.cfg, self.st, Store(pathlib.Path(self.d.name)), [], []
        )["needs"]
        self.assertFalse(
            [n for n in needs if "ready to merge now but waits" in n], needs
        )

    def test_no_line_for_a_pr_that_is_no_top_priority_or_must_rebase_first(self):
        """#3 is no priority, #9 a top priority that must rebase first: neither is told, even when counted ready."""
        files = {1: {"rk/a.py", "rk/b.py"}, 3: {"rk/b.py"}, 9: {"rk/a.py"}}
        st = state(1, 3, 9)
        st["post_merge"] = {"open": [{"n": 5, "kind": "ci_red"}]}
        st["develop_fixes"] = {"1": ["red-develop #5"], "9": PRIO}
        rows = [row(1, g3=FAIL), row(3), row(9, g3=FAIL)]
        p = mergeplan.build(
            self.cfg,
            st,
            rows,
            matrix([1, 3], {(1, 3): ["rk/b.py"]}, dev=[9], files=files),
        )
        mergeplan.finish(self.cfg, st, p, {3, 9})
        self.assertEqual([mergeplan.wave_index(st, n) for n in (1, 3, 9)], [1, 2, 2])
        self.assertEqual(
            (st["merge_plan"]["ready_early"], mergeplan.ready_lines(st)), ({}, [])
        )

    def test_the_rebase_list_names_the_earlier_prs_it_conflicts_with_and_the_abi_claims_it_jumps(
        self,
    ):
        """#5 (wave 3) conflicts with #1 (wave 1) and #2 (wave 2), shares a file with #9, which rebases first in wave 1,
        and bumps ROCKE_ABI_VERSION as #4 (wave 1) does: landing it now makes #1, #9 and #2 rebase and #4 renumber.
        """
        files = {
            1: {"rk/a.py", "rk/b.py"},
            2: {"rk/a.py", "rk/c.py"},
            4: {"rk/f4.py"},
            5: {"rk/b.py", "rk/c.py", "rk/s.py"},
            9: {"rk/s.py"},
        }
        st = state(1, 2, 4, 5, 9)
        st["post_merge"] = {"open": [{"n": 5, "kind": "ci_red"}]}
        st["develop_fixes"] = {
            "1": ["red-develop #5"],
            "2": PRIO,
            "4": PRIO,
            "5": PRIO,
            "9": PRIO,
        }
        bumps(st, (4, "binary", 7, "h4"), (5, "binary", 7, "h5"))
        rows = [
            row(1, g3=FAIL),
            row(2, size=100),
            row(4, size=150),
            row(5, size=200),
            row(9, g3=FAIL, size=50),
        ]
        M = matrix(
            [1, 2, 4, 5],
            {(1, 2): ["rk/a.py"], (1, 5): ["rk/b.py"], (2, 5): ["rk/c.py"]},
            dev=[9],
            files=files,
        )
        mergeplan.finish(
            self.cfg, st, mergeplan.build(self.cfg, st, rows, M), {2, 4, 5}
        )
        mp = st["merge_plan"]
        self.assertEqual(
            [mergeplan.wave_index(st, n) for n in (1, 9, 4, 2, 5)], [1, 1, 1, 2, 3]
        )
        self.assertEqual(
            mp["ready_early"]["5"],
            {
                "wave": 3,
                "why": "your priority",
                "waits": [1, 9],
                "rebase": [1, 9, 2],
                "renumber": [4],
                "key": f"{head(5)[:12]}:3",
            },
        )
        self.assertEqual(
            sorted(mp["ready_early"]), ["2", "5"]
        )  # #2 too: ready in wave 2 behind #1 and #9
        self.assertEqual(
            mergeplan.ready_lines(st),
            [
                "PRIORITY_READY_EARLY #2 (wave 2, your priority): ready now, but wave 1 still waits on #1, #9; landing it "
                "now makes #1 rebase; your call",
                "PRIORITY_READY_EARLY #5 (wave 3, your priority): ready now, but waves 1..2 still wait on #1, #9; landing "
                "it now makes #1, #9, #2 rebase and #4 renumber its ABI number; your call",
            ],
        )
        self.assertEqual(
            mergeplan.ready_effect({"rebase": [], "renumber": []}), "forces no rebase"
        )
        self.assertEqual(
            mergeplan.ready_effect({"rebase": [], "renumber": [3, 4]}),
            "makes #3, #4 renumber their ABI numbers",
        )


class TestAbiConflicts(unittest.TestCase):
    """ABI differences count as a conflict, so they help determine the merge order. Two claims in one ABI lane
    conflict even when git merges them cleanly."""

    def test_two_claims_in_one_lane_never_share_a_wave_and_the_later_renumbers(self):
        st = state(1, 2, 3, 4, 5)
        # #1 and #3 carry the same bump (one claim); #2 another binary claim; #4 and #5 two wire claims.
        bumps(
            st,
            (1, "binary", 7, "h1"),
            (3, "binary", 7, "h1"),
            (2, "binary", 7, "h2"),
            (4, "recipe", 3, "r4"),
            (5, "recipe", 3, "r5"),
        )
        rows = [row(1)] + [row(n, g2=PEND) for n in (2, 3, 4, 5)]
        M = matrix([1, 2, 3, 4, 5])  # git merges every pair cleanly
        p = mergeplan.build(CFG, st, rows, M)
        self.assertEqual(
            p["edges"],
            {
                "1-2": {"abi": ["binary"]},
                "2-3": {"abi": ["binary"]},
                "4-5": {"abi": ["recipe"]},
            },
        )
        self.assertEqual(p["waves"], [[1, 3, 4], [2, 5]])
        self.assertEqual(
            {r["n"]: r["plan_go"] for r in rows},
            {1: True, 3: True, 4: True, 2: False, 5: False},
        )
        self.assertEqual(
            p["advice"]["2"],
            "after #1 lands: renumber ROCKE_ABI_VERSION to develop + 1 (and BINARY_ABI)",
        )
        self.assertEqual(
            p["advice"]["5"],
            "after #4 lands: renumber ROCKE_RECIPE_ABI to develop + 1 (and RECIPE_ABI)",
        )
        self.assertEqual(
            p["advice"]["1"],
            "next: no rebase needed (merges clean with develop and every PR ahead); "
            "first in the ROCKE_ABI_VERSION lane: it takes 7",
        )
        # #4 ends wave 1: its advice names the PR it makes renumber.
        self.assertTrue(
            p["advice"]["4"].endswith(
                "; first in the ROCKE_RECIPE_ABI lane: it takes 3; its landing makes "
                "#5 rebase"
            )
        )
        self.assertEqual(
            (
                p["nodes"]["2"]["abi"],
                p["nodes"]["2"]["conflicts"],
                p["nodes"]["1"]["forces"],
            ),
            ([1, 3], [], [2]),
        )
        self.assertEqual(rows[0]["disrupts"], ["#2Ⓐ"])
        self.assertEqual(
            mergeplan.conflict_score(M), {n: [] for n in (1, 2, 3, 4, 5)}
        )  # only textual ones hold
        self.assertIn(
            "2 waves; ABI in turn: binary ×2, recipe ×2", mergeplan.summary(p)
        )

    def test_abi_conflicts_never_make_hubs(self):
        ns = list(range(1, 8))
        st = state(*ns)
        bumps(st, *[(n, "binary", 7, f"h{n}") for n in ns])
        p = mergeplan.build(CFG, st, [row(n, g2=PEND) for n in ns], matrix(ns))
        self.assertEqual(
            (p["hubs"], [len(w) for w in p["waves"]]), ([], [1] * 7)
        )  # one claim per wave
        self.assertTrue(all(node["hub_degree"] == 0 for node in p["nodes"].values()))
        self.assertEqual(
            [c["members"] for c in p["abi"]["binary"]["claims"]], p["waves"]
        )  # the lane follows the plan

    def test_a_pr_the_hold_rule_keeps_waiting_never_blocks_its_lane(self):
        """The hold rule exempts ABI overlaps (the next ready lane member may take the
        number), so a held PR's ABI conflicts are shown, not ordered, and it renumbers after the rest of its lane.
        """
        st = state(1, 2)
        bumps(st, (1, "binary", 7, "h1"), (2, "binary", 7, "h2"))
        rows = [row(1, held={"status": "held"}, static_ok=False), row(2)]
        p = mergeplan.build(CFG, st, rows, matrix([1, 2]))
        # Its ABI conflict with held #1 orders nothing: both are in wave 1.
        self.assertEqual(
            (p["waves"], p["edges"]), ([[1, 2]], {"1-2": {"abi": ["binary"]}})
        )
        self.assertTrue(
            rows[1]["plan_go"]
        )  # the held PR ahead of it does not hold it back
        self.assertEqual(
            [c["members"] for c in p["abi"]["binary"]["claims"]], [[2], [1]]
        )
        self.assertEqual(
            p["advice"]["1"],
            "held: it keeps its place: after #2 lands: renumber ROCKE_ABI_VERSION to "
            "develop + 1 (and BINARY_ABI)",
        )
        lanes = queue.abi_lanes(CFG, st, rows, list(st["prs"].values()), "d" * 40)
        self.assertEqual(
            (
                [g["members"] for g in lanes["binary"]["groups"]],
                lanes["binary"]["head"],
            ),
            ([[2], [1]], 0),
        )
        # Waiting for a held PR it conflicts with keeps a PR out of the lane's way too.
        rows = [
            row(
                1,
                blockers=[
                    "waits: landing it first would force held #9 to rebase or re-bless"
                ],
                static_ok=False,
            ),
            row(2),
        ]
        p = mergeplan.build(
            CFG, state(1, 2, _1=st["prs"]["1"], _2=st["prs"]["2"]), rows, matrix([1, 2])
        )
        self.assertEqual(
            ([c["members"] for c in p["abi"]["binary"]["claims"]], p["waves"]),
            ([[2], [1]], [[1, 2]]),
        )

    def test_a_top_priority_that_must_rebase_leads_its_lane(self):
        """A top priority that conflicts with develop still leads: not behind every clean PR, nor in the binary lane
        behind wave PRs, while its advice says "it lands first"."""
        st = state(1, 2, 3)
        st["develop_fixes"] = {"1": ["your priority: HIGH priority merge"]}
        bumps(st, (1, "binary", 6, "h1", [6]), (2, "binary", 7, "h2"))
        rows = [row(1, g3=FAIL), row(2, g2=PEND), row(3, g2=PEND)]
        p = mergeplan.build(CFG, st, rows, matrix([2, 3], dev=[1]))
        # #1 ends wave 1 after its rebase; #2, its rival in the binary lane, waits for wave 2.
        self.assertEqual((p["first_rebase"], p["waves"]), ([1], [[3], [2]]))
        self.assertEqual({r["n"]: r["plan_rank"] for r in rows}, {3: 0, 1: 1, 2: 2})
        self.assertEqual(
            [c["members"] for c in p["abi"]["binary"]["claims"]], [[1], [2]]
        )
        self.assertEqual(
            p["advice"]["1"],
            "rebase now: it ends wave 1 (conflicts with develop in x.py); first in the "
            "ROCKE_ABI_VERSION lane: renumber 6 to 7 = develop + 1 (and BINARY_ABI); "
            "keep develop's history line 6",
        )
        self.assertEqual(
            p["advice"]["2"],
            "after #1 (your priority) lands: renumber ROCKE_ABI_VERSION to develop + 1 "
            "(and BINARY_ABI)",
        )
        self.assertEqual(
            mergeplan.place(dict(st, merge_plan=p), 1), "wave 1, after its rebase"
        )


class TestAutoMerge(unittest.TestCase):
    def setUp(self):
        self.st = state(1, 2, 3, 4)
        self.st["develop_fixes"] = {"1": ["your priority: blocking other work"]}
        for k in ("1", "3"):
            self.st["prs"][k]["auto_merge"] = {"by": "bob", "at": now()}
        self.st["prs"]["3"]["abi_cache"] = {
            "head": head(3),
            "lanes": {"binary": {"to": 7}},
            "families": [],
        }
        self.st["abi_lanes"] = {
            "binary": {
                "develop": 6,
                "head": None,
                "groups": [
                    {"members": [4], "root": 4, "to": 7, "reserve": 7, "ready": False},
                    {"members": [3], "root": 3, "to": 7, "reserve": 8, "ready": False},
                ],
            }
        }
        M = matrix(
            [1, 2, 3, 4],
            {(1, 2): ["rk/a.py"], (2, 3): ["rk/b.py"], (3, 4): ["rk/abi.h"]},
        )
        rows = [
            row(1, g3=FAIL),
            row(2, g2=PEND),
            row(3, g1=["fail", f"our verdict {O}"]),
            row(4, g2=PEND),
        ]
        mergeplan.finish(CFG, self.st, mergeplan.build(CFG, self.st, rows, M))

    def test_a_top_priority_is_expected_first_others_alert_once(self):
        am = self.st["merge_plan"]["automerge"]
        self.assertFalse(am["1"]["alert"])
        self.assertTrue(
            am["1"]["note"].endswith(
                "forces rebases on #2 (R1); expected first (your priority)"
            ),
            am["1"],
        )
        self.assertTrue(am["3"]["alert"])
        # #4 is in wave 1 with #1 (no conflict with it), so it ranks ahead of #2.
        self.assertIn(
            "forces rebases on #4 (R1), #2 (R1); takes binary ABI 7 ahead of #4 in the binary lane",
            am["3"]["note"],
        )
        lines = mergeplan.automerge_lines(self.st)
        self.assertEqual(len(lines), 1)
        self.assertTrue(
            lines[0].startswith(
                "AUTOMERGE_JUMP #3: auto-merge on (bob): may land as soon as GitHub allows"
            )
        )
        self.assertEqual(
            mergeplan.automerge_lines(self.st), []
        )  # once per head and finding
        self.st["merge_plan"]["automerge"]["3"]["key"] = "pushed"
        self.assertEqual(len(mergeplan.automerge_lines(self.st)), 1)


class TestQueue(unittest.TestCase):
    """build_queue with a plan: its order, its head wave as the only batch source, only conflicts hold, and the lane
    head is the lane member in the earliest wave."""

    def test_order_batches_holds_and_lanes(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            str(n): rec(
                n,
                review_decision="APPROVED",
                files_cache={
                    "head": head(n),
                    "hot": ["RG"],
                    "areas": ["platform"],
                    "files": [],
                },
            )
            for n in (1, 2, 3, 4)
        }
        st["holds"] = {
            "4": {
                "head": head(4),
                "status": "held",
                "reason": "G5 golden",
                "since": now(),
            }
        }
        claim = {"from": 6, "to": 7, "history_line": True, "dropped_history": []}

        def claims(cfg, r, ref, develop_ref=None):
            lanes = (
                {"binary": dict(claim, hunk=f"h{r['n']}")} if r["n"] in (1, 3) else {}
            )
            r["abi_cache"] = {
                "head": r["head"],
                "lanes": lanes,
                "families": [],
                "notes": [],
            }
            return r["abi_cache"]

        # #1 and #2 conflict, #3 conflicts with held #4; #2 shares only a golden shard with #4 (care): it does not wait.
        M = matrix([1, 2, 3, 4], {(1, 2): ["rk/a.py"], (3, 4): ["rk/b.py"]})
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: M
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, r: f"refs/remotes/origin/pr{r['n']}"
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue.abi, "develop_counters", lambda cfg: {"binary": 6, "recipe": 2}
        ), mock.patch.object(
            queue.abi, "pr_claims", claims
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        mp = st["merge_plan"]
        # Held #4 keeps its place, a priority, and #1 is in wave 1 with it: no conflict with #4, and the better of
        # {#1, #2} (the older PR). #3 waits for the held PR, so its ABI claim orders nothing: {#2, #3} is wave 2.
        self.assertEqual(mp["waves"], [[4, 1], [2, 3]])
        self.assertEqual(
            [(r["n"], r["position"]) for r in q if r.get("position")], [(1, 1), (2, 2)]
        )
        self.assertTrue(
            any(
                b.startswith("waits:") and "held #4" in b
                for b in next(r for r in q if r["n"] == 3)["blockers"]
            )
        )
        self.assertEqual(
            [b["members"] for b in st["batches"]], [[1]]
        )  # #2 waits for #1
        self.assertEqual(
            st["pair_cache"][f"{head(1)[:12]}:{head(2)[:12]}"],
            {"conflicts": ["rk/a.py"], "undecided": False},
        )
        lane = st["abi_lanes"]["binary"]
        self.assertEqual(
            ([g["members"] for g in lane["groups"]], lane["head"]), ([[1], [3]], 0)
        )
        self.assertEqual(next(r for r in q if r["n"] == 1)["gates"]["G9"][0], "pass")
        self.assertTrue(report.plan_row(CFG, st, mp)[0].startswith("wave 1: #4 #1"))

    def test_a_ready_pr_behind_a_top_priority_is_not_batched_and_the_board_says_why(
        self,
    ):
        """A top priority goes in first: a ready PR it conflicts with waits for it."""
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["develop_fixes"] = {"1": ["your priority: blocking other work"]}
        st["prs"] = {
            "1": rec(1, review_decision="APPROVED", ci_rollup="FAILURE"),
            "2": rec(2, review_decision="APPROVED"),
        }
        M = matrix([1, 2], {(1, 2): ["rk/a.py"]})
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: M
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue, "abi_lanes", lambda *a: {}
        ), mock.patch.object(
            queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        self.assertEqual(
            ([r["n"] for r in q if r.get("position")], st["batches"]), ([2], [])
        )
        self.assertEqual(
            st["merge_plan"]["advice"]["2"],
            "wait: #1 (your priority) lands first (conflicts in a.py), then rebase",
        )
        with tempfile.TemporaryDirectory() as d:
            md = report.board(CFG, st, Store(pathlib.Path(d)), [], q)["md"]
        self.assertIn(
            "| Merge queue | no batch: #2 wait for the PRs ahead of them in the merge plan |",
            md,
        )

    def test_abi_claims_read_this_tick_order_the_plan(self):
        """The claims are read before the plan (queue.refresh_claims), so a lane's two claims never share a wave, even
        on the tick that first sees them; the lane follows the plan, and only its first
        claim may land now. The pair cache, which batches and holds read, stays textual.
        """
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {str(n): rec(n, review_decision="APPROVED") for n in (1, 2)}

        def claims(cfg, r, ref, develop_ref=None):
            r["abi_cache"] = {
                "head": r["head"],
                "lanes": {
                    "binary": {
                        "from": 6,
                        "to": 7,
                        "history_line": True,
                        "dropped_history": [],
                        "hunk": f"h{r['n']}",
                    }
                },
                "families": [],
                "notes": [],
            }
            return r["abi_cache"]

        M = matrix([1, 2])  # git merges the two cleanly
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: M
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, r: f"refs/remotes/origin/pr{r['n']}"
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue.abi, "develop_counters", lambda cfg: {"binary": 6, "recipe": 2}
        ), mock.patch.object(
            queue.abi, "pr_claims", claims
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        mp = st["merge_plan"]
        self.assertEqual(
            (mp["waves"], mp["edges"]), ([[1], [2]], {"1-2": {"abi": ["binary"]}})
        )
        self.assertEqual(
            (
                [(r["n"], r["position"]) for r in q if r.get("position")],
                st["batches"][0]["members"],
            ),
            ([(1, 1)], [1]),
        )
        lane = st["abi_lanes"]["binary"]
        self.assertEqual(
            ([g["members"] for g in lane["groups"]], lane["head"]), ([[1], [2]], 0)
        )
        self.assertIn(
            "lane position 2 (behind #1)",
            next(r for r in q if r["n"] == 2)["gates"]["G9"][1],
        )
        self.assertEqual(
            mp["advice"]["2"],
            "after #1 lands: renumber ROCKE_ABI_VERSION to develop + 1 (and BINARY_ABI)",
        )
        self.assertEqual(
            st["pair_cache"],
            {f"{head(1)[:12]}:{head(2)[:12]}": {"conflicts": [], "undecided": False}},
        )
        self.assertIn(
            "MERGE PLAN: wave 1 #1; 2 waves; ABI in turn: binary ×2",
            report.tick_summary(CFG, st, [], [], q),
        )

    def test_a_ready_pr_in_wave_1_with_a_priority_is_batched_and_leads_its_lane(self):
        """A PR in wave 1 may land now, so the batches and the ABI lanes take it: #3, clear of the priority, rather than
        #2 (older; their ABI claims conflict), which waits for the priority."""
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["develop_fixes"] = {"1": PRIO}
        st["prs"] = {
            "1": rec(1, review_decision="APPROVED", ci_rollup="FAILURE"),
            "2": rec(2, review_decision="APPROVED"),
            "3": rec(3, review_decision="APPROVED"),
        }

        def claims(cfg, r, ref, develop_ref=None):
            lanes = (
                {
                    "binary": {
                        "from": 6,
                        "to": 7,
                        "history_line": True,
                        "dropped_history": [],
                        "hunk": f"h{r['n']}",
                    }
                }
                if r["n"] in (2, 3)
                else {}
            )
            r["abi_cache"] = {
                "head": r["head"],
                "lanes": lanes,
                "families": [],
                "notes": [],
            }
            return r["abi_cache"]

        M = matrix(
            [1, 2, 3], {(1, 2): ["rk/a.py"]}
        )  # git merges #2 and #3 cleanly; both bump ROCKE_ABI_VERSION
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: M
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, r: f"refs/remotes/origin/pr{r['n']}"
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue.abi, "develop_counters", lambda cfg: {"binary": 6, "recipe": 2}
        ), mock.patch.object(
            queue.abi, "pr_claims", claims
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        mp = st["merge_plan"]
        self.assertEqual(mp["waves"], [[1, 3], [2]])
        self.assertEqual(
            (
                [(r["n"], r["position"]) for r in q if r.get("position")],
                [b["members"] for b in st["batches"]],
            ),
            ([(3, 1)], [[3]]),
        )
        lane = st["abi_lanes"]["binary"]
        self.assertEqual(
            ([g["members"] for g in lane["groups"]], lane["head"]), ([[3], [2]], 0)
        )
        self.assertEqual(
            mp["advice"]["2"],
            "wait: #1 (your priority) lands first (conflicts in a.py), then rebase; "
            "after #3 lands: renumber ROCKE_ABI_VERSION to "
            "develop + 1 (and BINARY_ABI)",
        )

    def test_no_develop_merge_falls_back_to_the_head_to_head_check(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {"1": rec(1, review_decision="APPROVED")}
        with mock.patch.object(
            mergeplan, "matrix", lambda cfg, st, rows, dev: None
        ), mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue, "abi_lanes", lambda *a: {}
        ), mock.patch.object(
            queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        self.assertEqual(q[0]["position"], 1)
        self.assertEqual([b["members"] for b in st["batches"]], [[1]])
        self.assertIn(
            "no near-merge PR could be merged with develop",
            st["merge_plan"]["unavailable"],
        )
        self.assertTrue(
            report.plan_row(CFG, st, st["merge_plan"])[0].startswith(
                "not computed this tick"
            )
        )


class TestLanes(unittest.TestCase):
    def test_a_top_priority_leads_the_not_ready_lane_members(self):
        """A tier-0 PR whose CI an infra failure turned red stays ahead of tier-2 PRs with fewer failing gates in the
        binary lane: tier comes before the blocker count; a ready member still leads (the readiness rule).
        """
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        recs = {n: rec(n) for n in (1, 2, 3)}

        def claims(cfg, r, ref, develop_ref=None):
            r["abi_cache"] = {
                "head": r["head"],
                "lanes": {
                    "binary": {
                        "from": 6,
                        "to": 7,
                        "history_line": True,
                        "dropped_history": [],
                        "hunk": f"h{r['n']}",
                    }
                },
                "families": [],
                "notes": [],
            }
            return r["abi_cache"]

        rows = [
            {
                "n": 1,
                "static_ok": False,
                "blockers": ["G1: x", "G3: CI FAILURE"],
                "tier": 0,
                "disruption": 0,
                "size": 9000,
            },
            {
                "n": 2,
                "static_ok": False,
                "blockers": ["G2: approvals"],
                "tier": 2,
                "disruption": 0,
                "size": 10,
            },
            {
                "n": 3,
                "static_ok": True,
                "blockers": [],
                "tier": 2,
                "disruption": 0,
                "size": 10,
            },
        ]
        with mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, r: "ref"
        ), mock.patch.object(
            queue.abi, "develop_counters", lambda cfg: {"binary": 6, "recipe": 2}
        ), mock.patch.object(
            queue.abi, "pr_claims", claims
        ):
            queue.refresh_claims(
                CFG, st, list(recs.values()), "d" * 40
            )  # before the plan, in build_queue
            lanes = queue.abi_lanes(CFG, st, rows, list(recs.values()), "d" * 40)
        self.assertEqual(
            [g["members"] for g in lanes["binary"]["groups"]], [[3], [1], [2]]
        )
        self.assertEqual(lanes["binary"]["head"], 0)


class TestRenders(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.cfg = dict(
            CFG,
            paths={
                "working_dir": self.d.name,
                "repo": "/nonexistent",
                "html": str(pathlib.Path(self.d.name) / "x.html"),
                "daily_dir": self.d.name,
            },
        )
        st = state(1, 2, 3, 4, 5, _2={"title": "<img src=x onerror=alert(1)>"})
        st["enabled"] = True
        st["codeowners"] = {"teams": {"plat": ["me"]}}
        st["develop_fixes"] = {"1": ["your priority: blocking other work"]}
        M = matrix([1, 2, 3, 4], {(1, 2): ["rk/platform/x.py"], (3, 4): None}, dev=[5])
        M["facts"] = {
            n: {
                "rg": {"conv"} if n in (2, 3) else set(),
                "dg": set(),
                "abi": set(),
                "cmk": set(),
                "cat": set(),
                "idx": {},
            }
            for n in (1, 2, 3, 4)
        }
        rows = [
            row(1, g3=FAIL),
            row(2, g2=PEND),
            row(3, g2=PEND),
            row(4, g2=PEND),
            row(5, g2=PEND),
        ]
        mergeplan.finish(self.cfg, st, mergeplan.build(self.cfg, st, rows, M))
        self.st = st

    def tearDown(self):
        self.d.cleanup()

    def test_board_row_needs_and_fingerprint(self):
        now_s, next_s = report.plan_row(self.cfg, self.st, self.st["merge_plan"])
        # #3 has no conflict with #1: wave 1 with it (of #3 and #4, whose pair is not checked yet, the older). #2 and #4
        # wait for them: wave 2. No list of the priorities: the table has them.
        self.assertEqual(now_s, "wave 1: #1 #3 (needs: me ×2)")
        self.assertEqual(
            next_s, "2 waves; 1 rebase first; partial: 1 pairs, 0 PRs not checked yet"
        )
        b = report.board(self.cfg, self.st, self.store, [], [])
        self.assertIn("| Merge plan | wave 1: #1 #3", b["md"])
        same = report.board(self.cfg, self.st, self.store, [], [])
        self.assertEqual(b["fp"], same["fp"])
        self.st["merge_plan"]["fp"] = "changed"
        self.assertNotEqual(
            b["fp"], report.board(self.cfg, self.st, self.store, [], [])["fp"]
        )
        self.assertIn(
            "MERGE PLAN: wave 1 #1 #3; 2 waves; 1 rebase first; partial",
            report.tick_summary(self.cfg, self.st, [], [], []),
        )

    def test_approvals_follow_the_plan(self):
        """The "Ready for your approval" list is sorted by wave."""
        self.st["prs"]["4"]["ours"][
            "at"
        ] = "2026-11-03T00:00:00Z"  # the oldest verdict: it led before the plan
        appr = report.approval_candidates(self.cfg, self.st)
        # The develop fix first; in wave 2, the older verdict first.
        self.assertEqual(
            [(a["n"], a["place"]) for a in appr],
            [
                (1, "wave 1"),
                (3, "wave 1"),
                (4, "wave 2"),
                (2, "wave 2"),
                (5, "after its rebase"),
            ],
        )
        self.assertTrue(report.approval_line(appr[1]).endswith("· merge plan: wave 1"))
        self.assertTrue(report.approval_line(appr[2]).endswith("· merge plan: wave 2"))

    def test_md_section_and_html_card(self):
        md = "\n".join(report.plan_md(self.cfg, self.st))
        # One row per wave, each PR with its class: #2 waits for #1, #4 for #3 (not checked yet).
        self.assertIn("| wave 1 | #1 T0 ✓, #3 R1 ✓ | #2 (R1), #4 (R1) | me ×2 |", md)
        self.assertIn("| wave 2 | #2 R1, #4 R1 | none |", md)
        self.assertNotIn("| first |", md)
        self.assertIn(
            "| #5 | R1 | after its rebase | rebase now: you're next (conflicts with develop in x.py)",
            md,
        )
        self.assertIn("golden shard conv", md)
        page = webdash.render(
            self.cfg,
            self.st,
            self.store,
            [],
            [],
            report.board(self.cfg, self.st, self.store, [], []),
        )
        card = webdash.plan_card(self.cfg, self.st)
        self.assertIn(card, page)
        # No list of the priorities above the table, and no "first" row.
        self.assertNotIn("<b>First:</b>", card)
        self.assertNotIn("<td class='nw'>first</td>", card)
        self.assertIn(
            "<tr><td class='nw'>wave 1</td><td><div class='chips'><a class='chip chip-prio'",
            card,
        )
        # The operator's priorities wear a green border wherever their chip shows.
        self.assertIn(
            "<a class='chip chip-prio' href='https://github.com/ROCm/r/pull/1'", card
        )
        self.assertIn("<a class='chip' href='https://github.com/ROCm/r/pull/3'", card)
        self.assertIn("<a href='#plan'>Merge plan", page)
        self.assertIn(
            "</h2><form id='prio' class='prio'", card
        )  # the "Request PR priority" box opens it
        self.assertIn("<table class='grid' aria-label='Conflict grid'>", card)
        self.assertIn("<td class='g fail' tabindex='0' data-tip=", card)  # ✗ #1 × #2
        self.assertIn(">✗</td>", card)
        self.assertIn(">?</td>", card)  # #3 × #4 not checked yet
        self.assertIn(">◆</td>", card)  # #2 × #3 share a golden shard
        self.assertNotIn("<img src=x", card)  # titles are untrusted
        self.st["merge_plan"] = {
            "unavailable": "no near-merge PR could be merged with develop here"
        }
        self.assertIn("Not computed this tick", webdash.plan_card(self.cfg, self.st))

    def test_a_top_priority_that_needs_a_rebase_ends_wave_1(self):
        """A top priority that conflicts with develop does not sink into "after a rebase" with everyone else: it
        ends wave 1."""
        st = state(1, 2, 3, 5)
        st["develop_fixes"] = {"1": ["your priority: blocking other work"]}
        M = matrix([2, 3], {(2, 3): ["rk/platform/x.py"]}, dev=[1, 5])
        mergeplan.finish(
            self.cfg,
            st,
            mergeplan.build(self.cfg, st, [row(n, g2=PEND) for n in (1, 2, 3, 5)], M),
        )
        self.assertEqual(
            (st["merge_plan"]["waves"], st["merge_plan"]["first_rebase"]),
            ([[2], [3]], [1]),
        )
        md = "\n".join(report.plan_md(self.cfg, st))
        self.assertIn("| wave 1 | #2 R1 ✓, #1 T0, after its rebase | #3 (R1) |", md)
        self.assertIn(
            "| after a rebase | #5 R1 | — |", md
        )  # the top priority is not repeated there
        card = webdash.plan_card(self.cfg, st)
        self.assertIn("<tr><td class='nw'>wave 1</td>", card)
        self.assertIn("#1 T0, after its rebase</a>", card)
        self.assertEqual(card.count("<td class='nw'>after a rebase</td>"), 1)
        self.assertNotIn(
            "#1 T0",
            card.split("<td class='nw'>after a rebase</td>")[1].split("</tr>")[0],
        )
        self.assertTrue(
            report.plan_row(self.cfg, st, st["merge_plan"])[0].startswith(
                "wave 1: #2, then #1 after its rebase (needs: "
            )
        )

    def test_a_plan_stored_before_the_waves_still_renders(self):
        """A plan stored before 2026-11-05 (its "first" group, "joined", nodes in wave 0) is read for at most one tick:
        nothing crashes on it."""
        st = state(1, 2, 3)
        st["codeowners"] = {"teams": {"plat": ["me"]}}
        st["merge_plan"] = old = {
            "v": 1,
            "at": now(),
            "develop": "d" * 40,
            "scope": [1, 2, 3],
            "first": [1, 2],
            "joined": [2],
            "first_rebase": [],
            "why": {"1": "your priority"},
            "waves": [[3]],
            "hubs": [],
            "groups": {},
            "edges": {},
            "nodes": {
                str(n): {
                    "cls": "R1",
                    "w": 4,
                    "wave": w,
                    "rank": i,
                    "go": w < 2,
                    "conflicts": [],
                    "abi": [],
                    "care": [],
                    "forces": [],
                    "cost": 0,
                    "care_cost": 0,
                    "hub": False,
                    "hub_degree": 0,
                }
                for i, (n, w) in enumerate(((1, 0), (2, 0), (3, 1)))
            },
            "advice": {
                "1": "first (your priority): no rebase needed",
                "2": "first: no rebase needed",
                "3": "x",
            },
            "dev_conflicts": {},
            "unknown": [],
            "partial": None,
            "merges": 0,
            "secs": 0.0,
            "held_back": {},
            "jump_told": {},
            "abi": {},
            "automerge": {},
        }
        self.assertTrue(report.plan_row(self.cfg, st, old)[0].startswith("wave 1: #3"))
        self.assertIn(
            "| wave 1 | #3 R1 ✓ | none |", "\n".join(report.plan_md(self.cfg, st))
        )
        self.assertIn("<td class='nw'>wave 1</td>", webdash.plan_card(self.cfg, st))
        self.assertEqual(mergeplan.summary(old), "wave 1 #3; 1 wave")
        self.assertEqual([mergeplan.place(st, n) for n in (3, 9)], ["wave 1", ""])
        self.assertEqual(
            [a["n"] for a in report.approval_candidates(self.cfg, st)], [1, 2, 3]
        )

    def test_abi_marks_lanes_and_the_collapsed_rebase_advice(self):
        """The ABI indicator is a separate icon in the matrix, and the rebase advice is collapsible, in the merge plan
        box."""
        st = state(1, 2, 3)
        bumps(
            st, (1, "binary", 7, "h1"), (2, "binary", 7, "h2"), (3, "binary", 7, "h3")
        )
        M = matrix(
            [1, 2, 3], {(1, 2): ["rk/platform/x.py"]}
        )  # #1 and #2 also conflict in git; #3 only bumps
        mergeplan.finish(
            self.cfg,
            st,
            mergeplan.build(self.cfg, st, [row(n, g2=PEND) for n in (1, 2, 3)], M),
        )
        st["abi_lanes"] = {
            "binary": {
                "develop": 6,
                "head": None,
                "groups": [{"members": [n]} for n in (1, 2, 3)],
            }
        }
        card = webdash.plan_card(self.cfg, st)
        self.assertIn(
            "<td class='g fail two' tabindex='0' data-tip=", card
        )  # #1 × #2: both marks
        self.assertIn(">✗<span class='abi'>Ⓐ</span></td>", card)
        self.assertIn(
            "<td class='g abi' tabindex='0' data-tip=", card
        )  # #1 × #3: git merges them cleanly
        self.assertIn(
            "Ⓐ both bump ROCKE_ABI_VERSION: the later renumbers to develop + 1",
            html.unescape(card),
        )
        self.assertIn(
            "<b class='g abi'>Ⓐ</b> ABI: both bump ROCKE_ABI_VERSION", card
        )  # the legend
        self.assertIn("1 textual conflict, 3 ABI conflicts", card)
        self.assertIn(
            "binary lane, ROCKE_ABI_VERSION (develop 6): <a class='chip'", card
        )
        self.assertIn(
            "<details id='d-plan-advice'><summary>Rebase advice (3 PRs)</summary>", card
        )
        self.assertNotIn("d-plan-advice' open", card)  # collapsed by default
        md = "\n".join(report.plan_md(self.cfg, st, details=True))
        self.assertIn("1 textual conflicts, 3 ABI conflicts", md)
        self.assertIn("- binary, ROCKE_ABI_VERSION (develop 6): #1 → #2 → #3", md)
        self.assertIn("<details><summary>Rebase advice (3 PRs)</summary>", md)
        self.assertNotIn(
            "<details>", "\n".join(report.plan_md(self.cfg, st))
        )  # `rk merge-plan` stays plain text

    def test_md_dashboard_follows_the_page(self):
        """The md dashboard in the page's order: the approval lists, the Mergeable Candidates (the PRs that passed CI
        and can merge now, and their gates), then the merge plan; no ready queue, batch or ABI lane tables. A candidate
        whose CI still runs is not mergeable: counted under them, and listed under Not ready.
        """
        full = lambda n, **kw: dict(
            {
                "n": n,
                "author": "a",
                "size": 100,
                "hot": [],
                "blockers": [],
                "cleared": False,
                "gates": {k: ["pass", ""] for k in report.GATES},
            },
            **kw,
        )
        q = [
            full(2, position=1, stage="ready"),
            full(3, stage="candidate", blockers=["G3: CI running"]),
            full(4, stage=None, blockers=["G1: our verdict 🟠"]),
        ]
        self.st["batches"] = [
            {
                "members": [2],
                "status": "validated",
                "lines": 100,
                "check_prefixes": False,
            }
        ]
        self.st["prs"]["3"]["ci_rollup"] = "PENDING"
        board = report.board(self.cfg, self.st, self.store, [], q)
        md = report.dashboard(self.cfg, self.st, [], q, [], board["md"])
        heads = [
            "## Status board",
            "**Needs plat approval**",
            "## Mergeable Candidates (and their gates)",
            "## Merge plan",
            "## Not ready (first blocker)",
        ]
        self.assertEqual(
            [md.index(h) for h in heads], sorted(md.index(h) for h in heads)
        )
        self.assertIn("| 1 | ready · batch 1: validated | #2 | a | 100 | ✓ | ✓ |", md)
        # CI still runs: not mergeable
        self.assertNotIn("|  | candidate | #3 | a | 100 |", md)
        self.assertIn("+1 waiting on CI: #3 CI running", md)
        for gone in (
            "## Ready queue",
            "## Merge batches",
            "## ABI lanes",
            "Ready for your approval",
            "Close to merge",
        ):
            self.assertNotIn(gone, md)
        rest = md[md.index("## Not ready") :]
        self.assertIn("| #4 |", rest)
        self.assertIn("| #3 |", rest)  # a candidate whose CI runs is not ready yet
        self.assertIn("<details><summary>Rebase advice (5 PRs)</summary>", md)


if __name__ == "__main__":
    unittest.main()
