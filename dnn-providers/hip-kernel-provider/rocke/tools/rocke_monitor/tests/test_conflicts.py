# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge-conflict hot files: live conflicts of the PRs GitHub reports conflicting, the
conflicts authors resolved in develop merges, once each, within the per-tick bounds; the ranking, episodes, directories
and the develop PRs that last changed each file. Temp git repos, no network."""

import datetime as dt
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import conflicts, gitops  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

UTC = dt.timezone.utc
PAST, FUTURE = dt.datetime(2000, 1, 1, tzinfo=UTC), dt.datetime(2100, 1, 1, tzinfo=UTC)


def g(repo, *a, check=True):
    return subprocess.run(
        ["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", *a],
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


class TestCollect(unittest.TestCase):
    def commit(self, files, msg):
        for f, text in files.items():
            p = pathlib.Path(self.origin, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        g(self.origin, "add", "-A")
        g(self.origin, "commit", "-qm", msg)
        return g(self.origin, "rev-parse", "HEAD")

    def publish(self, n):
        g(self.origin, "update-ref", f"refs/pull/{n}/head", "HEAD")
        return g(self.origin, "rev-parse", "HEAD")

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.origin, self.mirror = f"{self.d.name}/origin", f"{self.d.name}/mirror"
        g(self.d.name, "init", "-q", "-b", "develop", "origin")
        self.commit(
            {"rk/a.txt": "1\n", "rk/b.txt": "1\n", "other/c.txt": "1\n"}, "base"
        )
        g(self.origin, "checkout", "-q", "-b", "p7")
        self.commit({"rk/a.txt": "pr7\n"}, "pr 7")
        self.h7 = self.publish(7)
        g(self.origin, "checkout", "-q", "-b", "p9", "develop")
        self.h9a = self.commit({"rk/b.txt": "pr9\n"}, "pr 9")
        g(self.origin, "checkout", "-q", "-b", "p11", "develop")
        self.commit({"other/c.txt": "pr11\n"}, "pr 11")
        g(self.origin, "checkout", "-q", "develop")
        self.commit(
            {"rk/a.txt": "dev\n", "rk/b.txt": "dev\n"}, "develop changes a and b (#101)"
        )
        # #9 merges develop and resolves the conflict on b; #11 merges develop cleanly
        g(self.origin, "checkout", "-q", "p9")
        g(self.origin, "merge", "-q", "develop", check=False)  # conflicts on b
        pathlib.Path(self.origin, "rk/b.txt").write_text("pr9 and dev\n")
        g(self.origin, "add", "-A")
        g(self.origin, "commit", "-qm", "Merge develop into p9")
        self.h9 = self.publish(9)
        g(self.origin, "checkout", "-q", "p11")
        g(self.origin, "merge", "-q", "--no-edit", "develop")
        self.h11 = self.publish(11)
        g(self.origin, "checkout", "-q", "develop")
        g(self.d.name, "init", "-q", "mirror")
        g(self.mirror, "remote", "add", "origin", self.origin)
        self.dev = gitops.fetch_develop(self.mirror)
        for n in (9, 11):
            gitops.fetch_pr(
                self.mirror, n
            )  # the queue fetches open heads; #7 is fetched by the collector
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.cfg = {
            "repo": "ROCm/r",
            "rocke_path": "rk",
            "paths": {"repo": self.mirror},
        }
        self.st = empty_state()
        self.st["develop"]["sha"] = self.dev
        rec = {
            "status": "open",
            "draft": False,
            "base": "develop",
            "mergeable": "MERGEABLE",
        }
        self.st["prs"] = {
            "7": dict(rec, n=7, head=self.h7, mergeable="CONFLICTING"),
            "9": dict(rec, n=9, head=self.h9),
            "11": dict(rec, n=11, head=self.h11, status="merged", gone_at=now()),
            "12": dict(rec, n=12, head=self.h7, draft=True, mergeable="CONFLICTING"),
        }

    def rows(self, source=None):
        return [
            r
            for r in self.store.read_jsonl(conflicts.FILE)
            if source in (None, r["source"])
        ]

    def test_live_and_resolved_once_each(self):
        out = conflicts.collect(self.cfg, self.st, self.store)
        self.assertEqual((out["live"], out["resolved"], out["errors"]), (1, 2, []))
        (live,) = self.rows("live")
        self.assertEqual(
            (live["pr"], live["head"], live["develop"], live["files"]),
            (7, self.h7[:12], self.dev[:12], ["rk/a.txt"]),
        )  # the draft #12 is left out
        res = {r["pr"]: r for r in self.rows("resolved")}
        self.assertEqual(
            (res[9]["files"], res[9]["parent"]), (["rk/b.txt"], self.h9a[:12])
        )
        self.assertEqual(
            res[11]["files"], []
        )  # a clean develop merge counts in the merges, not the files
        self.assertEqual(self.rows("start")[0]["ts"], self.st["conflicts"]["since"])
        self.assertEqual(
            self.st["conflicts"]["offenders"]["by"],
            {"rk/a.txt": [101], "rk/b.txt": [101]},
        )
        again = conflicts.collect(self.cfg, self.st, self.store)
        self.assertEqual((again["live"], again["resolved"], again["checks"]), (0, 0, 0))
        hot = conflicts.hot_files(self.store, PAST, FUTURE, rocke_path="rk")
        self.assertEqual(
            [(f["path"], f["prs"], f["episodes"]) for f in hot["files"]],
            [("rk/a.txt", [7], 1), ("rk/b.txt", [9], 1)],
        )
        self.assertEqual((hot["merges"], hot["merges_conflicted"]), (2, 1))

    def test_develop_moves_a_row_only_when_the_paths_change_or_it_is_due(self):
        conflicts.collect(self.cfg, self.st, self.store)
        self.commit({"other/d.txt": "1\n"}, "develop moves elsewhere (#102)")
        self.st["develop"]["sha"] = gitops.fetch_develop(self.mirror)
        out = conflicts.collect(self.cfg, self.st, self.store)
        self.assertEqual(
            (out["checks"], out["live"]), (1, 0)
        )  # checked again, nothing new to say
        self.commit({"other/e.txt": "1\n"}, "and again (#103)")
        self.st["develop"]["sha"] = gitops.fetch_develop(self.mirror)
        out = conflicts.collect(
            dict(self.cfg, conflicts={"refresh_h": 0}), self.st, self.store
        )
        self.assertEqual(
            out["live"], 1
        )  # due: the PR shows in every window while it keeps conflicting
        self.assertEqual(len(self.rows("live")), 2)

    def test_the_bounds_leave_work_for_the_next_tick(self):
        out = conflicts.collect(
            dict(self.cfg, conflicts={"max_checks": 1}), self.st, self.store
        )
        self.assertEqual((out["checks"], out["live"], out["resolved"]), (1, 1, 0))
        self.assertGreater(out["deferred"], 0)
        out = conflicts.collect(self.cfg, self.st, self.store)
        self.assertEqual((out["live"], out["resolved"]), (0, 2))
        out = conflicts.collect(
            dict(self.cfg, conflicts={"max_fetches": 0}),
            dict(self.st, conflicts={}),
            Store(pathlib.Path(self.d.name) / "s2"),
        )
        self.assertEqual(
            out["live"], 1
        )  # #7 was fetched the first time; nothing to fetch now

    def test_a_failing_pr_is_reported_and_the_others_go_on(self):
        self.st["prs"]["8"] = {
            "n": 8,
            "head": "8" * 40,
            "status": "open",
            "draft": False,
            "base": "develop",
            "mergeable": "CONFLICTING",
        }  # GitHub has no refs/pull/8
        out = conflicts.collect(self.cfg, self.st, self.store)
        self.assertEqual(out["live"], 1)
        self.assertTrue(out["errors"][0].startswith("#8: "))


class TestHotFiles(unittest.TestCase):
    R = "team/provider/rocke"

    def test_ranking_episodes_and_directories(self):
        golden = f"{self.R}/platform/tests/golden/rocke_representative_ir_sha256.json"
        dg = f"{self.R}/platform/tests/instances/differential/golden/llvm_gfx_all.json"
        abi = f"{self.R}/platform/cpp/include/rocke/abi.h"
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            store.append_jsonl(
                conflicts.FILE,
                [
                    {
                        "ts": "2026-10-27T00:00:00Z",
                        "source": "start",
                        "scan_from": "2026-10-19T00:00:00Z",
                    },
                    {
                        "ts": "2026-10-27T01:00:00Z",
                        "source": "live",
                        "pr": 1,
                        "head": "h1",
                        "develop": "d1",
                        "files": [golden, dg],
                    },
                    {
                        "ts": "2026-10-27T05:00:00Z",
                        "source": "live",
                        "pr": 1,
                        "head": "h1",
                        "develop": "d2",
                        "files": [golden, dg],
                    },  # the same head as develop moves: one episode
                    # #1 merges develop on top of h1: the episode the live source saw, not a new one
                    {
                        "ts": "2026-10-27T06:00:00Z",
                        "source": "resolved",
                        "pr": 1,
                        "merge": "m1",
                        "parent": "h1",
                        "develop": "d2",
                        "files": [golden],
                    },
                    {
                        "ts": "2026-10-28T00:00:00Z",
                        "source": "resolved",
                        "pr": 2,
                        "merge": "m2",
                        "parent": "h2",
                        "develop": "d3",
                        "files": [abi],
                    },
                    {
                        "ts": "2026-10-29T00:00:00Z",
                        "source": "resolved",
                        "pr": 2,
                        "merge": "m3",
                        "parent": "h3",
                        "develop": "d4",
                        "files": [abi],
                    },
                    {
                        "ts": "2026-10-29T01:00:00Z",
                        "source": "resolved",
                        "pr": 3,
                        "merge": "m4",
                        "parent": "h4",
                        "develop": "d4",
                        "files": [dg],
                    },
                    {
                        "ts": "2026-10-29T02:00:00Z",
                        "source": "resolved",
                        "pr": 4,
                        "merge": "m5",
                        "parent": "h5",
                        "develop": "d4",
                        "files": [],
                    },
                    {
                        "ts": "2026-11-03T00:00:00Z",
                        "source": "live",
                        "pr": 9,
                        "head": "h9",
                        "develop": "d9",
                        "files": [abi],
                    },  # after the window
                ],
            )
            store.append_jsonl(conflicts.FILE, [{"broken": True}])
            with (store.dir / conflicts.FILE).open("a") as f:
                f.write(
                    '{"ts": "2026-10-2\n'
                )  # a write cut short is skipped, not fatal
            hot = conflicts.hot_files(
                store,
                dt.datetime(2026, 10, 26, tzinfo=UTC),
                dt.datetime(2026, 11, 2, tzinfo=UTC),
                rocke_path=self.R,
            )
        self.assertEqual(
            [(f["short"], f["prs"], f["episodes"]) for f in hot["files"]],
            [
                (
                    "rocke/platform/tests/instances/differential/golden/llvm_gfx_all.json",
                    [1, 3],
                    2,
                ),
                (
                    "rocke/platform/cpp/include/rocke/abi.h",
                    [2],
                    2,
                ),  # one PR, two merges: ties broken by episodes
                (
                    "rocke/platform/tests/golden/rocke_representative_ir_sha256.json",
                    [1],
                    1,
                ),
            ],
        )
        self.assertEqual(
            [(d["dir"], d["prs"], d["files"], d["episodes"]) for d in hot["dirs"]],
            [
                ("rocke/platform/tests/instances/differential", 2, 1, 2),
                ("rocke/platform/cpp/include/rocke", 1, 1, 2),
                ("rocke/platform/tests/golden", 1, 1, 1),
            ],
        )
        self.assertEqual(
            (hot["merges"], hot["merges_conflicted"], hot["prs"]), (5, 4, [1, 2, 3])
        )
        self.assertEqual(
            conflicts.tracked(hot),
            "tracked since 2026-10-27 00:00Z; develop merges on PR branches since 2026-10-19",
        )
        text = "\n".join(conflicts.render_md(dict(hot, offenders={dg: [1234, 1111]})))
        self.assertIn(
            "| 1 | `rocke/platform/tests/instances/differential/golden/llvm_gfx_all.json` | 2 | 2 | #1, #3 | "
            "#1234, #1111 |",
            text,
        )
        self.assertIn("| `rocke/platform/cpp/include/rocke/` | 1 | 1 | 2 |", text)
        self.assertEqual(conflicts.area("CMakeLists.txt", self.R), "(root)")
        self.assertEqual(
            conflicts.area("projects/x/y/z/w/v.cpp", self.R), "projects/x/y/z"
        )

    def test_a_file_in_the_folder_shows_from_the_folder_s_name(self):
        """A file in the folder the configuration names shows from that folder's name on, whatever the folder; a file
        outside it in full."""
        folder = "src/kernels"
        self.assertEqual(
            conflicts.short("src/kernels/core/a.cpp", folder), "kernels/core/a.cpp"
        )
        self.assertEqual(
            conflicts.area("src/kernels/core/x/y/z/a.cpp", folder),
            "kernels/core/x/y/z",
        )
        self.assertEqual(conflicts.short("src/other/a.cpp", folder), "src/other/a.cpp")
        self.assertEqual(
            conflicts.within("src/kernels/core/a.cpp", folder), "core/a.cpp"
        )
        self.assertEqual(conflicts.within("src/other/a.cpp", folder), "src/other/a.cpp")
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            store.append_jsonl(
                conflicts.FILE,
                [
                    {
                        "ts": "2026-10-27T01:00:00Z",
                        "source": "live",
                        "pr": 5,
                        "head": "h5",
                        "develop": "d1",
                        "files": ["src/kernels/core/a.cpp"],
                    }
                ],
            )
            cfg = {"rocke_path": "/src/kernels/", "paths": {"repo": ""}}
            with mock.patch.object(conflicts, "offenders", return_value={}):
                hot = conflicts.weekly(
                    cfg,
                    {},
                    store,
                    dt.datetime(2026, 10, 26, tzinfo=UTC),
                    dt.datetime(2026, 11, 2, tzinfo=UTC),
                )
        self.assertEqual([f["short"] for f in hot["files"]], ["kernels/core/a.cpp"])

    def test_which_prs_count_as_conflicting(self):
        rec = {
            "status": "open",
            "draft": False,
            "base": "develop",
            "mergeable": "CONFLICTING",
        }
        self.assertTrue(conflicts.conflicting(rec))
        # GitHub recomputes after develop moves: UNKNOWN keeps the last definite answer
        self.assertTrue(
            conflicts.conflicting(
                dict(rec, mergeable="UNKNOWN", mergeable_known="CONFLICTING")
            )
        )
        self.assertFalse(
            conflicts.conflicting(
                dict(rec, mergeable="UNKNOWN", mergeable_known="MERGEABLE")
            )
        )
        self.assertFalse(conflicts.conflicting(dict(rec, draft=True)))
        self.assertFalse(
            conflicts.conflicting(dict(rec, base="users/x/stack"))
        )  # conflicts with its base, not develop
        self.assertFalse(conflicts.conflicting(dict(rec, status="merged")))


class TestTickStepAndCli(unittest.TestCase):
    def test_a_failing_collector_never_fails_the_tick(self):
        import rkmon
        from rkmon_lib import ci_times, gh

        def boom(*a, **kw):
            raise RuntimeError("boom")

        def down(*a, **kw):
            raise gh.GhError("gh api: HTTP 502")

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            with mock.patch.object(conflicts, "collect", boom), mock.patch.object(
                ci_times, "refresh", down
            ):
                lines = rkmon.insight_steps({}, empty_state(), store)
            self.assertEqual(
                lines,
                [
                    "(conflicts not recorded: boom)",
                    "(ci not recorded: gh api: HTTP 502)",
                ],
            )
            self.assertIn("conflicts", rkmon.TIMING)  # on the tick's "time:" line
            self.assertEqual(
                rkmon.insight_steps({"repo": "ROCm/r"}, empty_state(), store), []
            )  # nothing to do: quiet

    def test_the_cli_prints_the_tables(self):
        import rkmon
        from rkmon_lib import ci_times

        out = []
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            store.append_jsonl(
                conflicts.FILE,
                [
                    {
                        "ts": now(),
                        "source": "live",
                        "pr": 5,
                        "head": "h",
                        "develop": "d",
                        "files": ["x/y.txt"],
                    }
                ],
            )
            cfg = {
                "repo": "ROCm/r",
                "rocke_path": "rk",
                "paths": {"repo": f"{d}/nonexistent"},
            }
            got = {"recorded": 0, "calls": 0, "pending": 0, "left": 0, "error": None}
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(rkmon, "Store", lambda: store), mock.patch.object(
                ci_times, "collect", lambda *a, **kw: got
            ), mock.patch(
                "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
            ):
                self.assertEqual(rkmon.main(["conflict-files", "--days", "3"]), 0)
                self.assertEqual(
                    rkmon.main(["ci-times"]), 0
                )  # nothing recorded: reads first (capped)
        text = "\n".join(out)
        self.assertIn("## Merge-conflict hot files, last 3 days", text)
        self.assertIn("| 1 | `x/y.txt` | 1 | 1 | #5 | — |", text)
        self.assertIn("(develop's last changes not read", text)
        self.assertIn(
            "(read 0 heads in 0 API calls; 0 still running, 0 left for the next pass)",
            text,
        )
        self.assertIn("## CI time per PR, heads pushed in the last 7 days", text)


if __name__ == "__main__":
    unittest.main()
