# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Insights from recent PRs: findings by cause, fixes matched back to their finding,
open develop issues by cause, the insights report's tables, the collapsed dashboard card, and where the time goes per
PR."""

import datetime as dt
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

from rkmon_lib import daylog, insights, report, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

R, O = "\U0001f534", "\U0001f7e0"


class TestInsights(unittest.TestCase):
    def test_a_cause_comes_from_the_title_and_impact_not_the_file_list(self):
        cases = {
            f"N1 {R} `golden/x.md`: **Absolute TFLOP/s figures in §7**": "public",
            f"N4 {R} `t.py:56`: **The new test asks for fork, which Windows does not have**": "windows",
            f"N1 {O} `differential/golden/llvm_gfx_all.json`: **The PR bundles #1111's change**": "scope",
            f"N1 {O} `golden/rocke_representative_ir_sha256.json:2`: **Edits the golden that #2345 split**": "golden",
            f"N1 {O} `abi.h:90`: **#1234 already took binary ABI 4**": "abi",
            f"N1 {O} `ir.h:476`: **The rebase conflicts with #2222**": "skew",
            f"N1 {O} `mem.cpp:1`: **Fixed in Python only; the C++ engine keeps the old count**": "parity",
            f"N1 {O} `CMakeLists.txt:674`: **The selector test never runs: CTest does not collect it**": "coverage",
            f"N1 {R} `x.cpp:145`: **The HipDNN Linux Superbuild fails at Build**": "ci",
            f"N2 {O} `abi.py:155`: **The fp6 word index overflows i32**": "correctness",
            f"N1 {O} `a.py:1`: **The catalog swap is half done**": "other",
            "The PR no longer merges into develop, so no CI can run on it.": "skew",
            # older unbolded posts: the location ends at the first ": " after a backtick, ")" or quote
            f"N1 {O} `sdpa/bind.py:173`; `common.py:626`, `:776`: Outside `_ARCHES` the layer builds every grid": "correctness",
            f"N1 {R} commits `9b44078b` (`case_study.md`): `9b44078b` wrote an absolute kernel timing": "public",
            # nothing in the title or impact: the file list decides
            f"N1 {R} `lower_llvm.py:1271`; `differential/golden/llvm_gfx_all.json`: **Problem:** the fix also lowers "
            f"the align of every vector load": "golden",
        }
        for text, key in cases.items():
            self.assertEqual(insights.classify(text)[0], key, text)
        self.assertEqual(
            insights._title(
                f"N1 {O} `a.py:1`, `:7`: Outside `_ARCHES` the layer builds every grid"
            ),
            "Outside `_ARCHES` the layer builds every grid",
        )
        self.assertEqual(
            insights._title(
                f"N1 {R} `x.py:2`: **Problem:** the cache thrashes<br>• impact: slow"
            ),
            "the cache thrashes",
        )

    def test_findings_by_cause_with_fixes_matched_back(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {"paths": {"daily_dir": d}}
            daylog.save(
                store,
                cfg,
                5,
                "finding",
                f"N1 {O} `g.json:2`: **Edits the golden that #1 split**<br>• impact: x",
            )
            daylog.save(
                store,
                cfg,
                5,
                "finding",
                f"N2 {R} `w.py:3`: **Fails on Windows: cp1252**",
            )
            daylog.save(
                store,
                cfg,
                6,
                "finding",
                f"N1 {O} `h.json:1`: **Golden not re-blessed**",
            )
            daylog.save(
                store,
                cfg,
                5,
                "fixed",
                f"our {now()[5:10]} N1 confirmed fixed before merge",
            )
            daylog.save(
                store, cfg, 5, "verify", "fu: 3 qwen claims corrected"
            )  # not a finding
            fs = insights.findings(store)
        self.assertEqual(len(fs), 3)
        golden, windows = insights.by_cause(fs)
        self.assertEqual(
            (
                golden["key"],
                golden["n"],
                golden["orange"],
                golden["fixed"],
                golden["last7"],
            ),
            ("golden", 2, 2, 1, 2),
        )
        self.assertEqual(
            golden["prs"],
            {5: ["Edits the golden that #1 split"], 6: ["Golden not re-blessed"]},
        )
        self.assertEqual(
            (windows["key"], windows["red"], windows["fixed"]), ("windows", 1, 0)
        )

    def test_open_develop_issues_by_cause(self):
        st = empty_state()
        st["post_merge"] = {
            "open": [
                {
                    "n": 1,
                    "title": "[rocke] x times out intermittently on Windows gfx110X",
                    "rocke": True,
                    "kind": "stability",
                    "label": guard_labels.STABILITY,
                },
                {
                    "n": 2,
                    "title": "[rocke] y fails on Windows since #3",
                    "rocke": True,
                    "kind": "stability",
                    "label": guard_labels.STABILITY,
                },
                {
                    "n": 4,
                    "title": "[rocke] z fails on develop since #5",
                    "rocke": True,
                    "kind": "ci_red",
                    "label": guard_labels.RED,
                },
                {
                    "n": 6,
                    "title": "[develop] rocFFT smoketest times out",
                    "rocke": False,
                },
            ]
        }
        rows = insights.develop({}, st)
        self.assertEqual(
            [(r["key"], [i["n"] for i in r["issues"]]) for r in rows],
            [("broke", [4]), ("windows", [2]), ("timeout", [1])],
        )

    def test_report_tables_first_table_under_each_heading(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "rocke_insights_2026-10-30.md"
            p.write_text(
                "# T\n\n## The patterns\n\n| # | Pattern | Fix |\n|---|---|---|\n| 1 | **Skew** \\| merge | daily |\n"
                "\ntext\n\n| a | b |\n|---|---|\n| c | d |\n\n## Recs\n\n| Change | Impact |\n|:--|--:|\n| x | y |\n",
                encoding="utf-8",
            )
            (pathlib.Path(d) / "rocke_insights_2026-10-01.md").write_text(
                "## Old\n", encoding="utf-8"
            )
            self.assertEqual(insights.latest_report({"paths": {"insights_dir": d}}), p)
            t = insights.report_tables(p)
        self.assertEqual([h for h, _ in t], ["The patterns", "Recs"])
        self.assertEqual(
            t[0][1], [["#", "Pattern", "Fix"], ["1", "**Skew** | merge", "daily"]]
        )
        self.assertEqual(t[1][1], [["Change", "Impact"], ["x", "y"]])

    def test_the_dashboard_card_is_collapsed_and_escapes_findings(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {
                "deployment": {"phase": "queue"},
                "merge_authority": "arm-cleared",
                "operator": {"github_login": "me"},
                "repo": "ROCm/r",
                "paths": {
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                    "insights_dir": d,
                },
            }
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            daylog.save(
                store,
                cfg,
                5,
                "finding",
                f"N1 {O} `g.json:2`: **Golden <i>not</i> re-blessed**",
            )
            (pathlib.Path(d) / "rocke_insights_2026-10-30.md").write_text(
                "## P\n\n| a |\n|---|\n| **b** |\n", encoding="utf-8"
            )
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn("<section id='insights'", page)
        self.assertIn("href='#insights'", page)
        self.assertIn("<details id='d-insights'><summary>", page)
        self.assertNotIn("d-insights' open", page)
        self.assertIn("1 🔴/🟠 finding on 1 PR · 0 confirmed fixed before merge", page)
        self.assertIn("Golden &lt;i&gt;not&lt;/i&gt; re-blessed", page)
        self.assertNotIn("<i>not</i>", page)
        self.assertIn("<td><b>b</b></td>", page)
        self.assertIn(
            "No merge conflict recorded in the last 7 days (not tracked yet)", page
        )
        self.assertIn(
            "No finished CI recorded for heads pushed in the last 7 days yet", page
        )

    def test_the_card_lists_conflict_files_and_ci_times_escaped(self):
        # The highest offending files for merge conflicts, and CI time per PR.
        from rkmon_lib import ci_times, conflicts

        bad = "lib/kern/platform/<img src=x onerror=alert(1)>.json"
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {
                "deployment": {"phase": "queue"},
                "merge_authority": "arm-cleared",
                "operator": {"github_login": "me"},
                "repo": "ROCm/r",
                "rocke_path": "lib/kern",
                "paths": {
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                    "insights_dir": d,
                },
            }
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {
                "5": {"n": 5, "title": "t5", "author": "cat", "status": "open"}
            }
            t = now()
            store.append_jsonl(
                conflicts.FILE,
                [
                    {"ts": t, "source": "start", "scan_from": t},
                    {
                        "ts": t,
                        "source": "live",
                        "pr": 5,
                        "head": "h5",
                        "develop": "d1",
                        "files": [bad],
                    },
                    {
                        "ts": t,
                        "source": "live",
                        "pr": 6,
                        "head": "h6",
                        "develop": "d1",
                        "files": [bad],
                    },
                ],
            )
            st["conflicts"] = {"offenders": {"by": {bad: [3333]}}}
            created = (
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            done = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            store.append_jsonl(
                ci_times.FILE,
                [
                    {
                        "ts": t,
                        "pr": 5,
                        "head": "a" * 40,
                        "first_created": created,
                        "wall_s": 3 * 3600,
                        "queue_s": 5,
                        "outcome": "success",
                        "complete": True,
                        "superseded": False,
                        "reruns": 0,
                        "runs": [
                            {
                                "id": 1,
                                "name": "<script>wf</script>",
                                "path": "x.yml",
                                "status": "completed",
                                "conclusion": "success",
                                "created_at": created,
                                "updated_at": done,
                                "jobs": {
                                    "n": 1,
                                    "slow": [
                                        ["<b>job</b>", "gpu<", 600, 60, "success"]
                                    ],
                                    "queue": {"gpu<": [60]},
                                },
                            }
                        ],
                    }
                ],
            )
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn(
            "· CI median 3.0 h push→result · top conflict file: &lt;img src=x onerror=alert(1)&gt;.json "
            "(2 PRs)</p>",
            page,
        )
        self.assertIn("<h3>Merge-conflict hot files", page)
        self.assertIn(
            "<code title='lib/kern/platform/&lt;img src=x "
            "onerror=alert(1)&gt;.json'>kern/platform/&lt;img src=x onerror=alert(1)&gt;.json</code>",
            page,
        )
        self.assertIn(
            "href='https://github.com/ROCm/r/pull/3333'", page
        )  # the develop PR that last changed it
        self.assertIn(
            "<td><code>kern/platform/</code></td><td class='num'>2</td>", page
        )
        self.assertIn(
            "Push → CI result: median 3.0 h · mean 3.0 h · p90 3.0 h · n=1 heads · runner queue wait "
            "median 1 min, p90 1 min (1 jobs)",
            page,
        )
        self.assertIn(
            "<td>&lt;script&gt;wf&lt;/script&gt;</td><td class='num'>3.0 h</td>", page
        )
        self.assertIn(
            "<td>&lt;b&gt;job&lt;/b&gt;</td><td class='num'>10 min</td>", page
        )
        self.assertIn("<td>gpu&lt;</td>", page)
        for raw in ("<img src=x", "<script>wf", "<b>job"):
            self.assertNotIn(raw, page)


def git(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class TestPerPrStatistics(unittest.TestCase):
    """Where the time goes, per PR: rebases, conflicts with develop, review and fix
    rounds, CI runner waits; the top five of each and the whole window, from what the monitor records.
    """

    END = dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        repo = d / "clone"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "develop")
        sha = {}
        for name in ("base", "a1", "a2"):
            (repo / "f.txt").write_text(name + "\n", encoding="utf-8")
            git(repo, "add", "-A")
            git(repo, "commit", "-qm", name)
            sha[name] = git(repo, "rev-parse", "HEAD")[:10]
        git(repo, "checkout", "-q", "-b", "rebased", sha["base"])
        (repo / "g.txt").write_text("rebased\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "rebased")
        sha["r"] = git(repo, "rev-parse", "HEAD")[:10]
        self.sha = sha
        self.store = Store(d / "state")
        self.cfg = {"repo": "ROCm/r", "rocke_path": "rk", "paths": {"repo": str(repo)}}
        self.st = empty_state()
        self.st["prs"]["10"] = {
            "n": 10,
            "status": "open",
            "draft": False,
            "base": "develop",
            "mergeable": "CONFLICTING",
        }
        a1, a2 = sha["a1"], sha["a2"]

        def push(pr, ts, old=a1, new=a2):
            return {"ts": ts, "kind": "PUSH", "pr": pr, "old": old, "new": new}

        def conflict(pr, ts):
            return {"ts": ts, "kind": "CONFLICT", "pr": pr}

        self.store.append_jsonl(
            "events.jsonl",
            [
                # #7: a push, a force-push (a rebase: the old head is no ancestor of the new), one the clone cannot judge.
                push(7, "2026-11-01T10:00:00Z"),
                push(7, "2026-11-02T10:00:00Z", a2, sha["r"]),
                push(7, "2026-11-03T10:00:00Z", "deadbeef00", "feedface00"),
                conflict(8, "2026-11-01T00:00:00Z"),
                push(8, "2026-11-01T03:00:00Z"),  # 3 h
                conflict(9, "2026-11-02T00:00:00Z"),
                push(9, "2026-11-02T01:00:00Z"),  # seen conflicting until 06:00
                conflict(
                    10, "2026-11-04T12:00:00Z"
                ),  # still conflicting: until the window's end
                conflict(11, "2026-10-20T12:00:00Z"),
                push(11, "2026-10-22T14:00:00Z"),  # its last 2 h are in the window
                conflict(12, "2026-11-03T00:00:00Z"),
                conflict(12, "2026-11-03T10:00:00Z"),  # GitHub flip-flopped
                push(12, "2026-11-03T11:00:00Z"),
                conflict(13, "2026-11-04T00:00:00Z"),
                push(13, "2026-11-04T00:30:00Z"),  # the sixth: not in the top 5
                push(15, "2026-11-01T00:00:00Z"),
            ],
        )  # before our review: no fix round
        self.store.append_jsonl(
            "conflicts.jsonl",
            [
                {
                    "ts": "2026-11-02T06:00:00Z",
                    "source": "live",
                    "pr": 9,
                    "head": "h9",
                    "files": ["rk/a.py"],
                },
                {
                    "ts": "2026-11-02T12:00:00Z",
                    "source": "resolved",
                    "pr": 7,
                    "merge": "m1",
                    "files": [],
                },
                {
                    "ts": "2026-11-01T12:00:00Z",
                    "source": "resolved",
                    "pr": 14,
                    "merge": "m2",
                    "files": ["rk/b.py"],
                },
                {
                    "ts": "2026-11-03T12:00:00Z",
                    "source": "resolved",
                    "pr": 14,
                    "merge": "m3",
                    "files": [],
                },
                {
                    "ts": "2026-10-01T12:00:00Z",
                    "source": "resolved",
                    "pr": 14,
                    "merge": "m0",
                    "files": [],
                },
            ],
        )
        with self.store.ledger_path.open("a", encoding="utf-8") as f:
            for ts, pr in (
                ("2026-10-10T00:00:00Z", 7),
                ("2026-10-25T00:00:00Z", 7),
                ("2026-11-01T09:00:00Z", 7),
                ("2026-11-02T11:00:00Z", 7),
                ("2026-11-02T00:00:00Z", 15),
            ):
                f.write(
                    json.dumps(
                        {
                            "ts": ts,
                            "kind": "post",
                            "pr": pr,
                            "head": "h",
                            "ref": "u",
                            "note": "",
                        }
                    )
                    + "\n"
                )

        def head(pr, h, created, queue):
            return {
                "ts": created,
                "pr": pr,
                "head": h * 40,
                "first_created": created,
                "queue_s": 0,
                "runs": [{"id": 1, "jobs": {"n": 1, "slow": [], "queue": queue}}],
            }

        from rkmon_lib import ci_times

        self.store.append_jsonl(
            ci_times.FILE,
            [
                head(
                    7, "a", "2026-11-01T10:05:00Z", {"gpu-a": [90, 630], "cpu-b": [30]}
                ),
                head(7, "b", "2026-11-02T10:05:00Z", {"gpu-a": [300]}),
                head(16, "c", "2026-11-03T10:05:00Z", {"cpu-b": [12, 18]}),
                head(16, "d", "2026-10-01T10:05:00Z", {"cpu-b": [9999]}),
            ],
        )  # outside the window

    def stats(self) -> tuple[dict, dict]:
        s = insights.pr_statistics(self.cfg, self.st, self.store, days=14, end=self.END)
        return s, {c["key"]: c for c in s["categories"]}

    def test_each_statistic(self):
        s, c = self.stats()
        self.assertEqual(
            (s["start"], s["end"]), ("2026-10-22T12:00:00Z", "2026-11-05T12:00:00Z")
        )
        self.assertEqual(
            [x["key"] for x in s["categories"]],
            ["rebased", "conflicted", "review_rounds", "fix_rounds", "runner_waits"],
        )
        self.assertEqual(
            c["rebased"]["overall"],
            {
                "rebases": 4,
                "force_pushes": 1,
                "develop_merges": 3,
                "prs": 2,
                "pushes_not_checked": 1,
            },
        )
        self.assertEqual(
            c["rebased"]["top"],
            [
                {"pr": 7, "value": 2, "detail": "1 force-push(es), 1 develop merge(s)"},
                {
                    "pr": 14,
                    "value": 2,
                    "detail": "0 force-push(es), 2 develop merge(s)",
                },
            ],
        )
        self.assertEqual(
            c["conflicted"]["overall"], {"conflicts": 7, "prs": 6, "hours": 36.5}
        )
        self.assertEqual(
            [(x["pr"], x["value"], x["detail"]) for x in c["conflicted"]["top"]],
            [
                (10, 24.0, "1 time(s), 24.0 h in all"),
                (9, 6.0, "1 time(s), 6.0 h in all"),
                (8, 3.0, "1 time(s), 3.0 h in all"),
                (11, 2.0, "1 time(s), 2.0 h in all"),
                (12, 1.0, "2 time(s), 1.0 h in all"),
            ],
        )
        self.assertEqual(c["review_rounds"]["overall"], {"posts": 4, "prs": 2})
        self.assertEqual(
            [(x["pr"], x["value"]) for x in c["review_rounds"]["top"]],
            [(7, 3), (15, 1)],
        )
        self.assertEqual(
            c["fix_rounds"]["overall"], {"rounds": 2, "pushes": 3, "prs": 1}
        )
        self.assertEqual(
            c["fix_rounds"]["top"],
            [
                {
                    "pr": 7,
                    "value": 2,
                    "detail": "2 round(s), 3 push(es) after our " "reviews",
                }
            ],
        )
        self.assertEqual(
            c["runner_waits"]["overall"],
            {"jobs": 6, "prs": 2, "median_min": 1.0, "p90_min": 10.5, "total_h": 0.3},
        )
        self.assertEqual(
            c["runner_waits"]["top"],
            [
                {
                    "pr": 7,
                    "value": 17.5,
                    "detail": "4 job(s) on 2 head(s), the longest 10.5 min",
                },
                {
                    "pr": 16,
                    "value": 0.5,
                    "detail": "2 job(s) on 1 head(s), the longest 0.3 min",
                },
            ],
        )
        self.assertEqual(
            c["runner_waits"]["labels"],
            [
                {
                    "label": "gpu-a",
                    "jobs": 3,
                    "median_min": 5.0,
                    "p90_min": 10.5,
                    "total_min": 17.0,
                },
                {
                    "label": "cpu-b",
                    "jobs": 3,
                    "median_min": 0.3,
                    "p90_min": 0.5,
                    "total_min": 1.0,
                },
            ],
        )

    def test_the_force_push_verdicts_are_cached(self):
        first, _ = self.stats()
        self.assertEqual(
            len(
                json.loads(
                    (self.store.dir / insights.CACHE).read_text(encoding="utf-8")
                )["ancestry"]
            ),
            3,
        )
        with mock.patch.object(
            insights.gitops, "has_commit", side_effect=AssertionError("asked git again")
        ):
            again, _ = self.stats()
        self.assertEqual(again, first)
        self.store.append_jsonl(
            "events.jsonl",
            [
                {
                    "ts": "2026-11-04T09:00:00Z",
                    "kind": "PUSH",
                    "pr": 7,
                    "old": self.sha["r"],
                    "new": self.sha["a2"],
                }
            ],
        )
        changed, c = self.stats()
        self.assertEqual(c["rebased"]["overall"]["force_pushes"], 2)
        self.assertNotEqual(changed["fp"], first["fp"])

    def test_the_numbers_a_reading_may_cite_and_the_markdown(self):
        s, _ = self.stats()
        nums = insights.numbers(s)
        for n in ("14", "7", "24", "36.5", "17.5", "10.5", "0.3"):
            self.assertIn(n, nums)
        self.assertNotIn("2026", nums)  # the window's dates are not figures
        self.assertEqual(
            [
                m.group(0)
                for m in insights.NUMBER.finditer(
                    "#7 waited 3. Then 2.5 h; x86 1.5.2 50%"
                )
            ],
            ["7", "3", "2.5", "50"],
        )
        self.assertEqual(
            (insights.norm(3.0), insights.norm("3.50"), insights.norm(17.5)),
            ("3", "3.5", "17.5"),
        )
        md = "\n".join(insights.stats_md(s))
        self.assertIn("## Where the time goes, per PR (last 14 days)", md)
        self.assertIn(
            "### Rebased: force-pushes (a rebase or an amend), and develop merges into the PR",
            md,
        )
        self.assertIn("| #10 | 24.0 | 1 time(s), 24.0 h in all |", md)
        self.assertIn("| gpu-a | 3 | 5.0 | 10.5 | 17.0 |", md)
        self.assertNotIn("| gpu-a |", "\n".join(insights.stats_md(s, labels=False)))
        reading = {
            "summary": "Runner waits first.",
            "model": "Read.Model-2",
            "at": "2026-11-05T11:00:00Z",
            "ranking": [{"category": "runner_waits", "pr": 7, "note": "17.5 min"}],
        }
        md = "\n".join(insights.stats_md(s, reading))
        self.assertIn(
            "The analyst's reading (Read.Model-2, 2026-11-05T11:00:00Z): Runner waits first.\n\n"
            "- CI runner waits #7: 17.5 min",
            md,
        )

    def test_rk_insights_and_the_dashboard(self):
        import rkmon

        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            self.store, "load", lambda: self.st
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["insights", "--days", "30"]), 0)
        self.assertIn("## Where the time goes, per PR (last 30 days)", out[0])
        html = webdash.pr_stats_html(self.cfg, self.st, self.store)
        self.assertIn(
            "<h3>Where the time goes, per PR <span class='c'>last 14 days, the top 5 of each</span></h3>",
            html,
        )
        self.assertNotIn("Runner label", html)  # the CI section above has them
        with mock.patch.object(
            insights, "pr_statistics", side_effect=RuntimeError("<boom>")
        ):
            self.assertEqual(
                webdash.pr_stats_html(self.cfg, self.st, self.store),
                "<p class='sub'>Per-PR statistics not computed: &lt;boom&gt;</p>",
            )


if __name__ == "__main__":
    unittest.main()
