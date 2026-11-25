# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the weekly statistics (no GitHub: search and review fetches are patched)."""

import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import stats  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

UTC = dt.timezone.utc


class TestWeeks(unittest.TestCase):
    def test_bounds(self):
        s, e, lab = stats.week_bounds("2026-W44")
        self.assertEqual(
            (s.isoformat(), e.isoformat(), lab),
            ("2026-10-26T00:00:00+00:00", "2026-11-02T00:00:00+00:00", "2026-W44"),
        )
        s, e, lab = stats.week_bounds(None, dt.datetime(2026, 10, 28, 15, tzinfo=UTC))
        self.assertEqual(lab, "2026-W43")  # the last complete week


class TestCompute(unittest.TestCase):
    def test_friction_incidents_and_render(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "2": {
                    "n": 2,
                    "author": "bob",
                    "status": "open",
                    "draft": False,
                    "created_at": "2026-10-20T00:00:00Z",
                    "mergeable": "CONFLICTING",
                }
            }
            store.append_jsonl(
                "events.jsonl",
                [
                    {"ts": "2026-10-27T10:00:00Z", "kind": "MERGED", "pr": 1},
                    {"ts": "2026-10-27T12:00:00Z", "kind": "CONFLICT", "pr": 2},
                    {
                        "ts": "2026-10-27T13:00:00Z",
                        "kind": "CI",
                        "pr": 2,
                        "state": "FAILURE",
                    },
                    {
                        "ts": "2026-10-28T13:00:00Z",
                        "kind": "CI",
                        "pr": 2,
                        "state": "FAILURE",
                    },
                ],
            )
            (store.dir / "incidents.json").write_text(
                json.dumps(
                    [
                        {
                            "start": "2026-10-27T05:13:59Z",
                            "end": "2026-10-28T14:07:36Z",
                            "culprit": 1,
                            "fixed_by": 3,
                            "contributing": [],
                            "summary": "golden stale",
                        }
                    ]
                )
            )
            merged = [
                {
                    "number": 1,
                    "title": "t1",
                    "createdAt": "2026-10-25T00:00:00Z",
                    "mergedAt": "2026-10-27T10:00:00Z",
                    "author": {"login": "amy"},
                }
            ]
            searches = iter([merged, [], [], merged])
            with mock.patch.object(
                stats, "_search", lambda q: next(searches)
            ), mock.patch.object(
                stats.gh, "rest", lambda path, jq=None, paginate=False: "zed"
            ), mock.patch.object(
                stats,
                "our_reviews",
                lambda cfg, prs, end: {
                    1: [
                        {
                            "at": "2026-10-26T09:00:00Z",
                            "verdict": "\U0001f7e2",
                            "follow_up": False,
                        }
                    ]
                },
            ):
                s, e, lab = stats.week_bounds("2026-W44")
                out = stats.compute(
                    {
                        "repo": "ROCm/r",
                        "scope_label": "x",
                        "operator": {"github_login": "me"},
                    },
                    st,
                    store,
                    s,
                    e,
                )
        self.assertEqual(
            out["friction"][1], ["broke develop", "likely forced a rebase of #2"]
        )
        self.assertEqual(
            out["authors"][3], "zed"
        )  # the fixer, outside the search results
        self.assertEqual(out["friction"][2], ["CI failed 2×"])
        self.assertAlmostEqual(out["red_hours"], 32.9, places=1)
        self.assertEqual(out["reviews"]["posts"], 1)
        text = stats.render(out, lab)
        self.assertIn("| PRs merged | 1 |", text)
        self.assertIn("#1 (amy)", text)
        self.assertIn("32.9 h", text)
        # nothing recorded yet: the new sections say so instead of failing
        self.assertIn(
            "| CI push → result | — (no finished CI recorded in the window) |", text
        )
        self.assertIn("## Merge-conflict hot files", text)
        # Where the time goes, per PR: #2 conflicts with develop to the week's end.
        self.assertIn("## Where the time goes, per PR (last 7 days)", text)
        self.assertRegex(text, r"\| #2 \| [0-9.]+ \| 1 time\(s\), [0-9.]+ h in all \|")
        self.assertNotIn(
            "| Runner label | Jobs |", text
        )  # the CI section has the runner labels

    def test_conflict_files_and_ci_times_sections(self):
        # The files PRs conflict on most, and CI time per PR, every week.
        from rkmon_lib import ci_times, conflicts

        rk = "team/provider/rocke"
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "5": {
                    "n": 5,
                    "author": "cat",
                    "status": "open",
                    "draft": False,
                    "created_at": "2026-10-20T00:00:00Z",
                    "mergeable": "MERGEABLE",
                }
            }
            store.append_jsonl(
                conflicts.FILE,
                [
                    {
                        "ts": "2026-10-26T08:00:00Z",
                        "source": "start",
                        "scan_from": "2026-10-18T08:00:00Z",
                    },
                    {
                        "ts": "2026-10-27T01:00:00Z",
                        "source": "live",
                        "pr": 5,
                        "head": "h5",
                        "develop": "d1",
                        "files": [f"{rk}/platform/cpp/include/rocke/abi.h"],
                    },
                ],
            )
            store.append_jsonl(
                ci_times.FILE,
                [
                    {
                        "ts": "2026-10-28T00:00:00Z",
                        "pr": 5,
                        "head": "a" * 40,
                        "first_created": "2026-10-27T10:00:00Z",
                        "wall_s": 4 * 3600,
                        "queue_s": 5,
                        "outcome": "success",
                        "complete": True,
                        "superseded": False,
                        "reruns": 0,
                        "runs": [
                            {
                                "id": 1,
                                "name": "TheRock Multi-Arch CI",
                                "path": "x.yml",
                                "status": "completed",
                                "conclusion": "success",
                                "created_at": "2026-10-27T10:00:00Z",
                                "updated_at": "2026-10-27T14:00:00Z",
                            }
                        ],
                    }
                ],
            )
            with mock.patch.object(stats, "_search", lambda q: []), mock.patch.object(
                stats, "our_reviews", lambda cfg, prs, end: {}
            ), mock.patch.object(
                ci_times, "collect", lambda *a, **kw: {"left": 0, "error": None}
            ):
                s, e, lab = stats.week_bounds("2026-W44")
                text = stats.render(
                    stats.compute(
                        {
                            "repo": "ROCm/r",
                            "rocke_path": rk,
                            "scope_label": "x",
                            "operator": {"github_login": "me"},
                        },
                        st,
                        store,
                        s,
                        e,
                    ),
                    lab,
                )
        self.assertIn(
            "| Top merge-conflict file | `rocke/platform/cpp/include/rocke/abi.h` (1 PR, 1 episode) |",
            text,
        )
        self.assertIn(
            "| CI push → result | median 4.0 h · mean 4.0 h · p90 4.0 h · n=1 heads |",
            text,
        )
        self.assertIn(
            "| 1 | `rocke/platform/cpp/include/rocke/abi.h` | 1 | 1 | #5 | — |", text
        )
        self.assertIn("## CI time per PR", text)
        self.assertIn("| #5 (cat) | 4.0 h | 4.0 h | 1 / 1 | 0 |", text)
        self.assertIn("| TheRock Multi-Arch CI | 4.0 h | 4.0 h | 1 |", text)
        self.assertIn("- Merge-conflict files: tracked since 2026-10-26 08:00Z", text)


class TestScoped(unittest.TestCase):
    def test_a_shared_label_counts_a_pr_only_when_it_changes_files_in_its_folder(self):
        """The week's PRs: those a scope label finds, and those a shared label finds that change files in that label's
        folder, two labels with two folders, each PR once (snapshot.in_scope, from the scope cache's answers).
        """
        cfg = {
            "repo": "o/r",
            "scope": {
                "labels": ["s"],
                "path_labels": {"area: a": "projects/a", "area: b": "projects/b"},
            },
        }
        found = {
            "s": [1],
            "area: a": [7, 8],
            "area: b": [7, 9],
        }
        st = {
            "scope_cache": {
                "7": {
                    "head": "7" * 40,
                    "folders": {"projects/a": False, "projects/b": True},
                },
                "8": {"head": "8" * 40, "folders": {"projects/a": True}},
                "9": {"head": "9" * 40, "folders": {"projects/b": False}},
            }
        }

        def search(q):
            label = q.split('label:"', 1)[1].split('"', 1)[0]
            return [{"number": n, "headRefOid": str(n) * 40} for n in found[label]]

        with mock.patch.object(stats, "_search", search):
            got = stats._scoped(cfg, st, "is:merged")
        self.assertEqual(sorted(p["number"] for p in got), [1, 7, 8])


if __name__ == "__main__":
    unittest.main()
