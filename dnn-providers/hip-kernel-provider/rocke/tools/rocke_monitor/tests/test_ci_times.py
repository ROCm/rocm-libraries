# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI time per PR: which runs are a head's CI, push → result with re-run attempts,
superseded and cancelled heads, a recorded head never read again, the per-pass caps, job timings and the summary. No
network: `gh` is mocked."""

import datetime as dt
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import ci_times, gh  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

UTC = dt.timezone.utc
A, B, C = "a" * 40, "b" * 40, "c" * 40


def run(
    id_,
    name,
    created,
    updated,
    event="pull_request",
    status="completed",
    conclusion="success",
    started=None,
    attempt=1,
    path=None,
):
    return {
        "id": id_,
        "name": name,
        "path": path or f".github/workflows/{name.lower().replace(' ', '-')}.yml",
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "created_at": created,
        "run_started_at": started or created,
        "updated_at": updated,
        "run_attempt": attempt,
    }


def job(name, label, created, started, completed, conclusion="success"):
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": created,
        "started_at": started,
        "completed_at": completed,
        "labels": [label],
    }


REF = dt.datetime.now(UTC).replace(
    microsecond=0
)  # one clock for a test's times, so their differences are exact


def ago(minutes: float) -> str:
    return (REF - dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestHeadRow(unittest.TestCase):
    def test_a_heads_ci_and_push_to_result(self):
        runs = [
            run(
                1,
                "Auto Label PR",
                "2026-10-30T10:00:00Z",
                "2026-10-30T10:00:20Z",
                event="pull_request_target",
            ),
            run(2, "pre-commit", "2026-10-30T10:00:02Z", "2026-10-30T10:03:00Z"),
            # the auto-retry: attempt 2 started later, the run's created_at stays
            run(
                3,
                "TheRock Multi-Arch CI",
                "2026-10-30T10:00:03Z",
                "2026-10-30T13:00:02Z",
                started="2026-10-30T11:30:00Z",
                attempt=2,
            ),
            run(
                4,
                "Running Copilot Code Review",
                "2026-10-30T12:00:00Z",
                "2026-10-30T15:00:00Z",
                event="dynamic",
            ),
            run(5, "TheRock CI", "2026-11-02T10:00:00Z", "2026-11-02T18:00:00Z"),
        ]  # a label days on: not this push
        r = ci_times.head_row(7, A, runs)
        self.assertEqual(
            (r["first_created"], r["last_completed"], r["wall_s"], r["queue_s"]),
            ("2026-10-30T10:00:02Z", "2026-10-30T13:00:02Z", 3 * 3600, 0),
        )
        self.assertEqual(
            (r["outcome"], r["complete"], r["superseded"], r["reruns"], r["later"]),
            ("success", True, False, 1, 1),
        )
        self.assertEqual([x["id"] for x in r["runs"]], [2, 3])

    def test_outcomes_and_superseded(self):
        base = [run(2, "pre-commit", "2026-10-30T10:00:02Z", "2026-10-30T10:03:00Z")]
        cancelled = base + [
            run(
                3,
                "TheRock",
                "2026-10-30T10:00:03Z",
                "2026-10-30T10:20:00Z",
                conclusion="cancelled",
            )
        ]
        r = ci_times.head_row(7, A, cancelled, replaced_at="2026-10-30T10:19:00Z")
        self.assertEqual((r["outcome"], r["superseded"]), ("cancelled", True))
        failed = base + [
            run(
                3,
                "TheRock",
                "2026-10-30T10:00:03Z",
                "2026-10-30T12:00:00Z",
                conclusion="failure",
            )
        ]
        r = ci_times.head_row(
            7, A, failed, replaced_at="2026-10-30T12:30:00Z"
        )  # pushed after CI finished
        self.assertEqual((r["outcome"], r["superseded"]), ("failure", False))
        r = ci_times.head_row(
            7, A, failed, replaced_at="2026-10-30T11:00:00Z"
        )  # pushed while CI still ran
        self.assertTrue(r["superseded"])
        mixed = failed + [
            run(
                4,
                "Component CI",
                "2026-10-30T10:00:04Z",
                "2026-10-30T10:30:00Z",
                conclusion="cancelled",
            )
        ]
        self.assertEqual(ci_times.head_row(7, A, mixed)["outcome"], "mixed")
        running = base + [
            run(
                3,
                "TheRock",
                "2026-10-30T10:00:03Z",
                "2026-10-30T10:30:00Z",
                status="in_progress",
                conclusion=None,
            )
        ]
        r = ci_times.head_row(7, A, running)
        self.assertEqual(
            (r["outcome"], r["complete"], r["wall_s"]), ("incomplete", False, None)
        )
        # a fork's runs waiting for a maintainer's approval never ran
        waiting = [
            run(
                2,
                "pre-commit",
                "2026-11-01T09:23:12Z",
                "2026-11-01T09:23:12Z",
                conclusion="action_required",
            ),
            run(
                1,
                "Libraries PR Bot",
                "2026-11-01T09:23:12Z",
                "2026-11-01T09:43:35Z",
                event="pull_request_target",
            ),
        ]
        self.assertIsNone(ci_times.head_row(7, A, waiting))

    def test_a_label_rerun_replaces_the_run_it_cancelled(self):
        # labels added minutes after the push start a new Multi-Arch run each and cancel the one before: the
        # head's result is each workflow's newest run, and its CI ran to the end
        runs = [
            run(2, "pre-commit", "2026-11-01T19:23:47Z", "2026-11-01T19:26:00Z"),
            run(
                3,
                "TheRock Multi-Arch CI",
                "2026-11-01T19:23:48Z",
                "2026-11-01T19:28:30Z",
                conclusion="cancelled",
            ),
            run(
                4,
                "TheRock Multi-Arch CI",
                "2026-11-01T19:27:57Z",
                "2026-11-01T19:35:00Z",
                conclusion="cancelled",
            ),
            run(
                5,
                "TheRock Multi-Arch CI",
                "2026-11-01T19:34:32Z",
                "2026-11-01T22:34:32Z",
                conclusion="failure",
            ),
        ]
        r = ci_times.head_row(7, A, runs)
        self.assertEqual(
            (r["outcome"], r["complete"], r["superseded"], r["wall_s"]),
            ("failure", True, False, 3 * 3600 + 645),
        )
        runs[-1]["conclusion"] = "success"
        self.assertEqual(ci_times.head_row(7, A, runs)["outcome"], "success")

    def test_job_summary_slowest_and_queue_by_label(self):
        jobs = [
            job(
                "Build Linux",
                "aws-linux",
                "2026-10-30T10:00:00Z",
                "2026-10-30T10:05:00Z",
                "2026-10-30T11:05:00Z",
            ),
            job(
                "Test gfx110X",
                "windows-gfx110X",
                "2026-10-30T11:05:00Z",
                "2026-10-30T11:20:00Z",
                "2026-10-30T11:40:00Z",
            ),
            # copied from attempt 1 by the re-run: started before this attempt created it
            job(
                "Build Windows",
                "aws-windows",
                "2026-10-30T12:00:00Z",
                "2026-10-30T10:01:00Z",
                "2026-10-30T10:58:00Z",
            ),
            job(
                "Skipped",
                "ubuntu",
                "2026-10-30T10:00:00Z",
                "2026-10-30T10:00:00Z",
                "2026-10-30T10:00:00Z",
                conclusion="skipped",
            ),
        ]
        s = ci_times.job_summary(jobs, keep=2)
        self.assertEqual(s["n"], 3)
        self.assertEqual(
            s["slow"],
            [
                ["Build Linux", "aws-linux", 3600, 300, "success"],
                ["Build Windows", "aws-windows", 3420, None, "success"],
            ],
        )
        self.assertEqual(s["queue"], {"aws-linux": [300], "windows-gfx110X": [900]})


class TestCollect(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.store = Store(pathlib.Path(self.d.name))
        self.cfg = {"repo": "ROCm/r", "paths": {"repo": "/nonexistent"}}
        self.st = empty_state()
        self.st["prs"] = {
            "7": {
                "n": 7,
                "head": A,
                "status": "open",
                "ci_rollup": "SUCCESS",
                "head_seen_at": ago(300),
            },
            "8": {
                "n": 8,
                "head": C,
                "status": "open",
                "ci_rollup": "PENDING",
                "head_seen_at": ago(30),
            },
        }
        # #7 moved from b… to a… two hours after b's CI started; b's runs were cancelled by the push
        self.store.append_jsonl(
            "events.jsonl",
            [{"kind": "PUSH", "pr": 7, "old": B[:10], "new": A[:10], "ts": ago(400)}],
        )
        self.calls = []
        self.runs = {
            A: [
                run(10, "pre-commit", ago(420), ago(415)),
                run(
                    11,
                    "TheRock Multi-Arch CI",
                    ago(420),
                    ago(240),
                    started=ago(330),
                    attempt=2,
                ),
                run(
                    12,
                    "Libraries PR Bot",
                    ago(420),
                    ago(419),
                    event="pull_request_target",
                ),
            ],
            B: [
                run(20, "pre-commit", ago(600), ago(598)),
                run(
                    21,
                    "TheRock Multi-Arch CI",
                    ago(600),
                    ago(399),
                    conclusion="cancelled",
                ),
            ],
        }
        self.jobs = {
            11: [
                job("Build", "aws-linux", ago(420), ago(410), ago(300)),
                job("Test", "linux-gfx942", ago(300), ago(270), ago(245)),
            ]
        }
        expand = mock.patch.object(
            ci_times.gitops,
            "expand",
            lambda repo, shorts: {B[:10]: B} if B[:10] in shorts else {},
        )
        expand.start()
        self.addCleanup(expand.stop)

    def rest(self, path, jq=None, paginate=False, timeout=120):
        self.calls.append(path)
        if "actions/runs?head_sha=" in path:
            return self.runs[path.split("head_sha=")[1].split("&")[0]]
        if "/jobs" in path:
            return self.jobs.get(int(path.split("/runs/")[1].split("/")[0]), [])
        raise AssertionError(path)

    def test_records_finished_and_superseded_heads_once(self):
        with mock.patch.object(gh, "rest", self.rest):
            out = ci_times.collect(self.cfg, self.st, self.store)
            self.assertEqual(
                (out["recorded"], out["pending"], out["calls"], out["left"]),
                (2, 1, 3, 0),
            )
            rows = {r["head"]: r for r in self.store.read_jsonl(ci_times.FILE)}
            a, b = rows[A], rows[B]
            self.assertEqual(
                (a["outcome"], a["superseded"], a["reruns"], a["wall_s"]),
                ("success", False, 1, 180 * 60),
            )
            self.assertEqual(
                [x["id"] for x in a["runs"]], [10, 11]
            )  # the bot is not CI
            jobs = next(x for x in a["runs"] if x["id"] == 11)[
                "jobs"
            ]  # the longest workflow's jobs
            self.assertEqual(
                jobs["queue"], {"aws-linux": [600], "linux-gfx942": [1800]}
            )
            self.assertEqual(
                (b["outcome"], b["superseded"], b["pr"]), ("cancelled", True, 7)
            )
            self.assertNotIn(
                f"head_sha={C}", " ".join(self.calls)
            )  # #8's CI is still running: not asked
            self.calls.clear()
            out = ci_times.collect(self.cfg, self.st, self.store)
        self.assertEqual(
            (out["recorded"], out["calls"]), (0, 0)
        )  # a recorded head is never read again
        self.assertEqual(self.calls, [])

    def test_caps_errors_and_heads_without_ci(self):
        with mock.patch.object(gh, "rest", self.rest):
            out = ci_times.collect(self.cfg, self.st, self.store, max_calls=1)
        self.assertEqual(
            (out["recorded"], out["left"], out["calls"]), (1, 1, 1)
        )  # #7's current head, jobs skipped
        self.assertNotIn(
            "jobs",
            next(
                x
                for x in self.store.read_jsonl(ci_times.FILE)[0]["runs"]
                if x["id"] == 11
            ),
        )

        def down(path, jq=None, paginate=False, timeout=120):
            raise gh.GhError("gh api: HTTP 502")

        with mock.patch.object(gh, "rest", down):
            out = ci_times.collect(self.cfg, self.st, self.store)
        self.assertEqual(
            (out["recorded"], out["left"]), (0, 1)
        )  # stops at GitHub trouble, the next pass goes on
        self.assertIn("502", out["error"])
        self.runs[B] = [
            run(30, "Libraries PR Bot", ago(600), ago(590), event="pull_request_target")
        ]
        with mock.patch.object(gh, "rest", self.rest):
            out = ci_times.collect(self.cfg, self.st, self.store)
        self.assertEqual(
            out["no_ci"], 1
        )  # a replaced head without CI is recorded, so never read again
        self.assertEqual(self.store.read_jsonl(ci_times.FILE)[-1]["outcome"], "none")

    def test_a_current_head_without_ci_is_asked_again_later(self):
        self.runs[A] = []
        self.store.append_jsonl(
            ci_times.FILE, [{"head": B, "pr": 7, "outcome": "none", "runs": []}]
        )
        with mock.patch.object(gh, "rest", self.rest):
            ci_times.collect(self.cfg, self.st, self.store)
            self.assertIn(A, self.st["ci_times"]["empty"])
            self.calls.clear()
            ci_times.collect(self.cfg, self.st, self.store)
            self.assertEqual(self.calls, [])
            self.st["ci_times"]["empty"][A] = ago(13 * 60)
            ci_times.collect(self.cfg, self.st, self.store)
            self.assertEqual(len(self.calls), 1)

    def test_refresh_is_hourly(self):
        with mock.patch.object(gh, "rest", self.rest):
            self.assertIsNotNone(ci_times.refresh(self.cfg, self.st, self.store))
            self.calls.clear()
            self.assertIsNone(ci_times.refresh(self.cfg, self.st, self.store))
            self.st["ci_times"]["at"] = ago(61)
            self.assertEqual(
                ci_times.refresh(self.cfg, self.st, self.store)["recorded"], 0
            )
        self.assertEqual(self.calls, [])


class TestSummary(unittest.TestCase):
    def test_summary_and_render(self):
        def row(
            pr,
            head,
            created,
            wall_h,
            outcome="success",
            superseded=False,
            reruns=0,
            jobs=None,
        ):
            done = (ci_times._ts(created) + dt.timedelta(hours=wall_h)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            long = run(
                1,
                "TheRock Multi-Arch CI",
                created,
                done,
                conclusion="cancelled" if outcome == "cancelled" else "success",
            )
            if jobs:
                long["jobs"] = jobs
            return {
                "ts": now(),
                "pr": pr,
                "head": head,
                "first_created": created,
                "wall_s": wall_h * 3600,
                "queue_s": 2,
                "outcome": outcome,
                "complete": True,
                "superseded": superseded,
                "reruns": reruns,
                "runs": [
                    long,
                    run(
                        2,
                        "PR 7 @ abc",
                        created,
                        created,
                        path=".github/workflows/gfx1250_test.yml",
                    ),
                ],
            }

        jobs = {
            "n": 2,
            "slow": [["Build | Linux", "aws-linux", 3000, 60, "success"]],
            "queue": {"win-gpu": [600, 1200]},
        }
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            store.append_jsonl(
                ci_times.FILE,
                [
                    row(7, "h1", "2026-10-27T10:00:00Z", 1, jobs=jobs),
                    row(
                        9, "h2", "2026-10-28T10:00:00Z", 3, outcome="failure", reruns=1
                    ),
                    row(
                        7,
                        "h3",
                        "2026-10-29T10:00:00Z",
                        0.5,
                        outcome="cancelled",
                        superseded=True,
                    ),
                    row(5, "h4", "2026-10-20T10:00:00Z", 9),  # pushed before the window
                    # a newer push came while its CI ran, which still ran to the end: in the times and in the cut-short
                    # share (the slowest CI is the likeliest to be overtaken)
                    row(9, "h5", "2026-10-30T10:00:00Z", 2, superseded=True),
                ],
            )
            s = ci_times.summary(
                store,
                dt.datetime(2026, 10, 26, tzinfo=UTC),
                dt.datetime(2026, 11, 2, tzinfo=UTC),
            )
            none = ci_times.summary(
                store,
                dt.datetime(2026, 10, 26, tzinfo=UTC),
                dt.datetime(2026, 10, 27, tzinfo=UTC),
            )
        self.assertEqual(
            (s["heads"], s["done"], s["cut"], s["cut_share"]), (4, 3, 2, 0.5)
        )
        self.assertEqual(
            (s["overall"]["median"], s["overall"]["mean"], s["overall"]["p90"]),
            (7200, 7200, 10800),
        )
        self.assertEqual(
            [(p["pr"], p["mean_s"], p["n"], p["heads"], p["failed"]) for p in s["prs"]],
            [(9, 9000, 2, 2, 1), (7, 3600, 1, 2, 0)],
        )
        self.assertEqual(
            [(w["name"], w["n"]) for w in s["workflows"]],
            [("TheRock Multi-Arch CI", 3), ("PR 7 @ abc", 4)],
        )
        self.assertEqual(
            (s["job_wait"]["median"], s["labels"][0]["label"], s["reruns"]),
            (900, "win-gpu", 1),
        )
        self.assertEqual(
            ci_times.headline(s), "median 2.0 h · mean 2.0 h · p90 3.0 h · n=3 heads"
        )
        text = "\n".join(ci_times.render_md(s, who=lambda n: f"#{n} (dev{n})"))
        self.assertIn("| Push → CI result | 2.0 h | 2.0 h | 3.0 h | 3 heads |", text)
        self.assertIn("| #9 (dev9) | 2.5 h | 3.0 h | 2 / 2 | 1 |", text)
        self.assertIn(
            "| Build \\| Linux | 50 min | 1 |", text
        )  # a pipe in a GitHub name cannot break the table
        self.assertIn("2 of 4 heads (50%)", text)
        self.assertEqual(ci_times.p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 9)
        self.assertEqual(ci_times.p90([1, 2]), 2)
        self.assertIsNone(none["overall"])
        self.assertIn("No finished CI recorded", "\n".join(ci_times.render_md(none)))
        self.assertEqual(
            ci_times.headline(none), "— (no finished CI recorded in the window)"
        )


if __name__ == "__main__":
    unittest.main()
