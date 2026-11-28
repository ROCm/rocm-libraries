# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk shard-times N` (rkmon_lib/shard_times.py): a PR's shard times against develop's same shard on the same pool, from
the CTest entry times in the job logs; GitHub is a fake behind gh.rest and gh.raw."""

import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sandbox  # noqa: E402
from rkmon_lib import gh, shard_times  # noqa: E402

REPO = "o/r"
CFG = {"repo": REPO}
HEAD = "a" * 40
DEV = ["d" * 40, "e" * 40, "f" * 40]
S1 = "Linux::gfx94X / Test hipkernelprovider (shard 1/2) (gfx94X-dcgpu)"
S2 = "Linux::gfx94X / Test hipkernelprovider (shard 2/2) (gfx94X-dcgpu)"
POOL = ["linux-pool-a"]


def ctest(entries: dict, started=()) -> str:
    """A shard's log: CTest's Start and finished lines, with the runner's timestamps."""
    out = ["2026-11-10T01:00:00.1Z ##[group]Run ctest"]
    for i, (name, secs) in enumerate(entries.items(), 1):
        out.append(f"2026-11-10T01:00:01.1Z       Start {i}: {name}")
        state = "***Timeout" if secs >= 300 else "   Passed"
        out.append(
            f"2026-11-10T01:00:02.1Z {i}/9 Test  #{i}: {name} ......{state}  {secs:.2f} sec"
        )
    for name in started:
        out.append(f"2026-11-10T01:30:00.1Z       Start 9: {name}")
    out.append(
        "2026-11-10T01:30:00.2Z ##[error]The action 'Test' has timed out after 30 minutes."
    )
    return "\n".join(out) + "\n"


class FakeGitHub:
    """The reads shard_times makes: the PR, check runs per commit, jobs, logs and the base branch's commits."""

    def __init__(self):
        self.runs: dict[str, list] = {}
        self.jobs: dict[int, dict] = {}
        self.logs: dict[int, str] = {}
        self.calls: list[str] = []

    def add(self, sha, name, job, conclusion, labels, start, end, log):
        self.runs.setdefault(sha, []).append(
            {
                "name": name,
                "status": "completed",
                "conclusion": conclusion,
                "started": start,
                "completed": end,
                "url": f"https://github.com/{REPO}/actions/runs/1/job/{job}",
            }
        )
        self.jobs[job] = {
            "labels": labels,
            "runner_name": "r-1",
            "runner_group_name": "",
            "started_at": start,
            "completed_at": end,
        }
        self.logs[job] = log

    def rest(self, path, jq=None, paginate=False, timeout=120):
        self.calls.append(path)
        if path == f"repos/{REPO}/pulls/7":
            return {"head": HEAD, "base": "develop"}
        if path.startswith(f"repos/{REPO}/commits?sha=develop"):
            return DEV
        if "/check-runs" in path:
            return self.runs.get(path.split("/commits/")[1].split("/")[0], [])
        if "/actions/jobs/" in path:
            return self.jobs[int(path.rsplit("/", 1)[1])]
        raise gh.GhError(f"HTTP 404 {path}")

    def raw(self, path, timeout=120):
        self.calls.append(path)
        job = int(path.split("/jobs/")[1].split("/")[0])
        if job not in self.logs:
            raise gh.GhError("HTTP 404")
        return self.logs[job]


class TestShardTimes(unittest.TestCase):
    def setUp(self):
        self.gh = FakeGitHub()
        for p in (
            mock.patch.object(shard_times.gh, "rest", self.gh.rest),
            mock.patch.object(shard_times.gh, "raw", self.gh.raw),
        ):
            p.start()
            self.addCleanup(p.stop)

    def develop_runs(self, entries, labels=POOL, minutes=(20, 22, 21)):
        for k, (sha, m) in enumerate(zip(DEV, minutes)):
            for s, name in enumerate((S1, S2)):
                self.gh.add(
                    sha,
                    name,
                    500 + 10 * k + s,
                    "success",
                    labels,
                    "2026-11-10T00:00:00Z",
                    f"2026-11-10T00:{m:02d}:00Z",
                    ctest(entries),
                )

    def report(self, **kw):
        lines, ok = shard_times.report(CFG, 7, **kw)
        self.assertTrue(ok)
        return lines

    def test_entry_times_parse_finished_timeouts_and_the_cut(self):
        done, unfinished = shard_times.entry_times(
            ctest(
                {"rocke_pytest_core": 81.5, "rocke_pytest": 300.05},
                started=["rocke_conv_pytest"],
            )
            + "  3/9 Test  #3: rocke_gtest ...***Exception: SegFault  0.10 sec\n"
        )
        self.assertEqual(
            done,
            {"rocke_pytest_core": 81.5, "rocke_pytest": 300.05, "rocke_gtest": 0.1},
        )
        self.assertEqual(unfinished, ["rocke_conv_pytest"])

    def test_a_shard_the_pr_slows_reads_pr_added(self):
        """A new entry and a slower shared entry, on both timed-out shards: PR-added, with the entries named."""
        self.develop_runs({"rocke_pytest_core": 100.0, "rocke_gtest": 50.0})
        for name, job, extra in ((S1, 1, 40.0), (S2, 2, 75.0)):
            self.gh.add(
                HEAD,
                name,
                job,
                "failure",
                POOL,
                "2026-11-10T02:00:00Z",
                "2026-11-10T02:30:00Z",
                ctest(
                    {
                        "rocke_pytest_core": 101.0,
                        "rocke_gtest": 50.0 + extra,
                        "rocke_new_pytest": 30.0,
                    }
                ),
            )
        lines = self.report()
        self.assertEqual(lines[-1], "VERDICT PR-added +70..105 s per shard")
        self.assertIn("2 shard job(s), failed or timed out", lines[0])
        self.assertIn("rocke_gtest +75 s", "\n".join(lines))
        self.assertIn("rocke_new_pytest new 30 s", "\n".join(lines))
        self.assertIn("develop: 3 run(s) on the same pool, 20–22 min", "\n".join(lines))

    def test_the_same_entry_times_read_as_the_pool(self):
        self.develop_runs({"rocke_pytest_core": 100.0}, minutes=(28, 30, 29))
        self.gh.add(
            HEAD,
            S1,
            1,
            "failure",
            POOL,
            "2026-11-10T02:00:00Z",
            "2026-11-10T02:30:00Z",
            ctest({"rocke_pytest_core": 103.0}, started=["rocke_conv_pytest"]),
        )
        self.gh.add(
            HEAD,
            S2,
            2,
            "success",
            POOL,
            "2026-11-10T02:00:00Z",
            "2026-11-10T02:25:00Z",
            ctest({"rocke_pytest_core": 99.0}),
        )
        lines = self.report()
        self.assertEqual(lines[-1], "VERDICT pool (develop same pool 28–30 min)")
        self.assertIn(
            "1 shard job(s), failed or timed out", lines[0]
        )  # only the failed one
        self.assertIn("unfinished at the cut: rocke_conv_pytest", "\n".join(lines))
        self.assertIn("2 shard job(s), every shard", self.report(every=True)[0])

    def test_another_pool_or_an_old_run_is_not_compared(self):
        self.develop_runs({"rocke_pytest_core": 100.0}, labels=["linux-pool-b"])
        self.gh.add(
            HEAD,
            S1,
            1,
            "failure",
            POOL,
            "2026-11-12T02:00:00Z",
            "2026-11-12T02:30:00Z",
            ctest({"rocke_pytest_core": 300.0}),
        )
        lines = self.report()
        self.assertTrue(lines[-1].startswith("VERDICT not compared"), lines)
        self.assertIn(
            "develop: no run of this shard on its pool within 24 h", "\n".join(lines)
        )

    def test_reads_are_gets_and_a_failure_says_so(self):
        self.gh.add(
            HEAD,
            S1,
            1,
            "failure",
            POOL,
            "2026-11-10T02:00:00Z",
            "2026-11-10T02:30:00Z",
            "x",
        )
        del self.gh.logs[1]
        lines = self.report()
        self.assertIn(f"{S1}: failure; log not read", lines)
        with mock.patch.object(
            shard_times.gh, "rest", side_effect=gh.GhError("HTTP 502")
        ):
            lines, ok = shard_times.report(CFG, 7)
        self.assertFalse(ok)
        self.assertTrue(lines[0].startswith("(not read: HTTP 502"))

    def test_the_rk_command(self):
        import rkmon

        self.develop_runs({"rocke_pytest_core": 100.0})
        self.gh.add(
            HEAD,
            S1,
            1,
            "failure",
            POOL,
            "2026-11-10T02:00:00Z",
            "2026-11-10T02:30:00Z",
            ctest({"rocke_pytest_core": 100.0}),
        )
        out = io.StringIO()
        sandbox.own_home(self)  # the command's store is there
        with mock.patch.object(
            rkmon, "load_config", lambda: dict(CFG)
        ), contextlib.redirect_stdout(out):
            rc = rkmon.main(["shard-times", "7"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("VERDICT pool", out.getvalue())


if __name__ == "__main__":
    unittest.main()
