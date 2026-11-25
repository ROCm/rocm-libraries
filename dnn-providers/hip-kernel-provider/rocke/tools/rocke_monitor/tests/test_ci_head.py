# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ci_head.py: CI at head per failing job: first error, failing tests, develop's latest run of the job, a
known-infrastructure label, re-runs marked (agent_script_overlap.md, fix 1)."""

import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))
import ci_head  # noqa: E402
from rkmon_lib import gh  # noqa: E402

CFG = {"repo": "o/r", "ci_infra_signatures": ["pwsh: command not found"]}
HEAD, DEV1, DEV2 = "a" * 40, "d" * 40, "e" * 40


def run(name, conclusion, job, started="2026-11-10T01:00:00Z", status="completed"):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started": started,
        "completed": started,
        "url": f"https://github.com/ROCm/r/actions/runs/1/job/{job}",
    }


class FakeReader:
    def __init__(self, runs: dict, logs: dict, broken=()):
        self.runs_by_sha, self.logs, self.broken = runs, logs, set(broken)

    def runs(self, sha):
        if sha in self.broken:
            raise gh.GhError("HTTP 502")
        return self.runs_by_sha.get(sha, [])

    def log(self, r):
        return self.logs.get(int(r["url"].rsplit("/", 1)[1]))


PYTEST_LOG = (
    "2026-11-10T01:02:03.1234567Z ##[group]Run tests\n"
    "2026-11-10T01:02:04.1234567Z FAILED tests/test_a.py::test_one - AssertionError: x\n"
    "2026-11-10T01:02:04.1234567Z FAILED tests/test_a.py::test_two[p] - AssertionError: y\n"
    "2026-11-10T01:02:05.1234567Z ##[error]\x1b[31mProcess tests failed: 2 failed\x1b[0m\n"
    "2026-11-10T01:02:06.1234567Z ##[error]Process completed with exit code 1.\n"
)
CTEST_LOG = (
    "The following tests FAILED:\n"
    "\t 12 - rocke_pytest (Timeout)\n"
    "\t 40 - rocke_library_gdn_pytest (Failed)\n"
    "Test #41: rocke_conv_pytest ..........***Failed  3.2 sec\n"
    "##[error]The action 'Test' has timed out after 30 minutes.\n"
)


class TestCiHead(unittest.TestCase):
    def test_each_failing_job_has_its_first_error_tests_develops_run_and_a_label(self):
        reader = FakeReader(
            {
                HEAD: [
                    run("unit", "failure", 1),
                    run("shard", "failure", 2, "2026-11-10T00:00:00Z"),
                    run("shard", "failure", 3),
                    run("win", "failure", 4),
                    run("setup", "failure", 5, "2026-11-10T00:00:00Z"),
                    run("setup", "success", 6),
                    run("lint", "success", 7),
                ],
                DEV1: [
                    run("shard", "-", 8, status="in_progress")
                ],  # still running: the next develop commit is read
                DEV2: [run("unit", "success", 9), run("shard", "failure", 10)],
            },
            {
                1: PYTEST_LOG,
                3: CTEST_LOG,
                4: "##[error]pwsh: command not found\n",
                10: CTEST_LOG,
            },
        )
        out = ci_head.report(CFG, HEAD, [DEV1, DEV2], reader)
        self.assertEqual(
            out,
            [
                "FAILED shard (attempt 2 of 2: a re-run): The action 'Test' has timed out after 30 minutes.",
                "   failing tests (3): rocke_pytest, rocke_library_gdn_pytest, rocke_conv_pytest",
                "   develop eeeeeeeeee: failure, the same first error",
                "   known infrastructure (develop fails the same); job https://github.com/ROCm/r/actions/runs/1/job/3",
                "FAILED unit: Process tests failed: 2 failed",
                "   failing tests (2): tests/test_a.py::test_one, tests/test_a.py::test_two[p]",
                "   develop eeeeeeeeee: success",
                "   not known infrastructure; job https://github.com/ROCm/r/actions/runs/1/job/1",
                "FAILED win: pwsh: command not found",
                "   develop: no completed run of this job in its last 2 commits",
                "   known infrastructure (signature `pwsh: command not found`); job https://github.com/ROCm/r/actions/runs/1/job/4",
                "failing tests by name (5, from the failing jobs' logs):",
                "   rocke_conv_pytest: 1 job: shard",
                "   rocke_library_gdn_pytest: 1 job: shard",
                "   rocke_pytest: 1 job: shard",
                "   tests/test_a.py::test_one: 1 job: unit",
                "   tests/test_a.py::test_two[p]: 1 job: unit",
                "re-run: setup (attempt 2 of 2): success",
                "CIHEAD 3 failing, 2 known infrastructure, 2 re-run",
            ],
        )

    def test_the_failing_tests_by_name_across_shards(self):
        """A failure in a test the PR does not change (a C++ mirror test, CTest entry 87), on two shards, is listed by
        test name across the failing jobs, not only inside each job's lines."""
        linux = "Linux::release / Test gfx94X-dcgpu / Test hipkernelprovider / Test hipkernelprovider (shard 1/2) (gfx94X)"
        win = "Windows::release / Test gfx110X-all / Test hipkernelprovider / Test hipkernelprovider (shard 2/2) (gfx110X)"
        mirror = "The following tests FAILED:\n\t 87 - rocke_dense_spec_cpp_mirror (Failed)\n"
        reader = FakeReader(
            {HEAD: [run(linux, "failure", 1), run(win, "failure", 2)]},
            {1: mirror + "\t 12 - rocke_pytest (Timeout)\n", 2: mirror},
        )
        out = ci_head.report(CFG, HEAD, [], reader)
        i = out.index("failing tests by name (2, from the failing jobs' logs):")
        self.assertEqual(
            out[i + 1 : i + 3],
            [
                "   rocke_dense_spec_cpp_mirror: 2 jobs: Test hipkernelprovider (shard 1/2) (gfx94X); Test hipkernelprovider "
                "(shard 2/2) (gfx110X)",
                "   rocke_pytest: 1 job: Test hipkernelprovider (shard 1/2) (gfx94X)",
            ],
        )
        self.assertEqual(ci_head.by_test_lines({}), [])

    def test_a_read_that_fails_says_so(self):
        reader = FakeReader({HEAD: [run("unit", "failure", 1)]}, {}, broken=[DEV1])
        out = ci_head.report(CFG, HEAD, [DEV1], reader)
        self.assertEqual(
            out[0],
            "FAILED unit: log not read (https://github.com/ROCm/r/actions/runs/1/job/1)",
        )
        reader = FakeReader(
            {HEAD: [run("unit", "failure", 1)]}, {1: PYTEST_LOG}, broken=[DEV1]
        )
        self.assertTrue(
            ci_head.report(CFG, HEAD, [DEV1], reader)[2].startswith(
                "   develop dddddddddd: not read (HTTP"
            )
        )
        out = ci_head.report(CFG, HEAD, [DEV1], FakeReader({}, {}, broken=[HEAD]))
        self.assertEqual(out[-1], "CIHEAD not read")
        self.assertTrue(
            out[0].startswith("NOT RUN: the check runs at aaaaaaaaaa were not read")
        )

    def test_the_checks_script_runs_it_with_develops_newest_commits(self):
        from rkmon_lib import review

        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(
            {
                "rocke_path": "rk",
                "repo": "o/r",
                "paths": {"working_dir": "/w"},
            },
            7,
            HEAD,
            "follow-up",
            plan,
            None,
        )
        sec = script.split("sec 'CI at head'", 1)[1].split("\nsec ", 1)[0]
        # the base branch's newest commits: develop's here (BREF), a feature branch's for a PR into one
        self.assertIn("BREF=refs/remotes/origin/develop\n", script)
        self.assertIn("DVS=$(git -C $R0 rev-list -n 6 $BREF ", sec)
        self.assertIn(f"python -I -B $M/tools/ci_head.py {HEAD} --develop $DVS", sec)
        self.assertIn(
            "ci_attempts.py", sec
        )  # earlier attempts' failures behind a re-run stay listed


if __name__ == "__main__":
    unittest.main()
