# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ci_attempts.py: a re-run hides its earlier attempt's failed jobs from the check-run view (failures in
attempt 1 behind a hand re-run still make the head red, never infra; L-108)."""

import contextlib
import io
import pathlib
import sys
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci_attempts.py"
sys.path.insert(0, str(TOOL.parent))
import ci_attempts  # noqa: E402

RUN = {"id": 41659828671, "name": "TheRock Multi-Arch CI", "run_attempt": 2}
SHARD1 = "Windows::release / Test gfx110X-all / Test hipkernelprovider (shard 1/2) (gfx110X-all)"
SHARD2 = "Windows::release / Test gfx110X-all / Test hipkernelprovider (shard 2/2) (gfx110X-all)"
MATHLIBS = "Linux::release / Build Multi-Arch Stages / math-libs (gfx125X-dcgpu) / Stage - Math Libs (gfx125X-dcgpu)"


def job(name, attempt, conclusion=None, status="completed"):
    return {
        "name": name,
        "run_attempt": attempt,
        "conclusion": conclusion,
        "status": status,
    }


class TestEarlierFailures(unittest.TestCase):
    def test_attempt_one_failures_show_with_their_rerun_state(self):
        jobs = [
            job(SHARD1, 1, "failure"),
            job(SHARD2, 1, "failure"),
            job("pre-commit", 1, "success"),
            job(SHARD1, 2, status="in_progress"),
            job(SHARD2, 2, status="queued"),
            job(MATHLIBS, 2, "failure"),
        ]
        out = ci_attempts.earlier_failures(RUN, jobs)
        self.assertEqual(
            out,
            [
                f"FAILED in attempt 1 of 2 (TheRock Multi-Arch CI): {SHARD1}; attempt 2: in_progress",
                f"FAILED in attempt 1 of 2 (TheRock Multi-Arch CI): {SHARD2}; attempt 2: queued",
            ],
        )

    def test_the_latest_attempts_own_failures_are_not_repeated(self):
        """The check-run view already lists the latest attempt; only what it hides is printed."""
        self.assertEqual(
            ci_attempts.earlier_failures(RUN, [job(MATHLIBS, 2, "failure")]), []
        )

    def test_a_failure_the_rerun_passed_says_so(self):
        """A job that passed only on a re-run is a flake, tagged so; one still failing is not."""
        out = ci_attempts.earlier_failures(
            RUN, [job(SHARD1, 1, "timed_out"), job(SHARD1, 2, "success")]
        )
        self.assertEqual(
            out,
            [
                f"FAILED in attempt 1 of 2 (TheRock Multi-Arch CI): {SHARD1}; attempt 2: success; "
                "FLAKE: passed only on a re-run"
            ],
        )
        out = ci_attempts.earlier_failures(
            RUN, [job(SHARD1, 1, "failure"), job(SHARD1, 2, "failure")]
        )
        self.assertNotIn("FLAKE", out[0])

    def test_a_job_missing_from_the_latest_attempt_says_absent(self):
        out = ci_attempts.earlier_failures(RUN, [job(SHARD1, 1, "failure")])
        self.assertTrue(
            out[0].endswith("attempt 2: absent (not re-run, or not started yet)"), out
        )


class TestPaging(unittest.TestCase):
    def test_pages_until_the_total(self):
        pages = {
            1: [job(f"j{i}", 1, "success") for i in range(50)],
            2: [job("j50", 1, "failure")],
        }
        seen = []

        def get(path):
            seen.append(path)
            n = int(path.rsplit("page=", 1)[1])
            return {"total_count": 51, "jobs": pages[n]}

        jobs = ci_attempts.all_jobs("ROCm/rocm-libraries", 1, get)
        self.assertEqual(len(jobs), 51)
        self.assertTrue(all("filter=all" in p for p in seen), seen)

    def test_an_unreadable_page_is_none(self):
        self.assertIsNone(
            ci_attempts.all_jobs("ROCm/rocm-libraries", 1, lambda path: None)
        )


class TestMain(unittest.TestCase):
    def run_main(self, get, argv=("d3d5e177b7", "o/r")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ci_attempts.main(["ci_attempts.py", *argv], get)
        return rc, buf.getvalue()

    def test_the_repository_is_the_one_given(self):
        """The runs are read in the repository the caller names (the checks pass the configuration's); the tool
        names none of its own, and without one it says how to run it."""
        asked = []
        self.assertEqual(
            self.run_main(lambda path: asked.append(path) or {"workflow_runs": []}),
            (0, ""),
        )
        self.assertEqual(
            asked, ["repos/o/r/actions/runs?head_sha=d3d5e177b7&per_page=100"]
        )
        self.assertEqual(
            self.run_main(asked.append, ("d3d5e177b7",)),
            (2, "usage: python -I ci_attempts.py SHA OWNER/REPO\n"),
        )
        self.assertEqual(len(asked), 1)

    def test_runs_without_a_rerun_print_nothing(self):
        def get(path):
            if "head_sha=" in path:
                return {
                    "workflow_runs": [
                        {"id": 1, "name": "TheRock Multi-Arch CI", "run_attempt": 1}
                    ]
                }
            raise AssertionError(f"a run with one attempt needs no job list: {path}")

        self.assertEqual(self.run_main(get), (0, ""))

    def test_a_rerun_prints_its_hidden_failures(self):
        def get(path):
            if "head_sha=" in path:
                return {"workflow_runs": [RUN]}
            return {
                "total_count": 2,
                "jobs": [
                    job(SHARD1, 1, "failure"),
                    job(SHARD1, 2, status="in_progress"),
                ],
            }

        rc, out = self.run_main(get)
        self.assertEqual(rc, 0)
        self.assertIn(
            f"FAILED in attempt 1 of 2 (TheRock Multi-Arch CI): {SHARD1}; attempt 2: in_progress",
            out,
        )

    def test_an_api_error_says_not_read_exit_1(self):
        rc, out = self.run_main(lambda path: None)
        self.assertEqual(rc, 1)
        self.assertIn("not read (GitHub API error)", out)

    def test_a_job_list_not_read_says_so_exit_1(self):
        def get(path):
            return {"workflow_runs": [RUN]} if "head_sha=" in path else None

        self.assertEqual(
            self.run_main(get),
            (
                1,
                "-- run 41659828671 (TheRock Multi-Arch CI): earlier attempts not read (GitHub API error)\n",
            ),
        )


if __name__ == "__main__":
    unittest.main()
