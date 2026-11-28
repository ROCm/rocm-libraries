# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ci_sha.py: at a commit, the runs of the watched workflows and the logs of their jobs that test the watched
folder, as the configuration names them (develop_watched_workflows, develop_rocke_job_pattern); the tool names none of
its own. gh is faked; no network."""

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci_sha.py"
sys.path.insert(0, str(TOOL.parent))
import ci_sha  # noqa: E402
from rkmon_lib import config  # noqa: E402

RUNS = (
    b"11\tBuild and test\tcompleted\tfailure\t2026-11-01T00:00:00Z\n"
    b"12\tBuild and test\tcompleted\tcancelled\t2026-11-01T00:00:00Z\n"
    b"13\tLint\tcompleted\tsuccess\t2026-11-01T00:00:00Z"
)
JOBS = (
    b"101\tLinux / unit tests (shard 1)\n"
    b"102\tLinux / build\n"
    b"103\tWindows / Unit Tests (shard 2)"
)
LOG = b"2026-11-01T00:00:00Z FAILED tests/test_a.py::test_x\nok\n"


class TestConfigured(unittest.TestCase):
    def run_tool(self, job_pattern: str) -> tuple[list, list, int]:
        """The tool at a commit, with a configuration that watches "Build and test" and takes JOB_PATTERN for the
        jobs that test the folder: (the GETs it made, the lines it said, its exit code).
        """
        asked, said = [], []

        def ghget(*args):
            asked.append(args[0])
            if "/actions/runs?" in args[0]:
                return RUNS
            if args[0].endswith("/jobs?filter=all&per_page=50"):
                return JOBS
            return b"a job [failure, attempt 1]"

        with tempfile.TemporaryDirectory() as d:
            logs = pathlib.Path(d) / "ci_sha"
            logs.mkdir()
            for job in ("101", "102", "103"):
                (logs / f"job_{job}.log").write_bytes(LOG)
            cfg = {
                "repo": "o/r",
                "develop_watched_workflows": ["Build and test"],
                "develop_rocke_job_pattern": job_pattern,
                "paths": {"scratch": d},
            }
            with mock.patch.object(
                ci_sha.store, "load_config", return_value=cfg
            ), mock.patch.object(ci_sha, "ghget", ghget), mock.patch.object(
                ci_sha, "say", said.append
            ):
                code = ci_sha.main(["ci_sha.py", "a" * 40, "FAILED "])
        return asked, said, code

    def test_the_runs_and_the_jobs_are_those_the_configuration_names(self):
        """The watched workflow's runs are listed and the one not cancelled is read; of its jobs, those the job
        pattern matches without regard to case, whose logs are grepped."""
        asked, said, code = self.run_tool("unit")
        self.assertEqual(code, 0)
        self.assertEqual(
            [line for line in said if b"\tcompleted\t" in line], RUNS.split(b"\n")[:2]
        )
        self.assertEqual(
            [a for a in asked if "/jobs?" in a],
            ["repos/o/r/actions/runs/11/jobs?filter=all&per_page=50"],
        )
        self.assertEqual(
            [a for a in asked if "/actions/jobs/" in a],
            ["repos/o/r/actions/jobs/101", "repos/o/r/actions/jobs/103"],
        )
        self.assertEqual(said.count(b"     FAILED tests/test_a.py::test_x"), 2, said)

    def test_a_blank_job_pattern_reads_every_job_of_the_watched_runs(self):
        asked, said, code = self.run_tool("")
        self.assertEqual(
            [a for a in asked if "/actions/jobs/" in a],
            [f"repos/o/r/actions/jobs/{job}" for job in (101, 102, 103)],
        )

    def test_it_refuses_when_the_configuration_cannot_be_read(self):
        out = io.StringIO()
        with mock.patch.object(
            ci_sha.store,
            "load_config",
            side_effect=config.NoConfiguration("no configuration at /x/config.json"),
        ), contextlib.redirect_stdout(out):
            self.assertEqual(ci_sha.main(["ci_sha.py", "a" * 40, "x"]), 4)
        self.assertTrue(out.getvalue().startswith("REFUSED"), out.getvalue())


if __name__ == "__main__":
    unittest.main()
