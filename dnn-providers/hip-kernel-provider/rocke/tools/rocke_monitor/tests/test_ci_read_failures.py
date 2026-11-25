# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The CI tools the checks run under `net` (rkmon_lib/review.py, NET_FN) exit 1 when a GitHub read failed, with gh's
whole error where net reads it, so a read that failed on the network runs once more: tools/ci_head.py (the check runs
at the head or at a develop commit, a job's log), tools/ci_sha.py (the runs, a run's job list, a job's details or log),
tools/ci_attempts.py (the runs, a run's jobs: its lines and exit codes in tests/test_ci_attempts.py) and
tools/prthreads.py (the three lists). A read that works exits 0. gh is faked; no network.
"""

import contextlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))
import ci_attempts  # noqa: E402
import ci_head  # noqa: E402
import ci_sha  # noqa: E402
import prthreads  # noqa: E402
from rkmon_lib import gh, review  # noqa: E402

HEAD, DEV = "a" * 40, "d" * 40
NETWORK = "error connecting to api.github.com"
# gh's errors for other reads that failed on the network: each starts with the request's address (Go's `Get "URL":`),
# which fills the start of the error a tool's line shows.
URL = f"https://api.github.com/repos/o/r/commits/{HEAD}/check-runs?per_page=100"
URL_FIRST = {
    "a TLS handshake timeout": f'Get "{URL}": net/http: TLS handshake timeout',
    "a refused connection": f'Get "{URL}": dial tcp: connect: connection refused',
    "a reset connection": f'Get "{URL}": read tcp: read: connection reset by peer',
}
# What net looks for in the output and the errors of a command that failed (grep -iE).
NET = re.compile(review.NET_ERRORS, re.I)
# gh that fails as it does when the network is down: its error on stderr, exit 1.
GH_DOWN = [
    sys.executable,
    "-c",
    f"import sys; sys.stderr.write({NETWORK!r} + '\\n'); sys.exit(1)",
]


def gh_error(err: str, path: str) -> gh.GhError:
    """gh's error ERR for a read of PATH, as gh.rest and gh.raw raise it."""
    return gh.GhError(f"gh api: {err} [{path}]")


def failed_run(name: str, job: int) -> dict:
    """A check run of the job JOB that failed."""
    return {
        "name": name,
        "status": "completed",
        "conclusion": "failure",
        "started": "2026-11-10T01:00:00Z",
        "completed": "2026-11-10T01:00:00Z",
        "url": f"https://github.com/ROCm/r/actions/runs/1/job/{job}",
    }


def answer(got):
    """A faked read's answer: GOT, raised when it is an exception."""
    if isinstance(got, Exception):
        raise got
    return got


class TestCiHead(unittest.TestCase):
    def main(self, runs, log=lambda path: "", develop=()) -> tuple[int, list[str], str]:
        """ci_head at HEAD with DEVELOP's commits, its two GitHub reads faked: RUNS(sha) the check runs at a commit and
        LOG(path) a job's log, each an exception for a read that fails. Its exit code, its lines and its stderr; the
        commits whose check runs it asked for are in self.asked."""
        self.asked = []

        def check_runs(cfg, sha):
            self.asked.append(sha)
            return answer(runs(sha))

        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(
            ci_head.store, "load_config", return_value={"repo": "o/r"}
        ), mock.patch.object(ci_head.gh, "configure"), mock.patch.object(
            ci_head.ci_runs, "check_runs", check_runs
        ), mock.patch.object(
            ci_head.gh, "raw", lambda path, timeout=120: answer(log(path))
        ), contextlib.redirect_stdout(
            out
        ), contextlib.redirect_stderr(
            err
        ):
            code = ci_head.main([HEAD, "--develop", *develop])
        return code, out.getvalue().splitlines(), err.getvalue()

    def test_check_runs_not_read_exit_1_with_gh_s_error_on_the_first_line_and_on_stderr(
        self,
    ):
        exc = gh_error(NETWORK, URL)
        code, lines, err = self.main(lambda sha: exc)
        self.assertEqual(code, 1)
        self.assertEqual(lines[-1], ci_head.NOT_READ)
        self.assertIn(NETWORK, lines[0])
        self.assertEqual(err, f"{exc}\n")

    def test_an_error_that_starts_with_the_address_reaches_net_whole(self):
        # The first line shows only the start of gh's error, which the address fills; stderr has the error whole.
        for name, text in URL_FIRST.items():
            with self.subTest(name):
                exc = gh_error(text, URL)
                code, lines, err = self.main(lambda sha: exc)
                self.assertEqual(code, 1)
                self.assertEqual(err, f"{exc}\n")
                self.assertRegex("\n".join(lines) + "\n" + err, NET)

    def test_a_log_not_read_exit_1_with_gh_s_error_on_stderr(self):
        exc = gh_error(NETWORK, "repos/o/r/actions/jobs/1/logs")
        code, lines, err = self.main(
            lambda sha: [failed_run("unit", 1)], lambda path: exc
        )
        self.assertEqual(code, 1)
        self.assertEqual(
            lines[0],
            "FAILED unit: log not read (https://github.com/ROCm/r/actions/runs/1/job/1)",
        )
        self.assertEqual(err, f"{exc}\n")

    def test_develops_runs_not_read_exit_1_and_are_asked_for_once(self):
        exc = gh_error(NETWORK, f"repos/o/r/commits/{DEV}/check-runs")
        code, lines, err = self.main(
            lambda sha: (
                [failed_run("lint", 1), failed_run("unit", 2)] if sha == HEAD else exc
            ),
            lambda path: "##[error]Process completed with exit code 1.\n",
            develop=[DEV],
        )
        self.assertEqual(code, 1)
        self.assertEqual(
            [ln for ln in lines if "not read" in ln],
            [f"   develop {DEV[:10]}: not read ({str(exc)[:80]})"] * 2,
        )
        self.assertEqual(self.asked, [HEAD, DEV])
        self.assertEqual(err, f"{exc}\n")

    def test_every_read_works_exit_0(self):
        self.assertEqual(
            self.main(lambda sha: []),
            (0, ["CIHEAD 0 failing, 0 known infrastructure, 0 re-run"], ""),
        )


RUNS = b"11\tBuild and test\tcompleted\tfailure\t2026-11-01T00:00:00Z"
JOBS = b"101\tLinux / unit tests (shard 1)"


class TestCiSha(unittest.TestCase):
    def run_tool(self, ghget=None, gh_cmd=GH_DOWN) -> tuple[int, list, str]:
        """The tool at HEAD, gh being GH_CMD and its reads with retry GHGET (the tool's own when None): (its exit code,
        the lines it said, what it wrote on stderr)."""
        said, err = [], io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "repo": "o/r",
                "develop_watched_workflows": ["Build and test"],
                "develop_rocke_job_pattern": "",
                "paths": {"scratch": d},
            }
            with mock.patch.object(
                ci_sha.store, "load_config", return_value=cfg
            ), mock.patch.object(ci_sha, "gh", return_value=gh_cmd), mock.patch.object(
                ci_sha, "ghget", ghget or ci_sha.ghget
            ), mock.patch.object(
                ci_sha.time, "sleep"
            ), mock.patch.object(
                ci_sha, "say", said.append
            ), contextlib.redirect_stderr(
                err
            ):
                code = ci_sha.main(["ci_sha.py", HEAD, "FAILED "])
        return code, said, err.getvalue()

    def test_runs_not_read_exit_1_with_gh_s_error_on_stderr(self):
        code, said, err = self.run_tool()
        self.assertEqual(code, 1)
        self.assertEqual(
            said, [f"-- CI runs for {HEAD}: not read (GitHub API error)".encode()]
        )
        self.assertIn(NETWORK, err)

    def test_a_job_list_not_read_exit_1(self):
        def ghget(*args):
            return RUNS if "/actions/runs?" in args[0] else None

        code, said, _ = self.run_tool(ghget)
        self.assertEqual(code, 1)
        self.assertIn(b"-- run 11: job list not read (GitHub API error)", said)

    def test_a_log_not_read_exit_1(self):
        def ghget(*args):
            if "/actions/runs?" in args[0]:
                return RUNS
            if "/jobs?" in args[0]:
                return JOBS
            return b"unit [failure, attempt 1]"

        code, said, err = self.run_tool(ghget)
        self.assertEqual(code, 1)
        self.assertIn(
            b"-- job 101: unit [failure, attempt 1] (log not available)", said
        )
        self.assertIn(NETWORK, err)

    def test_every_read_works_exit_0(self):
        def ghget(*args):
            if "/actions/runs?" in args[0]:
                return RUNS
            if "/jobs?" in args[0]:
                return JOBS
            return b"unit [failure, attempt 1]"

        log = [sys.executable, "-c", "print('FAILED tests/test_a.py::test_x')"]
        code, said, _ = self.run_tool(ghget, log)
        self.assertEqual(code, 0)
        self.assertIn(b"     FAILED tests/test_a.py::test_x", said)


class TestPrthreads(unittest.TestCase):
    def run_tool(self, ghr) -> tuple[int, list[str]]:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"RKMON_REPO": "o/r", "RKMON_SCRATCH": d}
        ), mock.patch.object(prthreads, "GHR", ghr), contextlib.redirect_stdout(out):
            code = prthreads.main(["prthreads.py", "7", "2026-11-01T00:00:00Z"])
        return code, out.getvalue().splitlines()

    def test_lists_not_read_exit_1_and_each_is_named(self):
        code, lines = self.run_tool([sys.executable, "-c", "import sys; sys.exit(1)"])
        self.assertEqual(code, 1)
        self.assertEqual(
            lines,
            [
                "== review comments (threads): not read (GitHub API error)",
                "== issue comments: not read (GitHub API error)",
                "== reviews: not read (GitHub API error)",
            ],
        )

    def test_lists_not_kept_or_gh_not_run_say_so_exit_1(self):
        # The scratch folder is a file: no list can be kept, and gh is never run.
        names = ["review comments (threads)", "issue comments", "reviews"]
        with tempfile.TemporaryDirectory() as d:
            scratch = pathlib.Path(d, "scratch")
            scratch.write_text("")
            out = io.StringIO()
            with mock.patch.dict(
                os.environ, {"RKMON_REPO": "o/r", "RKMON_SCRATCH": str(scratch)}
            ), mock.patch.object(
                prthreads.subprocess, "run"
            ) as run, contextlib.redirect_stdout(
                out
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                code = prthreads.main(["prthreads.py", "7", "2026-11-01T00:00:00Z"])
            self.assertEqual(code, 1)
            self.assertEqual(
                out.getvalue().splitlines(),
                [
                    f"== {name}: not kept ({scratch}/thr7/{f}: Not a directory)"
                    for name, f in zip(names, ["rc.json", "ic.json", "rv.json"])
                ],
            )
            run.assert_not_called()
        code, lines = self.run_tool([os.path.join(TOOLS, "no-such-gh")])
        self.assertEqual(code, 1)
        self.assertEqual(
            lines,
            [
                f"== {name}: not read (gh did not run: No such file or directory)"
                for name in names
            ],
        )

    def test_lists_read_exit_0(self):
        code, lines = self.run_tool([sys.executable, "-c", "print('[]')"])
        self.assertEqual(
            (code, lines),
            (
                0,
                [
                    "== review comments (threads)",
                    "== issue comments",
                    "== reviews with bodies",
                ],
            ),
        )


class TestCiAttempts(unittest.TestCase):
    def fetch(self, done) -> tuple[dict | None, str]:
        """ci_attempts.fetch with gh's run DONE (a CompletedProcess, or an exception it raises): what it returns, and
        what it wrote on stderr."""
        err = io.StringIO()
        with mock.patch.object(
            ci_attempts.subprocess,
            "run",
            (
                mock.Mock(side_effect=done)
                if isinstance(done, Exception)
                else mock.Mock(return_value=done)
            ),
        ), mock.patch.object(ci_attempts.time, "sleep"), contextlib.redirect_stderr(
            err
        ):
            got = ci_attempts.fetch("repos/o/r/actions/runs?head_sha=a")
        return got, err.getvalue()

    def test_a_read_that_fails_writes_gh_s_whole_error_on_stderr_each_time(self):
        for name, text in {"DNS": NETWORK, **URL_FIRST}.items():
            with self.subTest(name):
                got, err = self.fetch(
                    subprocess.CompletedProcess([], 1, "", text + "\n")
                )
                self.assertIsNone(got)
                self.assertEqual(err, f"{text}\n{text}\n")
                self.assertRegex(err, NET)

    def test_gh_that_does_not_run_or_answers_no_json_says_so_on_stderr(self):
        got, err = self.fetch(FileNotFoundError(2, "No such file or directory"))
        self.assertEqual((got, err.count("No such file or directory")), (None, 2))
        got, err = self.fetch(subprocess.CompletedProcess([], 0, "Server Error", ""))
        self.assertIsNone(got)
        self.assertIn("repos/o/r/actions/runs?head_sha=a: the answer is not JSON", err)

    def test_a_read_that_works_writes_nothing_on_stderr(self):
        self.assertEqual(
            self.fetch(subprocess.CompletedProcess([], 0, '{"workflow_runs": []}', "")),
            ({"workflow_runs": []}, ""),
        )


if __name__ == "__main__":
    unittest.main()
