# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk ci-fails`, `rk job-log` and `rk heads`. The orchestrator copies their lines into a post spec's FAILOK and
RUNOK and into briefs, so each line keeps its shape."""

import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import ci_runs, gh  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402

CFG = {
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "deployment": {"phase": "review"},
    "merge_authority": "propose",
}
HEAD = "c" * 40
URL = "https://github.com/ROCm/r/runs/"


def run(name, status, conclusion, started, k):
    """A check run as ci_runs.check_runs' jq leaves it."""
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started": started,
        "url": f"{URL}{k}",
    }


# GitHub's order. A re-run that has not started does not replace a started attempt (pre-commit), a later attempt does
# (Linux / build), one that started at the same time does not (dup); a cancelled latest attempt is no failure.
RUNS = [
    run("Linux / build", "completed", "failure", "2026-11-05T10:00:00Z", 1),
    run("Linux / build", "completed", "success", "2026-11-05T11:00:00Z", 2),
    run("Windows / stage", "completed", "failure", "2026-11-05T10:00:00Z", 3),
    run("Linux / test (shard 1/2)", "in_progress", "-", "2026-11-05T10:30:00Z", 4),
    run("pre-commit", "completed", "success", "2026-11-05T10:00:00Z", 5),
    run("pre-commit", "queued", "-", "", 6),
    run("Math CI Summary", "completed", "timed_out", "2026-11-05T10:05:00Z", 7),
    run("Windows / sanity", "queued", "-", "", 8),
    run("bot", "completed", "action_required", "2026-11-05T10:00:00Z", 9),
    run("label", "completed", "cancelled", "2026-11-05T10:00:00Z", 10),
    run("dup", "completed", "failure", "2026-11-05T10:00:00Z", 11),
    run("dup", "completed", "success", "2026-11-05T10:00:00Z", 12),
]


def cli(argv, **patches):
    """rkmon.main(argv) with gh's functions replaced by `patches`: (exit code, printed lines)."""
    out = []
    with tempfile.TemporaryDirectory() as d:
        store = Store(pathlib.Path(d))
        with mock.patch.object(rkmon, "load_config", lambda: CFG), mock.patch.object(
            rkmon, "Store", lambda: store
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            with mock.patch.multiple(gh, **patches):
                code = rkmon.main(argv)
        assert (
            not store.ledger_path.exists() and not store.journal_path.exists()
        )  # read-only
    return code, "\n".join(out).split("\n")


class TestCiFails(unittest.TestCase):
    def test_latest_attempts_by_name(self):
        calls = []

        def rest(path, jq=None, paginate=False, timeout=120):
            calls.append((path, paginate))
            if path == "repos/ROCm/r/pulls/7":
                self.assertEqual(jq, ".head.sha")
                return HEAD
            self.assertIn('started: (.started_at // "")', jq)
            return list(RUNS)

        code, lines = cli(["ci-fails", "7"], rest=rest)
        self.assertEqual(code, 0)
        self.assertEqual(
            lines,
            [
                f"head {HEAD}",
                f"RUN\tLinux / test (shard 1/2)\tin_progress\t{URL}4",
                f"FAIL\tMath CI Summary\ttimed_out\t{URL}7",
                f"RUN\tWindows / sanity\tqueued\t{URL}8",
                f"FAIL\tWindows / stage\tfailure\t{URL}3",
                f"FAIL\tbot\taction_required\t{URL}9",
                f"FAIL\tdup\tfailure\t{URL}11",
                "runs 12 names 9",
            ],
        )
        self.assertEqual(
            calls,
            [
                ("repos/ROCm/r/pulls/7", False),
                (f"repos/ROCm/r/commits/{HEAD}/check-runs?per_page=100", True),
            ],
        )

    def test_the_picture_premises_read(self):
        last = ci_runs.latest(RUNS)
        self.assertEqual(
            list(last),
            [
                "Linux / build",
                "Windows / stage",
                "Linux / test (shard 1/2)",
                "pre-commit",
                "Math CI Summary",
                "Windows / sanity",
                "bot",
                "label",
                "dup",
            ],
        )  # first-seen order
        self.assertEqual(
            [k for k, r in last.items() if ci_runs.failed(r)],
            ["Windows / stage", "Math CI Summary", "bot", "dup"],
        )
        self.assertEqual(
            [k for k, r in last.items() if ci_runs.running(r)],
            ["Linux / test (shard 1/2)", "Windows / sanity"],
        )
        self.assertEqual(last["pre-commit"]["url"], f"{URL}5")

    def test_a_read_failure_is_said_and_exits_1(self):
        def rest(path, jq=None, paginate=False, timeout=120):
            if paginate:
                raise gh.GhError("gh api --paginate: HTTP 502")
            return HEAD

        self.assertEqual(
            cli(["ci-fails", "7"], rest=rest),
            (1, [f"head {HEAD}", "(check runs not read: gh api --paginate: HTTP 502)"]),
        )


class TestJobLog(unittest.TestCase):
    def test_count_then_failing_lines_and_the_log_is_kept(self):
        log = (
            "2026-11-05T10:00:00.1Z ##[group]Run tests\n"
            "Error: the runner lost its connection\n"
            "ok\n"
            "summary: 3 warnings, error: 0\n"  # left out
            "***Failed    1.20 sec\n"
            + "x" * 300
            + " FAILED\n"
            + "".join(f"test_{i} FAILED\n" for i in range(30))
            + "tail without a newline: timed out"
        )
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"TMPDIR": d}
        ):
            code, lines = cli(
                ["job-log", "127262331729"],
                raw=lambda path, timeout=120: (
                    log
                    if path == "repos/ROCm/r/actions/jobs/127262331729/logs"
                    else self.fail(path)
                ),
            )
            self.assertEqual(
                (pathlib.Path(d) / "joblog_127262331729.txt").read_text(
                    encoding="utf-8"
                ),
                log,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            lines[0], "36"
        )  # wc -l: the unterminated last line is not counted
        self.assertEqual(
            lines[1:4],
            [
                "2:Error: the runner lost its connection",
                "5:***Failed    1.20 sec",
                "6:" + "x" * 258,
            ],
        )  # each cut to 260 characters
        self.assertEqual(
            lines[4:], [f"{i + 7}:test_{i} FAILED" for i in range(22)]
        )  # at most 25
        self.assertEqual(len(lines), 26)

    def test_a_log_kept_removes_the_ones_kept_more_than_a_day_ago(self):
        # This user's only, as the temporary folder may be shared; nothing else there.
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"TMPDIR": d}
        ):
            tmp = pathlib.Path(d)
            for name, hours in (
                ("joblog_1.txt", 25),
                ("joblog_2.txt", 23),
                ("other_1.txt", 25),
            ):
                (tmp / name).write_text("old\n", encoding="utf-8")
                t = time.time() - hours * 3600
                os.utime(tmp / name, (t, t))
            with mock.patch("os.getuid", return_value=os.getuid() + 1):
                ci_runs.prune_job_logs()  # as another user
            self.assertEqual(len(list(tmp.iterdir())), 3)
            code, _ = cli(["job-log", "3"], raw=lambda path, timeout=120: "log\n")
            self.assertEqual(code, 0)
            self.assertEqual(
                sorted(p.name for p in tmp.iterdir()),
                ["joblog_2.txt", "joblog_3.txt", "other_1.txt"],
            )

    def test_the_log_is_kept_in_tmpdir_even_while_that_folder_is_missing(self):
        """TMPDIR names the temporary folder: a log is kept there, the folder made when it is missing, and never in
        the system's temporary folder in its place."""
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as system:
            tmp = pathlib.Path(d) / "tmp"
            with mock.patch.dict(os.environ, {"TMPDIR": str(tmp)}), mock.patch.object(
                ci_runs.tempfile, "tempdir", system
            ):
                code, _ = cli(["job-log", "4"], raw=lambda path, timeout=120: "log\n")
            self.assertEqual(code, 0)
            self.assertEqual(
                (tmp / "joblog_4.txt").read_text(encoding="utf-8"), "log\n"
            )
            self.assertEqual(list(pathlib.Path(system).iterdir()), [])

    def test_a_log_it_cannot_keep_in_tmpdir_is_said(self):
        """TMPDIR names no folder it can write in (here a file): rk job-log says the log is not kept, and keeps it
        nowhere else."""
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as system:
            tmp = pathlib.Path(d) / "tmp"
            tmp.write_text("a file, not a folder", encoding="utf-8")
            with mock.patch.dict(os.environ, {"TMPDIR": str(tmp)}), mock.patch.object(
                ci_runs.tempfile, "tempdir", system
            ):
                code, lines = cli(
                    ["job-log", "4"], raw=lambda path, timeout=120: "log\n"
                )
            self.assertEqual(code, 1)
            self.assertEqual(len(lines), 1, lines)
            self.assertTrue(lines[0].startswith("(log not kept: "), lines)
            self.assertEqual(list(pathlib.Path(system).iterdir()), [])

    def test_an_unread_log_is_said(self):
        def raw(path, timeout=120):
            raise gh.GhError(
                "gh api repos/ROCm/r/actions/jobs/5/logs: gh: Not Found (HTTP 404)"
            )

        self.assertEqual(
            cli(["job-log", "5"], raw=raw),
            (
                1,
                [
                    "(log not read: gh api repos/ROCm/r/actions/jobs/5/logs: gh: Not Found (HTTP 404))"
                ],
            ),
        )


class TestHeads(unittest.TestCase):
    def test_one_line_per_pr_as_jq_printed_it(self):
        prs = {
            "7": {
                "number": 7,
                "sha": "a" * 40,
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
                "updated_at": "2026-11-05T10:00:00Z",
            },
            "8": {
                "number": 8,
                "sha": "b" * 40,
                "state": "closed",
                "merged": True,
                "mergeable_state": None,
                "updated_at": "2026-11-05T11:00:00Z",
            },
        }

        def rest(path, jq=None, paginate=False, timeout=120):
            n = path.rsplit("/", 1)[1]
            if n not in prs:
                raise gh.GhError(f"gh api {path}: gh: Not Found (HTTP 404)")
            return prs[n]

        self.assertEqual(
            cli(["heads", "7", "8"], rest=rest),
            (
                0,
                [
                    "7 aaaaaaaaaa open merged=false mergeable=blocked updated=2026-11-05T10:00:00Z",
                    "8 bbbbbbbbbb closed merged=true mergeable=null updated=2026-11-05T11:00:00Z",
                ],
            ),
        )
        code, lines = cli(["heads", "9", "7"], rest=rest)
        self.assertEqual(
            (code, lines[0]),
            (1, "9 (not read: gh api repos/ROCm/r/pulls/9: gh: Not Found (HTTP 404))"),
        )
        self.assertTrue(lines[1].startswith("7 aaaaaaaaaa open"))


if __name__ == "__main__":
    unittest.main()
