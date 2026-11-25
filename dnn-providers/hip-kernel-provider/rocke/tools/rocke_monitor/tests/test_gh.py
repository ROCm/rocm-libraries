# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gh wrapper retries every GitHub failure except a JSON 4xx (lessons L-505); a token is logged in to a folder on
gh's standard input alone."""

import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import gh  # noqa: E402

# As gh 2.46 reports them: a JSON 4xx prints "gh: <message> (HTTP 4xx)" and its JSON body on stdout; a
# network failure prints Go's error; a DNS failure "error connecting to <host>".
DNS = "error connecting to api.github.com\ncheck your internet connection or https://githubstatus.com\n"
REFUSED = 'Get "https://api.github.com/x": proxyconnect tcp: dial tcp 127.0.0.1:9: connect: connection refused\n'
NOT_FOUND = (
    '{"message":"Not Found","documentation_url":"https://docs.github.com/rest","status":"404"}',
    "gh: Not Found (HTTP 404)\n",
)


class TestTransient(unittest.TestCase):
    def test_transient_errors_are_retried(self):
        for err in (
            "gh: Server Error (HTTP 502)",
            "HTTP 500: Internal Server Error",
            "HTTP 504 Gateway Timeout",
            # mid-pagination: a retry, never a failed tick
            "gh api --paginate: stream error: stream ID 3; CANCEL; received from peer",
            "read: connection reset by peer",
            "timed out after 120s",
            # a non-JSON error page from the API
            "invalid character 'u' looking for beginning of value",
            # a truncated GraphQL response, and a DNS failure (L-505)
            "gh api graphql: unexpected end of JSON input",
            DNS,
            REFUSED,
            'Post "https://api.github.com/graphql": net/http: TLS handshake timeout',
            "gh: Too Many Requests (HTTP 429)",
            "gh: HTTP 404",  # 429 passes in time; a 4xx without JSON
            # A GraphQL error has no HTTP status: retried, at worst 20 s for a permanent one
            "gh: Something went wrong while executing your query. This may be the result of a timeout",
            "gh: Could not resolve to a PullRequest with the number of 999999999.",
            "exit 1 with no message",
        ):
            self.assertTrue(gh.transient(err), err)

    def test_permanent_errors_are_not(self):
        for err in (
            "gh: Not Found (HTTP 404)",
            "gh: Bad credentials (HTTP 401)",
            "gh: No commit found for SHA: 0000000000000000000000000000000000000001 (HTTP 422)",
            "gh api graphql: gh: Resource not accessible by integration (HTTP 403)",
            "HTTP 401: Bad credentials",
            "HTTP 422: Validation Failed",
        ):
            self.assertFalse(gh.transient(err), err)


class TestRun(unittest.TestCase):
    def run_gh(
        self,
        results: list[tuple[int, str, str]],
        call=lambda: gh._run(["api", "x"], 60),
    ):
        """`call` against canned gh answers, (returncode, stdout, stderr) each: (its result or GhError, the gh calls,
        the backoff sleeps)."""
        calls, sleeps = [], []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            rc, out, err = results[len(calls) - 1]
            return subprocess.CompletedProcess(cmd, rc, out, err)

        with mock.patch.object(gh.subprocess, "run", fake_run), mock.patch.object(
            gh.time, "sleep", sleeps.append
        ):
            try:
                return call(), calls, sleeps
            except gh.GhError as exc:
                return exc, calls, sleeps

    def test_error_connecting_is_retried(self):
        """An "error connecting to api.github.com" is asked again, never a failed tick (lessons L-505)."""
        out, calls, sleeps = self.run_gh(
            [(1, "", DNS), (1, "", REFUSED), (0, '{"ok": 1}', "")]
        )
        self.assertEqual((out, len(calls), sleeps), ('{"ok": 1}', 3, [5, 15]))
        out, calls, _ = self.run_gh([(1, "", REFUSED)] * 3)
        self.assertEqual((type(out), len(calls)), (gh.GhError, 3))
        self.assertIn("connection refused", str(out))
        out, calls, _ = self.run_gh(
            [(1, "", DNS), (0, '{"sha": "abc"}', "")],
            call=lambda: gh.rest("repos/ROCm/r/pulls/7"),
        )
        self.assertEqual((out, len(calls)), ({"sha": "abc"}, 2))

    def test_json_4xx_fails_fast(self):
        out, calls, sleeps = self.run_gh([(1, *NOT_FOUND)])
        self.assertEqual(
            (str(out), len(calls), sleeps),
            ("gh api x: gh: Not Found (HTTP 404)", 1, []),
        )
        out, calls, _ = self.run_gh(
            [
                (
                    1,
                    '{"message":"Bad credentials","status":"401"}',
                    "gh: Bad credentials (HTTP 401)\n",
                )
            ]
        )
        self.assertEqual((type(out), len(calls)), (gh.GhError, 1))
        out, calls, sleeps = self.run_gh(
            [
                (
                    1,
                    '{"message":"Too Many Requests"}',
                    "gh: Too Many Requests (HTTP 429)\n",
                ),
                (0, "[]", ""),
            ]
        )
        self.assertEqual((out, len(calls), sleeps), ("[]", 2, [5]))

    def test_raw_returns_the_body_as_it_comes(self):
        """`rk job-log` counts a log's lines and greps them: nothing stripped or parsed, and a byte that is not UTF-8
        cannot fail the read."""
        log = "[\n2026-11-05T10:00:00Z Error: x\n\n"  # it even starts like JSON
        out, calls, _ = self.run_gh(
            [(0, log, "")], call=lambda: gh.raw("repos/o/r/actions/jobs/5/logs")
        )
        self.assertEqual(
            (out, calls), (log, [["gh", "api", "repos/o/r/actions/jobs/5/logs"]])
        )
        seen = []
        with mock.patch.object(
            gh.subprocess,
            "run",
            lambda cmd, **kw: seen.append(kw)
            or subprocess.CompletedProcess(cmd, 0, "", ""),
        ):
            gh.raw("repos/o/r/actions/jobs/5/logs")
        self.assertEqual(seen[0]["errors"], "replace")
        with self.assertRaises(gh.GhError):
            gh.raw("--method=POST")


class TestLoginWithToken(unittest.TestCase):
    def test_the_token_goes_on_stdin_into_the_folder_s_own_file(self):
        """gh.login_with_token: `gh auth login --with-token --insecure-storage` under the folder's login, the token on
        its standard input, in no argument, and no variable gh would read a token from passed on; the folder 700 and
        its files 600; "" once gh took it, else gh's own words."""
        token = "invented-read-" + "token-Q7"
        ran = []

        def took(cmd, **kw):
            ran.append((cmd, kw))
            login = pathlib.Path(kw["env"]["GH_CONFIG_DIR"], gh.HOSTS_FILE)
            login.write_text("github.com: {}\n")
            login.chmod(0o644)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "gh-read")
            with mock.patch.object(gh.subprocess, "run", took), mock.patch.dict(
                os.environ, {"GH_TOKEN": "another"}
            ):
                self.assertEqual(gh.login_with_token(folder, token + "\n"), "")
            ((cmd, kw),) = ran
            self.assertEqual(
                cmd[1:],
                [
                    "auth",
                    "login",
                    "--hostname",
                    "github.com",
                    "--with-token",
                    "--insecure-storage",
                ],
            )
            self.assertEqual(kw["input"], token + "\n")
            self.assertEqual(kw["env"]["GH_CONFIG_DIR"], folder)
            self.assertNotIn("GH_TOKEN", kw["env"])
            self.assertNotIn(token, " ".join(cmd))
            self.assertEqual(stat.S_IMODE(os.stat(folder).st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE(os.stat(os.path.join(folder, gh.HOSTS_FILE)).st_mode),
                0o600,
            )
            refused = subprocess.CompletedProcess(
                [], 1, "", "error validating token: HTTP 401: Bad credentials\n"
            )
            with mock.patch.object(gh.subprocess, "run", lambda cmd, **kw: refused):
                self.assertEqual(
                    gh.login_with_token(folder, token),
                    "error validating token: HTTP 401: Bad credentials",
                )


if __name__ == "__main__":
    unittest.main()
