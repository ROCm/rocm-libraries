# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The environment PR code runs with (worker/pr_env.py): an allow-list.

- A variable reaches PR code only when the list names it, or it starts with a listed prefix and is not shaped like a
  secret's: a planted X_PASSWORD and X_KEY do not, nor ROCKE_API_KEY or LC_TOKEN, nor XDG's folders, gh's or git's
  settings or ssh's agent, nor any login folder, token, key, model setting or variable of the monitor's own. A secret's
  word (KEY, PAT, ...) counts as a word of the name only: ROCKE_KEYBOARD and ROCKE_PATH pass.
- The locale's, the search paths, rocKE's switches and RKMON_JOB do.
- PYTHONDONTWRITEBYTECODE is 1 whatever the caller's says: a Python run through `pr_env.py CMD` writes no bytecode.
- `pr_env.py CMD` runs CMD with the list's variables and its caller's HOME, TMPDIR and RKMON_RAMDISK, nothing else.
- `data(env)` and `pr_env.py --print0` give the launcher the list's variables only, NUL-separated; `parse` reads such
  data back as data.
- `host(env)` is the list's variables and the caller's HOME, TMPDIR and RKMON_RAMDISK: PR code's environment on a
  host; the launcher's data never holds RKMON_RAMDISK.
"""

import io
import os
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stderr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

import pr_env  # noqa: E402

# What a job's environment holds that PR code must never get: invented values.
SECRET_LIKE = {
    "X_PASSWORD": "fake-password",
    "X_KEY": "fake-key",
    "MY_TOKEN": "fake-token",
    "GH_TOKEN": "fake-token",
    "GITHUB_TOKEN": "fake-token",
    "GH_CONFIG_DIR": "/secrets/gh-read",
    "ANTHROPIC_API_KEY": "fake-key",
    "ANTHROPIC_BASE_URL": "https://example.com",
    "CLAUDE_CONFIG_DIR": "/work/claude/config",
    "ATLASSIAN_API_TOKEN": "fake-token",
    "SSH_AUTH_SOCK": "/tmp/agent.sock",
    "SUDO_USER": "rkmon",
    "RKMON_HOME": "/work/monitor",
    "RKMON_SECRETS_DIR": "/secrets",
    "RKMON_GH_WRITE_DIR": "/secrets-write/gh",
    "RKMON_WRITE_BROKER": "/opt/x/docker/rk_write.py",
    "RKMON_PR_RUNNER": "/opt/x/docker/rk_prcode.py",
    "RKMON_OPERATOR_LOGIN": "<user_login>",
    "RKMON_TRACKER_KEYS": "ABC",
    "RKMON_WORKER_URL": "https://example.com/v1",
    "RKMON_WORKER_JSON": '{"key": "fake-key"}',
    "RKMON_WORKER_MODEL": "m",
    "RKMON_GH": "/usr/bin/gh",
    "RKMON_TMP": "/work/tmp",
    "RKMON_PROBES": "/work/probes",
    "HOME": "/home/rkmon",
    "TMPDIR": "/work/tmp",
    "USER": "rkmon",
    "MAIL": "/var/mail/rkmon",
}
NEEDED = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "PATH": "/usr/bin:/bin",
    "PYTHONPATH": "/x",
    "ROCM_PATH": "/opt/rocm",
    "HIP_VISIBLE_DEVICES": "",
    "ROCKE_LLVM_FLAVOR": "llvm22",
    "RKMON_JOB": "/work/monitor/state/jobs/checks-pr7-1",
    "RKMON_BUILD": "/ramdisk/slots/rkpr1/build",
    "RKMON_OUT": "/ramdisk/slots/rkpr1/out",
    "RKMON_WORKER_REMOTE": "1",
    "RKMON_RERUN_VIA": "rk pr-run",
    "RKMON_RESULTS": "/work/monitor/state/jobs/checks-pr7-1/out/tests",
    "RKMON_TREES": "/work/wt7m:/work/repo",
}
# What PR code gets on a host only (pr_env.host): the RAM disk's folder, where worker/build_native.py builds a PR's
# tree.
HOST_ONLY = {"RKMON_RAMDISK": "/ramdisk"}
# What never reaches PR code, a listed prefix's name too: XDG's folders, gh's and git's settings, ssh's agent, and every
# name shaped like a secret's, in any case.
NEVER = (
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    "GH_TOKEN",
    "GH_HOST",
    "GH_CONFIG_DIR",
    "GH_ENTERPRISE_TOKEN",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0",
    "GIT_CONFIG_PARAMETERS",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "SSH_ASKPASS",
    "ROCKE_TOKEN",
    "ROCKE_GH_TOKEN",
    "ROCKE_KEY",
    "ROCKE_API_KEY",
    "ROCKE_APIKEY",
    "ROCKE_api_key",
    "ROCKE_SECRET",
    "ROCKE_CLIENT_SECRET",
    "ROCKE_PASSWORD",
    "ROCKE_DB_PASSWD",
    "ROCKE_PAT",
    "ROCKE_CREDENTIALS",
    "ROCKE_CREDS",
    "ROCKE_SESSION_COOKIE",
    "LC_TOKEN",
    "LC_SECRET_KEY",
)


class TestAllowed(unittest.TestCase):
    def test_only_listed_names_and_prefixes_pass(self):
        got = pr_env.allowed({**SECRET_LIKE, **NEEDED})
        self.assertEqual(got, {**NEEDED, **pr_env.FIXED})

    def test_a_planted_password_and_key_do_not_reach_pr_code(self):
        got = pr_env.allowed({"X_PASSWORD": "p", "X_KEY": "k", "LANG": "C"})
        self.assertEqual(got, {"LANG": "C", **pr_env.FIXED})

    def test_never_xdg_gh_git_config_ssh_or_a_secret_shaped_name(self):
        for name in NEVER:
            with self.subTest(name=name):
                self.assertFalse(pr_env.listed(name))
                self.assertEqual(
                    pr_env.allowed({name: "x", "LANG": "C"}),
                    {"LANG": "C", **pr_env.FIXED},
                )
                self.assertNotIn(name.encode(), pr_env.data({name: "x"}))
                self.assertNotIn(name, pr_env.host({name: "x"}))

    def test_a_secrets_word_counts_as_a_word_of_the_name_only(self):
        for name in (
            "ROCKE_KEYBOARD",
            "ROCKE_MONKEY_TEST",
            "ROCKE_PATH",
            "ROCKE_PATCH_DIR",
            "LC_PAPER",
        ):
            with self.subTest(name=name):
                self.assertTrue(pr_env.listed(name))

    def test_no_python_writes_bytecode_whatever_the_caller_says(self):
        for given in (
            {},
            {"PYTHONDONTWRITEBYTECODE": ""},
            {"PYTHONDONTWRITEBYTECODE": "0"},
        ):
            with self.subTest(given=given):
                self.assertEqual(pr_env.allowed(given)["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(pr_env.host(given)["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(
                    pr_env.parse(pr_env.data(given))["PYTHONDONTWRITEBYTECODE"], "1"
                )

    def test_a_python_run_through_it_writes_no_bytecode(self):
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "worker" / "pr_env.py"),
                sys.executable,
                "-c",
                "import sys; print(sys.flags.dont_write_bytecode)",
            ],
            env={"PATH": os.environ.get("PATH") or "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "1"), p.stderr)

    def test_no_listed_name_looks_like_a_secret(self):
        for name in pr_env.NAMES:
            for word in (
                "TOKEN",
                "KEY",
                "PASSWORD",
                "SECRET",
                "LOGIN",
                "URL",
                "CONFIG_DIR",
            ):
                self.assertNotIn(word, name, name)
            self.assertFalse(pr_env.secret_shaped(name), name)

    def test_the_prefixes_are_the_locale_and_rockes(self):
        self.assertEqual(set(pr_env.PREFIXES), {"LC_", "ROCKE_"})
        self.assertTrue(pr_env.listed("LC_CTYPE"))
        self.assertTrue(pr_env.listed("ROCKE_X"))
        self.assertFalse(pr_env.listed("ROCKEX"))
        self.assertFalse(pr_env.listed("XLC_ALL"))


class TestMain(unittest.TestCase):
    def test_runs_the_command_with_the_list_and_the_callers_home_and_tmpdir(self):
        seen = []

        def fake_exec(path, argv, env):
            seen.append((path, argv, env))

        rc = pr_env.main(
            ["pytest", "-q"],
            execvpe=fake_exec,
            environ={**SECRET_LIKE, **NEEDED, **HOST_ONLY},
        )
        self.assertEqual(rc, 0)
        path, argv, env = seen[0]
        self.assertEqual((path, argv), ("pytest", ["pytest", "-q"]))
        self.assertEqual(
            env,
            {
                **NEEDED,
                **HOST_ONLY,
                **pr_env.FIXED,
                "HOME": "/home/rkmon",
                "TMPDIR": "/work/tmp",
            },
        )

    def test_print0_writes_the_listed_variables_only(self):
        out = io.BytesIO()
        rc = pr_env.main(
            ["--print0"], environ={**SECRET_LIKE, **NEEDED, **HOST_ONLY}, out=out
        )
        self.assertEqual(rc, 0)
        got = dict(item.split(b"=", 1) for item in out.getvalue().split(b"\0") if item)
        self.assertEqual(
            got,
            {k.encode(): v.encode() for k, v in {**NEEDED, **pr_env.FIXED}.items()},
        )
        self.assertTrue(out.getvalue().endswith(b"\0"))

    def test_data_is_nul_separated(self):
        self.assertEqual(
            pr_env.data({"LANG": "C", "X_KEY": "k", "PATH": "/a b"}),
            b"LANG=C\0PATH=/a b\0PYTHONDONTWRITEBYTECODE=1\0",
        )

    def test_parse_reads_the_data_back_as_data(self):
        self.assertEqual(
            pr_env.parse(b"A=1\0B=x=y\0junk\0C=\xff\0D=\0"),
            {"A": "1", "B": "x=y", "D": ""},
        )
        env = {**NEEDED, "PATH": "/a b:/c"}
        self.assertEqual(pr_env.parse(pr_env.data(env)), {**env, **pr_env.FIXED})

    def test_host_is_the_list_and_the_callers_home_tmpdir_and_ramdisk(self):
        self.assertEqual(
            pr_env.host({**SECRET_LIKE, **NEEDED, **HOST_ONLY}),
            {
                **NEEDED,
                **HOST_ONLY,
                **pr_env.FIXED,
                "HOME": "/home/rkmon",
                "TMPDIR": "/work/tmp",
            },
        )
        self.assertEqual(
            pr_env.host({"HOME": "", "LANG": "C"}), {"LANG": "C", **pr_env.FIXED}
        )
        self.assertEqual(
            pr_env.allowed({**NEEDED, **HOST_ONLY}), {**NEEDED, **pr_env.FIXED}
        )

    def test_no_command_is_a_usage_error(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(pr_env.main([], execvpe=None, environ={}), 2)
        self.assertIn("usage", err.getvalue())

    def test_a_command_that_cannot_start_exits_127(self):
        def fail(path, argv, env):
            raise FileNotFoundError(2, "No such file or directory")

        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(pr_env.main(["nope"], execvpe=fail, environ={}), 127)
        self.assertIn("nope: No such file or directory", err.getvalue())


if __name__ == "__main__":
    unittest.main()
