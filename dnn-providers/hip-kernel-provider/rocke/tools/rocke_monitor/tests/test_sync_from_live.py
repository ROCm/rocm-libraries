# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/sync_from_live.py in scratch repositories: its usage, its refusals (a deny list not named among them), one sync
committed as rkmon with the blank config.json, nothing to sync the second time, and a failing privacy check or test that
commits nothing. The source's tree carries its own stand-in privacy check, `rkmon.py selftest` and test, as the sync
runs the synced tree's own; the deny list the calls name is a file name only (the stand-in check never reads it). The
stand-in has the check's command line and parser too, as the real module has: a sync takes them from the package its
script is in, which is the published tree itself when tools/publish_package.py runs the sync
(tests/test_publish_package.py). Here that package is this one, so the sync reads the deny list's options with the real
check's parser.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYNC = ROOT / "tools" / "sync_from_live.py"
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}
TREE = {
    "rkmon_lib/__init__.py": "",
    "rkmon_lib/privacy.py": "import argparse, os, sys\n\n\n"
    "def command(*args):\n"
    "    return [sys.executable, '-B', '-m', 'rkmon_lib.privacy', *map(str, args)]\n\n\n"
    "def parser():\n"
    "    ap = argparse.ArgumentParser(allow_abbrev=False)\n"
    "    ap.add_argument('paths', nargs='*')\n"
    "    ap.add_argument('--deny')\n"
    "    return ap\n\n\n"
    "if __name__ == '__main__':\n"
    "    print('check', sys.argv[-1] == '.')\n"
    "    w = os.environ.get('FAKE_PRIV_WORD')\n"
    "    if w and sys.argv[-1] != '.' and w in open(sys.argv[-1]).read():\n"
    "        sys.exit(1)\n"
    "    sys.exit(1 if os.environ.get('FAKE_PRIV') == ('files' if sys.argv[-1] == '.' else 'diff')"
    " else 0)\n",
    "tools/make_package.py": "BLANK_CONFIG = '{\"blank\": true}\\n'\n",
    # The stand-in's `rk selftest` runs its tests, and tells them so.
    "rkmon.py": "import os, subprocess, sys\n"
    "argv = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests']\n"
    "env = dict(os.environ, RAN_BY='rkmon.py selftest')\n"
    "sys.exit(subprocess.call(argv, env=env) if sys.argv[1:] == ['selftest'] else 2)\n",
    "tests/test_x.py": "import os, unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n"
    "        self.assertEqual(os.environ.get('RAN_BY'), 'rkmon.py selftest')\n"
    "        self.assertNotEqual(os.environ.get('FAKE_TESTS'), 'fail')\n",
    "config.json": '{"live": "values"}\n',
}
DENY = ("--", "--deny", "deny.txt")


class TestSyncFromLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        self.src, self.tgt = d / "live", d / "pkg"
        for repo in (self.src, self.tgt):
            repo.mkdir()
            self.git(repo, "init", "-q", "-b", "main")
        for rel, text in TREE.items():
            (self.src / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.src / rel).write_text(text, encoding="utf-8")
        self.git(self.src, "add", "-A")
        self.git(self.src, "commit", "-q", "-m", "live")
        self.git(self.src, "branch", "sync")
        (self.tgt / "old.txt").write_text("old\n", encoding="utf-8")
        self.git(self.tgt, "add", "-A")
        self.git(self.tgt, "commit", "-q", "-m", "first")

    def git(self, repo, *args) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, **GIT_ENV),
        ).stdout

    def sync(self, *args, **env):
        e = dict(os.environ, **GIT_ENV, PYTHONDONTWRITEBYTECODE="1", **env)
        e.pop("GIT_AUTHOR_NAME")
        return subprocess.run(
            [sys.executable, "-B", str(SYNC), *args],
            cwd=self.tgt,
            env=e,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_usage(self):
        p = self.sync(str(self.src))
        self.assertEqual(p.returncode, 2)
        self.assertIn(
            "#   tools/sync_from_live.py SOURCE_CHECKOUT REF -- --deny FILE [PRIVACY_ARGS...]",
            p.stdout,
        )

    def test_refusals(self):
        self.assertEqual(
            self.sync(str(self.src), "nosuch", *DENY).stdout,
            f"no commit nosuch in {self.src}\n",
        )
        (self.tgt / "stray.txt").write_text("x\n", encoding="utf-8")
        p = self.sync(str(self.src), "sync", *DENY)
        self.assertEqual(
            (p.returncode, p.stdout),
            (1, "the work tree has changes: commit or drop them\n"),
        )

    def test_the_deny_list_is_required(self):
        """The privacy check requires the deny list: arguments that name none, as the check reads them, are a usage
        error, before anything changes: an abbreviated option, which the check refuses, and a later blank --deny=,
        which it reads last, name none. `--deny=FILE` names it too."""
        for args in (
            (),
            ("--",),
            ("--", "--deny"),
            ("--", "--deny", "--config", "c"),
            ("--", "--den", "deny.txt"),
            ("--", "--deny", "deny.txt", "--deny="),
        ):
            with self.subTest(args=args):
                p = self.sync(str(self.src), "sync", *args)
                self.assertEqual(
                    (p.returncode, p.stdout),
                    (
                        2,
                        "the privacy check requires the deny list: name it with --deny FILE after --; "
                        "nothing changed\n",
                    ),
                )
                self.assertEqual(self.git(self.tgt, "status", "--porcelain"), "")
                self.assertEqual(
                    self.git(self.tgt, "log", "--format=%s").split(), ["first"]
                )
        p = self.sync(str(self.src), "sync", "--", "--deny=deny.txt")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_one_sync_then_nothing_to_sync(self):
        p = self.sync(str(self.src), "sync", *DENY)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        out = p.stdout.splitlines()
        self.assertEqual(
            out[1:5],
            [
                "privacy check, the files:",
                "check True",
                "privacy check, the staged change (the commit's history):",
                "check False",
            ],
        )
        self.assertEqual(out[5:6] + out[-2:-1], ["tests:", "OK"])
        self.assertRegex(
            out[-1],
            r"^committed [0-9a-f]+ rocKE monitor package: sync \(rkmon <rkmon@localhost>\)$",
        )
        self.assertEqual(
            (self.tgt / "config.json").read_text(encoding="utf-8"), '{"blank": true}\n'
        )
        self.assertFalse((self.tgt / "old.txt").exists())
        self.assertEqual(
            self.git(self.tgt, "log", "-1", "--format=%an <%ae> %cn <%ce>").strip(),
            "rkmon <rkmon@localhost> rkmon <rkmon@localhost>",
        )
        p = self.sync(str(self.src), "sync", *DENY)
        self.assertEqual(
            (p.returncode, p.stdout),
            (0, "nothing to sync: this repository already carries sync's tree\n"),
        )

    def test_the_check_of_the_staged_change_reads_only_what_the_commit_adds(self):
        """A line the sync removes is in the history already: the check over the staged change never reads it; a line it
        adds is read."""
        (self.tgt / "old.txt").write_text("OLDWORD\n", encoding="utf-8")
        self.git(self.tgt, "commit", "-q", "-am", "old word")
        p = self.sync(str(self.src), "sync", *DENY, FAKE_PRIV_WORD="OLDWORD")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.git(self.tgt, "reset", "-q", "--hard", "HEAD~1")
        p = self.sync(
            str(self.src), "sync", *DENY, FAKE_PRIV_WORD="unittest"
        )  # in an added test file
        self.assertEqual(p.returncode, 1)
        self.assertEqual(
            p.stdout.splitlines()[-1],
            "the privacy check failed on the staged change: nothing committed",
        )

    def test_a_failed_check_or_test_commits_nothing(self):
        for env, why in (
            ({"FAKE_PRIV": "files"}, "the privacy check failed on the files"),
            ({"FAKE_PRIV": "diff"}, "the privacy check failed on the staged change"),
            ({"FAKE_TESTS": "fail"}, "the tests failed"),
        ):
            with self.subTest(why=why):
                p = self.sync(str(self.src), "sync", *DENY, **env)
                self.assertEqual(p.returncode, 1)
                self.assertEqual(p.stdout.splitlines()[-1], f"{why}: nothing committed")
                self.assertEqual(
                    self.git(self.tgt, "log", "--format=%s").split(), ["first"]
                )
                self.git(self.tgt, "reset", "-q", "--hard")


if __name__ == "__main__":
    unittest.main()
