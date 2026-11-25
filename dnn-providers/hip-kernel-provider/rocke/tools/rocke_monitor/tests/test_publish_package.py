# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/publish_package.py in scratch repositories: it publishes the committed HEAD of the package checkout it lives
in (one commit, as rkmon), refuses a package work tree with changes, and refuses a tree whose config.json is a
machine's (a live toolkit's), so it never publishes from the live toolkit."""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_sync_from_live import DENY, GIT_ENV, TREE  # noqa: E402

BLANK = '{"blank": true}\n'


class TestPublishPackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        self.pkg, self.tgt = d / "pkg", d / "clean"
        for repo in (self.pkg, self.tgt):
            repo.mkdir()
            self.git(repo, "init", "-q", "-b", "main")
        for rel, body in dict(TREE, **{"config.json": BLANK}).items():
            (self.pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.pkg / rel).write_text(body, encoding="utf-8")
        for name in ("publish_package.py", "sync_from_live.py"):
            shutil.copy(ROOT / "tools" / name, self.pkg / "tools" / name)
        self.git(self.pkg, "add", "-A")
        self.git(self.pkg, "commit", "-q", "-m", "package")
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

    def publish(self, *args):
        e = dict(os.environ, **GIT_ENV, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.pkg / "tools" / "publish_package.py"),
                *args,
            ],
            cwd=self.tmp.name,
            env=e,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_it_publishes_the_package_head_once(self):
        head = self.git(self.pkg, "rev-parse", "HEAD").strip()
        p = self.publish(str(self.tgt), *DENY)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(
            p.stdout.splitlines()[-1],
            rf"^committed [0-9a-f]+ rocKE monitor package: publish {head[:10]} "
            r"\(rkmon <rkmon@localhost>\)$",
        )
        self.assertEqual((self.tgt / "config.json").read_text(encoding="utf-8"), BLANK)
        self.assertTrue((self.tgt / "tools" / "publish_package.py").is_file())
        self.assertFalse((self.tgt / "old.txt").exists())
        p = self.publish(str(self.tgt), *DENY)
        self.assertEqual(p.returncode, 0)
        self.assertIn("nothing to sync", p.stdout)

    def test_it_requires_the_deny_list(self):
        """The privacy check requires the deny list: a publish that names none is a usage error, and the target
        repository is left as it was."""
        p = self.publish(str(self.tgt))
        self.assertEqual(p.returncode, 2)
        self.assertIn("requires the deny list", p.stdout)
        self.assertEqual(self.git(self.tgt, "status", "--porcelain"), "")
        self.assertEqual(self.git(self.tgt, "log", "--format=%s").split(), ["first"])

    def test_it_never_publishes_a_live_tree_or_uncommitted_work(self):
        (self.pkg / "stray.txt").write_text("x\n", encoding="utf-8")
        p = self.publish(str(self.tgt))
        self.assertEqual(p.returncode, 1)
        self.assertIn("has changes: commit them", p.stdout)
        (self.pkg / "stray.txt").unlink()
        (self.pkg / "config.json").write_text(
            '{"operator": {"github_login": "someone"}}\n', encoding="utf-8"
        )
        self.git(self.pkg, "commit", "-q", "-am", "a machine's settings")
        p = self.publish(str(self.tgt))
        self.assertEqual(p.returncode, 1)
        self.assertIn("is not the blank configuration", p.stdout)
        self.assertEqual(self.git(self.tgt, "log", "--format=%s").split(), ["first"])

    def test_usage(self):
        self.assertEqual(self.publish().returncode, 2)


if __name__ == "__main__":
    unittest.main()
