# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""worker/mkwt.py on a real repository: the PR's merge worktree merges develop, or the PR's base branch when
RKMON_BASE names one (a PR on a feature branch), and its log says which; a conflict is said and its files listed; a
second run leaves nothing of the first in the trees; a PR the clone cannot fetch ends it with git's exit code; no PR
number is its usage (exit 2). Each worktree goes into a folder the PR runners' group reads (2750), on the RAM disk too,
and one that cannot be made so ends it."""

import grp
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MKWT = ROOT / "worker" / "mkwt.py"
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ENV.update(
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.com",
)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


class TestMkwt(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        base = pathlib.Path(self.d.name)
        self.up = up = base / "upstream"
        up.mkdir()
        git("init", "-q", "-b", "develop", cwd=up)
        (up / "p").mkdir()
        (up / "p" / "a.txt").write_text("a\n")
        git("add", "-A", cwd=up)
        git("commit", "-qm", "base", cwd=up)
        git("checkout", "-qb", "users/x/feature", cwd=up)
        (up / "p" / "f.txt").write_text("feature\n")
        git("add", "-A", cwd=up)
        git("commit", "-qm", "the feature branch", cwd=up)
        git("checkout", "-qb", "pr7", cwd=up)
        (up / "p" / "pr.txt").write_text("the PR\n")
        git("add", "-A", cwd=up)
        git("commit", "-qm", "the PR, on the feature branch", cwd=up)
        git("update-ref", "refs/pull/7/head", "pr7", cwd=up)
        git("checkout", "-q", "develop", cwd=up)
        (up / "p" / "d.txt").write_text("develop moved on\n")
        git("add", "-A", cwd=up)
        git("commit", "-qm", "develop", cwd=up)
        self.work = base / "work"
        self.work.mkdir()
        self.repo = self.work / "repo"
        git("clone", "-q", "--no-checkout", str(up), str(self.repo), cwd=base)
        self.env = dict(
            ENV,
            RKMON_WORKING_DIR=str(self.work),
            RKMON_REPO_DIR=str(self.repo),
            RKMON_REPO="o/r",
            RKMON_ROCKE_PATH="p",
            RKMON_SPARSE="p",
            RKMON_CLONE_SINCE="",
        )

    def run_mkwt(self, *args, **env):
        return subprocess.run(
            [sys.executable, "-B", str(MKWT), *args],
            capture_output=True,
            text=True,
            timeout=120,
            env={k: v for k, v in dict(self.env, **env).items() if v is not None},
        )

    def mkwt(self, **env):
        p = self.run_mkwt("7", **env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def test_develop_by_default(self):
        out = self.mkwt()
        self.assertNotIn(" base=", out.splitlines()[0])
        self.assertTrue(
            (self.work / "wt7m" / "p" / "d.txt").exists()
        )  # develop's change is merged in
        self.assertEqual(
            out.splitlines()[1:],
            ["behind/ahead: 1\t2", "conflicting files vs develop:"],
        )

    def test_the_base_branch_when_named(self):
        out = self.mkwt(RKMON_BASE="users/x/feature")
        self.assertTrue(out.splitlines()[0].endswith(" base=users/x/feature"), out)
        self.assertIn("behind/ahead: 0\t1", out)
        merged = self.work / "wt7m" / "p"
        self.assertTrue((merged / "pr.txt").exists())
        self.assertTrue((merged / "f.txt").exists())
        self.assertFalse(
            (merged / "d.txt").exists()
        )  # the base branch's tree, not develop's

    def test_a_conflict_is_said_and_its_files_listed(self):
        git("checkout", "-q", "develop", cwd=self.up)
        (self.up / "p" / "pr.txt").write_text("develop's own pr.txt\n")
        git("add", "-A", cwd=self.up)
        git("commit", "-qm", "develop adds pr.txt too", cwd=self.up)
        out = self.mkwt()
        self.assertIn("!! merge with develop has conflicts in wt7m", out)
        tail = out.split("conflicting files vs develop:\n", 1)[1]
        self.assertEqual(tail.splitlines()[0], "p/pr.txt")

    def test_a_second_run_leaves_nothing_of_the_first(self):
        self.mkwt()
        (self.work / "wt7" / "p" / "stray.txt").write_text("x\n")
        (self.work / "wt7m" / "p" / "a.txt").write_text("edited\n")
        self.mkwt()
        self.assertFalse((self.work / "wt7" / "p" / "stray.txt").exists())
        self.assertEqual((self.work / "wt7m" / "p" / "a.txt").read_text(), "a\n")

    def test_usage_and_a_failing_fetch(self):
        p = self.run_mkwt()
        self.assertEqual((p.returncode, p.stdout), (2, "usage: mkwt.py <PR number>\n"))
        p = self.run_mkwt(
            "9"
        )  # no refs/pull/9/head: git's fetch fails, and so does the script, with its code
        self.assertEqual(p.returncode, 128, p.stdout + p.stderr)
        self.assertEqual(p.stdout, "")

    def test_a_missing_setting_names_the_fix(self):
        p = self.run_mkwt("7", RKMON_REPO=None)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(
            p.stderr,
            "rkenv: RKMON_REPO is not set (check config.json with `rk doctor`)\n",
        )

    def run_as_container(self, group: str, **env):
        """mkwt.py with `group` standing in for the PR runners' group (pr_user.group)."""
        code = (
            "import runpy, sys; sys.path.insert(0, sys.argv[1]); import pr_user; "
            f"pr_user.group = lambda *a: {group!r}; sys.argv = sys.argv[2:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        return subprocess.run(
            [sys.executable, "-B", "-c", code, str(MKWT.parent), str(MKWT), "7"],
            capture_output=True,
            text=True,
            timeout=120,
            env={k: v for k, v in dict(self.env, **env).items() if v is not None},
        )

    def tree_mode(self, path: pathlib.Path) -> tuple[int, int]:
        s = path.stat()
        return stat.S_IMODE(s.st_mode), s.st_gid

    def test_each_worktree_goes_into_a_folder_the_runners_group_reads(self):
        """In the container the PR runners' group reads each worktree and none of them writes it;
        here the test user's own group stands in for theirs."""
        p = self.run_as_container(grp.getgrgid(os.getgid()).gr_name)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        for name in ("wt7", "wt7m", "wtdev"):
            self.assertEqual(
                self.tree_mode(self.work / name), (0o2750, os.getgid()), name
            )

    def test_with_the_ramdisk_its_trees_are_the_folders_made(self):
        ram = self.work / "ram"
        p = self.run_as_container(
            grp.getgrgid(os.getgid()).gr_name, RKMON_RAMDISK=str(ram)
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        for name in ("wt7", "wt7m"):
            self.assertTrue((self.work / name).is_symlink(), name)
            self.assertEqual(self.tree_mode(ram / name), (0o2750, os.getgid()), name)
        self.assertEqual(self.tree_mode(self.work / "wtdev"), (0o2750, os.getgid()))

    def test_a_folder_that_cannot_be_made_a_tree_ends_it(self):
        mine = set(os.getgroups()) | {os.getgid()}
        other = next((g for g in grp.getgrall() if g.gr_gid not in mine), None)
        if other is None or os.geteuid() == 0:
            self.skipTest("no group this user cannot give a folder")
        p = self.run_as_container(other.gr_name)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn(f"!! {self.work}/wt7: not made a tree PR code reads", p.stdout)
        # no worktree added
        self.assertFalse((self.work / "wt7" / ".git").exists())


if __name__ == "__main__":
    unittest.main()
