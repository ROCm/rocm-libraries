# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The working volume's modes, the RAM disk's folder and the slots, as each start gives them (docker/rk_layout.py), on a
working folder and a RAM disk's folder in a temporary folder, with this user's ids standing in for the image's users
(the runners' group: another group of this user's when it has one).

- A volume from before these modes (umask 002) gets them: the working folder 2771 with the SHARED default ACL; the
  state, the logs and the other shared folders 2770, their files 0660, private ones private; the trees PR code reads
  2750, their files 0640, the runners' group, no ACL; the scratch folder 2771 with SCRATCH, by which the runners'
  group enters the folders in it and reads none of its files, while a job's or an agent's tree there keeps its modes;
  the cache 2771 with its develop-side results and mutation history as trees.
- The RAM disk's folder is laid out as the working folder (2771 with SHARED, its scratch folder, a PR's worktree there a
  tree), and holds the slots: root's 0755, each slot folder its runner's 2751, the records' folder 2770, the inputs
  0750 and 2750. A container without a RAM disk gets a folder of its own; one that is a link gets no slots.
- The folders the monitor's code fills are made empty when missing, as trees.
- A second start changes nothing and walks nothing.
- The system's temporary folders are root's, the monitor's group's, 1770 and no ACL: no PR runner writes or reads
  them; what is in them is left as it is.
- Nothing is done through a link, a slot's contents are never walked, an entry another user owns is left as it is,
  `--plan` changes nothing, and a file system that keeps no ACL gets the modes alone, with a line that says so.
"""

import errno
import importlib.util
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "worker"), str(ROOT), str(ROOT / "tests")]
sys.dont_write_bytecode = True

import perms  # noqa: E402
from rkmon_lib import paths  # noqa: E402  (the caches' names)
from test_perms import runners  # noqa: E402


def load():
    spec = importlib.util.spec_from_file_location(
        "rk_layout", ROOT / "docker" / "rk_layout.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rk_layout = load()


def mode(p) -> int:
    return stat.S_IMODE(os.lstat(p).st_mode)


def gid(p) -> int:
    return os.lstat(p).st_gid


def access(p):
    try:
        return perms.acl_entries(os.getxattr(p, perms.ACCESS))
    except OSError:
        return None


class Volume(unittest.TestCase):
    """A working folder as an older start left it: umask 002, folders 2775, files 0664."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = pathlib.Path(self.tmp.name) / "work"
        self.outside = pathlib.Path(self.tmp.name) / "outside"
        self.outside.mkdir()
        (self.outside / "f").write_text("not the volume's")
        os.chmod(self.outside / "f", 0o644)
        old = os.umask(0o002)
        try:
            for d in (
                "monitor/state",
                "logs",
                "claude",
                "repo/.git/objects",
                "repo/src",
                "wt7/src",
                "tmp/v7abc/wt_v1/src",
                "tmp/winsim_x",
                "cache/prcode",
            ):
                (self.work / d).mkdir(parents=True)
            for f in (
                "monitor/state/state.json",
                "logs/tick.log",
                "rocke_merge_queue.md",
                "repo/src/a.py",
                "repo/.git/config",
                "wt7/src/a.py",
                "tmp/v7abc/draft.md",
                "tmp/v7abc/wt_v1/src/a.py",
                "tmp/winsim_x/a.txt",
                "cache/prcode/golden.lines",
            ):
                (self.work / f).write_text("x")
            # The RAM disk's folder: every user's and sticky, as compose mounts a tmpfs, with a PR's worktree, a review
            # directory and a kept slot's build in it (a container's own folder keeps them across a restart).
            self.ramdisk = pathlib.Path(self.tmp.name) / "ramdisk"
            for d in ("wt7m/src", "tmp/v7fr", "slots/rkpr1/build"):
                (self.ramdisk / d).mkdir(parents=True)
            for f in ("wt7m/src/a.py", "tmp/v7fr/draft.md", "slots/rkpr1/build/kept.o"):
                (self.ramdisk / f).write_text("x")
        finally:
            os.umask(old)
        os.chmod(self.ramdisk, 0o1777)
        os.chmod(self.work / "repo" / "src" / "a.py", 0o775)
        (self.work / "claude" / "login.json").write_text("{}")
        os.chmod(self.work / "claude" / "login.json", 0o600)
        os.chmod(self.work / "claude", 0o700)
        os.chmod(self.r("slots/rkpr1/build/kept.o"), 0o644)
        for d, _, fs in os.walk(self.work):
            os.chmod(d, os.lstat(d).st_mode | stat.S_ISGID)
        own = os.getgid()
        others = [g for g in os.getgroups() if g != own]
        self.pr = others[0] if others else own
        # An agent's worktree and a job's tree in the scratch folder, as perms.tree_folder makes their folders.
        for t in ("tmp/v7abc/wt_v1", "tmp/winsim_x"):
            os.chown(self.w(t), -1, self.pr)
            os.chmod(self.w(t), perms.TREE_DIR_MODE)
        uid = os.getuid()
        self.who = types.SimpleNamespace(
            root=uid,
            monitor=uid,
            group=own,
            writer=uid + 100000,
            pr=self.pr,
            runners={"rkpr1": uid, "rkpr2": uid},
        )
        # The system's temporary folders, as the image has them: every user's, sticky.
        self.tmp_dirs = [pathlib.Path(self.tmp.name) / n for n in ("tmp", "var_tmp")]
        for d in self.tmp_dirs:
            d.mkdir()
            os.chown(d, -1, self.pr)
            os.chmod(d, 0o1777)
        (self.tmp_dirs[0] / "left.txt").write_text("x")
        os.chmod(self.tmp_dirs[0] / "left.txt", 0o666)
        self.lines = []
        try:
            os.setxattr(self.work, perms.DEFAULT, perms.acl_bytes(perms.SHARED))
            os.removexattr(self.work, perms.DEFAULT)
        except OSError as exc:
            if exc.errno in (errno.EOPNOTSUPP, errno.ENOTSUP):
                self.skipTest("this file system keeps no ACL")
            raise

    def apply(self, plan=False):
        layout = rk_layout.Layout(
            str(self.work),
            self.who,
            say=self.lines.append,
            plan=plan,
            tmp_dirs=[str(d) for d in self.tmp_dirs],
            ramdisk=str(self.ramdisk),
        )
        layout.apply()
        return layout

    def w(self, rel: str) -> pathlib.Path:
        return self.work / rel

    def r(self, rel: str) -> pathlib.Path:
        return self.ramdisk / rel


class TestOlderVolume(Volume):
    def test_the_working_folder_and_its_shared_folders_get_the_shared_modes(self):
        self.apply()
        self.assertEqual(mode(self.work), 0o2771)
        self.assertTrue(perms.same_acl(perms.default_acl(self.work), perms.SHARED))
        for d in ("monitor", "monitor/state", "logs"):
            self.assertEqual(mode(self.w(d)), 0o2770, d)
            self.assertEqual(gid(self.w(d)), self.who.group, d)
            self.assertTrue(
                perms.same_acl(perms.default_acl(self.w(d)), perms.SHARED), d
            )
        for f in ("monitor/state/state.json", "logs/tick.log", "rocke_merge_queue.md"):
            self.assertEqual(mode(self.w(f)), 0o660, f)
        # what is made there afterwards: 0660 and 2770, whatever the umask
        old = os.umask(0o027)
        try:
            (self.w("monitor/state") / "new.json").write_text("y")
            (self.w("logs") / "sub").mkdir()
        finally:
            os.umask(old)
        self.assertEqual(mode(self.w("monitor/state/new.json")), 0o660)
        self.assertEqual(mode(self.w("logs/sub")), 0o2770)

    def test_a_private_folder_and_file_stay_private(self):
        self.apply()
        self.assertEqual(mode(self.w("claude")), 0o2700)
        self.assertIsNone(perms.default_acl(self.w("claude")))
        self.assertEqual(mode(self.w("claude/login.json")), 0o600)

    def test_the_trees_pr_code_reads_are_the_runners_groups_to_read(self):
        self.apply()
        for d in ("repo", "repo/src", "repo/.git", "wt7", "wt7/src"):
            self.assertEqual(mode(self.w(d)), 0o2750, d)
            self.assertEqual(gid(self.w(d)), self.pr, d)
            self.assertIsNone(perms.default_acl(self.w(d)), d)
        self.assertEqual(
            mode(self.w("repo/src/a.py")), 0o750
        )  # its owner ran it: the group runs it too
        for f in ("repo/.git/config", "wt7/src/a.py"):
            self.assertEqual(mode(self.w(f)), 0o640, f)
            self.assertEqual(gid(self.w(f)), self.pr, f)
            self.assertIsNone(access(self.w(f)), f)

    def test_the_scratch_folder_lets_the_runners_group_enter_and_read_nothing(self):
        self.apply()
        self.assertEqual(mode(self.w("tmp")), 0o2771)
        self.assertTrue(
            perms.same_acl(perms.default_acl(self.w("tmp")), perms.scratch(self.pr))
        )
        self.assertEqual(runners(self.w("tmp/v7abc"), self.pr), "x")
        self.assertEqual(runners(self.w("tmp/v7abc/draft.md"), self.pr), "")
        self.assertEqual(mode(self.w("tmp/v7abc/draft.md")), 0o660)

    def test_a_tree_in_the_scratch_folder_keeps_its_modes_and_all_it_holds(self):
        self.apply()
        # an agent's worktree, a job's tree
        for d in ("tmp/v7abc/wt_v1", "tmp/winsim_x"):
            self.assertEqual(mode(self.w(d)), perms.TREE_DIR_MODE, d)
            self.assertEqual(gid(self.w(d)), self.pr, d)
            self.assertIsNone(perms.default_acl(self.w(d)), d)
            self.assertIsNone(access(self.w(d)), d)
        # what they hold, as the umask made it
        self.assertEqual(mode(self.w("tmp/v7abc/wt_v1/src")), 0o2775)
        for f in ("tmp/v7abc/wt_v1/src/a.py", "tmp/winsim_x/a.txt"):
            self.assertEqual(mode(self.w(f)), 0o664, f)
            self.assertIsNone(access(self.w(f)), f)

    def test_the_cache_keeps_the_develop_side_results_and_the_mutation_history_as_trees(
        self,
    ):
        self.apply()
        self.assertEqual(mode(self.w("cache")), 0o2771)
        self.assertEqual(mode(self.w("cache/prcode")), 0o2770)
        self.assertEqual(mode(self.w("cache/prcode/golden.lines")), 0o660)
        for d in (f"cache/{name}" for name in paths.CACHES):
            self.assertEqual(mode(self.w(d)), 0o2750, d)
            self.assertEqual(gid(self.w(d)), self.pr, d)
            self.assertIsNone(perms.default_acl(self.w(d)), d)

    def test_the_slots_are_made_and_a_slots_contents_are_never_walked(self):
        self.apply()
        self.assertEqual(mode(self.r("slots")), 0o755)
        self.assertIsNone(perms.default_acl(self.r("slots")))
        for r in ("rkpr1", "rkpr2"):
            self.assertEqual(mode(self.r(f"slots/{r}")), 0o2751, r)
            self.assertEqual(gid(self.r(f"slots/{r}")), self.who.group, r)
            self.assertEqual(mode(self.r(f"slots/inputs/{r}")), 0o2750, r)
            self.assertEqual(gid(self.r(f"slots/inputs/{r}")), self.pr, r)
        self.assertEqual(
            mode(self.r("slots/rkpr1/build/kept.o")), 0o644
        )  # a kept slot's build
        self.assertEqual(mode(self.r("slots/locks")), 0o2770)
        self.assertTrue(
            perms.same_acl(perms.default_acl(self.r("slots/locks")), perms.SHARED)
        )
        self.assertEqual(mode(self.r("slots/inputs")), 0o750)
        self.assertEqual(gid(self.r("slots/inputs")), self.pr)
        self.assertFalse(self.w("slots").exists())  # the slots are on the RAM disk

    def test_the_ramdisk_folder_is_laid_out_as_the_working_folder(self):
        self.apply()
        self.assertEqual(mode(self.ramdisk), 0o2771)
        self.assertEqual(gid(self.ramdisk), self.who.group)
        self.assertTrue(perms.same_acl(perms.default_acl(self.ramdisk), perms.SHARED))
        self.assertIn(
            f"rk_layout: {self.ramdisk}: the monitor's, group rkmon, 2771: a PR runner passes through it",
            self.lines,
        )
        # its scratch folder, paths.tmp while the RAM disk is on: the runners' group enters the review directories, and
        # reads none of their files
        self.assertEqual(mode(self.r("tmp")), 0o2771)
        self.assertTrue(
            perms.same_acl(perms.default_acl(self.r("tmp")), perms.scratch(self.pr))
        )
        self.assertEqual(runners(self.r("tmp/v7fr"), self.pr), "x")
        self.assertEqual(runners(self.r("tmp/v7fr/draft.md"), self.pr), "")
        # a PR's worktree there: a tree
        for d in ("wt7m", "wt7m/src"):
            self.assertEqual(mode(self.r(d)), 0o2750, d)
            self.assertEqual(gid(self.r(d)), self.pr, d)
        self.assertEqual(mode(self.r("wt7m/src/a.py")), 0o640)

    def test_a_container_without_a_ramdisk_gets_a_folder_of_its_own(self):
        shutil.rmtree(self.ramdisk)
        self.apply()
        self.assertEqual(mode(self.ramdisk), 0o2771)
        self.assertTrue(perms.same_acl(perms.default_acl(self.ramdisk), perms.SHARED))
        self.assertEqual(sorted(os.listdir(self.ramdisk)), ["slots", "tmp"])
        self.assertTrue(
            perms.same_acl(perms.default_acl(self.r("tmp")), perms.scratch(self.pr))
        )
        self.assertEqual(mode(self.r("slots/rkpr2")), 0o2751)

    def test_the_system_temporary_folders_are_closed_to_the_runners(self):
        self.apply()
        for d in self.tmp_dirs:
            self.assertEqual(mode(d), 0o1770, d)
            self.assertEqual(gid(d), self.who.group, d)
            self.assertIsNone(perms.default_acl(d), d)
        self.assertEqual(
            mode(self.tmp_dirs[0] / "left.txt"), 0o666
        )  # what is in them: as it is
        self.assertIn(
            f"rk_layout: {self.tmp_dirs[0]}: root's, group rkmon, 1770: no PR runner writes or reads it",
            self.lines,
        )

    def test_the_folders_the_monitors_code_fills_are_made_as_trees(self):
        self.apply()
        for d in ("tools", "probes"):
            self.assertEqual(mode(self.w(d)), 0o2750, d)
            self.assertEqual(gid(self.w(d)), self.pr, d)
            self.assertEqual(os.listdir(self.w(d)), [], d)

    def test_a_second_start_changes_nothing_and_walks_nothing(self):
        self.apply()
        self.lines.clear()
        with mock.patch.object(
            rk_layout.Layout, "walk", side_effect=AssertionError("walked")
        ):
            self.apply()
        self.assertEqual(self.lines, [])


class TestRefusals(Volume):
    def test_nothing_is_done_through_a_link(self):
        (self.w("monitor") / "link").symlink_to(self.outside / "f")
        (self.w("wt9")).symlink_to(self.outside)
        (self.w("repo/src") / "dir").symlink_to(self.outside)
        os.rmdir(self.tmp_dirs[1])
        self.tmp_dirs[1].symlink_to(self.outside)
        outside = mode(self.outside)
        self.apply()
        self.assertEqual(mode(self.outside / "f"), 0o644)
        self.assertEqual(mode(self.outside), outside)
        self.assertTrue(os.path.islink(self.w("wt9")))
        self.assertTrue(
            any(
                f"{self.tmp_dirs[1]} is a link, no folder or missing" in ln
                for ln in self.lines
            )
        )

    def test_a_ramdisk_folder_that_is_a_link_gets_no_slots(self):
        shutil.rmtree(self.ramdisk)
        self.ramdisk.symlink_to(self.outside)
        outside = mode(self.outside)
        self.apply()
        self.assertEqual(mode(self.outside), outside)
        self.assertEqual(os.listdir(self.outside), ["f"])
        self.assertTrue(
            any(f"{self.ramdisk} is a link or no folder" in ln for ln in self.lines)
        )
        # nor one whose parent is missing: none is made
        self.lines.clear()
        none = self.outside / "none" / "ramdisk"
        rk_layout.Layout(
            str(self.work), self.who, say=self.lines.append, ramdisk=str(none)
        ).apply()
        self.assertFalse(none.parent.exists())
        self.assertTrue(any("no RAM disk folder made" in ln for ln in self.lines))

    def test_an_entry_another_user_owns_is_left_as_it_is(self):
        layout = rk_layout.Layout(str(self.work), self.who, say=self.lines.append)
        layout.owners = {
            os.getuid() + 1
        }  # neither the monitor nor the writer owns this user's files
        fd = os.open(self.w("logs"), os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.assertEqual(layout.walk(fd, rk_layout.SHARED), 0)
        finally:
            os.close(fd)
        self.assertEqual(mode(self.w("logs/tick.log")), 0o664)

    def test_a_working_folder_that_is_not_the_monitors_is_left_as_it_is(self):
        self.who.monitor = self.who.writer = os.getuid() + 1
        f = self.w("logs/tick.log")
        before = mode(f)
        rk_layout.Layout(str(self.work), self.who, say=self.lines.append).work_folder()
        self.assertEqual(mode(f), before)
        self.assertTrue(any("is not the monitor's" in ln for ln in self.lines))

    def test_plan_changes_nothing_and_says_what_would_change(self):
        def modes():
            return {
                p: mode(p)
                for p in (*self.work.rglob("*"), self.ramdisk, *self.ramdisk.rglob("*"))
            }

        before = modes()
        self.apply(plan=True)
        self.assertEqual(modes(), before)
        self.assertEqual([mode(d) for d in self.tmp_dirs], [0o1777, 0o1777])
        self.assertFalse(self.w("tools").exists())
        self.assertTrue(
            any(
                "would be given the permissions of a tree folder" in ln
                for ln in self.lines
            )
        )
        self.assertTrue(any("tools would be made" in ln for ln in self.lines))

    def test_a_file_system_with_no_acl_gets_the_modes_alone(self):
        def refuse(*a, **k):
            raise OSError(errno.EOPNOTSUPP, "Operation not supported")

        with mock.patch.object(rk_layout.os, "setxattr", refuse), mock.patch.object(
            rk_layout.os, "removexattr", refuse
        ), mock.patch.object(rk_layout.os, "getxattr", refuse):
            self.apply()
        self.assertEqual(mode(self.w("monitor/state/state.json")), 0o660)
        self.assertEqual(mode(self.w("repo/src")), 0o2750)
        said = [ln for ln in self.lines if "keeps no ACL" in ln]
        self.assertEqual(len(said), 1)


class TestMain(unittest.TestCase):
    def test_an_image_without_its_users_changes_nothing(self):
        with mock.patch.object(
            rk_layout.pwd, "getpwnam", side_effect=KeyError("rkmon")
        ), mock.patch.object(
            rk_layout.Layout, "apply", side_effect=AssertionError("applied")
        ), mock.patch(
            "sys.stderr"
        ):
            self.assertEqual(rk_layout.main([]), 1)

    def test_each_start_lays_out_the_images_folders(self):
        """The working folder, the RAM disk's folder with the slots, and the system's temporary folders."""
        seen = []
        with mock.patch.object(
            rk_layout, "ids", lambda: types.SimpleNamespace(monitor=1, writer=2)
        ), mock.patch.object(
            rk_layout.Layout,
            "apply",
            lambda self: seen.append(
                (self.work, self.ramdisk, self.tmp_dirs, self.plan)
            ),
        ):
            self.assertEqual(rk_layout.main([]), 0)
            self.assertEqual(rk_layout.main(["--plan"]), 0)
        lay = rk_layout.LAYOUT
        self.assertEqual(
            seen,
            [
                (lay.WORK_DIR, lay.RAMDISK, lay.SYSTEM_TMP, False),
                (lay.WORK_DIR, lay.RAMDISK, lay.SYSTEM_TMP, True),
            ],
        )

    def test_the_trees_are_the_layouts_names(self):
        """TREES and SCRATCH and CACHE name the default layout's folders (rkmon_lib/paths.py UNDER_WORKING and
        worktree, deploy.venv_commands); the code is in a volume of its own, root's (container_layout.CODE_DIR), and
        none of them."""
        sys.path.insert(0, str(ROOT))
        from rkmon_lib import container_layout, paths

        under = paths.UNDER_WORKING
        for key in ("repo", "worker_tools", "probes", "venv"):
            self.assertIn(under[key], rk_layout.TREES, key)
        self.assertEqual(rk_layout.SCRATCH, under["tmp"])
        self.assertEqual(rk_layout.CACHE, under["cache"])
        self.assertNotIn("code", rk_layout.TREES + rk_layout.MADE)
        self.assertFalse(
            container_layout.CODE_DIR.startswith(container_layout.WORK_DIR + "/")
        )
        self.assertIn("venv-*", rk_layout.TREES)  # deploy.venv_commands: venv-<which>
        # paths.worktree: wt<N>, wt<N>m, wtdev, wtbase, wtbatch, wtsanity, wtbisect, in the working folder
        cfg = {"paths": {"working_dir": "/w"}}
        for name in (
            "wt7",
            "wt7m",
            "wtdev",
            "wtbase",
            "wtbatch",
            "wtsanity",
            "wtbisect",
        ):
            self.assertEqual(paths.worktree(cfg, name), f"/w/{name}")
            self.assertEqual(rk_layout.Layout.kind_of(name), rk_layout.TREE, name)
        for name in (
            "monitor",
            "logs",
            "claude",
            "workers",
            "lessons",
            "scratch",
            "build",
            "agentic",
            ".config",
        ):
            self.assertEqual(rk_layout.Layout.kind_of(name), rk_layout.SHARED, name)


if __name__ == "__main__":
    unittest.main()
