# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The modes of the container's folders (worker/perms.py).

- An ACL's entries and its extended attribute's value turn into each other, in the kernel's order; a value that is no
  POSIX ACL is refused.
- Under umask 027, a folder with the SHARED default ACL gives what is made in it 0660 and 2770 (its group's, by its
  setgid bit), and its subfolders the same ACL; SCRATCH adds the runners' group, which enters the folders made there
  and reads none of the files (a program made there it may run, and never read).
- `tree_folder` makes the empty folder a tree PR code reads goes into: the group asked for, 2750, no ACL (an inherited
  one dropped), so what the umask makes there is 0640; a plain folder where the group does not exist; a tree that is
  there already is left as it is, and a link or a folder holding something else is refused and named. `is_tree` tells
  such a folder: a folder, the group's, 2750.
- The command line: `tree --group GROUP DIR...` exits 0, 1 when a folder could not be made so, 2 on a usage error.

The ACL cases skip on a file system that keeps no ACL.
"""

import errno
import grp
import io
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

import perms  # noqa: E402


def mode(p) -> int:
    return stat.S_IMODE(os.lstat(p).st_mode)


def runners(p, gid: int) -> str:
    """What the access ACL of `p` gives the group `gid` (a named group entry), through its mask: "rwx"-style."""
    entries = perms.acl_entries(os.getxattr(p, perms.ACCESS))
    named = next(e[2] for e in entries if e[0] == "group" and e[1] == gid)
    mask = next(e[2] for e in entries if e[0] == "mask")
    return "".join(c for c in named if c in mask)


class Folder(unittest.TestCase):
    """A temporary folder, umask 027 while a test runs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        old = os.umask(perms.UMASK)
        self.addCleanup(os.umask, old)
        self.group = grp.getgrgid(os.getgid()).gr_name  # a group this user is in

    def acls_or_skip(self, path, entries) -> None:
        try:
            perms.set_default_acl(path, entries)
        except OSError as exc:
            if exc.errno in (errno.EOPNOTSUPP, errno.ENOTSUP):
                self.skipTest("this file system keeps no ACL")
            raise


class TestAclValue(unittest.TestCase):
    def test_entries_and_value_turn_into_each_other_in_the_kernels_order(self):
        entries = (
            ("other", None, ""),
            ("group", 1998, "rx"),
            ("user_obj", None, "rwx"),
            ("mask", None, "rwx"),
            ("group_obj", None, "rw"),
        )
        back = perms.acl_entries(perms.acl_bytes(entries))
        self.assertEqual(
            back,
            (
                ("user_obj", None, "rwx"),
                ("group_obj", None, "rw"),
                ("group", 1998, "rx"),
                ("mask", None, "rwx"),
                ("other", None, ""),
            ),
        )
        self.assertTrue(perms.same_acl(back, entries))
        self.assertFalse(perms.same_acl(back, perms.SHARED))
        self.assertFalse(perms.same_acl(None, perms.SHARED))

    def test_a_value_that_is_no_posix_acl_is_refused(self):
        for data in (b"", b"\x01\x00\x00\x00", b"abc"):
            with self.assertRaises(ValueError):
                perms.acl_entries(data)

    def test_scratch_adds_the_runners_group_entering_only(self):
        self.assertIn(("group", 1998, "x"), perms.scratch(1998))
        self.assertIn(("mask", None, "rwx"), perms.scratch(1998))
        self.assertEqual(perms.scratch(1998)[:2], perms.SHARED[:2])


class TestDefaultAcl(Folder):
    def test_a_shared_folder_gives_0660_files_and_2770_folders_whatever_the_umask(self):
        top = self.d / "shared"
        top.mkdir()
        os.chmod(top, perms.SHARED_DIR_MODE)
        self.acls_or_skip(top, perms.SHARED)
        self.assertTrue(perms.same_acl(perms.default_acl(top), perms.SHARED))
        (top / "f").write_text("x")
        (top / "sub").mkdir()
        (top / "sub" / "g").write_text("y")
        self.assertEqual(mode(top / "f"), perms.SHARED_FILE_MODE)
        self.assertEqual(mode(top / "sub"), perms.SHARED_DIR_MODE)
        self.assertEqual(mode(top / "sub" / "g"), perms.SHARED_FILE_MODE)
        self.assertTrue(perms.same_acl(perms.default_acl(top / "sub"), perms.SHARED))

    def test_a_scratch_folder_lets_the_runners_group_enter_what_is_made_there_and_read_none(
        self,
    ):
        top = self.d / "tmp"
        top.mkdir()
        os.chmod(top, 0o2771)
        gid = os.getgid()
        self.acls_or_skip(top, perms.scratch(gid))
        review = top / "review"
        review.mkdir()
        (review / "draft.md").write_text("x")
        # a build's program
        os.close(os.open(review / "prog", os.O_CREAT | os.O_WRONLY, 0o777))
        self.assertEqual(runners(review, gid), "x")
        self.assertEqual(runners(review / "draft.md", gid), "")
        self.assertEqual(runners(review / "prog", gid), "x")  # it runs, never reads
        self.assertTrue(perms.same_acl(perms.default_acl(review), perms.scratch(gid)))

    def test_a_folder_with_no_default_acl_has_none_and_dropping_none_is_no_error(self):
        (self.d / "plain").mkdir()
        self.assertIsNone(perms.default_acl(self.d / "plain"))
        perms.drop_acls(self.d / "plain")
        self.assertIsNone(perms.default_acl(self.d / "plain"))

    def test_drop_acls_removes_an_access_acl_only_when_the_folder_has_one(self):
        """For a user in no group of the folder's, the kernel takes the folder's set-group-ID bit with any access ACL
        removed, even one that is not there."""
        top = self.d / "plain"
        top.mkdir()
        real, removed = os.removexattr, []

        def removexattr(path, name, *args, **kwargs):
            removed.append(name)
            return real(path, name, *args, **kwargs)

        with mock.patch.object(perms.os, "removexattr", removexattr):
            perms.drop_acls(top)
            self.assertNotIn(perms.ACCESS, removed)
            access = (
                ("user_obj", None, "rwx"),
                ("user", os.getuid() + 1, "rx"),
                ("group_obj", None, "rx"),
                ("mask", None, "rx"),
                ("other", None, "x"),
            )
            try:
                os.setxattr(top, perms.ACCESS, perms.acl_bytes(access))
            except OSError:
                self.skipTest("this file system keeps no ACL")
            perms.drop_acls(top)
        self.assertIn(perms.ACCESS, removed)
        with self.assertRaises(OSError) as got:
            os.getxattr(top, perms.ACCESS)
        self.assertEqual(got.exception.errno, errno.ENODATA)

    def test_drop_acls_removes_the_default_acl(self):
        top = self.d / "shared"
        top.mkdir()
        self.acls_or_skip(top, perms.SHARED)
        perms.drop_acls(top)
        self.assertIsNone(perms.default_acl(top))


class TestTreeFolder(Folder):
    def test_a_tree_folder_is_the_groups_2750_with_no_acl_and_its_files_0640(self):
        top = self.d / "work"
        top.mkdir()
        self.acls_or_skip(
            top, perms.SHARED
        )  # the working folder's own ACL, which the tree must not inherit
        wt = top / "wt7"
        self.assertEqual(perms.tree_folder(str(wt), self.group), "")
        s = os.lstat(wt)
        self.assertEqual(stat.S_IMODE(s.st_mode), perms.TREE_DIR_MODE)
        self.assertEqual(s.st_gid, os.getgid())
        self.assertIsNone(perms.default_acl(wt))
        (wt / "a.py").write_text("x")
        (wt / "sub").mkdir()
        self.assertEqual(mode(wt / "a.py"), perms.TREE_FILE_MODE)
        self.assertEqual(mode(wt / "sub"), perms.TREE_DIR_MODE)

    def test_an_empty_folder_there_already_is_made_a_tree_folder(self):
        wt = self.d / "wt7"
        wt.mkdir(mode=0o700)
        self.assertEqual(perms.tree_folder(str(wt), self.group), "")
        self.assertEqual(mode(wt), perms.TREE_DIR_MODE)

    def test_a_tree_there_already_is_left_as_it_is(self):
        wt = self.d / "wt7"
        self.assertEqual(perms.tree_folder(str(wt), self.group), "")
        (wt / "a.py").write_text("x")
        self.assertEqual(perms.tree_folder(str(wt), self.group), "")
        self.assertEqual((wt / "a.py").read_text(), "x")

    def test_is_tree_tells_a_tree_folder(self):
        gid = os.getgid()
        wt, shared, f = self.d / "wt7", self.d / "shared", self.d / "f"
        self.assertEqual(perms.tree_folder(str(wt), self.group), "")
        shared.mkdir()
        os.chmod(shared, perms.SHARED_DIR_MODE)
        f.write_text("x")
        os.chmod(f, perms.TREE_DIR_MODE)
        self.assertTrue(perms.is_tree(os.lstat(wt), gid))
        self.assertFalse(perms.is_tree(os.lstat(wt), gid + 1))  # another group's
        self.assertFalse(perms.is_tree(os.lstat(shared), gid))  # 2770
        self.assertFalse(perms.is_tree(os.lstat(f), gid))  # no folder

    def test_a_folder_holding_something_else_is_refused_and_named(self):
        wt = self.d / "wt7"
        wt.mkdir(mode=0o700)
        (wt / "a.py").write_text("x")
        why = perms.tree_folder(str(wt), self.group)
        self.assertIn(f"{wt}: not empty", why)
        self.assertEqual(mode(wt), 0o700)

    def test_a_link_is_refused_and_its_target_left_as_it_is(self):
        target = self.d / "elsewhere"
        target.mkdir(mode=0o700)
        (self.d / "wt7").symlink_to(target)
        why = perms.tree_folder(str(self.d / "wt7"), self.group)
        self.assertIn("a link", why)
        self.assertEqual(mode(target), 0o700)

    def test_a_missing_parent_is_named(self):
        why = perms.tree_folder(str(self.d / "no" / "wt7"), self.group)
        self.assertIn("not made", why)

    def test_where_the_group_does_not_exist_a_plain_folder_is_made(self):
        def none(name):
            raise KeyError(name)

        wt = self.d / "wt7"
        self.assertEqual(perms.tree_folder(str(wt), "rkpr", getgrnam=none), "")
        self.assertTrue(wt.is_dir())
        self.assertEqual(mode(wt), 0o750)


class TestCommandLine(Folder):
    def test_tree_folders_exit_0(self):
        a, b = self.d / "a", self.d / "b"
        self.assertEqual(perms.main(["tree", "--group", self.group, str(a), str(b)]), 0)
        self.assertEqual((mode(a), mode(b)), (perms.TREE_DIR_MODE,) * 2)

    def test_a_folder_not_made_so_exits_1_and_says_why(self):
        (self.d / "a").mkdir()
        (self.d / "a" / "f").write_text("x")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = perms.main(["tree", "--group", self.group, str(self.d / "a")])
        self.assertEqual(rc, 1)
        self.assertIn("perms: ", err.getvalue())
        self.assertIn("not empty", err.getvalue())

    def test_a_usage_error_exits_2(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(perms.main(["tree", str(self.d)]), 2)
            self.assertEqual(perms.main([]), 2)
        self.assertIn("usage:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
