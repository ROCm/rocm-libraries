# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A file written whole (rkmon_lib/whole_file.py): the text or the bytes at the mode given whatever the umask, with no
file left beside it; the old file kept on a failure; never a new file made through a link; an owner given by the new
file's descriptor; the syncs of a durable write; and its one home: the state's writer, the dashboard's pages and the
private files write through it, and the new file's name and the folder's sync are its own.
"""

import fnmatch
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import private_files, state_store, webdash, whole_file  # noqa: E402


class TestWholeFile(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)

    def names(self) -> list:
        return sorted(p.name for p in self.d.iterdir())

    def test_the_text_at_its_mode_and_nothing_beside_it(self):
        page = self.d / "p.html"
        umask = os.umask(0o077)
        try:
            whole_file.write(page, "<p>é</p>", 0o644)
        finally:
            os.umask(umask)
        self.assertEqual(page.read_text(encoding="utf-8"), "<p>é</p>")
        self.assertEqual(page.stat().st_mode & 0o777, 0o644)  # the umask took no bit
        whole_file.write(page, b"<p>two</p>", 0o600)
        self.assertEqual(
            (page.read_bytes(), page.stat().st_mode & 0o777), (b"<p>two</p>", 0o600)
        )
        self.assertEqual(self.names(), ["p.html"])

    def test_a_failure_keeps_the_old_file(self):
        page = self.d / "p.html"
        whole_file.write(page, "old", 0o644)
        with mock.patch.object(
            os, "replace", side_effect=OSError(28, "No space left on device")
        ):
            with self.assertRaises(OSError):
                whole_file.write(page, "new", 0o644)
        self.assertEqual(page.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.names(), ["p.html"])

    def test_the_new_file_is_never_made_through_a_link(self):
        page, theirs = self.d / "p.html", self.d / "theirs"
        theirs.write_text("theirs", encoding="utf-8")
        with mock.patch.object(os, "urandom", return_value=bytes(4)):
            new = whole_file.beside(page)
            self.assertEqual(new.name, f".p.html.{os.getpid()}.00000000.new")
            new.symlink_to(theirs)
            with self.assertRaises(FileExistsError):
                whole_file.write(page, "mine", 0o644)
        self.assertEqual(theirs.read_text(encoding="utf-8"), "theirs")
        self.assertFalse(page.exists())

    def test_an_owner_is_given_by_the_new_file_s_descriptor_before_it_is_filled(self):
        """OWNER: the new file's owner and group, given by its open file (never by a path, which another user may have
        planted) while it is still empty; when that fails, the old file stays and nothing is left beside it.
        """
        page, given = self.d / "p.json", []
        by_path = AssertionError("an owner given by a path")
        with (
            mock.patch.object(
                os,
                "fchown",
                lambda fd, uid, gid: given.append((os.fstat(fd).st_size, uid, gid)),
            ),
            mock.patch.object(os, "chown", side_effect=by_path),
            mock.patch.object(os, "lchown", side_effect=by_path),
        ):
            whole_file.write(page, "{}\n", 0o640, owner=(4242, 4343))
            whole_file.write(page, "{}\n", 0o640)  # no owner: none given
        self.assertEqual(given, [(0, 4242, 4343)])
        self.assertEqual(page.read_text(encoding="utf-8"), "{}\n")
        with mock.patch.object(
            os, "fchown", side_effect=OSError(1, "Operation not permitted")
        ):
            with self.assertRaises(OSError):
                whole_file.write(page, "[]\n", 0o640, owner=(4242, 4343))
        self.assertEqual(page.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(self.names(), ["p.json"])

    def test_a_durable_write_syncs_the_file_then_its_folder(self):
        page = self.d / "p.json"
        with mock.patch.object(os, "fsync", wraps=os.fsync) as synced:
            whole_file.write(page, "{}\n", 0o644)
            self.assertEqual(synced.call_count, 0)
            whole_file.write(page, "{}\n", 0o644, durable=True)
            self.assertEqual(synced.call_count, 2)
        with mock.patch.object(
            os, "fsync", side_effect=OSError(5, "Input/output error")
        ):
            with self.assertRaises(OSError):
                whole_file.write(page, "[]\n", 0o644, durable=True)
        self.assertEqual(page.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(self.names(), ["p.json"])
        whole_file.sync_folder(self.d / "no such folder")  # nothing when it cannot

    def test_every_writer_writes_through_it(self):
        for module, name in (
            (state_store, "_sync_folder"),
            (private_files, "_sync_folder"),
            (private_files, "_fill"),
            (private_files, "NEW"),
            (private_files, "NEW_FILE"),
            (webdash, "write_whole"),
        ):
            self.assertFalse(hasattr(module, name), f"{module.__name__}.{name}")
        with mock.patch.object(whole_file, "write") as wrote:
            state_store.write_file(self.d / "s.json", {"b": 1, "a": "é"})
        self.assertEqual(
            wrote.call_args.args[:2], (self.d / "s.json", '{\n "a": "é",\n "b": 1\n}\n')
        )
        self.assertEqual(wrote.call_args.kwargs, {"durable": True})
        # A private file's kin are the new file a write of it leaves when it stops part-way.
        new = whole_file.beside(self.d / ".env").name
        self.assertTrue(fnmatch.fnmatchcase(new, private_files.own_names(".env")[2]))


if __name__ == "__main__":
    unittest.main()
