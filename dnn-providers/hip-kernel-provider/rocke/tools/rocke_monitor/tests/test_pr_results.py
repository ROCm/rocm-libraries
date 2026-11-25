# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The results a job left in its slot, read as data (worker/pr_results.py).

- Only regular files with one link, the slot owner's, each up to a size cap, are read; a link (at a file, a folder on
  the way, or the start's own name), a FIFO, a hard link and a file past the cap are refused and named, never
  followed, never read, and a FIFO never blocks the reader.
- The JSON a job wrote is read by those rules and parsed as data: what holds no JSON, one nested past what the parser
  goes included, is refused, never a ValueError or a RecursionError.
- Another user's folder is read only once no process of that user runs. The processes are read from /proc, each one's
  id, real and effective user ids, parent and state; one that cannot be read is passed over.
- A copy keeps what the rules allow, names the rest, stops at its total cap and at its cap on names, opens no folder
  deeper than its depth cap (the launcher's too), never writes through a link it finds at its destination, and gives
  what it writes explicit permissions (files 0660 or 0770, folders 2770).
- The command line: exit 0 when nothing was refused, 1 when something was, 2 when nothing could be read.
"""

import io
import os
import pathlib
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

import pr_results  # noqa: E402


class Slot(unittest.TestCase):
    """A slot's `out` folder in a temporary folder, and a file of the monitor's beside it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        top = pathlib.Path(self.tmp.name)
        self.slot = top / "slot"
        self.out = self.slot / "out"
        (self.out / "sub").mkdir(parents=True)
        self.mine = top / "monitor"
        self.mine.mkdir()
        self.secret = self.mine / "state.json"
        self.secret.write_text("the monitor's\n")
        self.dest = top / "copy"


class TestRead(Slot):
    def test_a_regular_file_is_read(self):
        (self.out / "sub" / "r.txt").write_text("result\n")
        self.assertEqual(pr_results.read(str(self.out), "sub/r.txt"), b"result\n")

    def test_a_link_to_a_monitor_file_is_refused(self):
        (self.out / "r.txt").symlink_to(self.secret)
        with self.assertRaisesRegex(pr_results.Refused, "r.txt: a link"):
            pr_results.read(str(self.out), "r.txt")

    def test_a_link_on_the_way_is_refused(self):
        (self.out / "via").symlink_to(self.mine)
        with self.assertRaisesRegex(pr_results.Refused, "via is a link"):
            pr_results.read(str(self.out), "via/state.json")

    def test_a_link_at_the_starts_own_name_is_refused(self):
        os.rename(self.out, self.slot / "was_out")
        self.out.symlink_to(self.mine)
        with self.assertRaisesRegex(pr_results.Refused, "a link or no folder"):
            pr_results.read(str(self.out), "state.json")

    def test_a_fifo_is_refused_without_blocking(self):
        """The read runs in a thread joined with a time limit: a read that blocks fails the test, never hangs it."""
        os.mkfifo(self.out / "f")
        got = []

        def read():
            try:
                pr_results.read(str(self.out), "f")
            except pr_results.Refused as exc:
                got.append(str(exc))

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(10)
        if t.is_alive():  # blocked opening the FIFO: a writer that opens it lets it go
            os.close(os.open(self.out / "f", os.O_WRONLY | os.O_NONBLOCK))
            t.join(10)
            self.fail("reading a FIFO blocked")
        self.assertEqual(len(got), 1, got)
        self.assertIn("f: not a regular file", got[0])

    def test_a_hard_link_is_refused(self):
        (self.out / "a").write_text("x")
        os.link(self.out / "a", self.out / "b")
        with self.assertRaisesRegex(pr_results.Refused, "b: 2 links"):
            pr_results.read(str(self.out), "b")

    def test_a_file_of_another_owner_is_refused(self):
        (self.out / "r.txt").write_text("x")
        fd = os.open(self.out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with self.assertRaisesRegex(pr_results.Refused, "not the slot owner's"):
                pr_results._open_file(fd, "r.txt", os.getuid() + 1, "r.txt")
        finally:
            os.close(fd)

    def test_a_file_past_the_cap_is_refused(self):
        (self.out / "big").write_bytes(b"x" * 100)
        with mock.patch.object(pr_results, "FILE_MAX", 10):
            with self.assertRaisesRegex(pr_results.Refused, "big: over 10 bytes"):
                pr_results.read(str(self.out), "big")
        with self.assertRaisesRegex(pr_results.Refused, "big: over 50 bytes"):
            pr_results.read(str(self.out), "big", cap=50)

    def test_dot_dot_is_refused(self):
        with self.assertRaisesRegex(pr_results.Refused, "'..'"):
            pr_results.read(str(self.out), "../monitor/state.json")

    def fake_proc(self, name: str, procs: list[tuple[int, int, str]]) -> str:
        """A /proc with these (pid, uid, state) processes."""
        proc = pathlib.Path(self.tmp.name) / name
        for pid, uid, state in procs:
            (proc / str(pid)).mkdir(parents=True)
            (proc / str(pid) / "status").write_text(
                f"Name:\tx\nState:\t{state}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
            )
        (proc / "self").mkdir(parents=True, exist_ok=True)
        return str(proc)

    def test_each_process_by_its_ids_its_parent_and_its_state(self):
        me = os.getuid()
        proc = self.fake_proc(
            "p3", [(7, me, "S (sleeping)"), (8, me + 5, "Z (zombie)")]
        )
        with open(pathlib.Path(proc) / "7" / "status", "a") as f:
            f.write("PPid:\t1\n")
        (pathlib.Path(proc) / "9").mkdir()  # gone while /proc was read: no status
        self.assertEqual(
            sorted(pr_results.processes(proc)),
            [(7, (me, me), 1, "S"), (8, (me + 5, me + 5), 0, "Z")],
        )
        mine = [p for p in pr_results.processes() if p[0] == os.getpid()]
        self.assertEqual(
            mine, [(os.getpid(), (os.getuid(), os.geteuid()), os.getppid(), "R")]
        )

    def test_another_users_folder_is_read_only_once_its_processes_are_gone(self):
        (self.out / "r.txt").write_text("result\n")
        me = os.getuid()
        proc = self.fake_proc("p1", [(7, me, "S (sleeping)"), (8, me + 5, "R")])
        self.assertEqual(pr_results.running(me, proc), [7])
        with mock.patch.object(pr_results.os, "geteuid", lambda: me + 5):
            with self.assertRaisesRegex(pr_results.Refused, "1 process"):
                pr_results.open_root(str(self.out), proc)
        zombie = self.fake_proc("p2", [(9, me, "Z (zombie)")])
        self.assertEqual(pr_results.running(me, zombie), [])
        with mock.patch.object(pr_results.os, "geteuid", lambda: me + 5):
            fd, owner = pr_results.open_root(str(self.out), zombie)
        os.close(fd)
        self.assertEqual(owner, me)
        # The reader's own folder (a host's build folder): its own processes run, of course.
        fd, _owner = pr_results.open_root(str(self.out), proc)
        os.close(fd)


class TestReadJson(Slot):
    def test_json_is_read_as_data(self):
        (self.out / "sub" / "l.json").write_text('{"a": [1, 2]}')
        self.assertEqual(
            pr_results.read_json(str(self.out), "sub/l.json"), {"a": [1, 2]}
        )

    def test_what_holds_no_json_is_refused(self):
        """Not JSON, not text, or nested deeper than the parser goes: Refused, never a ValueError or a
        RecursionError."""
        for data in (b"{", b'"\x80"', b"[" * 200000 + b"]" * 200000):
            (self.out / "l.json").write_bytes(data)
            with self.assertRaisesRegex(pr_results.Refused, "l.json: no JSON"):
                pr_results.read_json(str(self.out), "l.json")

    def test_the_readers_rules_hold(self):
        (self.out / "l.json").symlink_to(self.secret)
        with self.assertRaisesRegex(pr_results.Refused, "l.json: a link"):
            pr_results.read_json(str(self.out), "l.json")
        (self.out / "big.json").write_text("[" + "1, " * 40 + "1]")
        with self.assertRaisesRegex(pr_results.Refused, "big.json: over 50 bytes"):
            pr_results.read_json(str(self.out), "big.json", cap=50)


class TestCopy(Slot):
    def test_a_tree_keeps_its_regular_files_and_names_the_rest(self):
        (self.out / "a.txt").write_text("a")
        (self.out / "sub" / "b.sh").write_text("b")
        os.chmod(self.out / "sub" / "b.sh", 0o750)
        (self.out / "link").symlink_to(self.secret)
        (self.out / "dirlink").symlink_to(self.mine)
        os.mkfifo(self.out / "sub" / "fifo")
        refused = pr_results.copy(str(self.out), "", str(self.dest))
        self.assertEqual((self.dest / "a.txt").read_text(), "a")
        self.assertEqual((self.dest / "sub" / "b.sh").read_text(), "b")
        self.assertEqual(stat.S_IMODE(os.stat(self.dest / "a.txt").st_mode) & 0o110, 0)
        self.assertTrue(os.stat(self.dest / "sub" / "b.sh").st_mode & stat.S_IXUSR)
        self.assertFalse(
            (self.dest / "link").exists() or (self.dest / "link").is_symlink()
        )
        self.assertFalse((self.dest / "dirlink").exists())
        self.assertEqual(
            sorted(refused),
            ["dirlink: a link", "link: a link", "sub/fifo: not a regular file"],
        )
        self.assertEqual(self.secret.read_text(), "the monitor's\n")

    def test_the_copies_have_explicit_permissions(self):
        (self.out / "sub" / "r.txt").write_text("r")
        os.chmod(self.out / "sub" / "r.txt", 0o600)
        (self.out / "x.sh").write_text("x")
        os.chmod(self.out / "x.sh", 0o700)
        old = os.umask(0o077)
        try:
            self.assertEqual(pr_results.copy(str(self.out), "", str(self.dest)), [])
        finally:
            os.umask(old)

        def mode(p):
            return stat.S_IMODE(os.stat(p).st_mode)

        self.assertEqual(mode(self.dest / "sub" / "r.txt"), 0o660)
        self.assertEqual(mode(self.dest / "x.sh"), 0o770)
        self.assertEqual(mode(self.dest), 0o2770)
        self.assertEqual(mode(self.dest / "sub"), 0o2770)

    def test_a_subfolder_and_a_single_file(self):
        (self.out / "sub" / "r.txt").write_text("r")
        self.assertEqual(
            pr_results.copy(str(self.out), "sub", str(self.dest / "s")), []
        )
        self.assertEqual((self.dest / "s" / "r.txt").read_text(), "r")
        self.assertEqual(
            pr_results.copy(str(self.out), "sub/r.txt", str(self.dest / "f")), []
        )
        self.assertEqual((self.dest / "f" / "r.txt").read_text(), "r")

    def test_a_link_at_the_destination_is_not_written_through(self):
        (self.out / "r.txt").write_text("from the job")
        self.dest.mkdir()
        (self.dest / "r.txt").symlink_to(self.secret)
        refused = pr_results.copy(str(self.out), "", str(self.dest))
        self.assertEqual(len(refused), 1)
        self.assertIn("r.txt", refused[0])
        self.assertEqual(self.secret.read_text(), "the monitor's\n")

    def test_a_name_the_folder_holds_is_refused_never_replaced(self):
        (self.out / "status").write_text("gates pass\n")
        (self.out / "r.txt").write_text("r")
        self.dest.mkdir()
        (self.dest / "status").write_text("the monitor's\n")
        refused = pr_results.copy(str(self.out), "", str(self.dest))
        self.assertEqual((self.dest / "status").read_text(), "the monitor's\n")
        self.assertEqual((self.dest / "r.txt").read_text(), "r")
        self.assertEqual(len(refused), 1)
        self.assertTrue(refused[0].startswith("status: "), refused)

    def test_the_copy_stops_at_its_total_cap(self):
        for name in ("a", "b", "c"):
            (self.out / name).write_bytes(b"x" * 40)
        refused = pr_results.copy(str(self.out), "", str(self.dest), total=100)
        self.assertEqual(sorted(os.listdir(self.dest)), ["a", "b", "sub"])
        self.assertEqual(len(refused), 1)
        self.assertIn("c: past the copy's", refused[0])

    def test_the_copy_stops_at_its_cap_on_names(self):
        """Each name counts, a folder's too; the first one past the cap is named, and the copy ends there."""
        for name in ("a", "b"):
            (self.out / name).write_text(name)
        for name in ("c", "d"):
            (self.out / "sub" / name).write_text(name)
        refused = pr_results.copy(str(self.out), "", str(self.dest), entries=4)
        self.assertEqual(
            refused,
            [
                "sub/d: past the copy's 4 entries: it and all that follows are not copied"
            ],
        )
        got = sorted(p.relative_to(self.dest).as_posix() for p in self.dest.rglob("*"))
        self.assertEqual(got, ["a", "b", "sub", "sub/c"])

    def test_a_copy_takes_no_more_names_than_entries_max(self):
        many = self.out / "many"
        many.mkdir()
        for i in range(pr_results.ENTRIES_MAX + 2):
            (many / f"{i:06}").touch()
        refused = pr_results.copy(str(self.out), "many", str(self.dest))
        self.assertEqual(len(refused), 1, refused[:3])
        self.assertIn(f"past the copy's {pr_results.ENTRIES_MAX} entries", refused[0])
        self.assertEqual(len(os.listdir(self.dest)), pr_results.ENTRIES_MAX)

    def test_a_copy_opens_no_folder_past_the_depth_cap(self):
        """A tree deeper than Python's recursion limit: the copy, and the command line, go DEPTH_MAX folders down,
        refuse the next one with all it holds, and end as a copy that refused something does.
        """
        deep, levels = self.out / "deep", 1200
        deep.mkdir()
        fd = os.open(deep, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for _ in range(levels):
                os.mkdir("d", dir_fd=fd)
                nxt = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                os.close(fd)
                fd = nxt
            os.close(os.open("far.txt", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd))
        finally:
            os.close(fd)
        (deep / "d" / "near.txt").write_text("n")
        cap = pr_results.DEPTH_MAX
        refused = pr_results.copy(str(self.out), "deep", str(self.dest))
        too_deep = "/".join(["deep"] + ["d"] * (cap + 1))
        self.assertEqual(
            refused,
            [
                f"{too_deep}: more than {cap} folders down: it and everything in it are not copied"
            ],
        )
        self.assertEqual((self.dest / "d" / "near.txt").read_text(), "n")
        self.assertTrue((self.dest / "/".join(["d"] * cap)).is_dir())
        self.assertFalse((self.dest / "/".join(["d"] * (cap + 1))).exists())
        err = io.StringIO()
        with redirect_stderr(err):
            rc = pr_results.main(
                ["copy", str(self.out), "deep", str(self.dest / "again")]
            )
        self.assertEqual(rc, 1)
        self.assertIn(
            f"refused {too_deep}: more than {cap} folders down", err.getvalue()
        )

    def test_a_missing_start_is_refused(self):
        with self.assertRaises(pr_results.Refused):
            pr_results.copy(str(self.out), "nope", str(self.dest))


class TestMain(Slot):
    def test_exit_codes(self):
        (self.out / "r.txt").write_text("r")
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(
                pr_results.main(["copy", str(self.out), "", str(self.dest)]), 0
            )
            (self.out / "link").symlink_to(self.secret)
            self.assertEqual(
                pr_results.main(["copy", str(self.out), "", str(self.dest)]), 1
            )
            self.assertEqual(
                pr_results.main(["copy", str(self.slot / "nope"), "", str(self.dest)]),
                2,
            )
            self.assertEqual(pr_results.main(["read", str(self.out), "link"]), 2)
            self.assertEqual(pr_results.main(["what"]), 2)
        self.assertIn("refused link: a link", err.getvalue())

    def test_read_prints_the_file(self):
        (self.out / "r.txt").write_text("result\n")
        out = io.BytesIO()
        self.assertEqual(pr_results.main(["read", str(self.out), "r.txt"], out=out), 0)
        self.assertEqual(out.getvalue(), b"result\n")


if __name__ == "__main__":
    unittest.main()
