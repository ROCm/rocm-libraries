# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package folder's lock (rkmon_lib/package_lock.py): one lock file in the host's records' folder;
shared holds share it, and an exclusive one, a package update's, refuses every other hold and is refused by any, at
once, naming the holders, each as its holder's file names it, since when, the earliest first. A holder that is gone,
however it ended, names nothing and blocks nothing, and the next hold removes its file; a hold that ends removes its
own, wherever its folder moved. A folder with no records' folder, or one that cannot be written, has no lock, and no
hold makes one.
"""

import datetime
import fcntl
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import package_lock, private_files, records  # noqa: E402

# A command in a process of its own: it holds the lock of the package folder argv[2], exclusive when argv[4] says so,
# else shared, naming the command argv[3]; says whether it holds it, then holds it until its stdin closes. With argv[4]
# "ends", it holds it shared and ends at once, letting nothing go itself.
HOLDER = """\
import os, sys
sys.path.insert(0, sys.argv[1])
from rkmon_lib import package_lock
try:
    held = package_lock.hold(sys.argv[2], sys.argv[3], exclusive=sys.argv[4] == "exclusive")
except package_lock.Taken:
    print("taken", flush=True)
    sys.exit(0)
print("held" if held.held else "none", flush=True)
if sys.argv[4] == "ends":
    os._exit(0)
sys.stdin.read()
"""
UPDATE = "rkmon package update"
AT = "2026-01-02T03:04:05Z"
LATER = "2026-01-03T04:05:06Z"


def mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


class Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pkg = pathlib.Path(tmp.name) / "pkg"
        self.folder = self.pkg / records.FOLDER
        self.folder.mkdir(parents=True, mode=0o700)

    def holder(self, command: str, how: str = "shared") -> subprocess.Popen:
        """A command that holds this folder's lock HOW (HOLDER), in a process of its own, once it said so."""
        p = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-I",
                "-c",
                HOLDER,
                str(ROOT),
                str(self.pkg),
                command,
                how,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.ended, p)
        self.assertEqual(p.stdout.readline(), "held\n")
        return p

    def ended(self, p: subprocess.Popen) -> None:
        p.kill()
        p.wait()
        p.stdin.close()
        p.stdout.close()

    def holders_files(self) -> list:
        return sorted(
            n for n in os.listdir(self.folder) if n.startswith(package_lock.HOLDER)
        )

    def named(self, name: str, data, locked: bool = True) -> None:
        """A holder's file NAME holding DATA, locked by this test until it ends when LOCKED, as a holder keeps it."""
        path = self.folder / name
        path.write_text(
            data if isinstance(data, str) else json.dumps(data), encoding="utf-8"
        )
        if locked:
            fd = os.open(path, os.O_RDONLY)
            self.addCleanup(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX)


class TestNames(unittest.TestCase):
    def test_the_lock_is_in_the_records_folder(self):
        self.assertEqual(package_lock.LOCK.rsplit("/", 1)[0], records.FOLDER)
        self.assertEqual(records.FOLDER, private_files.RECORDS_FOLDER)
        self.assertEqual(package_lock.HOLDER, package_lock.LOCK.rsplit("/", 1)[1] + ".")


class TestHold(Case):
    def test_shared_holds_share_it(self):
        self.holder("rkmon start")
        self.holder("rkmon stop --drain")
        with package_lock.hold(self.pkg, "rkmon check") as held:
            self.assertTrue(held.held)
            self.assertEqual(
                sorted(h.command for h in package_lock.holders(self.pkg)),
                ["rkmon check", "rkmon start", "rkmon stop --drain"],
            )
            self.assertEqual(len(self.holders_files()), 3)
        self.assertEqual(len(self.holders_files()), 2)

    def test_an_exclusive_hold_is_refused_by_the_holders_named_the_earliest_first(self):
        self.holder("rkmon start")
        self.holder("rkmon shell")
        before = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        with self.assertRaises(package_lock.Taken) as caught:
            package_lock.hold(self.pkg, UPDATE, exclusive=True)
        taken = caught.exception
        self.assertFalse(taken.update)
        self.assertEqual(str(taken), "other commands have the package folder's lock")
        self.assertEqual(
            sorted(h.command for h in taken.holders), ["rkmon shell", "rkmon start"]
        )
        for h in taken.holders:
            self.assertLessEqual(h.since, before)
            self.assertGreater(h.since, before - datetime.timedelta(minutes=5))
        self.assertEqual(taken.holders, tuple(package_lock.holders(self.pkg)))
        self.assertEqual(len(self.holders_files()), 2)  # the refused hold named nothing

    def test_an_update_refuses_every_other_hold(self):
        self.holder(UPDATE, "exclusive")
        with self.assertRaises(package_lock.Taken) as caught:
            package_lock.hold(self.pkg, "rkmon start")
        self.assertTrue(caught.exception.update)
        self.assertEqual(caught.exception.holders, ())
        self.assertEqual(
            str(caught.exception), "a package update has the package folder's lock"
        )
        with self.assertRaises(package_lock.Taken) as caught:
            package_lock.hold(self.pkg, UPDATE, exclusive=True)
        self.assertFalse(caught.exception.update)
        self.assertEqual([h.command for h in caught.exception.holders], [UPDATE])

    def test_none_waits(self):
        def flock(
            fd, how
        ):  # the lock another holds: a hold that would wait for it fails the test
            self.assertTrue(how & fcntl.LOCK_NB, "a hold waits")
            raise BlockingIOError

        with mock.patch.object(package_lock.fcntl, "flock", side_effect=flock):
            for exclusive in (False, True):
                with self.assertRaises(package_lock.Taken):
                    package_lock.hold(self.pkg, UPDATE, exclusive=exclusive)

    def test_a_hold_that_ends_lets_go_and_removes_its_file(self):
        held = package_lock.hold(self.pkg, "rkmon start")
        self.assertEqual(len(self.holders_files()), 1)
        held.release()
        self.assertFalse(held.held)
        self.assertEqual(self.holders_files(), [])
        held.release()  # again: nothing to let go
        with package_lock.hold(self.pkg, UPDATE, exclusive=True) as update:
            self.assertTrue(update.held)
            self.assertEqual(
                [h.command for h in package_lock.holders(self.pkg)], [UPDATE]
            )

    def test_a_hold_s_file_goes_from_its_folder_wherever_that_moved(self):
        # as an update's own hold, whose folder its renames move to FOLDER.prev
        held = package_lock.hold(self.pkg, UPDATE, exclusive=True)
        prev = self.pkg.parent / "pkg.prev"
        os.rename(self.pkg, prev)
        (self.pkg / records.FOLDER).mkdir(parents=True, mode=0o700)
        self.assertEqual(
            len(os.listdir(prev / records.FOLDER)), 2
        )  # the lock file and the hold's own
        held.release()
        self.assertEqual(os.listdir(prev / records.FOLDER), ["package.lock"])
        self.assertEqual(os.listdir(self.pkg / records.FOLDER), [])

    def test_the_files_are_600(self):
        with package_lock.hold(self.pkg, "rkmon start"):
            (own,) = self.holders_files()
            self.assertEqual(mode(self.pkg / package_lock.LOCK), 0o600)
            self.assertEqual(mode(self.folder / own), 0o600)
            self.assertEqual((self.pkg / package_lock.LOCK).read_bytes(), b"")
            data = json.loads((self.folder / own).read_text(encoding="utf-8"))
            self.assertEqual(data["command"], "rkmon start")
            self.assertIsNotNone(records.parsed(data["since"]))
            self.assertEqual(sorted(data), ["command", "since"])


class TestGone(Case):
    def test_a_holder_that_is_gone_blocks_nothing_and_its_file_goes(self):
        p = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-I",
                "-c",
                HOLDER,
                str(ROOT),
                str(self.pkg),
                "rkmon start",
                "ends",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        out, _ = p.communicate(timeout=60)
        self.assertEqual((p.returncode, out), (0, "held\n"))
        self.assertEqual(
            len(self.holders_files()), 1
        )  # left by a process that ended without letting go
        self.assertEqual(package_lock.holders(self.pkg), [])
        with package_lock.hold(self.pkg, UPDATE, exclusive=True):
            self.assertEqual(
                [h.command for h in package_lock.holders(self.pkg)], [UPDATE]
            )
            self.assertEqual(len(self.holders_files()), 1)  # the update's own
        self.assertEqual(self.holders_files(), [])

    def test_a_killed_holder_blocks_nothing(self):
        p = self.holder("rkmon apply")
        with self.assertRaises(package_lock.Taken):
            package_lock.hold(self.pkg, UPDATE, exclusive=True)
        p.kill()
        p.wait()
        with package_lock.hold(self.pkg, UPDATE, exclusive=True) as held:
            self.assertTrue(held.held)

    def test_a_file_no_one_locks_names_nothing(self):
        self.named(
            "package.lock.x", {"command": "rkmon start", "since": AT}, locked=False
        )
        self.assertEqual(package_lock.holders(self.pkg), [])
        with package_lock.hold(self.pkg, "rkmon check"):
            self.assertNotIn("package.lock.x", self.holders_files())

    def test_a_locked_file_that_names_no_command_names_nothing(self):
        for i, data in enumerate(
            ("", "[]", "{}", {"command": ""}, {"command": 3}, "\xff")
        ):
            self.named(f"package.lock.{i}", data)
        self.assertEqual(package_lock.holders(self.pkg), [])


class TestHolders(Case):
    def test_the_earliest_first_and_one_with_no_time_last(self):
        self.named("package.lock.a", {"command": "rkmon apply", "since": LATER})
        self.named("package.lock.b", {"command": "rkmon start", "since": AT})
        self.named("package.lock.c", {"command": "rkmon shell", "since": "soon"})
        self.named("package.lock.d", {"command": "rkmon check"})
        self.assertEqual(
            package_lock.holders(self.pkg),
            [
                package_lock.Holder("rkmon start", records.parsed(AT)),
                package_lock.Holder("rkmon apply", records.parsed(LATER)),
                package_lock.Holder("rkmon check", None),
                package_lock.Holder("rkmon shell", None),
            ],
        )

    def test_none_in_a_folder_with_no_records_folder(self):
        self.assertEqual(package_lock.holders(self.pkg.parent / "elsewhere"), [])


class TestNoLock(Case):
    def test_a_folder_with_no_records_folder_has_none_and_none_is_made(self):
        other = self.pkg.parent / "other"
        (other / "docker").mkdir(parents=True)
        with package_lock.hold(other, "rkmon build") as held:
            self.assertFalse(held.held)
            with package_lock.hold(other, UPDATE, exclusive=True) as update:
                self.assertFalse(update.held)
        self.assertEqual(os.listdir(other / "docker"), [])

    @unittest.skipIf(os.geteuid() == 0, "root writes in a read-only folder")
    def test_a_read_only_records_folder_has_none(self):
        os.chmod(self.folder, 0o500)
        self.addCleanup(os.chmod, self.folder, 0o700)
        with package_lock.hold(self.pkg, "rkmon start") as held:
            self.assertFalse(held.held)
        self.assertEqual(os.listdir(self.folder), [])

    @unittest.skipIf(os.geteuid() == 0, "root writes in a read-only folder")
    def test_a_hold_whose_file_cannot_be_made_is_held_unnamed(self):
        package_lock.hold(
            self.pkg, "rkmon start"
        ).release()  # the lock file, as an earlier hold leaves it
        os.chmod(self.folder, 0o500)
        self.addCleanup(os.chmod, self.folder, 0o700)
        with package_lock.hold(self.pkg, "rkmon start") as held:
            self.assertTrue(held.held)
            self.assertEqual(package_lock.holders(self.pkg), [])
            with self.assertRaises(package_lock.Taken) as caught:
                package_lock.hold(self.pkg, UPDATE, exclusive=True)
            self.assertEqual(caught.exception.holders, ())

    def test_a_file_system_with_no_flock_has_none(self):
        with mock.patch.object(
            package_lock.fcntl, "flock", side_effect=OSError(38, "no flock")
        ):
            with package_lock.hold(self.pkg, "rkmon start") as held:
                self.assertFalse(held.held)

    def test_a_name_another_file_has_is_passed_over(self):
        names = iter(("a" * 16, "b" * 16))
        self.named(f"package.lock.{'a' * 16}", {"command": "rkmon apply", "since": AT})
        with mock.patch.object(
            package_lock.secrets, "token_hex", lambda n: next(names)
        ):
            with package_lock.hold(self.pkg, "rkmon start"):
                self.assertEqual(
                    [h.command for h in package_lock.holders(self.pkg)],
                    ["rkmon apply", "rkmon start"],
                )
                self.assertIn(f"package.lock.{'b' * 16}", self.holders_files())


if __name__ == "__main__":
    unittest.main()
