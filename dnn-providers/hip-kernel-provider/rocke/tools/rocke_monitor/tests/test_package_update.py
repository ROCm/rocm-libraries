# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rkmon package update` (rkmon_lib/package_update.py): a package folder in a home folder of
the test's replaced by a newer package, its local files carried along, and the way back that gives the earlier folder
back exactly as it was; of the new package, the files the package build takes, so that what only a folder in use or a
clone holds stays out (the reviewer's two runs), and its private files and its links to files outside it refused;
a tarball the build made, and the tarballs refused; what an earlier update left beside it cleared, or put back; a
refusal or a failure at any step, or an interrupt, changing nothing; no git on the host; the fetch from a repository
of the test's, of the package's own folder alone, and its refusals; the image row of a new package that needs a new
image, and the next line of the state the update leaves, as the package now in the folder reads it, by its version or
its code (the local files carried with it among them), a deploy step's command with its step's words, since the update
asks nothing of the container; with --folder, the update each fix and dry run names carries it, and the folder's own
rkmon names the command after it; what an update that an interrupt stopped changed; and the lock of each folder
the update moves or makes, held as rkmon holds it from before the update moves it until it ends, a hold refused
changing nothing.
"""

import ast
import contextlib
import errno
import io
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import make_package as build  # noqa: E402
from rkmon_lib import (  # noqa: E402
    deploy,
    deploy_plan,
    derived,
    file_types,
    image,
    package_files,
    package_lock,
    package_update,
    paths,
    prcode,
    private_files,
    records,
    settings_file,
    state_facts,
)
from rkmon_lib.checks import Status  # noqa: E402
from fakerepo import git  # noqa: E402
from test_deploy_plan import doc  # noqa: E402

Stopped = package_update.Stopped
# The fix of a --from that gives no package.
FROM_FIX = "rkmon package update --from DIR, DIR being a package folder or a tarball the package build made"
PLACE = "lib/tools/monitor"  # the package's folder in TestFetch's repository: one no code names
# The folder of the tests' package in use, as a command that names it (--folder) shows it; and its own rkmon, which
# names the commands that run there after an update that named it.
NAMED = "--folder ~/rocke_monitor"
OWN = "~/rocke_monitor/docker/rkmon"
# The words before the confirm in state 5's next line after an update, deploy step 3's: the update asks nothing of
# the container, so nothing has passed since a start (states.NOT_STARTED).
CONFIRM_WORDS = "step 3, confirm:"
# The local files of a package folder in use: its own private files (600), a certificate and a note of yours (644).
LOCAL = {
    "docker/.env": ("RKMON_A=1\n", 0o600),
    "docker/compose.override.yaml": ("services: {}\n", 0o600),
    "docker/.env.prev": ("RKMON_A=0\n", 0o600),
    "docker/compose.override.yaml.prev": ("services: {}\n", 0o600),
    "docker/.rkmon/records.json": ("{}\n", 0o600),
    "docker/ca-certificates/site.crt": ("certificate\n", 0o644),
    "docker/notes.txt": ("notes\n", 0o644),
}
LINK = "docker/ca-certificates/site.pem"  # a link of yours, to a certificate outside the package
CARRIED_AS_IS = ("docker/ca-certificates/site.crt", LINK, "docker/notes.txt")
# Code the newer package dropped.
OLDER_ONLY = {"docker/old.sh": "#!/bin/sh\n", "rkmon_lib/gone.py": "\n"}
# Why an update leaves a file of docker/ in FOLDER.prev alone: the new package has another there, or it is a script
# the new package lacks.
TAKEN = "the new package has its own, which takes its place"
NOT_CARRIED = "the new package has none, and an update never carries a script, the image's recipe or the wrapper"


def write(path: pathlib.Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def make_package(top: pathlib.Path, version: str, more=None) -> pathlib.Path:
    """A package of VERSION at TOP, with MORE ({path: text}; None for no file there). The files an image keeps its
    own copies of (code_volume.IMAGE_BOUND) are the same in every version, unless MORE changes them. Its
    .dockerignore is this package's, which says what its code is (image.code_sha256). It has every file of a whole
    code tree (code_volume.WHOLE), by which a fetch finds it (package_files.package_folders).
    """
    files = {
        "rk": ("#!/bin/sh\n", 0o755),
        "rkmon.py": ("# rkmon\n", 0o755),
        "rkmon_lib/__init__.py": ("", 0o644),
        "rkmon_lib/store.py": ("# store\n", 0o644),
        "rkmon_lib/code.py": (f"VERSION = {version!r}\n", 0o644),
        "docker/rkmon": ("#!/bin/sh\n", 0o755),
        "docker/entrypoint.sh": ("#!/bin/sh\n", 0o755),
        "docker/Dockerfile": ("FROM base\n", 0o644),
        "docker/compose.yaml": ("services: {}\n", 0o644),
        "docker/ca-certificates/README.md": ("certificates\n", 0o644),
        "VERSION": (f"{version}\n", 0o644),
        ".gitignore": ((ROOT / ".gitignore").read_text(encoding="utf-8"), 0o644),
        ".dockerignore": ((ROOT / ".dockerignore").read_text(encoding="utf-8"), 0o644),
        "config.json": (package_files.BLANK_CONFIG, 0o644),
    }
    files.update({rel: (text, 0o644) for rel, text in (more or {}).items()})
    for rel, (text, mode) in files.items():
        if text is not None:
            write(top / rel, text, mode)
    return top


def in_use(top: pathlib.Path) -> None:
    """TOP's local files, as a package folder in use has them, and a lock the host's records leave."""
    (top / records.FOLDER).mkdir(mode=0o700)
    for rel, (text, mode) in LOCAL.items():
        write(top / rel, text, mode)
    write(top / records.RECORDS_LOCK, "", 0o600)
    os.symlink("/etc/ssl/certs/site.pem", top / LINK)


def snapshot(top: pathlib.Path) -> dict:
    """{path: what is there} for everything under TOP: a link's target, a file's bytes and permissions, a folder."""
    out = {}
    for here, folders, files in os.walk(top):
        for name in folders + files:
            p = pathlib.Path(here) / name
            rel = p.relative_to(top).as_posix()
            if p.is_symlink():
                out[rel] = ("link", os.readlink(p))
            elif p.is_dir():
                out[rel] = ("folder",)
            else:
                out[rel] = ("file", p.read_bytes(), p.stat().st_mode & 0o777)
    return out


def rows(done_or_caught) -> list:
    return [(r.name, r.status, r.found) for r in done_or_caught.rows]


def left_row(prev: pathlib.Path, rel: str, why: str, dry_run: bool = False) -> tuple:
    """The row of REL, a file of docker/ that an update leaves in PREV alone, for WHY."""
    word = "would be left" if dry_run else "left"
    return (
        rel,
        Status.PASS,
        f"{word} in {prev / rel}, which the next update removes: {why}",
    )


class Case(unittest.TestCase):
    """A home folder with the package folder ~/rocke_monitor in use (1.0) and a newer package, ~/new (2.0)."""

    def setUp(self):
        self.tmp = pathlib.Path(os.path.realpath(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        patcher = mock.patch.dict(os.environ, {"HOME": str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = make_package(self.home / "rocke_monitor", "1.0", OLDER_ONLY)
        in_use(self.folder)
        self.new = make_package(self.home / "new", "2.0")
        self.prev = self.home / "rocke_monitor.prev"
        self.stage, self.fetched, self.unpacked, self.aside, self.old = (
            self.home / f"rocke_monitor{ending}"
            for ending in (
                package_update.STAGE,
                package_update.FETCHED,
                package_update.UNPACKED,
                package_update.ASIDE,
                package_update.OLD,
            )
        )

    def updated(self, before: dict) -> dict:
        """What the package folder holds after an update from ~/new, BEFORE being what it held."""
        kept = {rel: before[rel] for rel in [*LOCAL, LINK]}
        return {**snapshot(self.new), "docker/.rkmon": ("folder",), **kept}

    def assert_stopped(self, caught, word, text, fix=None):
        self.assertEqual(
            (caught.exception.word, caught.exception.text, str(caught.exception.fix)),
            (word, text, str(fix)),
        )

    def assert_home(self, *names):
        self.assertEqual(sorted(os.listdir(self.home)), sorted(names))

    def no_git(self):
        """No git on this host, as shutil.which finds it."""
        real = shutil.which

        def which(name, *args, **kwargs):
            return None if name == "git" else real(name, *args, **kwargs)

        return mock.patch("shutil.which", which)


class TestUpdate(Case):
    def test_there_and_back(self):
        before = snapshot(self.folder)
        done = package_update.update(self.folder, self.new)
        self.assertEqual(
            (done.word, done.text),
            (
                "UPDATED",
                "~/rocke_monitor has the package from ~/new; the one it replaced is ~/rocke_monitor.prev",
            ),
        )
        self.assertEqual(snapshot(self.prev), before)
        after = snapshot(self.folder)
        self.assertEqual(after, self.updated(before))
        self.assertEqual(
            rows(done),
            [
                (n, Status.PASS, f"carried to {self.folder / n} at 600")
                for n in private_files.PACKAGE_FILES
            ]
            + [
                (rel, Status.PASS, f"carried to {self.folder / rel} as it is")
                for rel in CARRIED_AS_IS
            ]
            + [left_row(self.prev, "docker/old.sh", NOT_CARRIED)],
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")
        done = package_update.update(self.folder, self.prev)
        self.assertEqual(
            (done.word, done.text, done.rows),
            (
                "UPDATED",
                "~/rocke_monitor and ~/rocke_monitor.prev changed places, each with its own local files",
                (),
            ),
        )
        self.assertEqual((snapshot(self.folder), snapshot(self.prev)), (before, after))
        package_update.update(self.folder, self.prev)
        self.assertEqual((snapshot(self.folder), snapshot(self.prev)), (after, before))
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_a_link_to_the_folder_goes_on_pointing_at_it(self):
        link = self.home / "rkmon"
        os.symlink(self.folder, link)
        done = package_update.update(link, self.new)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(os.readlink(link), str(self.folder))
        self.assertFalse(package_update.differs(link, self.new))

    def test_a_looser_private_file_is_600_in_both_folders(self):
        (self.folder / "docker/.env").chmod(0o644)
        done = package_update.update(self.folder, self.new)
        self.assertEqual(
            rows(done)[0],
            (
                "docker/.env",
                Status.PASS,
                f"carried to {self.folder}/docker/.env at 600; its permissions were 644, and are 600 in both "
                "folders now",
            ),
        )
        for top in (self.folder, self.prev):
            self.assertEqual((top / "docker/.env").stat().st_mode & 0o777, 0o600)

    def test_what_an_earlier_update_left_goes(self):
        before = snapshot(self.folder)
        make_package(self.prev, "0.9")
        for p in (self.stage, self.fetched, self.unpacked, self.old):
            write(p / "x.txt", "left\n")
        done = package_update.update(self.folder, self.new)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(snapshot(self.prev), before)
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_an_earlier_folder_left_aside_is_the_way_back_again(self):
        before = snapshot(self.folder)
        earlier = snapshot(make_package(self.old, "0.9"))
        done = package_update.update(self.folder, self.prev)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (earlier, before)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_the_new_packages_own_private_files_stay_where_they_are(self):
        for n in private_files.PACKAGE_FILES:
            write(self.new / n, "theirs\n", 0o600)
        new_before, before = snapshot(self.new), snapshot(self.folder)
        package_update.update(self.folder, self.new)
        self.assertEqual(snapshot(self.new), new_before)
        after = snapshot(self.folder)
        self.assertEqual(
            {n: after[n] for n in private_files.PACKAGE_FILES},
            {n: before[n] for n in private_files.PACKAGE_FILES},
        )

    def test_a_refused_carry_changes_nothing(self):
        earlier = snapshot(make_package(self.prev, "0.9"))
        (self.folder / "docker/.env.prev").unlink()
        os.symlink(self.folder / "docker/.env", self.folder / "docker/.env.prev")
        before = snapshot(self.folder)
        with self.assertRaises(private_files.Refused) as caught:
            package_update.update(self.folder, self.new)
        self.assertEqual(
            [(r.name, r.status) for r in caught.exception.rows],
            [("docker/.env.prev", Status.FAIL)],
        )
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_the_same_code_is_nothing_to_do(self):
        same = make_package(self.home / "same", "1.0", OLDER_ONLY)
        write(same / "rkmon_lib/__pycache__/code.cpython-39.pyc", "cache\n")
        write(same / ".pytest_cache/v/cache", "cache\n")
        before = snapshot(self.folder)
        done = package_update.update(self.folder, same)
        self.assertEqual(
            (done.word, done.text, done.rows),
            (
                "NOTHING TO DO",
                "~/rocke_monitor has the package from ~/same already",
                (),
            ),
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("new", "rocke_monitor", "same")

    def test_differs(self):
        cases = (
            ("the same code", lambda p: None, False),
            (
                "other bytes",
                lambda p: write(p / "rkmon_lib/code.py", "VERSION = 1\n"),
                True,
            ),
            (
                "the owner's execute bit",
                lambda p: (p / "docker/entrypoint.sh").chmod(0o644),
                True,
            ),
            (
                "others' permissions alone",
                lambda p: (p / "VERSION").chmod(0o664),
                False,
            ),
            # any execute bit makes a file one the package runs (code_volume.runs), as the image's build reads it
            (
                "a group's execute bit alone",
                lambda p: (p / "VERSION").chmod(0o654),
                True,
            ),
            ("a file added", lambda p: write(p / "rkmon_lib/added.py", "\n"), True),
            ("a file removed", lambda p: (p / "rkmon_lib/gone.py").unlink(), True),
            (
                "a script of docker/ dropped",
                lambda p: (p / "docker/old.sh").unlink(),
                True,
            ),
            (
                "a local file of docker/ added",
                lambda p: write(p / "docker/mine.txt", "\n"),
                True,
            ),
            (
                "a link added",
                lambda p: os.symlink("code.py", p / "rkmon_lib/link"),
                True,
            ),
            (
                "a file the package's ignore rules leave out",
                lambda p: write(p / "docker/.env.old", "\n"),
                False,
            ),
            (
                "another config.json",
                lambda p: write(p / "config.json", "{}\n"),
                False,
            ),
        )
        for i, (name, change, expected) in enumerate(cases):
            with self.subTest(name):
                new = make_package(self.tmp / f"case{i}", "1.0", OLDER_ONLY)
                change(new)
                self.assertIs(package_update.differs(self.folder, new), expected)
        for target, expected in (("code.py", False), ("VERSION", True)):
            with self.subTest("a link to", target=target):
                a = make_package(self.tmp / "a", "1.0")
                b = make_package(self.tmp / f"b-{target}", "1.0")
                os.symlink("code.py", a / "rkmon_lib/link")
                os.symlink(target, b / "rkmon_lib/link")
                self.assertIs(package_update.differs(a, b), expected)
                shutil.rmtree(a)
        with self.subTest("a link, by its file"):
            a = make_package(self.tmp / "a", "1.0")
            b = make_package(self.tmp / "b-file", "1.0")
            os.symlink("code.py", a / "rkmon_lib/link")
            shutil.copy2(b / "rkmon_lib/code.py", b / "rkmon_lib/link")
            self.assertIs(package_update.differs(a, b), False)

    def test_local(self):
        for rel, expected in (
            ("docker/.env.mine", True),
            ("docker/ca-certificates/site.crt", True),
            ("docker/compose.local.yaml", True),
            ("docker/install.sh", False),
            ("docker/rk_tool.py", False),
            ("docker/Dockerfile", False),
            ("docker/rkmon", False),
            ("rkmon_lib/mine.txt", False),
            ("notes.txt", False),
        ):
            with self.subTest(rel):
                self.assertIs(package_update.local(rel), expected)

    def test_a_local_file_whose_place_the_new_package_takes_stays_behind(self):
        """In FOLDER.prev alone, which the next update removes, and so does a script of docker/ that the new package
        lacks: each has a row that says where it is. A local file the same as the new package's has none.
        """
        write(self.folder / "docker/extra", "mine\n")
        write(self.new / "docker/extra/README.md", "theirs\n")
        write(self.folder / "docker/changed.txt", "mine\n")
        write(self.new / "docker/changed.txt", "theirs\n")
        write(self.folder / "docker/hook.sh", "#!/bin/sh\n")
        for top in (self.folder, self.new):
            write(top / "docker/same.txt", "the same\n")
        self.assertEqual(
            package_update.others(self.folder, self.new), list(CARRIED_AS_IS)
        )
        left = ["docker/changed.txt", "docker/extra", "docker/hook.sh", "docker/old.sh"]
        self.assertEqual(package_update.left(self.folder, self.new), left)
        reasons = [(rel, NOT_CARRIED if rel.endswith(".sh") else TAKEN) for rel in left]
        dry = package_update.update(self.folder, self.new, dry_run=True)
        self.assertEqual(
            [r for r in rows(dry) if r[0] in left],
            [left_row(self.prev, rel, why, dry_run=True) for rel, why in reasons],
        )
        done = package_update.update(self.folder, self.new)
        self.assertEqual(
            [r for r in rows(done) if r[0] in left],
            [left_row(self.prev, rel, why) for rel, why in reasons],
        )
        self.assertEqual(
            snapshot(self.folder)["docker/extra/README.md"][1], b"theirs\n"
        )
        for rel in left:
            self.assertTrue((self.prev / rel).is_file(), rel)
        self.assertEqual((self.prev / "docker/extra").read_text(), "mine\n")

    def test_what_no_package_includes_is_neither_carried_nor_left_with_a_row(self):
        """What no package includes (package_files.excluded) in FOLDER's docker/, a backup a write keeps, a compiled
        file or a cache, is no local file to carry and no file to give a row: the update reads package_files' rule.
        """
        for rel in (
            "docker/rkmon.bak-20261123120000",
            "docker/tool.pyc",
            "docker/__pycache__/tool.cpython-314.pyc",
        ):
            write(self.folder / rel, "left by a copy\n")
        self.assertEqual(
            package_update.others(self.folder, self.new), list(CARRIED_AS_IS)
        )
        self.assertEqual(package_update.left(self.folder, self.new), ["docker/old.sh"])

    def test_no_package(self):
        empty = self.home / "empty"
        empty.mkdir()
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, empty)
        self.assert_stopped(
            caught,
            "REFUSED",
            "~/empty has no package: neither it nor its one folder has rkmon.py and rkmon_lib/; nothing changed",
            FROM_FIX,
        )
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.home / "nothing")
        self.assert_stopped(
            caught,
            "REFUSED",
            "there is no folder or file at ~/nothing; nothing changed",
            FROM_FIX,
        )
        with self.assertRaises(Stopped) as caught:
            package_update.update(empty, self.new)
        self.assert_stopped(
            caught,
            "REFUSED",
            "~/empty has no package: no rkmon.py and rkmon_lib/; nothing changed",
            "rkmon package update --from DIR --folder FOLDER, FOLDER being the package folder to update",
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("empty", "new", "rocke_monitor")

    def test_a_named_folder_s_fixes_name_it(self):
        """With --folder (NAMED): each command a fix names updates that folder too."""
        empty = self.home / "empty"
        empty.mkdir()
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, empty, named=True)
        self.assertEqual(
            str(caught.exception.fix),
            f"rkmon package update --from DIR {NAMED}, DIR being a package folder or a tarball the package build made",
        )
        with self.no_git(), self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new, named=True)
        self.assertEqual(
            str(caught.exception.fix),
            f"install git, then run rkmon package update --from ~/new {NAMED}",
        )

    def test_a_package_one_folder_down(self):
        unpacked = self.home / "unpacked"
        shutil.copytree(self.new, unpacked / "rocke_monitor", symlinks=True)
        write(unpacked / "README.txt", "beside it\n")
        self.assertEqual(package_files.package_in(unpacked), unpacked / "rocke_monitor")
        done = package_update.update(self.folder, unpacked)
        self.assertEqual(
            done.text,
            "~/rocke_monitor has the package from ~/unpacked/rocke_monitor; the one it replaced is "
            "~/rocke_monitor.prev",
        )
        shutil.copytree(self.new, unpacked / "second", symlinks=True)
        self.assertIsNone(package_files.package_in(unpacked))  # two: which one?
        self.assertIsNone(package_files.package_in(self.home / "nothing"))

    def test_dry_runs_change_nothing(self):
        before = snapshot(self.folder)
        with mock.patch.object(shutil, "copy2", wraps=shutil.copy2) as copy2:
            done = package_update.update(self.folder, self.new, dry_run=True)
        # the new package is staged; of the folder's files, its other local files alone are copied, into the stage,
        # which goes, so that the next line reads the code the update would put in the folder
        self.assertTrue(copy2.call_args_list)
        self.assertEqual(
            [
                (
                    pathlib.Path(c.args[0]).relative_to(self.folder).as_posix(),
                    pathlib.Path(c.args[1]).relative_to(self.stage).as_posix(),
                )
                for c in copy2.call_args_list
                if pathlib.Path(c.args[0]).is_relative_to(self.folder)
            ],
            [(rel, rel) for rel in CARRIED_AS_IS],
        )
        self.assertEqual(
            (done.word, done.text),
            ("PASSED", "the package from ~/new can replace ~/rocke_monitor"),
        )
        # the rows the update would give: carry's checks ran, and nothing was set or carried
        self.assertEqual(
            rows(done),
            [
                (n, Status.PASS, f"would be carried to {self.folder / n} at 600")
                for n in private_files.PACKAGE_FILES
            ]
            + [
                (rel, Status.PASS, f"would be carried to {self.folder / rel} as it is")
                for rel in CARRIED_AS_IS
            ]
            + [left_row(self.prev, "docker/old.sh", NOT_CARRIED, dry_run=True)],
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("new", "rocke_monitor")
        earlier = snapshot(make_package(self.prev, "0.9"))
        done = package_update.update(self.folder, self.prev, dry_run=True)
        self.assertEqual(
            (done.word, done.text),
            (
                "PASSED",
                "~/rocke_monitor and ~/rocke_monitor.prev can change places",
            ),
        )
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )

    def test_no_way_back(self):
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.prev)
        self.assert_stopped(
            caught,
            "REFUSED",
            "~/rocke_monitor.prev has no package, so there is no way back; nothing changed",
            "rkmon package update --from DIR, DIR being an earlier package",
        )

    def renaming(self, fails):
        """os.rename, but for the renames FAILS ((from, to) -> what it raises: an exception, or AFTER for an interrupt
        once the rename is done)."""
        real = os.rename

        def rename(src, dst):
            what = fails.get((pathlib.Path(src), pathlib.Path(dst)))
            if what is None:
                return real(src, dst)
            if what == "after":
                real(src, dst)
                raise KeyboardInterrupt
            raise what

        return mock.patch("os.rename", rename)

    def denied(self, path):
        return PermissionError(errno.EACCES, "Permission denied", str(path))

    def test_a_failed_rename_changes_nothing(self):
        earlier = snapshot(make_package(self.prev, "0.9"))
        before = snapshot(self.folder)
        with self.renaming({(self.stage, self.folder): self.denied(self.stage)}):
            with self.assertRaises(Stopped) as caught:
                package_update.update(self.folder, self.new)
        self.assert_stopped(
            caught,
            "ERROR",
            "the update of ~/rocke_monitor stopped: ~/rocke_monitor.update-new: Permission denied; nothing "
            "changed",
        )
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")
        with self.renaming({(self.prev, self.folder): self.denied(self.prev)}):
            with self.assertRaises(Stopped) as caught:
                package_update.update(self.folder, self.prev)
        self.assert_stopped(
            caught,
            "ERROR",
            "~/rocke_monitor and ~/rocke_monitor.prev could not change places: ~/rocke_monitor.prev: "
            "Permission denied; nothing changed",
        )
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_what_an_earlier_update_left_and_cannot_be_cleared_away(self):
        if os.geteuid() == 0:
            self.skipTest("root removes what a folder's permissions keep")
        write(self.stage / "x.txt", "left\n")
        self.stage.chmod(0o500)
        self.addCleanup(self.stage.chmod, 0o700)
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new)
        self.assertEqual(caught.exception.word, "ERROR")
        self.assertRegex(  # the file named is the folder's, or its file's, as Python's version says
            caught.exception.text,
            r"^what an earlier update left beside ~/rocke_monitor could not be cleared away: "
            r"~/rocke_monitor\.update-new(/x\.txt)?: Permission denied; nothing changed$",
        )
        self.assertEqual(snapshot(self.folder), before)

    def test_an_interrupt_changes_nothing(self):
        earlier = snapshot(make_package(self.prev, "0.9"))
        before = snapshot(self.folder)
        for name, fails in (
            ("before a rename", {(self.stage, self.folder): KeyboardInterrupt()}),
            ("after a rename", {(self.stage, self.folder): "after"}),
            ("after the first rename", {(self.prev, self.old): "after"}),
        ):
            with self.subTest(name):
                with self.renaming(fails), self.assertRaises(KeyboardInterrupt):
                    package_update.update(self.folder, self.new)
                self.assertEqual(
                    (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
                )
                self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")
        with self.subTest("while the new package is copied"):
            with mock.patch("shutil.copy2", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    package_update.update(self.folder, self.new)
            self.assertEqual(
                (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
            )
            self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")
        with self.subTest("while a local file is copied"):
            real_copy2 = shutil.copy2

            def copy2(src, dst, **kwargs):
                if pathlib.Path(src) == self.folder / "docker/notes.txt":
                    raise KeyboardInterrupt
                return real_copy2(src, dst, **kwargs)

            with mock.patch("shutil.copy2", copy2):
                with self.assertRaises(KeyboardInterrupt):
                    package_update.update(self.folder, self.new)
            self.assertEqual(
                (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
            )
            self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")
        with self.subTest("in the way back"):
            with self.renaming({(self.aside, self.prev): "after"}):
                with self.assertRaises(KeyboardInterrupt):
                    package_update.update(self.folder, self.prev)
            self.assertEqual(
                (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
            )
            self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_what_an_interrupt_changed(self):
        """interrupted, FOLDER's os.stat before the update being BEFORE: nothing changed while FOLDER is the folder it
        was, or none was there; the update is done once FOLDER is another (its renames done); FOLDER gone, the update
        stopped part-way."""
        before = os.stat(self.folder)
        nothing = ("nothing changed", False)
        self.assertEqual(package_update.interrupted(self.folder, before), nothing)
        self.assertEqual(package_update.interrupted(self.home / "none", None), nothing)
        package_update.update(self.folder, self.new)
        self.assertEqual(
            package_update.interrupted(self.folder, before),
            (
                "the update is done; only its clean-up stopped, which the next update finishes",
                True,
            ),
        )
        os.rename(self.folder, self.home / "elsewhere")
        self.assertEqual(
            package_update.interrupted(self.folder, before),
            ("the update stopped part-way: ~/rocke_monitor is not there", False),
        )

    def test_a_rename_back_that_fails_names_its_fix(self):
        earlier = snapshot(make_package(self.prev, "0.9"))
        before = snapshot(self.folder)
        fails = {
            (self.stage, self.folder): self.denied(self.stage),
            (self.prev, self.folder): self.denied(self.prev),
        }
        with self.renaming(fails), self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new)
        self.assert_stopped(
            caught,
            "ERROR",
            "~/rocke_monitor.prev could not be renamed back to ~/rocke_monitor: Permission denied",
            "mv ~/rocke_monitor.prev ~/rocke_monitor",
        )
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new)
        self.assert_stopped(
            caught,
            "REFUSED",
            "~/rocke_monitor is gone, and ~/rocke_monitor.prev has a package; nothing changed",
            "mv ~/rocke_monitor.prev ~/rocke_monitor puts it in its place",
        )
        os.rename(self.prev, self.folder)
        done = package_update.update(self.folder, self.prev, dry_run=True)
        self.assertEqual(done.word, "PASSED")
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_a_way_back_that_stopped_part_way(self):
        make_package(self.prev, "0.9")
        for name, gone, fix in (
            (
                "after its second rename",
                self.prev,
                "~/rocke_monitor.prev puts it back",
            ),
            ("after its first rename", self.folder, "~/rocke_monitor puts it back"),
            (
                "with both folders there",
                None,
                "~/rocke_monitor.update-aside.old moves it out of the way: look at what is in it before you "
                "remove it",
            ),
        ):
            with self.subTest(name):
                if gone is None:
                    make_package(self.aside, "0.8")
                else:
                    os.rename(gone, self.aside)
                with self.assertRaises(Stopped) as caught:
                    package_update.update(self.folder, self.new)
                self.assert_stopped(
                    caught,
                    "REFUSED",
                    "~/rocke_monitor.update-aside is left from a way back that stopped part-way; nothing changed",
                    f"mv ~/rocke_monitor.update-aside {fix}",
                )
                if gone is None:
                    shutil.rmtree(self.aside)
                else:
                    os.rename(self.aside, gone)

    def test_no_git(self):
        with self.no_git(), self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new)
        self.assert_stopped(
            caught,
            "REFUSED",
            "git, which reads which files of ~/new a package includes, is not on this host; nothing changed",
            "install git, then run rkmon package update --from ~/new",
        )
        self.assertTrue(caught.exception.earlier_step)
        self.assert_home("new", "rocke_monitor")

    def test_a_cloud_synced_folder_is_refused_by_its_own_name(self):
        synced = self.home / "Dropbox"
        folder = synced / "rocke_monitor"
        shutil.copytree(self.folder, folder, symlinks=True)
        before = snapshot(folder)
        with self.assertRaises(private_files.Refused) as caught:
            package_update.update(folder, self.new)
        self.assertEqual(
            [(r.name, r.status) for r in caught.exception.rows],
            [("package folder", Status.FAIL)],
        )
        self.assertIn(str(folder), caught.exception.rows[0].found)
        self.assertNotIn(package_update.STAGE, caught.exception.rows[0].found)
        self.assertEqual(snapshot(folder), before)
        self.assertEqual(os.listdir(synced), ["rocke_monitor"])


class TestTheLocks(Case):
    """update's HOLD, as rkmon's: the lock of each folder beside the package folder that the update moves or
    makes, held exclusive (rkmon_lib/package_lock.py) before the update moves it, until the command ends, which the
    test's stack stands for; the package folder's own lock its caller holds. The earlier FOLDER.prev is in use.
    """

    UPDATE = "rkmon package update --from ~/new"

    def setUp(self):
        super().setUp()
        in_use(make_package(self.prev, "0.9"))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # the holds, the renames beside the package folder and the next line, in order
        self.seen = []

    def hold(self, folder):
        """HOLD as rkmon's: FOLDER's lock, held exclusive in the test's stack; kept in `seen` with whether FOLDER has
        a records' folder, and so a lock, as it is held."""
        self.seen.append(("hold", folder.name, os.path.isdir(folder / records.FOLDER)))
        self.stack.enter_context(package_lock.hold(folder, self.UPDATE, exclusive=True))

    def starts(self, folder) -> str:
        """What a command that starts now in FOLDER finds, holding its lock shared as each one does (rkmon's
        hold_lock)."""
        try:
            package_lock.hold(folder, "rkmon start").release()
        except package_lock.Taken as exc:
            return "an update runs" if exc.update else "taken"
        return "runs"

    @contextlib.contextmanager
    def watched(self):
        """The renames beside the package folder, and the start of the next line, the update's last step, after its
        clean-up, kept in `seen` with what a command that starts then in the package folder finds.
        """
        rename, next_line = os.rename, package_update.next_line

        def renamed(src, dst, *args, **kwargs):
            if pathlib.Path(src).parent == self.home:
                self.seen.append(
                    ("rename", pathlib.Path(src).name, pathlib.Path(dst).name)
                )
            return rename(src, dst, *args, **kwargs)

        def next_of(folder, named=False):
            self.seen.append(("next line", self.starts(folder)))
            return next_line(folder, named)

        with mock.patch("os.rename", renamed):
            with mock.patch.object(package_update, "next_line", next_of):
                yield

    def test_each_folder_is_held_before_the_update_moves_it_until_it_ends(self):
        """FOLDER.prev's lock, then STAGE's as it is made, its records' folder made for it, both before the renames:
        STAGE takes the package folder's place with its lock, so that through the clean-up and the next line a
        command that starts there is refused, as while any update runs; until the update ends.
        """
        with self.watched():
            done = package_update.update(self.folder, self.new, hold=self.hold)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(
            self.seen,
            [
                ("hold", "rocke_monitor.prev", True),
                ("hold", "rocke_monitor.update-new", True),
                ("rename", "rocke_monitor.prev", "rocke_monitor.update-old"),
                ("rename", "rocke_monitor", "rocke_monitor.prev"),
                ("rename", "rocke_monitor.update-new", "rocke_monitor"),
                ("next line", "an update runs"),
            ],
        )
        self.assertEqual(
            stat.S_IMODE((self.folder / records.FOLDER).stat().st_mode), 0o700
        )
        self.assertEqual(
            package_lock.holders(self.folder),
            [package_lock.Holder(self.UPDATE, mock.ANY)],
        )
        self.assertEqual(self.starts(self.folder), "an update runs")
        self.stack.close()
        self.assertEqual(
            (self.starts(self.folder), package_lock.holders(self.folder)), ("runs", [])
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_the_way_back_holds_the_earlier_folder_before_it_moves_it(self):
        """The way back holds FOLDER.prev's lock before the folders change places: that folder takes the package
        folder's place with its lock, so that a command that starts there before the update ends is refused.
        """
        with self.watched():
            done = package_update.update(self.folder, self.prev, hold=self.hold)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(
            self.seen,
            [
                ("hold", "rocke_monitor.prev", True),
                ("rename", "rocke_monitor", "rocke_monitor.update-aside"),
                ("rename", "rocke_monitor.prev", "rocke_monitor"),
                ("rename", "rocke_monitor.update-aside", "rocke_monitor.prev"),
                ("next line", "an update runs"),
            ],
        )
        self.stack.close()
        self.assertEqual(self.starts(self.folder), "runs")

    def test_a_hold_refused_changes_nothing(self):
        """While a command runs in FOLDER.prev, holding its lock, the update, the way back and their dry runs are
        refused as HOLD refuses, naming it; so is an update whose hold of STAGE is refused, and STAGE goes. Nothing
        changed."""
        shell = "~/rocke_monitor.prev/docker/rkmon shell"
        before = snapshot(self.folder)
        with package_lock.hold(self.prev, shell):
            earlier = snapshot(self.prev)
            for name, source, dry_run in (
                ("the update", self.new, False),
                ("its dry run", self.new, True),
                ("the way back", self.prev, False),
                ("the way back's dry run", self.prev, True),
            ):
                with self.subTest(name):
                    with self.assertRaises(package_lock.Taken) as caught:
                        package_update.update(
                            self.folder, source, dry_run=dry_run, hold=self.hold
                        )
                    self.assertEqual(
                        [h.command for h in caught.exception.holders], [shell]
                    )
                    self.assertEqual(
                        (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
                    )
                    self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

        def refused(folder):
            if folder == self.stage:
                raise package_lock.Taken(False)

        earlier = snapshot(self.prev)
        with self.assertRaises(package_lock.Taken):
            package_update.update(self.folder, self.new, hold=refused)
        self.assertEqual(
            (snapshot(self.folder), snapshot(self.prev)), (before, earlier)
        )
        self.assert_home("new", "rocke_monitor", "rocke_monitor.prev")

    def test_the_stage_has_a_lock_where_the_folder_has_one(self):
        """A records' folder with no records to carry: STAGE's is made for its lock all the same. A package folder
        with none has no lock, and STAGE none either: its hold holds nothing."""
        (self.folder / records.FILE).unlink()
        package_update.update(self.folder, self.new, hold=self.hold)
        self.assertEqual(
            package_lock.holders(self.folder),
            [package_lock.Holder(self.UPDATE, mock.ANY)],
        )
        self.stack.close()
        shutil.rmtree(self.folder / records.FOLDER)
        self.seen.clear()
        package_update.update(
            self.folder, make_package(self.home / "newer", "3.0"), hold=self.hold
        )
        self.assertEqual(
            self.seen,
            [
                ("hold", "rocke_monitor.prev", True),
                ("hold", "rocke_monitor.update-new", False),
            ],
        )
        self.assertFalse((self.folder / records.FOLDER).exists())


class TestWhatComesIn(Case):
    """Of the new package, the files the package build takes (package_files): what only a folder in use or a clone
    holds stays where it is, the reviewer's two runs; the private files and the links to files outside it that it
    holds are refused, nothing changed."""

    def test_another_folder_in_use_as_the_source(self):
        # its private files, a .env it moved aside, its certificate and its records stay with it; this folder's own
        # come along, each with its row
        other = make_package(self.home / "other", "2.0")
        in_use(other)
        (other / "docker/notes.txt").unlink()
        write(other / "docker/.env.old", "RKMON_A=old\n", 0o600)
        write(other / "docker/ca-certificates/site.crt", "theirs\n")
        before, theirs = snapshot(self.folder), snapshot(other)
        done = package_update.update(self.folder, other)
        self.assertEqual(snapshot(self.folder), self.updated(before))
        self.assertEqual(
            rows(done),
            [
                (n, Status.PASS, f"carried to {self.folder / n} at 600")
                for n in private_files.PACKAGE_FILES
            ]
            + [
                (rel, Status.PASS, f"carried to {self.folder / rel} as it is")
                for rel in CARRIED_AS_IS
            ]
            + [left_row(self.prev, "docker/old.sh", NOT_CARRIED)],
        )
        self.assertEqual(snapshot(other), theirs)

    def test_a_clone_as_the_source(self):
        # its config.json, a monitor's state and a venv (with no .gitignore of its own) stay out; the config.json
        # that comes is the blank one
        clone = make_package(self.home / "clone", "2.0")
        git(clone, "init", "-q", "-b", "main")
        git(clone, "add", "-A")
        git(clone, "commit", "-qm", "the package")
        write(clone / "config.json", '{"repo": "<org>/<repo>"}\n', 0o600)
        write(clone / "state/state.json", "{}\n")
        write(clone / ".venv/pyvenv.cfg", "home = /usr/bin\n")
        write(clone / ".venv/bin/python", "invented\n")
        before = snapshot(self.folder)
        # under a umask that keeps each new file its owner's alone: the blank config.json is 644 all the same
        umask = os.umask(0o077)
        try:
            done = package_update.update(self.folder, clone)
        finally:
            os.umask(umask)
        self.assertEqual(done.word, "UPDATED")
        self.assertEqual(snapshot(self.folder), self.updated(before))

    def test_a_read_only_source(self):
        # as the image's copy of the package is: each file comes its owner's to write, as the build copies it
        (self.new / "rkmon_lib/code.py").chmod(0o444)
        self.assertEqual(package_update.update(self.folder, self.new).word, "UPDATED")
        self.assertEqual(
            (self.folder / "rkmon_lib/code.py").stat().st_mode & 0o777, 0o644
        )

    def test_private_files_in_the_new_package(self):
        # a settings file, by its header whatever its name, and a file git tracks though its name is a secret's; a
        # dry run finds them too
        write(self.new / "docker/deploy.yaml", settings_file.dump(doc({})), 0o600)
        write(self.new / "docker/.env", "RKMON_A=2\n", 0o600)
        git(self.new, "init", "-q", "-b", "main")
        git(self.new, "add", "-A")
        git(self.new, "add", "-f", "docker/.env")
        git(self.new, "commit", "-qm", "the package")
        before = snapshot(self.folder)
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                with self.assertRaises(Stopped) as caught:
                    package_update.update(self.folder, self.new, dry_run=dry_run)
                self.assert_stopped(
                    caught,
                    "REFUSED",
                    "the package from ~/new has private files, which no package includes; nothing changed (fix each "
                    "FAIL above)",
                )
                # exit 4, as content no package includes, not 3 as an earlier step
                self.assertFalse(caught.exception.earlier_step)
                self.assertEqual(
                    [
                        (*row, str(r.fix))
                        for row, r in zip(rows(caught.exception), caught.exception.rows)
                    ],
                    [
                        (
                            "the new package",
                            Status.FAIL,
                            "private files inside it, which no package carries: docker/.env, docker/deploy.yaml",
                            "take each out of ~/new, then run rkmon package update --from ~/new",
                        )
                    ],
                )
                self.assertEqual(snapshot(self.folder), before)
                self.assert_home("new", "rocke_monitor")

    def test_links_in_the_new_package(self):
        # a link to a file outside it is refused before anything is copied; a link to a file in it is that file, as
        # the build copies it
        os.symlink("code.py", self.new / "rkmon_lib/alias.py")
        key = self.home / ".ssh/id_test"
        write(key, "a key\n", 0o600)
        os.symlink(key, self.new / "docker/key")
        # config.json, which the copy writes blank, may be a link to a file anywhere
        write(self.home / "elsewhere.json", "{}\n", 0o600)
        (self.new / "config.json").unlink()
        os.symlink(self.home / "elsewhere.json", self.new / "config.json")
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.new)
        self.assert_stopped(
            caught,
            "REFUSED",
            "the package from ~/new has links to files outside it, which the copy would bring in: docker/key; "
            "nothing changed",
            "take each out of ~/new, or put its file in its place, then run rkmon package update --from ~/new",
        )
        self.assertFalse(caught.exception.earlier_step)
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home(".ssh", "elsewhere.json", "new", "rocke_monitor")
        (self.new / "docker/key").unlink()
        self.assertEqual(package_update.update(self.folder, self.new).word, "UPDATED")
        self.assertEqual(
            (self.folder / "config.json").read_text(encoding="utf-8"),
            package_files.BLANK_CONFIG,
        )
        self.assertEqual(
            snapshot(self.folder)["rkmon_lib/alias.py"],
            snapshot(self.new)["rkmon_lib/code.py"],
        )


def constants(module, names) -> list:
    """The lines of MODULE's source that write one of NAMES as a string of its own, not a part of an f-string (a
    backup's name that a write makes, say)."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    parts = {
        id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values
    }
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value in names and id(n) not in parts
    ]


# The names of what a developer's copy of the package makes or leaves (paths' lists), and their parts.
LEFT_BY_A_COPY = {
    *paths.COPY_OWN_FOLDERS,
    *paths.COPY_OWN_LEFTOVERS,
    "__pycache__",
    ".pytest_cache",
    ".pyc",
    ".bak-",
}


class TestWhatNoPackageIncludes(unittest.TestCase):
    def test_what_a_developers_copy_makes_or_leaves_by_their_one_names(self):
        """No package includes a folder a developer's copy makes at its top (paths.COPY_OWN_FOLDERS, sandbox/ too),
        a cache, a compiled file or a backup it leaves (paths.COPY_OWN_LEFTOVERS), or git's data, wherever they lie:
        package_files reads paths' lists and writes no copy of their names."""
        for folder in paths.COPY_OWN_FOLDERS:
            self.assertIn(folder, package_files.SKIP_PARTS)
            for rel in (f"{folder}/x.py", f"worker/{folder}/x.py"):
                self.assertTrue(package_files.excluded(rel), rel)
        for rel in (
            "rkmon_lib/__pycache__/x.cpython-314.pyc",
            "tests/.pytest_cache/v",
            "x.pyc",
            "docker/rkmon.bak-20261123120000",
            ".git/HEAD",
        ):
            self.assertTrue(package_files.excluded(rel), rel)
        for rel in (
            "rkmon.py",
            "worker/review_sandbox.py",
            "docs/state.md",
            "a.pyc.md",
        ):
            self.assertFalse(package_files.excluded(rel), rel)
        self.assertEqual(constants(package_files, LEFT_BY_A_COPY), [])

    def test_an_install_of_the_worker_tools_copies_what_a_package_includes(self):
        """An install of the worker tools copies no path a package never includes (package_files.excluded), and
        finds none of them stale: deploy.py writes no copy of their names."""
        with tempfile.TemporaryDirectory() as d:
            pkg = pathlib.Path(d, "package")
            for rel in (
                "tool.py",
                "sub/tool2.py",
                "state/x.json",
                "sandbox/x.py",
                "dist/x.py",
                "__pycache__/tool.cpython-314.pyc",
                "tool.pyc",
                "tool.py.bak-20261123120000",
            ):
                (pkg / "worker" / rel).parent.mkdir(parents=True, exist_ok=True)
                (pkg / "worker" / rel).write_text("x\n", encoding="utf-8")
            cfg = {"paths": {"working_dir": str(pathlib.Path(d, "work"))}}
            deploy.install_worker_tools(cfg, pkg)
            dest = pathlib.Path(paths.get(cfg, "worker_tools"))
            installed = {
                p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
            }
            values = {prcode.VALUES_FILE, prcode.MANIFEST}
            self.assertEqual(installed - values, {"tool.py", "sub/tool2.py"})
            self.assertEqual(deploy.stale_worker_tools(cfg, pkg), [])
        self.assertEqual(constants(deploy, LEFT_BY_A_COPY), [])


class TestInstallInto(unittest.TestCase):
    """make_package --install-into: the built folder copied over the package in a folder below the top of a sandbox
    checkout, which it finds as the update finds the package in a fetched commit (package_files.package_folders): the
    build names no folder of the repository."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = pathlib.Path(tmp.name)
        self.built = make_package(self.base / "built", "1.0")
        self.checkout = self.base / "checkout"
        self.checkout.mkdir()
        git(self.checkout, "init", "-q")

    def install(self) -> str:
        """The install's refusal, or "" when it installs."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                build.install_into(self.built, self.checkout)
        except SystemExit as exc:
            return str(exc)
        return ""

    def version(self, place: str) -> str:
        return (self.checkout / place / "VERSION").read_text(encoding="utf-8")

    def test_the_build_replaces_the_package_in_the_checkout(self):
        make_package(self.checkout / "lib" / "tools" / "pkg", "0.9")
        self.assertEqual(self.install(), "")
        self.assertEqual(
            sorted(p.name for p in self.checkout.iterdir()), [".git", "lib"]
        )
        self.assertFalse(
            package_update.differs(self.built, self.checkout / "lib" / "tools" / "pkg")
        )

    def test_a_checkout_with_no_package_below_its_top(self):
        refused = (
            f"--install-into: {self.checkout} has no package in a folder below its top for the built folder to "
            "replace"
        )
        self.assertEqual(self.install(), refused)
        make_package(
            self.checkout, "0.9"
        )  # the package's own repository: its top is the package
        self.assertEqual(self.install(), refused)
        self.assertEqual(self.version("."), "0.9\n")

    def test_a_checkout_with_two_packages(self):
        for place in ("a/pkg", "b/pkg"):
            make_package(self.checkout / place, "0.9")
        self.assertEqual(
            self.install(),
            f"--install-into: {self.checkout} has a package in each of a/pkg and b/pkg, and the built folder "
            "replaces one",
        )
        self.assertEqual((self.version("a/pkg"), self.version("b/pkg")), ("0.9\n",) * 2)


class TestTarball(Case):
    """--from a tarball: unpacked beside the package folder, removed whatever the outcome, then taken as a folder is;
    one the package build made, one that is no tarball or no whole one, and the members refused.
    """

    def tarball(self, members=(), name="new.tar.gz") -> pathlib.Path:
        """~/NAME: ~/new as the package build packs it, then MEMBERS, each (TarInfo, its bytes or None)."""
        path = self.home / name
        with tarfile.open(path, "w:gz") as t:
            t.add(self.new, arcname=build.NAME, filter=build.tar_entry)
            for info, data in members:
                t.addfile(info, None if data is None else io.BytesIO(data))
        return path

    def test_a_tarball_the_build_made(self):
        tarball = self.tarball()
        before = snapshot(self.folder)
        done = package_update.update(self.folder, tarball)
        self.assertEqual(
            (done.word, done.text),
            (
                "UPDATED",
                "~/rocke_monitor has the package from ~/new.tar.gz; the one it replaced is "
                "~/rocke_monitor.prev",
            ),
        )
        self.assertEqual(snapshot(self.folder), self.updated(before))
        self.assert_home("new", "new.tar.gz", "rocke_monitor", "rocke_monitor.prev")
        done = package_update.update(self.folder, tarball)
        self.assertEqual(
            (done.word, done.text),
            (
                "NOTHING TO DO",
                "~/rocke_monitor has the package from ~/new.tar.gz already",
            ),
        )
        self.assert_home("new", "new.tar.gz", "rocke_monitor", "rocke_monitor.prev")

    def test_what_an_earlier_unpack_left_is_cleared_first(self):
        # else its folder would sit beside the tarball's, and the tarball would hold no package
        make_package(self.unpacked / "rocke_monitor-old", "0.9")
        done = package_update.update(self.folder, self.tarball())
        self.assertEqual(done.word, "UPDATED")
        self.assert_home("new", "new.tar.gz", "rocke_monitor", "rocke_monitor.prev")

    def test_what_gives_no_package(self):
        notes = self.home / "notes.txt"
        write(notes, "no tarball\n")
        whole = self.tarball().read_bytes()
        cut, damaged = self.home / "cut.tar.gz", self.home / "damaged.tar.gz"
        cut.write_bytes(whole[: len(whole) // 2])
        middle = len(whole) // 2
        damaged.write_bytes(
            whole[:middle] + bytes([whole[middle] ^ 0x55]) + whole[middle + 1 :]
        )
        empty = self.home / "empty.tar.gz"
        with tarfile.open(empty, "w:gz") as t:
            t.add(self.new / "VERSION", arcname="VERSION")
        before = snapshot(self.folder)
        for path, text in (
            (notes, "~/notes.txt is not a tarball, or not a whole one"),
            (cut, "~/cut.tar.gz is not a tarball, or not a whole one"),
            (damaged, "~/damaged.tar.gz is not a tarball, or not a whole one"),
            (
                empty,
                "~/empty.tar.gz has no package: neither it nor its one folder has rkmon.py and rkmon_lib/",
            ),
        ):
            with self.subTest(path.name):
                with self.assertRaises(Stopped) as caught:
                    package_update.update(self.folder, path)
                self.assert_stopped(
                    caught, "REFUSED", f"{text}; nothing changed", FROM_FIX
                )
                self.assertEqual(snapshot(self.folder), before)
                self.assert_home(
                    "cut.tar.gz",
                    "damaged.tar.gz",
                    "empty.tar.gz",
                    "new",
                    "new.tar.gz",
                    "notes.txt",
                    "rocke_monitor",
                )

    def member(self, name, kind=tarfile.REGTYPE, link=""):
        """A member of a tarball, (TarInfo, its bytes or None): the file NAME, or a KIND to LINK."""
        info = tarfile.TarInfo(name)
        info.type, info.linkname = kind, link
        data = b"escaped\n" if kind == tarfile.REGTYPE else None
        info.size = len(data or b"")
        return info, data

    def refused(self, members) -> str:
        """The text of the refusal of ~/bad.tar.gz, ~/new then MEMBERS: its fix, nothing changed and nothing in
        ~/escaped asserted."""
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, self.tarball(members, "bad.tar.gz"))
        self.assertEqual(
            (caught.exception.word, str(caught.exception.fix)), ("REFUSED", FROM_FIX)
        )
        self.assertFalse(os.path.lexists(self.home / "escaped"))
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("bad.tar.gz", "new", "rocke_monitor")
        return caught.exception.text

    def test_members_that_are_refused(self):
        # by unpack's own checks, before tarfile's data filter where this Python has it
        escaped, docker = self.home / "escaped", f"{build.NAME}/docker"
        leaves = "is a link that leaves the folder, to"
        beside = f"{self.folder.name}{package_update.UNPACKED}-other"
        for name, member, said in (
            (
                "absolute",
                self.member(str(escaped)),
                f"{escaped}, which has a path that leaves the folder",
            ),
            (
                "..",
                self.member(f"{build.NAME}/../../escaped"),
                f"{build.NAME}/../../escaped, which has a path that leaves the folder",
            ),
            (
                "a device",
                self.member(f"{docker}/dev", tarfile.CHRTYPE),
                f"{docker}/dev, which is a device",
            ),
            (
                "a link out",
                self.member(f"{docker}/out", tarfile.SYMTYPE, "../../../escaped"),
                f"{docker}/out, which {leaves} ../../../escaped",
            ),
            (
                "a link to an absolute path",
                self.member(f"{docker}/out", tarfile.SYMTYPE, str(escaped)),
                f"{docker}/out, which {leaves} {escaped}",
            ),
            (
                "a hard link out",
                self.member(f"{docker}/out", tarfile.LNKTYPE, "../escaped"),
                f"{docker}/out, which {leaves} ../escaped",
            ),
            (
                "a link to a folder beside, whose name begins with the folder's",
                self.member(f"{docker}/out", tarfile.SYMTYPE, f"../../../{beside}"),
                f"{docker}/out, which {leaves} ../../../{beside}",
            ),
        ):
            with self.subTest(name):
                self.assertEqual(
                    self.refused([member]),
                    f"~/bad.tar.gz has {said}; nothing changed",
                )

    def test_a_chain_of_links(self):
        # d, a link that stays inside while f is missing, then f, a link to its own folder, which takes d out: the
        # file through d is refused where tarfile has no data filter, as the host's Python may not; where it has one
        # that writes a link's text as it reads it, d stays inside. Nothing lands outside either way.
        docker = f"{build.NAME}/docker"
        members = [
            self.member(f"{docker}/d", tarfile.SYMTYPE, "f/../../.."),
            self.member(f"{docker}/f", tarfile.SYMTYPE, "."),
            self.member(f"{docker}/d/escaped"),
        ]
        alone = {"filter": "fully_trusted"} if hasattr(tarfile, "data_filter") else {}
        with mock.patch.object(package_files, "_FILTER", alone):
            self.assertEqual(
                self.refused(members),
                f"~/bad.tar.gz has {docker}/d/escaped, which would land outside the folder, through a link before "
                "it; nothing changed",
            )
        with contextlib.suppress(Stopped):
            package_update.update(self.folder, self.tarball(members, "bad.tar.gz"))
        self.assertFalse(os.path.lexists(self.home / "escaped"))

    def test_the_data_filter(self):
        # where this Python has it, it refuses a member that leaves the folder, unpack's own checks left out
        if not hasattr(tarfile, "data_filter"):
            self.skipTest("this Python's tarfile has no data filter")
        name = f"{build.NAME}/../../escaped"
        with mock.patch.object(package_files, "_unsafe", return_value=""):
            text = self.refused([self.member(name)])
        self.assertRegex(
            text,
            rf"^~/bad\.tar\.gz has {re.escape(name)}, which is refused by tarfile's data filter: .+; nothing "
            "changed$",
        )


class TestFetch(Case):
    """The fetch from ~/rkmon-deploy.yaml's updates section, from a repository of the test's at <org>/<repo>
    acme/widgets: its develop branch holds a newer package (2.0) in PLACE, a folder no code names, a big file in a
    folder beside it, and files elsewhere."""

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.folder)
        make_package(self.folder, "1.0")
        self.origin = self.tmp / "origin" / "acme" / "widgets"
        make_package(self.origin / PLACE, "2.0")
        self.beside = f"{PLACE.rsplit('/', 1)[0]}/other_tool/big.txt"
        for rel in (
            self.beside,
            "lib/README.md",
            "README.md",
            "elsewhere/z.txt",
        ):
            write(self.origin / rel, f"{rel}\n")
        git(self.origin, "init", "-q", "-b", "develop")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-qm", "develop: first")
        git(self.origin, "config", "uploadpack.allowFilter", "true")
        base = (self.tmp / "origin").as_uri()
        patcher = mock.patch.object(
            derived, "github_address", lambda repo: f"{base}/{repo}"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.file = self.home / "rkmon-deploy.yaml"
        self.settings(
            {"updates.repository": "acme/widgets", "updates.branch": "develop"}
        )
        image = records.Image("", "rocke-monitor", "1.0")
        confirmed = records.Confirmed(
            str(self.file),
            "sha256:0",
            records.timestamp(),
            "sha256:0",
            "1.0",
            "sha256:0",
            "rocke-monitor",
            image,
            0,
        )
        records.confirm(self.folder, confirmed)

    def settings(self, changes) -> None:
        """~/rkmon-deploy.yaml, the deploy template with CHANGES, yours only."""
        if self.file.exists():
            self.file.unlink()
        fd = os.open(self.file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(settings_file.dump(doc(changes)))

    def test_the_update(self):
        done = package_update.update(self.folder)
        self.assertEqual(
            (done.word, done.text),
            (
                "UPDATED",
                "~/rocke_monitor has the package from the newest commit of develop in acme/widgets; the one it "
                "replaced is ~/rocke_monitor.prev",
            ),
        )
        self.assertFalse(package_update.differs(self.folder, self.origin / PLACE))
        self.assertEqual(
            rows(done),
            [
                (
                    records.FILE,
                    Status.PASS,
                    f"carried to {self.folder / records.FILE} at 600",
                )
            ],
        )
        self.assertEqual(
            records.read(self.folder).confirmed.settings_file, "~/rkmon-deploy.yaml"
        )
        self.assert_home(
            "new", "rkmon-deploy.yaml", "rocke_monitor", "rocke_monitor.prev"
        )
        done = package_update.update(self.folder)
        self.assertEqual(
            (done.word, done.text),
            (
                "NOTHING TO DO",
                "~/rocke_monitor has the package from the newest commit of develop in acme/widgets already",
            ),
        )
        self.assert_home(
            "new", "rkmon-deploy.yaml", "rocke_monitor", "rocke_monitor.prev"
        )

    def test_only_the_packages_folder_is_fetched(self):
        into = self.tmp / "clone"
        url = derived.github_address("acme/widgets")
        new = package_update.fetch(url, "develop", into)
        self.assertEqual(new, into / PLACE)
        self.assertFalse(package_update.differs(new, self.origin / PLACE))
        listed = git(into, "rev-list", "--objects", "--missing=print", "HEAD").split()
        missing = {line[1:] for line in listed if line.startswith("?")}
        blob = {
            rel: git(self.origin, "rev-parse", f"HEAD:{rel}")
            for rel in (self.beside, "elsewhere/z.txt")
        }
        self.assertLessEqual(set(blob.values()), missing)
        own = git(self.origin, "rev-parse", f"HEAD:{PLACE}/rkmon_lib/code.py")
        self.assertNotIn(own, missing)
        self.assertFalse((into / "elsewhere").exists())

    def test_a_fetch_with_no_git(self):
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp / "nothing")}):
            with self.assertRaises(package_update.FetchFailed) as caught:
                package_update.fetch("file:///nothing", "develop", self.tmp / "clone")
        self.assertEqual(str(caught.exception), "git: No such file or directory")

    def test_a_fetch_that_fails(self):
        self.settings(
            {"updates.repository": "acme/widgets", "updates.branch": "nosuch"}
        )
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assertEqual(caught.exception.word, "ERROR")
        self.assertRegex(
            caught.exception.text,
            r"^the fetch of the newest commit of nosuch in acme/widgets failed: .*nosuch.*; nothing changed$",
        )
        self.assertEqual(
            str(caught.exception.fix),
            "update from a package folder instead: rkmon package update --from DIR",
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("new", "rkmon-deploy.yaml", "rocke_monitor")

    def test_a_branch_without_the_package(self):
        git(self.origin, "checkout", "-qb", "bare")
        git(self.origin, "rm", "-rq", PLACE)
        git(self.origin, "commit", "-qm", "bare: no package")
        git(self.origin, "checkout", "-q", "develop")
        self.settings({"updates.repository": "acme/widgets", "updates.branch": "bare"})
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assert_stopped(
            caught,
            "ERROR",
            "the fetch of the newest commit of bare in acme/widgets failed: it has no package; nothing changed",
            "update from a package folder instead: rkmon package update --from DIR",
        )
        self.assert_home("new", "rkmon-deploy.yaml", "rocke_monitor")

    def test_a_branch_with_two_packages(self):
        """A commit with a package in two folders: the fetch takes neither, and names both."""
        git(self.origin, "checkout", "-qb", "two")
        make_package(self.origin / "other" / "monitor", "2.1")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-qm", "two: a second package")
        git(self.origin, "checkout", "-q", "develop")
        self.settings({"updates.repository": "acme/widgets", "updates.branch": "two"})
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assert_stopped(
            caught,
            "ERROR",
            "the fetch of the newest commit of two in acme/widgets failed: it has a package in each of "
            f"{PLACE} and other/monitor; nothing changed",
            "update from a package folder instead: rkmon package update --from DIR",
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("new", "rkmon-deploy.yaml", "rocke_monitor")

    def test_a_repository_that_is_the_package(self):
        """A repository whose top folder is the package: the fetch takes the whole tree."""
        origin = self.tmp / "origin" / "acme" / "monitor"
        make_package(origin, "3.0")
        git(origin, "init", "-q", "-b", "develop")
        git(origin, "add", "-A")
        git(origin, "commit", "-qm", "develop: the package")
        git(origin, "config", "uploadpack.allowFilter", "true")
        into = self.tmp / "clone"
        url = derived.github_address("acme/monitor")
        self.assertEqual(package_update.fetch(url, "develop", into), into)
        self.assertFalse(package_update.differs(into, origin))

    def test_a_dry_run(self):
        # the newer package reads this package's layout, and your file changed after the last confirm
        schema(self.origin / PLACE)
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-qm", "develop: the deploy schema")
        before = snapshot(self.folder)
        done = package_update.update(self.folder, dry_run=True)
        self.assertEqual(
            (done.word, done.text, done.next),
            (
                "PASSED",
                "the package from the newest commit of develop in acme/widgets can replace ~/rocke_monitor",
                "next: rkmon package update, then rkmon confirm",
            ),
        )
        self.assertEqual(snapshot(self.folder), before)
        self.assert_home("new", "rkmon-deploy.yaml", "rocke_monitor")

    def test_a_named_folder_s_refusal_and_dry_run_name_it(self):
        """With --folder (NAMED), and no --from: the refusal's fix and the dry run's next line update that folder."""
        schema(self.origin / PLACE)
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-qm", "develop: the deploy schema")
        done = package_update.update(self.folder, dry_run=True, named=True)
        self.assertEqual(
            (done.word, done.next),
            ("PASSED", f"next: rkmon package update {NAMED}, then {OWN} confirm"),
        )
        self.settings({})
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder, named=True)
        self.assertEqual(
            str(caught.exception.fix),
            f"rkmon package update --from DIR {NAMED}, or an updates section in ~/rkmon-deploy.yaml (rkmon template "
            "section updates prints it)",
        )

    def test_git_pointed_at_another_repository(self):
        other = self.tmp / "other"
        other.mkdir()
        git(other, "init", "-q")
        env = {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other)}
        with mock.patch.dict(os.environ, env):
            done = package_update.update(self.folder)
        self.assertEqual(done.word, "UPDATED")
        self.assertFalse(package_update.differs(self.folder, self.origin / PLACE))
        self.assertEqual(git(other, "status", "--porcelain"), "")

    def assert_earlier_step(self, why, within="your settings file"):
        """An update with no --from refused as an earlier step, WHY, its fix naming both ways."""
        before = snapshot(self.folder)
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assert_stopped(
            caught,
            "REFUSED",
            f"there is no --from, and {why}; nothing changed",
            f"rkmon package update --from DIR, or an updates section in {within} (rkmon template section updates "
            "prints it)",
        )
        self.assertTrue(caught.exception.earlier_step)
        self.assertEqual(snapshot(self.folder), before)

    def test_no_settings_file_confirmed(self):
        shutil.rmtree(self.folder / records.FOLDER)
        self.assert_earlier_step("no settings file is confirmed in ~/rocke_monitor")

    def test_the_settings_file_gone(self):
        self.file.unlink()
        self.assert_earlier_step(
            "~/rkmon-deploy.yaml, the settings file confirmed here, is gone"
        )

    def test_no_updates_section(self):
        self.settings({})
        self.assert_earlier_step(
            "~/rkmon-deploy.yaml has no updates section", "~/rkmon-deploy.yaml"
        )

    def test_no_git(self):
        with self.no_git(), self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assert_stopped(
            caught,
            "REFUSED",
            "git, which fetches the newest commit of develop in acme/widgets and reads which of its files a package "
            "includes, is not on this host; nothing changed",
            "install git, then run rkmon package update",
        )
        self.assertTrue(caught.exception.earlier_step)

    def test_an_updates_section_not_filled_in(self):
        self.settings(
            {"updates.repository": "<org>/<repo>", "updates.branch": "<branch>"}
        )
        with self.assertRaises(Stopped) as caught:
            package_update.update(self.folder)
        self.assert_stopped(
            caught,
            "REFUSED",
            "the updates section of ~/rkmon-deploy.yaml is not filled in; nothing changed (fix each FAIL above)",
        )
        self.assertFalse(caught.exception.earlier_step)
        self.assertEqual(
            [(r.name, r.status) for r in caught.exception.rows],
            [("updates.repository", Status.FAIL), ("updates.branch", Status.FAIL)],
        )

    def test_a_settings_file_others_may_read(self):
        self.file.chmod(0o644)
        with self.assertRaises(private_files.Refused) as caught:
            package_update.update(self.folder)
        self.assertEqual({r.name for r in caught.exception.rows}, {"settings file"})
        self.assert_home("new", "rkmon-deploy.yaml", "rocke_monitor")


# The deploy schema's place in a package, and the layout this package reads.
SCHEMA = pathlib.Path(file_types.DEPLOY.schema)
LAYOUT = file_types.DEPLOY.version


def schema(top: pathlib.Path, layout: int = LAYOUT) -> None:
    """The deploy schema of the package at TOP: this package's, but for the layout it reads, LAYOUT."""
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    data["properties"]["schema_version"]["const"] = layout
    write(top / SCHEMA.relative_to(paths.PACKAGE), json.dumps(data))


class TestWhatComesNext(Case):
    """What an update says of what comes after it: the image row, when the new package needs a new image, and the
    next line of the state it leaves the folder in, which the package now in the folder decides. ~/rkmon-deploy.yaml
    is confirmed here with the package 1.0 and its code, and each package reads this package's layout.
    """

    def setUp(self):
        super().setUp()
        self.file = self.home / "rkmon-deploy.yaml"
        self.settings({})
        (self.folder / records.FILE).unlink()
        for top in (self.folder, self.new):
            schema(top)
        self.confirm("1.0")

    def settings(self, changes) -> None:
        """~/rkmon-deploy.yaml, the deploy template with CHANGES, yours only."""
        if self.file.exists():
            self.file.unlink()
        fd = os.open(self.file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(settings_file.dump(doc(changes)))

    def confirm(self, version: str, code: str = "") -> None:
        """~/rkmon-deploy.yaml, as it is now, confirmed here with the package VERSION and the code CODE ("": the
        code of ~/rocke_monitor, image.code_sha256)."""
        values = settings_file.read(self.file, file_types.DEPLOY)
        confirmed = records.Confirmed(
            str(self.file),
            "sha256:0",
            records.timestamp(),
            deploy_plan.file_sha256(values),
            version,
            code or image.code_sha256(self.folder),
            "rocke-monitor",
            records.Image("", "rocke-monitor", version),
            0,
        )
        records.confirm(self.folder, confirmed)

    def brought(self, source: pathlib.Path) -> str:
        """The code that the update from SOURCE puts in ~/rocke_monitor: SOURCE's files, with the local files of
        its docker/ that the update carries (package_update.others), as image.code_sha256 reads them.
        """
        tree = self.tmp / "brought"
        shutil.copytree(source, tree, symlinks=True)
        for rel in package_update.others(self.folder, source):
            shutil.copy2(self.folder / rel, tree / rel, follow_symlinks=False)
        return image.code_sha256(tree)

    def update(self, source, then=None) -> package_update.Done:
        """The update from SOURCE, a folder in your home folder, after its dry run, whose next line runs the update,
        then names THEN, the command that the update's own next line names, without a deploy step's words: when THEN
        is None, that line's words after "next: "."""
        dry = package_update.update(self.folder, source, dry_run=True)
        done = package_update.update(self.folder, source)
        self.assertTrue(done.next.startswith("next: "), done.next)
        then = done.next[len("next: ") :] if then is None else then
        again = f"rkmon package update --from ~/{source.name}"
        self.assertEqual(
            (dry.word, dry.next), ("PASSED", f"next: {again}, then {then}")
        )
        return done

    def test_the_same_image_then_a_confirm(self):
        done = self.update(self.new, then="rkmon confirm")
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )
        self.assertEqual({r.status for r in done.rows}, {Status.PASS})

    def test_once_the_records_keep_a_first_tick_the_command_alone(self):
        """This folder's records, which the update carries, keep a first tick: deploy step 8 has passed after a start,
        so the next line names the confirm without its step (states.step_8_passed).
        """
        records.first_tick(self.folder, records.FirstTick(records.timestamp()))
        done = self.update(self.new, then="rkmon confirm")
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon confirm"))

    def test_a_new_image(self):
        write(self.new / "docker/Dockerfile", "FROM base:2\n")
        write(self.new / "requirements/base.txt", "tool==2\n")
        expected = (
            package_update.IMAGE_ROW,
            Status.WARN,
            "the package from ~/new needs a new image: it changes files the image keeps its own copies of: "
            "docker/Dockerfile, requirements/base.txt",
            "build an image from the new package (QUICK_BUILD_GUIDE.md), then set image.tag in your deploy settings "
            "file to its tag",
        )
        dry = package_update.update(self.folder, self.new, dry_run=True)
        done = package_update.update(self.folder, self.new)
        for name, outcome in (("the dry run", dry), ("the update", done)):
            with self.subTest(name):
                row = outcome.rows[-1]
                self.assertEqual(
                    (row.name, row.status, row.found, str(row.fix)), expected
                )
        self.assertEqual(
            (dry.word, [r.name for r in dry.rows], dry.next),
            (
                "PASSED",
                [r.name for r in done.rows],
                "next: rkmon package update --from ~/new, then rkmon confirm",
            ),
        )
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )

    def test_a_named_folder_s_next_lines(self):
        """With --folder (NAMED), there and back: the dry run's next line runs the update of that folder from here,
        then the command that the folder's own rkmon (OWN) runs there; the update's next line names that command,
        with its deploy step's words.
        """
        dry = package_update.update(self.folder, self.new, dry_run=True, named=True)
        done = package_update.update(self.folder, self.new, named=True)
        self.assertEqual(
            (dry.word, dry.next, done.word, done.next),
            (
                "PASSED",
                f"next: rkmon package update --from ~/new {NAMED}, then {OWN} confirm",
                "UPDATED",
                f"next: {CONFIRM_WORDS} {OWN} confirm",
            ),
        )
        self.assertEqual(
            (
                package_update.next_line(self.folder, named=True),
                package_update.next_line(self.folder),
            ),
            (
                f"next: {CONFIRM_WORDS} {OWN} confirm",
                f"next: {CONFIRM_WORDS} rkmon confirm",
            ),
        )
        back = package_update.update(self.folder, self.prev, dry_run=True, named=True)
        done = package_update.update(self.folder, self.prev, named=True)
        # the earlier package's own records: a state docker's answers decide
        self.assertEqual(
            (back.word, back.next, done.word, done.next),
            (
                "PASSED",
                f"next: rkmon package update --from ~/rocke_monitor.prev {NAMED}, then {OWN} next",
                "UPDATED",
                f"next: {OWN} next",
            ),
        )

    def test_a_package_that_reads_a_later_layout(self):
        schema(self.new, LAYOUT + 1)
        done = self.update(self.new)
        self.assertEqual(
            done.next, "next: rkmon migrate --settings ~/rkmon-deploy.yaml"
        )

    def test_a_package_that_reads_an_earlier_layout(self):
        schema(self.new, LAYOUT - 1)
        done = self.update(self.new)
        self.assertEqual(
            done.next,
            "next: rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one",
        )

    def test_nothing_confirmed(self):
        shutil.rmtree(self.folder / records.FOLDER)
        done = self.update(self.new)
        self.assertEqual(
            done.next,
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check "
            "--settings FILE)",
        )

    def test_the_way_back_is_the_earlier_package_s_state(self):
        package_update.update(self.folder, self.new)
        # your file, changed after the earlier package's confirm, which its folder keeps
        self.settings({"host.resources.cpus": 6})
        done = self.update(self.prev, then="rkmon confirm")
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )

    def test_the_way_back_reads_the_earlier_folder_s_own_records(self):
        package_update.update(self.folder, self.new)
        # the newer package confirmed in its folder: the earlier folder's records still confirm 1.0, its own
        self.confirm("2.0")
        done = self.update(self.prev)
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon next"))

    def test_a_state_that_needs_docker_s_answers(self):
        # what the update brings is what was confirmed, its code too, and your file is unchanged
        self.confirm("2.0", self.brought(self.new))
        done = self.update(self.new)
        self.assertEqual(done.next, "next: rkmon next")

    def test_other_code_under_the_confirmed_version(self):
        """A later commit of updates.branch: a module changed and VERSION kept. The package changed:
        the next line is the confirm's (state 5), the dry run's too."""
        self.confirm("2.0", self.brought(self.new))
        write(self.new / "rkmon_lib/code.py", "VERSION = '2.0'\nSTEP = 2\n")
        done = self.update(self.new, then="rkmon confirm")
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )

    def test_a_local_file_the_update_carries_is_code(self):
        """A note of yours in docker/, which the package's .dockerignore lets in, is code the update puts in the
        folder: confirmed with ~/new's code alone, the package changed; the dry run reads the note as the update
        does."""
        self.confirm("2.0", image.code_sha256(self.new))
        self.assertIn("docker/notes.txt", package_update.others(self.folder, self.new))
        done = self.update(self.new, then="rkmon confirm")
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )

    def test_records_that_do_not_read(self):
        write(self.folder / records.FILE, "{}\n", 0o600)
        done = self.update(self.new)
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon next"))

    def looser(self, mode: int) -> tuple:
        """(the dry run, the update), FOLDER's records at MODE, which the update's carry sets to 600 and the dry run
        leaves as they are."""
        held = self.folder / records.FILE
        os.chmod(held, mode)
        dry = package_update.update(self.folder, self.new, dry_run=True)
        self.assertEqual(os.lstat(held).st_mode & 0o777, mode)
        return dry, package_update.update(self.folder, self.new)

    def test_records_others_can_read(self):
        """The dry run reads them as the carry leaves them, and its row says that the carry sets them to 600."""
        dry, done = self.looser(0o644)
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )
        self.assertEqual(
            dry.next, "next: rkmon package update --from ~/new, then rkmon confirm"
        )
        self.assertIn(
            (
                records.FILE,
                Status.PASS,
                f"would be carried to {self.folder / records.FILE} at 600; its permissions are 644, and would be "
                "600 in both folders",
            ),
            rows(dry),
        )

    def test_records_you_cannot_read(self):
        """The update reads them once its carry has set them to 600; the dry run cannot, and its line says so."""
        if os.geteuid() == 0:
            self.skipTest("root reads a file whatever its permissions")
        dry, done = self.looser(0o200)
        self.assertEqual(
            (done.word, done.next), ("UPDATED", f"next: {CONFIRM_WORDS} rkmon confirm")
        )
        self.assertEqual(
            dry.next,
            "next: rkmon package update --from ~/new, which sets ~/rocke_monitor/docker/.rkmon/records.json to "
            "600 first, then rkmon next",
        )

    def test_a_package_that_says_no_layout(self):
        (self.new / SCHEMA.relative_to(paths.PACKAGE)).unlink()
        done = self.update(self.new)
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon next"))

    def test_the_way_back_to_records_that_break_a_rule(self):
        package_update.update(self.folder, self.new)
        os.chmod(self.prev / records.FILE, 0o644)
        done = self.update(self.prev)
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon next"))

    def test_facts_that_an_error_of_the_system_keeps_from_being_read(self):
        denied = PermissionError(errno.EACCES, "Permission denied")
        with mock.patch.object(state_facts, "gather", side_effect=denied):
            done = self.update(self.new)
        self.assertEqual((done.word, done.next), ("UPDATED", "next: rkmon next"))


if __name__ == "__main__":
    unittest.main()
