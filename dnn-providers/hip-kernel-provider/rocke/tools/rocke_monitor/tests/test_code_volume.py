# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The code is root's (rkmon_lib/code_volume.py): root installs a release in the code volume and goes back to the code
kept (docker/rk_root.py), and the writer's broker (docker/rk_write.py) runs only the code the launcher's rule picks,
once it has checked that this code is root's and read-only all the way.

The tests are not root. A fake root (FakeRoot) reports the tests' own files as root's, the folders above the tests'
folder as root's and read-only, and the paths a test names as another user's; the image test's isolation stage
(tests/container/stage_isolation.py) runs the real users.
"""

import contextlib
import fcntl
import importlib.util
import io
import json
import os
import pathlib
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    checks,
    code_volume,
    container_layout,
    doctor,
    features,
    paths,
    releases,
    writer,
    writer_run,
)


def script(name: str):
    """docker/NAME, loaded as a module of its own, as the image runs it."""
    spec = importlib.util.spec_from_file_location(
        f"{name[:-3]}_of_the_code_tests", ROOT / "docker" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rk_root = script("rk_root.py")
rk_write = script("rk_write.py")
OTHER_UID = 4242


class FakeRoot:
    """os.lstat as root would see the tests' files were they root's: the tests' own (this user's, this user's group)
    are root's, the folders above BASE root's and mode 755, and each path in OTHER another user's (OTHER_UID).
    """

    def __init__(self, base: pathlib.Path):
        self.base, self.other = str(base), set()

    def __call__(self, path):
        path = os.fspath(path)
        s = os.lstat(path)
        if not (path == self.base or path.startswith(self.base + "/")):
            return self.stat(s, 0, 0, stat.S_IFDIR | 0o755)
        uid = (
            OTHER_UID
            if path in self.other
            else (0 if s.st_uid == os.getuid() else s.st_uid)
        )
        return self.stat(s, uid, 0 if s.st_gid == os.getgid() else s.st_gid, s.st_mode)

    @staticmethod
    def stat(s, uid, gid, mode):
        return os.stat_result(
            (mode, s.st_ino, s.st_dev, s.st_nlink, uid, gid, s.st_size, 0, 0, 0)
        )


def write(top: pathlib.Path, rel: str, text: str, mode: int = 0o644) -> pathlib.Path:
    p = top / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    p.chmod(mode)
    return p


# The data file the tests' packages change from one version to the next, which their rkmon.py reads from its own tree.
DATA = "rkmon_lib/data.json"
# rkmon.py of the tests' packages: it prints the code it runs (RKMON_CODE) and the data file of its own tree.
RKMON_PY = (
    "import os, sys\n"
    "here = os.path.dirname(os.path.realpath(__file__))\n"
    f"data = open(os.path.join(here, {DATA!r})).read().strip()\n"
    "print(os.environ.get('RKMON_CODE'), data, ' '.join(sys.argv[1:]), flush=True)\n"
)


def package(top: pathlib.Path, version: str, bytecode: bool = False) -> pathlib.Path:
    """A whole package at TOP (code_volume.WHOLE): the launcher, rkmon.py, a module, a data file, a script a job
    starts by its path, and VERSION; with BYTECODE, Python's bytecode too, which an install leaves out.
    """
    top.mkdir(parents=True)
    shutil.copy2(ROOT / "rk", top / "rk")
    write(top, "rkmon.py", RKMON_PY)
    write(top, "rkmon_lib/store.py", "# the code\n")
    write(top, DATA, json.dumps({"version": version}) + "\n")
    write(top, "tools/job.py", "# a script a job starts by its path\n", 0o755)
    write(top, "VERSION", f"# the package's version\n{version}\n")
    if bytecode:
        write(top, "rkmon_lib/__pycache__/store.cpython-312.pyc", "bytecode\n")
    return top


def archive(top: pathlib.Path, *extra: tarfile.TarInfo) -> io.BytesIO:
    """The package at TOP as `rkmon apply` hands it to root: a tar archive of one folder, with EXTRA members."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tf:
        tf.add(top, arcname="rocke_monitor")
        for member in extra:
            tf.addfile(
                member, io.BytesIO(b"x" * member.size) if member.isreg() else None
            )
    data.seek(0)
    return data


def member(name: str, kind=tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    m = tarfile.TarInfo(name)
    m.type, m.linkname, m.mode = kind, link, 0o644
    m.size = 1 if kind == tarfile.REGTYPE else 0
    return m


class Base(unittest.TestCase):
    """A code volume over an image of the tests' own (its code, and its digest in a file), with a fake root."""

    def setUp(self):
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.base = pathlib.Path(d.name).resolve()
        self.image = package(self.base / "opt" / "rocke_monitor", "2026.11.18")
        self.idf = self.base / "image_code_id"
        self.idf.write_text("img1\n")
        self.code = self.base / "code"
        self.root = FakeRoot(self.base)
        self.chowned = []
        # The worker tools' values file, which the tests' own user reads as it scans its processes.
        self.values = self.base / "work" / code_volume.WORKER_VALUES
        self.volume = rk_root.Volume(
            folder=str(self.code),
            image=str(self.image),
            id_file=str(self.idf),
            chown=self.chown,
            lstat=self.root,
            scans=[(os.getuid(), os.getgid(), str(self.values))],
        )
        self.src = self.base / "src"

    def chown(self, path, uid, gid, follow_symlinks=True):
        """The fake root's chown: noted, and PATH is no other user's from then on."""
        self.chowned.append((path, uid, gid))
        self.root.other.discard(os.fspath(path))

    def kept_digest(self) -> str:
        """The digest of the tests' package as an install keeps it: without Python's bytecode."""
        shutil.rmtree(self.src / "rkmon_lib" / "__pycache__", ignore_errors=True)
        return code_volume.digest(self.src)

    def pkg(self, version: str) -> io.BytesIO:
        """The tests' package at VERSION, as its archive."""
        shutil.rmtree(self.src, ignore_errors=True)
        return archive(package(self.src, version, bytecode=True))

    def install(self, version: str) -> pathlib.Path:
        """The package at VERSION installed: the release that runs then."""
        self.assertTrue(
            self.volume.install(self.pkg(version))[-1].startswith("INSTALLED: ")
        )
        return self.current()

    def current(self) -> pathlib.Path:
        return (self.code / code_volume.CURRENT).resolve()

    def kept(self) -> str:
        return os.readlink(self.code / code_volume.KEPT)

    def released(self) -> list:
        return sorted(os.listdir(self.code / code_volume.RELEASES))

    def state(self) -> tuple:
        """What a refusal must leave as it was: the volume's names, its links and its releases."""
        names = sorted(os.listdir(self.code)) if self.code.is_dir() else []
        links = [
            os.readlink(self.code / n) if os.path.islink(self.code / n) else None
            for n in (code_volume.CURRENT, code_volume.KEPT)
        ]
        return (
            names,
            links,
            self.released() if (self.code / "releases").is_dir() else [],
        )

    def which_code(self) -> str:
        """The code the launcher picks for a process tree that starts now, the health check's among them (rk
        --which-code, the image's rk, in the container)."""
        env = dict(
            os.environ,
            RKMON_CONTAINER="1",
            RKMON_CODE_DIR=str(self.code),
            RKMON_IMAGE_CODE=str(self.image),
            RKMON_IMAGE_ID_FILE=str(self.idf),
        )
        for name in ("RKMON_CODE", "RKMON_WRAPPER"):
            env.pop(name, None)
        p = subprocess.run(
            ["sh", str(self.image / "rk"), "--which-code"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.strip()


class Install(Base):
    def test_install_then_back(self):
        """A package becomes a release over this image: root's and read-only (no write bit for anyone; the folders and
        the scripts it runs readable and searchable), its record written, no bytecode; CURRENT names it and KEPT the
        code CURRENT named before it, each in one rename. rk_root's code-back runs the code kept again.
        """
        first = self.install("2026.11.19")
        self.assertRegex(first.name, r"^2026\.11\.19-[0-9a-f]{12}$")
        self.assertEqual(first.parent, self.code / code_volume.RELEASES)
        self.assertEqual(self.kept(), str(self.image))
        self.assertEqual(
            os.readlink(self.code / code_volume.CURRENT), f"releases/{first.name}"
        )
        info = code_volume.release_info(first)
        self.assertEqual(
            (info["id"], info["version"], info["base_image"], info["digest"]),
            (first.name, "2026.11.19", "img1", self.kept_digest()),
        )
        modes = {
            p.relative_to(first).as_posix(): stat.S_IMODE(p.stat().st_mode)
            for p in [first, *first.rglob("*")]
        }
        self.assertEqual(modes["."], code_volume.DIR_MODE)
        self.assertEqual(modes["rkmon_lib"], code_volume.DIR_MODE)
        self.assertEqual(modes["rk"], code_volume.RUN_MODE)
        self.assertEqual(modes["tools/job.py"], code_volume.RUN_MODE)
        self.assertEqual(modes[DATA], code_volume.FILE_MODE)
        self.assertEqual(modes[code_volume.RELEASE_FILE], code_volume.FILE_MODE)
        self.assertEqual([p for p, m in modes.items() if m & 0o222], [])
        self.assertFalse([p for p in modes if "__pycache__" in p or p.endswith(".pyc")])
        self.assertEqual(self.volume.problems(), [])
        self.assertEqual(code_volume.running(self.code, "img1"), first)
        # A second install keeps the first; a third prunes it, as neither link names it any more.
        second = self.install("2026.11.20")
        self.assertEqual(
            (self.kept(), self.released()),
            (f"releases/{first.name}", sorted([first.name, second.name])),
        )
        said = self.volume.install(self.pkg("2026.11.21"))
        third = self.current()
        self.assertEqual(
            said,
            [
                f"INSTALLED: release {third.name} (2026.11.21) over this image, in place of release {second.name}, "
                "which is kept: rkmon rollback goes back to it"
            ],
        )
        self.assertEqual(self.released(), sorted([second.name, third.name]))
        self.assertEqual(
            self.volume.back(),
            [
                f"RESTORED: release {second.name} runs again, in place of release {third.name}"
            ],
        )
        self.assertEqual(self.current(), second)
        self.assertFalse(os.path.lexists(self.code / code_volume.KEPT))
        self.assertEqual(
            self.volume.back(), ["NOTHING TO DO: no code is kept to go back to"]
        )
        self.assertEqual(
            self.volume.start(), []
        )  # the next start prunes the release no link names
        self.assertEqual(self.released(), [second.name])

    def test_each_install_leaves_the_releases_folder_mode_755_whatever_the_umask(
        self,
    ):
        """The entry point runs the root step with umask 027, and the launcher and the write broker run as other
        users: the releases folder is mode 755 after each install, else the image's own code runs for them.
        """
        os.umask(0o027)
        folder = self.install("2026.11.19").parent
        self.assertEqual(oct(stat.S_IMODE(folder.stat().st_mode)), oct(0o755))
        folder.chmod(0o750)
        self.install("2026.11.20")
        self.assertEqual(oct(stat.S_IMODE(folder.stat().st_mode)), oct(0o755))

    def test_no_set_id_bit_reaches_a_release(self):
        """A file the archive gives the set-user-id and set-group-id bits is put in place as any script is: 0o555.
        The modes root gives are held too, not only the one the tests' user ends with: a write by a user without
        CAP_FSETID drops those bits, and root keeps them."""
        setid = member("rocke_monitor/tools/setid.py")
        setid.mode = 0o6755
        data = archive(package(self.src, "2026.11.19"), setid)
        with mock.patch.object(rk_root.os, "fchmod", wraps=os.fchmod) as fchmod:
            said = self.volume.install(data)
        self.assertTrue(said[-1].startswith("INSTALLED: "), said)
        made = self.current() / "tools" / "setid.py"
        self.assertEqual(oct(stat.S_IMODE(made.stat().st_mode)), oct(0o555))
        given = {c.args[1] for c in fchmod.call_args_list}
        self.assertIn(0o555, given)
        self.assertEqual([oct(m) for m in given if m & 0o7000], [])

    def test_back_to_the_image_s_own_code(self):
        first = self.install("2026.11.19")
        self.assertEqual(
            self.volume.back(),
            [
                f"RESTORED: the image's own code runs again, in place of release {first.name}"
            ],
        )
        self.assertFalse(os.path.lexists(self.code / code_volume.CURRENT))
        self.assertIsNone(code_volume.running(self.code, "img1"))

    def test_the_same_code_again(self):
        self.assertEqual(
            self.volume.install(archive(self.image)),
            ["NOTHING TO DO: this code runs already (the image's own code)"],
        )
        first = self.install("2026.11.19")
        before = self.state()
        self.assertEqual(
            self.volume.install(self.pkg("2026.11.19")),
            [f"NOTHING TO DO: this code runs already (release {first.name})"],
        )
        self.assertEqual(self.state(), before)

    def new_image(self, version: str, image_id: str) -> None:
        """Another image in the image's place: its own code the tests' package at VERSION (by files outside
        IMAGE_BOUND, so that the same releases install over it), its code digest IMAGE_ID.
        """
        write(self.image, DATA, json.dumps({"version": version}) + "\n")
        write(self.image, "VERSION", f"# the package's version\n{version}\n")
        self.idf.write_text(f"{image_id}\n")

    def test_a_release_runs_only_over_its_image(self):
        """After a new image, a release installed over the old one does not run. code-back puts the release kept in
        CURRENT's place whatever image runs, and says it is back, not that it runs: it runs once its own image does
        (the launcher's rule). A release kept that is not whole is refused, nothing changed.
        """
        first = self.install("2026.11.19")
        second = self.install("2026.11.20")
        self.idf.write_text("img2\n")
        self.assertIsNone(code_volume.running(self.code, "img2"))
        self.assertEqual(
            self.volume.back(),
            [
                f"RESTORED: release {first.name} (installed over another image) is back, in place of release "
                f"{second.name} (installed over another image)"
            ],
        )
        self.assertEqual(self.current(), first)
        self.assertEqual(self.which_code(), str(self.image))
        self.idf.write_text("img1\n")
        self.assertEqual(self.which_code(), str(first))
        third = self.install("2026.11.21")
        self.assertEqual(self.kept(), f"releases/{first.name}")
        first.chmod(0o755)
        (first / "rkmon.py").unlink()
        first.chmod(code_volume.DIR_MODE)
        before = self.state()
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.back()
        self.assertEqual(stop.exception.code, rk_root.REFUSED)
        self.assertEqual(
            stop.exception.lines,
            [
                f"REFUSED: the code kept (release {first.name}) is not whole: it runs on no image; rkmon apply "
                "installs a package's code; nothing changed"
            ],
        )
        self.assertEqual((self.state(), self.current()), (before, third))

    def test_back_after_a_new_image_runs_the_release_kept_on_its_own_image(self):
        """An install over a new image keeps the release CURRENT named, installed over the image before it, and
        code-back puts that release back: once the image before runs again (a rollback of the image change), the
        launcher runs it, not that image's own code. The same code over the new image is a release of its own, so that
        the one kept stays as it was."""
        first = self.install("2026.11.19")
        self.new_image("2026.11.17", "img2")
        self.assertEqual(self.which_code(), str(self.image))
        for version in ("2026.11.20", "2026.11.19"):
            with self.subTest(version=version):
                said = self.volume.install(self.pkg(version))
                second = self.current()
                self.assertEqual(
                    said,
                    [
                        f"INSTALLED: release {second.name} ({version}) over this image, in place of release "
                        f"{first.name} (installed over another image), which is kept: rkmon rollback goes back to it"
                    ],
                )
                self.assertNotEqual(second, first)
                self.assertEqual(self.kept(), f"releases/{first.name}")
                self.assertEqual(self.which_code(), str(second))
                self.new_image("2026.11.18", "img1")
                self.assertEqual(self.which_code(), str(self.image))
                self.assertEqual(
                    self.volume.back(),
                    [
                        f"RESTORED: release {first.name} runs again, in place of release {second.name} (installed "
                        "over another image)"
                    ],
                )
                self.assertEqual(self.which_code(), str(first))
                self.assertEqual(code_volume.release_info(first)["base_image"], "img1")
                self.new_image("2026.11.17", "img2")

    def test_an_install_in_place_of_a_release_that_is_not_whole_keeps_the_image_s_code(
        self,
    ):
        """A release CURRENT names that is not whole runs on no image: the image's own code runs. An install in its
        place keeps that code, never the release, so that rollback goes back to code that runs.
        """
        first = self.install("2026.11.19")
        first.chmod(0o755)
        (first / "rkmon.py").unlink()
        first.chmod(code_volume.DIR_MODE)
        self.assertEqual(self.which_code(), str(self.image))
        said = self.volume.install(self.pkg("2026.11.20"))
        second = self.current()
        self.assertEqual(
            said,
            [
                f"INSTALLED: release {second.name} (2026.11.20) over this image, in place of the image's own code, "
                "which is kept: rkmon rollback goes back to it"
            ],
        )
        self.assertEqual(self.kept(), str(self.image))

    def test_refusals_change_nothing(self):
        """An archive that does not read, or that holds what a package does not, a package that is not whole, one that
        needs a new image (its list capped), and an image with no digest: each refused, nothing changed.
        """
        first = self.install("2026.11.19")
        before = self.state()
        not_whole = package(self.base / "part", "2026.11.20")
        (not_whole / "rkmon.py").unlink()
        new_image = package(self.base / "new_image", "2026.11.20")
        for n in range(12):  # what the image's venvs are made from
            write(new_image, f"requirements/r{n:02}.txt", "pytest\n")
        for name, data, code, words in (
            (
                "not a tar",
                io.BytesIO(b"not a tar archive"),
                rk_root.REFUSED,
                "is not a package's tar archive",
            ),
            (
                "absolute",
                archive(self.src, member("/etc/x")),
                rk_root.REFUSED,
                "names /etc/x, outside its folder",
            ),
            (
                "dot dot",
                archive(self.src, member("rocke_monitor/../x")),
                rk_root.REFUSED,
                "outside its folder",
            ),
            (
                "a link",
                archive(self.src, member("rocke_monitor/l", tarfile.SYMTYPE, "/etc")),
                rk_root.REFUSED,
                "a link or a special file",
            ),
            (
                "a fifo",
                archive(self.src, member("rocke_monitor/f", tarfile.FIFOTYPE)),
                rk_root.REFUSED,
                "a link or a special file",
            ),
            (
                "a hard link",
                archive(
                    self.src,
                    member("rocke_monitor/h", tarfile.LNKTYPE, "rocke_monitor/rk"),
                ),
                rk_root.REFUSED,
                "has rocke_monitor/h, a link or a special file",
            ),
            (
                "a device",
                archive(self.src, member("rocke_monitor/d", tarfile.CHRTYPE)),
                rk_root.REFUSED,
                "has rocke_monitor/d, a link or a special file",
            ),
            (
                "twice",
                archive(self.src, member("rocke_monitor/rk")),
                rk_root.REFUSED,
                "the archive names rocke_monitor/rk twice; nothing changed",
            ),
            (
                "a folder in a file's place",
                archive(self.src, member("rocke_monitor/rk", tarfile.DIRTYPE)),
                rk_root.REFUSED,
                "the archive names rocke_monitor/rk twice; nothing changed",
            ),
            (
                "a file under a file",
                archive(self.src, member("rocke_monitor/rk/x/y")),
                rk_root.REFUSED,
                "the archive names rocke_monitor/rk twice; nothing changed",
            ),
            (
                "a release's record",
                archive(self.src, member(f"rocke_monitor/{code_volume.RELEASE_FILE}")),
                rk_root.REFUSED,
                f"the package has {code_volume.RELEASE_FILE}, the record root writes",
            ),
            (
                "not whole",
                archive(not_whole),
                rk_root.REFUSED,
                "not a whole package: rkmon.py missing or empty",
            ),
            (
                "new image",
                archive(new_image),
                rk_root.NEW_IMAGE,
                # its fix is the host's to say, from its one home (releases.new_image_fix)
                "requirements/r09.txt and 2 more differ from this image's; nothing changed",
            ),
        ):
            with self.subTest(name=name):
                with self.assertRaises(rk_root.Stop) as stop:
                    self.volume.install(data)
                self.assertEqual(stop.exception.code, code, stop.exception.lines)
                self.assertIn(words, stop.exception.lines[0])
                self.assertTrue(
                    stop.exception.lines[0].endswith("nothing changed"),
                    stop.exception.lines,
                )
                self.assertEqual(self.state(), before)
        self.idf.write_text("")
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.install(self.pkg("2026.11.20"))
        self.assertEqual(stop.exception.code, rk_root.ERROR)
        self.assertEqual((self.state(), self.current()), (before, first))

    def test_one_change_at_a_time(self):
        """While another change holds the volume's lock, install and back refuse, and change nothing."""
        self.install("2026.11.19")
        before = self.state()
        fd = os.open(self.code / code_volume.LOCK, os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        for change in (
            lambda: self.volume.install(self.pkg("2026.11.20")),
            self.volume.back,
        ):
            with self.assertRaises(rk_root.Stop) as stop:
                change()
            self.assertEqual(stop.exception.code, rk_root.REFUSED)
            self.assertIn(
                "another change of the code runs now", stop.exception.lines[0]
            )
        self.assertEqual(self.state(), before)

    def test_the_start(self):
        """At each start the volume's folder is made root's, what a change that died left goes, and so does each
        release no link names; a volume that fails its check stops the start, with its fix, and CURRENT goes first:
        no release runs while it fails, not even for the health check's rk.
        """
        first = self.install("2026.11.19")
        (self.code / f"{code_volume.STAGE}99").mkdir()
        (self.code / ".current-99").symlink_to("releases/none")
        (self.code / code_volume.RELEASES / "orphan").mkdir()
        self.root.other.add(str(self.code))
        self.assertEqual(self.volume.start(), [f"{self.code} is now root's"])
        self.assertEqual(self.chowned, [(str(self.code), 0, 0)])
        self.assertEqual(
            sorted(os.listdir(self.code)),
            sorted([code_volume.LOCK, "current", "previous", "releases"]),
        )
        self.assertEqual(self.released(), [first.name])
        self.root.other.clear()
        self.assertEqual(self.which_code(), str(first))
        (first / DATA).chmod(0o664)  # a data file another user may write
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.start()
        current = self.code / code_volume.CURRENT
        self.assertEqual(stop.exception.code, rk_root.FAILS)
        self.assertEqual(
            stop.exception.lines,
            [
                f"FAIL: the code volume ({self.code}) is not root's and read-only all the way: {first / DATA} may be "
                f"written by another user (mode 0664); so that no release runs, root removed {current}: the image's "
                "own code runs",
                "  the fix: rkmon apply, on the host, installs this package's code again, as root, and empties the "
                "code volume first; until then the monitor does not start",
            ],
        )
        self.assertFalse(os.path.lexists(current))
        self.assertEqual(self.which_code(), str(self.image))
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.start()
        self.assertTrue(
            stop.exception.lines[0].endswith(
                "; no release runs: the image's own code does"
            ),
            stop.exception.lines,
        )

    def test_a_lock_another_user_made_goes(self):
        """A LOCK that is not root's one file was made while another user could write the volume's folder: the start
        puts a file of root's in its place and checks the volume. A check that cannot be made, whatever stops it,
        fails the start as a failed check does, and no release runs."""
        first = self.install("2026.11.19")
        lock = self.code / code_volume.LOCK
        for name, make in (
            ("a folder", lock.mkdir),
            ("a link", lambda: lock.symlink_to(self.base / "elsewhere")),
            ("a second name of a release's file", lambda: os.link(first / DATA, lock)),
            ("another user's file", lambda: self.root.other.add(str(lock))),
        ):
            with self.subTest(lock=name):
                if name != "another user's file":
                    lock.unlink()
                make()
                self.assertEqual(self.volume.start(), [])
                self.assertEqual(
                    (lock.is_symlink(), lock.is_file(), lock.stat().st_nlink),
                    (False, True, 1),
                )
                self.assertEqual(stat.S_IMODE((first / DATA).stat().st_mode), 0o444)
        lock.unlink()
        lock.mkdir()
        (first / DATA).chmod(0o664)
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.start()
        self.assertEqual(stop.exception.code, rk_root.FAILS)
        self.assertIn(
            "is not root's and read-only all the way", stop.exception.lines[0]
        )
        (first / DATA).chmod(0o444)
        self.volume.link(code_volume.CURRENT, f"releases/{first.name}")
        lock.unlink()
        lock.mkdir()
        remove = rk_root.VOLUME.remove
        with mock.patch.object(
            rk_root.VOLUME,
            "remove",
            lambda p: os.fspath(p) != str(lock) and remove(p),
        ), self.assertRaises(rk_root.Stop) as stop:
            self.volume.start()
        current = self.code / code_volume.CURRENT
        self.assertEqual(
            (stop.exception.code, stop.exception.lines[0]),
            (
                rk_root.FAILS,
                f"FAIL: the code volume ({self.code}) could not be checked: [Errno 21] Is a directory: '{lock}'; so "
                f"that no release runs, root removed {current}: the image's own code runs",
            ),
        )
        self.assertEqual(self.which_code(), str(self.image))

    def test_the_start_checks_the_volume_whatever_the_login_s_move_does(self):
        """rk_root's start checks the code volume whatever the write login's move does: a move that raises is an ERROR
        line and exit 1, and a volume that fails its check exits 5 all the same; the entry point starts the monitor
        on exit 0 alone (tests/test_entrypoint.py)."""
        first = self.install("2026.11.19")

        def move():
            raise PermissionError(13, "Permission denied", "/secrets-write/gh")

        def start() -> tuple:
            out = io.StringIO()
            with mock.patch.object(rk_root.os, "geteuid", lambda: 0), mock.patch.object(
                rk_root, "_owner", lambda: (os.getuid(), os.getgid())
            ), mock.patch.object(
                rk_root, "write_volume", lambda: []
            ), mock.patch.object(
                rk_root, "move_write_login", move
            ), mock.patch.object(
                rk_root, "copies", lambda: ([], False)  # tests/test_root_copies.py
            ), contextlib.redirect_stdout(
                out
            ):
                code = rk_root.main(["start"], volume=self.volume)
            return code, out.getvalue().splitlines()

        moved = "ERROR: the write login's move: [Errno 13] Permission denied: '/secrets-write/gh'"
        self.assertEqual(start(), (rk_root.ERROR, [moved]))
        self.assertEqual(self.current(), first)
        (first / DATA).chmod(0o664)
        code, said = start()
        self.assertEqual((code, said[0]), (rk_root.FAILS, moved))
        self.assertTrue(
            said[1].startswith(f"FAIL: the code volume ({self.code}) is not root's"),
            said,
        )
        self.assertFalse(os.path.lexists(self.code / code_volume.CURRENT))

    def emptied(self, release: pathlib.Path) -> str:
        """What an install says when it empties a volume whose RELEASE holds a script another user owns."""
        job = release / "tools" / "job.py"
        self.root.other.add(str(job))  # a script a job starts, another user's
        return (
            f"the code volume was not root's and read-only all the way ({job} is another user's (uid {OTHER_UID})): "
            "it was emptied first, so the image's own code runs until the release is in place"
        )

    def test_an_install_over_a_volume_that_fails_its_check_empties_it_first(self):
        """Over a code volume that fails its check, the package is unpacked, checked and made a release first, so that
        what is refused changes nothing; the volume is emptied only then, and the lines say so.
        """
        first = self.install("2026.11.19")
        emptied = self.emptied(first)
        before = self.state()
        part = package(self.base / "part", "2026.11.20")
        (part / "rkmon.py").unlink()
        for data in (io.BytesIO(b"not a tar archive"), archive(part)):
            with self.assertRaises(rk_root.Stop) as stop:
                self.volume.install(data)
            self.assertEqual(stop.exception.code, rk_root.REFUSED)
            self.assertTrue(
                stop.exception.lines[-1].endswith("nothing changed"),
                stop.exception.lines,
            )
            self.assertEqual((self.state(), self.current()), (before, first))
        said = self.volume.install(self.pkg("2026.11.20"))
        second = self.current()
        self.assertEqual(
            said,
            [
                emptied,
                f"INSTALLED: release {second.name} (2026.11.20) over this image, in place of the image's own code, "
                "which is kept: rkmon rollback goes back to it",
            ],
        )
        self.assertEqual(self.released(), [second.name])
        self.assertEqual(self.kept(), str(self.image))
        emptied = self.emptied(second)
        self.assertEqual(
            self.volume.install(archive(self.image)),
            [emptied, "NOTHING TO DO: this code runs already (the image's own code)"],
        )
        self.assertEqual(self.state(), ([code_volume.LOCK], [None, None], []))

    def test_an_emptied_volume_keeps_the_move_s_marks_that_pass_its_check(self):
        """The one-time move's marks (docker/rk_root.py copies) stay when an install empties the code volume, while
        they are root's and read-only to every other user, so that no copy is made twice; else they go with the rest.
        """
        emptied = self.emptied(self.install("2026.11.19"))
        marks = self.code / code_volume.MOVED
        marks.mkdir(mode=0o755)
        mark = marks / "writer.teams_webhook"
        mark.touch(mode=0o644)
        self.assertEqual(self.volume.install(self.pkg("2026.11.20"))[0], emptied)
        self.assertEqual(os.listdir(marks), [mark.name])
        self.root.other.add(str(mark))  # a mark another user owns
        self.assertEqual(
            self.volume.install(self.pkg("2026.11.21"))[0],
            f"the code volume was not root's and read-only all the way ({mark} is another user's (uid {OTHER_UID})): "
            "it was emptied first, so the image's own code runs until the release is in place",
        )
        self.assertFalse(os.path.lexists(marks))

    def test_a_step_after_the_emptying_that_fails_says_what_changed(self):
        """A step of an install that fails after the code volume was emptied says so, and not that nothing changed:
        the image's own code runs. So does any error after a change."""
        emptied = self.emptied(self.install("2026.11.19"))

        def code_install(**failing) -> tuple:
            """The install of the next package, each os function in FAILING raising its error: the exit code and the
            lines of the Stop that ends it."""
            data = self.pkg("2026.11.20")
            with contextlib.ExitStack() as stack:
                for name, exc in failing.items():
                    stack.enter_context(
                        mock.patch.object(rk_root.os, name, mock.Mock(side_effect=exc))
                    )
                try:
                    self.volume.install(data)
                except rk_root.Stop as stop:
                    return stop.code, stop.lines
            self.fail("the install did not end short")

        self.assertEqual(
            code_install(rename=OSError(28, "No space left on device")),
            (
                rk_root.ERROR,
                [
                    emptied,
                    "ERROR: the release was not put in place ([Errno 28] No space left on device): the code volume "
                    "was emptied, and the image's own code runs",
                ],
            ),
        )
        self.assertEqual(
            self.state(), ([code_volume.LOCK, "releases"], [None, None], [])
        )
        self.assertEqual(self.which_code(), str(self.image))
        self.root.other.add(str(self.code))
        self.assertEqual(
            code_install(mkdir=PermissionError(13, "Permission denied")),
            (
                rk_root.ERROR,
                [
                    f"{self.code} is now root's",
                    "ERROR: the code volume: [Errno 13] Permission denied",
                ],
            ),
        )
        emptied = self.emptied(self.install("2026.11.19"))
        remove = rk_root.VOLUME.remove
        current = self.code / code_volume.CURRENT
        with mock.patch.object(
            rk_root.VOLUME,
            "remove",
            lambda p: os.fspath(p) != str(current) and remove(p),
        ), self.assertRaises(rk_root.Stop) as stop:
            self.volume.install(self.pkg("2026.11.20"))
        self.assertEqual(
            (stop.exception.code, stop.exception.lines),
            (
                rk_root.ERROR,
                [
                    f"ERROR: {emptied.split(': it was emptied')[0]}, and root could not empty it: {current} could "
                    "not be removed, the rest of it was; the release was not put in place"
                ],
            ),
        )
        self.assertEqual(self.which_code(), str(self.image))

    def test_root_alone_runs_it(self):
        """rk_root.py runs as root only, its one argument an action; code-install reads the archive on stdin."""
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rk_root.main(["code-install", "x"]), rk_root.USAGE)
            with mock.patch.object(rk_root.os, "geteuid", lambda: 1000):
                self.assertEqual(
                    rk_root.main(["code-install"], io.BytesIO()), rk_root.USAGE
                )
        out = io.StringIO()
        with mock.patch.object(
            rk_root.os, "geteuid", lambda: 0
        ), contextlib.redirect_stdout(out):
            code = rk_root.main(
                ["code-install"], self.pkg("2026.11.19"), volume=self.volume
            )
            self.assertEqual(
                (code, self.current().parent), (rk_root.DONE, self.code / "releases")
            )
            self.assertEqual(
                rk_root.main(["code-back"], volume=self.volume), rk_root.DONE
            )
            self.assertEqual(
                rk_root.main(["code-install"], io.BytesIO(b"junk"), volume=self.volume),
                rk_root.REFUSED,
            )
        self.assertIn("INSTALLED: release ", out.getvalue())
        self.assertIn("RESTORED: the image's own code runs again", out.getvalue())


class InUse(Base):
    """A release that neither link names goes only when nothing uses it (code_volume.in_use, rk_root's prune)."""

    def stays(self, release: pathlib.Path) -> str:
        return (
            f"release {release.name} stays: it is in use (a process runs it, or the worker tools name it); the next "
            "start or install removes what nothing uses then"
        )

    def sleeper(self, **kw) -> subprocess.Popen:
        """A process of the tests' own user that lives until the test ends."""
        p = subprocess.Popen(["sleep", "600"], **kw)
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        return p

    def test_a_release_in_use_stays(self):
        """A process that works in a release, or runs it (its PIN), and the worker tools' values file, which names the
        rk each PR run starts, keep it; it goes at the first start or install after nothing uses it.
        """
        first = self.install("2026.11.19")
        self.install("2026.11.20")  # KEPT names the first
        works_in = self.sleeper(cwd=first / "rkmon_lib")
        said = self.volume.install(self.pkg("2026.11.21"))
        self.assertEqual(said[1:], [self.stays(first)])
        self.assertIn(first.name, self.released())
        works_in.kill()
        works_in.wait()
        runs = self.sleeper(
            cwd=self.base, env=dict(os.environ, **{code_volume.PIN: str(first)})
        )
        self.assertEqual(self.volume.start(), [self.stays(first)])
        runs.kill()
        runs.wait()
        write(
            self.values.parent, self.values.name, json.dumps({"rk": str(first / "rk")})
        )
        self.assertEqual(self.volume.start(), [self.stays(first)])
        self.values.unlink()
        self.assertEqual(self.volume.start(), [])
        self.assertNotIn(first.name, self.released())

    def test_no_release_goes_while_a_scan_fails(self):
        """A scan that fails, ends with an error or takes too long removes no release, and says why."""
        self.install("2026.11.19")
        self.install("2026.11.20")
        before = self.released()
        uid = os.getuid()
        for name, scan, why in (
            (
                "an error",
                mock.Mock(
                    return_value=subprocess.CompletedProcess([], 1, "[]", "its words\n")
                ),
                f"the scan as uid {uid} ended with exit 1: its words",
            ),
            (
                "too long",
                mock.Mock(side_effect=subprocess.TimeoutExpired("scan", 60)),
                "Command 'scan' timed out after 60 seconds",
            ),
            (
                "not the names",
                mock.Mock(
                    return_value=subprocess.CompletedProcess([], 0, "not json", "")
                ),
                "Expecting value: line 1 column 1 (char 0)",
            ),
        ):
            with self.subTest(scan=name), mock.patch.object(
                rk_root.subprocess, "run", scan
            ):
                self.assertEqual(
                    self.volume.install(self.pkg("2026.11.21"))[1:],
                    [
                        f"no release was removed: the releases in use could not be found ({why}); the next start "
                        "or install removes what nothing uses then"
                    ],
                )
                self.assertEqual(len(self.released()), 3)
                self.assertTrue(set(before) <= set(self.released()))
                self.volume.back()  # the release before it runs again, and the next install is a change again

    def test_who_scans(self):
        """Root, the monitor's user, which reads the worker tools' values file, and the write user scan their own
        processes, each in a process of its ids; the values file is the worker tools'.
        """
        users = {"rkmon": (1300, 1300), "rkwrite": (1301, 1300)}

        def getpwnam(name):
            uid, gid = users[name]
            return pwd.struct_passwd((name, "x", uid, gid, "", "/", "/bin/sh"))

        with mock.patch.object(rk_root.pwd, "getpwnam", getpwnam):
            self.assertEqual(
                rk_root._scans(),
                [
                    (0, 0, None),
                    (1300, 1300, f"/work/{code_volume.WORKER_VALUES}"),
                    (1301, 1300, None),
                ],
            )
        self.assertEqual(
            (container_layout.MONITOR.name, container_layout.WRITER.name),
            ("rkmon", "rkwrite"),
        )
        self.assertEqual(
            code_volume.WORKER_VALUES,
            f"{paths.UNDER_WORKING['worker_tools']}/{paths.worker_module('pr_user').VALUES_FILE}",
        )
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, '["a"]', ""))
        with mock.patch.object(rk_root.subprocess, "run", run):
            self.assertEqual(rk_root._scan("/code", 1300, 1300, "/v"), {"a"})
        args, kw = run.call_args
        self.assertEqual(
            (args[0][1:], kw["user"], kw["group"], kw["extra_groups"], kw["env"]),
            (
                [
                    "-I",
                    str(ROOT / "rkmon_lib" / "code_volume.py"),
                    "in-use",
                    "/code",
                    "/v",
                ],
                1300,
                1300,
                [],
                {},
            ),
        )

    def test_in_use(self):
        """code_volume.in_use: the releases the processes in PROC work in or run (their PIN), and the one whose rk the
        values file names; a values file that is no regular file, or holds no values, names none.
        """
        proc, releases = self.base / "proc", self.code / code_volume.RELEASES
        for name in ("a/rkmon_lib", "b", "c", "d"):
            (releases / name).mkdir(parents=True)
        (self.code / "releases-other" / "x").mkdir(parents=True)
        pin = code_volume.PIN.encode()
        for pid, cwd, environ in (
            ("1", releases / "a" / "rkmon_lib", b""),
            (
                "2",
                self.base,
                b"A=1\0" + pin + b"=" + os.fsencode(releases / "b") + b"\0",
            ),
            ("3", None, pin + b"=releases/d\0"),  # a PIN that is no path
            ("4", self.code / "releases-other" / "x", b""),  # beside the releases
            ("self", releases / "d", b""),  # no process's number
        ):
            (proc / pid).mkdir(parents=True)
            if cwd is not None:
                (proc / pid / "cwd").symlink_to(cwd)
            (proc / pid / "environ").write_bytes(environ)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.code)  # where the PIN that is no path would name release d
        self.assertEqual(code_volume.in_use(self.code, proc=proc), {"a", "b"})
        values = self.base / "values.json"
        values.write_text(json.dumps({"rk": str(releases / "c" / "rk")}))
        self.assertEqual(code_volume.in_use(self.code, values, proc), {"a", "b", "c"})
        for text in ("[1]", "not the values", '{"rk": 3}', "{}"):
            values.write_text(text)
            self.assertEqual(code_volume.in_use(self.code, values, proc), {"a", "b"})
        values.unlink()
        os.mkfifo(values)  # no writer: a read would wait for one
        self.assertEqual(code_volume.in_use(self.code, values, proc), {"a", "b"})
        with self.assertRaises(OSError):
            code_volume.in_use(self.code, proc=self.base / "no_proc")
        cli = [sys.executable, "-I", str(ROOT / "rkmon_lib" / "code_volume.py")]
        p = subprocess.run(
            [*cli, "in-use", str(self.code)], capture_output=True, text=True, timeout=60
        )
        self.assertEqual((p.returncode, p.stdout), (0, "[]\n"), p.stderr)
        p = subprocess.run([*cli, "in-use"], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2)


class TheDoctorsRow(Base):
    """The doctor's row of the code volume (doctor.code_volume_check): the start's check, as the monitor's user reads
    the volume, and the fix the start's FAIL line gives."""

    def row(self, wrapper: str = "rkmon") -> checks.Check:
        env = {"RKMON_CODE_DIR": str(self.code), "RKMON_WRAPPER": wrapper}
        with mock.patch.object(releases, "IMAGE_CODE", self.image):
            return doctor.code_volume_check(env, lstat=self.root)

    def test_pass_or_fail_with_the_start_s_fix(self):
        """PASS while the volume is root's and read-only all the way, what a change of the code leaves while it runs
        aside: its STAGE folder, root's alone to read, and its links. FAIL as the start fails, with the start's fix.
        """
        first = self.install("2026.11.19")
        passed = checks.Check(
            "code volume", "PASS", f"{self.code} is root's and read-only all the way"
        )
        self.assertEqual(self.row(), passed)
        stage = self.code / f"{code_volume.STAGE}{OTHER_UID}"
        write(stage, DATA, "{}\n", 0o666)
        link = self.code / f".{code_volume.CURRENT}-{OTHER_UID}"
        link.symlink_to(self.base / "elsewhere")
        self.assertEqual(
            sorted(self.volume.problems()),
            [
                f"{link} is a link that leaves the code (to {self.base / 'elsewhere'})",
                f"{stage / DATA} may be written by another user (mode 0666)",
            ],
        )
        self.assertEqual(self.row(), passed)
        (first / DATA).chmod(0o664)  # a data file another user may write
        row = self.row()
        with self.assertRaises(rk_root.Stop) as stop:
            self.volume.start()
        said = f"{first / DATA} may be written by another user (mode 0664)"
        self.assertEqual(
            (row.status, row.found),
            (
                checks.Status.FAIL,
                f"{self.code} is not root's and read-only all the way: {said}; it blocks the monitor's start",
            ),
        )
        current = self.code / code_volume.CURRENT
        self.assertEqual(
            stop.exception.lines,
            [
                f"FAIL: the code volume ({self.code}) is not root's and read-only all the way: {said}; so that no "
                f"release runs, root removed {current}: the image's own code runs",
                f"  the fix: {row.fix}; until then the monitor does not start",
            ],
        )
        self.assertEqual(
            str(row.fix),
            "rkmon apply, on the host, installs this package's code again, as root, and empties the code volume "
            "first",
        )
        self.assertEqual(self.row("./rkmon").fix.commands, ("./rkmon apply",))

    def test_the_doctor_shows_it_in_the_container(self):
        """In the container the doctor says which code runs and shows the code volume's row, whether a Claude session
        runs or not; a host has no code volume."""
        rows = {}
        for box in ("1", ""):
            env = {"RKMON_CONTAINER": box, "RKMON_CODE_DIR": str(self.code)}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                features, "uses_claude", return_value=False
            ), mock.patch.object(releases, "IMAGE_CODE", self.image):
                rows[box] = [c.name for c in doctor.container_checks({})]
        self.assertEqual(
            [n for n in rows["1"] if n.startswith("code")], ["code", "code volume"]
        )
        self.assertEqual([n for n in rows[""] if n.startswith("code")], [])


class Broker(Base):
    """The broker's pick and its check of the code, with the code volume and the image's digest of the tests."""

    def setUp(self):
        super().setUp()
        for name, value in (
            ("CODE_DIR", str(self.code)),
            ("IMAGE_ID_FILE", str(self.idf)),
            ("CODE", str(self.image)),
        ):
            p = mock.patch.object(rk_write, name, value)
            p.start()
            self.addCleanup(p.stop)

    def refused(self, **kw) -> tuple:
        """(the broker's exit code, what it said, whether it ran anything) for `rk post 7 f.md` as the write user."""
        ran, out = [], io.StringIO()
        me = mock.Mock(pw_name=rk_write.WRITE_USER, pw_dir=str(self.base / "home"))
        with mock.patch.object(
            rk_write.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(
            rk_write, "log", lambda line: None
        ), contextlib.redirect_stdout(
            out
        ):
            code = rk_write.main(
                ["post", "7", "f.md"],
                run=lambda *a: ran.append(a) or 0,
                lstat=self.root,
                **kw,
            )
        return code, " ".join(out.getvalue().split()), bool(ran)

    def test_it_picks_current_by_the_launcher_s_rule_and_never_by_a_caller(self):
        self.assertEqual(rk_write.picked(), str(self.image))
        first = self.install("2026.11.19")
        caller = {
            "RKMON_CODE": str(self.image),
            "RKMON_CODE_DIR": "/elsewhere",
            "RKMON_IMAGE_CODE": "/elsewhere",
            "RKMON_IMAGE_ID_FILE": "/elsewhere",
            "PYTHONPATH": "/elsewhere",
        }
        with mock.patch.dict(os.environ, caller):
            self.assertEqual(rk_write.picked(), str(first))
        self.assertEqual(code_volume.running(self.code, "img1"), first)
        self.idf.write_text("img2\n")  # a new image: its own code runs
        self.assertEqual(rk_write.picked(), str(self.image))
        self.assertIsNone(code_volume.running(self.code, "img2"))
        self.idf.write_text("img1\n")
        code_volume.remove(
            first
        )  # not whole: the image's code runs, as the launcher's rule says
        self.assertEqual(rk_write.picked(), str(self.image))
        self.assertIsNone(code_volume.running(self.code, "img1"))

    def test_the_check_of_the_whole_tree(self):
        """Each folder from / down, and each file and folder of the tree, its data and the scripts a job starts by
        their paths among them: root's and no other user's to write; no link that leaves the tree.
        """
        tree = self.install("2026.11.19")
        self.assertEqual(rk_write.problems(str(tree), self.root), [])
        for path, change, said in (
            (
                tree / DATA,
                lambda p: self.root.other.add(str(p)),
                f"is another user's (uid {OTHER_UID})",
            ),
            (
                tree / DATA,
                lambda p: p.chmod(0o664),
                "may be written by another user (mode 0664)",
            ),
            (
                tree / "tools" / "job.py",
                lambda p: p.chmod(0o557),
                "may be written by another user (mode 0557)",
            ),
            (
                tree / "tools",
                lambda p: p.chmod(0o575),
                "may be written by another user (mode 0575)",
            ),
            (
                self.base / "code",
                lambda p: p.chmod(0o775),
                "may be written by another user (mode 0775)",
            ),
        ):
            with self.subTest(path=path, said=said):
                before = stat.S_IMODE(os.lstat(path).st_mode)
                change(path)
                self.assertEqual(
                    rk_write.problems(str(tree), self.root), [f"{path} {said}"]
                )
                path.chmod(before)
                self.root.other.clear()
        os.chmod(tree / "rkmon_lib", 0o755)
        (tree / "rkmon_lib" / "out").symlink_to("/etc")
        (tree / "rkmon_lib" / "in").symlink_to("store.py")
        os.mkfifo(tree / "rkmon_lib" / "fifo")
        self.assertEqual(
            rk_write.problems(str(tree), self.root),
            [
                f"{tree / 'rkmon_lib' / 'fifo'} is neither a file nor a folder",
                f"{tree / 'rkmon_lib' / 'out'} is a link that leaves the code (to /etc)",
            ],
        )

    def test_it_refuses_code_another_user_could_change(self):
        """The broker runs nothing while the code it would run fails its check, or the code volume's links are another
        user's, and says what is wrong (the first MOST things, then how many more) and the fix.
        """
        tree = self.install("2026.11.19")
        self.assertEqual(self.refused()[0], 0)
        os.chmod(tree / "rkmon_lib", 0o755)
        for n in range(12):
            self.root.other.add(str(write(tree, f"rkmon_lib/x{n:02}.py", "#\n")))
        os.chmod(tree / "rkmon_lib", 0o555)
        code, said, ran = self.refused()
        self.assertEqual((code, ran), (4, False))
        self.assertTrue(
            said.startswith(
                f"REFUSED: the writer: the code it would run ({tree}) is not root's"
            ),
            said,
        )
        self.assertIn(
            f"x09.py is another user's (uid {OTHER_UID}) and 2 more; the fix: rkmon apply",
            said,
        )
        self.root.other.clear()
        self.root.other.add(str(self.code / code_volume.KEPT))
        self.assertEqual(self.refused()[0], 4)
        self.root.other.clear()
        self.root.other.add(str(self.code / code_volume.CURRENT))
        self.assertEqual(self.refused()[0], 4)

    def test_it_passes_on_the_wrapper_s_name_checked_and_runs_the_same_command(self):
        """The name the host's wrapper was run by reaches the writer, which its fixes show, while it is a path that
        ends in rkmon (container_layout.shown_wrapper), else as rkmon: unset, with another character, of another
        name, too long. It is shown only: the broker runs the same command with any."""
        tree = self.install("2026.11.19")
        me = mock.Mock(pw_name=rk_write.WRITE_USER, pw_dir=str(self.base / "home"))
        for given, shown in (
            (None, "rkmon"),
            ("./rkmon", "./rkmon"),
            ("/opt/pkg/docker/rkmon", "/opt/pkg/docker/rkmon"),
            ("./rkmon; id", "rkmon"),
            ("$(id)/rkmon", "rkmon"),
            ("./rkmon-team", "rkmon"),
            ("a" * rk_write.LAYOUT.WRAPPER_MAX + "/rkmon", "rkmon"),
        ):
            ran = []
            with self.subTest(given=given), mock.patch.object(
                rk_write.pwd, "getpwuid", lambda uid: me
            ), mock.patch.object(rk_write, "log", lambda line: None):
                code = rk_write.main(
                    ["post", "7", "f.md"],
                    run=lambda *a: ran.append(a) or 0,
                    environ={} if given is None else {"RKMON_WRAPPER": given},
                    lstat=self.root,
                )
                [(argv, env, _copies)] = ran
                self.assertEqual(code, 0)
                self.assertEqual(env["RKMON_WRAPPER"], shown)
                self.assertEqual(
                    argv,
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-c",
                        rk_write.BOOT,
                        str(tree),
                        "post",
                        "7",
                        "f.md",
                    ],
                )

    def test_it_runs_current_s_writer_and_its_data(self):
        """End to end, with the tests' user as the writer: the broker runs CURRENT's release, its own writer
        (rkmon_lib/writer_run.py) and its rkmon.py, which reads the release's data file; of the caller's environment,
        only sudo's word of who called and the wrapper's name, checked, reach the writer, and RKMON_CODE and the like
        never do."""
        tree = self.install("2026.11.19")
        os.chmod(tree, 0o755)
        for folder in ("rkmon_lib", "worker"):
            os.chmod(tree / folder, 0o755) if (tree / folder).exists() else None
        for src in [
            *(ROOT / "rkmon_lib").glob("*.py"),
            *(ROOT / "worker").glob("*.py"),
        ]:
            rel = src.relative_to(ROOT).as_posix()
            if not (tree / rel).exists():
                (tree / rel).parent.mkdir(exist_ok=True)
                shutil.copy2(src, tree / rel)
        ran = []

        def run(argv, env, cwd):
            p = subprocess.run(
                argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=60
            )
            ran.append((argv, env, p))
            return p.returncode

        me = mock.Mock(pw_name=rk_write.WRITE_USER, pw_dir=str(self.base / "home"))
        caller = {
            "SUDO_UID": str(os.getuid()),
            "SUDO_GID": str(os.getgid()),
            "SUDO_USER": "rkmon",
            "RKMON_WRAPPER": "./rkmon",
            "RKMON_CODE": str(self.image),
            "PYTHONPATH": str(self.image),
            "GH_TOKEN": "x",
        }
        with mock.patch.object(
            rk_write.pwd, "getpwuid", lambda uid: me
        ), mock.patch.object(rk_write, "log", lambda line: None):
            code = rk_write.main(
                ["post", "7", "f.md", "--dry-run"],
                run=run,
                environ=caller,
                lstat=self.root,
            )
        [(argv, env, p)] = ran
        self.assertEqual(code, 0, p.stderr)
        self.assertEqual(
            argv,
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                rk_write.BOOT,
                str(tree),
                "post",
                "7",
                "f.md",
                "--dry-run",
            ],
        )
        self.assertEqual(
            env,
            {
                "PATH": rk_write.PATH,
                "LANG": "C.UTF-8",
                "HOME": me.pw_dir,
                "USER": rk_write.WRITE_USER,
                **{k: caller[k] for k in rk_write.SUDO_VARS},
                "RKMON_WRAPPER": "./rkmon",
            },
        )
        self.assertEqual(
            p.stdout.split(),
            [
                str(tree),
                '{"version":',
                '"2026.11.19"}',
                "post",
                "7",
                "f.md",
                "--dry-run",
            ],
        )


class Remove(unittest.TestCase):
    def test_root_changes_no_mode_on_the_way(self):
        """Root removes a tree without a chmod: a folder of it that another user swaps for a link between a check and a
        chmod would carry the mode to the link's target. The link goes, and what it named stays as it was.
        """
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        base = pathlib.Path(d.name)
        victim = write(base / "victim", "f", "x\n").parent
        victim.chmod(0o500)
        self.addCleanup(victim.chmod, 0o700)
        sub = write(base / "tree" / "sub", "g", "y\n").parent
        islink = os.path.islink

        def swapped(path):
            """os.path.islink, with SUB swapped for a link to VICTIM the first time it is asked about."""
            if os.fspath(path) == str(sub) and not islink(sub):
                shutil.rmtree(sub)
                sub.symlink_to(victim)
                return False
            return islink(path)

        with mock.patch.object(code_volume.os, "geteuid", lambda: 0):
            with mock.patch.object(code_volume.os.path, "islink", swapped):
                self.assertTrue(code_volume.remove(base / "tree"))
        self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o500)
        self.assertEqual((victim / "f").read_text(), "x\n")


class WhereTheCodeIs(unittest.TestCase):
    def test_the_broker_s_words_are_the_writer_s(self):
        """The broker's one copy of the package's words, the writer's refusal line (rkmon_lib/writer.py loads more
        than the standard library, so the broker cannot load it), is the writer's; it logs to the writer's log.
        """
        self.assertEqual(rk_write.REFUSAL.format("why"), writer.refusal("why"))
        self.assertEqual(rk_write.LOG, writer_run.LOG)

    def test_the_image_s_code_and_the_code_volume(self):
        """The broker and the root steps run from the image's own code, and take the code volume's names from the
        layout (tests/test_container_layout.py: another layout moves them); the writer's run runs the tree it is in;
        the image's build writes the code's digest where the layout says."""
        self.assertEqual((rk_write.CODE, rk_root.CODE), (str(ROOT), str(ROOT)))
        self.assertEqual(writer_run.TREE, str(ROOT))
        volume = rk_root.Volume()
        self.assertEqual(
            (volume.folder, volume.image, volume.id_file),
            (container_layout.CODE_DIR, str(ROOT), container_layout.IMAGE_ID_FILE),
        )
        self.assertEqual(
            (str(releases.code_dir({})), str(releases.IMAGE_ID_FILE)),
            (container_layout.CODE_DIR, container_layout.IMAGE_ID_FILE),
        )
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"> {container_layout.IMAGE_ID_FILE}", dockerfile)


if __name__ == "__main__":
    unittest.main()
