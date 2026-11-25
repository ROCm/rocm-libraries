# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Private files (rkmon_lib/private_files.py): the one check of a file that may hold a private value (a regular file,
not a link, yours, permissions 600, outside the package folder, git work trees and cloud-synced folders, links
resolved), the one way to write one (a new file at 600, synced, put in place, never over a file unless told), the
package's own private files at their places in the package folder and their carry to another package folder, the
place rule the monitor's folders follow, and rk doctor's rows of the secrets folder and its files.
"""

import errno
import fnmatch
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

from rendered import one_line  # noqa: E402
from rkmon_lib import doctor  # noqa: E402
from rkmon_lib import private_files as pf  # noqa: E402
from rkmon_lib.checks import Check, Command, Status, Steps, render  # noqa: E402

PASSED = "is yours only, outside the package, git and cloud sync"
SYNCED = "its sync service copies it off this machine"
# The command a caller ran, naming its file: a fix that names another file runs it again with that one.
RETRY = pf.again(("rkmon", "check", "--settings", "given.yaml"), "given.yaml")


def quoted(path) -> str:
    """PATH as a fix's command shows it, quoted where the shell needs it (checks.Command): a test's folder is
    in TMPDIR, whose path may hold a space."""
    return shlex.quote(str(path))


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(os.path.realpath(self.tmp.name))
        self.pkg = self.d / "pkg"
        (self.pkg / "docker").mkdir(parents=True)
        home = mock.patch.dict(os.environ, {"HOME": str(self.d / "home")})
        home.start()  # a fix's suggested ~/NAME is free in this home folder
        self.addCleanup(home.stop)

    def file(self, rel: str, mode: int = 0o600, text: str = "a: 1\n") -> pathlib.Path:
        p = self.d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        p.chmod(mode)
        return p

    def repo(self, rel: str = "repo") -> pathlib.Path:
        top = self.d / rel
        top.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(top)], check=True)
        return top

    def failed(self, path, name: str = "settings file") -> list:
        rows = pf.problems(path, self.pkg, name)
        self.assertTrue(rows, path)
        self.assertEqual({r.status for r in rows}, {Status.FAIL}, rows)
        self.assertEqual({r.name for r in rows}, {name})
        return rows


class TestProblems(Case):
    def test_a_file_of_yours_at_600_outside_the_three_folders_passes(self):
        p = self.file("s.yaml")
        self.assertEqual(
            pf.problems(p, self.pkg, "settings file"),
            [Check("settings file", Status.PASS, f"{p} {PASSED}")],
        )
        self.assertEqual(pf.problems(p, self.pkg)[0].name, "private file")
        for mode in (0o400, 0o700):  # nothing for others, readable by you
            p.chmod(mode)
            self.assertEqual([r.status for r in pf.problems(p, self.pkg)], ["PASS"])

    def test_others_may_not_read_it_and_you_must(self):
        p = self.file("s.yaml")
        for mode, words in (
            (0o644, "others on this machine can read it"),
            (0o640, "others on this machine can read it"),
            (0o604, "others on this machine can read it"),
            (0o620, "others on this machine have access to it"),
            (0o601, "others on this machine have access to it"),
            (0o200, "you cannot read it"),
            (0o000, "you cannot read it"),
            (0o244, "others on this machine can read it and you cannot read it"),
        ):
            with self.subTest(mode=oct(mode)):
                p.chmod(mode)
                self.assertEqual(
                    self.failed(p),
                    [
                        Check(
                            "settings file",
                            Status.FAIL,
                            f"{p} has permissions {mode:03o}: {words}",
                            Command(("chmod", "600", p)),
                        )
                    ],
                )
        self.assertEqual(str(self.failed(p)[0].fix), f"chmod 600 {quoted(p)}")

    def test_it_is_yours(self):
        p = self.file("s.yaml")
        with mock.patch.object(pf.os, "getuid", return_value=os.getuid() + 1):
            (row,) = self.failed(p)
        self.assertEqual(
            row.found, f"{p} belongs to {pf._user(os.getuid())}, not to you"
        )
        self.assertEqual(row.fix.argv[:2], ("sudo", "chown"))
        self.assertEqual(row.fix.argv[2], pf._user(os.getuid() + 1))
        self.assertEqual(row.fix.argv[3:], (str(p),))

    def test_no_file_a_link_or_no_regular_file_is_one_row(self):
        none = self.d / "none.yaml"
        self.assertEqual(
            self.failed(none),
            [
                Check(
                    "settings file",
                    Status.FAIL,
                    f"no file at {none}",
                    Steps(
                        (
                            Command(("ls", "-la", self.d)),
                            "shows what is there: name the file's full path",
                        )
                    ),
                )
            ],
        )
        p = self.file("s.yaml", 0o644)
        link = self.d / "link.yaml"
        link.symlink_to(p)
        self.assertEqual(
            self.failed(link),
            [
                Check(
                    "settings file",
                    Status.FAIL,
                    f"{link} is a link, not a file",
                    Steps((Command(("rm", link)), ", then", Command(("cp", p, link)))),
                )
            ],
        )
        fifo = self.d / "fifo"
        os.mkfifo(fifo)
        for odd in (self.d / "pkg", fifo):
            with self.subTest(odd=odd.name):
                (row,) = self.failed(odd)
                self.assertEqual(row.found, f"{odd} is not a regular file")
                self.assertEqual(
                    str(row.fix),
                    f"ls -la {quoted(odd)} shows what is there: name a regular file's full path",
                )

    def test_a_fix_that_names_another_file_is_the_command_you_ran(self):
        none = self.d / "none.yaml"
        (row,) = pf.problems(none, self.pkg, retry=RETRY)
        self.assertEqual(
            row.fix, Steps((RETRY(pf.FILE), ", FILE being the file's full path"))
        )
        self.assertEqual(
            str(row.fix),
            "rkmon check --settings FILE, FILE being the file's full path",
        )
        p = self.file("s.yaml")
        link = self.d / "link.yaml"
        link.symlink_to(p)
        (row,) = pf.problems(link, self.pkg, retry=RETRY)
        self.assertEqual(
            str(row.fix),
            f"name the file it points to: rkmon check --settings {quoted(p)}",
        )
        gone = self.d / "gone.yaml"  # a link to nothing: removed, or another file named
        gone.symlink_to(self.d / "nothing.yaml")
        self.assertEqual(pf.problems(gone, self.pkg)[0].fix, Command(("rm", gone)))
        self.assertEqual(
            str(pf.problems(gone, self.pkg, retry=RETRY)[0].fix),
            "rkmon check --settings FILE, FILE being the file's full path",
        )
        (row,) = pf.problems(self.pkg, self.pkg, retry=RETRY)
        self.assertEqual(
            str(row.fix),
            "rkmon check --settings FILE, FILE being a regular file's full path",
        )

    def test_again_runs_your_command_with_another_file(self):
        for argv, given, want in (
            (
                ("rkmon", "build", "--settings", "a.yaml", "--no-cache"),
                "a.yaml",
                ("rkmon", "build", "--settings", "~/b.yaml", "--no-cache"),
            ),
            (
                ("rkmon", "build", "--settings=a.yaml"),
                "a.yaml",
                ("rkmon", "build", "--settings=~/b.yaml"),
            ),
            (
                ("rkmon", "settings", "a.yaml"),
                "a.yaml",
                ("rkmon", "settings", "~/b.yaml"),
            ),
            (  # no file named: --settings and the file after the command
                ("rkmon", "check"),
                "",
                ("rkmon", "check", "--settings", "~/b.yaml"),
            ),
            (
                ("a.yaml",),
                "a.yaml",
                ("a.yaml", "--settings", "~/b.yaml"),
            ),  # never the program
        ):
            with self.subTest(argv=argv):
                self.assertEqual(pf.again(argv, given)("~/b.yaml"), Command(want))
        self.assertEqual(
            str(
                pf.again(("migrate", "--out", "o.yaml"), "o.yaml", "--out")("~/b.yaml")
            ),
            "migrate --out ~/b.yaml",
        )
        self.assertEqual(
            str(pf.again(("x", "a b.yaml"), "a b.yaml")("~/c d.yaml")),
            "x ~/'c d.yaml'",
        )

    def test_outside_the_package_folder(self):
        p = self.file("pkg/docker/my-settings.yaml")
        (row,) = self.failed(p)
        self.assertEqual(
            row.found,
            f"{p} is inside the package folder: the image's build context and a commit would carry it",
        )
        move = Command(("mv", p, "~/my-settings.yaml"))
        self.assertEqual(row.fix, Steps((move, ", then name it by its new path")))
        self.assertEqual(
            str(row.fix),
            f"mv {quoted(p)} ~/my-settings.yaml, then name it by its new path",
        )
        (named,) = pf.problems(p, self.pkg, retry=RETRY)
        self.assertEqual(
            named.fix, Steps((move, ", then", RETRY("~/my-settings.yaml")))
        )
        self.assertEqual(
            str(named.fix),
            f"mv {quoted(p)} ~/my-settings.yaml, then rkmon check --settings ~/my-settings.yaml",
        )
        # The home folder has one of that name: the move takes a free one, over no file.
        self.file("home/my-settings.yaml")
        self.assertEqual(
            self.failed(p)[0].fix.parts[0], Command(("mv", p, "~/my-settings-2.yaml"))
        )
        # A package folder in a git work tree: the one row says the package (one move fixes both).
        subprocess.run(["git", "init", "-q", str(self.pkg)], check=True)
        self.assertEqual(
            [r.found for r in self.failed(p)],
            [row.found],
        )

    def test_outside_git(self):
        top = self.repo()
        p = self.file("repo/deep/s.yaml")
        self.assertEqual(
            [r.found for r in self.failed(p)],
            [f"{p} is inside the git work tree {top}: a commit there would publish it"],
        )
        worktree = self.d / "worktree"  # a linked worktree's .git is a file
        worktree.mkdir()
        (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
        q = self.file("worktree/s.yaml")
        self.assertIn(f"inside the git work tree {worktree}:", self.failed(q)[0].found)

    def test_outside_cloud_synced_folders(self):
        for folder in (
            "OneDrive",
            "OneDrive" + " - Example Org",  # split: the privacy check flags the name
            "Dropbox",
            "Dropbox (Personal)",
            "iCloudDrive",
            "iCloud Drive",
            "Google Drive",
            "GoogleDrive-account",
            "My Drive",
            "Box",
            "Box Sync",
            "Box-Box",
            "CloudStorage",
            "Mobile Documents",
        ):
            with self.subTest(folder=folder):
                p = self.file(f"home/{folder}/sub/s.yaml")
                self.assertEqual(
                    [r.found for r in self.failed(p)],
                    [
                        f"{p} is inside the cloud-synced folder {self.d / 'home' / folder}: "
                        + SYNCED
                    ],
                )
        for folder in ("toolbox", "outbox", "my-dropbox", "drive", "notes"):
            with self.subTest(folder=folder):
                p = self.file(f"home/{folder}/s.yaml")
                self.assertEqual([r.status for r in pf.problems(p, self.pkg)], ["PASS"])

    def test_a_link_cannot_hide_where_a_file_lies(self):
        self.file("Dropbox/s.yaml")
        (self.d / "plain").symlink_to(self.d / "Dropbox")
        self.assertIn(
            f"inside the cloud-synced folder {self.d / 'Dropbox'}:",
            self.failed(self.d / "plain" / "s.yaml")[0].found,
        )
        self.file("pkg/docker/s.yaml")
        (self.d / "elsewhere").symlink_to(self.pkg / "docker")
        self.assertIn(
            "is inside the package folder",
            self.failed(self.d / "elsewhere" / "s.yaml")[0].found,
        )

    def test_every_rule_it_breaks_is_a_row(self):
        self.repo()
        p = self.file("repo/s.yaml", 0o644)
        with mock.patch.object(pf.os, "getuid", return_value=os.getuid() + 1):
            rows = self.failed(p)
        self.assertEqual(
            [r.found.removeprefix(f"{p} ").split(" ", 1)[0] for r in rows],
            ["belongs", "has", "is"],
        )

    def test_a_path_with_a_home_folder(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.d)}):
            p = self.file("s.yaml")
            self.assertEqual(
                pf.problems("~/s.yaml", self.pkg)[0].found, f"{p} {PASSED}"
            )

    def test_a_row_shows_the_file_as_the_caller_shows_it(self):
        # SHOWN, as the command's own lines show the file: the file, its folder and the paths beside it; any other
        # path absolute, and every one without SHOWN.
        p = self.file("home/s.yaml", 0o644)
        (row,) = pf.problems(p, self.pkg, shown="~/s.yaml")
        self.assertEqual(
            (row.found, str(row.fix)),
            (
                "~/s.yaml has permissions 644: others on this machine can read it",
                "chmod 600 ~/s.yaml",
            ),
        )
        with self.assertRaises(pf.Refused) as cm:
            pf.checked(p, self.pkg, shown="~/s.yaml")
        self.assertEqual(cm.exception.rows, [row])
        self.assertTrue(pf.problems(p, self.pkg)[0].found.startswith(f"{p} has "))
        p.chmod(0o600)
        self.assertEqual(
            pf.problems(p, self.pkg, shown="~/s.yaml")[0].found, f"~/s.yaml {PASSED}"
        )
        none = self.d / "home" / "none.yaml"
        (row,) = pf.problems(none, self.pkg, shown="~/none.yaml")
        self.assertEqual(
            (row.found, str(row.fix)),
            (
                "no file at ~/none.yaml",
                "ls -la ~/ shows what is there: name the file's full path",
            ),
        )
        link = self.d / "home" / "link.yaml"
        link.symlink_to(p)
        (row,) = pf.problems(link, self.pkg, shown="~/link.yaml")
        self.assertEqual(
            (row.found, str(row.fix)),
            (
                "~/link.yaml is a link, not a file",
                f"rm ~/link.yaml, then cp {quoted(p)} ~/link.yaml",
            ),
        )
        synced = self.file("home/Box/s.yaml")
        (row,) = pf.problems(synced, self.pkg, retry=RETRY, shown="~/Box/s.yaml")
        self.assertEqual(
            (row.found, str(row.fix)),
            (
                f"~/Box/s.yaml is inside the cloud-synced folder {self.d / 'home' / 'Box'}: {SYNCED}",
                "mv ~/Box/s.yaml ~/s-2.yaml, then rkmon check --settings ~/s-2.yaml",
            ),
        )


class TestPlaces(Case):
    def test_the_place_rule_for_files_and_folders(self):
        top = self.repo()
        cases = {
            self.d / "work": "",
            self.d / "work" / "not-yet": "",
            top: f"it is inside the git work tree {top}: a commit there would publish it",
            top
            / "work"
            / "not-yet": f"it is inside the git work tree {top}: a commit there would publish it",
            self.d
            / "Dropbox"
            / "work": f"it is inside the cloud-synced folder {self.d / 'Dropbox'}: "
            + SYNCED,
            self.pkg: "it is inside the package folder: the image's build context and a commit would carry it",
            self.pkg
            / "state": "it is inside the package folder: the image's build context and a commit would "
            "carry it",
        }
        for path, why in cases.items():
            with self.subTest(path=str(path)):
                self.assertEqual(pf.place_problem(path, self.pkg), why)

    def test_git_top(self):
        top = self.repo()
        self.assertEqual(pf.git_top(top), str(top))
        self.assertEqual(pf.git_top(top / "a" / "b.txt"), str(top))
        self.assertEqual(pf.git_top(self.d / "a" / "b.txt"), "")
        (self.d / "into-repo").symlink_to(top)
        self.assertEqual(pf.git_top(self.d / "into-repo" / "b.txt"), str(top))


class TestChecked(Case):
    def test_a_file_that_breaks_no_rule_comes_back_absolute(self):
        p = self.file("s.yaml")
        self.assertEqual(pf.checked(str(p), self.pkg, "settings file"), p)
        with mock.patch.dict(os.environ, {"HOME": str(self.d)}):
            self.assertEqual(pf.checked("~/s.yaml", self.pkg), p)

    def test_a_file_that_breaks_a_rule_is_refused_with_each_fail_row(self):
        self.repo()
        for path, rules in (
            (self.file("open.yaml", 0o644), 1),  # its permissions
            (self.file("repo/s.yaml", 0o644), 2),  # its permissions and its place
            (self.d / "none.yaml", 1),  # no file
        ):
            with self.subTest(path=path.name):
                with self.assertRaises(pf.Refused) as cm:
                    pf.checked(path, self.pkg, "settings file", retry=RETRY)
                self.assertIsInstance(cm.exception, ValueError)
                self.assertEqual(
                    cm.exception.rows,
                    pf.problems(path, self.pkg, "settings file", retry=RETRY),
                )
                self.assertEqual(
                    [r.status for r in cm.exception.rows], [Status.FAIL] * rules
                )

    def test_a_refusal_is_its_rows_then_the_rules_line(self):
        p = self.file("s.yaml", 0o644)
        with self.assertRaises(pf.Refused) as cm:
            pf.checked(p, self.pkg, "settings file")
        shown = str(cm.exception)
        self.assertEqual(shown, render(cm.exception.rows, "problems") + "\n" + pf.RULES)
        self.assertTrue(
            one_line(shown).startswith(
                f"FAIL settings file: {p} has permissions 644: "
            ),
            shown,
        )
        self.assertTrue(
            one_line(shown).endswith(f"fix: chmod 600 {quoted(p)} {pf.RULES}"), shown
        )

    def test_the_rules_line_ends_a_refusal_once_whatever_its_rows(self):
        """Where README.md says what a private file is and where it may lie, its section "Where things live":
        one line after the rows, however many rules the file breaks."""
        self.repo()
        p = self.file("repo/s.yaml", 0o644)  # its permissions and its place
        with self.assertRaises(pf.Refused) as cm:
            pf.checked(p, self.pkg, "settings file")
        self.assertEqual(len(cm.exception.rows), 2)
        lines = str(cm.exception).splitlines()
        self.assertEqual((lines[-1], lines.count(pf.RULES)), (pf.RULES, 1))
        self.assertTrue(pf.RULES.endswith(' README.md, "Where things live".'), pf.RULES)


class TestTighten(Case):
    def test_permissions_that_break_the_rule_are_set_to_600(self):
        for mode in (0o644, 0o640, 0o604, 0o620, 0o601, 0o777, 0o200, 0o000, 0o244):
            with self.subTest(mode=oct(mode)):
                p = self.file(f"s-{mode:03o}.yaml", mode)
                self.assertTrue(pf.tighten(p))
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
                self.assertEqual(p.read_text(encoding="utf-8"), "a: 1\n")
                self.assertEqual([r.status for r in pf.problems(p, self.pkg)], ["PASS"])

    def test_permissions_the_rule_allows_stay(self):
        for mode in (0o600, 0o400, 0o700):
            with self.subTest(mode=oct(mode)):
                p = self.file(f"s-{mode:03o}.yaml", mode)
                self.assertFalse(pf.tighten(p))
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), mode)

    def test_anything_but_a_regular_file_of_yours_is_left_as_it_is(self):
        target = self.file("target.yaml", 0o644)
        link = self.d / "link.yaml"
        link.symlink_to(target)
        fifo = self.d / "fifo"
        os.mkfifo(fifo)
        fifo.chmod(0o644)
        self.pkg.chmod(0o755)
        for p in (link, fifo, self.d / "none.yaml", self.pkg):
            with self.subTest(p=p.name):
                self.assertFalse(pf.tighten(p))
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)  # not followed
        self.assertEqual(stat.S_IMODE(os.lstat(fifo).st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(self.pkg.stat().st_mode), 0o755)
        with mock.patch.object(pf.os, "getuid", return_value=os.getuid() + 1):
            self.assertFalse(pf.tighten(target))  # another's
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_a_file_changed_since_its_check_is_left_as_it_is(self):
        target = self.file("target.yaml", 0o644)
        mode_rows = pf._mode_rows

        def then_a_link(p, name, st):  # the file becomes a link after its check
            rows = mode_rows(p, name, st)
            p.unlink()
            p.symlink_to(target)
            return rows

        def then_a_pipe(p, name, st):  # the file becomes a pipe after its check
            rows = mode_rows(p, name, st)
            p.unlink()
            os.mkfifo(p)
            p.chmod(0o644)
            return rows

        for swap in (then_a_link, then_a_pipe):
            with self.subTest(swap=swap.__name__):
                p = self.file("s.yaml", 0o644)
                with mock.patch.object(pf, "_mode_rows", side_effect=swap):
                    self.assertFalse(pf.tighten(p))
                # What is there now, the link's file or the pipe, keeps its permissions.
                self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o644)
                p.unlink()


class TestWrite(Case):
    def test_a_new_file_at_600_whatever_the_umask(self):
        text = "a: 1\r\nb: ü\n"
        for umask in (0o000, 0o022, 0o277):
            with self.subTest(umask=oct(umask)):
                old = os.umask(umask)
                try:
                    p = pf.write(
                        self.d / f"new-{umask:o}.yaml", text, self.pkg, retry=RETRY
                    )
                finally:
                    os.umask(old)
                self.assertEqual(p, self.d / f"new-{umask:o}.yaml")
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
                self.assertEqual(p.read_bytes(), text.encode("utf-8"))
        self.assertEqual(  # nothing else is left beside them
            sorted(os.listdir(self.d)),
            ["new-0.yaml", "new-22.yaml", "new-277.yaml", "pkg"],
        )
        self.assertEqual(
            [r.status for r in pf.problems(self.d / "new-22.yaml", self.pkg)],
            ["PASS"],
        )

    def test_a_missing_folder_is_made_yours_only(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.d / "home")}):
            p = pf.write("~/conf/s.yaml", "a: 1\n", self.pkg, retry=RETRY)
        self.assertEqual(p, self.d / "home" / "conf" / "s.yaml")
        self.assertEqual(p.read_text(encoding="utf-8"), "a: 1\n")
        self.assertEqual(stat.S_IMODE(p.parent.stat().st_mode) & 0o077, 0)

    def test_the_caller_names_its_command(self):
        # A refusal's fix names another file with the command the caller ran: a write without it is a mistake.
        with self.assertRaises(TypeError):
            pf.write(self.d / "s.yaml", "a: 1\n", self.pkg)
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg"])

    def test_a_dry_run_checks_the_path_and_writes_nothing(self):
        """A command's --dry-run runs every guard and changes nothing: a path that passes the checks comes back, with
        nothing written and no folder made; one that breaks a rule is refused with the rows a real write gives.
        """
        p = self.d / "new" / "s.yaml"
        self.assertEqual(pf.write(p, "a: 1\n", self.pkg, retry=RETRY, dry_run=True), p)
        self.assertFalse((self.d / "new").exists())
        mine = self.file("mine.yaml", text="mine\n")
        self.assertEqual(  # what replace would put in its place
            pf.write(
                mine, "other\n", self.pkg, replace=True, retry=RETRY, dry_run=True
            ),
            mine,
        )
        own = self.pkg / pf.PACKAGE_FILES[0]  # its place is fixed: no retry
        self.assertEqual(pf.write(own, "a=1\n", self.pkg, dry_run=True), own)
        with self.assertRaises(TypeError):  # your own file: a fix names another
            pf.write(self.d / "t.yaml", "a: 1\n", self.pkg, dry_run=True)
        self.repo()
        for path in (mine, self.d / "repo" / "x.yaml", self.pkg / "docker" / "x.yaml"):
            with self.subTest(path=path):
                with self.assertRaises(pf.Refused) as dry:
                    pf.write(path, "other\n", self.pkg, retry=RETRY, dry_run=True)
                with self.assertRaises(pf.Refused) as real:
                    pf.write(path, "other\n", self.pkg, retry=RETRY)
                self.assertEqual(dry.exception.rows, real.exception.rows)
        self.assertEqual(mine.read_text(encoding="utf-8"), "mine\n")
        self.assertEqual(stat.S_IMODE(mine.stat().st_mode), 0o600)
        self.assertEqual(sorted(os.listdir(self.d)), ["mine.yaml", "pkg", "repo"])
        self.assertEqual(sorted(os.listdir(self.d / "repo")), [".git"])
        self.assertEqual(sorted(os.listdir(self.pkg)), ["docker"])
        self.assertEqual(sorted(os.listdir(self.pkg / "docker")), [])

    def test_never_over_a_file_unless_told(self):
        p = self.file("s.yaml", 0o644, "mine\n")
        with self.assertRaises(pf.Refused) as cm:
            pf.write(p, "other\n", self.pkg, retry=RETRY)
        self.assertIsInstance(cm.exception, ValueError)
        fix = Steps(
            (
                RETRY(str(self.d / "s-2.yaml")),
                ", or move that one aside first:",
                Command(("mv", p, self.d / "s.yaml.old")),
            )
        )
        self.assertEqual(
            cm.exception.rows, [Check("private file", Status.FAIL, f"{p} exists", fix)]
        )
        self.assertEqual(  # the one way a refusal is shown
            str(cm.exception),
            render(cm.exception.rows, "problems") + "\n" + pf.RULES,
        )
        self.assertTrue(
            one_line(str(cm.exception)).startswith(
                f"FAIL private file: {p} exists fix: rkmon check --settings {quoted(self.d / 's-2.yaml')}, or"
            ),
            str(cm.exception),
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "mine\n")
        before = p.stat().st_ino
        self.assertEqual(pf.write(p, "other\n", self.pkg, replace=True, retry=RETRY), p)
        self.assertEqual(p.read_text(encoding="utf-8"), "other\n")
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertNotEqual(p.stat().st_ino, before)  # a new file renamed into place
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "s.yaml"])

    def test_its_rows_show_the_file_as_the_caller_shows_it(self):
        # SHOWN, the file as the command's lines show it, and the free names beside it beside SHOWN.
        p = self.file("home/s.yaml", text="mine\n")
        with self.assertRaises(pf.Refused) as cm:
            pf.write(p, "other\n", self.pkg, retry=RETRY, shown="~/s.yaml")
        (row,) = cm.exception.rows
        self.assertEqual(
            (row.found, str(row.fix)),
            (
                "~/s.yaml exists",
                "rkmon check --settings ~/s-2.yaml, or move that one aside first: mv ~/s.yaml ~/s.yaml.old",
            ),
        )
        link = self.d / "home" / "link.yaml"
        link.symlink_to(p)
        with self.assertRaises(pf.Refused) as cm:
            pf.write(link, "other\n", self.pkg, True, retry=RETRY, shown="~/link.yaml")
        self.assertEqual(
            cm.exception.rows[0].found,
            "~/link.yaml is a link, not a file",
        )
        synced = self.d / "home" / "Box" / "s.yaml"
        with self.assertRaises(pf.Refused) as cm:
            pf.write(synced, "other\n", self.pkg, retry=RETRY, shown="~/Box/s.yaml")
        self.assertEqual(
            cm.exception.rows[0].found,
            f"~/Box/s.yaml is inside the cloud-synced folder {self.d / 'home' / 'Box'}: {SYNCED}",
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "mine\n")

    def test_a_suggested_name_is_free(self):
        p = self.file("s.yaml")
        self.file("s-2.yaml")
        self.file("s.yaml.old")
        with self.assertRaises(pf.Refused) as cm:
            pf.write(p, "other\n", self.pkg, retry=RETRY)
        self.assertEqual(
            str(cm.exception.rows[0].fix),
            f"rkmon check --settings {quoted(self.d / 's-3.yaml')}, or move that one aside first: mv {quoted(p)} "
            f"{quoted(self.d / 's.yaml-2.old')}",
        )

    def test_aside_names_a_free_place_beside_the_file(self):
        p = self.file("s.yaml")
        self.assertEqual(pf.aside(p), self.d / "s.yaml.old")
        self.file("s.yaml.old")
        self.assertEqual(pf.aside(str(p)), self.d / "s.yaml-2.old")
        (self.d / "s.yaml-2.old").symlink_to(self.d / "nothing")  # takes its name too
        with mock.patch.dict(os.environ, {"HOME": str(self.d)}):
            self.assertEqual(pf.aside("~/s.yaml"), self.d / "s.yaml-3.old")
        self.assertEqual(  # it names a place, and makes nothing
            sorted(os.listdir(self.d)), ["pkg", "s.yaml", "s.yaml-2.old", "s.yaml.old"]
        )

    def test_refused_in_the_three_folders_with_nothing_written(self):
        self.repo()
        for rel in ("pkg/docker/x.yaml", "repo/x.yaml", "Dropbox/x.yaml"):
            with self.subTest(rel=rel):
                p = self.d / rel
                for replace in (False, True):
                    with self.assertRaises(pf.Refused) as cm:
                        pf.write(p, "a: 1\n", self.pkg, replace=replace, retry=RETRY)
                    (row,) = cm.exception.rows
                    self.assertEqual(row.fix, RETRY("~/x.yaml"))
                    self.assertEqual(str(row.fix), "rkmon check --settings ~/x.yaml")
                    self.assertTrue(row.found.startswith(f"{p} is inside the "))
                    self.assertFalse(os.path.lexists(p))
        self.assertFalse((self.d / "Dropbox").exists())

    def test_refused_onto_a_link_or_a_folder(self):
        real = self.file("real.yaml", text="mine\n")
        link = self.d / "link.yaml"
        link.symlink_to(real)
        folder = self.d / "folder.yaml"
        folder.mkdir()
        for path, found, fix in (
            (
                link,
                f"{link} is a link, not a file",
                f"name the file it points to: rkmon check --settings {quoted(real)}",
            ),
            (
                folder,
                f"{folder} is not a regular file",
                "rkmon check --settings FILE, FILE being a regular file's full path",
            ),
        ):
            for replace in (False, True):
                with self.subTest(path=path.name, replace=replace):
                    with self.assertRaises(pf.Refused) as cm:
                        pf.write(
                            path, "other\n", self.pkg, replace=replace, retry=RETRY
                        )
                    self.assertEqual([r.found for r in cm.exception.rows], [found])
                    self.assertEqual([str(r.fix) for r in cm.exception.rows], [fix])
        self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_text(encoding="utf-8"), "mine\n")

    def test_a_retry_that_names_only_a_folder_names_nothing_in_a_link_s_place(self):
        """A RETRY that names only another folder (folder_only, as a secret's has): a link in the place written is
        removed, the link alone, and the command run again, whether it points to a file or to nothing; the folder's
        edit is never the fix."""
        real = self.file("real.yaml", text="mine\n")
        folder = pf.folder_only(lambda path: Command(("rkmon", "folder", str(path))))
        for to in (real, self.d / "nothing.yaml"):
            for replace in (False, True):
                with self.subTest(to=to.name, replace=replace):
                    link = self.d / f"link-{to.stem}-{replace}.yaml"
                    link.symlink_to(to)
                    with self.assertRaises(pf.Refused) as cm:
                        pf.write(
                            link, "other\n", self.pkg, replace=replace, retry=folder
                        )
                    self.assertEqual(
                        [(r.found, str(r.fix)) for r in cm.exception.rows],
                        [
                            (
                                f"{link} is a link, not a file",
                                f"rm {quoted(link)} (the link alone), then run the command again",
                            )
                        ],
                    )
                    self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_text(encoding="utf-8"), "mine\n")

    def test_a_dry_run_refuses_as_the_write_would_and_makes_nothing(self):
        self.repo()
        taken = self.file("s.yaml", text="mine\n")
        for p in (
            taken,
            self.d / "pkg" / "docker" / "x.yaml",
            self.d / "repo" / "x.yaml",
        ):
            with self.subTest(path=p):
                with self.assertRaises(pf.Refused) as real:
                    pf.write(p, "other\n", self.pkg, retry=RETRY)
                with self.assertRaises(pf.Refused) as dry:
                    pf.write(p, "other\n", self.pkg, retry=RETRY, dry_run=True)
                self.assertEqual(dry.exception.rows, real.exception.rows)
        self.assertEqual(taken.read_text(encoding="utf-8"), "mine\n")
        with mock.patch.dict(os.environ, {"HOME": str(self.d / "home")}):
            p = pf.write("~/conf/s.yaml", "a: 1\n", self.pkg, retry=RETRY, dry_run=True)
        self.assertEqual(p, self.d / "home" / "conf" / "s.yaml")
        self.assertFalse((self.d / "home").exists())  # no file, and no folder for it
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "repo", "s.yaml"])

    def test_the_new_file_beside_it_is_among_its_kin(self):
        """The new file a write puts beside the file is NEW_FILE: a package's own file's new one, left by a write that
        stopped part-way, is among its kin (own_names), which the ignore lists hold wherever it is.
        """
        made = []
        real_open = os.open

        def recording(path, *a, **k):
            made.append(os.path.basename(path))
            return real_open(path, *a, **k)

        with mock.patch.object(pf.os, "open", recording):
            pf.write(self.pkg / "docker" / ".env", "RKMON_TAG=t1\n", self.pkg)
        self.assertTrue(
            [m for m in made if fnmatch.fnmatchcase(m, pf.own_names(".env")[2])], made
        )

    def test_a_failed_write_leaves_the_file_as_it_was(self):
        p = self.file("s.yaml", text="mine\n")
        with mock.patch.object(pf.os, "fsync", side_effect=OSError(5, "I/O error")):
            with self.assertRaises(OSError):
                pf.write(p, "other\n", self.pkg, replace=True, retry=RETRY)
        self.assertEqual(p.read_text(encoding="utf-8"), "mine\n")
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "s.yaml"])

    def test_the_file_and_its_folder_are_synced(self):
        with mock.patch.object(pf.os, "fsync", wraps=os.fsync) as fsync:
            pf.write(self.d / "s.yaml", "a: 1\n", self.pkg, retry=RETRY)
        self.assertEqual(fsync.call_count, 2)

    def test_a_file_another_writer_makes_meanwhile_stays(self):
        # Another writer makes the file after the check, just before this write puts its own in place.
        p = self.d / "s.yaml"

        def theirs_first(real):
            def put(src, dst, *args, **kwargs):
                pathlib.Path(dst).write_text("theirs\n", encoding="utf-8")
                return real(src, dst, *args, **kwargs)

            return put

        with mock.patch.object(pf.os, "link", theirs_first(os.link)), mock.patch.object(
            pf.os, "replace", theirs_first(os.replace)
        ):
            with self.assertRaises(pf.Refused) as cm:
                pf.write(p, "mine\n", self.pkg, retry=RETRY)
        self.assertEqual([r.found for r in cm.exception.rows], [f"{p} exists"])
        self.assertEqual(p.read_text(encoding="utf-8"), "theirs\n")
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "s.yaml"])

    def test_where_no_hard_link_is_made_the_file_is_made_in_its_place(self):
        no_link = OSError(errno.EPERM, os.strerror(errno.EPERM))
        p = self.d / "s.yaml"
        with mock.patch.object(pf.os, "link", side_effect=no_link):
            self.assertEqual(pf.write(p, "mine\n", self.pkg, retry=RETRY), p)
        self.assertEqual(p.read_text(encoding="utf-8"), "mine\n")
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "s.yaml"])

        theirs = self.d / "theirs.yaml"

        def made_meanwhile(src, dst):
            pathlib.Path(dst).write_text("theirs\n", encoding="utf-8")
            raise no_link

        with mock.patch.object(pf.os, "link", side_effect=made_meanwhile):
            with self.assertRaises(pf.Refused) as cm:
                pf.write(theirs, "mine\n", self.pkg, retry=RETRY)
        self.assertEqual([r.found for r in cm.exception.rows], [f"{theirs} exists"])
        self.assertEqual(theirs.read_text(encoding="utf-8"), "theirs\n")

        failed, fsync, calls = self.d / "failed.yaml", os.fsync, []

        def second_fails(
            fd,
        ):  # the file beside it is synced; the one made in its place is not
            calls.append(fd)
            if len(calls) == 2:
                raise OSError(errno.EIO, os.strerror(errno.EIO))
            return fsync(fd)

        with mock.patch.object(pf.os, "link", side_effect=no_link):
            with mock.patch.object(pf.os, "fsync", side_effect=second_fails):
                with self.assertRaises(OSError):
                    pf.write(failed, "mine\n", self.pkg, retry=RETRY)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(os.listdir(self.d)), ["pkg", "s.yaml", "theirs.yaml"])


class TestPackageFiles(Case):
    """The package's own private files: at their places in the package folder, the file rules and one place rule."""

    OWN = "is yours only, at its place in the package folder, outside cloud sync"

    def test_the_list(self):
        self.assertEqual(
            pf.PACKAGE_FILES,
            (
                "docker/.env",
                "docker/compose.override.yaml",
                "docker/.env.prev",
                "docker/compose.override.yaml.prev",
                "docker/.rkmon/records.json",
            ),
        )

    def test_their_kin_and_their_places(self):
        """Beside each of them, its copies and the new file a write of it left hold what it holds: their places are the
        package's own too, and so is all of RECORDS_FOLDER; the same names anywhere else are not.
        """
        self.assertEqual(pf.own_names(".env"), (".env", ".env.*", "..env.*.new"))
        self.assertEqual(
            pf.own_places(),
            [
                "docker/.env",
                "docker/.env.*",
                "docker/..env.*.new",
                "docker/compose.override.yaml",
                "docker/compose.override.yaml.*",
                "docker/.compose.override.yaml.*.new",
                "docker/.env.prev",
                "docker/.env.prev.*",
                "docker/..env.prev.*.new",
                "docker/compose.override.yaml.prev",
                "docker/compose.override.yaml.prev.*",
                "docker/.compose.override.yaml.prev.*.new",
                "docker/.rkmon",
            ],
        )
        for rel in (
            "docker/.env",
            "docker/.env.prev",
            "docker/.env.bak",
            "docker/..env.4242.0a1b2c3d.new",
            "docker/compose.override.yaml.mine",
            "docker/.rkmon",
            "docker/.rkmon/",
            "docker/.rkmon/records.json",
            "docker/.rkmon/.records.json.4242.0a1b2c3d.new",
        ):
            with self.subTest(rel=rel):
                self.assertTrue(pf.in_own_places(rel))
        for rel in (
            ".env",
            "docker/x/.env",
            "tests/compose.override.yaml",
            "docker/records.json",
            "docker/.rkmon.d/records.json",
            "docker/env",
        ):
            with self.subTest(rel=rel):
                self.assertFalse(pf.in_own_places(rel))

    def test_each_at_its_place_has_the_file_rules(self):
        subprocess.run(["git", "init", "-q", str(self.pkg)], check=True)
        for name in pf.PACKAGE_FILES:
            with self.subTest(name=name):
                p = self.file(f"pkg/{name}")
                self.assertEqual(
                    pf.problems(p, self.pkg, "deploy file"),
                    [Check("deploy file", Status.PASS, f"{p} {self.OWN}")],
                )
                p.chmod(0o644)
                self.assertEqual(
                    [(r.found, r.fix) for r in self.failed(p)],
                    [
                        (
                            f"{p} has permissions 644: others on this machine can read it",
                            Command(("chmod", "600", p)),
                        )
                    ],
                )
                p.chmod(0o600)
                yours = f"{p} belongs to {pf._user(os.getuid())}, not to you"
                with mock.patch.object(pf.os, "getuid", return_value=os.getuid() + 1):
                    self.assertEqual([r.found for r in self.failed(p)], [yours])
                p.unlink()
                p.symlink_to(self.file("elsewhere.txt"))
                self.assertEqual(
                    [r.found for r in self.failed(p)],
                    [f"{p} is a link, not a file"],
                )
                p.unlink()

    def test_each_is_written_at_its_place(self):
        for name in pf.PACKAGE_FILES:
            with self.subTest(name=name):
                p = self.pkg / name
                self.assertEqual(
                    pf.write(p, "a=1\n", self.pkg), p
                )  # its place is fixed: no retry
                self.assertEqual(p.read_text(encoding="utf-8"), "a=1\n")
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
                with self.assertRaises(pf.Refused) as cm:
                    pf.write(p, "a=2\n", self.pkg, retry=RETRY)
                aside = Command(("mv", p, p.with_name(p.name + ".old")))
                self.assertEqual(
                    cm.exception.rows,
                    [Check("private file", Status.FAIL, f"{p} exists", aside)],
                )
                self.assertEqual(pf.write(p, "a=2\n", self.pkg, replace=True), p)
                self.assertEqual(p.read_text(encoding="utf-8"), "a=2\n")
                self.assertEqual([r.status for r in pf.problems(p, self.pkg)], ["PASS"])

    def test_any_other_path_in_the_package_folder_is_refused(self):
        for rel in (
            "docker/x.yaml",
            "docker/.env.old",
            "docker/.rkmon/other.json",
            "docker/sub/.env",
            ".env",
        ):
            with self.subTest(rel=rel):
                p = self.pkg / rel
                with self.assertRaises(TypeError):  # your own file: a fix names another
                    pf.write(p, "a: 1\n", self.pkg)
                with self.assertRaises(pf.Refused) as cm:
                    pf.write(p, "a: 1\n", self.pkg, retry=RETRY)
                self.assertEqual(
                    [r.found for r in cm.exception.rows],
                    [
                        f"{p} is inside the package folder: the image's build context and a commit would carry it"
                    ],
                )
                self.assertFalse(os.path.lexists(p))
                q = self.file(f"pkg/{rel}")
                self.assertEqual(
                    [r.found for r in self.failed(q)], [cm.exception.rows[0].found]
                )

    def test_their_place_rule_is_a_package_folder_outside_cloud_sync(self):
        pkg = self.d / "Dropbox" / "pkg"
        p = self.file("Dropbox/pkg/docker/.env")
        move = Steps(
            (
                Command(("mv", pkg, "~/pkg")),
                ", then use the package folder from there",
            )
        )
        cloud = f"is inside the cloud-synced folder {self.d / 'Dropbox'}: {SYNCED}"
        self.assertEqual(
            pf.problems(p, pkg),
            [Check("private file", Status.FAIL, f"{p} {cloud}", move)],
        )
        self.assertEqual(
            str(move), f"mv {quoted(pkg)} ~/pkg, then use the package folder from there"
        )
        q = pkg / "docker" / "compose.override.yaml"
        with self.assertRaises(pf.Refused) as cm:
            pf.write(q, "services: {}\n", pkg)
        self.assertEqual(
            cm.exception.rows,
            [Check("private file", Status.FAIL, f"{q} {cloud}", move)],
        )
        self.assertFalse(q.exists())

    def test_a_link_to_the_package_folder_is_resolved(self):
        p = self.file("pkg/docker/.env")
        (self.d / "via").symlink_to(self.pkg)
        via = self.d / "via" / "docker" / ".env"
        self.assertEqual(
            [r.found for r in pf.problems(via, self.pkg)], [f"{via} {self.OWN}"]
        )
        self.assertEqual(
            [r.found for r in pf.problems(p, self.d / "via")], [f"{p} {self.OWN}"]
        )


class TestCarry(Case):
    def setUp(self):
        super().setUp()
        self.old, self.new = self.d / "old", self.d / "new"
        self.file("old/docker/.env", text="RKMON_TAG=x\n")
        self.file("old/docker/.rkmon/records.json", text="{}\n")
        self.file("old/docker/compose.yaml", 0o644, "services: {}\n")
        (self.new / "docker").mkdir(parents=True)

    def carried(self, name: str) -> Check:
        return Check(name, Status.PASS, f"carried to {self.new / name} at 600")

    def test_each_of_the_package_files_comes_along_at_600(self):
        for i, name in enumerate(pf.PACKAGE_FILES):
            self.file(f"old/{name}", text=f"{i}\n")
        self.assertEqual(
            pf.carry(self.old, self.new),
            [self.carried(name) for name in pf.PACKAGE_FILES],
        )
        for i, name in enumerate(pf.PACKAGE_FILES):
            with self.subTest(name=name):
                p = self.new / name
                self.assertEqual(p.read_text(encoding="utf-8"), f"{i}\n")
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertNotIn("compose.yaml", os.listdir(self.new / "docker"))

    def test_the_files_the_old_folder_has_come_along(self):
        self.assertEqual(
            pf.carry(self.old, self.new),
            [self.carried("docker/.env"), self.carried("docker/.rkmon/records.json")],
        )
        self.assertEqual(sorted(os.listdir(self.new / "docker")), [".env", ".rkmon"])

    def test_older_permissions_are_set_to_600_in_both_folders(self):
        env = self.old / "docker" / ".env"
        records = self.old / "docker" / ".rkmon" / "records.json"
        env.chmod(0o644)
        records.chmod(0o200)  # you cannot read it
        rows = pf.carry(self.old, self.new)
        self.assertEqual([r.status for r in rows], ["PASS", "PASS"])
        self.assertEqual(
            [r.found for r in rows],
            [
                f"carried to {self.new}/docker/.env at 600; its permissions were 644, and are 600 in both folders "
                "now",
                f"carried to {self.new}/docker/.rkmon/records.json at 600; its permissions were 200, and are 600 in "
                "both folders now",
            ],
        )
        for p in (
            env,
            records,
            self.new / "docker" / ".env",
            self.new / "docker" / ".rkmon" / "records.json",
        ):
            with self.subTest(p=str(p)):
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertEqual(
            (self.new / "docker" / ".rkmon" / "records.json").read_text(
                encoding="utf-8"
            ),
            "{}\n",
        )

    def test_a_dry_run_checks_alone(self):
        """A dry run runs the checks and says what the carry would do; nothing is set or written."""
        env = self.old / "docker" / ".env"
        env.chmod(0o644)
        self.assertEqual(
            [
                (r.name, r.status, r.found)
                for r in pf.carry(self.old, self.new, dry_run=True)
            ],
            [
                (
                    "docker/.env",
                    Status.PASS,
                    f"would be carried to {self.new}/docker/.env at 600; its permissions are 644, and would be 600 "
                    "in both folders",
                ),
                (
                    "docker/.rkmon/records.json",
                    Status.PASS,
                    f"would be carried to {self.new}/docker/.rkmon/records.json at 600",
                ),
            ],
        )
        self.assertEqual(stat.S_IMODE(env.stat().st_mode), 0o644)
        self.assertEqual(os.listdir(self.new / "docker"), [])
        (self.new / "docker" / ".env").write_text("RKMON_TAG=y\n")
        with self.assertRaises(pf.Refused) as cm:
            pf.carry(self.old, self.new, dry_run=True)
        self.assertEqual(
            [r.found for r in cm.exception.rows],
            [f"{self.new / 'docker' / '.env'} exists"],
        )

    def test_a_reader_of_a_file_as_the_carry_leaves_it(self):
        """checked's CARRIED passes over the permissions, which the carry sets to 600; every other rule holds."""
        records = self.old / "docker" / ".rkmon" / "records.json"
        records.chmod(0o644)
        with self.assertRaises(pf.Refused):
            pf.checked(records, self.old)
        self.assertEqual(pf.checked(records, self.old, carried=True), records)
        records.unlink()
        records.symlink_to(self.old / "docker" / "compose.yaml")
        with self.assertRaises(pf.Refused) as cm:
            pf.checked(records, self.old, carried=True)
        self.assertEqual(
            [r.found for r in cm.exception.rows],
            [f"{records} is a link, not a file"],
        )

    def test_refused_with_nothing_written(self):
        env = self.old / "docker" / ".env"
        records = self.old / "docker" / ".rkmon" / "records.json"
        compose = self.old / "docker" / "compose.yaml"
        cases = {
            "a link": (
                lambda: (env.unlink(), env.symlink_to(compose)),
                f"{env} is a link, not a file",
            ),
            "no regular file": (
                lambda: (env.unlink(), os.mkfifo(env)),
                f"{env} is not a regular file",
            ),
            "one the new folder has": (
                lambda: (self.new / "docker" / ".env").write_text("RKMON_TAG=y\n"),
                f"{self.new / 'docker' / '.env'} exists",
            ),
        }
        for what, (make, found) in cases.items():
            with self.subTest(what=what):
                self.reset()
                make()
                with self.assertRaises(pf.Refused) as cm:
                    pf.carry(self.old, self.new)
                self.assertEqual([r.found for r in cm.exception.rows], [found])
                self.assertEqual(cm.exception.rows[0].name, "docker/.env")
                self.assertFalse((self.new / "docker" / ".rkmon").exists())
        # The new folder's own file moves aside; the carry runs again after.
        new_env = self.new / "docker" / ".env"
        self.assertEqual(
            cm.exception.rows[0].fix,
            Command(("mv", new_env, new_env.with_name(".env.old"))),
        )
        self.reset()
        with mock.patch.object(pf.os, "getuid", return_value=os.getuid() + 1):
            with self.assertRaises(pf.Refused) as cm:
                pf.carry(self.old, self.new)
        self.assertEqual(
            [r.found for r in cm.exception.rows],
            [
                f"{p} belongs to {pf._user(os.getuid())}, not to you"
                for p in (env, records)
            ],
        )
        # Refused, it changes nothing: a file at other permissions stays so.
        self.reset()
        env.chmod(0o644)
        records.unlink()
        records.symlink_to(compose)
        with self.assertRaises(pf.Refused):
            pf.carry(self.old, self.new)
        self.assertEqual(stat.S_IMODE(env.stat().st_mode), 0o644)
        self.assertEqual(os.listdir(self.new / "docker"), [])

    def reset(self):
        """The two folders as setUp made them."""
        env = self.old / "docker" / ".env"
        if os.path.lexists(env):
            env.unlink()
        self.file("old/docker/.env", text="RKMON_TAG=x\n")
        p = self.new / "docker" / ".env"
        if os.path.lexists(p):
            p.unlink()

    def test_a_cloud_synced_package_folder_takes_none(self):
        new = self.d / "Dropbox" / "new"
        (new / "docker").mkdir(parents=True)
        with self.assertRaises(pf.Refused) as cm:
            pf.carry(self.old, new)
        move = Command(("mv", new, "~/new"))
        self.assertEqual(
            cm.exception.rows,
            [
                Check(
                    "package folder",
                    Status.FAIL,
                    f"{new} is inside the cloud-synced folder {self.d / 'Dropbox'}: {SYNCED}",
                    Steps((move, ", then use the package folder from there")),
                )
            ],
        )
        self.assertEqual(os.listdir(new / "docker"), [])

    def test_a_package_folder_in_git_takes_them(self):
        subprocess.run(["git", "init", "-q", str(self.new)], check=True)
        self.assertEqual(
            pf.carry(self.old, self.new, ["docker/.env"]),
            [self.carried("docker/.env")],
        )

    def test_a_failure_part_way_leaves_nothing_written(self):
        (self.new / "docker").rmdir()  # the folders it makes go too
        put = pf._put

        def records_fail(p, data, replace):
            if p.name != "records.json":
                return put(p, data, replace)
            p.parent.mkdir(parents=True)  # its folder is made, then the disk is full
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        with mock.patch.object(pf, "_put", side_effect=records_fail):
            with self.assertRaises(OSError):
                pf.carry(self.old, self.new)
        self.assertEqual(os.listdir(self.new), [])
        self.assertEqual(
            (self.old / "docker" / ".env").read_text(encoding="utf-8"), "RKMON_TAG=x\n"
        )

    def test_a_file_changed_since_its_check_is_refused(self):
        records = self.old / "docker" / ".rkmon" / "records.json"
        compose = self.old / "docker" / "compose.yaml"
        put = pf._put

        def then_a_link(
            p, data, replace
        ):  # the next file becomes a link after the check
            put(p, data, replace)
            records.unlink()
            records.symlink_to(compose)

        with mock.patch.object(pf, "_put", side_effect=then_a_link):
            with self.assertRaises(pf.Refused) as cm:
                pf.carry(self.old, self.new)
        self.assertEqual(
            [r.found for r in cm.exception.rows],
            [f"{records} is a link, not a file"],
        )
        self.assertEqual(os.listdir(self.new / "docker"), [])
        self.assertEqual(stat.S_IMODE(compose.stat().st_mode), 0o644)

    def test_only_the_package_files(self):
        for name in (
            "docker/compose.yaml",
            "docker/x.yaml",
            ".env",
            "../.env",
            "/etc/hostname",
            "",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as cm:
                    pf.carry(self.old, self.new, [name])
                self.assertIs(type(cm.exception), ValueError)
        self.assertEqual(os.listdir(self.new / "docker"), [])


class TestTheSecretsFolder(Case):
    def test_the_secrets_folder_s_files_follow_the_private_file_rules(self):
        """rk doctor's rows of the secrets folder (doctor.secrets_checks): the folder, then each of its files by the
        private-file rules, a login's folder left to its own rows; none while the configuration names no secrets
        folder.
        """
        work = self.d / "work"
        d = work / ".config"
        (d / "gh-read").mkdir(parents=True)  # a login's folder: its files are its own
        d.chmod(0o700)
        elsewhere = self.d / "elsewhere"
        files = {
            "admin_token": 0o600,
            "teams_webhook": 0o644,
            "worker_headers": 0o200,
        }
        for name, mode in files.items():
            (d / name).write_text("x\n", encoding="utf-8")
            (d / name).chmod(mode)
        elsewhere.write_text("x\n", encoding="utf-8")
        elsewhere.chmod(0o644)
        link = d / "worker_api_key"
        link.symlink_to(elsewhere)
        cfg = {"paths": {"working_dir": str(work)}}

        def rows() -> list:
            return [
                (r.name, r.status, r.found, str(r.fix or ""))
                for r in doctor.secrets_checks(cfg)
            ]

        token, hook, headers = (d / n for n in files)
        self.assertEqual(
            rows(),
            [
                ("secrets folder", "PASS", f"{d} permissions 700", ""),
                ("secret admin_token", "PASS", f"{token} {PASSED}", ""),
                (
                    "secret teams_webhook",
                    "FAIL",
                    f"{hook} has permissions 644: others on this machine can read it",
                    f"chmod 600 {quoted(hook)}",
                ),
                (
                    "secret worker_api_key",
                    "FAIL",
                    f"{link} is a link, not a file",
                    f"rm {quoted(link)}, then cp {quoted(elsewhere)} {quoted(link)}",
                ),
                (
                    "secret worker_headers",
                    "FAIL",
                    f"{headers} has permissions 200: you cannot read it",
                    f"chmod 600 {quoted(headers)}",
                ),
            ],
        )
        # A loose folder: its own fix, before its files'.
        d.chmod(0o755)
        self.assertEqual(
            rows()[0],
            (
                "secrets folder",
                "FAIL",
                f"{d} permissions 755",
                f"chmod 700 {quoted(d)}",
            ),
        )
        cfg = {"paths": {}}
        self.assertEqual(rows(), [])


if __name__ == "__main__":
    unittest.main()
