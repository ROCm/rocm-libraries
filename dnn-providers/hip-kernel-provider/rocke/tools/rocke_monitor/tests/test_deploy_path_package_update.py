# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon package update and the package folder's lock, end to end: docker/rkmon run on a copy of the package against the
stand-in docker, as tests/deploy_path_fixture.py sets it up. package update --folder replaces an older install's package
folder, and the commands that run there next are named by that folder's own rkmon; it refuses while an apply or rollback
is under way there; a Ctrl-C changes nothing, or stops only the clean-up after the update's renames. Each command holds
the package folder's lock as its row says: none starts while a package update runs, and no update while other commands
run, each named, since when; a command that ended blocks nothing. An update holds the lock of each folder it moves or
makes, from before it moves it until it ends; run by the rkmon of the folder it moves, it is refused, and its next line
names the updated folder's own rkmon.
"""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
from deploy_path_fixture import DeployPathCase  # noqa: E402
from rendered import finished, one_line  # noqa: E402
from test_package_update import in_use, make_package, snapshot  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    commands,
    image,
    outcome,
    package_lock,
    package_update,
    private_files,
    program,
    progress,
    records,
)

# package update of the older install's folder in the test's home folder, from the newer package there
# (TestPackageUpdate.package_folders); and that folder's own rkmon, which names the commands that run there next
UPDATE_FOLDER = (
    "package",
    "update",
    "--from",
    "~/new",
    "--folder",
    "~/rocke_monitor",
)
FOLDER_RKMON = "~/rocke_monitor/docker/rkmon"
# A program that runs the wrapper (argv[1]) on the arguments after argv[2], as its main does, with argv[2], a method of
# its Wrapper or a function of a module of the package (Wrapper.next_line_of, rkmon_lib.package_update.interrupted),
# pressing Ctrl-C as it is called (SIGINT to this process), which it says on stderr first
PRESSED_AT = """
import importlib, importlib.machinery, importlib.util, os, signal, sys
wrapper, at, *args = sys.argv[1:]
loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", wrapper)
mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("rkmon_wrapper", loader))
loader.exec_module(mod)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(wrapper))))
where, name = at.rsplit(".", 1)
owner = mod.Wrapper if where == "Wrapper" else importlib.import_module(where)
real = getattr(owner, name)


def press(*a, **kw):
    sys.stderr.write("pressed\\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGINT)
    return real(*a, **kw)


setattr(owner, name, press)
sys.argv = [wrapper, *args]
sys.exit(mod.main())
"""


class TestPackageUpdate(DeployPathCase):
    def package_folders(self) -> pathlib.Path:
        """An older install's package folder in your home folder, ~/rocke_monitor (1.0, in use), and a newer
        package, ~/new (2.0), as tests/test_package_update.py makes them (UPDATE_FOLDER). Returns the older folder.
        """
        folder = make_package(self.home / "rocke_monitor", "1.0")
        in_use(folder)
        make_package(self.home / "new", "2.0")
        return folder

    def test_package_update_of_another_folder(self):
        """package update --folder (the move from an older install), run by this package's rkmon: its dry run,
        PASSED with the rows of the files the update would carry, its next line the update, then the command that the
        folder's own rkmon runs there; the update, UPDATED with the rows of the files carried, its next line that
        command, and the folder it replaced kept beside it; run again, NOTHING TO DO with that next line. A refusal
        names --folder in its fix, and this command again."""
        folder = self.package_folders()
        update = " ".join(UPDATE_FOLDER)
        rc, out, err = self.rkmon(*UPDATE_FOLDER, "--dry-run")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            f"PASS docker/.env: would be carried to {folder}/docker/.env at 600",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: the package from ~/new can replace ~/rocke_monitor",
                    f"next: {self.w} {update}, then {FOLDER_RKMON} next",
                )
            ),
            out,
        )
        self.assertFalse((self.home / "rocke_monitor.prev").exists())
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            f"PASS docker/.env: carried to {folder}/docker/.env at 600", one_line(out)
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "UPDATED: ~/rocke_monitor has the package from ~/new; the one it replaced is "
                    "~/rocke_monitor.prev",
                    f"next: {FOLDER_RKMON} next",
                )
            ),
            out,
        )
        prev = self.home / "rocke_monitor.prev"
        self.assertEqual(
            [(f / "VERSION").read_text(encoding="utf-8") for f in (folder, prev)],
            ["2.0\n", "1.0\n"],
        )
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: ~/rocke_monitor has the package from ~/new already",
                    f"next: {FOLDER_RKMON} next",
                )
            ),
            out,
        )
        (self.home / "empty").mkdir()
        typed = (
            "package",
            "update",
            "--from",
            "~/empty",
            "--folder",
            "~/rocke_monitor",
        )
        rc, out, err = self.rkmon(*typed)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(
            one_line(out).endswith(
                "REFUSED: ~/empty has no package: neither it nor its one folder has rkmon.py and rkmon_lib/; nothing "
                f"changed fix: {self.w} package update --from DIR --folder ~/rocke_monitor, DIR being a package "
                f"folder or a tarball the package build made next: {self.w} {' '.join(typed)}"
            ),
            out,
        )

    def test_package_update_refuses_while_a_change_is_under_way_in_the_folder(self):
        """package update --folder while an apply runs in that folder, holding its lock (rkmon_lib/package_lock.py):
        REFUSED, naming the apply as its hold names it, the folder and since when, then this command again; once that
        apply stopped part-way (its record kept, no process running it): REFUSED, then the apply again by the
        folder's own rkmon, which finishes it there (state 3), and an adopt the same, in the form rkmon help
        names it. Nothing changed.
        """
        folder = self.package_folders()
        (folder / records.FILE).unlink()  # the records the apply below begins with
        apply = f"{FOLDER_RKMON} apply"
        with package_lock.hold(folder, apply), records.begin(folder, "apply"):
            (held,) = package_lock.holders(folder)
            rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertEqual(
            out,
            finished(
                f"REFUSED: {apply} runs in ~/rocke_monitor, since {progress.shown_time(held.since)}; nothing "
                "changed",
                f"next: {self.w} {' '.join(UPDATE_FOLDER)}",
            ),
        )
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the apply in ~/rocke_monitor stopped part-way; nothing changed",
                    f"next: {FOLDER_RKMON} apply",
                )
            ),
            out,
        )
        records.begin(folder, "apply").finish()
        records.begin(folder, "adopt").release()
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the adopt in ~/rocke_monitor stopped part-way; nothing changed",
                    f"next: {FOLDER_RKMON} adopt --from DIR",
                )
            ),
            out,
        )
        self.assertEqual((folder / "VERSION").read_text(encoding="utf-8"), "1.0\n")
        self.assertFalse((self.home / "rocke_monitor.prev").exists())

    def test_a_command_refuses_while_a_package_update_runs(self):
        """While a package update holds this folder's lock (rkmon_lib/package_lock.py), each command that holds it
        too refuses at once, before its banner: REFUSED, a package update runs, nothing changed, then the same command
        again as you typed it; exit 4, and no docker call. status, next and logs hold none, and run.
        """
        self.confirmed()
        # package update --folder of another folder reads this package's files: it holds this folder's lock too
        folder = self.package_folders()
        calls, kept = self.log.read_text(), records.read(self.pkg)
        with package_lock.hold(self.pkg, "rkmon package update", exclusive=True):
            (update,) = package_lock.holders(self.pkg)
            since = progress.shown_time(update.since)
            for typed, refused in (
                (("start",), "a package update runs"),
                (("--dry-run", "apply"), "a package update runs"),
                (("confirm", "--yes"), "a package update runs"),
                (("pause",), "a package update runs"),
                (
                    ("package", "update", "--from", "~/new"),
                    f"rkmon package update runs, since {since}",
                ),
                (UPDATE_FOLDER, "a package update runs"),
            ):
                with self.subTest(typed=typed):
                    rc, out, err = self.rkmon(*typed, **self.running())
                    self.assertEqual((rc, err), (4, ""), out)
                    self.assertEqual(
                        out,
                        finished(
                            f"REFUSED: {refused}; nothing changed",
                            f"next: {self.w} {' '.join(typed)}",
                        ),
                    )
            self.assertEqual(self.log.read_text(), calls)
            for typed in (("status",), ("next",), ("logs",)):
                with self.subTest(typed=typed):
                    rc, out, err = self.rkmon(*typed, **self.running())
                    self.assertEqual(rc, 0, out + err)
                    self.assertNotIn("REFUSED", out + err)
        self.assertEqual(records.read(self.pkg), kept)
        self.assertEqual((folder / "VERSION").read_text(encoding="utf-8"), "1.0\n")

    def test_package_update_refuses_while_commands_run(self):
        """package update while other commands of this package run here, each holding this folder's lock: REFUSED,
        naming each as its hold names it and since when, the earliest first, nothing changed, then the update again;
        exit 4, before its banner. Once they end, it runs.
        """
        self.confirmed()
        update = ("package", "update", "--from", "~/new")
        with package_lock.hold(self.pkg, f"{self.w} start"):
            with package_lock.hold(self.pkg, f"{self.w} shell"):
                held = package_lock.holders(self.pkg)
                rc, out, err = self.rkmon(*update)
        self.assertEqual((rc, err), (4, ""), out)
        runs = "; ".join(
            f"{h.command} runs, since {progress.shown_time(h.since)}" for h in held
        )
        self.assertEqual(
            out,
            finished(
                f"REFUSED: {runs}; nothing changed",
                f"next: {self.w} {' '.join(update)}",
            ),
        )
        self.assertEqual(len(held), 2)
        rc, out, err = self.rkmon(*update)
        self.assertNotIn("runs, since", out + err)
        self.assertTrue(out.startswith("== package update: "), out)
        self.assertEqual(package_lock.holders(self.pkg), [])

    def test_a_command_that_ended_blocks_no_update(self):
        """A command of the folder's package that held its lock and ended without letting it go, as a process that
        is killed does: its holder's file names nothing, package update --folder runs, and leaves no holder's file in
        either folder.
        """
        folder = self.package_folders()
        ended = (
            "import os, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from rkmon_lib import package_lock\n"
            "package_lock.hold(sys.argv[2], sys.argv[3])\n"
            "os._exit(0)\n"
        )
        argv = [sys.executable, "-B", "-I", "-c", ended, str(ROOT), str(folder)]
        subprocess.run([*argv, f"{FOLDER_RKMON} start"], check=True, timeout=60)

        def holders_files(top) -> list:
            names = os.listdir(top / records.FOLDER)
            return [n for n in names if n.startswith(package_lock.HOLDER)]

        self.assertTrue((folder / records.FOLDER).is_dir())

        self.assertEqual(len(holders_files(folder)), 1)
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "UPDATED: ~/rocke_monitor has the package from ~/new", one_line(out)
        )
        for top in (folder, self.home / "rocke_monitor.prev"):
            self.assertEqual(holders_files(top), [], top)

    def test_a_command_whose_folder_an_update_replaced_as_it_started_refuses(self):
        """A command holds this folder's lock, its hold naming it as you typed it, and lets it go as it ends
        (Wrapper.hold_lock, in this process); one whose package folder a package update replaced after it started and
        before it held the lock refuses: REFUSED, nothing changed, then the command again.
        """
        self.confirmed()
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        typed = ["start", "--on"]
        ns = commands.table_parser(image.package_version(self.pkg)).parse_args(typed)
        with mock.patch.object(sys, "path", list(sys.path)):
            w = mod.Wrapper(self.w)
            w.typed = typed
            with contextlib.ExitStack() as held:
                w.hold_lock(ns, held)
                self.assertEqual(
                    package_lock.holders(self.pkg),
                    [package_lock.Holder(f"{program.name()} start --on", mock.ANY)],
                )
            self.assertEqual(package_lock.holders(self.pkg), [])
            w = mod.Wrapper(self.w)
            w.typed = typed
            os.rename(self.pkg, self.d / "pkg.prev")
            shutil.copytree(self.d / "pkg.prev", self.pkg, symlinks=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.ExitStack() as held:
                with self.assertRaises(mod.Stopped) as caught:
                    w.hold_lock(ns, held)
        self.assertEqual(caught.exception.code, 4)
        self.assertEqual(
            out.getvalue(),
            finished(
                "REFUSED: a package update replaced this package folder as the command started; nothing changed",
                f"next: {program.name()} start --on",
            ),
        )
        self.assertEqual(package_lock.holders(self.pkg), [])

    def test_package_update_holds_the_folders_it_moves_or_makes_until_it_ends(self):
        """package update holds the lock of each folder it moves or makes, exclusive, from before it moves it until
        its last line (package_update.update's HOLD; Wrapper.hold_folder): the new package's folder, made beside the
        one it replaces, takes that one's place with its lock. So once the renames are done, through the clean-up and
        the next line, a command that starts in either folder is refused, as while any update runs, the update named
        as you typed it. Once it ends, both locks are free, and neither folder keeps a holder's file.
        """
        folder = self.package_folders()
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        make_package(self.home / "newer", "3.0")
        typed = (
            "package",
            "update",
            "--from",
            "~/newer",
            "--folder",
            "~/rocke_monitor",
        )
        seen = self.d / "seen.json"
        # the update as rkmon runs it, its next line (package_update.next_line, its last step after the clean-up)
        # first saying what a command that starts then in each folder finds, holding its lock as each one does
        probe = (
            "import importlib.machinery, importlib.util, json, sys\n"
            "pkg, w, seen = sys.argv[1:4]\n"
            "sys.path.insert(0, pkg)\n"
            "from rkmon_lib import package_lock, package_update\n"
            "next_line = package_update.next_line\n"
            "def starts(top):\n"
            "    try:\n"
            "        package_lock.hold(top, 'rkmon start').release()\n"
            "    except package_lock.Taken as exc:\n"
            "        return 'an update runs' if exc.update else 'taken'\n"
            "    return 'runs'\n"
            "def probed(folder, named=False):\n"
            "    found = {}\n"
            "    for top in (str(folder), f'{folder}.prev'):\n"
            "        found[top] = [starts(top), [h.command for h in package_lock.holders(top)]]\n"
            "    with open(seen, 'w', encoding='utf-8') as f:\n"
            "        json.dump(found, f)\n"
            "    return next_line(folder, named)\n"
            "package_update.next_line = probed\n"
            "loader = importlib.machinery.SourceFileLoader('rkmon_wrapper', w)\n"
            "wrapper = importlib.util.module_from_spec(\n"
            "    importlib.util.spec_from_loader('rkmon_wrapper', loader)\n"
            ")\n"
            "loader.exec_module(wrapper)\n"
            "sys.argv = [w, *sys.argv[4:]]\n"
            "sys.exit(wrapper.main())\n"
        )
        p = subprocess.run(
            [sys.executable, "-B", "-I", "-c", probe, str(self.pkg), self.w, str(seen)]
            + list(typed),
            cwd=self.home,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual((p.returncode, p.stderr), (0, ""), p.stdout)
        self.assertIn(
            "UPDATED: ~/rocke_monitor has the package from ~/newer",
            one_line(p.stdout),
        )
        update = f"{self.w} {' '.join(typed)}"
        tops = [os.path.realpath(folder), os.path.realpath(folder) + ".prev"]
        self.assertEqual(
            json.loads(seen.read_text(encoding="utf-8")),
            {top: ["an update runs", [update]] for top in tops},
        )
        for top in tops:
            self.assertEqual(package_lock.holders(top), [])
            package_lock.hold(top, "rkmon start").release()
            names = os.listdir(pathlib.Path(top) / records.FOLDER)
            self.assertEqual(
                [n for n in names if n.startswith(package_lock.HOLDER)], [], top
            )

    def test_package_update_refuses_while_a_command_runs_in_a_folder_it_moves(self):
        """The way back puts FOLDER.prev in the folder's place, and every update moves an earlier one away: package
        update holds its lock, exclusive, before it moves it (package_update.update's HOLD). So while a command of
        that package runs there, holding its lock, the way back, an update and their dry runs refuse as an update
        does while commands run: the command, the folder and since when, nothing changed, then the update again.
        Once it ends, the way back runs, and neither folder keeps a holder's file."""
        folder = self.package_folders()
        prev = self.home / "rocke_monitor.prev"
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        make_package(self.home / "newer", "3.0")
        named = ("--folder", "~/rocke_monitor")
        back = ("package", "update", "--from", "~/rocke_monitor.prev", *named)
        onward = ("package", "update", "--from", "~/newer", *named)
        shell = "~/rocke_monitor.prev/docker/rkmon shell"
        with package_lock.hold(prev, shell):
            (held,) = package_lock.holders(prev)
            since = progress.shown_time(held.since)
            for typed in (back, (*back, "--dry-run"), onward, (*onward, "--dry-run")):
                with self.subTest(typed=typed):
                    rc, out, err = self.rkmon(*typed)
                    self.assertEqual((rc, err), (4, ""), out)
                    self.assertTrue(
                        out.endswith(
                            finished(
                                f"REFUSED: {shell} runs in ~/rocke_monitor.prev, since {since}; nothing changed",
                                f"next: {self.w} {' '.join(typed)}",
                            )
                        ),
                        out,
                    )

        def versions() -> list:
            return [(f / "VERSION").read_text(encoding="utf-8") for f in (folder, prev)]

        self.assertEqual(versions(), ["2.0\n", "1.0\n"])
        rc, out, err = self.rkmon(*back)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "UPDATED: ~/rocke_monitor and ~/rocke_monitor.prev changed places",
            one_line(out),
        )
        self.assertEqual(versions(), ["1.0\n", "2.0\n"])
        for top in (folder, prev):
            names = os.listdir(top / records.FOLDER)
            self.assertEqual(
                [n for n in names if n.startswith(package_lock.HOLDER)], [], top
            )

    def test_package_update_run_by_the_folder_it_moves_names_folders_rkmon(self):
        """package update --folder FOLDER run by FOLDER.prev's own rkmon: the command holds FOLDER.prev's lock as its
        package folder's (hold_lock), and the update must hold that lock alone before it moves the folder
        (package_update.update's HOLD). So the update, its dry run and the way back are refused, and the same command
        again could only be refused again: the next line names the update by FOLDER's own rkmon, which holds no lock
        of FOLDER.prev's, without --folder, typed as `--folder FOLDER` or `--folder=FOLDER`. Nothing changed, and
        neither folder keeps a holder's file."""
        folder = make_package(self.home / "rocke_monitor", "2.0")
        # in use, as its commands ran: its records, its lock's file
        in_use(folder)
        package_lock.hold(folder, "rkmon start").release()
        prev = deploy_folder.copy(self.home / "rocke_monitor.prev")
        (prev / records.FOLDER).mkdir(mode=0o700)  # a records' folder: it has a lock
        make_package(self.home / "newer", "3.0")
        w = str(prev / "docker" / "rkmon")
        named, glued = ("--folder", "~/rocke_monitor"), "--folder=~/rocke_monitor"
        for typed, again in (
            (
                ("package", "update", *named, "--from", "~/newer"),
                "package update --from ~/newer",
            ),
            (
                ("package", "update", "--from", "~/newer", glued, "--dry-run"),
                "package update --from ~/newer --dry-run",
            ),
            (
                ("package", "update", "--from", "~/rocke_monitor.prev", *named),
                "package update --from ~/rocke_monitor.prev",
            ),
        ):
            with self.subTest(typed=typed):
                before = snapshot(folder)
                p = subprocess.run(
                    [w, *typed],
                    cwd=self.home,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=120,
                )
                self.assertEqual((p.returncode, p.stderr), (4, ""), p.stdout)
                self.assertRegex(
                    one_line(p.stdout),
                    rf"REFUSED: {re.escape(str(checks.Command((w, *typed))))} runs in ~/rocke_monitor\.prev, since "
                    r"[^;]+; nothing changed next: ",
                )
                self.assertTrue(
                    p.stdout.endswith(finished(f"next: {FOLDER_RKMON} {again}")),
                    p.stdout,
                )
                self.assertEqual(snapshot(folder), before)
                self.assertEqual(
                    [
                        (f / "VERSION").read_text(encoding="utf-8")
                        for f in (folder, prev)
                    ],
                    ["2.0\n", (ROOT / "VERSION").read_text(encoding="utf-8")],
                )
                for top in (folder, prev):
                    names = os.listdir(top / records.FOLDER)
                    self.assertEqual(
                        [n for n in names if n.startswith(package_lock.HOLDER)], [], top
                    )

    def test_a_private_file_s_refusal_in_an_update_shows_itself(self):
        """A private file the update carries that breaks a rule (docker/.env.prev, a link) refuses the update, nothing
        changed: the refusal as it shows itself (private_files.Refused), its rows then where the rules are, then
        REFUSED and this command again (exit 4)."""
        folder = self.package_folders()
        kept = folder / "docker/.env.prev"
        kept.unlink(missing_ok=True)
        os.symlink(folder / "docker/.env", kept)
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (4, ""), out)
        said = one_line(out)
        self.assertIn("FAIL docker/.env.prev: ", said)
        self.assertIn(
            f"{private_files.RULES} REFUSED: a private file the update reads or carries breaks a rule; "
            f"{outcome.FIX_EACH_FAIL}",
            said,
        )
        self.assertTrue(
            out.endswith(finished(f"next: {self.w} {' '.join(UPDATE_FOLDER)}")), out
        )
        self.assertEqual((folder / "VERSION").read_text(encoding="utf-8"), "1.0\n")

    def test_package_update_interrupted_changes_nothing(self):
        """Ctrl-C during package update --folder, while git reads which files the new package holds (a stand-in git
        that sends rkmon the interrupt): INTERRUPTED, nothing changed, then this command again; exit 130, no
        traceback, and nothing left beside the folder."""
        folder = self.package_folders()
        interrupting = self.d / "interrupting"
        interrupting.mkdir()
        git = interrupting / "git"
        git.write_text(
            "#!/usr/bin/env python3\nimport os, signal, time\nos.kill(os.getppid(), signal.SIGINT)\ntime.sleep(60)\n",
            encoding="utf-8",
        )
        git.chmod(0o755)
        before = sorted(os.listdir(self.home))
        rc, out, err = self.rkmon(*UPDATE_FOLDER, PATH=f"{interrupting}:{self.bin}")
        self.assertEqual((rc, err), (130, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: nothing changed",
                    f"next: {self.w} {' '.join(UPDATE_FOLDER)}",
                )
            ),
            out,
        )
        self.assertEqual(sorted(os.listdir(self.home)), before)
        self.assertEqual((folder / "VERSION").read_text(encoding="utf-8"), "1.0\n")

    def test_a_ctrl_c_once_the_update_has_ended_ends_it_by_the_signal(self):
        """Once package update's change has ended, stopped by a first Ctrl-C (a stand-in git that sends rkmon the
        interrupt) or with nothing to do, what is left is to say its outcome: a Ctrl-C meanwhile, as what changed or
        the next line is read (package_update.interrupted, next_line_of), ends the command by the signal, with no
        traceback (ctrl_c_ends)."""
        folder = self.package_folders()
        interrupting = self.d / "interrupting"
        interrupting.mkdir()
        git = interrupting / "git"
        ran = self.d / "git_ran"
        git.write_text(
            "#!/usr/bin/env python3\nimport os, signal, time\n"
            f"open({str(ran)!r}, 'w').close()\nos.kill(os.getppid(), signal.SIGINT)\ntime.sleep(60)\n",
            encoding="utf-8",
        )
        git.chmod(0o755)

        def pressed_at(at: str, **env) -> None:
            p = subprocess.run(
                [sys.executable, "-I", "-c", PRESSED_AT, self.w, at, *UPDATE_FOLDER],
                cwd=self.home,
                env=dict(self.env, **env),
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=120,
            )
            self.assertEqual(
                (p.returncode, p.stderr), (-signal.SIGINT, "pressed\n"), p.stdout
            )

        pressed_at(
            "rkmon_lib.package_update.interrupted", PATH=f"{interrupting}:{self.bin}"
        )
        self.assertTrue(ran.exists())
        self.assertEqual((folder / "VERSION").read_text(encoding="utf-8"), "1.0\n")
        rc, out, err = self.rkmon(*UPDATE_FOLDER)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("UPDATED", out)
        pressed_at("Wrapper.next_line_of")

    def test_an_interrupt_after_the_update_s_renames(self):
        """Ctrl-C once the update's renames are done, FOLDER another folder than before the update, as docker/rkmon
        reads it (Wrapper.update_interrupted, in this process): INTERRUPTED, only the clean-up stopped, then the next
        line of FOLDER's state, by its own rkmon; FOLDER gone: the update stopped part-way, then this command again.
        """
        folder = self.package_folders()
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = mod.Wrapper(self.w)
        w.typed = list(UPDATE_FOLDER)
        before = os.stat(folder)
        # update_interrupted leaves Ctrl-C ending the command by the signal (ctrl_c_ends): this process's goes back
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))

        def interrupted() -> tuple:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = w.update_interrupted(str(folder), before, here=False)
            return rc, out.getvalue()

        with mock.patch.dict(os.environ, {"HOME": str(self.home)}), mock.patch.object(
            sys, "path", list(sys.path)
        ):
            package_update.update(folder, self.home / "new")
            self.assertEqual(
                interrupted(),
                (
                    130,
                    finished(
                        "INTERRUPTED: the update is done; only its clean-up stopped, which the next update finishes",
                        f"next: {FOLDER_RKMON} next",
                    ),
                ),
            )
            os.rename(folder, self.home / "elsewhere")
            self.assertEqual(
                interrupted(),
                (
                    130,
                    finished(
                        "INTERRUPTED: the update stopped part-way: ~/rocke_monitor is not there",
                        f"next: {program.name()} {' '.join(UPDATE_FOLDER)}",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
