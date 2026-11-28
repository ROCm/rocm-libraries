# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The host's records (rkmon_lib/records.py): written at 600 through private_files and read
back as written; a settings file under the home folder kept as `~/…`; a confirm of the same fingerprint keeps its
first time; the change put back and the last doctor result; when deploy step 8 first passed, which keeps its first
time too; an apply, rollback or adopt under way, and whether it still runs, when its process ends however it ends, and
a look at it waited out; which command takes the place of one that stopped part-way; and the files the reader refuses,
with the fix.
"""

import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import private_files, program, records  # noqa: E402
from rkmon_lib.checks import Command, Edit, Steps  # noqa: E402

AT = "2026-01-02T03:04:05Z"
LATER = "2026-01-03T04:05:06Z"


def confirmed(**changes) -> records.Confirmed:
    values = dict(
        settings_file="/srv/settings/rkmon-deploy.yaml",
        fingerprint="sha256:" + "a" * 64,
        at=AT,
        file_sha256="sha256:" + "b" * 64,
        package_version="1.2.3",
        code_sha256="sha256:" + "e" * 64,
        container_name="rocke-monitor",
        image=records.Image("", "rocke-monitor", "1.2.3"),
        changes=3,
    )
    values.update(changes)
    return records.Confirmed(**values)


FIX = Steps(
    (
        Edit("model_bindings.fast.base_url", "~/rkmon-deploy.yaml", "<url>"),
        "then",
        Command(("rkmon", "confirm")),
    )
)
PUT_BACK = records.PutBack(
    "sha256:" + "a" * 64,
    "doctor",
    LATER,
    "model fast: no answer from the binding's base_url",
    FIX,
)
DOCTOR = records.Doctor(LATER, 41, ("model fast", "session login: gh"))
WRITTEN = records.Written("sha256:" + "c" * 64, "none", "sha256:" + "d" * 64, "")
FIRST_TICK = records.FirstTick(LATER)


def mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


class Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.pkg = self.tmp / "pkg"
        (self.pkg / "docker").mkdir(parents=True)
        self.file = self.pkg / records.FILE

    def write_file(self, data) -> None:
        self.file.parent.mkdir(exist_ok=True)
        text = data if isinstance(data, str) else json.dumps(data)
        self.file.write_text(text, encoding="utf-8")
        os.chmod(self.file, 0o600)


class TestNames(unittest.TestCase):
    def test_the_records_are_a_package_file(self):
        self.assertIn(records.FILE, private_files.PACKAGE_FILES)
        for rel in (records.FILE, records.RECORDS_LOCK, records.UNDER_WAY_LOCK):
            self.assertEqual(posix_parent(rel), records.FOLDER, rel)
        self.assertEqual(records.FOLDER, private_files.RECORDS_FOLDER)


def posix_parent(rel: str) -> str:
    return rel.rsplit("/", 1)[0]


class TestReadAndWrite(Case):
    def test_no_file_no_records(self):
        self.assertEqual(records.read(self.pkg), records.Records())

    def test_read_back_as_written(self):
        records.confirm(self.pkg, confirmed())
        records.put_back(self.pkg, PUT_BACK)
        records.doctor_result(self.pkg, DOCTOR)
        records.change_written(self.pkg, lambda w: WRITTEN)
        records.first_tick(self.pkg, FIRST_TICK)
        records.begin(self.pkg, "apply").finish(code=True)
        installed = records.read(self.pkg).installed
        self.assertIsNotNone(records.parsed(installed.at))
        self.assertEqual(
            records.read(self.pkg),
            records.Records(
                confirmed(), None, PUT_BACK, DOCTOR, WRITTEN, FIRST_TICK, installed
            ),
        )
        self.assertEqual(mode(self.file), 0o600)
        self.assertEqual(mode(self.file.parent), 0o700)
        data = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(
            list(data),
            [
                "kind",
                "schema_version",
                "confirmed",
                "under_way",
                "put_back",
                "doctor",
                "written",
                "first_tick",
                "installed",
            ],
        )
        self.assertEqual(data["first_tick"], {"at": LATER})
        self.assertEqual(data["installed"], {"at": installed.at})
        self.assertEqual((data["kind"], data["schema_version"]), ("host_records", 1))
        self.assertEqual(
            data["put_back"]["fix"]["steps"][2], {"command": ["rkmon", "confirm"]}
        )
        self.assertEqual(data["doctor"]["failing"], ["model fast", "session login: gh"])
        self.assertEqual(
            data["confirmed"]["image"],
            {"registry": "", "name": "rocke-monitor", "tag": "1.2.3"},
        )

    def test_the_fingerprints_kept(self):
        # deliver.wrote's: CHANGE has those kept before, NOT_KEPT while none are, and the other records stay.
        records.confirm(self.pkg, confirmed())
        seen = []

        def change(w):
            seen.append(w)
            return dataclasses.replace(w, env=WRITTEN.env)

        records.change_written(self.pkg, change)
        records.change_written(self.pkg, change)
        self.assertEqual(
            seen, [records.NOT_KEPT, records.Written(WRITTEN.env, "", "", "")]
        )
        self.assertEqual(records.NOT_KEPT, records.Written("", "", "", ""))
        self.assertEqual(records.read(self.pkg).confirmed, confirmed())

    def test_records_an_older_package_wrote_keep_no_fingerprint(self):
        records.confirm(self.pkg, confirmed())
        data = json.loads(self.file.read_text(encoding="utf-8"))
        del data["written"]
        self.write_file(data)
        self.assertIsNone(records.read(self.pkg).written)
        self.assertEqual(records.read(self.pkg).confirmed, confirmed())

    def test_the_image_reference(self):
        image = records.Image("registry.example.com/team", "rocke-monitor", "1.2.3")
        self.assertEqual(image.reference, "rocke-monitor:1.2.3")

    def test_the_lock_files(self):
        records.confirm(self.pkg, confirmed())
        with records.begin(self.pkg, "apply"):
            pass
        for rel in (records.RECORDS_LOCK, records.UNDER_WAY_LOCK):
            path = self.pkg / rel
            self.assertEqual((path.read_bytes(), mode(path)), (b"", 0o600), rel)

    def test_a_field_besides_its_own_is_passed_over(self):
        records.confirm(self.pkg, confirmed())
        data = json.loads(self.file.read_text(encoding="utf-8"))
        data["confirmed"]["newer"] = 1
        data["newer_section"] = {}
        self.write_file(data)
        self.assertEqual(records.read(self.pkg).confirmed, confirmed())

    def test_unreadable(self):
        records.confirm(self.pkg, confirmed())
        records.put_back(self.pkg, PUT_BACK)
        good = json.loads(self.file.read_text(encoding="utf-8"))

        def changed(path, value):
            data = json.loads(json.dumps(good))
            where = data
            for key in path[:-1]:
                where = where[key]
            if value is KeyError:
                del where[path[-1]]
            else:
                where[path[-1]] = value
            return data

        for data, why in (
            ("{", "it is not JSON"),
            ("[]", 'its kind is not "host_records"'),
            (changed(("kind",), "deploy_settings"), 'its kind is not "host_records"'),
            (
                changed(("schema_version",), 2),
                "its layout is 2, and this package reads layout 1",
            ),
            (changed(("confirmed",), []), "its confirmed is no mapping"),
            (changed(("confirmed", "at"), KeyError), "its confirmed has no at"),
            (
                changed(("confirmed", "changes"), True),
                "its confirmed.changes is no int",
            ),
            (
                changed(("confirmed", "image"), "a:b"),
                "its confirmed.image is no mapping",
            ),
            (
                changed(("confirmed", "image", "tag"), KeyError),
                "its confirmed.image has no tag",
            ),
            (
                changed(("confirmed", "image", "registry"), False),
                "its confirmed.image.registry is no str",
            ),
            (changed(("put_back", "fix"), {"steps": 1}), "its put_back.fix is no fix"),
            (
                changed(("put_back", "reason"), "update"),
                "its put_back.reason is none of rollback, restart, load, doctor, first apply",
            ),
            (
                changed(("under_way",), {"command": "update", "since": AT}),
                "its under_way.command is none of apply, rollback, adopt",
            ),
            (
                changed(
                    ("under_way",), {"command": "apply", "since": AT, "installed": 1}
                ),
                "its under_way.installed is no bool",
            ),
            (
                changed(("under_way",), {"command": "apply"}),
                "its under_way has no since",
            ),
        ):
            with self.subTest(why=why):
                self.write_file(data)
                elsewhere = {"HOME": str(self.tmp / "home")}  # the file shows absolute
                with mock.patch.dict(os.environ, elsewhere):
                    with self.assertRaises(records.Unreadable) as caught:
                        records.read(self.pkg)
                fix = caught.exception.fix
                self.assertEqual(
                    str(caught.exception),
                    f"{self.file} is no host records this package reads: {why} (fix: {fix})",
                )
                self.assertEqual(
                    fix.parts,
                    (
                        Command(("mv", str(self.file), f"{self.file}.old")),
                        ", then",
                        Command((program.name(), "confirm", "--settings", "FILE")),
                    ),
                )

    def test_unreadable_under_the_home_folder(self):
        self.write_file("{")
        (self.file.parent / "records.json.old").write_text("")
        with mock.patch.dict(os.environ, {"HOME": str(self.pkg)}):
            with self.assertRaises(records.Unreadable) as caught:
                records.read(self.pkg)
        shown = "~/" + records.FILE
        self.assertEqual(
            caught.exception.fix.parts[0],
            Command(("mv", shown, f"{shown}-2.old")),
        )
        self.assertTrue(str(caught.exception).startswith(f"{shown} is no host records"))

    def test_the_reader_checks_a_private_file(self):
        records.confirm(self.pkg, confirmed())
        os.chmod(self.file, 0o644)
        with self.assertRaises(private_files.Refused):
            records.read(self.pkg)
        records.doctor_result(self.pkg, DOCTOR)  # a writer sets it to 600 first
        self.assertEqual(mode(self.file), 0o600)
        self.assertEqual(records.read(self.pkg).doctor, DOCTOR)
        elsewhere = self.tmp / "elsewhere.json"
        os.replace(self.file, elsewhere)
        self.file.symlink_to(elsewhere)
        for call in (
            lambda: records.read(self.pkg),
            lambda: records.doctor_result(self.pkg, DOCTOR),
        ):
            with self.assertRaises(private_files.Refused):
                call()
        self.assertTrue(self.file.is_symlink())


class TestConfirm(Case):
    def test_a_file_under_the_home_folder_is_kept_as_home_relative(self):
        home = self.tmp / "home"
        (home / "settings").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            records.confirm(
                self.pkg,
                confirmed(settings_file=str(home / "settings" / "rkmon-deploy.yaml")),
            )
        saved = records.read(self.pkg).confirmed.settings_file
        self.assertEqual(saved, "~/settings/rkmon-deploy.yaml")
        self.assertIn(
            '"~/settings/rkmon-deploy.yaml"', self.file.read_text(encoding="utf-8")
        )

    def test_a_file_elsewhere_is_kept_absolute(self):
        records.confirm(self.pkg, confirmed())
        self.assertEqual(
            records.read(self.pkg).confirmed.settings_file,
            "/srv/settings/rkmon-deploy.yaml",
        )

    def test_the_same_fingerprint_keeps_its_first_time(self):
        records.confirm(self.pkg, confirmed())
        again = confirmed(at=LATER, file_sha256="sha256:" + "c" * 64, changes=0)
        records.confirm(self.pkg, again)
        self.assertEqual(
            records.read(self.pkg).confirmed,
            confirmed(file_sha256=again.file_sha256, changes=0),
        )
        other = confirmed(fingerprint="sha256:" + "d" * 64, at=LATER)
        records.confirm(self.pkg, other)
        self.assertEqual(records.read(self.pkg).confirmed, other)

    def test_a_confirm_leaves_the_other_records(self):
        records.put_back(self.pkg, PUT_BACK)
        records.doctor_result(self.pkg, DOCTOR)
        records.confirm(self.pkg, confirmed())
        r = records.read(self.pkg)
        self.assertEqual((r.put_back, r.doctor), (PUT_BACK, DOCTOR))


class TestFirstTick(Case):
    """When deploy step 8 first passed after a start: the first command that sees a tick since a start
    keeps it, a later one changes nothing, and the deployment's volumes removed clear it.
    """

    def test_the_first_one_stays(self):
        records.confirm(self.pkg, confirmed())
        self.assertIsNone(records.read(self.pkg).first_tick)
        records.first_tick(self.pkg, FIRST_TICK)
        written = os.stat(self.file)
        kept = records.first_tick(self.pkg, records.FirstTick(AT))
        self.assertEqual(kept.first_tick, FIRST_TICK)
        self.assertEqual(records.read(self.pkg).first_tick, FIRST_TICK)
        # FILE is not written again: the same file, as it was.
        self.assertEqual(
            (os.stat(self.file).st_ino, os.stat(self.file).st_mtime_ns),
            (written.st_ino, written.st_mtime_ns),
        )
        self.assertEqual(records.read(self.pkg).confirmed, confirmed())

    def test_cleared(self):
        records.confirm(self.pkg, confirmed())
        records.first_tick(self.pkg, FIRST_TICK)
        records.clear_first_tick(self.pkg)
        self.assertIsNone(records.read(self.pkg).first_tick)
        self.assertEqual(records.read(self.pkg).confirmed, confirmed())
        records.first_tick(self.pkg, records.FirstTick(AT))
        self.assertEqual(records.read(self.pkg).first_tick, records.FirstTick(AT))

    def test_records_that_keep_none_are_not_written(self):
        # A fresh folder gets no records, and records that keep no first tick stay as they are.
        self.assertEqual(records.clear_first_tick(self.pkg), records.Records())
        self.assertFalse(os.path.lexists(self.file))
        records.confirm(self.pkg, confirmed())
        written = os.stat(self.file)
        records.clear_first_tick(self.pkg)
        self.assertEqual(
            (os.stat(self.file).st_ino, os.stat(self.file).st_mtime_ns),
            (written.st_ino, written.st_mtime_ns),
        )

    def test_records_an_older_package_wrote_keep_none(self):
        # Their layout is the same: a section they lack reads as none.
        records.confirm(self.pkg, confirmed())
        data = json.loads(self.file.read_text(encoding="utf-8"))
        del data["first_tick"]
        self.write_file(data)
        self.assertIsNone(records.read(self.pkg).first_tick)


class TestInstalled(Case):
    """Whether the last apply put a release in: its end keeps it, with the time; a command that ends with
    nothing changed leaves it, and so does an apply that puts none in and keeps no settings; an apply that puts none in
    and keeps the settings before it, and the code back (a rollback, an apply's put-back), clear it.
    """

    def test_kept_until_an_apply_puts_none_in_or_the_code_is_back(self):
        records.begin(self.pkg, "apply").finish(code=True)
        installed = records.read(self.pkg).installed
        self.assertIsNotNone(records.parsed(installed.at))
        self.assertFalse(records.begin(self.pkg, "rollback").stop(False))
        self.assertEqual(records.read(self.pkg).installed, installed)
        records.begin(self.pkg, "apply").finish(code=False)
        self.assertIsNone(records.read(self.pkg).installed)
        records.begin(self.pkg, "apply").finish(code=True)
        records.begin(self.pkg, "rollback").finish(code=False)
        self.assertIsNone(records.read(self.pkg).installed)

    def test_an_apply_s_end_keeps_what_its_change_did(self):
        # Lease.applied: a release went in, Installed now; else the settings before the change kept, none (its
        # rollback puts them back alone); else (the confirmed change delivered again, which keeps none) Installed as
        # it was, since the settings kept are still those from before the apply it records
        lease = records.begin(self.pkg, "apply")
        lease.record_install()
        lease.applied()
        self.assertIsNotNone(records.read(self.pkg).installed)
        data = json.loads(self.file.read_text(encoding="utf-8"))
        data["installed"] = {"at": AT}
        self.write_file(data)
        records.begin(self.pkg, "apply").applied()
        self.assertEqual(records.read(self.pkg).installed, records.Installed(AT))
        lease = records.begin(self.pkg, "apply")
        lease.record_kept()
        lease.applied()
        self.assertIsNone(records.read(self.pkg).installed)
        lease = records.begin(self.pkg, "apply")
        lease.record_kept()
        lease.record_install()
        lease.applied()
        installed = records.read(self.pkg).installed
        self.assertIsNotNone(installed)
        self.assertNotEqual(installed, records.Installed(AT))

    def test_records_an_older_package_wrote_keep_none(self):
        records.begin(self.pkg, "apply").finish(code=True)
        data = json.loads(self.file.read_text(encoding="utf-8"))
        del data["installed"]
        self.write_file(data)
        self.assertIsNone(records.read(self.pkg).installed)


class TestPutBack(Case):
    def test_put_back_and_cleared(self):
        records.put_back(self.pkg, PUT_BACK)
        self.assertEqual(records.read(self.pkg).put_back, PUT_BACK)
        records.clear_put_back(self.pkg)
        self.assertIsNone(records.read(self.pkg).put_back)

    def test_with_no_fix(self):
        rolled = records.PutBack(PUT_BACK.fingerprint, "rollback", LATER, "", None)
        records.put_back(self.pkg, rolled)
        self.assertEqual(records.read(self.pkg).put_back, rolled)

    def test_its_reasons(self):
        for reason in records.REASONS:
            records.put_back(
                self.pkg, records.PutBack("sha256:x", reason, AT, "", None)
            )
        with self.assertRaises(ValueError):
            records.put_back(
                self.pkg, records.PutBack("sha256:x", "other", AT, "", None)
            )
        self.assertEqual(records.read(self.pkg).put_back.reason, records.REASONS[-1])


HOLDER = """\
import sys
sys.path.insert(0, sys.argv[1])
from rkmon_lib import records
lease = records.begin(sys.argv[2], "apply")
print("held", flush=True)
sys.stdin.read()
"""


class TestUnderWay(Case):
    def test_finished(self):
        lease = records.begin(self.pkg, "apply")
        under_way = records.read(self.pkg).under_way
        self.assertEqual(under_way.command, "apply")
        self.assertIsNotNone(records.parsed(under_way.since))
        self.assertTrue(records.runs(self.pkg))
        lease.finish()
        self.assertIsNone(records.read(self.pkg).under_way)
        self.assertFalse(records.runs(self.pkg))
        with self.assertRaises(RuntimeError):
            lease.finish()

    def test_let_go_without_finishing(self):
        with records.begin(self.pkg, "rollback") as lease:
            self.assertTrue(records.runs(self.pkg))
        self.assertEqual(records.read(self.pkg).under_way.command, "rollback")
        self.assertFalse(records.runs(self.pkg))
        lease.release()  # again: nothing to let go

    def test_a_second_one_is_busy(self):
        with records.begin(self.pkg, "apply"):
            started = time.monotonic()
            with mock.patch.object(records, "BUSY_WAIT", 0.2):
                with self.assertRaises(records.Busy) as caught:
                    records.begin(self.pkg, "rollback")
            self.assertGreaterEqual(time.monotonic() - started, 0.2)
            self.assertEqual(caught.exception.command, "apply")
            self.assertEqual(
                str(caught.exception), "an apply runs in this package folder already"
            )
            self.assertEqual(records.read(self.pkg).under_way.command, "apply")

    def test_busy_names_the_folder_it_runs_in(self):
        self.assertEqual(
            str(records.Busy("rollback", "~/rocke_monitor")),
            "a rollback runs in ~/rocke_monitor already",
        )

    def test_a_look_is_waited_out(self):
        records.begin(self.pkg, "apply").finish()
        look = os.open(self.pkg / records.UNDER_WAY_LOCK, os.O_RDONLY)
        fcntl.flock(look, fcntl.LOCK_SH)  # as runs() holds it, a moment, to look
        done = threading.Timer(0.2, os.close, (look,))
        done.start()
        self.addCleanup(done.join)
        with records.begin(self.pkg, "rollback") as lease:
            self.assertEqual(records.read(self.pkg).under_way.command, "rollback")
            lease.finish()

    def test_one_that_stopped_part_way_is_taken_over(self):
        with records.begin(self.pkg, "apply"):
            pass
        with records.begin(self.pkg, "rollback") as lease:
            self.assertEqual(records.read(self.pkg).under_way.command, "rollback")
            lease.finish()
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_process_that_ends_lets_go(self):
        holder = subprocess.Popen(
            [sys.executable, "-B", "-I", "-c", HOLDER, str(ROOT), str(self.pkg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            self.assertEqual(holder.stdout.readline(), b"held\n")
            self.assertTrue(records.runs(self.pkg))
            with mock.patch.object(records, "BUSY_WAIT", 0.1):
                with self.assertRaises(records.Busy):
                    records.begin(self.pkg, "apply")
        finally:
            holder.kill()
            holder.wait()
            holder.stdin.close()
            holder.stdout.close()
        self.assertFalse(records.runs(self.pkg))
        self.assertEqual(records.read(self.pkg).under_way.command, "apply")

    def test_an_install_carries_on_until_the_change_ends(self):
        # recorded (Lease.record_install), it carries on into the command that takes the place of one that stopped
        # part-way, until one finishes
        with records.begin(self.pkg, "apply") as lease:
            self.assertFalse(lease.installed)
            self.assertFalse(records.read(self.pkg).under_way.installed)
            lease.record_install()
            self.assertTrue(lease.installed)
        self.assertTrue(records.read(self.pkg).under_way.installed)
        with records.begin(self.pkg, "rollback") as lease:
            self.assertTrue(lease.installed)
            under_way = records.read(self.pkg).under_way
            self.assertEqual(
                (under_way.command, under_way.installed), ("rollback", True)
            )
            lease.finish()
        with records.begin(self.pkg, "apply") as lease:
            self.assertFalse(lease.installed)
            lease.finish()
        with self.assertRaises(RuntimeError):
            lease.record_install()
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_keep_carries_on_until_the_change_ends(self):
        # recorded (Lease.record_kept), it carries on into the command that takes the place of one that stopped
        # part-way, so that the apply that finishes the change clears Installed (Lease.applied), until one finishes
        lease = records.begin(self.pkg, "apply")
        lease.record_install()
        lease.applied()
        with records.begin(self.pkg, "apply") as lease:
            self.assertFalse(lease.kept)
            self.assertFalse(records.read(self.pkg).under_way.kept)
            lease.record_kept()
            self.assertTrue(lease.kept)
        self.assertTrue(records.read(self.pkg).under_way.kept)
        with records.begin(self.pkg, "apply") as lease:
            self.assertTrue(lease.kept)
            self.assertTrue(records.read(self.pkg).under_way.kept)
            lease.applied()
        self.assertIsNone(records.read(self.pkg).installed)
        with records.begin(self.pkg, "apply") as lease:
            self.assertFalse(lease.kept)
            lease.finish()
        with self.assertRaises(RuntimeError):
            lease.record_kept()

    def test_the_first_baseline_carries_on_until_the_change_ends(self):
        # the doctor's baseline (begin's failing) is kept in the record; the command that takes the place of one that
        # stopped part-way keeps it over its own (records.baseline), until one finishes; a command with no baseline
        # of its own (the doctor did not run) keeps it too
        with records.begin(self.pkg, "apply", ["ramdisk"]) as lease:
            self.assertEqual(lease.failing, ("ramdisk",))
        under_way = records.read(self.pkg).under_way
        self.assertEqual((under_way.baseline, under_way.failing), (True, ("ramdisk",)))
        with records.begin(self.pkg, "apply", ["ramdisk", "model fast"]) as lease:
            self.assertEqual(lease.failing, ("ramdisk",))
        with records.begin(self.pkg, "rollback") as lease:
            self.assertEqual(lease.failing, ("ramdisk",))
            lease.finish()
        with records.begin(self.pkg, "apply") as lease:
            self.assertIsNone(lease.failing)
            under_way = records.read(self.pkg).under_way
            self.assertEqual((under_way.baseline, under_way.failing), (False, ()))
        with records.begin(self.pkg, "apply", []) as lease:
            self.assertEqual(lease.failing, ())
            lease.finish()
        with records.begin(self.pkg, "apply", []) as lease:
            pass
        with records.begin(self.pkg, "apply", ["model fast"]) as lease:
            self.assertEqual(lease.failing, ())
            lease.finish()

    def test_a_rollback_s_code_back_carries_on_until_it_ends(self):
        # begin records CODE_BACK as given. A rollback decides by the records (records.code_back): after a rollback
        # that stopped part-way, as that one decided, whatever the last apply did; after an apply that stopped
        # part-way, whether a release went in during it; else whether the last apply put one in (Installed)
        self.assertFalse(records.code_back(records.read(self.pkg)))
        records.begin(self.pkg, "apply").finish(code=True)
        self.assertTrue(records.code_back(records.read(self.pkg)))
        with records.begin(self.pkg, "rollback", [], code_back=False):
            pass
        found = records.read(self.pkg)
        self.assertEqual(
            (found.under_way.command, found.under_way.code_back), ("rollback", False)
        )
        self.assertFalse(records.code_back(found))
        with records.begin(self.pkg, "rollback", [], records.code_back(found)):
            self.assertIs(records.read(self.pkg).under_way.code_back, False)
        records.begin(self.pkg, "rollback", [], code_back=True).finish(code=False)
        self.assertFalse(records.code_back(records.read(self.pkg)))
        with records.begin(self.pkg, "rollback", [], code_back=True):
            pass
        self.assertTrue(records.code_back(records.read(self.pkg)))
        with records.begin(self.pkg, "apply") as lease:
            self.assertIs(records.read(self.pkg).under_way.code_back, False)
            self.assertFalse(records.code_back(records.read(self.pkg)))
            lease.record_install()
            self.assertTrue(records.code_back(records.read(self.pkg)))

    def test_an_apply_s_put_back_puts_the_code_back_by_the_records(self):
        # while an apply is under way: whether a release went in during it; else, when its delivery kept no settings
        # of its own, whether the last apply put one in (Installed), since the settings kept are those from before it
        with records.begin(self.pkg, "apply"):
            self.assertFalse(records.code_back(records.read(self.pkg)))
        lease = records.begin(self.pkg, "apply")
        lease.record_install()
        self.assertTrue(records.code_back(records.read(self.pkg)))
        lease.applied()
        with records.begin(self.pkg, "apply") as lease:
            self.assertTrue(records.code_back(records.read(self.pkg)))
            lease.record_kept()
            self.assertFalse(records.code_back(records.read(self.pkg)))
            lease.record_install()
            self.assertTrue(records.code_back(records.read(self.pkg)))

    def test_a_command_that_stops_after_one_part_way_leaves_the_record(self):
        # stop(changed): as finish() when nothing changed and the change was not part-way when the lease was taken,
        # else as release()
        lease = records.begin(self.pkg, "apply")
        self.assertFalse(lease.part_way)
        self.assertFalse(lease.stop(False))
        self.assertIsNone(records.read(self.pkg).under_way)
        self.assertTrue(records.begin(self.pkg, "apply").stop(True))
        self.assertEqual(records.read(self.pkg).under_way.command, "apply")
        lease = records.begin(self.pkg, "rollback")
        self.assertTrue(lease.part_way)
        self.assertTrue(lease.stop(False))
        self.assertEqual(records.read(self.pkg).under_way.command, "rollback")
        self.assertFalse(records.runs(self.pkg))

    def test_a_record_without_the_install_reads_as_none(self):
        with records.begin(self.pkg, "apply", ["ramdisk"]):
            pass
        data = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertIs(data["under_way"]["installed"], False)
        self.assertEqual(
            (data["under_way"]["baseline"], data["under_way"]["failing"]),
            (True, ["ramdisk"]),
        )
        self.assertIs(data["under_way"]["code_back"], False)
        self.assertIs(data["under_way"]["kept"], False)
        self.assertIs(data["under_way"]["first_apply"], False)
        names = ("installed", "baseline", "failing", "code_back", "kept", "first_apply")
        for name in names:
            del data["under_way"][name]
        self.write_file(data)
        under_way = records.read(self.pkg).under_way
        self.assertEqual(
            (
                under_way.command,
                under_way.installed,
                under_way.baseline,
                under_way.failing,
                under_way.code_back,
                under_way.kept,
                under_way.first_apply,
            ),
            ("apply", False, False, (), False, False, False),
        )

    def test_a_first_apply_keeps_its_mark_until_it_ends(self):
        """The first apply over an older package marks its change as it begins
        it (begin's first_apply), the command that takes its place carries the mark on, and a first apply that failed
        keeps it in the change put back (FIRST_APPLY). records.first_apply reads the two, the record under way first:
        an adopt that stopped part-way after a failed first apply is none."""
        self.assertFalse(records.first_apply(records.read(self.pkg)))
        with records.begin(self.pkg, "apply", (), first_apply=True) as lease:
            self.assertTrue(lease.first_apply)
        kept = records.read(self.pkg)
        self.assertTrue(kept.under_way.first_apply)
        self.assertTrue(records.first_apply(kept))
        with records.begin(self.pkg, "apply") as lease:
            self.assertTrue(lease.first_apply)
            self.assertTrue(records.read(self.pkg).under_way.first_apply)
            lease.finish()
        self.assertFalse(records.first_apply(records.read(self.pkg)))
        with records.begin(self.pkg, "apply") as lease:
            self.assertFalse(lease.first_apply)
            self.assertFalse(records.read(self.pkg).under_way.first_apply)
            lease.finish()
        for reason in records.REASONS:
            with self.subTest(reason=reason):
                put_back = dataclasses.replace(PUT_BACK, reason=reason)
                records.put_back(self.pkg, put_back)
                self.assertIs(
                    records.first_apply(records.read(self.pkg)),
                    reason == records.FIRST_APPLY,
                )
        with records.begin(self.pkg, "adopt"):
            pass
        self.assertFalse(records.first_apply(records.read(self.pkg)))

    def test_an_adopt_takes_an_adopts_place_alone(self):
        """takes_place: a command that stopped part-way is taken over by itself, and an apply and a rollback by each
        other; an adopt by an adopt alone, which takes no apply's or rollback's place: PartWay, nothing written and
        the lock let go. A rollback after an adopt that stopped part-way decides by the last apply alone (code_back).
        """
        for command, part_way, takes in (
            ("apply", "apply", True),
            ("rollback", "apply", True),
            ("apply", "rollback", True),
            ("adopt", "adopt", True),
            ("adopt", "apply", False),
            ("adopt", "rollback", False),
            ("apply", "adopt", False),
            ("rollback", "adopt", False),
        ):
            with self.subTest(command=command, part_way=part_way):
                self.assertIs(records.takes_place(command, part_way), takes)
        lease = records.begin(self.pkg, "apply")
        lease.record_install()
        lease.applied()
        with records.begin(self.pkg, "adopt"):
            pass
        found = records.read(self.pkg)
        self.assertEqual(found.under_way.command, "adopt")
        self.assertTrue(records.code_back(found))
        before = self.file.read_bytes()
        for command in ("apply", "rollback"):
            with self.subTest(command=command):
                with self.assertRaises(records.PartWay) as caught:
                    records.begin(self.pkg, command)
                self.assertEqual(caught.exception.command, "adopt")
                self.assertEqual(
                    str(caught.exception),
                    "an adopt stopped part-way in this package folder",
                )
                self.assertEqual(self.file.read_bytes(), before)
                self.assertFalse(records.runs(self.pkg))
        with records.begin(self.pkg, "adopt") as lease:
            self.assertTrue(lease.part_way)
            lease.finish()
        self.assertIsNone(records.read(self.pkg).under_way)
        records.begin(self.pkg, "apply").release()
        with self.assertRaises(records.PartWay) as caught:
            records.begin(self.pkg, "adopt")
        self.assertEqual(
            str(caught.exception), "an apply stopped part-way in this package folder"
        )

    def test_no_lock_file_runs_nothing(self):
        self.assertFalse(records.runs(self.pkg))

    def test_its_commands(self):
        with self.assertRaises(ValueError):
            records.begin(self.pkg, "start")
        self.assertFalse(os.path.lexists(self.pkg / records.UNDER_WAY_LOCK))


class TestTimes(unittest.TestCase):
    def test_a_time_and_back(self):
        when = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        self.assertEqual(records.timestamp(when), AT)
        self.assertEqual(records.parsed(AT), when)
        self.assertIsNone(records.parsed("13:58"))
        self.assertIsNotNone(records.parsed(records.timestamp()))


if __name__ == "__main__":
    unittest.main()
