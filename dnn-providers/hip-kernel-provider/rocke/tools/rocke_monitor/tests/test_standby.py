# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk standby-off` (rkmon_lib/standby.py), which `rkmon standby off` runs in the running container:
while the monitor is in standby, the one-time move from an older install ended when one is open, its lines first, then
standby.on cleared and RELEASED, exit 0; NOTHING TO DO while it is not in standby, nothing written. Standby stays on
when the move cannot end or standby.on cannot be cleared: REFUSED with the fix, exit 4, when nothing changed; ERROR
with the fix, exit 1, after the lines of what the end did, and standby off, run again, ends it. The mark of an
adoption goes last; while it says that rkmon adopt stopped part-way, standby off refuses, its fix adopt again, however
rk standby-off is run. rk's parser runs it.
"""

import contextlib
import errno
import io
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import adopt, features, standby, state_store, states, steps  # noqa: E402
from rkmon_lib.checks import FIX_LABEL  # noqa: E402

# The move's own tests' fixture: the container's layout in a folder of the test's, the clock the test's.
from test_one_time_move import (
    LATER,
    MOVE,
    RC,
    STANDBY_OFF,
    Base,
    until_line,
)  # noqa: E402

RELEASED = f"RELEASED: {states.RELEASED}"
NOTHING_TO_DO = f"NOTHING TO DO: {states.NOT_IN_STANDBY}\n"


def one_line(text: str) -> str:
    """TEXT's words one space apart, wherever outcome.finish broke its lines."""
    return " ".join(text.split())


def mark(home, applied_at: str = "") -> None:
    """The mark of an adoption in HOME, as rkmon adopt's apply writes it: APPLIED_AT "" until apply has finished."""
    state_store.write_file(
        adopt.mark_file(home), {"kind": adopt.MARK_KIND, "applied_at": applied_at}
    )


class OffTest(Base):
    def setUp(self):
        super().setUp()
        self.layer()

    def off(self, cfg=None) -> tuple:
        """rk standby-off, with CFG (the configuration the monitor runs, by default): its exit code and its output."""
        out = io.StringIO()
        cfg = RC.load(self.home).values if cfg is None else cfg
        return standby.off(cfg, self.home, out), out.getvalue()

    def moved(self) -> bytes:
        """The one-time move from an older install, made by the start: open, the monitor in standby. config.json's
        content."""
        data = self.older_config()
        self.start()
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(features.standby(self.home))
        return data

    def renamed(self) -> str:
        return f"renamed {self.config_file} {MOVE.MIGRATED}, read-only"

    def test_with_the_move_open_it_ends_the_move_then_standby(self):
        data = self.moved()
        self.clock = LATER
        rc, out = self.off()
        self.assertEqual(rc, 0, out)
        lines = out.splitlines()
        self.assertEqual(lines[0], self.renamed())
        self.assertIn(until_line(LATER), lines)
        self.assertEqual(lines[-1], RELEASED)
        self.assertEqual((self.home / MOVE.MIGRATED).read_bytes(), data)
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))
        self.assertEqual(self.until(), LATER)
        self.assertIs(state_store.read(self.home)["standby"]["on"], False)
        self.assertFalse(features.standby(self.home))
        # Once released: nothing to do, nothing written.
        before = self.state_file().read_bytes()
        self.assertEqual(self.off(), (0, NOTHING_TO_DO))
        self.assertEqual(self.state_file().read_bytes(), before)

    def test_with_no_move_it_ends_standby(self):
        """Standby set with no move from an older install, as rkmon adopt sets it."""
        state_store.set(self.home, "standby.on", True)
        self.assertTrue(features.standby(self.home))
        self.assertEqual(self.off(), (0, RELEASED + "\n"))
        self.assertIs(state_store.read(self.home)["standby"]["on"], False)
        self.assertFalse(features.standby(self.home))
        self.assertIsNone(MOVE.record(self.home))

    def test_nothing_to_do_while_the_monitor_is_not_in_standby(self):
        self.assertEqual(self.off(), (0, NOTHING_TO_DO))
        self.assertFalse(self.state_file().exists())

    def test_a_move_not_made_yet_is_refused_with_its_fix(self):
        data = self.older_config()
        self.assertTrue(MOVE.due(self.home))
        rc, out = self.off()
        self.assertEqual(rc, 4, out)
        stop, start = steps.command_line("stop"), steps.command_line("start")
        self.assertEqual(
            one_line(out),
            "REFUSED: the one-time move cannot end: it is not made yet; the monitor's start makes it; standby stays "
            f"on; nothing changed {FIX_LABEL}restart the monitor, whose start makes the move: {stop}, then {start}; "
            f"then run {STANDBY_OFF}",
        )
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertFalse(self.state_file().exists())
        self.assertTrue(features.standby(self.home))

    def test_a_configuration_that_names_no_secrets_folder_is_refused_with_its_fix(
        self,
    ):
        data = self.moved()
        before = self.state_file().read_bytes()
        rc, out = self.off(dict(RC.load(self.home).values, paths={}))
        self.assertEqual(rc, 4, out)
        doctor = steps.command_line("doctor")
        self.assertEqual(
            one_line(out),
            "REFUSED: the one-time move cannot end: the configuration names no secrets folder, so the older copies of "
            f"the secrets cannot be found; standby stays on; nothing changed {FIX_LABEL}run {doctor} and put right "
            f"what it says, then run {STANDBY_OFF}",
        )
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertEqual(self.state_file().read_bytes(), before)
        self.assertTrue(features.standby(self.home))

    def test_a_config_json_that_cannot_be_renamed_is_refused_with_its_fix(self):
        data = self.moved()
        real = os.replace

        def replace(src, dst, *args, **kwargs):
            if pathlib.Path(src) == self.config_file:
                raise PermissionError(errno.EACCES, "Permission denied", str(src))
            return real(src, dst, *args, **kwargs)

        with mock.patch.object(os, "replace", side_effect=replace):
            rc, out = self.off()
        self.assertEqual(rc, 4, out)
        self.assertEqual(
            one_line(out),
            f"REFUSED: the one-time move cannot end: {self.config_file} cannot be renamed {MOVE.MIGRATED}: "
            f"Permission denied; standby stays on; nothing changed {FIX_LABEL}make {self.home} writable for the "
            f"monitor, then run {STANDBY_OFF}",
        )
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertTrue(features.standby(self.home))

    def test_a_lock_that_cannot_be_taken_is_refused_with_its_fix(self):
        self.moved()
        folder = self.state_file().parent
        lock = folder / state_store.LOCK_FILE
        denied = PermissionError(errno.EACCES, "Permission denied", str(lock))
        with mock.patch.object(state_store, "locked_in", side_effect=denied):
            rc, out = self.off()
        self.assertEqual(rc, 4, out)
        self.assertEqual(
            one_line(out),
            f"REFUSED: the one-time move cannot end: the internal state's lock cannot be taken in {folder}: "
            f"Permission denied; standby stays on; nothing changed {FIX_LABEL}make {folder} writable for the "
            f"monitor, then run {STANDBY_OFF}",
        )
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(features.standby(self.home))

    def test_an_end_stopped_part_way_says_what_it_did_and_standby_off_again_ends_it(
        self,
    ):
        self.moved()
        real = state_store.write_file
        record = MOVE.record_file(self.home)

        def write_file(file, data):
            if pathlib.Path(file) == record:
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        self.clock = LATER
        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            rc, out = self.off()
        self.assertEqual(rc, 1, out)
        lines = out.splitlines()
        self.assertEqual(lines[0], self.renamed())
        self.assertIn(until_line(LATER), lines)
        self.assertTrue(
            one_line(out).endswith(
                f"ERROR: the record of the one-time move cannot be written in {record.parent}: Permission denied; "
                f"standby stays on {FIX_LABEL}make {record.parent} writable for the monitor, then run {STANDBY_OFF}"
            ),
            out,
        )
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(features.standby(self.home))
        rc, out = self.off()
        self.assertEqual(rc, 0, out)
        self.assertEqual(
            out.splitlines()[0], f"{self.config_file} is not there: nothing is renamed"
        )
        self.assertEqual(out.splitlines()[-1], RELEASED)
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))
        self.assertFalse(features.standby(self.home))

    def test_standby_that_cannot_be_cleared_after_the_end_is_an_error(self):
        """The move ended, and standby.on cannot be written: ERROR after the end's lines, with the internal state's
        fix; standby off, run again, has no move to end, and ends standby."""
        self.moved()
        real = state_store.set
        denied = PermissionError(
            errno.EACCES, "Permission denied", str(self.state_file())
        )

        def refuse(home, name, value, *args, **kwargs):
            if name == "standby.on":
                raise denied
            return real(home, name, value, *args, **kwargs)

        with mock.patch.object(state_store, "set", side_effect=refuse):
            rc, out = self.off()
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.splitlines()[0], self.renamed())
        unwritable = state_store.unwritable(self.home, denied)
        self.assertTrue(
            one_line(out).endswith(
                f"ERROR: {unwritable}; standby stays on {FIX_LABEL}{unwritable.fix('./rkmon')}"
            ),
            out,
        )
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(features.standby(self.home))
        self.assertEqual(self.off(), (0, RELEASED + "\n"))
        self.assertFalse(features.standby(self.home))

    def test_standby_that_cannot_be_cleared_with_no_move_is_refused(self):
        state_store.set(self.home, "standby.on", True)
        before = self.state_file().read_bytes()
        denied = PermissionError(
            errno.EACCES, "Permission denied", str(self.state_file())
        )
        with mock.patch.object(state_store, "write_file", side_effect=denied):
            rc, out = self.off()
        self.assertEqual(rc, 4, out)
        unwritable = state_store.unwritable(self.home, denied)
        self.assertEqual(
            one_line(out),
            f"REFUSED: {unwritable}; standby stays on; nothing changed {FIX_LABEL}{unwritable.fix('./rkmon')}",
        )
        self.assertEqual(self.state_file().read_bytes(), before)
        self.assertTrue(features.standby(self.home))

    def test_an_internal_state_that_cannot_be_read_keeps_standby_on(self):
        """An internal state that cannot be read counts as standby (features.standby): standby off refuses with the
        internal state's own fix, and changes nothing."""
        cfg = RC.load(self.home).values
        self.state_file().parent.mkdir(parents=True, exist_ok=True)
        self.state_file().write_text("{not json", encoding="utf-8")
        self.assertTrue(features.standby(self.home))
        with self.assertRaises(state_store.StateError) as caught:
            state_store.read(self.home)
        rc, out = self.off(cfg)
        self.assertEqual(rc, 4, out)
        fix = caught.exception.fix("./rkmon")
        said = f"REFUSED: {caught.exception}; standby stays on; nothing changed"
        if fix is not None:
            said += f" {FIX_LABEL}{fix}"
        self.assertEqual(one_line(out), one_line(said))
        self.assertEqual(self.state_file().read_text(encoding="utf-8"), "{not json")

    def test_an_adoption_that_has_not_finished_is_refused_with_its_fix(self):
        """While the mark of an adoption says that rkmon adopt stopped part-way (adopt.unfinished), standby off
        refuses, nothing changed, its fix adopt again: the move stays open and the mark stays. rk standby-off checks
        it itself, whatever runs it."""
        data = self.moved()
        mark(self.home)
        before = self.state_file().read_bytes()
        adopt_cmd, again = steps.command_line("adopt"), steps.command_line(
            "adopt --from DIR"
        )
        said = (
            f"REFUSED: {adopt_cmd} stopped part-way; standby stays on; nothing changed {FIX_LABEL}run {again} "
            "again"
        )
        rc, out = self.off()
        self.assertEqual((rc, one_line(out)), (4, said))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(rkmon.main(["standby-off"]), 4)
        self.assertEqual(one_line(out.getvalue()), said)
        self.assertEqual(self.state_file().read_bytes(), before)
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(adopt.marked(self.home))
        self.assertTrue(features.standby(self.home))

    def test_the_mark_of_a_finished_adoption_goes_last(self):
        """Once rkmon adopt has finished, its mark holds standby until standby off removes it, last: standby.on that
        cannot be cleared after the end leaves it; a mark that cannot be removed is an ERROR, standby on still, with
        its fix; standby off, run again, removes it and ends standby."""
        self.moved()
        mark(self.home, LATER)
        real = state_store.set
        denied = PermissionError(
            errno.EACCES, "Permission denied", str(self.state_file())
        )

        def refuse(home, name, value, *args, **kwargs):
            if name == "standby.on":
                raise denied
            return real(home, name, value, *args, **kwargs)

        with mock.patch.object(state_store, "set", side_effect=refuse):
            rc, out = self.off()
        self.assertEqual(rc, 1, out)
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))
        self.assertTrue(adopt.marked(self.home))
        p = adopt.mark_file(self.home)
        unlink = pathlib.Path.unlink

        def kept(path, *args, **kwargs):
            if path == p:
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return unlink(path, *args, **kwargs)

        with mock.patch.object(pathlib.Path, "unlink", autospec=True, side_effect=kept):
            rc, out = self.off()
        self.assertEqual(rc, 1, out)
        self.assertEqual(
            one_line(out),
            f"ERROR: the mark of the adoption, {p}, cannot be removed: Permission denied; standby stays on "
            f"{FIX_LABEL}make {self.home} writable for the monitor, then run {STANDBY_OFF}",
        )
        self.assertIs(state_store.read(self.home)["standby"]["on"], False)
        self.assertTrue(features.standby(self.home))
        self.assertEqual(self.off(), (0, RELEASED + "\n"))
        self.assertFalse(adopt.marked(self.home))
        self.assertFalse(features.standby(self.home))

    def test_rk_standby_off_runs_it(self):
        self.moved()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(rkmon.main(["standby-off"]), 0)
        self.assertEqual(out.getvalue().splitlines()[0], self.renamed())
        self.assertEqual(out.getvalue().splitlines()[-1], RELEASED)
        self.assertFalse(features.standby(self.home))


if __name__ == "__main__":
    unittest.main()
