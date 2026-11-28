# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The table of the commands that send (rkmon_lib/outward.py): each row one rk command, its columns of one form, each
feature a capability; and the writer's broker loads it, through rkmon_lib/writer.py, with the standard library only.
The layers that read it are tested where they are: the stop (test_control, test_writer), a mirror's refusal
(test_mirror), the feature gates (test_write_gates), the writer's commands (test_writer, test_gh_split), the session's
guard (test_orchestrator).
"""

import contextlib
import io
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkmon  # noqa: E402
from rkmon_lib import features, outward  # noqa: E402


class TestTheTable(unittest.TestCase):
    def test_each_row_is_one_rk_command(self):
        names = [r.name for r in outward.TABLE]
        self.assertEqual(len(names), len(set(names)))
        quiet = io.StringIO()
        for name in names:
            # its help: rk's parser takes it
            with self.subTest(cmd=name), contextlib.redirect_stdout(quiet):
                with contextlib.redirect_stderr(quiet), self.assertRaises(
                    SystemExit
                ) as cm:
                    rkmon.main([name, "--help"])
                self.assertEqual(cm.exception.code, 0)
            self.assertEqual(outward.row(name).name, name)
        self.assertIsNone(outward.row("status"))

    def test_each_column_has_one_of_its_values(self):
        for r in outward.TABLE:
            with self.subTest(cmd=r.name):
                self.assertIn(
                    r.dest,
                    (
                        outward.GITHUB,
                        outward.TEAMS,
                        outward.WEB,
                        outward.JIRA,
                        outward.NONE,
                    ),
                )
                self.assertIn(r.gate, (outward.OWN, outward.GATED, outward.NONE))
                # a send meets one gate; a command that sends nothing meets none
                self.assertEqual(r.gate == outward.NONE, r.dest == outward.NONE)
                self.assertIn(r.writer, (outward.SUDO, outward.HOST, ""))
                self.assertIn(r.handler, (outward.HALF, outward.POST_LOCKED, ""))
                for name in r.features:
                    self.assertIn(name, features.CAPABILITIES)
                # the outward gate's words for what is sent: a GATED send's alone
                self.assertEqual(bool(r.what), r.gate == outward.GATED)
                # the writer's half runs as the writer, sent there through sudo
                if r.handler == outward.HALF:
                    self.assertEqual(r.writer, outward.SUDO)
                if r.message:
                    self.assertEqual(r.dest, outward.TEAMS)
                if r.protective:
                    self.assertEqual(r.dest, outward.GITHUB)
                self.assertEqual(r.stops, r.dest != outward.NONE and not r.protective)
                self.assertIn(r.session, (True, False))
                # a row that neither sends nor reaches the writer is there for the session's guard alone
                if r.dest == outward.NONE and not r.writer:
                    self.assertFalse(r.session)

    def test_names_picks_rows_by_their_columns(self):
        self.assertEqual(outward.names(protective=True), ("disarm", "automerge-off"))
        self.assertEqual(
            outward.names(gate=outward.GATED),
            (outward.TEAMS_SEND, outward.WEB_PUSH, outward.ATLASSIAN),
        )
        self.assertEqual(
            outward.names(dest=outward.NONE, handler=outward.HALF),
            (outward.WEB_REQUESTS,),
        )
        with self.assertRaises(AttributeError):
            outward.names(destination=outward.WEB)

    def test_the_broker_loads_it_with_the_standard_library_only(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import writer; "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('rkmon_lib'))))"
        )
        p = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(
            p.stdout.split(),
            ["rkmon_lib", "rkmon_lib.gh", "rkmon_lib.outward", "rkmon_lib.writer"],
        )


class TestOwnWrites(unittest.TestCase):
    def test_a_write_is_the_monitors_own_only_with_the_mark_at_one(self):
        """A write a person typed carries no mark; the mark's word is an option, so no rk command and no
        writer's command reads as it."""
        self.assertTrue(outward.own({outward.OWN_VAR: "1"}))
        for environ in ({}, {outward.OWN_VAR: ""}, {outward.OWN_VAR: "0"}):
            with self.subTest(environ=environ):
                self.assertFalse(outward.own(environ))
        self.assertTrue(outward.OWN_WORD.startswith("--"))
        self.assertIsNone(outward.row(outward.OWN_WORD))


if __name__ == "__main__":
    unittest.main()
