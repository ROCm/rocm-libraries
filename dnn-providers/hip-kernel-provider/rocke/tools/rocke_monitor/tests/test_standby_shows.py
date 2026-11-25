# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How standby shows: while the monitor is in standby the doctor has its WARN row, with when standby began,
why, and what ends it, which is state 14's next command, and the admin dashboard's top line shows that row, every value
escaped; the team dashboard's copy never does. Neither while standby is off."""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import checks, doctor, report, state_store, states, webdash  # noqa: E402
from rkmon_lib.checks import Status  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

SINCE = "2026-11-21T07:38:08Z"  # invented, as the internal state keeps a time
REASON = "moved from an older install"
WRAPPER = {"RKMON_WRAPPER": "rkmon"}  # the wrapper's name, as it passes it in
# The doctor's row in standby, as checks.render prints it.
ROW = (
    "  WARN  standby: on since 2026-11-21 07:38 UTC (moved from an older install): nothing is sent\n"
    "        fix: rkmon standby off, once rkmon status and the dashboards read right"
)


class Home(unittest.TestCase):
    """A monitor home of the test's own, whose internal state the test writes."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = pathlib.Path(tmp.name)
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def standby(self, on: bool = True, since: str = SINCE, reason: str = REASON):
        for name, value in (
            ("standby.on", on),
            ("standby.since", since),
            ("standby.reason", reason),
        ):
            state_store.set(self.home, name, value)


class TestDoctorRow(Home):
    def test_the_row_while_standby_is_on(self):
        self.standby()
        rows = doctor.standby_checks(environ=WRAPPER)
        self.assertEqual(checks.render(rows), ROW)
        self.assertEqual([r.status for r in rows], [Status.WARN])

    def test_no_row_while_standby_is_off(self):
        self.assertEqual(doctor.standby_checks(environ=WRAPPER), [])  # no state yet
        self.standby(on=False)
        self.assertEqual(doctor.standby_checks(environ=WRAPPER), [])

    def test_its_fix_is_state_14s_next_command(self):
        """One home for what ends standby: the doctor's fix and state 14's next command are the same steps, the
        commands as the operator types them here (the wrapper's name, which it passes in).
        """
        self.standby()
        (row,) = doctor.standby_checks(environ={})
        (fourteen,) = [s for s in states.TABLE if s.n == 14]
        self.assertEqual(row.fix, fourteen.next_command(None))
        (row,) = doctor.standby_checks(environ={"RKMON_WRAPPER": "./rkmon"})
        self.assertEqual(
            str(row.fix),
            "./rkmon standby off, once ./rkmon status and the dashboards read right",
        )

    def test_a_blank_time_or_reason_is_left_out(self):
        for since, reason, found in (
            ("", REASON, "on (moved from an older install): nothing is sent"),
            (SINCE, "", "on since 2026-11-21 07:38 UTC: nothing is sent"),
            ("", "", "on: nothing is sent"),
            # a time in another form is shown as it is
            ("yesterday", "", "on since yesterday: nothing is sent"),
        ):
            with self.subTest(since=since, reason=reason):
                self.standby(since=since, reason=reason)
                (row,) = doctor.standby_checks(environ=WRAPPER)
                self.assertEqual(row.found, found)

    def test_the_features_rows_carry_it_after_their_headline(self):
        names = [c.name for c in doctor.feature_checks({})]
        self.assertNotIn("standby", names)
        self.standby()
        rows = doctor.feature_checks({})
        self.assertEqual([c.name for c in rows[:2]], ["features", "standby"])
        self.assertEqual(rows[1].status, Status.WARN)


class TestAdminTopLine(Home):
    def page(self, variant: str) -> str:
        state = self.home / "state"
        state.mkdir(parents=True, exist_ok=True)
        store = Store(state)
        cfg = {
            "operator": {"github_login": "me"},
            "repo": "o/r",
            "codeowner_teams": {"platform": ["plat"]},
            "paths": {
                "html": str(self.home / "x.html"),
                "daily_dir": str(self.home),
                "working_dir": str(self.home),
                "root": str(self.home),
            },
        }
        st = empty_state()
        board = report.board(cfg, st, store, [], [])
        with mock.patch.dict(os.environ, WRAPPER):
            return webdash.render(cfg, st, store, [], [], board, variant)

    def test_the_admin_dashboards_top_line_while_standby_is_on(self):
        self.standby()
        local = self.page(webdash.LOCAL)
        header = local[local.index("<header") : local.index("<nav")]
        self.assertIn("<p class='standby' role='status'>", header)
        self.assertIn(
            "⚠ standby: on since 2026-11-21 07:38 UTC (moved from an older install): nothing is sent</span> · "
            "fix: rkmon standby off, once rkmon status and the dashboards read right</p>",
            header,
        )
        self.assertNotIn("class='standby'", self.page(webdash.PUBLISHED))

    def test_its_values_are_escaped(self):
        self.standby(reason="<b>moved</b> & kept")
        local = self.page(webdash.LOCAL)
        self.assertIn("(&lt;b&gt;moved&lt;/b&gt; &amp; kept): nothing is sent", local)
        self.assertNotIn("<b>moved</b>", local)

    def test_no_line_while_standby_is_off(self):
        self.assertNotIn("class='standby'", self.page(webdash.LOCAL))
        self.standby(on=False)
        self.assertNotIn("class='standby'", self.page(webdash.LOCAL))


if __name__ == "__main__":
    unittest.main()
