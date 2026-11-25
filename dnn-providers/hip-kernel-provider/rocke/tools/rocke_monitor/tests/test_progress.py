# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where a deployment or a build is on its path (rkmon_lib/progress.py): the report of a fresh
folder whose container name is taken; each state's rows and next line, which is the state's from what has passed since
the start (a deploy step's command with its step until a tick has run since then, and for good once this folder's
records keep one; in standby's state and state 16 the doctor, then the first tick) but on a fresh folder; step 7 from
the doctor's record; a file judged with --settings; the build path's five steps; a time as the rows show it; the report
while docker cannot say whose the name is (the container's steps unknown and why, the state's next line only where the
records decide it alone); step 5 before the first start, from what the confirmed settings need; `after step N`, which a
REDO need or a known TODO one gives, and a row not read does not; and the state's lines of rkmon status.
"""

import dataclasses
import datetime
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (  # noqa: E402
    outcome,
    program,
    progress,
    records,
    states,
    wrapping,
)
from rkmon_lib.checks import WIDTH, Check, Command, Status  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_states import CASES, FILE, RUNS  # noqa: E402

FIX = Command(("rkmon", "check"))
PASS = Check("docker", Status.PASS, "server 29.0.0")
WARN = Check("disk", Status.WARN, "12 GiB free", FIX)
FAIL = Check("container.name", Status.FAIL, "taken", FIX)
# the last doctor result since the start: every check passed
PASSED = records.Doctor("2026-11-15T09:15:00Z", 41, ())


def facts(**changes):
    return dataclasses.replace(RUNS, **changes)


# A note such as a taken container name gives: progress renders whatever note it is given.
NOTE = (
    "the container name {} is used by another deployment on this host: give container.name another value in your "
    "file"
)


def statuses(report):
    return [r.status for r in report.rows]


def found(report, n):
    return report.rows[n - 1].found


class Base(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(program, "name", return_value="rkmon")
        patcher.start()
        self.addCleanup(patcher.stop)


class TestFreshFolder(Base):
    def test_the_report_of_a_fresh_folder_whose_container_name_is_taken(self):
        f = facts(**CASES[2], container_name="rocke-monitor")
        report = progress.deploy(f, note=NOTE.format("rocke-monitor"))
        self.assertEqual(
            progress.render(report),
            "The deploy path (QUICK_DEPLOY_GUIDE.md):\n"
            "  1. TODO  template: no deploy settings file named yet\n"
            "  2. TODO  check: after step 1\n"
            "  3. TODO  confirm: after step 2\n"
            "  4. TODO  image: after step 3\n"
            "  5. TODO  login: after step 3\n"
            "  6. TODO  start: after step 3\n"
            "  7. TODO  doctor: after step 6\n"
            "  8. TODO  first tick: after step 6\n"
            "note: the container name rocke-monitor is used by another deployment on this host: give container.name "
            "another value\n"
            "      in your file\n"
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml\n"
            "(have a file already? rkmon next --settings FILE)",
        )

    def test_step_1s_command_alone_once_the_records_keep_a_first_tick(self):
        # Deploy step 8 passed after a start of this folder's, whose confirm is gone now.
        report = progress.deploy(facts(**CASES[2], first_tick_recorded=True))
        self.assertEqual(report.next, "next: rkmon template deploy ~/rkmon-deploy.yaml")
        self.assertEqual(
            report.also, "(have a file already? rkmon next --settings FILE)"
        )

    def test_without_a_note(self):
        report = progress.deploy(facts(**CASES[2]))
        self.assertEqual(report.notes, ())
        self.assertNotIn("note:", progress.render(report))

    def test_a_file_that_is_gone(self):
        report = progress.deploy(facts(file_exists=False))
        self.assertEqual(report.rows[0].status, progress.REDO)
        self.assertEqual(
            found(report, 1), f"the settings file you confirmed, {FILE}, is gone"
        )
        self.assertEqual(found(report, 2), "after step 1")
        self.assertEqual(
            report.next,
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml",
        )


class TestStateLines(Base):
    """What rkmon status says of the deployment's state, before its next line."""

    def test_the_states_sentence(self):
        f = facts(**CASES[11])
        self.assertEqual(
            progress.state_lines(f),
            [
                "state: a change waits: you confirmed 2 changes at 13:58, and the monitor runs the settings before them"
            ],
        )

    def test_wrapped_with_no_indent_and_its_fix(self):
        f = facts(**CASES[10], standby=True)
        lines = progress.state_lines(f)
        self.assertEqual(
            lines,
            [
                "state: the last change you confirmed was put back after a failed doctor check: model fast: no "
                "answer from the",
                "binding's base_url; in standby: it sends nothing until you run rkmon standby off",
                "fix: set model_bindings.fast.base_url in ~/rkmon-deploy.yaml to an address the container reaches",
            ],
        )
        self.assertTrue(all(len(line) <= WIDTH for line in lines))

    def test_a_note_and_a_state_not_known(self):
        # A fresh folder whose name is taken: state 2, with the note rkmon next gives it.
        f = facts(**CASES[2], container_name="rocke-monitor")
        self.assertEqual(
            progress.state_lines(f, note="the name is taken"),
            [
                "state: no deploy settings are confirmed in this package folder",
                "note: the name is taken",
            ],
        )
        # While docker cannot say whose the name is, a state the records do not decide alone is not known.
        why = "docker does not answer"
        self.assertEqual(
            progress.state_lines(facts(), unknown_why=why),
            ["state: unknown (docker does not answer)"],
        )
        self.assertEqual(
            progress.state_lines(facts(**CASES[5]), unknown_why=why),
            ["state: your settings file changed after the last confirm"],
        )


class TestStates(Base):
    """Each state's rows, and its next line."""

    def report(self, n, **kwargs):
        kwargs.setdefault("confirmed_once", True)
        kwargs.setdefault("logins_read", True)
        kwargs.setdefault("started", "2026-11-15 09:12 UTC")
        kwargs.setdefault("tick", "2026-11-15 09:20 UTC")
        kwargs.setdefault("doctor", PASSED)
        return progress.deploy(facts(**CASES[n]), **kwargs)

    def test_the_next_line_is_the_states(self):
        # From what has passed since the start: the doctor's result and the last tick since then.
        for n in CASES:
            if n == 2:
                continue
            for doctor, tick in (
                (PASSED, "2026-11-15 09:20 UTC"),
                (PASSED, ""),
                (None, None),
            ):
                with self.subTest(state=n, doctor=doctor, tick=tick):
                    f = facts(**CASES[n])
                    since = progress.since_start(doctor, tick)
                    self.assertEqual(
                        self.report(n, doctor=doctor, tick=tick).next,
                        states.next_line(states.current(f), f, since),
                    )

    def test_a_deploy_steps_command_carries_its_step_until_a_tick_has_run(self):
        # Before step 8 has passed since the start, the states whose next command is a deploy step's name it;
        # after it, the line is the command alone.
        for n, before in (
            (5, "next: step 3, confirm: rkmon confirm"),
            (7, "next: step 5, login: rkmon login"),
            (12, "next: step 6, start: rkmon start"),
        ):
            with self.subTest(state=n):
                self.assertEqual(self.report(n, tick="").next, before)
                self.assertEqual(
                    self.report(n).next, "next: " + before.split(": ", 2)[2]
                )

    def test_state_16_runs_your_confirmed_settings(self):
        report = self.report(16)
        self.assertEqual(statuses(report)[:6], ["DONE"] * 6)
        self.assertEqual(
            found(report, 3),
            "~/rkmon-deploy.yaml and this package are what you confirmed at 13:58",
        )
        self.assertEqual(found(report, 6), "running since 2026-11-15 09:12 UTC")
        self.assertEqual(found(report, 7), "41 checks passed at 2026-11-15 09:15 UTC")
        self.assertEqual(found(report, 8), "at 2026-11-15 09:20 UTC")
        self.assertEqual(report.next, "done: the monitor runs your confirmed settings")
        self.assertEqual(report.notes, ())

    def test_state_16_names_the_doctor_then_the_first_tick(self):
        # Until the doctor has passed since the start: step 7, its row TODO, with what failed when a check did.
        report = self.report(16, doctor=None, tick="")
        self.assertEqual(report.rows[6].status, progress.TODO)
        self.assertEqual(found(report, 7), progress.NOT_SINCE_START)
        self.assertEqual(report.next, "next: step 7, doctor: rkmon doctor")
        # Once a tick has run since the start, step 8 has passed: the doctor's command alone.
        report = self.report(16, doctor=None)
        self.assertEqual(report.rows[6].status, progress.TODO)
        self.assertEqual(report.next, "next: rkmon doctor")
        failed = records.Doctor(
            "2026-11-15T09:15:00Z", 41, ("model fast", "session login")
        )
        report = self.report(16, doctor=failed, tick="")
        self.assertEqual(report.rows[6].status, progress.TODO)
        self.assertEqual(
            found(report, 7),
            "2 of 41 checks failed at 2026-11-15 09:15 UTC: model fast and session login",
        )
        self.assertEqual(report.next, "next: step 7, doctor: rkmon doctor")
        # Then, while the first tick since the start is to come: step 8.
        report = self.report(16, tick="")
        self.assertEqual(report.rows[6].status, progress.DONE)
        self.assertEqual(report.rows[7].status, progress.TODO)
        self.assertEqual(found(report, 8), "none since the start")
        self.assertEqual(report.next, "next: step 8, first tick: rkmon first-tick")
        # A first tick nobody could read names no step.
        report = self.report(16, tick=None)
        self.assertEqual(found(report, 8), progress.UNKNOWN)
        self.assertEqual(report.next, "done: the monitor runs your confirmed settings")

    def test_standby_names_the_doctor_then_the_first_tick(self):
        # A start in standby is followed by step 7, then step 8, then standby's own line once a tick has run
        # since the start.
        for doctor, tick, want in (
            (None, "", "next: step 7, doctor: rkmon doctor"),
            (PASSED, "", "next: step 8, first tick: rkmon first-tick"),
            (
                PASSED,
                "2026-11-15 09:20 UTC",
                "next: rkmon standby off, once rkmon status and the dashboards read right",
            ),
        ):
            with self.subTest(doctor=doctor, tick=tick):
                self.assertEqual(self.report(14, doctor=doctor, tick=tick).next, want)

    def test_once_the_records_keep_a_first_tick_no_line_names_a_step(self):
        # Deploy step 8 passed after an earlier start; after this one, each line is the command alone, the walk of
        # steps 7 and 8 in standby and state 16 included.
        for n, doctor, want in (
            (7, None, "next: rkmon login"),
            (12, None, "next: rkmon start"),
            (14, None, "next: rkmon doctor"),
            (14, PASSED, "next: rkmon first-tick"),
            (16, None, "next: rkmon doctor"),
            (16, PASSED, "next: rkmon first-tick"),
        ):
            with self.subTest(state=n, doctor=doctor):
                f = facts(**CASES[n], first_tick_recorded=True)
                report = progress.deploy(
                    f, confirmed_once=True, logins_read=True, tick="", doctor=doctor
                )
                self.assertEqual(report.next, want)

    def test_the_doctor_after_the_start(self):
        # A container that does not run has no doctor result since its start: step 7 waits for step 6.
        report = self.report(12)
        self.assertEqual(report.rows[5].status, progress.REDO)
        self.assertEqual(
            (report.rows[6].status, found(report, 7)), (progress.TODO, "after step 6")
        )

    def test_state_1_asks_nothing_of_the_container(self):
        report = self.report(1)
        self.assertEqual(
            statuses(report), ["DONE", "DONE", "REDO", "DONE"] + ["TODO"] * 4
        )
        self.assertEqual(
            found(report, 3),
            "the container name rocke-monitor is used by another deployment on this host",
        )
        self.assertEqual([found(report, n) for n in (5, 6, 7, 8)], ["unknown"] * 4)
        text = progress.render(report)
        self.assertIn(
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rocke-monitor-2 (docker compose "
            "ls --all\n"
            "     lists the names in use)\n"
            "next: rkmon confirm --settings ~/rkmon-deploy.yaml",
            text,
        )

    def test_state_4_redoes_the_template(self):
        report = self.report(4)
        self.assertEqual(report.rows[0].status, progress.REDO)
        self.assertEqual(
            found(report, 1),
            "~/rkmon-deploy.yaml is in layout 1, older than this package's layout 2",
        )
        self.assertEqual(
            report.next, "next: rkmon migrate --settings ~/rkmon-deploy.yaml"
        )

    def test_state_5_redoes_the_confirm_or_does_it_first(self):
        report = self.report(5)
        self.assertEqual(report.rows[2].status, progress.REDO)
        self.assertEqual(
            found(report, 3), "your settings file changed after the last confirm"
        )
        # A deployment that runs keeps its later steps done.
        self.assertEqual(statuses(report)[3:6], ["DONE"] * 3)
        report = self.report(5, confirmed_once=False)
        self.assertEqual(report.rows[2].status, progress.TODO)
        self.assertEqual(found(report, 3), "not confirmed yet")
        self.assertEqual(report.next, "next: rkmon confirm")
        # Before the first start, the start waits for the confirm, REDO or TODO.
        for once in (True, False):
            f = facts(file_changed=True, container=states.NO_CONTAINER)
            report = progress.deploy(f, confirmed_once=once, logins_read=True)
            self.assertEqual(found(report, 6), "after step 3")
            self.assertEqual(found(report, 8), "after step 6")

    def test_state_6_holds_the_logins_back(self):
        report = self.report(6, logins_read=False)
        self.assertEqual(
            found(report, 4),
            "the image rocke-monitor:2026.11.15 is not on this host",
        )
        self.assertEqual(found(report, 5), "after step 4")
        self.assertEqual(report.next, "next: rkmon image load IMAGE_FILE")

    def test_state_7_and_8(self):
        report = self.report(7)
        self.assertEqual(
            found(report, 5),
            "your confirmed settings need claude and admin_access_token, which are missing or wrong",
        )
        self.assertEqual(report.next, "next: rkmon login")
        report = self.report(8)
        self.assertEqual(
            found(report, 6), "this deployment's container does not exist yet"
        )
        self.assertEqual(found(report, 7), "after step 6")
        self.assertEqual(report.next, "next: rkmon start")

    def test_logins_the_container_did_not_list_hold_nothing_back(self):
        report = self.report(8, logins_read=False)
        self.assertEqual(found(report, 5), progress.LOGINS_NOT_READ)
        self.assertEqual(
            found(report, 6), "this deployment's container does not exist yet"
        )
        self.assertEqual(report.next, "next: rkmon start")

    def test_state_12_redoes_the_start(self):
        report = self.report(12)
        self.assertEqual(report.rows[5].status, progress.REDO)
        self.assertEqual(found(report, 6), "the container is stopped")
        self.assertEqual(found(report, 8), "after step 6")
        self.assertEqual(report.next, "next: rkmon start")

    def test_state_11_has_the_confirm_done(self):
        # The confirm is done: its row says the change that waits, which no note says again; the next line names
        # apply.
        report = self.report(11)
        self.assertEqual(statuses(report), ["DONE"] * 8)
        self.assertEqual(
            found(report, 3),
            "a change waits: you confirmed 2 changes at 13:58, and the monitor runs the settings before them",
        )
        self.assertEqual(report.notes, ())
        self.assertEqual(report.next, "next: rkmon apply")
        # In a state the table decides before it, the confirmed fingerprint differing from the delivered layer's too:
        # the row says what you confirmed, as before the first start, with nothing delivered yet.
        for n, changes in (
            (9, dict(CASES[11], hand_edited=("docker/.env",))),
            (8, dict(CASES[8], delivered_fingerprint="")),
        ):
            with self.subTest(state=n):
                f = facts(**changes)
                self.assertEqual(states.current(f).n, n)
                report = progress.deploy(f, confirmed_once=True, logins_read=True)
                self.assertEqual(
                    (report.rows[2].status, found(report, 3)),
                    (
                        progress.DONE,
                        "~/rkmon-deploy.yaml and this package are what you confirmed at 13:58",
                    ),
                )

    def test_a_state_no_row_says_is_the_note(self):
        for n in progress.NOTED_STATES:
            with self.subTest(state=n):
                f = facts(**CASES[n])
                said = wrapping.wrap(
                    states.sentence(states.current(f), f),
                    WIDTH,
                    progress.NOTE,
                    " " * len(progress.NOTE),
                )
                self.assertEqual(list(self.report(n).notes[: len(said)]), said)
        for n in set(CASES) - set(progress.NOTED_STATES) - {1, 10}:
            with self.subTest(state=n):
                self.assertEqual(self.report(n).notes, ())
        # State 10's fix follows its note.
        self.assertEqual(
            self.report(10).notes[-1],
            "fix: set model_bindings.fast.base_url in ~/rkmon-deploy.yaml to an address the container reaches",
        )


class TestJudged(Base):
    """rkmon next --settings FILE: steps 1 and 2 are FILE's."""

    def judged(self, *checks, exists=True, layout=2, recorded=False):
        return progress.Judged("~/other.yaml", exists, layout, tuple(checks), recorded)

    def test_a_file_this_folder_has_not_recorded(self):
        f = facts(**CASES[2])
        report = progress.deploy(f, self.judged(PASS, WARN))
        self.assertEqual(statuses(report)[:3], ["DONE", "DONE", "TODO"])
        self.assertEqual(found(report, 1), "~/other.yaml")
        self.assertEqual(found(report, 2), "2 checks passed (1 warning)")
        self.assertEqual(found(report, 3), "not confirmed yet")
        self.assertEqual(found(report, 4), "after step 3")
        self.assertEqual(
            report.next, "next: step 3, confirm: rkmon confirm --settings ~/other.yaml"
        )
        self.assertEqual(report.also, "")
        # A check that fails, a file that is not there, a file in an older layout.
        report = progress.deploy(f, self.judged(PASS, FAIL))
        self.assertEqual(found(report, 2), "1 of 2 checks fail")
        self.assertEqual(found(report, 3), "after step 2")
        self.assertEqual(
            report.next, "next: step 2, check: rkmon check --settings ~/other.yaml"
        )
        report = progress.deploy(f, self.judged(exists=False))
        self.assertEqual(found(report, 1), "no file ~/other.yaml")
        self.assertEqual(
            report.next, "next: step 1, template: rkmon template deploy ~/other.yaml"
        )
        report = progress.deploy(f, self.judged(PASS, layout=1))
        self.assertEqual(report.next, "next: rkmon migrate --settings ~/other.yaml")

    def test_another_file_than_this_folders(self):
        report = progress.deploy(facts(**CASES[11]), self.judged(PASS))
        self.assertEqual(found(report, 3), f"this folder's settings file is {FILE}")
        self.assertEqual(
            report.next, "next: step 3, confirm: rkmon confirm --settings ~/other.yaml"
        )
        # The folder's own state says nothing of another file.
        self.assertEqual(report.notes, ())

    def test_this_folders_file(self):
        f = facts(**CASES[11])
        report = progress.deploy(f, self.judged(PASS, recorded=True))
        self.assertEqual(
            report.next, states.next_line(states.current(f), f, states.NOT_STARTED)
        )
        report = progress.deploy(f, self.judged(FAIL, recorded=True))
        self.assertEqual(
            report.next, "next: step 2, check: rkmon check --settings ~/other.yaml"
        )

    def test_each_command_alone_once_deploy_step_8_has_passed(self):
        # After a start, once a tick has run since then or this folder's records keep one.
        for f, kwargs in (
            (facts(**CASES[2], first_tick_recorded=True), {}),
            (facts(**CASES[11]), dict(doctor=PASSED, tick="2026-11-15 09:20 UTC")),
        ):
            with self.subTest(state=states.current(f).n):
                for judged, want in (
                    (self.judged(exists=False), "rkmon template deploy ~/other.yaml"),
                    (self.judged(PASS, FAIL), "rkmon check --settings ~/other.yaml"),
                    (self.judged(PASS), "rkmon confirm --settings ~/other.yaml"),
                ):
                    report = progress.deploy(f, judged, **kwargs)
                    self.assertEqual(report.next, f"next: {want}")


class TestBuild(Base):
    def test_the_five_steps(self):
        report = progress.build("~/rkmon-build.yaml", (PASS, FAIL), "", False)
        self.assertEqual(
            progress.render(report),
            "The build path (QUICK_BUILD_GUIDE.md):\n"
            "  1. DONE  template: ~/rkmon-build.yaml\n"
            "  2. TODO  check: 1 of 2 checks fail\n"
            "  3. TODO  build: after step 2\n"
            "  4. TODO  test: after step 3\n"
            "  5. TODO  ship: after step 3\n"
            "next: step 2, check: rkmon check --settings ~/rkmon-build.yaml",
        )
        report = progress.build("~/rkmon-build.yaml", (PASS,), "rkmon:1", False)
        self.assertEqual(found(report, 3), "the image rkmon:1 is not on this host")
        self.assertEqual(
            report.next,
            "next: step 3, build: rkmon build --settings ~/rkmon-build.yaml",
        )
        report = progress.build("~/rkmon-build.yaml", (PASS,), "rkmon:1", True)
        self.assertEqual(found(report, 3), "rkmon:1 is on this host")
        self.assertEqual(
            [found(report, n) for n in (4, 5)], [progress.NOTHING_RECORDED] * 2
        )
        self.assertEqual(
            report.next,
            "next: step 4, test: rkmon image test --settings ~/rkmon-build.yaml",
        )


class TestRender(Base):
    def test_without_the_next_line(self):
        report = progress.deploy(facts(**CASES[2]))
        text = progress.render(report, next=False)
        self.assertNotIn("next:", text)
        self.assertNotIn("have a file already", text)

    def test_without_the_notes(self):
        # rkmon doctor's rows: no notes and no next line, as its state's row says the sentence and the fix
        # (states.row), and it ends with its own outcome.
        report = progress.deploy(
            facts(**CASES[9]), confirmed_once=True, logins_read=True
        )
        self.assertTrue(report.notes)
        bare = progress.render(report, next=False, notes=False)
        self.assertEqual(
            progress.render(report), "\n".join([bare, *report.notes, report.next])
        )

    def test_every_line_fits(self):
        long = facts(logins_missing=tuple(f"login_{i}" for i in range(30)))
        for report in (
            progress.deploy(long, confirmed_once=True, logins_read=True),
            progress.deploy(facts(**CASES[1])),
            progress.deploy(facts(**CASES[2]), note=NOTE.format("x" * 40)),
        ):
            for line in progress.render(report).splitlines():
                self.assertLessEqual(len(line), WIDTH, line)
        text = progress.render(
            progress.deploy(long, confirmed_once=True, logins_read=True)
        )
        self.assertIn("\n" + " " * progress.NAME_COLUMN + "login_", text)

    def test_a_long_next_line_is_wrapped_as_an_outcome_s(self):
        """A next line that a long path makes wider than a check row is wrapped as outcome.finish wraps one."""
        long = f"~/{'a_folder_of_settings_files_with_a_long_name/' * 2}rkmon-build.yaml"
        ran = [Check("settings", Status.PASS, "ok")]
        report = progress.build(long, ran, "rocke-monitor:1", False)
        self.assertGreater(len(report.next), WIDTH)
        lines = progress.render(report).splitlines()
        for line in lines:
            self.assertLessEqual(len(line), WIDTH, line)
        after = outcome.next_lines(report.next)
        self.assertEqual((len(after) > 1, lines[-len(after) :]), (True, after))

    def test_a_time_as_the_rows_show_it(self):
        east = datetime.timezone(datetime.timedelta(hours=2))
        when = datetime.datetime(2026, 11, 15, 11, 12, 30, tzinfo=east)
        self.assertEqual(progress.shown_time(when), "2026-11-15 09:12 UTC")


class TestDockerUnknown(Base):
    """While docker cannot say whose the container name is, nothing of the container is asked: the records' and the
    file's steps as they are, the container's steps unknown and why, and one next line.
    """

    WHY = "docker does not answer"
    UNASKED = dict(
        name_owner="UNKNOWN", container=states.NO_CONTAINER, image_here=False
    )

    def report(self, n=16, **kwargs):
        kwargs.setdefault("confirmed_once", True)
        kwargs.setdefault("unknown_why", self.WHY)
        return progress.deploy(facts(**{**self.UNASKED, **CASES[n]}), **kwargs)

    def test_the_container_steps_read_unknown_and_step_2_is_next(self):
        report = self.report()
        self.assertEqual(statuses(report), ["DONE"] * 3 + ["TODO"] * 5)
        self.assertEqual(
            [found(report, n) for n in range(4, 9)],
            ["unknown (docker does not answer)"] * 5,
        )
        self.assertFalse(any(row.known for row in report.rows[3:]))
        # Its check says what docker answers; the state after the records' is not known, so it says nothing.
        self.assertEqual(
            report.next,
            "next: step 2, check: rkmon check --settings ~/rkmon-deploy.yaml",
        )
        self.assertEqual(report.notes, ())
        report = self.report(note="docker does not answer (no daemon)")
        self.assertEqual(report.notes, ("note: docker does not answer (no daemon)",))
        # Once this folder's records keep a first tick, its command alone.
        f = facts(**{**self.UNASKED, **CASES[16]}, first_tick_recorded=True)
        report = progress.deploy(f, confirmed_once=True, unknown_why=self.WHY)
        self.assertEqual(
            report.next, "next: rkmon check --settings ~/rkmon-deploy.yaml"
        )

    def test_a_state_the_records_decide_keeps_its_next_line_and_notes(self):
        for n, once in ((3, True), (5, True), (5, False)):
            with self.subTest(state=n, confirmed_once=once):
                unknown = self.report(n, confirmed_once=once)
                known = self.report(n, confirmed_once=once, unknown_why="")
                self.assertEqual(unknown.next, known.next)
                self.assertEqual(unknown.notes, known.notes)
        # Nothing of the container is asked, so nothing has passed since a start: confirm's command carries its step.
        self.assertEqual(
            self.report(5, confirmed_once=False).next,
            "next: step 3, confirm: rkmon confirm",
        )
        self.assertTrue(self.report(3).notes)

    def test_an_unknown_step_does_not_wait_for_a_redo(self):
        why = "docker's answer cannot be read"
        report = self.report(5, unknown_why=why)
        self.assertEqual(report.rows[2].status, progress.REDO)
        self.assertEqual(
            [found(report, n) for n in range(4, 9)], [f"unknown ({why})"] * 5
        )

    def test_a_fresh_folder_is_told_why_and_names_step_1(self):
        why = "docker does not answer (Cannot connect to the Docker daemon)"
        report = self.report(2, confirmed_once=False, note=why)
        self.assertEqual(
            [found(report, n) for n in range(1, 9)],
            ["no deploy settings file named yet"]
            + [f"after step {n}" for n in (1, 2, 3, 3, 3, 6, 6)],
        )
        self.assertEqual(
            progress.render(report).split("\n")[-3:],
            [
                f"note: {why}",
                "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml",
                "(have a file already? rkmon next --settings FILE)",
            ],
        )


class TestAfter(Base):
    def test_a_redo_holds_a_todo_back_and_a_row_not_read_does_not(self):
        # A REDO need holds back a later TODO row, one not read too.
        f = facts(container=states.NO_CONTAINER, file_changed=True)
        rows = progress.deploy(f, confirmed_once=True, logins_read=False).rows
        self.assertEqual(rows[2].status, progress.REDO)
        self.assertEqual([r.found for r in rows[4:6]], ["after step 3"] * 2)
        # A row not read holds back nothing: the start is next.
        rows = progress.deploy(
            facts(container=states.NO_CONTAINER), logins_read=False
        ).rows
        self.assertEqual(rows[4].found, progress.LOGINS_NOT_READ)
        self.assertEqual(
            rows[5].found, "this deployment's container does not exist yet"
        )


if __name__ == "__main__":
    unittest.main()
