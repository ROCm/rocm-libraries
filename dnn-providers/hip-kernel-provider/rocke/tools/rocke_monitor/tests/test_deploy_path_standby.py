# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""pause, resume and standby end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. pause and resume run rk pause or rk resume in the running container, and
standby off rk standby-off; while the monitor says it runs in standby, a start, an apply and each next line say so.
"""

import json
import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
from deploy_path_fixture import GOOD, DeployPathCase  # noqa: E402
from rendered import finished, one_line  # noqa: E402
from rkmon_lib import progress, records  # noqa: E402


class TestStandby(DeployPathCase):
    def test_pause_and_resume(self):
        """pause and resume: rk pause or rk resume in the running container, PAUSED or RESUMED, each with the
        state's next line, state 15's resume after a pause and state 16's after a resume (step 7's doctor, while none
        has passed since the start); NOTHING TO DO when the monitor says it is paused, or not, already, with nothing
        run; refused while the container is stopped.
        """
        self.confirmed()
        before = self.delivered()
        made = deploy_folder.plan(self.pkg, self.settings)
        paused = {"applied": {"fingerprint": made.fingerprint}, "paused": True}
        self.answer(
            [
                [
                    before[1][0],
                    json.dumps(paused),
                    0,
                    {},
                    {"after": r"(?ms)rk pause$(?!.*rk resume$)"},
                ],
                [
                    r"rk (pause|resume)$",
                    "enabled=x frozen=False cleared=[] prioritized=[]",
                    0,
                ],
                *self.apply_answers(before, made, made),
            ]
        )
        # state 16's line while no doctor has passed since the start
        doctor = f"next: step 7, doctor: {self.w} doctor"
        for command, line, next_line in (
            ("resume", "NOTHING TO DO: the monitor is not paused", doctor),
            (
                "pause",
                "PAUSED: the monitor starts no new work; the work in flight finishes",
                f"next: {self.w} resume",
            ),
            ("pause", "NOTHING TO DO: the monitor is paused", f"next: {self.w} resume"),
            ("resume", "RESUMED: the monitor starts new work again", doctor),
        ):
            with self.subTest(command=command, line=line):
                rc, out, err = self.rkmon(command, **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(out.endswith(finished(line, next_line)), out)
        ran = [
            c.split()[-1] for c in self.calls() if re.search(r"rk (pause|resume)$", c)
        ]
        self.assertEqual(ran, ["pause", "resume"])
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("pause", **stopped)
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "REFUSED: the container is not running; nothing changed", one_line(out)
        )

    def test_a_start_in_standby_says_so(self):
        """A start after which the monitor says it runs in standby (rk applied --json): STARTED says so, then
        the state's next line, step 7's doctor, as after any start."""
        w = self.w
        self.confirmed()
        layer, applied = self.delivered()
        standby = [
            applied[0],
            json.dumps(dict(json.loads(applied[1]), standby=True)),
            0,
        ]
        self.answer(
            [
                layer,
                standby,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: step 7, doctor: {w} doctor",
                )
            ),
            out,
        )

    def test_a_put_back_start_in_standby_says_so(self):
        """In state 10 (put back), a start of the running container, a stop left in it until the start clears it
        (rk start), while the monitor runs in standby: STARTED gives the state's words, then says the monitor runs in
        standby."""
        w = self.w
        self.confirmed()
        layer, applied = self.delivered()
        standby = [
            applied[0],
            json.dumps(dict(json.loads(applied[1]), standby=True)),
            0,
        ]
        fingerprint = self.confirm_record().fingerprint
        records.put_back(
            self.pkg,
            records.PutBack(fingerprint, "rollback", records.timestamp(), "", None),
        )
        self.answer(
            [
                layer,
                standby,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"compose ps -q rkmon$", "0123456789ab", 0],
                [r"rk logins --json", "[]", 0],
                [r"rk stopped$", "", 1, {}, {"after": r"(?m)rk start$"}],
                [r"rk stopped$", "rk stop --drain", 0],
            ]
        )
        self.log.unlink(missing_ok=True)
        rc, out, err = self.rkmon("start", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"STARTED: the last change you confirmed was put back by {w} rollback, and the monitor runs the settings "
            f"from before it, in standby: it sends nothing until you run {w} standby off next: ",
            one_line(out),
        )

    def test_an_apply_in_standby_says_so(self):
        """An apply after which the monitor says it runs in standby (rk applied --json): APPLIED says so, then
        the state's next line, step 7's doctor while none has passed since the restart, as after any apply that
        restarts."""
        w = self.w
        rows, made, was = self.change_waits()

        def in_standby(row) -> list:
            """ROW, the stand-in docker's answer, with standby on where it is the monitor's applied record."""
            try:
                said = json.loads(row[1])
            except ValueError:
                return row
            if not isinstance(said, dict) or "applied" not in said:
                return row
            return [row[0], json.dumps(dict(said, standby=True)), *row[2:]]

        image_id = "sha256:" + "1" * 64
        self.answer(
            [
                [r"compose ps -q rkmon$", "0123456789ab", 0],
                [r"^inspect --format \{\{\.Image\}\} 0123456789ab$", image_id, 0],
                [r"^image inspect --format \{\{\.Id\}\} ", image_id, 0],
                *[in_standby(row) for row in rows],
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "APPLIED: the deployment runs your confirmed settings (1 change), in standby: it sends nothing "
                    f"until you run {w} standby off",
                    f"next: step 7, doctor: {w} doctor",
                )
            ),
            out,
        )

    def test_a_monitor_in_standby_says_so_on_each_line(self):
        """While the running monitor says it is in standby (rk applied --json), the state is standby's:
        status says the monitor sends nothing until standby off; the next lines walk deploy steps 7 and 8, as after a
        start in standby, and then name standby off, as first-tick ends; and a change that waits says it is in standby
        too."""
        w = self.w
        self.confirmed()
        layer, applied = self.delivered()
        standby = [
            applied[0],
            json.dumps(dict(json.loads(applied[1]), standby=True)),
            0,
        ]
        container = [
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"compose ps -q rkmon$", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
            [r"rk logins --json", "[]", 0],
            [r"rk stopped$", "", 1],
        ]
        never = [r"rk tick-due", "last tick never", 1]
        self.answer([layer, standby, *container, never])
        in_standby = f"in standby: it sends nothing until you run {w} standby off"
        rc, out, err = self.rkmon("status", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"state: the monitor runs your confirmed settings, {in_standby} next: step 7, doctor: {w} doctor",
            one_line(out),
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 7, doctor: {w} doctor")), out
        )
        records.doctor_result(self.pkg, records.Doctor("2026-11-15T09:10:00Z", 3, ()))
        rc, out, err = self.rkmon("next", **self.running())
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 8, first tick: {w} first-tick")),
            out,
        )
        tick = [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0]
        self.answer([layer, standby, *container, tick])
        rc, out, err = self.rkmon("first-tick", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: the first tick ran at 2026-11-15T09:05:00Z",
                    f"next: {w} standby off, once {w} status and the dashboards read right",
                )
            ),
            out,
        )
        # A change waits, in standby: the state's sentence says so.
        self.write_settings(memory_gib=13)
        self.confirmed()
        rc, out, err = self.rkmon("status", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f", and the monitor runs the settings before them; {in_standby} next: {w} apply",
            one_line(out),
        )

    def test_standby_off(self):
        """standby off: rk standby-off in the running container, in state 14 alone; rk's lines as it says them,
        then the deployment state's next line: RELEASED, and the done line of a monitor that runs your confirmed
        settings; rk's REFUSED, nothing changed, and standby off again once its fix is done. rk's ERROR part-way, or any
        other exit, then this command again; under --dry-run, PASSED, and rk does not run. NOTHING TO DO while the
        monitor is not in standby; REFUSED while standby is on in another state, that state's own words saying why,
        with its next line. No docker call reads another deployment.
        """
        w = self.w
        self.confirmed()
        layer, applied = self.delivered()
        records.doctor_result(self.pkg, records.Doctor("2026-11-15T09:10:00Z", 3, ()))
        off = r"rk standby-off$"
        # A docker call that reads another deployment: a container by its name, a volume, another compose project's,
        # or this package's monitors on the host.
        elsewhere = re.compile(
            r"^(?:exec|cp|volume|container) |label=org\.opencontainers\.image\.title="
            r"|^compose (?:.* )?(?:-p|--project-name|--project-directory|-f|--file)[ =]"
        )

        def monitor(standby: bool, **more) -> list:
            """The running monitor's answer to rk applied --json: your confirmed settings, in standby or not."""
            said = json.dumps(dict(json.loads(applied[1]), standby=standby))
            return [applied[0], said, 0, {}, more]

        def run(*rows, typed=("standby", "off")) -> tuple:
            self.log.write_text("", encoding="utf-8")
            self.answer(
                [
                    layer,
                    *rows,
                    [r"image inspect", GOOD, 0],
                    [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                    [r"compose ps -q rkmon$", "0123456789ab", 0],
                    [
                        r"inspect --format \{\{\.State\.StartedAt\}\}",
                        "2026-11-15T09:00:00Z",
                        0,
                    ],
                    [r"rk logins --json", "[]", 0],
                    [r"rk stopped$", "", 1],
                    [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0],
                ]
            )
            rc, out, err = self.rkmon(*typed, **self.running())
            for call in self.docker_calls():
                with self.subTest(typed=typed, call=call):
                    self.assertIsNone(elsewhere.search(call))
            return rc, out, err

        done = "done: the monitor runs your confirmed settings"
        again = f"next: {w} standby off"
        ended = "the one-time move from an older install ended"
        released = "RELEASED: the monitor sends again, as your settings allow"
        rc, out, err = run(
            monitor(False, after=f"(?m){off}"),
            monitor(True),
            [off, f"{ended}\n{released}\n", 0],
        )
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out, finished(ended, released, done))
        self.assertEqual(len([c for c in self.calls() if re.search(off, c)]), 1)
        # rk's REFUSED: standby stays on, and the next line is standby's
        refused = "REFUSED: the one-time move cannot end yet; standby stays on; nothing changed"
        fix = "fix: restart the monitor, then run standby off"
        rc, out, err = run(monitor(True), [off, f"{refused}\n{fix}\n", 4])
        self.assertEqual(rc, 4, out + err)
        self.assertEqual(
            out,
            finished(
                refused, fix, f"{again}, once {w} status and the dashboards read right"
            ),
        )
        # rk's ERROR, part-way: what the end did, why it stopped, then this command again
        renamed = "renamed the older install's config.json, read-only"
        error = "ERROR: the record of the move cannot be written; standby stays on"
        rc, out, err = run(monitor(True), [off, f"{renamed}\n{error}\n{fix}\n", 1])
        self.assertEqual(rc, 1, out + err)
        self.assertEqual(out, finished(renamed, error, fix, again))
        rc, out, err = run(monitor(True), [off, "", 137])
        self.assertEqual(rc, 1, out + err)
        self.assertEqual(
            out,
            finished(
                "ERROR: rk standby-off exited 137 (the lines above say why)", again
            ),
        )
        # an exit 1 with no ERROR line of rk's, a traceback's: rkmon's own ERROR
        traceback = "Traceback (most recent call last):\nOSError: [Errno 28] No space left on device"
        rc, out, err = run(monitor(True), [off, traceback, 1])
        self.assertEqual(rc, 1, out + err)
        self.assertEqual(
            out,
            f"{traceback}\n"
            + finished(
                "ERROR: rk standby-off exited 1 (the lines above say why)", again
            ),
        )
        # --dry-run: the command shown, not run
        rc, out, err = run(monitor(True), typed=("standby", "off", "--dry-run"))
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(out, r"compose exec -T --user rkmon .*rkmon rk standby-off\n")
        self.assertTrue(
            out.endswith(finished("PASSED: the monitor can leave standby", again)), out
        )
        self.assertEqual(self.called(off), [-1])
        # Not in standby: nothing to do, and rk does not run
        rc, out, err = run(monitor(False))
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out, finished("NOTHING TO DO: the monitor is not in standby", done)
        )
        self.assertEqual(self.called(off), [-1])
        # A change waits (state 11): standby stays on until apply has run it, as the state's words and its next
        # line say
        self.write_settings(memory_gib=13)
        self.confirmed()
        at = progress.shown_time(records.parsed(self.confirm_record().at))
        rc, out, err = run(monitor(True))
        self.assertEqual(rc, 4, out + err)
        self.assertEqual(
            out,
            finished(
                "REFUSED: the monitor does not run the settings you confirmed, so standby stays on; a change waits: "
                f"you confirmed new settings at {at}, and the monitor runs the settings before them; nothing changed",
                f"next: {w} apply",
            ),
        )
        self.assertEqual(self.called(off), [-1])
        rc, out, err = run(monitor(False))
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            finished(
                "NOTHING TO DO: the monitor is not in standby", f"next: {w} apply"
            ),
        )


if __name__ == "__main__":
    unittest.main()
