# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""apply end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. apply runs the doctor, drains, puts this package's code and the confirmed
settings in, restarts, sets the work volume up (rk provision) and loads the settings, then runs the doctor again: a new
failure, or a load or a restart that fails, puts the change back, and its doctor after, with no check failing, is deploy
step 7 passed since the restart. It refuses a busy port, another deployment's volume, a private file the release would
carry and code that needs a new image; a Ctrl-C, a drain that runs out of time or a code-install that fails ends it with
what changed; before the first start it has nothing to do. The lines apply and rollback end with are those
QUICK_USER_GUIDE.md's "Change a setting" quotes.
"""

import io
import json
import os
import pathlib
import re
import signal
import socket
import sys
import tarfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
import sandbox  # noqa: E402
from deploy_path_fixture import (  # noqa: E402
    DRAINED,
    ELSEWHERE,
    GOOD,
    INSTALLED,
    LAYER_PUT_BACK,
    LAYER_READ,
    LAYER_RUN,
    LOAD,
    MAKES,
    MODEL_FIX,
    NEXT_LINE,
    OUTCOME_LINE,
    RESTART,
    STARTED,
    TICK_SINCE_START,
    UNREAD,
    DeployPathCase,
)
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    code_volume,
    deliver,
    image,
    outcome,
    progress,
    records,
)

# docker's answer while the container restarts
RESTARTING = "Error response from daemon: Container 0123456789ab is restarting, wait until the container is running"
# rk's last tick before the running container's start (STARTED)
TICK_BEFORE_START = [r"rk tick-due", "last tick 2026-11-15T08:55:00Z, 8 min ago", 0]


class TestApply(DeployPathCase):
    def test_apply_makes_the_change_between_two_doctors(self):
        """apply on the running container: the doctor, the drain, the record of the apply, this package's code (a
        release over the image, code-install, the archive of the image's own files on its input, in the running
        container as root while it is of the image the change runs), the settings with the previous ones kept, the
        container made again, the load, the drain lifted and the doctor again, which finds no new failure: APPLIED with
        the count of changes, the record of the apply gone, and the state's next line: step 8's rkmon first-tick, as
        the doctor after, with no check failing, is step 7 passed since the restart (record_doctor), and no tick has
        run since then; a tick at once is asked of the tick loop first (rk tick-now), once. Apply again has nothing to
        do, with that line. --dry-run says what changes and what apply will do, and changes nothing.
        """
        rows, made, was = self.change_waits()
        image_id = "sha256:" + "1" * 64
        self.answer(
            [
                [r"compose ps -q rkmon$", "0123456789ab", 0],
                [r"^inspect --format \{\{\.Image\}\} 0123456789ab$", image_id, 0],
                [r"^image inspect --format \{\{\.Id\}\} ", image_id, 0],
                *STARTED,
                TICK_BEFORE_START,
                *rows,
            ]
        )
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon("apply", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines()[:3],
            [
                "== apply: make the confirmed change on this deployment (dry run: nothing changes)",
                "Your changes:",
                "  ~ host.resources.memory_gib  12 -> 13",
            ],
        )
        self.assertIn(
            "\n\n1 change. apply will: let the work in flight finish (up to 60 minutes), write the new settings (the "
            "previous ones\nkept), make the container again, run the doctor, and put the previous settings back on a "
            "new failure.\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: 1 change can be applied; nothing changed",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        nothing = ("code-install", LAYER_RUN, "up -d", "rk drain", "rk doctor")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))
        self.log.unlink()
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        at = records.parsed(self.confirm_record().at)
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                "  PASS  doctor before: 3 checks",
                f"  {DRAINED[0]}",
                "  drain: done",
                f"  code: {INSTALLED}",
                "  wrote: the new settings (the previous ones kept), docker/.env and the override",
                "  restart: docker compose up -d --force-recreate",
                f"  loaded: the monitor runs the settings you confirmed at {progress.shown_time(at)}",
                "  PASS  doctor after: 3 checks, no new failure",
                "  tick: asked for a tick at once: the tick loop starts it within seconds (after the one in progress, "
                "if any)",
                "APPLIED: the deployment runs your confirmed settings (1 change)",
                *rewrapped(
                    f"next: step 8, first tick: {self.w} first-tick"
                ).splitlines(),
            ],
        )
        order = self.called(
            r"rk doctor --fresh --json$",
            r"rk drain --timeout-minutes 60$",
            r"exec -T --user 0 -e RKMON_AS=root rkmon \S+ code-install$",
            LAYER_RUN,
            r"compose up -d --force-recreate$",
            r"rk applied --load --json$",
            r"rk undrain$",
        )
        self.assertEqual(order, sorted(order), self.calls())
        self.assertNotIn(-1, order, self.calls())
        doctors = [i for i, c in enumerate(self.calls()) if "rk doctor" in c]
        self.assertEqual(len(doctors), 2)
        self.assertGreater(doctors[1], order[-1])
        ticks = [i for i, c in enumerate(self.calls()) if "rk tick-now" in c]
        self.assertEqual(len(ticks), 1, self.calls())
        self.assertGreater(ticks[0], doctors[1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), made.env)
        self.assertEqual(self.kept_env(), ran)
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNone(kept.put_back)
        self.assertEqual(kept.doctor, records.Doctor(mock.ANY, 3, ()))
        data = self.stdin.read_bytes()
        self.assertIn(b"deliver(", data)
        start = (
            data.index(b"ustar") - 257
        )  # the archive's first header (tar's magic is at 257)
        with tarfile.open(fileobj=io.BytesIO(data[start:])) as tar:
            members = {m.name: m for m in tar.getmembers()}
        self.assertEqual(
            sorted(members), image.code_files(self.pkg), "the image's own files"
        )
        self.assertEqual(members["docker/rkmon"].mode, 0o755)
        self.assertEqual(members["rkmon_lib/deliver.py"].mode, 0o644)
        self.assertEqual({(m.uid, m.gid) for m in members.values()}, {(0, 0)})
        self.assertNotIn(deliver.ENV_FILE, members)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: no change waits for apply; the monitor runs your confirmed settings",
                    f"next: step 8, first tick: {self.w} first-tick",
                )
            ),
            out,
        )

    def test_a_passing_doctor_after_is_step_7_passed_since_the_restart(self):
        """The doctor after apply's change, with no check failing, is deploy step 7 passed since the restart: this
        folder's records keep its result as they keep rkmon doctor's (record_doctor), so rkmon next's
        step 7 row says how many checks passed and when, and its next line, as apply's, names step 8's command while
        no tick has run since the restart."""
        rows, made, was = self.change_waits()
        self.answer([*STARTED, TICK_BEFORE_START, *rows])
        self.assertIsNone(records.read(self.pkg).doctor)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        first_tick = f"next: step 8, first tick: {self.w} first-tick"
        self.assertTrue(
            out.endswith(
                finished(
                    "APPLIED: the deployment runs your confirmed settings (1 change)",
                    first_tick,
                )
            ),
            out,
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(
            out,
            r"(?m)^  7\. DONE  doctor: 3 checks passed at \d{4}-\d\d-\d\d \d\d:\d\d UTC$",
        )
        self.assertIn("  8. TODO  first tick: none since the start\n", out)
        self.assertTrue(out.endswith(rewrapped(f"\n{first_tick}\n")), out)

    def test_a_new_doctor_failure_puts_the_change_back(self):
        """A new failure of the doctor after the change (deployment.new_failures) puts back the settings kept, and the
        code that ran (code-back), and makes the container again: its FAIL row, ROLLED BACK (exit 1) with its fix,
        and state 10's next line, which waits for what failed to be fixed. The records keep the change put back, why
        and its fix, and the fingerprints of what the put-back wrote: nothing reads as changed outside your file (state
        9); and no release that the last apply put in, though an apply before it put one in (records.Installed): the
        code is back."""
        rows, made, was = self.change_waits()
        records.begin(self.pkg, "apply").finish(code=True)
        after = r"(?m)compose up -d --force-recreate$"
        self.answer(
            [
                [
                    r"rk doctor --fresh --json$",
                    self.doctor("model fast"),
                    0,
                    {},
                    {"after": after},
                ]
            ]
            + rows
        )
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        failure = "model fast: no answer from the binding's base_url"
        self.assertIn(f"\n  FAIL  doctor after: {failure}\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                    "was put back"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(MODEL_FIX))
                + finished(
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)"
                )
            ),
            out,
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        calls = self.calls()
        runs = [i for i, c in enumerate(calls) if re.search(LAYER_RUN, c)]
        ups = [i for i, c in enumerate(calls) if re.search(after, c)]
        backs = [i for i, c in enumerate(calls) if c.endswith(" code-back")]
        self.assertEqual((len(runs), len(ups), len(backs)), (2, 2, 1), calls)
        self.assertTrue(runs[1] < backs[0] < ups[1], calls)
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNone(kept.installed)
        self.assertEqual(
            (kept.put_back.fingerprint, kept.put_back.reason, kept.put_back.failure),
            (made.fingerprint, "doctor", failure),
        )
        self.assertEqual(kept.put_back.fix, MODEL_FIX)
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            f"the last change you confirmed was put back after a failed doctor check: {failure}",
            one_line(out),
        )
        self.assertNotIn("changed outside your file", out)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm "
                    "first)"
                )
            ),
            out,
        )

    def test_a_failure_from_before_the_change_does_not_put_it_back(self):
        """A check that failed before the change, the doctor's baseline, and fails after it is no new failure (apply's
        step 7): the WARN row names it failing already, the doctor after finds no new failure, and the change stays.
        With a check failing, the doctor after is not deploy step 7 passed since the restart: the records keep no
        result of it, and the next line names step 7's command.
        """
        rows, made, was = self.change_waits()
        self.answer(
            [
                [r"rk doctor --fresh --json$", self.doctor("model fast"), 0],
                *STARTED,
                TICK_BEFORE_START,
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  WARN  doctor before: 4 checks; failing already: model fast\n", out
        )
        self.assertIn("\n  PASS  doctor after: 4 checks, no new failure\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "APPLIED: the deployment runs your confirmed settings (1 change)",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        kept = records.read(self.pkg)
        self.assertEqual((kept.put_back, kept.doctor), (None, None))

    def test_change_a_setting_quotes_the_lines_apply_and_rollback_end_with(self):
        """QUICK_USER_GUIDE.md's "Change a setting" quotes the lines apply and rollback end with, the program as
        ./rkmon: each APPLIED or RESTORED line, and each next or done line, that it quotes is one they print here, and
        each one they print is quoted. After APPLIED: step 7's doctor when a check fails in the doctor after, even one
        that failed before the change; else step 8's first-tick while no tick has run since the restart; else the done
        line; each with its step's number and name until deploy step 8 has passed once (a first tick kept). After
        RESTORED: apply."""
        guide = (ROOT / "QUICK_USER_GUIDE.md").read_text(encoding="utf-8")
        section = guide.split("## Change a setting\n", 1)[1].split("\n## ", 1)[0]
        quoted = {
            span
            for span in re.findall(r"`([^`]+)`", one_line(section))
            if re.match(r"(?:APPLIED|RESTORED|next|done): ", span)
        }
        printed = set()
        # deploy step 8 passed, the last tick, the checks failing before the change and after it, rollback after apply
        cases = (
            (True, TICK_BEFORE_START, (), True),
            (True, TICK_SINCE_START, (), False),
            (True, TICK_BEFORE_START, ("model fast",), False),
            (False, TICK_BEFORE_START, (), False),
            (False, TICK_BEFORE_START, ("model fast",), False),
        )
        for passed, tick, failing, rollback in cases:
            with self.subTest(step_8_passed=passed, tick=tick[1], failing=failing):
                self.setUp()
                rows, made, was = self.change_waits()
                if passed:
                    records.first_tick(
                        self.pkg, records.FirstTick("2026-11-15T08:00:00Z")
                    )
                doctor = [r"rk doctor --fresh --json$", self.doctor(*failing), 0]
                self.answer([doctor, *STARTED, tick, *rows])
                for command in ("apply", "rollback")[: 1 + rollback]:
                    rc, out, err = self.rkmon(command, **self.running())
                    self.assertEqual(rc, 0, out + err)
                    nxt = list(NEXT_LINE.finditer(out))[-1].start()
                    last = max(
                        m.start() for m in OUTCOME_LINE.finditer(out) if m.start() < nxt
                    )
                    printed |= {
                        one_line(part).replace(self.w, "./rkmon")
                        for part in (out[last:nxt], out[nxt:])
                    }
        self.assertEqual(quoted, printed)

    def test_apply_again_judges_the_doctor_against_the_first_baseline(self):
        """apply again after one that stopped part-way judges the doctor against the first run's baseline, which the
        record of the change keeps (state 3): a check that the change that stopped part-way made fail is never
        failing already, so it puts the change back when it still fails."""
        rows, made, was = self.change_waits()
        self.answer([[LOAD, "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        under_way = records.read(self.pkg).under_way
        self.assertEqual((under_way.baseline, under_way.failing), (True, ()))
        self.log.unlink()
        layer = deliver.texts(made)[2]
        self.answer(
            [
                [r"rk doctor --fresh --json$", self.doctor("model fast"), 0],
                [LAYER_READ, layer, 0, {}, {"after": f"^(?!(?s:.*){LAYER_RUN})"}],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            "\n  WARN  doctor before: 4 checks; failing since the change that stopped part-way: model fast\n",
            out,
        )
        failure = "model fast: no answer from the binding's base_url"
        self.assertIn(f"\n  FAIL  doctor after: {failure}\n", out)
        self.assertIn("\nROLLED BACK: ", out)
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure), ("doctor", failure)
        )
        self.assertIsNone(kept.under_way)

    def test_a_load_that_fails_puts_the_change_back(self):
        """A configuration the monitor refuses to load (rk applied --load --json: the refusal and its fix) puts the
        change back before the doctor runs again: its FAIL row with rk's refusal, then the put-back and the load
        of the settings put back, ROLLED BACK with rk's fix, the records keeping why (state 10: a failed load, the
        refusal in rk's words)."""
        rows, made, was = self.change_waits()
        fix = checks.Command(("rkmon", "logins"))
        refused = {"refused": "the login gh is missing", "fix": checks.fix_to_json(fix)}
        self.answer(rows[:5] + [[LOAD, json.dumps(refused), 4]] + rows[5:])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertNotIn("doctor after", out)
        self.assertIn("\n  FAIL  load: the login gh is missing\n", out)
        self.assertIn(
            "\n  loaded: the monitor runs the settings from before this apply\n", out
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                    "was put back",
                    "fix: rkmon logins",
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)",
                )
            ),
            out,
        )
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("load", "the login gh is missing", fix),
        )
        self.assertEqual(sum("rk doctor" in c for c in self.calls()), 1, self.calls())
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            "the last change you confirmed was put back after a failed load: the login gh is missing",
            one_line(out),
        )

    def test_a_load_that_times_out_puts_the_change_back(self):
        """A monitor that does not load the confirmed settings within the load's wait (states.LOAD_WAIT_MINUTES, a few
        seconds in this test's copy of the package, made before the confirms, which record its code) puts the change
        back: the FAIL row that says so, ROLLED BACK with the fix that the container's log says why, and state
        10's sentence names it.
        """
        states_py = self.pkg / "rkmon_lib" / "states.py"
        text = states_py.read_text(encoding="utf-8")
        self.assertEqual(text.count("\nLOAD_WAIT_MINUTES = 2\n"), 1)
        states_py.write_text(
            text.replace("\nLOAD_WAIT_MINUTES = 2\n", "\nLOAD_WAIT_MINUTES = 0.02\n"),
            encoding="utf-8",
        )
        rows, made, was = self.change_waits()
        old = json.dumps({"applied": {"fingerprint": was.fingerprint}, "paused": False})
        self.answer([[LOAD, old, 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        failure = "the monitor did not load your confirmed settings within 0.02 minutes"
        self.assertIn(f"\n  FAIL  load: {failure}\n", out)
        self.assertTrue(
            one_line(out).endswith(
                one_line(
                    finished(
                        "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                        "was put back",
                        f"fix: {self.w} logs --tail 50 says why",
                        f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)",
                    )
                )
            ),
            out,
        )
        logs = checks.Steps(
            (checks.Command((self.w, "logs", "--tail", "50")), "says why")
        )
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("load", failure, logs),
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            f"the last change you confirmed was put back after a failed load: {failure}",
            one_line(out),
        )

    def test_a_restart_that_fails_puts_the_change_back(self):
        """A container that does not start again after the change (docker compose up fails) puts the change back:
        the restart's FAIL row, the put-back and its own restart, ROLLED BACK with the fix that the container's
        log says why, and state 10's sentence names the failed restart."""
        rows, made, was = self.change_waits()
        self.answer(
            [[RESTART, "", 0, {}, {"after": LAYER_PUT_BACK}], [RESTART, "", 1], *rows]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  FAIL  restart: docker compose up exited 1\n", out)
        self.assertNotIn("doctor after", out)
        self.assertTrue(
            one_line(out).endswith(
                one_line(
                    finished(
                        "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                        "was put back",
                        f"fix: {self.w} logs --tail 50 says why",
                        f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)",
                    )
                )
            ),
            out,
        )
        self.assertEqual(sum("rk doctor" in c for c in self.calls()), 1, self.calls())
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure),
            ("restart", "docker compose up exited 1"),
        )
        self.assertIsNone(kept.under_way)
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            "the last change you confirmed was put back after a failed restart: docker compose up exited 1",
            one_line(out),
        )

    def test_a_doctor_that_cannot_run_before_the_change_changes_nothing(self):
        """A doctor that does not run before the change (apply's step 1: rk doctor fails in the container, or what it
        says cannot be read) leaves no baseline to judge the change by: ERROR, exit 1, nothing changed, and the command
        again; no drain, no record of the apply."""
        rows, made, was = self.change_waits()
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        for answer, why in (
            ((RESTARTING, 1), f"rk doctor in the container exited 1: {RESTARTING}"),
            (
                ("Traceback (most recent call last):\nMemoryError", 0),
                "what rk doctor in the container said cannot be read: MemoryError",
            ),
        ):
            with self.subTest(why=why):
                self.answer([[r"rk doctor --fresh --json$", *answer], *rows])
                start = len(self.calls())
                rc, out, err = self.rkmon("apply", **self.running())
                self.assertEqual(rc, 1, out + err)
                self.assertTrue(
                    out.endswith(
                        finished(
                            f"ERROR: the doctor did not run before the change ({why}); nothing changed",
                            f"next: {self.w} apply",
                        )
                    ),
                    out,
                )
                none = ("rk drain", "code-install", LAYER_RUN, "up -d")
                self.assertEqual(self.called(*none, start=start), [-1] * len(none))
                self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
                self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_doctor_that_cannot_run_after_the_change_puts_it_back(self):
        """A doctor that does not run after the change (apply's step 7) cannot say that nothing new fails, so the change
        is put back as for a new failure: the FAIL row that says why, ROLLED BACK with the doctor's own report as the
        fix, and state 10's sentence names it."""
        rows, made, was = self.change_waits()
        self.answer(
            [
                [
                    r"rk doctor --fresh --json$",
                    RESTARTING,
                    1,
                    {},
                    {"after": f"(?m){RESTART}"},
                ],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        failure = f"the doctor did not run (rk doctor in the container exited 1: {RESTARTING})"
        self.assertIn(f" FAIL doctor after: {failure} ", one_line(out))
        fix = checks.Command((self.w, "doctor"))
        self.assertTrue(
            out.endswith(
                finished(
                    "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                    "was put back"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(fix))
                + finished(
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)"
                )
            ),
            out,
        )
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("doctor", failure, fix),
        )
        self.assertIsNone(kept.under_way)
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            f"the last change you confirmed was put back after a failed doctor check: {failure}",
            one_line(out),
        )

    def test_an_older_rk_s_doctor_report_is_read_as_text(self):
        """An image whose rk doctor has no --json (it exits 2: unrecognized arguments): apply reads its report instead
        (deployment.doctor_rows), before and after the change; a new FAIL in it puts the change back, whose fix is the
        doctor's whole report (rkmon doctor)."""
        rows, made, was = self.change_waits()
        older = (
            "usage: rk doctor [-h] [--fresh]\nrk: error: unrecognized arguments: --json"
        )
        report = "  PASS  ramdisk: fine\n  PASS  config: fine\n  PASS  session login: gh: fine\n"
        failing = (
            f"{report}  FAIL  model fast: no answer from the binding's base_url\n"
            "        fix: set model_bindings.fast.base_url in ~/rkmon-deploy.yaml to an address the container reaches\n"
        )
        self.answer(
            [
                [r"rk doctor --fresh --json$", older, 2],
                [r"rk doctor --fresh$", failing, 4, {}, {"after": f"(?m){RESTART}"}],
                [r"rk doctor --fresh$", report, 0],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  PASS  doctor before: 3 checks\n", out)
        failure = "model fast: no answer from the binding's base_url"
        self.assertIn(f"\n  FAIL  doctor after: {failure}\n", out)
        fix = checks.Command((self.w, "doctor"))
        self.assertTrue(
            out.endswith(
                finished(
                    "ROLLED BACK: the previous settings are back and the container runs them; your confirmed change "
                    "was put back"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(fix))
                + finished(
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)"
                )
            ),
            out,
        )
        kept = records.read(self.pkg)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("doctor", failure, fix),
        )

    def test_apply_while_a_login_is_missing_before_the_first_start_has_nothing_to_do(
        self,
    ):
        """apply in state 7 before the first start (a login your confirmed settings need is not set yet, and no
        container exists): NOTHING TO DO, exit 0, as the first start takes your confirmed settings, with the state's
        sentence and its next step, the logins; nothing is made or written. Once this folder's container exists, the
        first start has passed: apply refuses there, with the same next line."""
        self.confirmed()
        rows = [
            {
                "name": "github_read",
                "what": "gh's read login",
                "state": "MISSING",
                "found": "no login",
                "needed_by": [],
                "command": "login github_read",
            }
        ]
        made = {
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }
        need = "your confirmed settings need github_read, which is missing or wrong"
        login = f"next: step 5, login: {self.w} login"
        self.answer(
            [
                *deploy_folder.secrets_volumes(True, ["github_read"]),
                [r"image inspect", GOOD, 0],
            ]
        )
        rc, out, err = self.rkmon("apply")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    f"NOTHING TO DO: the first start takes your confirmed settings; {need}",
                    login,
                )
            ),
            out,
        )
        self.assertEqual([c for c in self.calls() if MAKES.search(c)], [])
        self.assertFalse((self.pkg / deliver.ENV_FILE).exists())
        self.answer(
            [
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk logins --json", json.dumps(rows), 0],
                [r"image inspect", GOOD, 0],
            ]
        )
        rc, out, err = self.rkmon("apply", **made)
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(finished(f"REFUSED: {need}; nothing changed", login)), out
        )

    def test_apply_before_the_first_start_has_nothing_to_do(self):
        """apply in state 8, never started (no container yet, or one started some other way, whose work volume is not
        set up): NOTHING TO DO, exit 0, as the first start takes your confirmed settings, with the state's sentence and
        its next step, the start; nothing is made, and nothing runs in the container."""
        self.confirmed()
        start = f"next: step 6, start: {self.w} start"
        other_way = (
            "this deployment's container runs, but its work volume is not set up (it was started some other way "
            f"than {self.w} start)"
        )
        set_up = [r"rk provision --check$", "not set up yet", 1]
        for case, answers, env, sentence in (
            (
                "no container",
                deploy_folder.secrets_volumes(True),
                {},
                "this deployment's container does not exist yet",
            ),
            (
                "started some other way",
                [set_up, [r"rk logins --json", "[]", 0], [r"rk stopped$", "", 1]],
                self.running(),
                other_way,
            ),
        ):
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.answer([*answers, [r"image inspect", GOOD, 0]])
                rc, out, err = self.rkmon("apply", **env)
                self.assertEqual((rc, err), (0, ""), out)
                self.assertTrue(
                    out.endswith(
                        finished(
                            "NOTHING TO DO: the first start takes your confirmed settings; "
                            + sentence,
                            start,
                        )
                    ),
                    out,
                )
                self.assertEqual([c for c in self.calls() if MAKES.search(c)], [])
                ran = [c for c in self.calls() if " exec " in c]
                self.assertTrue(
                    all(c.endswith(("--check", "--json", "rk stopped")) for c in ran),
                    ran,
                )
                self.assertFalse((self.pkg / deliver.ENV_FILE).exists())

    def test_apply_refuses_a_private_file_the_release_would_carry(self):
        """Root's code-install looks for no private file (docker/rk_root.py), so apply's host side does, as a build
        does (deployment.stray_rows): a private file in the package folder that is not its own refuses a change that
        restarts, under --dry-run too, before the doctor: FAILED, exit 4, with its row, and nothing changed. The file
        is there before the confirms: one that comes after a confirm changes this package's code (state 5).
        """
        self.stray()
        rows, _made, _was = self.change_waits()
        self.answer(rows)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        for dry in (("--dry-run",), ()):
            rc, out, err = self.rkmon("apply", *dry, **self.running())
            self.assertEqual(rc, 4, out + err)
            self.assertIn("FAIL  the package folder: ", out)
            self.assertIn("docker/notes.yaml", out)
            self.assertIn("\nFAILED: ", out)
        nothing = ("rk doctor", "rk drain", "code-install", LAYER_RUN, "up -d")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)

    def test_apply_puts_back_a_hand_edit_of_docker_env(self):
        """A hand edit of docker/.env (state 9) is apply's to put back: --dry-run shows it under "Changed outside
        your file", and apply writes your file's value back, the settings kept before staying as they are; then
        nothing reads as changed outside your file."""
        self.confirmed()
        before = self.delivered()
        made = deploy_folder.plan(self.pkg, self.settings)
        self.answer(self.apply_answers(before, made, made))
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        edited = ran.replace(f"\nRKMON_PORT={self.port}\n", "\nRKMON_PORT=18620\n")
        self.assertNotEqual(edited, ran)
        dotenv.write_text(edited, encoding="utf-8")
        rc, out, err = self.rkmon("apply", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\nChanged outside your file (apply puts your file's value back):\n"
            f"  ~ features.dashboards.admin.port, as docker/.env has it  18620 -> {self.port}\n",
            out,
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), edited)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay\n",
            out,
        )
        self.assertTrue(
            out.endswith(rewrapped(f"\nnext: step 7, doctor: {self.w} doctor\n")), out
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(self.kept_env())
        rc, out, err = self.rkmon("next", **self.running())
        self.assertNotIn("changed outside your file", out)

    def test_apply_on_a_stopped_container(self):
        """On a stopped container apply puts this package's code and the settings in place through one-off containers
        of the image, and the container stays stopped: APPLIED, then state 12's next line, the start; no doctor,
        drain or restart. The records keep that the apply put a release in (records.Installed), which rollback reads.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("apply", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                f"  code: {INSTALLED}",
                "  wrote: the new settings (the previous ones kept), docker/.env and the override",
                "APPLIED: your confirmed settings and this package's code are in place; the container stays "
                "stopped",
                *rewrapped(f"next: step 6, start: {self.w} start").splitlines(),
            ],
        )
        self.assertNotEqual(
            self.called(r"run --rm -T --no-deps -e RKMON_AS=root rkmon code-install$"),
            [-1],
        )
        none = ("rk doctor", "rk drain", "up -d", r" exec ")
        self.assertEqual(self.called(*none), [-1] * len(none))
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNotNone(kept.installed)

    def test_apply_interrupted_before_and_after_the_write(self):
        """Ctrl-C during apply, exit 130: during the drain, nothing changed, the drain is lifted,
        the record of the apply goes, and the command again; once the settings are written, the record of the apply
        stays (state 3), whose next line is apply, which finishes the change, the settings kept by the first write
        staying the ones kept."""
        rows, made, was = self.change_waits()
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.answer([[r"rk drain --timeout-minutes 60$", DRAINED[0], "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: nothing changed; the drain is lifted and new reviews start again; your confirmed "
                    "change still waits",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        # rk drain goes on in the container once docker's client is stopped: SIGINT ends it first, its own way, its
        # drain lifted, so that it sets none after rk undrain
        ended = re.escape(r"pkill -INT -f rkmon\.py drain( |$)") + "$"
        ended, undrained = self.called(ended, r"rk undrain$")
        self.assertTrue(-1 < ended < undrained, self.calls())
        self.assertEqual(self.called("code-install", LAYER_RUN), [-1, -1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)
        self.log.unlink()
        self.answer([[r"compose up -d --force-recreate$", "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: after writing the new settings (the previous ones are kept)",
                    f"next: {self.w} apply (it finishes the change, or puts the previous settings back on a new "
                    "failure)",
                )
            ),
            out,
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), made.env)
        self.assertEqual(self.kept_env(), ran)
        self.assertEqual(records.read(self.pkg).under_way.command, "apply")
        # state 3: the layer is the new one, and apply finishes the change
        self.log.unlink()
        self.answer([[LAYER_READ, deliver.texts(made)[2], 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay\n",
            out,
        )
        self.assertTrue(
            out.endswith(rewrapped(f"\nnext: step 7, doctor: {self.w} doctor\n")), out
        )
        self.assertNotEqual(self.called(r"compose up -d --force-recreate$"), [-1])
        self.assertIsNone(records.read(self.pkg).under_way)
        self.assertEqual(self.kept_env(), ran)

    def test_ctrl_c_before_apply_s_change_ends_it_by_the_signal(self):
        """Ctrl-C before apply's change, during its guards (here check 15's ports), ends apply by the signal, as at a
        question (main), with no traceback: no doctor, no drain, nothing written, no record.
        """
        rows, made, was = self.change_waits()
        ran = (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8")
        self.answer([[r"\{\{\.Ports\}\}$", "", "SIGINT"], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("INTERRUPTED", out)
        self.assertNotEqual(self.called(r"\{\{\.Ports\}\}$", start=start), [-1])
        none = ("rk doctor", "rk drain", "code-install", LAYER_RUN)
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        dotenv = (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8")
        self.assertEqual(dotenv, ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_ctrl_c_once_apply_s_change_ended_or_stopped_ends_it_by_the_signal(self):
        """Ctrl-C once apply's change has ended ends apply by the signal, with no traceback: a second Ctrl-C while
        INTERRUPTED is said, after the first stopped the change at the restart, the record of the apply kept (state
        3); then, apply finishing the change, Ctrl-C while APPLIED's next line is read, the change made and its record
        gone."""
        rows, made, was = self.change_waits()
        self.answer(
            [
                [RESTART, "", "SIGINT"],
                ["", "", "SIGINT", {}, {"after": f"(?m){RESTART}"}],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("INTERRUPTED", out)
        self.assertEqual(records.read(self.pkg).under_way.command, "apply")
        self.log.unlink()
        self.answer(
            [
                ["", "", "SIGINT", {}, {"after": f"(?ms){LOAD}.* rk doctor "}],
                [LAYER_READ, deliver.texts(made)[2], 0],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotRegex(out, r"(?m)^APPLIED:")
        self.assertNotIn(-1, self.called(RESTART, r"rk undrain$"))
        self.assertIsNone(records.read(self.pkg).under_way)
        dotenv = (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8")
        self.assertEqual(dotenv, made.env)

    def test_apply_that_cannot_read_the_layer_in_place_changes_nothing(self):
        """apply whose records do not say the layer in place is the plan's, while the settings volume cannot be read
        (apply's step 5): ERROR, never a keep (deliver.keep_previous): nothing written, the code this run put in put
        back, the drain lifted, the record of the apply gone, and apply again."""
        rows, made, was = self.change_waits()
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.answer(
            [[LAYER_READ, UNREAD, 1, {}, {"after": r"(?m)code-install$"}], *rows]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the settings volume's deploy.json could not be read, to keep it for rollback (docker "
                    f"compose run exited 1: {UNREAD}): nothing changed",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        order = self.called(r"code-install$", r"code-back$", r"rk undrain$")
        self.assertNotIn(-1, order)
        self.assertEqual(order, sorted(order))
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(self.kept_env())
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_apply_whose_team_s_skill_changed_after_its_plan_refuses(self):
        """A team's skill that changed after apply's plan, before deliver reads the skills (apply's step 5;
        deliver.SkillsChanged): REFUSED, exit 3, an earlier step not done: nothing written, the code this run put in
        put back, the drain lifted, the record of the apply gone, and the state's next line, the confirm (state 5).
        With the skills as confirmed again, apply runs again: when the code it put in cannot be put back, ERROR says
        so, and the record of the apply stays."""
        skills = self.home / "team-skills"
        (skills / "agents").mkdir(parents=True)
        (skills / "agents" / "a.md").write_text("# a\n", encoding="utf-8")
        team = {"review_automation": {"skills_folder": str(skills)}}
        self.write_settings(features=team)
        self.confirmed()
        before = self.delivered()
        was = deploy_folder.plan(self.pkg, self.settings)
        self.write_settings(features=team, memory_gib=13)
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        rows = self.apply_answers(before, was, made)
        added = skills / "agents" / "b.md"
        # the read of the layer in place, just before deliver: a skill is added meanwhile
        read = [LAYER_READ, deliver.texts(was)[2], 0, {}]
        read.append({"after": r"(?m)code-install$", "makes": str(added)})
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        changed = f"the team's skills in {skills} changed since your confirm (agents/b.md): nothing changed"
        self.answer([read, *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    f"REFUSED: {changed}", f"next: step 3, confirm: {self.w} confirm"
                )
            ),
            out,
        )
        order = self.called(r"code-install$", r"code-back$", r"rk undrain$")
        self.assertNotIn(-1, order)
        self.assertEqual(order, sorted(order))
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(self.kept_env())
        self.assertIsNone(records.read(self.pkg).under_way)
        added.unlink()
        self.log.write_text("", encoding="utf-8")
        failed = [r"code-back$", "FAILED: the code volume could not be written", 1]
        self.answer([failed, read, *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            f"ERROR: {changed}, but this package's code went in and could not be put back",
            one_line(out),
        )
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertIsNotNone(records.read(self.pkg).under_way)

    def test_apply_again_after_a_ctrl_c_at_the_restart_keeps_the_settings_that_ran(
        self,
    ):
        """apply again after one that Ctrl-C stopped at the restart, its settings written (state 3), keeps the settings
        kept before, those that ran, though the settings volume cannot be read then: the records say the layer in
        place is the plan's (deliver.keep_previous), so rollback can still reach them.
        """
        rows, made, was = self.change_waits()
        ran = (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8")
        self.answer([[r"compose up -d --force-recreate$", "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertEqual(self.kept_env(), ran)
        self.log.unlink()
        before_the_write = f"^(?!(?s:.*){LAYER_RUN})"
        self.answer([[LAYER_READ, UNREAD, 1, {}, {"after": before_the_write}], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay\n",
            out,
        )
        self.assertEqual(self.kept_env(), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_put_back_after_a_stopped_apply_puts_back_the_code_it_installed(self):
        """An apply that stopped part-way after this package's code went in (state 3) leaves the install in its record
        (records.UnderWay.installed). Apply again, whose code-install finds that code in place (NOTHING TO DO), then
        puts that code back too on a new doctor failure (code-back, apply's step 8), with the settings from before the
        first apply: ROLLED BACK."""
        rows, made, was = self.change_waits()
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.answer([[r"compose up -d --force-recreate$", "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(records.read(self.pkg).under_way.installed)
        self.log.unlink()
        after = r"(?m)compose up -d --force-recreate$"
        same = "NOTHING TO DO: this code runs already (release 1)"
        self.answer(
            [
                *rows[:2],  # the layer after the write and after the put-back
                # before them, the layer the first apply wrote (state 3)
                [LAYER_READ, deliver.texts(made)[2], 0],
                [r"code-install$", same, 0],
                [
                    r"rk doctor --fresh --json$",
                    self.doctor("model fast"),
                    0,
                    {},
                    {"after": after},
                ],
                *rows[2:],
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(f"\n  code: {same}\n", out)
        self.assertIn("\nROLLED BACK: ", out)
        calls = self.calls()
        runs = [i for i, c in enumerate(calls) if re.search(LAYER_RUN, c)]
        ups = [i for i, c in enumerate(calls) if re.search(after, c)]
        backs = [i for i, c in enumerate(calls) if c.endswith(" code-back")]
        self.assertEqual((len(runs), len(ups), len(backs)), (2, 2, 1), calls)
        self.assertTrue(runs[1] < backs[0] < ups[1], calls)
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_ctrl_c_during_apply_s_code_install_waits_for_it_and_its_record(self):
        """Ctrl-C while apply's code-install runs (apply's step 4) is taken once it has ended and the record of the
        apply says the release that went in (records.UnderWay.installed), so that a put-back puts it back: INTERRUPTED,
        exit 130, while this package's code went in, then state 3's next line; nothing after the install runs.
        """
        rows, made, was = self.change_waits()
        ended = self.d / "ended"
        self.answer(
            [[r"code-install$", INSTALLED, "SIGINT", {}, {"ends": str(ended)}], *rows]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: while this package's code went in (the code before it is kept)",
                    f"next: {self.w} apply (it finishes the change, or puts the previous settings back on a new "
                    "failure)",
                )
            ),
            out,
        )
        self.assertTrue(ended.exists())
        self.assertIn(f"\n  code: {INSTALLED}\n", out)
        self.assertIn("Ctrl-C: the command above runs to its end first", err)
        self.assertTrue(records.read(self.pkg).under_way.installed)
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])

    def test_apply_again_that_ends_short_leaves_the_change_part_way(self):
        """Apply again after one that stopped part-way (state 3), ended short before it changed anything (here its
        code-install fails), leaves the record of the apply, the first one's install in it: the change is part-way
        still (records.Lease.stop), and the next line is state 3's."""
        rows, made, was = self.change_waits()
        self.answer([[r"compose up -d --force-recreate$", "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.log.unlink()
        broken = "ERROR: the code volume could not be read"
        self.answer(
            [
                [r"code-install$", broken, 1],
                [LAYER_READ, deliver.texts(made)[2], 0],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    f"ERROR: this package's code did not go in (code-install exited 1: {broken}); the settings are "
                    "as they were",
                    f"next: {self.w} apply (it finishes the change, or puts the previous settings back on a new "
                    "failure)",
                )
            ),
            out,
        )
        under_way = records.read(self.pkg).under_way
        self.assertEqual((under_way.command, under_way.installed), ("apply", True))
        self.assertFalse(records.runs(self.pkg))

    def test_a_drain_that_runs_out_of_time_changes_nothing(self):
        """A drain that runs out of time (--timeout-minutes; rk drain lifts it then): ERROR, exit 1, nothing changed,
        and the command with twice the time, or later."""
        rows, made, was = self.change_waits()
        timeout = (
            "drain: nothing new starts; the work in flight finishes first (up to 0 minutes): 2 reviews\n"
            "DRAIN TIMEOUT after 0 minutes: 2 reviews still in flight; the drain is lifted"
        )
        self.answer([[r"rk drain --timeout-minutes 0.0001$", timeout, 1], *rows])
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon(
            "apply", "--timeout-minutes", "0.0001", **self.running()
        )
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            "\n  DRAIN TIMEOUT after 0 minutes: 2 reviews still in flight; the drain is lifted\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the work in flight did not finish within 0.0001 minutes; the drain is lifted; nothing "
                    "changed",
                    f"next: {self.w} apply --timeout-minutes 0.0002, or {self.w} apply later",
                )
            ),
            out,
        )
        none = ("code-install", LAYER_RUN, "up -d", r"rk undrain$")
        self.assertEqual(self.called(*none), [-1] * len(none))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_apply_with_no_drain_drains_nothing(self):
        """apply --no-drain (apply's step 2): no drain, so the reviews in flight stop with the restart and start again
        later; every other step runs as without it. --dry-run says so. Its doctor after, with no check failing, is
        deploy step 7 passed since the restart, and a tick has run since then: the done line, and no tick asked (rk
        tick-now).
        """
        rows, made, was = self.change_waits()
        self.answer([*STARTED, TICK_SINCE_START, *rows])
        rc, out, err = self.rkmon("apply", "--no-drain", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "1 change. apply will: stop the work in flight (it starts again later), write the new settings (the "
            "previous ones kept), make the container again,",
            one_line(out),
        )
        rc, out, err = self.rkmon("apply", "--no-drain", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("drain", out)
        self.assertEqual(self.called(r"rk drain", r"rk undrain$"), [-1, -1])
        order = self.called(
            r"rk doctor --fresh --json$", r"code-install$", LAYER_RUN, RESTART, LOAD
        )
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(order, sorted(order), self.calls())
        self.assertTrue(
            out.endswith(
                finished(
                    "APPLIED: the deployment runs your confirmed settings (1 change)",
                    "done: the monitor runs your confirmed settings",
                )
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk tick-now" in c])
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )

    def test_a_drain_another_command_lifts_changes_nothing(self):
        """A drain that another command lifts before the reviews in flight finish (rk undrain, or rk start, meanwhile)
        ends rk drain with exit 1 before its limit, as the limit does once it passes, so the time it took tells the two
        apart: ERROR, exit 1, nothing changed, and the command again; not the limit's outcome. rk drain's own
        line says that the drain was lifted (UNDRAINED): an exit 1 with none, a traceback's, is rk drain's failure,
        which the ERROR names with its last line.
        """
        # rk drain's own word for a drain lifted meanwhile, which the wrapper reads
        from rkmon_lib.drain import UNDRAINED

        rows, made, was = self.change_waits()
        lifted = f"{UNDRAINED}: the flag was lifted meanwhile (rk undrain, or rk start)"
        failed = "Traceback (most recent call last):\nOSError: [Errno 28] No space left on device"
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        for said, why in (
            (
                lifted,
                "another command lifted the drain before the work in flight finished",
            ),
            (
                failed,
                "rk drain in the container exited 1 (OSError: [Errno 28] No space left on device)",
            ),
        ):
            with self.subTest(said=said[:9]):
                self.answer(
                    [
                        [r"rk drain --timeout-minutes 60$", f"{DRAINED[0]}\n{said}", 1],
                        *rows,
                    ]
                )
                rc, out, err = self.rkmon("apply", **self.running())
                self.assertEqual(rc, 1, out + err)
                shown = "".join(f"\n  {x}" for x in (DRAINED[0], *said.splitlines()))
                self.assertIn(f"{shown}\n", out)
                self.assertTrue(
                    out.endswith(
                        finished(
                            f"ERROR: {why}; nothing changed", f"next: {self.w} apply"
                        )
                    ),
                    out,
                )
                none = ("code-install", LAYER_RUN, "up -d", r"rk undrain$")
                self.assertEqual(self.called(*none), [-1] * len(none))
                self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
                self.assertIsNone(records.read(self.pkg).under_way)

    def test_code_that_needs_a_new_image_is_refused(self):
        """Code that cannot run on the image (code-install's exit, code_volume.NEW_IMAGE) refuses the apply, nothing
        changed, the drain lifted, with the fix that builds a new image (releases.new_image_fix), then confirm.
        """
        rows, made, was = self.change_waits()
        self.answer([[r"code-install$", "", code_volume.NEW_IMAGE], *rows])
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            " REFUSED: this package's code needs a new image; nothing changed fix: build an image from the new "
            "package (QUICK_BUILD_GUIDE.md), then set image.tag in your deploy settings file to its tag, then "
            f"{self.w} confirm next: {self.w} apply",
            one_line(out),
        )
        self.assertTrue(out.endswith(rewrapped(f"\nnext: {self.w} apply\n")), out)
        self.assertNotEqual(self.called(r"rk undrain$"), [-1])
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_a_code_install_that_fails_says_what_it_changed(self):
        """A code-install that fails (here after it emptied a code volume that failed its check: docker/rk_root.py)
        ends the apply short: ERROR, exit 1, with code-install's own last line, which says what became of the code
        volume, the settings as they were, the drain lifted, the record of the apply gone, and the command again;
        never "nothing changed", which code-install's lines alone can say."""
        rows, made, was = self.change_waits()
        emptied = (
            "the code volume was not root's and read-only all the way (/code/releases): it was emptied first, so the "
            "image's own code runs until the release is in place",
            "ERROR: the release was not put in place ([Errno 28] No space left on device): the code volume was "
            "emptied, and the image's own code runs",
        )
        self.answer([[r"code-install$", "\n".join(emptied), 1], *rows])
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(f"\n  code: {emptied[0]}\n  code: {emptied[1]}\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    f"ERROR: this package's code did not go in (code-install exited 1: {emptied[1]}); the settings "
                    "are as they were",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$"), [-1])
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_apply_and_rollback_run_rk_provision_before_each_load(self):
        """apply, then rollback, each runs rk provision in the running container once the settings it puts in place
        are delivered and the container made again, before the load (provision_now), as every start runs it once the
        layer is delivered, so that a changed model binding is probed again: its rows are said, as it indents
        them, and nothing else of it."""
        rows, _made, _was = self.change_waits()
        row = "  PASS  local review loop's server: its model list and chat route answer at their own places"
        said = f"{row}\nprovisioned at 2026-11-22 08:00 UTC: nothing else to do"
        self.answer([[r"rk provision$", said, 0], *rows])
        for command in ("apply", "rollback"):
            with self.subTest(command=command):
                start = len(self.calls())
                rc, out, err = self.rkmon(command, **self.running())
                self.assertEqual(rc, 0, out + err)
                lines = out.splitlines()
                restart = next(
                    i for i, x in enumerate(lines) if x.startswith("  restart: ")
                )
                self.assertEqual(lines[restart + 1], row, out)
                self.assertTrue(lines[restart + 2].startswith("  loaded: "), out)
                self.assertNotIn("nothing else to do", out)
                order = self.called(
                    r"compose up -d --force-recreate$",
                    r"rk provision$",
                    r"rk applied --load --json$",
                    start=start,
                )
                self.assertEqual(order, sorted(order), self.calls())
                self.assertNotIn(-1, order, self.calls())

    # rk provision's first row, as it renders it, before a run that does not end well
    PROVISIONED_ROW = "  PASS  folders: in place in /work"

    def provision_ends(self, answer: list, *shown: str) -> None:
        """apply with ANSWER as the stand-in docker's answer to rk provision before the load (provision_now), a run that
        does not end well: SHOWN, the lines said between the restart and the load, its rows and nothing else of it,
        then the one line that says what failed; apply goes on, and the load and the doctor after judge it: APPLIED,
        exit 0."""
        rows, _made, _was = self.change_waits()
        self.answer([answer, *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        lines = out.splitlines()
        restart = next(i for i, x in enumerate(lines) if x.startswith("  restart: "))
        loaded = next(i for i, x in enumerate(lines) if x.startswith("  loaded: "))
        self.assertEqual(lines[restart + 1 : loaded], list(shown), out)
        self.assertIn("APPLIED: the deployment runs your confirmed settings", out)

    def test_a_provision_that_fails_with_a_traceback_says_its_last_line(self):
        """rk provision before the load that fails with a traceback (provision_ends): its rows are said, not the
        traceback's lines; its last line is what failed."""
        said = (
            f"{self.PROVISIONED_ROW}\nTraceback (most recent call last):\n"
            '  File "/opt/rocke_monitor/rkmon_lib/provision.py", line 303, in _run\n'
            "    deploy.run(cmds)\n"
            "OSError: [Errno 28] No space left on device"
        )
        self.provision_ends(
            [r"rk provision$", said, 1],
            self.PROVISIONED_ROW,
            "  set-up: the work volume is not set up in full (exit code 1: OSError: [Errno 28] No space left on "
            "device); apply goes on",
        )

    def test_a_provision_refused_says_its_refusal(self):
        """rk provision before the load that refuses, as rk does when the configuration cannot run (exit 4, its
        REFUSED line, then its fix: provision_ends): its REFUSED line is what failed, not the fix after it.
        """
        refused = "REFUSED: your confirmed settings cannot be read: settings.yaml: no such file"
        self.provision_ends(
            [
                r"rk provision$",
                f"{refused}\nfix: {self.w} confirm, then {self.w} apply",
                4,
            ],
            f"  set-up: the work volume is not set up in full (exit code 4: {refused}); apply goes on",
        )

    def test_a_provision_step_that_fails_is_named(self):
        """rk provision before the load whose step fails (provision_ends): its FAIL row is said with its fix, and the
        line names the step."""
        failed = (
            "  FAIL  clone: the clone of <org>/<repo> failed: the network is down",
            "        fix: start again, once the network is back",
        )
        self.provision_ends(
            [r"rk provision$", "\n".join((self.PROVISIONED_ROW, *failed)), 1],
            self.PROVISIONED_ROW,
            *failed,
            "  set-up: the work volume is not set up in full (clone failed); apply goes on",
        )

    def test_a_provision_not_done_in_time_goes_on_in_the_container(self):
        """rk provision before the load that is not done within its limit (PROVISION_MINUTES, about a second in this
        test's copy of the package, made before the confirms, which record its code), its client stopped then: one
        line says so (provision_ends)."""
        rkmon = self.pkg / "docker" / "rkmon"
        text = rkmon.read_text(encoding="utf-8")
        self.assertEqual(text.count("\nPROVISION_MINUTES = 30\n"), 1)
        rkmon.write_text(
            text.replace("\nPROVISION_MINUTES = 30\n", "\nPROVISION_MINUTES = 0.02\n"),
            encoding="utf-8",
        )
        slow = [r"rk provision$", self.PROVISIONED_ROW, 0, {}, {"sleeps": 30}]
        self.provision_ends(
            slow,
            "  set-up: not done within 0.02 minutes, and goes on in the container; apply goes on",
        )

    def test_apply_refuses_a_busy_port_or_a_volume_of_another_deployment(self):
        """Check 15 at apply, as at the start: a dashboard port the change opens that another program or
        container uses refuses it, with the key that sets the port, and so does a volume this deployment uses that
        another deployment's compose project made, with the fix that gives container.name a name of its own: REFUSED
        (exit 4), nothing run and nothing changed. Neither refusal comes from another monitor's settings, which apply
        never reads, whatever docker would answer."""
        self.confirmed()
        before = self.delivered()
        was = deploy_folder.plan(self.pkg, self.settings)
        port = sandbox.free_port()
        self.write_settings(port=port)
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        rows = [*self.apply_answers(before, was, made), *ELSEWHERE]
        none = (LAYER_RUN, "rk doctor", "rk drain", "code-install", "up -d")
        self.log.unlink(missing_ok=True)
        self.answer(rows)
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
            s.listen()
            rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"FAIL port: 127.0.0.1:{port} is in use by another program or container fix: set "
            "features.dashboards.admin.port in ~/rkmon-deploy.yaml to a port that is free on this host",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: a port the dashboards need is in use; nothing changed (fix each FAIL above)",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertEqual(self.called(*none), [-1] * len(none))
        self.assert_reads_no_other_monitor()
        work = f"{deploy_folder.NAME}_rkmon-work"
        config = {"name": deploy_folder.NAME, "volumes": {"rkmon-work": {"name": work}}}
        theirs = {
            "Driver": "local",
            "Labels": "com.docker.compose.project=other-deployment",
            "Name": work,
        }
        self.log.unlink()
        self.answer(
            [
                [r"^compose config --format json$", json.dumps(config), 0],
                [r"^volume ls ", json.dumps(theirs), 0],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual((rc, err), (4, ""), out)
        self.assertIn(
            f"FAIL volume: {work} is used by another deployment on this host fix: set container.name in "
            f"~/rkmon-deploy.yaml to a name of its own, such as {deploy_folder.NAME}-2 (docker compose ls "
            "--all lists the names in use)",
            one_line(out),
        )
        self.assertIn(
            "REFUSED: a volume this deployment would use is another deployment's; nothing changed (fix each FAIL "
            "above)",
            one_line(out),
        )
        self.assertEqual(self.called(*none), [-1] * len(none))
        self.assert_reads_no_other_monitor()


if __name__ == "__main__":
    unittest.main()
