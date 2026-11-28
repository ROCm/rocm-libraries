# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The first apply over a container that runs an older package's code, end to end: docker/rkmon run on a copy of the
package against the stand-in docker, as tests/deploy_path_fixture.py sets it up. It shows that package's doctor without
comparing it, drains the container that package's way, lifting that drain with that package's rk start where it changes
nothing, and on a failure stops the new container, naming the way back to that package; run again after an interrupt, it
ends as one. A stopped container is this package's when the records or its settings volume hold this package's delivery.
A docker/.env that an older package wrote gives docker compose the confirmed variables it lacks.
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

from deploy_path_fixture import (  # noqa: E402
    DRAINED,
    INSTALLED,
    LAYER_READ,
    LAYER_RUN,
    LOAD,
    MODEL_FIX,
    RESTART,
    DeployPathCase,
)
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import checks, deliver, outcome, progress, records  # noqa: E402


class TestOlderPackage(DeployPathCase):
    # the way back to the older package, as the fix of a first apply that failed names it
    WAY_BACK = 'go back to the earlier package: QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor", step 7'

    def test_the_first_apply_over_an_older_package(self):
        """The first apply over a container that runs an older package's code, which apply knows by its rk, which
        has no `rk applied`: the older package's doctor, shown and not compared; its own drain, `rk stop --drain`, then
        the monitor stopped; no release, since this package's image carries its code; the settings, the older package's
        docker/.env and override kept; the container made again from this package's compose file, whose start clears
        the stop the older drain left; the load; rk stopped, which finds no stop left, so nothing more clears it; the
        doctor after, with no FAIL. APPLIED says the monitor runs in standby, and the next line is standby off's; the
        records keep no change put back and none under way. Under --dry-run, apply says it would stop the new
        container on a failure, and runs nothing.
        """
        rows, made, older = self.older_package()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "make the container again, run the doctor, and stop the new container on a failure.",
            one_line(out),
        )
        nothing = ("rk doctor", "rk stop --drain", LAYER_RUN, "up -d")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))
        self.log.unlink()
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        at = progress.shown_time(records.parsed(self.confirm_record().at))
        w = self.w
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                "  PASS  doctor before: 2 checks of the older package's own, not compared with the doctor after",
                "  drain: the older package's own, up to 60 minutes",
                *self.OLDER_DRAINED,
                "  drain: done",
                "  wrote: the new settings (the previous ones kept), docker/.env and the override",
                "  restart: docker compose up -d --force-recreate",
                f"  loaded: the monitor runs the settings you confirmed at {at}",
                "  PASS  doctor after: 3 checks, no new failure",
                *finished(
                    "APPLIED: the deployment runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: {w} standby off, once {w} status and the dashboards read right",
                ).splitlines(),
            ],
        )
        order = self.called(
            r"rk doctor --fresh$",
            r"rk stop --drain --wait 60$",
            LAYER_RUN,
            RESTART,
            LOAD,
        )
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(order, sorted(order), self.calls())
        # after the load, rk stopped finds no stop left, so no rk start runs
        asked = self.called(r"rk stopped$", r"(?m)rk start$", start=order[-1])
        self.assertEqual((asked[0] > order[-1], asked[1]), (True, -1), self.calls())
        # the older doctor, asked with --json first, then the doctor after
        doctors = [
            i
            for i, c in enumerate(self.calls())
            if re.search(r"rk doctor --fresh --json$", c)
        ]
        self.assertEqual(len(doctors), 2, self.calls())
        self.assertLess(doctors[0], order[0])
        self.assertGreater(doctors[1], order[-1])
        none = ("code-install", "rk undrain", "rk drain ", "compose stop")
        self.assertEqual(self.called(*none), [-1] * len(none), self.calls())
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )
        self.assertEqual(self.kept_env(), older[0])
        self.assertEqual(
            (self.pkg / (deliver.OVERRIDE_FILE + deliver.KEPT)).read_text(
                encoding="utf-8"
            ),
            older[1],
        )
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNone(kept.put_back)
        self.assertIsNone(kept.installed)

    def test_a_first_apply_whose_stop_is_left_clears_it_after_the_load(self):
        """A first apply over an older package whose new container starts with the stop the older drain left still in
        its state, which its start's rk start did not clear: after the load, rk stopped finds it, and apply says so
        and clears it with rk start, then runs the doctor after; APPLIED, exit 0.
        """
        rows, _made, _older = self.older_package()
        left = {"after": rf"(?ms){RESTART}(?!.*rk start$)"}
        stopped = [r"rk stopped$", "stopped by rk stop --drain", 0, {}, left]
        self.answer([stopped, *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        lines = out.splitlines()
        loaded = next(i for i, x in enumerate(lines) if x.startswith("  loaded: "))
        self.assertEqual(
            lines[loaded + 1 : loaded + 3],
            [
                "  the monitor was stopped (stopped by rk stop --drain): clearing it with rk start",
                "  PASS  doctor after: 3 checks, no new failure",
            ],
        )
        self.assertIn(
            "APPLIED: the deployment runs your confirmed settings, in standby",
            one_line(out),
        )
        load = self.called(LOAD)[0]
        order = self.called(r"(?m)rk start$", r"rk doctor --fresh --json$", start=load)
        self.assertNotIn(-1, [load, *order], self.calls())
        self.assertLess(order[0], order[1], self.calls())

    def first_apply_fails(self, rows: list, failed: str, failure: str, fix) -> str:
        """The first apply over an older package (older_package) with ROWS first among the stand-in docker's answers,
        which fail its step FAILED, with FAILURE and FIX: exit 1, and what first_apply_failed says. What apply
        printed.
        """
        older_rows, made, older = self.older_package()
        self.answer([*rows, *older_rows])
        rc, out, err = self.rkmon(
            "apply", **self.running(), FAKE_DOCKER_STOPPED=str(self.d / "stopped")
        )
        self.assertEqual(rc, 1, out + err)
        self.first_apply_failed(out, made, older, (failed, failure, fix))
        return out

    def first_apply_failed(self, out: str, made, older, failed: tuple, start=0):
        """What a first apply over an older package that failed printed (OUT) and left, FAILED its step, what
        failed and its fix: the step's FAIL row, then the new container stopped (docker compose stop, among the calls
        from START on), ERROR with both ways forward, the fix or the way back to the earlier package, and state 10's
        next line. The records keep the change that failed, that of the plan MADE, why and both ways forward, and none
        under way; the older package's docker/.env and override, OLDER, are kept as they were.
        """
        failed, failure, fix = failed
        stopped = self.d / "stopped"
        self.assertIn(
            f"\n  FAIL  {failed}: {failure}\n  stop: docker compose stop\n", out
        )
        w = self.w
        steps = fix.parts if isinstance(fix, checks.Steps) else (fix,)
        both = checks.Steps((*steps, f"; or {self.WAY_BACK}"))
        self.assertTrue(
            out.endswith(
                finished(
                    f"ERROR: the new monitor failed its first load: {failure}; its container is stopped, and the older "
                    "monitor's files are kept, so it can run again"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(both))
                + finished(
                    f"next: {w} apply, once what failed is fixed (a fix in your file: {w} confirm first)"
                )
            ),
            out,
        )
        self.assertTrue(stopped.exists(), "the new container stopped")
        self.assertNotEqual(self.called(r"compose stop$", start=start), [-1])
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertEqual(
            (kept.put_back.fingerprint, kept.put_back.reason, kept.put_back.failure),
            (made.fingerprint, records.FIRST_APPLY, failure),
        )
        self.assertEqual(kept.put_back.fix, both)
        self.assertEqual(self.kept_env(), older[0])
        self.assertEqual(
            (self.pkg / (deliver.OVERRIDE_FILE + deliver.KEPT)).read_text(
                encoding="utf-8"
            ),
            older[1],
        )

    def test_a_first_apply_over_an_older_package_whose_load_fails(self):
        """A first apply over an older package whose load fails cannot put the older monitor back: the new
        container is stopped, state 10 (first_apply_fails), and no doctor runs after it. rkmon next names it, with both
        ways forward; rollback has nothing to do and names the way back to the earlier package; a start then starts
        the new container on the settings that apply put in place, as STARTED says."""
        w = self.w
        fix = checks.Command((w, "logins"))
        refused = {"refused": "the login gh is missing", "fix": checks.fix_to_json(fix)}
        out = self.first_apply_fails(
            [[LOAD, json.dumps(refused), 4]], "load", "the login gh is missing", fix
        )
        self.assertNotIn("doctor after:", out)
        stopped = {**self.running(), "FAKE_DOCKER_STOPPED": str(self.d / "stopped")}
        rc, out, err = self.rkmon("next", **stopped)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "the first apply over an older package failed and stopped the new container: the login gh is missing",
            one_line(out),
        )
        self.assertIn(f"fix: {w} logins; or {self.WAY_BACK}", one_line(out))
        rc, out, err = self.rkmon("rollback", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: no earlier version of this package is kept; to go back to the earlier package, see "
                    'QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor", step 7',
                    f"next: {w} apply, once what failed is fixed (a fix in your file: {w} confirm first)",
                )
            ),
            out,
        )
        rc, out, err = self.rkmon("start", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "STARTED: the first apply over an older package failed and stopped the new container: the login gh is "
            "missing, and the monitor runs the settings that apply put in place, in standby: it sends nothing until "
            f"you run {w} standby off next: ",
            one_line(out),
        )

    def test_a_first_apply_over_an_older_package_whose_doctor_fails(self):
        """A first apply over an older package after which a check of the doctor fails: no check fails before it
        that the new doctor can be compared with, so any FAIL stops the new container, state 10 (first_apply_fails).
        """
        failing = [
            r"rk doctor --fresh --json$",
            self.doctor("model fast"),
            0,
            {},
            {"after": f"(?m){RESTART}"},
        ]
        self.first_apply_fails(
            [failing],
            "doctor after",
            "model fast: no answer from the binding's base_url",
            MODEL_FIX,
        )

    def test_a_first_apply_whose_stop_cannot_be_cleared_stops_the_new_container(self):
        """A first apply over an older package after whose load the stop the older drain left is still there, and the
        rk start that clears it fails: the new container stopped, state 10 (first_apply_fails), and no
        doctor runs after it."""
        after = {"after": f"(?m){RESTART}"}
        left = [r"rk stopped$", "stopped by rk stop --drain", 0, {}, after]
        refused = [r"rk start$", "rk: the state is locked", 1, {}, after]
        out = self.first_apply_fails(
            [left, refused],
            "stop cleared",
            "rk start, which clears the stop left in the monitor, exited 1",
            checks.Steps(
                (checks.Command((self.w, "logs", "--tail", "50")), "says why")
            ),
        )
        self.assertIn(
            "\n  the monitor was stopped (stopped by rk stop --drain): clearing it with rk start\n",
            out,
        )
        self.assertNotIn("doctor after:", out)

    # What the older package's rk answers to the first apply's doctor and drain: its doctor, without --json; its
    # drain, rk stop --drain, whose stop stays until its own rk start, as it has no rk undrain
    OLDER_REPORT = (
        "PASS  ramdisk: fine\n"
        "FAIL  claude session: on\n"
        "      fix: needs the Claude session's key: ./rkmon setup --section claude\n"
        "WARN  Claude session settings: no autoCompactWindow\n"
        "      fix: ./rkmon setup --section claude\n"
        "OFF   services: off\n"
    )

    def test_the_older_doctor_s_failing_rows_are_shown_as_it_gives_them(self):
        """The doctor before the first apply over an older package is that package's own, shown and not
        compared: the count of its checks that ran, WARN as one fails, then each failing row as that doctor gives it, a
        FAIL as FAIL, and no fix, as it judges nothing; the apply goes on, and the doctor after counts none of them.
        """
        rows, _made, _older = self.older_package()
        self.answer([[r"rk doctor --fresh$", self.OLDER_REPORT, 4], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  WARN  doctor before: 3 checks of the older package's own, not compared with the doctor after; "
            "failing:\n  FAIL  claude session: on\n  drain: the older package's own, up to 60 minutes\n",
            out,
        )
        self.assertNotIn("fix:", out)
        self.assertIn("PASS  doctor after: 3 checks, no new failure", out)
        self.assertIn(
            "APPLIED: the deployment runs your confirmed settings, in standby",
            one_line(out),
        )

    def test_an_older_doctor_that_cannot_run_is_a_warn(self):
        """An older package's doctor that does not run judges nothing: its row says why, a WARN, and
        the apply goes on."""
        rows, _made, _older = self.older_package()
        broken = (
            "Traceback (most recent call last):\nOSError: [Errno 5] Input/output error"
        )
        self.answer([[r"rk doctor --fresh$", broken, 1], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "WARN doctor before: the older package's doctor did not run (rk doctor in the container exited 1: OSError: "
            "[Errno 5] Input/output error) drain: the older package's own, up to 60 minutes",
            one_line(out),
        )
        self.assertIn(
            "APPLIED: the deployment runs your confirmed settings, in standby",
            one_line(out),
        )

    def older_drain_ends(self, rows: list, code: int, *lines: str) -> str:
        """The first apply over an older package (older_package), ROWS first among the stand-in docker's answers, which
        end it in the older package's drain: exit CODE, apply's last LINES, the outcome and the next line,
        and its stop lifted by that package's own rk start after the drain, never by rk undrain, which its rk lacks;
        nothing written or recorded, and the older package's docker/.env and override as they were. What apply
        printed."""
        older_rows, _made, older = self.older_package()
        self.answer([*rows, *older_rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, code, out + err)
        self.assertTrue(out.endswith(finished(*lines)), out)
        drain, start = self.called(r"rk stop --drain --wait 60$", r"(?m)rk start$")
        self.assertNotEqual(drain, -1, self.calls())
        self.assertGreater(start, drain, self.calls())
        none = ("rk undrain", LAYER_RUN, "code-install", RESTART, "compose stop")
        self.assertEqual(self.called(*none), [-1] * len(none), self.calls())
        kept = records.read(self.pkg)
        self.assertEqual((kept.under_way, kept.put_back, kept.written), (None,) * 3)
        for rel, text in zip((deliver.ENV_FILE, deliver.OVERRIDE_FILE), older):
            self.assertEqual((self.pkg / rel).read_text(encoding="utf-8"), text)
        return out

    def test_a_failed_older_drain_is_lifted_by_the_older_rk_start(self):
        """An older package's drain that fails changes nothing (older_drain_ends): ERROR, exit 1, and apply again."""
        failed = (
            "DRAINING: nothing new starts\nOSError: [Errno 28] No space left on device"
        )
        self.older_drain_ends(
            [[r"rk stop --drain --wait 60$", failed, 1]],
            1,
            "ERROR: the older package's drain exited 1; the older monitor runs again; nothing changed",
            f"next: {self.w} apply",
        )

    def test_an_older_drain_the_monitor_is_not_stopped_after_changes_nothing(self):
        """An older package's drain that exits 0 while rk stopped then finds the monitor running (another rk start
        meanwhile) changes nothing (older_drain_ends): ERROR, exit 1, and apply again.
        """
        self.older_drain_ends(
            [[r"rk stopped$", "", 1]],
            1,
            "ERROR: the older monitor was not stopped after its drain; the older monitor runs again; nothing changed",
            f"next: {self.w} apply",
        )
        drain = self.called(r"rk stop --drain --wait 60$")[0]
        self.assertNotEqual(self.called(r"rk stopped$", start=drain), [-1])

    def test_an_older_drain_at_its_limit_says_the_work_in_flight_was_stopped(self):
        """An older package's drain whose limit passes stops the work still in flight and exits 0, as
        its rk does (DRAIN TIMEOUT): the monitor is stopped, so the apply goes on, and `drain: done` says how the drain
        ended."""
        rows, _made, _older = self.older_package()
        said = (
            "DRAINING: nothing new starts; waiting up to 60 min for the reviews in flight",
            "DRAIN TIMEOUT after 60 min: #12, #34 still in flight; they are stopped and start again at their heads "
            "after rk start",
        )
        self.answer([[r"rk stop --drain --wait 60$", "\n".join(said), 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n".join(
                (
                    "  drain: the older package's own, up to 60 minutes",
                    *said,
                    "  drain: done at its limit of 60 minutes: the work still in flight was stopped",
                    "  wrote: ",
                )
            ),
            out,
        )
        self.assertIn(
            "APPLIED: the deployment runs your confirmed settings, in standby",
            one_line(out),
        )

    def test_ctrl_c_in_the_older_drain_is_lifted_by_the_older_rk_start(self):
        """Ctrl-C in an older package's drain changes nothing (older_drain_ends): INTERRUPTED, exit 130, the drain
        lifted, and apply again. Its rk stop --drain goes on in the container when docker's client is stopped, and would
        stop the monitor once the work in flight finishes: SIGINT ends it there first, its own way out, which clears
        its drain and stops nothing."""
        ctrl_c = [r"rk stop --drain --wait 60$", self.OLDER_DRAINED[0], "SIGINT"]
        self.older_drain_ends(
            [ctrl_c],
            130,
            "INTERRUPTED: nothing changed; the drain is lifted and new reviews start again; your confirmed change "
            "still waits",
            f"next: {self.w} apply",
        )
        ended, start = self.called(
            re.escape(r"pkill -INT -f rkmon\.py stop --drain") + "$", r"(?m)rk start$"
        )
        self.assertNotEqual(ended, -1, self.calls())
        self.assertLess(ended, start, self.calls())

    # state 3's next line while a first apply over an older package stopped part-way, which puts nothing back
    FIRST_APPLY_NEXT = (
        "next: {} apply (it finishes the first apply, or, on a new failure, stops the new container and keeps the "
        "older monitor's files)"
    )

    def first_apply_interrupted(self, rows: list) -> tuple:
        """The first apply over an older package (older_package), ROWS first among the stand-in docker's answers,
        interrupted at its first load, once the new container runs this package's code: exit 130, INTERRUPTED,
        the older monitor's files kept, and state 3's next line, which says that apply finishes the first apply, with
        no put-back; the records keep the apply under way, marked as a first apply. rkmon next says the same, and
        rollback has nothing to do and names the way back to the earlier package. The new monitor's own drain and
        undrain end well. (the plan apply delivers, the older package's docker/.env and override, the count of the
        stand-in's calls so far).
        """
        older_rows, made, older = self.older_package()
        mark = str(self.d / "interrupted")
        self.answer(
            [
                *rows,
                [LOAD, "", "SIGINT", {}, {"unless": mark, "makes": mark}],
                [r"rk drain --timeout-minutes 60$", "\n".join(DRAINED), 0],
                [r"rk undrain$", "UNDRAINED: the drain since 13:58 is lifted", 0],
                *older_rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        w = self.w
        finishes = self.FIRST_APPLY_NEXT.format(w)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: after writing the new settings (the older monitor's files are kept)",
                    finishes,
                )
            ),
            out,
        )
        self.assertNotRegex(out, r"\bputs?\b")
        under_way = records.read(self.pkg).under_way
        self.assertEqual((under_way.command, under_way.first_apply), ("apply", True))
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(f"{w} apply stopped part-way", one_line(out))
        self.assertIn("\n" + finished(finishes), out)
        self.assertNotRegex(out, r"\bputs?\b")
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: no earlier version of this package is kept; to go back to the earlier package, see "
                    'QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor", step 7',
                    finishes,
                )
            ),
            out,
        )
        self.assertEqual(records.read(self.pkg).under_way, under_way)
        return made, older, len(self.calls())

    def test_a_first_apply_run_again_after_its_restart_ends_as_one(self):
        """A first apply over an older package interrupted after its restart (first_apply_interrupted), then run again:
        the new container runs this package's code, so the new monitor's own
        drain, then steps 3 to 7: the record of the apply, the settings written again, the older package's files the
        ones kept still, the restart, the load, the drain lifted and the doctor after. APPLIED in standby, with no
        code-install and nothing put back; the records keep no change under way and none put back.
        """
        made, older, start = self.first_apply_interrupted([])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        at = progress.shown_time(records.parsed(self.confirm_record().at))
        w = self.w
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                f"  {DRAINED[0]}",
                "  drain: done",
                "  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay",
                "  restart: docker compose up -d --force-recreate",
                f"  loaded: the monitor runs the settings you confirmed at {at}",
                "  PASS  doctor after: 3 checks, no new failure",
                *finished(
                    "APPLIED: the deployment runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: {w} standby off, once {w} status and the dashboards read right",
                ).splitlines(),
            ],
        )
        order = self.called(
            r"rk drain --timeout-minutes 60$",
            LAYER_RUN,
            RESTART,
            LOAD,
            r"rk undrain$",
            r"rk doctor --fresh --json$",
            start=start,
        )
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(order, sorted(order), self.calls())
        none = ("code-install", "code-back", "rk stop --drain", "compose stop")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )
        self.assertEqual(self.kept_env(), older[0])
        kept = records.read(self.pkg)
        self.assertEqual((kept.under_way, kept.put_back), (None, None))

    def test_a_first_apply_run_again_after_its_restart_stops_on_a_new_failure(self):
        """A first apply over an older package interrupted after its restart (first_apply_interrupted), then run again,
        its doctor after failing: it ends as a first apply does (first_apply_failed): the new container stopped,
        the records keeping FIRST_APPLY, ERROR, exit 1, with both ways forward. Nothing is put back: the settings this
        apply wrote stay, written once in this run, and the older package's files stay the ones kept.
        """
        failing = [
            r"rk doctor --fresh --json$",
            self.doctor("model fast"),
            0,
            {},
            {"after": f"(?ms){RESTART}.*{RESTART}"},
        ]
        made, older, start = self.first_apply_interrupted([failing])
        rc, out, err = self.rkmon(
            "apply", **self.running(), FAKE_DOCKER_STOPPED=str(self.d / "stopped")
        )
        self.assertEqual(rc, 1, out + err)
        failure = "model fast: no answer from the binding's base_url"
        failed = ("doctor after", failure, MODEL_FIX)
        self.first_apply_failed(out, made, older, failed, start)
        self.assertNotRegex(out, r"\bputs?\b")
        writes = [c for c in self.calls()[start:] if re.search(LAYER_RUN, c)]
        self.assertEqual(len(writes), 1, self.calls()[start:])
        self.assertEqual(
            self.called("code-install", "code-back", start=start), [-1, -1]
        )
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )

    def test_a_first_apply_interrupted_before_its_restart_makes_the_seven_steps_again(
        self,
    ):
        """A first apply over an older package interrupted at its restart, once the settings are written: the
        older drain's stop lifted by the older rk's own rk start, INTERRUPTED, the older monitor's files kept, and
        state 3's next line; the records keep the apply under way, marked. Run again, over the older code, which still
        runs: the seven steps, the change begun once, in the place of the one that stopped part-way: the older doctor
        and drain, the settings written again, the older package's files the ones kept still, the restart, the load
        and the doctor after; APPLIED, the records keeping no change under way and none put back.
        """
        twice = f"(?ms){RESTART}.*{RESTART}"
        rows, made, older = self.older_package(new=twice)
        mark = str(self.d / "interrupted")
        self.answer(
            [
                [RESTART, "", "SIGINT", {}, {"unless": mark, "makes": mark}],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 130, out + err)
        w = self.w
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: after writing the new settings (the older monitor's files are kept)",
                    self.FIRST_APPLY_NEXT.format(w),
                )
            ),
            out,
        )
        drained, started, undrained = self.called(
            r"rk stop --drain", r"(?m)rk start$", r"rk undrain$"
        )
        self.assertLess(-1, drained, self.calls())
        self.assertLess(drained, started, self.calls())
        self.assertEqual(undrained, -1, self.calls())
        self.assertTrue(records.read(self.pkg).under_way.first_apply)
        start = len(self.calls())
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        at = progress.shown_time(records.parsed(self.confirm_record().at))
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                "  PASS  doctor before: 2 checks of the older package's own, not compared with the doctor after",
                "  drain: the older package's own, up to 60 minutes",
                *self.OLDER_DRAINED,
                "  drain: done",
                "  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay",
                "  restart: docker compose up -d --force-recreate",
                f"  loaded: the monitor runs the settings you confirmed at {at}",
                "  PASS  doctor after: 3 checks, no new failure",
                *finished(
                    "APPLIED: the deployment runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: {w} standby off, once {w} status and the dashboards read right",
                ).splitlines(),
            ],
        )
        order = self.called(
            r"rk doctor --fresh$",
            r"rk stop --drain --wait 60$",
            LAYER_RUN,
            RESTART,
            LOAD,
            start=start,
        )
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(order, sorted(order), self.calls())
        drains = [c for c in self.calls()[start:] if re.search("rk stop --drain", c)]
        self.assertEqual(len(drains), 1, self.calls()[start:])
        self.assertEqual(self.kept_env(), older[0])
        self.assertEqual(
            (self.pkg / (deliver.OVERRIDE_FILE + deliver.KEPT)).read_text(
                encoding="utf-8"
            ),
            older[1],
        )
        kept = records.read(self.pkg)
        self.assertEqual((kept.under_way, kept.put_back), (None, None))

    def test_a_first_apply_that_stopped_part_way_waits_for_the_file_you_confirmed(
        self,
    ):
        """A first apply over an older package that stopped part-way (first_apply_interrupted), your settings file
        changed since the confirm: apply cannot finish it and puts nothing back, so it refuses, nothing changed,
        with both ways forward, the file put back as you confirmed it, then apply, or the way back to the earlier
        package."""
        self.first_apply_interrupted([])
        self.write_settings(memory_gib=13)
        start = len(self.calls())
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 4, out + err)
        w = self.w
        apply = checks.Command((w, "apply"))
        fix = checks.Steps(
            (
                "put your settings file back as you confirmed it, then",
                apply,
                f"; or {self.WAY_BACK}",
            )
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: your settings file changed after the last confirm, so the first apply that stopped "
                    "part-way cannot be finished; nothing changed"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(fix))
                + finished(
                    f"next: {apply}, once your settings file is as you confirmed it"
                )
            ),
            out,
        )
        none = (r"rk (stop|drain|doctor)", LAYER_RUN, "compose up", "compose stop")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertTrue(records.read(self.pkg).under_way.first_apply)

    def older_stopped(self) -> tuple:
        """A deployment an older package made (older_package), its container stopped: (the stand-in docker's answers,
        the plan apply delivers, the older package's docker/.env and override, the stand-in's environment, in which
        the container is stopped until a compose up makes it run)."""
        rows, made, older = self.older_package()
        stopped = self.d / "stopped"
        stopped.touch()
        return (
            rows,
            made,
            older,
            {**self.running(), "FAKE_DOCKER_STOPPED": str(stopped)},
        )

    def test_the_first_apply_over_an_older_package_whose_container_is_stopped(self):
        """Over a stopped container an older package made, which apply knows
        as neither the records nor the settings volume hold a delivery of this package: rkmon next gives state 11's
        line, and start refuses with it, making nothing. apply makes steps 3 to 7, with no doctor before and no drain,
        since nothing runs: the record of the apply, the settings, the older package's docker/.env and override kept,
        the container made from this package's compose file, rk provision and the load, the doctor after. APPLIED in
        standby; the records keep no change under way and none put back. Under --dry-run, apply says it would stop
        the new container on a failure, and runs nothing."""
        rows, made, older, env = self.older_stopped()
        self.answer(rows)
        w = self.w
        rc, out, err = self.rkmon("next", **env)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "and the stopped container has the settings before them", one_line(out)
        )
        self.assertIn("\n" + finished(f"next: {w} apply"), out)
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "REFUSED: a change waits: you confirmed new settings", one_line(out)
        )
        self.assertTrue(out.endswith(finished(f"next: {w} apply")), out)
        rc, out, err = self.rkmon("apply", "--dry-run", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "apply will: write the new settings (the previous ones kept), make the container again, run the doctor, "
            "and stop the new container on a failure.",
            one_line(out),
        )
        nothing = ("compose up", "compose start", r" exec ", LAYER_RUN)
        self.assertEqual(self.called(*nothing), [-1] * len(nothing), self.calls())
        self.log.unlink()
        rc, out, err = self.rkmon("apply", **env)
        self.assertEqual(rc, 0, out + err)
        at = progress.shown_time(records.parsed(self.confirm_record().at))
        self.assertEqual(
            out.splitlines(),
            [
                "== apply: make the confirmed change on this deployment",
                "  wrote: the new settings (the previous ones kept), docker/.env and the override",
                "  restart: docker compose up -d --force-recreate",
                f"  loaded: the monitor runs the settings you confirmed at {at}",
                "  PASS  doctor after: 3 checks, no new failure",
                *finished(
                    "APPLIED: the deployment runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: {w} standby off, once {w} status and the dashboards read right",
                ).splitlines(),
            ],
        )
        order = self.called(
            LAYER_RUN, RESTART, r" rk provision$", LOAD, r"rk doctor --fresh --json$"
        )
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(order, sorted(order), self.calls())
        none = ("rk doctor --fresh$", "rk stop --drain", "rk drain", "code-install")
        self.assertEqual(self.called(*none), [-1] * len(none), self.calls())
        # nothing runs in the container before the restart makes it again
        self.assertGreater(self.called(r" exec ")[0], order[1], self.calls())
        self.assertEqual(self.kept_env(), older[0])
        self.assertEqual(
            (self.pkg / (deliver.OVERRIDE_FILE + deliver.KEPT)).read_text(
                encoding="utf-8"
            ),
            older[1],
        )
        kept = records.read(self.pkg)
        self.assertEqual((kept.under_way, kept.put_back), (None, None))

    def test_a_first_apply_over_an_older_stopped_container_whose_load_fails(self):
        """A first apply over a stopped container an older package made, whose load fails: it ends as every
        failed first apply does (first_apply_failed), the new container stopped again, with no doctor or drain
        before it."""
        rows, made, older, env = self.older_stopped()
        fix = checks.Command((self.w, "logins"))
        refused = {"refused": "the login gh is missing", "fix": checks.fix_to_json(fix)}
        self.answer([[LOAD, json.dumps(refused), 4], *rows])
        rc, out, err = self.rkmon("apply", **env)
        self.assertEqual(rc, 1, out + err)
        self.first_apply_failed(
            out, made, older, ("load", "the login gh is missing", fix)
        )
        none = ("rk doctor", "rk stop --drain", "rk drain", "code-install")
        self.assertEqual(self.called(*none), [-1] * len(none), self.calls())
        self.assertLess(self.called(RESTART)[0], self.called(r"compose stop$")[0])

    def apply_keeps_the_stopped_container(self, rows: list) -> None:
        """apply on a stopped container of this package's own deployment (change_waits), with ROWS first among the
        stand-in docker's answers: apply_stopped's way, as test_apply_on_a_stopped_container has it, not a first apply
        over an older package: this package's code and the settings in place through one-off containers, APPLIED, and
        the container stays stopped."""
        self.answer(rows)
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("apply", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines()[1:],
            [
                f"  code: {INSTALLED}",
                "  wrote: the new settings (the previous ones kept), docker/.env and the override",
                "APPLIED: your confirmed settings and this package's code are in place; the container stays "
                "stopped",
                *rewrapped(f"next: step 6, start: {self.w} start").splitlines(),
            ],
        )
        none = ("rk doctor", "rk drain", "up -d", r" exec ", "compose stop")
        self.assertEqual(self.called(*none), [-1] * len(none), self.calls())
        self.assertIsNone(records.read(self.pkg).put_back)

    def test_a_stopped_container_this_package_delivered_to_is_its_own_by_the_records(
        self,
    ):
        """A stopped container of this package's deployment whose settings volume's layer reads as none: the records
        keep this package's delivery (records.Written's layer, recorded before the container was made), so it is this
        package's own, and apply keeps the stopped container (apply_keeps_the_stopped_container).
        """
        rows, _made, _was = self.change_waits()
        none = f"cat: {deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}: No such file or directory"
        before = {"after": f"^(?!(?s:.*){LAYER_RUN})"}
        self.apply_keeps_the_stopped_container(
            [[LAYER_READ, none, 1, {}, before], *rows]
        )

    def test_a_stopped_container_this_package_delivered_to_is_its_own_by_its_volume(
        self,
    ):
        """A stopped container of this package's deployment whose records keep no delivery (records.Written empty):
        its settings volume holds this package's layer, which no older package's has, so it is this package's own, and
        apply keeps the stopped container (apply_keeps_the_stopped_container)."""
        rows, _made, _was = self.change_waits()
        records.change_written(self.pkg, lambda w: records.NOT_KEPT)
        self.assertEqual(records.read(self.pkg).written.layer, "")
        self.apply_keeps_the_stopped_container(rows)

    def test_a_docker_env_this_package_never_wrote_gives_compose_the_confirmed_variables(
        self,
    ):
        """A docker/.env this package's deliver never wrote, an older package's, which a move in place carries,
        lacks this package's variables, such as RKMON_CPUS, which its compose file needs: until the records keep the
        fingerprint of a docker/.env deliver wrote (records.Written's env), docker compose gets the confirmed plan's
        variables, as with no docker/.env, the confirmed image's tag among them, also in the reads of the container's
        state (compose ps), which pass none of their own; a record of the other files alone does not count. Once
        deliver has written docker/.env, docker compose reads that file, but for a variable it lacks that every deploy
        settings make (a later package's, which the docker/.env an earlier package's deliver wrote does not have): that
        one comes from the confirmed settings, and the file stays as it was."""
        rows, made, _older = self.older_package()
        self.answer(rows)
        log = self.d / "calls_and_variables.log"
        names = ("RKMON_CPUS", "RKMON_TAG")

        def variables() -> set:
            """The variables of NAMES that the reads of the container's state in `rkmon next` had, each NAME=value:
            the stand-in docker logs each call, then those of its variables, in one file here.
            """
            log.unlink(missing_ok=True)
            rc, out, err = self.rkmon(
                "next",
                **self.running(),
                FAKE_DOCKER_LOG=str(log),
                FAKE_DOCKER_ENV=",".join(names),
                FAKE_DOCKER_ENV_LOG=str(log),
            )
            self.assertEqual(rc, 0, out + err)
            seen, reads, call = set(), 0, ""
            for line in log.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    call = line.split("\t", 1)[1]
                    reads += bool(re.match(r"compose (-p \S+ )?ps ", call))
                elif re.match(r"compose (-p \S+ )?ps ", call):
                    seen.add(line)
            self.assertGreater(reads, 0, log.read_text(encoding="utf-8"))
            return seen

        plan = {f"{n}={made.variables[n]}" for n in names}
        self.assertEqual(variables(), plan)
        deliver.wrote(self.pkg, layer=deliver.texts(made)[2])
        self.assertEqual(variables(), plan)
        (self.pkg / deliver.ENV_FILE).write_text(
            deliver.texts(made)[0], encoding="utf-8"
        )
        deliver.wrote(self.pkg, env=deliver.texts(made)[0])
        self.assertEqual(variables(), set())
        # one an earlier package's deliver wrote, which a package update carries, lacks RKMON_CPUS
        lacking = re.sub(r"(?m)^RKMON_CPUS=.*\n", "", deliver.texts(made)[0])
        self.assertNotIn("\nRKMON_CPUS=", lacking)
        (self.pkg / deliver.ENV_FILE).write_text(lacking, encoding="utf-8")
        deliver.wrote(self.pkg, env=lacking)
        self.assertEqual(variables(), {f"RKMON_CPUS={made.variables['RKMON_CPUS']}"})
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), lacking
        )


if __name__ == "__main__":
    unittest.main()
