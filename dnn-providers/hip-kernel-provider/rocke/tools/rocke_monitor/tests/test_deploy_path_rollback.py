# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rollback, and apply's put-back, end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. rollback puts back the settings the last apply kept, and the code that ran with
them when that apply put a release in, on a running container (the doctor, the drain, the restart and the load) or a
stopped one. It has nothing to do while nothing is kept, and refuses a busy port of the settings it puts back and an
image that is not here; a failure or a Ctrl-C part-way says what changed and keeps its record, so that rollback again
finishes it.
"""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import re
import signal
import socket
import sys
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
    INSTALLED,
    LAYER_KEPT_READ,
    LAYER_PUT_BACK,
    LAYER_READ,
    LAYER_RUN,
    LAYER_WRITTEN,
    LOAD,
    MODEL_FIX,
    RESTART,
    UNREAD,
    DeployPathCase,
)
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import checks, deliver, outcome, records  # noqa: E402

# state 3's next line after a rollback stopped part-way
FINISHES = "(it finishes putting back the code and settings from before the last apply)"
# code-install's outcome where the release it puts in runs already (docker/rk_root.py)
RUNS_ALREADY = "NOTHING TO DO: this code runs already (release 1)"


class TestRollback(DeployPathCase):
    def test_rollback_puts_back_the_code_and_settings_the_last_apply_kept(self):
        """rollback after an apply: the doctor, the drain, the record of the rollback, the settings
        the apply kept, then the code it kept (code-back, in a container of the image they name) as the apply put a
        release in (records.Installed), though the kept layer names this package, the container made again, the load of
        the layer put back, the drain lifted and the doctor again, whose result the records keep as rkmon doctor's
        (record_doctor), as no check fails: RESTORED, and state 10's next line, plain apply. The records keep the change
        put back by rollback, with no failure, and the fingerprints of what it wrote, and no release the last apply put
        in: a second rollback has nothing to do. --dry-run says what it will do and changes nothing.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIsNotNone(records.read(self.pkg).installed)
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "rollback will: let the work in flight finish (up to 60 minutes), put back the code and settings from "
            "before your last apply, make the container again, and run the doctor.",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: the code and settings from before your last apply can be put back; nothing changed",
                    f"next: {self.w} rollback",
                )
            ),
            out,
        )
        nothing = (LAYER_RUN, "code-back", "up -d", "rk drain", "rk doctor")
        self.assertEqual(self.called(*nothing, start=start), [-1] * len(nothing))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), made.env)
        start = len(self.calls())
        # an earlier doctor's result, a failure, which the doctor after the rollback's change replaces
        records.doctor_result(
            self.pkg, records.Doctor("2026-11-15T08:00:00Z", 4, ("model fast",))
        )
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines(),
            [
                "== rollback: go back to the code and settings before the last apply",
                "  PASS  doctor before: 3 checks",
                f"  {DRAINED[0]}",
                "  drain: done",
                "  put back: the settings from before your last apply, docker/.env and the override",
                "  code: RESTORED: the image's code runs again",
                "  restart: docker compose up -d --force-recreate",
                "  loaded: the monitor runs the settings from before your last apply",
                "  PASS  doctor after: 3 checks, no new failure",
                "RESTORED: the deployment runs the code and settings from before your last apply",
                *rewrapped(f"next: {self.w} apply").splitlines(),
            ],
        )
        order = self.called(
            r"rk doctor --fresh --json$",
            r"rk drain --timeout-minutes 60$",
            LAYER_RUN,
            r"code-back$",
            r"compose up -d --force-recreate$",
            r"rk applied --load --json$",
            r"rk undrain$",
            start=start,
        )
        self.assertEqual(order, sorted(order), self.calls())
        self.assertNotIn(-1, order, self.calls())
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNone(kept.installed)
        self.assertEqual(kept.doctor, records.Doctor(mock.ANY, 3, ()))
        put_back = kept.put_back
        self.assertEqual(
            (put_back.fingerprint, put_back.reason, put_back.failure, put_back.fix),
            (made.fingerprint, "rollback", "", None),
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            f"the last change you confirmed was put back by {self.w} rollback",
            one_line(out),
        )
        self.assertNotIn("changed outside your file", out)
        self.assertTrue(out.endswith(rewrapped(f"\nnext: {self.w} apply\n")), out)
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: no earlier version of this package is kept",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )

    def test_rollback_with_no_kept_layer_has_nothing_to_do(self):
        """rollback when the settings volume keeps no earlier layer (the first apply over an older package keeps
        none), though docker/.env's kept copy is there: NOTHING TO DO and the state's next line, before the doctor and
        the drain; nothing run, nothing changed. --dry-run says the same."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        kept_file = f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_KEPT}"
        none_kept = f"cat: {kept_file}: No such file or directory"
        self.answer([[LAYER_KEPT_READ, none_kept, 1], *rows])
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        for dry in ((), ("--dry-run",)):
            with self.subTest(dry=dry):
                start = len(self.calls())
                rc, out, err = self.rkmon("rollback", *dry, **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(
                    out.endswith(
                        rewrapped(
                            "\nNOTHING TO DO: no earlier version of this package is kept\n"
                            f"next: step 7, doctor: {self.w} doctor\n"
                        )
                    ),
                    out,
                )
                self.assertNotEqual(self.called(LAYER_KEPT_READ, start=start), [-1])
                none = (LAYER_RUN, "rk doctor", "rk drain", "code-back", "up -d")
                self.assertEqual(self.called(*none, start=start), [-1] * len(none))
                self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
                self.assertIsNone(records.read(self.pkg).under_way)

    def test_rollback_that_cannot_read_the_settings_volume_changes_nothing(self):
        """rollback while the settings volume cannot be read, its kept layer (docker fails; not a layer that is not
        there, which cat says): ERROR, nothing changed, and rollback again; neither the doctor nor the drain runs.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.answer([[LAYER_KEPT_READ, UNREAD, 1], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the settings volume could not be read (docker compose run exited 1: "
                    f"{UNREAD}); nothing changed",
                    f"next: {self.w} rollback",
                )
            ),
            out,
        )
        none = (LAYER_RUN, "rk doctor", "rk drain", "code-back", "up -d")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_rollback_after_an_apply_that_put_no_release_in_leaves_the_code(self):
        """rollback puts the code back only when the last apply put a release in (records.code_back): after an apply
        whose code-install had nothing to do, as over a new image that carries this package's code, the settings go
        back and the code stays, though the kept layer names another package. --dry-run says so.
        """
        rows, made, was = self.change_waits()
        nothing = "NOTHING TO DO: this code runs already (the image's own code)"
        self.answer([[r"code-install$", nothing, 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"\n  code: {nothing}\n", out)
        self.assertIsNone(records.read(self.pkg).installed)
        earlier = json.loads(deliver.texts(was)[2])
        earlier["package_version"] = "0.0.1"
        self.answer([[LAYER_KEPT_READ, json.dumps(earlier), 0], *rows])
        rc, out, err = self.rkmon("rollback", "--dry-run", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "rollback will: let the work in flight finish (up to 60 minutes), put back the settings from before "
            "your last apply, make the container again, and run the doctor.",
            one_line(out),
        )
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("  code: ", out)
        self.assertEqual(self.called(r"code-back$", start=start), [-1])
        self.assertNotEqual(self.called(LAYER_RUN, start=start), [-1])
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the deployment runs the code and settings from before your last apply",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )

    def applied_then_edited(self) -> tuple:
        """An apply that put a release in (change_waits), then docker/.env edited by hand (state 9) and the stand-in
        docker's log emptied: (its answers for an apply that delivers your confirmed change again, whose code-install
        has nothing to do, the plan that apply delivers, the plan the first apply kept, and the records' Installed after
        the first apply)."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"\n  code: {INSTALLED}\n", out)
        installed = records.read(self.pkg).installed
        self.assertIsNotNone(installed)
        dotenv = self.pkg / deliver.ENV_FILE
        edited = made.env.replace(f"\nRKMON_PORT={self.port}\n", "\nRKMON_PORT=18620\n")
        self.assertNotEqual(edited, made.env)
        dotenv.write_text(edited, encoding="utf-8")
        self.log.unlink()
        rows = self.apply_answers(deploy_folder.runs(made), was, made)
        return [[r"code-install$", RUNS_ALREADY, 0], *rows], made, was, installed

    def test_rollback_after_an_apply_that_kept_nothing_puts_back_the_code_of_the_apply_before(
        self,
    ):
        """An apply that delivers your confirmed change again (a hand edit of docker/.env, state 9) puts no release in
        and keeps no settings of its own: the settings kept stay those from before the apply that put a release in, and
        the records keep that release (records.Installed). A rollback then puts back the code from before that apply
        with its settings (code-back), as its RESTORED line says."""
        rows, made, was, installed = self.applied_then_edited()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"\n  code: {RUNS_ALREADY}\n"
            "  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay\n",
            out,
        )
        self.assertEqual(records.read(self.pkg).installed, installed)
        self.assertEqual(self.kept_env(), was.env)
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n  put back: the settings from before your last apply, docker/.env and the override\n"
            "  code: RESTORED: the image's code runs again\n",
            out,
        )
        self.assertNotEqual(self.called(r"code-back$", start=start), [-1])
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the deployment runs the code and settings from before your last apply",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        self.assertIsNone(records.read(self.pkg).installed)

    def test_a_stopped_apply_that_kept_nothing_leaves_the_release_the_records_keep(
        self,
    ):
        """The same apply on a stopped container: it puts no release in and keeps no settings of its own, so the
        records keep the release the apply before it put in (records.Installed), which a rollback puts back with the
        settings kept."""
        rows, made, was, installed = self.applied_then_edited()
        self.answer(rows)
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("apply", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"\n  code: {RUNS_ALREADY}\n"
            "  wrote: your confirmed settings again, docker/.env and the override; the settings kept before stay\n",
            out,
        )
        self.assertEqual(records.read(self.pkg).installed, installed)
        self.assertEqual(self.kept_env(), was.env)

    def test_a_put_back_after_an_apply_that_kept_nothing_puts_back_the_code_of_the_apply_before(
        self,
    ):
        """The same apply, its load refused: its put-back puts back the settings kept, those from before the apply that
        put a release in, and with them the code that ran with them (records.code_back, as rollback decides: this apply
        kept no settings of its own, and the last apply put a release in), the container made again: ROLLED BACK. A
        rollback then finds the code and settings from before that apply in place, and has nothing to do.
        """
        rows, made, was, installed = self.applied_then_edited()
        fix = checks.Command(("rkmon", "logins"))
        refused = {"refused": "the login gh is missing", "fix": checks.fix_to_json(fix)}
        self.answer(rows[:6] + [[LOAD, json.dumps(refused), 4]] + rows[6:])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  FAIL  load: the login gh is missing\n", out)
        self.assertIn(
            "\n  put back: the settings from before this apply, docker/.env and the override\n"
            "  code: RESTORED: the image's code runs again\n"
            "  restart: docker compose up -d --force-recreate\n",
            out,
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
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        self.assertIsNone(records.read(self.pkg).installed)
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "\nNOTHING TO DO: no earlier version of this package is kept\n", out
        )
        self.assertEqual(self.called(r"code-back$", LAYER_RUN, start=start), [-1, -1])

    def test_rollback_after_an_apply_that_kept_the_settings_and_put_no_release_in_leaves_the_code(
        self,
    ):
        """An apply that put a release in, then another confirmed change whose apply puts none in (the release runs
        already) and keeps the settings before it: the records keep no release (records.Installed), so a rollback puts
        back those settings, from after the first apply, and leaves the code that ran with them, which runs.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIsNotNone(records.read(self.pkg).installed)
        self.write_settings(memory_gib=14)
        self.confirmed()
        newer = deploy_folder.plan(self.pkg, self.settings)
        self.log.unlink()
        rows = self.apply_answers(deploy_folder.runs(made), made, newer)
        self.answer([[r"code-install$", RUNS_ALREADY, 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"\n  code: {RUNS_ALREADY}\n"
            "  wrote: the new settings (the previous ones kept), docker/.env and the override\n",
            out,
        )
        self.assertIsNone(records.read(self.pkg).installed)
        self.assertEqual(self.kept_env(), made.env)
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("  code: ", out)
        self.assertEqual(self.called(r"code-back$", start=start), [-1])
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the deployment runs the code and settings from before your last apply",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), made.env
        )

    def test_a_put_back_that_puts_the_code_back_makes_the_container_again(self):
        """apply's put-back (Wrapper.put_back_change, in this process) after a change that did not make the container
        again: when the code goes back with the settings (records.code_back), it makes the container again, which runs
        the code it was made with; when the code stays, it does not."""
        self.confirmed()
        plan = deploy_folder.plan(self.pkg, self.settings)
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = mod.Wrapper(self.w)
        w.cmd = "apply"
        first = records.begin(self.pkg, "apply")
        first.record_install()
        first.applied()
        for coded in (True, False):
            with self.subTest(coded=coded), mock.patch.object(
                sys, "path", list(sys.path)
            ):
                change = mod.Change()
                change.lease = records.begin(self.pkg, "apply")
                self.assertIs(records.code_back(records.read(self.pkg)), coded)
                out = io.StringIO()
                with mock.patch.object(
                    w, "kept_fingerprint", return_value="sha256:kept"
                ), mock.patch.object(
                    w.lib("deliver"), "put_back", return_value=True
                ), mock.patch.object(
                    w, "root_code", return_value=(0, [])
                ) as root_code, mock.patch.object(
                    w, "restart", return_value=None
                ) as restart, mock.patch.object(
                    w, "wait_loaded", return_value=None
                ), mock.patch.object(
                    w, "state_next_line", return_value="next: apply"
                ), contextlib.redirect_stdout(
                    out
                ):
                    failure = ("the login gh is missing", None)
                    rc = w.put_back_change(plan, change, False, "load", failure)
                self.assertEqual(rc, 1, out.getvalue())
                self.assertEqual(
                    [c.args for c in root_code.call_args_list],
                    [("code-back",)] if coded else [],
                )
                self.assertEqual(restart.call_count, 1 if coded else 0)
                self.assertIsNone(records.read(self.pkg).installed)

    def test_rollback_refuses_a_busy_port_of_the_settings_it_puts_back(self):
        """rollback on the running container checks the dashboards' ports of the settings it puts back (the kept
        docker/.env's) before the doctor and the drain, as apply checks its own, under --dry-run
        too: one that another program or container uses refuses, its FAIL row's fix to stop what uses it, then rollback
        again; REFUSED, nothing run, nothing changed, and rollback again. A port of the settings in place alone refuses
        nothing, nor one this deployment's container publishes: in its compose project, whatever the kept docker/.env
        names."""
        self.confirmed()
        before = self.delivered()
        was = deploy_folder.plan(self.pkg, self.settings)
        port = sandbox.free_port()
        self.write_settings(port=port)
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        rows = self.apply_answers(before, was, made)
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        for dry in ((), ("--dry-run",)):
            with self.subTest(dry=dry), socket.socket() as s:
                s.bind(("127.0.0.1", self.port))
                s.listen()
                start = len(self.calls())
                rc, out, err = self.rkmon("rollback", *dry, **self.running())
                self.assertEqual(rc, 4, out + err)
                self.assertIn(
                    f"FAIL port: 127.0.0.1:{self.port} is in use by another program or container fix: stop the "
                    f"program or container that uses it, then {self.w} rollback",
                    one_line(out),
                )
                self.assertTrue(
                    out.endswith(
                        finished(
                            "REFUSED: a port the dashboards need is in use; nothing changed (fix each FAIL above)",
                            " ".join(("next:", self.w, "rollback", *dry)),
                        )
                    ),
                    out,
                )
                none = (LAYER_RUN, "rk doctor", "rk drain", "code-back", "up -d")
                self.assertEqual(self.called(*none, start=start), [-1] * len(none))
                self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
                self.assertIsNone(records.read(self.pkg).under_way)
        # the kept copy's compose project changed by hand: the port is this deployment's container's still
        project = made.variables["COMPOSE_PROJECT_NAME"]
        kept = self.pkg / (deliver.ENV_FILE + deliver.KEPT)
        text = kept.read_text(encoding="utf-8")
        edited = text.replace(
            f"\nCOMPOSE_PROJECT_NAME={project}\n",
            "\nCOMPOSE_PROJECT_NAME=another-project\n",
        )
        self.assertNotEqual(edited, text)
        kept.write_text(edited, encoding="utf-8")
        ours = [
            rf"^ps --filter label=com\.docker\.compose\.project={re.escape(project)} --format",
            f"127.0.0.1:{self.port}->8080/tcp",
            0,
        ]
        for bound, answers in ((port, rows), (self.port, [ours, *rows])):
            with self.subTest(bound=bound), socket.socket() as s:
                s.bind(("127.0.0.1", bound))
                s.listen()
                self.answer(answers)
                rc, out, err = self.rkmon("rollback", "--dry-run", **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertIn(
                    "\nPASSED: the code and settings from before your last apply can be put back",
                    out,
                )

    def test_rollback_again_after_a_ctrl_c_at_code_back_puts_the_code_back(self):
        """rollback again after one that Ctrl-C stopped at code-back (state 3) puts the code back too: the record
        of the first keeps whether the code goes back (records.UnderWay.code_back), which the second reads: RESTORED,
        and state 10's next line."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.answer([[r"code-back$", "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: while putting back the code and settings from before your last apply",
                    f"next: {self.w} rollback (it finishes putting back the code and settings from before the "
                    "last apply)",
                )
            ),
            out,
        )
        part_way = records.read(self.pkg).under_way
        self.assertEqual((part_way.command, part_way.code_back), ("rollback", True))
        self.answer(rows)
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn("\n  code: RESTORED: the image's code runs again\n", out)
        self.assertNotEqual(self.called(r"code-back$", start=start), [-1])
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the deployment runs the code and settings from before your last apply",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertIsNone(records.read(self.pkg).under_way)

    def test_rollback_whose_put_back_fails_says_what_changed(self):
        """rollback whose put-back fails: ERROR with what the settings volume and the files have, the drain
        lifted, no restart. When the one-off container changed nothing, the record of the rollback goes, and rollback
        again; when what it changed is not known, the record stays (state 3), and rollback finishes it.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        unknown = (
            "ended with exit code 1: the settings volume has either its layer as it was or the kept one; "
            "docker/.env and the override are as they were"
        )
        for code, said, next_line, under_way in (
            (
                deliver.UNCHANGED,
                "could not change it: nothing changed",
                f"next: {self.w} rollback",
                None,
            ),
            (1, unknown, f"next: {self.w} rollback {FINISHES}", "rollback"),
        ):
            with self.subTest(code=code):
                self.answer(
                    [[LAYER_RUN, "", code, {}, {"after": LAYER_WRITTEN}], *rows]
                )
                start = len(self.calls())
                rc, out, err = self.rkmon("rollback", **self.running())
                self.assertEqual(rc, 1, out + err)
                self.assertTrue(
                    out.endswith(
                        finished(
                            "ERROR: the settings from before your last apply could not be put back: the one-off "
                            f"container that writes the settings volume {said}",
                            next_line,
                        )
                    ),
                    out,
                )
                self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
                self.assertEqual(
                    self.called("code-back", "up -d", start=start), [-1, -1]
                )
                self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
                part_way = records.read(self.pkg).under_way
                self.assertEqual(part_way and part_way.command, under_way)

    def test_rollback_that_ends_short_after_the_put_back_keeps_its_record(self):
        """rollback that ends short once the settings are back: the code that does not go back (code-back fails),
        then, run again, a container that does not start: ERROR with what is back and what failed, the drain lifted,
        the record of the rollback kept (state 3), and rollback again, which carries on with the code. Run once more,
        a doctor that finds a new failure ends it: ERROR with that failure and its fix, the change put back by rollback
        with them (state 10), the record gone."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        unwritten = (
            "ERROR: the code volume could not be written: no space left on device"
        )
        self.answer([[r"code-back$", unwritten, 1], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            "\n  put back: the settings from before your last apply, docker/.env and the override\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the settings from before your last apply are back, but not its code (code-back exited 1: "
                    f"{unwritten})",
                    f"next: {self.w} rollback {FINISHES}",
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
        self.assertEqual(self.called("up -d", start=start), [-1])
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        part_way = records.read(self.pkg).under_way
        self.assertEqual((part_way.command, part_way.code_back), ("rollback", True))
        # again: the code goes back, and the container does not start
        self.answer([[RESTART, "", 1, {}, {"after": LAYER_PUT_BACK}], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  code: RESTORED: the image's code runs again\n", out)
        self.assertIn("\n  FAIL  restart: docker compose up exited 1\n", out)
        self.assertTrue(
            one_line(out).endswith(
                one_line(
                    finished(
                        "ERROR: the code and settings from before your last apply are back, but the container did not "
                        "start again: docker compose up exited 1",
                        f"fix: {self.w} logs --tail 50 says why",
                        f"next: {self.w} rollback {FINISHES}",
                    )
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
        self.assertEqual(records.read(self.pkg).under_way.command, "rollback")
        # once more: it ends, and the doctor after it finds a new failure (the third restart's)
        third = f"(?ms)(?:{RESTART}.*){{3}}"
        self.answer(
            [
                [
                    r"rk doctor --fresh --json$",
                    self.doctor("model fast"),
                    0,
                    {},
                    {"after": third},
                ],
                *rows,
            ]
        )
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        failure = "model fast: no answer from the binding's base_url"
        self.assertIn(f"\n  FAIL  doctor after: {failure}\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the code and settings from before your last apply are back, with a failure: "
                    f"{failure}"
                )
                + "".join(f"{x}\n" for x in outcome.fix_lines(MODEL_FIX))
                + finished(
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)"
                )
            ),
            out,
        )
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("rollback", failure, MODEL_FIX),
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            f"the last change you confirmed was put back by {self.w} rollback, which ended with a failure: "
            f"{failure}",
            one_line(out),
        )

    def test_rollback_whose_restart_fails_keeps_its_record(self):
        """rollback whose container does not start again once the code and settings are back, on its first run:
        the restart's FAIL row, ERROR with what is back and what failed, whose fix is the container's log, the drain
        lifted, the record of the rollback kept (state 3), and rollback again, which finishes the put-back.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIsNone(records.read(self.pkg).under_way)
        self.answer([[RESTART, "", 1, {}, {"after": LAYER_PUT_BACK}], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  FAIL  restart: docker compose up exited 1\n", out)
        self.assertTrue(
            one_line(out).endswith(
                one_line(
                    finished(
                        "ERROR: the code and settings from before your last apply are back, but the container did not "
                        "start again: docker compose up exited 1",
                        f"fix: {self.w} logs --tail 50 says why",
                        f"next: {self.w} rollback {FINISHES}",
                    )
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        self.assertEqual(records.read(self.pkg).under_way.command, "rollback")

    def test_rollback_whose_load_fails_ends_with_the_failure(self):
        """rollback whose settings put back the monitor refuses to load: the load's FAIL row in rk's words, the
        drain lifted, no doctor after it; ERROR with the failure and rk's fix, the change put back by rollback with them
        (state 10: apply, once what failed is fixed), the record of the rollback gone.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        fix = checks.Command(("rkmon", "logins"))
        refused = {"refused": "the login gh is missing", "fix": checks.fix_to_json(fix)}
        self.answer(
            [[LOAD, json.dumps(refused), 4, {}, {"after": LAYER_PUT_BACK}], *rows]
        )
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("\n  FAIL  load: the login gh is missing\n", out)
        self.assertNotIn("doctor after", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the code and settings from before your last apply are back, with a failure: the login gh "
                    "is missing",
                    "fix: rkmon logins",
                    f"next: {self.w} apply, once what failed is fixed (a fix in your file: {self.w} confirm first)",
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
        doctors = [c for c in self.calls()[start:] if "rk doctor" in c]
        self.assertEqual(len(doctors), 1, doctors)
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertEqual(
            (kept.put_back.reason, kept.put_back.failure, kept.put_back.fix),
            ("rollback", "the login gh is missing", fix),
        )

    def test_rollback_interrupted_before_the_put_back_changes_nothing(self):
        """Ctrl-C during rollback's drain, before anything is put back, exit 130: the drain
        lifted, the record of the rollback gone; INTERRUPTED, nothing changed, and rollback again, which puts back the
        settings from before the last apply."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.answer([[r"rk drain --timeout-minutes 60$", DRAINED[0], "SIGINT"], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: nothing changed; the drain is lifted and new reviews start again",
                    f"next: {self.w} rollback",
                )
            ),
            out,
        )
        self.assertNotEqual(self.called(r"rk undrain$", start=start), [-1])
        none = (LAYER_RUN, "code-back", "up -d")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)
        self.answer(rows)
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the deployment runs the code and settings from before your last apply",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        self.assertEqual(dotenv.read_text(encoding="utf-8"), was.env)

    def test_ctrl_c_outside_rollback_s_change_ends_it_by_the_signal(self):
        """Ctrl-C outside rollback's change ends rollback by the signal, as at a question (main), with no traceback:
        during its guards (here check 15's ports), no doctor, no drain, nothing put back, no record; then, rollback
        again putting back the code and settings, while RESTORED's next line is read, what it put back recorded and the
        record of the rollback gone."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.answer([[r"\{\{\.Ports\}\}$", "", "SIGINT"], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("INTERRUPTED", out)
        self.assertNotEqual(self.called(r"\{\{\.Ports\}\}$", start=start), [-1])
        none = ("rk doctor", "rk drain", LAYER_RUN, "code-back", "up -d")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        dotenv = self.pkg / deliver.ENV_FILE
        self.assertEqual(dotenv.read_text(encoding="utf-8"), made.env)
        self.assertIsNone(records.read(self.pkg).under_way)
        after = f"(?ms)code-back$.*{LOAD}.* rk doctor "
        self.answer([["", "", "SIGINT", {}, {"after": after}], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotRegex(out, r"(?m)^RESTORED:")
        back = ("code-back", RESTART, r"rk undrain$")
        self.assertNotIn(-1, self.called(*back, start=start))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), was.env)
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertEqual(kept.put_back.reason, "rollback")

    def test_ctrl_c_once_a_change_on_a_stopped_container_ended_ends_it_by_the_signal(
        self,
    ):
        """On a stopped container, Ctrl-C once apply's or rollback's change has ended, while the outcome's next line is
        read, ends the command by the signal, with no traceback: what the change did is recorded, as after its outcome
        (apply: the release it put in; rollback: what it put back), and no change is under way.
        """
        rows, made, was = self.change_waits()
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        self.log.unlink()
        self.answer([["", "", "SIGINT", {}, {"after": LAYER_RUN}], *rows])
        rc, out, err = self.rkmon("apply", **stopped)
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotRegex(out, r"(?m)^APPLIED:")
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertIsNotNone(kept.installed)
        dotenv = self.pkg / deliver.ENV_FILE
        self.assertEqual(dotenv.read_text(encoding="utf-8"), made.env)
        self.log.unlink()
        self.answer([["", "", "SIGINT", {}, {"after": r"(?m)code-back$"}], *rows])
        rc, out, err = self.rkmon("rollback", **stopped)
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotRegex(out, r"(?m)^RESTORED:")
        kept = records.read(self.pkg)
        self.assertIsNone(kept.under_way)
        self.assertEqual(kept.put_back.reason, "rollback")
        self.assertEqual(dotenv.read_text(encoding="utf-8"), was.env)

    def test_ctrl_c_during_a_change_on_a_stopped_container_ends_it_interrupted(self):
        """On a stopped container, Ctrl-C during apply's or rollback's change ends it with
        INTERRUPTED, exit 130, with no traceback, the record of the change kept (state 3): apply stopped while the
        settings are written; then rollback, stopped while the code is put back."""
        rows, made, was = self.change_waits()
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        self.answer([[LAYER_RUN, "", "SIGINT"], *rows])
        rc, out, err = self.rkmon("apply", **stopped)
        self.assertEqual(rc, 130, out + err)
        self.assertNotIn("Traceback", err)
        self.assertRegex(out, r"(?m)^INTERRUPTED: ")
        self.assertEqual(records.read(self.pkg).under_way.command, "apply")
        self.answer([[r"code-back$", "", "SIGINT"], *rows])
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **stopped)
        self.assertEqual(rc, 130, out + err)
        self.assertNotIn("Traceback", err)
        self.assertRegex(out, r"(?m)^INTERRUPTED: ")
        self.assertNotEqual(self.called(r"code-back$", start=start), [-1])
        self.assertEqual(records.read(self.pkg).under_way.command, "rollback")

    def test_rollback_with_nothing_kept_has_nothing_to_do(self):
        """Before an apply has kept anything (a first start keeps nothing), rollback has nothing to do: NOTHING TO
        DO with the state's next line, nothing changed and nothing recorded."""
        self.confirmed()
        before = self.delivered()
        made = deploy_folder.plan(self.pkg, self.settings)
        self.answer(self.apply_answers(before, made, made))
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: no earlier version of this package is kept",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        none = (LAYER_RUN, "code-back", "rk drain", "rk doctor")
        self.assertEqual(self.called(*none), [-1] * len(none))
        self.assertIsNone(records.read(self.pkg).put_back)

    def test_rollback_on_a_stopped_container(self):
        """On a stopped container rollback puts back the settings the last apply kept through a one-off container, and
        the container stays stopped: no doctor, drain or restart; the code stays when the last apply put no
        release in, its code-install having nothing to do (no code-back). RESTORED, and state 10's next line.
        """
        rows, made, was = self.change_waits()
        nothing = "NOTHING TO DO: this code runs already (the image's own code)"
        self.answer([[r"code-install$", nothing, 0], *rows])
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        start = len(self.calls())
        self.answer(rows)
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("rollback", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines()[:2],
            [
                "== rollback: go back to the code and settings before the last apply",
                "  put back: the settings from before your last apply, docker/.env and the override",
            ],
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "RESTORED: the code and settings from before your last apply are in place; the container stays "
                    "stopped",
                    f"next: {self.w} apply",
                )
            ),
            out,
        )
        none = ("code-back", "rk doctor", "rk drain", "up -d", r" exec ")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertNotEqual(self.called(LAYER_RUN, start=start), [-1])
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"), was.env
        )
        self.assertEqual(records.read(self.pkg).put_back.reason, "rollback")

    def test_rollback_on_a_stopped_container_puts_back_the_code_the_last_apply_put_in(
        self,
    ):
        """On a stopped container rollback puts back the code too when the last apply put a release in: code-back
        once the settings are back, through a one-off container of the image they name, its line said after `code:`;
        no doctor, drain or restart. RESTORED, the container stopped, and no release that the last apply put in kept
        in the records (records.Installed)."""
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIsNotNone(records.read(self.pkg).installed)
        start = len(self.calls())
        stopped = dict(self.running(), FAKE_DOCKER_RUNNING="0")
        rc, out, err = self.rkmon("rollback", **stopped)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out.splitlines(),
            [
                "== rollback: go back to the code and settings before the last apply",
                "  put back: the settings from before your last apply, docker/.env and the override",
                "  code: RESTORED: the image's code runs again",
                "RESTORED: the code and settings from before your last apply are in place; the container stays "
                "stopped",
                *rewrapped(f"next: {self.w} apply").splitlines(),
            ],
        )
        put, back = self.called(LAYER_RUN, r" run --rm .*code-back$", start=start)
        self.assertTrue(-1 < put < back, self.calls()[start:])
        none = ("rk doctor", "rk drain", "up -d", r" exec ")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertIsNone(records.read(self.pkg).installed)

    def rollback_after_a_hand_edit(self, file_changed: bool):
        """rollback after an apply, docker/.env edited by hand to name another compose project and image (state 9),
        and with FILE_CHANGED your file changed after the confirm too (state 5): RESTORED, the kept layer's read and the
        put-back in this deployment's compose project and on its image, never in those docker/.env names, and
        every docker call with the confirmed project in its environment. The stand-in docker answers the read and the
        put-back only in them, and as docker does for an image that is not here otherwise.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        dotenv = self.pkg / deliver.ENV_FILE
        project = f"\nCOMPOSE_PROJECT_NAME={deploy_folder.NAME}\n"
        edited = dotenv.read_text(encoding="utf-8").replace(
            project, "\nCOMPOSE_PROJECT_NAME=otherdeploy\n"
        )
        edited = edited.replace("\nRKMON_TAG=rkmon-test\n", "\nRKMON_TAG=other-tag\n")
        dotenv.write_text(edited, encoding="utf-8")
        found = self.dotenv()
        self.assertEqual(
            (found["COMPOSE_PROJECT_NAME"], found["RKMON_TAG"]),
            ("otherdeploy", "other-tag"),
        )
        if file_changed:
            self.write_settings(memory_gib=14)
        ours = {"COMPOSE_PROJECT_NAME": deploy_folder.NAME, "RKMON_TAG": "rkmon-test"}
        one_offs = (LAYER_RUN, LAYER_KEPT_READ)
        rows = [[*r[:3], ours, *r[4:]] if r[0] in one_offs else r for r in rows]
        no_image = "Error response from daemon: No such image: rocke-monitor:other-tag"
        self.answer(rows + [[p, no_image, 1] for p in one_offs])
        start = len(self.calls())
        env_log = self.d / "env.log"
        rc, out, err = self.rkmon(
            "rollback",
            **self.running(),
            FAKE_DOCKER_ENV="COMPOSE_PROJECT_NAME",
            FAKE_DOCKER_ENV_LOG=str(env_log),
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\nRESTORED: the deployment runs the code and settings from before your last apply\n",
            out,
        )
        calls = self.calls()[start:]
        self.assertEqual([c for c in calls if "other" in c], [])
        put_back = [c for c in calls if re.search(LAYER_RUN, c)]
        self.assertEqual(len(put_back), 1, calls)
        self.assertIn(f"\tcompose -p {deploy_folder.NAME} run ", put_back[0])
        self.assertNotEqual(
            self.called(LAYER_KEPT_READ, LAYER_READ, start=start), [-1, -1]
        )
        projects = set(env_log.read_text(encoding="utf-8").splitlines())
        self.assertEqual(projects, {f"COMPOSE_PROJECT_NAME={deploy_folder.NAME}"})
        self.assertEqual(dotenv.read_text(encoding="utf-8"), was.env)
        self.assertEqual(records.read(self.pkg).put_back.reason, "rollback")

    def test_rollback_puts_back_in_this_deployment_s_project_and_image(self):
        self.rollback_after_a_hand_edit(file_changed=False)

    def test_rollback_after_your_file_changed_takes_the_confirm_s_project_and_image(
        self,
    ):
        self.rollback_after_a_hand_edit(file_changed=True)

    def test_rollback_refuses_while_the_confirmed_image_is_not_here(self):
        """rollback puts back through containers of the confirmed image: while that image is not on this host
        (state 6), REFUSED with state 6's fix and next line; nothing run and nothing changed.
        """
        rows, made, was = self.change_waits()
        self.answer(rows)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.write_settings(memory_gib=13, image_tag="other-tag")
        self.confirmed()
        self.answer([[r"image inspect .*:other-tag", "", 1], *rows])
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        start = len(self.calls())
        rc, out, err = self.rkmon("rollback", **self.running())
        self.assertEqual(rc, 3, out + err)
        self.assertIn(
            "REFUSED: rollback puts back through containers of the image your confirmed settings name, and the image "
            "rocke-monitor:other-tag is not on this host; nothing changed next: ",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(
                rewrapped(f"\nnext: step 4, image: {self.w} image load IMAGE_FILE\n")
            ),
            out,
        )
        none = (LAYER_RUN, LAYER_READ, LAYER_KEPT_READ, "rk doctor", "rk drain")
        self.assertEqual(self.called(*none, start=start), [-1] * len(none))
        self.assertEqual(dotenv.read_text(encoding="utf-8"), ran)
        self.assertIsNone(records.read(self.pkg).under_way)


if __name__ == "__main__":
    unittest.main()
