# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deploy step 5, the logins, end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. The login walk shows each login and secret as a check row, what needs each one
read from your confirmed settings, and runs its commands in one-off containers of the image you confirmed, handed your
confirmed settings and nothing else; state 7 reads the same list. A token, a key and a secret go in on stdin and are
never printed; an ssh key's host keys are shown, and recorded once accepted; a value made here is shown once, on a
terminal only. rkmon web-host install runs as the writer in the walk's one-off container, on a terminal of its own that
rkmon relays.
"""

import importlib.machinery
import importlib.util
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import time
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
    GOOD,
    NEXT_LINE,
    PUBLISHED,
    RESTART,
    DeployPathCase,
)
from rendered import finished, one_line  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    container_layout,
    deliver,
    file_types,
    secrets_registry,
    settings_file,
)

# What a feature needs set while it is on, beside its switch (the confirm fails each one blank), as PUBLISHED is
# the team dashboard's: the remote GPU tests' cluster and nodes, the develop guard's labels (once the merge queue
# is live), and ticket creation's JIRA project, on the JIRA site the atlassian CLI logs in to (JIRA_SITE, under
# integrations).
GPU_TESTS = {
    "enabled": True,
    "cluster": {
        "login_host": "login.example.com",
        "partition": "gpu",
        "ssh_user": "widgets-ci",
        "accounts": ["widgets"],
        "gpu_types": {"gfx942": "mi300x"},
    },
    "node": {"python": "~/venvs/tests/bin/python"},
}
DEVELOP_GUARD = {
    "enabled": True,
    "ci_red_label": "ci-red",
    "stability_label": "stability",
}
TICKETS = {"project_key": "PROJ", "follow_up_tickets": {"enabled": True}}
JIRA_SITE = {"atlassian_cli": {"jira_url": "https://jira.example.com"}}


class TestLogins(DeployPathCase):
    def test_while_an_older_package_runs_state_7_reads_the_logins_as_step_5_does(self):
        """While the running container runs an older package's code, its rk lists that package's logins, under
        that package's names and from its own secrets: state 7 reads them as deploy step 5 does, in the walk's one-off
        of the image you confirmed, handed your confirmed settings (walk_inside). A login the older rk calls missing
        that is set for this package goes on to the first apply: rkmon next gives state 11's line, and apply makes it.
        One the older rk has no row for that is missing for this package gives state 7: rkmon next gives login's line,
        and apply refuses with it, nothing changed. Each state read asks the older rk `rk applied --json` once.
        """
        rows, _made, _older = self.older_package()
        older_logins = [
            {
                "name": "gh-read",
                "what": "gh's read login",
                "needed_by": ["the monitor"],
                "state": "SET",
                "found": "op-login reads only",
                "command": "login gh-read",
            },
            {
                "name": "claude",
                "what": "the Claude gateway (secret claude_env)",
                "needed_by": ["Headless Claude"],
                "state": "MISSING",
                "found": "/secrets/claude_env",
                "command": "setup --section claude",
            },
        ]
        in_container = r"compose exec .* rk logins --json$"
        one_off = r"compose run .* rk logins --json$"
        applied = r"rk applied --json$"
        w = self.w
        # state 7 first: its apply changes nothing, so state 11's runs on the same deployment
        for unset, n in (({"admin_access_token": ("MISSING", "not set")}, 7), ({}, 11)):
            with self.subTest(state=n):
                self.answer(
                    [
                        # the new container, once made again, lists this package's logins itself
                        [
                            in_container,
                            json.dumps(self.login_rows()),
                            0,
                            {},
                            {"after": f"(?m){RESTART}"},
                        ],
                        [in_container, json.dumps(older_logins), 0],
                        [one_off, json.dumps(self.login_rows(**unset)), 0],
                        *rows,
                    ]
                )
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon("next", **self.running())
                self.assertEqual(rc, 0, out + err)
                line = f"next: {w} apply" if n == 11 else f"next: {w} login"
                self.assertTrue(out.endswith("\n" + finished(line)), out)
                self.assertEqual(self.called(in_container), [-1], self.calls())
                self.assertNotEqual(self.called(one_off), [-1], self.calls())
                asked = [c for c in self.calls() if re.search(applied, c)]
                self.assertEqual(len(asked), 1, self.calls())
                self.log.unlink()
                rc, out, err = self.rkmon("apply", **self.running())
                restart = self.called(RESTART)[0]
                if n == 11:
                    self.assertEqual(rc, 0, out + err)
                    self.assertIn("\nAPPLIED: ", out)
                    self.assertNotEqual(restart, -1, self.calls())
                    before = self.calls()[:restart]
                    asked = [c for c in before if re.search(applied, c)]
                    self.assertEqual(len(asked), 1, self.calls())
                else:
                    self.assertEqual(rc, 3, out + err)
                    self.assertIn(
                        "\nREFUSED: your confirmed settings need admin_access_token, which is missing or wrong; "
                        "nothing changed\n",
                        out,
                    )
                    self.assertEqual(restart, -1, self.calls())
                    self.assertEqual(self.called("rk stop "), [-1], self.calls())

    def test_a_container_of_this_package_lists_its_own_logins_for_state_7(self):
        """A running container of this package's code (its rk answers `rk applied --json`) lists its logins itself:
        state 7 reads `rk logins --json` in it, and no one-off container runs for them. Its rk is asked `rk applied
        --json` once."""
        self.confirmed()
        missing = self.login_rows(admin_access_token=("MISSING", "not set"))
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"compose exec .* rk logins --json$", json.dumps(missing), 0],
                [r"compose run .* rk logins --json$", "[]", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "your confirmed settings need admin_access_token, which is missing or wrong",
            one_line(out),
        )
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 5, login: {self.w} login")), out
        )
        self.assertNotEqual(
            self.called(r"compose exec .* rk logins --json$"), [-1], self.calls()
        )
        self.assertEqual(
            self.called(r"compose run .* rk logins --json$"), [-1], self.calls()
        )
        asked = [c for c in self.calls() if re.search(r"rk applied --json$", c)]
        self.assertEqual(len(asked), 1, self.calls())

    def test_the_walk_ends_with_the_doctor_s_rows_on_the_logins(self):
        """`rkmon login --no-input` ends with rk doctor's WARN and FAIL rows on the logins, each in full as
        checks.render prints it (a found text over two lines, and the fix), and with no other row, once the first start
        has made the container and while it runs your confirmed settings; before it, with none, and the doctor does
        not run; before you apply them (none delivered here), one line in their place says
        that the doctor's findings on them come with rkmon apply, as the container still runs your earlier ones, and
        the doctor does not run either.
        """
        fix = checks.Command(("rk", "logins"))
        on_logins = [
            checks.Check(
                "GitHub write login", "FAIL", "no login in the write volume", fix
            ),
            checks.Check(
                "Teams webhook",
                "WARN",
                "the alerts' webhook file is missing;" + " the alerts wait for it" * 6,
                fix,
            ),
        ]
        others = [
            checks.Check("ramdisk", "FAIL", "not mounted", fix),
            checks.Check("GitHub read login", "PASS", "it reads acme/widgets"),
        ]
        report = checks.render([others[0], on_logins[0], others[1], on_logins[1]])
        self.assertGreater(len(checks.render(on_logins[1:]).splitlines()), 2)
        self.confirmed()
        logins = [r"rk logins --json", json.dumps(self.login_rows()), 0]
        self.answer([[r"image inspect", GOOD, 0], logins, [r"rk doctor$", report, 4]])
        rc, out, err = self.rkmon("login", "--no-input")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("The doctor, on the logins:", out)
        self.assertFalse([c for c in self.calls() if "rk doctor" in c])
        self.env.update(
            FAKE_DOCKER_PROJECT=deploy_folder.NAME,
            FAKE_DOCKER_WORKDIR=os.path.realpath(self.pkg / "docker"),
        )
        rc, out, err = self.rkmon("login", "--no-input")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"\nThe doctor's findings on these settings come with {self.w} apply: the container still runs your "
            "earlier ones.\n",
            out,
        )
        self.assertNotIn("The doctor, on the logins:", out)
        self.assertFalse([c for c in self.calls() if "rk doctor" in c])
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                logins,
                [r"rk doctor$", report, 4],
            ]
        )
        rc, out, err = self.rkmon("login", "--no-input")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("The doctor's findings on these settings", out)
        shown = out.split("The doctor, on the logins:\n", 1)[-1]
        for row in on_logins:
            self.assertIn(
                "\n".join("  " + x for x in checks.render([row]).splitlines()), shown
            )
        self.assertNotIn("ramdisk", shown)
        self.assertNotIn("GitHub read login", shown)
        self.assertNotIn("no login check fails", shown)

    def test_the_walk_s_one_offs_are_handed_your_confirmed_settings_and_nothing_else_is(
        self,
    ):
        """Deploy step 5 before the first start, with no container and no layer delivered: each command of the
        walk (rk logins, rk secret-set) runs in a one-off container handed the layer your confirmed settings make, the
        text the first start delivers: the variable's name alone on docker's command line, its value in docker's
        environment and in no output. The doctor waits for the first start. Neither the start nor any of its one-off
        containers is handed it; once the container runs, the walk's commands still run in a one-off container handed
        it, and the doctor runs in the container without it."""
        name = container_layout.HANDED_LAYER_VARIABLE
        self.confirmed()
        layer = deliver.texts(deploy_folder.plan(self.pkg, self.settings))[2]
        rows = [
            {
                "name": "admin_access_token",
                "what": "the admin dashboard's access token",
                "needed_by": ["the admin dashboard"],
                "state": "MISSING",
                "found": "",
                "command": "secret admin_access_token --generate",
            }
        ]
        seen = self.d / "env.log"
        watch = {"FAKE_DOCKER_ENV": name, "FAKE_DOCKER_ENV_LOG": str(seen)}
        one_off = rf"^compose run --rm -T -e {name} -e RKMON_WRAPPER=\S+ rkmon rk "
        walk = [
            [one_off + "logins --json$", json.dumps(rows), 0, {name: layer}],
            [one_off + "secret-set admin_access_token$", "", 0, {name: layer}],
            [r"rk (logins|secret-set)", "not handed your confirmed settings", 9],
        ]

        def handed(calls: list, said: str) -> list:
            """The calls handed the layer, each a walk's one-off container, the layer in no output nor call."""
            ones = [c for c in calls if f" {name} " in c]
            for c in ones:
                self.assertRegex(c, one_off)
            self.assertNotIn(layer, said)
            self.assertNotIn(json.loads(layer)["fingerprint"], said + "\n".join(calls))
            self.assertEqual(
                seen.read_text(encoding="utf-8").count(f"{name}="), len(ones)
            )
            return ones

        self.answer([[r"image inspect", GOOD, 0], *walk])
        rc, out, err = self.rkmon("login", **watch)
        self.assertEqual(rc, 4, out + err)
        self.assertIn("admin_access_token", out)
        self.assertNotIn("The doctor, on the logins:", out)
        self.answer(
            walk
        )  # the stand-in docker reads the stdin of a call it answers: secret-set's alone
        rc, out2, err2 = self.rkmon(
            "secret", "admin_access_token", "-", stdin="Invented-Token-Q\n", **watch
        )
        self.assertEqual(rc, 0, out2 + err2)
        self.assertIn("Invented-Token-Q", self.stdin.read_text(encoding="utf-8"))
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        ones = handed(calls, out + err + out2 + err2)
        self.assertEqual([c for c in calls if " rk " in c], ones)
        self.assertEqual(len(ones), 2)
        self.assertFalse([c for c in calls if "rk doctor" in c])
        # the first start: none handed
        seen.unlink()
        self.log.unlink()
        stopped = self.d / "stopped"
        stopped.touch()
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon(
            "start", FAKE_DOCKER_RUNNING="1", FAKE_DOCKER_STOPPED=str(stopped), **watch
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn("STARTED: the monitor runs your confirmed settings", out)
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertIn("compose up -d", calls)
        self.assertEqual(handed(calls, out + err), [])
        # the container runs: the walk's one-off containers handed it, the doctor in the container without it
        seen.unlink()
        self.log.unlink()
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"rk stopped$", "", 1],
                *walk,
                [r"rk doctor$", "", 0],
            ]
        )
        rc, out, err = self.rkmon("login", **self.running(), **watch)
        self.assertEqual(rc, 4, out + err)
        self.assertIn("The doctor, on the logins:", out)
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertEqual(len(handed(calls, out + err)), 1)
        (doctor,) = [c for c in calls if c.endswith(" rk doctor")]
        self.assertTrue(doctor.startswith("compose exec "), doctor)

    def test_the_walk_runs_the_image_you_confirmed_beside_a_container_of_another(self):
        """A first apply's order: confirm, image, login, then apply. docker/.env names the image the running container
        runs, an older package's, and your confirm names a new one: each one-off container of the walk runs the one
        you confirmed, its RKMON_IMAGE and RKMON_TAG in docker compose's environment over docker/.env's, which stays
        as it is. Until you apply them, the doctor, which checks the settings that run, does not run, and a line says
        that its findings on yours come with rkmon apply. While the image you confirmed is not on this host, the walk,
        a login's key and a secret are refused before any one-off container, and the next line names the image's load.
        """
        name = container_layout.HANDED_LAYER_VARIABLE
        self.write_settings(image_tag="older")
        self.confirmed()
        running = (
            self.delivered()
        )  # what the running container runs: docker/.env names the older image
        self.write_settings()
        deploy_folder.confirm(self.pkg, self.settings, self.home)
        made = deploy_folder.plan(self.pkg, self.settings)
        self.assertEqual(self.dotenv()["RKMON_TAG"], "older")
        image = {
            "RKMON_IMAGE": made.variables["RKMON_IMAGE"],
            "RKMON_TAG": "rkmon-test",
        }
        handed = dict(image, **{name: deliver.texts(made)[2]})
        rows = [
            {
                "name": "github_read",
                "what": "the read login",
                "needed_by": ["review automation"],
                "state": "SET",
                "found": "reads acme/widgets",
                "command": "login github_read",
            }
        ]
        one_off = rf"^compose run --rm -T -e {name} -e RKMON_WRAPPER=\S+ rkmon rk "
        walk = [
            [one_off + "logins --json$", json.dumps(rows), 0, handed],
            [one_off + "secret-set admin_access_token$", "", 0, handed],
            [r"rk (logins|secret-set)", "not on the image you confirmed", 9],
        ]
        self.answer(
            [
                *running,
                [r"image inspect", GOOD, 0],
                [r"rk stopped$", "", 1],
                *walk,
                [r"rk doctor$", "", 0],
            ]
        )
        rc, out, err = self.rkmon("login", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("PASSED: every login and secret your settings need is set", out)
        self.assertIn(
            f"\nThe doctor's findings on these settings come with {self.w} apply: the container still runs your "
            "earlier ones.\n",
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk doctor" in c])
        # the stand-in docker reads the stdin of a call it answers: the state's guard asks none of these
        self.answer([*running, *walk])
        rc, out, err = self.rkmon(
            "secret",
            "admin_access_token",
            "-",
            stdin="Invented-Token-W\n",
            **self.running(),
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn("SET: admin_access_token", out)
        self.assertEqual(self.dotenv()["RKMON_TAG"], "older")
        # the image you confirmed not here: refused before any one-off container, the image's load next
        for args, typed in (
            (("login",), None),
            (("login", "claude"), "Invented-Key-W\n"),
            (("secret", "admin_access_token", "-"), "Invented-Token-W\n"),
        ):
            with self.subTest(args=args):
                self.answer(
                    [
                        *running,
                        [r"image inspect .*:rkmon-test$", "", 1],
                        [r"image inspect", GOOD, 0],
                    ]
                )
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon(*args, stdin=typed, **self.running())
                self.assertEqual((rc, err), (3, ""), out)
                self.assertTrue(
                    out.endswith(
                        finished(
                            "REFUSED: the image rocke-monitor:rkmon-test is not on this host; nothing changed",
                            f"next: step 4, image: {self.w} image load IMAGE_FILE",
                        )
                    ),
                    out,
                )
                self.assertFalse([c for c in self.calls() if "compose run" in c])

    def test_a_layer_longer_than_one_variable_carries_is_refused_before_any_one_off(
        self,
    ):
        """The walk hands its one-off containers your confirmed settings by one variable, which carries
        container_layout.HANDED_LAYER_MAX bytes at most: settings that make a longer layer are refused in plain words,
        with the fix, before any one-off container, by the walk, a login's key and a secret alike.
        """
        values = settings_file.read(self.settings, file_types.DEPLOY).values
        values["repository"]["scope_labels"] = [
            f"label-{i:05d}-" + "x" * 48 for i in range(2500)
        ]
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, values
        )
        self.settings.write_text(settings_file.dump(doc), encoding="utf-8")
        self.confirmed()
        layer = deliver.texts(deploy_folder.plan(self.pkg, self.settings))[2]
        size = len(layer.encode("utf-8"))
        self.assertGreater(size, container_layout.HANDED_LAYER_MAX)
        for args, typed in (
            (("login",), None),
            (("login", "claude"), "Invented-Key-Z\n"),
            (("secret", "admin_access_token", "-"), "Invented-Token-Z\n"),
        ):
            with self.subTest(args=args):
                # the stand-in docker reads the stdin of a call it answers: none, once a value is typed
                self.answer([] if typed else [[r"image inspect", GOOD, 0]])
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon(*args, stdin=typed)
                self.assertEqual((rc, err), (4, ""), out)
                said = one_line(out)
                self.assertIn(
                    f"REFUSED: your confirmed settings make a layer of {size} bytes, more than the "
                    f"{container_layout.HANDED_LAYER_MAX} that one variable hands a one-off container; nothing "
                    f"changed fix: shorten ~/{self.settings.name} (its longest lists), then "
                    f"{self.w} confirm next: {self.w} {' '.join(args)}",
                    said,
                )
                self.assertFalse([c for c in self.calls() if "compose run" in c])

    def test_the_login_walk_shows_check_rows_and_ends_with_one_outcome(self):
        """Deploy step 5: each login and secret as a check row, the command that sets it as its fix. What needs
        each one comes from your confirmed settings, read here on the host, and its state from rk logins: SET is a
        PASS, MISSING a FAIL, one it could not check a WARN; one that only a feature in prototype would need is OFF,
        its row naming the mode, but the atlassian login is a WARN while ticket creation is in prototype, with the
        command that sets it. With --no-input it asks nothing, on a terminal too. On a terminal it offers each
        FAIL row's command in turn, and shows that row again after it. PASSED once none fails, then step 6; else
        FAILED, exit 4, then the command again; CANCELLED, exit 1, when you said no to each one left, then the
        deployment state's next line, here state 7's. `rkmon login` runs in the deployment states once the
        image is here: without it, the state's refusal (state 6). `rkmon logins` is no command: its error names
        the walk, and that of `logins --check` names --no-input, which goes with no NAME.
        """
        w = self.w
        # Review automation live, which posts with the write login; the alerts and ticket creation in prototype.
        self.write_settings(
            features={
                "review_automation": {"mode": "live"},
                "alerts": {"mode": "prototype", "develop_breaks": {"enabled": True}},
                "ticket_creation": {"mode": "prototype", **TICKETS},
            },
            integrations=JIRA_SITE,
        )
        self.confirmed()
        # This folder's container exists (stopped), with your confirmed settings, which the doctor checks.
        self.env.update(
            FAKE_DOCKER_PROJECT=deploy_folder.NAME,
            FAKE_DOCKER_WORKDIR=os.path.realpath(self.pkg / "docker"),
        )
        delivered = self.delivered()
        what = {e.name: e.what for e in secrets_registry.walk_entries({})}
        prototype = "features.{}.mode is prototype: nothing goes out until it is live"
        write_fails = checks.Check(
            "github_write",
            "FAIL",
            f"not set: {what['github_write']}, needed by review posting",
            checks.Command((w, "login", "github_write")),
        )
        shown = checks.render(
            [
                checks.Check("github_read", "PASS", "reads acme/widgets"),
                write_fails,
                checks.Check(
                    "atlassian",
                    "WARN",
                    f"not set: {what['atlassian']}; {prototype.format('ticket_creation')}, so tickets are prepared "
                    f"without it, and with it {w} check reads your JIRA project's required fields early",
                    checks.Command((w, "login", "atlassian")),
                ),
                checks.Check("claude", "PASS", "set"),
                checks.Check(
                    "admin_access_token",
                    "WARN",
                    f"not checked: {what['admin_access_token']}, needed by admin dashboard",
                    checks.Command((w, "secret", "admin_access_token", "--generate")),
                ),
                checks.Check(
                    "teams_webhook",
                    "OFF",
                    f"{what['teams_webhook']}: nothing that is on needs it yet ({prototype.format('alerts')})",
                ),
                checks.Check(
                    "web_host_ssh_key",
                    "OFF",
                    f"{what['web_host_ssh_key']}: nothing that is on needs it",
                ),
                checks.Check(
                    "gpu_ssh_key",
                    "OFF",
                    f"{what['gpu_ssh_key']}: nothing that is on needs it",
                ),
            ]
        )
        read = ("SET", "reads acme/widgets")

        def answer(rows):
            self.answer(
                [
                    *delivered,
                    [r"image inspect", GOOD, 0],
                    [r"rk logins --json", json.dumps(rows), 0],
                    [r"rk doctor", "", 0],
                ]
            )

        answer(
            self.login_rows(
                github_read=read, github_write="MISSING", admin_access_token="UNKNOWN"
            )
        )
        left = (
            "of the logins and secrets your settings need, 1 of 4 is missing or wrong"
        )
        rc, out, err = self.rkmon("login", "--no-input")
        self.assertEqual((rc, err), (4, ""), out)
        self.assertIn(shown + "\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    f"FAILED: {left} (fix each FAIL above)",
                    f"next: {w} login --no-input",
                )
            ),
            out,
        )
        rc, out = sandbox.on_a_terminal(
            [w, "login", "--no-input"], self.home, self.env, typed=b"y\n"
        )
        self.assertEqual(rc, 4, out)
        self.assertNotIn("it now?", out)
        # On a terminal: a no to the one FAIL row is CANCELLED, and nothing runs.
        rc, out = sandbox.on_a_terminal([w, "login"], self.home, self.env, typed=b"n\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("github_write: set it now? [Y/n] ", out)
        self.assertTrue(
            out.endswith(
                finished(
                    f"CANCELLED: you said no; {left}", f"next: step 5, login: {w} login"
                ).replace("\n", "\r\n")
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "auth login" in c])
        # A yes runs the row's command, then shows its row again in place of that command's outcome: here it is
        # still not set, so FAILED.
        rc, out = sandbox.on_a_terminal([w, "login"], self.home, self.env, typed=b"y\n")
        self.assertEqual(rc, 4, out)
        self.assertEqual(
            len([c for c in self.calls() if "gh-write auth login" in c]), 1
        )
        self.assertNotIn("SET: github_write", out)
        again = checks.render([write_fails])
        self.assertIn(
            again.replace("\n", "\r\n") + "\r\n\r\nThe doctor, on the logins:", out
        )
        self.assertTrue(
            out.endswith(
                finished(
                    f"FAILED: {left} (fix each FAIL above)", f"next: {w} login"
                ).replace("\n", "\r\n")
            ),
            out,
        )
        # None fails: PASSED, with the warnings counted, then step 6.
        answer(self.login_rows(github_read=read, admin_access_token="UNKNOWN"))
        rc, out, err = self.rkmon("login")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: every login and secret your settings need is set (2 warnings)",
                    f"next: step 6, start: {w} start",
                )
            ),
            out,
        )
        # No image here: the state's refusal, an earlier step, and nothing asked of the container; docker is
        # asked only whether it runs, which the next line's step words read.
        self.answer([[r"image inspect", "", 1]])
        self.log.unlink()
        rc, out, err = self.rkmon("login")
        self.assertEqual((rc, err), (3, ""), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the image rocke-monitor:rkmon-test is not on this host; nothing changed",
                    f"next: step 4, image: {w} image load IMAGE_FILE",
                )
            ),
            out,
        )
        asked = [c.split("\t")[1] for c in self.calls()]
        self.assertEqual(
            [c for c in asked if c.startswith("compose")],
            ["compose ps --status running --quiet rkmon"],
        )
        for args, said in (
            (("logins",), f'{w}: unknown command "logins": {w} login replaces it'),
            (
                ("logins", "--check"),
                f'{w} logins: unknown option "--check": {w} login --no-input replaces it',
            ),
            (
                ("login", "claude", "--no-input"),
                f"{w} login: --no-input goes with the walk over each login ({w} login), with no NAME",
            ),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.rkmon(*args), (2, "", said + "\n"))

    def test_the_login_walk_s_claude_row_asks_for_the_key_without_echo(self):
        """Deploy step 5's claude row: while the claude login is missing, a FAIL row whose fix is `rkmon login
        claude`; on a terminal a yes asks for the key without echo, rk secret-set claude gets it on its stdin, as the
        monitor, then rk claude-test makes the login's test call and shows its row, which the walk does not show
        again. A test call that fails leaves claude's row among those left, though the key is set: FAILED, exit 4. The
        key is in no output nor docker's log.
        """
        w, key = self.w, "test-claude-key-" + "Wk4"
        self.confirmed()
        self.env.update(
            FAKE_DOCKER_PROJECT=deploy_folder.NAME,
            FAKE_DOCKER_WORKDIR=os.path.realpath(self.pkg / "docker"),
        )
        delivered = self.delivered()  # your confirmed settings, which the doctor checks
        what = secrets_registry.entry({}, "claude").what
        read = ("SET", "reads acme/widgets")
        unset = json.dumps(self.login_rows(github_read=read, claude="MISSING"))
        set_ = json.dumps(self.login_rows(github_read=read))
        # rk secret-set claude has run: rk logins lists the key as set
        made = self.d / "claude-set"
        login = checks.Command((w, "login", "claude"))
        claude = checks.Check(
            "claude", "FAIL", f"not set: {what}, needed by Claude session", login
        )
        where = "at Anthropic's own API"
        answered = checks.Check(
            "claude", "PASS", f"one test call answered, {where} (model claude-a1)"
        )
        refused = checks.Check(
            "claude", "FAIL", f"the key is refused, {where} (HTTP 401)", login
        )

        def walk(test_row, code):
            """The walk on a terminal, a yes to claude's row and its key typed; rk claude-test shows TEST_ROW and
            exits CODE."""
            for f in (made, self.log, self.stdin):
                f.unlink(missing_ok=True)
            self.answer(
                [
                    *delivered,
                    [r"image inspect", GOOD, 0],
                    [r"rk logins --json", unset, 0, {}, {"unless": str(made)}],
                    [r"rk logins --json", set_, 0],
                    [r"rk secret-set claude$", "", 0, {}, {"makes": str(made)}],
                    [r"rk claude-test$", checks.render([test_row]), code],
                    [r"rk doctor", "", 0],
                ]
            )
            return sandbox.on_a_terminal(
                [w, "login"],
                self.home,
                self.env,
                replies=[
                    (b"claude: set it now? [Y/n] ", b"y\n"),
                    (
                        b"The Claude session's key: an Anthropic API key, or your gateway's token (not shown): ",
                        key.encode() + b"\n",
                    ),
                ],
            )

        rc, out = walk(answered, 0)
        self.assertEqual(rc, 0, out)
        for row in (checks.Check("github_read", "PASS", "reads acme/widgets"), claude):
            self.assertIn(checks.render([row]).replace("\n", "\r\n"), out)
        tested = checks.render([answered]).replace("\n", "\r\n")
        self.assertIn(tested + "\r\n\r\nThe doctor, on the logins:", out)
        self.assertEqual(
            (out.count("  FAIL  claude: "), out.count("  PASS  claude: ")), (1, 1), out
        )
        (set_call,) = [c for c in self.calls() if "secret-set" in c]
        self.assertTrue(set_call.endswith(" rk secret-set claude"), set_call)
        self.assertNotIn("rkwrite", set_call)  # the monitor's own login
        self.assertEqual(self.stdin.read_text().strip(), key)
        self.assertNotIn(key, out + "\n".join(self.calls()))
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: every login and secret your settings need is set",
                    f"next: step 6, start: {w} start",
                ).replace("\n", "\r\n")
            ),
            out,
        )
        # The test call fails: its row, and claude's left, though rk logins lists the key as set now.
        rc, out = walk(refused, 4)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            checks.render([refused]).replace("\n", "\r\n")
            + "\r\n\r\nThe doctor, on the logins:",
            out,
        )
        self.assertNotIn("  PASS  claude: ", out)
        self.assertNotIn(key, out + "\n".join(self.calls()))
        self.assertTrue(
            out.endswith(
                finished(
                    "FAILED: of the logins and secrets your settings need, 1 of 3 is missing or wrong (fix each "
                    "FAIL above)",
                    f"next: {w} login",
                ).replace("\n", "\r\n")
            ),
            out,
        )

    def test_the_walk_with_each_login_missing_in_prototype_and_live(self):
        """Deploy step 5 with nothing set and each feature on. In prototype the walk needs no login that
        sends: github_write is OFF while review automation and the merge queue are both in prototype, and teams_webhook
        and web_host_ssh_key while the alerts and the team dashboard are, each row naming the modes; atlassian is a WARN
        while ticket creation is in prototype, with the command that sets it; gpu_ssh_key is needed whenever remote GPU
        tests are on. Going live makes each one needed, whatever the image reads: here rk logins lists them as it read
        them in prototype, and the walk fails each, with the command that sets it. FAILED, exit 4, then the command
        again."""
        w = self.w

        def features(mode: str) -> dict:
            return {
                "review_automation": {"mode": mode, "remote_gpu_tests": GPU_TESTS},
                "merge_queue": {"mode": mode, "develop_guard": DEVELOP_GUARD},
                "dashboards": {"team": {**PUBLISHED, "mode": mode}},
                "alerts": {"mode": mode, "develop_breaks": {"enabled": True}},
                "ticket_creation": {"mode": mode, **TICKETS},
            }

        entries = secrets_registry.walk_entries({})
        what = {e.name: e.what for e in entries}
        fix = {
            e.name: checks.Command((w, *secrets_registry.set_command(e)))
            for e in entries
        }
        prototype = "features.{}.mode is prototype: nothing goes out until it is live"

        def missing(name: str, needs: str):
            text = f"not set: {what[name]}, needed by {needs}"
            return checks.Check(name, "FAIL", text, fix[name])

        def off(name: str, *modes: str):
            waits = "; ".join(prototype.format(m) for m in modes)
            text = f"{what[name]}: nothing that is on needs it yet ({waits})"
            return checks.Check(name, "OFF", text)

        self.write_settings(features=features("prototype"), integrations=JIRA_SITE)
        self.confirmed()
        # as the image lists them: each one the settings need MISSING, each other NOT NEEDED
        unset = self.login_rows("MISSING")
        self.answer(
            [[r"image inspect", GOOD, 0], [r"rk logins --json", json.dumps(unset), 0]]
        )
        rc, out, err = self.rkmon("login")
        self.assertEqual((rc, err), (4, ""), out)
        shown = checks.render(
            [
                missing("github_read", "the monitor, always"),
                off("github_write", "review_automation", "merge_queue"),
                checks.Check(
                    "atlassian",
                    "WARN",
                    f"not set: {what['atlassian']}; {prototype.format('ticket_creation')}, so tickets are prepared "
                    f"without it, and with it {w} check reads your JIRA project's required fields early",
                    fix["atlassian"],
                ),
                missing("claude", "Claude session"),
                missing("admin_access_token", "admin dashboard"),
                off("teams_webhook", "alerts"),
                off("web_host_ssh_key", "dashboards.team"),
                missing("gpu_ssh_key", "remote GPU tests"),
            ]
        )
        self.assertIn(shown + "\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "FAILED: of the logins and secrets your settings need, 4 of 4 are missing or wrong (fix each "
                    "FAIL above)",
                    f"next: {w} login",
                )
            ),
            out,
        )
        # Live: each one needed, though rk logins still lists them as it read them in prototype.
        self.write_settings(features=features("live"), integrations=JIRA_SITE)
        self.confirmed()
        rc, out, err = self.rkmon("login")
        self.assertEqual((rc, err), (4, ""), out)
        rows = re.findall(r"(?m)^  (PASS|FAIL|WARN|OFF) +(\w+): ", out)
        self.assertEqual(rows, [("FAIL", name) for name in what])
        said = one_line(out)
        for name in what:
            with self.subTest(name=name):
                self.assertIn(f"{name}: not set: {what[name]}, needed by ", said)
                self.assertIn(f"fix: {fix[name]}", said)
        self.assertIn(
            f"github_write: not set: {what['github_write']}, needed by review posting, ",
            said,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "FAILED: of the logins and secrets your settings need, 8 of 8 are missing or wrong (fix each "
                    "FAIL above)",
                    f"next: {w} login",
                )
            ),
            out,
        )

    def test_state_7_and_the_walk_read_one_list(self):
        """State 7 reads the login walk's list, which deploy step 5 shows: what needs each login comes from your
        confirmed settings, read here on the host. Going live needs a login that the running container, which still
        runs your settings in prototype, reads as not needed: state 7 sends you to rkmon login before rkmon apply, and
        the walk fails its row."""
        w = self.w
        self.confirmed()
        running = self.delivered()  # the container runs your settings in prototype
        # as it lists them: github_write NOT NEEDED
        rows = json.dumps(self.login_rows())
        self.write_settings(features={"review_automation": {"mode": "live"}})
        self.confirmed()
        self.answer(
            [
                *running,
                [r"image inspect", GOOD, 0],
                [r"rk stopped$", "", 1],
                [r"rk logins --json", rows, 0],
                [r"rk doctor$", "", 0],
            ]
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "your confirmed settings need github_write, which is missing or wrong",
            one_line(out),
        )
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 5, login: {w} login")), out
        )
        rc, out, err = self.rkmon("login", "--no-input", **self.running())
        self.assertEqual((rc, err), (4, ""), out)
        what = secrets_registry.entry({}, "github_write").what
        fails = checks.Check(
            "github_write",
            "FAIL",
            f"not set: {what}, needed by review posting",
            checks.Command((w, "login", "github_write")),
        )
        self.assertIn(checks.render([fails]) + "\n", out)
        self.assertIn(
            f"\nThe doctor's findings on these settings come with {w} apply: the container still runs your "
            "earlier ones.\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "FAILED: of the logins and secrets your settings need, 1 of 4 is missing or wrong (fix each "
                    "FAIL above)",
                    f"next: {w} login --no-input",
                )
            ),
            out,
        )

    def test_a_token_and_a_secret_go_in_on_stdin_never_printed(self):
        token = "test-read-token-value"
        self.confirmed()
        self.answer(
            [[r"read-login", "", 0], [r"auth login", "", 0], [r"secret-set", "", 0]]
        )
        rc, out, err = self.rkmon(
            "login", "github_read", "--with-token", stdin=token + "\n"
        )
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn(token, out + err + "\n".join(self.calls()))
        (ran,) = [c for c in self.calls() if "read-login" in c]
        self.assertTrue(ran.endswith(" rk read-login"), ran)
        self.assertEqual(self.stdin.read_text().strip(), token)
        # The write login: as the write user, which the entry point drops to after it maps the ids.
        self.stdin.unlink()
        rc, out, err = self.rkmon(
            "login", "github_write", "--with-token", stdin=token + "\n"
        )
        self.assertEqual(rc, 0, out + err)
        write = [c for c in self.calls() if "gh-write auth login" in c][-1]
        self.assertIn("-e RKMON_AS=rkwrite rkmon gh-write", write)
        self.assertNotIn("--user", write)
        hook = self.home / "webhook"
        hook.write_text("https://hooks.example.invalid/x\n")
        hook.chmod(0o644)
        rc, out, err = self.rkmon("secret", "teams_webhook", str(hook))
        self.assertEqual(rc, 4)
        self.assertIn(f"chmod 600 {hook}", one_line(out))
        hook.chmod(0o600)
        self.stdin.unlink()
        rc, out, err = self.rkmon("secret", "teams_webhook", str(hook))
        self.assertEqual(rc, 0, out + err)
        # the writer's: as the writer itself, from the host (its entry point gives the writer's rk to its broker)
        self.assertIn(
            "-e RKMON_AS=rkwrite rkmon rk secret-set teams_webhook",
            "\n".join(self.calls()),
        )
        self.assertEqual(
            self.stdin.read_text().strip(), "https://hooks.example.invalid/x"
        )
        self.assertNotIn("hooks.example.invalid", out + err)

    def test_the_read_login_takes_a_token_alone_and_refuses_one_that_can_write(self):
        """`rkmon login github_read`, with --with-token or without: a token alone, typed or on stdin (the read login
        has no browser flow, whose token could write), which rk read-login logs in, in the walk's one-off container
        handed your confirmed settings, as the monitor, whose login it is. SET says when the monitor
        starts using it. A token that can write: rk read-login's own refusal stands, exit 4, then the command again; no
        token on stdin: FAILED, and rk is not asked. The token is in no output nor docker's log.
        """
        token = "test-read-token-" + "Zq7"
        self.confirmed()
        self.answer(
            [
                [
                    r"rk read-login",
                    "github_read: set (/secrets/gh-read; reader reads only)",
                    0,
                ]
            ]
        )
        rc, out, err = self.rkmon("login", "github_read", stdin=token + "\n")
        self.assertEqual(rc, 0, out + err)
        (ran,) = [c for c in self.calls() if "read-login" in c]
        self.assertIn(
            f"\tcompose run --rm -T -e {container_layout.HANDED_LAYER_VARIABLE} ", ran
        )  # handed your confirmed settings
        self.assertTrue(ran.endswith(" rk read-login"), ran)
        self.assertNotIn("rkwrite", ran)
        self.assertEqual(self.stdin.read_text().strip(), token)
        self.assertIn("SET: github_read; the next GitHub read uses it\n", out)
        self.assertNotIn(token, out + err + "\n".join(self.calls()))
        refused = (
            "REFUSED: github_read: nothing changed:\n"
            "  FAIL  github_read: op-login can write to acme/widgets (HTTP 422 (validation): writes are allowed); "
            "the read login must not"
        )
        self.answer([[r"rk read-login", refused, 4]])
        self.stdin.unlink()
        rc, out, err = self.rkmon(
            "login", "github_read", "--with-token", stdin=token + "\n"
        )
        self.assertEqual(rc, 4, out + err)
        self.assertIn(refused, out)
        self.assertTrue(
            out.endswith(f"next: {self.w} login github_read --with-token\n"), out
        )
        self.assertNotIn("SET", out)
        self.assertNotIn(token, out + err + "\n".join(self.calls()))
        self.stdin.unlink(missing_ok=True)
        before = len(self.calls())
        rc, out, err = self.rkmon("login", "github_read", stdin="")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAILED: no token on stdin for github_read; nothing changed\n", out
        )
        self.assertEqual([c for c in self.calls()[before:] if "read-login" in c], [])

    def test_a_secret_from_stdin_goes_in_on_stdin_never_printed(self):
        """`rkmon secret NAME -`, or a value piped in: rk secret-set gets it on its stdin, never in an argument, and it
        is in no output nor docker's log."""
        value = "https://hooks.example.invalid/" + "made-up-Zq7"
        self.confirmed()
        self.answer([[r"secret-set", "", 0]])
        for args in (("secret", "teams_webhook", "-"), ("secret", "teams_webhook")):
            with self.subTest(args=args):
                self.stdin.unlink(missing_ok=True)
                rc, out, err = self.rkmon(*args, stdin=value + "\n")
                self.assertEqual(rc, 0, out + err)
                set_ = [c for c in self.calls() if "secret-set" in c][-1]
                self.assertTrue(set_.endswith(" rk secret-set teams_webhook"))
                self.assertEqual(self.stdin.read_text().strip(), value)
                self.assertNotIn(value, out + err + "\n".join(self.calls()))

    def test_an_exit_1_with_no_error_line_of_rks_ends_with_the_wrappers_error(self):
        """An exit 1 of rk in the container, rk secret-set's, rk read-login's, rk web-host-install's or rk adopt
        apply's, is its own ERROR only when its ERROR line says so (own_outcome): that line stands, then the command
        again; a traceback, which exits 1 too and says no outcome line, ends with rkmon's own ERROR, then the command
        again."""
        self.confirmed()
        traceback = (
            "Traceback (most recent call last):\n"
            "PermissionError: [Errno 13] Permission denied: '/secrets/teams_webhook'"
        )
        own = "ERROR: teams_webhook: /secrets/teams_webhook: Permission denied; nothing removed"
        old = self.d / "old"
        (old / "monitor" / "state").mkdir(parents=True)
        (old / "monitor" / "state" / "state.json").write_text(json.dumps({"prs": {}}))
        install, adopt = ("web-host", "install"), ("adopt", "--from", str(old), "--yes")
        # what adopt reads before the first start: the image, the secrets' volumes and the logins
        adopting = [
            [r"image inspect", GOOD, 0],
            *deploy_folder.secrets_volumes(True),
            [r"rk logins --json", "[]", 0],
        ]
        for args, said, end, rows in (
            (("secret", "teams_webhook", "--remove"), own, (), ()),
            (
                ("secret", "teams_webhook", "--remove"),
                traceback,
                (
                    "ERROR: docker exited 1, so teams_webhook is not removed (the lines above say why)",
                ),
                (),
            ),
            (
                ("secret", "teams_webhook", "-"),
                traceback,
                (
                    "ERROR: docker exited 1, so teams_webhook is not set (the lines above say why)",
                ),
                (),
            ),
            (
                ("login", "github_read"),
                traceback,
                (
                    "ERROR: docker exited 1, so github_read is not set (the lines above say why)",
                ),
                (),
            ),
            (
                install,
                "ERROR: ssh lost web.example.com during the install (Connection reset by peer)",
                (),
                (),
            ),
            (
                install,
                traceback,
                (
                    "ERROR: docker exited 1, so the request service is not installed (the lines above say why)",
                ),
                (),
            ),
            # adopt last: an apply that stops part-way leaves state 3, whose next line is adopt again
            (
                adopt,
                "ERROR: a made-up reason; the earlier monitor's state is in place, in standby",
                (),
                adopting,
            ),
            (
                adopt,
                traceback,
                ("ERROR: rk adopt apply exited 1 (the lines above say why)",),
                adopting,
            ),
        ):
            with self.subTest(args=args, said=said[:9]):
                asked = r"rk (secret-set|read-login|web-host-install|adopt apply)"
                self.answer([[asked, said, 1], *rows])
                self.stdin.unlink(missing_ok=True)
                rc, out, err = self.rkmon(*args, stdin="made-up-Zq7\n")
                self.assertEqual(rc, 1, out + err)
                self.assertTrue(
                    out.endswith(
                        f"{said}\n" + finished(*end, f"next: {self.w} {' '.join(args)}")
                    ),
                    out,
                )

    def test_the_web_host_install_runs_as_the_writer_for_your_confirmed_settings(self):
        """`rkmon web-host install`: rk web-host-install as the writer, the keeper of the web host's key,
        in the walk's one-off container, handed your confirmed settings and the wrapper's name, which the writer's
        fixes show (container_layout.shown_wrapper); its outcome line is rk's, with its exit code,
        then the deployment state's next line, or the command again (after rk's INTERRUPTED too, whose line
        says what is left on the web host). Under --dry-run, PASSED, and nothing runs; docker's own failure is ERROR.
        Ctrl-C that reaches rkmon itself: INTERRUPTED, exit 130, with what may be left on the web
        host, then the command again. Before your settings are confirmed it is refused, and docker is not asked.
        """
        installed = (
            "INSTALLED: the request service on web.example.com, op's user service rocke-monitor.service, "
            "answers on port 8080"
        )
        rc, out, err = self.rkmon("web-host", "install")
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(out.startswith("REFUSED: "), out)
        self.assertNotIn("web-host-install", "\n".join(self.calls()))
        self.confirmed()
        self.answer([[r"rk web-host-install$", installed, 0]])
        rc, out, err = self.rkmon("web-host", "install")
        self.assertEqual(rc, 0, out + err)
        (run,) = [c for c in self.calls() if "web-host-install" in c]
        self.assertIn(
            f"\tcompose run --rm -T -e {container_layout.HANDED_LAYER_VARIABLE} -e RKMON_WRAPPER={self.w} "
            "-e RKMON_AS=rkwrite rkmon rk web-host-install",
            run,
        )
        self.assertTrue(run.endswith(" rk web-host-install"), run)
        self.assertEqual(out.splitlines()[0], installed)
        self.assertTrue(NEXT_LINE.search(out.splitlines()[-1]), out)
        # --dry-run: the docker command only
        before = len(self.calls())
        rc, out, err = self.rkmon("--dry-run", "web-host", "install")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "PASSED: the writer can install the request service from a one-off container of the image your "
            f"confirmed settings name next: {self.w} web-host install",
            one_line(out),
        )
        self.assertEqual(
            [c for c in self.calls()[before:] if "web-host-install" in c], []
        )
        # rk's own failure and its Ctrl-C, its code and its line; then docker's; then Ctrl-C that reaches rkmon
        refused = "FAILED: web.example.com: port 8080 is held by PID 4242, which is not an http.server; nothing changed"
        stopped = (
            "INTERRUPTED: some of the copies may be on web.example.com, and rocke-monitor.service is as it "
            "was"
        )
        for said, code, rc_, end in (
            (refused, 4, 4, refused),
            (stopped, 130, 130, stopped),
            (
                "Error response from daemon: no such image",
                125,
                1,
                "ERROR: docker exited 125, so the request service is not installed (the lines above say why)",
            ),
            (
                "",
                "SIGINT",
                130,
                "INTERRUPTED: the install was stopped: the web host may have some of its new copies, and its request "
                "service may not run",
            ),
        ):
            with self.subTest(code=code):
                self.answer([[r"rk web-host-install$", said, code]])
                rc, out, err = self.rkmon("web-host", "install")
                self.assertEqual(rc, rc_, out + err)
                self.assertTrue(
                    one_line(out).endswith(f"{end} next: {self.w} web-host install"),
                    out,
                )
        # rk's own refusal in standby, which no state shows with no container (an adopt sets it before the first
        # start): once the monitor says it is in standby, in a one-off container, the state's next line follows it
        standby = (
            "REFUSED: the monitor is in standby, and the monitor you moved from may still use the web host; nothing "
            "changed"
        )
        said = json.dumps({"applied": None, "paused": False, "standby": True})
        self.answer(
            [[r"rk web-host-install$", standby, 4], [r"rk applied --json$", said, 1]]
        )
        before = len(self.calls())
        rc, out, err = self.rkmon("web-host", "install")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            one_line(out).endswith(f"{standby} next: step 6, start: {self.w} start"),
            out,
        )
        (asked,) = [c for c in self.calls()[before:] if "rk applied --json" in c]
        self.assertIn("\tcompose run --rm -T ", asked)

    def test_the_web_host_install_on_a_terminal_reads_what_rk_says_too(self):
        """On a terminal, rk web-host-install runs on one of its own (compose run -it), which rkmon relays both ways
        (relayed), so that a Ctrl-C reaches rk in the container's terminal, and a copy of what rk
        says is kept: its exit 1 is its own ERROR only with its ERROR line (own_outcome), which stands, then the
        command again; a traceback ends with rkmon's own ERROR, then the command again.
        """
        self.confirmed()
        argv = [self.w, "web-host", "install"]
        for said, end in (
            (
                "ERROR: ssh lost web.example.com during the install (Connection reset by peer)",
                (),
            ),
            (
                "Traceback (most recent call last):\nOSError: [Errno 101] Network is unreachable",
                (
                    "ERROR: docker exited 1, so the request service is not installed (the lines above say why)",
                ),
            ),
        ):
            with self.subTest(said=said[:9]):
                self.answer([[r"rk web-host-install$", said, 1]])
                rc, out = sandbox.on_a_terminal(argv, self.home, self.env)
                self.assertEqual(rc, 1, out)
                self.assertTrue(
                    out.replace("\r\n", "\n").endswith(
                        f"{said}\n" + finished(*end, f"next: {self.w} web-host install")
                    ),
                    out,
                )
        runs = [c for c in self.calls() if "web-host-install" in c]
        self.assertEqual(len(runs), 2, runs)
        for run in runs:
            self.assertIn("\tcompose run --rm -it ", run)

    def test_a_terminal_of_its_own_is_relayed_only_from_yours(self):
        """A docker compose run whose words are kept (compose's kept: walk_inside's) is relayed on a terminal of its
        own (relayed) only when it asks docker for one (-it) and your input and output are terminals (interactive):
        with no terminal of yours, what it says comes through a pipe (streamed), whatever its arguments.
        """
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        with mock.patch.dict(os.environ, self.env), mock.patch.object(
            sys, "path", list(sys.path)
        ):
            w = mod.Wrapper(self.w)
            for terminal, way in ((False, "streamed"), (True, "relayed")):
                with self.subTest(terminal=terminal), mock.patch.object(
                    w, "interactive", return_value=terminal
                ), mock.patch.object(w, "only_ours"), mock.patch.object(
                    w, "show"
                ), mock.patch.object(
                    mod, "relayed", return_value=(0, "")
                ) as relayed, mock.patch.object(
                    mod, "streamed", return_value=(0, "")
                ) as streamed:
                    said = w.compose("run", "--rm", "-it", "rkmon", "rk", kept=True)
                    self.assertEqual(said, (0, ""))
                    ran = {
                        "relayed": relayed.call_count,
                        "streamed": streamed.call_count,
                    }
                    self.assertEqual(ran, {"relayed": 0, "streamed": 0, way: 1})

    def test_a_ctrl_c_that_reaches_rkmon_on_a_terminal_ends_what_it_relays(self):
        """On a terminal, a Ctrl-C that reaches rkmon itself while it relays rk web-host-install's terminal (relayed;
        here the stand-in docker sends it): INTERRUPTED, exit 130, then the command again; your terminal's mode is as
        it was before (relayed set it raw meanwhile), and the run on the terminal of its own is ended with it (the
        stand-in, in a session of its own and deaf to its terminal's hangup, would else make its `ends` file half a
        second after the Ctrl-C).
        """
        self.confirmed()
        ends, before, after = (
            self.d / n for n in ("ends", "before.stty", "after.stty")
        )
        self.answer([[r"rk web-host-install$", "", "SIGINT", {}, {"ends": str(ends)}]])
        # rkmon run by a shell that writes the terminal's mode (termios.tcgetattr) before it and after it
        mode = f"{shlex.quote(sys.executable)} -c 'import termios; print(termios.tcgetattr(0))'"
        script = f'{mode} > "$1"; after=$2; shift 2; "$@"; rc=$?; {mode} > "$after"; exit $rc'
        rc, out = sandbox.on_a_terminal(
            ["/bin/sh", "-c", script, "sh", str(before), str(after)]
            + [self.w, "web-host", "install"],
            self.home,
            self.env,
        )
        self.assertEqual(rc, 130, out)
        self.assertTrue(
            out.replace("\r\n", "\n").endswith(
                finished(
                    "INTERRUPTED: the install was stopped: the web host may have some of its new copies, and its "
                    "request service may not run",
                    f"next: {self.w} web-host install",
                )
            ),
            out,
        )
        self.assertEqual(before.read_text(), after.read_text())
        time.sleep(1)
        self.assertFalse(ends.exists())

    def test_a_walk_whose_confirmed_settings_went_once_the_guard_passed_is_refused(
        self,
    ):
        """walk_inside, here for `rkmon web-host install`: when the confirmed settings file goes (or changes) once the
        state guard has passed, there are no confirmed settings to hand the one-off container, so the walk stops with
        the state's refusal (state 2: the file you confirmed is gone), exit 3, and no one-off container runs.
        """
        self.confirmed()
        self.answer([[r"rk web-host-install$", "INSTALLED: on web.example.com", 0]])
        # rkmon as it runs, the settings file you confirmed taken away once its state guard has passed
        probe = (
            "import importlib.machinery, importlib.util, os, sys\n"
            "w, settings = sys.argv[1:3]\n"
            "loader = importlib.machinery.SourceFileLoader('rkmon_wrapper', w)\n"
            "wrapper = importlib.util.module_from_spec(\n"
            "    importlib.util.spec_from_loader('rkmon_wrapper', loader)\n"
            ")\n"
            "loader.exec_module(wrapper)\n"
            "guard = wrapper.Wrapper.state_guard\n"
            "def guarded(self, *args, **kw):\n"
            "    facts = guard(self, *args, **kw)\n"
            "    os.remove(settings)\n"
            "    return facts\n"
            "wrapper.Wrapper.state_guard = guarded\n"
            "sys.argv = [w, *sys.argv[3:]]\n"
            "sys.exit(wrapper.main())\n"
        )
        p = subprocess.run(
            [sys.executable, "-B", "-I", "-c", probe, self.w, str(self.settings)]
            + ["web-host", "install"],
            cwd=self.home,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        self.assertFalse(self.settings.exists())
        self.assertTrue(p.stdout.startswith("REFUSED: "), p.stdout)
        self.assertIn(
            "the settings file you confirmed, ~/rkmon-deploy.yaml, is gone",
            one_line(p.stdout),
        )
        self.assertNotIn("web-host-install", "\n".join(self.calls()))

    def test_the_claude_login_is_the_key_alone_on_stdin_never_printed(self):
        """`rkmon login claude`: the Claude session's key alone, typed or on stdin, never a flag's; rk secret-set
        claude gets it on its stdin, as the monitor (the login is the monitor's own), and it is in no output nor
        docker's log. The outcome says when the session reads it, and while a session runs (rk orchestrate --running,
        asked in the running container only), the steps that start it again: one that has not started yet reads it as
        it starts. No key on stdin: FAILED, and nothing is set."""
        key = "test-claude-key-" + "Zq7"
        self.confirmed()
        self.answer([[r"secret-set", "", 0]])
        rc, out, err = self.rkmon("login", "claude", stdin=key + "\n")
        self.assertEqual(rc, 0, out + err)
        set_ = [c for c in self.calls() if "secret-set" in c][-1]
        self.assertTrue(set_.endswith(" rk secret-set claude"), set_)
        self.assertNotIn("rkwrite", set_)  # the monitor's own login
        self.assertEqual(self.stdin.read_text().strip(), key)
        self.assertNotIn(key, out + err + "\n".join(self.calls()))
        self.assertIn(
            "SET: claude; the Claude session reads it when it next starts\n", out
        )

        def asked():
            return [c for c in self.calls() if c.endswith(" rk orchestrate --running")]

        self.assertEqual(asked(), [])  # no container runs: none is started to ask
        self.stdin.unlink()
        rc, out, err = self.rkmon("login", "claude", stdin=key + "\n", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"SET: claude; the Claude session reads it when it next starts ({self.w} stop --drain, then "
            f"{self.w} start)",
            one_line(out),
        )
        self.assertIn("\tcompose exec -T --user rkmon ", asked()[-1])
        self.assertEqual(self.stdin.read_text().strip(), key)
        self.assertNotIn(key, out + err + "\n".join(self.calls()))
        self.stdin.unlink()
        # The container runs, and its session has not started yet: it reads the key as it starts.
        self.answer(
            [
                [r"secret-set", "", 0],
                [r" rk orchestrate --running$", "ORCHESTRATOR not running", 1],
            ]
        )
        rc, out, err = self.rkmon("login", "claude", stdin=key + "\n", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "SET: claude; the Claude session reads it when it next starts\n", out
        )
        self.assertEqual(len(asked()), 2)
        self.stdin.unlink()
        before = len(self.calls())
        rc, out, err = self.rkmon("login", "claude", stdin="")
        self.assertEqual(rc, 4, out + err)
        self.assertIn("FAILED: no key on stdin for claude; nothing changed\n", out)
        self.assertEqual([c for c in self.calls()[before:] if "secret-set" in c], [])

    def test_the_claude_login_makes_its_test_call_in_the_walk_s_one_off(self):
        """Once the key is set, `rkmon login claude` makes the login's test call, rk claude-test, in the walk's one-off
        container on the deployment's network, and shows its row. A call that does not pass is ERROR (exit
        1): the key stays set, and the next line is the command again. The key is in no output nor docker's
        log."""
        key = "test-claude-key-" + "Zq7"
        self.confirmed()
        self.answer([[r"secret-set", "", 0]])
        rc, out, err = self.rkmon("login", "claude", stdin=key + "\n")
        self.assertEqual(rc, 0, out + err)
        calls = self.calls()
        at = max(i for i, c in enumerate(calls) if "secret-set" in c)
        (test,) = [c for c in calls if "claude-test" in c]
        self.assertGreater(calls.index(test), at)
        self.assertIn(
            f"	compose run --rm -T -e {container_layout.HANDED_LAYER_VARIABLE} ",
            test,
        )  # handed your confirmed settings
        self.assertTrue(test.endswith(" rk claude-test"), test)
        refused = "FAIL  claude  the key is refused, at Anthropic's own API (HTTP 401, authentication_error)"
        self.answer([[r"secret-set", "", 0], [r"rk claude-test", refused, 4]])
        self.stdin.unlink()
        rc, out, err = self.rkmon("login", "claude", stdin=key + "\n")
        self.assertEqual(rc, 1, out + err)
        self.assertIn(refused, out)
        self.assertIn(
            "ERROR: claude is set, but its test call did not pass (the lines above say why) "
            f"next: {self.w} login claude",
            one_line(out),
        )
        self.assertEqual(self.stdin.read_text().strip(), key)  # set
        self.assertNotIn(key, out + err + "\n".join(self.calls()))

    def test_an_ssh_key_shows_its_host_s_own_keys_and_records_them_once_accepted(self):
        """Once the ssh key of a host is set, `rkmon secret NAME FILE` shows the host's own keys as rk host-key NAME
        reads them in the walk's one-off container, where ssh reaches the host, by their fingerprints, and on a
        terminal asks whether to record them: yes records the lines it showed, on rk host-key NAME
        --record's stdin, as the monitor (SET, exit 0); no, or no terminal, records nothing (CANCELLED, exit 1). The
        key is in no output nor docker's log."""
        line = "web.example.com ssh-ed25519 AAAAExampleWebHostKey"
        shown = {
            "host": "web.example.com",
            "port": 22,
            "lines": [line],
            "fingerprints": ["ED25519 SHA256:example1"],
            "recorded": False,
            "others": 0,
            "why": "",
        }
        key = self.home / "web_key"
        key.write_text("made-up-ssh-key-Zq7\n")
        key.chmod(0o600)
        self.confirmed()
        self.answer(
            [
                [r"secret-set", "", 0],
                [r"rk host-key web_host_ssh_key --json$", json.dumps(shown), 0],
                [r"rk host-key web_host_ssh_key --record$", "recorded", 0],
            ]
        )
        argv = [self.w, "secret", "web_host_ssh_key", str(key)]
        question = b"Record them as the web host's own keys, the only ones ssh trusts it by? [y/N] "
        rc, out = sandbox.on_a_terminal(
            argv, self.home, self.env, replies=[(question, b"y\n")]
        )
        self.assertEqual(rc, 0, out)
        flat = one_line(out)
        self.assertIn(
            "The web host's own keys (web.example.com), as the container reads them; compare their fingerprints "
            "with those its admin gives you: ED25519 SHA256:example1 Record them",
            flat,
        )
        self.assertIn(
            "SET: web_host_ssh_key; the web host's own keys (web.example.com) are recorded",
            flat,
        )
        calls = self.calls()
        (set_,) = [c for c in calls if "secret-set web_host_ssh_key" in c]
        (read,) = [c for c in calls if "host-key web_host_ssh_key --json" in c]
        (record,) = [c for c in calls if "host-key web_host_ssh_key --record" in c]
        self.assertLess(calls.index(set_), calls.index(read))
        self.assertLess(calls.index(read), calls.index(record))
        self.assertIn("RKMON_AS=rkwrite", set_)  # the writer's key, as the writer
        # the host's keys, as the monitor, for your confirmed settings
        for c in (read, record):
            self.assertIn(
                f"\tcompose run --rm -T -e {container_layout.HANDED_LAYER_VARIABLE} ",
                c,
            )
            self.assertNotIn("RKMON_AS=", c)
        self.assertTrue(self.stdin.read_text().endswith(f"\n{line}\n"))
        self.assertNotIn("made-up-ssh-key", out + "\n".join(calls))
        # No, as the question's default: nothing recorded.
        for answer in (b"n\n", b"\n"):
            with self.subTest(answer=answer):
                before = len(self.calls())
                rc, out = sandbox.on_a_terminal(
                    argv, self.home, self.env, replies=[(question, answer)]
                )
                self.assertEqual(rc, 1, out)
                self.assertIn(
                    "CANCELLED: you said no: web_host_ssh_key is set, and the web host's own keys as the container "
                    "reads them are not recorded",
                    one_line(out),
                )
                self.assertFalse([c for c in self.calls()[before:] if "--record" in c])
        # No terminal: the keys shown, nothing recorded, and the command named sets the key from its file, also
        # when it came on stdin: on a terminal, `secret NAME -` asks for it on one line, which no ssh key is.
        for args, stdin in (
            (argv[1:], None),
            (argv[1:3] + ["-"], "made-up-ssh-key-Zq7\n"),
        ):
            with self.subTest(args=args):
                before = len(self.calls())
                rc, out, err = self.rkmon(*args, stdin=stdin)
                self.assertEqual(rc, 1, out + err)
                self.assertIn("ED25519 SHA256:example1", out)
                self.assertIn(
                    "CANCELLED: web_host_ssh_key is set, but the web host's own keys are recorded only once you "
                    f"accept them, on a terminal ({self.w} secret web_host_ssh_key FILE)",
                    one_line(out),
                )
                self.assertFalse([c for c in self.calls()[before:] if "--record" in c])
                self.assertNotIn("made-up-ssh-key", out + err + "\n".join(self.calls()))

    def test_an_ssh_key_s_host_keys_recorded_already_unread_or_unnamed(self):
        """`rkmon secret gpu_ssh_key FILE` once the key is set: the GPU login host's own keys recorded already, SET and
        nothing asked; none read, ERROR (exit 1) and the command again while none is recorded, SET saying so while
        some are; no host named yet, SET saying its keys are not read; an answer from rk host-key that is no JSON, its
        lines, then ERROR. Keys that replace others recorded: a warning before the question. The key is set first
        every time, as the monitor, and is in no output."""
        line = "[login.example.com]:2222 ssh-ed25519 AAAAExampleGpuHostKey"
        base = {
            "host": "login.example.com",
            "port": 2222,
            "lines": [line],
            "fingerprints": ["ED25519 SHA256:example1"],
            "recorded": False,
            "others": 0,
            "why": "",
        }
        unnamed = "your settings name no host for it (features.review_automation.remote_gpu_tests.cluster.login_host)"
        key = self.home / "gpu_key"
        key.write_text("made-up-ssh-key-Zq7\n")
        key.chmod(0o600)
        self.confirmed()
        argv = ("secret", "gpu_ssh_key", str(key))
        again = f"next: {self.w} secret gpu_ssh_key {key}"
        for answer, code, said in (
            (
                dict(base, recorded=True),
                0,
                "SET: gpu_ssh_key; the GPU login host's own keys (login.example.com) are recorded already",
            ),
            (
                dict(base, lines=[], fingerprints=[], why="Connection refused"),
                1,
                "ERROR: gpu_ssh_key is set, but the GPU login host's own keys were not read from the container "
                "(Connection refused): until one is recorded, ssh does not reach login.example.com "
                + again,
            ),
            (
                dict(
                    base, lines=[], fingerprints=[], why="Connection refused", others=2
                ),
                0,
                "SET: gpu_ssh_key; the GPU login host's own keys were not read (Connection refused), and those "
                "recorded for it (2 keys) are kept",
            ),
            (
                dict(base, host="", lines=[], fingerprints=[], why=unnamed),
                0,
                f"SET: gpu_ssh_key; {unnamed}, so the GPU login host's own keys are not read yet",
            ),
            (
                "REFUSED: the configuration cannot be read",
                1,
                "REFUSED: the configuration cannot be read ERROR: gpu_ssh_key is set, but the GPU login host's own "
                "keys could not be read (the lines above say why) " + again,
            ),
        ):
            with self.subTest(said=said):
                text = json.dumps(answer) if isinstance(answer, dict) else answer
                read = isinstance(answer, dict) and answer["lines"]
                self.answer(
                    [
                        [r"secret-set", "", 0],
                        # rk host-key's own exit code: 4 when it reads none, or refuses
                        [r"rk host-key gpu_ssh_key --json$", text, 0 if read else 4],
                    ]
                )
                before = len(self.calls())
                rc, out, err = self.rkmon(*argv)
                self.assertEqual(rc, code, out + err)
                self.assertIn(said, one_line(out))
                calls = self.calls()[before:]
                (set_,) = [c for c in calls if "secret-set gpu_ssh_key" in c]
                self.assertNotIn("RKMON_AS=", set_)  # the monitor's key
                self.assertFalse([c for c in calls if "--record" in c])
                self.assertNotIn("made-up-ssh-key", out + err + "\n".join(calls))
        # Keys that replace others recorded: the warning, then the question.
        self.answer(
            [
                [r"secret-set", "", 0],
                [
                    r"rk host-key gpu_ssh_key --json$",
                    json.dumps(dict(base, others=1)),
                    0,
                ],
                [r"rk host-key gpu_ssh_key --record$", "recorded", 0],
            ]
        )
        question = b"Record them as the GPU login host's own keys, the only ones ssh trusts it by? [y/N] "
        rc, out = sandbox.on_a_terminal(
            [self.w, *argv], self.home, self.env, replies=[(question, b"y\n")]
        )
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "ED25519 SHA256:example1 They replace the 1 key recorded for it now: unless the host was set up anew, "
            "ask its admin why its keys changed before you accept them. Record them",
            one_line(out),
        )
        self.assertIn(
            "SET: gpu_ssh_key; the GPU login host's own keys (login.example.com) are recorded",
            one_line(out),
        )

    def test_the_login_walk_records_the_host_s_own_key_of_an_ssh_key_that_is_set(self):
        """Deploy step 5 for the ssh key of a host, set while the host's own key is not recorded, as one the
        one-time move copies: a FAIL row whose fix is `rkmon login`, which shows that key and records it once you
        accept it. On a terminal the walk asks no question of its own for it: it shows the host's own
        keys as rk host-key NAME reads them, a yes records them, and the ssh key is not asked for again; the row is
        shown again, then PASSED. A no records nothing, and is your no: CANCELLED (exit 1).
        """
        w = self.w
        line = "web.example.com ssh-ed25519 AAAAExampleWebHostKey"
        shown = {
            "host": "web.example.com",
            "port": 22,
            "lines": [line],
            "fingerprints": ["ED25519 SHA256:example1"],
            "recorded": False,
            "others": 0,
            "why": "",
        }
        what = secrets_registry.entry({}, "web_host_ssh_key").what
        wrong = {
            "name": "web_host_ssh_key",
            "what": what,
            "needed_by": ["team dashboard (publish)"],
            "state": "WRONG",
            "found": "the key is set; the web host's own key is not recorded",
            "command": "login",
            "host_key": True,
        }
        done = dict(
            wrong,
            state="SET",
            found="the writer's volume; the web host's own key is recorded",
            command="secret web_host_ssh_key",
            host_key=False,
        )
        recorded = self.d / "recorded"
        # the team dashboard live, published to its web host over ssh
        self.write_settings(
            features={"dashboards": {"team": {**PUBLISHED, "mode": "live"}}}
        )
        self.confirmed()
        listed = self.login_rows()  # each other one the settings need is set

        def rows(row: dict) -> str:
            return json.dumps([row if r["name"] == row["name"] else r for r in listed])

        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [
                    r"rk logins --json",
                    rows(wrong),
                    0,
                    {},
                    {"unless": str(recorded)},
                ],
                [r"rk logins --json", rows(done), 0],
                [r"rk host-key web_host_ssh_key --json$", json.dumps(shown), 0],
                [
                    r"rk host-key web_host_ssh_key --record$",
                    "recorded",
                    0,
                    {},
                    {"makes": str(recorded)},
                ],
            ]
        )
        fail = checks.render(
            [
                checks.Check(
                    "web_host_ssh_key",
                    "FAIL",
                    "wrong (the key is set; the web host's own key is not recorded): "
                    f"{what}, needed by team dashboard (publish)",
                    checks.Steps(
                        (
                            checks.Command((w, "login")),
                            "shows it, and records it once you accept it",
                        )
                    ),
                )
            ]
        )
        rc, out, err = self.rkmon("login")  # no terminal: the rows alone
        self.assertEqual((rc, err), (4, ""), out)
        self.assertIn(fail + "\n", out)
        self.assertFalse([c for c in self.calls() if "host-key" in c])
        question = b"Record them as the web host's own keys, the only ones ssh trusts it by? [y/N] "
        rc, out = sandbox.on_a_terminal(
            [w, "login"], self.home, self.env, replies=[(question, b"y\n")]
        )
        self.assertEqual(rc, 0, out)
        self.assertNotIn("it now?", out)
        flat = one_line(out)
        self.assertIn(
            "The web host's own keys (web.example.com), as the container reads them; compare their fingerprints "
            "with those its admin gives you: ED25519 SHA256:example1 Record them",
            flat,
        )
        passed = checks.render(
            [checks.Check("web_host_ssh_key", "PASS", done["found"])]
        )
        self.assertIn(passed.replace("\n", "\r\n"), out)
        self.assertIn("PASSED: every login and secret your settings need is set", flat)
        calls = self.calls()
        # the ssh key is not asked for again
        self.assertFalse([c for c in calls if "secret-set" in c])
        (read,) = [c for c in calls if "host-key web_host_ssh_key --json" in c]
        (record,) = [c for c in calls if "host-key web_host_ssh_key --record" in c]
        self.assertLess(calls.index(read), calls.index(record))
        self.assertTrue(self.stdin.read_text().endswith(f"{line}\n"))
        # No: nothing recorded, the row shown again as it was, and the walk is CANCELLED.
        recorded.unlink()
        before = len(self.calls())
        rc, out = sandbox.on_a_terminal(
            [w, "login"], self.home, self.env, replies=[(question, b"n\n")]
        )
        self.assertEqual(rc, 1, out)
        self.assertFalse([c for c in self.calls()[before:] if "--record" in c])
        self.assertEqual(out.count(fail.replace("\n", "\r\n")), 2)
        self.assertIn(
            "CANCELLED: you said no; of the logins and secrets your settings need, 1 of 4 is missing or wrong",
            one_line(out),
        )

    def test_a_key_typed_on_a_terminal_no_answer_cancels_and_ctrl_c_stops(self):
        """On a terminal, `rkmon login claude` asks for the key and `rkmon secret NAME` for the value, without echo:
        an empty answer, or an end of input, is CANCELLED (exit 1) on a line of its own after the question; Ctrl-C ends
        the command as at any question, by the signal (exit 130 in the shell), on a line of its own. Nothing is
        set."""
        self.confirmed()
        self.answer([[r"secret-set", "", 0]])
        for args, prompt, what in (
            (
                ("login", "claude"),
                "The Claude session's key: an Anthropic API key, or your gateway's token (not shown): ",
                "no key typed for claude",
            ),
            (
                ("secret", "teams_webhook"),
                "teams_webhook (not shown; for a key file, give its path instead): ",
                "no value typed for teams_webhook",
            ),
        ):
            for typed in (b"\n", b"\x04", b"\x03"):
                with self.subTest(args=args, typed=typed):
                    before = len(self.calls())
                    rc, out = sandbox.on_a_terminal(
                        [self.w, *args],
                        self.home,
                        self.env,
                        replies=[(prompt.encode(), typed)],
                    )
                    if typed == b"\x03":  # Ctrl-C
                        self.assertEqual(rc, -signal.SIGINT, out)
                        self.assertTrue(out.endswith(f"{prompt}\r\n"), out)
                    else:
                        self.assertEqual(rc, 1, out)
                        self.assertIn(
                            f"{prompt}\r\nCANCELLED: {what}; nothing changed\r\n", out
                        )
                    self.assertEqual(
                        [c for c in self.calls()[before:] if "secret-set" in c], []
                    )

    def test_a_made_value_is_shown_once_on_a_terminal_only(self):
        """`rkmon secret admin_access_token --generate`: a new value, into the container on rk secret-set's stdin and
        shown once, on the terminal (rk secret-set's own line never shows it)."""
        self.confirmed()
        self.answer([[r"secret-set", "admin_access_token: set (not shown)", 0]])
        rc, out = sandbox.on_a_terminal(
            [self.w, "secret", "admin_access_token", "--generate"], self.home, self.env
        )
        self.assertEqual(rc, 0, out)
        value = self.stdin.read_text().strip()
        self.assertRegex(value, r"^[A-Za-z0-9_-]{32}$")
        self.assertEqual(out.count(value), 1, out)
        self.assertIn(
            f"admin_access_token, shown this once (keep it where you keep passwords; {self.w} secret "
            "admin_access_token --generate makes another):",
            one_line(out),
        )
        self.assertNotIn(value, "\n".join(self.calls()))


if __name__ == "__main__":
    unittest.main()
