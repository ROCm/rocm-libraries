# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Another deployment on this host, end to end: docker/rkmon run on a copy of the package against the stand-in docker,
as tests/deploy_path_fixture.py sets it up. Before the first confirm this folder has no compose project, whatever the
other deployment is called. A container name another deployment on this host uses refuses each command that would reach
the container (state 1), the probes saying that what they cannot read is unknown, and its port is not taken for this
deployment's; an answer from docker that cannot tell whose the name is stops the command; no message names the other
deployment's folder. rkmon remove refuses that container name too, and before the first start runs docker compose in
this deployment's compose project.
"""

import pathlib
import re
import shlex
import socket
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
from deploy_path_fixture import FRESH_ENV, GOOD, DeployPathCase  # noqa: E402
from rendered import finished  # noqa: E402
from rkmon_lib import commands  # noqa: E402

# The docker calls that name no compose project, volume or container: docker's version, images, and a one-off
# container of an image alone, with no network, no volume and no name.
NAMES_NONE = re.compile(
    r"^(?:version --format |compose version\b|image (?:inspect|ls)\b|load -i |save |push |tag |build |buildx "
    r"|run --rm --network none (?!.*(?:--mount|--volume|-v |--name)))"
)


class TestOtherDeployments(DeployPathCase):
    def test_before_the_first_confirm_no_command_names_a_deployment(self):
        """Before the first confirm this folder has no compose project, whatever another deployment on this host
        is called (here rkmon, from another folder): each command, its first example and a few more, names no compose
        project, volume or container to docker (NAMES_NONE); check and confirm ask only whose the settings file's own
        container.name is, of its containers, volumes and network. None of them confirms anything.
        """
        own = re.escape(deploy_folder.NAME)
        file_own = re.compile(
            rf"^(?:ps --all --filter label=com\.docker\.compose\.project={own} "
            rf"|volume ls --filter label=com\.docker\.compose\.project={own} "
            rf"|network inspect .* {own}_default$)"
        )
        for name in ("rocke-monitor.tar.gz", "webhook.txt", "package.tgz"):
            (self.home / name).write_text("x\n", encoding="utf-8")
        runs = [
            (
                row.name,
                [
                    f"{self.home}/{word[2:]}" if word.startswith("~/") else word
                    for word in shlex.split(row.examples[0][0])
                ],
            )
            for row in commands.TABLE
        ]
        file = str(self.settings)
        runs += [
            ("check", ["check", "--settings", file]),
            ("next", ["next", "--settings", file]),
            ("login", ["login", "--no-input"]),
            ("login", ["login", "github_write"]),
            ("start", ["--dry-run", "start"]),
            ("remove", ["remove", "--delete-data", "--yes"]),
        ]
        # another deployment runs on this host, from another folder, in the compose project rkmon
        another = {"FAKE_DOCKER_WORKDIR": "/srv/other/docker"}
        asked_own = False
        for name, argv in runs:
            with self.subTest(argv=argv):
                if self.log.is_file():
                    self.log.unlink()
                rc, out, err = self.rkmon(*argv, **FRESH_ENV.get(name, {}), **another)
                self.assertIsNone(
                    self.confirm_record(), f"{argv}: exit {rc}\n{out}{err}"
                )
                for call in self.docker_calls():
                    mine = name in ("check", "confirm") and file_own.match(call)
                    asked_own = asked_own or bool(mine)
                    self.assertTrue(
                        NAMES_NONE.match(call) or mine,
                        f"{' '.join(argv)}: docker {call}",
                    )
        self.assertTrue(
            asked_own, "the check asks whose the file's own container name is"
        )

    def test_another_deployments_container_name(self):
        """Another folder's deployment made a container of this folder's container name: each command that would reach
        the container refuses (state 1) with the fix and the next line, and runs no docker compose; status says the
        container is unknown and next each step it cannot read, exit 0; logs, login and first-tick say so too, then
        refuse; the check fails the name, and the port that container holds is not this deployment's; no message names
        that folder. Nothing asks that container: no docker compose command runs.
        """
        self.confirmed()
        self.log.unlink()
        other = {
            "FAKE_DOCKER_PROJECT": "rkmon-test",
            "FAKE_DOCKER_WORKDIR": "/srv/other/docker",
        }
        printed = ""
        refusal = (
            "REFUSED: the container name rkmon-test is used by another deployment on this host; nothing changed "
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rkmon-test-2 (docker "
            "compose ls --all lists the names in use) "
            f"next: step 3, confirm: {self.w} confirm --settings ~/rkmon-deploy.yaml"
        )
        for args in (
            ("start",),
            ("--dry-run", "start"),
            ("stop",),
            ("doctor",),
            ("remove",),
            ("shell",),
            ("rk", "status"),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args, **other)
                printed += out + err
                self.assertEqual(rc, 4, out + err)
                self.assertIn(refusal, self.flat(out))
        for args, what in (
            (("logs",), "the container's log"),
            (("login",), "the logins"),
            (("first-tick",), "the first tick"),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args, **other)
                printed += out + err
                self.assertEqual(rc, 4, out + err)
                self.assertIn(f"{what}: unknown\n", out)
                self.assertIn(refusal, self.flat(out))
        rc, out, err = self.rkmon("status", **other)
        printed += out + err
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            self.flat(out).startswith(
                "deployment rkmon-test (this package folder) · container unknown · "
            ),
            out,
        )
        self.assertIn(
            "\nstate: the container name rkmon-test is used by another deployment on this host\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                f"\nnext: step 3, confirm: {self.w} confirm --settings ~/rkmon-deploy.yaml\n"
            ),
            out,
        )
        # next: what this folder's records say, the steps of the container unknown, the fix and the state's command.
        rc, out, err = self.rkmon("next", **other)
        printed += out + err
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "  3. REDO  confirm: the container name rkmon-test is used by another deployment on this host\n",
            out,
        )
        for n, name in ((5, "login"), (6, "start"), (7, "doctor"), (8, "first tick")):
            self.assertIn(f"  {n}. TODO  {name}: unknown\n", out)
        self.assertIn(
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rkmon-test-2",
            self.flat(out),
        )
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"next: step 3, confirm: {self.w} confirm --settings ~/rkmon-deploy.yaml"
                )
            ),
            out,
        )
        self.assertFalse(
            [c for c in self.calls() if c.split("\t", 1)[1].startswith("compose ")]
        )
        # The check: the name fails, and the port that container holds is busy, not this deployment's.
        self.answer(
            [
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
                [r"DockerRootDir", str(self.d), 0],
                [r"\{\{\.Ports\}\}", f"127.0.0.1:{self.port}->8080/tcp", 0],
            ]
        )
        with socket.socket() as s:
            s.bind(("127.0.0.1", self.port))
            s.listen()
            rc, out, err = self.rkmon("check", **other)
        printed += out + err
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAIL  container name: rkmon-test is used by another deployment on this host",
            out,
        )
        self.assertIn(
            f"fix: set container.name in {self.settings} to a name of its own, such as rkmon-test-2",
            self.flat(out),
        )
        self.assertRegex(out, rf"(?m)^  FAIL  port: .*127\.0\.0\.1:{self.port}")
        self.assertNotIn("/srv/other", printed)
        # Its own container, made from this folder: the name is this deployment's, and so is its port.
        rc, out, err = self.rkmon(
            "check",
            FAKE_DOCKER_PROJECT="rkmon-test",
            FAKE_DOCKER_WORKDIR=str(self.pkg / "docker"),
        )
        self.assertIn(
            "PASS  container name: rkmon-test: no other deployment on this host uses it",
            out,
        )
        self.assertIn("PASS  ports", out)

    def test_remove_delete_data_refuses_another_deployment_s_container_name(self):
        """rkmon remove --delete-data --yes in state 1: another folder's deployment made a container of this
        folder's container name, and volumes of that compose project's names are on this host. The state's refusal,
        exit 4, before anything is asked of the volumes: no volume listed or removed, no docker compose command, and no
        message names that folder."""
        self.confirmed()
        self.log.unlink()
        self.answer(
            [
                *deploy_folder.volumes(deploy_folder.VOLUMES),
                *deploy_folder.secrets_volumes(True),
            ]
        )
        rc, out, err = self.rkmon(
            "remove",
            "--delete-data",
            "--yes",
            FAKE_DOCKER_PROJECT="rkmon-test",
            FAKE_DOCKER_WORKDIR="/srv/other/docker",
        )
        self.assertEqual(rc, 4, out + err)
        self.assertEqual(
            self.flat(out),
            "REFUSED: the container name rkmon-test is used by another deployment on this host; nothing changed "
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rkmon-test-2 (docker "
            f"compose ls --all lists the names in use) next: step 3, confirm: {self.w} confirm --settings "
            "~/rkmon-deploy.yaml",
        )
        asked = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertEqual(
            [a for a in asked if a.startswith(("compose ", "volume "))], [], asked
        )
        self.assertNotIn("/srv/other", out + err)

    def test_remove_runs_in_this_deployment_s_compose_project_before_the_first_start(
        self,
    ):
        """rkmon remove --delete-data --yes in states 4 to 6, confirmed and never started, so with no docker/.env,
        and with the volumes the logins made left: with the file in another layout (4) or changed since the confirm
        (5), there is no confirmed plan either, so none of its variables, and docker compose cannot read the compose
        files with variables put in; in 6 there is no image. docker compose down still runs in this deployment's
        compose project, the confirmed container.name (compose_env), never in the one compose.yaml names; the volumes
        are found in a read of the files with no variable put in; then each volume goes, and the state's next line
        follows."""
        name = deploy_folder.NAME
        secrets = [f"{name}_rkmon-secrets", f"{name}_rkmon-secrets-write"]
        # docker compose's refusal of the files with variables put in while one they need has no value
        unread = [
            r"^compose config --format json$",
            "error while interpolating services.rkmon.image: required variable RKMON_IMAGE is missing a value: "
            "see ./rkmon next",
            1,
            {"RKMON_IMAGE": ""},
        ]
        layout_1 = "kind: deploy_settings\nschema_version: 1\nrepository:\n  name: acme/widgets\n"
        for state, change, next_line in (
            (
                4,
                lambda: self.settings.write_text(layout_1, encoding="utf-8"),
                f"next: {self.w} migrate --settings ~/rkmon-deploy.yaml",
            ),
            (
                5,
                lambda: self.write_settings(memory_gib=16),
                f"next: step 3, confirm: {self.w} confirm",
            ),
            (6, lambda: None, f"next: step 4, image: {self.w} image load IMAGE_FILE"),
        ):
            with self.subTest(state=state):
                self.write_settings()
                self.confirmed()
                change()
                self.settings.chmod(0o600)
                self.assertFalse((self.pkg / "docker" / ".env").exists())
                self.log.unlink(missing_ok=True)
                needs = {"COMPOSE_PROJECT_NAME": name}
                if state < 6:  # and nothing of the plan's: there is none
                    needs["RKMON_IMAGE"] = ""
                self.answer(
                    [
                        [r"image inspect", "", 1],
                        [r"^compose down$", "", 0, needs],
                        [r"^compose down$", "not this deployment's compose project", 1],
                        unread,
                        *deploy_folder.volumes(secrets, env=False),
                        *deploy_folder.secrets_volumes(True),
                    ]
                )
                rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(
                    self.flat(out).startswith(
                        f"REMOVED: this deployment's volumes {next_line}"
                    ),
                    out,
                )
                asked = [c.split("\t", 1)[1] for c in self.calls()]
                self.assertIn("compose down", asked)
                self.assertIn(f"volume rm {' '.join(secrets)}", asked)

    def test_a_fresh_folder_beside_another_deployment(self):
        """Nothing recorded in this folder yet, and another deployment runs on this host (its compose project rkmon,
        from another folder): this folder has no compose project before its first confirm, so a command refuses as an
        earlier step (exit 3) with state 2's sentence and the step to take, status and next say the state with no
        note, and docker is asked nothing; once this folder records and confirms a file whose container.name is free,
        the commands run.
        """
        w, other = self.w, {"FAKE_DOCKER_WORKDIR": "/srv/other/docker"}
        rc, out, err = self.rkmon("start", **other)
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                    self.state_2_next(),
                )
            ),
            out,
        )
        self.assertIn(
            f"next: step 1, template: {w} template deploy ~/rkmon-deploy.yaml",
            self.flat(out),
        )
        rc, out, err = self.rkmon("status", **other)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "deployment: none confirmed yet (this package folder) · no container\n"
                "state: no deploy settings are confirmed in this package folder\n"
            ),
            out,
        )
        self.assertNotIn("note:", out)
        rc, out, err = self.rkmon("next", **other)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "The deploy path (QUICK_DEPLOY_GUIDE.md):\n"
                "  1. TODO  template: no deploy settings file named yet\n"
                "  2. TODO  check: after step 1\n"
                "  3. TODO  confirm: after step 2\n"
                "  4. TODO  image: after step 3\n"
                "  5. TODO  login: after step 3\n"
                "  6. TODO  start: after step 3\n"
                "  7. TODO  doctor: after step 6\n"
                "  8. TODO  first tick: after step 6\n"
                + finished(
                    f"next: step 1, template: {w} template deploy ~/rkmon-deploy.yaml",
                    f"(have a file already? {w} next --settings FILE)",
                )
            ),
            out,
        )
        self.assertEqual(self.calls(), [])
        self.answer([[r"image inspect", "", 1]])
        confirm = ("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(self.rkmon(*confirm, **other)[0], 0)
        self.answer(
            [[r"image inspect", GOOD, 0]]
        )  # the start's guards run under --dry-run too
        rc, out, err = self.rkmon("--dry-run", "start", **other)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("docker compose up -d", out)

    def test_whose_the_container_name_is_unknown_stops_the_command(self):
        """docker's answer about the name's containers cannot be read: the command stops (ERROR, exit 1); docker gives
        no answer: an earlier step (REFUSED, exit 3). Nothing of the container is asked; next and status say unknown:
        next with the records' and the file's steps, the container's steps unknown and why, the reason in its note,
        and the state's next line where the records decide it alone (step 3 here), else step 2's check.
        """
        w = self.w
        self.confirmed()
        self.write_settings(
            memory_gib=13
        )  # changed after the confirm: state 5, which the records decide alone
        self.log.unlink()
        unreadable = [[r"^ps --all --filter", "not a JSON answer", 0]]
        self.answer(unreadable)
        rc, out, err = self.rkmon("stop")
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            "ERROR: docker's answer about its containers is neither JSON lines nor one JSON value, so whose they are "
            f"is unknown; nothing changed next: {w} stop",
            self.flat(out),
        )
        rc, out, err = self.rkmon("next")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "  3. REDO  confirm: your settings file changed after the last confirm\n",
            out,
        )
        self.assertEqual(
            re.findall(r"(?m)^  [4-8]\. TODO  [a-z ]+: (.*)$", out),
            ["unknown (docker's answer cannot be read)"] * 5,
        )
        self.assertIn(
            "note: docker's answer about its containers is neither JSON lines nor one JSON value, so whose they are "
            f"is unknown next: step 3, confirm: {w} confirm",
            self.flat(out),
        )
        no_answer = [
            [
                r"^ps --all --filter",
                "Cannot connect to the Docker daemon. Is it running?",
                1,
            ]
        ]
        self.answer(no_answer)
        rc, out, err = self.rkmon("doctor")
        self.assertEqual(rc, 3, out + err)
        self.assertIn(
            "REFUSED: docker does not answer (Cannot connect to the Docker daemon. Is it running?); nothing changed "
            f"next: {w} doctor",
            self.flat(out),
        )
        rc, out, err = self.rkmon("status")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            "deployment rkmon-test (this package folder) · container unknown\n"
            "state: your settings file changed after the last confirm\n"
            "note: docker does not answer (Cannot connect to the Docker daemon. Is it running?)\n"
            f"next: step 3, confirm: {w} confirm\n",
        )
        self.assertFalse(
            [c for c in self.calls() if c.split("\t", 1)[1].startswith("compose ")]
        )
        # The file as confirmed again: the state after the records' is docker's to say: step 2's check, which says
        # what docker answers.
        self.write_settings()
        self.log.unlink()
        self.answer(no_answer)
        rc, out, err = self.rkmon("next")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("  3. DONE  confirm: ", out)
        self.assertEqual(
            re.findall(r"(?m)^  [4-8]\. TODO  [a-z ]+: (.*)$", out),
            ["unknown (docker does not answer)"] * 5,
        )
        self.assertTrue(
            out.endswith(
                "\nnote: docker does not answer (Cannot connect to the Docker daemon. Is it running?)\n"
                + finished(
                    f"next: step 2, check: {w} check --settings ~/rkmon-deploy.yaml"
                )
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "\tcompose " in c])


if __name__ == "__main__":
    unittest.main()
