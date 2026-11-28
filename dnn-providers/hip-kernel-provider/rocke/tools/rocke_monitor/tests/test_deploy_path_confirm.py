# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deploy steps 1 to 3, the template, the check and the confirm, end to end: docker/rkmon run on a copy of the package
against the stand-in docker, as tests/deploy_path_fixture.py sets it up. The template step writes your settings file
outside the package folder. The check passes a ready host and fails what it lacks (docker, its compose, free ports, room
for the container's share of its memory and CPUs), and settings and secrets left in the package folder; a docker
question takes none of your input. The confirm refuses a template, a file in the package, a file in an earlier layout, a
file that holds a secret's value (the value never shown) and what a check fails, and records the settings it confirms in
this folder's records, writing nothing else; it asks no container before this folder's own exists. After the first start
it shows only what changes, and after a package update it names the package first. The start and apply refuse settings
never confirmed, or changed since.
"""

import io
import json
import os
import pathlib
import re
import shutil
import stat
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
import sandbox  # noqa: E402
from deploy_path_fixture import (  # noqa: E402
    GOOD,
    INSTALLED,
    LABELS,
    MAKES,
    PUBLISHED,
    DeployPathCase,
)
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import (  # noqa: E402
    claude_env_shape,
    container_layout,
    deliver,
    deploy_plan,
    deployment,
    file_types,
    image,
    records,
    settings_file,
)


class TestConfirm(DeployPathCase):
    def test_the_template_step_writes_your_file_then_names_the_check(self):
        """Deploy step 1: the banner, then WROTE, the file yours only, and step 2's line; with no profile, the
        schema's first. Under --dry-run, PASSED, nothing written, and the deployment state's next line. A file
        named from the folder you ran it in is that folder's, and a refusal's fix names the free name and the file as
        its other lines do, `~/` in your home folder. Docker is not asked while no settings file is recorded.
        """
        f = self.home / "new.yaml"
        rc, out, err = self.rkmon("--dry-run", "template", "deploy", "new.yaml")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertFalse(f.exists())
        self.assertEqual(
            out,
            "== deploy step 1 of 8, template: write a deploy settings file for you to fill in (dry run: nothing "
            "changes)\n"
            + finished(
                "PASSED: ~/new.yaml can be written (profile full; yours only, permissions 600); nothing written",
                self.state_2_next(),
            ),
        )
        rc, out, err = self.rkmon("template", "deploy", "new.yaml")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        banner, wrote = out.splitlines()[:2]
        self.assertEqual(
            banner,
            "== deploy step 1 of 8, template: write a deploy settings file for you to fill in",
        )
        self.assertRegex(
            wrote,
            r"^WROTE ~/new\.yaml \(profile full; yours only, permissions 600\): fill in the \d+ values marked "
            r"<\.\.\.>$",
        )
        self.assertEqual(
            out,
            f"{banner}\n"
            + finished(
                wrote, f"next: step 2, check: {self.w} check --settings ~/new.yaml"
            ),
        )
        rc, out, err = self.rkmon("template", "deploy", "new.yaml")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"fix: {self.w} template deploy ~/new-2.yaml, or move that one aside first: mv ~/new.yaml "
            "~/new.yaml.old",
            " ".join(out.split()),
        )
        self.assertEqual(self.calls(), [])

    def test_the_template_step_refuses_a_file_in_the_package(self):
        """Deploy step 1 writes your settings file outside the package folder: a file named in it is refused, with
        nothing written, and the fix names one in your home folder; then REFUSED and the deployment state's next line,
        state 2's in a folder with no settings recorded."""
        inside = self.pkg / "docker" / "mine.yaml"
        rc, out, err = self.rkmon("template", "deploy", str(inside))
        self.assertEqual(rc, 4, out + err)
        self.assertFalse(inside.exists())
        flat = " ".join(out.split())
        self.assertIn(f"FAIL private file: {inside} is inside the package folder", flat)
        self.assertIn(f"fix: {self.w} template deploy ~/mine.yaml", flat)
        self.assertTrue(
            out.endswith(
                "\n\n" + finished("REFUSED: nothing written", self.state_2_next())
            ),
            out,
        )
        # A file of the package's there too: each refusal its row, then one outcome.
        compose = self.pkg / "docker" / "compose.yaml"
        before = compose.read_bytes()
        rc, out, err = self.rkmon("template", "deploy", str(compose))
        self.assertEqual(rc, 4, out + err)
        self.assertEqual(compose.read_bytes(), before)
        self.assertEqual(len(re.findall(r"(?m)^  FAIL ", out)), 2, out)
        self.assertIn(f"{compose} exists", " ".join(out.split()))
        self.assertTrue(
            out.endswith(
                "\n\n" + finished("REFUSED: nothing written", self.state_2_next())
            ),
            out,
        )
        self.assertEqual(self.calls(), [])

    def test_a_file_in_an_earlier_layout_is_refused_and_left_as_it_is(self):
        """`rkmon confirm --settings FILE` reads nothing of a file in an earlier layout, and records nothing: a layout
        1 file is refused with the command that moves it to this layout; a file in the layout before layout 1 (the image's
        settings and the deployment's in one file among them), with a new file to start from, in migrate's words.
        """
        where = (
            "~/" + self.settings.name
        )  # as the reader shows a path under the home folder
        files = (
            (
                "kind: deploy_settings\nschema_version: 1\nrepository:\n  name: acme/widgets\n",
                f"{where} is in layout 1, older than this package's layout 2",
                f"fix: {self.w} migrate --settings {where}",
            ),
            (
                "rkmon_settings: 1\nconfig:\n  repo: acme/widgets\n"
                "  image:\n    base: ubuntu:24.04\n"
                "  rocm:\n    version: 10.1.0-5\n",
                f"{where} is in the layout before layout 1 (it starts with rkmon_settings), which this package does "
                "not read",
                f"fix: start from a new file, {self.w} template deploy NEW_FILE (a build file: {self.w} template "
                "build NEW_FILE), and copy your values into it",
            ),
        )
        for text, found, fix in files:
            with self.subTest(found=found):
                self.settings.write_text(text, encoding="utf-8")
                self.settings.chmod(0o600)
                rc, out, err = self.rkmon(
                    "confirm", "--settings", str(self.settings), "--yes"
                )
                self.assertEqual(rc, 4, out + err)
                said = one_line(out + err)
                self.assertIn(found, said)
                self.assertIn(fix, said)
                self.assertIn("nothing confirmed", said)
                self.assertEqual(self.settings.read_text(encoding="utf-8"), text)
                self.assertEqual(
                    [p.name for p in self.home.iterdir()], [self.settings.name]
                )
                self.assertIsNone(self.confirm_record())

    def test_confirm_shows_the_settings_and_records_them_on_this_host(self):
        """Deploy step 3: with no settings file named or confirmed, the state's refusal (state 2, an
        earlier step); a file with a password in an address is refused, never shown, and nothing recorded; with no
        terminal and no --yes, nothing confirmed (exit 2); with --yes, the summary and its checks, then CONFIRMED and
        the state's next line (the image here, the logins: state 7): this folder's records keep the settings'
        fingerprint and the file's own digest, no value of the file, and nothing else is written. NOTHING TO DO while
        the file and this package are what you confirmed, with the same next line; under --dry-run, a changed file's
        summary, with no question and nothing recorded."""
        # No settings recorded, none named: the state's refusal (state 2), an earlier step.
        rc, out, err = self.rkmon("confirm", "--yes")
        self.assertEqual((rc, err), (3, ""), out)
        self.assertIn(
            "\n"
            + finished(
                "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                self.state_2_next(),
            ),
            out,
        )
        # An address with a user and a password in it, beside a binding's: refused, never shown.
        binding = {
            "provider": "openai_compatible",
            "base_url": "https://models.example.com/v1",
            "model_id": "m",
            "api_key_secret_name": "b1_api_key",
        }
        self.write_settings(bindings={"b1": binding})
        text = self.settings.read_text(encoding="utf-8")
        self.settings.write_text(
            text.replace(
                "https://models.example.com/v1",
                "https://u:SECRET-MARK-x3@models.example.com/v1",
            ),
            encoding="utf-8",
        )
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 4, out + err)
        self.assertNotIn("SECRET-MARK", out + err)
        self.assertIsNone(self.confirm_record())
        self.write_settings(bindings={"b1": binding})
        self.answer(
            [
                *self.image_code(),
                *deploy_folder.secrets_volumes(False),
                [r"image inspect", GOOD, 0],
                [r"rk logins --json", "[]", 0],
            ]
        )
        # no terminal: nothing confirmed
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings))
        self.assertEqual(rc, 2, out + err)
        self.assertIn("no terminal to ask on", err)
        self.assertIsNone(self.confirm_record())
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "container rkmon-test · review automation prototype · merge queue prototype",
            out,
        )
        for heading in (
            "Container",
            "Image",
            "Updates: none",
            "Repository",
            "Host",
            "Monitor",
            "Review automation",
            "Merge queue",
            "Dashboards",
            "Alerts",
            "Ticket creation",
            "Weekly report",
            "Integrations",
            "Model bindings",
            "Logins",
        ):
            self.assertIn(f"\n{heading}", out)
        version = re.escape(LABELS["org.rocke.package.version"])
        self.assertRegex(
            out, rf"(?m)^  tag: +rkmon-test \(on this host: package {version},"
        )
        self.assertIn(
            "PASS  image: rocke-monitor:rkmon-test was built from this package's image recipe",
            out,
        )
        self.assertRegex(out, r"(?m)^  github_cli\.username: +op-login$")
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file",
                    f"next: step 5, login: {self.w} login",
                )
            ),
            out,
        )
        for mark in ("SECRET-MARK", "SECRET_MARK"):
            self.assertNotIn(mark, out + err)
        # The records: the settings' fingerprint and the file's own digest, no value of the file; nothing else written.
        doc = settings_file.read(self.settings, file_types.DEPLOY)
        version, code = image.package_version(self.pkg), image.code_sha256(self.pkg)
        c = self.confirm_record()
        self.assertEqual(
            (c.fingerprint, c.file_sha256, c.package_version, c.code_sha256),
            (
                deploy_plan.fingerprint(doc, version, code),
                deploy_plan.file_sha256(doc),
                version,
                code,
            ),
        )
        kept = (self.pkg / records.FILE).read_bytes()
        self.assertNotIn(b"op-login", kept)
        self.assertFalse((self.pkg / deliver.ENV_FILE).exists())
        # The file and this package as you confirmed them: NOTHING TO DO, and the records as they were.
        rc, out, err = self.rkmon("confirm", "--yes")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertRegex(
            one_line(out),
            r"NOTHING TO DO: ~/rkmon-deploy\.yaml and this package are what you confirmed at "
            rf"\d{{4}}-\d\d-\d\d \d\d:\d\d UTC next: step 5, login: {re.escape(self.w)} login$",
        )
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)
        # Under --dry-run, the file changed since: its summary, no question, nothing recorded.
        self.write_settings(bindings={"b1": binding}, memory_gib=13)
        rc, out, err = self.rkmon("--dry-run", "confirm")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("container rkmon-test · review automation prototype", out)
        self.assertNotIn("Are these settings right", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: ~/rkmon-deploy.yaml can be confirmed; nothing recorded",
                    f"next: {self.w} confirm",
                )
            ),
            out,
        )
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)

    def test_a_key_written_out_at_its_default_is_confirmed_again(self):
        """NOTHING TO DO needs the file's own values as confirmed too, which state 5 reads: an edit that writes a
        key out at its default changes your file but not the settings it makes, and the confirm shows it and records
        the file's new digest, with the same fingerprint; then NOTHING TO DO."""
        self.confirmed()
        before = self.confirm_record()
        # features.alerts.message_cooldown_minutes, which the file left out, written out at its default
        text = self.settings.read_text(encoding="utf-8")
        self.assertEqual(text.count("\n  alerts:\n"), 1, text)
        self.settings.write_text(
            text.replace(
                "\n  alerts:\n", "\n  alerts:\n    message_cooldown_minutes: 30\n"
            ),
            encoding="utf-8",
        )
        doc = settings_file.read(self.settings, file_types.DEPLOY)
        version, code = image.package_version(self.pkg), image.code_sha256(self.pkg)
        self.assertEqual(
            deploy_plan.fingerprint(doc, version, code), before.fingerprint
        )
        self.assertNotEqual(deploy_plan.file_sha256(doc), before.file_sha256)
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("confirm", "--yes")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file", out
        )
        after = self.confirm_record()
        self.assertEqual(
            (after.fingerprint, after.file_sha256),
            (before.fingerprint, deploy_plan.file_sha256(doc)),
        )
        rc, out, err = self.rkmon("confirm", "--yes")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("NOTHING TO DO: ", out)

    def test_confirm_says_who_can_reach_a_served_team_dashboard(self):
        """While this host serves the team dashboard, the first confirm's summary says who can reach it: after the
        check rows, before the outcome. With the team dashboard off, no such line."""
        self.answer([[r"image inspect", "", 1]])
        team = sandbox.free_port()
        served = {"enabled": True, "mode": "live", "delivery": "serve"}
        served["serve"] = {"address": "0.0.0.0", "port": team}
        served["public_url"] = "https://dashboards.example.com/my-team/"
        self.write_settings(features={"dashboards": {"team": served}})
        rc, out, err = self.rkmon(
            "--dry-run", "confirm", "--settings", str(self.settings)
        )
        self.assertEqual((rc, err), (0, ""), out)
        line = f"\nnote: anyone who reaches 0.0.0.0:{team} can read the team dashboard and send a priority request\n"
        outcome = "PASSED: ~/rkmon-deploy.yaml can be confirmed; nothing recorded"
        self.assertIn(line, out)
        self.assertLess(out.index("\nThe checks:\n"), out.index(line))
        self.assertLess(out.index(line), out.index(outcome))
        self.write_settings()
        rc, out, err = self.rkmon(
            "--dry-run", "confirm", "--settings", str(self.settings)
        )
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(outcome, out)
        self.assertNotIn("anyone who reaches", out)

    def test_check_and_confirm_say_the_team_dashboard_in_prototype_is_this_host_s_only(
        self,
    ):
        """Check 7: while the team dashboard is served in prototype, its page is served on 127.0.0.1 only,
        whatever serve.address says, and a row of check 7's own says so at rkmon check and at rkmon confirm, beside the
        other rows of the checks across keys: here check 12's WARN, as the package's repository defaults (invented
        ones) are another repository's.
        """
        shutil.copytree(HERE / "data" / "repository", self.pkg / "repository")
        self.answer(
            [
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
                [r"image inspect", "", 1],
            ]
        )
        served = {"enabled": True, "mode": "prototype", "delivery": "serve"}
        served["priority_requests"] = {"enabled": False}
        served["serve"] = {"address": "0.0.0.0", "port": sandbox.free_port()}
        served["public_url"] = "https://dashboards.example.com/widgets/"
        self.write_settings(features={"dashboards": {"team": served}})
        row = (
            'PASS team dashboard: in prototype, its page is served on 127.0.0.1 only, not on "0.0.0.0" '
            "(features.dashboards.team.serve.address), until it is live"
        )
        for args in (("check",), ("--dry-run", "confirm")):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args, "--settings", str(self.settings))
                self.assertEqual((rc, err), (0, ""), out)
                self.assertIn("WARN repository defaults: ", self.flat(out))
                self.assertIn(row, self.flat(out))

    def test_check_16_reads_only_this_deployment_s_own_secrets_volume(self):
        """Check 16, at confirm, while this deployment's secrets keep an older package's Claude settings: a
        one-off container of the image reads their shape, never a value, with no network, removed after the run, and
        this deployment's own secrets volume alone, mounted read-only and never filled from the image. Before that,
        docker is asked only listings filtered on the file's container.name, by its compose project label, and no
        other call names the volume: with no such volume, as on a fresh deployment, one another compose project made,
        or the name another folder's, no container runs and no row shows. A SKIP while the image is not here, or docker
        could not run the container, that names what to do before confirm runs again."""
        volume = f"{deploy_folder.NAME}_{container_layout.SECRETS_VOLUME}"
        label = "com.docker.compose.project"
        project_volumes = rf"^volume ls --filter label={re.escape(label)}={re.escape(deploy_folder.NAME)} "
        listed = [project_volumes, f"{deploy_folder.NAME}_rkmon-work\n{volume}", 0]
        # a volume of that name that another compose project made: docker lists it by its name, never by this label
        made = [
            r"^volume ls --filter name=",
            json.dumps({"Name": volume, "Labels": f"{label}=another"}),
            0,
        ]
        another_folder = {
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": "/srv/other/docker",
        }
        mount = (
            f"--mount type=volume,source={volume},target={container_layout.SECRETS_DIR},"
            "readonly,volume-nocopy "
        )
        shape = claude_env_shape.Shape(
            variables=("ANTHROPIC_AUTH_TOKEN", *claude_env_shape.MODEL_VARS)
        )
        said = json.dumps(
            {"shape": claude_env_shape.to_json(shape), "move": "", "login": False}
        )
        reader = [re.escape(mount), said, 0]
        here = [*self.image_code(), [r"image inspect", GOOD, 0]]
        ref = "rocke-monitor:rkmon-test"
        fail = (
            "FAIL Claude Code: claude_env sends the key as a bearer token, and integrations.claude_code."
            'api_key_sent_as says "header" fix: set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml '
            'to "bearer" PASS Claude Code: claude_env sets ANTHROPIC_MODEL, '
        )
        skip = (
            "SKIP Claude Code: this deployment's secrets are not read for claude_env, an older package's Claude "
            "settings: "
        )
        failing = "docker: Error response from daemon: no space left on device"
        for case, answers, env, code, shown in (
            (
                "no volume",
                [
                    *here,
                    [project_volumes, f"{deploy_folder.NAME}_rkmon-work", 0],
                    reader,
                ],
                {},
                0,
                "",
            ),
            ("another compose project made it", [*here, made, reader], {}, 0, ""),
            (
                "the name is another folder's",
                [*here, listed, reader],
                another_folder,
                4,
                "",
            ),
            ("its own", [*here, listed, reader], {}, 4, fail),
            (
                "no image here",
                [[r"image inspect", "", 1], listed, reader],
                {},
                0,
                f"{skip}{ref} is not on this host; load the image (step 4), then run {self.w} confirm again",
            ),
            (
                "no answer",
                [*here, listed, [re.escape(mount), failing, 125]],
                {},
                0,
                f"{skip}docker could not run a one-off container of {ref} ({failing}); fix what docker says, then run "
                f"{self.w} confirm again",
            ),
        ):
            with self.subTest(case=case):
                self.answer([*answers, [r"rk logins --json", "[]", 0]])
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon(
                    "--dry-run", "confirm", "--settings", str(self.settings), **env
                )
                self.assertEqual((rc, err), (code, ""), out)
                if shown:
                    self.assertIn(shown, one_line(out))
                else:
                    self.assertNotRegex(out, r"(?m)^  [A-Z]+ +Claude Code: ")
                # Each call the stand-in docker logs starts a line with its folder and a tab; the probe's lines have
                # none.
                calls = re.split(
                    r"\n(?=[^\n\t]*\t)", self.log.read_text(encoding="utf-8")
                )
                named = [c.split("\t", 1)[1] for c in calls if volume in c]
                runs = [c for c in named if c.startswith("run ")]
                self.assertEqual(named, runs)
                self.assertEqual(len(runs), int(case in ("its own", "no answer")))
                for c in runs:
                    self.assertTrue(
                        c.startswith(
                            "run --rm --network none "
                            + " ".join(container_layout.RUN_LIMITS)
                            + f" {mount}--entrypoint python3 {ref} -I -c "
                        ),
                        c,
                    )
                    self.assertEqual(c.count("--mount"), 1, c)
                    self.assertRegex(
                        c,
                        rf" {container_layout.IMAGE_CODE} {container_layout.SECRETS_DIR}/claude_env "
                        rf"{container_layout.SECRETS_DIR}/[a-z_]+$",
                    )

    def test_check_16_shows_when_there_is_nothing_to_confirm(self):
        """Check 16's SKIP while the image is not here asks for the confirm again once it is: a confirm of the file and
        this package as confirmed, NOTHING TO DO, reads check 16 again and shows its rows but its PASSes, the records
        as they were: its SKIP while the image is still not here; once it is, its FAIL, and the confirm is FAILED
        (exit 4), with this command again."""
        volume = f"{deploy_folder.NAME}_{container_layout.SECRETS_VOLUME}"
        label = "com.docker.compose.project"
        project_volumes = rf"^volume ls --filter label={re.escape(label)}={re.escape(deploy_folder.NAME)} "
        mount = (
            f"--mount type=volume,source={volume},target={container_layout.SECRETS_DIR},"
            "readonly,volume-nocopy "
        )
        shape = claude_env_shape.Shape(
            variables=("ANTHROPIC_AUTH_TOKEN", *claude_env_shape.MODEL_VARS)
        )
        said = json.dumps(
            {"shape": claude_env_shape.to_json(shape), "move": "", "login": False}
        )
        rest = [
            [project_volumes, f"{deploy_folder.NAME}_rkmon-work\n{volume}", 0],
            [re.escape(mount), said, 0],
            [r"rk logins --json", "[]", 0],
        ]
        skip = (
            "SKIP Claude Code: this deployment's secrets are not read for claude_env, an older package's Claude "
            "settings: rocke-monitor:rkmon-test is not on this host; load the image (step 4), then run "
            f"{self.w} confirm again"
        )
        self.answer([[r"image inspect", "", 1], *rest])
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(skip, one_line(out))
        self.assertIn("CONFIRMED: ", out)
        kept = (self.pkg / records.FILE).read_bytes()
        rc, out, err = self.rkmon("confirm")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            f"{skip} NOTHING TO DO: ~/rkmon-deploy.yaml and this package are what you confirmed at ",
            one_line(out),
        )
        self.answer([*self.image_code(), [r"image inspect", GOOD, 0], *rest])
        rc, out, err = self.rkmon("confirm")
        self.assertEqual((rc, err), (4, ""), out)
        fail = (
            "FAIL Claude Code: claude_env sends the key as a bearer token, and integrations.claude_code."
            'api_key_sent_as says "header" fix: set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml '
            'to "bearer"'
        )
        self.assertRegex(
            one_line(out),
            re.escape(fail)
            + r" FAILED: 1 of \d+ checks failed; nothing confirmed \(fix each FAIL above\) next: "
            + re.escape(f"{self.w} confirm")
            + "$",
        )
        self.assertNotIn("PASS Claude Code", one_line(out))
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)

    def test_confirm_calls_each_model_from_the_deployment_s_network(self):
        """Models are called as the container will reach them: with the image here and the deployment's
        network (the one docker compose makes for its project), from a one-off container of the image on that network,
        which mounts nothing; without either, or with a network of another project, a SKIP that names step 7, where
        the doctor calls it from the container. A model that does not answer is a WARN, and the confirm goes on.
        """
        url = "https://models.example.com/v1"
        binding = {
            "provider": "openai_compatible",
            "base_url": url,
            "model_id": "m",
            "api_key_secret_name": "b1_api_key",
        }
        self.write_settings(
            bindings={"b1": binding},
            features={
                "review_automation": {"passes": {"first_review_pass": {"model": "b1"}}}
            },
        )
        typed = ("confirm", "--settings", str(self.settings), "--yes")
        here = [
            *self.image_code(),
            [r"image inspect", GOOD, 0],
            [r"rk logins --json", "[]", 0],
        ]
        network = r"network inspect .* rkmon-test_default$"
        probe = r"run --rm --network rkmon-test_default "
        opus = "SKIP model opus: checked at step 7, from the container (it needs the claude login, step 5)"

        def run(answers) -> str:
            shutil.rmtree(self.pkg / records.FOLDER, ignore_errors=True)
            self.log.write_text("", encoding="utf-8")
            self.answer(answers)
            rc, out, err = self.rkmon(*typed)
            self.assertEqual(rc, 0, out + err)
            self.assertIn(opus, one_line(out))
            return one_line(out)

        for case, answers in (
            ("no image here", [[r"image inspect", "", 1]]),
            ("no network of the deployment", here),
            ("another project's network", [[network, "other", 0], *here]),
        ):
            with self.subTest(case=case):
                out = run(answers)
                self.assertIn(
                    "SKIP model b1: checked at step 7, from the container", out
                )
                self.assertNotIn("--network rkmon-test_default", self.log.read_text())
        out = run(
            [[network, "rkmon-test", 0], [probe, json.dumps({url: ""}), 0], *here]
        )
        self.assertIn(
            "PASS model b1: answers, called from a one-off container on this deployment's network",
            out,
        )
        [called] = [
            c for c in self.calls() if "\trun --rm --network rkmon-test_default " in c
        ]
        self.assertIn("rocke-monitor:rkmon-test -I -c ", called)
        self.assertTrue(called.rstrip().endswith(f" {url}"), called)
        for mounted in (" -v ", "--volume", "--mount", "--env-file"):
            self.assertNotIn(mounted, called)
        out = run(
            [
                [network, "rkmon-test", 0],
                [probe, json.dumps({url: "timed out"}), 0],
                *here,
            ]
        )
        self.assertIn(
            f"WARN model b1: no answer at {url}, called from a one-off container on this deployment's network "
            "(timed out)",
            out,
        )
        self.assertIn("CONFIRMED: ", out)

    def test_a_new_container_name_while_this_folder_s_container_exists(self):
        """Check 14: while this folder's container exists, a confirm of a new container.name is refused, for it
        would start a second, empty deployment: REFUSED, the fix that sets the name back, exit 4, and nothing
        recorded. With no container of this folder's, the new name is confirmed."""
        self.confirmed()
        text = self.settings.read_text(encoding="utf-8")
        self.assertEqual(text.count(f'name: "{deploy_folder.NAME}"\n'), 1, text)
        self.settings.write_text(
            text.replace(f'name: "{deploy_folder.NAME}"\n', 'name: "rkmon-new"\n'),
            encoding="utf-8",
        )
        kept = (self.pkg / records.FILE).read_bytes()
        typed = ("confirm", "--settings", str(self.settings), "--yes")
        self.answer(
            [
                [r"image inspect", "", 1],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            ]
        )
        ours = {
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }
        rc, out, err = self.rkmon(*typed, **ours)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(
            one_line(out).endswith(
                "== confirm: what changes on this deployment REFUSED: this package folder runs the deployment "
                "rkmon-test; a new container.name would start a second, empty one; nothing changed fix: set "
                'container.name in ~/rkmon-deploy.yaml to "rkmon-test"; for a second deployment on this host, see '
                f'QUICK_DEPLOY_GUIDE.md, "Several deployments" next: {self.w} {" ".join(typed)}'
            ),
            out,
        )
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)
        # No container of this folder's: the new name is this deployment's.
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon(*typed)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(self.confirm_record().container_name, "rkmon-new")

    def test_confirm_asks_no_container_before_this_folder_s_own_exists(self):
        """Confirm asks rk in a container only while this folder's own container exists (confirm writes only the
        host's records; before the first start, a one-off container would make the deployment's network and volumes,
        under the name the last confirm recorded). With the image here, a confirm of a changed file before the first
        start shows the logins from the settings alone and runs no one-off container, nor does the check. In state 1,
        with the image here, a new container.name in your file, the state's own fix, is confirmed; its next line is
        the state's, as rkmon next gives it, whose compose questions are read-only and come once the new name is known
        to be no other deployment's: nothing is made, and no one-off container runs."""
        self.confirmed()
        self.answer(
            [
                *self.image_code(),
                [r"image inspect", GOOD, 0],
                [r"rk logins --json", "[]", 0],
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
            ]
        )
        self.write_settings(memory_gib=13)
        self.log.write_text("", encoding="utf-8")
        rc, out, err = self.rkmon("confirm", "--yes")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("(checked at step 5, login)", out)
        self.assertIn(
            "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file", out
        )
        self.assertNotIn("\tcompose run ", self.log.read_text(encoding="utf-8"))
        rc, out, err = self.rkmon("check")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertNotIn("\tcompose run ", self.log.read_text(encoding="utf-8"))
        # State 1: the container name the last confirm recorded is another deployment's, and your file names another.
        other = {
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": "/srv/other/docker",
        }
        text = self.settings.read_text(encoding="utf-8")
        self.settings.write_text(
            text.replace(f'name: "{deploy_folder.NAME}"\n', 'name: "rkmon-new"\n'),
            encoding="utf-8",
        )
        self.log.write_text("", encoding="utf-8")
        rc, out, err = self.rkmon("confirm", "--yes", **other)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file", out
        )
        self.assertEqual(self.confirm_record().container_name, "rkmon-new")
        calls = self.calls()
        compose = [i for i, c in enumerate(calls) if "\tcompose " in c]
        self.assertEqual(
            [
                calls[i]
                for i in compose
                if MAKES.search(calls[i]) or "\tcompose run " in calls[i]
            ],
            [],
        )
        owner = "\tps --all --filter label=com.docker.compose.project=rkmon-new "
        self.assertLess(
            next(i for i, c in enumerate(calls) if owner in c), compose[0], calls
        )

    def test_later_confirm_says_who_can_reach_the_team_dashboard_a_change_serves(self):
        """After the first start, the exposure sentence follows the changes when a change serves the team dashboard,
        or moves the address or port it is served on: what runs, as docker/.env has it, serves it elsewhere or
        not at all. Not while it stays where it is served."""
        self.confirmed()
        running = self.delivered()  # what runs: the team dashboard off
        asked = [
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"rk stopped$", "", 1],
            [r"rk logins --json", "[]", 0],
        ]

        def confirm(port, dry=True, memory_gib=12) -> str:
            served = {"enabled": True, "mode": "live", "delivery": "serve"}
            served["serve"] = {"address": "0.0.0.0", "port": port}
            served["public_url"] = "https://dashboards.example.com/my-team/"
            self.write_settings(
                features={"dashboards": {"team": served}}, memory_gib=memory_gib
            )
            self.answer([*running, *asked])
            typed = ("--dry-run", "confirm") if dry else ("confirm", "--yes")
            rc, out, err = self.rkmon(*typed, **self.running())
            self.assertEqual((rc, err), (0, ""), out)
            return out

        def note(port) -> str:
            at = f"0.0.0.0:{port}"
            return f"\nnote: anyone who reaches {at} can read the team dashboard and send a priority request\n"

        first, second = sandbox.free_port(), sandbox.free_port()
        out = confirm(first)  # a change serves it
        self.assertIn(note(first), out)
        self.assertLess(
            out.index("\nPackage defaults that changed: "), out.index(note(first))
        )
        self.assertLess(out.index(note(first)), out.index(" apply will: "))
        self.assertIn("\nCONFIRMED: ", confirm(first, dry=False))
        running = self.delivered()  # what runs: the team dashboard served at FIRST
        out = confirm(first, memory_gib=13)  # it stays where it is served
        self.assertIn("\n  ~ host.resources.memory_gib  12 -> 13\n", out)
        self.assertNotIn("anyone who reaches", out)
        self.assertIn(note(second), confirm(second))  # a change moves it

    def test_confirm_after_the_first_start_shows_only_what_changes(self):
        """After the first start, confirm shows only what changes: each change as what runs now -> what apply
        makes it, with how it applies; docker/.env edited by hand, variable by variable, under the deploy key it is
        made from; the count, and what apply will do; then the checks of what changes alone (the model a change
        brings, called from the deployment's network; no image row while the image stays). --dry-run records nothing;
        the confirm records the count, which the state's sentence names once the hand edit of docker/.env, which comes
        first (state 9), is gone; a second confirm has nothing to do.
        """
        self.confirmed()
        delivered = self.delivered()  # what runs: what the first confirm made
        url = "https://fast.example.com/v1"
        binding = {
            "provider": "openai_compatible",
            "base_url": url,
            "model_id": "m",
            "api_key_secret_name": "fast_api_key",
        }
        self.write_settings(
            memory_gib=13,
            bindings={"fast": binding},
            features={
                "review_automation": {
                    "passes": {"first_review_pass": {"model": "fast"}}
                }
            },
        )
        dotenv = self.pkg / deliver.ENV_FILE
        ran = dotenv.read_text(encoding="utf-8")
        self.assertIn(f"\nRKMON_PORT={self.port}\n", ran)
        edited = ran.replace(f"\nRKMON_PORT={self.port}\n", "\nRKMON_PORT=18620\n")
        dotenv.write_text(edited, encoding="utf-8")
        self.answer(
            [
                *delivered,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [r"network inspect .* rkmon-test_default$", "rkmon-test", 0],
                [r"run --rm --network rkmon-test_default ", json.dumps({url: ""}), 0],
                [r"rk logins --json", "[]", 0],
            ]
        )
        typed = ("confirm", "--settings", str(self.settings))
        kept = (self.pkg / records.FILE).read_bytes()
        rc, out, err = self.rkmon(*typed, "--dry-run", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        head, _, rest = out.partition("\n\n")
        self.assertRegex(
            head,
            r"^== confirm: what changes on this deployment \(dry run: nothing changes\)\n"
            r"your deploy settings file: ~/rkmon-deploy\.yaml \(changed \d{4}-\d\d-\d\d \d\d:\d\d UTC\)$",
        )
        changes = [
            "Your changes:",
            "  ~ host.resources.memory_gib  12 -> 13",
            "        restart: the work in flight finishes first",
            '  ~ features.review_automation.passes.first_review_pass.model  "default" -> "fast"',
            "        restart: the work in flight finishes first",
            "  + model_bindings.fast",
            "        restart: the work in flight finishes first",
            "Changed outside your file (apply puts your file's value back):",
            f"  ~ features.dashboards.admin.port, as docker/.env has it  18620 -> {self.port}",
            "        restart",
            "Deploy keys you left blank whose default changed: none",
            "Package defaults that changed: none",
            "",
            "4 changes. apply will: let the work in flight finish (up to 60 minutes), write the new settings (the "
            "previous ones",
            "kept), make the container again, run the doctor, and put the previous settings back on a new failure.",
            "",
            "The checks:",
        ]
        self.assertEqual(rest.splitlines()[: len(changes)], changes)
        rows = one_line(rest.partition("\nThe checks:\n")[2])
        # the checks of what changes: the model the change brings alone, and no image row while the image stays
        self.assertIn(
            "PASS model fast: answers, called from a one-off container on this deployment's network",
            rows,
        )
        self.assertNotIn("model opus", rows)
        self.assertNotRegex(out, r"(?m)^  [A-Z]{4}  image: ")
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: 4 changes can be confirmed; nothing recorded",
                    f"next: {self.w} {' '.join(typed)}",
                )
            ),
            out,
        )
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)
        rc, out, err = self.rkmon(*typed, "--yes", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        put_back = f"next: {self.w} apply (--dry-run shows what it puts back)"
        self.assertTrue(
            out.endswith(
                finished(
                    "CONFIRMED: 4 changes wait for apply; nothing runs differently yet",
                    put_back,
                )
            ),
            out,
        )
        self.assertEqual(self.confirm_record().changes, 4)
        self.assertEqual(
            dotenv.read_text(encoding="utf-8"), edited
        )  # nothing runs differently
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        edit_note = f"note: changed outside your file: docker/.env differs from what {self.w} last wrote there"
        self.assertTrue(one_line(out).endswith(f"{edit_note} {put_back}"), out)
        dotenv.write_text(ran, encoding="utf-8")
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        # The confirm's row says the change that waits, which no note says again.
        self.assertRegex(
            one_line(out),
            r" 3\. DONE confirm: a change waits: you confirmed 4 changes at \d{4}-\d\d-\d\d \d\d:\d\d UTC, and the "
            r"monitor runs the settings before them 4\. ",
        )
        self.assertNotIn("\nnote: ", out)
        self.assertTrue(out.endswith(rewrapped(f"\nnext: {self.w} apply\n")), out)
        rc, out, err = self.rkmon(*typed, "--yes", **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        # The file's WARN shows above it, as at each confirm.
        self.assertRegex(
            out,
            r"^== confirm: what changes on this deployment\n(?:  WARN  [^\n]+\n(?: {8}[^\n]*\n)*)+\n"
            r"NOTHING TO DO: ~/rkmon-deploy\.yaml and this package are what you confirmed at [^\n]+ UTC\n"
            + re.escape(rewrapped(f"next: {self.w} apply\n"))
            + "$",
        )
        self.assertIn(f"WARN {deployment.RAM_DISK_ROW}: ", one_line(out))
        self.assertEqual(self.confirm_record().changes, 4)

    def test_confirm_names_the_web_host_install_under_a_change_its_service_is_made_from(
        self,
    ):
        """With the team dashboard published and its priority box on, the web host needs the request service: a change
        of a key that service is made from (its site folder) says, under how it applies, that the service takes it
        when you run rkmon web-host install after apply; a change of another key does not.
        """
        team = {
            **PUBLISHED,
            "public_url": "http://dashboards.example.com:8080/",
            "priority_requests": {"enabled": True},
        }
        self.write_settings(features={"dashboards": {"team": team}})
        self.confirmed()
        delivered = self.delivered()
        moved = {
            **team,
            "publish": {**PUBLISHED["publish"], "site_folder": "pages/gadgets"},
        }
        self.write_settings(memory_gib=13, features={"dashboards": {"team": moved}})
        self.answer(
            [
                *delivered,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [r"rk logins --json", "[]", 0],
            ]
        )
        typed = ("confirm", "--settings", str(self.settings), "--dry-run")
        rc, out, err = self.rkmon(*typed, **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        restart = "        restart: the work in flight finishes first"
        changes = [
            "Your changes:",
            "  ~ host.resources.memory_gib  12 -> 13",
            restart,
            '  ~ features.dashboards.team.publish.site_folder  "public_html/widgets" -> "pages/gadgets"',
            restart,
            f"        web host: the request service there takes it when you run {self.w} web-host install after "
            "apply",
        ]
        self.assertEqual(out.partition("\n\n")[2].splitlines()[: len(changes)], changes)

    def test_confirm_after_the_first_start_without_what_runs_shows_every_setting(self):
        """When what runs cannot be read, the confirm says why and shows every setting, as before the first start; it
        records no count. The layer that cannot be read is not what rkmon last wrote there, state 9's; with no
        fingerprint kept (records an older package left), the state's sentence names new settings. While the monitor
        does not say which settings it runs, the confirm shows and records nothing: rk's refusal as rk says it, then
        the confirm again.
        """
        self.confirmed()
        _layer, applied = self.delivered()  # docker/.env and the override in place
        self.write_settings(memory_gib=13)
        layer_file = f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}"
        answers = [
            *self.image_code(),
            [re.escape(layer_file) + "$", f"cat: {layer_file}: No such file", 1],
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"rk stopped$", "", 1],
            [r"rk logins --json", "[]", 0],
        ]
        typed = ("confirm", "--settings", str(self.settings), "--yes")
        refusal = "REFUSED: the applied record cannot be read\nfix: rk applied --load"
        self.answer([[r"rk applied --json", refusal, 4], *answers])
        kept = (self.pkg / records.FILE).read_bytes()
        rc, out, err = self.rkmon(*typed, **self.running())
        self.assertEqual((rc, err), (4, ""), out)
        self.assertEqual(
            out,
            f"== confirm: what changes on this deployment\n{refusal}\n"
            + finished(f"next: {self.w} {' '.join(typed)}"),
        )
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)
        self.answer([applied, *answers])
        rc, out, err = self.rkmon(*typed, **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "\nnote: what changes cannot be shown (the settings volume has no settings that can be read): every "
            "setting is shown\n",
            out,
        )
        self.assertRegex(
            out, r"\n  resources\.memory_gib: +13\n"
        )  # the summary's host section
        self.assertTrue(
            out.endswith(
                finished(
                    "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file",
                    f"next: {self.w} apply (--dry-run shows what it puts back)",
                )
            ),
            out,
        )
        self.assertEqual(self.confirm_record().changes, 0)
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            f"note: changed outside your file: the settings volume's deploy.json differs from what {self.w} last "
            "wrote there",
            one_line(out),
        )
        path = self.pkg / records.FILE
        older = json.loads(path.read_text(encoding="utf-8"))
        del older["written"]
        path.write_text(json.dumps(older), encoding="utf-8")
        rc, out, err = self.rkmon("next", **self.running())
        self.assertIn(
            "3. DONE confirm: a change waits: you confirmed new settings at ",
            one_line(out),
        )

    def test_confirm_after_a_package_update_names_the_package_first(self):
        """After a package update, confirm names the package first, as a restart, then the defaults that changed
        (the package before it in the folder beside this one, `<folder>.prev`, made what runs); the image row checks
        that the image suits this package; a login a change needs that is not set yet is a SKIP row that names the
        command that sets it."""
        self.confirmed()
        prev = deploy_folder.copy(self.d / "pkg.prev")
        (prev / "VERSION").write_text("2026.01.01\n", encoding="utf-8")
        defaults = json.loads(
            (prev / "config.defaults.json").read_text(encoding="utf-8")
        )
        drain = defaults["update"]["drain_timeout_min"]
        defaults["update"]["drain_timeout_min"] = drain + 30
        (prev / "config.defaults.json").write_text(
            json.dumps(defaults), encoding="utf-8"
        )
        then = deploy_plan.plan(
            settings_file.read(self.settings, file_types.DEPLOY),
            "2026.01.01",
            image.code_sha256(prev),
            deploy_plan.Host(os.getuid(), os.getgid(), self.confirm_record().at),
        )
        env, override, layer = deliver.texts(then)
        for rel, text in ((deliver.ENV_FILE, env), (deliver.OVERRIDE_FILE, override)):
            if text is not None:
                (self.pkg / rel).write_text(text, encoding="utf-8")
                (self.pkg / rel).chmod(0o600)
        layer_file = f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}"
        applied = {"applied": {"fingerprint": then.fingerprint}, "paused": False}
        self.write_settings(features={"review_automation": {"mode": "live"}})
        logins = [
            {"name": "github_write", "state": "NOT NEEDED", "what": "the write login"}
        ]
        self.answer(
            [
                *self.image_code(),
                [re.escape(layer_file) + "$", layer, 0],
                [r"rk applied --json", json.dumps(applied), 0],
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [r"rk logins --json", json.dumps(logins), 0],
            ]
        )
        typed = ("confirm", "--settings", str(self.settings), "--dry-run")
        rc, out, err = self.rkmon(*typed, **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        now = image.package_version(self.pkg)
        changes = [
            f"This package: 2026.01.01 -> {now}",
            "        restart: the work in flight finishes first",
            "Your changes:",
            '  ~ features.review_automation.mode  "prototype" -> "live"',
            "        restart: the work in flight finishes first",
            "Deploy keys you left blank whose default changed: none",
            "Package defaults that changed, for your information (no key of your file sets them):",
            f"  ~ update.drain_timeout_min  {drain + 30} -> {drain}",
            "",
        ]
        self.assertEqual(out.split("\n\n", 1)[1].splitlines()[: len(changes)], changes)
        rows = one_line(out.partition("\nThe checks:\n")[2])
        self.assertIn(
            "PASS image: rocke-monitor:rkmon-test was built from this package's image recipe",
            rows,
        )
        self.assertIn(
            f"SKIP github_write: not set yet; {self.w} login sets it before apply", rows
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: 2 changes can be confirmed; nothing recorded",
                    f"next: {self.w} {' '.join(typed[:-1])}",
                )
            ),
            out,
        )

    def confirm_dry_run(self, before: list) -> list:
        """confirm --dry-run on the running container, which runs what BEFORE (delivered) delivered: its lines from
        those that say what changes on."""
        self.answer(
            [
                *before,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [r"rk logins --json", "[]", 0],
            ]
        )
        typed = ("confirm", "--settings", str(self.settings), "--dry-run")
        rc, out, err = self.rkmon(*typed, **self.running())
        self.assertEqual((rc, err), (0, ""), out)
        return out.split("\n\n", 1)[1].splitlines()

    def test_other_code_under_the_same_version_is_confirmed_then_applied(self):
        """A package update to a later commit of updates.branch: a module changed, VERSION kept. The package changed
        (state 5): apply refuses, naming the confirm; confirm names the code change, a restart; once it
        is confirmed, apply puts the new code in (code-install, the module as it is now in its archive) and makes the
        container again."""
        self.confirmed()
        before = self.delivered()
        was = deploy_folder.plan(self.pkg, self.settings)
        module = self.pkg / "rkmon_lib" / "drain.py"
        later = module.read_text(encoding="utf-8") + "# a later commit\n"
        module.write_text(later, encoding="utf-8")
        self.answer([[r"image inspect", GOOD, 0]])
        rc, out, err = self.rkmon("apply", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: this package changed after the last confirm; nothing changed",
                    f"next: step 3, confirm: {self.w} confirm",
                )
            ),
            out,
        )
        now = image.package_version(self.pkg)
        self.assertEqual(
            self.confirm_dry_run(before)[:5],
            [
                f"This package: {now} -> {now} (other code under the same version)",
                "        restart: the work in flight finishes first",
                "Your changes: none",
                "Deploy keys you left blank whose default changed: none",
                "Package defaults that changed: none",
            ],
        )
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        self.assertNotEqual(made.fingerprint, was.fingerprint)
        self.answer(self.apply_answers(before, was, made))
        self.log.unlink()
        self.stdin.unlink(missing_ok=True)
        rc, out, err = self.rkmon("apply", **self.running())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"\n  code: {INSTALLED}\n", out)
        self.assertIn("\n  restart: docker compose up -d --force-recreate\n", out)
        self.assertIn(
            "\nAPPLIED: the deployment runs your confirmed settings (1 change)\n", out
        )
        data = self.stdin.read_bytes()
        start = (
            data.index(b"ustar") - 257
        )  # the archive's first header (tar's magic is at 257)
        with tarfile.open(fileobj=io.BytesIO(data[start:])) as tar:
            got = tar.extractfile("rkmon_lib/drain.py").read().decode("utf-8")
        self.assertEqual(got, later)

    def test_confirm_after_an_update_to_a_later_commit_reads_the_kept_package(self):
        """After a package update to a later commit under the same version, the package beside this folder
        (`<folder>.prev`), whose version and code made what runs, is the package then: confirm names the code change
        and the package defaults that changed between the two commits."""
        self.confirmed()
        before = self.delivered()
        prev = deploy_folder.copy(self.d / "pkg.prev")
        self.assertEqual(image.code_sha256(prev), self.confirm_record().code_sha256)
        defaults_file = self.pkg / "config.defaults.json"
        defaults = json.loads(defaults_file.read_text(encoding="utf-8"))
        drain = defaults["update"]["drain_timeout_min"]
        defaults["update"]["drain_timeout_min"] = drain + 30
        defaults_file.write_text(json.dumps(defaults), encoding="utf-8")
        now = image.package_version(self.pkg)
        self.assertEqual(
            self.confirm_dry_run(before)[:6],
            [
                f"This package: {now} -> {now} (other code under the same version)",
                "        restart: the work in flight finishes first",
                "Your changes: none",
                "Deploy keys you left blank whose default changed: none",
                "Package defaults that changed, for your information (no key of your file sets them):",
                f"  ~ update.drain_timeout_min  {drain} -> {drain + 30}",
            ],
        )

    def test_a_key_set_for_a_feature_that_is_off_is_a_warn(self):
        """A value set for a feature that is off is a WARN row of the confirm: nothing reads it, and the
        confirm goes on."""
        key = "features.review_automation.remote_gpu_tests"
        doc = settings_file.read(self.settings, file_types.DEPLOY)
        gpu = doc.values["features"]["review_automation"]["remote_gpu_tests"]
        self.assertEqual(gpu, {"enabled": False})  # off, as the templates write it
        gpu["cluster"] = {"partition": "gpu"}
        self.settings.write_text(settings_file.dump(doc), encoding="utf-8")
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"WARN {key}.cluster.partition: set, and {key} is off: nothing reads it fix: delete "
            f"{key}.cluster.partition from ~/rkmon-deploy.yaml, or turn it on: set {key}.enabled in "
            "~/rkmon-deploy.yaml to true",
            one_line(out),
        )
        self.assertIn("CONFIRMED: ", out)

    def test_check_and_confirm_read_the_team_s_skills_folder(self):
        """Check 18 at rkmon check and at rkmon confirm: a link in the team's skills folder the settings
        name is a FAIL that names the key, and the confirm refuses, recording nothing; with the file itself in its
        place, the PASS row names the package's file it replaces, among the check's rows and the confirm's checks.
        """
        verifier = (
            self.home / "team-skills" / "team_members" / "rocke_review_verifier.md"
        )
        verifier.parent.mkdir(parents=True)
        text = (
            self.pkg / "agentic" / "team_members" / "rocke_review_verifier.md"
        ).read_text(encoding="utf-8")
        outside = self.home / "verifier.md"
        outside.write_text(text + "\n## The team's rule\n", encoding="utf-8")
        verifier.symlink_to(outside)
        self.write_settings(
            features={"review_automation": {"skills_folder": "~/team-skills"}}
        )
        failed = (
            "FAIL skills folder: features.review_automation.skills_folder ~/team-skills may contain only Markdown "
            "files, each at team_members/, agents/, review_book.md or skills/rocke-monitor/review_procedure.md, "
            "and it has team_members/rocke_review_verifier.md (a link) fix: take each out of ~/team-skills (a "
            "link: put the file itself in its place), then run"
        )
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check", "--settings", "~/rkmon-deploy.yaml")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(failed, one_line(out))
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon(
            "confirm", "--settings", "~/rkmon-deploy.yaml", "--yes"
        )
        self.assertEqual(rc, 4, out + err)
        self.assertIn(failed, one_line(out))
        self.assertIn("nothing confirmed", out)
        self.assertIsNone(self.confirm_record())
        verifier.unlink()
        outside.rename(verifier)
        passed = (
            "PASS skills folder: the team's skills in ~/team-skills replace the package's "
            "team_members/rocke_review_verifier.md"
        )
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check", "--settings", "~/rkmon-deploy.yaml")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(passed, one_line(out))
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon(
            "confirm", "--settings", "~/rkmon-deploy.yaml", "--yes"
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn(passed, one_line(out.split("\nThe checks:\n", 1)[1]))
        self.assertIsNotNone(self.confirm_record())

    def test_start_and_apply_refuse_settings_not_confirmed_or_changed_since(self):
        """Settings never confirmed, or changed since: the start refuses with the state's sentence and next line (state
        2, then state 5, the confirm again: an earlier step, exit 3), and so does the apply; nothing runs, and next
        says the confirm is to do again."""
        self.answer([[r"image inspect", GOOD, 0]])
        rc, out, err = self.rkmon("start")
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
        self.confirmed()
        # A change that is not in the .env (a feature's mode) changes the settings too.
        live = {"ci_red_label": "ci-red", "stability_label": "stability"}
        self.write_settings(
            features={"merge_queue": {"mode": "live", "develop_guard": live}}
        )
        self.answer([[r"image inspect", GOOD, 0]])
        # Docker gives no start time here, so no tick is known to have run since the start: the step words.
        refused = finished(
            "REFUSED: your settings file changed after the last confirm; nothing changed",
            f"next: step 3, confirm: {self.w} confirm",
        )
        for args in (("start",), ("apply",)):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args, FAKE_DOCKER_RUNNING="1")
                self.assertEqual(rc, 3, out + err)
                self.assertTrue(out.endswith(refused), out)
        self.assertFalse(
            any(
                "rk init" in c or c.endswith("up -d") or "rk stop" in c
                for c in self.calls()
            )
        )
        rc, out, err = self.rkmon("next")
        self.assertIn(
            "  3. REDO  confirm: your settings file changed after the last confirm\n",
            out,
        )
        self.assertIn(rewrapped(f"next: step 3, confirm: {self.w} confirm"), out)

    def test_a_fail_refuses_the_confirm(self):
        """The confirm refuses what the settings alone cannot run, and an image here that was not built from this
        package's image recipe: its own copy of a file an image keeps is not this package's, or its own code
        cannot be read."""
        self.write_settings(
            features={
                "dashboards": {
                    "admin": {"enabled": False},
                    "team": {
                        "enabled": True,
                        "mode": "live",
                        "delivery": "serve",
                        "priority_requests": {"enabled": False},
                    },
                }
            }
        )
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAIL  features: features.dashboards.team is enabled but features.dashboards.admin is not",
            out,
        )
        self.assertIsNone(self.confirm_record())
        self.write_settings()
        for code, found in (
            (
                self.image_code(changed=["docker/Dockerfile"]),
                "rocke-monitor:rkmon-test was not built from this package's image recipe: its copies of "
                "docker/Dockerfile are not this package's",
            ),
            (
                [[deploy_folder.IMAGE_CODE, "exec: python3: not found", 127]],
                "rocke-monitor:rkmon-test: its own code could not be read (exec: python3: not found)",
            ),
        ):
            with self.subTest(found=found):
                self.answer(
                    [*code, [r"image inspect", GOOD, 0], [r"rk logins --json", "[]", 0]]
                )
                rc, out, err = self.rkmon(
                    "confirm", "--settings", str(self.settings), "--yes"
                )
                self.assertEqual(rc, 4, out + err)
                self.assertIn(f"FAIL image: {found}", one_line(out))
                self.assertIn("QUICK_BUILD_GUIDE.md", out)
                self.assertRegex(
                    out,
                    r"\n\nFAILED: 1 of \d+ checks failed; nothing confirmed \(fix each FAIL above\)\n"
                    + re.escape(
                        finished(
                            f"next: {self.w} confirm --settings {self.settings} --yes"
                        )
                    )
                    + "$",
                )
                self.assertIsNone(self.confirm_record())

    def test_confirm_answered_no_is_cancelled(self):
        """On a terminal, a no to the question confirms nothing: CANCELLED, exit 1, and the deployment state's next
        line, state 2's while nothing is confirmed; under --dry-run the question is not asked, and the outcome is
        the real run's. Nothing is recorded."""
        self.answer(
            [
                *self.image_code(),
                [r"image inspect", GOOD, 0],
                [r"rk logins --json", "[]", 0],
            ]
        )
        typed = ("confirm", "--settings", str(self.settings))
        rc, out = sandbox.on_a_terminal(
            [self.w, *typed], self.home, self.env, typed=b"n\n"
        )
        self.assertEqual(rc, 1, out)
        self.assertTrue(
            out.endswith(
                finished("CANCELLED: nothing confirmed", self.state_2_next()).replace(
                    "\n", "\r\n"
                )
            ),
            out,
        )
        self.assertIsNone(self.confirm_record())
        rc, out, err = self.rkmon("--dry-run", *typed)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertNotIn("Are these settings right", out)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "PASSED: ~/rkmon-deploy.yaml can be confirmed; nothing recorded",
                    f"next: {self.w} {' '.join(typed)}",
                )
            ),
            out,
        )
        self.assertIsNone(self.confirm_record())

    def test_secret_values_in_a_settings_file_are_refused_never_shown(self):
        """A settings file never has a secret's value in it: the confirm and the check (through the reader,
        rkmon_lib/settings_file.py) refuse a secret_values section, naming it and how many entries it has, never a
        value; a confirmed file that gained one since is no longer what was confirmed, and `rkmon secret NAME` refuses
        with that state's sentence (state 5), never reading it.
        """
        value = "made-up-" + "hook-Zq7"
        self.write_settings(more=f"secret_values:\n  teams_webhook: {value}\n")
        where = (
            "~/" + self.settings.name
        )  # as the reader shows a path under the home folder
        read = (
            f"FAIL deploy settings file: {where} has a secret_values section (1 entry), and no secret goes in "
            "a settings file"
        )
        for args, outcome in (
            (("confirm", "--settings", str(self.settings), "--yes"), "FAILED: "),
            (("check", "--settings", str(self.settings)), "FAILED: "),
        ):
            with self.subTest(args=args):
                # docker is here, so the check's prerequisites pass and its refusal is the file's (exit 4)
                self.answer(
                    [[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]]
                )
                rc, out, err = self.rkmon(*args)
                self.assertEqual(rc, 4, out + err)
                flat = one_line(out)
                self.assertIn(read, flat)
                self.assertIn(f"delete secret_values from {where}", flat)
                self.assertIn(outcome, flat)
                self.assertNotIn(value, out + err)
        self.assertIsNone(self.confirm_record())
        # A file the confirm recorded and that gained the section since: `rkmon secret NAME`, typed on a terminal,
        # refuses in that state, an earlier step's; nothing goes into the container.
        self.write_settings()
        self.confirmed()
        self.write_settings(more=f"secret_values:\n  teams_webhook: {value}\n")
        self.answer([[r"secret-set", "", 0]])
        rc, out = sandbox.on_a_terminal(
            [self.w, "secret", "teams_webhook"], self.home, self.env
        )
        self.assertEqual(rc, 3, out)
        self.assertEqual(
            one_line(out),
            "REFUSED: your settings file changed after the last confirm; nothing changed next: "
            f"step 3, confirm: {self.w} confirm",
        )
        self.assertNotIn(value, out)
        self.assertFalse([c for c in self.calls() if "secret-set" in c])

    def test_the_check_passes_a_ready_host(self):
        self.answer(
            [
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
                [r"DockerRootDir", str(self.d), 0],
            ]
        )
        # The host's own disk and memory decide those two checks; every other one is the settings' and the fake's.
        self.write_settings(memory_gib=8, ramdisk_gib=0)
        rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertTrue(
            out.startswith(
                "== deploy step 2 of 8, check: check this host and your deploy settings; change nothing\n"
            ),
            out,
        )
        for line in (
            "PASS  docker: server 29.0.0",
            "PASS  docker compose",
            "PASS  your settings file",
            "PASS  ports",
            "PASS  container name: rkmon-test: no other deployment on this host uses it",
        ):
            self.assertIn(line, out)
        self.assertNotIn("buildx", out)  # nothing is built here
        others = [
            x
            for x in out.splitlines()
            if x.lstrip().startswith("FAIL")
            and " disk:" not in x
            and " memory:" not in x
        ]
        self.assertEqual(others, [], out)
        if rc == 0:
            self.assertRegex(out, r"(?m)^PASSED: \d+ checks( \(\d+ warnings?\))?$")
            self.assertIn(
                f"next: step 3, confirm: {self.w} confirm --settings ~/rkmon-deploy.yaml",
                one_line(out),
            )
        self.settings.chmod(0o644)
        rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertEqual(rc, 4)
        self.assertIn(one_line(f"chmod 600 {self.settings}"), one_line(out))
        self.assertRegex(
            out,
            r"(?m)^FAILED: \d+ of \d+ checks failed; nothing changed \(fix each FAIL above\)\n"
            + re.escape(finished(f"next: {self.w} check --settings {self.settings}"))
            + r"\Z",
        )

    def test_check_and_confirm_judge_the_container_s_share_of_this_host(self):
        """rkmon check and rkmon confirm run the deploy file's rules on the container's share of this host
        (coherence.resources, checks 8 and 9), with this host's memory and CPUs, those the tests set
        (deploy_folder.HOST_SIZE): a RAM disk smaller than the starting reviews and builds need is a WARN in both, and
        the confirm records the file; a memory_gib or cpus more than this host has is a FAIL in both, with this
        host's figure, so a file the check fails cannot be confirmed, the records as they were.
        """
        self.answer(
            [
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
                [r"DockerRootDir", str(self.d), 0],
                [r"image inspect", "", 1],
            ]
        )
        warn = (
            r"WARN RAM disk: host\.resources\.ramdisk_gib 1 is less than the starting reviews and builds need "
            r"\(about \d+ GiB\)"
        )
        self.write_settings(ramdisk_gib=1)
        rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertRegex(one_line(out), warn)
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertRegex(one_line(out), warn)
        self.assertIn("CONFIRMED: ", out)
        kept = (self.pkg / records.FILE).read_bytes()
        gib, cpus = deploy_folder.HOST_MEMORY_GIB, deploy_folder.HOST_CPUS
        for settings, fail in (
            (
                {"memory_gib": 4096},
                f"FAIL memory: host.resources.memory_gib 4096 is more than this host has ({gib} GiB)",
            ),
            (
                {"cpus": cpus + 1},
                f"FAIL CPUs: host.resources.cpus {cpus + 1} is more than this host has ({cpus})",
            ),
        ):
            self.write_settings(**settings)
            for args in (("check",), ("confirm", "--yes")):
                with self.subTest(settings=settings, args=args):
                    rc, out, err = self.rkmon(*args, "--settings", str(self.settings))
                    self.assertEqual(rc, 4, out + err)
                    self.assertIn(fail, one_line(out))
        self.assertEqual((self.pkg / records.FILE).read_bytes(), kept)

    def test_the_check_asks_about_the_ports_your_file_makes(self):
        """The check asks about the dashboards' ports docker/.env would publish for your file, as deliver writes them
        (deploy_plan): the team dashboard's too while it is served, a FAIL with the key that sets it while another
        program uses it. While a row of the file itself fails, the ports it makes are not known yet: no ports row.
        """
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        team = sandbox.free_port()
        served = {"enabled": True, "mode": "live", "delivery": "serve"}
        served["serve"] = {"address": "127.0.0.1", "port": team}
        self.write_settings(features={"dashboards": {"team": served}})
        with socket.socket() as s:
            s.bind(("127.0.0.1", team))
            s.listen()
            rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"FAIL port: 127.0.0.1:{team} is in use by another program or container fix: set "
            f"features.dashboards.team.serve.port in {self.settings} to a port that is free on this host",
            self.flat(out),
        )
        self.assertNotIn(f"127.0.0.1:{self.port} is in use", out)
        text = self.settings.read_text(encoding="utf-8")
        self.settings.write_text(text.replace("cpus: 2", 'cpus: "two"'), "utf-8")
        rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertEqual(rc, 4, out + err)
        self.assertIn('"two" is not a number', out)
        self.assertNotRegex(out, r"(?m)^  [A-Z]+  ports?: ")

    def test_the_check_without_docker_is_an_earlier_step(self):
        """docker that does not answer, or its compose missing, is a prerequisite: FAILED with exit 3, the command
        again; with no settings file named or recorded, the same."""
        for answers in (
            [[r"^version", "Cannot connect to the Docker daemon", 1]],
            [[r"^version", "29.0.0", 0], [r"compose version", "", 1]],
        ):
            with self.subTest(answers=answers):
                self.answer(answers)
                rc, out, err = self.rkmon("check", "--settings", str(self.settings))
                self.assertEqual(rc, 3, out + err)
                self.assertRegex(out, r"(?m)^  FAIL  docker( compose)?: ")
                self.assertTrue(
                    out.endswith(
                        "\n"
                        + finished(f"next: {self.w} check --settings {self.settings}")
                    ),
                    out,
                )
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check")
        self.assertEqual(rc, 3, out + err)
        self.assertIn("FAIL  your settings file: none named", out)
        self.assertTrue(out.endswith("\n" + finished(f"next: {self.w} check")), out)

    def test_a_docker_question_takes_none_of_your_input(self):
        """A docker question (ask_docker) runs with no input: with yours a pipe that holds what you typed and stays
        open, the check's docker answers at once (the stand-in reads its input whenever it has one, to its end), and
        nothing you typed reaches it."""
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        with tempfile.TemporaryFile() as said:
            p = subprocess.Popen(
                [self.w, "check"],
                cwd=self.home,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=said,
                stderr=subprocess.STDOUT,
            )
            try:
                p.stdin.write(b"typed\n")
                p.stdin.flush()
                rc = p.wait(timeout=120)
            finally:
                p.stdin.close()
                if p.poll() is None:
                    p.kill()
                    p.wait()
            said.seek(0)
            out = said.read().decode("utf-8", "replace")
        self.assertEqual(rc, 3, out)
        self.assertIn("server 29.0.0", out)
        self.assertFalse(self.stdin.exists())

    def test_the_check_finds_settings_and_secrets_left_in_the_package(self):
        (self.pkg / "docker" / "old-settings.yaml").write_text(
            "kind: deploy_settings\nschema_version: 1\n"
        )
        (self.pkg / "docker" / "teams_webhook").write_text("x")
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check", "--settings", str(self.settings))
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAIL  the package folder: private files inside it: docker/old-settings.yaml, docker/teams_webhook",
            out,
        )
        self.assertIn("move each out of the package folder", out)
        self.assertRegex(out, r"(?m)^FAILED: \d+ of \d+ checks failed")


if __name__ == "__main__":
    unittest.main()
