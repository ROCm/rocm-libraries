# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deploy step 4, the image, end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. The image step loads the image and refuses one that does not suit the settings
or this network's TLS, setting nothing up; the image a load brings is the one checked, and one of another tag than your
confirmed settings name is a settings change. The image's own code is read from a one-off container of it with no
network and no mount. Each image command ends with one outcome line.
"""

import pathlib
import re
import shutil
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
from deploy_path_fixture import GOOD, LABELS, MAKES, DeployPathCase  # noqa: E402
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import (  # noqa: E402
    container_layout,
    deliver,
    deployment,
    file_types,
    records,
    settings_file,
)

# A FAIL row, as checks.render prints it.
FAIL_ROW = re.compile(r"(?m)^  FAIL ")


class TestImage(DeployPathCase):
    def test_confirm_then_the_image(self):
        """Deploy step 3, then step 4: the confirm refuses the template until its <...> are filled in, and a
        file inside the package folder, recording nothing; the real file's confirm records it in this folder's records
        (the file as messages show it, the container name, the image) and writes nothing else, the first start's to
        deliver; a changed image.tag, confirmed, is the image the records name. The image step loads that image,
        checks its labels and this network's TLS, and sets nothing up: the first start does, from the delivered
        settings, so the step makes no compose container, volume or network, sends nothing on stdin, and prints no
        FAIL row on a correct deployment (a login that step 5 sets is not checked there).
        """
        # The confirm refuses the template (settings_file.template) until its <...> are filled in.
        tpl = self.home / "tpl.yaml"
        tpl.write_text(settings_file.template(file_types.DEPLOY))
        tpl.chmod(0o600)
        self.assertEqual(deployment.settings_kind(tpl), "deploy")
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("confirm", "--settings", str(tpl), "--yes")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "integrations.github_cli.username: still the template's <user_login>",
            one_line(out),
        )
        self.assertIn("nothing confirmed", one_line(out))
        self.assertIsNone(self.confirm_record())
        inside = self.pkg / "docker" / "my-settings.yaml"
        shutil.copy(self.settings, inside)
        inside.chmod(0o600)
        rc, out, err = self.rkmon("confirm", "--settings", str(inside), "--yes")
        self.assertEqual(rc, 4, out + err)
        self.assertIn("inside the package folder", one_line(out))
        inside.unlink()
        self.assertIsNone(self.confirm_record())
        # The real file: recorded in this folder's records, and nothing else written (the first start delivers it).
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "== deploy step 3 of 8, confirm: check your deploy settings, show them, and confirm them\n"
            ),
            out,
        )
        # No image here: its row waits for the image step.
        self.assertIn("SKIP  image: checked at step 4, once it is on this host", out)
        c = self.confirm_record()
        self.assertEqual(
            (c.settings_file, c.container_name, c.image.reference),
            ("~/rkmon-deploy.yaml", "rkmon-test", "rocke-monitor:rkmon-test"),
        )
        for name in (deliver.ENV_FILE, deliver.OVERRIDE_FILE):
            self.assertFalse((self.pkg / name).exists(), name)
        # The summary: the monitor's ids, the settings'.
        self.assertRegex(
            out, r"(?m)^  users\.monitor_uid: +4242\n  users\.monitor_gid: +4343$"
        )
        # A changed image.tag, confirmed: the image the records name.
        self.write_settings(image_tag="split-1")
        self.confirmed()
        self.assertEqual(self.confirm_record().image.tag, "split-1")
        # The image step: the file loaded, its labels and this network's TLS checked, and nothing set up.
        image_file = self.home / "rocke-monitor-split-1.tar.gz"
        image_file.write_text("x")
        self.answer(
            [
                *deploy_folder.secrets_volumes(False),
                [r"^load -i", "Loaded image: rocke-monitor:split-1", 0],
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "TLS ok", 0],
            ]
        )
        self.log.unlink(missing_ok=True)
        self.stdin.unlink(missing_ok=True)
        rc, out, err = self.rkmon("image", "load", str(image_file))
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "== deploy step 4 of 8, image: make sure the image is on this host and suits this package\n"
            ),
            out,
        )
        self.assertIn("ubuntu:26.04, ROCm 10.1.0-5", out)
        self.assertIn("PASS  TLS in the image", out)
        # One outcome: what the step brought, then the state's next line: no volume keeps the logins yet, so every one
        # the settings need is missing (state 7); before the first start, with its step.
        self.assertTrue(
            out.endswith(
                rewrapped(
                    f"\nLOADED rocke-monitor:split-1\nnext: step 5, login: {self.w} login\n"
                )
            ),
            out,
        )
        self.assertNotRegex(out, FAIL_ROW)
        self.assertEqual([c for c in self.calls() if MAKES.search(c)], [])
        self.assertFalse(self.stdin.exists())

    def test_the_image_step_refuses_what_does_not_suit_the_settings(self):
        """Deploy step 4: an image that is not here, was not built from this package's image recipe (its
        own copies of the files an image keeps are not this package's: code_volume.IMAGE_BOUND), or does not trust
        this network's TLS is FAILED, exit 4, with the command again; a file that has no image of the name the
        settings give is FAILED too, its fix naming the key and the name of the image in it.
        """
        self.confirmed()
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("image", "check")
        self.assertEqual(rc, 4, out + err)
        self.assertIn("FAIL  image: no image rocke-monitor:rkmon-test here", out)
        self.assertIn(
            "rkmon image load IMAGE_FILE, IMAGE_FILE being the image's file",
            " ".join(out.split()),
        )
        self.assertTrue(
            out.endswith(
                "\n\n"
                + finished(
                    "FAILED: 1 of 1 checks failed; nothing changed (fix each FAIL above)",
                    f"next: {self.w} image check",
                )
            ),
            out,
        )
        self.answer(
            [
                *self.image_code(changed=["docker/Dockerfile"]),
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "TLS ok", 0],
            ]
        )
        rc, out, err = self.rkmon("image", "check")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAIL image: rocke-monitor:rkmon-test was not built from this package's image recipe: its copies of "
            "docker/Dockerfile are not this package's fix: build an image from the new package "
            "(QUICK_BUILD_GUIDE.md), then set image.tag in your deploy settings file to its tag",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(
                "\n\n"
                + finished(
                    "FAILED: 1 of 2 checks failed; nothing changed (fix each FAIL above)",
                    f"next: {self.w} image check",
                )
            ),
            out,
        )
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [
                    r"--entrypoint python3",
                    "TLS untrusted unable to get local issuer",
                    0,
                ],
            ]
        )
        rc, out, err = self.rkmon("image", "check")
        self.assertEqual(rc, 4)
        self.assertIn(
            "this network inspects TLS, and the image does not trust its root certificate",
            out,
        )
        # The probe's container has the compose file's limits.
        from rkmon_lib import container_layout

        probes = [c for c in self.calls() if "--entrypoint python3" in c]
        self.assertTrue(probes)
        for c in probes:
            self.assertIn(" ".join(container_layout.RUN_LIMITS), c)
        self.assertIn("operating_system.ca_certificates", out)
        self.answer([[r"^load -i", "Loaded image: ubuntu:26.04", 0]])
        other = self.home / "other.tar"
        other.write_text("x")
        rc, out, err = self.rkmon("image", "load", str(other))
        self.assertEqual(rc, 4, out + err)
        said = one_line(out)
        self.assertIn(
            f"FAILED: {other} has no rocke-monitor image (ubuntu:26.04): your deploy settings name "
            "rocke-monitor (image.name)",
            said,
        )
        self.assertIn(
            f'fix: set image.name in ~/rkmon-deploy.yaml to "ubuntu", then run {self.w} confirm; or load the '
            "file the build path saved (QUICK_BUILD_GUIDE.md)",
            said,
        )
        self.assertTrue(
            out.endswith("\n" + finished(f"next: {self.w} image load {other}")), out
        )

    def test_the_image_check_changes_nothing(self):
        """rkmon image check checks the image here and changes nothing: PASSED, then the state's next line,
        the logins while a volume that keeps them is missing (state 7), else the start (state 8); no FAIL row, no
        Docker object made, nothing given to a container on its stdin, and this folder's files and records as they
        were. The first start sets the work volume up (test_the_first_start_delivers_your_confirmed_settings).
        """
        self.confirmed()

        def files() -> dict:
            return {
                rel: (self.pkg / rel).read_bytes()
                for rel in (records.FILE, deliver.ENV_FILE, deliver.OVERRIDE_FILE)
                if (self.pkg / rel).exists()
            }

        before = files()
        for held, want in (
            (False, f"step 5, login: {self.w} login"),
            (True, f"step 6, start: {self.w} start"),
        ):
            with self.subTest(held=held):
                self.answer(
                    [
                        *deploy_folder.secrets_volumes(held),
                        [r"image inspect", GOOD, 0],
                        [r"--entrypoint python3", "TLS ok", 0],
                    ]
                )
                self.log.unlink(missing_ok=True)
                self.stdin.unlink(missing_ok=True)
                rc, out, err = self.rkmon("image", "check")
                self.assertEqual((rc, err), (0, ""), out)
                self.assertTrue(
                    out.endswith(
                        "\n"
                        + finished(
                            "PASSED: rocke-monitor:rkmon-test suits this package",
                            f"next: {want}",
                        )
                    ),
                    out,
                )
                self.assertNotRegex(out, FAIL_ROW)
                self.assertEqual([c for c in self.calls() if MAKES.search(c)], [])
                self.assertFalse(self.stdin.exists())
                self.assertEqual(files(), before)

    def test_an_image_file_that_is_not_there(self):
        """image load of a file that is not there: the file's FAIL row, its fix naming the command with
        IMAGE_FILE in the file's place, FAILED, exit 4, and the command again; under --dry-run too, which passes
        nothing it cannot load. docker is not asked to load anything."""
        missing = str(self.home / "missing.tgz")
        for typed in (
            ("image", "load", missing),
            ("--dry-run", "image", "load", missing),
        ):
            with self.subTest(typed=typed):
                rc, out, err = self.rkmon(*typed)
                self.assertEqual((rc, err), (4, ""), out)
                again = " ".join(typed[:-1])
                self.assertIn(
                    f"FAIL the image's file: no file at ~/missing.tgz fix: {self.w} {again} IMAGE_FILE, IMAGE_FILE "
                    "being the image's file the build path saved (QUICK_BUILD_GUIDE.md)",
                    one_line(out),
                )
                self.assertTrue(
                    out.endswith(
                        "\n\n"
                        + finished(
                            "FAILED: ~/missing.tgz is not loaded; nothing changed (fix each FAIL above)",
                            f"next: {self.w} {again} {missing}",
                        )
                    ),
                    out,
                )
                self.assertNotIn("PASSED", out)
        self.assertFalse([c for c in self.calls() if "\tload " in c])

    def test_a_new_image_is_checked_itself_then_named_in_the_settings(self):
        """The image a load brings is the one checked (not the deployment's current one). One of another tag than your
        confirmed settings name is a settings change: with the deployment's container or without
        it, the load ends with image.tag to set, then the confirm, and nothing goes into the deployment. Once confirmed,
        the settings name it: a pull brings it from their registry, and a check checks it, each ending with the state's
        next line.
        """
        self.confirmed()
        self.delivered()
        (self.home / "new.tar.gz").write_text("x")
        base = [
            [r"^load -i", "Loaded image: rocke-monitor:new", 0],
            [r"image inspect .* rocke-monitor:new$", GOOD, 0],
            [r"image inspect", "", 1],  # the deployment's own tag is not here
            [r"--entrypoint python3", "TLS ok", 0],
        ]
        named = finished(
            "LOADED rocke-monitor:new",
            f'next: set image.tag in ~/rkmon-deploy.yaml to "new", then {self.w} confirm',
        )
        for made in ("0123456789ab", ""):  # the deployment's container, then none
            with self.subTest(container=bool(made)):
                self.answer(base + [[r"compose ps -a -q rkmon", made, 0]])
                rc, out, err = self.rkmon(
                    "image", "load", str(self.home / "new.tar.gz")
                )
                self.assertEqual(rc, 0, out + err)
                self.assertIn(
                    f"rocke-monitor:new: package {LABELS['org.rocke.package.version']}",
                    out,
                )
                self.assertTrue(out.endswith("\n" + named), out)
                # the apply moves it
                self.assertEqual(self.dotenv()["RKMON_TAG"], "rkmon-test")
        # The settings name it once confirmed: a pull from their registry, then a check.
        self.write_settings(image_tag="new", registry="registry.example.com/team")
        self.confirmed()
        self.answer(base[1:] + [[r"compose ps -a -q rkmon", "0123456789ab", 0]])
        self.log.unlink()
        rc, out, err = self.rkmon("image", "pull")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "pull registry.example.com/team/rocke-monitor:new",
            "\n".join(self.calls()),
        )
        self.assertIn("\n" + finished("PULLED rocke-monitor:new") + "next:", out)
        self.assertNotIn("set image.tag", out)
        rc, out, err = self.rkmon("image", "check")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\n" + finished("PASSED: rocke-monitor:new suits this package") + "next:",
            out,
        )

    def test_an_image_of_its_own_name(self):
        """image.name names the image the deployment runs: the confirm records it, and the load (a file of another
        name is refused, with the fix) and the check take it, with a new tag in the settings too.
        """
        self.write_settings(image_name="team/rkmon")
        self.confirmed()
        self.assertEqual(self.confirm_record().image.reference, "team/rkmon:rkmon-test")
        (self.home / "img.tar.gz").write_text("x")
        self.answer([[r"^load -i", "Loaded image: rocke-monitor:rkmon-test", 0]])
        rc, out, err = self.rkmon("image", "load", str(self.home / "img.tar.gz"))
        self.assertEqual(rc, 4, out + err)
        said = one_line(out)
        self.assertIn(
            "has no team/rkmon image (rocke-monitor:rkmon-test): your deploy settings name team/rkmon "
            "(image.name)",
            said,
        )
        self.assertIn(
            'fix: set image.name in ~/rkmon-deploy.yaml to "rocke-monitor", then run',
            said,
        )
        self.answer(
            [
                *deploy_folder.secrets_volumes(False),
                [r"^load -i", "Loaded image: team/rkmon:rkmon-test", 0],
                [r"image inspect .* team/rkmon:rkmon-test$", GOOD, 0],
                [r"--entrypoint python3", "TLS ok", 0],
            ]
        )
        rc, out, err = self.rkmon("image", "load", str(self.home / "img.tar.gz"))
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "PASS  image: team/rkmon:rkmon-test was built from this package's image recipe",
            out,
        )
        self.assertTrue(
            out.endswith(
                rewrapped(
                    f"\nLOADED team/rkmon:rkmon-test\nnext: step 5, login: {self.w} login\n"
                )
            ),
            out,
        )
        # Once the first start has delivered the settings, a new tag confirmed: the check checks team/rkmon:new.
        self.delivered()
        self.write_settings(image_name="team/rkmon", image_tag="new")
        self.confirmed()
        self.answer(
            [
                [r"image inspect .* team/rkmon:new$", GOOD, 0],
                [r"--entrypoint python3", "TLS ok", 0],
            ]
        )
        rc, out, err = self.rkmon("image", "check", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("PASSED: team/rkmon:new suits this package", out)

    def test_each_image_command_ends_with_one_outcome(self):
        """Deploy step 4's other outcomes: a load before any confirm is LOADED, then the state's next line;
        a pull or a check then is the state's refusal, exit 3; a pull with no registry is REFUSED, exit 4, its fix
        naming the key; a docker command that fails is ERROR, exit 1, then the command again; under --dry-run, PASSED
        with what the real run would do, nothing loaded and nothing set up."""
        image_file = self.home / "monitor.tar.gz"
        image_file.write_text("x")
        self.answer([[r"^load -i", "Loaded image: rocke-monitor:rkmon-test", 0]])
        rc, out, err = self.rkmon("image", "load", str(image_file))
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                "\n" + finished("LOADED rocke-monitor:rkmon-test", self.state_2_next())
            ),
            out,
        )
        for command in (("image", "pull"), ("image", "check")):
            rc, out, err = self.rkmon(*command)
            self.assertEqual((rc, err), (3, ""), out)
            self.assertTrue(
                out.endswith(
                    "\n"
                    + finished(
                        "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                        self.state_2_next(),
                    )
                ),
                out,
            )
        # Settings confirmed, with no registry: the pull is refused, and the state's next line (no image here) is
        # the load.
        self.confirmed()
        self.answer([[r"image inspect", "", 1]])
        self.log.unlink()
        rc, out, err = self.rkmon("image", "pull")
        self.assertEqual((rc, err), (4, ""), out)
        said = one_line(out)
        self.assertIn(
            "REFUSED: your deploy settings name no registry (image.registry); nothing pulled",
            said,
        )
        self.assertIn(
            "fix: set image.registry in ~/rkmon-deploy.yaml to your registry's address, then run "
            f"{self.w} confirm; or load the image from a file: {self.w} image load IMAGE_FILE",
            said,
        )
        self.assertTrue(
            out.endswith(
                "\n" + finished(f"next: step 4, image: {self.w} image load IMAGE_FILE")
            ),
            out,
        )
        self.assertFalse(any(c.startswith("pull") for c in self.calls()))
        # docker load fails: ERROR, and the command again.
        self.answer([[r"^load -i", "", 1]])
        rc, out, err = self.rkmon("image", "load", str(image_file))
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "ERROR: docker load exited 1 (the lines above say why)",
                    f"next: {self.w} image load {image_file}",
                )
            ),
            out,
        )
        # Under --dry-run: what the real run would do; nothing loaded, nothing set up.
        self.answer(
            [[r"image inspect", GOOD, 0], [r"--entrypoint python3", "TLS ok", 0]]
        )
        self.log.unlink()
        rc, out, err = self.rkmon("--dry-run", "image", "load", str(image_file))
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"PASSED: {image_file} can be loaded",
                    f"next: {self.w} image load {image_file}",
                )
            ),
            out,
        )
        rc, out, err = self.rkmon("--dry-run", "image", "check")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "PASSED: rocke-monitor:rkmon-test suits this package",
                    f"next: {self.w} image check",
                )
            ),
            out,
        )
        self.assertFalse(
            any(c.startswith("load") for c in self.calls()),
            self.calls(),
        )
        # A registry, confirmed, and docker pull fails: ERROR, and the command again.
        self.write_settings(registry="registry.example.com/team")
        self.confirmed()
        self.answer([[r"^pull ", "", 1]])
        rc, out, err = self.rkmon("image", "pull")
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "ERROR: docker pull exited 1 (the lines above say why)",
                    f"next: {self.w} image pull",
                )
            ),
            out,
        )
        # Under --dry-run the pull names the registry's address, with the key that sets it, as the push does; nothing
        # pulled.
        self.log.unlink()
        rc, out, err = self.rkmon("--dry-run", "image", "pull")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    "PASSED: rocke-monitor:rkmon-test can be pulled from registry.example.com/team (image.registry)",
                    f"next: {self.w} image pull",
                )
            ),
            out,
        )
        self.assertFalse(any(c.startswith("pull") for c in self.calls()), self.calls())

    def test_the_image_s_own_code_is_read_with_no_network_and_no_mount(self):
        """The image row reads the image's own code from a one-off container of it with no network, under
        the limits every container of the image runs with (container_layout.RUN_LIMITS), which mounts nothing: no
        volume, no folder and no file of variables."""
        self.answer(
            [
                *self.image_code(),
                [r"image inspect", GOOD, 0],
                [r"rk logins --json", "[]", 0],
            ]
        )
        rc, out, err = self.rkmon(
            "--dry-run", "confirm", "--settings", str(self.settings)
        )
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("PASS  image: ", out)
        [probe] = [c for c in self.calls() if deploy_folder.IMAGE_CODE in c]
        self.assertIn(
            "\trun --rm --network none "
            + " ".join(container_layout.RUN_LIMITS)
            + " --entrypoint python3 rocke-monitor:rkmon-test -I -c ",
            probe,
        )
        for mounted in (" -v ", "--volume", "--mount", "--env-file"):
            self.assertNotIn(mounted, probe)


if __name__ == "__main__":
    unittest.main()
