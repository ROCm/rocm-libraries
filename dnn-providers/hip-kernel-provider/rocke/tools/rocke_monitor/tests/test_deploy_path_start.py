# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deploy step 6, the start, and step 8, the first tick, end to end: docker/rkmon run on a copy of the package against
the stand-in docker, as tests/deploy_path_fixture.py sets it up. The first start delivers the confirmed settings and
this package's code, and sets the work volume up from them (rk provision) before rk init, a start cut short being
finished by the next; a later one starts what is in place and never a change. Both refuse a busy port, another
deployment's volume, a login that is missing, a private file that is not the package's and an image or code that does
not suit, and a Ctrl-C ends them INTERRUPTED. The first tick waits for a tick after the start.
"""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import re
import signal
import stat
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
from deploy_path_fixture import (  # noqa: E402
    ELSEWHERE,
    GOOD,
    INSTALLED,
    LABELS,
    LAYER_READ,
    LAYER_RUN,
    OWN_CODE,
    STARTED,
    UNREAD,
    DeployPathCase,
)
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import code_volume, deliver, image, records, steps  # noqa: E402

OTHER = json.dumps(dict(LABELS, **{"org.rocke.package.version": "1999.01.01"}))
# the start's code-install, which leaves the code kept for a rollback as it is (code_volume.LEAVE_KEPT: docker/rkmon
# start_code); apply's has no option
START_INSTALL = f"code-install {code_volume.LEAVE_KEPT}"


class TestStart(DeployPathCase):
    def test_start_refuses_a_busy_port(self):
        """The check's row for the port (a busy port, exit 4), with the key that sets it, then REFUSED and the
        deployment state's next line: here state 8's, the start. Nothing is delivered, and no code goes in: the
        start only read the image's own code (image_code).
        """
        self.confirmed()
        self.answer([[r"image inspect", GOOD, 0]])
        with socket.socket() as s:
            s.bind(("127.0.0.1", self.port))
            s.listen()
            rc, out, err = self.rkmon("start")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"  FAIL  port: 127.0.0.1:{self.port} is in use by another program or container\n"
            "        fix: set features.dashboards.admin.port in ~/rkmon-deploy.yaml to a port that is free on this "
            "host\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: a port the dashboards need is in use; nothing changed (fix each FAIL above)",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        self.assertFalse(
            [
                c
                for c in self.calls()
                if "rk init" in c
                or c.endswith("up -d")
                or "code-install" in c
                or ("--entrypoint python3" in c and deploy_folder.IMAGE_CODE not in c)
            ]
        )
        self.assertFalse((self.pkg / deliver.ENV_FILE).exists())

    def test_start_refuses_an_unfit_image_and_reads_no_other_monitor(self):
        """Check rows, then REFUSED (exit 4), nothing delivered and no code in: an image not built from this
        package's image recipe, with the fix that builds one. Another monitor on this host that posts reviews for the
        same repository refuses nothing, these settings posting reviews too: neither the start nor the check reads
        another deployment, and no check takes the place of the one that did.
        """
        self.confirmed()
        self.answer(
            [
                *self.image_code(changed=["docker/entrypoint.sh", "rk"]),
                [r"image inspect", GOOD, 0],
            ]
        )
        rc, out, err = self.rkmon("start")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "FAIL image: rocke-monitor:rkmon-test was not built from this package's image recipe: its copies of "
            "docker/entrypoint.sh, rk are not this package's fix: build an image from the new package "
            "(QUICK_BUILD_GUIDE.md), then set image.tag in your deploy settings file to its tag",
            one_line(out),
        )
        self.assertFalse(
            [
                c
                for c in self.calls()
                if "code-install" in c
                or ("--entrypoint python3" in c and deploy_folder.IMAGE_CODE not in c)
            ]
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the image rocke-monitor:rkmon-test does not suit this package folder; nothing "
                    "changed (fix each FAIL above)",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        self.write_settings(
            features={
                "review_automation": {"mode": "live"},
                "merge_queue": {"enabled": False},
            }
        )
        self.confirmed()
        self.log.unlink(missing_ok=True)
        self.answer(
            [
                [r"^version", "29.0.0", 0],
                [r"compose version", "2.40.0", 0],
                [r"image inspect", GOOD, 0],
                *ELSEWHERE,
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon("check")
        self.assertNotIn("another monitor", out)
        self.assert_reads_no_other_monitor()
        # the first start: no container until its compose up, which starts it
        stopped = self.d / "stopped"
        stopped.touch()
        self.log.unlink()
        rc, out, err = self.rkmon(
            "start", FAKE_DOCKER_RUNNING="1", FAKE_DOCKER_STOPPED=str(stopped)
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn("STARTED: the monitor runs your confirmed settings", out)
        self.assert_reads_no_other_monitor()

    def test_start_refuses_while_a_login_is_missing(self):
        """A login your confirmed settings need that is not set yet is state 7's: the start refuses with its
        sentence and the login step (an earlier step, exit 3), before anything is delivered or runs. Before the first
        start, the volume that keeps it holds nothing for it; once this folder's container exists, `rk logins` lists it
        as missing, its JSON read whatever docker prints after it."""
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
        logins = [r"rk logins --json", json.dumps(rows) + "\n Container x Removed", 1]
        made = {
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }
        for case, answers, env in (
            (
                "before the first start",
                deploy_folder.secrets_volumes(True, ["github_read"]),
                {},
            ),
            ("once the container exists", [logins], made),
        ):
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.answer([*answers, [r"image inspect", GOOD, 0]])
                rc, out, err = self.rkmon("start", **env)
                self.assertEqual(rc, 3, out + err)
                self.assertTrue(
                    out.endswith(
                        finished(
                            "REFUSED: your confirmed settings need github_read, which is missing or wrong; "
                            "nothing changed",
                            f"next: step 5, login: {self.w} login",
                        )
                    ),
                    out,
                )
        self.assertFalse(
            [
                c
                for c in self.calls()
                if "rk init" in c or c.endswith("up -d") or "--entrypoint python3" in c
            ]
        )
        self.assertFalse((self.pkg / deliver.ENV_FILE).exists())

    def test_the_start_runs_rk_init_then_the_container(self):
        """Deploy step 6, the settings confirmed: with no image here, state 6's refusal (an earlier step, exit 3),
        and nothing runs. Else, never started (state 8), under --dry-run (before the command or after it): where
        your confirmed settings go, then this package's code in by root's code-install in a one-off container
        (docker/rk_root.py), rk provision in another (it sets the work volume up from them while its record is
        missing: tests/test_provision.py), rk init in another (rk seeds the state, ON, only while
        there is none, and never undoes an rk pause: tests/test_provision.py and tests/test_control.py), rk resume
        too with --on, then docker compose up -d; PASSED, and the command without --dry-run. The one-offs run while a
        stopped monitor's container runs too: after an image load, the running container may still be the old image's,
        whose rk lacks what the new start asks. Only read-only questions run, each in docker/, and nothing is written.
        While the container runs, its work volume set up, the start has nothing to do, and the state's line follows;
        nothing stops it.
        """
        self.confirmed()
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("start")
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the image rocke-monitor:rkmon-test is not on this host; nothing changed",
                    f"next: step 4, image: {self.w} image load IMAGE_FILE",
                )
            ),
            out,
        )
        self.answer([[r"image inspect", GOOD, 0], [r"rk logins --json", "[]", 0]])
        e = f"-e RKMON_WRAPPER={self.w}"
        banner = steps.banner(steps.DEPLOY_STEPS[5], dry_run=True)
        delivery = (
            "your confirmed settings go into docker/.env, docker/compose.override.yaml and the settings volume's "
            "deploy.json"
        )
        for typed in (
            ("--dry-run", "start"),
            ("start", "--dry-run"),
            ("--dry-run", "start", "--on"),
        ):
            with self.subTest(typed=typed):
                rc, out, err = self.rkmon(*typed)
                self.assertEqual((rc, err), (0, ""), out)
                resume = f"docker compose run --rm -T {e} rkmon rk resume"
                runs = [
                    banner,
                    delivery,
                    f"docker compose run --rm -T --no-deps -e RKMON_AS=root rkmon {START_INSTALL}",
                    f"docker compose run --rm -T {e} rkmon rk provision",
                    f"docker compose run --rm -T {e} rkmon rk init --if-new --on",
                    *([resume] if "--on" in typed else []),
                    "docker compose up -d",
                ]
                again = " ".join(w for w in typed if w != "--dry-run")
                self.assertEqual(
                    out,
                    "\n".join(runs)
                    + "\n"
                    + finished(
                        "PASSED: the monitor can start", f"next: {self.w} {again}"
                    ),
                )
        # under --dry-run, of the image only its own code is read (image_code)
        self.assertFalse(
            [
                c
                for c in self.calls()
                if "rk provision" in c
                or "rk init" in c
                or "code-install" in c
                or c.endswith("up -d")
                or ("--entrypoint python3" in c and deploy_folder.IMAGE_CODE not in c)
            ]
        )
        self.assertEqual(
            {c.split("\t", 1)[0] for c in self.calls()}, {str(self.pkg / "docker")}
        )
        self.assertFalse((self.pkg / deliver.ENV_FILE).exists())
        self.assertFalse((self.pkg / deliver.OVERRIDE_FILE).exists())
        # the container runs, its work volume set up, and rk stop has not stopped the monitor in it: nothing to do
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"rk stopped$", "", 1],
                [r"rk provision --check$", "provisioned", 0],
            ]
        )
        self.log.unlink()
        rc, out, err = self.rkmon("start", FAKE_DOCKER_RUNNING="1")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertEqual(
            out,
            steps.banner(steps.DEPLOY_STEPS[5], dry_run=False)
            + "\n"
            + finished(
                "NOTHING TO DO: the container is running",
                # the state's line: no doctor has passed since the start
                f"next: step 7, doctor: {self.w} doctor",
            ),
        )
        # whether its work volume is set up is asked, read-only; nothing runs, and nothing stops
        self.assertFalse(
            [
                c
                for c in self.calls()
                if ("rk provision" in c and not c.endswith("rk provision --check"))
                or "rk init" in c
                or "code-install" in c
                or "up -d" in c
                or "compose stop" in c
                or "--entrypoint python3" in c
            ]
        )

    def test_the_first_start_delivers_your_confirmed_settings(self):
        """The first start (state 8): your confirmed settings into place by deliver, their one writer, and
        this package's code by code-install, before rk provision sets the work volume up from them, then rk init and
        the container's start: docker/.env and the
        compose override as the plan makes them, yours only, and the layer through a one-off container in the compose
        project the settings name, what it replaces kept (none yet, as cat says of the settings volume), each file's
        fingerprint kept in the host's records; then STARTED and step 7. A first start that stopped once the
        layer was in place delivers again, without keeping that layer as the one before it.
        """
        self.confirmed()
        stopped = self.d / "stopped"  # no container until the start's compose up
        stopped.touch()
        layer_file = f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}"
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [LAYER_READ, f"cat: {layer_file}: No such file or directory", 1],
                [r"--entrypoint python3", "", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        env = {"FAKE_DOCKER_RUNNING": "1", "FAKE_DOCKER_STOPPED": str(stopped)}
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "\nyour confirmed settings go into docker/.env, docker/compose.override.yaml and the settings volume's "
            "deploy.json\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        made = deploy_folder.plan(self.pkg, self.settings)
        env_text, override, layer = deliver.texts(made)
        for rel, text in (
            (deliver.ENV_FILE, env_text),
            (deliver.OVERRIDE_FILE, override),
        ):
            with self.subTest(file=rel):
                path = self.pkg / rel
                self.assertIsNotNone(text)
                self.assertEqual(path.read_text(encoding="utf-8"), text)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        # the fingerprints in the host's records (a hand edit shows: state 9), none yet of a layer before it
        self.assertEqual(
            records.read(self.pkg).written,
            records.Written(
                env=deliver.fingerprint(env_text),
                override=deliver.fingerprint(override),
                layer=deliver.fingerprint(layer),
                kept_layer="",
            ),
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        run = " ".join(("compose", "-p", deploy_folder.NAME, *deliver.LAYER_RUN))
        order = [
            calls.index(run),
            next(i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            next(i for i, c in enumerate(calls) if c.endswith("rk init --if-new --on")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)
        gid = made.variables["RKMON_GID"]
        self.assertIn(
            f"\ndeliver({layer!r}, {gid}, True, None)\n",
            self.stdin.read_text(encoding="utf-8"),
        )
        # stopped after the layer went in: the same plan again, the layer in place not kept as the one before it
        stopped.touch()
        self.stdin.unlink()
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [
                    re.escape(f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}") + "$",
                    layer,
                    0,
                ],
                [r"--entrypoint python3", "", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("STARTED: the monitor runs your confirmed settings", out)
        self.assertIn(
            f"\ndeliver({layer!r}, {gid}, False, None)\n",
            self.stdin.read_text(encoding="utf-8"),
        )

    def test_a_first_start_that_cannot_read_the_settings_volume_starts_nothing(self):
        """The first start while the settings volume cannot be read (docker fails; not a volume with no layer yet,
        which cat says): ERROR, never a keep (deliver.keep_previous): nothing written, nothing started, and the start
        again."""
        self.confirmed()
        stopped = self.d / "stopped"
        stopped.touch()
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [LAYER_READ, UNREAD, 1],
                [r"rk stopped$", "", 1],
            ]
        )
        env = {"FAKE_DOCKER_RUNNING": "1", "FAKE_DOCKER_STOPPED": str(stopped)}
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the settings volume's deploy.json could not be read, to keep it for rollback (docker "
                    f"compose run exited 1: {UNREAD}): nothing changed; nothing started",
                    f"next: {self.w} start",
                )
            ),
            out,
        )
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertIsNone(records.read(self.pkg).written)

    def test_a_first_start_whose_team_s_skill_changed_after_its_plan_refuses(self):
        """A team's skill that changed after the first start's plan, before deliver reads the skills
        (deliver.SkillsChanged): REFUSED, exit 3, an earlier step not done: nothing written, nothing started, and the
        state's next line, the confirm (state 5)."""
        skills = self.home / "team-skills"
        (skills / "agents").mkdir(parents=True)
        (skills / "agents" / "a.md").write_text("# a\n", encoding="utf-8")
        self.write_settings(
            features={"review_automation": {"skills_folder": str(skills)}}
        )
        self.confirmed()
        stopped = self.d / "stopped"
        stopped.touch()
        none = f"cat: {deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}: No such file or directory"
        added = skills / "agents" / "b.md"
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [LAYER_READ, none, 1, {}, {"makes": str(added)}],
                [r"rk stopped$", "", 1],
            ]
        )
        env = {"FAKE_DOCKER_RUNNING": "1", "FAKE_DOCKER_STOPPED": str(stopped)}
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    f"REFUSED: the team's skills in {skills} changed since your confirm (agents/b.md): nothing "
                    "changed; nothing started",
                    f"next: step 3, confirm: {self.w} confirm",
                )
            ),
            out,
        )
        self.assertEqual(self.called(LAYER_RUN, "up -d"), [-1, -1])
        self.assertIsNone(records.read(self.pkg).written)

    def test_a_first_start_cut_short_is_finished_by_the_next(self):
        """A first start whose rk provision fails (a step that did not finish: tests/test_provision.py) stops there:
        ERROR, which says the work volume is not set up in full and that what was in place stays as it is, as a later
        start's repair that fails leaves it too, and never names rk provision (not a command for you), with the
        start again as its next line; neither rk init nor the container runs. An rk init that fails is said
        the same way.
        The next start delivers the same settings again, puts this package's code in (code-install, which has nothing
        to do while it is in place) and runs rk provision again, which finishes the work volume (its steps in place
        stay as they are), then rk init and the container."""
        self.confirmed()
        stopped = self.d / "stopped"  # no container until the start's compose up
        stopped.touch()
        env = {"FAKE_DOCKER_RUNNING": "1", "FAKE_DOCKER_STOPPED": str(stopped)}
        both = [
            [r"image inspect", GOOD, 0],
            [r"--entrypoint python3", "", 0],
            [r"rk stopped$", "", 1],
        ]
        self.answer([[r" rk provision$", "  FAIL  clone: not made", 1], *both])
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                "  FAIL  clone: not made\n"
                + finished(
                    "ERROR: the work volume is not set up in full (what was in place stays as it is), so the "
                    "monitor did not start (the lines above say why)",
                    f"next: {self.w} start",
                )
            ),
            out,
        )
        self.assertEqual(
            [c for c in self.calls() if "rk init" in c or c.endswith("up -d")], []
        )
        # The next start: the layer in place delivered again, then each step. An rk init that fails is said by what it
        # left undone, never by its command.
        layer = deliver.texts(deploy_folder.plan(self.pkg, self.settings))[2]
        delivered = [
            re.escape(f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}") + "$",
            layer,
            0,
        ]
        self.answer([delivered, [r" rk init --if-new --on$", "", 1], *both])
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "ERROR: the monitor's state is not seeded, so the monitor did not start (the lines above say "
                    "why)",
                    f"next: {self.w} start",
                )
            ),
            out,
        )
        # The start after it finishes.
        self.answer([delivered, *both])
        self.log.unlink()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        order = [
            next(i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            next(i for i, c in enumerate(calls) if c.endswith("rk init --if-new --on")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)

    def test_a_container_started_some_other_way_waits_for_the_first_start(self):
        """A container started some other way than rkmon start (a hand docker compose up after a first start cut
        short) runs, but its work volume is not set up (rk provision --check in it exits 1), so its entry point waits:
        never started (state 8), and rkmon next names start; adopt refuses, with the start's line. The start stops
        it (nothing runs in it yet) and says so, then makes the first start as for none: the delivery, this package's
        code (code-install), rk provision and rk init (the seed poll) in one-offs, never in that container, then
        compose up; STARTED, then step 7's line, as rkmon next's after it."""
        self.confirmed()
        # the work volume's setup record, once rk provision has run
        provisioned = self.d / "provisioned"
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "", 0],
                [r"rk logins --json", "[]", 0],
                [r"rk stopped$", "", 1],
                [
                    r"run --rm .* rk provision$",
                    "provisioned",
                    0,
                    {},
                    {"makes": str(provisioned)},
                ],
                [
                    r"rk provision --check$",
                    "not set up yet",
                    1,
                    {},
                    {"unless": str(provisioned)},
                ],
                [r"rk provision --check$", "provisioned", 0],
            ]
        )
        env = dict(self.running(), FAKE_DOCKER_STOPPED=str(self.d / "stopped"))
        start = f"next: step 6, start: {self.w} start"
        rc, out, err = self.rkmon("next", **env)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(
            "this deployment's container runs, but its work volume is not set up (it was started some other way than "
            f"{self.w} start)",
            one_line(out),
        )
        self.assertTrue(out.endswith("\n" + finished(start)), out)
        rc, out, err = self.rkmon(
            "adopt", "--from", str(self.d / "copy"), "--yes", **env
        )
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the container is running: it waits for its first start; nothing changed",
                    start,
                )
            ),
            out,
        )
        self.log.unlink()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"\nthe container was started some other way than {self.w} start, and waits for its first start: it "
            "stops, then the first start runs\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        order = [
            calls.index("compose stop"),
            next(i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            next(i for i, c in enumerate(calls) if c.endswith("rk init --if-new --on")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)
        for one_off in order[1:4]:
            self.assertTrue(
                calls[one_off].startswith("compose run --rm "), calls[one_off]
            )
        self.assertTrue(provisioned.exists())
        rc, out, err = self.rkmon("next", **env)
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 7, doctor: {self.w} doctor")), out
        )

    def test_the_start_never_stops_a_container_whose_work_volume_is_set_up(self):
        """Before it stops a container that waits for its first start, the start asks again whether its work volume is
        set up (rk provision --check), the fact state 8 reads: one whose volume was set up meanwhile (here, as the
        delivery runs) has a monitor that may have started, so nothing stops, and nothing more runs: NOTHING TO DO, then
        the state's line, as while the container runs."""
        self.confirmed()
        # the work volume's setup record, made while the delivery runs
        meanwhile = self.d / "provisioned"
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [
                    r"--entrypoint python3 deliver ",
                    "",
                    0,
                    {},
                    {"makes": str(meanwhile)},
                ],
                [r"--entrypoint python3", "", 0],
                [r"rk logins --json", "[]", 0],
                [r"rk stopped$", "", 1],
                [
                    r"rk provision --check$",
                    "not set up yet",
                    1,
                    {},
                    {"unless": str(meanwhile)},
                ],
                [r"rk provision --check$", "provisioned", 0],
            ]
        )
        env = dict(self.running(), FAKE_DOCKER_STOPPED=str(self.d / "stopped"))
        self.log.unlink()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(meanwhile.exists(), out)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: the container is running",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        self.assertNotIn("waits for its first start", out)
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertFalse(
            [
                c
                for c in calls
                if ("rk provision" in c and not c.endswith("rk provision --check"))
                or "rk init" in c
                or "code-install" in c
                or "up -d" in c
                or "compose stop" in c
            ],
            calls,
        )

    def test_your_shell_s_variables_never_change_what_compose_runs(self):
        """docker compose reads COMPOSE_* and RKMON_* over docker/.env's: your shell's never reach it, as rkmon drops
        them as it starts. The first start asks of the confirmed image, delivers in the confirmed compose project and
        writes docker/.env as the confirmed settings make it, whatever your shell exports; docker's connection
        variables reach each docker command as your shell has them."""
        self.confirmed()
        self.log.unlink()  # the start's calls alone
        stopped = self.d / "stopped"
        stopped.touch()
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        shell = {
            "RKMON_IMAGE": "shell-image",
            "RKMON_TAG": "shell-tag",
            "RKMON_PORT": "1",
            "RKMON_CPUS": "1",
            "COMPOSE_PROJECT_NAME": "shell-project",
            "COMPOSE_FILE": str(self.d / "shell.yaml"),
            "COMPOSE_PROFILES": "shell",
        }
        connection = {
            "DOCKER_HOST": "unix:///run/shell.sock",
            "DOCKER_CONTEXT": "shell-context",
            "DOCKER_CONFIG": str(self.d / "shell-config"),
        }
        seen = self.d / "env.log"
        rc, out, err = self.rkmon(
            "start",
            FAKE_DOCKER_RUNNING="1",
            FAKE_DOCKER_STOPPED=str(stopped),
            FAKE_DOCKER_ENV_LOG=str(seen),
            FAKE_DOCKER_ENV=",".join([*shell, *connection]),
            **shell,
            **connection,
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn("STARTED: the monitor runs your confirmed settings", out)
        made = deploy_folder.plan(self.pkg, self.settings)
        self.assertEqual(
            (self.pkg / deliver.ENV_FILE).read_text(encoding="utf-8"),
            deliver.texts(made)[0],
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertTrue(
            [
                c
                for c in calls
                if c.startswith("image inspect")
                and c.endswith(" rocke-monitor:rkmon-test")
            ],
            calls,
        )
        self.assertIn(
            " ".join(("compose", "-p", deploy_folder.NAME, *deliver.LAYER_RUN)), calls
        )
        self.assertFalse([c for c in calls if "shell" in c], calls)
        lines = seen.read_text(encoding="utf-8").splitlines()
        for name, value in shell.items():
            self.assertNotIn(f"{name}={value}", lines)
        for name, value in connection.items():
            self.assertEqual(lines.count(f"{name}={value}"), len(calls), name)

    def test_a_later_start_starts_what_is_in_place(self):
        """After the first start, a start starts what is in place and never applies a change: with the container
        stopped (state 12), this package's code goes in (code-install, which has nothing to do while it is in place),
        then the container starts, and nothing is delivered; while it runs, nothing to do, and the state's line; a
        hand edit of docker/.env is state 9's refusal, and a change you confirmed since is state 11's, both naming
        apply (exit 4), docker/.env as it was; a hand edit while that change waits is state 9's.
        """
        self.confirmed()
        stopped = self.d / "stopped"
        stopped.touch()
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        env = {
            "FAKE_DOCKER_RUNNING": "1",
            "FAKE_DOCKER_STOPPED": str(stopped),
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("your confirmed settings go into", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        self.assertIn(f"\n  code: {OWN_CODE}\n", out)
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertFalse([c for c in calls if "--entrypoint python3 deliver " in c])
        order = [
            next(i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual((rc, err), (0, ""), out)
        # the state's line, as rkmon next gives it: no doctor has passed since the start (state 16, step 7)
        self.assertTrue(
            out.endswith(
                finished(
                    "NOTHING TO DO: the container is running",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        stopped.touch()
        self.log.unlink()
        dotenv = self.pkg / deliver.ENV_FILE
        kept = dotenv.read_text(encoding="utf-8")
        dotenv.write_text(kept + "RKMON_CPUS=2\n", encoding="utf-8")
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual((rc, err), (4, ""), out)
        edited = finished(
            f"REFUSED: changed outside your file: docker/.env differs from what {self.w} last wrote there; nothing "
            "changed",
            f"next: {self.w} apply (--dry-run shows what it puts back)",
        )
        self.assertTrue(out.endswith(edited), out)
        self.assertEqual(dotenv.read_text(encoding="utf-8"), kept + "RKMON_CPUS=2\n")
        dotenv.write_text(kept, encoding="utf-8")
        self.write_settings(memory_gib=13)
        self.confirmed()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertIn("REFUSED: a change waits: you confirmed ", one_line(out))
        self.assertIn(
            ", and the stopped container has the settings before", one_line(out)
        )
        self.assertTrue(out.endswith(rewrapped(f"\nnext: {self.w} apply\n")), out)
        self.assertEqual(dotenv.read_text(encoding="utf-8"), kept)
        dotenv.write_text(kept + "RKMON_CPUS=2\n", encoding="utf-8")
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(out.endswith(edited), out)
        self.assertFalse(
            [
                c
                for c in self.calls()
                if "rk init" in c
                or "up -d" in c
                or "code-install" in c
                or "--entrypoint python3" in c
            ]
        )

    def test_a_start_shows_what_the_container_says_as_it_starts(self):
        """The container's entry point says rk start's lines in the container's log as it starts, the one-time move's
        report among them: a start that starts the container shows them, read from the log from the container's
        start on, up to rk start's line on the Claude side, each without the log's time and name; what comes after it,
        and a line the entry point does not say, it leaves to the log."""
        self.confirmed()
        stopped = self.d / "stopped"
        stopped.touch()
        at = "2026-11-15T09:00:05Z rkmon-container: "
        claude = (
            "The Claude side: the container's orchestrator (rk orchestrate) sends /rocke-monitor resume when it "
            "starts."
        )
        said = (
            "STARTED: the monitor runs again (it was stopped by rk stop at 2026-11-15T08:50:00Z)",
            "the one-time move: config.json is split by owner",
            claude,
        )
        log = "\n".join(
            (
                "a line the entry point does not say",
                *(at + line for line in said),
                f"{at}the loop that restarts the admin dashboard starts",
            )
        )
        since = (
            r"compose logs --follow --no-log-prefix --since 2026-11-15T09:00:00Z rkmon$"
        )
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                *STARTED,
                [since, log, 0],
            ]
        )
        env = {
            "FAKE_DOCKER_RUNNING": "1",
            "FAKE_DOCKER_STOPPED": str(stopped),
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        shown = "".join(f"  {line}\n" for line in said)
        self.assertIn(f"\nwhat the container says as it starts:\n{shown}", out)
        self.assertNotIn("the entry point does not say", out)
        self.assertNotIn("restarts the admin dashboard", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        calls = self.docker_calls()
        self.assertLess(calls.index("compose up -d"), calls.index(since[:-1]))

    def test_what_the_container_says_as_it_starts_is_waited_for_a_time_at_most(self):
        """A start waits for what the container says as it starts for a time at most (Wrapper.start_said, in this
        process): the log's reader stops then, and a log that has said nothing by then is left to rkmon logs, which a
        line of its own names."""
        self.confirmed()
        self.answer([*STARTED, [r"compose logs --follow ", "", 0, {}, {"sleeps": 60}]])
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        out = io.StringIO()
        with mock.patch.dict(
            os.environ, dict(self.env, **self.running())
        ), mock.patch.object(sys, "path", list(sys.path)):
            w = mod.Wrapper(self.w)
            with contextlib.redirect_stdout(out):
                w.start_said(seconds=2)
        self.assertRegex(
            out.getvalue(),
            r"\Athe container said nothing of its start within 2 seconds: \S+ logs shows what it says\n\Z",
        )

    def test_a_stop_then_a_start_after_a_code_only_update_starts(self):
        """A package update that changes code alone keeps the image, whose labels name the older package but which
        was built from this package's image recipe (its own copies of the files an image keeps are this package's:
        code_volume.IMAGE_BOUND): the image check passes it, and after a stop (state 12) the start puts this
        package's code in, a release over that image in a one-off container, before rk provision and the container's
        start, as apply puts it in: STARTED. What goes in is this package's code: the files an image's build copies
        (image.code_files), on code-install's input."""
        self.confirmed()
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", OTHER, 0],
                [r"--entrypoint python3", "TLS ok", 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [rf"{START_INSTALL}$", INSTALLED, 0],
            ]
        )
        env = self.stopped_after_a_start()
        rc, out, err = self.rkmon("image", "check", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("\nrocke-monitor:rkmon-test: package 1999.01.01, built ", out)
        self.assertIn(
            "\n  PASS  image: rocke-monitor:rkmon-test was built from this package's image recipe\n",
            out,
        )
        self.log.unlink()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"\n  code: {INSTALLED}\n", out)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings",
                    f"next: step 7, doctor: {self.w} doctor",
                )
            ),
            out,
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        order = [
            next(i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)
        self.assertTrue(calls[order[0]].startswith("compose run --rm "), calls)
        with tarfile.open(self.stdin) as tar:
            self.assertEqual(sorted(tar.getnames()), sorted(image.code_files(self.pkg)))

    def test_a_start_makes_its_running_container_again_once_a_release_went_in(self):
        """A container that runs with a stop left in it (state 12: rk stop --drain, and no rkmon stop after it) runs the
        code it started with: once the start's code-install put a release in, the container is made again (docker
        compose up -d --force-recreate, as apply's restart makes it), and the start says why; then the stop is
        cleared. While that code ran already (code-install's NOTHING TO DO), it stays as it runs.
        """
        self.confirmed()
        for said, up in (
            (INSTALLED, "compose up -d --force-recreate"),
            (OWN_CODE, "compose up -d"),
        ):
            with self.subTest(code=said):
                self.answer(
                    [
                        *self.delivered(),
                        [r"image inspect", GOOD, 0],
                        [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                        [r"compose ps -q rkmon$", "0123456789ab", 0],
                        [r"rk logins --json", "[]", 0],
                        [r"rk stopped$", "rk stop --drain", 0],
                        [rf"{START_INSTALL}$", said, 0],
                    ]
                )
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon("start", **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertIn(
                    "\nSTARTED: the monitor runs your confirmed settings\n", out
                )
                self.assertIn(
                    "the monitor was stopped (rk stop --drain): clearing it with rk start",
                    out,
                )
                remade = "the container runs the code from before: it is made again, so that it runs the code that"
                self.assertEqual(remade in out, said == INSTALLED, out)
                calls = [c.split("\t", 1)[1] for c in self.calls()]
                self.assertEqual(
                    [c for c in calls if c.startswith("compose up ")], [up], calls
                )
                installed = next(
                    i for i, c in enumerate(calls) if c.endswith(f" {START_INSTALL}")
                )
                self.assertLess(installed, calls.index(up), calls)

    def test_a_start_puts_in_no_code_where_a_rollback_put_code_back(self):
        """In state 10 (put back), the code that runs is the one the rollback or the put-back put back: the start puts
        in none (no code-install), so a private file in the package folder that is not its own refuses nothing, and
        it starts what is in place. That release's id ties its code to its image, so the start asks no image recipe
        of the image either: one whose own copies are not this package's starts, its code never read.
        """
        self.stray()
        self.confirmed()
        rows = self.delivered()
        fingerprint = self.confirm_record().fingerprint
        put_back = records.PutBack(
            fingerprint, "rollback", records.timestamp(), "", None
        )
        records.put_back(self.pkg, put_back)
        self.answer(
            [
                *self.image_code(changed=["docker/entrypoint.sh", "rk"]),
                *rows,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f"STARTED: the last change you confirmed was put back by {self.w} rollback, and the monitor runs the "
            "settings from before it next: ",
            one_line(out),
        )
        calls = [c.split("\t", 1)[1] for c in self.calls()]
        self.assertIn("compose up -d", calls)
        self.assertFalse([c for c in calls if "code-install" in c], calls)
        self.assertEqual(self.called(deploy_folder.IMAGE_CODE), [-1])

    def test_start_refuses_a_private_file_that_is_not_the_package_s(self):
        """Never started (state 8) or stopped (12), the start puts this package's code in, so a private file in the
        package folder that is not its own, which the release would carry, refuses it first (strays), under --dry-run
        too: FAILED, exit 4, with its row, before anything is delivered or goes in. Such a file, a file the release
        would carry, changes this package's code: after a confirm, the start asks for the confirm first (state 5).
        """
        self.confirmed()
        self.stray()
        self.answer([[r"image inspect", GOOD, 0]])
        rc, out, err = self.rkmon("start")
        self.assertEqual(rc, 3, out + err)
        self.assertIn(
            "\nREFUSED: this package changed after the last confirm; nothing changed\n",
            out,
        )
        self.confirmed()
        for state in (8, 12):
            rows, env = [], {}
            if state == 12:
                container = [r"compose ps -a -q rkmon$", "0123456789ab", 0]
                rows = [*self.delivered(), container, [r"rk stopped$", "", 1]]
                env = self.stopped_after_a_start()
            for dry in (("--dry-run",), ()):
                with self.subTest(state=state, dry=dry):
                    self.answer([*rows, [r"image inspect", GOOD, 0]])
                    self.log.unlink(missing_ok=True)
                    rc, out, err = self.rkmon(*dry, "start", **env)
                    self.assertEqual(rc, 4, out + err)
                    self.assertIn("FAIL  the package folder: ", out)
                    self.assertIn("docker/notes.yaml", out)
                    self.assertIn("\nFAILED: ", out)
                    self.assertNotIn("your confirmed settings go into", out)
                    nothing = ("code-install", LAYER_RUN, " rk provision$", "up -d")
                    self.assertEqual(self.called(*nothing), [-1] * len(nothing))
            if state == 8:
                self.assertFalse((self.pkg / deliver.ENV_FILE).exists())

    def test_start_says_the_code_that_did_not_go_in(self):
        """A code-install that fails ends the start (here after a stop, state 12): ERROR, exit 1, with code-install's
        own last line, which says what became of the code, the monitor not started; then the start again.
        """
        self.confirmed()
        emptied = [
            "code-install: the code volume failed its check (a release with no rk)",
            "code-install: emptied the code volume: the image's own code runs",
        ]
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [rf"{START_INSTALL}$", "\n".join(emptied), 1],
            ]
        )
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    f"ERROR: this package's code did not go in (code-install exited 1: {emptied[1]}), so the monitor "
                    "did not start",
                    f"next: {self.w} start",
                )
            ),
            out,
        )
        nothing = (" rk provision$", "rk init", "up -d")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))

    def test_a_start_whose_code_needs_a_new_image_is_refused(self):
        """Code that cannot run on the image (code-install's exit, code_volume.NEW_IMAGE) refuses the start (here after
        a stop, state 12) as it refuses apply: REFUSED, exit 4, with the fix that builds a new image
        (releases.new_image_fix), then confirm, and the state's next line; the monitor not started.
        """
        self.confirmed()
        self.answer(
            [
                *self.delivered(),
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [rf"{START_INSTALL}$", "", code_volume.NEW_IMAGE],
            ]
        )
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            " REFUSED: this package's code needs a new image, so the monitor did not start fix: build an image from "
            "the new package (QUICK_BUILD_GUIDE.md), then set image.tag in your deploy settings file to its tag, then "
            f"{self.w} confirm next: step 6, start: {self.w} start",
            one_line(out),
        )
        self.assertTrue(
            out.endswith(rewrapped(f"\nnext: step 6, start: {self.w} start\n")), out
        )
        nothing = (" rk provision$", "rk init", "up -d")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))

    def test_ctrl_c_during_the_start_ends_it_interrupted(self):
        """Ctrl-C during the start, from its first change on, exit 130: INTERRUPTED, which says where
        the start stopped, then the state's next line, which says what to run; no later step runs. Never started
        (state 8): while your confirmed settings went in, with what changed; while the work volume was set up; while
        the monitor's state was seeded. After a stop (state 12): while this package's code went in, once code-install
        has ended and said what it did (it runs to its end, so that its one-off container goes); a second Ctrl-C,
        while the state's next line is read, ends the start by the signal, with no traceback.
        """
        self.confirmed()
        stopped = self.d / "stopped"  # no container until the start's compose up
        stopped.touch()
        env = {"FAKE_DOCKER_RUNNING": "1", "FAKE_DOCKER_STOPPED": str(stopped)}
        both = [
            [r"image inspect", GOOD, 0],
            [r"--entrypoint python3", "", 0],
            [r"rk stopped$", "", 1],
        ]
        layer = deliver.texts(deploy_folder.plan(self.pkg, self.settings))[2]
        delivered = [LAYER_READ, layer, 0]
        went_in = (
            "putting your confirmed settings in place: the settings volume has either its layers as they were or the new one, "
            f"the earlier one kept; {deliver.ENV_FILE} and the override are as they were; their kept copies (.prev) "
            "are the files as they were"
        )
        for rows, said, later in (
            ([[LAYER_RUN, "", "SIGINT"]], went_in, ("code-install", " rk provision$")),
            (
                [delivered, [r" rk provision$", "", "SIGINT"]],
                "setting up the work volume",
                ("rk init", "up -d"),
            ),
            (
                [delivered, [r" rk init --if-new --on$", "", "SIGINT"]],
                "seeding the monitor's state",
                ("up -d",),
            ),
        ):
            with self.subTest(said=said):
                self.answer([*rows, *both])
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon("start", **env)
                self.assertEqual(rc, 130, out + err)
                self.assertTrue(
                    out.endswith(
                        finished(
                            f"INTERRUPTED: the start stopped while {said}",
                            f"next: step 6, start: {self.w} start",
                        )
                    ),
                    out,
                )
                self.assertEqual(self.called(*later), [-1] * len(later))
        ended = self.d / "ended"
        after_a_stop = [
            *self.delivered(),
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"rk stopped$", "", 1],
            [rf"{START_INSTALL}$", INSTALLED, "SIGINT", {}, {"ends": str(ended)}],
        ]
        self.answer(after_a_stop)
        self.log.unlink()
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, 130, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "INTERRUPTED: the start stopped while putting this package's code in",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        self.assertTrue(ended.exists())
        self.assertIn(f"\n  code: {INSTALLED}\n", out)
        self.assertIn("Ctrl-C: the command above runs to its end first", err)
        nothing = (" rk provision$", "rk init", "up -d")
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))
        # the second Ctrl-C: at the first docker call after code-install, the state's next line's
        self.answer(
            [["", "", "SIGINT", {}, {"after": rf"(?m){START_INSTALL}$"}], *after_a_stop]
        )
        self.log.unlink()
        rc, out, err = self.rkmon("start", **self.stopped_after_a_start())
        self.assertEqual(rc, -signal.SIGINT, out + err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("INTERRUPTED", out)
        self.assertEqual(self.called(*nothing), [-1] * len(nothing))

    def test_start_refuses_a_volume_of_another_deployment(self):
        """Check 15 at the start: a volume this deployment would use, as docker compose config names it,
        that another deployment's compose project made, or that another deployment's container mounts, refuses the
        start before any container runs on it (exit 4), with the fix that gives container.name a name of its own and
        no other folder's path; one that only this folder's container, no container or a container of no compose
        project mounts is this deployment's."""
        self.confirmed()
        work = f"{deploy_folder.NAME}_rkmon-work"
        here = os.path.realpath(self.pkg / "docker")
        config = {
            "name": deploy_folder.NAME,
            "volumes": {
                "rkmon-work": {"name": work},
                "rkmon-secrets": {"name": f"{deploy_folder.NAME}_rkmon-secrets"},
            },
        }

        def volume(project=None) -> dict:
            labels = f"com.docker.compose.project={project}" if project else ""
            return {"Driver": "local", "Labels": labels, "Name": work}

        def user(project, folder) -> dict:
            return {
                "Names": f"{project}-rkmon-1",
                "Labels": f"com.docker.compose.project={project},"
                f"com.docker.compose.project.working_dir={folder}",
            }

        def answers(found, users) -> list:
            return [
                [r"image inspect", GOOD, 0],
                [r"^compose config --format json$", json.dumps(config, indent=2), 0],
                [r"^volume ls ", json.dumps(found), 0],
                [
                    rf"^ps --all --filter volume={re.escape(work)} ",
                    "".join(json.dumps(u) + "\n" for u in users),
                    0,
                ],
            ]

        for case, found, users in (
            ("made by another deployment", volume("other-deployment"), []),
            (
                "mounted by another deployment's container",
                volume(deploy_folder.NAME),
                [user("other-deployment", "/srv/other/docker")],
            ),
        ):
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.answer(answers(found, users))
                rc, out, err = self.rkmon("start")
                self.assertEqual((rc, err), (4, ""), out)
                self.assertIn(
                    f"FAIL volume: {work} is used by another deployment on this host fix: set container.name in "
                    f"~/rkmon-deploy.yaml to a name of its own, such as {deploy_folder.NAME}-2 (docker compose ls "
                    "--all lists the names in use)",
                    one_line(out),
                )
                self.assertTrue(
                    out.endswith(
                        finished(
                            "REFUSED: a volume this deployment would use is another deployment's; nothing changed "
                            "(fix each FAIL above)",
                            f"next: step 6, start: {self.w} start",
                        )
                    ),
                    out,
                )
                self.assertNotIn("/srv/other", out)
                self.assertFalse(
                    [c for c in self.calls() if " run " in c or c.endswith("up -d")]
                )
                # each volume listed by its exact name: docker's name filter is a pattern that would also match a
                # part of another volume's name
                self.assertIn(
                    f"volume ls --filter name=^{deploy_folder.NAME}_rkmon-secrets$ --filter name=^{work}$ "
                    "--format {{json .}}",
                    [c.split("\t", 1)[1] for c in self.calls()],
                )
        for case, found, users in (
            (
                "mounted by this folder's container",
                volume(deploy_folder.NAME),
                [user(deploy_folder.NAME, here)],
            ),
            ("mounted by none", volume(), []),
            (
                "mounted by a container of no compose project",
                volume(),
                [{"Names": "backup", "Labels": ""}],
            ),
        ):
            with self.subTest(case=case):
                self.answer(answers(found, users))
                rc, out, err = self.rkmon("--dry-run", "start")
                self.assertEqual((rc, err), (0, ""), out)
                self.assertIn("PASSED: the monitor can start", out)

    def test_the_first_tick_waits_for_a_tick_after_the_start(self):
        """Deploy step 8: it runs in its deployment states while the container runs: with none made, the
        state's refusal (state 8: its sentence, and the start, exit 3); with one stopped, what the command needs, and
        the state's next line (state 12: exit 4). Once a tick has run since the start, rk status, then PASSED with its
        time and the done line, which names each feature that is on with its mode, and no tick asked; no tick in time:
        FAILED, with the commands that say why, then the command again (exit 4). While none has run since the start,
        it asks the tick loop for a tick at once (rk tick-now), once, and says what came of it: asked, asked of the
        next loop to start, or, when rk tick-now cannot write its request (its ERROR), no tick asked, and why; while
        the monitor in the container is stopped, nothing asked, REFUSED with the state's next line (exit 4).
        """
        self.confirmed()
        running = [
            [r"image inspect", GOOD, 0],
            [r"rk logins --json", "[]", 0],
            [r"rk stopped", "", 1],
            [r"compose ps -q", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
        ]
        self.answer(running)
        rc, out, err = self.rkmon("first-tick")
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: this deployment's container does not exist yet; nothing changed",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        made = [[r"compose ps -a -q", "0123456789ab", 0]]
        self.answer(self.delivered() + made + running)
        rc, out, err = self.rkmon("first-tick")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the container is not running; nothing changed",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk tick-due" in c])
        tick = [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0]
        self.answer(running + [tick])
        rc, out, err = self.rkmon("first-tick", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("rk status", err)
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: the first tick ran at 2026-11-15T09:05:00Z",
                    "done: the monitor runs (review automation prototype, merge queue prototype, the admin "
                    "dashboard). Your first day: QUICK_USER_GUIDE.md, Your first day",
                )
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk tick-now" in c])
        never = [r"rk tick-due", "last tick never", 1]
        unwritten = "/work/state/tick_now could not be written ([Errno 28] No space left on device)"
        for said, rc_now, line in (
            (
                "ASKED: a tick at once",
                0,
                "asked for a tick at once: the tick loop starts it within seconds",
            ),
            (
                "WAITING: a tick at once",
                3,
                "asked for a tick at once: no tick loop runs now; the next one",
            ),
            (
                f"ERROR: {unwritten}; no tick asked",
                1,
                f"no tick asked ({unwritten}): the tick loop ticks at its own time",
            ),
        ):
            with self.subTest(said=said):
                self.log.unlink(missing_ok=True)
                self.answer(running + [never, [r"rk tick-now$", said, rc_now]])
                rc, out, err = self.rkmon(
                    "first-tick", "--timeout-minutes", "0.001", FAKE_DOCKER_RUNNING="1"
                )
                self.assertEqual(rc, 4, out + err)
                self.assertIn(line, one_line(out))
                self.assertNotIn(said, out)
                self.assertTrue(
                    one_line(out).endswith(
                        f"FAILED: no tick in 0.001 minutes fix: {self.w} logs --tail 50 and {self.w} rk status say "
                        f"why next: {self.w} first-tick --timeout-minutes 0.001"
                    ),
                    out,
                )
                self.assertEqual(
                    len([c for c in self.calls() if "rk tick-now" in c]), 1
                )
        # the monitor in the container is stopped: rk stopped says so once rk tick-now has refused
        self.log.unlink(missing_ok=True)
        refused = "REFUSED: the monitor is stopped (rk stop, 2026-11-15T09:01:00Z); rk start resumes it; no tick asked"
        stopped = [r"rk stopped", "", 0, {}, {"after": r"rk tick-now"}]
        self.answer(
            [stopped, *self.delivered(), *running, never, [r"rk tick-now$", refused, 4]]
        )
        rc, out, err = self.rkmon("first-tick", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the monitor in the container is stopped; nothing changed",
                    f"next: {self.w} start",
                )
            ),
            out,
        )
        self.assertNotIn("asked for a tick", out)

    def test_the_first_tick_ends_with_the_states_line_outside_state_16(self):
        """Deploy step 8's PASSED: in state 16 the deploy path's done line; in any other, the state's next
        line as rkmon next gives it, here a paused monitor's (state 15)."""
        self.confirmed()
        layer, applied = self.delivered()
        running = [
            [r"image inspect", GOOD, 0],
            [r"rk logins --json", "[]", 0],
            [r"rk stopped$", "", 1],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"compose ps -q rkmon$", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
            [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0],
        ]
        done = (
            "done: the monitor runs (review automation prototype, merge queue prototype, the admin dashboard). Your "
            "first day: QUICK_USER_GUIDE.md, Your first day"
        )
        ran = "PASSED: the first tick ran at 2026-11-15T09:05:00Z"
        for paused, want in ((False, done), (True, f"next: {self.w} resume")):
            with self.subTest(paused=paused):
                said = dict(json.loads(applied[1]), paused=paused)
                self.answer([layer, [applied[0], json.dumps(said), 0], *running])
                rc, out, err = self.rkmon("first-tick", **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(out.endswith(finished(ran, want)), out)
        rc, out, err = self.rkmon("next", **self.running())
        self.assertTrue(out.endswith("\n" + finished(f"next: {self.w} resume")), out)


if __name__ == "__main__":
    unittest.main()
