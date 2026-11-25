# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""docker/rkmon, the container's one command, against a stand-in for docker (tests/fakes/fake_docker) on a PATH of
ordinary tools only, so no test ever reaches a real docker: each command maps to its docker compose command, printed
first, and --dry-run prints them and runs none; a command for the container runs in it when it is up (exec), else in a
one-off container (run --rm), and an exec runs as the monitor's user (the writer's logins and secrets: as the writer,
never through sudo); a terminal gets one (-it), none otherwise (-T); it works from any folder and from a link on the
PATH, and sends a settings file in on stdin (never a mount); rkmon rk exits with docker's exit code, and start, stop
and doctor end with one outcome. A step of the deploy path starts with its banner. With no settings
confirmed in the package folder, as in the package itself, a start is the state's refusal; what a start runs is
tests/test_deploy_path_start.py's, which confirms them. A command of the deploy side runs on a deployment, a copy of the
package whose settings are confirmed and delivered (tests/deploy_folder.py), and only in the deployment states its row
names: elsewhere it refuses with the state's sentence, or with what it needs of the
container, and the state's next line. rkmon status puts the deployment's state first, and rkmon doctor shows it as a
row and keeps its result for rkmon next. --image-code runs rk on the image's own code, and a container another folder
made is never touched."""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
import sandbox  # noqa: E402
from rendered import finished, one_line, state_2_next  # noqa: E402
from rkmon_lib import (
    build_plan,
    checks,
    commands,
    confirm,
    container_layout,
    file_types,
    identity,
    image,
    records,
    settings_file,
    states,
    steps,
    wrapping,
)  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402

WRAPPER = HERE.parent / "docker" / "rkmon"
DOCKER_DIR = str(WRAPPER.parent)
ENTRY = f"{container_layout.IMAGE_CODE}/docker/entrypoint.sh"
PS = "compose ps --status running --quiet rkmon"  # the read-only question: is the monitor's container up?
# The first line of start, doctor and first-tick under --dry-run: their steps' banners.
START = steps.banner(steps.DEPLOY_STEPS[5], dry_run=True)
DOCTOR = steps.banner(steps.DEPLOY_STEPS[6], dry_run=True)
FIRST_TICK = steps.banner(steps.DEPLOY_STEPS[7], dry_run=True)
# docker's read-only questions about a deployment (rkmon_lib/state_facts.py's, through docker/rkmon): whose the
# container name is, the image, whether the container runs or exists, its volumes, and, in it or a one-off container,
# the logins, whether rk stop stopped the monitor, whether its work volume is set up, the delivered layer and what the
# monitor loaded
READ_ONLY = re.compile(
    r"^(ps --all --filter |image inspect |inspect --format |compose ps |compose config (--no-interpolate )?--format "
    r"json$|volume ls "
    r"|compose (run --rm -T|exec -T --user rkmon) .*(rk (logins --json|stopped|applied --json|provision --check)$"
    r"|--entrypoint cat )"
    # the one-off container that reads the volumes that keep the logins before the first start: no network, and
    # only volumes that exist, mounted read-only
    r"|run --rm --network none (--cap-(drop|add) \S+ |--pids-limit \d+ )*"
    r"(--mount type=volume,source=\S+,target=\S+,readonly,volume-nocopy )+--entrypoint python3 )"
)
# rk drain's lines as it drains (rkmon_lib/drain.py), the work in flight, two reviews, finished
DRAINED = (
    "drain: nothing new starts; the work in flight finishes first (up to 60 minutes): 2 reviews",
    "DRAINED: nothing is in flight after 12 minutes",
)
# A program that runs the wrapper's run_whole (argv[1]: the wrapper) under ctrl_c_interrupts, of a child that sends it
# Ctrl-C (SIGINT) argv[2] times, then makes the file argv[3] a second later; once the run ends, it says whether the
# press was taken after the child made its file.
WHOLE_RUN = """
import importlib.machinery, importlib.util, os, subprocess, sys
loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", sys.argv[1])
mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("rkmon_wrapper", loader))
loader.exec_module(mod)
child = (
    "import os, signal, sys, time\\n"
    "for _ in range(int(sys.argv[1])):\\n"
    "    os.kill(os.getppid(), signal.SIGINT)\\n"
    "    time.sleep(0.2)\\n"
    "time.sleep(1)\\n"
    "open(sys.argv[2], 'w').close()\\n"
)
mod.ctrl_c_interrupts()
try:
    mod.run_whole([sys.executable, "-c", child, *sys.argv[2:]], stdout=subprocess.DEVNULL)
except KeyboardInterrupt:
    print("taken once it ended" if os.path.exists(sys.argv[3]) else "taken before it ended")
"""


# the deploy settings' model binding (deploy) whose key is a secret no fixed entry names: rkmon secret takes the key's
# name once settings that name it are confirmed
GLM_BINDINGS = {
    "glm": {
        "provider": "openai_compatible",
        "base_url": "https://models.example.com/v1",
        "model_id": "glm-model",
        "api_key_secret_name": "glm_api_key",
    }
}


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.bin = sandbox.safe_path(self.base)
        shutil.copy(HERE / "fakes" / "fake_docker", pathlib.Path(self.bin) / "docker")
        (pathlib.Path(self.bin) / "docker").chmod(0o755)
        self.log = self.base / "docker.log"
        self.cwd = self.base / "elsewhere"
        self.cwd.mkdir()
        (self.cwd / "my-settings.yaml").write_text("config: {}\n", encoding="utf-8")
        (self.cwd / "my-settings.yaml").chmod(0o600)  # a private file, yours only
        self.env = {
            "PATH": self.bin,
            "HOME": str(self.base / "home"),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
        }
        self.w = str(WRAPPER)
        self.docker_dir = DOCKER_DIR  # the folder docker runs in: the package's docker/, or the deployment's (deploy)
        self.answers = self.base / "answers.json"
        self.ready = (
            []
        )  # the stand-in docker's answers about the deployment (deploy), after a test's own

    def answer(self, rows=()) -> None:
        """The stand-in docker answers ROWS ([pattern, output, exit code]: tests/fakes/fake_docker) first, then as the
        deployment is (deploy)."""
        self.answers.write_text(json.dumps([*rows, *self.ready]), encoding="utf-8")
        self.env["FAKE_DOCKER_ANSWERS"] = str(self.answers)

    def deploy(self, made: bool = True, bindings=None, features=None) -> None:
        """The commands from here on run on a deployment (tests/deploy_folder.py): a copy of the package whose deploy
        settings, ~/rkmon-deploy.yaml, are confirmed and delivered as the first start delivers them. MADE: its
        container exists, this folder's, and runs while a call says FAKE_DOCKER_RUNNING=1, else it is stopped (state
        12); else there is none (state 8). BINDINGS: the settings' model bindings, none by default; FEATURES: their
        features over deploy_folder.FEATURES.
        """
        pkg = deploy_folder.copy(self.base / "pkg")
        home = pathlib.Path(self.env["HOME"])
        home.mkdir(exist_ok=True)
        self.settings = home / "rkmon-deploy.yaml"
        deploy_folder.write_settings(
            self.settings, sandbox.free_port(), features=features, bindings=bindings
        )
        deploy_folder.confirm(pkg, self.settings, home)
        # the delivered layer, which the monitor loaded; and rk stop has not stopped it
        self.ready = [
            *deploy_folder.delivered(pkg, self.settings),
            [r"rk stopped$", "", 1],
        ]
        self.pkg, self.w = pkg, str(pkg / "docker" / "rkmon")
        self.docker_dir = os.path.realpath(pkg / "docker")
        if made:
            self.make()
        self.answer()

    def make(self) -> None:
        """The deployment's container exists (deploy): docker says so, and whose its name is, this folder's."""
        self.ready.append([r"compose ps -a -q rkmon$", "0123456789ab", 0])
        self.env.update(
            FAKE_DOCKER_PROJECT=deploy_folder.NAME, FAKE_DOCKER_WORKDIR=self.docker_dir
        )
        self.answer()

    def after_removal(self, *rows) -> None:
        """Once a docker compose down is done (FAKE_DOCKER_DOWN), the stand-in docker answers ROWS first, then as the
        deployment is (deploy) with no container: none has its name, and it is not made.
        """
        after = self.base / "after_removal.json"
        gone = [r"^ps --all --filter ", "", 0]
        ready = [r for r in self.ready if "compose ps -a" not in r[0]]
        after.write_text(json.dumps([*rows, gone, *ready]), encoding="utf-8")
        self.env["FAKE_DOCKER_DOWN"] = str(after)

    def acted(self) -> list:
        """The docker calls that were not one of a deployment's read-only questions (READ_ONLY)."""
        return [c for c in self.calls() if not READ_ONLY.match(c[1])]

    def rkmon(self, *args, **env):
        p = subprocess.run(
            [self.w, *args],
            cwd=self.cwd,
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        return p.returncode, p.stdout, p.stderr

    def calls(self) -> list[tuple[str, str]]:
        """The stand-in docker's calls (FAKE_DOCKER_LOG): each one's folder and its arguments, joined by spaces; an
        argument of several lines, as the program a one-off container's python3 runs, keeps them.
        """
        if not self.log.is_file():
            return []
        out = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            if out and "\t" not in line:
                out[-1] = (out[-1][0], f"{out[-1][1]}\n{line}")
            else:
                out.append(tuple(line.split("\t", 1)))
        return out

    def dry_run_end(self, text: str, *typed: str) -> list:
        """The lines a dry run ends with once its guards pass: PASSED with TEXT, what the real run would do,
        then the command you typed (TYPED) without --dry-run."""
        return finished(
            f"PASSED: {text}", f"next: {self.w} {' '.join(typed)}"
        ).splitlines()

    def test_with_no_settings_confirmed_a_command_that_needs_them_is_refused(self):
        """With no settings confirmed, this folder has no compose project: a command that needs one (start) is state
        2's refusal, status says the state, and docker is asked nothing."""
        # status, which only reads, says the same under --dry-run: the state first; no container runs to ask.
        rc, out, err = self.rkmon("--dry-run", "status")
        self.assertEqual(
            (rc, out.splitlines(), err),
            (
                0,
                [
                    "deployment: none confirmed yet (this package folder) · no container",
                    "state: no deploy settings are confirmed in this package folder",
                    f"next: step 1, template: {self.w} template deploy ~/rkmon-deploy.yaml",
                    f"(have a file already? {self.w} next --settings FILE)",
                ],
                "",
            ),
        )
        # The start, with no settings confirmed here: the state's refusal (state 2, an earlier step), under
        # --dry-run before the command or after it, as without.
        refused = (
            START
            + "\n"
            + finished(
                "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                state_2_next(self.w),
            )
        )
        for args in (("--dry-run", "start"), ("start", "--dry-run")):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args)
                self.assertEqual((rc, out, err), (3, refused, ""))
        self.assertEqual(self.calls(), [])

    def test_setup_is_an_unknown_command_that_names_what_replaces_it(self):
        """Setup, with any of its options, before a deployment or on one, is a usage error that names your
        deploy settings file, login and secret, and, with Claude Code, /rkmon-setup; it changes nothing.
        """
        for deployed in (False, True):
            if deployed:
                self.deploy()  # the wrapper of the deployment's own package folder (self.w)
            said = (
                f'{self.w}: unknown command "setup": set up a deployment with your deploy settings file, {self.w} '
                f"login and {self.w} secret, as {self.w} help deploy-path shows; with Claude Code, /rkmon-setup "
                "helps you write the file\n"
            )
            for args in (
                ("setup",),
                ("--dry-run", "setup"),
                ("setup", "--section", "gh"),
                ("setup", "--settings", "my-settings.yaml"),
            ):
                with self.subTest(deployed=deployed, args=args):
                    self.assertEqual(self.rkmon(*args), (2, "", said))
        self.assertEqual(self.acted(), [])

    def test_on_a_deployment_each_command_maps_to_its_docker_compose_command(self):
        """On a deployment whose container is stopped (state 12), as a dry run: each command of the deploy side that
        runs there is its docker compose command, a command for the container in a one-off container; stop has nothing
        to do, then the state's next line; shell and rk, which run in the running container alone, refuse with what
        they need and the state's next line. None of the commands shown ran. No container runs, so nothing has
        passed since a start: the start's command carries its step."""
        self.deploy()
        self.answer(deploy_folder.volumes(labeled=deploy_folder.VOLUMES))
        e = f"-e RKMON_WRAPPER={self.w}"
        start = f"next: step 6, start: {self.w} start"
        cases = {
            ("login", "github_write"): [
                "docker compose run --rm -T -e RKMON_AS=rkwrite rkmon gh-write auth login --scopes read:org,repo",
                *self.dry_run_end(
                    "the login github_write can be set", "login", "github_write"
                ),
            ],
            # the writer's login, into its volume
            ("login", "atlassian"): [
                "docker compose run --rm -T -e RKMON_AS=rkwrite rkmon atlassian auth login",
                *self.dry_run_end(
                    "the login atlassian can be set", "login", "atlassian"
                ),
            ],
            # the read login: rk read-login in the walk's one-off container, handed your confirmed settings, as the
            # monitor, whose login it is
            ("login", "github_read"): [
                f"docker compose run --rm -T -e RKMON_CONFIRMED_LAYER {e} rkmon rk read-login",
                *self.dry_run_end("github_read can be set", "login", "github_read"),
            ],
            # the container does not run: nothing to stop, and the next line is the state's
            ("stop",): finished(
                "NOTHING TO DO: the container is not running", start
            ).splitlines(),
            ("doctor",): [
                DOCTOR,
                f"docker compose run --rm -T {e} rkmon rk doctor",
                *self.dry_run_end("rk doctor can run in the container", "doctor"),
            ],
            ("logs", "-f"): ["docker compose logs -f rkmon"],
            # the log's options are docker compose's own
            ("logs", "--since", "42m", "--tail", "100"): [
                "docker compose logs --tail 100 --since 42m rkmon"
            ],
            ("remove",): [
                "docker compose down",
                *self.dry_run_end("the container can be removed", "remove"),
            ],
            # its volumes, each by its name; no question under --dry-run
            ("remove", "--delete-data"): [
                "docker compose down",
                f"docker volume rm {' '.join(sorted(deploy_folder.VOLUMES))}",
                *self.dry_run_end(
                    "the container and its volumes can be removed",
                    "remove",
                    "--delete-data",
                ),
            ],
        }
        for args, want in cases.items():
            with self.subTest(args=args):
                rc, out, err = self.rkmon("--dry-run", *args)
                self.assertEqual((rc, out.splitlines(), err), (0, want, ""))
        refused = finished(
            "REFUSED: the container is not running; nothing changed", start
        )
        for args in (("shell",), ("rk", "pause"), ("rk", "usage", "--by-review")):
            with self.subTest(args=args):
                rc, out, err = self.rkmon("--dry-run", *args)
                self.assertEqual((rc, out, err), (4, refused, ""))
        shown = {w[len("docker ") :] for want in cases.values() for w in want}
        self.assertFalse(shown & {c[1] for c in self.calls()})
        self.assertTrue(all(c[0] == self.docker_dir for c in self.calls()))

    def test_each_command_refuses_outside_its_states(self):
        """Each command of the deploy side runs only in the deployment states its row names (commands.Row.states),
        and only while the container exists or runs, as its row needs (Row.container):
        elsewhere REFUSED with the state's sentence, or with what it needs, then the state's next line; exit 3 in the
        states of an earlier step (2, and 4 to 8), else 4; and docker is asked its read-only questions alone. Each
        command that refuses so, in states 2 (no settings confirmed), 8 (the container never made), 12 (made, and
        stopped), 6 (no image), 5 (your file edited after the confirm), 4 (your file in an earlier layout), 3 (an
        apply stopped part-way, then an adopt, each refusing the other: states.runs_here), 14 (standby, the container
        running: start and adopt aside, which say their own while it runs) and 1 (another deployment's container
        name). A next command that is a deploy step's carries its step: no container runs, or, in 14, the doctor
        has not passed since the start. Before the first start (state 8), apply has nothing to do (its own tests), so
        it is left out there."""
        rows = [r for r in commands.TABLE if r.states]

        def words(row) -> list:
            """ROW's words, with a value for each argument and option it requires: for secret's NAME a secret's, as a
            name no secret has is a usage error once settings are confirmed, said before the state's refusal.
            """
            value = "teams_webhook" if row.name == "secret" else "x"
            typed = [*row.words, *(value for a in row.args if not a.optional)]
            return typed + [
                w for o in row.options if o.required for w in (o.flags[-1], "x")
            ]

        def refused(
            n: int,
            sentence: str,
            next_line: str,
            made: bool = True,
            under_way: str = "",
            running: bool = False,
        ) -> None:
            """In state N, each row that refuses there: one whose states leave N out, or that does not finish
            UNDER_WAY, the command that stopped part-way, in state 3 (states.runs_here), with SENTENCE, the state's;
            one whose need of the container fails (its container stopped, or none MADE), with that need. RUNNING: the
            container runs, and start and adopt, which say their own while it does (state_guard's RUNNING; their own
            tests), are left out.
            """
            self.log.unlink(missing_ok=True)
            env = {"FAKE_DOCKER_RUNNING": "1"} if running else {}
            checked = []
            for row in rows:
                if running and row.name in ("start", "adopt"):
                    continue
                if n == 8 and row.name == "apply":
                    continue  # nothing to do before the first start: its own tests
                if not states.runs_here(row.name, row.states, n, under_way):
                    said = sentence
                elif not running and (
                    row.container == commands.RUNS
                    or (row.container == commands.EXISTS and not made)
                ):
                    said = commands.NEEDS[row.container]
                else:
                    continue
                checked.append(row.name)
                with self.subTest(state=n, command=row.name):
                    rc, out, err = self.rkmon(*words(row), **env)
                    self.assertEqual(
                        (rc, err), (3 if n in (2, 4, 5, 6, 7, 8) else 4, ""), out
                    )
                    self.assertIn(f"REFUSED: {said}; nothing changed", one_line(out))
                    self.assertTrue(one_line(out).endswith(one_line(next_line)), out)
            self.assertTrue(checked, n)
            self.assertEqual(self.acted(), [])

        refused(
            2,
            "no deploy settings are confirmed in this package folder",
            state_2_next(self.w),
        )
        self.deploy(made=False)
        w, file = self.w, "~/rkmon-deploy.yaml"
        refused(
            8,
            "this deployment's container does not exist yet",
            f"next: step 6, start: {w} start",
            made=False,
        )
        self.make()
        refused(12, "the container is stopped", f"next: step 6, start: {w} start")
        self.answer([[r"image inspect", "", 1]])
        refused(
            6,
            "the image rocke-monitor:rkmon-test is not on this host",
            f"next: step 4, image: {w} image load IMAGE_FILE",
        )
        self.answer()
        kept = self.settings.read_bytes()
        self.settings.write_bytes(kept.replace(b"memory_gib: 12", b"memory_gib: 13", 1))
        refused(
            5,
            "your settings file changed after the last confirm",
            f"next: step 3, confirm: {w} confirm",
        )
        layout = file_types.DEPLOY.version
        self.settings.write_bytes(
            kept.replace(f"schema_version: {layout}".encode(), b"schema_version: 1", 1)
        )
        refused(
            4,
            f"{file} is in layout 1, older than this package's layout {layout}",
            f"next: {w} migrate --settings {file}",
        )
        self.settings.write_bytes(kept)
        records.begin(self.pkg, "apply").release()
        refused(
            3,
            f"{w} apply stopped part-way",
            f"next: {w} apply (it finishes the change, or puts the previous settings back on a new failure)",
            under_way="apply",
        )
        records.begin(self.pkg, "apply").finish()
        records.begin(self.pkg, "adopt").release()
        refused(
            3,
            f"{w} adopt stopped part-way",
            f"next: {w} adopt --from DIR (from the same copy, it finishes putting the earlier monitor's state in "
            "place)",
            under_way="adopt",
        )
        records.begin(self.pkg, "adopt").finish()
        # the container runs, and the monitor says it is in standby (rk applied --json)
        applied = next(r for r in self.ready if r[0] == r"rk applied --json")
        said = dict(json.loads(applied[1]), standby=True)
        self.answer([[applied[0], json.dumps(said), 0]])
        refused(
            14,
            f"the monitor runs your confirmed settings, in standby: it sends nothing until you run {w} standby off",
            f"next: step 7, doctor: {w} doctor",
            running=True,
        )
        self.answer()
        self.env["FAKE_DOCKER_WORKDIR"] = "/srv/other/docker"
        refused(
            1,
            f"the container name {deploy_folder.NAME} is used by another deployment on this host",
            f"next: step 3, confirm: {w} confirm --settings {file}",
        )

    def test_with_the_container_up_its_commands_run_in_it(self):
        self.deploy(bindings=GLM_BINDINGS)
        e = f"-e RKMON_WRAPPER={self.w}"
        for args, want in (
            # an exec runs as the monitor's user
            (
                ("stop",),
                [
                    "docker compose stop",
                    *self.dry_run_end("the container can stop", "stop"),
                ],
            ),
            # the drain, rk stop and the container's stop
            (
                ("stop", "--drain"),
                [
                    f"docker compose exec -T --user rkmon {e} -e PYTHONUNBUFFERED=1 rkmon rk drain --timeout-minutes "
                    "60",
                    f"docker compose exec -T --user rkmon {e} rkmon rk stop",
                    "docker compose stop",
                    *self.dry_run_end(
                        "the work in flight can finish, then the container can stop",
                        "stop",
                        "--drain",
                    ),
                ],
            ),
            (
                ("doctor",),
                [
                    DOCTOR,
                    f"docker compose exec -T --user rkmon {e} rkmon rk doctor",
                    *self.dry_run_end("rk doctor can run in the container", "doctor"),
                ],
            ),
            (
                ("first-tick",),
                [
                    FIRST_TICK,
                    "docker compose exec -T rkmon rk tick-due --minutes 0",
                    "docker compose exec -T rkmon rk tick-now",
                    *self.dry_run_end(
                        "it asks for a tick at once while none has run since the start, then waits up to "
                        f"{commands.FIRST_TICK_TIMEOUT_MINUTES} minutes for the first tick after the start",
                        "first-tick",
                    ),
                ],
            ),
            (
                ("rk", "kill"),
                [f"docker compose exec -T --user rkmon {e} rkmon rk kill"],
            ),
            (
                ("shell",),
                [f"docker compose exec -T --user rkmon {e} rkmon {ENTRY} shell"],
            ),
            # the writer's logins: as the writer itself, its rk the broker's (never through sudo)
            (
                ("login", "github_write"),
                [
                    f"docker compose exec -T --user rkwrite rkmon {ENTRY} gh-write auth login --scopes read:org,repo",
                    *self.dry_run_end(
                        "the login github_write can be set", "login", "github_write"
                    ),
                ],
            ),
            (
                ("login", "atlassian"),
                [
                    f"docker compose exec -T --user rkwrite rkmon {ENTRY} atlassian auth login",
                    *self.dry_run_end(
                        "the login atlassian can be set", "login", "atlassian"
                    ),
                ],
            ),
            # the read login: rk read-login in the walk's one-off container, as the monitor, whose login it is
            (
                ("login", "github_read"),
                [
                    f"docker compose run --rm -T -e RKMON_CONFIRMED_LAYER {e} rkmon rk read-login",
                    *self.dry_run_end("github_read can be set", "login", "github_read"),
                ],
            ),
            # a secret: in the walk's one-off container, handed your confirmed settings, while the container runs
            # too; one the writer keeps as the writer itself, its rk the broker's, which shows the wrapper's name
            (
                ("secret", "teams_webhook", "--remove"),
                [
                    f"docker compose run --rm -T -e RKMON_CONFIRMED_LAYER {e} -e RKMON_AS=rkwrite rkmon rk secret-set "
                    "teams_webhook --remove",
                    *self.dry_run_end(
                        "teams_webhook can be removed",
                        "secret",
                        "teams_webhook",
                        "--remove",
                    ),
                ],
            ),
            (
                ("secret", "glm_api_key", "--remove"),
                [
                    f"docker compose run --rm -T -e RKMON_CONFIRMED_LAYER {e} rkmon rk secret-set glm_api_key "
                    "--remove",
                    *self.dry_run_end(
                        "glm_api_key can be removed",
                        "secret",
                        "glm_api_key",
                        "--remove",
                    ),
                ],
            ),
        ):
            with self.subTest(args=args):
                rc, out, _err = self.rkmon("--dry-run", *args, FAKE_DOCKER_RUNNING="1")
                self.assertEqual((rc, out.splitlines()), (0, want))

    def test_it_runs_docker_in_its_folder_prints_the_command_and_returns_its_exit_code(
        self,
    ):
        # No settings confirmed: nothing runs, and the refusal is the deployment state's, state 2's.
        rc, out, err = self.rkmon("start")
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                    state_2_next(self.w),
                )
            ),
            out,
        )
        self.assertFalse(any(" up " in c[1] or "rk init" in c[1] for c in self.calls()))
        # On a deployment, rkmon rk passes rk's code through, from the running container.
        self.deploy()
        rc, out, err = self.rkmon(
            "rk", "doctor", FAKE_DOCKER_RUNNING="1", FAKE_DOCKER_EXIT="7"
        )
        self.assertEqual(rc, 7)
        ran = f"compose exec -T --user rkmon -e RKMON_WRAPPER={self.w} rkmon rk doctor"
        self.assertEqual(err.splitlines(), [f"docker {ran}"])
        self.assertEqual(self.calls()[-1], (self.docker_dir, ran))
        # A login set: SET, then the deployment state's next line: here state 12's, the container stopped.
        rc, out, err = self.rkmon("login", "github_write")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    "SET: github_write; the next GitHub write uses it",
                    f"next: step 6, start: {self.w} start",
                ),
            ),
            err,
        )

    def test_stop_says_what_it_did(self):
        """rkmon stop: STOPPED, then the deployment state's next line, here the stopped container's (state 12);
        NOTHING TO DO while the container does not run (the dry runs show it); ERROR when docker cannot stop it, then
        the command again.
        """
        self.deploy()
        rc, out, err = self.rkmon(
            "stop",
            FAKE_DOCKER_RUNNING="1",
            FAKE_DOCKER_STOPPED=str(self.base / "stopped"),
        )
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            finished(
                "STOPPED: the monitor's container",
                f"next: step 6, start: {self.w} start",
            ),
        )
        self.assertIn((self.docker_dir, "compose stop"), self.calls())
        rc, out, err = self.rkmon("stop", FAKE_DOCKER_RUNNING="1", FAKE_DOCKER_EXIT="1")
        self.assertEqual(rc, 1, out + err)
        self.assertEqual(
            out,
            finished(
                "ERROR: docker compose stop exited 1 (the lines above say why)",
                f"next: {self.w} stop",
            ),
        )

    def drained(self, *rows, minutes: str = "60") -> None:
        """The stand-in docker answers ROWS first; then rk drain for MINUTES, once the work in flight, two reviews,
        finished; rk stop, rk undrain and rk start, as each says once it is done; then as the deployment is (deploy).
        """
        self.answer(
            [
                *rows,
                [
                    rf"rk drain --timeout-minutes {re.escape(minutes)}$",
                    "\n".join(DRAINED),
                    0,
                ],
                [
                    r" rk stop$",
                    "STOPPED: 1 process(es); nothing new starts until rk start.",
                    0,
                ],
                [r" rk undrain$", "UNDRAINED: the drain since 13:58 is lifted", 0],
                [r" rk start$", "STARTED: the monitor runs again", 0],
            ]
        )

    def stop_calls(self) -> list:
        """The calls of stop --drain that change something (acted): rk drain, rk undrain, rk stop, rk start and docker
        compose stop, each by its last words."""
        last = re.compile(r"(rk (drain|undrain|stop|start)\b.*|^compose stop)$")
        return [m.group(1) for c in self.acted() if (m := last.search(c[1]))]

    def test_stop_drain_lets_the_work_in_flight_finish_first(self):
        """rkmon stop --drain: rk drain in the running container, its lines said as they come, indented,
        then rk stop in it, which records the stop and lifts the drain in one write, then docker compose stop:
        STOPPED, then the state's next line (state 12). --timeout-minutes is the drain's limit (60 minutes by
        default); without --drain it is a usage error, and nothing is asked. While the container does not run:
        NOTHING TO DO, and no drain."""
        self.deploy()
        self.drained(minutes="15")
        stopped = self.base / "stopped"
        rc, out, err = self.rkmon(
            "stop",
            "--drain",
            "--timeout-minutes",
            "15",
            FAKE_DOCKER_RUNNING="1",
            FAKE_DOCKER_STOPPED=str(stopped),
        )
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            f"  {DRAINED[0]}\n  drain: done\n"
            + finished(
                "STOPPED: the monitor's container",
                f"next: step 6, start: {self.w} start",
            ),
        )
        self.assertEqual(
            self.stop_calls(),
            ["rk drain --timeout-minutes 15", "rk stop", "compose stop"],
        )
        self.assertTrue(stopped.exists())
        rc, out, err = self.rkmon(
            "stop", "--timeout-minutes", "5", FAKE_DOCKER_RUNNING="1"
        )
        self.assertEqual(
            (rc, out, err),
            (2, "", f"{self.w} stop: --timeout-minutes goes with --drain\n"),
        )
        self.log.unlink()
        rc, out, err = self.rkmon("stop", "--drain")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            finished(
                "NOTHING TO DO: the container is not running",
                f"next: step 6, start: {self.w} start",
            ),
        )
        self.assertEqual(self.stop_calls(), [])

    def test_stop_drain_stops_nothing_when_the_drain_does_not_end_well(self):
        """The drain that does not end well stops nothing: the reviews still in flight after --timeout-minutes,
        ERROR, exit 1, the drain lifted (rk drain's own lines said), and the command with twice as long, or later; rk
        drain refused while another drain is on, REFUSED, exit 4. rk stop that fails: its lines said, the drain lifted
        (rk undrain), the container not stopped, ERROR; docker compose stop that fails: ERROR, the monitor in the
        container stopped. Each ERROR and refusal names the command again, with --drain.
        """
        self.deploy()
        timeout = (
            "drain: nothing new starts; the work in flight finishes first (up to 0.0001 minutes): review #5",
            "DRAIN TIMEOUT after 0 minutes: review #5 still in flight; the drain is lifted",
        )
        held = "a drain is in force already (rk drain, since 13:58): it ends with that command, or rk start"
        w = self.w
        for case, rows, args, rc_out, calls in (
            (
                "timeout",
                [[r"rk drain --timeout-minutes 0\.0001$", "\n".join(timeout), 1]],
                ("--timeout-minutes", "0.0001"),
                (
                    1,
                    f"  {timeout[0]}\n  {timeout[1]}\n"
                    + finished(
                        "ERROR: the work in flight did not finish within 0.0001 minutes; the drain is lifted; "
                        "nothing changed",
                        f"next: {w} stop --drain --timeout-minutes 0.0002, or {w} stop --drain later",
                    ),
                ),
                ["rk drain --timeout-minutes 0.0001"],
            ),
            (
                "refused",
                [[r"rk drain --timeout-minutes 60$", f"REFUSED: {held}", 4]],
                (),
                (
                    4,
                    f"  REFUSED: {held}\n"
                    + finished(
                        f"REFUSED: {held}; nothing changed", f"next: {w} stop --drain"
                    ),
                ),
                ["rk drain --timeout-minutes 60"],
            ),
            (
                "rk stop",
                [
                    [
                        r" rk stop$",
                        "NOT STOPPED: the state lock is still in use after 120 s",
                        1,
                    ]
                ],
                (),
                (
                    1,
                    f"  {DRAINED[0]}\n  drain: done\n  NOT STOPPED: the state lock is still in use after 120 s\n"
                    + finished(
                        "ERROR: rk stop in the container exited 1, so the container was not stopped (the lines "
                        "above say why)",
                        f"next: {w} stop --drain",
                    ),
                ),
                ["rk drain --timeout-minutes 60", "rk stop", "rk undrain"],
            ),
            (
                "compose stop",
                [[r"^compose stop$", "", 1]],
                (),
                (
                    1,
                    f"  {DRAINED[0]}\n  drain: done\n"
                    + finished(
                        "ERROR: docker compose stop exited 1 (the lines above say why); the monitor in the container "
                        "is stopped, and the container runs on",
                        f"next: {w} stop --drain",
                    ),
                ),
                ["rk drain --timeout-minutes 60", "rk stop", "compose stop"],
            ),
        ):
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.drained(*rows)
                rc, out, err = self.rkmon(
                    "stop", "--drain", *args, FAKE_DOCKER_RUNNING="1"
                )
                self.assertEqual((rc, out), rc_out, err)
                self.assertEqual(self.stop_calls(), calls)

    def test_ctrl_c_during_stop_drain_undoes_what_it_started(self):
        """Ctrl-C during stop --drain, exit 130: during the drain, the drain lifted (rk undrain),
        nothing changed, and the command again. rk stop and docker compose stop each run to their end whatever Ctrl-C
        does (in a session of their own, the press said to wait), and the press is taken then: after rk stop, with the
        container up, rk start in it undoes the stop and the drain, nothing changed; after docker compose stop, with the
        container down, nothing is undone, and the state's next line says what to run; and when rk start fails, the
        monitor stays stopped, as that line says."""
        self.deploy()
        ends = self.base / "ended"
        stopped = self.base / "stopped"
        w = self.w
        waits = "Ctrl-C: the command above runs to its end first"
        undone = finished(
            "INTERRUPTED: nothing changed; the drain is lifted and new reviews start again",
            f"next: {w} stop --drain",
        )
        for case, rows, said, calls, waited in (
            (
                "the drain",
                [[r"rk drain --timeout-minutes 60$", DRAINED[0], "SIGINT"]],
                undone,
                ["rk drain --timeout-minutes 60", "rk undrain"],
                False,
            ),
            (
                "rk stop",
                [[r" rk stop$", "", "SIGINT", {}, {"ends": str(ends)}]],
                f"  {DRAINED[0]}\n  drain: done\n{undone}",
                ["rk drain --timeout-minutes 60", "rk stop", "rk start"],
                True,
            ),
            (
                "docker compose stop",
                [
                    [
                        r"^compose stop$",
                        "",
                        "SIGINT",
                        {},
                        {"makes": str(stopped), "ends": str(ends)},
                    ]
                ],
                f"  {DRAINED[0]}\n  drain: done\n"
                + finished(
                    "INTERRUPTED: the container stopped before Ctrl-C was taken, so it stays stopped",
                    f"next: step 6, start: {w} start",
                ),
                ["rk drain --timeout-minutes 60", "rk stop", "compose stop"],
                True,
            ),
            (
                "rk start fails",
                [
                    [r" rk stop$", "", "SIGINT", {}, {"ends": str(ends)}],
                    [r" rk start$", "ERROR: the state cannot be written", 1],
                    # the monitor in the running container stays stopped
                    [r"rk stopped$", "rk stop", 0, {}, {"after": r"(?m) rk stop$"}],
                ],
                f"  {DRAINED[0]}\n  drain: done\n"
                + finished(
                    "INTERRUPTED: rk start in the container exited 1 (ERROR: the state cannot be written), so the "
                    "monitor in it stays stopped",
                    f"next: step 6, start: {w} start",
                ),
                ["rk drain --timeout-minutes 60", "rk stop", "rk start"],
                True,
            ),
        ):
            with self.subTest(case=case):
                for path in (self.log, ends, stopped):
                    path.unlink(missing_ok=True)
                self.drained(*rows)
                rc, out, err = self.rkmon(
                    "stop",
                    "--drain",
                    FAKE_DOCKER_RUNNING="1",
                    FAKE_DOCKER_STOPPED=str(stopped),
                )
                self.assertEqual(rc, 130, out + err)
                self.assertTrue(out.endswith(said), out)
                self.assertEqual(self.stop_calls(), calls)
                self.assertEqual(ends.exists(), waited)
                self.assertEqual(waits in err, waited, err)

    def test_a_whole_run_takes_ctrl_c_once_it_has_ended(self):
        """run_whole: the command runs in a session of its own, so that the terminal's Ctrl-C does not reach it, and on
        to its end; a Ctrl-C meanwhile is said to wait, and is taken once the command has ended, as the handler before
        it takes it (KeyboardInterrupt, after ctrl_c_interrupts); a second one ends this command at once by the signal,
        the command going on to its end."""
        for presses in (1, 2):
            with self.subTest(presses=presses):
                ended = self.base / f"ended-{presses}"
                p = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        WHOLE_RUN,
                        self.w,
                        str(presses),
                        str(ended),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                waits = "Ctrl-C: the command above runs to its end first\n"
                if presses == 1:
                    self.assertEqual(
                        (p.returncode, p.stdout, p.stderr),
                        (0, "taken once it ended\n", waits),
                    )
                else:
                    self.assertEqual(
                        (p.returncode, p.stdout), (-signal.SIGINT, ""), p.stderr
                    )
                    self.assertEqual(p.stderr.count(waits), 1, p.stderr)
                    for _ in range(100):  # the command goes on to its end
                        if ended.exists():
                            break
                        time.sleep(0.1)
                    self.assertTrue(ended.exists())

    def test_the_doctor_ends_with_one_outcome_over_its_rows(self):
        """rkmon doctor prints the deployment's state as a row, here PASS (the monitor runs your confirmed settings),
        then rk doctor's report, then one outcome over the rows (deploy step 7): PASSED with their count, then the
        state's next line, step 8's while no tick has run since the container started; FAILED, exit 3 for a missing
        prerequisite, else 4, then the command again; ERROR for any other exit of rk doctor.
        Its result goes to this folder's records, but for an ERROR.
        """
        self.deploy()
        # the container started before this doctor, and no tick has run since
        self.ready += [
            [r"compose ps -q rkmon$", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
            [r"rk tick-due", "last tick never", 1],
        ]
        state = "  PASS  state: the monitor runs your confirmed settings\n\n"
        fix = checks.Command(("rk", "doctor", "--fresh"))
        passing = checks.render(
            [
                checks.Check("docker", "PASS", "29.0.0"),
                checks.Check("GitHub read login", "WARN", "expires in 5 days", fix),
            ]
        )
        failing = checks.render(
            [
                checks.Check("docker", "PASS", "29.0.0"),
                checks.Check("GitHub read login", "FAIL", "not set", fix),
            ]
        )
        failed = "FAILED: 1 of 3 checks failed; nothing changed (fix each FAIL above)"
        again = f"next: {self.w} doctor"
        one_failing = (3, ("GitHub read login",))
        for code, report, want_rc, end, kept in (
            (
                0,
                passing,
                0,
                finished(
                    "PASSED: 3 checks (1 warning)",
                    f"next: step 8, first tick: {self.w} first-tick",
                ),
                (3, ()),
            ),
            (4, failing, 4, finished(failed, again), one_failing),
            (3, failing, 3, finished(failed, again), one_failing),
            (
                7,
                "",
                1,
                finished(
                    "ERROR: rk doctor in the container exited 7 (the lines above say why)",
                    again,
                ),
                one_failing,  # the last whole report's
            ),
        ):
            with self.subTest(code=code):
                self.answer([["rk doctor", report, code]])
                rc, out, err = self.rkmon("doctor", FAKE_DOCKER_RUNNING="1")
                self.assertEqual(rc, want_rc, out + err)
                self.assertTrue(
                    out.endswith(
                        f"{state}{report}\n\n{end}" if report else f"{state}{end}"
                    ),
                    out,
                )
                doctor = records.read(self.pkg).doctor
                self.assertEqual((doctor.checks, doctor.failing), kept)
                self.assertIsNotNone(records.parsed(doctor.at))

    def test_the_doctor_fails_the_state_row_outside_the_confirmed_settings(self):
        """In any state but the monitor running your confirmed settings, the doctor's state row FAILs with the state's
        fix, or its next command, and the outcome is FAILED: exit 3 in an earlier step's state, else 4; the row
        counts in the result it keeps. Under --fails the report is the failing names, the state's first, and nothing
        is kept."""
        self.deploy()
        passing = checks.render([checks.Check("docker", "PASS", "29.0.0")])
        self.answer([["rk doctor", passing, 0]])
        rc, out, err = self.rkmon("doctor")  # the container is stopped: state 12
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            "  FAIL  state: the container is stopped\n"
            f"        fix: {self.w} start\n\n{passing}\n\n"
            + finished(
                "FAILED: 1 of 2 checks failed; nothing changed (fix each FAIL above)",
                f"next: {self.w} doctor",
            ),
            out,
        )
        doctor = records.read(self.pkg).doctor
        self.assertEqual((doctor.checks, doctor.failing), (2, ("state",)))
        # No image on this host (state 6), an earlier step's state: exit 3.
        self.answer([["rk doctor", passing, 0], [r"image inspect", "", 1]])
        rc, out, err = self.rkmon("doctor")
        self.assertEqual(rc, 3, out + err)
        self.assertIn(
            f"FAIL state: the image rocke-monitor:{deploy_folder.NAME} is not on this host fix: {self.w} image "
            "load IMAGE_FILE",
            one_line(out),
        )
        doctor = records.read(self.pkg).doctor
        self.assertEqual((doctor.checks, doctor.failing), (2, ("state",)))
        # --fails: the names alone; the result kept is the last whole report's.
        self.answer([["rk doctor", "session login", 4]])
        rc, out, err = self.rkmon("doctor", "--fails")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            out.endswith(
                "session login\n\n"
                + finished(
                    "FAILED: 2 checks failed; nothing changed",
                    f"next: {self.w} doctor --fails",
                )
            ),
            out,
        )
        self.assertEqual(records.read(self.pkg).doctor, doctor)

    def test_status_puts_the_state_first(self):
        """rkmon status: a line that names this folder's deployment, its container and the features its
        confirmed settings turn on; the state, with its fix; the next line rkmon next ends with; then, while the
        container runs, rk status from it. A hand edit of docker/.env shows as state 9 (changed outside your file),
        in rkmon next too."""
        self.deploy()
        head = (
            f"deployment {deploy_folder.NAME} (this package folder) · {{}} · review automation prototype · merge "
            "queue prototype · the admin dashboard"
        )
        rc, out, err = self.rkmon("status")
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(
            out,
            finished(head.format("stopped"))
            + f"state: the container is stopped\nnext: step 6, start: {self.w} start\n",
        )
        self.assertFalse([c for c in self.calls() if "rk status" in c[1]])
        self.answer(
            [
                [r"compose ps -q rkmon$", "0123456789ab", 0],
                [
                    r"inspect --format \{\{\.State\.StartedAt\}\}",
                    "2026-11-15T09:00:00Z",
                    0,
                ],
                [r"rk status$", "monitor: running", 0],
            ]
        )
        rc, out, err = self.rkmon("status", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(
            out,
            finished(head.format("running since 2026-11-15 09:00 UTC"))
            + "state: the monitor runs your confirmed settings\n"
            f"next: step 7, doctor: {self.w} doctor\n\nmonitor: running\n",
        )
        # A hand edit of docker/.env: state 9, whose next line puts back what your confirmed settings make.
        env = self.pkg / "docker" / ".env"
        env.write_text(
            env.read_text(encoding="utf-8") + "MY_NOTE=5\n", encoding="utf-8"
        )
        edited = (
            f"changed outside your file: docker/.env differs from what {self.w} last wrote there",
            f"next: {self.w} apply (--dry-run shows what it puts back)",
        )
        rc, out, err = self.rkmon("status")
        self.assertEqual(rc, 0, out + err)
        # one line: the program's path, a temporary folder's, moves the line breaks
        self.assertTrue(one_line(out).endswith(f"state: {edited[0]} {edited[1]}"), out)
        rc, out, err = self.rkmon("next")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(one_line(out).endswith(f"note: {edited[0]} {edited[1]}"), out)

    def test_status_exits_0_whatever_docker_or_rk_say(self):
        """rkmon status exits 0: when docker does not answer, the state is what the records decide alone, and
        rk is not asked; when rk status fails, the monitor's status is unknown, and why; when the running monitor does
        not say which settings it runs, the state is unknown: rk's refusal follows, its fix with it, or what rk said.
        """
        daemon = "Cannot connect to the Docker daemon. Is it running?"
        self.answer([[r"^ps --all --filter", daemon, 1]])
        rc, out, err = self.rkmon("status")
        self.assertEqual(rc, 0, out + err)
        # with no settings confirmed, no container has this folder's name: docker is not asked
        self.assertTrue(
            out.startswith(
                "deployment: none confirmed yet (this package folder) · no container\n"
                "state: no deploy settings are confirmed in this package folder\n"
            ),
            out,
        )
        self.assertEqual(self.calls(), [])
        self.deploy()
        self.answer([[r"^ps --all --filter", daemon, 1]])
        rc, out, err = self.rkmon("status")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                f"deployment {deploy_folder.NAME} (this package folder) · container unknown · "
            ),
            out,
        )
        self.assertIn(
            f"\nstate: unknown (docker does not answer)\nnote: docker does not answer ({daemon})\n",
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk status" in c[1]])
        self.answer([[r"rk status --queue$", "", 5]])
        rc, out, err = self.rkmon("status", "--queue", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                "the monitor's status: unknown (rk status exited 5: the lines above say why)\n"
            ),
            out,
        )
        self.assertIn(
            (
                self.docker_dir,
                f"compose exec -T --user rkmon -e RKMON_WRAPPER={self.w} rkmon rk status --queue",
            ),
            self.calls(),
        )
        refusal = "REFUSED: the applied record cannot be read\nfix: rk applied --load"
        running = f"deployment {deploy_folder.NAME} (this package folder) · running\n"
        for said, code, why in (
            (refusal, 4, f")\n{refusal}"),
            ("Traceback\nno record", 1, "; rk applied --json exited 1: no record)"),
        ):
            with self.subTest(code=code):
                self.answer(
                    [
                        [r"rk applied --json", said, code],
                        [r"rk status$", "monitor: running", 0],
                    ]
                )
                rc, out, err = self.rkmon("status", FAKE_DOCKER_RUNNING="1")
                self.assertEqual(rc, 0, out + err)
                self.assertEqual(
                    out,
                    f"{running}state: unknown (the running monitor did not say which settings it runs{why}\n"
                    "\nmonitor: running\n",
                )

    def test_remove_before_the_first_start(self):
        """Before the first start there is no container, and there may be volumes: rkmon login made those that
        keep the logins, or a start cut short made others too. rkmon remove has nothing to do then, and says that the
        data stays; with --delete-data, asked first (with no terminal, only with --yes), it removes each volume of
        this deployment's by its name, found never by a pattern: those docker compose made for its compose project, by
        their label, and those its compose file names that are here, made otherwise. Then the state's next line,
        docker asked again: with the data gone, the logins. When docker cannot remove them, ERROR says the data stays,
        with the fix. While another deployment uses each of them, or with no volume either, it has nothing to do.
        """
        self.deploy(made=False)
        w, name = self.w, deploy_folder.NAME
        secrets = [f"{name}_rkmon-secrets", f"{name}_rkmon-secrets-write"]
        stays = f"the data stays in this deployment's volumes ({w} remove --delete-data removes them)"
        data = f"{name}'s volumes: its state, its clone, its secrets and its logins, for good"
        none_here = [*deploy_folder.volumes(), *deploy_folder.secrets_volumes(False)]
        # (made by docker compose, made otherwise, removed): a start cut short made the settings volume, and docker
        # compose one from an earlier compose file; the code volume was made otherwise; and the stand-in docker's
        # listing by name gives a volume whose name holds the work volume's, as a name filter that is not exact would
        cases = {
            "logins set, never started": (secrets, [], secrets),
            "a start cut short": (
                [*secrets, f"{name}_rkmon-settings", f"{name}_old"],
                [f"{name}_rkmon-code", f"{name}_rkmon-work-copy"],
                [
                    *secrets,
                    f"{name}_rkmon-settings",
                    f"{name}_old",
                    f"{name}_rkmon-code",
                ],
            ),
        }
        for case, (labeled, unlabeled, removed) in cases.items():
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.env.pop("FAKE_DOCKER_DOWN", None)
                here = deploy_folder.volumes(labeled, unlabeled)
                self.answer([*here, *deploy_folder.secrets_volumes(True)])
                rc, out, err = self.rkmon("remove")
                self.assertEqual(
                    (rc, out, err),
                    (
                        0,
                        finished(
                            f"NOTHING TO DO: there is no container; {stays}",
                            f"next: step 6, start: {w} start",
                        ),
                        "",
                    ),
                )
                rc, out, err = self.rkmon("remove", "--delete-data")
                self.assertEqual((rc, out), (2, ""))
                self.assertIn(
                    f"{w} remove --delete-data: no terminal to ask on: {w} remove --delete-data --yes removes {data}",
                    one_line(err),
                )
                self.assertEqual(self.acted(), [])
                self.after_removal(*none_here)
                rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
                self.assertEqual(
                    (rc, out),
                    (
                        0,
                        finished(
                            "REMOVED: this deployment's volumes",
                            f"next: step 5, login: {w} login",
                        ),
                    ),
                    err,
                )
                rm = f"volume rm {' '.join(sorted(removed))}"
                self.assertEqual(
                    err.splitlines(), ["docker compose down", f"docker {rm}"]
                )
                self.assertEqual(
                    self.acted(),
                    [(self.docker_dir, "compose down"), (self.docker_dir, rm)],
                )
        # docker cannot remove a volume: ERROR, which says the data stays, with the fix, then the command again
        self.log.unlink(missing_ok=True)
        self.env.pop("FAKE_DOCKER_DOWN", None)
        self.answer(
            [
                [r"^volume rm ", "", 1],
                *deploy_folder.volumes(secrets),
                *deploy_folder.secrets_volumes(True),
            ]
        )
        typed = ("remove", "--delete-data", "--yes")
        rc, out, err = self.rkmon(*typed)
        self.assertEqual(
            (rc, one_line(out)),
            (
                1,
                "ERROR: docker volume rm exited 1: the data stays in the volumes it could not remove (the lines "
                "above say which, and why) fix: once what the lines above say is fixed, "
                f"{w} {' '.join(typed)} removes them next: {w} {' '.join(typed)}",
            ),
            err,
        )
        # every volume another deployment uses: each stays, and there is nothing to do
        self.log.unlink(missing_ok=True)
        users = [
            [
                rf"^ps --all --filter volume={re.escape(v)} ",
                json.dumps(
                    {
                        "Names": "other-rkmon-1",
                        "Labels": "com.docker.compose.project=other,"
                        "com.docker.compose.project.working_dir=/srv/other/docker",
                    }
                ),
                0,
            ]
            for v in secrets
        ]
        listed = "".join(
            json.dumps({"Name": v, "Labels": f"com.docker.compose.project={name}"})
            + "\n"
            for v in secrets
        )
        self.answer(
            [
                [
                    r"^volume ls (--filter name=\S+ )+--format \{\{json \.\}\}$",
                    listed,
                    0,
                ],
                *users,
                *deploy_folder.volumes(secrets),
                *deploy_folder.secrets_volumes(True),
            ]
        )
        rc, out, err = self.rkmon(*typed)
        self.assertEqual(
            (rc, out, err),
            (
                0,
                finished(
                    f"NOTHING TO DO: there is no container; {secrets[0]} and {secrets[1]} stay, as another "
                    "deployment on this host uses them",
                    f"next: step 6, start: {w} start",
                ),
                "",
            ),
        )
        self.assertEqual(self.acted(), [])
        # neither a container nor a volume: nothing to do, and nothing asked
        self.log.unlink(missing_ok=True)
        self.answer(none_here)
        for args in (("remove",), ("remove", "--delete-data")):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args)
                self.assertEqual(
                    (rc, out, err),
                    (
                        0,
                        finished(
                            "NOTHING TO DO: this deployment has no container and no volume on this host",
                            f"next: step 5, login: {w} login",
                        ),
                        "",
                    ),
                )
        self.assertEqual(self.acted(), [])

    def test_remove_the_container_and_its_data(self):
        """rkmon remove on a deployment that has started, its container stopped (state 12): REMOVED, the
        container (docker compose down), and its data stays; with --delete-data, its volumes too, asked first: a no is
        CANCELLED (exit 1), and with no terminal and no --yes, a usage error that names the command with --yes;
        nothing removed either way. Then the state's next line, docker asked again once the container is gone: the
        logins are read in their volumes, as before a first start, and with the data gone, they are the next step.
        ERROR when docker cannot remove them, or its answer about the volumes cannot be read, then the command again.
        """
        self.deploy()
        w, name = self.w, deploy_folder.NAME
        here = [
            *deploy_folder.volumes(deploy_folder.VOLUMES),
            *deploy_folder.secrets_volumes(True),
        ]
        data = f"the container and {name}'s volumes: its state, its clone, its secrets and its logins, for good"
        self.answer(here)
        rc, out, err = self.rkmon("remove", "--delete-data")
        self.assertEqual((rc, out), (2, ""))
        self.assertIn(
            f"{w} remove --delete-data: no terminal to ask on: {w} remove --delete-data --yes removes {data}",
            one_line(err),
        )
        rc, text = sandbox.on_a_terminal(
            [w, "remove", "--delete-data"], self.cwd, self.env, typed=b"n\n"
        )
        self.assertEqual(rc, 1, text)
        self.assertIn(f"Remove {data}? [y/N] ", text)
        self.assertIn(
            finished(
                "CANCELLED: nothing removed", f"next: step 6, start: {w} start"
            ).replace("\n", "\r\n"),
            text,
        )
        self.assertEqual(self.acted(), [])
        # the container alone: its volumes hold the logins, so the start is next
        self.after_removal(*here)
        rc, out, err = self.rkmon("remove")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    f"REMOVED: the container; the data stays in this deployment's volumes ({w} remove "
                    "--delete-data removes them)",
                    f"next: step 6, start: {w} start",
                ),
            ),
            err,
        )
        self.assertEqual(err.splitlines(), ["docker compose down"])
        self.assertEqual(self.acted(), [(self.docker_dir, "compose down")])
        # and its data
        self.log.unlink()
        self.answer(here)
        self.after_removal(
            *deploy_folder.volumes(), *deploy_folder.secrets_volumes(False)
        )
        rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    "REMOVED: the container and its volumes",
                    f"next: step 5, login: {w} login",
                ),
            ),
            err,
        )
        rm = f"volume rm {' '.join(sorted(deploy_folder.VOLUMES))}"
        self.assertEqual(
            self.acted(), [(self.docker_dir, "compose down"), (self.docker_dir, rm)]
        )
        # docker cannot remove them, or its answer about the volumes cannot be read: what ran, then ERROR, which says
        # what is gone and what stays, then the command again
        self.env.pop("FAKE_DOCKER_DOWN", None)
        down = (self.docker_dir, "compose down")
        for rows, env, typed, said, ran in (
            (
                [],
                {"FAKE_DOCKER_EXIT": "1"},
                ("remove",),
                "docker compose down exited 1 (the lines above say why, and what it removed); the data stays in "
                "this deployment's volumes",
                [down],
            ),
            (
                [[r"^volume rm ", "", 1]],
                {},
                ("remove", "--delete-data", "--yes"),
                "docker volume rm exited 1: the container is removed already, and the data stays in the volumes it "
                "could not remove (the lines above say which, and why) fix: once what the lines above say is "
                f"fixed, {w} remove --delete-data --yes removes them",
                [down, (self.docker_dir, rm)],
            ),
            (
                [[r"^volume ls --filter label=", "", 1]],
                {},
                ("remove", "--delete-data", "--yes"),
                "docker's answer about this deployment's volumes cannot be read; nothing removed",
                [],
            ),
        ):
            with self.subTest(said=said):
                self.log.unlink(missing_ok=True)
                self.answer([*rows, *here])
                rc, out, err = self.rkmon(*typed, **env)
                self.assertEqual(
                    (rc, one_line(out)),
                    (1, f"ERROR: {said} next: {w} {' '.join(typed)}"),
                    err,
                )
                self.assertEqual(self.acted(), ran)

    def test_remove_delete_data_forgets_the_first_tick(self):
        """rkmon remove --delete-data: once it has removed the volumes, this folder's records forget the
        first tick (records.clear_first_tick), so the next start is a first start, and the next line names its deploy
        step again. A plain remove keeps it, as the data stays, so the line names no step; so do --dry-run, and a
        removal docker cannot finish."""
        self.deploy()
        w = self.w
        here = [
            *deploy_folder.volumes(deploy_folder.VOLUMES),
            *deploy_folder.secrets_volumes(True),
        ]
        first = records.FirstTick("2026-11-15T09:05:00Z")
        records.first_tick(self.pkg, first)
        self.answer(here)
        self.after_removal(*here)
        rc, out, err = self.rkmon("remove")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    f"REMOVED: the container; the data stays in this deployment's volumes ({w} remove "
                    "--delete-data removes them)",
                    f"next: {w} start",
                ),
            ),
            err,
        )
        self.assertEqual(records.read(self.pkg).first_tick, first)
        self.env.pop("FAKE_DOCKER_DOWN", None)
        for typed, rows, code in (
            (("--dry-run", "remove", "--delete-data"), [], 0),
            (("remove", "--delete-data", "--yes"), [[r"^volume rm ", "", 1]], 1),
        ):
            with self.subTest(typed=typed):
                self.answer([*rows, *here])
                rc, out, err = self.rkmon(*typed)
                self.assertEqual(rc, code, out + err)
                self.assertEqual(records.read(self.pkg).first_tick, first)
        self.answer(here)
        self.after_removal(
            *deploy_folder.volumes(), *deploy_folder.secrets_volumes(False)
        )
        rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    "REMOVED: the container and its volumes",
                    f"next: step 5, login: {w} login",
                ),
            ),
            err,
        )
        self.assertIsNone(records.read(self.pkg).first_tick)

    def test_remove_keeps_a_volume_another_deployment_uses(self):
        """rkmon remove --delete-data keeps each volume another deployment on this host uses, as check 15 reads
        it at the start (identity.volume_owner): one whose compose project label names another project, or that a
        container of another compose project mounts. It removes the others, and its outcome names the volume kept and
        why, under --dry-run too. docker lists each volume by its exact name: its name filter is a pattern that would
        match a part of another volume's name.
        """
        self.deploy()
        w, name = self.w, deploy_folder.NAME
        work, secrets = f"{name}_rkmon-work", f"{name}_rkmon-secrets"
        here = [
            *deploy_folder.volumes(deploy_folder.VOLUMES),
            *deploy_folder.secrets_volumes(True),
        ]
        other = {
            "Names": "other-rkmon-1",
            "Labels": "com.docker.compose.project=other,"
            "com.docker.compose.project.working_dir=/srv/other/docker",
        }

        def whose(made_by: dict, used=()) -> list:
            """docker's answers about whose each volume is: the compose project its label names, MADE_BY's or this
            deployment's; the volumes USED, each mounted by another deployment's container.
            """
            listed = "".join(
                json.dumps(
                    {
                        "Name": v,
                        "Labels": f"com.docker.compose.project={made_by.get(v, name)}",
                    }
                )
                + "\n"
                for v in deploy_folder.VOLUMES
            )
            return [
                [
                    r"^volume ls (--filter name=\S+ )+--format \{\{json \.\}\}$",
                    listed,
                    0,
                ],
                *(
                    [
                        rf"^ps --all --filter volume={re.escape(v)} ",
                        json.dumps(other),
                        0,
                    ]
                    for v in used
                ),
            ]

        for case, made_by, used, kept in (
            (
                "made by another deployment",
                {work: "other"},
                (),
                f"{work} stays, as another deployment on this host uses it",
            ),
            (
                "and one mounted by another deployment's container",
                {work: "other"},
                (secrets,),
                f"{secrets} and {work} stay, as another deployment on this host uses them",
            ),
        ):
            with self.subTest(case=case):
                self.log.unlink(missing_ok=True)
                self.answer([*whose(made_by, used), *here])
                rm = f"volume rm {' '.join(sorted(set(deploy_folder.VOLUMES) - {work, *used}))}"
                rc, out, err = self.rkmon("--dry-run", "remove", "--delete-data")
                self.assertEqual(
                    (rc, one_line(out)),
                    (
                        0,
                        f"docker compose down docker {rm} PASSED: the container and its other volumes can be "
                        f"removed; {kept} next: {w} remove --delete-data",
                    ),
                    err,
                )
                self.assertEqual(self.acted(), [])
                self.after_removal(
                    *deploy_folder.volumes(), *deploy_folder.secrets_volumes(False)
                )
                rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(
                    one_line(out).startswith(
                        f"REMOVED: the container and its other volumes; {kept} next: "
                    ),
                    out,
                )
                self.assertEqual(
                    self.acted(),
                    [(self.docker_dir, "compose down"), (self.docker_dir, rm)],
                )
                # each volume listed by its exact name
                filters = [
                    v
                    for _, args in self.calls()
                    if args.startswith("volume ls --filter name=")
                    for v in re.findall(r"--filter name=(\S+)", args)
                ]
                self.assertTrue(filters)
                self.assertEqual(
                    [v for v in filters if not re.fullmatch(r"\^[\w-]+\$", v)], []
                )
                self.env.pop("FAKE_DOCKER_DOWN", None)

    def test_remove_reads_the_compose_files_with_docker_env_s_variables(self):
        """rkmon remove --delete-data on a deployment whose container serves the team dashboard: the compose
        override publishes its port on the address docker/.env gives (host_ip: "${RKMON_TEAM_HOST}"), and docker
        compose cannot read the compose files with no variable put in. With docker/.env there, they are read with its
        variables put in, so the volumes the files declare, made otherwise than by docker compose, are found by their
        names, and removed."""
        served = {"enabled": True, "mode": "live", "delivery": "serve"}
        served["serve"] = {"address": "127.0.0.1", "port": sandbox.free_port()}
        served["public_url"] = "https://dashboards.example.com/my-team/"
        self.deploy(features={"dashboards": {"team": served}})
        w = self.w
        override = self.pkg / "docker" / "compose.override.yaml"
        self.assertIn(
            'host_ip: "${RKMON_TEAM_HOST}"', override.read_text(encoding="utf-8")
        )
        # docker compose's refusal of the override's address while no variable is put in
        unread = [
            r"^compose config --no-interpolate ",
            "services.rkmon.ports.[]: invalid ip address: ${RKMON_TEAM_HOST}",
            1,
        ]
        self.answer(
            [
                unread,
                *deploy_folder.volumes(unlabeled=deploy_folder.VOLUMES),
                *deploy_folder.secrets_volumes(True),
            ]
        )
        self.after_removal(
            *deploy_folder.volumes(), *deploy_folder.secrets_volumes(False)
        )
        rc, out, err = self.rkmon("remove", "--delete-data", "--yes")
        self.assertEqual(
            (rc, out),
            (
                0,
                finished(
                    "REMOVED: the container and its volumes",
                    f"next: step 5, login: {w} login",
                ),
            ),
            err,
        )
        rm = f"volume rm {' '.join(sorted(deploy_folder.VOLUMES))}"
        self.assertEqual(
            self.acted(), [(self.docker_dir, "compose down"), (self.docker_dir, rm)]
        )

    def test_usage_errors_run_nothing(self):
        """A usage error is the command table parser's (rkmon_lib/commands.py), or the command's own: exit 2, one
        line that names the command, and nothing run."""
        listed = f"{self.w} help lists the commands"
        for args, words in (
            (  # setup went, and the deploy path replaces it
                ("setup", "--settings", "missing.yaml"),
                f'{self.w}: unknown command "setup": set up a deployment with your deploy settings file',
            ),
            (
                ("build",),
                f"{self.w} build: the following arguments are required: --settings",
            ),
            (
                ("build", "--settings", "my-settings.yaml", "extra"),
                f"{self.w} build: unrecognized arguments: extra",
            ),
            (  # docker is given only the flags build lists
                ("build", "--settings", "my-settings.yaml", "--", "--label", "x=y"),
                f"{self.w} build: unrecognized arguments: ",
            ),
            (
                ("build", "--settings", "my-settings.yaml", "--print"),
                f'{self.w} build: unknown option "--print": {self.w} build --settings FILE --dry-run '
                "replaces it",
            ),
            (
                ("image", "save"),
                f"{self.w} image save: the following arguments are required: IMAGE_FILE",
            ),
            (
                ("image", "nothing"),
                f'{self.w} image: unknown command "image nothing": {listed}',
            ),
            (
                ("image", "push"),
                f"{self.w} image push: the following arguments are required: --settings",
            ),
            (("rebuild",), f'{self.w}: unknown command "rebuild": {listed}'),
            (
                ("login", "github"),
                f"{self.w} login: "
                + wrapping.joined(
                    [e.name for e in sr.entries({}) if e.shape == sr.LOGIN], "or"
                ),
            ),  # the registry's logins, in its order: github_read, github_write, atlassian or claude
            (
                ("login", "gh_read_token"),
                f"{self.w} login: gh_read_token is the older name of the login github_read: {self.w} login "
                "github_read sets it",
            ),  # an older name: the registry's, with its entry's name now
            (
                ("login", "claude_env"),
                f"{self.w} login: claude_env is the older name of the login claude: {self.w} login claude sets it",
            ),  # the older secret that kept the Claude session's key
            (
                ("login", "claude", "--with-token"),
                f"{self.w} login: --with-token goes with {wrapping.joined(sr.GH_SIDES, 'or')}; claude takes its "
                "key typed, or on stdin, with no flag",
            ),
            (
                ("login", "--with-token"),
                f"{self.w} login: --with-token goes with a login's NAME",
            ),
            (
                ("login", "github_write", "--move"),
                f"{self.w} login: unrecognized arguments: --move",
            ),  # the start's root phase moves an earlier deployment's write login by itself
            (("deploy",), f'{self.w}: unknown command "deploy": {listed}'),
            # a command another replaced: no alias, and the error names the one that replaces it
            (
                ("down", "--volumes", "--yes"),
                f'{self.w}: unknown command "down": {self.w} remove [--delete-data] replaces it',
            ),
            (
                ("help", "preflight"),
                f'{self.w} help: unknown command "preflight": {self.w} check --settings FILE replaces it\n',
            ),
            (
                ("help", "nothing"),
                f'{self.w} help: no command or topic "nothing": {self.w} help lists them\n',
            ),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args)
                self.assertEqual((rc, out), (2, ""))
                self.assertIn(words, err)
        self.assertEqual(self.calls(), [])
        for args in ((), ("help",), ("--help",)):
            rc, out, _err = self.rkmon(*args)
            self.assertEqual(rc, 0)
            self.assertIn("rkmon start", out)
            # as push takes it (the program's path may wrap the line)
            self.assertIn("rkmon image push --settings FILE", one_line(out))
            self.assertNotIn("push [--settings", out)

    def test_each_command_of_the_table_has_its_handler(self):
        """Each row of the command table runs the wrapper's cmd_ method of its words (`image load`: cmd_image_load),
        and each cmd_ method is a row's."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        handlers = {n for n in dir(mod.Wrapper) if n.startswith("cmd_")}
        rows = {
            "cmd_" + r.name.replace(" ", "_").replace("-", "_") for r in commands.TABLE
        }
        self.assertEqual(handlers, rows)

    def test_a_compose_question_docker_does_not_answer_is_no_container(self):
        """made() reads a container from docker's answer alone: a compose error, such as a variable docker/.env
        lacks, comes back with the answer's text and says nothing of a container."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        error = "required variable RKMON_CPUS is missing a value: see ./rkmon next\n"
        for answer, made in (
            ((0, "0123456789ab\n"), True),
            ((0, ""), False),
            ((15, error), False),
        ):
            with self.subTest(answer=answer):
                w.ask_docker = lambda *args, timeout=60, a=answer: a
                self.assertEqual(w.made(), made)

    def test_compose_env_takes_only_what_docker_env_lacks_from_the_confirmed_settings(
        self,
    ):
        """Once docker/.env this package's deliver wrote is there (dotenv_delivered), compose_env takes from the
        confirmed settings (confirmed_plan) each variable the file lacks that every deploy settings make, a later
        package's (RKMON_CPUS here), and no other: those the file has stay its own, and the team dashboard's two, there
        only while it is served, stay out, so that a change that waits for apply (the dashboard served) never reaches
        what runs. With none lacking, it reads no confirmed plan."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.pkg = str(HERE.parent)
        names = list(w.lib("deploy_plan").ENV_KEYS)
        team = ("RKMON_TEAM_HOST", "RKMON_TEAM_PORT")
        on_file = {k: "on-file" for k in names if k not in team}
        plans = []
        w.confirmed = lambda: mock.Mock(container_name="rkmon-test")
        w.dotenv_delivered = lambda: True
        w.dotenv = lambda: dict(on_file)
        w.confirmed_plan = lambda: plans.append(1) or mock.Mock(
            variables={k: "confirmed" for k in names}
        )

        def set_here() -> dict:
            return {k: os.environ[k] for k in names if k in os.environ}

        with mock.patch.dict(os.environ, {}):
            for k in names:
                os.environ.pop(k, None)
            w.compose_env()
            self.assertEqual(
                (set_here(), plans), ({"COMPOSE_PROJECT_NAME": "rkmon-test"}, [])
            )
            del on_file["RKMON_CPUS"]
            w.compose_env()
            self.assertEqual(
                set_here(),
                {"COMPOSE_PROJECT_NAME": "rkmon-test", "RKMON_CPUS": "confirmed"},
            )

    def test_a_question_in_a_one_off_container_leaves_the_volumes_as_they_are(self):
        """A question in the container (quiet_inside; as the writer, writer_quiet) runs in the running container, else
        in a one-off whose entry point it tells that its command only reads (only_reads: the layout's
        LEAVE_VOLUMES_VARIABLE), so that its root phase walks no file of the volumes and writes no ids record
        (docker/entrypoint.sh); the running container's exec has no root phase, and no such variable.
        """
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.pkg, w.image_code = str(HERE.parent), False
        leave = f"{container_layout.LEAVE_VOLUMES_VARIABLE}=1"
        for running in (False, True):
            asked = []
            w.running = lambda running=running: running
            w.ask_docker = lambda *args, timeout=60: asked.append(args) or (0, "")
            with self.subTest(running=running):
                w.quiet_inside("rk", "stopped")
                w.writer_quiet("rk", "writer-secrets", "--json")
                self.assertEqual(
                    [a[:2] for a in asked],
                    [("compose", "exec" if running else "run")] * 2,
                )
                for args in asked:
                    self.assertEqual(leave in args, not running, args)
                    if not running:
                        self.assertEqual(args[args.index(leave) - 1], "-e", args)

    def test_a_refusal_in_a_delivery_shows_itself_above_its_outcome(self):
        """A delivery or a put back that a private file's refusal stops (private_files.Refused: deliver's own, or
        carried by deliver.Failed's `above` once its run has ended) shows the refusal as it shows itself, its rows then
        where the rules are, above the outcome line, whose words say that the lines above say why (said_above): here
        the first start's delivery; any other failure, its own words alone."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.pkg, w.dry = str(HERE.parent), 0
        w.keep_previous = lambda plan: False
        w.again_line = lambda: "next: rkmon start"
        dl, pf = w.lib("deliver"), w.lib("private_files")
        row = checks.Check(
            "private file",
            checks.Status.FAIL,
            "docker/.env is not a regular file",
            checks.Command(("ls", "-la", "docker/.env")),
        )
        refusal = pf.Refused([row])
        changed = "the settings volume has the new layer; docker/.env and the override are as they were"
        why = "docker/.env could not be written (the lines above say why)"
        failed = dl.Failed(0, changed, why, above=str(refusal))

        def fails(*args, **kwargs):
            raise failed

        out = io.StringIO()
        with mock.patch.object(dl, "deliver", fails), contextlib.redirect_stdout(out):
            self.assertEqual(w.first_delivery(None), 1)
        self.assertTrue(
            out.getvalue().endswith(
                f"{refusal}\n"
                + finished(
                    f"ERROR: {why}: {changed}; nothing started", "next: rkmon start"
                )
            ),
            out.getvalue(),
        )
        for exc, said, words in (
            (refusal, f"{refusal}\n", "the lines above say why"),
            (
                OSError(28, "No space left on device"),
                "",
                "[Errno 28] No space left on device",
            ),
        ):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(w.said_above(exc), words)
            self.assertEqual(out.getvalue(), said)

    def test_a_command_its_handler_leaves_on_an_exception_ends_with_its_outcome(self):
        """A command whose handler ends on an exception ends with its outcome line, then this command again (main,
        ended_on): a private file that breaks a rule shows itself, its rows then where the rules are, then REFUSED,
        nothing written there, exit 4; an error the command does not handle, its traceback on stderr, then ERROR with
        what it is, exit 1."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = mod.Wrapper(self.w)
        w.hold_lock = lambda ns, held: None
        row = checks.Check(
            "private file",
            checks.Status.FAIL,
            "my-settings.yaml can be read by others",
            checks.Command(("chmod", "600", "my-settings.yaml")),
        )
        refusal = w.lib("private_files").Refused([row])
        program = w.program
        for exc, code, above, outcome, traced in (
            (
                refusal,
                4,
                f"{refusal}\n",
                f"REFUSED: a private file {program} status reads or writes breaks a rule; nothing is written there "
                "(fix each FAIL above)",
                False,
            ),
            (
                KeyError("x"),
                1,
                "",
                f"ERROR: {program} status stopped on an error it does not handle (KeyError: 'x'); the traceback "
                "above says where",
                True,
            ),
        ):

            def fails(ns, exc=exc):
                raise exc

            w.handler = lambda row, fails=fails: fails
            out, errs = io.StringIO(), io.StringIO()
            with mock.patch.object(os, "chdir"), mock.patch.dict(os.environ):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errs):
                    self.assertEqual(w.main(["status"]), code)
            self.assertEqual(
                out.getvalue(),
                above + finished(outcome, f"next: {program} status"),
            )
            if traced:
                self.assertTrue(
                    errs.getvalue().startswith("Traceback (most recent call last):\n"),
                    errs.getvalue(),
                )
                self.assertTrue(errs.getvalue().endswith("KeyError: 'x'\n"))
            else:
                self.assertEqual(errs.getvalue(), "")

    def test_a_settings_file_that_cannot_be_read_gives_its_fix_a_line_of_its_own(self):
        """image pull and image check refuse while the recorded settings file cannot be read (settings_cfg,
        refuse_unread), each fix on a line of its own, never inside the sentence: the one row's found text is the
        refusal, its fix in the place of the state's; several rows show first, each with its fix, then the refusal and
        the state's next line."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.pkg = str(HERE.parent)
        refused = []
        w.refuse = (
            lambda need="", fix=None, facts=None: refused.append((need, fix)) or 4
        )
        w.state_next_line = lambda: "next: rkmon image check"
        file = self.cwd / "my-settings.yaml"
        file.write_bytes(b"kind: deploy_settings\nschema_version: 2\nname: \xff\n")
        cfg, problem = w.settings_cfg(str(file))
        self.assertIsNone(cfg)
        (row,) = w.problems
        self.assertEqual(problem, row.found)
        self.assertNotIn(checks.FIX_LABEL, problem)
        self.assertEqual(w.refuse_unread(problem), 4)
        self.assertEqual(refused, [(row.found, row.fix)])
        file.write_text(
            'kind: deploy_settings\nschema_version: 2\na: "\\ud800"\nb: "\\udc00"\n',
            encoding="utf-8",
        )
        cfg, problem = w.settings_cfg(str(file))
        self.assertEqual((cfg, len(w.problems)), (None, 2))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(w.refuse_unread(problem), 4)
        self.assertEqual(len(refused), 1)
        self.assertEqual(
            out.getvalue(),
            checks.render(w.problems, "problems")
            + "\n"
            + finished(
                f"REFUSED: {settings_file.shown(str(file))} cannot be read; nothing changed (fix each FAIL above)",
                "next: rkmon image check",
            ),
        )

    def test_a_docker_question_with_no_answer_in_time_says_so(self):
        """ask_docker's answer when docker gives none in time: confirm.TIMED_OUT and the wait, never the command
        (check 16's SKIP shows it); docker that cannot start: 127 and why."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.engine, w.pkg = "docker", str(HERE.parent)

        def late(cmd, timeout, **_kw):
            raise subprocess.TimeoutExpired(cmd, timeout)

        def missing(cmd, **_kw):
            raise FileNotFoundError(2, "No such file or directory", cmd[0])

        for run, answer in (
            (late, (124, "timed out after 120 seconds")),
            (missing, (127, "[Errno 2] No such file or directory: 'docker'")),
        ):
            with self.subTest(run=run.__name__), mock.patch.object(
                mod.subprocess, "run", run
            ):
                self.assertEqual(
                    w.ask_docker("run", "--rm", "ref", timeout=120), answer
                )
        self.assertEqual(confirm.TIMED_OUT, 124)

    def test_one_listing_of_a_compose_project_s_volumes(self):
        """The deploy path's check (facts) and check 16's (own_secrets_volume) read a compose project's volumes from
        one listing (project_volumes): docker's volumes filtered on its compose project label, by name; None when
        docker gives no answer, which each of them reads as no volume."""
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", self.w)
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = object.__new__(mod.Wrapper)
        w.engine, w.pkg, w.here = "docker", str(HERE.parent), str(self.base)
        w.owner_of = lambda project: (identity.NONE, "", None)
        w.whose = lambda project: identity.NONE
        w.recorded = lambda: ""
        listing = (
            "volume",
            "ls",
            "--filter",
            "label=com.docker.compose.project=rkmon-test",
            "--format",
            "{{.Name}}",
        )
        secrets = f"rkmon-test_{container_layout.SECRETS_VOLUME}"
        for listed, volumes in (
            (
                (0, f"rkmon-test_rkmon-work\n{secrets}\n"),
                ["rkmon-test_rkmon-work", secrets],
            ),
            ((0, ""), []),
            ((1, "Cannot connect to the Docker daemon"), None),
        ):
            asked = []

            def docker(*args, timeout=60, variables=None, listed=listed):
                asked.append(args)
                if args[:2] == ("volume", "ls"):
                    return listed
                return (0, "27.0.1") if args[0] == "version" else (1, "")

            w.ask_docker = docker
            with self.subTest(listed=listed):
                self.assertEqual(w.project_volumes("rkmon-test"), volumes)
                self.assertEqual(asked, [listing])
                facts = w.facts({"deployment": {"project": "rkmon-test"}})
                self.assertEqual(facts.project_volumes, volumes or [])
                self.assertEqual(
                    w.own_secrets_volume("rkmon-test"),
                    secrets if volumes else "",
                )
                self.assertEqual(asked.count(listing), 3)

    def test_secret_usage_errors_run_nothing(self):
        """`rkmon secret` takes a name the registry lists for a secret (or a model binding's key), its value from a
        file, stdin or typed, never from a flag's value; --generate only for a value made here, and only onto a
        terminal. Its own usage errors are `rkmon secret: ...`, with the table's usage and the names NAME takes,
        said in any state, docker not asked: a name the registry refuses among them, in its words (a login's, an older
        name, or one no secret has: secrets_registry.name_refusal). An empty stdin is no usage error: in the command's
        states, FAILED, exit 4, and the command again.
        """
        secrets = ", ".join(e.name for e in sr.entries({}) if e.shape != sr.LOGIN)
        usage = (
            f"; usage: {self.w} secret NAME [FILE | -] [--generate | --remove] (NAME: {secrets}, or the secret a "
            "model binding's api_key_secret_name names)\n"
        )
        unknown = f"{self.w} secret: unrecognized arguments:"
        for args, words in (
            (
                ("secret",),
                f"{self.w} secret: the following arguments are required: NAME",
            ),
            (
                ("secret", "Bad-Name"),
                f"{self.w} secret: a name given is no secret's, and is not shown: it does not read as a name, and may "
                f"be a value{usage}",
            ),
            (
                ("secret", "teams-hook"),
                f"{self.w} secret: no secret is named teams-hook{usage}",
            ),
            (("secret", "teams_webhook", "--value", "made-up"), f"{unknown} --value\n"),
            (("secret", "teams_webhook", "--value=made-up"), f"{unknown} --value\n"),
            (
                ("secret", "teams_webhook", "hook", "--remove"),
                f"{self.w} secret: --remove takes no FILE{usage}",
            ),
            (
                ("secret", "teams_webhook", "hook", "--generate"),
                f"{self.w} secret: --generate takes no FILE{usage}",
            ),
            (
                ("secret", "teams_webhook", "--generate", "--remove"),
                f"{self.w} secret: --generate and --remove go apart{usage}",
            ),
            (("secret", "teams_webhook", "a", "b"), f"{unknown} b\n"),
            (
                ("secret", "github_write"),
                f"{self.w} secret: github_write is a login: {self.w} login github_write sets it{usage}",
            ),
            (
                ("secret", "webhost_ed25519"),
                f"{self.w} secret: webhost_ed25519 is the older name of the secret web_host_ssh_key: {self.w} "
                f"secret web_host_ssh_key sets it{usage}",
            ),
            (
                ("secret", "admin_token"),
                f"{self.w} secret: admin_token is the older name of the secret admin_access_token: {self.w} secret "
                f"admin_access_token --generate sets it{usage}",
            ),
            (
                ("secret", "claude"),
                f"{self.w} secret: claude is a login: {self.w} login claude sets it{usage}",
            ),
            (
                ("secret", "claude_env"),
                f"{self.w} secret: claude_env is the older name of the login claude: {self.w} login claude sets "
                f"it{usage}",
            ),
            (
                ("secret", "teams_webhook", "--generate"),
                f"{self.w} secret: --generate goes with admin_access_token, whose value is made here; give "
                "teams_webhook's from a file, from stdin (-) or typed",
            ),
            (
                ("secret", "admin_access_token", "--generate"),
                f"{self.w} secret admin_access_token --generate: its value is shown once, on your terminal, and "
                "the output is no terminal: nothing made",
            ),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args)
                self.assertEqual((rc, out), (2, ""))
                self.assertIn(words, err)
                self.assertNotIn("made-up", err)  # a flag's value is never shown
                self.assertNotIn(
                    "Bad-Name", err
                )  # nor a NAME that names no secret: it may be a value
        self.assertEqual(self.calls(), [])
        self.deploy()
        rc, out, err = self.rkmon("secret", "teams_webhook")  # stdin, empty
        self.assertEqual(
            (rc, out, err),
            (
                4,
                finished(
                    "FAILED: stdin has no value for teams_webhook; nothing changed",
                    f"next: {self.w} secret teams_webhook",
                ),
                "",
            ),
        )
        self.assertEqual(self.acted(), [])

    def test_a_name_the_confirmed_settings_give_no_secret_is_refused_before_docker(
        self,
    ):
        """Once settings are confirmed, rkmon secret knows the secrets they name (confirmed_values): a name no model
        binding gives, and a binding's key by its older name, are usage errors in secrets_registry.name_refusal's
        words, docker not asked, while the binding's key goes on to rk secret-set. Before the first confirm only the
        fixed entries are known: a name that reads as a secret's goes on to the state's refusal, as a model binding
        may name it."""
        rc, out, _err = self.rkmon("secret", "glm_api_key")
        self.assertEqual(rc, 3, out)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: no deploy settings are confirmed in this package folder; nothing changed",
                    state_2_next(self.w),
                )
            ),
            out,
        )
        self.deploy(bindings=GLM_BINDINGS)
        before = self.calls()
        for name, why in (
            ("b_api_key", "no secret is named b_api_key"),
            (
                "glm_key",
                f"glm_key is the older name of the secret glm_api_key: {self.w} secret glm_api_key sets it",
            ),
        ):
            with self.subTest(name=name):
                rc, out, err = self.rkmon("secret", name)
                self.assertEqual((rc, out), (2, ""))
                self.assertTrue(err.startswith(f"{self.w} secret: {why}; usage: "), err)
        self.assertEqual(self.calls(), before)
        rc, out, err = self.rkmon("--dry-run", "secret", "glm_api_key", "-")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn(" rk secret-set glm_api_key\n", out)

    def test_secret_commands_in_the_container(self):
        """Each one runs rk secret-set in the container, the value on its stdin only: --remove, a model binding's key
        (one your confirmed settings name), stdin (-), and a value made here; a secret the writer keeps, as the
        writer (RKMON_AS: the entry point gives the writer's rk to its broker). A dry run ends with what the real run
        would do. Each in the walk's one-off container, handed your confirmed settings by the variable's name
        alone (deploy step 5): here the container is stopped."""
        self.deploy(bindings=GLM_BINDINGS)
        e = f"-e RKMON_CONFIRMED_LAYER -e RKMON_WRAPPER={self.w}"
        for args, want, would in (
            (
                ("secret", "teams_webhook", "--remove"),
                f"docker compose run --rm -T {e} -e RKMON_AS=rkwrite rkmon rk secret-set teams_webhook --remove",
                "teams_webhook can be removed",
            ),
            (
                ("secret", "glm_api_key", "-"),
                f"docker compose run --rm -T {e} rkmon rk secret-set glm_api_key",
                "glm_api_key can be set",
            ),
            (
                ("secret", "admin_access_token", "--generate"),
                f"docker compose run --rm -T {e} rkmon rk secret-set admin_access_token",
                "admin_access_token can be made and set",
            ),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon("--dry-run", *args)
                self.assertEqual(
                    (rc, out.splitlines(), err),
                    (0, [want, *self.dry_run_end(would, *args)], ""),
                )

    def test_a_fix_that_names_another_file_is_the_command_you_ran(self):
        repo = self.cwd / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "hook").write_text("config: {}\n", encoding="utf-8")
        (repo / "hook").chmod(0o600)
        self.deploy()  # the secret's command, in its states: on a deployment
        args = ("secret", "teams_webhook", "repo/hook")
        rc, out, _err = self.rkmon(*args)
        self.assertEqual(rc, 4)
        file = self.cwd / args[-1]
        move = checks.Command(("mv", file, "~/hook"))
        again = checks.Command((self.w, *args[:-1], "~/hook"))
        row = checks.Check(
            "the secret's file",
            "FAIL",
            f"{file} is inside the git work tree {repo}: a commit there would publish it",
            checks.Steps((move, ", then", again)),
        )
        self.assertIn(checks.render([row], "problems") + "\n", out)
        self.assertIn(f"fix: mv {file} ~/hook, then", one_line(out))
        self.assertEqual(self.acted(), [])  # the file goes nowhere

    def test_the_deployments_rkmon_through_a_link_on_the_path_is_named_as_called(self):
        """The deployment's rkmon, through a link to it on the PATH, called by its name: the commands it gives name
        it so (RKMON_WRAPPER), and every docker call runs in the deployment's folder."""
        self.deploy()
        self.log.unlink(missing_ok=True)
        os.symlink(self.w, pathlib.Path(self.bin) / "rkmon")
        p = subprocess.run(
            ["rkmon", "--dry-run", "doctor"],
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertEqual(
            p.stdout.splitlines(),
            [
                DOCTOR,
                "docker compose run --rm -T -e RKMON_WRAPPER=rkmon rkmon rk doctor",
                *finished(
                    "PASSED: rk doctor can run in the container", "next: rkmon doctor"
                ).splitlines(),
            ],
        )
        self.assertTrue(all(c[0] == self.docker_dir for c in self.calls()))

    def test_a_terminal_gets_one(self):
        # rkmon rk runs rk on the terminal, in the running container; rkmon doctor reads rk doctor's report (-T), to
        # end it with one outcome
        self.deploy()
        rc, text = sandbox.on_a_terminal(
            [self.w, "--dry-run", "rk", "doctor"],
            self.cwd,
            dict(self.env, TERM="xterm", FAKE_DOCKER_RUNNING="1"),
        )
        self.assertEqual(rc, 0)
        self.assertIn(
            f"docker compose exec -it --user rkmon -e RKMON_WRAPPER={self.w} rkmon rk doctor",
            text,
        )
        self.assertIn("\033[2m", text)  # dim on a terminal

    def test_the_wrapper_itself(self):
        """Python on the host (as tools/make_package.py, which builds the folder the image is built from), run
        directly: executable, with a python3 #! line, and it compiles."""
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/usr/bin/env python3\n"))
        self.assertTrue(os.access(WRAPPER, os.X_OK))
        compile(text, str(WRAPPER), "exec")

    def test_image_ls_has_a_line_for_each_tag_of_the_package_s_images_from_two_questions(
        self,
    ):
        """rkmon image ls: a line for each tag of each image with the package's title label, with what its labels say
        (image.describe), and none for an image with no tag; docker is asked twice, whatever the number of images:
        their IDs (an image's once for each of its tags), then each one's tags and labels.
        """
        labels = {
            build_plan.PACKAGE_LABEL: "2026.01.02",
            build_plan.DATE_LABEL: "2026-01-02T03:04:05Z",
            build_plan.BASE_LABEL: "ubuntu:24.04",
            build_plan.ROCM_LABEL: "7.0.0",
        }
        tagged = {
            "Id": "sha256:" + "a" * 64,
            "RepoTags": ["team/rkmon:1", "team/rkmon:latest"],
            "Config": {"Labels": labels},
        }
        untagged = {
            "Id": "sha256:" + "b" * 64,
            "RepoTags": [],
            "Config": {"Labels": labels},
        }
        listing = f"image ls --quiet --no-trunc --filter label=org.opencontainers.image.title={image.TITLE}"
        inspect = f"image inspect {tagged['Id']} {untagged['Id']}"
        self.answer(
            [
                [
                    f"^{re.escape(listing)}$",
                    f"{tagged['Id']}\n{tagged['Id']}\n{untagged['Id']}\n",
                    0,
                ],
                [f"^{re.escape(inspect)}$", json.dumps([tagged, untagged]), 0],
            ]
        )
        line = image.describe(labels)
        self.assertEqual(
            self.rkmon("image", "ls"),
            (0, f"team/rkmon:1  {line}\nteam/rkmon:latest  {line}\n", ""),
        )
        self.assertEqual(
            [c for _, c in self.calls() if c.startswith("image ")], [listing, inspect]
        )
        self.answer()  # no image of the package's here
        self.assertEqual(
            self.rkmon("image", "ls"), (0, "no image of this package here\n", "")
        )

    def test_image_code_and_another_folders_container(self):
        self.assertEqual(self.rkmon("--dry-run", "start", "--yes")[0], 2)
        self.deploy()
        e = f"-e RKMON_WRAPPER={self.w}"
        rc, out, _err = self.rkmon(
            "--dry-run",
            "--image-code",
            "rk",
            "doctor",
            FAKE_DOCKER_RUNNING="1",
        )
        self.assertEqual(
            out.splitlines(),
            [
                f"docker compose exec -T --user rkmon {e} -e RKMON_CODE=image rkmon rk doctor"
            ],
        )
        # The container name is another deployment's (its container made from another folder): state 1, nothing runs,
        # and no message names that folder.
        self.log.unlink()
        rc, out, err = self.rkmon("doctor", FAKE_DOCKER_WORKDIR="/srv/other/docker")
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"REFUSED: the container name {deploy_folder.NAME} is used by another deployment on this host; nothing "
            f"changed fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as "
            f"{deploy_folder.NAME}-2 (docker compose ls --all lists the names in use) next: step 3, confirm: "
            f"{self.w} confirm --settings ~/rkmon-deploy.yaml",
            one_line(out),
        )
        self.assertNotIn("/srv/other", out + err)
        self.assertFalse(
            any(c[1].startswith("compose") for c in self.calls())
        )  # only the read-only questions ran
        rc, out, err = self.rkmon(
            "doctor", FAKE_DOCKER_RUNNING="1"
        )  # this folder's container
        self.assertEqual(rc, 0, out + err)


if __name__ == "__main__":
    unittest.main()
