# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one writer of what a deployment runs (rkmon_lib/deliver.py): docker/.env and the
override at 600 with the previous ones kept, a link refused with nothing written, and the layer through the one-off
container in the plan's compose project and image, never those docker/.env names (here docker compose is a fake that
picks them as the real one does, docker/.env among what it reads, and runs the program on a folder of the test's, as
the test's own user). A run that stops before its commit changes nothing; one that stops after it, never starts or is
interrupted says what it may have changed, and a delivery of the same plan then ends where it would have ended.
Putting back what was kept, in the compose project and with the image it is given. The fingerprint of each file
written, kept in the records as it is written, against which state 9 finds a hand edit; and the change plan's rule of
what apply would change.
"""

import ast
import dataclasses
import hashlib
import itertools
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import deliver, deploy_plan, image, private_files, records  # noqa: E402
from test_deploy_plan import CODE, PACKAGE, SERVED, doc  # noqa: E402

HOST = deploy_plan.Host(os.getuid(), os.getgid(), "2026-01-02T03:04:05Z")
RAMDISK = {"host.resources.ramdisk_gib": 2}
PROJECT = "rocke-monitor"  # the plans' compose project: container.name's default
IMAGE = "rocke-monitor:1.2.3"  # the plans' image
# docker compose's exit code when nothing names the project: compose.yaml's name requires COMPOSE_PROJECT_NAME
NO_PROJECT = 1
COMPOSE_IMAGE = (
    "rocke-monitor",
    "local",
)  # compose.yaml's image, when nothing names one

# Faults for the one-off run's program, run just before its call, at its commit: the rename of the new layer into
# place.
_AT_COMMIT = """\
_rename = os.replace


def _at_commit(src, dst):
    if dst == CURRENT:
        {before}
    _rename(src, dst)
    if dst == CURRENT:
        {after}


os.replace = _at_commit
"""
NO_SPACE = _AT_COMMIT.format(
    before='raise OSError(28, "No space left on device")', after="pass"
)
KILLED_BEFORE = _AT_COMMIT.format(
    before="os.kill(os.getpid(), signal.SIGKILL)", after="pass"
)
KILLED_AFTER = _AT_COMMIT.format(
    before="pass", after="os.kill(os.getpid(), signal.SIGKILL)"
)
KILLED = 128 + 9  # docker's exit code for a container that SIGKILL ended
# A fault that stops the program (SIGKILL) just before its Nth call that may change the settings volume.
STOP = """\
_steps = [0]


def _stopping(real):
    def call(*args, **kwargs):
        _steps[0] += 1
        if _steps[0] == {n}:
            os.kill(os.getpid(), signal.SIGKILL)
        return real(*args, **kwargs)

    return call


for _name in ("open", "mkdir", "link", "replace", "unlink", "rmdir", "chown", "chmod", "fchown", "fchmod", "fsync"):
    setattr(os, _name, _stopping(getattr(os, _name)))
"""
# The team's skills: three versions of the folder, each file's path in it -> its text.
VERIFIER = "team_members/rocke_review_verifier.md"
V1 = {VERIFIER: "# v1\n", "review_book.md": "# book\n"}
V2 = {VERIFIER: "# v2\n", "agents/rocke-review-verifier.md": "# agent\n"}
V3 = {VERIFIER: "# v3\n"}

KEPT_AS_THEY_WERE = "; their kept copies (.prev) are the files as they were"
UNKNOWN_KEPT = (
    "the settings volume has either its layers as they were or the new one, the"
    " earlier one kept; docker/.env and the override are as they were"
    + KEPT_AS_THEY_WERE
)
UNKNOWN = (
    "the settings volume has either its layer as it was or the new one;"
    " docker/.env and the override are as they were"
)
# why the layer in the settings volume cannot be read, as docker/rkmon's read says it
UNREAD = (
    "docker compose run exited 1: Error response from daemon: no space left on device"
)


def plan(changes=None):
    return deploy_plan.plan(doc(changes), PACKAGE, CODE, HOST)


def layer_text(p) -> str:
    return deploy_plan.layer_text(p.layer)


class Compose:
    """docker compose for the tests, which picks its compose project and image as the real one does: the project from
    `-p`, else COMPOSE_PROJECT_NAME in its environment (this process's, with the variables it is given over it), else
    docker/.env's, else none, and then it refuses (NO_PROJECT); the image from RKMON_IMAGE and RKMON_TAG in its
    environment, else docker/.env's, else compose.yaml's defaults. Each call is kept (CALLS: its arguments, input and
    variables), and the project and image each one ran under (RAN). Deliver's one-off run (LAYER_RUN) runs its
    program with this Python on the project's settings volume, the folder VOLUMES/<project>_rkmon-settings (made when
    it is not there, as docker makes a volume), the owner being the test's own user, with FAULT (Python lines) just
    before the program's call; then THEN(), when given. Its exit code is docker's: 128 + N for a program that signal N
    ended. With CODE, that exit code, and nothing run.
    """

    def __init__(self, volumes, pkg, code=None, fault="", then=None):
        self.volumes, self.pkg, self.code = volumes, pkg, code
        self.fault, self.then = fault, then
        self.calls, self.ran, self.stderr = [], [], ""

    def volume(self, project: str) -> pathlib.Path:
        return self.volumes / f"{project}_{deliver.SETTINGS_VOLUME}"

    def __call__(self, args: list, stdin: bytes, variables: dict) -> int:
        self.calls.append((list(args), stdin, dict(variables)))
        dotenv = image.dotenv(self.pkg / deliver.ENV_FILE)
        found = {**dotenv, **os.environ, **variables}
        args = list(args)
        project = found.get("COMPOSE_PROJECT_NAME") or ""
        if args[:1] == ["-p"]:
            project, args = args[1], args[2:]
        name = found.get("RKMON_IMAGE") or COMPOSE_IMAGE[0]
        tag = found.get("RKMON_TAG") or COMPOSE_IMAGE[1]
        self.ran.append((project, f"{name}:{tag}"))
        if self.code is not None:
            return self.code
        if not project:
            return NO_PROJECT
        assert args == list(deliver.LAYER_RUN), args
        folder = self.volume(project)
        folder.mkdir(exist_ok=True)
        head, call = stdin.decode("utf-8").rstrip("\n").rsplit("\n", 1)
        program = f"{head}\n{self.fault}\n{call}\n".encode("utf-8")
        argv = [sys.executable, "-B", "-I", "-", str(folder), str(os.getuid())]
        done = subprocess.run(argv, input=program, stderr=subprocess.PIPE, check=False)
        self.stderr = done.stderr.decode("utf-8", "replace")
        if self.then:
            self.then()
        return 128 - done.returncode if done.returncode < 0 else done.returncode


def interrupted(args: list, stdin: bytes, variables: dict) -> int:
    raise KeyboardInterrupt


def mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


class Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.pkg = self.tmp / "pkg"
        (self.pkg / "docker").mkdir(parents=True)
        self.volumes = self.tmp / "volumes"
        self.volumes.mkdir()
        self.compose = self.fake()
        self.volume = self.compose.volume(PROJECT)
        self.env = self.pkg / deliver.ENV_FILE
        self.override = self.pkg / deliver.OVERRIDE_FILE

    def fake(self, code=None, fault="", then=None) -> Compose:
        return Compose(self.volumes, self.pkg, code, fault, then)

    def deliver_plan(self, p, keep_previous=True, compose=None):
        compose = compose or self.compose
        deliver.deliver(p, keep_previous, package_dir=self.pkg, compose=compose)

    def put_back_kept(self, compose=None, variables=None):
        """deliver.put_back in the deployment's compose project and image: VARIABLES, the plans' by default."""
        variables = plan().variables if variables is None else variables
        compose = compose or self.compose
        return deliver.put_back(variables, package_dir=self.pkg, compose=compose)

    def kept(self, path: pathlib.Path) -> pathlib.Path:
        return path.with_name(path.name + deliver.KEPT)

    def text(self, path: pathlib.Path):
        return path.read_text(encoding="utf-8") if os.path.lexists(path) else None

    def layer(self, name=deliver.LAYER_FILE):
        return self.text(self.volume / name)

    def host_files(self) -> dict:
        files = (self.env, self.override, self.kept(self.env), self.kept(self.override))
        return {f.name: self.text(f) for f in files}

    def volume_files(self) -> dict:
        names = sorted(os.listdir(self.volume))
        return {name: self.text(self.volume / name) for name in names}

    def everything(self) -> tuple:
        """What a delivery may change: the host's files with their kept copies, and the settings volume's files."""
        return self.host_files(), self.volume_files()

    def read_layer(self) -> tuple:
        """The layer in this settings volume, as docker/rkmon reads it for deliver.keep_previous: (its text, "")."""
        return self.layer(), ""

    def layers(self, current, previous) -> dict:
        """The settings volume's files when it holds CURRENT and PREVIOUS, plans (None: none), and nothing else."""
        out = {deliver.LAYER_FILE: layer_text(current)}
        if previous is not None:
            out[deliver.LAYER_KEPT] = layer_text(previous)
        return dict(sorted(out.items()))

    def team(self, files: dict) -> dict:
        """The team's skills folder, holding FILES (each one's path in it -> its text) and nothing else: the changes
        to the settings that name it."""
        top = self.tmp / "team"
        if top.exists():
            shutil.rmtree(top)
        top.mkdir()
        for rel, text in files.items():
            (top / rel).parent.mkdir(parents=True, exist_ok=True)
            (top / rel).write_text(text, encoding="utf-8")
        return {deploy_plan.SKILLS_KEY: str(top)}

    def skills(self, name=deliver.SKILLS_FOLDER):
        """The skills folder NAME in the settings volume: each file's path in it -> its text; None when it is not
        there."""
        top = self.volume / name
        if not os.path.lexists(top):
            return None
        found = sorted(p for p in top.rglob("*") if p.is_file())
        return {
            p.relative_to(top).as_posix(): p.read_text(encoding="utf-8") for p in found
        }

    def tree(self) -> dict:
        """The settings volume: each path in it -> its mode and its bytes (None for a folder)."""
        return {
            p.relative_to(self.volume).as_posix(): (
                mode(p),
                None if p.is_dir() else p.read_bytes(),
            )
            for p in sorted(self.volume.rglob("*"))
        }

    def called(self, compose=None) -> tuple:
        """The last one-off run's call: its function's name and its arguments, as values."""
        stdin = (compose or self.compose).calls[-1][1]
        call = stdin.decode("utf-8").rstrip("\n").rsplit("\n", 1)[1]
        node = ast.parse(call, mode="eval").body
        return node.func.id, tuple(ast.literal_eval(a) for a in node.args)


class TestNames(unittest.TestCase):
    def test_the_package_files_hold_the_two_files_and_their_kept_copies(self):
        e, o = deliver.ENV_FILE, deliver.OVERRIDE_FILE
        self.assertEqual(
            private_files.PACKAGE_FILES[:4],
            (e, o, e + deliver.KEPT, o + deliver.KEPT),
        )
        self.assertEqual((e, o), ("docker/.env", "docker/compose.override.yaml"))

    def test_what_deliver_writes(self):
        self.assertEqual(deliver.LAYER_NAME, "the settings volume's deploy.json")
        self.assertEqual(
            deliver.WRITES,
            (deliver.ENV_FILE, deliver.OVERRIDE_FILE, deliver.LAYER_NAME),
        )

    def test_the_one_off_run(self):
        """A service of its own, never the monitor's: the compose file gives it the settings volume and no other
        (tests/test_container_layout.py holds it to that)."""
        run = deliver.LAYER_RUN
        self.assertEqual(run[:4], ("run", "--rm", "-T", "--no-deps"))
        self.assertEqual(run[run.index("--user") + 1], "0:0")
        self.assertEqual(run[run.index("--entrypoint") + 1], "python3")
        self.assertFalse({"-v", "--volume"} & set(run), run)
        self.assertNotEqual(deliver.LAYER_SERVICE, deploy_plan.SERVICE)
        self.assertEqual(
            run[run.index(deliver.LAYER_SERVICE) :],
            (deliver.LAYER_SERVICE, "-I", "-", deliver.LAYER_WRITE_DIR, "0"),
        )


class TestDeliver(Case):
    def test_a_first_delivery(self):
        p = plan(RAMDISK)
        self.deliver_plan(p)
        self.assertEqual(self.text(self.env), p.env)
        self.assertEqual(self.text(self.override), p.override)
        for f in (self.env, self.override):
            self.assertEqual(mode(f), 0o600, f)
            self.assertFalse(os.path.lexists(self.kept(f)), "nothing was there to keep")
        self.assertEqual(self.volume_files(), self.layers(p, None), "nothing to keep")
        self.assertEqual(mode(self.volume / deliver.LAYER_FILE), 0o640)
        self.assertEqual(os.stat(self.volume / deliver.LAYER_FILE).st_gid, HOST.gid)
        self.assertEqual(mode(self.volume), 0o750)
        self.assertFalse(deliver.kept(self.pkg))

    def test_the_layer_comes_through_the_one_off_run(self):
        p = plan()
        self.deliver_plan(p)
        [(args, stdin, variables)] = self.compose.calls
        self.assertEqual(args, ["-p", PROJECT, *deliver.LAYER_RUN])
        self.assertEqual(variables, p.variables)
        program = stdin.decode("utf-8")
        self.assertIn(deliver.LAYER_PROGRAM, program)
        call = f"deliver({layer_text(p)!r}, {HOST.gid}, True, None)"  # None: no team's skills
        self.assertTrue(program.endswith(call + "\n"))
        self.assertEqual(stdin, deliver.layer_program(call))

    def test_a_first_start_runs_in_the_plans_project_and_image(self):
        # no docker/.env: compose alone names no project (compose.yaml's name requires COMPOSE_PROJECT_NAME)
        self.deliver_plan(plan())
        self.assertEqual(self.compose.ran, [(PROJECT, IMAGE)])
        self.assertEqual(os.listdir(self.volumes), [self.volume.name])

    def test_a_new_container_name_runs_in_its_own_project(self):
        first = plan()
        second = plan({"container.name": "team-b", "image.tag": "2.0.0"})
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.assertEqual(
            self.compose.ran,
            [(PROJECT, IMAGE), ("team-b", "rocke-monitor:2.0.0")],
        )
        self.assertEqual(self.volume_files(), self.layers(first, None), "as it was")
        self.volume = self.compose.volume("team-b")
        self.assertEqual(self.volume_files(), self.layers(second, None))
        self.assertEqual(self.text(self.env), second.env)

    def test_the_previous_files_are_kept(self):
        first, second = plan(RAMDISK), plan({"host.resources.cpus": 6})
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.assertEqual(self.text(self.env), second.env)
        self.assertEqual(self.text(self.kept(self.env)), first.env)
        self.assertIsNone(self.text(self.override), "the second plan makes none")
        self.assertEqual(self.text(self.kept(self.override)), first.override)
        for f in (self.env, self.kept(self.env), self.kept(self.override)):
            self.assertEqual(mode(f), 0o600, f)
        self.assertEqual(self.volume_files(), self.layers(second, first))
        self.assertEqual(mode(self.volume / deliver.LAYER_KEPT), 0o640)
        self.assertTrue(deliver.kept(self.pkg))

    def test_a_file_that_was_not_there_is_kept_as_none(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        self.deliver_plan(plan({**RAMDISK, **SERVED}))
        self.assertEqual(self.text(self.kept(self.override)), plan(RAMDISK).override)
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        self.assertFalse(os.path.lexists(self.kept(self.override)))

    def test_without_keeping_the_kept_set_stays(self):
        first, second = plan(), plan(RAMDISK)
        third = plan({"host.resources.memory_gib": 12})
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.deliver_plan(third, keep_previous=False)
        self.assertEqual(self.text(self.env), third.env)
        self.assertIsNone(self.text(self.override))
        self.assertEqual(self.text(self.kept(self.env)), first.env)
        self.assertFalse(os.path.lexists(self.kept(self.override)))
        self.assertEqual(self.volume_files(), self.layers(third, first))
        self.assertEqual(
            self.called(), ("deliver", (layer_text(third), HOST.gid, False, None))
        )

    def test_a_file_an_older_package_wrote_looser_is_kept_at_600(self):
        self.env.write_text("RKMON_TAG=old\n", encoding="utf-8")
        os.chmod(self.env, 0o644)
        self.deliver_plan(plan())
        self.assertEqual(self.text(self.kept(self.env)), "RKMON_TAG=old\n")
        self.assertEqual(mode(self.kept(self.env)), 0o600)
        self.assertEqual(mode(self.env), 0o600)
        self.assertEqual(self.compose.ran, [(PROJECT, IMAGE)], "not docker/.env's tag")

    def test_a_link_is_refused_and_nothing_is_written(self):
        self.deliver_plan(plan())
        target = self.pkg.parent / "elsewhere"
        target.write_text("RKMON_TAG=mine\n", encoding="utf-8")
        for name in (deliver.ENV_FILE, deliver.OVERRIDE_FILE + deliver.KEPT):
            with self.subTest(name=name):
                link = self.pkg / name
                was = self.text(link)
                if was is not None:
                    link.unlink()
                link.symlink_to(target)
                before = (self.everything(), len(self.compose.calls))
                with self.assertRaises(private_files.Refused):
                    self.deliver_plan(plan(RAMDISK))
                after = (self.everything(), len(self.compose.calls))
                self.assertEqual(after, before)
                self.assertTrue(os.path.islink(link))
                self.assertEqual(target.read_text(encoding="utf-8"), "RKMON_TAG=mine\n")
                link.unlink()
                if was is not None:
                    private_files.write(link, was, self.pkg)


class TestARunThatStops(Case):
    """A delivery whose one-off run stops: before its commit, after it, before it starts, or by an interrupt."""

    def setUp(self):
        super().setUp()
        self.first, self.second, self.new = plan(), plan(RAMDISK), plan(SERVED)
        self.deliver_plan(self.first)
        self.deliver_plan(self.second)

    def failed(self, keep_previous=True, **fake) -> deliver.Failed:
        with self.assertRaises(deliver.Failed) as caught:
            self.deliver_plan(self.new, keep_previous, compose=self.fake(**fake))
        return caught.exception

    def test_before_its_commit_nothing_changes(self):
        before = self.everything()
        for keep in (True, False):
            with self.subTest(keep_previous=keep):
                compose = self.fake(fault=NO_SPACE)
                with self.assertRaises(deliver.Failed) as caught:
                    self.deliver_plan(self.new, keep, compose=compose)
                failed = caught.exception
                self.assertEqual((failed.code, failed.changed), (deliver.UNCHANGED, ""))
                self.assertEqual(
                    str(failed),
                    "the one-off container that writes the settings volume could not"
                    " change it: nothing changed",
                )
                self.assertIn("No space left on device", compose.stderr)
                self.assertEqual(self.everything(), before, "kept copies put back")

    def test_killed_before_its_commit_a_delivery_again_ends_as_it_would_have(self):
        files = self.host_files()
        failed = self.failed(fault=KILLED_BEFORE)
        self.assertEqual((failed.code, failed.changed), (KILLED, UNKNOWN_KEPT))
        self.assertIn(f"ended with exit code {KILLED}: {UNKNOWN_KEPT}", str(failed))
        kept = {
            ".env.prev": files[".env"],
            "compose.override.yaml.prev": files["compose.override.yaml"],
        }
        self.assertEqual(self.host_files(), {**files, **kept})
        self.assertEqual(
            self.layer(), layer_text(self.second), "the layers as they were"
        )
        self.assertEqual(self.layer(deliver.LAYER_KEPT), layer_text(self.first))
        # the delivered layer is not the new plan's, nor do the records say so: the delivery again keeps the
        # previous ones
        self.assertIs(deliver.keep_previous(self.pkg, self.new, self.read_layer), True)
        self.deliver_plan(self.new)
        self.assertEqual(self.volume_files(), self.layers(self.new, self.second))
        self.assertEqual(self.text(self.env), self.new.env)
        self.assertEqual(self.text(self.kept(self.env)), self.second.env)
        self.assertEqual(self.text(self.kept(self.override)), self.second.override)

    def test_killed_after_its_commit_a_delivery_again_ends_as_it_would_have(self):
        files = self.host_files()
        failed = self.failed(fault=KILLED_AFTER)
        self.assertEqual((failed.code, failed.changed), (KILLED, UNKNOWN_KEPT))
        self.assertEqual(self.layer(), layer_text(self.new), "the new layer")
        self.assertEqual(self.text(self.env), files[".env"])
        # the delivered layer is the new plan's, though the records do not say so yet: the delivery again keeps
        # nothing, and finishes the journal
        self.assertIs(deliver.keep_previous(self.pkg, self.new, self.read_layer), False)
        self.deliver_plan(self.new, keep_previous=False)
        self.assertEqual(self.volume_files(), self.layers(self.new, self.second))
        self.assertEqual(self.text(self.env), self.new.env)
        self.assertEqual(self.text(self.kept(self.env)), self.second.env)

    def test_killed_after_its_commit_a_put_back_puts_back_the_earlier_layer(self):
        self.failed(fault=KILLED_AFTER)
        self.assertTrue(self.put_back_kept())
        self.assertEqual(self.volume_files(), self.layers(self.second, self.second))
        self.assertEqual(self.text(self.env), self.second.env)
        self.assertEqual(self.text(self.override), self.second.override)

    def test_a_run_that_never_starts(self):
        for keep, unknown in ((True, UNKNOWN_KEPT), (False, UNKNOWN)):
            with self.subTest(keep_previous=keep):
                files, layers = self.host_files(), self.volume_files()
                failed = self.failed(keep, code=125)
                self.assertEqual((failed.code, failed.changed), (125, unknown))
                self.assertIn("ended with exit code 125", str(failed))
                self.assertEqual(self.volume_files(), layers)
                self.assertEqual(self.text(self.env), files[".env"])
                if keep:
                    self.assertEqual(self.text(self.kept(self.env)), files[".env"])
                else:
                    self.assertEqual(self.host_files(), files)

    def test_an_interrupt(self):
        files = self.host_files()
        with self.assertRaises(deliver.Interrupted) as caught:
            self.deliver_plan(self.new, compose=interrupted)
        self.assertIsInstance(caught.exception, KeyboardInterrupt)
        self.assertEqual(caught.exception.changed, UNKNOWN_KEPT)
        self.assertEqual(self.text(self.env), files[".env"])
        self.assertEqual(self.text(self.kept(self.env)), files[".env"])

    def test_a_file_that_cannot_be_written_after_the_run(self):
        for rel, files in (
            (deliver.ENV_FILE, "docker/.env and the override are as they were"),
            (
                deliver.OVERRIDE_FILE,
                "docker/.env is the new one, and the override is as it was",
            ),
        ):
            with self.subTest(file=rel):
                self.setUp()
                path = self.pkg / rel

                def no_file():
                    path.unlink()
                    path.mkdir()

                failed = self.failed(then=no_file)
                self.assertEqual(failed.code, 0)
                # private_files refuses a folder in the file's place: the refusal shows itself above the outcome
                self.assertTrue(
                    str(failed).startswith(
                        f"{rel} could not be written (the lines above say why): "
                    ),
                    str(failed),
                )
                self.assertIn(
                    f"/{rel} is not a regular file", " ".join(failed.above.split())
                )
                self.assertTrue(
                    failed.above.endswith(private_files.RULES), failed.above
                )
                self.assertEqual(
                    failed.changed,
                    "the settings volume has the new layer, the earlier one kept; "
                    + files
                    + KEPT_AS_THEY_WERE,
                )
                self.assertEqual(
                    self.volume_files(), self.layers(self.new, self.second)
                )
                if rel == deliver.OVERRIDE_FILE:
                    self.assertEqual(self.text(self.env), self.new.env)

    def test_a_put_back_then_a_delivery(self):
        self.assertTrue(
            self.put_back_kept()
        )  # the two layers are one file, by two names
        self.deliver_plan(self.new)
        self.assertEqual(self.volume_files(), self.layers(self.new, self.first))


class TestKeepPrevious(Case):
    """Whether a delivery keeps what is in place (deliver.keep_previous): the records first, then the layer in the
    settings volume, read only when they do not say; never a keep while it cannot be read.
    """

    def setUp(self):
        super().setUp()
        self.first, self.second = plan(), plan(RAMDISK)
        self.deliver_plan(self.first)
        self.reads = 0

    def keep(self, made, found=None, why="") -> bool:
        """keep_previous for MADE, a plan, the layer in the settings volume read as (FOUND, WHY)."""

        def read():
            self.reads += 1
            return found, why

        return deliver.keep_previous(self.pkg, made, read)

    def test_the_plan_the_records_say_is_in_place_keeps_nothing_unread(self):
        # whatever the settings volume holds: the plan's layer, a hand edit, none, or one that cannot be read
        for found, why in (
            (layer_text(self.first), ""),
            ("{}", ""),
            (None, ""),
            (None, UNREAD),
        ):
            with self.subTest(found=found, why=why):
                self.assertIs(self.keep(self.first, found, why), False)
        self.assertEqual(self.reads, 0)

    def test_another_plan_keeps_what_is_in_place(self):
        self.assertIs(self.keep(self.second, layer_text(self.first)), True)
        self.assertIs(self.keep(self.second, "{}"), True)
        self.assertIs(self.keep(self.second, None), True)  # none there: kept as none
        self.assertEqual(self.reads, 3)

    def test_the_plan_s_layer_in_place_keeps_nothing(self):
        # a run that stopped after the layer's rename, before its fingerprint was kept
        self.assertIs(self.keep(self.second, layer_text(self.second)), False)
        self.assertEqual(self.reads, 1)

    def test_records_that_do_not_say_leave_it_to_the_layer_in_place(self):
        # an older package's records keep no layer's fingerprint; a folder never delivered to keeps none
        records.change_written(self.pkg, lambda w: dataclasses.replace(w, layer=""))
        self.assertIs(self.keep(self.first, layer_text(self.first)), False)
        self.assertIs(self.keep(self.first, "{}"), True)
        other = self.tmp / "other"
        (other / "docker").mkdir(parents=True)
        self.assertIsNone(records.read(other).written)

        def read():
            return layer_text(self.first), ""

        self.assertIs(deliver.keep_previous(other, self.first, read), False)
        self.assertIs(deliver.keep_previous(other, self.second, read), True)

    def test_a_layer_that_cannot_be_read_is_never_a_keep(self):
        before = self.everything()
        with self.assertRaises(deliver.Failed) as caught:
            self.keep(self.second, None, UNREAD)
        self.assertEqual((caught.exception.code, caught.exception.changed), (0, ""))
        self.assertEqual(
            str(caught.exception),
            f"the settings volume's deploy.json could not be read, to keep it for rollback ({UNREAD}): nothing changed",
        )
        self.assertEqual(self.everything(), before)


class TestPutBack(Case):
    def test_what_was_kept_goes_back(self):
        first, second = plan(RAMDISK), plan(SERVED)
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.assertTrue(self.put_back_kept())
        self.assertEqual(self.text(self.env), first.env)
        self.assertEqual(self.text(self.override), first.override)
        self.assertEqual(self.layer(), layer_text(first))
        for f in (self.env, self.override):
            self.assertEqual(mode(f), 0o600, f)
        self.assertEqual(mode(self.volume / deliver.LAYER_FILE), 0o640)
        self.assertEqual(mode(self.volume), 0o750)
        # the kept copies stay, so a put back that stopped part-way runs again to the same end
        after = self.everything()
        self.assertEqual(self.text(self.kept(self.env)), first.env)
        self.assertTrue(self.put_back_kept())
        self.assertEqual(self.everything(), after)
        self.assertTrue(self.compose.calls[-1][1].endswith(b"put_back()\n"))

    def test_it_runs_in_the_project_and_with_the_image_it_is_given(self):
        # never in those docker/.env names, which a hand edit changed here
        first, second = plan(), plan({"image.tag": "2.0.0"})
        self.deliver_plan(first)
        self.deliver_plan(second)
        text = self.text(self.env).replace(
            f"COMPOSE_PROJECT_NAME={PROJECT}\n", "COMPOSE_PROJECT_NAME=otherdeploy\n"
        )
        text = text.replace("RKMON_TAG=2.0.0\n", "RKMON_TAG=9.9.9\n")
        found = image.dotenv_text(text)
        self.assertEqual(
            (found["COMPOSE_PROJECT_NAME"], found["RKMON_TAG"]),
            ("otherdeploy", "9.9.9"),
        )
        private_files.write(self.env, text, self.pkg, replace=True)
        self.assertTrue(self.put_back_kept(variables=second.variables))
        args, _, variables = self.compose.calls[-1]
        self.assertEqual(args, ["-p", PROJECT, *deliver.LAYER_RUN])
        self.assertEqual(variables, second.variables)
        self.assertEqual(self.compose.ran[-1], (PROJECT, "rocke-monitor:2.0.0"))
        self.assertFalse(self.compose.volume("otherdeploy").exists())
        self.assertEqual(self.layer(), layer_text(first))

    def test_variables_that_name_no_project(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        before = (self.everything(), len(self.compose.calls))
        for variables in ({}, dict(plan().variables, COMPOSE_PROJECT_NAME="")):
            with self.subTest(variables=variables):
                with self.assertRaises(ValueError) as caught:
                    self.put_back_kept(variables=variables)
                self.assertEqual(
                    str(caught.exception),
                    "no compose project (COMPOSE_PROJECT_NAME) is given to put back in:"
                    " nothing changed",
                )
                self.assertEqual((self.everything(), len(self.compose.calls)), before)

    def test_an_override_with_no_kept_copy_goes(self):
        first, second = plan(), plan(RAMDISK)
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.assertTrue(self.put_back_kept())
        self.assertIsNone(self.text(self.override))
        self.assertEqual(self.text(self.env), first.env)

    def test_nothing_kept(self):
        self.assertFalse(self.put_back_kept())
        self.deliver_plan(plan())
        self.assertFalse(self.put_back_kept())
        self.assertEqual(len(self.compose.calls), 1, "only the delivery ran")

    def test_what_was_kept_differs_until_it_is_put_back(self):
        """kept_differs (rkmon rollback): False with nothing kept, a first delivery's among them; True once a delivery
        kept another version; False once it is put back, the kept copies staying; True again while the records say
        another layer is in place."""
        self.assertFalse(deliver.kept_differs(self.pkg))
        self.deliver_plan(plan())
        self.assertFalse(
            deliver.kept_differs(self.pkg), "a first delivery keeps nothing"
        )
        self.deliver_plan(plan(RAMDISK))
        self.assertTrue(deliver.kept_differs(self.pkg))
        self.assertTrue(self.put_back_kept())
        self.assertFalse(deliver.kept_differs(self.pkg))
        other = "sha256:" + "0" * 64
        records.change_written(self.pkg, lambda w: dataclasses.replace(w, layer=other))
        self.assertTrue(deliver.kept_differs(self.pkg), "another layer is in place")

    def test_a_volume_that_keeps_no_layer(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        os.unlink(self.volume / deliver.LAYER_KEPT)
        before = self.everything()
        with self.assertRaises(deliver.Failed) as caught:
            self.put_back_kept()
        self.assertEqual(caught.exception.code, deliver.NOTHING_KEPT)
        self.assertEqual(
            str(caught.exception),
            "the settings volume keeps no earlier layer to put back: nothing changed",
        )
        self.assertEqual(self.everything(), before)

    def test_a_run_that_stops(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        files = self.host_files()
        unknown = (
            "the settings volume has either its layer as it was or the kept one;"
            " docker/.env and the override are as they were"
        )
        for compose, code, changed in (
            (self.fake(fault=NO_SPACE), deliver.UNCHANGED, ""),
            (self.fake(code=1), 1, unknown),
            (self.fake(fault=KILLED_BEFORE), KILLED, unknown),
        ):
            with self.subTest(code=code):
                with self.assertRaises(deliver.Failed) as caught:
                    self.put_back_kept(compose=compose)
                failed = caught.exception
                self.assertEqual((failed.code, failed.changed), (code, changed))
                self.assertEqual(self.host_files(), files)
        with self.assertRaises(deliver.Interrupted) as caught:
            self.put_back_kept(compose=interrupted)
        self.assertEqual(caught.exception.changed, unknown)
        self.assertTrue(self.put_back_kept())
        self.assertEqual(self.volume_files(), self.layers(plan(), plan()))

    def test_a_link_is_refused(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(RAMDISK))
        kept = self.kept(self.env)
        os.unlink(kept)
        kept.symlink_to(self.env)
        calls = len(self.compose.calls)
        with self.assertRaises(private_files.Refused):
            self.put_back_kept()
        self.assertEqual(len(self.compose.calls), calls)


class TestOnHost(Case):
    def test_what_deliver_wrote(self):
        self.assertEqual(deliver.on_host(self.pkg), (None, None))
        p = plan(RAMDISK)
        self.deliver_plan(p)
        self.assertEqual(deliver.on_host(self.pkg), (p.env, p.override))
        self.deliver_plan(plan())
        self.assertEqual(deliver.on_host(self.pkg), (plan().env, None))

    def test_what_deliver_kept(self):
        self.assertEqual(deliver.kept_on_host(self.pkg), (None, None))
        p = plan(RAMDISK)
        self.deliver_plan(p)
        self.assertEqual(deliver.kept_on_host(self.pkg), (None, None))
        self.deliver_plan(plan())
        self.assertEqual(deliver.kept_on_host(self.pkg), (p.env, p.override))
        os.chmod(self.kept(self.env), 0o644)
        with self.assertRaises(private_files.Refused):
            deliver.kept_on_host(self.pkg)
        self.assertEqual(
            mode(self.kept(self.env)), 0o644, "a reader changes no permissions"
        )

    def test_a_reader_checks_and_changes_nothing(self):
        self.deliver_plan(plan(RAMDISK))
        os.chmod(self.env, 0o644)
        with self.assertRaises(private_files.Refused):
            deliver.on_host(self.pkg)
        self.assertEqual(mode(self.env), 0o644, "a reader changes no permissions")
        os.chmod(self.env, 0o600)
        target = self.pkg.parent / "elsewhere"
        target.write_text("services: {}\n", encoding="utf-8")
        self.override.unlink()
        self.override.symlink_to(target)
        with self.assertRaises(private_files.Refused):
            deliver.on_host(self.pkg)


class TestWritten(Case):
    """The fingerprint of each file deliver writes, kept in this folder's records once it is written (state 9)."""

    def written(self) -> records.Written:
        return records.read(self.pkg).written

    def unchanged(self) -> None:
        """What is in place is what the fingerprints kept say."""
        found = (*deliver.on_host(self.pkg), self.layer())
        self.assertEqual(deliver.changed_outside(self.written(), *found), ())

    def test_each_delivery_keeps_them(self):
        fp = deliver.fingerprint
        first, second = plan(RAMDISK), plan()
        third = plan({"host.resources.memory_gib": 12})
        self.deliver_plan(first)
        layer = fp(layer_text(first))
        self.assertEqual(
            self.written(),
            records.Written(fp(first.env), fp(first.override), layer, ""),
            "the volume kept no layer before",
        )
        self.unchanged()
        self.deliver_plan(second)
        self.assertEqual(
            self.written(),
            records.Written(
                fp(second.env), deliver.NO_FILE, fp(layer_text(second)), layer
            ),
        )
        self.unchanged()
        self.deliver_plan(third, keep_previous=False)
        self.assertEqual(
            (self.written().layer, self.written().kept_layer),
            (fp(layer_text(third)), layer),
            "without keeping, the kept layer's stays",
        )
        self.unchanged()

    def test_a_put_back_keeps_what_it_put_back(self):
        fp = deliver.fingerprint
        first, second = plan(RAMDISK), plan(SERVED)
        self.deliver_plan(first)
        self.deliver_plan(second)
        self.assertTrue(self.put_back_kept())
        layer = fp(layer_text(first))
        self.assertEqual(
            self.written(),
            records.Written(fp(first.env), fp(first.override), layer, layer),
        )
        self.unchanged()

    def test_a_run_that_changes_nothing_keeps_none(self):
        self.deliver_plan(plan())
        before = self.written()
        for fake in (dict(fault=NO_SPACE), dict(code=125)):
            with self.subTest(**fake):
                with self.assertRaises(deliver.Failed):
                    self.deliver_plan(plan(RAMDISK), compose=self.fake(**fake))
                self.assertEqual(self.written(), before)

    def test_a_fingerprint_that_cannot_be_written(self):
        # The layer's first, then each file's once it is written: Failed says what changed, as for a file.
        self.deliver_plan(plan())
        real = records.change_written
        for calls, files in (
            (1, "docker/.env and the override are as they were"),
            (2, "docker/.env is the new one, and the override is as it was"),
            (3, "docker/.env and the override are the new ones"),
        ):
            done = []

            def change_written(package_dir, change):
                done.append(change)
                if len(done) == calls:
                    raise OSError(28, "No space left on device")
                return real(package_dir, change)

            with self.subTest(calls=calls):
                with mock.patch.object(records, "change_written", change_written):
                    with self.assertRaises(deliver.Failed) as caught:
                        self.deliver_plan(plan(RAMDISK))
                failed = caught.exception
                self.assertEqual(failed.code, 0)
                self.assertTrue(
                    str(failed).startswith(f"{records.FILE} could not be written ("),
                    str(failed),
                )
                self.assertEqual(
                    failed.changed,
                    "the settings volume has the new layer, the earlier one kept; "
                    + files
                    + KEPT_AS_THEY_WERE,
                )


class TestChangedOutside(unittest.TestCase):
    """State 9's rule: each file deliver writes against the fingerprint of what it last wrote there."""

    def test_what_differs(self):
        p = plan(RAMDISK)
        env, override, layer = deliver.texts(p)
        fp = deliver.fingerprint
        written = records.Written(fp(env), fp(override), fp(layer), "")
        e, o, n = deliver.WRITES
        for found, names in (
            ((env, override, layer), ()),
            ((env + "# mine\n", override, layer), (e,)),
            ((None, override, layer), (e,)),
            ((env, None, layer), (o,)),
            ((env, override + "\n", layer), (o,)),
            ((env, override, layer.replace("\n", "\r\n")), (n,)),
            ((None, None, None), deliver.WRITES),
        ):
            with self.subTest(names=names):
                self.assertEqual(deliver.changed_outside(written, *found), names)
        none = dataclasses.replace(written, override=deliver.NO_FILE)
        self.assertEqual(deliver.changed_outside(none, env, None, layer), ())
        self.assertEqual(
            deliver.changed_outside(none, env, "", layer),
            (o,),
            "an empty file is a file",
        )

    def test_a_file_with_no_fingerprint_kept_is_unchanged(self):
        # As an older package left the records: none kept, or none for a file.
        e, o, n = deliver.WRITES
        self.assertEqual(deliver.changed_outside(None, "a", "b", "c"), ())
        self.assertEqual(deliver.changed_outside(records.NOT_KEPT, "a", "b", "c"), ())
        written = records.Written("", deliver.NO_FILE, "", "")
        self.assertEqual(deliver.changed_outside(written, "a", "b", "c"), (o,))

    def test_the_fingerprint(self):
        self.assertEqual(deliver.fingerprint(None), deliver.NO_FILE)
        for text in ("", "RKMON_CPUS=4\n", "é\n"):
            with self.subTest(text=text):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.assertEqual(deliver.fingerprint(text), "sha256:" + digest)

    def test_wrote(self):
        fp = deliver.fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            pkg = pathlib.Path(tmp)
            (pkg / "docker").mkdir()
            with self.assertRaises(TypeError):
                deliver.wrote(pkg, docker_env="RKMON_CPUS=4\n")
            self.assertIsNone(records.read(pkg).written, "nothing kept")
            deliver.wrote(pkg, env="E\n", override=None, layer="L1\n")
            first = records.Written(fp("E\n"), deliver.NO_FILE, fp("L1\n"), "")
            self.assertEqual(records.read(pkg).written, first)
            deliver.wrote(pkg, layer="L2\n", keep=True)
            kept = dataclasses.replace(first, layer=fp("L2\n"), kept_layer=fp("L1\n"))
            self.assertEqual(records.read(pkg).written, kept)
            deliver.wrote(pkg, put_back=True)
            back = dataclasses.replace(kept, layer=fp("L1\n"))
            self.assertEqual(records.read(pkg).written, back)


class TestDiffering(unittest.TestCase):
    """What apply would change, the change plan's rule (rkmon_lib/change_plan.py): whole texts against a plan's, by
    WRITES' names."""

    def test_what_deliver_writes_for_a_plan(self):
        p = plan(RAMDISK)
        self.assertEqual(deliver.texts(p), (p.env, p.override, layer_text(p)))
        self.assertIsNone(deliver.texts(plan())[1], "no override: no file")

    def test_what_differs(self):
        p = plan(RAMDISK)
        env, override, layer = deliver.texts(p)
        e, o, n = deliver.WRITES
        for found, names in (
            ((env, override, layer), ()),
            ((env + "# mine\n", override, layer), (e,)),
            ((None, override, layer), (e,)),
            ((env, None, layer), (o,)),
            ((env, override + "\n", layer), (o,)),
            ((env, override, layer.replace("\n", "\r\n")), (n,)),
            ((None, None, None), deliver.WRITES),
        ):
            with self.subTest(names=names):
                self.assertEqual(deliver.differing(p, *found), names)
        q = plan()
        found = (q.env, "", layer_text(q))
        self.assertEqual(deliver.differing(q, *found), (o,), "an empty file is a file")


class TestSkills(Case):
    """The team's skills the confirmed settings name (skills_folder), read on this host and checked against
    the layer's record of them, go beside the layer in the settings volume (SKILLS_FOLDER), kept (SKILLS_KEPT) and put
    back with it."""

    def test_a_delivery_puts_them_beside_the_layer(self):
        p = plan(self.team(V1))
        self.deliver_plan(p)
        self.assertEqual(
            self.called(), ("deliver", (layer_text(p), HOST.gid, True, V1))
        )
        self.assertEqual(self.skills(), V1)
        self.assertIsNone(self.skills(deliver.SKILLS_KEPT), "none were there to keep")
        top = self.volume / deliver.SKILLS_FOLDER
        for path in (top, top / "team_members"):
            self.assertEqual(
                (mode(path), os.stat(path).st_gid), (0o750, HOST.gid), path
            )
        for rel in V1:
            self.assertEqual(
                (mode(top / rel), os.stat(top / rel).st_gid), (0o640, HOST.gid), rel
            )
        self.assertEqual(
            sorted(os.listdir(self.volume)), [deliver.LAYER_FILE, deliver.SKILLS_FOLDER]
        )

    def test_they_are_kept_and_put_back_with_the_layer(self):
        first = plan(self.team(V1))
        self.deliver_plan(first)
        self.deliver_plan(plan(self.team(V2)))
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V2, V1))
        self.assertTrue(self.put_back_kept())
        self.assertEqual(self.layer(), layer_text(first))
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V1, V1))
        self.assertEqual(mode(self.volume / deliver.SKILLS_FOLDER / VERIFIER), 0o640)

    def test_a_plan_without_them_takes_them_away_and_keeps_them(self):
        first = plan(self.team(V1))
        self.deliver_plan(first)
        self.deliver_plan(plan())
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (None, V1))
        self.assertTrue(self.put_back_kept())
        self.assertEqual((self.layer(), self.skills()), (layer_text(first), V1))

    def test_with_none_in_place_none_are_kept(self):
        self.deliver_plan(plan())
        self.deliver_plan(plan(self.team(V1)))
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V1, None))
        self.assertTrue(self.put_back_kept())
        self.assertEqual((self.layer(), self.skills()), (layer_text(plan()), None))
        self.assertEqual(
            sorted(os.listdir(self.volume)), [deliver.LAYER_FILE, deliver.LAYER_KEPT]
        )

    def test_without_keeping_the_kept_ones_stay(self):
        self.deliver_plan(plan(self.team(V1)))
        self.deliver_plan(plan(self.team(V2)))
        self.deliver_plan(plan(self.team(V3)), keep_previous=False)
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V3, V1))

    def test_files_changed_since_the_confirm_are_refused_with_nothing_changed(self):
        changes = self.team(V1)
        top = changes[deploy_plan.SKILLS_KEY]
        p = plan(changes)
        self.deliver_plan(plan())
        before = (self.everything(), len(self.compose.calls), records.read(self.pkg))
        more = {f"{c}.md": "# more\n" for c in "abcd"}
        for files, names in (
            ({**V1, VERIFIER: "# mine\n"}, VERIFIER),
            ({**V1, "agents/new.md": "# new\n"}, "agents/new.md"),
            ({"review_book.md": V1["review_book.md"]}, VERIFIER),
            ({**V1, **more}, "a.md, b.md, c.md and 1 more"),
        ):
            with self.subTest(names=names):
                self.team(files)
                with self.assertRaises(deliver.SkillsChanged) as caught:
                    self.deliver_plan(p)
                self.assertEqual(
                    (caught.exception.code, caught.exception.changed), (0, "")
                )
                self.assertEqual(
                    str(caught.exception),
                    f"the team's skills in {top} changed since your confirm ({names}): nothing changed",
                )
                self.assertEqual(
                    (
                        self.everything(),
                        len(self.compose.calls),
                        records.read(self.pkg),
                    ),
                    before,
                )

    def test_a_skill_that_is_no_utf8_text_is_refused(self):
        changes = self.team(V1)
        book = pathlib.Path(changes[deploy_plan.SKILLS_KEY]) / "review_book.md"
        book.write_bytes(b"# \xff\n")
        with self.assertRaises(deliver.Failed) as caught:
            self.deliver_plan(plan(changes))
        self.assertNotIsInstance(caught.exception, deliver.SkillsChanged)
        self.assertEqual(
            str(caught.exception),
            f"the team's skill {book} is no UTF-8 text: nothing changed",
        )
        self.assertEqual(self.compose.calls, [])

    def test_a_link_in_the_team_s_folder_is_never_followed(self):
        changes = self.team(V1)
        secret = self.tmp / "secret"
        secret.write_text("a token\n", encoding="utf-8")
        (pathlib.Path(changes[deploy_plan.SKILLS_KEY]) / "agents.md").symlink_to(secret)
        self.deliver_plan(plan(changes))
        self.assertEqual(self.skills(), V1)

    def test_the_program_writes_nothing_outside_the_folder(self):
        """The root program's own guard: a path that is not one inside the skills folder changes nothing."""
        self.deliver_plan(plan())
        before = self.tree()
        p = plan(RAMDISK)
        args = ["-p", PROJECT, *deliver.LAYER_RUN]
        for rel in ("../escape.md", "/escape.md", "team//escape.md", "./escape.md", ""):
            with self.subTest(rel=rel):
                call = f"deliver({layer_text(p)!r}, {HOST.gid}, True, {{{rel!r}: 'x'}})"
                self.assertEqual(
                    self.compose(args, deliver.layer_program(call), p.variables),
                    deliver.UNCHANGED,
                )
                self.assertIn(
                    "no path in the team's skills folder", self.compose.stderr
                )
                self.assertEqual(self.tree(), before)
        self.assertEqual(sorted(os.listdir(self.volumes)), [self.volume.name])


class TestAStopAnywhere(Case):
    """A run stopped just before any one of its program's steps, then the next run's recover(), leave the
    settings volume as it was or as the whole run leaves it, never a mix: the layers with their skills. Between the
    stop and the next run, a reader of the volume takes the layer in place with the skills that go with it
    (deliver.skills_with_layer).
    """

    def view(self) -> tuple:
        """What a reader of the settings volume takes: the layer, and the team's skills that go with it, each file's
        path in them -> its bytes (None for none)."""
        top = deliver.skills_with_layer(self.volume / deliver.SKILLS_FOLDER)
        files = None
        if top is not None:
            found = sorted(p for p in top.rglob("*") if p.is_file())
            files = {p.relative_to(top).as_posix(): p.read_bytes() for p in found}
        return (self.volume / deliver.LAYER_FILE).read_bytes(), files

    def sweep(self, run) -> None:
        """RUN(compose), a delivery or a put back from the volume as it is now, stopped in turn just before each of its
        program's steps that may change the volume: after each stop, the reader's view, then after recover() the
        volume, as they were until the commit, then as RUN leaves them."""
        saved = self.tmp / "saved"
        shutil.copytree(self.volume, saved, symlinks=True)
        before, seen_before = self.tree(), self.view()
        run(self.compose)
        after, seen_after = self.tree(), self.view()
        self.assertNotEqual(after, before)
        self.assertNotEqual(seen_after, seen_before)
        args, variables = ["-p", PROJECT, *deliver.LAYER_RUN], plan().variables
        found, seen = [], []
        for n in itertools.count(1):
            self.assertLess(n, 500, "a run of 500 steps")
            shutil.rmtree(self.volume)
            shutil.copytree(saved, self.volume, symlinks=True)
            try:
                run(self.fake(fault=STOP.format(n=n)))
            except deliver.Failed as exc:
                self.assertEqual(exc.code, KILLED, exc)
            else:
                break  # the run has fewer steps than N
            now = self.view()
            seen.append(
                "before"
                if now == seen_before
                else "after" if now == seen_after else now
            )
            self.assertEqual(
                self.fake()(args, deliver.layer_program("recover()"), variables), 0
            )
            now = self.tree()
            found.append(
                "before" if now == before else "after" if now == after else now
            )
        self.assertEqual(self.tree(), after)
        commit = found.index("after")
        self.assertGreater(commit, 0)
        self.assertEqual(found, ["before"] * commit + ["after"] * (len(found) - commit))
        self.assertEqual(seen, found)

    def test_a_delivery_of_skills_over_skills(self):
        self.deliver_plan(plan(self.team(V1)))
        self.deliver_plan(plan(self.team(V2)))
        p = plan(self.team(V3))
        self.sweep(lambda compose: self.deliver_plan(p, True, compose))
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V3, V2))

    def test_a_delivery_of_none_over_none(self):
        self.deliver_plan(plan(self.team(V1)))
        self.deliver_plan(plan())
        p = plan(RAMDISK)
        self.sweep(lambda compose: self.deliver_plan(p, True, compose))
        self.assertEqual(
            (self.skills(), self.skills(deliver.SKILLS_KEPT)), (None, None)
        )

    def test_a_put_back(self):
        self.deliver_plan(plan(self.team(V1)))
        self.deliver_plan(plan(self.team(V2)))
        self.sweep(lambda compose: self.put_back_kept(compose))
        self.assertEqual((self.skills(), self.skills(deliver.SKILLS_KEPT)), (V1, V1))


if __name__ == "__main__":
    unittest.main()
