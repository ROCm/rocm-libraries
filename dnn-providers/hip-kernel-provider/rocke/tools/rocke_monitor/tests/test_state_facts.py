# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The state table's facts (rkmon_lib/state_facts.py): a package folder whose confirmed settings deliver
put in place, a settings file in a home folder of the test's, and a docker that answers as each test says. Each of the
16 states from what makes it; what docker is asked, and when: nothing before a confirm, the owner alone while the
name is another deployment's or unknown or the records decide the state, nothing past the first state its answers
decide; the files changed outside the settings file, against the fingerprints of what deliver last wrote there,
while a change waits and after a put-back too, a file with no fingerprint kept being unchanged, and, while the
confirmed settings are the delivered ones, the same files as the change plan lists, a map's entries in another order
being no change; the change put back; a first tick the records keep, with or without a confirm; the facts of the
package in another folder, as an update asks for them; the package changed by its version, or by its code under the
same version.
"""

import dataclasses
import datetime
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import (  # noqa: E402
    change_plan,
    deliver,
    deploy_plan,
    file_types,
    identity,
    image,
    private_files,
    program,
    records,
    settings_file,
    state_facts,
    states,
)
from rkmon_lib.checks import Command, Edit, Steps  # noqa: E402
from test_deliver import HOST, RAMDISK, Compose  # noqa: E402
from test_deploy_plan import FAST, PACKAGE, TAG, doc  # noqa: E402
from test_package_update import schema  # noqa: E402

NAME = "rocke-monitor"
IMAGE = f"rocke-monitor:{TAG}"
LAYOUT = file_types.DEPLOY.version
CPUS = "host.resources.cpus"
# The .dockerignore of the tests' packages: their code is VERSION and rkmon_lib/ (image.code_sha256).
CODE_FILES = "*\n!VERSION\n!rkmon_lib\n"
STARTED = datetime.datetime(2026, 1, 2, 4, 0, tzinfo=datetime.timezone.utc)
NOW = STARTED + datetime.timedelta(minutes=10)
LATER = "2026-01-02T05:06:07Z"
FIX = Steps(
    (
        Edit("model_bindings.fast.base_url", "~/rkmon-deploy.yaml", "<url>"),
        "then",
        Command(("rkmon", "confirm")),
    )
)
# docker's questions, in the order gather asks them
ASKED = (
    "owner",
    "image_here",
    "logins_missing",
    "container",
    "set_up",
    "layer",
    "monitor",
)


class Docker:
    """docker for the tests: ANSWERS, by question; each question asked kept in `asked`. A question about anything
    but the confirmed container name or image fails the test."""

    def __init__(self, answers: dict):
        self.answers, self.asked = answers, []

    def _answer(self, question: str, about: str):
        expected = IMAGE if question in ("image_here", "logins_missing") else NAME
        if about != expected:
            raise AssertionError(f"{question} asked of {about}, not {expected}")
        self.asked.append(question)
        return self.answers[question]

    def owner(self, project):
        return self._answer("owner", project)

    def image_here(self, image):
        return self._answer("image_here", image)

    def logins_missing(self, image):
        return self._answer("logins_missing", image)

    def container(self, project):
        return self._answer("container", project)

    def set_up(self, project):
        return self._answer("set_up", project)

    def layer(self, project):
        return self._answer("layer", project)

    def monitor(self, project):
        return self._answer("monitor", project)


def put(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


class Case(unittest.TestCase):
    """A package folder whose confirmed settings deliver put in place, its container running them: state 16, until a
    test changes something."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        home = root / "home"
        home.mkdir()
        patch = mock.patch.dict(os.environ, {"HOME": str(home)})
        patch.start()
        self.addCleanup(patch.stop)
        self.pkg = root / "pkg"
        (self.pkg / "docker").mkdir(parents=True)
        (self.pkg / ".dockerignore").write_text(CODE_FILES, encoding="utf-8")
        self.version(PACKAGE)
        self.volumes = root / "volumes"
        self.volumes.mkdir()
        self.compose = Compose(self.volumes, self.pkg)
        self.volume = self.compose.volume(NAME)
        self.file = home / "rkmon-deploy.yaml"
        self.env = self.pkg / deliver.ENV_FILE
        self.override = self.pkg / deliver.OVERRIDE_FILE
        self.plan = self.confirm()
        self.deliver(self.plan)

    def version(self, version: str) -> None:
        """This package's version."""
        (self.pkg / "VERSION").write_text(version + "\n", encoding="utf-8")

    def settings(self, changes=None) -> settings_file.Document:
        """The settings file, the template with CHANGES as the writer writes it, read back."""
        put(self.file, settings_file.dump(doc(changes)))
        return settings_file.read(self.file, file_types.DEPLOY)

    def confirm(self, changes=None, **record) -> deploy_plan.DeployPlan:
        """The settings file with CHANGES, confirmed (RECORD: fields of the record in place of the plan's): what
        they make."""
        d = self.settings(changes)
        code = image.code_sha256(self.pkg)
        p = deploy_plan.plan(d, PACKAGE, code, HOST)
        container = p.layer["container"]
        values = dict(
            settings_file=str(self.file),
            fingerprint=p.fingerprint,
            at=HOST.confirmed_at,
            file_sha256=deploy_plan.file_sha256(d),
            package_version=PACKAGE,
            code_sha256=code,
            container_name=container["project"],
            image=records.Image(**container["image"]),
            changes=4,
        )
        records.confirm(self.pkg, records.Confirmed(**{**values, **record}))
        return p

    def deliver(self, p: deploy_plan.DeployPlan) -> None:
        deliver.deliver(p, package_dir=self.pkg, compose=self.compose)

    def delivered(self) -> str:
        """The layer in the settings volume."""
        return (self.volume / deliver.LAYER_FILE).read_text(encoding="utf-8")

    def docker(self, **answers) -> Docker:
        """docker answering ANSWERS, the rest as for a container that runs the delivered layer and has loaded it."""
        running = dict(
            owner=identity.OURS,
            image_here=True,
            logins_missing=(),
            container=(states.RUNNING, STARTED),
            set_up=True,
            layer=self.delivered(),
            monitor=state_facts.Monitor(self.plan.fingerprint, False, False),
        )
        return Docker({**running, **answers})

    def gather(self, **answers) -> tuple:
        """(the facts, the questions docker was asked), docker answering ANSWERS (`docker`)."""
        docker = self.docker(**answers)
        return state_facts.gather(self.pkg, docker, now=NOW), docker.asked

    def state(self, **answers) -> int:
        return states.current(self.gather(**answers)[0]).n

    def running(self) -> states.Facts:
        """The facts of the deployment as setUp leaves it."""
        return states.Facts(
            name_owner=identity.OURS,
            container_name=NAME,
            settings_file="~/rkmon-deploy.yaml",
            file_exists=True,
            under_way="",
            under_way_runs=False,
            first_apply=False,
            file_layout=LAYOUT,
            package_layout=LAYOUT,
            file_changed=False,
            package_changed=False,
            image=IMAGE,
            image_here=True,
            registry=False,
            logins_missing=(),
            container=states.RUNNING,
            running_minutes=10.0,
            not_set_up=False,
            put_back="",
            put_back_fix=None,
            confirmed_fingerprint=self.plan.fingerprint,
            confirmed_at="2026-01-02 03:04 UTC",
            waiting_changes=4,
            delivered_fingerprint=self.plan.fingerprint,
            loaded_fingerprint=self.plan.fingerprint,
            hand_edited=(),
            standby=False,
            paused=False,
            first_tick_recorded=False,
        )

    def older_records(self) -> None:
        """The records as an older package left them: no fingerprint kept."""
        path = self.pkg / records.FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["written"]
        put(path, json.dumps(data))

    def records_only(self, **changes) -> states.Facts:
        """The facts with docker asked nothing but the owner, with CHANGES."""
        return dataclasses.replace(
            self.running(),
            image_here=False,
            container=states.NO_CONTAINER,
            running_minutes=0.0,
            delivered_fingerprint="",
            loaded_fingerprint="",
            **changes,
        )


class TestNames(unittest.TestCase):
    def test_words_for_each_reason_a_change_is_put_back(self):
        self.assertEqual(set(state_facts.PUT_BACK), set(records.REASONS))


class TestRuns(Case):
    def test_a_deployment_that_runs_its_confirmed_settings(self):
        facts, asked = self.gather()
        self.assertEqual(facts, self.running())
        self.assertEqual(states.current(facts).n, 16)
        self.assertEqual(asked, list(ASKED))

    def test_how_long_it_has_run_is_from_now_by_default(self):
        started = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            minutes=3
        )
        docker = self.docker(container=(states.RUNNING, started))
        facts = state_facts.gather(self.pkg, docker)
        self.assertAlmostEqual(facts.running_minutes, 3.0, delta=0.5)


class TestRecords(Case):
    """States 2 to 5, which the records, the settings file and this package decide: docker is asked whose the name is,
    and nothing else."""

    def test_no_confirm(self):
        os.unlink(self.pkg / records.FILE)
        facts, asked = self.gather()
        self.assertEqual(states.current(facts).n, 2)
        self.assertEqual((facts.settings_file, facts.name_owner), ("", identity.NONE))
        self.assertEqual(asked, [], "nothing is asked before a confirm")

    def test_the_confirmed_file_is_gone(self):
        os.unlink(self.file)
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(file_exists=False))
        self.assertEqual(states.current(facts).n, 2)
        self.assertEqual(asked, ["owner"])

    def test_an_apply_that_stopped_part_way(self):
        records.begin(self.pkg, "apply").release()
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(under_way="apply"))
        self.assertEqual(states.current(facts).n, 3)
        self.assertEqual(asked, ["owner"])

    def test_an_apply_that_runs_is_no_stop(self):
        with records.begin(self.pkg, "rollback"):
            facts, _ = self.gather()
        self.assertEqual((facts.under_way, facts.under_way_runs), ("rollback", True))
        self.assertEqual(states.current(facts).n, 16)

    def test_a_file_in_another_layout(self):
        text = self.file.read_text(encoding="utf-8")
        for layout in (LAYOUT - 1, LAYOUT + 1):
            with self.subTest(layout=layout):
                put(
                    self.file,
                    text.replace(
                        f"schema_version: {LAYOUT}\n", f"schema_version: {layout}\n"
                    ),
                )
                facts, asked = self.gather()
                self.assertEqual(
                    facts, self.records_only(file_layout=layout, file_changed=True)
                )
                self.assertEqual(states.current(facts).n, 4)
                self.assertEqual(asked, ["owner"])

    def test_your_file_changed(self):
        self.settings({CPUS: 6})
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(file_changed=True))
        self.assertEqual(states.current(facts).n, 5)
        self.assertEqual(asked, ["owner"])

    def test_a_file_that_cannot_be_read_as_settings_has_changed(self):
        put(self.file, "kind: [\n")
        facts, _ = self.gather()
        self.assertEqual(facts, self.records_only(file_changed=True))
        self.assertEqual(states.current(facts).n, 5)

    @unittest.skipIf(os.geteuid() == 0, "root reads a file whatever its mode")
    def test_a_file_you_cannot_read_has_changed(self):
        os.chmod(self.file, 0)
        facts, _ = self.gather()
        self.assertEqual(facts, self.records_only(file_changed=True))

    def test_a_comment_is_no_change(self):
        with open(self.file, "a", encoding="utf-8") as f:
            f.write("# a note of yours\n")
        self.assertEqual(self.state(), 16)

    def test_this_package_changed(self):
        self.version("1.2.4")
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(package_changed=True))
        self.assertEqual(states.current(facts).n, 5)
        self.assertEqual(asked, ["owner"])

    def test_other_code_under_the_same_version(self):
        """A module changed, VERSION kept (a later commit of updates.branch): the package changed."""
        module = self.pkg / "rkmon_lib" / "tick.py"
        module.parent.mkdir()
        module.write_text("STEP = 1\n", encoding="utf-8")
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(package_changed=True))
        self.assertEqual((states.current(facts).n, asked), (5, ["owner"]))
        # a file the package's .dockerignore leaves out is no code: the package as confirmed
        module.unlink()
        (self.pkg / "docker" / "notes.txt").write_text("notes\n", encoding="utf-8")
        self.assertEqual(self.gather()[0], self.running())

    def test_a_first_tick_the_records_keep(self):
        # Deploy step 8 passed after a start; the records keep it, with or without a confirm, and docker says
        # nothing of it.
        records.first_tick(self.pkg, records.FirstTick(LATER))
        facts, asked = self.gather()
        self.assertEqual(
            facts, dataclasses.replace(self.running(), first_tick_recorded=True)
        )
        self.assertEqual(asked, list(ASKED))
        records.clear_first_tick(self.pkg)
        self.assertEqual(self.gather()[0], self.running())
        os.unlink(self.pkg / records.FILE)
        records.first_tick(self.pkg, records.FirstTick(LATER))
        facts, asked = self.gather()
        self.assertEqual(states.current(facts).n, 2)
        self.assertTrue(facts.first_tick_recorded)
        self.assertEqual(asked, [])

    def test_records_this_package_does_not_read(self):
        put(self.pkg / records.FILE, "{}")
        docker = self.docker()
        with self.assertRaises(records.Unreadable):
            state_facts.gather(self.pkg, docker, now=NOW)
        self.assertEqual(docker.asked, [])


class TestThePackage(Case):
    """The facts of the package in another folder, PACKAGE, with this folder's records, as an update asks for them:
    its VERSION and the layout its own deploy schema fixes, not this package's; nothing read when it fixes none.
    """

    def setUp(self):
        super().setUp()
        self.new = self.file.parent / "new"

    def package(self, version: str = PACKAGE, layout: int = LAYOUT) -> pathlib.Path:
        """The package folder ~/new: its VERSION VERSION, its deploy schema fixing LAYOUT, and the code of this
        package's files (CODE_FILES)."""
        schema(self.new, layout)
        (self.new / "VERSION").write_text(version + "\n", encoding="utf-8")
        (self.new / ".dockerignore").write_text(CODE_FILES, encoding="utf-8")
        return self.new

    def gather(self, **answers) -> tuple:
        """(the facts of the package in ~/new, the questions docker was asked), docker answering ANSWERS."""
        docker = self.docker(**answers)
        facts = state_facts.gather(self.pkg, docker, now=NOW, package=self.new)
        return facts, docker.asked

    def test_a_package_like_this_one(self):
        self.package()
        facts, asked = self.gather()
        self.assertEqual(facts, self.running())
        self.assertEqual(asked, list(ASKED))

    def test_its_version_not_this_package_s(self):
        self.version("1.2.4")
        self.package()
        self.assertEqual(self.gather()[0], self.running())
        self.package("1.2.4")
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(package_changed=True))
        self.assertEqual((states.current(facts).n, asked), (5, ["owner"]))

    def test_its_code_not_this_package_s(self):
        """The package an update brings, other code under the confirmed version: state 5, not docker's."""
        self.package()
        (self.new / "rkmon_lib").mkdir()
        (self.new / "rkmon_lib" / "tick.py").write_text("STEP = 2\n", encoding="utf-8")
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(package_changed=True))
        self.assertEqual((states.current(facts).n, asked), (5, ["owner"]))

    def test_the_layout_it_reads(self):
        for layout in (LAYOUT - 1, LAYOUT + 1):
            with self.subTest(layout=layout):
                self.package(layout=layout)
                facts, asked = self.gather()
                self.assertEqual(facts, self.records_only(package_layout=layout))
                self.assertEqual((states.current(facts).n, asked), (4, ["owner"]))

    def test_no_confirm(self):
        os.unlink(self.pkg / records.FILE)
        self.package(layout=LAYOUT + 1)
        facts, asked = self.gather()
        self.assertEqual(
            (states.current(facts).n, facts.package_layout, asked), (2, LAYOUT + 1, [])
        )

    def test_a_package_that_fixes_no_layout(self):
        self.new.mkdir()
        # records this package does not read, which the refusal comes before
        put(self.pkg / records.FILE, "{}")
        with self.assertRaises(state_facts.NoLayout) as caught:
            self.gather()
        self.assertEqual(
            str(caught.exception),
            "the package in ~/new says no layout of its deploy file",
        )


class TestOwner(Case):
    def test_another_deployments_name(self):
        facts, asked = self.gather(owner=identity.OTHER)
        self.assertEqual(facts, self.records_only(name_owner=identity.OTHER))
        self.assertEqual(states.current(facts).n, 1)
        self.assertEqual(asked, ["owner"])

    def test_a_name_docker_cannot_say_whose(self):
        facts, asked = self.gather(owner="unknown")
        self.assertEqual(facts, self.records_only(name_owner="unknown"))
        self.assertEqual(asked, ["owner"])

    def test_a_name_no_deployment_uses_yet_is_asked_on(self):
        facts, asked = self.gather(owner=identity.NONE)
        self.assertEqual(states.current(facts).n, 16)
        self.assertEqual(asked, list(ASKED))


class TestDocker(Case):
    """States 6 to 8 and 12, which docker's answers decide: nothing is asked past the first that decides."""

    def test_no_image(self):
        facts, asked = self.gather(image_here=False)
        self.assertEqual(states.current(facts).n, 6)
        self.assertEqual(asked, list(ASKED[:2]))

    def test_the_registry_is_the_records(self):
        self.confirm(image=records.Image("registry.example.com/team", NAME, TAG))
        facts, _ = self.gather(image_here=False)
        self.assertEqual((facts.registry, facts.image), (True, IMAGE))

    def test_logins_missing(self):
        facts, asked = self.gather(logins_missing=["claude", "admin_access_token"])
        self.assertEqual(facts.logins_missing, ("claude", "admin_access_token"))
        self.assertEqual(states.current(facts).n, 7)
        self.assertEqual(asked, list(ASKED[:3]))

    def test_never_started(self):
        facts, asked = self.gather(container=(states.NO_CONTAINER, None))
        self.assertEqual(states.current(facts).n, 8)
        self.assertEqual(asked, list(ASKED[:4]))

    def test_stopped(self):
        facts, asked = self.gather(container=(states.STOPPED, None))
        self.assertEqual(
            facts,
            dataclasses.replace(
                self.running(),
                container=states.STOPPED,
                running_minutes=0.0,
                loaded_fingerprint="",
            ),
        )
        self.assertEqual(states.current(facts).n, 12)
        self.assertEqual(
            asked,
            [q for q in ASKED if q not in ("set_up", "monitor")],
            "a stopped container is asked nothing",
        )

    def test_a_running_container_whose_work_volume_is_not_set_up(self):
        # It was started some other way than rkmon start, and waits for its first start: never started.
        facts, asked = self.gather(set_up=False)
        self.assertEqual(
            facts,
            dataclasses.replace(
                self.running(),
                not_set_up=True,
                delivered_fingerprint="",
                loaded_fingerprint="",
            ),
        )
        self.assertEqual(states.current(facts).n, 8)
        self.assertEqual(asked, list(ASKED[:5]))


class TestDelivered(Case):
    """States 10, 11 and 13 to 15: the change put back, the delivered layer and what the running monitor says."""

    def put_back(self, reason: str, fingerprint=None, fix=None, failure="") -> None:
        record = records.PutBack(
            fingerprint or self.plan.fingerprint, reason, LATER, failure, fix
        )
        records.put_back(self.pkg, record)

    def test_put_back_after_a_failed_doctor_check_names_it(self):
        self.put_back("doctor", fix=FIX, failure="ramdisk: not mounted")
        facts, asked = self.gather()
        self.assertEqual(
            (facts.put_back, facts.put_back_fix),
            (
                "the last change you confirmed was put back after a failed doctor check: ramdisk: not mounted",
                FIX,
            ),
        )
        self.assertEqual(states.current(facts).n, 10)
        self.assertEqual(asked, list(ASKED))

    def test_put_back_after_a_failed_load_or_restart_names_it(self):
        self.put_back("load", failure="the login gh is missing")
        facts, _ = self.gather()
        self.assertEqual(
            facts.put_back,
            "the last change you confirmed was put back after a failed load: the login gh is missing",
        )
        self.assertEqual(states.current(facts).n, 10)
        self.put_back("restart", failure="docker compose up exited 1")
        self.assertEqual(
            self.gather()[0].put_back,
            "the last change you confirmed was put back after a failed restart: docker compose up exited 1",
        )
        self.put_back("load")
        self.assertEqual(
            self.gather()[0].put_back,
            "the last change you confirmed was put back after a failed load",
        )

    def test_a_failed_first_apply_over_an_older_package_names_it(self):
        """The first apply over an older package that failed is state 10 too, though nothing was put back: its
        sentence says that it stopped the new container, then what failed, and its fix is the record's.
        """
        self.put_back(records.FIRST_APPLY, fix=FIX, failure="the login gh is missing")
        facts, _ = self.gather()
        self.assertEqual(
            (facts.put_back, facts.put_back_fix),
            (
                "the first apply over an older package failed and stopped the new container: the login gh is "
                "missing",
                FIX,
            ),
        )
        self.assertEqual(states.current(facts).n, 10)
        self.assertTrue(facts.first_apply)

    def test_a_first_apply_that_stopped_part_way_is_one(self):
        # The records mark the change under way as a first apply over an older package (records.first_apply)
        records.begin(self.pkg, "apply", (), first_apply=True).release()
        facts, asked = self.gather()
        self.assertEqual(facts, self.records_only(under_way="apply", first_apply=True))
        self.assertEqual(states.current(facts).n, 3)
        self.assertEqual(asked, ["owner"])

    def test_put_back_by_a_rollback_as_rkmon_was_run(self):
        self.put_back("rollback")
        by = "the last change you confirmed was put back by "
        self.assertEqual(self.gather()[0].put_back, by + "rkmon rollback")
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(self.gather()[0].put_back, by + "./rkmon rollback")
        self.put_back("rollback", failure="ramdisk: not mounted")
        self.assertEqual(
            self.gather()[0].put_back,
            by + "rkmon rollback, which ended with a failure: ramdisk: not mounted",
        )

    def test_a_change_put_back_before_the_last_confirm_is_no_more(self):
        self.put_back("doctor", fingerprint="sha256:" + "0" * 64, fix=FIX)
        facts, _ = self.gather()
        self.assertEqual((facts.put_back, facts.put_back_fix), ("", None))
        self.assertEqual(states.current(facts).n, 16)

    def test_a_change_waits(self):
        confirmed = self.confirm({CPUS: 6})
        facts, asked = self.gather()
        self.assertEqual(
            (facts.confirmed_fingerprint, facts.delivered_fingerprint),
            (confirmed.fingerprint, self.plan.fingerprint),
        )
        self.assertEqual(facts.hand_edited, (), "a change that waits is no hand edit")
        self.assertEqual(states.current(facts).n, 11)
        self.assertEqual(asked, list(ASKED))

    def test_nothing_delivered(self):
        # A layer deliver did not write is changed outside your file; with no fingerprint kept, it is the change that
        # waits.
        layers = (None, "not JSON", "[]", '{"fingerprint": 3}')
        for older, edited, n in ((False, (deliver.LAYER_NAME,), 9), (True, (), 11)):
            if older:
                self.older_records()
            for layer in layers:
                with self.subTest(layer=layer, older=older):
                    facts, _ = self.gather(layer=layer)
                    self.assertEqual(facts.delivered_fingerprint, "")
                    self.assertEqual(facts.hand_edited, edited)
                    self.assertEqual(states.current(facts).n, n)

    def test_not_loaded(self):
        facts, _ = self.gather(monitor=state_facts.Monitor("", False, False))
        self.assertEqual(facts.loaded_fingerprint, "")
        self.assertEqual(states.current(facts).n, 13)

    def test_not_loaded_yet(self):
        started = NOW - datetime.timedelta(minutes=1)
        answers = dict(
            container=(states.RUNNING, started),
            monitor=state_facts.Monitor("", False, False),
        )
        facts, _ = self.gather(**answers)
        self.assertEqual(facts.running_minutes, 1.0)
        self.assertEqual(
            states.current(facts).n, 16, "the monitor has two minutes to load it"
        )

    def test_standby(self):
        facts, _ = self.gather(
            monitor=state_facts.Monitor(self.plan.fingerprint, True, False)
        )
        self.assertEqual((facts.standby, states.current(facts).n), (True, 14))

    def test_paused(self):
        facts, _ = self.gather(
            monitor=state_facts.Monitor(self.plan.fingerprint, False, True)
        )
        self.assertEqual((facts.paused, states.current(facts).n), (True, 15))


class TestHandEdited(Case):
    """State 9: the files deliver writes, against the fingerprints of what it last wrote there, which the records
    keep: whatever the settings confirmed since, and after a put-back."""

    def edited_layer(self) -> str:
        layer = json.loads(self.delivered())
        layer["settings"]["host"]["resources"]["cpus"] = 6
        return deploy_plan.layer_text(layer)

    def test_docker_env_edited(self):
        put(self.env, self.plan.env.replace("RKMON_CPUS=4\n", "RKMON_CPUS=6\n"))
        facts, _ = self.gather()
        self.assertEqual(facts.hand_edited, ("docker/.env",))
        self.assertEqual(states.current(facts).n, 9)

    def test_docker_env_removed(self):
        os.unlink(self.env)
        self.assertEqual(self.gather()[0].hand_edited, ("docker/.env",))

    def test_an_override_added(self):
        put(self.override, "services: {}\n")
        self.assertEqual(
            self.gather()[0].hand_edited, ("docker/compose.override.yaml",)
        )

    def test_the_override_removed(self):
        self.plan = self.confirm(RAMDISK)
        self.deliver(self.plan)
        self.assertEqual(
            self.state(), 16, "the override deliver wrote is what it makes"
        )
        os.unlink(self.override)
        self.assertEqual(
            self.gather()[0].hand_edited, ("docker/compose.override.yaml",)
        )

    def test_the_layer_edited(self):
        facts, _ = self.gather(layer=self.edited_layer())
        self.assertEqual(facts.delivered_fingerprint, self.plan.fingerprint)
        self.assertEqual(facts.hand_edited, (deliver.LAYER_NAME,))
        self.assertEqual(states.current(facts).n, 9)

    def test_all_three(self):
        os.unlink(self.env)
        put(self.override, "services: {}\n")
        facts, _ = self.gather(layer=self.edited_layer())
        self.assertEqual(facts.hand_edited, deliver.WRITES)

    def test_while_a_change_waits(self):
        self.confirm({CPUS: 6})
        self.assertEqual(self.state(), 11)
        put(self.env, self.plan.env + "# a note of yours\n")
        facts, _ = self.gather()
        self.assertEqual(
            (facts.hand_edited, states.current(facts).n), (("docker/.env",), 9)
        )

    def test_after_a_put_back(self):
        # What the put-back wrote is what deliver last wrote: the change put back (state 10), until a hand edit.
        first = self.plan
        self.plan = self.confirm({CPUS: 6})
        self.deliver(self.plan)
        self.assertTrue(
            deliver.put_back(
                self.plan.variables, package_dir=self.pkg, compose=self.compose
            )
        )
        records.put_back(
            self.pkg,
            records.PutBack(self.plan.fingerprint, "rollback", LATER, "", None),
        )
        self.assertEqual(self.delivered(), deploy_plan.layer_text(first.layer))
        facts, _ = self.gather()
        self.assertEqual((facts.hand_edited, states.current(facts).n), ((), 10))
        put(self.env, first.env + "# a note of yours\n")
        facts, _ = self.gather(layer=self.edited_layer())
        self.assertEqual(
            (facts.hand_edited, states.current(facts).n),
            (("docker/.env", deliver.LAYER_NAME), 9),
        )

    def test_a_file_with_no_fingerprint_kept_is_unchanged(self):
        # As an older package left the records: no fingerprint kept for a file, or none at all.
        put(self.env, self.plan.env + "# a note of yours\n")
        put(self.override, "services: {}\n")
        records.change_written(self.pkg, lambda w: dataclasses.replace(w, env=""))
        self.assertEqual(self.gather()[0].hand_edited, (deliver.OVERRIDE_FILE,))
        self.older_records()
        facts, _ = self.gather(layer=self.edited_layer())
        self.assertEqual((facts.hand_edited, states.current(facts).n), ((), 16))

    def test_a_maps_entries_in_another_order(self):
        a, b = "model_bindings.a", "model_bindings.b"
        self.plan = self.confirm({a: FAST, b: FAST})
        self.deliver(self.plan)
        text = self.file.read_text(encoding="utf-8")
        self.settings({b: FAST, a: FAST})
        self.assertNotEqual(self.file.read_text(encoding="utf-8"), text)
        facts, _ = self.gather()
        self.assertEqual((facts.file_changed, facts.hand_edited), (False, ()))
        self.assertEqual(states.current(facts).n, 16)

    def test_a_reader_checks_the_files_and_changes_none(self):
        os.chmod(self.env, 0o644)
        with self.assertRaises(private_files.Refused):
            self.gather()
        self.assertEqual(os.stat(self.env).st_mode & 0o777, 0o644)


class TestTheChangePlan(Case):
    """State 9 and the change plan, while the confirmed settings are the delivered ones: the files state 9 names, by
    the fingerprints deliver kept, are those confirm's plan says apply puts back (deliver.differing).
    """

    def listed(self, layer=None) -> tuple:
        """The files state 9 names, after the test's edit, LAYER being the delivered layer's text (by default as
        deliver wrote it), the change plan listing the same."""
        found = self.delivered() if layer is None else layer
        facts, _ = self.gather(layer=found)
        applied = change_plan.Applied(found, *deliver.on_host(self.pkg))
        read = settings_file.read(self.file, file_types.DEPLOY)
        code = image.code_sha256(self.pkg)
        package = change_plan.Package(PACKAGE, file_types.DEPLOY, {}, code)
        p = change_plan.plan(
            applied, change_plan.Confirmed(read, HOST), package, package
        )
        self.assertEqual(states.current(facts).n, 9)
        self.assertEqual(tuple(c.key for c in p.outside), facts.hand_edited)
        self.assertEqual((p.yours, p.blank_defaults), ((), ()))
        return facts.hand_edited

    def test_a_comment(self):
        put(self.env, self.plan.env + "# a note of yours\n")
        self.assertEqual(self.listed(), (deliver.ENV_FILE,))

    def test_docker_env_removed(self):
        os.unlink(self.env)
        self.assertEqual(self.listed(), (deliver.ENV_FILE,))

    def test_an_override_added(self):
        put(self.override, "services: {}\n")
        self.assertEqual(self.listed(), (deliver.OVERRIDE_FILE,))

    def test_the_layer_edited(self):
        layer = json.loads(self.delivered())
        layer["container"]["address"] = "0.0.0.0"
        found = deploy_plan.layer_text(layer)
        self.assertEqual(self.listed(found), (deliver.LAYER_NAME,))

    def test_all_three(self):
        put(self.env, self.plan.env.replace("RKMON_CPUS=4\n", "RKMON_CPUS=6\n"))
        put(self.override, "services: {}\n")
        layer = json.loads(self.delivered())
        layer["container"]["address"] = "0.0.0.0"
        self.assertEqual(self.listed(deploy_plan.layer_text(layer)), deliver.WRITES)


if __name__ == "__main__":
    unittest.main()
