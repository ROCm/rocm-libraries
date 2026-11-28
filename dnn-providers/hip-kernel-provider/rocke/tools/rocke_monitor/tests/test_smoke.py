# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container tests' parts that run without a container (tests/container/): the released tarball's stage skips
outside a container and changes nothing there; the image's own stages are left to their own guards, and those on the
deployment they share (stages.py set_up: basics, misconfigured, deploy, reviewer, parallel-reviews,
parallel-reviews-local and stop) run only as root in the image, each a stage of the image's test; that deployment's
settings, their confirmation and its service's environment; the deploy-path stage's change of a setting, its stop and
its tries of a write to the settings volume, and the deploy path's scope and results under the name of the stage that
walks it, and how its stages read what a command says, wait for the container and run a command in it; how the adopt
stage makes the earlier monitor's copies, and how it and the one-time-move stage read a move and its end, and the latter
the test call of the claude login the move copied; how the several-deployments stage reads each deployment's docker
calls and folder reads; where the team-prototype stage asks for its page; the no-restart stage's keys, their edits and
what it reads of each; what the first-apply stage reads of a folder, of the volumes and of docker's times, and the
earlier image it needs; the fake models' server, which prints nothing for a client gone before its answer; the open PRs
the stand-in GitHub serves the stages, its write login, and what the deploy path reads of the stand-ins' log; and the
image test's own rules: the build's labels and its record, its limits, its arguments and its containers' host names.
"""

import contextlib
import copy
import http.client
import http.server
import io
import json
import math
import os
import pathlib
import pwd
import re
import runpy
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.request
import warnings
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTAINER = ROOT / "tests" / "container"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CONTAINER))
sys.path.insert(0, str(ROOT / "tests" / "fakes"))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
import fake_models  # noqa: E402
import fake_net  # noqa: E402
import fake_servers  # noqa: E402
import fakerepo  # noqa: E402
import image  # noqa: E402
import stage_adopt  # noqa: E402
import stage_deploy  # noqa: E402
import stage_deploy_path  # noqa: E402
import stage_first_apply  # noqa: E402
import stage_isolation  # noqa: E402
import stage_models  # noqa: E402
import stage_no_restart  # noqa: E402
import stage_one_time_move  # noqa: E402
import stage_several_deployments  # noqa: E402
import stage_stop  # noqa: E402
import stage_team_prototype  # noqa: E402
import stage_web_host  # noqa: E402
import stages  # noqa: E402
import test_build_plan as plans  # noqa: E402  (its build settings documents; its own tests run in their module)
from rkmon_lib import (  # noqa: E402
    adopt,
    build_plan,
    checks,
    coherence,
    container_layout,
    control,
    deploy_tree,
    doctor,
    file_types,
    gh,
    identity,
    one_time_move,
    registry,
    settings_file,
    steps,
)

# The released tarball's stage, as tarball.py runs it in a throwaway container; the image's own, as image.py does.
THROWAWAY = ["tarball"]
IMAGE_TABLES = [stages.STAGES] + [__import__(f).STAGES for f in stages.IMAGE_FILES]
IMAGE = sorted(name for table in IMAGE_TABLES for name in table)


class TestOutsideAContainer(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.home = pathlib.Path(d.name) / "home"
        self.called = []
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            mock.patch.object(stages, "tarball", self.recorder("tarball"))
        )
        for table in IMAGE_TABLES:
            stack.enter_context(
                mock.patch.dict(table, {k: self.recorder(k) for k in table})
            )

    def recorder(self, stage: str):
        def record(*_args):
            self.called.append(stage)
            return 0

        return record

    def argv(self, stage: str) -> list:
        tarball = ["--tarball", str(self.home.parent / "x.tar.gz")]
        return [stage, "--home", str(self.home)] + (
            tarball if stage == "tarball" else []
        )

    def main(self, stage: str, inside: bool) -> tuple:
        """`stages.py STAGE` with every stage a recorder, in a container or not: (its exit code, what it printed)."""
        out = io.StringIO()
        with mock.patch.object(doctor, "in_container", return_value=inside):
            with contextlib.redirect_stdout(out):
                return stages.main(self.argv(stage)), out.getvalue()

    def test_the_tarball_stage_skips_outside_a_container(self):
        for stage in THROWAWAY:
            with self.subTest(stage):
                rc, out = self.main(stage, inside=False)
                self.assertEqual(rc, 0, out)
                self.assertIn(f"SKIP  stage {stage} (not inside a container", out)
                self.assertTrue(out.endswith(f"STAGE {stage}: PASS\n"), out)
                self.assertEqual(self.called, [])
                self.assertFalse(self.home.exists())

    def test_inside_a_container_each_runs(self):
        for stage in THROWAWAY:
            with self.subTest(stage):
                self.called.clear()
                self.assertEqual(self.main(stage, inside=True)[0], 0)
                self.assertEqual(self.called, [stage])

    def test_the_image_s_own_stages_run_anywhere_by_their_own_guards(self):
        self.assertEqual(
            IMAGE,
            [
                "atlassian",
                "basics",
                "canary",
                "deploy",
                "isolation",
                "misconfigured",
                "parallel-reviews",
                "parallel-reviews-local",
                "reviewer",
                "rocm",
                "stop",
            ],
        )
        for stage in IMAGE:
            with self.subTest(stage):
                self.called.clear()
                self.assertEqual(self.main(stage, inside=False)[0], 0)
                self.assertEqual(self.called, [stage])


class TestTheImageStagesDeployment(unittest.TestCase):
    """The deployment the image's stages share (stages.py set_up), in its parts that run anywhere: a stage on it runs
    only as root in the image, and a set-up step that ends otherwise than in a deployment is the stage's FAIL row; the
    deploy settings are deploy.yaml's, the fake servers' ports moved, with this package's image and the container users'
    ids, and a stage's own values whole at their keys; rkmon confirm's reading takes them, from a private file, and
    refuses a value it does not take, on the tests' own host whatever machine runs them; the service's environment is
    compose's, each value filled in as compose fills it in."""

    def setUp(self):
        # the host rkmon confirm judges the container's share of (deployment.host_size): the tests' own
        held = mock.patch.dict(os.environ, deploy_folder.HOST_SIZE)
        held.start()
        self.addCleanup(held.stop)

    def run_stage(self, body, inside: bool, euid: int = 0) -> tuple:
        """image_stage("x") around BODY, in the image or not, as the user EUID: (its exit code, what it printed)."""
        out = io.StringIO()
        with mock.patch.object(stages, "in_the_image", return_value=inside):
            with mock.patch.object(stages.os, "geteuid", return_value=euid):
                with contextlib.redirect_stdout(out):
                    stage = stages.image_stage("x")(body)
                    rc = stage(pathlib.Path("/nonexistent"), stages.FIRST_PORT)
        return rc, out.getvalue()

    def test_a_stage_on_it_runs_only_as_root_in_the_image(self):
        ran = []

        def body(st, _home, port):
            ran.append(port)
            st.check(True, "a check")

        self.assertEqual(
            self.run_stage(body, inside=False),
            (
                0,
                "SKIP  stage x (not inside the image (docker/Dockerfile))\nSTAGE x: PASS\n",
            ),
        )
        rc, out = self.run_stage(body, inside=True, euid=1000)
        self.assertEqual(rc, 1)
        self.assertTrue(out.startswith("FAIL  stage x runs as root in the image"), out)
        self.assertEqual(ran, [])
        self.assertEqual(
            self.run_stage(body, inside=True), (0, "PASS  a check\nSTAGE x: PASS\n")
        )
        self.assertEqual(ran, [stages.FIRST_PORT])
        on_it = {
            **stages.STAGES,
            **stage_deploy.STAGES,
            **stage_models.STAGES,
            **stage_stop.STAGES,
        }
        for name, stage in on_it.items():
            with self.subTest(name):
                out = io.StringIO()
                with mock.patch.object(stages, "in_the_image", return_value=False):
                    with mock.patch.object(stages, "set_up") as set_up:
                        with contextlib.redirect_stdout(out):
                            rc = stage(pathlib.Path("/nonexistent"), stages.FIRST_PORT)
                self.assertEqual(rc, 0)
                self.assertIn(
                    f"SKIP  stage {name} (not inside the image", out.getvalue()
                )
                set_up.assert_not_called()

    def test_a_set_up_step_that_fails_is_the_stage_s_fail_row(self):
        def body(_st, _home, _port):
            raise stages.SetUpFailed(
                "rk provision, in a one-off container (exit 2)", "said"
            )

        self.assertEqual(
            self.run_stage(body, inside=True),
            (
                1,
                "FAIL  the deployment's set-up: rk provision, in a one-off container (exit 2)\n      said\n"
                "STAGE x: FAIL\n",
            ),
        )

    def test_the_image_by_its_mark_and_its_entry_point(self):
        for mark, entry, inside in (
            ("1", pathlib.Path(__file__), True),
            ("1", pathlib.Path("/nonexistent/entrypoint.sh"), False),
            ("", pathlib.Path(__file__), False),
        ):
            with self.subTest(mark=mark, entry=entry):
                with mock.patch.dict(os.environ, {"RKMON_CONTAINER": mark}):
                    with mock.patch.object(stages, "ENTRY", entry):
                        self.assertIs(stages.in_the_image(), inside)

    def test_nothing_written_into_the_image_s_code(self):
        """The row that nothing was written into the image's code (stages.code_unchanged) fails on a file rewritten in
        place, which leaves every name as it was, and on a file made; the code as it was passes it.
        """
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / ".dockerignore").write_text("*.pyc\n", encoding="utf-8")
            (root / "a.py").write_text("A = 1\n", encoding="utf-8")
            with mock.patch.object(stages.container_layout, "IMAGE_CODE", d):
                before = stages.code_state()

                def row() -> str:
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        stages.code_unchanged(stages.Stage("x"), before)
                    return out.getvalue().splitlines()[0]

                passed = f"PASS  nothing was written into the image's code ({d})"
                failed = passed.replace("PASS", "FAIL", 1)
                self.assertEqual(row(), passed)
                (root / "a.py").write_text("A = 2\n", encoding="utf-8")
                self.assertEqual(row(), failed)
                (root / "a.py").write_text("A = 1\n", encoding="utf-8")
                self.assertEqual(row(), passed)
                (root / "b.py").write_text("B = 1\n", encoding="utf-8")
                self.assertEqual(row(), failed)

    def test_the_deploy_settings(self):
        values = stages.deploy_values()
        url = values["model_bindings"]["worker"]["base_url"]
        self.assertEqual(url, f"http://127.0.0.1:{stages.FIRST_PORT + 2}/v1")
        self.assertEqual(values["image"], {"tag": stages.image.package_version()})
        monitor = container_layout.MONITOR
        self.assertEqual(
            values["host"],
            {"users": {"monitor_uid": monitor.uid, "monitor_gid": monitor.gid}},
        )
        moved = stages.deploy_values(19001)
        url = moved["model_bindings"]["worker"].pop("base_url")
        self.assertEqual(url, "http://127.0.0.1:19003/v1")
        values["model_bindings"]["worker"].pop("base_url")
        self.assertEqual(moved, values)
        first = stages.FIRST_PORT
        self.assertEqual(
            [
                stages._moved(p, 19001)
                for p in range(first - 1, first + stages.FAKES + 1)
            ],
            [first - 1, 19001, 19002, 19003, 19004, first + stages.FAKES],
        )
        binding = {
            "provider": "ollama",
            "model_id": "m:1",
            "base_url": "http://127.0.0.1:9/v1",
        }
        over = stages.deploy_values(
            over={
                "model_bindings.worker": binding,
                "features.alerts.develop_breaks": {"enabled": False},
                "integrations.a.b": 1,
            }
        )
        self.assertEqual(over["model_bindings"]["worker"], binding)
        alerts = over["features"]["alerts"]
        self.assertEqual(alerts["develop_breaks"], {"enabled": False})
        self.assertEqual(alerts["needs_you"], {"enabled": True})
        self.assertEqual(over["integrations"]["a"], {"b": 1})

    def test_rkmon_confirm_s_reading_takes_them_from_a_private_file(self):
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            dep = stages.Deployment(home, {})
            dep.confirm(stages.deploy_values())
            path = home / "rkmon-deploy.yaml"
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(
                dep.plan.variables["RKMON_TAG"], stages.image.package_version()
            )
            # the package's code, as rkmon confirm takes it
            self.assertEqual(dep.plan.layer["code_sha256"], stages.image.code_sha256())
            bad = stages.deploy_values(over={"features.merge_queue.mode": "x"})
            with self.assertRaises(stages.SetUpFailed) as refused:
                dep.confirm(bad)
        self.assertTrue(refused.exception.what.startswith("rkmon confirm "))
        self.assertIn("features.merge_queue.mode", refused.exception.said)

    def test_rkmon_confirm_s_host_is_the_tests_own(self):
        """The host rkmon confirm judges the container's share of is the tests' own (deploy_folder.HOST_SIZE): a
        machine too small for that share takes the settings all the same."""
        deployment = stages.deployment
        with mock.patch.object(
            deployment, "host_memory_gb", return_value=1.0
        ), mock.patch.object(deployment.os, "cpu_count", return_value=1):
            with tempfile.TemporaryDirectory() as d:
                dep = stages.Deployment(pathlib.Path(d), {})
                dep.confirm(stages.deploy_values())
        self.assertEqual(
            dep.plan.variables["RKMON_TAG"], stages.image.package_version()
        )

    def test_rkmon_confirm_s_rows_are_confirm_s_own(self):
        """The stage takes the rows rkmon confirm adds to the file's own (deployment.confirm_rows), against this
        package: the team's skills folder the settings name among them, whose FAIL is SetUpFailed.
        """
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            missing = home / "no-skills"
            values = stages.deploy_values(over={coherence.SKILLS_KEY: str(missing)})
            with self.assertRaises(stages.SetUpFailed) as refused:
                stages.Deployment(home, {}).confirm(values)
        said = " ".join(refused.exception.said.split())
        self.assertTrue(refused.exception.what.startswith("rkmon confirm "))
        shown = settings_file.shown(missing)
        self.assertIn(f"{coherence.SKILLS_KEY} {shown}: no such folder", said)

    def test_each_secret_is_set_as_its_keeper_with_the_layer_handed(self):
        """`rkmon secret NAME` as set_up runs it (Deployment.secret): rk secret-set NAME in a one-off container, as the
        writer for a secret the writer keeps (secrets_registry: the alerts' webhook), else as the monitor, by the
        wrapper's name; handed the layer the confirmed settings make, which goes with the one-off container; the value
        on its input."""
        with tempfile.TemporaryDirectory() as d:
            dep = stages.Deployment(pathlib.Path(d), {})
            dep.confirm(stages.deploy_values())
        layer = stages.deliver.texts(dep.plan)[2]
        writer = {"RKMON_AS": container_layout.WRITER.name}
        monitor = {"RKMON_WRAPPER": stages.WRAPPER}
        for name, who in (
            (stages.WEBHOOK, writer),
            (stages.ACCESS_TOKEN, monitor),
            ("first_review_pass_api_key", monitor),
        ):
            with self.subTest(name=name):
                with mock.patch.dict(
                    stages.os.environ, {"PATH": "/usr/bin"}, clear=True
                ), mock.patch.object(dep, "step") as step, mock.patch.object(
                    stages.shutil, "rmtree"
                ) as rmtree:
                    dep.secret(name, f"v-{name}")
                _what, _words, argv, env = step.call_args.args
                self.assertEqual(
                    argv, ["sh", str(stages.ENTRY), "rk", "secret-set", name]
                )
                self.assertEqual(step.call_args.kwargs, {"input": f"v-{name}\n"})
                keeper = {k: env.get(k) for k in [*writer, *monitor]}
                self.assertEqual(keeper, {**dict.fromkeys(keeper), **who})
                self.assertEqual(env[container_layout.HANDED_LAYER_VARIABLE], layer)
                rmtree.assert_called_once_with(
                    container_layout.HANDED_LAYER_DIR, ignore_errors=True
                )
                self.assertEqual(dep.secrets[name], f"v-{name}")

    def test_the_delivery_keeps_what_it_replaces_as_deliver_decides(self):
        """set_up's delivery (Deployment.deliver) hands deliver the decision of the rule's one home,
        deliver.keep_previous, made with the layer in place read from the settings folder.
        """
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            dep = stages.Deployment(home, {})
            dep.confirm(stages.deploy_values())
            settings = home / "settings"
            settings.mkdir()
            (settings / stages.deliver.LAYER_FILE).write_text("in place", "utf-8")
            decided, asked = object(), []

            def keep_previous(pkg, plan, read):
                asked.append((pkg, plan, read()))
                return decided

            with mock.patch.object(
                stages.container_layout, "SETTINGS_DIR", str(settings)
            ), mock.patch.object(
                stages.deliver, "keep_previous", keep_previous
            ), mock.patch.object(
                stages.deliver, "deliver"
            ) as delivered:
                dep.deliver()
        self.assertEqual(asked, [(home / "package", dep.plan, ("in place", ""))])
        delivered.assert_called_once_with(
            dep.plan, decided, package_dir=home / "package", compose=dep.compose
        )

    def test_the_service_s_environment_is_compose_s(self):
        with tempfile.TemporaryDirectory() as d:
            dep = stages.Deployment(pathlib.Path(d), {})
            dep.confirm(stages.deploy_values())
        variables = dict(dep.plan.variables)
        variables.pop("RKMON_TEAM_HOST", None)
        env = stages.service_environment(variables)
        for name in ("RKMON_PORT", "RKMON_ADMIN_HOST", "RKMON_UID", "RKMON_GID"):
            self.assertEqual(env[name], str(variables[name]), name)
        self.assertEqual(env["RKMON_TEAM_HOST"], "127.0.0.1")
        self.assertNotIn("RKMON_IMAGE", env)
        del variables["RKMON_PORT"]
        with self.assertRaisesRegex(ValueError, r"^RKMON_PORT is not set: see "):
            stages.service_environment(variables)

    def test_a_value_compose_fills_in(self):
        given = {"A": "1", "E": ""}
        for value, filled in (
            ("${A}", "1"),
            ("${B}", ""),
            ("${A:-x}", "1"),
            ("${B:-127.0.0.1}", "127.0.0.1"),
            ("${E:-x}", "x"),
            ("${A:?why}", "1"),
            ("plain", "plain"),
        ):
            with self.subTest(value=value):
                self.assertEqual(stages._filled(value, given), filled)
        for value, said in (
            ("${E:?why}", "E is not set: why"),
            ("${B:?why}", "B is not set: why"),
            ('"${A}"', '"${A}": not a form this reader fills in as compose does'),
            ("${A-x}", "${A-x}: not a form this reader fills in as compose does"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as refused:
                    stages._filled(value, given)
                self.assertEqual(str(refused.exception), said)


class TestTheImageTest(unittest.TestCase):
    """image.py's own rules: each of the build's labels is there and is one of the package's prefixes, the history
    holds no download location, the build record is the build's, the hip stage's GPU targets are the build plan's,
    every container runs with the compose file's limits (one capability left out on request) and a host name its hosts
    file gives an address, and its names, its capabilities and the build settings file it builds from are checked
    before anything runs. Its stages are the package's list of them (steps.IMAGE_TEST_STAGES): it runs them in that
    order, its --help lists them with what each checks, and it refuses a stage the list does not name before anything
    runs.
    """

    def main(self, *args) -> tuple:
        """image.py ARGS: its exit code and what it printed."""
        out = io.StringIO()
        with mock.patch.object(image.hostrun, "setup_output"):
            with contextlib.redirect_stdout(out):
                return image.main(list(args)), out.getvalue()

    def test_it_runs_the_package_s_stages_in_their_order(self):
        source = pathlib.Path(image.__file__).read_text(encoding="utf-8")
        blocks = re.findall(r'^    if "([a-z-]+)" in stages:$', source, re.M)
        self.assertEqual(blocks, list(steps.IMAGE_TEST_STAGES))
        self.assertIs(image.stage_list(str(ROOT)), steps.IMAGE_TEST_STAGES)

    def test_its_help_lists_each_stage_and_what_it_checks(self):
        rc, out = self.main("--help")
        self.assertEqual(rc, 0)
        self.assertIn("# Usage: tests/container/image.py ", out)
        lines = out.splitlines()
        # where each stage's line is: the help's last lines, in the list's order
        found = [
            next(
                (
                    i
                    for i, line in enumerate(lines)
                    if re.fullmatch(rf"#\s+{name}\s+{re.escape(what)}", line)
                ),
                None,
            )
            for name, what in steps.IMAGE_TEST_STAGES.items()
        ]
        last = range(len(lines) - len(found), len(lines))
        self.assertEqual(found, list(last), out)

    def test_the_stages_on_the_shared_deployment_are_its_own(self):
        """Each stage on the deployment the image's stages share (stages.py set_up: basics, misconfigured, deploy,
        reviewer, parallel-reviews, parallel-reviews-local and stop) is a stage of the image's test, run as root in a
        throwaway container, in a home of its own under /tmp, its row PASS when the stage passed, with what the
        package's list says it checks.
        """
        on_it = {
            **stages.STAGES,
            **stage_deploy.STAGES,
            **stage_models.STAGES,
            **stage_stop.STAGES,
        }
        self.assertEqual(
            sorted(on_it),
            [
                "basics",
                "deploy",
                "misconfigured",
                "parallel-reviews",
                "parallel-reviews-local",
                "reviewer",
                "stop",
            ],
        )
        for name in on_it:
            for rc, status in ((0, "PASS"), (1, "FAIL")):
                with self.subTest(stage=name, rc=rc):
                    res, out = image.hostrun.Results(), io.StringIO()
                    with mock.patch.object(
                        image.hostrun, "capture", return_value=(rc, "said")
                    ) as capture, contextlib.redirect_stdout(out):
                        image.deployment_stage(
                            name,
                            steps.IMAGE_TEST_STAGES[name],
                            lambda stage: ["docker", "run", "--name", f"x-{stage}"],
                            ["--cpus", "4"],
                            "img:t",
                            res,
                        )
                    command = (
                        "\n    python3 -B /opt/rocke_monitor/tests/container/stages.py "
                        f"{name} --home /tmp/{name}-stage"
                    )
                    capture.assert_called_once_with(
                        ["docker", "run", "--name", f"x-{name}", "-e", "RKMON_AS=root"]
                        + ["--cpus", "4", "img:t", "shell", "-c", command]
                    )
                    self.assertEqual(
                        out.getvalue().splitlines()[-1],
                        f"{status}  {name}: {steps.IMAGE_TEST_STAGES[name]}",
                    )
                    self.assertEqual(res.failed, rc)

    def test_a_stage_it_does_not_list_is_refused_before_anything_runs(self):
        with mock.patch.object(image.hostrun, "pick_engine") as engine:
            rc, out = self.main(
                "--stages", "labels,nope", "--no-build", "--image", "x:t"
            )
        self.assertEqual(rc, 2)
        self.assertEqual(
            out,
            f"unknown stage: nope (the stages: {','.join(steps.IMAGE_TEST_STAGES)})\n",
        )
        engine.assert_not_called()

    def test_a_label_of_another_prefix(self):
        names = build_plan.label_names()
        labels = {k: "x" for k in names}
        self.assertEqual(image.labels_findings(labels, "", (), names), [])
        labels["com.example.vendor"] = "x"
        self.assertEqual(
            image.labels_findings(labels, "", (), names),
            [
                "label com.example.vendor is neither org.opencontainers.* nor org.rocke.*"
            ],
        )

    def test_each_label_of_the_build_and_the_history(self):
        """Each label the build gives is there, blank or not; no download location's value is in the history, where a
        step reads each from its build secret."""
        names = build_plan.label_names()
        blank = {k: "" for k in names[1:]}
        self.assertEqual(
            image.labels_findings(blank, "", (), names), [f"no label {names[0]}"]
        )
        variables = image.locations(str(ROOT))
        self.assertEqual(
            variables,
            tuple(build_plan.secret_variable(d.id) for d in build_plan.DOWNLOADS),
        )
        read = "RUN |1 X=1 /bin/sh -c GITHUB_CLI_PACKAGE_URL=$(sh /opt/rkmon/build/param.sh github_cli_package_url)"
        given = (
            "RUN |1 ROCM_PACKAGE_URL=https://files.example.com/x.run /bin/sh -c install"
        )
        labels = {k: "x" for k in names}
        self.assertEqual(image.labels_findings(labels, read, variables, names), [])
        self.assertEqual(
            image.labels_findings(labels, given, variables, names),
            ["the history has a value of ROCM_PACKAGE_URL"],
        )

    def test_the_build_record(self):
        """The image's build record is the build's (build_plan.record), its fingerprint the label's; none, no JSON,
        another kind, another fingerprint, a registry or an address is a finding."""
        p = plans.make_plan()
        record = build_plan.record(p)
        self.assertEqual(
            image.record_findings(json.dumps(record), p.labels, build_plan), []
        )
        self.assertEqual(
            image.record_findings(None, p.labels, build_plan),
            [f"no build record at {build_plan.RECORD_PATH}"],
        )
        self.assertEqual(
            image.record_findings("x", p.labels, build_plan),
            ["the build record is no JSON"],
        )
        self.assertEqual(
            image.record_findings(
                json.dumps(dict(record, kind="other")), p.labels, build_plan
            ),
            [
                f"the build record is no {build_plan.RECORD_KIND} of layout {build_plan.RECORD_SCHEMA_VERSION}"
            ],
        )
        other = dict(p.labels, **{build_plan.FINGERPRINT_LABEL: "sha256:" + "0" * 64})
        changed = dict(
            record,
            image=dict(record["image"], registry="registry.example.com/team"),
            base="https://registry.example.com/ubuntu:26.04",
        )
        self.assertEqual(
            image.record_findings(json.dumps(changed), other, build_plan),
            [
                f"the build record's fingerprint is not the label {build_plan.FINGERPRINT_LABEL}'s",
                "the build record names a registry",
                "the build record has an address",
            ],
        )

    def test_the_hip_stage_s_targets_are_the_build_plan_s(self):
        """The hip stage passes when rk doctor's check passed in the image for build_plan.HIP_CHECK_TARGETS, and its
        line names them; other targets, a check that did not pass, or no ROCm fail."""
        targets = build_plan.HIP_CHECK_TARGETS
        line = f"hip: the image's hipcc compiles <hip/hip_runtime.h> for {', '.join(targets)} (rk doctor's check)"
        printed = f"PASS found fix\ntargets {' '.join(targets)}\nhip=0"
        self.assertEqual(image.hip_result(printed, build_plan), ("PASS", line))
        for out in (
            printed.replace("hip=0", "hip=1"),
            printed.replace("\ntargets ", "\ntargets gfx1100 "),
        ):
            with self.subTest(out=out):
                self.assertEqual(image.hip_result(out, build_plan), ("FAIL", line))
        self.assertEqual(
            image.hip_result("hip=none", build_plan),
            ("FAIL", line + ": the image has no ROCm"),
        )
        with mock.patch.object(build_plan, "HIP_CHECK_TARGETS", ("gfx1100", "gfx1201")):
            status, what = image.hip_result(printed, build_plan)
            self.assertEqual(status, "FAIL")
            self.assertIn("for gfx1100, gfx1201 (", what)
            other = "PASS found fix\ntargets gfx1100 gfx1201\nhip=0"
            self.assertEqual(image.hip_result(other, build_plan)[0], "PASS")

    def test_the_hip_stage_prints_a_fix_only_when_the_check_has_one(self):
        """The hip stage's program (HIP, run in the image) prints rk doctor's check: its status and what it found, then
        its fix only when it has one."""
        program = image.HIP.split('python3 -B -c "', 1)[1].rsplit('" "$record"', 1)[0]
        fix = checks.Steps((checks.Command(("rk", "doctor")),))
        with tempfile.TemporaryDirectory() as d:
            record = pathlib.Path(d) / "rocm.json"
            record.write_text(json.dumps({"llvm_bin": d}), encoding="utf-8")
            for check, first in (
                (checks.Check("hipcc", "PASS", "it compiles"), "PASS it compiles"),
                (
                    checks.Check("hipcc", "FAIL", "no headers", fix),
                    f"FAIL no headers {fix}",
                ),
            ):
                out = io.StringIO()
                patched = (
                    mock.patch.object(doctor, "hip_check", return_value=check),
                    mock.patch.object(sys, "argv", ["-c", str(record)]),
                    contextlib.redirect_stdout(out),
                    warnings.catch_warnings(),
                )
                with self.subTest(status=check.status), contextlib.ExitStack() as s:
                    for p in patched:
                        s.enter_context(p)
                    # the program, a one-off in the image, leaves its record's file for its exit to close
                    warnings.simplefilter("ignore", ResourceWarning)
                    with self.assertRaises(SystemExit):
                        exec(program, {"__name__": "__main__"})
                self.assertEqual(out.getvalue().splitlines()[0], first)

    def test_a_build_is_of_the_test_s_own_tag(self):
        """--settings FILE --tag TAG: rkmon build takes the tag from the file alone, so the test builds from a private
        copy of FILE whose image.tag is TAG (the image it removes is its own), in a folder gone after the build.
        """
        built = []

        def quiet(argv):
            copied = pathlib.Path(argv[argv.index("--settings") + 1])
            mode = copied.stat().st_mode & 0o777
            built.append((argv, mode, json.loads(copied.read_text(encoding="utf-8"))))
            return 1  # the build fails: nothing to test

        with tempfile.TemporaryDirectory() as d:
            mine = pathlib.Path(d, "build.json")
            header = {
                "kind": file_types.BUILD.kind,
                "schema_version": file_types.BUILD.version,
            }
            values = plans.build_doc(image={"name": "rkmon-x", "tag": "t1"}).values
            mine.write_text(json.dumps({**header, **values}), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.dict(os.environ), mock.patch.object(
                image.hostrun, "setup_output"
            ):
                with mock.patch.object(
                    image.hostrun, "pick_engine", return_value="docker"
                ):
                    with mock.patch.object(image.hostrun, "quiet", side_effect=quiet):
                        with contextlib.redirect_stdout(out):
                            rc = image.main(["--settings", str(mine), "--tag", "t9"])
        self.assertEqual(rc, 1, out.getvalue())
        self.assertIn("== building rkmon-x:t9 ", out.getvalue())
        [(argv, mode, data)] = built
        self.assertEqual(argv[-3:-1], ["build", "--settings"])
        self.assertEqual(mode, 0o600)
        self.assertEqual(data["image"], {"name": "rkmon-x", "tag": "t9"})
        self.assertFalse(os.path.exists(argv[-1]))

    def test_the_image_a_build_settings_file_names(self):
        """--settings FILE: the image name the file's plan gives; a file this package does not build (layout 1) gives
        none, and the test refuses it before anything runs."""
        with tempfile.TemporaryDirectory() as d:
            mine = pathlib.Path(d, "build.json")
            header = {
                "kind": file_types.BUILD.kind,
                "schema_version": file_types.BUILD.version,
            }
            values = plans.build_doc(image={"name": "rkmon-x"}).values
            mine.write_text(json.dumps({**header, **values}), encoding="utf-8")
            self.assertEqual(image.built_name(str(ROOT), str(mine)), "rkmon-x")
            older = pathlib.Path(d, "older.json")
            older.write_text(
                json.dumps(dict(header, schema_version=1)), encoding="utf-8"
            )
            self.assertEqual(image.built_name(str(ROOT), str(older)), "")
            out = io.StringIO()
            with mock.patch.object(image.hostrun, "setup_output"):
                with contextlib.redirect_stdout(out):
                    rc = image.main(["--settings", str(older), "--engine", "no-engine"])
        self.assertEqual(rc, 2)
        self.assertIn("no build settings file this package reads", out.getvalue())

    def test_the_limits_without_one_capability(self):
        flags = list(container_layout.RUN_LIMITS)
        self.assertEqual(image.limits(str(ROOT)), flags)
        for cap in container_layout.CAPABILITIES:
            with self.subTest(cap=cap):
                less = image.limits(str(ROOT), cap)
                self.assertEqual(len(less), len(flags) - 2)
                self.assertNotIn(cap, less)
                self.assertIn("--pids-limit", less)

    def test_a_container_s_host_name_has_an_address(self):
        """A stage's throwaway container has no network, so docker gives its host name no address, and sudo would say
        on each call that it cannot resolve the host: each container gets a host name of docker's form (12 hex
        digits), another for each, which its hosts file gives the loopback address."""
        words = image.run_words("docker", str(ROOT), "", "x-stage-1")
        self.assertEqual(words[:5], ["docker", "run", "--rm", "--network", "none"])
        self.assertEqual(words[words.index("--name") + 1], "x-stage-1")
        host = words[words.index("--hostname") + 1]
        self.assertRegex(host, r"\A[0-9a-f]{12}\Z")
        self.assertEqual(
            words[words.index("--add-host") + 1],
            f"{host}:{container_layout.LOOPBACK}",
        )
        other = image.run_words("docker", str(ROOT), "", "x-stage-2")
        self.assertNotEqual(other[other.index("--hostname") + 1], host)

    def test_names_and_capabilities_checked_first(self):
        for args, said in (
            (["--names", "two words"], "--names"),
            (["--without-cap", "NET_ADMIN"], "--without-cap NET_ADMIN"),
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with mock.patch.object(image.hostrun, "setup_output"):
                    with contextlib.redirect_stdout(out):
                        rc = image.main(args + ["--no-build", "--image", "x:t"])
                self.assertEqual(rc, 2)
                self.assertIn(said, out.getvalue())


class TestTheDeployPathsChange(unittest.TestCase):
    """The setting the deploy-path stage changes once the path is done (stage_deploy_path.py CHANGE), edited as a
    person edits it (edited): one line of the template the stage writes, which then sets that key alone otherwise.
    """

    def test_the_change_is_one_line_of_the_template_and_changes_its_key_alone(self):
        name, value = stage_deploy_path.CHANGE
        text = settings_file.template(file_types.DEPLOY, stage_deploy_path.PROFILE)
        new = stage_deploy_path.edited(text, name, value)
        old_lines, new_lines = text.splitlines(), new.splitlines()
        self.assertEqual(len(new_lines), len(old_lines))
        self.assertEqual(sum(a != b for a, b in zip(old_lines, new_lines)), 1)
        before = settings_file.loads(text, file_types.DEPLOY).values
        after = settings_file.loads(new, file_types.DEPLOY).values
        self.assertNotEqual(registry.get(before, name), value)
        self.assertEqual(registry.get(after, name), value)
        *parents, key = name.split(".")
        section = before
        for p in parents:
            section = section[p]
        section[key] = value
        self.assertEqual(after, before)

    def test_the_key_s_line_is_found_under_its_sections(self):
        text = "monitor:\n  every: 10\nother:\n  every: 5\n"
        self.assertEqual(
            stage_deploy_path.edited(text, "monitor.every", 15),
            "monitor:\n  every: 15\nother:\n  every: 5\n",
        )
        self.assertEqual(
            stage_deploy_path.edited(text, "other.every", "x"),
            'monitor:\n  every: 10\nother:\n  every: "x"\n',
        )

    def test_a_key_no_line_sets_or_two_lines_set_is_refused(self):
        for text, name in (
            ("monitor:\n  every: 10\n", "monitor.each"),
            ("monitor:\n  every: 10\n", "every"),
            ("monitor:\n  every: 10\n  every: 5\n", "monitor.every"),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                stage_deploy_path.edited(text, name, 15)


class TestTheDeployPathsStop(unittest.TestCase):
    """How the deploy-path stage reads `rkmon stop --drain` (stage_deploy_path.py stopped_drained): rk drain's own
    first line, the drain's done line, the outcome line and a next line that gives `rkmon start` make a stop with a
    drain; any other end does not.
    """

    DRAIN = "  drain: nothing new starts; nothing is in flight\n"

    def test_the_drain_done_then_the_stop_and_its_next_line(self):
        for done, said in (
            ("  drain: done\n", "next: ./rkmon start"),
            ("  drain: done after 2 minutes\n", "next: step 6, start: ./rkmon start"),
        ):
            with self.subTest(done=done, said=said):
                out = (
                    self.DRAIN
                    + done
                    + " Container x-rkmon-1 Stopped\nSTOPPED: the monitor's container\n"
                    + said
                )
                self.assertTrue(stage_deploy_path.stopped_drained(out))

    def test_no_drain_no_stop_or_another_next_line(self):
        for out in (
            # the work in flight did not finish in time: nothing stopped
            self.DRAIN
            + "  DRAIN TIMEOUT after 5 minutes: review #5 still in flight; the drain is lifted\n"
            "ERROR: the work in flight did not finish within 5 minutes; the drain is lifted; nothing changed\n"
            "next: ./rkmon stop --drain --timeout-minutes 10, or ./rkmon stop --drain later",
            # a stop with no drain, and a container that does not run
            "STOPPED: the monitor's container\nnext: ./rkmon start",
            "NOTHING TO DO: the container is not running\nnext: ./rkmon start",
            # the drain and the stop, then a next line that gives another command
            self.DRAIN
            + "  drain: done\nSTOPPED: the monitor's container\nnext: ./rkmon doctor",
            # the drain, then no outcome line
            self.DRAIN + "  drain: done\nnext: ./rkmon start",
            # the wrapper's done line with no line of rk drain's: no sign that its drain ran
            "  drain: done\nSTOPPED: the monitor's container\nnext: ./rkmon start",
        ):
            with self.subTest(out=out):
                self.assertFalse(stage_deploy_path.stopped_drained(out))


class TestTheNoRestartStage(unittest.TestCase):
    """How the no-restart stage (stage_no_restart.py) changes its keys and reads what their readers show: each key's
    two values fit its schema, and its edit, on the template the deploy path writes with the stage's values filled in,
    changes that key alone; what each reader shows is read from what the stage saw (a Seen) and from the commands'
    output as rk words it, wrapped or not; and confirm's line under a change says how it applies.
    """

    PUSHED = "2026-01-05T10:00:00Z"
    # A Seen after the change: the last full tick's record, the PRs' records, and what the commands showed
    AFTER = {
        "idle": {
            "fp": {"prs": "x"},
            "full_at": "2026-01-05T10:02:00Z",
            "why": [f"{stage_no_restart.FULL_TICK_MAX} is 0"],
            "due": {
                "settle #2": "2026-01-05T18:20:00Z",
                "gap #3": "2026-01-05T18:20:00Z",
                "weekly report": "2026-01-05T09:00:00Z",
            },
        },
        "prs": {
            "2": {
                "head_seen_at": "2026-01-05T10:01:00Z",
                "head_committed_at": PUSHED,
                "ours": None,
            },
            "3": {
                "head_seen_at": "2026-01-05T10:01:00Z",
                "head_committed_at": "2002-02-02T00:00:00Z",
                "ours": {"at": PUSHED},
            },
        },
        "spend": {"at": "2026-01-05T10:02:00Z", "card": True},
        "cap": 3000,
        "pipes": 2,
        "decided": stage_no_restart.STARTED,
        "develop_issues": "acme/issues",
        "split": 7000,
        "labels": {
            "extra": ["ci-extra-2"],
            "red": "ci-red-2",
            "stability": "stability-2",
        },
    }

    def test_each_key_s_values_fit_and_its_edit_changes_it_alone(self):
        doc = settings_file.loads(
            settings_file.template(file_types.DEPLOY, stage_deploy_path.PROFILE),
            file_types.DEPLOY,
        )
        stage_no_restart.filled(doc.values)
        text = settings_file.dump(doc)
        before = settings_file.loads(text, file_types.DEPLOY)
        self.assertEqual(settings_file.misfits(before), [])
        new, want = text, copy.deepcopy(before.values)
        for key, check in stage_no_restart.CHECKS.items():
            with self.subTest(key=key):
                self.assertEqual(registry.get(before.values, key), check.values[0])
                self.assertNotEqual(check.values[0], check.values[1])
                self.assertNotEqual(check.shown[0], check.shown[1])
            new = stage_deploy_path.edited(new, key, check.values[1])
            deploy_tree.put(want, key, check.values[1])
        after = settings_file.loads(new, file_types.DEPLOY)
        self.assertEqual(settings_file.misfits(after), [])
        self.assertEqual(after.values, want)
        self.assertEqual(len(new.splitlines()), len(text.splitlines()))

    def test_what_each_reader_shows_of_a_seen(self):
        shows = {k: c.show(self.AFTER) for k, c in stage_no_restart.CHECKS.items()}
        self.assertEqual(
            shows, {k: c.shown[1] for k, c in stage_no_restart.CHECKS.items()}
        )
        self.assertTrue(stage_no_restart.recorded(self.AFTER))

    def test_a_full_tick_for_another_reason_and_no_card_or_timer(self):
        seen = copy.deepcopy(self.AFTER)
        seen["idle"]["why"] = ["no full tick on record"]
        seen["idle"]["due"] = {}
        seen["spend"] = {"at": None, "card": False}
        for key, want in (
            (stage_no_restart.FULL_TICK_MAX, False),
            (stage_no_restart.SPEND, False),
            ("features.weekly_report.enabled", False),
            ("features.review_automation.schedule.push_cooldown_minutes", None),
        ):
            with self.subTest(key=key):
                self.assertEqual(stage_no_restart.CHECKS[key].show(seen), want)

    def test_nothing_seen_shows_nothing(self):
        for key, check in stage_no_restart.CHECKS.items():
            with self.subTest(key=key):
                self.assertIsNone(check.show({}))
        self.assertFalse(stage_no_restart.recorded({}))
        unread = dict(self.AFTER, idle=dict(self.AFTER["idle"], fp=None))
        self.assertFalse(stage_no_restart.recorded(unread))

    def test_the_large_pr_is_over_the_cap_at_its_first_value_only(self):
        """The large PR is too large for the autopilot at the cap's first value and not at its second
        (rkmon_lib/review.py oversized), so that the advance's decision on it changes with the cap, once its in-depth
        review is approved, as the stage approves it.
        """
        from rkmon_lib import review

        large = stage_no_restart.LARGE
        key = f"{stage_no_restart.RA}.large_prs.in_depth_review_over_lines"
        st = {
            "prs": {
                str(large): {
                    "n": large,
                    "head": "1" * 40,
                    "size": stage_no_restart.LARGE_LINES,
                }
            },
            review.APPROVED: [review.approval(large)],
        }
        plan = [{"kind": "FIRST_ROUND", "pr": large, "text": f"#{large}"}]
        for cap, over in zip(stage_no_restart.CHECKS[key].values, (True, False)):
            with self.subTest(cap=cap):
                cfg = {"autopilot_max_lines": cap}
                self.assertEqual(bool(review.oversized(cfg, st, plan)), over)

    def test_what_the_stage_reads_of_the_monitor_s_state(self):
        """SEEN_SCRIPT on a state folder, as the stage runs it in the container: the advance's last decision on the
        large PR is the last of the tick's files and the review driver's, in their order, the session's lines among
        them; another PR's line or the large PR's other lines are none."""
        from rkmon_lib import gh_cap, idle, orchestrator, scaler, spend

        large, left = stage_no_restart.LARGE, stage_no_restart.LEFT
        files = {
            "20260105T100000Z-tick.md": f"- {left} #{large} (first-round, 30 lines > 20): too large\n\n"
            f"{left} #{large} (first-round, 30 lines > 20): too large\n",
            "20260105T100100Z-driver.md": f"{stage_no_restart.STARTED} #{large} (first-round): staged\n",
            "20260105T100100Z-tick.md": f"REVIEW_READY #{large}: staged\n"
            f"{left} #{large + 1} (first-round, 40 lines > 30): too large\n"
            f"{left} #{large}0 (first-round, 90 lines > 40): too large\n",
            "20260105T100200Z-tick.md": "TICK 2026-01-05T10:02:00Z full\n",
        }
        with tempfile.TemporaryDirectory() as d:
            state = pathlib.Path(d)
            prs = {n: dict(rec, size=3) for n, rec in self.AFTER["prs"].items()}
            st = {
                idle.KEY: self.AFTER["idle"],
                "prs": prs,
                "spend": {"at": self.AFTER["spend"]["at"]},
            }
            (state / "state.json").write_text(json.dumps(st), encoding="utf-8")
            (state / gh_cap.RECORD).write_text(
                json.dumps({"cap_per_hour": 3000}), encoding="utf-8"
            )
            (state / scaler.STATE).write_text(
                json.dumps({"pipes": {"start": 2}}), encoding="utf-8"
            )
            (state / spend.SUMMARY).write_text("{}", encoding="utf-8")
            (state / orchestrator.TICKS).mkdir()
            for name, text in files.items():
                (state / orchestrator.TICKS / name).write_text(text, encoding="utf-8")
            args = json.dumps(stage_no_restart.seen_args(str(state)))
            out = subprocess.run(
                [sys.executable, "-I", "-B", "-c", stage_no_restart.SEEN_SCRIPT, args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        seen = json.loads(out.splitlines()[-1])
        read = ("idle", "prs", "spend", "cap", "pipes", "decided")
        self.assertEqual(seen, {k: self.AFTER[k] for k in read})

    def test_the_last_read_of_the_open_develop_issues(self):
        """The repository of the tick's last read of the open develop issues (rkmon_lib/breakage.py fetch_open), from
        the stand-ins' log; no other read of issues counts."""
        from rkmon_lib import breakage, gh

        def reads(repo: str) -> list:
            cfg = {
                "repo": "acme/widgets",
                "incident_issue": {
                    "repo": repo,
                    "labels": ["ci-red"],
                    "stability_label": "stability",
                },
            }
            with mock.patch.object(gh, "rest", return_value=[]) as rest:
                breakage.fetch_open(cfg)
            return [
                {"path": "/" + c.args[0], "status": 200} for c in rest.call_args_list
            ]

        others = [
            {
                "path": "/repos/acme/widgets/issues/3/comments?per_page=100",
                "status": 200,
            },
            {
                "path": "/repos/acme/other/issues?labels=x&state=all&since=2026-01-05T00:00:00Z",
                "status": 200,
            },
            {**reads("acme/refused")[0], "status": 401},
        ]
        log = reads("") + reads("acme/issues") + others
        self.assertEqual(stage_no_restart.issues_read(log), "acme/issues")
        self.assertEqual(stage_no_restart.issues_read(reads("")), "acme/widgets")
        self.assertIsNone(stage_no_restart.issues_read(others))

    def test_rk_s_refusals_and_its_split_line_are_read_unwrapped(self):
        from rkmon_lib import breakage, split, wrapping

        checks = stage_no_restart.CHECKS

        def refusal(at: int, labels_at: int) -> str:
            """rk issue-file's refusal of the issue file of AT, under the develop guard's labels at their values
            LABELS_AT, as the settings give them to rk (breakage.labels and extra_labels: incident_issue.labels, the
            CI-red label first)."""
            value = {key: check.values[labels_at] for key, check in checks.items()}
            cfg = {
                "incident_issue": {
                    "labels": [
                        value[stage_no_restart.CI_RED],
                        *value[stage_no_restart.EXTRA],
                    ],
                    "stability_label": value[stage_no_restart.STABILITY],
                }
            }
            with tempfile.TemporaryDirectory() as d:
                path = pathlib.Path(d) / "issue.md"
                path.write_text(stage_no_restart.issue_text(at), encoding="utf-8")
                with self.assertRaises(ValueError) as said:
                    breakage.parse_file(str(path), cfg)
            return "\n".join(wrapping.wrap(f"REFUSED: {said.exception}", 60))

        for at in (0, 1):
            with self.subTest(at=at):
                self.assertEqual(
                    stage_no_restart.refused_labels(refusal(at, at)),
                    {
                        "extra": checks[stage_no_restart.EXTRA].shown[at],
                        "red": checks[stage_no_restart.CI_RED].shown[at],
                        "stability": checks[stage_no_restart.STABILITY].shown[at],
                    },
                )
        # the file of after the change, under the labels of before it, carries neither kind's label
        self.assertEqual(stage_no_restart.refused_labels(refusal(1, 0)), {})
        res = {"parts": [], "data": [], "max_lines": 8000, "links_between_parts": []}
        lines = split.summary(1, "fr", pathlib.Path("/r"), [], res, 20, effort="high")
        self.assertEqual(
            stage_no_restart.split_size("\n".join(wrapping.wrap(lines[0], 40))), 8000
        )
        self.assertIsNone(
            stage_no_restart.split_size("SPLIT #1: the diff was not read")
        )

    def test_confirm_s_line_under_each_change(self):
        from rkmon_lib import change_plan, confirm

        lines = []
        for key, check in stage_no_restart.CHECKS.items():
            change = change_plan.Change(key, *check.values, change_plan.NO_RESTART)
            lines += confirm._setting(change)
        other = change_plan.Change(
            "monitor.tick_interval_minutes", 1, 15, change_plan.NEW_CONTAINER
        )
        lines += confirm._setting(other)
        out = "\n".join(
            ["Your changes:", *lines, "CONFIRMED: 14 changes wait for apply"]
        )
        for key in stage_no_restart.CHECKS:
            with self.subTest(key=key):
                self.assertEqual(
                    stage_no_restart.applies(out, key),
                    confirm.APPLIES[change_plan.NO_RESTART],
                )
        self.assertEqual(
            stage_no_restart.applies(out, "monitor.tick_interval_minutes"),
            confirm.APPLIES[change_plan.NEW_CONTAINER],
        )
        self.assertEqual(stage_no_restart.applies(out, "monitor.tick"), "")


class TestTheFirstApplyStage(unittest.TestCase):
    """What the first-apply stage (stage_first_apply.py) reads and makes on its own: the earlier deployment's
    docker/.env, which names its own compose project and image; the deploy settings its own setup reads; a folder's
    contents and what changed in them; what a failed first apply may change in the volumes, the write login it keeps
    byte for byte, and the modes the container's layout gives them; the earlier doctor's failing rows as apply shows
    them; the next line in standby after the first apply; a stop mark left in the new monitor's state, as its rk stopped
    says it; what was written between two readings, and its modes; docker's times; the earlier package's folder,
    copied out of the folder of its image's own code that its entry point names; and the stage skipped without the
    earlier image, which the image test hands it.
    """

    def test_the_earlier_env_names_its_own_project_and_image(self):
        text = stage_first_apply.older_env(
            "l2test-unit-first-apply",
            "registry.example.com:5000/rocke-monitor:unit-earlier",
            18999,
            1001,
            1002,
        )
        self.assertEqual(
            text.splitlines(),
            [
                "COMPOSE_PROJECT_NAME=l2test-unit-first-apply",
                "RKMON_IMAGE=registry.example.com:5000/rocke-monitor",
                "RKMON_TAG=unit-earlier",
                "RKMON_ADMIN_HOST=127.0.0.1",
                "RKMON_PORT=18999",
                "RKMON_UID=1001",
                "RKMON_GID=1002",
            ],
        )

    def test_the_earlier_settings_name_the_stand_ins_in_the_earlier_layout(self):
        self.assertEqual(
            json.loads(stage_first_apply.earlier_settings("a/b/rocke")),
            {
                "kind": "deploy_settings",
                "schema_version": 1,
                "repository": {
                    "name": stage_deploy_path.REPOSITORY,
                    "path": "a/b/rocke",
                    "scope_labels": [stage_deploy_path.LABEL],
                },
                "integrations": {"github_cli": {"account": stage_deploy_path.LOGIN}},
            },
        )

    def test_a_folder_s_contents_and_what_changed_in_them(self):
        tree, findings = stage_first_apply.tree, stage_first_apply.tree_findings
        with tempfile.TemporaryDirectory() as d:
            top = pathlib.Path(d)
            (top / "docker").mkdir()
            env = top / "docker" / ".env"
            env.write_text("A=1\n", encoding="utf-8")
            env.chmod(0o644)
            (top / "VERSION").write_text("1\n", encoding="utf-8")
            os.symlink("VERSION", top / "latest")
            before = tree(top)
            self.assertEqual(before["latest"], ["link", "VERSION"])
            self.assertEqual(before["docker"][0], "folder")
            self.assertEqual(before["docker/.env"][:2], ["file", 0o644])
            self.assertEqual(findings(before, tree(top)), [])
            env.chmod(0o600)
            (top / "VERSION").unlink()
            (top / "new").write_text("x", encoding="utf-8")
            self.assertEqual(
                findings(before, tree(top)),
                ["VERSION is gone", "new was added", "docker/.env changed"],
            )

    def test_what_a_failed_first_apply_may_change_in_the_volumes(self):
        kept = stage_first_apply.kept_findings

        def file(uid=1000, mode=0o600, sha="a", at=5):
            return {
                "uid": uid,
                "gid": 1000,
                "mode": mode,
                "sha256": sha,
                "mtime_ns": at,
            }

        def layout(path, have):  # a stand-in layout: the others' bits taken off
            return have & 0o770

        config, state = "/work/monitor/config.json", "/work/monitor/state/s.json"
        log, key, rk = "/work/monitor/logs/l", "/secrets/k", "/work/monitor/rk"
        stop, pid = stage_first_apply.STOP_WRITES[0], stage_first_apply.DASHBOARD_PID
        before = {
            config: file(mode=0o664),
            state: file(),
            log: file(),
            key: file(mode=0o644),
            rk: file(mode=0o775),
            stop: file(),
            pid: file(),
        }

        def exact(path):
            return path == config or path.startswith("/secrets/")

        # a file the new package added, and the older monitor's own writes before the new container was made
        same = {**before, state: file(sha="b", at=9), "/work/monitor/n": file(at=20)}
        self.assertEqual(kept(before, same, 10, exact, layout), ([], []))
        # the changes it allows, each said: the mode the container's layout gives in the work volume, what the new
        # container's stop writes, and the older dashboard's pid file gone
        allows = {**before, config: file(mode=0o660), stop: file(sha="b", at=10)}
        del allows[pid]
        self.assertEqual(
            kept(before, allows, 10, exact, layout),
            (
                [],
                [
                    "/work/monitor/config.json: its mode was 664, now 660, as the container's layout gives it",
                    f"{stop}: written after the new container was made",
                    f"{pid} is gone",
                ],
            ),
        )
        # what breaks the claim: in the secrets, a mode the container's layout would give is a change too
        after = {
            config: file(sha="b", at=1, mode=0o664),
            state: file(sha="b", at=10),
            key: file(uid=1001, mode=0o640),
            rk: file(mode=0o755),
            stop: file(),
            pid: file(),
        }
        self.assertEqual(
            kept(before, after, 10, exact, layout),
            (
                [
                    "/secrets/k: its uid was 1000, now 1001",
                    "/secrets/k: its mode was 644, now 640",
                    "/work/monitor/config.json: its bytes changed",
                    "/work/monitor/logs/l is gone",
                    "/work/monitor/rk: its mode was 775, now 755",
                    "/work/monitor/state/s.json: written after the new container was made",
                ],
                [],
            ),
        )

    def test_the_write_login_is_read_and_kept_byte_for_byte(self):
        """The stage reads the GitHub write login's volume too, where it plants a stand-in write login, which a failed
        first apply must keep byte for byte, with its owner and its mode, as each secret (kept_exactly).
        """
        s = stage_first_apply
        self.assertEqual(s.VOLUMES["rkmon-secrets-write"], s.SECRETS_WRITE)
        self.assertIn(s.SECRETS_WRITE, s.READ)
        self.assertTrue(s.WRITE_LOGIN.startswith(s.SECRETS_WRITE + "/"))
        for path in (
            s.WRITE_LOGIN,
            f"{s.SECRETS}/claude_env",
            s.CONFIG_JSON,
            s.PLANTED,
        ):
            self.assertTrue(s.kept_exactly(path), path)
        self.assertFalse(s.kept_exactly(s.STOP_WRITES[0]))

        def layout(path, have):  # the container's layout gives the secrets no mode
            return have

        # the write login as VOLUME_FILES reads it: the write user's, 600
        login = {"uid": 1999, "gid": 1000, "mode": 0o600, "sha256": "a", "mtime_ns": 5}
        before = {s.WRITE_LOGIN: login}
        kept = s.kept_findings(
            before, {s.WRITE_LOGIN: dict(login)}, 10, s.kept_exactly, layout
        )
        self.assertEqual(kept, ([], []))
        changed = {**login, "uid": 1000, "mode": 0o640, "sha256": "b", "mtime_ns": 1}
        self.assertEqual(
            s.kept_findings(
                before, {s.WRITE_LOGIN: changed}, 10, s.kept_exactly, layout
            ),
            (
                [
                    f"{s.WRITE_LOGIN}: its uid was 1999, now 1000",
                    f"{s.WRITE_LOGIN}: its mode was 600, now 640",
                    f"{s.WRITE_LOGIN}: its bytes changed",
                ],
                [],
            ),
        )

    def test_the_modes_rk_layout_gives_an_older_volume(self):
        """The stage reads them from docker/rk_layout.py, by the kind of the top folder a file is in."""
        mode = stage_first_apply.layout_mode(ROOT)
        home = "/work/monitor"
        self.assertEqual(mode(f"{home}/config.json", 0o664), 0o660)
        self.assertEqual(mode(f"{home}/rk", 0o775), 0o770)
        self.assertEqual(mode(f"{home}/state/published.html", 0o644), 0o660)
        self.assertEqual(mode(f"{home}/state/internal_state.json", 0o600), 0o600)
        # a file of a tree PR code reads: its group reads it and writes none
        self.assertEqual(mode("/work/repo/README.md", 0o664), 0o640)
        # a file of the working folder itself: a shared one
        self.assertEqual(mode("/work/notes", 0o664), 0o660)

    def test_the_earlier_doctor_s_failing_rows_as_apply_shows_them(self):
        """The earlier package's doctor's report and apply's lines as a run gives them: each FAIL row shown as that
        doctor gives it; a row said otherwise, or only named, is a finding, and so is a report with no FAIL row.
        """
        report = (
            "PASS  admin dashboard: on\n"
            "FAIL  claude session: on\n"
            "      fix: needs the Claude session's key (the secret claude_env: ./rkmon setup --section claude): "
            "./rkmon setup --section claude (on a host: --section services)\n"
            "WARN  Claude session settings: /work/claude-workspace/.claude/settings.json does not set "
            "autoCompactWindow 300000\n"
            "      fix: ./rkmon setup --section claude\n"
        )
        lead = "== apply: make the confirmed change on this deployment\n"
        drain = (
            "  drain: the older package's own, up to 60 minutes\n"
            "DRAINING: nothing new starts; waiting up to 60 min for the reviews in flight\n"
        )
        doctor = "  WARN  doctor before: 49 checks of the older package's own, not compared with the doctor after"
        shown = stage_first_apply.doctor_shown
        out = f"{lead}{doctor}; failing:\n  FAIL  claude session: on\n{drain}"
        self.assertEqual(shown(report, out), [])
        unseen = ["FAIL claude session: on (not shown as the earlier doctor gives it)"]
        self.assertEqual(
            shown(report, out.replace("  FAIL  claude", "  PASS  claude")), unseen
        )
        named = f"{lead}{doctor}; failing: claude\n        session\n{drain}"
        self.assertEqual(shown(report, named), unseen)
        passing = report.replace("FAIL  claude", "PASS  claude")
        self.assertEqual(
            shown(passing, out), ["the earlier package's doctor fails no check"]
        )

    def test_the_next_line_after_the_first_apply(self):
        """The next line in standby after the first apply as runs give it: APPLIED's, first-tick's before the new
        monitor's first tick has run since its start (with step 8's words while the records keep no first tick), or
        standby off's; first-tick's, once the tick has run, standby off's alone; no other line.
        """
        ok = stage_first_apply.applied_next
        tick = "next: ./rkmon first-tick"
        off = "next: ./rkmon standby off, once ./rkmon status and the dashboards read right"
        self.assertTrue(ok(tick))
        self.assertTrue(ok("next: step 8, first tick: ./rkmon first-tick"))
        self.assertTrue(ok(off))
        self.assertTrue(ok(off, ticked=True))
        self.assertFalse(ok(tick, ticked=True))
        self.assertFalse(ok("next: ./rkmon doctor"))
        self.assertFalse(ok(f"{tick} --timeout-minutes 5"))

    def test_a_stop_mark_left_as_rk_stopped_says_it(self):
        """The new monitor's rk stopped as a run gives it: no stop mark left is its exit 1 and "not stopped"; a stop
        mark left, or a container that does not answer, is a finding, with what was said.
        """
        left = stage_first_apply.stop_left
        self.assertEqual(left(1, "not stopped\n"), "")
        said = (
            "stopped by rk stop --drain at 2026-01-02T03:04:05+00:00; rk start resumes"
        )
        self.assertEqual(left(0, said + "\n"), f"rk stopped: exit 0: {said}")
        gone = "Error response from daemon: container 0123456789ab is not running"
        self.assertEqual(left(1, gone), f"rk stopped: exit 1: {gone}")
        self.assertEqual(left(126, ""), "rk stopped: exit 126: no answer")

    def test_what_was_written_between_two_readings(self):
        before = {"/a": {"sha256": "1"}, "/b": {"sha256": "1"}, "/c": {"sha256": "1"}}
        after = {"/a": {"sha256": "1"}, "/b": {"sha256": "2"}, "/d": {"sha256": "1"}}
        self.assertEqual(
            stage_first_apply.written(before, after),
            ["/b (changed)", "/d (added)", "/c (gone)"],
        )

    def test_the_modes_of_what_was_written_between_two_readings(self):
        def f(mode, at):
            return {"mode": mode, "mtime_ns": at}

        first = {"/a": f(0o664, 1), "/b": f(0o644, 1)}
        before = {"/a": f(0o660, 2), "/b": f(0o660, 2), "/c": f(0o660, 2)}
        after = {
            "/a": f(0o664, 3),
            "/b": f(0o660, 2),
            "/c": f(0o660, 4),
            "/d": f(0o664, 5),
        }
        self.assertEqual(
            stage_first_apply.written_modes(before, after, first),
            ["/a: 664, 660, 664", "/c: -, 660, 660", "/d: -, -, 664"],
        )

    def test_docker_s_times(self):
        epoch = stage_first_apply.epoch_ns
        self.assertEqual(epoch("1970-01-02T00:00:00.000000001Z"), 86400 * 10**9 + 1)
        self.assertEqual(epoch("1970-01-01T00:00:01.5Z"), 15 * 10**8)
        later = epoch("2026-11-23T01:02:04Z") - epoch("2026-11-23T01:02:03.25Z")
        self.assertEqual(later, 75 * 10**7)
        for bad in (
            "2026-11-23 01:02:03Z",
            "2026-11-23T01:02:03.1234567890Z",
            "2026-11-23T01:02:03+01:00",
            "",
        ):
            with self.subTest(stamp=bad), self.assertRaises(ValueError):
                epoch(bad)

    def test_an_image_s_own_code_is_in_the_folder_its_entry_point_names(self):
        folder = stage_first_apply.code_folder
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        entry = re.search(r"(?m)^ENTRYPOINT (.+)$", dockerfile).group(1)
        self.assertEqual(folder(entry), container_layout.IMAGE_CODE)
        # an earlier package's image, whose code is in another folder
        self.assertEqual(
            folder(
                '["/usr/bin/tini","--","/opt/rocke_pr_monitor/docker/entrypoint.sh"]'
            ),
            "/opt/rocke_pr_monitor",
        )
        for bad in (
            "",
            "null",
            "[]",
            '["/bin/sh"]',
            '["opt/x/docker/entrypoint.sh"]',
            '["/docker/entrypoint.sh"]',
            '["/opt/x/entrypoint.sh"]',
            '{"/opt/x/docker/entrypoint.sh": 1}',
        ):
            with self.subTest(entrypoint=bad), self.assertRaises(ValueError):
                folder(bad)

    def test_the_earlier_package_is_copied_out_of_the_folder_its_image_names(self):
        """An earlier package's image whose own code is in another folder than this package's: the stage copies
        that folder out of it, as the deployment's folder, beside which nothing is left, with the docker/.env of the
        earlier package's settings step."""
        code = "/opt/rocke_pr_monitor"
        t = stage_first_apply.FirstApply(
            "docker",
            "rocke-monitor:l2test-unit",
            str(ROOT),
            "l2test-unit",
            [],
            mock.Mock(),
            "rocke-pr-monitor:l2test-earlier",
        )
        self.addCleanup(shutil.rmtree, t.top, True)
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            for name in ("rk", "docker/rkmon"):
                info = tarfile.TarInfo(f"rocke_pr_monitor/{name}")
                info.size, info.mode = len(name), 0o555
                tar.addfile(info, io.BytesIO(name.encode()))
        copied = []

        def must(*args, timeout=300):
            if args[:2] == ("image", "inspect"):
                return f'["/usr/bin/tini","--","{code}/docker/entrypoint.sh"]'
            return ""

        def run(argv, **_how):
            copied.append(argv[2])
            if argv[2].endswith(f":{code}"):
                return subprocess.CompletedProcess(argv, 0, data.getvalue(), b"")
            return subprocess.CompletedProcess(argv, 1, b"", b"no such folder")

        with mock.patch.object(t, "must", side_effect=must), mock.patch.object(
            t, "docker", return_value=(0, "")
        ), mock.patch.object(stage_first_apply.subprocess, "run", side_effect=run):
            t.earlier_deployment()
        self.assertEqual(copied, [f"{t.scope}-earlier-code:{code}"])
        self.assertEqual((t.folder / "docker" / "rkmon").read_text(), "docker/rkmon")
        self.assertEqual([p.name for p in t.top.iterdir()], [stage_first_apply.FOLDER])
        env = t.folder / stage_first_apply.ENV
        self.assertEqual(
            env.read_text(encoding="utf-8"),
            stage_first_apply.older_env(
                t.scope, t.earlier_derived, t.port, os.getuid(), os.getgid()
            ),
        )
        self.assertEqual(
            stat.S_IMODE(env.stat().st_mode), stage_first_apply.OLDER_ENV_MODE
        )

    def test_without_the_earlier_image_the_stage_is_skipped(self):
        res, out = mock.Mock(), io.StringIO()
        with mock.patch.object(
            stage_first_apply, "FirstApply"
        ) as run, contextlib.redirect_stdout(out):
            stage_first_apply.first_apply_stage(
                "docker", "x:t", str(ROOT), "l2test-unit", [], res, ""
            )
        run.assert_not_called()
        res.result.assert_not_called()
        self.assertTrue(
            out.getvalue().startswith(
                "SKIP  first-apply: it needs an image of the package before this one: "
            ),
            out.getvalue(),
        )

    def test_the_image_test_hands_the_stage_its_earlier_image(self):
        hr = image.hostrun
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ))
            stack.enter_context(mock.patch.object(hr, "setup_output"))
            stack.enter_context(
                mock.patch.object(hr, "pick_engine", return_value="docker")
            )
            stack.enter_context(mock.patch.object(hr, "quiet", return_value=0))
            stack.enter_context(
                mock.patch.object(image, "image_labels", return_value={})
            )
            stage = stack.enter_context(
                mock.patch.object(image.stage_first_apply, "first_apply_stage")
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rc = image.main(
                ["--no-build", "--image", "x:t", "--earlier-image", "y:t"]
                + ["--stages", "first-apply", "--names", "l2test-unit"]
            )
        self.assertEqual(rc, 0)
        stage.assert_called_once()
        args = stage.call_args.args
        self.assertEqual(
            (args[0], args[1], args[3], args[-1]),
            ("docker", "x:t", "l2test-unit", "y:t"),
        )


class TestTheFakeModels(unittest.TestCase):
    """The fake reviewer models the model stages run (tests/fakes/fake_models.py) print nothing for a client that goes
    away before its answer, as a worker that a stop ends mid-chat does; an error of their own prints its traceback.
    """

    def test_a_client_that_goes_away_before_its_answer(self):
        body = json.dumps({"model": fake_models.OLLAMA_GOOD, "messages": []}).encode()
        chat = (
            b"POST /api/chat HTTP/1.1\r\nHost: fake\r\nContent-Length: %d\r\n\r\n"
            % len(body)
        )
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err):
            srv = fake_models.start(d, "ollama", delay_s=1.0)
            # so that stop() waits for the request's thread
            srv.srv.daemon_threads = False
            try:
                with socket.create_connection(srv.srv.server_address[:2]) as c:
                    c.sendall(chat + body)
                    deadline = time.monotonic() + 30
                    while srv.handler.chats == 0 and time.monotonic() < deadline:
                        time.sleep(0.01)
                    # The client goes at once, with a reset: the answer, a second later, finds it gone.
                    c.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
            finally:
                srv.stop()
            log = fake_models.read_log(d, "ollama")
        # The chat was read, and its answer never went out (the fake logs an answer as it sends it).
        self.assertEqual((srv.handler.chats, log), (1, []))
        self.assertEqual(err.getvalue(), "")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                raise ValueError("an error of the fake's own")
            except ValueError:
                srv.srv.handle_error(None, ("127.0.0.1", 1))
        self.assertIn("ValueError: an error of the fake's own", err.getvalue())


class TestTheStandInsOpenPRs(unittest.TestCase):
    """The open PRs the stand-in GitHub serves (tests/fakes/fake_net.py CONFIG "prs", tests/fakerepo.py
    pull_request): each at refs/pull/N/head of the repository github.com serves, from develop's head, with the lines,
    pushes and date CONFIG asks for, and our formal reviews on the head of the push each names; and the models' and stop
    stages' (stage_models.make_prs), each one push of three lines, dated as the repository's own commits. They are open:
    a search finds them, and none for issues, or for merged or closed PRs.
    """

    AT = "2026-01-05T10:00:00Z"
    # CONFIG "prs": a PR of 30 changed lines; one pushed at AT; one of two pushes, its first reviewed at AT
    PRS = [
        {"number": 1, "lines": 30},
        {"number": 2, "at": AT},
        {
            "number": 3,
            "pushes": 2,
            "reviews": [{"push": 1, "at": AT, "body": "LINE 1\n\nNo findings.\n"}],
        },
    ]

    def test_the_prs_config_asks_for(self):
        with tempfile.TemporaryDirectory() as d:
            state = pathlib.Path(d)
            develop, served = fake_net.repository(state, "acme/widgets", self.PRS, "me")
            bare = state / "git" / "acme" / "widgets.git"
            self.assertEqual(
                fakerepo.git(bare, "rev-parse", "refs/heads/develop"), develop
            )
            self.assertEqual([pr["number"] for pr in served], [1, 2, 3])
            for pr in served:
                n = pr["number"]
                with self.subTest(pr=n):
                    self.assertEqual(
                        fakerepo.git(bare, "rev-parse", f"refs/pull/{n}/head"),
                        pr["head"],
                    )
                    self.assertEqual(
                        (pr["pushes"][-1], pr["base_sha"]), (pr["head"], develop)
                    )
            large, pushed, reviewed = served
            self.assertEqual(
                (large["additions"], len(large["pushes"]), large["reviews"]),
                (30, 1, []),
            )
            # its commit's dates, in the time zone they were given in (UTC)
            dates = fakerepo.git(
                bare,
                "log",
                "-1",
                "--date=format:%Y-%m-%dT%H:%M:%SZ",
                "--format=%cd %ad",
                pushed["head"],
            )
            self.assertEqual(dates.split(), [self.AT] * 2)
            self.assertEqual((pushed["committed_at"], pushed["reviews"]), (self.AT, []))
            [review] = reviewed["reviews"]
            self.assertEqual(len(reviewed["pushes"]), 2)
            self.assertEqual(
                (
                    review["user"]["login"],
                    review["commit_id"],
                    review["submitted_at"],
                    review["state"],
                    review["body"],
                ),
                (
                    "me",
                    reviewed["pushes"][0],
                    self.AT,
                    "COMMENTED",
                    self.PRS[2]["reviews"][0]["body"],
                ),
            )

    def test_the_models_and_stop_stages_prs(self):
        with tempfile.TemporaryDirectory() as d:
            fakerepo.make(d)
            origin = pathlib.Path(d) / "origin"
            develop = fakerepo.git(origin, "rev-parse", "develop")
            prs = stage_models.make_prs(origin)
            self.assertEqual([pr["number"] for pr in prs], list(stage_models.PRS))
            for pr in prs:
                n = pr["number"]
                with self.subTest(pr=n):
                    self.assertEqual(
                        (pr["pushes"], pr["base_sha"], pr["additions"]),
                        ([pr["head"]], develop, 3),
                    )
                    self.assertEqual(
                        pr["committed_at"], fakerepo.ENV["GIT_COMMITTER_DATE"]
                    )
                    self.assertEqual(
                        fakerepo.git(origin, "rev-parse", f"refs/pull/{n}/head"),
                        pr["head"],
                    )
                    self.assertIn(f"+change {n}\n", pr["diff"])
            self.assertEqual(
                fakerepo.git(origin, "rev-parse", "--abbrev-ref", "HEAD"), "develop"
            )

    def test_a_search_finds_the_open_prs_only(self):
        """The searches of the tick (rkmon_lib/snapshot.py), of the develop issues (rkmon_lib/breakage.py) and of the
        weekly report (rkmon_lib/stats.py compute), as the stand-in GitHub answers them.
        """
        from rkmon_lib import snapshot

        cfg = {"repo": "acme/widgets"}
        label = 'repo:acme/widgets is:pr label:"x" '
        week = "2026-01-05T00:00:00Z..2026-01-11T23:59:59Z"
        searches = {
            snapshot.label_query(cfg, ["x"]): True,
            label + "created:" + week: True,
            label + "commenter:me updated:>=2026-01-05T00:00:00Z": True,
            snapshot.label_query(cfg, ["x"], kind="issue", repo="acme/issues"): False,
            label + "is:merged merged:" + week: False,
            label + "is:unmerged is:closed closed:" + week: False,
        }
        with tempfile.TemporaryDirectory() as d:
            fakerepo.make(d)
            prs = stage_models.make_prs(pathlib.Path(d) / "origin")
            urls, stop = fake_servers.start(pathlib.Path(d) / "fakes", prs=prs)
            try:
                found = {q: self.search(urls["github"], q) for q in searches}
            finally:
                stop()
        numbers = [pr["number"] for pr in prs]
        self.assertTrue(numbers)
        self.assertEqual(
            found, {q: numbers if served else [] for q, served in searches.items()}
        )

    @staticmethod
    def search(base: str, q: str) -> list:
        """The numbers a GraphQL search for Q finds, at the stand-in GitHub at BASE."""
        query = (
            "query($q: String!) { search(query: $q, type: ISSUE) { nodes { number } } }"
        )
        req = urllib.request.Request(
            base + "/graphql",
            data=json.dumps({"query": query, "variables": {"q": q}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return [n["number"] for n in json.load(r)["data"]["search"]["nodes"]]


class TestTheStandInsWriteLogin(unittest.TestCase):
    """The stand-in GitHub's write login (tests/fakes/fake_net.py CONFIG github.write_token), as the stand-ins serve it
    on api.github.com: the login check's probes (a comment with no body, a branch on a commit that does not exist)
    answered as GitHub answers a login that may make them (422), the read login's refused (403), a token it does not
    know 401; a branch whose commit is not an id by its shape refused for its shape (422) whatever the login; a comment
    on a pull request as `gh pr comment` makes one (the pull request looked up by its number, then the comment)
    answered with the comment's URL, and logged as the write login's write; every other write refused (403). Each
    call logged with the token it carried, by its kind.
    """

    READ, WRITE = "read-token", "write-token"
    PROBE = "/repos/acme/widgets/issues/1/comments"
    PUSH = "/repos/acme/widgets/git/refs"
    LOOKUP = (
        "query PullRequestByNumber($owner: String!, $repo: String!, $pr_number: Int!) { repository(owner: $owner, "
        "name: $repo) { pullRequest(number: $pr_number) { id number url } } }"
    )
    COMMENT = "mutation CommentCreate($input:AddCommentInput!){addComment(input: $input){commentEdge{node{url}}}}"

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.state = pathlib.Path(d.name)
        github = {"login": "op-login", "token": self.READ, "write_token": self.WRITE}
        config = {
            "github": {**github, "repository": "acme/widgets"},
            "gateway": {"host": "llm-gateway.example.com"},
        }
        attrs = {"config": config, "state": self.state, "prs": [], "log_path": None}
        handler = type("NetHandler", (fake_net.Net,), attrs)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.port = server.server_address[1]

    def post(self, token: str, path: str, body: dict) -> tuple:
        """A POST of BODY to api.github.com's PATH, with TOKEN as gh sends one: (its status, its answer)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            headers = {"Host": fake_net.API, "Authorization": f"token {token}"}
            conn.request("POST", path, json.dumps(body), headers)
            r = conn.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            conn.close()

    def logged(self, *keys: str) -> list:
        lines = (self.state / "requests.log").read_text(encoding="utf-8").splitlines()
        return [tuple(json.loads(x).get(k) for k in keys) for x in lines]

    def test_the_login_checks_probe(self):
        """Each of its writes: the comment with no body, and the push of a branch on a commit that does not exist."""
        branch = {"ref": gh.PROBE_REF, "sha": gh.PROBE_SHA}
        for path, sent, answer in (
            (self.PROBE, {"body": ""}, fake_net.VALIDATION_FAILED),
            (self.PUSH, branch, fake_net.OBJECT_MISSING),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.post(self.READ, path, sent)[0], 403)
                self.assertEqual(self.post(self.WRITE, path, sent), (422, answer))
                self.assertEqual(self.post("another", path, sent)[0], 401)
        self.assertEqual(
            self.logged("token", "status", "probe"),
            [("login", 403, True), ("write", 422, True), ("other", 401, None)] * 2,
        )

    def test_a_branch_of_the_wrong_shape_is_refused_for_its_shape_whatever_the_login(
        self,
    ):
        """A branch whose commit is not an id by its shape (no name and no commit, or a commit too short): GitHub
        refuses it for its shape (422, gh.SHAPE_REFUSED) before it asks what the token may do, so its answer says
        nothing of the login; a token it does not know is still answered 401."""
        for sent in ({"ref": "", "sha": ""}, {"ref": gh.PROBE_REF, "sha": "0"}):
            with self.subTest(sent=sent):
                for token in (self.READ, self.WRITE):
                    status, said = self.post(token, self.PUSH, sent)
                    self.assertEqual(status, 422)
                    self.assertTrue(said["message"].startswith(gh.SHAPE_REFUSED), said)
                self.assertEqual(self.post("another", self.PUSH, sent)[0], 401)
        self.assertEqual(
            self.logged("token", "status", "probe"),
            [("login", 422, False), ("write", 422, False), ("other", 401, None)] * 2,
        )

    def test_a_comment_as_gh_pr_comment_makes_one(self):
        lookup = {"owner": "acme", "repo": "widgets", "pr_number": 7}
        status, found = self.post(
            self.WRITE, "/graphql", {"query": self.LOOKUP, "variables": lookup}
        )
        pr = found["data"]["repository"]["pullRequest"]
        self.assertEqual(
            (status, pr["number"], pr["url"]),
            (200, 7, f"https://{fake_net.WEB}/acme/widgets/pull/7"),
        )
        made = {"input": {"subjectId": pr["id"], "body": "a comment"}}
        status, said = self.post(
            self.WRITE, "/graphql", {"query": self.COMMENT, "variables": made}
        )
        self.assertEqual(status, 200)
        self.assertRegex(
            said["data"]["addComment"]["commentEdge"]["node"]["url"],
            r"^https://github\.com/acme/widgets/pull/7#issuecomment-\d+$",
        )
        self.assertEqual(
            self.logged("token", "status", "write"),
            [("write", 200, None), ("write", 200, True)],
        )

    def test_every_other_write_is_refused(self):
        made = {"input": {"subjectId": f"{fake_net.PR_ID}7", "body": "a comment"}}
        other = {"input": {"subjectId": "I_an_issue", "body": "a comment"}}
        cases = {
            "the read login's comment": (
                self.READ,
                "/graphql",
                {"query": self.COMMENT, "variables": made},
            ),
            "a comment on no pull request": (
                self.WRITE,
                "/graphql",
                {"query": self.COMMENT, "variables": other},
            ),
            "a comment with a body, through REST": (
                self.WRITE,
                self.PROBE,
                {"body": "x"},
            ),
            "a branch on another commit": (
                self.WRITE,
                self.PUSH,
                {"ref": gh.PROBE_REF, "sha": "1" * 40},
            ),
            "a tag on the id no object has": (
                self.WRITE,
                self.PUSH,
                {"ref": "refs/tags/rkmon-write-probe", "sha": fake_net.NULL_ID},
            ),
        }
        for name, (token, path, body) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self.post(token, path, body)[0], 403)
        self.assertEqual(
            self.logged("token", "write", "probe"),
            [("login", True, False)] + [("write", True, False)] * 4,
        )


class TestWhatTheDeployPathReadsOfTheStandIns(unittest.TestCase):
    """What the deploy path reads of the stand-ins' log (stage_deploy_path.py requests_findings), by the marks of the
    rkmon commands: each GitHub call with a login's token; a read, the clone and the gateway reached with its key (each
    unless asked for none); no write but the logins' probes before the command from which on the deployment sends, and
    from then on at least one, each the write login's and answered. One finding for each way a log differs from that.
    """

    SENDS = "./rkmon standby off"
    MARKS = [(10.0, "./rkmon start"), (20.0, SENDS), (30.0, SENDS)]
    SENT = {"method": "POST", "path": "/graphql", "token": "write", "write": True}

    @staticmethod
    def call(t: float, **changed) -> dict:
        """A GitHub read with the read login's token, at time T, but for CHANGED."""
        api = {"host": "api.github.com", "method": "GET", "path": "/user"}
        return {"t": t, **api, "status": 200, "token": "login", **changed}

    def log(self, *more: dict, clone: bool = True, gateway: bool = True) -> list:
        """A log the deploy path passes with, but for the clone when CLONE is false and the gateway when GATEWAY is: a
        read, the clone, the gateway reached with its key, each login's probe; then MORE.
        """
        probe = {"method": "POST", "path": "/repos/acme/widgets/issues/1/comments"}
        called = {
            "host": stage_deploy_path.GATEWAY,
            "method": "GET",
            "path": "/v1/models",
            "status": 200,
        }
        upload = {
            "host": "github.com",
            "method": "POST",
            "path": "/acme/widgets.git/git-upload-pack",
        }
        return [
            self.call(11.0),
            *([{"t": 12.0, **upload, "status": 200}] if clone else []),
            *(
                [{"t": 13.0, **called, "key_headers": ["x"], "key": "right"}]
                if gateway
                else []
            ),
            self.call(14.0, **probe, status=403, write=True, probe=True),
            self.call(15.0, **probe, status=422, token="write", write=True, probe=True),
            *more,
        ]

    def test_a_log_with_no_write(self):
        bad, good = stage_deploy_path.requests_findings(self.log(), self.MARKS)
        self.assertEqual(bad, [])
        self.assertIn("1 GitHub reads with the read login's token, 1 fetches", good[0])
        self.assertTrue(good[0].endswith("no write but 2 login probes"), good[0])

    def test_a_send_after_the_command_that_sends(self):
        log = self.log(self.call(25.0, **self.SENT, probe=False))
        bad, good = stage_deploy_path.requests_findings(log, self.MARKS, self.SENDS)
        self.assertEqual(bad, [])
        self.assertTrue(
            good[0].endswith(
                f"; after {self.SENDS}, 1 sent with the write login's token"
            )
        )

    def test_no_clone_or_gateway_asked_for(self):
        clone, gateway = (
            "no clone of the repository",
            "no call reached the gateway with its key",
        )
        cases = {  # (clone, gateway): what the log holds and the stage asks for; the findings when both are asked for
            "no clone": ((False, True), [clone]),
            "no gateway call": ((True, False), [gateway]),
            "neither": ((False, False), [clone, gateway]),
        }
        for name, ((c, g), wanted) in cases.items():
            with self.subTest(case=name):
                log = self.log(clone=c, gateway=g)
                found = stage_deploy_path.requests_findings(
                    log, self.MARKS, clone=c, gateway=g
                )
                self.assertEqual(found[0], [])
                bad = stage_deploy_path.requests_findings(log, self.MARKS)[0]
                self.assertEqual(bad, wanted)

    def test_each_way_a_log_differs(self):
        sent = {**self.SENT, "probe": False}
        none_after = (
            f"no write with the write login's token went out after {self.SENDS}"
        )
        cases = {
            "a write, nothing sends": (
                "",
                [self.call(25.0, **sent)],
                ["a write, during"],
            ),
            "a send before it": (
                self.SENDS,
                [self.call(18.0, **sent)],
                ["a write, during ./rkmon start: POST", none_after],
            ),
            "no send after it": (self.SENDS, [], [none_after]),
            "the read login's write after it": (
                self.SENDS,
                [self.call(25.0, **{**sent, "token": "login"}, status=403)],
                [f"a write, during {self.SENDS}", none_after],
            ),
            "a send refused after it": (
                self.SENDS,
                [self.call(25.0, **sent, status=403)],
                [f"a write, during {self.SENDS}", none_after],
            ),
            "a call with no token": (
                "",
                [self.call(16.0, token="none")],
                ["without a login's token (none)"],
            ),
        }
        for name, (sends, more, wanted) in cases.items():
            with self.subTest(case=name):
                bad = stage_deploy_path.requests_findings(
                    self.log(*more), self.MARKS, sends
                )[0]
                self.assertEqual(len(bad), len(wanted), bad)
                for x, want in zip(bad, wanted):
                    self.assertIn(want, x)


class TestTheDeployPathsStageName(unittest.TestCase):
    """The deploy path under the name of the stage that walks it (stage_deploy_path.py DeployPath): the deploy-path
    stage's own, or that of a stage that walks the path to make the deployment it checks (STAGES, each class of a
    stage module that is a DeployPath). The scope that names its objects and its temporary folder, and each of its
    results, the path's own among them, take that name.
    """

    # Each class that walks the deploy path: the arguments of its own a run takes, the name its scope ends with, and
    # the words its results start with (several-deployments' deployments name their letter too).
    STAGES = (
        (stage_deploy_path.DeployPath, (), "deploy-path", "deploy-path: "),
        (stage_web_host.WebHost, (), "web-host", "web-host: "),
        (stage_no_restart.NoRestart, (), "no-restart", "no-restart: "),
        (stage_adopt.Adopt, (), "adopt", "adopt: "),
        (stage_one_time_move.OneTimeMove, (), "one-time-move", "one-time-move: "),
        (stage_team_prototype.TeamPrototype, (), "team-prototype", "team-prototype: "),
        (
            stage_first_apply.FirstApply,
            ("rocke-monitor:l2test-earlier",),
            "first-apply",
            "first-apply: ",
        ),
        (
            stage_several_deployments.Deployment,
            ("a",),
            "several-deployments-a",
            "several-deployments: deployment a, ",
        ),
    )

    def run_of(self, stage_class, *own) -> tuple:
        """A run of STAGE_CLASS, with OWN its own arguments, made as the image test makes one, but run no further; and
        the results it gives."""
        results = []
        res = mock.Mock()
        res.result.side_effect = lambda status, text: results.append((status, text))
        t = stage_class(
            "docker",
            "rocke-monitor:l2test-unit",
            str(ROOT),
            "l2test-unit",
            [],
            res,
            *own,
        )
        self.addCleanup(t.top.rmdir)  # all a run makes before it starts
        return t, results

    def test_the_scope_and_the_results_take_the_stage_s_name(self):
        for stage_class, own, stage, said in self.STAGES:
            with self.subTest(stage=stage):
                t, results = self.run_of(stage_class, *own)
                self.assertEqual(t.scope, f"l2test-unit-{stage}")
                self.assertTrue(t.top.name.startswith(f"l2test-unit-{stage}-"))
                self.assertIs(t.result(True, "a check"), True)
                self.assertIs(t.result(False, "another check"), False)
                # one of the path's own checks, which every such stage runs: a step's, here confirm's that exited 1
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIs(t.step(3, 1, ""), False)
                self.assertEqual(
                    results,
                    [
                        ("PASS", f"{said}a check"),
                        ("FAIL", f"{said}another check"),
                        ("FAIL", f"{said}step 3, confirm records the settings file"),
                    ],
                )

    def test_every_class_that_walks_the_deploy_path_is_one_of_stages(self):
        """Each class of a stage module (tests/container/stage_*.py) whose bases name DeployPath, or a class that does,
        is one STAGES runs."""
        import ast

        # each class of a stage module, as module.Class -> the names of its bases
        bases = {}
        for f in sorted((ROOT / "tests" / "container").glob("stage_*.py")):
            for node in ast.parse(f.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ClassDef):
                    bases[f"{f.stem}.{node.name}"] = {
                        ast.unparse(b).rpartition(".")[2] for b in node.bases
                    }
        walks = {"stage_deploy_path.DeployPath"}
        while True:
            more = {
                c
                for c, named in bases.items()
                if named & {w.rpartition(".")[2] for w in walks}
            }
            if more <= walks:
                break
            walks |= more
        self.assertIn("stage_several_deployments.Deployment", walks)
        self.assertEqual(
            walks, {f"{c.__module__}.{c.__name__}" for c, *_ in self.STAGES}
        )


class TestTheDeployPathsReaders(unittest.TestCase):
    """How the deploy path's stages read a command's output and wait for the container (stage_deploy_path.py): says
    finds a text however either is wrapped, and never one that is not there whole; polled reads until its answer is
    ready, or until the time it is given has passed, and an answer it cannot read is never ready.
    """

    def test_says_reads_each_run_of_white_space_as_one_space(self):
        out = "FAILED: web_host_ssh_key is not set;\n  nothing changed\n"
        for text, said in (
            ("web_host_ssh_key is not set; nothing changed", True),
            ("is not\nset;   nothing", True),
            ("web_host_ssh_key is set", False),
            ("nothing changed.", False),
        ):
            with self.subTest(text=text):
                self.assertIs(stage_deploy_path.says(out, text), said)

    def test_polled_reads_until_its_answer_is_ready(self):
        answers = iter({"n": n} for n in range(1, 10))
        with mock.patch.object(stage_deploy_path.time, "sleep") as slept:
            got = stage_deploy_path.polled(
                lambda: next(answers), lambda a: a["n"] == 3, 60
            )
        self.assertEqual(got, {"n": 3})
        self.assertEqual(slept.call_count, 2)

    def test_an_answer_it_cannot_read_is_never_ready_and_the_time_ends_it(self):
        ready, reads = mock.Mock(return_value=True), []

        def unread():
            reads.append(len(reads))
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

        # each look at the clock, 10 seconds after the last
        clock = iter(range(0, 100, 10))
        with mock.patch.object(
            stage_deploy_path.time, "time", side_effect=lambda: next(clock)
        ), mock.patch.object(stage_deploy_path.time, "sleep"):
            got = stage_deploy_path.polled(unread, ready, 25)
        self.assertEqual(got, {"unread": "Expecting value: line 1 column 1 (char 0)"})
        self.assertEqual(len(reads), 3)
        ready.assert_not_called()


class TestTheDeployPathsCommandsInItsContainer(unittest.TestCase):
    """How a stage that walks the deploy path runs a command in the deployment's container (stage_deploy_path.py
    DeployPath.inside, the one helper of every such stage): docker exec in the container docker compose runs for the
    deployment (monitor), as the monitor's user unless another is given, with the command's stdin, environment and
    folder; as_monitor in the folder of the code the container runs, RuntimeError when either of its commands fails.
    """

    CONTAINER = "l2test-unit-deploy-path-rkmon-1"

    def run_of(self, *answers) -> tuple:
        """A deploy path, run no further, whose engine lists CONTAINER for docker ps and answers each other command with
        the next of ANSWERS, (its exit code, its output); and the commands it is given: (the words after the
        engine's, the stdin)."""
        t = stage_deploy_path.DeployPath(
            "docker",
            "rocke-monitor:l2test-unit",
            str(ROOT),
            "l2test-unit",
            [],
            mock.Mock(),
        )
        self.addCleanup(t.top.rmdir)  # all a run makes before it starts
        calls, left = [], list(answers)

        def run(cmd, input=None, **_how):
            calls.append((cmd[1:], input))
            rc, out = (0, self.CONTAINER) if cmd[1] == "ps" else left.pop(0)
            return subprocess.CompletedProcess(cmd, rc, stdout=f"{out}\n".encode())

        patched = mock.patch.object(
            stage_deploy_path.subprocess, "run", side_effect=run
        )
        patched.start()
        self.addCleanup(patched.stop)
        return t, calls

    def test_a_command_runs_as_the_monitor_s_user_with_its_stdin_environment_and_folder(
        self,
    ):
        t, calls = self.run_of((0, "said"))
        got = t.inside(
            "rk", "issue-file", stdin=b"text", env={"RKMON_X": "1"}, workdir="/w"
        )
        self.assertEqual(got, (0, "said"))
        (ps, _), last = calls[0], calls[-1]
        self.assertEqual(ps[0], "ps")
        self.assertIn("label=com.docker.compose.oneoff=False", ps)
        self.assertEqual(
            last,
            (
                [
                    *("exec", "-i", "-u", container_layout.MONITOR.name),
                    *(
                        "-e",
                        "RKMON_X=1",
                        "-w",
                        "/w",
                        self.CONTAINER,
                        "rk",
                        "issue-file",
                    ),
                ],
                b"text",
            ),
        )

    def test_another_user_and_no_stdin(self):
        t, calls = self.run_of((3, "refused"))
        self.assertEqual(t.inside("id", user="root"), (3, "refused"))
        self.assertEqual(
            calls[-1], (["exec", "-i", "-u", "root", self.CONTAINER, "id"], b"")
        )

    def test_as_monitor_runs_in_the_folder_of_the_code_and_fails_with_either_command(
        self,
    ):
        user, code = container_layout.MONITOR.name, "/code/releases/r1"
        t, calls = self.run_of((0, code), (0, "out"))
        self.assertEqual(t.as_monitor("python3", "-c", "x"), "out")
        self.assertEqual(
            [words for words, _stdin in calls if words[0] == "exec"],
            [
                ["exec", "-i", "-u", user, self.CONTAINER, "rk", "--which-code"],
                [
                    *("exec", "-i", "-u", user, "-w", code, self.CONTAINER),
                    *("python3", "-c", "x"),
                ],
            ],
        )
        for answers, said in (
            (
                ((0, code), (2, "Traceback")),
                "python3 -c in the deployment's container exited 2: Traceback",
            ),
            (
                ((1, "rk: not found"),),
                "rk --which-code in the deployment's container exited 1: rk: not found",
            ),
        ):
            with self.subTest(said=said):
                t, _calls = self.run_of(*answers)
                with self.assertRaisesRegex(RuntimeError, re.escape(said)):
                    t.as_monitor("python3", "-c", "x")

    def test_no_stage_that_walks_the_path_has_a_helper_of_its_own(self):
        for stage_class, *_ in TestTheDeployPathsStageName.STAGES:
            with self.subTest(stage=stage_class.__name__):
                self.assertIs(stage_class.inside, stage_deploy_path.DeployPath.inside)


class TestTheDeployPathsSettingsVolume(unittest.TestCase):
    """How the deploy-path stage tries a write to the settings volume in the running container and reads the tries
    (stage_deploy_path.py WRITE_TRIES, read_only_findings): none for a volume nothing writes, each try refused and
    root's by the read-only mount; a finding for each try that wrote, for root's refused otherwise, and for a user whose
    tries did not run. The tries, made in a folder of this machine that may be written, write and leave nothing.
    """

    FOLDER = container_layout.SETTINGS_DIR
    NEW = f"{FOLDER}/write-try"
    LAYER = f"{FOLDER}/{container_layout.LAYER_FILE}"

    def test_a_volume_nothing_writes(self):
        ro = stage_deploy_path.READ_ONLY
        monitor = container_layout.MONITOR.name
        good = {
            monitor: {self.NEW: ro, self.LAYER: "Permission denied"},
            "root": {self.NEW: ro, self.LAYER: ro},
        }
        self.assertEqual(stage_deploy_path.read_only_findings(good), [])
        cases = {
            "the monitor's user wrote": (
                {monitor: {self.NEW: ro, self.LAYER: "written"}},
                f"as {monitor}, {self.LAYER} was opened to write",
            ),
            "root stopped by a permission alone": (
                {"root": {self.NEW: ro, self.LAYER: "Permission denied"}},
                f"as root, {self.LAYER}: Permission denied, not read-only file system",
            ),
            "root's tries not run": (
                {"root": "python3: not found"},
                "as root, the two tries did not run: python3: not found",
            ),
        }
        for what, (changed, said) in cases.items():
            with self.subTest(what):
                found = stage_deploy_path.read_only_findings({**good, **changed})
                self.assertEqual(found, [said])

    def test_the_tries_write_where_they_may_and_leave_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            layer = pathlib.Path(d, container_layout.LAYER_FILE)
            layer.write_text("{}\n", encoding="utf-8")

            def tried(folder: str, layer_file: str) -> dict:
                return json.loads(
                    subprocess.run(
                        [sys.executable, "-I", "-c", stage_deploy_path.WRITE_TRIES]
                        + [folder, layer_file],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                    ).stdout
                )

            self.assertEqual(
                tried(d, str(layer)),
                {os.path.join(d, "write-try"): "written", str(layer): "written"},
            )
            self.assertEqual(os.listdir(d), [layer.name])
            self.assertEqual(layer.read_text(encoding="utf-8"), "{}\n")
            gone = os.path.join(d, "gone")
            self.assertEqual(
                set(tried(gone, os.path.join(gone, layer.name)).values()),
                {"No such file or directory"},
            )


class TestTheIsolationProof(unittest.TestCase):
    """The isolation and canary stages' verdicts (stage_isolation.py Proof): a check fails when a fake secret is in its
    output, and when sudo says in it that it wrote no audit record (the container lacks AUDIT_WRITE).
    """

    def verdict(self, good: bool, out: str) -> tuple:
        p = stage_isolation.Proof()
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            good = p.expect(good, "a check", out)
        return good, p.failed, said.getvalue()

    def test_a_clean_output(self):
        self.assertEqual(
            self.verdict(True, "done\n"), (True, 0, "[01] PASS: a check\n")
        )
        self.assertEqual(self.verdict(False, "done\n")[:2], (False, 1))

    def test_a_fake_secret(self):
        out = f"token: {stage_isolation.READ_TOKEN}\n"
        self.assertEqual(
            self.verdict(True, out),
            (False, 1, f"[01] FAIL (read {stage_isolation.READ_TOKEN}): a check\n"),
        )

    def test_sudo_without_its_audit_record(self):
        out = "sudo: unable to send audit message: Operation not permitted\ndone\n"
        self.assertEqual(
            self.verdict(True, out),
            (False, 1, "[01] FAIL (sudo wrote no audit record): a check\n"),
        )


class TestTheIsolationStageReleases(unittest.TestCase):
    """The releases the isolation stage has root install (stage_isolation.py release_archive): the image's code as a
    package's tar archive, its bytecode left out and a link kept a link (root refuses one), and its writer's runner
    marked with the release's name, which it says when Python loads it."""

    def test_the_image_s_code_with_its_writer_s_runner_marked(self):
        runner = stage_isolation.WRITER_RUNNER
        with tempfile.TemporaryDirectory() as d:
            pkg = pathlib.Path(d, "rocke_monitor")
            (pkg / "rkmon_lib" / "__pycache__").mkdir(parents=True)
            (pkg / "rk").write_text("#!/bin/sh\n")
            (pkg / runner).write_text("def main(argv):\n    return 0\n")
            (pkg / "rkmon_lib" / "__pycache__" / "writer_run.pyc").write_bytes(b"x")
            (pkg / "rk2").symlink_to("rk")
            with mock.patch.object(stage_isolation, "PKG", str(pkg)):
                path = stage_isolation.release_archive("one", d)
            with tarfile.open(path) as tar:
                members = {m.name: m for m in tar.getmembers()}
                body = tar.extractfile(f"rocke_monitor/{runner}").read()
            loaded = pathlib.Path(d, "loaded.py")
            loaded.write_bytes(body)
            said = io.StringIO()
            with contextlib.redirect_stderr(said):
                scope = runpy.run_path(str(loaded))
        self.assertEqual(
            sorted(members),
            [
                "rocke_monitor",
                "rocke_monitor/rk",
                "rocke_monitor/rk2",
                "rocke_monitor/rkmon_lib",
                f"rocke_monitor/{runner}",
            ],
        )
        self.assertTrue(members["rocke_monitor/rk2"].issym())
        self.assertTrue(body.startswith(b"def main(argv):\n    return 0\n"), body)
        self.assertEqual(said.getvalue(), stage_isolation.MARK.format("one") + "\n")
        self.assertEqual(scope["main"]([]), 0)


class TestTheIsolationStageTries(unittest.TestCase):
    """The isolation stage's tries to change the code volume (stage_isolation.py code_tries), run as this user: on a
    tree of this user's each try works and the script names it; on names that are not there none does, and it says
    NOTHING CHANGED."""

    def tries(self, folder: str, file: str, link: str) -> str:
        script = stage_isolation.code_tries([folder], [file], [link])
        return subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, check=True
        ).stdout

    def test_on_a_tree_of_this_user_s_each_try_works(self):
        with tempfile.TemporaryDirectory() as d:
            folder, file, link = (os.path.join(d, n) for n in ("code", "rk", "current"))
            os.mkdir(folder)
            pathlib.Path(file).write_text("x\n")
            os.symlink(file, link)
            out = self.tries(folder, file, link)
        self.assertEqual(
            out,
            f"CHANGED as {pwd.getpwuid(os.geteuid()).pw_name}: made:{folder}/{stage_isolation.TRY} "
            f"renamed:{folder} wrote:{file} opened:{file} pointed:{link} removed:{link}\n",
        )

    def test_on_names_that_are_not_there_no_try_works(self):
        with tempfile.TemporaryDirectory() as d:
            gone = os.path.join(d, "gone")
            out = self.tries(
                gone, os.path.join(gone, "rk"), os.path.join(gone, "current")
            )
        self.assertEqual(
            out, f"NOTHING CHANGED as {pwd.getpwuid(os.geteuid()).pw_name}\n"
        )


class TestTheSeveralDeploymentsStageReads(unittest.TestCase):
    """How the several-deployments stage reads a deployment's docker calls, the docker guard's log lines
    (stage_several_deployments.py others_findings): a call that names another deployment's object, or runs in or names
    its folder, is a finding; an image is the host's. c's container name is a's: until a's container starts, a's
    compose project is c's as much as a's; from then on, c names it only exactly, and only in a listing (LISTINGS).
    And how it reads a deployment's folder reads, the folder reads hook's log lines (read_findings): each file opened
    or folder listed in another deployment's folder is a finding, with the rkmon command it ran in, and so is a log
    with fewer starts of ./rkmon than the deployment ran rkmon commands.
    """

    SCOPE = "l2test-unit-several-deployments-"
    A, B, C = SCOPE + "a", SCOPE + "b", SCOPE + "c"
    MARKS = [(100.0, "./rkmon check"), (200.0, "./rkmon start")]

    @staticmethod
    def call(command, argv, names=(), t=150.0, cwd="/tmp/own-top/package/docker"):
        """A line of the guard's log (tests/fakes/docker_guard): COMMAND as it reads ARGV, and what ARGV names."""
        verdict = "narrowed" if command in ("ps", "volume ls", "compose ls") else "pass"
        return {
            "t": t,
            "argv": list(argv),
            "cwd": cwd,
            "command": command,
            "verdict": verdict,
            "names": [list(n) for n in names],
        }

    def findings(self, calls, own=None, ids=None, until=math.inf) -> list:
        """others_findings of CALLS, a deployment's (OWN: a's by default) with b the other (or a and b, for c)."""
        own = own or (self.A, self.A)
        others = {s: f"/tmp/{s[-1]}-top" for s in (self.A, self.B) if s != own[0]}
        return stage_several_deployments.others_findings(
            calls, self.MARKS, own, others, ids or {}, until
        )

    def test_its_own_objects_and_the_engine_s_questions_are_no_finding(self):
        A = self.A
        calls = [
            self.call(
                "ps", ["ps", "-a", "--filter", f"name=^{A}$"], [("name", f"^{A}$")]
            ),
            self.call("compose ps", ["compose", "ps"], [("project", A)]),
            self.call(
                "volume inspect",
                ["volume", "inspect", f"{A}_rkmon-work"],
                [("volume", f"{A}_rkmon-work")],
            ),
            self.call("version", ["version"]),
        ]
        self.assertEqual(self.findings(calls), [])

    def test_another_s_volume_compose_project_name_or_folder(self):
        A, B = self.A, self.B
        for c in (
            self.call(
                "volume inspect",
                ["volume", "inspect", f"{B}_rkmon-work"],
                [("volume", f"{B}_rkmon-work")],
            ),
            self.call("compose ps", ["compose", "ps"], [("project", B)]),
            self.call(
                "ps", ["ps", "-a", "--filter", f"name=^{B}"], [("name", f"^{B}")]
            ),
            self.call(
                "compose config", ["compose", "config"], cwd="/tmp/b-top/package/docker"
            ),
            self.call(
                "cp",
                ["cp", "/tmp/b-top/package/x", f"{A}-rkmon-1:/x"],
                [("container", f"{A}-rkmon-1")],
            ),
        ):
            with self.subTest(argv=c["argv"]):
                found = self.findings([c])
                self.assertEqual(len(found), 1, found)
                self.assertTrue(
                    found[0].startswith("during ./rkmon check: docker "), found
                )

    def test_a_container_by_its_id(self):
        full = "f" * 64
        c = self.call("inspect", ["inspect", full[:12]], [("container", full[:12])])
        self.assertEqual(len(self.findings([c], ids={full: f"{self.B}-rkmon-1"})), 1)
        self.assertEqual(self.findings([c], ids={full: f"{self.A}-rkmon-1"}), [])

    def test_an_image_is_the_host_s(self):
        image = f"rocke-monitor:{self.B}"
        c = self.call("image inspect", ["image", "inspect", image], [("image", image)])
        self.assertEqual(self.findings([c]), [])

    def test_once_a_runs_c_names_a_s_compose_project_only_in_a_listing(self):
        A, own = self.A, (self.C, self.A)
        exact = identity.exact_name(A)
        commands = [
            self.call("compose stop", ["compose", "-p", A, "stop"], [("project", A)]),
            self.call(
                "compose exec",
                ["compose", "-p", A, "exec", "rkmon", "ls", "/secrets"],
                [("project", A)],
            ),
            self.call("compose up", ["compose", "-p", A, "up", "-d"], [("project", A)]),
            self.call(
                "volume inspect",
                ["volume", "inspect", f"{A}_rkmon-secrets"],
                [("volume", f"{A}_rkmon-secrets")],
            ),
        ]
        listings = [
            self.call(
                "ps",
                ["ps", "--all", "--filter", f"label={identity.PROJECT}={A}"],
                [("project", A)],
            ),
            self.call("ps", ["ps", "--filter", f"name={exact}"], [("name", exact)]),
            self.call("compose ps", ["compose", "-p", A, "ps"], [("project", A)]),
            self.call(
                "compose ls",
                ["compose", "ls", "--filter", f"name={exact}"],
                [("name", exact)],
            ),
            self.call(
                "compose config",
                ["compose", "-p", A, "config", "--format", "json"],
                [("project", A)],
            ),
        ]
        # a's container started at 120, before each call (at 150)
        self.assertEqual(
            self.findings(commands + listings, own, until=120.0),
            [
                f"during ./rkmon check: docker compose -p {A} stop: the project {A}",
                f"during ./rkmon check: docker compose -p {A} exec rkmon ls /secrets: the project {A}",
                f"during ./rkmon check: docker compose -p {A} up -d: the project {A}",
                f"during ./rkmon check: docker volume inspect {A}_rkmon-secrets: the volume {A}_rkmon-secrets",
            ],
        )
        # before it starts, the objects of a's compose project are c's as much as a's
        self.assertEqual(self.findings(commands + listings, own), [])
        # a name filter that may match a part of a's names, in a listing or not
        loose = self.call("ps", ["ps", "--filter", f"name={A}"], [("name", A)])
        self.assertEqual(len(self.findings([loose], own, until=120.0)), 1)

    def test_a_read_in_another_s_folder_and_a_command_the_hook_did_not_see(self):
        folders = {"/tmp/b-top/": "b"}
        started = [
            {"t": 101.0, "pid": 1, "started": ["/usr/bin/python3", "./rkmon", "check"]},
            {"t": 201.0, "pid": 2, "started": ["/usr/bin/python3", "./rkmon", "start"]},
        ]
        reads = stage_several_deployments.read_findings
        self.assertEqual(reads(started, self.MARKS, folders), [])
        read = {
            "t": 202.0,
            "pid": 2,
            "event": "open",
            "path": "/tmp/b-top/settings/rkmon-deploy.yaml",
        }
        self.assertEqual(
            reads([*started, read], self.MARKS, folders),
            [
                "during ./rkmon start: open /tmp/b-top/settings/rkmon-deploy.yaml, in deployment b's folder"
            ],
        )
        found = reads(started[:1], self.MARKS, folders)
        self.assertEqual(len(found), 1, found)
        self.assertIn("1 starts of ./rkmon, for 2 commands", found[0])


class TestTheTeamPrototypeStageAsks(unittest.TestCase):
    """Where the team-prototype stage asks for the team dashboard's page (stage_team_prototype.py served): in the
    container, at the container's own address, where docker delivers what reaches the port it publishes, and at the
    team port; never at 127.0.0.1, where a page served on 127.0.0.1 alone would answer while the published port reached
    nothing. Its GET asks the address and the port it is given.
    """

    def test_the_page_is_asked_at_the_container_s_own_address(self):
        results, asked = [], []
        res = mock.Mock()
        res.result.side_effect = lambda status, text: results.append((status, text))
        t = stage_team_prototype.TeamPrototype(
            "docker", "rocke-monitor:l2test-unit", str(ROOT), "l2test-unit", [], res
        )
        self.addCleanup(t.top.rmdir)  # all a run makes before it starts
        with mock.patch.object(t, "own_address", return_value="172.30.0.7"):
            with mock.patch.object(
                t, "as_monitor", side_effect=lambda *a: asked.append(a) or "200\n"
            ):
                self.assertIs(t.served(), True)
        port = str(container_layout.TEAM_PORT)
        self.assertEqual(
            asked,
            [("python3", "-I", "-c", stage_team_prototype.GET, "172.30.0.7", port)],
        )
        self.assertEqual(results[0][0], "PASS")
        self.assertIn("at the container's own address", results[0][1])

    class Page(http.server.BaseHTTPRequestHandler):
        """A page that answers 200, empty, and logs nothing."""

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    def test_the_get_asks_the_address_and_the_port_it_is_given(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.Page)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        port = str(server.server_address[1])
        with tempfile.TemporaryDirectory() as tmp:
            for address, said in (("127.0.0.1", "200"), ("127.0.0.3", "")):
                with self.subTest(address=address):
                    p = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            "-c",
                            stage_team_prototype.GET,
                            address,
                            port,
                        ],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(p.stdout.strip(), said)
                    self.assertEqual(p.returncode != 0, not said)


class TestTheAdoptStagesCopies(unittest.TestCase):
    """The earlier monitor's copies the adopt stage makes (stage_adopt.py earlier_work): each one adopt stages as it
    stages a copy of a monitor's /work (rkmon_lib/adopt.py stage), the package's synthetic production configuration
    with the copy's own value, so that the two differ, which the move reads as config.json, of the bytes whose sha256
    earlier_work gives; its PR records, the work in flight cleared; and its secret's file and its lock left behind.
    """

    def test_adopt_stages_each_copy(self):
        production = json.loads(
            ROOT.joinpath(*stage_adopt.PRODUCTION).read_text(encoding="utf-8")
        )
        shas = set()
        with tempfile.TemporaryDirectory() as tmp:
            top = pathlib.Path(tmp)
            for n, earlier in enumerate(stage_adopt.COPIES):
                with self.subTest(copy=n):
                    folder = top / f"earlier-{n}"
                    sha = stage_adopt.earlier_work(folder, str(ROOT), n)
                    shas.add(sha)
                    older = one_time_move.older_settings(folder / "monitor")
                    self.assertEqual((older.problem, older.sha256), ("", sha))
                    config = folder / "monitor" / "config.json"
                    values = json.loads(config.read_text(encoding="utf-8"))
                    self.assertEqual(
                        values, {**production, "drift_batch": earlier["drift_batch"]}
                    )
                    staged = top / f"staged-{n}"
                    manifest = adopt.stage(adopt.old_folders(folder, values), staged)
                    self.assertEqual(manifest["records"]["prs"], len(earlier["prs"]))
                    self.assertEqual(list(manifest["cleared"]), ["review_ready"])
                    state = folder / "monitor" / "state"
                    self.assertEqual(
                        sorted(manifest["left"]),
                        sorted(
                            [str(state / stage_adopt.SECRET[0]), str(state / "lock")]
                        ),
                    )
                    self.assertTrue(adopt.staged_config(staged).is_file())
                    for part in ("lessons", "posts", "posted", "daily", "spend"):
                        self.assertEqual(manifest["counts"][part], 1, part)
        self.assertEqual(len(shas), len(stage_adopt.COPIES))


class TestHowTheAdoptStageReadsAMoveAndItsEnd(unittest.TestCase):
    """How the adopt stage reads the monitor's home (stage_adopt.py FACTS). moved_findings: none for the move the start
    after an adopt makes, after the time given, of the copy's config.json; one for each way the home differs from that,
    and with no record, that the move's file and time match none either. held_findings: none for a send standby held,
    kept as an unsent item; one for each way it was not. released_findings: none once standby off has ended the move;
    one for each way the home differs from that.
    """

    MOVED_AT, ENDED_AT = "2026-11-22T09:00:00Z", "2026-11-22T10:00:00Z"

    def moved(self, **changed) -> dict:
        """The monitor's home as the start after an adopt leaves it, but for CHANGED."""
        return {
            "standby": True,
            "reason": adopt.REASON,
            "until": self.MOVED_AT,
            "adopted_at": "2026-11-22T08:59:00Z",
            "record": {
                "moved_at": self.MOVED_AT,
                "ended_at": "",
                "config_json_sha256": "s1",
            },
            "config_json": "s1",
            "migrated": "",
            "migrated_mode": None,
            "mark": {"kind": adopt.MARK_KIND, "applied_at": "2026-11-22T08:59:30Z"},
            "prs": ["101", "202"],
            "state_folder": ["internal_state.json", "one_time_move.json", "state.json"],
            "moved": 1,
            **changed,
        }

    def released(self, **changed) -> dict:
        """The monitor's home once standby off has ended the move, but for CHANGED."""
        record = {**self.moved()["record"], "ended_at": self.ENDED_AT}
        ended = {"standby": False, "until": self.ENDED_AT, "record": record}
        ended.update(config_json="", migrated="s1", migrated_mode=0o440, mark=None)
        return self.moved(**{**ended, **changed})

    def test_a_move_after_an_adopt(self):
        moved_findings = stage_adopt.moved_findings
        self.assertEqual(moved_findings(self.moved(), "s1", ["202", "101"]), [])
        before = "2026-11-22T08:00:00Z"
        self.assertEqual(moved_findings(self.moved(), "s1", ["101", "202"], before), [])

    def test_each_way_a_move_differs(self):
        record = self.moved()["record"]
        cases = {
            "no record": (self.moved(record=None), "no open record"),
            "an ended move": (
                self.moved(record={**record, "ended_at": self.ENDED_AT}),
                "no open record",
            ),
            "a move made before": (self.moved(), "no open record of a move made after"),
            "another config.json": (
                self.moved(config_json="s2"),
                "not of the copy's config.json",
            ),
            "until at another time": (
                self.moved(until=self.ENDED_AT),
                "reviews.unnamed_are_ours_until is",
            ),
            "no standby": (self.moved(standby=False), "not on for the adoption"),
            "the move's standby": (
                self.moved(reason=one_time_move.REASON),
                "not on for the adoption",
            ),
            "an adopt cut short": (
                self.moved(mark={"kind": adopt.MARK_KIND, "applied_at": ""}),
                "not recorded as finished",
            ),
            "other records": (self.moved(prs=["101"]), "the PR records are"),
            "the secret came along": (
                self.moved(state_folder=[stage_adopt.SECRET[0], "state.json"]),
                "came along",
            ),
        }
        for name, (facts, said) in cases.items():
            with self.subTest(case=name):
                before = self.MOVED_AT if name == "a move made before" else ""
                found = stage_adopt.moved_findings(facts, "s1", ["101", "202"], before)
                self.assertEqual(len(found), 3 if name == "no record" else 1, found)
                self.assertIn(said, found[0])

    def test_a_dry_run_that_changed_nothing(self):
        copy = "/tmp/scope-adopt-x/earlier-1"
        out = f"PASSED: the plan can be applied\nnext: ./rkmon adopt --from {copy}"
        held = ("next: ./rkmon start", ["volume v", "work f 2 1.5 /work/monitor/a"])
        self.assertEqual(stage_adopt.dry_run_findings(0, out, copy, held, held), [])

    def test_each_way_a_dry_run_differs(self):
        copy = "/tmp/scope-adopt-x/earlier-1"
        out = f"PASSED: the plan can be applied\nnext: ./rkmon adopt --from {copy}"
        said, held = "next: ./rkmon start", ["volume v", "work f 2 1.5 /work/monitor/a"]
        cases = {
            "exit 1": (1, out, said, held, ["exit 1"]),
            "no PASSED": (
                0,
                out.replace("PASSED", "ADOPTED"),
                said,
                held,
                ["outcome ''"],
            ),
            "the next line keeps --dry-run": (
                0,
                out + " --dry-run",
                said,
                held,
                ["its next line is"],
            ),
            "rkmon next says otherwise": (
                0,
                out,
                "next: ./rkmon x",
                held,
                ["says otherwise"],
            ),
            "a file written": (
                0,
                out,
                said,
                [*held, "work f 3 2.5 /work/adopt-backup-1/b"],
                ["new: work f 3 2.5 /work/adopt-backup-1/b"],
            ),
            "a file changed": (
                0,
                out,
                said,
                ["volume v", "work f 4 2.5 /work/monitor/a"],
                ["gone: work f 2 1.5", "new: work f 4 2.5"],
            ),
        }
        for name, (rc, text, after, files, wanted) in cases.items():
            with self.subTest(case=name):
                found = stage_adopt.dry_run_findings(
                    rc, text, copy, (said, held), (after, files)
                )
                self.assertEqual(len(found), len(wanted), found)
                for x, want in zip(found, wanted):
                    self.assertIn(want, x)

    NOTICE = "Automated overlap notice by scope-adopt\n\nAn invented overlap notice.\n"
    HELD = (
        f"UNSENT: an overlap notice for pull request {stage_adopt.NOTICE_PR}, kept as an unsent item: the monitor is "
        "in standby: nothing goes out until standby ends (./rkmon standby off)"
    )
    KEPT = {
        "type": "comment",
        "for": f"pull request {stage_adopt.NOTICE_PR}",
        "why": "standby",
        "content": NOTICE,
    }

    def test_a_send_standby_held(self):
        found = stage_adopt.held_findings(0, self.HELD + "\n", [self.KEPT], self.NOTICE)
        self.assertEqual(found, [])

    def test_each_way_a_send_was_not_held(self):
        url = f"https://{fake_net.WEB}/acme/widgets/pull/{stage_adopt.NOTICE_PR}"
        posted = f"posted: {url}#issuecomment-1"
        cases = {
            "sent": (0, posted, [], ["its last line 'posted:", "no unsent item"]),
            "refused": (
                4,
                "REFUSED: an overlap notice is not sent",
                [],
                ["exit 4", "no unsent"],
            ),
            "held, nothing kept": (0, self.HELD, [], ["no unsent item"]),
            "kept for prototype": (
                0,
                self.HELD,
                [{**self.KEPT, "why": "prototype"}],
                ["no unsent"],
            ),
            "another text kept": (
                0,
                self.HELD,
                [{**self.KEPT, "content": "x"}],
                ["no unsent"],
            ),
        }
        for name, (rc, out, kept, wanted) in cases.items():
            with self.subTest(case=name):
                found = stage_adopt.held_findings(rc, out, kept, self.NOTICE)
                self.assertEqual(len(found), len(wanted), found)
                for x, want in zip(found, wanted):
                    self.assertIn(want, x)

    def test_the_end(self):
        self.assertEqual(stage_adopt.released_findings(self.released(), "s1"), [])

    def test_each_way_the_end_differs(self):
        cases = {
            "config.json still there": (
                self.released(config_json="s1"),
                "config.json is still there",
            ),
            "another file renamed": (
                self.released(migrated="s2"),
                "config.json.migrated is not",
            ),
            "config.json.migrated writable": (
                self.released(migrated_mode=0o640),
                "not read-only",
            ),
            "the move not ended": (
                self.released(record=self.moved()["record"]),
                "the move's record ended at ''",
            ),
            "until at another time": (self.released(until=self.MOVED_AT), "until is"),
            "standby on": (self.released(standby=True), "standby True"),
            "the mark left": (
                self.released(
                    mark={"kind": adopt.MARK_KIND, "applied_at": self.MOVED_AT}
                ),
                "the mark",
            ),
        }
        for name, (facts, said) in cases.items():
            with self.subTest(case=name):
                found = stage_adopt.released_findings(facts, "s1")
                self.assertEqual(len(found), 1, found)
                self.assertIn(said, found[0])


class TestHowTheOneTimeMoveStageReadsTheMoveAndItsEnd(unittest.TestCase):
    """How the one-time-move stage reads the container (stage_one_time_move.py FACTS and FILES). moved_findings: none
    for the move as the first start makes it on the older install the stage makes (older_files), with its report in the
    start's log; ended_findings: none for its end, as standby off makes it. For each way the container differs from
    those, a finding that says what differs.
    """

    KEY = "sk-invented-0123"
    MOVED_AT, ENDED_AT = "2026-11-22T09:00:00Z", "2026-11-22T10:00:00Z"

    def setUp(self):
        m = stage_one_time_move
        self.made = m.older_files(str(ROOT), self.KEY)
        # the start's log: the move's report (MOVE_REPORT), as the first start writes it
        self.log = [
            f"moved: {m.CONFIG_JSON}, an older install's configuration, split by owner, once; the monitor is in "
            "standby: it sends nothing until you run ./rkmon standby off",
            f"carried into the internal state: {m.CARRIED}, codeowner_teams",
            "repo: config.json's value is not carried; your deploy settings' repository.name runs",
            f"drift_batch: config.json's {m.OLDER['drift_batch']} is not carried; the package's 6 runs",
            "dropped (no key of this package has the name): no_such_key",
            f"{m.UNTIL} is {self.MOVED_AT}: the reviews the older monitor posted before then, whose line 1 names no "
            "automated reviewer, are this deployment's own; ./rkmon standby off sets it again, to its own time",
        ]
        self.facts = {
            "older_install": True,
            "standby": True,
            "reason": one_time_move.REASON,
            "until": self.MOVED_AT,
            "carried": m.OLDER[m.CARRIED],
            "record": {
                "moved_at": self.MOVED_AT,
                "ended_at": "",
                "config_json_sha256": m.sha256(self.made[m.CONFIG_JSON]),
                "report": [
                    f"repo: config.json's {json.dumps(m.OLDER['repo'])} is not carried"
                ],
            },
            "log": self.log,
        }
        self.files = {
            path: {"owner": "rkmon", "mode": 0o600, "sha256": m.sha256(text)}
            for path, text in self.made.items()
        }
        self.files[m.MIGRATED] = None
        for path, (user, sha) in m.copied(self.made, self.KEY).items():
            self.files[path] = {"owner": user, "mode": 0o600, "sha256": sha}

    def moved(self, facts=None, files=None) -> list:
        """moved_findings of the move as the first start makes it, but for FACTS and FILES."""
        return stage_one_time_move.moved_findings(
            {**self.facts, **(facts or {})},
            {**self.files, **(files or {})},
            self.made,
            self.KEY,
        )

    def ended(self, facts=None, files=None) -> list:
        """ended_findings of the end as standby off makes it, but for FACTS and FILES."""
        m = stage_one_time_move
        record = {**self.facts["record"], "ended_at": self.ENDED_AT}
        end = {**self.facts, "standby": False, "until": self.ENDED_AT, "record": record}
        migrated = {
            "owner": "rkmon",
            "mode": 0o400,
            "sha256": self.files[m.CONFIG_JSON]["sha256"],
        }
        end_files = {**self.files, m.MIGRATED: migrated, m.CONFIG_JSON: None}
        end_files.update(dict.fromkeys([*m.COPIES, *m.UNCLAIMED]))
        return m.ended_findings(
            {**end, **(facts or {})},
            {**end_files, **(files or {})},
            self.made,
            self.KEY,
        )

    def test_the_move_and_its_end(self):
        self.assertEqual(self.moved(), [])
        self.assertEqual(self.ended(), [])

    def test_the_ends_of_rk_start(self):
        # each rk start ends with its line on the Claude side (rkmon_lib/control.py start), after the move's report
        m = stage_one_time_move
        ended = f"{control.CLAUDE_SIDE}the container's orchestrator (rk orchestrate) sends /rocke-monitor resume"
        self.assertEqual(m.starts_ended(self.facts), 0)
        self.assertEqual(m.starts_ended({"log": [*self.log, ended]}), 1)
        self.assertEqual(
            m.starts_ended({"log": [*self.log, ended, "a line", ended]}), 2
        )
        self.assertEqual(m.starts_ended({}), 0)

    def test_the_claude_login_the_move_copied_works_when_its_test_call_passes(self):
        """login_findings: none for rk claude-test's exit 0 and the claude login's row PASS, as the command prints it;
        a finding for a FAIL (exit 4), for a WARN (an answer that does not confirm the key), for another exit, and
        for no row of the claude login."""
        at = "at https://llm-gateway.example.com"
        fix = checks.Steps((checks.Command(("./rkmon", "login", "claude")),))

        def row(status: str, found: str, name: str = "claude") -> str:
            fixed = fix if status != "PASS" else None
            return checks.render([checks.Check(name, status, found, fixed)])

        passed = row("PASS", f"one test call answered, {at} (model claude-opus-5-5)")
        self.assertEqual(stage_one_time_move.login_findings(0, passed), [])
        for rc, out in (
            (4, row("FAIL", f"the key is refused, {at} (HTTP 401)")),
            (0, row("WARN", f"the test call could not be answered, {at} (HTTP 429)")),
            (1, passed),
            (0, ""),
            (0, row("PASS", "on", name="claude session")),
        ):
            with self.subTest(rc=rc, out=out):
                found = stage_one_time_move.login_findings(rc, out)
                self.assertEqual(len(found), 1, found)
                self.assertTrue(found[0].startswith(f"rk claude-test: exit {rc}, "))

    def test_each_way_the_move_differs(self):
        m, files = stage_one_time_move, self.files
        webhook, key = f"{m.WRITE_SECRETS}/teams_webhook", f"{m.SECRETS}/claude_key"
        whole = m.sha256(self.made[f"{m.SECRETS}/claude_env"])
        cases = {
            "standby off": ({"standby": False}, {}, "not on for the move"),
            "the adoption's standby": (
                {"reason": adopt.REASON},
                {},
                "not on for the move",
            ),
            "until at another time": (
                {"until": self.ENDED_AT},
                {},
                "not the move's time",
            ),
            "the state key not carried": ({"carried": None}, {}, "not config.json's"),
            "not an older install's": (
                {"older_install": None},
                {},
                "an older install's",
            ),
            "no record": ({"record": {}}, {}, "no open record of a move"),
            "the private value shown": (
                {"log": [*self.log, f"repo {m.OLDER['repo']}"]},
                {},
                "a private key's value",
            ),
            "the private value not kept": (
                {"record": {**self.facts["record"], "report": []}},
                {},
                "does not keep",
            ),
            "two reports": ({"log": [*self.log, self.log[0]]}, {}, "reports 2 moves"),
            "no line of the name dropped": (
                {"log": self.log[:4] + self.log[5:]},
                {},
                "has no line",
            ),
            "config.json changed": (
                {},
                {m.CONFIG_JSON: {**files[m.CONFIG_JSON], "sha256": "x"}},
                "not as the older install left it",
            ),
            "an older copy gone": (
                {},
                {f"{m.SECRETS}/teams_webhook": None},
                "not as the older install left it",
            ),
            "a copy not its keeper's": (
                {},
                {webhook: {**files[webhook], "owner": "rkmon"}},
                "not the copy the move makes",
            ),
            "a copy of mode 640": (
                {},
                {key: {**files[key], "mode": 0o640}},
                "not the copy the move makes",
            ),
            "the claude login's whole file copied": (
                {},
                {key: {**files[key], "sha256": whole}},
                "not the copy the move makes",
            ),
            "config.json.migrated already": (
                {},
                {m.MIGRATED: {"owner": "rkmon", "mode": 0o400, "sha256": "y"}},
                "there before the end",
            ),
        }
        for name, (facts, changed, said) in cases.items():
            with self.subTest(case=name):
                found = self.moved(facts, changed)
                self.assertTrue(any(said in x for x in found), found)

    def test_each_way_the_end_differs(self):
        m, files = stage_one_time_move, self.files
        writable = {**files[m.CONFIG_JSON], "mode": 0o600}
        cases = {
            "config.json still there": (
                {},
                {m.CONFIG_JSON: files[m.CONFIG_JSON]},
                "config.json is still there",
            ),
            "config.json not renamed": (
                {},
                {m.MIGRATED: None},
                "not config.json's bytes",
            ),
            "config.json.migrated writable": (
                {},
                {m.MIGRATED: writable},
                "not read-only",
            ),
            "an older copy left": (
                {},
                {f"{m.SECRETS}/claude_env": files[f"{m.SECRETS}/claude_env"]},
                "the older copy",
            ),
            "a file no entry claims left": (
                {},
                {m.UNCLAIMED[0]: files[m.UNCLAIMED[0]]},
                "the older copy",
            ),
            "a new copy gone": ({}, {f"{m.SECRETS}/claude_key": None}, "the new copy"),
            "standby on": ({"standby": True}, {}, "standby is True"),
            "until at the move's time": (
                {"until": self.MOVED_AT},
                {},
                "the move's record ended at",
            ),
        }
        for name, (facts, changed, said) in cases.items():
            with self.subTest(case=name):
                found = self.ended(facts, changed)
                self.assertTrue(any(said in x for x in found), found)


if __name__ == "__main__":
    unittest.main()
