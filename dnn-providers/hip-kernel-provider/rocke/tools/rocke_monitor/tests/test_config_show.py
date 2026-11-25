# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk config show (rkmon_lib/config_show.py): the running configuration, each value under the name of the
file that holds it and, with --source, the layer it came from, deploy only for the keys your deploy settings file sets;
never a secret, and no file written; a configuration that cannot run is shown with its problems, and one that cannot
be loaded is refused with its fix.
"""

import contextlib
import copy
import dataclasses
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    applied,
    checks,
    config,
    config_show,
    container_layout,
    deploy_plan,
    derived,
    doctor,
    file_types,
    image,
    paths,
    registry,
    runtime_config,
    settings_file,
)

RC = runtime_config
# A first setup's answers, which config.json holds besides the defaults: a configuration that runs.
ANSWERS = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "operator": {"github_login": "<user_login>"},
    "scope": {"labels": ["team: widgets"]},
}


def found(resolved: RC.Resolved) -> dict:
    return {name: (value, source) for name, value, source in config_show.rows(resolved)}


class TestRows(unittest.TestCase):
    def test_each_value_is_named_as_its_file_names_it_with_its_layer(self):
        carried = {"repo": "acme/widgets", "loop_heartbeat_min": 11}
        r = RC.resolve(
            registry.package_defaults(),
            {"tick_timeout_min": 7},
            None,
            {"base_branch": "main"},
            derived.Layout(given=RC.given(carried)),
            carried,
        )
        rows = found(r)
        self.assertEqual(rows["repo"], ("acme/widgets", RC.CONFIG_JSON))
        self.assertEqual(rows["loop_heartbeat_min"], (11, RC.CONFIG_JSON))
        self.assertEqual(rows["tick_timeout_min"], (7, RC.REPOSITORY))
        self.assertEqual(
            rows["huge_pr_lines"],
            (registry.default("huge_pr_lines"), RC.PACKAGE_SOURCE),
        )
        self.assertEqual(  # the internal state file's name for it
            rows["repository.base_branch"], ("main", RC.STATE)
        )
        self.assertNotIn("base_branch", rows)
        self.assertEqual(rows["codeowner_org"], ("acme", RC.DERIVED))
        self.assertEqual(len(rows), len(r.source))
        self.assertIn("loop_heartbeat_min = 11", config_show.lines(r))
        width = max(len(s) for s in RC.SOURCES if s in r.source.values())
        self.assertIn(
            f"{RC.REPOSITORY:<{width}}  tick_timeout_min = 7",
            config_show.lines(r, source=True),
        )

    def test_a_delivered_layer_s_values_are_named_by_the_deploy_settings_file(self):
        settings = {
            "repository": {"name": "acme/widgets", "folder": "a/b/rocke"},
            "host": {"resources": {"cpus": 4}},
        }
        set_in_file = ["repository.name", "repository.folder"]
        r = RC.from_settings(settings)._replace(
            layer={"settings": settings, "set_in_file": set_in_file}
        )
        rows = found(r)
        self.assertEqual(rows["repository.name"], ("acme/widgets", RC.DEPLOY))
        self.assertEqual(rows["repository.folder"], ("a/b/rocke", RC.DEPLOY))
        # a value of the layer your file does not set: the package's default
        self.assertEqual(rows["host.resources.cpus"], (4, RC.PACKAGE_SOURCE))
        for internal in ("repo", "rocke_path"):  # the code's names never show
            self.assertNotIn(internal, rows)
        self.assertEqual(rows["codeowner_org"], ("acme", RC.DERIVED))

    def test_a_map_of_the_deploy_settings_file_is_one_value(self):
        """A map's entries are its value, not keys of their own: a label's name with a colon or a dot stays inside
        its map (settings_file.map_keys, registry.leaves)."""
        folders = {"team: widgets": "a/b", "v1.2": "c/d"}
        settings = {
            "repository": {
                "name": "acme/widgets",
                "folder": "a/b/rocke",
                "shared_label_folders": folders,
            }
        }
        set_in_file = [
            "repository.name",
            "repository.folder",
            "repository.shared_label_folders",
        ]
        r = RC.from_settings(settings)._replace(
            layer={"settings": settings, "set_in_file": set_in_file}
        )
        rows = found(r)
        self.assertEqual(rows["repository.shared_label_folders"], (folders, RC.DEPLOY))
        self.assertEqual(
            [n for n in rows if n.startswith("repository.shared_label_folders.")], []
        )
        self.assertIn(
            f"repository.shared_label_folders = {json.dumps(folders)}",
            config_show.lines(r),
        )

    def test_the_last_probe_s_record_shows_without_how_it_sent_the_key(self):
        """worker.probe shows as the internal state holds it but for key_sent (config_show.LEFT_OUT), a digest of the
        headers that carry the reviewer model's key, which stays in the internal state.
        """
        carried = {"repo": "acme/widgets", "rocke_path": "a/b/rocke"}
        probe = {
            "at": "2026-01-01T00:00:00Z",
            "url": "http://127.0.0.1:11434",
            "model": "m",
            "backend": "ollama",
            "key_sent": "0123456789abcdef",
            "chat": "ok",
            "tools": "ok",
        }
        r = RC.resolve(
            registry.package_defaults(),
            {},
            None,
            {"worker": {"probe": probe}},
            derived.Layout(given=RC.given(carried)),
            carried,
        )
        shown = config.safe_value({k: v for k, v in probe.items() if k != "key_sent"})
        self.assertEqual(found(r)["model_bindings.worker.probe"], (shown, RC.STATE))
        self.assertNotIn(
            probe["key_sent"], "\n".join(config_show.lines(r, source=True))
        )


class TestShow(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = self.dir / "home"
        self.home.mkdir()
        for patch in (
            mock.patch.object(doctor, "deployment_image", return_value=False),
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def config_json(self, data: dict) -> None:
        (self.home / "config.json").write_text(json.dumps(data), encoding="utf-8")

    def show(self, *args: str) -> tuple:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(["config", "show", *args])
        return code, out.getvalue()

    def test_it_never_shows_a_secret_and_writes_no_file(self):
        secrets = self.dir / "secrets"
        secrets.mkdir(mode=0o700)
        files = {
            "worker_api_key": "invented-key-4711",
            "admin_access_token": "invented-token-4712",
            "teams_webhook": "https://example.com/hook/invented-hook-4713",
        }
        for name, value in files.items():
            (secrets / name).write_text(value + "\n", encoding="utf-8")
            os.chmod(secrets / name, 0o600)
        token = (
            "gh" + "p_" + "Q7" * 18
        )  # a token's form, made here so that no file holds one
        self.config_json(
            dict(
                ANSWERS,
                paths={
                    "working_dir": str(self.dir / "work"),
                    "secrets_dir": str(secrets),
                },
                clone_url="https://someone:invented-password@example.com/acme/widgets.git?key=invented-query#top",
                review_checklist=token,
            )
        )
        before = sorted(self.dir.rglob("*"))
        code, out = self.show("--source")
        self.assertEqual(code, 0, out)
        for secret in (*files.values(), "invented-password", "invented-query", token):
            self.assertNotIn(secret, out)
        self.assertIn('clone_url = "https://example.com/acme/widgets.git"', out)
        self.assertIn(f'review_checklist = "{config.HIDDEN}"', out)
        self.assertIn(f'paths.secrets_dir = "{secrets}"', out)  # where the secrets are
        self.assertEqual(sorted(self.dir.rglob("*")), before)

    def test_the_keys_of_the_install_without_a_container_are_gone(self):
        """Neither the defaults nor the registry has the keys of the install without a container, and a config.json
        that still has them runs and shows none of them: they are not read."""
        defaults = json.loads(
            (ROOT / "config.defaults.json").read_text(encoding="utf-8")
        )
        for name in (
            "wsl_distro",
            "services",
            "services.enabled",
            "services.claude_timer",
        ):
            with self.subTest(name=name):
                self.assertIs(registry.get(defaults, name), registry.MISSING)
                self.assertIsNone(registry.kind(name))
        self.config_json(
            dict(
                ANSWERS,
                paths={"working_dir": str(self.dir / "work")},
                wsl_distro="Ubuntu",
                services={"enabled": True, "claude_timer": True},
            )
        )
        code, out = self.show("--source")
        self.assertEqual(code, 0, out)
        for word in ("wsl_distro", "services", "claude_timer"):
            self.assertNotIn(word, out)

    def test_a_configuration_that_cannot_run_is_shown_with_its_problems(self):
        self.config_json(ANSWERS)
        with mock.patch.object(config, "problems", return_value=["a problem"]):
            code, out = self.show()
        self.assertEqual(code, 1)
        self.assertIn('repo = "acme/widgets"\n', out)
        self.assertTrue(out.endswith("\n\nIt cannot run yet:\n  - a problem\n"), out)

    def test_no_configuration_to_read_is_refused(self):
        code, out = self.show()
        self.assertEqual(code, 4)
        self.assertTrue(out.startswith("REFUSED: no configuration at "), out)

    def test_a_configuration_that_cannot_be_loaded_is_refused_with_its_fix(self):
        """Each error of the load is refused as rk refuses it (applied.refuse): why, then the fix the error names; a
        layer that cannot run names rkmon apply or rkmon rollback."""
        self.config_json(ANSWERS)
        error = RC.LayerError(
            "your confirmed settings are of the package 1.0, and this code is 2.0"
        )
        said = "".join(f"{line}\n" for line in applied.cannot_run(error, "REFUSED"))
        self.assertIn(f"\n{checks.FIX_LABEL}", said)
        with mock.patch.object(RC, "load", side_effect=error):
            self.assertEqual(self.show(), (4, said))


class TestDeploySource(unittest.TestCase):
    """`rk config show --source` in the container once apply has delivered a deploy settings file, its layer as
    deploy_plan.plan makes it: deploy for exactly the keys the file sets, package for each default it leaves
    out."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        home = self.dir / "home"
        home.mkdir()
        self.settings_dir = self.dir / "settings"
        self.settings_dir.mkdir()
        for patch in (  # the container's load, its settings folder the test's
            mock.patch.object(doctor, "deployment_image", return_value=True),
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.object(RC, "SETTINGS_DIR", str(self.settings_dir)),
            mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.template = settings_file.loads(
            settings_file.template(file_types.DEPLOY, "full"), file_types.DEPLOY
        )

    def shown(self, values: dict) -> dict:
        """What `rk config show --source` shows, {name: (value, source)}, once the deploy settings file of VALUES is
        delivered."""
        doc = dataclasses.replace(self.template, values=values)
        host = deploy_plan.Host(uid=1234, gid=1235, confirmed_at="2026-01-02T03:04:05Z")
        package = image.package_version(paths.PACKAGE)
        made = deploy_plan.plan(doc, package, image.code_sha256(paths.PACKAGE), host)
        (self.settings_dir / RC.LAYER_FILE).write_text(
            deploy_plan.layer_text(made.layer), encoding="utf-8"
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rkmon.main(["config", "show", "--source"])
        rows = (
            re.fullmatch(r"(\S+) +(\S+) = (.*)", line)
            for line in out.getvalue().splitlines()
        )
        return {m[2]: (json.loads(m[3]), m[1]) for m in rows if m}

    def test_an_off_feature_s_section_left_out_gets_no_deploy_row(self):
        """The file leaves out Ticket creation, a feature that is off, and has an empty atlassian_cli section: each
        of their keys shows at the package's default."""
        values = copy.deepcopy(self.template.values)
        del values["features"]["ticket_creation"]
        self.assertEqual(values["integrations"]["atlassian_cli"], {})
        shown = self.shown(values)
        for section in ("features.ticket_creation", "integrations.atlassian_cli"):
            under = {
                n: s for n, (_v, s) in shown.items() if n.startswith(f"{section}.")
            }
            self.assertTrue(under, section)
            self.assertEqual(set(under.values()), {RC.PACKAGE_SOURCE}, section)
        # the deploy rows are exactly the keys the file sets, at the file's values
        doc = dataclasses.replace(self.template, values=values)
        self.assertEqual(
            {n: v for n, (v, s) in shown.items() if s == RC.DEPLOY},
            {k: registry.get(values, k) for k in deploy_plan.set_in_file(doc)},
        )

    def test_a_file_that_sets_one_key_gets_exactly_one_deploy_row(self):
        shown = self.shown({"repository": {"name": "acme/widgets"}})
        self.assertEqual(
            {n: v for n, (v, s) in shown.items() if s == RC.DEPLOY},
            {"repository.name": "acme/widgets"},
        )
        self.assertEqual(shown["container.name"][1], RC.PACKAGE_SOURCE)

    def test_a_map_or_a_list_is_one_key_deploy_when_your_file_sets_it(self):
        """A map (model_bindings, repository.shared_label_folders) and a list (repository.scope_labels) are one key
        each: deploy at your file's value when it sets them, package at the default when it leaves them out.
        """
        mine = {
            "model_bindings": {
                "fast": {"provider": "claude", "model_id": "claude-haiku-5-5"}
            },
            "repository.shared_label_folders": {"team: widgets": "a/b"},
            "repository.scope_labels": ["area: widgets"],
        }
        values = {"repository": {"name": "acme/widgets"}}
        for name, value in mine.items():
            registry.put(values, name, copy.deepcopy(value))
        set_in = self.shown(values)
        left_out = self.shown({"repository": {"name": "acme/widgets"}})
        for name, value in mine.items():
            with self.subTest(name=name):
                self.assertEqual(set_in[name], (value, RC.DEPLOY))
                entries = [  # a map's entries are no deploy rows of their own
                    n
                    for n, (_v, s) in set_in.items()
                    if n.startswith(f"{name}.") and s == RC.DEPLOY
                ]
                self.assertEqual(entries, [])
                self.assertEqual(left_out[name][1], RC.PACKAGE_SOURCE)
                self.assertNotEqual(left_out[name][0], value)


if __name__ == "__main__":
    unittest.main()
