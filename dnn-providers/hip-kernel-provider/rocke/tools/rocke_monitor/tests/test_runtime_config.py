# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The running configuration (rkmon_lib/runtime_config.py): each key comes from its
owner's layer and says which, config.json being one more source below the delivered layer and the internal state; a
delivered layer runs when it was confirmed for this package and this layout of the deploy settings, and the monitor's
start records it as applied once, while a load records nothing and a configuration with a problem is never recorded; a
layer that cannot run stops the load and names its fix; each error of the load names its fix, and the doctor shows that
fix alone; a configuration that cannot be read starts no part of the monitor and no tick, and the doctor fails with its
fix, standby playing no part, while the emergency brake, which reads only what names the folders, stops the monitor all
the same; the layer is read in the container, where it is delivered, and the outward gate reads each feature's mode
from it; in a one-off container of the logins' walk (deploy step 5) the layer the root phase writes from your confirmed
settings runs in place of the delivered one, so that step 5's commands run with neither config.json nor a delivered
layer, and a folder or file of it that is not root's alone is refused in plain words; the host's view of a layer's
settings is what the container's load runs; config.load is the running configuration; a configuration file runs as
config.normalize reads it, but for the keys listed for it, each a difference the round trip's list of intended
differences says why of (tools/round_trip_differences.json): over the file a first setup writes, production's and an
older setup's; and the repository defaults for the configuration's repository and folder set their package keys,
where config.json holds none of its own.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

import rk_root  # noqa: E402  (docker/rk_root.py: the root phase's hand-layer)
from rkmon_lib import (  # noqa: E402
    applied,
    atlassian,
    bindings,
    checks,
    config,
    confirm,
    container_layout,
    control,
    deliver,
    deploy_tree,
    derived,
    doctor,
    draft_stage,
    drain,
    features,
    file_types,
    ghlogins,
    image,
    orchestrator,
    paths,
    post,
    registry,
    renames,
    repository_defaults,
    runtime_config,
    settings_file,
    slots,
    state_store,
    store,
    writer,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
import home_repository  # noqa: E402
import rkmon  # noqa: E402
import round_trip  # noqa: E402
from test_secrets_registry import a_value  # noqa: E402  (an invented value of a shape)

PACKAGE = paths.PACKAGE
perms = paths.worker_module("perms")
RC = runtime_config
# Whether an environment is the image's own (doctor.deployment_image), the real one: Base replaces it in each test.
DEPLOYMENT_IMAGE = doctor.deployment_image
# A first setup's answers: what config.json holds besides the defaults.
ANSWERS = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "operator.github_login": "<user_login>",
    "scope.labels": ["team: widgets"],
}
# The same answers in a delivered layer's settings: in the container, its configuration runs.
SETTINGS = {
    "repository": {
        "name": "acme/widgets",
        "folder": "a/b/rocke",
        "scope_labels": ["team: widgets"],
    },
    "integrations": {"github_cli": {"username": "<user_login>"}},
}


def set_in(settings) -> list:
    """The keys a test's layer lists as set in your file (its set_in_file): each key of its SETTINGS, which hold your
    file's own keys and no default; none for settings that are no mapping."""
    if not isinstance(settings, dict):
        return []
    return registry.leaves(settings, settings_file.map_keys(file_types.DEPLOY.load()))


# Where the running configuration of a configuration file differs from config.normalize's, today's reader: the keys
# of each file, each a difference the round trip's list of intended differences says why of (round_trip.DIFFERENCES).
# The dashboards' ports, derived: inside the container, where the file a first setup writes holds none.
PORTS = ("admin_dashboard.port", "team_dashboard.port")
# The derived keys whose rule their readers held: each file holds a blank, and the running configuration has the value
# its reader took for it.
FROM_THE_READER = (
    "clone_url",
    "followup_scan.dir",
    "worker.tag",
    "draft_stage",
    "verifier_name",
    "review_first_line_no_draft",
)
# The known_hosts files of the hosts ssh reaches, derived: in the working folder, where ssh's callers write the host's
# own keys the state records. Each file holds a blank.
HOST_KEYS = ("gpu.known_hosts", "publish.html.known_hosts")
HOST_KEYS_FILES = {
    "gpu.known_hosts": derived.GPU_KNOWN_HOSTS,
    "publish.html.known_hosts": derived.WEB_HOST_KNOWN_HOSTS,
}
# The local worker's keys no binding reads (rkmon_lib/removed_keys.py): its key's and its headers' variables, its
# headers' file, and where the setup interview found its model. Each file an earlier setup wrote holds them.
WORKER_KEYS = (
    "worker.api_key_env",
    "worker.header_env",
    "worker.headers_file",
    "worker.source",
    "worker.omp_provider",
)
# Keys no feature reads, and the package's own values (rkmon_lib/removed_keys.py): no key of the registry, so the
# running configuration holds none.
REMOVED = (
    *WORKER_KEYS,
    "admin_dashboard.publish_host",
    "admin_dashboard.token_file",
    "confluence.site",
    "confluence.space",
    "container.mounts",
    "container.pr_uid",
    "container.write_uid",
    "paths.rocm_reference_clang",
    "update.path",
    "update.test_timeout_min",
    "wsl_distro",
)
# The keys of the install without a container's services: no key of the registry either; an older
# setup's file holds them, and production's does not.
SERVICES = ("services.enabled", "services.claude_timer")
# The state keys: no file an older setup wrote holds them, and the running configuration has each at the internal
# state's default.
STATE_KEYS = (
    "standby.on",
    "standby.reason",
    "standby.since",
    "reviews.automated_reviewer_names_used",
    "reviews.unnamed_are_ours_until",
    "remote_gpu_tests.login_host_keys",
    "provisioned.at",
    "provisioned.package_version",
    "provisioned.older_install",
    "provisioned.adopted_at",
)
# The package key the registry adds: the variables each GPU job sets first, which the repository defaults set. A file
# without it runs with config.defaults.json's: no variables (its gpu.env keeps the variables it sets).
NEW_PACKAGE_KEYS = ("gpu.base_env",)
# The names the home repository's defaults give its develop issues: no configuration file holds them, and the running
# configuration of a file that names the home repository has the repository defaults' values.
REPOSITORY_NAMES = (
    "incident_issue.title_pattern",
    "incident_issue.test_name_pattern",
    "incident_issue.umbrella_test_names",
)
# The package keys of the sizing and the settle: no file an older setup wrote holds them, and the running
# configuration has each at its package default.
SIZING_AND_SETTLE = (
    "settle_burst_min",
    "settle_develop_fix_min",
    "resources.work_memory_gib",
    "ramdisk.review_estimate_gib",
    "ramdisk.build_estimate_gib",
)
# The remote GPU tests' job options the settings add: no file an older setup wrote holds them, and the running
# configuration has each at its default.
GPU_JOB_OPTIONS = ("gpu.qos", "gpu.gpus_per_job")
# The deploy keys no older file holds: the running configuration has each at its default.
NEW_DEPLOY_KEYS = (
    "features.merge_queue.drift_notices",
    "claude.api_key_sent_as",
    "claude.api_key_header",
    "claude.api_key_scheme",
    "gh.max_requests_per_hour",
    "features.review_automation.automated_reviewer_name",
)
# The modes of the features that send and had none: a file an older setup wrote holds none
# and has those features off, and the running configuration has each at its default, prototype, which is what
# features.mode takes without it.
MODES_THEY_LACK = (
    "features.alerts.mode",
    "features.dashboards.team.mode",
    "features.ticket_creation.mode",
)
# Production's file: the removed keys, the state keys, the new package keys, the sizing and the settle, the GPU
# jobs' options, the new deploy keys, and four more deploy keys it does not hold, each at its default.
PRODUCTION = (
    *REMOVED,
    *STATE_KEYS,
    *NEW_PACKAGE_KEYS,
    *REPOSITORY_NAMES,
    *SIZING_AND_SETTLE,
    *GPU_JOB_OPTIONS,
    *NEW_DEPLOY_KEYS,
    "split_part_lines",
    "features.merge_queue.queue_notices",
    "effort.first_pass",
    "jira.default_epic",
)
# The rows of the round trip's list of intended differences: each says why of a difference.
LISTED = round_trip.rows(round_trip.DIFFERENCES)


def data_file(name: str) -> dict:
    return json.loads((ROOT / "tests" / "data" / name).read_text(encoding="utf-8"))


def leaves(data, prefix: str = "") -> dict:
    """Nested DATA's leaves by dotted key; an empty mapping is a leaf."""
    out = {}
    for k, v in (data or {}).items():
        if isinstance(v, dict) and v:
            out.update(leaves(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def registry_key(leaf: str) -> str:
    """The registry's key the dotted LEAF names, or the key it is an entry of."""
    name = leaf
    while registry.kind(name) == registry.ENTRY:
        name = name.rsplit(".", 1)[0]
    return name


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = self.dir / "home"
        self.home.mkdir()
        self.settings_dir = self.dir / "settings"
        self.settings_dir.mkdir()
        for patch in (
            mock.patch.object(doctor, "deployment_image", return_value=False),
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.object(
                container_layout, "ROCM_RECORD", str(self.dir / "no-rocm.json")
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def config_json(self, data) -> None:
        (self.home / "config.json").write_text(json.dumps(data), encoding="utf-8")

    def layer(self, **over) -> dict:
        """A layer (a_layer) delivered in the test's settings folder."""
        layer = self.a_layer(**over)
        (self.settings_dir / RC.LAYER_FILE).write_text(
            json.dumps(layer), encoding="utf-8"
        )
        return layer

    @staticmethod
    def a_layer(**over) -> dict:
        """A layer confirmed for this package and this layout of the deploy settings, OVER over its keys."""
        settings = over.get("settings", {"repository": {"name": "acme/widgets"}})
        layer = {
            "kind": RC.LAYER_KIND,
            "schema_version": RC.LAYER_SCHEMA_VERSION,
            "deploy_schema_version": file_types.DEPLOY.version,
            "package_version": image.package_version(PACKAGE),
            "confirmed_at": "2026-01-02T03:04:05Z",
            "fingerprint": "sha256:" + "0" * 64,
            "settings": settings,
            "set_in_file": set_in(settings),
            "container": {},
        }
        layer.update(over)
        return layer

    def load(self) -> RC.Resolved:
        return RC.load(self.home, self.settings_dir)

    def pair(self, resolved: RC.Resolved, name: str) -> tuple:
        return registry.get(resolved.values, name), resolved.source.get(name)

    def rk(self, *argv: str) -> tuple:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(list(argv))
        return code, out.getvalue()


class TestResolve(Base):
    def test_each_key_comes_from_its_owners_layer_and_says_which(self):
        r = RC.resolve(
            registry.package_defaults(),
            {"loop_heartbeat_min": 7},
            {"repo": "acme/widgets"},
            {"standby": {"on": True}},
            derived.Layout(),
        )
        self.assertEqual(self.pair(r, "repo"), ("acme/widgets", RC.DEPLOY))
        self.assertEqual(  # a deploy key at its default: the package's
            self.pair(r, "rocke_path"),
            (registry.default("rocke_path"), RC.PACKAGE_SOURCE),
        )
        self.assertEqual(self.pair(r, "loop_heartbeat_min"), (7, RC.REPOSITORY))
        self.assertEqual(
            self.pair(r, "tick_timeout_min"),
            (registry.default("tick_timeout_min"), RC.PACKAGE_SOURCE),
        )
        self.assertEqual(self.pair(r, "standby.on"), (True, RC.STATE))
        self.assertEqual(self.pair(r, "standby.reason"), ("", RC.STATE))
        self.assertEqual(self.pair(r, "codeowner_org"), ("acme", RC.DERIVED))
        names = {k.name for k in registry.keys()}
        self.assertLessEqual(set(r.source), names)
        self.assertLessEqual(set(r.source.values()), set(RC.SOURCES))
        self.assertEqual(
            set(leaves(r.values)) - names,
            {k for k in leaves(r.values) if registry.kind(k) == registry.ENTRY},
        )

    def test_config_json_is_one_more_source_below_the_layer_and_the_state(self):
        carried = {
            "repo": "other/name",
            "loop_heartbeat_min": 11,
            "tick_timeout_min": 9,
            "standby": {"on": False, "reason": "a reason"},
            "codeowner_org": "given-org",
            "no_such_key": 1,
        }
        args = (registry.package_defaults(), {"loop_heartbeat_min": 7})
        layout = derived.Layout(given=RC.given(carried))
        r = RC.resolve(
            *args, {"repo": "acme/widgets"}, {"standby": {"on": True}}, layout, carried
        )
        self.assertEqual(
            self.pair(r, "repo"), ("acme/widgets", RC.DEPLOY)
        )  # the layer wins
        self.assertEqual(self.pair(r, "loop_heartbeat_min"), (11, RC.CONFIG_JSON))
        self.assertEqual(self.pair(r, "tick_timeout_min"), (9, RC.CONFIG_JSON))
        self.assertEqual(self.pair(r, "standby.on"), (True, RC.STATE))  # the state wins
        self.assertEqual(self.pair(r, "standby.reason"), ("a reason", RC.CONFIG_JSON))
        self.assertEqual(self.pair(r, "codeowner_org"), ("given-org", RC.CONFIG_JSON))
        self.assertNotIn("no_such_key", list(r.values))
        r = RC.resolve(*args, None, {}, layout, carried)  # no layer delivered
        self.assertEqual(self.pair(r, "repo"), ("other/name", RC.CONFIG_JSON))
        self.assertEqual(
            self.pair(r, "rocke_path"),
            (registry.default("rocke_path"), RC.PACKAGE_SOURCE),
        )


class TestDeploySource(unittest.TestCase):
    """deploy_source: a key of the deploy settings file is deploy when the delivered layer lists it, or the map
    that holds it, among the keys your file sets (set_in_file), and package otherwise.
    """

    def test_a_key_is_deploy_when_your_file_sets_it_or_the_map_that_holds_it(self):
        layer = {
            "set_in_file": [
                "repository.name",
                "model_bindings",  # a map: one key
                "repository.scope_labels",  # a list: one key
            ]
        }
        for name, want in (
            ("repository.name", RC.DEPLOY),
            ("model_bindings", RC.DEPLOY),
            ("model_bindings.fast", RC.DEPLOY),  # an entry of a map your file sets
            ("repository.scope_labels", RC.DEPLOY),
            ("repository.folder", RC.PACKAGE_SOURCE),
            ("repository.shared_label_folders", RC.PACKAGE_SOURCE),  # a map left out
            ("features.ticket_creation.labels", RC.PACKAGE_SOURCE),  # a list left out
            ("repository", RC.PACKAGE_SOURCE),  # a section is no key
            ("repository.name_of_another", RC.PACKAGE_SOURCE),  # no key inside one set
        ):
            with self.subTest(name=name):
                self.assertEqual(RC.deploy_source(layer, name), want)

    def test_a_layer_that_lists_no_key_your_file_sets_shows_each_as_the_package_s(
        self,
    ):
        for layer in ({"set_in_file": []}, {}):
            with self.subTest(layer=layer):
                self.assertEqual(
                    RC.deploy_source(layer, "repository.name"), RC.PACKAGE_SOURCE
                )


class TestLoad(Base):
    def test_a_delivered_layer_runs_and_the_monitor_s_start_records_it_as_applied_once(
        self,
    ):
        times = iter(f"2026-01-02T03:0{n}:00Z" for n in range(10))
        for patch in (
            mock.patch.object(store, "now", side_effect=lambda: next(times)),
            mock.patch.object(doctor, "deployment_image", return_value=True),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.config_json({"repo": "other/name"})
        layer = self.layer(settings=SETTINGS)
        loaded = self.load()
        self.assertEqual(self.pair(loaded, "repo"), ("acme/widgets", RC.DEPLOY))
        self.assertEqual(loaded.layer, layer)
        self.assertTrue(RC.record_applied(self.home, loaded))
        record = config.state_dir(self.home) / RC.APPLIED_FILE
        self.assertEqual(
            RC.applied(self.home),
            {
                "kind": RC.APPLIED_KIND,
                "schema_version": RC.APPLIED_SCHEMA_VERSION,
                "fingerprint": layer["fingerprint"],
                "package_version": layer["package_version"],
                "applied_at": "2026-01-02T03:00:00Z",
            },
        )
        self.assertEqual(stat.S_IMODE(record.stat().st_mode), perms.SHARED_FILE_MODE)
        self.assertFalse(RC.record_applied(self.home, self.load()))
        self.assertEqual(  # the same layer is recorded once
            RC.applied(self.home)["applied_at"], "2026-01-02T03:00:00Z"
        )
        self.layer(settings=SETTINGS, fingerprint="sha256:" + "1" * 64)
        self.assertTrue(RC.record_applied(self.home, self.load()))
        self.assertEqual(
            (RC.applied(self.home)["fingerprint"], RC.applied(self.home)["applied_at"]),
            ("sha256:" + "1" * 64, "2026-01-02T03:02:00Z"),
        )
        (self.home / "config.json").unlink()
        self.assertEqual(
            self.pair(self.load(), "repo"), ("acme/widgets", RC.DEPLOY)
        )  # the layer alone runs

    def test_one_comparison_says_whether_the_applied_record_holds_a_layer(self):
        """`holds`: record_applied asks it before it writes, and a reader of the record asks the same."""
        layer = {
            "fingerprint": "sha256:" + "0" * 64,
            "package_version": "1.2.3",
            "settings": {},
        }
        record = {
            "kind": RC.APPLIED_KIND,
            "fingerprint": layer["fingerprint"],
            "package_version": "1.2.3",
            "applied_at": "2026-01-02T03:04:05Z",
        }
        self.assertTrue(RC.holds(record, layer))
        for key in RC.APPLIED_MARK:
            self.assertFalse(RC.holds(dict(record, **{key: "other"}), layer), key)
        self.assertFalse(RC.holds(None, layer))
        self.assertFalse(RC.holds(record, None))
        container = mock.patch.object(doctor, "deployment_image", return_value=True)
        container.start()
        self.addCleanup(container.stop)
        self.layer(settings=SETTINGS)
        loaded = self.load()
        with mock.patch.object(RC, "holds", return_value=True) as asked:
            self.assertFalse(RC.record_applied(self.home, loaded))
        asked.assert_called_once_with(None, loaded.layer)
        self.assertIsNone(RC.applied(self.home))

    def test_a_load_records_nothing_and_a_configuration_with_a_problem_is_never_recorded(
        self,
    ):
        """The monitor writes the applied record as it starts on a layer; a load only reads, whoever makes it (the
        doctor, rk config show), and a layer whose configuration cannot run is never recorded.
        """
        container = mock.patch.object(doctor, "deployment_image", return_value=True)
        container.start()
        self.addCleanup(container.stop)
        self.layer(settings=SETTINGS)
        loaded = self.load()
        self.assertEqual(loaded.problems, [])
        self.assertIsNone(RC.applied(self.home))
        repository = dict(SETTINGS["repository"], name="not a repository")
        self.layer(settings=dict(SETTINGS, repository=repository))
        loaded = self.load()
        self.assertTrue(loaded.problems)
        with self.assertRaises(config.ConfigError) as e:
            RC.record_applied(self.home, loaded)
        self.assertIn(loaded.problems[0], str(e.exception))
        self.assertIsNone(RC.applied(self.home))
        no_layer = RC.Resolved({}, {}, [])
        self.assertFalse(RC.record_applied(self.home, no_layer))
        self.assertIsNone(RC.applied(self.home))

    def test_a_layer_that_cannot_run_stops_the_load_and_names_apply_or_rollback(self):
        self.config_json({"repo": "acme/widgets"})
        version, layout = image.package_version(PACKAGE), file_types.DEPLOY.version
        for over, found in (
            (
                {"package_version": "0.0.0"},
                f"it was confirmed for the package '0.0.0', and this is the package {version!r}",
            ),
            (
                {"deploy_schema_version": layout - 1},
                f"it was confirmed for layout {layout - 1} of the deploy settings, and this package reads layout "
                f"{layout}",
            ),
            (
                {"kind": "deploy_settings"},
                "it is of the kind 'deploy_settings' at layout 1, not 'confirmed_deploy_settings' at layout 1",
            ),
            ({"settings": None}, "it has no settings"),
        ):
            self.layer(**over)
            with self.assertRaises(RC.LayerError) as e:
                self.load()
            self.assertIn(found, str(e.exception))
        (self.settings_dir / RC.LAYER_FILE).write_text("{", encoding="utf-8")
        with self.assertRaises(RC.LayerError) as e:
            self.load()
        self.assertIn("are no JSON", str(e.exception))
        self.assertIsInstance(e.exception, config.ConfigError)
        self.assertIsNone(RC.applied(self.home))
        self.assertEqual(
            str(RC.LayerError.fix("rkmon")),
            "rkmon apply (this package's code, delivered with your settings), or rkmon rollback",
        )

    def test_without_a_layer_config_json_runs_and_without_either_nothing_does(self):
        with self.assertRaises(config.ConfigError) as e:
            self.load()
        self.assertEqual(
            (str(e.exception), e.exception.fix("rkmon")),
            (
                f"no configuration at {self.home / 'config.json'}",
                checks.Steps(doctor.apply_commands()),
            ),
        )
        self.config_json({"repo": "acme/widgets"})
        self.assertEqual(
            self.pair(self.load(), "repo"), ("acme/widgets", RC.CONFIG_JSON)
        )
        self.assertIsNone(RC.applied(self.home))
        self.config_json([])
        with self.assertRaises(config.ConfigError) as e:
            self.load()
        self.assertIn("is no JSON object", str(e.exception))

    def test_beside_a_layer_config_json_is_never_read(self):
        """Once a layer is delivered, the one-time move splits config.json by owner
        (rkmon_lib/one_time_move.py), and the load never reads it."""
        self.config_json({"repo": "other/name", "standby": {"on": True}})
        self.assertEqual(self.pair(self.load(), "standby.on"), (True, RC.CONFIG_JSON))
        self.layer(settings=SETTINGS)
        self.assertEqual(self.pair(self.load(), "standby.on"), (False, RC.STATE))
        self.config_json([])  # no JSON object: read, it would stop the load
        self.assertEqual(self.pair(self.load(), "repo"), ("acme/widgets", RC.DEPLOY))

    def test_the_internal_state_wins_over_config_json(self):
        self.config_json({"repo": "acme/widgets", "standby": {"on": False}})
        state_store.set(self.home, "standby.on", True)
        self.assertEqual(self.pair(self.load(), "standby.on"), (True, RC.STATE))
        state_store.path(self.home).write_text(
            '{"kind": "internal_state"}', encoding="utf-8"
        )
        with self.assertRaises(state_store.StateError):
            self.load()

    def test_the_hosts_view_of_a_layer_is_what_the_containers_load_runs(self):
        layer = self.layer()
        state_store.set(self.home, "standby.on", True)
        with mock.patch.object(doctor, "deployment_image", return_value=True):
            loaded = self.load()
        with (  # a host's own process: its environment is no container's
            mock.patch.object(doctor, "deployment_image", DEPLOYMENT_IMAGE),
            mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}),
        ):
            shown = RC.from_settings(
                layer["settings"], state=state_store.read(self.home)
            )
        self.assertEqual(shown.values, loaded.values)
        self.assertEqual(shown.source, loaded.source)
        self.assertEqual(
            registry.get(shown.values, "paths.working_dir"), container_layout.WORK_DIR
        )
        self.assertEqual(
            registry.get(shown.values, "admin_dashboard.bind"),
            container_layout.LISTEN_ADDRESS,
        )

    def test_the_layer_hands_out_how_the_claude_session_sends_its_key(self):
        """integrations.claude_code's three keys reach the running configuration from the delivered layer, for the
        Claude session, whose variables they make (orchestrator.session_variables)."""
        sent = {
            "api_key_sent_as": "header_and_bearer",
            "api_key_header": "X-Gateway-Key",
            "api_key_scheme": "Bearer",
        }
        integrations = dict(SETTINGS["integrations"], claude_code=sent)
        self.layer(settings=dict(SETTINGS, integrations=integrations))
        with mock.patch.object(doctor, "deployment_image", return_value=True):
            loaded = self.load()
        for name, value in sent.items():
            with self.subTest(key=name):
                self.assertEqual(
                    self.pair(loaded, f"claude.{name}"), (value, RC.DEPLOY)
                )

    def test_the_layer_is_read_where_it_is_delivered_in_the_container(self):
        self.config_json({"repo": "other/name"})
        self.layer()
        with mock.patch.object(RC, "SETTINGS_DIR", str(self.settings_dir)):
            self.assertEqual(
                self.pair(RC.load(self.home), "repo"), ("other/name", RC.CONFIG_JSON)
            )  # a host has no delivered layer
            with mock.patch.object(doctor, "deployment_image", return_value=True):
                self.assertEqual(
                    self.pair(RC.load(self.home), "repo"), ("acme/widgets", RC.DEPLOY)
                )

    def test_the_container_s_venv_is_the_image_s_where_the_image_has_one(self):
        """In the container, paths.venv is derived: the image's checks venv (container_layout.IMAGE_VENV) where
        the image has one, else the one in the working folder."""
        self.layer()
        venv = self.dir / "venv"
        with (
            mock.patch.object(doctor, "deployment_image", return_value=True),
            mock.patch.object(container_layout, "IMAGE_VENV", str(venv)),
        ):
            work = f"{container_layout.WORK_DIR}/venv"
            self.assertEqual(self.pair(self.load(), "paths.venv"), (work, RC.DERIVED))
            venv.mkdir()
            self.assertEqual(
                self.pair(self.load(), "paths.venv"), (str(venv), RC.DERIVED)
            )

    def test_the_container_s_llvm_tools_are_the_image_s_rocm_s(self):
        """In the container, paths.rocm_llvm_bin is derived at each load from the image's ROCm: the folder its ROCm
        record names (container_layout.ROCM_RECORD), over one a config.json gives, as the image's ROCm is the only one
        there; where it records none, config.json's while the load reads no layer (beside a layer, config.json is never
        read), else blank. A host reads no image's record.
        """
        self.layer()
        record = pathlib.Path(container_layout.ROCM_RECORD)
        llvm = "/opt/rocm-9.9.0/lib/llvm/bin"
        key = "paths.rocm_llvm_bin"
        with mock.patch.object(doctor, "deployment_image", return_value=True):
            self.assertEqual(self.pair(self.load(), key), ("", RC.DERIVED))
            self.config_json({"paths": {"rocm_llvm_bin": "/l"}})
            self.assertEqual(self.pair(self.load(), key), ("", RC.DERIVED))
            (self.settings_dir / RC.LAYER_FILE).unlink()  # a load that reads no layer
            self.assertEqual(self.pair(self.load(), key), ("/l", RC.CONFIG_JSON))
            record.write_text(json.dumps({"llvm_bin": llvm}), encoding="utf-8")
            self.assertEqual(self.pair(self.load(), key), (llvm, RC.DERIVED))
            self.config_json({"repo": "acme/widgets"})
            self.assertEqual(self.pair(self.load(), key), (llvm, RC.DERIVED))
            self.layer()
            self.assertEqual(self.pair(self.load(), key), (llvm, RC.DERIVED))
        self.assertEqual(
            self.pair(RC.load(self.home), key), ("", RC.DERIVED)
        )  # a host's load

    def test_config_load_is_the_running_configuration(self):
        self.config_json({"repo": "acme/widgets"})
        state_store.set(self.home, "standby.on", True)
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)}):
            cfg = config.load(check=False)
            self.assertIs(registry.get(cfg, "standby.on"), True)
            self.assertEqual(
                cfg["loop_heartbeat_min"], registry.default("loop_heartbeat_min")
            )
            self.assertEqual(cfg, self.load().values)
            with self.assertRaises(registry.UnknownKey):  # the configuration is strict
                cfg.get("no_such_key")
            with self.assertRaises(config.ConfigError) as e:
                config.load()
            self.assertTrue(
                str(e.exception).startswith(
                    f"{self.home / 'config.json'} cannot run yet:\n  - "
                ),
                str(e.exception),
            )
            self.assertIn(
                f"{deploy_tree.name('rocke_path')} is blank", str(e.exception)
            )
            state_store.path(self.home).write_text("[]", encoding="utf-8")
            with self.assertRaises(state_store.StateError):
                config.load(check=False)

    def differences(self, raw) -> tuple:
        """The leaves where the running configuration of the configuration file RAW differs from config.normalize's,
        today's reader; today's reader's leaves; the running configuration's leaves; the load.
        """
        self.config_json(raw)
        text = (self.home / "config.json").read_text(encoding="utf-8")
        want = leaves(config.normalize(renames.migrate(json.loads(text))[0]))
        loaded = self.load()
        got = leaves(loaded.values)
        differ = {
            k
            for k in set(want) | set(got)
            if want.get(k, registry.MISSING) != got.get(k, registry.MISSING)
        }
        return differ, want, got, loaded

    @staticmethod
    def reader_s(key: str, got: dict):
        """The value the reader of the derived key KEY took for a blank, in the running configuration GOT."""
        cfg = registry.nest(dict(got, **{key: ""}))
        return {
            "worker.tag": lambda: bindings.worker_tag(cfg),
            "draft_stage": lambda: draft_stage.for_config(cfg).name,
            "verifier_name": lambda: bindings.verifier_name(cfg),
            "review_first_line_no_draft": lambda: bindings.no_draft_line(cfg),
            "clone_url": lambda: derived.github_address(got["repo"]),
            "followup_scan.dir": lambda: str(
                pathlib.Path(got["paths.toolkit"]) / "state" / "followups"
            ),
        }[key]()

    def test_a_configuration_file_runs_as_config_normalize_reads_it_but_for_the_keys_listed_with_their_reasons(
        self,
    ):
        """The file a first setup writes from today's defaults, production's (tests/data/production_config.json) and
        the one an older setup wrote (tests/data/older_setup_config.json): each differs from today's reader only in
        the keys listed for it, each a difference the round trip's list of intended differences says why of.
        """
        raw = config.load_defaults()
        for name, value in dict(
            ANSWERS, **{"paths.working_dir": str(self.dir / "work")}
        ).items():
            registry.put(raw, name, value)
        for name, data, listed in (
            (
                "today's defaults",
                raw,
                (*PORTS, *FROM_THE_READER, *HOST_KEYS),
            ),
            (
                "production's",
                data_file("production_config.json"),
                (*PRODUCTION, *FROM_THE_READER, *HOST_KEYS),
            ),
            (
                "an older setup's",
                data_file("older_setup_config.json"),
                (
                    *STATE_KEYS,
                    *NEW_PACKAGE_KEYS,
                    *REPOSITORY_NAMES,
                    *SIZING_AND_SETTLE,
                    *GPU_JOB_OPTIONS,
                    *NEW_DEPLOY_KEYS,
                    *MODES_THEY_LACK,
                    *FROM_THE_READER,
                    *HOST_KEYS,
                    *WORKER_KEYS,
                    "confluence.space",
                    "paths.rocm_reference_clang",
                    "update.path",
                    "update.test_timeout_min",
                    "wsl_distro",
                    *SERVICES,
                ),
            ),
        ):
            with self.subTest(file=name):
                differ, today_s, got, loaded = self.differences(data)
                self.assertEqual(differ, set(listed))
                for k in differ:
                    # the round trip's list says why, of the kind the round trip gives it; a key whose value the
                    # deploy plan makes now is compared with the plan's, which no setting changes
                    difference = round_trip.Difference(
                        k,
                        round_trip.kind_of(k),
                        today_s.get(k, registry.MISSING),
                        (
                            round_trip.planned({})[k]
                            if k in round_trip.PLANNED
                            else got.get(k, registry.MISSING)
                        ),
                    )
                    self.assertIsNotNone(round_trip.listed(difference, LISTED), k)
                    if k in PORTS:
                        continue
                    if k in FROM_THE_READER:
                        self.assertEqual(got[k], self.reader_s(k, got), k)
                        continue
                    if k in HOST_KEYS:
                        self.assertEqual(
                            got[k],
                            str(
                                pathlib.Path(got["paths.working_dir"])
                                / HOST_KEYS_FILES[k]
                            ),
                            k,
                        )
                        continue
                    if k in (*REMOVED, *SERVICES):
                        self.assertIsNone(registry.kind(k), k)
                        self.assertNotIn(k, got)
                    else:  # the key's default (a map's entry: its map's), from its owner's layer, but a deploy
                        # key's from the package
                        name = registry_key(k)
                        want = registry.default(name)
                        for part in k[len(name) :].split(".")[1:]:
                            want = want[part]
                        owner = registry.key(name).owner
                        self.assertEqual(
                            (got[k], loaded.source[name]),
                            (
                                want,
                                (
                                    RC.PACKAGE_SOURCE
                                    if owner == registry.DEPLOY
                                    else owner
                                ),
                            ),
                            k,
                        )
                if name == "today's defaults":
                    self.assertEqual(
                        (got["admin_dashboard.port"], got["team_dashboard.port"]),
                        (container_layout.ADMIN_PORT, container_layout.TEAM_PORT),
                    )


class InContainer(Base):
    """rk in the container: the test's home, and the layer delivered in the test's settings folder."""

    def setUp(self):
        super().setUp()
        for patch in (
            mock.patch.object(doctor, "deployment_image", return_value=True),
            mock.patch.object(RC, "SETTINGS_DIR", str(self.settings_dir)),
            mock.patch.dict(
                os.environ, {"RKMON_HOME": str(self.home), "RKMON_WRAPPER": "rkmon"}
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)


class TestUnreadable(InContainer):
    """A configuration that cannot be read runs no part of the monitor: `rk features --runs` refuses
    every part and `rk tick-min` fails, so the entry point starts no part and runs no tick (tests/test_entrypoint.py),
    and the doctor fails with the fix. Standby plays no part in it: nothing is written.
    """

    def layers_that_cannot_run(self):
        """Each delivered layer that cannot run, set up in turn, with what its refusal says of it: one of
        another package, one that does not list the keys your file sets, one of another layout of the deploy
        settings, one that is no JSON, a link to no file (a layer delivered all the same: runtime_config.delivered),
        and none where this monitor has run one, as its applied record says, or where a record that cannot be read
        leaves that unknown.
        """
        version, layout = image.package_version(PACKAGE), file_types.DEPLOY.version
        layer = self.settings_dir / RC.LAYER_FILE
        self.layer(package_version="0.0.0")
        yield f"cannot run: it was confirmed for the package '0.0.0', and this is the package {version!r}"
        self.layer(set_in_file=None)
        yield "cannot run: it does not list the keys your file sets"
        self.layer(deploy_schema_version=layout - 1)
        yield f"cannot run: it was confirmed for layout {layout - 1} of the deploy settings"
        layer.write_text("{", encoding="utf-8")
        yield "are no JSON"
        layer.unlink()
        layer.symlink_to(self.settings_dir / "no-such-layer.json")
        yield "cannot be read: it is a link to a file that is not there"
        layer.unlink()
        self.layer(settings=SETTINGS)
        self.assertTrue(RC.record_applied(self.home, RC.load(self.home)))
        (self.settings_dir / RC.LAYER_FILE).unlink()
        yield "are missing, and this monitor has run your confirmed settings before (applied at "
        (config.state_dir(self.home) / RC.APPLIED_FILE).write_text(
            "{", encoding="utf-8"
        )
        yield "are missing, and whether this monitor has run your confirmed settings before is not known: "

    def test_no_part_starts_and_no_tick_runs_while_the_layer_cannot_run(self):
        self.layer()
        runs = {
            part: self.rk("features", "--runs", part)[0] for part in features.COMPONENTS
        }
        self.assertLessEqual(
            set(runs.values()), {0, 1}, runs
        )  # a layer that runs: each part runs or not
        code, out = self.rk("tick-min")
        self.assertEqual((code, out.strip().isdigit()), (0, True), out)
        for found in self.layers_that_cannot_run():
            before = sorted(self.dir.rglob("*"))
            for part in features.COMPONENTS:
                code, out = self.rk("features", "--runs", part)
                self.assertEqual(code, 4, part)
                self.assertTrue(
                    out.startswith("REFUSED: your confirmed settings at "), out
                )
            for argv, refused in ((["tick-min"], 4), (["features"], 1)):
                code, out = self.rk(*argv)
                self.assertEqual(code, refused, argv)
                self.assertIn(found, " ".join(out.split()))
            self.assertTrue(self.rk("features")[1].startswith("Nothing runs: "))
            self.assertEqual(sorted(self.dir.rglob("*")), before)  # nothing written
        self.assertFalse(
            state_store.path(self.home).exists()
        )  # no standby: no state at all

    def test_rk_refuses_in_cannot_run_s_words(self):
        """Wherever rk cannot load the configuration it says what applied.cannot_run says, as rk start and
        `rk applied --load` do: why, then the fix the error names (here a layer's: rkmon apply or rkmon rollback).
        """
        self.layer(package_version="0.0.0")
        with self.assertRaises(RC.LayerError) as caught:
            RC.load(self.home)
        said = "".join(
            f"{line}\n" for line in applied.cannot_run(caught.exception, "REFUSED")
        )
        self.assertIn(f"\n{checks.FIX_LABEL}", said)
        for argv in (["features", "--runs", "web"], ["paths"], ["tick-min"]):
            self.assertEqual(self.rk(*argv), (4, said), argv)

    def test_no_part_starts_without_a_configuration(self):
        with self.assertRaises(config.NoConfiguration):
            RC.load(self.home)
        code, out = self.rk("features", "--runs", "web")
        self.assertEqual(code, 4)
        self.assertTrue(out.startswith("REFUSED: no configuration at "), out)
        with mock.patch.object(rkmon, "cmd_features", return_value=0) as summary:
            code, out = self.rk("features")
        self.assertEqual(code, 0)
        self.assertTrue(
            out.startswith("Every optional feature is off: no configuration at "), out
        )
        self.assertIn(
            f"\n{checks.FIX_LABEL}{checks.Steps(doctor.apply_commands())}\n", out
        )
        defaults = config.normalize(config.load_defaults())
        summary.assert_called_once_with(defaults)  # then the summary of the defaults

    def test_the_doctor_fails_with_the_fix(self):
        """With no configuration, its fix is your deploy settings confirmed and applied; with each layer that cannot
        run, rkmon apply or rkmon rollback."""

        def said() -> str:
            out = io.StringIO()
            with (
                mock.patch.object(doctor, "prerequisites", return_value=[]),
                contextlib.redirect_stdout(out),
            ):
                code = doctor.main()
            self.assertEqual(code, doctor.EXIT_CONFIG)
            return " ".join(out.getvalue().split())

        text = said()
        self.assertTrue(
            text.startswith("FAIL configuration: no configuration at "), text
        )
        fixed = str(checks.Steps(doctor.apply_commands()))
        self.assertEqual(fixed, "rkmon confirm and rkmon apply")
        self.assertIn(fixed, text)
        fix = " ".join(str(RC.LayerError.fix("rkmon")).split())
        for found in self.layers_that_cannot_run():
            text = said()
            self.assertTrue(
                text.startswith("FAIL configuration: your confirmed settings at "),
                text,
            )
            self.assertIn(found, text)
            self.assertIn(fix, text)

    def test_the_doctor_s_rows_as_json_carry_each_fix(self):
        """rk doctor --json: the rows alone, one JSON list on one line, each read back as the row it was
        (checks.check_from_json), its fix with it; the report's exit code."""
        out = io.StringIO()
        with (
            mock.patch.object(doctor, "prerequisites", return_value=[]),
            contextlib.redirect_stdout(out),
        ):
            code = doctor.main(as_json=True)
        self.assertEqual(code, doctor.EXIT_CONFIG)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        rows = [checks.check_from_json(x) for x in json.loads(lines[0])]
        self.assertEqual(rows[0].name, "configuration")
        self.assertEqual(rows[0].status, checks.Status.FAIL)
        self.assertEqual(rows[0].fix, checks.Steps(doctor.apply_commands()))


class TestTheHandedLayer(InContainer):
    """The layer handed to a one-off container of the logins' walk (deploy step 5): the entry point's root phase writes
    it (docker/rk_root.py hand-layer), and rk there reads it in place of the delivered one, so that step 5's commands
    answer for your confirmed settings before the first start too, and write no configuration. A folder or file of it
    that is not root's alone is refused in plain words, its removal the fix; a layer that cannot run, as a delivered one
    cannot, the image check. Here the test's user stands for root.
    """

    def setUp(self):
        super().setUp()
        self.handed = self.dir / "handed"
        self.path = self.handed / RC.LAYER_FILE
        self.what = (
            f"the confirmed deploy settings handed to this container ({self.path})"
        )
        for patch in (
            mock.patch.object(RC, "HANDED_DIR", str(self.handed)),
            mock.patch.object(RC, "ROOT_UID", os.getuid()),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def hand(self, data: bytes | None = None, **over) -> dict:
        """A layer (a_layer, OVER over its keys), or DATA, written as the root phase writes it, the test's user its
        owner."""
        layer = self.a_layer(**over)
        rk_root.hand_layer(
            json.dumps(layer).encode() if data is None else data,
            str(self.path),
            (os.getuid(), os.getgid()),
        )
        return layer

    def test_the_root_phase_writes_it_for_the_monitor_and_the_writer_to_read(self):
        """rk_root.py hand-layer: the folder mode 750 and the file 640, root's and the monitor's group's (here the
        test's user's), the layer whole, put in place by a rename (no other file left), and whole again each time. An
        empty layer, or one longer than HANDED_MAX bytes, is refused, ERROR, and nothing is written; nothing is written
        in a folder in its place that is not root's (another user's, or a link) either, which rk's read refuses with
        the image's fix (test_a_folder_or_file_that_is_not_root_s_alone_is_refused_in_plain_words); root alone runs
        it, and it prints its refusal.
        """
        layer = self.hand()
        modes = (self.handed.stat().st_mode, self.path.stat().st_mode)
        self.assertEqual(tuple(map(stat.S_IMODE, modes)), (0o750, 0o640))
        self.assertEqual(os.listdir(self.handed), [RC.LAYER_FILE])
        self.assertEqual(RC.handed_layer(), layer)
        layer = self.hand(confirmed_at="2026-02-03T04:05:06Z")
        self.assertEqual(RC.handed_layer(), layer)
        written, owner = self.path.read_bytes(), (os.getuid(), os.getgid())
        link = self.dir / "link"
        link.symlink_to(self.handed)
        not_written = (
            "ERROR: your confirmed settings handed to this container are not written"
        )
        for data, path, owned_by, why in (
            (b"", self.path, owner, "they are empty"),
            (
                b"{" * (rk_root.HANDED_MAX + 1),
                self.path,
                owner,
                f"they are longer than {rk_root.HANDED_MAX} bytes",
            ),
        ):
            with self.subTest(why=why):
                with self.assertRaises(rk_root.Stop) as stop:
                    rk_root.hand_layer(data, str(path), owned_by)
                self.assertEqual(
                    (stop.exception.code, stop.exception.lines),
                    (rk_root.ERROR, [f"{not_written}: {why}"]),
                )
                self.assertEqual(self.path.read_bytes(), written)
                self.assertEqual(os.listdir(self.handed), [RC.LAYER_FILE])
        other = json.dumps(self.a_layer(confirmed_at="2026-02-03T04:05:07Z")).encode()
        for path, owned_by in (
            (link / RC.LAYER_FILE, owner),  # a link in the folder's place
            (self.path, (os.getuid() + 1, os.getgid())),  # another user's folder
        ):
            with self.subTest(path=path, owner=owned_by):
                self.assertEqual(rk_root.hand_layer(other, str(path), owned_by), [])
                self.assertEqual(self.path.read_bytes(), written)
                self.assertEqual(os.listdir(self.handed), [RC.LAYER_FILE])
        real = rk_root.hand_layer
        for euid, data, said, code in (
            (0, b"", f"{not_written}: they are empty\n", rk_root.ERROR),
            (
                0,
                json.dumps(self.a_layer(confirmed_at="2026-03-04T05:06:07Z")).encode(),
                "",
                rk_root.DONE,
            ),
            (1000, b"{}", "", rk_root.USAGE),
        ):
            out = io.StringIO()
            with (
                self.subTest(euid=euid, code=code),
                mock.patch.object(rk_root.os, "geteuid", return_value=euid),
                mock.patch.object(
                    rk_root, "hand_layer", lambda d: real(d, str(self.path), owner)
                ),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    (
                        rk_root.main(["hand-layer"], stdin=io.BytesIO(data)),
                        out.getvalue(),
                    ),
                    (code, said),
                )
        self.assertEqual(RC.handed_layer()["confirmed_at"], "2026-03-04T05:06:07Z")

    def test_it_runs_in_place_of_the_delivered_layer(self):
        """In the container a handed layer runs in place of the delivered one, and the resolved configuration's layer
        is the one it read; without it, the delivered one runs again. A host never reads it.
        """
        self.layer(settings=SETTINGS)
        repository = dict(SETTINGS["repository"], name="acme/gadgets")
        handed = self.hand(settings=dict(SETTINGS, repository=repository))
        loaded = RC.load(self.home)
        self.assertEqual(loaded.layer, handed)
        self.assertEqual(self.pair(loaded, "repo"), ("acme/gadgets", RC.DEPLOY))
        with mock.patch.object(doctor, "deployment_image", return_value=False):
            self.assertEqual(
                self.pair(RC.load(self.home, self.settings_dir), "repo"),
                ("acme/widgets", RC.DEPLOY),
            )
        shutil.rmtree(self.handed)
        self.assertEqual(
            self.pair(RC.load(self.home), "repo"), ("acme/widgets", RC.DEPLOY)
        )

    def one_off(self) -> pathlib.Path:
        """A one-off container of the logins' walk before the first start: the container's environment, its secrets
        volumes and working folder the test's folders, and no GitHub and no atlassian CLI asked. The monitor's
        secrets folder."""
        secrets = self.dir / "secrets"
        env = dict(os.environ, RKMON_CONTAINER="1")
        env.pop(
            writer.BROKER_VAR, None
        )  # the monitor's rk: it sets the monitor's secrets itself
        for patch in (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(container_layout, "SECRETS_DIR", str(secrets)),
            mock.patch.object(
                container_layout, "WRITE_SECRETS_DIR", str(self.dir / "secrets-write")
            ),
            mock.patch.object(container_layout, "WORK_DIR", str(self.dir / "work")),
            mock.patch.object(ghlogins, "check", return_value=({"sides": {}}, True)),
            mock.patch.object(
                atlassian, "status", return_value={"found": False, "logged_in": False}
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        writer.forget()
        self.addCleanup(writer.forget)
        return secrets

    def test_step_5_s_commands_run_with_no_config_json_and_no_delivered_layer(self):
        """A one-off container before the first start has neither config.json nor a delivered layer: there rk logins
        and rk secret-set refuse, as there is no configuration, and write nothing. Handed your confirmed settings, rk
        logins lists what they need and rk secret-set sets a secret, its value never printed, and neither writes a
        configuration or a layer."""
        secrets = self.one_off()
        before = sorted(self.dir.rglob("*"))
        for argv in (["logins", "--json"], ["secret-set", "admin_access_token"]):
            with self.subTest(argv=argv), mock.patch.object(
                sys, "stdin", io.StringIO("")
            ):
                code, out = self.rk(*argv)
                self.assertEqual(code, 4, out)
                self.assertIn(
                    f"no configuration at {self.home / 'config.json'}",
                    " ".join(out.split()),
                )
        self.assertEqual(sorted(self.dir.rglob("*")), before)
        self.hand(settings=SETTINGS)
        cfg = config.load(check=False)
        self.assertEqual(registry.get(cfg, "repo"), "acme/widgets")

        def walk() -> dict:
            code, out = self.rk("logins", "--json")
            rows = {r["name"]: r for r in json.loads(out)}
            todo = any(r["state"] in ("MISSING", "WRONG") for r in rows.values())
            self.assertEqual(code, 1 if todo else 0, out)
            self.assertEqual(list(rows), [e.name for e in sr.walk_entries(cfg)])
            return rows

        self.assertNotEqual(walk()["admin_access_token"]["state"], "SET")
        value = a_value(sr.entry(cfg, "admin_access_token").shape, 5)
        with mock.patch.object(sys, "stdin", io.StringIO(value)):
            code, out = self.rk("secret-set", "admin_access_token")
        self.assertEqual(code, 0, out)
        self.assertNotIn(value.strip(), out)
        (kept,) = [p for p in secrets.rglob("*") if p.is_file()]
        self.assertEqual(kept.read_text(encoding="utf-8"), value.strip() + "\n")
        self.assertEqual(walk()["admin_access_token"]["state"], "SET")
        self.assertFalse((self.home / "config.json").exists())
        self.assertEqual(list(self.settings_dir.iterdir()), [])

    def test_the_confirm_reads_the_volumes_for_what_rk_logins_says_is_needed(self):
        """The confirm's rule and rk logins' agree: over settings that turn on each needed_when condition, the logins
        and secrets whose places the confirm reads before the first start (confirm.login_places, from the configuration
        your confirmed settings make) are those that rk logins, handed the same settings, says something needs (its
        rows with a needed_by)."""
        from test_confirm import GLM, read

        self.one_off()
        live = {
            "review_automation": {
                "mode": "live",
                "remote_gpu_tests": {"enabled": True},
            },
            "merge_queue": {"mode": "live", "auto_merge": {"enabled": True}},
            "alerts": {
                "mode": "live",
                "develop_breaks": {"enabled": True},
                "needs_you": {"enabled": True},
            },
            "ticket_creation": {
                "mode": "live",
                "follow_up_tickets": {"enabled": True},
                "found_issue_tickets": {"enabled": True},
            },
            "dashboards": {
                "team": {
                    "enabled": True,
                    "mode": "live",
                    "delivery": "publish",
                    "publish": {"web_host": "web.example.com"},
                }
            },
        }
        served = {"enabled": True, "mode": "live", "delivery": "serve"}
        drafts = {"passes": {"first_review_pass": {"model": "glm"}}}
        said, every = set(), set()
        for name, parts in (
            ("nothing that sends live", {}),
            (
                "everything live, a model binding's key",
                {
                    "features": dict(
                        live,
                        review_automation=dict(live["review_automation"], **drafts),
                    ),
                    "model_bindings": {"glm": GLM},
                },
            ),
            (
                "the team dashboard served",
                {"features": {"dashboards": {"team": served}}},
            ),
        ):
            made = read(**parts)
            places = {n for n, _, _ in confirm.login_places(made.run.values)}
            self.hand(data=deliver.texts(made.plan)[2].encode())
            code, out = self.rk("logins", "--json")
            rows = json.loads(out)
            with self.subTest(settings=name):
                self.assertEqual(
                    places, {r["name"] for r in rows if r["needed_by"]}, out
                )
            said.update(w for r in rows for w in r["needed_by"])
            every.update(
                sr.condition_words(c)
                for e in sr.walk_entries(made.run.values)
                for c in e.needed_when
            )
        self.assertEqual(every - said, set())  # each condition held in one of them

    def test_a_folder_or_file_that_is_not_root_s_alone_is_refused_in_plain_words(self):
        """A forged copy: the folder or the file a link, another user's, or one a user other than root may write;
        the folder a file, or the file a folder. The layer is not read, the refusal says which in plain words, and the
        fix is an image this package built, in whose own file system the root phase alone makes them. rk refuses as it
        does for any configuration that cannot be read, and the doctor fails with the fix.
        """
        real = self.dir / "handed.real"
        fstat = os.fstat

        def clear():
            for p in (self.handed, real):
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.exists():
                    shutil.rmtree(p)

        def others_file(fd):
            """os.fstat, but a regular file is another user's."""
            s = fstat(fd)
            if not stat.S_ISREG(s.st_mode):
                return s
            return os.stat_result(
                (s.st_mode, s.st_ino, s.st_dev, s.st_nlink, s.st_uid + 1, s.st_gid)
                + (s.st_size, int(s.st_atime), int(s.st_mtime), int(s.st_ctime))
            )

        def handed_link():
            self.hand()
            self.handed.rename(real)
            self.handed.symlink_to(real)

        def file_link():
            self.hand()
            self.path.rename(self.handed / "elsewhere.json")
            self.path.symlink_to(self.handed / "elsewhere.json")

        def folder_a_file():
            self.handed.write_text("{}", encoding="utf-8")

        def file_a_folder():
            self.handed.mkdir(0o750)
            self.path.mkdir(0o750)

        me = os.getuid()
        for make, patch, found in (
            (
                self.hand,
                mock.patch.object(RC, "ROOT_UID", me + 1),
                f"{self.handed} is not root's (user {me} owns it)",
            ),
            (
                self.hand,
                mock.patch.object(RC.os, "fstat", others_file),
                f"{self.path} is not root's (user {me + 1} owns it)",
            ),
            (
                lambda: (self.hand(), self.handed.chmod(0o770)),
                contextlib.nullcontext(),
                f"users other than root may write {self.handed} (permissions 770)",
            ),
            (
                lambda: (self.hand(), self.path.chmod(0o646)),
                contextlib.nullcontext(),
                f"users other than root may write {self.path} (permissions 646)",
            ),
            (handed_link, contextlib.nullcontext(), f"{self.handed} is a link"),
            (file_link, contextlib.nullcontext(), f"{self.path} is a link"),
            (folder_a_file, contextlib.nullcontext(), f"{self.handed} is no folder"),
            (file_a_folder, contextlib.nullcontext(), f"{self.path} is no file"),
        ):
            clear()
            make()
            with self.subTest(found=found), patch:
                with self.assertRaises(RC.HandedError) as caught:
                    RC.load(self.home)
                self.assertEqual(
                    str(caught.exception), f"{self.what} are not read: {found}"
                )
                self.assertEqual(
                    str(caught.exception.fix("rkmon")),
                    "run an image built by this package (QUICK_BUILD_GUIDE.md): set image.tag in your deploy "
                    "settings file to its tag, then rkmon confirm",
                )
                said = applied.cannot_run(caught.exception, "REFUSED")
                self.assertEqual(
                    self.rk("features", "--runs", "web"),
                    (4, "".join(f"{line}\n" for line in said)),
                )
                out = io.StringIO()
                with (
                    mock.patch.object(doctor, "prerequisites", return_value=[]),
                    contextlib.redirect_stdout(out),
                ):
                    self.assertEqual(doctor.main(), doctor.EXIT_CONFIG)
                text = " ".join(out.getvalue().split())
                self.assertIn(
                    f"FAIL configuration: {self.what} are not read: {found}", text
                )
                self.assertIn("set image.tag in your deploy settings file", text)

    def test_a_handed_layer_that_cannot_run_names_the_image_check(self):
        """A handed layer of another package, or one that is no JSON, cannot run, as a delivered one cannot,
        and the delivered layer does not run in its place: the fix is the image check, as the host's package makes the
        layer and the image's code reads it."""
        version = image.package_version(PACKAGE)
        self.layer(settings=SETTINGS)
        for data, found in (
            (
                json.dumps(self.a_layer(package_version="0.0.0")).encode(),
                f"cannot run: it was confirmed for the package '0.0.0', and this is the package {version!r}",
            ),
            (b"{", "are no JSON"),
        ):
            self.hand(data=data)
            with self.subTest(found=found):
                with self.assertRaises(RC.HandedError) as caught:
                    RC.load(self.home)
                said = " ".join(str(caught.exception).split())
                self.assertTrue(said.startswith(f"{self.what} "), said)
                self.assertIn(found, said)
                self.assertEqual(
                    str(caught.exception.fix("rkmon")), "rkmon image check"
                )


class TestTheGate(InContainer):
    """The outward gate (post.outward_gate; the writer runs rk, and so its gate) reads each feature's mode from the
    delivered layer: config.json gives no deploy key, so what it says there plays no part.
    """

    def test_the_layer_s_mode_is_the_gate_s_whatever_config_json_says(self):
        live = {"enabled": True, "mode": "live"}
        self.config_json(
            {"repo": "acme/widgets", "features": {"review_automation": live}}
        )
        prototype = dict(live, mode="prototype")
        self.layer(settings=dict(SETTINGS, features={"review_automation": prototype}))
        with self.assertRaises(post.PostRefused) as refused:
            post.outward_gate(config.load(), {}, "the review", post.POST_FEATURES)
        self.assertIn(
            "features.review_automation.mode is prototype: nothing goes out until it is live",
            str(refused.exception),
        )
        self.layer(settings=dict(SETTINGS, features={"review_automation": live}))
        self.assertIsNone(
            post.outward_gate(config.load(), {}, "the review", post.POST_FEATURES)
        )

    def test_each_kind_of_alert_goes_out_while_its_own_switch_in_the_layer_is_on(self):
        """Production's file: the develop-break alerts on and the Needs-you alerts off. Each kind's sender reads
        its own switch from the layer: the Needs-you message is refused, a develop-break alert is not.
        """
        alerts = {
            "mode": "live",
            "develop_breaks": {"enabled": True},
            "needs_you": {"enabled": False},
        }
        self.config_json({"repo": "acme/widgets"})
        self.layer(settings=dict(SETTINGS, features={"alerts": alerts}))
        cfg = config.load()
        self.assertIsNone(post.outward_gate(cfg, {}, "the alert", ("chat-alerts",)))
        with self.assertRaises(post.PostRefused) as refused:
            post.outward_gate(cfg, {}, "the Needs-you message", ("needs-you-alerts",))
        self.assertIn(
            "features.alerts.needs_you.enabled is true",
            str(refused.exception),
        )


class TestTheTickInterval(InContainer):
    """The minutes between ticks come from your deploy settings file alone: the delivered layer's
    monitor.tick_interval_minutes is what `rk tick-min` prints, which the entry point's tick loop takes, and the
    orchestrator's ticks; a variable of the environment (RKMON_TICK_MIN) changes nothing.
    """

    def test_the_tick_interval_is_the_file_s(self):
        self.layer(settings=dict(SETTINGS, monitor={"tick_interval_minutes": 7}))
        with mock.patch.dict(os.environ, {"RKMON_TICK_MIN": "3"}):
            self.assertEqual(self.rk("tick-min"), (0, "7\n"))
            self.assertEqual(orchestrator.settings(config.load())["tick_min"], 7)


class TestBrake(InContainer):
    """The emergency brake (rkmon.py BRAKE: rk stop, kill, undrain and stopped) reads only what names the folders, the
    container's layout (config.folders; config.json is not read once a layer is delivered), never the internal
    state or the delivered layer: neither keeps it from stopping the monitor, while rk tick-min, which the entry point
    reads, refuses both. rk stop --drain drains with the whole configuration; when it cannot be loaded, no review in
    flight can finish, and the stop says NOT DRAINED, why and the fix, and goes at once.
    """

    def setUp(self):
        super().setUp()
        # No PR runner (prcode.stop_all), whatever this machine has, and a configuration folder of the test's own.
        env = {k: v for k, v in os.environ.items() if k != slots.RUNNER_VAR}
        env["XDG_CONFIG_HOME"] = str(self.dir / "xdg")
        for patch in (
            mock.patch.dict(os.environ, env, clear=True),
            # The container's working folder, the test's own.
            mock.patch.object(container_layout, "WORK_DIR", str(self.dir / "work")),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.config_json(
            {"repo": "acme/widgets", "paths": {"working_dir": str(self.dir / "work")}}
        )
        self.layer(settings={"repository": {"name": "acme/gadgets"}})

    def broken_state(self) -> None:
        path = state_store.path(self.home)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"kind": "internal_state", "schema_version": 1}
        body["jira"] = {"accounts": {"bad login.x": 5}}
        path.write_text(json.dumps(body), encoding="utf-8")

    def brakes(self, broken) -> None:
        """BROKEN keeps the configuration from loading; each brake command stops the monitor all the same."""
        broken()
        with self.assertRaises(config.ConfigError) as caught:
            config.load(check=False)
        exc = caught.exception
        refused = "".join(f"{line}\n" for line in applied.cannot_run(exc, "REFUSED"))
        self.assertEqual(
            self.rk("tick-min"), (4, refused)
        )  # the entry point runs no tick
        self.assertEqual(self.rk("stopped"), (1, "not stopped\n"))
        code, out = self.rk("stop")
        self.assertEqual(code, 0, out)
        self.assertTrue(
            out.splitlines()[-1].startswith("STOPPED: 0 process(es); "), out
        )
        code, out = self.rk("stopped")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("stopped by rk stop at "), out)
        code, out = self.rk("kill", "--wait", "0")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.splitlines()[-1].startswith("KILLED 0 process(es) "), out)
        self.assertTrue(self.rk("stopped")[1].startswith("stopped by rk kill at "))
        state = store.Store()
        with state.locked(wait=True):
            st = state.load()
            st["draining"] = {"why": drain.WHY, "since": store.now(), "until_min": 1}
            state.save(st)
        code, out = self.rk("undrain")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith(f"{drain.UNDRAINED}: the drain since "), out)
        self.assertNotIn("draining", state.load())
        code, out = self.rk("stop", "--drain")
        self.assertEqual(code, 0, out)
        lines = applied.cannot_run(exc, drain.NOT_DRAINED)
        said = "".join(f"{line}\n" for line in lines)
        self.assertTrue(out.startswith(said), out)
        self.assertIn(f"\n{checks.FIX_LABEL}", said)  # the fix the error names
        self.assertNotIn("DRAINING", out)
        self.assertTrue(
            out.splitlines()[-1].startswith("STOPPED: 0 process(es); "), out
        )
        self.assertTrue(self.rk("stopped")[1].startswith("stopped by rk stop at "))

    def test_a_state_that_breaks_its_schema_never_keeps_the_brake_from_stopping(self):
        self.brakes(self.broken_state)

    def test_a_layer_that_cannot_run_never_keeps_the_brake_from_stopping(self):
        self.brakes(lambda: self.layer(package_version="0.0.0"))

    def test_a_layer_that_is_a_link_to_no_file_is_delivered_and_config_json_never_read(
        self,
    ):
        """A link to no file counts as delivered (runtime_config.delivered, the one test of it): the load stops on it,
        with its fix, and nothing falls back to config.json, whose repository is not the folders'.
        """
        layer = self.settings_dir / RC.LAYER_FILE

        def dangle() -> None:
            layer.unlink()
            layer.symlink_to(self.settings_dir / "no-such-layer.json")

        self.brakes(dangle)
        self.assertTrue(RC.delivered())
        with self.assertRaises(RC.LayerError) as caught:
            config.load(check=False)
        self.assertEqual(
            str(caught.exception),
            f"your confirmed settings at {layer} cannot be read: it is a link to a file that is not there",
        )
        self.assertEqual(
            str(caught.exception.fix("rkmon")), str(RC.LayerError.fix("rkmon"))
        )
        self.assertEqual(
            registry.get(config.folders(), "repo"), registry.default("repo")
        )

    def test_a_config_json_beside_the_layer_is_never_read(self):
        """It is the one-time move's to split: one that is no JSON object keeps neither the configuration nor
        the folders from loading."""
        self.config_json([])
        self.assertEqual(registry.get(config.load(check=False), "repo"), "acme/gadgets")
        self.assertEqual(
            registry.get(config.folders(), "paths.working_dir"), str(self.dir / "work")
        )

    def test_the_drain_runs_with_the_whole_configuration_and_the_rest_with_the_folders(
        self,
    ):
        whole, folders = config.load(check=False), config.folders()
        self.assertEqual(
            (registry.get(whole, "repo"), registry.get(folders, "repo")),
            # the layer's, then none: the folders read neither the layer nor config.json beside it
            ("acme/gadgets", registry.default("repo")),
        )
        self.assertEqual(
            registry.get(folders, "paths.working_dir"), str(self.dir / "work")
        )
        for argv, cfg, drains in (
            (["stop", "--drain"], whole, True),
            (["stop"], folders, False),
        ):
            with mock.patch.object(control, "stop", return_value=0) as stop:
                self.assertEqual(self.rk(*argv), (0, ""), argv)
            self.assertEqual(stop.call_args.args[0], cfg, argv)
            self.assertEqual(stop.call_args.kwargs["drain"], drains, argv)


class TestFixes(Base):
    """Each error of the configuration's load names its fix (config.ConfigError.fix), and the doctor shows that fix
    alone: setup where there is no configuration; a config.json that is no configuration mended, or moved aside for
    setup to write it anew; the internal state's own (tests/test_state_store.py); setup again, or the settings
    edited, where the configuration cannot run."""

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def doctor_says(self) -> str:
        out = io.StringIO()
        with (
            mock.patch.object(doctor, "prerequisites", return_value=[]),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(doctor.main(), doctor.EXIT_CONFIG)
        return " ".join(out.getvalue().split())

    def test_each_error_of_the_load_names_its_fix_and_the_doctor_shows_it_alone(self):
        path, state = self.home / "config.json", state_store.path(self.home)
        where = settings_file.shown(path)
        anew = (
            f"move it aside: mv {where} {where}.old, then rkmon confirm and rkmon apply"
        )
        broken = {"kind": "internal_state", "schema_version": 1, "standby": {"on": 1}}
        for text, state_body, fix in (
            (None, None, "rkmon confirm and rkmon apply"),
            ("{", None, f"mend it at the line and column above, or {anew}"),
            ("[]", None, anew),
            (
                '{"repo": "acme/widgets"}',
                broken,
                f"delete standby.on from {settings_file.shown(state)}",
            ),
        ):
            if text is not None:
                path.write_text(text, encoding="utf-8")
            if state_body is not None:
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps(state_body), encoding="utf-8")
            with self.assertRaises(config.ConfigError) as e:
                config.load(check=False)
            self.assertEqual(str(e.exception.fix("rkmon")), fix, text)
            said = self.doctor_says()
            why = " ".join(str(e.exception).split())
            self.assertTrue(said.startswith(f"FAIL configuration: {why} "), said)
            self.assertIn(f"{checks.FIX_LABEL}{fix}", said)
            self.assertEqual(said.count(checks.FIX_LABEL), 1, said)

    def test_rk_features_says_why_nothing_runs_and_the_fix(self):
        """`rk features`, the setup summary, while the configuration cannot be loaded: with none yet, every optional
        feature is off, then the fix and the summary of the defaults, exit 0; with one that cannot be read, nothing
        runs, why and the fix, exit 1. `rk features --runs` refuses both, exit 4."""
        path = self.home / "config.json"
        for text, lead, code in (
            (None, "Every optional feature is off", 0),
            ("{", "Nothing runs", 1),
        ):
            if text is not None:
                path.write_text(text, encoding="utf-8")
            with self.assertRaises(config.ConfigError) as e:
                config.load(check=False)
            said = "".join(
                f"{line}\n" for line in applied.cannot_run(e.exception, lead)
            )
            self.assertIn(f"\n{checks.FIX_LABEL}", said)
            refused = applied.cannot_run(e.exception, "REFUSED")
            with mock.patch.object(rkmon, "cmd_features", return_value=0) as summary:
                self.assertEqual(
                    self.rk("features"), (code, said + ("\n" if code == 0 else ""))
                )
                self.assertEqual(
                    self.rk("features", "--runs", "web"),
                    (4, "".join(f"{line}\n" for line in refused)),
                )
            self.assertEqual(
                summary.call_count, 1 - code, text
            )  # the defaults' summary

    def test_the_drafter_the_quality_lane_and_the_ci_head_report_refuse_it(self):
        """A configuration they cannot load: REFUSED, why and the fix, exit 4, as every rk command says it
        (applied.refuse), never a traceback."""
        paths = [str(ROOT / "worker"), str(ROOT / "tools")]
        sys.path[:0] = paths
        self.addCleanup(lambda: [sys.path.remove(p) for p in paths])
        import ci_head
        import drafter
        import quality_lane

        (self.home / "config.json").write_text("{", encoding="utf-8")
        with self.assertRaises(config.ConfigError) as e:
            config.load()
        said = "".join(
            f"{line}\n" for line in applied.cannot_run(e.exception, "REFUSED")
        )
        for main, argv in (
            (drafter.main, ["ws", "1"]),
            (quality_lane.main, ["ws", "1"]),
            (ci_head.main, ["0" * 40]),
        ):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(argv), 4)
            self.assertEqual(out.getvalue(), said)

    def test_a_configuration_that_cannot_run_names_the_settings_edited_then_applied(
        self,
    ):
        self.config_json({"repo": "acme/widgets"})
        fix = "fix it in your deploy settings file, then rkmon confirm and rkmon apply"
        self.assertEqual(str(doctor.cannot_run_fix()), fix)
        with self.assertRaises(config.ConfigError) as e:
            config.load()
        self.assertEqual(str(e.exception.fix("rkmon")), fix)
        self.assertNotIn(
            "rkmon confirm", str(e.exception)
        )  # the fix is the error's own, not in its words
        with self.assertRaises(config.ConfigError) as e:
            RC.record_applied(self.home, RC.Resolved({}, {}, ["a problem"]))
        self.assertEqual(str(e.exception.fix("rkmon")), fix)


class TestRepositoryDefaults(Base):
    """The repository defaults for the configuration's repository and folder set their package keys, over
    config.defaults.json's; config.json's value of one of them runs only when it is its own: neither blank nor
    config.defaults.json's."""

    def configured(self, changes=None) -> dict:
        """config.json with config.defaults.json's value of every package key, for the home repository and its
        folder, with CHANGES (dotted)."""
        raw = config.load_defaults()
        answers = dict(
            ANSWERS, repo=home_repository.REPOSITORY, rocke_path=home_repository.FOLDER
        )
        for name, value in dict(answers, **(changes or {})).items():
            registry.put(raw, name, value)
        return raw

    def test_the_defaults_for_the_repository_and_its_folder_set_their_keys(self):
        names = repository_defaults.keys_of(home_repository.VALUES)
        self.assertIn("gpu.base_env", names)
        self.config_json(self.configured())
        loaded = self.load()
        for name in names:
            self.assertEqual(
                self.pair(loaded, name),
                (registry.get(home_repository.VALUES, name), RC.REPOSITORY),
            )
        # The repository whatever its case, through a delivered layer's settings too.
        settings = {
            "repository": {
                "name": home_repository.REPOSITORY.upper(),
                "folder": home_repository.FOLDER,
            }
        }
        shown = RC.from_settings(settings)
        for name in names:
            self.assertEqual(
                self.pair(shown, name),
                (registry.get(home_repository.VALUES, name), RC.REPOSITORY),
            )

    def test_a_value_of_its_own_in_config_json_runs_and_a_blank_or_generic_one_does_not(
        self,
    ):
        pattern = registry.get(home_repository.VALUES, "develop_rocke_job_pattern")
        self.config_json(
            self.configured(
                {
                    "ci_absent_modules": ["mine"],
                    "develop_rocke_job_pattern": "",
                    "gpu.base_env": {},
                }
            )
        )
        loaded = self.load()
        self.assertEqual(
            self.pair(loaded, "ci_absent_modules"), (["mine"], RC.CONFIG_JSON)
        )
        self.assertEqual(
            self.pair(loaded, "develop_rocke_job_pattern"), (pattern, RC.REPOSITORY)
        )
        self.assertEqual(
            self.pair(loaded, "gpu.base_env"),
            ({"ROCKE_BACKEND": "python"}, RC.REPOSITORY),
        )
        # A key no repository defaults set keeps config.json's value, whatever it is.
        self.config_json(
            self.configured(
                {"loop_heartbeat_min": registry.default("loop_heartbeat_min")}
            )
        )
        self.assertEqual(
            self.pair(self.load(), "loop_heartbeat_min")[1], RC.CONFIG_JSON
        )

    def test_another_folder_or_repository_takes_the_generic_values(self):
        generic = registry.package_defaults()
        for repo, folder in (
            (home_repository.REPOSITORY, "other/folder"),
            ("acme/widgets", home_repository.FOLDER),
        ):
            with self.subTest(repo=repo, folder=folder):
                self.assertEqual(
                    RC.repository_for(None, {"repo": repo, "rocke_path": folder}), {}
                )
                self.config_json(self.configured({"repo": repo, "rocke_path": folder}))
                loaded = self.load()
                for name in repository_defaults.keys_of(home_repository.VALUES):
                    self.assertEqual(
                        self.pair(loaded, name),
                        (registry.get(generic, name), RC.CONFIG_JSON),
                    )
                shown = RC.from_settings(
                    {"repository": {"name": repo, "folder": folder}}
                )
                for name in repository_defaults.keys_of(home_repository.VALUES):
                    self.assertEqual(
                        self.pair(shown, name),
                        (registry.get(generic, name), RC.PACKAGE_SOURCE),
                    )
        self.assertEqual(
            RC.repository_for(
                None,
                {
                    "repo": home_repository.REPOSITORY,
                    "rocke_path": home_repository.FOLDER,
                },
            ),
            home_repository.VALUES,
        )


if __name__ == "__main__":
    unittest.main()
