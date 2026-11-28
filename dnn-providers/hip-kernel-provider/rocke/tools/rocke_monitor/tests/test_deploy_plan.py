# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What confirmed deploy settings make (rkmon_lib/deploy_plan.py): each profile's docker/.env,
override and layer, golden; the layer's text, the same for the same settings; the team dashboard's port only while
the container serves it; the RAM disk; the ids a blank key stands for; the fingerprint's inputs and what it leaves
out, the package's version and the SHA-256 of its code among its inputs; the keys the file sets; the plan made
again from a delivered layer; the file's own SHA-256; and the documents the plan refuses.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (  # noqa: E402
    deploy_plan,
    deploy_tree,
    file_types,
    runtime_config,
    settings_file,
    webserve,
)

TAG = "1.2.3"
PACKAGE = "1.2.3"
CODE = "sha256:" + "c" * 64  # the SHA-256 of the package's code (image.code_sha256)
HOST = deploy_plan.Host(uid=1234, gid=1235, confirmed_at="2026-01-02T03:04:05Z")
PROFILES = ("full", "review-only", "merge-queue")
TEAM = "features.dashboards.team"
SERVED = {
    f"{TEAM}.enabled": True,
    f"{TEAM}.mode": "live",
    f"{TEAM}.delivery": "serve",
    f"{TEAM}.serve.address": "0.0.0.0",
    f"{TEAM}.serve.port": 18700,
}
FAST = {"provider": "claude", "model_id": "claude-haiku-5-5"}

ENV = """\
# docker/.env: the values docker compose reads for this deployment, made from your confirmed deploy settings.
# rkmon writes it at the deployment's first start, and at each rkmon apply and rkmon rollback (the previous file
# kept as .env.prev).
# Change your deploy settings file instead, then run rkmon confirm and rkmon apply: an edit here shows as
# "changed outside your file" in rkmon status, and rkmon apply puts it back.
COMPOSE_PROJECT_NAME=rocke-monitor
RKMON_IMAGE=rocke-monitor
RKMON_TAG=1.2.3
RKMON_ADMIN_HOST=127.0.0.1
RKMON_PORT=18614
RKMON_CPUS=4
RKMON_MEMORY=8g
RKMON_UID=1234
RKMON_GID=1235
"""

OVERRIDE = """\
# docker/compose.override.yaml: what docker compose adds to compose.yaml for this deployment, made from your
# confirmed deploy settings: the team dashboard's port while the container serves it, and the RAM disk, a tmpfs
# that counts against the container's memory limit. rkmon writes it with docker/.env (the previous file kept as
# compose.override.yaml.prev).
# Change your deploy settings file instead, then run rkmon confirm and rkmon apply: an edit here shows as
# "changed outside your file" in rkmon status, and rkmon apply puts it back.
services:
  rkmon:
    ports:
      - target: 18615
        published: "${RKMON_TEAM_PORT}"
        host_ip: "${RKMON_TEAM_HOST}"
    tmpfs:
      - /ramdisk:size=3g,mode=1777,exec
"""


@functools.lru_cache(maxsize=None)
def _template(profile: str) -> settings_file.Document:
    return settings_file.loads(
        settings_file.template(file_types.DEPLOY, profile), file_types.DEPLOY
    )


def doc(changes=None, profile="full") -> settings_file.Document:
    """PROFILE's starting template, its image tag TAG, with CHANGES (dotted key -> value; None blanks the key)."""
    base = _template(profile)
    values = copy.deepcopy(base.values)
    for key, value in {"image.tag": TAG, **(changes or {})}.items():
        deploy_tree.put(values, key, value)
    return dataclasses.replace(base, values=values)


def plan(changes=None, profile="full", package=PACKAGE, host=HOST, code=CODE):
    return deploy_plan.plan(doc(changes, profile), package, code, host)


def effective_without_updates(d) -> dict:
    values = settings_file.effective(d)
    del values["updates"]
    return values


class TestProfiles(unittest.TestCase):
    """Each starting template's three files, golden."""

    def test_env(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                self.assertEqual(plan(profile=profile).env, ENV)

    def test_no_override(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                self.assertEqual(plan(profile=profile).override, "")

    def test_layer(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                d = doc(profile=profile)
                p = deploy_plan.plan(d, PACKAGE, CODE, HOST)
                layer = dict(p.layer)
                settings = layer.pop("settings")
                set_keys = layer.pop("set_in_file")
                self.assertEqual(
                    layer,
                    {
                        "kind": "confirmed_deploy_settings",
                        "schema_version": 1,
                        "deploy_schema_version": 2,
                        "package_version": PACKAGE,
                        "code_sha256": CODE,
                        "confirmed_at": "2026-01-02T03:04:05Z",
                        "fingerprint": p.fingerprint,
                        "container": {
                            "project": "rocke-monitor",
                            "ports": {"admin": 18614},
                            "address": "127.0.0.1",
                            "image": {
                                "registry": "",
                                "name": "rocke-monitor",
                                "tag": TAG,
                            },
                        },
                    },
                )
                self.assertEqual(settings, effective_without_updates(d))
                self.assertIn("updates", deploy_tree.parts())
                self.assertEqual(set(settings), set(deploy_tree.parts()) - {"updates"})
                # the profile's own values are the layer's, each set in the file
                (made,) = [
                    x
                    for x in settings_file.profiles(file_types.DEPLOY)
                    if x.name == profile
                ]
                for key, value in made.values.items():
                    self.assertEqual(deploy_tree.get(settings, key), value, key)
                    if not settings_file.blank(value):
                        self.assertIn(key, set_keys)
                # each key set is one the file gives a value
                self.assertTrue(set_keys)
                for key in set_keys:
                    self.assertFalse(
                        settings_file.blank(deploy_tree.get(d.values, key)), key
                    )

    def test_layer_text(self):
        p = plan()
        text = deploy_plan.layer_text(p.layer)
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(json.loads(text), p.layer)
        # a whole number given as a float is written whole
        self.assertEqual(
            deploy_plan.layer_text(plan({"host.resources.cpus": 4.0}).layer), text
        )

    def test_layer_text_whatever_the_order_of_a_maps_entries(self):
        a, b = "model_bindings.a", "model_bindings.b"
        swapped = (plan({a: FAST, b: FAST}), plan({b: FAST, a: FAST}))
        orders = [list(x.layer["settings"]["model_bindings"]) for x in swapped]
        self.assertEqual(orders, [["a", "b"], ["b", "a"]])
        texts = {deploy_plan.layer_text(x.layer) for x in swapped}
        self.assertEqual(len(texts), 1)
        self.assertEqual(swapped[0].fingerprint, swapped[1].fingerprint)

    def test_variables_are_the_env_lines(self):
        p = plan(SERVED)
        lines = [x for x in p.env.splitlines() if not x.startswith("#")]
        self.assertEqual(lines, [f"{k}={v}" for k, v in p.variables.items()])
        self.assertEqual(list(p.variables), [k for k in deploy_plan.ENV_KEYS])


class TestEnv(unittest.TestCase):
    def test_what_goes_in_from_the_settings(self):
        p = plan(
            {
                "container.name": "rkmon-two",
                "image.registry": "registry.example.com/team",
                "image.name": "monitor",
                "image.tag": "9.9",
                "features.dashboards.admin.port": 19000,
                "host.resources.cpus": 2.5,
                "host.resources.memory_gib": 12,
            }
        )
        self.assertEqual(p.variables["COMPOSE_PROJECT_NAME"], "rkmon-two")
        self.assertEqual(p.variables["RKMON_IMAGE"], "monitor")
        self.assertEqual(p.variables["RKMON_TAG"], "9.9")
        self.assertEqual(p.variables["RKMON_PORT"], "19000")
        self.assertEqual(p.variables["RKMON_CPUS"], "2.5")
        self.assertEqual(p.variables["RKMON_MEMORY"], "12g")
        self.assertEqual(
            p.layer["container"],
            {
                "project": "rkmon-two",
                "ports": {"admin": 19000},
                "address": "127.0.0.1",
                "image": {
                    "registry": "registry.example.com/team",
                    "name": "monitor",
                    "tag": "9.9",
                },
            },
        )

    def test_a_whole_number_is_written_whole(self):
        p = plan({"host.resources.cpus": 6.0, "host.resources.memory_gib": 16.0})
        self.assertEqual(p.variables["RKMON_CPUS"], "6")
        self.assertEqual(p.variables["RKMON_MEMORY"], "16g")

    def test_blank_ids_are_this_hosts(self):
        p = plan({"host.users.monitor_uid": None, "host.users.monitor_gid": None})
        self.assertEqual(
            (p.variables["RKMON_UID"], p.variables["RKMON_GID"]), ("1234", "1235")
        )

    def test_ids_set_are_the_files(self):
        p = plan({"host.users.monitor_uid": 2000, "host.users.monitor_gid": 3000})
        self.assertEqual(
            (p.variables["RKMON_UID"], p.variables["RKMON_GID"]), ("2000", "3000")
        )

    def test_no_variable_of_the_earlier_files(self):
        for name in (
            "RKMON_SETTINGS",
            "RKMON_CONFIRMED",
            "RKMON_WRITE_UID",
            "RKMON_PR_UID",
        ):
            self.assertNotIn(name, plan(SERVED).env)

    def test_every_variable_is_named_with_its_keys(self):
        schema_keys = {
            path
            for path, _node, _required in settings_file.schema_keys(
                file_types.DEPLOY.load()
            )
        }
        named = [k for keys in deploy_plan.ENV_KEYS.values() for k in keys]
        for key in named + list(deploy_plan.OVERRIDE_KEYS):
            self.assertIn(key, schema_keys)
        self.assertEqual(set(plan(SERVED).variables), set(deploy_plan.ENV_KEYS))

    def test_what_docker_reads_takes_a_new_container(self):
        # docker reads these when it makes the container: none takes effect without a new one; the
        # container's name is no class
        root = file_types.DEPLOY.load()
        named = [k for keys in deploy_plan.ENV_KEYS.values() for k in keys]
        for key in set(named + list(deploy_plan.OVERRIDE_KEYS)):
            applies = settings_file.schema_at(root, key).get("x-applies")
            if key == "container.name":
                self.assertIsNone(applies)
            else:
                self.assertEqual(applies, "new_container", key)

    def test_headers(self):
        for line in deploy_plan.ENV_HEADER + deploy_plan.OVERRIDE_HEADER:
            self.assertTrue(line.startswith("# "), line)
            self.assertLessEqual(len(line), settings_file.WIDTH, line)


class TestTeamDashboard(unittest.TestCase):
    def test_served_opens_its_port(self):
        p = plan(SERVED)
        self.assertEqual(p.variables["RKMON_TEAM_HOST"], "0.0.0.0")
        self.assertEqual(p.variables["RKMON_TEAM_PORT"], "18700")
        self.assertIn(
            "    ports:\n"
            "      - target: 18615\n"
            '        published: "${RKMON_TEAM_PORT}"\n'
            '        host_ip: "${RKMON_TEAM_HOST}"\n',
            p.override,
        )
        self.assertNotIn("    tmpfs:", p.override)
        self.assertEqual(p.layer["container"]["ports"], {"admin": 18614, "team": 18700})

    def test_an_ipv6_address(self):
        p = plan({**SERVED, f"{TEAM}.serve.address": "::1"})
        self.assertEqual(p.variables["RKMON_TEAM_HOST"], "::1")
        self.assertIn('host_ip: "${RKMON_TEAM_HOST}"', p.override)

    def test_served_in_prototype_too(self):
        """In prototype the team dashboard's page is made and served as in live: its port opens."""
        p = plan({**SERVED, f"{TEAM}.mode": "prototype"})
        self.assertTrue(deploy_plan.served(p.layer["settings"]))
        self.assertEqual(p.variables["RKMON_TEAM_PORT"], "18700")
        self.assertEqual(p.layer["container"]["ports"], {"admin": 18614, "team": 18700})

    def test_in_prototype_the_port_opens_on_this_host_only(self):
        """Check 7: in prototype the page is this host's only, whatever serve.address says; live, the
        address is the file's."""
        for address in ("0.0.0.0", "::", "10.1.2.3"):
            with self.subTest(address=address):
                served = {**SERVED, f"{TEAM}.serve.address": address}
                p = plan({**served, f"{TEAM}.mode": "prototype"})
                self.assertEqual(p.variables["RKMON_TEAM_HOST"], "127.0.0.1")
                self.assertIn('host_ip: "${RKMON_TEAM_HOST}"', p.override)
                self.assertEqual(plan(served).variables["RKMON_TEAM_HOST"], address)

    def test_not_served(self):
        for why, changes in (
            ("published", {f"{TEAM}.delivery": "publish"}),
            ("off", {f"{TEAM}.enabled": False}),
        ):
            with self.subTest(why=why):
                p = plan({**SERVED, **changes})
                self.assertFalse(deploy_plan.served(p.layer["settings"]))
                self.assertNotIn("RKMON_TEAM_HOST", p.variables)
                self.assertNotIn("RKMON_TEAM_PORT", p.variables)
                self.assertEqual(p.override, "")
                self.assertEqual(p.layer["container"]["ports"], {"admin": 18614})

    def test_the_container_s_own_check_wants_the_same_values(self):
        """The container compares the variables docker compose gave it with the settings it runs, the delivered
        layer's (webserve.publish_env, the doctor's row): it wants the plan's own, the team dashboard's while served,
        in prototype on this host's own address (features.team_host).
        """
        prototype = {**SERVED, f"{TEAM}.mode": "prototype"}
        for changes in (
            {},
            SERVED,
            prototype,
            {"features.dashboards.admin.port": 19000},
        ):
            with self.subTest(changes=changes):
                p = plan(changes)
                cfg = runtime_config.from_settings(p.layer["settings"]).values
                wanted = {name: want for name, want, _ in webserve.publish_env(cfg)}
                self.assertEqual({name: p.variables[name] for name in wanted}, wanted)
                ports = {"RKMON_ADMIN_HOST", "RKMON_PORT"}
                if deploy_plan.served(p.layer["settings"]):
                    ports |= {"RKMON_TEAM_HOST", "RKMON_TEAM_PORT"}
                self.assertEqual(set(wanted), ports)


class TestRamDisk(unittest.TestCase):
    def test_none(self):
        self.assertEqual(plan({deploy_plan.RAMDISK_GIB: 0}).override, "")

    def test_whole(self):
        p = plan({deploy_plan.RAMDISK_GIB: 6})
        self.assertTrue(
            p.override.endswith("    tmpfs:\n      - /ramdisk:size=6g,mode=1777,exec\n")
        )
        self.assertNotIn("    ports:", p.override)

    def test_a_part_rounds_up(self):
        p = plan({deploy_plan.RAMDISK_GIB: 0.25})
        self.assertIn("/ramdisk:size=1g,", p.override)

    def test_with_the_team_port(self):
        self.assertEqual(
            plan({**SERVED, deploy_plan.RAMDISK_GIB: 2.5}).override, OVERRIDE
        )


class TestFingerprint(unittest.TestCase):
    def test_its_inputs(self):
        d = doc()
        self.assertEqual(
            deploy_plan.fingerprint(d, PACKAGE, CODE),
            settings_file.fingerprint(
                {
                    "deploy_schema_version": 2,
                    "package_version": PACKAGE,
                    "code_sha256": CODE,
                    "settings": effective_without_updates(d),
                }
            ),
        )
        p = deploy_plan.plan(d, PACKAGE, CODE, HOST)
        self.assertEqual(p.fingerprint, deploy_plan.fingerprint(d, PACKAGE, CODE))
        self.assertEqual(p.layer["fingerprint"], p.fingerprint)
        self.assertRegex(p.fingerprint, r"^sha256:[0-9a-f]{64}$")

    def test_what_changes_it(self):
        base = plan().fingerprint
        self.assertNotEqual(plan(package="1.2.4").fingerprint, base)
        # other code under the same version, such as a later commit of updates.branch
        self.assertNotEqual(plan(code="sha256:" + "d" * 64).fingerprint, base)
        self.assertNotEqual(plan({"host.resources.cpus": 5}).fingerprint, base)
        self.assertNotEqual(plan({"image.tag": "1.2.4"}).fingerprint, base)

    def test_what_it_leaves_out(self):
        base = plan().fingerprint
        for why, p in (
            (
                "updates",
                plan(
                    {
                        "updates.repository": "<org>/<repo>",
                        "updates.branch": "main",
                    }
                ),
            ),
            (
                "the confirm's time",
                plan(host=HOST._replace(confirmed_at="2027-01-01T00:00:00Z")),
            ),
            (
                "the ids a blank stands for",
                plan(host=HOST._replace(uid=4321, gid=4322)),
            ),
            ("a blank for its default", plan({"host.resources.cpus": None})),
            ("a whole number as a float", plan({"host.resources.cpus": 4.0})),
        ):
            with self.subTest(why=why):
                self.assertEqual(p.fingerprint, base)

    def test_a_set_id_is_in_it(self):
        base = plan().fingerprint
        self.assertNotEqual(plan({"host.users.monitor_uid": 2000}).fingerprint, base)


class TestSetInFile(unittest.TestCase):
    """The keys of the layer's settings your file sets, which rk config show tells from the defaults: each key
    the file gives a value that is not blank, a map one key, in the schema's order; a section left out or empty sets
    none, and `updates` is left out. Which keys they are is no part of the fingerprint.
    """

    def test_a_key_is_set_when_the_file_gives_it_a_value(self):
        p, blank = plan(), plan({"host.resources.cpus": None})
        self.assertIn("host.resources.cpus", p.layer["set_in_file"])  # at its default
        self.assertNotIn("host.resources.cpus", blank.layer["set_in_file"])
        self.assertEqual(  # it runs as it did
            (blank.layer["settings"], blank.fingerprint),
            (p.layer["settings"], p.fingerprint),
        )

    def test_a_section_left_out_or_empty_sets_none(self):
        values = copy.deepcopy(_template("full").values)
        del values["features"]["ticket_creation"]
        values["integrations"]["atlassian_cli"] = {}
        d = dataclasses.replace(_template("full"), values=values)
        p = deploy_plan.plan(d, PACKAGE, CODE, HOST)
        for section in ("features.ticket_creation", "integrations.atlassian_cli"):
            with self.subTest(section=section):
                inside = [
                    k
                    for k in p.layer["set_in_file"]
                    if k == section or k.startswith(f"{section}.")
                ]
                self.assertEqual(inside, [])
                self.assertTrue(deploy_tree.get(p.layer["settings"], section))

    def test_a_map_is_one_key(self):
        set_keys = plan({"model_bindings.fast": FAST}).layer["set_in_file"]
        self.assertIn("model_bindings", set_keys)
        self.assertEqual([k for k in set_keys if k.startswith("model_bindings.")], [])

    def test_updates_are_left_out(self):
        p = plan({"updates.repository": "<org>/<repo>", "updates.branch": "main"})
        self.assertEqual(
            [k for k in p.layer["set_in_file"] if k.split(".")[0] == "updates"], []
        )

    def test_in_the_schema_s_order_whatever_the_file_s(self):
        d = doc()
        turned = dataclasses.replace(d, values=dict(reversed(d.values.items())))
        self.assertEqual(deploy_plan.set_in_file(turned), deploy_plan.set_in_file(d))
        parts = list(file_types.DEPLOY.load()["properties"])
        first = [k.split(".")[0] for k in deploy_plan.set_in_file(d)]
        self.assertEqual(first, sorted(first, key=parts.index))


class TestFromLayer(unittest.TestCase):
    """The plan made again from a delivered layer, so that a hand edit is told from what deliver wrote."""

    def again(self, p, uid=HOST.uid, gid=HOST.gid):
        layer = json.loads(deploy_plan.layer_text(p.layer))
        return deploy_plan.from_layer(layer, uid, gid)

    def test_the_same_plan(self):
        for changes in (None, {deploy_plan.RAMDISK_GIB: 2.5}, SERVED):
            with self.subTest(changes=changes):
                p = plan(changes)
                again = self.again(p)
                self.assertEqual(
                    (again.env, again.override, again.fingerprint, again.variables),
                    (p.env, p.override, p.fingerprint, p.variables),
                )
                self.assertEqual(
                    deploy_plan.layer_text(again.layer), deploy_plan.layer_text(p.layer)
                )

    def test_on_the_ids_it_is_given(self):
        again = self.again(plan(), 4321, 4322)
        ids = (again.variables["RKMON_UID"], again.variables["RKMON_GID"])
        self.assertEqual(ids, ("4321", "4322"))

    def test_what_the_layer_holds(self):
        layer = json.loads(deploy_plan.layer_text(plan().layer))
        held = dict(
            deploy_schema_version=1,
            package_version="1.2.4",
            code_sha256="sha256:" + "d" * 64,
            confirmed_at="2027-01-01T00:00:00Z",
        )
        again = deploy_plan.from_layer({**layer, **held}, HOST.uid, HOST.gid)
        self.assertEqual({k: again.layer[k] for k in held}, held)
        self.assertNotEqual(again.fingerprint, layer["fingerprint"])

    def test_a_layer_whose_settings_were_edited(self):
        p = plan()
        layer = json.loads(deploy_plan.layer_text(p.layer))
        layer["settings"]["host"]["resources"]["cpus"] = 6
        again = deploy_plan.from_layer(layer, HOST.uid, HOST.gid)
        self.assertEqual(
            again.fingerprint, plan({"host.resources.cpus": 6}).fingerprint
        )
        self.assertNotEqual(again.fingerprint, layer["fingerprint"])
        self.assertNotEqual(
            deploy_plan.layer_text(again.layer), deploy_plan.layer_text(layer)
        )


class TestFileSha256(unittest.TestCase):
    def test_the_files_own_values(self):
        base = deploy_plan.file_sha256(doc())
        self.assertEqual(
            base,
            settings_file.fingerprint(
                {k: v for k, v in doc().values.items() if k != "updates"}
            ),
        )
        # a default written out is a change of your file, though the settings are the same
        written = doc({"host.resources.cpus": 4})
        blank = doc({"host.resources.cpus": None})
        self.assertNotEqual(
            deploy_plan.file_sha256(written), deploy_plan.file_sha256(blank)
        )
        self.assertEqual(
            deploy_plan.fingerprint(written, PACKAGE, CODE),
            deploy_plan.fingerprint(blank, PACKAGE, CODE),
        )

    def test_updates_and_comments_are_left_out(self):
        base = deploy_plan.file_sha256(doc())
        self.assertEqual(deploy_plan.file_sha256(doc({"updates.branch": "main"})), base)
        text = settings_file.template(file_types.DEPLOY, "full")
        commented = settings_file.loads(
            text.replace("\n", "\n# a comment of yours\n", 1), file_types.DEPLOY
        )
        self.assertEqual(
            deploy_plan.file_sha256(commented),
            deploy_plan.file_sha256(_template("full")),
        )


class TestRefused(unittest.TestCase):
    def test_another_layout_or_kind(self):
        for d in (
            settings_file.Document(file_types.DEPLOY, 1, {}),
            settings_file.Document(file_types.BUILD, file_types.BUILD.version, {}),
        ):
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    deploy_plan.plan(d, PACKAGE, CODE, HOST)


class TestTeamSkills(unittest.TestCase):
    """While skills_folder is set, the fingerprint and the file's own SHA-256 cover each of the
    folder's files by path and SHA-256, read from the folder itself, so that a changed skill is a change to confirm
    (state 5); the layer records them, so the plan made again from it gives its fingerprint.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = pathlib.Path(tmp.name)
        self.verifier = self.folder / "team_members" / "rocke_review_verifier.md"
        self.verifier.parent.mkdir()
        self.verifier.write_text("# v1\n", encoding="utf-8")
        self.changes = {deploy_plan.SKILLS_KEY: str(self.folder)}

    def test_a_changed_skill_changes_the_fingerprint_and_the_file_s_sha256(self):
        made = plan(self.changes)
        rel = "team_members/rocke_review_verifier.md"
        self.assertEqual(
            made.layer["skills"], {rel: hashlib.sha256(b"# v1\n").hexdigest()}
        )
        self.assertNotEqual(made.fingerprint, plan().fingerprint)
        d = doc(self.changes)
        self.assertEqual(deploy_plan.fingerprint(d, PACKAGE, CODE), made.fingerprint)

        def both():
            return deploy_plan.fingerprint(d, PACKAGE, CODE), deploy_plan.file_sha256(d)

        first = both()
        self.verifier.write_text("# v2\n", encoding="utf-8")  # a changed file
        second = both()
        self.assertNotEqual(second[0], first[0])
        self.assertNotEqual(second[1], first[1])
        (self.folder / "agents").mkdir()  # an empty folder is no file
        self.assertEqual(both(), second)
        (self.folder / "review_book.md").write_text("# book\n", encoding="utf-8")
        third = both()  # an added file
        self.assertNotEqual(third[0], second[0])
        self.assertNotEqual(third[1], second[1])
        # the plan made again from the layer delivered before gives that layer's fingerprint
        again = deploy_plan.from_layer(made.layer, HOST.uid, HOST.gid)
        self.assertEqual(again.fingerprint, made.fingerprint)
        self.assertEqual(again.layer, made.layer)

    def test_a_blank_folder_leaves_both_as_they_were(self):
        made, d = plan(), doc()
        self.assertNotIn("skills", made.layer)
        values = deploy_plan.settings(d)
        self.assertIsNone(deploy_plan.skills(values))
        self.assertEqual(
            made.fingerprint,
            settings_file.fingerprint(
                {
                    "deploy_schema_version": d.file_type.version,
                    "package_version": PACKAGE,
                    "code_sha256": CODE,
                    "settings": values,
                }
            ),
        )
        self.assertEqual(
            deploy_plan.file_sha256(d),
            settings_file.fingerprint(deploy_plan._without(d.values)),
        )

    def test_a_link_or_a_pipe_in_the_folder_is_never_read(self):
        """A link is never followed, so no file outside the folder (a secret, say) gets into the fingerprint, the
        layer or the delivered copy; a pipe never holds the read. Check 18 names both, and its FAIL stops the
        confirm."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        outside = pathlib.Path(tmp.name)
        (outside / "token").write_text("not a skill\n", encoding="utf-8")
        (self.folder / "review_book.md").symlink_to(outside / "token")
        (self.folder / "agents").symlink_to(outside, target_is_directory=True)
        os.mkfifo(self.folder / "team_members" / "pipe.md")
        rel = "team_members/rocke_review_verifier.md"
        values = deploy_plan.settings(doc(self.changes))
        self.assertEqual(deploy_plan.skill_files(values), {rel: b"# v1\n"})
        self.assertEqual(
            plan(self.changes).layer["skills"],
            {rel: hashlib.sha256(b"# v1\n").hexdigest()},
        )
        self.assertEqual(deploy_plan.skills_path(values), self.folder)
        self.assertIsNone(deploy_plan.skill_files(deploy_plan.settings(doc())))


if __name__ == "__main__":
    unittest.main()
