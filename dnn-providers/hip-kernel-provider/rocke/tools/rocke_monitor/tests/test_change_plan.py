# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What confirm shows changing on a running deployment (rkmon_lib/change_plan.py): your changes with their classes, a
map's entry as one change and a new section key by key; the files
changed outside your file, by deliver.differing against the layer that runs, each with its hand edits as apply
writes them and your own changes left out, and none whose hand edit apply writes as it is or after a package update;
the line rule of a file's hand edits; the keys you left blank whose default changed, told from the keys you blanked
yourself; a package's keys added and removed; the package line, for another version or for other code under the
same one, and the package defaults; the count and whether a change restarts; a key classed no_restart only with its
check in the image test's no-restart stage.
"""

import copy
import dataclasses
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "container"))

import stage_no_restart  # noqa: E402
from rkmon_lib import change_plan, code_volume, deliver, deploy_plan  # noqa: E402
from rkmon_lib import file_types, paths, registry, settings_file  # noqa: E402
from rkmon_lib.change_plan import NEW_CONTAINER, NO_RESTART, Change  # noqa: E402
from rkmon_lib.deploy_tree import MISSING  # noqa: E402
from test_deploy_plan import CODE, FAST, HOST, PACKAGE, SERVED, doc  # noqa: E402

RAW = json.loads(
    (ROOT / "schema" / "deploy_settings.schema.json").read_text(encoding="utf-8")
)
CPUS = "host.resources.cpus"
RAMDISK = "host.resources.ramdisk_gib"
ADMIN_PORT = "features.dashboards.admin.port"
NOW = change_plan.Package(PACKAGE, file_types.DEPLOY, {}, CODE)
OTHER_CODE = "sha256:" + "d" * 64  # other code under the same version


def node(schema: dict, dotted: str) -> dict:
    """The schema's own node of a section's key ("": the schema's)."""
    for part in dotted.split(".") if dotted else ():
        schema = schema["properties"][part]
    return schema


def deploy_type(change) -> settings_file.FileType:
    """The deploy settings' file type of a package whose schema is this one's with CHANGE made to it (a function of
    the schema, a mapping)."""
    schema = copy.deepcopy(RAW)
    change(schema)
    deploy = file_types.DEPLOY
    return settings_file.FileType(deploy.kind, deploy.version, schema)


def package(version="1.2.2", change=lambda schema: None, defaults=None, code=CODE):
    return change_plan.Package(version, deploy_type(change), defaults or {}, code)


def default(dotted: str, value):
    """A schema change: the key DOTTED's default is VALUE."""
    return lambda schema: node(schema, dotted).update(default=value)


def applies(dotted: str, cls: str):
    """A schema change: the key DOTTED's class is CLS."""
    return lambda schema: node(schema, dotted).update({"x-applies": cls})


def without(dotted: str):
    """A schema change: no key DOTTED."""
    around, _, name = dotted.rpartition(".")

    def change(schema):
        del node(schema, around)["properties"][name]

    return change


def made(changes, pkg: change_plan.Package):
    """doc(CHANGES) as PKG reads it, and its plan."""
    d = dataclasses.replace(doc(changes), file_type=pkg.deploy)
    return d, deploy_plan.plan(d, pkg.version, pkg.code, HOST)


def applied(changes=None, pkg=NOW, env=None, override=None, layer=None):
    """What runs: the files deliver wrote for doc(CHANGES) with PKG (deliver.texts), each one ENV, OVERRIDE or LAYER
    names edited on this host: that function of its text (None: no file)."""
    _, made_plan = made(changes, pkg)
    edits = (env, override, layer)
    texts = [edit(x) if edit else x for edit, x in zip(edits, deliver.texts(made_plan))]
    return change_plan.Applied(texts[2], texts[0], texts[1])


def confirmed(changes=None, pkg=NOW) -> change_plan.Confirmed:
    return change_plan.Confirmed(made(changes, pkg)[0], HOST)


def opened(text: str) -> str:
    """The layer's text TEXT with the admin dashboard's address, which no deploy key sets, edited by hand."""
    layer = json.loads(text)
    layer["container"]["address"] = "0.0.0.0"
    return deploy_plan.layer_text(layer)


def port_edited(text: str) -> str:
    """docker/.env's text TEXT with the admin dashboard's port edited by hand."""
    return text.replace("RKMON_PORT=18614\n", "RKMON_PORT=18620\n")


def plan(was=None, now=None, then=NOW, current=NOW, **on_host):
    """What changes from what doc(WAS) made with THEN to doc(NOW) with CURRENT, ON_HOST as `applied` takes it."""
    return change_plan.plan(
        applied(was, then, **on_host), confirmed(now, current), then, current
    )


class TestYours(unittest.TestCase):
    """Your changes, each with its class."""

    def test_nothing_changes(self):
        p = plan()
        self.assertEqual(p, change_plan.ChangePlan(None, (), (), (), ()))
        self.assertEqual(p.total, 0)
        self.assertFalse(p.restarts)

    def test_a_value_you_change(self):
        p = plan(now={CPUS: 6})
        self.assertEqual(p.yours, (Change(CPUS, 4, 6, NEW_CONTAINER),))
        self.assertEqual(p.yours[0].mark, "~")
        self.assertEqual(p.outside, ())  # docker/.env's RKMON_CPUS follows from it
        self.assertEqual((p.total, p.restarts), (1, True))

    def test_a_no_restart_change_restarts_nothing(self):
        both = package(PACKAGE, applies(CPUS, NO_RESTART))
        p = plan(now={CPUS: 6}, then=both, current=both)
        self.assertEqual(p.yours, (Change(CPUS, 4, 6, NO_RESTART),))
        self.assertEqual((p.total, p.restarts), (1, False))

    def test_a_key_without_a_class_takes_the_nearest_around_it(self):
        # a binding's fields lie inside a oneOf: the map's class is theirs
        both = package(PACKAGE, applies("model_bindings", NO_RESTART))
        was = {"model_bindings.fast": FAST}
        now = {"model_bindings.fast": {**FAST, "model_id": "claude-sonnet-5-5"}}
        p = plan(was, now, both, both)
        key = "model_bindings.fast.model_id"
        self.assertEqual(
            p.yours,
            (Change(key, "claude-haiku-5-5", "claude-sonnet-5-5", NO_RESTART),),
        )

    def test_container_name_has_no_class(self):
        p = plan(now={"container.name": "rocke-monitor-2"})
        self.assertEqual(
            p.yours,
            (Change("container.name", "rocke-monitor", "rocke-monitor-2", ""),),
        )
        self.assertEqual(p.outside, ())  # COMPOSE_PROJECT_NAME follows from it
        self.assertTrue(p.restarts)

    def test_a_map_entry_added_or_removed_is_one_change(self):
        added = plan(now={"model_bindings.fast": FAST})
        self.assertEqual(
            added.yours,
            (Change("model_bindings.fast", MISSING, FAST, NEW_CONTAINER),),
        )
        self.assertEqual(added.yours[0].mark, "+")
        removed = plan(was={"model_bindings.fast": FAST})
        self.assertEqual(
            removed.yours,
            (Change("model_bindings.fast", FAST, MISSING, NEW_CONTAINER),),
        )
        self.assertEqual(removed.yours[0].mark, "-")

    def test_the_schema_order_with_a_removed_key_after_its_section(self):
        both = {"model_bindings.b": FAST}
        p = plan(
            was={**both, "model_bindings.a": FAST},
            now={**both, "model_bindings.c": FAST, CPUS: 6},
        )
        marks = [(c.mark, c.key) for c in p.yours]
        self.assertEqual(
            marks,
            [("~", CPUS), ("+", "model_bindings.c"), ("-", "model_bindings.a")],
        )


class TestOutside(unittest.TestCase):
    """The changes outside your file: each file deliver writes whose whole text is not what deliver wrote for the
    layer that runs, by the change plan's rule (deliver.differing), as it is and with its hand edits as apply writes
    them, your own changes left out; none whose hand edit apply writes as it is, and none after a package update.
    """

    def lines(self, change: Change) -> tuple:
        """CHANGE's lines that are not in both its texts: (OLD's, NEW's)."""
        old, new = (set(x.splitlines()) for x in (change.old, change.new))
        return old - new, new - old

    def test_docker_env_edited_on_this_host(self):
        _, made_plan = made(None, NOW)
        p = plan(env=port_edited)
        where = deliver.ENV_FILE
        self.assertNotEqual(port_edited(made_plan.env), made_plan.env)
        self.assertEqual(
            p.outside,
            (
                Change(
                    where,
                    port_edited(made_plan.env),
                    made_plan.env,
                    NEW_CONTAINER,
                    where,
                ),
            ),
        )
        self.assertEqual(p.outside[0].mark, "~")
        self.assertEqual((p.yours, p.total, p.restarts), ((), 1, True))

    def test_your_change_beside_a_hand_edit_of_its_file(self):
        # cpus is yours, and the admin port was edited by hand in the same docker/.env: the row outside holds the
        # port alone
        _, made_plan = made(None, NOW)
        p = plan(now={CPUS: 6}, env=port_edited)
        self.assertEqual(p.yours, (Change(CPUS, 4, 6, NEW_CONTAINER),))
        where = deliver.ENV_FILE
        self.assertEqual(
            p.outside,
            (
                Change(
                    where,
                    port_edited(made_plan.env),
                    made_plan.env,
                    NEW_CONTAINER,
                    where,
                ),
            ),
        )
        self.assertEqual(
            self.lines(p.outside[0]), ({"RKMON_PORT=18620"}, {"RKMON_PORT=18614"})
        )
        self.assertEqual(p.total, 2)

    def test_lines_your_change_adds_stay_out(self):
        # serving the team dashboard adds its address and port to docker/.env: yours alone
        _, made_plan = made(None, NOW)
        p = plan(now=SERVED, env=port_edited)
        self.assertEqual([c.key for c in p.outside], [deliver.ENV_FILE])
        self.assertEqual(p.outside[0].new, made_plan.env)

    def test_your_change_of_the_key_edited_by_hand(self):
        # the hand edit -> the value apply makes, which is yours
        p = plan(now={ADMIN_PORT: 18700}, env=port_edited)
        self.assertEqual(p.yours, (Change(ADMIN_PORT, 18614, 18700, NEW_CONTAINER),))
        self.assertEqual([c.key for c in p.outside], [deliver.ENV_FILE])
        self.assertEqual(
            self.lines(p.outside[0]), ({"RKMON_PORT=18620"}, {"RKMON_PORT=18700"})
        )

    def test_a_hand_edit_your_file_now_makes_is_no_change(self):
        def cpus_edited(text):
            return text.replace("RKMON_CPUS=4\n", "RKMON_CPUS=6\n")

        p = plan(now={CPUS: 6}, env=cpus_edited)
        self.assertEqual(p.yours, (Change(CPUS, 4, 6, NEW_CONTAINER),))
        self.assertEqual((p.outside, p.total), ((), 1))
        # beside another change of yours in the same file
        p = plan(now={CPUS: 6, ADMIN_PORT: 18700}, env=cpus_edited)
        self.assertEqual([c.key for c in p.yours], [CPUS, ADMIN_PORT])
        self.assertEqual((p.outside, p.total), ((), 2))
        p = plan(
            now={RAMDISK: 2}, override=lambda text: made({RAMDISK: 2}, NOW)[1].override
        )
        self.assertEqual((p.outside, p.total), ((), 1))

    def test_a_comment_is_a_change(self):
        p = plan(env=lambda text: text + "# a note of yours\n")
        self.assertEqual(
            [(c.mark, c.key) for c in p.outside], [("~", deliver.ENV_FILE)]
        )
        self.assertEqual((p.total, p.restarts), (1, True))
        p = plan(override=lambda text: "# only a comment\n")
        self.assertEqual(
            [(c.mark, c.key) for c in p.outside], [("-", deliver.OVERRIDE_FILE)]
        )

    def test_docker_env_removed_on_this_host(self):
        _, made_plan = made(None, NOW)
        p = plan(env=lambda text: None)
        where = deliver.ENV_FILE
        self.assertEqual(
            p.outside, (Change(where, MISSING, made_plan.env, NEW_CONTAINER, where),)
        )
        self.assertEqual(p.outside[0].mark, "+")

    def test_your_changes_alone_change_no_file_outside(self):
        # docker/.env's RKMON_TEAM_HOST and RKMON_TEAM_PORT, and the override's port, follow from the switch
        p = plan(now=SERVED)
        self.assertEqual(p.outside, ())
        self.assertIn("features.dashboards.team.enabled", [c.key for c in p.yours])
        p = plan(now={RAMDISK: 2, "container.name": "rocke-monitor-2"})
        self.assertEqual((p.outside, len(p.yours)), ((), 2))

    def test_the_override_edited_on_this_host(self):
        ramdisk = {RAMDISK: 2}
        _, made_plan = made(ramdisk, NOW)
        edited = made_plan.override.replace("size=2g", "size=8g")
        p = plan(ramdisk, ramdisk, override=lambda text: edited)
        where = deliver.OVERRIDE_FILE
        self.assertEqual(
            p.outside,
            (Change(where, edited, made_plan.override, NEW_CONTAINER, where),),
        )
        p = plan(ramdisk, ramdisk, override=lambda text: "# mine\n" + text)
        self.assertEqual([c.key for c in p.outside], [where], "a comment is a change")

    def test_the_override_removed_or_made_on_this_host(self):
        gone = plan({RAMDISK: 2}, {RAMDISK: 2}, override=lambda text: None)
        self.assertEqual([c.mark for c in gone.outside], ["+"])
        _, made_plan = made({RAMDISK: 2}, NOW)
        extra = plan(override=lambda text: made_plan.override)
        self.assertEqual([(c.mark, c.new) for c in extra.outside], [("-", MISSING)])

    def test_your_change_to_the_override_beside_a_hand_made_one(self):
        _, apply_plan = made({RAMDISK: 2}, NOW)
        p = plan(now={RAMDISK: 2}, override=lambda text: "services: {}\n")
        self.assertEqual(p.yours, (Change(RAMDISK, 0, 2, NEW_CONTAINER),))
        where = deliver.OVERRIDE_FILE
        self.assertEqual(
            p.outside,
            (
                Change(
                    where, "services: {}\n", apply_plan.override, NEW_CONTAINER, where
                ),
            ),
        )

    def test_the_layer_edited(self):
        _, made_plan = made(None, NOW)
        text = deploy_plan.layer_text(made_plan.layer)
        p = plan(layer=opened)
        name = deliver.LAYER_NAME
        self.assertEqual(
            p.outside, (Change(name, opened(text), text, NEW_CONTAINER, name),)
        )
        self.assertEqual(p.yours, ())
        p = plan(now={CPUS: 6}, layer=opened)
        self.assertEqual([c.key for c in p.outside], [name])

    def test_the_layers_settings_edited(self):
        # what runs is the layer's: the edit shows among your changes alone. Outside, the layer's one line unlike
        # deliver's text for its settings is its fingerprint, which apply writes as it is, and docker/.env, made from
        # the key the edit changes, is as apply writes it
        def edited(text):
            layer = json.loads(text)
            layer["settings"]["host"]["resources"]["cpus"] = 8
            return deploy_plan.layer_text(layer)

        p = plan(layer=edited)
        self.assertEqual(p.yours, (Change(CPUS, 8, 4, NEW_CONTAINER),))
        self.assertEqual(p.outside, ())
        # beside a change of yours, the fingerprint apply writes is another
        p = plan(now={RAMDISK: 2}, layer=edited)
        self.assertEqual([c.key for c in p.outside], [deliver.LAYER_NAME])
        self.assertEqual(
            [line.split(":")[0].strip() for line in self.lines(p.outside[0])[1]],
            ['"fingerprint"'],
        )

    def test_all_three_in_their_order(self):
        _, made_plan = made({RAMDISK: 2}, NOW)
        p = plan(
            env=lambda text: text + "# mine\n",
            override=lambda text: made_plan.override,
            layer=opened,
        )
        self.assertEqual([c.key for c in p.outside], list(deliver.WRITES))
        self.assertEqual([c.found_in for c in p.outside], list(deliver.WRITES))
        self.assertEqual(p.total, 3)

    def test_a_new_package_lists_no_file(self):
        p = plan(
            then=package(),
            env=lambda text: text + "# mine\n",
            override=lambda text: "services: {}\n",
            layer=opened,
        )
        self.assertEqual((p.package, p.outside), (("1.2.2", PACKAGE), ()))


class TestHandEdits(unittest.TestCase):
    """A file as found with only its lines edited by hand as apply writes them (hand_edits_applied): your changes
    elsewhere stay out, lines added by hand go, a hand edit and a change of yours that overlap take apply's lines
    together, and a file added or removed is apply's whole."""

    RAN = "a=1\nb=2\nc=3\nd=4\n"

    def applied(self, found, made, ran=RAN):
        return change_plan.hand_edits_applied(found, ran, made)

    def test_a_hand_edit_beside_your_change(self):
        self.assertEqual(
            self.applied("a=1\nb=9\nc=3\nd=4\n", "a=1\nb=2\nc=7\nd=4\n"), self.RAN
        )
        self.assertEqual(self.applied("a=1\nb=9\nc=3\nd=4\n", self.RAN), self.RAN)

    def test_your_change_of_the_line_edited_by_hand(self):
        made = "a=1\nb=5\nc=3\nd=4\n"
        self.assertEqual(self.applied("a=1\nb=9\nc=3\nd=4\n", made), made)

    def test_lines_added_or_removed_by_hand(self):
        added = "a=1\nb=2\n# mine\nc=3\nd=4\n"
        self.assertEqual(self.applied(added, "a=1\nb=2\nc=3\nd=4\ne=5\n"), self.RAN)
        self.assertEqual(
            self.applied("a=1\nc=3\nd=4\n", "a=1\nb=2\nc=3\nd=5\n"), self.RAN
        )
        # lines of yours at the very place of the hand's: apply's lines there
        yours = "a=1\nb=2\nx=0\nc=3\nd=4\n"
        self.assertEqual(self.applied(added, yours), yours)

    def test_lines_your_change_adds_beside_a_hand_edit(self):
        made = "a=1\nb=2\nx=0\nc=3\nd=4\n"
        self.assertEqual(self.applied("a=1\nb=9\nc=3\nd=4\n", made), self.RAN)
        self.assertEqual(self.applied("a=1\nb=2\nc=9\nd=4\n", made), self.RAN)

    def test_overlapping_runs_take_apply_lines_together(self):
        # line by line where a run changes as many lines as it holds
        found, made = "a=1\nb=9\nc=9\nd=4\n", "a=1\nb=2\nc=7\nd=8\n"
        self.assertEqual(self.applied(found, made), "a=1\nb=2\nc=7\nd=4\n")
        self.assertEqual(self.applied("a=1\nb=2\nc=3\n# mine\nd=4\n", made), self.RAN)
        # a run of yours that adds a line, under a hand edit; a line added by hand inside one
        made = "a=1\nb=2\nc=7\ne=5\nd=4\n"
        self.assertEqual(self.applied("a=1\nb=2\nc=9\nd=4\n", made), made)
        made = "a=1\nb=2\nc=7\nd=8\ne=5\n"
        self.assertEqual(self.applied("a=1\nb=2\nc=3\n# mine\nd=4\n", made), made)

    def test_a_file_added_or_removed_is_apply_whole(self):
        made = "a=1\n"
        self.assertEqual(self.applied(None, made), made)
        self.assertEqual(self.applied("x=1\n", made, ran=None), made)
        self.assertIsNone(self.applied("a=1\nb=9\nc=3\nd=4\n", None))
        self.assertEqual(self.applied(self.RAN, made), self.RAN, "no hand edit")


class TestPackage(unittest.TestCase):
    """After a package update: the package line, the blank keys whose default changed, the keys it adds and removes,
    and the package defaults."""

    def test_a_blank_key_whose_default_changed(self):
        then = package(change=default(CPUS, 2))
        p = plan({CPUS: None}, {CPUS: None}, then)
        self.assertEqual(p.package, ("1.2.2", PACKAGE))
        self.assertEqual(p.yours, ())
        self.assertEqual(p.blank_defaults, (Change(CPUS, 2, 4, NEW_CONTAINER),))
        self.assertEqual(p.outside, ())  # docker/.env's RKMON_CPUS follows from it
        self.assertEqual((p.total, p.restarts), (2, True))

    def test_a_key_you_blank_whose_default_changed_is_yours(self):
        then = package(change=default(CPUS, 2))
        p = plan({CPUS: 5}, {CPUS: None}, then)
        self.assertEqual(p.yours, (Change(CPUS, 5, 4, NEW_CONTAINER),))
        self.assertEqual(p.blank_defaults, ())

    def test_a_new_package_alone(self):
        p = plan(then=package())
        self.assertEqual(p, change_plan.ChangePlan(("1.2.2", PACKAGE), (), (), (), ()))
        self.assertEqual((p.total, p.restarts), (1, True))

    def test_other_code_under_the_same_version(self):
        """A package update to a later commit of updates.branch: the package line, with the same version, a restart;
        no file outside your file, as apply writes every file anew."""
        then = NOW._replace(code=OTHER_CODE)
        p = plan(then=then, env=lambda text: text + "# mine\n")
        self.assertEqual(p, change_plan.ChangePlan((PACKAGE, PACKAGE), (), (), (), ()))
        self.assertEqual((p.total, p.restarts), (1, True))
        # the same code: no package line, and the hand edit is a change outside your file
        same = plan(env=lambda text: text + "# mine\n")
        self.assertEqual((same.package, len(same.outside)), (None, 1))

    def test_a_key_the_package_adds(self):
        key = "monitor.parallel_builds"
        then = package(change=without(key))
        set_by_you = plan(then=then)  # the template sets it
        self.assertEqual(set_by_you.yours, (Change(key, MISSING, 1, NEW_CONTAINER),))
        left_blank = plan({key: None}, {key: None}, then)
        self.assertEqual(left_blank.yours, ())
        self.assertEqual(
            left_blank.blank_defaults, (Change(key, MISSING, 1, NEW_CONTAINER),)
        )

    def test_a_section_the_package_adds_is_one_change_for_each_key(self):
        then = package(change=without("monitor"))
        blank = {
            f"monitor.{k}": None for k in RAW["properties"]["monitor"]["properties"]
        }
        p = plan(blank, blank, then)
        self.assertEqual(
            [(c.mark, c.key, c.applies) for c in p.blank_defaults],
            [
                ("+", "monitor.tick_interval_minutes", NEW_CONTAINER),
                ("+", "monitor.full_tick_max_interval_minutes", NO_RESTART),
                ("+", "monitor.parallel_builds", NEW_CONTAINER),
            ],
        )

    def test_a_section_the_package_removes_is_one_change_for_each_key(self):
        current = package(PACKAGE, without("monitor"))
        p = plan(then=package(), current=current)
        self.assertEqual(
            [(c.mark, c.key, c.applies) for c in p.blank_defaults],
            [
                ("-", "monitor.tick_interval_minutes", NEW_CONTAINER),
                ("-", "monitor.full_tick_max_interval_minutes", NO_RESTART),
                ("-", "monitor.parallel_builds", NEW_CONTAINER),
            ],
        )

    def test_a_map_entry_is_no_section_whatever_its_schema(self):
        entry = {"type": "object", "properties": {"x": {"type": "integer"}}}
        root = {
            "properties": {
                "s": {"type": "object", "properties": {"t": entry}},
                "m": {"type": "object", "additionalProperties": entry},
                "both": {
                    "type": "object",
                    "properties": {"fixed": entry},
                    "additionalProperties": entry,
                },
            }
        }
        sections = ["s", "s.t", "m", "m.e", "both", "both.fixed", "both.e", "s.t.x"]
        self.assertEqual(
            [k for k in sections if change_plan._section(root, k)],
            ["s", "s.t", "both", "both.fixed"],
        )

    def test_a_key_the_package_removes_keeps_its_class_then(self):
        def extra(schema):
            node(schema, "monitor")["properties"]["extra_minutes"] = {
                "description": "Minutes. Default: 5.",
                "type": "integer",
                "default": 5,
                "x-class": "public",
                "x-applies": NO_RESTART,
            }

        p = plan(then=package(change=extra))
        self.assertEqual(
            p.blank_defaults,
            (Change("monitor.extra_minutes", 5, MISSING, NO_RESTART),),
        )
        self.assertEqual(p.blank_defaults[0].mark, "-")

    def test_the_package_defaults_that_changed_are_not_counted(self):
        then = package(PACKAGE, defaults={"a.b": 1, "c": 2, "e": 5})
        current = change_plan.Package(
            PACKAGE, file_types.DEPLOY, {"a.b": 1, "c": 3, "d": 4}, CODE
        )
        p = plan(then=then, current=current)
        self.assertEqual(
            p.package_defaults,
            (
                Change("c", 2, 3, NEW_CONTAINER),
                Change("d", MISSING, 4, NEW_CONTAINER),
                Change("e", 5, MISSING, NEW_CONTAINER),
            ),
        )
        self.assertEqual((p.total, p.restarts), (0, False))

    def test_a_layer_another_package_made_is_refused(self):
        with self.assertRaisesRegex(ValueError, "made by package 1.2.3, not 9.9.9"):
            change_plan.plan(applied(), confirmed(), package("9.9.9"), NOW)


class TestPackageIn(unittest.TestCase):
    """The package in a folder, as `plan` reads it (change_plan.package_in): its VERSION, its deploy settings' file
    type with the folder's own schema, its package defaults, and its code's SHA-256 by its own .dockerignore; a
    folder without them has no package.
    """

    SCHEMA = pathlib.Path(file_types.DEPLOY.schema).relative_to(paths.PACKAGE)
    DEFAULTS = registry.PACKAGE_FILE.relative_to(paths.PACKAGE)

    def folder(
        self, top: str, defaults='{"update": {"drain_timeout_min": 90}}'
    ) -> pathlib.Path:
        """A package folder in TOP: VERSION 1.2.2, the schema with the CPUs' default 2, and DEFAULTS' text."""
        folder = pathlib.Path(top) / "rkmon.prev"
        schema = copy.deepcopy(RAW)
        default(CPUS, 2)(schema)
        (folder / self.SCHEMA).parent.mkdir(parents=True)
        (folder / self.SCHEMA).write_text(json.dumps(schema), encoding="utf-8")
        (folder / "VERSION").write_text(
            "# the package's version\n1.2.2\n", encoding="utf-8"
        )
        (folder / self.DEFAULTS).write_text(defaults, encoding="utf-8")
        (folder / ".dockerignore").write_text("*\n!VERSION\n", encoding="utf-8")
        return folder

    def test_the_package_in_a_folder(self):
        with tempfile.TemporaryDirectory() as t:
            then = change_plan.package_in(self.folder(t))
            self.assertEqual(then.version, "1.2.2")
            # its code: the files its .dockerignore lets in, VERSION alone
            folder = pathlib.Path(t) / "rkmon.prev"
            code = code_volume.digest(folder, ["VERSION"])
            self.assertEqual(then.code, f"sha256:{code}")
            self.assertEqual(then.defaults, {"update": {"drain_timeout_min": 90}})
            deploy = file_types.DEPLOY
            self.assertEqual(
                (then.deploy.kind, then.deploy.version), (deploy.kind, deploy.version)
            )
            # the folder's own schema: its default, not this package's
            p = plan({CPUS: None}, {CPUS: None}, then)
            self.assertEqual(p.package, ("1.2.2", PACKAGE))
            self.assertEqual(p.blank_defaults, (Change(CPUS, 2, 4, NEW_CONTAINER),))

    def test_a_folder_without_a_package(self):
        with tempfile.TemporaryDirectory() as t:
            folder = self.folder(t)
            (folder / "VERSION").unlink()
            with self.assertRaisesRegex(ValueError, "rkmon.prev has no package$"):
                change_plan.package_in(folder)
        with tempfile.TemporaryDirectory() as t:
            folder = self.folder(t)
            (folder / self.SCHEMA).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rkmon.prev has no package$"):
                change_plan.package_in(folder)
        with tempfile.TemporaryDirectory() as t:
            folder = self.folder(t)
            (folder / ".dockerignore").unlink()  # no code a release installs
            with self.assertRaisesRegex(ValueError, "rkmon.prev has no package$"):
                change_plan.package_in(folder)
        for text, said in (
            ("{", "cannot be read"),
            ("[]", "has no package defaults"),
        ):
            with self.subTest(defaults=text), tempfile.TemporaryDirectory() as t:
                with self.assertRaisesRegex(
                    ValueError, f"config.defaults.json {said}$"
                ):
                    change_plan.package_in(self.folder(t, text))


class TestClasses(unittest.TestCase):
    def test_the_schemas_classes_are_these(self):
        found = set()

        def walk(value):
            if isinstance(value, dict):
                if "x-applies" in value:
                    found.add(value["x-applies"])
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        walk(RAW)
        self.assertLessEqual(found, set(change_plan.CLASSES))

    def test_a_key_is_classed_no_restart_only_with_its_check_in_the_no_restart_stage(
        self,
    ):
        """Each key the deploy schema classes no_restart is one the image test's no-restart stage changes on a running
        deployment, checking that it is read at its next use (tests/container/stage_no_restart.py CHECKS), and each
        key that stage checks is classed no_restart: no key changes class without its check.
        """
        self.assertEqual(
            stage_no_restart.no_restart_keys(RAW), set(stage_no_restart.CHECKS)
        )


if __name__ == "__main__":
    unittest.main()
