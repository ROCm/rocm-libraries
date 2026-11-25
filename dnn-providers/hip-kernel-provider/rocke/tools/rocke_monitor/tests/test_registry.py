# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The registry (rkmon_lib/registry.py): every key has one owner; the deploy keys are the names the
deploy schema's keys give, and every key of the schema makes one or is named with the step that carries it; the package
keys are config.defaults.json's, the state keys the state schema's; one function per derived key, the derived keys last;
each key described, in words the docs' words lint takes; each package and derived key's description ending with its one
closing sentence; a package key in GiB named so; the package's defaults read from their one home, the tuned limits among
them; and the running configuration refuses a read of a name the registry does not know.
"""

import copy
import json
import pathlib
import re
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    ci_times,
    config,
    conflicts,
    deploy_tree,
    derived,
    drain,
    file_types,
    idle,
    mergeplan,
    mirror,
    offload,
    overlaps,
    registry,
    retention,
    scoreboard,
    settings_file,
)

import doc_lints  # noqa: E402


def owned(owner: str) -> list:
    return [k for k in registry.keys() if k.owner == owner]


def covers(file_key: str, leaf: str) -> bool:
    """FILE_KEY is the schema key LEAF, or a section or map around it; a map's entry by its name stands for NAME.*."""
    have, want = file_key.split("."), leaf.split(".")
    return len(have) <= len(want) and all(
        h == w or w == "*" for h, w in zip(have, want)
    )


class TestOwners(unittest.TestCase):
    def test_the_registry_finds_no_problem(self):
        self.assertEqual(registry.check(), [])

    def test_each_key_has_one_owner(self):
        names = [k.name for k in registry.keys()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({k.owner for k in registry.keys()}, set(registry.OWNERS))

    def test_no_key_lies_inside_another(self):
        names = {k.name for k in registry.keys()}
        for name in names:
            parts = name.split(".")
            for i in range(1, len(parts)):
                self.assertNotIn(".".join(parts[:i]), names, name)

    def test_each_key_is_described(self):
        for k in registry.keys():
            with self.subTest(key=k.name):
                self.assertTrue(k.description.strip())

    def test_each_description_is_in_words_the_docs_words_lint_takes(self):
        """A key's, a section's and an entry's description read as the docs do: the docs' words lint
        (tools/doc_lints.py, tests/doc_words.yaml) finds nothing in them."""
        texts = [
            doc_lints.Text(f"registry: {name}", 1, text, doc_lints.USER)
            for name, text in registry.descriptions().items()
        ]
        found = doc_lints.words(texts, doc_lints.load_words())
        self.assertEqual([str(p) for p in found], [])

    def test_the_check_names_a_key_owned_twice_or_left_undescribed(self):
        row = owned(registry.PACKAGE)[0]
        found = registry.check(
            registry.keys()
            + (row._replace(owner=registry.STATE),)
            + (registry.Key("merge_plan.max_merges.x", registry.PACKAGE, ""),)
        )
        self.assertIn(f"{row.name}: owned by package and state", found)
        self.assertIn("merge_plan.max_merges.x: no description", found)
        self.assertIn(
            "merge_plan.max_merges.x is inside the key merge_plan.max_merges", found
        )


class TestClosingSentences(unittest.TestCase):
    def test_every_package_and_derived_key_ends_with_its_closing_sentence(self):
        """Each package key's description ends with its default's closing sentence (settings_file.default_sentence,
        the value config.defaults.json's), or "Blank means ..." for blank text; each derived key's with "Derived:
        RULE."; none has a second one. The registry's closing sentences are checked as the schemas' are
        (test_file_types)."""
        self.assertEqual(registry.closing_problems(), [])

    def test_a_closing_sentence_that_is_not_its_key_s_is_named(self):
        package, derived_ = registry.PACKAGE, registry.DERIVED
        rows = (
            registry.Key("a.number", package, "A number. Default: 5.", 6),
            registry.Key("a.bare", package, "A number.", 6),
            registry.Key(
                "a.twice", package, "A number. Blank means none. Default: 6.", 6
            ),
            registry.Key(
                "a.inline", package, "Jobs (blank: any job). Default: none.", []
            ),
            registry.Key("a.older", package, "A list. Empty means none.", []),
            registry.Key("a.empty", package, "A list. Default: [].", []),
            registry.Key(
                "a.means", package, "A map. Default: none, which means all.", {}
            ),
            registry.Key(
                "a.map",
                package,
                'A map. Default: {"x": 1, "y": ["z"]}.',
                {"x": 1, "y": ["z"]},
            ),
            registry.Key("a.other", package, 'A map. Default: {"x": 2}.', {"x": 1}),
            registry.Key(
                "a.nested", package, 'A map. Default: {"x": {"y": 1}}.', {"x": {"y": 1}}
            ),
            registry.Key("a.text", package, 'A text. Default: "".', ""),
            registry.Key(
                "a.after", derived_, "A path. Derived: <WORKING_DIR>/x. More words."
            ),
            registry.Key("a.derived", derived_, "A path. Blank means <WORKING_DIR>/x."),
            registry.Key("a.alone", package, "Default: 6.", 6),
            registry.Key(
                "a.fine", package, "A number. Default: 6, which means six.", 6
            ),
            registry.Key("a.list", package, 'A list. Default: ["x", 2].', ["x", 2]),
            registry.Key("a.blank", package, "A text. Blank means none.", ""),
            registry.Key(
                "a.rule",
                derived_,
                "A path. Derived: blank, which means <WORKING_DIR>/x.",
            ),
            registry.Key(
                "a.state", registry.STATE, "A state key: its schema's check reads it."
            ),
        )
        self.assertEqual(
            registry.closing_problems(rows),
            [
                "a.number: its description ends with 'Default: 5.', not 'Default: 6.'",
                "a.bare: its description does not end with its closing sentence, 'Default: 6.'",
                "a.twice: its description has a second closing sentence, before 'Default: 6.'",
                "a.inline: its description has a second closing sentence, before 'Default: none.'",
                "a.older: its description does not end with its closing sentence, 'Default: none.'",
                "a.empty: its description ends with 'Default: [].', not 'Default: none.'",
                "a.means: its description ends with 'Default: none, which means all.', not 'Default: none.'",
                "a.other: its description ends with 'Default: {\"x\": 2}.', not 'Default: {\"x\": 1}.'",
                "a.nested: its default cannot be written in a closing sentence: the writer cannot write dict here",
                "a.text: its description ends with 'Default: \"\".', not 'Blank means <what a blank one means>.'",
                "a.after: its description does not end with its closing sentence, 'Derived: <its rule>.'",
                "a.derived: its description ends with 'Blank means <WORKING_DIR>/x.', not 'Derived: <its rule>.'",
                "a.alone: its description says nothing before its closing sentence",
            ],
        )


class TestDeployKeys(unittest.TestCase):
    def test_the_deploy_keys_are_the_names_the_deploy_schema_s_leaves_give(self):
        """The deploy keys, against the deploy schema read on its own (not the carry's rules, which the keys are made
        from): each name a carried leaf gives in its x-internal, a map entry's (NAME.*.FIELD) as its map; the local
        worker's keys, one per field of a binding (WORKER_FIELDS renames two); and the two keys a rule makes from keys
        that feed others: whether a type of ticket is on, and the local worker's effort.
        """
        named, fields = set(), set()
        for leaf, node, _required in settings_file.schema_keys(
            file_types.DEPLOY.load()
        ):
            if leaf in settings_file.HEADER_KEYS or leaf in deploy_tree.NOT_CARRIED:
                continue
            internal = node.get("x-internal") or []
            for name in [internal] if isinstance(internal, str) else internal:
                named.add(name.split(".*.", 1)[0])
                if name.startswith("model_bindings.*."):
                    fields.add(name.split(".*.", 1)[1])
        worker = {f"worker.{deploy_tree.WORKER_FIELDS.get(f, f)}" for f in fields}
        self.assertEqual(
            {k.name for k in owned(registry.DEPLOY)},
            named | worker | {"atlassian.enabled", "worker.think"},
        )

    def test_every_key_of_the_deploy_schema_makes_a_deploy_key_or_waits_for_its_step(
        self,
    ):
        made = [f for k in owned(registry.DEPLOY) for f in k.file_keys]
        for leaf, _node, _required in settings_file.schema_keys(
            file_types.DEPLOY.load()
        ):
            if leaf in settings_file.HEADER_KEYS:
                continue
            with self.subTest(leaf=leaf):
                self.assertTrue(
                    any(covers(f, leaf) for f in made)
                    or leaf in deploy_tree.NOT_CARRIED,
                    leaf,
                )

    def test_each_file_key_of_a_deploy_key_is_in_the_schema(self):
        for k in owned(registry.DEPLOY):
            for f in k.file_keys:
                with self.subTest(key=k.name, file_key=f):
                    self.assertIsNotNone(deploy_tree.node(f)[1], f)

    def test_a_deploy_key_carries_its_schema_s_default_readers_and_class(self):
        k = registry.key("idle_full_tick_min")
        self.assertEqual(k.file_keys, ("monitor.full_tick_max_interval_minutes",))
        self.assertEqual(
            k.default,
            deploy_tree.get(deploy_tree.internal_defaults(), "idle_full_tick_min"),
        )
        self.assertEqual(k.applies, "no_restart")
        self.assertEqual(k.klass, "public")
        self.assertIn(
            "monitor.full_tick_max_interval_minutes: ",
            registry.descriptions()["idle_full_tick_min"],
        )
        self.assertEqual(registry.key("worker.model").klass, "private")


class TestPackageKeys(unittest.TestCase):
    def test_the_package_keys_are_config_defaults_json_s(self):
        data = json.loads(registry.PACKAGE_FILE.read_text(encoding="utf-8"))
        package = {k.name for k in owned(registry.PACKAGE)}
        found = []

        def walk(node, prefix):
            for key, value in node.items():
                path = prefix + key
                if path in package:
                    found.append(path)
                elif isinstance(value, dict) and value:
                    walk(value, path + ".")
                else:
                    self.fail(f"config.defaults.json holds {path}, no package key")

        walk(data, "")
        self.assertEqual(sorted(found), sorted(package))

    def test_a_package_key_s_default_is_config_defaults_json_s(self):
        data = json.loads(registry.PACKAGE_FILE.read_text(encoding="utf-8"))
        for k in owned(registry.PACKAGE):
            with self.subTest(key=k.name):
                self.assertEqual(k.default, registry.get(data, k.name))

    def test_a_package_key_in_gib_ends_in_gib(self):
        """A package key whose description gives its number in GiB ends in _gib, as a settings file's key does:
        the sizes beside memory_gib and ramdisk_gib among them."""
        found = [
            k.name
            for k in owned(registry.PACKAGE)
            if re.search(r"\bin GiB\b", k.description)
        ]
        self.assertIn("resources.work_memory_gib", found)
        for name in found:
            with self.subTest(key=name):
                self.assertTrue(name.endswith("_gib"), name)

    def test_config_defaults_json_holds_no_description(self):
        data = json.loads(registry.PACKAGE_FILE.read_text(encoding="utf-8"))
        self.assertNotIn("_doc", data)

    def test_the_tuned_limits_are_the_package_s_and_the_readers_have_no_copy(self):
        """The heartbeat of 180 minutes and the merge plan's 200 merges in 30 seconds have one home,
        config.defaults.json: the tick loop's settings and the merge plan read them from there.
        """
        self.assertEqual(registry.default("loop_heartbeat_min"), 180)
        self.assertEqual(idle.settings({})["loop_heartbeat_min"], 180)
        self.assertEqual(
            mergeplan.settings({}),
            {"max_merges": 200, "budget_s": 30, "timeout_s": 120},
        )

    def test_the_readers_lay_their_section_over_the_package_s(self):
        for module, section in (
            (ci_times, "ci_times"),
            (conflicts, "conflicts"),
            (mergeplan, "merge_plan"),
            (mirror, "mirror"),
            (offload, "offload"),
            (overlaps, "overlaps"),
            (retention, "retention"),
            (scoreboard, "scoreboard"),
        ):
            with self.subTest(section=section):
                self.assertEqual(module.DEFAULTS, registry.package_section(section))
        # The update section is the drain's limits, which rk drain reads through the registry (drain.TIMEOUT_KEY,
        # drain.POLL_KEY): no other key of it is read.
        self.assertEqual(
            {f"update.{k}" for k in registry.package_section("update")},
            {drain.TIMEOUT_KEY, drain.POLL_KEY},
        )
        self.assertEqual(
            idle.settings({}), {k: registry.default(k) for k in idle.SETTINGS}
        )

    def test_a_section_is_a_copy(self):
        registry.package_section("merge_plan")["max_merges"] = 1
        self.assertEqual(registry.default("merge_plan.max_merges"), 200)


class TestStateKeys(unittest.TestCase):
    def test_the_state_keys_are_the_state_schema_s(self):
        schema = file_types.INTERNAL_STATE.load()
        rows = [
            path
            for path, _node, _required in settings_file.schema_keys(schema)
            if path not in settings_file.HEADER_KEYS
        ]
        maps = {p.split(".*.", 1)[0] for p in rows if ".*." in p}
        self.assertEqual(
            sorted(f for k in owned(registry.STATE) for f in k.file_keys),
            sorted(p for p in rows if p not in maps),
        )

    def test_a_state_key_is_named_as_the_configuration_reads_it(self):
        k = registry.key("codeowner_teams")
        self.assertEqual(
            (k.owner, k.file_keys, k.default),
            (registry.STATE, ("repository.codeowner_teams",), {}),
        )
        self.assertEqual(registry.key("standby.on").default, False)


class TestDerivedKeys(unittest.TestCase):
    def test_one_function_per_derived_key(self):
        self.assertEqual(
            [k.name for k in owned(registry.DERIVED)], list(derived.FUNCTIONS)
        )
        self.assertTrue(all(callable(f) for f in derived.FUNCTIONS.values()))
        self.assertEqual(
            len({id(f) for f in derived.FUNCTIONS.values()}), len(derived.FUNCTIONS)
        )

    def test_the_derived_keys_come_last(self):
        owners = [k.owner for k in registry.keys()]
        first = owners.index(registry.DERIVED)
        self.assertEqual(set(owners[first:]), {registry.DERIVED})

    def test_a_configuration_file_holds_no_path_it_never_set(self):
        defaults = registry.defaults()
        for name in registry.DERIVED_NOT_IN_FILES:
            with self.subTest(key=name):
                self.assertIs(registry.get(defaults, name), registry.MISSING)
        self.assertEqual(registry.get(defaults, "models.draft"), "worker")


class TestNames(unittest.TestCase):
    def test_a_name_is_a_key_a_section_an_entry_or_unknown(self):
        self.assertEqual(registry.kind("merge_plan.max_merges"), registry.KEY)
        self.assertEqual(registry.kind("merge_plan"), registry.SECTION)
        self.assertEqual(registry.kind("hot_spots.ABI"), registry.ENTRY)
        self.assertIsNone(registry.kind("merge_plan.max_merge"))
        with self.assertRaises(KeyError):
            registry.key("merge_plan.max_merge")

    def test_the_loader_s_defaults_and_descriptions_are_the_registry_s(self):
        self.assertEqual(config.load_defaults(), registry.defaults())
        self.assertEqual(config.doc(), registry.descriptions())
        self.assertEqual(config.DEFAULTS, registry.PACKAGE_FILE)

    def test_a_section_is_described_before_its_keys_and_an_entry_after_its_key(self):
        """The names that are no key keep their descriptions (PARTS): a section's comes before its first key, an
        entry's after its key; a described part that is neither a section nor inside a key is a problem.
        """
        order = list(registry.descriptions())
        for part in ("mirror", "offload", "offload.budget_shares", "gpu"):
            first = next(n for n in order if n.startswith(part + "."))
            self.assertLess(order.index(part), order.index(first), part)
        self.assertEqual(
            order.index("gpu.base_env.ROCKE_BACKEND"), order.index("gpu.base_env") + 1
        )
        self.assertEqual(
            {registry.kind(p) for p in registry.PARTS},
            {registry.SECTION, registry.ENTRY},
        )
        parts = dict(registry.PARTS, nowhere="x", **{"merge_plan.max_merges": "y"})
        with mock.patch.object(registry, "PARTS", parts):
            found = registry.check()
        self.assertEqual(
            found,
            [
                "nowhere: described, but neither a section nor inside a key",
                "merge_plan.max_merges: described as a key and as a part",
            ],
        )

    def test_nest_and_get(self):
        nested = registry.nest({"a.b": 1, "a.c": [2], "d": "x"})
        self.assertEqual(nested, {"a": {"b": 1, "c": [2]}, "d": "x"})
        self.assertEqual(registry.get(nested, "a.c"), [2])
        self.assertIs(registry.get(nested, "a.e"), registry.MISSING)

    def test_a_key_without_a_default_gives_missing_itself(self):
        self.assertIs(registry.default("paths.root"), registry.MISSING)
        self.assertIs(copy.deepcopy(registry.MISSING), registry.MISSING)
        self.assertIs(copy.copy(registry.MISSING), registry.MISSING)


class TestValue(unittest.TestCase):
    """Every reader's fallback is the key's default, from the registry."""

    def test_a_key_left_out_takes_its_default(self):
        self.assertEqual(
            registry.value({}, "review.pipes"), registry.default("review.pipes")
        )
        self.assertEqual(
            registry.value({"review": None}, "review.pipes"),
            registry.default("review.pipes"),
        )
        self.assertEqual(registry.value({"review": {"pipes": 2}}, "review.pipes"), 2)
        self.assertIsNone(registry.value({}, "paths.root"))

    def test_a_blank_value_takes_the_default_where_the_reader_says_so(self):
        dflt = registry.default("team_dashboard.mode")
        for given, blank, want in (
            ("", (), ""),
            (None, (), None),
            ("", registry.BLANK, dflt),
            (None, registry.BLANK, dflt),
            ("serve", registry.BLANK, "serve"),
        ):
            with self.subTest(given=given, blank=blank):
                cfg = {"team_dashboard": {"mode": given}}
                self.assertEqual(
                    registry.value(cfg, "team_dashboard.mode", blank), want
                )
        port = registry.default("admin_dashboard.public_port")
        for given, blank, want in (
            (0, registry.BLANK, 0),
            (0, registry.FALSY, port),
            (False, registry.FALSY, port),
            (8080, registry.FALSY, 8080),
        ):
            with self.subTest(given=given, blank=blank):
                cfg = {"admin_dashboard": {"public_port": given}}
                self.assertEqual(
                    registry.value(cfg, "admin_dashboard.public_port", blank), want
                )

    def test_a_default_is_a_copy(self):
        registry.value({}, "hold_label_patterns").append("x")
        self.assertNotIn("x", registry.default("hold_label_patterns"))

    def test_a_name_the_registry_does_not_know_is_refused_even_when_set(self):
        with self.assertRaises(KeyError):
            registry.value({"review": {"pipe": 2}}, "review.pipe")

    def test_a_number_is_the_value_else_the_default(self):
        dflt = registry.default("review.pipes")
        for given, want in (
            (2, 2),
            ("4", 4),
            ("x", dflt),
            ("", dflt),
            (None, dflt),
            ([], dflt),
        ):
            with self.subTest(given=given):
                self.assertEqual(
                    registry.number({"review": {"pipes": given}}, "review.pipes"), want
                )
        self.assertEqual(registry.number({}, "review.pipes"), dflt)
        mem = registry.number({}, "checks.min_free_memory_gb", float)
        self.assertEqual(
            (mem, type(mem)),
            (float(registry.default("checks.min_free_memory_gb")), float),
        )

    def test_a_section_s_defaults(self):
        gpu = registry.defaults("gpu")
        self.assertEqual(
            gpu,
            registry.nest(
                {
                    k.name.split(".", 1)[1]: k.default
                    for k in registry.keys()
                    if k.name.startswith("gpu.") and k.default is not registry.MISSING
                }
            ),
        )
        gpu["max_cases"] = -1
        self.assertEqual(
            registry.defaults("gpu")["max_cases"], registry.default("gpu.max_cases")
        )
        self.assertEqual(registry.defaults("nowhere"), {})
        self.assertEqual(registry.defaults("review.pipes"), {})
        self.assertEqual(registry.defaults(), config.load_defaults())


class TestStrict(unittest.TestCase):
    def test_the_configuration_refuses_a_read_of_a_name_the_registry_does_not_know(
        self,
    ):
        cfg = registry.strict(registry.defaults())
        self.assertEqual(cfg, registry.defaults())
        self.assertEqual(
            cfg["merge_plan"]["max_merges"], registry.default("merge_plan.max_merges")
        )
        self.assertIn("merge_plan", cfg)
        reads = (
            lambda: cfg["no_such_key"],
            lambda: cfg.get("no_such_key", 1),
            lambda: "no_such_key" in cfg,
            lambda: cfg.setdefault("no_such_key", 1),
            lambda: cfg.pop("no_such_key", None),
            lambda: cfg["merge_plan"].get("max_merge"),
            lambda: registry.get(cfg, "merge_plan.max_merge"),
            lambda: cfg.copy()["no_such_key"],
            lambda: copy.deepcopy(cfg)["merge_plan"]["max_merge"],
        )
        for read in reads:
            with self.assertRaises(registry.UnknownKey) as e:
                read()
            self.assertNotIsInstance(e.exception, KeyError)
        self.assertIn("merge_plan.max_merge is no key", str(e.exception))
        cfg["hot_spots"]["any name"] = {}  # a key's own value is read freely
        self.assertIn("any name", cfg["hot_spots"])
        with self.assertRaises(KeyError):  # a key that is not there
            registry.strict({})["merge_plan"]


if __name__ == "__main__":
    unittest.main()
