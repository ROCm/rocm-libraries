# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Layout 1's deploy settings schema (schema/layout1/deploy_settings.schema.json, rkmon_lib/schema.py) and the
template made from it, the image's name in every schema, and README.md's settings reference: every template key has
its description (the schema's: one home) and a schema entry; the template starts with the schema line editors read,
and validates; the commented model_bindings examples validate once uncommented; the deploy guide's table and the FAQ
follow the order of the deploy settings file's parts. README.md's settings reference follows the deploy schema the
deploy settings file is read with (file_types.DEPLOY) and the build schema the build settings file is read with
(file_types.BUILD), each build key with what the build is given from it, and no key of the deploy settings file feeds
an advanced key.
"""

import json
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (
    answers,
    build_plan,
    deploy_tree,
    deployment,
    file_types,
    from_layout1,
    image,
    registry,
    schema,
    settings_file,
)  # noqa: E402


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def readme_part(title: str) -> str:
    """README.md's part under its heading "### TITLE", to the next heading."""
    part = text("README.md").split(f"\n### {title}\n", 1)[1]
    return re.split(r"\n#{1,3} ", part, maxsplit=1)[0]


DEPLOY = schema.load("deploy")
# Layout 1's tree: the deploy file's sections, but the header's, `advanced` and the parts beside the tree.
TREE_PARTS = tuple(
    k
    for k in DEPLOY["properties"]
    if k
    not in (
        "kind",
        "schema_version",
        "advanced",
        deploy_tree.SECRET_VALUES,
        *deploy_tree.OTHER_PARTS,
    )
)
TREE_KEYS = [
    k
    for k, _ in schema.leaves(DEPLOY)
    if k.split(".", 1)[0] in TREE_PARTS and k != "model_bindings"
]


class TestTheSchemas(unittest.TestCase):
    def test_every_key_has_a_description_a_type_and_a_default(self):
        for key in TREE_KEYS:
            with self.subTest(key=key):
                node = schema.node(DEPLOY, key)
                self.assertTrue(str(node.get("description") or "").strip(), key)
                self.assertTrue(node.get("type"), key)
                self.assertIn("default", node, key)
                if "enum" in node:
                    self.assertIn(node["default"], node["enum"], key)
                self.assertEqual(
                    schema.validate(node, node["default"], key + "."), [], key
                )
        for dotted, node in schema.sections(DEPLOY):
            self.assertTrue(node.get("title") and node.get("description"), dotted)

    def test_the_image_and_the_tools_are_named_alike_in_every_file(self):
        """The image's name: one pattern in the build settings file and in both layouts' deploy settings files,
        layout 1's taking a blank too; one default (image.DEFAULT_NAME) in layout 2's two files, and in layout 1's
        the name layout 1's package used (from_layout1.IMAGE_NAME), frozen with its schema. In layout 2's two files,
        the image's keys and the GitHub and atlassian CLIs' sections have the same names. The base's default is
        image.DEFAULT_BASE.
        """
        build, deploy = file_types.BUILD.load(), file_types.DEPLOY.load()
        name = schema.node(build, "image.name")
        self.assertEqual(schema.node(deploy, "image.name")["pattern"], name["pattern"])
        self.assertEqual(
            schema.node(DEPLOY, "container.image.name")["pattern"],
            "^$|" + name["pattern"],
        )
        for node in (name, schema.node(deploy, "image.name")):
            self.assertEqual(node["default"], image.DEFAULT_NAME)
        self.assertEqual(
            schema.node(DEPLOY, "container.image.name")["default"],
            from_layout1.IMAGE_NAME,
        )
        self.assertEqual(
            list(schema.node(build, "image")["properties"]),
            list(schema.node(deploy, "image")["properties"]),
        )
        self.assertEqual(
            list(schema.node(DEPLOY, "container.image")["properties"])[:2],
            ["name", "tag"],
        )
        for tool in ("github_cli", "atlassian_cli"):
            self.assertIn(tool, build["properties"])
            self.assertIn(tool, deploy["properties"]["integrations"]["properties"])
        self.assertEqual(
            schema.node(build, "operating_system.base_image")["default"],
            image.DEFAULT_BASE,
        )
        for key, node in schema.leaves(DEPLOY):
            self.assertFalse(
                key.startswith("integrations.") and key.endswith(".enabled"),
                f"{key}: an integration is required by the feature that needs it, never switched",
            )

    def test_the_headers(self):
        self.assertEqual(DEPLOY["required"], ["kind", "schema_version"])
        self.assertEqual(schema.node(DEPLOY, "kind")["enum"], ["deploy_settings"])

    def test_it_is_layout_1_s_in_every_package(self):
        """Layout 1's one schema is the deploy settings file's: the build settings file is read in layout 2 alone
        (file_types.BUILD)."""
        self.assertEqual(list(schema.FILES), ["deploy"])
        self.assertEqual(
            sorted(p.name for p in schema.FOLDER.glob("*.schema.json")),
            [schema.FILES["deploy"]],
        )
        self.assertEqual(schema.node(DEPLOY, "schema_version")["enum"], [1])
        self.assertEqual(
            schema.path("deploy", "/srv/pkg"),
            pathlib.Path("/srv/pkg/schema/layout1", schema.FILES["deploy"]),
        )
        self.assertEqual(schema.path("deploy"), schema.path("deploy", ROOT))

    def test_the_built_in_models_are_described(self):
        mb = DEPLOY["properties"]["model_bindings"]
        for name, model in mb["x-builtin"].items():
            self.assertIn(name, mb["description"])
            self.assertIn(model, mb["description"])
        self.assertTrue(set(mb["x-builtin"]) <= set(mb["x-reserved"]))


class TestTheTemplates(unittest.TestCase):
    def test_every_deploy_template_key_has_its_description_and_a_schema_entry(self):
        tpl = deploy_tree.template("2026.11.13")
        keys = schema.template_keys(tpl)
        self.assertEqual(
            sorted(k for k, _ in keys),
            sorted(["kind", "schema_version"] + TREE_KEYS + ["advanced"]),
        )
        self.assertIn("\nmodel_bindings:\n", tpl)
        for key, comment in keys:
            with self.subTest(key=key):
                node = schema.node(DEPLOY, key)
                self.assertIsNotNone(node, key)
                self.assertEqual(
                    " ".join(comment.split()),
                    " ".join(str(node["description"]).split()),
                )

    def test_a_key_without_its_description_is_caught(self):
        self.assertEqual(
            schema.template_keys(
                "kind: x\na:\n  b: 1  # b's\n  # c's, above it\n  c: 2\n  d: 3\n"
            ),
            [("kind", ""), ("a.b", "b's"), ("a.c", "c's, above it"), ("a.d", "")],
        )

    def test_each_key_comes_with_its_line_number(self):
        self.assertEqual(
            schema.template_lines(
                "# head\n\nkind: x\na:\n  b: 1  # b's\n  # c's, above it\n  c: 2\n"
            ),
            [(3, "kind", ""), (5, "a.b", "b's"), (7, "a.c", "c's, above it")],
        )

    def test_the_private_keys_are_those_marked_private(self):
        # A download location is private by its x-class, as every private key is: x-secret marks nothing.
        sch = {
            "properties": {
                "a": {"type": "string", "x-secret": True},
                "s": {
                    "properties": {
                        "b": {"type": "string", "x-class": "private"},
                        "c": {"type": "string", "x-class": "public"},
                    }
                },
                "d": {"type": "string"},
            }
        }
        self.assertEqual(schema.private_keys(sch), ["s.b"])

    def test_the_template_starts_with_the_schema_line_and_validates(self):
        path = ROOT / "schema" / "layout1" / "deploy_settings.schema.json"
        tpl = deploy_tree.template("2026.11.13", ROOT)
        self.assertEqual(tpl.splitlines()[0], f"# yaml-language-server: $schema={path}")
        json.loads(path.read_text(encoding="utf-8"))
        data = answers.parse(deploy_tree.template("2026.11.13"))
        self.assertEqual(schema.validate(DEPLOY, data, ""), [])
        self.assertEqual(data["container"]["image"]["tag"], "2026.11.13")

    def test_the_model_binding_examples_validate_once_uncommented(self):
        tpl = deploy_tree.template("2026.11.13")
        lines = tpl.splitlines()
        at = lines.index("model_bindings:")
        block = []
        for ln in lines[at + 1 :]:
            if ln and not ln.startswith("  "):
                break
            block.append(ln)
        examples = [
            ln
            for ln in block
            if ln.startswith("  # ") and not ln.startswith("  # Example")
        ]
        self.assertGreater(len(examples), 8)
        filled = (
            "\n".join("  " + ln[4:] for ln in examples)
            .replace("<your gateway URL>", "https://gateway.example.invalid/api/v1")
            .replace("<tag>", "latest")
        )
        data = answers.parse(
            "kind: deploy_settings\nschema_version: 1\nmodel_bindings:\n"
            + filled
            + "\n"
        )
        self.assertEqual(sorted(data["model_bindings"]), ["glm-flash", "qwen-local"])
        self.assertEqual(
            data["model_bindings"]["glm-flash"]["model_id"], "GLM-5.3-Flash"
        )
        self.assertEqual(
            schema.validate(
                schema.node(DEPLOY, "model_bindings"),
                data["model_bindings"],
                "model_bindings.",
            ),
            [],
        )
        self.assertNotIn(
            "gateway.example.invalid", tpl
        )  # no gateway address in the package


class TestTheDocsFollowTheTree(unittest.TestCase):
    def test_the_readme_follows_the_deploy_schema(self):
        """README.md's deploy settings, from the schema the deploy settings file is read with (file_types.DEPLOY): a row
        for every key but the model bindings, and a row in bold for every section, before its keys, all in the file's
        order; then the model bindings, a table for each form of a binding, by its provider, with a row for each of its
        keys."""
        layout = file_types.DEPLOY.load()
        rows = re.findall(
            r"(?m)^\| (\*\*)?`([\w.]+)`(?:\*\*)? \|", readme_part("Deploy settings")
        )
        heads = [d for d, _ in schema.sections(layout)]
        self.assertEqual([key for bold, key in rows if bold], heads)
        self.assertEqual(
            [key for bold, key in rows if not bold],
            [k for k, _ in schema.leaves(layout) if k != "model_bindings"],
        )
        order = [key for _, key in rows]
        for d in heads:
            first = next(k for k in order if k.startswith(d + "."))
            self.assertLess(order.index(d), order.index(first), d)
        forms = readme_part("Model bindings").split("| Key | What it is |")[1:]
        for branch in layout["$defs"]["binding"]["oneOf"]:
            provider = branch["properties"]["provider"]["const"]
            with self.subTest(provider=provider):
                head = f'| `model_bindings.<name>.provider` | "{provider}".'
                form = next((f for f in forms if head in f), "")
                self.assertTrue(form, "no table for the form")
                for key, _ in schema.leaves(branch):
                    self.assertIn(f"| `model_bindings.<name>.{key}` |", form)

    def test_the_readme_follows_the_build_schema(self):
        """README.md's build settings: a row for every key of the schema the build settings file is read with
        (file_types.BUILD), in its order, each with what the build is given from it (rkmon_lib/build_plan.py): a build
        argument, a build secret, the root certificates' build context, or nothing (—); a section's row in bold, with
        no such cell.
        """
        build = file_types.BUILD.load()
        rows = re.findall(
            r"(?m)^\| (\*\*)?`([\w.]+)`(?:\*\*)? \| ([^|]*?) \|",
            readme_part("Build settings"),
        )
        self.assertEqual(
            [key for bold, key, _ in rows if bold],
            [d for d, _ in schema.sections(build)],
        )
        self.assertEqual({given for bold, _, given in rows if bold}, {""})
        given = {key: given for bold, key, given in rows if not bold}
        self.assertEqual(list(given), [k for k, _ in schema.leaves(build)])
        for a in build_plan.ARGS:
            if a.key:
                self.assertEqual(given[a.key], f"build argument `{a.name}`", a.key)
        for d in build_plan.DOWNLOADS:
            self.assertEqual(given[d.key], f"build secret `{d.id}`", d.key)
        self.assertEqual(
            given[deployment.CA_KEY], f"build context `{build_plan.CA_CONTEXT}`"
        )
        named = {a.key for a in build_plan.ARGS} | {d.key for d in build_plan.DOWNLOADS}
        self.assertEqual(
            {v for k, v in given.items() if k not in named | {deployment.CA_KEY}},
            {"—"},
        )

    def test_the_deploy_guide_s_keys_to_fill_in_are_the_file_s_keys_in_order(self):
        """The keys the deploy guide's "The keys you must fill in" names (its generated required-keys section): the
        keys each template marks <...>, then the table's, a row's keys under the section it names; each one a key of
        the deploy settings file's schema (file_types.DEPLOY; <name>: any entry of its map), and in each part the
        file's sections in their order.
        """
        layout = file_types.DEPLOY.load()
        tops = list(layout["properties"])
        guide = text("QUICK_DEPLOY_GUIDE.md")
        part = guide.split("\n## The keys you must fill in\n", 1)[1]
        part = part.split("\n## ", 1)[0]
        marked = next(ln for ln in part.splitlines() if "`<...>`" in ln)
        parts = [re.findall(r"`([\w.]+)`", marked), []]
        for line in part.splitlines():
            m = re.match(r"\| .+? \| (?:in `([\w.]+)`: )?(.+) \|$", line)
            if m and not line.startswith(("| Required when ", "|---")):
                under = f"{m.group(1)}." if m.group(1) else ""
                parts[1] += [under + k for k in re.findall(r"`([\w.<>]+)`", m.group(2))]
        self.assertGreater(len(parts[0]), 3)
        self.assertGreater(len(parts[1]), 10)
        known = {path for path, _, _ in settings_file.schema_keys(layout)}
        for keys in parts:
            for key in keys:
                with self.subTest(key=key):
                    self.assertIn(key.replace(".<name>.", ".*."), known)
            order = [tops.index(k.split(".", 1)[0]) for k in keys]
            self.assertEqual(order, sorted(order))

    def test_the_faq_names_the_deploy_settings_file_s_parts_in_order(self):
        """The FAQ names the parts of the deploy settings file (file_types.DEPLOY), but its header, in their order."""
        parts = [
            p
            for p in file_types.DEPLOY.load()["properties"]
            if p not in settings_file.HEADER_KEYS
        ]
        names = ", ".join(f"`{p}`" for p in parts[:-1]) + f", then `{parts[-1]}`"
        self.assertIn(names, " ".join(text("docs/FAQ.md").split()))


def advanced() -> list:
    """The advanced keys: every key of the configuration no key of the deploy settings file feeds, the registry's
    package, state and derived keys (rkmon_lib/registry.py), each with its description there.
    """
    return [k.name for k in registry.keys() if k.owner != registry.DEPLOY]


class TestTheAdvancedKeys(unittest.TestCase):
    def test_no_key_of_the_deploy_settings_file_feeds_an_advanced_key(self):
        for key in advanced():
            with self.subTest(key=key):
                self.assertNotIn(
                    key,
                    deploy_tree.held(),
                    f"{key}: a key of the deploy settings file feeds it, so its description is the schema's",
                )


if __name__ == "__main__":
    unittest.main()
