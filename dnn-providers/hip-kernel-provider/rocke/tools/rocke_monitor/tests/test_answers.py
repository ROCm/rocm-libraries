# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The YAML subset's reader (rkmon_lib/answers.py): the subset's forms, and each form outside it refused with its
line; `plain=True` keeps each plain scalar's text for the caller to type, in YAML and in JSON; an escaped quote inside
double quotes ends neither the string nor starts a comment; a flow mapping can be a list's item, and so can a mapping
whose first key is quoted; `typed` gives a plain scalar's value by the subset's rules.
"""

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import answers  # noqa: E402
from rkmon_lib.answers import Plain  # noqa: E402


class TestTheSubset(unittest.TestCase):
    def test_the_yaml_subset(self):
        text = (
            "# answers\nadvanced:\n  repo: acme/widgets   # a comment\n  scope:\n    labels: [rocke, 'x: y']\n"
            "  worker:\n    context: 32768\n    temperature: 0.3\n  limits: { max_parallel: 1, timeout_min: 20 }\n"
            "  tracker_keys:\n  - TK\n  - OT\n  jira: {}\nsetup:\n  clone: true\n  venvs: no-such\n"
            '  note: "a # inside quotes"\n  none: ~\n'
        )
        a = answers.parse(text)
        self.assertEqual(a["advanced"]["scope"]["labels"], ["rocke", "x: y"])
        self.assertEqual(
            a["advanced"]["worker"], {"context": 32768, "temperature": 0.3}
        )
        self.assertEqual(
            a["advanced"]["limits"], {"max_parallel": 1, "timeout_min": 20}
        )
        self.assertEqual(a["advanced"]["tracker_keys"], ["TK", "OT"])
        self.assertEqual(a["advanced"]["jira"], {})
        self.assertEqual(
            a["setup"],
            {
                "clone": True,
                "venvs": "no-such",
                "note": "a # inside quotes",
                "none": None,
            },
        )

    def test_unsupported_forms_are_refused_with_the_line(self):
        for text, line in (
            ("a:\n\tb: 1\n", 2),
            ("a: &x 1\n", 1),
            ("a:\n  - [1, [2]]\n", 2),
            ("a: {b: {c: 1}}\n", 1),
            ("a: 1\na: 2\n", 2),
            ("a: |\n  x\n", 1),
        ):
            with self.subTest(text=text):
                with self.assertRaises(answers.AnswersError) as cm:
                    answers.parse(text)
                self.assertIn(f"line {line}", str(cm.exception))


class TestPlainMode(unittest.TestCase):
    TEXT = (
        "rocm:\n"
        "  version: 7.10\n"
        '  quoted: "7.10"\n'
        "  switch: true\n"
        "  blank:\n"
        "  nothing: null\n"
        "  word: prototype\n"
        "labels: [a, 7, 'b']\n"
        'map: {x: 1, y: "2"}\n'
        "items:\n"
        "  - 0755\n"
        "  - name: 22:22\n"
    )

    def test_each_plain_scalar_keeps_its_text(self):
        a = answers.parse(self.TEXT, plain=True)
        rocm = a["rocm"]
        self.assertEqual(rocm["version"], "7.10")
        self.assertIsInstance(rocm["version"], Plain)
        self.assertEqual(rocm["switch"], Plain("true"))
        self.assertIsInstance(rocm["switch"], Plain)
        self.assertEqual(rocm["nothing"], Plain("null"))
        self.assertIsInstance(rocm["word"], Plain)
        self.assertEqual(a["labels"], ["a", "7", "b"])
        self.assertEqual([type(x) for x in a["labels"]], [Plain, Plain, str])
        self.assertEqual(a["items"], ["0755", {"name": "22:22"}])
        self.assertIsInstance(a["items"][1]["name"], Plain)

    def test_quoted_text_and_blanks_are_not_plain(self):
        a = answers.parse(self.TEXT, plain=True)
        self.assertIs(type(a["rocm"]["quoted"]), str)
        self.assertIsNone(a["rocm"]["blank"])
        self.assertEqual(a["map"], {"x": "1", "y": "2"})
        self.assertIsInstance(a["map"]["x"], Plain)
        self.assertIs(type(a["map"]["y"]), str)

    def test_without_it_the_subsets_types_stay(self):
        a = answers.parse(self.TEXT)
        self.assertEqual(a["rocm"]["version"], 7.1)
        self.assertIs(a["rocm"]["switch"], True)
        self.assertIsNone(a["rocm"]["nothing"])
        self.assertEqual(a["labels"], ["a", 7, "b"])
        self.assertNotIsInstance(a["rocm"]["word"], Plain)

    def test_json_numbers_keep_their_text(self):
        text = '{"version": 7.10, "count": 3, "on": true, "none": null, "name": "x"}'
        a = answers.loads(text, plain=True)
        self.assertEqual(a["version"], Plain("7.10"))
        self.assertIsInstance(a["count"], Plain)
        self.assertIs(a["on"], True)
        self.assertIsNone(a["none"])
        self.assertIs(type(a["name"]), str)
        self.assertEqual(answers.loads(text)["version"], 7.1)

    def test_load_reads_a_file_either_way(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "s.yaml"
            f.write_text("a: 1.50\n", encoding="utf-8")
            self.assertEqual(answers.load(f, plain=True), {"a": Plain("1.50")})
            self.assertEqual(answers.load(f), {"a": 1.5})

    def test_the_source_names_the_text_in_a_message(self):
        with self.assertRaises(answers.AnswersError) as cm:
            answers.loads("- a\n", "~/s.yaml")
        self.assertEqual(
            str(cm.exception),
            "~/s.yaml: the top level is not a mapping of keys to values",
        )
        with self.assertRaises(answers.AnswersError) as cm:
            answers.loads('{"a": }', "~/s.json", plain=True)
        self.assertIn("~/s.json: not valid JSON", str(cm.exception))


class TestEscapes(unittest.TestCase):
    def test_an_escaped_quote_keeps_the_string_open(self):
        text = 'a: "say \\"hi\\" # not a comment" # a comment\n'
        for plain in (False, True):
            with self.subTest(plain=plain):
                self.assertEqual(
                    answers.parse(text, plain), {"a": 'say "hi" # not a comment'}
                )

    def test_one_escaped_quote_just_before_a_comment_mark(self):
        # The quote after the backslash is text, so the string goes on past " #" to the next quote.
        self.assertEqual(answers.parse('a: "5\\" # tall" # note\n'), {"a": '5" # tall'})

    def test_an_escaped_quote_in_a_flow_list(self):
        a = answers.parse('a: ["x\\", y", z]\n')
        self.assertEqual(a, {"a": ['x", y', "z"]})

    def test_an_escaped_backslash_ends_before_the_quote(self):
        self.assertEqual(answers.parse('a: "c:\\\\" # note\n'), {"a": "c:\\"})


class TestFlowMappingItem(unittest.TestCase):
    def test_a_flow_mapping_is_a_list_item(self):
        a = answers.parse("a:\n  - {name: x, size: 2}\n  - y\n")
        self.assertEqual(a, {"a": [{"name": "x", "size": 2}, "y"]})


class TestListOfMappings(unittest.TestCase):
    def test_a_quoted_first_key_starts_a_mapping(self):
        text = (
            "a:\n"
            '  - "yes": "y"\n'
            "    other: 1\n"
            "  - 'a b':\n"
            "      c: d\n"
            '  - "a: b": c\n'
            # Quoted scalars, which are no keys.
            '  - "a: b"\n'
            "  - 'x'\n"
        )
        self.assertEqual(
            answers.parse(text),
            {
                "a": [
                    {"yes": "y", "other": 1},
                    {"a b": {"c": "d"}},
                    {"a: b": "c"},
                    "a: b",
                    "x",
                ]
            },
        )


class TestTyped(unittest.TestCase):
    def test_the_subsets_rules(self):
        for text, want in (
            ("true", True),
            ("False", False),
            ("~", None),
            ("NULL", None),
            ("-12", -12),
            ("1.50", 1.5),
            ("1e5", "1e5"),
            ("2026.11.15", "2026.11.15"),
            ("x y", "x y"),
        ):
            with self.subTest(text=text):
                got = answers.typed(text)
                self.assertEqual(got, want)
                self.assertIs(type(got), type(want))


if __name__ == "__main__":
    unittest.main()
