# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/closing_sentences.py: each key's closing sentence written at the end of its description, the
one already there replaced, a second run changing nothing; a schema it cannot finish left as it was; and what it
prints and its exit status."""

import contextlib
import copy
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "closing_sentences.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOL.parent))
import closing_sentences as cs  # noqa: E402

from rkmon_lib import settings_file  # noqa: E402

SAMPLE = ROOT / "tests" / "data" / "sample.schema.json"


def small():
    """A small schema: the header's two keys and a key of each kind of closing sentence, two of them sharing one
    schema through $ref."""
    return {
        "$defs": {
            "port": {"type": "integer", "default": 8080, "description": "The port."}
        },
        "type": "object",
        "required": ["kind", "schema_version", "name"],
        "properties": {
            "kind": {"const": "small_settings", "description": "What the file is."},
            "schema_version": {
                "const": 1,
                "x-changed-by": "small migrate updates it",
                "description": "The file's layout.",
            },
            "name": {
                "type": "string",
                "default": "small",
                "description": 'The service\'s name. Default: "old".',
            },
            "front": {"$ref": "#/$defs/port"},
            "back": {"$ref": "#/$defs/port"},
            "page": {
                "type": "string",
                "default": "",
                "x-required-when": "the queue is live",
                "x-blank": "to link no page",
                "description": "A page to link.",
            },
            "folders": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "default": {},
                "description": "Folders by name.",
            },
        },
    }


def written(data) -> bytes:
    """A schema's file as the tool writes it."""
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


class TestWriteSentences(unittest.TestCase):
    def test_each_keys_sentence_at_the_end_of_its_description(self):
        data = small()
        # The two keys that share the port's schema change one description.
        self.assertEqual(cs.write_sentences(data), 6)
        props = data["properties"]
        self.assertEqual(
            [
                props[k]["description"]
                for k in ("kind", "schema_version", "name", "page", "folders")
            ],
            [
                "What the file is. Don't change it.",
                "The file's layout. Don't change it: small migrate updates it.",
                "The service's name. Required.",
                "A page to link. Required when the queue is live. Otherwise, leave blank to link no page.",
                "Folders by name. Default: none.",
            ],
        )
        self.assertEqual(
            data["$defs"]["port"]["description"], "The port. Default: 8080."
        )
        self.assertEqual(settings_file.closing_problems(data), [])

    def test_a_second_run_changes_nothing(self):
        data = small()
        cs.write_sentences(data)
        again = copy.deepcopy(data)
        self.assertEqual(cs.write_sentences(again), 0)
        self.assertEqual(again, data)

    def test_the_sample_schema_is_as_it_writes_it(self):
        data = json.loads(SAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(cs.write_sentences(data), 0)
        self.assertEqual(written(data), SAMPLE.read_bytes())

    def test_a_key_with_no_description_is_left_for_its_author(self):
        data = small()
        data["properties"]["name"]["description"] = ""
        cs.write_sentences(data)
        self.assertEqual(data["properties"]["name"]["description"], "")
        self.assertEqual(settings_file.closing_problems(data), ["name: no description"])

    def test_a_key_with_no_rule_leaves_the_schema_as_it_was(self):
        data = small()
        del data["$defs"]["port"]["default"]
        before = copy.deepcopy(data)
        with self.assertRaisesRegex(ValueError, "^front: no closing sentence: "):
            cs.write_sentences(data)
        self.assertEqual(data, before)

    def test_a_shared_schema_that_needs_two_sentences(self):
        data = small()
        data["required"].append("back")
        before = copy.deepcopy(data)
        with self.assertRaisesRegex(
            ValueError,
            "^front and back share one schema and need two closing sentences, "
            "'Default: 8080.' and 'Required.'$",
        ):
            cs.write_sentences(data)
        self.assertEqual(data, before)


class TestMain(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = pathlib.Path(tmp.name)

    def schema_file(self, data, name="small.schema.json"):
        path = self.folder / name
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    def run_main(self, *paths):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cs.main([str(p) for p in paths])
        return status, out.getvalue()

    def test_a_file_is_written_only_when_a_description_changes(self):
        path = self.schema_file(small())
        self.assertEqual(self.run_main(path), (0, f"{path}: 6 descriptions changed\n"))
        data = small()
        cs.write_sentences(data)
        self.assertEqual(path.read_bytes(), written(data))
        os.utime(path, (0, 0))
        self.assertEqual(self.run_main(path), (0, f"{path}: 0 descriptions changed\n"))
        self.assertEqual(path.stat().st_mtime, 0)
        self.assertEqual(path.read_bytes(), written(data))

    def test_each_problem_left_is_printed(self):
        data = small()
        data["properties"]["name"]["description"] = "Required."
        path = self.schema_file(data)
        self.assertEqual(
            self.run_main(path),
            (
                1,
                f"{path}: 5 descriptions changed\n"
                f"{path}: name: its description says nothing before its closing sentence\n",
            ),
        )

    def test_a_file_with_a_key_that_has_no_rule_is_left_as_it_was(self):
        data = small()
        del data["$defs"]["port"]["default"]
        path = self.schema_file(data)
        before = path.read_bytes()
        self.assertEqual(
            self.run_main(path),
            (
                1,
                f"{path}: left as it was: front: no closing sentence: give it a default, x-blank or "
                "x-required-when, or require it\n",
            ),
        )
        self.assertEqual(path.read_bytes(), before)

    def test_a_file_that_is_no_schema(self):
        listed = self.folder / "list.json"
        listed.write_text("[1, 2]\n", encoding="utf-8")
        missing = self.folder / "missing.json"
        path = self.schema_file(small())
        status, out = self.run_main(listed, missing, path)
        lines = out.splitlines()
        self.assertEqual(status, 2)
        self.assertEqual(
            lines[0], f"{listed}: cannot be read as a schema: a schema is a JSON object"
        )
        self.assertTrue(lines[1].startswith(f"{missing}: cannot be read as a schema: "))
        # The files after it are written all the same.
        self.assertEqual(lines[2:], [f"{path}: 6 descriptions changed"])

    def test_no_file_named(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as cm:
            cs.main([])
        self.assertEqual(cm.exception.code, 2)

    def test_as_a_command(self):
        path = self.schema_file(small())
        run = subprocess.run(
            [sys.executable, "-B", str(TOOL), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            (run.returncode, run.stdout, run.stderr),
            (0, f"{path}: 6 descriptions changed\n", ""),
        )


if __name__ == "__main__":
    unittest.main()
