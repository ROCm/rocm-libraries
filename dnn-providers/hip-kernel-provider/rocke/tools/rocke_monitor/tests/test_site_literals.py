# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No site literal: the home repository's folder, its watched workflows' names, the names its develop
issues are known by, and the develop guard's labels appear only in the repository defaults or the settings, never in
the code.

What is looked for (`literals`): each folder the package's repository defaults were written for, and each tail of it
of two parts or more (a path that ends in one is the folder under another top); their values of the watched workflows
(develop_watched_workflows), the watched jobs (develop_rocke_job_pattern) and incident_issue's names and labels; and
the develop guard's labels a layout 1 configuration has by default (rkmon_lib/from_layout1.py), which a settings file
names. A workflow's name is found as a whole string, as a workflow may bear the name of a tool the code names too
(pre-commit); every other literal inside a longer string too.

Where (`code_files`, `strings`): every file of the package but the tests' (TESTS: their data is invented, or the home
repository's through tests/home_repository.py) and the docs' (PROSE: the guides and the agentic files' prose). Its
strings: a Python file's string and bytes literals but its docstrings; a JSON file's string values but a schema's words
for people (its descriptions, and its x- notes but x-template, the value a settings file template holds); any other
file's lines but its comments (the shell scripts, the Dockerfile, the compose files, the ignore files, the service
unit). EXEMPT names each file or folder the scan leaves out, with why; each holds a literal still.
"""

import ast
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (  # noqa: E402
    from_layout1,
    package_files,
    registry,
    repository_defaults,
)

TESTS = "tests/"  # the tests' files
PROSE = (".md",)  # the docs' files, wherever they lie
# The files the scan leaves out (a folder's, by its name and a slash), with why.
EXEMPT = {
    "rkmon_lib/from_layout1.py": "it translates a layout 1 configuration by that layout's own meanings, its defaults "
    "with the home repository's workflows and the develop guard's labels among them",
    "schema/layout1/": "layout 1's own schemas and defaults, which the translation of a layout 1 file reads",
    "repository/": "the repository defaults: the one home of these values, which the code asks",
    "requirements/": "the Python packages the checks and the generators install, by their names on the package "
    "index, where a package's name holds a develop issue's label",
}


def literals() -> dict:
    """{literal: (what it is, whether it is found as a whole string only)}: see the module's docstring."""
    home = repository_defaults.home()
    values = home.values
    out = {}
    for folder in home.folders:
        parts = folder.split("/")
        for n in range(
            max(1, len(parts) - 1)
        ):  # the folder, and each tail of two parts or more
            out["/".join(parts[n:])] = ("the home repository's folder", False)
    for name in registry.get(values, "develop_watched_workflows") or ():
        out[name] = ("a watched workflow's name", True)
    for key in (
        "develop_rocke_job_pattern",
        "incident_issue.title_pattern",
        "incident_issue.test_name_pattern",
    ):
        if registry.get(values, key):
            out[registry.get(values, key)] = (key, False)
    for key in (
        "incident_issue.umbrella_test_names",
        "incident_issue.time_limit_labels",
    ):
        for name in registry.get(values, key) or ():
            out[name] = (key, False)
    for label in (
        from_layout1.CI_RED,
        *from_layout1.GUARD_LABELS,
        from_layout1.STABILITY_LABEL,
    ):
        out[label] = ("a develop guard's label", False)
    return out


def python_strings(text: str):
    """(line, string) for each string or bytes literal of Python TEXT but its docstrings."""
    tree = ast.parse(text)
    docs = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docs:
            continue
        if isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node.value, bytes):
            yield node.lineno, node.value.decode("utf-8", "replace")


def text_lines(text: str):
    """(line, text) for each line of TEXT but its comment lines."""
    for n, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("#"):
            yield n, line


def json_strings(data, schema: bool, where: str = ""):
    """(where, string) for each string value of DATA; of a SCHEMA, but its words for people."""
    if isinstance(data, dict):
        for key, value in data.items():
            if schema and (
                key == "description" or (key.startswith("x-") and key != "x-template")
            ):
                continue
            yield from json_strings(value, schema, f"{where}.{key}" if where else key)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from json_strings(value, schema, f"{where}[{i}]")
    elif isinstance(data, str):
        yield where, data


def strings(rel: str, text: str):
    """(where, string) for each string of the code in the file REL, whose text is TEXT."""
    if rel.endswith(".json"):
        return json_strings(json.loads(text), schema=rel.startswith("schema/"))
    if rel.endswith(".py") or text.startswith("#!/usr/bin/env python"):
        return python_strings(text)
    return text_lines(text)


def code_files() -> list:
    """Every file of the package but the tests' (TESTS) and the docs' (PROSE): relative, with / between their parts."""
    names = []
    for path in package_files.tracked(ROOT):
        rel = (path.relative_to(ROOT) if path.is_absolute() else path).as_posix()
        if not rel.startswith(TESTS) and not rel.endswith(PROSE):
            names.append(rel)
    return sorted(names)


def exempt(rel: str) -> bool:
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in EXEMPT)


def found(looked_for: dict, rel: str, text: str) -> list:
    """(where, literal, what it is, the string) for each literal of LOOKED_FOR (`literals`) in the file REL."""
    out = []
    for where, string in strings(rel, text):
        for literal, (what, whole) in looked_for.items():
            if string == literal if whole else literal in string:
                out.append((where, literal, what, string))
    return out


class TestSiteLiterals(unittest.TestCase):
    def test_no_code_holds_a_site_literal(self):
        looked_for = literals()
        problems = []
        for rel in code_files():
            if exempt(rel):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for where, literal, what, _string in found(looked_for, rel, text):
                problems.append(
                    f"{rel}:{where}: {what}, {literal!r}: the repository defaults or the settings hold it, "
                    "and the code asks them"
                )
        self.assertEqual(problems, [])

    def test_what_is_looked_for(self):
        """Every kind of site literal: the home repository's folder, its watched workflows' names, the names
        its develop issues are known by, and the develop guard's labels."""
        looked_for = literals()
        self.assertIn(
            repository_defaults.home().folders[0], looked_for
        )  # the folder the home repository's defaults were written for
        kinds = {what for what, _whole in looked_for.values()}
        for kind in (
            "the home repository's folder",
            "a watched workflow's name",
            "develop_rocke_job_pattern",
            "incident_issue.title_pattern",
            "incident_issue.test_name_pattern",
            "incident_issue.umbrella_test_names",
            "a develop guard's label",
        ):
            self.assertIn(kind, kinds)
        self.assertIn("rkmon_lib/breakage.py", code_files())
        self.assertIn("schema/deploy_settings.schema.json", code_files())

    def test_every_file_but_the_tests_and_the_docs_is_read(self):
        """No file is left out silently: the scan reads each file of the package outside the tests and the docs, the
        code that is no Python module too, and EXEMPT says why for each one it leaves out.
        """
        files = code_files()
        for rel in (
            "conftest.py",
            "tools/rocke-monitor.service",
            "VERSION",
            ".gitignore",
            ".dockerignore",
        ):
            self.assertIn(rel, files)
        self.assertTrue([rel for rel in files if rel.startswith("repository/")])
        self.assertFalse(
            [rel for rel in files if rel.startswith(TESTS) or rel.endswith(PROSE)]
        )

    def test_each_form_of_a_string_is_read_and_the_docs_are_not(self):
        looked_for = {
            "lib/kern": ("the folder", False),
            "Build and test": ("a workflow", True),
            "red-ci": ("a label", False),
        }
        python = (
            '"""The module\'s docstring names lib/kern."""\n'
            "# a comment names lib/kern\n"
            'FOLDER = "a/lib/kern/b"\n'
            'RUNS = b"Build and test"\n'
            'NOTE = "a lint in Build and test"\n'
            "def f(x):\n"
            '    """lib/kern in a docstring."""\n'
            '    return f"{x} (red-ci)"\n'
        )
        self.assertEqual(
            [(w, lit) for w, lit, _what, _s in found(looked_for, "x.py", python)],
            [(3, "lib/kern"), (4, "Build and test"), (8, "red-ci")],
        )
        shell = "# lib/kern in a comment\ncd lib/kern  # and here\n"
        self.assertEqual(
            [(w, lit) for w, lit, _what, _s in found(looked_for, "x.sh", shell)],
            [(2, "lib/kern")],
        )
        schema = json.dumps(
            {
                "properties": {
                    "folder": {
                        "description": "Example: lib/kern.",
                        "x-source": "such as red-ci",
                        "x-template": "lib/kern",
                        "default": "red-ci",
                    }
                }
            }
        )
        self.assertEqual(
            [
                (w, lit)
                for w, lit, _what, _s in found(looked_for, "schema/x.json", schema)
            ],
            [
                ("properties.folder.x-template", "lib/kern"),
                ("properties.folder.default", "red-ci"),
            ],
        )
        self.assertEqual(
            [
                w
                for w, _lit, _what, _s in found(
                    looked_for, "x.json", json.dumps({"description": "red-ci"})
                )
            ],
            ["description"],
        )  # a JSON file that is no schema has no words for people

    def test_each_exemption_is_needed(self):
        """Each exempt file or folder holds a literal still."""
        looked_for = literals()
        files = code_files()
        for e in EXEMPT:
            with self.subTest(exempt=e):
                held = [
                    rel
                    for rel in files
                    if exempt(rel)
                    and (rel == e or rel.startswith(e))
                    and found(
                        looked_for,
                        rel,
                        (ROOT / rel).read_text(encoding="utf-8", errors="replace"),
                    )
                ]
                self.assertTrue(held)


if __name__ == "__main__":
    unittest.main()
