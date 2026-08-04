# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_doc_symbols import (
    build_symbol_inventory,
    build_python_index,
    check_docstring_references,
    parse_missing_references,
    render_inventory,
)


class CheckDocSymbolsTest(unittest.TestCase):
    def test_builds_definitions_members_and_reexports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text(
                "from .mod import Thing, helper\n", encoding="utf-8"
            )
            (package / "mod.py").write_text(
                "\n".join(
                    [
                        "VALUE = 1",
                        "def helper():",
                        "    return None",
                        "class Thing:",
                        "    field: int",
                        "    @property",
                        "    def name(self):",
                        "        return 'thing'",
                        "    def run(self):",
                        "        return None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            symbols = build_symbol_inventory([root])

        self.assertEqual(symbols["pkg.mod.helper"], "function")
        self.assertEqual(symbols["pkg.helper"], "function")
        self.assertEqual(symbols["pkg.Thing"], "class")
        self.assertEqual(symbols["pkg.mod.Thing.run"], "method")
        self.assertEqual(symbols["pkg.mod.Thing.name"], "attribute")
        self.assertEqual(symbols["pkg.mod.Thing.field"], "attribute")
        self.assertEqual(symbols["pkg.mod.VALUE"], "data")

        inventory = render_inventory(symbols)
        self.assertIn(".. py:function:: pkg.helper()", inventory)
        self.assertIn(".. py:method:: pkg.mod.Thing.run()", inventory)
        self.assertLess(
            inventory.index(".. py:function:: pkg.helper()"),
            inventory.index(".. py:module:: pkg"),
        )

    def test_maps_only_doc_reference_warnings_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_docs = root / "source" / "docs"
            docs_root = root / "fixtures" / "dsl_docs"
            copied_docs.mkdir(parents=True)
            docs_root.mkdir(parents=True)
            warnings = "\n".join(
                [
                    f"{copied_docs / 'guide.md'}:7: WARNING: py:func reference "
                    "target not found: pkg.missing [ref.func]",
                    f"{root / 'api.rst'}:9: WARNING: py:class reference target "
                    "not found: external.Missing [ref.class]",
                ]
            )

            broken = parse_missing_references(
                warnings,
                copied_docs,
                docs_root,
                root / "platform",
            )

        self.assertEqual(len(broken), 1)
        self.assertEqual(broken[0].path, "dsl_docs/guide.md")
        self.assertEqual(broken[0].line, 7)
        self.assertEqual(broken[0].role, "func")
        self.assertEqual(broken[0].target, "pkg.missing")

    def test_checks_only_explicit_roles_in_python_docstrings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "base.py").write_text(
                "class Thing:\n" "    def run(self):\n" "        return None\n",
                encoding="utf-8",
            )
            (package / "mod.py").write_text(
                '"""Use :meth:`Thing.run`, :func:`available`, '
                ":class:`ValueError`, and :class:`pathlib.Path`.\n\n"
                'Broken: :func:`missing`.\n"""\n'
                "import pathlib\n"
                "from .base import Thing\n\n"
                "# A comment with :func:`comment_only` is not a docstring.\n"
                "def available():\n"
                "    return None\n",
                encoding="utf-8",
            )

            index = build_python_index([root], root)
            broken, external = check_docstring_references(index)

        self.assertEqual(len(index.references), 5)
        self.assertEqual([item.target for item in external], ["pathlib.Path"])
        self.assertEqual(
            [(item.role, item.target, item.found_kinds) for item in broken],
            [("func", "missing", ())],
        )
        self.assertNotIn("comment_only", [item.target for item in index.references])

    def test_does_not_resolve_unrelated_symbol_by_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "a.py").write_text(
                "def helper():\n    return None\n", encoding="utf-8"
            )
            (package / "b.py").write_text(
                '"""This module uses :func:`helper`."""\n', encoding="utf-8"
            )

            index = build_python_index([root], root)
            broken, external = check_docstring_references(index)

        self.assertEqual(external, [])
        self.assertEqual(
            [(item.role, item.target, item.found_kinds) for item in broken],
            [("func", "helper", ())],
        )

    def test_accepts_existing_symbol_with_a_different_normal_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "mod.py").write_text(
                "\n".join(
                    [
                        '"""Use :class:`available` and :func:`Thing.run`."""',
                        "def available():",
                        "    return None",
                        "class Thing:",
                        "    def run(self):",
                        "        return None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            index = build_python_index([root], root)
            broken, external = check_docstring_references(index)

        self.assertEqual(broken, [])
        self.assertEqual(external, [])

    def test_relative_reference_requires_a_matching_symbol_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "mod.py").write_text(
                '"""Use :class:`.available`."""\n'
                "def available():\n"
                "    return None\n",
                encoding="utf-8",
            )

            index = build_python_index([root], root)
            broken, external = check_docstring_references(index)

        self.assertEqual(external, [])
        self.assertEqual(
            [(item.role, item.target, item.found_kinds) for item in broken],
            [("class", ".available", ("function",))],
        )


@unittest.skipUnless(shutil.which("uvx"), "uvx is required for CLI integration tests")
class CheckDocSymbolsCliTest(unittest.TestCase):
    def _run_checker(
        self, reference: str
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs_root = root / "docs"
            python_root = root / "python"
            package = python_root / "pkg"
            docs_root.mkdir()
            package.mkdir(parents=True)
            (docs_root / "guide.md").write_text(
                "# Guide\n\n"
                "{py:func}`pkg.available`\n"
                ":func:`pkg.available`\n"
                "``:func:`pkg.literal` ``\n"
                "````markdown\n"
                "```\n"
                ":func:`pkg.in_a_fence`\n"
                "```\n"
                "````\n",
                encoding="utf-8",
            )
            (package / "__init__.py").write_text(
                f'"""API using :func:`{reference}`."""\n\n'
                "def available():\n    return None\n",
                encoding="utf-8",
            )
            report_path = root / "report.txt"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_doc_symbols.py")),
                    "--docs-root",
                    str(docs_root),
                    "--python-root",
                    str(python_root),
                    "--work-dir",
                    str(root / "work"),
                    "--output",
                    str(report_path),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            if not report_path.is_file():
                self.fail(
                    "checker did not create its report\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            report = report_path.read_text(encoding="utf-8")
        return result, report

    def test_cli_accepts_resolvable_symbol(self) -> None:
        result, report = self._run_checker("pkg.available")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Markdown symbol references: 2", report)
        self.assertIn("Python docstring symbol references: 1", report)
        self.assertIn("broken local references: 0", report)

    def test_cli_reports_missing_symbol_and_exits_one(self) -> None:
        result, report = self._run_checker("pkg.missing")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Markdown symbol references: 2", report)
        self.assertIn("Python docstring symbol references: 1", report)
        self.assertIn("broken local references: 1", report)
        self.assertIn(
            "python/pkg/__init__.py:1: error: unresolved local py:func "
            "target 'pkg.missing' (docstring)",
            report,
        )


if __name__ == "__main__":
    unittest.main()
