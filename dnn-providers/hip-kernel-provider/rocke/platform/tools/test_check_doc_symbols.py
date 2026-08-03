# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_doc_symbols import (
    ReferenceLocation,
    build_symbol_inventory,
    find_sphinx_references,
    normalize_sphinx_roles,
    parse_missing_references,
    render_inventory,
)


class CheckDocSymbolsTest(unittest.TestCase):
    def test_normalizes_roles_outside_fenced_code(self) -> None:
        source = (
            ":func:`pkg.good`\n"
            "```python\n"
            ":class:`pkg.Literal`\n"
            "```\n"
            "{py:meth}`pkg.Thing.method`\n"
        )

        normalized, replacements = normalize_sphinx_roles(source)

        self.assertEqual(replacements, 1)
        self.assertIn("{py:func}`pkg.good`", normalized)
        self.assertIn(":class:`pkg.Literal`", normalized)
        self.assertIn("{py:meth}`pkg.Thing.method`", normalized)
        self.assertEqual(source.count("\n"), normalized.count("\n"))

        references = find_sphinx_references(normalized, "guide.md")
        self.assertEqual(
            [(item.line, item.role, item.target) for item in references],
            [(1, "func", "pkg.good"), (5, "meth", "pkg.Thing.method")],
        )

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
            docs_root = root / "platform" / "dsl_docs"
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
                [
                    ReferenceLocation(
                        path="guide.md",
                        line=11,
                        role="func",
                        target="pkg.missing",
                    )
                ],
            )

        self.assertEqual(len(broken), 1)
        self.assertEqual(broken[0].path, "dsl_docs/guide.md")
        self.assertEqual(broken[0].line, 11)
        self.assertEqual(broken[0].role, "func")
        self.assertEqual(broken[0].target, "pkg.missing")


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
                f"# Guide\n\n{{py:func}}`{reference}`\n", encoding="utf-8"
            )
            (package / "__init__.py").write_text(
                "def available():\n    return None\n", encoding="utf-8"
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
        self.assertIn("Sphinx symbol references: 1", report)
        self.assertIn("broken references: 0", report)

    def test_cli_reports_missing_symbol_and_exits_one(self) -> None:
        result, report = self._run_checker("pkg.missing")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Sphinx symbol references: 1", report)
        self.assertIn("broken references: 1", report)
        self.assertIn(
            "docs/guide.md:3: error: unresolved py:func target 'pkg.missing'",
            report,
        )


if __name__ == "__main__":
    unittest.main()
