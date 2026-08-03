# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
