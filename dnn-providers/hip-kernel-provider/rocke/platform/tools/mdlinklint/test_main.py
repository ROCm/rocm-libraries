# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the standard-library Markdown link linter."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from main import (
    MAX_MARKDOWN_BYTES,
    Linter,
    github_anchor,
    links_in_line,
    markdown_files,
    run,
)


class MarkdownLinkLinterTest(unittest.TestCase):
    """Exercise parsing, filesystem containment, and CLI diagnostics."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    def write(self, relative: str, contents: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def linter(self, root: Path | None = None, link_root: Path | None = None) -> Linter:
        scan_root = root or self.root
        return Linter(scan_root, link_root or scan_root)

    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = run(arguments)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_links_in_line(self) -> None:
        links = links_in_line(
            "[good](one.md) and [`also good`](two.md#section) "
            "and `[not a link](no.md)`"
        )
        self.assertEqual([link.target for link in links], ["one.md", "two.md#section"])

    def test_link_parser_handles_destinations_escapes_and_unicode_columns(self) -> None:
        links = links_in_line(
            "é [angle](<a b.md>) [nested](dir/a_(b).md) "
            r"\[escaped](no.md) [title](yes.md \"Title\")"
        )
        self.assertEqual(
            [(link.target, link.column) for link in links],
            [("a b.md", 12), ("dir/a_(b).md", 30), ("yes.md", 70)],
        )

    def test_resolves_paths_fragments_directories_and_external_links(self) -> None:
        self.write(
            "guide.md",
            "# Guide title\n\n## Repeated Heading\n\n## Repeated Heading\n",
        )
        source = self.write(
            "index.md",
            "[guide](guide.md#guide-title)\n"
            "[duplicate](guide.md#repeated-heading-1)\n"
            "[directory](assets/)\n"
            "[remote](https://example.com/docs)\n"
            "[anchor](#local-anchor)\n"
            "\n"
            "# Local anchor",
        )
        (self.root / "assets").mkdir()

        linter = self.linter()
        self.assertEqual(linter.lint_file(source), [])
        self.assertEqual(linter.checked_links, 5)

    def test_reports_unresolved_path_and_fragment(self) -> None:
        source = self.write(
            "index.md", "[missing](missing.md)\n[fragment](guide.md#missing)\n"
        )
        self.write("guide.md", "# Present\n")

        diagnostics = self.linter().lint_file(source)
        self.assertEqual(len(diagnostics), 2)
        self.assertIn("unresolved local link", diagnostics[0].message)
        self.assertIn("unresolved fragment", diagnostics[1].message)

    def test_enforces_link_root_independently_from_scan_root(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        self.write("guide.md", "# Guide\n")
        source = self.write("docs/index.md", "[guide](../guide.md#guide)\n")

        diagnostics = self.linter(docs, docs).lint_file(source)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("escapes link root", diagnostics[0].message)
        self.assertEqual(self.linter(docs, self.root).lint_file(source), [])

    def test_rejects_symlink_escape(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        outside = self.write("outside.md", "# Outside\n")
        try:
            (docs / "escape.md").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"cannot create symlink: {error}")
        source = self.write("docs/index.md", "[escape](escape.md#outside)\n")

        diagnostics = self.linter(docs, docs).lint_file(source)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("through a symbolic link", diagnostics[0].message)

    def test_rejects_symlinked_markdown_source(self) -> None:
        target = self.write("target.txt", "# Target\n")
        try:
            (self.root / "source.md").symlink_to(target)
        except OSError as error:
            self.skipTest(f"cannot create symlink: {error}")
        with self.assertRaisesRegex(ValueError, "symlinked Markdown source"):
            markdown_files(self.root)

    def test_rejects_oversized_markdown_target(self) -> None:
        large = self.write("large.md", "")
        large.write_bytes(b"x" * (MAX_MARKDOWN_BYTES + 1))
        source = self.write("index.md", "[large](large.md#heading)\n")

        diagnostics = self.linter().lint_file(source)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("limit is", diagnostics[0].message)

    def test_reports_invalid_escapes_and_absolute_paths(self) -> None:
        source = self.write(
            "index.md",
            "[path](bad%2.md)\n[fragment](index.md#bad%2)\n[absolute](/tmp/a.md)\n",
        )

        diagnostics = self.linter().lint_file(source)
        self.assertEqual(len(diagnostics), 3)
        self.assertIn("invalid percent-encoding in local link", diagnostics[0].message)
        self.assertIn("invalid percent-encoding in fragment", diagnostics[1].message)
        self.assertIn("absolute local link", diagnostics[2].message)

    def test_reports_nul_in_decoded_path_without_crashing(self) -> None:
        source = self.write("index.md", "[nul](bad%00.md)\n")
        diagnostics = self.linter().lint_file(source)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("cannot resolve local link", diagnostics[0].message)

    def test_skips_fenced_and_inline_code(self) -> None:
        source = self.write(
            "index.md",
            "`[inline](missing.md)`\n```md\n[fenced](missing.md)\n```\n",
        )
        linter = self.linter()
        self.assertEqual(linter.lint_file(source), [])
        self.assertEqual(linter.checked_links, 0)

    def test_github_anchor_unicode_punctuation_and_spacing(self) -> None:
        self.assertEqual(
            github_anchor(" Héllo,   World! _Value-2 "), "héllo-world-_value-2"
        )

    def test_cli_sorts_diagnostics_and_reports_totals(self) -> None:
        self.write("z.md", "[missing](none.md)\n")
        self.write("a.md", "[missing](none.md)\n")

        exit_code, stdout, stderr = self.invoke("--root", str(self.root))

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr, "")
        lines = stdout.splitlines()
        self.assertTrue(lines[0].startswith("a.md:1:11: error:"))
        self.assertTrue(lines[1].startswith("z.md:1:11: error:"))
        self.assertEqual(
            lines[2],
            "mdlinklint: checked 2 Markdown files and 2 links; found 2 problems",
        )

    def test_cli_quiet_suppresses_only_success_summary(self) -> None:
        self.write("index.md", "# Index\n")
        exit_code, stdout, stderr = self.invoke("--root", str(self.root), "--quiet")
        self.assertEqual((exit_code, stdout, stderr), (0, "", ""))

    def test_cli_rejects_root_outside_link_root(self) -> None:
        docs = self.root / "docs"
        boundary = self.root / "boundary"
        docs.mkdir()
        boundary.mkdir()
        exit_code, stdout, stderr = self.invoke(
            "--root", str(docs), "--link-root", str(boundary)
        )
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("is outside link root", stderr)


if __name__ == "__main__":
    unittest.main()
