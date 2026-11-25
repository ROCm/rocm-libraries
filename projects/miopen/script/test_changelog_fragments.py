# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import changelog_fragments as fragments


class ChangelogFragmentsTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pending = self.root / "changelog.d"
        self.pending.mkdir()
        self.readme = self.pending / "README.md"
        self.readme.write_text("Instructions\n", encoding="utf-8")
        self.changelog = self.root / "CHANGELOG.md"
        self.history = "## MIOpen 3.0\n\n### Resolved Issues\n* Shipped fix.\n"
        self.active = "# Changelog\n\n## MIOpen 4.0 (Unreleased)\n\n### Resolved Issues\n* Existing fix.\n\n"
        self.changelog.write_text(self.active + self.history, encoding="utf-8")

    def run_cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                fragments.__file__,
                "--project",
                str(self.root),
                *map(str, args),
            ],
            capture_output=True,
            text=True,
        )

    def entry(self, name, text):
        path = self.pending / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_fold_orders_entries_and_preserves_released_history(self):
        second = self.entry("22.resolved.md", "* Second fix.\n  More detail.\n")
        first = self.entry("11.resolved.md", "* First fix.\n")
        added = self.entry("33.added.md", "* New feature.\n")
        self.assertNotEqual(self.run_cli("--check-empty").returncode, 0)
        result = self.run_cli("--fold")
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = (
            self.active.rstrip()
            + "\n* First fix.\n* Second fix.\n  More detail.\n\n\n### Added\n* New feature.\n\n"
            + self.history
        )
        self.assertEqual(self.changelog.read_text(encoding="utf-8"), expected)
        self.assertEqual(set(self.pending.iterdir()), {self.readme})
        for path in (first, second, added):
            self.assertFalse(path.exists())
        self.assertEqual(self.run_cli("--check-empty").returncode, 0)
        self.assertEqual(self.run_cli("--fold").returncode, 0)
        self.assertEqual(self.changelog.read_text(encoding="utf-8"), expected)

    def test_invalid_fragment_prevents_any_consumption(self):
        good = self.entry("11.added.md", "* A feature.\n")
        bad = self.entry("22.resolved.md", "* A fix.\n## Another release\n")
        result = self.run_cli("--fold")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(bad.name, result.stderr)
        self.assertTrue(good.exists())
        self.assertTrue(bad.exists())
        self.assertEqual(
            self.changelog.read_text(encoding="utf-8"), self.active + self.history
        )

    def test_refuses_to_write_into_a_shipped_release(self):
        note = self.entry("11.added.md", "* A feature.\n")
        self.changelog.write_text(self.history, encoding="utf-8")
        result = self.run_cli("--fold")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("(Unreleased)", result.stderr)
        self.assertTrue(note.exists())
        self.assertEqual(self.changelog.read_text(encoding="utf-8"), self.history)

    def test_validation_rejects_symlink_and_bad_category(self):
        bad = self.entry("11.fixed.md", "* A fix.\n")
        link = self.pending / "22.resolved.md"
        link.symlink_to(self.changelog)
        for path in (bad, link):
            with self.subTest(path=path):
                result = self.run_cli("--validate", path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(path.name, result.stderr)


if __name__ == "__main__":
    unittest.main()
