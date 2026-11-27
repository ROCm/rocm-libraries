# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_CTEST_DIR = Path(__file__).resolve().parent.parent
_PARSER = _CTEST_DIR / "parse_catch2_categories.py"

sys.path.insert(0, str(_CTEST_DIR))
import parse_catch2_categories as pcc  # noqa: E402


class TestInvalidConfigExitStatus(unittest.TestCase):
    def _run(self, yaml_text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_categories.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(_PARSER), str(path), "catch2-target", directory],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_invalid_tag(self):
        result = self._run("test_categories:\n  quick:\n    test_tags: ['not a tag']\n")
        self.assertEqual(result.returncode, pcc.EXIT_INVALID_CONFIG)
        self.assertIn("validation error", result.stderr)

    def test_yaml_syntax_error(self):
        result = self._run("test_categories: [unterminated\n")
        self.assertEqual(result.returncode, pcc.EXIT_INVALID_CONFIG)
        self.assertIn("Invalid YAML syntax", result.stderr)

    def test_missing_yaml_file_is_not_an_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_PARSER),
                    str(Path(directory) / "absent.yaml"),
                    "t",
                    directory,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("YAML file not found", result.stderr)

    def test_valid_yaml_exits_zero(self):
        result = self._run("test_categories:\n  quick:\n    test_tags: ['[smoke]']\n")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
