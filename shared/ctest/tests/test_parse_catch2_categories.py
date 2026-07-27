# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_CTEST_DIR = _TESTS_DIR.parent
_PARSER = _CTEST_DIR / "parse_catch2_categories.py"

sys.path.insert(0, str(_CTEST_DIR))
import parse_catch2_categories as pcc  # noqa: E402

MINIMAL_YAML = """
test_categories:
  quick:
    test_tags:
      - "[smoke]"
      - "[unit]"
    labels:
      - quick
      - pre-commit
execution_settings:
  category_timeouts:
    quick: 60
"""

FULL_YAML = """
test_categories:
  quick:
    test_tags:
      - "[smoke]"
      - "[unit]"
    exclude_tags:
      - "[slow]"
    exclude_tags_windows:
      - "[linux-only]"
    exclude_tags_linux:
      - "[windows-only]"
    labels:
      - quick
  all_tests:
    test_tags:
      - "[]"
    labels:
      - full
  excludes_only:
    exclude_tags:
      - "[slow]"
      - "[flaky]"
    labels:
      - comprehensive
execution_settings:
  timeout_multiplier: 2
  category_timeouts:
    quick: 60
    all_tests: 3600
    excludes_only: 300
  environment:
    CATCH2_LOG_LEVEL: info
"""


class TestValidateTag(unittest.TestCase):
    def test_accepts_valid_tags(self):
        for tag in ("[smoke]", "[unit]", "[]", "~[slow]", "[a-b.c_d]", "[abc*]"):
            self.assertIsNone(pcc.validate_tag(tag), msg=tag)

    def test_rejects_invalid_tags(self):
        for tag in ("smoke", "[smoke", "smoke]", "[smoke ]", "", "~smoke"):
            self.assertIsNotNone(pcc.validate_tag(tag), msg=tag)

    def test_rejects_non_strings(self):
        for tag in (123, None, ["[smoke]"]):
            err = pcc.validate_tag(tag)
            self.assertIsNotNone(err)
            self.assertIn("must be a string", err)


class TestValidateIdentifier(unittest.TestCase):
    def test_accepts_safe(self):
        for value in ("quick", "ex_gpu_gfx1150", "pre-commit", "v1.2.3"):
            self.assertIsNone(pcc.validate_identifier(value), msg=value)

    def test_rejects_unsafe(self):
        for value in ("bad name", "has/slash", "has*", ""):
            self.assertIsNotNone(pcc.validate_identifier(value), msg=value)


class TestValidateCategories(unittest.TestCase):
    def test_accepts_minimal(self):
        categories = {"quick": {"test_tags": ["[smoke]"], "labels": ["quick"]}}
        self.assertEqual(pcc.validate_categories(categories, False, True), [])

    def test_rejects_non_mapping(self):
        errors = pcc.validate_categories(["not", "a", "dict"], False, True)
        self.assertTrue(any("must be a mapping" in e for e in errors))

    def test_rejects_invalid_test_tag(self):
        categories = {"quick": {"test_tags": ["[smoke"], "labels": ["quick"]}}
        errors = pcc.validate_categories(categories, False, True)
        self.assertTrue(
            any("test_tags" in e and "Invalid tag syntax" in e for e in errors)
        )

    def test_rejects_invalid_exclude_tag(self):
        categories = {
            "quick": {
                "test_tags": ["[smoke]"],
                "exclude_tags": ["not a tag"],
                "labels": ["quick"],
            }
        }
        errors = pcc.validate_categories(categories, False, True)
        self.assertTrue(any("exclude_tags" in e for e in errors))

    def test_os_specific_excludes_only_checked_on_that_os(self):
        categories = {
            "quick": {
                "test_tags": ["[smoke]"],
                "exclude_tags_linux": ["not a tag"],
                "labels": ["quick"],
            }
        }
        # On Windows the bad linux-only tag is ignored; on Linux it errors.
        self.assertEqual(pcc.validate_categories(categories, True, False), [])
        self.assertTrue(pcc.validate_categories(categories, False, True))

    def test_rejects_unsafe_label(self):
        categories = {"quick": {"test_tags": ["[smoke]"], "labels": ["bad label"]}}
        errors = pcc.validate_categories(categories, False, True)
        self.assertTrue(any("label" in e for e in errors))


class TestBuildCatch2TagExpression(unittest.TestCase):
    def test_only_includes(self):
        self.assertEqual(
            pcc.build_catch2_tag_expression(["[smoke]", "[unit]"], []),
            "[smoke],[unit]",
        )

    def test_only_excludes(self):
        self.assertEqual(
            pcc.build_catch2_tag_expression([], ["[slow]", "[flaky]"]),
            "~[slow] ~[flaky]",
        )

    def test_include_and_exclude_distributes_excludes(self):
        # Catch2: ',' is OR, ' ' is AND, '~' negates. Because ',' binds looser
        # than ' ', excludes are duplicated per include clause.
        self.assertEqual(
            pcc.build_catch2_tag_expression(["[smoke]", "[unit]"], ["[slow]"]),
            "[smoke] ~[slow],[unit] ~[slow]",
        )

    def test_multiple_excludes(self):
        self.assertEqual(
            pcc.build_catch2_tag_expression(["[a]", "[b]"], ["[x]", "[y]"]),
            "[a] ~[x] ~[y],[b] ~[x] ~[y]",
        )

    def test_all_tests_sentinel_skipped(self):
        self.assertEqual(pcc.build_catch2_tag_expression(["[]"], []), "")

    def test_empty_inputs(self):
        self.assertEqual(pcc.build_catch2_tag_expression([], []), "")
        self.assertEqual(pcc.build_catch2_tag_expression(None, None), "")

    def test_sentinel_with_excludes_returns_excludes_only(self):
        self.assertEqual(pcc.build_catch2_tag_expression(["[]"], ["[slow]"]), "~[slow]")


class TestLoadYaml(unittest.TestCase):
    def test_loads_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(MINIMAL_YAML, encoding="utf-8")
            data = pcc.load_yaml(path)
            self.assertIn("test_categories", data)
            self.assertIn("quick", data["test_categories"])

    def test_missing_file_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                pcc.load_yaml(Path(tmp) / "missing.yaml")
            self.assertEqual(ctx.exception.code, 1)


class TestCliIntegration(unittest.TestCase):
    def _run_parser(self, yaml_text, target="rr-tests", install_file=None):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "test_categories.yaml"
            yaml_path.write_text(yaml_text, encoding="utf-8")
            install_path = None
            cmd = [sys.executable, str(_PARSER), str(yaml_path), target, tmp]
            if install_file is not None:
                install_path = Path(tmp) / install_file
                cmd.append(str(install_path))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            install_contents = (
                install_path.read_text(encoding="utf-8") if install_path else None
            )
            return result, install_contents

    def test_minimal_emits_suite_and_tag_expression(self):
        result, _ = self._run_parser(MINIMAL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("NAME rr-tests_quick_suite", result.stdout)
        self.assertIn('COMMAND rr-tests "[smoke],[unit]"', result.stdout)
        self.assertIn('LABELS "quick;pre-commit"', result.stdout)
        self.assertIn("TIMEOUT 60", result.stdout)

    def test_timeout_multiplier_applied(self):
        result, _ = self._run_parser(FULL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # quick timeout 60 * multiplier 2 == 120
        self.assertIn("TIMEOUT 120", result.stdout)

    def test_distributes_excludes_per_include(self):
        result, _ = self._run_parser(FULL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # The 'quick' category has 2 includes and [slow] + (linux) [windows-only]
        # excludes; each include clause must repeat "~[slow]".
        quick_line = next(
            line
            for line in result.stdout.splitlines()
            if "COMMAND rr-tests" in line and "[smoke]" in line
        )
        self.assertEqual(quick_line.count("~[slow]"), 2)

    def test_all_tests_sentinel_runs_bare_binary(self):
        result, _ = self._run_parser(FULL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("NAME rr-tests_all_tests_suite", result.stdout)
        # The [] sentinel with no excludes yields a bare COMMAND (no tag arg):
        # a COMMAND line that is exactly the binary with no quoted expression.
        self.assertIn("\n  COMMAND rr-tests\n", result.stdout)

    def test_excludes_only_category(self):
        result, _ = self._run_parser(FULL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('COMMAND rr-tests "~[slow] ~[flaky]"', result.stdout)

    def test_environment_propagated(self):
        result, _ = self._run_parser(FULL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('ENVIRONMENT "CATCH2_LOG_LEVEL=info"', result.stdout)

    def test_install_file_uses_relative_path(self):
        result, install_contents = self._run_parser(
            MINIMAL_YAML, install_file="install_CTestTestfile.cmake"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIsNotNone(install_contents)
        self.assertIn('add_test(rr-tests_quick_suite "../rr-tests"', install_contents)
        self.assertIn('"[smoke],[unit]"', install_contents)
        self.assertIn('LABELS "quick;pre-commit"', install_contents)

    def test_invalid_tag_exits_nonzero(self):
        bad_yaml = (
            "test_categories:\n"
            "  quick:\n"
            '    test_tags: ["[smoke"]\n'
            "    labels: [quick]\n"
        )
        result, _ = self._run_parser(bad_yaml)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid tag syntax", result.stderr)

    def test_invalid_identifier_exits_nonzero(self):
        bad_yaml = (
            "test_categories:\n"
            '  "bad/name":\n'
            '    test_tags: ["[smoke]"]\n'
            "    labels: [quick]\n"
        )
        result, _ = self._run_parser(bad_yaml)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
