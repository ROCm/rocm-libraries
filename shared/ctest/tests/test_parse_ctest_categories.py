# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_CTEST_DIR = _TESTS_DIR.parent
_PARSER = _CTEST_DIR / "parse_ctest_categories.py"

sys.path.insert(0, str(_CTEST_DIR))
import parse_ctest_categories as pcc  # noqa: E402

# A test_categories.yaml for the "pre-registered CTest" flavor: patterns are
# matched against test names already registered via add_test().
SAMPLE_YAML = """
test_categories:
  quick:
    test_patterns:
      - "smoke_test"
      - "unit_.*"
    exclude:
      - "unit_broken"
    labels:
      - pre-commit
  standard:
    test_patterns:
      - "integration_.*"
    labels: []
general_exclude:
  exclude_gpu_gfx942:
    test_patterns:
      - "gfx942_fail"
    labels:
      - ex_gpu_gfx942
"""

# The install-tree entries a project would have already written into the
# install CTestTestfile.cmake before the parser appends label properties.
INSTALL_ADD_TESTS = (
    "add_test(smoke_test ../mylib-test)\n"
    "add_test(unit_alpha ../mylib-test)\n"
    "add_test(unit_broken ../mylib-test)\n"
    "add_test(integration_one ../mylib-test)\n"
    "add_test(gfx942_fail ../mylib-test)\n"
)


class TestIsRegexPattern(unittest.TestCase):
    def test_bare_names_are_not_regex(self):
        for name in ("smoke_test", "unit-alpha", "Test123"):
            self.assertFalse(pcc.is_regex_pattern(name), msg=name)

    def test_metacharacters_are_regex(self):
        for name in ("unit_.*", "a|b", "foo$", "gfx.*fail", "a+b"):
            self.assertTrue(pcc.is_regex_pattern(name), msg=name)


class TestScanAddTestNames(unittest.TestCase):
    def test_parses_bare_and_quoted_names_dedup_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CTestTestfile.cmake"
            path.write_text(
                "add_test(alpha ../t)\n"
                "add_test(NAME beta COMMAND ../t)\n"
                'add_test("gamma smoke" ../t)\n'
                "add_test(alpha ../t)\n",  # duplicate -> dropped
                encoding="utf-8",
            )
            names = pcc.scan_add_test_names(path)
            self.assertEqual(names, ["alpha", "beta", "gamma smoke"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(pcc.scan_add_test_names("/no/such/file"), [])


class TestTierLabels(unittest.TestCase):
    def test_quick_inherits_all_tiers(self):
        self.assertEqual(
            pcc.tier_labels("quick"), ["quick", "standard", "comprehensive", "full"]
        )

    def test_standard_and_full(self):
        self.assertEqual(
            pcc.tier_labels("standard"), ["standard", "comprehensive", "full"]
        )
        self.assertEqual(pcc.tier_labels("full"), ["full"])

    def test_unknown_category_returns_itself(self):
        self.assertEqual(pcc.tier_labels("weekly"), ["weekly"])


class TestParseYaml(unittest.TestCase):
    def _parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(text, encoding="utf-8")
            return pcc.parse_yaml(path)

    def test_returns_categories_and_general_excludes(self):
        categories, general = self._parse(SAMPLE_YAML)
        self.assertIn("quick", categories)
        self.assertIn("standard", categories)
        self.assertIn("exclude_gpu_gfx942", general)

    def test_sets_defaults_for_missing_keys(self):
        categories, _ = self._parse(
            "test_categories:\n  quick:\n    labels: [pre-commit]\n"
        )
        self.assertEqual(categories["quick"]["test_patterns"], [])
        self.assertEqual(categories["quick"]["exclude"], [])
        self.assertEqual(categories["quick"]["test_labels"], [])

    def test_normalizes_non_dict_category_to_patterns(self):
        # A category whose value is a bare list is treated as test_patterns.
        categories, _ = self._parse(
            "test_categories:\n  quick:\n    - smoke_test\n    - unit_.*\n"
        )
        self.assertEqual(
            categories["quick"]["test_patterns"], ["smoke_test", "unit_.*"]
        )
        self.assertEqual(categories["quick"]["labels"], [])


class TestComputeInstallLabels(unittest.TestCase):
    def _labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(SAMPLE_YAML, encoding="utf-8")
            categories, general = pcc.parse_yaml(path)
        test_names = [
            "smoke_test",
            "unit_alpha",
            "unit_broken",
            "integration_one",
            "gfx942_fail",
        ]
        return pcc._compute_install_labels(categories, general, test_names)

    def test_exact_and_regex_patterns_get_tier_labels(self):
        labels = self._labels()
        self.assertEqual(
            labels["smoke_test"],
            ["quick", "standard", "comprehensive", "full", "pre-commit"],
        )
        # unit_alpha matches the "unit_.*" regex, so it gets the quick tier too.
        self.assertIn("quick", labels["unit_alpha"])

    def test_exclude_adds_category_exclude_label(self):
        labels = self._labels()
        self.assertIn("quick_exclude", labels["unit_broken"])

    def test_standard_category_does_not_get_quick_label(self):
        labels = self._labels()
        self.assertNotIn("quick", labels["integration_one"])
        self.assertIn("standard", labels["integration_one"])

    def test_general_exclude_labels_applied(self):
        labels = self._labels()
        self.assertEqual(labels["gfx942_fail"], ["ex_gpu_gfx942"])


class TestCliIntegration(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(_PARSER), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_build_tree_emits_exact_and_regex_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "c.yaml"
            yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")
            result = self._run(str(yaml_path))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # Exact pattern -> literal foreach entry.
        self.assertIn("smoke_test", result.stdout)
        # Regex pattern -> MATCHES branch.
        self.assertIn('if(_test MATCHES "^unit_.*$")', result.stdout)
        # Tier inheritance is reflected in the labels string.
        self.assertIn(
            'LABELS "quick;standard;comprehensive;full;pre-commit"', result.stdout
        )
        # Exclude produces a <category>_exclude label.
        self.assertIn('LABELS "quick_exclude"', result.stdout)

    def test_print_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "c.yaml"
            yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")
            result = self._run("--print-categories", str(yaml_path))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "quick;standard")

    def test_install_tree_emits_set_tests_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "c.yaml"
            yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")
            install_path = Path(tmp) / "install_CTestTestfile.cmake"
            install_path.write_text(INSTALL_ADD_TESTS, encoding="utf-8")
            result = self._run(str(yaml_path), str(install_path))
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            contents = install_path.read_text(encoding="utf-8")
        # Install tree uses set_tests_properties (ctest -L honours only this
        # form), with regex patterns expanded against the scanned test names.
        self.assertIn(
            'set_tests_properties("smoke_test" PROPERTIES LABELS '
            '"quick;standard;comprehensive;full;pre-commit")',
            contents,
        )
        self.assertIn(
            'set_tests_properties("unit_alpha" PROPERTIES LABELS '
            '"quick;standard;comprehensive;full;pre-commit")',
            contents,
        )
        # unit_broken picks up both the quick tier and the quick_exclude label.
        self.assertIn("quick_exclude", contents)
        # general_exclude label lands on the matching test.
        self.assertIn(
            'set_tests_properties("gfx942_fail" PROPERTIES LABELS "ex_gpu_gfx942")',
            contents,
        )

    def test_explicit_tests_override_expands_regex(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "c.yaml"
            yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")
            install_path = Path(tmp) / "install_CTestTestfile.cmake"
            install_path.write_text(INSTALL_ADD_TESTS, encoding="utf-8")
            result = self._run(
                str(yaml_path),
                str(install_path),
                "--explicit-tests",
                "unit_alpha;integration_one",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            contents = install_path.read_text(encoding="utf-8")
        # Only the explicitly named tests get labelled.
        self.assertIn('set_tests_properties("unit_alpha"', contents)
        self.assertIn('set_tests_properties("integration_one"', contents)
        self.assertNotIn('set_tests_properties("smoke_test"', contents)

    def test_missing_input_file_exits_nonzero(self):
        result = self._run("/no/such/file.yaml")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
