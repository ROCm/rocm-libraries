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
import parse_ctest_categories as pctc  # noqa: E402


class TestTimeoutGeneration(unittest.TestCase):
    def setUp(self):
        self.categories = {
            "quick": {
                "test_patterns": ["fast_test"],
                "test_labels": [],
                "exclude": [],
                "labels": [],
            },
            "standard": {
                "test_patterns": ["standard_test"],
                "test_labels": [],
                "exclude": [],
                "labels": [],
            },
        }
        self.execution_settings = {
            "default_timeout": 45,
            "timeout_multiplier": 2,
            "category_timeouts": {"quick": 60, "standard": 900},
        }

    def test_category_timeout_uses_category_value_and_multiplier(self):
        self.assertEqual(
            pctc.category_timeout("quick", self.execution_settings), 120
        )

    def test_category_timeout_falls_back_to_default(self):
        self.assertEqual(
            pctc.category_timeout("custom", self.execution_settings), 90
        )

    def test_missing_execution_settings_preserve_label_only_output(self):
        code = pctc.generate_cmake(self.categories, explicit_tests=["fast_test"])
        self.assertNotIn("TIMEOUT", code)

    def test_explicit_tests_limit_exact_match_output(self):
        code = pctc.generate_cmake(self.categories, explicit_tests=["fast_test"])
        self.assertIn("fast_test", code)
        self.assertNotIn("standard_test", code)

    def test_build_tree_output_emits_timeouts(self):
        code = pctc.generate_cmake(
            self.categories,
            explicit_tests=["fast_test", "standard_test"],
            execution_settings=self.execution_settings,
        )
        self.assertIn("PROPERTY TIMEOUT 120", code)
        self.assertIn("PROPERTY TIMEOUT 1800", code)

    def test_install_tree_uses_minimum_matching_tier_timeout(self):
        self.categories["standard"]["test_patterns"].append("fast_test")
        code = pctc.generate_cmake_install(
            self.categories,
            {},
            ["fast_test", "standard_test"],
            self.execution_settings,
        )
        self.assertIn(
            'set_tests_properties("fast_test" PROPERTIES '
            'LABELS "quick;standard;comprehensive;full" TIMEOUT 120)',
            code,
        )
        self.assertIn(
            'set_tests_properties("standard_test" PROPERTIES '
            'LABELS "standard;comprehensive;full" TIMEOUT 1800)',
            code,
        )

    def test_install_tree_preserves_gpu_exclusion_labels(self):
        general_excludes = {
            "exclude_gpu_gfx942": {
                "test_patterns": ["mx_test"],
                "labels": ["ex_gpu_gfx942"],
            }
        }
        code = pctc.generate_cmake_install(
            self.categories,
            general_excludes,
            ["fast_test", "mx_test"],
            self.execution_settings,
        )
        self.assertIn(
            'set_tests_properties("mx_test" PROPERTIES '
            'LABELS "ex_gpu_gfx942")',
            code,
        )


class TestCliIntegration(unittest.TestCase):
    def test_install_file_gets_labels_and_timeout(self):
        yaml_text = """
test_categories:
  quick:
    test_patterns:
      - fast_test
execution_settings:
  default_timeout: 300
  category_timeouts:
    quick: 60
"""
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "test_categories.yaml"
            install_path = Path(tmp) / "CTestTestfile.cmake"
            yaml_path.write_text(yaml_text, encoding="utf-8")
            install_path.write_text(
                'add_test(fast_test "../fast_test")\n', encoding="utf-8"
            )

            subprocess.run(
                [sys.executable, str(_PARSER), str(yaml_path), str(install_path)],
                check=True,
                text=True,
                capture_output=True,
            )

            generated = install_path.read_text(encoding="utf-8")
            self.assertIn(
                'set_tests_properties("fast_test" PROPERTIES '
                'LABELS "quick;standard;comprehensive;full" TIMEOUT 60)',
                generated,
            )


if __name__ == "__main__":
    unittest.main()
