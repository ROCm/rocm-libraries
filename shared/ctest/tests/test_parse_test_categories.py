# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_CTEST_DIR = _TESTS_DIR.parent
_PARSER = _CTEST_DIR / "parse_test_categories.py"

sys.path.insert(0, str(_CTEST_DIR))
import parse_test_categories as ptc  # noqa: E402


MINIMAL_YAML = """
test_categories:
  quick:
    test_yaml: smoke.yaml
    test_patterns:
      - "*quick*"
    exclude:
      - "*known_bug*"
    labels:
      - quick
execution_settings:
  category_timeouts:
    quick: 60
"""


class TestHelperFunctions(unittest.TestCase):
    def test_format_extra_args_empty(self):
        self.assertEqual(ptc._format_extra_args([]), "")
        self.assertEqual(ptc._format_extra_args(None), "")

    def test_format_extra_args_quotes_spaces(self):
        out = ptc._format_extra_args(["--flag", "value with spaces"])
        self.assertIn("'value with spaces'", out)
        self.assertTrue(out.startswith(" "))

    def test_rtest_script_basename(self):
        self.assertEqual(ptc._rtest_script_basename("rocblas-test"), "rocblas_rtest.py")
        self.assertEqual(ptc._rtest_script_basename("hipblas-test"), "hipblas_rtest.py")
        self.assertEqual(ptc._rtest_script_basename("my-lib-test"), "my-lib_rtest.py")
        self.assertEqual(
            ptc._rtest_script_basename("custom_gtest"), "custom_gtest_rtest.py"
        )

    def test_format_gtest_command_tail_yaml_and_filter(self):
        tail = ptc._format_gtest_command_tail(
            "rocblas-test",
            "*quick*-*known_bug*",
            "",
            test_yaml="rocblas_smoke.yaml",
        )
        self.assertEqual(
            tail,
            "rocblas-test --yaml rocblas_smoke.yaml --gtest_filter=*quick*-*known_bug*",
        )

    def test_format_gtest_command_tail_yaml_only(self):
        tail = ptc._format_gtest_command_tail(
            "rocblas-test", "", "", test_yaml="rocblas_smoke.yaml"
        )
        self.assertEqual(tail, "rocblas-test --yaml rocblas_smoke.yaml")

    def test_format_category_command_rtest_driver(self):
        cmd = ptc._format_category_command(
            True,
            "rocblas-test",
            "rocblas-test",
            "*quick*",
            "",
            "quick",
        )
        self.assertEqual(cmd, '"${Python3_EXECUTABLE}" rocblas_rtest.py -t ctest_quick')

    def test_format_category_command_direct_gtest(self):
        cmd = ptc._format_category_command(
            False,
            "rocblas-test",
            "rocblas-test",
            "*quick*-*known_bug*",
            " --iterations 2",
            "quick",
            test_yaml="smoke.yaml",
        )
        self.assertIn("--yaml smoke.yaml", cmd)
        self.assertIn("--gtest_filter=*quick*-*known_bug*", cmd)
        self.assertIn(" --iterations 2", cmd)

    def test_format_install_add_test_line_direct(self):
        line = ptc._format_install_add_test_line(
            False,
            "rocblas-test",
            "quick",
            "rocblas-test",
            "*quick*",
            "",
            "/usr/bin/python3",
            test_yaml="smoke.yaml",
        )
        self.assertIn('add_test(rocblas-test_quick_suite "../rocblas-test"', line)
        self.assertIn("--yaml smoke.yaml", line)


class TestValidation(unittest.TestCase):
    def test_validate_identifier_rejects_unsafe(self):
        self.assertIsNotNone(ptc.validate_identifier("bad;name"))

    def test_validate_gtest_pattern_accepts_wildcards(self):
        self.assertIsNone(ptc.validate_gtest_pattern("*quick*"))
        self.assertIsNone(ptc.validate_gtest_pattern("*pre_checkin*"))

    def test_validate_config_requires_patterns_or_yaml(self):
        errors = ptc.validate_config(
            {"quick": {"labels": ["quick"]}}, None, False, True
        )
        self.assertTrue(any("test_patterns and/or test_yaml" in e for e in errors))

    def test_validate_config_accepts_test_yaml_only(self):
        errors = ptc.validate_config(
            {"quick": {"test_yaml": "smoke.yaml", "labels": ["quick"]}},
            None,
            False,
            True,
        )
        self.assertEqual(errors, [])

    def test_parse_exclude_gpu_key(self):
        self.assertEqual(
            ptc.parse_exclude_gpu_key("exclude_gpu_gfx942"), ("gfx942", None)
        )
        self.assertEqual(
            ptc.parse_exclude_gpu_key("exclude_gpu_gfx942_linux"), ("gfx942", "linux")
        )
        self.assertEqual(ptc.parse_exclude_gpu_key("invalid"), (None, None))

    def test_exclude_gpu_key_applies(self):
        self.assertTrue(ptc.exclude_gpu_key_applies(None, False, True))
        self.assertTrue(ptc.exclude_gpu_key_applies("linux", False, True))
        self.assertFalse(ptc.exclude_gpu_key_applies("windows", False, True))

    def test_gpu_arch_matches(self):
        self.assertTrue(ptc.gpu_arch_matches("gfx1150", "gfx1150"))
        self.assertTrue(ptc.gpu_arch_matches("gfx1150", "gfx115X"))
        self.assertFalse(ptc.gpu_arch_matches("gfx1150", "gfx942"))


class TestCliIntegration(unittest.TestCase):
    def _run_parser(self, yaml_text, *extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "test_categories.yaml"
            yaml_path.write_text(yaml_text, encoding="utf-8")
            cmd = [
                sys.executable,
                str(_PARSER),
                str(yaml_path),
                "rocblas-test",
                tmp,
                *extra_args,
            ]
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_cli_direct_gtest_emits_yaml_and_filter(self):
        result = self._run_parser(MINIMAL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "COMMAND rocblas-test --yaml smoke.yaml --gtest_filter=*quick*-*known_bug*",
            result.stdout,
        )
        self.assertIn('LABELS "quick"', result.stdout)

    def test_cli_rtest_driver_emits_rtest_command(self):
        result = self._run_parser(MINIMAL_YAML, "--use-rtest-driver")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            'COMMAND "${Python3_EXECUTABLE}" rocblas_rtest.py -t ctest_quick',
            result.stdout,
        )

    def test_cli_invalid_yaml_exits_nonzero(self):
        result = self._run_parser("test_categories:\n  bad:\n    labels: [quick]\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation error", result.stderr)


if __name__ == "__main__":
    unittest.main()
