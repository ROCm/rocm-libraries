# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import platform

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

PATTERNS_ONLY_YAML = """
test_categories:
  quick:
    test_patterns:
      - "*quick*"
    labels:
      - quick
execution_settings:
  category_timeouts:
    quick: 60
  environment:
    OMP_NUM_THREADS: "4"
"""


class TestDevelopHelperFunctions(unittest.TestCase):
    def test_cmake_quote_simple(self):
        self.assertEqual(ptc._cmake_quote("foo"), "[[foo]]")
        self.assertEqual(ptc._cmake_quote("../mylib-test"), "[[../mylib-test]]")

    def test_cmake_quote_escapes_brackets(self):
        self.assertEqual(ptc._cmake_quote("a]]b"), "[=[a]]b]=]")

    def test_format_cmake_args_empty(self):
        self.assertEqual(ptc._format_cmake_args([]), "")

    def test_format_cmake_args_multiple(self):
        out = ptc._format_cmake_args(["--test-article", "my.article"])
        self.assertEqual(out, " [[--test-article]] [[my.article]]")

    def test_dedupe_preserve_order(self):
        self.assertEqual(
            ptc._dedupe_preserve_order(["quick", "ci", "quick", "extra"]),
            ["quick", "ci", "extra"],
        )

    def test_format_gtest_command_tail_with_command_args(self):
        tail = ptc._format_gtest_command_tail(
            "mylib-test",
            "*quick*",
            "",
            command_args_string=" [[--test-article]] [[article1]]",
        )
        self.assertEqual(
            tail,
            "mylib-test [[--test-article]] [[article1]] --gtest_filter=*quick*",
        )

    def test_format_install_add_test_line_custom_executable_and_args(self):
        line = ptc._format_install_add_test_line(
            False,
            "prefix",
            "quick",
            "mylib-test",
            "*quick*",
            "",
            "/usr/bin/python3",
            install_executable="../custom/binary",
            install_command_args_string=" [[--test-config]] [[cfg.json]]",
        )
        self.assertIn("add_test(prefix_quick_suite [[../custom/binary]]", line)
        self.assertIn("[[--test-config]] [[cfg.json]]", line)
        self.assertIn("--gtest_filter=*quick*", line)


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

    def test_ctest_python_command_platform(self):
        self.assertEqual(ptc._ctest_python_command(False), "python3")
        self.assertEqual(ptc._ctest_python_command(True), "python")

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
        self.assertEqual(cmd, "python3 rocblas_rtest.py -t ctest_quick")

    def test_format_category_command_rtest_driver_windows(self):
        cmd = ptc._format_category_command(
            True,
            "rocblas-test",
            "rocblas-test",
            "*quick*",
            "",
            "quick",
            is_windows=True,
        )
        self.assertEqual(cmd, "python rocblas_rtest.py -t ctest_quick")

    def test_format_install_add_test_line_rtest_windows(self):
        line = ptc._format_install_add_test_line(
            True,
            "rocblas-test",
            "quick",
            "rocblas-test",
            "*quick*",
            "",
            None,
            is_windows=True,
        )
        self.assertIn(
            'add_test(rocblas-test_quick_suite python "../rocblas_rtest.py"', line
        )

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
        self.assertIn("add_test(rocblas-test_quick_suite [[../rocblas-test]]", line)
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
    def _run_parser(self, yaml_text, *extra_args, install_file=None):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "test_categories.yaml"
            yaml_path.write_text(yaml_text, encoding="utf-8")
            install_path = None
            cmd = [
                sys.executable,
                str(_PARSER),
                str(yaml_path),
                "rocblas-test",
                tmp,
            ]
            if install_file is not None:
                install_path = Path(tmp) / install_file
                cmd.append(str(install_path))
            cmd.extend(extra_args)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            install_contents = (
                install_path.read_text(encoding="utf-8") if install_path else None
            )
            return result, install_contents

    def test_cli_direct_gtest_emits_yaml_and_filter(self):
        result, _ = self._run_parser(MINIMAL_YAML)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "COMMAND rocblas-test --yaml smoke.yaml --gtest_filter=*quick*-*known_bug*",
            result.stdout,
        )
        self.assertIn('LABELS "quick"', result.stdout)

    def test_cli_rtest_driver_emits_rtest_command(self):
        result, _ = self._run_parser(MINIMAL_YAML, "--use-rtest-driver")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        expected_py = ptc._ctest_python_command(platform.system() == "Windows")
        self.assertIn(
            f"COMMAND {expected_py} rocblas_rtest.py -t ctest_quick",
            result.stdout,
        )

    def test_cli_command_arg_injected_before_filter(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML,
            "--command-arg=--test-article",
            "--command-arg=article1",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "COMMAND rocblas-test [[--test-article]] [[article1]] --gtest_filter=*quick*",
            result.stdout,
        )

    def test_cli_test_name_prefix(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML, "--test-name-prefix", "provider_gtest"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("NAME provider_gtest_quick_suite", result.stdout)

    def test_cli_additional_label_merged_and_deduped(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML,
            "--additional-label",
            "ci",
            "--additional-label",
            "quick",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('LABELS "quick;ci"', result.stdout)

    def test_cli_environment_override(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML, "--environment", "OMP_NUM_THREADS=24"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('ENVIRONMENT "OMP_NUM_THREADS=24"', result.stdout)

    def test_cli_environment_modification_emitted(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML,
            "--environment-modification",
            "PATH=path_list_prepend:/opt/rocm/bin",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            'ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:/opt/rocm/bin"',
            result.stdout,
        )

    def test_cli_fixtures_required_emitted(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML, "--fixtures-required", "my_setup_fixture"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('FIXTURES_REQUIRED "my_setup_fixture"', result.stdout)

    def test_cli_install_command_arg_and_executable(self):
        result, install_contents = self._run_parser(
            PATTERNS_ONLY_YAML,
            "--install-command-arg=--test-config",
            "--install-command-arg=config.json",
            "--install-executable",
            "../provider_gtest",
            install_file="install_CTestTestfile.cmake",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIsNotNone(install_contents)
        self.assertIn(
            "add_test(rocblas-test_quick_suite [[../provider_gtest]]", install_contents
        )
        self.assertIn("[[--test-config]] [[config.json]]", install_contents)
        self.assertIn("--gtest_filter=*quick*", install_contents)

    def test_cli_invalid_yaml_exits_nonzero(self):
        result, _ = self._run_parser("test_categories:\n  bad:\n    labels: [quick]\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation error", result.stderr)

    def test_cli_invalid_environment_exits_nonzero(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML, "--environment", "not-a-kv-pair"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid --environment value", result.stderr)

    def test_cli_invalid_additional_label_exits_nonzero(self):
        result, _ = self._run_parser(
            PATTERNS_ONLY_YAML, "--additional-label", "bad;label"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid --additional-label value", result.stderr)


if __name__ == "__main__":
    unittest.main()
