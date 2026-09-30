# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import ci_parity_runner  # noqa: E402


class ParseGtestTest(unittest.TestCase):
    def test_ok_and_failed_lines(self):
        text = "\n".join(
            [
                "[ RUN      ] hiprand_32/hiprand_api_32.hiprand_generate_test/0",
                "[       OK ] hiprand_32/hiprand_api_32.hiprand_generate_test/0 (14423 ms)",
                "[  FAILED  ] 1 test, listed below:",
                "[  FAILED  ] hiprand_32/hiprand_api_32.hiprand_generate_test/3 (88 ms)",
            ]
        )
        cases = ci_parity_runner.parse_gtest_output(text)
        self.assertEqual(
            cases,
            [
                {
                    "id": "hiprand_32/hiprand_api_32.hiprand_generate_test/0",
                    "status": "passed",
                    "seconds": 14.423,
                },
                {
                    "id": "hiprand_32/hiprand_api_32.hiprand_generate_test/3",
                    "status": "failed",
                    "seconds": 0.088,
                },
            ],
        )

    def test_ctest_prefixed_ok_line(self):
        cases = ci_parity_runner.parse_gtest_output(
            "1: [       OK ] hiprand_linkage_tests.get_version_test (0 ms)"
        )
        self.assertEqual(cases[0]["id"], "hiprand_linkage_tests.get_version_test")
        self.assertEqual(cases[0]["status"], "passed")
        self.assertEqual(cases[0]["seconds"], 0.0)

    def test_typeparam_failure_uses_the_baseline_id(self):
        text = (
            "3: [  FAILED  ] sobol_tests/1.sobol_tests, where TypeParam = "
            "sobol_test_type<hiprandStateSobol64,unsigned long long> (10 ms)"
        )
        cases = ci_parity_runner.parse_gtest_output(text)
        self.assertEqual(cases[0]["id"], "sobol_tests/1.sobol_tests")
        self.assertEqual(cases[0]["status"], "failed")
        self.assertEqual(cases[0]["seconds"], 0.01)

    def test_duplicate_lines_count_once_and_keep_a_failure(self):
        text = "\n".join(
            [
                "1: [       OK ] case/0 (10 ms)",
                "[       OK ] case/0 (10 ms)",
                "3: [       OK ] sobol/1 (5 ms)",
                "[  FAILED  ] sobol/1, where TypeParam = T (5 ms)",
            ]
        )
        cases = ci_parity_runner.parse_gtest_output(text)
        self.assertEqual([case["id"] for case in cases], ["case/0", "sobol/1"])
        self.assertEqual(cases[1]["status"], "failed")

    def test_ctest_timeout_and_failure_lines(self):
        text = "\n".join(
            [
                "1/5 Test #1: test_hiprand_api .................***Timeout 7200.12 sec",
                "3/5 Test #3: test_hiprand_kernel ..............***Failed   32.39 sec",
                "4/5 Test #4: test_hiprand_linkage .............   Passed    0.05 sec",
            ]
        )
        binaries = ci_parity_runner.parse_ctest_binaries(text)
        self.assertEqual(
            [(item["name"], item["status"]) for item in binaries],
            [
                ("test_hiprand_api", "timeout"),
                ("test_hiprand_kernel", "failed"),
                ("test_hiprand_linkage", "passed"),
            ],
        )
        self.assertEqual(binaries[0]["seconds"], 7200.12)

    def test_summary_line_without_time_is_ignored(self):
        cases = ci_parity_runner.parse_gtest_output(
            "[  FAILED  ] 2 tests, listed below:"
        )
        self.assertEqual(cases, [])

    def test_ctest_result_line_matches_baseline_shape(self):
        line = ci_parity_runner.ctest_result_line(
            1, 5, "test_hiprand_api", True, 1.12
        )
        self.assertIn("1/5 Test #1: test_hiprand_api", line)
        self.assertIn("Passed", line)
        self.assertIn("1.12 sec", line)
        failed = ci_parity_runner.ctest_result_line(
            1, 1, "test_hiprand_api", False, 12.25
        )
        self.assertIn("***Failed", failed)

    def test_unsupported_marker(self):
        self.assertTrue(
            ci_parity_runner.looks_unsupported("fatal: UnimplementedInst V_ADD_CO_U32")
        )
        self.assertFalse(ci_parity_runner.looks_unsupported("[       OK ] Case (1 ms)"))


INSTALLED_CTEST = textwrap.dedent(
    """\
    add_test(test_hiprand_api "../test_hiprand_api")
    add_test(test_hiprand_cpp_wrapper "../test_hiprand_cpp_wrapper")
    add_test(test_hiprand_kernel "../test_hiprand_kernel")
    add_test(test_hiprand_linkage "../test_hiprand_linkage")
    add_test(test_hiprand_c_compile "../test_hiprand_c_compile")
    add_test(ffm_only "../ffm_only")
    set_tests_properties("test_hiprand_api" PROPERTIES LABELS "quick;standard;comprehensive;full")
    set_tests_properties("test_hiprand_cpp_wrapper" PROPERTIES LABELS "quick;standard;comprehensive;full")
    set_tests_properties("test_hiprand_kernel" PROPERTIES LABELS "quick;standard;comprehensive;full;ffm-quick;ffm-full")
    set_tests_properties("test_hiprand_linkage" PROPERTIES LABELS "quick;standard;comprehensive;full;ffm-quick;ffm-full")
    set_tests_properties("test_hiprand_c_compile" PROPERTIES LABELS "quick;standard;comprehensive;full")
    set_tests_properties("ffm_only" PROPERTIES LABELS "ffm-quick")
    """
)


class QuickLabelTest(unittest.TestCase):
    def test_quick_selects_the_five_baseline_binaries(self):
        selected = ci_parity_runner.select_labeled_tests(
            ci_parity_runner.parse_installed_tests(INSTALLED_CTEST),
            "^quick$",
        )
        self.assertEqual(
            [test["name"] for test in selected],
            [
                "test_hiprand_api",
                "test_hiprand_cpp_wrapper",
                "test_hiprand_kernel",
                "test_hiprand_linkage",
                "test_hiprand_c_compile",
            ],
        )

    def test_ctest_verbose_log_for_quick_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hiprand = root / "rocm" / "bin" / "hipRAND"
            hiprand.mkdir(parents=True)
            binary = root / "rocm" / "bin" / "test_hiprand_api"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
            (hiprand / "CTestTestfile.cmake").write_text(
                "\n".join(
                    [
                        'add_test(test_hiprand_api "../test_hiprand_api")',
                        'set_tests_properties("test_hiprand_api" PROPERTIES LABELS "quick;standard")',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            rocjitsu = root / "rocjitsu"
            rocjitsu.write_text(
                "#!/bin/sh\n"
                "if [ -n \"${GTEST_FILTER+x}\" ]; then\n"
                "  echo \"GTEST_FILTER is set: [$GTEST_FILTER]\"\n"
                "  exit 1\n"
                "fi\n"
                "echo '[ RUN      ] hiprand_32/hiprand_api_32.hiprand_generate_test/0'\n"
                "echo '[       OK ] hiprand_32/hiprand_api_32.hiprand_generate_test/0 (12 ms)'\n",
                encoding="utf-8",
            )
            rocjitsu.chmod(rocjitsu.stat().st_mode | stat.S_IEXEC)
            (root / "cfg.json").write_text("{}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                # The workflow passes --out parity/emu.json. ctest then
                # changes into that relative test dir, so the launcher path
                # written into CTestTestfile.cmake must already be absolute.
                result = ci_parity_runner.run_labeled_ctest(
                    "rocjitsu",
                    "cfg.json",
                    "rocm",
                    "^quick$",
                    "",
                    30,
                    {**os.environ, "GTEST_FILTER": ""},
                    False,
                    Path("parity"),
                )
            finally:
                os.chdir(previous)
            log = "\n".join(result["log_lines"])
            launcher = root / "parity" / "ctest-quick" / "rocjitsu-launch"
            self.assertIn(str(launcher), log)
            self.assertNotIn("Could not find executable", log)
            self.assertIn("Start 1: test_hiprand_api", log)
            self.assertIn(
                "1: [ RUN      ] hiprand_32/hiprand_api_32.hiprand_generate_test/0",
                log,
            )
            self.assertIn("1/1 Test #1: test_hiprand_api", log)
            self.assertIn("Passed", log)
            self.assertEqual(result["cases"][0]["status"], "passed")
            self.assertEqual(result["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
