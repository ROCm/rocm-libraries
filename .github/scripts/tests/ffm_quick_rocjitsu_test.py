# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import ffm_quick_rocjitsu as sweep  # noqa: E402


SAMPLE = textwrap.dedent(
    """
    add_test(test_hiprand_kernel "../test_hiprand_kernel")
    set_tests_properties("test_hiprand_kernel" PROPERTIES LABELS "quick;ffm-quick;ffm-full")
    add_test(rocblas-test_ffm-quick_suite python3 "../rocblas_rtest.py" -t ctest_ffm-quick)
    set_tests_properties(rocblas-test_ffm-quick_suite PROPERTIES LABELS "ffm-quick" TIMEOUT 7200)
    add_test(test_hip_api_gpus_ffm-quick_suite [[../test_hip_api]] --gtest_filter=Smoke*)
    set_tests_properties(test_hip_api_gpus_ffm-quick_suite PROPERTIES LABELS "ffm-quick;ffm-full" TIMEOUT 7200)
    add_test("bilinear ffm-quick" "../bilinear" -y configs/quick.yaml)
    set_tests_properties("bilinear ffm-quick" PROPERTIES LABELS "ffm-quick" SKIP_REGULAR_EXPRESSION "unsupported host device")
    add_test(excluded "../excluded")
    set_tests_properties("excluded" PROPERTIES LABELS "ffm-quick;ex_gpu_gfx110X")
    add_test(quick_only "../quick_only")
    set_tests_properties("quick_only" PROPERTIES LABELS "quick")
    """
)


class ParseInstalledCTestTest(unittest.TestCase):
    def test_selects_ffm_quick_commands(self):
        tests = sweep.parse_ctest_file(SAMPLE)
        selected = [test for test in tests.values() if sweep.selects_ffm_quick(test)]
        names = [test["name"] for test in selected]
        self.assertEqual(
            names,
            [
                "test_hiprand_kernel",
                "rocblas-test_ffm-quick_suite",
                "test_hip_api_gpus_ffm-quick_suite",
                "bilinear ffm-quick",
            ],
        )
        self.assertEqual(
            tests["rocblas-test_ffm-quick_suite"]["command"],
            ["python3", "../rocblas_rtest.py", "-t", "ctest_ffm-quick"],
        )
        self.assertEqual(
            tests["test_hip_api_gpus_ffm-quick_suite"]["command"][:2],
            ["../test_hip_api", "--gtest_filter=Smoke*"],
        )
        self.assertEqual(sweep.timeout_seconds(tests["test_hiprand_kernel"]), 1800)
        self.assertEqual(
            sweep.timeout_seconds(tests["rocblas-test_ffm-quick_suite"]), 3600
        )

    def test_skip_regex_matches_ctest_output(self):
        regex = sweep.parse_ctest_file(SAMPLE)["bilinear ffm-quick"]["skip_regex"]
        self.assertTrue(sweep.output_skips("unsupported host device\n", regex))
        self.assertFalse(sweep.output_skips("result mismatch\n", regex))

    def test_discover_orders_small_directories_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "bin" / "hipRAND"
            many = root / "bin" / "rocprim"
            one.mkdir(parents=True)
            many.mkdir()
            one.joinpath("CTestTestfile.cmake").write_text(
                'add_test(kernel "../kernel")\n'
                'set_tests_properties("kernel" PROPERTIES LABELS "ffm-quick")\n',
                encoding="utf-8",
            )
            many.joinpath("CTestTestfile.cmake").write_text(
                "\n".join(
                    f'add_test(t{i} "../t{i}")\n'
                    f'set_tests_properties("t{i}" PROPERTIES LABELS "ffm-quick")\n'
                    for i in range(3)
                ),
                encoding="utf-8",
            )
            found = sweep.discover(root)
            self.assertEqual([item["component"] for item in found], ["hipRAND", "rocprim"])


if __name__ == "__main__":
    unittest.main()
