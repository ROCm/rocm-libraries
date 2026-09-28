# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
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

    def test_summary_line_without_time_is_ignored(self):
        cases = ci_parity_runner.parse_gtest_output(
            "[  FAILED  ] 2 tests, listed below:"
        )
        self.assertEqual(cases, [])

    def test_unsupported_marker(self):
        self.assertTrue(
            ci_parity_runner.looks_unsupported("fatal: UnimplementedInst V_ADD_CO_U32")
        )
        self.assertFalse(ci_parity_runner.looks_unsupported("[       OK ] Case (1 ms)"))


if __name__ == "__main__":
    unittest.main()
