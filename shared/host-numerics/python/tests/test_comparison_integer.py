# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class ComparisonIntegerTests(unittest.TestCase):  # noqa: F405
    def test_integer_comparison_is_exact_beyond_float64_precision(self):
        observed_unsigned = np.asarray(
            [2**53, 2**63, np.iinfo(np.uint64).max],
            dtype=np.uint64,
        )
        expected_unsigned = np.asarray(
            [2**53 + 1, 2**63, np.iinfo(np.uint64).max - 1],
            dtype=np.uint64,
        )
        unsigned_report = hv.compare(
            hv.from_numpy(observed_unsigned),
            hv.from_numpy(expected_unsigned),
        )
        self.assertEqual(
            unsigned_report.mismatches,
            int(np.count_nonzero(observed_unsigned != expected_unsigned)),
        )
        self.assertFalse(unsigned_report.passed)

        observed_signed = np.asarray(
            [np.iinfo(np.int64).min, 2**53, np.iinfo(np.int64).max],
            dtype=np.int64,
        )
        expected_signed = np.asarray(
            [np.iinfo(np.int64).max, 2**53 + 1, np.iinfo(np.int64).max],
            dtype=np.int64,
        )
        signed_report = hv.compare(
            hv.from_numpy(observed_signed),
            hv.from_numpy(expected_signed),
        )
        self.assertEqual(
            signed_report.mismatches,
            int(np.count_nonzero(observed_signed != expected_signed)),
        )
        self.assertFalse(signed_report.passed)
