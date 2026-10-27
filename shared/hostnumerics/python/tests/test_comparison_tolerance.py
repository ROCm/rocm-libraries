# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _hostnumerics_test_support import *  # noqa: F403


class ComparisonToleranceTests(unittest.TestCase):  # noqa: F405
    def test_allclose_requires_explicit_tolerances_and_nan_policy(self):
        with self.assertRaises(TypeError):
            hv.allclose_comparison_options()
        with self.assertRaises(TypeError):
            hv.allclose_comparison_options(1e-8, 1e-5)

    def test_float32_difference_uses_comparison_precision(self):
        cases = [(1.0, -(2**-25), 1.0), (1.0, -(2**-25), 1.0 + 2**-25)]
        maximum = float(np.finfo(np.float32).max)
        cases.append((maximum, -maximum, 2.0 * maximum))
        for observed, expected, tolerance in cases:
            difference = abs(observed - expected)
            for detailed in (False, True):
                with self.subTest(values=(observed, expected), detailed=detailed):
                    options = hv.allclose_comparison_options(tolerance, 0.0, False)
                    options.compute_elementwise_statistics = detailed
                    options.compute_frobenius = detailed
                    options.max_reported_mismatches = 0
                    report = hv.compare(
                        hv.from_numpy(np.asarray([observed], dtype=np.float32)),
                        hv.from_numpy(np.asarray([expected], dtype=np.float32)),
                        options,
                    )
                    self.assertEqual(report.passed, difference <= tolerance)
                    self.assertEqual(report.mismatches, int(difference > tolerance))
                    if detailed:
                        self.assertEqual(report.max_absolute_difference, difference)

    def test_symmetric_unit_tolerance_matches_strict_scalar_oracle(self):
        cases = [
            (1.3125, 1.0, 0.1),
            (1.0, 1.3125, 0.1),
            (0.71875, 1.0, 0.1),
            (0.0, 1.0, 0.5),
            (1.0, 1.0002, 0.0002),
            (1.0, 1.001, 0.0002),
        ]
        for dtype, precision in (
            (np.float32, hv.ComparisonArithmeticPrecision.Float32),
            (np.float64, hv.ComparisonArithmeticPrecision.Float64),
        ):
            for observed, expected, tolerance in cases:
                a, b, tol = dtype(observed), dtype(expected), dtype(tolerance)
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = dtype(abs(dtype(a - b)))
                    scale = dtype(dtype(abs(a) + abs(b)) + dtype(1))
                    legacy = bool(a == b or difference < dtype(tol * scale))
                for detailed in (False, True):
                    with self.subTest(dtype=dtype, values=(a, b), detailed=detailed):
                        options = hv.ComparisonOptions()
                        options.relative_tolerance = tolerance
                        options.elementwise_tolerance_model = (
                            hv.ElementwiseToleranceModel.SymmetricUnitScaled
                        )
                        options.arithmetic_precision = precision
                        options.strict_elementwise_tolerance = True
                        options.compute_elementwise_statistics = detailed
                        options.compute_frobenius = detailed
                        options.max_reported_mismatches = 0
                        report = hv.compare(
                            hv.from_numpy(np.asarray([a], dtype=dtype)),
                            hv.from_numpy(np.asarray([b], dtype=dtype)),
                            options,
                        )
                        self.assertEqual(report.passed, legacy)

        complex_observed = hv.from_numpy(np.asarray([0.09 + 0.09j], dtype=np.complex64))
        complex_expected = hv.from_numpy(np.asarray([0.0 + 0.0j], dtype=np.complex64))
        options = hv.ComparisonOptions()
        options.relative_tolerance = 0.1
        options.elementwise_tolerance_model = (
            hv.ElementwiseToleranceModel.SymmetricUnitScaled
        )
        options.arithmetic_precision = hv.ComparisonArithmeticPrecision.Float32
        options.strict_elementwise_tolerance = True
        self.assertTrue(hv.compare(complex_observed, complex_expected, options).passed)
        options.complex_comparison_mode = hv.ComplexComparisonMode.Magnitude
        self.assertFalse(hv.compare(complex_observed, complex_expected, options).passed)

    def test_explicit_relative_norm_policy(self):
        options = hv.ComparisonOptions()
        options.all_close = False
        options.compute_frobenius = True
        options.relative_frobenius_tolerance = 1.0
        options.zero_expected_norm_is_nan = True
        options.non_finite_values_invalidate_relative_norms = True

        boundary = hv.compare(
            hv.from_numpy(np.asarray([2.0], dtype=np.float64)),
            hv.from_numpy(np.asarray([1.0], dtype=np.float64)),
            options,
        )
        self.assertEqual(boundary.relative_frobenius_error, 1.0)
        self.assertTrue(boundary.passed)

        zero = hv.from_numpy(np.asarray([0.0], dtype=np.float64))
        zero_result = hv.compare(zero, zero, options)
        self.assertTrue(np.isnan(zero_result.relative_frobenius_error))
        self.assertTrue(np.isnan(zero_result.relative_maximum_error))
        self.assertFalse(zero_result.passed)

    def test_complex_allclose_modes_match_numpy_magnitude_policy(self):
        observed_values = np.asarray([1.0 + 1.0j], dtype=np.complex128)
        expected_values = np.asarray([0.0 + 0.0j], dtype=np.complex128)
        observed = hv.from_numpy(observed_values)
        expected = hv.from_numpy(expected_values)

        magnitude = hv.allclose_comparison_options(1.0, 0.0, equal_nans=False)
        self.assertEqual(
            magnitude.complex_comparison_mode,
            hv.ComplexComparisonMode.Magnitude,
        )
        self.assertFalse(hv.compare(observed, expected, magnitude).passed)
        self.assertEqual(
            hv.compare(observed, expected, magnitude).passed,
            bool(np.allclose(observed_values, expected_values, atol=1.0, rtol=0.0)),
        )

        componentwise = hv.allclose_comparison_options(1.0, 0.0, equal_nans=False)
        componentwise.complex_comparison_mode = hv.ComplexComparisonMode.Componentwise
        self.assertTrue(hv.compare(observed, expected, componentwise).passed)

        magnitude.compute_elementwise_statistics = False
        magnitude.compute_frobenius = False
        self.assertFalse(hv.compare(observed, expected, magnitude).passed)

        boundary_observed_values = np.asarray([0.0 + 0.0j], dtype=np.complex128)
        boundary_expected_values = np.asarray([3.0 + 4.0j], dtype=np.complex128)
        boundary = hv.allclose_comparison_options(0.0, 1.0, equal_nans=False)
        boundary_result = hv.compare(
            hv.from_numpy(boundary_observed_values),
            hv.from_numpy(boundary_expected_values),
            boundary,
        )
        reverse_result = hv.compare(
            hv.from_numpy(boundary_expected_values),
            hv.from_numpy(boundary_observed_values),
            boundary,
        )
        self.assertEqual(
            boundary_result.passed,
            bool(
                np.allclose(
                    boundary_observed_values,
                    boundary_expected_values,
                    atol=0.0,
                    rtol=1.0,
                )
            ),
        )
        self.assertEqual(
            reverse_result.passed,
            bool(
                np.allclose(
                    boundary_expected_values,
                    boundary_observed_values,
                    atol=0.0,
                    rtol=1.0,
                )
            ),
        )

        nan_observed = hv.from_numpy(
            np.asarray([complex(np.nan, 1.0)], dtype=np.complex128)
        )
        nan_expected = hv.from_numpy(
            np.asarray([complex(1.0, np.nan)], dtype=np.complex128)
        )
        equal_nan = hv.allclose_comparison_options(0.0, 0.0, True)
        nan_result = hv.compare(nan_observed, nan_expected, equal_nan)
        self.assertTrue(nan_result.passed)
        self.assertEqual(nan_result.matched_nans, 1)

        search_observed = hv.from_numpy(np.asarray([0.09 + 0.09j], dtype=np.complex128))
        search_expected = hv.from_numpy(np.asarray([0.0 + 0.0j], dtype=np.complex128))
        self.assertIsNone(
            hv.find_allclose_tolerance(
                search_observed,
                search_expected,
                [0.1],
                [0.0],
                options=hv.allclose_comparison_options(1e-8, 1e-5, False),
            )
        )
        componentwise_search = hv.allclose_comparison_options(
            absolute_tolerance=1e-8, relative_tolerance=1e-5, equal_nans=False
        )
        componentwise_search.complex_comparison_mode = (
            hv.ComplexComparisonMode.Componentwise
        )
        self.assertIsNotNone(
            hv.find_allclose_tolerance(
                search_observed,
                search_expected,
                [0.1],
                [0.0],
                componentwise_search,
            )
        )

    def test_allclose_policies_match_numpy(self):
        real_values = np.asarray(
            [
                -5.0,
                -4.0,
                -3.0,
                -2.25,
                -2.0,
                -1.1,
                -1.02,
                -1.0,
                -0.05,
                -0.005,
                -0.0,
                0.0,
                0.005,
                0.05,
                1.0,
                1.02,
                1.1,
                1.5,
                2.0,
                2.25,
                2.5,
                3.0,
                4.0,
                5.0,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )

        real_types = [
            hv.ScalarType.Float16,
            hv.ScalarType.BFloat16,
            hv.ScalarType.Float8E4M3,
            hv.ScalarType.Float8E5M2,
            hv.ScalarType.Float8E4M3Fnuz,
            hv.ScalarType.Float8E5M2Fnuz,
            hv.ScalarType.Float32,
            hv.ScalarType.Float64,
        ]
        for scalar_type in real_types:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.from_numpy(
                    np.tile(real_values, real_values.size),
                    scalar_type,
                )
                expected = hv.from_numpy(
                    np.repeat(real_values, real_values.size),
                    scalar_type,
                )
                observed_values = hv.to_numpy(observed, np.float64)
                expected_values = hv.to_numpy(expected, np.float64)
                options = hv.allclose_comparison_options(
                    absolute_tolerance=1e-8, relative_tolerance=1e-5, equal_nans=False
                )
                options.compute_frobenius = False

                with np.errstate(invalid="ignore"):
                    oracle = np.isclose(
                        observed_values,
                        expected_values,
                        atol=options.absolute_tolerance,
                        rtol=options.relative_tolerance,
                        equal_nan=options.equal_nans,
                    )

                report = hv.compare(observed, expected, options)
                self.assertEqual(
                    report.mismatches,
                    int(np.count_nonzero(~oracle)),
                )
                self.assertEqual(report.passed, bool(np.all(oracle)))

        complex_values = np.asarray(
            [
                0.0 + 0.0j,
                -0.0 + 0.0j,
                1.0 + 1.0j,
                1.0002 + 1.0002j,
                1.001 + 1.0j,
                1.0 + 1.001j,
                complex(np.inf, 0.0),
                complex(0.0, np.inf),
                complex(np.nan, 0.0),
                complex(0.0, np.nan),
            ],
            dtype=np.complex128,
        )
        for scalar_type in [
            hv.ScalarType.ComplexFloat32,
            hv.ScalarType.ComplexFloat64,
        ]:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.from_numpy(
                    np.tile(complex_values, complex_values.size),
                    scalar_type,
                )
                expected = hv.from_numpy(
                    np.repeat(complex_values, complex_values.size),
                    scalar_type,
                )
                observed_values = hv.to_numpy(observed, np.complex128)
                expected_values = hv.to_numpy(expected, np.complex128)
                options = hv.allclose_comparison_options(
                    absolute_tolerance=1e-8, relative_tolerance=1e-5, equal_nans=False
                )
                options.compute_frobenius = False

                with np.errstate(invalid="ignore"):
                    oracle = np.isclose(
                        observed_values,
                        expected_values,
                        atol=options.absolute_tolerance,
                        rtol=options.relative_tolerance,
                        equal_nan=options.equal_nans,
                    )
                report = hv.compare(observed, expected, options)
                self.assertEqual(
                    report.mismatches,
                    int(np.count_nonzero(~oracle)),
                )

        for dtype, scalar_type in [
            (np.int8, hv.ScalarType.Int8),
            (np.int32, hv.ScalarType.Int32),
            (np.uint32, hv.ScalarType.UInt32),
        ]:
            observed_values = np.asarray([0, 1, 2, 3], dtype=dtype)
            expected_values = np.asarray([0, 1, 7, 3], dtype=dtype)
            report = hv.compare(
                hv.from_numpy(observed_values, scalar_type),
                hv.from_numpy(expected_values, scalar_type),
                hv.ComparisonOptions(),
            )
            self.assertEqual(
                report.mismatches,
                int(np.count_nonzero(observed_values != expected_values)),
            )

    def test_explicit_tolerance_uses_numpy_boundary(self):
        observed = hv.from_numpy(np.asarray([1.02, 0.0], dtype=np.float32))
        expected = hv.from_numpy(np.asarray([1.0, 1.0], dtype=np.float32))

        defaults = hv.allclose_comparison_options(
            absolute_tolerance=1e-8, relative_tolerance=1e-5, equal_nans=False
        )
        defaults.compute_frobenius = False
        self.assertEqual(hv.compare(observed, expected, defaults).mismatches, 2)

        overridden = hv.allclose_comparison_options(0.01, 0.02, equal_nans=False)
        overridden.compute_frobenius = False
        self.assertEqual(
            hv.compare(observed, expected, overridden).mismatches,
            1,
        )

        exact_boundary = hv.allclose_comparison_options(0.0, 1.0, equal_nans=False)
        exact_boundary.compute_frobenius = False
        boundary_observed = hv.from_numpy(np.asarray([0.0], dtype=np.float32))
        boundary_expected = hv.from_numpy(np.asarray([1.0], dtype=np.float32))
        self.assertTrue(
            hv.compare(boundary_observed, boundary_expected, exact_boundary).passed
        )

    def test_relative_policy_scales_at_large_magnitudes(self):
        cases = [
            (
                np.float32,
                hv.ScalarType.Float32,
                np.float32(1.0e6),
                np.float32(100.0),
                np.float32(500.0),
            ),
            (
                np.float64,
                hv.ScalarType.Float64,
                np.float64(1.0e12),
                np.float64(1.0),
                np.float64(5.0),
            ),
        ]
        for dtype, scalar_type, base, accepted_delta, rejected_delta in cases:
            with self.subTest(scalar_type=scalar_type):
                tolerance = 0.0002 if scalar_type == hv.ScalarType.Float32 else 1e-12
                options = hv.allclose_comparison_options(
                    tolerance, 2.0 * tolerance, equal_nans=False
                )
                options.compute_frobenius = False
                accepted = hv.compare(
                    hv.from_numpy(np.asarray([base], dtype=dtype)),
                    hv.from_numpy(np.asarray([base + accepted_delta], dtype=dtype)),
                    options,
                )
                rejected = hv.compare(
                    hv.from_numpy(np.asarray([base], dtype=dtype)),
                    hv.from_numpy(np.asarray([base + rejected_delta], dtype=dtype)),
                    options,
                )
                self.assertTrue(accepted.passed)
                self.assertFalse(rejected.passed)

    def test_float32_policy_rejects_opposite_signs_near_zero(self):
        observed = hv.from_numpy(np.asarray([-0.0001, -0.0002], dtype=np.float32))
        expected = hv.from_numpy(np.asarray([0.0001, 0.0002], dtype=np.float32))
        options = hv.allclose_comparison_options(0.0002, 0.0004, equal_nans=False)
        options.compute_frobenius = False
        report = hv.compare(observed, expected, options)
        self.assertFalse(report.passed)
        self.assertEqual(report.mismatches, 1)

    def test_signed_zero_matches_numpy(self):
        observed = hv.from_numpy(np.asarray([0.0, -0.0], dtype=np.float64))
        expected = hv.from_numpy(np.asarray([-0.0, 0.0], dtype=np.float64))
        options = hv.ComparisonOptions()
        options.compute_frobenius = False
        self.assertTrue(hv.compare(observed, expected, options).passed)
