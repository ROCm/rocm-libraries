# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import hostnumerics as hv


class SymmetricComparisonOracleTests(unittest.TestCase):
    """Retain the scalar classifications used by strict componentwise validation."""

    def check_cases(self, scalar_type, tolerance, cases):
        precision = (
            np.float64
            if scalar_type in (hv.ScalarType.Float64, hv.ScalarType.ComplexFloat64)
            else np.float32
        )
        for name, a, b, passed in cases:
            observed = hv.from_numpy(np.asarray([a]), scalar_type)
            expected = hv.from_numpy(np.asarray([b]), scalar_type)
            decoded_dtype = (
                np.complex128
                if scalar_type
                in (hv.ScalarType.ComplexFloat32, hv.ScalarType.ComplexFloat64)
                else np.float64
            )
            decoded_a = hv.to_numpy(observed, decoded_dtype)[0]
            decoded_b = hv.to_numpy(expected, decoded_dtype)[0]

            def component_close(x, y):
                x, y, tol = precision(x), precision(y), precision(tolerance)
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = precision(abs(precision(x - y)))
                    scale = precision(precision(abs(x) + abs(y)) + precision(1))
                    return bool(x == y or difference < precision(tol * scale))

            oracle = component_close(
                decoded_a.real, decoded_b.real
            ) and component_close(decoded_a.imag, decoded_b.imag)
            with self.subTest(type=scalar_type, case=name):
                self.assertEqual(oracle, passed)
            for swapped in (False, True):
                for detailed in (False, True):
                    options = hv.ComparisonOptions()
                    options.relative_tolerance = tolerance
                    options.elementwise_tolerance_model = (
                        hv.ElementwiseToleranceModel.SymmetricUnitScaled
                    )
                    options.arithmetic_precision = (
                        hv.ComparisonArithmeticPrecision.Float64
                        if precision == np.float64
                        else hv.ComparisonArithmeticPrecision.Float32
                    )
                    options.strict_elementwise_tolerance = True
                    options.compute_elementwise_statistics = detailed
                    options.compute_frobenius = detailed
                    report = hv.compare(
                        expected if swapped else observed,
                        observed if swapped else expected,
                        options,
                    )
                    with self.subTest(
                        type=scalar_type, case=name, swapped=swapped, detailed=detailed
                    ):
                        self.assertEqual(report.passed, passed)
                        self.assertEqual(report.mismatches, int(not passed))

    def test_real_classifications(self):
        for scalar_type, tolerance, within, outside, small, too_large in (
            (hv.ScalarType.Float32, 0.0002, 1.0002, 1.001, 0.00005, 0.0005),
            (hv.ScalarType.Float64, 1e-12, 1.0 + 2e-12, 1.0 + 1e-9, 5e-13, 5e-9),
            (hv.ScalarType.Float16, 0.01, 1.02, 1.1, 0.005, 0.05),
            (hv.ScalarType.BFloat16, 0.1, 2.25, 2.0, 0.05, 0.5),
            (hv.ScalarType.Float8E4M3, 0.125, 2.5, 2.0, None, None),
            (hv.ScalarType.Float8E4M3Fnuz, 0.125, 2.5, 2.0, None, None),
            (hv.ScalarType.Float8E5M2, 0.25, 1.5, 3.0, None, None),
            (hv.ScalarType.Float8E5M2Fnuz, 0.25, 1.5, 3.0, None, None),
        ):
            base = (
                2.0
                if scalar_type
                in (
                    hv.ScalarType.BFloat16,
                    hv.ScalarType.Float8E4M3,
                    hv.ScalarType.Float8E4M3Fnuz,
                )
                else 1.0
            )
            cases = [
                ("exact positive", 1.0, 1.0, True),
                ("exact negative", -1.0, -1.0, True),
                ("exact zero", 0.0, 0.0, True),
                ("signed zero", -0.0, 0.0, True),
                ("both NaN", np.nan, np.nan, False),
                ("one NaN", np.nan, 1.0, False),
                ("within", base, within, True),
                ("outside", 1.0, outside, False),
            ]
            if small is not None:
                cases.extend(
                    [
                        ("zero versus small", 0.0, small, True),
                        ("zero versus outside", 0.0, too_large, False),
                        ("negative within", -base, -within, True),
                        ("negative outside", -1.0, -outside, False),
                    ]
                )
            if scalar_type in (
                hv.ScalarType.Float32,
                hv.ScalarType.Float64,
                hv.ScalarType.Float16,
                hv.ScalarType.BFloat16,
                hv.ScalarType.Float8E5M2,
            ):
                cases.extend(
                    [
                        ("positive infinity", np.inf, np.inf, True),
                        ("negative infinity", -np.inf, -np.inf, True),
                        ("opposite infinities", np.inf, -np.inf, False),
                        ("infinity versus finite", np.inf, 1.0, False),
                    ]
                )
            if scalar_type in (hv.ScalarType.Float32, hv.ScalarType.Float64):
                dtype = (
                    np.float32 if scalar_type == hv.ScalarType.Float32 else np.float64
                )
                v, tol = dtype(10), dtype(tolerance)
                delta = dtype(
                    dtype(tol * dtype(dtype(2) * v + dtype(1))) / dtype(dtype(1) - tol)
                )
                for factor, passed in ((0.99, True), (1.01, False)):
                    cases.append(
                        (
                            "boundary " + str(factor),
                            v,
                            dtype(v + dtype(delta * dtype(factor))),
                            passed,
                        )
                    )
                magnitude, close_delta, far_delta, opposite = (
                    (1e6, 100.0, 500.0, 0.0002)
                    if dtype == np.float32
                    else (1e12, 1.0, 5.0, 1e-6)
                )
                cases.extend(
                    [
                        ("large within", magnitude, magnitude + close_delta, True),
                        ("large outside", magnitude, magnitude + far_delta, False),
                        ("opposite signs near zero", -opposite, opposite, False),
                    ]
                )
            elif scalar_type in (hv.ScalarType.Float16, hv.ScalarType.BFloat16):
                tol = np.float32(tolerance)
                delta = np.float32(tol * np.float32(9) / np.float32(1 - tol))
                factors = (
                    (0.9, 1.1) if scalar_type == hv.ScalarType.Float16 else (0.85, 1.15)
                )
                for factor, passed in zip(factors, (True, False)):
                    cases.append(
                        (
                            "boundary " + str(factor),
                            4.0,
                            np.float32(4 + np.float32(delta * np.float32(factor))),
                            passed,
                        )
                    )
            elif scalar_type in (
                hv.ScalarType.Float8E4M3,
                hv.ScalarType.Float8E4M3Fnuz,
            ):
                cases.extend(
                    [
                        ("boundary below", 4.0, 5.0, True),
                        ("boundary above", 2.0, 4.0, False),
                    ]
                )
            else:
                cases.extend(
                    [
                        ("boundary below", 2.0, 3.0, True),
                        ("boundary above", 1.0, 4.0, False),
                    ]
                )
            self.check_cases(scalar_type, tolerance, cases)

    def test_complex_classifications(self):
        for scalar_type, tolerance, delta, outside in (
            (hv.ScalarType.ComplexFloat32, 0.0002, 0.0002, 0.001),
            (hv.ScalarType.ComplexFloat64, 1e-12, 2e-12, 1e-9),
        ):
            self.check_cases(
                scalar_type,
                tolerance,
                [
                    ("exact", 1 + 2j, 1 + 2j, True),
                    ("zero", 0j, 0j, True),
                    ("real NaN", complex(np.nan, 0), 1 + 0j, False),
                    ("imaginary NaN", complex(0, np.nan), 1j, False),
                    (
                        "both NaN",
                        complex(np.nan, np.nan),
                        complex(np.nan, np.nan),
                        False,
                    ),
                    ("real infinity", complex(np.inf, 0), complex(np.inf, 0), True),
                    (
                        "imaginary infinity",
                        complex(0, np.inf),
                        complex(0, np.inf),
                        True,
                    ),
                    ("real outside", 1 + 0j, 1 + outside + 0j, False),
                    ("imaginary outside", 1j, complex(0, 1 + outside), False),
                    ("both within", 1 + 1j, complex(1 + delta, 1 + delta), True),
                    ("both outside", 1 + 1j, complex(1 + outside, 1 + outside), False),
                    (
                        "real within imaginary outside",
                        1 + 1j,
                        complex(1 + delta, 1 + outside),
                        False,
                    ),
                    (
                        "real outside imaginary within",
                        1 + 1j,
                        complex(1 + outside, 1 + delta),
                        False,
                    ),
                ],
            )

    def test_exact_integer_classifications(self):
        for dtype in (np.int8, np.int32, np.uint32):
            limits = np.iinfo(dtype)
            cases = [
                (42, 42, True),
                (0, 0, True),
                (1, 2, False),
                (0, 1, False),
                (limits.max, limits.max, True),
                (limits.min, limits.min, True),
                (limits.min, limits.max, False),
            ]
            cases.extend(
                [(10, 11, False)] if dtype == np.int8 else [(1000, 1001, False)]
            )
            if dtype != np.uint32:
                cases.extend([(-1, -1, True), (-1, 1, False)])
            if dtype == np.int32:
                cases.append((-100, -100, True))
            if dtype == np.uint32:
                cases.append((100, 200, False))
            for a, b, passed in cases:
                for detailed in (False, True):
                    options = hv.ComparisonOptions()
                    options.compute_elementwise_statistics = detailed
                    options.compute_frobenius = detailed
                    report = hv.compare(
                        hv.from_numpy(np.asarray([a], dtype=dtype)),
                        hv.from_numpy(np.asarray([b], dtype=dtype)),
                        options,
                    )
                    with self.subTest(dtype=dtype, a=a, b=b, detailed=detailed):
                        self.assertEqual(report.passed, passed)


if __name__ == "__main__":
    unittest.main()
