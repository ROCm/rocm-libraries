# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class ComparisonCoreTests(unittest.TestCase):  # noqa: F405
    def test_generation_and_comparison(self):
        seed = 17
        recipe = real_generation_recipe(
            hv.GenerationRecipe.uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
            ),
            seed=seed,
        )
        first = hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe)
        second = hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe)
        self.assertTrue(hv.compare(first, second).passed)
        expected = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 7
                for index in range(6)
            ],
            dtype=np.float32,
        ).reshape((2, 3), order="F")
        np.testing.assert_array_equal(
            hv.to_numpy(first),
            expected,
        )

        changed_values = hv.to_numpy(second).copy()
        changed_values[1, 2] += 1
        changed = hv.from_numpy(changed_values)
        options = hv.ComparisonOptions()
        options.max_reported_mismatches = 2
        result = hv.compare(changed, first, options)
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)
        self.assertEqual(result.reported_mismatches[0].index, 5)
        diagnostic = str(result)
        self.assertTrue(
            diagnostic.startswith(
                "comparison failed: 1 of 6 compared elements mismatched"
            )
        )
        self.assertIn("index 5 [1, 2]", diagnostic)
        self.assertIn("absolute difference", diagnostic)

    def test_comparison_program_matches_numpy(self):
        expected_values = np.asarray(
            [[3.0 + 4.0j, 1.0 - 2.0j], [0.0 + 0.0j, -2.0 + 1.0j]],
            dtype=np.complex128,
        )
        observed_values = expected_values.copy()
        observed_values[0, 1] += 0.25 - 0.5j
        observed_values[1, 0] += 1.0

        expected = hv.from_numpy(expected_values)
        observed = hv.from_numpy(observed_values)
        options = hv.ComparisonOptions()
        options.absolute_tolerance = 0.1
        options.relative_frobenius_tolerance = 1.0
        report = hv.compare(observed, expected, options)

        difference = observed_values - expected_values
        self.assertEqual(report.compared, expected_values.size)
        self.assertEqual(report.mismatches, 2)
        self.assertAlmostEqual(
            report.frobenius_difference,
            float(np.linalg.norm(difference)),
        )
        self.assertAlmostEqual(
            report.frobenius_expected,
            float(np.linalg.norm(expected_values)),
        )
        self.assertAlmostEqual(
            report.relative_frobenius_error,
            float(np.linalg.norm(difference) / np.linalg.norm(expected_values)),
        )
        self.assertAlmostEqual(
            report.relative_maximum_error,
            float(np.max(np.abs(difference)) / np.max(np.abs(expected_values))),
        )
        self.assertEqual(
            report.reported_mismatches[0].coordinates,
            [0, 1],
        )

        selected = hv.ComparisonOptions()
        selected.selection = hv.OutputSelection.strided(
            0, 2, index_order=hv.IndexOrder.FirstDimensionFastest
        )
        selected_report = hv.compare(observed, expected, selected)
        expected_flat_fortran = expected_values.reshape(-1, order="F")
        observed_flat_fortran = observed_values.reshape(-1, order="F")
        selected_indices = np.arange(0, expected_values.size, 2)
        selected_mismatches = np.count_nonzero(
            observed_flat_fortran[selected_indices]
            != expected_flat_fortran[selected_indices]
        )
        self.assertEqual(selected_report.compared, len(selected_indices))
        self.assertEqual(selected_report.mismatches, int(selected_mismatches))

    def test_comparison_nonfinite_ulp_and_sentinel(self):
        expected_values = np.asarray([np.inf, np.nan, 1.0], dtype=np.float64)
        observed_values = expected_values.copy()
        options = hv.ComparisonOptions()
        options.equal_nans = True
        report = hv.compare(
            hv.from_numpy(observed_values),
            hv.from_numpy(expected_values),
            options,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.matched_infinities, 1)
        self.assertEqual(report.matched_nans, 1)

        one_ulp = np.nextafter(np.float64(1.0), np.float64(2.0))
        ulp_options = hv.ComparisonOptions()
        ulp_options.compute_ulp = True
        ulp_options.ulp_type = hv.ScalarType.Float64
        ulp_options.maximum_ulp_tolerance = 1.0
        ulp_report = hv.compare(
            hv.from_numpy(np.asarray([one_ulp])),
            hv.from_numpy(np.asarray([1.0], dtype=np.float64)),
            ulp_options,
        )
        self.assertEqual(ulp_report.maximum_ulp, 1.0)
        self.assertTrue(ulp_report.ulp_passed)
        self.assertEqual(
            hv.encoded_ulp_distance(
                0.0,
                float(np.nextafter(np.float32(0.0), np.float32(1.0))),
                hv.ScalarType.Float32,
            ),
            1.0,
        )
        self.assertEqual(
            hv.encoded_ulp_distance(1.0, 2.0, hv.ScalarType.E5M3),
            8.0,
        )

        candidates = [1e-6, 1e-5, 1e-4, 1e-3]
        tolerance = hv.find_allclose_tolerance(
            hv.from_numpy(np.asarray([1.00009], dtype=np.float64)),
            hv.from_numpy(np.asarray([1.0], dtype=np.float64)),
            candidates,
            [0.0],
        )
        self.assertIsNotNone(tolerance)
        self.assertEqual(tolerance.absolute, 1e-4)

        sentinel_values = np.full(5, np.inf, dtype=np.float32)
        sentinel = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            sentinel_values.tobytes(),
            strides=[1, 3],
        )
        self.assertTrue(
            hv.check_unused_tensor_storage(sentinel, allocated_elements=5).passed
        )
        sentinel_values[2] = 0.0
        sentinel = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            sentinel_values.tobytes(),
            strides=[1, 3],
        )
        sentinel_report = hv.check_unused_tensor_storage(sentinel, allocated_elements=5)
        self.assertFalse(sentinel_report.passed)
        self.assertEqual(sentinel_report.reported_mismatches[0].index, 2)
