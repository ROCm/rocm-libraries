# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


from _hostnumerics_test_support import *  # noqa: F403


class GenerationTests(unittest.TestCase):  # noqa: F405
    def test_indexed_generation_matches_numpy(self):
        serial = hv.generate_tensor(
            hv.ScalarType.Float32,
            [2, 3],
            real_generation_recipe_last_dimension_fastest(
                hv.GenerationRecipe.serial_index(), seed=0
            ),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(serial, dtype=None),
            np.arange(6, dtype=np.float32).reshape((2, 3), order="C"),
        )

        complex_recipe = hv.GenerationRecipe.cartesian_last_dimension_fastest(
            hv.GenerationRecipe.sine(),
            hv.GenerationRecipe.cosine(),
            seed=0,
        )
        complex_values = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32, [2, 3], complex_recipe
        )
        indices = np.arange(6, dtype=np.float32).reshape((2, 3), order="C")
        np.testing.assert_allclose(
            hv.to_numpy(complex_values, dtype=None),
            np.sin(indices) + 1j * np.cos(indices),
            rtol=1e-6,
            atol=1e-6,
        )

        identity = hv.generate_tensor(
            hv.ScalarType.Float32,
            [3, 4],
            real_generation_recipe_last_dimension_fastest(
                hv.GenerationRecipe.identity(), seed=0
            ),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(identity, dtype=None), np.eye(3, 4, dtype=np.float32)
        )

        seed = 19
        random_component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
        )
        random_recipe = hv.GenerationRecipe.cartesian_last_dimension_fastest(
            random_component,
            random_component,
            seed,
        )
        random_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            ),
            dtype=None,
        )
        random_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            ),
            dtype=None,
        )
        expected_real = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="C")
        expected_imaginary = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_IMAGINARY_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="C")
        np.testing.assert_array_equal(random_values, random_repeat)
        np.testing.assert_array_equal(random_values.real, expected_real)
        np.testing.assert_array_equal(random_values.imag, expected_imaginary)
        self.assertFalse(np.array_equal(expected_real, expected_imaginary))
        self.assertTrue(np.all((-3 <= expected_real) & (expected_real <= 3)))

        candidates = np.asarray([-6.0, -1.5, 0.0, 4.0], dtype=np.float32)
        seed = 37
        candidate_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.choice(
                hv.ChoiceGenerationParameters(values=candidates.tolist())
            ),
            seed=seed,
        )
        selected = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 4], candidate_recipe),
            dtype=None,
        )
        expected = np.asarray(
            [
                candidates[
                    counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 4
                ]
                for index in range(8)
            ],
            dtype=np.float32,
        ).reshape((2, 4), order="C")
        np.testing.assert_array_equal(selected, expected)
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.choice(hv.ChoiceGenerationParameters(values=[]))

        point = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3, 2]))
        first_point_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=9.0)),
            seed=0,
        )
        hv.generate_at(point, 3, first_point_recipe)
        expected_point = np.zeros((2, 3, 2), dtype=np.float32)
        expected_point[0, 1, 1] = 9.0
        np.testing.assert_array_equal(hv.to_numpy(point, dtype=None), expected_point)

        last_dimension_fastest = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=7.0)),
            index_order=[2, 1, 0],
            seed=0,
        )
        hv.generate_at(point, 3, last_dimension_fastest)
        expected_point[1, 1, 0] = 7.0
        np.testing.assert_array_equal(hv.to_numpy(point, dtype=None), expected_point)
        with self.assertRaises(RuntimeError):
            hv.generate_at(point, point.size, last_dimension_fastest)

        affine_component = hv.GenerationRecipe.affine_index_remainder(
            hv.AffineIndexRemainderGenerationParameters(
                dimension_coefficients=[1, -1, 2],
                offset=-2,
                positive_divisor=5,
            )
        ).with_affine_value_mapping(
            hv.GenerationAffineValueParameters(scale=1.0, offset=1.0)
        )
        affine = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3, 2],
                real_generation_recipe_last_dimension_fastest(affine_component, seed=0),
            ),
            dtype=None,
        )
        expected_affine = np.empty((2, 3, 2), dtype=np.float32)
        for index in np.ndindex(expected_affine.shape):
            numerator = -2 + index[0] - index[1] + 2 * index[2]
            expected_affine[index] = cxx_remainder(numerator, 5) + 1
        np.testing.assert_array_equal(affine, expected_affine)

    def test_type_derived_generation(self):
        maximum_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.type_maximum(), seed=0
        )
        maximum_cases = (
            (hv.ScalarType.Float16, np.finfo(np.float16).max),
            (
                hv.ScalarType.BFloat16,
                np.asarray([0x7F7F0000], dtype=np.uint32).view(np.float32)[0],
            ),
            (hv.ScalarType.Float4E2M1, 6.0),
            (hv.ScalarType.Float6E2M3, 7.5),
            (hv.ScalarType.Float6E3M2, 28.0),
            (hv.ScalarType.Float8E4M3, 448.0),
            (hv.ScalarType.Int8, 127),
            (hv.ScalarType.Int32, np.iinfo(np.int32).max),
        )
        for scalar_type, expected in maximum_cases:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.to_numpy(
                    hv.generate_tensor(scalar_type, [3], maximum_recipe),
                    np.float64 if scalar_type == hv.ScalarType.Float64 else np.float32,
                )
                np.testing.assert_array_equal(
                    observed, np.full(3, expected, dtype=observed.dtype)
                )

        fp4_denormal = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float4E2M1,
                [2],
                real_generation_recipe_last_dimension_fastest(
                    hv.GenerationRecipe.type_denormal_minimum(), seed=0
                ),
            ),
            dtype=None,
        )
        np.testing.assert_array_equal(
            fp4_denormal, np.asarray([0.5, 0.5], dtype=np.float32)
        )

        nan_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E4M3Fnuz,
                [2],
                real_generation_recipe_last_dimension_fastest(
                    hv.GenerationRecipe.type_nan(), seed=0
                ),
            ),
            dtype=None,
        )
        self.assertTrue(np.isnan(nan_values).all())

        infinity = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E5M2,
                [2],
                real_generation_recipe_last_dimension_fastest(
                    hv.GenerationRecipe.type_infinity(), seed=0
                ),
            ),
            dtype=None,
        )
        self.assertTrue(np.isposinf(infinity).all())

        type_range_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.uniform_type_range(), seed=23
        )
        low_precision = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float4E2M1,
                [64],
                type_range_recipe,
            ),
            dtype=None,
        )
        self.assertTrue(np.all((-6.0 <= low_precision) & (low_precision <= 6.0)))
        float64_range = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float64, [64], type_range_recipe),
            dtype=None,
        )
        self.assertTrue(np.isfinite(float64_range).all())
        self.assertTrue(
            np.all(
                (-np.finfo(np.float64).max <= float64_range)
                & (float64_range <= np.finfo(np.float64).max)
            )
        )

        absolute_integer_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.absolute_uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
            ),
            seed=23,
        )
        unsigned_scale_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.E5M3,
                [64],
                absolute_integer_recipe,
            ),
            dtype=None,
        )
        self.assertTrue(
            np.all((0 <= unsigned_scale_values) & (unsigned_scale_values <= 3))
        )

        encoded_exponent_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.random_encoded_exponent(
                hv.RandomEncodedExponentGenerationParameters(
                    lower_unbiased_exponent=-3,
                    upper_unbiased_exponent=-1,
                    source_type=hv.ScalarType.Float32,
                )
            ),
            seed=29,
        )
        narrow = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                encoded_exponent_recipe,
            ),
            dtype=None,
        )
        narrow_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                encoded_exponent_recipe,
            ),
            dtype=None,
        )
        np.testing.assert_array_equal(narrow, narrow_repeat)
        exponent_bits = (narrow.view(np.uint32) >> 23) & np.uint32(0xFF)
        self.assertTrue(
            set(int(value) for value in exponent_bits).issubset({124, 125, 126})
        )

        raw_serial = hv.generate_tensor(
            hv.ScalarType.Float16,
            [2, 3],
            real_generation_recipe_last_dimension_fastest(
                hv.GenerationRecipe.raw_serial_dimension(
                    hv.DimensionGenerationParameters(dimension=1)
                ),
                seed=0,
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_serial.storage, dtype=np.uint16).reshape(2, 3),
            np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint16),
        )

        raw_zero = hv.generate_tensor(
            hv.ScalarType.E8M0,
            [4],
            real_generation_recipe_last_dimension_fastest(
                hv.GenerationRecipe.raw_constant(
                    hv.RawConstantGenerationParameters(bits=0)
                ),
                seed=0,
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_zero.storage, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
        )

        raw_fp4 = hv.generate_tensor(
            hv.ScalarType.Float4E2M1,
            [65],
            real_generation_recipe_last_dimension_fastest(
                hv.GenerationRecipe.uniform_raw_integer(
                    hv.UniformIntegerGenerationParameters(lower=0, upper=14)
                ),
                seed=31,
            ),
        )
        fp4_nibbles = np.frombuffer(raw_fp4.storage, dtype=np.uint8)
        fp4_nibbles = np.concatenate(
            (fp4_nibbles & np.uint8(0xF), fp4_nibbles >> np.uint8(4))
        )[:65]
        self.assertTrue(np.all(fp4_nibbles <= 14))

        raw_bits_recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.random_raw_bits(), seed=41
        )
        raw_bits = hv.generate_tensor(hv.ScalarType.UInt32, [32], raw_bits_recipe)
        raw_bits_repeat = hv.generate_tensor(
            hv.ScalarType.UInt32, [32], raw_bits_recipe
        )
        self.assertEqual(raw_bits.storage, raw_bits_repeat.storage)
        self.assertNotEqual(raw_bits.storage, bytes(len(raw_bits.storage)))

    def test_generation_recipe_modifiers(self):
        scaled_component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=1, upper=10)
        ).with_affine_value_mapping(
            hv.GenerationAffineValueParameters(scale=0.1, offset=2.0)
        )
        scaled = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                real_generation_recipe_last_dimension_fastest(
                    scaled_component, seed=37
                ),
            ),
            dtype=None,
        )
        scaled_tenths = np.rint((scaled - 2.0) * 10).astype(np.int32)
        self.assertTrue(np.all((1 <= scaled_tenths) & (scaled_tenths <= 10)))
        np.testing.assert_allclose(
            scaled, 2.0 + scaled_tenths.astype(np.float32) / 10.0
        )

        positive_component = hv.GenerationRecipe.uniform_real(
            hv.UniformRealGenerationParameters(lower=-0.5, upper=0.5)
        ).with_absolute_transform()
        positive = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                real_generation_recipe_last_dimension_fastest(
                    positive_component, seed=37
                ),
            ),
            dtype=None,
        )
        self.assertTrue(np.all((0 <= positive) & (positive <= 0.5)))

        constant_component = hv.GenerationRecipe.constant(
            hv.ConstantGenerationParameters(value=2.0)
        )
        alternating_component = constant_component.with_alternating_sign(
            hv.AlternatingSignGenerationParameters(
                dimensions=[0, 1],
                negative_when_odd=False,
            )
        )
        alternating = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3, 2],
                real_generation_recipe_last_dimension_fastest(
                    alternating_component, seed=0
                ),
            ),
            dtype=None,
        )
        expected_matrix = np.asarray([[-2, 2, -2], [2, -2, 2]], dtype=np.float32)
        np.testing.assert_array_equal(alternating[:, :, 0], expected_matrix)
        np.testing.assert_array_equal(alternating[:, :, 1], expected_matrix)

        opposite_component = constant_component.with_alternating_sign(
            hv.AlternatingSignGenerationParameters(
                dimensions=[0, 1],
                negative_when_odd=True,
            )
        )
        opposite = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3],
                real_generation_recipe_last_dimension_fastest(
                    opposite_component, seed=0
                ),
            ),
            dtype=None,
        )
        np.testing.assert_array_equal(opposite, -expected_matrix)


class PermutationGenerationTests(unittest.TestCase):
    def test_permutations_are_independent_of_layout(self):
        import itertools

        dimensions = (2, 3, 4)
        for order in itertools.permutations(range(3)):
            for scalar_type in (hv.ScalarType.Float32, hv.ScalarType.Float6E2M3):
                layout = hv.Layout(hv.Shape(dimensions), [-20, 5, 1], 20)
                component = hv.GenerationRecipe.serial_index()
                recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
                    component, seed=0
                ).with_index_order(order)
                whole = hv.generate_tensor(scalar_type, layout, recipe)
                pointwise = hv.Tensor(scalar_type, layout)
                for linear in range(24):
                    hv.generate_at(pointwise, linear, recipe)
                self.assertEqual(whole.storage, pointwise.storage)
                if scalar_type == hv.ScalarType.Float32:
                    expected = np.arange(24, dtype=np.float32).reshape(
                        tuple(dimensions[i] for i in order)
                    )
                    expected = expected.transpose(
                        tuple(order.index(i) for i in range(3))
                    )
                    np.testing.assert_array_equal(
                        hv.to_numpy(whole, dtype=None), expected
                    )

    def test_arbitrary_selection_and_invalid_order(self):
        dimensions = (2, 3, 4)
        shape = hv.Shape(dimensions)
        order = [1, 0, 2]
        expected = hv.from_numpy(np.zeros(dimensions, dtype=np.float32))
        observed_values = np.zeros(dimensions, dtype=np.float32)
        point = shape.coordinates(13, order)
        observed_values[tuple(point)] = 9
        observed = hv.from_numpy(observed_values)
        options = hv.ComparisonOptions()
        report = hv.compare(observed, expected, options, {13}, order)
        self.assertFalse(report.passed)
        self.assertEqual(report.reported_mismatches[0].coordinates, point)
        for invalid in ([0, 1], [0, 0, 2], [0, 1, 3]):
            with self.assertRaises(RuntimeError):
                recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
                    hv.GenerationRecipe.serial_index(),
                    seed=0,
                ).with_index_order(invalid)
                hv.generate_tensor(hv.ScalarType.Float32, dimensions, recipe)
            with self.assertRaises(RuntimeError):
                hv.compare(
                    observed,
                    expected,
                    options,
                    hv.IndexRange.all(shape.element_count),
                    invalid,
                )

    def test_contiguous_axis_order_conversion_and_selections(self):
        dimensions = (2, 3, 4)
        order = [1, 0, 2]
        layout = hv.Layout(hv.Shape(dimensions), [4, 8, 1], 0)
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.serial_index(),
            seed=0,
        ).with_index_order(order)
        source = hv.generate_tensor(hv.ScalarType.Float32, layout, recipe)
        expected = np.arange(24, dtype=np.float32).reshape(3, 2, 4).transpose(1, 0, 2)
        np.testing.assert_array_equal(hv.to_numpy(source, dtype=None), expected)
        converted = source.to(hv.ScalarType.Float64)
        np.testing.assert_array_equal(hv.to_numpy(converted, dtype=None), expected)
        for selection in (
            hv.IndexRange.all(24),
            hv.IndexRange(2, 24, 3),
            {1, 13, 23},
        ):
            options = hv.ComparisonOptions()
            report = hv.compare(source, converted, options, selection, order)
            self.assertTrue(report.passed)
            expected_count = (
                selection.count()
                if isinstance(selection, hv.IndexRange)
                else len(selection)
            )
            self.assertEqual(report.compared, expected_count)

    def test_invalid_order_is_rejected_for_empty_or_identical_comparisons(self):
        for dimensions in ((2, 3, 4), (2, 0, 4), ()):
            tensor = hv.Tensor(hv.ScalarType.Float32, hv.Shape(dimensions))
            options = hv.ComparisonOptions()
            for factory in (
                lambda: hv.IndexRange.all(hv.Shape(dimensions).element_count),
                lambda: set(),
            ):
                with self.assertRaises(RuntimeError):
                    hv.compare(tensor, tensor, options, factory(), [0, 0, 2])

    def test_prepared_type_generation_matches_checked_pointwise(self):
        dimensions = (2, 3)
        layout = hv.Layout(hv.Shape(dimensions), [-7, 2], 7)
        components = (
            hv.GenerationRecipe.type_maximum(),
            hv.GenerationRecipe.uniform_type_range(),
            hv.GenerationRecipe.random_encoded_exponent(
                hv.RandomEncodedExponentGenerationParameters(
                    lower_unbiased_exponent=-3,
                    upper_unbiased_exponent=1,
                    source_type=None,
                )
            ),
        )
        for component in components:
            recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
                component.with_zero_outside_main_diagonal(),
                seed=0,
            ).with_index_order([1, 0])
            whole = hv.generate_tensor(hv.ScalarType.Float32, layout, recipe)
            pointwise = hv.Tensor(hv.ScalarType.Float32, layout)
            for index in range(6):
                hv.generate_at(pointwise, index, recipe)
            self.assertEqual(whole.storage, pointwise.storage)
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.uniform_finite_encoded_value(),
            seed=0,
        )
        whole = hv.generate_tensor(hv.ScalarType.Float6E2M3, layout, recipe)
        pointwise = hv.Tensor(hv.ScalarType.Float6E2M3, layout)
        for index in range(6):
            hv.generate_at(pointwise, index, recipe)
        self.assertEqual(whole.storage, pointwise.storage)
