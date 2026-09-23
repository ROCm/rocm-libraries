# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class GenerationTests(unittest.TestCase):  # noqa: F405
    def test_indexed_generation_matches_numpy(self):
        serial = hv.generate_tensor(
            hv.ScalarType.Float32,
            [2, 3],
            real_generation_recipe(hv.GenerationRecipe.serial_index()),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(serial),
            np.arange(6, dtype=np.float32).reshape((2, 3), order="F"),
        )

        complex_recipe = hv.GenerationRecipe.cartesian(
            hv.GenerationRecipe.sine(),
            hv.GenerationRecipe.cosine(),
        )
        complex_values = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32, [2, 3], complex_recipe
        )
        indices = np.arange(6, dtype=np.float32).reshape((2, 3), order="F")
        np.testing.assert_allclose(
            hv.to_numpy(complex_values),
            np.sin(indices) + 1j * np.cos(indices),
            rtol=1e-6,
            atol=1e-6,
        )

        identity = hv.generate_tensor(
            hv.ScalarType.Float32,
            [3, 4],
            real_generation_recipe(hv.GenerationRecipe.identity()),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(identity), np.eye(3, 4, dtype=np.float32)
        )

        seed = 19
        random_component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
        )
        random_recipe = hv.GenerationRecipe.cartesian(
            random_component,
            random_component,
            hv.GenerationRecipeSettings(seed=seed),
        )
        random_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            )
        )
        random_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            )
        )
        expected_real = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="F")
        expected_imaginary = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_IMAGINARY_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="F")
        np.testing.assert_array_equal(random_values, random_repeat)
        np.testing.assert_array_equal(random_values.real, expected_real)
        np.testing.assert_array_equal(random_values.imag, expected_imaginary)
        self.assertFalse(np.array_equal(expected_real, expected_imaginary))
        self.assertTrue(np.all((-3 <= expected_real) & (expected_real <= 3)))

        candidates = np.asarray([-6.0, -1.5, 0.0, 4.0], dtype=np.float32)
        seed = 37
        candidate_recipe = real_generation_recipe(
            hv.GenerationRecipe.choice(
                hv.ChoiceGenerationParameters(values=candidates.tolist())
            ),
            seed=seed,
        )
        selected = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 4], candidate_recipe)
        )
        expected = np.asarray(
            [
                candidates[
                    counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 4
                ]
                for index in range(8)
            ],
            dtype=np.float32,
        ).reshape((2, 4), order="F")
        np.testing.assert_array_equal(selected, expected)
        with self.assertRaises(ValueError):
            hv.GenerationRecipe.choice(hv.ChoiceGenerationParameters(values=[]))

        point = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3, 2]))
        first_point_recipe = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=9.0))
        )
        hv.generate_at(point, 3, first_point_recipe)
        expected_point = np.zeros((2, 3, 2), dtype=np.float32)
        expected_point[1, 1, 0] = 9.0
        np.testing.assert_array_equal(hv.to_numpy(point), expected_point)

        last_dimension_fastest = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=7.0)),
            index_order=hv.IndexOrder.LastDimensionFastest,
        )
        hv.generate_at(point, 3, last_dimension_fastest)
        expected_point[0, 1, 1] = 7.0
        np.testing.assert_array_equal(hv.to_numpy(point), expected_point)
        with self.assertRaises(IndexError):
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
                real_generation_recipe(affine_component),
            )
        )
        expected_affine = np.empty((2, 3, 2), dtype=np.float32)
        for index in np.ndindex(expected_affine.shape):
            numerator = -2 + index[0] - index[1] + 2 * index[2]
            expected_affine[index] = cxx_remainder(numerator, 5) + 1
        np.testing.assert_array_equal(affine, expected_affine)

    def test_type_derived_generation(self):
        maximum_recipe = real_generation_recipe(hv.GenerationRecipe.type_maximum())
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
                real_generation_recipe(hv.GenerationRecipe.type_denormal_minimum()),
            )
        )
        np.testing.assert_array_equal(
            fp4_denormal, np.asarray([0.5, 0.5], dtype=np.float32)
        )

        nan_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E4M3Fnuz,
                [2],
                real_generation_recipe(hv.GenerationRecipe.type_nan()),
            )
        )
        self.assertTrue(np.isnan(nan_values).all())

        infinity = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E5M2,
                [2],
                real_generation_recipe(hv.GenerationRecipe.type_infinity()),
            )
        )
        self.assertTrue(np.isposinf(infinity).all())

        type_range_recipe = real_generation_recipe(
            hv.GenerationRecipe.uniform_type_range(),
            seed=23,
        )
        low_precision = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float4E2M1,
                [64],
                type_range_recipe,
            )
        )
        self.assertTrue(np.all((-6.0 <= low_precision) & (low_precision <= 6.0)))
        float64_range = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float64, [64], type_range_recipe)
        )
        self.assertTrue(np.isfinite(float64_range).all())
        self.assertTrue(
            np.all(
                (-np.finfo(np.float64).max <= float64_range)
                & (float64_range <= np.finfo(np.float64).max)
            )
        )

        absolute_integer_recipe = real_generation_recipe(
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
            )
        )
        self.assertTrue(
            np.all((0 <= unsigned_scale_values) & (unsigned_scale_values <= 3))
        )

        encoded_exponent_recipe = real_generation_recipe(
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
            )
        )
        narrow_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                encoded_exponent_recipe,
            )
        )
        np.testing.assert_array_equal(narrow, narrow_repeat)
        exponent_bits = (narrow.view(np.uint32) >> 23) & np.uint32(0xFF)
        self.assertTrue(
            set(int(value) for value in exponent_bits).issubset({124, 125, 126})
        )

        raw_serial = hv.generate_tensor(
            hv.ScalarType.Float16,
            [2, 3],
            real_generation_recipe(
                hv.GenerationRecipe.raw_serial_dimension(
                    hv.DimensionGenerationParameters(dimension=1)
                )
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_serial.storage, dtype=np.uint16).reshape(2, 3),
            np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint16),
        )

        raw_zero = hv.generate_tensor(
            hv.ScalarType.E8M0,
            [4],
            real_generation_recipe(
                hv.GenerationRecipe.raw_constant(
                    hv.RawConstantGenerationParameters(bits=0)
                )
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_zero.storage, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
        )

        raw_fp4 = hv.generate_tensor(
            hv.ScalarType.Float4E2M1,
            [65],
            real_generation_recipe(
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

        raw_bits_recipe = real_generation_recipe(
            hv.GenerationRecipe.random_raw_bits(),
            seed=41,
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
                real_generation_recipe(scaled_component, seed=37),
            )
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
                real_generation_recipe(positive_component, seed=37),
            )
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
                real_generation_recipe(alternating_component),
            )
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
                real_generation_recipe(opposite_component),
            )
        )
        np.testing.assert_array_equal(opposite, -expected_matrix)
