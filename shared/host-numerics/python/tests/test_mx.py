# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class MxTests(unittest.TestCase):  # noqa: F405
    def test_mx_generation_matches_decoded_numpy_values(self):
        for dimensions, block_axis in (((64, 3), 0), ((3, 64), 1)):
            with self.subTest(dimensions=dimensions, block_axis=block_axis):
                data_generation = hv.MxDataGeneration.preserve_range(
                    real_generation_recipe(
                        hv.GenerationRecipe.uniform_real(
                            hv.UniformRealGenerationParameters(-1, 1)
                        ),
                    ),
                    hv.MxRepresentedValueRange(-1, 1),
                )
                options = hv.MxGenerationOptions()
                options.data_type = hv.ScalarType.Float4E2M1
                options.scale_type = hv.ScalarType.E8M0
                options.leading_dimension = dimensions[0]
                options.block_axis = block_axis
                options.block_size = 32

                first = hv.generate_mx(dimensions, data_generation, options)
                second = hv.generate_mx(dimensions, data_generation, options)
                self.assertEqual(first.data.storage, second.data.storage)
                self.assertEqual(first.scales.storage, second.scales.storage)

                data = hv.to_numpy(first.data)
                scales = hv.to_numpy(first.scales).reshape(-1)
                scale_indices = hv.to_numpy(first.scale_indices, np.uint32)
                reference = hv.to_numpy(first.reference)
                expected = np.empty(dimensions, dtype=np.float32)
                for row in range(dimensions[0]):
                    for column in range(dimensions[1]):
                        scale_index = scale_indices[row, column]
                        expected[row, column] = data[row, column] * scales[scale_index]
                np.testing.assert_array_equal(reference, expected)

    def test_bounded_mx_generation_matches_independent_python_oracle(self):
        for dimensions, leading_dimension, block_axis, minimum, maximum in (
            ((9, 5), 12, 0, -1.0, 1.0),
            ((5, 9), 8, 1, -1.0, 1.0),
            ((9, 5), 12, 0, 0.0, 0.9),
        ):
            with self.subTest(
                dimensions=dimensions,
                block_axis=block_axis,
                minimum=minimum,
                maximum=maximum,
            ):
                seed = 12345
                data_generation = hv.MxDataGeneration.preserve_range(
                    real_generation_recipe(
                        hv.GenerationRecipe.uniform_real(
                            hv.UniformRealGenerationParameters(minimum, maximum)
                        ),
                        seed=seed,
                    ),
                    hv.MxRepresentedValueRange(minimum, maximum),
                )
                options = hv.MxGenerationOptions()
                options.data_type = hv.ScalarType.Float4E2M1
                options.scale_type = hv.ScalarType.E8M0
                options.leading_dimension = leading_dimension
                options.block_axis = block_axis
                options.block_size = 4

                observed = hv.generate_mx(dimensions, data_generation, options)
                expected_data, expected_scales, expected_indices, expected_reference = (
                    bounded_mx_fp4_oracle(
                        dimensions,
                        leading_dimension,
                        block_axis,
                        options.block_size,
                        seed,
                        minimum,
                        maximum,
                    )
                )
                self.assertEqual(observed.data.storage, expected_data)
                self.assertEqual(observed.scales.storage, expected_scales)
                np.testing.assert_array_equal(
                    hv.to_numpy(observed.scale_indices, np.uint32),
                    expected_indices,
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(observed.reference, np.float32),
                    expected_reference,
                )
                self.assertTrue(
                    np.all(
                        (hv.to_numpy(observed.reference) >= minimum)
                        & (hv.to_numpy(observed.reference) <= maximum)
                    )
                )

    def test_mx_scale_policy_is_independent_of_data_generation(self):
        data_generation = hv.MxDataGeneration.preserve_range(
            real_generation_recipe(
                hv.GenerationRecipe.uniform_real(
                    hv.UniformRealGenerationParameters(-1.0, 1.0)
                ),
            ),
            hv.MxRepresentedValueRange(-1.0, 1.0),
        )
        options = hv.MxGenerationOptions()
        options.data_type = hv.ScalarType.Float4E2M1
        options.scale_type = hv.ScalarType.E8M0
        options.block_axis = 0
        options.block_size = 4
        options.scale = hv.MxScaleGenerationMode.One

        observed = hv.generate_mx([8, 8], data_generation, options)
        np.testing.assert_array_equal(
            hv.to_numpy(observed.scales),
            np.ones(observed.scales.shape, dtype=np.float32),
        )
