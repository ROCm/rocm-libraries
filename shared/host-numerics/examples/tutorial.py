# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable walkthrough of the public Python API.

The numbered tests are intentionally small and ordered like a tutorial. CTest
runs this file so the examples stay synchronized with the extension module.
"""

import unittest

import numpy as np

import roc_host_numerics as hn


def real_recipe(component, seed):
    return hn.GenerationRecipe.real_only(
        component,
        hn.GenerationRecipeSettings(
            seed=seed,
            index_order=hn.IndexOrder.LastDimensionFastest,
        ),
    )


class HostNumericsTutorial(unittest.TestCase):
    def test_01_create_and_convert_tensors(self):
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = hn.from_numpy(values)
        self.assertEqual(tensor.type, hn.ScalarType.Float32)
        self.assertEqual(tensor.shape, [2, 2])
        np.testing.assert_array_equal(hn.to_numpy(tensor), values)

        fp4 = hn.from_numpy(
            np.asarray([0.0, 0.5, 1.0, 6.0], dtype=np.float32),
            hn.ScalarType.Float4E2M1,
        )
        self.assertEqual(len(fp4.storage), 2)
        np.testing.assert_array_equal(
            hn.to_numpy(fp4), np.asarray([0.0, 0.5, 1.0, 6.0], dtype=np.float32)
        )

    def test_02_generate_reproducible_tensors(self):
        real = real_recipe(
            hn.GenerationRecipe.uniform_real(
                hn.UniformRealGenerationParameters(lower=-1.0, upper=1.0)
            ),
            seed=17,
        )
        first = hn.generate_tensor(hn.ScalarType.Float32, [8], real)
        replay = hn.generate_tensor(hn.ScalarType.Float32, [8], real)
        self.assertEqual(first.storage, replay.storage)

        integers = hn.generate_tensor(
            hn.ScalarType.Int32,
            [8],
            real_recipe(
                hn.GenerationRecipe.uniform_integer(
                    hn.UniformIntegerGenerationParameters(lower=1, upper=4)
                ),
                seed=27,
            ),
        )
        self.assertTrue(np.any(hn.to_numpy(integers) != 0))

        complex_values = hn.generate_tensor(
            hn.ScalarType.ComplexFloat32,
            [8],
            hn.GenerationRecipe.cartesian(
                hn.GenerationRecipe.uniform_integer(
                    hn.UniformIntegerGenerationParameters(lower=1, upper=4)
                ),
                hn.GenerationRecipe.uniform_integer(
                    hn.UniformIntegerGenerationParameters(lower=-4, upper=-1)
                ),
                hn.GenerationRecipeSettings(seed=37),
            ),
        )
        self.assertTrue(np.all(hn.to_numpy(complex_values).imag != 0))

    def test_03_compose_broadcast_tensor_operations(self):
        column = hn.from_numpy(np.asarray([[1.0], [2.0]], dtype=np.float32))
        row = hn.from_numpy(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32))
        result = 2.0 * column + row
        np.testing.assert_array_equal(
            hn.to_numpy(result),
            np.asarray([[12.0, 22.0, 32.0], [14.0, 24.0, 34.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hn.to_numpy(hn.relu(hn.clip(result, -20.0, 20.0))),
            np.asarray([[12.0, 20.0, 20.0], [14.0, 20.0, 20.0]], dtype=np.float32),
        )

    def test_04_multiply_matrices(self):
        a_values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b_values = np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
        product = hn.matmul(hn.from_numpy(a_values), hn.from_numpy(b_values))
        np.testing.assert_array_equal(hn.to_numpy(product), a_values @ b_values)

        integer_product = hn.matmul(
            hn.from_numpy(np.asarray([[2, 3]], dtype=np.int8)),
            hn.from_numpy(np.asarray([[4], [5]], dtype=np.int8)),
            output_type=hn.ScalarType.Int32,
            accumulator_type=hn.ScalarType.Int32,
        )
        np.testing.assert_array_equal(
            hn.to_numpy(integer_product), np.asarray([[23]], dtype=np.int32)
        )

        complex_a = np.asarray([[1 + 1j, 2 - 1j]], dtype=np.complex64)
        complex_b = np.asarray([[3 + 0j], [1 + 2j]], dtype=np.complex64)
        complex_product = hn.matmul(
            hn.from_numpy(complex_a),
            hn.from_numpy(complex_b),
            output_type=hn.ScalarType.ComplexFloat32,
            accumulator_type=hn.ScalarType.ComplexFloat32,
            conjugate_a=True,
        )
        np.testing.assert_array_equal(
            hn.to_numpy(complex_product), np.conjugate(complex_a) @ complex_b
        )

        empty_product = hn.matmul(
            hn.from_numpy(np.empty((2, 0), dtype=np.float32)),
            hn.from_numpy(np.empty((0, 3), dtype=np.float32)),
        )
        np.testing.assert_array_equal(
            hn.to_numpy(empty_product), np.zeros((2, 3), dtype=np.float32)
        )

    def test_05_generate_mxfp4_data_and_scales(self):
        recipe = real_recipe(
            hn.GenerationRecipe.uniform_real(
                hn.UniformRealGenerationParameters(lower=-1.0, upper=1.0)
            ),
            seed=47,
        )
        data = hn.MxDataGeneration.preserve_range(
            recipe, hn.MxRepresentedValueRange(lower=-1.0, upper=1.0)
        )
        options = hn.MxGenerationOptions()
        options.data_type = hn.ScalarType.Float4E2M1
        options.scale_type = hn.ScalarType.E8M0
        options.leading_dimension = 32
        options.block_axis = 0
        options.block_size = 32
        mx = hn.generate_mx([32, 2], data, options)

        self.assertEqual(mx.data.type, hn.ScalarType.Float4E2M1)
        self.assertEqual(mx.scales.shape, [2, 1])
        self.assertEqual(mx.scale_indices.shape, mx.data.shape)
        self.assertEqual(mx.reference.shape, mx.data.shape)

    def test_06_use_reductions_normalization_and_sparsity(self):
        values = hn.from_numpy(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        sums = hn.reference_sum(
            values,
            output_type=hn.ScalarType.Float32,
            accumulator_type=hn.ScalarType.Float32,
            axes=[1],
        )
        np.testing.assert_array_equal(hn.to_numpy(sums), np.asarray([3.0, 7.0]))

        probabilities = hn.reference_softmax(values, axis=1)
        np.testing.assert_allclose(
            np.sum(hn.to_numpy(probabilities), axis=1), np.ones(2), rtol=1e-6
        )

        normalized = hn.reference_layer_norm(
            values,
            output_type=hn.ScalarType.Float32,
            statistics_type=hn.ScalarType.Float32,
            axis=1,
        )
        self.assertEqual(normalized.mean.shape, [2])

        pattern = hn.StructuredSparsityPattern()
        pattern.axis = 1
        pattern.fixed_positions = [0, 2]
        sparse = hn.apply_structured_sparsity(
            hn.from_numpy(np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)),
            pattern,
            emit_two_of_four_metadata=True,
        )
        self.assertEqual(hn.to_numpy(sparse.pruned)[0, 1], 0.0)
        self.assertIsNotNone(sparse.retained_indices)
        self.assertIsNotNone(sparse.two_of_four_metadata)

    def test_07_inspect_a_comparison_failure(self):
        expected = hn.from_numpy(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        observed = hn.from_numpy(np.asarray([1.0, 9.0, 3.0], dtype=np.float32))
        report = hn.compare(observed, expected)
        self.assertFalse(report.passed)
        self.assertEqual(report.mismatches, 1)
        self.assertTrue(
            str(report).startswith("comparison failed: 1 of 3 compared elements mismatched")
        )
        self.assertIn("index 1 [1]", str(report))


if __name__ == "__main__":
    unittest.main()
