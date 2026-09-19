# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class TensorOperationsTests(unittest.TestCase):  # noqa: F405
    def test_tensor_arithmetic_matches_numpy(self):
        x_values = np.asarray(
            [[[1.0, -2.0], [3.0, 4.0]], [[-1.0, 2.0], [5.0, -6.0]]],
            dtype=np.float16,
        )
        y_source = np.asarray(
            [[[0.25, 1.1], [-3.5, 2.25]], [[4.0, -0.5], [1.75, 3.0]]],
            dtype=np.float32,
        )
        x = hv.from_numpy(x_values)
        y = hv.from_numpy(y_source, hv.ScalarType.BFloat16)
        result = hv.add(
            hv.multiply(
                x,
                0.5,
                output_type=hv.ScalarType.Float32,
                compute_type=hv.ScalarType.Float32,
            ),
            hv.multiply(
                y,
                -1.25,
                output_type=hv.ScalarType.Float32,
                compute_type=hv.ScalarType.Float32,
            ),
            output_type=hv.ScalarType.Float32,
            compute_type=hv.ScalarType.Float32,
        )

        y_values = quantize_bfloat16(y_source)
        expected = np.empty(x_values.shape, dtype=np.float32)
        for index in np.ndindex(x_values.shape):
            value = np.float32(0.0)
            value = np.float32(value + np.float32(0.5) * np.float32(x_values[index]))
            value = np.float32(value + np.float32(-1.25) * y_values[index])
            expected[index] = value
        np.testing.assert_array_equal(hv.to_numpy(result), expected)

        y_only = hv.multiply(y, 3.0)
        self.assertEqual(y_only.type, hv.ScalarType.BFloat16)
        np.testing.assert_array_equal(
            hv.to_numpy(y_only),
            quantize_bfloat16(np.float32(3.0) * y_values),
        )

        scalar_coefficient = hv.from_numpy(np.asarray(2.0, dtype=np.float32))
        tensor_scaled = hv.multiply(
            x,
            scalar_coefficient,
            output_type=hv.ScalarType.Float32,
            compute_type=hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(tensor_scaled), np.float32(2.0) * x_values
        )
        np.testing.assert_array_equal(
            hv.to_numpy(hv.multiply(x, x)), x_values * x_values
        )

        float_values = hv.from_numpy(np.asarray([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(
            hv.to_numpy(2.0 * float_values + 1.0),
            np.asarray([3.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(1.0 + float_values),
            np.asarray([2.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(hv.subtract(float_values, np.float32(0.5))),
            np.asarray([0.5, 1.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(10.0 - float_values * 2.0 - 1.0),
            np.asarray([7.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(-float_values), np.asarray([-1.0, -2.0], dtype=np.float32)
        )

        # The low bits must survive Python scalar conversion before the
        # operation's documented Int32 modulo conversion is applied.
        large_integer = 2**53 + 1
        integer_value = hv.from_numpy(np.asarray(0, dtype=np.int32))
        self.assertEqual((integer_value + large_integer).item(), 1)
        integer_values = hv.from_numpy(np.asarray([1, 2], dtype=np.int32))
        integer_plus_fraction = integer_values + np.float64(1.75)
        self.assertEqual(integer_plus_fraction.type, hv.ScalarType.Int32)
        np.testing.assert_array_equal(
            hv.to_numpy(integer_plus_fraction), np.asarray([2, 3], dtype=np.int32)
        )

        with self.assertRaises(ValueError):
            hv.add(x, y)

        packed = hv.from_numpy(
            np.asarray([0.5, 1.0, -1.5, 3.0], dtype=np.float32),
            hv.ScalarType.Float4E2M1,
        )
        packed_product = hv.multiply(
            packed,
            2.0,
            output_type=hv.ScalarType.Float32,
            compute_type=hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(packed_product),
            np.asarray([1.0, 2.0, -3.0, 6.0], dtype=np.float32),
        )

        padded_x = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            np.asarray([1.0, 2.0, -99.0, 3.0, 4.0], dtype=np.float32).tobytes(),
            strides=[3, 1],
        )
        contiguous = hv.multiply(padded_x, 1.0)
        self.assertEqual(contiguous.strides, [2, 1])
        self.assertEqual(contiguous.offset, 0)
        np.testing.assert_array_equal(
            hv.to_numpy(contiguous),
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        column = np.asarray([[1.0], [2.0]], dtype=np.float32)
        row = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)
        broadcast = hv.add(
            hv.multiply(hv.from_numpy(column), 2.0),
            hv.multiply(hv.from_numpy(row), -1.0),
        )
        np.testing.assert_array_equal(hv.to_numpy(broadcast), 2.0 * column - row)

        empty = hv.add(
            hv.from_numpy(np.empty((0, 3), dtype=np.float32)),
            hv.from_numpy(row),
        )
        self.assertEqual(empty.shape, [0, 3])

        activation_values = np.asarray([-2.0, -0.5, 0.0, 2.0], dtype=np.float32)
        activation_input = hv.from_numpy(activation_values)
        np.testing.assert_array_equal(
            hv.to_numpy(hv.relu(activation_input)), np.maximum(activation_values, 0.0)
        )
        np.testing.assert_array_equal(
            hv.to_numpy(hv.clip(activation_input, -1.0, 1.0)),
            np.clip(activation_values, -1.0, 1.0),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(activation_input.relu()), np.maximum(activation_values, 0.0)
        )
        np.testing.assert_array_equal(
            hv.to_numpy(activation_input.clip(-1.0, 1.0)),
            np.clip(activation_values, -1.0, 1.0),
        )

    def test_native_operation_direct_bindings(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        input_tensor = hv.from_numpy(values)
        reduction = hv.reference_reduce(
            input_tensor,
            [1],
            hv.ReductionOperation.Sum,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(reduction),
            np.sum(values, axis=1, dtype=np.float32),
        )

        reduction_output = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2]))
        hv.reference_reduce_into(
            input_tensor,
            reduction_output,
            [1],
            hv.ReductionOperation.Sum,
            hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(reduction_output),
            np.sum(values, axis=1, dtype=np.float32),
        )

        layer_norm_outputs = hv.reference_layer_norm(input_tensor, axis=1)
        self.assertEqual(layer_norm_outputs.output.shape, [2, 3])
        self.assertEqual(layer_norm_outputs.mean.shape, [2])
        self.assertEqual(layer_norm_outputs.inverse_variance.shape, [2])

        pattern = hv.StructuredSparsityPattern()
        pattern.axis = 0
        sparse = hv.apply_structured_sparsity(
            hv.from_numpy(np.arange(1, 9, dtype=np.float32)), pattern
        )
        self.assertIsNotNone(sparse.retained_indices)
        self.assertIsNone(sparse.two_of_four_metadata)
