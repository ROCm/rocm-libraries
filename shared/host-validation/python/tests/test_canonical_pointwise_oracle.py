# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import roc_host_validation as hv


def wrap_int32(value):
    unsigned = int(value) & 0xFFFFFFFF
    return unsigned if unsigned < 0x80000000 else unsigned - 0x100000000


def exact_int32_gemm(left, right, initial, alpha, beta, output_scale):
    """Use unbounded Python integers and wrap after every defined Int32 operation."""

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.int32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = 0
            for reduction in range(left.shape[1]):
                product = wrap_int32(
                    int(left[row, reduction]) * int(right[reduction, column])
                )
                accumulation = wrap_int32(accumulation + product)
            combined = wrap_int32(
                wrap_int32(alpha * accumulation)
                + wrap_int32(beta * int(initial[row, column]))
            )
            result[row, column] = wrap_int32(combined * output_scale)
    return result


def per_k_block_scaled_gemm(left, right, scale_a, scale_b, block_a, block_b):
    """Apply each operand's scale directly at k, independent of block traversal."""

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = np.float32(0.0)
            for reduction in range(left.shape[1]):
                term = np.float32(left[row, reduction] * right[reduction, column])
                term = np.float32(term * scale_a[row, reduction // block_a])
                term = np.float32(term * scale_b[column, reduction // block_b])
                accumulation = np.float32(accumulation + term)
            result[row, column] = accumulation
    return result


def affine_tensor(values, scalar_type, strides, offset):
    values = np.asarray(values)
    maximum_offset = offset
    for extent, stride in zip(values.shape, strides):
        maximum_offset += (extent - 1) * stride
    storage = np.zeros(maximum_offset + 1, dtype=values.dtype)
    for coordinates in np.ndindex(values.shape):
        storage_index = offset + sum(
            coordinate * stride for coordinate, stride in zip(coordinates, strides)
        )
        storage[storage_index] = values[coordinates]
    return hv.Tensor.from_storage(
        scalar_type,
        list(values.shape),
        storage.tobytes(),
        strides=list(strides),
        offset=offset,
    )


class CanonicalPointwiseOracleTests(unittest.TestCase):
    def test_selected_int32_outputs_wrap_in_affine_storage(self):
        left = np.asarray(
            [
                [2_147_483_647, -2_147_483_648, 123_456_789],
                [-2_000_000_000, 2_000_000_000, -17],
            ],
            dtype=np.int32,
        )
        right = np.asarray([[-3, 5], [7, -11], [12_345, -54_321]], dtype=np.int32)
        initial = np.asarray(
            [[2_147_483_647, -2_147_483_648], [13_579, -24_680]],
            dtype=np.int32,
        )
        alpha = -3
        beta = 5
        output_scale = -7
        selected = [1, 2]
        output_layout = hv.Layout(hv.Shape([2, 2]), [7, 2], 1)

        request = hv.GemmRequest(
            hv.GemmOperand(hv.from_numpy(left)),
            hv.GemmOperand(hv.from_numpy(right)),
            hv.from_numpy(initial),
            output_type=hv.ScalarType.Int32,
            accumulator_type=hv.ScalarType.Int32,
            output_layout=output_layout,
        )
        request.epilogue.alpha = alpha
        request.epilogue.beta = beta
        request.epilogue.output_scale = output_scale
        request.output_selection = hv.OutputSelection.explicit_indices(selected)

        result = hv.reference_gemm_result(
            request, hv.GemmExecution(hv.GemmBackend.Canonical, True)
        )

        complete_expected = exact_int32_gemm(
            left, right, initial, alpha, beta, output_scale
        )
        expected = np.zeros_like(complete_expected)
        expected.reshape(-1)[selected] = complete_expected.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)

        expected_storage = np.zeros(11, dtype=np.int32)
        expected_storage[[3, 8]] = complete_expected.reshape(-1)[selected]
        np.testing.assert_array_equal(
            np.frombuffer(result.output.storage, dtype=np.int32), expected_storage
        )
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Canonical)
        self.assertEqual(result.run_info.output_elements_computed, len(selected))

    def test_unequal_mx_blocks_and_k_tail_use_per_k_scale_oracle(self):
        left = np.asarray(
            [
                [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, 0.5, -3.0, 1.5, 2.0],
                [-0.5, 1.0, 2.0, -4.0, 3.0, 0.5, -1.5, 2.0, 1.0, -2.0, 6.0],
            ],
            dtype=np.float32,
        )
        right = np.asarray(
            [
                [1.0, -0.5, 2.0],
                [2.0, 1.0, -1.0],
                [-1.5, 2.0, 0.5],
                [0.5, -2.0, 3.0],
                [4.0, 0.5, -0.5],
                [-2.0, 1.5, 1.0],
                [1.0, 3.0, -2.0],
                [2.0, -1.0, 0.5],
                [-0.5, 2.0, 4.0],
                [3.0, 1.0, -1.5],
                [1.5, -2.0, 2.0],
            ],
            dtype=np.float32,
        )
        block_a = 3
        block_b = 4
        scale_a_values = np.asarray(
            [[0.5, 2.0, 1.0, 4.0], [1.0, 0.5, 4.0, 2.0]], dtype=np.float32
        )
        scale_b_values = np.asarray(
            [[2.0, 1.0, 0.5], [1.0, 4.0, 2.0], [0.5, 2.0, 4.0]],
            dtype=np.float32,
        )
        # E8M0 stores powers of two as exponent+bias, with bias 127.
        scale_a = hv.Tensor.from_storage(
            hv.ScalarType.E8M0,
            list(scale_a_values.shape),
            bytes([126, 128, 127, 129, 127, 126, 129, 128]),
        )
        scale_b = hv.Tensor.from_storage(
            hv.ScalarType.E8M0,
            list(scale_b_values.shape),
            bytes([128, 127, 126, 127, 129, 128, 126, 128, 129]),
        )
        selected = [0, 4, 5]

        operand_a = hv.GemmOperand(hv.from_numpy(left, hv.ScalarType.Float4E2M1))
        operand_a.block_scale = hv.BlockScaleBinding(scale_a, block_a)
        operand_b = hv.GemmOperand(hv.from_numpy(right, hv.ScalarType.Float4E2M1))
        operand_b.block_scale = hv.BlockScaleBinding(scale_b, block_b)
        request = hv.GemmRequest(
            operand_a,
            operand_b,
            output_type=hv.ScalarType.Float32,
            accumulator_type=hv.ScalarType.Float32,
        )
        request.output_selection = hv.OutputSelection.explicit_indices(selected)

        result = hv.reference_gemm_result(
            request, hv.GemmExecution(hv.GemmBackend.Canonical, True)
        )

        expected_complete = per_k_block_scaled_gemm(
            left, right, scale_a_values, scale_b_values, block_a, block_b
        )
        expected = np.zeros_like(expected_complete)
        expected.reshape(-1)[selected] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Canonical)
        self.assertEqual(result.run_info.output_elements_computed, len(selected))

    def test_complex_affine_inputs_and_output_respect_explicit_selection(self):
        left = np.asarray(
            [
                [1.0 + 2.0j, -3.0 + 1.0j, 0.5 - 1.0j],
                [2.0 - 0.5j, 1.0 + 0.0j, -2.0 - 2.0j],
            ],
            dtype=np.complex64,
        )
        right = np.asarray(
            [
                [2.0 - 1.0j, 1.0 + 0.5j],
                [-1.0 + 2.0j, 3.0 - 1.0j],
                [0.5 + 1.0j, -2.0 + 0.0j],
            ],
            dtype=np.complex64,
        )
        initial = np.asarray(
            [[1.0 + 0.0j, -1.0 + 1.0j], [2.0 - 2.0j, 0.5 + 0.5j]],
            dtype=np.complex64,
        )
        alpha = np.complex64(0.5 - 0.25j)
        beta = np.complex64(-1.0 + 0.5j)
        selected = [0, 3]
        output_layout = hv.Layout(hv.Shape([2, 2]), [8, 3], 2)

        operand_a = hv.GemmOperand(
            affine_tensor(left, hv.ScalarType.ComplexFloat32, strides=[6, 1], offset=1)
        )
        operand_a.conjugate = True
        request = hv.GemmRequest(
            operand_a,
            hv.GemmOperand(
                affine_tensor(
                    right,
                    hv.ScalarType.ComplexFloat32,
                    strides=[1, 5],
                    offset=2,
                )
            ),
            affine_tensor(
                initial,
                hv.ScalarType.ComplexFloat32,
                strides=[5, 2],
                offset=1,
            ),
            output_type=hv.ScalarType.ComplexFloat32,
            accumulator_type=hv.ScalarType.ComplexFloat32,
            output_layout=output_layout,
        )
        request.epilogue.alpha = complex(alpha)
        request.epilogue.beta = complex(beta)
        request.output_selection = hv.OutputSelection.explicit_indices(selected)

        result = hv.reference_gemm_result(
            request, hv.GemmExecution(hv.GemmBackend.Canonical, True)
        )

        expected_complete = np.complex64(alpha) * (np.conjugate(left) @ right)
        expected_complete = np.asarray(
            expected_complete + np.complex64(beta) * initial, dtype=np.complex64
        )
        expected = np.zeros_like(expected_complete)
        expected.reshape(-1)[selected] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)

        expected_storage = np.zeros(14, dtype=np.complex64)
        expected_storage[[2, 13]] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(
            np.frombuffer(result.output.storage, dtype=np.complex64),
            expected_storage,
        )
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Canonical)
        self.assertEqual(result.run_info.output_elements_computed, len(selected))


if __name__ == "__main__":
    unittest.main()
