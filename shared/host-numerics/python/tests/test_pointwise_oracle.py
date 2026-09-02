# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import roc_host_numerics as hv


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


def quantize_bfloat16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    least_significant_bit = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + least_significant_bit
    result = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return result[()] if result.ndim == 0 else result


def stepwise_gemm(left, right, accumulator_type):
    """Round every product and sum in an independently implemented accumulator."""

    if accumulator_type == hv.ScalarType.Float16:
        quantize = lambda value: np.float32(np.float16(value))
    elif accumulator_type == hv.ScalarType.BFloat16:
        quantize = quantize_bfloat16
    else:
        raise ValueError("stepwise_gemm requires a reduced-precision accumulator")

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = np.float32(0.0)
            for reduction in range(left.shape[1]):
                product = quantize(
                    np.float32(left[row, reduction] * right[reduction, column])
                )
                accumulation = quantize(np.float32(accumulation + product))
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


def saturating_int8(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


def selected_values(values, selected):
    result = np.zeros_like(values)
    result.reshape(-1)[selected] = values.reshape(-1)[selected]
    return result


def modular_values(rows, columns, row_factor, column_factor, modulus, center, divisor):
    return (
        (
            np.arange(rows)[:, None] * row_factor
            + np.arange(columns)[None, :] * column_factor
        )
        % modulus
        - center
    ).astype(np.float32) / np.float32(divisor)


class PointwiseOracleTests(unittest.TestCase):
    def test_zero_matrix_extents_are_no_ops(self):
        for rows, columns, reductions in ((0, 4, 3), (2, 0, 3), (2, 4, 0)):
            with self.subTest(rows=rows, columns=columns, reductions=reductions):
                left = np.empty((rows, reductions), dtype=np.float32)
                right = np.empty((reductions, columns), dtype=np.float32)
                initial = np.arange(rows * columns, dtype=np.float32).reshape(
                    rows, columns
                )
                expected = np.float32(1.5) * initial
                for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
                    observed = hv.reference_gemm(
                        hv.from_numpy(left),
                        hv.from_numpy(right),
                        hv.from_numpy(initial),
                        hv.ScalarType.Float32,
                        hv.ScalarType.Float32,
                        alpha=np.nan if reductions == 0 else 1.0,
                        beta=1.5,
                        backend=backend,
                    )
                    self.assertEqual(observed.shape, [rows, columns])
                    np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_selected_affine_reduced_precision_accumulation_rounds_each_step(self):
        left = np.asarray(
            [
                [0.1] * 16,
                [0.2, -0.3, 0.4, -0.5] * 4,
            ],
            dtype=np.float32,
        )
        right = np.asarray(
            [[0.1, np.float32((index % 5) - 2) / 7.0] for index in range(16)],
            dtype=np.float32,
        )
        initial = np.zeros((2, 2), dtype=np.float32)
        selected = [0, 3]
        output_layout = hv.Layout(hv.Shape([2, 2]), [7, 2], 1)

        full_precision = np.asarray(left @ right, dtype=np.float32)
        for accumulator_type in (
            hv.ScalarType.Float16,
            hv.ScalarType.BFloat16,
        ):
            with self.subTest(accumulator_type=accumulator_type):
                complete_expected = stepwise_gemm(left, right, accumulator_type)
                expected = selected_values(complete_expected, selected)
                self.assertTrue(
                    np.any(
                        complete_expected.reshape(-1)[selected]
                        != full_precision.reshape(-1)[selected]
                    )
                )

                observed = hv.reference_gemm(
                    affine_tensor(left, hv.ScalarType.Float32, [19, 1], 2),
                    affine_tensor(right, hv.ScalarType.Float32, [3, 1], 1),
                    affine_tensor(initial, hv.ScalarType.Float32, [5, 2], 1),
                    hv.ScalarType.Float32,
                    accumulator_type,
                    output_selection=hv.OutputSelection.explicit_indices(selected),
                    output_layout=output_layout,
                    backend=hv.GemmBackend.Pointwise,
                )

                self.assertEqual(observed.strides, [7, 2])
                self.assertEqual(observed.offset, 1)
                np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_selected_affine_zero_coefficients_suppress_poisoned_inputs(self):
        selected = [0, 3]
        selection = hv.OutputSelection.explicit_indices(selected)
        output_layout = hv.Layout(hv.Shape([2, 2]), [7, 2], 1)
        finite_left = np.asarray([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        finite_right = np.asarray(
            [[2.0, -1.0], [0.5, 3.0], [-4.0, 2.0]], dtype=np.float32
        )
        finite_initial = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        alpha_zero = hv.reference_gemm(
            affine_tensor(
                np.full((2, 3), np.nan, dtype=np.float32),
                hv.ScalarType.Float32,
                [5, 1],
                1,
            ),
            affine_tensor(
                np.full((3, 2), np.nan, dtype=np.float32),
                hv.ScalarType.Float32,
                [3, 1],
                1,
            ),
            affine_tensor(finite_initial, hv.ScalarType.Float32, [5, 2], 1),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=0.0,
            beta=2.0,
            output_selection=selection,
            output_layout=output_layout,
            backend=hv.GemmBackend.Pointwise,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(alpha_zero), selected_values(2.0 * finite_initial, selected)
        )

        beta_zero = hv.reference_gemm(
            affine_tensor(finite_left, hv.ScalarType.Float32, [5, 1], 1),
            affine_tensor(finite_right, hv.ScalarType.Float32, [3, 1], 1),
            affine_tensor(
                np.full((2, 2), np.nan, dtype=np.float32),
                hv.ScalarType.Float32,
                [5, 2],
                1,
            ),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=1.0,
            beta=0.0,
            output_selection=selection,
            output_layout=output_layout,
            backend=hv.GemmBackend.Pointwise,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(beta_zero),
            selected_values(finite_left @ finite_right, selected),
        )

    def test_negative_stride_numpy_gemm_operands_match_numpy(self):
        left = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        right = (np.arange(15, dtype=np.float32).reshape(3, 5) - 4.0)[::-1, ::-1]
        initial = np.arange(20, dtype=np.float32).reshape(4, 5)[:, ::-1]

        left_tensor = hv.Tensor.from_numpy(left)
        right_tensor = hv.Tensor.from_numpy(right)
        initial_tensor = hv.Tensor.from_numpy(initial)
        self.assertTrue(any(stride < 0 for stride in left_tensor.strides))
        self.assertTrue(any(stride < 0 for stride in right_tensor.strides))

        observed = hv.reference_gemm(
            left_tensor,
            right_tensor,
            initial_tensor,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=0.75,
            beta=-0.25,
            backend=hv.GemmBackend.Pointwise,
        )
        expected = np.float32(0.75) * (left @ right) + np.float32(-0.25) * initial
        np.testing.assert_allclose(
            hv.to_numpy(observed), expected, rtol=1e-6, atol=1e-6
        )

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

        output = hv.Tensor(hv.ScalarType.Int32, output_layout)
        backend = hv.reference_gemm_into(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            output,
            accumulator_type=hv.ScalarType.Int32,
            alpha=alpha,
            beta=beta,
            output_scale=output_scale,
            output_selection=hv.OutputSelection.explicit_indices(selected),
            backend=hv.GemmBackend.Pointwise,
        )

        complete_expected = exact_int32_gemm(
            left, right, initial, alpha, beta, output_scale
        )
        expected = np.zeros_like(complete_expected)
        expected.reshape(-1)[selected] = complete_expected.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(output), expected)

        expected_storage = np.zeros(11, dtype=np.int32)
        expected_storage[[3, 8]] = complete_expected.reshape(-1)[selected]
        np.testing.assert_array_equal(
            np.frombuffer(output.storage, dtype=np.int32), expected_storage
        )
        self.assertEqual(backend, hv.GemmBackend.Pointwise)

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

        operand_a = hv.from_numpy(left, hv.ScalarType.Float4E2M1)
        operand_b = hv.from_numpy(right, hv.ScalarType.Float4E2M1)
        output = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3]))
        backend = hv.reference_gemm_into(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3])),
            output,
            block_scale_a=scale_a,
            block_scale_b=scale_b,
            block_size_a=block_a,
            block_size_b=block_b,
            output_selection=hv.OutputSelection.explicit_indices(selected),
            backend=hv.GemmBackend.Pointwise,
        )

        expected_complete = per_k_block_scaled_gemm(
            left, right, scale_a_values, scale_b_values, block_a, block_b
        )
        expected = np.zeros_like(expected_complete)
        expected.reshape(-1)[selected] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(output), expected)
        self.assertEqual(backend, hv.GemmBackend.Pointwise)

    def test_one_sided_block_scales_are_independent(self):
        left = np.ones((2, 4), dtype=np.float32)
        right = np.ones((4, 2), dtype=np.float32)
        initial = np.zeros((2, 2), dtype=np.float32)
        scale_a = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        scale_b = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

        only_a = hv.reference_gemm(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            block_scale_a=hv.from_numpy(scale_a),
            block_size_a=2,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(only_a),
            np.asarray([[10.0, 10.0], [18.0, 18.0]], dtype=np.float32),
        )

        only_b = hv.reference_gemm(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            block_scale_b=hv.from_numpy(scale_b),
            block_size_b=2,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(only_b),
            np.asarray([[10.0, 18.0], [10.0, 18.0]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "A block size requires a scale tensor"):
            hv.reference_gemm(
                hv.from_numpy(left),
                hv.from_numpy(right),
                hv.from_numpy(initial),
                hv.ScalarType.Float32,
                hv.ScalarType.Float32,
                block_size_a=2,
            )

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

        operand_a = affine_tensor(
            left, hv.ScalarType.ComplexFloat32, strides=[6, 1], offset=1
        )
        operand_b = affine_tensor(
            right,
            hv.ScalarType.ComplexFloat32,
            strides=[1, 5],
            offset=2,
        )
        initial_tensor = affine_tensor(
            initial,
            hv.ScalarType.ComplexFloat32,
            strides=[5, 2],
            offset=1,
        )
        output = hv.Tensor(hv.ScalarType.ComplexFloat32, output_layout)
        backend = hv.reference_gemm_into(
            operand_a,
            operand_b,
            initial_tensor,
            output,
            accumulator_type=hv.ScalarType.ComplexFloat32,
            alpha=complex(alpha),
            beta=complex(beta),
            conjugate_a=True,
            output_selection=hv.OutputSelection.explicit_indices(selected),
            backend=hv.GemmBackend.Pointwise,
        )

        expected_complete = np.complex64(alpha) * (np.conjugate(left) @ right)
        expected_complete = np.asarray(
            expected_complete + np.complex64(beta) * initial, dtype=np.complex64
        )
        expected = np.zeros_like(expected_complete)
        expected.reshape(-1)[selected] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(output), expected)

        expected_storage = np.zeros(14, dtype=np.complex64)
        expected_storage[[2, 13]] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(
            np.frombuffer(output.storage, dtype=np.complex64),
            expected_storage,
        )
        self.assertEqual(backend, hv.GemmBackend.Pointwise)


class GemmFinalizationOracleTests(unittest.TestCase):
    def test_epilogue_operations_precede_saturating_conversion(self):
        pre_activation = np.asarray(
            [-259.0, -5.0, -3.0, -1.0, 0.625, 0.75, 63.75, np.inf],
            dtype=np.float32,
        )
        bias = np.where(
            np.arange(pre_activation.size) % 2 == 0,
            np.float32(0.25),
            np.float32(-0.25),
        )
        inputs = (pre_activation - bias).reshape(1, -1)
        expected_raw = (
            np.where(pre_activation > 0.0, pre_activation, 0.25 * pre_activation) * 2.0
        ).reshape(1, -1)
        expected_output = saturating_int8(expected_raw)
        selected = [0, 2, 4, 6, 7]

        def run(selection):
            return hv.reference_epilogue(
                hv.from_numpy(inputs),
                hv.ScalarType.Int8,
                hv.ScalarType.Float32,
                bias=hv.from_numpy(bias),
                activation=hv.Activation.LeakyRelu,
                activation_parameter0=0.25,
                output_scale=2.0,
                output_conversion=hv.OutputConversion.SaturatingInt8,
                include_raw_output=True,
                output_selection=selection,
            )

        full = run(hv.OutputSelection.all())
        partial = run(hv.OutputSelection.explicit_indices(selected))

        self.assertIsNotNone(full.raw_output)
        np.testing.assert_array_equal(hv.to_numpy(full.raw_output), expected_raw)
        np.testing.assert_array_equal(hv.to_numpy(full.output), expected_output)
        np.testing.assert_array_equal(
            hv.to_numpy(partial.output),
            selected_values(hv.to_numpy(full.output), selected),
        )

    def test_int32_wrapping_precedes_saturating_conversion(self):
        targets = np.asarray(
            [[-129, -128, -127, -1, 0, 1, 126, 127, 128]], dtype=np.int32
        )
        left = np.asarray([[np.iinfo(np.int32).max]], dtype=np.int32)
        right = np.arange(2, 11, dtype=np.int32).reshape(1, -1)
        alpha = -3
        output_scale = -7
        alpha_accumulation = exact_int32_gemm(
            left, right, np.zeros_like(targets), alpha, 0, 1
        )
        inverse_scale = pow(output_scale & 0xFFFFFFFF, -1, 1 << 32)
        before_scale = np.asarray(
            [wrap_int32(int(value) * inverse_scale) for value in targets.flat],
            dtype=np.int32,
        ).reshape(targets.shape)
        initial = np.asarray(
            [
                wrap_int32(int(goal) - int(accumulation))
                for goal, accumulation in zip(
                    before_scale.flat, alpha_accumulation.flat
                )
            ],
            dtype=np.int32,
        ).reshape(targets.shape)

        wrapped = exact_int32_gemm(left, right, initial, alpha, 1, output_scale)
        np.testing.assert_array_equal(wrapped, targets)
        observed = hv.reference_gemm(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            hv.ScalarType.Int8,
            hv.ScalarType.Int32,
            alpha=alpha,
            beta=1,
            output_scale=output_scale,
            output_conversion=hv.OutputConversion.SaturatingInt8,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), saturating_int8(targets))

    def test_selected_float_results_match_full_results_across_backends(self):
        rows = 35
        reduction = 65
        columns = 37
        left = modular_values(rows, reduction, 17, 13, 29, 14, 7)
        right = modular_values(reduction, columns, 11, 19, 31, 15, 9)
        initial = modular_values(rows, columns, 5, 7, 17, 8, 6)
        alpha = np.float32(0.75)
        beta = np.float32(-0.25)
        selected = [
            0,
            columns - 1,
            columns,
            (rows // 2) * columns + columns // 2,
            (rows - 1) * columns,
            rows * columns - 1,
        ]
        unselected = np.ones(rows * columns, dtype=bool)
        unselected[selected] = False
        tensors = tuple(hv.from_numpy(values) for values in (left, right, initial))
        expected = float(alpha) * (
            left.astype(np.float64) @ right.astype(np.float64)
        ) + float(beta) * initial.astype(np.float64)
        relative_tolerance = 1.0e-4
        absolute_tolerance = 1.0e-5
        full_outputs = []
        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                full = hv.to_numpy(
                    hv.reference_gemm(
                        *tensors,
                        hv.ScalarType.Float32,
                        hv.ScalarType.Float32,
                        alpha=float(alpha),
                        beta=float(beta),
                        backend=backend,
                    )
                )
                partial = hv.to_numpy(
                    hv.reference_gemm(
                        *tensors,
                        hv.ScalarType.Float32,
                        hv.ScalarType.Float32,
                        alpha=float(alpha),
                        beta=float(beta),
                        output_selection=hv.OutputSelection.explicit_indices(selected),
                        backend=backend,
                    )
                )
                np.testing.assert_allclose(
                    full, expected, rtol=relative_tolerance, atol=absolute_tolerance
                )
                np.testing.assert_allclose(
                    partial.reshape(-1)[selected],
                    full.reshape(-1)[selected],
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                np.testing.assert_array_equal(partial.reshape(-1)[unselected], 0.0)
                full_outputs.append(full)

        np.testing.assert_allclose(
            full_outputs[0],
            full_outputs[1],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )


if __name__ == "__main__":
    unittest.main()
