# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _gemm_oracle_support import *  # noqa: F403


class GemmAffineOracleTests(unittest.TestCase):  # noqa: F405
    def test_zero_matrix_extents_are_no_ops(self):
        for rows, columns, reductions in ((0, 4, 3), (2, 0, 3), (2, 4, 0)):
            with self.subTest(rows=rows, columns=columns, reductions=reductions):
                left = np.empty((rows, reductions), dtype=np.float32)
                right = np.empty((reductions, columns), dtype=np.float32)
                initial = np.arange(rows * columns, dtype=np.float32).reshape(
                    rows, columns
                )
                expected = np.float32(1.5) * initial
                observed = reference_gemm(
                    hv.from_numpy(left),
                    hv.from_numpy(right),
                    hv.from_numpy(initial),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    alpha=np.nan if reductions == 0 else 1.0,
                    beta=1.5,
                    backend=hv.GemmBackend.Blocked,
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

                observed = reference_gemm(
                    affine_tensor(left, hv.ScalarType.Float32, [19, 1], 2),
                    affine_tensor(right, hv.ScalarType.Float32, [3, 1], 1),
                    affine_tensor(initial, hv.ScalarType.Float32, [5, 2], 1),
                    hv.ScalarType.Float32,
                    accumulator_type,
                    output_selection=hv.OutputSelection.explicit_indices(selected),
                    output_layout=output_layout,
                    backend=hv.GemmBackend.Blocked,
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

        alpha_zero = reference_gemm(
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
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(alpha_zero), selected_values(2.0 * finite_initial, selected)
        )

        beta_zero = reference_gemm(
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
            backend=hv.GemmBackend.Blocked,
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

        observed = reference_gemm(
            left_tensor,
            right_tensor,
            initial_tensor,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=0.75,
            beta=-0.25,
            backend=hv.GemmBackend.Blocked,
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
        result = reference_gemm_into(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            output,
            accumulator_type=hv.ScalarType.Int32,
            alpha=alpha,
            beta=beta,
            output_scale=output_scale,
            output_selection=hv.OutputSelection.explicit_indices(selected),
            backend=hv.GemmBackend.Blocked,
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
        self.assertIsNone(result)
