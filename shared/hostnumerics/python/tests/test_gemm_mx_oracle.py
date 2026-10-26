# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _gemm_oracle_support import *  # noqa: F403


class GemmMxOracleTests(unittest.TestCase):  # noqa: F405
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
            strides=None,
            offset=0,
        )
        scale_b = hv.Tensor.from_storage(
            hv.ScalarType.E8M0,
            list(scale_b_values.shape),
            bytes([128, 127, 126, 127, 129, 128, 126, 128, 129]),
            strides=None,
            offset=0,
        )
        selected = [0, 4, 5]

        operand_a = hv.from_numpy(left, hv.ScalarType.Float4E2M1)
        operand_b = hv.from_numpy(right, hv.ScalarType.Float4E2M1)
        output = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3]))
        result = reference_gemm_into(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3])),
            output,
            block_scale_a=scale_a,
            block_scale_b=scale_b,
            block_size_a=block_a,
            block_size_b=block_b,
            indices=set(selected),
            index_order=hv.LastDimensionFastest(),
            backend=hv.GemmBackend.Blocked,
            accumulator_type=hv.ScalarType.Float32,
            alpha=1.0,
            beta=0.0,
            compute_type_a=None,
            compute_type_b=None,
            math_mode=hv.MathMode.Default,
            output_conversion=hv.OutputConversion.Default,
            accumulation_rounding=hv.AccumulationRounding.TypeDefault,
            conjugate_a=False,
            conjugate_b=False,
        )

        expected_complete = per_k_block_scaled_gemm(
            left, right, scale_a_values, scale_b_values, block_a, block_b
        )
        expected = np.zeros_like(expected_complete)
        expected.reshape(-1)[selected] = expected_complete.reshape(-1)[selected]
        np.testing.assert_array_equal(hv.to_numpy(output, dtype=None), expected)
        self.assertIsNone(result)

    def test_mxfp4_scale_is_applied_after_the_complete_segment(self):
        reductions = 16
        left = np.ones((1, reductions), dtype=np.float32)
        right = np.ones((reductions, 1), dtype=np.float32)
        right[reductions // 2 :] = -1.0
        scale_a = hv.Tensor.from_storage(
            hv.ScalarType.E8M0,
            [1, 1],
            bytes([254]),  # 2**127: scaling either half separately overflows float32.
            strides=None,
            offset=0,
        )
        output = hv.Tensor(hv.ScalarType.Float32, hv.Shape([1, 1]))

        result = reference_gemm_into(
            hv.from_numpy(left, hv.ScalarType.Float4E2M1),
            hv.from_numpy(right, hv.ScalarType.Float4E2M1),
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([1, 1])),
            output,
            accumulator_type=hv.ScalarType.Float32,
            block_scale_a=scale_a,
            block_size_a=reductions,
            backend=hv.GemmBackend.Blocked,
            alpha=1.0,
            beta=0.0,
            compute_type_a=None,
            compute_type_b=None,
            math_mode=hv.MathMode.Default,
            indices=hv.IndexRange.all(),
            index_order=hv.LastDimensionFastest(),
            output_conversion=hv.OutputConversion.Default,
            accumulation_rounding=hv.AccumulationRounding.TypeDefault,
            conjugate_a=False,
            conjugate_b=False,
        )

        self.assertTrue(np.isfinite(hv.to_numpy(output, dtype=None)[0, 0]))
        np.testing.assert_array_equal(
            hv.to_numpy(output, dtype=None), np.zeros((1, 1), dtype=np.float32)
        )
        self.assertIsNone(result)

    def test_one_sided_block_scales_are_independent(self):
        left = np.ones((2, 4), dtype=np.float32)
        right = np.ones((4, 2), dtype=np.float32)
        initial = np.zeros((2, 2), dtype=np.float32)
        scale_a = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        scale_b = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

        only_a = reference_gemm(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            block_scale_a=hv.from_numpy(scale_a),
            block_size_a=2,
            alpha=1.0,
            beta=0.0,
            compute_type_a=None,
            compute_type_b=None,
            math_mode=hv.MathMode.Default,
            indices=hv.IndexRange.all(),
            index_order=hv.LastDimensionFastest(),
            backend=hv.GemmBackend.Automatic,
            output_conversion=hv.OutputConversion.Default,
            accumulation_rounding=hv.AccumulationRounding.TypeDefault,
            conjugate_a=False,
            conjugate_b=False,
            output_layout=None,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(only_a, dtype=None),
            np.asarray([[10.0, 10.0], [18.0, 18.0]], dtype=np.float32),
        )

        only_b = reference_gemm(
            hv.from_numpy(left),
            hv.from_numpy(right),
            hv.from_numpy(initial),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            block_scale_b=hv.from_numpy(scale_b),
            block_size_b=2,
            alpha=1.0,
            beta=0.0,
            compute_type_a=None,
            compute_type_b=None,
            math_mode=hv.MathMode.Default,
            indices=hv.IndexRange.all(),
            index_order=hv.LastDimensionFastest(),
            backend=hv.GemmBackend.Automatic,
            output_conversion=hv.OutputConversion.Default,
            accumulation_rounding=hv.AccumulationRounding.TypeDefault,
            conjugate_a=False,
            conjugate_b=False,
            output_layout=None,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(only_b, dtype=None),
            np.asarray([[10.0, 18.0], [10.0, 18.0]], dtype=np.float32),
        )

        with self.assertRaisesRegex(
            RuntimeError, "A block size requires a scale tensor"
        ):
            reference_gemm(
                hv.from_numpy(left),
                hv.from_numpy(right),
                hv.from_numpy(initial),
                hv.ScalarType.Float32,
                hv.ScalarType.Float32,
                block_size_a=2,
                alpha=1.0,
                beta=0.0,
                compute_type_a=None,
                compute_type_b=None,
                math_mode=hv.MathMode.Default,
                indices=hv.IndexRange.all(),
                index_order=hv.LastDimensionFastest(),
                backend=hv.GemmBackend.Automatic,
                output_conversion=hv.OutputConversion.Default,
                accumulation_rounding=hv.AccumulationRounding.TypeDefault,
                conjugate_a=False,
                conjugate_b=False,
                output_layout=None,
            )
