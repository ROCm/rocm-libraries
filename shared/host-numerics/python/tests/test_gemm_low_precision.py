# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class GemmLowPrecisionTests(unittest.TestCase):  # noqa: F405
    def test_int32_accumulator_gemm_matches_exact_integer_oracle(self):
        a = np.asarray([[1, 3], [2, 4]], dtype=np.int8)
        b = np.asarray([[5], [6]], dtype=np.int8)
        c = np.zeros((2, 1), dtype=np.int32)
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Int32,
            hv.ScalarType.Int32,
        )
        expected = gemm_int32_exact(a, b, c)
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_int32_accumulator_gemm_wraps_without_float_proxy(self):
        reduction = 140_000
        a = np.full((1, reduction), 127, dtype=np.int8)
        b = np.full((reduction, 1), 127, dtype=np.int8)
        c = np.asarray([[np.iinfo(np.int32).max]], dtype=np.int32)
        alpha = 2
        beta = 2
        output_scale = -3
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Int32,
            hv.ScalarType.Int32,
            alpha=alpha,
            beta=beta,
            output_scale=output_scale,
        )
        expected = gemm_int32_exact(
            a,
            b,
            c,
            alpha=alpha,
            beta=beta,
            output_scale=output_scale,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_float16_accumulator_rounds_each_step(self):
        a = np.full((1, 64), np.float16(0.1), dtype=np.float16)
        b = np.full((64, 1), np.float16(0.1), dtype=np.float16)
        result = hv.matmul(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.ScalarType.Float16,
            hv.ScalarType.Float16,
        )

        expected = np.float16(0)
        for reduction in range(a.shape[1]):
            product = np.float16(a[0, reduction] * b[reduction, 0])
            expected = np.float16(expected + product)
        np.testing.assert_array_equal(
            hv.to_numpy(result, np.float32),
            np.asarray([[expected]], dtype=np.float32),
        )

    def test_bfloat16_accumulator_rounds_product_and_sum_each_step(self):
        a = np.full((1, 16), np.float32(0.1), dtype=np.float32)
        b = np.full((16, 1), np.float32(0.1), dtype=np.float32)
        observed = hv.matmul(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.ScalarType.Float32,
            hv.ScalarType.BFloat16,
        )

        expected = np.float32(0.0)
        for reduction in range(a.shape[1]):
            product = quantize_bfloat16(np.float32(a[0, reduction] * b[reduction, 0]))
            expected = quantize_bfloat16(np.float32(expected + product))

        self.assertNotEqual(expected, matmul_float32(a, b)[0, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(observed),
            np.asarray([[expected]], dtype=np.float32),
        )

    def test_bfloat16_accumulator_rounding_policy_is_explicit(self):
        a = np.full((1, 16), np.float32(0.1), dtype=np.float32)
        b = np.full((16, 1), np.float32(0.1), dtype=np.float32)

        rounded = hv.matmul(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.ScalarType.Float32,
            hv.ScalarType.BFloat16,
            accumulation_rounding=hv.AccumulationRounding.AfterProductAndSum,
        )
        full_precision = hv.matmul(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.ScalarType.Float32,
            hv.ScalarType.BFloat16,
            accumulation_rounding=hv.AccumulationRounding.FullPrecision,
        )

        expected_rounded = np.float32(0.0)
        for reduction in range(a.shape[1]):
            product = quantize_bfloat16(np.float32(a[0, reduction] * b[reduction, 0]))
            expected_rounded = quantize_bfloat16(np.float32(expected_rounded + product))
        expected_full_precision = matmul_float32(a, b)
        self.assertNotEqual(expected_rounded, expected_full_precision[0, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(rounded),
            np.asarray([[expected_rounded]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(full_precision),
            expected_full_precision,
        )

    def test_xfloat32_truncates_operand_mantissas(self):
        a = np.asarray([[1.234567, -2.345678]], dtype=np.float32)
        b = np.asarray([[3.456789], [4.567891]], dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            math_mode=hv.MathMode.XFloat32,
        )

        def xfloat32(values):
            bits = values.view(np.uint32).copy()
            bits &= np.uint32(0xFFFFE000)
            return bits.view(np.float32)

        expected = xfloat32(a) @ xfloat32(b)
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_mixed_compute_input_quantization(self):
        a = hv.from_numpy(
            np.asarray([[1.25, 2.5]], dtype=np.float32),
            hv.ScalarType.Float8E4M3,
        )
        b = hv.from_numpy(
            np.asarray([[2.0], [3.0]], dtype=np.float32),
            hv.ScalarType.Float8E5M2,
        )
        c = hv.from_numpy(np.asarray([[1.0]], dtype=np.float32))
        observed = reference_gemm(
            a,
            b,
            c,
            hv.ScalarType.Float16,
            hv.ScalarType.Float32,
            beta=1.0,
            compute_type_a=hv.ScalarType.Float4E2M1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(observed, np.float32), np.asarray([[9.0]], np.float32)
        )

        pre_scaled = reference_gemm(
            hv.from_numpy(np.asarray([[1.1]], dtype=np.float32), hv.ScalarType.Float16),
            hv.from_numpy(np.asarray([[1.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0.0]], dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float8E4M3,
            pre_quantization_scales_a=[
                hv.from_numpy(np.asarray([3.0], dtype=np.float32))
            ],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(pre_scaled), np.asarray([[3.25]], dtype=np.float32)
        )

        vector_pre_scaled = reference_gemm(
            hv.from_numpy(
                np.asarray([[1.1], [1.1]], dtype=np.float32),
                hv.ScalarType.Float16,
            ),
            hv.from_numpy(np.asarray([[1.0]], dtype=np.float32)),
            hv.from_numpy(np.zeros((2, 1), dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float8E4M3,
            pre_quantization_scales_a=[
                hv.from_numpy(np.asarray([3.0, 4.0], dtype=np.float32))
            ],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(vector_pre_scaled),
            np.asarray([[3.25], [4.5]], dtype=np.float32),
        )

        combined_pre_scaled = reference_gemm(
            hv.from_numpy(np.asarray([[0.3]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[1.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0.0]], dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float8E4M3,
            pre_quantization_scales_a=[
                hv.from_numpy(np.asarray([0.7], dtype=np.float32)),
                hv.from_numpy(np.asarray([0.6], dtype=np.float32)),
            ],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(combined_pre_scaled),
            np.asarray([[0.125]], dtype=np.float32),
        )

    def test_compute_input_quantization_matches_numpy(self):
        a = np.asarray(
            [[1.001, -2.003, 0.3333], [4.007, -0.499, 2.999]],
            dtype=np.float32,
        )
        b = np.asarray(
            [[0.999, -1.001], [2.005, 0.249], [-3.003, 4.004]],
            dtype=np.float32,
        )
        c = np.zeros((2, 2), dtype=np.float32)
        scale_a_row = np.asarray([1.25, 0.75], dtype=np.float32)
        scale_a_reduction = np.asarray([0.5, 1.5, 2.0], dtype=np.float32)
        scale_b_reduction = np.asarray([1.0, 0.75, 1.25], dtype=np.float32)
        scale_b_column = np.asarray([2.0, 0.5], dtype=np.float32)

        scaled_a = np.float32(
            np.float32(a * scale_a_row[:, None]) * scale_a_reduction[None, :]
        )
        scaled_b = np.float32(
            np.float32(b * scale_b_reduction[:, None]) * scale_b_column[None, :]
        )
        quantized_a = scaled_a.astype(np.float16).astype(np.float32)
        quantized_b = quantize_bfloat16(scaled_b)
        expected = matmul_float32(quantized_a, quantized_b)
        self.assertFalse(np.array_equal(expected, matmul_float32(scaled_a, scaled_b)))

        arguments = dict(
            compute_type_a=hv.ScalarType.Float16,
            compute_type_b=hv.ScalarType.BFloat16,
            pre_quantization_scales_a=[
                hv.from_numpy(scale_a_row[:, None]),
                hv.from_numpy(scale_a_reduction[None, :]),
            ],
            pre_quantization_scales_b=[
                hv.from_numpy(scale_b_reduction[:, None]),
                hv.from_numpy(scale_b_column),
            ],
        )
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            backend=hv.GemmBackend.Blocked,
            **arguments,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_gemm_output_scale_and_saturating_conversion(self):
        scaled_half = reference_gemm(
            hv.from_numpy(np.asarray([[0.3333]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[3.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0.0]], dtype=np.float32)),
            hv.ScalarType.Float16,
            hv.ScalarType.Float32,
            output_scale=0.1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(scaled_half),
            np.asarray([[np.float16(np.float32(0.3333 * 3.0 * 0.1))]]),
        )

        saturated_int8 = reference_gemm(
            hv.from_numpy(np.asarray([[63.75]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[2.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0]], dtype=np.int8)),
            hv.ScalarType.Int8,
            hv.ScalarType.Float32,
            output_conversion=hv.OutputConversion.SaturatingInt8,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(saturated_int8), np.asarray([[127]], dtype=np.int8)
        )

    def test_saturating_int8_rounding_matches_numpy(self):
        a = np.asarray(
            [
                [
                    -129.5,
                    -128.5,
                    -127.5,
                    -2.5,
                    -1.5,
                    1.5,
                    2.5,
                    126.5,
                    127.5,
                    128.5,
                ]
            ],
            dtype=np.float32,
        )
        b = np.eye(a.shape[1], dtype=np.float32)
        c = np.zeros_like(a, dtype=np.int8)
        expected = np.clip(np.rint(a), -128, 127).astype(np.int8)

        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Int8,
            hv.ScalarType.Float32,
            backend=hv.GemmBackend.Blocked,
            output_conversion=hv.OutputConversion.SaturatingInt8,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_block_scaled_gemm_matches_numpy(self):
        a = np.ones((1, 16), dtype=np.float32)
        b = np.ones((16, 1), dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        scale_a = np.asarray([[2.0, 4.0]], dtype=np.float32)
        scale_b = np.asarray([[8.0, 16.0]], dtype=np.float32)
        expected = np.asarray(
            [
                [
                    np.sum(a[:, :8] @ b[:8, :] * 2.0 * 8.0)
                    + np.sum(a[:, 8:] @ b[8:, :] * 4.0 * 16.0)
                ]
            ],
            dtype=np.float32,
        )

        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            backend=hv.GemmBackend.Blocked,
            block_scale_a=hv.from_numpy(scale_a),
            block_scale_b=hv.from_numpy(scale_b),
            block_size_a=8,
            block_size_b=8,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)
