# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class GemmApiTests(unittest.TestCase):  # noqa: F405
    def test_gemm_accepts_owned_inputs_and_scaling_tensors(self):
        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pre_scale_a = np.asarray([2.0, 3.0], dtype=np.float32)
        pre_scale_b = np.asarray([0.5, 2.0], dtype=np.float32)
        scale_alpha = np.asarray([1.0, 2.0], dtype=np.float32)
        scale_a = np.asarray([2.0, 3.0], dtype=np.float32)
        scale_b = np.asarray([4.0, 5.0], dtype=np.float32)
        bias = np.asarray([1.0, -2.0], dtype=np.float32)

        scaled_a = np.float32(a_values * pre_scale_a[:, None])
        scaled_b = np.float32(b_values * pre_scale_b[None, :])
        accumulation = matmul_float32(scaled_a, scaled_b)
        effective_alpha = np.float32(
            np.float32(
                np.float32(np.float32(0.5) * scale_a[:, None]) * scale_b[None, :]
            )
            * scale_alpha[:, None]
        )
        combined = np.float32(
            np.float32(effective_alpha * accumulation)
            + np.float32(np.float32(-1.0) * c_values)
        )
        expected = np.float32(np.float32(combined + bias[:, None]) * np.float32(0.25))

        result = reference_gemm(
            hv.from_numpy(a_values),
            hv.from_numpy(b_values),
            hv.from_numpy(c_values),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=0.5,
            beta=-1.0,
            pre_quantization_scales_a=[hv.from_numpy(pre_scale_a[:, None])],
            pre_quantization_scales_b=[hv.from_numpy(pre_scale_b)],
            scale_alpha=hv.from_numpy(scale_alpha[:, None]),
            scale_a=hv.from_numpy(scale_a[:, None]),
            scale_b=hv.from_numpy(scale_b),
            bias=hv.from_numpy(bias[:, None]),
            output_scale=0.25,
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(hv.to_numpy(result), expected)

    def test_gemm_api_requires_explicit_c(self):
        self.assertFalse(hasattr(hv, "GemmOptions"))
        self.assertFalse(hasattr(hv, "GemmEpilogue"))
        self.assertFalse(hasattr(hv.GemmBackend, "Pointwise"))
        self.assertFalse(hasattr(hv.GemmBackend, "Blas"))
        self.assertFalse(hasattr(hv.GemmBackend, "Mixed"))

        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0], [6.0]], dtype=np.float32)
        operand_a = hv.from_numpy(a_values)
        operand_b = hv.from_numpy(b_values)
        np.testing.assert_array_equal(
            hv.to_numpy(
                reference_gemm(
                    operand_a,
                    operand_b,
                    hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 1])),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                )
            ),
            a_values @ b_values,
        )

        with self.assertRaises(TypeError):
            reference_gemm(operand_a, operand_b)

    def test_gemm_coefficients_accept_native_and_rank_zero_values(self):
        a = hv.from_numpy(np.asarray([[2.0]], dtype=np.float32))
        b = hv.from_numpy(np.asarray([[3.0]], dtype=np.float32))
        c = hv.from_numpy(np.asarray([[5.0]], dtype=np.float32))
        alpha = hv.from_numpy(np.asarray(4.0, dtype=np.float32))

        result = reference_gemm(
            a,
            b,
            c,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=alpha,
            beta=2.0,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(result), np.asarray([[34.0]], dtype=np.float32)
        )

        with self.assertRaisesRegex(ValueError, "rank-zero"):
            reference_gemm(
                a,
                b,
                c,
                hv.ScalarType.Float32,
                hv.ScalarType.Float32,
                alpha=a,
            )

    def test_gemm_options_apply_operand_quantization_and_block_scales(self):
        operand_a = hv.from_numpy(np.full((1, 8), 1.5, dtype=np.float32))
        operand_b = hv.from_numpy(np.full((8, 1), 2.0, dtype=np.float32))
        result = reference_gemm(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([1, 1])),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float16,
            compute_type_b=hv.ScalarType.BFloat16,
            pre_quantization_scales_a=[
                hv.from_numpy(np.asarray([2.0], dtype=np.float32))
            ],
            pre_quantization_scales_b=[
                hv.from_numpy(np.asarray([0.5], dtype=np.float32))
            ],
            block_scale_a=hv.from_numpy(np.asarray([[2.0, 4.0]], dtype=np.float32)),
            block_scale_b=hv.from_numpy(np.asarray([[3.0, 5.0]], dtype=np.float32)),
            block_size_a=4,
            block_size_b=4,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(result), np.asarray([[312.0]], dtype=np.float32)
        )

    def test_gemm_options_conjugate_operands(self):
        a_values = np.asarray(
            [[1.0 + 2.0j, 3.0 - 4.0j], [-2.0 + 0.5j, 1.5 + 3.0j]],
            dtype=np.complex64,
        )
        b_values = np.asarray([[2.0 - 1.0j], [0.5 + 2.0j]], dtype=np.complex64)
        operand_a = hv.from_numpy(a_values)
        operand_b = hv.from_numpy(b_values)
        result = reference_gemm(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.ComplexFloat32, hv.Shape([2, 1])),
            hv.ScalarType.ComplexFloat32,
            hv.ScalarType.ComplexFloat32,
            conjugate_a=True,
        )
        np.testing.assert_allclose(
            hv.to_numpy(result),
            np.conjugate(a_values) @ b_values,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gemm_into_writes_affine_output_for_blocked_execution(self):
        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
        output_layout = hv.Layout(hv.Shape([2, 3]), [9, 2], 1)
        operand_a = hv.from_numpy(a_values)
        operand_b = hv.from_numpy(b_values)
        c = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3]))
        result = hv.Tensor(hv.ScalarType.Float32, output_layout)
        reference_gemm_into(
            operand_a,
            operand_b,
            c,
            result,
            accumulator_type=hv.ScalarType.Float32,
            backend=hv.GemmBackend.Blocked,
        )
        expected = a_values @ b_values
        np.testing.assert_array_equal(hv.to_numpy(result), expected)
        self.assertEqual(result.strides, [9, 2])
        self.assertEqual(result.offset, 1)

        automatic = hv.Tensor(hv.ScalarType.Float32, output_layout)
        reference_gemm_into(
            operand_a,
            operand_b,
            c,
            automatic,
            accumulator_type=hv.ScalarType.Float32,
            backend=hv.GemmBackend.Automatic,
        )
        np.testing.assert_array_equal(hv.to_numpy(automatic), expected)

        storage = np.frombuffer(result.storage, dtype=np.float32)
        expected_storage = np.zeros(15, dtype=np.float32)
        expected_storage[[1, 3, 5, 10, 12, 14]] = expected.reshape(-1)
        np.testing.assert_array_equal(storage, expected_storage)

    def test_selected_output_gemm(self):
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c = np.zeros((2, 2), dtype=np.float32)
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            output_selection=hv.OutputSelection.explicit_indices([0, 3]),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(observed),
            np.asarray([[19.0, 0.0], [0.0, 50.0]], dtype=np.float32),
        )
        blocked = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            output_selection=hv.OutputSelection.explicit_indices([0, 3]),
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(blocked),
            np.asarray([[19.0, 0.0], [0.0, 50.0]], dtype=np.float32),
        )
        self.assertEqual(
            hv.OutputSelection.prime_stride(10, 10, 3).indices(10),
            [0, 3, 6, 9],
        )
