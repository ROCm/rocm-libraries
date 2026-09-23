# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class GemmFloatingTests(unittest.TestCase):  # noqa: F405
    def test_matmul_matches_numpy_and_supports_empty_reduction(self):
        a_values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b_values = np.asarray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
        a = hv.from_numpy(a_values)
        b = hv.from_numpy(b_values)

        np.testing.assert_array_equal(hv.to_numpy(hv.matmul(a, b)), a_values @ b_values)

        output = hv.from_numpy(np.full((2, 2), -99.0, dtype=np.float32))
        self.assertIsNone(hv.matmul_into(a, b, output))
        np.testing.assert_array_equal(hv.to_numpy(output), a_values @ b_values)

        empty_a = hv.from_numpy(np.empty((2, 0), dtype=np.float32))
        empty_b = hv.from_numpy(np.empty((0, 3), dtype=np.float32))
        empty_product = hv.matmul(empty_a, empty_b)
        self.assertEqual(empty_product.shape, [2, 3])
        np.testing.assert_array_equal(
            hv.to_numpy(empty_product), np.zeros((2, 3), dtype=np.float32)
        )

    def test_float32_gemm_matches_numpy(self):
        a = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
        b = np.arange(20, dtype=np.float32).reshape(5, 4) - 7
        c = np.arange(12, dtype=np.float32).reshape(3, 4)
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
        )
        expected = 2.0 * (a @ b) - c
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)
        blocked = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(hv.to_numpy(blocked), expected)

    def test_zero_gemm_scalars_suppress_non_finite_operands(self):
        finite_c = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        finite_a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        finite_b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        alpha_zero = reference_gemm(
            hv.from_numpy(np.full((2, 2), np.nan, dtype=np.float32)),
            hv.from_numpy(np.full((2, 2), np.inf, dtype=np.float32)),
            hv.from_numpy(finite_c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=0.0,
            beta=2.0,
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(hv.to_numpy(alpha_zero), 2.0 * finite_c)

        beta_zero = reference_gemm(
            hv.from_numpy(finite_a),
            hv.from_numpy(finite_b),
            hv.from_numpy(np.full((2, 2), np.inf, dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=1.0,
            beta=0.0,
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(hv.to_numpy(beta_zero), finite_a @ finite_b)

    def test_gemm_finalization_matches_numpy(self):
        a = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [-7.0, 8.0]], dtype=np.float32)
        c = np.asarray([[2.0, -3.0], [4.0, 5.0]], dtype=np.float32)
        alpha = np.float32(0.75)
        beta = np.float32(-0.5)
        scale_c = np.float32(2.0)
        output_scale = np.float32(1.25)

        accumulation = matmul_float32(a, b)
        combined = np.float32(
            np.float32(alpha * accumulation) + np.float32(beta * scale_c * c)
        )
        expected = np.float32(np.maximum(combined, np.float32(0.0)) * output_scale)

        product = hv.matmul(hv.from_numpy(a), hv.from_numpy(b), hv.ScalarType.Float32)
        combined_tensor = hv.add(
            hv.multiply(product, float(alpha)),
            hv.multiply(hv.from_numpy(c), float(beta * scale_c)),
        )
        observed = hv.reference_epilogue(
            combined_tensor,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            activation=hv.Activation.Relu,
            output_scale=float(output_scale),
        ).output
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_float64_gemm_matches_numpy(self):
        a = np.asarray([[0.25, -1.5], [2.0, 3.25]], dtype=np.float64)
        b = np.asarray([[4.0, 0.5], [-2.0, 1.25]], dtype=np.float64)
        c = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float64,
            hv.ScalarType.Float64,
            alpha=1.25,
            beta=0.5,
        )
        expected = 1.25 * (a @ b) + 0.5 * c
        np.testing.assert_allclose(
            hv.to_numpy(observed), expected, rtol=1e-15, atol=0.0
        )

    def test_complex_gemm_matches_numpy(self):
        a = np.asarray(
            [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0 + 0.5j, 4.0 + 3.0j]],
            dtype=np.complex64,
        )
        b = np.asarray([[2.0 - 1.0j], [0.5 + 3.0j]], dtype=np.complex64)
        c = np.asarray([[1.0j], [2.0 - 1.0j]], dtype=np.complex64)
        alpha = 0.5 + 0.25j
        beta = -1.0 + 0.5j
        observed = reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.ComplexFloat32,
            hv.ScalarType.ComplexFloat32,
            alpha=alpha,
            beta=beta,
        )
        expected = alpha * (a @ b) + beta * c
        np.testing.assert_allclose(
            hv.to_numpy(observed), expected, rtol=1e-6, atol=1e-6
        )
