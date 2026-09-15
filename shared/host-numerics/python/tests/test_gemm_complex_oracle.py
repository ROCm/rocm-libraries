# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _gemm_oracle_support import *  # noqa: F403


class GemmComplexOracleTests(unittest.TestCase):  # noqa: F405
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
        result = reference_gemm_into(
            operand_a,
            operand_b,
            initial_tensor,
            output,
            accumulator_type=hv.ScalarType.ComplexFloat32,
            alpha=complex(alpha),
            beta=complex(beta),
            conjugate_a=True,
            output_selection=hv.OutputSelection.explicit_indices(selected),
            backend=hv.GemmBackend.Blocked,
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
        self.assertIsNone(result)
