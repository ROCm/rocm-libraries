# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _gemm_oracle_support import *  # noqa: F403


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
        observed = reference_gemm(
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

    def test_selected_float_results_match_full_result_and_numpy(self):
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
        full = hv.to_numpy(
            reference_gemm(
                *tensors,
                hv.ScalarType.Float32,
                hv.ScalarType.Float32,
                alpha=float(alpha),
                beta=float(beta),
                backend=hv.GemmBackend.Blocked,
            )
        )
        partial = hv.to_numpy(
            reference_gemm(
                *tensors,
                hv.ScalarType.Float32,
                hv.ScalarType.Float32,
                alpha=float(alpha),
                beta=float(beta),
                output_selection=hv.OutputSelection.explicit_indices(selected),
                backend=hv.GemmBackend.Blocked,
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
