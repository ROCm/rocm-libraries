# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class ReferenceOperationsTests(unittest.TestCase):  # noqa: F405
    def test_reference_softmax_matches_numpy(self):
        source = np.asarray(
            [
                [[1.0, -2.0], [3.25, 4.0], [-1.5, 0.5]],
                [[100.0, 2.0], [99.0, -3.0], [98.0, 1.0]],
            ],
            dtype=np.float32,
        )
        input_tensor = hv.from_numpy(source, hv.ScalarType.Float16)
        result = hv.reference_softmax(input_tensor, axis=1)

        quantized = source.astype(np.float16).astype(np.float32)
        expected = np.empty_like(quantized)
        for batch in range(quantized.shape[0]):
            for column in range(quantized.shape[2]):
                maximum = np.max(quantized[batch, :, column])
                exponentials = np.empty(quantized.shape[1], dtype=np.float32)
                total = np.float32(0.0)
                for row in range(quantized.shape[1]):
                    value = np.float32(
                        np.exp(np.float32(quantized[batch, row, column] - maximum))
                    )
                    exponentials[row] = value
                    total = np.float32(total + value)
                for row in range(quantized.shape[1]):
                    expected[batch, row, column] = np.float32(exponentials[row] / total)

        np.testing.assert_allclose(hv.to_numpy(result), expected, rtol=1e-6, atol=1e-7)
        with self.assertRaises(IndexError):
            hv.reference_softmax(input_tensor, axis=source.ndim)

        padded_input = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            np.asarray([1.0, 2.0, -99.0, 3.0, 4.0], dtype=np.float32).tobytes(),
            strides=[3, 1],
        )
        contiguous = hv.reference_softmax(padded_input, axis=1)
        self.assertEqual(contiguous.strides, [2, 1])
        self.assertEqual(contiguous.offset, 0)

    def test_reference_layer_norm_matches_numpy(self):
        source = np.asarray(
            [
                [[1.0, -2.0], [3.25, 4.0], [-1.5, 0.5]],
                [[5.0, 2.0], [7.0, -3.0], [8.5, 1.0]],
            ],
            dtype=np.float32,
        )
        gamma_source = np.asarray([1.0, 0.5, -2.0], dtype=np.float32)
        beta_source = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
        input_tensor = hv.from_numpy(source, hv.ScalarType.Float16)
        gamma = hv.from_numpy(gamma_source, hv.ScalarType.Float16)
        beta = hv.from_numpy(beta_source, hv.ScalarType.BFloat16)
        epsilon = np.float32(1e-5)
        result = hv.reference_layer_norm(
            input_tensor,
            axis=1,
            epsilon=float(epsilon),
            gamma=gamma,
            beta=beta,
        )

        quantized = source.astype(np.float16).astype(np.float32)
        gamma_values = gamma_source.astype(np.float16).astype(np.float32)
        beta_values = quantize_bfloat16(beta_source)
        expected = np.empty_like(quantized)
        expected_mean = np.empty((2, 2), dtype=np.float32)
        expected_inverse = np.empty((2, 2), dtype=np.float32)
        for batch in range(quantized.shape[0]):
            for column in range(quantized.shape[2]):
                average = np.float32(0.0)
                second_moment = np.float32(0.0)
                for row in range(quantized.shape[1]):
                    value = quantized[batch, row, column]
                    delta = np.float32(value - average)
                    average = np.float32(average + delta / np.float32(row + 1))
                    delta_after_update = np.float32(value - average)
                    second_moment = np.float32(
                        second_moment + delta * delta_after_update
                    )
                inverse = np.float32(
                    1.0
                    / np.sqrt(
                        np.float32(
                            second_moment / np.float32(quantized.shape[1]) + epsilon
                        )
                    )
                )
                expected_mean[batch, column] = average
                expected_inverse[batch, column] = inverse
                for row in range(quantized.shape[1]):
                    normalized = np.float32(
                        np.float32(quantized[batch, row, column] - average) * inverse
                    )
                    expected[batch, row, column] = np.float32(
                        np.float32(normalized * gamma_values[row]) + beta_values[row]
                    )

        np.testing.assert_allclose(
            hv.to_numpy(result.output), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            hv.to_numpy(result.mean), expected_mean, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            hv.to_numpy(result.inverse_variance),
            expected_inverse,
            rtol=1e-6,
            atol=1e-6,
        )

        padded_input = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            np.asarray([1.0, 2.0, -99.0, 3.0, 4.0], dtype=np.float32).tobytes(),
            strides=[3, 1],
        )
        contiguous = hv.reference_layer_norm(padded_input, axis=1)
        self.assertEqual(contiguous.output.strides, [2, 1])
        self.assertEqual(contiguous.output.offset, 0)
        self.assertEqual(contiguous.mean.strides, [1])
        self.assertEqual(contiguous.inverse_variance.strides, [1])

    def test_reference_epilogue_matches_numpy(self):
        values = np.asarray([[-2.0, 1.0], [3.0, -4.0]], dtype=np.float32)
        bias = np.asarray([1.0, 2.0], dtype=np.float32)
        result = hv.reference_epilogue(
            hv.from_numpy(values),
            hv.ScalarType.Float16,
            hv.ScalarType.Float32,
            bias=hv.from_numpy(bias[:, None]),
            activation=hv.Activation.Relu,
            auxiliary_output_type=hv.ScalarType.BFloat16,
            output_scale=2.0,
            auxiliary_scale=3.0,
            include_raw_output=True,
            include_amax=True,
        )
        pre_activation = values + bias[:, None]
        activated = np.maximum(pre_activation, 0.0)
        np.testing.assert_array_equal(
            hv.to_numpy(result.output, np.float32), activated * 2.0
        )
        np.testing.assert_array_equal(hv.to_numpy(result.raw_output), activated * 2.0)
        np.testing.assert_array_equal(
            hv.to_numpy(result.auxiliary_output), pre_activation * 3.0
        )
        np.testing.assert_array_equal(
            hv.to_numpy(result.amax), np.asarray([5.0], dtype=np.float32)
        )

        gate = np.asarray([[0.5, 2.0], [-1.0, 0.25]], dtype=np.float32)
        gated = hv.reference_epilogue(
            hv.from_numpy(values),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            gate_residual=hv.from_numpy(gate),
            output_scale=2.0,
            include_raw_output=True,
        )
        raw = values * 2.0
        np.testing.assert_array_equal(hv.to_numpy(gated.raw_output), raw)
        np.testing.assert_array_equal(hv.to_numpy(gated.output), gate * raw + gate)

        selected = hv.reference_epilogue(
            hv.from_numpy(values),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            activation=hv.Activation.Relu,
            include_raw_output=True,
            output_selection=hv.OutputSelection.explicit_indices([1, 2]),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(selected.output),
            np.asarray([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(selected.raw_output),
            np.asarray([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32),
        )
        int8_values = np.asarray([[-200.0, -128.5], [126.5, 300.0]], dtype=np.float32)
        saturated = hv.reference_epilogue(
            hv.from_numpy(int8_values),
            hv.ScalarType.Int8,
            hv.ScalarType.Float32,
            output_conversion=hv.OutputConversion.SaturatingInt8,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(saturated.output),
            np.clip(np.rint(int8_values), -128, 127).astype(np.int8),
        )

    def test_reference_gradient_epilogue_matches_numpy(self):
        gradient = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        activation_input = np.asarray([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float32)
        result = hv.reference_epilogue(
            hv.from_numpy(gradient),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            activation=hv.Activation.Relu,
            activation_application=hv.ActivationApplication.Gradient,
            auxiliary_input=hv.from_numpy(activation_input),
        )
        expected = gradient * (activation_input > 0)
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)

        gelu = hv.reference_epilogue(
            hv.from_numpy(gradient),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            activation=hv.Activation.Gelu,
            activation_application=hv.ActivationApplication.Gradient,
            auxiliary_input=hv.from_numpy(activation_input),
        )
        coefficient0 = np.float32(0.7978845608028654)
        coefficient1 = np.float32(0.044715)
        activation_argument = (
            coefficient0
            * activation_input
            * (1.0 + coefficient1 * activation_input * activation_input)
        )
        gelu_derivative = 0.5 * (1.0 + np.tanh(activation_argument))
        gelu_derivative += (
            0.5
            * activation_input
            * (1.0 - np.tanh(activation_argument) ** 2)
            * coefficient0
            * (1.0 + 3.0 * coefficient1 * activation_input * activation_input)
        )
        np.testing.assert_allclose(
            hv.to_numpy(gelu.output),
            gradient * gelu_derivative,
            rtol=5e-6,
            atol=2e-5,
        )

    def test_configured_activation_family_matches_numpy(self):
        values = np.asarray([[-2.0, -0.5, 0.0], [0.5, 1.0, 2.0]], dtype=np.float32)
        parameter0 = np.float32(0.5)
        parameter1 = np.float32(1.5)

        def gelu(array):
            coefficient0 = np.float32(0.7978845608028654)
            coefficient1 = np.float32(0.044715)
            return (
                np.float32(0.5)
                * array
                * (
                    np.float32(1.0)
                    + np.tanh(
                        coefficient0
                        * array
                        * (np.float32(1.0) + coefficient1 * array * array)
                    )
                )
            )

        def gelu_derivative(array):
            coefficient0 = np.float32(0.0535161)
            coefficient1 = np.float32(0.398942)
            coefficient2 = np.float32(0.0356774)
            coefficient3 = np.float32(0.797885)
            cube = array * array * array
            first = coefficient0 * cube + coefficient1 * array
            second = coefficient2 * cube + coefficient3 * array
            return (
                np.float32(0.5) * np.tanh(second)
                + first * (np.float32(4.0) / (np.exp(-second) + np.exp(second)) ** 2)
                + np.float32(0.5)
            )

        sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-values))
        swish_sigmoid = np.float32(1.0) / (
            np.float32(1.0) + np.exp(-parameter0 * values)
        )
        cases = {
            hv.Activation.Absolute: np.abs(values),
            hv.Activation.ClippedRelu: np.where(
                values > parameter0,
                np.minimum(values, parameter1),
                np.minimum(np.float32(0.0), parameter1),
            ),
            hv.Activation.Gelu: gelu(values),
            hv.Activation.GeluDerivative: gelu_derivative(values),
            hv.Activation.GeluScaling: gelu(values) * parameter0,
            hv.Activation.LeakyRelu: np.where(values > 0, values, values * parameter0),
            hv.Activation.Relu: np.maximum(values, 0),
            hv.Activation.ReluDerivative: (values > 0).astype(np.float32),
            hv.Activation.Sigmoid: sigmoid,
            hv.Activation.Tanh: np.tanh(values * parameter0) * parameter1,
            hv.Activation.Silu: values * sigmoid,
            hv.Activation.Swish: values * swish_sigmoid,
            hv.Activation.Clamp: np.maximum(parameter0, np.minimum(values, parameter1)),
        }

        for activation, expected in cases.items():
            with self.subTest(activation=activation):
                result = hv.reference_epilogue(
                    hv.from_numpy(values),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    activation=activation,
                    activation_parameter0=float(parameter0),
                    activation_parameter1=float(parameter1),
                )
                np.testing.assert_allclose(
                    hv.to_numpy(result.output),
                    expected,
                    rtol=5e-6,
                    atol=2e-5,
                )

        gradient = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        hyperbolic_tangent = np.tanh(values * parameter0)
        gradient_factors = {
            hv.Activation.Absolute: np.sign(values),
            hv.Activation.ClippedRelu: (
                (values > parameter0) & (values < parameter1)
            ).astype(np.float32),
            hv.Activation.Gelu: gelu_derivative(values),
            hv.Activation.GeluScaling: gelu_derivative(values) * parameter0,
            hv.Activation.LeakyRelu: np.where(values > 0, np.float32(1.0), parameter0),
            hv.Activation.Relu: (values > 0).astype(np.float32),
            hv.Activation.Sigmoid: sigmoid * (np.float32(1.0) - sigmoid),
            hv.Activation.Tanh: parameter0
            * parameter1
            * (np.float32(1.0) - hyperbolic_tangent * hyperbolic_tangent),
            hv.Activation.Silu: sigmoid
            + values * sigmoid * (np.float32(1.0) - sigmoid),
            hv.Activation.Swish: swish_sigmoid
            + parameter0 * values * swish_sigmoid * (np.float32(1.0) - swish_sigmoid),
            hv.Activation.Clamp: ((values > parameter0) & (values < parameter1)).astype(
                np.float32
            ),
        }
        for activation, factor in gradient_factors.items():
            with self.subTest(gradient_activation=activation):
                result = hv.reference_epilogue(
                    hv.from_numpy(gradient),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    activation=activation,
                    activation_application=hv.ActivationApplication.Gradient,
                    auxiliary_input=hv.from_numpy(values),
                    activation_parameter0=float(parameter0),
                    activation_parameter1=float(parameter1),
                )
                np.testing.assert_allclose(
                    hv.to_numpy(result.output),
                    gradient * factor,
                    rtol=5e-6,
                    atol=2e-5,
                )

    def test_float64_activation_retains_float64_precision(self):
        values = np.asarray([[1.0000000001]], dtype=np.float64)
        result = hv.reference_epilogue(
            hv.from_numpy(values),
            hv.ScalarType.Float64,
            hv.ScalarType.Float64,
            activation=hv.Activation.Sigmoid,
        )
        expected = 1.0 / (1.0 + np.exp(-values))
        np.testing.assert_allclose(
            hv.to_numpy(result.output), expected, rtol=0.0, atol=1e-15
        )

    def test_reference_sum_matches_numpy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        observed = hv.reference_sum(
            hv.from_numpy(values),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            [0, 2],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(observed), np.sum(values, axis=(0, 2), dtype=np.float32)
        )

        complex_values = values.astype(np.complex64) * np.complex64(1.0 + 2.0j)
        complex_observed = hv.reference_sum(
            hv.from_numpy(complex_values),
            hv.ScalarType.ComplexFloat32,
            hv.ScalarType.ComplexFloat32,
            [1],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(complex_observed),
            np.sum(complex_values, axis=1, dtype=np.complex64),
        )

        integer_values = np.arange(12, dtype=np.int8).reshape(3, 4)
        integer_observed = hv.reference_sum(
            hv.from_numpy(integer_values),
            hv.ScalarType.Int32,
            hv.ScalarType.Int32,
            [1],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(integer_observed),
            np.sum(integer_values, axis=1, dtype=np.int32),
        )

        wrapping_values = np.asarray([np.iinfo(np.int32).max, 1], dtype=np.int32)
        wrapping_observed = hv.reference_sum(
            hv.from_numpy(wrapping_values),
            hv.ScalarType.Int32,
            hv.ScalarType.Int32,
            [0],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(wrapping_observed),
            np.sum(wrapping_values, dtype=np.int32),
        )

    def test_reference_maximum_absolute_matches_numpy(self):
        values = np.asarray(
            [[-1.5, 2.25, np.nan], [-7.0, 3.5, 0.25]],
            dtype=np.float32,
        )
        observed = hv.reference_maximum_absolute(
            hv.from_numpy(values),
            hv.ScalarType.Float16,
            hv.ScalarType.Float32,
        )
        expected = np.asarray(
            np.max(np.where(np.isnan(values), 0.0, np.abs(values))),
            dtype=np.float16,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

        all_nan = np.full((2, 3), np.nan, dtype=np.float32)
        all_nan_observed = hv.reference_maximum_absolute(
            hv.from_numpy(all_nan),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(all_nan_observed),
            np.asarray(0.0, dtype=np.float32),
        )
