# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import unittest

import numpy as np

import roc_host_validation as hv


def pack_bits(values, bits):
    result = bytearray((len(values) * bits + 7) // 8)
    for index, value in enumerate(values):
        bit_offset = index * bits
        for bit in range(bits):
            if (value >> bit) & 1:
                absolute = bit_offset + bit
                result[absolute // 8] |= 1 << (absolute % 8)
    return bytes(result)


def unpack_bits(storage, count, bits):
    result = []
    for index in range(count):
        bit_offset = index * bits
        value = 0
        for bit in range(bits):
            absolute = bit_offset + bit
            value |= ((storage[absolute // 8] >> (absolute % 8)) & 1) << bit
        result.append(value)
    return result


def decode_binary(raw, exponent_bits, mantissa_bits, bias, total_bits, signed=True):
    sign_mask = 1 << (total_bits - 1) if signed else 0
    negative = signed and bool(raw & sign_mask)
    magnitude = raw & (sign_mask - 1) if signed else raw
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    exponent = (magnitude >> mantissa_bits) & exponent_mask
    mantissa = magnitude & mantissa_mask
    fraction = mantissa / (1 << mantissa_bits)
    if exponent == 0:
        value = math.ldexp(fraction, 1 - bias)
    else:
        value = math.ldexp(1.0 + fraction, exponent - bias)
    return -value if negative else value


FORMATS = {
    hv.ScalarType.Float4E2M1: (4, 2, 1, 1, "finite"),
    hv.ScalarType.Float6E2M3: (6, 2, 3, 1, "finite"),
    hv.ScalarType.Float6E3M2: (6, 3, 2, 3, "finite"),
    hv.ScalarType.Float8E4M3: (8, 4, 3, 7, "e4m3"),
    hv.ScalarType.Float8E5M2: (8, 5, 2, 15, "e5m2"),
    hv.ScalarType.Float8E4M3Fnuz: (8, 4, 3, 8, "fnuz"),
    hv.ScalarType.Float8E5M2Fnuz: (8, 5, 2, 16, "fnuz"),
    hv.ScalarType.E5M3: (8, 5, 3, 15, "e5m3_scale"),
}


def expected_value(scalar_type, raw):
    total_bits, exponent_bits, mantissa_bits, bias, kind = FORMATS[scalar_type]
    if kind == "e4m3" and (raw & 0x7F) == 0x7F:
        return math.nan
    if kind == "e5m2" and (raw & 0x7F) >= 0x7C:
        if (raw & 0x7F) == 0x7C:
            return -math.inf if raw & 0x80 else math.inf
        return math.nan
    if kind == "fnuz" and raw == 0x80:
        return math.nan
    if kind == "e5m3_scale" and raw == 0xFF:
        return math.nan
    return decode_binary(
        raw,
        exponent_bits,
        mantissa_bits,
        bias,
        total_bits,
        signed=kind != "e5m3_scale",
    )


class CodecTests(unittest.TestCase):
    def assert_values_equal(self, observed, expected):
        observed = np.asarray(observed)
        expected = np.asarray(expected)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(observed), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(observed), np.isneginf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(observed[finite], expected[finite])
        zeros = finite & (expected == 0)
        np.testing.assert_array_equal(
            np.signbit(observed[zeros]), np.signbit(expected[zeros])
        )

    def test_exhaustive_low_precision_codecs(self):
        for scalar_type, format_info in FORMATS.items():
            with self.subTest(scalar_type=scalar_type):
                bits = format_info[0]
                raw = list(range(1 << bits))
                tensor = hv.Tensor.from_storage(
                    scalar_type, [len(raw)], pack_bits(raw, bits)
                )
                observed = hv.to_numpy(tensor, np.float32)
                expected = np.asarray(
                    [expected_value(scalar_type, value) for value in raw],
                    dtype=np.float32,
                )
                self.assert_values_equal(observed, expected)

                encodable = ~np.isnan(expected)
                encoded = hv.from_numpy(expected[encodable], scalar_type)
                round_trip = unpack_bits(
                    encoded.storage, int(encodable.sum()), bits
                )
                np.testing.assert_array_equal(
                    round_trip, np.asarray(raw, dtype=np.uint32)[encodable]
                )

    def test_exhaustive_float16_decode(self):
        raw = np.arange(1 << 16, dtype="<u2")
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float16, [raw.size], raw.tobytes()
        )
        observed = hv.to_numpy(tensor, np.float32)
        expected = raw.view("<f2").astype(np.float32)
        self.assert_values_equal(observed, expected)

    def test_exhaustive_bfloat16_decode_and_finite_round_trip(self):
        raw = np.arange(1 << 16, dtype="<u2")
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.BFloat16, [raw.size], raw.tobytes()
        )
        observed = hv.to_numpy(tensor, np.float32)
        expected_bits = raw.astype("<u4") << 16
        expected = expected_bits.view("<f4")
        self.assert_values_equal(observed, expected)

        encoded = hv.from_numpy(expected, hv.ScalarType.BFloat16)
        encoded_raw = np.frombuffer(encoded.storage, dtype="<u2")
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(encoded_raw[finite], raw[finite])

    def test_e8m0_has_no_zero_encoding(self):
        raw = bytes([0, 1, 127, 128, 254, 255])
        tensor = hv.Tensor.from_storage(hv.ScalarType.E8M0, [len(raw)], raw)
        observed = hv.to_numpy(tensor, np.float32)
        expected = np.asarray(
            [
                math.ldexp(1.0, -127),
                math.ldexp(1.0, -126),
                1.0,
                2.0,
                math.ldexp(1.0, 127),
                math.nan,
            ],
            dtype=np.float32,
        )
        self.assert_values_equal(observed, expected)


class TensorAndGemmTests(unittest.TestCase):
    def test_numpy_round_trip(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = hv.from_numpy(values)
        self.assertEqual(tensor.type, hv.ScalarType.Float32)
        self.assertEqual(tensor.shape, [3, 4])
        np.testing.assert_array_equal(hv.to_numpy(tensor), values)

    def test_affine_layout_decode(self):
        storage = np.asarray(
            [-99.0, 1.0, 2.0, -99.0, 3.0, 4.0, -99.0],
            dtype=np.float32,
        )
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            storage.tobytes(),
            strides=[1, 3],
            offset=1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(tensor),
            np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )

    def test_generation_and_comparison(self):
        first = hv.Tensor(
            hv.ScalarType.Float32, hv.Shape([2, 3])
        )
        second = hv.Tensor(
            hv.ScalarType.Float32, hv.Shape([2, 3])
        )
        hv.fill(
            first,
            hv.DataPattern.UniformInteger,
            seed=17,
            parameter0=-3,
            parameter1=3,
        )
        hv.fill(
            second,
            hv.DataPattern.UniformInteger,
            seed=17,
            parameter0=-3,
            parameter1=3,
        )
        self.assertTrue(hv.compare(first, second).passed)

        changed_values = hv.to_numpy(second).copy()
        changed_values[1, 2] += 1
        changed = hv.from_numpy(changed_values)
        options = hv.ComparisonOptions()
        options.max_reported_mismatches = 2
        result = hv.compare(changed, first, options)
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)
        self.assertEqual(result.reported_mismatches[0].index, 5)

    def test_indexed_generation_matches_numpy(self):
        options = hv.GenerationOptions()
        options.real.pattern = hv.GenerationPattern.SerialIndex
        serial = hv.generate_tensor(hv.ScalarType.Float32, [2, 3], options)
        np.testing.assert_array_equal(
            hv.to_numpy(serial),
            np.arange(6, dtype=np.float32).reshape((2, 3), order="F"),
        )

        options.real.pattern = hv.GenerationPattern.Sine
        options.imaginary.pattern = hv.GenerationPattern.Cosine
        complex_values = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32, [2, 3], options
        )
        indices = np.arange(6, dtype=np.float32).reshape((2, 3), order="F")
        np.testing.assert_allclose(
            hv.to_numpy(complex_values),
            np.sin(indices) + 1j * np.cos(indices),
            rtol=1e-6,
            atol=1e-6,
        )

        options.imaginary.pattern = hv.GenerationPattern.Zero
        options.real.pattern = hv.GenerationPattern.Identity
        identity = hv.generate_tensor(hv.ScalarType.Float32, [3, 4], options)
        np.testing.assert_array_equal(hv.to_numpy(identity), np.eye(3, 4, dtype=np.float32))

        options.real.pattern = hv.GenerationPattern.UniformInteger
        options.real.parameter0 = -3
        options.real.parameter1 = 3
        options.seed = 19
        options.real.stream = 0
        random_first = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [4, 4], options)
        )
        random_repeat = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [4, 4], options)
        )
        np.testing.assert_array_equal(random_first, random_repeat)
        options.real.stream = 1
        random_other_stream = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [4, 4], options)
        )
        self.assertFalse(np.array_equal(random_first, random_other_stream))
        self.assertTrue(np.all((-3 <= random_first) & (random_first <= 3)))

    def test_float32_gemm_matches_numpy(self):
        a = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
        b = np.arange(20, dtype=np.float32).reshape(5, 4) - 7
        c = np.arange(12, dtype=np.float32).reshape(3, 4)
        observed = hv.reference_gemm(
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
        tiled = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
            backend=hv.GemmBackend.Tiled,
        )
        np.testing.assert_array_equal(hv.to_numpy(tiled), expected)

    def test_float64_gemm_matches_numpy(self):
        a = np.asarray([[0.25, -1.5], [2.0, 3.25]], dtype=np.float64)
        b = np.asarray([[4.0, 0.5], [-2.0, 1.25]], dtype=np.float64)
        c = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        observed = hv.reference_gemm(
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

    def test_int32_accumulator_gemm_matches_numpy(self):
        a = np.asarray([[1, 3], [2, 4]], dtype=np.int8)
        b = np.asarray([[5], [6]], dtype=np.int8)
        c = np.zeros((2, 1), dtype=np.int32)
        observed = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Int32,
            hv.ScalarType.Int32,
        )
        expected = a.astype(np.int32) @ b.astype(np.int32)
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_float16_accumulator_rounds_each_step(self):
        a = np.full((1, 64), np.float16(0.1), dtype=np.float16)
        b = np.full((64, 1), np.float16(0.1), dtype=np.float16)
        c = np.zeros((1, 1), dtype=np.float16)
        observed = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float16,
            hv.ScalarType.Float16,
        )

        expected = np.float16(0)
        for reduction in range(a.shape[1]):
            product = np.float16(a[0, reduction] * b[reduction, 0])
            expected = np.float16(expected + product)
        np.testing.assert_array_equal(
            hv.to_numpy(observed, np.float32),
            np.asarray([[expected]], dtype=np.float32),
        )

    def test_xfloat32_truncates_operand_mantissas(self):
        a = np.asarray([[1.234567, -2.345678]], dtype=np.float32)
        b = np.asarray([[3.456789], [4.567891]], dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        observed = hv.reference_gemm(
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

    def test_complex_gemm_matches_numpy(self):
        a = np.asarray(
            [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0 + 0.5j, 4.0 + 3.0j]],
            dtype=np.complex64,
        )
        b = np.asarray(
            [[2.0 - 1.0j], [0.5 + 3.0j]], dtype=np.complex64
        )
        c = np.asarray([[1.0j], [2.0 - 1.0j]], dtype=np.complex64)
        alpha = 0.5 + 0.25j
        beta = -1.0 + 0.5j
        observed = hv.reference_gemm(
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
        observed = hv.reference_gemm(
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

        pre_scaled = hv.reference_gemm(
            hv.from_numpy(
                np.asarray([[1.1]], dtype=np.float32), hv.ScalarType.Float16
            ),
            hv.from_numpy(np.asarray([[1.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0.0]], dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float8E4M3,
            pre_quantization_scale_a=hv.from_numpy(
                np.asarray([3.0], dtype=np.float32)
            ),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(pre_scaled), np.asarray([[3.25]], dtype=np.float32)
        )

        vector_pre_scaled = hv.reference_gemm(
            hv.from_numpy(
                np.asarray([[1.1], [1.1]], dtype=np.float32),
                hv.ScalarType.Float16,
            ),
            hv.from_numpy(np.asarray([[1.0]], dtype=np.float32)),
            hv.from_numpy(np.zeros((2, 1), dtype=np.float32)),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            compute_type_a=hv.ScalarType.Float8E4M3,
            pre_quantization_scale_a=hv.from_numpy(
                np.asarray([3.0, 4.0], dtype=np.float32)
            ),
            pre_quantization_axis_a=hv.MatrixAxis.Row,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(vector_pre_scaled),
            np.asarray([[3.25], [4.5]], dtype=np.float32),
        )

    def test_block_scaled_tiled_gemm_matches_numpy(self):
        a = np.ones((1, 16), dtype=np.float32)
        b = np.ones((16, 1), dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        scale_a = np.asarray([[2.0, 4.0]], dtype=np.float32)
        scale_b = np.asarray([[8.0, 16.0]], dtype=np.float32)
        observed = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            backend=hv.GemmBackend.Tiled,
            block_scale_a=hv.from_numpy(scale_a),
            block_scale_b=hv.from_numpy(scale_b),
            block_size_a=8,
            block_size_b=8,
        )
        expected = np.asarray(
            [[np.sum(a[:, :8] @ b[:8, :] * 2.0 * 8.0)
              + np.sum(a[:, 8:] @ b[8:, :] * 4.0 * 16.0)]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_selected_output_gemm(self):
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c = np.zeros((2, 2), dtype=np.float32)
        observed = hv.reference_gemm(
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
        tiled = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            output_selection=hv.OutputSelection.explicit_indices([0, 3]),
            backend=hv.GemmBackend.Tiled,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(tiled),
            np.asarray([[19.0, 0.0], [0.0, 50.0]], dtype=np.float32),
        )
        self.assertEqual(
            hv.OutputSelection.prime_stride(10, 10, 3).indices(10),
            [0, 3, 6, 9],
        )

    def test_reference_epilogue_matches_numpy(self):
        values = np.asarray([[-2.0, 1.0], [3.0, -4.0]], dtype=np.float32)
        bias = np.asarray([1.0, 2.0], dtype=np.float32)
        result = hv.reference_epilogue(
            hv.from_numpy(values),
            hv.ScalarType.Float16,
            hv.ScalarType.Float32,
            bias=hv.from_numpy(bias),
            bias_axis=hv.MatrixAxis.Row,
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
        np.testing.assert_array_equal(
            hv.to_numpy(result.raw_output), activated * 2.0
        )
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

    def test_reference_gradient_epilogue_matches_numpy(self):
        gradient = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        activation_input = np.asarray(
            [[-1.0, 1.0], [2.0, -2.0]], dtype=np.float32
        )
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
        activation_argument = coefficient0 * activation_input * (
            1.0 + coefficient1 * activation_input * activation_input
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

    def test_tensor_contraction_matches_numpy_einsum(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) - 5
        b = np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2) - 7
        c = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        observed = hv.reference_contraction(
            hv.from_numpy(a),
            [0, 1, 3, 4],
            hv.from_numpy(b),
            [0, 2, 3, 4],
            hv.from_numpy(c),
            [0, 1, 2],
            [0, 1, 2],
            [3, 4],
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
        )
        expected = 2.0 * np.einsum("bmxy,bnxy->bmn", a, b) - c
        np.testing.assert_array_equal(hv.to_numpy(observed), expected)


if __name__ == "__main__":
    unittest.main()
