# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gc
import math
import unittest
import weakref

import numpy as np

import roc_host_validation as hv

GENERATION_REAL_RANDOM_DOMAIN = 0
GENERATION_IMAGINARY_RANDOM_DOMAIN = 0x243F6A8885A308D3
MX_DATA_RANDOM_DOMAIN = 0x3F84D5B5B5470917
MX_BOUNDED_SCALE_RANDOM_DOMAIN = 0xA24BAED4963EE407


def counter_random(seed, domain, index):
    mask = (1 << 64) - 1
    value = (
        seed
        ^ ((domain + 0x9E3779B97F4A7C15) & mask)
        ^ ((index * 0xBF58476D1CE4E5B9) & mask)
    )
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def indexed_uniform_unit(seed, domain, index):
    mantissa = counter_random(seed, domain, index) >> 11
    return (mantissa + 0.5) / (1 << 53)


def real_generation_recipe(
    component,
    *,
    seed=0,
    index_order=hv.IndexOrder.FirstDimensionFastest,
):
    return hv.GenerationRecipe.real_only(
        component,
        hv.GenerationRecipeSettings(seed=seed, index_order=index_order),
    )


def encode_fp4_e2m1(value):
    value = np.float32(value)
    sign = 0x8 if np.signbit(value) else 0
    magnitude = abs(float(value))
    positive_values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    if math.isnan(magnitude) or magnitude >= positive_values[-1]:
        return sign | 0x7
    upper = next(
        index
        for index, candidate in enumerate(positive_values)
        if candidate >= magnitude
    )
    if upper == 0:
        return sign
    lower = upper - 1
    lower_distance = magnitude - positive_values[lower]
    upper_distance = positive_values[upper] - magnitude
    if lower_distance < upper_distance:
        selected = lower
    elif upper_distance < lower_distance:
        selected = upper
    else:
        selected = lower if lower % 2 == 0 else upper
    return sign | selected


def decode_e8m0(raw):
    return math.nan if raw == 0xFF else math.ldexp(1.0, raw - 127)


def constrain_fp4_to_interval(raw, scale, minimum, maximum):
    fp4_values = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    sign = raw & 0x8
    magnitude = raw & 0x7
    for _ in range(8):
        represented = fp4_values[sign | magnitude] * scale
        if minimum <= represented <= maximum:
            return sign | magnitude
        negative = bool(sign)
        increase_magnitude = (
            represented < minimum if not negative else represented > maximum
        )
        if increase_magnitude:
            if magnitude == 0x7:
                break
            magnitude += 1
        else:
            if magnitude == 0:
                break
            magnitude -= 1
    raise ValueError("bounded interval has no representable FP4 value")


def bounded_mx_fp4_oracle(
    dimensions,
    leading_dimension,
    block_axis,
    block_size,
    seed,
    minimum,
    maximum,
):
    rows, columns = dimensions
    blocked_extent = dimensions[block_axis]
    free_extent = dimensions[1 - block_axis]
    block_count = (blocked_extent + block_size - 1) // block_size
    scale_count = block_count * free_extent
    source = np.empty(dimensions, dtype=np.float64)
    for column in range(columns):
        for row in range(rows):
            logical_index = row + column * rows
            unit = indexed_uniform_unit(seed, MX_DATA_RANDOM_DOMAIN, logical_index)
            source[row, column] = minimum + (maximum - minimum) * unit

    physical_raw = [0] * (leading_dimension * columns)
    scale_raw = bytearray(scale_count)
    scale_indices = np.empty(dimensions, dtype=np.uint32)
    reference = np.empty(dimensions, dtype=np.float32)
    fp4_values = np.asarray(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=np.float32,
    )

    for scale_index in range(scale_count):
        if block_axis == 0:
            block = scale_index % block_count
            free_coordinate = scale_index // block_count
        else:
            block = scale_index // free_extent
            free_coordinate = scale_index % free_extent
        block_element_count = min(block_size, blocked_extent - block * block_size)
        coordinates = []
        for offset in range(block_element_count):
            if block_axis == 0:
                coordinates.append((block * block_size + offset, free_coordinate))
            else:
                coordinates.append((free_coordinate, block * block_size + offset))

        maximum_magnitude = max(abs(source[row, column]) for row, column in coordinates)
        requested_scale = maximum_magnitude / 6.0
        selected_scale_raw = next(
            raw for raw in range(0xFF) if decode_e8m0(raw) >= requested_scale
        )
        if (
            selected_scale_raw < 0xFE
            and counter_random(seed, MX_BOUNDED_SCALE_RANDOM_DOMAIN, scale_index) & 1
        ):
            selected_scale_raw += 1
        scale_raw[scale_index] = selected_scale_raw
        scale_value = decode_e8m0(selected_scale_raw)

        for row, column in coordinates:
            logical_index = row + column * rows
            data_raw = encode_fp4_e2m1(source[row, column] / scale_value)
            data_raw = constrain_fp4_to_interval(
                data_raw, scale_value, minimum, maximum
            )
            physical_raw[row + column * leading_dimension] = data_raw
            scale_indices[row, column] = scale_index
            reference[row, column] = np.float32(fp4_values[data_raw] * scale_value)

    return (
        pack_bits(physical_raw, 4),
        bytes(scale_raw),
        scale_indices,
        reference,
    )


def cxx_remainder(value, divisor):
    return value % divisor if value >= 0 else -((-value) % divisor)


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
    hv.ScalarType.E4M3: (8, 4, 3, 7, "e4m3_scale"),
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
    if kind == "e4m3_scale":
        raw &= 0x7F
        if raw == 0x7F:
            return math.nan
        return decode_binary(
            raw,
            exponent_bits,
            mantissa_bits,
            bias,
            7,
            signed=False,
        )
    return decode_binary(
        raw,
        exponent_bits,
        mantissa_bits,
        bias,
        total_bits,
        signed=kind != "e5m3_scale",
    )


def quantize_bfloat16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    least_significant_bit = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + least_significant_bit
    quantized = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return quantized[()] if quantized.ndim == 0 else quantized


def matmul_float32(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            value = np.float32(0.0)
            for reduction in range(left.shape[1]):
                product = np.float32(left[row, reduction] * right[reduction, column])
                value = np.float32(value + product)
            result[row, column] = value
    return result


def wrap_int32(value):
    unsigned = int(value) & 0xFFFFFFFF
    return unsigned if unsigned < 0x80000000 else unsigned - 0x100000000


def add_int32(left, right):
    return wrap_int32(int(left) + int(right))


def multiply_int32(left, right):
    return wrap_int32(int(left) * int(right))


def gemm_int32_exact(left, right, initial, alpha=1, beta=0, output_scale=1):
    left = np.asarray(left)
    right = np.asarray(right)
    initial = np.asarray(initial)
    result = np.empty((left.shape[0], right.shape[1]), dtype=np.int32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = 0
            for reduction in range(left.shape[1]):
                product = multiply_int32(
                    int(left[row, reduction]), int(right[reduction, column])
                )
                accumulation = add_int32(accumulation, product)
            combined = add_int32(
                multiply_int32(int(alpha), accumulation),
                multiply_int32(int(beta), int(initial[row, column])),
            )
            result[row, column] = multiply_int32(combined, int(output_scale))
    return result


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
                round_trip = unpack_bits(encoded.storage, int(encodable.sum()), bits)
                expected_raw = np.asarray(raw, dtype=np.uint32)
                if scalar_type == hv.ScalarType.E4M3:
                    expected_raw &= 0x7F
                np.testing.assert_array_equal(round_trip, expected_raw[encodable])

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

    def test_numpy_tensor_preserves_positive_stride_and_gaps(self):
        storage = np.asarray(
            [
                1.0,
                np.inf,
                2.0,
                np.inf,
                3.0,
                4.0,
                np.inf,
                5.0,
                np.inf,
                6.0,
            ],
            dtype=np.float32,
        )
        values = storage.reshape(2, 5)[:, ::2]
        view = hv.Tensor.from_numpy(values)
        expected = hv.from_numpy(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

        self.assertEqual(view.type, hv.ScalarType.Float32)
        self.assertEqual(view.shape, [2, 3])
        self.assertEqual(view.strides, [5, 2])
        self.assertEqual(view.offset, 0)
        self.assertEqual(len(view.storage), storage.nbytes)
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertTrue(hv.compare(view, expected).passed)

        tolerance = hv.find_allclose_tolerance(view, expected, [0.0], [0.0])
        self.assertIsNotNone(tolerance)
        self.assertEqual(tolerance.absolute, 0.0)
        self.assertTrue(
            hv.check_unused_tensor_storage(view, allocated_elements=storage.size).passed
        )

    def test_numpy_tensor_preserves_negative_strides(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.shape, [4, 3])
        self.assertEqual(view.strides, [-6, -2])
        self.assertEqual(view.offset, 22)
        self.assertEqual(len(view.storage), 23 * values.itemsize)
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertTrue(hv.compare(view, hv.from_numpy(values)).passed)

    def test_tensor_conversion_preserves_layout(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        copied = view.to(hv.ScalarType.Float32)
        self.assertEqual(copied.strides, view.strides)
        self.assertEqual(copied.offset, view.offset)
        np.testing.assert_array_equal(hv.to_numpy(copied), values)

        converted = view.to(hv.ScalarType.Float16)
        self.assertEqual(converted.strides, view.strides)
        self.assertEqual(converted.offset, view.offset)
        np.testing.assert_array_equal(hv.to_numpy(converted), values.astype(np.float16))

        bfloat_source = hv.from_numpy(np.asarray([1.1, -2.25], dtype=np.float32))
        np.testing.assert_array_equal(
            hv.to_numpy(bfloat_source.to(hv.ScalarType.BFloat16)),
            quantize_bfloat16(np.asarray([1.1, -2.25], dtype=np.float32)),
        )

        packed = hv.from_numpy(
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
            hv.ScalarType.Float4E2M1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(packed.to(hv.ScalarType.Float32)),
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
        )

    def test_reference_axpby_matches_numpy(self):
        x_values = np.asarray(
            [[[1.0, -2.0], [3.0, 4.0]], [[-1.0, 2.0], [5.0, -6.0]]],
            dtype=np.float16,
        )
        y_source = np.asarray(
            [[[0.25, 1.1], [-3.5, 2.25]], [[4.0, -0.5], [1.75, 3.0]]],
            dtype=np.float32,
        )
        x = hv.from_numpy(x_values)
        y = hv.from_numpy(y_source, hv.ScalarType.BFloat16)
        result = hv.reference_axpby(
            x=x,
            y=y,
            output_type=hv.ScalarType.Float32,
            accumulator_type=hv.ScalarType.Float32,
            alpha=0.5,
            beta=-1.25,
        )

        y_values = quantize_bfloat16(y_source)
        expected = np.empty(x_values.shape, dtype=np.float32)
        for index in np.ndindex(x_values.shape):
            value = np.float32(0.0)
            value = np.float32(value + np.float32(0.5) * np.float32(x_values[index]))
            value = np.float32(value + np.float32(-1.25) * y_values[index])
            expected[index] = value
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)
        self.assertEqual(result.run_info.output_elements_written, expected.size)

        y_only = hv.reference_axpby(y=y, beta=3.0)
        np.testing.assert_array_equal(
            hv.to_numpy(y_only.output),
            np.float32(3.0) * y_values,
        )

        padded_x = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            np.asarray([1.0, 2.0, -99.0, 3.0, 4.0], dtype=np.float32).tobytes(),
            strides=[3, 1],
        )
        contiguous = hv.reference_axpby(x=padded_x)
        self.assertEqual(contiguous.output.strides, [2, 1])
        self.assertEqual(contiguous.output.offset, 0)
        np.testing.assert_array_equal(
            hv.to_numpy(contiguous.output),
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

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

        np.testing.assert_allclose(
            hv.to_numpy(result.output), expected, rtol=1e-6, atol=1e-7
        )
        self.assertEqual(result.run_info.slices_processed, 4)
        self.assertEqual(result.run_info.output_elements_written, source.size)
        with self.assertRaises(IndexError):
            hv.reference_softmax(input_tensor, axis=source.ndim)

        padded_input = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            np.asarray([1.0, 2.0, -99.0, 3.0, 4.0], dtype=np.float32).tobytes(),
            strides=[3, 1],
        )
        contiguous = hv.reference_softmax(padded_input, axis=1)
        self.assertEqual(contiguous.output.strides, [2, 1])
        self.assertEqual(contiguous.output.offset, 0)

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
        self.assertEqual(result.run_info.slices_processed, 4)
        self.assertEqual(result.run_info.output_elements_written, source.size)
        self.assertEqual(result.run_info.mean_elements_written, 4)
        self.assertEqual(result.run_info.inverse_variance_elements_written, 4)

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

    def test_numpy_tensor_owns_an_independent_copy(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        view = hv.Tensor.from_numpy(values)
        expected = hv.from_numpy(values.copy())
        self.assertTrue(hv.compare(view, expected).passed)

        values[1, 2] = -17.0
        np.testing.assert_array_equal(
            hv.to_numpy(view), np.arange(6, dtype=np.float32).reshape(2, 3)
        )
        self.assertTrue(hv.compare(view, expected).passed)

    def test_numpy_tensor_does_not_retain_source_owner(self):
        values = np.arange(6, dtype=np.float64)
        owner = weakref.ref(values)
        view = hv.Tensor.from_numpy(values)

        del values
        gc.collect()
        self.assertIsNone(owner())
        np.testing.assert_array_equal(hv.to_numpy(view), np.arange(6, dtype=np.float64))

    def test_tensor_clone_is_independent(self):
        tensor = hv.from_numpy(np.arange(6, dtype=np.float32))
        cloned = tensor.clone()
        recipe = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0))
        )
        hv.generate_at(cloned, 5, recipe)
        np.testing.assert_array_equal(
            hv.to_numpy(tensor), np.arange(6, dtype=np.float32)
        )
        self.assertEqual(hv.to_numpy(cloned)[5], -17.0)

    def test_numpy_tensor_accepts_read_only_array(self):
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        values.flags.writeable = False
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.type, hv.ScalarType.Int32)
        np.testing.assert_array_equal(hv.to_numpy(view), values)

    def test_numpy_tensor_rejects_storage_conversion(self):
        with self.assertRaises(TypeError):
            hv.Tensor.from_numpy([1.0, 2.0])
        with self.assertRaises(TypeError):
            hv.Tensor.from_numpy(np.arange(4, dtype=np.dtype(">f4")))
        with self.assertRaises(ValueError):
            hv.Tensor.from_numpy(
                np.arange(4, dtype=np.float64),
                hv.ScalarType.Float32,
            )
        with self.assertRaises(ValueError):
            hv.Tensor.from_numpy(
                np.arange(4, dtype=np.uint8),
                hv.ScalarType.Float8E4M3,
            )

        byte_storage = np.arange(16, dtype=np.uint8)
        byte_strided = np.ndarray(
            shape=(3,),
            dtype=np.int16,
            buffer=byte_storage,
            strides=(3,),
        )
        with self.assertRaises(ValueError):
            hv.Tensor.from_numpy(byte_strided)

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
        seed = 17
        recipe = real_generation_recipe(
            hv.GenerationRecipe.uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
            ),
            seed=seed,
        )
        first = hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe)
        second = hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe)
        self.assertTrue(hv.compare(first, second).passed)
        expected = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 7
                for index in range(6)
            ],
            dtype=np.float32,
        ).reshape((2, 3), order="F")
        np.testing.assert_array_equal(
            hv.to_numpy(first),
            expected,
        )

        changed_values = hv.to_numpy(second).copy()
        changed_values[1, 2] += 1
        changed = hv.from_numpy(changed_values)
        options = hv.ComparisonOptions()
        options.max_reported_mismatches = 2
        result = hv.compare(changed, first, options)
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)
        self.assertEqual(result.reported_mismatches[0].index, 5)

    def test_comparison_program_matches_numpy(self):
        expected_values = np.asarray(
            [[3.0 + 4.0j, 1.0 - 2.0j], [0.0 + 0.0j, -2.0 + 1.0j]],
            dtype=np.complex128,
        )
        observed_values = expected_values.copy()
        observed_values[0, 1] += 0.25 - 0.5j
        observed_values[1, 0] += 1.0

        expected = hv.from_numpy(expected_values)
        observed = hv.from_numpy(observed_values)
        options = hv.ComparisonOptions()
        options.absolute_tolerance = 0.1
        options.relative_frobenius_tolerance = 1.0
        report = hv.compare(observed, expected, options)

        difference = observed_values - expected_values
        self.assertEqual(report.compared, expected_values.size)
        self.assertEqual(report.mismatches, 2)
        self.assertAlmostEqual(
            report.frobenius_difference,
            float(np.linalg.norm(difference)),
        )
        self.assertAlmostEqual(
            report.frobenius_expected,
            float(np.linalg.norm(expected_values)),
        )
        self.assertAlmostEqual(
            report.relative_frobenius_error,
            float(np.linalg.norm(difference) / np.linalg.norm(expected_values)),
        )
        self.assertEqual(
            report.reported_mismatches[0].coordinates,
            [0, 1],
        )

        selected = hv.ComparisonOptions()
        selected.selection.index_order = hv.IndexOrder.FirstDimensionFastest
        selected.selection.stride = 2
        selected_report = hv.compare(observed, expected, selected)
        expected_flat_fortran = expected_values.reshape(-1, order="F")
        observed_flat_fortran = observed_values.reshape(-1, order="F")
        selected_indices = np.arange(0, expected_values.size, 2)
        selected_mismatches = np.count_nonzero(
            observed_flat_fortran[selected_indices]
            != expected_flat_fortran[selected_indices]
        )
        self.assertEqual(selected_report.compared, len(selected_indices))
        self.assertEqual(selected_report.mismatches, int(selected_mismatches))

    def test_complex_allclose_modes_match_numpy_magnitude_policy(self):
        observed_values = np.asarray([1.0 + 1.0j], dtype=np.complex128)
        expected_values = np.asarray([0.0 + 0.0j], dtype=np.complex128)
        observed = hv.from_numpy(observed_values)
        expected = hv.from_numpy(expected_values)

        magnitude = hv.allclose_comparison_options(1.0, 0.0)
        self.assertEqual(
            magnitude.complex_pointwise_mode,
            hv.ComplexPointwiseMode.Magnitude,
        )
        self.assertFalse(hv.compare(observed, expected, magnitude).passed)
        self.assertEqual(
            hv.compare(observed, expected, magnitude).passed,
            bool(np.allclose(observed_values, expected_values, atol=1.0, rtol=0.0)),
        )

        componentwise = hv.allclose_comparison_options(1.0, 0.0)
        componentwise.complex_pointwise_mode = hv.ComplexPointwiseMode.Componentwise
        self.assertTrue(hv.compare(observed, expected, componentwise).passed)

        magnitude.compute_pointwise_statistics = False
        magnitude.compute_frobenius = False
        self.assertFalse(hv.compare(observed, expected, magnitude).passed)

        boundary_observed_values = np.asarray([0.0 + 0.0j], dtype=np.complex128)
        boundary_expected_values = np.asarray([3.0 + 4.0j], dtype=np.complex128)
        boundary = hv.allclose_comparison_options(0.0, 1.0)
        boundary_result = hv.compare(
            hv.from_numpy(boundary_observed_values),
            hv.from_numpy(boundary_expected_values),
            boundary,
        )
        reverse_result = hv.compare(
            hv.from_numpy(boundary_expected_values),
            hv.from_numpy(boundary_observed_values),
            boundary,
        )
        self.assertEqual(
            boundary_result.passed,
            bool(
                np.allclose(
                    boundary_observed_values,
                    boundary_expected_values,
                    atol=0.0,
                    rtol=1.0,
                )
            ),
        )
        self.assertEqual(
            reverse_result.passed,
            bool(
                np.allclose(
                    boundary_expected_values,
                    boundary_observed_values,
                    atol=0.0,
                    rtol=1.0,
                )
            ),
        )

        nan_observed = hv.from_numpy(
            np.asarray([complex(np.nan, 1.0)], dtype=np.complex128)
        )
        nan_expected = hv.from_numpy(
            np.asarray([complex(1.0, np.nan)], dtype=np.complex128)
        )
        equal_nan = hv.allclose_comparison_options(0.0, 0.0, True)
        nan_result = hv.compare(nan_observed, nan_expected, equal_nan)
        self.assertTrue(nan_result.passed)
        self.assertEqual(nan_result.matched_nans, 1)

        search_observed = hv.from_numpy(np.asarray([0.09 + 0.09j], dtype=np.complex128))
        search_expected = hv.from_numpy(np.asarray([0.0 + 0.0j], dtype=np.complex128))
        self.assertIsNone(
            hv.find_allclose_tolerance(
                search_observed,
                search_expected,
                [0.1],
                [0.0],
            )
        )
        componentwise_search = hv.allclose_comparison_options()
        componentwise_search.complex_pointwise_mode = (
            hv.ComplexPointwiseMode.Componentwise
        )
        self.assertIsNotNone(
            hv.find_allclose_tolerance(
                search_observed,
                search_expected,
                [0.1],
                [0.0],
                componentwise_search,
            )
        )

    def test_comparison_nonfinite_ulp_and_sentinel(self):
        expected_values = np.asarray([np.inf, np.nan, 1.0], dtype=np.float64)
        observed_values = expected_values.copy()
        options = hv.ComparisonOptions()
        options.equal_nans = True
        report = hv.compare(
            hv.from_numpy(observed_values),
            hv.from_numpy(expected_values),
            options,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.matched_infinities, 1)
        self.assertEqual(report.matched_nans, 1)

        one_ulp = np.nextafter(np.float64(1.0), np.float64(2.0))
        ulp_options = hv.ComparisonOptions()
        ulp_options.compute_ulp = True
        ulp_options.ulp_type = hv.ScalarType.Float64
        ulp_options.maximum_ulp_tolerance = 1.0
        ulp_report = hv.compare(
            hv.from_numpy(np.asarray([one_ulp])),
            hv.from_numpy(np.asarray([1.0], dtype=np.float64)),
            ulp_options,
        )
        self.assertEqual(ulp_report.maximum_ulp, 1.0)
        self.assertTrue(ulp_report.ulp_passed)
        self.assertEqual(
            hv.encoded_ulp_distance(
                0.0,
                float(np.nextafter(np.float32(0.0), np.float32(1.0))),
                hv.ScalarType.Float32,
            ),
            1.0,
        )

        candidates = [1e-6, 1e-5, 1e-4, 1e-3]
        tolerance = hv.find_allclose_tolerance(
            hv.from_numpy(np.asarray([1.00009], dtype=np.float64)),
            hv.from_numpy(np.asarray([1.0], dtype=np.float64)),
            candidates,
            [0.0],
        )
        self.assertIsNotNone(tolerance)
        self.assertEqual(tolerance.absolute, 1e-4)

        sentinel_values = np.full(5, np.inf, dtype=np.float32)
        sentinel = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            sentinel_values.tobytes(),
            strides=[1, 3],
        )
        self.assertTrue(
            hv.check_unused_tensor_storage(sentinel, allocated_elements=5).passed
        )
        sentinel_values[2] = 0.0
        sentinel = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            sentinel_values.tobytes(),
            strides=[1, 3],
        )
        sentinel_report = hv.check_unused_tensor_storage(sentinel, allocated_elements=5)
        self.assertFalse(sentinel_report.passed)
        self.assertEqual(sentinel_report.reported_mismatches[0].index, 2)

    def test_default_comparison_policies_match_numpy(self):
        real_values = np.asarray(
            [
                -5.0,
                -4.0,
                -3.0,
                -2.25,
                -2.0,
                -1.1,
                -1.02,
                -1.0,
                -0.05,
                -0.005,
                -0.0,
                0.0,
                0.005,
                0.05,
                1.0,
                1.02,
                1.1,
                1.5,
                2.0,
                2.25,
                2.5,
                3.0,
                4.0,
                5.0,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )

        real_types = [
            hv.ScalarType.Float16,
            hv.ScalarType.BFloat16,
            hv.ScalarType.Float8E4M3,
            hv.ScalarType.Float8E5M2,
            hv.ScalarType.Float8E4M3Fnuz,
            hv.ScalarType.Float8E5M2Fnuz,
            hv.ScalarType.Float32,
            hv.ScalarType.Float64,
        ]
        for scalar_type in real_types:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.from_numpy(
                    np.tile(real_values, real_values.size),
                    scalar_type,
                )
                expected = hv.from_numpy(
                    np.repeat(real_values, real_values.size),
                    scalar_type,
                )
                observed_values = hv.to_numpy(observed, np.float64)
                expected_values = hv.to_numpy(expected, np.float64)
                options = hv.default_comparison_options(scalar_type)
                options.compute_frobenius = False

                with np.errstate(invalid="ignore"):
                    difference = np.abs(observed_values - expected_values)
                    tolerance = options.symmetric_relative_tolerance * (
                        np.abs(observed_values) + np.abs(expected_values) + 1.0
                    )
                oracle = observed_values == expected_values
                finite = np.isfinite(observed_values) & np.isfinite(expected_values)
                oracle |= finite & (difference < tolerance)

                report = hv.compare(observed, expected, options)
                self.assertEqual(
                    report.mismatches,
                    int(np.count_nonzero(~oracle)),
                )
                self.assertEqual(report.passed, bool(np.all(oracle)))

        complex_values = np.asarray(
            [
                0.0 + 0.0j,
                -0.0 + 0.0j,
                1.0 + 1.0j,
                1.0002 + 1.0002j,
                1.001 + 1.0j,
                1.0 + 1.001j,
                complex(np.inf, 0.0),
                complex(0.0, np.inf),
                complex(np.nan, 0.0),
                complex(0.0, np.nan),
            ],
            dtype=np.complex128,
        )
        for scalar_type in [
            hv.ScalarType.ComplexFloat32,
            hv.ScalarType.ComplexFloat64,
        ]:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.from_numpy(
                    np.tile(complex_values, complex_values.size),
                    scalar_type,
                )
                expected = hv.from_numpy(
                    np.repeat(complex_values, complex_values.size),
                    scalar_type,
                )
                observed_values = hv.to_numpy(observed, np.complex128)
                expected_values = hv.to_numpy(expected, np.complex128)
                options = hv.default_comparison_options(scalar_type)
                options.compute_frobenius = False

                def component_oracle(observed_component, expected_component):
                    with np.errstate(invalid="ignore"):
                        difference = np.abs(observed_component - expected_component)
                        tolerance = options.symmetric_relative_tolerance * (
                            np.abs(observed_component)
                            + np.abs(expected_component)
                            + 1.0
                        )
                    result = observed_component == expected_component
                    finite = np.isfinite(observed_component) & np.isfinite(
                        expected_component
                    )
                    return result | (finite & (difference < tolerance))

                oracle = component_oracle(
                    observed_values.real, expected_values.real
                ) & component_oracle(observed_values.imag, expected_values.imag)
                report = hv.compare(observed, expected, options)
                self.assertEqual(
                    report.mismatches,
                    int(np.count_nonzero(~oracle)),
                )

        for dtype, scalar_type in [
            (np.int8, hv.ScalarType.Int8),
            (np.int32, hv.ScalarType.Int32),
            (np.uint32, hv.ScalarType.UInt32),
        ]:
            observed_values = np.asarray([0, 1, 2, 3], dtype=dtype)
            expected_values = np.asarray([0, 1, 7, 3], dtype=dtype)
            report = hv.compare(
                hv.from_numpy(observed_values, scalar_type),
                hv.from_numpy(expected_values, scalar_type),
                hv.default_comparison_options(scalar_type),
            )
            self.assertEqual(
                report.mismatches,
                int(np.count_nonzero(observed_values != expected_values)),
            )

    def test_integer_comparison_is_exact_beyond_float64_precision(self):
        observed_unsigned = np.asarray(
            [2**53, 2**63, np.iinfo(np.uint64).max],
            dtype=np.uint64,
        )
        expected_unsigned = np.asarray(
            [2**53 + 1, 2**63, np.iinfo(np.uint64).max - 1],
            dtype=np.uint64,
        )
        unsigned_report = hv.compare(
            hv.from_numpy(observed_unsigned),
            hv.from_numpy(expected_unsigned),
        )
        self.assertEqual(
            unsigned_report.mismatches,
            int(np.count_nonzero(observed_unsigned != expected_unsigned)),
        )
        self.assertFalse(unsigned_report.passed)

        observed_signed = np.asarray(
            [np.iinfo(np.int64).min, 2**53, np.iinfo(np.int64).max],
            dtype=np.int64,
        )
        expected_signed = np.asarray(
            [np.iinfo(np.int64).max, 2**53 + 1, np.iinfo(np.int64).max],
            dtype=np.int64,
        )
        signed_report = hv.compare(
            hv.from_numpy(observed_signed),
            hv.from_numpy(expected_signed),
        )
        self.assertEqual(
            signed_report.mismatches,
            int(np.count_nonzero(observed_signed != expected_signed)),
        )
        self.assertFalse(signed_report.passed)

    def test_explicit_tolerance_strict_boundary(self):
        observed = hv.from_numpy(np.asarray([1.02, 0.0], dtype=np.float32))
        expected = hv.from_numpy(np.asarray([1.0, 1.0], dtype=np.float32))

        defaults = hv.default_comparison_options(hv.ScalarType.Float32)
        defaults.compute_frobenius = False
        self.assertEqual(hv.compare(observed, expected, defaults).mismatches, 2)

        overridden = hv.default_comparison_options(hv.ScalarType.Float32, 0.01)
        overridden.compute_frobenius = False
        self.assertEqual(
            hv.compare(observed, expected, overridden).mismatches,
            1,
        )

        exact_boundary = hv.default_comparison_options(hv.ScalarType.Float32, 0.5)
        exact_boundary.compute_frobenius = False
        boundary_observed = hv.from_numpy(np.asarray([0.0], dtype=np.float32))
        boundary_expected = hv.from_numpy(np.asarray([1.0], dtype=np.float32))
        self.assertFalse(
            hv.compare(boundary_observed, boundary_expected, exact_boundary).passed
        )
        exact_boundary.strict_tolerance = False
        self.assertTrue(
            hv.compare(boundary_observed, boundary_expected, exact_boundary).passed
        )

    def test_default_relative_policy_scales_at_large_magnitudes(self):
        cases = [
            (
                np.float32,
                hv.ScalarType.Float32,
                np.float32(1.0e6),
                np.float32(100.0),
                np.float32(500.0),
            ),
            (
                np.float64,
                hv.ScalarType.Float64,
                np.float64(1.0e12),
                np.float64(1.0),
                np.float64(5.0),
            ),
        ]
        for dtype, scalar_type, base, accepted_delta, rejected_delta in cases:
            with self.subTest(scalar_type=scalar_type):
                options = hv.default_comparison_options(scalar_type)
                options.compute_frobenius = False
                accepted = hv.compare(
                    hv.from_numpy(np.asarray([base], dtype=dtype)),
                    hv.from_numpy(np.asarray([base + accepted_delta], dtype=dtype)),
                    options,
                )
                rejected = hv.compare(
                    hv.from_numpy(np.asarray([base], dtype=dtype)),
                    hv.from_numpy(np.asarray([base + rejected_delta], dtype=dtype)),
                    options,
                )
                self.assertTrue(accepted.passed)
                self.assertFalse(rejected.passed)

    def test_default_float32_policy_rejects_opposite_signs_near_zero(self):
        observed = hv.from_numpy(
            np.asarray([-0.0001, -0.0002], dtype=np.float32)
        )
        expected = hv.from_numpy(np.asarray([0.0001, 0.0002], dtype=np.float32))
        options = hv.default_comparison_options(hv.ScalarType.Float32)
        options.compute_frobenius = False
        report = hv.compare(observed, expected, options)
        self.assertFalse(report.passed)
        self.assertEqual(report.mismatches, 1)

    def test_signed_zero_policy_is_explicit(self):
        observed = hv.from_numpy(np.asarray([0.0, -0.0], dtype=np.float64))
        expected = hv.from_numpy(np.asarray([-0.0, 0.0], dtype=np.float64))
        options = hv.ComparisonOptions()
        options.compute_frobenius = False
        self.assertTrue(hv.compare(observed, expected, options).passed)
        options.equal_signed_zero = False
        report = hv.compare(observed, expected, options)
        self.assertFalse(report.passed)
        self.assertEqual(report.signed_zero_mismatches, 2)

    def test_indexed_generation_matches_numpy(self):
        serial = hv.generate_tensor(
            hv.ScalarType.Float32,
            [2, 3],
            real_generation_recipe(hv.GenerationRecipe.serial_index()),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(serial),
            np.arange(6, dtype=np.float32).reshape((2, 3), order="F"),
        )

        complex_recipe = hv.GenerationRecipe.cartesian(
            hv.GenerationRecipe.sine(),
            hv.GenerationRecipe.cosine(),
        )
        complex_values = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32, [2, 3], complex_recipe
        )
        indices = np.arange(6, dtype=np.float32).reshape((2, 3), order="F")
        np.testing.assert_allclose(
            hv.to_numpy(complex_values),
            np.sin(indices) + 1j * np.cos(indices),
            rtol=1e-6,
            atol=1e-6,
        )

        identity = hv.generate_tensor(
            hv.ScalarType.Float32,
            [3, 4],
            real_generation_recipe(hv.GenerationRecipe.identity()),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(identity), np.eye(3, 4, dtype=np.float32)
        )

        seed = 19
        random_component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
        )
        random_recipe = hv.GenerationRecipe.cartesian(
            random_component,
            random_component,
            hv.GenerationRecipeSettings(seed=seed),
        )
        random_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            )
        )
        random_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.ComplexFloat32,
                [4, 4],
                random_recipe,
            )
        )
        expected_real = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="F")
        expected_imaginary = np.asarray(
            [
                -3 + counter_random(seed, GENERATION_IMAGINARY_RANDOM_DOMAIN, index) % 7
                for index in range(16)
            ],
            dtype=np.float32,
        ).reshape((4, 4), order="F")
        np.testing.assert_array_equal(random_values, random_repeat)
        np.testing.assert_array_equal(random_values.real, expected_real)
        np.testing.assert_array_equal(random_values.imag, expected_imaginary)
        self.assertFalse(np.array_equal(expected_real, expected_imaginary))
        self.assertTrue(np.all((-3 <= expected_real) & (expected_real <= 3)))

        candidates = np.asarray([-6.0, -1.5, 0.0, 4.0], dtype=np.float32)
        seed = 37
        candidate_recipe = real_generation_recipe(
            hv.GenerationRecipe.candidate_set(
                hv.CandidateSetGenerationParameters(values=candidates.tolist())
            ),
            seed=seed,
        )
        selected = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 4], candidate_recipe)
        )
        expected = np.asarray(
            [
                candidates[
                    counter_random(seed, GENERATION_REAL_RANDOM_DOMAIN, index) % 4
                ]
                for index in range(8)
            ],
            dtype=np.float32,
        ).reshape((2, 4), order="F")
        np.testing.assert_array_equal(selected, expected)
        with self.assertRaises(ValueError):
            hv.GenerationRecipe.candidate_set(
                hv.CandidateSetGenerationParameters(values=[])
            )

        point = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3, 2]))
        first_point_recipe = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=9.0))
        )
        hv.generate_at(point, 3, first_point_recipe)
        expected_point = np.zeros((2, 3, 2), dtype=np.float32)
        expected_point[1, 1, 0] = 9.0
        np.testing.assert_array_equal(hv.to_numpy(point), expected_point)

        last_dimension_fastest = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=7.0)),
            index_order=hv.IndexOrder.LastDimensionFastest,
        )
        hv.generate_at(point, 3, last_dimension_fastest)
        expected_point[0, 1, 1] = 7.0
        np.testing.assert_array_equal(hv.to_numpy(point), expected_point)
        with self.assertRaises(IndexError):
            hv.generate_at(point, point.size, last_dimension_fastest)

        affine_component = hv.GenerationRecipe.affine_index_remainder(
            hv.AffineIndexRemainderGenerationParameters(
                dimension_coefficients=[1, -1, 2],
                offset=-2,
                positive_divisor=5,
            )
        ).with_affine_value_mapping(
            hv.GenerationAffineValueParameters(scale=1.0, offset=1.0)
        )
        affine = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3, 2],
                real_generation_recipe(affine_component),
            )
        )
        expected_affine = np.empty((2, 3, 2), dtype=np.float32)
        for index in np.ndindex(expected_affine.shape):
            numerator = -2 + index[0] - index[1] + 2 * index[2]
            expected_affine[index] = cxx_remainder(numerator, 5) + 1
        np.testing.assert_array_equal(affine, expected_affine)

    def test_type_derived_generation(self):
        maximum_recipe = real_generation_recipe(hv.GenerationRecipe.type_maximum())
        maximum_cases = (
            (hv.ScalarType.Float16, np.finfo(np.float16).max),
            (
                hv.ScalarType.BFloat16,
                np.asarray([0x7F7F0000], dtype=np.uint32).view(np.float32)[0],
            ),
            (hv.ScalarType.Float4E2M1, 6.0),
            (hv.ScalarType.Float6E2M3, 7.5),
            (hv.ScalarType.Float6E3M2, 28.0),
            (hv.ScalarType.Float8E4M3, 448.0),
            (hv.ScalarType.Int8, 127),
            (hv.ScalarType.Int32, np.iinfo(np.int32).max),
        )
        for scalar_type, expected in maximum_cases:
            with self.subTest(scalar_type=scalar_type):
                observed = hv.to_numpy(
                    hv.generate_tensor(scalar_type, [3], maximum_recipe),
                    np.float64 if scalar_type == hv.ScalarType.Float64 else np.float32,
                )
                np.testing.assert_array_equal(
                    observed, np.full(3, expected, dtype=observed.dtype)
                )

        fp4_denormal = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float4E2M1,
                [2],
                real_generation_recipe(hv.GenerationRecipe.type_denormal_minimum()),
            )
        )
        np.testing.assert_array_equal(
            fp4_denormal, np.asarray([0.5, 0.5], dtype=np.float32)
        )

        nan_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E4M3Fnuz,
                [2],
                real_generation_recipe(hv.GenerationRecipe.type_nan()),
            )
        )
        self.assertTrue(np.isnan(nan_values).all())

        infinity = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float8E5M2,
                [2],
                real_generation_recipe(hv.GenerationRecipe.type_infinity()),
            )
        )
        self.assertTrue(np.isposinf(infinity).all())

        type_range_recipe = real_generation_recipe(
            hv.GenerationRecipe.uniform_type_range(),
            seed=23,
        )
        low_precision = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float4E2M1,
                [64],
                type_range_recipe,
            )
        )
        self.assertTrue(np.all((-6.0 <= low_precision) & (low_precision <= 6.0)))
        float64_range = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float64, [64], type_range_recipe)
        )
        self.assertTrue(np.isfinite(float64_range).all())
        self.assertTrue(
            np.all(
                (-np.finfo(np.float64).max <= float64_range)
                & (float64_range <= np.finfo(np.float64).max)
            )
        )

        absolute_integer_recipe = real_generation_recipe(
            hv.GenerationRecipe.absolute_uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-3, upper=3)
            ),
            seed=23,
        )
        unsigned_scale_values = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.E5M3,
                [64],
                absolute_integer_recipe,
            )
        )
        self.assertTrue(
            np.all((0 <= unsigned_scale_values) & (unsigned_scale_values <= 3))
        )

        encoded_exponent_recipe = real_generation_recipe(
            hv.GenerationRecipe.random_encoded_exponent(
                hv.RandomEncodedExponentGenerationParameters(
                    lower_unbiased_exponent=-3,
                    upper_unbiased_exponent=-1,
                    source_type=hv.ScalarType.Float32,
                )
            ),
            seed=29,
        )
        narrow = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                encoded_exponent_recipe,
            )
        )
        narrow_repeat = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                encoded_exponent_recipe,
            )
        )
        np.testing.assert_array_equal(narrow, narrow_repeat)
        exponent_bits = (narrow.view(np.uint32) >> 23) & np.uint32(0xFF)
        self.assertTrue(
            set(int(value) for value in exponent_bits).issubset({124, 125, 126})
        )

        raw_serial = hv.generate_tensor(
            hv.ScalarType.Float16,
            [2, 3],
            real_generation_recipe(
                hv.GenerationRecipe.raw_serial_dimension(
                    hv.DimensionGenerationParameters(dimension=1)
                )
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_serial.storage, dtype=np.uint16).reshape(2, 3),
            np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint16),
        )

        raw_zero = hv.generate_tensor(
            hv.ScalarType.E8M0,
            [4],
            real_generation_recipe(
                hv.GenerationRecipe.raw_constant(
                    hv.RawConstantGenerationParameters(bits=0)
                )
            ),
        )
        np.testing.assert_array_equal(
            np.frombuffer(raw_zero.storage, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
        )

        raw_fp4 = hv.generate_tensor(
            hv.ScalarType.Float4E2M1,
            [65],
            real_generation_recipe(
                hv.GenerationRecipe.uniform_raw_integer(
                    hv.UniformIntegerGenerationParameters(lower=0, upper=14)
                ),
                seed=31,
            ),
        )
        fp4_nibbles = np.frombuffer(raw_fp4.storage, dtype=np.uint8)
        fp4_nibbles = np.concatenate(
            (fp4_nibbles & np.uint8(0xF), fp4_nibbles >> np.uint8(4))
        )[:65]
        self.assertTrue(np.all(fp4_nibbles <= 14))

        raw_bits_recipe = real_generation_recipe(
            hv.GenerationRecipe.random_raw_bits(),
            seed=41,
        )
        raw_bits = hv.generate_tensor(hv.ScalarType.UInt32, [32], raw_bits_recipe)
        raw_bits_repeat = hv.generate_tensor(
            hv.ScalarType.UInt32, [32], raw_bits_recipe
        )
        self.assertEqual(raw_bits.storage, raw_bits_repeat.storage)
        self.assertNotEqual(raw_bits.storage, bytes(len(raw_bits.storage)))

    def test_generation_recipe_modifiers(self):
        scaled_component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=1, upper=10)
        ).with_affine_value_mapping(
            hv.GenerationAffineValueParameters(scale=0.1, offset=2.0)
        )
        scaled = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                real_generation_recipe(scaled_component, seed=37),
            )
        )
        scaled_tenths = np.rint((scaled - 2.0) * 10).astype(np.int32)
        self.assertTrue(np.all((1 <= scaled_tenths) & (scaled_tenths <= 10)))
        np.testing.assert_allclose(
            scaled, 2.0 + scaled_tenths.astype(np.float32) / 10.0
        )

        positive_component = hv.GenerationRecipe.uniform_real(
            hv.UniformRealGenerationParameters(lower=-0.5, upper=0.5)
        ).with_absolute_transform()
        positive = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [64],
                real_generation_recipe(positive_component, seed=37),
            )
        )
        self.assertTrue(np.all((0 <= positive) & (positive <= 0.5)))

        constant_component = hv.GenerationRecipe.constant(
            hv.ConstantGenerationParameters(value=2.0)
        )
        alternating_component = constant_component.with_alternating_sign(
            hv.AlternatingSignGenerationParameters(
                dimensions=[0, 1],
                negative_when_odd=False,
            )
        )
        alternating = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3, 2],
                real_generation_recipe(alternating_component),
            )
        )
        expected_matrix = np.asarray([[-2, 2, -2], [2, -2, 2]], dtype=np.float32)
        np.testing.assert_array_equal(alternating[:, :, 0], expected_matrix)
        np.testing.assert_array_equal(alternating[:, :, 1], expected_matrix)

        opposite_component = constant_component.with_alternating_sign(
            hv.AlternatingSignGenerationParameters(
                dimensions=[0, 1],
                negative_when_odd=True,
            )
        )
        opposite = hv.to_numpy(
            hv.generate_tensor(
                hv.ScalarType.Float32,
                [2, 3],
                real_generation_recipe(opposite_component),
            )
        )
        np.testing.assert_array_equal(opposite, -expected_matrix)

    def test_mx_generation_matches_decoded_numpy_values(self):
        for dimensions, block_axis in (((64, 3), 0), ((3, 64), 1)):
            with self.subTest(dimensions=dimensions, block_axis=block_axis):
                problem = hv.MxGenerationProblem()
                problem.data_type = hv.ScalarType.Float4E2M1
                problem.scale_type = hv.ScalarType.E8M0
                problem.shape = hv.Shape(list(dimensions))
                problem.leading_dimension = dimensions[0]
                problem.block_axis = block_axis
                problem.block_size = 32
                problem.data = hv.MxDataRecipe.bounded(
                    hv.MxBoundedDataParameters(-1, 1)
                )

                first = hv.generate_mx(problem)
                second = hv.generate_mx(problem)
                self.assertEqual(first.data.storage, second.data.storage)
                self.assertEqual(first.scales.storage, second.scales.storage)

                data = hv.to_numpy(first.data)
                scales = hv.to_numpy(first.scales)
                scale_indices = hv.to_numpy(first.scale_indices, np.uint32)
                reference = hv.to_numpy(first.reference)
                expected = np.empty(dimensions, dtype=np.float32)
                for row in range(dimensions[0]):
                    for column in range(dimensions[1]):
                        scale_index = scale_indices[row, column]
                        expected[row, column] = data[row, column] * scales[scale_index]
                np.testing.assert_array_equal(reference, expected)

    def test_bounded_mx_generation_matches_independent_python_oracle(self):
        for dimensions, leading_dimension, block_axis, minimum, maximum in (
            ((9, 5), 12, 0, -1.0, 1.0),
            ((5, 9), 8, 1, -1.0, 1.0),
            ((9, 5), 12, 0, 0.0, 0.9),
        ):
            with self.subTest(
                dimensions=dimensions,
                block_axis=block_axis,
                minimum=minimum,
                maximum=maximum,
            ):
                problem = hv.MxGenerationProblem()
                problem.data_type = hv.ScalarType.Float4E2M1
                problem.scale_type = hv.ScalarType.E8M0
                problem.shape = hv.Shape(list(dimensions))
                problem.leading_dimension = leading_dimension
                problem.block_axis = block_axis
                problem.block_size = 4
                problem.seed = 12345
                problem.data = hv.MxDataRecipe.bounded(
                    hv.MxBoundedDataParameters(minimum, maximum)
                )

                observed = hv.generate_mx(problem)
                expected_data, expected_scales, expected_indices, expected_reference = (
                    bounded_mx_fp4_oracle(
                        dimensions,
                        leading_dimension,
                        block_axis,
                        problem.block_size,
                        problem.seed,
                        minimum,
                        maximum,
                    )
                )
                self.assertEqual(observed.data.storage, expected_data)
                self.assertEqual(observed.scales.storage, expected_scales)
                np.testing.assert_array_equal(
                    hv.to_numpy(observed.scale_indices, np.uint32),
                    expected_indices,
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(observed.reference, np.float32),
                    expected_reference,
                )
                self.assertTrue(
                    np.all(
                        (hv.to_numpy(observed.reference) >= minimum)
                        & (hv.to_numpy(observed.reference) <= maximum)
                    )
                )

    def test_mx_scale_policy_is_independent_of_data_recipe(self):
        problem = hv.MxGenerationProblem()
        problem.data_type = hv.ScalarType.Float4E2M1
        problem.scale_type = hv.ScalarType.E8M0
        problem.shape = hv.Shape([8, 8])
        problem.block_axis = 0
        problem.block_size = 4
        problem.data = hv.MxDataRecipe.bounded(hv.MxBoundedDataParameters(-1.0, 1.0))
        problem.scale = hv.MxScaleGenerationMode.One

        observed = hv.generate_mx(problem)
        np.testing.assert_array_equal(
            hv.to_numpy(observed.scales),
            np.ones(observed.scales.shape[0], dtype=np.float32),
        )

    def test_gemm_object_api_retains_owned_inputs_and_scaling(self):
        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pre_scale_a = np.asarray([2.0, 3.0], dtype=np.float32)
        pre_scale_b = np.asarray([0.5, 2.0], dtype=np.float32)
        scale_alpha = np.asarray([1.0, 2.0], dtype=np.float32)
        scale_a = np.asarray([2.0, 3.0], dtype=np.float32)
        scale_b = np.asarray([4.0, 5.0], dtype=np.float32)
        bias = np.asarray([1.0, -2.0], dtype=np.float32)

        def make_request():
            operand_a = hv.GemmOperand(hv.from_numpy(a_values))
            operand_a.pre_quantization_scales = [
                hv.VectorBinding(hv.from_numpy(pre_scale_a), hv.MatrixAxis.Row)
            ]
            operand_b = hv.GemmOperand(hv.from_numpy(b_values))
            operand_b.pre_quantization_scales = [
                hv.VectorBinding(hv.from_numpy(pre_scale_b), hv.MatrixAxis.Column)
            ]
            problem = hv.GemmProblem(
                operand_a,
                operand_b,
                hv.from_numpy(c_values),
                output_type=hv.ScalarType.Float32,
                accumulator_type=hv.ScalarType.Float32,
            )
            request = hv.GemmRequest(
                problem,
                hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 2])),
            )
            request.epilogue.alpha = 0.5
            request.epilogue.beta = -1.0
            request.epilogue.scale_alpha = hv.VectorBinding(
                hv.from_numpy(scale_alpha), hv.MatrixAxis.Row
            )
            request.epilogue.scale_a = hv.from_numpy(scale_a)
            request.epilogue.scale_b = hv.from_numpy(scale_b)
            request.epilogue.bias = hv.VectorBinding(
                hv.from_numpy(bias), hv.MatrixAxis.Row
            )
            request.epilogue.output_scale = 0.25
            return request

        request = make_request()
        gc.collect()

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

        result = hv.reference_gemm_result(request)
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)
        np.testing.assert_array_equal(hv.to_numpy(hv.reference_gemm(request)), expected)
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Pointwise)

    def test_gemm_object_api_requires_explicit_c_and_d(self):
        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0], [6.0]], dtype=np.float32)
        operand_a = hv.GemmOperand(hv.from_numpy(a_values))
        operand_b = hv.GemmOperand(hv.from_numpy(b_values))
        request = hv.GemmRequest(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 1])),
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 1])),
            accumulator_type=hv.ScalarType.Float32,
        )

        np.testing.assert_array_equal(
            hv.to_numpy(hv.reference_gemm(request)), a_values @ b_values
        )

        request.epilogue.beta = 1.0
        np.testing.assert_array_equal(
            hv.to_numpy(hv.reference_gemm(request)), a_values @ b_values
        )
        with self.assertRaises(TypeError):
            hv.GemmRequest(operand_a, operand_b)

    def test_gemm_object_api_operand_quantization_and_block_scales(self):
        operand_a = hv.GemmOperand(
            hv.from_numpy(np.full((1, 8), 1.5, dtype=np.float32))
        )
        operand_a.compute_type = hv.ScalarType.Float16
        operand_a.pre_quantization_scales = [
            hv.VectorBinding(
                hv.from_numpy(np.asarray([2.0], dtype=np.float32)),
                hv.MatrixAxis.Row,
            )
        ]
        operand_a.block_scale = hv.BlockScaleBinding(
            hv.from_numpy(np.asarray([[2.0, 4.0]], dtype=np.float32)), 4
        )

        operand_b = hv.GemmOperand(
            hv.from_numpy(np.full((8, 1), 2.0, dtype=np.float32))
        )
        operand_b.compute_type = hv.ScalarType.BFloat16
        operand_b.pre_quantization_scales = [
            hv.VectorBinding(
                hv.from_numpy(np.asarray([0.5], dtype=np.float32)),
                hv.MatrixAxis.Column,
            )
        ]
        operand_b.block_scale = hv.BlockScaleBinding(
            hv.from_numpy(np.asarray([[3.0, 5.0]], dtype=np.float32)), 4
        )

        request = hv.GemmRequest(
            operand_a,
            operand_b,
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([1, 1])),
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([1, 1])),
            accumulator_type=hv.ScalarType.Float32,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(hv.reference_gemm(request)),
            np.asarray([[312.0]], dtype=np.float32),
        )

    def test_gemm_object_api_conjugates_operands(self):
        a_values = np.asarray(
            [[1.0 + 2.0j, 3.0 - 4.0j], [-2.0 + 0.5j, 1.5 + 3.0j]],
            dtype=np.complex64,
        )
        b_values = np.asarray([[2.0 - 1.0j], [0.5 + 2.0j]], dtype=np.complex64)
        operand_a = hv.GemmOperand(hv.from_numpy(a_values))
        operand_a.conjugate = True
        request = hv.GemmRequest(
            operand_a,
            hv.GemmOperand(hv.from_numpy(b_values)),
            hv.Tensor(hv.ScalarType.ComplexFloat32, hv.Shape([2, 1])),
            hv.Tensor(hv.ScalarType.ComplexFloat32, hv.Shape([2, 1])),
            accumulator_type=hv.ScalarType.ComplexFloat32,
        )

        np.testing.assert_allclose(
            hv.to_numpy(hv.reference_gemm(request)),
            np.conjugate(a_values) @ b_values,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gemm_object_api_allocates_affine_output_for_blocked_execution(self):
        a_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_values = np.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
        output_layout = hv.Layout(hv.Shape([2, 3]), [9, 2], 1)
        problem = hv.GemmProblem(
            hv.GemmOperand(hv.from_numpy(a_values)),
            hv.GemmOperand(hv.from_numpy(b_values)),
            hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3])),
            output_type=hv.ScalarType.Float32,
            accumulator_type=hv.ScalarType.Float32,
        )
        output = hv.GemmOutputOptions()
        output.layout = output_layout
        execution = hv.GemmExecution(hv.GemmBackend.Blocked, True)

        result = hv.reference_gemm_result(problem, output, execution)
        expected = a_values @ b_values
        np.testing.assert_array_equal(hv.to_numpy(result.output), expected)
        self.assertEqual(result.output.strides, [9, 2])
        self.assertEqual(result.output.offset, 1)
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Blocked)
        self.assertEqual(result.run_info.output_elements_written, expected.size)
        self.assertEqual(result.run_info.output_elements_covered, expected.size)

        automatic = hv.reference_gemm_result(
            problem, output, hv.GemmExecution(hv.GemmBackend.Automatic)
        )
        np.testing.assert_array_equal(hv.to_numpy(automatic.output), expected)
        self.assertEqual(automatic.run_info.backend_used, hv.GemmBackend.Blocked)

        storage = np.frombuffer(result.output.storage, dtype=np.float32)
        expected_storage = np.zeros(15, dtype=np.float32)
        expected_storage[[1, 3, 5, 10, 12, 14]] = expected.reshape(-1)
        np.testing.assert_array_equal(storage, expected_storage)

    def test_float32_gemm_matches_numpy(self):
        a = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
        b = np.arange(20, dtype=np.float32).reshape(5, 4) - 7
        c = np.arange(12, dtype=np.float32).reshape(3, 4)
        observed = hv.reference_gemm_flat(
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
        blocked_result = hv.reference_gemm_flat_result(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(hv.to_numpy(blocked_result.output), expected)
        self.assertEqual(blocked_result.run_info.backend_used, hv.GemmBackend.Blocked)
        self.assertIsNone(blocked_result.run_info.fallback_reason)
        self.assertEqual(blocked_result.run_info.output_elements_written, expected.size)
        self.assertEqual(blocked_result.run_info.output_elements_covered, expected.size)

    def test_zero_gemm_scalars_suppress_non_finite_operands(self):
        finite_c = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        finite_a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        finite_b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                alpha_zero = hv.reference_gemm_flat(
                    hv.from_numpy(np.full((2, 2), np.nan, dtype=np.float32)),
                    hv.from_numpy(np.full((2, 2), np.inf, dtype=np.float32)),
                    hv.from_numpy(finite_c),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    alpha=0.0,
                    beta=2.0,
                    backend=backend,
                )
                np.testing.assert_array_equal(hv.to_numpy(alpha_zero), 2.0 * finite_c)

                beta_zero = hv.reference_gemm_flat(
                    hv.from_numpy(finite_a),
                    hv.from_numpy(finite_b),
                    hv.from_numpy(np.full((2, 2), np.inf, dtype=np.float32)),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    alpha=1.0,
                    beta=0.0,
                    backend=backend,
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(beta_zero), finite_a @ finite_b
                )

    def test_gemm_finalization_matches_numpy_on_both_backends(self):
        a = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [-7.0, 8.0]], dtype=np.float32)
        c = np.asarray([[2.0, -3.0], [4.0, 5.0]], dtype=np.float32)
        alpha = np.float32(0.75)
        beta = np.float32(-0.5)
        output_scale = np.float32(1.25)

        accumulation = matmul_float32(a, b)
        combined = np.float32(np.float32(alpha * accumulation) + np.float32(beta * c))
        expected = np.float32(np.maximum(combined, np.float32(0.0)) * output_scale)

        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm_flat(
                    hv.from_numpy(a),
                    hv.from_numpy(b),
                    hv.from_numpy(c),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    alpha=float(alpha),
                    beta=float(beta),
                    activation=hv.Activation.Relu,
                    output_scale=float(output_scale),
                    backend=backend,
                )
                np.testing.assert_array_equal(hv.to_numpy(observed), expected)

    def test_float64_gemm_matches_numpy(self):
        a = np.asarray([[0.25, -1.5], [2.0, 3.25]], dtype=np.float64)
        b = np.asarray([[4.0, 0.5], [-2.0, 1.25]], dtype=np.float64)
        c = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        observed = hv.reference_gemm_flat(
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

    def test_int32_accumulator_gemm_matches_exact_integer_oracle(self):
        a = np.asarray([[1, 3], [2, 4]], dtype=np.int8)
        b = np.asarray([[5], [6]], dtype=np.int8)
        c = np.zeros((2, 1), dtype=np.int32)
        observed = hv.reference_gemm_flat(
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
        observed = hv.reference_gemm_flat(
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

    def test_int32_accumulator_rejects_fractional_scalar_proxy(self):
        values = np.ones((1, 1), dtype=np.int8)
        initial = np.zeros((1, 1), dtype=np.int32)
        with self.assertRaises(ValueError):
            hv.reference_gemm_flat(
                hv.from_numpy(values),
                hv.from_numpy(values),
                hv.from_numpy(initial),
                hv.ScalarType.Int32,
                hv.ScalarType.Int32,
                alpha=0.5,
            )

    def test_float16_accumulator_rounds_each_step(self):
        a = np.full((1, 64), np.float16(0.1), dtype=np.float16)
        b = np.full((64, 1), np.float16(0.1), dtype=np.float16)
        c = np.zeros((1, 1), dtype=np.float16)
        result = hv.reference_gemm_flat_result(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float16,
            hv.ScalarType.Float16,
            backend=hv.GemmBackend.Automatic,
        )

        expected = np.float16(0)
        for reduction in range(a.shape[1]):
            product = np.float16(a[0, reduction] * b[reduction, 0])
            expected = np.float16(expected + product)
        np.testing.assert_array_equal(
            hv.to_numpy(result.output, np.float32),
            np.asarray([[expected]], dtype=np.float32),
        )
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Pointwise)
        self.assertIsNotNone(result.run_info.fallback_reason)

    def test_bfloat16_accumulator_rounds_product_and_sum_each_step(self):
        a = np.full((1, 16), np.float32(0.1), dtype=np.float32)
        b = np.full((16, 1), np.float32(0.1), dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        observed = hv.reference_gemm_flat(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
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
        c = np.zeros((1, 1), dtype=np.float32)

        rounded = hv.reference_gemm_flat(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.BFloat16,
            accumulation_rounding=hv.AccumulationRounding.AfterProductAndSum,
        )
        full_precision = hv.reference_gemm_flat(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
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
        observed = hv.reference_gemm_flat(
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
        b = np.asarray([[2.0 - 1.0j], [0.5 + 3.0j]], dtype=np.complex64)
        c = np.asarray([[1.0j], [2.0 - 1.0j]], dtype=np.complex64)
        alpha = 0.5 + 0.25j
        beta = -1.0 + 0.5j
        observed = hv.reference_gemm_flat(
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
        observed = hv.reference_gemm_flat(
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

        pre_scaled = hv.reference_gemm_flat(
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

        vector_pre_scaled = hv.reference_gemm_flat(
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
            pre_quantization_axes_a=[hv.MatrixAxis.Row],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(vector_pre_scaled),
            np.asarray([[3.25], [4.5]], dtype=np.float32),
        )

        combined_pre_scaled = hv.reference_gemm_flat(
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

    def test_compute_input_quantization_matches_numpy_on_both_backends(self):
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
                hv.from_numpy(scale_a_row),
                hv.from_numpy(scale_a_reduction),
            ],
            pre_quantization_axes_a=[
                hv.MatrixAxis.Row,
                hv.MatrixAxis.Column,
            ],
            pre_quantization_scales_b=[
                hv.from_numpy(scale_b_reduction),
                hv.from_numpy(scale_b_column),
            ],
            pre_quantization_axes_b=[
                hv.MatrixAxis.Row,
                hv.MatrixAxis.Column,
            ],
        )
        backend_outputs = []
        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm_flat(
                    hv.from_numpy(a),
                    hv.from_numpy(b),
                    hv.from_numpy(c),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    backend=backend,
                    **arguments,
                )
                output = hv.to_numpy(observed)
                np.testing.assert_array_equal(output, expected)
                backend_outputs.append(output)
        np.testing.assert_array_equal(backend_outputs[0], backend_outputs[1])

    def test_gemm_output_scale_and_saturating_conversion(self):
        scaled_half = hv.reference_gemm_flat(
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

        saturated_int8 = hv.reference_gemm_flat(
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

    def test_saturating_int8_rounding_matches_numpy_on_both_backends(self):
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

        backend_outputs = []
        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm_flat(
                    hv.from_numpy(a),
                    hv.from_numpy(b),
                    hv.from_numpy(c),
                    hv.ScalarType.Int8,
                    hv.ScalarType.Float32,
                    backend=backend,
                    output_conversion=hv.OutputConversion.SaturatingInt8,
                )
                output = hv.to_numpy(observed)
                np.testing.assert_array_equal(output, expected)
                backend_outputs.append(output)
        np.testing.assert_array_equal(backend_outputs[0], backend_outputs[1])

    def test_block_scaled_gemm_matches_numpy_on_both_backends(self):
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

        backend_outputs = []
        for backend in (hv.GemmBackend.Pointwise, hv.GemmBackend.Blocked):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm_flat(
                    hv.from_numpy(a),
                    hv.from_numpy(b),
                    hv.from_numpy(c),
                    hv.ScalarType.Float32,
                    hv.ScalarType.Float32,
                    backend=backend,
                    block_scale_a=hv.from_numpy(scale_a),
                    block_scale_b=hv.from_numpy(scale_b),
                    block_size_a=8,
                    block_size_b=8,
                )
                output = hv.to_numpy(observed)
                np.testing.assert_array_equal(output, expected)
                backend_outputs.append(output)
        np.testing.assert_array_equal(backend_outputs[0], backend_outputs[1])

    def test_selected_output_gemm(self):
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c = np.zeros((2, 2), dtype=np.float32)
        observed = hv.reference_gemm_flat(
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
        blocked = hv.reference_gemm_flat_result(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            output_selection=hv.OutputSelection.explicit_indices([0, 3]),
            backend=hv.GemmBackend.Blocked,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(blocked.output),
            np.asarray([[19.0, 0.0], [0.0, 50.0]], dtype=np.float32),
        )
        self.assertEqual(blocked.run_info.output_elements_written, 2)
        self.assertEqual(blocked.run_info.output_elements_covered, 4)
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
        np.testing.assert_array_equal(hv.to_numpy(result.raw_output), activated * 2.0)
        np.testing.assert_array_equal(
            hv.to_numpy(result.auxiliary_output), pre_activation * 3.0
        )
        np.testing.assert_array_equal(
            hv.to_numpy(result.amax), np.asarray([5.0], dtype=np.float32)
        )
        self.assertEqual(result.run_info.output_elements_written, 4)
        self.assertEqual(result.run_info.raw_output_elements_written, 4)
        self.assertEqual(result.run_info.auxiliary_output_elements_written, 4)
        self.assertEqual(result.run_info.amax_elements_written, 1)

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
        self.assertEqual(selected.run_info.output_elements_written, 2)
        self.assertEqual(selected.run_info.raw_output_elements_written, 2)

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

    def test_native_operation_problem_and_request_bindings(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        input_tensor = hv.from_numpy(values)
        reduction_problem = hv.ReductionProblem(
            input_tensor,
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            [1],
            hv.ReductionOperation.Sum,
        )
        reduction_result = hv.reference_reduce(reduction_problem)
        np.testing.assert_array_equal(
            hv.to_numpy(reduction_result.output),
            np.sum(values, axis=1, dtype=np.float32),
        )
        self.assertEqual(reduction_result.run_info.output_elements_written, 2)
        self.assertEqual(reduction_result.run_info.input_elements_read, 6)

        reduction_output = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2]))
        reduction_request = hv.ReductionRequest(
            reduction_problem,
            reduction_output,
        )
        reduction_run = hv.reference_reduce(reduction_request)
        self.assertEqual(reduction_run.output_elements_written, 2)
        np.testing.assert_array_equal(
            hv.to_numpy(reduction_output),
            np.sum(values, axis=1, dtype=np.float32),
        )

        layer_norm_problem = hv.LayerNormProblem(
            input_tensor,
            hv.ScalarType.Float32,
            1,
            hv.ScalarType.Float32,
        )
        layer_norm_result = hv.reference_layer_norm(layer_norm_problem)
        self.assertIsNone(layer_norm_result.mean)
        self.assertIsNone(layer_norm_result.inverse_variance)

        pattern = hv.StructuredSparsityPattern()
        pattern.axis = 0
        sparse_problem = hv.StructuredSparsityProblem(
            hv.from_numpy(np.arange(1, 9, dtype=np.float32)),
            pattern,
        )
        sparse_result = hv.apply_structured_sparsity(sparse_problem)
        self.assertIsNone(sparse_result.retained_indices)
        self.assertIsNone(sparse_result.two_of_four_metadata)

    def test_structured_sparsity_all_fixed_two_of_four_patterns(self):
        values = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
        retained_position_sets = (
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 3),
        )

        for retained_positions in retained_position_sets:
            with self.subTest(retained_positions=retained_positions):
                pattern = hv.StructuredSparsityPattern()
                pattern.axis = 1
                pattern.fixed_positions = list(retained_positions)
                result = hv.apply_structured_sparsity(
                    hv.from_numpy(values), pattern, True
                )

                expected_pruned = np.zeros_like(values)
                expected_compressed = np.empty((2, 4), dtype=np.float32)
                expected_indices = np.empty((2, 4), dtype=np.uint8)
                for row in range(values.shape[0]):
                    for group in range(2):
                        source = values[row, group * 4 : (group + 1) * 4]
                        for retained_index, position in enumerate(retained_positions):
                            expected_pruned[row, group * 4 + position] = source[
                                position
                            ]
                            expected_compressed[row, group * 2 + retained_index] = (
                                source[position]
                            )
                            expected_indices[row, group * 2 + retained_index] = position

                np.testing.assert_array_equal(
                    hv.to_numpy(result.pruned), expected_pruned
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.compressed), expected_compressed
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.retained_indices), expected_indices
                )
                metadata = hv.encode_two_of_four_metadata(result.retained_indices, 1)
                nibble = retained_positions[0] | (retained_positions[1] << 2)
                np.testing.assert_array_equal(
                    hv.to_numpy(metadata.metadata),
                    np.full(
                        (2, 1),
                        nibble | (nibble << 4),
                        dtype=np.uint8,
                    ),
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.two_of_four_metadata),
                    hv.to_numpy(metadata.metadata),
                )
                self.assertEqual(metadata.run_info.sparsity_groups_encoded, 4)
                self.assertEqual(result.run_info.groups_processed, 4)
                self.assertEqual(result.run_info.compressed_elements_written, 8)

    def test_structured_sparsity_random_is_deterministic_and_self_consistent(self):
        values = np.arange(1, 65, dtype=np.float32).reshape(4, 16)
        pattern = hv.StructuredSparsityPattern()
        pattern.axis = 1
        pattern.selection = hv.StructuredSparsitySelection.Random
        pattern.seed = 0x12345678

        first = hv.apply_structured_sparsity(hv.from_numpy(values), pattern, True)
        second = hv.apply_structured_sparsity(hv.from_numpy(values), pattern, True)
        np.testing.assert_array_equal(
            hv.to_numpy(first.pruned), hv.to_numpy(second.pruned)
        )
        np.testing.assert_array_equal(
            hv.to_numpy(first.compressed), hv.to_numpy(second.compressed)
        )
        np.testing.assert_array_equal(
            hv.to_numpy(first.retained_indices),
            hv.to_numpy(second.retained_indices),
        )

        pruned = hv.to_numpy(first.pruned)
        compressed = hv.to_numpy(first.compressed)
        retained_indices = hv.to_numpy(first.retained_indices)
        metadata = hv.to_numpy(first.two_of_four_metadata)
        observed_position_sets = set()
        for row in range(values.shape[0]):
            for group in range(values.shape[1] // 4):
                positions = retained_indices[row, group * 2 : (group + 1) * 2]
                self.assertLess(positions[0], positions[1])
                observed_position_sets.add(tuple(int(x) for x in positions))
                expected_group = np.zeros(4, dtype=np.float32)
                for retained_index, position in enumerate(positions):
                    expected_group[position] = values[row, group * 4 + position]
                    self.assertEqual(
                        compressed[row, group * 2 + retained_index],
                        values[row, group * 4 + position],
                    )
                np.testing.assert_array_equal(
                    pruned[row, group * 4 : (group + 1) * 4],
                    expected_group,
                )
                nibble = int(positions[0]) | (int(positions[1]) << 2)
                metadata_byte = metadata[row, group // 2]
                observed_nibble = (
                    metadata_byte & 0xF if group % 2 == 0 else metadata_byte >> 4
                )
                self.assertEqual(observed_nibble, nibble)
        self.assertGreater(len(observed_position_sets), 1)

    def test_structured_sparsity_handles_strided_input_and_packed_values(self):
        storage = np.full(20, -99.0, dtype=np.float32)
        logical = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
        storage[0:8] = logical[0]
        storage[10:18] = logical[1]
        strided = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 8],
            storage.tobytes(),
            [10, 1],
        )
        pattern = hv.StructuredSparsityPattern()
        pattern.axis = 1
        pattern.fixed_positions = [1, 3]
        observed = hv.apply_structured_sparsity(strided, pattern)
        expected = np.zeros_like(logical)
        expected[:, 1::4] = logical[:, 1::4]
        expected[:, 3::4] = logical[:, 3::4]
        np.testing.assert_array_equal(hv.to_numpy(observed.pruned), expected)
        self.assertEqual(observed.pruned.strides, [8, 1])
        self.assertEqual(observed.compressed.strides, [4, 1])
        self.assertEqual(observed.retained_indices.strides, [4, 1])

        fp4_values = np.asarray(
            [[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5]],
            dtype=np.float32,
        )
        fp4 = hv.apply_structured_sparsity(
            hv.from_numpy(fp4_values, hv.ScalarType.Float4E2M1), pattern
        )
        expected_fp4 = np.zeros_like(fp4_values)
        expected_fp4[:, 1::4] = fp4_values[:, 1::4]
        expected_fp4[:, 3::4] = fp4_values[:, 3::4]
        np.testing.assert_array_equal(hv.to_numpy(fp4.pruned), expected_fp4)


if __name__ == "__main__":
    unittest.main()
