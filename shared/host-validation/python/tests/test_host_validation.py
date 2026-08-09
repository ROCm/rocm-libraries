# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gc
import math
import unittest
import weakref

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

    def test_numpy_tensor_view_positive_stride_and_gaps(self):
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
        view = hv.TensorView.from_numpy(values)
        expected = hv.from_numpy(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ).view()

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

    def test_numpy_tensor_view_negative_strides(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.TensorView.from_numpy(values)

        self.assertEqual(view.shape, [4, 3])
        self.assertEqual(view.strides, [-6, -2])
        self.assertEqual(view.offset, 22)
        self.assertEqual(len(view.storage), 23 * values.itemsize)
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertTrue(hv.compare(view, hv.from_numpy(values).view()).passed)

    def test_numpy_tensor_view_observes_mutation(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        view = hv.TensorView.from_numpy(values)
        expected = hv.from_numpy(values.copy()).view()
        self.assertTrue(hv.compare(view, expected).passed)

        values[1, 2] = -17.0
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertFalse(hv.compare(view, expected).passed)

    def test_numpy_tensor_view_retains_owner(self):
        values = np.arange(6, dtype=np.float64)
        owner = weakref.ref(values)
        view = hv.TensorView.from_numpy(values)

        del values
        gc.collect()
        self.assertIsNotNone(owner())
        np.testing.assert_array_equal(hv.to_numpy(view), np.arange(6, dtype=np.float64))

        del view
        gc.collect()
        self.assertIsNone(owner())

    def test_owning_tensor_view_retains_tensor(self):
        tensor = hv.from_numpy(np.arange(6, dtype=np.float32))
        view = tensor.view()

        del tensor
        gc.collect()
        np.testing.assert_array_equal(hv.to_numpy(view), np.arange(6, dtype=np.float32))

    def test_numpy_tensor_view_accepts_read_only_array(self):
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        values.flags.writeable = False
        view = hv.TensorView.from_numpy(values)

        self.assertEqual(view.type, hv.ScalarType.Int32)
        np.testing.assert_array_equal(hv.to_numpy(view), values)

    def test_numpy_tensor_view_rejects_storage_conversion(self):
        with self.assertRaises(TypeError):
            hv.TensorView.from_numpy([1.0, 2.0])
        with self.assertRaises(TypeError):
            hv.TensorView.from_numpy(np.arange(4, dtype=np.dtype(">f4")))
        with self.assertRaises(ValueError):
            hv.TensorView.from_numpy(
                np.arange(4, dtype=np.float64),
                hv.ScalarType.Float32,
            )
        with self.assertRaises(ValueError):
            hv.TensorView.from_numpy(
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
            hv.TensorView.from_numpy(byte_strided)

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
        first = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3]))
        second = hv.Tensor(hv.ScalarType.Float32, hv.Shape([2, 3]))
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
        selected.selection.index_order = hv.ComparisonIndexOrder.FirstDimensionFastest
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
        np.testing.assert_array_equal(
            hv.to_numpy(identity), np.eye(3, 4, dtype=np.float32)
        )

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

    def test_type_derived_generation(self):
        options = hv.GenerationOptions()
        options.real.pattern = hv.GenerationPattern.TypeMaximum
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
                    hv.generate_tensor(scalar_type, [3], options),
                    np.float64 if scalar_type == hv.ScalarType.Float64 else np.float32,
                )
                np.testing.assert_array_equal(
                    observed, np.full(3, expected, dtype=observed.dtype)
                )

        options.real.pattern = hv.GenerationPattern.TypeDenormalMinimum
        fp4_denormal = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float4E2M1, [2], options)
        )
        np.testing.assert_array_equal(
            fp4_denormal, np.asarray([0.5, 0.5], dtype=np.float32)
        )

        options.real.pattern = hv.GenerationPattern.TypeNaN
        nan_values = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float8E4M3Fnuz, [2], options)
        )
        self.assertTrue(np.isnan(nan_values).all())

        options.real.pattern = hv.GenerationPattern.TypeInfinity
        infinity = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float8E5M2, [2], options)
        )
        self.assertTrue(np.isposinf(infinity).all())

        options.real.pattern = hv.GenerationPattern.UniformTypeRange
        options.seed = 23
        low_precision = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float4E2M1, [64], options)
        )
        self.assertTrue(np.all((-6.0 <= low_precision) & (low_precision <= 6.0)))
        float64_range = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float64, [64], options)
        )
        self.assertTrue(np.isfinite(float64_range).all())
        self.assertTrue(
            np.all(
                (-np.finfo(np.float64).max <= float64_range)
                & (float64_range <= np.finfo(np.float64).max)
            )
        )

        options.real.pattern = hv.GenerationPattern.AbsoluteUniformInteger
        options.real.parameter0 = -3
        options.real.parameter1 = 3
        unsigned_scale_values = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.E5M3, [64], options)
        )
        self.assertTrue(
            np.all((0 <= unsigned_scale_values) & (unsigned_scale_values <= 3))
        )

        options.real.pattern = hv.GenerationPattern.RandomEncodedExponent
        options.real.parameter0 = -3
        options.real.parameter1 = -1
        options.real.source_type = hv.ScalarType.Float32
        options.seed = 29
        narrow = hv.to_numpy(hv.generate_tensor(hv.ScalarType.Float32, [64], options))
        narrow_repeat = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [64], options)
        )
        np.testing.assert_array_equal(narrow, narrow_repeat)
        exponent_bits = (narrow.view(np.uint32) >> 23) & np.uint32(0xFF)
        self.assertTrue(
            set(int(value) for value in exponent_bits).issubset({124, 125, 126})
        )

        options.real.pattern = hv.GenerationPattern.RawSerialDimension
        options.real.dimension = 1
        raw_serial = hv.generate_tensor(hv.ScalarType.Float16, [2, 3], options)
        np.testing.assert_array_equal(
            np.frombuffer(raw_serial.storage, dtype=np.uint16).reshape(2, 3),
            np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint16),
        )

        options.real.pattern = hv.GenerationPattern.RawConstant
        options.real.parameter0 = 0
        raw_zero = hv.generate_tensor(hv.ScalarType.E8M0, [4], options)
        np.testing.assert_array_equal(
            np.frombuffer(raw_zero.storage, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
        )

        options.real.pattern = hv.GenerationPattern.UniformRawInteger
        options.real.parameter0 = 0
        options.real.parameter1 = 14
        options.seed = 31
        raw_fp4 = hv.generate_tensor(hv.ScalarType.Float4E2M1, [65], options)
        fp4_nibbles = np.frombuffer(raw_fp4.storage, dtype=np.uint8)
        fp4_nibbles = np.concatenate(
            (fp4_nibbles & np.uint8(0xF), fp4_nibbles >> np.uint8(4))
        )[:65]
        self.assertTrue(np.all(fp4_nibbles <= 14))

        options.real.pattern = hv.GenerationPattern.RandomRawBits
        options.seed = 41
        raw_bits = hv.generate_tensor(hv.ScalarType.UInt32, [32], options)
        raw_bits_repeat = hv.generate_tensor(hv.ScalarType.UInt32, [32], options)
        self.assertEqual(raw_bits.storage, raw_bits_repeat.storage)
        self.assertNotEqual(raw_bits.storage, bytes(len(raw_bits.storage)))

    def test_generation_recipe_modifiers(self):
        options = hv.GenerationOptions()
        options.real.pattern = hv.GenerationPattern.UniformInteger
        options.real.parameter0 = 1
        options.real.parameter1 = 10
        options.real.value_scale = 0.1
        options.real.value_offset = 2.0
        options.seed = 37
        scaled = hv.to_numpy(hv.generate_tensor(hv.ScalarType.Float32, [64], options))
        scaled_tenths = np.rint((scaled - 2.0) * 10).astype(np.int32)
        self.assertTrue(np.all((1 <= scaled_tenths) & (scaled_tenths <= 10)))
        np.testing.assert_allclose(
            scaled, 2.0 + scaled_tenths.astype(np.float32) / 10.0
        )

        options.real.pattern = hv.GenerationPattern.UniformReal
        options.real.parameter0 = -0.5
        options.real.parameter1 = 0.5
        options.real.value_scale = 1.0
        options.real.value_offset = 0.0
        options.real.transform = hv.GenerationTransform.Absolute
        positive = hv.to_numpy(hv.generate_tensor(hv.ScalarType.Float32, [64], options))
        self.assertTrue(np.all((0 <= positive) & (positive <= 0.5)))

        options.real.pattern = hv.GenerationPattern.Constant
        options.real.parameter0 = 2.0
        options.real.transform = hv.GenerationTransform.Identity
        options.real.alternating_dimensions = [0, 1]
        alternating = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 3, 2], options)
        )
        expected_matrix = np.asarray([[-2, 2, -2], [2, -2, 2]], dtype=np.float32)
        np.testing.assert_array_equal(alternating[:, :, 0], expected_matrix)
        np.testing.assert_array_equal(alternating[:, :, 1], expected_matrix)

        options.real.negative_parity = 1
        opposite = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 3], options)
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
                recipe = hv.MxGenerationRecipe()
                recipe.mode = hv.MxGenerationMode.Bounded
                recipe.parameter0 = -1
                recipe.parameter1 = 1
                problem.data = recipe

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
        tiled_result = hv.reference_gemm_result(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.Float32,
            alpha=2.0,
            beta=-1.0,
            backend=hv.GemmBackend.Tiled,
        )
        np.testing.assert_array_equal(hv.to_numpy(tiled_result.output), expected)
        self.assertEqual(tiled_result.run_info.backend_used, hv.GemmBackend.Tiled)
        self.assertIsNone(tiled_result.run_info.fallback_reason)
        self.assertEqual(tiled_result.run_info.output_elements_computed, expected.size)

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
        result = hv.reference_gemm_result(
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
        self.assertEqual(result.run_info.backend_used, hv.GemmBackend.Canonical)
        self.assertIsNotNone(result.run_info.fallback_reason)

    def test_bfloat16_accumulator_rounds_product_and_sum_each_step(self):
        a = np.full((1, 16), np.float32(0.1), dtype=np.float32)
        b = np.full((16, 1), np.float32(0.1), dtype=np.float32)
        c = np.zeros((1, 1), dtype=np.float32)
        observed = hv.reference_gemm(
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

        rounded = hv.reference_gemm(
            hv.from_numpy(a),
            hv.from_numpy(b),
            hv.from_numpy(c),
            hv.ScalarType.Float32,
            hv.ScalarType.BFloat16,
            accumulation_rounding=hv.AccumulationRounding.AfterProductAndSum,
        )
        full_precision = hv.reference_gemm(
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
        b = np.asarray([[2.0 - 1.0j], [0.5 + 3.0j]], dtype=np.complex64)
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
            pre_quantization_scales_a=[
                hv.from_numpy(np.asarray([3.0, 4.0], dtype=np.float32))
            ],
            pre_quantization_axes_a=[hv.MatrixAxis.Row],
        )
        np.testing.assert_array_equal(
            hv.to_numpy(vector_pre_scaled),
            np.asarray([[3.25], [4.5]], dtype=np.float32),
        )

        combined_pre_scaled = hv.reference_gemm(
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
        for backend in (hv.GemmBackend.Canonical, hv.GemmBackend.Tiled):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm(
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
        scaled_half = hv.reference_gemm(
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

        saturated_int8 = hv.reference_gemm(
            hv.from_numpy(np.asarray([[63.75]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[2.0]], dtype=np.float32)),
            hv.from_numpy(np.asarray([[0]], dtype=np.int8)),
            hv.ScalarType.Int8,
            hv.ScalarType.Float32,
            output_conversion=hv.GemmOutputConversion.SaturatingInt8,
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
        for backend in (hv.GemmBackend.Canonical, hv.GemmBackend.Tiled):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm(
                    hv.from_numpy(a),
                    hv.from_numpy(b),
                    hv.from_numpy(c),
                    hv.ScalarType.Int8,
                    hv.ScalarType.Float32,
                    backend=backend,
                    output_conversion=hv.GemmOutputConversion.SaturatingInt8,
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
        for backend in (hv.GemmBackend.Canonical, hv.GemmBackend.Tiled):
            with self.subTest(backend=backend):
                observed = hv.reference_gemm(
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
        pattern.stream = 7

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
