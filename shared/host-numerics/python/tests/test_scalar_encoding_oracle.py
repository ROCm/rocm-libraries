# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

import roc_host_numerics as hv


@dataclass(frozen=True)
class BinaryFormat:
    """Published encoding facts used without consulting host-numerics metadata."""

    name: str
    scalar_type: object
    storage_bits: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    maximum_finite_raw: int
    finite_overflow_raw: int
    infinity_raw: Optional[int]
    nan_raw: Optional[int]
    signed: bool = True
    signed_zero: bool = True

    @property
    def sign_mask(self):
        return 1 << (self.storage_bits - 1) if self.signed else 0


# IEEE formats use infinity for finite overflow. OCP E5M2 deliberately differs:
# finite overflow saturates, while an infinity input retains the infinity code.
BINARY_FORMATS = (
    BinaryFormat(
        "fp16",
        hv.ScalarType.Float16,
        16,
        5,
        10,
        15,
        0x7BFF,
        0x7C00,
        0x7C00,
        0x7E00,
    ),
    BinaryFormat(
        "bf16",
        hv.ScalarType.BFloat16,
        16,
        8,
        7,
        127,
        0x7F7F,
        0x7F80,
        0x7F80,
        0x7FC0,
    ),
    BinaryFormat(
        "fp4_e2m1",
        hv.ScalarType.Float4E2M1,
        4,
        2,
        1,
        1,
        0x7,
        0x7,
        None,
        None,
    ),
    BinaryFormat(
        "fp6_e2m3",
        hv.ScalarType.Float6E2M3,
        6,
        2,
        3,
        1,
        0x1F,
        0x1F,
        None,
        None,
    ),
    BinaryFormat(
        "fp6_e3m2",
        hv.ScalarType.Float6E3M2,
        6,
        3,
        2,
        3,
        0x1F,
        0x1F,
        None,
        None,
    ),
    BinaryFormat(
        "ocp_fp8_e4m3",
        hv.ScalarType.Float8E4M3,
        8,
        4,
        3,
        7,
        0x7E,
        0x7E,
        None,
        0x7F,
    ),
    BinaryFormat(
        "ocp_fp8_e5m2",
        hv.ScalarType.Float8E5M2,
        8,
        5,
        2,
        15,
        0x7B,
        0x7B,
        0x7C,
        0x7F,
    ),
    BinaryFormat(
        "e4m3_scale",
        hv.ScalarType.E4M3,
        8,
        4,
        3,
        7,
        0x7E,
        0x7E,
        None,
        0x7F,
        signed=False,
        signed_zero=False,
    ),
    BinaryFormat(
        "e5m3_scale",
        hv.ScalarType.E5M3,
        8,
        5,
        3,
        15,
        0xFE,
        0xFE,
        None,
        0xFF,
        signed=False,
        signed_zero=False,
    ),
)

PACKED_FORMATS = tuple(
    format_spec for format_spec in BINARY_FORMATS if format_spec.storage_bits < 8
)


def finite_value(format_spec, raw):
    """Decode a finite code directly from sign/exponent/mantissa fields."""

    negative = format_spec.signed and bool(raw & format_spec.sign_mask)
    magnitude = raw & (format_spec.sign_mask - 1) if format_spec.signed else raw
    exponent_mask = (1 << format_spec.exponent_bits) - 1
    mantissa_mask = (1 << format_spec.mantissa_bits) - 1
    exponent = (magnitude >> format_spec.mantissa_bits) & exponent_mask
    mantissa = magnitude & mantissa_mask
    if exponent == 0:
        value = math.ldexp(
            mantissa / (1 << format_spec.mantissa_bits),
            1 - format_spec.exponent_bias,
        )
    else:
        value = math.ldexp(
            1.0 + mantissa / (1 << format_spec.mantissa_bits),
            exponent - format_spec.exponent_bias,
        )
    return -value if negative else value


def unpack_little_endian(storage, count, bits_per_value):
    """Unpack the byte stream specified by the little-endian bit convention."""

    values = []
    for index in range(count):
        bit_offset = index * bits_per_value
        value = 0
        for bit in range(bits_per_value):
            absolute_bit = bit_offset + bit
            value |= ((storage[absolute_bit // 8] >> (absolute_bit % 8)) & 1) << bit
        values.append(value)
    return values


def pack_little_endian(codes, bits_per_value):
    storage = bytearray((len(codes) * bits_per_value + 7) // 8)
    for index, code in enumerate(codes):
        bit_offset = index * bits_per_value
        for bit in range(bits_per_value):
            if (code >> bit) & 1:
                absolute_bit = bit_offset + bit
                storage[absolute_bit // 8] |= 1 << (absolute_bit % 8)
    return bytes(storage)


def encode_codes(format_spec, values):
    tensor = hv.from_numpy(
        np.asarray(values, dtype=np.float32), format_spec.scalar_type
    )
    storage = tensor.storage
    if format_spec.storage_bits == 16:
        return [int(value) for value in np.frombuffer(storage, dtype="<u2")]
    if format_spec.storage_bits == 8:
        return list(storage)
    return unpack_little_endian(storage, len(values), format_spec.storage_bits)


def float32_from_bits(raw):
    return np.asarray([raw], dtype=np.uint32).view(np.float32)[0]


class ScalarEncodingOracleTests(unittest.TestCase):
    def test_zero_subnormal_and_normal_boundary_encodings(self):
        for format_spec in BINARY_FORMATS:
            maximum_subnormal_raw = (1 << format_spec.mantissa_bits) - 1
            minimum_normal_raw = 1 << format_spec.mantissa_bits
            values = [
                np.float32(0.0),
                np.float32(-0.0),
                np.float32(finite_value(format_spec, 1)),
                np.float32(finite_value(format_spec, maximum_subnormal_raw)),
                np.float32(finite_value(format_spec, minimum_normal_raw)),
            ]
            expected = [
                0,
                format_spec.sign_mask if format_spec.signed_zero else 0,
                1,
                maximum_subnormal_raw,
                minimum_normal_raw,
            ]
            with self.subTest(format=format_spec.name):
                self.assertEqual(encode_codes(format_spec, values), expected)

        # E8M0 has neither zero nor subnormals: zero clamps to its minimum
        # exponent code, which also represents 2**-127.
        minimum = np.float32(math.ldexp(1.0, -127))
        next_value = np.float32(math.ldexp(1.0, -126))
        tensor = hv.from_numpy(
            np.asarray([0.0, -0.0, minimum, next_value], dtype=np.float32),
            hv.ScalarType.E8M0,
        )
        self.assertEqual(tensor.storage, bytes([0x00, 0x00, 0x00, 0x01]))

    def test_adjacent_midpoints_use_round_to_nearest_even(self):
        for format_spec in BINARY_FORMATS:
            maximum_subnormal_raw = (1 << format_spec.mantissa_bits) - 1
            minimum_normal_raw = 1 << format_spec.mantissa_bits
            unity_raw = format_spec.exponent_bias << format_spec.mantissa_bits
            adjacent_pairs = (
                (maximum_subnormal_raw, minimum_normal_raw),
                (unity_raw, unity_raw + 1),
                (unity_raw + 1, unity_raw + 2),
            )
            for lower_raw, upper_raw in adjacent_pairs:
                lower = np.float32(finite_value(format_spec, lower_raw))
                upper = np.float32(finite_value(format_spec, upper_raw))
                midpoint = np.float32((float(lower) + float(upper)) / 2.0)
                below = np.nextafter(midpoint, np.float32(-np.inf))
                above = np.nextafter(midpoint, np.float32(np.inf))
                tie_raw = lower_raw if lower_raw % 2 == 0 else upper_raw
                with self.subTest(
                    format=format_spec.name,
                    lower_raw=hex(lower_raw),
                    upper_raw=hex(upper_raw),
                ):
                    self.assertGreater(below, lower)
                    self.assertLess(above, upper)
                    self.assertEqual(
                        encode_codes(format_spec, [below, midpoint, above]),
                        [lower_raw, tie_raw, upper_raw],
                    )

        # E8M0 applies the same even-code tie rule between adjacent powers of
        # two. These pairs exercise both lower-even and lower-odd outcomes.
        values = []
        expected = []
        for lower_raw in (126, 127):
            lower = np.float32(math.ldexp(1.0, lower_raw - 127))
            upper = np.float32(math.ldexp(1.0, lower_raw + 1 - 127))
            midpoint = np.float32((float(lower) + float(upper)) / 2.0)
            values.extend(
                [
                    np.nextafter(midpoint, np.float32(-np.inf)),
                    midpoint,
                    np.nextafter(midpoint, np.float32(np.inf)),
                ]
            )
            expected.extend(
                [
                    lower_raw,
                    lower_raw if lower_raw % 2 == 0 else lower_raw + 1,
                    lower_raw + 1,
                ]
            )
        encoded = hv.from_numpy(
            np.asarray(values, dtype=np.float32), hv.ScalarType.E8M0
        )
        self.assertEqual(encoded.storage, bytes(expected))

    def test_finite_overflow_infinity_and_nan_policies(self):
        positive_nan = float32_from_bits(0x7FC00000)
        negative_nan = float32_from_bits(0xFFC00000)
        maximum_float32 = np.finfo(np.float32).max

        for format_spec in BINARY_FORMATS:
            maximum = np.float32(
                finite_value(format_spec, format_spec.maximum_finite_raw)
            )
            values = [maximum, maximum_float32, np.float32(np.inf), positive_nan]
            expected = [
                format_spec.maximum_finite_raw,
                format_spec.finite_overflow_raw,
                (
                    format_spec.infinity_raw
                    if format_spec.infinity_raw is not None
                    else format_spec.maximum_finite_raw
                ),
                (
                    format_spec.nan_raw
                    if format_spec.nan_raw is not None
                    else format_spec.maximum_finite_raw
                ),
            ]

            if format_spec.signed:
                values.extend(
                    [-maximum, -maximum_float32, np.float32(-np.inf), negative_nan]
                )
                expected.extend(
                    [
                        format_spec.sign_mask | format_spec.maximum_finite_raw,
                        format_spec.sign_mask | format_spec.finite_overflow_raw,
                        format_spec.sign_mask
                        | (
                            format_spec.infinity_raw
                            if format_spec.infinity_raw is not None
                            else format_spec.maximum_finite_raw
                        ),
                        format_spec.sign_mask
                        | (
                            format_spec.nan_raw
                            if format_spec.nan_raw is not None
                            else format_spec.maximum_finite_raw
                        ),
                    ]
                )
            else:
                # Unsigned scale NaNs are canonical regardless of input sign.
                values.append(negative_nan)
                expected.append(format_spec.nan_raw)

            with self.subTest(format=format_spec.name):
                self.assertEqual(encode_codes(format_spec, values), expected)
                if not format_spec.signed:
                    for invalid in (-1.0, -np.inf):
                        with self.subTest(format=format_spec.name, invalid=invalid):
                            with self.assertRaises(ValueError):
                                encode_codes(format_spec, [invalid])

        e8m0 = hv.from_numpy(
            np.asarray(
                [
                    math.ldexp(1.0, 127),
                    maximum_float32,
                    np.inf,
                    positive_nan,
                    negative_nan,
                ],
                dtype=np.float32,
            ),
            hv.ScalarType.E8M0,
        )
        self.assertEqual(e8m0.storage, bytes([0xFE, 0xFE, 0xFE, 0xFF, 0xFF]))
        for invalid in (-1.0, -np.inf):
            with self.subTest(format="e8m0", invalid=invalid):
                with self.assertRaises(ValueError):
                    hv.from_numpy(
                        np.asarray([invalid], dtype=np.float32),
                        hv.ScalarType.E8M0,
                    )

    def test_subbyte_codes_pack_across_byte_boundaries(self):
        for format_spec in PACKED_FORMATS:
            maximum_subnormal_raw = (1 << format_spec.mantissa_bits) - 1
            minimum_normal_raw = 1 << format_spec.mantissa_bits
            unity_raw = format_spec.exponent_bias << format_spec.mantissa_bits
            codes = [
                0,
                1,
                maximum_subnormal_raw,
                minimum_normal_raw,
                unity_raw,
                unity_raw + 1,
                format_spec.maximum_finite_raw,
                format_spec.sign_mask | 1,
                format_spec.sign_mask | format_spec.maximum_finite_raw,
                format_spec.sign_mask,
            ]
            values = np.asarray(
                [finite_value(format_spec, raw) for raw in codes], dtype=np.float32
            )
            observed = hv.from_numpy(values, format_spec.scalar_type)
            with self.subTest(format=format_spec.name):
                self.assertEqual(
                    observed.storage,
                    pack_little_endian(codes, format_spec.storage_bits),
                )
                self.assertEqual(
                    unpack_little_endian(
                        observed.storage, len(codes), format_spec.storage_bits
                    ),
                    codes,
                )


if __name__ == "__main__":
    unittest.main()
