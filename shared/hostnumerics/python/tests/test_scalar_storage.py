# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks scalar encodings against an oracle built from published format facts."""

from __future__ import annotations

import bisect
import functools
import math
import random
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

import hostnumerics as hn


@dataclass(frozen=True)
class BinaryFormat:
    """A binary floating format, described without consulting HostNumerics."""

    scalar_type: object
    storage_bits: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    maximum_finite_raw: int
    # Code that encodes infinity, or None.
    infinity_raw: Optional[int]
    # Code that encoding produces for NaN, or None if the format has no NaN.
    nan_raw: Optional[int]
    signed: bool = True
    # FNUZ formats use the negative-zero code for NaN.
    negative_zero_is_nan: bool = False
    # IEEE 16-bit formats round finite overflow to infinity; narrower formats saturate.
    finite_overflow_to_infinity: bool = False

    @property
    def sign_mask(self):
        return 1 << (self.exponent_bits + self.mantissa_bits) if self.signed else 0

    @property
    def magnitude_mask(self):
        return (1 << (self.exponent_bits + self.mantissa_bits)) - 1

    def is_nan(self, raw):
        magnitude = raw & self.magnitude_mask
        if self.negative_zero_is_nan:
            return raw == self.sign_mask
        if self.infinity_raw is not None:
            return magnitude > self.infinity_raw
        return self.nan_raw is not None and magnitude == self.nan_raw


# Fields: type, storage bits, exponent bits, fraction bits, bias, largest finite
# code, infinity code, produced NaN code.
# fmt: off
FORMATS = (
    BinaryFormat(hn.ScalarType.Float16,    16, 5, 10,  15, 0x7BFF, 0x7C00, 0x7E00,
                 finite_overflow_to_infinity=True),
    BinaryFormat(hn.ScalarType.BFloat16,   16, 8,  7, 127, 0x7F7F, 0x7F80, 0x7FC0,
                 finite_overflow_to_infinity=True),
    BinaryFormat(hn.ScalarType.Float4E2M1,  4, 2,  1,   1,    0x7,   None,   None),
    BinaryFormat(hn.ScalarType.Float6E2M3,  6, 2,  3,   1,   0x1F,   None,   None),
    BinaryFormat(hn.ScalarType.Float6E3M2,  6, 3,  2,   3,   0x1F,   None,   None),
    BinaryFormat(hn.ScalarType.Float8E4M3,  8, 4,  3,   7,   0x7E,   None,   0x7F),
    BinaryFormat(hn.ScalarType.Float8E5M2,  8, 5,  2,  15,   0x7B,   0x7C,   0x7F),
    BinaryFormat(hn.ScalarType.Float8E4M3Fnuz, 8, 4, 3, 8,   0x7F,   None,   0x80,
                 negative_zero_is_nan=True),
    BinaryFormat(hn.ScalarType.Float8E5M2Fnuz, 8, 5, 2, 16,  0x7F,   None,   0x80,
                 negative_zero_is_nan=True),
    # The E4M3 scale stores seven value bits in a byte and ignores the top bit.
    BinaryFormat(hn.ScalarType.E4M3,        8, 4,  3,   7,   0x7E,   None,   0x7F,
                 signed=False),
    BinaryFormat(hn.ScalarType.E5M3,        8, 5,  3,  15,   0xFE,   None,   0xFF,
                 signed=False),
)
# fmt: on

INTEGER_OPTIONS = (hn.IntegerRounding.NearestEven, hn.IntegerOverflow.Saturate)
DEFAULTS = hn.ScalarConversionOptions(*INTEGER_OPTIONS)
DIRECT = hn.ScalarConversionOptions(
    *INTEGER_OPTIONS, floating_source_precision=hn.FloatingSourcePrecision.Direct
)
SATURATE = hn.ScalarConversionOptions(
    *INTEGER_OPTIONS, floating_overflow=hn.FloatingOverflow.SaturateFinite
)


def finite_value(fmt, raw):
    """Decode a finite code from its sign, exponent and fraction fields."""

    negative = bool(raw & fmt.sign_mask)
    magnitude = raw & fmt.magnitude_mask
    exponent = magnitude >> fmt.mantissa_bits
    fraction = (magnitude & ((1 << fmt.mantissa_bits) - 1)) / (1 << fmt.mantissa_bits)
    if exponent == 0:
        value = math.ldexp(fraction, 1 - fmt.exponent_bias)
    else:
        value = math.ldexp(1.0 + fraction, exponent - fmt.exponent_bias)
    return -value if negative else value


def expected_decode(fmt, raw):
    if fmt.is_nan(raw):
        return math.nan
    if fmt.infinity_raw is not None and raw & fmt.magnitude_mask == fmt.infinity_raw:
        return -math.inf if raw & fmt.sign_mask else math.inf
    return finite_value(fmt, raw)


def pack_little_endian(codes, bits):
    """Pack codes least significant bit first, as the storage layout specifies."""

    code_bits = (np.asarray(codes, dtype=np.int64)[:, None] >> np.arange(bits)) & 1
    return np.packbits(code_bits.astype(np.uint8).ravel(), bitorder="little").tobytes()


def unpack_little_endian(storage, count, bits):
    code_bits = np.unpackbits(np.frombuffer(storage, dtype=np.uint8), bitorder="little")
    code_bits = code_bits[: count * bits].reshape(count, bits).astype(np.int64)
    return [int(code) for code in (code_bits << np.arange(bits)).sum(axis=1)]


def encode_codes(scalar_type, bits, values, options=DEFAULTS):
    storage = hn.encode_scalars(
        scalar_type, [float(value) for value in values], options
    )
    return unpack_little_endian(storage, len(values), bits)


@functools.cache
def rounding_grid(fmt):
    """Nonnegative finite values in code order, then one step above the largest."""

    positive = [finite_value(fmt, raw) for raw in range(fmt.maximum_finite_raw + 1)]
    return positive + [2 * positive[-1] - positive[-2]]


def nearest_code(fmt, value):
    """Round value to the nearest finite code with ties to the even code."""

    grid = rounding_grid(fmt)
    magnitude = abs(value)
    sign = fmt.sign_mask if math.copysign(1.0, value) < 0 else 0
    upper = bisect.bisect_left(grid, magnitude)
    if upper == len(grid):
        raw = upper
    elif upper == 0:
        raw = 0
    else:
        lower = upper - 1
        if grid[upper] == magnitude:
            raw = upper
        else:
            midpoint = (grid[lower] + grid[upper]) / 2
            if magnitude < midpoint:
                raw = lower
            elif magnitude > midpoint:
                raw = upper
            else:
                raw = lower if lower % 2 == 0 else upper
    if raw > fmt.maximum_finite_raw:
        raw = (
            fmt.infinity_raw
            if fmt.finite_overflow_to_infinity
            else fmt.maximum_finite_raw
        )
    if raw == 0 and (fmt.negative_zero_is_nan or not fmt.signed):
        return 0
    return sign | raw


class CodeTestCase(unittest.TestCase):
    def assert_codes(self, observed, expected, values):
        """Compare code lists, reporting the first mismatches instead of a full diff."""

        self.assertEqual(len(observed), len(expected))
        mismatches = [
            f"{float(values[index])!r}: {observed[index]:#x} != {int(expected[index]):#x}"
            for index in range(len(expected))
            if observed[index] != expected[index]
        ]
        if mismatches:
            self.fail(
                f"{len(mismatches)} mismatches, including " + "; ".join(mismatches[:5])
            )


class BinaryFloatTests(CodeTestCase):
    def test_every_code_decodes(self):
        for fmt in FORMATS:
            raw = list(range(1 << fmt.storage_bits))
            storage = pack_little_endian(raw, fmt.storage_bits)
            observed = hn.decode_scalars(fmt.scalar_type, storage, len(raw))
            with self.subTest(format=fmt.scalar_type):
                for code, value in zip(raw, observed):
                    expected = expected_decode(fmt, code)
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(value), hex(code))
                    else:
                        self.assertEqual(value, expected, hex(code))
                        self.assertEqual(
                            math.copysign(1.0, value), math.copysign(1.0, expected)
                        )

    def test_ieee_16_bit_decoding_matches_numpy(self):
        raw = np.arange(1 << 16, dtype="<u2")
        # Widening signaling NaNs warns; only their NaN-ness is compared.
        with np.errstate(invalid="ignore"):
            cases = (
                (hn.ScalarType.Float16, raw.view("<f2").astype(np.float64)),
                (
                    hn.ScalarType.BFloat16,
                    (raw.astype("<u4") << 16).view("<f4").astype(np.float64),
                ),
            )
        for scalar_type, expected in cases:
            with self.subTest(scalar_type=scalar_type):
                observed = np.asarray(
                    hn.decode_scalars(scalar_type, raw.tobytes(), raw.size)
                )
                np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
                finite = ~np.isnan(expected)
                np.testing.assert_array_equal(observed[finite], expected[finite])
                np.testing.assert_array_equal(
                    np.signbit(observed[finite]), np.signbit(expected[finite])
                )

    def test_every_finite_code_round_trips(self):
        for fmt in FORMATS:
            codes = [
                raw
                for raw in range(1 << fmt.storage_bits)
                if not fmt.is_nan(raw)
                and raw & ~(fmt.sign_mask | fmt.magnitude_mask) == 0
            ]
            values = [expected_decode(fmt, raw) for raw in codes]
            for options in (DEFAULTS, DIRECT):
                with self.subTest(
                    format=fmt.scalar_type, options=options.floating_source_precision
                ):
                    self.assert_codes(
                        encode_codes(
                            fmt.scalar_type, fmt.storage_bits, values, options
                        ),
                        codes,
                        values,
                    )

    def test_midpoints_round_to_nearest_even(self):
        for fmt in FORMATS:
            grid = np.asarray(rounding_grid(fmt)[:-1], dtype=np.float32)
            lower, upper = grid[:-1], grid[1:]
            midpoint = ((lower.astype(np.float64) + upper) / 2).astype(np.float32)
            lower_raw = np.arange(lower.size)
            tie_raw = lower_raw + lower_raw % 2
            # Just below, at and just above each midpoint.
            values = np.stack(
                [
                    np.nextafter(midpoint, lower),
                    midpoint,
                    np.nextafter(midpoint, upper),
                ],
                axis=1,
            ).ravel()
            codes = np.stack([lower_raw, tie_raw, lower_raw + 1], axis=1).ravel()
            expected = codes
            if fmt.signed:
                values = np.concatenate([values, -values])
                # FNUZ has no negative zero.
                negative = np.where(
                    (codes == 0) & fmt.negative_zero_is_nan, 0, codes | fmt.sign_mask
                )
                expected = np.concatenate([codes, negative])
            with self.subTest(format=fmt.scalar_type):
                self.assert_codes(
                    encode_codes(fmt.scalar_type, fmt.storage_bits, values),
                    expected,
                    values,
                )

    def test_random_values_match_nearest_code(self):
        generator = random.Random(20261105)
        for fmt in FORMATS:
            largest = finite_value(fmt, fmt.maximum_finite_raw)
            smallest = finite_value(fmt, 1)
            values = []
            for _ in range(3000):
                # Stay below 2^128, which overflows float.
                exponent = generator.uniform(
                    math.log2(smallest) - 2, min(math.log2(largest) + 1, 127.5)
                )
                value = 2.0**exponent
                values.append(
                    -value if fmt.signed and generator.random() < 0.5 else value
                )
            for options, round_to_float in ((DEFAULTS, True), (DIRECT, False)):
                sources = [
                    float(np.float32(value)) if round_to_float else value
                    for value in values
                ]
                expected = [nearest_code(fmt, value) for value in sources]
                with self.subTest(
                    format=fmt.scalar_type, options=options.floating_source_precision
                ):
                    self.assert_codes(
                        encode_codes(
                            fmt.scalar_type, fmt.storage_bits, values, options
                        ),
                        expected,
                        values,
                    )

    def test_source_precision_controls_double_rounding(self):
        # 1 + 2^-8 + 2^-30 rounds to the tie 1 + 2^-8 as a float, which rounds to 1.0 in BF16.
        value = 1.0 + 2.0**-8 + 2.0**-30
        self.assertEqual(encode_codes(hn.ScalarType.BFloat16, 16, [value]), [0x3F80])
        self.assertEqual(
            encode_codes(hn.ScalarType.BFloat16, 16, [value], DIRECT), [0x3F81]
        )

    def test_overflow_infinity_and_nan(self):
        maximum_float32 = float(np.finfo(np.float32).max)
        for fmt in FORMATS:
            maximum = finite_value(fmt, fmt.maximum_finite_raw)
            nan_raw = fmt.nan_raw if fmt.nan_raw is not None else fmt.maximum_finite_raw
            infinity_raw = (
                fmt.infinity_raw
                if fmt.infinity_raw is not None
                else fmt.maximum_finite_raw
            )
            overflow_raw = (
                fmt.infinity_raw
                if fmt.finite_overflow_to_infinity
                else fmt.maximum_finite_raw
            )
            values = [maximum, maximum_float32, math.inf, math.nan]
            default_codes = [
                fmt.maximum_finite_raw,
                overflow_raw,
                infinity_raw,
                nan_raw,
            ]
            saturated_codes = [fmt.maximum_finite_raw] * 3 + [nan_raw]
            if fmt.signed:
                values += [-maximum, -maximum_float32, -math.inf, -math.nan]
                # FNUZ has a single unsigned NaN; other signed formats keep the sign.
                negative_nan = (
                    nan_raw if fmt.negative_zero_is_nan else fmt.sign_mask | nan_raw
                )
                default_codes += [fmt.sign_mask | raw for raw in default_codes[:3]] + [
                    negative_nan
                ]
                saturated_codes += [fmt.sign_mask | fmt.maximum_finite_raw] * 3 + [
                    negative_nan
                ]
            with self.subTest(format=fmt.scalar_type):
                self.assertEqual(
                    encode_codes(fmt.scalar_type, fmt.storage_bits, values),
                    default_codes,
                )
                self.assertEqual(
                    encode_codes(fmt.scalar_type, fmt.storage_bits, values, SATURATE),
                    saturated_codes,
                )

    def test_bfloat16_truncation(self):
        truncate = hn.ScalarConversionOptions(
            *INTEGER_OPTIONS, bfloat16_rounding=hn.BFloat16Rounding.Truncate
        )
        below_next = 1.0 + 2.0**-7 - 2.0**-20
        self.assertEqual(
            encode_codes(
                hn.ScalarType.BFloat16,
                16,
                [below_next, -below_next, math.nan],
                truncate,
            ),
            [0x3F80, 0xBF80, 0xFFC1],
        )

    def test_unsigned_formats_reject_negative_values(self):
        for scalar_type in (hn.ScalarType.E4M3, hn.ScalarType.E5M3, hn.ScalarType.E8M0):
            for value in (-1.0, -math.inf, -(2.0**-140)):
                with self.subTest(scalar_type=scalar_type, value=value):
                    with self.assertRaises(ValueError):
                        hn.encode_scalars(scalar_type, [value], DEFAULTS)
            self.assertEqual(hn.encode_scalars(scalar_type, [-0.0], DEFAULTS), b"\x00")

    def test_packed_codes_cross_byte_boundaries(self):
        for fmt in FORMATS:
            if fmt.storage_bits >= 8:
                continue
            unity = fmt.exponent_bias << fmt.mantissa_bits
            codes = [
                0,
                1,
                unity,
                unity + 1,
                fmt.maximum_finite_raw,
                fmt.sign_mask | 1,
                fmt.sign_mask | fmt.maximum_finite_raw,
            ]
            values = [finite_value(fmt, raw) for raw in codes]
            with self.subTest(format=fmt.scalar_type):
                storage = hn.encode_scalars(fmt.scalar_type, values, DEFAULTS)
                self.assertEqual(storage, pack_little_endian(codes, fmt.storage_bits))
                self.assertEqual(
                    hn.decode_scalars(fmt.scalar_type, storage, len(codes)), values
                )


class ScaleTests(CodeTestCase):
    def test_e8m0_codes_are_powers_of_two_without_zero(self):
        observed = hn.decode_scalars(hn.ScalarType.E8M0, bytes(range(256)), 256)
        for code in range(255):
            self.assertEqual(observed[code], math.ldexp(1.0, code - 127))
        self.assertTrue(math.isnan(observed[255]))

        # Zero and values below the smallest code clamp to it; infinity clamps to 2^127.
        values = [
            0.0,
            2.0**-140,
            2.0**127,
            float(np.finfo(np.float32).max),
            math.inf,
            math.nan,
        ]
        self.assertEqual(
            encode_codes(hn.ScalarType.E8M0, 8, values), [0, 0, 254, 254, 254, 255]
        )

    def test_e8m0_rounds_to_even_codes(self):
        values, expected = [], []
        for lower_raw in range(254):
            lower = math.ldexp(1.0, lower_raw - 127)
            midpoint = lower * 1.5
            values += [
                np.nextafter(np.float32(midpoint), np.float32(0)),
                midpoint,
                np.nextafter(np.float32(midpoint), np.float32(math.inf)),
            ]
            expected += [lower_raw, lower_raw + lower_raw % 2, lower_raw + 1]
        self.assert_codes(encode_codes(hn.ScalarType.E8M0, 8, values), expected, values)

    def test_e8m0_flush_below_float32_normal(self):
        flush = hn.ScalarConversionOptions(
            *INTEGER_OPTIONS, e8m0_underflow=hn.E8M0Underflow.FlushBelowFloat32Normal
        )
        values = [1.75 * 2.0**-127, 2.0**-126, 1.75 * 2.0**-126]
        self.assertEqual(encode_codes(hn.ScalarType.E8M0, 8, values), [1, 1, 2])
        self.assertEqual(encode_codes(hn.ScalarType.E8M0, 8, values, flush), [0, 1, 2])

    def test_e8m0_zero_keeps_the_float32_exponent_field(self):
        observed = hn.decode_scalars(
            hn.ScalarType.E8M0Zero, bytes([0, 1, 127, 254, 255]), 5
        )
        self.assertEqual(observed[:4], [0.0, 2.0**-126, 1.0, 2.0**127])
        self.assertTrue(math.isnan(observed[4]))

        values = [0.0, 2.0**-127, 1.75, -2.0, 2.0**127 * 1.5, math.inf, math.nan]
        expected = [0, 0, 127, 128, 254, 255, 255]
        for options in (DEFAULTS, DIRECT):
            self.assertEqual(
                encode_codes(hn.ScalarType.E8M0Zero, 8, values, options), expected
            )


class NonFloatingTests(unittest.TestCase):
    def test_integer_policies(self):
        saturate = DEFAULTS
        wrap = hn.ScalarConversionOptions(
            hn.IntegerRounding.TowardZero, hn.IntegerOverflow.ModuloWrap
        )
        cases = (
            (hn.ScalarType.Int4, [9, -9, 7, -8], saturate, [7, -8, 7, -8]),
            (hn.ScalarType.Int4, [9, -9, 7, -8], wrap, [-7, 7, 7, -8]),
            (hn.ScalarType.Int8, [300, -300], saturate, [127, -128]),
            (hn.ScalarType.UInt8, [300, -1], wrap, [44, 255]),
            (
                hn.ScalarType.Int64,
                [-(2**63), 2**63 - 1],
                saturate,
                [-(2**63), 2**63 - 1],
            ),
            (hn.ScalarType.UInt64, [2**64 - 1], saturate, [2**64 - 1]),
        )
        for scalar_type, values, options, expected in cases:
            with self.subTest(scalar_type=scalar_type, values=values):
                storage = hn.encode_scalars(scalar_type, values, options)
                self.assertEqual(
                    hn.decode_scalars(scalar_type, storage, len(values)), expected
                )
        # Int4 packs two's-complement nibbles, low nibble first.
        self.assertEqual(
            hn.encode_scalars(hn.ScalarType.Int4, [-1, 2, 3], saturate), b"\x2f\x03"
        )

    def test_boolean_and_complex_values(self):
        self.assertEqual(
            hn.encode_scalars(hn.ScalarType.Boolean, [True, False], DEFAULTS),
            b"\x01\x00",
        )
        self.assertEqual(
            hn.decode_scalars(hn.ScalarType.Boolean, b"\x00\x02", 2), [False, True]
        )
        values = [1.5 - 2j, 3.0]
        storage = hn.encode_scalars(hn.ScalarType.ComplexFloat32, values, DEFAULTS)
        self.assertEqual(storage, np.asarray(values, dtype="<c8").tobytes())
        self.assertEqual(
            hn.decode_scalars(hn.ScalarType.ComplexFloat32, storage, 2),
            [1.5 - 2j, 3 + 0j],
        )

    def test_rejects_wrong_value_types_and_short_storage(self):
        with self.assertRaises(TypeError):
            hn.encode_scalars(hn.ScalarType.Int8, [1.5], DEFAULTS)
        with self.assertRaises(TypeError):
            hn.encode_scalars(hn.ScalarType.Float16, [1j], DEFAULTS)
        with self.assertRaises(RuntimeError):
            hn.decode_scalars(hn.ScalarType.Float16, b"\x00\x3c\x00", 2)
        with self.assertRaises(TypeError):
            hn.ScalarConversionOptions(hn.IntegerRounding.NearestEven)
        identity = hn.ScalarConversionOptions(
            hn.IntegerRounding.TowardZero, hn.IntegerOverflow.Identity
        )
        self.assertEqual(
            hn.encode_scalars(hn.ScalarType.Int4, [7, -8], identity), b"\x87"
        )
        for scalar_type, value in ((hn.ScalarType.Int4, 8), (hn.ScalarType.UInt8, -1)):
            with self.assertRaises(ValueError):
                hn.encode_scalars(scalar_type, [value], identity)


if __name__ == "__main__":
    unittest.main()
