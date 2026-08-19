# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import unittest
from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

import roc_host_validation as hv


MASK64 = (1 << 64) - 1
MX_DATA_RANDOM_DOMAIN = 0x3F84D5B5B5470917
MX_NORMAL_RANDOM_DOMAIN = 0x9216D5D98979FB1B
MX_BOUNDED_SCALE_RANDOM_DOMAIN = 0xA24BAED4963EE407
MX_UNBOUNDED_DATA_RANDOM_DOMAIN = 0xD1B54A32D192ED03
MX_UNBOUNDED_SCALE_RANDOM_DOMAIN = 0x94D049BB133111EB
TWO_PI = 6.28318530717958647692528676655900576


@dataclass(frozen=True)
class BinaryFormat:
    name: str
    scalar_type: object
    storage_bits: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    maximum_finite_raw: int
    infinity_raw: Optional[int] = None
    nan_raw: Optional[int] = None
    signed: bool = True

    @property
    def sign_mask(self):
        return 1 << (self.storage_bits - 1) if self.signed else 0


FP4_E2M1 = BinaryFormat("fp4_e2m1", hv.ScalarType.Float4E2M1, 4, 2, 1, 1, 0x07)
FP6_E2M3 = BinaryFormat("fp6_e2m3", hv.ScalarType.Float6E2M3, 6, 2, 3, 1, 0x1F)
FP6_E3M2 = BinaryFormat("fp6_e3m2", hv.ScalarType.Float6E3M2, 6, 3, 2, 3, 0x1F)
FP8_E4M3 = BinaryFormat(
    "fp8_e4m3", hv.ScalarType.Float8E4M3, 8, 4, 3, 7, 0x7E, nan_raw=0x7F
)
FP8_E5M2 = BinaryFormat(
    "fp8_e5m2",
    hv.ScalarType.Float8E5M2,
    8,
    5,
    2,
    15,
    0x7B,
    infinity_raw=0x7C,
    nan_raw=0x7F,
)
E4M3_SCALE = BinaryFormat(
    "e4m3_scale",
    hv.ScalarType.E4M3,
    8,
    4,
    3,
    7,
    0x7E,
    nan_raw=0x7F,
    signed=False,
)
E5M3_SCALE = BinaryFormat(
    "e5m3_scale",
    hv.ScalarType.E5M3,
    8,
    5,
    3,
    15,
    0xFE,
    nan_raw=0xFF,
    signed=False,
)

DATA_FORMATS = {
    format_spec.scalar_type: format_spec
    for format_spec in (FP4_E2M1, FP6_E2M3, FP6_E3M2, FP8_E4M3, FP8_E5M2)
}
SCALE_FORMATS = {
    E4M3_SCALE.scalar_type: E4M3_SCALE,
    E5M3_SCALE.scalar_type: E5M3_SCALE,
}


class RecipeKind(Enum):
    BOUNDED = auto()
    BOUNDED_ALTERNATING_SIGN = auto()
    UNBOUNDED = auto()
    IDENTITY = auto()
    CONSTANT = auto()
    SEQUENTIAL = auto()
    ROW_INDEX = auto()
    COLUMN_INDEX = auto()
    CHECKERBOARD = auto()
    SCALED_DIAGONAL = auto()
    TYPE_MAXIMUM = auto()
    TYPE_DENORMAL_MINIMUM = auto()
    TYPE_DENORMAL_MAXIMUM = auto()
    TYPE_NAN = auto()
    TYPE_INFINITY = auto()
    TRIGONOMETRIC = auto()
    NORMAL = auto()
    UNIFORM_INTEGER = auto()


@dataclass(frozen=True)
class Recipe:
    kind: RecipeKind
    lower: float = -1.0
    upper: float = 1.0
    maximum_magnitude: float = 1.0
    value: float = 0.0
    mean: float = 0.0
    standard_deviation: float = 1.0
    integer_lower: int = 0
    integer_upper: int = 1

    def native(self):
        factories = {
            RecipeKind.UNBOUNDED: hv.MxDataRecipe.unbounded,
            RecipeKind.IDENTITY: hv.MxDataRecipe.identity,
            RecipeKind.SEQUENTIAL: hv.MxDataRecipe.sequential,
            RecipeKind.ROW_INDEX: hv.MxDataRecipe.row_index,
            RecipeKind.COLUMN_INDEX: hv.MxDataRecipe.column_index,
            RecipeKind.CHECKERBOARD: hv.MxDataRecipe.checkerboard,
            RecipeKind.SCALED_DIAGONAL: hv.MxDataRecipe.scaled_diagonal,
            RecipeKind.TYPE_MAXIMUM: hv.MxDataRecipe.type_maximum,
            RecipeKind.TYPE_DENORMAL_MINIMUM: hv.MxDataRecipe.type_denormal_minimum,
            RecipeKind.TYPE_DENORMAL_MAXIMUM: hv.MxDataRecipe.type_denormal_maximum,
            RecipeKind.TYPE_NAN: hv.MxDataRecipe.type_nan,
            RecipeKind.TYPE_INFINITY: hv.MxDataRecipe.type_infinity,
            RecipeKind.TRIGONOMETRIC: hv.MxDataRecipe.trigonometric,
        }
        if self.kind in factories:
            return factories[self.kind]()
        if self.kind is RecipeKind.BOUNDED:
            return hv.MxDataRecipe.bounded(
                hv.MxBoundedDataParameters(self.lower, self.upper)
            )
        if self.kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
            return hv.MxDataRecipe.bounded_alternating_sign(
                hv.MxAlternatingSignDataParameters(self.maximum_magnitude)
            )
        if self.kind is RecipeKind.CONSTANT:
            return hv.MxDataRecipe.constant(self.value)
        if self.kind is RecipeKind.NORMAL:
            return hv.MxDataRecipe.normal(
                hv.MxNormalDataParameters(self.mean, self.standard_deviation)
            )
        if self.kind is RecipeKind.UNIFORM_INTEGER:
            return hv.MxDataRecipe.uniform_integer(
                hv.MxUniformIntegerDataParameters(
                    self.integer_lower, self.integer_upper
                )
            )
        raise AssertionError(f"no native MX recipe for {self.kind}")


@dataclass(frozen=True)
class MxCase:
    label: str
    data_type: object
    scale_type: object
    shape: tuple[int, ...]
    leading_dimension: int
    block_axis: int
    block_size: int
    data: Recipe
    scale: object = hv.MxScaleGenerationMode.Derived
    seed: int = 0x10203040


@dataclass(frozen=True)
class ExpectedMx:
    data_storage: bytes
    scale_storage: bytes
    scale_indices: np.ndarray
    reference: np.ndarray


def counter_random(seed, stream, index):
    """SplitMix-style counter RNG, with every operation wrapped to uint64."""

    value = (
        int(seed)
        ^ ((int(stream) + 0x9E3779B97F4A7C15) & MASK64)
        ^ ((int(index) * 0xBF58476D1CE4E5B9) & MASK64)
    ) & MASK64
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def indexed_uniform_unit(seed, stream, index):
    mantissa = counter_random(seed, stream, index) >> 11
    return (float(mantissa) + 0.5) / 9007199254740992.0


def indexed_uniform_integer(seed, stream, index, lower, upper):
    width = upper - lower + 1
    return lower + counter_random(seed, stream, index) % width


def finite_binary_value(format_spec, raw):
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


def raw_is_nan(format_spec, raw):
    if format_spec is FP8_E4M3:
        return (raw & 0x7F) == 0x7F
    if format_spec is FP8_E5M2:
        return (raw & 0x7F) > 0x7C
    return format_spec.nan_raw is not None and raw == format_spec.nan_raw


def raw_is_infinity(format_spec, raw):
    return (
        format_spec.infinity_raw is not None
        and (raw & 0x7F) == format_spec.infinity_raw
    )


def decode_binary(format_spec, raw):
    if raw_is_nan(format_spec, raw):
        return math.nan
    if raw_is_infinity(format_spec, raw):
        return -math.inf if raw & format_spec.sign_mask else math.inf
    return finite_binary_value(format_spec, raw)


def encode_binary(format_spec, value):
    """Quantize a float32 input by nearest value, breaking ties by raw parity."""

    value = np.float32(value)
    negative = bool(np.signbit(value))
    if np.isnan(value):
        if format_spec.nan_raw is None:
            return (
                format_spec.sign_mask if format_spec.signed and negative else 0
            ) | format_spec.maximum_finite_raw
        return (
            format_spec.sign_mask if format_spec.signed and negative else 0
        ) | format_spec.nan_raw

    if not format_spec.signed and value != 0.0 and negative:
        raise ValueError(f"{format_spec.name} cannot encode a negative value")

    sign = format_spec.sign_mask if format_spec.signed and negative else 0
    magnitude = float(abs(value))
    if math.isinf(magnitude):
        return sign | (
            format_spec.infinity_raw
            if format_spec.infinity_raw is not None
            else format_spec.maximum_finite_raw
        )
    if magnitude == 0.0:
        return sign if format_spec.signed else 0

    maximum = finite_binary_value(format_spec, format_spec.maximum_finite_raw)
    if magnitude >= maximum:
        return sign | format_spec.maximum_finite_raw

    positive_values = [
        finite_binary_value(format_spec, raw)
        for raw in range(format_spec.maximum_finite_raw + 1)
    ]
    upper = bisect_left(positive_values, magnitude)
    if upper == 0:
        selected = 0
    else:
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


def encode_e8m0(value):
    value = np.float32(value)
    if np.isnan(value):
        return 0xFF
    if value != 0.0 and np.signbit(value):
        raise ValueError("e8m0 cannot encode a negative value")
    if value <= np.float32(math.ldexp(1.0, -127)):
        return 0x00
    if np.isinf(value) or value >= np.float32(math.ldexp(1.0, 127)):
        return 0xFE

    exponent = math.floor(math.log2(float(value)))
    lower_raw = exponent + 127
    lower = decode_e8m0(lower_raw)
    upper = decode_e8m0(lower_raw + 1)
    lower_distance = float(value) - lower
    upper_distance = upper - float(value)
    if lower_distance < upper_distance:
        return lower_raw
    if upper_distance < lower_distance:
        return lower_raw + 1
    return lower_raw if lower_raw % 2 == 0 else lower_raw + 1


def decode_scale(scale_type, raw):
    if scale_type == hv.ScalarType.E8M0:
        return decode_e8m0(raw)
    return decode_binary(SCALE_FORMATS[scale_type], raw)


def encode_scale(scale_type, value):
    if scale_type == hv.ScalarType.E8M0:
        return encode_e8m0(value)
    return encode_binary(SCALE_FORMATS[scale_type], value)


def maximum_scale_raw(scale_type):
    if scale_type == hv.ScalarType.E8M0:
        return 0xFE
    return SCALE_FORMATS[scale_type].maximum_finite_raw


def finite_nonzero_scale_candidates(scale_type):
    candidates = []
    for raw in range(maximum_scale_raw(scale_type) + 1):
        value = decode_scale(scale_type, raw)
        if math.isfinite(value) and value > 0.0:
            candidates.append((value, raw))
    candidates.sort()
    return candidates


def finite_data_raw_candidates(format_spec):
    return [
        raw
        for raw in range(1 << format_spec.storage_bits)
        if math.isfinite(decode_binary(format_spec, raw))
    ]


def pack_little_endian(codes, bits_per_value):
    storage = bytearray((len(codes) * bits_per_value + 7) // 8)
    for index, code in enumerate(codes):
        bit_offset = index * bits_per_value
        for bit in range(bits_per_value):
            if (code >> bit) & 1:
                absolute_bit = bit_offset + bit
                storage[absolute_bit // 8] |= 1 << (absolute_bit % 8)
    return bytes(storage)


def explicit_scale_raw(scale_type, mode):
    if mode == hv.MxScaleGenerationMode.Derived:
        return None
    if mode == hv.MxScaleGenerationMode.Minimum:
        return encode_scale(scale_type, 0.0)
    if mode == hv.MxScaleGenerationMode.One:
        return encode_scale(scale_type, 1.0)
    if mode == hv.MxScaleGenerationMode.Two:
        return encode_scale(scale_type, 2.0)
    if mode == hv.MxScaleGenerationMode.Maximum:
        return maximum_scale_raw(scale_type)
    if mode == hv.MxScaleGenerationMode.NaN:
        return encode_scale(scale_type, math.nan)
    raise AssertionError(f"oracle has no MX scale formula for {mode}")


def generated_value(case, row, column, logical_index):
    recipe = case.data
    kind = recipe.kind
    if kind is RecipeKind.BOUNDED:
        unit = indexed_uniform_unit(case.seed, MX_DATA_RANDOM_DOMAIN, logical_index)
        return recipe.lower + unit * (recipe.upper - recipe.lower)
    if kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
        magnitude = recipe.maximum_magnitude * indexed_uniform_unit(
            case.seed, MX_DATA_RANDOM_DOMAIN, logical_index
        )
        return magnitude if logical_index % 2 == 0 else -magnitude
    if kind is RecipeKind.IDENTITY:
        return 1.0 if row == column else 0.0
    if kind is RecipeKind.CONSTANT:
        return recipe.value
    if kind is RecipeKind.SEQUENTIAL:
        return ((row % 256) * (case.shape[1] % 256) + column % 256) % 256
    if kind is RecipeKind.ROW_INDEX:
        return row % 256
    if kind is RecipeKind.COLUMN_INDEX:
        return column % 256
    if kind is RecipeKind.CHECKERBOARD:
        return 1.0 if (row + column) % 2 == 0 else 0.0
    if kind is RecipeKind.SCALED_DIAGONAL:
        return row + 1.0 if row == column else 0.0
    format_spec = DATA_FORMATS[case.data_type]
    if kind is RecipeKind.TYPE_MAXIMUM:
        return finite_binary_value(format_spec, format_spec.maximum_finite_raw)
    if kind is RecipeKind.TYPE_DENORMAL_MINIMUM:
        return finite_binary_value(format_spec, 1)
    if kind is RecipeKind.TYPE_DENORMAL_MAXIMUM:
        return finite_binary_value(format_spec, (1 << format_spec.mantissa_bits) - 1)
    if kind is RecipeKind.TYPE_NAN:
        return math.nan
    if kind is RecipeKind.TYPE_INFINITY:
        return math.inf
    if kind is RecipeKind.TRIGONOMETRIC:
        unit = indexed_uniform_unit(case.seed, MX_DATA_RANDOM_DOMAIN, logical_index)
        return math.cos(TWO_PI * unit)
    if kind is RecipeKind.NORMAL:
        first_index = logical_index * 2
        first = indexed_uniform_unit(case.seed, MX_NORMAL_RANDOM_DOMAIN, first_index)
        second = indexed_uniform_unit(
            case.seed, MX_NORMAL_RANDOM_DOMAIN, first_index + 1
        )
        standard_normal = math.sqrt(-2.0 * math.log(first)) * math.cos(TWO_PI * second)
        return recipe.mean + recipe.standard_deviation * standard_normal
    if kind is RecipeKind.UNIFORM_INTEGER:
        return indexed_uniform_integer(
            case.seed,
            MX_DATA_RANDOM_DOMAIN,
            logical_index,
            recipe.integer_lower,
            recipe.integer_upper,
        )
    raise AssertionError(f"oracle has no source formula for {kind}")


def scale_at_least(requested, candidates, allow_larger, seed, scale_index):
    values = [candidate[0] for candidate in candidates]
    selected = bisect_left(values, requested)
    if selected == len(candidates):
        selected -= 1
    elif (
        allow_larger
        and selected + 1 < len(candidates)
        and counter_random(seed, MX_BOUNDED_SCALE_RANDOM_DOMAIN, scale_index) & 1
    ):
        selected += 1
    return candidates[selected][1]


def constrain_raw_to_interval(
    format_spec, raw, scale, minimum, maximum, decoded_values
):
    """Clamp the initially nearest code to the same-sign codes in the interval."""

    represented = decoded_values[raw] * scale
    if minimum <= represented <= maximum:
        return raw

    sign = raw & format_spec.sign_mask
    magnitude = raw & (format_spec.sign_mask - 1)
    candidates = []
    for candidate_magnitude in range(format_spec.maximum_finite_raw + 1):
        candidate_raw = sign | candidate_magnitude
        candidate_value = decoded_values[candidate_raw] * scale
        if minimum <= candidate_value <= maximum:
            candidates.append(candidate_magnitude)
    if not candidates:
        raise ValueError("bounded interval has no representable value")
    selected_magnitude = min(
        candidates, key=lambda candidate: abs(candidate - magnitude)
    )
    return sign | selected_magnitude


def block_coordinates(case, block, free_coordinate):
    blocked_extent = case.shape[case.block_axis]
    first = block * case.block_size
    count = min(case.block_size, blocked_extent - first)
    if case.block_axis == 0:
        return [(first + offset, free_coordinate) for offset in range(count)]
    return [(free_coordinate, first + offset) for offset in range(count)]


def expected_mx(case):
    rows, columns = case.shape
    leading_dimension = case.leading_dimension or rows
    format_spec = DATA_FORMATS[case.data_type]
    blocked_extent = case.shape[case.block_axis]
    free_extent = case.shape[1 - case.block_axis]
    block_count = (blocked_extent + case.block_size - 1) // case.block_size
    scale_count = block_count * free_extent
    fixed_scale_raw = explicit_scale_raw(case.scale_type, case.scale)
    scale_candidates = (
        []
        if fixed_scale_raw is not None
        else finite_nonzero_scale_candidates(case.scale_type)
    )
    decoded_data = [
        decode_binary(format_spec, raw) for raw in range(1 << format_spec.storage_bits)
    ]
    finite_data_candidates = finite_data_raw_candidates(format_spec)
    maximum_data_value = finite_binary_value(
        format_spec, format_spec.maximum_finite_raw
    )

    physical_raw = [0] * (leading_dimension * columns)
    scale_raw = [0] * scale_count
    scale_indices = np.empty((rows, columns), dtype=np.uint32)
    reference = np.empty((rows, columns), dtype=np.float32)

    for scale_index in range(scale_count):
        if case.block_axis == 0:
            block = scale_index % block_count
            free_coordinate = scale_index // block_count
        else:
            block = scale_index // free_extent
            free_coordinate = scale_index % free_extent
        coordinates = block_coordinates(case, block, free_coordinate)

        if case.data.kind is RecipeKind.UNBOUNDED:
            if fixed_scale_raw is None:
                candidate_index = counter_random(
                    case.seed, MX_UNBOUNDED_SCALE_RANDOM_DOMAIN, scale_index
                ) % len(scale_candidates)
                selected_scale_raw = scale_candidates[candidate_index][1]
            else:
                selected_scale_raw = fixed_scale_raw
            scale_raw[scale_index] = selected_scale_raw
            scale_value = decode_scale(case.scale_type, selected_scale_raw)

            for row, column in coordinates:
                logical_index = row + column * rows
                physical_index = row + column * leading_dimension
                candidate_index = counter_random(
                    case.seed, MX_UNBOUNDED_DATA_RANDOM_DOMAIN, logical_index
                ) % len(finite_data_candidates)
                data_raw = finite_data_candidates[candidate_index]
                physical_raw[physical_index] = data_raw
                scale_indices[row, column] = scale_index
                with np.errstate(over="ignore", invalid="ignore"):
                    reference[row, column] = np.float32(
                        decoded_data[data_raw] * scale_value
                    )
            continue

        block_values = [
            generated_value(case, row, column, row + column * rows)
            for row, column in coordinates
        ]
        if fixed_scale_raw is not None:
            selected_scale_raw = fixed_scale_raw
        elif any(math.isnan(value) for value in block_values):
            selected_scale_raw = encode_scale(case.scale_type, math.nan)
        else:
            maximum_magnitude = max(
                (abs(value) for value in block_values if math.isfinite(value)),
                default=0.0,
            )
            if maximum_magnitude == 0.0:
                selected_scale_raw = encode_scale(case.scale_type, 1.0)
            else:
                selected_scale_raw = scale_at_least(
                    maximum_magnitude / maximum_data_value,
                    scale_candidates,
                    case.data.kind
                    in (RecipeKind.BOUNDED, RecipeKind.BOUNDED_ALTERNATING_SIGN),
                    case.seed,
                    scale_index,
                )

        scale_raw[scale_index] = selected_scale_raw
        scale_value = decode_scale(case.scale_type, selected_scale_raw)
        for (row, column), source_value in zip(coordinates, block_values):
            logical_index = row + column * rows
            physical_index = row + column * leading_dimension
            scaled_value = (
                source_value if source_value == 0.0 else source_value / scale_value
            )
            data_raw = encode_binary(format_spec, scaled_value)
            if case.data.kind is RecipeKind.BOUNDED:
                data_raw = constrain_raw_to_interval(
                    format_spec,
                    data_raw,
                    scale_value,
                    case.data.lower,
                    case.data.upper,
                    decoded_data,
                )
            elif case.data.kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
                maximum = case.data.maximum_magnitude
                data_raw = constrain_raw_to_interval(
                    format_spec,
                    data_raw,
                    scale_value,
                    -maximum,
                    maximum,
                    decoded_data,
                )

            physical_raw[physical_index] = data_raw
            scale_indices[row, column] = scale_index
            with np.errstate(over="ignore", invalid="ignore"):
                reference[row, column] = np.float32(
                    decoded_data[data_raw] * scale_value
                )

    return ExpectedMx(
        data_storage=pack_little_endian(physical_raw, format_spec.storage_bits),
        scale_storage=bytes(scale_raw),
        scale_indices=scale_indices,
        reference=reference,
    )


def make_problem(case):
    problem = hv.MxGenerationProblem()
    problem.data_type = case.data_type
    problem.scale_type = case.scale_type
    problem.shape = hv.Shape(list(case.shape))
    problem.leading_dimension = case.leading_dimension
    problem.block_axis = case.block_axis
    problem.block_size = case.block_size
    problem.seed = case.seed

    problem.data = case.data.native()
    problem.scale = case.scale
    return problem


class MxGenerationOracleTests(unittest.TestCase):
    def assert_float_values_equal(self, observed, expected):
        observed = np.asarray(observed, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        not_nan = ~np.isnan(expected)
        np.testing.assert_array_equal(observed[not_nan], expected[not_nan])
        zeros = not_nan & (expected == 0.0)
        np.testing.assert_array_equal(
            np.signbit(observed[zeros]), np.signbit(expected[zeros])
        )

    def assert_matches_oracle(self, case):
        expected = expected_mx(case)
        observed = hv.generate_mx(make_problem(case))
        rows = case.shape[0]
        leading_dimension = case.leading_dimension or rows

        self.assertEqual(observed.data.strides, [1, leading_dimension])
        self.assertEqual(observed.data.storage, expected.data_storage)
        self.assertEqual(observed.scales.storage, expected.scale_storage)
        self.assertEqual(observed.scale_indices.strides, [1, rows])
        self.assertEqual(observed.reference.strides, [1, rows])
        np.testing.assert_array_equal(
            hv.to_numpy(observed.scale_indices, np.uint32),
            expected.scale_indices,
        )
        self.assert_float_values_equal(
            hv.to_numpy(observed.reference, np.float32),
            expected.reference,
        )
        return observed, expected

    def test_every_mx_data_recipe_matches_exact_oracle(self):
        recipes = (
            Recipe(RecipeKind.BOUNDED, lower=-1.25, upper=0.875),
            Recipe(RecipeKind.BOUNDED_ALTERNATING_SIGN, maximum_magnitude=1.25),
            Recipe(RecipeKind.UNBOUNDED),
            Recipe(RecipeKind.IDENTITY),
            Recipe(RecipeKind.CONSTANT, value=1.0),
            Recipe(RecipeKind.CONSTANT, value=0.0),
            Recipe(RecipeKind.SEQUENTIAL),
            Recipe(RecipeKind.ROW_INDEX),
            Recipe(RecipeKind.COLUMN_INDEX),
            Recipe(RecipeKind.CHECKERBOARD),
            Recipe(RecipeKind.SCALED_DIAGONAL),
            Recipe(RecipeKind.CONSTANT, value=2.0),
            Recipe(RecipeKind.CONSTANT, value=-1.0),
            Recipe(RecipeKind.TYPE_MAXIMUM),
            Recipe(RecipeKind.TYPE_DENORMAL_MINIMUM),
            Recipe(RecipeKind.TYPE_DENORMAL_MAXIMUM),
            Recipe(RecipeKind.TYPE_NAN),
            Recipe(RecipeKind.TYPE_INFINITY),
            Recipe(RecipeKind.TRIGONOMETRIC),
            Recipe(RecipeKind.NORMAL, mean=0.25, standard_deviation=0.75),
            Recipe(RecipeKind.UNIFORM_INTEGER, integer_lower=-3, integer_upper=4),
        )

        for recipe in recipes:
            case = MxCase(
                label=recipe.kind.name,
                data_type=hv.ScalarType.Float8E5M2,
                scale_type=hv.ScalarType.E8M0,
                shape=(3, 4),
                leading_dimension=5,
                block_axis=1,
                block_size=3,
                data=recipe,
            )
            with self.subTest(mode=case.label):
                self.assert_matches_oracle(case)

    def test_all_supported_pairs_axes_partial_blocks_and_scale_selection(self):
        cases = (
            MxCase(
                "fp4_e8m0_axis0",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E8M0,
                (5, 3),
                7,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp6_e2m3_e8m0_axis1",
                hv.ScalarType.Float6E2M3,
                hv.ScalarType.E8M0,
                (3, 5),
                5,
                1,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp6_e3m2_e8m0_axis0",
                hv.ScalarType.Float6E3M2,
                hv.ScalarType.E8M0,
                (5, 2),
                6,
                0,
                3,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp8_e4m3_e8m0_axis1",
                hv.ScalarType.Float8E4M3,
                hv.ScalarType.E8M0,
                (3, 5),
                4,
                1,
                2,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp8_e5m2_e8m0_axis0",
                hv.ScalarType.Float8E5M2,
                hv.ScalarType.E8M0,
                (5, 3),
                8,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp4_e4m3_axis1",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E4M3,
                (3, 5),
                5,
                1,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp4_e5m3_axis0",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E5M3,
                (5, 3),
                6,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
        )
        for case in cases:
            with self.subTest(case=case.label):
                _, expected = self.assert_matches_oracle(case)
                if case.scale_type in (hv.ScalarType.E4M3, hv.ScalarType.E5M3):
                    self.assertGreater(len(set(expected.scale_storage)), 1)

    def test_fp6_packing_tails_are_exact(self):
        for data_type in (hv.ScalarType.Float6E2M3, hv.ScalarType.Float6E3M2):
            for leading_dimension in (5, 6, 7):
                case = MxCase(
                    label=f"{data_type.name}_ld{leading_dimension}",
                    data_type=data_type,
                    scale_type=hv.ScalarType.E8M0,
                    shape=(5, 1),
                    leading_dimension=leading_dimension,
                    block_axis=0,
                    block_size=4,
                    data=Recipe(RecipeKind.UNBOUNDED),
                    seed=0xA5A5,
                )
                with self.subTest(
                    data_type=data_type.name, leading_dimension=leading_dimension
                ):
                    observed, _ = self.assert_matches_oracle(case)
                    self.assertEqual(
                        len(observed.data.storage),
                        (leading_dimension * 6 + 7) // 8,
                    )

    def test_independent_constant_scales_are_exact(self):
        constants = (
            (hv.ScalarType.E8M0, 0x80),
            (hv.ScalarType.E4M3, 0x40),
            (hv.ScalarType.E5M3, 0x80),
        )
        for scale_type, raw_two in constants:
            case = MxCase(
                label=f"constant_{scale_type.name}",
                data_type=hv.ScalarType.Float4E2M1,
                scale_type=scale_type,
                shape=(3, 3),
                leading_dimension=4,
                block_axis=0,
                block_size=2,
                data=Recipe(RecipeKind.SEQUENTIAL),
                scale=hv.MxScaleGenerationMode.Two,
            )
            with self.subTest(scale_type=scale_type.name):
                observed, _ = self.assert_matches_oracle(case)
                self.assertEqual(observed.scales.storage, bytes([raw_two]) * 6)

    def test_invalid_bounds_are_rejected(self):
        invalid_data_recipes = (
            (
                "bounded_equal",
                Recipe(RecipeKind.BOUNDED, lower=1.0, upper=1.0),
            ),
            (
                "bounded_reversed",
                Recipe(RecipeKind.BOUNDED, lower=2.0, upper=-1.0),
            ),
            (
                "bounded_nonfinite",
                Recipe(RecipeKind.BOUNDED, lower=-1.0, upper=math.inf),
            ),
            (
                "alternating_nonfinite",
                Recipe(
                    RecipeKind.BOUNDED_ALTERNATING_SIGN,
                    maximum_magnitude=math.inf,
                ),
            ),
            (
                "normal_negative_sigma",
                Recipe(RecipeKind.NORMAL, mean=0.0, standard_deviation=-0.1),
            ),
            (
                "normal_nan_mean",
                Recipe(
                    RecipeKind.NORMAL,
                    mean=math.nan,
                    standard_deviation=1.0,
                ),
            ),
            (
                "integer_reversed",
                Recipe(
                    RecipeKind.UNIFORM_INTEGER,
                    integer_lower=4,
                    integer_upper=-3,
                ),
            ),
        )
        for label, recipe in invalid_data_recipes:
            case = MxCase(
                label,
                hv.ScalarType.Float8E5M2,
                hv.ScalarType.E8M0,
                (3, 3),
                3,
                0,
                2,
                recipe,
            )
            with self.subTest(recipe=label):
                with self.assertRaises(ValueError):
                    hv.generate_mx(make_problem(case))

        with self.assertRaises(TypeError):
            hv.MxUniformIntegerDataParameters(0, 1 << 40)

        no_infinity = MxCase(
            "fp4_infinity",
            hv.ScalarType.Float4E2M1,
            hv.ScalarType.E8M0,
            (3, 3),
            3,
            0,
            2,
            Recipe(RecipeKind.TYPE_INFINITY),
        )
        with self.assertRaises(ValueError):
            hv.generate_mx(make_problem(no_infinity))

        impossible_interval = MxCase(
            "unrepresentable_bounded_interval",
            hv.ScalarType.Float4E2M1,
            hv.ScalarType.E8M0,
            (3, 1),
            3,
            0,
            2,
            Recipe(RecipeKind.BOUNDED, lower=0.1, upper=0.2),
            scale=hv.MxScaleGenerationMode.One,
        )
        with self.assertRaises(ValueError):
            hv.generate_mx(make_problem(impossible_interval))

    def test_invalid_geometry_and_type_pairs_are_rejected(self):
        base = dict(
            label="invalid",
            data_type=hv.ScalarType.Float4E2M1,
            scale_type=hv.ScalarType.E8M0,
            shape=(3, 3),
            leading_dimension=3,
            block_axis=0,
            block_size=2,
            data=Recipe(RecipeKind.CONSTANT, value=1.0),
        )
        invalid_cases = (
            ("rank", MxCase(**(base | {"shape": (3,)})), ValueError),
            ("zero_extent", MxCase(**(base | {"shape": (0, 3)})), ValueError),
            (
                "short_leading_dimension",
                MxCase(**(base | {"leading_dimension": 2})),
                ValueError,
            ),
            ("block_axis", MxCase(**(base | {"block_axis": 2})), IndexError),
            ("zero_block", MxCase(**(base | {"block_size": 0})), ValueError),
            (
                "unsupported_data",
                MxCase(**(base | {"data_type": hv.ScalarType.Float16})),
                ValueError,
            ),
            (
                "unsupported_scale",
                MxCase(**(base | {"scale_type": hv.ScalarType.Float32})),
                ValueError,
            ),
            (
                "unsupported_pair",
                MxCase(
                    **(
                        base
                        | {
                            "data_type": hv.ScalarType.Float6E2M3,
                            "scale_type": hv.ScalarType.E4M3,
                        }
                    )
                ),
                ValueError,
            ),
        )
        for label, case, error in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(error):
                    hv.generate_mx(make_problem(case))


if __name__ == "__main__":
    unittest.main()
