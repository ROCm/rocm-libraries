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

import roc_host_numerics as hv
from _numeric_oracle_support import (
    counter_random,
    indexed_uniform_unit,
    pack_little_endian,
)


GENERATION_REAL_RANDOM_DOMAIN = 0
MX_BOUNDED_SCALE_RANDOM_DOMAIN = 0xA24BAED4963EE407
MX_UNBOUNDED_SCALE_RANDOM_DOMAIN = 0x94D049BB133111EB
TWO_PI = 6.28318530717958647692528676655900576

from _mx_format_oracle import *  # noqa: F403


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

    def native(self, shape, seed):
        def recipe(component):
            return hv.GenerationRecipe.real_only(
                component,
                hv.GenerationRecipeSettings(
                    seed=seed,
                    index_order=hv.IndexOrder.FirstDimensionFastest,
                ),
            )

        if self.kind is RecipeKind.UNBOUNDED:
            return (
                recipe(hv.GenerationRecipe.uniform_finite_encoded_value()),
                hv.MxDataQuantization.PreserveGeneratedEncoding,
                None,
                hv.MxScaleGenerationMode.RandomFinite,
            )
        if self.kind is RecipeKind.IDENTITY:
            component = hv.GenerationRecipe.identity()
        elif self.kind is RecipeKind.SEQUENTIAL:
            component = hv.GenerationRecipe.affine_index_remainder(
                hv.AffineIndexRemainderGenerationParameters([shape[1] % 256, 1], 0, 256)
            )
        elif self.kind is RecipeKind.ROW_INDEX:
            component = hv.GenerationRecipe.affine_index_remainder(
                hv.AffineIndexRemainderGenerationParameters([1, 0], 0, 256)
            )
        elif self.kind is RecipeKind.COLUMN_INDEX:
            component = hv.GenerationRecipe.affine_index_remainder(
                hv.AffineIndexRemainderGenerationParameters([0, 1], 0, 256)
            )
        elif self.kind is RecipeKind.CHECKERBOARD:
            component = hv.GenerationRecipe.affine_index_remainder(
                hv.AffineIndexRemainderGenerationParameters([1, 1], 0, 2)
            ).with_affine_value_mapping(hv.GenerationAffineValueParameters(-1.0, 1.0))
        elif self.kind is RecipeKind.SCALED_DIAGONAL:
            component = (
                hv.GenerationRecipe.serial_dimension(
                    hv.DimensionGenerationParameters(0)
                )
                .with_affine_value_mapping(hv.GenerationAffineValueParameters(1.0, 1.0))
                .with_zero_outside_main_diagonal()
            )
        elif self.kind is RecipeKind.TYPE_MAXIMUM:
            component = hv.GenerationRecipe.type_maximum()
        elif self.kind is RecipeKind.TYPE_DENORMAL_MINIMUM:
            component = hv.GenerationRecipe.type_denormal_minimum()
        elif self.kind is RecipeKind.TYPE_DENORMAL_MAXIMUM:
            component = hv.GenerationRecipe.type_denormal_maximum()
        elif self.kind is RecipeKind.TYPE_NAN:
            component = hv.GenerationRecipe.constant(
                hv.ConstantGenerationParameters(math.nan)
            )
        elif self.kind is RecipeKind.TYPE_INFINITY:
            component = hv.GenerationRecipe.type_infinity()
        elif self.kind is RecipeKind.TRIGONOMETRIC:
            component = hv.GenerationRecipe.uniform_real(
                hv.UniformRealGenerationParameters(0.0, TWO_PI)
            ).with_cosine_transform()
        if self.kind is RecipeKind.BOUNDED:
            return (
                recipe(
                    hv.GenerationRecipe.uniform_real(
                        hv.UniformRealGenerationParameters(self.lower, self.upper)
                    )
                ),
                hv.MxDataQuantization.PreserveRange,
                hv.MxRepresentedValueRange(self.lower, self.upper),
                None,
            )
        if self.kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
            dimensions = [0] + ([1] if shape[0] % 2 else [])
            component = hv.GenerationRecipe.uniform_real(
                hv.UniformRealGenerationParameters(0.0, self.maximum_magnitude)
            ).with_alternating_sign(
                hv.AlternatingSignGenerationParameters(dimensions, True)
            )
            return (
                recipe(component),
                hv.MxDataQuantization.PreserveRange,
                hv.MxRepresentedValueRange(
                    -self.maximum_magnitude, self.maximum_magnitude
                ),
                None,
            )
        if self.kind is RecipeKind.CONSTANT:
            component = hv.GenerationRecipe.constant(
                hv.ConstantGenerationParameters(self.value)
            )
        elif self.kind is RecipeKind.NORMAL:
            return (
                recipe(
                    hv.GenerationRecipe.normal(
                        hv.NormalGenerationParameters(
                            self.mean, self.standard_deviation
                        )
                    )
                ),
                hv.MxDataQuantization.Nearest,
                None,
                None,
            )
        elif self.kind is RecipeKind.UNIFORM_INTEGER:
            component = hv.GenerationRecipe.uniform_integer(
                hv.UniformIntegerGenerationParameters(
                    self.integer_lower, self.integer_upper
                )
            )
        elif self.kind not in {
            RecipeKind.IDENTITY,
            RecipeKind.SEQUENTIAL,
            RecipeKind.ROW_INDEX,
            RecipeKind.COLUMN_INDEX,
            RecipeKind.CHECKERBOARD,
            RecipeKind.SCALED_DIAGONAL,
            RecipeKind.TYPE_MAXIMUM,
            RecipeKind.TYPE_DENORMAL_MINIMUM,
            RecipeKind.TYPE_DENORMAL_MAXIMUM,
            RecipeKind.TYPE_NAN,
            RecipeKind.TYPE_INFINITY,
            RecipeKind.TRIGONOMETRIC,
        }:
            raise AssertionError(f"no native MX recipe for {self.kind}")
        return recipe(component), hv.MxDataQuantization.Nearest, None, None


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


def indexed_uniform_integer(seed, stream, index, lower, upper):
    width = upper - lower + 1
    return lower + counter_random(seed, stream, index) % width


def explicit_scale_raw(scale_type, mode):
    if mode in (
        hv.MxScaleGenerationMode.Derived,
        hv.MxScaleGenerationMode.RandomFinite,
    ):
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
        unit = indexed_uniform_unit(
            case.seed, GENERATION_REAL_RANDOM_DOMAIN, logical_index
        )
        return recipe.lower + unit * (recipe.upper - recipe.lower)
    if kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
        magnitude = recipe.maximum_magnitude * indexed_uniform_unit(
            case.seed, GENERATION_REAL_RANDOM_DOMAIN, logical_index
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
        unit = indexed_uniform_unit(
            case.seed, GENERATION_REAL_RANDOM_DOMAIN, logical_index
        )
        return math.cos(TWO_PI * unit)
    if kind is RecipeKind.NORMAL:
        first_index = logical_index * 2
        first = indexed_uniform_unit(
            case.seed, GENERATION_REAL_RANDOM_DOMAIN, first_index
        )
        second = indexed_uniform_unit(
            case.seed, GENERATION_REAL_RANDOM_DOMAIN, first_index + 1
        )
        standard_normal = math.sqrt(-2.0 * math.log(first)) * math.cos(TWO_PI * second)
        return recipe.mean + recipe.standard_deviation * standard_normal
    if kind is RecipeKind.UNIFORM_INTEGER:
        return indexed_uniform_integer(
            case.seed,
            GENERATION_REAL_RANDOM_DOMAIN,
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


__all__ = [name for name in globals() if not name.startswith("__")]
