# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gc
import math
import unittest
import weakref

import numpy as np

import roc_host_numerics as hv
from roc_host_numerics import _roc_host_numerics as native
from gemm_test_adapter import reference_gemm, reference_gemm_into
from _numeric_oracle_support import *  # noqa: F403

GENERATION_REAL_RANDOM_DOMAIN = 0
GENERATION_IMAGINARY_RANDOM_DOMAIN = 0x243F6A8885A308D3
MX_BOUNDED_SCALE_RANDOM_DOMAIN = 0xA24BAED4963EE407


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
            unit = indexed_uniform_unit(
                seed, GENERATION_REAL_RANDOM_DOMAIN, logical_index
            )
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


__all__ = [name for name in globals() if not name.startswith("__")]
