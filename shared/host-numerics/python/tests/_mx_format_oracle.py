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
from _numeric_oracle_support import decode_e8m0


GENERATION_REAL_RANDOM_DOMAIN = 0
MX_BOUNDED_SCALE_RANDOM_DOMAIN = 0xA24BAED4963EE407
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


__all__ = [name for name in globals() if not name.startswith("__")]
