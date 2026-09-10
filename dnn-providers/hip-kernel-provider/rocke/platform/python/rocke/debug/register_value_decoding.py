# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decode physical register storage into typed semantic values.

The current decoders consume one 32-bit word. This layer also owns future
multi-register scalar encodings, such as an f64 assembled from two words.
"""

from __future__ import annotations

import math
import struct
from typing import Any

DTYPES = ("f32", "f16x2", "bf16x2", "fp8e4m3x4", "bf8e5m2x4")
FLOAT8_FORMATS = ("ocp", "fnuz")


def _element(
    raw: int, value: float, classification: str, negative: bool
) -> dict[str, Any]:
    finite = math.isfinite(value)
    return {
        "raw_bits": raw,
        "class": classification,
        "sign": -1 if negative else 1,
        "value": value if finite else None,
    }


def _classify_ieee(raw: int, exponent_bits: int, mantissa_bits: int) -> str:
    exponent = (raw >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = raw & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        return "zero" if mantissa == 0 else "subnormal"
    if exponent == (1 << exponent_bits) - 1:
        return "infinity" if mantissa == 0 else "nan"
    return "normal"


def _decode_f32(raw: int) -> dict[str, Any]:
    value = struct.unpack("<f", struct.pack("<I", raw))[0]
    return _element(raw, value, _classify_ieee(raw, 8, 23), bool(raw >> 31))


def _decode_f16(raw: int) -> dict[str, Any]:
    value = struct.unpack("<e", struct.pack("<H", raw))[0]
    return _element(raw, value, _classify_ieee(raw, 5, 10), bool(raw >> 15))


def _decode_bf16(raw: int) -> dict[str, Any]:
    value = struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    return _element(raw, value, _classify_ieee(raw, 8, 7), bool(raw >> 15))


def _decode_float8(
    raw: int,
    exponent_bits: int,
    mantissa_bits: int,
    finite_only: bool,
    fnuz: bool,
) -> dict[str, Any]:
    negative = bool(raw >> 7)
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    exponent = (raw >> mantissa_bits) & exponent_mask
    mantissa = raw & mantissa_mask
    bias = 1 << (exponent_bits - 1) if fnuz else (1 << (exponent_bits - 1)) - 1
    if fnuz and raw == 0x80:
        return _element(raw, math.nan, "nan", negative=True)
    if exponent == 0:
        if mantissa == 0:
            return _element(raw, -0.0 if negative else 0.0, "zero", negative)
        value = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
        classification = "subnormal"
    elif (
        not fnuz
        and exponent == exponent_mask
        and (not finite_only or mantissa == mantissa_mask)
    ):
        if finite_only or mantissa != 0:
            return _element(raw, math.nan, "nan", negative)
        return _element(
            raw,
            -math.inf if negative else math.inf,
            "infinity",
            negative,
        )
    else:
        value = math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
        classification = "normal"
    if negative:
        value = -value
    return _element(raw, value, classification, negative)


def decode_word_value(
    raw: int, dtype: str, float8_format: str = "ocp"
) -> list[dict[str, Any]]:
    """Decode one physical word without adding display strings or tile layout."""
    if dtype not in DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; choose one of {', '.join(DTYPES)}"
        )
    if float8_format not in FLOAT8_FORMATS:
        raise ValueError(
            f"unsupported float8 format {float8_format!r}; choose one of "
            f"{', '.join(FLOAT8_FORMATS)}"
        )
    if not 0 <= raw <= 0xFFFFFFFF:
        raise ValueError(f"register word must fit in 32 bits, got {raw}")
    if dtype == "f32":
        decoded = [_decode_f32(raw)]
    elif dtype == "f16x2":
        decoded = [_decode_f16((raw >> shift) & 0xFFFF) for shift in (0, 16)]
    elif dtype == "bf16x2":
        decoded = [_decode_bf16((raw >> shift) & 0xFFFF) for shift in (0, 16)]
    elif dtype == "fp8e4m3x4":
        decoded = [
            _decode_float8(
                (raw >> shift) & 0xFF,
                4,
                3,
                finite_only=True,
                fnuz=float8_format == "fnuz",
            )
            for shift in (0, 8, 16, 24)
        ]
    else:
        decoded = [
            _decode_float8(
                (raw >> shift) & 0xFF,
                5,
                2,
                finite_only=False,
                fnuz=float8_format == "fnuz",
            )
            for shift in (0, 8, 16, 24)
        ]
    for index, element in enumerate(decoded):
        element["index"] = index
    return decoded
