#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decode stopped-wave AMDGPU registers in rocGDB.

Source this file from rocGDB, then use ``rocke decode``::

    (gdb) source tools/rocke_debug.py
    (gdb) rocke decode $v40 --dtype f32
    (gdb) rocke decode $v41 --dtype fp8e4m3x4 --format jsonl

The decoder is deliberately independent of rocGDB. Ordinary Python tests can
verify dtype semantics and structured output without a GPU or a stopped wave.
The rocGDB adapter only reads register expressions and the EXEC mask.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import struct
from collections.abc import Sequence
from typing import Any

SCHEMA = "rocke-register-v1"
DTYPES = ("f32", "f16x2", "bf16x2", "fp8e4m3x4", "bf8e5m2x4")
FLOAT8_FORMATS = ("ocp", "fnuz")


def _value_text(value: float, classification: str, negative: bool) -> str:
    if classification == "nan":
        return "nan"
    if classification == "infinity":
        return "-inf" if negative else "inf"
    if classification == "zero":
        return "-0" if negative else "0"
    return repr(value)


def _element(
    raw: int,
    width: int,
    value: float,
    classification: str,
    negative: bool,
) -> dict[str, Any]:
    finite = math.isfinite(value)
    return {
        "raw_bits": raw,
        "raw_hex": f"0x{raw:0{width // 4}x}",
        "class": classification,
        "sign": -1 if negative else 1,
        "value": value if finite else None,
        "value_text": _value_text(value, classification, negative),
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
    return _element(raw, 32, value, _classify_ieee(raw, 8, 23), bool(raw >> 31))


def _decode_f16(raw: int) -> dict[str, Any]:
    value = struct.unpack("<e", struct.pack("<H", raw))[0]
    return _element(raw, 16, value, _classify_ieee(raw, 5, 10), bool(raw >> 15))


def _decode_bf16(raw: int) -> dict[str, Any]:
    value = struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    return _element(raw, 16, value, _classify_ieee(raw, 8, 7), bool(raw >> 15))


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
        return _element(raw, 8, math.nan, "nan", negative=True)

    if exponent == 0:
        if mantissa == 0:
            return _element(raw, 8, -0.0 if negative else 0.0, "zero", negative)
        value = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
        classification = "subnormal"
    elif (
        not fnuz
        and exponent == exponent_mask
        and (not finite_only or mantissa == mantissa_mask)
    ):
        if finite_only or mantissa != 0:
            return _element(raw, 8, math.nan, "nan", negative)
        return _element(
            raw, 8, -math.inf if negative else math.inf, "infinity", negative
        )
    else:
        value = math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
        classification = "normal"

    if negative:
        value = -value
    return _element(raw, 8, value, classification, negative)


def decode_word(
    raw: int, dtype: str, float8_format: str = "ocp"
) -> list[dict[str, Any]]:
    """Decode one 32-bit register word into ordered packed elements.

    Packed elements are returned least-significant first, matching AMDGPU
    register byte order. ``fp8e4m3`` uses the OCP finite-only E4M3 encoding;
    ``bf8e5m2`` uses the OCP E5M2 encoding by default. Pass
    ``float8_format="fnuz"`` for targets whose conversion instructions use
    the legacy FNUZ encodings (bias 8/16, ``0x80`` NaN, no infinities).
    """
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


def decode_register(
    register: str,
    raw_words: Sequence[int],
    dtype: str,
    exec_mask: int | None = None,
    float8_format: str = "ocp",
) -> list[dict[str, Any]]:
    """Build stable per-lane records for one stopped-wave register."""
    records = []
    for lane, word in enumerate(raw_words):
        raw = int(word) & 0xFFFFFFFF
        records.append(
            {
                "schema": SCHEMA,
                "register": register,
                "lane": lane,
                "active": None if exec_mask is None else bool(exec_mask & (1 << lane)),
                "raw_bits": raw,
                "raw_hex": f"0x{raw:08x}",
                "dtype": dtype,
                "float8_format": float8_format if "8" in dtype else None,
                "elements": decode_word(raw, dtype, float8_format=float8_format),
            }
        )
    return records


def records_jsonl(records: Sequence[dict[str, Any]]) -> str:
    """Serialize records as strict, deterministic JSON Lines."""
    return "\n".join(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        for record in records
    )


def records_human(records: Sequence[dict[str, Any]]) -> str:
    """Render records as a compact table for an interactive rocGDB session."""
    lines = ["register lane active raw        dtype        values"]
    for record in records:
        active = (
            "?" if record["active"] is None else ("yes" if record["active"] else "no")
        )
        values = ", ".join(element["value_text"] for element in record["elements"])
        lines.append(
            f"{record['register']:<8} {record['lane']:>4} {active:<6} "
            f"{record['raw_hex']} {record['dtype']:<12} [{values}]"
        )
    return "\n".join(lines)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke decode", add_help=False)
    parser.add_argument("expression")
    parser.add_argument("--dtype", required=True, choices=DTYPES)
    parser.add_argument("--format", choices=("human", "jsonl"), default="human")
    parser.add_argument("--float8-format", choices=FLOAT8_FORMATS, default="ocp")
    parser.add_argument("--lane", action="append", type=int)
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--exec", dest="exec_expression", default="$exec")
    parser.add_argument("--help", action="help")
    return parser


try:
    import gdb  # type: ignore
except ModuleNotFoundError:
    gdb = None


if gdb is not None:

    def _gdb_words(value: Any) -> list[int]:
        try:
            lower, upper = value.type.range()
        except (gdb.error, RuntimeError):
            return [int(value)]
        return [int(value[index]) for index in range(lower, upper + 1)]

    class RockePrefix(gdb.Command):
        """rocKE commands for stopped AMDGPU waves."""

        def __init__(self) -> None:
            super().__init__("rocke", gdb.COMMAND_USER, prefix=True)

    class RockeDecode(gdb.Command):
        """Decode a physical register: rocke decode EXPR --dtype DTYPE."""

        def __init__(self) -> None:
            super().__init__("rocke decode", gdb.COMMAND_DATA)

        def invoke(self, argument: str, from_tty: bool) -> None:
            del from_tty
            try:
                args = _argument_parser().parse_args(shlex.split(argument))
                words = _gdb_words(gdb.parse_and_eval(args.expression))
                try:
                    exec_mask = int(gdb.parse_and_eval(args.exec_expression))
                except (gdb.error, RuntimeError):
                    exec_mask = None
                records = decode_register(
                    args.expression,
                    words,
                    args.dtype,
                    exec_mask=exec_mask,
                    float8_format=args.float8_format,
                )
                if args.lane is not None:
                    selected = set(args.lane)
                    records = [r for r in records if r["lane"] in selected]
                if args.active_only:
                    records = [r for r in records if r["active"] is True]
                rendered = (
                    records_jsonl(records)
                    if args.format == "jsonl"
                    else records_human(records)
                )
                if rendered:
                    gdb.write(rendered + "\n")
            except (ValueError, RuntimeError, gdb.error) as error:
                raise gdb.GdbError(str(error)) from error

    RockePrefix()
    RockeDecode()
