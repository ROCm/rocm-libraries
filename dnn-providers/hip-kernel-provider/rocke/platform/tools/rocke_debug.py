#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decode stopped-wave AMDGPU registers in rocGDB.

Source this file from rocGDB, then use ``rocke decode``::

    (gdb) source tools/rocke_debug.py
    (gdb) rocke decode $v40 --dtype f32
    (gdb) rocke decode $v41 --dtype fp8e4m3x4 --format jsonl
    (gdb) rocke value acc --manifest debug-manifest.json

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
from collections import Counter
from collections.abc import Sequence
from typing import Any

SCHEMA = "rocke-register-v1"
VALUE_SCHEMA = "rocke-debug-value/v1"
MANIFEST_SCHEMA = "rocke-debug-manifest/v1"
DTYPES = ("f32", "f16x2", "bf16x2", "fp8e4m3x4", "bf8e5m2x4")
FLOAT8_FORMATS = ("ocp", "fnuz")
VALUE_STATUSES = (
    "available",
    "replica_mismatch",
    "optimized_out",
    "location_unavailable",
    "inactive_lane",
    "stale_manifest",
    "unsupported_dtype",
    "unsupported_layout",
)
_LOGICAL_STORAGE_DTYPES = {
    "f16": "f16x2",
    "bf16": "bf16x2",
    "f32": "f32",
    "fp8e4m3": "fp8e4m3x4",
    "bf8e5m2": "bf8e5m2x4",
}


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


def load_manifest(path: str) -> dict[str, Any]:
    """Load and minimally validate a portable rocKE debug manifest."""
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load debug manifest {path!r}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        actual = manifest.get("schema") if isinstance(manifest, dict) else None
        raise ValueError(
            f"unsupported debug manifest schema {actual!r}; expected {MANIFEST_SCHEMA!r}"
        )
    values = manifest.get("values")
    if not isinstance(values, list):
        raise TypeError("debug manifest 'values' must be a list")
    if any(not isinstance(value, dict) for value in values):
        raise TypeError("every debug manifest value must be an object")
    return manifest


def manifest_value(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one uniquely named logical-value entry from ``manifest``."""
    matches = [
        value
        for value in manifest["values"]
        if value.get("logical", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"debug manifest must contain exactly one value named {name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _expanded_indices(
    index: list[int], packing: dict[str, Any]
) -> list[tuple[int, int]]:
    """Expand one physical slot's base coordinate into logical scalars."""
    kind = packing.get("kind")
    elements_per_slot = packing.get("elements_per_slot")
    if kind == "scalar" and elements_per_slot == 1:
        return [tuple(index)]
    axis = packing.get("axis")
    if (
        kind != "contiguous"
        or axis not in (0, 1)
        or not isinstance(elements_per_slot, int)
        or elements_per_slot <= 1
    ):
        raise ValueError(f"invalid layout packing {packing!r}")
    expanded = []
    for offset in range(elements_per_slot):
        logical = list(index)
        logical[axis] += offset
        expanded.append(tuple(logical))
    return expanded


def _validated_value_spec(
    value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int, int, int]:
    """Validate a logical-value manifest entry and return layout dimensions."""
    logical = value.get("logical")
    binding = value.get("binding")
    if not isinstance(logical, dict):
        raise TypeError("manifest value has no logical description")
    if not isinstance(binding, dict):
        raise TypeError("manifest value has no physical binding")
    if not isinstance(logical.get("name"), str) or not logical["name"]:
        raise ValueError("logical value name must be a non-empty string")
    dtype = logical.get("dtype")
    storage_dtype = binding.get("storage_dtype")
    if dtype not in ("f16", "bf16", "f32", "fp8e4m3", "bf8e5m2"):
        raise ValueError(f"unsupported logical dtype {dtype!r}")
    if storage_dtype not in DTYPES:
        raise ValueError(f"unsupported storage dtype {storage_dtype!r}")
    expected_storage = _LOGICAL_STORAGE_DTYPES[dtype]
    if storage_dtype != expected_storage:
        raise ValueError(
            f"logical dtype {dtype!r} requires storage dtype {expected_storage!r}, "
            f"not {storage_dtype!r}"
        )

    shape = logical.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(extent, int) or extent <= 0 for extent in shape)
    ):
        raise ValueError(
            f"value {logical.get('name')!r} has invalid tile shape {shape!r}"
        )
    layout = logical.get("layout")
    if not isinstance(layout, dict):
        raise TypeError(f"value {logical.get('name')!r} has no layout object")
    if not isinstance(layout.get("name"), str) or not layout["name"]:
        raise ValueError("layout name must be a non-empty string")
    if layout.get("role") not in ("a", "b", "acc"):
        raise ValueError(f"unsupported layout role {layout.get('role')!r}")
    wave_size = layout.get("wave_size")
    fragment_length = layout.get("fragment_length")
    if not isinstance(wave_size, int) or wave_size <= 0:
        raise ValueError(f"invalid layout wave size {wave_size!r}")
    if not isinstance(fragment_length, int) or fragment_length <= 0:
        raise ValueError(f"invalid layout fragment length {fragment_length!r}")
    replication_factor = layout.get("replication_factor")
    if not isinstance(replication_factor, int) or replication_factor <= 0:
        raise ValueError(f"invalid layout replication factor {replication_factor!r}")
    packing = layout.get("packing")
    if not isinstance(packing, dict):
        raise TypeError("layout packing must be an object")

    if binding.get("kind") != "amdgpu_registers":
        raise ValueError(f"unsupported physical binding {binding.get('kind')!r}")
    locations = binding.get("locations")
    if not isinstance(locations, list) or any(
        not isinstance(location, str) or not location for location in locations
    ):
        raise ValueError("value locations must be a list of non-empty expressions")
    packed_width = len(decode_word(0, storage_dtype))
    if len(locations) * packed_width != fragment_length:
        raise ValueError(
            f"{len(locations)} {storage_dtype} locations provide "
            f"{len(locations) * packed_width} elements, but layout requires "
            f"{fragment_length}"
        )

    coordinates = layout.get("coordinates")
    if not isinstance(coordinates, list):
        raise TypeError("layout coordinates must be a list")
    if len(coordinates) != wave_size * fragment_length:
        raise ValueError(
            f"layout has {len(coordinates)} coordinates; expected "
            f"{wave_size * fragment_length}"
        )
    seen_physical = set()
    logical_sources: Counter[tuple[int, int]] = Counter()
    for coordinate in coordinates:
        if not isinstance(coordinate, dict):
            raise TypeError("every layout coordinate must be an object")
        lane = coordinate.get("lane")
        slot = coordinate.get("slot")
        index = coordinate.get("index")
        if (
            not isinstance(lane, int)
            or not 0 <= lane < wave_size
            or not isinstance(slot, int)
            or not 0 <= slot < fragment_length
            or not isinstance(index, list)
            or len(index) != 2
            or any(not isinstance(axis, int) for axis in index)
        ):
            raise ValueError(f"invalid layout coordinate {coordinate!r}")
        physical = (lane, slot)
        if physical in seen_physical:
            raise ValueError(f"duplicate physical layout coordinate {physical!r}")
        seen_physical.add(physical)
        for logical_index in _expanded_indices(index, packing):
            if not (
                0 <= logical_index[0] < shape[0] and 0 <= logical_index[1] < shape[1]
            ):
                raise ValueError(
                    f"layout coordinate {logical_index!r} is outside shape {shape!r}"
                )
            logical_sources[logical_index] += 1

    expected_indices = {
        (axis0, axis1) for axis0 in range(shape[0]) for axis1 in range(shape[1])
    }
    if set(logical_sources) != expected_indices:
        missing = sorted(expected_indices - set(logical_sources))
        raise ValueError(
            f"layout does not cover shape {shape!r}; missing {missing[:4]!r}"
        )
    wrong_multiplicity = {
        index: count
        for index, count in logical_sources.items()
        if count != replication_factor
    }
    if wrong_multiplicity:
        sample = next(iter(sorted(wrong_multiplicity.items())))
        raise ValueError(
            f"layout source multiplicity does not match replication factor "
            f"{replication_factor}; first mismatch {sample!r}"
        )
    return logical, binding, wave_size, fragment_length, packed_width


def unavailable_value(
    value: dict[str, Any], status: str, detail: str
) -> dict[str, Any]:
    """Build an explicit unavailable logical-value record."""
    if status not in VALUE_STATUSES or status in (
        "available",
        "inactive_lane",
        "replica_mismatch",
    ):
        raise ValueError(f"invalid unavailable status {status!r}")
    logical = value.get("logical") or {}
    binding = value.get("binding") or {}
    return {
        "schema": VALUE_SCHEMA,
        "name": logical.get("name"),
        "dtype": logical.get("dtype"),
        "shape": logical.get("shape"),
        "status": status,
        "detail": detail,
        "machine_locations": binding.get("locations", []),
        "layout": logical.get("layout"),
        "elements": [],
        "tile": None,
    }


def unavailable_status_for_error(error: Exception) -> str:
    """Classify manifest/decoder failures for machine-readable output."""
    message = str(error).lower()
    if "dtype" in message:
        return "unsupported_dtype"
    if "layout" in message or "coordinate" in message or "shape" in message:
        return "unsupported_layout"
    return "stale_manifest"


def decode_logical_value(
    value: dict[str, Any],
    raw_locations: Sequence[Sequence[int]],
    exec_mask: int | None = None,
    float8_format: str = "ocp",
) -> dict[str, Any]:
    """Reconstruct a logical tile from lane-major physical register words."""
    logical, binding, wave_size, fragment_length, packed_width = _validated_value_spec(
        value
    )
    if logical["layout"]["packing"]["kind"] != "scalar":
        raise ValueError("packed logical fragment slots are not yet decodable")
    locations = binding["locations"]
    if len(raw_locations) != len(locations):
        raise ValueError(
            f"received {len(raw_locations)} physical locations; expected {len(locations)}"
        )
    if any(len(words) != wave_size for words in raw_locations):
        lengths = [len(words) for words in raw_locations]
        raise ValueError(
            f"physical location lane counts {lengths!r} do not match wave size {wave_size}"
        )

    coordinate_by_physical = {
        (coordinate["lane"], coordinate["slot"]): coordinate["index"]
        for coordinate in logical["layout"]["coordinates"]
    }
    shape = logical["shape"]
    sources_by_index: dict[tuple[int, int], list[dict[str, Any]]] = {}
    elements = []
    storage_dtype = binding["storage_dtype"]
    for location_index, (location, words) in enumerate(zip(locations, raw_locations)):
        for lane, word in enumerate(words):
            decoded = decode_word(
                int(word) & 0xFFFFFFFF,
                storage_dtype,
                float8_format=float8_format,
            )
            for packed_index, scalar in enumerate(decoded):
                slot = location_index * packed_width + packed_index
                if slot >= fragment_length:
                    continue
                index = coordinate_by_physical[(lane, slot)]
                active = None if exec_mask is None else bool(exec_mask & (1 << lane))
                status = "inactive_lane" if active is False else "available"
                element = {
                    "index": index,
                    "lane": lane,
                    "slot": slot,
                    "active": active,
                    "status": status,
                    "machine_location": location,
                    "packed_index": packed_index,
                    "raw_bits": scalar["raw_bits"],
                    "raw_hex": scalar["raw_hex"],
                    "class": scalar["class"],
                    "sign": scalar["sign"],
                    "value": scalar["value"],
                    "value_text": scalar["value_text"],
                }
                elements.append(element)
                sources_by_index.setdefault(tuple(index), []).append(element)

    tile: list[list[dict[str, Any]]] = []
    has_replica_mismatch = False
    for row in range(shape[0]):
        cells = []
        for column in range(shape[1]):
            index = [row, column]
            sources = sources_by_index[(row, column)]
            active_sources = [source for source in sources if source["active"] is True]
            unknown_sources = [source for source in sources if source["active"] is None]
            if active_sources:
                comparable = active_sources
                active = True
            elif unknown_sources:
                comparable = unknown_sources
                active = None
            else:
                comparable = sources
                active = False

            representative = comparable[0]
            agrees = all(
                source["raw_bits"] == representative["raw_bits"]
                for source in comparable[1:]
            )
            status = "inactive_lane" if active is False else "available"
            if not agrees and active is not False:
                status = "replica_mismatch"
                has_replica_mismatch = True
            cells.append(
                {
                    "index": index,
                    "active": active,
                    "status": status,
                    "source_count": len(sources),
                    "sources": sources,
                    "raw_bits": representative["raw_bits"] if agrees else None,
                    "raw_hex": representative["raw_hex"] if agrees else None,
                    "class": representative["class"] if agrees else None,
                    "sign": representative["sign"] if agrees else None,
                    "value": representative["value"] if agrees else None,
                    "value_text": (
                        representative["value_text"] if agrees else "<replica-mismatch>"
                    ),
                }
            )
        tile.append(cells)

    return {
        "schema": VALUE_SCHEMA,
        "name": logical["name"],
        "dtype": logical["dtype"],
        "storage_dtype": storage_dtype,
        "float8_format": float8_format if "8" in storage_dtype else None,
        "shape": shape,
        "status": "replica_mismatch" if has_replica_mismatch else "available",
        "detail": ("observable replicas disagree" if has_replica_mismatch else None),
        "exec_mask": None if exec_mask is None else f"0x{exec_mask:x}",
        "machine_locations": locations,
        "layout": {
            key: logical["layout"][key]
            for key in (
                "name",
                "role",
                "wave_size",
                "fragment_length",
                "replication_factor",
                "packing",
            )
        },
        "elements": elements,
        "tile": tile,
    }


def values_human(records: Sequence[dict[str, Any]]) -> str:
    """Render logical values and reconstructed tiles for humans."""
    lines = []
    for record in records:
        shape = "x".join(str(extent) for extent in record.get("shape") or [])
        layout = record.get("layout") or {}
        lines.append(
            f"{record.get('name')} {record.get('dtype')} [{shape}] "
            f"layout={layout.get('name', '?')} status={record['status']}"
        )
        if record["status"] not in ("available", "replica_mismatch"):
            lines.append(f"  {record.get('detail', '')}")
            continue
        if record["status"] == "replica_mismatch":
            lines.append(f"  {record['detail']}")
        lines.append("  locations: " + ", ".join(record["machine_locations"]))
        lines.append("  inactive lanes are prefixed with ~; unknown activity with ?")
        for row, cells in enumerate(record["tile"]):
            rendered = []
            for cell in cells:
                if cell is None:
                    rendered.append("<missing>")
                else:
                    prefix = (
                        "~"
                        if cell["active"] is False
                        else ("?" if cell["active"] is None else "")
                    )
                    rendered.append(prefix + cell["value_text"])
            lines.append(f"  {row:>3}: " + " ".join(rendered))
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


def _value_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke value", add_help=False)
    parser.add_argument("name")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--format", choices=("human", "jsonl"), default="human")
    parser.add_argument("--float8-format", choices=FLOAT8_FORMATS, default="ocp")
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

    class RockeValue(gdb.Command):
        """Render a logical value: rocke value NAME --manifest PATH."""

        def __init__(self) -> None:
            super().__init__("rocke value", gdb.COMMAND_DATA)

        def invoke(self, argument: str, from_tty: bool) -> None:
            del from_tty
            try:
                args = _value_argument_parser().parse_args(shlex.split(argument))
                value = manifest_value(load_manifest(args.manifest), args.name)
                raw_locations = []
                try:
                    binding = value.get("binding") or {}
                    for expression in binding.get("locations", []):
                        raw_locations.append(_gdb_words(gdb.parse_and_eval(expression)))
                except (gdb.error, RuntimeError) as error:
                    message = str(error)
                    status = (
                        "optimized_out"
                        if "optimized out" in message.lower()
                        else "location_unavailable"
                    )
                    record = unavailable_value(value, status, message)
                else:
                    try:
                        exec_mask = int(gdb.parse_and_eval(args.exec_expression))
                    except (gdb.error, RuntimeError):
                        exec_mask = None
                    try:
                        record = decode_logical_value(
                            value,
                            raw_locations,
                            exec_mask=exec_mask,
                            float8_format=args.float8_format,
                        )
                    except (TypeError, ValueError) as error:
                        record = unavailable_value(
                            value, unavailable_status_for_error(error), str(error)
                        )
                rendered = (
                    records_jsonl([record])
                    if args.format == "jsonl"
                    else values_human([record])
                )
                gdb.write(rendered + "\n")
            except (TypeError, ValueError, RuntimeError, gdb.error) as error:
                raise gdb.GdbError(str(error)) from error

    RockePrefix()
    RockeDecode()
    RockeValue()
