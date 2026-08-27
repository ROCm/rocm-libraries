# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render semantic rocKE logical values for interactive and offline output."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .logical_value_reconstruction import (
    reconstruct_logical_value,
    unavailable_logical_value,
)
from .register_value_decoding import decode_word_value

VALUE_SCHEMA = "rocke-debug-value/v1"


def value_text(
    value: float | None, classification: str | None, sign: int | None
) -> str:
    """Format one decoded scalar without changing its semantic record."""
    if classification == "nan":
        return "nan"
    if classification == "infinity":
        return "-inf" if sign == -1 else "inf"
    if classification == "zero":
        return "-0" if sign == -1 else "0"
    return repr(value)


def present_elements(
    elements: Sequence[dict[str, Any]], bit_width: int
) -> list[dict[str, Any]]:
    """Add display text and hexadecimal fields to decoded scalar elements."""
    presented = deepcopy(list(elements))
    for element in presented:
        element["raw_hex"] = f"0x{element['raw_bits']:0{bit_width // 4}x}"
        element["value_text"] = value_text(
            element["value"], element["class"], element["sign"]
        )
    return presented


def decode_word(
    raw: int, dtype: str, float8_format: str = "ocp"
) -> list[dict[str, Any]]:
    """Decode one word into the presentation-rich interactive representation."""
    decoded = decode_word_value(raw, dtype, float8_format=float8_format)
    width = 32 // len(decoded)
    return present_elements(decoded, width)


def _render_cell_value(cell: dict[str, Any]) -> str:
    if cell.get("status") == "replica_mismatch":
        return "<replica-mismatch>"
    return value_text(cell.get("value"), cell.get("class"), cell.get("sign"))


def present_logical_value(
    record: dict[str, Any], *, schema: str, bit_width: int
) -> dict[str, Any]:
    """Add presentation fields to a semantic logical-value record."""
    result = deepcopy(record)
    result["schema"] = schema
    for element in result.get("elements", []):
        element["raw_hex"] = f"0x{element['raw_bits']:0{bit_width // 4}x}"
        element["value_text"] = _render_cell_value(element)
    for row in result.get("tile") or []:
        for cell in row:
            if cell["raw_bits"] is not None:
                cell["raw_hex"] = f"0x{cell['raw_bits']:0{bit_width // 4}x}"
            else:
                cell["raw_hex"] = None
            cell["value_text"] = _render_cell_value(cell)
            for source in cell["sources"]:
                source["raw_hex"] = (
                    f"0x{source['raw_bits']:0{bit_width // 4}x}"
                )
                source["value_text"] = _render_cell_value(source)
    return result


def unavailable_value(
    value: dict[str, Any], status: str, detail: str
) -> dict[str, Any]:
    """Build an unavailable value for the interactive representation."""
    record = unavailable_logical_value(value, status, detail)
    record["schema"] = VALUE_SCHEMA
    return record


def decode_logical_value(
    value: dict[str, Any],
    raw_locations: Sequence[Sequence[int]],
    exec_mask: int | None = None,
    float8_format: str = "ocp",
) -> dict[str, Any]:
    """Reconstruct a logical value for interactive display or JSON output."""
    record = reconstruct_logical_value(
        value,
        raw_locations,
        exec_mask=exec_mask,
        float8_format=float8_format,
    )
    storage_dtype = record.get("storage_dtype")
    packed_width = len(decode_word_value(0, storage_dtype)) if storage_dtype else 1
    return present_logical_value(
        record, schema=VALUE_SCHEMA, bit_width=32 // packed_width
    )


def values_human(records: Sequence[dict[str, Any]]) -> str:
    """Render semantic or presentation-rich logical-value records for humans."""
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
                prefix = (
                    "~"
                    if cell["active"] is False
                    else ("?" if cell["active"] is None else "")
                )
                rendered.append(prefix + _render_cell_value(cell))
            lines.append(f"  {row:>3}: " + " ".join(rendered))
    return "\n".join(lines)
