# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Portable semantic manifests for headless rocGDB value rendering.

The compiler owns the PC-specific physical location of a value. rocKE owns the
logical dtype, shape, and lane/fragment layout. This module exports the latter
without duplicating the MMA coordinate formulas in debugger code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .arch import LayoutMap

DEBUG_MANIFEST_SCHEMA = "rocke-debug-manifest/v1"
_STORAGE_WIDTHS = {
    "f32": 1,
    "f16x2": 2,
    "bf16x2": 2,
    "fp8e4m3x4": 4,
    "bf8e5m2x4": 4,
}
_LOGICAL_STORAGE_DTYPES = {
    "f16": "f16x2",
    "bf16": "bf16x2",
    "f32": "f32",
    "fp8e4m3": "fp8e4m3x4",
    "bf8e5m2": "bf8e5m2x4",
}


class _IntegerBuilder:
    """Evaluate a ``LayoutMap`` with ordinary integers instead of SSA values."""

    @staticmethod
    def const_i32(value: int) -> int:
        return value

    @staticmethod
    def add(lhs: int, rhs: int) -> int:
        return lhs + rhs

    @staticmethod
    def mul(lhs: int, rhs: int) -> int:
        return lhs * rhs

    @staticmethod
    def mod(lhs: int, rhs: int) -> int:
        return lhs % rhs

    @staticmethod
    def div(lhs: int, rhs: int) -> int:
        return lhs // rhs


def evaluate_layout(layout: LayoutMap) -> list[dict[str, Any]]:
    """Evaluate every finite lane/slot coordinate in ``layout``.

    Ordering is lane-major and then slot-major. Keeping the evaluated table in
    the manifest lets the rocGDB extension remain independent of the rocKE
    Python package while ``LayoutMap`` stays the coordinate source of truth.
    """
    builder = _IntegerBuilder()
    coordinates = []
    for lane in range(layout.wave_size):
        for slot in range(layout.frag_len):
            coord = layout.coord(builder, lane, slot)
            if len(coord) != 2 or not all(isinstance(index, int) for index in coord):
                raise ValueError(
                    f"layout {layout.role!r} returned non-integer coordinate {coord!r}"
                )
            coordinates.append(
                {"lane": lane, "slot": slot, "index": [coord[0], coord[1]]}
            )
    return coordinates


def logical_value_manifest(
    *,
    name: str,
    dtype: str,
    shape: Sequence[int],
    layout: LayoutMap,
    layout_name: str,
    storage_dtype: str,
    locations: Sequence[str],
) -> dict[str, Any]:
    """Build one debugger-independent logical-value manifest entry.

    ``locations`` are ordered physical expressions, normally VGPRs. Each word
    contributes the number of packed elements described by ``storage_dtype``.
    The debugger validates that their combined width equals ``frag_len``.
    """
    normalized_shape = [int(extent) for extent in shape]
    if not name:
        raise ValueError("logical value name must not be empty")
    if len(normalized_shape) != 2 or any(extent <= 0 for extent in normalized_shape):
        raise ValueError(
            f"logical tile shape must have two positive extents: {shape!r}"
        )
    if not layout_name:
        raise ValueError("layout name must not be empty")
    if not locations or any(not expression for expression in locations):
        raise ValueError("at least one non-empty physical location is required")
    expected_storage_dtype = _LOGICAL_STORAGE_DTYPES.get(dtype)
    if expected_storage_dtype is None:
        raise ValueError(f"unsupported logical dtype {dtype!r}")
    if storage_dtype != expected_storage_dtype:
        raise ValueError(
            f"logical dtype {dtype!r} requires storage dtype "
            f"{expected_storage_dtype!r}, not {storage_dtype!r}"
        )
    provided_elements = len(locations) * _STORAGE_WIDTHS[storage_dtype]
    if provided_elements != layout.frag_len:
        raise ValueError(
            f"{len(locations)} {storage_dtype} locations provide "
            f"{provided_elements} elements, but layout requires {layout.frag_len}"
        )

    return {
        "name": name,
        "dtype": dtype,
        "shape": normalized_shape,
        "storage_dtype": storage_dtype,
        "locations": list(locations),
        "layout": {
            "name": layout_name,
            "role": "acc" if layout.role == "c" else layout.role,
            "wave_size": layout.wave_size,
            "fragment_length": layout.frag_len,
            "coordinates": evaluate_layout(layout),
        },
    }


def debug_manifest(*values: dict[str, Any]) -> dict[str, Any]:
    """Wrap logical-value entries in the portable manifest schema."""
    names = [value.get("name") for value in values]
    if len(set(names)) != len(names):
        raise ValueError(f"logical value names must be unique: {names!r}")
    return {"schema": DEBUG_MANIFEST_SCHEMA, "values": list(values)}
