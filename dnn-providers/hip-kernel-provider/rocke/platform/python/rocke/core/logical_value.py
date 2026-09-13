# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Backend-neutral descriptions of logical values and distributed layouts.

The architecture owns how physical fragment slots map to logical tile elements.
Printing and debugging backends consume the evaluated description but retain
their own transport, availability, and presentation policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .arch import LayoutMap

LOGICAL_DTYPES = (
    "f16",
    "bf16",
    "f32",
    "fp8e4m3",
    "bf8e5m2",
    "i32",
    "iu8",
    "iu4",
)


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
    """Evaluate every finite physical-to-logical relation in ``layout``.

    Ordering is lane-major and then slot-major. Multiple physical entries may
    intentionally name the same logical index when a layout is replicated.
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


def _expanded_indices(layout: LayoutMap, index: list[int]) -> list[tuple[int, int]]:
    if layout.packing is None:
        return [tuple(index)]
    expanded = []
    for offset in range(layout.packing.elements_per_slot):
        logical = list(index)
        logical[layout.packing.axis] += offset
        expanded.append(tuple(logical))
    return expanded


def validate_layout_relation(
    layout: LayoutMap, shape: Sequence[int]
) -> list[dict[str, Any]]:
    """Validate physical completeness, bounds, coverage, and multiplicity."""
    normalized_shape = tuple(int(extent) for extent in shape)
    if len(normalized_shape) != 2 or any(extent <= 0 for extent in normalized_shape):
        raise ValueError(
            f"logical tile shape must have two positive extents: {shape!r}"
        )
    coordinates = evaluate_layout(layout)
    physical = {(entry["lane"], entry["slot"]) for entry in coordinates}
    expected_physical = {
        (lane, slot)
        for lane in range(layout.wave_size)
        for slot in range(layout.frag_len)
    }
    if physical != expected_physical or len(coordinates) != len(physical):
        raise ValueError("layout does not cover each physical lane/slot exactly once")

    source_counts: dict[tuple[int, int], int] = {}
    for entry in coordinates:
        for index in _expanded_indices(layout, entry["index"]):
            if not (
                0 <= index[0] < normalized_shape[0]
                and 0 <= index[1] < normalized_shape[1]
            ):
                raise ValueError(
                    f"layout coordinate {index!r} is outside shape "
                    f"{normalized_shape!r}"
                )
            source_counts[index] = source_counts.get(index, 0) + 1

    expected_logical = {
        (axis0, axis1)
        for axis0 in range(normalized_shape[0])
        for axis1 in range(normalized_shape[1])
    }
    if set(source_counts) != expected_logical:
        missing = sorted(expected_logical - set(source_counts))
        raise ValueError(
            f"layout does not cover shape {normalized_shape!r}; "
            f"missing {missing[:4]!r}"
        )
    wrong_multiplicity = {
        index: count
        for index, count in source_counts.items()
        if count != layout.replication_factor
    }
    if wrong_multiplicity:
        sample = next(iter(sorted(wrong_multiplicity.items())))
        raise ValueError(
            f"layout source multiplicity does not match replication factor "
            f"{layout.replication_factor}; first mismatch {sample!r}"
        )
    return coordinates


def logical_value_description(
    *,
    name: str,
    dtype: str,
    shape: Sequence[int],
    layout: LayoutMap,
    layout_name: str,
) -> dict[str, Any]:
    """Build a transport-independent logical value description."""
    normalized_shape = [int(extent) for extent in shape]
    if not name:
        raise ValueError("logical value name must not be empty")
    if dtype not in LOGICAL_DTYPES:
        raise ValueError(f"unsupported logical dtype {dtype!r}")
    if len(normalized_shape) != 2 or any(extent <= 0 for extent in normalized_shape):
        raise ValueError(
            f"logical tile shape must have two positive extents: {shape!r}"
        )
    if not layout_name:
        raise ValueError("layout name must not be empty")

    packing = {"kind": "scalar", "elements_per_slot": 1}
    if layout.packing is not None:
        packing = {
            "kind": "contiguous",
            "axis": layout.packing.axis,
            "elements_per_slot": layout.packing.elements_per_slot,
        }

    coordinates = validate_layout_relation(layout, normalized_shape)
    return {
        "name": name,
        "dtype": dtype,
        "shape": normalized_shape,
        "layout": {
            "name": layout_name,
            "role": "acc" if layout.role == "c" else layout.role,
            "wave_size": layout.wave_size,
            "fragment_length": layout.frag_len,
            "replication_factor": layout.replication_factor,
            "packing": packing,
            "coordinates": coordinates,
        },
    }
