# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reconstruct logical tiles from stopped-wave register values and layouts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .register_value_decoding import DTYPES, decode_word_value

LOGICAL_VALUE_SCHEMA = "rocke-logical-value/v1"
LOGICAL_SNAPSHOT_SCHEMA = "rocke-logical-snapshot/v1"
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


def _expanded_indices(
    index: list[int], packing: dict[str, Any]
) -> list[tuple[int, int]]:
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
    if dtype not in _LOGICAL_STORAGE_DTYPES:
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
    replication_factor = layout.get("replication_factor")
    if not isinstance(wave_size, int) or wave_size <= 0:
        raise ValueError(f"invalid layout wave size {wave_size!r}")
    if not isinstance(fragment_length, int) or fragment_length <= 0:
        raise ValueError(f"invalid layout fragment length {fragment_length!r}")
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
    packed_width = len(decode_word_value(0, storage_dtype))
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
                0 <= logical_index[0] < shape[0]
                and 0 <= logical_index[1] < shape[1]
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
            "layout source multiplicity does not match replication factor "
            f"{replication_factor}; first mismatch {sample!r}"
        )
    return logical, binding, wave_size, fragment_length, packed_width


def unavailable_logical_value(
    value: dict[str, Any], status: str, detail: str
) -> dict[str, Any]:
    """Build a presentation-neutral unavailable logical-value record."""
    logical = value.get("logical") or {}
    binding = value.get("binding") or {}
    return {
        "schema": LOGICAL_VALUE_SCHEMA,
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
    message = str(error).lower()
    if "dtype" in message:
        return "unsupported_dtype"
    if "layout" in message or "coordinate" in message or "shape" in message:
        return "unsupported_layout"
    return "stale_manifest"


def reconstruct_logical_value(
    value: dict[str, Any],
    raw_locations: Sequence[Sequence[int]],
    exec_mask: int | None = None,
    float8_format: str = "ocp",
) -> dict[str, Any]:
    """Reconstruct a presentation-neutral logical tile from raw words."""
    logical, binding, wave_size, fragment_length, packed_width = _validated_value_spec(
        value
    )
    if logical["layout"]["packing"]["kind"] != "scalar":
        raise ValueError("packed logical fragment slots are not yet decodable")
    locations = binding["locations"]
    if len(raw_locations) != len(locations):
        raise ValueError(
            f"received {len(raw_locations)} physical locations; "
            f"expected {len(locations)}"
        )
    if any(len(words) != wave_size for words in raw_locations):
        lengths = [len(words) for words in raw_locations]
        raise ValueError(
            f"physical location lane counts {lengths!r} do not match "
            f"wave size {wave_size}"
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
            decoded = decode_word_value(
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
                    "class": scalar["class"],
                    "sign": scalar["sign"],
                    "value": scalar["value"],
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
                    "class": representative["class"] if agrees else None,
                    "sign": representative["sign"] if agrees else None,
                    "value": representative["value"] if agrees else None,
                }
            )
        tile.append(cells)
    return {
        "schema": LOGICAL_VALUE_SCHEMA,
        "name": logical["name"],
        "dtype": logical["dtype"],
        "storage_dtype": storage_dtype,
        "float8_format": float8_format if "8" in storage_dtype else None,
        "shape": shape,
        "status": "replica_mismatch" if has_replica_mismatch else "available",
        "detail": "observable replicas disagree" if has_replica_mismatch else None,
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


def logical_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every captured wave without retaining presentation strings."""
    specs = {
        value["logical"]["name"]: value
        for value in snapshot["values"]
    }
    waves = []
    for wave in snapshot["waves"]:
        records = []
        for captured in wave["values"]:
            spec = specs[captured["name"]]
            if captured["status"] != "available":
                record = unavailable_logical_value(
                    spec, captured["status"], captured.get("detail") or ""
                )
            else:
                raw_locations = [
                    location["raw_words"] for location in captured["locations"]
                ]
                exec_mask = int(wave["exec"], 16) if wave.get("exec") else None
                record = reconstruct_logical_value(
                    spec,
                    raw_locations,
                    exec_mask=exec_mask,
                    float8_format=snapshot["capture"].get("float8_format", "ocp"),
                )
            records.append(record)
        waves.append(
            {
                "thread_id": wave["thread_id"],
                "dispatch_id": wave.get("dispatch_id"),
                "workgroup": wave.get("workgroup"),
                "wave_position": wave.get("wave_position"),
                "pc": wave.get("pc"),
                "kernel_pc_offset": wave.get("kernel_pc_offset"),
                "exec": wave.get("exec"),
                "status": wave["status"],
                "values": records,
            }
        )
    return {
        "schema": LOGICAL_SNAPSHOT_SCHEMA,
        "capture": snapshot["capture"],
        "target": snapshot["target"],
        "waves": waves,
    }
