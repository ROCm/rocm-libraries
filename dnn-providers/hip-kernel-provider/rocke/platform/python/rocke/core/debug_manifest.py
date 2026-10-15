# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bind portable logical values to headless rocGDB register locations.

Logical dtype, shape, and layout live in :mod:`rocke.core.logical_value` and can
be shared by other printing backends. This module adds the debugger-specific
physical register binding and wraps bound values in a manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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


def register_value_binding(
    *, storage_dtype: str, locations: Sequence[str], fragment_length: int
) -> dict[str, Any]:
    """Describe the rocGDB register expressions carrying one lane fragment."""
    if storage_dtype not in _STORAGE_WIDTHS:
        raise ValueError(f"unsupported storage dtype {storage_dtype!r}")
    if not locations or any(not expression for expression in locations):
        raise ValueError("at least one non-empty physical location is required")
    provided_elements = len(locations) * _STORAGE_WIDTHS[storage_dtype]
    if provided_elements != fragment_length:
        raise ValueError(
            f"{len(locations)} {storage_dtype} locations provide "
            f"{provided_elements} elements, but layout requires {fragment_length}"
        )
    return {
        "kind": "amdgpu_registers",
        "storage_dtype": storage_dtype,
        "locations": list(locations),
    }


def bind_logical_value(
    description: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine reusable logical semantics with one observation binding."""
    dtype = description.get("dtype")
    storage_dtype = binding.get("storage_dtype")
    expected_storage_dtype = _LOGICAL_STORAGE_DTYPES.get(dtype)
    if expected_storage_dtype is None:
        raise ValueError(f"unsupported logical dtype {dtype!r}")
    if storage_dtype != expected_storage_dtype:
        raise ValueError(
            f"logical dtype {dtype!r} requires storage dtype "
            f"{expected_storage_dtype!r}, not {storage_dtype!r}"
        )
    return {"logical": dict(description), "binding": dict(binding)}


def debug_manifest(*values: dict[str, Any]) -> dict[str, Any]:
    """Wrap bound logical-value entries in the rocGDB manifest schema."""
    names = [value.get("logical", {}).get("name") for value in values]
    if len(set(names)) != len(names):
        raise ValueError(f"logical value names must be unique: {names!r}")
    return {"schema": DEBUG_MANIFEST_SCHEMA, "values": list(values)}
