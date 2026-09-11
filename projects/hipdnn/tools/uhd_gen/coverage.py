# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-field evidence required before learning a device-normalized expression."""
from __future__ import annotations

import json

from .features import signature_references


def device_field_coverage(df) -> dict:
    """Board identity is not variation: record the actual values of every field."""
    fields = {}
    for column in sorted(name for name in df.columns if name.startswith("device.")):
        series = df[column]
        values = [value.item() if hasattr(value, "item") else value
                  for value in series.dropna().drop_duplicates().tolist()]
        values.sort(key=lambda value: json.dumps(value, sort_keys=True))
        missing = int(series.isna().sum())
        if "device" in df.columns:
            per_device = df.groupby("device", dropna=False)[column].nunique(dropna=False)
            if (per_device > 1).any():
                raise ValueError(f"device field {column!r} changes for the same device identity")
        fields[column] = {"values": values, "distinct_values": len(values), "missing_rows": missing,
                          "varies": len(values) > 1 and missing == 0}
    return {"device_count": int(df["device"].nunique()) if "device" in df.columns else None,
            "fields": fields}


def unsafe_device_fields(expression: dict, coverage: dict) -> list[str]:
    return sorted(reference[1:] for reference in signature_references([expression])
                  if reference.startswith("$device.")
                  and not coverage["fields"].get(reference[1:], {}).get("varies", False))


def enforce_device_coverage(signature: list, coverage: dict) -> None:
    """Raw constants remain legal; every computed device dependency must vary."""
    for index, expression in enumerate(signature):
        if not isinstance(expression, dict):
            continue
        unsafe = unsafe_device_fields(expression, coverage)
        if unsafe:
            raise ValueError(
                f"feature {index} uses device fields without observed variation: {', '.join(unsafe)}; "
                "collect distinct values for each field or use its raw reference instead"
            )


def propose_features(df, kernel_fields: set[str], dim_tile_pairs: list[tuple[str, str]]) -> tuple[list, list]:
    """Published problem/device values and KMD fields, plus author-declared geometry."""
    coverage = device_field_coverage(df)
    signature = ["$" + name for name in df.columns
                 if name.startswith("device.") or name in kernel_fields]
    # Collection passes only published feature columns here, never envelope metadata.
    signature.extend("$" + name for name in df.columns
                     if not name.startswith(("kernel.", "device.")))
    omitted = []
    for dimension, tile in dim_tile_pairs:
        if dimension not in df.columns or tile not in kernel_fields or tile not in df.columns:
            raise ValueError(f"dim-to-tile pair {dimension}={tile} must name a published dimension and KMD field")
        grid = {"ceil_div": ["$" + dimension, "$" + tile]}
        signature.extend([grid, {"%": ["$" + dimension, "$" + tile]}])
        if "device.cu_count" in df.columns:
            normalized = {"/": [grid, "$device.cu_count"]}
            unsafe = unsafe_device_fields(normalized, coverage)
            if unsafe:
                omitted.append({"expression": normalized, "constant_device_fields": unsafe})
            else:
                signature.append(normalized)
    # The same pair may be supplied twice; preserve authored feature order, not duplicates.
    unique = {}
    for entry in signature:
        unique.setdefault(json.dumps(entry, sort_keys=True), entry)
    return list(unique.values()), omitted
