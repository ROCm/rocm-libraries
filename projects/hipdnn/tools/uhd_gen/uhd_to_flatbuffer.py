#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert UHD configuration to FlatBuffer UHD format.

The output format matches hipdnn_flatbuffers_sdk.data_objects.UHD,
which is read by UHD loader at runtime (RFC 0019 §9.2).
"""
from __future__ import annotations

import logging
from pathlib import Path

import flatbuffers

logger = logging.getLogger(__name__)

# File identifier for UHD FlatBuffers (from uhd.fbs)
UHD_FILE_IDENTIFIER = b"HUHD"

# UhdAdapter enum values (from uhd.fbs)
ADAPTER_STATIC_ORDER = 0
ADAPTER_TREE_DATA = 1
ADAPTER_TABLE = 2
ADAPTER_ONNX = 3
ADAPTER_CUSTOM_LIBRARY = 4


def convert_uhd(
    uhd_id: str,
    name: str,
    adapter: str,
    features_signature: list[str],
    features_hash: str,
    objective: str,
    score_units: str,
    score_calibrated: bool,
    score_transform: str,
    output_path: str | Path,
    model_artifact_path: str | None = None,
    model_hash: str | None = None,
    static_order_fields: list[str] | None = None,
    custom_library_symbol: str | None = None,
) -> None:
    """Convert UHD configuration to FlatBuffer UHD.

    Args:
        uhd_id: UUID/GUID identifier.
        name: Human-readable display name.
        adapter: Adapter type ("static_order", "tree_data", "table", "onnx", "custom_library").
        features_signature: Ordered list of feature expressions (JsonLogic format).
        features_hash: SHA-256 hash of features_signature.
        objective: "max" or "min".
        score_units: Score units (e.g., "tflops", "ms").
        score_calibrated: True if cross-engine comparable.
        score_transform: Transform applied to model output ("identity", "log", "log1p").
        output_path: Output path for .uhd FlatBuffer file.
        model_artifact_path: Relative path to model artifact (for tree_data/onnx/custom_library).
        model_hash: Checksum of model artifact.
        static_order_fields: Field names for static_order adapter (e.g., ["priority", "id"]).
        custom_library_symbol: Symbol name for custom_library adapter.

    Raises:
        ValueError: If adapter type is unknown or required fields are missing.
    """
    # Map string adapter to enum value
    adapter_enum_map = {
        "static_order": ADAPTER_STATIC_ORDER,
        "tree_data": ADAPTER_TREE_DATA,
        "table": ADAPTER_TABLE,
        "onnx": ADAPTER_ONNX,
        "custom_library": ADAPTER_CUSTOM_LIBRARY,
    }
    if adapter not in adapter_enum_map:
        raise ValueError(f"Unknown adapter type: {adapter}")

    adapter_value = adapter_enum_map[adapter]

    # Validate required fields per adapter type
    if adapter in ("tree_data", "onnx", "custom_library") and not model_artifact_path:
        raise ValueError(f"Adapter '{adapter}' requires model_artifact_path")

    buffer = build_uhd(
        uhd_id=uhd_id,
        name=name,
        adapter=adapter_value,
        features_signature=features_signature,
        features_hash=features_hash,
        objective=objective,
        score_units=score_units,
        score_calibrated=score_calibrated,
        score_transform=score_transform,
        model_artifact_path=model_artifact_path,
        model_hash=model_hash,
        static_order_fields=static_order_fields,
        custom_library_symbol=custom_library_symbol,
    )

    with open(output_path, "wb") as f:
        f.write(buffer)

    logger.info(
        "Converted UHD to FlatBuffer: %s (%d bytes, adapter=%s)",
        output_path,
        len(buffer),
        adapter,
    )


def build_uhd(
    uhd_id: str,
    name: str,
    adapter: int,
    features_signature: list[str],
    features_hash: str,
    objective: str,
    score_units: str,
    score_calibrated: bool,
    score_transform: str,
    model_artifact_path: str | None,
    model_hash: str | None,
    static_order_fields: list[str] | None,
    custom_library_symbol: str | None,
) -> bytes:
    """Build FlatBuffer UHD from configuration.

    Returns:
        FlatBuffer bytes for UHD.
    """
    builder = flatbuffers.Builder(1024)

    # Build string offsets
    id_offset = builder.CreateString(uhd_id)
    name_offset = builder.CreateString(name)
    hash_offset = builder.CreateString(features_hash)
    objective_offset = builder.CreateString(objective)

    # Build features_signature vector
    sig_offsets = [builder.CreateString(feature) for feature in features_signature]
    builder.StartVector(4, len(sig_offsets), 4)
    for offset in reversed(sig_offsets):
        builder.PrependUOffsetTRelative(offset)
    features_signature_vec = builder.EndVector()

    # Build UhdScoreMetadata table
    score_units_offset = builder.CreateString(score_units)
    score_transform_offset = builder.CreateString(score_transform)

    _start_uhd_score_metadata(builder)
    _add_uhd_score_metadata_units(builder, score_units_offset)
    _add_uhd_score_metadata_calibrated(builder, score_calibrated)
    _add_uhd_score_metadata_transform(builder, score_transform_offset)
    score_metadata_offset = _end_uhd_score_metadata(builder)

    # Build optional fields
    model_artifact_offset = None
    if model_artifact_path:
        model_artifact_offset = builder.CreateString(model_artifact_path)

    model_hash_offset = None
    if model_hash:
        model_hash_offset = builder.CreateString(model_hash)

    static_order_vec = None
    if static_order_fields:
        field_offsets = [builder.CreateString(field) for field in static_order_fields]
        builder.StartVector(4, len(field_offsets), 4)
        for offset in reversed(field_offsets):
            builder.PrependUOffsetTRelative(offset)
        static_order_vec = builder.EndVector()

    custom_library_symbol_offset = None
    if custom_library_symbol:
        custom_library_symbol_offset = builder.CreateString(custom_library_symbol)

    # Build UHD table
    _start_uhd(builder)
    _add_uhd_id(builder, id_offset)
    _add_uhd_name(builder, name_offset)
    _add_uhd_adapter(builder, adapter)
    _add_uhd_features_signature(builder, features_signature_vec)
    _add_uhd_features_hash(builder, hash_offset)
    _add_uhd_objective(builder, objective_offset)
    _add_uhd_score(builder, score_metadata_offset)

    if model_artifact_offset is not None:
        _add_uhd_model_artifact_path(builder, model_artifact_offset)
    if model_hash_offset is not None:
        _add_uhd_model_hash(builder, model_hash_offset)
    if static_order_vec is not None:
        _add_uhd_static_order_fields(builder, static_order_vec)
    if custom_library_symbol_offset is not None:
        _add_uhd_custom_library_symbol(builder, custom_library_symbol_offset)

    uhd_offset = _end_uhd(builder)

    builder.Finish(uhd_offset, file_identifier=UHD_FILE_IDENTIFIER)
    return bytes(builder.Output())


# UhdScoreMetadata table helpers (inline, no generated bindings)
def _start_uhd_score_metadata(builder: flatbuffers.Builder) -> None:
    builder.StartObject(3)  # 3 fields: units, calibrated, transform


def _add_uhd_score_metadata_units(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(0, s, 0)


def _add_uhd_score_metadata_calibrated(builder: flatbuffers.Builder, v: bool) -> None:
    builder.PrependBoolSlot(1, v, False)


def _add_uhd_score_metadata_transform(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(2, s, 0)


def _end_uhd_score_metadata(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()


# UHD table helpers
#
# Slot numbers are positions in uhd.fbs's `table UHD`, and they are load-bearing: a writer
# whose numbering is one behind the schema produces a buffer whose every field is read as
# its predecessor. That is not a parse error -- the reader sees a well-formed table with the
# wrong contents -- so it fails as a rejected descriptor rather than as anything that names
# the cause.
#
# This drifted exactly that way once already, when `derived` was inserted at slot 3 and this
# writer was not updated, silently shifting features_signature onward and making every UHD
# it emitted unloadable. TestUhdGenArtifact now loads a generated artifact through the C++
# loader, which is what would catch it next time; keep these in step with the schema, and
# add new fields only at the end.
def _start_uhd(builder: flatbuffers.Builder) -> None:
    builder.StartObject(13)  # 13 fields from uhd.fbs


def _add_uhd_id(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(0, s, 0)


def _add_uhd_name(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(1, s, 0)


def _add_uhd_adapter(builder: flatbuffers.Builder, adapter: int) -> None:
    builder.PrependUint8Slot(2, adapter, ADAPTER_STATIC_ORDER)


def _add_uhd_derived(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(3, vec, 0)


def _add_uhd_features_signature(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(4, vec, 0)


def _add_uhd_features_hash(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(5, s, 0)


def _add_uhd_objective(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(6, s, 0)


def _add_uhd_score(builder: flatbuffers.Builder, table: int) -> None:
    builder.PrependUOffsetTRelativeSlot(7, table, 0)


def _add_uhd_model_artifact_path(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(8, s, 0)


def _add_uhd_model_hash(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(9, s, 0)


def _add_uhd_static_order_fields(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(10, vec, 0)


def _add_uhd_custom_library_symbol(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(11, s, 0)


def _add_uhd_native_symbol(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(12, s, 0)


def _end_uhd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()
