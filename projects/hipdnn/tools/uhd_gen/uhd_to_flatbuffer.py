#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert UHD configuration to FlatBuffer UHD format.

The output format is hipdnn_flatbuffers_sdk.data_objects.UHD, read at runtime by
`UhdLoader::loadFromBuffer` (RFC 0019 §9.2).

Written through the flatc-generated object API rather than by hand. The previous
version hand-rolled the vtable -- `StartObject(11)` against a 13-field table, with
every slot from `features_signature` on one index low and `derived`/`native_symbol`
unwritten -- so every buffer it produced failed the runtime verifier. A generated
builder cannot drift: field identity is a name, and a schema addition renumbers
nothing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import flatbuffers

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.UHD import UHDT
from hipdnn_flatbuffers_sdk.data_objects.UhdAdapter import UhdAdapter
from hipdnn_flatbuffers_sdk.data_objects.UhdDerivedEntry import UhdDerivedEntryT
from hipdnn_flatbuffers_sdk.data_objects.UhdScoreMetadata import UhdScoreMetadataT

logger = logging.getLogger(__name__)

# File identifier for UHD FlatBuffers (from uhd.fbs).
UHD_FILE_IDENTIFIER = b"HUHD"

# Re-exported from the generated enum so the two cannot disagree. Kept as module
# constants because callers and tests name them.
ADAPTER_STATIC_ORDER = UhdAdapter.STATIC_ORDER
ADAPTER_TREE_DATA = UhdAdapter.TREE_DATA
ADAPTER_TABLE = UhdAdapter.TABLE
ADAPTER_ONNX = UhdAdapter.ONNX
ADAPTER_CUSTOM_LIBRARY = UhdAdapter.CUSTOM_LIBRARY
ADAPTER_NATIVE = UhdAdapter.NATIVE

ADAPTER_BY_NAME = {
    "static_order": ADAPTER_STATIC_ORDER,
    "tree_data": ADAPTER_TREE_DATA,
    "table": ADAPTER_TABLE,
    "onnx": ADAPTER_ONNX,
    "custom_library": ADAPTER_CUSTOM_LIBRARY,
    "native": ADAPTER_NATIVE,
}

# Adapters that score from a loaded artifact, so a path is mandatory. `native`
# resolves an in-process symbol and `UhdLoader` rejects it outright if it also
# carries an artifact path, so it is not in this set.
_ARTIFACT_ADAPTERS = ("tree_data", "table", "onnx", "custom_library")


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
    native_symbol: str | None = None,
    derived: list[tuple[str, str]] | None = None,
) -> None:
    """Convert UHD configuration to a FlatBuffer UHD written to @p output_path.

    Args:
        uhd_id: UUID/GUID identifier.
        name: Human-readable display name.
        adapter: One of ADAPTER_BY_NAME's keys.
        features_signature: Ordered list of feature expressions (JsonLogic format).
        features_hash: SHA-256 hash of features_signature.
        objective: "max" or "min". UhdLoader rejects anything else.
        score_units: Score units (e.g. "tflops", "ms").
        score_calibrated: True if cross-engine comparable.
        score_transform: Transform applied to model output ("identity", "log1p", ...).
        output_path: Destination for the .uhd.fb file.
        model_artifact_path: Path to the model artifact, relative to this descriptor.
        model_hash: Checksum of the model artifact.
        static_order_fields: Sort fields for the static_order adapter.
        custom_library_symbol: Scorer symbol inside a custom_library .so.
        native_symbol: Scorer symbol registered in-process, for the native adapter.
        derived: Ordered (name, JsonLogic expression) pairs forming $derived.*
            (RFC 0019 §6.4). Evaluated in order; later entries may reference earlier.

    Raises:
        ValueError: If the adapter is unknown or a required field is missing.
    """
    if adapter not in ADAPTER_BY_NAME:
        raise ValueError(f"Unknown adapter type: {adapter}")

    if adapter in _ARTIFACT_ADAPTERS and not model_artifact_path:
        raise ValueError(f"Adapter '{adapter}' requires model_artifact_path")

    if adapter == "native" and not native_symbol:
        raise ValueError("Adapter 'native' requires native_symbol")

    if adapter == "native" and model_artifact_path:
        raise ValueError(
            "Adapter 'native' must not carry model_artifact_path; use "
            "custom_library for a scorer shipped as a .so"
        )

    buffer = build_uhd(
        uhd_id=uhd_id,
        name=name,
        adapter=ADAPTER_BY_NAME[adapter],
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
        native_symbol=native_symbol,
        derived=derived,
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
    model_artifact_path: str | None = None,
    model_hash: str | None = None,
    static_order_fields: list[str] | None = None,
    custom_library_symbol: str | None = None,
    native_symbol: str | None = None,
    derived: list[tuple[str, str]] | None = None,
) -> bytes:
    """Build the FlatBuffer UHD for a configuration.

    @p adapter is a UhdAdapter enum value, not a name; convert_uhd() maps names.

    Returns:
        Finished FlatBuffer bytes, carrying the "HUHD" file identifier.
    """
    uhd = UHDT()
    uhd.id = uhd_id
    uhd.name = name
    uhd.adapter = adapter
    uhd.featuresSignature = list(features_signature)
    uhd.featuresHash = features_hash
    uhd.objective = objective

    score = UhdScoreMetadataT()
    score.units = score_units
    score.calibrated = score_calibrated
    score.transform = score_transform
    uhd.score = score

    if derived:
        entries = []
        for entry_name, expression in derived:
            entry = UhdDerivedEntryT()
            entry.name = entry_name
            entry.expression = expression
            entries.append(entry)
        uhd.derived = entries

    # Left unset rather than set empty: an absent FlatBuffer field reads back as
    # null, which is how the loader distinguishes "not supplied" from "supplied
    # empty" for every one of these.
    if model_artifact_path:
        uhd.modelArtifactPath = model_artifact_path
    if model_hash:
        uhd.modelHash = model_hash
    if static_order_fields:
        uhd.staticOrderFields = list(static_order_fields)
    if custom_library_symbol:
        uhd.customLibrarySymbol = custom_library_symbol
    if native_symbol:
        uhd.nativeSymbol = native_symbol

    builder = flatbuffers.Builder(1024)
    builder.Finish(uhd.Pack(builder), file_identifier=UHD_FILE_IDENTIFIER)
    return bytes(builder.Output())
