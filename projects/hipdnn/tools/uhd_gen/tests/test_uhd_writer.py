#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the UHD FlatBuffer writer.

Every structural case reads the buffer back through the generated accessors and
checks fields **by name**. The previous version asserted only `buffer[4:8] ==
b"HUHD"` and a non-zero length, which is why a writer whose vtable was two slots
short and misaligned from `derived` onward shipped green for as long as it did: the
file identifier sits before the root table and survives any vtable error.
"""
import tempfile
from pathlib import Path

import pytest

import uhd_gen  # noqa: F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.UHD import UHD
from hipdnn_flatbuffers_sdk.data_objects.UhdAdapter import UhdAdapter
from uhd_gen.uhd_to_flatbuffer import (
    ADAPTER_BY_NAME,
    ADAPTER_NATIVE,
    ADAPTER_STATIC_ORDER,
    ADAPTER_TREE_DATA,
    UHD_FILE_IDENTIFIER,
    build_uhd,
    convert_uhd,
)

SIGNATURE = ["$q.seqlen_q", "$kernel.block_size", "$device.cu_count"]


def _read(buffer: bytes) -> UHD:
    """Parse a finished buffer, asserting the identifier the loader checks first."""
    assert UHD.UHDBufferHasIdentifier(buffer, 0)
    return UHD.GetRootAs(buffer, 0)


def _signature_of(uhd: UHD) -> list[str]:
    return [
        uhd.FeaturesSignature(i).decode() for i in range(uhd.FeaturesSignatureLength())
    ]


def test_build_uhd_tree_data():
    """A tree_data UHD round-trips every field it was given."""
    buffer = build_uhd(
        uhd_id="test-uhd-id",
        name="Test UHD",
        adapter=ADAPTER_TREE_DATA,
        features_signature=SIGNATURE,
        features_hash="sha256:abc123",
        objective="max",
        score_units="tflops",
        score_calibrated=True,
        score_transform="log1p",
        model_artifact_path="model.bin",
        model_hash="sha256:def456",
    )

    uhd = _read(buffer)
    assert uhd.Id().decode() == "test-uhd-id"
    assert uhd.Name().decode() == "Test UHD"
    assert uhd.Adapter() == UhdAdapter.TREE_DATA
    assert _signature_of(uhd) == SIGNATURE
    assert uhd.FeaturesHash().decode() == "sha256:abc123"
    assert uhd.Objective().decode() == "max"
    assert uhd.ModelArtifactPath().decode() == "model.bin"
    assert uhd.ModelHash().decode() == "sha256:def456"

    score = uhd.Score()
    assert score is not None
    assert score.Units().decode() == "tflops"
    assert score.Calibrated() is True
    assert score.Transform().decode() == "log1p"


def test_features_signature_lands_in_features_signature():
    """The regression that motivated generating the bindings.

    `derived` sits between `adapter` and `features_signature` in uhd.fbs. A writer
    that predates it puts the signature one slot low, so the runtime reads a vector
    of strings as a vector of tables and the verifier rejects the whole buffer.
    Asserting the two fields separately is what distinguishes the two layouts.
    """
    buffer = build_uhd(
        uhd_id="id",
        name="n",
        adapter=ADAPTER_TREE_DATA,
        features_signature=SIGNATURE,
        features_hash="sha256:h",
        objective="max",
        score_units="tflops",
        score_calibrated=False,
        score_transform="identity",
        model_artifact_path="model.bin",
    )

    uhd = _read(buffer)
    assert _signature_of(uhd) == SIGNATURE
    assert uhd.DerivedLength() == 0, "signature must not land in `derived`"
    assert uhd.FeaturesHash().decode() == "sha256:h"


def test_derived_entries_round_trip():
    """`derived` had no writer at all before the bindings were generated."""
    derived = [
        ("tiles_m", '{"ceil_div": ["$q.seqlen_q", "$kernel.block_size"]}'),
        ("aspect", '{"/": ["$q.seqlen_q", "$q.seqlen_kv"]}'),
    ]
    buffer = build_uhd(
        uhd_id="id",
        name="n",
        adapter=ADAPTER_TREE_DATA,
        features_signature=SIGNATURE,
        features_hash="sha256:h",
        objective="max",
        score_units="tflops",
        score_calibrated=False,
        score_transform="identity",
        model_artifact_path="model.bin",
        derived=derived,
    )

    uhd = _read(buffer)
    assert uhd.DerivedLength() == 2
    # Order is the evaluation order: a later expression may reference an earlier one.
    for index, (name, expression) in enumerate(derived):
        entry = uhd.Derived(index)
        assert entry.Name().decode() == name
        assert entry.Expression().decode() == expression
    assert _signature_of(uhd) == SIGNATURE


def test_native_symbol_round_trips():
    """`native_symbol` is the schema's last field and was likewise unwritable."""
    buffer = build_uhd(
        uhd_id="id",
        name="n",
        adapter=ADAPTER_NATIVE,
        features_signature=SIGNATURE,
        features_hash="sha256:h",
        objective="max",
        score_units="tflops",
        score_calibrated=False,
        score_transform="identity",
        native_symbol="engine.sdpa.score",
    )

    uhd = _read(buffer)
    assert uhd.Adapter() == UhdAdapter.NATIVE
    assert uhd.NativeSymbol().decode() == "engine.sdpa.score"
    assert uhd.ModelArtifactPath() is None


def test_build_uhd_static_order():
    """static_order carries sort fields and no artifact."""
    buffer = build_uhd(
        uhd_id="static-uhd",
        name="Static Order UHD",
        adapter=ADAPTER_STATIC_ORDER,
        features_signature=[],
        features_hash="",
        objective="max",
        score_units="",
        score_calibrated=False,
        score_transform="identity",
        static_order_fields=["priority", "id"],
    )

    uhd = _read(buffer)
    assert uhd.Adapter() == UhdAdapter.STATIC_ORDER
    assert uhd.StaticOrderFieldsLength() == 2
    assert uhd.StaticOrderFields(0).decode() == "priority"
    assert uhd.StaticOrderFields(1).decode() == "id"
    assert uhd.ModelArtifactPath() is None


def test_absent_optionals_read_back_as_null():
    """An omitted optional must be absent, not empty.

    `UhdLoader` branches on presence -- a `native` UHD carrying a
    `model_artifact_path` is rejected outright -- so writing an empty string where
    the caller supplied nothing changes runtime behaviour.
    """
    buffer = build_uhd(
        uhd_id="id",
        name="n",
        adapter=ADAPTER_STATIC_ORDER,
        features_signature=[],
        features_hash="",
        objective="max",
        score_units="",
        score_calibrated=False,
        score_transform="identity",
    )

    uhd = _read(buffer)
    assert uhd.ModelArtifactPath() is None
    assert uhd.ModelHash() is None
    assert uhd.CustomLibrarySymbol() is None
    assert uhd.NativeSymbol() is None
    assert uhd.StaticOrderFieldsLength() == 0
    assert uhd.DerivedLength() == 0


def test_convert_uhd_to_file():
    """convert_uhd writes bytes that parse back with the fields it was given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.uhd.fb"
        convert_uhd(
            uhd_id="file-uhd",
            name="File UHD",
            adapter="tree_data",
            features_signature=SIGNATURE,
            features_hash="sha256:filehash",
            objective="min",
            score_units="ms",
            score_calibrated=False,
            score_transform="identity",
            output_path=output_path,
            model_artifact_path="model.bin",
        )

        assert output_path.exists()
        uhd = _read(output_path.read_bytes())
        assert uhd.Id().decode() == "file-uhd"
        assert uhd.Objective().decode() == "min"
        assert uhd.Score().Units().decode() == "ms"
        assert _signature_of(uhd) == SIGNATURE


def test_convert_uhd_rejects_unknown_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown adapter type"):
            convert_uhd(
                uhd_id="id",
                name="n",
                adapter="not_an_adapter",
                features_signature=[],
                features_hash="",
                objective="max",
                score_units="",
                score_calibrated=False,
                score_transform="identity",
                output_path=Path(tmpdir) / "x.uhd.fb",
            )


def test_convert_uhd_requires_model_path_for_tree_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="requires model_artifact_path"):
            convert_uhd(
                uhd_id="id",
                name="n",
                adapter="tree_data",
                features_signature=SIGNATURE,
                features_hash="sha256:h",
                objective="max",
                score_units="tflops",
                score_calibrated=False,
                score_transform="log1p",
                output_path=Path(tmpdir) / "x.uhd.fb",
            )


def test_convert_uhd_requires_symbol_for_native():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="requires native_symbol"):
            convert_uhd(
                uhd_id="id",
                name="n",
                adapter="native",
                features_signature=SIGNATURE,
                features_hash="sha256:h",
                objective="max",
                score_units="tflops",
                score_calibrated=False,
                score_transform="identity",
                output_path=Path(tmpdir) / "x.uhd.fb",
            )


def test_convert_uhd_rejects_artifact_on_native():
    """Mirrors UhdLoader, which errors on exactly this combination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="must not carry model_artifact_path"):
            convert_uhd(
                uhd_id="id",
                name="n",
                adapter="native",
                features_signature=SIGNATURE,
                features_hash="sha256:h",
                objective="max",
                score_units="tflops",
                score_calibrated=False,
                score_transform="identity",
                output_path=Path(tmpdir) / "x.uhd.fb",
                native_symbol="engine.score",
                model_artifact_path="model.bin",
            )


@pytest.mark.parametrize(
    "adapter_name,artifact",
    [
        ("static_order", None),
        ("tree_data", "model.bin"),
        ("table", "table.bin"),
        ("onnx", "model.onnx"),
        ("custom_library", "libscorer.so"),
    ],
)
def test_every_adapter_serializes_with_its_own_enum_value(adapter_name, artifact):
    """The name->enum map must agree with the generated enum, for every member."""
    buffer = build_uhd(
        uhd_id=f"{adapter_name}-uhd",
        name=f"{adapter_name} UHD",
        adapter=ADAPTER_BY_NAME[adapter_name],
        features_signature=SIGNATURE,
        features_hash="sha256:h",
        objective="max",
        score_units="tflops",
        score_calibrated=False,
        score_transform="identity",
        model_artifact_path=artifact,
    )

    uhd = _read(buffer)
    assert uhd.Adapter() == getattr(UhdAdapter, adapter_name.upper())
    assert _signature_of(uhd) == SIGNATURE


def test_adapter_map_covers_the_generated_enum():
    """A schema gaining an adapter must not leave the writer unable to emit it."""
    declared = {
        name
        for name in vars(UhdAdapter)
        if name.isupper() and isinstance(getattr(UhdAdapter, name), int)
    }
    mapped = {name.upper() for name in ADAPTER_BY_NAME}
    assert mapped == declared


def test_file_identifier_is_still_the_one_the_loader_checks():
    """Kept as its own case so its weakness is explicit.

    This assertion passes for a buffer whose every field is misplaced; it is here
    only to pin the four bytes `UhdLoader` matches before parsing.
    """
    buffer = build_uhd(
        uhd_id="id",
        name="n",
        adapter=ADAPTER_STATIC_ORDER,
        features_signature=[],
        features_hash="",
        objective="max",
        score_units="",
        score_calibrated=False,
        score_transform="identity",
    )
    assert buffer[4:8] == UHD_FILE_IDENTIFIER
