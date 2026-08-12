#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for UHD FlatBuffer writer."""
import tempfile
from pathlib import Path

import pytest

from uhd_gen.uhd_to_flatbuffer import (
    ADAPTER_STATIC_ORDER,
    ADAPTER_TREE_DATA,
    UHD_FILE_IDENTIFIER,
    build_uhd,
    convert_uhd,
)


def test_build_uhd_tree_data():
    """Test building a tree_data UHD FlatBuffer."""
    buffer = build_uhd(
        uhd_id="test-uhd-123",
        name="Test UHD",
        adapter=ADAPTER_TREE_DATA,
        features_signature=["$q.M", "$q.N", "$kernel.tile_m"],
        features_hash="sha256:abcd1234",
        objective="max",
        score_units="tflops",
        score_calibrated=False,
        score_transform="log1p",
        model_artifact_path="model.bin",
        model_hash=None,
        static_order_fields=None,
        custom_library_symbol=None,
    )

    assert len(buffer) > 0
    # FlatBuffer file identifier is at offset 4-7 (after the 4-byte size prefix)
    assert buffer[4:8] == UHD_FILE_IDENTIFIER


def test_build_uhd_static_order():
    """Test building a static_order UHD FlatBuffer."""
    buffer = build_uhd(
        uhd_id="static-uhd-456",
        name="Static Order UHD",
        adapter=ADAPTER_STATIC_ORDER,
        features_signature=[],  # static_order has no features
        features_hash="",
        objective="max",
        score_units="priority",
        score_calibrated=False,
        score_transform="identity",
        model_artifact_path=None,
        model_hash=None,
        static_order_fields=["priority", "id"],
        custom_library_symbol=None,
    )

    assert len(buffer) > 0
    # FlatBuffer file identifier is at offset 4-7 (after the 4-byte size prefix)
    assert buffer[4:8] == UHD_FILE_IDENTIFIER


def test_convert_uhd_to_file():
    """Test converting UHD to a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.uhd"

        convert_uhd(
            uhd_id="file-test-789",
            name="File Test UHD",
            adapter="tree_data",
            features_signature=["$q.batch", "$device.cu_count"],
            features_hash="sha256:xyz789",
            objective="min",
            score_units="ms",
            score_calibrated=True,
            score_transform="identity",
            output_path=output_path,
            model_artifact_path="my_model.bin",
            model_hash="md5:abc123",
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify file identifier (at offset 4-7)
        with open(output_path, "rb") as f:
            f.read(4)  # Skip size prefix
            file_id = f.read(4)
            assert file_id == UHD_FILE_IDENTIFIER


def test_convert_uhd_validation():
    """Test that convert_uhd validates adapter types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "invalid.uhd"

        with pytest.raises(ValueError, match="Unknown adapter type"):
            convert_uhd(
                uhd_id="invalid-adapter",
                name="Invalid",
                adapter="bogus_adapter",  # Invalid adapter type
                features_signature=[],
                features_hash="",
                objective="max",
                score_units="",
                score_calibrated=False,
                score_transform="identity",
                output_path=output_path,
            )


def test_convert_uhd_requires_model_path_for_tree_data():
    """Test that tree_data adapter requires model_artifact_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "missing_path.uhd"

        with pytest.raises(ValueError, match="requires model_artifact_path"):
            convert_uhd(
                uhd_id="missing-path",
                name="Missing Path",
                adapter="tree_data",
                features_signature=["$q.M"],
                features_hash="sha256:123",
                objective="max",
                score_units="tflops",
                score_calibrated=False,
                score_transform="log1p",
                output_path=output_path,
                # model_artifact_path missing!
            )


def test_all_adapter_types():
    """Test that all adapter enum values can be serialized."""
    adapters = [
        ("static_order", ADAPTER_STATIC_ORDER, None),
        ("tree_data", ADAPTER_TREE_DATA, "model.bin"),
        ("table", 2, "table.csv"),  # ADAPTER_TABLE
        ("onnx", 3, "model.onnx"),  # ADAPTER_ONNX
        ("custom_library", 4, "libscorer.so"),  # ADAPTER_CUSTOM_LIBRARY
    ]

    for adapter_name, adapter_value, model_path in adapters:
        buffer = build_uhd(
            uhd_id=f"test-{adapter_name}",
            name=f"{adapter_name} UHD",
            adapter=adapter_value,
            features_signature=["$q.x"] if model_path else [],
            features_hash="sha256:test",
            objective="max",
            score_units="score",
            score_calibrated=False,
            score_transform="identity",
            model_artifact_path=model_path,
            model_hash=None,
            static_order_fields=["priority", "id"] if adapter_name == "static_order" else None,
            custom_library_symbol="myScorer" if adapter_name == "custom_library" else None,
        )
        assert len(buffer) > 0
        assert buffer[4:8] == UHD_FILE_IDENTIFIER
