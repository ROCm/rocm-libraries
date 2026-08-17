# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for portable logical-value and LayoutMap debugger manifests."""

from __future__ import annotations

import pytest
from rocke.core import debug_manifest, evaluate_layout, logical_value_manifest
from rocke.core.arch import ArchTarget


def _mfma_acc_layout():
    op = ArchTarget.from_gfx("gfx942").mma.by_op_id("mfma_f32_16x16x16_f16")
    assert op is not None
    return op.acc_layout()


def test_evaluate_layout_uses_lane_and_fragment_source_of_truth():
    coordinates = evaluate_layout(_mfma_acc_layout())

    assert len(coordinates) == 64 * 4
    assert coordinates[:5] == [
        {"lane": 0, "slot": 0, "index": [0, 0]},
        {"lane": 0, "slot": 1, "index": [1, 0]},
        {"lane": 0, "slot": 2, "index": [2, 0]},
        {"lane": 0, "slot": 3, "index": [3, 0]},
        {"lane": 1, "slot": 0, "index": [0, 1]},
    ]
    assert coordinates[-1] == {"lane": 63, "slot": 3, "index": [15, 15]}


def test_logical_value_manifest_preserves_semantics_and_locations():
    value = logical_value_manifest(
        name="acc",
        dtype="f32",
        shape=(16, 16),
        layout=_mfma_acc_layout(),
        layout_name="mfma_f32_16x16x16_f16.acc",
        storage_dtype="f32",
        locations=("$v40", "$v41", "$v42", "$v43"),
    )
    manifest = debug_manifest(value)

    assert manifest["schema"] == "rocke-debug-manifest/v1"
    assert value["locations"] == ["$v40", "$v41", "$v42", "$v43"]
    assert value["layout"]["role"] == "acc"
    assert value["layout"]["fragment_length"] == 4
    assert value["layout"]["wave_size"] == 64


def test_manifest_rejects_invalid_shapes_locations_and_duplicate_names():
    kwargs = {
        "name": "acc",
        "dtype": "f32",
        "shape": (16, 16),
        "layout": _mfma_acc_layout(),
        "layout_name": "mfma_f32_16x16x16_f16.acc",
        "storage_dtype": "f32",
        "locations": ("$v40", "$v41", "$v42", "$v43"),
    }
    with pytest.raises(ValueError, match="two positive extents"):
        logical_value_manifest(**{**kwargs, "shape": (256,)})
    with pytest.raises(ValueError, match="physical location"):
        logical_value_manifest(**{**kwargs, "locations": ()})
    with pytest.raises(ValueError, match="provide 1 elements"):
        logical_value_manifest(**{**kwargs, "locations": ("$v40",)})
    with pytest.raises(ValueError, match="requires storage dtype"):
        logical_value_manifest(**{**kwargs, "storage_dtype": "f16x2"})

    value = logical_value_manifest(**kwargs)
    with pytest.raises(ValueError, match="unique"):
        debug_manifest(value, value)
