# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Computed features require variation in every underlying device field."""
import pytest

pd = pytest.importorskip("pandas")

from uhd_gen.coverage import device_field_coverage, enforce_device_coverage, propose_features


def _boards():
    return pd.DataFrame({"device": ["board-a", "board-b"], "device.cu_count": [120, 120],
                         "device.memory_bytes": [100, 200], "attention.query.dims[2]": [128, 256],
                         "kernel.tile_m": [32, 64]})


def test_two_boards_with_same_cu_count_do_not_support_cu_normalization():
    coverage = device_field_coverage(_boards())
    assert coverage["device_count"] == 2
    with pytest.raises(ValueError, match="device.cu_count"):
        enforce_device_coverage([{"/": ["$attention.query.dims[2]", "$device.cu_count"]}], coverage)
    # Another device field genuinely varies, and is independently safe.
    enforce_device_coverage([{"/": ["$attention.query.dims[2]", "$device.memory_bytes"]}], coverage)
    enforce_device_coverage(["$device.cu_count"], coverage)


def test_each_nested_dependency_needs_its_own_variation():
    expression = {"/": [{"*": ["$attention.query.dims[2]", "$device.memory_bytes"]}, "$device.cu_count"]}
    with pytest.raises(ValueError, match="device.cu_count"):
        enforce_device_coverage([expression], device_field_coverage(_boards()))


def test_auto_proposal_omits_only_unsafe_device_expression():
    frame = _boards().drop(columns="device")
    signature, omitted = propose_features(frame, {"kernel.tile_m"}, [("attention.query.dims[2]", "kernel.tile_m")])
    assert "$device.cu_count" in signature
    assert {"ceil_div": ["$attention.query.dims[2]", "$kernel.tile_m"]} in signature
    assert omitted == [{"expression": {"/": [{"ceil_div": ["$attention.query.dims[2]", "$kernel.tile_m"]}, "$device.cu_count"]},
                        "constant_device_fields": ["device.cu_count"]}]


def test_inconsistent_facts_for_same_device_are_rejected():
    frame = _boards()
    frame["device"] = "one-board"
    with pytest.raises(ValueError, match="device.memory_bytes"):
        device_field_coverage(frame)


def test_missing_device_values_are_not_evidence_of_variation():
    frame = pd.DataFrame({"device.cu_count": [120, 240, None]})
    with pytest.raises(ValueError, match="device.cu_count"):
        enforce_device_coverage([{"/": [512, "$device.cu_count"]}], device_field_coverage(frame))
