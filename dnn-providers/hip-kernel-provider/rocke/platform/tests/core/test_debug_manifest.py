# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for portable logical-value and LayoutMap debugger manifests."""

from __future__ import annotations

import pytest
from rocke.core import (
    DEBUG_DESCRIPTION_MAGIC,
    DEBUG_DESCRIPTION_SCHEMA,
    IRBuilder,
    automatic_debug_description,
    bind_logical_value,
    debug_description_symbol,
    debug_manifest,
    embed_debug_description,
    evaluate_layout,
    logical_value_description,
    register_value_binding,
)
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


def _bound_mfma_acc_value(**overrides):
    logical_kwargs = {
        "name": "acc",
        "dtype": "f32",
        "shape": (16, 16),
        "layout": _mfma_acc_layout(),
        "layout_name": "mfma_f32_16x16x16_f16.acc",
    }
    binding_kwargs = {
        "storage_dtype": "f32",
        "locations": ("$v40", "$v41", "$v42", "$v43"),
        "fragment_length": 4,
    }
    logical_kwargs.update(overrides.get("logical", {}))
    binding_kwargs.update(overrides.get("binding", {}))
    logical = logical_value_description(**logical_kwargs)
    binding = register_value_binding(**binding_kwargs)
    return bind_logical_value(logical, binding)


def test_debug_manifest_preserves_semantics_and_locations():
    value = _bound_mfma_acc_value()
    manifest = debug_manifest(value)

    assert manifest["schema"] == "rocke-debug-manifest/v1"
    assert value["binding"]["locations"] == ["$v40", "$v41", "$v42", "$v43"]
    assert value["binding"]["kind"] == "amdgpu_registers"
    assert value["logical"]["layout"]["role"] == "acc"
    assert value["logical"]["layout"]["fragment_length"] == 4
    assert value["logical"]["layout"]["wave_size"] == 64
    assert value["logical"]["layout"]["replication_factor"] == 1


def test_logical_description_has_no_debugger_binding():
    value = logical_value_description(
        name="acc",
        dtype="f32",
        shape=(16, 16),
        layout=_mfma_acc_layout(),
        layout_name="mfma_f32_16x16x16_f16.acc",
    )

    assert value["name"] == "acc"
    assert value["layout"]["packing"] == {
        "kind": "scalar",
        "elements_per_slot": 1,
    }
    assert "binding" not in value
    assert "locations" not in value


def test_logical_description_represents_replicated_and_packed_layouts():
    target = ArchTarget.from_gfx("gfx1151")
    fp16 = target.mma.by_op_id("wmma_f32_16x16x16_f16")
    iu8 = target.mma.by_op_id("wmma_i32_16x16x16_iu8")
    assert fp16 is not None
    assert iu8 is not None

    replicated = logical_value_description(
        name="a",
        dtype="f16",
        shape=(16, 16),
        layout=fp16.a_layout(),
        layout_name=f"{fp16.op_id}.a",
    )
    packed = logical_value_description(
        name="a",
        dtype="iu8",
        shape=(16, 16),
        layout=iu8.a_layout(),
        layout_name=f"{iu8.op_id}.a",
    )

    assert replicated["layout"]["replication_factor"] == 2
    assert replicated["layout"]["coordinates"][0]["index"] == [0, 0]
    assert replicated["layout"]["coordinates"][16 * 16]["index"] == [0, 0]
    assert packed["layout"]["replication_factor"] == 2
    assert packed["layout"]["packing"] == {
        "kind": "contiguous",
        "axis": 1,
        "elements_per_slot": 4,
    }


def test_manifest_rejects_invalid_shapes_locations_and_duplicate_names():
    with pytest.raises(ValueError, match="two positive extents"):
        _bound_mfma_acc_value(logical={"shape": (256,)})
    with pytest.raises(ValueError, match="physical location"):
        _bound_mfma_acc_value(binding={"locations": ()})
    with pytest.raises(ValueError, match="provide 1 elements"):
        _bound_mfma_acc_value(binding={"locations": ("$v40",)})
    with pytest.raises(ValueError, match="requires storage dtype"):
        _bound_mfma_acc_value(
            binding={
                "storage_dtype": "f16x2",
                "locations": ("$v40", "$v41"),
            }
        )

    value = _bound_mfma_acc_value()
    with pytest.raises(ValueError, match="unique"):
        debug_manifest(value, value)


def test_automatic_description_is_binding_free_and_embeddable():
    builder = IRBuilder("debug_description", capture_loc=True)
    accumulator = builder.mfma_f32_16x16x16_f16(
        builder.zero_vec_f16(4),
        builder.zero_vec_f16(4),
        builder.zero_vec_f32_4(),
    )
    builder.debug_value(accumulator)
    builder.ret()

    description = automatic_debug_description(builder.kernel)
    assert description is not None
    assert description["schema"] == DEBUG_DESCRIPTION_SCHEMA
    assert description["kernel"] == "debug_description"
    assert description["values"][0]["dwarf"] == {
        "name": accumulator.name[1:],
        "type": "vec<f32x4>",
    }
    assert "binding" not in description["values"][0]

    llvm = 'target triple = "amdgcn-amd-amdhsa"\n\ndefine void @k() { ret void }\n'
    embedded = embed_debug_description(llvm, builder.kernel)
    assert f"@{debug_description_symbol('debug_description')}" in embedded
    assert 'section ".rocke.debug"' in embedded
    assert embedded.endswith("define void @k() { ret void }\n")
    assert "".join(f"\\{byte:02X}" for byte in DEBUG_DESCRIPTION_MAGIC) in embedded
