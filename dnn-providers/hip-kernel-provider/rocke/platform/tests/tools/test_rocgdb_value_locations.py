# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from rocke.debug.rocgdb_value_locations import (
    bind_debug_description,
    kernel_symbol,
    register_pieces,
    symbol_address,
)


LOCATION = """Symbol "acc83" is multi-location:
  Range 0x100-0x120: a variable in $v0 [4-byte piece], and a variable in $v1 [4-byte piece], and a variable in $v2 [4-byte piece], and a variable in $v3 [4-byte piece]
  Range 0x120-0x140: a variable in $v8 [4-byte piece], and a variable in $v9 [4-byte piece], and a variable in $v10 [4-byte piece], and a variable in $v11 [4-byte piece]
.
"""


def _description(*names):
    return {
        "schema": "rocke-debug-description/v1",
        "kernel": "demo",
        "values": [
            {
                "dwarf": {"name": name, "type": "vec<f32x4>"},
                "logical": {
                    "name": name,
                    "dtype": "f32",
                    "shape": [2, 2],
                    "layout": {"fragment_length": 4},
                },
            }
            for name in names
        ],
    }


def test_symbol_address_reads_minimal_symbol_output():
    text = (
        'Symbol "metadata" is at 0x7fff123405c0 in a file compiled without debugging.\n'
    )
    assert symbol_address(text) == 0x7FFF123405C0


def test_kernel_symbol_ignores_inlined_source_frame_name():
    text = (
        "rocke_full_rank_snapshot_demo + 192 in section .text "
        "of memory://123#offset=0x100&size=200\n"
    )
    assert kernel_symbol(text) == "rocke_full_rank_snapshot_demo"


def test_register_pieces_selects_the_current_pc_range():
    assert [piece.expression for piece in register_pieces(LOCATION, 0x125)] == [
        "$v8",
        "$v9",
        "$v10",
        "$v11",
    ]


def test_register_pieces_rejects_pc_outside_location_ranges():
    with pytest.raises(ValueError, match="0 location ranges"):
        register_pieces(LOCATION, 0x200)


def test_description_binds_the_only_value_without_a_name():
    manifest, names = bind_debug_description(
        _description("acc83"), (), pc=0x110, location_text=lambda name: LOCATION
    )
    assert names == ["acc83"]
    assert manifest["values"][0]["binding"] == {
        "kind": "amdgpu_registers",
        "storage_dtype": "f32",
        "locations": ["$v0", "$v1", "$v2", "$v3"],
    }


def test_description_requires_a_choice_when_several_values_exist():
    with pytest.raises(ValueError, match="choose one of: acc, scores"):
        bind_debug_description(
            _description("scores", "acc"),
            (),
            pc=0x110,
            location_text=lambda name: LOCATION,
        )
