# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the text layout visualizer + structured describe()."""

from __future__ import annotations

from rocke.helpers.tiling.mma.warp_encoding import c_warp_encoding
from rocke.helpers.tiling.visualization import describe, render_forward_map, render_inverse_map
from rocke.helpers.tiling.traits import load_mma_traits


def _c_encoding(op_id: str = "mfma_f32_16x16x16f16"):
    return c_warp_encoding(load_mma_traits().get(op_id))


def test_describe_is_structured_and_machine_readable() -> None:
    info = describe(_c_encoding())
    assert info["num_lanes"] == 64
    assert info["num_vector_items"] == 4
    assert info["num_repeat"] == 1
    assert info["matrix_major_size"] == 16
    assert info["matrix_minor_size"] == 16
    assert info["hierarchical_lengths"] == ((1, 4, 4), (16,))


def test_forward_map_renders_expected_shape_and_cells() -> None:
    text = render_forward_map(_c_encoding(), axis_names=("row", "col"))
    lines = text.splitlines()
    assert "forward map (row,col)" in lines[0]
    assert "64 lanes x 4 regs" in lines[0]
    # lane 0 row: registers 0..3 hold rows 0..3 at col 0.
    lane0 = next(line for line in lines if line.strip().startswith("0 |"))
    assert "0,0" in lane0 and "3,0" in lane0


def test_inverse_map_renders_and_is_ascii() -> None:
    text = render_inverse_map(_c_encoding(), axis_names=("row", "col"))
    assert "inverse map" in text
    assert text.isascii()
    # element (row=2, col=0) is held by lane 0 register 2 (see forward derivation).
    assert "L0r2" in text
