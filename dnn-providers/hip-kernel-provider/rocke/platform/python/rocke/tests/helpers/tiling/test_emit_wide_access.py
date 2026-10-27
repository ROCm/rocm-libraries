# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A wide fragment access covers registers that sit at consecutive positions in memory.

``load_fragment`` and ``store_fragment`` move several registers in one vector access when the
innermost register bucket runs along the tensor's stride-1 axis. Those registers are consecutive
positions on that axis only when the bucket is the axis's innermost hierarchical level. The
interleaved accumulator layout of a 64 x 64 x 32 f16 tile on gfx90a puts its innermost register
bucket on an OUTER level of M: four consecutive registers hold M positions 0, 4, 8 and 12. A wide
access there would move M 0..3 instead, so these tests check:

- that layout on an M-stride-1 tensor gets no wide access (global and LDS), and the LDS conflict
  model, which mirrors the verbs, agrees;
- layouts whose innermost register bucket is the innermost level keep their vector width;
- for every TileMma operand layout and every axis made stride 1, each access run holds registers at
  consecutive positions on the stride-1 axis.

The GPU round trip of that accumulator through an M-contiguous LDS buffer lives with the other LDS
round trips in ``test_lds_descriptor_addressing.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rocke.core.ir", reason="rocke IR substrate required")

from rocke.core.ir import F16, F32
from rocke.helpers.tiling import (
    InterleavedStyle,
    TileMma,
    Tiling,
    make_tensor_desc,
    make_window,
)
from rocke.helpers.tiling.emit import _contiguous_run, _register_coord_offsets
from rocke.helpers.tiling.lds_conflict import access_width


def _mma(interleaved: bool) -> TileMma:
    style = {"style": InterleavedStyle()} if interleaved else {}
    return TileMma(
        (64, 64, 32),
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
        **style,
    )


def _tensor(shape, stride_one_axis: int, dtype):
    """A packed tensor of `shape` whose `stride_one_axis` is stride 1 and the other axis follows it."""
    rows, cols = shape
    strides = (1, rows) if stride_one_axis == 0 else (cols, 1)
    return make_tensor_desc(shape, strides, dtype)


def _width(tile_desc, tensor, is_lds: bool) -> int:
    return _contiguous_run(tile_desc, make_window(tensor, (0, 0)), tensor.dtype, is_lds)


@pytest.mark.parametrize("is_lds", [False, True])
def test_interleaved_accumulator_on_an_m_contiguous_tensor_gets_no_wide_access(
    is_lds: bool,
) -> None:
    c_desc = _mma(interleaved=True).c_desc
    tensor = _tensor(c_desc.shape, 0, F32)
    assert _width(c_desc, tensor, is_lds) == 1
    assert access_width(c_desc, tensor, False) == 1


@pytest.mark.parametrize(
    "interleaved,operand,stride_one_axis,dtype,width",
    [
        (False, "c_desc", 0, F32, 4),
        (False, "a_desc", 1, F16, 4),
        (False, "b_desc", 1, F16, 4),
        (True, "a_desc", 1, F16, 4),
        (True, "b_desc", 1, F16, 4),
    ],
)
@pytest.mark.parametrize("is_lds", [False, True])
def test_innermost_level_layouts_keep_their_vector_width(
    interleaved: bool,
    operand: str,
    stride_one_axis: int,
    dtype,
    width: int,
    is_lds: bool,
) -> None:
    desc = getattr(_mma(interleaved), operand)
    assert _width(desc, _tensor(desc.shape, stride_one_axis, dtype), is_lds) == width


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("operand", ["a_desc", "b_desc", "c_desc"])
@pytest.mark.parametrize("stride_one_axis", [0, 1])
def test_every_access_run_holds_consecutive_positions(
    interleaved: bool, operand: str, stride_one_axis: int
) -> None:
    # Every register run the verbs move in one access must step the stride-1 position by exactly 1
    # from one register to the next, so element i of the vector is the element at base + i.
    desc = getattr(_mma(interleaved), operand)
    dtype = F32 if operand == "c_desc" else F16
    width = _width(desc, _tensor(desc.shape, stride_one_axis, dtype), True)
    for start in range(0, desc.register_count, width):
        positions = [
            _register_coord_offsets(desc.layout, start + i)[stride_one_axis]
            for i in range(width)
        ]
        assert positions == list(
            range(positions[0], positions[0] + width)
        ), f"registers {start}..{start + width - 1} hold positions {positions}"
