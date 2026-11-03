# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The bank-conflict address map uses the LDS verbs' own index helper.

``lds_conflict.addr_map`` evaluates the helpers ``load_fragment`` / ``store_fragment`` build every LDS
index with, over plain ints, for a ``TensorDesc``; it carries each element's positions instead of
decoding them back from the index. These tests check, CPU only:

- on the geometries the measured bank-conflict corpus was taken on (the cooperative stores at every
  pad of the pad sweep, the MMA-operand reads at every read pad, and the swizzled stores), the map
  gives the same vector width and the same index for every (lane, run) as the formula it used before:
  ``sum((origin + coordinate) * stride)`` over the (swizzled) positions;
- the carried positions are the element's own: for an N-contiguous buffer described ``(N, K)`` with
  strides ``(1, N)`` -- where a decode by ``strides[0]`` cannot work -- the store and read pictures
  give each dword's (row, column) as (N, K) positions that index back to that dword;
- the (row, column) pictures refuse a descriptor that is not rank 2.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16
from rocke.helpers.tiling import lds_conflict as lc
from rocke.helpers.tiling.descriptors import make_tensor_desc, make_window
from rocke.helpers.tiling.emit import (
    _contiguous_run,
    _swizzle_lds_positions,
    _swizzle_vw,
    emit_tensor_coordinates,
)
from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import (
    _macro_coop_descs_crc,
)
from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
    _transpose_desc,
    _wave_descs_interleaved,
    b32_swizzle,
    b64_swizzle,
)

# The measured pad sweep's pads (in f16 elements) and the measured read rows' pads.
_STORE_PADS = (0, 8, 16, 24, 32, 40, 48, 56, 64)
_READ_PADS = (0, 2, 4, 6, 8)
_K_EXTENT = 32


def _previous_addresses(tile_desc, strides, origin, n_lanes, lds_swizzle):
    """The map as it was computed before it called the verbs' helper: the contiguous-run width
    (capped by a swizzle callable at its granularity, alignment 2 for f16) and, per (lane, run),
    ``sum(position * stride)`` over the run start's (swizzled) positions."""
    window = make_window(make_tensor_desc((1, 1), strides, F16), (0, 0))
    vw = _swizzle_vw(lds_swizzle, _contiguous_run(tile_desc, window, F16, True), 2)
    swizzle = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
    rows = []
    for lane in range(n_lanes):
        nb = lc.NumBuilder(lane)
        for reg0 in range(0, tile_desc.register_count, vw):
            coords = emit_tensor_coordinates(nb, tile_desc.layout, lane, reg0)
            positions = [origin[axis] + coord for axis, coord in enumerate(coords)]
            if lds_swizzle:
                positions = swizzle(nb, positions)
            index = sum(p * s for p, s in zip(positions, strides))
            rows.append((lane, reg0, index, tuple(positions)))
    return vw, rows


def _assert_matches_previous(tile_desc, strides, origin, n_lanes, lds_swizzle):
    tensor = make_tensor_desc(lc.desc_extents(tile_desc, n_lanes, origin), strides, F16)
    accesses, vw = lc.addr_map(
        tile_desc, tensor, origin=origin, n_lanes=n_lanes, lds_swizzle=lds_swizzle
    )
    previous_vw, previous = _previous_addresses(
        tile_desc, strides, origin, n_lanes, lds_swizzle
    )
    assert vw == previous_vw
    assert [
        (a["lane"], a["reg0"], a["base"], a["element_positions"][0]) for a in accesses
    ] == previous


# The cooperative stores the store corpus was measured on: f16 (K, free) tiles 128 and 256 wide,
# K 16 and 32, 4 to 16 waves (b32 / b64 / b128 stores), at every measured pad.
@pytest.mark.parametrize("pad", _STORE_PADS)
@pytest.mark.parametrize("tile_free", (128, 256))
@pytest.mark.parametrize("tile_k", (16, 32))
@pytest.mark.parametrize("n_waves", (4, 8, 16))
def test_cooperative_store_map_equals_the_previous_formula(
    pad: int, tile_free: int, tile_k: int, n_waves: int
) -> None:
    store = _transpose_desc(_macro_coop_descs_crc(tile_free, tile_k, n_waves))
    _assert_matches_previous(store, (tile_free + pad, 1), (0, 0), 64, False)


# The MMA-operand reads the read corpus was measured on: 256-wide tile, m_sub 2..16, k_sub 1, at
# every measured read pad, read by a wave whose free origin is not zero.
@pytest.mark.parametrize("pad", _READ_PADS)
@pytest.mark.parametrize("m_sub", (2, 4, 8, 16))
def test_operand_read_map_equals_the_previous_formula(pad: int, m_sub: int) -> None:
    read = _transpose_desc(_wave_descs_interleaved(m_sub, 1, 1)[0])
    _assert_matches_previous(read, (256 + pad, 1), (0, 64), 64, False)


# The swizzled stores: the built-in swizzle and the two bank swizzles the demos use.
@pytest.mark.parametrize(
    "lds_swizzle", (True, b32_swizzle, b64_swizzle), ids=("builtin", "b32", "b64")
)
def test_swizzled_store_map_equals_the_previous_formula(lds_swizzle) -> None:
    store = _transpose_desc(_macro_coop_descs_crc(256, _K_EXTENT, 16))
    _assert_matches_previous(store, (256, 1), (0, 0), 64, lds_swizzle)


def _n_contiguous_store():
    """A 64-wide cooperative tile in (free, K) order -- free innermost in the registers -- and its
    buffer described (N, K) with strides (1, N): an N-contiguous ``[K][N]`` buffer."""
    tile_free = 64
    return _macro_coop_descs_crc(tile_free, _K_EXTENT, 4), (1, tile_free), tile_free


def test_store_picture_carries_the_positions_of_an_n_contiguous_buffer() -> None:
    store, strides, tile_free = _n_contiguous_store()
    _acc, _vw, datum = lc.store_datum(
        store, tile_free, lc.GFX90A, strides, "f16", origin=(0, 0), lds_swizzle=False
    )
    assert datum
    per_dword = 2  # f16
    for row, column, dword, _bank in datum.values():
        assert 0 <= row < tile_free and 0 <= column < _K_EXTENT
        assert row + column * tile_free == dword * per_dword


def test_read_picture_carries_the_positions_of_an_n_contiguous_buffer() -> None:
    _store, strides, tile_free = _n_contiguous_store()
    read = _wave_descs_interleaved(2, 1, 1)[0]  # (free, K), free innermost
    _acc, _vw, datum = lc.read_datum(
        read, tile_free, lc.GFX90A, strides, "f16", origin=(0, 0), lds_swizzle=False
    )
    assert datum
    for row, column, dword, _bank in datum.values():
        assert 0 <= row < tile_free and 0 <= column < _K_EXTENT
        assert (row + column * tile_free) // 2 == dword


def test_pictures_refuse_a_descriptor_that_is_not_rank_2() -> None:
    store, _strides, _tile_free = _n_contiguous_store()
    with pytest.raises(
        ValueError,
        match=r"store_datum: the \(row, column\) bank-conflict picture needs a rank-2 LDS "
        r"descriptor -- strides \(2048, 1, 64\) have rank 3",
    ):
        lc.store_datum(
            store,
            64,
            lc.GFX90A,
            (2048, 1, 64),
            "f16",
            origin=(0, 0, 0),
            lds_swizzle=False,
        )
