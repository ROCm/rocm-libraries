# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The wide-GLOBAL-load fully-in-bounds gate (Principle 4). An unmasked wide global load is emitted
ONLY when the tile is provably in bounds; a ragged/odd tensor, a runtime length that can't be proven a
tile multiple, or a tile placed before the tensor start falls to the scalar masked path -- never an
out-of-bounds wide read. The valid region is always the tensor's own lengths: a window has none of its
own. LDS is exempt (its allocation is the full tile by construction). Offline, no GPU.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16, I32, IRBuilder
from rocke.helpers.tiling.descriptors import (
    make_tensor_desc,
    make_window,
)
from rocke.helpers.tiling.emit import _contiguous_run
from rocke.helpers.tiling.memory import cooperative_load_desc

# A (free, K) cooperative-load tile: free (axis 0) is stride-1, tile extent 128.
_COOP = cooperative_load_desc(128, 16, 4, vw=8)


def _vw(lengths, *, is_lds=False, origin=(0, 0)):
    tensor = make_tensor_desc(lengths, (1, lengths[0]), F16)  # axis 0 stride-1
    return _contiguous_run(_COOP, make_window(tensor, origin), F16, is_lds)


def _builder() -> IRBuilder:
    b = IRBuilder("wide_load_gate")
    b.kernel.attrs["max_workgroup_size"] = 64
    return b


def test_aligned_tensor_gets_wide_load() -> None:
    # length 256 is a multiple of the tile extent 128 -> no tile overhangs -> wide (vw 8).
    assert _vw((256, 16)) == 8


def test_ragged_run_axis_scalarizes() -> None:
    # length 250 is NOT a multiple of 128: the far tile overhangs, so an unmasked wide read would be
    # OOB -> scalarize (the masked path handles the clip safely).
    assert _vw((250, 16)) == 1


def test_a_narrower_region_is_a_narrower_tensor() -> None:
    # The gate keys on the tensor's length against the tile, whatever the allocation behind it: a
    # region of 128 rows inside a larger buffer is described as a 128-row tensor (wide); 200 rows
    # are ragged (scalar).
    assert _vw((128, 16)) == 8
    assert _vw((200, 16)) == 1


@pytest.mark.parametrize("axis", [0, 1])
def test_a_runtime_length_on_a_tile_axis_scalarizes(axis: int) -> None:
    # A runtime length can't be proven a tile multiple at build time -> scalarize, on the run axis
    # and on the other tile axis alike.
    b = _builder()
    runtime = b.param("L", I32, ge=0, le=1 << 20)
    lengths = [256, 16]
    lengths[axis] = runtime
    tensor = make_tensor_desc(tuple(lengths), (1, 256), F16)
    assert _contiguous_run(_COOP, make_window(tensor, (0, 0)), F16, False) == 1


def test_lds_access_is_exempt_from_the_gate() -> None:
    # Even a "ragged" length is wide for LDS: the allocation is the full tile, so no overhang. The K
    # stride (256) keeps every run aligned.
    tensor = make_tensor_desc((250, 16), (1, 256), F16)
    assert _contiguous_run(_COOP, make_window(tensor, (0, 0)), F16, True) == 8


@pytest.mark.parametrize("is_lds", [False, True])
@pytest.mark.parametrize("k_stride,vw", [(256, 8), (260, 4), (258, 2), (257, 1)])
def test_the_vector_run_divides_the_other_stride(
    is_lds: bool, k_stride: int, vw: int
) -> None:
    # Column k starts at element k * k_stride; a wide access claims the alignment of its width, so
    # the width must divide that stride in global memory and LDS alike.
    tensor = make_tensor_desc((256, 16), (1, k_stride), F16)
    assert _contiguous_run(_COOP, make_window(tensor, (0, 0)), F16, is_lds) == vw


# ---- #32: origin-alignment on the wide-load gate (compile-time mid-tile origin caught; SSA trusted) ----


def test_aligned_int_origin_stays_wide() -> None:
    assert _vw((256, 16), origin=(128, 0)) == 8  # origin 128 is on the tile grid (128)


def test_compile_time_mid_tile_origin_scalarizes() -> None:
    # origin 8 is NOT a multiple of the tile extent 128: the tile straddles the grid and can overhang
    # the far edge even though the extent (256) is aligned -> scalarize (masked-safe).
    assert _vw((256, 16), origin=(8, 0)) == 1


def test_ssa_origin_is_trusted_grid_aligned() -> None:
    # A runtime origin is trusted grid-aligned -> stays wide. PARTIAL guard: an SSA mid-tile origin
    # (attention) is not caught here; that is deferred with attention.
    b = _builder()
    assert _vw((256, 16), origin=(b.param("M0", I32), 0)) == 8


@pytest.mark.parametrize("origin", [(-128, 0), (0, -16), (-8, 0)])
def test_a_tile_placed_before_the_tensor_start_scalarizes(origin) -> None:
    # A negative origin (a convolution halo) puts part of the tile before the tensor: the masked
    # path's lower compare keeps those elements out, so no unmasked wide read -- even for a
    # tile-aligned negative origin, whose upper side alone would be proven.
    assert _vw((256, 16), origin=origin) == 1
    assert _vw((256, 16), origin=origin, is_lds=True) == 8  # LDS origins are trusted


# ---- a rank-2 tile on a rank-3 tensor: the run axis is read on the tensor's last axes ----


def test_vector_run_on_the_last_axis_of_a_rank_3_tensor_is_wide() -> None:
    # (G, free, K) with the free axis stride 1 at tensor axis 1. The (free, K) tile covers the last
    # two axes, so its run axis (tile axis 0) is tensor axis 1 -- stride 1 -> wide. Read without the
    # leading-axis offset it would be the batch axis (stride free * K) and scalarize.
    tensor = make_tensor_desc((3, 256, 16), (256 * 16, 1, 256), F16)
    assert _contiguous_run(_COOP, make_window(tensor, (2, 0, 0)), F16, False) == 8


def test_ragged_run_axis_of_a_rank_3_tensor_scalarizes_like_rank_2() -> None:
    # The fully-in-bounds gate checks only the tile axes: a ragged free length scalarizes exactly as
    # the rank-2 tensor of that length does, whatever the leading axis.
    tensor = make_tensor_desc((3, 250, 16), (250 * 16, 1, 250), F16)
    assert _contiguous_run(_COOP, make_window(tensor, (1, 0, 0)), F16, False) == 1
    assert _vw((250, 16)) == 1


# ---- #31: masking LDS is structurally unrepresentable -- the LDS verbs carry NO bounds parameter ----


def test_lds_verbs_have_no_bounds_parameter() -> None:
    import inspect

    from rocke.helpers.tiling.memory import lds_read, lds_store

    assert (
        "bounds" not in inspect.signature(lds_store).parameters
    ), "lds_store must not accept bounds"
    assert (
        "bounds" not in inspect.signature(lds_read).parameters
    ), "lds_read must not accept bounds"
