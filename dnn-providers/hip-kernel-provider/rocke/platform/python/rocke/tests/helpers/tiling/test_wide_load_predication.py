# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Wide accesses and their bound checks are two separate decisions.

The width of an access follows the layout and the strides alone: the innermost register run on the
tensor's stride-1 axis, as wide as the dtype allows and the other strides divide -- the same rule
in global memory and LDS. Whether a side of a global access is checked follows the shared bound-check
rule alone: a side is proven only when the ranges of the window's positions show it cannot fail. A
ragged tensor or an origin the IR knows nothing about keeps its width and is masked by its
validity; nothing is trusted, and a tile alignment is not a proof. Offline, no GPU.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.helpers.tiling import (
    SideCheck,
    explain_bounds_checks,
    make_tensor_desc,
    make_window,
)
from rocke.helpers.tiling.clipping import contiguous_run
from rocke.helpers.tiling.memory import cooperative_load_desc

# A (free, K) cooperative-load tile: free (axis 0) is stride-1, tile extent 128.
_COOP = cooperative_load_desc(128, 16, 4, vw=8)

NOT_NEEDED, PER_LANE = SideCheck.NOT_NEEDED, SideCheck.PER_LANE_CHECK


def _builder() -> tuple[IRBuilder, object]:
    b = IRBuilder("wide_load_gate")
    b.kernel.attrs["max_workgroup_size"] = 256
    return b, b.param("X", PtrType(F16, "global"))


def _vw(lengths, *, origin=(0, 0), k_stride=None):
    tensor = make_tensor_desc(lengths, (1, k_stride or lengths[0]), F16)
    return contiguous_run(_COOP, make_window(tensor, origin), F16)


def _sides(lengths, *, origin=(0, 0), b=None, ptr=None):
    """``((lower, upper) of axis 0, (lower, upper) of axis 1)`` of a free-contiguous window."""
    if b is None:
        b, ptr = _builder()
    tensor = make_tensor_desc(lengths, (1, 256), F16)
    plan = explain_bounds_checks(b, ptr, make_window(tensor, origin), _COOP)
    return tuple((axis.lower, axis.upper) for axis in plan.axes)


def test_an_aligned_tensor_is_wide_and_proven() -> None:
    # length 256 is a multiple of the tile extent 128 at origin 0: wide (vw 8), nothing checked.
    assert _vw((256, 16)) == 8
    assert _sides((256, 16)) == ((NOT_NEEDED, NOT_NEEDED), (NOT_NEEDED, NOT_NEEDED))


def test_a_ragged_run_axis_stays_wide_and_checks_its_upper_side() -> None:
    # length 250 is not a multiple of 128: the far tile, at origin 128, overhangs, so the upper side
    # of the run axis is checked -- the access keeps its width and is masked by its validity. The
    # near tile, at origin 0, is inside.
    assert _vw((250, 16), k_stride=256) == 8
    far = _sides((250, 16), origin=(128, 0))
    assert far == ((NOT_NEEDED, PER_LANE), (NOT_NEEDED, NOT_NEEDED))
    assert _sides((250, 16))[0] == (NOT_NEEDED, NOT_NEEDED)


def test_a_narrower_region_is_a_narrower_tensor() -> None:
    # The checks key on the tensor's length against the tile, whatever the allocation behind it: a
    # region of 256 rows inside a larger buffer is described as a 256-row tensor (proven); 200 rows
    # are ragged (the tile at origin 128 is checked).
    assert _sides((256, 16), origin=(128, 0))[0] == (NOT_NEEDED, NOT_NEEDED)
    assert _sides((200, 16), origin=(128, 0))[0] == (NOT_NEEDED, PER_LANE)


@pytest.mark.parametrize("axis", [0, 1])
def test_a_runtime_length_on_a_tile_axis_checks_that_upper_side(axis: int) -> None:
    # A runtime length stated with no multiple-of fact can end mid-tile: its upper side is checked;
    # the other axis stays proven and the access stays wide.
    b, ptr = _builder()
    runtime = b.param("L", I32, ge=0, le=1 << 20)
    lengths = [256, 16]
    lengths[axis] = runtime
    assert _vw((256, 16)) == 8
    sides = _sides(tuple(lengths), b=b, ptr=ptr)
    assert sides[axis] == (NOT_NEEDED, PER_LANE)
    assert sides[1 - axis] == (NOT_NEEDED, NOT_NEEDED)


@pytest.mark.parametrize("k_stride,vw", [(256, 8), (260, 4), (258, 2), (257, 1)])
def test_the_vector_run_divides_the_other_stride(k_stride: int, vw: int) -> None:
    # Column k starts at element k * k_stride; a wide access claims the alignment of its width, so
    # the width must divide that stride -- one rule for global memory and LDS.
    assert _vw((256, 16), k_stride=k_stride) == vw


def test_an_aligned_int_origin_is_proven() -> None:
    assert _sides((256, 16), origin=(128, 0))[0] == (NOT_NEEDED, NOT_NEEDED)


def test_an_int_origin_off_the_tile_grid_is_proven_when_the_tile_fits() -> None:
    # origin 8 is not a multiple of the extent 128, yet 8 + 128 <= 256: the tile is inside the
    # tensor, so nothing is checked. Alignment is not what decides a check -- the ranges are.
    assert _sides((256, 16), origin=(8, 0))[0] == (NOT_NEEDED, NOT_NEEDED)
    assert _sides((256, 16), origin=(136, 0))[0] == (NOT_NEEDED, PER_LANE)


def test_an_origin_the_ir_knows_nothing_about_checks_both_sides() -> None:
    # A kernel parameter can be anything: nothing is trusted, both sides of its axis are checked.
    b, ptr = _builder()
    origin = (b.param("M0", I32), 0)
    assert _sides((256, 16), origin=origin, b=b, ptr=ptr)[0] == (PER_LANE, PER_LANE)


@pytest.mark.parametrize(
    "origin,sides",
    [
        ((-128, 0), ((PER_LANE, NOT_NEEDED), (NOT_NEEDED, NOT_NEEDED))),
        ((0, -16), ((NOT_NEEDED, NOT_NEEDED), (PER_LANE, NOT_NEEDED))),
        ((-8, 0), ((PER_LANE, NOT_NEEDED), (NOT_NEEDED, NOT_NEEDED))),
    ],
)
def test_a_tile_placed_before_the_tensor_start_checks_its_lower_side(
    origin, sides
) -> None:
    # A negative origin (a convolution halo) puts part of the tile before the tensor: the lower side
    # of that axis is checked, the access stays wide.
    assert _vw((256, 16), origin=origin) == 8
    assert _sides((256, 16), origin=origin) == sides


# ---- a rank-2 tile on a rank-3 tensor: the run axis is read on the tensor's last axes ----


def test_vector_run_on_the_last_axes_of_a_rank_3_tensor_is_wide() -> None:
    # (G, free, K) with the free axis stride 1 at tensor axis 1. The (free, K) tile covers the last
    # two axes, so its run axis (tile axis 0) is tensor axis 1 -- stride 1 -> wide. Read without the
    # leading-axis offset it would be the batch axis (stride free * K) and scalarize.
    tensor = make_tensor_desc((3, 256, 16), (256 * 16, 1, 256), F16)
    assert contiguous_run(_COOP, make_window(tensor, (2, 0, 0)), F16) == 8


def test_a_ragged_run_axis_of_a_rank_3_tensor_is_checked_like_rank_2() -> None:
    # The leading axis at an int origin inside the tensor is proven on both sides; the ragged free
    # axis is checked on its upper side exactly as the rank-2 tensor of that length is.
    b, ptr = _builder()
    tensor = make_tensor_desc((3, 250, 16), (250 * 16, 1, 250), F16)
    plan = explain_bounds_checks(b, ptr, make_window(tensor, (1, 128, 0)), _COOP)
    assert [(axis.lower, axis.upper) for axis in plan.axes] == [
        (NOT_NEEDED, NOT_NEEDED),
        (NOT_NEEDED, PER_LANE),
        (NOT_NEEDED, NOT_NEEDED),
    ]


# ---- masking LDS is structurally unrepresentable -- the LDS verbs carry NO bounds parameter ----


def test_lds_verbs_have_no_bounds_parameter() -> None:
    import inspect

    from rocke.helpers.tiling.memory import lds_read, lds_store

    assert (
        "bounds" not in inspect.signature(lds_store).parameters
    ), "lds_store must not accept bounds"
    assert (
        "bounds" not in inspect.signature(lds_read).parameters
    ), "lds_read must not accept bounds"
