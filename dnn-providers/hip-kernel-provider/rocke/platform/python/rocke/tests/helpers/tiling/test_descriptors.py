# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the memory-layout value objects TensorDesc + TensorWindow (offline, no GPU)."""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.descriptors import make_tensor_desc, make_window


class _DT:
    """Minimal ir.Type-like dtype stub (only `.name` is needed offline)."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_tensor_desc_basics() -> None:
    td = make_tensor_desc((16, 16), (16, 1), _DT("f16"))
    assert td.rank == 2
    assert td.lengths == (16, 16)
    assert td.strides == (16, 1)


def test_tensor_desc_rank_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="lengths rank"):
        make_tensor_desc((16, 16), (1,), _DT("f16"))


def test_permute_swaps_lengths_and_strides() -> None:
    # An N-contiguous tensor described (N, K) = (32, 16), strides (1, 32); permute([1, 0]) gives its
    # (K, N) view -- a view of the same bytes and dtype, not the operand order.
    td = make_tensor_desc((32, 16), (1, 32), _DT("f16"))
    viewed = td.permute([1, 0])
    assert viewed.lengths == (16, 32)
    assert viewed.strides == (32, 1)


def test_permute_rejects_non_permutation() -> None:
    td = make_tensor_desc((16, 16), (16, 1), _DT("f16"))
    with pytest.raises(ValueError, match="must be a permutation"):
        td.permute([0, 0])


def test_make_window_positions_desc_with_origin() -> None:
    td = make_tensor_desc((16, 16), (16, 1), _DT("f16"))
    win = make_window(td, (0, 0))
    assert win.origin == (0, 0)
    assert win.tensor is td


# ---- a window holds a position only: no length, no clip of its own ----------------------------------


def test_a_window_has_no_bounds() -> None:
    # The valid region is the tensor's lengths; a narrower region is a narrower tensor. There is no
    # alias for the removed argument: Python's own keyword error names it.
    td = make_tensor_desc((16, 16), (16, 1), _DT("f16"))
    with pytest.raises(TypeError) as info:
        make_window(td, (0, 0), bounds=(8, 8))
    assert (
        str(info.value) == "make_window() got an unexpected keyword argument 'bounds'"
    )
    assert not hasattr(make_window(td, (0, 0)), "bounds")


def test_window_origin_count_must_match_tensor_rank() -> None:
    with pytest.raises(ValueError) as info:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0, 0))
    assert str(info.value) == (
        "make_window got 3 origins for a rank-2 tensor -- give one origin per tensor axis (a "
        "rank-t tile covers the last t axes; leading axes are positioned by their origin)"
    )


@pytest.mark.parametrize(
    "origin,type_text",
    [(1.5, "float"), ("m", "str"), (None, "NoneType"), (True, "bool")],
)
def test_window_origin_type_is_checked(origin, type_text: str) -> None:
    with pytest.raises(TypeError) as info:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, origin))
    assert str(info.value) == (
        f"make_window origin[1] is a {type_text} -- an origin is an int or an i32 IR value"
    )


def test_an_ir_origin_must_be_i32() -> None:
    from rocke.core.ir import I64, IRBuilder

    b = IRBuilder("origin_type")
    wide = b.param("W", I64)
    with pytest.raises(TypeError) as info:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (wide, 0))
    assert str(info.value) == (
        "make_window origin[0] is a Value of type i64 -- an origin is an int or an i32 IR value"
    )


def test_a_negative_origin_is_accepted() -> None:
    # A convolution halo places a window before the tensor start; the elements outside the tensor
    # read 0 and are not written (the verbs check that side).
    win = make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (-1, -16))
    assert win.origin == (-1, -16)


def test_origins_are_kept_as_ints_or_ir_values() -> None:
    import numpy as np

    from rocke.core.ir import I32, IRBuilder

    b = IRBuilder("origin_kinds")
    b.kernel.attrs["max_workgroup_size"] = 64
    m0 = b.param("M0", I32)
    k0 = b.param("K0", I32, ge=0, le=4096)
    win = make_window(
        make_tensor_desc((4, 16, 16), (256, 16, 1), _DT("f16")), (np.int64(3), m0, k0)
    )
    assert win.origin[0] == 3 and type(win.origin[0]) is int
    assert win.origin[1] is m0
    assert win.origin[2] is k0


def test_with_origin_replaces_one_axis_only() -> None:
    td = make_tensor_desc((4, 16, 16), (256, 16, 1), _DT("f16"))
    win = make_window(td, (1, 0, 0))
    moved = win.with_origin(2, 8)
    assert moved.origin == (1, 0, 8)
    assert moved.tensor is td
    assert win.origin == (1, 0, 0)  # the window is a value: the original is unchanged
    assert win.with_origin(0, -1).origin == (-1, 0, 0)


@pytest.mark.parametrize("axis", [3, -1])
def test_with_origin_refuses_an_axis_the_window_does_not_have(axis: int) -> None:
    win = make_window(
        make_tensor_desc((4, 16, 16), (256, 16, 1), _DT("f16")), (0, 0, 0)
    )
    with pytest.raises(ValueError) as info:
        win.with_origin(axis, 0)
    assert (
        str(info.value) == f"with_origin: axis {axis} does not exist on a rank-3 window"
    )


def test_with_origin_checks_the_origin_type() -> None:
    win = make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0))
    with pytest.raises(TypeError) as info:
        win.with_origin(1, 2.0)
    assert str(info.value) == (
        "with_origin origin for axis 1 is a float -- an origin is an int or an i32 IR value"
    )


def test_the_bounds_of_a_window_are_its_tensors_lengths() -> None:
    # The window keeps no region: the bound checks are built from the window's tensor, one
    # [0, length) per axis, so a check reaches the very length Value the author declared.
    from rocke.core.ir import F16, I32, IRBuilder
    from rocke.helpers.tiling.tensor_transforms import required_checks
    from rocke.helpers.tiling.tensor_transforms.graph import identity_graph

    b = IRBuilder("window_bounds")
    b.kernel.attrs["max_workgroup_size"] = 64
    ms = b.param("M", I32, ge=0, le=1 << 20)
    ks = b.param("K", I32, ge=1, le=1 << 20)
    win = make_window(make_tensor_desc((ms, ks), (ks, 1), F16), (0, 0))
    checks = required_checks(
        identity_graph(win.tensor.lengths), win.tensor, ((-5, None), (-5, None))
    )
    assert [(c.lo, c.hi) for c in checks] == [(0, ms), (0, ks)]
    assert checks[0].hi is ms and checks[1].hi is ks


# ---- one axis convention, no axis labels; leading axes positioned by the origin --------------------


def test_make_tensor_desc_takes_no_axis_labels() -> None:
    # Axes are positional (batch first, reduction last); there is no label argument and no alias.
    with pytest.raises(TypeError, match="unexpected keyword argument 'axis_roles'"):
        make_tensor_desc(
            (4, 16, 8), (128, 8, 1), _DT("f16"), axis_roles=("a", "b", "c")
        )


def test_a_window_takes_one_origin_per_tensor_axis() -> None:
    # A batched (G, M, K) tensor read with a rank-2 (M, K) tile is windowed at (g, m0, k0): the
    # leading batch axis is positioned by its origin, like any other axis.
    win = make_window(make_tensor_desc((4, 16, 8), (128, 8, 1), _DT("f16")), (3, 0, 8))
    assert win.origin == (3, 0, 8)
    assert win.tensor.rank == 3


def test_leading_axis_adds_origin_times_stride_after_the_tile_terms() -> None:
    # A rank-2 tile on a rank-3 window: the tile positions take the strides of the LAST two axes,
    # then the leading axis adds origin * stride.
    from rocke.helpers.tiling.emit import _address
    from rocke.helpers.tiling.lds_conflict import NumBuilder

    nb = NumBuilder(0)
    win = make_window(make_tensor_desc((4, 16, 8), (128, 8, 1), _DT("f16")), (3, 0, 0))
    m, k = 5, 6
    assert _address(nb, win, [m, k]) == m * 8 + k * 1 + 3 * 128
    # A tile that covers every axis has no leading axis: the plain strided sum.
    assert _address(nb, win, [3, m, k]) == 3 * 128 + m * 8 + k * 1
