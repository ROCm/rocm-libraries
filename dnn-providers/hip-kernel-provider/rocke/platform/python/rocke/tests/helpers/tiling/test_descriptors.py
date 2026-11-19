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
    win = make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0))
    assert win.origin == (0, 0)
    assert win.bounds is None  # clip defaults to the desc lengths
    assert win.tensor is not None


def test_make_window_bounds_override() -> None:
    win = make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0), (8, 8))
    assert win.bounds == (8, 8)


def test_window_origin_rank_must_match_tensor() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0, 0))
    msg = str(excinfo.value)
    assert "origin rank" in msg and "tensor rank" in msg


def test_window_bounds_rank_must_match_tensor() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (0, 0), (8,))
    assert "bounds rank" in str(excinfo.value)


def test_window_rejects_negative_compile_time_origin() -> None:
    # A negative origin places the tile before the buffer start; a tile-aligned one (-16 here) would
    # otherwise take the clip's compare-free fast path and emit that out-of-bounds address unmasked.
    with pytest.raises(ValueError) as excinfo:
        make_window(make_tensor_desc((16, 16), (16, 1), _DT("f16")), (-16, 0))
    msg = str(excinfo.value)
    assert "negative window origin" in msg and "axis 0" in msg


def test_window_allows_nonnegative_and_runtime_origin() -> None:
    # Zero/positive compile-time origins are fine, and a runtime (non-int) origin is trusted (the
    # deferred mid-tile/sliding-window case), so the negative guard must not fire on it.
    assert make_window(
        make_tensor_desc((16, 16), (16, 1), _DT("f16")), (16, 0)
    ).origin == (16, 0)
    runtime_origin = object()  # stands in for an SSA Value
    win = make_window(
        make_tensor_desc((16, 16), (16, 1), _DT("f16")), (runtime_origin, 0)
    )
    assert win.origin[0] is runtime_origin


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
