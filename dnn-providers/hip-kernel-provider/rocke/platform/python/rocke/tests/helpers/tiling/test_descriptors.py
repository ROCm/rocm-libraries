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
    # A col-major operand stored (N, K) presented as logical (K, N): a pure view, same dtype.
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
