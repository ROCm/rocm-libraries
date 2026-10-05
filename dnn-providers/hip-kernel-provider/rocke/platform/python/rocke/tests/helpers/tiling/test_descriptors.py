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


# ---- N-D axis roles + rank-reducing slice (at_index / squeeze) ---------------------------------

def _batched(dt=_DT("f16")):
    """A rank-3 batched operand (batch, M=free, K=contraction), strides (M*K, K, 1)."""
    return make_tensor_desc((4, 16, 8), (128, 8, 1), dt, ("batch", "free", "contraction"))


def test_axis_roles_recorded_and_located() -> None:
    td = _batched()
    assert td.axis_roles == ("batch", "free", "contraction")
    assert td.free_axis == 1
    assert td.contraction_axis == 0 + 2  # axis 2


def test_axis_roles_reject_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown axis role"):
        make_tensor_desc((4, 16), (16, 1), _DT("f16"), ("bogus", "free"))


def test_axis_roles_reject_wrong_length() -> None:
    with pytest.raises(ValueError, match="axis_roles rank"):
        make_tensor_desc((4, 16), (16, 1), _DT("f16"), ("batch",))


def test_axis_roles_reject_two_contraction_axes() -> None:
    with pytest.raises(ValueError, match="at most one 'contraction'"):
        make_tensor_desc((8, 8), (8, 1), _DT("f16"), ("contraction", "contraction"))


def test_permute_carries_axis_roles() -> None:
    td = make_tensor_desc((16, 8), (1, 16), _DT("f16"), ("free", "contraction"))
    viewed = td.permute([1, 0])
    assert viewed.axis_roles == ("contraction", "free")


def test_at_index_pins_batch_and_reduces_rank() -> None:
    win = make_window(_batched(), (0, 0, 0))
    reduced = win.at_index(0, 3)
    assert reduced.tensor.rank == 2
    assert reduced.tensor.lengths == (16, 8)
    assert reduced.tensor.strides == (8, 1)          # strides preserved
    assert reduced.tensor.axis_roles == ("free", "contraction")
    assert reduced.origin == (0, 0)
    assert reduced.pinned == ((3, 128),)             # batch offset carried: index 3 * stride 128


def test_at_index_reduces_bounds_too() -> None:
    win = make_window(_batched(), (0, 0, 0), (4, 16, 8))
    reduced = win.at_index(0, 1)
    assert reduced.bounds == (16, 8)                 # the batch bound entry is dropped


def test_at_index_refuses_free_or_contraction_axis() -> None:
    win = make_window(_batched(), (0, 0, 0))
    with pytest.raises(ValueError, match="may only reduce a 'batch' axis"):
        win.at_index(1, 0)                           # free
    with pytest.raises(ValueError, match="may only reduce a 'batch' axis"):
        win.at_index(2, 0)                           # contraction


def test_at_index_requires_declared_roles() -> None:
    win = make_window(make_tensor_desc((4, 16, 8), (128, 8, 1), _DT("f16")), (0, 0, 0))
    with pytest.raises(ValueError, match="needs declared axis_roles"):
        win.at_index(0, 0)


def test_at_index_refuses_positioned_batch_axis() -> None:
    # A nonzero origin on the batch axis conflicts with the pinned index -- fail fast, not silent.
    win = make_window(_batched(), (2, 0, 0))
    with pytest.raises(ValueError, match="POSITIONED batch axis"):
        win.at_index(0, 1)


def test_at_index_refuses_clipped_batch_axis() -> None:
    win = make_window(_batched(), (0, 0, 0), (2, 16, 8))  # batch bound 2 != length 4
    with pytest.raises(ValueError, match="CLIPPED batch axis"):
        win.at_index(0, 1)


def test_at_index_address_matches_full_rank(  ) -> None:
    # The reduced (M, K) window must address the same element the rank-3 window would at batch=i.
    from rocke.helpers.tiling.emit import _address
    from rocke.helpers.tiling.lds_conflict import NumBuilder

    nb = NumBuilder(0)
    win3 = make_window(_batched(), (0, 0, 0))
    i, m, k = 3, 5, 6
    full = _address(nb, win3, [i, m, k])
    reduced = win3.at_index(0, i)
    got = _address(nb, reduced, [m, k])
    assert got == full == i * 128 + m * 8 + k * 1


def test_squeeze_drops_unit_batch_axis() -> None:
    td = make_tensor_desc((1, 16, 8), (128, 8, 1), _DT("f16"), ("batch", "free", "contraction"))
    reduced = td.squeeze(0)
    assert reduced.rank == 2
    assert reduced.axis_roles == ("free", "contraction")
    assert reduced.strides == (8, 1)


def test_squeeze_refuses_non_unit_axis() -> None:
    with pytest.raises(ValueError, match="needs a length-1 axis"):
        _batched().squeeze(0)                        # batch length is 4


def test_squeeze_refuses_free_axis() -> None:
    td = make_tensor_desc((1, 8), (8, 1), _DT("f16"), ("free", "contraction"))
    with pytest.raises(ValueError, match="may only reduce a 'batch' axis"):
        td.squeeze(0)


def test_assert_mma_operand_passes_rank2_free_contraction() -> None:
    make_tensor_desc((16, 8), (8, 1), _DT("f16"), ("free", "contraction")).assert_mma_operand()


def test_assert_mma_operand_rejects_undeclared_roles() -> None:
    with pytest.raises(ValueError, match="must declare axis_roles"):
        make_tensor_desc((16, 8), (8, 1), _DT("f16")).assert_mma_operand()


def test_assert_mma_operand_accepts_b_orientation() -> None:
    # B is presented (K, N) = (contraction, free) so that A @ B^T maps onto the atom's A @ B; the
    # gate is order-agnostic (checks the axis SET, not the order).
    make_tensor_desc((8, 16), (1, 8), _DT("f16"), ("contraction", "free")).assert_mma_operand()


def test_assert_mma_operand_rejects_missing_contraction() -> None:
    with pytest.raises(ValueError, match="one free \\+ one contraction"):
        make_tensor_desc((16, 16), (16, 1), _DT("f16"), ("free", "free")).assert_mma_operand()


def test_assert_mma_operand_rejects_unreduced_rank3() -> None:
    with pytest.raises(ValueError, match="Reduce batch axes"):
        _batched().assert_mma_operand()
