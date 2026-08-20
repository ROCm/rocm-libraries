# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the fragment-transform solver + MMA safety (no GPU).

Covers: interleave_idx matches the reference layout tables; classify_transform labels a real
register reorder, rejects different-element transforms, and is an involution; validate_operands
accepts K-aligned operands and rejects a K-misaligned one with a constructive message.
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.helpers.tiling.encoding import WarpDistributionEncoding
from rocke.helpers.tiling.mma.warp_encoding import a_warp_encoding, b_warp_encoding
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import load_mma_traits
from rocke.helpers.tiling.transforms import (
    classify_transform,
    interleave_idx,
    k_distribution,
    validate_operands,
)


def _traits(op_id: str = "mfma_f32_16x16x16f16"):
    return load_mma_traits().get(op_id)


def _swap_register_axes(enc: WarpDistributionEncoding, i: int, j: int) -> WarpDistributionEncoding:
    """A register-slot reorder: swap two register (Y) axis entries. Same buckets, so still a valid
    bijection -- only the per-lane register slot order changes (an interleave)."""
    rmaj = list(enc.register_to_rh_major)
    rmin = list(enc.register_to_rh_minor)
    rmaj[i], rmaj[j] = rmaj[j], rmaj[i]
    rmin[i], rmin[j] = rmin[j], rmin[i]
    return dataclasses.replace(enc, register_to_rh_major=tuple(rmaj), register_to_rh_minor=tuple(rmin))


def test_interleave_idx_matches_reference_tables() -> None:
    assert interleave_idx(1, 2, 8) == (0, 2, 4, 6, 1, 3, 5, 7)          # CDNA16 <1,2,8>
    assert interleave_idx(1, 4, 16) == tuple((i % 4) * 4 + i // 4 for i in range(16))  # <1,4,16>
    assert interleave_idx(1, 16, 32, 32)[:4] == (0, 16, 1, 17)         # RDNA3 <1,16,32>
    assert interleave_idx(1, 1, 8) == tuple(range(8))                  # NOP: stride == 1
    assert interleave_idx(1, 8, 8) == tuple(range(8))                  # NOP: stride == count


def test_interleave_idx_gather_gt_1_deferred() -> None:
    with pytest.raises(NotImplementedError, match="gather"):
        interleave_idx(4, 8, 16)


def test_interleave_idx_bad_params_reject() -> None:
    with pytest.raises(ValueError, match="count % stride"):
        interleave_idx(1, 3, 8)


def test_classify_identity_is_reorder_identity() -> None:
    a = a_warp_encoding(_traits())
    plan = classify_transform(a, a)
    assert plan.tier == "reorder"
    assert plan.permutation == tuple(range(len(plan.permutation)))


def test_classify_register_reorder_is_lane_uniform_reorder() -> None:
    # A wave tile with k_iter=2 has >1 register axis; swapping them is the SOA<->AOS interleave.
    a = a_warp_encoding(_traits(), k_iter=2)
    interleaved = _swap_register_axes(a, 0, 2)
    plan = classify_transform(a, interleaved)
    assert plan.tier == "reorder"
    # a genuine permutation (not identity) and a bijection over the register slots
    nreg = RegisterMapper(a).num_vector_items
    assert plan.permutation != tuple(range(nreg))
    assert sorted(plan.permutation) == list(range(nreg))


def test_classify_reorder_is_involution() -> None:
    a = a_warp_encoding(_traits(), k_iter=2)
    interleaved = _swap_register_axes(a, 0, 2)
    fwd = classify_transform(a, interleaved).permutation
    back = classify_transform(interleaved, a).permutation
    # applying fwd then back returns identity
    composed = tuple(back[fwd[i]] for i in range(len(fwd)))
    assert composed == tuple(range(len(fwd)))


def test_classify_different_elements_rejected() -> None:
    a = a_warp_encoding(_traits())          # (M, K)
    a_bigger = a_warp_encoding(_traits(), k_iter=2)  # different fragment dims
    with pytest.raises(ValueError, match="different dimensions"):
        classify_transform(a, a_bigger)


def test_validate_operands_accepts_canonical() -> None:
    tr = _traits()
    a, b = a_warp_encoding(tr), b_warp_encoding(tr)
    ok, why = validate_operands(a, b)
    assert ok, why


def test_validate_operands_rejects_k_misaligned_with_message() -> None:
    # A wave tile whose K registers are reordered => A's K-distribution no longer matches B's.
    tr = _traits()
    a = a_warp_encoding(tr, k_iter=2)
    b = b_warp_encoding(tr, k_iter=2)
    a_k_scrambled = _swap_register_axes(a, 0, 2)  # reorders which K sits in which register slot
    if k_distribution(a_k_scrambled) == k_distribution(a):
        pytest.skip("swap did not perturb the K projection for this atom")
    ok, why = validate_operands(a_k_scrambled, b)
    assert not ok
    assert "not K-aligned" in why and "transform_fragment" in why


def test_k_distribution_a_equals_b_for_aligned_atom() -> None:
    tr = _traits()
    assert k_distribution(a_warp_encoding(tr)) == k_distribution(b_warp_encoding(tr))


def test_validate_operands_accepts_rectangular_per_atom() -> None:
    # A rectangular wave tile (64x32x32 -> m_sub=4, n_sub=2) tiles more M-atoms in A than N-atoms in
    # B, so the WHOLE-fragment K-lists differ in length (A repeats atom-K 4x, B 2x). The per-atom gate
    # must accept it (every issued 16x16x16 atom pairs the same K); the whole-fragment default rejects.
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import _wave_descs_interleaved

    a, b, _ = _wave_descs_interleaved(4, 2, 2)
    ok_atom, why = validate_operands(a.layout, b.layout, a_free_atoms=4, b_free_atoms=2)
    assert ok_atom, why
    ok_whole, _ = validate_operands(a.layout, b.layout)  # default 1,1 -> whole-fragment compare
    assert not ok_whole


def test_validate_operands_per_atom_still_rejects_k_mismatch() -> None:
    # The per-atom relaxation must NOT mask a genuine K divergence. A's atom-K is (0..7); pairing it
    # against a B chunked so its atom-K spans two K-groups (0..7,0..7) yields unequal atom signatures
    # -> the gate rejects, proving the relaxation only forgives the free-dim repeat, never the K.
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import _wave_descs_interleaved

    a, b, _ = _wave_descs_interleaved(4, 2, 2)
    ok, why = validate_operands(a.layout, b.layout, a_free_atoms=4, b_free_atoms=1)
    assert not ok
    assert "not K-aligned" in why


def test_a_desc_interleaved_is_broken() -> None:
    # a_desc(interleaved=True) does NOT produce a proper interleaved layout -- it is broken and raises.
    # Real interleaved layouts are custom static tile distributions (make_tile_desc).
    from rocke.helpers.tiling.mma import TileMma, Tiling
    mma = TileMma((16, 16, 32), a="f16", b="f16", c="f32", target="gfx90a",
                  tiling=Tiling(atom_shape=(16, 16, 16)))
    with pytest.raises(RuntimeError, match="BROKEN"):
        mma.a_desc(interleaved=True)


def test_b_desc_interleaved_is_broken() -> None:
    from rocke.helpers.tiling.mma import TileMma, Tiling
    mma = TileMma((16, 16, 32), a="f16", b="f16", c="f32", target="gfx90a",
                  tiling=Tiling(atom_shape=(16, 16, 16)))
    with pytest.raises(RuntimeError, match="BROKEN"):
        mma.b_desc(interleaved=True)
