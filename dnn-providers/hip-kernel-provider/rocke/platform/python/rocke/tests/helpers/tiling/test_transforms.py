# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the fragment-transform solver + MMA safety (no GPU).

Covers: interleave_idx matches the reference layout tables; classify_transform labels a real
register reorder, rejects different-element transforms, and is an involution; mma_pair_k_aligned
accepts K-aligned operands and rejects a K-misaligned one with a constructive message.
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.core.ir import F16, F32
from rocke.helpers.tiling.encoding import WarpDistributionEncoding
from rocke.helpers.tiling.mma.warp_encoding import (
    a_warp_encoding,
    b_warp_encoding,
    canonical_layouts,
)
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import load_mma_traits
from rocke.helpers.tiling.transforms import (
    as_forward_map,
    classify_transform,
    describe_edge,
    interleave_idx,
    k_distribution,
    mma_pair_compatible,
    mma_pair_k_aligned,
    reorder_between,
)


def _traits(op_id: str = "mfma_f32_16x16x16f16"):
    return load_mma_traits().get(op_id)


def _swap_register_axes(
    enc: WarpDistributionEncoding, i: int, j: int
) -> WarpDistributionEncoding:
    """A register-slot reorder: swap two register (Y) axis entries. Same buckets, so still a valid
    bijection -- only the per-lane register slot order changes (an interleave)."""
    rmaj = list(enc.register_to_rh_major)
    rmin = list(enc.register_to_rh_minor)
    rmaj[i], rmaj[j] = rmaj[j], rmaj[i]
    rmin[i], rmin[j] = rmin[j], rmin[i]
    return dataclasses.replace(
        enc, register_to_rh_major=tuple(rmaj), register_to_rh_minor=tuple(rmin)
    )


def test_interleave_idx_matches_reference_tables() -> None:
    assert interleave_idx(1, 2, 8) == (0, 2, 4, 6, 1, 3, 5, 7)  # CDNA16 <1,2,8>
    assert interleave_idx(1, 4, 16) == tuple(
        (i % 4) * 4 + i // 4 for i in range(16)
    )  # <1,4,16>
    assert interleave_idx(1, 16, 32, 32)[:4] == (0, 16, 1, 17)  # RDNA3 <1,16,32>
    assert interleave_idx(1, 1, 8) == tuple(range(8))  # NOP: stride == 1
    assert interleave_idx(1, 8, 8) == tuple(range(8))  # NOP: stride == count


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
    a = a_warp_encoding(_traits())  # (M, K)
    a_bigger = a_warp_encoding(_traits(), k_iter=2)  # different fragment dims
    with pytest.raises(ValueError, match="different dimensions"):
        classify_transform(a, a_bigger)


# --- describe_edge: the five edge kinds (labels invariant except an EXPLICIT relabel) --------------
def test_describe_edge_identity() -> None:
    a = a_warp_encoding(_traits())
    kind, why = describe_edge(a, a)
    assert kind == "identity" and why


def test_describe_edge_reorder_and_cross_lane() -> None:
    # a lane-uniform register permutation is a reorder (has a cost -> new register file)
    a = a_warp_encoding(_traits(), k_iter=2)
    kind, _ = describe_edge(a, _swap_register_axes(a, 0, 2))
    assert kind == "reorder"


def test_describe_edge_reposition_is_free_and_label_invariant() -> None:
    # register -> LDS: label INVARIANT, the datum's PHYSICAL storage placement changes -- never a LABEL
    # transpose (the memref axis order is positional). tgt_dims must NOT manufacture a "(M,K)->(K,M)" why.
    a = a_warp_encoding(_traits())
    kind, why = describe_edge(
        a, None, src_dims=("M", "K"), tgt_dims=("K", "M"), to_space="lds"
    )
    assert kind == "reposition"
    assert "invariant" in why and "lds" in why
    assert (
        "transpose" not in why and "->" not in why
    )  # storage != label; no label-dims transpose


def test_describe_edge_explicit_relabel_changes_label() -> None:
    # Relabel = the ONE sanctioned label change: a FINISHED tile reused as a downstream input (axis
    # permutation + rename). NOT AB-swap (that is machine-input routing, labels invariant). EXPLICIT.
    m = {(0, 0): (0, 0), (0, 1): (1, 0), (1, 0): (0, 1), (1, 1): (1, 1)}
    kind, why = describe_edge(
        m, m, src_dims=("M", "K"), tgt_dims=("N", "K"), relabel=True
    )
    assert kind == "relabel" and "CHANGES" in why
    # a transpose relabel (axes swapped) is also a relabel, and names the swap
    mt = {k: (c[1], c[0]) for k, c in m.items()}
    kind2, why2 = describe_edge(
        m, mt, src_dims=("M", "K"), tgt_dims=("K", "M"), relabel=True
    )
    assert kind2 == "relabel" and "swapped" in why2


def test_describe_edge_declared_relabel_must_be_consistent() -> None:
    # a "relabel" that is not a consistent axis permutation of the source is rejected, not silently applied
    s = {(0, 0): (0, 0), (0, 1): (1, 0)}
    t = {(0, 0): (7, 7), (0, 1): (9, 9)}
    with pytest.raises(ValueError, match="consistent axis permutation"):
        describe_edge(s, t, relabel=True)


def test_mma_pair_k_aligned_accepts_canonical() -> None:
    tr = _traits()
    a, b = a_warp_encoding(tr), b_warp_encoding(tr)
    d = mma_pair_k_aligned(a, b, a_canon=a, b_canon=b)
    assert d.severity == "ok", d.message


def test_mma_pair_k_aligned_rejects_k_misaligned() -> None:
    # A wave tile whose K registers are reordered => A's K-distribution no longer matches B's.
    tr = _traits()
    a = a_warp_encoding(tr, k_iter=2)
    b = b_warp_encoding(tr, k_iter=2)
    a_k_scrambled = _swap_register_axes(
        a, 0, 2
    )  # reorders which K sits in which register slot
    if k_distribution(a_k_scrambled) == k_distribution(a):
        pytest.skip("swap did not perturb the K projection for this atom")
    d = mma_pair_k_aligned(a_k_scrambled, b, a_canon=a, b_canon=b)
    assert d.severity in ("warning", "error")  # the driver rejects either
    # a register-axis swap keeps the K SET per lane but changes its order -> reorder-fixable warning
    # (names transform_fragment); if it also changed the set it would be a hard "different K sets" error.
    assert "transform_fragment" in d.message or "different K sets" in d.message


def test_k_distribution_a_equals_b_for_aligned_atom() -> None:
    tr = _traits()
    assert k_distribution(a_warp_encoding(tr)) == k_distribution(b_warp_encoding(tr))


def test_mma_pair_k_aligned_accepts_a_rectangular_wave_tile() -> None:
    # A rectangular wave tile (64x32x32 -> 4 M atoms, 2 N atoms) holds more A registers than B
    # registers per lane, yet every A row meets every B column at the same instruction K: the check
    # reads that off the wave tile's canonical layouts and accepts it.
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
        _wave_descs_interleaved,
    )

    a, b, _ = _wave_descs_interleaved(4, 2, 2)
    a_canon, b_canon, _ = canonical_layouts(_traits(), (4, 2, 2))
    d = mma_pair_k_aligned(a.layout, b.layout, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == "ok", d.message


def test_mma_pair_k_aligned_rejects_k_that_depends_on_the_free_row() -> None:
    # Swapping two K labels on the odd M rows of A and the odd N rows of B keeps each operand sound
    # and keeps A and B equal slot for slot, but C[0, 1] then multiplies A's K 0 with B's K 1: every
    # A row meets every B column, so K must not depend on the row.
    tr = _traits()
    a_canon, b_canon = a_warp_encoding(tr), b_warp_encoding(tr)

    def odd_rows_swap_k01(layout):
        return {
            slot: (row, {0: 1, 1: 0}.get(k, k)) if row % 2 else (row, k)
            for slot, (row, k) in as_forward_map(layout).items()
        }

    a, b = odd_rows_swap_k01(a_canon), odd_rows_swap_k01(b_canon)
    assert as_forward_map(a) == as_forward_map(b)  # equal slot for slot
    for check in (mma_pair_k_aligned, mma_pair_compatible):
        d = check(a, b, a_canon=a_canon, b_canon=b_canon)
        assert d.severity == "error", d.message
        assert d.message == (
            "A's K labels depend on its M row: instruction K 0 holds K 0 in M 0 but K 1 in M 1; "
            "the instruction multiplies every M row at one instruction K with the same K of the "
            "other operand -- no in-register reorder of one operand fixes it"
        )


def _canonical_pair():
    """The canonical A and B layouts of one 16x16x16 f16 MFMA, and A's forward map."""
    tr = _traits()
    a_canon, b_canon = a_warp_encoding(tr), b_warp_encoding(tr)
    return a_canon, b_canon, as_forward_map(a_canon)


def test_mma_pair_k_aligned_names_a_reorder_of_a_when_one_fixes_it() -> None:
    # Swapping the K labels of lane 0's first two registers leaves every row of A holding the K
    # labels B pairs with it, out of order: an in-register reorder of A fixes it.
    a_canon, b_canon, a_map = _canonical_pair()
    swapped = dict(a_map)
    swapped[(0, 0)], swapped[(0, 1)] = a_map[(0, 1)], a_map[(0, 0)]
    d = mma_pair_k_aligned(swapped, b_canon, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == "warning"
    assert d.message == (
        "A.K != B.K at instruction K 0 (A lane 0 register 0 holds K 1, B holds K 0); every row "
        "of A holds the K labels B pairs with it -> reconcilable by an in-register reorder of A "
        "(transform_fragment it first)"
    )


def test_mma_pair_k_aligned_refuses_different_k_sets() -> None:
    # Each lane of this instruction holds a disjoint group of four K. Moving A's K labels on by one
    # group gives every lane K that B never pairs with it there.
    a_canon, b_canon, a_map = _canonical_pair()
    moved = {slot: (row, (k + 4) % 16) for slot, (row, k) in a_map.items()}
    d = mma_pair_k_aligned(moved, b_canon, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == "error"
    assert d.message == (
        "A.K and B.K hold different K sets (lane 0, M 0: A [4, 5, 6, 7] vs B [0, 1, 2, 3] at the "
        "same instruction K) -- no in-register reorder reconciles them"
    )


def test_mma_pair_k_aligned_refuses_an_operand_of_another_wave_tile() -> None:
    # A two-atom A against the one-atom canonical: the canonical is not this operand's.
    a_canon, b_canon, _ = _canonical_pair()
    a_two_atoms, _, _ = canonical_layouts(_traits(), (2, 1, 1))
    d = mma_pair_k_aligned(a_two_atoms, b_canon, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == "error"
    assert d.message == (
        "A holds 512 (lane, register) positions but its canonical layout holds 256 -- pass "
        "canonical_layouts(traits, subtiles) of this wave tile"
    )


def test_mma_pair_k_aligned_refuses_replicas_that_disagree() -> None:
    # This wave32 instruction holds every A value in two lanes (0 and 16 hold M 0). Giving lane 16's
    # copy of instruction K 0 another K label makes the replicas disagree.
    tr = _traits("wmma_f32_16x16x16_f16_w32")
    a_canon, b_canon = a_warp_encoding(tr), b_warp_encoding(tr)
    a_map = as_forward_map(a_canon)
    assert a_map[(0, 0)] == a_map[(16, 0)] == (0, 0)
    replicas = dict(a_map)
    replicas[(16, 0)] = (0, 1)
    d = mma_pair_k_aligned(replicas, b_canon, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == "error"
    assert d.message == (
        "A's replicas of M 0 disagree on K: instruction K 0 holds K 0 in one and K 1 in another; "
        "every replica must hold the same values -- no in-register reorder of one operand fixes it"
    )


def test_interleaved_style_operand_desc_is_distinct() -> None:
    # The interleaved layout is now a first-class LayoutStyle (the old broken a_desc(interleaved=True)
    # register-reorder guard is gone). Its operand descriptors differ from the canonical ones (a
    # register reorder), while the canonical style's operand desc coincides with the canonical desc.
    # The accumulator-vs-machine C-oracle is checked at plan construction (so building succeeds only if
    # the style's C is machine-consistent).
    from rocke.helpers.tiling.mma import TileMma, Tiling
    from rocke.helpers.tiling.mma.styles import InterleavedStyle

    canon = TileMma(
        (32, 32, 32),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    inter = TileMma(
        (32, 32, 32),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
        style=InterleavedStyle(),
    )
    # canonical and interleaved styles produce DIFFERENT operand layouts over the same machine. canon is a
    # canonical-style object, so its own accessor IS the canonical reference (no helper needed here).
    assert inter.a_desc.layout != canon.a_desc.layout
    assert inter.b_desc.layout != canon.b_desc.layout


# --- reorder_between: derive the in-register reorder that bridges coalesced -> requested order ---
# All SYNTHETIC (no CRC): the interleave params are OUTPUTS of the classifier, proving genericity.


def _perm_maps(perm, nlanes=2):
    """A source frame + its within-lane register reorder by ``perm`` (same per-lane data, reordered)."""
    src = {(l, r): (r, l) for l in range(nlanes) for r in range(len(perm))}
    tgt = {(l, p): src[(l, perm[p])] for l in range(nlanes) for p in range(len(perm))}
    return src, tgt


def test_reorder_between_identity_is_none() -> None:
    src, _ = _perm_maps((0, 1, 2, 3))
    assert reorder_between(src, src, pack=2) is None  # no reorder -> no panel


def test_reorder_between_dword_vs_subdword_from_pack() -> None:
    # SAME permutation, different dtype pack: pack=1 (f32) is a dword renumber; pack=2 (f16) splits a
    # dword -> a real v_perm repack. The tier is DERIVED from pack, not hardcoded per kernel.
    src, tgt = _perm_maps(interleave_idx(1, 2, 4))  # (0,2,1,3)
    dword = reorder_between(src, tgt, pack=1)
    assert (
        dword is not None
        and dword.tier == "reorder (dword)"
        and dword.vperm_per_lane == 0
    )
    assert "interleave_idx(1, 2, 4)" in dword.label
    sub = reorder_between(src, tgt, pack=2)
    assert sub.tier == "reorder (sub-dword, 2x)" and sub.vperm_per_lane > 0
    # genericity: a DIFFERENT permutation yields a DIFFERENT derived label (nothing hardcoded)
    s2, t2 = _perm_maps(interleave_idx(1, 4, 8))
    other = reorder_between(s2, t2, pack=1).label
    assert "interleave_idx(" in other and other != dword.label


def test_reorder_between_cross_lane_when_not_lane_uniform() -> None:
    # Same per-lane SET on both lanes (guard passes) but lane 0 is identity while lane 1 is swapped ->
    # not lane-uniform -> cross_lane (its own arrow, NOT the two-panel within-lane model).
    src = {(0, 0): (0, 0), (0, 1): (1, 0), (1, 0): (0, 1), (1, 1): (1, 1)}
    tgt = {(0, 0): (0, 0), (0, 1): (1, 0), (1, 0): (1, 1), (1, 1): (0, 1)}
    rp = reorder_between(src, tgt, pack=1)
    assert rp is not None and rp.tier == "cross_lane" and rp.permutation is None


def test_reorder_between_raises_on_different_per_lane_data() -> None:
    # An element that changes lane means the two frames are the WRONG pair -> loud fail, never silently
    # reinterpreted as within-lane.
    src = {(0, 0): (0, 0), (1, 0): (1, 0)}
    tgt = {(0, 0): (1, 0), (1, 0): (0, 0)}
    with pytest.raises(ValueError, match="same per-lane data"):
        reorder_between(src, tgt, pack=1)


def test_observer_result_types_surfaced_at_package_root() -> None:
    # #34: the observers' return dataclasses are importable from the package root (so an exported
    # observer function's return type is reachable without diving into transforms/_core).
    import rocke.helpers.tiling as tiling
    from rocke.helpers.tiling import Diagnostic, ReorderPlan, TransformPlan

    for name in ("Diagnostic", "TransformPlan", "ReorderPlan"):
        assert name in tiling.__all__
    assert all(isinstance(t, type) for t in (Diagnostic, TransformPlan, ReorderPlan))


# ---- #47: a REPLICATED / duplicated-coordinate layout is REPORTED, not silently first-slot-classified ----


def test_classify_rejects_a_replicated_layout() -> None:
    # An element held at MULTIPLE (lane,reg) slots (e.g. gfx11 WMMA duplicates the operand across lane
    # halves) cannot be paired source<->target by coordinate. The classifier must report it, not let the
    # first slot silently win (the old `setdefault` mis-classified the reorder). Classify within one copy.
    replicated = {(0, 0): (7,), (1, 0): (7,)}  # element (7,) held by BOTH lanes
    with pytest.raises(ValueError, match="REPLICATED"):
        classify_transform(replicated, replicated)


def test_classify_rejects_same_set_different_multiplicity() -> None:
    # Same element SET, different MULTIPLICITY -- the old `set()` check passed this; the multiset check
    # rejects it as different elements (a same-size gather/broadcast preserves count but not the set).
    src = {(0, 0): (0,), (1, 0): (0,), (2, 0): (1,)}  # multiset {(0,): 2, (1,): 1}
    tgt = {(0, 0): (0,), (1, 0): (1,), (2, 0): (1,)}  # multiset {(0,): 1, (1,): 2}
    with pytest.raises(ValueError, match="different elements"):
        classify_transform(src, tgt)
