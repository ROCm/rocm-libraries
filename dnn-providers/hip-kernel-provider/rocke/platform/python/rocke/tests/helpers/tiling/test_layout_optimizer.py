# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the layout optimizer -- ``evaluate_transform`` / ``optimize_layout`` / ``enumerate_stripings``.

The optimizer discovers each transform (source->target) with rocKE's own solver and prices it against the cost
ladder; it never consults a table. These lock the ladder ordering + the validity gates on real layouts."""

from rocke.helpers.tiling.layouts.tile_distribution import make_tile_desc
from rocke.helpers.tiling.mma.mma_operation import TileMma, Tiling
from rocke.helpers.tiling.transforms import TransformPlan, as_forward_map
from rocke.helpers.tiling.layout_optimizer import (
    _price, enumerate_stripings, evaluate_transform, optimize_layout, recommend)


def _mma():
    return TileMma((64, 64, 32), a="f16", b="f16", c="f32", target="gfx90a",
                   tiling=Tiling(atom_shape=(16, 16, 16)))


def _interleaved_free_a():
    # a free-dim-contiguous interleaved A operand (M=64, K=32; per-lane 4x8)
    return make_tile_desc(shape=[64, 32], thread_tile=[4, 8], thread_dist=[16, 4], thread_order=[1, 0],
                          block_repeat=[1, 1], wave_dist=[1, 1], wave_size=64).layout


def _k_contig_twin(a_layout):
    # same OWNERSHIP as a_layout, K-contiguous register order (an intra-lane reorder away)
    fwd = as_forward_map(a_layout)
    kc = {}
    for lane in sorted({l for l, _ in fwd}):
        cs = [fwd[(l, r)] for (l, r) in fwd if l == lane]
        ms, ks = sorted({m for m, _k in cs}), sorted({k for _m, k in cs})
        for (m, k) in cs:
            kc[(lane, ms.index(m) * len(ks) + ks.index(k))] = (m, k)
    return kc


def test_identity_is_free():
    a = _interleaved_free_a()
    v = evaluate_transform(a, a)
    assert v.edge.kind == "identity" and v.edge.cost == 0.0 and v.works


def test_relabel_is_free():
    # a pure axis-swap at register identity (col<->row / M<->N) must be recognised as a free relabel, NOT a move
    a = as_forward_map(_interleaved_free_a())
    swapped = {k: (c[1], c[0]) for k, c in a.items()}
    v = evaluate_transform(a, swapped)
    assert v.edge.kind == "relabel" and v.edge.cost == 0.0


def test_cross_lane_when_ownership_differs():
    # interleaved free vs the CANONICAL atom layout have different lane ownership -> cross-lane (not a relabel)
    mma = _mma()
    v = evaluate_transform(_interleaved_free_a(), mma.a_layout, canon=mma.a_layout)
    assert v.edge.kind == "cross_lane" and v.works and v.edge.cost > 0.0


def test_lds_reposition_beats_register_cross_lane():
    mma = _mma()
    reg = evaluate_transform(_interleaved_free_a(), mma.a_layout, canon=mma.a_layout, through_lds=False)
    lds = evaluate_transform(_interleaved_free_a(), mma.a_layout, canon=mma.a_layout, through_lds=True)
    assert reg.edge.kind == "cross_lane"
    assert lds.edge.kind == "reposition_lds"
    assert lds.edge.cost < reg.edge.cost           # LDS bulk beats per-element cross-lane


def test_lds_measured_conflicts_raise_cost():
    mma = _mma()
    lb = evaluate_transform(_interleaved_free_a(), mma.a_layout, canon=mma.a_layout, through_lds=True)
    meas = evaluate_transform(_interleaved_free_a(), mma.a_layout, canon=mma.a_layout, through_lds=True,
                              lds_conflict_cost=(2.0, 5.0))
    assert meas.edge.cost == lb.edge.cost + 7.0     # lower bound + measured store+read bank conflicts


def test_both_movers_scale_with_registers():
    xl = TransformPlan("cross_lane", None, "element moves lane")
    reg32 = _price(xl, dtype_bits=16, through_lds=False, tile_regs=32, lds_conflict_cost=None)
    reg128 = _price(xl, dtype_bits=16, through_lds=False, tile_regs=128, lds_conflict_cost=None)
    lds32 = _price(xl, dtype_bits=16, through_lds=True, tile_regs=32, lds_conflict_cost=None)
    lds128 = _price(xl, dtype_bits=16, through_lds=True, tile_regs=128, lds_conflict_cost=None)
    assert reg128.cost > reg32.cost and lds128.cost > lds32.cost


def test_pairwise_k_match_gate():
    # per-operand soundness alone can pass while the PAIR is K-mismatched -> k_partner must catch it
    mma = _mma()
    a, b = mma.a_layout, mma.b_layout
    ok = evaluate_transform(a, a, canon=a, k_partner=b)
    assert ok.k_match == "ok" and ok.works
    b_bad = {k: (c[0], 0) for k, c in as_forward_map(b).items()}      # scramble B's K -> mismatch
    bad = evaluate_transform(a, a, canon=a, k_partner=b_bad)
    assert bad.k_match != "ok" and not bad.works


def test_optimize_ranks_cheapest_valid_first():
    mma = _mma()
    a = _interleaved_free_a()
    ranked = optimize_layout(a, {"twin": _k_contig_twin(a), "canonical": mma.a_layout}, canon=mma.a_layout)
    assert ranked[0][0] == "twin"                    # intra-lane (cost 0) beats canonical (cross-lane)
    assert ranked[0][1].edge.cost <= ranked[1][1].edge.cost


def test_enumerate_stripings_sweeps_valid_candidates():
    cands = enumerate_stripings([64, 32], 64)
    assert len(cands) >= 4                            # multiple lane stripings of the tile
    src = next(iter(cands.values()))
    ranked = optimize_layout(src, cands)
    assert ranked[0][1].edge.cost == 0.0             # source reaches itself for free (identity)


def test_cross_lane_scales_with_ops_per_register():
    xl = TransformPlan("cross_lane", None, "moves lane")
    one = _price(xl, dtype_bits=16, through_lds=False, tile_regs=64, lds_conflict_cost=None,
                 cross_lane_ops_per_reg=1.0)
    four = _price(xl, dtype_bits=16, through_lds=False, tile_regs=64, lds_conflict_cost=None,
                  cross_lane_ops_per_reg=4.0)
    assert four.cost > one.cost                       # flip&zip (more ops/reg) is dearer than a permute
    assert one.empirical and four.empirical


def test_recommend_decides_when_deterministic_wins():
    mma = _mma()
    a = _interleaved_free_a()
    ranked = optimize_layout(a, {"twin": _k_contig_twin(a), "canonical": mma.a_layout}, canon=mma.a_layout)
    status, picks, contenders = recommend(ranked)
    assert status == "decided" and picks[0][0] == "twin" and contenders == ()


def test_recommend_gives_lds_and_crosslane_equal_shots():
    # a cross-lane re-ownership WITH lds available -> both paths are contenders; neither falls out on cost
    mma = _mma()
    a = _interleaved_free_a()
    ranked = optimize_layout(a, {"canonical": mma.a_layout}, canon=mma.a_layout, through_lds=True)
    status, picks, contenders = recommend(ranked)
    assert status == "measure"
    assert {e.kind for e in contenders} == {"cross_lane", "reposition_lds"}
    assert all(e.empirical for e in contenders)
