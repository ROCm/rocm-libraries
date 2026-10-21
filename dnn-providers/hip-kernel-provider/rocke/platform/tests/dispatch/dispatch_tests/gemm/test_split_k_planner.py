# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The split-K planner: candidate limits, reduction caps and the cost regimes.

CPU-only. Values the fit owns (exact degrees on a given shape) are not pinned;
what is asserted is the behaviour the model is built to have -- the limits it
must never cross, and the direction it moves in each physical regime.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from rocke.helpers.split_k_planner import (
    MAX_GRID_DIM_Z,
    SplitKKernel,
    SplitKProblem,
    SplitKReduction,
    predict_split_k_cost,
    select_split_k_plan,
    split_k_cost_model,
)

GFX950 = split_k_cost_model("gfx950")
ATOMIC16 = SplitKReduction("atomic", "bf16", max_split_k=128)
WORKSPACE = SplitKReduction("workspace", "fp32", workspace=True)

# A tiny output (one or two tiles) over a very long reduction: latency-bound.
TINY_OUT_LONG_K = SplitKProblem(m=24, n=27, k=42 * 240 * 320)
# A wide dense GEMM over a short reduction: hundreds of tiles, compute-bound.
DENSE_SHORT_K = SplitKProblem(m=1024, n=2048, k=6272)
SMALL_TILE = SplitKKernel(32, 16, 64, waves_per_block=2)
BIG_TILE = SplitKKernel(128, 128, 64, waves_per_block=4)


def _plan(problem, kernel, reductions=(WORKSPACE,), **kw):
    return select_split_k_plan(problem, kernel, reductions, GFX950, **kw)


def test_cost_model_lookup():
    assert GFX950 is not None
    assert split_k_cost_model("gfx950:sramecc+:xnack-") is GFX950
    # No model: callers keep their previous heuristic. gfx942 is unfitted.
    assert split_k_cost_model("gfx90a") is None
    assert split_k_cost_model("gfx90a", num_cus=104) is None
    assert split_k_cost_model("gfx942") is None
    assert split_k_cost_model("gfx942:sramecc+:xnack-", num_cus=228) is None


@pytest.mark.parametrize(
    "problem,kernel",
    [
        (TINY_OUT_LONG_K, SMALL_TILE),
        (SplitKProblem(m=24, n=27, k=1_000_003), SMALL_TILE),
        (SplitKProblem(m=256, n=576, k=25_088), BIG_TILE),
        (SplitKProblem(m=1280, n=17280, k=12800), BIG_TILE),
    ],
)
def test_off_ladder_degree_must_beat_the_ladder_by_the_margin(problem, kernel):
    exact = dataclasses.replace(GFX950, ladder_margin=0.0)
    best = select_split_k_plan(problem, kernel, (WORKSPACE,), exact)
    plan = _plan(problem, kernel)
    ladder = min((t, d) for d, _, t in plan.candidates if d & (d - 1) == 0)
    if plan.split_k & (plan.split_k - 1):
        assert plan.split_k == best.split_k
        assert ladder[0] > best.predicted_cost * (1 + GFX950.ladder_margin)
    else:
        assert plan.predicted_cost == ladder[0]
        assert plan.predicted_cost <= best.predicted_cost * (1 + GFX950.ladder_margin)


def test_grouped_launch_keeps_the_group_stride_xcd_aligned():
    # Depthwise-like: one tile per group, groups sharing gridDim.z. Off the
    # power-of-two ladder, every degree keeps tiles_per_group * split_k a
    # multiple of the 8 XCDs; the ladder itself stays whole.
    def off_ladder(plan):
        return {d for d in _degrees(plan) if d & (d - 1)}

    grouped = SplitKProblem(m=16, n=144, k=201_600, batch=12)
    plan = _plan(grouped, SplitKKernel(32, 256, 16, 2))
    assert {2, 4} <= _degrees(plan)
    assert off_ladder(plan) and all(d % 8 == 0 for d in off_ladder(plan))
    # Two tiles per group halve the step; the batch on another axis lifts it.
    two_tiles = SplitKProblem(m=64, n=144, k=201_600, batch=12)
    plan = _plan(two_tiles, SplitKKernel(32, 256, 16, 2))
    assert 12 in _degrees(plan)
    assert all(d % 4 == 0 for d in off_ladder(plan))
    off_z = dataclasses.replace(grouped, z_batch=1)
    assert 7 in _degrees(_plan(off_z, SplitKKernel(32, 256, 16, 2)))
    # Ungrouped: no alignment.
    assert 7 in _degrees(_plan(TINY_OUT_LONG_K, SMALL_TILE))


def test_grouped_pool_contains_the_whole_ladder_under_a_tight_cap():
    # Only 5 K iterations: no multiple of 8 fits; the ladder still does.
    grouped = SplitKProblem(m=16, n=144, k=16 * 5, batch=12)
    plan = _plan(grouped, SplitKKernel(32, 256, 16, 2), allow_unsplit=False)
    assert _degrees(plan) == {2, 4}


def test_device_geometry_override():
    assert (GFX950.num_cus, GFX950.num_xcds) == (256, 8)
    other = split_k_cost_model("gfx950:sramecc+:xnack-", num_cus=228, num_xcds=6)
    assert other == dataclasses.replace(GFX950, num_cus=228, num_xcds=6)
    with pytest.raises(ValueError, match="num_cus"):
        split_k_cost_model("gfx950", num_cus=0)
    with pytest.raises(ValueError, match="num_xcds"):
        split_k_cost_model("gfx950", num_xcds=0)


@pytest.mark.parametrize(
    "field,value", [("m", 0), ("n", 0), ("k", 0), ("batch", 0), ("operand_bytes", 0)]
)
def test_problem_rejects_non_positive_extents(field, value):
    kw = {"m": 1, "n": 1, "k": 1}
    kw[field] = value
    with pytest.raises(ValueError, match=f"SplitKProblem.{field}"):
        SplitKProblem(**kw)


def test_kernel_rejects_non_positive_geometry():
    with pytest.raises(ValueError, match="SplitKKernel.waves_per_block"):
        SplitKKernel(64, 64, 64, waves_per_block=0)


def test_predict_rejects_bad_degree_and_missing_reduction():
    with pytest.raises(ValueError, match="split_k must be >= 1"):
        predict_split_k_cost(TINY_OUT_LONG_K, SMALL_TILE, 0, None, GFX950)
    with pytest.raises(ValueError, match="needs a reduction"):
        predict_split_k_cost(TINY_OUT_LONG_K, SMALL_TILE, 4, None, GFX950)


def test_latency_bound_problem_splits_deep():
    # One or two output tiles cannot occupy 256 CUs; the long reduction is
    # what there is to parallelise, so the planner splits it far past the
    # degree a two-blocks-per-CU fill formula would stop at.
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE)
    assert plan.split_k >= 512
    assert plan.reduction is WORKSPACE


def test_compute_bound_problem_barely_splits():
    # Hundreds of big tiles already saturate the MFMA pipes; splitting only
    # adds rounds and reduction traffic.
    plan = _plan(DENSE_SHORT_K, BIG_TILE)
    assert plan.split_k <= 4


def test_more_output_tiles_never_raise_the_degree():
    degrees = [
        _plan(SplitKProblem(m=128, n=n, k=1 << 20), SplitKKernel(64, 64, 64, 4)).split_k
        for n in (64, 512, 4096, 32768)
    ]
    assert degrees == sorted(degrees, reverse=True)


def test_longer_reduction_never_lowers_the_degree():
    degrees = [
        _plan(SplitKProblem(m=128, n=128, k=k), SplitKKernel(64, 64, 64, 4)).split_k
        for k in (1 << 12, 1 << 15, 1 << 18, 1 << 21)
    ]
    assert degrees == sorted(degrees)


def test_reduction_cap_is_respected():
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE, reductions=(ATOMIC16,))
    assert plan.split_k <= ATOMIC16.max_split_k
    for degree, name, _ in plan.candidates:
        assert name == "none" or degree <= ATOMIC16.max_split_k


def test_uncapped_reduction_wins_past_the_cap():
    # With both on offer, a problem that wants more than the 16-bit cap gets
    # it through the reduction that has no cap.
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE, reductions=(ATOMIC16, WORKSPACE))
    assert plan.split_k > ATOMIC16.max_split_k
    assert plan.reduction is WORKSPACE


def test_degree_never_exceeds_k_iterations():
    problem = SplitKProblem(m=16, n=16, k=40 * 64)  # 40 K iterations at tile_k 64
    plan = _plan(problem, SplitKKernel(16, 16, 64, 1))
    assert plan.split_k <= 40
    assert all(d <= 40 for d, _, _ in plan.candidates)


def test_grid_z_budget_is_shared_with_the_batch():
    problem = SplitKProblem(m=1, n=9, k=1 << 24, batch=1024)
    plan = _plan(problem, SplitKKernel(16, 16, 32, 1))
    assert plan.split_k * problem.batch <= MAX_GRID_DIM_Z
    assert all(d * problem.batch <= MAX_GRID_DIM_Z for d, _, _ in plan.candidates)


def test_max_split_k_argument_caps_every_reduction():
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE, max_split_k=16)
    assert plan.split_k <= 16


def _degrees(plan):
    return {d for d, _, _ in plan.candidates}


def test_candidates_include_unsplit_and_every_power_of_two():
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE)
    degrees = _degrees(plan)
    assert 1 in degrees
    assert {1 << i for i in range(1, 11)} <= degrees


def test_candidates_include_the_cu_count_divisors_and_multiples():
    cu304 = split_k_cost_model("gfx950", num_cus=304)
    plan = select_split_k_plan(TINY_OUT_LONG_K, SMALL_TILE, (WORKSPACE,), cu304)
    assert {19, 38, 76, 152, 304, 608, 912} <= _degrees(plan)


def test_candidates_are_not_every_degree():
    # A short reduction: past sqrt(k_iters) most degrees repeat an iteration
    # count, and those (when they also fill no round) are left out.
    problem = SplitKProblem(m=128, n=128, k=64 * 900)
    plan = _plan(problem, BIG_TILE)
    assert 600 not in _degrees(plan)
    assert len(_degrees(plan)) < 900


@pytest.mark.parametrize(
    "problem,kernel",
    [
        (TINY_OUT_LONG_K, SMALL_TILE),
        (SplitKProblem(m=24, n=27, k=1_000_003), SMALL_TILE),
        (SplitKProblem(m=256, n=576, k=25_088), BIG_TILE),
        (SplitKProblem(m=96, n=1728, k=42 * 60 * 80, batch=3), SMALL_TILE),
        (DENSE_SHORT_K, BIG_TILE),
    ],
)
# CU counts off the power-of-two ladder, and another XCD count, exercise the
# candidate rules beyond the ladder.
@pytest.mark.parametrize(
    "geometry", [{}, {"num_cus": 304}, {"num_cus": 228, "num_xcds": 6}]
)
def test_candidates_lose_nothing_against_every_degree(problem, kernel, geometry):
    # The pool only drops degrees the model can never prefer, so planning over
    # it matches planning over every integer up to the cap -- on a grouped
    # launch, every XCD-aligned one and the ladder. (No ladder margin: it trades cost on purpose.)
    cost = dataclasses.replace(
        split_k_cost_model("gfx950", **geometry), ladder_margin=0.0
    )
    plan = select_split_k_plan(problem, kernel, (WORKSPACE, ATOMIC16), cost)
    cap = max(_degrees(plan))
    step = 1
    if problem.batch > 1:
        group_tiles = math.ceil(problem.m / kernel.tile_m) * math.ceil(
            problem.n / kernel.tile_n
        )
        step = cost.num_xcds // math.gcd(group_tiles, cost.num_xcds)
    best = min(
        (predict_split_k_cost(problem, kernel, s, r, cost), s)
        for r in (WORKSPACE, ATOMIC16)
        for s in range(2, min(cap, r.max_split_k or cap) + 1)
        if s % step == 0 or s & (s - 1) == 0
    )
    unsplit = predict_split_k_cost(problem, kernel, 1, None, cost)
    assert plan.predicted_cost <= min(best[0], unsplit) * (1 + 1e-12)


def test_unsplit_can_be_excluded():
    plan = _plan(DENSE_SHORT_K, BIG_TILE, allow_unsplit=False)
    assert plan.split_k >= 2
    assert all(d >= 2 for d, _, _ in plan.candidates)


def test_no_admissible_candidate_raises():
    with pytest.raises(ValueError, match="no admissible split-K candidate"):
        _plan(DENSE_SHORT_K, BIG_TILE, reductions=(), allow_unsplit=False)


def test_time_unit_is_one_k_step_latency():
    # One CTA alone on the device, with throughput far above its demand, pays
    # exactly one latency unit per K iteration: the unit the constants use.
    fast = dataclasses.replace(
        GFX950,
        flops_per_latency=1e18,
        load_bytes_per_latency=1e18,
        cta_cost=0.0,
    )
    problem = SplitKProblem(m=16, n=16, k=64 * 37)
    t = predict_split_k_cost(problem, SplitKKernel(16, 16, 64, 1), 1, None, fast)
    assert math.isclose(t, 37.0, rel_tol=1e-9)


def test_z_batch_sets_the_grid_z_budget_independently_of_the_batch():
    # A kernel that puts the batch on another grid axis does not divide the
    # gridDim.z budget by it.
    on_z = SplitKProblem(m=1, n=9, k=1 << 24, batch=1024)
    off_z = SplitKProblem(m=1, n=9, k=1 << 24, batch=1024, z_batch=1)
    kern = SplitKKernel(16, 16, 32, 1)
    cap_on = max(d for d, _, _ in _plan(on_z, kern).candidates)
    cap_off = max(d for d, _, _ in _plan(off_z, kern).candidates)
    assert cap_on * 1024 <= MAX_GRID_DIM_Z < cap_off * 1024
    with pytest.raises(ValueError, match="z_batch"):
        SplitKProblem(m=1, n=1, k=1, z_batch=0)


def test_never_beyond_the_validated_range():
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE)
    assert plan.split_k <= GFX950.max_split_k
    assert all(d <= GFX950.max_split_k for d, _, _ in plan.candidates)
    narrow = dataclasses.replace(GFX950, max_split_k=64)
    assert (
        select_split_k_plan(TINY_OUT_LONG_K, SMALL_TILE, (WORKSPACE,), narrow).split_k
        <= 64
    )
