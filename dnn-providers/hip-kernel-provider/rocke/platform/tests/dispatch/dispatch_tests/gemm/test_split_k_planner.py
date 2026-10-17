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
    # No fitted model: callers keep their previous heuristic.
    assert split_k_cost_model("gfx942") is None


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


def test_candidates_are_powers_of_two_and_unsplit():
    plan = _plan(TINY_OUT_LONG_K, SMALL_TILE)
    degrees = {d for d, _, _ in plan.candidates}
    assert 1 in degrees
    assert all(d == 1 or (d & (d - 1)) == 0 for d in degrees)


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
