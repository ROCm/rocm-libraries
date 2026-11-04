# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The conv split-K resolver: one answer for dispatch, builders and benchmark.

CPU-only. Covers the conv-specific limits (reduction availability, the 16-bit
atomic cap, merged groups, gridDim.z), the override and explicit paths, and
that an arch without a fitted cost model resolves exactly as the previous
CK-formula rule did.
"""

from __future__ import annotations

import itertools

import pytest
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import (
    _wg_K,
    _wg_M,
    _wg_N,
    wgrad_atomic_epilogue_available,
)
from kernels.common.conv_implicit_gemm_wgrad_two_stage import wgrad_workspace_nbytes
from kernels.common.conv_split_k import (
    ATOMIC_16BIT_MAX_SPLIT_K,
    MAX_WGRAD_WS_BYTES,
    parse_split_k_override,
    resolve_conv_split_k,
)
from rocke.helpers.split_k import select_split_k_wgrad
from rocke.helpers.split_k_planner import MAX_GRID_DIM_Z

# Long reduction into a tiny dW, odd row (no packed atomic): a stem conv, the
# class a 128-degree ceiling under-splits.
STEM = ConvProblem(N=32, Hi=512, Wi=512, C=3, K=32, Y=3, X=3, sH=2, sW=2, pH=1, pW=1)
# Even dW row, wide channels: the packed atomic is buildable.
DENSE = ConvProblem(N=128, Hi=28, Wi=28, C=256, K=256, Y=3, X=3, pH=1, pW=1)
DEPTHWISE = ConvProblem(
    N=64, Hi=56, Wi=56, C=256, K=256, Y=3, X=3, pH=1, pW=1, groups=256
)


def _wgrad(problem, arch="gfx950", dtype="bf16", tile=(64, 32, 64, 512), **kw):
    tm, tn, tk, bs = tile
    return resolve_conv_split_k(
        "wgrad",
        problem,
        arch=arch,
        dtype=dtype,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        block_size=bs,
        **kw,
    )


def _previous_rule(problem, arch, dtype, tile, gm, split_k):
    """The dispatcher's split-K rule before the resolver existed."""
    tm, tn, tk, bs = tile
    grid_groups = max(1, problem.groups // gm)
    if split_k == -1:
        split_k = select_split_k_wgrad(
            wg_M=_wg_M(problem) * gm,
            wg_N=_wg_N(problem) * gm,
            wg_K=_wg_K(problem),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            arch=arch,
            groups=grid_groups,
            block_size=bs,
        ).split_k
    split_k = max(1, min(split_k, MAX_GRID_DIM_Z // grid_groups))
    atomic_ok = wgrad_atomic_epilogue_available(problem, dtype, None)[0]
    two_stage = ((not atomic_ok) or gm > 1) and split_k > 1
    if two_stage and wgrad_workspace_nbytes(problem, 8) > MAX_WGRAD_WS_BYTES:
        two_stage, split_k = False, 1
    return split_k, two_stage


@pytest.mark.parametrize(
    "problem,dtype,tile,gm,split_k",
    [
        (p, d, t, gm, sk)
        for p, d, t, gm, sk in itertools.product(
            (STEM, DENSE, DEPTHWISE),
            ("bf16", "fp16", "fp32"),
            ((64, 32, 64, 512), (128, 128, 32, 256), (16, 16, 32, 64)),
            (1, 4),
            (-1, 1, 8, 100000),
        )
        if gm == 1 or p is DEPTHWISE
    ],
)
def test_unfitted_arch_keeps_the_previous_rule(problem, dtype, tile, gm, split_k):
    got = _wgrad(problem, "gfx90a", dtype, tile, group_merge=gm, split_k=split_k)
    assert (got.split_k, got.two_stage) == _previous_rule(
        problem, "gfx90a", dtype, tile, gm, split_k
    )


def test_fwd_never_splits():
    r = resolve_conv_split_k(
        "fwd",
        DENSE,
        arch="gfx950",
        dtype="bf16",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        block_size=256,
    )
    assert (r.split_k, r.two_stage) == (1, False)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"direction": "bwd"}, "direction must be one of"),
        ({"reduction": "tree"}, "reduction must be one of"),
        ({"direction": "dgrad", "reduction": "atomic"}, "applies to wgrad only"),
    ],
)
def test_rejects_bad_arguments(kw, match):
    args = {"direction": "wgrad", "reduction": "auto"} | kw
    with pytest.raises(ValueError, match=match):
        resolve_conv_split_k(
            args.pop("direction"),
            DENSE,
            arch="gfx950",
            dtype="bf16",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            block_size=256,
            **args,
        )


def test_long_reduction_into_a_tiny_output_splits_deep():
    r = _wgrad(STEM)
    assert r.split_k >= 512
    assert r.two_stage  # odd dW row: the packed atomic cannot be built


def test_odd_dw_row_never_takes_the_atomic():
    for dtype in ("bf16", "fp16"):
        r = _wgrad(STEM, dtype=dtype, tile=(32, 16, 64, 128))
        assert r.split_k == 1 or r.two_stage


def test_merged_groups_reduce_through_the_workspace():
    r = _wgrad(DEPTHWISE, dtype="fp32", group_merge=4)
    assert r.split_k == 1 or r.two_stage


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_16bit_atomic_stays_under_its_accuracy_cap(dtype):
    for tile in ((64, 64, 64, 256), (16, 16, 32, 64), (32, 16, 64, 128)):
        r = _wgrad(DENSE, dtype=dtype, tile=tile, reduction="atomic")
        assert r.split_k <= ATOMIC_16BIT_MAX_SPLIT_K
        assert not r.two_stage


def test_fixed_workspace_reduction_never_returns_unsplit():
    r = _wgrad(DENSE, reduction="workspace", allow_unsplit=False)
    assert r.split_k > 1 and r.two_stage


def test_grid_z_budget_holds_for_grouped():
    r = _wgrad(DEPTHWISE, tile=(16, 16, 32, 64))
    assert r.split_k * DEPTHWISE.groups <= MAX_GRID_DIM_Z


def test_explicit_degree_is_kept_and_clamped_to_grid_z():
    assert _wgrad(DENSE, split_k=8).split_k == 8
    big = _wgrad(DEPTHWISE, split_k=100000)
    assert big.split_k == MAX_GRID_DIM_Z // DEPTHWISE.groups
    assert big.requested == 100000


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", 1),
        ("0", 1),
        ("1", 1),
        ("16", 16),
        (" Auto ", None),
        ("", None),
        (None, None),
    ],
)
def test_override_parses(value, expected):
    assert parse_split_k_override(value) == expected


@pytest.mark.parametrize("override", [1, 16])
def test_override(override):
    assert _wgrad(DENSE, override=override).split_k == override


def test_override_never_touches_an_explicit_degree():
    assert _wgrad(DENSE, split_k=8, override=1).split_k == 8


@pytest.mark.parametrize("value", ["-1", "-8", "lots"])
def test_override_parse_rejects_negative_and_garbage(value):
    with pytest.raises(ValueError, match="split-K override"):
        parse_split_k_override(value)


def test_override_rejects_a_degree_below_one():
    with pytest.raises(ValueError, match="override"):
        _wgrad(DENSE, override=0)


def test_resolution_ignores_the_environment(monkeypatch):
    # The resolver reads no ambient state: the benchmark's variable reaches it
    # only through ``override``.
    auto = _wgrad(DENSE)
    monkeypatch.setenv("ROCKE_CONV_SPLIT_K", "off")
    assert _wgrad(DENSE) == auto


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_override_moves_a_16bit_atomic_past_its_limit_to_the_workspace(dtype):
    over = 2 * ATOMIC_16BIT_MAX_SPLIT_K
    assert wgrad_atomic_epilogue_available(DENSE, dtype, None)[0]
    at_cap = _wgrad(DENSE, dtype=dtype, override=ATOMIC_16BIT_MAX_SPLIT_K)
    assert (at_cap.split_k, at_cap.two_stage) == (ATOMIC_16BIT_MAX_SPLIT_K, False)
    r = _wgrad(DENSE, dtype=dtype, override=over)
    assert (r.split_k, r.two_stage, r.requested) == (over, True, over)


def test_override_holds_a_fixed_16bit_atomic_to_its_limit():
    over = 2 * ATOMIC_16BIT_MAX_SPLIT_K
    r = _wgrad(DENSE, reduction="atomic", override=over)
    assert (r.split_k, r.two_stage, r.requested) == (
        ATOMIC_16BIT_MAX_SPLIT_K,
        False,
        over,
    )
    # fp32 has no such limit.
    r32 = _wgrad(DENSE, dtype="fp32", reduction="atomic", override=over)
    assert (r32.split_k, r32.two_stage) == (over, False)


def test_dgrad_explicit_and_unfitted_arch_unchanged():
    kw = {
        "arch": "gfx90a",
        "dtype": "bf16",
        "tile_m": 128,
        "tile_n": 64,
        "tile_k": 32,
        "block_size": 256,
    }
    assert resolve_conv_split_k("dgrad", DENSE, split_k=4, **kw).split_k == 4
    legacy = select_split_k_wgrad(
        wg_M=DENSE.N * DENSE.Hi * DENSE.Wi,
        wg_N=DENSE.cpg,
        wg_K=DENSE.Y * DENSE.X * DENSE.kpg,
        tile_m=128,
        tile_n=64,
        tile_k=32,
        arch="gfx90a",
    ).split_k
    assert resolve_conv_split_k("dgrad", DENSE, **kw).split_k == legacy


def test_dgrad_with_a_full_device_does_not_split():
    # N*Hi*Wi = 100k output rows: thousands of tiles before any split.
    r = resolve_conv_split_k(
        "dgrad",
        DENSE,
        arch="gfx950",
        dtype="bf16",
        tile_m=128,
        tile_n=128,
        tile_k=32,
        block_size=256,
    )
    assert r.split_k == 1


def test_benchmark_times_the_degree_dispatch_would_launch():
    # --run-from-cache times a split-K binary at its swept degrees; the one the
    # resolver picks must be among them even when it lies past the sweep list,
    # or the benchmark never measures what dispatch would run.
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _SPLIT_K_AUTO,
        _runtime_split_ks,
    )
    from benchmarks.common.kernel_cache import KernelIdentity

    ident = KernelIdentity.from_label(
        "wgrad-implicit_gemm-bf16-2d-t32x16x64-w2x1-a16x16x32-v8x1x1-compv3-"
        "default-grp-kouter-sk2-2stage-wsr8-pfa-wave64-gfx950-llvm22"
    )
    planned = _wgrad(
        STEM, tile=(32, 16, 64, 128), reduction="workspace", allow_unsplit=False
    ).split_k
    assert planned > max(_SPLIT_K_AUTO)
    degrees = _runtime_split_ks(STEM, ident)
    assert planned in degrees
    assert set(_SPLIT_K_AUTO) - {1} <= set(degrees)
    assert list(degrees) == sorted(degrees, reverse=True)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_dgrad_auto_never_picks_a_split_the_builder_rejects(dtype):
    # Odd channels-per-group: the packed 16-bit atomic cannot address dX, so
    # DgradConvSpec.validate rejects any split; auto must not hand it one.
    odd_cpg = ConvProblem(N=1, Hi=8, Wi=8, C=3, K=512, Y=7, X=7, pH=3, pW=3)
    r = resolve_conv_split_k(
        "dgrad",
        odd_cpg,
        arch="gfx950",
        dtype=dtype,
        tile_m=16,
        tile_n=16,
        tile_k=64,
        block_size=64,
    )
    assert r.split_k == 1


def test_merged_tiles_always_split():
    # The AOT cache holds merged kernels only in split-K form.
    r = _wgrad(DEPTHWISE, dtype="bf16", tile=(64, 64, 64, 256), group_merge=4)
    assert r.split_k > 1 and r.two_stage


# ---- regressions found in review --------------------------------------------

# Depthwise, batch 1, 4x4: the whole reduction fits in one K tile.
ONE_K_TILE_DW = ConvProblem(
    N=1, Hi=4, Wi=4, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64
)


def test_merged_tile_with_one_k_tile_falls_back_to_unsplit():
    r = _wgrad(ONE_K_TILE_DW, tile=(64, 64, 64, 256), group_merge=4)
    assert (r.split_k, r.two_stage) == (1, False)


def test_split_only_caller_gets_a_clear_error_when_nothing_can_split():
    with pytest.raises(ValueError, match="cannot be built"):
        _wgrad(
            ONE_K_TILE_DW,
            tile=(64, 64, 64, 256),
            group_merge=4,
            reduction="workspace",
            allow_unsplit=False,
        )


def test_atomic_request_on_a_merged_tile_does_not_raise():
    # A merged tile has no packed-atomic reduction; an atomic-binary caller
    # gets the unsplit kernel rather than an exception.
    r = _wgrad(DEPTHWISE, tile=(64, 64, 64, 256), group_merge=4, reduction="atomic")
    assert (r.split_k, r.two_stage) == (1, False)


def _dgrad(problem, dtype="bf16", **kw):
    return resolve_conv_split_k(
        "dgrad",
        problem,
        arch="gfx950",
        dtype=dtype,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        block_size=256,
        **kw,
    )


ODD_CPG = ConvProblem(N=1, Hi=8, Wi=8, C=3, K=512, Y=7, X=7, pH=3, pW=3)


def test_dgrad_override_is_held_to_what_the_kernel_can_build():
    assert _dgrad(ODD_CPG, override=8).split_k == 1
    assert _dgrad(DENSE, override=8).split_k == 8
    # The 16-bit atomic stays within its accuracy limit; fp32 does not have one.
    over = 2 * ATOMIC_16BIT_MAX_SPLIT_K
    r = _dgrad(DENSE, override=over)
    assert (r.split_k, r.requested) == (ATOMIC_16BIT_MAX_SPLIT_K, over)
    assert _dgrad(DENSE, dtype="fp32", override=over).split_k == over


def test_dgrad_groups_do_not_share_the_grid_z_budget():
    # dgrad rides the group on grid y: a forced degree above 65535 // groups
    # is still legal.
    many = ConvProblem(
        N=1, Hi=8, Wi=8, C=2048, K=2048, Y=3, X=3, pH=1, pW=1, groups=1024
    )
    assert _dgrad(many, override=128).split_k == 128


@pytest.mark.parametrize(
    "problem,dtype,wave,match",
    [
        (DENSE, "bf16", 32, "CDNA-only"),
        (DENSE, "int8", 64, "dtype_d"),
        (ODD_CPG, "bf16", 64, "even channels-per-group"),
        (ODD_CPG, "fp16", 64, "even channels-per-group"),
    ],
)
def test_dgrad_split_k_available_rejections(problem, dtype, wave, match):
    from kernels.common.conv_implicit_gemm_dgrad import dgrad_split_k_available

    ok, why = dgrad_split_k_available(problem, dtype, wave_size=wave)
    assert not ok
    assert match in why


def test_dgrad_split_k_available_accepts_fp32_odd_channels():
    from kernels.common.conv_implicit_gemm_dgrad import dgrad_split_k_available

    assert dgrad_split_k_available(ODD_CPG, "fp32") == (True, "ok")


def test_benchmark_workers_skip_combos_the_resolver_cannot_serve():
    # The JIT sweep builds single- and two-stage legs for every combo; a combo
    # with nothing to split must come back None, not raise and end the run.
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        WgradCombo,
        _build_wgrad_two_stage_one,
    )

    tiny = ConvProblem(N=1, Hi=7, Wi=7, C=512, K=512, Y=3, X=3, pH=1, pW=1)
    combo = WgradCombo(64, 64, 64, 2, 2, 32, "mem", "default", False, -1)
    assert _build_wgrad_two_stage_one((tuple(combo), tiny, "bf16", "gfx950")) is None


# Workspace bytes = groups * R * wg_M * wg_N * 4: R=8 fits the i32 ws_bytes
# kernarg, R=16 does not.
WIDE = ConvProblem(N=8, Hi=14, Wi=14, C=2048, K=2048, Y=3, X=3, pH=1, pW=1)


def test_wide_problem_straddles_the_workspace_limit():
    assert wgrad_workspace_nbytes(WIDE, 8) <= MAX_WGRAD_WS_BYTES
    assert wgrad_workspace_nbytes(WIDE, 16) > MAX_WGRAD_WS_BYTES


@pytest.mark.parametrize("split_k", [-1, 8])
def test_pinned_workspace_keeps_its_size_limit(split_k):
    with pytest.raises(ValueError, match="ws_bytes limit"):
        _wgrad(WIDE, reduction="workspace", ws_replicas=16, split_k=split_k)
    fits = _wgrad(
        WIDE,
        reduction="workspace",
        ws_replicas=8,
        split_k=split_k,
        allow_unsplit=False,
    )
    assert fits.two_stage and fits.split_k > 1


def test_auto_never_plans_a_workspace_its_replicas_overflow():
    for split_k in (-1, 8):
        r = _wgrad(WIDE, dtype="fp16", ws_replicas=16, split_k=split_k)
        assert not r.two_stage


def test_planner_rejects_a_batch_beyond_grid_z():
    from rocke.helpers.split_k_planner import (
        SplitKKernel,
        SplitKProblem,
        select_split_k_plan,
        split_k_cost_model,
    )

    with pytest.raises(ValueError, match="gridDim.z"):
        select_split_k_plan(
            SplitKProblem(m=1, n=1, k=1, batch=MAX_GRID_DIM_Z + 1),
            SplitKKernel(16, 16, 32, 1),
            [],
            split_k_cost_model("gfx950"),
        )


def test_two_stage_builder_spec_is_never_resolved_unsplit(monkeypatch):
    # A two-stage spec is a split-K kernel; auto must not hand it degree 1.
    from kernels.common import conv_split_k
    from kernels.common.conv_implicit_gemm import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
    )

    seen = {}
    real = conv_split_k.resolve_conv_split_k

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr(conv_split_k, "resolve_conv_split_k", spy)
    spec = WgradConvSpec(
        problem=DENSE,
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        split_k=-1,
        two_stage=True,
        ws_replicas=4,
    )
    build_implicit_gemm_conv_wgrad(spec, arch="gfx950")
    assert seen["allow_unsplit"] is False
    assert seen["ws_replicas"] == 4


def test_cost_fit_keeps_each_workspace_replica_count_its_own_option():
    # Two workspace binaries of one main loop that differ only in replica
    # count: each time must stay paired with its own setup traffic.
    from benchmarks.common.split_k_cost_fit import _decisions, _Kernel
    from rocke.helpers.split_k_planner import (
        SplitKKernel,
        SplitKProblem,
        SplitKReduction,
        split_k_cost_model,
    )

    problem = SplitKProblem(m=64, n=64, k=1 << 16)
    tile = SplitKKernel(64, 64, 64, 4)

    def kernel(reduction, times):
        return _Kernel(
            "log", "cmd", "label", problem, tile, reduction, ("loop",), times
        )

    def ws(copies):
        return SplitKReduction(
            "workspace", "fp32", workspace=True, workspace_copies=copies
        )

    (d,) = _decisions(
        [
            kernel(None, {1: 10.0}),
            kernel(ws(1), {8: 1.0}),
            kernel(ws(8), {8: 2.0}),
        ],
        split_k_cost_model("gfx950"),
        max_degree=8,
    )
    out = 64 * 64 * 2
    pairs = {(float(nb), float(ms)) for nb, ms in zip(d.setup_bytes, d.measured) if nb}
    assert (2 * 1 * 64 * 64 * 4 + out, 1.0) in pairs
    assert (2 * 8 * 64 * 64 * 4 + out, 2.0) in pairs


def test_cost_fit_degree_list_parses_ranges():
    import argparse

    from benchmarks.common.split_k_cost_fit import _parse_degrees

    assert _parse_degrees("2-5,8, 12-12") == [2, 3, 4, 5, 8, 12]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_degrees("1-4")


def test_cost_fit_round_trip_keeps_the_unsearched_fields():
    # The literal the fit prints must not drop the XCD count or the margin.
    from benchmarks.common.split_k_cost_fit import _model_of, _x_of
    from rocke.helpers.split_k_planner import split_k_cost_model

    for cost in (
        split_k_cost_model("gfx950"),
        split_k_cost_model("gfx950", num_cus=304, num_xcds=6),
    ):
        back = _model_of(_x_of(cost), cost.max_split_k)
        assert (back.num_xcds, back.ladder_margin) == (
            cost.num_xcds,
            cost.ladder_margin,
        )


def test_default_epilogue_never_resolves_to_the_packed_atomic():
    # Only cshuffle emits the packed 16-bit atomic: a direct-store kernel must
    # split through the workspace or not at all.
    kw = {
        "arch": "gfx950",
        "dtype": "bf16",
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "block_size": 256,
    }
    p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    for split_k in (-1, 8):
        assert resolve_conv_split_k(
            "wgrad", p, split_k=split_k, epilogue="cshuffle", **kw
        ) == resolve_conv_split_k("wgrad", p, split_k=split_k, **kw)
        r = resolve_conv_split_k("wgrad", p, split_k=split_k, epilogue="default", **kw)
        assert r.split_k == 1 or r.two_stage, r
    fixed = resolve_conv_split_k(
        "wgrad", p, reduction="atomic", epilogue="default", **kw
    )
    assert fixed.split_k == 1


def test_explicit_atomic_that_cannot_be_built_raises():
    # reduction="atomic" says the caller's binary is the packed-atomic one; a
    # problem it cannot serve must fail here, not come back with a degree the
    # builder then rejects. Degree 1 needs no reduction and still resolves.
    kw = {
        "arch": "gfx950",
        "dtype": "bf16",
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "block_size": 256,
        "reduction": "atomic",
    }
    p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    assert resolve_conv_split_k("wgrad", p, split_k=8, **kw).split_k == 8
    with pytest.raises(ValueError, match="default epilogue"):
        resolve_conv_split_k("wgrad", p, split_k=8, epilogue="default", **kw)
    with pytest.raises(ValueError, match="merged tile"):
        resolve_conv_split_k("wgrad", DEPTHWISE, split_k=8, group_merge=4, **kw)
    with pytest.raises(ValueError, match="packed atomic"):
        resolve_conv_split_k("wgrad", STEM, split_k=8, **kw)
    assert resolve_conv_split_k("wgrad", STEM, split_k=1, **kw).split_k == 1
