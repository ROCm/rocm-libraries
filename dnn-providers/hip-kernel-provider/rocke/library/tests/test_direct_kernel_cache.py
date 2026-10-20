# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the direct-conv kernel cache: what gets built, which
cached binary is offered for which problem, and what a launch plan binds.

Nothing is compiled -- the cache is filled with placeholder HSACOs -- and no
GPU is touched.
"""

from __future__ import annotations

import itertools
import json

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common.kernel_cache import KernelCache
from kernels.conv.direct_aot_jobs import _DENSE_MFMA_DGRAD_CHANNELS
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_variants import direct_mfma_knobs_reason
from rocke.core.arch import ArchTarget
from rocke.runtime.packing import compile_packer

_ARCH = "gfx950"


@pytest.fixture(scope="module")
def jobs():
    return dks.enumerate_direct_jobs(
        arch=_ARCH,
        target=ArchTarget.from_gfx(_ARCH),
        directions=("fwd", "dgrad", "wgrad"),
    )


def _problem(**kw) -> DirectConvProblem:
    shape = dict(
        N=2, H=28, W=28, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
    )
    shape.update(kw)
    return DirectConvProblem(**shape)


# The column-streamed depthwise kernel bounds its output rows against p_Ho
# instead of streaming them by input row, so it is the one padding-generic
# forward kernel.
_PADDING_GENERIC_FWD = {"direct_depthwise_col"}


def test_forward_entries_only_use_same_padding(jobs):
    """Row-streaming forward kernels are wrong for any other padding, so none
    may be built. The non-grouped kernel bounds its rows by the runtime output
    extents; its dilated rows take the padding their workloads use."""
    fwd = [
        j.identity
        for j in jobs
        if j.identity.direction == "direct_fwd"
        and j.identity.algorithm not in _PADDING_GENERIC_FWD
        and not (
            j.identity.algorithm == "direct_nongrouped"
            and j.identity.dilation_h * j.identity.dilation_w > 1
        )
    ]
    assert fwd
    assert all(i.pad_h == (i.filter_h - 1) // 2 for i in fwd)
    assert all(i.filter_h % 2 == 1 for i in fwd)


def test_only_the_nongrouped_kernel_is_dilated(jobs):
    dilated = [
        j.identity for j in jobs if j.identity.dilation_h * j.identity.dilation_w > 1
    ]
    assert dilated
    assert {i.algorithm for i in dilated} == {"direct_nongrouped"}


@pytest.mark.parametrize(
    "pad, stride, dil_h, dil_w", [(1, 1, 3, 1), (4, 1, 1, 4), (2, 1, 2, 2)]
)
def test_dilated_problem_gets_only_its_dilation(cache, pad, stride, dil_h, dil_w):
    """A dilated problem is served by the non-grouped binaries baked for its
    dilation only, and an undilated one never by a dilated binary."""
    problem = _problem(
        N=8,
        H=14,
        W=14,
        groups=1,
        cpg=512,
        kpg=512,
        PAD=pad,
        stride=stride,
        dil_h=dil_h,
        dil_w=dil_w,
    )
    plans, _ = dks.direct_plans(cache, problem, "fwd", _ARCH)
    assert plans
    assert {(p.identity.dilation_h, p.identity.dilation_w) for p in plans} == {
        (dil_h, dil_w)
    }
    (step,) = plans[0].steps
    assert step.conv_args.geom.Ho == problem.Ho and step.conv_args.geom.Wo == problem.Wo
    plans, _ = dks.direct_plans(
        cache, _problem(groups=1, cpg=512, kpg=512), "fwd", _ARCH
    )
    assert plans and all(
        p.identity.dilation_h == p.identity.dilation_w == 1 for p in plans
    )


def test_padding_generic_forward_entries_cover_other_paddings(jobs):
    pads = {
        (j.identity.filter_h, j.identity.stride_h, j.identity.pad_h)
        for j in jobs
        if j.identity.algorithm in _PADDING_GENERIC_FWD
    }
    # "valid" and "same" padding at both strides; intermediate paddings are
    # not cached.
    assert {(5, 1, 0), (5, 1, 2), (5, 2, 0), (5, 2, 2)} <= pads
    assert (5, 2, 1) not in pads


def test_every_capability_row_is_built(jobs):
    built = {
        (
            j.identity.algorithm,
            j.identity.filter_h,
            j.identity.pad_h,
            j.identity.stride_h,
            j.identity.cpg,
            j.identity.kpg,
        )
        for j in jobs
    }
    for variant, (_, caps_list) in dks.DIRECT_CAPABILITIES.items():
        for c in caps_list:
            assert (variant, c.KH, c.PAD, c.stride, c.cpg, c.kpg) in built, (variant, c)


def test_identities_are_unique(jobs):
    hashes = [j.identity.stable_hash() for j in jobs]
    assert len(hashes) == len(set(hashes))


@pytest.fixture
def cache(tmp_path, jobs):
    """A cache holding every job with a placeholder binary."""
    c = KernelCache(tmp_path, _ARCH)
    for job in jobs:
        c.put(job.identity, b"placeholder", {"kernel_name": job.identity.label()})
    return c


def _variants(plans):
    return {p.identity.algorithm for p in plans}


def test_padding_zero_is_a_baked_value(cache):
    """PAD=0 must not read as "runtime": a same-padding binary is wrong for it."""
    plans, _ = dks.direct_plans(cache, _problem(PAD=0), "fwd", _ARCH)
    assert plans == []
    plans, _ = dks.direct_plans(cache, _problem(PAD=0), "dgrad", _ARCH)
    assert plans and all(p.identity.pad_h == 0 for p in plans)


@pytest.mark.parametrize("groups", [1, 3, 16, 48])
def test_group_count_is_runtime(cache, groups):
    """Any group count is served; only tiles that do not divide it drop out."""
    plans, rejected = dks.direct_plans(cache, _problem(groups=groups), "fwd", _ARCH)
    assert plans
    assert all(
        "block_groups" in why
        for ident, why in rejected
        if ident.algorithm != "direct_nongrouped"
    )


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [3, 9])
def test_small_group_depthwise_offers_spatial(cache, groups, stride):
    """A depthwise problem with fewer groups than a wave gets the spatial
    kernel, launched for its own group count: the binary was compiled on the
    groups=8 probe, but each wave covers ``wave_size // groups`` columns of
    the real problem."""
    # At W=226 every grid differs from the one the groups=8 probe would get.
    problem = _problem(H=8, W=226, groups=groups, cpg=1, kpg=1, stride=stride)
    plans, _ = dks.direct_plans(cache, problem, "fwd", _ARCH)
    spatial = [p for p in plans if p.identity.algorithm == "direct_depthwise_spatial"]
    assert {json.loads(p.identity.knobs)["block_waves"] for p in spatial} == set(
        dks.DW_BLOCK_WAVES
    )
    for plan in spatial:
        waves = json.loads(plan.identity.knobs)["block_waves"]
        block_w = waves * (plan.identity.wave_size // groups)
        (step,) = plan.steps
        assert step.grid == (-(-problem.Wo // block_w), 1, problem.N)
        assert step.block == (waves * plan.identity.wave_size, 1, 1)
        probe_w = waves * (plan.identity.wave_size // 8)
        assert step.grid[0] != -(-problem.Wo // probe_w)


@pytest.mark.parametrize("stride", [1, 2])
def test_large_depthwise_filters_get_the_tiled_kernel(cache, stride):
    """Filters past the row-streaming grid (13x13 .. 31x31) are served by the
    output-stationary kernel, launched with one block per output-row tile.
    At stride 1 the column-streamed kernel competes with it (the spatial one
    too, but only below a wave of groups)."""
    problem = _problem(
        N=2, H=28, W=28, groups=192, cpg=1, kpg=1, KH=31, KW=31, PAD=15, stride=stride
    )
    plans, _ = dks.direct_plans(cache, problem, "fwd", _ARCH)
    expected = {"direct_depthwise_tiled"}
    if stride == 1:
        expected.add("direct_depthwise_col")
    assert plans and _variants(plans) == expected
    for plan in plans:
        if plan.identity.algorithm != "direct_depthwise_tiled":
            continue
        knobs = json.loads(plan.identity.knobs)
        (step,) = plan.steps
        assert step.grid == (
            -(-problem.Wo // knobs["block_w"]),
            -(-problem.groups // (64 * knobs["block_waves"])),
            problem.N * -(-problem.Ho // knobs["block_h"]),
        )


def test_no_spilling_row_streaming_depthwise_is_built(jobs):
    """A row-streaming depthwise kernel past the register file takes minutes
    to build and loses to the tiled kernel, so the grid leaves it out."""
    from kernels.common.conv_direct_grouped import (
        DirectDepthwiseSpec,
        depthwise_stream_register_reason,
    )

    stream = [j for j in jobs if j.identity.algorithm == "direct_depthwise"]
    assert stream
    for job in stream:
        caps = dks.DirectCaps(**job.caps)
        spec = DirectDepthwiseSpec(
            problem=dks.probe_problem(caps, "fp16", "direct_depthwise"),
            **job.spec_kwargs,
        )
        assert depthwise_stream_register_reason(spec) is None, job.identity.label()


def _nongrouped(plans):
    return [p for p in plans if p.identity.algorithm == "direct_nongrouped"]


@pytest.mark.parametrize(
    "C,K,H,W", [(64, 64, 56, 56), (640, 640, 64, 64), (96, 352, 28, 48)]
)
def test_nongrouped_channels_are_runtime(cache, C, K, H, W):
    """One non-grouped binary per tile serves any C/K it divides."""
    problem = _problem(groups=1, cpg=C, kpg=K, H=H, W=W)
    plans = _nongrouped(dks.direct_plans(cache, problem, "fwd", _ARCH)[0])
    assert plans
    assert all(p.identity.cpg == 0 and p.identity.kpg == 0 for p in plans)
    # Only the widths a sweep for this Wo would try are offered.
    from kernels.common.conv_direct_nongrouped import tile_w_candidates

    for plan in plans:
        spec = dks.make_spec(
            "direct_nongrouped", problem, json.loads(plan.identity.knobs)
        )
        assert spec.tile_w in tile_w_candidates(problem.Wo, spec.atom_tile)
        (step,) = plan.steps
        assert step.grid == spec.grid()


def test_nongrouped_grid_skips_spilling_tiles(jobs):
    """Tiles that would spill are not cached, and the cut leaves both atoms and
    both strides covered."""
    from kernels.common.conv_direct_nongrouped import (
        DirectNongroupedConvSpec,
        nongrouped_register_reason,
    )

    built = [j for j in jobs if j.identity.algorithm == "direct_nongrouped"]
    assert built
    for job in built:
        caps = dks.DirectCaps(**job.caps)
        spec = dks.make_spec(
            "direct_nongrouped",
            dks.probe_problem(caps, dks.DIRECT_DTYPE, "direct_nongrouped"),
            job.spec_kwargs,
        )
        assert nongrouped_register_reason(spec, _ARCH) is None, spec.kernel_name()
    assert {json.loads(j.identity.knobs)["atom"] for j in built} == {
        "32x32x16",
        "16x16x32",
    }
    assert {j.identity.stride_h for j in built} == {1, 2}

    # A tile filling the whole accumulator file is rejected however few waves
    # share the SIMD: it spilled in every such build on gfx950.
    full = DirectNongroupedConvSpec(
        problem=_problem(groups=1, cpg=64, kpg=256),
        tile_h=8,
        tile_w=32,
        tile_k=256,
        ck=16,
        waves_m=2,
        waves_n=2,
    )
    assert full.acc_vgprs == 256
    assert "accumulator file" in nongrouped_register_reason(full, _ARCH)


def test_nongrouped_only_serves_one_group(cache):
    plans, rejected = dks.direct_plans(cache, _problem(groups=16), "fwd", _ARCH)
    assert not _nongrouped(plans)
    assert all(i.algorithm != "direct_nongrouped" for i, _ in rejected)
    # Grouped forward has no stride-2 kernel; the non-grouped one does.
    plans, _ = dks.direct_plans(
        cache, _problem(groups=1, cpg=64, kpg=64, stride=2), "fwd", _ARCH
    )
    assert plans and _variants(plans) == {"direct_nongrouped"}


def test_depthwise_forward_offers_col_alongside_preload(cache):
    same = _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=2, stride=2)
    plans, _ = dks.direct_plans(cache, same, "fwd", _ARCH)
    # groups == wave_size still leaves the spatial kernel one column per wave.
    assert _variants(plans) == {
        "direct_depthwise",
        "direct_depthwise_spatial",
        "direct_depthwise_col",
        "direct_depthwise_tiled",
    }
    other = _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2)
    plans, _ = dks.direct_plans(cache, other, "fwd", _ARCH)
    assert _variants(plans) == {"direct_depthwise_col"}
    # The row tile is a capability: the grid follows the runtime height.
    for plan in plans:
        (step,) = plan.steps
        knobs = json.loads(plan.identity.knobs)
        n_h_tiles = -(-other.Ho // knobs["block_h"])
        assert step.grid[2] == other.N * n_h_tiles


def test_capabilities_outside_the_list_are_not_served(cache):
    assert dks.direct_plans(cache, _problem(cpg=24, kpg=24), "fwd", _ARCH)[0] == []
    # Grouped forward stride 2 has no rocke kernel.
    assert dks.direct_plans(cache, _problem(stride=2), "fwd", _ARCH)[0] == []


def test_dgrad_stride1_offers_mfma_pipeline_and_scalar(cache):
    plans, _ = dks.direct_plans(cache, _problem(), "dgrad", _ARCH)
    # cpg = kpg = 16 has the fixed-channel MFMA entries and also the
    # runtime-channel ones (one K-atom, one M-tile).
    assert _variants(plans) == {
        "direct_grouped_dgrad",
        "direct_grouped_dgrad_mfma",
        "direct_dgrad_mfma_rtc",
    }
    for plan in plans:
        if plan.identity.algorithm not in (
            "direct_grouped_dgrad_mfma",
            "direct_dgrad_mfma_rtc",
        ):
            continue
        # transpose, then the forward compute pass reading the transposed weights
        assert [s.buffers for s in plan.steps] == [
            ("w", None, "ws_t"),
            ("dy", "ws_t", "dx"),
        ]
        assert set(plan.workspaces) == {"ws_t"}


@pytest.mark.parametrize("cpg,kpg", _DENSE_MFMA_DGRAD_CHANNELS)
def test_dense_mfma_dgrad_channels_are_routed(cache, cpg, kpg):
    """Each tuned dense (cpg, kpg) row in ``_DENSE_MFMA_DGRAD_CHANNELS`` must
    reach the MFMA dgrad pipeline for a real problem with those channel
    counts, with a groups=1-compatible knob -- otherwise the row is buildable
    but never dispatched to."""
    problem = _problem(groups=1, cpg=cpg, kpg=kpg)
    plans, _ = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    mfma = [p for p in plans if p.identity.algorithm == "direct_grouped_dgrad_mfma"]
    assert mfma
    assert all(json.loads(p.identity.knobs)["block_groups"] == 1 for p in mfma)


def test_mfma_dgrad_grid_skips_spilling_preloads(jobs):
    """With waves_k > 1 the MFMA dgrad's fprop pass preloads its weights into
    registers; specs past the budget spill at the 512-VGPR ceiling, where
    comgr has produced a wrong-dX kernel. None of them is built."""
    built = [j for j in jobs if j.identity.algorithm == "direct_grouped_dgrad_mfma"]
    assert built
    for job in built:
        caps = dks.DirectCaps(**job.caps)
        probe = dks.probe_problem(caps, dks.DIRECT_DTYPE, "direct_grouped_dgrad_mfma")
        assert (
            direct_mfma_knobs_reason(probe, job.spec_kwargs) is None
        ), job.identity.label()

    # The spec that computed the wrong dX on gfx950 is rejected; halving its
    # Q tile brings it under the budget and it stays.
    problem = _problem(groups=1, cpg=48, kpg=192)
    knobs = dict(
        block_q=32,
        block_groups=1,
        block_h=8,
        waves_q=1,
        waves_k=2,
        runtime_k_loop=False,
        persistent_grid=False,
        fold_k32=True,
    )
    assert "spill" in direct_mfma_knobs_reason(problem, knobs)
    assert direct_mfma_knobs_reason(problem, dict(knobs, block_q=16)) is None


@pytest.mark.parametrize(
    "direction,problem",
    [
        ("fwd", _problem()),
        ("fwd", _problem(groups=1, cpg=128, kpg=96)),
        ("fwd", _problem(groups=64, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2)),
        # Only the column-streamed kernel serves non-"same" forward padding.
        ("fwd", _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2)),
        ("dgrad", _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2)),
        ("dgrad", _problem(cpg=8, kpg=8, stride=2)),
        ("dgrad", _problem()),
        ("wgrad", _problem()),
        ("wgrad", _problem(groups=1, cpg=48, kpg=192, KH=5, KW=5, PAD=0)),
        # Dense MFMA dgrad (runtime channels): transforms take two scalars.
        ("dgrad", _problem(groups=1, cpg=48, kpg=192)),
        ("dgrad", _problem(groups=1, cpg=24, kpg=24)),
    ],
)
def test_plans_pack_against_their_signatures(cache, direction, problem):
    plans, _ = dks.direct_plans(cache, problem, direction, _ARCH)
    assert plans
    names = ("x", "w", "y", "dy", "dx", "dw", "ws_t", "ws_coa")
    ptrs = {n: 0x1000 * (i + 1) for i, n in enumerate(names)}
    sizes = {n: 1 << 20 for n in names}
    for plan in plans:
        for step in plan.steps:
            compile_packer(step.signature)(dks.launch_values(step, ptrs, sizes))
            assert all(g > 0 for g in step.grid)


def _wgrad(plans):
    return [p for p in plans if p.identity.algorithm == "direct_wgrad"]


@pytest.mark.parametrize(
    "groups,cpg,kpg",
    [(1, 24, 24), (1, 48, 192), (1, 128, 128), (4, 64, 64), (32, 16, 16)],
)
def test_wgrad_channels_are_runtime(cache, groups, cpg, kpg):
    """One wgrad binary per knob serves any channel count its knobs fit, and
    the launch grid follows the real problem, not the probe it was built on."""
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    problem = _problem(groups=groups, cpg=cpg, kpg=kpg)
    plans, rejected = dks.direct_plans(cache, problem, "wgrad", _ARCH)
    plans = _wgrad(plans)
    assert plans
    c_tiles = -(-cpg // 16)
    for plan in plans:
        assert plan.identity.cpg == 0 and plan.identity.kpg == 0
        knobs = json.loads(plan.identity.knobs)
        assert knobs["waves_c"] <= c_tiles
        (step,) = plan.steps
        assert step.buffers == ("dy", "x", "dw")
        spec = dks.make_spec("direct_wgrad", problem, knobs)
        assert spec.runtime_channels
        assert (step.grid, step.block) == direct_launch_geometry(spec)
    # Every knob combination too wide for the channels is turned away, with
    # the reason; nothing else is.
    for ident, why in rejected:
        if ident.algorithm == "direct_wgrad":
            assert json.loads(ident.knobs)["waves_c"] > c_tiles, why
    offered = {json.loads(p.identity.knobs)["waves_c"] for p in plans}
    assert 1 in offered
    if c_tiles in {w[1] for w in dks.wgrad_waves()}:
        assert c_tiles in offered


def test_wgrad_serves_valid_and_same_padding_only_at_stride_1(cache):
    for pad in (0, 1):
        plans, _ = dks.direct_plans(cache, _problem(PAD=pad), "wgrad", _ARCH)
        assert _wgrad(plans) and all(p.identity.pad_h == pad for p in plans)
    # The kernel is stride-1 only; nothing is cached for stride 2.
    assert dks.direct_plans(cache, _problem(stride=2), "wgrad", _ARCH)[0] == []
    assert not [j for j in dks.DIRECT_CAPABILITIES["direct_wgrad"][1] if j.stride != 1]


def test_wgrad_needs_a_full_wave_tile_of_channels(cache):
    """cpg/kpg below the 16-wide wave tile are a per-problem rejection now."""
    plans, rejected = dks.direct_plans(
        cache, _problem(cpg=8, kpg=8, groups=32), "wgrad", _ARCH
    )
    assert not _wgrad(plans)
    assert any(i.algorithm == "direct_wgrad" for i, _ in rejected)


# ---------------------------------------------------------------------------
# Runtime-channel MFMA dgrad (direct_dgrad_mfma_rtc)
# ---------------------------------------------------------------------------


def _rtc(plans):
    return [p for p in plans if p.identity.algorithm == "direct_dgrad_mfma_rtc"]


def _jit_dgrad_mfma_combos(problem):
    """The (block_q, block_groups, block_h, waves) combos the JIT dgrad sweep
    builds for ``problem`` (benchmark_direct_conv._run_dgrad_sweep), as knobs."""
    out = set()
    for bq, bg, bh, (wq, wk, rk, pg, k32) in itertools.product(
        dks.BLOCK_Q, dks.BLOCK_GROUPS, dks.DGRAD_BLOCK_H, dks.DGRAD_WAVES
    ):
        if problem.groups % bg or bq // wq < 16:
            continue
        if k32:
            if problem.kpg % 32:
                continue
            n_atoms = problem.kpg // 32
        else:
            n_atoms = (problem.kpg + 15) // 16
        if n_atoms == 0 or n_atoms % wk:
            continue
        out.add((bq, bg, bh, wq, wk, rk, pg, k32))
    return out


@pytest.mark.parametrize(
    "groups,cpg,kpg",
    [
        (1, 48, 192),
        (1, 4, 320),
        (1, 24, 24),
        (1, 64, 128),
        (1, 16, 512),
        (4, 40, 20),
        (16, 16, 16),
    ],
)
def test_dgrad_rtc_serves_any_channels_its_tiles_fit(cache, groups, cpg, kpg):
    """One runtime-channel binary per tile count serves every channel count
    that rounds to it; the knobs offered are the JIT sweep's, less the
    duplicated runtime_k_loop splits and the configs that would spill."""
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    problem = _problem(groups=groups, cpg=cpg, kpg=kpg)
    plans = _rtc(dks.direct_plans(cache, problem, "dgrad", _ARCH)[0])
    assert plans
    offered = set()
    for plan in plans:
        ident = plan.identity
        assert (ident.cpg, ident.kpg) == (0, 0)
        knobs = json.loads(ident.knobs)
        atom = 32 if knobs["fold_k32"] else 16
        assert knobs["m_tiles"] == -(-cpg // 16)
        assert knobs["k_atoms"] == (kpg // atom if knobs["fold_k32"] else -(-kpg // 16))
        offered.add(
            (
                knobs["block_q"],
                knobs["block_groups"],
                knobs["block_h"],
                knobs["waves_q"],
                knobs["waves_k"],
                knobs["runtime_k_loop"],
                knobs["persistent_grid"],
                knobs["fold_k32"],
            )
        )
        *helpers, fprop = plan.steps
        assert helpers[0].buffers == ("w", None, "ws_t")
        for step in helpers:
            assert dict(step.scalars) == {"p_cpg": cpg, "p_kpg": kpg}
            assert [a["name"] for a in step.signature][-2:] == ["p_cpg", "p_kpg"]
        # The compute pass launches on the real (transposed) problem.
        spec = dks.make_spec("direct_dgrad_mfma_rtc", problem, knobs)
        assert spec.runtime_channels
        assert (fprop.grid, fprop.block) == direct_launch_geometry(spec)
        coalesced = knobs["waves_k"] > 1 or knobs["runtime_k_loop"]
        assert fprop.buffers[1] == ("ws_coa" if coalesced else "ws_t")
        assert len(helpers) == (2 if coalesced else 1)
    jit = _jit_dgrad_mfma_combos(problem)
    assert offered <= jit
    missing = jit - offered
    for bq, bg, bh, wq, wk, rk, pg, k32 in missing:
        knobs = dict(
            block_q=bq,
            block_groups=bg,
            block_h=bh,
            waves_q=wq,
            waves_k=wk,
            runtime_k_loop=rk,
            persistent_grid=pg,
            fold_k32=k32,
            k_atoms=(kpg // 32 if k32 else -(-kpg // 16)),
            m_tiles=-(-cpg // 16),
        )
        why = (
            (wk > 1 and rk)
            # A K split's LDS reduction is per group: the spec rejects it.
            or (wk > 1 and bg > 1)
            or knobs["k_atoms"] not in dks.dgrad_rtc_k_atoms(wk, k32)
            or knobs["m_tiles"] not in dks.DGRAD_RTC_M_TILES
            or dks.dgrad_rtc_spills(knobs, 3, 3, _ARCH)
        )
        assert why, f"JIT config {knobs} has no cached binary"


def test_dgrad_rtc_cached_scope(jobs):
    """The tile counts built, and only binaries that fit their registers."""
    built = [
        json.loads(j.identity.knobs)
        for j in jobs
        if j.identity.algorithm == "direct_dgrad_mfma_rtc"
    ]
    assert built
    assert {k["m_tiles"] for k in built} == set(dks.DGRAD_RTC_M_TILES)
    for k in built:
        assert k["k_atoms"] in dks.dgrad_rtc_k_atoms(k["waves_k"], k["fold_k32"])
        assert not dks.dgrad_rtc_spills(k, 3, 3, _ARCH)
        assert not (k["waves_k"] > 1 and k["runtime_k_loop"])
    # One transpose binary and one reorganize binary per fold_k32 serve them.
    helpers = [
        j.identity
        for j in jobs
        if j.identity.algorithm
        in ("direct_dgrad_transpose_rtc", "direct_dgrad_reorganize_rtc")
    ]
    assert sorted((i.algorithm, i.knobs) for i in helpers) == sorted(
        [
            ("direct_dgrad_reorganize_rtc", json.dumps({"fold_k32": False})),
            ("direct_dgrad_reorganize_rtc", json.dumps({"fold_k32": True})),
            ("direct_dgrad_transpose_rtc", json.dumps({})),
        ]
    )
    assert all((i.cpg, i.kpg) == (0, 0) for i in helpers)


@pytest.mark.parametrize(
    "groups,cpg,kpg,why",
    [
        # 640 dY channels: more K-atoms than any cached binary holds.
        (1, 48, 640, "K-atoms"),
        # 80 dX channels: five M-tiles, past the cached range.
        (1, 80, 192, "M-tiles"),
        # A ragged dX channel count: the epilogue stores 4 channels per lane.
        (1, 6, 64, "kpg % 4"),
    ],
)
def test_dgrad_rtc_rejects_what_no_binary_fits(cache, groups, cpg, kpg, why):
    problem = _problem(groups=groups, cpg=cpg, kpg=kpg)
    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    assert not _rtc(plans)
    reasons = [r for i, r in rejected if i.algorithm == "direct_dgrad_mfma_rtc"]
    assert reasons and any(why in r for r in reasons), reasons[:3]
