# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the direct-conv kernel cache: what gets built, which
cached binary is offered for which problem, and what a launch plan binds.

Nothing is compiled -- the cache is filled with placeholder HSACOs -- and no
GPU is touched.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common.kernel_cache import KernelCache
from kernels.common.conv_direct_grouped import DirectConvProblem
from rocke.core.arch import ArchTarget
from rocke.runtime.packing import compile_packer

_ARCH = "gfx950"


@pytest.fixture(scope="module")
def jobs():
    return dks.enumerate_direct_jobs(
        arch=_ARCH, target=ArchTarget.from_gfx(_ARCH), directions=("fwd", "dgrad")
    )


def _problem(**kw) -> DirectConvProblem:
    shape = dict(
        N=2, H=28, W=28, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
    )
    shape.update(kw)
    return DirectConvProblem(**shape)


def test_forward_entries_only_use_same_padding(jobs):
    """Forward kernels are wrong for any other padding, so none may be built."""
    fwd = [j.identity for j in jobs if j.identity.direction == "direct_fwd"]
    assert fwd
    assert all(i.pad_h == (i.filter_h - 1) // 2 for i in fwd)
    assert all(i.filter_h % 2 == 1 for i in fwd)


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
        c.put(job.identity, b"placeholder", {"kernel_name": job.identity.short_label()})
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


def test_nongrouped_only_serves_one_group(cache):
    plans, rejected = dks.direct_plans(cache, _problem(groups=16), "fwd", _ARCH)
    assert not _nongrouped(plans)
    assert all(i.algorithm != "direct_nongrouped" for i, _ in rejected)
    # Grouped forward has no stride-2 kernel; the non-grouped one does.
    plans, _ = dks.direct_plans(
        cache, _problem(groups=1, cpg=64, kpg=64, stride=2), "fwd", _ARCH
    )
    assert plans and _variants(plans) == {"direct_nongrouped"}


def test_capabilities_outside_the_list_are_not_served(cache):
    assert dks.direct_plans(cache, _problem(cpg=24, kpg=24), "fwd", _ARCH)[0] == []
    # Grouped forward stride 2 has no rocke kernel.
    assert dks.direct_plans(cache, _problem(stride=2), "fwd", _ARCH)[0] == []


def test_dgrad_stride1_offers_mfma_pipeline_and_scalar(cache):
    plans, _ = dks.direct_plans(cache, _problem(), "dgrad", _ARCH)
    assert _variants(plans) == {"direct_grouped_dgrad", "direct_grouped_dgrad_mfma"}
    for plan in plans:
        if plan.identity.algorithm != "direct_grouped_dgrad_mfma":
            continue
        # transpose, then the forward compute pass reading the transposed weights
        assert [s.buffers for s in plan.steps] == [
            ("w", None, "ws_t"),
            ("dy", "ws_t", "dx"),
        ]
        assert set(plan.workspaces) == {"ws_t"}


@pytest.mark.parametrize(
    "direction,problem",
    [
        ("fwd", _problem()),
        ("fwd", _problem(groups=1, cpg=128, kpg=96)),
        ("fwd", _problem(groups=64, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2)),
        ("dgrad", _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2)),
        ("dgrad", _problem(cpg=8, kpg=8, stride=2)),
        ("dgrad", _problem()),
    ],
)
def test_plans_pack_against_their_signatures(cache, direction, problem):
    plans, _ = dks.direct_plans(cache, problem, direction, _ARCH)
    assert plans
    names = ("x", "w", "y", "dy", "dx", "ws_t", "ws_coa")
    ptrs = {n: 0x1000 * (i + 1) for i, n in enumerate(names)}
    sizes = {n: 1 << 20 for n in names}
    for plan in plans:
        for step in plan.steps:
            compile_packer(step.signature)(dks.launch_values(step, ptrs, sizes))
            assert all(g > 0 for g in step.grid)
