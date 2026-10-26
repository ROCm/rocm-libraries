# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the direct-conv kernel cache: what gets built, which
cached binary is offered for which problem, and what a launch plan binds.

Nothing is compiled -- the cache is filled with placeholder HSACOs -- and no
GPU is touched.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import replace

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common.kernel_cache import KernelCache
from kernels.conv.direct_aot_jobs import _DENSE_MFMA_DGRAD_CHANNELS, _knob_grid
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_variants import (
    DIRECT_DGRAD_WMMA,
    DIRECT_WAVE_PARAMETRIC,
    direct_mfma_knobs_reason,
    direct_spec_for_problem,
)
from rocke.core.arch import ArchTarget
from rocke.runtime.packing import compile_packer

_ARCH = "gfx950"
# Variants the grid builds only for wave32 targets.
_WAVE32_ONLY = {DIRECT_DGRAD_WMMA}


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
        if variant in _WAVE32_ONLY:
            # No tuning for this arch at all (see the gfx1250 test below).
            assert not any(True for _ in _knob_grid(variant, _ARCH, "fp16"))
            continue
        for c in caps_list:
            assert (variant, c.KH, c.PAD, c.stride, c.cpg, c.kpg) in built, (variant, c)


def test_wave32_only_capability_rows_are_built_on_gfx1250():
    jobs = dks.enumerate_direct_jobs(
        arch="gfx1250",
        target=ArchTarget.from_gfx("gfx1250"),
        directions=("dgrad",),
    )
    built = {
        (j.identity.algorithm, j.identity.filter_h, j.identity.pad_h)
        for j in jobs
        if j.identity.algorithm in _WAVE32_ONLY
    }
    for variant in _WAVE32_ONLY:
        for c in dks.DIRECT_CAPABILITIES[variant][1]:
            assert (variant, c.KH, c.PAD) in built, (variant, c)


def test_identities_are_unique(jobs):
    hashes = [j.identity.stable_hash() for j in jobs]
    assert len(hashes) == len(set(hashes))


class _InMemoryEntriesCache(KernelCache):
    """A cache whose entries are held in memory, all linked to one placeholder
    binary on disk.

    Each entry's metadata is what :meth:`KernelCache.link` would write. One
    metadata file per job is over ten thousand small files, and writing then
    parsing them outlasts the per-test timeout on slow filesystems (Windows
    CI). The tests that build their own cache cover the on-disk layout.
    """

    def __init__(self, root, arch, identities):
        super().__init__(root, arch)
        key = hashlib.sha256(b"placeholder").hexdigest()
        # The binary's symbol is the first identity's, as when each is put.
        self._blob = self.put_blob(key, b"placeholder", identities[0].label())
        self._metas = {
            ident: {
                "identity": ident.to_dict(),
                "blob": key,
                "hsaco_bytes": self._blob.stat().st_size,
                "kernel_name": self.blob_kernel_name(key),
            }
            for ident in identities
        }

    def meta(self, identity):
        return self._metas.get(identity)

    def _load(self, subdir):
        # In metadata file-name order, as the on-disk walk yields them.
        return sorted(
            (
                (ident, self._blob, meta)
                for ident, meta in self._metas.items()
                if self._dir_for(ident).name == subdir
            ),
            key=lambda entry: entry[0].stable_hash(),
        )


@pytest.fixture(scope="module")
def cache(tmp_path_factory, jobs):
    """A cache holding every job with a placeholder binary.

    The tests that take it only read it (the ones that add or discard entries
    build their own), so the module shares one.
    """
    return _InMemoryEntriesCache(
        tmp_path_factory.mktemp("direct_cache"), _ARCH, [j.identity for j in jobs]
    )


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
    """Any group count is served; only tiles that do not divide it drop out.

    The LDS-halo kernels (dense, and grouped through merging) have shape rules
    of their own -- merge degree, chunk and tile fits -- so they are left out.
    """
    plans, rejected = dks.direct_plans(cache, _problem(groups=groups), "fwd", _ARCH)
    assert plans
    assert all(
        "block_groups" in why
        for ident, why in rejected
        if ident.algorithm not in dks._HALO_VARIANTS
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


_WIN = "direct_depthwise_dgrad_win"


def _win(plans):
    return [p for p in plans if p.identity.algorithm == _WIN]


@pytest.mark.parametrize("KH,PAD", [(3, 1), (7, 3), (5, 0), (11, 7)])
def test_windowed_depthwise_dgrad_serves_stride_1(cache, KH, PAD):
    """Stride-1 depthwise dgrad gets the windowed kernel beside the streaming
    one, at every padding; each plan is one dgrad-ABI launch whose grid is
    decoded from the runtime problem (row tiles of block_h, images folded by
    w_fold in the MFMA form)."""
    problem = _problem(N=3, H=13, W=17, groups=136, cpg=1, kpg=1, KH=KH, KW=KH, PAD=PAD)
    plans, _ = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    assert {_WIN, "direct_depthwise_dgrad"} <= _variants(plans)
    for plan in _win(plans):
        knobs = json.loads(plan.identity.knobs)
        (step,) = plan.steps
        assert step.buffers == ("dy", "w", "dx")
        assert [e["name"] for e in step.signature][:3] == ["A", "B", "D"]
        h_tiles = -(-problem.H // knobs["block_h"])
        if knobs.get("mfma"):
            fold = knobs["w_fold"]
            assert step.grid == (
                -(-problem.W // (32 // fold)),
                -(-problem.groups // (8 * knobs["block_waves"])),
                -(-problem.N // fold) * h_tiles,
            )
        else:
            assert step.grid == (
                -(-problem.W // knobs["block_w"]),
                -(
                    -problem.groups
                    // (64 * knobs["block_waves"] * knobs["ch_per_lane"])
                ),
                problem.N * h_tiles,
            )


def test_windowed_depthwise_dgrad_is_stride_1_only(cache):
    problem = _problem(groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=2, stride=2)
    plans, _ = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    assert plans and not _win(plans)


def test_windowed_depthwise_dgrad_cached_scope(jobs):
    """The cache holds the knob image of the tile policy, not a product: dot2
    exactly where it pairs taps, channel pairs only for 3x3, the MFMA form
    only for "same"-padded 5x5..9x9, and every tile within the unroll budget."""
    win = [j for j in jobs if j.identity.algorithm == _WIN]
    assert win
    mfma = [j for j in win if j.spec_kwargs.get("mfma")]
    valu = [j for j in win if not j.spec_kwargs.get("mfma")]
    assert {(j.identity.filter_h, j.identity.pad_h) for j in mfma} == {
        (k, (k - 1) // 2) for k in dks.DW_WIN_MFMA_FILTERS
    }
    for j in valu:
        kw, knobs = j.identity.filter_w, j.spec_kwargs
        assert knobs["dot2"] == (kw >= 5)
        assert knobs["ch_per_lane"] == 1 or kw == 3
        fmas = knobs["block_h"] * kw * kw * knobs["block_w"] * knobs["ch_per_lane"]
        assert fmas <= dks.DW_WIN_UNROLL_BUDGET
    # Every capability row has a binary (the smallest tile fits any filter).
    assert {(j.identity.filter_h, j.identity.pad_h) for j in valu} == {
        (c.KH, c.PAD) for c in dks.DIRECT_CAPABILITIES[_WIN][1]
    }


def test_windowed_depthwise_dgrad_mfma_needs_groups_multiple_of_8(cache):
    problem = _problem(groups=60, cpg=1, kpg=1, KH=7, KW=7, PAD=3)
    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    assert _win(plans) and not any(
        json.loads(p.identity.knobs).get("mfma") for p in _win(plans)
    )
    assert any(
        "groups % 8" in why
        for ident, why in rejected
        if ident.algorithm == _WIN and json.loads(ident.knobs).get("mfma")
    )


def test_windowed_depthwise_dgrad_binary_is_found_by_identity():
    """direct_job_for_identity rebuilds a windowed job from its identity alone
    (both forms), as --compile-all does for a named kernel."""
    target = ArchTarget.from_gfx(_ARCH)
    cells = [
        job
        for caps in dks.DIRECT_CAPABILITIES[_WIN][1]
        if (caps.KH, caps.PAD) == (7, 3)
        for _, job in dks._caps_jobs(
            _ARCH, target.wave_size, _WIN, "dgrad", caps, "bf16"
        )[1]
    ]
    assert any(j.spec_kwargs.get("mfma") for j in cells)
    for job in cells:
        assert dks.direct_job_for_identity(job.identity).identity == job.identity


_4C = "direct_grouped_dgrad_4c"
_FUSED = "direct_grouped_dgrad_fused"


@pytest.mark.parametrize("KH", [1, 3])
def test_dgrad_cpg4_offers_the_4c_kernel(cache, KH):
    """A stride-1 "same"-padded cpg = kpg = 4 dgrad gets the 4c kernel: one
    forward-ABI launch over the transposed problem that reads the original
    weights itself (no transform helper, no workspace), with the 4c grid
    (column tiles, group blocks, images). 3x3 also keeps the existing
    variants; a 1x1 has only the 4c one."""
    problem = _problem(
        N=3, H=13, W=17, groups=64, cpg=4, kpg=4, KH=KH, KW=KH, PAD=KH // 2
    )
    plans, _ = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    expected = {_4C, _FUSED}
    if KH == 3:
        expected |= {
            "direct_grouped_dgrad",
            "direct_grouped_dgrad_mfma",
            "direct_dgrad_mfma_rtc",
        }
    assert _variants(plans) == expected
    fourc = [p for p in plans if p.identity.algorithm == _4C]
    assert fourc
    for plan in fourc:
        knobs = json.loads(plan.identity.knobs)
        (step,) = plan.steps
        assert plan.workspaces == {}
        assert step.buffers == ("dy", "w", "dx")
        # Forward ABI: the A/D stride triples are the forward ones.
        names = [e["name"] for e in step.signature]
        assert "p_A_stride_n" in names and "p_dY_stride_n" not in names
        threads = (knobs["block_groups"] // 16) * 64
        if knobs["stage_rows"]:
            threads *= knobs["block_q"] // 4
        assert step.block == (threads, 1, 1)
        assert step.grid == (
            -(-problem.W // knobs["block_q"]),
            problem.groups // knobs["block_groups"],
            problem.N,
        )
        # The transposed problem: dY in, dX out (cpg and kpg swap).
        assert step.conv_args.to_launch_values(1, 2, 3, 4, 5, 6)["p_Hi"] == problem.H


def test_dgrad_cpg4_4c_cached_scope(jobs):
    """The 4c dgrad cache is the policy image: block_q in {4, 8, 16}, block
    groups in {16, 32, 64}, gathered weights, LDS-staged weights and the
    row-staged form (one q wave per 4 columns); 1x1 and 3x3, stride 1, "same"
    padding, cpg = kpg = 4."""
    fourc = [j for j in jobs if j.identity.algorithm == _4C]
    assert {
        (j.identity.filter_h, j.identity.pad_h, j.identity.stride_h) for j in fourc
    } == {(1, 0, 1), (3, 1, 1)}
    assert {(j.identity.cpg, j.identity.kpg) for j in fourc} == {(4, 4)}
    knobs = {json.dumps(j.spec_kwargs, sort_keys=True) for j in fourc}
    assert len(knobs) == len(dks.DGRAD_4C_BLOCK_Q) * len(
        dks.DGRAD_4C_BLOCK_GROUPS
    ) * len(dks.DGRAD_4C_FORMS)
    assert any(j.spec_kwargs["stage_rows"] for j in fourc)


def test_dgrad_cpg4_4c_rejects_shapes_it_cannot_serve(cache):
    """groups must divide block_groups; the rejected binaries say why."""
    problem = _problem(groups=48, cpg=4, kpg=4)
    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    fourc = [p for p in plans if p.identity.algorithm == _4C]
    assert fourc
    assert all(json.loads(p.identity.knobs)["block_groups"] == 16 for p in fourc)
    assert any(
        "not divisible by block_groups" in why
        for ident, why in rejected
        if ident.algorithm == _4C
    )
    # No 4c kernel for stride 2 or a 5x5 filter (no such capability row).
    for kw in ({"stride": 2}, {"KH": 5, "KW": 5, "PAD": 2}):
        plans, _ = dks.direct_plans(
            cache, _problem(groups=64, cpg=4, kpg=4, **kw), "dgrad", _ARCH
        )
        assert _4C not in _variants(plans)


def test_dgrad_cpg4_4c_lds_forms_are_gfx950_only():
    """gfx942 has no transpose LDS reads: its 4c cache is the gathered form."""
    target = ArchTarget.from_gfx("gfx942")
    for caps in dks.DIRECT_CAPABILITIES[_4C][1]:
        cell = dks._caps_jobs("gfx942", target.wave_size, _4C, "dgrad", caps, "fp16")[1]
        assert cell
        assert not any(j.spec_kwargs["dgrad_weights_lds"] for _, j in cell)


def test_dgrad_cpg4_4c_binary_is_found_by_identity():
    target = ArchTarget.from_gfx(_ARCH)
    for caps in dks.DIRECT_CAPABILITIES[_4C][1]:
        for _, job in dks._caps_jobs(
            _ARCH, target.wave_size, _4C, "dgrad", caps, "bf16"
        )[1]:
            assert dks.direct_job_for_identity(job.identity).identity == job.identity


def _fused_knobs(job_or_plan):
    return json.loads(job_or_plan.identity.knobs)


@pytest.mark.parametrize(
    "KH,cpg,kpg", [(3, 16, 16), (3, 12, 28), (1, 32, 8), (5, 8, 20), (7, 16, 4)]
)
def test_dgrad_fused_offers_one_forward_kernel(cache, KH, cpg, kpg):
    """A stride-1 "same"-padded dgrad with channel counts in 4-channel steps
    up to 32 (cpg != kpg too) gets the fused generic kernel: one forward-ABI
    launch over the transposed problem, binding the original weights (no
    transform helper, no workspace), with the generic grid over the
    transposed problem's tiles."""
    problem = _problem(
        N=3, H=13, W=17, groups=8, cpg=cpg, kpg=kpg, KH=KH, KW=KH, PAD=KH // 2
    )
    plans, _ = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    fused = [p for p in plans if p.identity.algorithm == _FUSED]
    assert fused
    for plan in fused:
        knobs = _fused_knobs(plan)
        (step,) = plan.steps
        assert plan.workspaces == {}
        assert step.buffers == ("dy", "w", "dx")
        names = [e["name"] for e in step.signature]
        assert "p_A_stride_n" in names and "p_dY_stride_n" not in names
        assert step.block == (knobs["block_groups"] * 64, 1, 1)
        assert step.grid == (
            -(-problem.W // knobs["block_q"]),
            problem.groups // knobs["block_groups"],
            problem.N * -(-problem.H // knobs["block_h"]),
        )
        values = step.conv_args.to_launch_values(1, 2, 3, 4, 5, 6)
        # The transposed problem: dY in (kpg channels), dX out (cpg).
        assert values["p_total_c"] == problem.groups * kpg
        assert values["p_total_k"] == problem.groups * cpg


def test_dgrad_fused_cached_scope(jobs):
    """The fused cache is the generic dgrad policy's image: 16-column strips
    (32 only on 4/8-row tiles with wide channel groups), block_groups at or
    below the per-channel rule (halved for 5x5/7x7), 4..32-row tiles (4/8 for
    5x5/7x7), every tile with the row guard, fold_k32 exactly where kpg
    allows it, LDS-staged weights on gfx950 and the waves-per-EU hint only
    for the 16-wide atom forms with a small weight footprint."""
    fused = [j for j in jobs if j.identity.algorithm == _FUSED]
    assert {(j.identity.filter_h, j.identity.stride_h) for j in fused} == {
        (k, 1) for k in dks.DGRAD_FUSED_FILTERS
    }
    assert all(j.identity.pad_h == j.identity.filter_h // 2 for j in fused)
    chans = set(dks.DGRAD_FUSED_CHANNELS)
    assert {(j.identity.cpg, j.identity.kpg) for j in fused} == {
        (c, k) for c in chans for k in chans
    }
    assert any(j.identity.cpg != j.identity.kpg for j in fused)
    for j in fused:
        knobs, i = j.spec_kwargs, j.identity
        caps = dks._caps_of(i)
        assert knobs["block_groups"] <= dks._dgrad_fused_rule_block_groups(caps)
        if i.filter_h >= 5:
            assert knobs["block_h"] in dks.DGRAD_FUSED_BLOCK_H_LARGE
        else:
            assert knobs["block_h"] in dks.DGRAD_FUSED_BLOCK_H
        if knobs["block_q"] == 32:
            assert knobs["block_h"] in (4, 8)
        # Whole-image tiles leave the workgroup once the image is done.
        assert knobs.get("row_exit", False) == (
            knobs["block_h"] in dks.DGRAD_FUSED_EXIT_BLOCK_H
        )
        assert knobs["fold_k32"] == (i.kpg % 32 == 0)
        assert knobs["dgrad_weights_lds"]
        if knobs["fold_k32"]:
            assert knobs["waves_per_eu"] == 0
        if "prefetch_rows" in knobs:
            # The row-stream stack: square groups on 16-column strips, the
            # two-row prefetch, LDS-only barrier and XCD order, with the
            # column pad the staged row's width calls for (or none).
            assert i.cpg == i.kpg and knobs["block_q"] == 16
            assert knobs["prefetch_rows"] == 2
            assert knobs["lds_only_sync"] and knobs["xcd_tiles"]
            assert knobs["lds_pad"] in (0, dks.DGRAD_FUSED_LDS_PAD)
            assert not (knobs["stage_out"] and knobs["lds_pad"] == 0) or (
                knobs["block_groups"] * i.kpg * 2 % 32 != 0
            )
    assert any("prefetch_rows" in j.spec_kwargs for j in fused)
    # Low thousands per arch and dtype, not a product of every knob.
    assert len(fused) < 4000


def test_dgrad_fused_cached_scope_gfx942():
    """gfx942 has neither transpose LDS reads nor the 16x16x32 atom: its
    fused cache takes the gathered weights and 16-wide atoms throughout, and
    no row-stream stack (the XCD order is gfx950-only)."""
    target = ArchTarget.from_gfx("gfx942")
    for caps in dks.DIRECT_CAPABILITIES[_FUSED][1]:
        cell = dks._caps_jobs(
            "gfx942", target.wave_size, _FUSED, "dgrad", caps, "fp16"
        )[1]
        assert cell, caps
        for _, j in cell:
            assert not j.spec_kwargs["dgrad_weights_lds"]
            assert not j.spec_kwargs["fold_k32"]
            # The XCD row-band order is built for gfx950 only, and the
            # row-stream stack needs it.
            assert "prefetch_rows" not in j.spec_kwargs


def test_dgrad_fused_rows_fit_the_weight_register_budget():
    """A 5x5 / 7x7 row is listed only where its fused weights fit the
    preload register budget at some knob (the rest is the pre-pass's)."""
    from kernels.common.conv_direct_grouped import PRELOAD_WEIGHT_VGPR_BUDGET

    rows = dks.DIRECT_CAPABILITIES[_FUSED][1]
    assert {(c.KH, c.cpg, c.kpg) for c in rows if c.KH == 7} == {
        (7, c, k) for c in (4, 8, 12, 16) for k in (4, 8, 12, 16)
    }
    for c in rows:
        regs = c.KH * c.KW * -(-c.kpg // 16) * -(-c.cpg // 16) * 2
        assert regs <= PRELOAD_WEIGHT_VGPR_BUDGET


def test_dgrad_fused_rejects_shapes_it_cannot_serve(cache):
    """groups must divide block_groups (plan_for re-validates on the real
    shape); stride 2, a channel count off the 4-channel grid and a 9x9 filter
    have no row."""
    problem = _problem(groups=3, cpg=8, kpg=8)
    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    fused = [p for p in plans if p.identity.algorithm == _FUSED]
    assert fused
    assert all(_fused_knobs(p)["block_groups"] == 1 for p in fused)
    assert any(
        "not divisible by block_groups" in why
        for ident, why in rejected
        if ident.algorithm == _FUSED
    )
    for kw in (
        {"stride": 2},
        {"cpg": 6, "kpg": 6},
        {"cpg": 64, "kpg": 64},
        {"KH": 9, "KW": 9, "PAD": 4},
        {"KH": 3, "KW": 3, "PAD": 0},
    ):
        plans, _ = dks.direct_plans(cache, _problem(**kw), "dgrad", _ARCH)
        assert _FUSED not in _variants(plans), kw


def test_dgrad_fused_binary_is_found_by_identity():
    target = ArchTarget.from_gfx(_ARCH)
    rows = dks.DIRECT_CAPABILITIES[_FUSED][1]
    for caps in rows[::7]:
        for _, job in dks._caps_jobs(
            _ARCH, target.wave_size, _FUSED, "dgrad", caps, "bf16"
        )[1]:
            assert dks.direct_job_for_identity(job.identity).identity == job.identity


def _generic_dgrad_policy_knobs(p: DirectConvProblem) -> dict:
    """The knobs the generic direct-MFMA dgrad policy picks for ``p`` (the
    fused, single-kernel form on gfx950), with a whole-image pick
    (``block_h = 0``) mapped to the cached tile height that serves it.

    A test-side statement of the policy so the enumeration can be checked
    against it until the policy itself lands in dispatch: per-channel
    block_groups (halved for 5x5/7x7, then halved until it divides groups),
    whole image or 8/4-row tiles, 32-column strips on tiled images with wide
    channel groups and enough waves, fold_k32 where kpg allows it, LDS-staged
    weights and the waves-per-EU hint on small 16-wide-atom footprints.
    """
    from kernels.common import conv_direct_grouped as dc

    caps = dks.DirectCaps(KH=p.KH, PAD=p.PAD, stride=1, cpg=p.cpg, kpg=p.kpg)
    # The short-image cost model scales by the per-channel row (unhalved).
    rule_bg = dks._dgrad_fused_rule_block_groups(replace(caps, KH=3))
    bg = dks._dgrad_fused_rule_block_groups(caps)
    while p.groups % bg:
        bg //= 2
    q_cols = -(-p.Wo // 16) * p.groups * p.N
    if p.H <= 8:
        bh = 0
    elif p.KH >= 5:
        bh = 4
    elif p.H <= 16:
        slots = 1024 * rule_bg
        costs = [
            ((h or p.H) + p.KH - 1) * max(slots, q_cols * (-(-p.H // h) if h else 1))
            for h in (0, 8, 4)
        ]
        bh = (0, 8, 4)[costs.index(min(costs))]
    else:
        bh = 4
        for h in (0, 8) if p.H <= 64 else (8,):
            if q_cols * (-(-p.H // h) if h else 1) >= 3072:
                bh = h
                break
    bq = 16
    chans = max(p.cpg, p.kpg)
    if (
        bh
        and -(-p.Wo // 32) * 32 <= -(-p.Wo // 16) * 16
        and (chans >= 16 or (p.KH >= 5 and chans >= 8))
        and -(-p.Wo // 32) * p.groups * p.N * -(-p.H // bh) >= 768
    ):
        bq = 32
    k32 = p.kpg % 32 == 0
    knobs = {
        "block_q": bq,
        "block_groups": bg,
        "block_h": bh or dc.dgrad_fused_block_h(p.H),
        "fold_k32": k32,
        "dgrad_weights_lds": True,
        "waves_per_eu": 0,
    }
    if knobs["block_h"] in dks.DGRAD_FUSED_EXIT_BLOCK_H:
        knobs["row_exit"] = True
    if not k32:
        main = dks.make_spec(_FUSED, p, knobs)
        if dc.preload_weight_vgprs(main) <= dks.DGRAD_FUSED_WPE_MAX_VGPRS:
            knobs["waves_per_eu"] = dks.DGRAD_FUSED_WPE
    return _generic_dgrad_stream_knobs(p, knobs, untiled=bh == 0)


def _fused_valid(p: DirectConvProblem, knobs: dict) -> bool:
    return dks.validate_spec(_FUSED, dks.make_spec(_FUSED, p, knobs), _ARCH)[0]


def _generic_dgrad_stream_knobs(p: DirectConvProblem, knobs: dict, untiled: bool):
    """The policy's row-stream stack on top of ``knobs`` (see
    _generic_dgrad_policy_knobs): square channel groups on 16-column strips
    in its measured box, not where waves_m = 2 is valid, only with the XCD
    row-band order valid; then the first of (pad, staged output), (pad,
    unstaged), (no pad, unstaged) that adds no launch rounds."""
    from kernels.common import conv_direct_grouped as dc

    if p.cpg != p.kpg or knobs["block_q"] != 16:
        return knobs
    if (p.Wo > 64) if untiled else (p.KH == 1 and p.cpg < 12):
        return knobs
    base = {**knobs, "prefetch_rows": 2, "lds_only_sync": True}
    if not _fused_valid(p, base) or _fused_valid(p, {**base, "waves_m": 2}):
        return knobs

    def rounds(kn):
        spec = dks.make_spec(_FUSED, p, kn)
        target = ArchTarget.from_gfx(_ARCH)
        per_cu = min(
            target.lds_capacity_bytes // dc.direct_conv_lds_bytes(spec),
            32 // (spec.threads_per_block // target.wave_size),
        )
        gx, gy, gz = dc.direct_conv_grid(spec)
        return -(-(gx * gy * gz) // (max(per_cu, 1) * 256))

    floor = rounds(base)
    stack = {**base, "xcd_tiles": True, "lds_pad": 0, "stage_out": False}
    if not _fused_valid(p, stack):
        return knobs
    pad = 8 if knobs["block_groups"] * p.kpg * 2 % 32 == 0 else 0
    for lds_pad, stage_out in ((pad, True), (pad, False)):
        cand = {**stack, "lds_pad": lds_pad, "stage_out": stage_out}
        if cand != stack and _fused_valid(p, cand) and rounds(cand) <= floor:
            return cand
    return stack


def test_dgrad_fused_policy_image_is_enumerated():
    """Every pick of the generic direct dgrad policy on a spread of grouped
    stride-1 problems is a cached identity of the fused variant, and that
    binary is valid on the problem (the guard that keeps the enumeration the
    policy's image: dispatch can only pick what was built)."""
    import random

    target = ArchTarget.from_gfx(_ARCH)
    rows = set(dks.DIRECT_CAPABILITIES[_FUSED][1])
    cached: dict = {}
    rng = random.Random(1)
    checked = streamed = 0
    for _ in range(300):
        KH = rng.choice((1, 3, 5, 7))
        p = _problem(
            N=rng.choice((1, 2, 8, 64)),
            H=rng.choice((3, 7, 8, 9, 12, 13, 15, 17, 28, 50, 64, 70)),
            W=rng.choice((5, 14, 17, 33, 64)),
            groups=rng.choice((1, 2, 3, 6, 8, 24)),
            cpg=rng.choice(dks.DGRAD_FUSED_CHANNELS),
            kpg=rng.choice(dks.DGRAD_FUSED_CHANNELS),
            KH=KH,
            KW=KH,
            PAD=KH // 2,
        )
        caps = dks.DirectCaps(KH=KH, PAD=KH // 2, stride=1, cpg=p.cpg, kpg=p.kpg)
        if caps not in rows:
            continue  # past the fused register budget: the pre-pass serves it
        if caps not in cached:
            cached[caps] = [
                json.loads(job.identity.knobs)
                for _, job in dks._caps_jobs(
                    _ARCH, target.wave_size, _FUSED, "dgrad", caps, "bf16"
                )[1]
            ]
        knobs = _generic_dgrad_policy_knobs(p)
        streamed += "prefetch_rows" in knobs
        assert knobs in cached[caps], (p, knobs)
        spec = dks.make_spec(_FUSED, p, knobs)
        assert dks.validate_spec(_FUSED, spec, _ARCH)[0], (p, knobs)
        checked += 1
    assert checked > 100
    assert streamed > 10


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
        if spec.split_c == 1:
            (step,) = plan.steps
            assert step.grid == spec.grid()
            continue
        # Split-C: the conv pass fills the fp32 workspace, the reduction
        # folds it into y.
        conv, fold = plan.steps
        assert conv.grid == spec.grid()
        assert conv.buffers == ("x", "w", "ws_split")
        assert fold.buffers == ("ws_split", "w", "y")
        assert plan.workspaces == {"ws_split": spec.workspace_bytes}


def test_every_split_kernel_has_its_reduction(jobs):
    """A split-C kernel is useless without the reduction of its degree, so the
    grid builds one per degree it uses -- and only those."""
    degrees = {
        json.loads(j.identity.knobs).get("split_c", 1)
        for j in jobs
        if j.identity.algorithm == "direct_nongrouped"
    } - {1}
    assert degrees
    reduces = [j for j in jobs if j.identity.algorithm == "direct_split_reduce"]
    assert {json.loads(j.identity.knobs)["split_c"] for j in reduces} == degrees
    for job in reduces:
        assert dks.direct_job_for_identity(job.identity) == job


def test_grouped_halo_serves_grouped_problems_only(cache):
    """The grouped LDS-halo kernel is offered to grouped problems, with merge
    degrees that make the merged channels fit its chunks; a dense problem gets
    the non-grouped kernel instead."""
    problem = _problem(groups=32, cpg=4, kpg=4, H=56, W=56)
    plans = dks.direct_plans(cache, problem, "fwd", _ARCH)[0]
    halo = [p for p in plans if p.identity.algorithm == "direct_grouped_halo"]
    assert halo
    for plan in halo:
        knobs = json.loads(plan.identity.knobs)
        spec = dks.make_spec("direct_grouped_halo", problem, knobs)
        assert (problem.cpg * knobs["group_merge"]) % spec.ck == 0
        assert spec.tile_k <= spec.merged_k
        (step,) = plan.steps
        assert step.grid == spec.grid()
    dense = dks.direct_plans(cache, _problem(groups=1, cpg=64, kpg=64), "fwd", _ARCH)[0]
    assert "direct_grouped_halo" not in _variants(dense)


def test_large_batch_is_split_into_launches(cache):
    """A batch whose activations pass the kernels' 32-bit buffer range runs as
    balanced per-chunk launches over windows of the same buffers."""
    problem = _problem(N=8, H=768, W=768, groups=1, cpg=256, kpg=128)
    assert dks._batch_chunks(problem) == [4, 4]
    assert dks._batch_chunks(replace(problem, N=2)) is None
    plans = _nongrouped(dks.direct_plans(cache, problem, "fwd", _ARCH)[0])
    assert plans
    in_img = 2 * 768 * 768 * 256
    out_img = 2 * 768 * 768 * 128
    for plan in plans:
        assert json.loads(plan.identity.knobs).get("split_c", 1) == 1
        assert [st.views for st in plan.steps] == [
            ((0, 4 * in_img), None, (0, 4 * out_img)),
            ((4 * in_img, 4 * in_img), None, (4 * out_img, 4 * out_img)),
        ]
        values = dks.launch_values(
            plan.steps[1], {"x": 0x1000, "w": 0x2000, "y": 0x3000}, {"w": 1 << 20}
        )
        assert values["A"] == 0x1000 + 4 * in_img
        assert values["A_bytes"] == 4 * in_img
        assert values["p_N"] == 4


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


def _code_object(text: bytes, scratch: int = 0, entry: int = 0) -> bytes:
    """A minimal ELF64 code object: ``.text`` and one 64-byte ``k.kd``."""
    import struct

    kd = bytearray(64)
    struct.pack_into("<I", kd, 4, scratch)
    struct.pack_into("<q", kd, 16, entry)
    kd_addr = 0x1000
    strtab = b"\0k.kd\0"
    symtab = bytes(24) + struct.pack("<IBBHQQ", 1, 0x11, 0, 2, kd_addr, 64)
    names = [b".text", b".rodata", b".symtab", b".strtab", b".shstrtab"]
    shstrtab = b"\0" + b"".join(n + b"\0" for n in names)
    name_off = [1 + sum(len(n) + 1 for n in names[:i]) for i in range(len(names))]
    blobs = [text, bytes(kd), symtab, strtab, shstrtab]
    # (type, addr, link): .text, .rodata, .symtab -> .strtab, .strtab, .shstrtab
    kinds = [(1, 0, 0), (1, kd_addr, 0), (2, 0, 4), (3, 0, 0), (3, 0, 0)]
    body, offsets = b"", []
    for blob in blobs:
        offsets.append(64 + len(body))
        body += blob
    shoff = 64 + len(body)
    sections = bytes(64)
    for (sh_type, addr, link), off, blob, name in zip(kinds, offsets, blobs, name_off):
        sections += struct.pack(
            "<IIQQQQIIQQ", name, sh_type, 0, addr, off, len(blob), link, 0, 1, 0
        )
    header = b"\x7fELF\x02\x01\x01" + bytes(9)
    header += struct.pack(
        "<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 5
    )
    return header + body + sections


def test_code_facts_read_text_descriptor_and_scratch():
    code, scratch = dks._elf_code_facts(_code_object(b"\x01\x02", scratch=12))
    assert scratch == 12
    assert len(code) == 32  # a SHA-256 digest, not the code itself
    # The entry-point offset is file layout, not code.
    assert dks._elf_code_facts(_code_object(b"\x01\x02", entry=256))[0] == (
        dks._elf_code_facts(_code_object(b"\x01\x02"))[0]
    )
    assert dks._elf_code_facts(_code_object(b"\x01\x03"))[0] != code
    assert dks._elf_code_facts(b"placeholder") is None


def _hint_pairs(jobs, kind):
    """``(waves_per_eu=None, waves_per_eu=3)`` identity pairs of one kernel kind:
    ``halo`` (grouped), ``split`` (split-C) or ``dense`` (unsplit non-grouped)."""
    by_key = {}
    for job in jobs:
        ident = job.identity
        if ident.algorithm not in dks._HALO_VARIANTS:
            continue
        knobs = json.loads(ident.knobs)
        if ident.algorithm == "direct_grouped_halo":
            this = "halo"
        else:
            this = "split" if knobs.get("split_c", 1) > 1 else "dense"
        if this != kind:
            continue
        we = knobs.pop("waves_per_eu", None)
        key = replace(ident, knobs=json.dumps(knobs, sort_keys=True))
        by_key.setdefault(key, {})[we] = ident
    return [(p[None], p[3]) for p in by_key.values() if set(p) == {None, 3}]


def test_prune_drops_spilling_and_hint_duplicate_new_entries(tmp_path, jobs):
    """A grouped-halo or split-C waves_per_eu=3 entry whose code equals its
    default twin's goes, as does one that spills; a hint that changed the code
    stays, entries without an ELF binary are left alone, and the dense unsplit
    grid is never touched."""
    (same_none, same_3), (diff_none, diff_3) = _hint_pairs(jobs, "halo")[:2]
    (spill_none, spill_3), (split_none, split_3) = _hint_pairs(jobs, "split")[:2]
    (dense_none, dense_3) = _hint_pairs(jobs, "dense")[0]
    c = KernelCache(tmp_path, _ARCH)
    binaries = {
        same_none: _code_object(b"\x01"),
        same_3: _code_object(b"\x01", entry=64),
        diff_none: _code_object(b"\x02"),
        diff_3: _code_object(b"\x03"),
        spill_none: _code_object(b"\x04", scratch=8),
        spill_3: b"placeholder",
        split_none: _code_object(b"\x05"),
        split_3: _code_object(b"\x05"),
        dense_none: _code_object(b"\x06", scratch=8),
        dense_3: _code_object(b"\x06", scratch=8),
    }
    for ident, blob in binaries.items():
        c.put(ident, blob, {"kernel_name": ident.label()})
    assert dks.prune_halo_entries(c, jobs, log=lambda _: None) == 3
    assert {i for i, _ in c.list_all("direct_fwd")} == {
        same_none,
        diff_none,
        diff_3,
        spill_3,
        split_none,
        dense_none,
        dense_3,
    }


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
    # The baked-channel grouped kernels cover cpg in {4, 8, 16, 32} at stride
    # 1; anything else grouped is left to the runtime-channel LDS-halo kernel.
    for problem in (_problem(cpg=24, kpg=24), _problem(stride=2)):
        plans = dks.direct_plans(cache, problem, "fwd", _ARCH)[0]
        assert plans
        assert _variants(plans) == {"direct_grouped_halo"}
    # A filter no fwd row carries has no kernel at all.
    assert dks.direct_plans(cache, _problem(KH=5, KW=5, PAD=2), "fwd", _ARCH)[0] == []


def test_dgrad_stride1_offers_mfma_pipeline_and_scalar(cache):
    plans, _ = dks.direct_plans(cache, _problem(), "dgrad", _ARCH)
    # cpg = kpg = 16 has the fixed-channel MFMA entries and also the
    # runtime-channel ones (one K-atom, one M-tile), and the single-kernel
    # fused-weight generic dgrad.
    assert _variants(plans) == {
        "direct_grouped_dgrad",
        "direct_grouped_dgrad_mfma",
        "direct_dgrad_mfma_rtc",
        _FUSED,
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


def test_dgrad_rtc_grid_skips_wide_q_heavy_preloads(jobs):
    """Runtime-channel preloading binaries with two Q subtiles per wave past
    the register cut computed a wrong dX on gfx950; none is built, and the
    plan check refuses them on an existing cache too."""
    built = [
        json.loads(j.identity.knobs)
        for j in jobs
        if j.identity.algorithm == "direct_dgrad_mfma_rtc"
    ]
    assert any(k["block_q"] == 32 and k["waves_k"] > 1 for k in built)
    for k in built:
        if k["block_q"] == 32 and k["waves_k"] > 1:
            assert dks.dgrad_rtc_vgprs(k, 3, 3) <= 216, k

    # The two binaries that failed verification on a dense 16 -> 512 bf16
    # dgrad are refused for it; their block_q 16 twins serve it.
    problem = _problem(groups=1, cpg=16, kpg=512, dtype="bf16")
    for block_h in (8, 16):
        knobs = dict(
            block_q=32,
            block_groups=1,
            block_h=block_h,
            waves_q=1,
            waves_k=2,
            runtime_k_loop=False,
            persistent_grid=False,
            fold_k32=True,
            k_atoms=16,
            m_tiles=1,
        )
        assert "wrong dX" in direct_mfma_knobs_reason(problem, knobs)
        spec, why = direct_spec_for_problem(
            "direct_dgrad_mfma_rtc", problem, knobs, _ARCH
        )
        assert spec is None and "wrong dX" in why
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
        # Stride-1 depthwise: the windowed kernel (both forms) as well.
        ("dgrad", _problem(groups=64, cpg=1, kpg=1, KH=7, KW=7, PAD=3)),
        ("dgrad", _problem(cpg=8, kpg=8, stride=2)),
        ("dgrad", _problem()),
        # cpg = kpg = 4: the 4c kernel (forward ABI, transposed problem) too.
        ("dgrad", _problem(groups=64, cpg=4, kpg=4)),
        # The fused generic dgrad: cpg != kpg, 5x5 and 1x1.
        ("dgrad", _problem(groups=6, cpg=20, kpg=8, KH=5, KW=5, PAD=2)),
        ("dgrad", _problem(groups=4, cpg=12, kpg=32, KH=1, KW=1, PAD=0)),
        ("dgrad", _problem(groups=32, cpg=4, kpg=4, KH=1, KW=1, PAD=0)),
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
    names = ("x", "w", "y", "dy", "dx", "dw", "ws_t", "ws_coa", "ws_split")
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
            # The preload budget the enumerator also applies: past it comgr's
            # AGPR spilling has miscomputed dX.
            or direct_mfma_knobs_reason(problem, knobs)
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


@pytest.mark.parametrize(
    "variant,problem",
    [
        (
            "direct_wgrad_dw",
            _problem(groups=192, cpg=1, kpg=1, stride=2, H=30, W=40),
        ),
        (
            "direct_wgrad_dw_cols",
            _problem(groups=3, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2, H=40, W=64),
        ),
        # A PAD 0 row: built only since the probe picks a width the kernel takes.
        ("direct_wgrad_dw_cols", _problem(groups=4, cpg=1, kpg=1, PAD=0, W=34)),
        (
            "direct_wgrad_gather",
            _problem(groups=1, cpg=3, kpg=24, stride=2, H=32, W=64),
        ),
        (
            "direct_wgrad_gather",
            _problem(groups=1, cpg=24, kpg=96, stride=2, H=30, W=40),
        ),
        (
            "direct_wgrad_dw_mfma",
            _problem(groups=192, cpg=1, kpg=1, KH=7, KW=7, PAD=3, H=56, W=56),
        ),
        # Small images: the packed-image binaries join in.
        (
            "direct_wgrad_dw_mfma",
            _problem(N=128, groups=1024, cpg=1, kpg=1, KH=7, KW=7, PAD=3, H=7, W=7),
        ),
        ("direct_wgrad_grouped_mfma", _problem(groups=32, cpg=4, kpg=4, H=56, W=56)),
        ("direct_wgrad_grouped_mfma", _problem(groups=32, cpg=8, kpg=8, H=56, W=56)),
    ],
)
def test_folded_wgrad_plans_run_the_pass_then_the_fold(cache, variant, problem):
    """A partial-slot wgrad is a two-step plan: the pass writes one slot per
    partition into the scratch ``ws`` and the fold sums them into dW. Both
    steps, and the scratch size, must agree on the partition and channel
    counts of the problem actually launched."""
    plans, _ = dks.direct_plans(cache, problem, "wgrad", _ARCH)
    plans = [p for p in plans if p.identity.algorithm == variant]
    assert plans
    names = ("dy", "x", "dw", "ws")
    ptrs = {n: 0x1000 * (i + 1) for i, n in enumerate(names)}
    sizes = {n: 1 << 20 for n in names}
    for plan in plans:
        main, fold = plan.steps
        assert main.buffers == ("dy", "x", "ws")
        assert fold.buffers == ("ws", None, "dw")
        spec = dks.make_spec(variant, problem, json.loads(plan.identity.knobs))
        assert (main.grid, main.block) == spec.launch_geometry()
        if hasattr(spec, "fold_channels"):
            channels, taps = spec.fold_channels(), spec.fold_taps()
        else:
            channels, taps = problem.groups, problem.KH * problem.KW
        assert dict(fold.scalars) == {
            "parts": spec.n_partitions(),
            "channels": channels,
        }
        assert (
            plan.workspaces
            == {"ws": spec.n_partitions() * taps * channels * 4}
            == {"ws": spec.workspace_bytes()}
        )
        # The fold's 64-output blocks cover every element of dW.
        assert fold.grid[0] * 64 >= channels * taps
        for step in plan.steps:
            compile_packer(step.signature)(dks.launch_values(step, ptrs, sizes))


def test_small_c_depthwise_wgrad_binary_bakes_its_channel_count(cache):
    """The small-C kernel bakes C, so a problem is only offered the binaries
    built for its own channel count."""
    problem = _problem(groups=3, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2, W=64)
    plans, rejected = dks.direct_plans(cache, problem, "wgrad", _ARCH)
    cols = [p for p in plans if p.identity.algorithm == "direct_wgrad_dw_cols"]
    assert cols
    assert all(json.loads(p.identity.knobs)["channels"] == 3 for p in cols)
    assert any(
        i.algorithm == "direct_wgrad_dw_cols" and "bakes C=" in why
        for i, why in rejected
    )


def _need_benchmark_direct_conv():
    """benchmark_direct_conv imports torch at module level (its conv
    references); the host-only CI venv that runs this file has none."""
    pytest.importorskip("torch")


def test_run_from_cache_variants_filter():
    """``--variants`` names cache variants (DIRECT_CAPABILITIES keys)."""
    from types import SimpleNamespace

    _need_benchmark_direct_conv()
    from benchmarks.common.benchmark_direct_conv import _cache_variants

    assert _cache_variants(SimpleNamespace(variants=None)) == frozenset()
    picked = _cache_variants(
        SimpleNamespace(variants="direct_grouped_dgrad_4c, direct_grouped_dgrad_fused")
    )
    assert picked == {"direct_grouped_dgrad_4c", "direct_grouped_dgrad_fused"}
    assert picked <= set(dks.DIRECT_CAPABILITIES)
    with pytest.raises(ValueError, match="unknown direct variant"):
        _cache_variants(SimpleNamespace(variants="direct_grouped_dgrad_4c,bogus"))


def test_run_from_cache_variants_filter_can_leave_nothing():
    """A ``--variants`` filter that keeps none of a case's plans or rejections
    leaves both lists empty, and the case reports the filter rather than an
    index error or a missing cache build."""
    from types import SimpleNamespace

    _need_benchmark_direct_conv()
    from benchmarks.common.benchmark_direct_conv import (
        _filter_variants,
        _no_plan_reason,
    )

    def plan(variant):
        return SimpleNamespace(identity=SimpleNamespace(algorithm=variant))

    plans = [plan("direct_grouped_dgrad"), plan("direct_dgrad_mfma_rtc")]
    rejected = [(SimpleNamespace(algorithm="direct_grouped_dgrad_mfma"), "why")]
    keep = frozenset({"direct_grouped_dgrad_4c"})
    got_plans, got_rejected = _filter_variants(plans, rejected, keep)
    assert got_plans == [] and got_rejected == []
    why = _no_plan_reason(got_rejected, "bf16", keep)
    assert "--variants direct_grouped_dgrad_4c" in why
    assert "--compile-all" not in why
    # Unfiltered, the same empty case points at the cache build; a kept
    # rejection is reported as is.
    assert "--compile-all --dtype bf16" in _no_plan_reason([], "bf16", frozenset())
    assert _filter_variants(plans, rejected, frozenset()) == (plans, rejected)
    kept = _filter_variants(plans, rejected, frozenset({"direct_grouped_dgrad_mfma"}))
    assert kept == ([], rejected)
    assert _no_plan_reason(kept[1], "bf16", keep) == "why"


def test_compile_all_refuses_variants_before_checking_them(capsys):
    """``--compile-all --variants`` is refused as a mode error, whether or not
    the names are valid: --variants only filters --run-from-cache."""
    from types import SimpleNamespace

    _need_benchmark_direct_conv()
    from benchmarks.common.benchmark_direct_conv import _cache_dispatch

    for names in ("direct_grouped_dgrad_4c", "bogus"):
        args = SimpleNamespace(compile_all=True, variants=names, directions=None)
        assert _cache_dispatch(args, _ARCH, []) == 2
        err = capsys.readouterr().err
        assert "--variants only filters --run-from-cache" in err
        assert "unknown direct variant" not in err


# ---------------------------------------------------------------------------
# A wave32 target (gfx1250): only the wave-parametric depthwise forward
# kernels exist there, and they are built and launched at its wave size.
# ---------------------------------------------------------------------------
_WAVE32_ARCH = "gfx1250"


@pytest.fixture(scope="module")
def wave32_jobs():
    return dks.enumerate_direct_jobs(
        arch=_WAVE32_ARCH,
        target=ArchTarget.from_gfx(_WAVE32_ARCH),
        directions=("fwd",),
    )


def test_wave32_forward_builds_the_depthwise_kernels_at_its_wave_size(wave32_jobs):
    """The MFMA variants are wave64 and drop out; the depthwise ones are built
    at wave_size 32, which the identity records and the spec carries."""
    assert {j.identity.algorithm for j in wave32_jobs} == DIRECT_WAVE_PARAMETRIC
    for job in wave32_jobs:
        assert job.identity.wave_size == 32
        variant = job.identity.algorithm
        caps = dks.DirectCaps(**job.caps)
        spec = dks.make_spec(
            variant,
            dks.probe_problem(caps, dks.DIRECT_DTYPE, variant, job.spec_kwargs),
            job.spec_kwargs,
            job.identity.wave_size,
        )
        assert spec.wave_size == 32, job.identity.label()


def _wave32_cache(tmp_path, jobs, spec_wave_size):
    """The wave32 jobs with placeholder binaries. ``spec_wave_size`` is the
    wave size the build recorded (``None``: an entry that predates the record)."""
    c = KernelCache(tmp_path, _WAVE32_ARCH)
    for job in jobs:
        meta = {"kernel_name": job.identity.label()}
        if spec_wave_size is not None:
            meta["spec_wave_size"] = spec_wave_size
        c.put(job.identity, b"placeholder", meta)
    return c


def test_wave32_build_records_its_wave_size(wave32_jobs):
    """The build records the wave size its binary was built at, which is what
    the run side checks before deriving a launch geometry from the identity."""
    from kernels.conv.direct_aot_jobs import direct_kernel_for_job

    job = wave32_jobs[0]
    _, meta = direct_kernel_for_job(job, _WAVE32_ARCH, dks.DIRECT_DTYPE)
    assert meta == {"spec_wave_size": 32}


def test_legacy_wave32_entries_are_not_launched(tmp_path, wave32_jobs):
    """A gfx1250 entry built before the wave size was recorded holds a binary
    built at the wave64 spec default under a wave32 identity: launching it with
    the identity's geometry would leave half of every channel tile unwritten,
    so it is rejected with a rebuild hint rather than offered."""
    c = _wave32_cache(tmp_path, wave32_jobs, None)
    problem = _problem(H=56, W=56, groups=192, cpg=1, kpg=1)
    plans, rejected = dks.direct_plans(c, problem, "fwd", _WAVE32_ARCH)
    assert plans == []
    assert rejected and all("--compile-all" in why for _, why in rejected)


def test_conflicting_wave_size_knob_is_rejected():
    problem = _problem(groups=64, cpg=1, kpg=1)
    knobs = dict(block_w=8, block_h=2, block_waves=1, unroll_rows=False)
    with pytest.raises(ValueError, match="conflicts"):
        dks.make_spec("direct_depthwise_tiled", problem, dict(knobs, wave_size=0), 32)
    spec = dks.make_spec(
        "direct_depthwise_tiled", problem, dict(knobs, wave_size=32), 32
    )
    assert spec.wave_size == 32


@pytest.mark.parametrize("groups", [9, 192])
def test_wave32_plans_launch_wave32_blocks(tmp_path, wave32_jobs, groups):
    c = _wave32_cache(tmp_path, wave32_jobs, 32)
    problem = _problem(H=56, W=56, groups=groups, cpg=1, kpg=1)
    plans, _ = dks.direct_plans(c, problem, "fwd", _WAVE32_ARCH)
    assert plans
    assert ("direct_depthwise_spatial" in _variants(plans)) == (groups < 32)
    for plan in plans:
        (step,) = plan.steps
        waves = json.loads(plan.identity.knobs)["block_waves"]
        assert step.block == (waves * 32, 1, 1), plan.identity.label()
