# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The emulated-TF32 register-ceiling guard, shared by fwd / dgrad / wgrad.

A BF16x3-emulated TF32 warp tile with >= 384 fp32 accumulators per lane sits at
the 512-VGPR ceiling, where comgr's VGPR->AGPR spill path miscompiled a two-stage
wgrad kernel (one accumulator register wrong; the same IR is correct at -O1).
Every direction rejects such tiles on gfx950; the native gfx942 atom and every
non-TF32 spec are untouched. CPU only.
"""

from __future__ import annotations

import functools

import pytest

from kernels.common._conv_implicit_gemm_common import (
    ConvDataSpec,
    ConvProblem,
    _TF32_EMULATED_MAX_ACC_PER_LANE,
)
from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec, is_valid_spec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    is_valid_dgrad_spec,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    is_valid_wgrad_spec,
)

_TF32 = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")
_FP32 = ConvDataSpec("fp32", "fp32", "fp32")
_PROBLEM = ConvProblem(N=32, Hi=14, Wi=14, C=256, K=256, Y=3, X=3, pH=1, pW=1)
_GUARD = "512-VGPR ceiling"

# 256x384 over 1x4 warps = 384 accumulators per lane (the miscompiled tile);
# 128x128 over 2x2 warps = 64.
_BIG = dict(tile_m=256, tile_n=384, warp_m=1, warp_n=4)
_SMALL = dict(tile_m=128, tile_n=128, warp_m=2, warp_n=2)


def _wgrad(geom, data, atom):
    return WgradConvSpec(
        problem=_PROBLEM,
        data=data,
        tile_k=32,
        warp_tile_m=atom[0],
        warp_tile_n=atom[1],
        warp_tile_k=atom[2],
        **geom,
    )


def _fwd(geom, data, atom):
    # cshuffle: the default epilogue rejects these tiles' vector_size_c first.
    return ImplicitGemmConvSpec(
        problem=_PROBLEM,
        data=data,
        epilogue="cshuffle",
        tile_k=32,
        warp_tile_m=atom[0],
        warp_tile_n=atom[1],
        warp_tile_k=atom[2],
        **geom,
    )


def _dgrad(geom, data, atom):
    return DgradConvSpec(
        problem=_PROBLEM,
        data=data,
        tile_k=32,
        warp_tile_m=atom[0],
        warp_tile_n=atom[1],
        warp_tile_k=atom[2],
        **geom,
    )


_DIRECTIONS = [
    ("fwd", _fwd, is_valid_spec),
    ("dgrad", _dgrad, is_valid_dgrad_spec),
    ("wgrad", _wgrad, is_valid_wgrad_spec),
]


def test_threshold_is_the_miscompiled_tile():
    tile = _BIG
    acc = (tile["tile_m"] // tile["warp_m"]) * (tile["tile_n"] // tile["warp_n"]) // 64
    assert acc == _TF32_EMULATED_MAX_ACC_PER_LANE


@pytest.mark.parametrize(
    "name,make,valid", _DIRECTIONS, ids=[d[0] for d in _DIRECTIONS]
)
def test_emulated_tf32_rejects_ceiling_tile_on_gfx950(name, make, valid):
    ok, why = valid(make(_BIG, _TF32, (16, 16, 32)), "gfx950")
    assert not ok and _GUARD in why, (name, why)


@pytest.mark.parametrize(
    "name,make,valid", _DIRECTIONS, ids=[d[0] for d in _DIRECTIONS]
)
def test_small_tile_is_not_caught_by_the_guard(name, make, valid):
    ok, why = valid(make(_SMALL, _TF32, (16, 16, 32)), "gfx950")
    assert _GUARD not in why, (name, why)


@pytest.mark.parametrize(
    "name,make,valid", _DIRECTIONS, ids=[d[0] for d in _DIRECTIONS]
)
def test_native_gfx942_and_fp32_are_exempt(name, make, valid):
    # gfx942's native xf32 has no split temporaries; it may still fail for
    # other reasons (LDS budget), just never for this one.
    _ok, why = valid(make(_BIG, _TF32, (16, 16, 8)), "gfx942")
    assert _GUARD not in why, (name, why)
    _ok, why = valid(make(_BIG, _FP32, (16, 16, 4)), "gfx950")
    assert _GUARD not in why, (name, why)


# ---------------------------------------------------------------------------
# One TF32 gate for every direction (review: dtype_d policy, no drift, the
# arch-free fragment lookup).
# ---------------------------------------------------------------------------


def test_tf32_atom_shapes_are_unique_across_arches():
    """The arch-free TF32 gate reads the fragment width off the MfmaAtom table
    by (m, n, k). That is sound only while no (m, n, k) is a TF32 atom on two
    arches with different fragment widths; a new atom that breaks it fails here
    rather than letting validate() check the wrong width."""
    from rocke.core.arch import ArchTarget, known_arches
    from rocke.helpers.atoms import mfma_atom

    arches = known_arches()
    seen = {}
    for arch in arches:
        for op in ArchTarget.from_gfx(arch).mma.ops:
            if op.a_dtype != "tf32":
                continue
            key = (op.m, op.n, op.k)
            seen.setdefault(key, set()).add(op.a_frag_len)
    assert seen, "no TF32 atom in any arch catalog"
    for key, widths in seen.items():
        assert len(widths) == 1, f"TF32 atom {key} has fragment widths {widths}"
        assert mfma_atom("tf32", *key).a_per_lane in widths, key


@pytest.mark.parametrize(
    "name,make,valid", _DIRECTIONS, ids=[d[0] for d in _DIRECTIONS]
)
def test_tf32_needs_an_fp32_output_in_every_direction(name, make, valid):
    data = ConvDataSpec("fp32", "fp32", "fp16", mma_dtype="tf32")
    ok, why = valid(make(_SMALL, data, (16, 16, 32)), "gfx950")
    assert not ok and "needs an fp32 output" in why, (name, why)


def test_dgrad_and_wgrad_share_one_tf32_gate():
    """The same violation gets the same message from both families."""
    misaligned = dict(lds_k_pad=2)
    for warp_tile in ((16, 16, 32), (16, 16, 24)):
        msgs = set()
        for make, valid in (
            (_dgrad, is_valid_dgrad_spec),
            (_wgrad, is_valid_wgrad_spec),
        ):
            spec = make(_SMALL, _TF32, warp_tile)
            if warp_tile == (16, 16, 32):
                from dataclasses import replace

                spec = replace(spec, **misaligned)
            msgs.add(valid(spec, "gfx950")[1])
        assert len(msgs) == 1, msgs


# ---------------------------------------------------------------------------
# The direct MFMA dgrad's weight preload (review: fp32 weights were counted
# at the 16-bit width, so the fp32 / tf32 grid passed the spill guard).
# ---------------------------------------------------------------------------

# The runtime-channel candidate of the review: 3x3, cpg 64 (4 M-tiles),
# kpg 256 (8 K-32 atoms, 2 per wave), one Q subtile.
_PRELOAD_KNOBS = dict(
    block_q=16,
    block_groups=1,
    block_h=8,
    waves_q=1,
    waves_k=4,
    runtime_k_loop=False,
    persistent_grid=False,
    fold_k32=True,
    k_atoms=8,
    m_tiles=4,
)


def _preload_problem(dtype):
    from kernels.common.conv_direct_grouped import DirectConvProblem

    return DirectConvProblem(
        N=2, H=28, W=28, groups=1, cpg=64, kpg=256, KH=3, KW=3, PAD=1, dtype=dtype
    )


def test_preload_fragment_follows_storage_and_emulation():
    from kernels.conv.direct_variants import dgrad_frag_vgprs

    for fold, regs16 in ((True, 4), (False, 2)):
        for dtype in ("fp16", "bf16"):
            assert dgrad_frag_vgprs(fold, dtype, "gfx950") == regs16
        # One element per register with fp32 storage; native TF32 (gfx942)
        # adds nothing.
        for dtype, arch in (("fp32", "gfx950"), ("tf32", "gfx942"), ("tf32", None)):
            assert dgrad_frag_vgprs(fold, dtype, arch) == 2 * regs16
    # BF16x3 split halves span the 8-wide emulated atom: a K-16 fragment is
    # zero-padded to it, so its split costs twice its own registers.
    assert dgrad_frag_vgprs(True, "tf32", "gfx950") == 8 + 8
    assert dgrad_frag_vgprs(False, "tf32", "gfx950") == 4 + 8


def test_review_preload_candidate_is_counted_at_fp32_width():
    """The 16-bit estimate admits 336 registers; fp32 weights plus
    accumulators need 624 (and gfx950's TF32 split 576 more), past the spill
    guard, on both arches and through both estimates."""
    from kernels.conv.direct_aot_jobs import dgrad_rtc_spills
    from kernels.conv.direct_variants import (
        _MFMA_PRELOAD_VGPR_BUDGET,
        dgrad_rtc_vgprs,
        direct_mfma_knobs_reason,
        direct_spec_for_problem,
    )

    k = _PRELOAD_KNOBS
    for dtype in ("fp16", "bf16"):
        assert dgrad_rtc_vgprs(k, 3, 3, dtype, "gfx950") == 336
        assert direct_mfma_knobs_reason(_preload_problem(dtype), k, "gfx950") is None
    assert dgrad_rtc_vgprs(k, 3, 3, "fp32", "gfx950") == 624
    assert dgrad_rtc_vgprs(k, 3, 3, "tf32", "gfx942") == 624
    assert dgrad_rtc_vgprs(k, 3, 3, "tf32", "gfx950") == 624 + 576
    assert 624 > _MFMA_PRELOAD_VGPR_BUDGET
    for dtype in ("fp32", "tf32"):
        problem = _preload_problem(dtype)
        for arch in ("gfx942", "gfx950"):
            assert "would spill" in direct_mfma_knobs_reason(problem, k, arch)
            assert dgrad_rtc_spills(k, 3, 3, arch, dtype)
            spec, why = direct_spec_for_problem(
                "direct_dgrad_mfma_rtc", problem, k, arch
            )
            assert spec is None and "would spill" in why


@functools.cache
def _dgrad_jobs(arch, dtype):
    """The direct dgrad cache's jobs for ``dtype`` on ``arch``.

    Enumerated once per (arch, dtype) for the whole module: both tests below
    read the fp32 and tf32 grids, and each enumeration walks the whole grid.
    """
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    return tuple(
        dks.enumerate_direct_jobs(
            arch=arch,
            target=ArchTarget.from_gfx(arch),
            directions=("dgrad",),
            dtype=dtype,
        )
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp32", "tf32"])
def test_no_fp32_storage_preload_job_is_over_budget(arch, dtype):
    """Every preloading MFMA dgrad binary the fp32 / tf32 cache builds fits
    the dtype-aware budget, and the review's candidate is not among them."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.conv.direct_variants import (
        DIRECT_MFMA_DGRAD,
        direct_mfma_knobs_reason,
    )

    preloading = [
        j
        for j in _dgrad_jobs(arch, dtype)
        if j.identity.algorithm in DIRECT_MFMA_DGRAD and j.spec_kwargs["waves_k"] > 1
    ]
    assert preloading
    for job in preloading:
        knobs = job.spec_kwargs
        variant = job.identity.algorithm
        probe = dks.probe_problem(dks.DirectCaps(**job.caps), dtype, variant, knobs)
        assert direct_mfma_knobs_reason(probe, knobs, arch) is None, job.identity
        assert not (
            "k_atoms" in knobs
            and dks.dgrad_rtc_spills(knobs, probe.KH, probe.KW, arch, dtype)
        ), job.identity
        assert not (
            probe.KH == 3
            and {k: knobs.get(k) for k in _PRELOAD_KNOBS} == _PRELOAD_KNOBS
        ), job.identity


# A baked-channel job the fp32-storage budget cuts: 3x3, cpg 48 (3 M-tiles),
# kpg 192 over four waves (three K-16 atoms each). At fp32 width it preloads
# 324 weight and 36 accumulator registers, between the fp32-storage budget
# and the 16-bit one, so only the smaller budget refuses it; the fp16 grid
# builds it (162 + 36).
_CUT_VARIANT = "direct_grouped_dgrad_mfma"
_CUT_CAPS = dict(KH=3, PAD=1, stride=1, cpg=48, kpg=192)
_CUT_KNOBS = dict(
    block_q=16,
    block_groups=1,
    block_h=8,
    waves_q=1,
    waves_k=4,
    runtime_k_loop=False,
    persistent_grid=False,
    fold_k32=False,
)


def _builds_cut_job(arch, dtype):
    """Does the direct dgrad cache of ``dtype`` on ``arch`` build the job?"""
    return any(
        j.identity.algorithm == _CUT_VARIANT
        and {k: j.caps[k] for k in _CUT_CAPS} == _CUT_CAPS
        and j.spec_kwargs == _CUT_KNOBS
        for j in _dgrad_jobs(arch, dtype)
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp32", "tf32"])
def test_fp32_storage_budget_cuts_a_job_the_16bit_budget_builds(arch, dtype):
    """The fp32-storage preload budget, not the 16-bit one, refuses the job:
    fp16 builds it, fp32 and tf32 do not, on both arches (gfx950's TF32
    split counts its weights three times over)."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.conv.direct_variants import direct_mfma_knobs_reason

    assert _builds_cut_job(arch, "fp16")
    assert not _builds_cut_job(arch, dtype)
    probe = dks.probe_problem(dks.DirectCaps(**_CUT_CAPS), dtype, _CUT_VARIANT)
    weights = 972 if (dtype, arch) == ("tf32", "gfx950") else 324
    assert direct_mfma_knobs_reason(probe, _CUT_KNOBS, arch) == (
        f"preloaded weights ({weights}) + accumulators (36) VGPRs > 256: "
        "the kernel would spill"
    )
