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
