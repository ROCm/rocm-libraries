# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 grouped and depthwise direct families.

Knob classes covered:

* the windowed Toeplitz MFMA forward and dgrad (``direct_depthwise_win`` and
  ``direct_depthwise_dgrad_win``): the dense channel form, the XCD tile
  order (``xcd_tiles``), the swizzled LDS staging (``lds_swz``), the LDS-DMA
  copies of the dY window and of the weights (``dma``, ``wdma``), the
  block-diagonal 2- and 4-channel groups, the grouped and stride-2 dgrad and
  16 waves;
* the row-tiled depthwise spatial forward (``block_h``);
* the depthwise wgrad kernels: channels per lane and the XCD pairing of the
  LDS-ring kernel, stride 2 / XCD pairing / swizzle / A-direct of the MFMA
  kernel;
* the grouped ping-pong dgrad (block_k 64) and the ping-pong forward with
  valid padding;
* the Winograd filter transform that reads a dgrad filter.

CPU: validator verdicts and reasons, kernel builds per knob class and dtype,
launch geometry, kernel names, the AOT grid counts of these families and the
packaged-launch refusal of the forward identities whose knobs no packaged
instance field carries (95 of them: 71 windowed and 24 row-tiled spatial).
GPU (gfx950): each class against a float64 numpy reference of the stored
operands in both dtypes, with a poisoned output.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_grouped_dw_gfx950.py -v
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.tests.gpu import current_device_arch_in

from kernels.common import conv_winograd_fwd_rt as wino
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_direct_dgrad_pp_mfma import (
    DgradPpMfmaSpec,
    build_dgrad_pp_mfma,
    dgrad_pp_mfma_problem,
    dgrad_pp_mfma_reason,
)
from kernels.common.conv_direct_dw_wgrad import (
    DwWgradLdsSpec,
    build_dw_wgrad_lds,
    is_valid_dw_wgrad_lds_spec,
)
from kernels.common.conv_direct_dw_wgrad_mfma import (
    DwWgradMfmaSpec,
    build_dw_wgrad_mfma,
    is_valid_dw_wgrad_mfma_spec,
)
from kernels.common.conv_direct_fwd_pp_mfma import (
    FwdPpMfmaSpec,
    build_fwd_pp_mfma,
    fwd_pp_mfma_reason,
)
from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_LDS_BUDGET,
    DirectConvProblem,
    DirectDepthwiseDgradWindowedSpec,
    DirectDepthwiseSpatialSpec,
    build_direct_depthwise_dgrad_windowed,
    build_direct_depthwise_spatial,
    direct_launch_geometry,
    is_valid_depthwise_dgrad_win_spec,
    is_valid_depthwise_spatial_spec,
)
from kernels.conv.direct_variants import (
    DIRECT_DEPTHWISE_WIN_FWD,
    direct_spec_for_problem,
)

from ._direct_conv_numeric import (
    direct_dgrad_reference,
    direct_fwd_reference,
    direct_wgrad_reference,
    launch_direct_conv,
    launch_direct_kernel,
    poisoned,
    stored_uniform,
)

_WIN_FWD = DIRECT_DEPTHWISE_WIN_FWD
_WIN_DGRAD = "direct_depthwise_dgrad_win"
_SPATIAL = "direct_depthwise_spatial"
_WGRAD_LDS = "direct_wgrad_dw"
_WGRAD_MFMA = "direct_wgrad_dw_mfma"
_PP_DGRAD = "direct_dgrad_pp_mfma"
_PP_FWD = "direct_fwd_pp_mfma"
_DTYPES = ("fp16", "bf16")
_TOL = {"fp16": 5e-3, "bf16": 2e-2}


def _dw(N, H, W, G, cpg, K, S, pad=None):
    """A depthwise (``cpg`` 1) or grouped (``cpg = kpg``) square-filter problem
    with "same" padding unless ``pad`` is given; the dtype is set per case."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=G,
        cpg=cpg,
        kpg=cpg,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2 if pad is None else pad,
        stride=S,
        dtype="fp16",
    )


# --------------------------------------------------------------- windowed rows

# (id, variant, problem, knobs). A forward row is the forward problem; a dgrad
# row is the dgrad problem (dX extents). Each is valid on gfx950 as listed, and
# together they cover every knob class of the windowed form.
_W = dict(prefetch_rows=2)
_WIN_ROWS = [
    # forward: swizzled staging + XCD tile order (3x3 at 8 waves, 5x5 with two
    # images folded), XCD alone with three rows in flight, at stride 2
    (
        "f_swz_xcd",
        _WIN_FWD,
        _dw(2, 28, 28, 64, 1, 3, 1),
        dict(block_waves=8, block_h=14, w_fold=1, lds_swz=True, xcd_tiles=True, **_W),
    ),
    (
        "f_swz_xcd_5x5",
        _WIN_FWD,
        _dw(3, 16, 20, 64, 1, 5, 1),
        dict(block_waves=4, block_h=16, w_fold=2, lds_swz=True, xcd_tiles=True, **_W),
    ),
    (
        "f_xcd_pf3",
        _WIN_FWD,
        _dw(2, 16, 24, 64, 1, 3, 1),
        dict(block_waves=8, block_h=8, w_fold=1, xcd_tiles=True, prefetch_rows=3),
    ),
    (
        "f_swz_xcd_s2",
        _WIN_FWD,
        _dw(2, 28, 28, 64, 1, 3, 2),
        dict(block_waves=8, block_h=14, w_fold=1, lds_swz=True, xcd_tiles=True, **_W),
    ),
    # forward: block-diagonal groups of 4 and 2 channels, swizzled cpg 16
    (
        "f_cpg4",
        _WIN_FWD,
        _dw(2, 13, 21, 16, 4, 3, 1),
        dict(block_waves=3, block_h=13, w_fold=2, **_W),
    ),
    (
        "f_cpg2",
        _WIN_FWD,
        _dw(2, 12, 12, 32, 2, 3, 1),
        dict(block_waves=2, block_h=8, w_fold=1, **_W),
    ),
    (
        "f_cpg16_swz",
        _WIN_FWD,
        _dw(2, 16, 16, 8, 16, 3, 1),
        dict(block_waves=4, block_h=8, w_fold=1, lds_swz=True, **_W),
    ),
    # forward: the dense channel form (cpg 32), with four images folded, and
    # with the dY / weight LDS-DMA copies (cpg 32 and 64)
    (
        "f_dense",
        _WIN_FWD,
        _dw(2, 14, 14, 4, 32, 3, 1),
        dict(dense=True, block_waves=4, block_h=14, w_fold=1, **_W),
    ),
    (
        "f_dense_wf4",
        _WIN_FWD,
        _dw(5, 9, 7, 4, 32, 3, 1),
        dict(dense=True, block_waves=2, block_h=4, w_fold=4, **_W),
    ),
    (
        "f_dense_dma",
        _WIN_FWD,
        _dw(2, 14, 14, 4, 32, 3, 1),
        dict(
            dense=True,
            lds_swz=True,
            dma=True,
            wdma=True,
            block_waves=4,
            block_h=14,
            w_fold=1,
            **_W,
        ),
    ),
    (
        "f_dense_dma_c64",
        _WIN_FWD,
        _dw(2, 12, 12, 2, 64, 3, 1),
        dict(
            dense=True,
            lds_swz=True,
            dma=True,
            wdma=True,
            block_waves=4,
            block_h=8,
            w_fold=1,
            **_W,
        ),
    ),
    # dgrad: stride 2 (3x3 with two images folded, 7x7 with four), grouped
    # (cpg 16; cpg 8 at stride 2), 16 waves, swizzled staging
    (
        "d_s2",
        _WIN_DGRAD,
        _dw(2, 24, 24, 64, 1, 3, 2),
        dict(mfma=True, block_waves=8, block_h=12, w_fold=2, **_W),
    ),
    (
        "d_s2_7x7",
        _WIN_DGRAD,
        _dw(2, 21, 19, 64, 1, 7, 2),
        dict(mfma=True, block_waves=4, block_h=8, w_fold=4, **_W),
    ),
    (
        "d_grp16",
        _WIN_DGRAD,
        _dw(2, 13, 21, 4, 16, 3, 1),
        dict(mfma=True, block_waves=4, block_h=8, w_fold=2, **_W),
    ),
    (
        "d_grp8_s2",
        _WIN_DGRAD,
        _dw(2, 24, 22, 8, 8, 3, 2),
        dict(mfma=True, block_waves=8, block_h=12, w_fold=1, **_W),
    ),
    (
        "d_w16",
        _WIN_DGRAD,
        _dw(2, 20, 20, 128, 1, 3, 1),
        dict(mfma=True, block_waves=16, block_h=10, w_fold=1, **_W),
    ),
    (
        "d_swz_wf4",
        _WIN_DGRAD,
        _dw(4, 12, 9, 64, 1, 5, 1),
        dict(mfma=True, lds_swz=True, block_waves=4, block_h=12, w_fold=4, **_W),
    ),
    # dgrad: the dense form, at stride 2, and with the LDS-DMA copies
    (
        "d_dense",
        _WIN_DGRAD,
        _dw(2, 16, 16, 4, 32, 3, 1),
        dict(mfma=True, dense=True, block_waves=4, block_h=16, w_fold=1, **_W),
    ),
    (
        "d_dense_s2",
        _WIN_DGRAD,
        _dw(2, 32, 32, 4, 32, 3, 2),
        dict(mfma=True, dense=True, block_waves=4, block_h=16, w_fold=1, **_W),
    ),
    (
        "d_dense_dma",
        _WIN_DGRAD,
        _dw(2, 14, 14, 4, 32, 3, 1),
        dict(
            mfma=True,
            dense=True,
            lds_swz=True,
            dma=True,
            wdma=True,
            block_waves=4,
            block_h=14,
            w_fold=1,
            **_W,
        ),
    ),
    (
        "d_dense_dma_c64",
        _WIN_DGRAD,
        _dw(2, 16, 16, 2, 64, 3, 1),
        dict(
            mfma=True,
            dense=True,
            lds_swz=True,
            dma=True,
            wdma=True,
            block_waves=4,
            block_h=16,
            w_fold=1,
            **_W,
        ),
    ),
]

# The name suffix each knob adds to the windowed kernels' names.
_WIN_NAME_FLAGS = {
    "dense": "_dn",
    "lds_swz": "_sw",
    "xcd_tiles": "_xc",
    "dma": "_dma",
    "wdma": "_wd",
}


def _win_spec(row, dtype):
    _, variant, problem, knobs = row
    spec, why = direct_spec_for_problem(
        variant, replace(problem, dtype=dtype), knobs, "gfx950"
    )
    assert spec is not None, why
    return spec


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WIN_ROWS, ids=lambda r: r[0])
def test_win_rows_are_valid_and_build(row, dtype):
    spec = _win_spec(row, dtype)
    ok, why = is_valid_depthwise_dgrad_win_spec(spec, arch="gfx950")
    assert ok, why
    kernel = build_direct_depthwise_dgrad_windowed(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert kernel.attrs["max_workgroup_size"] == spec.threads_per_block
    assert spec.mfma_lds_bytes() <= DW_DGRAD_MFMA_LDS_BUDGET
    name = spec.kernel_name()
    knobs = row[3]
    for knob, suffix in _WIN_NAME_FLAGS.items():
        assert (suffix in f"{name}_") == bool(knobs.get(knob)), (knob, name)
    grid, block = direct_launch_geometry(spec)
    assert block == (spec.threads_per_block, 1, 1)
    assert grid == spec.grid()


def test_win_kernel_names_are_unique():
    """The flags are all in the name, so no two rows share a symbol."""
    names = [
        _win_spec(row, dtype).kernel_name() for row in _WIN_ROWS for dtype in _DTYPES
    ]
    assert len(names) == len(set(names))


def test_win_reasons():
    def spec(row_id, **kw):
        row = next(r for r in _WIN_ROWS if r[0] == row_id)
        return replace(_win_spec(row, "fp16"), **kw)

    def why(s, arch="gfx950"):
        ok, reason = is_valid_depthwise_dgrad_win_spec(s, arch=arch)
        assert not ok
        return reason

    assert why(spec("f_swz_xcd", dma=True)) == (
        "dma needs the dense lds_swz form (wave64, stride-1 window)"
    )
    assert why(spec("f_dense_dma", dma=False)) == "wdma needs the dma form"
    assert why(spec("f_dense_dma", lds_swz=False)) == (
        "dma needs the dense lds_swz form (wave64, stride-1 window)"
    )
    # dense is a cpg % 32 == 0 form
    dense16 = spec("f_cpg16_swz", dense=True, lds_swz=False)
    assert "dense needs the wave64 grouped form with cpg % 32 == 0" in why(dense16)
    # a stride-2 dgrad needs an even row tile
    assert "stride=1, or 2 on the mfma dgrad form" in why(
        spec("d_s2", problem=replace(spec("d_s2").problem, stride=3))
    )
    assert why(spec("f_swz_xcd_s2", block_h=13)) == (
        "block_h=13 must be a multiple of out_stride=2"
    )
    # xcd_tiles is built for gfx950 only; the form needs a wave64 MFMA arch
    xcd = spec("f_swz_xcd")
    assert "mfma needs one of ('gfx950',) (got gfx942)" in why(xcd, "gfx942")
    assert is_valid_depthwise_dgrad_win_spec(xcd, arch="gfx950")[0]
    # the VALU form takes none of the MFMA-only knobs
    valu = DirectDepthwiseDgradWindowedSpec(
        problem=_dw(2, 14, 14, 64, 1, 3, 1), block_h=8, xcd_tiles=True
    )
    assert "lds_swz, xcd_tiles and dma require mfma" in why(valu)
    s2_valu = DirectDepthwiseDgradWindowedSpec(
        problem=_dw(2, 24, 24, 64, 1, 3, 2), block_w=8, block_h=8
    )
    assert "stride=1, or 2 on the mfma dgrad form" in why(s2_valu)


def test_win_dense_geometry():
    """The dense form: a wave owns 16 channels, a tile is 16 / w_fold columns
    wide, and one atom contracts 32 input channels of a tap."""
    row = next(r for r in _WIN_ROWS if r[0] == "f_dense_dma")
    s = _win_spec(row, "fp16")
    assert (s.block_ch, s.tile_w) == (16 * s.block_waves, 16)
    # 3x3 taps x cpg 32 / 32 per atom, per row of the tile
    assert s.unrolled_mfmas() == s.block_h * 9 * (s.problem.cpg // 32)
    # the dY ring is whole block-wide copies; one weight slot per row in flight
    assert s.dma_in_chunks() % s.threads_per_block == 0
    assert s.dma_in_chunks() >= s.swz_chunks()[0]
    assert s.wdma_slots() == min(s.prefetch_rows, s.problem.KH)
    assert s.wdma_slot_chunks() == 16 * 3 * s.problem.cpg * 2 // 16
    wf4 = _win_spec(next(r for r in _WIN_ROWS if r[0] == "f_dense_wf4"), "fp16")
    assert (wf4.tile_w, wf4.n_tiles) == (4, 2)  # 16 / 4 columns, 5 images / 4
    # the staged LDS grows by the weight slots under wdma
    no_w = replace(s, wdma=False)
    assert s.mfma_lds_bytes() == no_w.mfma_lds_bytes() + (
        s.wdma_slots() * s.block_waves * s.wdma_slot_chunks() * 16
    )


def test_win_swizzled_staging_is_unpadded():
    """lds_swz stages unpadded 16-byte chunks: it never needs more LDS than
    the padded layout of the same tile, and its chunk count is the buffer's."""
    for row_id in ("f_swz_xcd", "f_swz_xcd_5x5", "d_swz_wf4"):
        row = next(r for r in _WIN_ROWS if r[0] == row_id)
        s = _win_spec(row, "fp16")
        n_in, n_out = s.swz_chunks()
        assert n_in > 0 and n_out > 0
        assert s.mfma_lds_bytes() == 2 * (n_in + n_out) * 16
        plain = replace(s, lds_swz=False, xcd_tiles=False)
        assert plain.kernel_name() != s.kernel_name()


def test_win_stride2_forward_walks_the_stride1_grid():
    row = next(r for r in _WIN_ROWS if r[0] == "f_swz_xcd_s2")
    s = _win_spec(row, "bf16")
    assert s.fwd and s.out_stride == 2
    # the spec's problem is the dgrad the forward transposes: the stride-1 grid
    assert s.problem.stride == 1 and s.problem.H == 2 * row[2].Ho
    assert "_s2" in s.kernel_name()


# ---------------------------------------------------------------- spatial rows

_SPATIAL_ROWS = [
    (
        "g16_k3_bh64",
        DirectConvProblem(
            N=2, H=40, W=36, groups=16, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1,
            dtype="fp16",
        ),
        dict(block_waves=1, block_h=64),
    ),
    (
        "g8_k5_s2_bh8_w2",
        DirectConvProblem(
            N=2, H=33, W=30, groups=8, cpg=1, kpg=1, KH=5, KW=5, PAD=2, stride=2,
            dtype="fp16",
        ),
        dict(block_waves=2, block_h=8),
    ),
    (
        "g4_k9_bh16",
        DirectConvProblem(
            N=1, H=40, W=40, groups=4, cpg=1, kpg=1, KH=9, KW=9, PAD=4, stride=1,
            dtype="fp16",
        ),
        dict(block_waves=1, block_h=16),
    ),
]  # fmt: skip


def _spatial_spec(row, dtype):
    _, problem, knobs = row
    spec, why = direct_spec_for_problem(
        _SPATIAL, replace(problem, dtype=dtype), knobs, "gfx950"
    )
    assert spec is not None, why
    return spec


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _SPATIAL_ROWS, ids=lambda r: r[0])
def test_spatial_row_tiles_are_valid_and_build(row, dtype):
    spec = _spatial_spec(row, dtype)
    assert isinstance(spec, DirectDepthwiseSpatialSpec)
    assert is_valid_depthwise_spatial_spec(spec, arch="gfx950") == (True, "ok")
    kernel = build_direct_depthwise_spatial(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert f"_bh{spec.block_h}" in spec.kernel_name()
    p = spec.problem
    # one block per row tile: the row tiles ride on the grid's z axis
    grid, _ = direct_launch_geometry(spec)
    assert grid[2] == p.N * -(-p.Ho // spec.block_h)
    # the untiled form of the same layer keeps one tile and no "bh" in its name
    untiled = replace(spec, block_h=0)
    assert "_bh" not in untiled.kernel_name()
    assert direct_launch_geometry(untiled)[0][2] == p.N
    # a tile taller than the image does not change the answer's extents
    assert kernel.attrs["waves_per_eu"] == (1 if p.KH * p.KW > 64 else 2)


def test_spatial_block_h_reasons():
    spec = _spatial_spec(_SPATIAL_ROWS[0], "fp16")
    assert is_valid_depthwise_spatial_spec(replace(spec, block_h=-1)) == (
        False,
        "block_h must be >= 0 (got -1)",
    )
    assert is_valid_depthwise_spatial_spec(replace(spec, block_h=0))[0]


# ----------------------------------------------------------------- wgrad rows


def _dwp(C, KH=3, KW=3, PAD=1, stride=1, N=2, H=40, W=36, dtype="fp16"):
    return DirectConvProblem(
        N=N, H=H, W=W, groups=C, cpg=1, kpg=1, KH=KH, KW=KW, PAD=PAD, stride=stride,
        dtype=dtype,
    )  # fmt: skip


# (id, problem, spec knobs). The channel counts that are not a multiple of the
# channel tile give the XCD pairing something to pair.
_WGRAD_LDS_ROWS = [
    ("ch4_k5", _dwp(64, KH=5, KW=5, PAD=2), dict(lane_ch=4)),
    (
        "ch2_k7_xcdp",
        _dwp(72, KH=7, KW=7, PAD=3),
        dict(lane_ch=2, xcd_pair=True, target_blocks=512),
    ),
    (
        "ch4_s2_lw8_xcdp",
        _dwp(80, stride=2, H=30, W=40, N=4),
        dict(lane_ch=4, lanes_w=8, xcd_pair=True, target_blocks=24),
    ),
]
_WGRAD_MFMA_ROWS = [
    ("s2_k3", _dwp(64, stride=2, H=20, W=24), {}),
    (
        "s2_k5_swz_xcdp",
        _dwp(72, KH=5, KW=5, PAD=2, stride=2, H=21, W=26),
        dict(q_tiles=4, rows_per_chunk=8, xcd_pair=True, swz=True),
    ),
    (
        "s1_adir_swz_xcdp",
        _dwp(96),
        dict(rows_per_chunk=32, target_blocks=4096, adir=True, swz=True, xcd_pair=True),
    ),
    (
        "s2_k7_adir",
        _dwp(64, KH=7, KW=7, PAD=3, stride=2, H=28, W=28),
        dict(q_tiles=4, adir=True),
    ),
    (
        "unfold_k5_adir",
        _dwp(64, KH=5, KW=5, PAD=0, H=10, W=6),
        dict(q_tiles=5, unfold_n=2, adir=True),
    ),
]


def _lds_spec(row, dtype):
    _, problem, knobs = row
    return DwWgradLdsSpec(problem=replace(problem, dtype=dtype), **knobs)


def _mfma_spec(row, dtype):
    _, problem, knobs = row
    return DwWgradMfmaSpec(problem=replace(problem, dtype=dtype), **knobs)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WGRAD_LDS_ROWS, ids=lambda r: r[0])
def test_wgrad_lds_rows_are_valid_and_build(row, dtype):
    spec = _lds_spec(row, dtype)
    assert is_valid_dw_wgrad_lds_spec(spec, "gfx950") == (True, "ok")
    kernel = build_dw_wgrad_lds(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert kernel.attrs["max_workgroup_size"] == spec.threads
    assert spec.block_c == spec.lane_ch * spec.lanes_c
    name = spec.kernel_name()
    assert (f"_ch{spec.lane_ch}" in name) == (spec.lane_ch != 8)
    assert ("_xcdp" in name) == spec.xcd_pair
    if spec.xcd_pair:
        # the remap only runs with a ragged last channel tile, at least two
        # tiles and a partition count that is a whole number of XCDs
        p = spec.problem
        assert p.groups % spec.block_c != 0
        assert spec.n_channel_tiles() >= 2
        assert spec.n_partitions() % 8 == 0, spec.n_partitions()


def test_wgrad_lds_reasons():
    base = _lds_spec(_WGRAD_LDS_ROWS[0], "fp16")
    assert is_valid_dw_wgrad_lds_spec(replace(base, lane_ch=3), "gfx950") == (
        False,
        "lane_ch must be 2, 4 or 8 and the channel tile a multiple of 8",
    )
    wide = replace(base, lane_ch=8, problem=replace(base.problem, KH=7, KW=7, PAD=3))
    # 7x7 taps x 8 channels = 392 accumulators: why lane_ch exists
    assert is_valid_dw_wgrad_lds_spec(wide, "gfx950") == (
        False,
        "too many accumulators",
    )
    assert is_valid_dw_wgrad_lds_spec(replace(wide, lane_ch=2), "gfx950")[0]
    assert "wave64" in is_valid_dw_wgrad_lds_spec(base, "gfx1250")[1]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WGRAD_MFMA_ROWS, ids=lambda r: r[0])
def test_wgrad_mfma_rows_are_valid_and_build(row, dtype):
    spec = _mfma_spec(row, dtype)
    assert is_valid_dw_wgrad_mfma_spec(spec, "gfx950") == (True, "ok")
    kernel = build_dw_wgrad_mfma(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    name = spec.kernel_name()
    for flag, on in (("xcdp", spec.xcd_pair), ("swz", spec.swz), ("adir", spec.adir)):
        assert (f"_{flag}" in name) == on, (flag, name)
    grid, block = spec.launch_geometry()
    # stride 2 runs four stride-1 sub-problems side by side on z
    assert grid[2] == spec.n_z() * (4 if spec.problem.stride == 2 else 1)
    assert spec.phased == (spec.problem.stride == 2)
    if spec.xcd_pair:
        gx, gy = spec.n_channel_tiles(), spec.n_q()
        assert spec.problem.groups % 64 != 0 and gx >= 2
        assert (gy * grid[2]) % 8 == 0, (gy, grid[2])


def test_wgrad_mfma_stride2_sub_problem():
    """Stride 2: the main loop runs the stride-1 sub-problem of each input
    row / column parity, with its own filter extent and padding."""
    s3 = _mfma_spec(_WGRAD_MFMA_ROWS[0], "fp16")  # 3x3, pad 1
    assert (s3.pad, s3.kh, s3.kw, s3.hh) == (1, 2, 2, 10)
    s5 = _mfma_spec(_WGRAD_MFMA_ROWS[1], "fp16")  # 5x5, pad 2
    assert (s5.pad, s5.kh, s5.kw) == (1, 3, 3)
    flat = _mfma_spec(_WGRAD_MFMA_ROWS[2], "fp16")
    assert (flat.pad, flat.kh, flat.kw, flat.hh) == (1, 3, 3, 40)
    # the partial slots are per (q tile, z), not per phase
    assert s3.workspace_bytes() == s3.n_partitions() * 9 * 64 * 4


def test_wgrad_mfma_reasons():
    s = _mfma_spec(_WGRAD_MFMA_ROWS[0], "fp16")
    assert is_valid_dw_wgrad_mfma_spec(replace(s, unfold_n=2)) == (
        False,
        "stride 2: no image packing",
    )
    stride3 = replace(s, problem=replace(s.problem, stride=3))
    assert is_valid_dw_wgrad_mfma_spec(stride3) == (False, "stride 1 or 2 only")
    assert is_valid_dw_wgrad_mfma_spec(s, "gfx942") == (
        False,
        "ds_read_tr16_b64 needs gfx950",
    )


# ------------------------------------------------------------------ pp rows

# Grouped ping-pong dgrad: one group per block, block_k 64.
_PP_DGRAD_ROWS = [
    ("g4_c56", dict(groups=4, cpg=56, kpg=56)),
    ("g2_c64", dict(groups=2, cpg=64, kpg=64)),
    ("g3_c40_k72", dict(groups=3, cpg=40, kpg=72)),
]
# Ping-pong forward with valid padding (PAD 0): (id, KH, block_k).
_PP_FWD_ROWS = [("k3_bk64", 3, 64), ("k3_bk128", 3, 128), ("k5_bk128", 5, 128)]


def _pp_dgrad(row, dtype):
    return DirectConvProblem(
        N=2, H=20, W=24, KH=3, KW=3, PAD=1, stride=1, dtype=dtype, **row[1]
    )


def _pp_fwd(row, dtype):
    _, kh, _ = row
    return DirectConvProblem(
        N=2, H=20, W=24, groups=1, cpg=72, kpg=136, KH=kh, KW=kh, PAD=0, stride=1,
        dtype=dtype,
    )  # fmt: skip


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _PP_DGRAD_ROWS, ids=lambda r: r[0])
def test_pp_grouped_dgrad_rows(row, dtype):
    p = _pp_dgrad(row, dtype)
    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(p), block_k=64)
    assert dgrad_pp_mfma_reason(spec) is None
    kernel = build_dgrad_pp_mfma(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    # one group per block: the grid is the groups times a one-group grid
    one = replace(spec, problem=replace(spec.problem, groups=1))
    assert spec.launch_geometry()[0][0] == p.groups * one.launch_geometry()[0][0]
    # the dense 64-wide block stays refused, and so does 128 for a grouped one
    assert "block_k must be 64" in dgrad_pp_mfma_reason(replace(spec, block_k=128))


def test_pp_grouped_dgrad_refusals():
    p = _pp_dgrad(_PP_DGRAD_ROWS[0], "bf16")
    base = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(p), block_k=64)
    assert dgrad_pp_mfma_reason(replace(base, block_k=256)) == (
        "grouped: block_k must be 64 (128-byte weight rows)"
    )
    dense = replace(base, problem=replace(base.problem, groups=1))
    assert "block_k must be 128 or 256" in dgrad_pp_mfma_reason(dense)
    for arch in ("gfx942", "gfx90a"):
        assert "ds_read_b64_tr_b16" in dgrad_pp_mfma_reason(base, arch=arch)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _PP_FWD_ROWS, ids=lambda r: r[0])
def test_pp_forward_valid_padding_rows(row, dtype):
    spec = FwdPpMfmaSpec(problem=_pp_fwd(row, dtype), block_k=row[2])
    assert fwd_pp_mfma_reason(spec) is None
    kernel = build_fwd_pp_mfma(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert f"_p0_bk{row[2]}" in spec.kernel_name()
    p = spec.problem
    assert (p.Ho, p.Wo) == (20 - row[1] + 1, 24 - row[1] + 1)


def test_pp_forward_padding_refusals():
    spec = FwdPpMfmaSpec(problem=_pp_fwd(_PP_FWD_ROWS[1], "bf16"))
    p = spec.problem

    def why(**kw):
        return fwd_pp_mfma_reason(replace(spec, problem=replace(p, **kw)))

    assert why(PAD=0) is None
    assert why(PAD=2) == "'same' padding only: PAD must be 1 (got 2)"
    assert why(PAD=1) is None
    assert why(groups=2, cpg=36, kpg=68) == "groups must be 1"


# ------------------------------------------------------------- Winograd dgrad

_WINO_DGRAD_ROWS = [
    ("f4_fp16_v8", dict(mh=4, mw=4, vec=8, filter_rows=2), "fp16"),
    ("f4_bf16_v4", dict(mh=4, mw=4, io_dtype="bf16", vec=4), "bf16"),
]


def test_wino_dgrad_filter_spec():
    plain = wino.RtFilterSpec(mh=4, mw=4, vec=8, filter_rows=2)
    dg = wino.RtFilterSpec(mh=4, mw=4, vec=8, filter_rows=2, dgrad=True)
    assert dg.kernel_name() == plain.kernel_name() + "_dg"
    # the forward the dgrad runs as: C = the dgrad's K, K = its C
    p = ConvProblem(N=1, Hi=8, Wi=8, C=16, K=24, Y=3, X=3, pH=1, pW=1)
    geo = wino.RtGeometry(p, 4, 4)
    assert dg.reason(geo) is None
    g_plain, b_plain, v_plain = wino.stage_launch(plain, geo)
    g_dg, b_dg, v_dg = wino.stage_launch(dg, geo)
    assert (g_plain, b_plain) == (g_dg, b_dg)
    # the channel divisor moves from C / vec to K: the filter is read [C][3][3][K]
    assert v_plain["cgv"] == 16 // 8 and v_dg["cgv"] == 24
    assert {k: v for k, v in v_plain.items() if not k.startswith("cgv")} == {
        k: v for k, v in v_dg.items() if not k.startswith("cgv")
    }
    grouped = ConvProblem(N=1, Hi=8, Wi=8, C=16, K=24, Y=3, X=3, pH=1, pW=1, groups=2)
    assert dg.reason(wino.RtGeometry(grouped, 4, 4)) == (
        "the dgrad filter read is dense only"
    )
    assert plain.reason(wino.RtGeometry(grouped, 4, 4)) is None


@pytest.mark.parametrize("row", _WINO_DGRAD_ROWS, ids=lambda r: r[0])
def test_wino_dgrad_filter_builds(row):
    spec = wino.RtFilterSpec(dgrad=True, **row[1])
    kernel = wino.build_stage(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert [q.name for q in kernel.params] == [
        a["name"] for a in wino.stage_signature(spec)
    ]


def test_wino_dgrad_is_in_the_aot_grid():
    specs = wino.aot_stage_specs()
    dg = [s for s in specs if getattr(s, "dgrad", False)]
    fwd = [s for s in specs if isinstance(s, wino.RtFilterSpec) and not s.dgrad]
    assert len(dg) == len(fwd) == 68
    assert {s.kernel_name()[:-3] for s in dg} == {s.kernel_name() for s in fwd}


# ------------------------------------------------------------------- AOT grid

# Label counts of each family on gfx950 (fp16 + bf16 over the directions that
# have it) and what the families had before this change.
_AOT_COUNTS = {
    "direct_depthwise_win": (1707, 1518),
    "direct_depthwise_spatial": (96, 72),
    "direct_fwd_pp_mfma": (8, 6),
    _WIN_DGRAD: (2013, 1318),
    "direct_dgrad_pp_mfma": (8, 6),
    _WGRAD_MFMA: (1127, 912),
    _WGRAD_LDS: (458, 288),
}
_NEW_WIN_KNOBS = ("lds_swz", "xcd_tiles", "dma", "wdma", "dense")


@pytest.fixture(scope="module")
def gfx950_jobs():
    """Every job of the families above on gfx950, both dtypes, by direction."""
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    jobs = []
    for direction in ("fwd", "dgrad", "wgrad"):
        for dtype in _DTYPES:
            for job in dks.enumerate_direct_jobs(
                arch="gfx950", target=target, directions=(direction,), dtype=dtype
            ):
                if job.identity.algorithm in _AOT_COUNTS:
                    jobs.append((direction, dtype, job))
    return jobs


def test_aot_grid_counts(gfx950_jobs):
    labels: dict[str, set] = {}
    for _, dtype, job in gfx950_jobs:
        labels.setdefault(job.identity.algorithm, set()).add(job.identity.label())
    assert {a: len(s) for a, s in labels.items()} == {
        a: now for a, (now, _) in _AOT_COUNTS.items()
    }
    assert sum(now - before for now, before in _AOT_COUNTS.values()) == 1297


def test_aot_grid_other_arches_do_not_get_the_gfx950_knobs():
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    for arch in ("gfx942", "gfx1250"):
        target = ArchTarget.from_gfx(arch)
        for direction in ("fwd", "dgrad"):
            for job in dks.enumerate_direct_jobs(
                arch=arch, target=target, directions=(direction,), dtype="fp16"
            ):
                ident = job.identity
                knobs = json.loads(ident.knobs) if ident.knobs else {}
                if ident.algorithm in (_WIN_FWD, _WIN_DGRAD):
                    assert not any(knobs.get(k) for k in _NEW_WIN_KNOBS), (
                        arch,
                        ident.label(),
                    )
                if ident.algorithm == _SPATIAL:
                    assert not knobs.get("block_h"), (arch, ident.label())
                if ident.algorithm in (_PP_FWD, _PP_DGRAD):
                    pytest.fail(f"{ident.label()} on {arch}")


def test_aot_new_knob_labels_have_valid_specs(gfx950_jobs):
    """Each new-knob identity of the grid maps back from its label, and the
    spec that builds it is valid on gfx950 (a sample spread over the grid)."""
    from kernels.conv.direct_aot_jobs import direct_job_for_identity

    new = [job for _, _, job in gfx950_jobs if _is_new_knob(job.identity)]
    assert len(new) > 300
    for job in new[:: max(1, len(new) // 40)]:
        assert direct_job_for_identity(job.identity).identity == job.identity


def _is_new_knob(ident) -> bool:
    knobs = json.loads(ident.knobs) if ident.knobs else {}
    if ident.algorithm in (_WIN_FWD, _WIN_DGRAD):
        return any(knobs.get(k) for k in _NEW_WIN_KNOBS)
    if ident.algorithm == _SPATIAL:
        return bool(knobs.get("block_h"))
    if ident.algorithm == _WGRAD_LDS:
        return knobs.get("lane_ch", 8) != 8 or bool(knobs.get("xcd_pair"))
    if ident.algorithm == _WGRAD_MFMA:
        return any(knobs.get(k) for k in ("xcd_pair", "swz", "adir")) or (
            ident.stride_h == 2
        )
    return False


def test_packaged_launch_refuses_the_uncarried_forward_knobs(gfx950_jobs):
    """The packaged forward launch has no instance field for the swizzled /
    XCD-ordered / DMA / dense windowed forms or for the row-tiled spatial one:
    those 95 forward identities live in the rocke AOT cache only. The adapter
    refuses each with the knobs' own reason (an explicit, intended refusal,
    not a skip); every other identity of the two families is packaged."""
    from builders.common.convolution_forward import (
        aot_direct_instance_fields,
        gfx950_conv_fwd_aot_spec_for_identity,
    )

    refused: dict = {}
    packaged = 0
    for direction, dtype, job in gfx950_jobs:
        ident = job.identity
        if direction != "fwd" or ident.algorithm not in (_WIN_FWD, _SPATIAL):
            continue
        try:
            aot_direct_instance_fields(ident)
        except ValueError as err:
            assert _is_new_knob(ident), ident.label()
            assert str(err).startswith("knobs "), str(err)
            spec, reason = gfx950_conv_fwd_aot_spec_for_identity(ident)
            assert spec is None
            assert reason.startswith("knobs ") and "are not [" in reason
            refused[dtype, ident.algorithm] = (
                refused.get((dtype, ident.algorithm), 0) + 1
            )
        else:
            assert not _is_new_knob(ident), ident.label()
            packaged += 1
    assert refused == {
        ("fp16", _WIN_FWD): 42,
        ("bf16", _WIN_FWD): 29,
        ("fp16", _SPATIAL): 12,
        ("bf16", _SPATIAL): 12,
    }
    assert sum(refused.values()) == 95
    assert packaged > 1000


# ------------------------------------------------------------------------ GPU


def _tol_check(y, ref, dtype):
    y = np.asarray(y, dtype=np.float64)
    assert not np.isnan(y).any()
    rel = peak_relative_error(y, ref, floor=1.0)
    assert rel < _TOL[dtype], rel


def _compile(kernel):
    from rocke import compile_kernel

    return compile_kernel(kernel, arch="gfx950")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "row", [r for r in _WIN_ROWS if r[1] == _WIN_FWD], ids=lambda r: r[0]
)
def test_win_forward_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the windowed depthwise forward")
    spec = _win_spec(row, dtype)
    p = replace(row[2], dtype=dtype)
    art = _compile(build_direct_depthwise_dgrad_windowed(spec, arch="gfx950"))
    rng = np.random.default_rng(4321)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    grid, block = direct_launch_geometry(spec)
    y = launch_direct_conv(
        art,
        p,
        (x, w),
        poisoned((p.N, p.Ho, p.Wo, p.total_k), dtype, 0xFF),
        grid=grid,
        block=block,
    )
    ref = direct_fwd_reference(p, storage.decode(x, dtype), storage.decode(w, dtype))
    _tol_check(storage.decode(y, dtype), ref, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "row", [r for r in _WIN_ROWS if r[1] == _WIN_DGRAD], ids=lambda r: r[0]
)
def test_win_dgrad_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the windowed depthwise dgrad")
    spec = _win_spec(row, dtype)
    p = spec.problem  # dX extents; dY is Ho x Wo x K
    art = _compile(build_direct_depthwise_dgrad_windowed(spec, arch="gfx950"))
    rng = np.random.default_rng(1234)
    dy = stored_uniform(rng, (p.N, p.Ho, p.Wo, p.total_k), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    grid, block = direct_launch_geometry(spec)
    dx = launch_direct_conv(
        art,
        p,
        (dy, w),
        poisoned((p.N, p.H, p.W, p.total_c), dtype, 0xFF),
        grid=grid,
        block=block,
        direction="dgrad",
    )
    ref = direct_dgrad_reference(p, storage.decode(dy, dtype), storage.decode(w, dtype))
    _tol_check(storage.decode(dx, dtype), ref, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _SPATIAL_ROWS, ids=lambda r: r[0])
def test_spatial_row_tiles_match_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the row-tiled depthwise spatial")
    spec = _spatial_spec(row, dtype)
    p = spec.problem
    art = _compile(build_direct_depthwise_spatial(spec, arch="gfx950"))
    rng = np.random.default_rng(7)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    grid, block = direct_launch_geometry(spec)
    y = launch_direct_conv(
        art,
        p,
        (x, w),
        poisoned((p.N, p.Ho, p.Wo, p.total_k), dtype, 0xFF),
        grid=grid,
        block=block,
    )
    ref = direct_fwd_reference(p, storage.decode(x, dtype), storage.decode(w, dtype))
    _tol_check(storage.decode(y, dtype), ref, dtype)


def _run_dw_wgrad(spec, artifact, dtype):
    """Launch a depthwise wgrad kernel on a poisoned ``ws[part][tap][c]``
    scratch and return ``(dw[c, tap], reference[c, tap])``."""
    p = spec.problem
    rng = np.random.default_rng(42)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype, 0.5)
    dy = stored_uniform(rng, (p.N, p.Ho, p.Wo, p.total_k), dtype, 0.5)
    parts, taps = spec.n_partitions(), p.KH * p.KW
    assert spec.workspace_bytes() == parts * taps * p.groups * 4
    ws = DeviceArray(poisoned((parts, taps, p.groups), "fp32", 0xFF), "fp32")
    grid, block = spec.launch_geometry()
    launch_direct_kernel(
        artifact,
        p,
        (DeviceArray(dy, dtype), DeviceArray(x, dtype), ws),
        grid=grid,
        block=block,
        direction="wgrad",
    )
    dw = ws.read().sum(axis=0).T  # [c, tap]
    ref = direct_wgrad_reference(
        p, storage.decode(x, dtype), storage.decode(dy, dtype)
    ).reshape(p.groups, taps)
    return dw, ref


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WGRAD_LDS_ROWS, ids=lambda r: r[0])
def test_wgrad_lds_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the depthwise LDS-ring wgrad")
    spec = _lds_spec(row, dtype)
    art = _compile(build_dw_wgrad_lds(spec, arch="gfx950"))
    dw, ref = _run_dw_wgrad(spec, art, dtype)
    _tol_check(dw, ref, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WGRAD_MFMA_ROWS, ids=lambda r: r[0])
def test_wgrad_mfma_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the depthwise MFMA wgrad")
    spec = _mfma_spec(row, dtype)
    art = _compile(build_dw_wgrad_mfma(spec, arch="gfx950"))
    dw, ref = _run_dw_wgrad(spec, art, dtype)
    _tol_check(dw, ref, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _PP_DGRAD_ROWS, ids=lambda r: r[0])
def test_pp_grouped_dgrad_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the grouped ping-pong dgrad")
    p = _pp_dgrad(row, dtype)
    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(p), block_k=64)
    assert dgrad_pp_mfma_reason(spec) is None
    art = _compile(build_dgrad_pp_mfma(spec, arch="gfx950"))
    rng = np.random.default_rng(0)
    dy = stored_uniform(rng, (p.N, p.Ho, p.Wo, p.total_k), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    grid, block = spec.launch_geometry()
    # the spec's problem is the forward over the transposed problem
    dx = launch_direct_conv(
        art,
        spec.problem,
        (dy, w),
        poisoned((p.N, p.H, p.W, p.total_c), dtype, 0xFF),
        grid=grid,
        block=block,
    )
    ref = direct_dgrad_reference(p, storage.decode(dy, dtype), storage.decode(w, dtype))
    _tol_check(storage.decode(dx, dtype), ref, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _PP_FWD_ROWS, ids=lambda r: r[0])
def test_pp_forward_valid_padding_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the ping-pong forward")
    p = _pp_fwd(row, dtype)
    spec = FwdPpMfmaSpec(problem=p, block_k=row[2])
    assert fwd_pp_mfma_reason(spec) is None
    art = _compile(build_fwd_pp_mfma(spec, arch="gfx950"))
    rng = np.random.default_rng(0)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    grid, block = spec.launch_geometry()
    y = launch_direct_conv(
        art,
        p,
        (x, w),
        poisoned((p.N, p.Ho, p.Wo, p.total_k), dtype, 0xFF),
        grid=grid,
        block=block,
    )
    ref = direct_fwd_reference(p, storage.decode(x, dtype), storage.decode(w, dtype))
    _tol_check(storage.decode(y, dtype), ref, dtype)


@pytest.mark.parametrize("row", _WINO_DGRAD_ROWS, ids=lambda r: r[0])
def test_wino_dgrad_filter_equals_the_forward_filter_of_the_flipped_transpose(row):
    """K1 reading a dgrad filter ``[C][3][3][K]`` writes the same transformed
    filter as K1 on the forward filter ``[K][3][3][C]`` (the flipped transpose),
    for the same random filter; both outputs start poisoned."""
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    current_device_arch_in(("gfx950",), subject="the Winograd filter transform")
    _, knobs, io = row
    p = ConvProblem(N=1, Hi=8, Wi=8, C=16, K=24, Y=3, X=3, pH=1, pW=1)
    geo = wino.RtGeometry(p, knobs["mh"], knobs["mw"])
    rng = np.random.default_rng(5)
    w_fwd = stored_uniform(rng, (p.K, 3, 3, p.C), io)
    w_dg = np.ascontiguousarray(w_fwd[:, ::-1, ::-1, :].transpose(3, 1, 2, 0))
    u_elems = geo.n2 * p.K * p.cpg
    out = {}
    for dgrad, w in ((False, w_fwd), (True, w_dg)):
        spec = wino.RtFilterSpec(dgrad=dgrad, **knobs)
        art = _compile(wino.build_stage(spec, arch="gfx950"))
        grid, block, vals = wino.stage_launch(spec, geo)
        w_dev = DeviceArray(w, io)
        u_dev = DeviceArray(
            poisoned((u_elems,), spec.operand_dtype, 0xFF), spec.operand_dtype
        )
        KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=wino.stage_signature(spec),
        )(
            {
                "w": w_dev.data_ptr(),
                "u": u_dev.data_ptr(),
                "w_bytes": w_dev.nbytes,
                "u_bytes": u_dev.nbytes,
                **vals,
            },
            config=LaunchConfig(grid=grid, block=(block, 1, 1), fence=True),
        )
        out[dgrad] = storage.decode(u_dev.read(), spec.operand_dtype)
    assert not np.isnan(out[True]).any()
    assert np.array_equal(out[True], out[False])
