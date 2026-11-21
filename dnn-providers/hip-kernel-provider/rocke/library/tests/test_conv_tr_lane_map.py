# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only: the wave32 transpose-read lane map the conv readers share is right.

``ConvKOuterFragmentReader.wave32_lane_offsets`` is the ``ds_load_tr16_b128``
lane map of a K-outer conv LDS tile. The implicit-GEMM fragment reader and the
three gfx1250 direct wgrad builders (dense, image-per-block, small_c) all take
it from there, so a fix to the map reaches each of them. The wave32 dgrad WMMA
builders (``conv_direct_dgrad_wmma``, ``conv_dgrad_patch_wmma``,
``conv_dgrad_smallc_wmma``) still compute the same map inline. The C++ engine
mirrors the helper in ``rocke_conv_tr_lane_offsets32``; the byte-identity gate
keeps the two in step.
"""

from __future__ import annotations

import pytest

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvWgradSpec,
    build_direct_conv_wgrad,
)
from kernels.common._attention_bwd_frag import IntLaneBuilder
from kernels.common._conv_layouts import ConvKOuterFragmentReader


@pytest.mark.parametrize("n", [8, 16])
def test_wave32_lane_map_values(n):
    """Lane l addresses the 8-column block holding column l % 16 and supplies
    K row (l // 16) * n + l % 8 of it; the bases are added when given."""
    b = IntLaneBuilder()
    for lane in range(32):
        col, row = ConvKOuterFragmentReader.wave32_lane_offsets(b, lane, n=n)
        assert col == ((lane % 16) // 8) * 8
        assert row == (lane // 16) * n + lane % 8
        col, row = ConvKOuterFragmentReader.wave32_lane_offsets(
            b, lane, n=n, mn_base=100, k_base=1000
        )
        assert col == 100 + ((lane % 16) // 8) * 8
        assert row == 1000 + (lane // 16) * n + lane % 8


def _dense():
    p = DirectConvProblem(
        N=2, H=8, W=37, groups=1, cpg=64, kpg=64, KH=3, KW=3, PAD=1, stride=1
    )
    return DirectConvWgradSpec(problem=p, wave_size=32)


def _image():
    p = DirectConvProblem(
        N=2, H=8, W=37, groups=1, cpg=64, kpg=64, KH=3, KW=3, PAD=1, stride=1
    )
    return DirectConvWgradSpec(
        problem=p, wave_size=32, image_per_block=True, waves_k=2, waves_c=2
    )


def _small_c():
    p = DirectConvProblem(
        N=2, H=13, W=37, groups=1, cpg=3, kpg=32, KH=3, KW=3, PAD=1, stride=1
    )
    return DirectConvWgradSpec(problem=p, wave_size=32, small_c=True)


@pytest.mark.parametrize(
    "make_spec", [_dense, _image, _small_c], ids=["dense", "image", "small_c"]
)
def test_direct_wgrad_w32_builders_use_the_shared_lane_map(monkeypatch, make_spec):
    """Each wave32 direct wgrad builder takes its transpose-read address from
    the shared map, once, with the 16-element WMMA fragment."""
    calls = []
    real = ConvKOuterFragmentReader.wave32_lane_offsets

    def spy(b, lane, *, n, mn_base=None, k_base=None):
        calls.append(n)
        return real(b, lane, n=n, mn_base=mn_base, k_base=k_base)

    monkeypatch.setattr(
        ConvKOuterFragmentReader, "wave32_lane_offsets", staticmethod(spy)
    )
    build_direct_conv_wgrad(make_spec(), arch="gfx1250")
    assert calls == [16]
