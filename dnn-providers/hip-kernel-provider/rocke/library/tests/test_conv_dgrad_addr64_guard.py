# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``DgradConvSpec.addr64``'s validate() guards: the scope the once-per-CTA
i64 M-tile-origin decode has actually been wired for (2D, MFMA, stride 1
only -- ``folds_sub_gemm_record`` -- non-pointwise, non-patch,
non-lds_k_outer, non-dy_halo, non-tap_outer_k, non-uniform_k_decode,
plain sync K-loop, split_k == 1, and a Hi*Wi-multiple-of-tile_m precondition
the addr64 descriptors rely on -- see
make_dgrad_dx_descriptor_addr64/dgrad_m_unmerge_addr64's docstrings).
Grouped conv is in scope and accepted (the group fold rides the bounded
kpg/cpg axis, orthogonal to the i64 spatial decode). CPU only; GPU numeric
coverage (plain/grouped, plus a real >2^31-byte tile-origin run) is a
follow-up outside this file's scope.
"""

from __future__ import annotations

import pytest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

_FP16 = ConvDataSpec("fp16", "fp16", "fp16")
# Hi = Wi = 16, Y=X=3, pH=pW=1, sH=sW=1: Hi*Wi=256 is a multiple of
# tile_m=64, so this geometry satisfies the addr64 image-boundary
# precondition.
_PROBLEM = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)


def _dgrad(**kwargs):
    defaults = dict(
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )
    defaults.update(kwargs.pop("tile_overrides", {}))
    return DgradConvSpec(problem=kwargs.pop("problem", _PROBLEM), data=_FP16, **defaults, **kwargs)


def test_addr64_plain_validates():
    _dgrad(addr64=True).validate()


def test_addr64_grouped_validates():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
    _dgrad(problem=p, addr64=True).validate()


def test_addr64_default_is_off_and_unaffected():
    _dgrad().validate()


def test_addr64_rejects_is_3d():
    p = ConvProblem(
        N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1,
        Z=3, Di=16, sD=1, pD=1, dD=1,
    )
    with pytest.raises(ValueError, match="is_3d"):
        _dgrad(problem=p, addr64=True).validate()


def test_addr64_rejects_is_pointwise():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=1, X=1, pH=0, pW=0)
    with pytest.raises(ValueError, match="is_pointwise"):
        _dgrad(problem=p, addr64=True).validate()


def test_addr64_accepts_strided_when_records_are_tile_m_aligned():
    # Hi = Wi = 16, stride 2, no pad: every tilde record is 8x8 = 64, a
    # multiple of tile_m=64, so no M-tile crosses an image boundary --
    # general stride is now in scope (the record's own h_tilde_slice/
    # w_tilde_slice generalize the stride-1 decode; see the addr64 field's
    # docstring).
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=0, pW=0, sH=2, sW=2)
    _dgrad(problem=p, addr64=True).validate()


def test_addr64_rejects_patch():
    with pytest.raises(ValueError, match="patch"):
        _dgrad(addr64=True, patch=True).validate()


def test_addr64_rejects_lds_k_outer():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="lds_k_outer"):
        _dgrad(
            problem=p,
            addr64=True,
            lds_k_outer=True,
            tile_overrides=dict(warp_tile_n=16),
        ).validate()


def test_addr64_rejects_unroll_and_async_dma():
    with pytest.raises(ValueError, match="async_dma/unroll_k"):
        _dgrad(addr64=True, unroll_k=True).validate()
    with pytest.raises(ValueError, match="async_dma/unroll_k"):
        _dgrad(addr64=True, async_dma=True).validate()


def test_addr64_rejects_tap_outer_k():
    with pytest.raises(ValueError, match="tap_outer_k"):
        _dgrad(addr64=True, tap_outer_k=True).validate()


def test_addr64_rejects_dy_halo():
    with pytest.raises(ValueError, match="dy_halo"):
        _dgrad(addr64=True, dy_halo=1, tile_overrides=dict(halo_max_wo=16)).validate()


def test_addr64_rejects_uniform_k_decode():
    with pytest.raises(ValueError, match="uniform_k_decode"):
        _dgrad(addr64=True, uniform_k_decode=True).validate()


def test_addr64_rejects_wavelet_pipeline():
    with pytest.raises(ValueError, match="wavelet"):
        _dgrad(addr64=True, pipeline="wavelet").validate()


def test_addr64_rejects_split_k():
    with pytest.raises(ValueError, match="split_k"):
        _dgrad(addr64=True, split_k=2).validate()
    with pytest.raises(ValueError, match="split_k"):
        _dgrad(addr64=True, split_k=-1).validate()


def test_addr64_requires_hi_wi_multiple_of_tile_m():
    # Hi = Wi = 14: 196 is not a multiple of tile_m=64 -- an M-tile could
    # straddle an image boundary. Stride 1, one record spanning the whole
    # grid: h_tilde_slice*w_tilde_slice == Hi*Wi.
    p = ConvProblem(N=4, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="h_tilde_slice\\*w_tilde_slice"):
        _dgrad(problem=p, addr64=True).validate()


def test_addr64_rejects_strided_when_records_are_not_tile_m_aligned():
    # Hi = Wi = 16, stride 2, pad 1: each tilde record is 17x17 = 289, not a
    # multiple of tile_m=64.
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, sH=2, sW=2)
    with pytest.raises(ValueError, match="h_tilde_slice\\*w_tilde_slice"):
        _dgrad(problem=p, addr64=True).validate()


def test_addr64_kernel_name_tags_when_set():
    name = _dgrad(addr64=True).kernel_name()
    assert "a64" in name
    assert "a64" not in _dgrad().kernel_name()


# ---------------------------------------------------------------------
# Split-K resolver overflow guard (kernels.common.conv_split_k): splitting K
# partitions the reduction, not the absolute byte range a non-addr64
# launch's CTAs must reach (dY/dX byte offsets are absolute from the tensor
# base), so no split-K degree narrows an already-overflowing problem.
# resolve_conv_split_k(addr64=False) must fail fast instead of handing back
# a degree that still produces a corrupting kernel.
# ---------------------------------------------------------------------

from kernels.common.conv_split_k import resolve_conv_split_k

# N chosen so N*Hi*Wi*C*2 bytes (fp16) exceeds 2**31 - 1 with Hi=Wi=16, C=64.
_HUGE_N = 65537
_HUGE_PROBLEM = ConvProblem(N=_HUGE_N, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)


def _resolve(problem=_HUGE_PROBLEM, **kwargs):
    return resolve_conv_split_k(
        "dgrad",
        problem,
        arch="gfx950",
        dtype="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        block_size=256,
        **kwargs,
    )


def test_split_k_resolver_rejects_overflowing_non_addr64_dgrad():
    with pytest.raises(ValueError, match="addr64"):
        _resolve()


def test_split_k_resolver_allows_overflowing_addr64_dgrad():
    # No ValueError: addr64's i64 addressing makes the problem safe to
    # address at degree 1 (the only degree addr64 dgrad supports today), so
    # the guard does not apply.
    _resolve(addr64=True, split_k=1)


def test_split_k_resolver_unaffected_for_small_problems():
    small = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    _resolve(small)  # does not raise, addr64 default False
