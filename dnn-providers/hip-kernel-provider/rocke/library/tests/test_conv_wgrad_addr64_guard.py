# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``WgradConvSpec.addr64``'s validate() guards: the scope the K-loop's
per-iteration i64 tile-origin decode has actually been wired for (2D, MFMA,
non-pointwise, non-group_merge, non-streamk, plain sync K-loop, and a
Ho*Wo-multiple-of-tile_k precondition the addr64 descriptors rely on -- see
make_dy_descriptor_addr64/make_x_wgrad_descriptor_addr64's docstrings).
Grouped conv, split-K, and two-stage are all in scope and accepted. CPU only;
GPU numeric coverage (plain/grouped/split-K/two-stage, plus a real >2^31-byte
tile-origin run) lives outside this file's scope -- see the session's PR
description for those commands.
"""

from __future__ import annotations

import pytest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

_FP16 = ConvDataSpec("fp16", "fp16", "fp32")
# Ho = Wo = 16 (Hi=Wi=16, Y=X=3, pH=pW=1, sH=sW=1): 16 is a multiple of
# tile_k=16, so this geometry satisfies the addr64 image-boundary precondition.
_PROBLEM = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)


def _wgrad(**kwargs):
    defaults = dict(
        tile_m=64,
        tile_n=64,
        tile_k=16,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )
    defaults.update(kwargs.pop("tile_overrides", {}))
    return WgradConvSpec(problem=kwargs.pop("problem", _PROBLEM), data=_FP16, **defaults, **kwargs)


def test_addr64_plain_validates():
    _wgrad(addr64=True).validate()


def test_addr64_grouped_validates():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
    _wgrad(problem=p, addr64=True).validate()


def test_addr64_split_k_and_two_stage_validate():
    _wgrad(addr64=True, split_k=4).validate()
    _wgrad(addr64=True, split_k=4, two_stage=True).validate()


def test_addr64_default_is_off_and_unaffected():
    _wgrad().validate()


def test_addr64_rejects_is_3d():
    p = ConvProblem(
        N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1,
        Z=3, Di=16, sD=1, pD=1, dD=1,
    )
    with pytest.raises(ValueError, match="is_3d"):
        _wgrad(problem=p, addr64=True).validate()


def test_addr64_rejects_is_pointwise():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=1, X=1, pH=0, pW=0)
    with pytest.raises(ValueError, match="is_pointwise"):
        _wgrad(problem=p, addr64=True, tile_overrides=dict(tile_k=16)).validate()


def test_addr64_rejects_group_merge():
    p = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
    with pytest.raises(ValueError, match="group_merge"):
        _wgrad(problem=p, addr64=True, group_merge=2).validate()


def test_addr64_rejects_streamk():
    with pytest.raises(ValueError, match="stream-K"):
        _wgrad(addr64=True, streamk="dp_sk").validate()


def test_addr64_rejects_chiplet_swizzle():
    with pytest.raises(ValueError, match="chiplet_swizzle"):
        _wgrad(addr64=True, chiplet_swizzle=True).validate()


def test_addr64_rejects_unroll_and_prefetch_and_lds_k_outer():
    with pytest.raises(ValueError, match="async_dma/unroll_k/prefetch_addr"):
        _wgrad(addr64=True, unroll_k=True).validate()
    with pytest.raises(ValueError, match="async_dma/unroll_k/prefetch_addr"):
        _wgrad(addr64=True, prefetch_addr=True).validate()
    with pytest.raises(ValueError, match="prefetch_global_regs"):
        _wgrad(addr64=True, prefetch_global_regs=True).validate()
    with pytest.raises(ValueError, match="lds_k_outer"):
        _wgrad(addr64=True, lds_k_outer=True).validate()


def test_addr64_requires_ho_wo_multiple_of_tile_k():
    # Ho = Wo = 14 (Hi=Wi=14, Y=X=3, pH=pW=1, sH=sW=1): 196 is not a multiple
    # of tile_k=16 -- a K-tile could straddle an image boundary.
    p = ConvProblem(N=4, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="Ho\\*Wo"):
        _wgrad(problem=p, addr64=True).validate()


def test_addr64_kernel_name_tags_when_set():
    name = _wgrad(addr64=True).kernel_name()
    assert "a64" in name
    assert "a64" not in _wgrad().kernel_name()


# ---------------------------------------------------------------------
# Split-K resolver overflow guard (kernels.common.conv_split_k): splitting K
# partitions which CTA computes a given K_wg index, not the absolute byte
# offset that index maps to, so no split-K degree narrows the i32 byte range
# a non-addr64 launch's CTAs must reach. resolve_conv_split_k(addr64=False)
# must fail fast on an overflowing problem instead of handing back a degree
# that still produces a corrupting kernel.
# ---------------------------------------------------------------------

from kernels.common.conv_split_k import resolve_conv_split_k

# N chosen so N*Ho*Wo*K*2 bytes (fp16) exceeds 2**31 - 1 with Ho=Wo=16, K=64.
_HUGE_N = 65537
_HUGE_PROBLEM = ConvProblem(N=_HUGE_N, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)


def _resolve(problem=_HUGE_PROBLEM, **kwargs):
    return resolve_conv_split_k(
        "wgrad",
        problem,
        arch="gfx950",
        dtype="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=16,
        block_size=256,
        **kwargs,
    )


def test_split_k_resolver_rejects_overflowing_non_addr64_wgrad():
    with pytest.raises(ValueError, match="addr64"):
        _resolve()


def test_split_k_resolver_allows_overflowing_addr64_wgrad():
    # No ValueError: addr64's i64 addressing makes the problem safe to split
    # at any degree, so the guard does not apply.
    _resolve(addr64=True)


def test_split_k_resolver_unaffected_for_small_problems():
    small = ConvProblem(N=4, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    _resolve(small)  # does not raise, addr64 default False


def test_split_k_resolver_checks_the_operand_dtype_not_only_the_output_dtype():
    # At N=40000 (Ho=Wo=16, K=64): dY/X bytes at fp32 (4 bytes/elem) exceed
    # i32, but at fp16 (2 bytes/elem, the output dtype here) they do not.
    # Passing dtype_a/dtype_b="fp32" must catch the real operand size instead
    # of silently passing on the (here, too-small) output-dtype fallback.
    mixed = ConvProblem(N=40000, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="addr64"):
        _resolve(mixed, dtype_a="fp32", dtype_b="fp32")
    # The fallback (no dtype_a/dtype_b) sizes dY/X as the output dtype
    # (fp16) and so does not catch this particular mismatch -- callers with
    # a mixed-precision spec must pass dtype_a/dtype_b explicitly.
    _resolve(mixed)
