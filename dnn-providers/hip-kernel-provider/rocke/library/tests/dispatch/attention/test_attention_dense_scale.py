# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Both dense runners refuse a softmax scale outside [2**-64, 2**4] before they
build a kernel: NaN and +-inf as invalid input, a finite scale as a kernel limit."""

import math
from types import SimpleNamespace
from unittest import mock

import pytest
from kernels.common.attention_dense_spec import DENSE_MAX_SCALE, DENSE_MIN_SCALE
from kernels.gfx942 import attention_dense as gfx942_dense
from kernels.gfx950 import attention_dense as gfx950_dense

_RUNNERS = {
    "gfx942": (gfx942_dense.Gfx942AttentionDenseSpec, gfx942_dense),
    "gfx950": (gfx950_dense.Gfx950AttentionDenseSpec, gfx950_dense),
}


class _ReachedBuild(Exception):
    """Raised in place of the kernel build: the runner got past every check."""


def _launch(arch, scale):
    """Run ``arch``'s runner on shape-only stand-ins. A scale the guard passes
    raises :class:`_ReachedBuild`; one it refuses raises before the build."""
    spec_type, module = _RUNNERS[arch]
    spec = spec_type(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=64,
        dtype="bf16",
    )
    qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
    kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
    with (
        mock.patch.dict(module._DENSE_LAUNCHER_CACHE, clear=True),
        mock.patch.object(module, "build_attention_dense", side_effect=_ReachedBuild),
    ):
        module.run_attention_dense_torch(
            spec=spec,
            q=SimpleNamespace(shape=qshape),
            k=SimpleNamespace(shape=kvshape),
            v=SimpleNamespace(shape=kvshape),
            out=SimpleNamespace(shape=qshape),
            scale=scale,
        )


def test_bounds_match_the_hipdnn_packs():
    """isSupportedScale (AttentionDenseNative.cpp) hardcodes 0x1p-64F / 0x1p4F for
    both arches; moving these alone would make rocKE and hipDNN serve different
    scales."""
    msg = "update isSupportedScale in AttentionDenseNative.cpp and its gtest together"
    assert DENSE_MIN_SCALE == 2.0**-64, msg
    assert DENSE_MAX_SCALE == 2.0**4, msg


@pytest.mark.parametrize("arch", sorted(_RUNNERS))
def test_dense_scale_range_is_the_runners_range(arch):
    """``dense_scale_range`` is the public lookup the SDPA gate declines with; it
    must name the range the runners enforce, not drift to its own copy."""
    from dispatch.attention import dense_scale_range

    assert dense_scale_range(arch) == (DENSE_MIN_SCALE, DENSE_MAX_SCALE)


_REFUSED = [
    (float("nan"), ValueError, "scale must be finite"),
    (float("inf"), ValueError, "scale must be finite"),
    (float("-inf"), ValueError, "scale must be finite"),
] + [
    (scale, NotImplementedError, r"NOT_YET_IMPLEMENTED.*\[2\*\*-64, 2\*\*4\]")
    for scale in (
        0.0,
        -0.0,
        -0.125,
        1e-30,
        math.nextafter(DENSE_MIN_SCALE, 0.0),
        math.nextafter(DENSE_MAX_SCALE, math.inf),
        1e30,
    )
]


@pytest.mark.parametrize("arch", sorted(_RUNNERS))
@pytest.mark.parametrize("scale,error,match", _REFUSED)
def test_out_of_range_scale_refused_before_the_build(arch, scale, error, match):
    with pytest.raises(error, match=match):
        _launch(arch, scale)


@pytest.mark.parametrize("arch", sorted(_RUNNERS))
@pytest.mark.parametrize("scale", [DENSE_MIN_SCALE, 1 / math.sqrt(64), DENSE_MAX_SCALE])
def test_inclusive_bounds_reach_the_build(arch, scale):
    with pytest.raises(_ReachedBuild):
        _launch(arch, scale)
