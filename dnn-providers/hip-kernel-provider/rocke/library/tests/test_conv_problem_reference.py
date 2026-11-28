# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The float64 references of a ``ConvProblem`` (``builders.common.conv_reference``).

They hand a problem's geometry to ``rocke.numeric.conv``; these checks pin that
mapping: the per-axis order (depth first), the output extents the problem
itself computes, that forward, data gradient and weight gradient are the
adjoints of one map, and that running the batch in image slices leaves the
result unchanged. CPU only; no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from builders.common.conv_reference import (
    conv_dgrad_reference,
    conv_extents,
    conv_fwd_reference,
    conv_geometry,
    conv_wgrad_reference,
)
from kernels.common._conv_implicit_gemm_common import ConvProblem

# Every per-axis value differs from its neighbours, so a swapped axis changes
# the output shape or breaks the adjoint identities.
_PROBLEM_2D = ConvProblem(
    N=2, Hi=11, Wi=9, C=6, K=4, Y=3, X=2, sH=2, sW=1, pH=1, pW=2, dH=1, dW=2, groups=2
)  # fmt: skip
_PROBLEM_3D = ConvProblem(
    N=2, Di=7, Hi=8, Wi=9, C=6, K=9, Z=2, Y=3, X=2, sD=1, sH=2, sW=3, pD=1, pH=0,
    pW=2, dD=2, dH=1, dW=1, groups=3,
)  # fmt: skip


def test_extents_are_per_axis_depth_first():
    p = _PROBLEM_3D
    assert conv_extents(p) == ((7, 8, 9), (p.Do, p.Ho, p.Wo), (2, 3, 2))
    assert conv_extents(_PROBLEM_2D) == (
        (11, 9),
        (_PROBLEM_2D.Ho, _PROBLEM_2D.Wo),
        (3, 2),
    )


def test_geometry_is_per_axis_depth_first():
    assert conv_geometry(_PROBLEM_2D) == {
        "stride": (2, 1),
        "padding": (1, 2),
        "dilation": (1, 2),
        "groups": 2,
    }
    assert conv_geometry(_PROBLEM_3D) == {
        "stride": (1, 2, 3),
        "padding": (1, 0, 2),
        "dilation": (2, 1, 1),
        "groups": 3,
    }


@pytest.mark.parametrize("problem", [_PROBLEM_2D, _PROBLEM_3D], ids=["2d", "3d"])
def test_references_are_adjoints_in_the_kernel_layouts(problem):
    p = problem
    in_spatial, out_spatial, filter_spatial = conv_extents(p)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((p.N, *in_spatial, p.C))
    w = rng.standard_normal((p.K, *filter_spatial, p.cpg))
    dy = rng.standard_normal((p.N, *out_spatial, p.K))

    y = conv_fwd_reference(x, w, p)
    dx = conv_dgrad_reference(dy, w, p)
    dw = conv_wgrad_reference(x, dy, p)
    assert y.shape == dy.shape
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    forward = float(np.vdot(y, dy))
    assert float(np.vdot(x, dx)) == pytest.approx(forward, rel=1e-12)
    assert float(np.vdot(w, dw)) == pytest.approx(forward, rel=1e-12)


@pytest.mark.parametrize("problem", [_PROBLEM_2D, _PROBLEM_3D], ids=["2d", "3d"])
def test_image_slices_do_not_change_the_result(problem):
    """A slice cap below one image runs every image alone; the result is the
    whole-batch one (wgrad sums the slices)."""
    p = problem
    in_spatial, out_spatial, filter_spatial = conv_extents(p)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((p.N, *in_spatial, p.C))
    w = rng.standard_normal((p.K, *filter_spatial, p.cpg))
    dy = rng.standard_normal((p.N, *out_spatial, p.K))
    one_image = {"max_slice_bytes": 1}
    np.testing.assert_allclose(
        conv_fwd_reference(x, w, p, **one_image),
        conv_fwd_reference(x, w, p),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        conv_dgrad_reference(dy, w, p, **one_image),
        conv_dgrad_reference(dy, w, p),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        conv_wgrad_reference(x, dy, p, **one_image),
        conv_wgrad_reference(x, dy, p),
        rtol=1e-12,
        atol=1e-12,
    )
