# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the float64 convolution references in ``rocke.numeric.conv``.

The forward is checked against a scalar loop written straight from the defining
sum. The two gradients are checked through the adjoint identities
``<conv_fwd(x, w), dy> == <x, conv_dgrad(dy, w)> == <w, conv_wgrad(x, dy)>``,
which hold for exactly one pair of linear maps, so no second implementation of
either gradient is needed.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

from rocke.numeric.conv import conv_dgrad, conv_fwd, conv_output_spatial, conv_wgrad

# (in_spatial, filter_spatial, stride, padding, dilation, groups): 1-D, 2-D and 3-D,
# grouped and depthwise, every geometry knob above its default somewhere.
GEOMETRIES = [
    ((11,), (3,), 1, 1, 1, 1),
    ((12,), (4,), 3, 2, 2, 2),
    ((7, 9), (3, 3), 1, 1, 1, 1),
    ((9, 8), (3, 2), (2, 1), (1, 0), (1, 2), 2),
    ((6, 6), (1, 1), 2, 0, 1, 4),
    ((8, 7), (3, 3), 2, 2, 2, 6),
    ((5, 6, 4), (3, 2, 2), (1, 2, 1), (1, 0, 1), (2, 1, 1), 2),
]


def _case(rng, in_spatial, filter_spatial, groups, *, channels_per_group=3, kg=2):
    c, k = channels_per_group * groups, kg * groups
    x = rng.standard_normal((2, *in_spatial, c))
    w = rng.standard_normal((k, *filter_spatial, channels_per_group))
    return x, w


def _loop_conv_fwd(x, w, stride, padding, dilation, groups):
    """Scalar loop over every output element and tap, from the defining sum."""
    rank = x.ndim - 2
    per_axis = lambda v: (v,) * rank if isinstance(v, int) else v
    stride, padding, dilation = map(per_axis, (stride, padding, dilation))
    n, *in_spatial, _ = x.shape
    k, *filter_spatial, cg = w.shape
    kg = k // groups
    out_spatial = [
        (i + 2 * p - d * (f - 1) - 1) // s + 1
        for i, f, s, p, d in zip(in_spatial, filter_spatial, stride, padding, dilation)
    ]
    y = np.zeros((n, *out_spatial, k))
    for out_pos in itertools.product(*map(range, out_spatial)):
        for tap in itertools.product(*map(range, filter_spatial)):
            in_pos = [
                o * s - p + t * d
                for o, t, s, p, d in zip(out_pos, tap, stride, padding, dilation)
            ]
            if any(not 0 <= i < e for i, e in zip(in_pos, in_spatial)):
                continue
            for out_channel in range(k):
                group = out_channel // kg
                x_row = x[(slice(None), *in_pos, slice(group * cg, (group + 1) * cg))]
                y[(slice(None), *out_pos, out_channel)] += (
                    x_row @ w[(out_channel, *tap)]
                )
    return y


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"{len(g[0])}d")
def test_forward_matches_scalar_loop(geometry):
    in_spatial, filter_spatial, stride, padding, dilation, groups = geometry
    rng = np.random.default_rng(len(in_spatial) * 10 + groups)
    x, w = _case(rng, in_spatial, filter_spatial, groups)
    kw = {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups}
    got = conv_fwd(x, w, **kw)
    want = _loop_conv_fwd(x, w, stride, padding, dilation, groups)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"{len(g[0])}d")
def test_gradients_are_the_adjoints_of_the_forward(geometry):
    in_spatial, filter_spatial, stride, padding, dilation, groups = geometry
    rng = np.random.default_rng(100 + len(in_spatial) * 10 + groups)
    x, w = _case(rng, in_spatial, filter_spatial, groups)
    kw = {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups}
    y = conv_fwd(x, w, **kw)
    dy = rng.standard_normal(y.shape)
    dx = conv_dgrad(dy, w, in_spatial=in_spatial, **kw)
    dw = conv_wgrad(x, dy, filter_spatial=filter_spatial, **kw)
    assert dx.shape == x.shape and dw.shape == w.shape
    forward_pairing = float((y * dy).sum())
    scale = max(1.0, abs(forward_pairing))
    assert abs(float((x * dx).sum()) - forward_pairing) <= 1e-12 * scale
    assert abs(float((w * dw).sum()) - forward_pairing) <= 1e-12 * scale


def test_taps_in_the_padding_read_zero():
    """A 1x1 image with padding 1 and a 3x3 filter sees only the centre tap."""
    x = np.full((1, 1, 1, 1), 2.0)
    w = np.arange(9.0).reshape(1, 3, 3, 1)
    y = conv_fwd(x, w, padding=1)
    assert y.shape == (1, 1, 1, 1) and y[0, 0, 0, 0] == 2.0 * w[0, 1, 1, 0]


def test_input_extents_a_stride_skips_get_zero_data_gradient():
    """With stride 2 and a 1-wide filter, odd input columns feed no output."""
    dy = np.ones((1, 3, 1))
    dx = conv_dgrad(dy, np.ones((1, 1, 1)), in_spatial=(6,), stride=2)
    np.testing.assert_array_equal(dx[0, :, 0], [1, 0, 1, 0, 1, 0])


def test_output_spatial():
    assert conv_output_spatial((7, 9), (3, 3), stride=2, padding=1) == (4, 5)
    assert conv_output_spatial((12,), (4,), stride=3, padding=2, dilation=2) == (4,)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: conv_fwd(np.zeros((1, 4, 4, 6)), np.zeros((4, 3, 3, 2)), groups=4),
            "C=6, groups=4",
        ),
        (
            lambda: conv_fwd(np.zeros((1, 4, 4, 6)), np.zeros((4, 3, 3, 6)), groups=2),
            "w channels=6, expected C/groups=3",
        ),
        (
            lambda: conv_fwd(np.zeros((1, 2, 2, 1)), np.zeros((1, 3, 3, 1))),
            "output extents",
        ),
        (lambda: conv_fwd(np.zeros((4, 1)), np.zeros((1, 1, 1))), "x.ndim=2"),
        (lambda: conv_fwd(np.zeros((1, 4, 4, 1)), np.zeros((1, 3, 1))), "w.ndim=3"),
        (
            lambda: conv_fwd(np.zeros((1, 4, 4, 1)), np.zeros((1, 3, 3, 1)), stride=0),
            "expected stride >= 1",
        ),
        (
            lambda: conv_fwd(
                np.zeros((1, 4, 4, 1)), np.zeros((1, 3, 3, 1)), padding=(1,)
            ),
            "padding=(1,), expected an int or 2 ints",
        ),
        (
            lambda: conv_dgrad(
                np.zeros((1, 3, 3, 1)), np.zeros((1, 3, 3, 1)), in_spatial=(4, 4)
            ),
            "dy spatial=(3, 3), expected (2, 2)",
        ),
        (
            lambda: conv_dgrad(
                np.zeros((1, 2, 2, 2)), np.zeros((1, 3, 3, 1)), in_spatial=(4, 4)
            ),
            "w.shape[0]=1, expected K=2",
        ),
        (
            lambda: conv_wgrad(
                np.zeros((1, 4, 4, 1)), np.zeros((2, 2, 2, 1)), filter_spatial=(3, 3)
            ),
            "dy.shape[0]=2, expected N=1",
        ),
        (
            lambda: conv_wgrad(
                np.zeros((1, 4, 4, 1)), np.zeros((1, 3, 3, 1)), filter_spatial=(3, 3)
            ),
            "dy spatial=(3, 3), expected (2, 2)",
        ),
    ],
)
def test_refusals(call, message):
    with pytest.raises(ValueError, match=f"failed - .*{re.escape(message)}"):
        call()
