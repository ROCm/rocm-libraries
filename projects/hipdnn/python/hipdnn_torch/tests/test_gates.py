# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-CPU, provider-free tests for the override *gates* and pure helpers.

The gate is the contract that keeps the injection layer safe: anything the catalog
adapter cannot serve must be declined (returning ``(False, reason)``) so the caller
falls back to native PyTorch. These tests drive ``_gate`` directly with fake tensors
and a fake ``state`` -- no bootstrap, no GPU, no provider ``.so`` -- so they run in
CI on any box and pin the decline reasons that show up in the fallback census.
"""

from types import SimpleNamespace

import pytest

from hipdnn_torch.activation import GeluOverride, SiluOverride
from hipdnn_torch.conv import Conv2dFpropOverride, _pair
from hipdnn_torch.layernorm import LayerNormOverride


class _FakeDtype:
    """A stand-in for a ``torch.dtype`` -- a unique, hashable sentinel with a repr."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<dtype {self.name}>"


# The three dtypes the gates care about. Identity is all that matters (the gates and
# _tok compare against state.torch.float16 etc.), so distinct objects suffice.
F16 = _FakeDtype("f16")
BF16 = _FakeDtype("bf16")
F32 = _FakeDtype("f32")


class _FakeTensor:
    """Minimal tensor: just the attributes the gates read."""

    def __init__(self, shape, dtype=F16, is_cuda=True):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)


def _fake_torch():
    return SimpleNamespace(float16=F16, bfloat16=BF16, float32=F32)


def _make(cls):
    """Construct an override and give it just enough fake ``state`` for the gate."""
    ov = cls()
    ov.state = SimpleNamespace(torch=_fake_torch())
    return ov


# --------------------------------------------------------------------------- #
# conv._pair helper                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        (1, (1, 1)),
        (2, (2, 2)),
        ((3, 4), (3, 4)),
        ([5, 6], (5, 6)),
        ((7,), (7, 7)),  # single-element sequence broadcasts
        ("same", None),  # padding strings decline
        ("valid", None),
        ((1, 2, 3), None),  # rank-3 is not a 2D conv hyperparam
    ],
)
def test_pair_normalisation(value, expected):
    assert _pair(value) == expected


# --------------------------------------------------------------------------- #
# LayerNorm gate                                                               #
# --------------------------------------------------------------------------- #
def test_layernorm_gate_accepts_f16_last_axis():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((8, 128), F16)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(128,), n=128)
    assert ok, reason


def test_layernorm_gate_accepts_bf16():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((4, 2, 64), BF16)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(64,), n=64)
    assert ok, reason


def test_layernorm_gate_rejects_f32():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((8, 128), F32)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(128,), n=128)
    assert not ok
    assert "f32" in reason


def test_layernorm_gate_rejects_multi_axis():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((8, 16, 32), F16)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(16, 32), n=32)
    assert not ok
    assert "normalized_shape rank" in reason


def test_layernorm_gate_rejects_not_last_dim():
    ov = _make(LayerNormOverride)
    # normalized_shape of 16 but last dim is 32 -> can't map to 2D [M,N] last-axis.
    x = _FakeTensor((8, 32), F16)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(16,), n=32)
    assert not ok
    assert "last dim" in reason


def test_layernorm_gate_rejects_weight_dtype_mismatch():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((8, 128), F16)
    w = _FakeTensor((128,), F32)
    ok, reason = ov._gate(x, weight=w, bias=None, ns=(128,), n=128)
    assert not ok
    assert "weight dtype" in reason


def test_layernorm_gate_rejects_non_cuda():
    ov = _make(LayerNormOverride)
    x = _FakeTensor((8, 128), F16, is_cuda=False)
    ok, reason = ov._gate(x, weight=None, bias=None, ns=(128,), n=128)
    assert not ok
    assert "cuda" in reason


# --------------------------------------------------------------------------- #
# Activation gates (SiLU / GELU)                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [SiluOverride, GeluOverride])
def test_activation_gate_accepts_f16_bf16(cls):
    ov = _make(cls)
    for dt in (F16, BF16):
        ok, reason = ov._gate(_FakeTensor((32, 64), dt))
        assert ok, reason


@pytest.mark.parametrize("cls", [SiluOverride, GeluOverride])
def test_activation_gate_rejects_f32(cls):
    ov = _make(cls)
    ok, reason = ov._gate(_FakeTensor((32, 64), F32))
    assert not ok
    assert "f32" in reason


@pytest.mark.parametrize("cls", [SiluOverride, GeluOverride])
def test_activation_gate_rejects_non_cuda(cls):
    ov = _make(cls)
    ok, reason = ov._gate(_FakeTensor((32, 64), F16, is_cuda=False))
    assert not ok
    assert "cuda" in reason


def test_gelu_mode_tanh_vs_erf():
    ov = _make(GeluOverride)
    # tanh approximation needs hipdnn.PointwiseMode; give it a fake enum sentinel.
    tanh_mode = object()
    ov.state.hipdnn = SimpleNamespace(
        PointwiseMode=SimpleNamespace(GELU_APPROX_TANH_FWD=tanh_mode)
    )
    assert ov._mode(approximate="tanh") is tanh_mode
    # exact erf gelu is unsupported -> a reason string (routes to native fallback).
    assert ov._mode(approximate="none") == "erf-gelu unsupported"


def test_silu_mode():
    ov = _make(SiluOverride)
    swish = object()
    ov.state.hipdnn = SimpleNamespace(PointwiseMode=SimpleNamespace(SWISH_FWD=swish))
    assert ov._mode() is swish


# --------------------------------------------------------------------------- #
# Conv2d gate                                                                  #
# --------------------------------------------------------------------------- #
def test_conv_gate_accepts_f16_groups1():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F16)
    w = _FakeTensor((32, 16, 3, 3), F16)
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert ok, reason


def test_conv_gate_rejects_f32():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F32)
    w = _FakeTensor((32, 16, 3, 3), F32)
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert not ok
    assert "f32" in reason


def test_conv_gate_rejects_grouped():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F16)
    w = _FakeTensor((32, 8, 3, 3), F16)
    ok, reason = ov._gate(
        x, w, groups=2, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert not ok
    assert "groups" in reason


def test_conv_gate_rejects_string_padding():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F16)
    w = _FakeTensor((32, 16, 3, 3), F16)
    # _pair('same') -> None, and the gate declines non-integer padding.
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=_pair("same"), dilation=(1, 1)
    )
    assert not ok
    assert "stride/padding/dilation" in reason


def test_conv_gate_rejects_weight_dtype_mismatch():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F16)
    w = _FakeTensor((32, 16, 3, 3), BF16)
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert not ok
    assert "weight dtype" in reason


def test_conv_gate_rejects_non_rank4():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((16, 32, 32), F16)
    w = _FakeTensor((32, 16, 3), F16)
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert not ok
    assert "rank-4" in reason


def test_conv_gate_rejects_non_cuda():
    ov = _make(Conv2dFpropOverride)
    x = _FakeTensor((1, 16, 32, 32), F16, is_cuda=False)
    w = _FakeTensor((32, 16, 3, 3), F16)
    ok, reason = ov._gate(
        x, w, groups=1, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )
    assert not ok
    assert "cuda" in reason
