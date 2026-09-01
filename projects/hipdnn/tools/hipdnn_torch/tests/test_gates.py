# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-CPU, provider-free tests for native-routing + the translation helpers.

Under the pure-passthrough contract the overrides make **no** capability or
correctness pre-judgement: they build the graph from the tensor's actual dims +
strides at true rank and let hipDNN decide. So a ``_gate`` now has only the two
legitimate stops that keep the FFI safe -- a non-CUDA tensor (``execute()`` needs a
device pointer) and a dtype with no ``DataType`` enum (nothing to ``set_data_type``).
These tests pin exactly those two native-routing conditions, plus the *pure
translation helpers* (``_ntuple`` / ``_resolve_pads`` padding math, the
``[1,…,1,*ns]`` scale-view builder, the N-D matmul operand builder, and the
gelu/silu ``_mode`` map) -- with fake tensors and a fake ``state``, so they run in CI
on any box (no bootstrap, no GPU, no provider ``.so``).
"""

from types import SimpleNamespace

import pytest

from hipdnn_torch.activation import GeluOverride, SiluOverride
from hipdnn_torch.conv import Conv2dFpropOverride, _ntuple, _resolve_pads
from hipdnn_torch.layernorm import LayerNormOverride
from hipdnn_torch.linear import LinearOverride
from hipdnn_torch.rmsnorm import RmsNormOverride, _scale_view
from hipdnn_torch.sdpa import SdpaOverride


class _FakeDtype:
    """A stand-in for a ``torch.dtype`` -- a unique, hashable sentinel with a repr."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<dtype {self.name}>"


# F16/BF16/F32 are all graph-mappable (present in state.dtype_map); INT8 stands in
# for a dtype the graph builder cannot map, so the dtype stop declines it.
F16 = _FakeDtype("f16")
BF16 = _FakeDtype("bf16")
F32 = _FakeDtype("f32")
INT8 = _FakeDtype("int8")  # not in dtype_map -> "not graph-mappable"


def _contig_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class _FakeTensor:
    """Minimal tensor: just the attributes the gates + helpers read."""

    def __init__(self, shape, dtype=F16, is_cuda=True, stride=None):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._stride = tuple(stride) if stride is not None else _contig_strides(shape)

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def stride(self, i=None):
        return self._stride if i is None else self._stride[i]


def _fake_torch():
    return SimpleNamespace(float16=F16, bfloat16=BF16, float32=F32)


def _make(cls):
    """Construct an override with just enough fake ``state`` for the gate/helpers.
    ``dtype_map`` mirrors bootstrap's (f16/bf16/f32 are graph-mappable); only
    membership is checked by the gate."""
    ov = cls()
    ov.state = SimpleNamespace(
        torch=_fake_torch(),
        dtype_map={F16: "HALF", BF16: "BFLOAT16", F32: "FLOAT"},
    )
    return ov


# Each override's gate and the argument it inspects (all reduce to the input/query).
_GATED = [
    (LinearOverride, lambda ov, t: ov._gate(t)),
    (RmsNormOverride, lambda ov, t: ov._gate(t)),
    (LayerNormOverride, lambda ov, t: ov._gate(t)),
    (SdpaOverride, lambda ov, t: ov._gate(t)),
    (SiluOverride, lambda ov, t: ov._gate(t)),
    (GeluOverride, lambda ov, t: ov._gate(t)),
    (Conv2dFpropOverride, lambda ov, t: ov._gate(t, t)),
]


# --------------------------------------------------------------------------- #
# The two legitimate native-routing stops, for every override                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls, call", _GATED)
@pytest.mark.parametrize("dt", [F16, BF16, F32])
def test_gate_accepts_mappable_dtype_on_cuda(cls, call, dt):
    # f32 is graph-mappable now too: no dtype pre-filter beyond "has a DataType".
    ov = _make(cls)
    ok, reason = call(ov, _FakeTensor((8, 128, 4, 4), dt))
    assert ok, reason


@pytest.mark.parametrize("cls, call", _GATED)
def test_gate_routes_non_cuda_to_native(cls, call):
    ov = _make(cls)
    ok, reason = call(ov, _FakeTensor((8, 128, 4, 4), F16, is_cuda=False))
    assert not ok
    assert "cuda" in reason


@pytest.mark.parametrize("cls, call", _GATED)
def test_gate_routes_unmappable_dtype_to_native(cls, call):
    ov = _make(cls)
    ok, reason = call(ov, _FakeTensor((8, 128, 4, 4), INT8))
    assert not ok
    assert "not graph-mappable" in reason


# --------------------------------------------------------------------------- #
# conv._ntuple helper (rank-generic; n=2 for conv2d)                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        (1, (1, 1)),
        (2, (2, 2)),
        ((3, 4), (3, 4)),
        ([5, 6], (5, 6)),
        ((7,), (7, 7)),  # single-element sequence broadcasts
        ("same", None),  # padding strings are not an n-tuple
        ("valid", None),
        ((1, 2, 3), None),  # rank-3 is not a 2D conv hyperparam
    ],
)
def test_ntuple_normalisation(value, expected):
    assert _ntuple(value, 2) == expected


# --------------------------------------------------------------------------- #
# conv._resolve_pads translation ('valid'/'same'/numeric)                      #
# --------------------------------------------------------------------------- #
def test_resolve_pads_valid_is_zero():
    assert _resolve_pads("valid", (1, 1), (3, 3), 2) == ((0, 0), (0, 0))


def test_resolve_pads_same_odd_kernel_symmetric():
    # total = dil*(k-1) = 2; pre = 1, post = 1.
    assert _resolve_pads("same", (1, 1), (3, 3), 2) == ((1, 1), (1, 1))


def test_resolve_pads_same_even_kernel_trailing_heavy():
    # total = dil*(k-1) = 3; pre = 1, post = 2 (torch pads the trailing side more).
    assert _resolve_pads("same", (1, 1), (4, 4), 2) == ((1, 1), (2, 2))


def test_resolve_pads_same_dilated():
    # dil=2, k=3 -> total = 2*2 = 4; pre = 2, post = 2.
    assert _resolve_pads("same", (2, 2), (3, 3), 2) == ((2, 2), (2, 2))


def test_resolve_pads_numeric_is_symmetric():
    assert _resolve_pads((2, 3), (1, 1), (3, 3), 2) == ((2, 3), (2, 3))
    assert _resolve_pads(1, (1, 1), (3, 3), 2) == ((1, 1), (1, 1))


def test_resolve_pads_unknown_string_declines():
    assert _resolve_pads("wrap", (1, 1), (3, 3), 2) is None


# --------------------------------------------------------------------------- #
# rmsnorm._scale_view -- [1,...,1,*param.shape] rank-expanding view            #
# --------------------------------------------------------------------------- #
def test_scale_view_expands_leading_unit_dims():
    param = _FakeTensor((128,), F16)  # contiguous [128], stride (1,)
    dims, strides = _scale_view(param, 3)
    assert dims == [1, 1, 128]
    assert strides == [0, 0, 1]  # leading batch dims strided 0 (pure description)


def test_scale_view_multi_axis_normalized_shape():
    param = _FakeTensor((16, 32), F16)  # stride (32, 1)
    dims, strides = _scale_view(param, 4)
    assert dims == [1, 1, 16, 32]
    assert strides == [0, 0, 32, 1]


def test_scale_view_same_rank_is_identity_shape():
    param = _FakeTensor((8, 128), F16)
    dims, strides = _scale_view(param, 2)
    assert dims == [8, 128]
    assert strides == [128, 1]


# --------------------------------------------------------------------------- #
# linear._operands -- N-D matmul operands from actual dims/strides             #
# --------------------------------------------------------------------------- #
def test_operands_nd_input():
    ov = _make(LinearOverride)
    x = _FakeTensor((4, 8, 128), F16)  # [batch, M, K], stride (1024,128,1)
    w = _FakeTensor((256, 128), F16)  # [N, K], stride (128, 1)
    a_dims, a_str, b_dims, b_str, c_dims, rank1 = ov._operands(x, w, 256, 128)
    assert a_dims == [4, 8, 128] and a_str == [1024, 128, 1]
    # B: logical [1, K, N] with batch stride 0 and the transposed weight strides.
    assert b_dims == [1, 128, 256] and b_str == [0, 1, 128]
    assert c_dims == [4, 8, 256]
    assert rank1 is False


def test_operands_rank1_input_gets_unit_view():
    ov = _make(LinearOverride)
    x = _FakeTensor((128,), F16)  # [K]
    w = _FakeTensor((256, 128), F16)
    a_dims, a_str, b_dims, b_str, c_dims, rank1 = ov._operands(x, w, 256, 128)
    assert a_dims == [1, 128] and a_str == [0, 1]  # free [1,K] view
    assert b_dims == [128, 256] and b_str == [1, 128]
    assert c_dims == [1, 256]
    assert rank1 is True


# --------------------------------------------------------------------------- #
# Activation _mode maps (SiLU / GELU tanh vs erf)                             #
# --------------------------------------------------------------------------- #
def test_gelu_mode_tanh_vs_erf():
    ov = _make(GeluOverride)
    tanh_mode = object()
    erf_mode = object()
    ov.state.hipdnn = SimpleNamespace(
        PointwiseMode=SimpleNamespace(GELU_APPROX_TANH_FWD=tanh_mode, GELU_FWD=erf_mode)
    )
    assert ov._mode(approximate="tanh") is tanh_mode
    # exact erf gelu -> GELU_FWD (built into the graph; hipDNN decides support).
    assert ov._mode(approximate="none") is erf_mode


def test_silu_mode():
    ov = _make(SiluOverride)
    swish = object()
    ov.state.hipdnn = SimpleNamespace(PointwiseMode=SimpleNamespace(SWISH_FWD=swish))
    assert ov._mode() is swish
