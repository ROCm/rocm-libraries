# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.conv_fwd against torch's conv2d/conv3d, on vision
patch-embedding shapes (kernel == stride == patch)."""

import pytest
import torch
import torch.nn.functional as F

from hipdnn_vllm import stats
from hipdnn_vllm.declines import parse_op_error

# conv is opt-in: the MIOpen provider that serves these graphs is
# loaded only in a process started with conv in HIPDNN_VLLM_OPS.
pytestmark = pytest.mark.conv

TOL = {
    torch.bfloat16: dict(atol=5e-2, rtol=2e-2),
    torch.float16: dict(atol=1e-2, rtol=5e-3),
    torch.float32: dict(atol=2e-3, rtol=1e-3),
}
DTYPES = list(TOL)

# (in_ch, out_ch, kernel, input spatial, batch)
CONV2D = [
    pytest.param(3, 768, (16, 16), (224, 224), 1, id="siglip-b16-224"),
    pytest.param(3, 1152, (14, 14), (384, 384), 2, id="siglip-so400m-384"),
    pytest.param(3, 1024, (14, 14), (336, 336), 1, id="clip-l14-336"),
]
CONV3D = [
    # Qwen2-VL / Qwen2.5-VL patch embed: per-patch input [N, 3, 2, 14, 14].
    pytest.param(3, 1280, (2, 14, 14), (2, 14, 14), 64, id="qwen2.5-vl-patches"),
    pytest.param(3, 1152, (2, 16, 16), (4, 64, 64), 1, id="video-t4-64x64"),
]


def conv(x, w, b, stride, padding=None, dilation=None, groups=1, strict=True):
    n = x.dim() - 2
    padding = padding or [0] * n
    dilation = dilation or [1] * n
    return torch.ops.hipdnn_vllm.conv_fwd(x, w, b, list(stride), padding, dilation, groups, strict)


def ref(x, w, b, stride, padding=0, dilation=1, groups=1):
    fn = F.conv2d if x.dim() == 4 else F.conv3d
    y = fn(x.float(), w.float(), None if b is None else b.float(), stride, padding, dilation, groups)
    return y.to(x.dtype)


def _counts(op="conv_fwd"):
    out = {}
    for r in stats.snapshot()["counters"]:
        if r["op"] == op:
            key = (r["route"], r["reason"])
            out[key] = out.get(key, 0) + r["count"]
    return out


def assert_hipdnn_route(bias):
    """One hipDNN call: fully fused, or (bias only) hipDNN conv + ATen bias add."""
    got = _counts()
    if got == {("hipdnn", ""): 1}:
        return "fused" if bias else "plain"
    assert bias and got == {("hipdnn", "PARTIAL"): 1}, got
    assert _counts("conv_fprop") == {("hipdnn", ""): 1}
    return "partial"


def _check(C, dtype, cin, cout, k, sp, n, bias):
    torch.manual_seed(cout + n)
    x = torch.randn(n, cin, *sp, dtype=dtype, device="cuda")
    w = torch.randn(cout, cin, *k, dtype=dtype, device="cuda") * 0.05
    b = torch.randn(cout, dtype=dtype, device="cuda") if bias else None
    dims = len(k)
    assert C.conv_fwd_probe(x, w, b, list(k), [0] * dims, [1] * dims, 1) == ""
    stats.reset()
    y = conv(x, w, b, k)
    torch.testing.assert_close(y, ref(x, w, b, k), **TOL[dtype])
    assert y.is_contiguous() and y.dtype == dtype
    path = assert_hipdnn_route(bias)
    print(f"conv path: {path}")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("cin,cout,k,sp,n", CONV2D)
def test_conv2d_patch_embed(C, dtype, bias, cin, cout, k, sp, n):
    _check(C, dtype, cin, cout, k, sp, n, bias)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("cin,cout,k,sp,n", CONV3D)
def test_conv3d_patch_embed(C, dtype, bias, cin, cout, k, sp, n):
    _check(C, dtype, cin, cout, k, sp, n, bias)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_conv2d_padded_strided(C, dtype):
    """A non-patch conv (padding, stride != kernel), e.g. a downsampling stem."""
    x = torch.randn(2, 64, 56, 56, dtype=dtype, device="cuda")
    w = torch.randn(128, 64, 3, 3, dtype=dtype, device="cuda") * 0.05
    b = torch.randn(128, dtype=dtype, device="cuda")
    y = conv(x, w, b, [2, 2], [1, 1], strict=False)
    torch.testing.assert_close(y, ref(x, w, b, 2, 1), **TOL[dtype])


def test_decline_falls_back_and_strict_raises(C):
    x = torch.randn(1, 3, 32, 32, dtype=torch.float64, device="cuda")
    w = torch.randn(16, 3, 8, 8, dtype=torch.float64, device="cuda")
    stats.reset()
    y = conv(x, w, None, [8, 8], strict=False)
    torch.testing.assert_close(y, F.conv2d(x, w, None, 8))
    assert _counts() == {("fallback", "UNSUPPORTED_FEATURE:dtype"): 1}
    with pytest.raises(RuntimeError) as ei:
        conv(x, w, None, [8, 8], strict=True)
    op, decline = parse_op_error(ei.value)
    assert op == "conv_fwd" and decline.reason == "UNSUPPORTED_FEATURE:dtype"


def test_torch_compile_fullgraph(C):
    w = torch.randn(768, 3, 16, 16, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(768, dtype=torch.bfloat16, device="cuda")

    @torch.compile(fullgraph=True)
    def f(x):
        y = torch.ops.hipdnn_vllm.conv_fwd(x, w, b, [16, 16], [0, 0], [1, 1], 1, True)
        return y.flatten(2).transpose(1, 2)

    x = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device="cuda")
    out = f(x)
    assert out.shape == (1, 196, 768)
    torch.testing.assert_close(out, ref(x, w, b, 16).flatten(2).transpose(1, 2), **TOL[torch.bfloat16])


def test_hip_graph_capture(C):
    w = torch.randn(768, 3, 16, 16, dtype=torch.float16, device="cuda") * 0.05
    b = torch.randn(768, dtype=torch.float16, device="cuda")
    x = torch.randn(1, 3, 224, 224, dtype=torch.float16, device="cuda")
    conv(x, w, b, [16, 16])  # build the plan eagerly
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = conv(x, w, b, [16, 16])
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref(x, w, b, 16), **TOL[torch.float16])


@pytest.mark.parametrize(
    "stride,padding,dilation,match",
    [
        ([0, 16], [0, 0], [1, 1], "stride and dilation must be >= 1"),
        ([16, 16], [-1, 0], [1, 1], "padding >= 0"),
        ([16, 16], [0, 0], [0, 1], "stride and dilation must be >= 1"),
        ([1, 1], [0, 0], [20, 1], "larger than the padded input"),
    ],
)
def test_invalid_conv_arguments_are_rejected(C, stride, padding, dilation, match):
    """Bad stride/padding/dilation are argument errors, not a crash in the
    output-shape arithmetic (stride 0 divided by zero)."""
    w = torch.randn(8, 3, 16, 16, dtype=torch.float16, device="cuda")
    x = torch.randn(1, 3, 224, 224, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match=match):
        conv(x, w, None, stride, padding, dilation)
