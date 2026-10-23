# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.layer_norm against torch's layer_norm."""

import pytest
import torch
import torch.nn.functional as F

from hipdnn_vllm import stats
from hipdnn_vllm.declines import parse_op_error, parse_reason

TOL = {
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=4e-3, rtol=4e-3),
    torch.float32: dict(atol=2e-5, rtol=2e-5),
}
DTYPES = list(TOL)


def ref(x, w, b, eps):
    """vLLM LayerNorm.forward_native, evaluated in float32."""
    y = F.layer_norm(x.float(), (x.shape[-1],), None if w is None else w.float(),
                     None if b is None else b.float(), eps)
    return y.to(x.dtype)


def ln(x, w, b, eps, strict=True):
    return torch.ops.hipdnn_vllm.layer_norm(x, w, b, eps, strict)


def _counts(op="layer_norm"):
    out = {}
    for r in stats.snapshot()["counters"]:
        if r["op"] == op:
            out[(r["route"], r["reason"])] = out.get((r["route"], r["reason"]), 0) + r["count"]
    return out


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("hidden", [768, 1152, 1280])
@pytest.mark.parametrize("rows", [1, 7, 256, 729])
def test_matches_reference(C, dtype, bias, hidden, rows):
    torch.manual_seed(hidden + rows)
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    b = torch.randn(hidden, dtype=dtype, device="cuda") if bias else None
    assert C.layer_norm_probe(x, w, b, 1e-6) == ""
    stats.reset()
    y = ln(x, w, b, 1e-6)
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[dtype])
    assert y.shape == x.shape and y.dtype == dtype and y.is_contiguous()
    # Strict mode did not raise, and the probe built the exact plan: one launch.
    assert _counts() == {("hipdnn", ""): 1}


@pytest.mark.parametrize("rows", [3, 100, 729])
def test_rows_without_exact_plan_use_power_of_two_blocks(C, rows):
    x = torch.randn(rows, 1152, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1152, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1152, dtype=torch.bfloat16, device="cuda")
    C.plan_cache_clear()
    stats.reset()
    y = ln(x, w, b, 1e-6)
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[torch.bfloat16])
    # One launch per set bit of the row count, all on hipDNN.
    assert _counts() == {("hipdnn", ""): bin(rows).count("1")}


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_no_affine(C, dtype):
    x = torch.randn(33, 1024, dtype=dtype, device="cuda")
    torch.testing.assert_close(ln(x, None, None, 1e-5), ref(x, None, None, 1e-5), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_fp32_params_with_half_input(C, dtype):
    """LayerNorm(dtype=float32) on a half input: reduction in fp32, output in x.dtype."""
    x = torch.randn(17, 1152, dtype=dtype, device="cuda")
    w = torch.randn(1152, dtype=torch.float32, device="cuda")
    b = torch.randn(1152, dtype=torch.float32, device="cuda")
    reason = C.layer_norm_probe(x, w, b, 1e-6)
    y = ln(x, w, b, 1e-6, strict=False)
    assert y.dtype == dtype
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[dtype])
    if reason:
        parse_reason(reason)
        pytest.xfail(f"mixed-dtype affine declined (fallback verified): {reason}")


def test_rank3_and_strided_rows(C):
    x = torch.randn(2, 5, 1024, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    y = ln(x, w, b, 1e-6)
    assert y.shape == x.shape
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[torch.bfloat16])
    # A row-strided view (e.g. a slice of a fused projection) is described
    # without a copy. Engines may decline the padded row stride; the op must
    # then fall back (counted) and still be correct.
    big = torch.randn(9, 3072, dtype=torch.bfloat16, device="cuda")
    xs = big[:, 1024:2048]
    reason = C.layer_norm_probe(xs, w, b, 1e-6)
    stats.reset()
    torch.testing.assert_close(ln(xs, w, b, 1e-6, strict=False), ref(xs, w, b, 1e-6), **TOL[torch.bfloat16])
    if reason:
        parse_reason(reason)
        assert _counts() == {("fallback", reason): 1}
        pytest.xfail(f"row-strided input declined (fallback verified): {reason}")
    assert _counts() == {("hipdnn", ""): 1}


def test_sliced_token_rows(C):
    """hidden[:, :1] (a CLS-token slice, [B, 1, N]) collapses to rows with no
    copy; a single row is described packed and always runs on hipDNN."""
    w = torch.randn(1152, dtype=torch.float16, device="cuda")
    b = torch.randn(1152, dtype=torch.float16, device="cuda")
    h = torch.randn(1, 7, 1152, dtype=torch.float16, device="cuda")
    one = h[:, :1]
    stats.reset()
    torch.testing.assert_close(ln(one, w, b, 1e-6), ref(one, w, b, 1e-6), **TOL[torch.float16])
    assert _counts() == {("hipdnn", ""): 1}
    many = torch.randn(3, 7, 1152, dtype=torch.float16, device="cuda")[:, :1]
    stats.reset()
    y = ln(many, w, b, 1e-6, strict=False)
    torch.testing.assert_close(y, ref(many, w, b, 1e-6), **TOL[torch.float16])
    assert y.shape == many.shape
    routes = _counts()
    assert sum(routes.values()) == 1 and all(route in ("hipdnn", "fallback") for route, _ in routes), routes


def test_transposed_patch_embed_layout(C):
    """The first ViT LayerNorm sees conv(x).flatten(2).transpose(1, 2) + pos,
    a [B, S, N] tensor whose hidden dim has stride S. It is packed (as ATen
    does) and served by hipDNN."""
    w = torch.randn(1152, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1152, dtype=torch.bfloat16, device="cuda")
    feat = torch.randn(3, 1152, 27, 27, dtype=torch.bfloat16, device="cuda")
    pos = torch.randn(1, 729, 1152, dtype=torch.bfloat16, device="cuda")
    x = feat.flatten(2).transpose(1, 2) + pos
    assert x.stride(-1) != 1
    assert C.layer_norm_probe(x, w, b, 1e-6) == ""
    stats.reset()
    y = ln(x, w, b, 1e-6)
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[torch.bfloat16])
    assert y.shape == x.shape
    assert set(_counts()) == {("hipdnn", "")}


def test_decline_falls_back_and_strict_raises(C):
    x = torch.randn(4, 64, dtype=torch.float64, device="cuda")
    w = torch.randn(64, dtype=torch.float64, device="cuda")
    stats.reset()
    y = ln(x, w, None, 1e-5, strict=False)
    torch.testing.assert_close(y, F.layer_norm(x, (64,), w, None, 1e-5))
    assert _counts() == {("fallback", "UNSUPPORTED_FEATURE:dtype"): 1}
    with pytest.raises(RuntimeError) as ei:
        ln(x, w, None, 1e-5, strict=True)
    op, decline = parse_op_error(ei.value)
    assert op == "layer_norm" and decline.reason == "UNSUPPORTED_FEATURE:dtype"


def test_torch_compile_fullgraph(C):
    w = torch.randn(768, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(768, dtype=torch.bfloat16, device="cuda")

    @torch.compile(fullgraph=True, dynamic=True)
    def f(x):
        return torch.ops.hipdnn_vllm.layer_norm(x * 2, w, b, 1e-6, True) + 1

    for rows in (5, 64, 100):
        x = torch.randn(rows, 768, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(f(x), ref(x * 2, w, b, 1e-6) + 1, **TOL[torch.bfloat16])


@pytest.mark.parametrize("prebuilt", [True, False], ids=["exact", "blocks"])
def test_hip_graph_capture(C, prebuilt):
    """Capture records the same launches as eager warm-up: one exact plan when
    it was pre-built with the probe, else the power-of-two blocks."""
    w = torch.randn(1152, dtype=torch.float16, device="cuda")
    b = torch.randn(1152, dtype=torch.float16, device="cuda")
    x = torch.randn(729, 1152, dtype=torch.float16, device="cuda")
    C.plan_cache_clear()
    if prebuilt:
        assert C.layer_norm_probe(x, w, b, 1e-6) == ""
    ln(x, w, b, 1e-6)  # warm-up outside capture builds any block plans
    torch.cuda.synchronize()
    stats.reset()
    builds = stats.snapshot()["runtime"]["plan_builds"]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = ln(x, w, b, 1e-6)
    snap = stats.snapshot()
    rows = [r for r in snap["counters"] if r["op"] == "layer_norm"]
    launches = 1 if prebuilt else bin(729).count("1")
    assert rows == [
        {"op": "layer_norm", "route": "hipdnn", "reason": "", "phase": "capture", "count": launches}
    ]
    assert snap["runtime"]["plan_builds"] == builds  # nothing built under capture
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref(x, w, b, 1e-6), **TOL[torch.float16])
