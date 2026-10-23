# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.linear and the routed unquantized_gemm op
against torch.nn.functional.linear: dtypes, bias, odd M, strided operands,
torch.compile, HIP-graph capture, fallback counting and strict mode."""

import time

import pytest
import torch
import torch.nn.functional as F

from hipdnn_vllm import stats
from hipdnn_vllm.config import Config
from hipdnn_vllm.declines import DeclineError, Stage, parse_op_error

TOL = {
    torch.bfloat16: dict(atol=3e-2, rtol=2e-2),
    torch.float16: dict(atol=4e-3, rtol=4e-3),
    torch.float32: dict(atol=1e-4, rtol=1e-4),
}
DTYPES = list(TOL)
# (K, N): Qwen2.5-0.5B projections (q/k/v fused, o, gate_up fused, down) and odd sizes.
SHAPES = [(896, 1152), (896, 896), (896, 9728), (4864, 896), (100, 37), (33, 129)]
ODD_M = [1, 3, 7, 33, 257]


def _operands(m, k, n, dtype, bias, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + m * 31 + k + n)
    x = torch.randn(m, k, dtype=dtype, device="cuda", generator=g)
    w = torch.randn(n, k, dtype=dtype, device="cuda", generator=g) / k**0.5
    b = torch.randn(n, dtype=dtype, device="cuda", generator=g) if bias else None
    return x, w, b


def _check(y, x, w, b):
    ref = F.linear(x, w, b)
    assert y.shape == ref.shape and y.dtype == ref.dtype
    torch.testing.assert_close(y, ref, **TOL[x.dtype])


def _counts(op="linear"):
    rows = [r for r in stats.snapshot()["counters"] if r["op"] == op]
    out = {}
    for r in rows:
        key = (r["route"], r["reason"], r["phase"])
        out[key] = out.get(key, 0) + r["count"]
    return out


@pytest.fixture
def patched():
    """The vLLM patch installed with `linear` (strict off); removed after."""
    from hipdnn_vllm.ops import linear as lin

    lin.install_patch(Config(ops=frozenset({"linear"})))
    yield lin
    lin.uninstall_patch()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", ODD_M)
def test_matches_functional_linear(C, dtype, bias, k, n, m):
    x, w, b = _operands(m, k, n, dtype, bias)
    reason, engine = C.linear_probe(x, w, b)
    assert reason == "", reason
    assert "HIPBLASLT" in engine.upper(), engine
    _check(torch.ops.hipdnn_vllm.linear(x, w, b), x, w, b)


def test_hipblaslt_serves_all_dtypes(C):
    for dtype in DTYPES:
        x, w, b = _operands(5, 64, 48, dtype, True)
        reason, engine = C.linear_probe(x, w, b)
        print(f"linear {dtype}: engine={engine!r} reason={reason!r}")
        assert reason == "" and "HIPBLASLT" in engine.upper()


def test_concurrent_first_calls_build_one_plan(C):
    """Threads that miss the same key at once build it once: the Runtime
    re-checks the cache after taking its lock. linear_try releases the GIL."""
    import threading

    dtype = torch.bfloat16
    x, w, b = _operands(23, 640, 384, dtype, True, seed=5)  # a key no other test uses
    torch.cuda.synchronize()
    builds0 = C.stats_snapshot()["runtime"]["plan_builds"]
    start = threading.Barrier(4)
    results, errors = [], []

    def worker():
        try:
            start.wait()
            results.append(C.linear_try(x, w, b))
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert C.stats_snapshot()["runtime"]["plan_builds"] == builds0 + 1
    for y, reason, _ in results:
        assert reason == "", reason
        _check(y, x, w, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_padded_weight_row_stride(C, dtype):
    """vLLM pads weight rows on gfx11 (maybe_pad_weight_avoid_cache_aliasing):
    a [N, K] view with row stride > K must be served without a copy."""
    x, w, b = _operands(9, 896, 1152, dtype, True)
    pad = 64
    wp = F.pad(w, (0, pad))[..., :-pad]
    assert wp.stride() == (896 + pad, 1) and not wp.is_contiguous()
    assert C.linear_probe(x, wp, b)[0] == ""
    _check(torch.ops.hipdnn_vllm.linear(x, wp, b), x, w, b)


def test_strided_and_rank3_x(C):
    dtype = torch.bfloat16
    big = torch.randn(11, 1000, dtype=dtype, device="cuda")
    x = big[:, :896]  # row stride 1000
    _, w, b = _operands(11, 896, 640, dtype, True)
    _check(torch.ops.hipdnn_vllm.linear(x, w, b), x, w, b)
    x3 = torch.randn(2, 7, 896, dtype=dtype, device="cuda")
    y3 = torch.ops.hipdnn_vllm.linear(x3, w, None)
    assert y3.shape == (2, 7, 640)
    _check(y3, x3, w, None)
    xt = torch.randn(896, 13, dtype=dtype, device="cuda").t()  # column-major x
    _check(torch.ops.hipdnn_vllm.linear(xt, w, b), xt, w, b)


def test_decline_raises_with_reason(C):
    x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
    w = torch.randn(16, 32, dtype=torch.float64, device="cuda")
    assert C.linear_probe(x, w, None)[0] == "UNSUPPORTED_FEATURE:dtype"
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.linear(x, w, None)
    op, decline = parse_op_error(ei.value)
    assert op == "linear" and decline.stage is Stage.UNSUPPORTED_FEATURE


def test_one_plan_per_m(C):
    """Exact plans per M (no override shapes yet): a repeated M is a cache hit,
    a new M builds once. Prints the hipBLASLt plan build time."""
    dtype = torch.bfloat16
    _, w, b = _operands(1, 896, 4864, dtype, True)
    before = stats.snapshot()["runtime"]
    t0 = time.perf_counter()
    for m in (601, 602, 603):
        x = torch.randn(m, 896, dtype=dtype, device="cuda")
        torch.ops.hipdnn_vllm.linear(x, w, b)
    torch.cuda.synchronize()
    built = stats.snapshot()["runtime"]
    builds = built["plan_builds"] - before["plan_builds"]
    ns = built["plan_build_ns"] - before["plan_build_ns"]
    assert builds == 3
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(f"linear plan build: {ns / builds / 1e6:.2f} ms/plan ({wall_ms:.1f} ms wall)")
    for m in (601, 602, 603):
        x = torch.randn(m, 896, dtype=dtype, device="cuda")
        _check(torch.ops.hipdnn_vllm.linear(x, w, b), x, w, b)
    assert stats.snapshot()["runtime"]["plan_builds"] == built["plan_builds"]


# --- routed op (the vLLM patch target) ------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_routed_op_routes_to_hipdnn(C, patched, dtype, bias):
    stats.reset()
    x, w, b = _operands(37, 896, 1152, dtype, bias)
    _check(patched.hipdnn_unquantized_gemm(None, x, w, b), x, w, b)
    counts = _counts()
    assert counts.get(("hipdnn", "", "eager")) == 1, counts


def test_routed_op_fallback_is_counted_and_correct(C, patched):
    stats.reset()
    x = torch.randn(6, 40, dtype=torch.float64, device="cuda")
    w = torch.randn(24, 40, dtype=torch.float64, device="cuda")
    b = torch.randn(24, dtype=torch.float64, device="cuda")
    y = patched.hipdnn_unquantized_gemm(None, x, w, b)
    torch.testing.assert_close(y, F.linear(x, w, b))
    counts = _counts()
    assert counts == {("fallback", "UNSUPPORTED_FEATURE:dtype", "eager"): 1}, counts


def test_routed_op_strict_raises(C):
    from hipdnn_vllm.ops import linear as lin

    lin.install_patch(Config(ops=frozenset({"linear"}), strict=True))
    try:
        x = torch.randn(6, 40, dtype=torch.float64, device="cuda")
        w = torch.randn(24, 40, dtype=torch.float64, device="cuda")
        with pytest.raises(DeclineError) as ei:
            lin.hipdnn_unquantized_gemm(None, x, w, None)
        assert ei.value.decline.reason == "UNSUPPORTED_FEATURE:dtype"
        # A supported call still runs under strict.
        xb, wb, bb = _operands(5, 64, 32, torch.bfloat16, True)
        _check(lin.hipdnn_unquantized_gemm(None, xb, wb, bb), xb, wb, bb)
    finally:
        lin.uninstall_patch()
        lin._strict = False


def test_cpu_tensors_use_vllm_path(patched):
    x = torch.randn(3, 8)
    w = torch.randn(5, 8)
    torch.testing.assert_close(patched.hipdnn_unquantized_gemm(None, x, w, None), F.linear(x, w))


# --- torch.compile ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_torch_compile_fullgraph_dynamic_m(C, patched, dtype):
    torch._dynamo.reset()
    _, w, b = _operands(1, 896, 1152, dtype, True)

    def f(x):
        y = patched.hipdnn_unquantized_gemm(None, x, w, b)
        return torch.relu(y) + 1.0

    cf = torch.compile(f, fullgraph=True, dynamic=True)
    stats.reset()
    for m in (3, 17, 33):
        x = torch.randn(m, 896, dtype=dtype, device="cuda")
        torch.testing.assert_close(cf(x), torch.relu(F.linear(x, w, b)) + 1.0, **TOL[dtype])
    assert _counts().get(("hipdnn", "", "eager")) == 3, _counts()


def test_torch_compile_direct_op(C):
    torch._dynamo.reset()
    x, w, b = _operands(9, 128, 64, torch.bfloat16, True)
    cf = torch.compile(lambda x: torch.ops.hipdnn_vllm.linear(x, w, b) * 2, fullgraph=True)
    torch.testing.assert_close(cf(x), F.linear(x, w, b) * 2, **TOL[torch.bfloat16])


# --- HIP-graph capture ------------------------------------------------------------


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_capture_replay_uses_prebuilt_plan(C, patched, bias):
    """A size run eagerly first (vLLM's warm-up) is captured on hipDNN and
    replays with new inputs."""
    dtype = torch.bfloat16
    x, w, b = _operands(8, 896, 1152, dtype, bias)
    patched.hipdnn_unquantized_gemm(None, x, w, b)  # warm-up builds the plan
    torch.cuda.synchronize()
    stats.reset()
    builds = stats.snapshot()["runtime"]["plan_builds"]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = patched.hipdnn_unquantized_gemm(None, x, w, b)
    counts = _counts()
    assert counts == {("hipdnn", "", "capture"): 1}, counts
    assert stats.snapshot()["runtime"]["plan_builds"] == builds
    for seed in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        _check(y, x, w, b)
    assert _counts() == counts  # replays are not counted


def test_capture_miss_falls_back_without_building(C, patched):
    """A size never run eagerly declines CAPTURE_INVARIANT and captures the
    vLLM fallback instead of building a plan inside the capture."""
    dtype = torch.bfloat16
    x, w, b = _operands(19, 896, 640, dtype, True)
    stats.reset()
    builds = stats.snapshot()["runtime"]["plan_builds"]
    # The fallback (rocm_unquantized_gemm) may lazily initialise libraries on
    # first use; run it once eagerly so the capture itself stays legal.
    torch.ops.vllm.rocm_unquantized_gemm(x, w, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = patched.hipdnn_unquantized_gemm(None, x, w, b)
    assert _counts() == {("fallback", "CAPTURE_INVARIANT", "capture"): 1}, _counts()
    assert stats.snapshot()["runtime"]["plan_builds"] == builds
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    _check(y, x, w, b)


def test_capture_compiled_dynamic(C, patched):
    """torch.compile + capture, the way vLLM's piecewise graphs run it."""
    torch._dynamo.reset()
    dtype = torch.bfloat16
    _, w, b = _operands(1, 896, 1152, dtype, True)
    cf = torch.compile(lambda x: patched.hipdnn_unquantized_gemm(None, x, w, b), fullgraph=True)
    x = torch.randn(16, 896, dtype=dtype, device="cuda")
    cf(x)
    torch.cuda.synchronize()
    stats.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = cf(x)
    assert _counts() == {("hipdnn", "", "capture"): 1}, _counts()
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    _check(y, x, w, b)
