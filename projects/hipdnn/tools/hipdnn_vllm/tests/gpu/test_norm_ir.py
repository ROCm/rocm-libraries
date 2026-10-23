# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The dynamic-row rms_norm / fused_add_rms_norm ops and the
``hipdnn`` vLLM IR providers, against vLLM's ``native`` and ``vllm_c`` impls,
eagerly, under torch.compile and under HIP-graph capture."""

import pytest
import torch

from hipdnn_vllm import stats

from .test_rms_norm import TOL

vllm_ir = pytest.importorskip("vllm.ir")

EPS = 1e-6
HIDDEN = [896, 4096]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture(scope="module")
def ir_ops(hipdnn_state):
    import vllm.kernels  # noqa: F401  (registers vllm_c, aiter)

    from hipdnn_vllm.ops import norm_ir  # noqa: F401  (registers hipdnn)

    return vllm_ir.ops


@pytest.fixture
def rows_state(C):
    """Fresh row-policy state; exact rows {1, 2, 4, 8, 24}."""
    C.norm_rows_reset()
    C.norm_set_exact_rows([1, 2, 4, 8, 24])
    yield C
    C.norm_rows_reset()


def native_rms(ir_ops, x, w, eps):
    return ir_ops.rms_norm.impls["native"].impl_fn(x, w, eps)


def native_fused(ir_ops, x, r, w, eps):
    return ir_ops.fused_add_rms_norm.impls["native"].impl_fn(x, r, w, eps)


def vllm_c_rms(ir_ops, x, w, eps):
    return ir_ops.rms_norm.impls["vllm_c"].impl_fn(x, w, eps)


def vllm_c_fused(ir_ops, x, r, w, eps):
    # The vllm_c impl is in place; give it copies.
    return ir_ops.fused_add_rms_norm.impls["vllm_c"].impl_fn(x.clone(), r.clone(), w, eps)


def counts(C, op):
    out = {}
    for r in C.stats_snapshot()["counters"]:
        if r["op"] == op:
            key = (r["route"], r["reason"], r["phase"])
            out[key] = out.get(key, 0) + r["count"]
    return out


def launches(C):
    return C.stats_snapshot()["runtime"]["execute_calls"]


# ---------------------------------------------------------------------------
# correctness


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", [1, 7, 24, 37, 256, 1000])
def test_rms_rows_matches_native_and_vllm_c(ir_ops, rows_state, dtype, hidden, rows):
    torch.manual_seed(rows * 7 + hidden)
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    y = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
    assert y.shape == x.shape and y.dtype == dtype and y.is_contiguous()
    torch.testing.assert_close(y, native_rms(ir_ops, x, w, EPS), **TOL[dtype])
    torch.testing.assert_close(y, vllm_c_rms(ir_ops, x, w, EPS), **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", [1, 7, 24, 37, 1000])
def test_fused_add_matches_native_and_vllm_c(ir_ops, rows_state, dtype, hidden, rows):
    torch.manual_seed(rows * 13 + hidden)
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    r = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    x0, r0 = x.clone(), r.clone()
    y, res = torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, True)
    # Functional: inputs untouched.
    assert torch.equal(x, x0) and torch.equal(r, r0)
    ny, nres = native_fused(ir_ops, x, r, w, EPS)
    cy, cres = vllm_c_fused(ir_ops, x, r, w, EPS)
    # The residual is the rounded sum in both references.
    torch.testing.assert_close(res, nres, **TOL[dtype])
    torch.testing.assert_close(res, cres, **TOL[dtype])
    torch.testing.assert_close(y, ny, **TOL[dtype])
    torch.testing.assert_close(y, cy, **TOL[dtype])
    assert rows_state.fused_add_route(x, r, w, EPS) in ("fused", "decomposed")


def test_fused_add_route_is_counted(ir_ops, rows_state):
    C = rows_state
    x = torch.randn(5, 896, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x)
    w = torch.randn(896, dtype=torch.bfloat16, device="cuda")
    stats.reset()
    torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, True)
    route = C.fused_add_route(x, r, w, EPS)
    got = counts(C, "fused_add_rms_norm")
    if route == "decomposed":
        assert got == {("hipdnn", "PARTIAL", "eager"): 1}, got
    else:
        assert route == "fused"
        # 5 rows = 4 + 1: one launch per chunk.
        assert got == {("hipdnn", "", "eager"): 2}, got
    print(f"fused_add_rms_norm route: {route}")


@pytest.mark.parametrize("layout", ["padded_rows", "per_head_3d"])
def test_strided_and_3d_inputs(ir_ops, rows_state, layout):
    """A q-like view of a fused projection (padded row stride) and a per-head
    [tokens, heads, head_dim] norm. Where no engine takes the layout, the probe
    declines it (so the IR provider is skipped), strict raises and the
    non-strict op still returns the native result."""
    C = rows_state
    dtype = torch.bfloat16
    if layout == "padded_rows":
        x = torch.randn(37, 3 * 896, dtype=dtype, device="cuda")[:, :896]
    else:
        x = torch.randn(37, 14, 64, dtype=dtype, device="cuda")
    w = torch.randn(x.shape[-1], dtype=dtype, device="cuda")
    want = native_rms(ir_ops, x, w, EPS)
    torch.testing.assert_close(torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False), want, **TOL[dtype])
    reason = C.rms_norm_probe(x[:2], w, EPS)
    with ir_ops.rms_norm.set_priority(["hipdnn", "vllm_c", "native"]):
        provider = ir_ops.rms_norm.dispatch(x, w, EPS).provider
    if reason:
        print(f"{layout}: declined {reason}")
        assert provider != "hipdnn"
        with pytest.raises(RuntimeError, match=r"decline \[rms_norm\]"):
            torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
    else:
        assert provider == "hipdnn"
        torch.testing.assert_close(torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True), want, **TOL[dtype])


def test_noncollapsible_rows_fall_back(ir_ops, rows_state):
    """[T, 7, 64] sliced from [T, 14, 64]: no single row stride. The op falls
    back (counted); the IR provider declines it up front."""
    C = rows_state
    dtype = torch.bfloat16
    x = torch.randn(9, 14, 64, dtype=dtype, device="cuda")[:, :7, :]
    w = torch.randn(64, dtype=dtype, device="cuda")
    stats.reset()
    y = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False)
    torch.testing.assert_close(y, native_rms(ir_ops, x, w, EPS), **TOL[dtype])
    assert counts(C, "rms_norm") == {("fallback", "UNSUPPORTED_FEATURE:noncollapsible_rows", "eager"): 1}
    with ir_ops.rms_norm.set_priority(["hipdnn", "vllm_c", "native"]):
        assert ir_ops.rms_norm.dispatch(x, w, EPS).provider != "hipdnn"


# ---------------------------------------------------------------------------
# row policy


def test_exact_rows_prebuilt_and_pow2_chunks(ir_ops, rows_state):
    C = rows_state
    x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")

    before = C.norm_prebuild_totals()
    torch.ops.hipdnn_vllm.rms_norm_rows(x[:24], w, EPS, True)  # first eager call pre-builds
    after = C.norm_prebuild_totals()
    assert after["signatures"] == before["signatures"] + 1
    assert after["plans"] - before["plans"] == 5  # 1, 2, 4, 8, 24
    print(f"prebuild 5 exact plans: {(after['ns'] - before['ns']) / 1e9:.2f} s")

    def n_launches(rows):
        n0 = launches(C)
        torch.ops.hipdnn_vllm.rms_norm_rows(x[:rows], w, EPS, True)
        return launches(C) - n0

    assert n_launches(24) == 1  # exact
    assert n_launches(8) == 1  # exact (and a power of two)
    assert n_launches(37) == 3  # 32 + 4 + 1
    assert n_launches(1000) == bin(1000).count("1")
    assert n_launches(512) == 1


def test_prebuild_cost_default_capture_sizes(ir_ops, C):
    """Warm-up cost of vLLM's default capture sizes (max 512: 51 sizes) for a
    signature no other test builds."""
    sizes = [1, 2, 4, *range(8, 256, 8), *range(256, 513, 16)]
    assert len(sizes) == 51
    C.norm_rows_reset()
    C.norm_set_exact_rows(sizes)
    try:
        x = torch.randn(3, 1536, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(1536, dtype=torch.bfloat16, device="cuda")
        builds0 = C.stats_snapshot()["runtime"]["plan_builds"]
        torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
        pre = C.norm_prebuild_totals()
        assert pre["plans"] == 51 and pre["declines"] == 0, pre
        builds = C.stats_snapshot()["runtime"]["plan_builds"] - builds0
        print(f"prebuild of 51 capture sizes: {pre['ns'] / 1e9:.2f} s ({builds} plan builds)")
    finally:
        C.norm_rows_reset()


def test_no_exact_rows_when_vllm_config_absent(ir_ops, C):
    C.norm_rows_reset()
    try:
        x = torch.randn(24, 896, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(896, dtype=torch.bfloat16, device="cuda")
        n0 = launches(C)
        torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
        assert launches(C) - n0 == 2  # 16 + 8: no capture sizes known
        assert C.norm_exact_rows() is None  # asked again for the next signature
    finally:
        C.norm_rows_reset()


def test_exact_rows_from_vllm_config(ir_ops, C):
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CUDAGraphMode

    from hipdnn_vllm.ops.norm_ir import exact_rows_for_process

    cfg = VllmConfig(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, cudagraph_capture_sizes=[1, 2, 4, 24, 48]
        )
    )
    with set_current_vllm_config(cfg):
        rows = exact_rows_for_process()
    cc = cfg.compilation_config
    if cc.cudagraph_mode == CUDAGraphMode.NONE:
        assert rows == []
    else:
        assert rows == sorted(set(cc.cudagraph_capture_sizes))
    print(f"cudagraph_mode={cc.cudagraph_mode} exact rows={rows}")


# ---------------------------------------------------------------------------
# fallback and strict


def test_decline_falls_back_and_is_counted(ir_ops, rows_state):
    C = rows_state
    x = torch.randn(9, 128, dtype=torch.float64, device="cuda")
    w = torch.randn(128, dtype=torch.float64, device="cuda")
    stats.reset()
    y = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False)
    torch.testing.assert_close(y, native_rms(ir_ops, x, w, EPS))
    assert counts(C, "rms_norm") == {("fallback", "UNSUPPORTED_FEATURE:dtype", "eager"): 1}
    with pytest.raises(RuntimeError, match=r"hipdnn_vllm decline \[rms_norm\]: UNSUPPORTED_FEATURE:dtype"):
        torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)

    r = torch.randn_like(x)
    y, res = torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, False)
    ny, nres = native_fused(ir_ops, x, r, w, EPS)
    torch.testing.assert_close(y, ny)
    torch.testing.assert_close(res, nres)
    with pytest.raises(RuntimeError, match=r"decline \[fused_add_rms_norm\]"):
        torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, True)


# ---------------------------------------------------------------------------
# vLLM IR dispatch


PRIORITY = ["hipdnn", "vllm_c", "native"]


def test_ir_dispatch_selects_hipdnn(ir_ops, rows_state):
    C = rows_state
    x = torch.randn(37, 896, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x)
    w = torch.randn(896, dtype=torch.bfloat16, device="cuda")
    with ir_ops.rms_norm.set_priority(PRIORITY), ir_ops.fused_add_rms_norm.set_priority(PRIORITY):
        assert ir_ops.rms_norm.dispatch(x, w, EPS).provider == "hipdnn"
        assert ir_ops.fused_add_rms_norm.dispatch(x, r, w, EPS).provider == "hipdnn"
        # Args hipdnn does not take go to the next provider.
        assert ir_ops.rms_norm.dispatch(x, None, EPS).provider == "vllm_c"
        assert ir_ops.rms_norm.dispatch(x, w, EPS, 64).provider == "native"
        stats.reset()
        y = ir_ops.rms_norm(x, w, EPS)
        y2, res = ir_ops.fused_add_rms_norm(x, r, w, EPS)
        y3, res3 = ir_ops.fused_add_rms_norm.maybe_inplace(x.clone(), r.clone(), w, EPS)
    torch.testing.assert_close(y, native_rms(ir_ops, x, w, EPS), **TOL[torch.bfloat16])
    ny, nres = native_fused(ir_ops, x, r, w, EPS)
    for a, b in ((y2, ny), (res, nres), (y3, ny), (res3, nres)):
        torch.testing.assert_close(a, b, **TOL[torch.bfloat16])
    assert counts(C, "rms_norm").get(("hipdnn", "", "eager"), 0) >= 1


def test_ir_supports_args_under_fake_mode(ir_ops, rows_state):
    """vLLM's lowering pass calls supports_args with FakeTensors (symbolic rows)."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext

    mode = FakeTensorMode(shape_env=ShapeEnv())
    real = torch.randn(64, 896, dtype=torch.bfloat16, device="cuda")
    w_real = torch.randn(896, dtype=torch.bfloat16, device="cuda")
    ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC])
    fx = mode.from_tensor(real, symbolic_context=ctx)
    fw = mode.from_tensor(w_real, static_shapes=True)
    assert isinstance(fx.shape[0], torch.SymInt)
    with ir_ops.rms_norm.set_priority(PRIORITY):
        with mode:
            assert ir_ops.rms_norm.dispatch(fx, fw, EPS).provider == "hipdnn"


def test_strict_supports_args_raises(ir_ops, rows_state, monkeypatch):
    from hipdnn_vllm.declines import DeclineError
    from hipdnn_vllm.ops import norm_ir

    monkeypatch.setattr(norm_ir, "_STRICT", True)
    x = torch.randn(3, 896, dtype=torch.bfloat16, device="cuda")
    with ir_ops.rms_norm.set_priority(PRIORITY):
        with pytest.raises(DeclineError, match="no_weight"):
            ir_ops.rms_norm.dispatch(x, None, EPS)


# ---------------------------------------------------------------------------
# torch.compile and HIP-graph capture


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_torch_compile_dynamic_rows(ir_ops, rows_state, dtype):
    hidden = 896
    w = torch.randn(hidden, dtype=dtype, device="cuda")

    def block(x, r):
        y, res = torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x * 2.0, r, w, EPS, True)
        return torch.ops.hipdnn_vllm.rms_norm_rows(y + 1.0, w, EPS, True), res

    def ref(x, r):
        y, res = native_fused(ir_ops, x * 2.0, r, w, EPS)
        return native_rms(ir_ops, y + 1.0, w, EPS), res

    compiled = torch.compile(block, fullgraph=True, dynamic=True)
    for rows in (3, 24, 37, 200):
        x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
        r = torch.randn_like(x)
        got, res = compiled(x, r)
        want, wres = ref(x, r)
        torch.testing.assert_close(res, wres, **TOL[dtype])
        torch.testing.assert_close(got, want, atol=TOL[dtype]["atol"] * 4, rtol=TOL[dtype]["rtol"] * 4)


def test_torch_compile_ir_op(ir_ops, rows_state):
    """The IR op (torch-wrapped) inside a compiled graph dispatches to hipdnn."""
    C = rows_state
    w = torch.randn(896, dtype=torch.bfloat16, device="cuda")

    def f(x):
        return ir_ops.rms_norm(x * 0.5, w, EPS)

    compiled = torch.compile(f, fullgraph=True, dynamic=True)
    with ir_ops.rms_norm.set_priority(PRIORITY):
        stats.reset()
        for rows in (5, 37):
            x = torch.randn(rows, 896, dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(compiled(x), native_rms(ir_ops, x * 0.5, w, EPS), **TOL[torch.bfloat16])
    assert counts(C, "rms_norm").get(("hipdnn", "", "eager"), 0) >= 2


@pytest.mark.parametrize("rows", [24, 37])
def test_capture_replay(ir_ops, rows_state, rows):
    """24 is an exact (pre-built) size: one launch in the graph. 37 is not: the
    capture replays the 32 + 4 + 1 row-chunk plans its eager warm-up built.
    Nothing is built inside the capture, and both replay correctly."""
    C = rows_state
    dtype = torch.bfloat16
    hidden = 896
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    r = torch.randn_like(x)
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    # Eager warm-up (as vLLM does before capturing): pre-builds the exact rows
    # and, for 37, the chunk plans.
    torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
    torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, True)
    torch.cuda.synchronize()
    route = C.fused_add_route(x, r, w, EPS)
    assert route in ("fused", "decomposed"), route

    stats.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, True)
        fy, fres = torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, r, w, EPS, True)
    snap = C.stats_snapshot()
    assert snap["runtime"]["plan_builds"] == 0, snap["runtime"]
    # Counters count launches: one per exact call, one per chunk otherwise.
    per_call = 1 if rows == 24 else bin(rows).count("1")
    assert per_call == (1 if rows == 24 else 3)
    # Every launch served by hipDNN (strict would have raised on a decline).
    if route == "fused":
        assert counts(C, "rms_norm") == {("hipdnn", "", "capture"): per_call}, counts(C, "rms_norm")
        assert counts(C, "fused_add_rms_norm") == {("hipdnn", "", "capture"): per_call}
    else:
        # The decomposed path runs the rms_norm launches a second time on the sum.
        assert counts(C, "rms_norm") == {("hipdnn", "", "capture"): 2 * per_call}, counts(C, "rms_norm")
        assert counts(C, "fused_add_rms_norm") == {("hipdnn", "PARTIAL", "capture"): 1}

    for step in range(4):
        x.copy_(torch.randn_like(x) * (step + 1))
        r.copy_(torch.randn_like(r))
        w.copy_(torch.randn_like(w))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, native_rms(ir_ops, x, w, EPS), **TOL[dtype])
        ny, nres = native_fused(ir_ops, x, r, w, EPS)
        torch.testing.assert_close(fres, nres, **TOL[dtype])
        torch.testing.assert_close(fy, ny, **TOL[dtype])
