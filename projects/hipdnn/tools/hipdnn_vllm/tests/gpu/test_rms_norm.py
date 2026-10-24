# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.rms_norm against a float32 torch reference."""

import pytest
import torch

from hipdnn_vllm import stats
from hipdnn_vllm.declines import Stage, parse_op_error, parse_reason

TOL = {
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=4e-3, rtol=4e-3),
    torch.float32: dict(atol=2e-5, rtol=2e-5),
}
DTYPES = list(TOL)


def ref_rms_norm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def _run(x, w, eps):
    return torch.ops.hipdnn_vllm.rms_norm(x, w, eps)


# Every (rows, hidden, dtype) is a separate plan build, and a JIT engine build
# is slow, so the shape list samples rows x hidden instead of crossing them.
SHAPES = [(1, 64), (256, 128), (7, 2048), (1, 4096), (256, 4096), (7, 5120), (1, 8192)]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("rows,hidden", SHAPES)
def test_matches_reference(C, dtype, hidden, rows):
    torch.manual_seed(hidden * 31 + rows)
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    eps = 1e-6
    assert C.rms_norm_probe(x, w, eps) == ""
    y = _run(x, w, eps)
    torch.testing.assert_close(y, ref_rms_norm(x, w, eps), **TOL[dtype])
    assert y.is_contiguous() and y.dtype == dtype and y.shape == x.shape


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("hidden", [1, 33, 1001, 4097])
def test_odd_hidden_sizes(C, dtype, hidden):
    """Odd sizes must be served correctly. A shape the installed engines
    cannot serve must decline with a taxonomy reason, never return garbage."""
    torch.manual_seed(hidden)
    x = torch.randn(5, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    reason = C.rms_norm_probe(x, w, 1e-5)
    if reason:
        parse_reason(reason)
        with pytest.raises(RuntimeError) as ei:
            _run(x, w, 1e-5)
        assert parse_op_error(ei.value)[1].reason == reason
        pytest.xfail(f"hipDNN declined hidden={hidden} {dtype}: {reason}")
    torch.testing.assert_close(_run(x, w, 1e-5), ref_rms_norm(x, w, 1e-5), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_fp32_weight_with_half_input(C, dtype):
    x = torch.randn(9, 4096, dtype=dtype, device="cuda")
    w = torch.randn(4096, dtype=torch.float32, device="cuda")
    reason = C.rms_norm_probe(x, w, 1e-6)
    if reason:
        parse_reason(reason)
        pytest.xfail(f"mixed-dtype weight declined: {reason}")
    torch.testing.assert_close(_run(x, w, 1e-6), ref_rms_norm(x, w, 1e-6), **TOL[dtype])


def test_rank3_rank5_and_rank1_inputs(C):
    """Leading dims collapse into one row dim, so any rank is served."""
    w = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    for shape in [(2, 5, 1024), (2, 1, 3, 1, 1024)]:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        y = _run(x, w, 1e-6)
        assert y.shape == x.shape
        torch.testing.assert_close(y, ref_rms_norm(x, w, 1e-6), **TOL[torch.bfloat16])
    x1 = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    y1 = _run(x1, w, 1e-6)
    assert y1.shape == x1.shape
    torch.testing.assert_close(y1, ref_rms_norm(x1, w, 1e-6), **TOL[torch.bfloat16])


def test_noncollapsible_rows_decline(C):
    """Leading dims that no single row stride describes decline before hipDNN."""
    base = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    x = base[:, :4, :]
    w = torch.randn(64, dtype=torch.float16, device="cuda")
    assert C.rms_norm_probe(x, w, 1e-6) == "UNSUPPORTED_FEATURE:noncollapsible_rows"
    with pytest.raises(RuntimeError) as ei:
        _run(x, w, 1e-6)
    assert parse_op_error(ei.value)[1].reason == "UNSUPPORTED_FEATURE:noncollapsible_rows"


def test_strided_rows_decline(C):
    """Elements within a row strided (x[:, ::2]) decline before hipDNN."""
    stats.reset()
    base = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
    x = base[:, ::2]
    w = torch.randn(1024, dtype=torch.float16, device="cuda")
    assert C.rms_norm_probe(x, w, 1e-6) == "UNSUPPORTED_FEATURE:strided_rows"
    with pytest.raises(RuntimeError) as ei:
        _run(x, w, 1e-6)
    op, d = parse_op_error(ei.value)
    assert op == "rms_norm" and d.reason == "UNSUPPORTED_FEATURE:strided_rows"
    rows = C.stats_snapshot()["counters"]
    assert any(
        r["op"] == "rms_norm" and r["route"] == "fallback" and r["reason"] == "UNSUPPORTED_FEATURE:strided_rows"
        for r in rows
    )


def test_padded_row_stride(C):
    """Row stride > hidden (a view of a fused projection) is passed to hipDNN
    as-is: served correctly, or declined by hipDNN with a taxonomy reason."""
    big = torch.randn(6, 3072, dtype=torch.bfloat16, device="cuda")
    x = big[:, 1024:2048]
    assert x.stride() == (3072, 1)
    w = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    reason = C.rms_norm_probe(x, w, 1e-6)
    if reason:
        assert parse_reason(reason).stage in {Stage.NO_BUILD, Stage.NO_ENGINE, Stage.CHECK_SUPPORT, Stage.BUILD_PLANS}
        pytest.xfail(f"padded row stride declined by hipDNN: {reason}")
    torch.testing.assert_close(_run(x, w, 1e-6), ref_rms_norm(x, w, 1e-6), **TOL[torch.bfloat16])


def test_unsupported_dtype_declines(C):
    x = torch.randn(2, 64, dtype=torch.float64, device="cuda")
    w = torch.randn(64, dtype=torch.float64, device="cuda")
    assert C.rms_norm_probe(x, w, 1e-6) == "UNSUPPORTED_FEATURE:dtype"
    with pytest.raises(RuntimeError) as ei:
        _run(x, w, 1e-6)
    assert parse_op_error(ei.value)[1].reason == "UNSUPPORTED_FEATURE:dtype"


def test_bad_weight_shape_is_an_error_not_a_decline(C):
    x = torch.randn(2, 64, dtype=torch.float16, device="cuda")
    w = torch.randn(32, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError) as ei:
        _run(x, w, 1e-6)
    assert parse_op_error(ei.value) is None


def test_plan_cache_reuse_and_eps_in_key(C):
    x = torch.randn(3, 768, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(768, dtype=torch.bfloat16, device="cuda")
    _run(x, w, 1e-6)
    before = C.stats_snapshot()["runtime"]
    for _ in range(5):
        _run(torch.randn_like(x), w, 1e-6)
    after = C.stats_snapshot()["runtime"]
    assert after["plan_builds"] == before["plan_builds"]
    assert after["plan_cache_hits"] >= before["plan_cache_hits"] + 5
    y = _run(x, w, 1e-2)  # different eps -> different graph
    assert C.stats_snapshot()["runtime"]["plan_builds"] == after["plan_builds"] + 1
    torch.testing.assert_close(y, ref_rms_norm(x, w, 1e-2), **TOL[torch.bfloat16])


def test_lru_bound(C):
    C.set_plan_cache_size(2)
    try:
        w = torch.randn(256, dtype=torch.float16, device="cuda")
        for rows in (1, 2, 3, 4):
            _run(torch.randn(rows, 256, dtype=torch.float16, device="cuda"), w, 1e-6)
        rt = C.stats_snapshot()["runtime"]
        assert rt["plan_cache_size"] <= 2 and rt["plan_cache_evictions"] >= 2
    finally:
        C.set_plan_cache_size(4096)


def test_fake_impl_registered(hipdnn_state):
    x = torch.empty(3, 64, dtype=torch.bfloat16, device="meta")
    w = torch.empty(64, dtype=torch.bfloat16, device="meta")
    y = torch.ops.hipdnn_vllm.rms_norm(x, w, 1e-6)
    assert y.shape == (3, 64) and y.device.type == "meta"


def test_torch_compile_fullgraph(hipdnn_state):
    @torch.compile(fullgraph=True, backend="eager")
    def f(x, w):
        return torch.ops.hipdnn_vllm.rms_norm(x * 2, w, 1e-6) + 1

    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(f(x, w), ref_rms_norm(x * 2, w, 1e-6) + 1, **TOL[torch.bfloat16])
