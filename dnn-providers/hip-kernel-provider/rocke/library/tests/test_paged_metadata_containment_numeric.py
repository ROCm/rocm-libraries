# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hardware containment and recovery after ordered live metadata mutation."""

import math

import pytest

from tests._decode_numeric_support import gpu, decode_launch_session
from tests._decode_reference import _reference, _feature_reference


@pytest.mark.parametrize(
    "page,route,dim,ratio",
    [(page, "explicit", 64, 4) for page in (1, 16, 32, 64)]
    + [
        (1, route, dim, ratio)
        for route in ("auto", "3d")
        for dim, ratio in ((64, 1), (128, 4), (256, 16))
    ],
)
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
def test_live_metadata_failure_and_recovery(
    gpu, monkeypatch, page, route, dim, ratio, dtype_name, mode
):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("bounded decode targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    capacity, batch, hk = 65, 2, 2
    hq = hk * ratio
    req = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hk,
        seqlen_q=1,
        seqlen_k=capacity,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype_name,
        kv_block_size=page,
        algorithm="paged_decode_t32",
        target_ctas=8,
    )
    selected = dispatch_attention(req) if route == "explicit" else None
    problem = _problem(req)
    torch.manual_seed(112)
    qh = torch.randn(batch, hq, dim, dtype=dtype)
    pages = (capacity + page - 1) // page
    count = batch * pages
    kh = torch.randn(count, page, hk, dim, dtype=dtype)
    vh = torch.randn_like(kh)
    # Padding beyond declared K/V views provides a controlled fence for ID=count.
    kb = torch.full((count + 2, page, hk, dim), float("nan"), dtype=dtype).cuda()
    vb = torch.full((count + 2, page, hk, dim), float("nan"), dtype=dtype).cuda()
    kb[:count].copy_(kh)
    vb[:count].copy_(vh)
    k, v = kb[:count], vb[:count]
    qback = torch.full((batch + 2, hq, dim), 19.0, dtype=dtype).cuda()
    oback = torch.full((batch + 2, hq, dim), 23.0, dtype=dtype).cuda()
    q, out = qback[1:-1], oback[1:-1]
    q.copy_(qh)
    cu = torch.arange(batch + 1, dtype=torch.int32).cuda()
    used = torch.full((batch,), capacity, dtype=torch.int32).cuda()
    valid_table = torch.arange(count, dtype=torch.int32).reshape(batch, pages)
    table = valid_table.cuda()
    args = dict(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=used,
        block_table=table,
        softmax_scale=1 / math.sqrt(dim),
        softcap=0.0,
        backend="3d" if route == "explicit" else route,
        tuning_spec=selected.spec if selected is not None else None,
        use_graph=mode == "graph",
    )
    with decode_launch_session(torch, au, args, mode) as launch:

        logical_k = kh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
        logical_v = vh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
        reference = _reference(torch, qh, logical_k, logical_v, [capacity] * batch, 0)
        # Every failure is followed by a valid replay on the same buffers/graph.
        for kind, value in (
            ("page", -1),
            ("page", count),
            ("page", -2147483648),
            ("page", 2147483647),
            ("length", -1),
            ("length", capacity + 1),
            ("length", 2147483647),
            ("cu", -2147483648),
            ("cu", 2147483647),
            ("cu", 1),
        ):
            table.copy_(valid_table)
            used.copy_(torch.tensor([capacity, capacity], dtype=torch.int32))
            cu.copy_(torch.arange(batch + 1, dtype=torch.int32))
            if kind == "page":
                # Last page exercises final prefetch and later segments too.
                bad = valid_table.clone()
                bad[0, -1] = value
                table.copy_(bad)
            elif kind == "length":
                used.copy_(torch.tensor([value, capacity], dtype=torch.int32))
            else:
                # cu[0] affects only sequence zero.
                cu.copy_(torch.tensor([value, 1, 2], dtype=torch.int32))
            launch()
            assert torch.isnan(out[0].cpu()).all()
            torch.testing.assert_close(
                out[1].cpu().float(), reference[1], atol=0.01, rtol=0.02
            )
            table.copy_(valid_table)
            used.copy_(torch.tensor([capacity, capacity], dtype=torch.int32))
            cu.copy_(torch.arange(batch + 1, dtype=torch.int32))
            launch()
            torch.testing.assert_close(
                out.cpu().float(), reference, atol=0.01, rtol=0.02
            )
        # Unused table entries are poison, valid zero-length is not a failure.
        table.copy_(torch.full_like(valid_table, -2147483648))
        used.copy_(torch.tensor([0, 0], dtype=torch.int32))
        launch()
        assert torch.count_nonzero(out.cpu()) == 0
        assert torch.equal(
            qback.cpu()[[0, -1]], torch.full((2, hq, dim), 19.0, dtype=dtype)
        )
        assert torch.equal(
            oback.cpu()[[0, -1]], torch.full((2, hq, dim), 23.0, dtype=dtype)
        )


@pytest.mark.parametrize(
    "page,route",
    [(page, "explicit") for page in (1, 16, 32, 64)] + [(1, "auto"), (1, "3d")],
)
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_window_excluded_metadata_and_recovery(
    gpu, monkeypatch, page, route, dtype_name, mode, use_sinks
):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("bounded decode targets gfx942/gfx950")
    from dispatch.attention import (
        AttentionRequest,
        AttentionMaskType,
        dispatch_attention,
    )
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    capacity, dim, batch, hq, hk = 65, 64, 2, 8, 2
    torch.manual_seed(113)
    qh = torch.randn(batch, hq, dim, dtype=dtype)
    pages = (capacity + page - 1) // page
    count = batch * pages
    kh = torch.randn(count, page, hk, dim, dtype=dtype)
    vh = torch.randn_like(kh)
    qback = torch.full((batch + 2, hq, dim), 19.0, dtype=dtype).cuda()
    oback = torch.full((batch + 2, hq, dim), 23.0, dtype=dtype).cuda()
    q, out = qback[1:-1], oback[1:-1]
    q.copy_(qh)
    k, v = kh.cuda(), vh.cuda()
    cu = torch.arange(batch + 1, dtype=torch.int32).cuda()
    used = torch.full((batch,), capacity, dtype=torch.int32).cuda()
    valid_table = torch.arange(count, dtype=torch.int32).reshape(batch, pages)
    table = valid_table.cuda()
    sink_host = torch.linspace(-1, 2, hq).to(dtype) if use_sinks else None
    sinks = sink_host.cuda() if use_sinks else None
    logical_k = kh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
    logical_v = vh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
    features = dict(softcap=0, sinks=sink_host, alibi_slopes=None, qq_bias=None)

    for window in (1, 33):
        req = AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=1,
            seqlen_k=capacity,
            hdim_q=dim,
            hdim_v=dim,
            arch=arch,
            dtype=dtype_name,
            kv_block_size=page,
            algorithm="auto" if page == 1 else "paged_decode_t32",
            mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
            sliding_window=window,
            use_sinks=use_sinks,
            target_ctas=8,
        )
        selected = dispatch_attention(req) if route == "explicit" else None
        if selected is not None:
            assert selected.candidate.name == "attention_paged_decode_t32"
            assert selected.spec.kernel_spec.metadata_containment
            assert selected.spec.reduce_spec.metadata_containment
        args = dict(
            problem=_problem(req),
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            block_table=table,
            sinks=sinks,
            softmax_scale=1 / math.sqrt(dim),
            softcap=0.0,
            backend="3d" if route == "explicit" else route,
            tuning_spec=selected.spec if selected is not None else None,
            use_graph=mode == "graph",
        )
        table.copy_(valid_table)
        used.copy_(torch.full((batch,), capacity, dtype=torch.int32))
        with decode_launch_session(torch, au, args, mode) as launch:

            expected = _feature_reference(
                torch, qh, logical_k, logical_v, [capacity] * batch, window, features
            )
            begin = capacity - window
            live_page = begin // page
            for invalid in (-1, count, -2147483648, 2147483647):
                # A partly live page is retained; only wholly excluded pages
                # may have evicted IDs. The same graph observes each mutation.
                evicted = valid_table.clone()
                evicted[0, :live_page] = invalid
                table.copy_(evicted)
                launch()
                torch.testing.assert_close(
                    out.cpu().float(), expected, atol=0.01, rtol=0.02
                )
                consumed = evicted.clone()
                consumed[0, live_page] = invalid
                table.copy_(consumed)
                launch()
                assert torch.isnan(out[0].cpu()).all()
                torch.testing.assert_close(
                    out[1].cpu().float(), expected[1], atol=0.01, rtol=0.02
                )
                table.copy_(evicted)
                launch()
                torch.testing.assert_close(
                    out.cpu().float(), expected, atol=0.01, rtol=0.02
                )
            # Empty sequences must remain neutral even with a sink and poison.
            table.copy_(torch.full_like(valid_table, -2147483648))
            used.copy_(torch.zeros(batch, dtype=torch.int32))
            launch()
            assert torch.count_nonzero(out.cpu()) == 0
            assert torch.equal(
                qback.cpu()[[0, -1]], torch.full((2, hq, dim), 19.0, dtype=dtype)
            )
            assert torch.equal(
                oback.cpu()[[0, -1]], torch.full((2, hq, dim), 23.0, dtype=dtype)
            )
