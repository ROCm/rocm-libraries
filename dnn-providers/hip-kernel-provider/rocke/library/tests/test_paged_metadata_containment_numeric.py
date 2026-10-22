# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hardware containment and recovery after ordered live metadata mutation."""

import math

import pytest

from tests.test_strided_kv_decode_numeric import gpu, _reference


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
def test_live_metadata_failure_and_recovery(gpu, monkeypatch, page, dtype_name, mode):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("bounded decode targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    capacity, dim, batch, hq, hk = 65, 64, 2, 8, 2
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
    selected = dispatch_attention(req)
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
        backend="3d",
        tuning_spec=selected.spec,
        use_graph=mode == "graph",
    )
    owner = None
    graph = None
    if mode in ("prepared", "external"):
        owner = au.prepare_unified_attention_torch(**args)
    if mode == "external":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            owner.launch()

    def launch():
        if graph is not None:
            graph.replay()
        elif owner is not None:
            owner.launch()
        else:
            au.run_unified_attention_torch(**args)
        torch.cuda.synchronize()

    logical_k = kh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
    logical_v = vh.reshape(batch, pages * page, hk, dim).permute(0, 2, 1, 3)
    reference = _reference(torch, qh, logical_k, logical_v, [capacity] * batch, 0)
    try:
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
    finally:
        if owner is not None:
            owner.close()
        au.clear_attention_3d_graph_cache()
