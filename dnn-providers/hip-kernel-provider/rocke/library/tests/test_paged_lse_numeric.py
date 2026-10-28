# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hardware O/LSE agreement, bounded metadata recovery and graph ownership."""

import math

import pytest

from tests._decode_numeric_support import gpu
from tests._decode_reference import reference


@pytest.mark.gpu
@pytest.mark.parametrize(
    "page,route,dim,ratio",
    [(p, "explicit", 64, 4) for p in (1, 16, 32, 64)]
    + [(1, "auto", 128, 4), (1, "3d", 64, 1), (1, "3d", 256, 16)],
)
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
def test_bounded_lse_failure_empty_window_and_recovery(
    gpu, monkeypatch, page, route, dim, ratio, dtype_name, mode
):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("bounded decode LSE targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    batch, hk, capacity, window = 2, 2, 97, 33
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
        sliding_window=window,
        emit_lse=True,
        mask_type=2,
    )
    selected = dispatch_attention(req) if route == "explicit" else None
    problem = _problem(req)
    pages = (capacity + page - 1) // page
    count = batch * pages
    torch.manual_seed(13128)
    qh = torch.randn(batch, hq, dim, dtype=dtype)
    kh = torch.randn(count, page, hk, dim, dtype=dtype)
    vh = torch.randn_like(kh)
    valid = torch.arange(count - 1, -1, -1, dtype=torch.int32).reshape(batch, pages)
    q = qh.cuda()
    k, v = kh.cuda(), vh.cuda()
    oback = torch.full((batch + 2, hq, dim), 23.0, dtype=dtype).cuda()
    lback = torch.full((batch + 2, hq, 1), 29.0, dtype=torch.float32).cuda()
    out, lse = oback[1:-1], lback[1:-1]
    cu = torch.arange(batch + 1, dtype=torch.int32).cuda()
    used = torch.full((batch,), capacity, dtype=torch.int32).cuda()
    table = valid.cuda()
    args = dict(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        cu_seqlens_q=cu,
        seqused_k=used,
        block_table=table,
        softmax_scale=1 / math.sqrt(dim),
        softcap=0.0,
        backend="3d" if route == "explicit" else route,
        tuning_spec=selected.spec if selected else None,
        use_graph=mode == "graph",
    )
    owner = graph = None
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

    expected_o, expected_l = reference(
        torch, qh, kh, vh, valid, [capacity] * batch, page, window
    )

    def reset():
        table.copy_(valid)
        used.copy_(torch.tensor([capacity, capacity], dtype=torch.int32))
        cu.copy_(torch.arange(batch + 1, dtype=torch.int32))

    def compare(o, l):
        torch.testing.assert_close(out.cpu().float(), o, atol=0.01, rtol=0.02)
        torch.testing.assert_close(lse.cpu(), l, atol=0.004, rtol=0.004)

    try:
        launch()
        compare(expected_o, expected_l)
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
            reset()
            if kind == "page":
                bad = valid.clone()
                bad[0, -1] = value
                table.copy_(bad)
            elif kind == "length":
                used.copy_(torch.tensor([value, capacity], dtype=torch.int32))
            else:
                cu.copy_(torch.tensor([value, 1, 2], dtype=torch.int32))
            launch()
            assert torch.isnan(out[0].cpu()).all()
            assert torch.isnan(lse[0].cpu()).all()
            torch.testing.assert_close(
                out[1].cpu().float(), expected_o[1], atol=0.01, rtol=0.02
            )
            torch.testing.assert_close(
                lse[1].cpu(), expected_l[1], atol=0.004, rtol=0.004
            )
            reset()
            launch()
            compare(expected_o, expected_l)
        # Earlier pages are outside the sliding window and must not be audited.
        poison = valid.clone()
        for col in range(pages):
            if (col + 1) * page <= capacity - window:
                poison[:, col] = -2147483648
        table.copy_(poison)
        launch()
        compare(expected_o, expected_l)
        # A keyless row is valid, unlike a consumed malformed page or length.
        table.copy_(torch.full_like(valid, -2147483648))
        used.copy_(torch.zeros(batch, dtype=torch.int32))
        launch()
        assert torch.count_nonzero(out.cpu()) == 0
        assert torch.isneginf(lse.cpu()).all()
        reset()
        launch()
        compare(expected_o, expected_l)
        assert torch.equal(
            oback.cpu()[[0, -1]], torch.full((2, hq, dim), 23.0, dtype=dtype)
        )
        assert torch.equal(
            lback.cpu()[[0, -1]], torch.full((2, hq, 1), 29.0, dtype=torch.float32)
        )
        if mode == "graph":
            # Same tensors/scalars except LSE: capture must retain a new pointer.
            other = torch.full((batch, hq, 1), 41.0, dtype=torch.float32).cuda()
            lse.copy_(torch.full((batch, hq, 1), 37.0, dtype=torch.float32))
            au.run_unified_attention_torch(**dict(args, lse=other))
            torch.cuda.synchronize()
            torch.testing.assert_close(other.cpu(), expected_l, atol=0.004, rtol=0.004)
            assert torch.equal(lse.cpu(), torch.full((batch, hq, 1), 37.0))
    finally:
        if owner is not None:
            owner.close()
        au.clear_attention_3d_graph_cache()
