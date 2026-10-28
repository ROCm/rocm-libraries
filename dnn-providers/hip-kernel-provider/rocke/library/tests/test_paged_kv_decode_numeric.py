# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU page/tile loader numerics with CPU references after quantization."""

import math

import pytest

from tests._decode_numeric_support import gpu
from tests._decode_reference import _reference, _feature_reference


def _shuffled_cache(torch, logical, page, *, poison_tail=True):
    batch, heads, capacity, dim = logical.shape
    count = (capacity + page - 1) // page
    # Scatter logical pages into nonadjacent physical pages. Unreferenced pages
    # and final-page padding are NaN so an incorrect mapping cannot pass quietly.
    ids = torch.randperm(
        batch * count * 2, generator=torch.Generator().manual_seed(51)
    )[: batch * count].reshape(batch, count)
    pool = torch.full(
        (batch * count * 2, page, heads, dim), float("nan"), dtype=logical.dtype
    )
    for b in range(batch):
        for i in range(count):
            if not poison_tail:
                pool[ids[b, i]].zero_()
            n = min(page, capacity - i * page)
            pool[ids[b, i], :n] = logical[b, :, i * page : i * page + n].permute(
                1, 0, 2
            )
    return pool.cuda(), ids.to(torch.int32).cuda()


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_paged_tile_decode(
    gpu,
    monkeypatch,
    page,
    dim,
    dtype,
    graph=False,
    default_stream=False,
    tile=32,
    transport="plain",
):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("token-mapped loader targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import AttentionMaskType
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", str(int(graph)))
    batch, heads, capacity = 3, 2, 97
    ratio = {64: 1, 128: 4, 256: 16}[dim]
    qheads = heads * ratio
    window = 0 if graph else (33 if dim == 128 else 0)
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(913 + dim)
    qh = torch.randn(batch, qheads, dim, generator=rng).to(tdtype)
    kh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    vh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    for b in range(batch):
        vh[b] += b
    q = qh.cuda()
    # Legacy page-contained loaders require finite page padding: PV can read
    # masked V rows. New gather paths must safely handle NaN tails as well.
    poison_tail = (
        tile == 32
        and transport == "plain"
        and (page < 32 or (arch == "gfx950" and page == 64))
    )
    k, table = _shuffled_cache(torch, kh, page, poison_tail=poison_tail)
    v, _ = _shuffled_cache(torch, vh, page, poison_tail=poison_tail)
    cu = torch.arange(batch + 1, dtype=torch.int32).cuda()
    used = torch.tensor([capacity, capacity - 1, 0], dtype=torch.int32).cuda()
    storage = torch.full((q.numel() + 64,), 117, dtype=tdtype).cuda()
    out = storage[32:-32].view(q.shape)
    request = AttentionRequest(
        batch=batch,
        nhead_q=qheads,
        nhead_k=heads,
        seqlen_q=1,
        seqlen_k=capacity,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype,
        kv_block_size=page,
        sliding_window=window,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        target_ctas=8,
        algorithm="paged_decode_t32",
    )
    result = dispatch_attention(request)
    from dispatch.attention.common import _problem

    problem = _problem(request)
    runtime_spec = result.spec
    if tile != 32 or transport != "plain":
        from dataclasses import replace

        runtime_spec = replace(
            result.spec,
            reduce_spec=replace(result.spec.reduce_spec, metadata_containment=False),
            kernel_spec=replace(
                result.spec.kernel_spec,
                metadata_containment=False,
                tile_size_override=tile,
                use_kv_lds_swizzle=transport in ("swizzle", "swizzle_page_ids"),
                use_wave_page_id_load=transport == "swizzle_page_ids",
            ),
        )
    assert runtime_spec.kernel_spec.tile_size == tile
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    torch.cuda.synchronize()
    before = set(au._3D_GRAPHS)
    captured = {}
    # Cross both page and tile boundaries, then empty and one-token rows. No
    # padding in the page table beyond ceil(capacity/page).
    lengths_cases = (
        [capacity, capacity - 1, 0],
        [min(page + 1, capacity), 31, 32],
        [1, 33, 0],
    )
    for lengths in lengths_cases:
        with torch.cuda.stream(stream):
            used.copy_(torch.tensor(lengths, dtype=torch.int32).cuda())
            au.run_unified_attention_torch(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu,
                seqused_k=used,
                softmax_scale=1 / math.sqrt(dim),
                block_table=table,
                softcap=0,
                backend="3d",
                stream=stream.cuda_stream,
                tuning_spec=runtime_spec,
            )
        stream.synchronize()
        expected = _reference(torch, qh, kh, vh, lengths, window)
        atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)
        torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
        assert bool((storage[:32].cpu() == 117).all()) and bool(
            (storage[-32:].cpu() == 117).all()
        )
        if graph:
            keys = set(au._3D_GRAPHS) - before
            assert len(keys) == 1
            now = {key: au._3D_GRAPHS[key] for key in keys}
            assert all(now[key] is val for key, val in captured.items())
            captured = now
    if page == 1 and not graph:
        au.run_unified_attention_torch(
            problem=problem,
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            softmax_scale=1 / math.sqrt(dim),
            block_table=table,
            softcap=0,
            backend="3d",
            stream=stream.cuda_stream,
        )
        stream.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
    # The candidate remains T32-only. Check its original spec after direct
    # larger-tile launches so both runtime and public binding stay covered.
    binding = result.candidate.bind_torch(
        request,
        result.spec,
        {
            "problem": problem,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": cu,
            "seqused_k": used,
            "block_table": table,
        },
        stream=stream.cuda_stream,
    )
    binding.launch()
    stream.synchronize()
    torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("default_stream", [False, True])
def test_paged_tile_graph(gpu, monkeypatch, page, default_stream):
    test_paged_tile_decode(
        gpu, monkeypatch, page, 64, "bf16", graph=True, default_stream=default_stream
    )


@pytest.mark.parametrize("page", [1, 16, 64])
def test_paged_binding_metadata_replay(gpu, monkeypatch, page):
    """The candidate keeps device metadata live across valid in-place updates."""
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("token-mapped loader targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import AttentionMaskType
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
    req = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="bf16",
        kv_block_size=page,
        algorithm="paged_decode_t32",
        target_ctas=8,
    )
    result = dispatch_attention(req)
    p = _problem(req)
    torch.manual_seed(73)
    qh = torch.randn((1, 8, 64), dtype=torch.bfloat16)
    pages = (33 + page - 1) // page
    kh = torch.randn((2 * pages, page, 2, 64), dtype=torch.bfloat16)
    vh = torch.randn_like(kh)
    q, k, v = qh.cuda(), kh.cuda(), vh.cuda()
    out = torch.empty_like(q)
    cu = torch.tensor([0, 1], dtype=torch.int32).cuda()
    used = torch.tensor([33], dtype=torch.int32).cuda()
    table = torch.arange(pages, dtype=torch.int32).reshape(1, pages).cuda()
    tensors = {
        "problem": p,
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu,
        "seqused_k": used,
        "block_table": table,
    }
    binding = result.candidate.bind_torch(req, result.spec, tensors)
    for length, start in ((33, 0), (17, pages), (0, 0), (1, pages)):
        used.copy_(torch.tensor([length], dtype=torch.int32))
        table.copy_(
            torch.arange(start, start + pages, dtype=torch.int32).reshape(1, pages)
        )
        binding.launch()
        torch.cuda.synchronize()
        keys = (
            kh[start : start + pages]
            .reshape(1, pages * page, 2, 64)
            .permute(0, 2, 1, 3)
        )
        values = (
            vh[start : start + pages]
            .reshape(1, pages * page, 2, 64)
            .permute(0, 2, 1, 3)
        )
        expected = _reference(torch, qh, keys, values, [length], 0)
        torch.testing.assert_close(out.cpu().float(), expected, atol=0.01, rtol=0.02)
    au.clear_attention_3d_graph_cache()


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "feature", ["softcap", "sinks", "alibi", "qq_bias", "combined"]
)
def test_paged_features_numeric(gpu, page, dtype, feature):
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import AttentionMaskType
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("token-mapped loader targets gfx942/gfx950")
    batch, heads, qheads, capacity, dim = 3, 2, 8, 97, 128
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(1941)
    qh = (torch.randn(batch, qheads, dim, generator=rng) * 2).to(tdtype)
    kh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    vh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    enabled = lambda name: feature in (name, "combined")
    host_features = dict(
        softcap=1.5 if enabled("softcap") else 0.0,
        sinks=torch.linspace(-1, 2, qheads).to(tdtype) if enabled("sinks") else None,
        alibi_slopes=torch.linspace(0.01, 0.08, qheads) if enabled("alibi") else None,
        qq_bias=(
            torch.tensor([[1.25]], dtype=torch.float32) if enabled("qq_bias") else None
        ),
    )
    features = {
        name: value.cuda() if torch.is_tensor(value) else value
        for name, value in host_features.items()
    }
    window = 33 if feature == "combined" else 0
    req = AttentionRequest(
        batch=batch,
        nhead_q=qheads,
        nhead_k=heads,
        seqlen_q=1,
        seqlen_k=capacity,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype,
        kv_block_size=page,
        algorithm="paged_decode_t32",
        target_ctas=8,
        sliding_window=window,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        softcap=features["softcap"],
        use_sinks=enabled("sinks"),
        use_alibi=enabled("alibi"),
        use_qq_bias=enabled("qq_bias"),
    )
    result = dispatch_attention(req)
    p = _problem(req)
    gather = result.spec.kernel_spec.uses_paged_gather
    q = qh.cuda()
    k, table = _shuffled_cache(torch, kh, page, poison_tail=gather)
    v, _ = _shuffled_cache(torch, vh, page, poison_tail=gather)
    storage = torch.full((q.numel() + 64,), 117, dtype=tdtype).cuda()
    out = storage[32:-32].view(q.shape)
    used = torch.tensor([97, 33, 0], dtype=torch.int32).cuda()
    tensors = dict(
        problem=p,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32).cuda(),
        seqused_k=used,
        block_table=table,
    )
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    arguments = dict(
        tensors,
        **features,
        softmax_scale=1 / math.sqrt(dim),
        qq_bias_stride_0=1 if enabled("qq_bias") else 0,
        backend="3d",
        stream=stream.cuda_stream,
        tuning_spec=result.spec,
        use_graph=False,
    )
    binding = result.candidate.bind_torch(
        req,
        result.spec,
        tensors,
        **features,
        qq_bias_stride_0=arguments["qq_bias_stride_0"],
        stream=stream.cuda_stream,
    )
    atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)

    def check(lengths):
        stream.synchronize()
        expected = _feature_reference(torch, qh, kh, vh, lengths, window, host_features)
        torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
        assert bool((storage[:32].cpu() == 117).all())
        assert bool((storage[-32:].cpu() == 117).all())

    for lengths in ([97, 33, 0], [1, 32, 0]):
        used.copy_(torch.tensor(lengths, dtype=torch.int32))
        torch.cuda.synchronize()
        au.run_unified_attention_torch(**arguments)
        check(lengths)
        binding.launch()
        check(lengths)
        if page == 1:
            au.run_unified_attention_torch(**dict(arguments, tuning_spec=None))
            check(lengths)
    if feature == "combined":
        execution = au.prepare_unified_attention_torch(**arguments)
        try:
            execution.launch()
            check(lengths)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                execution.launch()
            for lengths in ([65, 17, 0], [1, 33, 0]):
                used.copy_(torch.tensor(lengths, dtype=torch.int32))
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    graph.replay()
                check(lengths)
            del graph
        finally:
            execution.close()


@pytest.mark.parametrize(
    "page,tile,transport",
    [
        (page, tile, transport)
        for page, tile in [(16, 64), (32, 64), (16, 128), (64, 128)]
        for transport in ["plain", "swizzle_page_ids"]
    ]
    + [
        (page, tile, transport)
        for page, tile in [(16, 48), (16, 80), (16, 96), (32, 96), (16, 112)]
        for transport in ["plain", "swizzle"]
    ],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_gfx950_whole_page_tiles_numeric(
    gpu, monkeypatch, page, tile, transport, dtype
):
    # Whole-page DMA requires finite physical page padding; token gather's
    # NaN-tail contract is covered separately. Both use shuffled physical pages.
    if gpu[1] != "gfx950":
        pytest.skip("gfx950 whole-page loader")
    test_paged_tile_decode(
        gpu, monkeypatch, page, 64, dtype, tile=tile, transport=transport
    )
