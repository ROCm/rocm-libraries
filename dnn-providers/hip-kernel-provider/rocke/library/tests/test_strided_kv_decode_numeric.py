# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Actual tiled split-KV launches on contiguous and padded non-paged caches.

Set ROCKE_REQUIRE_DECODE_GPU=gfx942/gfx950 to require hardware (no skips).
For shared paged graph coverage on gfx1250, set that target and select
``-k 'graph_replay and paged and bf16'`` (D64, GQA-8).
Reference attention is computed independently in FP32 on the CPU from logical
KV tensors, after rounding inputs to the tested fp16/bf16 dtype.
"""

import math
import os
from collections import OrderedDict
from dataclasses import replace

import pytest


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    if required:
        import torch
    else:
        torch = pytest.importorskip("torch")
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available():
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    arch = get_device_arch()
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        if required:
            pytest.fail(f"unsupported required decode target {arch}")
        pytest.skip("decode validation targets gfx942/gfx950/gfx1250")
    return torch, arch


def _reference(torch, q, k, v, lengths, window):
    output = torch.zeros_like(q, dtype=torch.float32)
    ratio = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        begin = max(0, length - window) if window else 0
        for head in range(q.shape[1]):
            keys = k[batch, head // ratio, begin:length].float()
            values = v[batch, head // ratio, begin:length].float()
            scores = (keys @ q[batch, head].float()) / math.sqrt(q.shape[2])
            output[batch, head] = scores.softmax(0) @ values
    return output


def _cache(torch, logical, layout):
    batch, heads, capacity, dim = logical.shape
    if layout == "bhsd":
        return logical.cuda()
    if layout == "bshd":
        return logical.permute(0, 2, 1, 3).contiguous().cuda().permute(0, 2, 1, 3)
    # Unit-D rows separated by 8 unused elements, plus batch/head padding.
    row_stride = dim + 8
    head_stride = (capacity + 1) * row_stride
    batch_stride = (heads + 1) * head_stride
    span = (
        (batch - 1) * batch_stride
        + (heads - 1) * head_stride
        + (capacity - 1) * row_stride
        + dim
    )
    backing = torch.full((span,), float("nan"), dtype=logical.dtype)
    view = backing.as_strided(logical.shape, (batch_stride, head_stride, row_stride, 1))
    view.copy_(logical)
    return backing.cuda().as_strided(logical.shape, view.stride())


def _paged_cache(torch, logical, block_size):
    batch, heads, capacity, dim = logical.shape
    blocks = (capacity + block_size - 1) // block_size
    backing = torch.zeros(batch, blocks * block_size, heads, dim, dtype=logical.dtype)
    backing[:, :capacity].copy_(logical.permute(0, 2, 1, 3))
    pages = backing.reshape(batch * blocks, block_size, heads, dim).flip(0).contiguous()
    table = torch.arange(batch * blocks - 1, -1, -1, dtype=torch.int32).view(
        batch, blocks
    )
    return pages.cuda(), table.cuda()


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "dim,block,tile",
    [(256, 16, None), (256, 32, None), (128, 64, None), (256, 64, 16), (256, 64, None)],
)
def test_strided_lds_admission_numeric(gpu, dtype, dim, block, tile, monkeypatch):
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    torch, arch = gpu
    request = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype,
        kv_layout="strided",
        kv_block_size=block,
        target_ctas=8,
    )
    problem = _problem(request)
    torch.manual_seed(314)
    storage_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    q_cpu = torch.randn(3, 8, dim).to(storage_dtype)
    k_cpu = torch.randn(3, 2, 65, dim).to(storage_dtype)
    v_cpu = torch.randn(3, 2, 65, dim).to(storage_dtype)
    lengths = [0, 64, 65]
    reference = _reference(torch, q_cpu, k_cpu, v_cpu, lengths, 0)
    tensors = dict(
        q=q_cpu.cuda(),
        k=_cache(torch, k_cpu, "bshd"),
        v=_cache(torch, v_cpu, "bhsd"),
        out=torch.empty_like(q_cpu).cuda(),
        cu_seqlens_q=torch.arange(4, dtype=torch.int32).cuda(),
        seqused_k=torch.tensor(lengths, dtype=torch.int32).cuda(),
    )
    spec = None
    if tile is not None or (arch == "gfx942" and dim == 256 and block == 64):
        supported = dispatch_attention(replace(request, kv_block_size=32)).spec
        spec = replace(
            supported,
            kernel_spec=replace(
                supported.kernel_spec, block_size=block, tile_size_override=tile or 32
            ),
        )
    kwargs = dict(
        problem=problem,
        **tensors,
        block_table=None,
        softmax_scale=1 / math.sqrt(dim),
        softcap=0,
        kv_layout="strided",
    )
    if arch == "gfx942" and dim == 256 and block == 64 and tile is None:
        monkeypatch.setattr(
            au,
            "compile_kernel",
            lambda *a, **kw: pytest.fail("must reject before compile"),
        )
        with pytest.raises(ValueError, match="74752 B LDS"):
            dispatch_attention(request)
        for explicit in (None, spec):
            with pytest.raises(ValueError, match="74752 B LDS"):
                au.run_unified_attention_torch(**kwargs, tuning_spec=explicit)
        return
    if tile is None:
        selected = dispatch_attention(request)
        binding = selected.candidate.bind_torch(
            request, selected.spec, dict(problem=problem, **tensors)
        )
        binding.launch()
    else:
        au.run_unified_attention_torch(**kwargs, tuning_spec=spec)
    atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)
    torch.testing.assert_close(
        tensors["out"].cpu().float(), reference, atol=atol, rtol=rtol
    )
    if tile is None:
        au.run_unified_attention_torch(**kwargs)
        torch.testing.assert_close(
            tensors["out"].cpu().float(), reference, atol=atol, rtol=rtol
        )


# Distinct batches, empty/partial segments, both physical orders, independent
# K/V strides, and final tiles without allocation padding. Repeated passes change
# the lengths in place to exercise binding/graph cache reuse.
_CASES = [
    (1, 64, 2, 4, "bhsd", "bhsd", 0),
    (17, 64, 2, 4, "bshd", "bshd", 0),
    (65, 128, 2, 8, "bhsd", "bshd", 0),
    (97, 128, 2, 8, "bshd", "bhsd", 17),
    (65, 256, 1, 8, "bhsd", "bhsd", 0),
    (129, 256, 2, 8, "padded", "bshd", 33),
    (1031, 64, 2, 4, "bshd", "padded", 0),
]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
def test_strided_kv_decode(
    gpu,
    monkeypatch,
    dtype,
    case,
    kv_layout,
    graph=False,
    default_stream=False,
    block_size=16,
):
    torch, arch = gpu
    from kernels.common import attention_unified as au

    if arch == "gfx1250" and (
        kv_layout == "strided"
        or dtype != "bf16"
        or case[1] != 64
        or case[3] // case[2] != 8
    ):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    monkeypatch.setenv("HIPDNN_GFX942_3D_GRAPH", str(int(graph)))
    monkeypatch.setenv("HIPDNN_GFX950_3D_GRAPH", str(int(graph)))
    monkeypatch.setenv("HIPDNN_GFX1250_3D_GRAPH", str(int(graph)))
    capacity, dim, kv_heads, q_heads, k_layout, v_layout, window = case
    batch = 3
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(701 + capacity + dim)
    q_host = torch.randn(batch, q_heads, dim, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, dim, generator=rng).to(tdtype)
    v_host = torch.randn(batch, kv_heads, capacity, dim, generator=rng).to(tdtype)
    # Large per-batch shifts make shared-cache bugs observable.
    for b in range(batch):
        v_host[b] += b
    q = q_host.cuda()
    if kv_layout == "strided":
        k, v = _cache(torch, k_host, k_layout), _cache(torch, v_host, v_layout)
        block_table = None
    else:
        k, block_table = _paged_cache(torch, k_host, block_size)
        v, _ = _paged_cache(torch, v_host, block_size)
    cu_q = torch.arange(batch + 1, dtype=torch.int32).cuda()
    lengths_host = [capacity, max(0, capacity - 3), 0]
    lengths = torch.tensor(lengths_host, dtype=torch.int32).cuda()
    count = q.numel()
    storage = torch.full((count + 64,), 117, dtype=tdtype).cuda()
    output = storage[32:-32].view(q.shape)
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=dim,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        sliding_window=window,
        clamp_arch=arch,
        target_ctas=8,
    )
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    torch.cuda.synchronize()
    # Paged cases exercise the direct runtime; strided cases also test dispatch.
    if kv_layout == "strided":
        from dispatch.attention import AttentionRequest, dispatch_attention

        request = AttentionRequest(
            batch=batch,
            nhead_q=q_heads,
            nhead_k=kv_heads,
            seqlen_q=1,
            seqlen_k=capacity,
            hdim_q=dim,
            hdim_v=dim,
            arch=arch,
            dtype=dtype,
            kv_layout="strided",
            kv_block_size=block_size,
            sliding_window=window,
            target_ctas=8,
        )
        result = dispatch_attention(request)
        assert result.candidate.name == "attention_strided_decode"
        binding = result.candidate.bind_torch(
            request,
            result.spec,
            dict(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=output,
                cu_seqlens_q=cu_q,
                seqused_k=lengths,
            ),
            stream=stream.cuda_stream,
        )
    graphs_before = set(au._3D_GRAPHS)
    captured_graphs = {}
    for step in range(3):
        if step:
            lengths_host = (
                [max(0, capacity - 1), capacity, min(1, capacity)]
                if step == 1
                else [capacity, min(1, capacity), capacity]
            )
            lengths.copy_(torch.tensor(lengths_host, dtype=torch.int32))
            torch.cuda.synchronize()
        if step and kv_layout == "strided":
            binding.launch()
        else:
            au.run_unified_attention_torch(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=output,
                cu_seqlens_q=cu_q,
                seqused_k=lengths,
                softmax_scale=1 / math.sqrt(dim),
                block_table=block_table,
                softcap=0.0,
                backend="3d",
                kv_layout=kv_layout,
                stream=stream.cuda_stream,
            )
        stream.synchronize()
        if graph:
            graph_keys = set(au._3D_GRAPHS) - graphs_before
            # Direct and explicit-spec launches have separate cache identities.
            assert len(graph_keys) == 1
            current_graphs = {key: au._3D_GRAPHS[key] for key in graph_keys}
            if kv_layout == "strided" and step:
                owned = binding.launch.execution._graphs
                assert len(owned) == 1
                current_graphs.update(owned)
            assert all(
                current_graphs[key] is value for key, value in captured_graphs.items()
            )
            captured_graphs = current_graphs
        expected = _reference(torch, q_host, k_host, v_host, lengths_host, window)
        atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
        torch.testing.assert_close(output.cpu().float(), expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            storage[:32].cpu(), torch.full((32,), 117, dtype=tdtype), atol=0, rtol=0
        )
        torch.testing.assert_close(
            storage[-32:].cpu(), torch.full((32,), 117, dtype=tdtype), atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
@pytest.mark.parametrize("default_stream", [False, True])
def test_strided_decode_graph_replay(
    gpu, monkeypatch, dtype, kv_layout, default_stream
):
    case = (17, 64, 2, 16, "bshd", "bshd", 0) if gpu[1] == "gfx1250" else _CASES[1]
    test_strided_kv_decode(
        gpu,
        monkeypatch,
        dtype,
        case,
        kv_layout,
        graph=True,
        default_stream=default_stream,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("window", [0, 17])
def test_strided_decode_tile_policy(gpu, monkeypatch, dtype, block_size, window):
    case = (*_CASES[2][:-1], window)
    test_strided_kv_decode(
        gpu,
        monkeypatch,
        dtype,
        case,
        "strided",
        block_size=block_size,
    )


@pytest.fixture
def concurrent_decode(gpu, monkeypatch):
    """Own streams and drain asynchronous references without serializing calls."""
    torch, arch = gpu
    from kernels.common import attention_unified as au
    from rocke.runtime.launcher import synchronize_and_release

    for target in ("GFX942", "GFX950", "GFX1250"):
        monkeypatch.setenv(f"HIPDNN_{target}_3D_GRAPH", "0")
    # Isolate graph lifetimes between tests, not between concurrent invocations.
    monkeypatch.setattr(au, "_3D_GRAPHS", OrderedDict())
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    try:
        yield torch, arch, streams
    finally:
        synchronize_and_release()


@pytest.mark.parametrize("mode", ["eager", "internal", "external"])
@pytest.mark.parametrize("kv_layout", ["paged", "strided"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_decode_concurrent_workspace(
    concurrent_decode, monkeypatch, mode, kv_layout, dtype
):
    torch, arch, streams = concurrent_decode
    if arch == "gfx1250" and (kv_layout != "paged" or dtype != "bf16"):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    from kernels.common import attention_unified as au
    from rocke.runtime.launcher import no_fence, synchronize_and_release

    batch, q_heads, kv_heads, capacity, dim = 3, 16, 2, 257, 64
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=dim,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        clamp_arch=arch,
        target_ctas=8,
    )
    requests, expected = [], []
    for index in range(2):
        rng = torch.Generator().manual_seed(1701 + index)
        q = torch.randn(batch, q_heads, dim, generator=rng).to(tdtype)
        k = torch.randn(batch, kv_heads, capacity, dim, generator=rng).to(tdtype)
        v = (torch.randn(batch, kv_heads, capacity, dim, generator=rng) + index * 4).to(
            tdtype
        )
        lengths = [capacity, capacity - 7, 0]
        expected.append(_reference(torch, q, k, v, lengths, 0))
        if kv_layout == "paged":
            kd, table = _paged_cache(torch, k, 16)
            vd, _ = _paged_cache(torch, v, 16)
        else:
            kd, vd, table = _cache(torch, k, "bshd"), _cache(torch, v, "bhsd"), None
        requests.append(
            {
                "problem": problem,
                "q": q.cuda(),
                "k": kd,
                "v": vd,
                "out": torch.empty_like(q, device="cuda"),
                "cu_seqlens_q": torch.arange(batch + 1, dtype=torch.int32).cuda(),
                "seqused_k": torch.tensor(lengths, dtype=torch.int32).cuda(),
                "softmax_scale": 1 / math.sqrt(dim),
                "block_table": table,
                "softcap": 0.0,
                "backend": "3d",
                "kv_layout": kv_layout,
            }
        )
    torch.cuda.synchronize()
    poison = torch.full((batch, q_heads, dim), float("nan"), dtype=tdtype).pin_memory()

    # Record addresses without retaining tensors: later churn checks graph-pool
    # ownership after the runtime's temporary launch references have drained.
    allocations = []
    launch = au._launch_3d_pipeline

    def observe(prepared, segment, reduce, stream, *, capturing):
        allocations.append(
            (
                capturing,
                tuple(
                    segment[name].data_ptr()
                    for name in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr")
                ),
            )
        )
        return launch(prepared, segment, reduce, stream, capturing=capturing)

    monkeypatch.setattr(au, "_launch_3d_pipeline", observe)
    for request, stream in zip(requests, streams):
        au.run_unified_attention_torch(**request, stream=stream.cuda_stream)
    synchronize_and_release()
    allocations.clear()

    graphs = []
    if mode == "internal":
        monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", "1")
        for request, stream in zip(requests, streams):
            with torch.cuda.stream(stream):
                # Zero means the current stream, also in the graph-cache key.
                au.run_unified_attention_torch(**request, stream=0)
        graphs = list(au._3D_GRAPHS.values())
        assert len(graphs) == 2
    elif mode == "external":
        capture_stream = torch.cuda.Stream()
        eager_request = dict(requests[0], out=torch.empty_like(requests[0]["out"]))
        # Both captures use the SAME stream but independent default graph pools.
        # Replay later uses two other streams: per-capture-stream scratch fails.
        for request in requests:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                au.run_unified_attention_torch(**request, stream=0)
            graphs.append(graph)
    if graphs:
        captured = [addresses for capturing, addresses in allocations if capturing]
        assert len(captured) == 2
        assert set(captured[0]).isdisjoint(captured[1])
        synchronize_and_release()
        allocations.clear()
        # Release cached free blocks and pressure the ordinary allocator before
        # replay; graph-private scratch must survive without Python tensor refs.
        torch.cuda.empty_cache()
        churn = [torch.full((batch, q_heads, 8, dim), 91.0).cuda() for _ in range(8)]
        torch.cuda.synchronize()

    with no_fence():
        for _ in range(8):
            for index, (request, stream) in enumerate(zip(requests, streams)):
                with torch.cuda.stream(stream):
                    request["out"].copy_(poison, non_blocking=True)
                    if mode == "external":
                        graphs[index].replay()
                    else:
                        au.run_unified_attention_torch(
                            **request, stream=stream.cuda_stream
                        )
            if mode == "external":
                au.run_unified_attention_torch(
                    **eager_request, stream=capture_stream.cuda_stream
                )
    synchronize_and_release()
    if mode == "eager":
        # Retained in-flight tensors must be distinct regardless of whether the
        # GPU scheduler actually overlaps kernels during this particular run.
        addresses = [address for _, workspace in allocations for address in workspace]
        assert len(addresses) == len(set(addresses))
    elif mode == "internal":
        assert list(au._3D_GRAPHS.values()) == graphs  # 0 and explicit handle agree
        assert not allocations  # cache hits replay without new scratch allocations
    atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
    for request, reference in zip(requests, expected):
        torch.testing.assert_close(
            request["out"].cpu().float(), reference, atol=atol, rtol=rtol
        )
    if mode == "external":
        for _, workspace in allocations:
            assert set(workspace).isdisjoint(captured[0] + captured[1])
        torch.testing.assert_close(
            eager_request["out"].cpu().float(), expected[0], atol=atol, rtol=rtol
        )
    if graphs:
        assert len(churn) == 8


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kv_layout", ["paged", "strided"])
@pytest.mark.parametrize("graph", [False, True])
def test_prepared_decode_lifecycle(gpu, monkeypatch, dtype, kv_layout, graph):
    """Changing bindings/lengths, graph eviction and scratch reuse on hardware."""
    torch, arch = gpu
    if arch == "gfx1250" and (dtype != "bf16" or kv_layout != "paged"):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    from kernels.common import attention_unified as au
    from rocke.runtime.launcher import no_fence, synchronize_and_release

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", str(int(graph)))
    batch, heads, dim, capacity = 2, 8, 64, 65
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(808)
    q_host = torch.randn(batch, heads, dim, generator=rng).to(tdtype)
    k_host = torch.randn(batch, 1, capacity, dim, generator=rng).to(tdtype)
    v_host = torch.randn(batch, 1, capacity, dim, generator=rng).to(tdtype)
    if kv_layout == "strided":
        k, v = _cache(torch, k_host, "bshd"), _cache(torch, v_host, "padded")
        table = None
    else:
        k, table = _paged_cache(torch, k_host, 16)
        v, _ = _paged_cache(torch, v_host, 16)
    lengths = torch.tensor([capacity, 0], dtype=torch.int32).cuda()
    stream = torch.cuda.Stream()
    args = dict(
        problem=au.UnifiedAttentionProblem(
            total_q=batch,
            num_seqs=batch,
            num_query_heads=heads,
            num_kv_heads=1,
            head_size=dim,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=capacity,
            dtype=dtype,
            clamp_arch=arch,
            target_ctas=8,
        ),
        q=q_host.cuda(),
        k=k,
        v=v,
        out=torch.empty_like(q_host).cuda(),
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32).cuda(),
        seqused_k=lengths,
        block_table=table,
        softmax_scale=1 / math.sqrt(dim),
        softcap=0.0,
        kv_layout=kv_layout,
        backend="3d",
        stream=stream.cuda_stream,
        use_graph=graph,
    )
    torch.cuda.synchronize()
    allocations = []
    launch = au._launch_3d_pipeline

    def observe(prepared, segment, reduce, stream, *, capturing):
        allocations.append(
            (
                capturing,
                tuple(
                    int(segment[n].data_ptr())
                    for n in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr")
                ),
            )
        )
        return launch(prepared, segment, reduce, stream, capturing=capturing)

    monkeypatch.setattr(au, "_launch_3d_pipeline", observe)
    execution = au.prepare_unified_attention_torch(**args)
    scratch = tuple(t.data_ptr() for t in execution._workspace)
    try:
        for step in range(24):
            lens = [capacity - step % 5, step % capacity]
            q_step = (q_host.float() + step / 64).to(tdtype)
            with torch.cuda.stream(stream):
                lengths.copy_(torch.tensor(lens, dtype=torch.int32), non_blocking=True)
                # New Q/output objects force graph rebinding, as in decode callers.
                q = q_step.cuda()
                output = torch.empty_like(q)
                with no_fence():
                    execution.launch(q=q, out=output)
            synchronize_and_release(stream.cuda_stream)
            expected = _reference(torch, q_step, k_host, v_host, lens, 0)
            atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
            torch.testing.assert_close(
                output.cpu().float(), expected, atol=atol, rtol=rtol
            )
            assert tuple(t.data_ptr() for t in execution._workspace) == scratch
            assert len(execution._graphs) <= 1
        eager = [p for capture, p in allocations if not capture]
        assert all(p == scratch for p in eager)
        # An outer capture on a different capture stream gets independent scratch.
        outer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(outer):
            execution.launch(q=q, out=output)
        captured = allocations[-1]
        assert captured[0] and set(captured[1]).isdisjoint(scratch)
        synchronize_and_release()
        with torch.cuda.stream(stream):
            outer.replay()
        synchronize_and_release(stream.cuda_stream)
        torch.testing.assert_close(output.cpu().float(), expected, atol=atol, rtol=rtol)
    finally:
        execution.close()
    assert not execution._graphs and execution._workspace is None
    churn = [
        torch.empty((batch, heads, capacity, dim), device="cuda") for _ in range(8)
    ]
    torch.cuda.synchronize()
    assert churn
