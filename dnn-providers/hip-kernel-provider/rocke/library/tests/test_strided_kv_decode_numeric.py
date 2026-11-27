# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Actual tiled split-KV launches on contiguous and padded non-paged caches.

Set ROCKE_REQUIRE_DECODE_GPU=gfx942/gfx950 to require hardware (no skips).
For shared paged graph coverage on gfx1250, set that target and select
``-k 'graph_replay and paged and bf16'`` (D64, GQA-8).
Reference attention is the FP64 numpy decode over the fp16/bf16-rounded inputs.

The decode launches run on ``rocke.runtime`` device arrays and HIP streams. Graph
replay, torch streams and outer torch graph captures belong to torch callers, so
those tests build the same inputs as torch tensors and need torch.
"""

import math
import os
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import Tolerance, assert_within_tolerance
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import Runtime, get_device_arch
from rocke.tests.lanes import TICKET, import_or_skip_until

from ._attention_unified_support import (
    decode_reference,
    paged_cache,
    paged_layout,
    random_half,
)


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    arch = get_device_arch()
    if arch is None:
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        if required:
            pytest.fail(f"unsupported required decode target {arch}")
        pytest.skip("decode validation targets gfx942/gfx950/gfx1250")
    return arch


@pytest.fixture(scope="module")
def torch_gpu(gpu):
    """Torch with a GPU, for the torch-caller features (graphs, torch streams)."""
    torch = import_or_skip_until("torch", TICKET)
    if not torch.cuda.is_available():
        pytest.skip("torch has no GPU here")
    return torch, gpu


def _reference(q, k, v, lengths, window):
    return decode_reference(
        q, k, v, lengths, 1 / math.sqrt(q.shape[2]), sliding_window=window
    )


def _strided_storage(logical, layout):
    """Host storage of ``logical`` ``[B, H, S, D]`` in ``layout`` and the element
    strides that view it as ``[B, H, S, D]``: ``bhsd`` as is, ``bshd`` token-major,
    ``padded`` with unit-D rows 8 unused (NaN) elements apart plus batch/head
    padding."""
    batch, heads, capacity, dim = logical.shape
    if layout == "bhsd":
        return logical, None
    if layout == "bshd":
        seq = capacity
        return logical.transpose(0, 2, 1, 3), (seq * heads * dim, dim, heads * dim, 1)
    row_stride = dim + 8
    head_stride = (capacity + 1) * row_stride
    batch_stride = (heads + 1) * head_stride
    span = (
        (batch - 1) * batch_stride
        + (heads - 1) * head_stride
        + (capacity - 1) * row_stride
        + dim
    )
    backing = np.full(span, np.nan, np.float32)
    strides = (batch_stride, head_stride, row_stride, 1)
    np.lib.stride_tricks.as_strided(
        backing, logical.shape, tuple(4 * x for x in strides)
    )[...] = logical
    return backing, strides


class _RockeArrays:
    """Device arrays and HIP streams from ``rocke.runtime``."""

    def half(self, values, dtype):
        return DeviceArray(storage.encode(values, dtype), dtype)

    def ints(self, values):
        return DeviceArray(np.asarray(values, np.int32))

    def set_ints(self, array, values):
        array.write(np.asarray(values, np.int32))

    def strided_kv(self, logical, layout, dtype):
        host, strides = _strided_storage(logical, layout)
        array = self.half(host, dtype)
        return array if strides is None else array.view(logical.shape, strides)

    def paged_kv(self, logical, block_size, dtype):
        pages, table = paged_cache(logical, block_size, dtype)
        return pages, self.ints(table)

    def guarded_output(self, shape, dtype, guard, fill):
        """An output view with ``guard`` elements of ``fill`` on each side."""
        count = math.prod(shape)
        backing = self.half(np.full(count + 2 * guard, fill, np.float32), dtype)
        return backing.view(shape, offset=guard), backing

    def read(self, array, dtype):
        return storage.decode(array.values(), dtype)

    def stream(self, default):
        return 0 if default else Runtime().create_stream()

    def handle(self, stream):
        return stream

    def sync(self, stream):
        Runtime().wait_stream(stream)

    def release(self, stream):
        if stream:
            Runtime().destroy_stream(stream)


class _TorchArrays:
    """The same inputs as torch tensors on torch streams."""

    def __init__(self, torch):
        self.torch = torch

    def _dtype(self, dtype):
        return self.torch.float16 if dtype == "fp16" else self.torch.bfloat16

    def half(self, values, dtype):
        bits = storage.encode(values, dtype)
        if dtype == "bf16":
            bits = bits.view(np.int16)
        tensor = self.torch.from_numpy(np.ascontiguousarray(bits))
        return tensor.view(self._dtype(dtype)).cuda()

    def ints(self, values):
        return self.torch.tensor(list(values), dtype=self.torch.int32).cuda()

    def set_ints(self, array, values):
        array.copy_(self.torch.tensor(list(values), dtype=self.torch.int32))
        self.torch.cuda.synchronize()

    def strided_kv(self, logical, layout, dtype):
        host, strides = _strided_storage(logical, layout)
        tensor = self.half(host, dtype)
        return tensor if strides is None else tensor.as_strided(logical.shape, strides)

    def paged_kv(self, logical, block_size, dtype):
        pages, table = paged_layout(logical, block_size)
        return self.half(pages, dtype), self.ints(table.ravel()).view(table.shape)

    def guarded_output(self, shape, dtype, guard, fill):
        count = math.prod(shape)
        backing = self.half(np.full(count + 2 * guard, fill, np.float32), dtype)
        return backing[guard:-guard].view(shape), backing

    def read(self, array, dtype):
        bits = array.detach().cpu().contiguous()
        if dtype == "bf16":
            return storage.decode(
                bits.view(self.torch.int16).numpy().view(np.uint16), dtype
            )
        return bits.float().numpy()

    def stream(self, default):
        torch = self.torch
        return torch.cuda.default_stream() if default else torch.cuda.Stream()

    def handle(self, stream):
        return stream.cuda_stream

    def sync(self, stream):
        stream.synchronize()

    def release(self, stream):
        pass


def _decode_inputs(rng, dtype, batch, q_heads, kv_heads, capacity, dim):
    q = random_half(rng, (batch, q_heads, dim), dtype)
    k = random_half(rng, (batch, kv_heads, capacity, dim), dtype)
    v = random_half(rng, (batch, kv_heads, capacity, dim), dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "dim,block,tile",
    [(256, 16, None), (256, 32, None), (128, 64, None), (256, 64, 16), (256, 64, None)],
)
def test_strided_lds_admission_numeric(gpu, dtype, dim, block, tile):
    from dispatch.attention.common import attention_problem
    from kernels.common import attention_unified as au

    from dispatch.attention import AttentionRequest, dispatch_attention

    arch = gpu
    if arch == "gfx950" and tile is not None and tile < block:
        pytest.skip("gfx950 tile_size_override is a multiple of block_size")
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
    problem = attention_problem(request)
    q, k, v = _decode_inputs(np.random.default_rng(314), dtype, 3, 8, 2, 65, dim)
    # Written to the output before each launch: a row the kernel leaves
    # unwritten fails the comparison instead of matching a zero reference.
    unwritten = np.full(q.shape, np.nan, np.float32)
    lengths = [0, 64, 65]
    reference = _reference(q, k, v, lengths, 0)
    arrays = _RockeArrays()
    tensors = dict(
        q=arrays.half(q, dtype),
        k=arrays.strided_kv(k, "bshd", dtype),
        v=arrays.strided_kv(v, "bhsd", dtype),
        out=arrays.half(unwritten, dtype),
        cu_seqlens_q=arrays.ints(range(4)),
        seqused_k=arrays.ints(lengths),
    )
    spec = None
    if tile is not None:
        supported = dispatch_attention(replace(request, kv_block_size=32)).spec
        spec = replace(
            supported,
            kernel_spec=replace(
                supported.kernel_spec, block_size=block, tile_size_override=tile
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
    if tile is None:
        selected = dispatch_attention(request)
        binding = selected.candidate.bind_torch(
            request, selected.spec, dict(problem=problem, **tensors)
        )
        binding.launch()
    else:
        au.run_unified_attention_torch(**kwargs, tuning_spec=spec)
    atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)
    out = arrays.read(tensors["out"], dtype)
    assert_within_tolerance(out, reference, Tolerance(atol=atol, rtol=rtol))
    if tile is None:
        tensors["out"].write(storage.encode(unwritten, dtype))
        au.run_unified_attention_torch(**kwargs)
        out = arrays.read(tensors["out"], dtype)
        assert_within_tolerance(out, reference, Tolerance(atol=atol, rtol=rtol))


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


def _strided_kv_decode(
    arch,
    arrays,
    monkeypatch,
    dtype,
    case,
    kv_layout,
    graph=False,
    default_stream=False,
    block_size=16,
):
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
    rng = np.random.default_rng(701 + capacity + dim)
    q_host, k_host, v_host = _decode_inputs(
        rng, dtype, batch, q_heads, kv_heads, capacity, dim
    )
    # Large per-batch shifts make shared-cache bugs observable.
    for b in range(batch):
        v_host[b] = storage.round_to(v_host[b] + b, dtype)
    q = arrays.half(q_host, dtype)
    if kv_layout == "strided":
        k = arrays.strided_kv(k_host, k_layout, dtype)
        v = arrays.strided_kv(v_host, v_layout, dtype)
        block_table = None
    else:
        k, block_table = arrays.paged_kv(k_host, block_size, dtype)
        v, _ = arrays.paged_kv(v_host, block_size, dtype)
    cu_q = arrays.ints(range(batch + 1))
    lengths_host = [capacity, max(0, capacity - 3), 0]
    lengths = arrays.ints(lengths_host)
    output, guarded = arrays.guarded_output(q_host.shape, dtype, 32, 117)
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
        target_ctas=8,
    )
    stream = arrays.stream(default_stream)
    try:
        # Paged cases exercise the direct runtime; strided cases also test dispatch.
        if kv_layout == "strided":
            from dispatch.attention.common import AttentionMaskType

            from dispatch.attention import AttentionRequest, dispatch_attention

            # A decode window is anchored at the end of the sequence (bottom-right
            # causal); NO_MASK + sliding_window is the non-causal top-left band.
            request = AttentionRequest(
                mask_type=(
                    AttentionMaskType.BOTTOM_RIGHT_CAUSAL
                    if window
                    else AttentionMaskType.NO_MASK
                ),
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
                stream=arrays.handle(stream),
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
                arrays.set_ints(lengths, lengths_host)
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
                    stream=arrays.handle(stream),
                )
            arrays.sync(stream)
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
                    current_graphs[key] is value
                    for key, value in captured_graphs.items()
                )
                captured_graphs = current_graphs
            expected = _reference(q_host, k_host, v_host, lengths_host, window)
            atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
            assert_within_tolerance(
                arrays.read(output, dtype), expected, Tolerance(atol=atol, rtol=rtol)
            )
            guards = arrays.read(guarded, dtype)
            np.testing.assert_array_equal(guards[:32], 117)
            np.testing.assert_array_equal(guards[-32:], 117)
        if kv_layout == "strided":
            binding.launch.close()
    finally:
        arrays.release(stream)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
def test_strided_kv_decode(gpu, monkeypatch, dtype, case, kv_layout):
    _strided_kv_decode(gpu, _RockeArrays(), monkeypatch, dtype, case, kv_layout)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
@pytest.mark.parametrize("default_stream", [False, True])
def test_strided_decode_graph_replay(
    torch_gpu, monkeypatch, dtype, kv_layout, default_stream
):
    """Graph replay captures through torch: torch tensors on torch streams."""
    torch, arch = torch_gpu
    case = (17, 64, 2, 16, "bshd", "bshd", 0) if arch == "gfx1250" else _CASES[1]
    _strided_kv_decode(
        arch,
        _TorchArrays(torch),
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
    _strided_kv_decode(
        gpu,
        _RockeArrays(),
        monkeypatch,
        dtype,
        case,
        "strided",
        block_size=block_size,
    )


@pytest.fixture
def concurrent_decode(torch_gpu, monkeypatch):
    """Own streams and drain asynchronous references without serializing calls.
    Torch streams and graphs: the torch-caller surface."""
    torch, arch = torch_gpu
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
    arrays = _TorchArrays(torch)
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
        target_ctas=8,
    )
    requests, expected = [], []
    for index in range(2):
        rng = np.random.default_rng(1701 + index)
        q, k, v = _decode_inputs(rng, dtype, batch, q_heads, kv_heads, capacity, dim)
        v = storage.round_to(v + index * 4, dtype)
        lengths = [capacity, capacity - 7, 0]
        expected.append(_reference(q, k, v, lengths, 0))
        if kv_layout == "paged":
            kd, table = arrays.paged_kv(k, 16, dtype)
            vd, _ = arrays.paged_kv(v, 16, dtype)
        else:
            kd = arrays.strided_kv(k, "bshd", dtype)
            vd = arrays.strided_kv(v, "bhsd", dtype)
            table = None
        requests.append(
            {
                "problem": problem,
                "q": arrays.half(q, dtype),
                "k": kd,
                "v": vd,
                "out": arrays.half(np.zeros(q.shape, np.float32), dtype),
                "cu_seqlens_q": arrays.ints(range(batch + 1)),
                "seqused_k": arrays.ints(lengths),
                "softmax_scale": 1 / math.sqrt(dim),
                "block_table": table,
                "softcap": 0.0,
                "backend": "3d",
                "kv_layout": kv_layout,
            }
        )
    torch.cuda.synchronize()
    poison = (
        arrays.half(np.full((batch, q_heads, dim), np.nan, np.float32), dtype)
        .cpu()
        .pin_memory()
    )

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
        assert_within_tolerance(
            arrays.read(request["out"], dtype),
            reference,
            Tolerance(atol=atol, rtol=rtol),
        )
    if mode == "external":
        for _, workspace in allocations:
            assert set(workspace).isdisjoint(captured[0] + captured[1])
        assert_within_tolerance(
            arrays.read(eager_request["out"], dtype),
            expected[0],
            Tolerance(atol=atol, rtol=rtol),
        )
    if graphs:
        assert len(churn) == 8


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kv_layout", ["paged", "strided"])
@pytest.mark.parametrize("graph", [False, True])
def test_prepared_decode_lifecycle(torch_gpu, monkeypatch, dtype, kv_layout, graph):
    """Changing bindings/lengths, graph eviction and scratch reuse on hardware,
    on torch streams and under an outer torch graph capture."""
    torch, arch = torch_gpu
    if arch == "gfx1250" and (dtype != "bf16" or kv_layout != "paged"):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    from kernels.common import attention_unified as au
    from rocke.runtime.launcher import no_fence, synchronize_and_release

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", str(int(graph)))
    batch, heads, dim, capacity = 2, 8, 64, 65
    arrays = _TorchArrays(torch)
    q_host, k_host, v_host = _decode_inputs(
        np.random.default_rng(808), dtype, batch, heads, 1, capacity, dim
    )
    if kv_layout == "strided":
        k = arrays.strided_kv(k_host, "bshd", dtype)
        v = arrays.strided_kv(v_host, "padded", dtype)
        table = None
    else:
        k, table = arrays.paged_kv(k_host, 16, dtype)
        v, _ = arrays.paged_kv(v_host, 16, dtype)
    lengths = arrays.ints([capacity, 0])
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
            target_ctas=8,
        ),
        q=arrays.half(q_host, dtype),
        k=k,
        v=v,
        out=arrays.half(np.zeros(q_host.shape, np.float32), dtype),
        cu_seqlens_q=arrays.ints(range(batch + 1)),
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
            q_step = storage.round_to(q_host + step / 64, dtype)
            with torch.cuda.stream(stream):
                lengths.copy_(torch.tensor(lens, dtype=torch.int32), non_blocking=True)
                # New Q/output objects force graph rebinding, as in decode callers.
                q = arrays.half(q_step, dtype)
                output = torch.empty_like(q)
                with no_fence():
                    execution.launch(q=q, out=output)
            synchronize_and_release(stream.cuda_stream)
            expected = _reference(q_step, k_host, v_host, lens, 0)
            atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
            assert_within_tolerance(
                arrays.read(output, dtype), expected, Tolerance(atol=atol, rtol=rtol)
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
        assert_within_tolerance(
            arrays.read(output, dtype), expected, Tolerance(atol=atol, rtol=rtol)
        )
    finally:
        execution.close()
    assert not execution._graphs and execution._workspace is None
    churn = [
        torch.empty((batch, heads, capacity, dim), device="cuda") for _ in range(8)
    ]
    torch.cuda.synchronize()
    assert churn
