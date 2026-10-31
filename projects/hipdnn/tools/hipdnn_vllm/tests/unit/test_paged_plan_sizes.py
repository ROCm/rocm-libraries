# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""P2a/P2b plan sizes: eager buckets, capture sizes, covering and prefix views."""

import types

import pytest

torch = pytest.importorskip("torch")

from hipdnn_vllm.attention.metadata import PagedBounds, PagedBuffers  # noqa: E402
from hipdnn_vllm.attention.paged import (  # noqa: E402
    _Plans,
    bound_size,
    capture_size,
    eager_size,
    graph_kwargs,
    next_pow2,
)
from hipdnn_vllm.attention.routing import P2A, P2B  # noqa: E402

BOUNDS = PagedBounds(max_num_seqs=48, max_decode_query_len=1, max_query_len=2048, max_model_len=4096)
SPEC = PagedBounds(max_num_seqs=48, max_decode_query_len=4, max_query_len=2048, max_model_len=4096)


def _descriptor(num_tokens, num_reqs, uniform):
    return types.SimpleNamespace(num_tokens=num_tokens, num_reqs=num_reqs, uniform=uniform)


@pytest.mark.parametrize("n,want", [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (33, 64), (1024, 1024)])
def test_next_pow2(n, want):
    assert next_pow2(n) == want


def test_bound_sizes():
    assert bound_size(P2A, BOUNDS) == (48, 1)
    assert bound_size(P2B, BOUNDS) == (48, 2048)
    assert bound_size(P2A, SPEC) == (48, 4)


@pytest.mark.parametrize(
    "path,bounds,num_reqs,max_q,want",
    [
        (P2A, BOUNDS, 1, 1, (1, 1)),
        (P2A, BOUNDS, 5, 1, (8, 1)),
        (P2A, BOUNDS, 33, 1, (48, 1)),  # capped at max_num_seqs
        (P2A, SPEC, 6, 4, (8, 4)),  # spec decode keeps the decode query length
        (P2A, SPEC, 6, 1, (8, 4)),
        (P2B, BOUNDS, 6, 77, (8, 128)),
        (P2B, BOUNDS, 1, 333, (1, 512)),
        (P2B, BOUNDS, 2, 2048, (2, 2048)),
        (P2B, BOUNDS, 48, 1500, (48, 2048)),  # 2048 is the bound
    ],
)
def test_eager_size(path, bounds, num_reqs, max_q, want):
    b, sq = eager_size(path, bounds, num_reqs, max_q)
    assert (b, sq) == want
    assert b >= min(num_reqs, bounds.max_num_seqs) and sq >= max_q


def test_eager_sizes_are_bounded():
    sizes = {eager_size(P2B, BOUNDS, n, q) for n in range(1, 49) for q in range(1, 2049, 7)}
    # log2(48) + 2 batch buckets times log2(2048) + 1 query buckets
    assert len({b for b, _ in sizes}) <= 7 and len({q for _, q in sizes}) <= 12
    assert len({eager_size(P2A, BOUNDS, n, 1) for n in range(1, 49)}) <= 7


def test_capture_size_follows_the_descriptor():
    # Uniform decode graph of 24 tokens at decode length 1: 24 requests.
    assert capture_size(P2A, BOUNDS, _descriptor(24, 24, True), 24, 1) == (24, 1)
    # Spec decode (k+1 = 4): 96 tokens = 24 requests of 4.
    assert capture_size(P2A, SPEC, _descriptor(96, 24, True), 24, 4) == (24, 4)
    # P2b holding a uniform-decode graph (P2a declined) keeps the decode length.
    assert capture_size(P2B, SPEC, _descriptor(96, 24, True), 24, 4) == (24, 4)
    # Mixed FULL graph of 64 tokens: up to 48 requests, queries up to 64 tokens.
    assert capture_size(P2B, BOUNDS, _descriptor(64, 48, False), 16, 4) == (48, 64)
    # The capture batch itself always fits.
    assert capture_size(P2B, BOUNDS, _descriptor(64, 4, False), 6, 70) == (6, 70)


@pytest.mark.parametrize("descriptor", [None, _descriptor(64, None, False)])
def test_capture_size_without_descriptor_is_the_bound(descriptor):
    assert capture_size(P2A, BOUNDS, descriptor, 3, 1) == (48, 1)
    assert capture_size(P2B, BOUNDS, descriptor, 3, 7) == (48, 2048)


def test_capture_size_from_v2_runner_metadata():
    """vLLM's V2 model runner captures FULL graphs with no request-bounding
    descriptor and builds the capture batch from the graph's bounds (num_reqs,
    promised max_query_len)."""
    # Uniform decode graph: 24 requests of 1 (or 4 with spec decode).
    assert capture_size(P2A, BOUNDS, None, 24, 1, from_metadata=True) == (24, 1)
    assert capture_size(P2A, SPEC, None, 24, 4, from_metadata=True) == (24, 4)
    # Mixed FULL graph of 64 tokens: min(64, 48) requests, queries up to 64.
    assert capture_size(P2B, BOUNDS, None, 48, 64, from_metadata=True) == (48, 64)
    assert capture_size(P2B, BOUNDS, None, 48, 4096, from_metadata=True) == (48, 2048)
    # set_forward_context fills in BatchDescriptor(num_tokens) (no num_reqs)
    # when the V2 runner passes none: the metadata still sizes the graph.
    assert capture_size(P2A, BOUNDS, _descriptor(24, None, False), 24, 1, from_metadata=True) == (24, 1)
    assert capture_size(P2B, BOUNDS, _descriptor(64, None, False), 48, 64, from_metadata=True) == (48, 64)


def test_v2_full_capture_forward_context_bounds_no_requests():
    """Mirror of vLLM's set_forward_context under a V2 FULL capture (no batch
    descriptor passed): the descriptor it fills in bounds no requests, so the
    capture batch's own class and size hold; a V1 descriptor still decides."""
    pytest.importorskip("vllm")
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context

    from hipdnn_vllm.attention.backend import _capture_descriptor, _captured_class
    from hipdnn_vllm.attention.metadata import BatchClass

    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=False),
        compilation_config=types.SimpleNamespace(fast_moe_cold_start=False, static_forward_context={}),
    )
    with set_forward_context({}, config, num_tokens=24, cudagraph_runtime_mode=CUDAGraphMode.FULL):
        filled = get_forward_context().batch_descriptor
        assert filled is not None and filled.num_tokens == 24 and filled.num_reqs is None
        assert _capture_descriptor() is None
        assert _captured_class(BatchClass.UNIFORM_DECODE, True) is BatchClass.UNIFORM_DECODE
        assert _captured_class(BatchClass.UNIFORM_DECODE, False) is BatchClass.MIXED
    v1 = BatchDescriptor(num_tokens=24, num_reqs=24, uniform=True)
    with set_forward_context({}, config, num_tokens=24, cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=v1):
        assert _capture_descriptor() is v1
        assert _captured_class(BatchClass.UNIFORM_DECODE, False) is BatchClass.UNIFORM_DECODE


def test_cover_picks_the_smallest_built_plan():
    plans = _Plans(container=None)
    plans.built[P2B] = {(48, 2048), (8, 128), (16, 64), (8, 2048)}
    assert plans.cover(P2B, 8, 100) == (8, 128)
    assert plans.cover(P2B, 9, 64) == (16, 64)
    assert plans.cover(P2B, 9, 65) == (48, 2048)
    assert plans.cover(P2B, 49, 1) is None
    assert plans.cover(P2A, 1, 1) is None


def test_graph_kwargs_binds_a_prefix_of_the_persistent_views():
    b, m = 48, 6
    bounds = PagedBounds(max_num_seqs=b, max_decode_query_len=1, max_query_len=512, max_model_len=96)
    qsl = torch.arange(b + 1, dtype=torch.int32)
    seq_lens = torch.ones(b, dtype=torch.int32)
    table = torch.zeros(b, m, dtype=torch.int32)
    n = 5
    views = PagedBuffers(bounds, "cpu").refresh(qsl[: n + 1], seq_lens[:n], table[:n])
    assert views.rows == n
    full = graph_kwargs(P2B, views, 16, -1)
    assert full["max_seq_len_q"] == 512 and full["seq_len_q"].shape == (b,)
    kw = graph_kwargs(P2B, views, 16, -1, (8, 64))
    assert kw["max_seq_len_q"] == 64 and kw["max_seq_len_kv"] == 96
    assert kw["ragged_offset_q"].shape == (9,) and kw["seq_len_q"].shape == (8,)
    assert kw["seq_len_kv"].shape == (8,) and kw["page_table"].shape == (8, m)
    # Zero-copy prefixes of the same storage.
    for name in ("ragged_offset_q", "seq_len_q", "seq_len_kv", "page_table"):
        assert kw[name].data_ptr() == full[name].data_ptr(), name
    assert kw["seq_len_q"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def _vllm_config(mode_name, sizes, max_num_seqs, spec_tokens):
    from vllm.config import CUDAGraphMode

    cc = types.SimpleNamespace(
        cudagraph_mode=CUDAGraphMode[mode_name],
        cudagraph_capture_sizes=list(sizes),
        max_cudagraph_capture_size=max(sizes),
        compile_sizes=[],
        mode=None,
        splitting_ops=[],
        is_attention_compiled_piecewise=lambda: True,
    )
    return types.SimpleNamespace(
        compilation_config=cc,
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        num_speculative_tokens=spec_tokens,
        lora_config=None,
    )


@pytest.mark.parametrize(
    "mode_name,spec_tokens,max_num_seqs",
    [("FULL_AND_PIECEWISE", 0, 256), ("FULL_AND_PIECEWISE", 3, 64), ("FULL", 0, 48), ("FULL_DECODE_ONLY", 0, 32),
     ("PIECEWISE", 0, 256), ("NONE", 0, 256)],
)  # fmt: skip
def test_capture_sizes_match_the_vllm_dispatcher(mode_name, spec_tokens, max_num_seqs):
    """capture_sizes() lists exactly the FULL graphs vLLM's dispatcher keys."""
    pytest.importorskip("vllm")
    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    from hipdnn_vllm.attention.backend import capture_sizes

    q = spec_tokens + 1
    sizes = [x * q for x in [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))]
    config = _vllm_config(mode_name, sizes, max_num_seqs, spec_tokens)
    bounds = PagedBounds(max_num_seqs=max_num_seqs, max_decode_query_len=q, max_query_len=2048, max_model_len=4096)
    ours = capture_sizes(config, bounds)
    mode = config.compilation_config.cudagraph_mode
    if mode == CUDAGraphMode.NONE:
        assert ours == ()
        return
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(mode, q)
    theirs = {
        (d.uniform, d.num_reqs, q if d.uniform else min(d.num_tokens, bounds.max_query_len))
        for d in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
    }
    assert set(ours) == theirs
    assert len(ours) == len(set(ours))


@pytest.mark.parametrize(
    "mode_name,spec_tokens,max_num_seqs",
    [("FULL_AND_PIECEWISE", 0, 256), ("FULL_AND_PIECEWISE", 3, 64), ("FULL_AND_PIECEWISE", 2, 64), ("FULL", 0, 48),
     ("FULL_DECODE_ONLY", 0, 32), ("PIECEWISE", 0, 256)],
)  # fmt: skip
def test_capture_sizes_match_the_v2_model_runner(monkeypatch, mode_name, spec_tokens, max_num_seqs):
    """capture_sizes() lists exactly the FULL graphs vLLM's V2 model runner
    (CudaGraphManager) captures, as (uniform, num_reqs, promised max_query_len)."""
    pytest.importorskip("vllm")
    cg = pytest.importorskip("vllm.v1.worker.gpu.cudagraph_utils")
    from vllm.config import CUDAGraphMode

    from hipdnn_vllm.attention.backend import capture_sizes

    q = spec_tokens + 1
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    config = _vllm_config(mode_name, sizes, max_num_seqs, spec_tokens)
    config.use_v2_model_runner = True
    config.speculative_config = None
    config.parallel_config = types.SimpleNamespace(tensor_parallel_size=1)
    config.cache_config = types.SimpleNamespace(use_kda_recoverssm=False)
    bounds = PagedBounds(max_num_seqs=max_num_seqs, max_decode_query_len=q, max_query_len=2048, max_model_len=4096)
    monkeypatch.setattr(cg, "get_pp_group", lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None, raising=False)
    mode = config.compilation_config.cudagraph_mode
    manager = cg.CudaGraphManager(config, torch.device("cpu"), mode, q)
    theirs = {
        (d.uniform_token_count is not None, d.num_reqs,
         d.uniform_token_count or d.max_query_len or min(d.num_tokens, bounds.max_query_len))
        for d in manager._capture_descs.get(CUDAGraphMode.FULL, [])
    }  # fmt: skip
    assert set(capture_sizes(config, bounds)) == theirs
