# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""P2a (paged uniform decode) and P2b (ragged + paged) through the HIPDNN
attention impl, against the TRITON_ATTN impl on the same metadata and cache.

Batches are laid out the way vLLM's model runner lays them out: persistent
``query_start_loc`` / ``seq_lens`` / block-table buffers sized for
``max_num_seqs``, sliced to the (padded) batch, padded requests with
``q_len = 0`` and ``seq_len = 0``, a stale tail past the padded batch, a
shuffled block table, K/V interleaved in one TRITON_ATTN cache, and Q as a
strided view of a fused QKV projection.

On archs with a bundled paged engine (gfx942/gfx950 AttentionPagedUnified,
RDNA Gfx1151WmmaAttention) strict mode is on, so any fallback fails the test.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

DTYPES = [torch.bfloat16, torch.float16]
MAX_SEQS = 16
PAGED_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201", "gfx942", "gfx950")


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _tol(dtype):
    return dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=5e-3, rtol=5e-3)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    pytest.importorskip("vllm")
    for name in ("HIPDNN_VLLM_OPS", "HIPDNN_VLLM_STRICT"):
        monkeypatch.delenv(name, raising=False)
    if _arch() in PAGED_ARCHS:
        monkeypatch.setenv("HIPDNN_VLLM_STRICT", "1")


# ------------------------------------------------------------------ setup --


class Runner:
    """Persistent buffers like vLLM's GPU model runner, plus one KV cache."""

    def __init__(self, *, hq, hkv, d, bs, dtype, layout="LBHNC", num_blocks=96, max_blocks=24,
                 max_tokens=640, max_query_len=512, decode_len=1, window=None, seed=0):
        from hipdnn_vllm.attention.metadata import PagedBounds, PagedBuffers

        self.g = torch.Generator(device="cuda").manual_seed(seed)
        self.hq, self.hkv, self.d, self.bs, self.dtype = hq, hkv, d, bs, dtype
        self.window = window
        self.decode_len = decode_len
        dev = "cuda"
        if layout == "LBHNC":
            self.kv_cache = torch.empty(num_blocks, hkv, bs, 2 * d, dtype=dtype, device=dev)
        else:  # LBNHC: physically [blocks, block_size, heads, 2D], logically permuted
            self.kv_cache = torch.empty(num_blocks, bs, hkv, 2 * d, dtype=dtype, device=dev).transpose(1, 2)
        self.qsl = torch.zeros(MAX_SEQS + 1, dtype=torch.int32, device=dev)
        self.seq_lens = torch.zeros(MAX_SEQS, dtype=torch.int32, device=dev)
        self.table = torch.zeros(MAX_SEQS, max_blocks, dtype=torch.int32, device=dev)
        width = (hq + 2 * hkv) * d
        self.qkv = torch.empty(max_tokens, width, dtype=dtype, device=dev)
        self.query = self.qkv[:, : hq * d].view(max_tokens, hq, d)  # token stride (hq+2hkv)*d
        self.out = torch.full((max_tokens, hq, d), float("nan"), dtype=dtype, device=dev)
        self.ref = torch.full((max_tokens, hq, d), float("nan"), dtype=dtype, device=dev)
        bounds = PagedBounds(max_num_seqs=MAX_SEQS, max_decode_query_len=decode_len,
                             max_query_len=max_query_len, max_model_len=max_blocks * bs)
        self.paged = PagedBuffers(bounds, dev)
        self.layer = types.SimpleNamespace(
            _k_scale=torch.tensor(1.0, device=dev), _v_scale=torch.tensor(1.0, device=dev),
            _q_scale=torch.tensor(1.0, device=dev),
        )
        # TRITON_ATTN 3D-decode scratch, sized as its builder does.
        segs = 16
        self.segm_out = torch.empty(max_tokens, hq, segs, 1 << (d - 1).bit_length(), dtype=torch.float32, device=dev)
        self.segm_max = torch.empty(max_tokens, hq, segs, dtype=torch.float32, device=dev)
        self.segm_sum = torch.empty(max_tokens, hq, segs, dtype=torch.float32, device=dev)
        self.metadata = None
        # vLLM's layer.sinks ([hq] Parameter) for both impls, or None.
        self.sinks = None

    def fill(self, seq_q, seq_kv, pad_reqs=0, pad_tokens=None, block_base=0, max_query_len=None):
        """Write a batch into the persistent buffers (in place) and fresh
        random cache / query contents. Requests use shuffled distinct physical
        blocks from ``[block_base, num_blocks)``. Returns the TRITON_ATTN
        metadata with ``metadata.hipdnn`` attached, built like
        HipdnnAttentionMetadataBuilder.build. ``max_query_len`` overrides the
        metadata's (vLLM's V2 runner builds capture batches with the graph's
        promised bound)."""
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

        from hipdnn_vllm.attention.metadata import build_info

        assert all(k >= q for q, k in zip(seq_q, seq_kv))
        n_real = len(seq_q)
        n = n_real + pad_reqs
        g = self.g
        # In place: no float32 temporary the size of the cache.
        self.kv_cache.normal_(generator=g)
        self.qkv.copy_(torch.randn(self.qkv.shape, generator=g, device="cuda", dtype=torch.float32))
        self.out.fill_(float("nan"))
        self.ref.fill_(float("nan"))
        # Stale runner tail past the padded batch (dummy runs leave it).
        self.qsl.fill_(10_000)
        self.seq_lens.fill_(4_321)
        self.table.copy_(torch.randint(0, self.kv_cache.shape[0], self.table.shape, generator=g,
                                       device="cuda", dtype=torch.int32))
        starts = [0]
        for q in seq_q:
            starts.append(starts[-1] + q)
        starts += [starts[-1]] * pad_reqs
        self.qsl[: n + 1].copy_(torch.tensor(starts, dtype=torch.int32))
        self.seq_lens[:n].copy_(torch.tensor(list(seq_kv) + [0] * pad_reqs, dtype=torch.int32))
        # Shuffled, distinct physical blocks per request.
        perm = block_base + torch.randperm(self.kv_cache.shape[0] - block_base, generator=g,
                                           device="cuda").to(torch.int32)
        used = 0
        for i, k in enumerate(seq_kv):
            nb = -(-k // self.bs)
            self.table[i, :nb] = perm[used : used + nb]
            used += nb
        assert used <= perm.numel()
        if pad_tokens is None:
            # vLLM pads a uniform batch to num_reqs_padded * q_len tokens.
            pad_tokens = pad_reqs * seq_q[0] if len(set(seq_q)) == 1 else 0
        tokens = starts[-1] + pad_tokens
        self.n_real_tokens = starts[-1]
        cm = types.SimpleNamespace(
            num_reqs=n, num_actual_tokens=tokens, max_query_len=max_query_len or max(seq_q),
            max_seq_len=max(seq_kv), query_start_loc=self.qsl[: n + 1],
            query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
            seq_lens=self.seq_lens[:n], block_table_tensor=self.table[:n], causal=True,
        )
        if self.metadata is None:
            md = TritonAttentionMetadata(
                num_actual_tokens=tokens, max_query_len=cm.max_query_len, query_start_loc=cm.query_start_loc,
                max_seq_len=cm.max_seq_len, seq_lens=cm.seq_lens, block_table=cm.block_table_tensor,
                slot_mapping=torch.zeros(tokens, dtype=torch.int64, device="cuda"),
                seq_threshold_3D=128 // self.hkv, num_par_softmax_segments=16,
                softmax_segm_output=self.segm_out, softmax_segm_max=self.segm_max,
                softmax_segm_expsum=self.segm_sum, causal=True,
            )
            self.metadata = md
        md = self.metadata
        md.num_actual_tokens, md.max_query_len, md.max_seq_len = tokens, cm.max_query_len, cm.max_seq_len
        md.query_start_loc, md.seq_lens, md.block_table = cm.query_start_loc, cm.seq_lens, cm.block_table_tensor
        md.hipdnn = build_info(cm, self.decode_len, False, self.paged)
        return md

    def impl(self, ops=None):
        import os

        from vllm.v1.attention.backend import AttentionType

        from hipdnn_vllm.attention.backend import HipdnnAttentionImpl

        if ops is not None:
            os.environ["HIPDNN_VLLM_OPS"] = ops
        try:
            return HipdnnAttentionImpl(
                num_heads=self.hq, head_size=self.d, scale=self.d**-0.5, num_kv_heads=self.hkv,
                alibi_slopes=None, sliding_window=self.window, kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER, sinks=self.sinks,
            )
        finally:
            if ops is not None:
                del os.environ["HIPDNN_VLLM_OPS"]

    def triton(self):
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        return TritonAttentionImpl(
            num_heads=self.hq, head_size=self.d, scale=self.d**-0.5, num_kv_heads=self.hkv,
            alibi_slopes=None, sliding_window=self.window, kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER, sinks=self.sinks,
        )

    def forward(self, impl, out, md=None):
        md = md or self.metadata
        t = md.num_actual_tokens
        impl.forward(self.layer, self.query[:t], None, None, self.kv_cache, md, out[:t])

    def check(self):
        n = self.n_real_tokens
        assert torch.isfinite(self.out[:n]).all()
        torch.testing.assert_close(self.out[:n].float(), self.ref[:n].float(), **_tol(self.dtype))
        # Padding tokens past the real batch are never written.
        assert torch.isnan(self.out[n:]).all()


def _hits(op, route="hipdnn"):
    from hipdnn_vllm import stats

    return sum(r["count"] for r in stats.snapshot()["counters"] if r["op"] == op and r["route"] == route)


def _run_eager(r, seq_q, seq_kv, op, **fill):
    md = r.fill(seq_q, seq_kv, **fill)
    r.forward(r.triton(), r.ref)
    impl = r.impl()
    before = _hits(op)
    r.forward(impl, r.out)
    torch.cuda.synchronize()
    served = _hits(op) - before
    if _arch() in PAGED_ARCHS:
        assert served == 1, f"{op} fell back on {_arch()}"
    r.check()
    return impl, md


def _random_sinks(r, dtype=torch.bfloat16):
    """gpt-oss attention sinks as vLLM holds them: one learned logit per query
    head in a ``[hq]`` Parameter, here random with head 0 at -inf (no sink)."""
    s = torch.randn(r.hq, generator=r.g, device="cuda", dtype=torch.float32) * 2
    s[0] = float("-inf")
    r.sinks = torch.nn.Parameter(s.to(dtype), requires_grad=False)
    return r.sinks


# ------------------------------------------------------------------ eager --


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hq,hkv,d,bs", [(14, 2, 64, 16), (32, 8, 64, 32), (16, 8, 128, 16), (8, 8, 128, 64),
                                         (8, 1, 256, 16), (16, 4, 256, 32)])
def test_decode(C, dtype, hq, hkv, d, bs):
    r = Runner(hq=hq, hkv=hkv, d=d, bs=bs, dtype=dtype, num_blocks=128, max_blocks=24)
    impl, md = _run_eager(r, [1] * 5, [1, 17, 300, 64, 129], "attention.uniform_decode", pad_reqs=3, pad_tokens=3)
    assert md.hipdnn.batch_class.value == "uniform_decode"
    print(f"{_arch()} {dtype} hq{hq}/hkv{hkv} D{d} bs{bs}: p2a={impl._paged.engine('p2a')} "
          f"p2b={impl._paged.engine('p2b')}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_spec_decode_uniform_query_len(C, dtype):
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=dtype, decode_len=4)
    _run_eager(r, [4] * 6, [4, 20, 97, 200, 33, 5], "attention.uniform_decode", pad_reqs=2, pad_tokens=8)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["LBHNC", "LBNHC"])
@pytest.mark.parametrize("window", [None, 48])
def test_mixed_prefill_and_decode(C, dtype, layout, window):
    r = Runner(hq=14, hkv=2, d=64, bs=16, dtype=dtype, layout=layout, window=window)
    # decodes, a whole prompt, and chunked-prefill chunks with cached context
    _run_eager(r, [1, 77, 1, 31, 128, 1], [300, 77, 9, 120, 290, 1], "attention.mixed", pad_tokens=5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hq,hkv,d,bs", [(32, 8, 64, 16), (16, 8, 128, 32), (8, 2, 256, 16)])
def test_chunked_prefill_with_context(C, dtype, hq, hkv, d, bs):
    r = Runner(hq=hq, hkv=hkv, d=d, bs=bs, dtype=dtype, num_blocks=160)
    _run_eager(r, [200, 64, 3], [380, 64, 3], "attention.mixed")


@pytest.mark.parametrize("dtype", DTYPES)
def test_prefill_without_context(C, dtype):
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=dtype)
    _run_eager(r, [333], [333], "attention.prefill_no_context")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "seq_q,seq_kv,op",
    [([1] * 4, [1, 300, 2049, 64], "attention.uniform_decode"),
     ([1, 77, 31, 1], [300, 77, 120, 2049], "attention.mixed")],
)
def test_kv_cache_above_4gib(C, dtype, seq_q, seq_kv, op):
    """Every page the batch reads starts past byte 2^32 of a KV cache larger
    than 4 GiB, so a 32-bit page or element offset reads the wrong pages.
    Both paged engines, AttentionPagedUnified (CDNA) and Gfx1151WmmaAttention
    (RDNA), address K/V pages with 64-bit offsets and must serve it."""
    hq, hkv, d, bs = 16, 8, 128, 16
    block_bytes = hkv * bs * 2 * d * torch.empty(0, dtype=dtype).element_size()
    first = -(-(1 << 32) // block_bytes)  # first block whose byte offset is >= 2^32
    num_blocks = first + 512
    free, _ = torch.cuda.mem_get_info()
    need = num_blocks * block_bytes + (1 << 30)
    if free < need:
        pytest.skip(f"needs {need >> 20} MiB free device memory, {free >> 20} MiB available")
    r = Runner(hq=hq, hkv=hkv, d=d, bs=bs, dtype=dtype, num_blocks=num_blocks, max_blocks=160)
    _run_eager(r, seq_q, seq_kv, op, block_base=first)
    for i, k in enumerate(seq_kv):
        assert int(r.table[i, : -(-k // bs)].min()) >= first


def test_attention_disabled_is_triton(C):
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=torch.bfloat16)
    r.fill([1, 1], [40, 9])
    r.forward(r.triton(), r.ref)
    r.forward(r.impl(ops="rms_norm"), r.out)
    torch.cuda.synchronize()
    torch.testing.assert_close(r.out[:2], r.ref[:2], rtol=0, atol=0)


def test_prefiltered_layer_never_builds(C, monkeypatch):
    monkeypatch.delenv("HIPDNN_VLLM_STRICT", raising=False)
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=torch.bfloat16)
    r.fill([1, 3], [40, 9])
    impl = r.impl()
    impl._layer_static = dataclasses.replace(impl._layer_static, logits_soft_cap=30.0)
    impl._routes.clear()
    r.forward(r.triton(), r.ref)
    r.forward(impl, r.out)
    torch.cuda.synchronize()
    assert not impl._paged.prepared
    torch.testing.assert_close(r.out[:4], r.ref[:4], rtol=0, atol=0)


# ------------------------------------------------------------------ sinks --

SINK_BATCHES = [
    pytest.param([1] * 5, [1, 17, 300, 64, 129], "attention.uniform_decode", dict(pad_reqs=3, pad_tokens=3),
                 id="p2a"),
    pytest.param([1, 77, 1, 31, 128, 1], [300, 77, 9, 120, 290, 1], "attention.mixed", dict(pad_tokens=5),
                 id="p2b"),
]


@pytest.mark.parametrize("sink_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("seq_q,seq_kv,op,fill", SINK_BATCHES)
def test_sinks(C, monkeypatch, sink_dtype, window, seq_q, seq_kv, op, fill):
    """gpt-oss attention sinks (GQA 8, D64, full and sliding-window layers)
    through P2a / P2b against TRITON_ATTN with the same sinks. The graph
    carries the sink_token, so where a paged engine claims it the call runs on
    hipDNN (strict mode); elsewhere it falls back with the engine's decline,
    never a sinks pre-filter. A bf16 sink is bound zero-copy; an FP32 one is
    converted once to the Q dtype and cached on the impl."""
    from hipdnn_vllm import stats

    r = Runner(hq=16, hkv=2, d=64, bs=16, dtype=torch.bfloat16, num_blocks=128, window=window)
    sinks = _random_sinks(r, sink_dtype)
    if _arch() in PAGED_ARCHS:
        impl, _ = _run_eager(r, seq_q, seq_kv, op, **fill)
    else:
        monkeypatch.delenv("HIPDNN_VLLM_STRICT", raising=False)
        r.fill(seq_q, seq_kv, **fill)
        r.forward(r.triton(), r.ref)
        impl = r.impl()
        stats.reset()
        r.forward(impl, r.out)
        torch.cuda.synchronize()
        reasons = {x["reason"] for x in stats.snapshot()["counters"] if x["op"] == op}
        assert _hits(op, "fallback") == 1 and "UNSUPPORTED_FEATURE:sinks" not in reasons, reasons
        r.check()
    bound = impl._paged._sink
    assert bound.shape == (1, r.hq, 1, 1) and bound.stride() == (r.hq, 1, 1, 1)
    assert bound.dtype == torch.bfloat16
    if sink_dtype == torch.bfloat16:
        assert bound.data_ptr() == sinks.data_ptr()
    else:
        assert impl._sink(torch.bfloat16) is bound
    paths = {p: impl._paged.engine(p) or impl._paged.status(p).reason for p in ("p2a", "p2b")}
    print(f"{_arch()} sinks {sink_dtype} window={window} {op}: {paths}")


def test_sinks_decline_falls_back_to_triton(C, monkeypatch):
    """A sink graph no engine takes falls back to TRITON_ATTN exactly as any
    other decline: the build-stage reason is recorded (not a sinks pre-filter)
    and the output is TRITON_ATTN's, bit for bit. The decline is forced with an
    FP32 sink_token, which the rocKE engines (whose kernels load the sink in
    the Q dtype) decline."""
    from hipdnn_vllm import stats
    from hipdnn_vllm.attention.routing import P2A, P2B

    monkeypatch.delenv("HIPDNN_VLLM_STRICT", raising=False)
    r = Runner(hq=16, hkv=2, d=64, bs=16, dtype=torch.bfloat16)
    sinks = _random_sinks(r)
    r.fill([1, 3], [40, 9])
    r.forward(r.triton(), r.ref)
    impl = r.impl()
    float_sink = sinks.detach().float().view(1, r.hq, 1, 1)
    monkeypatch.setattr(impl, "_sink", lambda dtype: float_sink)
    stats.reset()
    r.forward(impl, r.out)
    torch.cuda.synchronize()
    assert impl._paged.prepared
    assert impl._paged.status(P2A) is not None and impl._paged.status(P2B) is not None
    reasons = {x["reason"] for x in stats.snapshot()["counters"] if x["op"] == "attention.mixed"}
    assert _hits("attention.mixed") == 0 and _hits("attention.mixed", "fallback") == 1, reasons
    assert "UNSUPPORTED_FEATURE:sinks" not in reasons
    if _arch() in PAGED_ARCHS:
        assert reasons == {"NO_ENGINE"}, reasons
    torch.testing.assert_close(r.out[:4], r.ref[:4], rtol=0, atol=0)


# ------------------------------------------------------- capture / replay --


def _capture(r, impl, descriptor_uniform):
    """Capture one layer call as vLLM's V1 runner does (a batch descriptor),
    or, with ``descriptor_uniform=None``, as its V2 runner captures FULL
    graphs: no descriptor, so ``set_forward_context`` fills in
    ``BatchDescriptor(num_tokens)``."""
    import types

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context

    md = r.metadata
    descriptor = None
    if descriptor_uniform is not None:
        descriptor = BatchDescriptor(num_tokens=md.num_actual_tokens, num_reqs=md.hipdnn.num_reqs,
                                     uniform=descriptor_uniform)
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=False),
        compilation_config=types.SimpleNamespace(fast_moe_cold_start=False, static_forward_context={}),
    )
    ctx = set_forward_context({}, config, num_tokens=md.num_actual_tokens, cudagraph_runtime_mode=CUDAGraphMode.FULL,
                              batch_descriptor=descriptor)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with ctx, torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        r.forward(impl, r.out)
    torch.cuda.current_stream().wait_stream(s)
    return graph


def _replay_matches(r, graph, seq_q, seq_kv, **fill):
    """Refill the persistent buffers (same padded shape), rebind as build()
    would, replay the graph, and compare with eager TRITON_ATTN."""
    r.fill(seq_q, seq_kv, **fill)
    graph.replay()
    r.forward(r.triton(), r.ref)
    torch.cuda.synchronize()
    r.check()


@pytest.mark.parametrize("dtype", DTYPES)
def test_capture_replay_uniform_decode(C, dtype):
    from hipdnn_vllm.attention.routing import P2A

    r = Runner(hq=14, hkv=2, d=64, bs=16, dtype=dtype, num_blocks=128)
    impl, _ = _run_eager(r, [1] * 6, [5, 60, 1, 200, 31, 8], "attention.uniform_decode", pad_reqs=2)
    # TRITON_ATTN's capture metadata sets seq_lens to 1 (build_for_cudagraph_capture).
    r.fill([1] * 6, [1] * 6, pad_reqs=2)
    graph = _capture(r, impl, descriptor_uniform=True)
    captured = [route.path for (_, cap), route in impl._routes.items() if cap]
    assert captured and all(path == P2A for path in captured), captured
    _replay_matches(r, graph, [1] * 6, [300, 2, 17, 129, 64, 1], pad_reqs=2)
    _replay_matches(r, graph, [1] * 4, [9, 250, 33, 1], pad_reqs=4)  # fewer live requests


@pytest.mark.parametrize("dtype", DTYPES)
def test_capture_replay_mixed_graph(C, dtype):
    """A FULL graph captured for a non-uniform descriptor replays any batch
    with the same token count through P2b."""
    from hipdnn_vllm.attention.routing import P2B

    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=dtype, num_blocks=128)
    impl, _ = _run_eager(r, [1, 40, 1, 22], [70, 40, 3, 90], "attention.mixed")
    r.fill([16] * 4, [16] * 4)  # dummy capture batch: 64 tokens
    graph = _capture(r, impl, descriptor_uniform=False)
    captured = [route.path for (_, cap), route in impl._routes.items() if cap]
    assert captured and all(path == P2B for path in captured), captured
    _replay_matches(r, graph, [1, 60, 1, 2], [300, 60, 41, 7])
    _replay_matches(r, graph, [33, 31], [33, 200], pad_reqs=2)


def test_capture_replay_with_sinks(C):
    """The sink_token bound at capture (here the converted copy of FP32 sinks
    kept on the impl) stays valid for every replay."""
    from hipdnn_vllm.attention.routing import P2A

    r = Runner(hq=16, hkv=2, d=64, bs=16, dtype=torch.bfloat16, num_blocks=128)
    _random_sinks(r, torch.float32)
    impl, _ = _run_eager(r, [1] * 6, [5, 60, 1, 200, 31, 8], "attention.uniform_decode", pad_reqs=2)
    r.fill([1] * 6, [1] * 6, pad_reqs=2)
    graph = _capture(r, impl, descriptor_uniform=True)
    captured = [route.path for (_, cap), route in impl._routes.items() if cap]
    assert captured and all(path == P2A for path in captured), captured
    _replay_matches(r, graph, [1] * 6, [300, 2, 17, 129, 64, 1], pad_reqs=2)
    _replay_matches(r, graph, [1] * 4, [9, 250, 33, 1], pad_reqs=4)


def test_capture_without_eager_call_keeps_triton(C, monkeypatch):
    """A layer that never ran eagerly builds nothing under capture: the graph
    holds TRITON_ATTN (capture-invariant fallback)."""
    monkeypatch.delenv("HIPDNN_VLLM_STRICT", raising=False)
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=torch.bfloat16)
    r.fill([1] * 4, [10, 20, 30, 40])
    r.forward(r.triton(), r.ref)  # Triton JIT before capture
    impl = r.impl()
    before = _hits("attention.uniform_decode", "fallback")
    graph = _capture(r, impl, descriptor_uniform=True)
    assert not impl._paged.prepared
    assert _hits("attention.uniform_decode", "fallback") == before + 1
    _replay_matches(r, graph, [1] * 4, [7, 120, 3, 64])


# ------------------------------------------------------------- plan sizes --


@pytest.fixture
def sizes_used(monkeypatch):
    """Every (path, size) the paged path binds, in call order; a run's own
    launch is the last entry."""
    from hipdnn_vllm.attention import paged as paged_mod

    used = []
    real = paged_mod.graph_kwargs

    def spy(path, p, block_size, left_bound, size=None, sink=None):
        used.append((path, size))
        return real(path, p, block_size, left_bound, size, sink)

    monkeypatch.setattr(paged_mod, "graph_kwargs", spy)
    return used


def test_eager_calls_run_power_of_two_buckets(C, sizes_used):
    """Each eager call runs the plan of its (B, max_seq_len_q) bucket, not the
    upper bounds (B = 16, max_seq_len_q = 512 here); results still match."""
    from hipdnn_vllm import stats
    from hipdnn_vllm.attention.routing import P2A, P2B

    if _arch() not in PAGED_ARCHS:
        pytest.skip("no paged engine on this arch")
    stats.reset()
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=torch.bfloat16, num_blocks=160)
    impl, _ = _run_eager(r, [1] * 3, [40, 9, 100], "attention.uniform_decode")
    assert sizes_used[-1] == (P2A, (4, 1))
    cases = [
        ([1, 77, 3], [90, 77, 50], (4, 128)),
        ([200, 1], [380, 64], (2, 256)),
        ([5] * 9, [40] * 9, (16, 8)),
        ([1, 2], [64, 2], (2, 2)),
    ]
    for seq_q, seq_kv, want in cases:
        r.fill(seq_q, seq_kv)
        r.forward(r.triton(), r.ref)
        r.forward(impl, r.out)
        torch.cuda.synchronize()
        r.check()
        assert sizes_used[-1] == (P2B, want), (seq_q, sizes_used[-1])
    assert impl._paged.plan_sizes(P2A) == [(4, 1), (16, 1)]
    assert impl._paged.plan_sizes(P2B) == sorted([(2, 2), (2, 256), (4, 128), (16, 8), (16, 512)])
    plans = stats.snapshot()["attention_plans"]
    origins = {(s["path"], s["batch"], s["max_seq_len_q"]): s["origin"] for s in plans["sizes"]}
    assert origins[(P2A, 16, 1)] == "bound" and origins[(P2B, 16, 512)] == "bound"
    assert origins[(P2A, 4, 1)] == "eager" and origins[(P2B, 4, 128)] == "eager"
    assert plans["count"] == len(origins) == 7


@pytest.mark.parametrize("dtype", DTYPES)
def test_capture_holds_the_prebuilt_capture_size_plan(C, dtype, sizes_used):
    """The first eager call builds the plan of every FULL graph vLLM captures;
    a capture then holds the plan of its descriptor's padded request count,
    builds nothing, and replays batches with fewer live requests."""
    from hipdnn_vllm.attention.routing import P2A, P2B

    if _arch() not in PAGED_ARCHS:
        pytest.skip("no paged engine on this arch")
    r = Runner(hq=14, hkv=2, d=64, bs=16, dtype=dtype, num_blocks=128)
    r.paged.capture_sizes = ((False, 16, 64), (True, 4, 1), (True, 12, 1))
    impl, _ = _run_eager(r, [1] * 3, [5, 60, 200], "attention.uniform_decode")
    assert impl._paged.plan_sizes(P2A) == [(4, 1), (12, 1), (16, 1)]
    assert impl._paged.plan_sizes(P2B) == [(16, 64), (16, 512)]
    builds = C.stats_snapshot()["runtime"]["plan_builds"]
    r.fill([1] * 10, [1] * 10, pad_reqs=2)  # the capture batch: 12 padded requests
    graph = _capture(r, impl, descriptor_uniform=True)
    assert sizes_used[-1] == (P2A, (12, 1))
    assert C.stats_snapshot()["runtime"]["plan_builds"] == builds
    _replay_matches(r, graph, [1] * 10, [300, 2, 17, 129, 64, 1, 33, 90, 5, 250], pad_reqs=2)
    _replay_matches(r, graph, [1] * 7, [9, 250, 33, 1, 64, 2, 120], pad_reqs=5)
    # A capture size with no plan of its own uses the smallest built plan that holds it.
    r.fill([1] * 5, [1] * 5, pad_reqs=1)
    graph = _capture(r, impl, descriptor_uniform=True)
    assert sizes_used[-1] == (P2A, (12, 1))
    _replay_matches(r, graph, [1] * 5, [77, 3, 140, 1, 20], pad_reqs=1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_v2_runner_capture_sizes_from_the_capture_batch(C, dtype, sizes_used):
    """vLLM's V2 model runner captures FULL graphs with no request-bounding descriptor and
    builds the capture batch from the graph's bounds: num_reqs and a promised
    max_query_len. A uniform-decode graph then holds P2a of that request count,
    a mixed graph P2b of (num_reqs, promised max_query_len)."""
    from hipdnn_vllm.attention.routing import P2A, P2B

    if _arch() not in PAGED_ARCHS:
        pytest.skip("no paged engine on this arch")
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=dtype, num_blocks=128)
    r.paged.capture_from_metadata = True
    r.paged.capture_sizes = ((False, 16, 64), (True, 12, 1))
    impl, _ = _run_eager(r, [1] * 3, [5, 60, 200], "attention.uniform_decode")
    builds = C.stats_snapshot()["runtime"]["plan_builds"]
    r.fill([1] * 12, [1] * 12)  # uniform graph: 12 requests of 1
    graph = _capture(r, impl, descriptor_uniform=None)
    assert sizes_used[-1] == (P2A, (12, 1))
    _replay_matches(r, graph, [1] * 9, [300, 2, 17, 129, 64, 1, 33, 90, 5], pad_reqs=3)
    r.fill([4] * 16, [4] * 16, max_query_len=64)  # mixed graph: 64 tokens, 16 requests
    graph = _capture(r, impl, descriptor_uniform=None)
    assert sizes_used[-1] == (P2B, (16, 64))
    assert C.stats_snapshot()["runtime"]["plan_builds"] == builds
    _replay_matches(r, graph, [1, 60, 1, 2], [300, 60, 41, 7], pad_reqs=12, pad_tokens=0)


def test_a_new_kv_cache_is_prepared_again(C, monkeypatch):
    """vLLM profiles with a small KV cache, then allocates the real one. The
    first eager call on the new cache probes and builds its plans again; a
    capture before that keeps TRITON_ATTN in the graph."""
    from hipdnn_vllm.attention.routing import P2A

    monkeypatch.delenv("HIPDNN_VLLM_STRICT", raising=False)
    r = Runner(hq=16, hkv=8, d=128, bs=16, dtype=torch.bfloat16, num_blocks=64)
    r.fill([1] * 4, [10, 20, 30, 40])
    impl = r.impl()
    r.forward(impl, r.out)
    torch.cuda.synchronize()
    assert impl._paged.ready(r.kv_cache)
    r.kv_cache = torch.empty(160, r.hkv, r.bs, 2 * r.d, dtype=r.dtype, device="cuda")
    assert not impl._paged.ready(r.kv_cache)
    r.fill([1] * 4, [1] * 4)
    r.forward(r.triton(), r.ref)  # Triton JIT before capture
    before = _hits("attention.uniform_decode", "fallback")
    graph = _capture(r, impl, descriptor_uniform=True)
    assert _hits("attention.uniform_decode", "fallback") == before + 1
    _replay_matches(r, graph, [1] * 4, [7, 120, 3, 64])
    served = _hits("attention.uniform_decode")
    r.fill([1] * 4, [70, 2, 300, 15])
    r.forward(r.triton(), r.ref)
    r.forward(impl, r.out)
    torch.cuda.synchronize()
    r.check()
    assert impl._paged.ready(r.kv_cache)
    if _arch() in PAGED_ARCHS:
        assert _hits("attention.uniform_decode") == served + 1
        assert impl._paged.status(P2A) is None


# ------------------------------------------------------------------ probes --


def test_encoder_only_varlen_probe(C):
    """P2b replaces P1e for encoder-only layers only if an engine accepts
    ragged Q with non-paged ragged K/V. Report what this arch's engines say."""
    t, hq, d = 48, 8, 64
    q = torch.randn(t, hq, d, dtype=torch.bfloat16, device="cuda")
    off = torch.tensor([0, 16, 48], dtype=torch.int32, device="cuda")
    lens = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
    reason, detail = C.sdpa_fwd_probe(q, q, q, torch.empty_like(q), d**-0.5, causal=False,
                                      ragged_offset_q=off, seq_len_q=lens, max_seq_len_q=32,
                                      ragged_offset_kv=off, seq_len_kv=lens, max_seq_len_kv=32)
    print(f"{_arch()} encoder-only varlen (ragged Q, ragged K/V): {reason or 'served by ' + detail}")
