# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.sdpa_fwd and the P1e encoder-only path, on GPU.

* dense graphs: output vs torch SDPA, over strided BSHD views of a fused QKV
  projection permuted to BHSD (the P1e recipe), GQA, masks and windows;
* expected routes per arch (the P1e engines);
* the paged and ragged graph descriptions and their probes;
* torch.compile(fullgraph=True) and HIP-graph capture with the fake impl;
* HipdnnAttentionImpl P1e vs the TRITON_ATTN fallback, strict mode.
"""

from __future__ import annotations

import os
import types

import pytest
import torch
import torch.nn.functional as F

DTYPES = [torch.bfloat16, torch.float16]


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _qkv(tokens, hq, hkv, d, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(tokens, (hq + 2 * hkv) * d, dtype=dtype, device="cuda", generator=g)
    q, k, v = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
    return q.view(tokens, hq, d), k.view(tokens, hkv, d), v.view(tokens, hkv, d)


def _bhsd(t, count, length):
    return t.view(count, length, t.shape[1], t.shape[2]).transpose(1, 2)


def _ref(q, k, v, scale, left=-1, right=-1, bottom_right=True):
    """Float32 reference over BHSD tensors with hipDNN band semantics."""
    hq, hkv = q.shape[1], k.shape[1]
    kf = k.float().repeat_interleave(hq // hkv, dim=1)
    vf = v.float().repeat_interleave(hq // hkv, dim=1)
    sq, skv = q.shape[2], k.shape[2]
    mask = None
    if left >= 0 or right >= 0:
        off = skv - sq if bottom_right else 0
        i = torch.arange(sq, device=q.device)[:, None] + off
        j = torch.arange(skv, device=q.device)[None, :]
        mask = torch.ones(sq, skv, dtype=torch.bool, device=q.device)
        if left >= 0:
            mask &= j >= i - left
        if right >= 0:
            mask &= j <= i + right
    return F.scaled_dot_product_attention(q.float(), kf, vf, attn_mask=mask, scale=scale)


def _tol(dtype):
    return dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=5e-3, rtol=5e-3)


def _run_or_skip(C, q, k, v, o, scale, **kw):
    reason, detail = C.sdpa_fwd_probe(q, k, v, o, scale, **kw)
    if reason:
        pytest.skip(f"hipDNN declines on {_arch()}: {reason} ({detail.splitlines()[0] if detail else ''})")
    torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, o, scale, **kw)
    return detail


# ---------------------------------------------------------------- dense ----


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 2)])
@pytest.mark.parametrize("count,length", [(1, 1), (1, 77), (3, 128), (2, 300)])
def test_dense_strided_views(C, dtype, d, hq, hkv, count, length):
    q, k, v = _qkv(count * length, hq, hkv, d, dtype)
    out = torch.full((count * length, hq, d), float("nan"), dtype=dtype, device="cuda")
    qv, kv, vv, ov = (_bhsd(t, count, length) for t in (q, k, v, out))
    scale = d**-0.5
    _run_or_skip(C, qv, kv, vv, ov, scale)
    torch.cuda.synchronize()
    ref = _ref(qv, kv, vv, scale)
    torch.testing.assert_close(ov.float(), ref, **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "mask",
    [
        dict(causal=True, bottom_right=False),
        dict(causal=True, bottom_right=True),
        dict(left_bound=31, right_bound=31),  # vLLM encoder window W=32: (W-1, W-1)
        dict(causal=True, left_bound=63),  # decoder sliding window W=64
    ],
)
def test_dense_masks(C, dtype, mask):
    count, length, hq, hkv, d = 2, 160, 4, 4, 64
    q, k, v = _qkv(count * length, hq, hkv, d, dtype, seed=3)
    out = torch.empty(count * length, hq, d, dtype=dtype, device="cuda")
    qv, kv, vv, ov = (_bhsd(t, count, length) for t in (q, k, v, out))
    _run_or_skip(C, qv, kv, vv, ov, 0.125, **mask)
    torch.cuda.synchronize()
    right = 0 if mask.get("causal") else mask.get("right_bound", -1)
    ref = _ref(qv, kv, vv, 0.125, mask.get("left_bound", -1), right, mask.get("bottom_right", True))
    torch.testing.assert_close(ov.float(), ref, **_tol(dtype))


def _expected_p1e_dtypes(d: int = 128) -> list[torch.dtype]:
    """Dtypes with a P1e engine for head size ``d`` on this arch.

    gfx942/gfx950: ASM_SDPA ships bf16 head-dim-128 forward kernels only
    (fmha_v3_fwd hd128) in TheRock 10.0.0. RDNA: the bundled rocKE
    Gfx1151WmmaAttention takes bf16 and fp16 at head widths in multiples of 8."""
    arch = _arch()
    if arch in ("gfx942", "gfx950"):
        return [torch.bfloat16] if d == 128 else []
    if arch.startswith("gfx11") or arch.startswith("gfx12"):
        return [torch.bfloat16, torch.float16]
    return []


@pytest.mark.parametrize("d", [64, 128])
def test_p1e_routes_to_hipdnn_on_this_arch(C, d):
    expected = _expected_p1e_dtypes(d)
    if not expected:
        pytest.skip(f"no P1e engine listed for {_arch()} D{d}")
    names = {f"0x{e['id'] & 0xFFFFFFFFFFFFFFFF:016x}": e["name"] for e in C.engines(0)}
    for dtype in expected:
        q, k, v = _qkv(256, 12, 12, d, dtype)
        out = torch.empty(256, 12, d, dtype=dtype, device="cuda")
        views = [_bhsd(t, 2, 128) for t in (q, k, v, out)]
        reason, detail = C.sdpa_fwd_probe(*views, d**-0.5)
        assert reason == "", f"{_arch()} {dtype} D{d}: {reason} {detail}"
        print(f"{_arch()} {dtype} D{d}: engine {names.get(detail, detail)}")


def _hip_peek_last_error() -> int:
    import ctypes

    with open("/proc/self/maps", encoding="utf-8") as f:
        path = next(line.split(None, 5)[5].strip() for line in f if "libamdhip64.so" in line)
    return ctypes.CDLL(path).hipPeekAtLastError()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [64, 128])
def test_plan_build_leaves_no_hip_error(C, dtype, d):
    """A plan build that fails inside a provider (e.g. the stock ASM_SDPA with
    no code objects: hipErrorFileNotFound) must not leave the HIP last error
    set, or torch's next launch check raises it from unrelated code."""
    torch.cuda.synchronize()
    assert _hip_peek_last_error() == 0
    # A shape no other test builds, so the plan is not already cached.
    q, k, v = _qkv(3 * 136, 6, 6, d, dtype, seed=11)
    out = torch.empty(3 * 136, 6, d, dtype=dtype, device="cuda")
    reason, detail = C.sdpa_fwd_probe(*(_bhsd(t, 3, 136) for t in (q, k, v, out)), d**-0.5)
    print(f"{_arch()} {dtype} D{d}: {reason or 'built'} {detail.splitlines()[0] if detail else ''}")
    assert _hip_peek_last_error() == 0
    torch.randn(16, device="cuda").sum().item()


def test_decline_is_raised_and_counted(C):
    q = torch.zeros(1, 2, 4, 64, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match=r"hipdnn_vllm decline \[sdpa_fwd\]: UNSUPPORTED_FEATURE:dtype"):
        torch.ops.hipdnn_vllm.sdpa_fwd(q, q, q, q.clone(), 1.0)
    rows = C.stats_snapshot()["counters"]
    assert any(
        r["op"] == "sdpa_fwd" and r["reason"] == "UNSUPPORTED_FEATURE:dtype" and r["count"] >= 1
        for r in rows
    )


def test_malformed_call_is_an_error(C):
    q = torch.zeros(1, 6, 4, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, 4, 4, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of Hkv"):
        torch.ops.hipdnn_vllm.sdpa_fwd(q, k, k, q.clone(), 1.0)


# ------------------------------------------------- paged / ragged graphs ----


def _paged_case(batch=3, hq=8, hkv=2, d=64, bs=16, max_kv=100, chunk=5, dtype=torch.bfloat16):
    blocks = -(-max_kv // bs)
    cache = torch.zeros(batch * blocks + 2, hkv, bs, 2 * d, dtype=dtype, device="cuda")
    k_cache, v_cache = cache.split(d, dim=-1)
    # vLLM's block table is wider than the bound needs; only the first columns are bound.
    table = torch.arange(batch * (blocks + 3), dtype=torch.int32, device="cuda").view(batch, blocks + 3)
    seq_kv = torch.tensor([max_kv, 1, 0][:batch], dtype=torch.int32, device="cuda")
    q, _, _ = _qkv(batch * chunk, hq, hkv, d, dtype)
    o = torch.empty(batch * chunk, hq, d, dtype=dtype, device="cuda")
    offsets = torch.arange(0, batch * chunk + 1, chunk, dtype=torch.int32, device="cuda")
    seq_q = torch.full((batch,), chunk, dtype=torch.int32, device="cuda")
    return types.SimpleNamespace(**locals())


def test_paged_decode_layout(C):
    c = _paged_case()
    q = c.q[: c.batch].unsqueeze(2)  # [B, Hq, 1, D]
    o = c.o[: c.batch].unsqueeze(2)
    lay = C.sdpa_fwd_layout(
        q, c.k_cache, c.v_cache, o, 0.125, causal=True, left_bound=127,
        page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv,
    )
    assert lay["decline"] == ""
    assert lay["layout"] == "dense_q_paged_kv"
    t = lay["tensors"]
    # K/V: zero-copy views of the interleaved TRITON_ATTN cache.
    assert t["k"]["dims"] == [c.cache.shape[0], c.hkv, c.bs, c.d]
    assert t["k"]["strides"] == list(c.cache.stride()[:3]) + [1]
    # One page table [B, 1, ceil(max_kv/bs), 1] over vLLM's [B, M] block table.
    assert t["page_table"]["dims"] == [c.batch, 1, c.blocks, 1]
    assert t["page_table"]["strides"] == [c.table.stride(0), c.table.stride(0), 1, 1]
    assert t["seq_len_kv"]["dims"] == [c.batch, 1, 1, 1]
    assert lay["right_bound"] == 0 and lay["left_bound"] == 127 and lay["bottom_right"]
    assert lay["max_seq_len_kv"] == c.max_kv
    assert lay["slots"] == [1, 2, 3, 4, 8, 9]


def test_ragged_paged_layout(C):
    c = _paged_case()
    kw = dict(
        causal=True, ragged_offset_q=c.offsets, seq_len_q=c.seq_q, max_seq_len_q=c.chunk,
        page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv,
    )
    lay = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, **kw)
    t = lay["tensors"]
    assert lay["layout"] == "ragged_q_paged_kv"
    token_stride = c.q.stride(0)
    assert t["q"]["dims"] == [c.batch, c.hq, c.chunk, c.d]
    assert t["q"]["strides"] == [c.chunk * token_stride, c.d, token_stride, 1]
    # Q and O share one ragged-offset tensor; each multiplier is its token stride.
    assert t["q"]["ragged_uid"] == t["o"]["ragged_uid"] == t["ragged_offset_q"]["uid"]
    assert t["q"]["ragged_multiplier"] == token_stride
    assert t["o"]["ragged_multiplier"] == c.o.stride(0)
    assert t["ragged_offset_q"]["dims"] == [c.batch + 1, 1, 1, 1]
    assert lay["slots"] == [1, 2, 3, 4, 5, 7, 8, 9]
    if not lay["frontend_ragged_multiplier"]:
        assert lay["decline"] == "UNSUPPORTED_FEATURE:ragged_multiplier"
    # The plan key ignores the step's token count: same B and bounds, fewer tokens.
    fewer = C.sdpa_fwd_layout(c.q[:7], c.k_cache, c.v_cache, c.o[:7], 0.125, **kw)
    assert fewer["key"] == lay["key"]


def test_ragged_kv_layout(C):
    c = _paged_case()
    _, k, v = _qkv(c.batch * c.chunk, c.hq, c.hkv, c.d, torch.bfloat16)
    lay = C.sdpa_fwd_layout(
        c.q, k, v, c.o, 0.125, ragged_offset_q=c.offsets, seq_len_q=c.seq_q,
        max_seq_len_q=c.chunk, ragged_offset_kv=c.offsets, seq_len_kv=c.seq_q,
        max_seq_len_kv=c.chunk,
    )
    t = lay["tensors"]
    assert lay["layout"] == "ragged"
    assert t["k"]["ragged_uid"] == t["v"]["ragged_uid"] == t["ragged_offset_kv"]["uid"]
    assert t["k"]["ragged_multiplier"] == k.stride(0)


def test_paged_and_ragged_probes_report_a_route(C):
    """The paged and ragged graphs build or decline cleanly: a route, never a crash."""
    c = _paged_case()
    paged = dict(page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv, causal=True)
    decode = C.sdpa_fwd_probe(c.q[: c.batch].unsqueeze(2), c.k_cache, c.v_cache,
                              c.o[: c.batch].unsqueeze(2), 0.125, **paged)
    mixed = C.sdpa_fwd_probe(c.q, c.k_cache, c.v_cache, c.o, 0.125, ragged_offset_q=c.offsets,
                             seq_len_q=c.seq_q, max_seq_len_q=c.chunk, **paged)
    for name, (reason, detail) in (("p2a", decode), ("p2b", mixed)):
        print(f"{_arch()} {name}: {reason or 'hipdnn'} {detail.splitlines()[0] if detail else ''}")
        if reason:
            from hipdnn_vllm.declines import parse_reason

            parse_reason(reason)  # inside the taxonomy


def test_malformed_paged_call(C):
    c = _paged_case()
    with pytest.raises(RuntimeError, match="page-table columns"):
        C.sdpa_fwd_layout(c.q[: c.batch].unsqueeze(2), c.k_cache, c.v_cache,
                          c.o[: c.batch].unsqueeze(2), 0.125, page_table=c.table[:, :2],
                          seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv)


def _decode_args(c):
    """Positional sdpa_fwd_layout arguments of a paged uniform-decode call."""
    return (c.q[: c.batch].unsqueeze(2), c.k_cache, c.v_cache, c.o[: c.batch].unsqueeze(2), 0.125)


def test_paged_zero_block_size_declines(C):
    """An empty paged container (block_size 0) declines instead of dividing by zero."""
    c = _paged_case()
    q, _, _, o, scale = _decode_args(c)
    lay = C.sdpa_fwd_layout(q, c.k_cache[:, :, :0], c.v_cache[:, :, :0], o, scale,
                            page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv)
    assert lay["decline"] == "UNSUPPORTED_FEATURE:empty"


@pytest.mark.parametrize("aux", ["page_table", "seq_len_kv", "ragged_offset_q", "seq_len_q"])
def test_aux_tensor_on_another_device_is_rejected(C, aux):
    """Aux tensors are bound as device pointers: a host tensor must be refused."""
    c = _paged_case()
    kw = dict(causal=True, ragged_offset_q=c.offsets, seq_len_q=c.seq_q, max_seq_len_q=c.chunk,
              page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv)
    C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, **kw)  # valid as given
    kw[aux] = kw[aux].cpu()
    with pytest.raises(RuntimeError, match=f"{aux} must be on q's device"):
        C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, **kw)


def test_paged_max_seq_len_kv_above_int32_is_rejected(C):
    """hipDNN holds the paged bound in INT32; a larger bound must not be truncated."""
    c = _paged_case()
    with pytest.raises(RuntimeError, match="exceeds the INT32 bound"):
        C.sdpa_fwd_layout(*_decode_args(c), page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=2**31)


def _ragged_paged_kwargs(c):
    return dict(causal=True, ragged_offset_q=c.offsets, seq_len_q=c.seq_q, max_seq_len_q=c.chunk,
                page_table=c.table, seq_len_kv=c.seq_kv, max_seq_len_kv=c.max_kv)


def test_sink_layout_and_plan_key(C):
    """The sink_token is vLLM's [Hq] sinks viewed zero-copy as [1, Hq, 1, 1]
    (strides [Hq, 1, 1, 1]), bound in the last slot. The plan key lists it, so
    sink and sink-free graphs, and sinks of different dtypes, never share a
    plan; the sink values do not enter the key."""
    c = _paged_case()
    kw = _ragged_paged_kwargs(c)
    sinks = torch.randn(c.hq, dtype=torch.bfloat16, device="cuda")
    plain = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, **kw)
    lay = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, sink=sinks.view(1, c.hq, 1, 1), **kw)
    t = lay["tensors"]["sink"]
    assert "sink" not in plain["tensors"]
    assert t["uid"] == 10 and t["dims"] == [1, c.hq, 1, 1] and t["strides"] == [c.hq, 1, 1, 1]
    assert t["dtype"] == lay["tensors"]["q"]["dtype"]
    assert lay["slots"] == plain["slots"] + [10]
    as_float = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125,
                                 sink=sinks.float().view(1, c.hq, 1, 1), **kw)
    assert len({plain["key"], lay["key"], as_float["key"]}) == 3
    other = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125,
                              sink=torch.zeros(1, c.hq, 1, 1, dtype=torch.bfloat16, device="cuda"), **kw)
    assert other["key"] == lay["key"]


def test_malformed_sink_is_rejected(C):
    c = _paged_case()
    kw = _ragged_paged_kwargs(c)
    sinks = torch.randn(c.hq, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=r"sink must be \[1, Hq, 1, 1\]"):
        C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, sink=sinks, **kw)  # vLLM's raw [Hq]
    with pytest.raises(RuntimeError, match=r"sink must be \[1, Hq, 1, 1\]"):
        C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, sink=sinks[:-1].view(1, -1, 1, 1), **kw)
    with pytest.raises(RuntimeError, match="sink must be on q's device"):
        C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125, sink=sinks.cpu().view(1, c.hq, 1, 1), **kw)
    lay = C.sdpa_fwd_layout(c.q, c.k_cache, c.v_cache, c.o, 0.125,
                            sink=sinks.double().view(1, c.hq, 1, 1), **kw)
    assert lay["decline"] == "UNSUPPORTED_FEATURE:sink_dtype"


# Archs whose bundled providers must serve paged/ragged SDPA: RDNA through
# Gfx1151WmmaAttention, CDNA through AttentionPagedUnified.
_PAGED_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1151", "gfx1200", "gfx1201", "gfx942", "gfx950")


def _paged_numeric_case(dtype, d, bs, hq, hkv, seq_kv, seq_q, seed=7):
    """Random K/V in a shuffled paged cache plus Q (packed [T, Hq, D])."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_kv)
    max_kv = max(seq_kv)
    blocks = -(-max_kv // bs)
    num_blocks = batch * (blocks + 2) + 5
    cache = torch.randn(num_blocks, hkv, bs, 2 * d, dtype=dtype, device="cuda", generator=g)
    k_cache, v_cache = cache.split(d, dim=-1)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[: batch * (blocks + 2)]
    table = perm.to(torch.int32).view(batch, blocks + 2).contiguous()
    tokens = sum(seq_q)
    q = torch.randn(tokens, hq, d, dtype=dtype, device="cuda", generator=g)
    o = torch.full((tokens, hq, d), float("nan"), dtype=dtype, device="cuda")
    offsets = torch.tensor([0] + list(torch.tensor(seq_q).cumsum(0)), dtype=torch.int32, device="cuda")
    return types.SimpleNamespace(
        batch=batch, max_kv=max_kv, max_q=max(seq_q), k_cache=k_cache, v_cache=v_cache, table=table, q=q, o=o,
        offsets=offsets, seq_q=list(seq_q), seq_kv=list(seq_kv), bs=bs, d=d,
        seq_q_t=torch.tensor(seq_q, dtype=torch.int32, device="cuda"),
        seq_kv_t=torch.tensor(seq_kv, dtype=torch.int32, device="cuda"),
    )


def _gather_dense_ref(c, scale, left):
    """Gather each sequence's pages into dense K/V, then the dense reference
    (bottom-right causal, optional left bound) per sequence."""
    outs = []
    start = 0
    for b in range(c.batch):
        n_kv, n_q = c.seq_kv[b], c.seq_q[b]
        pages = c.table[b, : -(-n_kv // c.bs)].long()
        k = c.k_cache[pages].permute(1, 0, 2, 3).reshape(c.k_cache.shape[1], -1, c.d)[:, :n_kv]
        v = c.v_cache[pages].permute(1, 0, 2, 3).reshape(c.v_cache.shape[1], -1, c.d)[:, :n_kv]
        q = c.q[start : start + n_q].transpose(0, 1)
        outs.append(_ref(q[None], k[None], v[None], scale, left, 0, True)[0].transpose(0, 1))
        start += n_q
    return torch.cat(outs)


def _paged_run(C, c, scale, left, decode):
    kw = dict(causal=True, left_bound=left, page_table=c.table, seq_len_kv=c.seq_kv_t, max_seq_len_kv=c.max_kv)
    if decode:
        q, o = c.q.unsqueeze(2), c.o.unsqueeze(2)  # [B, Hq, 1, D]
    else:
        q, o = c.q, c.o
        kw.update(ragged_offset_q=c.offsets, seq_len_q=c.seq_q_t, max_seq_len_q=c.max_q)
    reason, detail = C.sdpa_fwd_probe(q, c.k_cache, c.v_cache, o, scale, **kw)
    if reason and _arch() not in _PAGED_ARCHS:
        pytest.skip(f"no paged SDPA engine bundled for {_arch()}: {reason}")
    names = {f"0x{e['id'] & 0xFFFFFFFFFFFFFFFF:016x}": e["name"] for e in C.engines(0)}
    assert reason == "", f"{_arch()} declined the {'paged' if decode else 'ragged+paged'} graph: {reason} {detail}"
    print(f"{_arch()} {'p2a' if decode else 'p2b'} {c.q.dtype} D{c.d} bs{c.bs}: engine {names.get(detail, detail)}")
    torch.ops.hipdnn_vllm.sdpa_fwd(q, c.k_cache, c.v_cache, o, scale, **kw)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d,bs", [(64, 16), (128, 32), (128, 16)])
@pytest.mark.parametrize("left", [-1, 47])
def test_paged_decode_matches_gathered_dense(C, dtype, d, bs, left):
    """p2a: one query per sequence over a paged K/V cache (vLLM decode)."""
    c = _paged_numeric_case(dtype, d, bs, hq=8, hkv=2, seq_kv=[100, 37, 1, 64], seq_q=[1, 1, 1, 1])
    scale = d**-0.5
    _paged_run(C, c, scale, left, decode=True)
    torch.testing.assert_close(c.o.float(), _gather_dense_ref(c, scale, left), **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d,bs", [(64, 16), (128, 32)])
@pytest.mark.parametrize("left", [-1, 47])
def test_ragged_paged_matches_gathered_dense(C, dtype, d, bs, left):
    """p2b: ragged Q (chunked prefill plus decode rows) over a paged K/V cache."""
    c = _paged_numeric_case(dtype, d, bs, hq=8, hkv=2, seq_kv=[100, 37, 9, 64], seq_q=[20, 37, 1, 5])
    scale = d**-0.5
    _paged_run(C, c, scale, left, decode=False)
    torch.testing.assert_close(c.o.float(), _gather_dense_ref(c, scale, left), **_tol(dtype))


# --------------------------------------------- torch.compile / capture ----


def test_torch_compile_fullgraph(C):
    dtype = _expected_p1e_dtypes()[0] if _expected_p1e_dtypes() else torch.bfloat16
    q, k, v = _qkv(256, 4, 4, 128, dtype)
    views = [_bhsd(t, 2, 128) for t in (q, k, v)]
    if C.sdpa_fwd_probe(*views, views[0].new_empty(views[0].shape), 0.125)[0]:
        pytest.skip("dense SDPA declined on this arch")

    def f(q, k, v):
        out = torch.empty_like(q)
        torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, out, 0.125)
        return out * 1.0

    eager = f(*views)
    compiled = torch.compile(f, fullgraph=True)(*views)
    torch.testing.assert_close(compiled, eager)


def test_hip_graph_capture_replay(C):
    dtype = _expected_p1e_dtypes()[0] if _expected_p1e_dtypes() else torch.bfloat16
    q, k, v = _qkv(256, 4, 4, 128, dtype)
    out = torch.empty(256, 4, 128, dtype=dtype, device="cuda")
    qv, kv, vv, ov = (_bhsd(t, 2, 128) for t in (q, k, v, out))
    _run_or_skip(C, qv, kv, vv, ov, 0.125)  # warm-up: plan built before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        torch.ops.hipdnn_vllm.sdpa_fwd(qv, kv, vv, ov, 0.125)
    q2, k2, v2 = _qkv(256, 4, 4, 128, dtype, seed=9)
    q.copy_(q2), k.copy_(k2), v.copy_(v2)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(ov.float(), _ref(qv, kv, vv, 0.125), **_tol(dtype))


# ------------------------------------------------ backend impl (P1e) ----


def _impl(monkeypatch, env: dict[str, str], **kw):
    pytest.importorskip("vllm")
    for name in ("HIPDNN_VLLM_OPS", "HIPDNN_VLLM_STRICT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    from vllm.v1.attention.backend import AttentionType

    from hipdnn_vllm.attention.backend import HipdnnAttentionImpl

    args = dict(num_heads=12, head_size=64, scale=0.125, num_kv_heads=12, alibi_slopes=None,
                sliding_window=None, kv_cache_dtype="auto", attn_type=AttentionType.ENCODER_ONLY)
    args.update(kw)
    return HipdnnAttentionImpl(**args)


def _encoder_batch(lengths, hq=12, hkv=12, d=64, dtype=torch.bfloat16):
    from hipdnn_vllm.attention.metadata import build_info

    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    tokens = starts[-1]
    q, k, v = _qkv(tokens + 3, hq, hkv, d, dtype, seed=5)  # +3 padding tokens
    qsl_cpu = torch.tensor(starts, dtype=torch.int32)
    cm = types.SimpleNamespace(
        num_reqs=len(lengths), num_actual_tokens=tokens, max_query_len=max(lengths),
        max_seq_len=max(lengths), query_start_loc=qsl_cpu.cuda(), query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        block_table_tensor=None, causal=False,
    )
    md = types.SimpleNamespace(
        num_actual_tokens=tokens, query_start_loc=cm.query_start_loc, seq_lens=cm.seq_lens,
        max_query_len=cm.max_query_len, hipdnn=build_info(cm, 1, True),
    )
    return q, k, v, md, tokens


def _forward(impl, q, k, v, md):
    out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=q.dtype, device="cuda")
    impl.forward(None, q, k, v, None, md, out)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lengths", [[7], [16, 16, 16, 5, 33, 33], [128, 1, 300]])
@pytest.mark.parametrize("hq,hkv,d", [(12, 12, 64), (16, 4, 128)])
def test_impl_p1e_matches_triton_fallback(C, monkeypatch, dtype, lengths, hq, hkv, d):
    from hipdnn_vllm import stats

    q, k, v, md, tokens = _encoder_batch(lengths, hq=hq, hkv=hkv, d=d, dtype=dtype)
    hip = _impl(monkeypatch, {}, num_heads=hq, num_kv_heads=hkv, head_size=d)
    probe_views = [_bhsd(t[: lengths[0]], 1, lengths[0]) for t in (q, k, v, q)]
    if C.sdpa_fwd_probe(*probe_views, 0.125)[0]:
        pytest.skip(f"dense SDPA {dtype} D{d} declined on {_arch()}")
    before = stats.snapshot()["counters"]
    out_hip = _forward(hip, q, k, v, md)
    after = stats.snapshot()["counters"]

    def hits(rows):
        return sum(r["count"] for r in rows
                   if r["op"] == "attention.encoder_only" and r["route"] == "hipdnn")

    assert hits(after) == hits(before) + 1
    # attention disabled -> TRITON_ATTN
    tri = _impl(monkeypatch, {"HIPDNN_VLLM_OPS": "rms_norm"}, num_heads=hq, num_kv_heads=hkv, head_size=d)
    out_tri = _forward(tri, q, k, v, md)
    torch.testing.assert_close(out_hip[:tokens].float(), out_tri[:tokens].float(), **_tol(dtype))
    # And against torch SDPA per request.
    start = 0
    for n in lengths:
        qv, kv, vv = (_bhsd(t[start : start + n], 1, n) for t in (q, k, v))
        ref = _ref(qv, kv, vv, 0.125)[0].transpose(0, 1)
        torch.testing.assert_close(out_hip[start : start + n].float(), ref, **_tol(dtype))
        start += n


def test_impl_prefiltered_feature_falls_back_and_strict_raises(C, monkeypatch):
    from hipdnn_vllm.declines import DeclineError

    q, k, v, md, tokens = _encoder_batch([9, 9])
    impl = _impl(monkeypatch, {}, logits_soft_cap=30.0)
    ref = _impl(monkeypatch, {"HIPDNN_VLLM_OPS": "rms_norm"}, logits_soft_cap=30.0)
    torch.testing.assert_close(_forward(impl, q, k, v, md), _forward(ref, q, k, v, md))
    strict = _impl(monkeypatch, {"HIPDNN_VLLM_STRICT": "1"}, logits_soft_cap=30.0)
    with pytest.raises(DeclineError, match="UNSUPPORTED_FEATURE:softcap"):
        _forward(strict, q, k, v, md)


def test_impl_engine_decline_falls_back(C, monkeypatch):
    """A graph no engine takes (head size 40) falls back with the build reason."""
    from hipdnn_vllm import stats

    q, k, v, md, tokens = _encoder_batch([10], hq=2, hkv=2, d=40)
    impl = _impl(monkeypatch, {}, num_heads=2, num_kv_heads=2, head_size=40)
    reason = C.sdpa_fwd_probe(*[_bhsd(t[:10], 1, 10) for t in (q, k, v, q)], 0.125)[0]
    if not reason:
        pytest.skip("an engine accepts head size 40 on this arch")
    out = _forward(impl, q, k, v, md)
    ref = _impl(monkeypatch, {"HIPDNN_VLLM_OPS": "rms_norm"}, num_heads=2, num_kv_heads=2, head_size=40)
    torch.testing.assert_close(out, _forward(ref, q, k, v, md))
    rows = stats.snapshot()["counters"]
    assert any(r["op"] == "attention.encoder_only" and r["reason"] == reason for r in rows)


def test_impl_under_capture_uses_fallback(C, monkeypatch):
    q, k, v, md, tokens = _encoder_batch([16, 16])
    _forward(_impl(monkeypatch, {"HIPDNN_VLLM_OPS": "rms_norm"}), q, k, v, md)  # Triton JIT
    impl = _impl(monkeypatch, {})
    out = torch.zeros(q.shape, dtype=q.dtype, device="cuda")
    impl.forward(None, q, k, v, None, md, out)  # eager warm-up (Triton JIT, plans)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        impl.forward(None, q, k, v, None, md, out)
    route = impl._routes[(md.hipdnn.batch_class, True)]
    assert route.path is None and route.reason == "CAPTURE_INVARIANT:p1e_eager_only"
    g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out[:tokens]).all()


if os.environ.get("HIPDNN_VLLM_PRINT_ENGINES"):

    def test_print_engines(C):
        for e in C.engines(0):
            print(e)
