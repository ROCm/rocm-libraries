# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unified split-KV (3D) LSE output vs an independent FP32 paged reference."""

import dataclasses
import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


requires_cdna3_or_4 = pytest.mark.skipif(
    _arch() is None, reason="needs a gfx942 or gfx950 GPU with ROCm torch"
)


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _paged_reference(
    q, k_cache, v_cache, q_lens, kv_lens, block_table, scale, softcap, sliding_window=0
):
    """FP32 O and natural-log LSE with the unified bottom-right causal mask
    (key k visible to query p iff k <= diag, diag = p + kv_len - q_len) and, when
    ``sliding_window`` > 0, diag - k < sliding_window. Rows with no key get
    O = 0 and LSE = -inf."""
    hq, d = q.shape[1], q.shape[2]
    hkv, bs = k_cache.shape[2], k_cache.shape[1]
    outs, lses, start = [], [], 0
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q[start : start + ql].float().transpose(0, 1)  # [hq, ql, d]
        nblk = (kl + bs - 1) // bs
        blocks = block_table[i, :nblk].long()
        k = k_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        v = v_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        k = k.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        v = v.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        s = torch.matmul(qi, k.transpose(-1, -2)) * scale
        if softcap > 0:
            s = softcap * torch.tanh(s / softcap)
        p = torch.arange(ql, device=q.device).view(-1, 1)
        kk = torch.arange(kl, device=q.device).view(1, -1)
        diag = p + (kl - ql)
        visible = kk <= diag
        if sliding_window > 0:
            visible = visible & (diag - kk < sliding_window)
        s = s.masked_fill(~visible, float("-inf"))
        lse = torch.logsumexp(s, dim=-1)  # [hq, ql]
        has = torch.isfinite(lse)
        prob = torch.exp(s - torch.where(has, lse, 0.0).unsqueeze(-1))
        prob = torch.where(has.unsqueeze(-1), prob, 0.0)
        outs.append(torch.matmul(prob, v).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
        start += ql
    return torch.cat(outs), torch.cat(lses).unsqueeze(-1)


def _case(
    q_lens,
    kv_lens,
    *,
    hq=32,
    hkv=8,
    d=128,
    dtype="bf16",
    bs=16,
    softcap=0.0,
    sliding_window=0,
    seed=0,
):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    total_q, num_seqs = sum(q_lens), len(q_lens)
    max_blocks = max(1, max((kl + bs - 1) // bs for kl in kv_lens))
    num_blocks = num_seqs * max_blocks + 3
    q = torch.randn(total_q, hq, d, device="cuda", dtype=tdt, generator=g)
    k_cache = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    v_cache = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    # Shuffled physical pages: catches any block-table indexing shortcut.
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[
        : num_seqs * max_blocks
    ]
    block_table = perm.view(num_seqs, max_blocks).to(torch.int32).contiguous()
    cu = torch.tensor(
        [0, *itertools.accumulate(q_lens)], dtype=torch.int32, device="cuda"
    )
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    problem = UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(max(kv_lens), 1),
        dtype=dtype,
        softcap=softcap,
        sliding_window=sliding_window,
        num_kv_blocks=num_blocks,
        emit_lse=True,
    )
    return problem, q, k_cache, v_cache, cu, seqused, block_table


def _run(problem, q, k_cache, v_cache, cu, seqused, block_table, *, lse, softcap):
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        block_table=block_table,
        softcap=softcap,
        backend="3d",
        lse=lse,
    )
    torch.cuda.synchronize()
    return out


_SHAPES = [
    # (q_lens, kv_lens)               what it covers
    ([1, 1, 1, 1], [4096, 777, 1500, 2049]),  # decode, ragged KV, partial last page
    ([4, 2], [3000, 1029]),  # spec-decode
    ([64], [64]),  # s_q == s_kv
    ([5], [3]),  # bottom-right s_q > s_kv: rows 0,1 empty
    ([1, 1], [0, 600]),  # kv_len == 0: whole sequence empty
]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("q_lens,kv_lens", _SHAPES)
def test_split_kv_output_and_lse(dtype, d, q_lens, kv_lens):
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens, d=d, dtype=dtype)
    lse = torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("nan"), device="cuda"
    )
    out = _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0)
    off = _run(
        dataclasses.replace(problem, emit_lse=False),
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=None,
        softcap=0.0,
    )
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(d), 0.0
    )
    tol = _tolerance(dtype)
    assert torch.equal(out, off), "LSE-on changed O"
    assert (out.float() - o_ref).abs().max().item() < tol
    assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref))
    finite = torch.isfinite(lse_ref)
    assert not torch.isnan(lse).any()
    assert (lse[finite] - lse_ref[finite]).abs().max().item() < tol


@requires_cdna3_or_4
@pytest.mark.gpu
def test_split_kv_lse_with_softcap():
    q_lens, kv_lens = [1, 1], [2048, 900]
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens, softcap=30.0)
    lse = torch.empty((problem.total_q, problem.num_query_heads, 1), device="cuda")
    _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=30.0)
    _, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 30.0
    )
    assert (lse - lse_ref).abs().max().item() < _tolerance("bf16")


def _assert_matches_reference(out, lse, o_ref, lse_ref, tol):
    assert torch.isfinite(out).all(), "non-finite O"
    assert (out.float() - o_ref).abs().max().item() < tol
    assert not torch.isnan(lse).any(), "an LSE row was left unwritten"
    assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref))
    finite = torch.isfinite(lse_ref)
    assert (lse[finite] - lse_ref[finite]).abs().max().item() < tol


def _nan_lse(problem):
    return torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("nan"), device="cuda"
    )


# Decode with ragged KV and spec-decode: the shapes the split-KV path serves.
_DECODE_SHAPES = [([1, 1, 1, 1], [4096, 777, 1500, 2049]), ([4, 2], [3000, 1029])]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_one_query_head_per_kv_head(q_lens, kv_lens):
    """qpk = 1 (Hq == Hkv): a different query-head-to-row layout than GQA."""
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens, hq=8, hkv=8)
    lse = _nan_lse(problem)
    out = _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance("bf16"))


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("window", [128, 1000])
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_sliding_window(q_lens, kv_lens, window):
    """Sliding window empties the segments left of the window for every row."""
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens, sliding_window=window)
    lse = _nan_lse(problem)
    out = _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0, window
    )
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance("bf16"))


_PINNED_SPLIT_KV_SPEC = {
    "gfx942": "gfx942_u3d_splitkv_seg16_t1xb",
    "gfx950": "gfx950_u3d_splitkv_seg16_t1xb",
}


@requires_cdna3_or_4
@pytest.mark.gpu
def test_pinned_split_kv_spec_end_to_end():
    """The dispatcher route: a request pinned to a split-KV tuning spec with an
    explicit segment count, bound to real tensors and launched."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens)
    request = AttentionRequest(
        batch=4,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch=_arch(),
        mask_type=2,  # bottom-right causal, as in _paged_reference
        dtype="bf16",
        emit_lse=True,
        algorithm="unified_tuning",
        spec_id=_PINNED_SPLIT_KV_SPEC[_arch()],
    )
    result = dispatch_attention(request)
    assert result.spec.reduce_spec.emit_lse is True
    assert result.spec.reduce_spec.num_segments == 16
    out, lse = torch.empty_like(q), _nan_lse(problem)
    tensors = dict(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=su, block_table=bt, lse=lse
    )
    result.bind_torch(tensors).launch()
    torch.cuda.synchronize()
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance("bf16"))


# (q_len, kv_len): the last KV tile holds only keys row 0 cannot see (bottom-right
# causal: row 0 sees kv_len - q_len + 1 keys), for any tile of 16, 32 or 64; 5/3
# also leaves rows 0 and 1 with no visible key at all.
_MASKED_TILE_SHAPES = [(4, 17), (4, 33), (4, 65), (5, 3)]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("emit_lse", [True, False])
@pytest.mark.parametrize("q_len,kv_len", _MASKED_TILE_SHAPES)
def test_masked_segment_does_not_underflow_the_merge(q_len, kv_len, emit_lse):
    """A segment whose tiles are all masked for a row stores max=0, sum=0. With
    every visible logit at -200 (Q=2, K=-12.5, D=64, scale 1/8), letting that
    max set the merge max underflows every real segment: O became 0 and LSE
    -inf for rows that do see keys. Keyless rows must stay O=0, LSE=-inf."""
    problem, q, kc, vc, cu, su, bt = _case([q_len], [kv_len], d=64)
    q.fill_(2.0)
    kc.fill_(-12.5)
    problem = dataclasses.replace(problem, emit_lse=emit_lse)
    lse = (
        torch.full(
            (problem.total_q, problem.num_query_heads, 1), float("nan"), device="cuda"
        )
        if emit_lse
        else None
    )
    out = _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(q, kc, vc, [q_len], [kv_len], bt, 0.125, 0.0)
    assert torch.isfinite(out).all(), "the merge produced a non-finite O"
    assert (out.float() - o_ref).abs().max().item() < _tolerance("bf16")
    if emit_lse:
        assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref))
        finite = torch.isfinite(lse_ref)
        assert finite.any(), "the case has no row with visible keys"
        assert (lse[finite] - lse_ref[finite]).abs().max().item() < _tolerance("bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
def test_auto_backend_refuses_lse_on_a_2d_routed_shape():
    problem, q, kc, vc, cu, su, bt = _case([256], [256])
    lse = torch.empty((problem.total_q, problem.num_query_heads, 1), device="cuda")
    out = torch.empty_like(q)
    with pytest.raises(NotImplementedError, match="LSE"):
        run_unified_attention_torch(
            problem=problem,
            q=q,
            k=kc,
            v=vc,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=su,
            softmax_scale=1.0 / math.sqrt(128),
            block_table=bt,
            softcap=0.0,
            backend="auto",
            lse=lse,
        )


@requires_cdna3_or_4
@pytest.mark.gpu
def test_cached_lse_off_2d_graph_never_serves_an_lse_request():
    """Regression: the gfx942 2D graph fast path replays a cache hit before any
    support gate, keyed on a signature without emit_lse. An LSE-off launch on
    these exact buffers must not be replayed for an LSE-on request.

    On gfx942 the test first proves the hazard is live: the LSE-off launch must
    capture a graph whose key the LSE-on request would hit. gfx950 has no 2D
    graph path, so there it checks only the plain refusal."""
    from kernels.common import attention_unified as au

    problem, q, kc, vc, cu, su, bt = _case([256], [256])
    off_problem = dataclasses.replace(problem, emit_lse=False)
    out = torch.empty_like(q)
    args = dict(
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=su,
        softmax_scale=1.0 / math.sqrt(128),
        block_table=bt,
        softcap=0.0,
        backend="auto",
    )
    before = set(au._2D_GRAPHS)
    run_unified_attention_torch(problem=off_problem, **args)  # warm the 2D graph
    captured = set(au._2D_GRAPHS) - before
    run_unified_attention_torch(problem=off_problem, **args)  # cache hit: replay
    if _arch() == "gfx942":
        assert len(captured) == 1, "LSE-off launch did not capture a 2D graph"
        (key,) = captured
        sig = au._cheap_2d_sig(problem)  # the LSE-on request's signature
        assert key[: len(sig)] == sig and id(out) in key[len(sig) :]
        assert set(au._2D_GRAPHS) - before == captured, "second launch was not a replay"
    else:
        assert not captured, "2D graph replay is expected only on gfx942"
    torch.cuda.synchronize()
    # A replay writes ``out``, never ``lse``: poison ``out`` in place (same
    # tensor, so the captured graph key still matches) to see any replay.
    out.fill_(float("nan"))
    lse = torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("nan"), device="cuda"
    )
    with pytest.raises(NotImplementedError, match="LSE"):
        run_unified_attention_torch(problem=problem, lse=lse, **args)
    torch.cuda.synchronize()
    assert torch.isnan(out).all(), "a 2D kernel ran for a refused LSE request"
    assert torch.isnan(lse).all(), "a kernel wrote LSE for a refused request"


def _aliases_last_lse_element(lse, like):
    """A tensor with ``like``'s 4-byte dtype and shape whose first element
    shares memory with the last element of ``lse``: the smallest possible
    overlap, and otherwise a valid input."""
    pool = torch.empty(lse.numel() - 1 + like.numel(), device="cuda")
    overlapping_lse = pool[: lse.numel()].view(lse.shape)
    alias = pool[lse.numel() - 1 :].view(like.dtype).view(like.shape)
    return overlapping_lse, alias


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        "rows",
        "heads",
        "dtype",
        "strided",
        "host",
        "cu_seqlens_q",
        "seqused_k",
        "block_table",
        "additive_bias",
    ],
)
def test_malformed_lse_buffer_is_refused_before_launch(case):
    problem, q, kc, vc, cu, su, bt = _case([4, 2], [3000, 1029])
    tq, hq = problem.total_q, problem.num_query_heads
    lse = torch.full((tq, hq, 1), float("nan"), device="cuda")
    if case == "rows":
        lse = torch.full((tq - 1, hq, 1), float("nan"), device="cuda")
    elif case == "heads":
        lse = torch.full((tq, hq - 1, 1), float("nan"), device="cuda")
    elif case == "dtype":
        lse = lse.half()
    elif case == "strided":
        lse = torch.full((tq, hq, 2), float("nan"), device="cuda")[..., :1]
    elif case == "host":
        lse = lse.cpu()
    elif case == "cu_seqlens_q":
        lse, cu = _aliases_last_lse_element(lse, cu)
    elif case == "seqused_k":
        lse, su = _aliases_last_lse_element(lse, su)
    elif case == "block_table":
        lse, bt = _aliases_last_lse_element(lse, bt)
    bias = {}
    if case == "additive_bias":
        # The reduce must never write LSE over a bias it (or a later replay)
        # reads. The alias is otherwise a valid bias (float32, long enough for
        # the broadcast strides), so only the overlap check can refuse it.
        problem = dataclasses.replace(problem, use_additive_bias=True)
        like = torch.empty(4096, device="cuda")
        lse, storage = _aliases_last_lse_element(lse, like)
        bias = dict(additive_bias=storage, additive_bias_sq_stride=0)
    out = torch.full_like(q, float("nan"))
    aliased = ("cu_seqlens_q", "seqused_k", "block_table", "additive_bias")
    with pytest.raises(
        ValueError, match=f"lse overlaps {case}" if case in aliased else "lse"
    ):
        run_unified_attention_torch(
            problem=problem,
            q=q,
            k=kc,
            v=vc,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=su,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            block_table=bt,
            softcap=0.0,
            backend="3d",
            lse=lse,
            **bias,
        )
    torch.cuda.synchronize()
    assert torch.isnan(out).all(), "a kernel ran for a malformed lse buffer"
