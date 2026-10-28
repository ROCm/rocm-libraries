# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unified paged attention LSE output -- split-KV (3D) reduce and tiled 2D
epilogues on gfx942/gfx950 -- vs an independent FP32 paged reference."""

import dataclasses
import itertools
import math
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from kernels.common import attention_unified as au  # noqa: E402


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
    q,
    k_cache,
    v_cache,
    q_lens,
    kv_lens,
    block_table,
    scale,
    softcap,
    sliding_window=0,
    additive_bias=None,
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
        if additive_bias is not None:
            # [seq, q_head, q_row, key], added after softcap, as the kernels do.
            s = s + additive_bias[i, :, :ql, :kl]
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


def _run(
    problem,
    q,
    k_cache,
    v_cache,
    cu,
    seqused,
    block_table,
    *,
    lse,
    softcap,
    backend="3d",
    tuning_spec=None,
):
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
        backend=backend,
        tuning_spec=tuning_spec,
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
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_split_kv_lse_nan_query_row_stays_nan(dtype):
    """The split-KV reduce follows the tiled 2D rule: a NaN query row gives
    NaN O and NaN LSE for that (token, head) only, never the keyless -inf.
    Its merged sum is NaN, and !(l == 0) keeps row_has_mass true."""
    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    nan_token, nan_head = 1, 5
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens, dtype=dtype)
    q[nan_token, nan_head] = float("nan")
    lse = torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("inf"), device="cuda"
    )
    out = _run(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(q.shape[-1]), 0.0
    )
    assert not torch.isposinf(lse).any(), "an LSE row was left unwritten"
    assert torch.isnan(lse[nan_token, nan_head]).all(), "NaN row got a non-NaN LSE"
    assert torch.isnan(out[nan_token, nan_head]).all(), "NaN row was hidden in O"
    rest = torch.ones(
        problem.total_q, problem.num_query_heads, dtype=torch.bool, device="cuda"
    )
    rest[nan_token, nan_head] = False
    assert torch.isfinite(lse.squeeze(-1)[rest]).all(), "NaN leaked into another LSE"
    assert torch.isfinite(out[rest]).all(), "NaN leaked into another (row, head)"
    assert (out[rest].float() - o_ref[rest]).abs().max().item() < _tolerance(dtype)
    torch.testing.assert_close(
        lse.squeeze(-1)[rest], lse_ref.squeeze(-1)[rest], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


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


# ---------------------------------------------------------------------------
# Tiled 2D LSE (gfx942 / gfx950 build_unified_attention_2d_tiled)
# ---------------------------------------------------------------------------

_LSE_ATOL = 1e-3
_LSE_RTOL = 1e-3

# Prefill shapes the tiled 2D path serves. With 32 query / 8 KV heads every one
# routes to 2D under backend="auto" (UnifiedAttentionProblem.select_path).
_PREFILL_SHAPES = [
    # (q_lens, kv_lens)                    what it covers
    ([256], [256]),  # s_q == s_kv; the gfx942 2D-graph-eligible shape
    ([37, 300, 1], [37, 812, 900]),  # ragged, causal s_q < s_kv, partial pages
    (
        [5],
        [3],
    ),  # bottom-right s_q > s_kv: rows 0,1 see no key (m=-inf, l=0 after Part 0)
    ([3, 64], [0, 64]),  # kv_len == 0: zero-trip KV loop (m=-inf, l=1)
]


def _forbid_split_kv(*_args, **_kwargs):
    raise AssertionError("a tiled 2D LSE test reached the split-KV (3D) path")


def _case2d(q_lens, kv_lens, **kw):
    """``_case`` for a tiled 2D LSE launch; fails fast, with the gate's reason,
    if the tiled 2D gate refuses the LSE problem on this GPU."""
    case = _case(q_lens, kv_lens, **kw)
    ok, why = au.supports_native_unified_attention_tiled(case[0], _arch())
    assert ok, f"the tiled 2D gate refuses this LSE problem: {why}"
    return case


def _run2d(
    problem,
    q,
    k_cache,
    v_cache,
    cu,
    seqused,
    block_table,
    *,
    lse,
    softcap,
    backend,
    tuning_spec=None,
):
    """One 2D launch. The split-KV runner is patched to fail, so a pass proves
    the tiled 2D path wrote O (and LSE)."""
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        return _run(
            problem,
            q,
            k_cache,
            v_cache,
            cu,
            seqused,
            block_table,
            lse=lse,
            softcap=softcap,
            backend=backend,
            tuning_spec=tuning_spec,
        )


def _run2d_lse_off(
    problem, q, k_cache, v_cache, cu, seqused, block_table, *, softcap, backend
):
    """The LSE-off launch of the kernel the LSE-on request runs. With LSE on,
    the gfx942 4-warp cohorts fall back to the default builder and the gfx942
    2D graph is bypassed; force both here so ``torch.equal`` compares
    one kernel with and without its LSE store. Both are no-ops on gfx950."""
    off = dataclasses.replace(problem, emit_lse=False)
    with mock.patch.object(
        au, "_gfx942_4warp_eligible", lambda problem, arch: False
    ), mock.patch.object(au, "_enable_2d_graph_replay", lambda problem, arch: False):
        return _run2d(
            off,
            q,
            k_cache,
            v_cache,
            cu,
            seqused,
            block_table,
            lse=None,
            softcap=softcap,
            backend=backend,
        )


def _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype):
    assert not torch.isnan(lse).any(), "an LSE row was left unwritten"
    assert torch.equal(out, off), "LSE-on changed O"
    assert torch.isfinite(out).all(), "non-finite O"
    assert (out.float() - o_ref).abs().max().item() < _tolerance(dtype)
    keyless = torch.isneginf(lse_ref.squeeze(-1))  # [total_q, Hq]
    assert (out[keyless] == 0).all(), "a row with no visible key has O != 0"
    assert torch.equal(
        torch.isneginf(lse), torch.isneginf(lse_ref)
    ), "the -inf (keyless) LSE rows differ from the reference"
    assert not torch.isposinf(lse).any(), "+inf LSE"
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(
        lse[finite], lse_ref[finite], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


# One explicit dispatcher geometry per tiled 2D epilogue branch (flags =
# dispatch/attention/axes.py _CODEPATH_KNOBS).
_BRANCH_VARIANT = {
    "gfx942": {
        "transposed32": "transposed_x8_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    "gfx950": {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
}
# A: transposed 32x32 (16 broadcast slots, and the 1-slot scalar state that
# auto routing ships); B: non-transposed 32x32 Acc_lds; C: striped 16x16.
_BRANCHES = ("transposed32", "transposed32_s1", "mfma32", "striped16")


def _epilogue_branch(kernel_spec) -> str:
    wide = bool(
        getattr(kernel_spec, "use_mfma_32x32", False)
        or getattr(kernel_spec, "use_mfma_32x32x8", False)
    )
    if not wide:
        return "striped16"
    return "transposed32" if kernel_spec.use_transposed_qk_32x32 else "mfma32"


def _branch_spec(problem, branch):
    """The explicit tuning spec that forces ``branch`` for ``problem``."""
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    base = branch.removesuffix("_s1")
    spec_id = f"{arch}_u2d_{_BRANCH_VARIANT[arch][base]}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    knobs = {"use_transposed_scalar_state": True} if branch.endswith("_s1") else {}
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    ks = spec.kernel_spec
    assert _epilogue_branch(ks) == base, f"{spec_id} selects {_epilogue_branch(ks)}"
    assert bool(getattr(ks, "use_transposed_scalar_state", False)) == (
        branch.endswith("_s1")
    )
    assert ks.emit_lse is problem.emit_lse
    return spec


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
def test_auto_backend_writes_lse_on_a_2d_routed_shape():
    """A shape the runner used to refuse: backend='auto' routes it to tiled
    2D, which now writes LSE itself (the split-KV runner is patched to fail)."""
    problem, q, kc, vc, cu, su, bt = _case2d([256], [256])
    assert problem.select_path() == "2d"
    lse = _nan_lse(problem)
    out = _run2d(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0, backend="auto")
    off = _run2d_lse_off(problem, q, kc, vc, cu, su, bt, softcap=0.0, backend="auto")
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, [256], [256], bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
def test_cached_lse_off_2d_graph_never_serves_an_lse_request():
    """Regression: the gfx942 2D graph fast path replays a cache hit before any
    support gate, keyed on _cheap_2d_sig (no emit_lse) + buffer ids. An LSE-off
    graph captured on these exact buffers must not be replayed for an LSE-on
    request: the replay would write O and leave lse unwritten.

    On gfx942 the test first proves the hazard is live (the LSE-off launch
    captures a graph whose key the LSE-on request would hit). Then the LSE-on
    call must write O and LSE, match the reference, and add no graph. gfx950 has
    no 2D graph path, so there only the served-and-correct half runs."""
    problem, q, kc, vc, cu, su, bt = _case2d([256], [256])
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
    torch.cuda.synchronize()
    if _arch() == "gfx942":
        assert len(captured) == 1, "LSE-off launch did not capture a 2D graph"
        (key,) = captured
        sig = au._cheap_2d_sig(problem)  # the LSE-on request's signature
        assert key[: len(sig)] == sig and id(out) in key[len(sig) :]
        assert set(au._2D_GRAPHS) - before == captured, "second launch was not a replay"
    else:
        assert not captured, "2D graph replay is expected only on gfx942"
    off_out = out.clone()
    graphs = set(au._2D_GRAPHS)
    # A replay writes ``out``, never ``lse``: poison both in place (same
    # tensors, so the captured graph key still matches).
    out.fill_(float("nan"))
    lse = _nan_lse(problem)
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        run_unified_attention_torch(problem=problem, lse=lse, **args)
    torch.cuda.synchronize()
    assert set(au._2D_GRAPHS) == graphs, "an LSE request captured a 2D graph"
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, [256], [256], bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_2d_result(out, off_out, lse, o_ref, lse_ref, "bf16")


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
@pytest.mark.parametrize("backend", ["3d", "tiled"])
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
def test_malformed_lse_buffer_is_refused_before_launch(case, backend):
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
            backend=backend,
            lse=lse,
            **bias,
        )
    torch.cuda.synchronize()
    assert torch.isnan(out).all(), "a kernel ran for a malformed lse buffer"


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_output_and_lse(backend, dtype, d, q_lens, kv_lens):
    """Arch-native routing. On gfx942 bf16 D256 this is the shape the 4-warp
    lean body serves with LSE off (with LSE on it takes the default builder)."""
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    if backend == "auto":
        assert problem.select_path() == "2d", "auto would not route this to 2D"
    lse = _nan_lse(problem)
    out = _run2d(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(problem, q, kc, vc, cu, su, bt, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(d), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_tiled_2d_lse_one_query_head_per_kv_head(dtype, q_lens, kv_lens):
    """Hq == Hkv (8:8): NQK = 1, so every CTA row is its own token. backend is
    'tiled' because auto sends the ragged 8:8 shape to split-KV."""
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens, hq=8, hkv=8, dtype=dtype)
    lse = _nan_lse(problem)
    out = _run2d(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0, backend="tiled")
    off = _run2d_lse_off(problem, q, kc, vc, cu, su, bt, softcap=0.0, backend="tiled")
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("window", [64, 4096])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_sliding_window(backend, window, dtype):
    """D128 sliding window: on gfx942 the LSE-off route is the 4-warp GQA body,
    so this is where the LSE-on fallback to the default builder is exercised. Seq 0
    (256 queries over 5000 keys) is cut by both windows; seq 1 (64/64) only by 64."""
    q_lens, kv_lens = [256, 64], [5000, 64]
    problem, q, kc, vc, cu, su, bt = _case2d(
        q_lens, kv_lens, dtype=dtype, sliding_window=window
    )
    lse = _nan_lse(problem)
    out = _run2d(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(problem, q, kc, vc, cu, su, bt, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0, window
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_softcap(backend):
    """Softcap is applied before the max, in log2 units: LSE must include it."""
    q_lens, kv_lens = [37, 300, 1], [37, 812, 900]
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens, softcap=30.0)
    lse = _nan_lse(problem)
    out = _run2d(problem, q, kc, vc, cu, su, bt, lse=lse, softcap=30.0, backend=backend)
    off = _run2d_lse_off(problem, q, kc, vc, cu, su, bt, softcap=30.0, backend=backend)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 30.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_pinned_branch_writes_lse(branch, d, dtype, q_lens, kv_lens):
    """Every epilogue branch of the arch's tiled 2D builder, forced by an
    explicit tuning spec. Branch B (mfma32) is reachable only this way."""
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    off_problem = dataclasses.replace(problem, emit_lse=False)
    lse = _nan_lse(problem)
    out = _run2d(
        problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(problem, branch),
    )
    off = _run2d(
        off_problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(d), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("branch", ["transposed32", "mfma32", "striped16"])
def test_dispatcher_2d_lse_end_to_end(branch):
    """Request -> pinned 2D tuning spec -> bind_torch(tensors with 'lse') ->
    launch. (The generic unified_2d candidate has no Torch binding; executable
    dispatch goes through the tuning candidates, as for split-KV.)"""
    from dispatch.attention import AttentionRequest, dispatch_attention

    arch = _arch()
    q_lens, kv_lens = [256, 256], [256, 700]  # total_q = batch * seqlen_q
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens)

    def request(emit_lse):
        return AttentionRequest(
            batch=2,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=256,
            seqlen_k=700,
            hdim_q=128,
            hdim_v=128,
            arch=arch,
            mask_type=2,  # bottom-right causal, as in _paged_reference
            dtype="bf16",
            emit_lse=emit_lse,
            algorithm="unified_tuning",
            spec_id=f"{arch}_u2d_{_BRANCH_VARIANT[arch][branch]}",
        )

    on = dispatch_attention(request(True))
    off = dispatch_attention(request(False))
    assert on.spec.path == "2d"
    assert _epilogue_branch(on.spec.kernel_spec) == branch
    assert on.spec.kernel_spec.emit_lse is True
    assert off.spec.kernel_spec.emit_lse is False
    tensors = dict(q=q, k=kc, v=vc, cu_seqlens_q=cu, seqused_k=su, block_table=bt)
    out, lse, off_out = torch.empty_like(q), _nan_lse(problem), torch.empty_like(q)
    on.bind_torch({**tensors, "out": out, "lse": lse}).launch()
    off.bind_torch({**tensors, "out": off_out}).launch()
    torch.cuda.synchronize()
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(128), 0.0
    )
    _assert_2d_result(out, off_out, lse, o_ref, lse_ref, "bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,score", [("fp16", -20.0), ("bf16", -200.0)])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_masked_leading_tile_very_negative_scores(branch, dtype, score):
    """Sliding window 64 over 256 tokens: some rows of each CTA see a fully
    masked first KV tile. Every visible score is `score` (natural log). Every
    row sees keys, so every LSE must be finite (about score + ln 64) and O must
    match the reference, on every epilogue branch."""
    problem, q, kc, vc, cu, su, bt = _case2d(
        [256], [256], dtype=dtype, sliding_window=64
    )
    a = 1.0 if dtype == "fp16" else 4.0
    q.fill_(a)
    kc.fill_(score / (a * math.sqrt(128)))
    lse = _nan_lse(problem)
    out = _run2d(
        problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(problem, branch),
    )
    off_problem = dataclasses.replace(problem, emit_lse=False)
    off = _run2d(
        off_problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, [256], [256], bt, 1.0 / math.sqrt(128), 0.0, sliding_window=64
    )
    assert torch.isfinite(lse_ref).all(), "the case must give every row a key"
    assert torch.isfinite(lse).all(), "a row with visible keys got LSE -inf/NaN"
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


_NAN_ROW, _NAN_HEAD = 100, 5


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_nan_query_row_stays_nan(branch, dtype):
    """A NaN in one query row gives NaN O and NaN LSE for that (row, head)
    only. Its softmax sum l is NaN; !(l == 0) keeps row_has_mass true, so
    lse_store propagates NaN instead of writing the keyless -inf."""
    problem, q, kc, vc, cu, su, bt = _case2d([256], [256], dtype=dtype)
    q[_NAN_ROW, _NAN_HEAD] = float("nan")
    lse = torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("inf"), device="cuda"
    )
    out = _run2d(
        problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(problem, branch),
    )
    off_problem = dataclasses.replace(problem, emit_lse=False)
    off = _run2d(
        off_problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, [256], [256], bt, 1.0 / math.sqrt(q.shape[-1]), 0.0
    )
    assert not torch.isposinf(lse).any(), "an LSE row was left unwritten"
    torch.testing.assert_close(out, off, rtol=0, atol=0, equal_nan=True)
    assert torch.isnan(lse[_NAN_ROW, _NAN_HEAD]).all(), "NaN row got a non-NaN LSE"
    assert torch.isnan(out[_NAN_ROW, _NAN_HEAD]).all(), "NaN row was hidden in O"
    rest = torch.ones(
        problem.total_q, problem.num_query_heads, dtype=torch.bool, device="cuda"
    )
    rest[_NAN_ROW, _NAN_HEAD] = False
    assert torch.isfinite(lse.squeeze(-1)[rest]).all(), "NaN leaked into another LSE"
    assert torch.isfinite(out[rest]).all(), "NaN leaked into another (row, head)"
    assert (out[rest].float() - o_ref[rest]).abs().max().item() < _tolerance(dtype)
    torch.testing.assert_close(
        lse.squeeze(-1)[rest], lse_ref.squeeze(-1)[rest], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


# (variant id, geometry suffix, knobs, epilogue branch). Per arch, so every
# collected case runs: gfx950 has no q-major grid.
_ROW_MAP_VARIANTS = {
    "gfx942": [
        (
            "qmajor_transposed32",
            "transposed_x8_nw1_mw32_t2xb_llvm",
            {"use_q_major_grid": True},
            "transposed32",
        ),
        (
            "qmajor_transposed32_s1",
            "transposed_x8_nw1_mw32_t2xb_llvm",
            {"use_q_major_grid": True, "use_transposed_scalar_state": True},
            "transposed32",
        ),
        (
            "qmajor_striped16",
            "narrow_nw1_mw16_t1xb_llvm",
            {"use_q_major_grid": True},
            "striped16",
        ),
        ("striped16_mw32", "narrow_nw1_mw32_t1xb_llvm", {}, "striped16"),
        ("striped16_mw32_nw2", "narrow_nw2_mw32_t1xb_llvm", {}, "striped16"),
    ],
    "gfx950": [
        ("striped16_mw32", "narrow_nw1_mw32_t1xb_llvm", {}, "striped16"),
        ("striped16_mw32_nw2", "narrow_nw2_mw32_t1xb_llvm", {}, "striped16"),
    ],
}


def _variant_spec(problem, geometry, knobs, branch):
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    spec_id = f"{arch}_u2d_{geometry}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    ks = spec.kernel_spec
    assert _epilogue_branch(ks) == branch, f"{spec_id} selects {_epilogue_branch(ks)}"
    for name, value in knobs.items():
        assert getattr(ks, name) == value, f"{spec_id}: {name} did not stick"
    assert ks.emit_lse is problem.emit_lse
    return spec


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "vid,geometry,knobs,branch",
    _ROW_MAP_VARIANTS.get(_arch(), []),
    ids=[v[0] for v in _ROW_MAP_VARIANTS.get(_arch(), [])],
)
def test_row_mapping_variant_writes_lse(
    vid, geometry, knobs, branch, dtype, q_lens, kv_lens
):
    """Swapped grid, two atoms per warp and a second warp change which lane
    and slot owns each (token, head). A wrong mapping writes the right values
    to the wrong index, which only a reference comparison sees. The ragged
    shape spans three sequences; [5]/[3] has keyless rows 0-1."""
    problem, q, kc, vc, cu, su, bt = _case2d(q_lens, kv_lens, dtype=dtype)
    off_problem = dataclasses.replace(problem, emit_lse=False)
    lse = _nan_lse(problem)
    out = _run2d(
        problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_variant_spec(problem, geometry, knobs, branch),
    )
    off = _run2d(
        off_problem,
        q,
        kc,
        vc,
        cu,
        su,
        bt,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_variant_spec(off_problem, geometry, knobs, branch),
    )
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, 1.0 / math.sqrt(q.shape[-1]), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens,branch",
    [
        ([37, 300, 1], [37, 812, 900], "transposed32"),
        ([5], [3], "striped16"),
    ],
)
def test_tiled_2d_lse_with_additive_bias(q_lens, kv_lens, branch):
    """Additive bias is added to the log2 score before the running max, so LSE
    must include it. gfx950 default routing: the ragged shape takes branch A
    (16 slots: bias turns scalar state off), 5/3 takes C and keeps two keyless
    rows. gfx942 has no bias kernel: the tiled 2D gate must refuse it."""
    problem, q, kc, vc, cu, su, bt = _case(q_lens, kv_lens)
    problem = dataclasses.replace(problem, use_additive_bias=True)
    arch = _arch()
    ok, why = au.supports_native_unified_attention_tiled(problem, arch)
    if arch == "gfx942":
        assert not ok and "only implemented on gfx950" in why, why
        return
    assert ok, f"the tiled 2D gate refuses LSE + bias: {why}"
    assert _epilogue_branch(au._tiled_spec_from_problem(problem, arch)) == branch
    g = torch.Generator(device="cuda").manual_seed(1)
    bias = torch.randn(
        len(q_lens),
        32,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        device="cuda",
        generator=g,
    )
    bias_args = dict(
        additive_bias=bias,
        additive_bias_batch_stride=bias.stride(0),
        additive_bias_head_stride=bias.stride(1),
        additive_bias_sq_stride=bias.stride(2),
    )
    scale = 1.0 / math.sqrt(128)

    def launch(prob, lse):
        out = torch.empty_like(q)
        with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
            run_unified_attention_torch(
                problem=prob,
                q=q,
                k=kc,
                v=vc,
                out=out,
                cu_seqlens_q=cu,
                seqused_k=su,
                softmax_scale=scale,
                block_table=bt,
                softcap=0.0,
                backend="tiled",
                lse=lse,
                **bias_args,
            )
        torch.cuda.synchronize()
        return out

    lse = _nan_lse(problem)
    out = launch(problem, lse)
    off = launch(dataclasses.replace(problem, emit_lse=False), None)
    o_ref, lse_ref = _paged_reference(
        q, kc, vc, q_lens, kv_lens, bt, scale, 0.0, additive_bias=bias
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")
