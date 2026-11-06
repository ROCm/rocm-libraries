# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ALiBi in the unified paged attention kernels (tiled 2D and split-KV 3D) on
gfx942 / gfx950, vs an FP32 paged reference.

The term is ``slope * (key - query)`` with the query at its diagonal position,
hipDNN's definition on the top-left diagonal; there the LSE is checked too. Slopes
are built in (hipDNN's per-head formula) or read from a caller tensor.
"""

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from rocke.numeric.sdpa_reference import alibi_slopes  # noqa: E402


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


_ARCH = _arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH is None, reason="needs a gfx942 or gfx950 GPU"),
]


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _reference(q, k_cache, v_cache, q_lens, kv_lens, block_table, scale, slopes, tl):
    """FP32 O and natural-log LSE; causal on the bottom-right diagonal, or the
    top-left one when ``tl``; ALiBi measured from the query's diagonal."""
    hq, d = q.shape[1], q.shape[2]
    hkv, bs = k_cache.shape[2], k_cache.shape[1]
    slope = slopes.float().view(hq, 1, 1)
    outs, lses, start = [], [], 0
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q[start : start + ql].float().transpose(0, 1)
        blocks = block_table[i, : (kl + bs - 1) // bs].long()
        k = k_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        v = v_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        k = k.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        v = v.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        s = torch.matmul(qi, k.transpose(-1, -2)) * scale
        p = torch.arange(ql, device=q.device).view(-1, 1)
        kk = torch.arange(kl, device=q.device).view(1, -1)
        diag = p if tl else p + (kl - ql)
        s = s + slope * (kk - diag)
        s = s.masked_fill(~(kk <= diag), float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        has = torch.isfinite(lse)
        prob = torch.exp(s - torch.where(has, lse, 0.0).unsqueeze(-1))
        prob = torch.where(has.unsqueeze(-1), prob, 0.0)
        outs.append(torch.matmul(prob, v).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
        start += ql
    return torch.cat(outs), torch.cat(lses).unsqueeze(-1)


def _run_case(
    q_lens,
    kv_lens,
    *,
    backend,
    hq=12,
    hkv=4,
    d=128,
    dtype="bf16",
    top_left=False,
    lse=False,
    external=False,
    seed=0
):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    bs = 16
    total_q, num_seqs = sum(q_lens), len(q_lens)
    max_blocks = max(1, max((kl + bs - 1) // bs for kl in kv_lens))
    num_blocks = num_seqs * max_blocks + 3
    q = torch.randn(total_q, hq, d, device="cuda", dtype=tdt, generator=g)
    kc = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    vc = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[
        : num_seqs * max_blocks
    ]
    bt = perm.view(num_seqs, max_blocks).to(torch.int32).contiguous()
    cu = torch.tensor(
        [0, *itertools.accumulate(q_lens)], dtype=torch.int32, device="cuda"
    )
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    slopes = torch.tensor(alibi_slopes(hq), dtype=torch.float32, device="cuda")
    if external:
        # Not the built-in vector, so a kernel ignoring the tensor fails.
        slopes = slopes.flip(0).contiguous()
    problem = UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=dtype,
        num_kv_blocks=num_blocks,
        use_alibi=True,
        causal_top_left=top_left,
        right_bound=0,
        emit_lse=lse,
    )
    out = torch.empty_like(q)
    lse_t = torch.full((total_q, hq, 1), float("nan"), device="cuda") if lse else None
    scale = 1.0 / math.sqrt(d)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=scale,
        block_table=bt,
        softcap=0.0,
        backend=backend,
        alibi_slopes=slopes if external else None,
        lse=lse_t,
    )
    torch.cuda.synchronize()
    o_ref, lse_ref = _reference(q, kc, vc, q_lens, kv_lens, bt, scale, slopes, top_left)
    tol = _tolerance(dtype)
    assert (out.float() - o_ref).abs().max().item() < tol
    if lse:
        finite = torch.isfinite(lse_ref)
        assert (lse_t[finite] - lse_ref[finite]).abs().max().item() < tol


_SHAPES = [
    ([1, 1, 1, 1], [4096, 777, 1500, 2049]),  # decode, ragged KV
    ([4, 2], [3000, 1029]),  # spec-decode
    ([128], [128]),  # prefill, s_q == s_kv
    ([96, 64], [300, 64]),  # chunked prefill
]


@pytest.mark.parametrize("external", [False, True], ids=["builtin", "external"])
@pytest.mark.parametrize("backend", ["tiled", "3d"])
@pytest.mark.parametrize("q_lens,kv_lens", _SHAPES)
def test_bottom_right(backend, q_lens, kv_lens, external):
    _run_case(q_lens, kv_lens, backend=backend, external=external)


@pytest.mark.parametrize("backend", ["tiled", "3d"])
@pytest.mark.parametrize("q_lens,kv_lens", [([128], [128]), ([96, 64], [300, 64])])
def test_top_left_with_lse(backend, q_lens, kv_lens):
    if _ARCH != "gfx950":
        pytest.skip("top-left alignment is gfx950-only in the unified kernels")
    _run_case(q_lens, kv_lens, backend=backend, top_left=True, lse=True)


def test_bottom_right_lse_is_refused():
    with pytest.raises((ValueError, NotImplementedError), match="top-left"):
        _run_case([4], [64], backend="tiled", lse=True)
