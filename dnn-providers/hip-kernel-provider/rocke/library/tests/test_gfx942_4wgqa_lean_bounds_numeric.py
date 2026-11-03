# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guard for the gfx942 lean 4-warp body on odd sequence lengths.

The lean natural-QK body (head_size 192 and 256 bf16 causal prefill) walks 64-key
tiles and 128-row Q blocks. When a sequence length is not a tile multiple, the last
tile covers keys and rows past the sequence. The body must not read block-table
entries, K/V rows or Q rows for them: an out-of-sequence block-table entry holds an
arbitrary page id, and the K/V load built from it faults.

Each block-table row here carries extra columns filled with a page id whose K/V
offset lies about 2 GiB past the cache, so any read past a sequence's last page
faults deterministically instead of depending on what the allocator left there.
Output is checked against an fp32 causal reference. The structural CPU twin is
``TestLeanBodyBoundsReads`` in ``test_d256_gfx942_fast_routing.py``.
"""

from __future__ import annotations

import math

import pytest
from kernels.common import attention_unified as au


def _gpu_ready():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "mi300" in name or "gfx942" in name


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)

_PAD_COLS = 8


def _run(seqs, hq, hkv, d, bs, seed=0):
    import torch
    import torch.nn.functional as F

    dt = torch.bfloat16
    pages = [-(-k // bs) for _, k in seqs]
    nblocks = sum(pages) + 3
    g = torch.Generator(device="cuda").manual_seed(seed)
    total_q = sum(s for s, _ in seqs)
    q = torch.randn(total_q, hq, d, device="cuda", dtype=dt, generator=g)
    kc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=dt, generator=g)
    vc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=dt, generator=g)
    perm = torch.randperm(nblocks, device="cuda", generator=g).to(torch.int32)
    poison = (1 << 30) // (bs * hkv * d)
    bt = torch.full(
        (len(seqs), max(pages) + _PAD_COLS), poison, dtype=torch.int32, device="cuda"
    )
    p0 = 0
    for i, n in enumerate(pages):
        bt[i, :n] = perm[p0 : p0 + n]
        p0 += n
    cu = torch.tensor(
        [0] + [s for s, _ in seqs], dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    kl = torch.tensor([k for _, k in seqs], dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(d)
    problem = au.UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=len(seqs),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(s for s, _ in seqs),
        max_seqlen_k=max(k for _, k in seqs),
        dtype="bf16",
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        num_kv_blocks=nblocks,
    )
    assert (
        au._gfx942_4warp_route(problem, "gfx942") is not None
    ), "case must exercise the lean 4-warp body"
    out = torch.full_like(q, float("nan"))
    au.run_unified_attention_torch(
        problem=problem,
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=kl,
        softmax_scale=scale,
        block_table=bt,
        softcap=0.0,
        backend="auto",
    )
    torch.cuda.synchronize()
    rep = hq // hkv
    st = 0
    worst = 0.0
    for i, (sq, skv) in enumerate(seqs):
        idx = bt[i, : pages[i]].long()
        k = kc[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        v = vc[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        k = k.repeat_interleave(rep, 0)
        v = v.repeat_interleave(rep, 0)
        qs = q[st : st + sq].float().permute(1, 0, 2)
        ii = torch.arange(sq, device="cuda").view(sq, 1)
        jj = torch.arange(skv, device="cuda").view(1, skv)
        mask = torch.where(jj <= ii + (skv - sq), 0.0, float("-inf"))
        ref = F.scaled_dot_product_attention(
            qs, k, v, attn_mask=mask, scale=scale
        ).permute(1, 0, 2)
        o = out[st : st + sq].float()
        assert torch.isfinite(o).all(), f"seq {i}: non-finite output"
        worst = max(worst, (o - ref).abs().max().item())
        st += sq
    return worst


@requires_gfx942_gpu
@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.parametrize(
    "seqs,hq,hkv,bs",
    [
        ([(129, 129)], 16, 2, 16),
        ([(129, 129)], 16, 2, 32),
        ([(2, 2)], 12, 2, 16),
        ([(339, 367)], 40, 8, 16),
        ([(513, 513), (77, 2001), (1023, 1023)], 16, 16, 16),
    ],
)
def test_lean_body_odd_lengths(d, seqs, hq, hkv, bs):
    assert _run(seqs, hq, hkv, d, bs) <= 5e-2
