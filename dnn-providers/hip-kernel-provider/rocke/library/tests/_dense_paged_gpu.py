# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU helpers the per-arch dense paged numeric tests share: a page cache scattered
from contiguous K/V, and a pinned paged spec run through dispatch, bind and launch."""

from __future__ import annotations

from ._dense_paged_stand_ins import pinned


def paged_cache(k, v, block_size):
    """Scatter one sequence's K/V into pages at permuted ids of a cache twice the
    needed size whose unused pages are NaN, so a wrong page or in-page row reads a
    NaN page or the wrong keys rather than a near miss."""
    import torch

    _, sk, hkv, d = k.shape
    npages = sk // block_size
    phys = torch.randperm(2 * npages, device=k.device)[:npages]
    k_cache = torch.full(
        (2 * npages, block_size, hkv, d), float("nan"), device=k.device, dtype=k.dtype
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    k_cache[phys] = k[0].view(npages, block_size, hkv, d)
    v_cache[phys] = v[0].view(npages, block_size, hkv, d)
    block_tables = phys.to(torch.int32).view(1, npages).contiguous()
    kv_lens = torch.tensor([sk], device=k.device, dtype=torch.int32)
    return k_cache, v_cache, block_tables, kv_lens


def run_dispatched_paged(arch, spec_id, *, window=1024):
    """Request -> dispatch -> bind -> launch of the paged spec ``spec_id`` on one
    fp16 D128 sequence, with the block count taken from the cache at bind.

    Returns ``(q, k, v, out)``; ``k``/``v`` are the contiguous tensors the cache was
    scattered from, for the caller's reference."""
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention

    sq, hq, hkv, d = 2048, 16, 4, 128
    request = AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sq,
        hdim_q=d,
        hdim_v=d,
        arch=arch,
        mask_type=1,
        dtype="fp16",
        sliding_window=window,
        layout="paged",
        kv_block_size=16,
    )
    result = dispatch_attention(pinned(request, spec_id))
    torch.manual_seed(0)
    q = torch.randn(1, sq, hq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, sq, hkv, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_tables, kv_lens = paged_cache(k, v, 16)
    out = torch.empty_like(q)
    result.bind_torch(
        {
            "q": q,
            "k": k_cache,
            "v": v_cache,
            "out": out,
            "block_tables": block_tables,
            "kv_lens": kv_lens,
        }
    ).launch()
    torch.cuda.synchronize()
    return q, k, v, out
