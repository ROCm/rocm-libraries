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


def run_dispatched_paged(
    arch, spec_id, *, window=1024, sq=2048, d=128, block_size=16, hq=16, hkv=4
):
    """Request -> dispatch -> bind -> launch of the paged spec ``spec_id`` (``None``:
    the unpinned auto route) on one fp16 sequence, with the block count taken from
    the cache at bind.

    Returns ``(result, q, k, v, out)``: the dispatch result, and ``k``/``v`` the
    contiguous tensors the cache was scattered from, for the caller's reference."""
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention

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
        kv_block_size=block_size,
    )
    result = dispatch_attention(
        request if spec_id is None else pinned(request, spec_id)
    )
    torch.manual_seed(0)
    q = torch.randn(1, sq, hq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, sq, hkv, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_tables, kv_lens = paged_cache(k, v, block_size)
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
    return result, q, k, v, out


def check_one_single_sequence_kernel_serves_two_caches(
    make_spec, run, reference, tolerance, launcher_for
):
    """Two single-sequence fp16 launches whose length and cache size differ run on
    one compiled kernel (``launcher_for(spec)`` returns the same launcher after
    both), and both match ``reference(q, k, v, scale)``. ``make_spec`` is the arch
    test's shipped-spec factory."""
    import dataclasses
    import math

    import torch

    d, hq, hkv, block_size = 128, 16, 4, 16
    scale = 1.0 / math.sqrt(d)
    launchers, sizes = [], []
    for sq in (1024, 2048):
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(1, sq, h, d, device="cuda", dtype=torch.float16)
            for h in (hq, hkv, hkv)
        )
        k_cache, v_cache, block_tables, kv_lens = paged_cache(k, v, block_size)
        spec = dataclasses.replace(
            make_spec(
                "fp16", d, hq, hkv, False, sq=sq, sk=sq, kv_block_size=block_size
            ),
            num_kv_blocks=k_cache.shape[0],
        )
        out = torch.full_like(q, float("nan"))
        run(
            spec=spec,
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            scale=scale,
            block_tables=block_tables,
            kv_lens=kv_lens,
        )
        torch.cuda.synchronize()
        max_abs = (reference(q, k, v, scale) - out.float()).abs().max().item()
        assert max_abs < tolerance("fp16"), f"sq{sq}: max_abs={max_abs:.3e}"
        sizes.append(k_cache.shape[0])
        launchers.append(launcher_for(spec))
    assert sizes[0] != sizes[1]
    assert launchers[0] is not None
    assert launchers[0] is launchers[1], "a second compile for another cache size"
