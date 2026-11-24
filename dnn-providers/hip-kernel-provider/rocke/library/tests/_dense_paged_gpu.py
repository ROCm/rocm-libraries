# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU helpers the per-arch dense paged numeric tests share: a page cache scattered
from contiguous K/V, and a pinned paged spec run through dispatch, bind and launch."""

from __future__ import annotations

import numpy as np

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray
from ._dense_paged_stand_ins import pinned


def paged_cache(rt, k, v, block_size, dtype, *, seed=0):
    """Scatter one sequence's K/V (fp32 ``[1, Sk, Hkv, D]`` arrays) into pages at
    permuted ids of a cache twice the needed size whose unused pages are NaN, so a
    wrong page or in-page row reads a NaN page or the wrong keys rather than a near
    miss. Returns device ``(k_cache, v_cache, block_tables, kv_lens)``."""
    _, sk, hkv, d = k.shape
    npages = sk // block_size
    phys = np.random.default_rng(seed).permutation(2 * npages)[:npages]
    k_cache = np.full((2 * npages, block_size, hkv, d), np.nan, dtype=np.float32)
    v_cache = np.full_like(k_cache, np.nan)
    k_cache[phys] = k[0].reshape(npages, block_size, hkv, d)
    v_cache[phys] = v[0].reshape(npages, block_size, hkv, d)
    block_tables = phys.astype(np.int32).reshape(1, npages)
    kv_lens = np.array([sk], dtype=np.int32)
    return (
        an.to_device(rt, k_cache, dtype),
        an.to_device(rt, v_cache, dtype),
        DeviceArray(rt, block_tables),
        DeviceArray(rt, kv_lens),
    )


def run_dispatched_paged(rt, arch, spec_id, *, window=1024):
    """Request -> dispatch -> bind -> launch of the paged spec ``spec_id`` on one
    fp16 D128 sequence, with the block count taken from the cache at bind.

    Returns fp32 ``(q, k, v, out)``; ``q``/``k``/``v`` are the contiguous arrays the
    cache was scattered from, for the caller's reference."""
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
    rng = np.random.default_rng(0)
    q = an.randn(rng, (1, sq, hq, d), "fp16")
    k = an.randn(rng, (1, sq, hkv, d), "fp16")
    v = an.randn(rng, (1, sq, hkv, d), "fp16")
    k_cache, v_cache, block_tables, kv_lens = paged_cache(rt, k, v, 16, "fp16")
    out = an.empty_device(rt, q.shape, "fp16")
    result.bind_torch(
        {
            "q": an.to_device(rt, q, "fp16"),
            "k": k_cache,
            "v": v_cache,
            "out": out,
            "block_tables": block_tables,
            "kv_lens": kv_lens,
        }
    ).launch()
    return q, k, v, an.read(out, "fp16")
