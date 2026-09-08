# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared torch flash-attention reference timer for the parity/bench harnesses.

Reconstructs the dense per-sequence causal Q/K/V from a paged KV cache (so the
numbers match the reference paged attention), then times torch's
``F.scaled_dot_product_attention`` (flash backend) on it. Kept arch-agnostic and
decoupled from any harness-local ``Scenario``/``data`` shape so the gfx942/gfx950
harnesses can consume one implementation instead of near-identical copies.
"""
from __future__ import annotations

from typing import Callable


def run_torch_flash(
    *,
    query,
    key_cache,
    value_cache,
    block_tables,
    query_lens,
    kv_lens_list,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    scale: float,
    warmup: int,
    attempts: int,
    bench_stream: int,
    time_lane_ms: Callable,
):
    """Time torch F.sdpa (flash) on the same shape as the paged reference.

    ``time_lane_ms(call_once, *, warmup, attempts, stream)`` and ``bench_stream``
    are injected by the caller so this stays free of harness-local timing infra.
    Returns ``(out[total_q, H, D], ms, backend_name)``.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    H = num_query_heads
    Hkv = num_kv_heads
    D = head_size
    dev = query.device
    nqpkv = H // Hkv

    # Gather dense per-seq tensors once (outside the timed region).
    seqs = []
    start = 0
    for i, (ql, kvl) in enumerate(zip(query_lens, kv_lens_list)):
        if ql <= 0:
            continue
        q_i = query[start : start + ql]  # [ql, H, D]
        start += ql
        nblk = (kvl + block_size - 1) // block_size
        blk = block_tables[i, :nblk]
        k_i = key_cache[blk].reshape(-1, Hkv, D)[:kvl]  # [kvl, Hkv, D]
        v_i = value_cache[blk].reshape(-1, Hkv, D)[:kvl]
        # [1, H, L, D]
        qh = q_i.permute(1, 0, 2).unsqueeze(0)
        kh = k_i.permute(1, 0, 2).unsqueeze(0)
        vh = v_i.permute(1, 0, 2).unsqueeze(0)
        if nqpkv > 1:
            kh = kh.repeat_interleave(nqpkv, dim=1)
            vh = vh.repeat_interleave(nqpkv, dim=1)
        seqs.append((qh, kh, vh, ql, kvl))

    # Every sequence was empty (all query_lens <= 0) -> nothing to time. Fail
    # explicitly here rather than letting the seqs[0] backend probe below
    # IndexError into a misleading "no working torch sdpa backend".
    if not seqs:
        raise ValueError(
            "run_torch_flash: no sequence with query_len > 0 "
            f"(query_lens={list(query_lens)}); nothing to time"
        )

    # Pick a working flash-first backend once.
    backend = None
    for be, name in (
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "efficient"),
        (SDPBackend.MATH, "math"),
    ):
        try:
            with sdpa_kernel(be):
                _ = F.scaled_dot_product_attention(
                    seqs[0][0][:, :, :64],
                    seqs[0][1][:, :, :64],
                    seqs[0][2][:, :, :64],
                    is_causal=True,
                    scale=scale,
                )
            backend = (be, name)
            break
        except Exception:  # noqa: BLE001
            continue
    if backend is None:
        raise NotImplementedError("no working torch sdpa backend")

    outs = [None] * len(seqs)

    def call_once():
        with sdpa_kernel(backend[0]):
            for j, (qh, kh, vh, ql, kvl) in enumerate(seqs):
                if ql == kvl:
                    o = F.scaled_dot_product_attention(
                        qh, kh, vh, is_causal=True, scale=scale
                    )
                else:
                    # query is a suffix of the kv stream (aiter convention)
                    mask = torch.ones(ql, kvl, device=dev, dtype=torch.bool)
                    mask = torch.triu(mask, diagonal=kvl - ql + 1).logical_not()
                    o = F.scaled_dot_product_attention(
                        qh, kh, vh, attn_mask=mask, scale=scale
                    )
                outs[j] = o

    ms = time_lane_ms(call_once, warmup=warmup, attempts=attempts, stream=bench_stream)

    # Assemble [total_q, H, D] in scenario order for the correctness compare.
    flat = [o.squeeze(0).permute(1, 0, 2) for o in outs]  # each [L, H, D]
    out = torch.cat(flat, dim=0).to(query.dtype)
    return out, ms, backend[1]
