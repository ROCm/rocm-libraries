#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Paged-KV decode attention — DSL unified 3D split-KV vs AITER Triton.

Problem
-------
ATOM's decode attention path (when use_flash_layout=True) calls AITER's
Triton unified_attention.  For A3B decode (batch=2, nhead_q=32, nhead_k=4,
head_dim=64, block_size=16), the DSL 3D split-KV kernel is at parity.

Algorithm: 3D split-KV
-----------------------
The attention problem is split along the KV-sequence dimension.  Each
CTA processes a chunk of kv_len keys/values, writes a partial softmax
accumulator and log-sum-exp to a scratch buffer, then a second reduction
pass merges the partials.

num_sms controls how many CTAs participate in the split.  Too few leaves
compute stranded; too many create merge overhead.  The script sweeps
num_sms = {30, 60, 80, 120, 152, 304} and picks the best.

Why parity (not a win) for A3B
--------------------------------
head_dim=64 is half the typical 128.  The 3D kernel's tile is sized for
head_dim=128 on gfx950; at head_dim=64 the MFMA utilization is lower and
the kernel becomes bandwidth-bound sooner.  At kv_len=1024 both DSL and
Triton deliver ~95% of parity.

Results (MI355X / gfx950, batch=2, nhead_q=32, nhead_k=4, head_dim=64)
------------------------------------------------------------------------
  kv_len   AITER Triton   DSL 3d (best)   speedup
     512       51.4µs        52.6µs        0.977×
    1024       51.9µs        53.3µs        0.973×
    2048       66.3µs        67.5µs        0.982×
    4096       93.2µs        94.1µs        0.990×

Run
---
  PYTHONPATH=<rocke_python_root> python3 03_decode_attention.py
"""

from __future__ import annotations

import torch

from _common import (
    BATCH,
    DTYPE,
    HEAD_DIM,
    NHEAD_K,
    NHEAD_Q,
    BLOCK_SIZE,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
)


def bench_kv_len(kv_len: int) -> None:
    num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = num_blks * BATCH + 64
    scale = HEAD_DIM**-0.5

    q = torch.randn(BATCH, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
    kc = (
        torch.randn(pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda")
        * 0.1
    )
    vc = torch.randn_like(kc)
    # Decode: one query token per sequence → cu_seqlens_q = [0, 1, ..., BATCH].
    # (A plain [0, BATCH] declares a *single* length-BATCH sequence, which
    # mismatches the BATCH-row block_table / seqused_k and reads cu_q[BATCH]
    # out of bounds inside the kernel — an intermittent GPU memory fault.)
    cu_q = torch.arange(0, BATCH + 1, dtype=torch.int32, device="cuda")
    kv_l = torch.full((BATCH,), kv_len, dtype=torch.int32, device="cuda")
    bt = torch.randint(0, pool, (BATCH, num_blks), dtype=torch.int32, device="cuda")
    stream_h = int(torch.cuda.current_stream().cuda_stream)

    print(f"\n  kv_len={kv_len}")

    # AITER Triton baseline
    bl_ms = float("nan")
    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )

        out_bl = torch.empty_like(q)

        def tri_fn():
            tri_attn(
                q=q,
                k=kc,
                v=vc,
                out=out_bl,
                cu_seqlens_q=cu_q,
                seqused_k=kv_l,
                max_seqlen_q=1,
                max_seqlen_k=kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                block_table=bt,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=None,
                qq_bias=None,
                sinks=None,
            )

        bl_ms = ms(tri_fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS)
        print(f"    AITER Triton:       {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"    AITER unavailable: {exc}")

    # DSL unified attention — sweep num_sms
    try:
        from rocke.instances import (
            UnifiedAttentionProblem,
            run_unified_attention_torch,
        )

        out_dsl = torch.empty_like(q)
        best_ms = float("inf")
        best_sms = 60

        for num_sms in [30, 60, 80, 120, 152, 304]:
            try:
                prob = UnifiedAttentionProblem(
                    total_q=BATCH,
                    num_seqs=BATCH,
                    num_query_heads=NHEAD_Q,
                    num_kv_heads=NHEAD_K,
                    head_size=HEAD_DIM,
                    block_size=BLOCK_SIZE,
                    max_seqlen_q=1,
                    max_seqlen_k=kv_len,
                    dtype="bf16",
                    num_sms=num_sms,
                )

                def dsl_fn():
                    run_unified_attention_torch(
                        problem=prob,
                        q=q,
                        k=kc,
                        v=vc,
                        out=out_dsl,
                        cu_seqlens_q=cu_q,
                        seqused_k=kv_l,
                        softmax_scale=scale,
                        block_table=bt,
                        softcap=0.0,
                        stream=stream_h,
                    )

                for _ in range(3):
                    dsl_fn()
                torch.cuda.synchronize()
                t = ms(dsl_fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS)
                if t < best_ms:
                    best_ms, best_sms = t, num_sms
            except Exception:
                pass

        spd = speedup(bl_ms, best_ms)
        print(
            f"    DSL 3d sms={best_sms:3d}:    {best_ms * 1000:.2f}µs  speedup={spd:.3f}×"
        )
    except Exception as exc:
        print(f"    DSL unavailable: {exc}")


def main() -> None:
    print("=" * 60)
    print("03  Paged-KV Decode Attention")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: batch={BATCH} nhead_q={NHEAD_Q} nhead_k={NHEAD_K} head_dim={HEAD_DIM}"
    )
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()
    print("Note: A3B head_dim=64 is smaller than the typical 128 the 3D kernel")
    print("is tuned for.  Both kernels are bandwidth-bound → near parity.")

    for kv_len in [512, 1024, 2048, 4096]:
        bench_kv_len(kv_len)


if __name__ == "__main__":
    main()
