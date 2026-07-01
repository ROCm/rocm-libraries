# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Qwen3-30B-A3B decode/prefill attention benchmarks — DSL vs AITER Triton.

Attention-specific harnesses excised from the platform Amdahl scripts
(07_full_decode_step.py and qwen3_30b_a3b_decode.py).  Platform files keep
the non-attention layers; this module owns the attention coverage.

Run::

    python3 -m builders.gfx950.qwen3_30b_a3b.decode_attention_bench
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import List

import torch

from rocke.examples.gfx950.qwen3_30b_a3b._common import (
    BATCH,
    BLOCK_SIZE,
    DTYPE,
    HEAD_DIM,
    ITERS,
    NHEAD_K,
    NHEAD_Q,
    REPEATS,
    WARMUP,
    ms,
    speedup,
)

NaN = float("nan")
DECODE_KV = 1024


# ── Result containers ──────────────────────────────────────────────────────────


@dataclass
class AttentionResult:
    name: str
    baseline_us: float
    dsl_us: float

    @property
    def speedup(self) -> float:
        return speedup(self.baseline_us / 1e3, self.dsl_us / 1e3)


@dataclass
class LayerResult:
    name: str
    baseline_us: float
    dsl_us: float
    spd: float
    notes: str = ""


# ── Decode attention benchmark (matches 07_full_decode_step.py logic) ─────────


def bench_decode_attn_simple(kv_len: int = DECODE_KV) -> AttentionResult:
    """Sweep num_sms to find best DSL vs AITER Triton for paged-KV decode.

    Returns an AttentionResult with baseline/dsl timings in µs.
    """
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
    cu_q = torch.arange(0, BATCH + 1, dtype=torch.int32, device="cuda")
    kv_l = torch.full((BATCH,), kv_len, dtype=torch.int32, device="cuda")
    bt = torch.randint(0, pool, (BATCH, num_blks), dtype=torch.int32, device="cuda")
    stream_h = int(torch.cuda.current_stream().cuda_stream)

    bl_ms = NaN
    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri,
        )

        out_bl = torch.empty_like(q)

        def tri_fn():
            tri(
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
    except Exception:
        pass

    dsl_ms = NaN
    try:
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        out_dsl = torch.empty_like(q)
        best = float("inf")
        for sms in [30, 60, 80, 120]:
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
                    num_sms=sms,
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
                if t < best:
                    best = t
            except Exception:
                pass
        dsl_ms = best
    except Exception:
        traceback.print_exc()

    return AttentionResult(
        f"decode_attn kv={kv_len}",
        bl_ms * 1000,
        dsl_ms * 1000,
    )


# ── Full decode attention benchmark (matches qwen3_30b_a3b_decode.py logic) ───


def _ms(fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS) -> float:
    from statistics import median

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters / 1e3)
    return median(samples)


def _spd(bl, dsl):
    if bl != bl or dsl != dsl or dsl == 0:
        return NaN
    return bl / dsl


def bench_decode_attn(kv_len: int = DECODE_KV) -> LayerResult:
    """Full decode attention benchmark: sweeps num_sms, reports best DSL vs AITER."""
    num_seqs = BATCH
    total_q = num_seqs
    num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = num_blks * num_seqs + 64

    q = torch.randn(total_q, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
    kc = (
        torch.randn(pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda")
        * 0.1
    )
    vc = torch.randn_like(kc)
    cu_q = torch.tensor([0, num_seqs], dtype=torch.int32, device="cuda")
    kv_lens = torch.full((num_seqs,), kv_len, dtype=torch.int32, device="cuda")
    bt = torch.randint(0, pool, (num_seqs, num_blks), dtype=torch.int32, device="cuda")
    scale = HEAD_DIM**-0.5
    stream_h = int(torch.cuda.current_stream().cuda_stream)
    print(f"\n[Decode attn] kv={kv_len}")

    # Production baseline: AITER Triton unified_attention
    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )

        out_tri = torch.empty_like(q)

        def tri_fn():
            tri_attn(
                q=q,
                k=kc,
                v=vc,
                out=out_tri,
                cu_seqlens_q=cu_q,
                seqused_k=kv_lens,
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
                backend="triton",
            )

        bl_ms = _ms(tri_fn)
        print(f"  AITER Triton paged-decode: {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  SKIP AITER: {exc}")
        bl_ms = NaN

    # DSL: unified_attention — sweep num_sms
    try:
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        best_dsl_ms = float("inf")
        best_sms = 60
        best_path = "n/a"
        out2 = torch.empty_like(q)
        for num_sms in [30, 60, 80, 120, 152, 304]:
            try:
                prob = UnifiedAttentionProblem(
                    total_q=total_q,
                    num_seqs=num_seqs,
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
                        out=out2,
                        cu_seqlens_q=cu_q,
                        seqused_k=kv_lens,
                        softmax_scale=scale,
                        block_table=bt,
                        softcap=0.0,
                        stream=stream_h,
                    )

                for _ in range(3):
                    dsl_fn()
                torch.cuda.synchronize()
                t = _ms(dsl_fn)
                path = prob.select_path()
                if t < best_dsl_ms:
                    best_dsl_ms = t
                    best_sms = num_sms
                    best_path = path
            except Exception:
                pass
        dsl_ms = best_dsl_ms
        print(
            f"  DSL unified({best_path}, sms={best_sms}):  "
            f"{dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN
        best_sms = 60
        best_path = "n/a"

    return LayerResult(
        "decode_attn",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"kv={kv_len} GQA-8 hdim={HEAD_DIM} path={best_path} num_sms={best_sms}",
    )


# ── Prefill attention benchmark (matches qwen3_30b_a3b_decode.py logic) ───────


def bench_prefill_attn() -> List[LayerResult]:
    """Prefill attention benchmark with per-sql tuned configs vs AITER Triton."""
    print("\n[Prefill Attention] — DSL vs AITER Triton paged-prefill")
    results: List[LayerResult] = []
    stream_h = int(torch.cuda.current_stream().cuda_stream)

    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

    except Exception as exc:
        print(f"  Cannot import attention kernels: {exc}")
        return results

    # Per-sql tuned configurations discovered by sweep against AITER Triton paged-prefill
    # on MI355X / gfx950 (bf16, hd=64, nhq=32, nhk=4, BS=16, num_seqs=1).
    # Each entry is (num_warps, block_m_per_warp, tile_size_mult_of_BS, use_transposed_qk_32x32).
    PREFILL_CFG = {
        128: (1, 16, 8, False),
        256: (2, 16, 8, False),
        512: (2, 32, 4, True),
        1024: (4, 32, 4, True),
        2048: (4, 32, 8, True),
    }

    from kernels.common import attention_unified as _au

    _orig_nw = _au._select_2d_num_warps
    _orig_mw = _au._select_2d_block_m_per_warp
    _orig_T = _au._select_2d_tile_size
    _orig_tr = _au._enable_transposed_qk_32x32
    _orig_pv = _au._enable_register_pv

    def _apply_cfg(nw_v, mw_v, t_mult, trans):
        _au._select_2d_num_warps = lambda p, _n=nw_v: _n
        _au._select_2d_block_m_per_warp = lambda p, _m=mw_v: _m
        _au._select_2d_tile_size = lambda p, _t=t_mult: _t * p.block_size
        if trans:
            _au._enable_transposed_qk_32x32 = lambda p, _m=mw_v: (
                p.dtype == "bf16" and p.head_size in (64, 128) and _m == 32
            )
            _au._enable_register_pv = lambda p: False
        else:
            _au._enable_transposed_qk_32x32 = lambda p: False
            _au._enable_register_pv = _orig_pv
        _au._ATTN_TILED_CACHE.clear()
        _au._2D_LAUNCHERS.clear()

    def _restore_cfg():
        _au._select_2d_num_warps = _orig_nw
        _au._select_2d_block_m_per_warp = _orig_mw
        _au._select_2d_tile_size = _orig_T
        _au._enable_transposed_qk_32x32 = _orig_tr
        _au._enable_register_pv = _orig_pv
        _au._ATTN_TILED_CACHE.clear()
        _au._2D_LAUNCHERS.clear()

    try:
        for sql in [128, 256, 512, 1024, 2048]:
            pool = 2048
            nblk = (sql + BLOCK_SIZE - 1) // BLOCK_SIZE
            q = torch.randn(sql, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
            kc = (
                torch.randn(
                    pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda"
                )
                * 0.1
            )
            vc = torch.randn_like(kc)
            bt = torch.randint(0, pool, (1, nblk), dtype=torch.int32, device="cuda")
            cu_q = torch.tensor([0, sql], dtype=torch.int32, device="cuda")
            kvl = torch.tensor([sql], dtype=torch.int32, device="cuda")
            scale = HEAD_DIM**-0.5
            out = torch.empty_like(q)
            out2 = torch.empty_like(q)

            def tri():
                tri_attn(
                    q=q,
                    k=kc,
                    v=vc,
                    out=out,
                    cu_seqlens_q=cu_q,
                    seqused_k=kvl,
                    max_seqlen_q=sql,
                    max_seqlen_k=sql,
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
                    backend="triton",
                )

            for _ in range(5):
                tri()
                torch.cuda.synchronize()

            try:
                bl_ms = _ms(tri)
            except Exception:
                bl_ms = NaN

            nw_v, mw_v, t_mult, trans = PREFILL_CFG[sql]
            _apply_cfg(nw_v, mw_v, t_mult, trans)
            cfg_tag = f"nw{nw_v}_mw{mw_v}_T{t_mult}{'_tr' if trans else ''}"

            best_dsl = float("inf")
            for num_sms in [120]:
                try:
                    prob = UnifiedAttentionProblem(
                        total_q=sql,
                        num_seqs=1,
                        num_query_heads=NHEAD_Q,
                        num_kv_heads=NHEAD_K,
                        head_size=HEAD_DIM,
                        block_size=BLOCK_SIZE,
                        max_seqlen_q=sql,
                        max_seqlen_k=sql,
                        dtype="bf16",
                        num_sms=num_sms,
                    )

                    def dsl_fn():
                        run_unified_attention_torch(
                            problem=prob,
                            q=q,
                            k=kc,
                            v=vc,
                            out=out2,
                            cu_seqlens_q=cu_q,
                            seqused_k=kvl,
                            softmax_scale=scale,
                            block_table=bt,
                            softcap=0.0,
                            stream=stream_h,
                        )

                    for _ in range(5):
                        dsl_fn()
                        torch.cuda.synchronize()
                    dsl_us = _ms(dsl_fn) * 1000
                    if dsl_us < best_dsl:
                        best_dsl = dsl_us
                except Exception:
                    pass

            bl_us = bl_ms * 1000
            spd = _spd(bl_us, best_dsl)
            flag = "  *** REGRESSION ***" if spd == spd and spd < 0.95 else ""
            print(
                f"  sq={sql:5d}: AITER={bl_us:.2f}µs  DSL({cfg_tag})={best_dsl:.2f}µs  "
                f"spd={spd:.3f}x{flag}"
            )
            results.append(
                LayerResult(
                    f"prefill_sq{sql}", bl_us, best_dsl, spd, f"sq={sql} cfg={cfg_tag}"
                )
            )
    finally:
        _restore_cfg()
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B attention benchmarks (DSL vs AITER Triton)."
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "prefill", "all"],
        default="all",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=DECODE_KV,
        help="KV sequence length for decode benchmark.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "No GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: Qwen3-30B-A3B  batch={BATCH}  bf16  head_dim={HEAD_DIM}")

    if args.mode in ("decode", "all"):
        r = bench_decode_attn(kv_len=args.kv_len)
        print(
            f"\n  decode_attn: baseline={r.baseline_us:.2f}µs  "
            f"dsl={r.dsl_us:.2f}µs  speedup={r.spd:.3f}x"
        )

    if args.mode in ("prefill", "all"):
        bench_prefill_attn()


if __name__ == "__main__":
    main()
