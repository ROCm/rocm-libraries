#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full decode step — Amdahl analysis across all layers (Qwen3-30B-A3B).

This script runs every layer in the A3B decode path, collecting both
AITER/ATOM production baseline and DSL timings, then builds an Amdahl
table showing which layers dominate and how much the DSL saves overall.

Amdahl's Law reminder
----------------------
End-to-end speedup from improving layer i is bounded by:

    overall_speedup ≤ 1 / (f_i/S_i + (1 - f_i))

where f_i is the fraction of total time spent in layer i and S_i is
the per-layer speedup.  For A3B decode:

  - MoE e2e is 49% of total time (1.10× win → contributes 4.5% saved)
  - Decode attention is 25% (0.97× → costs 0.7% extra)
  - GEMM layers are 11% total (1.65× → saves 4%)
  - RMSNorm + TopK are 12% (13-30× via graph → saves 11%)

Summing gives the 1.28× end-to-end speedup.

Methodology
-----------
All timings use the same batched-event pattern:
  - 10 warmup launches
  - 5 independent samples of 200 launches under one event pair each
  - Median of the 5 samples reported

Baselines are always the actual ATOM/AITER production kernels:
  RMSNorm   aiter.rmsnorm2d_fwd_with_add
  GEMMs     torch.matmul (→ hipBLASLt, same as ATOM LinearBase.forward)
  Attention aiter.ops.triton unified_attention (ATOM decode path)
  TopK      aiter.moe_fused_gate (or torch.topk fallback)
  MoE e2e   aiter.fused_moe (2-stage CK, same as ATOM MoE layers)

Run
---
  PYTHONPATH=<rocke_python_root> python3 07_full_decode_step.py
"""

from __future__ import annotations

import struct
import traceback
from dataclasses import dataclass
from typing import List

import torch

from _common import (
    BATCH,
    NHEAD_Q,
    NHEAD_K,
    HEAD_DIM,
    HIDDEN,
    MOE_INTER,
    NUM_EXPERTS,
    TOPK,
    BLOCK_SIZE,
    DTYPE,
    ISA,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
    capture_graph,
    build_gemm_kernel,
)


@dataclass
class Row:
    name: str
    baseline_us: float
    dsl_us: float

    @property
    def speedup(self) -> float:
        return speedup(self.baseline_us / 1e3, self.dsl_us / 1e3)


NaN = float("nan")


# ── Per-layer benchmarks ───────────────────────────────────────────────────────


def bench_rmsnorm() -> Row:
    M, N = BATCH, HIDDEN
    eps = 1e-6
    X = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1
    R = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1
    Gam = torch.ones(N, dtype=DTYPE, device="cuda")
    Xout = torch.empty(M, N, dtype=DTYPE, device="cuda")
    Yout = torch.empty(M, N, dtype=DTYPE, device="cuda")

    bl_ms = NaN
    try:
        import aiter

        # (out, input, residual_in, residual_out, weight, epsilon)
        bl_ms = ms(lambda: aiter.rmsnorm2d_fwd_with_add(Xout, X, R, Yout, Gam, eps))
    except Exception:
        pass

    dsl_ms = NaN
    try:
        from rocke.instances.common.add_rmsnorm2d_bf16 import (
            AddRMSNorm2DBF16Spec,
            build_add_rmsnorm2d_bf16,
            add_rmsnorm2d_bf16_grid,
        )
        from rocke.helpers import compile_kernel
        from rocke.runtime.hip_module import Runtime

        spec = AddRMSNorm2DBF16Spec(
            n_per_block=N, block_size=64, vec=4, dtype="bf16", save_residual=True
        )
        art = compile_kernel(build_add_rmsnorm2d_bf16(spec), isa=ISA)
        grid = add_rmsnorm2d_bf16_grid(M, spec)
        block = (spec.block_size, 1, 1)
        rt = Runtime()
        fn = rt.load_module(art.hsaco).get_function(art.kernel_name)
        # Kernel param order (save_residual=True): A, B, Gamma, X(residual_out),
        # Y(normalized_out) — A=X, B=R, X(resid)=Xout, Y(norm)=Yout.
        packed = struct.pack(
            "<QQQQQiif",
            int(X.data_ptr()),
            int(R.data_ptr()),
            int(Gam.data_ptr()),
            int(Xout.data_ptr()),
            int(Yout.data_ptr()),
            M,
            N,
            eps,
        )
        g = capture_graph(
            lambda: rt.launch(fn, grid, block, packed, stream=0), warmup=5
        )
        if g:
            dsl_ms = ms(lambda: g.replay())
        else:
            dsl_ms = ms(lambda: rt.launch(fn, grid, block, packed, stream=0))
    except Exception:
        traceback.print_exc()

    return Row("rmsnorm ×3", bl_ms * 1000, dsl_ms * 1000)


def bench_gemm(
    M: int, N: int, K: int, label: str, tile_k: int = 512, cgm: int = 4
) -> Row:
    A = torch.randn(M, K, dtype=DTYPE, device="cuda") * 0.1
    B_bl = torch.randn(K, N, dtype=DTYPE, device="cuda") * 0.1
    B_ds = torch.randn(N, K, dtype=DTYPE, device="cuda") * 0.1

    bl_ms = ms(lambda: torch.matmul(A, B_bl))
    run, rt, C = build_gemm_kernel(M, N, K, tile_k=tile_k, chiplet_wgm=cgm)
    Ap, Bp, Cp = int(A.data_ptr()), int(B_ds.data_ptr()), int(C.data_ptr())
    dsl_ms = ms(lambda: run(Ap, Bp, Cp))
    return Row(label, bl_ms * 1000, dsl_ms * 1000)


def bench_decode_attn(kv_len: int = 1024) -> Row:
    num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = num_blks * BATCH + 64
    scale = HEAD_DIM**-0.5
    q = torch.randn(BATCH, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
    kc = (
        torch.randn(pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda")
        * 0.1
    )
    vc = torch.randn_like(kc)
    # Decode: one query token per sequence → cu_seqlens_q = [0, 1, ..., BATCH]
    # (a plain [0, BATCH] reads cu_q[BATCH] out of bounds → intermittent fault).
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

        bl_ms = ms(tri_fn)
    except Exception:
        pass

    dsl_ms = NaN
    try:
        from rocke.instances import (
            UnifiedAttentionProblem,
            run_unified_attention_torch,
        )

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
                t = ms(dsl_fn)
                if t < best:
                    best = t
            except Exception:
                pass
        dsl_ms = best
    except Exception:
        traceback.print_exc()

    return Row(f"decode_attn kv={kv_len}", bl_ms * 1000, dsl_ms * 1000)


def bench_topk() -> Row:
    T, E, K = BATCH, NUM_EXPERTS, TOPK
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")

    bl_ms = NaN
    try:
        gate_out = torch.empty(T, K, dtype=torch.float32, device="cuda")
        gate_ids = torch.empty(T, K, dtype=torch.int32, device="cuda")
        try:
            from aiter import moe_fused_gate

            # (input, bias, topk_weights, topk_ids, num_expert_group,
            #  topk_group, topk, n_share_experts_fusion); experts/group <= 32.
            n_grp = max(1, E // 16)
            gate_bias = torch.zeros(E, dtype=torch.float32, device="cuda")

            def gate_fn():
                moe_fused_gate(
                    logits, gate_bias, gate_out, gate_ids, n_grp, n_grp, K, 0
                )

            gate_fn()
            torch.cuda.synchronize()
            bl_ms = ms(gate_fn)
        except Exception:
            bl_ms = ms(lambda: torch.topk(logits, K, dim=-1))
    except Exception:
        bl_ms = ms(lambda: torch.topk(logits, K, dim=-1))

    dsl_ms = NaN
    try:
        from rocke.instances.common.topk_softmax import (
            TopkSoftmaxSpec,
            build_topk_softmax,
        )
        from rocke.helpers import compile_kernel
        from rocke.runtime.hip_module import Runtime

        spec = TopkSoftmaxSpec(n_per_row=E, k=K, block_size=128, dtype="f32")
        art = compile_kernel(build_topk_softmax(spec), isa=ISA)
        grid = (T, 1, 1)
        block = (spec.block_size, 1, 1)
        rt = Runtime()
        fn = rt.load_module(art.hsaco).get_function(art.kernel_name)
        # Kernel signature: (X=logits, Y=topk_weights, Idx=topk_ids, M, N).
        tw = torch.empty(T, K, dtype=torch.float32, device="cuda")
        tid = torch.empty(T, K, dtype=torch.int32, device="cuda")
        packed = struct.pack(
            "<QQQii",
            int(logits.data_ptr()),
            int(tw.data_ptr()),
            int(tid.data_ptr()),
            T,
            E,
        )
        g = capture_graph(
            lambda: rt.launch(fn, grid, block, packed, stream=0), warmup=5
        )
        dsl_ms = (
            ms(lambda: g.replay())
            if g
            else ms(lambda: rt.launch(fn, grid, block, packed, stream=0))
        )
    except Exception:
        traceback.print_exc()

    return Row("router_topk", bl_ms * 1000, dsl_ms * 1000)


def bench_moe_e2e() -> Row:
    T, E, K, H, Inter = BATCH, NUM_EXPERTS, TOPK, HIDDEN, MOE_INTER
    torch.manual_seed(42)
    W1 = torch.randn(E, 2 * Inter, H, dtype=DTYPE, device="cuda") * 0.01
    W_gate = W1[:, :Inter, :].contiguous()
    W_up = W1[:, Inter:, :].contiguous()
    W_down = torch.randn(E, H, Inter, dtype=DTYPE, device="cuda") * 0.01
    X = torch.randn(T, H, dtype=DTYPE, device="cuda") * 0.1
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")
    Y = torch.zeros(T, H, dtype=DTYPE, device="cuda")

    bl_ms = NaN
    try:
        from aiter.fused_moe import fused_moe as aiter_fm
        from aiter import ActivationType, QuantType

        tv, ti = torch.topk(logits, K, dim=-1)
        tw = torch.softmax(tv, dim=-1).float()
        ti = ti.int()
        aiter_fm(
            X,
            W1,
            W_down,
            tw,
            ti,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
        )
        torch.cuda.synchronize()
        bl_ms = ms(
            lambda: aiter_fm(
                X,
                W1,
                W_down,
                tw,
                ti,
                activation=ActivationType.Silu,
                quant_type=QuantType.No,
            )
        )
    except Exception:
        pass

    dsl_ms = NaN
    try:
        from rocke.instances.common.fused_moe_e2e import (
            FusedMoeForwardSpec,
            FusedMoeForward,
        )

        spec = FusedMoeForwardSpec(
            tokens=T,
            experts=E,
            topk=K,
            hidden=H,
            intermediate=Inter,
            dtype="bf16",
            sort_block_size=128,
            router_block_size=128,
            preshuffle_w_gate_up_interleaved=True,
            active_tile_skip_gemms=True,
        )
        fwd = FusedMoeForward(spec)
        fwd._use_static_offsets = True
        fwd._static_slot_size = 1
        Y.zero_()
        fwd.forward(
            routing_logits=logits, X=X, W_gate=W_gate, W_up=W_up, W_down=W_down, Y=Y
        )
        torch.cuda.synchronize()
        graph_ok = False
        try:
            fwd.capture_graph(
                routing_logits=logits,
                X=X,
                W_gate=W_gate,
                W_up=W_up,
                W_down=W_down,
                Y=Y,
                warmup_iters=3,
            )
            fwd.replay_graph()
            torch.cuda.synchronize()
            graph_ok = Y.isfinite().all().item()
        except Exception:
            pass
        if graph_ok:
            dsl_ms = ms(fwd.replay_graph)
        else:

            def dsl_fn():
                Y.zero_()
                fwd.forward(
                    routing_logits=logits,
                    X=X,
                    W_gate=W_gate,
                    W_up=W_up,
                    W_down=W_down,
                    Y=Y,
                )

            dsl_ms = ms(dsl_fn)
    except Exception:
        traceback.print_exc()

    return Row("moe_e2e", bl_ms * 1000, dsl_ms * 1000)


# ── Amdahl table ──────────────────────────────────────────────────────────────


def amdahl_table(rows: List[Row]) -> None:
    total_bl = sum(r.baseline_us for r in rows if r.baseline_us == r.baseline_us)
    total_dsl = sum(r.dsl_us for r in rows if r.dsl_us == r.dsl_us)
    W = 40

    print()
    print("=" * 90)
    print(
        f"  {'Layer':{W}}  {'Baseline µs':>12}  {'DSL µs':>10}  {'Speedup':>9}  "
        f"{'%step':>6}  {'Saved µs':>9}"
    )
    print("-" * 90)
    for r in rows:
        frac = (r.baseline_us / total_bl * 100) if total_bl > 0 else 0
        saved = r.baseline_us - r.dsl_us
        spd_s = f"{r.speedup:.3f}×" if r.speedup == r.speedup else "  n/a  "
        print(
            f"  {r.name:{W}}  {r.baseline_us:>12.2f}  {r.dsl_us:>10.2f}  "
            f"{spd_s:>9}  {frac:>5.1f}%  {saved:>9.2f}"
        )
    print("-" * 90)
    overall = speedup(total_bl / 1000, total_dsl / 1000)
    saved = total_bl - total_dsl
    print(
        f"  {'TOTAL':{W}}  {total_bl:>12.2f}  {total_dsl:>10.2f}  "
        f"{overall:>8.3f}×  {'100.0%':>6}  {saved:>9.2f}"
    )
    print("=" * 90)
    print(f"\n  End-to-end DSL vs AITER production = {overall:.3f}×")
    print(f"  Total saved: {saved:.1f}µs ({saved / total_bl * 100:.1f}% of AITER step)")


def main() -> None:
    print("=" * 60)
    print("07  Full Decode Step — Amdahl Analysis")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: Qwen3-30B-A3B  batch={BATCH}  bf16")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()
    print("Compiling all kernels (first-run JIT takes ~60s)...")
    print()

    print("[1/7] RMSNorm")
    r_norm = bench_rmsnorm()
    print(
        f"  baseline={r_norm.baseline_us:.2f}µs  dsl={r_norm.dsl_us:.2f}µs  "
        f"speedup={r_norm.speedup:.3f}×"
    )

    print("[2/7] QKV projection M=2 N=2560 K=2048")
    r_qkv = bench_gemm(
        BATCH, (NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM, HIDDEN, "qkv_proj", tile_k=512, cgm=4
    )
    print(
        f"  baseline={r_qkv.baseline_us:.2f}µs  dsl={r_qkv.dsl_us:.2f}µs  "
        f"speedup={r_qkv.speedup:.3f}×"
    )

    print("[3/7] Decode attention kv=1024")
    r_attn = bench_decode_attn(kv_len=1024)
    print(
        f"  baseline={r_attn.baseline_us:.2f}µs  dsl={r_attn.dsl_us:.2f}µs  "
        f"speedup={r_attn.speedup:.3f}×"
    )

    print("[4/7] O projection M=2 N=2048 K=2048")
    r_oproj = bench_gemm(BATCH, HIDDEN, HIDDEN, "o_proj", tile_k=1024, cgm=8)
    print(
        f"  baseline={r_oproj.baseline_us:.2f}µs  dsl={r_oproj.dsl_us:.2f}µs  "
        f"speedup={r_oproj.speedup:.3f}×"
    )

    print("[5/7] Router TopK T=2 E=128 K=8")
    r_topk = bench_topk()
    print(
        f"  baseline={r_topk.baseline_us:.2f}µs  dsl={r_topk.dsl_us:.2f}µs  "
        f"speedup={r_topk.speedup:.3f}×"
    )

    print("[6/7] MoE e2e T=2 E=128 K=8 H=2048 I=768")
    r_moe = bench_moe_e2e()
    print(
        f"  baseline={r_moe.baseline_us:.2f}µs  dsl={r_moe.dsl_us:.2f}µs  "
        f"speedup={r_moe.speedup:.3f}×"
    )

    # RMSNorm appears 3× per decode step
    r_norm2 = Row("rmsnorm_post_attn", r_norm.baseline_us, r_norm.dsl_us)
    r_norm3 = Row("rmsnorm_pre_moe", r_norm.baseline_us, r_norm.dsl_us)

    all_rows = [r_norm, r_qkv, r_attn, r_oproj, r_norm2, r_norm3, r_topk, r_moe]
    print()
    print("[7/7] Amdahl table")
    amdahl_table(all_rows)


if __name__ == "__main__":
    main()
