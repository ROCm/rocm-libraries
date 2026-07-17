#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end fused MoE forward — DSL FusedMoeForward vs AITER fused_moe.

Problem
-------
The MoE forward pass computes:
  Y = sum_{k in topk} w_k * down_k(silu(gate_k(X)) * up_k(X))

where gate_k, up_k, down_k are per-expert linear projections.

ATOM/AITER uses aiter.fused_moe — a highly optimized 2-stage CK kernel
pipeline (ck_moe_stage1 for gate+up+silu, ck_moe_stage2 for down+reduce)
that runs in ~101µs for A3B shapes (T=2, E=128, K=8, H=2048, I=768).

The DSL FusedMoeForward pipeline is:
  topk_softmax  → routing weights and IDs
  (sort skip)   → static offsets eliminate the sort kernel for decode
  gather        → scatter X into expert-grouped buffer
  interleaved gate+up GEMM → fused (gate_i, up_i) column pairs + SiLU
  down GEMM     → per-expert projection back to hidden
  topk reduce   → weighted sum into Y

Key optimizations applied to reach 1.10×
------------------------------------------
1. BF16 tile selection
   gfx950 BF16 only supports 16×16 MFMA atoms (not 32×32).  The default
   tile uses a 32×32×16 atom (F16-only) → garbage output on BF16 tensors.
   Fixed by _default_bf16_gemm_tile(): warp_tile=(16,16,32), warp_m=2,
   warp_n=2, giving tile_m=32, tile_n=32, tile_k=32, block_size=256.

2. FP16/BF16 dtype mismatch bug (now fixed)
   BatchedGemmSpec.to_universal_spec() previously defaulted DataSpec()
   to dtype_a=dtype_b=dtype_c="fp16".  Reading BF16 bits as FP16 gives
   values ~1e36 (finite, non-NaN in BF16) — a silent correctness bug.
   Fix: all GEMM spec classes now carry a dtype field threaded through
   DataSpec.

3. Static-offset mode (skip the sort)
   For decode (T=2, E=128, K=8: T*K*E=2048), the histogram+scan+scatter
   sort takes 28µs by itself.  Static-offset mode pre-computes fixed
   offsets [0, slot_size, 2*slot_size, ...] so the sort is never
   launched.  slot_size=1 gives minimal waste for sparse routing.

4. Active-tile skip
   With only T*K=16 active (token, expert) pairs out of E=128 possible
   expert slots, 87.5% of GEMM tiles are empty.  active_tile_skip_gemms
   uses a SortedTokenIds sentinel (=-1 for empty slots) so the GEMM
   kernel skips all-empty tiles entirely.

5. CUDA-graph capture
   The entire DSL pipeline (topk → gather → GEMM → reduce) is captured
   into one HIP graph.  Replay cost is ~0.5µs vs ~15µs dispatch overhead
   for the multi-kernel chain.

6. 128-expert sort_block_size
   A3B has 128 experts; the default sort_block_size=64 would assert.
   Setting sort_block_size=128 fixes this.

Results (MI355X / gfx950, T=2, E=128, K=8, H=2048, I=768, bf16)
-----------------------------------------------------------------
  Backend                      Latency   Speedup
  AITER fused_moe (2-stage CK) 101.3µs    1.00×
  DSL FusedMoeForward + graph   92.3µs    1.10×

Run
---
  PYTHONPATH=<rocke_python_root> python3 06_moe_e2e.py
"""

from __future__ import annotations

import torch

from _common import (
    BATCH,
    NUM_EXPERTS,
    TOPK,
    HIDDEN,
    MOE_INTER,
    DTYPE,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
)


def main() -> None:
    T, E, K, H, Inter = BATCH, NUM_EXPERTS, TOPK, HIDDEN, MOE_INTER
    torch.manual_seed(42)

    # Weight tensors
    W1 = torch.randn(E, 2 * Inter, H, dtype=DTYPE, device="cuda") * 0.01
    W_gate = W1[:, :Inter, :].contiguous()
    W_up = W1[:, Inter:, :].contiguous()
    W_down = torch.randn(E, H, Inter, dtype=DTYPE, device="cuda") * 0.01
    X = torch.randn(T, H, dtype=DTYPE, device="cuda") * 0.1
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")
    Y = torch.zeros(T, H, dtype=DTYPE, device="cuda")

    print("=" * 60)
    print("06  Fused MoE Forward — end-to-end pipeline")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: T={T} E={E} K={K} H={H} I={Inter}  dtype=bf16")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()

    # AITER fused_moe baseline
    bl_ms = float("nan")
    try:
        from aiter.fused_moe import fused_moe as aiter_fm
        from aiter import ActivationType, QuantType

        tv, ti = torch.topk(logits, K, dim=-1)
        tw = torch.softmax(tv, dim=-1).float()
        ti = ti.int()

        out_bl = aiter_fm(
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
        print(f"  AITER fused_moe (2-stage CK):        {bl_ms * 1000:.2f}µs")
        print(
            f"  AITER output: max={out_bl.abs().max():.4e}  "
            f"{'finite' if out_bl.isfinite().all() else 'NON-FINITE'}"
        )
    except Exception as exc:
        print(f"  AITER unavailable: {exc}")

    # DSL FusedMoeForward
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
            sort_block_size=128,  # required: A3B has 128 experts
            router_block_size=128,
            preshuffle_w_gate_up_interleaved=True,
            active_tile_skip_gemms=True,  # skip empty expert tiles
        )
        fwd = FusedMoeForward(spec)
        print(f"  DSL tile: {fwd.spec.gemm_tile}")
        print(
            f"  static_offsets: {fwd._use_static_offsets} "
            f"(forcing True for decode shape)"
        )

        # Force static-offset mode: skip the sort
        fwd._use_static_offsets = True
        fwd._static_slot_size = 1  # minimal slot waste for sparse T*K=16

        # Compile warmup
        Y.zero_()
        fwd.forward(
            routing_logits=logits, X=X, W_gate=W_gate, W_up=W_up, W_down=W_down, Y=Y
        )
        torch.cuda.synchronize()
        print(
            f"  DSL output (compile): max={Y.abs().max():.4e}  "
            f"{'finite' if Y.isfinite().all() else 'NON-FINITE'}"
        )

        # CUDA-graph capture
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
            graph_ok = Y.isfinite().all().item() and Y.abs().max().item() > 1e-8
        except Exception as ge:
            print(f"  CUDA graph failed: {ge}")

        if graph_ok:
            dsl_ms = ms(fwd.replay_graph, warmup=WARMUP, iters=ITERS, repeats=REPEATS)
            mode = "graph"
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

            dsl_ms = ms(dsl_fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS)
            mode = "dynamic"

        spd = speedup(bl_ms, dsl_ms)
        print(
            f"  DSL FusedMoeForward [{mode}]:        {dsl_ms * 1000:.2f}µs  "
            f"speedup={spd:.3f}×"
        )

    except Exception:
        import traceback

        traceback.print_exc()

    print()
    print("  Key insight: static-offset + active_tile_skip eliminates sorting")
    print("  and skips 112/128 empty expert GEMM tiles — saves ~30µs.")
    print("  CUDA graph capture removes ~15µs of Python/HIP dispatch overhead.")


if __name__ == "__main__":
    main()
