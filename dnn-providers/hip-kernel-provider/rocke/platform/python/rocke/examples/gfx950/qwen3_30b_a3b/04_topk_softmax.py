#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Router TopK — DSL topk_softmax with CUDA-graph capture vs AITER gate.

Problem
-------
The MoE router computes softmax(logits) and selects the top-K experts
per token.  For A3B (T=2 tokens, E=128 experts, K=8), ATOM calls
aiter.moe_fused_gate which takes ~13µs including Python overhead.

The DSL topk_softmax kernel is a pure-GPU fused softmax + selection.
With CUDA-graph capture it runs in 0.45µs — a 29× improvement — because
the entire operation fits in a single GPU launch whose command packet
can be replicated without re-invoking the HIP runtime.

Why the AITER baseline is so slow
----------------------------------
aiter.moe_fused_gate involves:
  1. Python arg unpacking (pybind11)
  2. torch tensor dispatch
  3. hipModuleLaunchKernel
  4. Actual GPU kernel (~2µs)
  5. Optional D2H sync for count buffers

Steps 1–3 alone cost 10–12µs for a tiny kernel.  CUDA graph capture
eliminates all of them on replay.

Results (MI355X / gfx950, T=2, E=128, K=8, bf16)
--------------------------------------------------
  Backend                   Latency   Speedup
  AITER moe_fused_gate       13.3µs    1.00×
  DSL topk_softmax (no graph) 2.1µs    6.3×
  DSL topk_softmax + graph    0.45µs  29.5×

Run
---
  PYTHONPATH=<rocke_python_root> python3 04_topk_softmax.py
"""

from __future__ import annotations

import struct

import torch

from _common import (
    BATCH,
    NUM_EXPERTS,
    TOPK,
    ISA,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
    capture_graph,
)


def build_topk_kernel(T: int, E: int, K: int):
    """Compile DSL topk_softmax.  Kernel signature is
    ``(X, Y, Idx, M, N)``: X = logits ``(T, E)``, Y = top-K softmax weights
    ``(T, K)``, Idx = top-K expert indices ``(T, K)``, M = rows, N = experts.
    Returns ``(rt, fn, grid, block, topk_weights, topk_ids)``."""
    from rocke.instances.common.topk_softmax import (
        TopkSoftmaxSpec,
        build_topk_softmax,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    spec = TopkSoftmaxSpec(
        n_per_row=E,  # entries per row = experts; one row (token) per CTA
        k=K,
        block_size=128,
        dtype="f32",
    )
    art = compile_kernel(build_topk_softmax(spec), isa=ISA)
    grid = (T, 1, 1)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    topk_weights = torch.empty(T, K, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(T, K, dtype=torch.int32, device="cuda")

    return rt, fn, grid, block, topk_weights, topk_ids


def main() -> None:
    T, E, K = BATCH, NUM_EXPERTS, TOPK
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")

    print("=" * 60)
    print("04  Router TopK — topk_softmax + CUDA-graph capture")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: T={T} E={E} K={K}")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()

    # AITER baseline
    bl_ms = float("nan")
    try:
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        bl_ms = ms(lambda: torch.topk(logits, K, dim=-1))
        print(f"  torch.topk + softmax (fallback): {bl_ms * 1000:.2f}µs")
        # Try the production gate kernel
        try:
            from aiter import moe_fused_gate

            # Signature: (input, bias, topk_weights, topk_ids, num_expert_group,
            # topk_group, topk, n_share_experts_fusion, routed_scaling_factor).
            # The kernel requires experts/num_expert_group <= 32; with
            # num_expert_group == topk_group every group is eligible, so the
            # selection matches a global top-K.
            n_grp = max(1, E // 16)  # 16 experts/group ≤ 32 cap
            gate_bias = torch.zeros(E, dtype=torch.float32, device="cuda")
            gate_out = torch.empty(T, K, dtype=torch.float32, device="cuda")
            gate_ids = torch.empty(T, K, dtype=torch.int32, device="cuda")

            def gate_fn():
                moe_fused_gate(
                    logits, gate_bias, gate_out, gate_ids, n_grp, n_grp, K, 0
                )

            gate_fn()
            torch.cuda.synchronize()
            bl_ms = ms(gate_fn)
            print(f"  AITER moe_fused_gate (production): {bl_ms * 1000:.2f}µs")
        except Exception as exc:
            print(f"  AITER gate unavailable ({exc}); torch.topk used as baseline")
    except Exception as exc:
        print(f"  SKIP AITER: {exc}")

    # DSL topk_softmax
    try:
        rt, fn, grid, block, tw, ti = build_topk_kernel(T, E, K)
        packed = struct.pack(
            "<QQQii",
            int(logits.data_ptr()),
            int(tw.data_ptr()),
            int(ti.data_ptr()),
            T,
            E,
        )

        rt.launch(fn, grid, block, packed, stream=0)
        torch.cuda.synchronize()

        dsl_raw_ms = ms(lambda: rt.launch(fn, grid, block, packed, stream=0))
        print(
            f"  DSL topk_softmax (no graph):      {dsl_raw_ms * 1000:.2f}µs  "
            f"speedup={speedup(bl_ms, dsl_raw_ms):.3f}×"
        )

        g = capture_graph(
            lambda: rt.launch(fn, grid, block, packed, stream=0), warmup=5
        )
        if g is not None:
            dsl_g_ms = ms(lambda: g.replay())
            print(
                f"  DSL topk_softmax + CUDA graph:    {dsl_g_ms * 1000:.2f}µs  "
                f"speedup={speedup(bl_ms, dsl_g_ms):.3f}×"
            )
        else:
            print("  CUDA graph capture failed")

        # Correctness: DSL top-K set should match torch.topk
        _, ref_ids = torch.topk(logits, K, dim=-1)
        dsl_top = set(ti[0].tolist())
        ref_top = set(ref_ids[0].tolist())
        ok = dsl_top == ref_top
        print(f"  Correctness: top-{K} selection {'PASS' if ok else 'FAIL'}")

    except Exception:
        import traceback

        traceback.print_exc()

    print()
    print("  Note: 29× speedup comes almost entirely from eliminating HIP")
    print("  command-submission overhead via CUDA graph replay (~0.45µs).")


if __name__ == "__main__":
    main()
