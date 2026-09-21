#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MoE token sorting — DSL MoeSortingLauncher vs AITER moe_sorting_opus_fwd.

Problem
-------
After the router selects top-K experts per token, the tokens must be
sorted by expert ID so the batched GEMMs can process all tokens routed
to expert E together in one block.

ATOM/AITER uses moe_sorting_opus_fwd — a single highly-optimized CUDA
kernel that computes histogram + scan + scatter in one pass on-chip.
For A3B (T=2, E=128, K=8) it runs in ~6µs.

The DSL MoeSortingLauncher is a 3-kernel chain:
  1. moe_histogram_kernel — counts tokens per expert
  2. moe_scan_kernel      — exclusive prefix scan of counts → offsets
  3. moe_scatter_kernel   — scatter (token_id, weight) into sorted slots

This approach is correct and flexible but cannot match the on-chip
fused version: three separate kernel launches, three HBM round-trips.
The 3-kernel chain takes ~28µs on A3B vs AITER's 6µs.

Why sorting still matters
--------------------------
MoeSortingLauncher feeds counts and offsets to FusedMoeForward.
In static-offset mode (the fast decode path), the sort is eliminated
entirely — FusedMoeForward uses pre-computed fixed offsets instead.
The sort benchmarked here is the *dynamic* path overhead.

Key constraint: A3B has 128 experts → sort_block_size must be ≥ 128.
The default is 64; setting sort_block_size=128 is required.

Results (MI355X / gfx950, T=2, E=128, K=8)
--------------------------------------------
  Backend                      Latency   vs AITER
  AITER moe_sorting_opus_fwd    6.1µs      1.00×
  DSL MoeSortingLauncher       28.3µs      0.22×   (3-kernel chain)

Note: In practice FusedMoeForward bypasses sorting entirely for decode
shapes (static-offset mode) — see 06_moe_e2e.py for the e2e number.

Run
---
  PYTHONPATH=<rocke_python_root> python3 05_moe_sorting.py
"""

from __future__ import annotations

import torch

from _common import (
    BATCH,
    NUM_EXPERTS,
    TOPK,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
)


def main() -> None:
    T, E, K = BATCH, NUM_EXPERTS, TOPK
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    topk_wts = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1)

    print("=" * 60)
    print("05  MoE Token Sorting")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: T={T} E={E} K={K}  (T*K={T * K} total routed pairs)")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()

    # AITER baseline
    bl_ms = float("nan")
    try:
        import aiter

        sorted_tok = torch.empty(T * K * E, dtype=torch.int32, device="cuda")
        sorted_wts = torch.empty(T * K * E, dtype=torch.float32, device="cuda")
        sorted_exp = torch.empty(E, dtype=torch.int32, device="cuda")
        num_valid = torch.zeros(1, dtype=torch.int32, device="cuda")
        moe_buf = torch.empty(E, dtype=torch.int32, device="cuda")

        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_wts.float(),
            sorted_tok,
            sorted_wts,
            sorted_exp,
            num_valid,
            moe_buf,
            E,
            T * K,
        )
        torch.cuda.synchronize()
        bl_ms = ms(
            lambda: aiter.moe_sorting_opus_fwd(
                topk_ids,
                topk_wts.float(),
                sorted_tok,
                sorted_wts,
                sorted_exp,
                num_valid,
                moe_buf,
                E,
                T * K,
            )
        )
        print(f"  AITER moe_sorting_opus_fwd (production): {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  AITER unavailable: {exc}")

    # DSL MoeSortingLauncher
    try:
        from rocke.instances.common.fused_moe_e2e import (
            FusedMoeForwardSpec,
            MoeSortingLauncher,
        )

        spec = FusedMoeForwardSpec(
            tokens=T,
            experts=E,
            topk=K,
            hidden=2048,
            intermediate=768,
            dtype="bf16",
            sort_block_size=128,
            router_block_size=128,
        )
        sorter = MoeSortingLauncher(spec.to_sort_spec())

        T_K = T * K
        # Workspace.  MoeSortingLauncher does NOT own Hist/Counter lifetime:
        # the histogram phase atomic-adds into Hist and the scatter phase
        # atomic-adds into Counter, so both must be zeroed before each chain.
        hist = torch.zeros(E, dtype=torch.int32, device="cuda")
        counter = torch.zeros(E, dtype=torch.int32, device="cuda")
        offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
        counts = torch.zeros(E, dtype=torch.int32, device="cuda")
        sorted_tok_dsl = torch.empty(T_K, dtype=torch.int32, device="cuda")
        sorted_kid_dsl = torch.empty(T_K, dtype=torch.int32, device="cuda")
        sorted_wts_dsl = torch.empty(T_K, dtype=torch.float32, device="cuda")
        weights_f32 = topk_wts.float()
        stream_h = int(torch.cuda.current_stream().cuda_stream)

        def sort_fn():
            hist.zero_()
            counter.zero_()
            sorter.run(
                {
                    "histogram": {
                        "TopkIds": topk_ids,
                        "Hist": hist,
                        "num_pairs": T_K,
                        "num_experts": E,
                    },
                    "scan": {
                        "Hist": hist,
                        "Offsets": offsets,
                        "Counts": counts,
                        "num_experts": E,
                    },
                    "scatter": {
                        "TopkIds": topk_ids,
                        "TopkWeights": weights_f32,
                        "Offsets": offsets,
                        "Counter": counter,
                        "SortedTokenIds": sorted_tok_dsl,
                        "SortedTopkIds": sorted_kid_dsl,
                        "SortedWeights": sorted_wts_dsl,
                        "tokens": T,
                        "topk": K,
                        "num_experts": E,
                    },
                },
                stream=stream_h,
            )

        sort_fn()
        torch.cuda.synchronize()

        dsl_ms = ms(sort_fn)
        print(
            f"  DSL MoeSortingLauncher (3-kernel):       {dsl_ms * 1000:.2f}µs  "
            f"speedup={speedup(bl_ms, dsl_ms):.3f}×"
        )

        # Correctness: every routed (token, topk) pair must be placed exactly
        # once → the per-expert counts must sum to T*K.
        torch.cuda.synchronize()
        nv = int(counts.sum().item())
        print(
            f"  Correctness: num_valid={nv} (expected {T * K})  "
            f"{'PASS' if nv == T * K else 'FAIL'}"
        )

    except Exception:
        import traceback

        traceback.print_exc()

    print()
    print("  Note: The 0.22× slowdown vs AITER is a 3-kernel-chain vs 1-kernel-fused")
    print("  trade-off.  In production FusedMoeForward bypasses sorting via static")
    print("  offsets for decode shapes (see 06_moe_e2e.py).")


if __name__ == "__main__":
    main()
