# gfx950 Deep Fusion — Consolidated Learnings & Improvements (2026-06-02 → 2026-06-03)

Synthesis of **all** 2026-06-* analysis notes for the gfx950 CK DSL deep-fusion
prototype: a single fused kernel computing
`conv0 3x3 -> ReLU -> conv1 1x1 -> ReLU -> 2x2 s2 maxpool`, fp16 in/weights,
fp32 MFMA accumulation, no HBM intermediates.

> **Bottom line:** the prototype went from a **~143 useful TFLOP/s** starting
> baseline (2026-06-02) to **~284–287 useful TFLOP/s (~0.178 ms)** after the
> 2026-06-03 optimization sequence — a **~2.0× speedup**, all bit-stable. The
> ~280 TFLOP/s figure is the achieved result; the ~132–143 numbers below are the
> *starting point*, not the destination.

Source docs (chronological):

- `2026-06-02-1723-gfx950-deep-fusion-experiments.md`
- `2026-06-02-1723-gfx950-roofline-bottlenecks.md`
- `2026-06-02-1725-gfx950-async-direct-probe-results.md`
- `2026-06-03-0406-gfx950-rocprof-baseline-counters.md`
- `2026-06-03-0439-gfx950-conv1-lds-vectorization-results.md`
- `2026-06-03-0447-gfx950-cshuffle-store-vectorization-results.md`
- `2026-06-03-1416-gfx950-barrier-merge-results.md`
- `2026-06-03-1620-gfx950-best-config-rocprof-counters.md`
- `2026-06-03-1753-gfx950-leverC-rocprof-and-async.md`
- `2026-06-03-1848-gfx950-leverD-mdecode-bypass.md`

## Prototype Under Test

| Item | Value |
|------|-------|
| Graph | concat(C=8) → conv0 3x3 K0=32 → ReLU → conv1 1x1 K1=24 → ReLU → 2x2 s2 maxpool |
| Full shape | in `[1,2160,3840,8]` → pool out `[1,1080,1920,24]` |
| MFMA atom | fp16 `32x32x16` (`mfma_f32_32x32x16_f16`) — confirmed selected, not scalar fallback |
| Accumulation | fp32 |
| Layout | NHWC in / NHWK out |
| CTA ownership | each CTA owns a final pooled tile, expands back through conv1/conv0 halo locally (no inter-CTA comms, no HBM intermediates) |
| Correctness | NumPy ref @ tol 1e-2; final max_abs_diff = 0.00195312, bad = 0/49766400 |

## Performance Trajectory (full target shape)

This is the core result. Each row is a cumulative step; the speedup column is
relative to the 2026-06-02 baseline.

| # | Step | Time (ms) | Useful TFLOP/s | Δ vs prev | Cum. speedup | Mechanism |
|---|------|-----------|----------------|-----------|--------------|-----------|
| 0 | Baseline (`mem`, tk16, pool_tile 4x8) | ~0.357 | ~143 | — | 1.00× | tuned starting config |
| 1 | conv1 LDS read vectorization | ~0.253 | ~201 | +41% | 1.41× | conv1 LDS reads → single `ds_read_b128`, dropped dead K-mask |
| 2 | Barrier merge (3→2) | ~0.246 | ~207 | +3% | 1.45× | disjoint LDS tiles let W1 load overlap cshuffle |
| 3 | Config switch (pool_tile 4x4, tile_m 64, tk32) | ~0.224 | ~228 | +10% | 1.59× | more padded MFMA but VALU/CTA 1720→690 |
| 4 | Lever A — defer conv1 epilogue past pool | ~0.219 | ~233 | +2% | 1.63× | `relu(max)=max(relu)`, fewer epilogue ops |
| 5 | Lever B — vectorize maxpool gather | ~0.218 | ~234 | +0.4% | 1.63× | `ds_read2_b64` for pool window |
| 6 | Lever C — eliminate conv1→pool LDS handoff | ~0.184 | ~277 | +18% | 1.94× | register-resident intra-lane pool, no LDS round-trip |
| 7 | Lever D — bypass A-descriptor m-decode | ~0.178 | ~284–287 | +2.7% | ~2.0× | `decompose_m=False`, drop magic-const ÷Wo/÷Ho round-trip |

Final: **~0.178 ms, ~284–287 useful TFLOP/s, bit-identical to step 6.**

## Bottleneck Evolution

| Phase | Limiter | Evidence |
|-------|---------|----------|
| 2026-06-02 hypothesis | MFMA operand delivery / LDS staging | ISA-mix deltas on regressions |
| 2026-06-03 baseline (counters) | **VALU-bound** (not MFMA, not HBM) | MfmaUtil 6.2%, VALUBusy 47.7%, VALU:MFMA 61:1, MemUnitStalled 0.06% |
| After conv1 LDS vec | LDS wait largely cleared | LDS wait 52%→10%, latency 125→57 cyc |
| Lever C / D (final) | **still VALU-bound** | VALUBusy ~63%, VALU/LDS reduced but VALU stays critical path |

Key reframe from 2026-06-02 → 2026-06-03: the real limiter was **not** HBM
bandwidth nor MFMA throughput, but **scalar/VALU work** (coordinate arithmetic,
LDS staging, sync) feeding a lightly-utilized MFMA unit. Every winning lever cut
VALU/LDS work on the critical path rather than adding compute.

## Roofline / Padding Accounting

| Quantity | Value | Note |
|----------|-------|------|
| conv0 useful FLOPs | 38.22 G | 2·P·K0·R·S·C |
| conv1 useful FLOPs | 12.74 G | 2·P·K1·K0 |
| total useful FLOPs | 50.96 G | pool comparisons (149 M) excluded from TFLOP/s |
| total HW-padded FLOPs | 59.45 G | rectangular MFMA tiles actually issued (baseline tk16) |
| useful / hardware ratio | **85.7%** | ~16.7% excess MFMA work at baseline |
| padding source 1 | conv0 K: 72 → 80 | tk16 rounds K_gemm=R·S·C=72 to 80 |
| padding source 2 | conv1 N: K1=24 → 32 | output-channel pad to tile_n |

**Counterintuitive config finding (step 3):** moving to pool_tile 4x4 / tile_m
64 / tk32 *increased* padded MFMA work yet ran faster, because it slashed
per-CTA VALU (1720→690). Confirms the limiter is VALU, not MFMA padding — minimal
padded-FLOPs is the wrong thing to optimize for on this topology.

## Dead-Ends & Correct Negatives

| Approach | Result | Why it failed |
|----------|--------|---------------|
| input-footprint LDS cache | regressed 0.357→0.539 ms | balloons VALU/ds_read/waitcnt/barrier around same MFMA count |
| direct footprint (LDS→MFMA) | regressed; **fails at full shape** (max_abs_diff 0.637) | scalar per-fragment gathers kill operand delivery; incorrect at scale |
| `async_dma=True` | wrong output + 27% slower | nothing to hide — memory pipes already idle; broken for fused carrier |
| `unroll_k=True` | race (max_abs_diff ~0.881) | drops the K-tile barrier |
| cshuffle store vectorization | **impossible** (no code change) | MFMA C-fragment is same-col/diff-row stride-32; row-major store needed by reads — fundamentally unvectorizable |

These are valuable: they bound the design space. The cshuffle-store and
async results are *correct negatives* — proven not worth revisiting.

## Key Learnings

| # | Learning | Evidence |
|---|----------|----------|
| 1 | Single-kernel fusion is **correct** across toy → full 2160×3840, no HBM intermediates | bit-stable max_abs_diff 0.00195312 through all 7 steps |
| 2 | Limiter is **VALU/scalar work**, not HBM bandwidth nor MFMA throughput | MfmaUtil 6.2%, VALUBusy 47.7%, MemUnitStalled 0.06% |
| 3 | Biggest single win was eliminating the **conv1→pool LDS handoff** (lever C, +18%) | register-resident intra-lane pool, 0.218→0.184 ms |
| 4 | conv1 LDS read vectorization was the biggest early win (+41%) | single `ds_read_b128`, LDS wait 52%→10% |
| 5 | Schedule-only changes (barrier merge) are nearly free wins | 3→2 barriers, +3%, pure overlap |
| 6 | Optimize for **VALU**, not padded-MFMA minimization | step-3 config trades more padding for less VALU and wins |
| 7 | Occupancy is **not** the deciding factor | direct-footprint had best waves/CU yet lost |
| 8 | m-decode round-trip (÷Wo/÷Ho magic constants) was pure overhead | lever D bypass, +2.7%, bit-identical |
| 9 | Some transforms are provably impossible/unsafe (cshuffle-store, async, unroll_k) | correct negatives above |
| 10 | Hard ceiling: 85.7% useful/HW from padding tax | roofline accounting |

## Confirmed Defaults

| Change | Status | Rationale |
|--------|--------|-----------|
| conv1 LDS read vectorization | **kept** | +41%, clears LDS wait |
| barrier merge (3→2) | **kept** | free schedule win |
| pool_tile 4x4 / tile_m 64 / tk32 | **kept** | minimizes VALU (the real limiter) |
| lever A (deferred conv1 epilogue) | **kept** | `relu(max)=max(relu)` |
| lever B (vectorized maxpool gather) | **kept** | `ds_read2_b64` |
| lever C (register-resident pool) | **kept** | +18%, eliminates conv1→pool handoff |
| lever D (`decompose_m=False`) | **kept** | +2.7%, drops magic-const m-decode |
| `pipeline=mem`, `async_dma=False` | **kept** | async broken + slower |
| input-footprint / direct footprint | **rejected** | regress / incorrect at full shape |
| `unroll_k`, cshuffle-store-vec | **rejected** | race / provably impossible |

## Remaining Open Levers

| Lever | Status | Note |
|-------|--------|------|
| conv0→conv1 LDS handoff (handoff #1) | **open, hard** | inherent M↔K0 transpose; not cheaply eliminable like handoff #2 was |
| further VALU reduction | open | still VALU-bound at ~63% busy; coordinate arithmetic remains the critical path |
| compare vs unfused multi-kernel fp16 pipeline | open | quantify true end-to-end fusion win |
| int8/int4 MFMA + packing, true virtual concat, production autotune | out of scope | path to production graph beyond this fp16 proof |
