# FP32 gfx1100 Optimization Iterations Diary

## Iteration 0 — Baseline (completed)
- Non-MI tuning campaign: 1,030 shapes, peak 24.1 TFLOPS (6144x8192)
- VOPD diagonal pairing: works but only +3-4% (DRAM-bound at MT128x64)

## Iteration 1 — gfx1151-Inspired Sweep (in progress)
**Finding: DU=8 + PGR=1 gives +5-6.5% over DU=16**
- 4096x4096: 20.1 → 21.4 TFLOPS (non-VOPD DU=8 PGR=1 TT[8,8])
- 6144x8192: 24.1 → 25.5 TFLOPS (VOPD DU=8 PGR=1 TT[8,8])
- TT[7,8] competitive with TT[8,8]
- StaggerU=8 helps slightly

**Next: Re-tune all 580 medium+large shapes with DU=8**

## Iteration 2 — LdsPad + DU=8 Optimization
**Finding: LdsPadA=4, LdsPadB=4-8 with DU=8 gives +11-16%!**

| Shape | Iter 0 Best | Iter 2 Best | Improvement |
|-------|------------|------------|-------------|
| 6144x8192 | 24.1 T | **25.6 T** | +6.2% |
| 4096x4096 | 20.1 T | **22.5 T** | +11.9% |
| 3072x4096 | 19.6 T | **22.2 T** | +13.3% |
| 1536x2048 | 10.0 T | **11.6 T** | +16.0% |

Best params: TT[8,8] WG[16,8,1] DU=8 PGR=1 PLR=0 SU=0 LdsPadA=4 LdsPadB=4-8
LdsPad reduces bank conflicts in LDS reads — huge impact on non-MI FP32.

**Next: Run full sweep with these new params on all 1,030 shapes**

**Full re-tuning complete:**
- Wave 3 (320 medium): 6 new solutions added (DU=8+LdsPad won)
- Wave 4 (260 large): 14 new solutions added
- Merged logic: 1,030 shapes, 178 solutions
- Deployed to library/.../gfx1100/GridBased/

## Iteration 3 — Exploration Sweep
Tried PGR=2, PLR=1, WG[8,16,1], StaggerU=4, LdsPad=8 on 4096x4096:
- No improvement beyond 22.5 TFLOPS
- WG[8,16,1] competitive but not better
- PGR=2 at 22.3 TFLOPS — close but no cigar
- LPA=8 shows up in top configs

Also tried GlobalSplitU=2,4 — made performance WORSE (21.7T vs 22.8T).

## Final Performance Summary (after all iterations)

| Shape | hipBLASLt WMMA | Iter 0 | Iter 2 (DU=8+LdsPad) | Improvement |
|-------|---------------|--------|----------------------|-------------|
| 6144x8192 | 3.1 T | 24.1 T | **25.6 T** | +6.2% / **8.3x vs WMMA** |
| 4096x4096 | 3.5 T | 20.1 T | **22.8 T** | +13.4% / **6.5x vs WMMA** |
| 3072x4096 | ~3.0 T | 19.6 T | **22.2 T** | +13.3% / **7.4x vs WMMA** |
| 1536x2048 | ~2.5 T | 10.0 T | **11.6 T** | +16.0% / **4.6x vs WMMA** |
| Blog kernel | N/A | 38.5 T | N/A | N/A (hand-crafted assembly) |

Key winning params: TT[8,8] WG[16,8,1] DU=8 PGR=1 PLR=0 SU=0 **LdsPadA=4 LdsPadB=8**

### Remaining gap to blog kernel (38.5T vs 22.8T at 4096x4096)
The blog kernel uses:
1. MT128x128 (vs our MT128x64) — 2x compute/load ratio
2. All 64 FMAs VOPD dual-issued (vs our 88% diagonal)
3. Interleaved ds_load/v_dual_fmac (vs our sequential MAC block)
4. Occupancy 7 (vs our 2) — hand-optimized register allocation
5. No kernel-arg overhead, no bounds checking

Closing this gap requires fundamental kernel writer changes (register allocation, scheduling, tile size), not parameter tuning.

## Iteration 4 — TT[8,16] MT128x128 VGPR Analysis
TT[8,16] with PGR=0 DU=4 gives 22.8 TFLOPS (= TT[8,8] best) but uses 256 VGPRs (occupancy 1).
Same perf at 2x compute/load ratio proves occupancy is the limiting factor.

**VGPR map for TT[8,8] (128 VGPRs total):**
- v0-v63: ValuC (64)
- v64-v65: GlobalReadOffset A,B (2)
- v66-v67: LocalWriteAddr A,B (2)
- v68-v69: LocalReadAddr A,B (2)
- v70-v71: alignment gap (2)
- v72-v79: G2L_A/ValuA (8, overlapped)
- v80-v87: G2L_A cont (8, extra G2L buffer)
- v88-v95: G2L_B/ValuB (8, overlapped)
- v96: G2L_B cont (1)
- v97: Serial (1)
Total named: 98, but declared as 128 (extra G2L buffer space)

**Key insight**: G2L buffer is 17 VGPRs for A, even though ValuA is only 8.
The extra 9 VGPRs are for G2L double-buffering (PGR=1) or extra load elements.
With PGR=0, G2L overlaps with ValuA/B exactly, but the allocator still reserves extra space.

**To get TT[8,16] occupancy 2**: Need 128-192 VGPRs (vs current 256).
- ValuC: 128 (fixed)
- ValuA: 8 (fixed)
- ValuB: 16 (fixed, TT1=16 so 16 B values per thread)
- Addresses: 6 (fixed)
- Serial: 1
- Subtotal: 159 VGPRs
- Overhead: 256-159 = 97 VGPRs of G2L/padding/alignment waste

Eliminating 65+ VGPRs of waste → ~191 VGPRs → occupancy 2.
**CORRECTION**: With PGR=0, TT[8,16] already uses only 168 VGPRs (occ 9!).
TT[8,8] PGR=0 uses 96 VGPRs (occ 16). PGR=1 adds ~32 VGPRs for double-buffer.

The G2L elimination (Change 1) mainly helps PGR=1 configs, which need double-buffered G2L.
With PGR=0, G2L overlaps with ValuA/B and no separate allocation is needed.

**Bottom line**: The VGPR budget isn't the bottleneck for PGR=0 configs.
TT[8,16] PGR=0 DU=4 already achieves 22.8 TFLOPS (same as TT[8,8] best).
The MT128x128 arithmetic intensity advantage is eaten by DU=4 loop overhead.

## Iteration 5 — MAC Interleaving Investigation

### SIA=1 Schedule Analysis (per K-iteration for TT[8,8] DU=8):
1. 16 ds_load (A[0..7] + B[0..7]) — back-to-back, ~16 issue cycles
2. ~8 instructions (global reads, pointer updates)
3. s_waitcnt lgkmcnt(0) — STALLS ~12 cycles (LDS latency = ~20, only 8 instr between load and wait)
4. 36 MAC instructions (VOPD) or 64 (non-VOPD)
5. Next iter's ds_loads

### Measured Stall: ~14% of total loop time
- 8 iterations × 12 stall cycles = 96 wasted cycles per tile
- Total loop time ≈ 8 × (16+8+12+36) = 576 cycles
- Stall fraction: 96/576 = 16.7%
- Eliminating stall → ~17% improvement → 22.8 * 1.17 = **26.7 TFLOPS**

### Why Simple Reordering Doesn't Help
The `waitCode` is always `s_waitcnt lgkmcnt(0)` — it waits for ALL pending LDS ops.
Even if we reorder reads and other ops, the lgkmcnt(0) blocks until everything completes.

### What Would Help
1. **Replace lgkmcnt(0) with precise lgkmcnt(N)**: Start MACs when first A+B loaded,
   let remaining loads complete during early MACs. Requires dependency tracking.
2. **Software pipelining**: Issue next iter's ds_loads BEFORE current iter's MAC block,
   so reads and compute overlap across iterations (like the blog kernel).
3. **PrefetchLocalRead=1**: Tensile already supports this — it loads next iter's A/B
   values during current iter's MAC. But PLR=1 always lost in our sweeps.

### Decision
The ~12-cycle stall per iteration is real but modest. The architectural change to
implement precise lgkmcnt tracking is complex and risky. The ~17% theoretical gain
would bring us to ~27 TFLOPS — still far from the blog kernel's 38.5 TFLOPS.

The remaining gap (27 vs 38.5 = 1.43x) comes from the blog kernel's:
- 100% VOPD (vs 88% diagonal) — ~6% improvement
- 128x128 tile at high occupancy — ~30% improvement (2x compute/load + better occupancy)
- Optimized memory patterns — remaining ~7%

These require fundamental kernel architecture changes (VGPR interleaving, custom
tile mapping) beyond the scope of SIA scheduler modifications.
