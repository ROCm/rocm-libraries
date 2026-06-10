# MXFP6 GEMM — Handoff

**Date:** 2026-06-10 · **Branch:** `zhewan/ck/mxfp6-standalone`
**Machine:** AMD Instinct MI350X (gfx950, CDNA4), ROCm 7.0.2.1

Problem: `D[M,N] = A[M,K] · B[K,N]`, MXFP6 E2M3 inputs + per-32-block E8M0 scales, FP16/BF16/F32 out.
MFMA = `v_mfma_scale_f32_32x32x64_f8f6f4` (cbsz:2 blgp:2, 32 cyc/inst on FP6).

---

## TL;DR

**One kernel paradigm — `lds_gemm_hybrid_dripA` — shape-routed by `mxfp6_dispatch.hpp`.**
@8192³ FP16 ≈ **2230 TFLOPs**. Beats the previous register-direct + pure-LDS v18 dispatcher
on **all 12 benchmarked shapes (+6~98%)**, including non-pow2 N (5120/7680/9216) where v18
needed dedicated 18/20-acc mixed tiles — the hybrid paradigm wins those outright.

Tile routing (`choose_tile`):
- **256×256** (16-acc arithmetic-intensity sweet spot) — workhorse for every shape whose
  256×256 grid fills the machine (WG ≥ #CU).
- **128×256** (8-acc) — ONLY for WG-starved small-M shapes (256×256 grid < #CU); halves the
  M-tile to double WG count and fill idle CUs (+20~98% there). Same kernel, different tile args.

---

## Kernel essence (`lds_gemm_hybrid_dripA`, tile-general)

Distilled from the full optimization history (register-direct big-tile era → deep-K LDS era →
hybrid era). The load-bearing levers, all measured:

- **Big register tile + LOW occupancy (occ1).** A huge per-wave 16-acc tile (256 AGPR) maximizes
  arithmetic intensity / register reuse. occ1 latency-bound is the *intended* regime — raising
  occupancy (occ2) was proven net-negative (−14~30%: halving acc collapses intensity and trades
  the latency wall for an L2/bandwidth wall). The 256-AGPR tile is also why occ stays 1 — the
  merged VGPR pool (512) is full; this is structural, not a bug.
- **32×32×64 MFMA, not 16×16×128.** Same FP6 FLOP/cyc (both 4096) but 16×16×128 needs 2× the
  instructions + 2× operand bandwidth for the same output — pure overhead. (This is why CK's
  mxfp6, which uses 16×16×128, is ~2× slower.)
- **A staged deep-K in LDS (KT=192), double-buffered (RDB).** The deep-K MFMA window exceeds the
  load latency so latency is hidden by compute without register spill. RDB (prefetch after the
  barrier) makes one barrier/tile cover both RAW and WAR — correct for all shapes.
- **B streamed DIRECT HBM→VGPR (coalesced), not through LDS.** Bypasses B's whole LDS round-trip
  (kills the Era-2 ds-read wall, profiled 40%→0%). B MUST be coalesced (`load_b_shuf` from the
  `preshuffle_B` layout) — a raw column-major scatter erases the win (−3%).
- **drip-A.** A's cooperative `buffer_load_lds`es are dripped 1/MFMA-quartet across the compute
  window instead of bursted after the barrier (removes the top ATT stall). Tuned schedule:
  `ADRIP_START=1` (skip the sub-head hotspot quartet), PFD=5 B-ring. `HARD_WAIT=1`
  (`wait_vmcnt(0)` before each DB barrier) is faster AND safe (de-pollutes the A/B-shared vmcnt).
- **tiled-scale.** A wave's K-tile scales are contiguous per lane → one `global_load_dwordxN`
  per K-tile. (The #1 correctness pitfall of the LDS port.)
- **swz0 best on this machine** (the L2-locality remap underperforms the raw WG order here).

Profile (RCV, @8192³): occ1 latency-bound, MFMA matrix-unit busy ~30%, ~2.37× theoretical
headroom locked behind the occ1 wall. Stall ≈ vmcnt(B HBM) 24% + lgkmcnt(A LDS) 21% + v_mfma
backpressure (real work) 24% + epilogue store drain ~10%. No single reclaimable villain remains.

---

## Files

| File | What |
|---|---|
| `mxfp6_dispatch.hpp` | **`choose_tile` + `dispatch_gemm`** — the unified shape→tile router |
| `mxfp6_lds_hybrid.hpp` | **`lds_gemm_hybrid_dripA`** (tile-general) + `load_b_shuf` + `issue_A_chunks` |
| `mxfp6_lds.hpp` | shared device helpers (`read_op`, `asm_load_dwordxN_nowait`) + host `tile_scale` |
| `mxfp6_asm_utils.hpp` | MFMA wrappers, `ds_read_fp6x32_plain`, `store_acc_t`, wait/M0 helpers |
| `mxfp6_preprocess.hpp` | `quantize_to_mxfp6`, `preprocess_B`, `preprocess_scale`, `preshuffle_B` |
| `mxfp6_types.hpp` / `mxfp6_reference.hpp` | types / CPU reference GEMM |
| `test_dispatch.cpp` | end-to-end correctness (fresh-alloc + CPU ref) + 12-shape benchmark |
| `profile_out/` | RCV traces + annotated ASM of the hybrid paradigm (analysis artifacts) |

Build & run: `make test_dispatch && ./test_dispatch`

---

## Performance (FP16, vs v18 best-per-shape, same machine 2026-06-10)

| shape | v18 best | hybrid | Δ | tile |
|---|---|---|---|---|
| 8192×8192 | 1835 | 2231 | +22% | 256×256 |
| 8192×4096 | 1810 | 2160 | +19% | 256×256 |
| 4096×8192 | 1773 | 2199 | +24% | 256×256 |
| 8192×9216 | 1743 | 2163 | +24% | 256×256 |
| 8192×7680 | 1794 | 2200 | +22% | 256×256 |
| 8192×5120 | 1665 | 2065 | +24% | 256×256 |
| 4096×5120 | 1570 | 1670 | +6% | 256×256 |
| 4096×4096 | 1652 | 1976 | +20% | 256×256 |
| 2048×8192 | 1679 | 1906 | +14% | 256×256 |
| 2048×4096 | 997 | 1601 | +61% | 128×256 |
| 2048×2048 | 516 | 993 | +92% | 128×256 |
| 1024×4096 | 678 | 844 | +24% | 128×256 |

(Absolute numbers vary ±5% run-to-run from the SCLK/1000W power cap; kernel is stable.)

## Validation
- Fresh-alloc 0x5A-poison vs CPU ref, both tile paths (256×256 and 128×256), incl. partial-grid
  / non-square / k_tiles==1. `test_dispatch` runs the end-to-end correctness gate.
- ⚠️ `k_tiles==1` needs `wait_vmcnt(0)` in the odd-tail (no compute-window margin) — present.

## Dead ends — do NOT re-try without a new idea
- **occ2 / smaller acc tile:** −14~30%. occ gate is discrete (≤256 VGPR for occ2); even 8-acc
  stays occ1 (B-ring eats ~135 arch VGPR), and filling CUs via smaller tiles loses more to
  strength than it gains. The 256-AGPR tile is the whole point.
- **split-K for small-M:** atomic崩 (298); deterministic (partial buffer + reduce) only +6% —
  the reduce + short-K fixed-overhead eat the fill-CU gain. 128×256 (+25%) beats it.
- **persistent / epilogue-overlap:** the epilogue store drain shares the HW vmcnt with B-direct's
  ring, so the next tile's `vmcnt` wait is held by the in-flight stores — overlap doesn't fire.
  (Plus grid-stride framework −5%.) The store-after-final-MFMA variant spills.
- **16×16×128 MFMA, wider 128×512 tile:** see kernel essence / scale-path NDB≤1 limit.

## Repro
```bash
make test_dispatch && ./test_dispatch
# CK reference (same machine): /home/AMD/zhewan/ck-bench-6732acf/bin/tile_example_mx_flatmm
```
NFS backup: source mirrored to `/home/AMD/zhewan/rocm-libraries-ck/mxfp6_gemm/`.
