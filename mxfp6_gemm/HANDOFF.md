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
  M-tile to double WG count and fill idle CUs. Same kernel, different tile args. (occ1 — an
  occ2 variant was tried and is steady-state-neutral; see dead-ends.)

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

Built as a library (`libmxfp6gemm`) with a device-free public header; the test links it and
drives it through the public API. Layout:

| File | What |
|---|---|
| `include/mxfp6/gemm.hpp` | **public API** — `mxfp6::gemm(OutType,…)` + `choose_tile` (device-free; host TUs can include it) |
| `include/mxfp6/preprocess.hpp` | host: `quantize_to_mxfp6` / `preprocess_B` / `preshuffle_B` / `preprocess_scale` / `tile_scale` |
| `include/mxfp6/{types,reference}.hpp` | data types / CPU reference GEMM |
| `src/gemm.cpp` | library impl (HIP): `gemm()` + `choose_tile()`, instantiates F32/F16/BF16 |
| `src/dispatch.hpp` | internal `detail::dispatch_gemm<OutT>` (shape→tile launch) |
| `src/lds_hybrid.hpp` | **`lds_gemm_hybrid_dripA`** kernel + `load_b_shuf` + `issue_A_chunks` |
| `src/lds.hpp` · `src/asm_utils.hpp` | device helpers (`read_op`, `asm_load*`) · MFMA/store/wait |
| `tests/test_dispatch.cpp` | end-to-end correctness (fresh-alloc + CPU ref) + perf, via the public API |
| `CMakeLists.txt` | HIP build: static lib `mxfp6gemm` + `ctest` |
| `profile_out/` | RCV traces + annotated ASM of the hybrid paradigm (analysis artifacts) |

Build & test: `cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950 && cmake --build build -j && ctest --test-dir build`

Use the library: include `<mxfp6/gemm.hpp>`, link `mxfp6gemm`, then
`choose_tile(M,N)` → host preprocess (tile scales with the returned MPW/NPW) → `gemm(OutType, …)`.

---

## Performance (FP16, vs v18 best-per-shape, same machine 2026-06-10)

⚠️ The hybrid column below is from the original 12-shape sweep where small-M shapes ran first
on a still-ramping GPU (cold-biased). Steady-state re-measurement (see vs-CK table) is higher
for those, e.g. 2048×4096 = 1686 steady vs 1517 swept. Trends/ratios hold either way.

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
| 2048×4096 | 997 | 1517† | +52% | 128×256 |
| 2048×2048 | 516 | 1014† | +96% | 128×256 |
| 1024×4096 | 678 | 848† | +25% | 128×256 |

† cold-biased swept value; steady-state is higher (2048×4096 ≈ 1686, 2048×2048 ≈ 1047).
Absolute TFLOPs also swing ±10% run-to-run from the SCLK/1000W power cap; warm up to steady
state before comparing. The vs-CK table below is all steady-state.

## vs CK (same machine 2026-06-10, K=8192, `tile_example_mx_flatmm`, FP16, **steady-state**, all warmed)

| shape | CK FP8 | CK FP6 | ours FP6 | ours / CK-FP6 | ours / CK-FP8 |
|---|---|---|---|---|---|
| 2048×4096 | 1608 | 1019 | 1686 | 1.66× | 1.05× |
| 2048×8192 | 1757 | 1079 | 2053 | 1.90× | 1.17× |
| 4096×4096 | 1759 | 1079 | 2185 | 2.03× | 1.24× |
| 4096×8192 | 1820 | 1097 | 2214 | 2.02× | 1.22× |
| 8192×8192 | 1843 | 1109 | 2234 | 2.21× | 1.21× |

(ours = median of 25 windows after 120 warm iters; CK = warmup=100 repeat=100.)

**Same-precision 1.66~2.21× CK MXFP6; and our FP6 beats CK's FP8 by 1.05~1.24×.** On this machine CK
FP8 > FP6 (FP6 pinned by the 1000W power cap + CK's 16×16×128). NOTE: an external 2026-05-04
table showed CK FP6 (3200) > FP8 (3019) at much higher absolutes — that is a *different
machine / CK build* (no power cap); do NOT compare its numbers to this machine's.

## Validation
- Fresh-alloc 0x5A-poison vs CPU ref, both tile paths (256×256 and 128×256), incl. partial-grid
  / non-square / k_tiles==1. `test_dispatch` runs the end-to-end correctness gate.
- ⚠️ `k_tiles==1` needs `wait_vmcnt(0)` in the odd-tail (no compute-window margin) — present.

## Dead ends — do NOT re-try without a new idea
- **occ2, both ways:** (a) by shrinking acc (16→≤6): −14~30% (strength collapse, latency→
  bandwidth wall). (b) the 8-acc 128×256 small-M tile CAN be forced to occ2 (MIN_OCC=2 → 251
  VGPR, 0 spill) without losing strength — but **steady-state it's ~0%** (a 256-WG shape has no
  2nd WG to fill the occ2 slot). An earlier "+6~10%" was a warm-up artifact (always timing occ1
  first on a ramping GPU). Reverted to occ1. The 256×256 workhorse is occ1-locked anyway (507
  VGPR). Lesson: warm to steady state AND alternate A/B/A/B before trusting an occ delta.
- **split-K for small-M:** atomic崩 (298); deterministic (partial buffer + reduce) only +6% —
  the reduce + short-K fixed-overhead eat the fill-CU gain. 128×256 (+25%) beats it.
- **persistent / epilogue-overlap:** the epilogue store drain shares the HW vmcnt with B-direct's
  ring, so the next tile's `vmcnt` wait is held by the in-flight stores — overlap doesn't fire.
  (Plus grid-stride framework −5%.) The store-after-final-MFMA variant spills.
- **16×16×128 MFMA, wider 128×512 tile:** see kernel essence / scale-path NDB≤1 limit.

## Repro
```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950 && cmake --build build -j && ctest --test-dir build
# CK reference (same machine): /home/AMD/zhewan/ck-bench-6732acf/bin/tile_example_mx_flatmm
```
NFS backup: source mirrored to `/home/AMD/zhewan/rocm-libraries-ck/mxfp6_gemm/`.
