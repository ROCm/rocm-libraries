# MXFP6 GEMM — Handoff

**Date:** 2026-06-11 · **Branch:** `zhewan/ck/mxfp6-standalone`
**Machine:** AMD Instinct MI350X (gfx950, CDNA4), ROCm 7.0.2.1

> 2026-06-11 session = **no code change** — a measurement/audit pass: confirmed every perf knob is
> already at its optimum, hard-measured the utilization picture, fair-corrected the vs-CK comparison,
> and wrote a perf-theory doc. All findings folded in below. Net: kernel is at its structural ceiling.

Problem: `D[M,N] = A[M,K] · B[K,N]`, MXFP6 E2M3 inputs + per-32-block E8M0 scales, FP16/BF16/F32 out.
MFMA = `v_mfma_scale_f32_32x32x64_f8f6f4` (cbsz:2 blgp:2, 32 cyc/inst on FP6).

---

## TL;DR

**One kernel paradigm — `lds_gemm_hybrid_dripA` — shape-routed by `mxfp6_dispatch.hpp`.**
@8192³ FP16 ≈ **2292 TFLOPs** warm / **2158** cold (2026-06-11: +2.6% from a **swapped-MFMA coalesced-store epilogue** —
see "Epilogue" in kernel essence; was 2230 with the scattered store). Beats the previous register-direct + pure-LDS v18 dispatcher
on **all 12 benchmarked shapes (+6~98%)**, including non-pow2 N (5120/7680/9216) where v18
needed dedicated 18/20-acc mixed tiles — the hybrid paradigm wins those outright.

Tile routing (`choose_tile`):
- **256×256** (16-acc arithmetic-intensity sweet spot) — workhorse for every shape whose
  256×256 grid fills the machine (WG ≥ #CU).
- **128×256** (8-acc) — ONLY for WG-starved small-M shapes (256×256 grid < #CU); halves the
  M-tile to double WG count and fill idle CUs. Same kernel, different tile args. (occ1 — an
  occ2 variant was tried and is steady-state-neutral; see dead-ends.)

@8192³: warm ≈ 2292 / cold ≈ 2158 TFLOPs (~25% peak; was 2230 before the coalesced-store epilogue). ⚠️ **2048×4096 is the weak shape** (warm ~1707 /
cold ~1555, now ~tie vs CK FP8): filling 256 CUs at M=2048 forces an 8-acc tile (area math: MT×NT=32768 ⟺ 8 acc), whose
MFMA window (768 cyc) can't hide B's HBM latency → B-VMEM-exposed, ~18.7% peak. It's the only shape
that loses to CK FP8 (cold 0.90×). Structurally locked, not a tuning miss.

---

## Kernel essence (`lds_gemm_hybrid_dripA`, tile-general)

Distilled from the full optimization history (register-direct big-tile era → deep-K LDS era →
hybrid era). The load-bearing levers, all measured:

- **Big register tile + LOW occupancy (occ1).** A huge per-wave 16-acc tile (256 AGPR) maximizes
  arithmetic intensity / register reuse. occ1 latency-bound is the *intended* regime — raising
  occupancy (occ2) was proven net-negative (−14~30%: halving acc collapses intensity and trades
  the latency wall for an L2/bandwidth wall). The 256-AGPR tile is also why occ stays 1 — the
  merged VGPR pool (512) is full; this is structural, not a bug.
- **32×32×64 MFMA, not 16×16×128.** Same FP6 FLOP/cyc (both 4096) but for a large 32-aligned tile
  16×16×128 needs 2× the instructions (and re-fetches operands per small output) — pure overhead
  here. (This is why CK's mxfp6, which defaults to 16×16×128, is ~2× slower on big aligned GEMMs.)
  ⚠️ 16×16×128 is NOT useless in general — it wins on small/skinny/non-32-aligned shapes (decode,
  GEMV, small MoE: 32×32 wastes ≥50% on M=16) and enables high-occupancy small-tile designs
  (¼ the acc VGPR). It's a granularity-vs-instruction-count trade; we picked the big-aligned side.
  See `docs/perf_optimization_guide.md` §7.
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
- **Swapped-MFMA coalesced store, +2.6% @8192³ (2026-06-11).** Root cause of store backpressure: the
  stock MFMA (`src0=B,src1=A`, TransposeC) outputs lane=M-row, so direct stores to row-major D scatter
  ~32 cache-lines/instr (coalesced-store proxy measured +7.6% recoverable). FIX at the SOURCE: swap the
  MFMA operands (`src0=A,src1=B` → `mfma_scale_f32_32x32x64_fp6_swapC`) so acc comes out as Cᵀ
  (lane=N-col); then storing D[m][n] with n=base+lane%32 is NATURALLY COALESCED (consecutive lanes →
  consecutive N), no transpose/LDS/barrier, any OutT. VGPR 508 / spill 0 / LDS stays 72KB / occ1.
  An LDS-transpose epilogue gives the SAME +2.7% but needs 132KB LDS + barriers + is FP16-only — the
  swap is strictly cleaner (kept it). ⚠️ NOT `dwordx4`: tested, no gain — `dwordx4` LDS-read alignment
  (LROW%8==0) conflicts with bank-conflict-free write (LROW≡2 mod4), and store-instr count isn't the
  bottleneck (issue 4.4% busy; coalescing already gives full 64B cache-line transactions). ⚠️ Store
  OVERLAP (hide stores in compute) is a separate FAILED idea (drain −1.1%, B-LDS −1.2%, nt −22%):
  stores are end-loaded + single VM_CNT. The win is COALESCING, not width or overlap. See
  `mxfp6_epilogue_drain`.

**Profile & utilization (hard-measured 2026-06-11; RCV + rocprofv3 counters + `.s` metadata, @8192³ FP16):**
- **occ1 CONFIRMED three ways** (was doubted, now settled): `.s` metadata 251 arch VGPR + 256 AGPR =
  **507/512 merged → VGPR-bound occ1** (spill 0; LDS 72KB alone would allow 2); RCV `occupancy.json`
  ≡ 1. ⚠️ rocprofv3 CSV `VGPR_Count`/`Accum_VGPR_Count` **misreport** on the gfx950 merged pool
  (showed 256/0; truth 251/256) — trust `.s num_vgpr/num_agpr`, not the CSV.
- **MFMA duty cycle ~22–24% of nominal peak (9227@2.2GHz) = ~27–29% of the actual sustained clock**
  (~1.83 GHz measured under load — power/thermal-limited, but only ~880–910 W, NOT hitting the 1000 W
  cap). Matrix unit ~71% idle on memory latency; ~2.4× headroom locked behind the occ1 wall. ~2× CK's duty.
- **HBM BW ~11%** (measured FETCH 312 MB + WRITE 163 MB = 474 MB/dispatch; L2 read-hit **89.9%**) →
  far from bandwidth-bound.
- **Coalescing optimal**: writes 1.0× (req bytes == FP16 output 134 MB, zero waste); A (lane×16
  `buffer_load`) + B (`preshuffle_B`) instruction-level 100% coalesced. **Bank conflict 9.5% of
  LDS-active but LDS off critical path → <1% runtime (red herring)**.
- Stall mix (ATT): vmcnt(B HBM) ~24% + lgkmcnt(A LDS) ~21% + v_mfma backpressure ~24% + epilogue ~10%.
- **Net: latency-bound BY DESIGN** — big-tile occ1 trades occupancy for arithmetic intensity (the
  winning side; occ2 is −14~30%). The 71% idle is *structurally stranded* (VGPR + B-HBM-latency both
  locked on gfx950), not waste. ⚠️ HBM round-trip "~880 cyc" used in analysis is an **estimate**, not
  measured (real cost = ~22 cyc/load issue-stall + ~460–500 cyc/hit data-wait; order is right).

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
| `docs/perf_optimization_guide.md` | **perf theory** (AI / roofline / latency-hiding / big-tile / deep-K / MFMA / occ), formula-derived with worked examples & this kernel's real numbers |
| `docs/PERF_SUMMARY.md` | executive summary (vs-CK, the structural ceiling, what it'd take to go higher) |

Build & test: `cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950 && cmake --build build -j && ctest --test-dir build`

Use the library: include `<mxfp6/gemm.hpp>`, link `mxfp6gemm`, then
`choose_tile(M,N)` → host preprocess (tile scales with the returned MPW/NPW) → `gemm(OutType, …)`.

---

> ⚠️ The tables in this section (both vs-v18 and vs-CK below) were measured BEFORE the 2026-06-11
> coalesced-store epilogue. Add ~+2.6% to every "ours/hybrid" number (8192³ 2231→~2287). They are a
> conservative floor — the vs-CK verdicts only improve.

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

## vs CK (same machine, K=8192, `tile_example_mx_flatmm`, FP16) — FAIR matched, refreshed 2026-06-12

Methodology: cold = stock `tile_example_mx_flatmm` (RotatingMemWrapper rotates inputs → cold L2/rep);
warm = `tile_example_mx_flatmm_warm` (single-buffer). Ours measured the same way (warm = single buffer;
cold = rotate 6 input sets → cold L2). Both sustained back-to-back (no turbo bonus,
[[feedback_interleave_inflation]]). Report BOTH regimes ([[feedback_bench_cold_and_warm]]).
⚠️ ours numbers below are the **current swapped-MFMA coalesced-store kernel** (2026-06-11).
CK FP4 (`pk_fp4`, half the bits) added for cross-precision context — it's a *different tier*.

**COLD-vs-COLD** (rotated; = single-call / real-inference where B is evicted between layers):
| shape | CK FP8 | CK FP6 | CK FP4 | **ours FP6** | /CK-FP6 | /CK-FP8 |
|---|---|---|---|---|---|---|
| 2048×4096 | 1569 | 1004 | 1707 | 1555 | 1.55× | **0.99×** |
| 2048×8192 | 1704 | 1080 | 3060 | 1965 | 1.82× | 1.15× |
| 4096×4096 | 1693 | 1079 | 3047 | 2099 | 1.95× | 1.24× |
| 4096×8192 | 1789 | 1100 | 3321 | 2123 | 1.93× | 1.19× |
| 8192×8192 | 1831 | 1109 | 3482 | 2158 | 1.95× | 1.18× |

**WARM-vs-WARM** (single-buffer / L2-hot; = repeated-call cache-resident):
| shape | CK FP8 | CK FP6 | CK FP4 | **ours FP6** | /CK-FP6 | /CK-FP8 |
|---|---|---|---|---|---|---|
| 2048×4096 | 1626 | 1057 | 1856 | 1707 | 1.62× | 1.05× |
| 2048×8192 | 1724 | 1101 | 3189 | 2105 | 1.91× | 1.22× |
| 4096×4096 | 1732 | 1100 | 3182 | 2236 | 2.03× | 1.29× |
| 4096×8192 | 1809 | 1107 | 3419 | 2243 | 2.03× | 1.24× |
| 8192×8192 | 1845 | 1111 | 3559 | 2292 | 2.06× | 1.24× |

**Verdict:** vs same-precision **CK FP6 we win 1.55~2.06×** (every shape, both regimes). vs the
higher-precision **CK FP8: WARM we beat it 1.05~1.29×; COLD we beat it 1.15~1.24× except 2048×4096
which is now ~tie (0.99×, was 0.90× before the coalesced-store epilogue)** — that area-locked 8-acc
shape is the only non-win and it's now break-even. **CK FP4** (half-bit tier) is ~1.7~3.6K — our FP6
sits between CK FP8 and CK FP4 (faster than FP8, can't reach FP4's bit-count advantage).
On this machine CK FP8 > CK FP6 (FP6 pinned by 1000W cap + CK's 16×16×128). The external 2026-05-04
table (CK FP6 3200 > FP8 3019) is a *different machine/build* (no power cap); do NOT compare.

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
- **Per-shape tuning of drip-A / K_TILE / SWZ on the current hybrid (2026-06-11):** all swept on
  both tile paths (8192³ 256×256 and 2048×4096 128×256). Production defaults (PFD5/START1/STR1/PER1,
  KT192, SWZ0) are already the optimum everywhere — **zero gain**. ⚠️ Sequential sweeps show a false
  +1~2% (KT256/SWZ8 looked better on 2048×4096) — it's an **idle→turbo-boost artifact** (host gaps
  between measurement windows let the clock boost; `usleep` dose-response 0/2/10ms → 1673/1730/1737).
  Interleaved A/B or back-to-back kills it; the defaults win. (memory `feedback_interleave_inflation`.)
- **Larger shapes don't break the ceiling (2026-06-11):** 12288²/16384²/16384³/deep-K all plateau at
  ~23–24% peak. Cold (realistic) even rises slightly 8192³ 2038 → ~2120–2163 (warm/cold converge as
  A+B ≫ L2; deep-K K=16384 marginally best cold ~2163). WARM 8192³ (2237) is the warm peak; bigger is
  not faster. The occ1 latency ceiling is shape-independent for CU-filling shapes.
- **Using the spare LDS (only 72/160 KB used) — nothing to exploit:** LDS is slack, not the binding
  constraint (VGPR occ1 + B-HBM-latency are). Deeper K (KT256 = 96 KB) is worse (2227→2106); B-in-LDS
  is the disproven pure-LDS path (−28~35% small-M); triple-buffer A buys nothing (A already 0% stall).
  The 72 KB is the *win* of taking B out of LDS (B-direct), not waste.

## Repro
```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950 && cmake --build build -j && ctest --test-dir build
# CK reference (same machine, ⚠️ args need leading dash):
#   cold (stock): /home/AMD/zhewan/ck-bench-6732acf/bin/tile_example_mx_flatmm -m=8192 -n=8192 -k=8192 -mx_prec=fp6xfp6 -v=0 -warmup=50 -repeat=100
#   warm:         .../tile_example_mx_flatmm_warm  (same args; rotating off)        -mx_prec=fp6xfp6|fp8xfp8
# dev sweep/bench drivers: hipcc -std=c++17 --offload-arch=gfx950 -O3 -Iinclude -Isrc <driver>.cpp [src/gemm.cpp] -o <bin>
# rocprofv3 utilization: --pmc <counters> --output-format csv --kernel-include-regex lds_gemm_hybrid_dripA
#   (⚠️ FETCH_SIZE+WRITE_SIZE exceed counter slots together — collect separately; TCP_TCC req unit = 16 B)
```
NFS backup: source mirrored to `/home/AMD/zhewan/rocm-libraries-ck/mxfp6_gemm/`.
