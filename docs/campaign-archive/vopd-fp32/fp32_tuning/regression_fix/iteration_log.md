# VOPD Regression-Fix — Autonomous Iteration Log

Started 2026-06-02. Goal: fix all real FP32 regressions in merged_v3 logic vs baseline on gfx1100 (RX 7900 XTX).

## Iteration 0 — Methodology reconciliation (CRITICAL)

Before tuning, validated the benchmark methodology. Found the premise in memory was built on
unreliable numbers. Established ground truth:

- **My FP32 end-to-end measurement is correct.** `hipblaslt-bench --precision f32_r --compute_type s
  --transA T --transB N` on 4096^3 gives ~3.45 TFLOPS for the *heuristic* pick, ~5.7 TFLOPS for
  best-of-all-solutions. RESULTS.md independently confirms hipBLASLt FP32 baseline = 3.468 TFLOPS
  on 4096^3. Match.
- **`hipblaslt_bench_results.csv` is NOT a valid FP32 baseline.** Its numbers (4096^3 = 19721,
  2048x8192x8192 = 39733) do not match FP32 (3.5 TF) or cold FP16. Likely a mislabeled / cache-hot
  / different-tool file. MUST NOT be used as the regression reference.
- **Campaign Tensile CSVs are cache-resident.** Peaks of 131072-196608 GFlops are physically
  impossible (FP32 peak 30.7 TF, VOPD peak 61.4 TF). Tensile's in-process bench reuses L2; not
  real end-to-end throughput. The "+40% to +1294% VOPD wins" in memory derive from these and are
  NOT trustworthy as end-to-end claims.
- **Device lib DOES contain FP32 kernels** (792 `Cijk_SS_GG_B` in Kernels.so hsaco). The logic-file
  naming `Cijk_Ailk_Bjlk_S_B` differs from the binary's `Cijk_SS_GG_B` — earlier zero-grep was a
  false alarm.
- GPU: must be warmed (cold GPU under-ramps short runs). `power_dpm_force_performance_level=high`
  HURT throughput on RDNA3; keep `auto` + warmup iters.

### Correct A/B definition (what "regression" really means)
Same kernel pool (hsaco) for both logic files. Only the heuristic's *selection* differs.
- Rebuild device lib from baseline logic (.bak_pre_vopd_campaign) → bench `--algo_method heuristic`
  (default, what real users get) per shape, FP32 TN, warmed, cold-cache.
- Rebuild device lib from merged_v3 logic → same bench.
- **Regression = merged_v3 heuristic GFlops < baseline heuristic GFlops** (beyond ~3% noise).
- Also track best-of-all (`--algo_method all`) as the ceiling: if heuristic << all, the logic is
  mis-selecting and is fixable by logic edit alone (no re-tune needed).

Next: build harness, run baseline-lib A/B vs merged_v3-lib.

## Iteration 1 — Confirmed VOPD is real; found the ACTUAL fixable problem

### VOPD is genuinely integrated (disassembly proof)
Disassembled the deployed `Kernels.so` hsaco with llvm-objdump:
- **188 `v_dual_fmac_f32`** instructions + **9272 total `v_dual_*`** dual-issue ops, 0 `v_wmma`.
So the dual-issue FP32 path IS compiled into the runtime. The integration works. The memory note
that VOPD is "fully working" is correct at the kernel level.

### The real, fixable problem: HEURISTIC MIS-SELECTION (not a tuning regression)
A/B bench of merged_v3 (cold-cache, FP32 TN, warmed GPU), heuristic vs best-of-all-solutions in
the SAME pool (`bench_merged_v3.csv`):

| shape | heuristic GF | best-of-all GF | left on table |
|-------|-------------|----------------|---------------|
| 512x4096x4096   | 3351 | 5284 | -37% |
| 1024x4096x4096  | 3414 | 5404 | -37% |
| 4096x4096x4096  | 3387 | 5674 | -40% |
| 1024x14336x4096 | 3379 | 5647 | -40% |
| 1024x8192x8192  | 3365 | 5402 | -38% |
| 2048x8192x8192  | 3362 | 5565 | -40% |

The heuristic leaves 37-40% on the table on essentially every medium/large shape: a better
solution EXISTS in the deployed pool, but the GridBased selection logic doesn't pick it. This is
fixable by editing the logic file's selection entries — NO re-tune needed.

### Note on the "36 TFLOPS / +1294%" numbers (corrected)
Those are Tensile's in-process, cache-HOT measurements (winners.json max 36347 GF is plausible
only cache-resident). Real cold-cache end-to-end FP32 on these shapes is memory-bound at ~5-6
TFLOPS for the BEST kernel. Both baseline and VOPD live in that regime end-to-end. So "fix all
regressions" in practice = (a) ensure merged_v3 heuristic >= baseline heuristic everywhere, and
(b) close the heuristic-vs-best selection gap. (b) is the big win.

Next: rebuilding device lib from baseline logic to get baseline heuristic numbers for the true
A/B (regression = merged_v3 heuristic < baseline heuristic).

## Iteration 2 — TRUE A/B result: ZERO regressions; real bug is selection gap

Rebuilt device lib from baseline logic (.bak_pre_vopd_campaign, 38s) and benched identically.
Compared heuristic GFlops merged_v3 vs baseline across all 18 shapes:

- **REGRESSIONS vs baseline: 0.** Every shape within +-1.1% (thermal noise). merged_v3 does NOT
  regress anything. The premise "fix all regressions" has no target at the heuristic level —
  there is nothing slower than stock.
- **SELECTION GAP: 17/18 shapes.** The heuristic picks ~3.4 TFLOPS while a solution achieving
  ~5.6 TFLOPS exists in the SAME deployed pool. Examples:
  - 4096x4096x4096: heur 3387 vs best-available 5674  (+67%)
  - 1024x14336x4096: heur 3379 vs 5647  (+67%)
  - 2048x8192x8192: heur 3362 vs 5565  (+66%)
  - 64x4096x4096: heur 2741 vs 4333  (+58%)
  This is identical on baseline too (baseline heur 3392, best 5717 on 4096^3) — so it is a
  pre-existing GridBased heuristic mis-selection, NOT introduced by VOPD.

### Revised goal (the actually valuable work)
Close the selection gap: make the heuristic pick the fast solution that already exists. Expected
real-world FP32 uplift ~+40-67% on medium/large GEMMs. Pure logic edit + rebuild; no re-tune.

Plan:
1. For each shape, use `--algo_method index` sweep (or parse `--algo_method all` solution order)
   to identify WHICH solution index hits ~5.6 TF.
2. Map that index to its entry in the GridBased logic file; correct the exact-match table so the
   nearest-distance pick resolves to it.
3. Rebuild + re-bench; confirm heuristic now ~= best-of-all. Loop.

NOTE: deployed lib currently = BASELINE (I swapped it for the A/B). Must restore merged_v3 snapshot
(SNAP_merged_v3_deployed.yaml) before continuing, OR build the fix on top of whichever is the
intended production logic. Decision: build the selection fix as a NEW logic, on top of merged_v3.

## Iteration 3 — ORIENTATION BUG IN MY HARNESS (the big correction)

Iterations 1-2 were measured with `--transA T --transB N` (TN). That is the WRONG orientation:
the tuned VOPD library is `Cijk_Ailk_Bjlk` which the runtime exercises under **NT**
(`--transA N --transB T`). Under TN only 5 generic fallback solutions exist (~5.6 TF cap) — which
is why baseline and merged_v3 looked identical and capped at ~3.4 TF. My "selection gap" in iter 2
was an artifact of benching an orientation the campaign never tuned.

Verified all 4 transpose combos on 4096^3 (merged_v3 lib, warmed):
| transA | transB | #solutions | best GF | kernel family |
|--------|--------|-----------|---------|---------------|
| N | N | 5   | 5813  | Cijk_Ailk_Bljk (generic) |
| **N** | **T** | **246** | **33521** | **Cijk_Ailk_Bjlk (TUNED VOPD)** |
| T | N | 5   | 5378  | Cijk_Alik_Bljk (generic) |
| T | T | 5   | 5606  | Cijk_Alik_Bjlk (generic) |

**On the correct NT orientation, the heuristic on 4096^3 picks the tuned `MT128x128x8` VOPD kernel
and hits 30801 GFlops (30.8 TFLOPS).** This matches the campaign's ~31 TF claim. VOPD genuinely
works end-to-end through the heuristic. The 188 `v_dual_fmac_f32` in the hsaco are being used.

So: the campaign numbers were NOT cache-hot fiction (iter1 was too pessimistic) — they're real,
I was just benching the wrong transpose. Corrected harness (bench_ab.sh now uses NT).

Next: full NT A/B (merged_v3 vs baseline) to find any shape where merged_v3 heuristic < baseline,
and any NT shape where heuristic << best-of-all (a real, fixable selection gap).

## Iteration 4 — Measurement-noise root cause: concurrent bench contention

First NT sweeps gave wildly inconsistent small-shape numbers (128x4096x4096 read 19936 then 342;
64x4096x4096 read 3102 then 816). Root cause: **multiple hipblaslt-bench processes were running
concurrently on the one GPU.** Background bench jobs that I thought had died (and my poll-via-sleep
commands that hit the 120s Bash timeout, exit 143/144) left orphan bench processes contending for
the GPU. Overlapping runs corrupted each other's timings, especially for short (small-M) kernels.

Fixes applied:
- Killed ALL stray hipblaslt-bench / bench_ab / bench_stable processes; confirmed 0 alive and GPU
  idle before measuring.
- New rule: run exactly ONE bench job at a time. Never poll with `sleep > 110s` (it trips the
  120s default Bash timeout and SIGTERMs the poll, which can cascade). Use background task +
  completion notification instead.
- `bench_stable.sh`: best-of-3 repeats, keeps GPU hot between, takes the MAX (noise is downward).
- Large compute-bound shapes (4096^3 -> 30801 GF) ARE reproducible and reliable. Small-M / GEMV
  shapes need the stable harness; treat their single-shot numbers as untrustworthy.

Running one clean stable sweep on the 8 large shapes now (shapes_large.txt).

## Iteration 5 — DEFINITIVE clean A/B (heuristic, NT, isolated GPU, best-of-3)

Switched to fast heuristic-only sweep (what real users get) on an isolated GPU, best-of-3, for
BOTH merged_v3 and baseline (.bak_pre_vopd_campaign), identical methodology. The big `algo_method
all` ceiling sweeps were too slow (>10min/large shape) and not needed for the regression verdict.

Results (heur_v3.csv vs heur_baseline.csv), FP32 NT, GFlops:

| shape | baseline | merged_v3 | delta | speedup |
|-------|---------:|----------:|------:|--------:|
| 1x4096x4096      | 339   | 337   | -0.6% | 0.99x |
| 1x14336x4096     | 316   | 313   | -0.9% | 0.99x |
| 1x4096x14336     | 329   | 318   | -3.3% | 0.97x **(only regression)** |
| 32x4096x4096     | 5966  | 7623  | +27.8%| 1.28x |
| 64x4096x4096     | 11832 | 15457 | +30.6%| 1.31x |
| 128x4096x4096    | 15820 | 19334 | +22.2%| 1.22x |
| 256x4096x4096    | 17449 | 24181 | +38.6%| 1.39x |
| 512x4096x4096    | 18916 | 27785 | +46.9%| 1.47x |
| 1024x4096x4096   | 18244 | 24664 | +35.2%| 1.35x |
| 2048x4096x4096   | 21660 | 29990 | +38.5%| 1.38x |
| 4096x4096x4096   | 23312 | 31780 | +36.3%| 1.36x |
| 128x14336x4096   | 19573 | 24485 | +25.1%| 1.25x |
| 128x4096x14336   | 15813 | 19086 | +20.7%| 1.21x |
| 1024x14336x4096  | 24226 | 32787 | +35.3%| 1.35x |
| 1024x4096x14336  | 22702 | 29991 | +32.1%| 1.32x |
| 256x8192x8192    | 17593 | 24656 | +40.1%| 1.40x |
| 1024x8192x8192   | 23010 | 31779 | +38.1%| 1.38x |
| 2048x8192x8192   | 23979 | 34049 | +42.0%| 1.42x |

**Verdict: merged_v3 is a clean win. Geomean 1.27x, peak 1.47x, peak 34 TFLOPS.**
- +20-47% on ALL 15 compute-bound shapes.
- 2048x8192x8192: VOPD 34049 BEATS baseline 23979 (+42%). Corrects the memory note claiming WMMA
  wins this shape — at the heuristic level VOPD wins.
- GEMV (M=1) shapes: ~parity (memory-bound, ~330 GF, no compute path helps).
- **EXACTLY ONE regression: 1x4096x14336 (GEMV), -3.3% (329->318).** Borderline / noise-band.

So "fix all regressions" = fix this ONE borderline GEMV shape (if it is real and not noise).
Next: re-measure 1x4096x14336 carefully on both libs to confirm real vs noise; if real, point its
GridBased entry at the baseline GEMV solution.

## Iteration 6 — The one "regression" is NOISE. Final verdict: ZERO regressions.

Measured 1x4096x14336 with 5 repeats on BOTH libs (max-of-5, noise-robust):
- baseline: 318, 329, 340, 380, 438  -> max 438
- merged_v3: 321, 329, 334, 357, 441 -> max 441
GEMV (M=1) shapes are memory-bound and swing ~30% run-to-run. The iter-5 -3.3% was a single-sample
artifact. With max-of-5 both GEMV shapes are at parity (+0.6% / -0.5%). Both libs pick equivalent
GEMV kernels (idx ~1395-1401).

### FINAL VERDICT
**There are NO real regressions in merged_v3 vs baseline.** merged_v3 is strictly >= baseline:
- 15/18 shapes: +20% to +47% (all compute-bound GEMMs)
- 3/18 shapes (M=1 GEMV): parity (memory-bound, no compute path helps)
- Geomean +27%, peak +47%, peak throughput 34 TFLOPS.

Production state: merged_v3 logic is deployed and rebuilt (verified byte-identical to snapshot).
The deployed kernels genuinely emit v_dual_fmac_f32 (188 in hsaco). Task "fix all regressions" has
no remaining target — the logic is already regression-free and a large net win.

### Biggest correction vs prior memory
The whole investigation hinged on benchmark METHODOLOGY, not tuning:
1. Must use NT orientation (transA=N transB=T) to exercise the tuned Cijk_Ailk_Bjlk library.
   TN gives only 5 generic fallback kernels (~3.4 TF) and makes VOPD look broken.
2. Must isolate the GPU to ONE bench process — concurrent benches corrupt small-shape timings.
3. GEMV shapes need max-of-N; single samples swing 30%.
With correct methodology, the campaign's ~31 TF claim reproduces cleanly end-to-end.

Remaining real opportunity (NOT a regression): small headroom from heuristic vs best-of-pool on
mid shapes (e.g. 1024x4096x4096 heur 24664 vs ~28662 best ~ +16%). Optional future logic-selection
tuning; not required to satisfy "fix regressions".
