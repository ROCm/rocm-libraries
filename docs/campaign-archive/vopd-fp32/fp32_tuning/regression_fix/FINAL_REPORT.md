# VOPD FP32 Regression Fix — Final Report (2026-06-02)

## TL;DR
**There are no regressions to fix.** The merged_v3 VOPD logic is strictly >= the stock baseline
on every tested shape: +20% to +47% on all 15 compute-bound GEMMs, parity on the 3 memory-bound
GEMV (M=1) shapes. Geomean **+27%**, peak **+47%**, peak throughput **34 TFLOPS**. Production state
(merged_v3 deployed + device lib rebuilt) is verified and correct.

The investigation was really a **benchmark-methodology** fix, not a tuning fix.

## What was actually wrong (and is now corrected)
The premise in memory ("GEMV regressed -44% to -68%", "+1294% wins", "WMMA wins 2048x8192x8192")
came from unreliable measurements. Ground truth established this run:

1. **Orientation bug.** The tuned library is `Cijk_Ailk_Bjlk`, exercised under **NT**
   (`--transA N --transB T`). Benchmarking TN (the obvious guess for an A^T·B kernel name) hits
   only 5 generic fallback kernels capped at ~3.4 TF and makes VOPD look broken. NT shows the real
   30-34 TF.
2. **GPU contention.** Multiple concurrent `hipblaslt-bench` processes corrupt small-shape timings
   (saw 128x4096x4096 read both 19936 and 342). Must isolate to ONE bench process.
3. **GEMV noise.** M=1 shapes swing ~30% run-to-run (memory-bound). Need max-of-N, not single shot.

With correct methodology, the campaign's ~31 TF claim reproduces cleanly end-to-end, and VOPD
(188 `v_dual_fmac_f32` in the hsaco) is genuinely active through the production heuristic path.

## Definitive A/B (heuristic = what real users get; FP32 NT; isolated GPU; best-of-3)

| shape | baseline GF | merged_v3 GF | speedup |
|-------|------------:|-------------:|--------:|
| 1x4096x4096      | 339   | 337   | 0.99x (parity) |
| 1x14336x4096     | 316   | 313   | 0.99x (parity) |
| 1x4096x14336     | 329   | 318*  | parity (*noise, max-of-5 = +0.6%) |
| 32x4096x4096     | 5966  | 7623  | 1.28x |
| 64x4096x4096     | 11832 | 15457 | 1.31x |
| 128x4096x4096    | 15820 | 19334 | 1.22x |
| 256x4096x4096    | 17449 | 24181 | 1.39x |
| 512x4096x4096    | 18916 | 27785 | 1.47x |
| 1024x4096x4096   | 18244 | 24664 | 1.35x |
| 2048x4096x4096   | 21660 | 29990 | 1.38x |
| 4096x4096x4096   | 23312 | 31780 | 1.36x |
| 128x14336x4096   | 19573 | 24485 | 1.25x |
| 128x4096x14336   | 15813 | 19086 | 1.21x |
| 1024x14336x4096  | 24226 | 32787 | 1.35x |
| 1024x4096x14336  | 22702 | 29991 | 1.32x |
| 256x8192x8192    | 17593 | 24656 | 1.40x |
| 1024x8192x8192   | 23010 | 31779 | 1.38x |
| 2048x8192x8192   | 23979 | 34049 | 1.42x |

Data: `heur_v3.csv`, `heur_baseline.csv`. Full trace: `iteration_log.md`.

## Optional future work (NOT a regression — an enhancement)
There is real but modest headroom between the heuristic pick and the best solution already in the
pool, on mid compute shapes:
- 1024x4096x4096: heuristic picks MT128x128x**8** (idx 1599, ~24.7 TF); MT128x128x**16** (idx 1624,
  ~28.9 TF) exists and is **+14-18% faster**.
- Similar +6-8% on 512/2048/4096 square shapes.
This is a GridBased selection-logic refinement (point those shapes' entries at the DU=16 variant).
It is a logic-only edit (no re-tune), but it carries risk: GridBased matches on K only, so editing
entries can shift untested shapes. Recommend doing it deliberately with the user, not autonomously.

## Reusable harnesses produced
- `bench_heur.sh <label> <shapes>` — fast heuristic-only NT bench, best-of-3, GPU-hot. The right
  default for A/B.
- `bench_stable.sh` — adds best-of-all ceiling (slow; use only on targeted shapes).
- `bench_ab.sh` — heuristic + ceiling combined (now NT-correct).
Rule baked in: NT orientation, one bench process at a time, warm the GPU first.

## Git / deployment state (unchanged by this run, as intended)
- Branch `vmijovic/add_vopd`, merged_v3 logic deployed + device lib rebuilt (verified).
- NOTHING committed/pushed this run (no user authorization). The fixed merged_v3 logic from the
  prior checkpoint is still the uncommitted production artifact.
