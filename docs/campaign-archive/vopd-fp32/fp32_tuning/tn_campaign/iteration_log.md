# VOPD TN-orientation Tuning Campaign — Log

Goal: tune FP32 VOPD for the TN orientation (transA=T transB=N, A^T*B, library Cijk_Alik_Bljk) on
gfx1100. TN was never tuned; today it falls back to ~5 generic kernels (~5.9 TFLOPS on 4096^3).
Full grid, same 1030-shape set as the NT campaign.

## Setup (done)
- gfx1100 == navi31. TN baseline = navi31/Equality/navi31_Cijk_Alik_Bljk_SB_Bias_HAS_SAV_UserArgs.yaml
  (1273 lines, snapshot: SNAP_tn_baseline_navi31.yaml). No gfx1100 TN GridBased file exists yet;
  the campaign will CREATE gfx1100/GridBased/gfx1100_Cijk_Alik_Bljk_S_B_UserArgs.yaml.
- TN wave configs built by copying vopd_campaign/wave{1-4}.yaml and flipping ONLY the transpose
  line to `TransposeA: true, TransposeB: false`. Verified: 1 line changed, shape counts preserved
  (50/240/320/250 = 860 tuned + GEMV; 1030 total across waves... NT had 1030).

## Go/no-go SMOKE TEST — PASSED
1 shape (4096^3), 1 config (MT128x128/DU16/TT8x8), TN, EnableVOPD=1:
- Tensile clientExit=0 (PASS), ran in ~4.4s.
- **TN VOPD = 22077 GFlops (22 TFLOPS)** untuned, vs ~5.9 TF generic fallback = **3.7x already**.
- Kernel emits 15 v_dual_fmac_f32 (519 total v_dual_*). Dual-issue confirmed on TN.
- VOPD is orientation-independent as predicted; no code blocker.
(Minor non-fatal warnings: DirectToLdsMetadata bool, GlobalReadPerMfma int type mismatches — same
as NT campaign, cosmetic.)

Tensile invocation (from NT logs):
`~/TheRock/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile <wave.yaml> <out_dir>`

## Waves
- [x] wave1 (50 shapes) — DONE 12:24, clientExit=0 PASS, 815s. Peak 365 GF (tiny shapes 16x16..128x16, expected low).
- [~] wave2 (239 shapes) — RUNNING (driver run_waves.sh, started 12:24:27)
- [ ] wave3 (320 shapes) — queued in driver
- [ ] wave4 (250 shapes) — queued in driver
Driver: `run_waves.sh 2 3 4` chains them sequentially; log `run_waves.log`.
NT timing reference: wave2 1h23m, wave3 1h02m, wave4 3h37m => waves2-4 ~6h.

## Autonomous pipeline (launched 12:29)
Full downstream automated so it completes without babysitting:
- `run_waves.sh 2 3 4` — chains tuning waves 2->3->4 (driver PID, run_waves.log).
- `watcher.sh` (PID 988327, watcher.log) — polls for clientExit in all 4 wave logs, then runs:
  1. `build_merge_tn.sh` — per-wave LibraryLogic gen + force_merge waves 1->2->3->4 = merged_tn/
     (build_merge.log). Untested shapes fall back to navi31 generic TN kernels (already in build).
  2. `deploy_verify_tn.sh` — capture TN baseline (tn_before.csv), deploy merged_tn to
     gfx1100/GridBased/gfx1100_Cijk_Alik_Bljk_S_B_UserArgs.yaml, rebuild device lib, bench
     (tn_after.csv). (deploy_verify.log)
- Done marker: "PIPELINE_COMPLETE" in watcher.log; "TN_VERIFY_DONE" in deploy_verify.log.
Recipe verified against NT: KNAME=Cijk_Alik_Bljk_S_B_UserArgs, create_library.yaml = GridBased.
TN waves cover 860 unique shapes incl 20 M=1 + 10 N=1 GEMV.

Smoke result preview: untuned TN MT128x128/DU16 already = 22 TFLOPS (vs 5.9 generic). Expect tuned
TN to approach NT's ~31 TF on large shapes.

## CAMPAIGN COMPLETE — 2026-06-02 21:27 (PIPELINE_COMPLETE)

All 4 waves done; merge + deploy + rebuild + cold-cache A/B verify all succeeded.

### Final cold-cache results (tn_before.csv vs tn_after.csv, FP32 TN, best-of-3, isolated GPU)
- **Peak: 31.9 TFLOPS** (2048x8192x8192) end-to-end. (Tensile cache-warm peak 33.1 TF on 6144x8192x8192.)
- **Geomean speedup 4.63x, max 9.46x** vs the untuned generic fallback.
- 4096^3: 3427 -> 29558 (8.63x). 1024x14336x4096: 3399 -> 29731 (8.75x). 2048x8192x8192: 3373 -> 31894 (9.46x).
- GEMV M=1: parity (~1.0x, memory-bound, expected).

### Winner divergence vs NT (waves1-3, 610 common shapes)
TN winners share NT's tile vocabulary but differ: only 26% identical full config; 64% same tile shape.
Key systematic diffs: TN prefers DepthU=16 (NT spreads DU8/4); 41 shapes are pure MT0<->MT1 swaps
(orientation transpose mirror, e.g. NT 64x128 -> TN 128x64); 103 differ only in WGM. Confirms tuning
TN separately was correct, not reusable from NT by relabeling.

### Deployment state
- Tuned TN logic deployed: gfx1100/GridBased/gfx1100_Cijk_Alik_Bljk_S_B_UserArgs.yaml (59631 lines).
  Source: tn_campaign/merged_tn/. Device lib rebuilt. TN now live alongside NT.
- NT production logic (Cijk_Ailk_Bjlk) untouched. Nothing committed/pushed (no user authorization).

### Waves final
- wave1 50 shapes DONE | wave2 239 DONE | wave3 319 DONE | wave4 249 DONE (clientExit=0 each).
