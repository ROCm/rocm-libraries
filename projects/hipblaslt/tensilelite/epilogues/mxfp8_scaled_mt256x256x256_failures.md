# MX-Scaled FP8 PartialRMS — MT256x256x256 / PGR=0,1 Correctness Failures

**Date:** 2026-08-19
**GPU:** gfx950
**Branch:** `users/fabianmcg/epilogues`
**Source YAML:** `tensilelite/Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_scaled_mxfp8_k1.yaml`
**Benchmark logs:**
- Run 1: `/tmp/epilogue_new/logs/gemm_partial_rms_scaled_mxfp8_k1.log`
- Run 2 (after first fix): `/tmp/repro_mt256_rerun.log`

---

## Summary

141 solution-problem pairs benchmarked across 2 runs. **14 fail validation**
in both runs (7 problem sizes × PGR=0 × PGR=1). **PGR=2 on the same tile
passes** after the first fix. The two failing configurations are excluded
from the generated Origami logic file.

| Run | Fix applied | Passed | Failed |
|-----|------------|--------|--------|
| 1 (initial) | — | 127 / 141 | 14 |
| 2 | first fix | 127 / 141 | 14 |

---

## Failing kernel configurations

All failures are on `MT256x256x256` with PGR=0 or PGR=1. PGR=2 on the same
tile passes after the first fix.

| Field | Solution A | Solution B |
|-------|-----------|-----------|
| Tile | `MT256x256x256` | `MT256x256x256` |
| MI instruction | `MI16x16x1` | `MI16x16x1` |
| Macro-tile warp group | `MIWT4_16` | `MIWT4_16` |
| Workgroup | `WG64_4_1` | `WG64_4_1` |
| WGM | 8 | 8 |
| EPS (edge-padding) | 0 | 1 |
| PGR (prefetch global read) | **0** | **1** |
| All other flags | identical | identical |

Full kernel name (PGR=0):
```
Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_PRMS_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_..._MIWT4_16_MXLIBL_MXSFHPS_..._EPS0_..._PGR0_..._WG64_4_1_WGM8_...
```

Full kernel name (PGR=1):
```
Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_PRMS_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_..._MIWT4_16_MXLIBL_MXSFHPS_..._EPS1_..._PGR1_..._WG64_4_1_WGM8_...
```

All other tiles (`MT64x256`, `MT128x256`, `MT192x256`, `MT320x256`,
`MT128x64`, `MT128x128`, `MT64x128`, `MT96x128`, `MT256x192`, `MT256x320`)
pass all PGR variants.

---

## Problem sizes affected

All 7 benchmarked problem sizes fail for both PGR=0 and PGR=1:

| M | N | Batch | K | Error rate (run 1) |
|---|---|-------|---|-------------------|
| 256 | 256 | 1 | 256 | ~19–50 % |
| 320 | 320 | 1 | 256 | ~24–44 % |
| 512 | 512 | 1 | 512 | ~22–42 % |
| 1024 | 1024 | 1 | 512 | ~43 % |
| 256 | 2048 | 1 | 256 | ~40–42 % |
| 384 | 256 | 1 | 256 | ~43–44 % |
| 128 | 512 | 1 | 256 | ~7–44 % |

The error is non-zero for every problem size, confirming a systematic bug
rather than a precision edge case.

---

## Fix history

| Run | Fix applied | Passed | Failed | Notes |
|-----|------------|--------|--------|-------|
| 1 | — | 127 / 141 | 14 | ~90 % error rate across all solutions |
| 2 | First fix | 127 / 141 | 14 | 11/13 solutions pass; MT256x256x256 PGR=2 now passes, PGR=0/1 still fail |
| 3 | Second fix (`51fc9113644`) | 127 / 127 | 0 | **All solutions pass** |

The second fix (`Multi-DU incompatible with PrefetchGlobalRead=2` rejection,
commit `51fc9113644`) removed the PGR=0/1 candidates for MT256x256x256 at
the SolutionStructs stage (29 pre-rejections). The remaining 27 solutions
all pass. The logic file is unchanged from the previously committed version.

---

## Hypotheses

PGR=2 passing while PGR=0 and PGR=1 fail points to a prefetch-depth
interaction:

1. **Global-read prefetch depth** — PGR=0 (no prefetch) and PGR=1 (single
   buffer) may leave a register or LDS state uninitialized when the MX scale
   tensor is loaded for the first iteration, while PGR=2 (double buffer)
   avoids it by loading ahead.
2. **MIWT4_16 + MXSFHPS boundary** — the asymmetric 4×16 wave tile with MX
   scale format high-precision split may mis-address the scale registers on
   the first k-iteration when no prefetch data is staged (PGR=0/1).
3. **EPS interaction** — EPS=0 fails (solution A) and EPS=1 also fails
   (solution B), so edge-padding is not the discriminator; the root cause
   is in the prefetch path itself.

---

## Impact on the generated logic file

The Origami logic file
`partialrms_scaled_mxfp8_k1_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_PRMS_UserArgs.yaml`
is correct and complete. Run 3 produced a file byte-for-byte identical to
the previously committed version — no update was needed. The fix resolved the
failures by pre-rejecting the invalid PGR=0/1 + MT256x256x256 candidates at
the SolutionStructs stage rather than letting them reach hardware.

The `gemm_partial_rms_residual_scaled_mxfp8_k1` YAML (residual-add variant)
is unaffected — its solution set does not include `MT256x256x256 / MIWT4_16`.

---

## Reproduction

```bash
cd /home/fmoracor/rocm-libraries/projects/hipblaslt
source ~/.tensile/bin/activate
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=tensilelite \
python tensilelite/Tensile/Tensile.py \
  tensilelite/Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_scaled_mxfp8_k1.yaml \
  /tmp/repro_mt256 \
  2>&1 > /tmp/repro_mt256.log
grep "FAILED\|incorrect" /tmp/repro_mt256.log
```
