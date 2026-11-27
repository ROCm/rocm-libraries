# OOB BBS TN exact tuning, category 18: Medium batch (gfx1250, BF16 in/out, FP32 compute, TN)

Three variants are provided for this category: `fused`, `pure_lda` and `tdmsi_tf`.

## fused: fused GEMM, no 256 B alignment

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with LDA = LDB = K and LDC = LDD = M, i.e. the leading dimensions are not padded to 256 B.

- `fused/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size (fused GEMM: Bias (f32) + ScaleAlphaVec, ActivationFuncCall, `Exact: [M, N, batch, K]`, LDA = LDB = K, LDC = LDD = M, RotatingBufferSize 384).
- `fused/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below, 10 solutions.

### Results (hipblaslt-bench, GPU0, fused GEMM, LDA = LDB = K, tuned winner vs OOB)

Category mean: speedup vs OOB **+45.7%**, vs MAIDAS **43.0% -> 58.3%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P11 | 320x6240x135x7808 | 1656.75 | 2595.00 | 2983.50 | -13.0% | 63.8% | 55.5% |
| P12 | 2080x4496x184x752 | 1024.48 | 3915.00 | 1866.00 | +109.8% | 26.2% | 54.9% |
| P13 | 448x6464x214x448 | 297.29 | 890.00 | 693.33 | +28.4% | 33.4% | 42.9% |
| P14 | 3248x1248x233x3984 | 2503.24 | 5409.00 | 3960.00 | +36.6% | 46.3% | 63.2% |
| P15 | 5232x1328x283x2272 | 3015.34 | 7046.00 | 5190.00 | +35.8% | 42.8% | 58.1% |
| P16 | 96x1744x353x6960 | 668.28 | 1079.33 | 868.00 | +24.3% | 61.9% | 77.0% |
| P17 | 288x1520x355x4816 | 778.54 | 1302.50 | 859.00 | +51.6% | 59.8% | 90.6% |
| P18 | 3264x336x373x2496 | 895.84 | 1947.00 | 1635.00 | +19.1% | 46.0% | 54.8% |
| P19 | 1120x656x382x512 | 139.01 | 479.14 | 384.86 | +24.5% | 29.0% | 36.1% |
| P20 | 4944x1984x484x672 | 2578.11 | 12476.00 | 5198.00 | +140.0% | 20.7% | 49.6% |

### Winner key parameters

| # | size | winner |
|---|---|---|
| P11 | 320x6240x135x7808 | MT160x224x128, AG0, CD2_2, CLS0, MIWT5_7, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG32_4_1 |
| P12 | 2080x4496x184x752 | MT256x256x64, AG1, CD1_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |
| P13 | 448x6464x214x448 | MT64x128x64, AG0, CD1_1, CLS0, MIWT4_4, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG16_4_1 |
| P14 | 3248x1248x233x3984 | MT256x256x64, AG1, CD2_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |
| P15 | 5232x1328x283x2272 | MT256x224x64, AG1, CD2_2, CLS0, MIWT8_7, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB1, WG32_4_1 |
| P16 | 96x1744x353x6960 | MT96x256x128, AG0, CD1_1, CLS0, MIWT3_8, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB4, WG32_4_1 |
| P17 | 288x1520x355x4816 | MT144x256x128, AG0, CD2_1, CLS0, MIWT9_4, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB4, WG16_8_1 |
| P18 | 3264x336x373x2496 | MT256x192x64, AG1, CD2_2, CLS0, MIWT8_6, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB2, WG32_4_1 |
| P19 | 1120x656x382x512 | MT128x96x64, AG0, CD1_1, CLS0, MIWT4_3, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_4_1 |
| P20 | 4944x1984x484x672 | MT256x256x64, AG1, CD4_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |

## pure_lda: pure GEMM, 256 B aligned leading dimensions

Pure GEMM (no Bias / ScaleAlphaVec / activation) with the leading dimensions padded for 256 B alignment (BF16, 2 B per element). `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`, RotatingBufferSize 384.

- `pure_lda/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]` with the leading dimensions of the original problem, RotatingBufferSize 384.
- `pure_lda/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`: merged ExactLogic for the 10 sizes below.

### Results (hipblaslt-bench, GPU0, pure GEMM, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+55.7%**, vs MAIDAS **43.4% -> 62.5%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P11 | 320x6240x135x7808 | 1656.75 | 2541.50 | 2593.50 | -2.0% | 65.2% | 63.9% |
| P12 | 2080x4496x184x752 | 1024.48 | 3828.00 | 1755.00 | +118.1% | 26.8% | 58.4% |
| P13 | 448x6464x214x448 | 297.29 | 885.00 | 617.67 | +43.3% | 33.6% | 48.1% |
| P14 | 3248x1248x233x3984 | 2503.24 | 5376.00 | 3729.00 | +44.2% | 46.6% | 67.1% |
| P15 | 5232x1328x283x2272 | 3015.34 | 7131.00 | 4926.00 | +44.8% | 42.3% | 61.2% |
| P16 | 96x1744x353x6960 | 668.28 | 1091.00 | 872.33 | +25.1% | 61.3% | 76.6% |
| P17 | 288x1520x355x4816 | 778.54 | 1243.50 | 804.00 | +54.7% | 62.6% | 96.8% |
| P18 | 3264x336x373x2496 | 895.84 | 1957.00 | 1558.50 | +25.6% | 45.8% | 57.5% |
| P19 | 1120x656x382x512 | 139.01 | 479.29 | 354.00 | +35.4% | 29.0% | 39.3% |
| P20 | 4944x1984x484x672 | 2578.11 | 12234.00 | 4569.00 | +167.8% | 21.1% | 56.4% |

### Problem sizes and leading dimensions

| # | size (MxNxBatchxK) | LDA | LDB | LDC | LDD |
|---|---|---|---|---|---|
| P11 | 320x6240x135x7808 | 7808 | 7808 | 320* | 320* |
| P12 | 2080x4496x184x752 | 768 | 768 | 2176 | 2176 |
| P13 | 448x6464x214x448 | 448* | 448* | 448* | 448* |
| P14 | 3248x1248x233x3984 | 4096 | 4096 | 3328 | 3328 |
| P15 | 5232x1328x283x2272 | 2304 | 2304 | 5248 | 5248 |
| P16 | 96x1744x353x6960 | 7040 | 7040 | 96* | 96* |
| P17 | 288x1520x355x4816 | 4864 | 4864 | 384 | 384 |
| P18 | 3264x336x373x2496 | 2496* | 2496* | 3264* | 3264* |
| P19 | 1120x656x382x512 | 512 | 512 | 1152 | 1152 |
| P20 | 4944x1984x484x672 | 768 | 768 | 4992 | 4992 |

`*`: not a multiple of 256 B.

## tdmsi_tf: TDM store + tail folding, fused GEMM with CompactLoopStore, 256 B aligned leading dimensions

Winners of the TDM store re-tune (`TDMStoreInst` in the search space, [#15133](https://github.com/ROCm/rocm-libraries/pull/15133)) on a build with tail folding ([#16312](https://github.com/ROCm/rocm-libraries/pull/16312)), re-tuned with a wider `DepthU` list, then rebuilt as fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with `CompactLoopStore` on. Leading dimensions padded for 256 B alignment (same as `pure_lda`), RotatingBufferSize 384.

- `tdmsi_tf/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the TDM store tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `TDMStoreInst: [false, true]`, `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`. The logic winners come from re-tuning these yamls with a wider `DepthU` list (e.g. `[128, 256, 512]` instead of `[128]`, where LDS allows); only the P12 yaml here already contains it.
- `tdmsi_tf/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below; the winners with the epilogue added and `CompactLoopStore: true`.

### Results (hipblaslt-bench C++ API, GPU0, fused GEMM + CompactLoopStore, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+69.6%**, vs MAIDAS **54.7% -> 86.0%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P11 | 320x6240x135x7808 | 1656.75 | 1891.00 | 1599.50 | +18.2% | 87.6% | 103.6% |
| P12 | 2080x4496x184x752 | 1024.48 | 3397.00 | 1502.00 | +126.2% | 30.2% | 68.2% |
| P13 | 448x6464x214x448 | 297.29 | 821.50 | 468.50 | +75.3% | 36.2% | 63.5% |
| P14 | 3248x1248x233x3984 | 2503.24 | 4245.00 | 2929.00 | +44.9% | 59.0% | 85.5% |
| P15 | 5232x1328x283x2272 | 3015.34 | 5749.00 | 3561.00 | +61.4% | 52.4% | 84.7% |
| P16 | 96x1744x353x6960 | 668.28 | 870.25 | 591.00 | +47.3% | 76.8% | 113.1% |
| P17 | 288x1520x355x4816 | 778.54 | 870.00 | 648.67 | +34.1% | 89.5% | 120.0% |
| P18 | 3264x336x373x2496 | 895.84 | 1680.50 | 967.00 | +73.8% | 53.3% | 92.6% |
| P19 | 1120x656x382x512 | 139.01 | 441.86 | 225.86 | +95.6% | 31.5% | 61.5% |
| P20 | 4944x1984x484x672 | 2578.11 | 8334.00 | 3811.00 | +118.7% | 30.9% | 67.6% |

### Winner key parameters

`TDMStore1` = `TDMStoreInst: true` (TDM store), `TDMStore0` = buffer store.

| # | size | winner |
|---|---|---|
| P11 | 320x6240x135x7808 | MT160x256x128, AG0, CD2_2, CLS1, MIWT5_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW1, VWA1, VWB4, WG32_4_1, TDMStore1 |
| P12 | 2080x4496x184x752 | MT256x256x128, AG0, CD1_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P13 | 448x6464x214x448 | MT112x128x64, AG0, CD1_1, CLS1, MIWT7_4, PAP0, PGR2, PLR1, RAP0, SS0, SVW8, VWA1, VWB4, WG16_4_1, TDMStore1 |
| P14 | 3248x1248x233x3984 | MT256x256x128, AG0, CD1_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P15 | 5232x1328x283x2272 | MT256x224x128, AG0, CD1_1, CLS1, MIWT8_7, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_4_1, TDMStore1 |
| P16 | 96x1744x353x6960 | MT96x256x128, AG0, CD1_1, CLS1, MIWT3_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW1, VWA1, VWB4, WG32_4_1, TDMStore1 |
| P17 | 288x1520x355x4816 | MT144x256x128, AG0, CD2_2, CLS1, MIWT9_4, PAP0, PGR2, PLR1, RAP0, SS0, SVW8, VWA1, VWB4, WG16_8_1, TDMStore1 |
| P18 | 3264x336x373x2496 | MT256x176x128, AG0, CD1_2, CLS1, MIWT4_11, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG64_2_1, TDMStore1 |
| P19 | 1120x656x382x512 | MT128x112x64, AG0, CD1_1, CLS1, MIWT4_7, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_2_1, TDMStore1 |
| P20 | 4944x1984x484x672 | MT256x256x128, AG0, CD4_2, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |

## Notes

- The `fused` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, fused GEMM (Bias + ScaleAlphaVec), LDA = LDB = K.
- The `pure_lda` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, pure GEMM, 256 B aligned leading dimensions.
- The `tdmsi_tf` results above are measured with hipblaslt-bench (C++ API) on GPU0: tuned winner vs OOB baseline, fused GEMM with CompactLoopStore, 256 B aligned leading dimensions, build develop + #15133 (84a8b39, includes tail folding).
