# OOB BBS TN exact tuning, category 2 (gfx1250, BF16 in/out, FP32 compute, TN)

Two variants, each in its own directory with a logic file and tuning configs. Both logic files hold only the 10 sizes of this
category, use the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true on every solution), and key
ExactLogic on [M, N, 1, K] only (no strides).

### Pure GEMM, LDA/LDB padded (`pure_LdaPadded/`)

- `pure_LdaPadded/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128.
- `pure_LdaPadded/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search for that size
  (all tuning rounds merged into one file, including a ClusterDim / ClusterAIDMapping / WorkGroupMapping sweep in which
  cluster candidates use ClusterAIDMapping -2; identical solutions removed; the header comment lists the merged sets
  and the candidate count). Group 1 = winner, group 2 = previous logic kernel (logic_1109a).
  Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

- `fused_ldaK/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
  Winners tuned as fused GEMM (Bias (f32) + ScaleAlphaVec, activation none) with unpadded leading dimensions.
- `fused_ldaK/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search of the
  fused retune for that size (all candidate sets merged into one file, including a ClusterDim sweep; identical
  solutions removed; the header comment gives the candidate count). Group 1 = winner of the pure padded logic
  (previous winner), group 2 = previous logic kernel (logic_1109a).
  Fused GEMM, Bias (f32) + ScaleAlphaVec, activation none, ActivationFuncCall true,
  `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K, LDC = LDD = M),
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

## Results, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`; Tensile client, GPU3, median of rounds 1+)

Category mean: gain vs previous **+5.4%**, vs MAIDAS **62.2% -> 64.9%**
(logic_1109a: 57.1%; 8 of 10 sizes have a new winner; vs MAIDAS = MAIDAS us / measured us).
Previous = winner of the previous pure padded logic of this directory; logic_1109a = previous logic kernel. All three
kernels of a size are measured in the same client run; if the previous winner measured faster it is kept.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | logic_1109a us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 464x1872x6736 | 6784 | 1 | 7.99 | 17.38 | 17.38 | 16.27 | +6.8% | 46.0% | 49.1% | new |
| 544x1728x7488 | 7552 | 4 | 8.71 | 18.52 | 18.43 | 17.94 | +2.7% | 47.2% | 48.5% | new |
| 1744x5376x4096 | 4096 | 7 | 25.85 | 39.21 | 32.88 | 32.44 | +1.3% | 78.6% | 79.7% | new |
| 2016x6112x4608 | 4608 | 8 | 36.67 | 55.76 | 46.22 | 46.22 | +0.0% | 79.3% | 79.3% | previous kept |
| 2960x1792x8176 | 8192 | 3 | 28.32 | 53.07 | 46.15 | 43.38 | +6.4% | 61.4% | 65.3% | new |
| 3760x1168x2176 | 2176 | 9 | 8.33 | 12.24 | 12.19 | 12.19 | +0.0% | 68.3% | 68.3% | previous kept |
| 4176x4048x752 | 768 | 0 | 10.63 | 29.12 | 26.44 | 20.63 | +28.2% | 40.2% | 51.5% | new |
| 6128x4784x3504 | 3584 | 5 | 67.70 | 91.57 | 88.28 | 83.53 | +5.7% | 76.7% | 81.0% | new |
| 6592x2592x3296 | 3328 | 6 | 37.51 | 52.55 | 50.93 | 50.36 | +1.1% | 73.6% | 74.5% | new |
| 7056x896x1248 | 1280 | 2 | 7.25 | 16.74 | 14.33 | 14.04 | +2.1% | 50.6% | 51.6% | new |

## Results, fused GEMM, LDA = LDB = K (`fused_ldaK/`; Tensile client, GPU0 and GPU3 (column), median of rounds 1+)

Category mean: gain vs previous **+10.6%**, vs MAIDAS **51.6% -> 55.1%**
(9 of 10 sizes have a new winner; the others keep the previous winner).
Fused GEMM: Bias (f32) + ScaleAlphaVec, activation none, LDA = LDB = K, LDC = LDD = M. Previous = winner of the
earlier pure padded logic, measured in the same client run as the new winner; a size keeps the previous winner
when it measured faster. Rows marked GPU0 come from one bench of all sizes on GPU0; rows marked GPU3 are the sizes
whose winner was re-selected afterwards, re-measured on GPU3 (compare gains, not absolute us, across GPUs).

| size (MxNxK) | sol | GPU | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|
| 464x1872x6736 | 1 | GPU0 | 7.99 | 23.09 | 21.21 | +8.9% | 34.6% | 37.7% | new |
| 544x1728x7488 | 4 | GPU0 | 8.71 | 19.05 | 18.85 | +1.1% | 45.7% | 46.2% | new |
| 1744x5376x4096 | 7 | GPU3 | 25.85 | 33.56 | 33.51 | +0.1% | 77.0% | 77.1% | new |
| 2016x6112x4608 | 8 | GPU0 | 36.67 | 48.17 | 47.93 | +0.5% | 76.1% | 76.5% | new |
| 2960x1792x8176 | 3 | GPU0 | 28.32 | 68.86 | 68.86 | +0.0% | 41.1% | 41.1% | kept |
| 3760x1168x2176 | 9 | GPU3 | 8.33 | 13.30 | 13.27 | +0.2% | 62.6% | 62.7% | new |
| 4176x4048x752 | 0 | GPU0 | 10.63 | 34.56 | 24.64 | +40.3% | 30.8% | 43.2% | new |
| 6128x4784x3504 | 5 | GPU0 | 67.70 | 125.44 | 114.82 | +9.2% | 54.0% | 59.0% | new |
| 6592x2592x3296 | 6 | GPU0 | 37.51 | 55.83 | 54.93 | +1.6% | 67.2% | 68.3% | new |
| 7056x896x1248 | 2 | GPU0 | 7.25 | 26.82 | 18.56 | +44.5% | 27.0% | 39.1% | new |

## Winner key parameters, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 464x1872x6736 | 1 | MT64x64x512, MIWT4_2, WG16_4_2, CD4_2, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 544x1728x7488 | 4 | MT64x64x512, MIWT4_2, WG16_4_2, CD1_4, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 1744x5376x4096 | 7 | MT256x160x128, MIWT8_5, WG32_4_1, CD1_4, TPS None, SS1, VWA8, VWB1, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 2016x6112x4608 | 8 | MT256x192x128, MIWT16_3, WG16_8_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA4, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |
| 2960x1792x8176 | 3 | MT192x112x256, MIWT3_7, WG64_2_1, CD4_4, CAIDMn2, TPS None, SS0, VWA1, VWB1, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM1 | new |
| 3760x1168x2176 | 9 | MT128x160x128, MIWT4_5, WG32_4_1, CD4_4, TPS None, SS1, VWA4, VWB1, SVW4, HPLR1, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | previous kept |
| 4176x4048x752 | 0 | MT256x320x128, MIWT8_10, WG32_4_1, CD2_2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM0, LPA4, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 6128x4784x3504 | 5 | MT256x160x128, MIWT8_5, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB1, SVW8, HPLR0, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6592x2592x3296 | 6 | MT256x288x64, MIWT8_9, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB1, SVW8, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 7056x896x1248 | 2 | MT128x112x128, MIWT2_7, WG64_2_1, CD4_4, TPS None, SS1, VWA2, VWB1, SVW2, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |

## Winner key parameters, fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 464x1872x6736 | 1 | MT128x256x64, MIWT4_8, WG32_4_1, CD2_1, TPS StreamK, SS1, VWA4, VWB8, SVW4, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 544x1728x7488 | 4 | MT64x64x512, MIWT4_2, WG16_4_2, CD1_4, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 1744x5376x4096 | 7 | MT256x160x128, MIWT8_5, WG32_4_1, CD1_2, TPS None, SS1, VWA8, VWB1, SVW8, HPLR1, LDSSI1, TDMIM0, LPA4, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 2016x6112x4608 | 8 | MT256x192x128, MIWT16_3, WG16_8_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB1, SVW8, HPLR1, LDSSI0, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 2960x1792x8176 | 3 | MT192x112x256, MIWT3_7, WG64_2_1, CD1_1, TPS None, SS1, VWA1, VWB1, SVW1, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | kept |
| 3760x1168x2176 | 9 | MT128x80x128, MIWT4_5, WG32_2_1, CD4_4, CAIDMn2, TPS None, SS1, VWA4, VWB1, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 4176x4048x752 | 0 | MT256x320x64, MIWT8_10, WG32_4_1, CD2_2, CAIDMn2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 6128x4784x3504 | 5 | MT256x256x64, MIWT8_8, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB8, SVW8, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6592x2592x3296 | 6 | MT256x288x64, MIWT8_9, WG32_4_1, CD4_1, TPS None, SS1, VWA8, VWB1, SVW8, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM4 | new |
| 7056x896x1248 | 2 | MT128x112x128, MIWT2_7, WG64_2_1, CD2_1, TPS DataParallel, SS1, VWA2, VWB1, SVW2, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on each logic file: 0 rejects.
- The two tables differ in LDA alignment (with LDA = K, K*2 bytes not a multiple of 128 makes TDM loads go
  indirect) and in the fused epilogue, so the two variants can pick different kernels for the same size.
- The tables come from different GPUs and runs (pure: GPU3; fused: GPU0, plus GPU3 rows where marked); compare gains
  within a row, not absolute us across tables.
- `pure_LdaPadded/logic/`: 2 solution(s) (2 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
- `fused_ldaK/logic/`: 1 solution(s) (1 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
