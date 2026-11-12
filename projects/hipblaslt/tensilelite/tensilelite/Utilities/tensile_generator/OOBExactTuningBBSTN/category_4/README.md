# OOB BBS TN exact tuning, category 4 (gfx1250, BF16 in/out, FP32 compute, TN)

Two variants, each in its own directory with a logic file and tuning configs. Both logic files hold only the 10 sizes of this
category, use the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true on every solution), and key
ExactLogic on [M, N, 1, K] only (no strides).

### Pure GEMM, LDA/LDB padded (`pure_LdaPadded/`)

- `pure_LdaPadded/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 9 solutions.
  Winners tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128.
- `pure_LdaPadded/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search for that size
  (all tuning rounds merged into one file, including a ClusterDim / ClusterAIDMapping / WorkGroupMapping sweep in which
  cluster candidates use ClusterAIDMapping -2; identical solutions removed; the header comment lists the merged sets
  and the candidate count). Group 1 = winner, group 2 = previous logic kernel (logic_1109a).
  Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

- `fused_ldaK/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 9 solutions.
  Winners tuned as fused GEMM (Bias (f32) + ScaleAlphaVec, activation none) with unpadded leading dimensions.
- `fused_ldaK/tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search of the
  fused retune for that size (all candidate sets merged into one file, including a ClusterDim sweep; identical
  solutions removed; the header comment gives the candidate count). Group 1 = winner of the pure padded logic
  (previous winner), group 2 = previous logic kernel (logic_1109a).
  Fused GEMM, Bias (f32) + ScaleAlphaVec, activation none, ActivationFuncCall true,
  `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K, LDC = LDD = M),
  EnqueuesPerSync 200 / NumWarmups 80 / NumBenchmarks 3, RotatingBufferSize 384.

## Results, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`; Tensile client, GPU3, median of rounds 1+)

Category mean: gain vs previous **+1.6%**, vs MAIDAS **78.6% -> 79.8%**
(logic_1109a: 74.7%; 6 of 10 sizes have a new winner; vs MAIDAS = MAIDAS us / measured us).
Previous = winner of the previous pure padded logic of this directory; logic_1109a = previous logic kernel. All three
kernels of a size are measured in the same client run; if the previous winner measured faster it is kept.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | logic_1109a us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 11600x7936x6048 | 6144 | 3 | 338.04 | 439.93 | 422.39 | 417.37 | +1.2% | 80.0% | 81.0% | new |
| 168464x3200x6368 | 6400 | 3 | 2037.76 | 2532.19 | 2459.32 | 2459.32 | +0.0% | 82.9% | 82.9% | same as previous |
| 315952x3696x1424 | 1536 | 2 | 1153.90 | 1631.13 | 1557.18 | 1518.32 | +2.6% | 74.1% | 76.0% | new |
| 411648x3248x5008 | 5120 | 5 | 4031.32 | 5032.19 | 4790.60 | 4790.60 | +0.0% | 84.2% | 84.2% | same as previous |
| 540528x7600x1008 | 1024 | 0 | 2886.03 | 4395.07 | 4090.05 | 3968.57 | +3.1% | 70.6% | 72.7% | new |
| 563008x2192x8080 | 8192 | 4 | 6000.28 | 7388.88 | 7141.75 | 7141.75 | +0.0% | 84.0% | 84.0% | same as previous |
| 573776x7936x6512 | 6528 | 7 | 17463.58 | 23132.10 | 21484.55 | 20794.85 | +3.3% | 81.3% | 84.0% | new |
| 721520x6032x1072 | 1152 | 1 | 3227.64 | 4878.06 | 4562.38 | 4490.00 | +1.6% | 70.7% | 71.9% | new |
| 814128x5376x2912 | 2944 | 6 | 7758.79 | 10297.35 | 9857.11 | 9455.81 | +4.2% | 78.7% | 82.1% | new |
| 838752x5696x3584 | 3584 | 8 | 10095.91 | 13496.80 | 12680.45 | 12680.45 | +0.0% | 79.6% | 79.6% | previous kept |

## Results, fused GEMM, LDA = LDB = K (`fused_ldaK/`; Tensile client, GPU0 and GPU3 (column), median of rounds 1+)

Category mean: gain vs previous **+2.4%**, vs MAIDAS **64.7% -> 66.2%**
(8 of 10 sizes have a new winner; the others keep the previous winner).
Fused GEMM: Bias (f32) + ScaleAlphaVec, activation none, LDA = LDB = K, LDC = LDD = M. Previous = winner of the
earlier pure padded logic, measured in the same client run as the new winner; a size keeps the previous winner
when it measured faster. Rows marked GPU0 come from one bench of all sizes on GPU0; rows marked GPU3 are the sizes
whose winner was re-selected afterwards, re-measured on GPU3 (compare gains, not absolute us, across GPUs).

| size (MxNxK) | sol | GPU | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|
| 11600x7936x6048 | 2 | GPU0 | 338.04 | 548.91 | 548.91 | +0.0% | 61.6% | 61.6% | kept |
| 168464x3200x6368 | 8 | GPU3 | 2037.76 | 2765.89 | 2761.78 | +0.1% | 73.7% | 73.8% | new |
| 315952x3696x1424 | 1 | GPU0 | 1153.90 | 1891.11 | 1803.51 | +4.9% | 61.0% | 64.0% | new |
| 411648x3248x5008 | 4 | GPU0 | 4031.32 | 6280.07 | 5959.90 | +5.4% | 64.2% | 67.6% | new |
| 540528x7600x1008 | 0 | GPU0 | 2886.03 | 5168.10 | 4987.73 | +3.6% | 55.8% | 57.9% | new |
| 563008x2192x8080 | 3 | GPU0 | 6000.28 | 9536.69 | 9536.69 | +0.0% | 62.9% | 62.9% | kept |
| 573776x7936x6512 | 6 | GPU0 | 17463.58 | 26489.40 | 25949.00 | +2.1% | 65.9% | 67.3% | new |
| 721520x6032x1072 | 1 | GPU0 | 3227.64 | 5543.49 | 5343.64 | +3.7% | 58.2% | 60.4% | new |
| 814128x5376x2912 | 5 | GPU0 | 7758.79 | 11624.10 | 11434.50 | +1.7% | 66.7% | 67.9% | new |
| 838752x5696x3584 | 7 | GPU3 | 10095.91 | 13119.10 | 12772.85 | +2.7% | 77.0% | 79.0% | new |

## Winner key parameters, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 11600x7936x6048 | 3 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 168464x3200x6368 | 3 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | same as previous |
| 315952x3696x1424 | 2 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 411648x3248x5008 | 5 | MT256x256x128, MIWT8_8, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR2, LDSSI1, LDSSBR1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG0, PGL0, PAP0, WGM16 | same as previous |
| 540528x7600x1008 | 0 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS DataParallel, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM32 | new |
| 563008x2192x8080 | 4 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL1, PAP0, WGM8 | same as previous |
| 573776x7936x6512 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA4, VWB2, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 721520x6032x1072 | 1 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM32 | new |
| 814128x5376x2912 | 6 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM32 | new |
| 838752x5696x3584 | 8 | MT256x288x64, MIWT8_9, WG32_4_1, CD4_4, TPS None, SS1, VWA8, VWB1, SVW8, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |

## Winner key parameters, fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 11600x7936x6048 | 2 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | kept |
| 168464x3200x6368 | 8 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 315952x3696x1424 | 1 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM32 | new |
| 411648x3248x5008 | 4 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR0, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 540528x7600x1008 | 0 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS DataParallel, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP1, WGM32 | new |
| 563008x2192x8080 | 3 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL1, PAP0, WGM8 | kept |
| 573776x7936x6512 | 6 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR0, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 721520x6032x1072 | 1 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM32 | new |
| 814128x5376x2912 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB2, SVW8, HPLR0, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 838752x5696x3584 | 7 | MT256x256x128, MIWT8_8, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR2, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL0, PAP0, WGM32 | new |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on each logic file: 0 rejects.
- The two tables differ in LDA alignment (with LDA = K, K*2 bytes not a multiple of 128 makes TDM loads go
  indirect) and in the fused epilogue, so the two variants can pick different kernels for the same size.
- The tables come from different GPUs and runs (pure: GPU3; fused: GPU0, plus GPU3 rows where marked); compare gains
  within a row, not absolute us across tables.
- `pure_LdaPadded/logic/`: 6 solution(s) (7 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
- `fused_ldaK/logic/`: 8 solution(s) (9 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
