# OOB BBS TN exact tuning, category 3 (gfx1250, BF16 in/out, FP32 compute, TN)

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
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 3, RotatingBufferSize 384.

## Results, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`; Tensile client, GPU3, median of rounds 1+)

Category mean: gain vs previous **+1.5%**, vs MAIDAS **83.4% -> 84.7%**
(logic_1109a: 76.8%; 7 of 10 sizes have a new winner; vs MAIDAS = MAIDAS us / measured us).
Previous = winner of the previous pure padded logic of this directory; logic_1109a = previous logic kernel. All three
kernels of a size are measured in the same client run; if the previous winner measured faster it is kept.

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | logic_1109a us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 4464x5968x6464 | 6528 | 6 | 107.79 | 144.04 | 131.79 | 131.79 | +0.0% | 81.8% | 81.8% | previous kept |
| 4928x4848x7216 | 7296 | 0 | 105.99 | 144.75 | 134.28 | 133.13 | +0.9% | 78.9% | 79.6% | new |
| 5504x6096x5440 | 5504 | 9 | 111.83 | 134.35 | 129.71 | 129.62 | +0.1% | 86.2% | 86.3% | new |
| 6000x4928x7216 | 7296 | 3 | 133.61 | 173.76 | 162.93 | 150.60 | +8.2% | 82.0% | 88.7% | new |
| 6496x7056x5664 | 5760 | 2 | 160.19 | 225.71 | 205.01 | 203.07 | +1.0% | 78.1% | 78.9% | new |
| 7008x5808x6288 | 6400 | 4 | 156.05 | 197.10 | 189.06 | 189.06 | +0.0% | 82.5% | 82.5% | previous kept |
| 7040x6912x6256 | 6272 | 1 | 184.96 | 260.22 | 215.00 | 214.93 | +0.0% | 86.0% | 86.1% | new |
| 7200x6304x5888 | 5888 | 8 | 161.62 | 210.84 | 203.58 | 203.58 | +0.0% | 79.4% | 79.4% | previous kept |
| 7808x7088x5424 | 5504 | 5 | 186.47 | 227.91 | 214.87 | 205.98 | +4.3% | 86.8% | 90.5% | new |
| 7856x5120x7200 | 7296 | 7 | 175.45 | 218.48 | 189.69 | 189.09 | +0.3% | 92.5% | 92.8% | new |

## Results, fused GEMM, LDA = LDB = K (`fused_ldaK/`; Tensile client, GPU0 and GPU3 (column), median of rounds 1+)

Category mean: gain vs previous **+3.4%**, vs MAIDAS **66.0% -> 68.3%**
(9 of 10 sizes have a new winner; the others keep the previous winner).
Fused GEMM: Bias (f32) + ScaleAlphaVec, activation none, LDA = LDB = K, LDC = LDD = M. Previous = winner of the
earlier pure padded logic, measured in the same client run as the new winner; a size keeps the previous winner
when it measured faster. Rows marked GPU0 come from one bench of all sizes on GPU0; rows marked GPU3 are the sizes
whose winner was re-selected afterwards, re-measured on GPU3 (compare gains, not absolute us, across GPUs).

| size (MxNxK) | sol | GPU | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | winner |
|---|---|---|---|---|---|---|---|---|---|
| 4464x5968x6464 | 6 | GPU0 | 107.79 | 155.92 | 146.95 | +6.1% | 69.1% | 73.4% | new |
| 4928x4848x7216 | 0 | GPU0 | 105.99 | 186.69 | 186.69 | +0.0% | 56.8% | 56.8% | kept |
| 5504x6096x5440 | 9 | GPU3 | 111.83 | 137.25 | 137.24 | +0.0% | 81.5% | 81.5% | new |
| 6000x4928x7216 | 3 | GPU3 | 133.61 | 216.40 | 215.03 | +0.6% | 61.7% | 62.1% | new |
| 6496x7056x5664 | 2 | GPU0 | 160.19 | 282.13 | 275.23 | +2.5% | 56.8% | 58.2% | new |
| 7008x5808x6288 | 4 | GPU0 | 156.05 | 241.94 | 231.95 | +4.3% | 64.5% | 67.3% | new |
| 7040x6912x6256 | 1 | GPU0 | 184.96 | 308.09 | 300.02 | +2.7% | 60.0% | 61.6% | new |
| 7200x6304x5888 | 8 | GPU0 | 161.62 | 232.90 | 216.01 | +7.8% | 69.4% | 74.8% | new |
| 7808x7088x5424 | 5 | GPU0 | 186.47 | 278.18 | 263.70 | +5.5% | 67.0% | 70.7% | new |
| 7856x5120x7200 | 7 | GPU0 | 175.45 | 239.26 | 228.15 | +4.9% | 73.3% | 76.9% | new |

## Winner key parameters, pure GEMM, LDA/LDB padded (`pure_LdaPadded/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 4464x5968x6464 | 6 | MT224x256x128, MIWT14_4, WG16_8_1, CD2_2, CAIDMn2, TPS None, SS1, VWA2, VWB4, SVW2, HPLR2, LDSSI1, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |
| 4928x4848x7216 | 0 | MT192x256x128, MIWT12_4, WG16_8_1, CD1_1, TPS None, SS1, VWA4, VWB4, SVW4, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 5504x6096x5440 | 9 | MT192x384x128, MIWT12_6, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB2, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 6000x4928x7216 | 3 | MT192x320x128, MIWT12_5, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB1, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6496x7056x5664 | 2 | MT384x160x128, MIWT6_10, WG64_2_1, CD1_1, TPS None, SS1, VWA2, VWB2, SVW2, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM32 | new |
| 7008x5808x6288 | 4 | MT320x256x128, MIWT10_8, WG32_4_1, CD1_1, TPS None, SS1, VWA2, VWB4, SVW2, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | previous kept |
| 7040x6912x6256 | 1 | MT256x256x128, MIWT8_8, WG32_4_1, CD2_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS0, AG1, PGL2, PAP0, WGM8 | new |
| 7200x6304x5888 | 8 | MT160x384x128, MIWT10_6, WG16_8_1, CD1_1, TPS None, SS1, VWA2, VWB2, SVW2, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | previous kept |
| 7808x7088x5424 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR1, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS0, AG1, PGL2, PAP0, WGM8 | new |
| 7856x5120x7200 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, CD4_4, CAIDMn2, TPS None, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |

## Winner key parameters, fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size | sol | kernel | winner |
|---|---|---|---|
| 4464x5968x6464 | 6 | MT224x256x128, MIWT14_4, WG16_8_1, CD2_2, CAIDMn2, TPS None, SS0, VWA2, VWB4, SVW8, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS0, AG1, PGL0, PAP0, WGM8 | new |
| 4928x4848x7216 | 0 | MT192x256x128, MIWT12_4, WG16_8_1, CD1_1, TPS None, SS1, VWA4, VWB4, SVW4, HPLR0, LDSSI1, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | kept |
| 5504x6096x5440 | 9 | MT192x384x128, MIWT12_6, WG16_8_1, CD4_4, TPS None, SS1, VWA4, VWB2, SVW4, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 6000x4928x7216 | 3 | MT192x320x128, MIWT12_5, WG16_8_1, CD1_1, TPS DataParallel, SS1, VWA4, VWB1, SVW4, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA1024, PLR1, CLS1, AG1, PGL0, PAP0, WGM8 | new |
| 6496x7056x5664 | 2 | MT384x160x128, MIWT6_10, WG64_2_1, CD1_1, TPS None, SS0, VWA2, VWB2, SVW8, HPLR0, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL0, PAP0, WGM16 | new |
| 7008x5808x6288 | 4 | MT320x256x128, MIWT10_8, WG32_4_1, CD2_1, CAIDMn2, TPS DataParallel, SS0, VWA2, VWB4, SVW8, HPLR1, LDSSI0, TDMIM0, LPA8, LBSPPA512, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 7040x6912x6256 | 1 | MT256x256x128, MIWT8_8, WG32_4_1, CD1_1, TPS None, SS1, VWA8, VWB4, SVW8, HPLR1, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |
| 7200x6304x5888 | 8 | MT240x256x128, MIWT15_4, WG16_8_1, CD2_1, TPS None, SS0, VWA1, VWB4, SVW8, HPLR2, LDSSI0, TDMIM0, LPA8, LBSPPA256, PLR1, CLS0, AG0, PGL0, PAP0, WGM8 | new |
| 7808x7088x5424 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, CD2_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI0, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM16 | new |
| 7856x5120x7200 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, CD2_1, TPS StreamK, SS1, VWA8, VWB2, SVW8, HPLR2, LDSSI1, TDMIM1, LPA8, LBSPPA2048, PLR1, CLS1, AG1, PGL2, PAP0, WGM8 | new |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on each logic file: 0 rejects.
- The two tables differ in LDA alignment (with LDA = K, K*2 bytes not a multiple of 128 makes TDM loads go
  indirect) and in the fused epilogue, so the two variants can pick different kernels for the same size.
- The tables come from different GPUs and runs (pure: GPU3; fused: GPU0, plus GPU3 rows where marked); compare gains
  within a row, not absolute us across tables.
- `pure_LdaPadded/logic/`: 2 solution(s) (2 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
- `fused_ldaK/logic/`: 3 solution(s) (3 size(s)) use TDMIterateMode 1, which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
