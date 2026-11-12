# OOB BBS TN exact tuning, category 4 (gfx1250, BF16 in/out, FP32 compute, TN)

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 9 solutions.
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size. Group 1 = new winner, group 2 = previous logic kernel
  (logic_1109a, for an A/B in the same client run). Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

## Results (Tensile client, GPU0, pure GEMM, LDA/LDB padded, median of rounds 1+)

Category mean: gain vs previous **+4.0%**, vs MAIDAS **81.2% -> 84.4%**
(vs MAIDAS = MAIDAS us / measured us).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | |
|---|---|---|---|---|---|---|---|---|---|
| 11600x7936x6048 | 6144 | 3 | 338.04 | 410.16 | 397.92 | +3.1% | 82.4% | 85.0% | |
| 168464x3200x6368 | 6400 | 8 | 2037.76 | 2318.78 | 2273.76 | +2.0% | 87.9% | 89.6% | |
| 315952x3696x1424 | 1536 | 2 | 1153.90 | 1500.41 | 1443.70 | +3.9% | 76.9% | 79.9% | |
| 411648x3248x5008 | 5120 | 5 | 4031.32 | 4591.15 | 4441.07 | +3.4% | 87.8% | 90.8% | |
| 540528x7600x1008 | 1024 | 0 | 2886.03 | 4123.49 | 3817.50 | +8.0% | 70.0% | 75.6% | |
| 563008x2192x8080 | 8192 | 4 | 6000.28 | 6797.51 | 6641.06 | +2.4% | 88.3% | 90.4% | |
| 573776x7936x6512 | 6528 | 6 | 17463.58 | 21282.30 | 19918.90 | +6.8% | 82.1% | 87.7% | |
| 721520x6032x1072 | 1152 | 1 | 3227.64 | 4524.37 | 4253.48 | +6.4% | 71.3% | 75.9% | |
| 814128x5376x2912 | 2944 | 2 | 7758.79 | 9434.62 | 9092.14 | +3.8% | 82.2% | 85.3% | |
| 838752x5696x3584 | 3584 | 7 | 10095.91 | 12077.20 | 12008.05 | +0.6% | 83.6% | 84.1% | |

## Winner key parameters

| size | sol | kernel | change vs previous kernel |
|---|---|---|---|
| 11600x7936x6048 | 3 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR1, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR0->1); SS/VWA/SVW(VWA4->8,SVW4->8) |
| 168464x3200x6368 | 8 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR1, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR2->1); TPS(TPSDP->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 315952x3696x1424 | 2 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR1, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | TDMIM(TDMIM0->1); HPLR(HPLR2->1); TPS(TPSDP->N,WASG->None); CLS(CLS0->1); MT(MT256x256x128->256x320x128,MIWT8_8->8_10); PAP/XCC(PXCCM0->None) |
| 411648x3248x5008 | 5 | MT256x256x128, MIWT8_8, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB4, SVW8, LPA8, LBSPPA1024, HPLR2, LDSSI1, PLR1, CLS0, AG0, LDSSBR1 | LSI(LDSSI0->1); BRot(LDSSBR0->1); TPS(TPSDP->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 540528x7600x1008 | 0 | MT256x320x128, MIWT8_10, WG32_4_1, TPS DataParallel, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR2, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR0->2); SS/VWA/SVW(VWA4->8,SVW4->8) |
| 563008x2192x8080 | 4 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR2, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR1->2); SS/VWA/SVW(VWA4->8,SVW4->8); TPS(TPSDP->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 573776x7936x6512 | 6 | MT256x320x128, MIWT8_10, WG32_4_1, TPS StreamK, CD1_1, SS1, VWA4, VWB2, SVW4, LPA8, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS1, AG1 | TPS(TPSDP->SK,WASG->DWQ) |
| 721520x6032x1072 | 1 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR2, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | TDMIM(TDMIM0->1); TPS(TPSDP->N,WASG->None); CLS(CLS0->1); MT(MT256x256x128->256x320x128,MIWT8_8->8_10); PAP/XCC(PXCCM0->None) |
| 814128x5376x2912 | 2 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA2048, HPLR1, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | TDMIM(TDMIM0->1); HPLR(HPLR2->1); TPS(TPSDP->N,WASG->None); CLS(CLS0->1); MT(MT256x256x128->256x320x128,MIWT8_8->8_10); PAP/XCC(PXCCM0->None) |
| 838752x5696x3584 | 7 | MT256x288x64, MIWT8_9, WG32_4_1, TPS None, CD4_4, SS1, VWA8, VWB1, SVW8, LPA8, LBSPPA1024, HPLR0, LDSSI1, PLR1, CLS1, AG1 | HPLR(HPLR2->0); TPS(TPSDP->N,WASG->None); CLS(CLS0->1); MT(MT256x256x128->256x288x64,MIWT8_8->8_9); CD(CD1_1->4_4); PAP/XCC(PXCCM0->None) |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on this logic file: 0 rejects.
- The logic file uses the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall True on every solution), but the
  numbers above are pure GEMM (no bias / alpha vector / activation); fused performance is not measured.
- Measured with padded leading dimensions (LDA = LDB = ceil(K/128)*128); with LDA = K the numbers differ.
- 6 solution(s) (7 size(s)) use TDMIterateMode 1 (LdsBlockSizePerPadA 2048, DepthU 128, VWA 8), which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
