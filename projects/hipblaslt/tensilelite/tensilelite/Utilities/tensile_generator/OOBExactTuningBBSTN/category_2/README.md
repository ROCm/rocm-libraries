# OOB BBS TN exact tuning, category 2 (gfx1250, BF16 in/out, FP32 compute, TN)

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size. Group 1 = new winner, group 2 = previous logic kernel
  (logic_1109a, for an A/B in the same client run). Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.

## Results (Tensile client, GPU0, pure GEMM, LDA/LDB padded, median of rounds 1+)

Category mean: gain vs previous **+9.8%**, vs MAIDAS **59.4% -> 65.3%**
(vs MAIDAS = MAIDAS us / measured us).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | |
|---|---|---|---|---|---|---|---|---|---|
| 464x1872x6736 | 6784 | 1 | 7.99 | 16.60 | 16.60 | +0.0% | 48.1% | 48.1% | same kernel as previous |
| 544x1728x7488 | 7552 | 4 | 8.71 | 17.80 | 17.54 | +1.5% | 48.9% | 49.6% | |
| 1744x5376x4096 | 4096 | 7 | 25.85 | 37.76 | 30.85 | +22.4% | 68.4% | 83.8% | |
| 2016x6112x4608 | 4608 | 8 | 36.67 | 54.58 | 44.13 | +23.7% | 67.2% | 83.1% | |
| 2960x1792x8176 | 8192 | 3 | 28.32 | 50.78 | 43.64 | +16.3% | 55.8% | 64.9% | |
| 3760x1168x2176 | 2176 | 9 | 8.33 | 11.77 | 11.61 | +1.4% | 70.7% | 71.7% | |
| 4176x4048x752 | 768 | 0 | 10.63 | 28.14 | 25.71 | +9.5% | 37.8% | 41.4% | |
| 6128x4784x3504 | 3584 | 5 | 67.70 | 86.91 | 83.61 | +4.0% | 77.9% | 81.0% | |
| 6592x2592x3296 | 3328 | 6 | 37.51 | 50.28 | 48.50 | +3.7% | 74.6% | 77.3% | |
| 7056x896x1248 | 1280 | 2 | 7.25 | 16.13 | 13.92 | +15.9% | 45.0% | 52.1% | |

## Winner key parameters

| size | sol | kernel | change vs previous kernel |
|---|---|---|---|
| 464x1872x6736 | 1 | MT64x64x512, MIWT4_2, WG16_4_2, TPS None, CD1_1, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS0, AG0 | - |
| 544x1728x7488 | 4 | MT64x64x512, MIWT4_2, WG16_4_2, TPS None, CD1_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS0, AG0 | CD(CD1_1->1_2) |
| 1744x5376x4096 | 7 | MT256x160x128, MIWT8_5, WG32_4_1, TPS None, CD1_2, SS1, VWA8, VWB1, SVW8, LPA8, LBSPPA2048, HPLR1, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR0->1); SS/VWA/SVW(VWA4->8,SVW4->8); MT(MT192x256x128->256x160x128,MIWT12_4->8_5,WG16_8_1->32_4_1); CD(CD1_1->1_2) |
| 2016x6112x4608 | 8 | MT256x192x128, MIWT16_3, WG16_8_1, TPS None, CD4_4, SS1, VWA8, VWB1, SVW8, LPA4, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS1, AG1 | SS/VWA/SVW(SS0->1,VWA1->8); LPA(LPA8->4); CLS(CLS0->1); MT(MT224x224x128->256x192x128,MIWT7_7->16_3,WG32_4_1->16_8_1); CD(CD1_1->4_4); AG(AG0->1) |
| 2960x1792x8176 | 3 | MT192x112x256, MIWT3_7, WG64_2_1, TPS None, CD1_1, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA512, HPLR0, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR1->0); SS/VWA/SVW(VWA4->1,SVW4->1); CLS(CLS0->1); MT(MT128x192x128->192x112x256,MIWT4_6->3_7,WG32_4_1->64_2_1) |
| 3760x1168x2176 | 9 | MT128x160x128, MIWT4_5, WG32_4_1, TPS None, CD4_4, SS1, VWA4, VWB1, SVW4, LPA8, LBSPPA1024, HPLR1, LDSSI1, PLR1, CLS1, AG1 | CD(CD4_2->4_4) |
| 4176x4048x752 | 0 | MT320x256x64, MIWT10_8, WG32_4_1, TPS None, CD2_1, SS1, VWA2, VWB8, SVW2, LPA8, LBSPPA256, HPLR2, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR0->2); SS/VWA/SVW(SS0->1,VWA1->2,SVW8->2); MT(MT288x256x128->320x256x64,MIWT9_8->10_8); CD(CD1_1->2_1) |
| 6128x4784x3504 | 5 | MT256x160x128, MIWT8_5, WG32_4_1, TPS None, CD1_1, SS1, VWA8, VWB1, SVW8, LPA8, LBSPPA2048, HPLR0, LDSSI1, PLR1, CLS1, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); SS/VWA/SVW(VWA4->8,SVW4->8) |
| 6592x2592x3296 | 6 | MT256x288x64, MIWT8_9, WG32_4_1, TPS None, CD2_1, SS1, VWA8, VWB1, SVW8, LPA8, LBSPPA1024, HPLR0, LDSSI1, PLR1, CLS1, AG1 | LSI(LDSSI0->1); CD(CD1_1->2_1) |
| 7056x896x1248 | 2 | MT128x112x128, MIWT2_7, WG64_2_1, TPS None, CD2_1, SS1, VWA2, VWB1, SVW2, LPA8, LBSPPA512, HPLR0, LDSSI0, PLR1, CLS0, AG1 | SS/VWA/SVW(VWA4->2,SVW4->2); CLS(CLS1->0); MT(MT256x144x128->128x112x128,MIWT4_9->2_7); CD(CD1_1->2_1) |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on this logic file: 0 rejects.
- The logic file uses the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall True on every solution), but the
  numbers above are pure GEMM (no bias / alpha vector / activation); fused performance is not measured.
- Measured with padded leading dimensions (LDA = LDB = ceil(K/128)*128); with LDA = K the numbers differ.
- 2 solution(s) (2 size(s)) use TDMIterateMode 1 (LdsBlockSizePerPadA 2048, DepthU 128, VWA 8), which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
