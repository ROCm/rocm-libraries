# OOB BBS TN exact tuning, category 1 (gfx1250, BF16 in/out, FP32 compute, TN)

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size with the full candidate search for that size
  (all tuning rounds merged into one file, identical solutions removed; the header comment lists the merged sets and
  the candidate count). Group 1 = new winner, group 2 = previous logic kernel (logic_1109a).
  Category 1 candidates contain no StreamK; where the previous logic kernel is StreamK it is included only as a
  comparison ruler in a separate second BenchmarkProblems section.
  Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 1000 / NumWarmups 300 / NumBenchmarks 5, RotatingBufferSize 384.
- `tuning_fused_ldaK/M<m>_N<n>_K<k>.yaml`: the full candidate search of the fused retune for that size, same layout as
  `tuning/`, as fused GEMM with unpadded leading dimensions: Bias (f32) + ScaleAlphaVec, activation none,
  ActivationFuncCall true (the problem type of the logic file), `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K,
  LDC = LDD = M), EnqueuesPerSync 1000 / NumWarmups 300 / NumBenchmarks 5, RotatingBufferSize 384.
  The logic file serves both variants: its ExactLogic keys are [M, N, 1, K] only (no strides).

## Results (Tensile client, GPU0, pure GEMM, LDA/LDB padded, median of rounds 1+)

Category mean: gain vs previous **+17.1%**, vs MAIDAS **45.3% -> 52.9%**
(vs MAIDAS = MAIDAS us / measured us).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | |
|---|---|---|---|---|---|---|---|---|---|
| 96x224x240 | 256 | 0 | 1.70 | 3.26 | 2.73 | +19.4% | 52.1% | 62.2% | |
| 160x224x976 | 1024 | 6 | 1.99 | 4.41 | 4.22 | +4.4% | 45.2% | 47.2% | |
| 224x464x608 | 640 | 9 | 2.06 | 4.70 | 3.85 | +22.2% | 43.8% | 53.6% | |
| 256x224x144 | 256 | 1 | 1.70 | 4.23 | 3.28 | +29.2% | 40.0% | 51.7% | |
| 576x208x736 | 768 | 8 | 2.05 | 4.47 | 4.29 | +4.1% | 45.9% | 47.8% | |
| 608x432x32 | 128 | 4 | 1.86 | 3.76 | 3.20 | +17.4% | 49.6% | 58.2% | |
| 656x112x32 | 128 | 5 | 1.91 | 3.02 | 2.56 | +17.7% | 63.4% | 74.6% | |
| 688x768x912 | 1024 | 3 | 2.47 | 6.42 | 5.71 | +12.4% | 38.5% | 43.2% | |
| 720x432x352 | 384 | 7 | 1.98 | 4.82 | 4.33 | +11.3% | 41.1% | 45.8% | |
| 832x896x16 | 128 | 2 | 1.75 | 5.19 | 3.91 | +32.6% | 33.7% | 44.7% | |

## Winner key parameters

| size | sol | kernel | change vs previous kernel |
|---|---|---|---|
| 96x224x240 | 0 | MT32x16x256, MIWT1_1, WG32_2_1, TPS None, CD4_4, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA512, HPLR0, LDSSI0, PLR1, CLS0, AG1 | TPS(TPSSK->N,WASG->None); MT(MT16x16x256->32x16x256,WG16_2_4->32_2_1); CD(CD2_2->4_4); PAP/XCC(PXCCM0->None); AG(AG0->1) |
| 160x224x976 | 6 | MT16x16x512, MIWT1_1, WG16_2_1, TPS None, CD4_2, SS0, VWA1, VWB1, SVW8, LPA8, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS0, AG1 | SS/VWA/SVW(SS1->0,SVW1->8); CLS(CLS1->0) |
| 224x464x608 | 9 | MT64x16x256, MIWT2_1, WG32_2_1, TPS None, CD4_2, SS0, VWA2, VWB1, SVW8, LPA8, LBSPPA1024, HPLR0, LDSSI0, PLR1, CLS0, AG1 | SS/VWA/SVW(VWA1->2); TPS(TPSSK->N,WASG->None); CLS(CLS1->0); MT(MT16x64x256->64x16x256,MIWT1_1->2_1,WG16_8_1->32_2_1); CD(CD1_1->4_2); PAP/XCC(PXCCM0->None) |
| 256x224x144 | 1 | MT32x16x128, MIWT2_1, WG16_2_1, TPS None, CD4_2, SS0, VWA2, VWB1, SVW8, LPA8, LBSPPA512, HPLR0, LDSSI0, PLR1, CLS0, AG0 | TPS(TPSSK->N,WASG->None); CLS(CLS1->0); MT(MT128x16x64->32x16x128,WG64_2_1->16_2_1); CD(CD1_2->4_2); PAP/XCC(PXCCM0->None); AG(AG1->0) |
| 576x208x736 | 8 | MT32x32x256, MIWT1_1, WG32_4_1, TPS None, CD4_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA512, HPLR0, LDSSI1, PLR3, CLS0, AG0 | PLR(PLR1->3) |
| 608x432x32 | 4 | MT32x16x32, MIWT1_1, WG32_2_1, TPS None, CD2_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA256, HPLR0, LDSSI0, PLR0, CLS0, AG0 | SS/VWA/SVW(SS0->1,SVW8->1); TPS(TPSSK->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 656x112x32 | 5 | MT192x16x32, MIWT3_1, WG64_2_1, TPS None, CD4_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA256, HPLR0, LDSSI0, PLR0, CLS0, AG1 | SS/VWA/SVW(SS0->1,VWA2->1,SVW8->1); MT(MT32x16x32->192x16x32,MIWT2_1->3_1,WG16_2_1->64_2_1); CD(CD2_1->4_2) |
| 688x768x912 | 3 | MT32x48x256, MIWT1_3, WG32_2_1, TPS None, CD1_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA512, HPLR0, LDSSI0, PLR1, CLS0, AG1 | TPS(TPSSK->N,WASG->None); MT(WG32_2_2->32_2_1); CD(CD1_1->1_2); PAP/XCC(PXCCM0->None); AG(AG0->1) |
| 720x432x352 | 7 | MT32x16x128, MIWT1_1, WG32_2_1, TPS None, CD4_2, SS1, VWA1, VWB1, SVW1, LPA8, LBSPPA256, HPLR0, LDSSI0, PLR2, CLS0, AG1 | PLR(PLR1->2,CLR0->1); TPS(TPSDP->N,WASG->None); CLS(CLS1->0); MT(MT16x16x256->32x16x128,WG16_2_1->32_2_1); CD(CD1_1->4_2); PAP/XCC(PXCCM0->None) |
| 832x896x16 | 2 | MT16x64x32, MIWT1_4, WG16_2_1, TPS None, CD2_2, SS1, VWA1, VWB4, SVW1, LPA8, LBSPPA256, HPLR0, LDSSI0, PLR0, CLS0, AG1 | SS/VWA/SVW(SS0->1,SVW8->1); TPS(TPSSK->N,WASG->None); CLS(CLS1->0); MT(MT16x128x32->16x64x32,WG16_4_1->16_2_1); PAP/XCC(PXCCM0->None) |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on this logic file: 0 rejects.
- The logic file uses the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall True on every solution), but the
  numbers above are pure GEMM (no bias / alpha vector / activation); fused performance is not measured.
- Measured with padded leading dimensions (LDA = LDB = ceil(K/128)*128); with LDA = K the numbers differ.
