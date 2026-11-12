# OOB BBS TN exact tuning, category 3 (gfx1250, BF16 in/out, FP32 compute, TN)

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, 10 solutions.
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile tuning config per size. Group 1 = new winner, group 2 = previous logic kernel
  (logic_1109a, for an A/B in the same client run). Pure GEMM, `Exact: [M, N, 1, K, M, M, LDA, LDB]` with LDA = LDB = ceil(K/128)*128,
  EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384.
- `tuning_fused_ldaK/M<m>_N<n>_K<k>.yaml`: the same kernels per size as `tuning/`, as fused GEMM with unpadded
  leading dimensions: Bias (f32) + ScaleAlphaVec, activation none, ActivationFuncCall true (the problem type of the
  logic file), `Exact: [M, N, 1, K, M, M, K, K]` (LDA = LDB = K, LDC = LDD = M), same benchmark settings.
  The logic file serves both variants: its ExactLogic keys are [M, N, 1, K] only (no strides).

## Results (Tensile client, GPU0, pure GEMM, LDA/LDB padded, median of rounds 1+)

Category mean: gain vs previous **+8.6%**, vs MAIDAS **82.3% -> 89.2%**
(vs MAIDAS = MAIDAS us / measured us).

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | previous us | new us | gain vs previous | previous vs MAIDAS | new vs MAIDAS | |
|---|---|---|---|---|---|---|---|---|---|
| 4464x5968x6464 | 6528 | 6 | 107.79 | 136.29 | 125.62 | +8.5% | 79.1% | 85.8% | |
| 4928x4848x7216 | 7296 | 0 | 105.99 | 135.41 | 123.84 | +9.3% | 78.3% | 85.6% | |
| 5504x6096x5440 | 5504 | 9 | 111.83 | 129.94 | 124.93 | +4.0% | 86.1% | 89.5% | |
| 6000x4928x7216 | 7296 | 3 | 133.61 | 161.74 | 151.65 | +6.7% | 82.6% | 88.1% | |
| 6496x7056x5664 | 5760 | 2 | 160.19 | 208.76 | 189.90 | +9.9% | 76.7% | 84.4% | |
| 7008x5808x6288 | 6400 | 4 | 156.05 | 180.61 | 175.87 | +2.7% | 86.4% | 88.7% | |
| 7040x6912x6256 | 6272 | 1 | 184.96 | 246.22 | 199.41 | +23.5% | 75.1% | 92.8% | |
| 7200x6304x5888 | 5888 | 8 | 161.62 | 196.27 | 189.44 | +3.6% | 82.3% | 85.3% | |
| 7808x7088x5424 | 5504 | 5 | 186.47 | 209.03 | 201.93 | +3.5% | 89.2% | 92.3% | |
| 7856x5120x7200 | 7296 | 7 | 175.45 | 202.19 | 176.83 | +14.3% | 86.8% | 99.2% | |

## Winner key parameters

| size | sol | kernel | change vs previous kernel |
|---|---|---|---|
| 4464x5968x6464 | 6 | MT224x256x128, MIWT14_4, WG16_8_1, TPS None, CD2_2, SS1, VWA2, VWB4, SVW2, LPA8, LBSPPA512, HPLR2, LDSSI1, PLR1, CLS1, AG1 | LSI(LDSSI0->1); HPLR(HPLR0->2); SS/VWA/SVW(SS0->1,SVW8->2); CD(CD1_1->2_2) |
| 4928x4848x7216 | 0 | MT192x256x128, MIWT12_4, WG16_8_1, TPS None, CD1_1, SS1, VWA4, VWB4, SVW4, LPA8, LBSPPA1024, HPLR0, LDSSI1, PLR1, CLS1, AG1 | LSI(LDSSI0->1); SS/VWA/SVW(SS0->1,VWA2->4,SVW8->4); MT(MIWT6_8->12_4,WG32_4_1->16_8_1) |
| 5504x6096x5440 | 9 | MT192x384x128, MIWT12_6, WG16_8_1, TPS None, CD4_4, SS1, VWA4, VWB2, SVW4, LPA8, LBSPPA1024, HPLR1, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR0->1); CD(CD4_2->4_4) |
| 6000x4928x7216 | 3 | MT192x320x128, MIWT12_5, WG16_8_1, TPS None, CD1_1, SS1, VWA4, VWB1, SVW4, LPA8, LBSPPA1024, HPLR1, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR0->1); SS/VWA/SVW(SS0->1,SVW8->4) |
| 6496x7056x5664 | 2 | MT384x160x128, MIWT6_10, WG64_2_1, TPS None, CD1_1, SS1, VWA2, VWB2, SVW2, LPA8, LBSPPA512, HPLR2, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR0->2); SS/VWA/SVW(SS0->1,SVW8->2); PLR(CLR1->0); TPS(TPSDP->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 7008x5808x6288 | 4 | MT320x256x128, MIWT10_8, WG32_4_1, TPS None, CD1_1, SS1, VWA2, VWB4, SVW2, LPA8, LBSPPA512, HPLR2, LDSSI0, PLR1, CLS1, AG1 | HPLR(HPLR0->2); TPS(TPSDP->N,WASG->None); PAP/XCC(PXCCM0->None) |
| 7040x6912x6256 | 1 | MT256x256x128, MIWT8_8, WG32_4_1, TPS None, CD2_1, SS1, VWA8, VWB4, SVW8, LPA8, LBSPPA2048, HPLR2, LDSSI1, PLR1, CLS0, AG1, TDMIM1 (AF0EM 8) | LSI(LDSSI0->1); TDMIM(TDMIM0->1); HPLR(HPLR1->2); SS/VWA/SVW(SS0->1,VWA2->8); TPS(TPSSK->N,WASG->None); MT(MT320x224x128->256x256x128,MIWT10_7->8_8); CD(CD1_1->2_1); PAP/XCC(PXCCM0->None); AG(AG0->1) |
| 7200x6304x5888 | 8 | MT160x384x128, MIWT10_6, WG16_8_1, TPS None, CD1_1, SS1, VWA2, VWB2, SVW2, LPA8, LBSPPA512, HPLR1, LDSSI0, PLR1, CLS1, AG1 | LSI(LDSSI1->0); HPLR(HPLR2->1); SS/VWA/SVW(SS0->1,VWA1->2,SVW8->2); CLS(CLS0->1); MT(MT240x256x128->160x384x128,MIWT15_4->10_6); AG(AG0->1) |
| 7808x7088x5424 | 5 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD2_1, SS1, VWA8, VWB2, SVW8, LPA8, LBSPPA1024, HPLR1, LDSSI1, PLR1, CLS0, AG1 | TPS(TPSDP->N,WASG->None); CLS(CLS1->0); CD(CD1_1->2_1); PAP/XCC(PXCCM0->None) |
| 7856x5120x7200 | 7 | MT256x320x128, MIWT8_10, WG32_4_1, TPS None, CD4_4, SS1, VWA8, VWB2, SVW8, LPA4, LBSPPA1024, HPLR1, LDSSI1, PLR1, CLS1, AG1 | LSI(LDSSI0->1); SS/VWA/SVW(VWA4->8,SVW4->8); LPA(LPA8->4); TPS(TPSDP->N,WASG->None); MT(MT384x224x128->256x320x128,MIWT12_7->8_10); CD(CD1_1->4_4); PAP/XCC(PXCCM0->None) |

## Requirements / notes

- gfx1250 (`Device 73f0`), Tensile with TDM (TDMInst 3), TDMIterateMode, LDSSegmentInterleave, HalfPLR, ClusterDim and
  AdaptiveGemm support. TensileCreateLibrary --architecture=gfx1250 on this logic file: 0 rejects.
- The logic file uses the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall True on every solution), but the
  numbers above are pure GEMM (no bias / alpha vector / activation); fused performance is not measured.
- Measured with padded leading dimensions (LDA = LDB = ceil(K/128)*128); with LDA = K the numbers differ.
- 1 solution(s) (1 size(s)) use TDMIterateMode 1 (LdsBlockSizePerPadA 2048, DepthU 128, VWA 8), which sets
  AssertFree0ElementMultiple 8: they are only selected when M % 8 == 0 (true for these exact sizes).
