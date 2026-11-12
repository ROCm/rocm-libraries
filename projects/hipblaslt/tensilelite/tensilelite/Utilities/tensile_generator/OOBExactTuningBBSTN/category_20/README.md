# OOB BBS TN exact tuning, category 20: Very large batch (gfx1250, BF16 in/out, FP32 compute, TN)

Two variants are provided for this category: `fused` and `pure_lda`.

## fused: fused GEMM, no 256 B alignment

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with LDA = LDB = K and LDC = LDD = M, i.e. the leading dimensions are not padded to 256 B.

- `fused/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size (fused GEMM: Bias (f32) + ScaleAlphaVec, ActivationFuncCall, `Exact: [M, N, batch, K]`, LDA = LDB = K, LDC = LDD = M, RotatingBufferSize 384).
- `fused/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below, 10 solutions.

### Results (hipblaslt-bench, GPU0, fused GEMM, LDA = LDB = K, tuned winner vs OOB)

Category mean: speedup vs OOB **+127.6%**, vs MAIDAS **44.6% -> 80.8%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P31 | 1616x224x8398x128 | 1021.03 | 5056.00 | 1927.00 | +162.4% | 20.2% | 53.0% |
| P32 | 2768x32x8803x208 | 1066.43 | 2076.50 | 1249.00 | +66.3% | 51.4% | 85.4% |
| P33 | 176x144x9760x1120 | 732.88 | 1550.50 | 618.50 | +150.7% | 47.3% | 118.5% |
| P34 | 96x816x13015x576 | 1486.27 | 2571.00 | 1346.00 | +91.0% | 57.8% | 110.4% |
| P35 | 80x96x15029x2736 | 1290.01 | 1356.67 | 1332.33 | +1.8% | 95.1% | 96.8% |
| P36 | 384x384x22154x240 | 1374.01 | 4635.00 | 3113.00 | +48.9% | 29.6% | 44.1% |
| P37 | 80x3616x23009x16 | 1452.15 | 5564.00 | 4122.00 | +35.0% | 26.1% | 35.2% |
| P38 | 528x32x28366x384 | 1268.70 | 2031.50 | 1100.00 | +84.7% | 62.5% | 115.3% |
| P39 | 4048x16x51732x16 | 2154.55 | 15599.00 | 2184.00 | +614.2% | 13.8% | 98.7% |
| P40 | 224x128x98666x96 | 1170.81 | 2795.00 | 2300.00 | +21.5% | 41.9% | 50.9% |

### Winner key parameters

| # | size | winner |
|---|---|---|
| P31 | 1616x224x8398x128 | MT128x224x32, AG1, CD1_1, CLS0, MIWT4_7, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_4_1 |
| P32 | 2768x32x8803x208 | MT192x32x64, AG0, CD1_1, CLS0, MIWT12_1, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG16_4_1 |
| P33 | 176x144x9760x1120 | MT176x160x64, AG0, CD1_1, CLS0, MIWT11_5, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG16_4_1 |
| P34 | 96x816x13015x576 | MT96x144x64, AG0, CD1_1, CLS0, MIWT3_9, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG32_2_1 |
| P35 | 80x96x15029x2736 | MT80x96x128, AG0, CD1_1, CLS0, MIWT5_3, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG16_4_1 |
| P36 | 384x384x22154x240 | MT192x192x32, AG0, CD1_1, CLS0, MIWT6_6, PAP0, PLR1, RAP0, SS1, SVW2, VWA2, VWB2, WG32_4_1 |
| P37 | 80x3616x23009x16 | MT80x160x32, AG0, CD1_1, CLS0, MIWT5_5, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG16_4_1 |
| P38 | 528x32x28366x384 | MT192x32x64, AG1, CD1_1, CLS0, MIWT12_1, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG16_4_1 |
| P39 | 4048x16x51732x16 | MT128x16x32, AG0, CD1_1, CLS0, MIWT4_1, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_2_1 |
| P40 | 224x128x98666x96 | MT112x128x64, AG0, CD1_1, CLS0, MIWT7_4, PAP0, PLR1, RAP0, SS1, SVW1, VWA1, VWB4, WG16_4_1 |

## pure_lda: pure GEMM, 256 B aligned leading dimensions

Pure GEMM (no Bias / ScaleAlphaVec / activation) with the leading dimensions padded for 256 B alignment (BF16, 2 B per element). `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`, RotatingBufferSize 384.

- `pure_lda/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]` with the leading dimensions of the original problem, RotatingBufferSize 384.
- `pure_lda/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`: merged ExactLogic for the 10 sizes below.

### Results (hipblaslt-bench, GPU0, pure GEMM, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+157.3%**, vs MAIDAS **44.6% -> 89.3%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P31 | 1616x224x8398x128 | 1021.03 | 5070.00 | 1355.00 | +274.2% | 20.1% | 75.4% |
| P32 | 2768x32x8803x208 | 1066.43 | 2081.00 | 1079.50 | +92.8% | 51.2% | 98.8% |
| P33 | 176x144x9760x1120 | 732.88 | 1539.50 | 579.00 | +165.9% | 47.6% | 126.6% |
| P34 | 96x816x13015x576 | 1486.27 | 2572.00 | 1274.00 | +101.9% | 57.8% | 116.7% |
| P35 | 80x96x15029x2736 | 1290.01 | 1349.67 | 1351.33 | -0.1% | 95.6% | 95.5% |
| P36 | 384x384x22154x240 | 1374.01 | 4607.00 | 3115.00 | +47.9% | 29.8% | 44.1% |
| P37 | 80x3616x23009x16 | 1452.15 | 5560.00 | 4296.00 | +29.4% | 26.1% | 33.8% |
| P38 | 528x32x28366x384 | 1268.70 | 2037.00 | 956.00 | +113.1% | 62.3% | 132.7% |
| P39 | 4048x16x51732x16 | 2154.55 | 15650.00 | 1935.00 | +708.8% | 13.8% | 111.3% |
| P40 | 224x128x98666x96 | 1170.81 | 2807.00 | 2010.50 | +39.6% | 41.7% | 58.2% |

### Problem sizes and leading dimensions

| # | size (MxNxBatchxK) | LDA | LDB | LDC | LDD |
|---|---|---|---|---|---|
| P31 | 1616x224x8398x128 | 128 | 128 | 1664 | 1664 |
| P32 | 2768x32x8803x208 | 256 | 256 | 2816 | 2816 |
| P33 | 176x144x9760x1120 | 1152 | 1152 | 176* | 176* |
| P34 | 96x816x13015x576 | 576* | 576* | 96* | 96* |
| P35 | 80x96x15029x2736 | 2816 | 2816 | 80* | 80* |
| P36 | 384x384x22154x240 | 256 | 256 | 384 | 384 |
| P37 | 80x3616x23009x16 | 16* | 16* | 80* | 80* |
| P38 | 528x32x28366x384 | 384 | 384 | 640 | 640 |
| P39 | 4048x16x51732x16 | 16* | 16* | 4096 | 4096 |
| P40 | 224x128x98666x96 | 128 | 128 | 224* | 224* |

`*`: not a multiple of 256 B.

## Notes

- The `fused` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, fused GEMM (Bias + ScaleAlphaVec), LDA = LDB = K.
- The `pure_lda` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, pure GEMM, 256 B aligned leading dimensions.
