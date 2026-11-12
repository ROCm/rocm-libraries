# OOB BBS TN exact tuning, category 19: Large batch (gfx1250, BF16 in/out, FP32 compute, TN)

Two variants are provided for this category: `fused` and `pure_lda`.

## fused: fused GEMM, no 256 B alignment

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with LDA = LDB = K and LDC = LDD = M, i.e. the leading dimensions are not padded to 256 B.

- `fused/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size (fused GEMM: Bias (f32) + ScaleAlphaVec, ActivationFuncCall, `Exact: [M, N, batch, K]`, LDA = LDB = K, LDC = LDD = M, RotatingBufferSize 384).
- `fused/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below, 10 solutions.

### Results (hipblaslt-bench, GPU0, fused GEMM, LDA = LDB = K, tuned winner vs OOB)

Category mean: speedup vs OOB **+116.4%**, vs MAIDAS **35.0% -> 62.4%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P21 | 336x144x2108x7776 | 1388.88 | 2268.50 | 1465.50 | +54.8% | 61.2% | 94.8% |
| P22 | 656x736x2147x640 | 650.33 | 4984.00 | 1277.00 | +290.3% | 13.0% | 50.9% |
| P23 | 208x2256x2367x192 | 450.34 | 1568.50 | 1214.50 | +29.1% | 28.7% | 37.1% |
| P24 | 1488x1568x2773x16 | 929.52 | 10780.00 | 2414.00 | +346.6% | 8.6% | 38.5% |
| P25 | 624x1472x2930x928 | 2644.58 | 5448.00 | 5278.00 | +3.2% | 48.5% | 50.1% |
| P26 | 48x720x3028x208 | 116.22 | 210.23 | 147.77 | +42.3% | 55.3% | 78.7% |
| P27 | 1776x16x4186x784 | 978.63 | 2013.00 | 816.00 | +146.7% | 48.6% | 119.9% |
| P28 | 3120x288x6158x64 | 1187.00 | 5382.00 | 2249.00 | +139.3% | 22.1% | 52.8% |
| P29 | 3840x48x6527x208 | 940.19 | 2281.00 | 1404.50 | +62.4% | 41.2% | 66.9% |
| P30 | 160x5488x7242x32 | 1354.00 | 5867.00 | 3920.00 | +49.7% | 23.1% | 34.5% |

### Winner key parameters

| # | size | winner |
|---|---|---|
| P21 | 336x144x2108x7776 | MT192x160x128, AG1, CD2_1, CLS0, MIWT6_5, PAP0, PLR1, RAP0, SS0, SVW8, VWA2, VWB1, WG32_4_1 |
| P22 | 656x736x2147x640 | MT128x192x64, AG1, CD1_1, CLS0, MIWT8_6, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB2, WG16_4_1 |
| P23 | 208x2256x2367x192 | MT208x192x64, AG0, CD1_1, CLS0, MIWT13_3, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG16_8_1 |
| P24 | 1488x1568x2773x16 | MT256x128x32, AG1, CD1_1, CLS0, MIWT8_4, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB4, WG32_4_1 |
| P25 | 624x1472x2930x928 | MT208x256x128, AG0, CD1_2, CLS0, MIWT13_4, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB4, WG16_8_1 |
| P26 | 48x720x3028x208 | MT48x160x64, AG0, CD1_1, CLS0, MIWT3_5, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG16_4_1 |
| P27 | 1776x16x4186x784 | MT128x16x128, AG0, CD1_1, CLS0, MIWT4_1, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_2_1 |
| P28 | 3120x288x6158x64 | MT256x144x32, AG1, CD1_1, CLS0, MIWT4_9, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG64_2_1 |
| P29 | 3840x48x6527x208 | MT128x48x64, AG0, CD1_1, CLS0, MIWT4_3, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG32_2_1 |
| P30 | 160x5488x7242x32 | MT160x160x32, AG0, CD1_1, CLS0, MIWT5_5, PAP0, PLR1, RAP0, SS0, SVW8, VWA1, VWB1, WG32_4_1 |

## pure_lda: pure GEMM, 256 B aligned leading dimensions

Pure GEMM (no Bias / ScaleAlphaVec / activation) with the leading dimensions padded for 256 B alignment (BF16, 2 B per element). `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`, RotatingBufferSize 384.

- `pure_lda/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]` with the leading dimensions of the original problem, RotatingBufferSize 384.
- `pure_lda/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`: merged ExactLogic for the 10 sizes below.

### Results (hipblaslt-bench, GPU0, pure GEMM, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+135.9%**, vs MAIDAS **34.9% -> 65.9%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P21 | 336x144x2108x7776 | 1388.88 | 2322.00 | 1649.50 | +40.8% | 59.8% | 84.2% |
| P22 | 656x736x2147x640 | 650.33 | 5006.00 | 1157.00 | +332.7% | 13.0% | 56.2% |
| P23 | 208x2256x2367x192 | 450.34 | 1569.00 | 1036.00 | +51.4% | 28.7% | 43.5% |
| P24 | 1488x1568x2773x16 | 929.52 | 10846.00 | 2183.00 | +396.8% | 8.6% | 42.6% |
| P25 | 624x1472x2930x928 | 2644.58 | 5456.00 | 5274.00 | +3.5% | 48.5% | 50.1% |
| P26 | 48x720x3028x208 | 116.22 | 211.38 | 128.23 | +64.8% | 55.0% | 90.6% |
| P27 | 1776x16x4186x784 | 978.63 | 2004.00 | 799.00 | +150.8% | 48.8% | 122.5% |
| P28 | 3120x288x6158x64 | 1187.00 | 5398.00 | 1829.00 | +195.1% | 22.0% | 64.9% |
| P29 | 3840x48x6527x208 | 940.19 | 2275.00 | 1409.50 | +61.4% | 41.3% | 66.7% |
| P30 | 160x5488x7242x32 | 1354.00 | 5862.00 | 3628.00 | +61.6% | 23.1% | 37.3% |

### Problem sizes and leading dimensions

| # | size (MxNxBatchxK) | LDA | LDB | LDC | LDD |
|---|---|---|---|---|---|
| P21 | 336x144x2108x7776 | 7808 | 7808 | 384 | 384 |
| P22 | 656x736x2147x640 | 640 | 640 | 768 | 768 |
| P23 | 208x2256x2367x192 | 192* | 192* | 208* | 208* |
| P24 | 1488x1568x2773x16 | 16* | 16* | 1536 | 1536 |
| P25 | 624x1472x2930x928 | 1024 | 1024 | 640 | 640 |
| P26 | 48x720x3028x208 | 256 | 256 | 48* | 48* |
| P27 | 1776x16x4186x784 | 896 | 896 | 1792 | 1792 |
| P28 | 3120x288x6158x64 | 64* | 64* | 3200 | 3200 |
| P29 | 3840x48x6527x208 | 256 | 256 | 3840 | 3840 |
| P30 | 160x5488x7242x32 | 32* | 32* | 160* | 160* |

`*`: not a multiple of 256 B.

## Notes

- The `fused` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, fused GEMM (Bias + ScaleAlphaVec), LDA = LDB = K.
- The `pure_lda` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, pure GEMM, 256 B aligned leading dimensions.
