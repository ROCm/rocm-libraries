# OOB BBS TN exact tuning, category 17: Small batch (gfx1250, BF16 in/out, FP32 compute, TN)

Two variants are provided for this category: `fused` and `pure_lda`.

## fused: fused GEMM, no 256 B alignment

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with LDA = LDB = K and LDC = LDD = M, i.e. the leading dimensions are not padded to 256 B.

- `fused/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size (fused GEMM: Bias (f32) + ScaleAlphaVec, ActivationFuncCall, `Exact: [M, N, batch, K]`, LDA = LDB = K, LDC = LDD = M, RotatingBufferSize 384).
- `fused/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below, 10 solutions.

### Results (hipblaslt-bench, GPU0, fused GEMM, LDA = LDB = K, tuned winner vs OOB)

Category mean: speedup vs OOB **+84.9%**, vs MAIDAS **39.5% -> 59.3%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P01 | 3264x2416x7x400 | 23.79 | 136.36 | 47.86 | +184.9% | 17.4% | 49.7% |
| P02 | 1392x4656x13x5488 | 313.61 | 661.00 | 582.60 | +13.5% | 47.4% | 53.8% |
| P03 | 6592x4848x26x2080 | 1127.82 | 2934.00 | 1556.00 | +88.6% | 38.4% | 72.5% |
| P04 | 4928x6656x40x16 | 182.26 | 1507.00 | 342.00 | +340.6% | 12.1% | 53.3% |
| P05 | 5280x7616x58x7232 | 9902.51 | 19659.00 | 15890.00 | +23.7% | 50.4% | 62.3% |
| P06 | 4016x7792x60x1200 | 1606.54 | 4898.00 | 2621.00 | +86.9% | 32.8% | 61.3% |
| P07 | 6896x6208x80x2848 | 6062.45 | 11380.00 | 10011.00 | +13.7% | 53.3% | 60.6% |
| P08 | 3504x7760x86x2784 | 4136.66 | 8306.00 | 6774.00 | +22.6% | 49.8% | 61.1% |
| P09 | 4704x3440x102x7040 | 6763.73 | 11514.00 | 11722.00 | -1.8% | 58.7% | 57.7% |
| P10 | 3856x5088x105x3680 | 4704.23 | 13666.00 | 7764.00 | +76.0% | 34.4% | 60.6% |

### Winner key parameters

| # | size | winner |
|---|---|---|
| P01 | 3264x2416x7x400 | MT256x256x64, AG1, CD2_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |
| P02 | 1392x4656x13x5488 | MT256x256x128, AG1, CD1_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1 |
| P03 | 6592x4848x26x2080 | MT256x256x64, AG1, CD2_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |
| P04 | 4928x6656x40x16 | MT128x128x32, AG1, CD1_1, CLS0, MIWT8_4, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB4, WG16_4_1 |
| P05 | 5280x7616x58x7232 | MT256x256x128, AG1, CD2_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1 |
| P06 | 4016x7792x60x1200 | MT256x256x128, AG1, CD1_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1 |
| P07 | 6896x6208x80x2848 | MT256x256x128, AG0, CD1_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1 |
| P08 | 3504x7760x86x2784 | MT256x256x64, AG1, CD2_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |
| P09 | 4704x3440x102x7040 | MT256x256x128, AG0, CD1_1, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1 |
| P10 | 3856x5088x105x3680 | MT256x256x64, AG1, CD2_2, CLS0, MIWT8_8, PAP0, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1 |

## pure_lda: pure GEMM, 256 B aligned leading dimensions

Pure GEMM (no Bias / ScaleAlphaVec / activation) with the leading dimensions padded for 256 B alignment (BF16, 2 B per element). `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`, RotatingBufferSize 384.

- `pure_lda/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]` with the leading dimensions of the original problem, RotatingBufferSize 384.
- `pure_lda/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`: merged ExactLogic for the 10 sizes below.

### Results (hipblaslt-bench, GPU0, pure GEMM, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+99.9%**, vs MAIDAS **39.7% -> 63.2%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P01 | 3264x2416x7x400 | 23.79 | 134.86 | 43.64 | +209.0% | 17.6% | 54.5% |
| P02 | 1392x4656x13x5488 | 313.61 | 659.60 | 544.20 | +21.2% | 47.5% | 57.6% |
| P03 | 6592x4848x26x2080 | 1127.82 | 2832.00 | 1468.00 | +92.9% | 39.8% | 76.8% |
| P04 | 4928x6656x40x16 | 182.26 | 1506.00 | 293.50 | +413.1% | 12.1% | 62.1% |
| P05 | 5280x7616x58x7232 | 9902.51 | 19396.00 | 15469.00 | +25.4% | 51.1% | 64.0% |
| P06 | 4016x7792x60x1200 | 1606.54 | 4957.00 | 2344.00 | +111.5% | 32.4% | 68.5% |
| P07 | 6896x6208x80x2848 | 6062.45 | 11382.00 | 9784.00 | +16.3% | 53.3% | 62.0% |
| P08 | 3504x7760x86x2784 | 4136.66 | 8257.00 | 6203.00 | +33.1% | 50.1% | 66.7% |
| P09 | 4704x3440x102x7040 | 6763.73 | 11467.00 | 11779.00 | -2.6% | 59.0% | 57.4% |
| P10 | 3856x5088x105x3680 | 4704.23 | 13634.00 | 7609.00 | +79.2% | 34.5% | 61.8% |

### Problem sizes and leading dimensions

| # | size (MxNxBatchxK) | LDA | LDB | LDC | LDD |
|---|---|---|---|---|---|
| P01 | 3264x2416x7x400 | 512 | 512 | 3328 | 3328 |
| P02 | 1392x4656x13x5488 | 5504 | 5504 | 1392* | 1392* |
| P03 | 6592x4848x26x2080 | 2176 | 2176 | 6656 | 6656 |
| P04 | 4928x6656x40x16 | 16* | 16* | 4992 | 4992 |
| P05 | 5280x7616x58x7232 | 7232* | 7232* | 5280* | 5280* |
| P06 | 4016x7792x60x1200 | 1280 | 1280 | 4096 | 4096 |
| P07 | 6896x6208x80x2848 | 2944 | 2944 | 6912 | 6912 |
| P08 | 3504x7760x86x2784 | 2816 | 2816 | 3584 | 3584 |
| P09 | 4704x3440x102x7040 | 7040 | 7040 | 4704* | 4704* |
| P10 | 3856x5088x105x3680 | 3712 | 3712 | 3968 | 3968 |

`*`: not a multiple of 256 B.

## Notes

- The `fused` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, fused GEMM (Bias + ScaleAlphaVec), LDA = LDB = K.
- The `pure_lda` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, pure GEMM, 256 B aligned leading dimensions.
