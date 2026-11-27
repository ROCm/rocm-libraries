# OOB BBS TN exact tuning, category 17: Small batch (gfx1250, BF16 in/out, FP32 compute, TN)

Three variants are provided for this category: `fused`, `pure_lda` and `tdmsi_tf`.

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

## tdmsi_tf: TDM store + tail folding, fused GEMM with CompactLoopStore, 256 B aligned leading dimensions

Winners of the TDM store re-tune (`TDMStoreInst` in the search space, [#15133](https://github.com/ROCm/rocm-libraries/pull/15133)) on a build with tail folding ([#16312](https://github.com/ROCm/rocm-libraries/pull/16312)), re-tuned with a wider `DepthU` list, then rebuilt as fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with `CompactLoopStore` on. Leading dimensions padded for 256 B alignment (same as `pure_lda`), RotatingBufferSize 384.

- `tdmsi_tf/tuning/Problem_<nn>_40_<category>_batch_m_<m>_n_<n>_batch_<b>_k_<k>_exact.yaml`: the TDM store tuning yaml of one size, pure GEMM (no Bias / ScaleAlphaVec / activation), `TDMStoreInst: [false, true]`, `Exact: [M, N, batch, K, LDD, LDC, LDA, LDB]`. The logic winners come from re-tuning these yamls with a wider `DepthU` list (e.g. `[128, 256, 512]` instead of `[128]`, where LDS allows); the yamls here do not contain it (only P12, in category 18, does).
- `tdmsi_tf/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: merged ExactLogic for the 10 sizes below; the winners with the epilogue added and `CompactLoopStore: true`.

### Results (hipblaslt-bench C++ API, GPU0, fused GEMM + CompactLoopStore, 256 B aligned leading dimensions, tuned winner vs OOB)

Category mean: speedup vs OOB **+110.1%**, vs MAIDAS **48.0% -> 79.3%** (speedup = (OOB us - tuned us) / tuned us; vs MAIDAS = MAIDAS us / measured us).

| # | size (MxNxBatchxK) | MAIDAS us | OOB us | tuned us | speedup vs OOB | OOB vs MAIDAS | tuned vs MAIDAS |
|---|---|---|---|---|---|---|---|
| P01 | 3264x2416x7x400 | 23.79 | 131.22 | 40.70 | +222.4% | 18.1% | 58.4% |
| P02 | 1392x4656x13x5488 | 313.61 | 562.00 | 375.50 | +49.7% | 55.8% | 83.5% |
| P03 | 6592x4848x26x2080 | 1127.82 | 2088.50 | 1328.50 | +57.2% | 54.0% | 84.9% |
| P04 | 4928x6656x40x16 | 182.26 | 1513.43 | 274.29 | +451.8% | 12.0% | 66.4% |
| P05 | 5280x7616x58x7232 | 9902.51 | 15388.00 | 12025.00 | +28.0% | 64.4% | 82.3% |
| P06 | 4016x7792x60x1200 | 1606.54 | 4611.00 | 2042.00 | +125.8% | 34.8% | 78.7% |
| P07 | 6896x6208x80x2848 | 6062.45 | 10665.00 | 7146.00 | +49.2% | 56.8% | 84.8% |
| P08 | 3504x7760x86x2784 | 4136.66 | 7509.00 | 4908.00 | +53.0% | 55.1% | 84.3% |
| P09 | 4704x3440x102x7040 | 6763.73 | 9982.00 | 8175.00 | +22.1% | 67.8% | 82.7% |
| P10 | 3856x5088x105x3680 | 4704.23 | 7719.00 | 5443.00 | +41.8% | 60.9% | 86.4% |

### Winner key parameters

`TDMStore1` = `TDMStoreInst: true` (TDM store), `TDMStore0` = buffer store.

| # | size | winner |
|---|---|---|
| P01 | 3264x2416x7x400 | MT256x256x64, AG0, CD1_2, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1, TDMStore1 |
| P02 | 1392x4656x13x5488 | MT256x256x128, AG0, CD1_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P03 | 6592x4848x26x2080 | MT256x256x128, AG0, CD2_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P04 | 4928x6656x40x16 | MT128x128x32, AG1, CD1_1, CLS1, MIWT8_4, PAP0, PGR0, PLR0, RAP0, SS1, SVW8, VWA8, VWB4, WG16_4_1, TDMStore0 |
| P05 | 5280x7616x58x7232 | MT256x256x128, AG0, CD2_2, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P06 | 4016x7792x60x1200 | MT256x256x128, AG0, CD1_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P07 | 6896x6208x80x2848 | MT256x240x128, AG0, CD1_1, CLS1, MIWT4_15, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB1, WG64_2_1, TDMStore1 |
| P08 | 3504x7760x86x2784 | MT256x256x64, AG1, CD2_2, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW8, VWA8, VWB8, WG32_4_1, TDMStore0 |
| P09 | 4704x3440x102x7040 | MT256x256x128, AG0, CD1_1, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |
| P10 | 3856x5088x105x3680 | MT256x256x128, AG0, CD2_2, CLS1, MIWT8_8, PAP0, PGR2, PLR1, RAP0, SS1, SVW4, VWA4, VWB4, WG32_4_1, TDMStore1 |

## Notes

- The `fused` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, fused GEMM (Bias + ScaleAlphaVec), LDA = LDB = K.
- The `pure_lda` results above are measured with hipblaslt-bench on GPU0: tuned winner vs OOB baseline, pure GEMM, 256 B aligned leading dimensions.
- The `tdmsi_tf` results above are measured with hipblaslt-bench (C++ API) on GPU0: tuned winner vs OOB baseline, fused GEMM with CompactLoopStore, 256 B aligned leading dimensions, build develop + #15133 (84a8b39, includes tail folding).
