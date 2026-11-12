# OOB BBS TN exact tuning, category 14 (M = 1; gfx1250, BF16 in/out, FP32 compute, TN)

Two sets, as agreed for every category: pure GEMM with LDA/LDB padded to 256 B, and fused GEMM with LDA = LDB = K.
Both logic files hold only the 10 sizes of this category (no older library), every kernel ActivationFuncCall 1 (AFC1).

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: pure GEMM + LDA/LDB padding set. ExactLogic for the 10 sizes, 5 solutions.
  The winners were tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128;
  the logic stores them in the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true).
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Pure GEMM,
  `Exact: [M, N, 1, K, M, M, LDA, LDB]` with the padded LDA = LDB. Group 1 = winner, group 2 = previous (10/27 final3) kernel.
- `fused_ldaK/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: fused GEMM without LDA padding set. ExactLogic for the 10 sizes, 10 solutions.
- `tuning_fused_ldaK/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Fused GEMM (Bias f32 + ScaleAlphaVec,
  activation none, ActivationFuncCall true), `Exact: [M, N, 1, K, M, M, K, K]`. Group 1 = winner, group 2 (if present) = previous kernel.
- Both: EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384. ExactLogic keys are [M, N, 1, K] only.

## Results (hipblaslt-bench, g04-2 GPU0, ROCm 10.3.0a20261111, develop 12e22a11ed1)

`--rotating 384 --use_gpu_timer --api_method cpp -i 1000`, -j = ~40 ms of warmup, best of 3 reps, no epilogue flags,
LDC = LDD = M, default heuristic (the logic's ExactLogic picks the kernel). vs MAIDAS = MAIDAS us / us;
vs 10/22 develop = 10/22 develop us / us - 1.

Category mean vs MAIDAS: fused GEMM (LDA = K) **94.4%**, pure GEMM + padding (this logic) **110.9%** (the same winners as pure-problem-type kernels 111.6%), best pure kernel at LDA = K 101.3%.
The sheet ("BBS TN OOB exact re-tune Winner" 256B-alignment columns, "OOB status" column D) uses the
pure-problem-type numbers, like the other categories' pure GEMM numbers; "us" below is this logic file as delivered.

### Pure GEMM + LDA/LDB padding (top-level logic)

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 10/22 develop us | us | vs MAIDAS | vs 10/22 develop | pure problem type us |
|---|---|---|---|---|---|---|---|---|
| 1x1696x418336 | 418432 | 0 | 102.09 | 153.53 | 97.87 | 104.3% | +56.9% | 97.77 |
| 1x7632x36352 | 36352 | 1 | 34.66 | 36.61 | 36.55 | 94.8% | +0.2% | 36.76 |
| 1x18976x187728 | 187776 | 0 | 487.28 | 475.13 | 451.16 | 108.0% | +5.3% | 450.92 |
| 1x40992x90624 | 90624 | 2 | 424.82 | 423.81 | 394.49 | 107.7% | +7.4% | 394.14 |
| 1x107408x5968 | 6016 | 3 | 90.36 | 90.48 | 81.53 | 110.8% | +11.0% | 80.66 |
| 1x108576x23952 | 24064 | 3 | 355.70 | 380.38 | 308.11 | 115.4% | +23.5% | 306.61 |
| 1x182656x14736 | 14848 | 2 | 369.52 | 328.63 | 291.64 | 126.7% | +12.7% | 290.85 |
| 1x467584x4256 | 4352 | 3 | 268.56 | 314.24 | 224.15 | 119.8% | +40.2% | 222.94 |
| 1x489984x17424 | 17536 | 2 | 1152.60 | 1197.60 | 900.98 | 127.9% | +32.9% | 899.18 |
| 1x615664x928 | 1024 | 4 | 81.23 | 121.33 | 87.27 | 93.1% | +39.0% | 83.48 |

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size (MxNxK) | sol | MAIDAS us | 10/22 develop us | 10/27 after us | us | vs MAIDAS | vs 10/22 develop |
|---|---|---|---|---|---|---|---|
| 1x1696x418336 | 0 | 102.09 | 153.53 | 108.39 | 136.75 | 74.7% | +12.3% |
| 1x7632x36352 | 1 | 34.66 | 36.61 | 35.19 | 37.65 | 92.1% | -2.8% |
| 1x18976x187728 | 2 | 487.28 | 475.13 | 446.30 | 598.70 | 81.4% | -20.6% |
| 1x40992x90624 | 3 | 424.82 | 423.81 | 377.34 | 394.08 | 107.8% | +7.5% |
| 1x107408x5968 | 4 | 90.36 | 90.48 | 88.51 | 92.57 | 97.6% | -2.3% |
| 1x108576x23952 | 5 | 355.70 | 380.38 | 324.58 | 344.37 | 103.3% | +10.5% |
| 1x182656x14736 | 6 | 369.52 | 328.63 | 316.18 | 322.91 | 114.4% | +1.8% |
| 1x467584x4256 | 7 | 268.56 | 314.24 | 237.52 | 253.89 | 105.8% | +23.8% |
| 1x489984x17424 | 8 | 1152.60 | 1197.60 | 995.63 | 1171.59 | 98.4% | +2.2% |
| 1x615664x928 | 9 | 81.23 | 121.33 | 112.63 | 118.09 | 68.8% | +2.7% |

## Winner key parameters

| size | set | kernel |
|---|---|---|
| 1x1696x418336 | pure+pad | MT16x256x256, MIWT1_4, WG16_8_1, TPSSK, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB2, SVW1, TDMIM0, AFEM1, SU32 |
| 1x1696x418336 | fused | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU16, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x7632x36352 | pure+pad | MT16x64x512, MIWT1_1, WG16_8_1, TPSN, GSU2, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x7632x36352 | fused | MT16x32x256, MIWT1_1, WG16_4_1, TPSN, GSU3, CD1_1, PGL2, HPLR0, LDSSI0, SS0, VWA1, VWB1, SVW8, TDMIM0, AFEM1, SU0 |
| 1x18976x187728 | pure+pad | MT16x256x256, MIWT1_4, WG16_8_1, TPSSK, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB2, SVW1, TDMIM0, AFEM1, SU32 |
| 1x18976x187728 | fused | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS0, VWA1, VWB1, SVW8, TDMIM0, AFEM1, SU0 |
| 1x40992x90624 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x40992x90624 | fused | MT16x192x256, MIWT1_3, WG16_8_1, TPSSK, GSU0, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x107408x5968 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x107408x5968 | fused | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS0, VWA1, VWB1, SVW8, TDMIM0, AFEM1, SU0 |
| 1x108576x23952 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x108576x23952 | fused | MT16x64x512, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x182656x14736 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x182656x14736 | fused | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS0, VWA1, VWB1, SVW8, TDMIM0, AFEM1, SU0 |
| 1x467584x4256 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x467584x4256 | fused | MT16x64x256, MIWT1_1, WG16_8_1, TPSSK, GSU0, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x489984x17424 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x489984x17424 | fused | MT16x192x256, MIWT1_3, WG16_8_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 1x615664x928 | pure+pad | MT16x64x256, MIWT1_1, WG16_8_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU32 |
| 1x615664x928 | fused | MT16x256x256, MIWT1_4, WG16_8_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB2, SVW1, TDMIM0, AFEM1, SU32 |

## Requirements / notes

- gfx1250 (256 CUs), Tensile with TDM, StreamK, ClusterDim and AdaptiveGemm support.
  TensileCreateLibrary --architecture=gfx1250 on each logic file (develop 12e22a11ed1): 0 rejects, AFC1 kernels only.
- 10/27 after us is the 1027C sheet column (measured with HIPBLASLT_BENCH_FREQ_ALL=true, full develop library);
  the new numbers leave FREQ_ALL unset, and memory-bound sizes measure slower than 10/27 this way.
- Correctness: hipblaslt-bench --verify_strict with an f32 bias vector + ScaleAlpha vector on every delivered kernel
  of both sets, forced by solution index on reduced sizes (dimensions above 2048 become 1024 + (x mod 512));
  category 14 also at full size. All pass (max norm_error 3.3e-05). The heuristic picked the ExactLogic winner for all sizes.
- The tuning configs were generated from the winning solutions (one ForkParameters step per kernel);
  they were not re-run with the Tensile client on 20261111. The numbers above are hipblaslt-bench.
