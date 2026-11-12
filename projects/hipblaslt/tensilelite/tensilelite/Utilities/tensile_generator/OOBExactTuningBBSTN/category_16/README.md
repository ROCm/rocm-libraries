# OOB BBS TN exact tuning, category 16 (K = 1; gfx1250, BF16 in/out, FP32 compute, TN)

Two sets, as agreed for every category: pure GEMM with LDA/LDB padded to 256 B, and fused GEMM with LDA = LDB = K.
Both logic files hold only the 10 sizes of this category (no older library), every kernel ActivationFuncCall 1 (AFC1).

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: pure GEMM + LDA/LDB padding set. ExactLogic for the 10 sizes, 6 solutions.
  The winners were tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128;
  the logic stores them in the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true).
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Pure GEMM,
  `Exact: [M, N, 1, K, M, M, LDA, LDB]` with the padded LDA = LDB. Group 1 = winner, group 2 = previous (10/27 final3) kernel.
- `fused_ldaK/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: fused GEMM without LDA padding set. ExactLogic for the 10 sizes, 6 solutions.
- `tuning_fused_ldaK/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Fused GEMM (Bias f32 + ScaleAlphaVec,
  activation none, ActivationFuncCall true), `Exact: [M, N, 1, K, M, M, K, K]`. Group 1 = winner, group 2 (if present) = previous kernel.
- Both: EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384. ExactLogic keys are [M, N, 1, K] only.

## Results (hipblaslt-bench, g04-2 GPU0, ROCm 10.3.0a20261111, develop 12e22a11ed1)

`--rotating 384 --use_gpu_timer --api_method cpp -i 1000`, -j = ~40 ms of warmup, best of 3 reps, no epilogue flags,
LDC = LDD = M, default heuristic (the logic's ExactLogic picks the kernel). vs MAIDAS = MAIDAS us / us;
vs 10/22 develop = 10/22 develop us / us - 1.

Category mean vs MAIDAS: fused GEMM (LDA = K) **25.1%**, pure GEMM + padding (this logic) **24.5%** (the same winners as pure-problem-type kernels 29.1%), best pure kernel at LDA = K 30.7%.
The sheet ("BBS TN OOB exact re-tune Winner" 256B-alignment columns, "OOB status" column D) uses the
pure-problem-type numbers, like the other categories' pure GEMM numbers; "us" below is this logic file as delivered.

### Pure GEMM + LDA/LDB padding (top-level logic)

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 10/22 develop us | us | vs MAIDAS | vs 10/22 develop | pure problem type us |
|---|---|---|---|---|---|---|---|---|
| 640x1020544x1 | 128 | 0 | 88.60 | 572.97 | 377.76 | 23.5% | +51.7% | 303.89 |
| 2080x1033808x1 | 128 | 1 | 311.43 | 1713.15 | 1368.17 | 22.8% | +25.2% | 1217.86 |
| 12912x36976x1 | 128 | 2 | 66.52 | 438.05 | 278.44 | 23.9% | +57.3% | 235.49 |
| 22256x233872x1 | 128 | 3 | 699.02 | 4118.91 | 2869.31 | 24.4% | +43.6% | 2335.89 |
| 56624x55664x1 | 128 | 2 | 424.00 | 2515.90 | 1738.20 | 24.4% | +44.7% | 1419.74 |
| 169552x16384x1 | 128 | 0 | 377.28 | 2259.07 | 1833.50 | 20.6% | +23.2% | 1354.50 |
| 172256x5152x1 | 128 | 4 | 125.67 | 917.59 | 482.14 | 26.1% | +90.3% | 426.09 |
| 227056x19424x1 | 128 | 2 | 596.33 | 3637.11 | 2398.77 | 24.9% | +51.6% | 2008.87 |
| 414944x7088x1 | 128 | 5 | 407.03 | 2607.80 | 1515.98 | 26.8% | +72.0% | 1346.93 |
| 639824x1408x1 | 128 | 5 | 133.74 | 918.59 | 484.16 | 27.6% | +89.7% | 432.75 |

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size (MxNxK) | sol | MAIDAS us | 10/22 develop us | 10/27 after us | us | vs MAIDAS | vs 10/22 develop |
|---|---|---|---|---|---|---|---|
| 640x1020544x1 | 0 | 88.60 | 572.97 | 384.41 | 357.85 | 24.8% | +60.1% |
| 2080x1033808x1 | 1 | 311.43 | 1713.15 | 1504.09 | 1480.00 | 21.0% | +15.8% |
| 12912x36976x1 | 2 | 66.52 | 438.05 | 281.90 | 283.70 | 23.4% | +54.4% |
| 22256x233872x1 | 3 | 699.02 | 4118.91 | 2797.60 | 2750.85 | 25.4% | +49.7% |
| 56624x55664x1 | 4 | 424.00 | 2515.90 | 1638.09 | 1622.43 | 26.1% | +55.1% |
| 169552x16384x1 | 5 | 377.28 | 2259.07 | 1579.99 | 1553.02 | 24.3% | +45.5% |
| 172256x5152x1 | 4 | 125.67 | 917.59 | 466.90 | 473.75 | 26.5% | +93.7% |
| 227056x19424x1 | 1 | 596.33 | 3637.11 | 2312.13 | 2270.94 | 26.3% | +60.2% |
| 414944x7088x1 | 3 | 407.03 | 2607.80 | 1530.44 | 1496.73 | 27.2% | +74.2% |
| 639824x1408x1 | 5 | 133.74 | 918.59 | 503.48 | 506.32 | 26.4% | +81.4% |

## Winner key parameters

| size | set | kernel |
|---|---|---|
| 640x1020544x1 | pure+pad | MT128x256x32, MIWT4_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA4, VWB8, SVW4, TDMIM0, AFEM1, SU32 |
| 640x1020544x1 | fused | MT128x256x32, MIWT4_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA4, VWB8, SVW4, TDMIM0, AFEM1, SU32 |
| 2080x1033808x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 2080x1033808x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 12912x36976x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 12912x36976x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 22256x233872x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 22256x233872x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 56624x55664x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 56624x55664x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 169552x16384x1 | pure+pad | MT128x256x32, MIWT4_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA4, VWB8, SVW4, TDMIM0, AFEM1, SU32 |
| 169552x16384x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 172256x5152x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 172256x5152x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 227056x19424x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 227056x19424x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 414944x7088x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 414944x7088x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 639824x1408x1 | pure+pad | MT256x128x32, MIWT8_4, WG32_4_1, TPSDP, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |
| 639824x1408x1 | fused | MT256x128x32, MIWT8_4, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM0, AFEM1, SU32 |

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
- K = 1: padding LDA/LDB to 128 only spreads A and B out, so the pure+pad set is slower than pure GEMM with LDA = K.
