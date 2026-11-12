# OOB BBS TN exact tuning, category 13 (Very Large GEMMs; gfx1250, BF16 in/out, FP32 compute, TN)

Two sets, as agreed for every category: pure GEMM with LDA/LDB padded to 256 B, and fused GEMM with LDA = LDB = K.
Both logic files hold only the 10 sizes of this category (no older library), every kernel ActivationFuncCall 1 (AFC1).

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: pure GEMM + LDA/LDB padding set. ExactLogic for the 10 sizes, 8 solutions.
  The winners were tuned as pure GEMM (no bias / ScaleAlphaVec / activation) with LDA = LDB = ceil(K/128)*128;
  the logic stores them in the fused problem type (Bias + ScaleAlphaVec, ActivationFuncCall true).
- `tuning/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Pure GEMM,
  `Exact: [M, N, 1, K, M, M, LDA, LDB]` with the padded LDA = LDB. Group 1 = winner, group 2 = previous (10/27 final3) kernel.
- `fused_ldaK/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: fused GEMM without LDA padding set. ExactLogic for the 10 sizes, 4 solutions.
- `tuning_fused_ldaK/M<m>_N<n>_K<k>.yaml`: one Tensile config per size for that set. Fused GEMM (Bias f32 + ScaleAlphaVec,
  activation none, ActivationFuncCall true), `Exact: [M, N, 1, K, M, M, K, K]`. Group 1 = winner, group 2 (if present) = previous kernel.
- Both: EnqueuesPerSync 400 / NumWarmups 160 / NumBenchmarks 5, RotatingBufferSize 384. ExactLogic keys are [M, N, 1, K] only.

## Results (hipblaslt-bench, g04-2 GPU0, ROCm 10.3.0a20261111, develop 12e22a11ed1)

`--rotating 384 --use_gpu_timer --api_method cpp -i 1000`, -j = ~40 ms of warmup, best of 3 reps, no epilogue flags,
LDC = LDD = M, default heuristic (the logic's ExactLogic picks the kernel). vs MAIDAS = MAIDAS us / us;
vs 10/22 develop = 10/22 develop us / us - 1.

Category mean vs MAIDAS: fused GEMM (LDA = K) **61.2%**, pure GEMM + padding (this logic) **70.1%** (the same winners as pure-problem-type kernels 73.9%), best pure kernel at LDA = K 63.8%.
The sheet ("BBS TN OOB exact re-tune Winner" 256B-alignment columns, "OOB status" column D) uses the
pure-problem-type numbers, like the other categories' pure GEMM numbers; "us" below is this logic file as delivered.

### Pure GEMM + LDA/LDB padding (top-level logic)

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 10/22 develop us | us | vs MAIDAS | vs 10/22 develop | pure problem type us |
|---|---|---|---|---|---|---|---|---|
| 8224x11424x14160 | 14208 | 0 | 800.44 | 1436.22 | 1122.01 | 71.3% | +28.0% | 1084.85 |
| 8816x9968x15312 | 15360 | 1 | 787.56 | 1496.89 | 1153.60 | 68.3% | +29.8% | 1086.40 |
| 9344x13600x13344 | 13440 | 2 | 999.09 | 1711.61 | 1401.37 | 71.3% | +22.1% | 1370.25 |
| 10080x12480x11120 | 11136 | 0 | 823.30 | 1577.16 | 1277.29 | 64.5% | +23.5% | 1187.11 |
| 11312x12368x8928 | 8960 | 2 | 738.52 | 1348.57 | 1085.82 | 68.0% | +24.2% | 1055.41 |
| 11840x8576x11360 | 11392 | 3 | 676.05 | 1237.69 | 931.17 | 72.6% | +32.9% | 846.29 |
| 13120x13264x14880 | 14976 | 4 | 1510.57 | 2464.11 | 2046.35 | 73.8% | +20.4% | 1933.22 |
| 13328x15344x10080 | 10112 | 5 | 1212.13 | 2453.06 | 1825.93 | 66.4% | +34.3% | 1694.64 |
| 13568x12752x9328 | 9344 | 6 | 947.76 | 1757.93 | 1292.28 | 73.3% | +36.0% | 1250.84 |
| 13952x14736x13616 | 13696 | 7 | 1635.25 | 2953.81 | 2287.41 | 71.5% | +29.1% | 2178.32 |

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size (MxNxK) | sol | MAIDAS us | 10/22 develop us | 10/27 after us | us | vs MAIDAS | vs 10/22 develop |
|---|---|---|---|---|---|---|---|
| 8224x11424x14160 | 0 | 800.44 | 1436.22 | 1305.45 | 1342.61 | 59.6% | +7.0% |
| 8816x9968x15312 | 1 | 787.56 | 1496.89 | 1263.34 | 1272.97 | 61.9% | +17.6% |
| 9344x13600x13344 | 2 | 999.09 | 1711.61 | 1570.89 | 1596.70 | 62.6% | +7.2% |
| 10080x12480x11120 | 3 | 823.30 | 1577.16 | 1385.38 | 1406.25 | 58.5% | +12.2% |
| 11312x12368x8928 | 3 | 738.52 | 1348.57 | 1144.87 | 1216.68 | 60.7% | +10.8% |
| 11840x8576x11360 | 3 | 676.05 | 1237.69 | 1042.85 | 1024.15 | 66.0% | +20.9% |
| 13120x13264x14880 | 3 | 1510.57 | 2464.11 | 2371.26 | 2439.56 | 61.9% | +1.0% |
| 13328x15344x10080 | 3 | 1212.13 | 2453.06 | 2169.50 | 2127.71 | 57.0% | +15.3% |
| 13568x12752x9328 | 3 | 947.76 | 1757.93 | 1466.21 | 1516.95 | 62.5% | +15.9% |
| 13952x14736x13616 | 3 | 1635.25 | 2953.81 | 2714.45 | 2663.25 | 61.4% | +10.9% |

## Winner key parameters

| size | set | kernel |
|---|---|---|
| 8224x11424x14160 | pure+pad | MT256x256x128, MIWT8_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI1, SS1, VWA8, VWB4, SVW8, TDMIM1, AFEM8, SU0 |
| 8224x11424x14160 | fused | MT256x256x128, MIWT8_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB4, SVW8, TDMIM1, AFEM8, SU0 |
| 8816x9968x15312 | pure+pad | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 8816x9968x15312 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA4, VWB1, SVW4, TDMIM0, AFEM1, SU32 |
| 9344x13600x13344 | pure+pad | MT256x256x128, MIWT8_8, WG32_4_1, TPSN, GSU1, CD2_2, PGL2, HPLR1, LDSSI1, SS1, VWA8, VWB4, SVW8, TDMIM1, AFEM8, SU0 |
| 9344x13600x13344 | fused | MT224x384x128, MIWT14_6, WG16_8_1, TPSN, GSU1, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA2, VWB2, SVW2, TDMIM0, AFEM1, SU32 |
| 10080x12480x11120 | pure+pad | MT256x256x128, MIWT8_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI1, SS1, VWA8, VWB4, SVW8, TDMIM1, AFEM8, SU0 |
| 10080x12480x11120 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 11312x12368x8928 | pure+pad | MT256x256x128, MIWT8_8, WG32_4_1, TPSN, GSU1, CD2_2, PGL2, HPLR1, LDSSI1, SS1, VWA8, VWB4, SVW8, TDMIM1, AFEM8, SU0 |
| 11312x12368x8928 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 11840x8576x11360 | pure+pad | MT320x256x128, MIWT10_8, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA2, VWB4, SVW2, TDMIM0, AFEM1, SU0 |
| 11840x8576x11360 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13120x13264x14880 | pure+pad | MT320x256x128, MIWT10_8, WG32_4_1, TPSN, GSU1, CD4_4, PGL2, HPLR1, LDSSI0, SS1, VWA2, VWB4, SVW2, TDMIM0, AFEM1, SU0 |
| 13120x13264x14880 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13328x15344x10080 | pure+pad | MT320x256x128, MIWT10_8, WG32_4_1, TPSN, GSU1, CD2_2, PGL2, HPLR1, LDSSI0, SS1, VWA2, VWB4, SVW2, TDMIM0, AFEM1, SU0 |
| 13328x15344x10080 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13568x12752x9328 | pure+pad | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD2_2, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13568x12752x9328 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13952x14736x13616 | pure+pad | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD2_2, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |
| 13952x14736x13616 | fused | MT256x352x128, MIWT8_11, WG32_4_1, TPSN, GSU1, CD1_1, PGL2, HPLR1, LDSSI0, SS1, VWA8, VWB1, SVW8, TDMIM1, AFEM8, SU0 |

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
- Fused set: 8 sizes moved from the 10/27 kernel to an AFC1 CD1_1 candidate: 11312x12368x8928, 13568x12752x9328 and 8224x11424x14160 because their 10/27 kernels were AFC0, the others because the candidate was faster. Group 2 of their tuning config is the 10/27 kernel.
- These compute-bound sizes vary with clock: the slowest and fastest of the 3 reps differ by up to 1% on one size and set (0.5% on average).
