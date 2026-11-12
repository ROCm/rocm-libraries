# OOB BBS TN exact tuning, category 15 (N = 1; gfx1250, BF16 in/out, FP32 compute, TN)

Two sets, as agreed for every category: pure GEMM with LDA/LDB padded to 256 B, and fused GEMM with LDA = LDB = K.
Both logic files hold only the 10 sizes of this category (no older library), every kernel ActivationFuncCall 1 (AFC1).

- `gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: pure GEMM + LDA/LDB padding set. ExactLogic for the 10 sizes, 5 solutions.
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

Category mean vs MAIDAS: fused GEMM (LDA = K) **99.3%**, pure GEMM + padding (this logic) **122.2%** (the same winners as pure-problem-type kernels 122.3%), best pure kernel at LDA = K 113.2%.
The sheet ("BBS TN OOB exact re-tune Winner" 256B-alignment columns, "OOB status" column D) uses the
pure-problem-type numbers, like the other categories' pure GEMM numbers; "us" below is this logic file as delivered.

### Pure GEMM + LDA/LDB padding (top-level logic)

| size (MxNxK) | LDA=LDB | sol | MAIDAS us | 10/22 develop us | us | vs MAIDAS | vs 10/22 develop | pure problem type us |
|---|---|---|---|---|---|---|---|---|
| 2240x1x922208 | 922240 | 0 | 279.18 | 415.93 | 254.44 | 109.7% | +63.5% | 254.19 |
| 5040x1x571264 | 571264 | 1 | 424.14 | 432.65 | 310.94 | 136.4% | +39.1% | 310.04 |
| 12928x1x240752 | 240768 | 2 | 355.90 | 432.72 | 362.54 | 98.2% | +19.4% | 362.90 |
| 32528x1x247840 | 247936 | 3 | 1146.70 | 1064.51 | 844.46 | 135.8% | +26.1% | 843.53 |
| 42128x1x114480 | 114560 | 4 | 683.69 | 824.12 | 531.39 | 128.7% | +55.1% | 531.64 |
| 42720x1x18784 | 18816 | 4 | 113.21 | 122.36 | 98.01 | 115.5% | +24.8% | 98.46 |
| 540080x1x6768 | 6784 | 4 | 526.15 | 552.58 | 393.72 | 133.6% | +40.3% | 392.44 |
| 578352x1x14768 | 14848 | 4 | 1203.74 | 1197.74 | 907.82 | 132.6% | +31.9% | 906.99 |
| 596864x1x11200 | 11264 | 4 | 746.76 | 764.06 | 714.17 | 104.6% | +7.0% | 713.08 |
| 732000x1x4896 | 4992 | 4 | 498.81 | 523.72 | 392.05 | 127.2% | +33.6% | 390.83 |

### Fused GEMM, LDA = LDB = K (`fused_ldaK/`)

| size (MxNxK) | sol | MAIDAS us | 10/22 develop us | 10/27 after us | us | vs MAIDAS | vs 10/22 develop |
|---|---|---|---|---|---|---|---|
| 2240x1x922208 | 0 | 279.18 | 415.93 | 279.59 | 299.14 | 93.3% | +39.0% |
| 5040x1x571264 | 1 | 424.14 | 432.65 | 304.11 | 368.12 | 115.2% | +17.5% |
| 12928x1x240752 | 2 | 355.90 | 432.72 | 387.27 | 419.67 | 84.8% | +3.1% |
| 32528x1x247840 | 3 | 1146.70 | 1064.51 | 892.79 | 992.52 | 115.5% | +7.3% |
| 42128x1x114480 | 4 | 683.69 | 824.12 | 587.94 | 726.86 | 94.1% | +13.4% |
| 42720x1x18784 | 4 | 113.21 | 122.36 | 97.67 | 120.12 | 94.3% | +1.9% |
| 540080x1x6768 | 4 | 526.15 | 552.58 | 454.88 | 529.50 | 99.4% | +4.4% |
| 578352x1x14768 | 4 | 1203.74 | 1197.74 | 1000.36 | 1205.09 | 99.9% | -0.6% |
| 596864x1x11200 | 5 | 746.76 | 764.06 | 701.51 | 757.86 | 98.5% | +0.8% |
| 732000x1x4896 | 4 | 498.81 | 523.72 | 446.28 | 510.97 | 97.6% | +2.5% |

## Winner key parameters

| size | set | kernel |
|---|---|---|
| 2240x1x922208 | pure+pad | MT128x16x512, MIWT2_1, WG64_2_1, TPSSK, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU32 |
| 2240x1x922208 | fused | MT128x16x512, MIWT2_1, WG64_2_1, TPSSK, GSU0, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU32 |
| 5040x1x571264 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU8, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 5040x1x571264 | fused | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU8, CD1_1, PGL0, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU32 |
| 12928x1x240752 | pure+pad | MT64x16x512, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 12928x1x240752 | fused | MT64x16x512, MIWT1_1, WG64_2_1, TPSSK, GSU0, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 32528x1x247840 | pure+pad | MT256x16x256, MIWT4_1, WG64_2_1, TPSN, GSU2, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA2, VWB1, SVW2, TDMIM0, AFEM1, SU0 |
| 32528x1x247840 | fused | MT64x16x512, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 42128x1x114480 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 42128x1x114480 | fused | MT192x16x256, MIWT3_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 42720x1x18784 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 42720x1x18784 | fused | MT192x16x256, MIWT3_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 540080x1x6768 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 540080x1x6768 | fused | MT192x16x256, MIWT3_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 578352x1x14768 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 578352x1x14768 | fused | MT192x16x256, MIWT3_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 596864x1x11200 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 596864x1x11200 | fused | MT128x16x256, MIWT2_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA2, VWB1, SVW2, TDMIM0, AFEM1, SU0 |
| 732000x1x4896 | pure+pad | MT64x16x256, MIWT1_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL2, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |
| 732000x1x4896 | fused | MT192x16x256, MIWT3_1, WG64_2_1, TPSN, GSU1, CD1_1, PGL1, HPLR0, LDSSI0, SS1, VWA1, VWB1, SVW1, TDMIM0, AFEM1, SU0 |

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
