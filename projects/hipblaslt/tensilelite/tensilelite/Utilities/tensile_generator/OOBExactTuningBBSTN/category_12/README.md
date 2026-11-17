# OOB BBS TN exact tuning, category 12: Large M and K (gfx1250, BF16 in/out, FP32 compute, TN)

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with unpadded leading
dimensions, LDA = LDB = K and LDC = LDD = M.

- `fused_gemm/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, keyed on `[M, N, 1, K]`.
- `fused_gemm/tuning/bf16_gfx1250_<category>_<M>x<N>x1x<K>.yaml`: one Tensile config per size that builds exactly that size's winner kernel and nothing else. Each was checked with `--build-only`: it produces one kernel whose name matches the logic entry byte for byte.

All 10 sizes hold the winner merged on 11/01. The 11/17 round analysed this category but did not retune it: in the 10/31 tuning every cluster shape lost to CD1_1 on each size's own winning tile except 33264x7152x10736. That one does win with a cluster, but it already ships CD4_4 and sits at the Cs*Ck <= 16 cap, so there is no larger cluster left to try.

## Results (Tensile client, gfx1250, 256 CU; `vs MAIDAS` = MAIDAS us / measured us)

Category mean: gain vs previous **+0.00%**, vs MAIDAS **71.7% -> 71.7%**.

`previous us` is the old winner measured in the same client run as the new one, so no gain mixes runs, GPUs or builds.

| size (MxNxK) | MAIDAS us | previous us | new us | gain | previous vs MAIDAS | new vs MAIDAS |
|---|---:|---:|---:|---:|---:|---:|
| 22768x1632x61712 | 1348.41 | 1981.20 | 1981.20 | +0.00% | 68.1% | 68.1% |
| 25680x1312x59792 | 1203.58 | 1800.42 | 1800.42 | +0.00% | 66.8% | 66.8% |
| 31104x2496x61968 | 2744.50 | 4133.29 | 4133.29 | +0.00% | 66.4% | 66.4% |
| 31648x1728x52896 | 1685.77 | 2415.49 | 2415.49 | +0.00% | 69.8% | 69.8% |
| 33264x7152x10736 | 1472.17 | 2232.25 | 2232.25 | +0.00% | 66.0% | 66.0% |
| 35216x208x19952 | 167.20 | 206.09 | 206.09 | +0.00% | 81.1% | 81.1% |
| 48864x6144x18320 | 3152.26 | 4338.37 | 4338.37 | +0.00% | 72.7% | 72.7% |
| 51904x848x31376 | 872.47 | 1274.80 | 1274.80 | +0.00% | 68.4% | 68.4% |
| 59472x7632x35168 | 9146.77 | 11912.96 | 11912.96 | +0.00% | 76.8% | 76.8% |
| 64848x1328x60608 | 3075.42 | 3789.80 | 3789.80 | +0.00% | 81.2% | 81.2% |

### Winner key parameters

| size (MxNxK) | MT | MIWaveTile | GSU | GSUC | ClusterDim | WGM | HalfPLR | K splitting |
|---|---|---|---|---|---|---|---|---|
| 22768x1632x61712 | 384x208x128 | 6_13 | 1 | 0 | 1_1 | 8 | 1 | MultipleBuffer |
| 25680x1312x59792 | 256x272x128 | 4_17 | 1 | 0 | 1_1 | 8 | 1 | MultipleBuffer |
| 31104x2496x61968 | 256x320x128 | 4_20 | 1 | 0 | 1_1 | 8 | 2 | MultipleBuffer |
| 31648x1728x52896 | 384x192x128 | 6_12 | 1 | 0 | 1_1 | 8 | 2 | MultipleBuffer |
| 33264x7152x10736 | 384x224x64 | 12_7 | 1 | 0 | 4_4 | 1 | 1 | MultipleBuffer |
| 35216x208x19952 | 160x224x128 | 5_7 | 1 | 0 | 1_2 | 1 | 0 | MultipleBuffer |
| 48864x6144x18320 | 320x256x64 | 10_8 | 1 | 0 | 4_4 | 1 | 2 | MultipleBuffer |
| 51904x848x31376 | 320x288x128 | 10_9 | 1 | 0 | 1_1 | 8 | 1 | MultipleBuffer |
| 59472x7632x35168 | 256x320x64 | 4_20 | 1 | 0 | 4_4 | 1 | 2 | MultipleBuffer |
| 64848x1328x60608 | 384x224x128 | 12_7 | 1 | 0 | 2_2 | 1 | 1 | MultipleBuffer |
