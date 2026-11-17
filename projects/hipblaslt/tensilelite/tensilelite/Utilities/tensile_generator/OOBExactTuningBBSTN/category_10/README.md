# OOB BBS TN exact tuning, category 10: Large M and N (gfx1250, BF16 in/out, FP32 compute, TN)

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with unpadded leading
dimensions, LDA = LDB = K and LDC = LDD = M.

- `fused_gemm/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, keyed on `[M, N, 1, K]`.
- `fused_gemm/tuning/bf16_gfx1250_<category>_<M>x<N>x1x<K>.yaml`: one Tensile config per size that builds exactly that size's winner kernel and nothing else. Each was checked with `--build-only`: it produces one kernel whose name matches the logic entry byte for byte.

All 10 sizes hold the winner merged on 11/01. The 11/17 round analysed this category but did not retune it: in the 10/31 tuning every cluster shape lost to CD1_1 when measured on each size's own winning tile, so repeating that sweep was not worth the GPU time.

## Results (Tensile client, gfx1250, 256 CU; `vs MAIDAS` = MAIDAS us / measured us)

Category mean: gain vs previous **+0.00%**, vs MAIDAS **72.1% -> 72.1%**.

`previous us` is the old winner measured in the same client run as the new one, so no gain mixes runs, GPUs or builds.

| size (MxNxK) | MAIDAS us | previous us | new us | gain | previous vs MAIDAS | new vs MAIDAS |
|---|---:|---:|---:|---:|---:|---:|
| 16880x31072x7216 | 2213.46 | 3228.50 | 3228.50 | +0.00% | 68.6% | 68.6% |
| 21536x53248x5200 | 3570.83 | 5343.15 | 5343.15 | +0.00% | 66.8% | 66.8% |
| 24208x32416x2336 | 1162.42 | 1782.30 | 1782.30 | +0.00% | 65.2% | 65.2% |
| 25520x15760x7296 | 1719.18 | 2115.65 | 2115.65 | +0.00% | 81.3% | 81.3% |
| 26656x47824x1728 | 1422.94 | 2003.58 | 2003.58 | +0.00% | 71.0% | 71.0% |
| 36912x62832x4448 | 6234.40 | 8307.00 | 8307.00 | +0.00% | 75.0% | 75.0% |
| 48272x61088x5360 | 9331.90 | 13223.61 | 13223.61 | +0.00% | 70.6% | 70.6% |
| 51664x14976x7696 | 3521.58 | 4923.90 | 4923.90 | +0.00% | 71.5% | 71.5% |
| 51968x61744x4544 | 8583.74 | 10370.60 | 10370.60 | +0.00% | 82.8% | 82.8% |
| 56848x44448x1280 | 2083.91 | 3061.88 | 3061.88 | +0.00% | 68.1% | 68.1% |

### Winner key parameters

| size (MxNxK) | MT | MIWaveTile | GSU | GSUC | ClusterDim | WGM | HalfPLR | K splitting |
|---|---|---|---|---|---|---|---|---|
| 16880x31072x7216 | 256x352x128 | 8_11 | 1 | 0 | 1_1 | 32 | 1 | MultipleBuffer |
| 21536x53248x5200 | 256x352x128 | 8_11 | 1 | 0 | 1_1 | 32 | 1 | MultipleBuffer |
| 24208x32416x2336 | 256x352x128 | 8_11 | 0 | 0 | 1_1 | 1 | 1 | ? |
| 25520x15760x7296 | 256x256x128 | 8_8 | 1 | 0 | 2_4 | 1 | 1 | MultipleBuffer |
| 26656x47824x1728 | 256x384x64 | 8_12 | 1 | 0 | 1_1 | 32 | 3 | MultipleBuffer |
| 36912x62832x4448 | 256x384x64 | 8_12 | 1 | 0 | 2_4 | 1 | 3 | MultipleBuffer |
| 48272x61088x5360 | 256x352x128 | 8_11 | 1 | 0 | 1_1 | 32 | 1 | MultipleBuffer |
| 51664x14976x7696 | 256x352x128 | 8_11 | 1 | 0 | 1_1 | 32 | 1 | MultipleBuffer |
| 51968x61744x4544 | 256x352x128 | 8_11 | 1 | 0 | 2_4 | 1 | 1 | MultipleBuffer |
| 56848x44448x1280 | 256x384x64 | 8_12 | 1 | 0 | 1_1 | 32 | 3 | MultipleBuffer |
