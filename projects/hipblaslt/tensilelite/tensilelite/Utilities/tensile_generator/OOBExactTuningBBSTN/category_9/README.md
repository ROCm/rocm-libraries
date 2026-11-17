# OOB BBS TN exact tuning, category 9: Large K, very small M and N (gfx1250, BF16 in/out, FP32 compute, TN)

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with unpadded leading
dimensions, LDA = LDB = K and LDC = LDD = M.

- `fused_gemm/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, keyed on `[M, N, 1, K]`.
- `fused_gemm/tuning/bf16_gfx1250_<category>_<M>x<N>x1x<K>.yaml`: one Tensile config per size that builds exactly that size's winner kernel and nothing else. Each was checked with `--build-only`: it produces one kernel whose name matches the logic entry byte for byte.

2 of the 10 sizes have a new winner from the 11/17 retune; the other 8 still hold the winner merged on 11/01.

## Results (Tensile client, gfx1250, 256 CU; `vs MAIDAS` = MAIDAS us / measured us)

Category mean: gain vs previous **+2.53%**, vs MAIDAS **95.7% -> 96.9%**.

`previous us` is the old winner measured in the same client run as the new one, so no gain mixes runs, GPUs or builds.

| size (MxNxK) | MAIDAS us | previous us | new us | gain | previous vs MAIDAS | new vs MAIDAS |
|---|---:|---:|---:|---:|---:|---:|
| 16x64x231696 | 10.86 | 12.93 | 12.93 | +0.00% | 84.0% | 84.0% |
| 16x64x659520 | 25.96 | 16.81 | 16.81 | +0.00% | 154.5% | 154.5% |
| 16x128x941728 | 43.81 | 26.31 | 26.31 | +0.00% | 166.5% | 166.5% |
| 32x32x423200 | 16.46 | 16.19 | 16.19 | +0.00% | 101.7% | 101.7% |
| 48x80x620224 | 23.38 | 22.71 | 22.71 | +0.00% | 103.0% | 103.0% |
| 48x128x796576 | 34.02 | 32.92 | 32.92 | +0.00% | 103.3% | 103.3% |
| **80x80x49872** | 5.03 | 12.72 | 10.81 | +17.72% | 39.6% | 46.6% |
| 80x80x322768 | 17.00 | 22.16 | 22.16 | +0.00% | 76.7% | 76.7% |
| 96x112x254816 | 14.39 | 21.01 | 21.01 | +0.00% | 68.5% | 68.5% |
| **128x32x222368** | 10.06 | 16.92 | 15.73 | +7.57% | 59.4% | 63.9% |

### What changed

| size | before | after |
|---|---|---|
| 80x80x49872 | MT16x128x256 MultipleBuffer | **MT16x32x256 MultipleBufferSingleKernel** |
| 128x32x222368 | MT32x32x512 MultipleBuffer | **MT16x32x512 MultipleBufferSingleKernel** |

Two levers did the work, and both only pay off under conditions worth recording:

- **MultipleBufferSingleKernel** folds the separate PostGSU reduction launch into the main kernel. It wins only on short kernels: +10% at 10.8 us and +7% at 15.7 us, but -29% at 20.9 us and -27% at 21.9 us. The break-even is around 16-20 us, which is where the fixed cost of the second launch stops being a large share of the total.
- **Finer macro tiles** raise the tile count so GlobalSplitU does not have to be large to fill 256 CUs. Both category-9 winners moved to MT16x32.

What did *not* work, measured rather than assumed: removing macro-tile edge waste. Zero-waste tiles lost on every size that has any. On 80x80x49872, holding the tile count at 5 and taking the N-direction waste from 37.5% to 0% moved performance by less than 0.4%, inside run noise.

### Winner key parameters

| size (MxNxK) | MT | MIWaveTile | GSU | GSUC | ClusterDim | WGM | HalfPLR | K splitting |
|---|---|---|---|---|---|---|---|---|
| 16x64x231696 | 64x16x512 | 1_1 | 64 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 16x64x659520 | 32x32x512 | 1_1 | 128 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 16x128x941728 | 32x32x512 | 1_1 | n1 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 32x32x423200 | 32x32x256 | 1_1 | 128 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 48x80x620224 | 64x32x512 | 2_1 | 64 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 48x128x796576 | 80x64x256 | 5_2 | 128 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 80x80x49872 | 16x32x256 | 1_1 | 24 | 0 | 1_1 | 1 | 0 | MultipleBufferSingleKernel |
| 80x80x322768 | 80x64x128 | 5_2 | n1 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 96x112x254816 | 96x96x128 | 3_3 | 128 | 1 | 1_1 | 1 | 0 | MultipleBuffer |
| 128x32x222368 | 16x32x512 | 1_1 | 32 | 0 | 1_1 | 1 | 0 | MultipleBufferSingleKernel |
