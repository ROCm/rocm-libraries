# OOB BBS TN exact tuning, category 11: Large N and K (gfx1250, BF16 in/out, FP32 compute, TN)

Fused GEMM (Bias (f32) + ScaleAlphaVec, ActivationFuncCall) with unpadded leading
dimensions, LDA = LDB = K and LDC = LDD = M.

- `fused_gemm/logic/gfx1250_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs.yaml`: ExactLogic for the 10 sizes below, keyed on `[M, N, 1, K]`.
- `fused_gemm/tuning/bf16_gfx1250_<category>_<M>x<N>x1x<K>.yaml`: one Tensile config per size that builds exactly that size's winner kernel and nothing else. Each was checked with `--build-only`: it produces one kernel whose name matches the logic entry byte for byte.

2 of the 10 sizes have a new winner from the 11/17 retune; the other 8 still hold the winner merged on 11/01.

## Results (Tensile client, gfx1250, 256 CU; `vs MAIDAS` = MAIDAS us / measured us)

Category mean: gain vs previous **+1.16%**, vs MAIDAS **66.7% -> 67.2%**.

`previous us` is the old winner measured in the same client run as the new one, so no gain mixes runs, GPUs or builds.

| size (MxNxK) | MAIDAS us | previous us | new us | gain | previous vs MAIDAS | new vs MAIDAS |
|---|---:|---:|---:|---:|---:|---:|
| **368x8432x30048** | 64.25 | 166.03 | 155.43 | +6.82% | 38.7% | 41.3% |
| 752x49760x35952 | 821.74 | 1236.44 | 1236.44 | +0.00% | 66.5% | 66.5% |
| 912x13120x53472 | 439.91 | 635.89 | 635.89 | +0.00% | 69.2% | 69.2% |
| **1040x10928x28400** | 208.05 | 387.51 | 369.78 | +4.80% | 53.7% | 56.3% |
| 4144x40208x24656 | 2392.15 | 3455.37 | 3455.37 | +0.00% | 69.2% | 69.2% |
| 4464x10560x62896 | 1701.90 | 2616.70 | 2616.70 | +0.00% | 65.0% | 65.0% |
| 4528x54432x36224 | 5049.81 | 6176.39 | 6176.39 | +0.00% | 81.8% | 81.8% |
| 4768x44400x17904 | 2216.82 | 3191.50 | 3191.50 | +0.00% | 69.5% | 69.5% |
| 5216x25360x18784 | 1494.06 | 2089.59 | 2089.59 | +0.00% | 71.5% | 71.5% |
| 7360x60368x64608 | 16201.29 | 19772.14 | 19772.14 | +0.00% | 81.9% | 81.9% |

### What changed

| size | before | after |
|---|---|---|
| 368x8432x30048 | MT128x128x128 MultipleBuffer | **MT128x112x128 MultipleBuffer** |
| 1040x10928x28400 | MT208x256x128 MultipleBuffer | **MT208x256x128 StreamK** |

Two levers did the work, and both only pay off under conditions worth recording:

- **MultipleBufferSingleKernel** folds the separate PostGSU reduction launch into the main kernel. It wins only on short kernels: +10% at 10.8 us and +7% at 15.7 us, but -29% at 20.9 us and -27% at 21.9 us. The break-even is around 16-20 us, which is where the fixed cost of the second launch stops being a large share of the total.
- **Finer macro tiles** raise the tile count so GlobalSplitU does not have to be large to fill 256 CUs. Both category-9 winners moved to MT16x32.

What did *not* work, measured rather than assumed: removing macro-tile edge waste. Zero-waste tiles lost on every size that has any. On 80x80x49872, holding the tile count at 5 and taking the N-direction waste from 37.5% to 0% moved performance by less than 0.4%, inside run noise.

### Winner key parameters

| size (MxNxK) | MT | MIWaveTile | GSU | GSUC | ClusterDim | WGM | HalfPLR | K splitting |
|---|---|---|---|---|---|---|---|---|
| 368x8432x30048 | 128x112x128 | 2_7 | 1 | 0 | 1_1 | 8 | 0 | MultipleBuffer |
| 752x49760x35952 | 256x304x128 | 4_19 | 1 | 0 | 1_1 | 8 | 1 | MultipleBuffer |
| 912x13120x53472 | 256x256x128 | 8_8 | 1 | 0 | 1_1 | 8 | 2 | MultipleBuffer |
| 1040x10928x28400 | 208x256x128 | 13_4 | 0 | 0 | 1_1 | 8 | 2 | StreamK |
| 4144x40208x24656 | 384x224x128 | 12_7 | 1 | 0 | 1_1 | 8 | 1 | MultipleBuffer |
| 4464x10560x62896 | 256x384x64 | 8_12 | 1 | 0 | 2_2 | 1 | 3 | MultipleBuffer |
| 4528x54432x36224 | 320x288x128 | 10_9 | 1 | 0 | 2_4 | 1 | 1 | MultipleBuffer |
| 4768x44400x17904 | 320x288x64 | 10_9 | 1 | 0 | 2_4 | 1 | 1 | MultipleBuffer |
| 5216x25360x18784 | 256x320x128 | 4_20 | 1 | 0 | 1_1 | 8 | 2 | MultipleBuffer |
| 7360x60368x64608 | 320x288x64 | 10_9 | 1 | 0 | 4_4 | 1 | 1 | MultipleBuffer |
