<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Conv engines: a map for heuristics work

This page is for building UHD heuristics and getting good out-of-the-box (non-autotuned) selection for the conv engine group. It covers three things:
- which engine is built for which regime;
- what each engine's configs vary;
- where timing data comes from.

What each engine computes is in `README.md`. How the catalogs are produced is in `generators/README.md`.

## Engines by regime

No single engine is meant to win everywhere. Each one targets a regime, and most graphs admit several engines. Picking the right engine, and the right config within it, is the heuristic's job.

| Engine(s) | Built for | Directions | Archs | Configs |
|---|---|---|---|---|
| `CONV_IGEMM_FWD/BWD` (K1) | Dense and grouped implicit GEMM, the general workhorse; compute-bound shapes; NCHW via direct channels-first epilogue (`cf`) or staging | fwd, bwd-data | gfx908-gfx950 (MFMA), gfx11/gfx12 (WMMA), FMA everywhere | 1648 |
| `CONV_IGEMM_NCHW_BWD` (K7) | NCHW bwd-data in place: strided pointwise, patchify, 1-D/vocoder ConvTranspose, dilated rows, 3-D stride 2. Weak on large compute-bound GEMMs (K1 wins there). | bwd-data | MFMA, WMMA, FMA | 1839 |
| `CONV_DIRECT_FWD/BWD` (K2) | Stride-1 halo-tile direct conv, 3x3 / 5x5 / 7x7 / 1x7 / 3x3x3. The `rows` kind (hipconv-class filter-row reuse) is the large-channel dense path on gfx950 and gfx90a, 2-D and 3x3x3; on gfx90a its ping-pong tiles win from about 256 channels (16-bit) and at every channel count in fp32, while the halo tile keeps 96-192-channel 16-bit layers. | fwd, bwd-data (stride 1) | MFMA, WMMA, FMA | 490 |
| `CONV_WINOGRAD_FWD/BWD` (K5) | 3x3 stride 1 (stride 2 by phases). `coop` is the cooperative WMMA/MFMA F(2,3), and F(3,2) for stride 2; the fused VALU and register variants target gfx906, grouped, and fp32 on RDNA (including 24-wide K blocks). | fwd, bwd-data | all | 324 |
| `CONV_FFT_FWD/BWD/WRW` | Large filters: dense 5x5 and up with many channels (fp32 especially; 16-bit from about 11x11), 1-D filters of 15+ taps, large-kernel depthwise (31x31: RepLKNet / SLaK class). Configs vary the transform size (`th` x `tw`: 16, 32, 64 square, 1 x 32 / 1 x 64 rows: the tile must hold the dilated filter; larger tiles for larger filters and images), the transform block (`pb` tiles; `tb` threads, the least that covers the row pairs or 256) and the product tile (`gn` 64 or 16 output channels, `tm` 4 or 8 tiles a thread). Numerical note FFT. | fwd, bwd-data, wrw (stride 1) | all (FMA; no matrix cores) | 198 fwd, 198 bwd, 132 wrw |
| `CONV_WRW` (K3) | Backward weights. Several loops: row ring, output rows (any stride), height-stride phased ring, gfx950 direct (hipconv `direct_wgrad` class), small-C rows for stems, and FMA rows for exact fp32 on RDNA. At one split, configs with `dws` 1 write dw directly with no reduce. A generic tier (`loop` 8) covers the filter widths, strides and dilations no specialized config has, and x/dy in any strides (see "Two tiers" below). | wrw | MFMA, WMMA, FMA | 4755 |
| `CONV_GROUP_FWD/BWD` (K4) | Small-group grouped conv (2 ≤ C/g ≤ 16, ResNeXt and RegNet): one 16x16 MMA serves 16/(C/g) groups through a block-diagonal filter. Wide groups (17 ≤ C/g ≤ 64) pad each group to 16-channel blocks. K1's `fuse` knob is its implicit-GEMM alternative. | fwd (s1, s2), bwd-data (s1; s2 by phases) | MFMA, WMMA; fp32 by FMA on RDNA | 2042 |
| `CONV_GROUP_WRW` (K4) | Backward weights for small groups (2 ≤ C/g ≤ 16): the same block-diagonal 16x16 MMA, dy positions of a th x tw tile flattened into the atom's k-steps (any plane width), x and dy read in place; deterministic split reduce. Groups of 24 and 32 (`nbi` 2, MFMA only) as dense per-group block products. K3 (`CONV_WRW`) is its implicit-GEMM alternative; groups past 32 channels stay with K3. | wrw (any stride on rows and columns per config: s1, s2; any depth stride) | MFMA, WMMA; fp32 also by FMA (all archs, the only fp32 path on RDNA) | 3132 |
| `CONV_STEM_BWD` | Bwd-data for stems with C ≤ 4 (7x7 s2, 3x3 s2, patchify) | bwd-data | MFMA, WMMA, FMA | 1187 |
| `CONV_SMALLC_FWD` | Forward stems with C ≤ 4 | fwd | MFMA, WMMA, FMA | 606 |
| `CONV_DEPTHWISE_MFMA_FWD/BWD` | Depthwise on matrix cores (gfx950 hipconv class), NHWC and NCHW | fwd, bwd-data (s1); fwd (s2) | gfx908-gfx950 | 640 |
| `CONV_DEPTHWISE_ROWS_FWD/BWD` | Depthwise by channel rows: 1-D/vocoder depthwise, dilated, strided ConvTranspose1d | fwd, bwd-data | all (VALU) | 222 |
| `CONV_DEPTHWISE_VALU_FWD/BWD` | General VALU depthwise, plus the strided bwd `phase` variant | fwd, bwd-data | all | 168 |
| `CONV_DEPTHWISE_VALU_WRW` | Depthwise backward weights (register strips; NHWC and NCHW) | wrw | all | 42 |
| `CONV_NAIVE_FWD/BWD/WRW` | Correctness floor: serves every graph | all | all | 9 |
| `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` (hipblaslt-provider) | Explicit GEMM (im2col/col2im) | all | all | BLAS-internal |

The config counts are the full catalog. A graph admits only the configs whose gates accept it. For example, K7 admits about 28-50 per graph and `CONV_STEM_BWD` about 4-9 (at most 21).

### Two tiers

An engine's configs come in two tiers:
- **Specialized** configs fix the geometry at compile time (filter width, stride, dilation, sometimes filter height) and serve the hot shapes. There can be many of them.
- **Generic** configs take the geometry at runtime. There are few of them, and each is admitted only where no specialized config's class serves the graph. They exist to beat the BLAS/naive fallback on the long tail, not to compete with the specialized tier.

In `CONV_WRW` the generic tier is `loop` 8. It has a runtime filter, strides and dilation, `fr` taps per block, and reads x and dy through their strides, so padded or sliced activations are served too. Each generic config carries a `spec` string: the specialized classes the catalog serves on its architectures (`<atom_id><kind><layout>.<s>.<sw>.<dw>.<fr>`). The host declines the generic config wherever one of those classes serves the graph. A graph that a specialized matrix-core config serves therefore never sees the generic configs. The FMA loops (exact fp32) are not counted as specialized.

Measured against the BLAS path, 16-bit channels-last:
- gfx950: 1.5-1.9x faster on 2-D 5x5/7x7 and 3-D 1x5x5.
- gfx1201: 2.6-3.2x faster on the same shapes, and 1.7x on 1-D k3.
- 1-D k9: 0.69x on gfx950 and 0.96x on gfx1201, so BLAS still wins there. Heuristics should rank BLAS above the generic tier for it.
- ASPP-class filters (at most 9 taps, dilation 6 or more): BLAS won on every arch measured (d18 16-bit 0.6-0.7x, d6 fp32 0.6x). The generic tier declines them.
- RDNA fp32 runs the generic tier's FMA form, because RDNA has no fp32 matrix core. On gfx1201, against BLAS, it was faster on 5x5 (1.6x), 1-D k9 (1.4x), 3x3 stride 3 (4.2x) and 5x5x5 (3.5x). It was slower on stems and patchify-class width strides (4.2-7.9x behind), so it declines fewer than 8 input channels per group and width strides of 4 or more.

## What the knobs mean

Every config is a set of integer knobs, listed in each engine's `*.kdp.json` under `kernelDescriptors[].metadata`. The recurring ones:

| Knob | Meaning |
|---|---|
| `atom` / `atom_id` | Matrix-core instruction shape (MFMA 16x16 or 32x32, gfx950 x32/x16, WMMA gfx11/gfx12) or FMA |
| `bm`, `bn`, `bk` (also `bq`, `bc`, `bmq`) | Block tile in the GEMM's M, N and K |
| `wm`, `wn` | Wave layout |
| `threads`, `vec` | Block size; vector width in elements |
| `stages`, `dma` | LDS pipeline depth; gfx950 direct-to-LDS loads |
| `swz` | XOR-swizzled LDS layout |
| `xcd` | XCD-contiguous block order (gfx942/gfx950) |
| `cf` | Write channels-first output directly instead of staging it |
| `wdirect` | bwd-data reads the packed filter in place |
| `pack` | Taps × channels flattened in the reduction |
| `fuse` | F small groups run as one block-diagonal group |
| `halo` | Extra dy/x window beyond the tile (dilation and filter reach) |
| `bpu` / `split` | Split of the reduction over blocks: fp32 partials, then a fixed-order reduce. Deterministic, no atomics. |
| `dws` (K3) | At one split, write dw directly (no workspace, no reduce) |
| `loop` (K3), `kind` (K2), `variant` (K5, depthwise) | Which kernel body runs |

Engine-specific knobs (`taps`, `stride`, `flat`, `ring`, `phase`, `fold`, ...) are documented in the engine's `*Problem.hpp`.

**The hardest choice is often split-K.** The planner picks the split count from a device-unit model (`packs/DeviceUnits.hpp`: CUs on gfx9, WGPs on gfx10+) and fills whole dispatch rounds. Configs that differ only in `bpu` exist so autotune can override it.

## Timing data

**Per-config timings.** With autotune (`HIPDNN_FORCE_BENCHMARKING=1`), each ingestor engine times every admitted config on the first execute and keeps the fastest. Every candidate's time and its problem features are logged as a JSON record at INFO plugin log level (`BenchmarkPlan::logCandidateTiming`). That is the per-config training data the `hipdnn-uhd` skill's pipeline consumes. The BLAS engines tune internally and report only their best.

**Problem sets in use for measuring this group**, as dnn-benchmarking graph JSON:
- **conv-kpi subset:** 1998 shapes chosen from a 9471-shape real-model corpus to keep its range: every direction, dtype, layout, conv type and filter class. It covers 31 models.
- **rapid:** 86 shapes. Every conv class, direction, dtype and layout.
- **rdna_av:** 414 audio/video shapes (vocoders, video VAEs, conformer, segmentation).
- **depthwise** and **pointwise** tiers.

Reference results per GPU are each engine's autotuned best time per graph, against MIOpen in search mode (gfx950, gfx942, gfx90a, gfx1201, gfx1100/gfx1101, gfx908, gfx906). They come as per-graph CSVs: MIOpen time, our best time, best engine, and MIOpen's solver.

Ask the conv owners for the graph sets and the latest results; they are not in this repo.

## Things that bite

- **First-run cost.** With autotune, every admitted config compiles through HIPRTC on first use. Measured median cold compiles are 0.3-0.8 s per config on CDNA and 0.4-1.5 s on RDNA. The only configs over 10 s are K7's fp32 FMA stride-8 configs (10-20 s). Summed over its admitted configs, a graph's median first-run cost is about 30 s. The worst measured graph is a ResNeXt-50 grouped NCHW bwd-data graph, where K1 alone admits 91-131 configs: 134 s on gfx942 and 224 s on gfx950. Good OOTB selection means picking well without compiling the whole candidate list. The HIPRTC disk cache lives under `<cache>/hip-kernel-provider/hiprtc/<key>.co`. The key is a sha256 over the HIPRTC version, the source, every embedded include and every option (architecture included). A pre-populated cache directory is how commonly needed instances can ship pre-compiled.
- **Catalogs are still moving.** K5 `coop`, K1 core (`swz`, `dma`, `fuse`, `cf`), K3, K4 and the depthwise engines all change week to week. Models trained now will need retraining.
- **Timing noise.** Jobs sharing a node can inflate times unevenly between engines; one shape read 0.85 sharing a node and 1.29 alone. Kernels under ~50 µs are noisy per shape. Measure isolated, and weigh by absolute time.
- **Configs that spill registers still win in places.** Some RDNA K3 ring tiles spill and are still the fastest. Don't prune on spills.
- **TF32.** Three-bf16 configs register only under `HIP_KERNEL_PROVIDER_CONV_TF32`. Leave it unset for normal fp32 selection.
- **Winograd workspace.** `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB` caps K5's workspace. `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES` overrides the last-level-cache size its planner assumes; HIP doesn't report MALL.
- **Winograd engines carry the `WINOGRAD` numerical note** in their engine descriptor, so callers can opt out of them for numerical reasons.
- **bf16 Winograd is declined** unless `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_BF16` is set. The matrix-core bf16 operands round the transformed tiles to bf16, past the precision model, and MIOpen has no bf16 Winograd to match. The instances stay; once plans can carry a numerical note, the gate becomes a tag. Deferred cost, full NCHW tier at 762934be264: bf16 Winograd was our fastest on 284 graphs on gfx1201 (503 ms call-weighted lost to the next engine, of 4498 ms) and 73 on gfx1100 (440 ms). The NHWC tier has none.
- **fp32 F(4,3) Winograd is declined** unless `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_F43` is set. Its transforms put about 27 fp32 ulps of each tile's sum of |products| into the result on a 3x3 c128 layer (about 1 for F(2,3); MIOpen's fp32 Winograd needs a 5.7 coefficient). The multipass F(4,3) instances stay. Deferred cost: F(4,3) was 25-36% faster than the best F(2,3) on fp32 mrcnn c256 layers on gfx1201 (200²: 19.1 vs 24.4 ms). That bounds the loss to about 300 ms call-weighted on gfx1201 NCHW (fp32 Winograd-best total 1120 ms), with smaller amounts on gfx1201 NHWC (210 ms total) and gfx1100 (147 ms NCHW, 155 ms NHWC). Follow-up once plans can carry a numerical note: tag F(4,3) instead of declining it, and try better points (0, ±1, ±1/2) for less error.
- **All conv engines here carry the `runtime_compilation` behavior note.**
- **NCHW 1x1 half-precision wrw trails MIOpen by design.** MIOpen's `GemmWrw1x1_stride1` zeroes dw, then runs one rocBLAS GEMM per image with beta=1 and fp16/bf16 output (HHS), so dw is rounded to half precision after every image. The BLAS engines accumulate the whole reduction in fp32 (HSS) and round once. rocprofv3 on gfx1201, fp16: deeplab c256 k1024 128^2 n16 is 16 GEMMs of about 150 µs (2.47 ms) for MIOpen against 4 fp32-output calls of about 1.05 ms plus a 0.11 ms cast (4.45 ms) for us; unet c64 k2 512^2 n16 is 1.27 ms against about 1.85 ms (16 GEMMs plus a 0.25 ms cast). We don't match it because it rounds N times. It could be added later as a separate lower-precision plan or engine carrying its own numerical note (like `WINOGRAD`), so callers can filter it out.
