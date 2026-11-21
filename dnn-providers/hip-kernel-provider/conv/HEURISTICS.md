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
| `CONV_IGEMM_FWD/BWD` (K1) | Dense and grouped implicit GEMM, the general workhorse; compute-bound shapes; backward data of "valid" small 3-D layers as one GEMM over dy's positions plus a column sum (col2im); NCHW via direct channels-first epilogue (`cf`) or staging; 1-D channels-first (N C L) read in place by the `cfx` configs (no activation staging) | fwd, bwd-data | gfx908-gfx950 (MFMA), gfx11/gfx12 (WMMA including native gfx1250), FMA everywhere | 3998 (2256 fwd, 1742 bwd; 1435 of them patch configs, 38 col2im, 340 `cfx`) |
| `CONV_IGEMM_FWD/BWD_TF32` (K1) | K1's TF32 configs (three-bf16 everywhere with bf16 matrix cores, xf32 on gfx942) for fp32 graphs. Numerical note `TENSOR_CORE`. | fwd, bwd-data | MFMA, WMMA including native gfx1250 | 101 (52 fwd, 49 bwd) |
| `CONV_IGEMM_NCHW_BWD` (K7) | NCHW bwd-data in place: strided pointwise, patchify, 1-D/vocoder ConvTranspose, dilated rows, 3-D stride 2. Weak on large compute-bound GEMMs (K1 wins there). | bwd-data | MFMA, WMMA including native gfx1250, FMA | 2531 |
| `CONV_IGEMM_NCHW_BWD_TF32` (K7) | K7's three-bf16 configs for fp32 graphs (RDNA, strided 3-D filters of 27+ taps). Numerical note `TENSOR_CORE`. | bwd-data | MFMA, WMMA including native gfx1250 | 112 |
| `CONV_DIRECT_FWD/BWD` (K2) | Stride-1 halo-tile direct conv, 3x3 / 5x5 / 7x7 / 1x7 / 3x3x3. The `rows` kind (hipconv-class filter-row reuse) is the large-channel dense path on gfx950 and gfx90a, 2-D and 3x3x3; on gfx90a its ping-pong tiles win from about 256 channels (16-bit) and at every channel count in fp32, while the halo tile keeps 96-192-channel 16-bit layers. | fwd, bwd-data (stride 1) | MFMA, WMMA including native gfx1250, FMA | 798 (399 fwd, 399 bwd) |
| `CONV_DIRECT_FWD/BWD_TF32` (K2) | K2's three-bf16 configs for fp32 graphs. Numerical note `TENSOR_CORE`. | fwd, bwd-data (stride 1) | MFMA, WMMA | 42 (21 fwd, 21 bwd) |
| `CONV_WINOGRAD_FWD/BWD` (K5) | 3x3 stride 1 (stride 2 by phases). `coop` is the cooperative WMMA/MFMA F(2,3), and F(3,2) for stride 2; the fused VALU and register variants target gfx906, grouped, and fp32 on RDNA (including 24-wide K blocks). | fwd, bwd-data | all | 288 (144 fwd, 144 bwd) |
| `CONV_WINOGRAD_FWD/BWD_BF16` (K5) | K5 for bf16 graphs: the matrix-core bf16 operands round the transformed tiles to bf16, past the convolution's own precision model. Numerical notes `WINOGRAD`, `WINOGRAD_TILE_4x4`. | fwd, bwd-data | all | 126 (63 fwd, 63 bwd) |
| `CONV_WINOGRAD_FWD/BWD_F43` (K5) | fp32 F(4,3) multipass: about 27 fp32 ulps of each tile's sum of \|products\| on a 3x3 c128 layer (about 1 for F(2,3)); 25-36% faster than the best F(2,3) on fp32 mrcnn c256 layers on gfx1201. Numerical notes `WINOGRAD`, `WINOGRAD_TILE_6x6`. | fwd, bwd-data (stride 1) | all | 32 (16 fwd, 16 bwd) |
| `CONV_WINOGRAD_WRW` (K5) | 3x3 stride-1 backward weights as F(3,2), F(3,4) and F(3,6): a GEMM per Winograd position over all tiles on MFMA/WMMA or an FMA thread tile, fixed-order split reduce | wrw (stride 1) | all | 84 |
| `CONV_WINOGRAD_ND_FWD` | FLOAT 2-D 3×3, unit stride, non-unit dilation: residue-subgrid F(2,3), exact-fp32 native GEMM and bounded spatial workspace; weights read anew each execution | fwd | gfx1250 | 4 |
| `CONV_FFT_FWD/BWD/WRW` | Large filters: dense 5x5 and up with many channels (fp32 especially; 16-bit from about 11x11), 1-D filters of 15+ taps, large-kernel depthwise (31x31: RepLKNet / SLaK class). Configs vary the transform size (`th` x `tw`: 16, 32, 64 square, 1 x 32 / 1 x 64 rows: the tile must hold the dilated filter; larger tiles for larger filters and images), the transform block (`pb` tiles; `tb` threads, the least that covers the row pairs or 256) and the product tile (`gn` 64 or 16 output channels, `tm` 4 or 8 tiles a thread). Numerical note FFT. | fwd, bwd-data, wrw (stride 1) | all (FMA; no matrix cores) | 198 fwd, 198 bwd, 132 wrw |
| `CONV_WRW` (K3) | Backward weights. Several loops: row ring, output rows (any stride), height-stride phased ring, gfx950 direct (hipconv `direct_wgrad` class), small-C rows for stems, and FMA rows for exact fp32 on RDNA. At one split, configs with `dws` 1 write dw directly with no reduce. A generic tier (`loop` 8) covers the filter widths, strides and dilations no specialized config has, and x/dy in any strides (see "Two tiers" below); native loop 11 also gathers admitted strided operands directly. | wrw | MFMA, WMMA including native gfx1250, FMA | 7240 |
| `CONV_WRW_TF32` (K3) | K3's three-bf16 configs for fp32 graphs. Native gfx1250 uses the existing operand-split, fp32-partials and ordered-reduce algorithm on K32 BF16 atoms. Numerical note `TENSOR_CORE`. | wrw | MFMA, WMMA including native gfx1250 (not gfx908) | 884 |
| `CONV_GROUP_FWD/BWD` (K4) | Small-group grouped conv (2 ≤ C/g ≤ 16, ResNeXt and RegNet): one 16x16 MMA serves 16/(C/g) groups through a block-diagonal filter. Wide groups (17 ≤ C/g ≤ 64) pad each group to 16-channel blocks. Row-streaming configs (`rs`, channels-last; small groups and groups of exactly 32 or 64 channels) stream each band's input rows once with the filter in registers, the fastest K4 form on channels-last graphs that are bandwidth-bound; the halo-staged and `cf` configs serve channels-first graphs. K1's `fuse` knob is its implicit-GEMM alternative. | fwd (s1, s2), bwd-data (s1; s2 by phases) | MFMA, WMMA; fp32 by FMA on RDNA | 3724 (1524 fwd, 2200 bwd; 1500 row-streaming) |
| `CONV_GROUP_FWD/BWD_TF32` (K4) | K4's three-bf16 configs for fp32 wide-group graphs (17-64 channels a group). Numerical note `TENSOR_CORE`. | fwd, bwd-data | MFMA, WMMA (not gfx908) | 264 (144 fwd, 120 bwd) |
| `CONV_GROUP_WRW` (K4) | Backward weights for small groups (2 ≤ C/g ≤ 16): the same block-diagonal 16x16 MMA, dy positions of a th x tw tile flattened into the atom's k-steps (any plane width), x and dy read in place; deterministic split reduce. Groups of 24 and 32 (`nbi` 2, MFMA only) as dense per-group block products. K3 (`CONV_WRW`) is its implicit-GEMM alternative; groups past 32 channels stay with K3. | wrw (any stride on rows and columns per config: s1, s2; any depth stride) | MFMA, WMMA; fp32 also by FMA (all archs, the only fp32 path on RDNA) | 3132 |
| `CONV_GROUP_VALU_FWD/BWD` | Grouped forward and backward data on vector ALU, the forward counterpart of `CONV_GROUP_VALU_WRW`: a wave owns `kb` output channels of a group, lanes along the output row, the filter as wave-uniform scalar loads. Groups of 2 to 16 channels where a per-group GEMM is mostly empty (ResNeXt / ResNeXt3D, small RegNet and StyleGAN2 groups) and 3-D grouped layers; wide groups (C/g ≥ 56) stay with Winograd. 3x3 and 3x3x3, any stride (backward data reads dy upsampled); NCHW and NHWC. | fwd, bwd-data | `valu` pack gfx906, gfx908 (180 fwd, 168 bwd); `mma` pack gfx90a-gfx950, gfx11, gfx12, gfx1250 (30 fwd, 30 bwd) | 408 |
| `CONV_GROUP_VALU_WRW` | Backward weights on vector ALU (MIOpen's `ConvAsmBwdWrW3x3` design in HIP, generalized): groups of 4 to 56 channels where a per-group GEMM is mostly empty, few-channel dense and pointwise layers, and most wrw on gfx906. 3x3 stride 1/2 and 5x5 (3-D: a tile per depth tap, any depth filter); NCHW and NHWC. | wrw | `valu` pack gfx906, gfx908 (293); `mma` pack gfx90a-gfx950, gfx11, gfx12, gfx1250 (212) | 505 |
| `CONV_STEM_BWD` | Bwd-data for stems with C ≤ 4 (7x7 s2, 3x3 s2, patchify, 3-D and space-to-depth geometry) | bwd-data | MFMA, WMMA including native gfx1250, FMA | 3034 |
| `CONV_SMALLC_FWD` | Forward stems with C ≤ 4 | fwd | MFMA, WMMA including native gfx1250, FMA | 1057 |
| `CONV_SMALLC_BWD` | Stride-1 bwd-data with K ≤ 4 (3x3 and 3x3x3 heads), the forward kernels run over dy | bwd-data | MFMA, WMMA including native gfx1250, FMA | 268 |
| `CONV_NARROW_FWD/BWD/WRW` | At most 8 channels on one side: output layers (C wide into K <= 8) forward and backward weights, stem input gradients (C <= 8) backward data and backward weights. The taps go on the GEMM's N axis; the wide tensor is read once. Forward / backward-data configs fix the filter, width stride, narrow-channel range (`kmin..ko`), reduction-channel tier (`cmin..cm`), depth streaming (`ds`), prefetch depth (`pd`) and tile; backward weights the side, filter, narrow-channel range and tile. | fwd, bwd-data, wrw | MFMA (gfx950 x32 atoms and its own fp32 tiles; gfx908-gfx942 16x16; gfx908's own bf16), WMMA (gfx11, gfx12), vector ALU everywhere | 8758 (2572 fwd, 2848 bwd, 3338 wrw) |
| `CONV_DEPTHWISE_MFMA_FWD/BWD` | Depthwise on matrix cores (gfx950 hipconv class), NHWC and NCHW | fwd, bwd-data (s1); fwd (s2) | gfx908-gfx950 | 640 |
| `CONV_DEPTHWISE_ROWS_FWD/BWD` | Depthwise by channel rows: 1-D/vocoder depthwise, dilated, strided ConvTranspose1d | fwd, bwd-data | all (VALU) | 222 |
| `CONV_DEPTHWISE_VALU_FWD/BWD` | General VALU depthwise, plus the strided bwd `phase` variant | fwd, bwd-data | all | 342 (162 fwd, 180 bwd) |
| `CONV_DEPTHWISE_VALU_WRW` | Depthwise backward weights (register strips; NHWC and NCHW) | wrw | all | 1140 |
| `CONV_NAIVE_FWD/BWD/WRW` | Coverage-first production kernels: serve graphs no faster engine serves, within a workload budget | all | all | 9 |
| `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` (hipblaslt-provider) | Explicit GEMM (im2col/col2im) | all | all | BLAS-internal |
| `ROCBLAS_CONV_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_ENGINE_DETERMINISTIC`, `ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC` (hipblaslt-provider) | The exact and TF32 explicit-GEMM engines, bitwise reproducible (rocBLAS atomics not allowed, no hipBLASLt workspace); the default engines report `NONDETERMINISTIC`. Cost against the default engines: `hipblaslt-provider/docs/engines/EXPLICIT_GEMM.md`. | all | all | BLAS-internal |
| `ROCBLAS_CONV_TF32_ENGINE`, `HIPBLASLT_CONV_TF32_ENGINE`; `ROCBLAS_CONV_RPR_ENGINE`, `HIPBLASLT_CONV_RPR_ENGINE` (hipblaslt-provider) | Explicit GEMM with TF32 (numerical note `TENSOR_CORE`), and fp16 1x1 backward weights with a per-image rounded reduction (`REDUCED_PRECISION_REDUCTION`) | all; wrw | all (TF32 rounds only where the library has a TF32 path: hipBLASLt gfx942 / gfx950, rocBLAS gfx942) | BLAS-internal |

The config counts are the full catalog of each engine: every dtype and arch pack. A graph admits only the configs whose gates accept it. For example, K7 admits about 28-50 per graph and `CONV_STEM_BWD` about 4-9 (at most 21).

### Two tiers

An engine's configs come in two tiers:
- **Specialized** configs fix the geometry at compile time (filter width, stride, dilation, sometimes filter height) and serve the hot shapes. There can be many of them.
- **Generic** configs take the geometry at runtime. There are few of them, and each is admitted only where no specialized config's class serves the graph. They exist to beat the BLAS/naive fallback on the long tail, not to compete with the specialized tier.

In `CONV_WRW` the generic tier is `loop` 8. It has a runtime filter, strides and dilation, `fr` taps per block, and reads x and dy through their strides, so padded or sliced activations are served too. Each generic config carries a `spec` string: the specialized classes the catalog serves on its architectures (`<atom_id><kind><layout>.<s>.<sw>.<dw>.<fr>`). The host declines the generic config wherever one of those classes serves the graph. A graph that a specialized matrix-core config serves therefore never sees the generic configs. The FMA loops (exact fp32) are not counted as specialized.

Measured against the BLAS path, 16-bit channels-last:
- gfx950: 1.5-1.9x faster on 2-D 5x5/7x7 and 3-D 1x5x5.
- gfx1201: 2.6-3.2x faster on the same shapes, and 1.7x on 1-D k3.
- 1-D k9: 0.69x on gfx950 and 0.96x on gfx1201, so BLAS still wins there. Heuristics should rank BLAS above the generic tier for it.
- gfx11 (gfx1100, gfx1101): R(2+1)D's 3x1x1 layers 1.7-4.0x, Inception's 5x5 1.2-1.75x, HiFi-GAN's 1x7 rows to one output channel 1.3-1.5x. On R(2+1)D's 14² and 7² maps and its wide 1x3x3 layers BLAS stays ahead (0.2-0.5x), and autotune keeps it there; heuristics should rank BLAS first for them.
- ASPP-class filters (at most 9 taps, dilation 6 or more), best generic config against the best BLAS engine on deeplab 3x3 d12-d36 and SSD's fp32 3x3 d6: gfx942 (MI325X) geomean 0.64x and gfx90a 0.73x (the generic tier wins only channels-first d24/d36, where BLAS is weakest), gfx1201 16-bit 0.8-3.3x (geomean 1.5x) but fp32 (FMA form) 0.4-0.7x, gfx1101 16-bit 0.9-12x and fp32 1.3-2.2x. The generic tier serves them on gfx11 and on gfx12's matrix atoms, and declines them on gfx942, gfx90a, gfx12's FMA form and the architectures not measured (gfx950, gfx908, gfx906).
- Narrow groups (fewer than 16 input and fewer than 16 output channels a group): the generic tier ran 14-20x behind `CONV_GROUP_WRW` at 4 channels a group and 30-5000x behind the vector-ALU engines on depthwise layers (gfx1101). It declines them.
- Stems (fewer than 8 input channels a group) and output rows of fewer than 4 positions: BLAS measured 2.5-30x faster on gfx1101 (ViT / ConvNeXt / R(2+1)D stems, Inception's 5x5 over a 5x5 map). The generic tier declines them.
- These three declines (narrow groups, stems, narrow rows) leave every graph they touch served on gfx1100, gfx1101, gfx1201, gfx942 and gfx90a (668 corpus wrw graphs; BLAS alone on 21-22 of them). In an A/B of 188 of them on gfx1201, gfx942 and gfx90a no graph's best time moved by more than 5%, apart from one gfx90a stem at 0.93x (one round). Where the generic tier was the fastest hipkernel engine before (stems and Inception's 5x5 over a 5x5 map), BLAS ran faster (4-42x on gfx1201).
- RDNA fp32 runs the generic tier's FMA form, because RDNA has no fp32 matrix core. On gfx1201, against BLAS, it is faster on 5x5 (1.6x), 1-D k9 (1.4x), 3x3 stride 3 (4.2x) and 5x5x5 (3.5x). It is slower on stems and patchify-class width strides (4.2-7.9x behind), so it declines width strides of 4 or more (and stems, as the whole tier does).

Native gfx1250 fp16/bf16 alternatives include narrow one-wave small-C WRW tiles,
longer pointwise ring chunks, and direct 3×3 tiles with twice as many output
channels per block. They retain existing defaults and matcher/resource gates.
Rank them from complete-operation timings: a wider chunk can lose on short rows
or underfilled grids, and a small-C tile is not a replacement for wide groups.

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
| `cf` | 1 writes channels-first output directly; K1 2 also reads bounded channels-first HALF pointwise input directly |
| `cfx` (K1) | Read 1-D channels-first activations in place (no staging); with `halo` h, the halo rows hold h input pixels per output row |
| `split_policy` (K1) | 0 retains automatic partitioning; 1 explicitly disables split reduction for the bounded native HALF backward-data candidate |
| `layout_id` (grouped VALU) | 0 NCHW, 1 NHWC; keeps both compiled layouts distinct when singleton extents make both applicable |
| `wdirect` | bwd-data reads the packed filter in place |
| `pack` | Taps × channels flattened in the reduction |
| `fuse` | F small groups run as one block-diagonal group |
| `halo` | Extra dy/x window beyond the tile (dilation and filter reach) |
| `bpu` / `split` | Split of the reduction over blocks: fp32 partials, then a fixed-order reduce. Deterministic, no atomics. |
| `dws` (K3) | At one split, 1 writes dw when channels are contiguous; bounded HALF loop-12 mode 2 also writes channels-first multi-tap filters directly |
| `loop` (K3), `kind` (K2), `variant` (K5, depthwise) | Which kernel body runs |

Engine-specific knobs (`taps`, `stride`, `flat`, `ring`, `phase`, `fold`, ...) are documented in the engine's `*Problem.hpp`.

**The hardest choice is often split-K.** The planner picks the split count from a device-unit model (`packs/DeviceUnits.hpp`: CUs on gfx9 and gfx1250, WGPs on legacy gfx10/gfx11/gfx1200/gfx1201) and fills whole dispatch rounds. Configs that differ only in `bpu` exist so autotune can override it.

## Timing data

**Per-config timings.** With autotune (`HIPDNN_FORCE_BENCHMARKING=1`), each ingestor engine times every admitted config on the first execute and keeps the fastest. Every candidate's time and its problem features are logged as a JSON record at INFO plugin log level (`BenchmarkPlan::logCandidateTiming`). That is the per-config training data for UHD heuristics. The BLAS engines tune internally and report only their best.

**Problem sets in use for measuring this group**, as hipDNN graph JSON:
- **Model-corpus subset:** 1998 shapes chosen from an internal 9471-shape real-model corpus to keep its range: every direction, dtype, layout, conv type and filter class. It covers 31 models.
- **Quick set:** 86 shapes. Every conv class, direction, dtype and layout.
- **Audio/video set:** 414 shapes (vocoders, video VAEs, conformer, segmentation).
- **Depthwise** and **pointwise** sets.

Reference results per GPU are each engine's autotuned best time per graph, against MIOpen in search mode (gfx950, gfx942, gfx90a, gfx1201, gfx1100/gfx1101, gfx908, gfx906). They come as per-graph CSVs: MIOpen time, our best time, best engine, and MIOpen's solver.

Ask the conv owners for the graph sets and the latest results; they are not in this repo.

## Things that bite

- **First-run cost.** With autotune, every admitted config compiles through HIPRTC on first use. Measured median cold compiles are 0.3-0.8 s per config on CDNA and 0.4-1.5 s on RDNA. The only configs over 10 s are K7's fp32 FMA stride-8 configs (10-20 s). Summed over its admitted configs, a graph's median first-run cost is about 30 s. The worst measured graph is a ResNeXt-50 grouped NCHW bwd-data graph, where K1 alone admits 91-131 configs: 134 s on gfx942 and 224 s on gfx950. Good OOTB selection means picking well without compiling the whole candidate list. The HIPRTC disk cache lives under `<cache>/hip-kernel-provider/hiprtc/<key>.co`. The key is a sha256 over the compiler identity, the source, every embedded include and every option (architecture included); see [the provider README](../README.md#hiprtc-code-object-cache). A pre-populated cache directory is how commonly needed instances can ship pre-compiled.
- **Catalogs are not final.** K5 `coop`, K1 core (`swz`, `dma`, `fuse`, `cf`), K3, K4 and the depthwise engines are under active tuning. Retrain heuristic models whenever a catalog changes.
- **Timing noise.** Jobs sharing a node can inflate times unevenly between engines; one shape read 0.85 sharing a node and 1.29 alone. Kernels under ~50 µs are noisy per shape. Measure isolated, and weigh by absolute time.
- **Configs that spill registers still win in places.** Some RDNA K3 ring tiles spill and are still the fastest. Don't prune on spills.
- **Lower-precision engines are offered by default.** The TF32 engines (`*_TF32`), the bf16 and F(4,3) Winograd engines and the BLAS reduced-precision-reduction engines serve by default, each reporting its numerical note. A caller that wants exact fp32 deselects `TENSOR_CORE` (and the Winograd notes) through the hipDNN API's cuDNN-compatible `deselect_numeric_notes`; heuristics trained for exact selection should drop those engines the same way.
- **Winograd workspace.** `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB` caps K5's workspace. `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES` overrides the last-level-cache size its planner assumes; HIP doesn't report MALL.
- **Winograd engines carry the `WINOGRAD` numerical note** with their tile note (`WINOGRAD_TILE_4x4` forward and backward data, `WINOGRAD_TILE_6x6` for F(4,3); backward weights mixes tiles and names none), so callers can opt out of them for numerical reasons.
- **bf16 Winograd** (`CONV_WINOGRAD_FWD/BWD_BF16`): the matrix-core bf16 operands round the transformed tiles to bf16, past the precision model, and MIOpen has no bf16 Winograd to match. On the full corpus in NCHW, bf16 Winograd is our fastest on 284 graphs on gfx1201 (503 ms call-weighted ahead of the next engine, of 4498 ms) and 73 on gfx1100 (440 ms). In NHWC it is fastest on none.
- **fp32 F(4,3) Winograd** (`CONV_WINOGRAD_FWD/BWD_F43`): its transforms put about 27 fp32 ulps of each tile's sum of |products| into the result on a 3x3 c128 layer (about 1 for F(2,3); MIOpen's fp32 Winograd needs a 5.7 coefficient). It is 25-36% faster than the best F(2,3) on fp32 mrcnn c256 layers on gfx1201 (200²: 19.1 vs 24.4 ms), about 300 ms call-weighted on gfx1201 NCHW (fp32 Winograd-best total 1120 ms), less on gfx1201 NHWC (210 ms total) and gfx1100 (147 ms NCHW, 155 ms NHWC). Better points (0, ±1, ±1/2) for less error are a follow-up.
- **All conv engines here carry the `runtime_compilation` behavior note.**
- **NCHW 1x1 half-precision wrw trails MIOpen by design.** MIOpen's `GemmWrw1x1_stride1` zeroes dw, then runs one rocBLAS GEMM per image with beta=1 and fp16/bf16 output (HHS), so dw is rounded to half precision after every image. The BLAS engines accumulate the whole reduction in fp32 (HSS) and round once. rocprofv3 on gfx1201, fp16: deeplab c256 k1024 128^2 n16 is 16 GEMMs of about 150 µs (2.47 ms) for MIOpen against 4 fp32-output calls of about 1.05 ms plus a 0.11 ms cast (4.45 ms) for us; unet c64 k2 512^2 n16 is 1.27 ms against about 1.85 ms (16 GEMMs plus a 0.25 ms cast). We don't match it because it rounds N times. The hipblaslt-provider's `ROCBLAS_CONV_RPR_ENGINE` and `HIPBLASLT_CONV_RPR_ENGINE` run that form and report `REDUCED_PRECISION_REDUCTION`, so callers can filter them out.
- **Native atoms are distinct contracts.** gfx1250 uses K32 for 16-bit WMMA and exact K4 for fp32, not the gfx1200/gfx1201 fragments. Native stem and WRW LDS limits also clamp to the device's per-block capacity; more LDS is not proof that a tile is faster.
- **WRW loop IDs are stable.** The existing packed-row loop 7 and double-buffered generic loop 8 remain. Native per-tap runtime GEMM is loop 11, flattened tap/channel GEMM is loop 12, small-K scalar is loop 9 and the explicit FLOAT/FP64 accuracy floor is loop 10. Distinct scheduling contracts remain separate candidates.
- **WRW numerical authority.** The independent validator uses compensated FP64 direct convolution at unchanged per-type tolerances; MIOpen disagreement is diagnostic. Loop 10 is an accuracy floor, not a general performance claim.
