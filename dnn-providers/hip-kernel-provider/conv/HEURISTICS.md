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
| `CONV_IGEMM_FWD/BWD` (K1) | Dense and grouped implicit GEMM, the general workhorse; compute-bound shapes; NCHW via direct channels-first epilogue (`cf`) or staging | fwd, bwd-data | gfx908-gfx950 (MFMA), gfx11/gfx12 (WMMA), FMA everywhere | 1628 |
| `CONV_IGEMM_NCHW_BWD` (K7) | NCHW bwd-data in place: strided pointwise, patchify, 1-D/vocoder ConvTranspose, dilated rows, 3-D stride 2. Weak on large compute-bound GEMMs (K1 wins there). | bwd-data | MFMA, WMMA, FMA | 1839 |
| `CONV_DIRECT_FWD/BWD` (K2) | Stride-1 halo-tile direct conv, 3x3 / 5x5 / 7x7 / 1x7 / 3x3x3. The `rows` kind is the gfx950 dense 3x3 path (hipconv-class). | fwd, bwd-data (stride 1) | MFMA, WMMA, FMA | 442 |
| `CONV_WINOGRAD_FWD/BWD` (K5) | 3x3 stride 1 (stride 2 by phases). `coop` is the cooperative WMMA/MFMA F(2,3); the fused VALU and register variants target gfx906 and grouped. | fwd, bwd-data | all | 266 |
| `CONV_WRW` (K3) | Backward weights. Several loops: row ring, output rows (any stride), height-stride phased ring, gfx950 direct (hipconv `direct_wgrad` class), small-C rows for stems, and FMA rows for exact fp32 on RDNA. | wrw | MFMA, WMMA, FMA | 2434 |
| `CONV_STEM_BWD` | Bwd-data for stems with C ≤ 4 (7x7 s2, 3x3 s2, patchify) | bwd-data | MFMA, WMMA, FMA | 2708 |
| `CONV_SMALLC_FWD` | Forward stems with C ≤ 4 | fwd | MFMA, WMMA, FMA | 597 |
| `CONV_DEPTHWISE_MFMA_FWD/BWD` | Depthwise on matrix cores (gfx950 hipconv class), NHWC and NCHW | fwd, bwd-data (s1); fwd (s2) | gfx908-gfx950 | 640 |
| `CONV_DEPTHWISE_ROWS_FWD/BWD` | Depthwise by channel rows: 1-D/vocoder depthwise, dilated, strided ConvTranspose1d | fwd, bwd-data | all (VALU) | 222 |
| `CONV_DEPTHWISE_VALU_FWD/BWD` | General VALU depthwise, plus the strided bwd `phase` variant | fwd, bwd-data | all | 168 |
| `CONV_DEPTHWISE_VALU_WRW` | Depthwise backward weights | wrw | all | 42 |
| `CONV_NAIVE_FWD/BWD/WRW` | Correctness floor: serves every graph | all | all | 9 |
| `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` (hipblaslt-provider) | Explicit GEMM (im2col/col2im) | all | all | BLAS-internal |

The config counts are the full catalog. A graph admits only the configs whose gates accept it. For example, K7 admits about 28-50 per graph and `CONV_STEM_BWD` about 8-24.

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

- **First-run cost.** With autotune, every admitted config compiles through HIPRTC on first use. Measured cold compiles are about 0.5-8 s per config, and no surveyed config is over 10 s. On a graph that admits 30-50 configs, that is a large first-run cost. Good OOTB selection means picking well without compiling the whole candidate list. The HIPRTC disk cache lives under `<cache>/hip-kernel-provider/hiprtc/<key>.co`. The key is a sha256 over the HIPRTC version, the source, every embedded include and every option (architecture included). A pre-populated cache directory is how commonly needed instances can ship pre-compiled.
- **Catalogs are still moving.** K5 `coop`, K1 core (`swz`, `dma`, `fuse`, `cf`), the depthwise engines, and a small-group grouped engine in progress all change week to week. Models trained now will need retraining.
- **Timing noise.** Jobs sharing a node can inflate times unevenly between engines; one shape read 0.85 sharing a node and 1.29 alone. Kernels under ~50 µs are noisy per shape. Measure isolated, and weigh by absolute time.
- **Configs that spill registers still win in places.** Some RDNA K3 ring tiles spill and are still the fastest. Don't prune on spills.
- **TF32.** Three-bf16 configs register only under `HIP_KERNEL_PROVIDER_CONV_TF32`. Leave it unset for normal fp32 selection.
- **Winograd workspace.** `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB` caps K5's workspace. `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES` overrides the last-level-cache size its planner assumes; HIP doesn't report MALL.
- **Winograd engines carry the `WINOGRAD` numerical note** in their engine descriptor, so callers can opt out of them for numerical reasons.
- **All conv engines here carry the `runtime_compilation` behavior note.**
