<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_WINOGRAD (K5) — Winograd 3x3 convolution: fused, cooperative and multipass, all three directions

> Engine ids: `hipkernel:CONV_WINOGRAD_FWD`, `hipkernel:CONV_WINOGRAD_BWD`, `hipkernel:CONV_WINOGRAD_WRW` · Generator: `conv/generators/gen_winograd.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/` (`ConvWinograd.cpp`, `ConvWinogradCore.h`, `…Fused.h`, `…Sreg.h`, `…Wmma.h`, `…WmmaReg.h`, `…WmmaCoop.h`, `…Wrw.h`, `…Args.h`) · Pack: `packs/ConvWinogradNative.cpp`, `packs/ConvWinogradProblem.hpp`

## At a glance
- **What:** Winograd minimal filtering for 3x3 filters.
  - Forward and backward data: F(2,3) for every dtype; F(4,3) for fp32 multipass only, in the `CONV_WINOGRAD_FWD/BWD_F43` sibling engines; F(3,2) on the four 2x2 phase sub-filters for stride 2.
  - Backward weights: F(3,2), F(3,4) and F(3,6), by duality with F(r,3).
- **Covers:**
  - fwd/bwd: 2-D 3x3, stride 1 or 2 (the same on both axes), dilation 1, groups, any padding, both convolution modes, any layout (staged when a variant does not read it). fp32 and fp16; bf16 in the `CONV_WINOGRAD_FWD/BWD_BF16` sibling engines.
  - wrw: stride 1, fp32 / fp16 / bf16 under per-dtype accuracy rules.
  - No 1-D or 3-D, no other filter sizes, no TF32 form.
- **Variants:**
  - `multipass`: transforms around a batched GEMM, any arch, NCHW and NHWC;
  - `fused`: VALU, every arch, `v_dot2_f32_f16` for fp16;
  - `wmma` and `wreg`: register-built WMMA operands, gfx11/gfx12 and native gfx1250 16-bit atoms;
  - `sreg`: register VALU, wave64 gfx9;
  - `coop`: cooperative WMMA (gfx11/gfx12 and gfx1250) or MFMA (gfx908+); native gfx1250 uses K32 chunks and the four-bin F(2,3) form, while older RDNA retains its 16-bin layout;
  - `wrw`: F(3,r) GEMM per Winograd position.
- **Arches:** all. gfx1250 adds exact-fp32 K4 WMMA to multipass GEMM; other fp32 paths retain VALU or their existing backward-weights matrix atoms.
- **Grids:** persistent variants size their grid by occupancy: units x blocks that are actually resident (registers and LDS). A second, partly empty round measured 15-26% slower.
- **Wins when:**
  - NCHW fp16 3x3 on CDNA2 / CDNA1 (gfx90a 1.30x fwd / 1.50x bwd on its graphs), against CK and older implicit GEMMs;
  - fp32 3x3 wrw on detection backbones: Mask R-CNN c256 200², 0.52-0.68x of MIOpen's time on CDNA.
- **Loses when:**
  - RDNA fp16 small-C against `ConvWinoFury` and `ConvWinoRage`: on gfx1101, kernel against kernel on Fury's graphs, Fury is up to 1.9x faster (backward data with 32 dy channels) and 1.2x faster on the geometric mean;
  - fp32 Mask R-CNN 200² fwd/bwd against `ConvBinWinogradRxSf2x3g1` (1.13-1.24x), F(2,3) only;
  - gfx906 fwd/bwd against MIOpen's asm Winograd (about 1.55x behind, weighted).
  - NHWC 3x3 fp16 backward weights and backward data against MIOpen's transpose-wrapped asm (`Transposed*`); other engines serve those graphs.
  - Kernel against kernel, K5 trails MIOpen's Winograd on gfx906 and RDNA (0.55-0.70x) and is near parity on gfx90a/gfx908 (0.96-0.98x); it leads on gfx950/gfx942 (1.03-1.32x) except against Rage (0.93x). See "Next steps".
- **Precision:** every engine reports the `WINOGRAD` numerical note, with `WINOGRAD_TILE_4x4` on the fwd/bwd and BF16 engines and `WINOGRAD_TILE_6x6` on the F43 engines (wrw mixes F(3,2), F(3,4) and F(3,6) and names no tile). Transforms are fp32; 16-bit operands are rounded once after their transform, except the input transform of `f16math` configs. Accumulation is fp32. An output whose window lies wholly in padding is exactly 0. Deterministic.
- **Instances:** current per-engine counts are in [HEURISTICS.md](../../HEURISTICS.md). The BF16 and F43 siblings remain separate numerical-note engines.
- **vs MIOpen** (usage-weighted speedup on the graphs where it is our best engine, search mode), fwd / bwd / wrw:

  | GPU | fwd | bwd | wrw |
  |---|---|---|---|
  | gfx950 | 1.07x | 1.10x | 1.86x |
  | gfx942 | 1.02x | 1.01x | 1.67x |
  | gfx90a | 1.30x | 1.50x | 1.00x |
  | gfx1201 | 0.87x | 0.83x | 1.11x |
  | gfx1100 | 0.83x | 0.81x | 1.28x |

## Why it exists
- 3x3 stride-1 layers are the bulk of CNN and UNet FLOPs. F(2,3) cuts multiplies 2.25x, and on VALU-bound or fp32 paths (gfx906, RDNA fp32, NCHW fp16 on CDNA) that beats any direct or GEMM form.
- MIOpen's fast Winograd kernels are binary-only assembly (no readable source): `ConvBinWinograd*`, Fury and Rage. They are NCHW-only and fp16- or fp32-only per family, with no bf16. K5 is one readable HIP source with runtime shapes for every arch, layout and dtype.
- Backward weights by Winograd (`CONV_WINOGRAD_WRW`) closes the RDNA fp32 Mask R-CNN / RetinaNet wrw gap against `ConvBinWinogradRxSf3x2`. It is a 4x-5x multiply reduction where `CONV_WRW` runs exact FMA.

## How it works
**Forward form.** Every launch is a forward problem: in → out, a 3x3 correlation with top/left padding.
- Backward data: in = dy, out = dx, padding 2 - p. The filter transform rotates the filter 180 degrees and swaps its K/C roles in its address walk, and reflects it for true convolution.
- Stride 2 runs as F(3,2) on the phases. The four 2x2 phase sub-filters form one stride-1 convolution over 4x the channels. Forward gathers phase (ph, pw) of x as virtual channel 4c + 2ph + pw while loading; backward data scatters virtual output channel 4c + phase into dx while storing.

**F(m,3) per tile:** Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A. Each of the (m+2)² Winograd positions ("bins") is an independent GEMM over input channels.
- **multipass:**
  - filter transform U[G][P][K][C];
  - input transform V[G][P][C][T];
  - batched GEMM M = U·V per (group, position), fp32 accumulate;
  - output transform to y.
  - The batch is sliced to keep V and M under the workspace limit: 2 GiB by default, `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB` overrides. An `llc` config slices instead to fit the RDNA last-level cache (MALL size table per chip; `…_LLC_BYTES` overrides).
- **fused / persistent variants:** one kernel loops over items (group, BK output channels, BT tiles). Per input-channel chunk it loads the patches, applies Bᵀ d B, multiplies against the pre-transformed U, and applies Aᵀ M A in the epilogue straight to y. A small pre-pass writes U in the layout the variant reads.

**Backward weights F(3,r)** (`wrw`): dW = Gᵀ [ Σ_tiles (A g Aᵀ)_dy ⊙ (Bᵀ d B)_x ] G, using F(r,3)'s matrices.
- x patches go through F(r,3)'s data transform, and r x r dy tiles through its transposed output transform.
- A GEMM per position runs over all tiles, with both operands tile-contiguous, split over tiles into fp32 partials that are summed in launch order.
- Then dW = Gᵀ M G, with the true-convolution reflection; taps whose x window is all padding are exactly 0.
- r = 2, 4, 6 give 2.25x, 4x and 5.06x fewer multiplies.

## Implementation design
- **fused (VALU, every arch):**
  - The next chunk's patches and U are loaded during the current Hadamard. Bᵀ d B goes to double-buffered LDS, one barrier per chunk.
  - The per-bin Hadamard is a register-blocked BK x BT GEMM: fp32 FMA, or `v_dot2_f32_f16` on fp16 channel pairs.
  - `pf 1` prefetches two chunks ahead: 5-13% on RDNA fp32.
  - RDNA has an 8x16 micro-tile that reads 0.75 LDS bytes per FMA instead of 1, because the Hadamard is LDS-bound there (Mask R-CNN c256 200²: 14% faster on gfx1201, and about 23% faster on gfx1100 together with depth-2 prefetch).
  - 16- and 24-wide K blocks serve small groups and K = 24 / 48 layers.
- **coop (cooperative matrix core):**
  - The whole block transforms each 16-channel input chunk once into LDS, so one transformed value feeds all of the block's output channels.
  - Wave w owns transform row w % 4 (4 bins) and a K slice. A fragments (U) come from a fragment-major pre-pass buffer, one contiguous 512-byte run per fragment, prefetched a chunk ahead.
  - Epilogue: the wave folds its 4 bins to 2 values in registers, writes them to LDS, then applies Aᵀ over the rows.
  - RDNA and CDNA share one body over the atom traits (wave32 WMMA, wave64 MFMA 16x16x16).
  - Single input buffer (`sb 1`, CDNA): one V buffer instead of two, so the next chunk's transform waits for a barrier after the current chunk's MFMAs. Half the LDS: 2 x 2-tile blocks take 16 KiB, so MI325X's 64 KiB holds 3 of them (registers bound it there) instead of 2. Measured on MI325X: HRNet c32 56² 1.07x, Inception c32 k32 149² 1.09-1.10x, realesrgan c192 k64 128² backward data 1.06x. gfx950's 160 KiB already holds 3 blocks, so it gains nothing there.
- **16-bin layout** (`bins 16`, RDNA F(2,3)):
  - A wave owns all 16 bins of 16 channels x 16 tiles, so the output transform runs in registers: no LDS round trip and no epilogue barriers.
  - It wins where the output side dominates: backward data with few dy channels. Measured on UNet c128 k64 512² backward data against the 4-bin layout: 7.76 against 9.07 ms on gfx1201, 8.66 against 9.83 ms on gfx1100.
  - Its grid counts on 6 waves a SIMD on the 1536-register RDNA parts, 4 elsewhere. On gfx11 the kernel is built for exactly that, 240 VGPRs: at 242 a SIMD holds 5, and the grid's last blocks run a second round (gfx1101 UNet c128 k64 512² backward data with 8 K waves: 16.8 against 12.3 ms). The gfx12 build is not held to it.
- **wreg / sreg:** register-operand variants.
  - In `wreg` (gfx11/gfx12), each wave owns one transform row and builds its WMMA B fragments from the loaded input rows in registers: no LDS for V, and no barrier in the channel loop. gfx11's half-waves exchange channels with `v_permlanex16`.
  - In `sreg` (wave64 gfx9), lane = tile, and the transformed filter is a wave-uniform operand through the scalar cache.
- **Grids and order.** Fused / sreg use 2 blocks per dispatch unit. Coop uses about 1024 resident threads per unit, capped on CDNA by a shared occupancy model (`k5CoopCdnaWpe`, `k5CoopLdsBytes`, which counts the single input buffer) for registers and LDS (160 KiB a CU on gfx950, 64 KiB on earlier CDNA). Items run K blocks fastest, or tile blocks fastest on RDNA when a group's U is ≥ 4 MiB.
- **Codegen.**
  - Branch-free 32-bit transform offsets with an out-of-range sentinel for padding: the fused input slice is kept under about 1 GiB.
  - Chunk loads use the chunk as a scalar offset, so a load needs no VALU.
  - The fp16 first chunk is peeled so MFMAs start from zero.
  - A `readfirstlane` on the U offset avoids waterfall loops.
  - A V swizzle spreads LDS banks: it cuts CDNA conflict cycles 2.7x on gfx950 UNet. On gfx11, whose B fragment is a tile's 8 dwords read as two 4-dword halves, the halves swap on tile bit 2 xor bit 3: about 10x fewer conflict cycles on gfx1101 ESRGAN c192 k64 backward data.
  - On gfx11 the cooperative kernel splits tile indices into column, row and image by multiply-high reciprocals set once per launch, instead of runtime divisions in every item's job setup and epilogue.
- **Gates.**
  - Problem gates, everything the naive parser gates, plus: 2-D; 3x3 filter; dilation 1; equal stride 1 or 2 (wrw 1); n, c·stride², k and groups within 32 bits; extents and padding within 32 bits.
  - fwd/bwd bf16 only with the switch.
  - Per config: dtype and direction equal to the graph's; atom on the device; WMMA / coop not fp32; sreg not bf16; `f16math` only for fp16; coop bf16 MFMA from gfx90a; per-chip coop configs (`gfx942`: gfx940-942; `gfx950`) only on their chip; `sb` only on the CDNA transform-row layout; F(3,2) exactly for stride 2; F(4,3) only fp32; a device reporting no units declined for persistent variants.
  - wrw per-dtype rules (below); every plan within launch, buffer and item-count limits.
  - On gfx1250 only, extent-one activation axes are classified jointly: choose NCHW
    when both x and y match it, otherwise NHWC when both match that order, otherwise
    stage the mixed/other layouts. Other architectures retain their individual
    NCHW-first classification and existing staging decisions.
- **wrw.**
  - x and dy are read in place when packed channels-first. Packed channels-last operands go through 64 x 64 LDS transposes, which measured 3-23% faster than transforming in place on gfx906 / gfx90a UNet; others are copied.
  - GEMM on MFMA (gfx9 16x16, gfx950 16x16x32), WMMA, or FMA thread tiles.
  - Tile splits are chosen for about 2048 threads per unit; batch slices stay under the workspace limit.
- **Large tensors.** Fused slices are capped so one slice's input is under about 1 GiB. Multipass and wrw slice the batch under the workspace limit. A single image past those limits is declined, and other engines serve it.
- **Catalog.** Variants x tiles per dtype and arch family (coop: `gfx11`, `gfx12`, `gfx9` for every MFMA part, and per-chip `gfx942` (gfx940-942) and `gfx950`). The knobs are `m`, `llc`, the GEMM tile, `bk`, `bt`, `ccv`, `tm`, `tn`, `waves`, `bins`, `f16math`, `pf` and `sb`. The per-chip fp16 configs add, on gfx942, cross-item prefetch on 2 x 4 and 4 x 2 tiles and the single input buffer on 2 x 2, 4 x 2 and 2 x 4 tiles; on gfx950, prefetch on 4 x 2 tiles (prefetch on 2 x 4 tiles ran large-C UNet layers 4-5% slower there). Configs ship where they win measured layers (realesrgan, UNet, Mask R-CNN, HRNet, Inception, IGM).

## Fusion

The forward engines (`CONV_WINOGRAD_FWD`, `_FWD_BF16`, `_FWD_F43`) also claim a convolution followed
by a chain of pointwise nodes, under the same engine ids and catalog configs: scale and bias (per
channel, a device scalar or a pass-by-value scalar), relu / clipped relu / clamp / leaky relu, gelu,
gelu-tanh, swish / silu, sigmoid, tanh, elu, abs and softplus. The chain is generated into the
kernel that writes y (`ConvEpilogue.h`) and runs on the fp32 result before its single rounding:

| Variant | Where the epilogue runs |
|---|---|
| multipass | the output-transform pass (`ConvWinogradOutput`), both layouts |
| fused VALU, register VALU | the in-kernel output transform |
| WMMA, register WMMA | the in-kernel output transform |
| cooperative (F(2,3), F(3,2) by phases, 16-bin layout) | the in-kernel epilogue, per stored pair |

An output whose 3x3 window lies wholly in the padding is exactly 0 from the convolution and gets the
epilogue of 0, like every other output. The filter, input and GEMM passes and the staging copies
take no epilogue. The generated code depends on the chain's pattern only; values and operand
pointers are a second kernel argument. With no epilogue the kernels compile to the same code as
without the hook. Backward data and backward weights decline fused graphs.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: gfx950 (both layouts) relu 1.04x, gelu 1.07x, silu 1.05x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.18x on gfx950 (usage-weighted; the conv README's fusion table).

## Strengths and weaknesses
Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. W/L counts our-best graphs that are more than 5% faster or slower. Losses are counted against MIOpen's best, which is often its own Winograd.

| GPU | fwd graphs (W/L) | bwd graphs (W/L) | wrw graphs (W/L) |
|---|---|---|---|
| gfx950 | 44 (20/14) | 48 (25/14) | 10 (6/2) |
| gfx942 (MI325X) | 61 (26/26) | 63 (27/32) | 9 (5/3) |
| gfx90a | 57 (28/20) | 52 (38/11) | 15 (5/9) |
| gfx908 | 49 (19/22) | 41 (28/8) | 43 (22/11) |
| gfx906 | 77 (0/76) | 84 (1/81) | 76 (57/9) |
| gfx1201 | 102 (34/61) | 87 (31/52) | 14 (5/7) |
| gfx1100 | 95 (35/46) | 77 (11/57) | 10 (4/5) |

- **Strong:**
  - NCHW fp16 on gfx90a / gfx908, against CK and V4R4/V4R5 implicit GEMM: realesrgan c192 fwd 0.45x and bwd 0.54x of MIOpen's time.
  - fp32 wrw on detection layers.
  - gfx906 wrw, where MIOpen's universal GEMM is very slow.
- **Weak:**
  - RDNA fp16 small-C (c32-c192, k32-k64) against Rage / Fury: the cost is VALU. On gfx1101 backward data with 32 dy channels, K5 issues 2-3x Fury's VALU per WMMA, in the fp32 output transform of each item's 16 bins and the per-item job setup.
  - RDNA fp32 large layers, where F(2,3) runs (F(4,3) is the F43 engines').
  - gfx906 fused, where the Hadamard runs near peak but staging, loads and barriers cost as much again.
  - CDNA small-C against Rage: Inception c32 149² trails by 1.6-1.75x, realesrgan c160 k32 backward data by 1.33-1.38x, HRNet c32 56² by 1.18x (gfx950) and 1.27x (MI325X).

## Dilated Winograd

`hipkernel:CONV_WINOGRAD_ND_FWD` is a separate gfx1250 engine for FLOAT 2-D
3x3, unit-stride forward convolution with non-unit dilation on at least one
axis. Both convolution modes, groups, asymmetric padding and safe positive
strided views are supported. It does not add backward paths, 16-bit storage,
rank-3 convolution, unit-dilation candidates or fused pointwise graphs.

Each output axis is split into dilation residues. F(2,3) operates inside those
subgrids, without inserted filter zeros or materialized space-to-batch copies.
Filter, input, GEMM and output transforms use fp32; GEMM uses native exact-fp32
16x16x4 WMMA. Filter U and spatially chunked V/M buffers share a 256 MiB cap.
The filter is transformed from its current binding on every execution.

Four candidates combine `gemm_bm` 64/128 and `chunk_tiles` 256/4096;
`gemm_bn` 64 and `gemm_bk` 16 are fixed. The engine reports `WINOGRAD` and
`WINOGRAD_TILE_4x4`, with deterministic fixed-order sums and no atomics.
Its generator is `gen_winograd_nd.py`; kernels and descriptors live in
`descriptors/conv/winograd_nd/`, and matching/preparation in `ConvWinogradNd*`.
`TestGpuConvWinogradNd.cpp` checks independent FP64 direct convolution,
reduction/output-tile tails, chunk boundaries, rebinding and bitwise relaunches.

## Numerical precision
- **Error model.** Winograd's rounding error scales with the magnitude of the whole tile it shares, Σ|products| over the neighbourhood, not with the output alone. An infinity in a tile poisons its neighbours (inf x structural zero gives NaN). Both are what the `WINOGRAD` note declares; the non-finite behaviour is documented, not fixed.
- **Accumulation:** fp32 everywhere. Transforms are fp32, and 16-bit operands are rounded once after their transform. `f16math` configs (fp16 `wreg`, `coop`, `sreg`) do the input transform in packed fp16, two roundings, as MIOpen's asm does. MIOpen's Fury accumulates in fp16 on gfx11; K5 never does.
- **Test tolerances** (relative to the largest Σ|products| of the neighbourhood): fp32 1e-5 for F(2,3) and 5e-5 for F(4,3), fp16 2e-3, bf16 1.6e-2.
- **wrw tolerances:** the dynamic-tolerance model against an fp64-accumulated reference. fp16 up to 12 eps·σ; bf16 at the same absolute level; fp32 with the base summation allowance (c = 4).
- **wrw admission rules** keep each dtype inside its band:
  - fp16: F(3,2) on fp16 operands (matrix cores, or `v_dot2` FMA), and F(3,4) / F(3,6) on fp32 operands.
  - bf16: fp32 operands only, because bf16 operands measured past output rounding.
  - fp32: F(3,2), plus F(3,4) from 512 products per dW element. F(3,6) is not admitted; it measured past MIOpen's Winograd error.
  - F(3,4) uses points 0, ±1, 1/2, -2: about 3.7x less fp32 summation error than Lavin's set.
- **Lower-accuracy forms as sibling engines** (offered by default; a caller excludes them through the hipDNN API's cuDNN-compatible `deselect_numeric_notes`):
  - **bf16 fwd/bwd** (`CONV_WINOGRAD_FWD/BWD_BF16`; `WINOGRAD`, `WINOGRAD_TILE_4x4`): matrix-core operands round the transformed tiles to bf16, about a third of bf16's unit roundoff of each tile's Σ|products|. That is past the convolution's precision model, and MIOpen has no bf16 Winograd. On the full corpus in NCHW it is our fastest on 284 graphs on gfx1201 (503 ms call-weighted ahead of the next engine, of 4,498 ms) and 73 on gfx1100 (440 ms); in NHWC, fastest on none.
  - **fp32 F(4,3)** (`CONV_WINOGRAD_FWD/BWD_F43`; `WINOGRAD`, `WINOGRAD_TILE_6x6`): transforms with entries up to 8 and 1/24 put about 27 fp32 ulps of Σ|products| into a c128 layer, against about 1 for F(2,3) and the 5.7 coefficient MIOpen's fp32 Winograd needs. F(4,3) is 25-36% faster on fp32 Mask R-CNN c256 on gfx1201 (200²: 19.1 against 24.4 ms), about 300 ms call-weighted on gfx1201 NCHW.
  - Notes are per engine, so these configs are engines of their own; the exact-tile engines list none of them.
- **TF32:** none. A three-bf16 multipass form measured 0.21-0.58x of the exact configs. A paying form needs the split inside the fused cooperative kernel.
- **Determinism:** fixed-order sums, no atomics. Every config is relaunched over a re-filled workspace and must reproduce its bits.

## Comparison with MIOpen
- **`ConvBinWinograd3x3U`, `ConvBinWinogradRxS`, `ConvBinWinoRxS<2,3>/<3,2>`, `…RxSf2x3g1`:** binary asm for gfx9 / gfx10 / gfx11 / gfx12, fp32 and fp16, NCHW. They also serve arbitrary RxS by decomposing the filter into 3x3 pieces, and stride 2 by decimation. They win gfx906 fwd/bwd and RDNA fp32 Mask R-CNN 200² (1.13-1.24x).
- **`ConvWinoFuryRxS` (gfx11 / gfx12) and `ConvWinoRageRxS` (gfx942 / gfx950 / gfx12):** fp16 only, the WMMA / MFMA fused kernels. Fury builds B fragments in registers; K5's `wreg` and 16-bin `coop` take that idea. They lead on small-C fp16: on gfx1101, Fury's own graphs run 1.2x faster under Fury than under K5's best config (geometric mean, kernel against kernel; 0.83-1.9x per graph); 1.2-1.4x on most CDNA graphs, and up to 1.7x on Inception / EfficientNet.
- **`ConvMPBidirectWinograd`** (multipass, disabled by default) and **`ConvWinograd3x3MultipassWrW<3-x>`**: readable multipass sources. K5's multipass covers the same ground with its own GEMM and both layouts. Our wrw beats MIOpen's multipass wrw and `GemmWrwUniversal` on fp32 detection layers.
- MIOpen wraps NHWC in transposes (`Transposed*`) and has no bf16 Winograd.

## Prior art considered
- **MIOpen's asm Winograd families:** studied through solver contracts, perf models and instruction mixes. The kernels have no comments or labels.
  - Taken: F(3,2) by stride phases.
  - Taken: Fury's register-built B operands, and its keep-U-resident idea.
  - Taken: packed-fp16 input transforms, as an opt-in `f16math` knob.
- **Public Winograd literature:** Lavin and Gray's F(m,r) construction, and the standard interpolation-point sets. The F(3,4) points were chosen by an fp32 error simulation.
- **K1:** atoms, element conversion and buffer loads are shared.

No code was copied.

## Next steps and deferred ideas

**Remaining gap to MIOpen's asm Winograd, kernel against kernel.**
- **Graphs:** 3x3 graphs of the 1,998-shape corpus subset where MIOpen's exhaustive-search winner is a Winograd solver.
- **MIOpen time:** the MIOpen run that executed that solver: whichever of its default and deterministic engines times within 25% of the solver's Find time. Graphs where neither matches are left out.
- **K5 time:** its best config.
- **Ratio:** usage-weighted MIOpen time ÷ K5 time, so below 1 means K5 is slower. The BF16 and F43 engines are not included.

| GPU | K5 vs MIOpen Winograd | Graphs (K5 faster / slower by >5%) | K5 excess (all / without `Transposed*`) |
|---|---|---|---|
| gfx950 | 1.03x | 159 (66 / 69) | 16 / 9 ms |
| gfx942 (MI325X) | 1.32x | 213 (88 / 100) | 34 / 14 ms |
| gfx90a | 0.96x | 88 (25 / 50) | 50 / 11 ms |
| gfx908 | 0.98x | 148 (61 / 71) | 155 / 78 ms |
| gfx906 | 0.56x | 286 (29 / 249) | 3405 / 742 ms |
| gfx1100 | 0.55x | 299 (87 / 175) | 1803 / 322 ms |
| gfx1201 | 0.70x | 267 (77 / 167) | 412 / 261 ms |

- By solver, K5 is faster than:
  - `ConvBinWinogradRxSf3x2` on every arch except gfx906 (1.01-1.44x);
  - `…RxSf2x3` and `…RxSf2x3g1` on the CDNA arches except gfx90a's `RxSf2x3g1` (1.10-2.36x).
- K5 trails:
  - `ConvWinoRageRxS` at 0.72-0.93x;
  - `ConvWinoFuryRxS` at 0.39-0.71x;
  - `…RxSf2x3` / `…RxSf2x3g1` on RDNA at 0.62-1.00x;
  - `ConvBinWinograd3x3U` (gfx908) at 0.77x;
  - every gfx906 solver at 0.59-0.77x.
- Users see little of this: on these graphs autotune picks other engines where K5 is slow. Against the best of all engines, the same graphs run 1.14-1.69x faster than MIOpen on gfx950, gfx942, gfx1100 and gfx1201, 1.05x on gfx908, 1.41x on gfx906, and 0.93x on gfx90a.
- None of the levers below needs assembly.

1. **Channels-last 3x3 Winograd, mostly fp16 backward weights and backward data.** MIOpen serves these graphs with its NCHW asm wrapped in layout transposes (`Transposed*`), and most of K5's excess sits here (gfx906 2663 ms, gfx1100 1481 ms, gfx908 77 ms, gfx90a 39 ms). Give the wrw and bwd variants a fast NHWC path, or at least a transpose-wrapped one, so they don't run the slow configs.
2. **RDNA fp16 small-C against Fury / Rage.**
   - **Current gap:** kernel against kernel on gfx1101, on the graphs Fury wins on gfx1100 (full size), K5's best config runs at 0.84x of Fury on the geometric mean. That ranges from 0.53x (ESRGAN c160 k32 backward data) to 1.20x (HRNet c64 28²). gfx1201 against Rage and Fury is not measured at the current code.
   - **What remains is VALU:** 2-3x Fury's per WMMA on backward data with 32 dy channels (ESRGAN c160 k32, HRNet c256 k32). It comes from the fp32 output transform of each item's 16 bins and the per-item job setup. Fury transforms its output in packed fp16, which K5's precision rules out. Candidates: more tiles per item for small C without more accumulators, and cheaper job setup.
   - **gfx12:** apply the gfx11 changes there too (the 16-bin occupancy build, the reciprocal tile split, the bit 2 xor bit 3 swizzle) once gfx1201 can be measured.
   - **Measured on gfx1101 without gain:**
     - cross-item input prefetch;
     - a predicate-free interior epilogue for the 16-bin layout (it spills);
     - 16-bin configs with 2 K waves (they spill);
     - dropping the per-bin scheduling barrier.
3. **gfx942 / gfx950 against Rage** (0.93x on both; 7 / 6 ms excess, mostly fp16 small-C backward data: realesrgan c160 k32, HRNet c32, Inception c32 149²).
   - What is left is per-item overhead at low occupancy. Profiled on MI325X (HRNet c32 56² forward), K5 issues about 1.2x Rage's VALU, and Rage keeps 12 waves a CU against K5's 12 with the single input buffer (8 with two). Most of an item is load latency, transform VALU and the epilogue's LDS round trip with 2 barriers per 16-channel slab, not MFMA (about 3% busy).
   - Measured and not taken:
     - An in-kernel filter transform (U built per block in registers, filter staged through LDS) costs more than the 3-4 µs filter launch it removes.
     - A 32-bank V swizzle for gfx942 removes the modelled bank conflicts but times the same.
     - 4 waves a SIMD (128 VGPRs) spills.
     - Multiply-shift division for the tile index split is neutral.
   - Next: a cheaper epilogue (fewer barriers per item, Z in fewer passes), 16x16x32 MFMA with 32-channel chunks on gfx950, and transform VALU.
   - Autotune ranks some coop configs by a tuning time that differs from their steady-state time by up to 10% (e.g. 2 x 4 tiles without prefetch). It then picks the prefetch sibling, 3-5% slower in steady state, on a few large-C layers.
4. **gfx908 against `ConvBinWinograd3x3U`** (0.77x).
5. **gfx906 fused** (0.56-0.77x): reader/writer wave specialization, or a persistent design with U in LDS, aimed at near-peak fp32 VALU.
6. **gfx90a / gfx908 multipass backward weights** against `ConvWinograd3x3MultipassWrW` (0.66-0.93x on its graphs).
7. **F(4,3) accuracy:** try points 0, ±1, ±1/2 for less error.
8. **TF32:** split inside the cooperative kernel, not in multipass.
9. **Other filter sizes by 3x3 decomposition**, as MIOpen's RxS does: low value. On the non-3x3 graphs where MIOpen picks Winograd (1x1, 5x5, 7x7, 1xN), `CONV_DIRECT`, `CONV_IGEMM` and `CONV_FFT` already run 1.37-2.56x faster than it on every arch except gfx90a (0.63x, 4 ms over 14 graphs).

Gate for each item: parity with the MIOpen solver on the graphs it wins, and no regression elsewhere.

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestConvWinogradProblem.cpp` (host):
  - **Gates:** one refusal per field; direction exclusivity; bf16 fwd/bwd graphs and F(4,3) fp32 configs admitted by the matches (the catalogs put them in their sibling engines); stride-2 phase gather/scatter and 16 positions.
  - **Variant placement:** each variant on its arch family and dtypes; the 16-bin layout on RDNA stride 1; coop MFMA from gfx908 and bf16 from gfx90a; per-chip coop configs only on their chip and the single input buffer only on CDNA; fused stride-2 chunking; inconsistent register-variant fields refused.
  - **Plans:** LLC slicing and workspace; occupancy-sized grids (16-bin on RDNA, CDNA, the single input buffer); item order for a large U; fused slices clear of the out-of-range offset.
  - **wrw:** error-band admission (FMA every dtype, fp32 F(3,4) from 512 products, half-operand rules), tile splits, batch slicing, NHWC transpose staging, cuDNN compute types.
- `TestGpuConvWinograd.cpp` (device):
  - every admitted config, fp32/fp16/bf16 (the BF16 and F43 engines' configs included), fwd and bwd;
  - NCHW, NHWC and mixed layouts, groups, padding 0-2 and asymmetric, odd extents, deep channels, true convolution;
  - the LLC plan with a forced small cache;
  - wrw against an fp64 reference with sliced and split-heavy cases;
  - the determinism relaunch; every config parses;
  - fused forward cases: each distinct kernel path among the admitted configs on the pointwise menu,
    NaN/infinity operands and clip boundaries, against the CPU convolution followed by the chain in
    double; only the forward matcher claims a fused graph.
- `TestConvDescriptorShards.cpp` checks that all three engines report `WINOGRAD` and nothing else.
