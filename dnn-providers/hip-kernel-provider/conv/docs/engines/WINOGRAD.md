<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_WINOGRAD (K5) — Winograd 3x3 convolution: fused, cooperative and multipass, all three directions

> Engine ids: `hipkernel:CONV_WINOGRAD_FWD`, `hipkernel:CONV_WINOGRAD_BWD`, `hipkernel:CONV_WINOGRAD_WRW` · Generator: `conv/generators/gen_winograd.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/` (`ConvWinograd.cpp`, `ConvWinogradCore.h`, `…Fused.h`, `…Sreg.h`, `…Wmma.h`, `…WmmaReg.h`, `…WmmaCoop.h`, `…Wrw.h`, `…Args.h`) · Pack: `packs/ConvWinogradNative.cpp`, `packs/ConvWinogradProblem.hpp`

## At a glance
- **What:** Winograd minimal filtering for 3x3 filters.
  - Forward and backward data: F(2,3) for every dtype; F(4,3) for fp32 multipass only, declined by default; F(3,2) on the four 2x2 phase sub-filters for stride 2.
  - Backward weights: F(3,2), F(3,4) and F(3,6), by duality with F(r,3).
- **Covers:**
  - fwd/bwd: 2-D 3x3, stride 1 or 2 (the same on both axes), dilation 1, groups, any padding, both convolution modes, any layout (staged when a variant does not read it). fp32 and fp16; bf16 only with `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_BF16`.
  - wrw: stride 1, fp32 / fp16 / bf16 under per-dtype accuracy rules.
  - No 1-D or 3-D, no other filter sizes, no TF32 form.
- **Variants:**
  - `multipass`: transforms around a batched GEMM, any arch, NCHW and NHWC;
  - `fused`: VALU, every arch, `v_dot2_f32_f16` for fp16;
  - `wmma` and `wreg`: register-built WMMA operands, gfx11/gfx12;
  - `sreg`: register VALU, wave64 gfx9;
  - `coop`: cooperative WMMA (gfx11/gfx12) or MFMA (gfx908+), with a 16-bin RDNA layout;
  - `wrw`: F(3,r) GEMM per Winograd position.
- **Arches:** all. Matrix cores are used only for 16-bit operands; fp32 runs on VALU.
- **Grids:** persistent variants size their grid by occupancy: units x blocks that are actually resident (registers and LDS). A second, partly empty round measured 15-26% slower.
- **Wins when:**
  - NCHW fp16 3x3 on CDNA2 / CDNA1 (gfx90a 1.30x fwd / 1.50x bwd on its graphs), against CK and older implicit GEMMs;
  - fp32 3x3 wrw on detection backbones: Mask R-CNN c256 200², 0.52-0.68x of MIOpen's time on CDNA.
- **Loses when:**
  - RDNA fp16 small-C backward data against `ConvWinoRage/Fury` (1.4-1.7x);
  - fp32 Mask R-CNN 200² fwd/bwd against `ConvBinWinogradRxSf2x3g1` (1.13-1.24x), with F(4,3) declined;
  - gfx906 fwd/bwd against MIOpen's asm Winograd (about 1.55x behind, weighted).
- **Precision:** the `WINOGRAD` numerical note on all three engines. Transforms are fp32; 16-bit operands are rounded once after their transform, except the input transform of `f16math` configs. Accumulation is fp32. An output whose window lies wholly in padding is exactly 0. Deterministic.
- **Instances:** 420: 168 fwd, 168 bwd, 84 wrw. 16 fp32 F(4,3) multipass configs (8 fwd, 8 bwd) are present but declined by default, and all bf16 fwd/bwd configs are declined by default.
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
  - RDNA has an 8x16 micro-tile that reads 0.75 LDS bytes per FMA instead of 1, because the Hadamard is LDS-bound there (Mask R-CNN c256 200²: 14% on gfx1201; gfx1100 29.9 → 23.0 ms with depth-2 prefetch).
  - 16- and 24-wide K blocks serve small groups and K = 24 / 48 layers.
- **coop (cooperative matrix core):**
  - The whole block transforms each 16-channel input chunk once into LDS, so one transformed value feeds all of the block's output channels.
  - Wave w owns transform row w % 4 (4 bins) and a K slice. A fragments (U) come from a fragment-major pre-pass buffer, one contiguous 512-byte run per fragment, prefetched a chunk ahead.
  - Epilogue: the wave folds its 4 bins to 2 values in registers, writes them to LDS, then applies Aᵀ over the rows.
  - RDNA and CDNA share one body over the atom traits (wave32 WMMA, wave64 MFMA 16x16x16).
- **16-bin layout** (`bins 16`, RDNA F(2,3)):
  - A wave owns all 16 bins of 16 channels x 16 tiles, so the output transform runs in registers: no LDS round trip and no epilogue barriers.
  - It wins where the output side dominates: backward data with few dy channels. Measured on UNet c128 k64 512² backward data: gfx1201 9.07 → 7.76 ms, gfx1100 9.83 → 8.66 ms.
  - Its grid follows the residency of its 256-VGPR waves: 6 a SIMD on the 1536-register RDNA parts, 4 elsewhere.
- **wreg / sreg:** register-operand variants.
  - In `wreg` (gfx11/gfx12), each wave owns one transform row and builds its WMMA B fragments from the loaded input rows in registers: no LDS for V, and no barrier in the channel loop. gfx11's half-waves exchange channels with `v_permlanex16`.
  - In `sreg` (wave64 gfx9), lane = tile, and the transformed filter is a wave-uniform operand through the scalar cache.
- **Grids and order.** Fused / sreg use 2 blocks per dispatch unit. Coop uses about 1024 resident threads per unit, capped on CDNA by a shared occupancy model (`k5CoopCdnaWpe`, `k5CoopLdsBytes`) for registers and LDS (160 KiB a CU on gfx950, 64 KiB before). Items run K blocks fastest, or tile blocks fastest on RDNA when a group's U is ≥ 4 MiB.
- **Codegen.**
  - Branch-free 32-bit transform offsets with an out-of-range sentinel for padding: the fused input slice is kept under about 1 GiB.
  - Chunk loads use the chunk as a scalar offset, so a load needs no VALU.
  - The fp16 first chunk is peeled so MFMAs start from zero.
  - A `readfirstlane` on the U offset avoids waterfall loops.
  - A V swizzle spreads LDS banks: CDNA conflict cycles fell 2.7x on gfx950 UNet.
- **Gates.**
  - Problem gates, everything the naive parser gates, plus: 2-D; 3x3 filter; dilation 1; equal stride 1 or 2 (wrw 1); n, c·stride², k and groups within 32 bits; extents and padding within 32 bits.
  - fwd/bwd bf16 only with the switch.
  - Per config: dtype and direction equal to the graph's; atom on the device; WMMA / coop not fp32; sreg not bf16; `f16math` only for fp16; coop bf16 MFMA from gfx90a; F(3,2) exactly for stride 2; F(4,3) only fp32 with `HIP_KERNEL_PROVIDER_CONV_WINOGRAD_F43`; a device reporting no units declined for persistent variants.
  - wrw per-dtype rules (below); every plan within launch, buffer and item-count limits.
- **wrw.**
  - x and dy are read in place when packed channels-first. Packed channels-last operands go through 64 x 64 LDS transposes, which measured 3-23% faster than transforming in place on gfx906 / gfx90a UNet; others are copied.
  - GEMM on MFMA (gfx9 16x16, gfx950 16x16x32), WMMA, or FMA thread tiles.
  - Tile splits are chosen for about 2048 threads per unit; batch slices stay under the workspace limit.
- **Large tensors.** Fused slices are capped so one slice's input is under about 1 GiB. Multipass and wrw slice the batch under the workspace limit. A single image past those limits is declined, and other engines serve it.
- **Catalog.** Variants x tiles per dtype and arch family. The knobs are `m`, `llc`, the GEMM tile, `bk`, `bt`, `ccv`, `tm`, `tn`, `waves`, `bins`, `f16math` and `pf`. Configs were kept where they won measured layers (realesrgan, UNet, Mask R-CNN, HRNet, Inception, IGM).

## Strengths and weaknesses
Measured: an autotuned sweep of the 1,998-shape model-corpus subset against MIOpen in search mode. W/L counts our-best graphs that are more than 5% faster or slower. Losses are counted against MIOpen's best, which is often its own Winograd.

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
  - RDNA fp16 small-C (c32-c192, k32-k64) against Rage / Fury: the cost is per-item overhead (load latency, barriers, the Z round trip), not the MMAs.
  - RDNA fp32 large layers, where F(4,3) is declined.
  - gfx906 fused, where the Hadamard runs near peak but staging, loads and barriers cost as much again.
  - CDNA small-C: Inception c32 and EfficientNet c24 still trail Rage by 1.5-1.7x, HRNet c32 by about 1.15-1.2x.

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
- **Declined forms** (env-gated; no sibling engine exists yet):
  - **bf16 fwd/bwd** (`HIP_KERNEL_PROVIDER_CONV_WINOGRAD_BF16`): matrix-core operands round the transformed tiles to bf16, about a third of bf16's unit roundoff of each tile's Σ|products|. That is past the convolution's precision model, and MIOpen has no bf16 Winograd. Deferred cost, full NCHW tier: fastest on 284 graphs on gfx1201 (503 ms call-weighted lost of 4,498 ms) and 73 on gfx1100 (440 ms); none in the NHWC tier.
  - **fp32 F(4,3)** (`HIP_KERNEL_PROVIDER_CONV_WINOGRAD_F43`): transforms with entries up to 8 and 1/24 put about 27 fp32 ulps of Σ|products| into a c128 layer, against about 1 for F(2,3) and the 5.7 coefficient MIOpen's fp32 Winograd needs. Deferred cost: F(4,3) was 25-36% faster on fp32 Mask R-CNN c256 on gfx1201 (200²: 19.1 against 24.4 ms), bounding the loss at about 300 ms call-weighted on gfx1201 NCHW.
  - Both become tagged plans or sibling engines once callers can deselect by numerical note. Engines report notes per engine, so a tag cannot sit on individual configs.
- **TF32:** none. A three-bf16 multipass form measured 0.21-0.58x of the exact configs. A paying form needs the split inside the fused cooperative kernel.
- **Determinism:** fixed-order sums, no atomics. Every config is relaunched over a re-filled workspace and must reproduce its bits.

## Comparison with MIOpen
- **`ConvBinWinograd3x3U`, `ConvBinWinogradRxS`, `ConvBinWinoRxS<2,3>/<3,2>`, `…RxSf2x3g1`:** binary asm for gfx9 / gfx10 / gfx11 / gfx12, fp32 and fp16, NCHW. They also serve arbitrary RxS by decomposing the filter into 3x3 pieces, and stride 2 by decimation. They win gfx906 fwd/bwd and RDNA fp32 Mask R-CNN 200² (1.13-1.24x).
- **`ConvWinoFuryRxS` (gfx11 / gfx12) and `ConvWinoRageRxS` (gfx942 / gfx950 / gfx12):** fp16 only, the WMMA / MFMA fused kernels. Fury builds B fragments in registers; K5's `wreg` and 16-bin `coop` take that idea. They lead on small-C fp16: 1.4-1.7x on RDNA backward data, 1.2-1.4x on most CDNA graphs, and up to 1.7x on Inception / EfficientNet.
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
1. **RDNA fp16 small-C backward data against Rage / Fury** (largest RDNA Winograd excess; 1.4-1.7x). Autotune does not pick the 16-bin config where it would win, so the planner needs register counts at plan time. Per-item overhead (loads, about 5 barriers per item) is the cost to cut.
2. **bf16 and F(4,3) as tagged plans** after note filtering lands. Payoff: about 0.5 s (bf16) and 0.3 s (F(4,3)) call-weighted on gfx1201 NCHW. Also try F(4,3) on points 0, ±1, ±1/2 for less error.
3. **CDNA small-C:** extra VALU and bank conflicts in V stores and B reads; LDS-resident U to fuse the 3-4 µs filter-transform launch.
4. **gfx906 fused:** reader/writer wave specialization, or a persistent design with U in LDS.
5. **TF32:** split inside the cooperative kernel, not in multipass.
6. **Other filter sizes by 3x3 decomposition**, as MIOpen's RxS does. Not yet measured; `CONV_DIRECT`, `CONV_IGEMM` and `CONV_FFT` serve those today.

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestConvWinogradProblem.cpp` (host):
  - **Gates:** one refusal per field; direction exclusivity; bf16 and F(4,3) declined unless permitted; stride-2 phase gather/scatter and 16 positions.
  - **Variant placement:** each variant on its arch family and dtypes; the 16-bin layout on RDNA stride 1; coop MFMA from gfx908 and bf16 from gfx90a; fused stride-2 chunking; inconsistent register-variant fields refused.
  - **Plans:** LLC slicing and workspace; occupancy-sized grids (16-bin on RDNA, CDNA); item order for a large U; fused slices clear of the out-of-range offset.
  - **wrw:** error-band admission (FMA every dtype, fp32 F(3,4) from 512 products, half-operand rules), tile splits, batch slicing, NHWC transpose staging, cuDNN compute types.
- `TestGpuConvWinograd.cpp` (device):
  - every admitted config, fp32/fp16/bf16 (bf16 and F(4,3) under their switches), fwd and bwd;
  - NCHW, NHWC and mixed layouts, groups, padding 0-2 and asymmetric, odd extents, deep channels, true convolution;
  - the LLC plan with a forced small cache;
  - wrw against an fp64 reference with sliced and split-heavy cases;
  - the determinism relaunch; every config parses.
- `TestConvDescriptorShards.cpp` checks that all three engines report `WINOGRAD` and nothing else.
