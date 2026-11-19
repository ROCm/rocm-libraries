<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_STEM_BWD — backward data for stems (at most 4 input channels) by stride-phase super-pixels

> Engine id: `hipkernel:CONV_STEM_BWD` · Generator: `conv/generators/gen_conv_stem_bwd.py` (ships the configs in `conv/generators/conv_stem_bwd_kept.txt`) · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/stem_bwd/kernels/` (`ConvStemBwd.cpp`, `ConvStemBwdCore.h`) · Pack: `packs/ConvStemBwdProblem.cpp`, `packs/ConvStemBwdNative.cpp`

## At a glance

- **What:** dx as one GEMM per block of super-pixels (the `sh x sw` dx pixels of one stride-phase group): M the super-pixels, N the input channels times the phases, the reduction the output channels times the window offsets of one phase. A pre-pass rearranges the filter into that operand.
- **Covers:** bwd-data with C <= 4, one group; fp32, fp16, bf16; dy packed NCHW or NHWC (read in place), dx packed either way (written through its strides), filter K C R S or K R S C; 1-D, 2-D, and 3-D with unit depth stride and up to 8 depth taps; unit height/width dilation; H/W strides, filters and padding up to 64; both conv modes. No TF32 form. Catalog geometries: 7x7 s2, 3x3 s2, 6x6 and 5x5 s2, 2x2 s2, 3x3 / 5x5 / 7x7 s1, 1x1, 4x4 s4 patchify, 11x11 s4; 3-D 1x7x7 and 3x7x7 at stride (1, 2, 2) and 3x3x3 s1. Stride-1 configs can also run on a 2x2 space-to-depth view of dy.
- **Arches:** MFMA 16x16 on gfx908-gfx950 (16x16x32 for 16-bit on gfx950; gfx908's own bf16 instructions), WMMA on gfx11/gfx12, an FMA loop for fp32 on every architecture and for 16-bit on gfx906.
- **Wins when:** RGB stems in training: 7x7 s2 (ResNet, DeepLab, Mask R-CNN), 3x3 s2 NHWC (EfficientNet), patchify (ConvNeXt), 3-D video stems. 1.4-2.3x MIOpen call-weighted on every GPU, up to 5-9x on single graphs.
- **Loses when:** stride-1 3x3 RGB stems at high resolution (U-Net 512²) against MIOpen's asm Winograd (`ConvWinoRageRxS` / `ConvWinoFuryRxS`, 0.8-0.9x), and one fp32 3x3 stem per CDNA GPU against `ConvAsm3x3U`.
- **Precision:** fp32 accumulation, the whole reduction in one block, every dx element written once. Deterministic. No numerical notes.
- **Instances:** 3034 configs in 12 packs (per dtype and architecture family): 2249 vector configs (chosen from the universe by a per-config bench on seven architectures, plus the native gfx1250 candidates), and 785 channels-last single-element twins of the kept channels-last vector configs; 4-9 admitted per graph typically, at most 21.
- **vs MIOpen:** replaces `GemmBwdRest` (col2im), the stride-2 Winograd `ConvBinWinogradRxSf3x2`, CK grouped bwd and the HIP direct kernels on stems.

## Why it exists

- With C <= 4, dx has 1-4 channels: any GEMM whose N axis is the input channels leaves 12-15 of 16 matrix-core columns idle, and MIOpen's col2im path materializes a column buffer and sums the overlapping columns into dx.
- Stem backward data runs at full input resolution once per training step; without this engine it is most of MIOpen's backward-data lead over our engines on some GPUs (on gfx1101, 86 of the 106 ms by which `GemmBwdRest` leads; indicative).
- Folding the stride phases into N fills the matrix columns (12 of 16 for an RGB stem at stride 2) and makes every dx element a single accumulator.

## How it works

For a super-pixel `(i, j)` standing for dx pixels `(sh*i + p, sw*j + q)`:

```
D[(i, j)][(c, p, q)] = sum over (u, v, k) of  dy[k][i + OH - u][j + OW - v] * B[(u, v, k)][(c, p, q)]
```

`B` holds `w[k][c][r][s]` for the filter row and column that pair phase `(p, q)` with window offset `(u, v)` (`r = sh*(u - OH) + p + ph`), zero where the pair falls outside the filter. The window of one axis (stride `s`, filter extent `f`, padding `pad`) has offset `floor((s - 1 + pad) / s)` and extent `offset - ceil((pad - f + 1) / s) + 1`: 7x7 s2 p3 is a 4x4 window, 3x3 s2 p1 a 2x2 window, patchify a 1x1 window. A config fixes the strides and window, so it serves every (filter, padding) pair that yields that window.

- **3-D** (`ft`): the dx planes go to the grid's z, the depth taps into the reduction (`k' = (tap, k)`), unit depth stride.
- **Space-to-depth** (`sd` = 2, stride 1): the stride-1 problem is run on 2x2 super-pixels over a space-to-depth view of dy; the 4 sub-pixel phases fill N (4C columns) and the sub-pixels join the reduction. Matrix-core configs only (on FMA it measured 1.3-1.5x slower).

## Implementation design

- **Two launches:** `ConvStemBwdTransform` builds `B` (`nb x u x v x kp`, `k` padded per chunk) in the workspace, reflecting the filter for true convolution; the main kernel stages a dy window and a chunk of `kc` output channels in LDS per step (dy read as 16-byte column vectors or scalars for NCHW, 16-byte channel vectors for NHWC, or single channels where K is not whole vectors, as r(2+1)d's 45), multiplies, and writes each dx element once through dx's strides.
- **Tile:** `th` super-rows per block (one wave each on a matrix atom; 16 threads each on FMA) by `fm` fragments of super-columns; grid (super-columns, super-rows, images x dx planes).
- **Gates** (`parseStemBwdProblem`, `stemBwdLaunchFor`, every field): everything the naive parser gates; fp32/fp16/bf16; groups 1; C <= 4; a trivial depth axis or unit depth stride and dilation with 1-8 taps and depth padding <= 64; unit H/W dilation; H/W strides, filters and padding <= 64; dy packed channels-first or channels-last, dx and the filter packed either way; extents and K within 32 bits; images x dx depth <= 65535; one dy image under 2^31 bytes. Per config: atom and `atom_id` (FMA takes fp32, or 16-bit only where no matrix atom exists for the type); layout and `vec` (NHWC: 16 bytes with K whole vectors, single elements with K not; NCHW: scalar or 16 bytes with whole-vector dy rows, space-to-depth scalars only for partial rows); `u`, `v`, `sh`, `sw`, `ft`, `sd` equal to the problem's; `nb` >= C x phases, <= 64, whole fragments; `kc` whole atom steps and 16-byte units; block <= 1024 threads; LDS within the device's (160 KiB on gfx950, 64 KiB elsewhere, with the kernel's own arithmetic); grid within one dispatch; the transformed filter within 2^31 bytes.
- **Workspace:** the transformed filter only.
- **Codegen:** shares K1's `readfirstlane` buffer descriptors (without it the current HIPRTC emits waterfall loops around the dy loads).
- **Catalog selection:** every universe config a graph admits is timed (transform plus main kernel, best of 8 launches) on gfx950, gfx942, gfx90a, gfx908, gfx1201, gfx1101 and gfx906 over the 82 corpus stems and stems of every covered geometry in three dtypes and two layouts; configs are visited best first and dropped when a kept config is within 2% on every (architecture, graph) they serve. Every graph the universe serves is served, and no measured graph's best time grows by more than 2%. The method is in the kept list's header; re-run it after kernel changes.

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. STEM_BWD is admitted on 33 graphs per GPU; counted over those where it is our best:

| GPU | best on | W/L | call-weighted factor | weighted time saved | weighted excess |
|---|---|---|---|---|---|
| gfx950 | 21 | 18/2 | 2.27x | 14 ms | 0.0 ms |
| gfx942 (MI325X) | 24 | 19/4 | 1.72x | 13 ms | 0.2 ms |
| gfx90a | 30 | 25/2 | 1.66x | 44 ms | 0.7 ms |
| gfx908 | 24 | 23/1 | 1.94x | 60 ms | 0.5 ms |
| gfx906 | 26 | 18/5 | 1.41x | 77 ms | 4.6 ms |
| gfx1201 | 19 | 17/2 | 1.58x | 40 ms | 0.4 ms |
| gfx1100 | 25 | 17/7 | 1.36x | 37 ms | 1.0 ms |

- **Wins:** ResNet 7x7 s2 n128 fp16 (5.1x on gfx950, 1.8x on gfx1201), DeepLab 7x7 s2 at 1024² (2.0-2.4x), EfficientNet-V2 3x3 s2 NHWC fp16 (4.5-6.2x), ConvNeXt 4x4 s4 patchify NHWC bf16 (6x on gfx1201), CSN 3x7x7 3-D (1.95x on gfx950).
- **Losses:** U-Net 3x3 s1 c3 fp16 NCHW stems against `ConvWinoRageRxS` / `ConvWinoFuryRxS` (0.8-0.9x, with the space-to-depth configs; 0.56x without them); a fp32 3x3 stem against `ConvAsm3x3U` on gfx90a, gfx908 and gfx906 (0.4-0.6x, one graph each); gfx950 patchify NHWC bf16 against hipconv's `ConvHipConv` (2 of 4 graphs, up to 1.4x).

## Numerical precision

- fp32 accumulation (fp16/bf16 products exact on the 16-bit atoms; fp32 on fp32 MFMA or FMA); the whole reduction over (window, output channels, depth taps) runs inside one block in a fixed order; one rounding at the store; no split, no atomics. The device suite relaunches every config and compares bits.
- Window offsets a phase does not use hold zero filter entries and are multiplied: an infinite dy value can meet those zeros and give NaN where an exact reference gives ±inf. This is an open parity item.
- No TF32 sibling: a three-bf16 form (3K) measures 0.25-0.70x of exact on every architecture, so none ships.

## Comparison with MIOpen

Speedup = MIOpen time / STEM_BWD time, call-weighted over the graphs where STEM_BWD is our best, by MIOpen's chosen solver:

| MIOpen solver | Approach | Speedup |
|---|---|---|
| `GemmBwdRest` | rocBLAS GEMM into columns, then col2im | 1.2-2.1x (every GPU) |
| `ConvBinWinogradRxSf3x2` | stride-2 Winograd (backward via transposition) | 1.2-4.9x |
| `ConvHipImplicitGemmGroupBwdXdlops` (CK) | grouped implicit GEMM (NHWC stems) | 4.2-7.9x |
| `ConvHipDirectFwd` | HIP direct kernel run backward | 1.5-1.8x on gfx90a/gfx908, parity on gfx906 |
| `ConvWinoRageRxS`, `ConvWinoFuryRxS` | asm Winograd, stride 1 (Rage on gfx942/gfx950/gfx12, Fury on gfx11) | 0.8-0.9x (U-Net 3x3 s1, gfx942 and RDNA) |
| `ConvAsm3x3U` | asm 3x3 direct, fp32 | 0.4-0.6x (one graph per CDNA GPU) |

MIOpen's GEMM path writes columns and scatters them; its CK path runs a GEMM whose N axis is the 3-4 input channels. STEM_BWD's phase folding fills the matrix columns and writes dx once, which is where the 4-8x NHWC wins come from. Winograd wins stride-1 3x3 where its multiply count is lower.

## Prior art considered

- The sub-pixel (stride-phase) decomposition of transposed convolution and the space-to-depth (pixel-shuffle) view, both standard reformulations.
- MIOpen's col2im backward data and direct kernels as baselines; K1's atoms and buffer loads reused. No code was copied.

## Next steps and deferred ideas

1. **Stride-1 3x3 RGB stems vs RDNA Winograd** (0.8-0.9x on U-Net 512²): the losses with weight; a Winograd-domain or larger space-to-depth (`sd` 3, 4) variant would cut multiplies.
2. **inf/NaN parity:** skip or mask the zero window entries instead of multiplying, A/B'd against the compiled-out build.
3. **Spilling configs:** the fp32 FMA 4x4-window configs spill; retune or gate them (spills alone are not a reason to prune).
4. **3-D at depth stride 2** (3x3x3 s2 stems) is declined today; K1 serves it. Folding depth phases into N like the H/W phases would extend the engine.
5. **Re-select the kept list** after any kernel change, with the bench method recorded in the list's header.

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestConvStemBwdProblem.cpp` (host, 31 tests): window offsets and extents of the stems, every tap of every phase falls in the window, space-to-depth offsets, extents, shifts and pairings; acceptance; layouts and window read from the graph; one refusal per graph field and per config field; gate cases one step from each limit; FMA takes 16-bit only without a matrix core for it; vector rules (channels-last single elements only for K not whole vectors); columns cover channels x phases; atoms per architecture; grid and transformed-filter limits; launch arithmetic for 2-D, 3-D and space-to-depth; layouts, strides and the flip; 1-D graphs pad the leading axes; LDS mirrors the kernel; every field is a define; compute types.
- `TestGpuConvStemBwd.cpp` (device): every admitted config in fp32, fp16 and bf16 on 7x7 s2 (NCHW, NHWC, whole rows, one and four channels, odd K), 3x3 s2, 6x6 / 5x5 / 2x2 s2, 3x3 / 5x5 / 7x7 s1 (odd extents, whole rows, one-sided padding, true convolution), 1x1, patchify, 11x11 s4, asymmetric padding, true convolution, mixed dx/filter layouts, 3-D of depth one, 1x7x7 / 3x7x7 s(1,2,2) (NHWC, leading depth padding, true convolution) and 3x3x3 s1, and channels-last output channel counts that are not whole vectors in 2-D and 3-D (r(2+1)d's 45, mixed dx/filter layouts); bitwise relaunch; catalog knobs distinguish every config and every config fits its architectures.
