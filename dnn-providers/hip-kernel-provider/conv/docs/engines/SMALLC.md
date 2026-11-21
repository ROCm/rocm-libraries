<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_SMALLC_FWD / BWD — few-channel stems forward, and stride-1 backward data into few channels

> Engine ids: `hipkernel:CONV_SMALLC_FWD`, `hipkernel:CONV_SMALLC_BWD` · Generator: `conv/generators/gen_smallc.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/smallc/kernels/` (`ConvSmallc.cpp`, `ConvSmallcCore.h`) · Pack: `packs/ConvSmallcNative.cpp`, `packs/ConvSmallcProblem.hpp`

## At a glance

- **What:** a block stages its input window and its formatted weights in LDS once and runs the whole flattened (channel, depth, row, column) reduction as a handful of 16x16 matrix-core steps; no channel loop.
- **Covers:** fwd with at most 4 input channels (configs bound C at 1, 3 or 4); bwd-data at unit stride with at most 4 dy channels, run as the forward kernel over dy. fp32, fp16, bf16; any activation strides (NCHW, NHWC, views); 2-D and 3-D; groups 1; any padding; both conv modes. No TF32 form. Compile-time geometries: fwd 7x7 s2, 3x3 s2, 3x3 s1, 4x4 s4 (patchify), 1x7x7 and 3x7x7 at stride (1, 2, 2), 3x3x3 s1 and s2; bwd 3x3 and 3x3x3 at stride 1, and on gfx942 1x1(x1) from up to 8 dy channels (1x1 heads' backward data). Dilation 1.
- **Arches:** MFMA 16x16 on gfx908 (fp16, fp32, and bf16 on the half-rate 16x16x8 atom), gfx90a/gfx942 (fp16, bf16, fp32), gfx950 (16x16x32 for 16-bit, 16x16x4 fp32); WMMA on gfx11/gfx12 (fp16, bf16); an FMA kernel for fp32 (fwd on every architecture, bwd on gfx11/gfx12).
- **Wins when:** RGB / grey stems on gfx950, gfx942 and RDNA (1.5-2.2x MIOpen call-weighted), single-channel 3-D medical inputs (3-5x), VAE / diffusion output layers in backward data (c96-c320 dx from 3-4 dy channels, 1.4-2.2x), and a 3-D 1x1 head's backward data (c64 dx from 6 dy channels at large 3-D: 0.25 ms against 0.38 ms for the best BLAS engine, MI300A).
- **Loses when:** stems on gfx90a and gfx908 (1.6-2.2x behind MIOpen's stride-2 Winograd, CK, asm implicit GEMM and direct kernels), large-K NHWC 16-bit stems (sdxl c4 to k320, 1.1-1.4x behind CK), gfx950 patchify (behind hipconv's patch-embed kernel).
- **Precision:** fp32 accumulation, the reduction done inside one block, one rounding at the store. Deterministic. No numerical notes.
- **Instances:** 1057 fwd + 268 bwd configs, each shipped to the architecture groups whose sweeps it won (`archs`).
- **vs MIOpen:** competes with CK grouped/3-D implicit GEMM, the Winograd `ConvBinWinogradRxSf3x2` / `f2x3g1` and `GemmFwdRest` on stems: 1.0-1.3x vs Winograd and 1.8-3.0x vs CK 3-D on gfx950/942/RDNA, 0.5-0.6x on gfx90a/gfx908.

## Why it exists

- With C <= 4 an implicit GEMM fills only C x taps of each reduction step and gathers one scalar per reduction element; K1 and the BLAS engines spend their time on gathers and padding there.
- Stems run once per image at full resolution, so they are a fixed share of every CNN forward pass; output layers of VAEs and diffusion models (dx of c96-c320 from 3 or 4 channels) are the same shape in backward data.
- One small kernel over an LDS-staged window covers both: the whole reduction (C x KD x R x S, rounded up to whole MMA steps) fits a few MMA steps.

## How it works

GEMM view per block: `D[pixel][n] = sum_k A[pixel][k] B[k][n]`, with the reduction flattened as `k = rho * SP + s`, `rho = (c, t, r)` a filter row and `s < SP` a column tap (`SP` is S rounded up to a power of two; the extra taps carry zero weights).

```
block: stage the input window [c][t][row][col] (zeros outside the image and for c >= C)
       stage the BN x KP formatted weights
       each wave: FM rows x 16 columns x 16*FN output channels, KP / ka MMA steps from LDS
       epilogue: through a per-wave LDS slot in the output's layout (ep 0),
                 or straight from registers for channels-fastest output (ep 1)
```

With the window staged as `[c][t][row][col]`, the A element of (pixel, k) sits at the pixel's window base plus `rowOff(rho) + s * dilation`, so a lane's consecutive k are runs of adjacent window columns; on gfx11 WMMA (where every lane holds the same k) every offset is a compile-time constant.

Backward data at unit stride is the forward convolution over dy that writes dx: input channels K, output channels C, extents swapped, padding `(filter - 1) * dilation - prePadding` (negative where the graph pads past the filter), and the filter transposed (k <-> c) and reflected by the weight pre-pass.

## Implementation design

- **Two launches:** `ConvSmallcFormat` writes the formatted weights `wf[n][kp]` into the workspace (reflected for true convolution, transposed for backward data, zero past C, the filter or K); `ConvSmallc` runs the convolution. Workspace: the formatted weights only (`kpad x kpl` elements).
- **Tile:** `WM x WQ x WN` waves; output tile `TH = WM * FM` rows x `TW = 16 * WQ` columns of one output plane x `BN = 16 * FN * WN` channels; grid x walks (tiles, output planes, images), grid y the output-channel blocks.
- **FMA kernel:** 256 threads, `BN / FN` channel groups x runs of `FM` adjacent columns; weights staged transposed `[kp][n]` so a thread reads its `FN` weights as vectors; fp32 accumulation.
- **Gates** (`parseSmallcProblem`, `planFor`, kernel match): everything the naive parser gates; fp32/fp16/bf16; groups 1; rank 1-3 with positive extents; bwd requires unit stride on every axis; non-negative activation strides with one kernel-input image (its strided span) under 2^31 bytes; per config the channel bound, filter (KD, R, S), strides and dilations equal to the graph's; `ep` 1 only for channels-fastest output; grid x within 2^31 - 1 and 2^32 - 1 work items, grid y within 65535, formatted weights within 2^31 elements; the atom on the device and taking the dtype; the device named in the config's `archs` list.
- **Addressing:** 64-bit image bases, 32-bit buffer offsets within an image (window loads read zeros out of range).
- **Catalog:** per geometry and channel bound, 4 shared tiles (plus two 7x7 tiles for gfx90a), the register-epilogue twins only for NHWC 3x3 s1, 3x3 s2 and 4x4 s4 with 3-4 channels on per-arch tiles, and 5 FMA tiles for fp32. LDS-fit filtering per atom (160 KiB on gfx950's 16x16x32 configs). Each config is admitted on correctness alone; autotune picks.

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. Counted over the graphs where the engine is our autotuned best (W/L: faster / slower by more than 5%):

| GPU | FWD best on (of admitted) | FWD W/L | FWD factor | BWD best on (of admitted) | BWD W/L | BWD factor |
|---|---|---|---|---|---|---|
| gfx950 | 40 (46) | 29/9 | 1.96x | 11 (11) | 9/1 | 1.71x |
| gfx942 (MI325X) | 28 (46) | 16/10 | 2.23x | 7 (11) | 7/0 | 1.44x |
| gfx90a | 38 (46) | 8/30 | 1.10x | 9 (11) | 7/2 | 1.56x |
| gfx908 | 26 (38) | 4/20 | 1.14x | 2 (2) | 0/2 | 0.77x |
| gfx906 | 6 (10, fp32 only) | 5/1 | 3.32x | none | - | - |
| gfx1201 | 32 (46) | 17/11 | 1.68x | 10 (11) | 9/1 | 1.77x |
| gfx1100 | 27 (46) | 12/10 | 1.54x | 7 (11) | 7/0 | 1.86x |

- **Wins:** nnU-Net 3-D single-channel 3x3x3 fp32 fwd (3-5x vs `GemmFwdRest` / CK 3-D), ResNet and DeepLab 7x7 s2 fp16 stems (1.2-2.2x), R(2+1)D / CSN 3-D stems (2.2-2.5x), SD/Flux VAE and Wan-VAE output layers in bwd (1.6-2.2x).
- **Losses:** gfx90a and gfx908 stems are the only material ones (about 11 ms call-weighted excess each; elsewhere at most 2 ms): 7x7 s2 fp16/fp32 against `ConvBinWinogradRxSf3x2` and CK (1.6-1.9x) and against MIOpen's direct and asm kernels (2x). The kernel is LDS-bound there; extra tiles do not close it.
- 3x3 s1 fp16 NCHW stems trail `ConvWinoRageRxS` on gfx942 (1.3x, three graphs) and marginally on gfx950; NHWC bf16 c4 stems (sdxl) trail CK by 1.1-1.4x on gfx950, gfx942 and RDNA.
- gfx906 and gfx103x get only the fp32 FMA path; their 16-bit stems fall to K1.

## Fusion

`CONV_SMALLC_FWD` claims a forward convolution followed by a chain of pointwise nodes (scale, bias,
relu / clipped relu / clamp / leaky relu, gelu, gelu-tanh, swish / silu, sigmoid, tanh, elu, abs,
softplus; per-channel, one-element and pass-by-value operands) under its own engine id. Every
store applies the chain to the fp32 accumulator before the single rounding: the LDS-slot
epilogue (ep 0) before the slot, the register epilogue (ep 1) and the FMA kernel's vector and
scalar stores. A thread's channels are fixed, so their per-channel operands load once, before the
stores. The weight pre-pass compiles as unfused, and an unfused graph's kernels are unchanged.
The chain is compiled in as generated code that depends on the pattern only; operand pointers and
values are a second kernel argument. The backward-data configs decline fused graphs.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.08x, gelu 1.46x, silu 1.23x; gfx950 (both layouts) relu 1.06x, gelu 1.25x, silu 1.11x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.65x on MI325X and 1.34x on gfx950 (usage-weighted; the conv README's fusion table).

## Numerical precision

- fp32 accumulation (fp16/bf16 products exact in fp32 on the 16x16 atoms, fp32 on the 16x16x4 atom or FMA); the whole reduction runs inside one block in a fixed order; one rounding at the store. No split, no atomics. The device suite relaunches every config and compares bits.
- Channel padding (c >= C) and tap padding (s >= S) are stored as zeros and multiplied: an infinite input or weight can meet those zeros and give NaN where an exact reference gives ±inf. This is an open parity item.
- No TF32 sibling: a three-bf16 form (3C channels) measures 0.27-0.75x of exact on every architecture, so none ships.

## Comparison with MIOpen

Speedup = MIOpen time / SMALLC time, call-weighted over the graphs where SMALLC is our best, by MIOpen's chosen solver:

| MIOpen solver | Where it competes | gfx950 | gfx942 | gfx90a | gfx908 | gfx1201 / gfx1100 |
|---|---|---|---|---|---|---|
| `ConvHipImplicitGemmGroupFwdXdlops` (CK) | 2-D stems | 1.84x | 0.98x | 0.60x | 0.62x | 1.05x / 0.98x |
| `ConvHipImplicitGemm3DGroupFwdXdlops` (CK) | 3-D stems | 3.01x | 2.91x | 1.77x | 4.70x | 2.00x / 1.85x |
| `ConvBinWinogradRxSf3x2` | 7x7 / 3x3 s2 stems | 1.32x | 1.13x | 0.61x | 0.53x | 1.02x / 1.08x |
| `ConvBinWinogradRxSf2x3g1` | 3x3 s1 stems | 1.32x | 1.19x | - | 0.71x | 1.04x / 1.06x |
| `GemmFwdRest` | 3-D single-channel inputs | 3.71x | - | 4.79x | - | 4.12x / 3.06x |
| `ConvHipDirectFwd` (gfx90a), `ConvAsmImplicitGemmGTCDynamicFwdXdlops` (gfx908) | stems | - | - | 0.46x | 0.52x | - |

Backward data (CONV_SMALLC_BWD) against CK grouped and 3-D bwd: 1.3-2.2x on every GPU except gfx908 (two graphs, 0.77x). gfx906 (fp32 forward only): 1.21x against `ConvBinWinogradRxSf2x3g1`, 5.65x against `GemmFwdRest`. On gfx950, 4x4 s4 patchify trails hipconv's patch-embed kernel (`ConvHipConv`) by 1.45x on microsecond graphs.

MIOpen's stride-2 Winograd computes fewer multiplies on 7x7 s2 stems; SMALLC does every multiply but from LDS with no gathers, so it wins where matrix-core throughput is high (gfx950, gfx942, RDNA) and loses where it is not (gfx90a, gfx908).

## Prior art considered

- hipconv's patch-embed family solves the related case of filters no larger than the stride with one MFMA GEMM over a gathered input; SMALLC stages an overlapping window once instead, so one structure serves 7x7 s2, 3x3 s1 and patchify.
- K1's implicit-GEMM view, with the reduction flattened across channels and taps so that a few channels still fill the MMA steps.
- K1's matrix-core atoms, element conversions and buffer loads are reused. No code was copied.

## Next steps and deferred ideas

1. **gfx90a/gfx908 stems** (about 11 ms call-weighted each): the kernel is LDS-bound; a register-staged window or a Winograd-style stride-2 form would be needed. Extra tiles on gfx90a gain 5-11% and no more.
2. **Large-K NHWC 16-bit stems** (sdxl c4 to k320; 0.76-0.97 of MIOpen on gfx950, gfx90a, gfx1201): more output channels per block or a wider register epilogue.
3. **inf/NaN parity:** mask channel and tap padding instead of multiplying zeros, A/B'd against the compiled-out build.
4. **Long-tail geometries:** every filter, stride and dilation is compile-time, so a stem outside the list (5x5, 11x11 s4, dilated) goes to K1. A runtime-geometry generic tier would cover them; the stem corpus shows little demand so far.
5. **16-bit on gfx906 and gfx103x:** no configs; K1's FMA serves them.

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestConvSmallcProblem.cpp` (host): acceptance of any layout and rank; refusal of groups, other directions and over-large images; admission by atom and ship list; channel bound and geometry must match; one refusal per malformed config field; FMA thread coverage; register epilogue only for channels-fastest output; bwd unit stride on every axis, dy channel bound, dy as the kernel input and dx as its output; tile, grid and formatted-weight arithmetic; compute types.
- `TestGpuConvSmallc.cpp` (device): every admitted config in fp32, fp16 and bf16 on 7x7 s2 (NCHW, NHWC, one channel, many tiles, true convolution), 3x3 s2 (padded and unpadded), 3x3 s1 with 2 and 4 channels, 4x4 s4, 3x7x7 and 1x7x7 s(1,2,2), 3x3x3 single-channel and s2; bwd 3x3 with 1, 3 and 4 dy channels (130-channel dx, padding past the filter, NCHW, true convolution, many tiles) and 3x3x3 in both layouts. Bitwise relaunch of every config; catalog parses and bwd configs are unit-stride.
