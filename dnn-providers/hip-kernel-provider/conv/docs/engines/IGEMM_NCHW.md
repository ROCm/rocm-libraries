<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_IGEMM_NCHW_BWD (K7) — channels-first backward data with the col2im fused, in place

> Engine id: `hipkernel:CONV_IGEMM_NCHW_BWD` · Generator: `conv/generators/gen_igemm_nchw.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/igemm_nchw/kernels/` (`ConvIgemmNchw.cpp`, `ConvIgemmNchwCore.h`) · Pack: `packs/ConvIgemmNchwProblem.cpp`, `packs/ConvIgemmNchwNative.cpp`

## At a glance

- **What:** backward data as an implicit GEMM along the width with the col2im fused: a block owns a run of dx positions covering every stride phase, so each product lands in its own accumulator; dy, the filter and dx are read and written in place.
- **Covers:** bwd-data only; fp32, fp16, bf16 (three-bf16 TF32 form on RDNA 3-D strided only); packed NCHW / NCDHW dy and dx with a packed K C T R S filter; 1-D, 2-D, 3-D; groups 1; cross-correlation only; width strides 1, 2, 3, 4, 5, 6, 8 with the filter widths the catalog carries; any padding; depth/height any stride, dilation at unit stride.
- **Arches:** WMMA on gfx11/gfx12 (fp16, bf16); MFMA 16x16 on gfx908-gfx950 (fp16; bf16 from gfx90a; fp32 on the 16x16x4 atom); an FMA kernel everywhere (the fp32 path on RDNA).
- **Wins when:** strided NCHW backward data and ConvTranspose: vocoder 1-D (HiFi-GAN upsampling and dilated residual blocks), strided 1x1, patchify, 2x2x2 and 3x3x3 stride-2 3-D (nnU-Net). 1.4-2.4x MIOpen, call-weighted, with almost no losses.
- **Loses when:** large compute-bound GEMMs (wide channels, unit stride 2-D): its matrix-core loop reaches only 13-20% of peak there, and K1 or Winograd serve those.
- **Precision:** fp32 accumulation, every dx element written once by one block, no split, no atomics: deterministic. No numerical notes on the exact engine; `CONV_IGEMM_NCHW_BWD_TF32` reports `TENSOR_CORE`.
- **Instances:** 1845 exact configs, and 84 three-bf16 configs in `CONV_IGEMM_NCHW_BWD_TF32`; about 28-50 admitted per graph.
- **vs MIOpen:** replaces `GemmBwdRest` / `GemmBwd1x1_stride2` (GEMM into a column buffer, then a col2im scatter) and CK grouped bwd on NCHW; 1.5-2.6x faster than CK grouped bwd and 1.4-4.4x faster than `GemmBwdRest` on the graphs it wins.

## Why it exists

- K1 reads channels-last, so NCHW dy and dx go through staging transposes, and for strided problems its per-phase GEMMs are thin. With K1 alone, NCHW backward data is the largest group of losses on RDNA (55 graphs on gfx1100 and 42 on gfx1201 in a 414-shape audio/video set).
- Vocoders and 3-D segmentation networks run NCHW ConvTranspose at strides 2-8 with long, narrow rows; MIOpen serves them with a GEMM into a column buffer plus a col2im pass that sums the overlapping columns into dx.
- A separate kernel that fuses col2im by stride phase removes the staging, the workspace and the scatter. It is a narrow specialist; K1 keeps compute-bound NCHW.

## How it works

The convolution runs along the width; depth and height are rows. A dx row `(d, h)` reduces, one pair after another in the k loop, over every (dy row, filter row) pair that reaches it; a dx row no pair reaches is written as zeros.

Along the width, with `x + pad = stride * q + phase`:

```
dx[c][x] = sum over row pairs, k, j of  w[k][c][td][th][phase + stride*j] * dy[k][row'][q - j]
```

A block owns `BN` channels and `BMQ` consecutive phase indices `q`, all `stride` phases. Per k-step it stages the contiguous filter run `w[k][c0 .. c0+BN][0 .. S)` and the dy row window `dy[k][q0 - (taps-1) .. q0 + BMQ)`; phase `p` takes tap `p + stride*j` as the A rows shifted by `j`. MIOpen's GEMM path computes `col[c][s][q] = sum_k w[k][c][s] dy[k][q]` and then scatters it; here every product is accumulated where it lands, and each dx element is stored once.

At stride 1 (1-D convolutions, dilated residual blocks) there is one phase and the taps shift by the width dilation; the dy window's `halo` covers the farthest tap.

**Flat form:** when nothing pads, every filter extent is at most its stride, and dx is exactly the strides times dy on every axis (strided 1x1, patchify, 2x2x2 s2), each dy position owns a disjoint dx block. `flat` configs then run one GEMM per (depth, height) filter class over the dy positions of all images, computing only the phases a tap reaches (the rest are zero), so narrow planes fill whole tiles.

## Implementation design

**Kernels:**
- Matrix-core kernel on K1's atoms: `WM x WN x WP` waves over phase indices, channels and phase groups; A rows are `q`, B rows are channels, the reduction is `k`. LDS tiles are row-major in k, filled by transposing stores (each thread loads the same positions of `KV` consecutive k rows). fp32 configs (one k per lane) keep the FMA kernel's k-major tiles, stored untransposed.
- FMA kernel: threads over (channels, tile indices), each computing `TQ` phase indices of `TC` channels for all phases; its main loop is specialized per `pad mod stride` so every phase's register shift is compile-time; with `S = stride * taps` exactly there is no per-phase branch, so the FMAs of different phases interleave (dual-issue on gfx11/gfx12). `stages` 2 double-buffers the LDS tile (one barrier per k-step).
- Loads are branch-free: out-of-range positions get a byte offset the buffer's range check zeroes; a vector load straddling the row end falls back to elements (the range check is per dword).

**Host gates** (`parseNchwProblem`, `nchwLaunchFor`): everything the naive parser gates, plus fp32/fp16/bf16; one group; CROSS_CORRELATION; dy, dx and the filter packed channels-first; depth/height padding >= 0 and dilation only where that axis's stride is 1; width dilation only at stride 1; extents, strides, padding and the filter run within 32 bits; one dy image and the whole filter within 2^31 bytes; images times dx rows within 65535 (grid z). Per config: the atom and its `atom_id`; the kind's own fields set and the other kind's zero; tile against the atom step or thread tile; `vec` (1, 2, 4, 8; at most 16 bytes) dividing the tile, the phase span and the filter run; `stride` equal to the problem's (at most 8) and `taps` exactly `ceil(S / stride)`; `halo` covering `(taps - 1) * dilation` (+1 past stride 1); `flat` only on flat problems; LDS within 64 KiB; grid within launch limits.

**Workspace:** none, except the three-bf16 form's split pre-pass.

**Catalog** (per stride: the tiles that won, or came within 3%, on a measured shape on any GPU of the family, then systematic grids for the one-tap and stride-1 classes):

| Width stride | Filter widths served | Notes |
|---|---|---|
| 1 | 3, 5, 7, 9, 11 | width dilation where its reach fits a catalog halo (halos sized for dilation 1 and 5) |
| 2 | 1-4 | per row or `flat`; 3x3 and 3x3x3 s2 |
| 3 | 7-9 | vocoder S7 s3 |
| 4 | 1-8 | per row or `flat`; patchify 4x4 |
| 5, 6, 8 | 6-10, 7-12, 9-16 | ConvTranspose1d `k = 2s` |

Counts: 1845 exact configs (WMMA 1020, MFMA 16x16 684, FMA 141) and 84 three-bf16 configs, which the `CONV_IGEMM_NCHW_BWD_TF32` sibling lists (offered by default, reporting `TENSOR_CORE`).

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. Counted over the graphs where K7 is our autotuned best (it is admitted on 258 of the 634 bwd graphs; 205 on gfx906):

| GPU | best on | W/L | call-weighted factor | weighted time saved |
|---|---|---|---|---|
| gfx950 | 64 | 61/1 | 1.61x | 16 ms |
| gfx942 (MI325X) | 49 | 47/2 | 2.00x | 17 ms |
| gfx90a | 48 | 42/4 | 1.83x | 55 ms |
| gfx908 | 48 | 40/4 | 1.40x | 32 ms |
| gfx906 | 24 | 18/5 | 1.61x | 155 ms |
| gfx1201 | 76 | 75/1 | 2.42x | 261 ms |
| gfx1100 | 80 | 78/1 | 2.33x | 338 ms |

- Heaviest wins: nnU-Net 3-D fp32 3x3x3 and 2x2x2 s2 (1.5-4.6x on RDNA, 2.8x on gfx950), HiFi-GAN 1-D upsampling and residual bwd (1.6-3.3x), DeepLab 1x1 s2 pointwise bwd (2.8x on gfx1201).
- Losses are microsecond graphs (call-weighted excess 0.3 ms or less per GPU), except gfx906, where two 3-D fp32 graphs trail `GemmBwdRest` by 1.18x (14 ms).
- Where it is not best: compute-bound large GEMMs. Dilated ASPP routed through K7 measured 0.4-0.5x of MIOpen; the MMA loop (transposing 4-byte LDS stores on both operands, one block per CU, A fragments reloaded per tap, no 32x32 atoms) caps it at 13-20% of peak. K1 and Winograd keep that regime.
- Compile cost: the fp32 FMA stride-8 configs are the slowest conv compiles in the group (10-20 s each); gfx11 WMMA `t11` configs spill.

## Numerical precision

- fp32 accumulation; fp16/bf16 on MFMA/WMMA accumulate in fp32; one rounding at the store. One block writes each dx element once; no split-K, no atomics. The device suite relaunches every config and compares bits.
- Padding and out-of-range taps load zeros and are multiplied; rows no dy row reaches are stored as exact zeros.
- **Three-bf16** (`CONV_IGEMM_NCHW_BWD_TF32`, fp32 graphs): dy and w are split along K into 3K bf16 channels by a pre-pass (high, low, and high-with-infinities-zeroed for the cross term), run on the bf16 matrix-core configs, dx written in fp32. Admitted only on RDNA for 3-D problems with all strides >= 2 and at least 27 taps, the only place it measured faster than exact (1.07x and 1.46x on gfx1201; 0.05-0.79x elsewhere). Infinities stay infinite (tested).
- Tested against hipDNN's CPU reference with a per-term bound times the reduction length plus 2 storage epsilons; three-bf16 against fp64 with a TF32 bound.

## Comparison with MIOpen

| MIOpen solver | Approach | Measured on the graphs K7 wins |
|---|---|---|
| `GemmBwdRest` | GEMM into a column buffer (rocBLAS), then a col2im scatter over dx | K7 1.4-4.4x faster (3-D fp32, 1-D fp16) |
| `GemmBwd1x1_stride2` | 1x1 GEMM plus strided scatter | 1.6-3.0x faster |
| `ConvHipImplicitGemmGroupBwdXdlops` (CK) | grouped implicit GEMM, NCHW handled by transposes | 1.5-2.6x faster (all GPUs) |
| `ConvAsmImplicitGemmGTCDynamicBwdXdlops` | asm implicit GEMM (gfx908) | 1.5x faster |
| `ConvWinoRageRxS`, `ConvWinoFuryRxS` | asm Winograd (Rage on gfx942/gfx950/gfx12, Fury on gfx11) | 2.0-2.7x faster |

K7 wins by never materializing columns or staged layouts: one pass reads dy and the filter and writes dx. It does not compete on wide-channel stride-1 2-D, where MIOpen's CK and Winograd solvers and our K1/K5 are faster.

## Prior art considered

- MIOpen's backward-data GEMM path (GEMM then col2im) as the baseline whose scatter is fused here.
- The stride-phase (sub-pixel) decomposition of transposed convolution, the same idea K1 uses for strided backward data and the depthwise rows engine uses for strided ConvTranspose1d.
- K1's matrix-core atoms and buffer-load helpers are reused unchanged. No code was copied from elsewhere.

## Next steps and deferred ideas

1. **Forward direction for NCHW 1-D** (HiFi-GAN-class forward convolutions): deferred; the same row-window staging applies. Payoff limited to the 1-D NCHW forward graphs where K1's staging dominates.
2. **A high-throughput NCHW matrix-core loop** (no transposing stores, more blocks per CU, A fragments reused across taps, 32x32 atoms) would let K7 compete on compute-bound NCHW; as measured it would be a new kernel, so K1 stays the owner there.
3. **Lift the images x rows <= 65535 grid gate** by folding rows into grid x: graphs past it fall to K1 today. Payoff unmeasured.
4. **Groups and true convolution:** declined today (K1 serves them); both are filter-indexing changes.
5. **fp16 wide-halo FMA configs:** not shipped, because HIPRTC fails to compile the `_Float16` form on gfx942 (bf16 compiles). Add them once the compiler handles it; it matters only where fp16 has no matrix core (gfx906).
6. **Catalog balance:** retune or gate the spilling WMMA `t11` configs and the 10-20 s fp32 FMA stride-8 compiles.

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestConvIgemmNchwProblem.cpp` (host): acceptance; argument forms (width convolution over rows, strided rows); one refusal per graph field and per config field; direction; dtype; grid limits; atoms per architecture; stride and taps follow the filter; flat configs and flat refusals; stride-1 taps shift by the dilation; LDS arithmetic mirrors the kernel; three-bf16 only for fp32 and only where it pays, with the tripled problem and its split.
- `TestGpuConvIgemmNchw.cpp` (device): every admitted config in fp32, fp16 and bf16 over 1-D ConvTranspose shapes (k4 s2 to k16 s8, partial and past-stride padding, wide rows, 70 channels, output extent 1), strided 1x1 and patchify, height- and depth-strided rows, 2x2x2 and 3x3x3 s2 3-D (padded and unpadded), stride-1 k3-k11 with dilation 3 and 5, height dilation; bitwise relaunch of every config; three-bf16 and its infinity cases; catalog invariants (three-bf16 only for fp32 where it pays, knobs distinguish every config, LDS fits, a device reporting no units is served like a counted one).
