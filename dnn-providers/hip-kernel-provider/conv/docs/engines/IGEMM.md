<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_IGEMM_FWD / BWD (K1) — implicit GEMM, the general workhorse for forward and backward data

> Engine ids: `hipkernel:CONV_IGEMM_FWD`, `hipkernel:CONV_IGEMM_BWD` · Generator: `conv/generators/gen_igemm.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/` (`ConvIgemm.cpp`, `ConvIgemmCore.h`) · Pack: `packs/ConvIgemmNative.cpp`, `packs/ConvIgemmProblem.hpp`

## At a glance

- **What:** implicit GEMM on matrix cores (MFMA, WMMA) or FMA micro-tiles, channels-last native, runtime shapes; a config fixes only the atom, the tile and a few code-path knobs.
- **Covers:** fwd and bwd-data; fp32, fp16, bf16 (TF32-class forms in the `CONV_IGEMM_FWD/BWD_TF32` sibling engines); any layout (NHWC native; 1-D channels-first N C L read in place by the `cfx` configs; other packed NCHW by tiled transpose or the `cf` epilogue; anything else by strided copy); 1-D/2-D/3-D; any groups; any stride, dilation and padding; both conv modes; images past 2^31 pixels on large-image configs; patch convolutions (filter equal to the stride, no padding: ViT, ConvNeXt and U-Net patchify) as one GEMM on the patch configs.
- **Arches:** MFMA on gfx908, gfx90a, gfx942, gfx950 (gfx950's double-rate 16x16x32 / 32x32x16 atoms; xf32 on gfx942); WMMA on gfx11 and gfx12; an FMA kernel admitted for every type on every architecture (the only fp32 path on RDNA, the only path on gfx906, gfx103x and gfx1250; the last two are not validated on hardware).
- **Wins when:** compute-bound dense and grouped layers in fwd and bwd-data, NCHW 16-bit backward data, 3-D fp32, and everything on gfx906 (where MIOpen falls to im2col + rocBLAS). It serves every fwd/bwd graph of the 2K subset.
- **Loses when:** gfx90a/gfx908 NCHW fp16 forward vs MIOpen's older XDLOPS implicit GEMMs (1.2-1.4x behind), gfx906 NCHW fp16 forward vs its V4R1 kernel (1.55x), stride-1 3x3 where Winograd (K5) or direct (K2) win, strided NCHW bwd-data where K7 wins.
- **Precision:** fp32 accumulation, one rounding (split-K partials in fp32, summed in split order). Deterministic. No numerical notes on the exact engines; the TF32 siblings report `TENSOR_CORE`.
- **Instances:** 1898 fwd + 1470 bwd exact configs (6 of each the gfx950 flex tile, 15 fwd + 18 bwd the gfx942 register-staged one (6 fwd of them 256 x 256), 9 of those reading the backward-data filter in place; 60 of each the tall 1-wide-vector tiles (256 x 64 and 256 x 128 on the 16x16 atoms, for reductions of a few channels) and their twins; 214 fwd + 126 bwd of them `cfx`, served only on 1-D channels-first graphs), all runtime-geometry (no compile-time shape tier), 37 large-image twins among them, 752 fwd + 491 bwd patch configs admitted only on patch convolutions (44 fwd + 88 bwd of them the patch tile), and 38 col2im backward-data configs; the TF32 siblings list 48 fwd + 45 bwd TF32-class configs (9 + 6 of them patch configs).
- **vs MIOpen:** competes with CK grouped implicit GEMM (`ConvHipImplicitGemm{,3D}Group{Fwd,Bwd}Xdlops`), the older HIP and asm implicit GEMMs, and `GemmFwdRest`/`GemmBwdRest`. Call-weighted factor where K1 is our best: 0.93-1.44x fwd, 1.15-1.47x bwd on matrix-core GPUs; 3.0x / 1.8x on gfx906.

## Why it exists

- hipDNN needed one engine that serves every forward and backward-data convolution fast enough to replace MIOpen's GEMM-class solvers, on every architecture from gfx906 to gfx12, without CK and without per-shape compiles.
- It is the floor of the fast engines: specialized engines (K2 direct, K5 Winograd, K7, K4, SMALLC, STEM_BWD, depthwise) only need to beat it in their regime; anything they decline still has K1.
- Its atoms, buffer-load helpers and element conversions (`ConvIgemmCore.h`) are the shared core that K2, K3, K4, K7, SMALLC and STEM_BWD build on.

## How it works

Per group, forward is the GEMM `y[m][n] = sum_k A[m][k] B[n][k]` with `m` = (image, output position), `n` = output channel, `k` = (tap, input channel), tap outer so a k-step of `BK` channels stays within one tap and every row's tap offset is one add. `A` is never formed: each block gathers its A tile from x with per-row bounds checks.

Backward data runs one GEMM per (group, stride phase): rows are the dx positions of one phase (`(i + pad) mod stride` fixed), columns the input channels, the reduction (taps that reach the phase, output channels). Within a phase the dy gather is the forward gather at stride 1 with an offset and a negative tap step, so the same kernel serves it without zero upsampling, and every dx element is written once. The phases ride on grid z next to the groups.

**1-D channels-first** (`cfx` configs, `K1HaloCf`, `K1TilesCf`): PyTorch's only conv1d layout is N C L, where a channel's positions are contiguous. The matrix-core configs run the halo loop on it: per channel chunk a block loads the run of input pixels its rows' taps reach, from an MV-aligned base (MV = 16 bytes of elements), as one vector per channel and MV pixels (element loads at image and tensor edges), transposes each unit of KV channels x MV pixels in registers, and stores one LDS row per pixel. Every tap then reads its rows at an offset, at any stride, padding and output extent: a row's LDS row is its tap-0 pixel minus the run's base, and a tap outside the row's image reads the zero row. Halo `h` stages up to h input pixels per output row (h 2 for forward stride 2). The FMA kernel reads N C L in place without a halo (each channel's MV positions as one vector where they are one input run, element loads otherwise) and stores the output channels-first itself. Both directions: backward data gathers dy at unit stride in each stride phase, so it always runs halo 1. The filter is staged as for any other config (forward K C S to K S C, backward data transposed); the activations are not.

**Patch convolutions** (patch configs, `withPatchView`): one group, filter extent equal to the stride, no padding, unit dilation, input extents a whole number of patches.
- Forward folds the width: the channels-last input is read as W / S pixels of S C channels and the K T R S C filter as K T R 1 (S C), the same bytes. Each filter row's taps become one contiguous run of the reduction, so the loads are whole vectors (ViT's 16 x 3 = 48-channel runs on 8-wide vectors) where the plain configs gather 3 channels per tap. Only the width must tile exactly; the height can be any geometry.
- Backward data folds every axis: `dx[(n, od, oh, ow)][(t, r, s, c)] = sum_k dy[(n, od, oh, ow)][k] w[k][(t, r, s, c)]`, one GEMM over dy's positions with T R S C columns (`K1_PATCH`: the epilogue writes column runs of S C to dx pixel `(od T + t, oh R + r, ow S + s)`). Every dx element is written once, with no stride phases; N is the whole filter rather than C per phase (3 for RGB stems).
- Patch configs never split K: their GEMMs run in microseconds, where the reduce launch and the partials cost more than an under-filled grid.

**Col2im backward data** (`patch` 3, `withCol2imView`): where dy has fewer positions than dx ("valid" or partly padded layers, small 3-D maps), the GEMM over dx rows multiplies mostly padding: a 3x3x3 layer over small 3-D does 2.8 times the useful multiply-adds there. The col2im configs run the patch view's backward-data GEMM for any filter, stride, padding and dilation: rows are dy's positions, columns the filter's taps x channels in the filter's own order (K T R S C or K C T R S, so both layouts read it in place), the reduction the output channels, and the B operand the graph's filter rows (K-major, so these are direct-filter configs). The GEMM writes its fp32 columns to the config's workspace, and `ConvIgemmCol2im` sums, for each dx element (four channels a thread), the columns of the dy positions it reaches, tap by tap in filter order, then rounds once.
- Served for one group, a packed unreflected K T R S C or K C T R S filter, layers that are not patch convolutions (the patch view writes those with no columns in between), columns within 1 GiB, and only where the GEMM over dy's positions has fewer multiply-adds than the GEMMs over dx rows: dy positions x taps below dx positions x the taps a stride phase reaches. "Same" layers never qualify.
- It sums only the products of dy positions that exist, as the reference defines backward data; the GEMMs over dx rows multiply padding's zeros by the filter.
- Never split: its grids are wide (the filter's T R S C over 128-256 columns a block).
- The patch tile (`patch` 2, `k1PatchMma` in `ConvIgemmPatch.h`) is a dedicated GEMM for the same view: two XOR-swizzled LDS stages with one barrier a k-step (the next step's global loads in flight under the current step's matrix-core work), a forward cursor that walks a thread's 16-byte granule through the filter rows without division, and backward data reading either the staged `[(t, r, s, c)][k]` filter or, with `wdirect`, the graph's K T R S C filter in place: K-major in LDS and read by `ds_read_b64_tr_b16` on gfx950, stored transposed elsewhere. It runs on the 16-bit 16x16 matrix-core atoms of every matrix-core architecture (fp32 patch convolutions run on the patch view of the general loop), needs the read operands within 32-bit buffer offsets and its two stages within the device's LDS, and serves 4- and 8-wide channel vectors.

```
grid = (row tiles, column tiles, groups x stride phases x splits)
block: stage A (gathered) and B (filter) tiles in LDS -> MMA/FMA over the k-steps -> fp32 epilogue
```

## Implementation design

**Kernel structure** (`k1ConvMma`, `k1ConvFma`):
- Matrix-core block of `WM x WN` waves, tile `BM x BN`, k-step `BK`. One LDS buffer per operand: the next k-step's global loads are issued into registers before the current step's multiplies and stored after a barrier. gfx950 `dma` configs instead load straight into a ring of 2-4 swizzled LDS stages.
- Atoms: MFMA 16x16 and 32x32 (wave64), gfx950 16x16x32 / 32x32x16, gfx908 bf16 forms, WMMA gfx11 (both half-waves hold all k) and gfx12 (half-waves split k), fp32 MFMA, and xf32 on gfx942.
- FMA kernel: k-major LDS tiles, each thread a `TM x TN` register tile in runs of 4x4; fp16 multiplies pairs of k with `v_dot2_f32_f16`.
- Flex tile (`flex`, `k1ConvFlex`, gfx950 16-bit): 256 flattened output positions x 256, 192 or 128 channels on eight waves, each owning 64 positions x half the channels (192: whole tiles of 192 and 384 channels).
  - The block's rows run on across planes and images, so no spatial tile is padded.
  - Both operands reach LDS by 16-byte DMA in 64-channel (128-byte) rows: X double-buffered, W triple-buffered. A lane's X rows are fixed for the block and keep their offset at tap 0 plus one in-image bit per tap of each axis, so a step's DMA is a uniform tap delta and one mask test per row.
  - The reduction walks 64-channel slices outer and taps inner, so consecutive steps read shifted copies of the same rows (L1 hits).
  - The two wave groups (the channel halves) ping-pong one barrier apart, with `s_setprio` around the multiplies; X's DMA goes out in a step's first half, W's (one step further ahead) in its second.
  - The epilogue stages each wave's tile in LDS and stores whole 16-byte runs.
- Register-staged flex form (gfx942: 16x16x16 and 16x16x4 atoms, one LDS stage fed from registers a step ahead, `ConvIgemmFlexReg.h`): 256 x 128 on 128-byte steps, and a 16-bit forward 256 x 256 on 64-byte steps (32 channels), which reads a third fewer operand bytes per MMA and serves grids of at least a block a CU (on fewer tiles its splits run twice as deep and measured slower). Forward computes the step after next's load offsets in the store phase and issues the loads one per five MMAs of the second compute phase, so the store phase holds only the LDS stores: 1.03-1.09x on MI300A's 3-D layers, and 1.046x time-weighted (roofline efficiency 49% -> 51%) over the K1-won forward corpus graphs on MI325X, none repeatably slower beyond noise but two small SD UNet layers (0.97x). The 256 x 256 tile measured 1.07x over the 256 x 128 on 256-channel mid-size 3-D outputs. Forward tiles whose rows reach the input at every tap the steps visit, on whole 64-channel slices (unpadded layers, interior tiles), take a load's offset as a per-row base plus the step's uniform delta with no masks: 1.05-1.10x on MI300A's unpadded 3-D layers. Backward data may read the graph's K T R S C filter in place (`wdirect`): a thread loads two 16-bit (or one fp32) filter rows of a 16-byte column granule and stores each column's dword into that column's row of the stage the forward form uses (32 lanes a granule, so a dword store covers the 32 banks once). The per-call transposed copy of the filter is gone, which on wide filters over few positions cost more than the GEMM.
- Epilogue: fp32 accumulators converted once and stored channels-last; `cf` configs write packed channels-first output directly; split configs write fp32 partials indexed like the output. Both store 16-byte output vectors wherever a wave's strip row (channels-first: a run of positions) holds whole ones, whatever the reduction's channel vector, with an element fallback where a run is unaligned or crosses a boundary (patch backward data keeps its vector, which divides its S C runs). A 1-wide reduction vector (two to seven reduction channels) then writes a wide output in vectors, not one element and one decoded output position at a time.

**Knobs** (each a compile-time define; every config field is a knob so candidates are distinguishable):

| Knob | Effect | Served where |
|---|---|---|
| `atom`, `bm/bn/bk`, `wm/wn`, `threads`, `vec` | atom, tile, waves, FMA block, channel vector | `vec` must divide the reduction channels |
| `halo` | stage `BM` + shift-range rows once per channel chunk; every tap reads them at an offset | stride 1, output extents equal input ("same"), shift range within 64 rows |
| `xcd` | XCD-contiguous block order (blocks sharing an A tile land on one L2) | gfx942, gfx950 only |
| `swz` | XOR-swizzled unpadded LDS read in 16-byte chunks (no bank conflicts) | 16-bit, `vec` 8 |
| `dma` 2-4 | gfx950 buffer loads straight into LDS stages | gfx950, 32-bit-addressable problems |
| `wdirect` | bwd-data reads the K T R S C/g filter in place (no transposed copy) | packed KRSC, unreflected filter |
| `pack` 1 / 2 | 1: k-steps of whole taps for groups narrower than `BK`; 2: flat (taps x channels end to end) for channel counts `BK` does not divide; flat configs also run swizzled and with gfx950 DMA (a thread's slot takes its row's swizzled chunk) | as named |
| `cf` | channels-first output from the epilogue | packed NCHW/NCDHW output |
| `cfx` | 1-D channels-first activations read in place (`K1HaloCf` on matrix cores, `K1TilesCf` on FMA) and the output written channels-first; `halo` h stages h input pixels a row | 1-D problems with packed N C L input and output whose block's pixel run fits the halo rows and the device's LDS (`cfxFits`); pointwise unit-stride unpadded channels-first 2-D and 3-D problems as their 1-D line (`withChannelsFirstLineView`) |
| `fuse` 2-16 | run F small groups as one block-diagonal group (filter expanded in the workspace) | fused group <= 64 channels each way |
| `large` | 64-bit positions within an image | only images past 2^31 - 1 pixels (and only them) |
| `patch` | 1: the patch view above on the general loop (forward width fold, backward data one GEMM with the patch scatter); 2: the patch tile; 3: col2im backward data (above) | 1, 2: patch convolutions only; backward data writes channels-last dx with a vector dividing S C. 3: backward data `withCol2imView` serves, direct-filter plain tiles |
| `math` `bf16x3` | three-bf16 TF32 form (the `_TF32` engines only) | fp32 graphs |
| `split_policy` | 0 the analytic split-K plan, 1 none, N (2-64) exactly N splits (`splitPlanFor`) | twins served only where the plan differs from the default's |
| `flex` | the flex tile (above): `bm` 256, `bn` 256, 192 or 128, `bk` 64 (register-staged: 128 bytes at `bn` 128, 32 at its 16-bit forward `bn` 256), 4 x 2 waves, `vec` 16 bytes, `xcd` 1, any `split_policy`, every other knob off but `wdirect` on the register-staged form's backward data | gfx950 (LDS-DMA form) or gfx942 (register-staged form); 32-bit-addressable problems; backward data at unit stride; at most 10 taps along each filter axis; the register-staged 256 x 256 tile on grids of at least a block a CU |

**Host side:**
- Gates: everything the naive parser gates (`parseConvNaiveProblem`, every graph, node and tensor field), plus `N`, `C`, `K`, groups, extents, padding, stride and dilation within 32 bits, row tiles and grid y/z within launch limits. Fwd and bwd only.
- Staging: activations that are not packed channels-last are staged into the workspace (packed channels-first by a batched 64x64 LDS transpose, other strides by a strided copy), except on `cfx` configs, which read 1-D channels-first activations in place. A pointwise convolution (filter 1, unit stride, no padding) on packed channels-first activations is the 1-D problem over each channel's D x H x W pixels as one line, so the `cfx` configs serve it with no staging transpose: forward is one launch, backward data one launch plus the filter transpose. The filter is staged to K T R S C/g (forward) or `[g][tap][c][k]` (backward data) unless `wdirect` applies; true convolution reflects it through negative strides in the same copy.
- Split-K (`splitPlanFor`), by the config's `split_policy`:
  - 0, the analytic default: split only when the tiles under-fill the device: fewer blocks than the CUs hold on gfx9 (2 a CU; for blocks of 8 waves, their resident blocks a CU from threads, LDS and accumulators, `residentBlocksOf`, at most 2; keeping at least 2 k-iterations per split. Smaller blocks measured faster split for 2 than for an occupancy-derived 1), fewer than 1 per WGP on gfx10+ (keeping at least 64, since a split costs about 50 k-steps of partial store there), 1 per CU on gfx1250 (keeping 8); at most 16 splits; partials at most 1 GiB.
  - 1: no split. N from 2 to 64: exactly N splits of an under-filled grid, where each keeps a k-iteration and the partials fit. A filled grid takes no explicit split: its partials would only size the engine's workspace, which is the largest any served config needs (a 154 MB 32-split workspace on a filled 3-channel backward-data grid measured 5-8% slower on gfx950 with the same winning kernel).
  - The gfx9 matrix-core configs that can split have a no-split twin, and the widest tiles (256 x 128 and up) a 32-split twin. The kernel match serves a twin only where its plan differs from the default's, so autotune times distinct plans only.
  - Flex tiles (one block a CU) under policy 0 take the tail split (`flexTailSplit`): the row tiles that fill whole rounds of the device's CUs run unsplit in one launch and the rest split in a second launch whose partials cover only their rows, or every row tile splits, whichever gives the fewest k-steps along the launch rounds, a split block paying a fixed cost (`flexSplitCost`: 20 k-steps for the LDS-DMA form, 14 for the register-staged form). A fused epilogue keeps the all-or-none split, since the reduce applies it over the whole output.
  - A reduce kernel adds the partials in split order.
- Addressing: buffer loads with 32-bit offsets from a per-block 64-bit base (out-of-range reads return zero, so padding needs no branch); `K1_BUFFER=0` switches to 64-bit global loads when a block's rows or the filter could span 2^31 bytes. A grid past 2^32 work items runs as consecutive launches (`tileBase`).
- Occupancy: the swizzled 128x128x64 WMMA tile with 4x2 waves gets `waves_per_eu` 4 on gfx1100, gfx1101 and gfx1151 (1.5x VGPR file), where the compiler otherwise spills it and runs 1.6x slower.
- Workspace: staged operands plus split partials (col2im: its fp32 columns, M x T R S C); nothing when the graph is NHWC/KRSC (fwd) or NHWC with `wdirect` (bwd) and unsplit. On `cfx` configs only the filter (forward K C S to K S C; backward data transposed) and split partials.

**Notable codegen choices:**
- Buffer descriptors pass through `readfirstlane`. Without it the current HIPRTC keeps a descriptor derived from a 64-bit division in VGPRs and wraps every load in a waterfall loop (the pattern appears in 93-95% of K1 code objects compiled without it and costs about 7% geomean on gfx950 and gfx1201).
- LDS-only barriers in the matrix-core kernel: `__syncthreads()` on gfx12 adds an L0 invalidate after each barrier.
- Branch-free `&` load predicates, a decode-once cursor stepped by additions, and magic division for the flat (`pack` 2) cursor (32-45% faster on C=3 stems than a subtract loop).
- bf16 stores use the hardware conversion on gfx950, software round-to-nearest-even elsewhere.
- RDNA family macros (`ConvArchFamily.h`) so gfx11-generic and gfx12-generic compiles keep their WMMA atoms.

**Catalog:** per atom family a handful of tiles (for example MFMA 32x32: 128x128, 256x128, 128x64, 64x128, 256x256; MFMA 16x16: 128x64, 64x64, 128x32, 256x16 and a `vec` 1 tile; WMMA: 13 tiles plus direct-filter and tap-packing variants; FMA: 3 per dtype), then generated twins: `flat`, `fuse`, `swz`, `cf`, `large`. Each config is admitted on correctness alone; priority (larger matrix-core tiles first, FMA last) is the declared order out of the box, and autotune picks.

## Fusion

`CONV_IGEMM_FWD` and `CONV_IGEMM_FWD_TF32` apply a fused pointwise epilogue
([conv README](../../README.md#fused-pointwise-epilogues)) in every path that writes y:

- the LDS-staged row epilogue and the channels-first (`cf`) epilogue, at the accumulator, before
  the strip; the strip and the staging transposes stay bitwise copies of the rounded value;
- the FMA kernels' direct store;
- split-K: the main kernel writes its fp32 partials unchanged and `ConvIgemmReduce` applies the
  chain once to each sum, walking the output's channels (channels-last, or the in-place
  channels-first output) with additions only;
- group fusion, halo, tap packing, DMA, large-image and three-bf16 / xf32 configs alike: the
  channel is `g * (K / groups) + column`.

The per-channel operands load once per output column a thread owns; columns past K load
nothing. Backward data takes no epilogue.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.10x, gelu 1.19x, silu 1.14x; gfx950 (both layouts) relu 1.14x, gelu 1.24x, silu 1.16x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.01x on MI325X and 1.05x on gfx950 (usage-weighted; the conv README's fusion table). Its losses are 1x1 layers whose bare convolution a BLAS GEMM wins (down to 0.60x on gfx950, channels-first 1x1 with a 384x384 image): the fused graph has no faster form there than the BLAS GEMM followed by a separate pass.

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. Counted over the graphs where K1 is our autotuned best; W/L = faster / slower by more than 5%; factor = call-weighted MIOpen time / K1 time.

| GPU | fwd best on | fwd W/L | fwd factor | bwd best on | bwd W/L | bwd factor |
|---|---|---|---|---|---|---|
| gfx950 | 180 | 99/47 | 1.21x | 102 | 80/9 | 1.26x |
| gfx942 (MI325X) | 205 | 120/47 | 1.13x | 96 | 69/15 | 1.17x |
| gfx90a | 228 | 62/125 | 0.93x | 122 | 64/30 | 1.15x |
| gfx908 | 303 | 68/183 | 0.93x | 140 | 73/45 | 1.18x |
| gfx906 | 380 | 245/122 | 2.99x | 165 | 126/30 | 1.76x |
| gfx1201 | 220 | 132/49 | 1.29x | 116 | 89/18 | 1.47x |
| gfx1100 | 203 | 123/43 | 1.44x | 129 | 102/12 | 1.43x |

- **Largest wins** (weighted time saved): NCHW fp16 dense bwd vs CK grouped bwd (1.2-1.6x, every GPU), NCHW fp16 dense fwd vs CK grouped fwd (1.3-1.4x on gfx950/942/RDNA), 3-D NCHW fp32 fwd/bwd vs CK 3-D or `GemmFwdRest` (1.15-2.5x), and gfx906 bf16, NHWC and NCHW fp16 bwd layers vs `GemmFwdRest`/`GemmBwdRest` (1.5-5x).
- **Largest losses:** gfx90a NCHW fp16 fwd vs `ConvHipImplicitGemmForwardV4R4/V4R5Xdlops` (1.2-1.4x); gfx908 Wan-VAE 3-D NHWC bf16 fwd vs CK 3-D (1.08x, but the heaviest weighted loss of any engine there) and NCHW fp16 vs the asm `ConvAsmImplicitGemmGTCDynamicFwdXdlops` (1.18x); gfx906 NCHW fp16 fwd vs `ConvHipImplicitGemmV4R1Fwd` (1.55x over 65 graphs).
- **1-D channels-first (`cfx`)**, against the same build without `cfx` configs (both autotuned, fresh cache, 20 iterations, interleaved, our engines only) on the 450 1-D NCL graphs of the internal corpus (HiFi-GAN, Conformer; fp16/bf16, stride 1): usage-weighted 1.061x on gfx950 (fwd 1.090x, bwd 1.034x), 1.099x on gfx90a (fwd 1.139x, bwd 1.063x), 1.051x on gfx1101 (fwd 1.054x, bwd 1.049x). The `cfx` configs are the best engine on 224 (gfx950), 294 (gfx90a) and 162 (gfx1101) of the graphs, which then stage no activation (12-19 GB of per-call staging workspace summed over those graphs). Kernel level (gfx950): stride-1 halo `cfx` is 1.0-1.12x the staged path. Width stride 2 (Whisper conv2, wav2vec2 k3/k2 s2) runs 0.70-0.91x, so autotune keeps the staged configs there. On 24 Whisper, wav2vec2 and HiFi-GAN layers in fp16 and fp32 (fp32 runs the FMA `cfx` configs on RDNA) the same A/B gives 1.015x on gfx950 and 1.31x on gfx1101, the strided layers within 4-7% either way.
- Call-weighted excess over MIOpen on K1's losing graphs (per run of every captured model): 1.4 ms gfx950, 3.4 ms gfx942, 14-19 ms RDNA, 54 ms gfx90a, about 630 ms each on gfx908 (mostly the Wan-VAE 3-D layers) and gfx906 (NCHW fp16 fwd). Against that, K1's wins save 48 ms (gfx950, gfx942) to 660 ms (gfx1100).
- **Compile cost:** a graph admits many K1 configs; the worst measured graph (ResNeXt-50 grouped NCHW bwd) admits 91-131, and compiling them all for autotune takes 134 s on gfx942 and 224 s on gfx950. Median cold compile is 0.3-0.8 s per config on CDNA, 0.4-1.5 s on RDNA.
- Against straight ports of the same configs to rocKE and FlyDSL (12 forward NHWC shapes, gfx950/gfx942/gfx1201), K1 is within about 1% of rocKE on gfx950, at parity on gfx942 and 2-4% behind on gfx1201; the C=3 stem (1.4x) and BK-64 fragment scheduling on gfx1201 remain.

## Numerical precision

- fp32 accumulation on every path: fp16/bf16 products on MFMA/WMMA accumulate in fp32; fp32 runs on fp32 MFMA (exact products) or FMA; one rounding to the output type. Split-K writes fp32 partials and adds them in split order before that rounding.
- Deterministic: fixed tile order, no atomics; the device suite relaunches every admitted config and compares bits.
- Tested against hipDNN's CPU reference with 2e-6 per reduction term (times the reduction length) plus 2 storage epsilons; TF32 forms at 1e-3 (xf32) and 3e-5 (three-bf16) per term.
- **TF32 forms** (`CONV_IGEMM_FWD/BWD_TF32`, offered by default, fp32 graphs only): three-bf16 on every matrix-core arch (operands split into hi + lo bf16 at the LDS store; lo·hi, hi·lo and hi·hi accumulated in fp32; about 2^-16 per product), and xf32 on gfx942 (operands rounded to nearest-even TF32 at fragment load, not truncated by the hardware). Infinities stay infinite in both. Measured against the best exact config on top fp32 shapes: xf32 1.07-1.27x faster on MI325X (7 of 8 shapes); three-bf16 0.5-0.7x on gfx950 and gfx942. The siblings report `TENSOR_CORE`, cuDNN's note for TF32, so a caller excludes them through the hipDNN API's cuDNN-compatible `deselect_numeric_notes`; the exact engines list no TF32 config.
- **`cfx` configs** run the same reduction order and atoms as their staged twins: on the bench, outputs are bitwise identical to the staged path with the same tile.
- **inf/NaN:** ±inf inputs give the fp32 result on native and TF32 configs (tested), except group-fusion configs: the zero blocks of the expanded filter turn another group's infinity into NaN (0 x inf). Those configs are excluded from the infinity test; it is an open parity item. Padding is read as zeros and multiplied, as in a zero-padded reference. Col2im instead sums only the products of existing dy positions (the reference's definition), so an infinite filter tap gives no NaN at dx elements whose taps reach past dy; its suite plants infinities in dy and, separately, in the filter.

## Comparison with MIOpen

| MIOpen solver | Where | Approach | Measured (subset, where K1 is our best) |
|---|---|---|---|
| `ConvHipImplicitGemm{,3D}Group{Fwd,Bwd}Xdlops` (CK) | gfx90a+, RDNA | AOT template instances per tile, grouped GEMM | K1 1.2-1.6x faster on NCHW fp16 bwd and fwd on gfx950/942/RDNA; 1.1-1.2x behind on some gfx90a/908 fp16 shapes |
| `ConvHipImplicitGemmForwardV4R4/V4R5Xdlops`, `ConvAsmImplicitGemmGTCDynamic{Fwd,Bwd}Xdlops` | gfx908, gfx90a | older HIP and hand-written asm implicit GEMMs | K1 1.2-1.4x behind on NCHW fp16 fwd; ahead on bwd (asm GTC bwd 1.23x) |
| `ConvHipImplicitGemmV4R1Fwd`, `V4R4Fwd` | gfx906 | VALU implicit GEMM | K1 FMA 1.25-1.55x behind on NCHW fp16 fwd and 3-D fp32 |
| `GemmFwdRest`, `GemmBwdRest` | all | im2col/col2im + rocBLAS | K1 1.5-5x faster where they are MIOpen's best (gfx906, 3-D, RDNA) |

K1 differs in kind: one source compiled at first use for the device in hand, runtime shapes (no per-shape compile), bwd-data by stride phases rather than col2im, NCHW served without MIOpen's transpose solvers, and three dtypes on every arch including fp32 and bf16 where MIOpen has no matrix-core path.

## Prior art considered

- **rocKE's implicit GEMM:** same main loop as K1; its dgrad techniques were taken: decode once per tap, halo rows staged once per channel chunk (`halo`), XCD-contiguous block order (`xcd`), occupancy-driven 4x1 tiles, no two-stage LDS on 64 KiB parts.
- **hipconv:** its swizzled LDS tiles and its merging of small groups suggested `swz` and `fuse`.
- **CK:** studied as the reference for tile shapes and pipelining; not linked, ported or used.
- **rocKE and FlyDSL straight ports of K1**: comparing their ISA with K1's identified the waterfall descriptor, the gfx12 barrier invalidate, branchy predicates and the stem cursor as HIP-side costs; K1 avoids all four (see Notable codegen choices). No code was copied.

## Next steps and deferred ideas

1. **Patch GEMMs at microsecond scale** (gfx950, NHWC bf16, autotuned, same GPU): with the patch tile, ViT-B/16 runs 1.6-1.8x (forward) and 1.3x (backward data) faster than the previous best engine, ConvNeXt 2x2 backward data 1.5x and the ConvNeXt 4x4 RGB stem 1.1-1.35x. MIOpen's hipconv patch kernel stays 1.1-1.45x ahead on these graphs: at the same 64 x 64 x 64 tile its kernel runs 1.35x faster on ViT-B/16 batch 16 (11 against 15 us), so the remaining gap is the inner loop (load and LDS-read scheduling against the matrix-core work), not the launch count or the tile shape; the patch tile's 256-row tiles also trail the general loop's at batch 128.
2. **Fragment read-ahead for BK-64 tiles** (load a k-step's fragments before its MMA loop): the largest same-config residual to the rocKE port (1.20x on gfx1201 BK-64). Gain bounded by that residual on the BK-64 picks.
3. **gfx90a/gfx908 forward pipeline** (more work per barrier, deeper load/LDS/MMA overlap) for NCHW fp16 and the Wan-VAE 3-D bf16 layers, the heaviest K1 losses on those GPUs. RDNA pipelining of the same kind (ring, register prefetch) measured no gain, so this needs a structural change.
4. **gfx906 FMA forward for NCHW fp16** (1.55x behind V4R1 on 65 graphs, the largest K1 weighted loss anywhere), beyond the `v_dot2` fp16 path the FMA kernel has.
5. **Out-of-the-box selection and compile cost:** the declared order takes the first admitted config; a native per-config scorer from tile fit, waves, split plan and occupancy, plus a precompiled code-object store, would avoid compiling 90-130 candidates on grouped graphs.
6. **Group-fusion inf parity:** mask the structural zeros of the expanded filter, A/B'd against the compiled-out build.
7. **Split knob:** expose the split count so autotune can override the analytic plan on shapes where it guesses wrong.
8. **Three-bf16 on CDNA:** it does not pay there (0.5-0.7x) and stays only as an autotune candidate in the TF32 siblings; a faster split would make it worth selecting.
9. **Workspace limits:** a per-family test that a caller's workspace cap declines at plan time, never at execute.
10. **Strided channels-first halos:** at width stride 2 the halo loads and transposes two pixels per output row (each used about 1.5 times for a 3-tap filter) and measures 0.70-0.91x of the staged path; splitting even and odd pixels into separate halo halves measured no gain. Candidates: gfx950's transposing LDS reads (raw channel rows by DMA, `ds_read_b64_tr_b16` at the fragment read, as K2 `rows cf`), so the transpose costs nothing.
11. **2-D/3-D channels-first input:** `K1TilesCf` and `K1HaloCf` are written for any spatial rank; the kernel match admits 1-D only. The per-tap in-place gather (`K1TilesCf` on matrix cores) measured 0.4-0.8x of the staged path on 1-D compute-bound shapes (unaligned shifted vector loads), so 2-D/3-D needs the halo form.

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestGpuConvIgemm.cpp`, device: every admitted config of each case in fp32, fp16 and bf16, fwd and bwd: NHWC/NCHW 3x3, small groups, grouped stride 2 (both layouts), 1x1, small channels, 1-D dilated and strided (k4 s2, k8 s4), NCDHW/NDHWC 3x3x3, asymmetric padding with dilation and stride, convolution mode, groups of 8 and 4 channels, stride 2 to an output of extent 1, and patch convolutions (4x4 and 16x16 RGB, 2x2 NHWC and NCHW, 2x2x2, 1-D, convolution mode, a width the stride does not divide). Each config is relaunched for bitwise determinism. TF32 cases (fp64 reference, TF32 bounds) and ±inf cases for TF32 and native configs; ±inf through the patch configs alone in all three types.
- Same file, `TestGpuConvIgemmCol2im` (col2im configs alone, fp32/fp16/bf16 where a matrix atom takes the type): 3-D "valid" in both layouts, padding on one side, stride 2, dilation 2, six channels in both layouts, 1-D; each plain, with ±inf in dy and with ±inf in the filter. `deviceCases` also runs 1x1 layers of six channels on one side in both layouts (1-wide reduction vectors). Host: the col2im view's shape and workspace, one decline per field (same padding, groups, patch convolutions, true convolution, one tap, columns past 1 GiB), and the col2im configs' catalog shape; the flex catalog test admits the direct filter on the register-staged form only.
- Same file, host: the patch view (forward width fold, backward data GEMM shape and filter staging, one decline per non-patch field, large images), patch configs served only on patch convolutions and never split; catalog invariants (atoms known and run only where their instructions exist; knobs distinguish every config; halo configs single-stage, no two-stage tiles; `wdirect` bwd only; `cf`, `flat`, `swz`, `dma`, `fuse` twins and their gates; `xcd` only on multi-XCD devices), split plans (gfx9 below 2 blocks per CU, RDNA below 1 per WGP with 64 k-steps, no units no split), buffer span, large-image configs exactly past 2^31 pixels, occupancy target, multi-launch grids, compute types.
- Same file, `cfx`: device suite `TestGpuConvIgemmChannelsFirst1d` (only `cfx` configs, at least one per case; fp32/fp16/bf16, ±inf in the input; fwd and bwd) over N C L "same" k3, a height-1 2-D graph, dilated k11, "valid" k7, stride 2 k3 and k2, pointwise, groups, an output of extent 1, a row shorter than a vector, convolution mode; host: the channels-first input/output predicates and staging, `cfxFits` (halo rows, LDS, negative image gaps, rank), configs served only on 1-D channels-first graphs, catalog shape of the `cfx` twins.
- `IntegrationConvCoverageCensus.cpp`: fwd/bwd tiers per architecture and gate-boundary graphs, including large images.
