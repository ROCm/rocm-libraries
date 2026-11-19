<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_WRW (K3) — backward weights as a row-direct implicit GEMM, deterministic split reduction

> Engine id: `hipkernel:CONV_WRW` · Generator: `conv/generators/gen_conv_wrw.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/wrw/kernels/` (`ConvWrw.cpp`, `ConvWrwCore.h`, `ConvWrwDirect.h`, `ConvWrwGeom.h`, `ConvWrwArgs.h`) · Pack: `packs/ConvWrwNative.cpp`, `packs/ConvWrwProblem.{hpp,cpp}`

## At a glance
- **What:** dW[k][c][tap] = Σ dy·x over images and output positions, as an implicit GEMM. A block owns all filter-width taps of its filter rows in registers and stages each input window once for all of them. Nine loop bodies (`loop` 0-8) cover strides, rings, stems, narrow maps, gfx950 DMA / direct, exact-fp32 FMA, and a runtime-geometry generic tier.
- **Covers:** backward weights. fp32, fp16 and bf16; a three-bf16 fp32 form in the `CONV_WRW_TF32` sibling engine, on the classes where it measured faster. x and dy both packed NCHW or both packed NHWC; the generic tier also reads any strides. dW is packed K C T R S or K T R S C. 1-D, 2-D and 3-D, groups, any stride, dilation and padding, both convolution modes.
- **Large tensors:** the batch must fit 32 bits. Images past 2 GiB run on block-based-buffer configs (`pb 1`).
- **Arches:** MFMA on gfx908-gfx950 and WMMA on gfx11/gfx12 (K1's atoms). FMA loops give exact fp32 on RDNA, and serve every dtype on gfx906. gfx950 extras: LDS DMA ring, LDS transpose reads, and the hipconv-class direct loop.
- **Wins when:** most wrw on RDNA (MIOpen falls to universal GEMM or naive there), NHWC bf16 3x3 / 3x3x3 (VAE, Wan VAE), vocoder 1-D, small-C stems, dilated rows, narrow ResNet / Inception maps (packed rows), pointwise on RDNA and gfx950.
- **Loses when:**
  - gfx90a NHWC bf16 VAE 3x3 trails universal GEMM / CK by 1.11-1.28x;
  - deeplab ASPP at large dilation on 65² maps and on gfx906/908;
  - NCHW 1x1 fp16 on gfx942 / gfx90a, where MIOpen rounds per image;
  - hifigan 1x16 stride-8 on gfx950.
- **Precision:** fp32 accumulation. Split partials are fp32 and summed in a fixed order; dW is rounded once. Deterministic, no atomics, no numerical note (`CONV_WRW_TF32` reports `TENSOR_CORE`).
- **Instances:** 8,050 configs, 882 of them three-bf16, in per-dtype, per-arch-group packs (`cdna`, `cdnapk`, `gfx950` (3-wide filters) / `gfx950b` (other widths), `direct`, `rdna11`, `rdna12`, `fma`, `fmardna`, `fma906`/`fma906b`, `generic`, `genericfma`, `x3*`), each under the repository's 500 KB file cap. Two tiers: specialized (geometry compile-time) and generic (`loop 8`, runtime geometry).
- **vs MIOpen:** usage-weighted speedup on the subset graphs where K3 is our fastest engine: 1.67x gfx950, 1.43x gfx942, 1.33x gfx90a, 1.79x gfx908, 4.50x gfx1201, 17.95x gfx1100. MIOpen ran in search mode.

## Why it exists
- MIOpen's wrw is split across CK grouped implicit GEMM (NHWC only, NCHW through 3-5 transposes, atomic split-K), im2col + rocBLAS (`GemmWrwUniversal`), hipconv `direct_wgrad` (gfx950, NHWC 16-bit), and legacy asm/OpenCL kernels. On RDNA and old CDNA it often falls to universal GEMM or naive, at 10-1000x our time.
- One HIP engine covers every layout, dtype, rank and stride on every arch, reads NCHW in place (positions are already the reduction axis), and stays deterministic. CK's and rocKE's split-K use atomics.
- The `loop 8` generic tier exists so the long tail (any filter, stride, dilation and strided views) beats the BLAS / naive fallback without growing the specialized catalog.

## How it works
Per group, filter row (t, r) and width tap s: D[k][c] += A[k][q]·B[c][q]. A is a dy row chunk (BK x BQ positions), B is the x window under it, and the reduction runs over output positions q.

**Width.** The window a chunk's taps reach, WIN = (BQ-1)·sw + (S-1)·dw + 1, is staged once and split into sw stride phases. Tap s reads phase (s·dw) mod sw, shifted by ⌊s·dw/sw⌋. Odd 16-bit shifts are built from aligned dword reads and `v_alignbyte`.

**Loops** (the `loop` knob):

| loop | name | what a step is | serves |
|---|---|---|---|
| 0 | output rows | one output-row chunk with FR filter rows' windows | any stride |
| 1 | row ring | one input row; the last R dy rows' fragments stay in registers, so one window feeds all R x S taps. With `wq` 2 or 4, a stage of that many steps of fewer positions between barriers | unit height stride (any dilation); height stride > 1 as one ring per phase; `wq` stages for output rows within a step's positions |
| 2 | DMA ring | loop 1 with 16-byte LDS DMA and 2-4 stages | gfx950, unit width stride |
| 3 / 4 | FMA ring / FMA rows | loops 1 / 0 on FMA thread tiles (TK x TC for all taps) | exact fp32 on RDNA; every dtype on gfx906 |
| 5 | direct | hipconv `direct_wgrad` class: a 32-column strip of one image plane over its rows, dy rows in a register ring, x read at every width tap | gfx950 16-bit NHWC, unit width stride and height dilation |
| 6 | small-C rows | loop 0 with the GEMM's N axis the (input channel, width tap) pairs | stems, ≤ 16 input channels a group |
| 7 | packed rows | several narrow output rows of one plane in one step, each padded to the taps' reach | narrow channels-first outputs (7², 14² s2, 35² s2) |
| 8 | generic taps | runtime filter, strides and dilations; dy chunk once, then the x positions each tap reads | the long tail; strided views; with `flat 1`, output rows shorter than a step |

**Width fold.** A width stride s that divides the filter width S (unit width dilation, one group, packed x and dy) is the same wrw as a stride-1 one: position u s + j of input channel c is folded channel j C + c at position u (`widthFoldOf`). The folded problem has C s input channels at unit width stride; depth and height are unchanged.
- Channels-first x is staged into the workspace by `ConvWrwFoldX` (one more launch: a tile of one row through LDS, read and written contiguously), shifted by the padding's phase r = p mod s and zero outside the row. The folded filter is then exactly S / s taps at padding (p - r) / s over ceil((L + r) / s) positions, for any width L.
- Channels-last x is already the folded layout (its width stride becomes s C) and is read in place, for widths s divides: ceil((S - p) / s) + ceil(p / s) folded taps at padding ceil(p / s), the ones reaching outside the filter computed and dropped.
- Specialized configs of unit width stride whose class is the folded one serve the graph through the fold; graphs whose own class has specialized configs keep them too, and autotune ranks both. HiFi-GAN's transposed-conv wrw (1x16, stride 8, padding 4) runs on the 1x2 configs (channels-first) or the 1x3 ones (channels-last) over C x 8 channels.
- `ConvWrwReduceFold` sums the folded partials in the quad reduce's order and writes folded (channel j C + c, tap tau) to dW tap (tau - pre) s + j + p - r of channel c, reflected for true convolution. Every dW element is exactly one folded element, so the kernel never writes dW itself (no `dws`).
- Measured (autotuned, fp16): HiFi-GAN n32 c128 k256 1x25600, 1x16 stride 8, 1.22 -> 0.29 ms on gfx950 (staging 95 µs, the 1x2 wrw 196 µs, the fold reduce 12 µs); its 1x16 layers 2.9-4.1x faster on gfx950 and 2.5-3x on MI325X. ViT 16x16 patch wrw on the WRW engine (channels-last, in place): 21 -> 0.45 ms on gfx950.

**Ring tracks.** The ring walks tracks: (image, output depth, height class ρ mod dh, width chunk). A track ends at its last step that has both an x row in the image and a dy row for some tap. Trailing pad-only steps are trimmed: at dilation 36 on 128 rows they are a third of the steps.

**Split and reduce.** Steps split evenly over grid z to fill the dispatch units. The count comes from `splitsWanted`: units x `bpu` blocks per unit, at most steps / 4, workspace ≤ 1 GiB, then the count that best fills the last round.
- Each split writes fp32 partials [split][g][tap][k][c].
- `ConvWrwReduce` adds them in a fixed order: 4 split groups of every-4th split, then the groups in order. It writes dW in its dtype and layout, reflected for true convolution. Its loads are non-temporal on gfx9 (plain on gfx950 at 64 or more splits) and plain on RDNA, where a non-temporal load misses the infinity cache holding the fresh partials (3x on gfx1201).
- K C T R S filters of more than one tap at up to 8 splits take the tile reduce (`ConvWrwReduceTile`): a block sums a tile of output by input channels with all taps in the same order and writes each output channel's run contiguously. The quad blocks' writes land `taps` elements apart, which cost the c512 3x3 reduce 6-9 µs on gfx942 and gfx950. At one split on gfx950 the row reduce (one output channel's run of 64 channels a block) runs instead; it measured 3-4 µs ahead there.
- From 32 splits into a quad grid that fills at most a third of the units, the quad reduce runs as `ConvWrwReduceDeep`: the same association (so the same bits), 16 loads a thread in flight instead of 4, and 64 quads a block halved down to 16 while the blocks leave a unit fewer than two. There the few waves wait on load latency: c32 k32 3x3 at about 500 splits ran 36 blocks for 14 µs on 18 MB of partials on gfx950, and the deep reduce took the layer 1.04-1.06x faster. Wider quad grids are bandwidth-bound and keep the quad reduce (deep ran them 1-3% slower).
- At one split into K T R S C (or one tap), `dws 1` configs write dW directly: no workspace and no reduce.

## Implementation design
- **Layouts.**
  - `layout 0`: NCHW read in place. `align 1` units start on vector boundaries, so a unit is wholly in or out of a row.
  - `layout 1`: NHWC transposed into LDS on the store. `align 2` reads dy at any element, for output channel counts that are not whole vectors: VAE 3-channel heads (3x3), 1x1, and HiFi-GAN's one-channel 1x7 output layer.
  - `layout 2`: NHWC kept channels-last on gfx950, with MFMA fragments by `ds_read_b64_tr_b16`.
  - The channels-first LDS pitch is an odd number of 16-byte units at unit width stride, which removes bank conflicts (deeplab d24).
  - The matrix-core ring's channels-last rows on CDNA (`k3ClPitch`) are an odd number of fragment-read-wide dword groups: 2 x odd dwords for 8-byte reads (16-bit), odd dwords for fp32, 4 x odd for gfx950's 16-byte reads. The 16 rows of one 8-byte read then land on distinct bank pairs, where the 24-dword pitch of a 34-position window put them on 4. In a bank model of the 64 x 64 x 32 bf16 tile the x-fragment reads go from 7.3 to 1.8 cycles per read and the dy reads and stores from 2 to 1. RDNA and the output-rows loop keep their pitch.
- **gfx950 direct loop (5).**
  - 8 waves; 16-byte DMA into 2-5 separate `__shared__` ring slots, with column-rotated channel groups. The rotation is chosen by a compile-time conflict check of the transpose-read lane map, and the DMA fetch address is its inverse.
  - 8-wave ping-pong whose memory phase is transpose reads plus DMA issue.
  - A persistent grid of one workgroup per CU, walking (tile, item) units in XCD-contiguous order, optionally `wq` items side by side.
  - `directPlanFor` picks the walk (cells or equal runs), the row segments (1-64, minimizing modelled busiest-workgroup time plus partial traffic) and the splits.
  - What each piece is worth on Flux n8 c256 k128 768², in µs, each line adding one piece to the one before: ring loop 2317; direct structure 2204; swizzle, DMA and transpose reads 1927; ping-pong 1874; XCD order 1833.
- **Row ring (loop 1).**
  - The grid walks XCD-contiguous order on every arch (`k1XcdOrder`, a no-op on one XCD): the channel tiles and depth taps of one split, which read the same x and dy rows, share an L2.
  - The cursor's rows, offsets and tap-row mask advance by additions within a track; the division and the buffer resources are recomputed only where a track starts.
  - Stages (`wq` 2 or 4): each step of a stage has its own registers and LDS slot, on threads rotated by THREADS / `wq` so the waves share the stores, and the stage multiplies its steps in order behind one barrier. A step's positions are a whole output row of narrow maps (`bq` 16 or 8 at `wq` 2 or 4, admitted only where an output row fits a step), so the atoms see no zero positions. The packed-rows loop (7) pads each row to the taps' reach and stages every filter row; the flat generic loop gathers each tap.
- **Host gates.**
  - Problem gates, everything the naive parser gates, plus: fp32/fp16/bf16; x/dy layouts as above; dW packed; batch and taps within 32 bits; per axis the reach of the 32-bit position math, (out + 512)·stride + filter·dilation + padding, within 32 bits.
  - Config gates: `wrwLaunchFor` reads and gates every config field — atom on device and type, loop rules, layout, tile against atom, vector, s/sw/dw equal to the filter's, fr, xcd, align, ns, wq, dws, pb, bpu. LDS is gated by the kernels' own `k3LdsBytes`, and load passes must fit their 32-bit mask.
  - Flattened positions (`flat 1`, loop 8): a step is `bq` consecutive positions of the flattened (image, output depth, row, column) order instead of one output row's chunk, so maps whose output rows are shorter than a step (small spatial, wide channels: 8 x 8 and 16 x 16 rows of 3-D volumes) fill the step instead of padding it. Each position is decomposed by magic division; x and dy are each one buffer, so flat configs are admitted only where the whole tensors and the positions stay within the 32-bit range, and only where an output row is shorter than `bq`. They serve beside the specialized classes (the two-tier decline does not apply). On a 1024-channel 3x3x3 layer with 8 x 8 output rows they run 1.8x faster than the row ring on MI325X; the BLAS path remains faster there (the generic loop's per-tap gathers cost about 2.5x the ring's time on large maps).
  - The generic tier declines wherever a specialized class (its `spec` key list) serves the graph. It declines ASPP-class filters (≤ 9 taps, dilation ≥ 6) on gfx942 and gfx90a (generic against BLAS on deeplab d12-d36 and SSD d6: geomean 0.64x and 0.73x), with gfx12's FMA form (0.4-0.7x), and on the architectures not measured (gfx950, gfx908, gfx906). It serves them on gfx11 (0.9-12x on gfx1101) and on gfx12's matrix atoms (16-bit, 0.8-3.3x, geomean 1.5x on gfx1201). The 3x3 deeplab and SSD dilations have specialized configs in both layouts, so the decline leaves only other sparse filters to BLAS. It declines narrow groups (fewer than 16 input and fewer than 16 output channels a group: depthwise, ResNeXt-class), which the vector-ALU wrw engines serve 14-5000x faster on gfx1101, and it declines stems (fewer than 8 input channels a group) and output rows of fewer than 4 positions, where BLAS measured 2.5-30x faster. Its FMA form also declines width strides ≥ 4.
- **Large images.** `pb 1` configs start their x and dy buffers at the block's first channel (channels-first loops 0, 1, 6) or at each step's first readable element (generic). An image may then pass 2^31 bytes as long as one block's planes or windows do not. These configs are admitted only for such images. Measured: 257³ stride 2 on gfx950, 2.08 ms against MIOpen's 6.70.
- **Three-bf16 (`math 1`).** `ConvWrwSplit` writes x and dy as 3N bf16 images: x as [hi | hi' | lo] and dy as [hi | lo | hi'], where hi' is hi with infinities zeroed (the copy that meets the other operand's lo), so an infinity stays infinite instead of becoming NaN. The tuned bf16 config then runs on the 3N-image problem, and the reduce writes fp32 dW. `threeBf16Pays` admits it only where it measured faster than exact fp32 K3:
  - gfx942 and gfx950: unit stride, ≥ 9 taps, ≥ 128 input and ≥ 64 output channels a group;
  - gfx11 and gfx12: stems (≤ 4 input channels), or ≥ 256 channels with ≥ 9 taps;
  - never on gfx90a.
- **Catalog.** Families × filters: atoms per dtype × tiles × loops × layouts, crossed with a measured FILTERS list. That list covers 3x3 (dilations 2, 4, 6, 12, 24 and 36 in both layouts, on the ring and the output rows), 1x1, 1-D 1x3-1x11 at dilations 1/3/5 (in both layouts), stride-2 and 1x16 stride-8 rows, 7x1/3x1/5x5, packed rows, 32-channel-group rings and 5x5x5. Extra tiles cover measured winners a family's list lacks.
  - Catalog breadth: neighbouring geometries of the top configs (halved or doubled blocks, 32/64 positions, `bpu`), found by a sweep around the shipped winners. They measured +10% on gfx950 and +17% on gfx1201 on a 12-shape set.
  - Wide pointwise tiles: 256x128 on gfx950, and RDNA ring tiles for long reductions.
  - gfx11 channels-last rows the channels-first-only filter rows leave out: 5x5 on the output rows and the 32 x 32 ring for 32-channel groups.
  - `bpu` 1/2/4/8 variants exist so autotune can override the split model.
- **Codegen choices.**
  - Every barrier is LDS-only on gfx12.
  - Per-lane load predicates use bitwise `&` (fewer spills, +1.4-2.3%).
  - Channels-first edge fix-ups shift a whole 16-byte run as one 128-bit integer, then mask. Its share is +9-13% on narrow-row configs.
  - Loads are issued straight-line, so the compiler's `vmcnt` waits stay exact.
  - The ring, DMA-ring and direct loops fit 256 VGPRs only with the current (nightly) LLVM. Older compilers spill the same source heavily.

## Strengths and weaknesses
Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. W/L counts shapes faster or slower by more than 5%.

| GPU | graphs K3 is our best | W/L vs MIOpen | weighted speedup | MIOpen solver it mostly meets |
|---|---|---|---|---|
| gfx950 | 294 | 199/54 | 1.67x | `GemmWrwUniversal`, CK group wrw (2-D, 3-D) |
| gfx942 (MI325X) | 266 | 203/31 | 1.43x | `GemmWrwUniversal`, CK 3-D group wrw |
| gfx90a | 276 | 137/114 | 1.33x | `GemmWrwUniversal`, CK group wrw |
| gfx908 | 259 | 125/96 | 1.79x | `GemmWrwUniversal`, CK group wrw |
| gfx906 (FMA) | 226 | 164/51 | 15.6x | `GemmWrwUniversal`, `ConvHipImplicitGemmV4R1WrW`, naive |
| gfx1201 | 326 | 284/26 | 4.50x | CK group wrw (WMMA), `GemmWrwUniversal` |
| gfx1100 | 359 | 327/23 | 17.95x | CK group wrw, `GemmWrwUniversal` |

- **Strong:**
  - Wan VAE 3x3x3 bf16: 0.20-0.64x of MIOpen's time on gfx90a-gfx950.
  - VAE NHWC bf16 on gfx908 / gfx942 and on RDNA, where MIOpen's universal GEMM is pathological.
  - nnU-Net fp32 3-D on RDNA and gfx908, where MIOpen runs naive or atomic CK.
  - Small-C stems: 7x7 s2 c3 0.62 ms against 0.87 on gfx950, 3.47 against 14.2 on gfx1201.
  - VAE 3-channel heads (gfx1201 c128 k3 768²: 2.25 ms against 23.9).
  - Deeplab dilated rows on gfx950 / gfx942 at d12 / d24 on 128² maps (0.68-0.95x of MIOpen's time).
  - Channels-last dilated rows run within reach of channels-first. Ratios are NHWC best engine over NCHW best engine, geomean over the corpus's fp16 layers, autotuned:
    - deeplab 3x3 d2-d36: 0.80 on gfx950, 0.91 on MI325X, 0.83 on gfx1101;
    - HiFi-GAN 1x3 / 1x7 / 1x11 at d3 / d5: 0.88 on gfx950, 1.16 on MI325X, 1.11 on gfx1101.

    DeepLabV3's c512 d4 layer (16 x 128², channels-last) takes 34 ms on gfx1101; channels-first takes 37-41.
  - The gfx950 direct loop is at parity with `ConvHipConv` on Flux / SDXL VAE (0.96-1.06x).
  - The ring's conflict-free channels-last pitch, cursor, XCD order and stages, `CONV_WRW` alone against the previous ring (autotuned, two interleaved rounds): geomean 1.14x over 36 of the corpus's heaviest wrw layers on MI325X (SD 1.5 UNet c1280 at 8²-16² 1.54-1.76x, Wan VAE 3-D 1.07-1.12x, Flux VAE 1.04-1.08x, nnU-Net fp32 1.04-1.06x) and 1.09x over 37 on gfx950 (ResNet-50 c256 at 14²-16² 1.38-1.41x, RegNet-Y grouped 1.34-1.44x, ViT patch embedding 1.27-1.35x; the 3-D bf16 layers there run on the direct loop).
  - gfx11's generic tier, on the corpus wrw layers the BLAS path served there (interleaved autotuned A/B, two rounds): 26 of them gain K3 configs, 1.51x in total time on gfx1100 and 1.40x on gfx1101 (14 and 12 faster by more than 5%, none slower). R(2+1)D's channels-last 3x1x1 layers run 1.7-4.0x, Inception's channels-last 5x5 1.2-1.75x and HiFi-GAN's 1x7 rows to one output channel 1.3-1.5x the BLAS path's speed.
- **Weak:**
  - gfx90a VAE NHWC bf16 3x3 at 96²-384²: 1.11-1.28x behind.
  - nnU-Net fp32 NCHW 3-D on gfx90a: 1.73x.
  - Deeplab 65² at dilation 36 (1.47-1.89x), and deeplab dilated on gfx906 / gfx908 (1.5-2.0x).
  - 3x3 stride 2 on mid-size outputs (about 17²-40²) that packed rows do not take, where 32/64-position chunks run half empty.
  - hifigan 1x16 stride-8 on gfx950 (4.2x).
  - Channels-last behind its own channels-first layer:
    - deeplab c2048 d24 on 128² maps, 1.2-1.3x on MI325X and gfx1101. The layout-1 ring costs this: on gfx1101 the same 64 x 32 x 32 tile runs 81 ms channels-last against 63 channels-first;
    - HiFi-GAN 1x11 d5 on gfx11, 1.6-2.0x;
    - 1x7 d3 at 32-64 channels on MI325X, 2.0-2.2x.

## Numerical precision
- **Accumulation:** fp32. 16-bit products are exact in the atoms; fp32 runs on fp32 MFMA (CDNA) or FMA (RDNA, gfx906). Split partials stay fp32, and the reduce sums them in an order that depends only on the split count, then rounds once. The `dws` direct store rounds the fp32 accumulators once.
- **Tolerance:** the device suite grades against hipDNN's CPU reference accumulated in fp64. The tolerance is 2e-6 x reduction length + 2 storage-eps x (|ref| + 1e-3); three-bf16 uses 3e-5 x reduction length.
- **inf/NaN:** infinities in x propagate as in the reference, and a test pins it for exact fp32 and the three-bf16 split. Known open parity item: the gfx950 direct loop (5) can turn inf x structural zero into NaN.
- **Determinism:** no atomics anywhere. The device suite relaunches each config over a re-filled workspace and compares bits.
- **Reduced precision:** the three-bf16 form drops lo x lo, about 2^-16 of each product. Only `CONV_WRW_TF32` lists it (882 configs; 7168 exact in `CONV_WRW`), reporting `TENSOR_CORE`, by which a caller excludes it (`deselect_numeric_notes`). MIOpen's `GemmWrw1x1_stride1` rounds dW to fp16 after every image. K3 does not; that form is the BLAS reduced-precision-reduction engines', which report `REDUCED_PRECISION_REDUCTION`.

## Comparison with MIOpen
- **CK `ConvHipImplicitGemm{,3D}GroupWrwXdlops`:** M = K, N = C·taps via descriptor im2col, NHWC only. NCHW goes through 3-5 full-tensor transposes. Split-K uses fp32 or packed 16-bit atomics; deterministic mode forces split 1. K3 wins where CK transposes or runs one split, and loses on some large NHWC bf16 3x3 on gfx90a.
- **`GemmWrwUniversal` / `GemmWrw1x1_stride1`:** im2col plus a BLAS GEMM per image. Strong on CDNA for large VAE layers and 1x1, and very slow on RDNA. The 1x1 form keeps an fp16 accumulator across images (see above).
- **`ConvHipConv` (`direct_wgrad`, gfx950):** the design source of loop 5. It is NHWC 16-bit only, stride 1, filters 2²-5²; loop 5 is at parity with it.
- **Legacy kernels:** `ConvAsmBwdWrW3x3/1x1`, `ConvOclBwdWrW2/53` and `ConvHipImplicitGemmV4R1WrW` serve gfx906 / gfx908. The dilated deeplab layers there are still MIOpen's.
- **Winograd wrw** (`ConvBinWinogradRxSf3x2`, multipass `<3-x>`): our counterpart is `CONV_WINOGRAD_WRW`.

## Prior art considered
- **hipconv `direct_wgrad`:** all taps in registers, a kh-deep dy ring, 16-byte swizzled DMA, transpose reads, ping-pong, √H row segments, fp32 partials with a fixed-order reduce. Reimplemented as the ring (every arch) and loop 5 (gfx950).
- **CK grouped wrw:** studied for mapping and split policy. Its atomic split-K was rejected for determinism.
- **rocKE wgrad PRs:** an AOT implicit GEMM; a direct wgrad with all taps in registers and a dy ring; atomics or slab folds. Their "per-slice slabs, no atomics" note is K3's scheme.
- **rocKE and FlyDSL ports of K3** (12 corpus shapes, fp16 / bf16, gfx950 / gfx1201 / gfx942):
  - K3 runs ahead of the best rocKE and FlyDSL flavours on gfx950 and gfx1201 (indicative).
  - The gaps the ports exposed come from catalog breadth or an ISA-level idiom (128-bit edge shift, predicate form); K3 has both.
  - Spills depend on the compiler version, not the language.

No code was copied.

## Next steps and deferred ideas
1. **Same-config gaps against the rocKE / FlyDSL ports:**
   - loop 6 register pins and explicit waits cost 1.19x on gfx1201 stems;
   - loop 7 live ranges allow one block per CU instead of two (1.20x on gfx950);
   - a single branch-free ring multiply path (1.05-1.11x on gfx1201 NHWC).

   Each is a measured, same-config gap.
2. **65² at large dilation:** needs a tap-split / flattened-position loop. A tap-split form measured slower (5.5 against 4.2 ms), so the flattened form is the candidate.
3. **gfx90a NHWC bf16 VAE 3x3:** a ring ping-pong `pp` knob for CDNA2, whose long MFMA phase suits it (a projection only; needs a probe). Also: d24/d36 there run at 61% of MFMA peak against asm igemm's 70%.
4. **Strided 3x3 on small maps:** a flattened-position variant of loop 1 phases, as K4 wrw does.
5. **inf parity in loop 5:** a select for structural zeros, A/B-checked against the compiled-out build.
6. **Channels-last dilated rows:**
   - the layout-1 ring's store-side transpose costs 1.2-1.3x at deeplab d24 against channels-first (same tile);
   - small-channel 1-D rows (c32-c64, 1x7 d3 on CDNA; 1x11 d5 on gfx11) want channels-last tiles of their own;

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestConvWrwProblem.cpp` (host):
  - **Gates:** one refusal per field (`RefusesOneFieldAtATime`, `OneField`); every loop's admission rules (ring phases and stages, DMA ring gfx950-only, the direct loop and its plan, small-C, packed rows, unaligned dy).
  - **Large images and layouts:** block-based buffers serve only images past the range; layouts are read from strides.
  - **Generic tier:** spec keys, the ASPP decline (and its admission on gfx11 and gfx12's matrix atoms), the narrow-group, stem and narrow-row declines, its FMA form.
  - **Arithmetic and planning:** LDS and load-pass arithmetic against the kernels'; ring track trimming; the split planner and `bpu`; reduce layouts and reflection; the reduce choice (quad, tile, row) and its tiles; the one-split direct store.
  - **TF32:** three-bf16 admitted only for fp32 graphs and only on paying classes.
- `TestGpuConvWrw.cpp` (device), fp32/fp16/bf16 against an fp64-accumulated reference:
  - every admitted config on NCHW and NHWC, 1-D/2-D/3-D, groups, stride 2, dilations 2/3/12, channels-last 3x3 at dilations 4, 6, 24 (three images) and 36 (65^2), channels-last 1x3 d5, 1x7 d3 and 1x11 d5, a one-output-channel 1x7 (HiFi-GAN's output layer) in both layouts, channels-last 5x5 and 32-channel groups, and 3-D 3x1x1 with input channels that are not whole vectors in both layouts;
  - padding below, at and past the filter; odd channels and partial tiles; both dW layouts; true convolution;
  - three-bf16 against exact fp32, and infinities;
  - small 3-D maps whose rows fit a ring stage's 16- and 8-position steps, in both layouts, and the ring configs alone with infinities in x in every element type;
  - the generic tier on every matrix arch (gfx11 included): channels-first and channels-last, 1-D/2-D/3-D, strided, dilated (ASPP-class dilations on gfx11 and gfx12), padded and strided-view operands, infinities in every element type, a tensor past 2 GiB (three images) and an image past 2 GiB (block-based buffers);
  - the determinism relaunch, and dW equal bit for bit to the fixed-order sum of the workspace's partials (every reduce);
  - knob uniqueness, generic spec keys, and atom ids and LDS fits per arch;
  - a specialized config for every deeplab / SSD 3x3 dilation (4-36) and HiFi-GAN dilated 1-D row (1x3, 1x7, 1x11 at d3 / d5) in both layouts on every matrix-core arch (`DilatedRowsHaveSpecializedConfigsInBothLayouts`): a dilated row the specialized tier misses falls to the generic tier (which declines ASPP-class dilations except on gfx11 and gfx12) or to the BLAS path.
- `TestConvDescriptorShards.cpp` checks the shipped shards' numerical notes (`CONV_WRW` none, `CONV_WRW_TF32` `TENSOR_CORE`; both deterministic) and that only `CONV_WRW_TF32` lists three-bf16 configs.
