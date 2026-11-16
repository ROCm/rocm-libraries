<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_DEPTHWISE_*: the depthwise convolution engines

> Engine ids: `hipkernel:CONV_DEPTHWISE_MFMA_FWD`, `hipkernel:CONV_DEPTHWISE_MFMA_BWD`, `hipkernel:CONV_DEPTHWISE_ROWS_FWD`, `hipkernel:CONV_DEPTHWISE_ROWS_BWD`, `hipkernel:CONV_DEPTHWISE_VALU_FWD`, `hipkernel:CONV_DEPTHWISE_VALU_BWD`, `hipkernel:CONV_DEPTHWISE_VALU_WRW` · Generators: `conv/generators/gen_depthwise_{mfma,rows,valu,valu_wrw}.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/depthwise_{mfma,rows,valu,valu_wrw}/kernels/` · Packs: `packs/ConvDepthwise{Mfma,Rows,Valu}{Problem,Native}.cpp`, `packs/ConvDepthwiseValuWrwNative.cpp`

## At a glance
- **What:** four engine families for depthwise convolution (one input channel and one output channel per group). They are bandwidth-bound VALU kernels with register and LDS tiling (`VALU`, `ROWS`), a CDNA matrix-core forward and backward-data kernel (`MFMA`), and a deterministic partial-plus-fold backward-weights engine (`VALU_WRW`) with VALU and 4x4x4-MFMA ("Hankel") configs.
- **Covers:** fwd, bwd-data and wrw. fp32, fp16 and bf16 (MFMA and Hankel are 16-bit only). NHWC and NCHW, packed. 1-D, 2-D and 3-D (MFMA and ROWS are 1-D and 2-D only). Any stride, dilation and padding on the floors. Both convolution modes. The wrw engine also takes depth multiplier 2 (K = 2C) on its VALU configs.
- **Large tensors:** VALU floors switch to a 64-bit index body past 2^31 bytes, and the rowtile core needs only rows below 2^31 bytes (a device test runs fp16 past 2 GiB). The other tile cores, the MFMA kernel and the Hankel configs need each image (or folded image group) under 2 GiB, with 64-bit image bases. ROWS uses 64-bit plane bases.
- **Arches:** VALU, ROWS and VALU_WRW (VALU configs) run on every architecture, wave32 and wave64, gfx906 through gfx12. MFMA runs on gfx908 (fp16) and gfx90a, gfx94x and gfx950; its 16x16x32 atom is dense on gfx950 and the 2:4-sparse `smfmac` on gfx94x (NHWC only). Hankel runs on gfx908 (fp16) and gfx90a, gfx942 and gfx950. RDNA has no matrix-core depthwise.
- **Wins when:** almost every depthwise layer on RDNA, gfx906-gfx942 (the class runs 8-16x faster than MIOpen search in usage-weighted time). On gfx950, the class is at 1.0-1.3x depending on direction and layout. Strided bwd-data (phase blocks), dilated and 1-D vocoder rows, and ConvTranspose1d are the largest margins.
- **Loses when:** gfx950 NHWC 16-bit wrw on some shapes: ConvNeXt 7x7 bf16 is 1.03x behind MIOpen's hipconv kernel overall (same node), from 0.85x (ahead) on 7x7 images to 1.49x behind on n16 64x64. Large filters (31x31) go to `CONV_FFT_*`. Depth multiplier above 1 in fwd/bwd goes to the grouped implicit GEMM, BLAS or naive.
- **Precision:** fp32 accumulation, one explicit `fmaf` per tap (never a separate multiply and add), and one rounding at the store. No numerical notes. Every config is bitwise deterministic, with no atomics; wrw folds its partials in a fixed order.
- **Instances:** MFMA 640 (384 fwd, 256 bwd); ROWS 222; VALU 342 (162 fwd, 180 bwd); VALU_WRW 1140 (66 VALU tile/floor and 162 of their `bpc` variants, 504 Hankel, 24 vtile and 72 of their `bpc` variants, and every tile and vtile config again at `fill` 1). Counted from the generators.
- **vs MIOpen:** replaces `ConvHipConv` (hipconv's MFMA depthwise and Hankel wgrad), the CK grouped depthwise forward, MIOpen's direct depthwise solvers, `ConvHipDirectFwd`, the GEMM and Winograd fallbacks and `ConvDirectNaive` on depthwise graphs.

## Selection guide

Autotune picks among every admitted config. This table says which engine serves which regime, so that a reader can predict the winner and the coverage.

| Regime | Engine (config kind) | Arch |
|---|---|---|
| fwd / unit-stride bwd, 2-D, 16-bit, NHWC or NCHW, 3x3-9x9 (stride 2: 3x3, 5x5 fwd) | `MFMA` | gfx908 (fp16), gfx90a, gfx94x, gfx950 |
| fwd, NHWC 2-D, 3x3/5x5/7x7, stride 1 or 2, any dtype | `VALU` `rowtile` (f16 via `v_fma_mix_f32`) | all |
| fwd / unit-stride bwd, NHWC 2-D, 3x3 halo, 5x5 and 7x7 fused, 9x9 LDS; NDHWC 3x3x3 | `VALU` `microtile` / `fused` / `lds` | all |
| fwd / unit-stride bwd, NCHW 2-D 3x3, 5x5, 7x7 (plane tiles, small whole planes) | `VALU` `lds` (channels-first) | all |
| anything depthwise: any filter, stride, dilation, 1-D/2-D/3-D, >2 GiB | `VALU` `floor` | all |
| strided bwd-data, 2-D, 3x3/5x5/7x7 at stride 2 | `VALU` `phase` | all |
| strided bwd-data, any filter, any stride, 1-D/2-D/3-D | `VALU` `gather` | all |
| fwd NCHW 1-D/2-D with dilation, 1xk vocoder rows (1x3 to 1x31) | `ROWS` `tile` / `band` | all |
| strided depthwise ConvTranspose1d (bwd-data, row-separable), NCHW or NHWC | `ROWS` `phase` / `lanes` | all |
| wrw, NHWC 2-D 16-bit, 3x3/5x5/7x7, stride 1 or 2 (both axes) | `VALU_WRW` `hankel` | gfx908 (fp16), gfx90a, gfx942, gfx950 |
| wrw, NHWC 2-D, 3x3/5x5/7x7, unit width stride, any dtype | `VALU_WRW` `vtile` | all |
| wrw, 3x3/5x5/7x7/1x4/1x31/3x3x3, either layout | `VALU_WRW` `tile` | all |
| wrw, anything depthwise (multiplier 1 or 2) | `VALU_WRW` `floor` | all |

## Why it exists
- Depthwise layers (MobileNet, EfficientNet, ConvNeXt, Conformer, Mamba conv1d, vocoder resamplers) are memory-bound. A per-group implicit GEMM has a 1x1 channel tile per group and wastes almost all of it. Dedicated kernels are the only route to a bandwidth-limited time.
- The VALU forward/backward-data cores are based on hipconv's hand-written VALU depthwise kernels (the same code MIOpen's direct depthwise solver uses), with the filter reflection for backward data and reworked code generation. ROWS reimplements techniques from open MIOpen RDNA pull requests (channel-row blocks, scalar filters, stride-phase ConvTranspose1d).
- On gfx950 a VALU 7x7 kernel cannot keep up with matrix-core depthwise: it takes 0.31 ms against MIOpen's 0.050 ms. So `MFMA` exists, and `VALU_WRW` carries the Hankel wgrad configs for the same reason.
- No other engine serves depthwise wrw at speed: GEMM and naive are 1.1-4.3x behind MIOpen, and run the Mamba conv1d wrw at a small fraction of its bandwidth floor.

## How it works

### CONV_DEPTHWISE_VALU_FWD / BWD
- Every config computes a forward convolution of `in` into `out`. Backward data at unit stride is the forward convolution of dy with the filter reflected and the padding mirrored (`pad' = dil (k - 1) - pad`). True convolution reflects once more, so the two cancel.
- Backward data at stride above 1 runs only on the `gather` variant (one thread per dx element, a sum over the taps that land on it, with no zero upsampling) or on the `phase` blocks (a thread owns the S x S dx pixels that share one dy window).
- Cores: the W-strip floor (NHWC/NDHWC) and the one-element-a-thread floors (NCHW/NCDHW), which take any filter, stride and dilation. `microtile` is a register 2-D halo with no LDS. `fused` is an LDS block stage plus a register micro-tile. `lds` is an LDS stage with taps read from LDS; it is register-light and suits big filters and 3x3x3. The channels-first `lds` plane tiles include whole small planes of 14x14 or 16x16 per block.
- `rowtile` gives each lane a TH x TW output tile of 1 or 2 channels. It streams the input rows through registers (each row feeds every output row whose filter row it is) and takes its spatial tile from the block index, with no LDS. For fp16 the operands stay 16-bit and the backend folds both conversions into `v_fma_mix_f32`. The result is the `fmaf` of the converted operands bit for bit, at half the operand registers. bf16 widens by a shift.

### CONV_DEPTHWISE_ROWS_FWD / BWD
- Every block works inside one channel row or plane, so the filter is uniform across the block and is held in scalar registers.
- Forward (channels-first, 1-D and 2-D): `tile` is a plane tile. `band` is a band of whole rows with no tile edge along the width. Both stage the input halo in LDS as fp32. With dilation d each thread's outputs step along the dilation grid, so neighbouring outputs share inputs and a patch row costs RW + KW - 1 LDS reads, not RW x KW. Any stride, dilation and padding.
- Backward data of a strided 1-D (or height-1, row-separable 2-D) depthwise convolution, which is a depthwise ConvTranspose1d, runs by stride phases. With `i + pad = s q + r`, `dx[i] = sum_m w[r + s m] dy[q - m]`. No multiply is spent on a skipped tap and no index needs a modulo. `phase` uses a block per row, channels-first or channels-last. `lanes` (channels-last) puts the lanes over (position, channel) with the channel fastest, so dy loads and dx stores are contiguous for any channel count.
- Unit-stride backward data runs as the reflected forward form.

### CONV_DEPTHWISE_MFMA_FWD / BWD
- **Channels-last core.** One MFMA multiplies a filter row by an input row. A holds the filter row's taps as a one-hot diagonal over 8 channels. B holds the input row, one 16-byte channel granule per lane. D is 8 channels x 32 output columns, and the off-diagonal products are zeros, which costs nothing in a memory-bound kernel.
- A filter row takes `ceil((kw + 1) / P)` passes, P = K / 8 input positions a pass. A wave streams down the image with one 16x16 fp32 accumulator per filter row, as a ring: output row y collects input rows y - pad .. y - pad + kh - 1, then leaves.
- **Channels-first core.** Channel pairs lie along the MFMA's K, and the filter row is a Toeplitz B operand. One MFMA gives 2 channels x 128 columns and covers a filter row up to `kw = P + 1 - 7 S` in one pass. Each wave loads its own rows (unaligned 16-byte loads, no LDS).
- On gfx94x the 16x16x32 atom is the 2:4-sparse `smfmac`. The channel-diagonal taps hold at most one nonzero per 4 channels, so they compress exactly. The channels-first input operand is dense, so channels-first uses the 16x16x16 atom there.
- Stride 2 (forward, 3x3 and 5x5): the block stages the 64 + halo input columns its 32 outputs read. Backward data runs at unit stride only, as the reflected forward form; strided backward data goes to `VALU` `phase`/`gather`.
- Narrow images share the MFMA columns (`fold`: 2 or 4 images channels-last, up to 8 channels-first).

### CONV_DEPTHWISE_VALU_WRW
- **Schedule (all configs).** The reduction over images x output rows is cut into partitions. A partial kernel sums one partition per channel and tap, in fp32, into a workspace. A fold kernel adds the partitions in a fixed order and rounds once into dw. At one partition the partial kernel writes dw itself, with the same rounding, and there is no workspace and no fold.
- **Partitions.** The config's `bpc` blocks per dispatch unit set the target (0 on a VALU variant means 4). Configs at `fill` 1 trade partitions for whole rounds: of target / 2 .. target partitions, the planner takes the count with the least rounds of blocks over the units times rows per partition (the busiest unit's rows; the most partitions on a tie). `fill` 0 keeps the target. Both forms ship and autotune picks: small batches gain from `fill` 1, while some n128 shapes (EfficientNet c1536 12² fp16 on gfx950) run 4-5% faster on the target. On gfx950 EfficientNet c960 30² n16, the target of 35 partitions of 15 blocks ran a third round for 13 blocks; two full rounds are about 1.17x faster. The VALU tiles and vector tiles also ship at `bpc` 1, 2 and 8, because few rows per partition re-read the filter's halo rows: ConvNeXt c512 14² n16 fp32 runs 1.6x faster at 1-2 blocks per unit, and its NCHW fp16 form 1.6x faster at one partition. Workspace is partitions x channels x taps x 4 bytes.
- **`floor`:** any filter, stride, dilation and padding, 1-D to 3-D, both layouts, multiplier 1 or 2.
- **`tile`:** compile-time filter (3x3, 5x5, 7x7, 1x4, 1x31, 3x3x3), every tap's sum in registers, dy read once a position. At unit width stride and dilation, x is read in 8-column register strips. Multiplier 2 is available for the 2-D square filters.
- **`vtile`:** channels-last, unit width stride and dilation. A thread owns `cv` consecutive channels (one vector load a column) and every tap's sum of each, walking its rows in strips of `strip` columns. A partition gives each row lane at least 4 rows.
- **`hankel`:** hipconv's Hankel construction on 4x4x4 MFMA, with 16 channels as the instruction's 16 blocks. For input row y, channel c and a 4-column step q:

```
A[s][k] = x[y][S (q + k) + s - pw]        (Hankel: the tap column shift s)
B[k][r] = dy[(y + ph - r) / S][q + k]     (the filter rows' dy rows; zero where S does not divide)
C[s][r] += sum_k A[s][k] B[k][r]          (= dw[c][r][s] over this row and step)
```

  Filters past 4 taps tile into 4x4 tap groups, each with its own accumulator. The dy rows sit in an LDS ring of KH + G - 1 slots plus a zero slot.
- **Stride-2 phase split.** At stride 2 each x row is staged as two phase rows, its even and its odd columns, so `A[s][k]` = phase `s % 2` at `k + s / 2`. That is again a shift along one row, and each lane's operand read stays one 64-bit window. Filters up to 8 wide.
- **Parity masks (inf/NaN).** The kernel never keeps a product whose dy operand is not one of the convolution's, so an inf or NaN in x never meets a (row, column) pairing that the convolution does not have.
  - Columns: where wo is not whole tiles, each step ANDs the lane's A with a block-uniform column mask.
  - Rows: an input row whose filter row has no dy row here (outside the image or partition, or at stride 2 the other parity) saves the accumulators before its products and restores them after. This costs a few moves a row, not a select per product.
- **Two kernel bodies (`form`).** Both compute the same products in the same order, with the same masks; they differ in the work around the matrix instructions.
  - `form` 0: any thread stages any unit of the step's rows, so each load selects its row offset and bounds per lane, and each lane derives its dy ring slot with a division per input row.
  - `form` 1: each wave stages whole rows. A row loads through its own buffer descriptor (a row outside the image or partition has zero records, so its loads read zeros); a lane's column offsets and LDS offset are computed once per tile. The ring slot of a lane's B column is the wave-uniform slot of the row stepped back by the lane's tap (a compare-free wrap and a 24-bit multiply), and only the tap groups with a missing filter row save and restore their sums. The kernel is bound by instruction issue, not by the MFMAs: on gfx950, ConvNeXt c512 14x14 bf16 at one schedule issues 7.3 VALU and 6.1 SALU instructions per MFMA in `form` 0 and 5.9 and 5.6 in `form` 1, with 18% fewer wave cycles.

## Implementation design

### Common
- Every family takes all shapes at runtime. A config bakes only the variant, the tile and, where noted, the filter and stride (HIPRTC `-D` defines with family prefixes `CONV_DW_`, `CONV_DWR_`, `CONV_DWM_`).
- The shared problem parse gates: depthwise (`c == k == groups`; wrw allows `k = mult x c`); x and y packed in the same layout (NHWC-like or NCHW-like) with the filter packed; every extent, pad, stride and dilation within 32 bits; fwd or bwd-data (wrw has its own parse). Axes of extent 1 accept any stride.
- Codegen choices common to the tile cores:
  - Branch-free per-lane predicates: an out-of-range element loads an in-range address or a buffer offset past the record count, and its value is selected away or reads hardware zero. The loads of a group stay in flight together.
  - Whole-tensor or whole-image buffer resources at 32-bit offsets, so out-of-range outputs are stored past the end and dropped. This avoids exec-masked store blocks that would swallow each output's FMA chain.
  - `readfirstlane` on values that are wave-uniform but not provably so, such as `tid / BK`.
  - An LDS-only barrier on gfx12, which avoids the L0 invalidate of `__syncthreads`.
  - 16-byte channel-vector LDS staging where C allows it, and compile-time, fully unrolled staging trip counts.

### VALU fwd/bwd
- Kernel match per variant: `floor` takes everything. `microtile`/`fused`/`lds` need unit stride and dilation and their compile-time filter. `rowtile` needs its baked stride on height and width, unit dilation and its filter. `phase` needs 2-D bwd-data at its stride and filter with unit dilation, and channels-last vectors of at most 16 bytes. `gather` is bwd-data only, and with `phase` the only form for strided bwd-data.
- Large tensors: the tile cores have only the 32-bit form (x, y and launched threads under 2^31) and decline beyond it. A second 64-bit body would set the register count of every launch; for fused 7x7 on gfx950 that is 177 VGPRs against 77. The floors switch bodies at run time. The `rowtile` core rebases per row, so one body serves every size whose rows fit.
- Launch: 256-thread blocks for floors and gather; the variant's tile for the rest; `rowtile` uses 64 lanes x channel groups by output-row tiles.
- Catalog: 4 floors, 3 microtile, 4 fused, 2 NHWC LDS (9x9, 3x3x3), 10 NCHW LDS plane tiles, 22 stride-1 `rowtile` (3x3, 5x5, 7x7 tiles x vec 1/2), 10 stride-2 `rowtile` (fwd only), 4 `gather` and 12 `phase` (bwd only), each x 3 dtypes. With no ranker, the floors come first in declared order.

### ROWS fwd/bwd
- Gates: forward is channels-first, 1-D (as height 1) or 2-D. Bwd-data takes either layout but not 3-D. `phase`/`lanes` need a row-separable problem: the height axis is the identity, the width has unit dilation, and the stride is the config's. The dy window holds `mmax` taps a phase, so a runtime-filter config admits up to 32 taps a phase. The LDS halo must fit 64 KiB (as fp32 elements).
- The filter is compile-time (1x3, 1x7, 1x9, 1x31, 3x3, 5x5, 9x9; ConvTranspose 12/29/43 taps) or read at runtime (`kh = kw = 0`).
- On gfx9, fp16 5x5 band taps are kept as opaque fp32 VGPRs to avoid an LLVM instruction-selection crash in the mixed-precision FMA combine.
- Catalog: 25 forward-form configs (tile and band, also used for unit-stride bwd) and 24 phase/lanes configs, x 3 dtypes; 222 kernels. Each config is the fastest on at least one target shape on gfx1100, gfx1201 or gfx942.

### MFMA fwd/bwd
- Gates (every config field): layout matches the problem; 2-D (1-D as height 1); not gather-only; the config's stride (1 or 2) on both axes; unit dilation; atom 16 or 32 and runnable on the device (`mfmaRunsOn`); filter equals the config's; waves a power of two up to 16 (channels-last also needs `C % (8 x waves) == 0`, while channels-first takes any C); depth 1-4; fold a power of two up to 4 (NHWC) or 8 (NCHW); rows positive; LDS within 64 KiB; folded image bytes within 32-bit buffer offsets.
- Channels-last block: 8 x `waves` channels. It stages every input row once with coalesced 16-byte loads (`depth` rows in flight) into a double-buffered LDS row and writes output rows back through another double-buffered LDS row, with one LDS barrier per row. The streaming body has no branches. Channels-first claims no LDS.
- Catalog: per filter class a list of (waves, depth, fold, rows) tuples, small filters and large filters, x atoms 16 and 32 x 2 layouts x fp16/bf16. Stride-2 lists are forward only.

### VALU_WRW
- Gates: `floor` takes everything. `tile` needs its filter. `hankel` must pass every field: arch and dtype; multiplier 1; NHWC 2-D; its filter; stride s on both axes (s = 2 only for filters at most 8 wide); unit dilation; C a multiple of 8; each image under 2 GiB; q in 4..64 (multiple of 4); g 1-4; nwg 1/2/4; bpc 1/2/4/8 (the VALU variants: 0/1/2/8); `form` 0 or 1; fill 0 or 1 (0 on the floor); LDS within 160 KiB on gfx950 or 64 KiB elsewhere; x and dy pointers declared 16-byte aligned. Only `hankel` configs may set `form`. `vtile` needs NHWC 2-D, multiplier 1, its filter, unit width stride and dilation, `cv` in {1, 2, 4, 8} dividing C, `strip` in {2, 4, 8}, and pointers aligned to a `cv` vector. A device reporting no dispatch units is declined.
- Hankel block: 4 waves x `nwg` groups of 16 channels. Each step stages `g` input rows and the dy rows entering the ring, with the next step's global loads in flight under the current step's MFMAs. LDS rows use an odd number of dword pairs as pitch, plus a skew every 8 rows, so the 16 channel rows of an operand read hit distinct banks. There is no swizzle.
- Catalog: floor plus 6 tiles, plus 4 multiplier-2 configs, x 2 layouts x 3 dtypes (66), and the 9 tiles again at `bpc` 1, 2 and 8 (162). Hankel: 11 schedules (q, g, nwg, bpc) in both forms plus 7 in `form` 1 alone (8-, 24- and 28-column tiles for 7-, 24- and 28-column images, and four blocks a unit at two rows a step), x filters 3/5/7 x stride 1/2 x fp16/bf16 (348), again at `fill` 1 in `form` 0 (132) and in the two `form` 1 schedules at four blocks a unit and two rows a step (24), where whole rounds beat the partition target on small EfficientNet batches (504 in all); each schedule won on some gfx950 ConvNeXt, EfficientNet or MobileNet shape. 8 vtile shapes x 3 dtypes (24), and again at `bpc` 1, 2 and 8 (72). Every tile and vtile config also ships at `fill` 1. Floors come first.

## Strengths and weaknesses
- **Usage-weighted, depthwise class, all engines, against MIOpen search** (model corpus; the class is served almost entirely by these engines and FFT):
  - Full corpus, native layouts: gfx942 MI325X 16.0x, gfx1201 12.3x, gfx1100 17.9x, gfx950 1.31x.
  - 2K-shape subset: gfx90a 24.4x; usage-weighted MIOpen/ours about 13x on MI325X, 16x on gfx90a, 8x on gfx1201, 12x on gfx1100 and 8x on gfx906.
- **gfx950:** on a dedicated depthwise set the class runs 1.28x overall: wrw about 1.13x, NCHW fp16 wrw about 1.67x. NHWC 16-bit Hankel wrw: `form` 1 runs 1.11-1.47x faster than `form` 0 on the ConvNeXt 7x7 bf16 shapes (1.18x usage-weighted) and 1.09-1.15x on EfficientNet, MobileNet, MobileFaceNet and YOLO NHWC fp16 wrw (autotuned, same node), which puts ConvNeXt at 1.03x behind hipconv overall. MFMA forward and backward data are about at parity with hipconv (1.02x). MFMA is the best engine on about 110 graphs.
- **gfx942:** Hankel runs ConvNeXt c128 wrw in 0.187 ms against MIOpen's 17.3 ms. `form` 1 measures 1.04-1.37x over `form` 0 on MI325X, 1.07-1.49x on gfx90a and 0.98-1.53x on gfx908 (kernel bench, ConvNeXt 7x7 and EfficientNet 3x3 shapes).
- **VALU rowtile + `v_fma_mix`:** measured against the VALU cores without `rowtile` in a 60-cell, three-dtype bench (cold, L2 flushed; indicative):
  - gfx950: 1.16x over all cells, 1.35x on the 18 NHWC 16-bit cells.
  - gfx942: 1.15x and 1.33x.
  - gfx1201: 1.08x and 1.19x; the top 7x7 shape runs at 79% of DRAM bandwidth there.
- **Strided bwd-data:** the `phase` variant measured 8.5-9.1x MIOpen on gfx1201. Strided depthwise on an RDNA3 part is about 4x faster than MIOpen's Winograd fallback on a bigger GPU.
- **ROWS:** on the 90 targeted RDNA graphs (resampler ConvTranspose1d, dilated DeepLab, Conformer k31/k9, 9x9), our-time/MIOpen-time is about 0.23 (gfx1100) and 0.24 (gfx1201), estimated from kernel benchmark ratios (indicative). Three near-parity cells remain on gfx1201.
- **Weak spots:**
  - gfx950 NHWC 16-bit wrw, where the VALU and scalar issue rate binds.
  - NCHW 7x7 forward on RDNA, the largest depthwise excess in the full corpus.
  - The gfx942 NHWC 3-D 16-bit LDS core, about 13% slower than an alternative staging form measured for it (cause unknown).
  - Depth multiplier above 1 in fwd/bwd, which no depthwise engine serves.
  - Large-kernel depthwise, which goes to FFT.
  - RDNA has no matrix-core path by design: the diagonal channel block keeps about 3/32 of a WMMA useful, which is below VALU fp32 throughput.

## Numerical precision
- Accumulation is fp32 everywhere.
  - VALU/ROWS: one `__builtin_fmaf` per tap. An explicit FMA prevents the SLP vectorizer from pairing multiplies into `v_pk_mul_f32`, which would leave two roundings.
  - fp16 output rounds as a separate step: an empty `asm` keeps the backend from folding the last FMA into `v_fma_mixlo_f16`, which differs in about 1 in 2000 near-halfway cases. So `rowtile` fp16 is bitwise the `fmaf` of the converted operands, and a gtest checks it.
  - MFMA/Hankel: 16-bit products are exact in fp32 and accumulate in fp32 inside the instruction. gfx90a's fp16 MFMA flushes denormal inputs (a hardware property; the test widens the bound there).
- Wrw: fp32 partials per (partition, channel, tap), folded in partition order, rounded once; one partition rounds its sums once into dw. There are no atomics.
- Error model in the device tests: `|got - want| <= 1e-6 x taps (or reduction length) + 2 eps_storage (|want| + 1e-3)` against hipDNN's CPU reference. True convolution is checked by reflecting the filter around the cross-correlation reference.
- Determinism: no `nondeterministic` note on any engine. Each family's device test relaunches every config and requires a bitwise-identical result.
- inf/NaN parity: Hankel configs are masked to match the reference, and a dedicated test puts an inf in x where some taps never pair with dy. The VALU `tile`/`vtile` wrw variants and dy-side padding are not yet covered by a parity test (open item). The floors and VALU forward cores only multiply real taps.
- No reduced-precision siblings: there is no TF32 or three-bf16 form, and fp32 stays on VALU.

## Fusion
- `VALU_FWD`, `ROWS_FWD` and `MFMA_FWD` claim a forward depthwise convolution followed by a chain of pointwise nodes (the conv engines' epilogue menu: scale, bias, relu / clipped relu / clamp / leaky relu, gelu, gelu-tanh, swish/silu, sigmoid, tanh, elu, abs, softplus; per-channel, device-scalar and pass-by-value operands) under their own engine ids. The backward engines decline fused graphs.
- Mechanism: the generated epilogue runs on each fp32 result before the single rounding to the output type, at every forward store:
  - `VALU`: the W-strip and channels-first floors (2-D and 3-D), microtile, fused halo, LDS tiles (2-D, 3-D, channels-first plane tiles) and rowtile;
  - `ROWS`: the plane tile and the band;
  - `MFMA`: the channel-diagonal core before it rounds into the LDS output row (the row copy stays a bitwise copy), and the channel-pair core.
- Every thread owns fixed channels, so a store site loads its channels' operands once (`ConvDepthwiseEpilogue.h`). With fusion off the macros expand to nothing and the kernels compile to the same instructions as a build without the hook.
- 16-bit rowtile configs pass the accumulator through an empty `asm` before the epilogue: without it, instruction selection fails for some chains (gelu-tanh, bias + relu) in the current compiler.
- No config is added: the epilogue is part of a config's compile, so a fused pattern compiles once per config and serves every graph of that pattern.

**Known cost.** A fused graph a depthwise engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.01x, gelu 1.19x, silu 1.10x; gfx950 (both layouts) relu 1.05x, gelu 1.16x, silu 1.09x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.55x on MI325X and 1.60x on gfx950 (all three depthwise engines) (usage-weighted; the conv README's fusion table).

## Comparison with MIOpen
- **`ConvHipConv` (hipconv):** gfx950-centred MFMA depthwise (1-D Toeplitz on 16x16x32, sparse on wide filters) and the Hankel 4x4x4 wgrad.
  - `MFMA` adopts the one-hot channel-diagonal idea but applies it to 2-D rows with an accumulator ring, plus a separate channels-first Toeplitz core. It also brings the sparse atom to gfx94x.
  - `hankel` reimplements the wgrad construction and adds stride 2 (phase split), the parity masks, an LDS ring with conflict-free pitches, and a deterministic fold instead of any cross-block accumulation.
  - On gfx950 forward and backward data are at parity, and NHWC 16-bit wrw is about 0.85x.
- **MIOpen's direct depthwise forward solvers** (2-D and 3-D; the hipconv VALU kernels harvested into MIOpen) and the **CK grouped depthwise forward:** the `VALU` cores start from the same validated kernels and run 1.10-1.21x faster than them (cold geomean) through the codegen choices listed above. They add bwd-data (reflected forward, gather, phase), true convolution and every architecture.
- **`ConvHipDirectFwd`:** bakes every shape into compile-time macros and accumulates fp16 in fp16. Only its phase-blocked strided idea was taken, and it accumulates in fp32 here.
- **GEMM / Winograd / `ConvDirectNaive`:** these serve depthwise in MIOpen where nothing better applies, notably strided bwd-data via zero-upsampled dy and all wrw on older arches. That is the source of the 8-24x class ratios on gfx906-gfx942 and RDNA.
- Comparisons above use MIOpen search mode (5 timed iterations) unless stated.

## Prior art considered
- hipconv v0.5 depthwise (1-D Toeplitz fwd/dgrad on MFMA, Hankel wgrad, 2-D Toeplitz) was studied. The 2-D Toeplitz is not used (upstream marks it opt-in and never better than 1-D); the 1-D MFMA Toeplitz is deferred.
- hipconv's VALU depthwise kernels (rocm-libraries PR #10945) are the base of the `VALU` cores.
- Open MIOpen RDNA PRs (#15100, #15130, #15233) supplied the channel-row, scalar-filter, dilation-grid and ConvTranspose1d phase techniques, reimplemented in `ROWS`.
- MIOpen `ConvHipDirectFwd` contributed phase blocking for strided backward data. hipconv's zero-upsampled dy was rejected, because it wastes 3/4 of the MACs at stride 2.
- rocKE and FlyDSL: ports of the depthwise VALU forward to both, compared cell by cell.
  - Where the ports were faster, the cause was HIP-expressible codegen: buffer loads with hardware-zero bounds, branch-free predicates, wave-uniform values made scalar, and keeping guarded stores from swallowing FMA chains. The `VALU` cores use all four.
  - `rowtile` reimplements rocKE's register-tiled kernel with `v_fma_mix_f32` (NHWC 16-bit 5x5/7x7) and runs ahead of it on gfx950 and gfx942, level on gfx1201.
- No code was copied from CK, rocKE or FlyDSL.

## Next steps and deferred ideas
1. **gfx950 NHWC 16-bit wrw (ConvNeXt 1.03x behind hipconv; n16 64x64 1.49x).** The Hankel kernel is still issue-bound (about 6 VALU and 6 SALU instructions per MFMA). Measured and rejected: reading each lane's A window at its own 2-byte LDS offset (no byte align) runs 1.23x slower from unaligned-access stalls; a DPP form of the tap shift cannot fold into the byte align (VOP3, no DPP on gfx9) and adds instructions. Remaining candidates: the stride-2 parity lanes (every row saves and restores), the per-step barriers, and the fold launch on small shapes.
2. **inf/NaN parity for VALU `tile`/`vtile` wrw and dy-side padding.** This is required for behaviour parity with cuDNN and PyTorch. Masks can cost several percent of kernel time, so measure the cost against the compiled-out build.
3. **NCHW 7x7 forward on RDNA.** This is the largest depthwise excess in the full corpus (on the order of 60 ms over about 165 shapes on one RDNA GPU).
4. **Host-computed magic numbers for the thread decode.** These measured +8-10% on gfx1201 NCHW floors. They need an argument-block change.
5. **Depth multiplier above 1 for fwd/bwd.** wrw already carries `mult` 2. The forward cores would need the same channel mapping.
6. **gfx942 NHWC 3-D 16-bit LDS core**, about 13% slower than an alternative staging form (unexplained).
7. **Deferred:** 1-D MFMA Toeplitz for long 1-D filters on gfx942/gfx950 (pays only where VALU becomes compute-bound); MFMA strided backward data and 3-D.

## Tests
- Host-only gate tests, which check each field one at a time where noted and check that every catalog config serves some case:
  - `TestConvDepthwiseValuProblem.cpp` covers the problem parse, reflection and padding, gather-only strided bwd, layouts, and 32-bit tile limits.
  - `TestConvDepthwiseMfmaProblem.cpp` covers every config field, atom-per-device, the fold/buffer bound, stride 2 forward only, and the compute type.
  - `TestConvDepthwiseRowsProblem.cpp` covers variants, refusal per field, phase row-separability and launches.
  - `TestGpuConvDepthwiseValu.cpp` also holds the host tests for `phase`, `rowtile`, `hankel` and `vtile` refusals, the 16-byte alignment gate, the multiplier's workspace, and the wrw partition planner (whole rounds, one partition without workspace).
- Device tests (`TestGpuConvDepthwise{Valu,Mfma,Rows}.cpp`) run every admitted shipped config, in each dtype, through the engines' own match, prepare and launch, against the CPU reference.
  - Covered: odd and narrow extents, folds that do not divide the batch, asymmetric padding, single-pixel and k x k-to-1-pixel geometries, true convolution and strided cases.
  - Each config is relaunched for a bitwise-repeat check.
  - Extra cases: `rowtile` fp16 bit-exactness (mutation-checked) and fp16 past 2 GiB; Hankel inf parity (fp16, bf16); the wrw multiplier; and wrw planned for one dispatch unit, where the `bpc` 1 configs run one partition and write dw in the partial kernel.
- Fusion: `TestGpuConvDepthwise{Valu,Rows,Mfma}Fused*` run every forward variant (both layouts, 1-D/2-D/3-D, strides, channels and planes off the tiles) fused with the menu chains in each dtype, plus NaN/infinity operands and clip boundaries, against the CPU convolution followed by the chain in double. `TestConvDepthwiseValuProblemFusion` checks that forward takes the epilogue only when the engine asks for it and that backward data declines a fused graph.
- Integration: the conv coverage census (`IntegrationConvCoverageCensus`) records, in its checked-in coverage map, which engine serves each census graph. The depthwise engines serve about 200 of those rows, mostly `VALU_FWD`/`VALU_BWD`, `VALU_WRW` and `ROWS_FWD`.
