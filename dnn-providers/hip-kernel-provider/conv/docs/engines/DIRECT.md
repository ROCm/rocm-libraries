<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_DIRECT (K2) — stride-1 direct convolution on matrix cores, halo tiles and filter-row reuse

> Engine ids: `hipkernel:CONV_DIRECT_FWD`, `hipkernel:CONV_DIRECT_BWD` · Generator: `conv/generators/gen_direct.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/direct/kernels/` (`ConvDirect.cpp`, `ConvDirectCore.h`, `ConvDirectRows.h`, `ConvDirectArgs.h`) · Pack: `packs/ConvDirectNative.cpp`, `packs/ConvDirectProblem.hpp`

## At a glance
- **What:** direct convolution with every filter tap run as a small GEMM over an LDS-staged input window. Two kernel kinds: the halo tile (`mma`, `fma`) and `rows`, hipconv's filter-row-reuse form with LDS DMA and a persistent tile walk.
- **Covers:** forward and backward data. fp32, fp16 and bf16; a three-bf16 fp32 form in the `CONV_DIRECT_FWD/BWD_TF32` sibling engines. Any activation layout: channels-last is native; packed channels-first 16-bit 2-D graphs run in place on gfx950 (`rows` `cf`) and are staged elsewhere; other layouts are staged. 2-D and 3-D; 1-D graphs (and 2-D ones of one row) under 1 x S filters on the `rows` line configs.
- **Geometry limits:** filters 3x3, 5x5, 7x7, 1x7, 7x1, 3x3x3 and 5x5x5, and 1x3, 1x5, 1x7 and 1x11 on one-row images, fixed at compile time. Groups 1, stride 1. Dilation 1, or the dilation a `rows` config fixes: 2 and 4 for 2-D 3x3, 3 and 5 for one-row 1 x S (an axis the filter does not span takes any dilation). Any padding, asymmetric included, and both convolution modes.
- **Large tensors:** served past 2 GiB as long as one forward-form input image stays under 2 GiB (32-bit offsets inside an image, 64-bit image bases).
- **Arches:** MFMA on gfx908 to gfx950 and WMMA on gfx11/gfx12, both on the implicit-GEMM core's atoms. FMA micro-tiles everywhere, gfx906 included, as the floor and the exact-fp32 path on RDNA. `rows` runs on gfx950 (16-byte LDS DMA) and on gfx908, gfx90a and gfx942 (dword LDS DMA); its channels-first form `cf` on gfx950 only.
- **Wins when:** dense stride-1 3x3 and 3x3x3 layers with 16-bit channels-last data (diffusion VAEs and UNets, video VAEs). On RDNA it also wins exact-fp32 3-D layers, where MIOpen falls back to naive or GEMM kernels.
- **Loses when:**
  - the largest NHWC bf16 VAE maps on gfx950 (384² to 1024², 4-8 images) trail `ConvHipConv` by 1.01-1.03x;
  - 384-channel 3x3x3 layers on gfx90a and gfx942 trail CK's 3-D grouped forward by 1.09-1.16x;
  - NCHW 16-bit 3x3 usually goes to `CONV_WINOGRAD`;
  - anything strided or grouped, and dilations the `rows` configs do not fix (DeepLab's 6-36), go to `CONV_IGEMM`.
- **Precision:** fp32 accumulation of exact products, rounded once at the store (split-C sums fp32 partials in a fixed order). Deterministic. No numerical note on the exact engines; the TF32 siblings report `TENSOR_CORE`.
- **Instances:** 1556 exact configs, 778 per direction: 250 halo-tile MMA, 196 2-D and 3-D `rows` (70 of them channels-first `cf`, 30 dilated), 288 `rows` line tiles, 44 FMA (the floor on every architecture); and 18 three-bf16 per direction in the TF32 siblings. Each matrix-core config ships only to the architecture groups whose autotuned sweeps it won (`archs`).
- **vs MIOpen:** usage-weighted speedup on the subset graphs where K2 is our fastest engine: 1.49x gfx950, 1.23x gfx942, 1.22x gfx90a, 1.03x gfx908, 2.16x gfx906, 2.80x gfx1201, 3.88x gfx1100. MIOpen ran in search mode.

## Why it exists
- An implicit GEMM (`CONV_IGEMM`) re-gathers the input for every tap. A direct kernel stages each input window once per channel chunk and reuses it across all R x S taps from LDS. On stride-1 3x3 / 3x3x3 16-bit layers that measured 1.1-2.5x faster than K1 on gfx942, gfx950, gfx1100 and gfx1201.
- MIOpen's fastest kernels for this class are `ConvHipConv` (hipconv, gfx950) and CK's grouped implicit GEMM (NHWC). K2 gives every architecture one plain-HIP engine for the class, with exact fp32 where the hardware allows it.
- The `rows` kind brings hipconv's `direct` / `direct_l1` design (weights as the matrix M operand read through L1, input rows reused across filter rows, LDS DMA, 8-wave ping-pong) beyond gfx950: to gfx908, gfx90a and gfx942 with dword DMA, and to 3x3x3.
- hipconv serves channels-first graphs through three wrapper transposes. `cf` reads and writes packed channels-first activations in place on gfx950, so those graphs pay no staging.

## How it works
Backward data at stride 1 is the forward convolution of dy with the filter rotated 180 degrees and its K/C roles swapped, padded by F - 1 - p. Both changes, and the reflection for true convolution, happen in a weight pre-pass, so one kernel serves both directions. The pre-pass also formats the filter into the exact order a config reads (MMA fragment order, per-tap LDS tiles, or the `rows` lane-major runs), zero-padded to whole chunks and blocks. The inner loops therefore carry no bounds checks.

**Halo tile (`mma`, `fma`).** A block owns `th x tw` output pixels of `ds` output planes of one image, times `bn` output channels:

```
for each channel chunk (bc channels):
    stage input halo (th+R-1) x (tw+S-1) x bc of one plane in LDS (double-buffered)
    for each tap (r, s):  acc += halo shifted by (r, s)  x  W[tap]   # one GEMM step per tap
store acc (via an LDS strip, 16-byte rows)
```

3-D has two modes, both configs:
- **ring** streams each input plane once into KD rotating accumulator sets; the plane loop is unrolled by KD, so set indices are compile-time.
- **non-ring** keeps one accumulator set and re-stages the KD input planes of each output plane (L2 hits).

Weight fragments come straight from global memory (L1/L2-resident), prefetched a few steps ahead in registers. `xpf 1` configs keep that register ring across channel chunks: a chunk's last steps load the next chunk's first steps, so a chunk starts with its weights in flight and only its halo waits on the barrier. `bl 1` configs instead stage per-tap weight tiles in LDS. The FMA kernel keeps an fp32 channel-major halo and slides TM + S - 1 values across the S taps.

**Rows (`rows`).** D = W x X^T per 16x16 MFMA atom: weights are the M operand (output channels) and the input is the N operand (16 output columns of one row).
- A wave owns PW output rows x 16 columns x 16·FN channels.
- One step is one filter column and one K slice. It loads R x FN weight fragments and PW + R - 1 input-row fragments, then runs R x FN x PW MMAs, so every input fragment serves R filter rows.
- The halo is filled by LDS DMA into two separate LDS stages.
- At most one block per dispatch unit walks its share of the tiles in XCD-contiguous order; a tile's last chunk prefetches the next tile's first.
- 3x3x3 runs per output plane over the depth taps whose input plane lies in the image.

**Channels-first rows (`cf`, gfx950).** For packed N C H W 16-bit 2-D graphs whose width is a multiple of 8.
- The halo is fetched by 16-byte LDS DMA as it lies in memory: runs of 8 columns of each channel row, from the 8-aligned column at or left of the tile's first input column. A run lies wholly inside or wholly outside its image row, so out-of-image runs read as zero through the buffer range check.
- Once a chunk lands, the block transposes it into the channels-last stage: two `ds_read_b64_tr_b16` give a lane 8 channels of one column, written as one 16-byte granule. The step loop is the `rows` loop.
- One raw stage (the next chunk's DMA overlaps the steps) and one channels-last stage; two barriers a chunk.
- The epilogue stores channels-first: 16 lanes write 16 consecutive columns of a channel row.

**Dilated rows (`dil`, 2-D).** A config fixes the filter dilation d. The halo grows by (R - 1) d rows and (S - 1) d columns, filter row r of output row p reads halo row p + r d, and each step's columns shift by s d, so a dilated 3x3 layer still stages its input once per channel chunk for all nine taps. With d up to the wave's rows the step's input fragments are one run of PW + (R - 1) d rows; past them, R separate runs of PW. Backward data pads dy by (R - 1) d - p. The halo bound keeps d at 2 and 4 for 3x3 (32-channel chunks on gfx950, 16 on the 64 KiB parts); larger dilations stay on `CONV_IGEMM`.

**Line tiles (`line`, one-row images).** For 1-D graphs and 2-D graphs of one row under a 1 x S filter (audio convolutions). A 2-D rows tile on a one-row image leaves every row but one idle, and its weight fragments then serve one MFMA each. A line tile is instead wp x pw segments of 16 wq columns of the row, end to end, staged as one halo line of the tile plus the filter's reach (S - 1) d; segment p reads at column offset 16 wq p, and output row p of the step loop is segment p. Each weight fragment serves the wave's 8 segments and the staged line serves every tap. Channels-first (16-bit, in place, rows of whole 16-byte runs) and channels-last; dilation fixed per config (1, 3, 5). Channels-first line tiles come 32, 64 and 128 output channels wide on gfx950 and MI325X: on K = 32 a 128-wide block leaves three quarters of every MFMA idle (MI325X forward 1.15x over the corpus's one-row layers with the 32- and 64-wide tiles, gfx950 32-channel layers 1.24-1.27x with the 32-wide tile).

**Narrow channels.** 32- and 64-wide output blocks (fn 1 and 2) at 32-channel chunks on gfx950 and 16-channel chunks on the dword-DMA parts, channels-last and channels-first: on K <= 64 layers a 128-wide block leaves half or more of every MFMA idle, and on C <= 64 reductions (backward data of K <= 64 layers) a 64-channel chunk half of every step. Such a tile's chunk is three short steps, too few to hide its DMA, transpose and barriers, so the 8-row narrow tiles run two or three blocks per unit (`bpu`). 3x3x3 has two 32-wide tiles of 1024 pixels (32 x 32 and 16 x 64) for 32-channel outputs (nnU-Net c32 and c64 to k32 at 128^3-160^3): 1.3-1.6x over the 64-wide tile on gfx950.

## Implementation design
- **Kernel structure.**
  - Halo tile: `WM x WN` waves of the atom, or a 256-thread FMA block. The halo is double-buffered with one barrier per chunk, and the next chunk's halo loads issue after the first weight loads so in-order `vmcnt` never stalls them.
  - Rows, block shape: `WP x WQ x WK` waves.
  - Rows, LDS: the halo is stored in 16-byte granules rotated per column, so 16-byte reads are conflict-free. A DMA instruction writes 64 consecutive granules; without 16-byte DMA it takes four dword sub-instructions with the same fetch addresses.
  - Rows, ping-pong (`pp 1`): the two channel halves run as two wave groups one barrier apart, and each step is "memory phase | barrier | MMAs at raised priority | barrier".
  - Rows, ping-pong halo fill on gfx950 (16-byte DMA, no unfold): the first `XG` waves each own one 8-column strip of the halo and fill all its rows. A lane's offset within a row is computed once per chunk, so each DMA row costs one scalar offset; a step's memory phase is about 26 VALU and 53 SALU instructions. Every other config deals the halo's DMA instructions round-robin over all waves: without ping-pong, strips leave the DMA on fewer waves, which measured 1-2% slower.
  - Rows, ping-pong staged epilogue (same configs): each wave stages through two slots at the end of the stage the last chunk read, double-buffered by output row, and no barrier follows. The next tile's DMA into that stage starts with the step-0 rows, which lie below the slots; every later DMA step comes after a barrier that the lagging wave group reaches only after its epilogue.
  - Rows, epilogue: `epi 1` (16-bit output) transposes through LDS so 8 lanes store 16 channels of a pixel as 16-byte pieces, with XOR-keyed slots for conflict-free writes. `epi 0` stores directly.
  - Rows, batch unfold (`un` 2 or 4, 2-D): an MFMA's 16 columns hold 16/un columns of un consecutive images, so narrow maps (24², 40²) fill the atom.
  - Rows, gfx908: 8-wave blocks run two waves per SIMD at 128 VGPRs and 128 AGPRs, so its tiles keep 8-12 accumulator sets per wave; bf16 runs on the 16x16x8 atom.
  - Split-C (`split 1`): a config splits the channel reduction over grid z. Each split writes fp32 partials, and `ConvDirectReduce` adds them in split order.
- **Host gates (every field).** Everything the naive engine's parser gates, plus:
  - direction forward or backward data;
  - fp32/fp16/bf16, with x, w and y of one type;
  - spatial rank 2 or 3 (or 1, a one-row image), groups 1, stride 1 on every axis, one dilation of the axes the filter spans (1 in 3-D);
  - a filter in the instantiated set;
  - n, c, k, extents, filter and padding within 32 bits;
  - one forward-form input image under 2^31 bytes.

  Per config, the kernel match also requires:
  - direction, dtype, filter and dilation equal to the graph's; line configs take one-row images only, and only they take 1-D graphs;
  - the atom runs on the device and takes the graph's element type (three-bf16 only for fp32 graphs);
  - the ship list names the device; `rows` LDS fits the device (160 KiB gfx950, 64 KiB elsewhere);
  - `cf`: gfx950, 16-bit, 2-D, no ping-pong, unfold, split-C or staged epilogue; both activations packed channels-first and input rows of whole 16-byte runs;
  - the channel vector divides the reduction channels;
  - the formatted weights stay under 2^31 elements and bytes, and the grid fits one dispatch.
- **Planner.**
  - Halo tile: picks `tw`, then the largest `th` whose halo fits the compile-time LDS bound, minimizing padded pixel slots plus halo loads. Depth is split until there are 4 (ring) or 8 blocks per dispatch unit (`DeviceUnits.hpp`: CUs on gfx9, WGPs on gfx10+).
  - Split-C, halo tile: only below 2 blocks per unit, with at least 2 chunks per split and at most 16 splits. A grid that needs no split declines the split config; its split-0 twin runs the same tile in one launch.
  - `rows` tiles are fixed, and the grid is min(tiles, bpu x units): `bpu` blocks resident per unit (1 to 3, each with its own halo stages, all within the device's LDS), so one block's chunk DMA, transposes and barriers overlap another's MMAs. A line tile is wp pw x 16 wq columns of the row. Rows split-C runs only where the tiles leave three quarters of the units idle: it measured 1.4x on n1 c640 64², but 0.7-0.9x at 128-250 tiles.
  - gfx950 16-bit rows tiles smaller than 16 x 32 x 128 decline past 8 persistent rounds, where the large tile is ahead. Tiles compare by the output channels they compute (on K < 128 the reference tile's columns past K are idle, so a tile as wide as K is not smaller), and bpu resident blocks as one bpu times larger tile.
- **Workspace.** In order: staged activations (none for `cf`), formatted weights, the three-bf16 input copy, then split partials. Each region is 256-byte aligned. Non-channels-last activations go through K1's tiled transpose (packed channels-first) or a strided copy. The weight pre-pass runs on every execution; its 16-byte form costs about 0.016 ms on a 384x384x27 bf16 filter on gfx1201. Backward data with channels-last filter rows formats VEC x VEC squares: VEC output channels' 16-byte runs, transposed in registers into the input channels' 16-byte run pieces.
- **Cross-chunk prefetch (`xpf 1`).** Halo tiles with register-path weights on channels-last activations. Thread traces of the 3x3x3 halo tiles on gfx942 show about half of the wave time in `vmcnt` waits: weights come from L2 every step, and the in-order counter makes each chunk's first weight waits also wait for the halo loads issued before them. The `xpf` twin prefetches the next chunk's first steps (PD + 1 ring slots dividing a chunk's steps, so slots line up across chunks), skips the depth sets of ring configs whose plane is outside the output with scalar branches and reads their A fragments a step ahead, and pads 16-bit halo rows by 8 bytes so 8-byte fragment reads of 16 consecutive pixels cover the 32 banks once. `xpf 0` configs compile as before (`CONV_DIRECT_XPF` is defined only when set).
- **Catalog.** The knobs are the atom, filter, `bm/bn/bc`, waves, `vec`, `ring`, `bl`, `split`, `xpf`, FMA `tm/tn/nt/pipe`, and rows `pw/fn/wp/wq/wk/pp/epi/un/cf/dil/line/bpu`. Configs exist because they won a measured cell. The dilated and line tiles cover DeepLab's dilation 2 / 4 stages and HiFi-GAN-style 3-, 7- and 11-tap stacks at dilation 1, 3 and 5. The K-fill tiles are 16-, 32- and 96-wide, for K <= 16 to-RGB heads and K = 32 / 96 VAE layers. The 192-wide bf16 tiles on gfx90a are for Wan VAE's K = 192 / 384. The 48-wide rows tiles are for 48² / 96² maps.
- **Codegen choices.**
  - On gfx950 and RDNA, weights are read by buffer loads: the lane's column is the vector offset and the step a wave-uniform scalar offset, so a load needs no VALU address math. Measured gain about 3.9% weighted on gfx1201.
  - Rows recompute lane-derived values from an opaque, masked lane id, so the compiler does not hoist dozens of invariants into live registers.
  - Rows use `s_barrier` without fences, so DMA into the next stage stays in flight. Each stage is its own `__shared__` object, so waits track the right DMA.
  - gfx950 rounds bf16 pairs with the hardware convert, which is bit-identical to the software round for every non-NaN input (checked exhaustively).
  - Halo tiles keep the compiler's memory waits exact. The halo loader selects its offsets by masks rather than lane-varying branches, and an empty asm with a memory clobber keeps the loads where they are issued. The `xpf` tail loads the cursor's next chunk on every path; past the last chunk the cursor is back at the first chunk, which is in range. A branch on either test splits the loads into paths, and the waits merge to `vmcnt(0)` at every chunk boundary, which also waits for the previous plane's output stores. Measured against the same tree without them (autotuned, engine alone, two interleaved rounds):
    - MI300A 3x3x3 bf16, large and mid planes: 1.06x forward and 1.07x backward (roofline efficiency 42% -> 45% and 37% -> 40%);
    - the K2-won corpus graphs: MI325X 1.03x forward and 1.02x backward geomean, gfx950 neutral.
  - The kernel bodies are always inlined into their entry points: the largest register-path fp32 tiles otherwise compile to a call that spills to scratch.

## Fusion

`CONV_DIRECT_FWD` and `CONV_DIRECT_FWD_TF32` also claim a forward convolution followed by a
chain of pointwise nodes, under the same engine ids and catalog configs. The chain is the
conv epilogue menu (`ConvEpilogue.hpp`): scale and bias (per-channel or scalar operands, device
or pass-by-value), relu with its clips and slope (relu, clipped relu, clamp, leaky relu), gelu,
gelu-tanh, swish/silu, sigmoid, tanh, elu, abs and softplus. The provider generates the pattern's
code into the kernel's compile options; operand pointers and values are runtime arguments, so
one code object serves every graph of a pattern.

The epilogue runs on the fp32 accumulator and the kernel rounds once:

- FMA kernel: per thread, before its output stores.
- Halo-tile kernel: before the accumulators go to the wave's LDS strip.
- Rows kernels (2-D, batch unfold, 3-D, in-place channels-first): once per tile on the
  accumulators, before either store, so the LDS-staged 16-byte store copies rounded values
  bitwise.
- Split-C configs: the main kernel writes raw fp32 partials and `ConvDirectReduce` applies the
  epilogue to each summed element, deriving the channel with no division per element.
- Three-bf16 (TF32) configs: the same fp32 accumulator, rounded once to fp32.

The weight format, the three-bf16 input split and the staging copies take no epilogue. Backward
data does not fuse. With no pointwise node the compile options, and so the code, are those of
the unfused engine.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.02x, gelu 1.06x, silu 1.04x; gfx950 (both layouts) relu 1.02x, gelu 1.09x, silu 1.05x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.06x on MI325X and 1.08x on gfx950 (usage-weighted; the conv README's fusion table).

## Strengths and weaknesses
Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X. W/L counts shapes faster or slower by more than 5%.

| GPU | graphs K2 is our best | W/L vs MIOpen | weighted speedup | MIOpen solver it mostly meets |
|---|---|---|---|---|
| gfx950 | 177 | 80/46 | 1.49x | `ConvHipConv` (108), CK 3-D group fwd/bwd |
| gfx942 (MI325X) | 150 | 96/19 | 1.23x | CK group fwd/bwd (2-D and 3-D) |
| gfx90a | 148 | 102/23 | 1.22x | CK group fwd/bwd |
| gfx908 | 107 | 45/35 | 1.03x | CK group fwd/bwd |
| gfx906 (FMA only) | 58 | 54/3 | 2.16x | GEMM, naive |
| gfx1201 | 171 | 130/14 | 2.80x | CK group fwd (WMMA), GEMM, naive (fp32 3-D) |
| gfx1100 | 177 | 156/6 | 3.88x | CK group fwd, GEMM, naive (fp32 3-D) |

- **Strong:**
  - narrow-channel, dilated and one-row (1 x S) 16-bit NCHW layers (Real-ESRGAN, U-Net, DeepLab dilation 2 / 4, HiFi-GAN), against the same build without the narrow, dilated, line and `bpu` tiles (autotuned, fresh cache, 20 iterations, two interleaved rounds, our engines only, 97 graphs, usage-weighted): gfx950 1.27x (roofline efficiency 25% -> 32%; HiFi-GAN layers 1.5-1.7x, 3x3 K <= 64 forward 1.28x, dilated 1.31-1.35x), MI325X 1.06x (29% -> 31%), gfx90a 1.04x (51% -> 53%). On 38 other layers (gfx950, MI325X) no layer is repeatably slower; gfx950 gains on Wan VAE c96 3x3x3 (1.14x, a 64-wide tile the small-tile rule now admits) and EfficientNet 24- and 48-channel NHWC layers (1.4-1.7x);
  - packed channels-first 16-bit 2-D graphs on gfx950 (`cf`): 1.14x weighted over the staged path on 101 Flux/SD/Wan/Real-ESRGAN layers, up to 1.58x (MI350X, autotuned, same node);
  - gfx908 16-bit `rows`: EfficientNet fp16 1.27-1.34x and Wan c384 240x416 bf16 1.18x over the halo tile (MI100, autotuned, same node);
  - Wan VAE 3x3x3 bf16 (fwd 0.54-0.85x of MIOpen's time, bwd 0.36-0.47x on gfx90a-gfx950);
  - SD/SDXL/Flux VAE and UNet NHWC 3x3 on CDNA2/3 and RDNA;
  - nnU-Net fp32 NCHW 3-D on RDNA (50-100x, against MIOpen's naive kernels).
  - `rows` measured 1.28x over the halo tile on gfx950 (Flux c256 768²), and its gfx90a ping-pong tiles 1.04-1.12x over the halo tile from 256 channels.
- **Weak:**
  - gfx950's largest-map VAE bf16 layers against `ConvHipConv` (384²-1024², c256-c512, 4-8 images, 1.01-1.03x);
  - 40² c1280 tiles that fill a persistent round badly (with the 4-image unfold they trail `ConvHipConv`'s batch-packed `direct` tile by about 1.1x);
  - gfx90a/gfx942 384-channel 3x3x3 forward against CK (1.09-1.16x);
  - gfx908 bf16, which runs on the half-rate 16x16x8 atom: `rows` reaches 60-66 TFLOP/s (65-72% of the atom's peak) against CK's 72-78%, so CK keeps most large bf16 layers. gfx908 fp32 is mixed.
  - On gfx942, the 16-bit `rows` tiles (no ping-pong) take most 2-D 3x3 layers from the halo tile in a kernel-only sweep, but for only about 7% of the weighted time. Two 3x3x3 `rows` tiles (16 x 16 x 256 and 16 x 32 x 128) ship for gfx942: they take 256- and 512-channel layers on mid-size 3-D outputs on MI325X (1.2-1.3x) and are neutral on MI300A, where the `xpf` halo tiles take those layers.
- **Coverage gaps:** 1-D only under 1 x 3/5/7/11 filters (line tiles), dilation only where a `rows` config fixes it, no generic tier for other filter sizes, and images past 2 GiB are declined. `CONV_IGEMM` is the floor for all of these; it reads 1-D channels-first (N C L) graphs in place on its `cfx` configs.

## Numerical precision
- Every config accumulates in fp32. 16-bit products are exact in the atoms, and fp32 runs exact on the fp32 MFMA atoms (CDNA) or FMA (everywhere). The output is rounded once to its type. Split-C writes fp32 partials and rounds once in an in-order reduce.
- **Test tolerance:** the device suite grades against hipDNN's CPU reference convolution. The tolerance is (2e-6 x reduction length) + 2 storage-eps x (|ref| + 1e-3), and three-bf16 uses 3e-5 x reduction length.
- **inf/NaN:** an infinity planted in the input reaches every output it touches as that infinity (NaN where +inf and -inf meet), as in the reference. Padding never multiplies a real infinity by a structural zero: halo positions outside the image and padded channels read zero on both operands. The three-bf16 split keeps infinities, which a dedicated test pins.
- **Determinism:** fixed-order sums, no atomics. Each config is launched twice over a re-filled workspace and must reproduce its bits.
- **Reduced precision:** the three-bf16 form computes xh·wh + xh·wl + xl·wh on bf16 atoms over 3C reduction channels, with fp32 output. Its error is about 2^-16 of each product. It is admitted only for fp32 graphs, and only the `CONV_DIRECT_FWD/BWD_TF32` siblings list it; they report `TENSOR_CORE`, by which a caller excludes them (`deselect_numeric_notes`).

## Comparison with MIOpen
- **`ConvHipConv`** (hipconv; MIOpen's pick on gfx950 in our sweeps) uses the same `direct` / `direct_l1` design as K2's `rows`, NHWC 16-bit only. On the 70 subset graphs where hipconv runs `direct_l1`, K2 is 1.05x faster usage-weighted (MI355X, autotuned, alternating with exhaustively tuned hipconv on one GPU). The graphs it trails are the largest VAE maps (384²-1024²) and a 96² c1280 UNet layer, each by at most 1.03x. Its main ping-pong tile matches hipconv's 16 x 32 x 128 `direct_l1` tile, including the strip-owned halo DMA and the barrier-free staged epilogue. hipconv serves NCHW through three transposes; K2's `cf` configs read NCHW in place. Not adopted: in-workgroup channel split, dynamic per-XCD tile claiming, and the K-partitioned L2 grid (a slice-major K walk measured within 1% on c512-c1280 layers). Kernel-argument preload does not apply: LLVM preloads only scalar kernel arguments, and every K2 kernel takes one argument block.
- **CK `ConvHipImplicitGemm{,3D}Group{Fwd,Bwd}Xdlops`** (gfx908-gfx942; the same solver names cover CK's WMMA instances on RDNA): an implicit GEMM, NHWC only. K2 beats it on most 3x3 / 3x3x3 layers by reusing the staged halo across taps. It loses on the 384-channel 3-D shapes above.
- **Winograd** (`ConvBinWinograd*`, `ConvWinoRage/Fury`) serves NCHW fp16 3x3. Our matching engine is `CONV_WINOGRAD`.
- **GEMM / naive fallbacks** (`GemmFwdRest`, `ConvDirectNaiveConv*`): exact fp32 3-D on RDNA has no optimized MIOpen solver. K2's FMA kernels are 50-100x faster there.

## Prior art considered
- **hipconv `direct` / `direct_l1`** (gfx950, plain HIP): weights as the M operand via L1, input-row reuse across filter rows, swizzled 16-byte DMA, 8-wave ping-pong with priority, K-fill tiles, batch-packed tiles for narrow maps, an analytic split plan. K2 `rows` reimplements these and generalizes them to dword DMA (gfx90a, gfx942), exact fp32 on the 16x16x4 atom, and 3x3x3. hipconv has no exact fp32.
- **CK grouped forward**: studied as MIOpen's main competitor on CDNA, not as a design source.
- **K1 (`CONV_IGEMM`)**: K2 reuses its MMA atoms, element conversion, buffer-load helpers and staging kernels.
- **RDNA WMMA ping-pong**: measured in K1 at 3-30% slower, so the rows ping-pong stays CDNA-only.

No code was copied.

## Next steps and deferred ideas
1. **gfx950's largest VAE maps vs `ConvHipConv`** (1.01-1.03x behind on 384²-1024² layers). Measured and not adopted: a row-fastest tile walk (no gain) and hipconv's DMA spread over every step with the drain after the last step's MMAs (0.2-0.3% faster, within run-to-run spread). csplit and dynamic XCD claiming remain unported.
2. **3x3x3 on gfx90a/gfx942 against CK at 384 channels.** On gfx90a, rows3d measured 1.07-1.14x over the halo tile on 96-channel Wan VAE layers and 1.08x on one 384-channel layer. Wider rows3d tiles, or K-fill tiles for 384, are the candidates. Dense 3x3x3 is the largest single block of our CDNA time.
3. **Generic tier** (runtime filter for other sizes, 1-D): low priority, because `CONV_IGEMM` already serves those graphs.
4. **Images past 2 GiB:** a block-based-buffer variant on the generic tier only, so normal graphs gain no candidates. `CONV_IGEMM` serves these today.
5. **`cf` beyond gfx950**: gfx90a/gfx942 have dword LDS DMA but no transpose read; the transpose would need `ds_read_u16` gathers. The `cf` epilogue stores 2-byte elements; an LDS-staged channels-first epilogue is untried.
6. **RDNA `rows`**: needs a wave32 lane map and a register-staged fill (no LDS DMA). Low payoff: RDNA losses in this class are Winograd-served NCHW shapes.

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestConvDirectProblem.cpp` (host, no device):
  - **Gates:** each gated field refused one at a time; the buffer-range and LDS declines; ship lists and arch families; malformed metadata; cuDNN compute types.
  - **Planner:** tile versus halo bound, split and unfold rules, the small-tile decline, the rows one-block-per-unit grid.
  - **TF32:** three-bf16 admitted only for fp32 graphs.
- `TestGpuConvDirect.cpp` (device):
  - every admitted config on fp32/fp16/bf16, forward and backward data;
  - NHWC and staged NCHW, asymmetric padding, convolution mode, a split-C case;
  - the three-bf16 forms under the switch, and planted infinities (native and three-bf16);
  - the determinism relaunch;
  - every config parses, and every filter/dtype has a config on every architecture.
- `TestConvDescriptorShards.cpp` checks the shipped shards' numerical notes (exact engines none, TF32 siblings `TENSOR_CORE`; all deterministic) and that only the TF32 siblings list three-bf16 configs.
- No in-repo device test goes past 2 GiB. The large-tensor runs use sampled fp64 checks outside the unit suites.
