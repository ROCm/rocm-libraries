<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_GROUP (K4): grouped convolution on block-diagonal matrix-core tiles

> Engine ids: `hipkernel:CONV_GROUP_FWD`, `hipkernel:CONV_GROUP_BWD`, `hipkernel:CONV_GROUP_WRW` · Generator: `conv/generators/gen_group.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/group/kernels/` (`ConvGroup.cpp`, `ConvGroupCore.h`, `ConvGroupArgs.h`) · Pack: `packs/ConvGroupNative.cpp`, `packs/ConvGroupProblem.hpp`

## At a glance
- **What:** a direct convolution in which each 16-channel block multiplies on a 16x16 matrix-core atom against the block-diagonal expansion of its groups' filters, so one multiply serves 16 / (C/g) whole groups. The input halo is staged once in LDS, or (row-streaming `rs` configs, channels-last) the band's input rows stream through LDS once each while the filter stays in registers. Backward weights keeps the diagonal blocks of the same products and reduces through a deterministic split and reduce.
- **Covers:** fwd, bwd-data, wrw. fp32, fp16 and bf16, plus three-bf16 fp32 sibling engines for wide groups (`CONV_GROUP_FWD/BWD_TF32`, fwd/bwd only, offered by default, reporting `TENSOR_CORE`). Activations NHWC/NDHWC are staged, or NCHW/NCDHW are read in place (`cf`). Filters 3x3 and 3x3x3 (1x3x3 runs as 2-D); 2-D and 3-D; unit dilation; any padding; both convolution modes.
- **Groups:** small groups have C/g in {2, 4, 8, 16} with C a multiple of 16. Wide groups have 17 to 64 channels a group (a multiple of 8) in fwd/bwd, and 24 or 32 channels a group in wrw (MFMA only).
- **Strides:** row stride equals column stride, 1 or 2 per config. Forward takes any depth stride. Backward data takes strides 1 or 2 on every axis, run by stride phases. Wrw takes row/column stride 1 or 2 and any depth stride.
- **Large tensors:** element offsets are 64-bit from the image base, and the grid is a flattened 64-bit block id. No >2 GiB device test exists (unverified).
- **Arches:** MFMA 16x16x16 f16/bf16 and 16x16x4 f32 on gfx908 to gfx950 (gfx908 has its own bf16 atom). WMMA on gfx11 and gfx12. An fp32 FMA kernel compiled per channels-a-group runs on every arch, gfx906 included, and is RDNA's only fp32 path. gfx906 has no 16-bit configs.
- **Wins when:** groups of 2 to 16 channels (ResNeXt g32, RegNet, ResNeXt3D), where a per-group implicit GEMM fills 1/16 to 1/4 of its tile. It is the best non-MIOpen engine on roughly 40-110 graphs per direction per GPU in the full corpus.
- **Loses when:** gfx950 NHWC 16-channel groups (ResNeXt c512 g32) at 28x28-32x32 against MIOpen's hipconv grouped kernel, by about 1.2x with the row-streaming configs (at parity from 56x56). Also mid-width compute-bound groups (Winograd does fewer MACs). Most grouped wrw goes to `CONV_GROUP_VALU_WRW` (see GROUP_VALU.md).
- **Precision:** fp32 accumulation and one rounding at the store. Wrw keeps fp32 partials and reduces them in a fixed order. No numerical notes on the exact engines (the TF32 siblings report `TENSOR_CORE`). Results are bitwise deterministic, and the device test relaunches every config to check it.
- **Instances:** fwd 1524 (672 row-streaming) + 144 three-bf16; bwd 2200 (828 row-streaming) + 120 three-bf16; wrw 3132 in 7 packs (each tile at 3 `bpu` targets that share one compiled kernel). Counted from the generator's output.
- **vs MIOpen:** competes with `ConvHipImplicitGemmGroup{Fwd,Bwd,Wrw}Xdlops` and their 3-D forms, `ConvHipConv`, and the grouped Winograd solvers. On the full corpus the grouped class (all engines) runs at 1.41x on gfx950, 2.06x on MI325X, 1.44x on gfx1201 and 1.77x on gfx1100 against MIOpen search.

## Why it exists
- In a grouped convolution with C/g of 2 to 16, an implicit GEMM per group has a reduction of C/g x taps and an N of K/g. The 16x16 or 32x32 matrix tile is mostly empty, and each load reads a 4- to 32-byte channel slice.
- Expanding a 16-channel block's filters into a block-diagonal 16x16 matrix turns 16 / cg groups into one dense-shaped multiply. The wasted MACs (the off-diagonal zeros) cost less than the wasted tile and the narrow loads.
- Backward weights uses the same block-diagonal products, because K3's per-group GEMM runs 2-5x behind MIOpen on ResNeXt g32 wrw.

## How it works
Forward, per output plane and per depth tap:

```
stage halo[(TH-1)*st + R][(TW-1)*st + S][16*CB (+pad)]       # once per depth tap, in LDS
for each wave item (output row, channel block):
  for each tap (r, s):
    A = halo fragment: 16 output columns x 16 channels       # from LDS
    B = expanded filter of this block and tap: 16 x 16       # block-diagonal, zero off-group
    acc[item][frag] += A x B                                 # 16x16 atom
store acc (fp32 -> T)
```

- **Filter expansion.** A small kernel writes each phase's expanded filter into the workspace. Small groups get block-diagonal [block][tap][16][16]. Wide groups get dense per group, padded to `nbi` blocks each way. For true convolution, the expansion reflects the filter. For backward data it also swaps the K/C roles.
- **Backward data.** dy is gathered into dx with the filter rotated. At stride 2 the work splits by stride phase: dx[st*j + p] sums the ceil(F/st) compact taps that reach phase p. Each phase is a unit-stride gather with its own padding and expanded filter, and runs as one launch. With `pf` (row/column stride 2), one launch per depth phase runs all four row/column phases from one staged dy halo into an LDS output tile, then writes whole dx lines.
- **Backward weights.** Per 16-channel block, D[k][c] += dy[k][q] x[c][q + tap], keeping the cg x cg diagonal blocks. A workgroup owns `cb` blocks, one depth tap, and a split's run of `th` x `tw` dy tiles. The tile's positions are flattened into the atom's k-steps, so 56-, 28-, 14- and 7-wide planes waste no lanes. Positions past the plane read a zero pixel in both operands. Partials [split][g][k][c][tap] go to an fp32 workspace, and `ConvGroupWrwReduce` sums them in a fixed order into dw. At one split the workgroups write dw themselves, with the reduce's rounding of one slot, and the plan has no workspace and no reduce launch.
- **FMA kernel (fp32).** The kernel has the same grid, halo and expansion layout, but multiplies only the group's own inputs (no block-diagonal zeros). Output pixels run fastest across threads, so a wave reads its filter with wave-uniform scalar loads.

## Implementation design
- **Matrix-core kernel (`k4ConvMma`).** 4 waves own TH output rows x 16·WP columns x CB channel blocks of one output plane. The halo is a static LDS array; each pixel is padded by 16 bytes so that a fragment's 16 lanes hit different banks. At stride 2, each halo row stores its even columns and then its odd ones, so a fragment's lanes (columns 2 apart) read consecutive pixels. The halo is staged once per depth tap, behind a barrier, and is single-buffered. B fragments come from the expanded filter in global memory, read through cache.
- **Row-streaming kernel (`k4ConvMmaRows`, `rs` 1-4).** Small groups, and wide groups of exactly 2 or 4 blocks (32 or 64 channels a group, `nbi` 2 or 4: a workgroup holds whole groups, a wave one output block of its group against every input block); channels-last; forward at stride 1 or 2, backward data by its unit-stride phases. A workgroup of `cb` waves owns 16 `wp` output columns x a band of `th` output rows x `cb` 16-channel blocks (one a wave). Each wave loads its block's expanded filter (every tap) into registers once. The band's input rows (and depth taps) stream top to bottom: a row segment of (16 `wp` - 1) stride + S pixels is staged in a double-buffered LDS row (at stride 2 its even columns first, then its odd ones, so a fragment's lanes read consecutive pixels), with `rs` later rows' loads in flight in registers. Each wave keeps a ring of R output-row accumulators: input row t adds filter row r into output row (t - r) / stride where the stride divides t - r, so after row t the output row (t - (R - 1)) / stride is complete and is stored straight from registers (filter as the A operand, input as B: a lane holds consecutive output channels of one pixel). Input rows are read once a band (plus the halo columns) and every output written once. Loads and stores go through per-row buffer records (rows outside the tensor have an empty record; columns past the row read zero or are dropped), so the loop has no branch around memory operations and its waits count only the rows in flight. Each input and output row must be under 2 GiB (gated); bases are 64-bit.
- **Wide kernel (`k4ConvMmaWide`).** A workgroup owns `nbo` of a group's output blocks and stages the group's `nbi` input blocks. Channels need not be a multiple of 16, only of the 16-byte vector.
- **Channels-first in place (`cf`).** The halo is gathered one channel per thread-load along rows, which makes the loads coalesced. The output tile is parked in the halo's LDS as [row][channel][column] and written one row of columns at a time. Configs without `cf` read NHWC staged by the implicit-GEMM engine's staging kernels; on NCHW graphs that costs a transpose each way. In-place `cf` measures 1.5-2.5x faster than staged (indicative).
- **Host gates (`parseGroupProblem`):**
  - everything `parseIgemmProblem` gates (types, layouts, packing);
  - groups ≥ 2, with C == K and the group sizes as above (depthwise, where cg = 1, is declined);
  - unit dilation on all three axes;
  - row stride == column stride, and backward data strides ≤ 2;
  - each phase's smallest-tile grid within the launch limit (at most 2^32 - 1 work items).
- **Kernel gates:**
  - the atom runs on the device and its element type matches the graph's;
  - the filter (t, r, s) and stride equal the config's;
  - the channel tile divides C (wide configs: `nbi` equals ceil(cg / 16));
  - the halo plus any output tile fits 64 KB of LDS;
  - `cf` only when both activations are packed channels-first;
  - `pf` only for backward data at row/column stride 2 where every depth phase has all four phases;
  - three-bf16 only for fp32 graphs, on wide groups, on gfx90a, gfx942, gfx950, gfx11 or gfx12.
- **Wrw gates (`parseGroupWrwProblem`):**
  - everything `parseWrwProblem` gates;
  - x and dy both packed channels-first or both packed channels-last;
  - dw packed K C T R S or K T R S C;
  - small groups, or wide groups of 17-32 (a multiple of 8);
  - unit dilation;
  - row stride == column stride.
- **Wrw planner.** `groupWrwLaunch` sizes the splits with K3's `splitsWanted`, targeting `bpu` workgroups per dispatch unit, at least 4 tiles per split, and partials within 1 GiB. The reduce groups slots in powers of two (G = slot count rounded up), and its association depends only on the slot count. Below 4 blocks per workgroup, 4 / `cb` waves share a block and add their sums in LDS in wave order. On gfx950, 16-bit atoms read LDS fragments with `ds_read_b64_tr_b16` transpose reads.
- **Workspace.** fwd/bwd: staged activations (unless `cf`), then each phase's expanded filter, then the three-bf16 split copies. wrw: fp32 partials only, capped at 1 GiB; none at one split.
- **Catalog knobs:**

  | Knob | Meaning |
  |---|---|
  | `atom` | 16x16 atom, or `fma` |
  | `t`, `r`, `s`, `stride` | compile-time filter and stride |
  | `th`, `wp`, `cb` | output rows, 16-column fragments, channel blocks |
  | `cf`, `pf` | in-place channels-first; phase fusion |
  | `nbi`, `nbo` | wide groups' blocks per group and output blocks per workgroup |
  | `cg`, `oc`, `px` | FMA: channels a group, output channels and columns per thread |
  | `tw`, `bpu` | wrw: tile columns; split target |
  | `math` | 1 for three-bf16 |
  | `rs` | row streaming: 1-4 input rows in flight (`th` the band's rows, `wp` 1/2/4 fragments, `cb` waves); 0 otherwise |

  The tile lists come from bench winners per target (one-block tiles, for example, are RDNA's best at 16 channels a group). Every tile that fits LDS ships for every atom. Shapes are runtime. There is no generic/specialized tier split: every config is specialized on filter and stride, and other filters go to K1, `CONV_GROUP_VALU`, or Winograd.
- **Codegen.** `readfirstlane` on the FMA kernel's chunk index makes the filter loads scalar. Per-tap B offsets are compile-time. Out-of-plane depth taps are skipped uniformly across the workgroup.

## Fusion

`CONV_GROUP_FWD` and `CONV_GROUP_FWD_TF32` claim a forward convolution followed by a chain of
pointwise nodes (scale, bias, relu / clipped relu / clamp / leaky relu, gelu, gelu-tanh,
swish / silu, sigmoid, tanh, elu, abs, softplus; per-channel and scalar operands) under their own
engine ids, with no extra catalog config. The pack compiles the generated epilogue
(`ConvEpilogue.h`) into the convolution kernel only; the filter expansion, the staging copies and
the three-bf16 split stay as they are. Every store path applies it to the fp32 accumulator before
the single rounding: small groups on matrix cores and on FMA, wide groups (the padded channels past
the group width are staged but never written, and load no operand), channels-last and in-place
channels-first. The channel is the output channel itself (`group x channels a group + column`),
known per accumulator column, so the per-channel operands load once per column. Staged
channels-first outputs keep their bitwise transpose. Backward data and backward weights decline
fused graphs. With no epilogue the kernels compile to the same code as without fusion support.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.06x, gelu 1.26x, silu 1.17x; gfx950 (both layouts) relu 1.07x, gelu 1.20x, silu 1.11x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.33x on MI325X and 1.44x on gfx950 (usage-weighted; the conv README's fusion table).

## Strengths and weaknesses
- **Best-engine counts.** On the full 9,471-shape internal corpus (autotuned, our engines only), K4 is the best non-MIOpen engine on:

  | GPU | NCHW fwd / bwd / wrw | NHWC fwd / bwd / wrw |
  |---|---|---|
  | gfx950 | 73 / 38 / 53 | 58 / 62 / 46 |
  | MI325X | 69 / 43 / 51 | 64 / 56 / 62 |
  | gfx1201 | 41 / 23 / 23 | 75 / 62 / 42 |
  | gfx1100 | 37 / 29 / 14 | 110 / 65 / 42 |

- **Kernel-time measurements per feature** (indicative):
  - ResNeXt3D kernels run 1.1-18x MIOpen search on gfx950, gfx942, gfx1201 and gfx1101.
  - RDNA fp32 grouped runs 1.4-5.5x MIOpen.
  - fp16 forward runs 2.7-6.7x MIOpen.
  - Stride-2 forward with LDS de-interleave runs 1.6-2.7x MIOpen.
  - Phase-fused stride-2 backward data runs 1.3-4.7x over the per-phase launches.
  - Wide-group phase fusion runs RegNet cg56 stride 2 in 16.3 ms against 18.5 ms per phase on gfx1101.
- **Wrw.** It is 1.3-14.5x over K3 on small groups. Weighted excess against MIOpen on the 18 grouped wrw graphs is 0.37 ms on gfx950, 0.09 ms on gfx90a, 5.3 ms on gfx1201 and 0.09 ms on gfx1100 (indicative). In the full corpus, though, `CONV_GROUP_VALU_WRW` wins far more wrw graphs (about 290-360 per GPU).
- **Weaknesses:**
  - gfx950 NHWC ResNeXt g32 at 16 channels a group (c512, 14x14 and 28x28) is 2.4-2.5x behind MIOpen's `ConvHipConv` in fwd and bwd (full corpus). On MI325X, NHWC ResNeXt3D cg32 wrw is about 3.3-3.4x behind MIOpen's 3-D grouped wrw.
  - Mid-width groups (cg 32-64) are compute-bound, and Winograd F(2,3) does 2.25x fewer MACs. Wide-group configs win mainly on gfx950.
  - Wide wrw on WMMA, and wrw at cg 56/64, measured behind K3 on every graph, so they are not shipped.
  - Only 3x3 / 3x3x3 filters and unit dilation are covered.
- **First-run cost.** Each admitted config costs about 0.5-0.8 s of cold compile; the number admitted per grouped graph is not measured for the current catalog.

## Numerical precision
- **Accumulation.** Every atom accumulates in fp32, and fp32 graphs on MFMA use the exact 16x16x4 f32 atom. The FMA kernel accumulates in fp32. Each output rounds once to the element type. Wrw partials are fp32, and the reduce rounds once into dw.
- **Determinism.** No atomics; the sums run in a fixed order. The wrw association depends only on the split count, which depends on the shape and the device's dispatch units. The UED therefore carries no `nondeterministic` note, and the device test asserts bitwise equality on relaunch for every admitted config.
- **Tolerance.** Results are compared with a CPU reference at 2e-6 x reduction length + 2 x storage epsilon x (|ref| + 1e-3). The reduction length is (C/g) x taps for fwd/bwd and N x output positions for wrw. This is not yet the #15481 fp64 form.
- **Three-bf16 (`math` 1).** Only `CONV_GROUP_FWD/BWD_TF32` list it (144 fwd, 120 bwd configs; the exact engines 852 and 1372), for wide-group fp32 fwd/bwd. x (or dy) and the expanded filter are split into hi/lo bf16 parts, and three part products accumulate in fp32. The error is about 2^-16 per product, and infinities propagate as in fp32 (tested). It measured 1.4-2.4x over exact on gfx950 and 1.1-1.8x on gfx90a. For small groups and wrw the form measures 0.28-0.77x of exact, so they have none.
- **inf/NaN parity, row-streaming configs.** When a fragment of the input holds an infinity or NaN (a wave-wide check per fragment), and groups are narrower than the block, that fragment multiplies one group at a time: the input keeps only the group's channels and only the group's output rows take the product, so a structural zero never meets another group's infinity. Wide groups of whole blocks have no structural zeros. Tested with +inf and -inf inputs on groups of 2, 4, 8, 16 and 32 channels, fwd and bwd-data, stride 1 and 2, in all three types.
- **inf/NaN parity (open) for the other matrix-core configs.** On exact matrix-core small-group configs, an inf input times a structural zero of the block-diagonal expansion gives NaN where cuDNN and PyTorch give inf. Wide-group padding channels and the FMA kernel do not multiply structural zeros of real data the same way (unverified for wide-group padding).

## Comparison with MIOpen
- **`ConvHipImplicitGemmGroup{Fwd,Bwd,Wrw}Xdlops` and the `3DGroup` forms.** These are CK grouped implicit GEMMs, one GEMM per group, so their tiles are mostly empty at small cg. K4 replaces the per-group GEMM with block-diagonal 16x16 tiles, or with the FMA kernel for fp32 where there is no matrix core.
- **`ConvHipConv`.** hipconv's grouped direct kernel, the strongest MIOpen competitor on gfx950/gfx942 NHWC cg16. It wins those graphs by 2.4-2.5x (see weaknesses).
- **Grouped Winograd (`ConvBinWinogradRxSf2x3/3x2`, transposed variants).** These win the wide compute-bound groups. In hipDNN they belong to `CONV_WINOGRAD`.
- **Headline.** The grouped class with every hipDNN engine runs 1.41x (gfx950), 2.06x (MI325X), 1.44x (gfx1201) and 1.77x (gfx1100) against MIOpen search (full corpus). On the 1,998-shape subset it runs 2.00x (gfx90a), 1.52x (gfx1201) and 1.80x (gfx1100). These factors are not K4 alone.

## Prior art considered
- MIOpen's CK grouped solvers, as the baseline whose tile under-use motivated the design.
- hipconv's grouped direct kernel, for halo staging and channels-first in-place I/O.
- K1's group fusion (`fuse`), which is the same block-diagonal idea inside an implicit GEMM. K4 keeps it as a direct kernel with an LDS halo.
- K3's split planner and fixed-order reduce, reused for wrw.

K4 reuses K1's atoms and staging. No CK or hipconv code was copied.

## Next steps and deferred ideas
1. **inf/NaN parity for the halo-staged small-group configs.** Port the row-streaming kernel's per-fragment group-at-a-time path, with an A/B against the compiled-out build. This is required for behaviour parity with cuDNN and PyTorch.
2. **gfx950 NHWC cg16 at 28x28-32x32 (about 1.2x behind `ConvHipConv`).** Candidates: LDS DMA for the row segments, fusing the filter expansion into the kernel's prologue with vector loads (an element-wise gather there measured 13 us slower than the separate launch), and the gfx950 16x16x32 atom with two taps a step.
3. **Row streaming in place for channels-first activations** (NCHW graphs are served through staging today), and wide groups whose channels are not whole blocks (24, 48, 56).
4. **Coverage.** Add C/g not dividing 16 and not a multiple of 8 (3, 6, 12), non-unit dilation, and filters other than 3x3 / 3x3x3. Demand for these is not measured; they are served by K1 and `CONV_GROUP_VALU` today.
5. **Mid-width groups.** Leave these to grouped Winograd (K5) rather than K4.
6. **Large-tensor test.** Add a sampled >2 GiB boundary case to confirm the 64-bit offsets.

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestGpuConvGroup.cpp`:
  - Device test (`TestGpuConvGroup` Float/Half/BFloat16): runs every shipped config admitted for 2-D and 3-D, NCHW and NHWC, small and wide, stride 1/2 and depth-stride cases, in all three directions, against the CPU reference, with a bitwise relaunch check.
  - `TestGpuConvGroupThreeBf16`: the bound and inf propagation.
  - `TestGpuConvGroupRowsInfinities`: row-streaming configs with +inf and -inf inputs, groups of 2-32 channels, fwd and bwd-data (stride 1 and 2), three types.
  - `TestGpuConvGroupWrwOneSplit`: wrw planned for one dispatch unit, so configs run at one split and write dw themselves, against the reference and the relaunch check.
  - Opt-in timing test: exact against three-bf16.
- Host tests in the same file:
  - one decline per gated graph field (`DeclinesEveryUnservedGraph`, fwd/bwd and wrw);
  - kernel gates per knob, including FMA, wide, row-streaming (`RowsKernelGatesEveryField`: every field, stride 2 admitted in both directions, wide configs only for groups of exactly `nbi` blocks with whole groups a workgroup, rows of 2 GiB declined) and wrw;
  - every config fits its LDS, and the knobs tell configs apart;
  - three-bf16 is admitted only for fp32 wide-group fwd/bwd graphs;
  - launches past one dispatch are declined;
  - wrw splits fill the device and size the workspace, and one split writes dw directly (no workspace);
  - cuDNN compute types per element type.
- End to end, the conv engine group's integration suites run with `HIP_KERNEL_PROVIDER_ENABLE_CONV`.
