<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_GROUP_VALU_FWD / BWD / WRW — grouped convolution on vector ALU, lanes along the row

> Engine ids: `hipkernel:CONV_GROUP_VALU_FWD`, `hipkernel:CONV_GROUP_VALU_BWD`, `hipkernel:CONV_GROUP_VALU_WRW` · Generators: `conv/generators/gen_group_valu.py`, `conv/generators/gen_group_valu_wrw.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/group_valu/kernels/`, `.../group_valu_wrw/kernels/` · Packs: `packs/ConvGroupValuProblem.cpp`, `packs/ConvGroupValuNative.cpp`, `packs/ConvGroupValuWrwProblem.cpp`, `packs/ConvGroupValuWrwNative.cpp`

## At a glance
- **What:** direct convolution on VALU. A wave owns a small tile of one group's channels, its lanes lie along the output row, and the filter-column halo comes from neighbouring lanes (DPP / lane permute), so each input element is loaded once a row. WRW is MIOpen's hand-written `ConvAsmBwdWrW3x3` (conv3x3wrw.s) design, written in HIP and generalized. FWD/BWD apply the same row layout to forward and backward data.
- **Covers:** fwd, bwd-data, wrw · fp32/fp16/bf16 (compute type unset or fp32, no TF32 sibling) · NCHW (width packed) and NHWC (channels packed) · 1-D/2-D/3-D · any group count (the tile must divide the channels a group) · padding, stride and dilation on height and depth are runtime and unrestricted; the config fixes the width's · tensors over 2 GiB if the images one pass spans stay under 2 GiB.
- **Filter classes:** FWD/BWD 3x3 and 3x3x3, width stride 1 or 2. WRW 3x3 (width stride 1 or 2), 1x1 and 5x5, plus depthwise-only 5x5 at stride 2 and 7x7. 3-D WRW reuses the 2-D configs, one tile per depth tap.
- **Row width:** a whole row must fit in one wave (lanes x `p` output columns). That is at most 512 columns on wave64 and 256 on wave32 with the widest shipped `p`. Wider rows are declined.
- **Arches:** every arch, VALU only. Both packs ship both layouts. The `valu` pack (gfx906, gfx908) ships every tile. The `mma` pack (gfx90a, gfx942, gfx950, gfx11, gfx12) ships the few tiles that won there.
- **Wins when:** a per-group GEMM would leave the matrix tile mostly empty: ResNeXt / RegNet groups of 4 to 56 channels, ResNeXt3D 3-D grouped layers, few-channel dense and pointwise wrw, 3-D depthwise wrw, and most wrw on gfx906 (no matrix cores).
- **Loses when:** channels per group are wide (FWD/BWD at 56 or more go to Winograd; WRW to K3 / BLAS), on large pointwise wrw (a GEMM), on dense 3x3 wrw on gfx906 (MIOpen's asm is about 1.3x faster), on channels-last layers on the matrix-core arches (on MI325X another engine is faster on every grouped NHWC graph of the corpus; see Strengths), and on rows too wide for a wave.
- **Precision:** fp32 accumulation and one rounding at the store. 16-bit products are exact in fp32, and the fp16 `dot2` path adds two exact products to the fp32 sum. Fixed-order reductions with no atomics give bitwise-repeatable results, so the engine carries no numerical notes.
- **Instances:** FWD 180 `valu` + 30 `mma`, BWD 168 + 30, WRW 293 `valu` + 212 `mma` (913 in all). Every config is admitted on correctness alone and autotune picks among them.
- **vs MIOpen:** WRW replaces `ConvAsmBwdWrW3x3`, which is limited to gfx8/gfx90x, 2-D NCHW 3x3 pad 1, fp32/fp16, even N for fp16, and one code object per shape. On gfx906 it is 5.5x faster on ResNeXt-50 g32 56² n128 fp16 wrw and 17.6x on its stride-2 layer. On ResNeXt3D grouped wrw it is about 33x faster than MIOpen's best solver.

## Why it exists
- On gfx906 (no matrix cores) MIOpen's only fast grouped wrw is a per-shape assembly kernel, and an implicit GEMM leaves its tiles mostly empty at 4 to 32 channels a group: one ResNeXt-50 wrw layer takes about 20 ms on our implicit-GEMM engine against MIOpen's 2 ms.
- The asm kernel's limits are too narrow to keep: 2-D, NCHW, 3x3, pad 1, no bf16, even batch for fp16, no 3-D, no large tensors, and one code object per shape. The engine keeps its schedule and drops those limits.
- Without this engine, 3-D grouped layers (ResNeXt3D, ir-CSN) fall to a BLAS fallback at 2-17 ms per layer on every arch. The lane-along-row design runs them in 0.1-0.3 ms.
- FWD/BWD apply the same layout to forward and backward data, for gfx906 grouped layers and 3-D grouped layers.

## How it works
**WRW** (`ConvGroupValuWrw.cpp`), for a tile of `cb` input by `kb` output channels of one group:

```
wave: acc[cb][kb][fr][fs] (fp32, in registers; cb*kb*fr*fs <= 160)
lanes: lane l owns output columns [l*p, l*p+p) of one row and the p*sw input columns under them
for each reduction row (image, output row) of this wave's share:
    load x window (one vector load per channel and filter row), dy (p columns per output channel)
    halo: left/right neighbour lanes' columns via DPP row shift (gfx9) or ds_bpermute (wave32)
    acc[c][k][r][s] += x[c][row*sh + r*dh - ph][col*sw + s*dw - pw] * dy[k][row][col]
end: butterfly-reduce acc over the lanes -> LDS sum over the waves in wave order
     -> write dw (one split) or fp32 partials per split -> fold kernel adds the splits in order
```

- Narrow rows put several rows side by side in one wave. Lanes per row is a power of two, so a row never straddles a lane group.
- The `pass` form gives each lane group of a wave its own row of a pass. That supports any height stride, dilation and padding in either layout.
- The `ring` form (NCHW, unit height stride and dilation, 2-D) has a lane group walk a run of at least 8 consecutive rows of one image. The filter's input rows sit in a register ring unrolled by the filter height, so each input row is read once rather than once per filter row. The next row's loads are issued before the current row's multiplies.
- 3-D: the depth tap is a grid axis. Reduction rows cover images x output depth x output rows, and a row whose depth slice falls in padding adds nothing.

**FWD/BWD** (`ConvGroupValu.cpp`):

- A wave owns `kb` output channels of one group, and its lanes own `p` output columns each. Per input channel and filter row, a lane loads its input window once, takes the halo from its neighbours, and multiplies every filter column and output channel from registers.
- No sum crosses lanes, so there is no reduction step.
- A prepare kernel writes the filter as fp32 `[group][kb block][ci][t][r][s][kb]` into the workspace. The main kernel reads it with wave-uniform scalar loads.
- BWD is the same kernel over dy, with the filter's roles swapped and reflected, padding `span - 1 - pad`, and dy read as if upsampled by the stride.
- Along the width, the upsampling is compile-time (the config's `sw`, `p` a multiple of it), so taps that would read the inserted zeros are never emitted.
- Along height and depth it is a runtime per-row check. Rows are ordered by phase so that a pass's rows mostly agree, and a wave skips filter rows that none of its rows read.

## Implementation design
- **Loads:**
  - Buffer loads through a `readfirstlane` descriptor (no waterfall loops), with out-of-range offsets returning zero.
  - The access width (16, 8, 4 bytes or the element) is the widest that the window bytes, every stride and the declared pointer alignment all divide.
  - NCHW loads run along the width. NHWC loads run along the channels, one column at a time, each column checked.
- **Ragged rows (`RAGGED`, `COLMASK`):** where the width is not a whole number of lane windows, loads go element by element and elements past the row read zero.
  - WRW drops the products of output columns past the row, selecting their x operand to 0, so an inf/NaN at a row end cannot reach a tap the reference does not have.
  - FWD/BWD do not store those columns.
- **Index math:** offsets are 32-bit within the images one pass spans, added to a 64-bit image base. The host declines a config whose pass reach exceeds 2^31 bytes.
- **Barrier:** WRW's workgroup reduction uses an LDS-only barrier (local release/acquire fences around `s_barrier`), not a full memory fence.
- **dot2 (WRW, fp16, NCHW, width stride 1, `p` even, no ragged edge):** pairs of adjacent output columns multiply on `v_dot2_f32_f16`. The window is kept as dwords, and taps at odd offsets read one half-dword-shifted copy (`alignbit`) shared by every output channel. Configs without `dot2` convert to fp32 and use FMA.
- **Host gates (WRW `planFor`, FWD/BWD `planFor`):**
  - Filter rows and columns, width stride, width dilation and width pre-padding must equal the config's.
  - Layout must match: the lane axis must be packed in both x and dy.
  - Channels per group must be divisible by the tile (`cb`/`kb`; FWD/BWD apply this to the kernel roles, swapped for BWD).
  - Wave must be 32 or 64, and the row must fit in one wave.
  - Rows (images x depth x rows) and grid sizes must fit 31 bits.
  - Compute type must be unset or fp32.
  - Extents and channel counts must fit 32 bits.
  - `ring` additionally needs unit height stride and dilation and a 2-D problem.
  - `dot2` additionally needs whole-dword windows.
  - Each gate has a decline test (below).
- **Planner (WRW):**
  - Splits aim for twice as many workgroups as CUs, with each wave doing at least four passes.
  - The fp32 partials (splits x tiles x `cb kb fr fs` floats) are capped at 256 MiB by halving the splits. One split writes dw directly with no fold.
  - A fold kernel (256 threads per block) adds the splits in order and writes dw through its strides.
- **Planner (FWD/BWD):** about four workgroups per CU along the rows. The workspace is the prepared filter only.
- **Catalog:**
  - A config fixes the direction (FWD/BWD), dtype, layout, filter class, tile (`kb`, `cb`), `p`, `nw` (waves per workgroup) and, for WRW, `dot2` and `ring`.
  - Every extent, channel count, group count and the height/depth geometry stay runtime.
  - The tiles are the ones that won somewhere in each pack's autotuned sweeps: ResNeXt, RegNet, HRNet, Real-ESRGAN, RetinaNet, ResNeXt3D and StyleGAN2 grouped layers, mostly measured on gfx906.
  - The FWD/BWD `valu` tiles cover each sweep graph's best config within 5%, plus one `kb = cb = 1` tile per class for channel counts no wider tile divides.
- **Two packs:**
  - `valu` targets gfx906, which has no matrix core for these types, and gfx908, whose vector unit is fast enough to matter. It ships both layouts and all tiles.
  - On gfx90a, gfx942, gfx950, gfx11 and gfx12 the matrix-core engines serve most layers. This engine won only narrow ones there (groups of 4 to 8, few-channel dense and pointwise wrw), by 1.1-2.7x in channels-first sweeps. So `mma` ships fewer tiles, in both layouts, and autotune decides where they win.
  - In both packs every FWD/BWD tile and every WRW `pass` tile ships in both layouts. The WRW `ring` form is channels-first by construction (its register ring walks the packed width of consecutive rows). The depthwise set below follows the same rule.
- **Depthwise tiles (WRW):**
  - Both packs carry 106 `cb = kb = 1` configs on every arch, 67 NCHW and 39 NHWC: one channel's filter per wave, lanes along the row.
  - Filter classes: 3x3 s1/s2, 5x5 s1/s2, 7x7; `pass` in both layouts and `ring` channels-first; `p` of at least 2, because channels-first 16-bit windows are whole dwords. NHWC uses the same tiles.
  - They compete under autotune with `CONV_DEPTHWISE_VALU_WRW`.
  - FWD/BWD have no dedicated depthwise set. Their `kb = cb = 1` fallback tiles also match depthwise (gfx906/gfx908 only), so autotune can also pick them for depthwise.

## Fusion

`CONV_GROUP_VALU_FWD` claims a forward convolution followed by a chain of pointwise nodes (the
same menu as the other conv engines: scale, bias, relu family, gelu, gelu-tanh, swish / silu,
sigmoid, tanh, elu, abs, softplus; per-channel and scalar operands) under its own engine id, with
no extra catalog config. A wave owns KB output channels, so it loads their per-channel operands
once, before its row loop, and applies the generated epilogue (`ConvEpilogue.h`) to each fp32 sum
before the single rounding at the store, in both layouts and for masked columns. The prepare
kernel is unchanged. Backward data and backward weights decline fused graphs. With no epilogue the
kernel compiles to the same code as without fusion support.

## Strengths and weaknesses
All numbers come from autotuned hipDNN runs (fresh cache) compared with MIOpen in search mode, or with our other engines where stated. They are indicative.
- **gfx906 wrw, ResNeXt (fp16, NCHW):**

  | Layer | hipDNN | MIOpen | Speedup |
  |---|---|---|---|
  | g32 c128 56² n128 | 0.36 ms | 1.98 ms | 5.5x |
  | same layer, n16 | 0.057 ms | 0.070 ms | 1.2x |
  | g32 c256 56² stride 2 | 0.66 ms | 11.7 ms | 17.6x |
  | ResNeXt-101 g32 c512 64² | 3.4 ms | 32.7 ms | 9.7x |
  | ResNeXt-101 g32 c1024 16² n128 | 1.65 ms | 1.90 ms | 1.15x |
- **gfx906 3-D grouped:**
  - On a 705-graph gfx906 set, the engine is our best on 14-16 of the 18-26 3-D grouped graphs per direction.
  - One ResNeXt3D wrw layer runs in 0.28 ms, against 17 ms on our other engines and MIOpen's 9.3 ms.
  - 3-D depthwise wrw (ir-CSN) runs 6.5-10x faster than `CONV_DEPTHWISE_VALU_WRW`.
- **Matrix-core arches:**
  - Over 125 grouped wrw graphs, our engines with this WRW run 1.18x faster than without it on MI325X, 1.23x on gfx1201 and 1.22x on gfx90a, with 7-8 fewer losses against MIOpen per arch.
  - FWD/BWD spot checks, against our engines without it: 1.02x on gfx90a, 1.06x on MI325X, 1.02x on gfx1201. These are small, as expected, since the matrix-core engines (K4, K1 `fuse`) cover most of the space.
  - Channels-last on MI325X: the `mma` NHWC configs serve every grouped NHWC graph of the real-model corpus (80 layers in fp32, fp16 and bf16: 240 graphs, all directions). Another engine is faster on all of them. The closest is `CONV_GROUP_FWD`, 1.44x faster (ResNeXt-50 c128 g32 56² n1 fp32), and the median gap to the best engine is about 13x. This engine's NHWC time is a median 3.2x its NCHW time on the same layer.
  - Depthwise wrw, channels-last, on MI325X: over the corpus's depthwise wrw NHWC graphs (142 layers at their own dtype and in fp32: 284 graphs), the `cb = kb = 1` NHWC configs serve 272. They are the best engine on 18, by 1.04-1.79x over `CONV_DEPTHWISE_VALU_WRW` (geometric mean 1.43x): ir-CSN 3-D layers in fp32 and fp16, MobileNetV3 5x5 fp32 and MobileFaceNet 3x3. The 12 declined graphs are 1x31 filters and 7x7 filters without padding, which no shipped filter class matches.
- **Weaknesses:**
  - Dense 3x3 wrw on gfx906 is about 1.3x behind MIOpen's asm.
  - HRNet c64 28² n128 wrw on gfx906 is 1.33x behind MIOpen. The hipDNN engine that serves that shape best is not recorded.
  - Large 1x1 wrw on gfx906 needs a GEMM-shaped kernel. The 1x1 tiles serve only narrow pointwise layers.
  - Rows wider than one wave are declined.

## Numerical precision
- **Accumulation:**
  - fp32 throughout. 16-bit inputs are converted and multiplied exactly, since fp16 x fp16 and bf16 x bf16 products are exact in fp32.
  - fp32 inputs round per FMA.
  - `dot2` adds two exact fp16 products to the fp32 sum per instruction.
  - There is one rounding to the output type at the store or fold.
- **Determinism:**
  - FWD/BWD: every output element belongs to exactly one lane and is summed in a fixed order.
  - WRW: lane butterfly, then the waves in order through LDS, then the splits in order in the fold. No atomics.
  - The GPU tests relaunch each config and require bitwise-identical output.
  - `numerical_notes` is empty: no nondeterminism, no reduced precision.
- **Tolerance in tests:** compared against hipDNN's CPU reference accumulated in double. The bound is a few units of fp32 rounding of the sum of term magnitudes, plus two units of the output type's rounding.
- **inf/NaN parity:** an inf at a ragged row's last column must reach exactly the outputs (FWD) or taps (WRW) that the reference reaches it with, and no others. Both families test this in both layouts.

## Comparison with MIOpen
- **`ConvAsmBwdWrW3x3`** (WRW).
  - Its limits:
    - gfx8 and gfx90x only (fp32 off on gfx90a by default; stride 1 only on gfx9).
    - 2-D NCHW, 3x3, pad 1, dilation 1, fp32/fp16.
    - fp16 needs even N.
    - Output width at most 512.
    - C or K must be a multiple of 4 x groups.
    - Size caps below 2^29 elements.
  - Every shape is an assembler constant, giving one code object per problem.
  - Its lanes split into per-input-channel parts, rotate dy rows across the parts with swizzles, and pair two images per dword for fp16.
  - The HIP form instead:
    - gives every lane the whole channel tile, so no rotation is needed;
    - pairs adjacent columns for `dot2`, so batch parity does not matter;
    - reduces across lanes with a halving butterfly rather than log2(64) DPP adds;
    - takes shapes at runtime;
    - adds bf16, NHWC, 1x1/5x5/7x7, strided and dilated height, 3-D, ragged widths, deterministic splits and large tensors.
  - Measured ratios are under Strengths.
- **FWD/BWD:** MIOpen has no VALU grouped counterpart of this design. On gfx906, grouped fwd/bwd go to its direct, Winograd or GEMM solvers. The per-shape MIOpen winner is not recorded here (unsure which solver dominates).
- **3-D:** MIOpen's asm wrw is 2-D only. 3-D grouped layers run on MIOpen's generic solvers, which this engine beats by about 33x on the ResNeXt3D layer cited above.

## Prior art considered
- MIOpen `conv3x3wrw.s` / `ConvAsmBwdWrW3x3`, studied for its schedule:
  - **Adopted:** the per-wave register tile of every tap, lanes along the width with a neighbour-lane halo, native groups, the batch split across waves with an LDS reduction, and the row ring with the next row in flight.
  - **Not adopted:** image pairing, per-problem assembly, and dy rotation by swizzle.
- Codegen practices from the other engines in this group: `readfirstlane` descriptors, branch-free predicates with out-of-range offsets, and 32-bit offsets on 64-bit bases.
- No code was copied. Everything is plain HIP compiled through HIPRTC.

## Next steps and deferred ideas
1. **Dense 3x3 wrw on gfx906** (about 1.3x behind MIOpen's asm): deeper ring pipelining (more than one row in flight). The asm kernel's `pipe_lines_depth` up to 16 is the evidence that it pays.
2. **Large 1x1 wrw on gfx906:** a GEMM-shaped VALU/`dot2` kernel. This is the largest measured gfx906 wrw excess, though it may belong in another engine.
3. **Rows wider than a wave:** split a row across passes, or let a lane own more columns. This would widen coverage for high-resolution grouped layers that are declined today.
4. **Workspace cap:** WRW's fixed 256 MiB cap does not yet consult a user workspace limit. An investigation of fixed caps across engines is open.
5. **Small depthwise fp16 wrw on gfx906:** re-measure this engine's tiles on small depthwise fp16 wrw layers, where measured timings are inconsistent.
6. **Channels-last speed:** NHWC runs about 3.2x this engine's NCHW time on MI325X. The likely cause is the load pattern: lanes lie along the row, so in NHWC each lane reads its columns one at a time, a few channels (`cb`) wide, with neighbouring lanes a whole pixel of channels apart. NCHW reads one contiguous vector per row. Staging a channels-packed tile through LDS, or wider `cb` tiles for NHWC, could close part of the gap. This is unmeasured.

## Tests
- `src/tests/engines/kernel_ingestor_engine/packs/TestConvGroupValuWrwProblem.cpp` and `TestConvGroupValuProblem.cpp`, host-only:
  - graph match (layouts, dtypes, ranks, groups, modes, compute type, extents past 32 bits, 3-D);
  - one kernel-match decline per field (dtype, layout, filter class, tile divisibility, row fits wave, `dot2`, `ring`, direction, malformed configs);
  - the planner (lanes, load widths, splits and workspace, alignment narrowing, NHWC accesses, BWD role swap, padding mirror and upsampling, large tensors with small images served, images past 32-bit offsets declined, grid limits, ring items covering each row once, 3-D tiles per depth tap).
- `TestGpuConvGroupValuWrw.cpp` and `TestGpuConvGroupValu.cpp`, on the device:
  - every shipped config for the device's arch that serves each case, fp32/fp16/bf16, against the double CPU reference;
  - cases: groups of 4 and 8 with odd batch, both layouts, ragged widths, width stride 2, 1x1 (2-D and 1-D), 5x5, 3x3x3 stride 1 and 2, true convolution, height stride/dilation/padding, multi-split fold, depthwise WRW in both layouts (every depthwise filter class, 3-D);
  - the inf-at-row-end parity cases;
  - a bitwise relaunch check per config;
  - a sampled fp16 case past 2 GiB.
