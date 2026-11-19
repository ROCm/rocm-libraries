<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_NARROW_FWD / BWD / WRW — convolutions with at most 8 channels on one side

> Engine ids: `hipkernel:CONV_NARROW_FWD`, `hipkernel:CONV_NARROW_BWD`, `hipkernel:CONV_NARROW_WRW` · Generator: `conv/generators/gen_narrow.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/narrow/kernels/` (`ConvNarrow.cpp`, `ConvNarrowWrw.cpp`, `ConvNarrowArgs.h`) · Pack: `packs/ConvNarrowPlan.cpp`, `packs/ConvNarrowNative.cpp`, `packs/ConvNarrowWrwNative.cpp`

## At a glance

- **What:** the taps go on the GEMM's N axis. Each pixel of the wide operand is multiplied once by every (narrow channel, tap) column of the filter, and each output element gathers its taps' products. The wide tensor is read once, with a band and strip halo, so the kernels run at the bandwidth roofline's shape instead of a GEMM tile that is mostly padding.
- **Covers:**
  - **Forward:** network output layers, C wide into K <= 8. Examples are VAE and super-resolution RGB heads (3x3, 3x3x3), 1x1 heads and 1x7 vocoder outputs.
  - **Backward data:** stem input gradients, K wide into C <= 8 (7x7 s2, 3x7x7 s(1,2,2), 3x3 s2, 3x3 and 3x3x3 s1, 1x1). The 7x7 and 3x7x7 stems need 160 KiB of LDS, so only gfx950 serves them.
  - **Backward weights:** both sides. Output layers take 3x3x3, 3x3, 1x1(x1) and 1x7 filters; stems take 7x7, 3x7x7, 3x3, 4x4 and 3x3x3.
  - **Common to all three:** fp32, fp16 and bf16; NCHW and NHWC read in place; 1-D, 2-D and 3-D; groups 1; any padding and dilation. Every extent, channel count and stride is a runtime argument, except the width stride for forward and backward data, which each config fixes. There is no TF32 form.
- **Arches:**
  - gfx950: MFMA 16x16x32 for 16-bit data, and 16x16x4 for fp32, with tiles sized to its 160 KiB of LDS.
  - gfx90a and gfx942: MFMA 16x16.
  - gfx908: 16x16 for fp16 and fp32, and its own half-rate 16x16x8 for bf16.
  - gfx1100-gfx1151 and gfx1200 / gfx1201: WMMA.
  - Every architecture, gfx906 and gfx1250 included: a vector-ALU form. It is the only fp32 path on RDNA.
- **Wins when:**
  - Forward output layers. VAE and super-resolution RGB heads gain 1.9-2.2x on gfx950 and gfx942, 4.6-5.5x on gfx908 and 2.0x on gfx1101. Wan 3x3x3 heads gain 1.5-2.2x on gfx950 and 4.8-5.2x on gfx908.
  - Backward weights of output layers on gfx908 (5.4-8.5x) and gfx1101 (1.5-2.1x), and of single-channel 3-D stems (1.1-1.25x).
- **Loses when:**
  - Backward data of 2-D RGB stems, against `CONV_STEM_BWD`. It wins only 3x3 stride-1 fp32 and fp16 stems, by 4-16%.
  - Backward weights of output layers on gfx950 and gfx942, against `CONV_WRW`. Autotune keeps `CONV_WRW` there.
  - Wan 3x3x3 heads on gfx942 gain only 1.1-1.3x.
- **Precision:** fp32 accumulation. Forward and backward data sum every output element in one thread, in a fixed order, and round once. Backward weights add fp32 partials in split order. Deterministic, no atomics. No numerical notes.
- **Instances:** 8758 configs. Forward has 2572 and backward data 2848, in 14 catalog files per direction. Backward weights has 3338.

## Why it exists

An output layer such as a VAE's c128 → RGB 3x3 head is memory-bound. It reads 128 channels per pixel and writes 3. As an implicit GEMM (M pixels, N = K = 3, reduction C x taps), the N axis fills 3 of 16 matrix-core columns. The explicit-GEMM engines also write an im2col buffer of 9 x 128 channels per pixel. Both run at a small fraction of the bandwidth roofline. Before this engine, the output-layer forward graphs of the A/B below ran at 18% of attainable on gfx950, 19% on gfx942 and 7% on gfx908.

The transposed form is a GEMM whose N is narrow channels x taps (27 columns for a 3x3 RGB head, two 16-wide tiles at 84% use). Its reduction is the wide channel axis, and each input pixel is a single GEMM row. The output is a gather of products, with no im2col.

## How it works

Forward and backward data (`ConvNarrow.cpp`) compute, for in pixel p (an x pixel forward, a dy pixel backward), narrow channel j and tap (t, r, s):

```
P[p][j, t, r, s] = sum_c in[p][c] w'[c][j, t, r, s]
forward:        y[o][j]  = sum_{t,r,s} P[o * stride - pad + (t, r, s) * dilation][j, t, r, s]
backward data:  dx[i][j] = sum_{t,r,s} P[(i + pad - (t, r, s) * dilation) / stride][j, t, r, s]
```

The backward term counts only where the division is exact and the dy pixel exists.

A block owns three things: a segment of output planes, a band of output rows, and a strip of output columns of one image. It streams the in rows its band reads, `rb` rows of `wr` pixels per step:

- The matrix-core lanes load their pixels' channel runs from memory into registers. They run `pd` steps ahead, with `pd + 1` register buffers.
- The step's products go through LDS: M = `rb x wr` rows by NCP columns, in fp32, with two buffers and one barrier per step.
- Each thread then adds the products of its `op` output pixels.

The filter is prepared once by a first launch, as the products' B operand, in the element type, zero past the channels. The main kernel holds it in registers when one column group's fragments fit in 64 VGPRs, and in LDS otherwise.

3-D filters have two modes:

- `ds` 1: depth streaming, for unit depth stride and dilation. Every in plane is read once with the products of every depth tap, and is added into a ring of `fd` output planes.
- `ds` 0: one pass per depth tap, which takes any depth stride.

Backward weights (`ConvNarrowWrw.cpp`) form a GEMM with these axes:

- **M:** the wide channels.
- **N:** narrow channels x taps.
- **Reduction:** the wide operand's pixels.

The side decides which operand is which:

- **Output layer:** the wide operand is x, the narrow operand is dy, and the narrow window is mapped as in backward data.
- **Stem:** the wide operand is dy, the narrow operand is x, and the window is mapped as in forward.

Each step proceeds as follows:

1. A block stages the narrow window its taps reach, one step ahead in registers.
2. It expands the window into the B operand in LDS.
3. It multiplies against its lanes' wide-operand pixel runs.

The steps (images x planes x bands x strips) split across blocks into fp32 partials `[split][wide channel][column]`. `ConvNarrowWrwReduce` adds them in split order and writes dw in its layout.

## Implementation design

- **Launches:**
  - Forward and backward data take two launches: `ConvNarrowPrepare` writes the filter in `[group][NCP][cpad]` (or `[group][ci][NCP]` fp32 for the vector ALU), and `ConvNarrow` runs the convolution.
  - Backward weights takes two launches: `ConvNarrowWrw` writes the partials, and `ConvNarrowWrwReduce` sums them.
  - Workspace holds the prepared filter or the partials (within 256 MiB).
- **Config knobs, forward and backward data:**
  - Narrow channels: `kmin` to `ko`.
  - Filter `fd` x `fr` x `fs`, and width stride `sw`.
  - Depth mode `ds`.
  - Tile: `wr`, `rb`, `nw` waves, `op`.
  - Matrix cores only: the reduction-channel tier `cmin..cm` (32, 64, 128, 320, 512 channels; `cm` is the compile-time k-loop extent) and the prefetch depth `pd`.
- **Config knobs, backward weights:** the side, `kmin..ko`, the filter, `wr`, `rb`, `nw`, and `mpw`. `mpw` is row tiles per wave on matrix cores and column groups on the vector ALU.
- **Exactly one config per tier:** the channel ranges and tiers mean exactly one config per (tile, atom) admits a graph.
- **Planning (`planFor`, `planWrw`):**
  - Column strips are as wide as the product row covers with every tap.
  - Bands and segments halve until the grid has four blocks per dispatch unit. Whichever axis's halo grows less is halved.
  - LDS within the device: 160 KiB on gfx950, 64 KiB elsewhere.
- **Gates (`parseNarrowProblem`, `parseNarrowWrwProblem`, every field):**
  - Everything the naive parser gates, plus fp32, fp16 and bf16, and groups 1.
  - Narrow side 1-8 and narrower than the wide side.
  - Compute type fp32 or unset.
  - Pointers aligned to the element.
  - Extents and channels within 32 bits.
  - One image of each tensor under 2^31 bytes (images index 32-bit; image bases are 64-bit).
  - Per config:
    - the atom on the device and taking the dtype;
    - the device in the config's `archs`;
    - filter and width stride equal to the graph's;
    - the channel tier;
    - streaming only at unit depth stride;
    - LDS and the grid within limits.
- **Addressing:** buffer loads with 32-bit offsets within an image, and out-of-range reads return zero. Vector loads apply where a pixel's channels are packed and the channel count is whole 16-byte runs. The loads are unconditional, and out-of-range lanes read the hardware's zero. Conditional loads made the compiler drain the whole load queue each step.
- **Catalog (`gen_narrow.py`):**
  - Tiles per atom family are taken from a sweep over output-layer, stem and 3-D shapes on gfx950, gfx908 and gfx1101.
  - MFMA: `64x1` with prefetch 3, plus the streaming `32x2`; on 3-D, also `64x2` over 8 waves.
  - WMMA: the streaming `32x2` and `64x2` over 8 waves; on 3-D, also `64x1`.
  - The vector ALU: `32x4`, `64x2` and `32x2` (whose 64 product rows carry 8-channel heads within 64 KiB); on 3-D, also the streaming `32x2`.
  - Configs whose A buffers would exceed 128 VGPRs, or whose LDS exceeds the ship group's, are not generated.
  - Each config is admitted on correctness alone, and autotune picks.

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines and the BLAS engines, 20 iterations, two interleaved rounds, best of the rounds per graph. The set is 90 edge-channel graphs (C or K <= 8; output layers, stems and 3-D medical and video layers from an internal real-model corpus) in all three directions. Speedup = before / after, time-weighted. Efficiency is the time-weighted fraction of the attainable (roofline) time. gfx942 is MI325X. gfx1101 has no efficiency figures: the V710's bandwidth is not published.

| GPU | all 90: eff before -> after | all 90: speedup | output-layer fwd (11): eff | output-layer fwd: speedup | output-layer wrw (16): speedup | stem bwd (18): speedup |
|---|---|---|---|---|---|---|
| gfx950 | 22% -> 25% | 1.15x | 18% -> 36% | 1.98x | 1.00x | 1.01x |
| gfx942 | 20% -> 21% | 1.07x | 19% -> 27% | 1.37x | 1.00x | 1.00x |
| gfx908 | 10% -> 23% | 2.42x | 7% -> 33% | 4.85x | 5.52x | 1.09x |
| gfx1101 | - | 1.03x | - | 1.38x | 1.13x | 0.99x |

- **Where the engine is the autotuned best:**
  - Forward output layers on 10 of 11 graphs (7 on gfx1101).
  - Backward data on 3-5 of 18 stem graphs on the CDNA GPUs (1 on gfx1101): 3x3 stride-1 RGB stems and single-channel 3-D stems.
  - Backward weights on 7 of 16 output-layer graphs on gfx908 and 4 on gfx1101, none on gfx950. On gfx942, 1 output-layer graph and 2 single-channel 3-D stems.
- **gfx908:** the backward-weights and backward-data gains include the gfx908 bf16 atoms of `CONV_WRW` and `CONV_SMALLC`, which serve those bf16 graphs there for the first time (Wan and VAE heads 1.4-3.6x).
- **Control set:** 60 graphs no narrow config serves gave a geomean of 1.000-1.003 on every GPU, with no repeatable regression on the CDNA GPUs. On gfx1101, same-engine graphs drift by up to 10% between rounds on that node. A four-round interleaved recheck of the three graphs it flagged found no repeatable regression. Per-round ratios were 0.99-1.12 on the two `CONV_WRW` graphs and 0.89-1.01 on the rocBLAS control, whose engine is unchanged.

## Numerical precision

- **Forward and backward data:** fp32 accumulation. 16-bit products are exact on the 16-bit atoms. One thread sums every output element in a fixed order, and it is rounded once. The device suite relaunches each config and compares bits.
- **Padding:** products of pixels outside the input are never gathered. An infinite or NaN weight therefore meets padding as an exact reference does, which is not at all. An infinite input value meets only the filter's own entries.
- **Backward weights:** infinities and NaN give the result of an explicitly padded GEMM: im2col with x's padding as zeros, as PyTorch, the BLAS engines and `CONV_WRW` compute it.
  - **Stems:** an infinite dy value meets x's padding as inf x 0, which gives NaN.
  - **Output layers:** an infinite or NaN x pixel meets only the taps that reach a dy pixel. A tap that reaches none (beyond dy, or a stride phase between dy pixels) makes no product.
  - **Implementation:** the matrix cores multiply every column, so a step whose A fragments hold a non-finite value runs them with those values zeroed. Each lane then adds the non-finite values' products on the taps that reach a dy pixel. A wave-wide vote on the loaded fragments decides this, so finite data pays only the vote.
  - **Tests:** the device suite plants infinities and NaN in x's border pixels (padding, and a stride-2 column and row no output reads), and in a stem's interior dy.
- **Folded filter columns (`sf` 1):** this kernel option, not in the catalog, multiplies the padding columns of the window by the filter, with the same consequence for infinite inputs.

## Prior art considered

- The "taps on N" transposition is the standard way to turn a narrow-output correlation into a GEMM with a wide reduction, as in the col2im formulation of transposed convolution run forward.
- K1's atoms (`ConvIgemmCore.h`) and buffer loads are reused.
- No code was copied.

## Next steps and deferred ideas

1. **Bandwidth gap on 3-D heads:** the Wan VAE c96 → RGB 3x3x3 head on gfx950 is load-throughput-bound at about 2.4 TB/s. The `ds` 0 passes re-read each plane per depth tap, and streaming is bound by MFMA and LDS occupancy. A depth-tap-folded reduction, the `sf` analogue on depth, would read each plane once at the `ds` 0 occupancy.
2. **Backward data of output layers** (narrow dy into wide dx) is SMALLC_BWD's (K <= 4, stride 1). Its efficiency on NCHW 16-bit heads is the next item for this graph class.
3. **Re-sweep tiles** on gfx942, gfx90a, gfx12 and gfx906. The catalog's tiles come from gfx950, gfx908 and gfx1101; the Wan 3x3x3 head on gfx942 runs at 22% of attainable (31% on gfx950).
4. **gfx1250 matrix cores:** gfx1250 runs the vector-ALU configs only. Its K32 WMMA atoms (`wmma125_*`) need their own lane mapping in the A and B loads.

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestConvNarrowProblem.cpp` (host):
  - acceptance of output layers and stem gradients, and one refusal per gated graph field;
  - the matching config admits, and each config mismatch and malformed field declines;
  - strip, band, segment and prepared-filter arithmetic, the LDS formula with registers or the LDS filter, and vector-load rules;
  - images past 2 GiB decline;
  - backward weights: both sides accept, the rest decline, each mismatch declines, and the split, channel-block and partial arithmetic.
- `TestGpuConvNarrow.cpp` (device):
  - every admitted config against a reference, in three dtypes and both layouts, across these shapes:
    - output-layer 3x3 and 3x3x3 heads (streaming and per-tap);
    - 1x1, 1x7, stem 7x7 s2, 3x3 s1 and 3x7x7;
    - odd extents, padding past the filter, dilation, strides, and true convolution;
    - backward weights on both sides;
  - planted inf and NaN;
  - a channels-last tensor past 2 GiB, served by 64-bit image bases;
  - bitwise relaunch;
  - every catalog config parses, and the knobs and ship list distinguish every config.
