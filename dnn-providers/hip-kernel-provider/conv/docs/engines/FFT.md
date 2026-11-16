# CONV_FFT — tiled overlap-save FFT convolution for large filters

> Engine ids: `hipkernel:CONV_FFT_FWD`, `hipkernel:CONV_FFT_BWD`, `hipkernel:CONV_FFT_WRW` · Generator: `conv/generators/gen_fft.py` · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/fft/kernels/` · Pack: `packs/ConvFftProblem.cpp`, `packs/ConvFftNative.cpp`

## At a glance
- **What:** overlap-save tiled FFT convolution. Real 2-D (or 1-D row) transforms are done in fp32 registers and LDS. A per-bin complex GEMM runs over channels, then the inverse transform. Written fresh in HIP.
- **Covers:** fwd, bwd-data and wrw · fp32/fp16/bf16 tensors, compute type unset or fp32 · any element strides (NCHW, NHWC, non-packed) · 1-D and 2-D · any groups, with a fused path for depthwise · any padding and dilation, both conv modes · unit stride only · every extent within 32 bits.
- **Arches:** all of them. The engine uses VALU FMA only (no matrix cores) and has no arch gate. It runs through runtime compilation and is tested on gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1101 and gfx1201.
- **Wins when:** fp32 dense filters of 5x5 and up, and 1-D filters of 7 taps and up. Also 16-bit dense filters from about 11x11, and large-kernel depthwise (13x13 to 31x31, the RepLKNet/SLaK class), where wrw gains the most.
- **Loses when:** 3x3 and other small filters (Winograd and MMA direct engines win), and 16-bit dense filters below about 11x11. An admission floor keeps it off those graphs.
- **Precision:** fp32 transforms and products, rounded once to the output type. An output's error scales with the transform's magnitude, not with the output's own size. Reports numerical note `FFT`. Deterministic: no atomics, and a relaunch test checks bitwise repeat.
- **Instances:** one generic tier, where everything is runtime except the transform size. Fwd and bwd each have 198 configs, wrw has 132.
- **Workspace:** frequency-domain spectra, capped at 1 GiB. Tiles are processed in chunks below the cap.
- **vs MIOpen:** MIOpen's `fft` solver is a fixed-size fp32 NCHW 5x5 special case. CONV_FFT beats it on 27x27 and 28x28 images and loses on 14x14 and 7x7. Against MIOpen's Find winner on gfx950, CONV_FFT is 2-17x faster in fp32 and 1.1-3.3x faster in fp16 from 11x11.

## Why it exists
- MIOpen's FFT solver bakes the image size, batch, channel and filter counts into hand-written code. It covers 5x5 filters, pad 2, fp32 NCHW on four image sizes, and no corpus graph meets those conditions. hipDNN uses it only as a test reference and does not port it.
- No other engine in the catalog gets cheaper as the filter grows. Without this engine, large-kernel depthwise wrw on our other engines takes 55-440 ms, where MIOpen takes 10-35 ms, and dense fp32 filters of 7x7 and up run at the FMA rate.
- Arithmetic advantage: direct convolution costs r·s MACs per output and channel. The frequency-domain product costs 4·F/(oh·ow), where F = th·(tw/2+1) bins. The ratio is 4.5 at most for 3x3 on any tile, 25 for 11x11 on 32x32, and 116 for 31x31 on 64x64.
- Corpus value is small. Under 1% of usage-weighted corpus work lies above the floor. The engine's value lies outside the corpus: large-kernel CNNs, long 1-D filters, and any fp32 filter of 7x7 and up.

## How it works
Overlap-save: an output tile of oh x ow points equals the circular cross-correlation of a th x tw input window with the zero-padded, dilated filter. That result is valid on the first oh x ow points, where oh = th − spanH + 1 and span = (r−1)·dil + 1. Tiles step by (oh, ow) over the output, so any image size, pre/post padding and dilation work.

```
fwd/bwd:  W_f   = conj(FFT(pad(w)))                          once, per (g, a, b)
          per chunk of tiles:
            X_f = FFT(window)                                per (tile, input channel)
            Y_f[g][bin][b][tile] = Σ_a X_f[g][bin][a][tile] · W_f[g][bin][a][b]   complex GEMM per (g, bin)
            y   = crop(IFFT(Y_f))                            per (tile, output channel)
wrw:      per chunk: X_f, DY_f(zero-padded dy tile); DW_f[g][bin][k][c] (+)= Σ_tile conj(DY_f)·X_f
          dw = taps of IFFT(DW_f) at the dilated lags
```

- **bwd-data** correlates dy with the reflected filter, using padding span − 1 − pad, with the channel roles swapped.
- **Conv mode** reflects the filter in fwd and wrw, and un-reflects it in bwd.
- **Depthwise** has one input and one output channel per group, so there is no product to amortize the transforms over. fwd and bwd run a filter transform, then one fused transform-multiply-inverse launch per `pb` tiles of a channel. wrw keeps per-partition sums of conj(DY)·X in registers, folds the partitions in order, then inverts.

## Implementation design
- **Transforms:** a real tile becomes th x (tw/2+1) bins. Two real rows at a time go through one complex register FFT of compile-time length (radix 2, fully unrolled, trivial twiddles special-cased). Hermitian symmetry splits the result back into two rows. Columns then run as complex register FFTs through LDS. 1-D tiles pair two tiles instead of two rows. Twiddles come from a 512-entry table, rounded once from double.
- **Transform blocks:** a block transforms `pb` tiles (or `pb` filters) of one channel with `tb` threads. It stages the real tiles in LDS, with rows padded to tw + 1 floats, using coalesced cooperative loads. Tile geometry is computed once per block. Spectra are written bin-major, with the block's `pb` tiles contiguous, which is the operand layout the product expects.
- **Product (fwd/bwd):** a complex GEMM per (group, bin). Each block has 256 threads and each thread computes 4 output channels x `tm` tiles. Block tiles are 64x64 or 128x64 (`gn` 64, `tm` 4 or 8) or 256x16 (`gn` 16). The next 16 (or 8) channels are prefetched into registers during the current step. **Product (wrw):** contracts over tiles, with `4096/gn` input channels by `gn` dy channels per block.
- **Launches:** fwd and bwd run filter, data, product and output launches, chunked over tiles. wrw runs data(x), data(dy) and product per chunk; chunk 0 writes and later chunks add, in stream order, followed by an inverse launch. Depthwise runs filter + fused, or partial-sum + fold + inverse. All grids are one-dimensional.
- **Host gates** (`parseFftProblem`, `configOf`, `planFor`; each has its own decline test):
  - Graph: single conv node; compute type unset or fp32; rank 1 or 2 (3-D declined); unit stride; n, c, k, groups, extents, filter, padding, dilation and dilated span all within 32 bits.
  - Kernel: dtype and direction per config.
  - Fit: span ≤ tile on both axes; a 1-D tile needs a one-row filter.
  - Admission floor, below.
  - Workspace: filter (or dw) spectra ≤ half the cap; a chunk must hold at least min(tiles, 64) tiles; depthwise wrw LDS within 64 KiB; grids ≤ 2^31−1 blocks and < 2^32 work-items.
  - Config: th ∈ {1, 8..64 pow2}, tw ∈ 8..64 pow2, pb·bins·8 B ≤ 64 KiB, tb a multiple of 64 up to 256 with at least one thread per row pair, gn ∈ {16, 32, 64}, tm 4 (or 8 with gn 64). Depthwise takes only gn 64 / tm 4.
- **Admission floor** (`fftAdvantage`, calibrated on gfx950 by benching every config of every engine). Direct MACs divided by product MACs must reach 4 for fp32, 16 for fp16/bf16 (the direct engines run 16-bit matrix cores), and 1 for depthwise, where the transforms are counted (2.5·n·log2 n each). The problem also needs at least 2^26 MACs. `HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR=0` disables the floor, for tests and calibration.
- **Workspace:** fixed spectra (g·F·ca·cb complex fp32) plus per-chunk data and product spectra, 256-byte aligned, all under `WORKSPACE_CAP_BYTES` = 1 GiB. Tests lower the cap with `HIP_KERNEL_PROVIDER_CONV_FFT_WORKSPACE_CAP` to force several chunks.
- **Catalog:** 22 transform configs: 16x16 (pb 8, 16), 32x32 (pb 2, 4, 8), 64x64 (pb 1, 2), 1x32 (pb 64, 128) and 1x64 (pb 32, 64), each at the least `tb` that covers the row pairs and at 256. These are crossed with 3 product tiles and 3 dtypes, giving 198 fwd and 198 bwd configs. wrw's product has no `tm`, so it has 132 configs. The only compile-time geometry is the transform size, so there is no per-shape specialized tier to add.
- **Codegen choices:** element offsets are 64-bit (`long long` strides) while per-tile index math is 32-bit. Filter taps are scattered tap by tap (r·s writes, not th·tw), and dw is gathered the same way. LDS is limited to 64 KiB on every target, so one module compiles everywhere and the depthwise wrw spectra fit 64 KiB parts.

## Strengths and weaknesses
Engine bench: every admitted config of every hip-kernel-provider engine, autotune-equivalent, with the FFT floor off and NCHW layout. Ratios are baseline time / CONV_FFT time, and above 1 CONV_FFT is faster. "Ours" means the best non-FFT hip-kernel-provider engine; BLAS engines are not included.

| class (gfx950) | fp32 fwd/bwd | fp32 wrw | fp16 fwd/bwd | fp16 wrw |
|---|---|---|---|---|
| dense 5x5 (c128, 64x64) | 1.8x | 1.4x | 0.25x | 0.47x |
| dense 7x7 | 2.3-4.1x | 2.2-6.4x | 0.29-0.45x | 1.3-2.9x |
| dense 11x11 | 5.8-8.4x | 8.7-14.2x | 0.92-1.19x | 3.1-4.2x |
| dense 13x13-15x15 | 6.5-9.7x | 10.6-16.1x | 1.15-1.57x | 3.6-4.8x |
| dense 31x31 | 15.0x | 26.7x | 2.2x | 7.1x |
| 1-D dense k7 / k15 / k31 (c512) | 1.3 / 2.3 / 3.2x | 1.3 / 3.9 / 5.4x | 0.22-0.47x | 0.23-1.36x |
| depthwise 13x13 (c256, 56x56) | 2.0-2.1x | 24x | 2.2-2.3x | 27x |
| depthwise 31x31 | 3.9-8.1x | 55-65x | 4.5-8.4x | 57-63x |

- **gfx1201, gfx942 (MI300X) and gfx90a** follow the same pattern. fp32 dense from 11x11 runs 3-13x faster than ours. Depthwise wrw runs 23-94x faster. 16-bit dense filters below about 11x11 lose (0.4-0.9x).
- **gfx90a** gains less: 2.9-4.9x on dense 11x11-15x15 fp32. On 64 KiB-LDS CDNA, the transforms run below the bandwidth roofline.
- **gfx1101** results come from a partial bench and match gfx1201.
- **Weak spots:**
  - **Small images.** Power-of-two tiles waste bins: a 7x7 image in a 16x16 tile uses about half of them, and a 14x14 image needs 4 tiles.
  - **Short 1-D filters in 16-bit.** k7 and k11 run at 0.2-0.5x.
  - **64x64 tiles** are limited by VGPRs at 256 threads.
  - **MI325X perf** has not been measured. The gfx942 numbers above come from an MI300X and are indicative only.

## Fusion

`CONV_FFT_FWD` claims a forward convolution followed by a chain of pointwise nodes (scale, bias,
relu / clipped relu / clamp / leaky relu, gelu, gelu-tanh, swish / silu, sigmoid, tanh, elu, abs,
softplus; per-channel, one-element and pass-by-value operands) under its own engine id. The
pass that writes y applies the chain to each point's fp32 value (the inverse transform's output
times its scale) before the single rounding: `ConvFftOutput`, and `ConvFftDepthwise` for the
depthwise path. A block works on one output channel, so the chain's per-channel operands load
once a block, with no per-element channel arithmetic. The transforms and the product compile and
run as unfused, and an unfused graph's kernels are unchanged. The chain is compiled into the
output pass as generated code that depends on the pattern only; operand pointers and values are
a second kernel argument. Backward data and backward weights decline fused graphs.

## Numerical precision
- **Arithmetic:** transforms and products run in fp32 for every tensor type. Inputs are converted once on load and outputs are rounded once on store. There is no reduced-precision product, so no TF32 or RPR sibling exists.
- **Error model:** FFT error is absolute, about c·log2(n)·ε times the transform's rms. It is not relative to each output. An output much smaller than its neighbours therefore carries a large relative error, which is why the engine reports the `FFT` note.
- **Measured:** an fp32 model of the same transform math, checked against an fp64 CPU reference on gfx950 for 16x16 to 64x64 and 1x64 tiles in all three directions: max |err| / rms(ref) is 15.6-22.8 units of fp32 rounding. Every case passes the #15481 model with c = 4; the largest c needed is 0.78. MIOpen's own FFT, on its own shapes, needs c = 0.12-0.60 with 12-42 units.
- **Gtest bound:** 8·log2(th·tw) u32 of the largest reference magnitude, plus 2 storage ε relative. The #15481 atol term is relative to σ, which is the right shape for this error. An rtol-only check would flag near-zero outputs.
- **Determinism:** no atomics. Every sum has one owner and a fixed order: chunks in stream order, and depthwise partitions folded in order. The device suite relaunches each config over a re-filled workspace and compares bits.
- **inf/NaN:** a non-finite input enters every bin of its tile's transform, so every output of every tile that reads it becomes non-finite. A direct engine affects only the outputs whose windows touch that input. This divergence is inherent to FFT and has no dedicated test.

## Comparison with MIOpen
- **MIOpen `fft` solver:** transforms the whole padded image with no tiling, at 12x12, 18x18 or 32x32 (radix 2/3). It runs seven launches with a per-bin cgemm. Image size, batch and channel counts are compile-time constants. It handles fp32 NCHW only, 5x5 / pad 2 / stride 1, with no groups or dilation, and fwd and bwd only.
- **On MIOpen's own shapes** (GoogLeNet 5x5 branches and AlexNet conv2, fp32, N 16-128, gfx950), CONV_FFT against MIOpen's FFT is:
  - 1.06-1.56x on AlexNet conv2 (27x27) and 0.91-1.14x on GoogLeNet 3b (28x28);
  - 1.06-1.21x against MIOpen's Find winner on those images;
  - 0.60-1.00x on the 14x14 and 7x7 layers.
- **On gfx90a**, MIOpen's hand-scheduled FFT also leads on the 27x27 and 28x28 images (per-layer geomean 0.79-1.02x).
- **Against MIOpen's Find best** (normal mode with perf-db configs and no tuning search, gfx950), CONV_FFT's speedups are:

  | filter | fp32 fwd/bwd | fp32 wrw | fp16 fwd/bwd | fp16 wrw |
  |---|---|---|---|---|
  | 5x5–7x7 | 2.0-5.0x | 1.6-4.7x | 0.39-0.63x | 0.36-0.97x |
  | 11x11–31x31 | 6.1-17.1x | 5.0-14.5x | 1.1-3.3x | 1.2-2.6x |
  | depthwise 13x13–31x31 | 7.4-21.5x | 6.1-12.7x | 7.0-18.9x | 3.8-6.8x |

- **gfx1201:** CONV_FFT runs 1.4-6.7x faster than MIOpen's Find winner on fp32 dense, and 8-29x faster on depthwise fwd/bwd.

## Prior art considered
- MIOpen's `fft` solver, used as a test reference and as an example of what not to bake.
- cuDNN `FFT` / `FFT_TILING`: 32x32 tiles, stride 1, and the `FFT` numerical note.
- Mathieu, Henaff & LeCun 2013: transform once and reuse across filters.
- fbfft (Vasilache et al. 2014): custom batched small FFTs, with tiling as the way to scale.
- Lavin & Gray 2015: the crossover against Winograd sits around 7-9 taps.
- Highlander & Rodriguez 2016: overlap tiles a few times the filter size.
- Zlateski et al. 2019: roofline view, in which transforms are the bandwidth term.
- FlashFFTConv 2023: four-step FFT on matrix units, for long 1-D filters.
- Only techniques were taken; no FFT library code was used.

## Next steps and deferred ideas
1. **Large-kernel depthwise occupancy.** A 64-point transform split across 2-4 lanes per row would lift 64x64 tiles off the VGPR limit. This regime has the best value-to-cost ratio, with the largest measured gaps.
2. **1-D tiles beyond 64**, using a four-step FFT with 256-1024 points. This would serve wav2vec2/HuBERT positional convs (k128, g16) and SincNet-style front ends, which are declined today.
3. **Radix-3 tiles (12, 18, 24)** to win the 14x14 and 7x7 images where MIOpen's FFT still leads.
4. **Transforms on 64 KiB-LDS CDNA (gfx90a/gfx942)**, which run below the bandwidth roofline.
5. **Opt-in TF32 / three-bf16 product** as a tagged sibling engine. This would lift the 16-bit floor off 16x, since the product is the fp32 bottleneck.
6. **3-D and strided convs.** Strided convs would need a phase decomposition; cuDNN also limits FFT to stride 1. Neither appears in current demand.
7. **MI325X perf run** for the gfx942 rows.

## Tests
- **`TestConvFftProblem` (host):**
  - Accepts per direction, layout, dtype, rank, groups, dilation, mode and compute type.
  - One decline per gated field: 3-D, stride, compute type, extents, dtype, direction, fit, 1-D tile against filter height, the floor and its switch, and each malformed config field (th, tw, pb, LDS, tb, gn, tm).
  - Plans: tiles per direction, flip mapping, chunking under the cap, filter spectra over the cap, and depthwise fusion, partitions and floor.
  - Each gate's test fails when that gate is removed (mutation-checked).
- **`TestGpuConvFft` (device):** 17 shape families x fwd/bwd/wrw x fp32/fp16/bf16, run against the CPU double reference. fp32 covers every fitting catalog config.
  - The families include MIOpen's three 5x5 shapes, odd extents in NCHW and NHWC, an asymmetric dilated grouped shape with post-padding, 31x31, 1-D rows (dilated and grouped), 1 x s on 2-D, conv mode, depthwise, and forced multi-chunk.
  - Every config gets a bitwise relaunch check.
  - `TestConvFftCatalog` checks that every shipped config parses and names its direction.
