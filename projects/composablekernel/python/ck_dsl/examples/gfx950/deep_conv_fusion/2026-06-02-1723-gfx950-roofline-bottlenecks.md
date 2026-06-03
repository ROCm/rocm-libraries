# gfx950 Deep Fusion Roofline Bottlenecks

This note summarizes the current roofline and bottleneck read on the gfx950 CK
DSL deep-fusion prototype. It is an experiment record for the fixed-shape fp16
prototype, not a statement of production readiness.

## Kernel Under Test

The measured prototype is the gfx950-only CK DSL kernel:

```text
virtual concat input, represented as logical C=8
-> conv0 3x3, C=8, K0=32, fp16 x fp16 -> fp32
-> ReLU epilogue and fp16 C-shuffle LDS staging
-> conv1 1x1, K0=32, K1=24, fp16 x fp16 -> fp32
-> ReLU epilogue and fp16 C-shuffle LDS staging
-> inline 2x2 stride-2 maxpool
-> final fp16 store
```

The full target shape is:

```text
conv0 input:        N=1, H=2160, W=3840, C=8
conv0 output:       N=1, H=2160, W=3840, K0=32
conv1 output:       N=1, H=2160, W=3840, K1=24
pool output:        N=1, H=1080, W=1920, K1=24
pool tile:          4x8 pooled pixels
conv tile per CTA:  8x16 = 128 conv pixels
tile_m/tile_n/k:    128 x 32 x 16
threads per CTA:    128 = 2 waves
grid:               270 x 240 = 64,800 CTAs
```

`tile_k=16` is the current tuned default because conv0 has
`K_gemm = R * S * C = 3 * 3 * 8 = 72`. With `tile_k=16`, the hardware K loop
rounds that to 80. Larger K tiles would increase padding: `tile_k=32` rounds to
96 and `tile_k=64` rounds to 128.

## Roofline Accounting

Let:

```text
P  = 2160 * 3840 = 8,294,400 conv pixels
Po = 1080 * 1920 = 2,073,600 pooled pixels
B  = (1080 / 4) * (1920 / 8) = 64,800 CTAs
```

Useful convolution FLOPs count the logical conv work. Maxpool comparisons are
listed separately and are not included in TFLOP/s:

```text
conv0 useful FLOPs = 2 * P * K0 * R * S * C
                   = 2 * 8,294,400 * 32 * 3 * 3 * 8
                   = 38,220,595,200

conv1 useful FLOPs = 2 * P * K1 * K0
                   = 2 * 8,294,400 * 24 * 32
                   = 12,740,198,400

total useful FLOPs = 50,960,793,600 = 50.96 GFLOPs

pool comparisons   = Po * K1 * 3
                   = 2,073,600 * 24 * 3
                   = 149,299,200 comparisons
```

Hardware-padded MFMA work counts the rectangular MFMA tiles the kernel asks the
hardware to execute:

```text
conv0 hardware FLOPs per CTA =
  2 * tile_m * tile_n * ceil(72 / 16) * 16
= 2 * 128 * 32 * 80
= 655,360

conv1 hardware FLOPs per CTA =
  2 * tile_m * padded_K1 * K0
= 2 * 128 * 32 * 32
= 262,144

total hardware FLOPs =
  B * (655,360 + 262,144)
= 64,800 * 917,504
= 59,454,259,200 = 59.45 GFLOPs
```

The useful-to-hardware ratio is:

```text
useful / hardware = 50,960,793,600 / 59,454,259,200
                  = 0.8571 = 85.7%
```

Equivalently, the current tile shape executes about 16.7% more MFMA work than
the logical conv graph. The excess is dominated by conv0 K padding
(`72 -> 80`) and conv1 output-channel padding (`K1=24 -> tile_n=32`).

## Measured Full-Shape Timings

The local gfx950 full-shape measurements recorded so far are:

```text
baseline, tile_k=16:
  time                  ~= 0.387 ms
  useful throughput     ~= 50.96 GFLOPs / 0.387 ms = 131.7 TFLOP/s
  hardware throughput   ~= 59.45 GFLOPs / 0.387 ms = 153.6 TFLOP/s

baseline repeat:
  time                  ~= 0.369 ms
  useful throughput     ~= 138.1 TFLOP/s
  hardware throughput   ~= 161.1 TFLOP/s

input-footprint cache:
  time                  ~= 0.539 ms
  useful throughput     ~= 94.5 TFLOP/s
  hardware throughput   ~= 110.3 TFLOP/s

direct footprint:
  time                  ~= 0.469 ms
  useful throughput     ~= 108.7 TFLOP/s
  hardware throughput   ~= 126.8 TFLOP/s
```

The baseline number is therefore around `132 useful TFLOP/s` for the recorded
run, or `154 hardware-padded TFLOP/s` after accounting for tile padding.

## Probe Findings

The current probes point away from "missing MFMA" and toward scheduling,
operand delivery, and LDS/register overhead:

- **Intrinsic selection**: the example selects gfx950's fp16
  `32x32x16` MFMA atom for `m=32, n=32`, and the manifest/kernel naming records
  `tile.mfma_f32_32x32x16_f16` / `a32x32x16`. This is not accidentally falling
  back to scalar math or a narrower pre-gfx950 atom.

- **Occupancy shape**: each CTA has 128 threads, i.e. two 64-lane waves, and the
  full image launches 64,800 CTAs. The launch has ample grid parallelism. The
  prototype does not currently set an explicit `amdgpu-waves-per-eu` occupancy
  hint, so the backend is choosing occupancy from resource usage.

- **ISA/resource evidence still missing**: there is no saved ISA/resource dump in
  the experiment record yet. The next pass should preserve the disassembly plus
  VGPR, SGPR, LDS, scratch, and waves-per-CU metadata for baseline,
  input-cache, and direct-footprint variants.

- **Code-shape evidence**: the baseline uses the normal implicit-GEMM A/B tile
  materialization followed by MFMA. The direct-footprint path replaces the
  materialized A tile with scalar LDS fragment gathers and additional coordinate
  arithmetic in the MFMA operand path. That is consistent with the measured
  slowdown.

## Why Input Cache Regressed

The input-footprint cache loads the unique conv0 input footprint for each CTA
into LDS. For the current tile:

```text
conv tile       = 8 x 16
conv0 halo      = +2 rows x +2 cols for 3x3
footprint       = 10 x 18 x 8 = 1,440 fp16 elements
materialized A  = 128 x 80 = 10,240 fp16 element slots per CTA
```

The cache reduces repeated global input loads in principle, but the measured
kernel slows from about `0.387 ms` to about `0.539 ms`. The likely reason is
that the baseline's global input loads are already cache-friendly enough for
this shape, while the cache variant adds:

- one extra LDS allocation and fill for every CTA;
- extra synchronization before conv0 A materialization;
- scalar address math to map `(row, k)` back into the cached footprint;
- LDS reads from the footprint plus LDS writes into the regular A tile.

That extra LDS and VALU work sits directly on the critical path feeding MFMA, so
it costs more than the avoided HBM traffic saves.

## Why Direct Footprint Regressed

The direct-footprint variant goes further by avoiding the intermediate im2col A
tile and reading MFMA A fragments directly from the cached footprint. It is
correct, but the repeat timing regressed from about `0.369 ms` to about
`0.469 ms`.

The bottleneck is the current access pattern. Each MFMA A fragment performs
scalar LDS gathers with per-fragment coordinate reconstruction:

```text
row, k -> local_oh/local_ow, r/s/c -> footprint row/col -> scalar LDS load
```

That removes one materialization step, but it also removes the regular,
vectorizable LDS layout that the MFMA loop wants. The result is more VALU index
work, less regular LDS access, and worse operand delivery to MFMA. The direct
path should stay opt-in until the footprint LDS layout is organized around the
actual MFMA fragment order.

## Current Bottleneck Read

The dominant bottleneck is not proven HBM input bandwidth. If input bandwidth
were the limiter, the footprint cache should have helped. Instead, both
cache-oriented variants regressed.

The current best hypothesis is:

```text
effective performance is limited by MFMA operand delivery,
LDS staging/gather overhead, synchronization, and scalar coordinate arithmetic,
not by the amount of useful conv math alone.
```

The 85.7% useful/hardware ratio also matters: even perfect scheduling of the
current tile cannot report hardware throughput as useful throughput without
paying the `K1=24 -> 32` and `K_gemm=72 -> 80` padding tax.

## Next Measurements

Recommended next measurements, in priority order:

1. Save ISA/resource reports for baseline, input-cache, and direct-footprint:
   VGPRs, SGPRs, LDS bytes, scratch, waves per CU, MFMA count, DS read/write
   count, and code size.
2. Run counter-based profiling for baseline vs cache variants: MFMA active,
   VALU active, LDS bank conflicts, LDS latency/stalls, VMEM bytes, L2 hit rate,
   and achieved occupancy.
3. Sweep padding-sensitive geometry: `tile_n=24` if the DSL/MFMA mapping can
   support it, alternate `pool_tile_h x pool_tile_w`, and `tile_k=8/16/32` to
   separate K-padding from loop overhead.
4. Sweep pipeline variants on the baseline path only: `mem`, `compv3`,
   `compv4`, and async DMA where valid. The cache paths currently disable async
   DMA, so compare them separately.
5. Add stage-isolation toggles or microbenchmarks for conv0 only, conv0+conv1,
   and conv0+conv1+pool to attribute time to C-shuffle staging, W1 loading,
   conv1 MFMA, and maxpool/store.
6. Revisit direct footprint only after designing an LDS footprint layout whose
   rows/columns match MFMA A fragment consumption. Then compare scalar gathers
   against vectorized LDS reads.
7. Compare against the unfused multi-kernel pipeline with the same dtype and
   shape to quantify the actual end-to-end fusion win, not just single-kernel
   roofline efficiency.
