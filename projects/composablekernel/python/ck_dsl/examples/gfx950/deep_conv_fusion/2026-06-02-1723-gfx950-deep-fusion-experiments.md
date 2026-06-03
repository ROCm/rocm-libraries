# gfx950 Deep Fusion Experiments

Status snapshot from 2026-06-02 17:23.

## Target Shape

The motivating model block is:

```text
Input0:  (1, 4, 2160, 3840)
Input1:  (1, 4, 2160, 3840)
Concat:  channel concat -> (1, 8, 2160, 3840)
Conv0:   3x3, C=8 -> K0=32, same spatial size
Conv1:   1x1, K0=32 -> K1=24, same spatial size
Pool:    2x2 stride 2 maxpool -> (1, 24, 1080, 1920)
```

CK DSL represents the prototype in NHWC/NHWK order:

```text
Input:       [N=1, H=2160, W=3840, C=8]
Conv0 out:   [1, 2160, 3840, 32]
Conv1 out:   [1, 2160, 3840, 24]
Pool output: [1, 1080, 1920, 24]
```

The original graph is int8/int4. The current gfx950 prototype intentionally uses
fp16 inputs and weights with fp32 MFMA accumulation so the experiment stays
focused on tile ownership, on-chip producer/consumer flow, and inline pooling.

## Implemented Fused Graph

The checked-in prototype is `instances/gfx950/deep_fused_conv_pool.py`, with the
verification/emission harness in `examples/gfx950/deep_conv_fusion/deep_fused_conv_pool_verify.py`.
It is gfx950-only and shape-specialized.

The implemented single-kernel graph is:

```text
logical NHWC input A
-> implicit-GEMM conv0, 3x3, fp16 x fp16 -> fp32
-> ReLU accumulator epilogue
-> fp16 C-shuffle LDS tile
-> 1x1 conv1 MFMA, fp16 x fp16 -> fp32
-> ReLU accumulator epilogue
-> fp16 C-shuffle LDS tile
-> inline 2x2 stride-2 maxpool
-> final NHWK fp16 global store
```

No conv0 or conv1 intermediate tensor is written to HBM. Each CTA owns a
rectangular tile of final pooled outputs, expands that tile backward to the
required conv1 positions and conv0 halo, then computes the upstream values
locally. This avoids inter-CTA communication. The model's two-input concat is
still represented by the current harness as one logical post-concat `C=8` input;
true two-pointer virtual concat descriptor work remains separate from this
kernel dataflow proof.

## Current Tuned Defaults

The current tuned defaults are:

```text
tile_m=128
tile_n=32
tile_k=16
pool_tile=4x8
warp_m=2
warp_n=1
warp_tile=32x32x16
block_size=128
conv0_epilogue=ReLU
conv1_epilogue=ReLU
cache_input_footprint=False
direct_conv0_from_input_cache=False
```

With 2x2 stride-2 pooling, `pool_tile=4x8` maps to an 8x16 conv tile, so
`tile_m=128`. The grid is:

```text
(1, pool_ho / 4, pool_wo / 8)
```

For the full target output `[1, 1080, 1920, 24]`, this gives:

```text
grid=(1, 270, 240)
```

The default harness shape remains the smaller target-channel smoke test:

```text
N=1, H=16, W=16, C=8, K0=32, K1=24
pool_tile=4x8
tile_k=16
grid=(1, 2, 1)
```

## Correctness Milestones

Correctness has progressed through these milestones:

1. Toy dataflow: `N=1, H=W=8, C=8, K0=8, K1=8` proved the fused chain at a
   minimal size.
2. Target channels, small spatial: `N=1, H=W=16, C=8, K0=32, K1=24` is the
   default harness shape and verifies the intended channel topology across more
   than one CTA in the spatial grid.
3. Tiled image schedule: `H=32, W=64, C=8, K0=32, K1=24` with `pool_tile=4x8`
   gives `grid=(1, 4, 4)` and verifies rectangular 2D CTA tiling without
   inter-CTA communication.
4. Full exercise shape: `N=1, H=2160, W=3840, C=8, K0=32, K1=24` has been
   compiled, launched, and timed with the tuned fp16 prototype.

The harness compares against a NumPy reference that performs conv0, ReLU,
fp16 rounding, conv1, ReLU, fp16 rounding, and maxpool. The verification path
reports `max_abs_diff` and `bad_count` at a default tolerance of `1e-2`.

## Measured Timings

Current local gfx950 measurements for the full target fp16 prototype:

```text
pool_tile=4x8, tile_n=32, tile_k=16
time ~= 0.387 ms
useful throughput ~= 132 TFLOP/s
```

A later repeat timing of the baseline during direct-footprint experiments
reported about `0.369 ms`. Treat the spread between `0.369 ms` and `0.387 ms`
as run-to-run or setup variation until a fuller benchmark sweep is captured.

These timings are for the fused kernel path, not a production comparison against
the final unfused int8/int4 graph.

## tile_k Experiments

The important current result is that `tile_k=16` is the tuned default for this
fused schedule. It matches the selected 32x32x16 fp16 MFMA atom and avoids extra
padded work for the target `K0=32, K1=24` channel sizes. Larger generic conv
defaults are not the best fit here because the fused kernel has small channel
counts, two MFMA stages, and LDS staging between stages.

## Input-Cache and Direct-Footprint Experiments

Two input-footprint variants were prototyped:

```text
baseline, tile_k=16, no input cache:      ~0.387 ms
input footprint cached in LDS:            ~0.539 ms
direct footprint from cached LDS to MFMA: ~0.469 ms
repeat baseline for direct test:          ~0.369 ms
```

The input-cache mode loads each CTA's unique conv0 input footprint into LDS, then
materializes the implicit-GEMM A tile from that footprint. It is numerically
correct but slower than the baseline.

The direct-footprint mode skips materializing the im2col A tile and has MFMA A
fragments gather directly from the cached footprint. It also verifies
correctness, but remains slower than the baseline because the current
implementation uses scalar LDS fragment gathers and additional coordinate
arithmetic in the MFMA inner path.

Both modes should stay opt-in until the footprint LDS layout and fragment access
pattern are vectorized.

## Current Bottleneck Hypotheses

The latest data suggests the bottleneck is not HBM input bandwidth for the
original input footprint. Caching the input footprint in LDS increases latency,
which points instead to on-chip scheduling and LDS overhead.

Working hypotheses:

- MFMA scheduling and the two C-shuffle LDS materializations dominate the current
  critical path.
- The direct-footprint path pays too much scalar LDS gather and coordinate
  arithmetic overhead to recover the saved A-tile materialization.
- The small `K0=32, K1=24` topology makes padded or over-general tile choices
  expensive; `tile_k=16` is currently better aligned to useful work.
- The final-output CTA ownership model is correct, but any halo recompute and
  per-CTA footprint work must remain cheaper than writing intermediates to HBM.
- True virtual concat, quantized int8/int4 MFMA or packing, and production
  autotuning are still outside this fp16 dataflow proof.

## Next Useful Measurements

The next high-signal measurements are a repeatable baseline sweep around
`tile_k`, pool-tile geometry, and C-shuffle staging cost; a direct comparison
against an unfused multi-kernel fp16 reference; and a vectorized direct-footprint
variant before spending more effort on input caching.
