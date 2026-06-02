# gfx950 Deep Fusion Plan

This page captures the implementation plan for a gfx950-first CK DSL deep fusion
prototype. It is a planning document, not a statement of shipped support.

The goal is to demonstrate deep kernel fusion with the MFMA and dtype paths that
CK DSL already supports on gfx950, then use that prototype to identify the
minimal reusable abstractions needed for broader conv / quant / pool fusion.

## Target Workload Shape

The motivating graph is a CNN-style chain:

```text
virtual concat
-> conv0
-> epilogue chain
-> conv1
-> epilogue chain
-> maxpool
-> final store
```

The original model graph uses int8 / int4 quantized convolutions. The gfx950 v1
prototype should intentionally use supported fp16 / bf16 MFMA first:

```text
Input0:  NHWC fp16 or bf16
Input1:  NHWC fp16 or bf16
W0:      KRSC fp16 or bf16
W1:      KRSC fp16 or bf16
Output:  NHWC/NHWK fp16 or bf16
Acc:     fp32
```

Quantized behavior should be represented by epilogue transforms where useful
(`scale`, `clamp`, `relu`, fake-quant), but real int8 MFMA, int4 activation
packing, and int4 convolution are deferred until the scheduling model is proven.

### Exercise Target Shape

The optimization target for this exercise is the encoder block shown in the
model summary:

```text
Input0:      (1, 4, 2160, 3840)
Input1:      (1, 4, 2160, 3840)
Concat:      2 x int8 -> int8
             2 x (1, 4, 2160, 3840) -> (1, 8, 2160, 3840)

Conv0:       3 x 3, int8 -> int32
             (1, 8, 2160, 3840) -> (1, 32, 2160, 3840)
Quant0:      int32 -> int8
ReLU0:       int8 -> int8
Quant0b:     int8 -> int4

Conv1:       1 x 1, int4 -> int32
             (1, 32, 2160, 3840) -> (1, 24, 2160, 3840)
Quant1:      int32 -> int4
ReLU1:       int4 -> int4

MaxPool:     2 x 2
             (1, 24, 2160, 3840) -> (1, 24, 1080, 1920)
FinalQuant:  int4 -> int4

Fusion I/O:  2 x (1, 4, 2160, 3840) -> (1, 24, 1080, 1920)
```

For layout in CK DSL, represent these as NHWC during the prototype:

```text
Input0/Input1: [N=1, H=2160, W=3840, C=4]
Virtual concat logical input: [1, 2160, 3840, 8]
Conv0 output: [1, 2160, 3840, K0=32]
Conv1 output: [1, 2160, 3840, K1=24]
Pool output:  [1, 1080, 1920, 24]
```

The current checked-in bring-up harness uses target channel counts at a small
spatial size:

```text
N=1, H=W=16, C=8, K0=32, K1=24, R=S=3, pool=2x2 stride 2
pool_tile=4x8
tile_n=32, tile_k=32
grid=(1, 2, 1)
```

That toy shape verifies the single-kernel dataflow
`conv0 -> 1x1 conv1 -> maxpool` across multiple CTAs in both spatial tile
axes. It is not the final correctness or performance target. It preserves the
model's target channel sizes (`concat C=8`, `K0=32`, `K1=24`) before scaling H/W
to the full 2160 x 3840 image.

## Why Start With Supported MFMA

The first milestone is not "add every quantized primitive." It is to prove that
CK DSL can author a single kernel where a downstream non-elementwise stage
consumes an upstream stage's tile without materializing the intermediate in
global memory.

Using existing gfx950 MFMA paths keeps the hard problem focused on:

- producer-consumer tile ownership;
- descriptor-driven virtual concat;
- reusable epilogue transforms on accumulator fragments;
- inline pooling when the final image is split across CTAs;
- correctness and performance accounting for recomputed halos.

Once this works with fp16 / bf16, int8 / int4 support can be added as a dtype and
packing extension rather than as a simultaneous compiler, scheduling, and ISA
bring-up effort.

## Contract

Add a new gfx950-focused instance or experimental builder with a narrow v1
validator:

```text
Y = pool(epi1(conv1(epi0(conv0(concat(X0, X1), W0)), W1)))

X0, X1:     NHWC, same N/H/W, channel-concatenated logically
W0:         KRSC for conv0
W1:         KRSC for conv1
Conv0 acc:  fp32
Conv1 acc:  fp32
Epi0/Epi1:  static epilogue op chains
Pool:       maxpool v1
Y:          final output tensor
```

The v1 contract should be stricter than the eventual target:

- gfx950 only;
- fp16 first, bf16 after parity;
- NHWC input layout and KRSC weights;
- static shape fields in the spec for N/H/W/C/K/R/S;
- one fixed pool form first, preferably 2x2 stride 2 maxpool;
- conv1 should start as 1x1 so the first fused prototype avoids a second spatial
  halo problem;
- no inter-CTA communication;
- no graph capture or automatic pattern matching.

The one-CTA constraint was only for bring-up. The current prototype uses
final-output tile ownership: each CTA owns a rectangular tile of pooled output
pixels and computes the required conv1 and conv0 source region locally. The
current tuned prototype uses `pool_tile=4x8`, `tile_n=32`, and `tile_k=32` to
reduce padded MFMA work for the target `K0=32`, `K1=24` channel topology.

### Shape Milestones

Use these shape milestones to avoid conflating "dataflow works" with "target
shape works":

1. **Toy correctness**: `N=1, H=W=8, C=8, K0=8, K1=8`.
   This remains useful as the fastest smoke test if compile time becomes noisy.
2. **Target channels, small spatial**: `N=1, H=W=16, C=8, K0=32, K1=24`.
   This is the current default harness shape and verifies the exact channel
   topology of the model summary with a rectangular `grid=(1,2,1)` split.
3. **Tiled image schedule**: `N=1, H/W` large enough to require multiple CTAs,
   still with `C=8, K0=32, K1=24`.
   The tuned 2D tiled schedule has been checked at `H=32, W=64`; with
   `pool_tile=4x8` it yields `grid=(1,4,4)`. It proves rectangular spatial
   tiling without inter-CTA communication.
4. **Full exercise shape**:
   `N=1, H=2160, W=3840, C=8, K0=32, K1=24, pool output 1080x1920`.
   This is the performance target. Current fp16 prototype timing with
   `pool_tile=4x8`, `tile_n=32`, `tile_k=32` is about `0.39 ms`
   (`~130 useful TFLOP/s`) on the local gfx950 run.

Correctness claims must state which milestone was verified.

## Placement

Use `instances/gfx950/` if the first implementation hard-codes gfx950 lane math,
tile geometry, or a specific hand-fused schedule.

Use `instances/common/` only after the implementation is expressed in terms of
target-neutral helpers such as `MfmaAtom`, `WarpGrid`, transform DAG descriptors,
and epilogue callbacks, with `is_valid_spec(spec, arch)` rejecting unsupported
architectures.

Recommended v1 placement:

```text
instances/gfx950/deep_fused_conv_pool.py
examples/gfx950/deep_fused_conv_pool_verify.py
```

Promotion to `instances/common/` should be a follow-up, not a requirement for
the proof of concept.

## Execution Model

The fused kernel should be scheduled around the final output tile, not around an
already-materialized intermediate conv tile.

Each CTA owns a tile of final pooled outputs:

```text
pooled output tile
=> required conv1 output positions
=> required conv0 output positions
=> required original input positions and halo
```

The CTA computes all upstream values needed for its final output tile locally.
This avoids inter-block communication, which is not available inside a normal
CK DSL kernel.

For a 2x2 stride 2 pool, each pooled output needs a 2x2 patch of conv1 outputs.
If conv1 is 1x1, each conv1 output needs only the corresponding conv0 channel
vector. Conv0 then owns the only spatial halo in the first milestone.

## Virtual Concat

Do not implement concat as a separate device operation. Implement it as input
descriptor logic for conv0:

```text
logical channel c:
  if c < C0:
    load X0[n, h, w, c]
  else:
    load X1[n, h, w, c - C0]
```

The descriptor callback should present a single logical NHWC tensor to the conv0
loader. This keeps concat free except for a channel predicate and preserves the
single-kernel fusion story.

## Epilogue Chain

The first reusable abstraction should be a device-level epilogue transform that
can be used in two modes:

```text
accumulator fragment -> transformed value -> store
accumulator fragment -> transformed value -> next stage input tile
```

Initial ops should stay close to existing `FusedEpilogue` semantics:

- bias add;
- scale;
- relu;
- clamp;
- cast to fp16 / bf16;
- optional fake-quant as `round/clamp/dequant` while final output remains fp16.

This is still static in-kernel composition. It should not launch recursive child
kernels from the epilogue.

## Stage Interface

The deep fusion prototype should not try to compose existing top-level kernels.
Top-level kernels own incompatible grid mappings, argument layouts, LDS
allocation, barriers, main loops, and epilogue contracts.

Instead, define a small internal stage contract for hand-authored fusion:

```text
Stage.produce(tile_coords) -> tile fragment or LDS tile
Stage.consume(tile, tile_coords) -> next tile fragment
```

For v1, this can be a local convention inside the fused instance rather than a
public helper API. Extract it only after the first kernel demonstrates the right
dataflow.

## Inline Pooling

Pooling is a windowed reduction and should not use MFMA. Inline pooling should
reduce conv1 results in registers or LDS before the final store.

The scheduler must account for output splitting:

- A CTA cannot read conv1 results produced by another CTA unless those results
  were stored to global memory, which would break deep fusion.
- Therefore each CTA computes every conv1 value needed by its own pool windows.
- Tile boundaries are handled by backward planning from pooled output
  coordinates to upstream conv coordinates.
- Overlap between neighboring CTAs is allowed for the prototype and should be
  measured. If overlap becomes too expensive, the schedule is wrong for the
  chosen shape.

For v1, use non-overlapping 2x2 stride 2 maxpool. Larger or overlapping pooling
windows should remain follow-up work.

## Milestones

### 1. Conv Epilogue Hook

Add a conv-local fused epilogue callback that can transform fp32 accumulators
before direct or cshuffle store.

Deliverable:

```text
conv_implicit_gemm fp16/bf16
-> bias
-> scale
-> relu/clamp
-> fp16/bf16 store
```

This may initially duplicate some `FusedEpilogue` behavior rather than forcing
the GEMM fusion API to fit conv.

### 2. Fused Conv0 -> Conv1 Skeleton

Author a shape-specialized gfx950 kernel where conv0 output stays on chip and
feeds a 1x1 conv1.

Deliverable:

```text
virtual concat
-> conv0
-> epilogue transform
-> conv1 1x1
-> final store
```

Start with a small spatial tile and conservative LDS use. Optimize after
correctness.

### 3. Inline MaxPool

Extend the skeleton so each CTA owns pooled output coordinates and computes the
required conv1 patch locally.

Deliverable:

```text
virtual concat
-> conv0
-> epilogue transform
-> conv1 1x1
-> relu/clamp
-> 2x2 stride 2 maxpool
-> final store
```

### 4. Fake Quant Epilogues

Add fake-quant epilogue ops to mimic the quantized model chain while staying on
supported fp16 / bf16 MFMA.

Deliverable:

```text
acc_fp32
-> scale
-> round/clamp
-> dequant or cast
```

Use this to compare numerical behavior against a reference model before adding
real int8 / int4 compute.

### 5. Performance Pass

Tune only after the dataflow is correct:

- tile size for final pooled outputs at the full `1080 x 1920 x 24` output
  shape;
- conv0 halo size and recompute overhead;
- LDS layout for conv0 intermediates;
- LDS layout for conv1 1x1 accumulation over `K0=32 -> K1=24`;
- cshuffle vs direct epilogue for final stores;
- vector width and coalescing for virtual concat loads;
- global-load reuse for the two `1 x 4 x 2160 x 3840` inputs;
- chiplet swizzle if the output tile mapping benefits from L2 reuse.

## Deferred Work

The following are explicitly out of scope for the gfx950 v1 proof:

- int8 MFMA bring-up in CK DSL;
- int4 activation tensor type;
- packed int4 conv input/output;
- general conv1 spatial kernels beyond 1x1;
- overlapping maxpool windows;
- automatic graph fusion through `compile_fn`;
- recursive kernel launches from epilogue code;
- production autotuning across arbitrary shapes.

## Test Harness

Add a gfx950 verification harness with a pure torch / numpy reference:

```text
examples/gfx950/deep_fused_conv_pool_verify.py
```

Required behavior:

- generate `X0`, `X1`, `W0`, `W1`, bias/scale tensors;
- compute the reference as explicit concat, conv0, epilogue, conv1, epilogue,
  maxpool, and final cast;
- compile and launch the fused CK DSL kernel;
- compare output with a tolerance appropriate for fp16 / bf16 accumulation and
  epilogue order;
- report intermediate shape planning, including final output tile, conv1 patch,
  conv0 output patch, and input halo;
- support `--arch gfx950`, `--dtype fp16|bf16`, `--verify`, `--bench`, and shape
  overrides for the v1 static shape family.

The benchmark should compare against a multi-kernel reference pipeline, not just
against torch eager. The primary metric is avoided intermediate HBM traffic and
end-to-end latency for the target shape.

## Success Criteria

The proof is successful when all of the following are true:

- one CK DSL kernel computes the full v1 chain;
- no conv0 or conv1 intermediate is written to global memory;
- pooling is inline and does not require inter-CTA communication;
- correctness matches the reference within the documented tolerance;
- the benchmark reports end-to-end latency against the unfused pipeline;
- the implementation clearly separates reusable pieces from gfx950-specific
  scheduling choices.

## Estimated Effort

For a fixed-shape gfx950 prototype using supported MFMA:

```text
conv epilogue hook:       1-2 weeks
conv0 -> conv1 skeleton:  3-5 weeks
inline pooling:           1-2 weeks
fake quant + validation:  1-2 weeks
performance pass:         1-2 weeks
```

Expected total: 6-10 engineer-weeks for a credible prototype.

Turning this into a reusable deep fusion framework is a separate 3-6 month
effort, mostly because the reusable scheduler must reason about tile ownership,
halo expansion, recompute, LDS pressure, and fusion legality across operator
families.
