# gfx1151 Deep Fusion Native-Integer Pipeline Plan

Date: 2026-06-04

Scope: turn the existing gfx1151 deep-fused `conv0 3x3 -> conv1 1x1 -> maxpool`
kernel from integer storage plus fp16-WMMA emulation into a real
`iu8`/`iu4 -> i32` WMMA pipeline, and drive it toward the roofline target.

Reference repos inspected under `/workspace`:

- `/workspace/rocBLAS` cloned from `ROCm/rocBLAS`. This repo is now retired in
  favor of `ROCm/rocm-libraries`; it is mainly useful here as the rocBLAS host
  dispatch layer into Tensile, not as a source-level grouped-conv kernel.
- `/workspace/HipKittens` already present. Used for concrete high-performance
  AMD kernel scheduling and load/staging patterns.
- `/workspace/aiter` already present. Used for low-bit GEMM pipeline patterns,
  especially the `opus_gemm` explicit-layout kernels.
- CK Tile grouped convolution in
  `include/ck_tile/ops/grouped_convolution`.

## Current State

The estimate document is correct about the performance model, but the current
tree has already moved past the "no integer WMMA plumbing" starting point:

- `wmma_i32_16x16x16_iu8` is already present in the gfx1151/gfx11-generic arch
  specs, fragment maps, LLVM declarations, backend emission, and a standalone
  bit-exact `iu8` GEMM probe.
- `iu4` is not wired yet. `ir.py` knows the op-id name in the integer-accumulator
  set, but the arch catalog, lane maps, intrinsic declaration, and backend table
  do not expose it.
- The production deep-fusion kernel still uses real int8/int4 HBM storage but
  dequantizes to fp16 LDS fragments and feeds `wmma_f32_16x16x16_f16`.
- The latest measured direct-conv0 implementation is much better than the older
  32 ms baseline: the multilever note reports about 6.7 to 6.9 ms at the full
  shape, with only scheduler hints giving a small positive delta. Use that as
  the active baseline unless intentionally comparing against the older direct
  kernel.

## Library Findings

### CK Tile Grouped Conv

Useful ideas:

- The grouped-conv forward path is an implicit-GEMM adapter with separate
  prolog, GEMM pipeline, and epilogue, so the right shape for CK DSL remains
  "specialized descriptors plus a GEMM-like inner loop".
- Its LDS descriptors use bank-aware XOR permutations and data-type-dependent
  packing. This is worth copying as a layout principle when we add packed i8/i4
  LDS fragments.
- It already separates 1x1/3x3 convolution specializations and validates vector
  access sizes up front. The native-int kernel should keep this kind of
  specialization instead of adding generic runtime convolution transforms.

Not useful for the target hot path:

- The grouped-conv descriptor stack is too general for this fixed
  `N=1,H=2160,W=3840,C=8,K0=32,K1=24` fusion. Carrying its runtime
  split-image/split-N and transform machinery into CK DSL would add exactly the
  coordinate arithmetic we are trying to remove.
- CK Tile disables the async grouped-conv path outside gfx950. Treat
  `raw_ptr_buffer_load_lds` on gfx1151 as an experiment, not as the first
  implementation dependency.

### HipKittens

Useful ideas:

- Precompute global buffer resources and LDS byte bases, then issue raw
  buffer-load-to-LDS operations with minimal per-iteration address math.
- Use small ping-pong LDS buffers and place `s_waitcnt`, `s_barrier`,
  `sched_barrier`, and `s_setprio` deliberately around load/DS-read/WMMA
  clusters.
- Size tile memory ops to the matrix instruction layout and eagerly use
  swizzled LDS offsets, instead of making the compiler rediscover the layout.
- Avoid replacing an LDS handoff with cross-lane register shuffles unless the
  shuffle count is demonstrably lower. This matches the existing rejection of
  conv0-to-conv1 butterfly fusion.

### AITER / Opus GEMM

Useful ideas:

- The `a8w8` pipeline keeps all A/B layouts explicit: global layout, shared
  layout, register layout, and accumulator layout are separate compile-time
  objects. This is the right model for adding packed integer fragments to CK DSL.
- It uses a 2x4 wave mapping, half-tile staging, four LDS buffers, and manually
  interleaved async loads with DS reads and MFMA/WMMA calls. The same principle
  applies, but our tiny K counts mean the full AITER pipeline depth may not pay.
- It assumes B is already in a kernel-friendly layout when possible. For this
  fused conv, W0 and W1 are static weights, so a one-time packed/preshuffled
  weight layout is a high-value follow-up after the native path is correct.

### rocBLAS

Useful ideas:

- rocBLAS routes int8 GEMM through typed Tensile problem selection. The lesson is
  to specialize and autotune concrete low-bit problems, not to build a single
  generic convolution kernel.
- The public dispatch supports int8/int32 types, but the cloned repo does not
  expose the generated low-level Tensile kernels in a form that is directly
  reusable for this CK DSL fused op.

## Implementation Plan

### Phase 1: Complete Integer WMMA Atom Coverage

1. Add `wmma_i32_16x16x16_iu4` to the gfx1151 and gfx11-generic arch specs.
2. Add the iu4 fragment maps in `core/arch/target.py`: same 16x16 accumulator map
   as f16/iu8, but A/B operands are `<2 x i32>` with each slot holding eight
   signed int4 K-values.
3. Add the LLVM intrinsic declaration for
   `llvm.amdgcn.wmma.i32.16x16x16.iu4`.
4. Add the backend mapping in `Gfx11RdnaBackend` with signedness flags set to
   signed and clamp disabled.
5. Add a standalone `iu4` GEMM probe mirroring the existing `iu8` probe. It must
   verify bit-exact `C = A @ B.T` for signed int4 values and pin the nibble order.

Exit criteria: `iu8` and `iu4` standalone probes build for `gfx11-generic` and
verify bit-exact on the Strix Halo board.

### Phase 2: Add Packed Integer Fragment Helpers

Add CK DSL helpers for the packed fragment ABI rather than threading ad hoc bit
twiddling through the fused kernel:

- `pack_i8x4_to_i32` and `pack_i8x16_to_v4i32`.
- `pack_i4x8_to_i32` and `pack_i4x16_to_v2i32`.
- LDS/global load helpers that return packed integer WMMA fragments directly.
- Optional vectorized fast paths for the target shape: `C=8` means conv0 K tiles
  are two contiguous channel groups per 16-wide WMMA K step.

Use the current `cvt_pk_i8_f32x4`/`v_perm_b32` lowering as the model for compact
packing. Avoid scalar per-nibble chains in the inner loop.

Exit criteria: fragment helpers have focused unit/probe coverage and generated
ISA shows packed integer operations instead of fp16 dequant/pack.

### Phase 3: Native `iu8` Conv0

Replace conv0's fp16-WMMA body with `wmma_i32_16x16x16_iu8`:

- Keep the direct input-footprint staging idea; it was the big proven win.
- Store the input footprint and W0 as integer bytes or packed i32 lanes, not f16.
- Build each conv0 A fragment from the footprint as `<4 x i32>`.
- Build each W0 B fragment as `<4 x i32>`, padding K=72 to 80 exactly as the
  estimate assumes.
- Accumulate in `<8 x i32>`.
- Remove the fp16 emulation `rint` snap from the conv0 epilogue.

Quantization is now the next major VALU risk. The current scales are powers of
two (`m0=1/16`, `m0b=1/2`), so add an integer round-to-nearest-even shift path
for these constants before falling back to f32 scaling for generic experiments.

Exit criteria: conv0-only reference path is bit-exact, generated ISA contains
`v_wmma_i32_16x16x16_iu8`, and fp16 conversion instructions disappear from the
conv0 hot loop.

### Phase 4: Native `iu4` Conv1

Replace conv1's fp16-WMMA body with `wmma_i32_16x16x16_iu4`:

- Change the conv0-to-conv1 handoff from f16 LDS to signed int4 codes packed as
  bytes or i32 words.
- Keep the LDS handoff. The previous butterfly/register-transpose analysis still
  applies; native int does not change the cross-lane geometry.
- Load W1 in its existing packed-int4 HBM format and assemble `<2 x i32>` B
  fragments with no fp16 dequant.
- Assemble `<2 x i32>` A fragments from the packed conv0 output codes.
- Accumulate in `<8 x i32>`.
- Use integer round-to-nearest-even shift for `m1=1/4` and keep the ReLU as an
  integer max against zero.

Exit criteria: generated ISA contains `v_wmma_i32_16x16x16_iu4`, W1 dequant to
fp16 is gone, and the full conv0+conv1 integer reference is bit-exact.

### Phase 5: Integer Maxpool And Final Pack

Maxpool compares int4 codes, so it should not round-trip through f16:

- Store conv1 post-quant/ReLU codes as integer int4/int8 values in LDS.
- Vectorize the 2x2 max over integer codes.
- Final `mf=1.0` can be a no-op clamp for the target scale; preserve a generic
  fallback only if needed for alternate scale experiments.
- Keep the existing final output layout: three i32 words per pooled pixel, eight
  signed nibbles per word.

Exit criteria: full fused kernel is bit-exact against the existing integer numpy
reference and emits no fp16 conversions in the conv/quant/maxpool hot path except
for any explicitly retained generic-scale fallback.

### Phase 6: Performance Levers After Correctness

Adopt only the proven f16-emulation levers initially:

- Keep direct-conv0 enabled.
- Keep early W1 loading.
- Use scheduler hints (`compv3`) as the first default candidate, since it was the
  only positive composition lever.
- Do not enable `waves_per_eu=2` by default; it was VGPR-negative.
- Do not enable masked maxpool by default; it activated redundant waves.
- Do not revive butterfly conv0-to-conv1 handoff unless a new integer-specific
  fragment map eliminates the cross-lane transpose, which is unlikely.

Then run a second-stage sweep inspired by HipKittens/AITER:

- Packed W0/W1 preshuffle layouts for direct fragment loads.
- Optional raw buffer-load-to-LDS staging for dense W0/W1 tiles.
- LDS swizzle/padding variants for packed byte/nibble storage.
- Grid ordering variants only after the native integer kernel is stable; the APU
  partition is small, so chiplet-aware scheduling is less likely to dominate.

## Performance Targets

Use two target ladders:

- Against the older estimate baseline, native int should land in the original
  11 to 16 ms near-term band immediately.
- Against the current direct-conv0 baseline of roughly 6.7 to 6.9 ms, the first
  meaningful native-int target is below 5 ms. A credible roofline chase starts
  around 3 to 4 ms, and the padded matrix floor remains about 2.2 ms.

The main risk is not matrix throughput. The main risk is scalar packing,
coordinate arithmetic, quantization, and LDS handoff overhead. The plan therefore
prioritizes integer quant fast paths and packed-fragment helpers before adding
deeper async scheduling.

## Validation Runbook

1. Run standalone `iu8` and `iu4` probes first.
2. Add a conv0-only debug mode and verify exact int32 accumulators before quant.
3. Add conv0 quant and verify int8/int4 intermediate codes.
4. Add conv1-only-on-staged-codes and verify exact int32 accumulators.
5. Enable full fusion and compare against the existing integer-exact numpy
   reference at toy, multi-CTA, and full shapes.
6. For every performance quote, record median, spread, generated ISA checks, and
   whether the expected `v_wmma_i32_*_iu8/iu4` instructions were present.
