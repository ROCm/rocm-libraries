# gfx950 Deep Fusion Optimization Digest

This is the compact record for the gfx950 CK DSL deep-fusion prototype. It
replaces the longer dated experiment notes in this directory.

## Scope

Prototype graph:

```text
logical C=8 input
-> conv0 3x3, K0=32
-> ReLU
-> conv1 1x1, K1=24
-> ReLU
-> 2x2 stride-2 maxpool
-> final NHWK store
```

The current prototype is a gfx950-only fp16/fp32-accumulation proof. It does not
yet implement the original int8/int4 packed model path, true two-pointer virtual
concat, production autotuning, or generic graph fusion.

Full target shape:

```text
input:     [1, 2160, 3840, 8]
conv0:     [1, 2160, 3840, 32]
conv1:     [1, 2160, 3840, 24]
pool out:  [1, 1080, 1920, 24]
```

Correctness target used in the optimization records: NumPy reference, tolerance
`1e-2`, final verified result `max_abs_diff=0.00195312`, `bad=0/49766400`.

## Result

The optimized fused kernel moved from about `0.357 ms` / `143 useful TFLOP/s` to
about `0.178 ms` / `284-287 useful TFLOP/s`, roughly a 2x speedup.

The important lesson is that the kernel was not HBM-bound and not MFMA-bound.
The wins came from removing VALU, LDS, and synchronization overhead on the
operand-delivery path.

## Current Winning Configuration

Keep these as the known-good gfx950 defaults for the fp16 prototype:

- `pool_tile=4x4`
- `tile_m=64`
- `tile_n=32`
- `tile_k=32`
- `warp_m=2`
- `warp_n=1`
- `block_size=128`
- `pipeline=mem`
- `async_dma=False`
- conv1 LDS reads vectorized
- merged conv0-cshuffle and W1-load barrier
- conv1 epilogue deferred until after pool
- maxpool gather vectorized when using the LDS fallback path
- register-resident intra-lane pool for the best geometry
- `decompose_m=False` A-descriptor path for the fused conv0 load

This configuration intentionally accepts more padded MFMA work than the early
`4x8/tk16` shape. It wins because it cuts per-CTA overhead: lower VALU count,
lower VGPR/LDS pressure, lower LDS wait, and fewer handoff costs.

## Optimization Chain

- Baseline: `pool_tile=4x8`, `tile_m=128`, `tile_k=16`, `pipeline=mem`.
  Measured about `0.357 ms`, `143 useful TFLOP/s`.
- Conv1 LDS read vectorization: replaced scalar `ds_read_b16` gathers with wide
  `ds_read_b128` where K has no tail. This was the first large win:
  about `0.253 ms`, `201 TFLOP/s`.
- Barrier merge: collapsed two independent producer barriers into one before the
  conv1 MFMA. Same instruction counts, but lower critical-path serialization:
  about `0.246 ms`, `207 TFLOP/s`.
- Geometry change to `4x4/tile_m=64/tk32`: more padded MFMA but much less VALU
  per CTA. Measured about `0.224 ms`, `228 TFLOP/s`.
- Lever A: deferred conv1 epilogue past maxpool using
  `relu(max(x)) == max(relu(x))`. Measured about `0.219 ms`, `233 TFLOP/s`.
- Lever B: vectorized maxpool LDS gather to wide reads. Structurally cleaned up
  the read path, but wall-clock was nearly flat: about `0.218 ms`,
  `234 TFLOP/s`.
- Lever C: eliminated the conv1-to-pool LDS handoff for the best geometry by
  doing the 2x2 pool register-resident within each lane. This was the largest
  later win: about `0.184 ms`, `277 TFLOP/s`.
- Lever D: bypassed redundant A-descriptor `m -> (n, ho, wo)` magic-division
  decode by passing `(n, ho, wo)` directly. Final measured range:
  `0.1777-0.1792 ms`, `284-287 TFLOP/s`.

## Bottleneck Read

Hardware counters reframed the problem:

- Early baseline: `MfmaUtil` about `6%`, `VALUBusy` about `48%`,
  `MemUnitStalled` about `0.06%`, and LDS wait dominated.
- After vectorizing conv1 LDS reads, LDS wait dropped sharply and VALU became the
  clear limiter.
- Best later captures still showed `VALUBusy` around `63%`, with MFMA and global
  memory mostly idle.

The right optimization target is therefore VALU/LDS/synchronization overhead,
not additional MFMA throughput or input-footprint caching.

Useful FLOP accounting for the target shape:

- conv0 useful work: `38.22 GFLOP`
- conv1 useful work: `12.74 GFLOP`
- total useful conv work: `50.96 GFLOP`
- early `4x8/tk16` hardware-padded work: `59.45 GFLOP`, `85.7%` useful/hardware
- later `4x4/tk32` hardware-padded work: more padding, but faster due to much
  lower overhead

## Rejected Or Closed Paths

- Input-footprint LDS cache: slower. It adds LDS fill, sync, and address math
  without recovering a meaningful HBM stall.
- Direct footprint path: slower, and later found incorrect at full target shape.
  It removes regular A-tile materialization but replaces it with scalar LDS
  gathers and more coordinate arithmetic.
- `async_dma=True`: incorrect for this fused carrier and slower even when judged
  only by timing. There is no memory-latency wall to hide.
- `unroll_k=True`: incorrect because it drops synchronization needed by the
  fused rectangular K-tile carrier.
- C-shuffle store vectorization: blocked by MFMA C-fragment geometry. Per-lane
  accumulator values are same-column/different-row, while downstream conv1 reads
  require row-major same-row/contiguous-column layout. A layout that fixes stores
  breaks the read vectorization.
- Further maxpool LDS-read work: mostly closed. The best path avoids the
  conv1-to-pool LDS handoff entirely.

## Remaining Work

- The conv0-to-conv1 handoff is still the hard structural handoff. It is an
  inherent M-to-K0 transpose, unlike the conv1-to-pool handoff that was removed
  with intra-lane pooling.
- Further gains likely require cutting residual VALU in coordinate arithmetic,
  conversion, and staging code. Expect incremental wins unless a new structural
  handoff reduction appears.
- Compare the final fused fp16 kernel against an unfused multi-kernel fp16
  pipeline for the same shape. The notes so far establish single-kernel progress,
  not the full end-to-end production win.
- Real int8/int4 MFMA, packing, quantization, true virtual concat, and production
  autotune remain separate follow-up work.
