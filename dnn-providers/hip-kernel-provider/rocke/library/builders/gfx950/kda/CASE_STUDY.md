# Chunkwise KDA on gfx950 — optimization case study

How the chunkwise gated delta-rule linear attention (KDA) prefill kernels in
`kernels/gfx950/kda_chunkwise.py` were built and tuned against an existing
reference implementation on a 32-CU gfx950 partition.

Per [`AGENTS.md`](../../../../AGENTS.md) §Compliance this document records
**methodology and levers only**. No measured numbers, ratios, or achieved
bandwidths appear here; those live on the protected Confluence page. Everything
below is replayable — each claim names the harness that produced it, so the
numbers can be regenerated on demand rather than quoted.

## 1. What the workload is

KDA is a gated delta-rule linear attention. Tokens are grouped into chunks of
`C`, and per chunk the token-serial recurrence factors into six
**state-independent** tiles plus a short **state-dependent** recurrence. The
full contract, including the overflow-avoiding factorization, is in the module
docstring of `kernels/gfx950/kda_chunkwise.py`.

The split matters more than any individual optimization:

- The six tiles (`A`, `GK`, `GQ`, `Aqk`, `Kt`, `dec`) depend only on that
  chunk's `q/k/g/beta`. Fully parallel over the sequence.
- The recurrence carrying the state `S` across chunks is serial per
  (batch, head), and parallel only across heads.

Two halves with opposite parallelism profiles, which is what makes the fusion
question (§5) interesting rather than obvious.

## 2. Harnesses

Nothing was tuned without a correctness gate attached to the same run. Three
independent references, each checking a different claim:

| Harness | Reference | What it establishes |
| --- | --- | --- |
| `kda_chunk_prep.py` | float64 tile oracle | The six tiles, individually |
| `kda_chunk_split.py` | token-serial float64 walk | The chunkwise factorization itself |
| `kda_chunk_fused.py` | token-serial float64 walk | Same, for the fused path |

The tile oracle is deliberately built by a different route than the kernel: it
forms the pairwise exponent *difference* directly, with no midpoint factoring
and no `C x C x DK` reuse. So agreement checks the factorization rather than
restating it. The token-serial oracle is not chunked at all.

`tests/test_kda_chunkwise_gfx950_numeric.py` runs all three across the gate
range, plus a bitwise-equality check between the split and fused paths (they
share one emitted scan body, so tolerance-based agreement would be too weak a
statement).

The gate range is swept to `-5`, the reference `gate_lower_bound`. This is not
padding: a 32-token chunk accumulates up to 160 nats there, which overflows
fp32 if the ratio `Gamma_i / Gamma_j` is formed directly. Only that end of the
sweep exercises the factoring and the clamps.

## 3. Method: attribute before optimizing

Every lever below came from an attribution step, not a guess. Two techniques
did the work.

### 3.1 Phase ablation

A build-time switch dropped individual phases (cumsum, elementwise, the `C x C`
MFMAs, the triangular solve, each store) and the kernel was re-timed. Each
ablation breaks correctness, so it only ever answers "what does this phase
cost" — and because the compiler dead-codes anything whose only consumer was
ablated, each delta is an *upper* bound on the named phase.

This is what ranked the phases and pointed at the triangular solve and the
elementwise sweep as the two worth restructuring. The scaffolding was removed
once it had served its purpose; it is recorded here rather than kept in the
kernel, since a phase switch that breaks correctness has no business shipping.

### 3.2 LDS ballast: separating occupancy from instruction count

A wall-clock number cannot distinguish "starved of concurrency" from
"serialized on a dependency chain". To separate them, an **unused** LDS
allocation was added to hold the instruction stream fixed while varying only
the workgroups resident per CU.

**This probe was initially broken and gave a confidently wrong answer.** An LDS
allocation with no accesses is eliminated outright, leaving
`group_segment_fixed_size` unchanged — so the probe measured nothing while
appearing to show that occupancy was irrelevant. That conclusion survived
several rounds of tuning before it was caught.

The fix is one keep-alive store from one thread. With the allocation actually
live, dropping from two workgroups per CU to one turned out to cost a large
multiple on the tile work — which reframed the whole fusion question in §5.

**Lesson, and the reason this is written down:** a probe that reports "no
effect" is indistinguishable from a probe that does not run. Verify the
mechanism moved before trusting a null result. For an LDS probe, that means
reading `group_segment_fixed_size` out of the code object, not reasoning about
the source.

## 4. Levers that paid off, in the order they were found

1. **128-bit LDS access throughout.** Every staging tile's padded row pitch is
   constrained to a multiple of 8 elements so operands move as `ds_read_b128` /
   `ds_write_b128` rather than scalar transactions. The pad is also bank-conflict
   padding: the MFMA operand reads are at a row stride of `DK`, and an unpadded
   128-element bf16 row puts every lane in the same 16 banks.

   The `pad_cb` knob exists because of a bug caught here: a `C x C` bf16 tile at
   a pitch that is a multiple of 4 but not 8 puts odd rows on an 8-byte
   boundary, silently violating the `ds_read_b128` alignment contract. That is
   now an `is_valid_spec` rejection rather than a comment.

2. **One fused elementwise sweep.** The separate passes producing `GK`, `GQ` and
   the three MFMA operands all read the same `q`, `k` and cumulative-gate values.
   Merging them into a single sweep, where a thread owns eight consecutive
   channels of one row, cut both the LDS traffic and the exponential count —
   each `exp2` result feeds several consumers.

3. **Blocked triangular solve.** The solve's arithmetic splits into per-block
   forward substitution (irreducibly scalar VALU) and a rank update against
   already-solved blocks (a matmul, so MFMA). The substitution shrinks as the
   *square* of the block size, so smaller blocks move more of the `O(C^3)` work
   onto the MFMA pipe at the cost of one more block step. `solve_block` exposes
   the trade; the degenerate `solve_block == chunk` is the original unblocked
   scalar solve.

   This step introduced a race: `ds_write_b128` followed by a `ds_read_b128`
   from a *different* lane needs an explicit `s_waitcnt lgkmcnt(0)` between
   them. It presented as a correctness failure only at the smaller block sizes,
   where the cross-lane distance is shortest.

4. **Keeping every product in `A B^T` form.** The state recurrence is written
   transposed throughout, which puts the contraction on the fastest axis of both
   operands for all five of its matrix products. No operand ever needs an LDS
   transpose. This is a layout decision, not an optimization applied afterward —
   it is why §5's scan is as cheap as it is.

5. **No fp32 staging tile for the residual.** In `R^T = V^T - Z^T`, the
   accumulator's lane mapping puts each lane's slots at one chunk row and four
   runs of four consecutive `v` channels. So `V` is subtracted straight into the
   accumulator with four short vector loads, and only the bf16 result reaches
   LDS. The same observation lets `O` go straight to HBM from the accumulators:
   a slot's column index is `lane % 32`, so one store instruction is already
   coalesced across 32 consecutive channels.

## 5. The fusion that lost, and why that is the interesting result

The obvious design fuses both halves: one workgroup per (batch, head) walks its
chunks in order, tiles are produced and consumed in LDS, and the only HBM
traffic is the inputs and the output. It is strictly less traffic than
materializing the tiles — the split path writes every tile once and reads it
back — and `build_kda_chunk_fused` implements exactly that.

It is measurably slower, and the ablation harness plus the (repaired) ballast
probe explain why without ambiguity:

- Ablating the scan showed the fused kernel's **tile** phase alone costs far
  more than the standalone tile kernel doing identical work.
- A looped tile-only kernel writing to HBM destinations ruled out the chunk loop
  structure and the LDS-versus-HBM destination switch.
- Forcing the standalone tile kernel down to one workgroup per CU with ballast
  reproduced the fused kernel's tile-phase cost almost exactly.

So the cost is entirely occupancy. Holding the tile builder's staging tiles and
the scan's operands live simultaneously puts the workgroup over half the LDS
budget, so one workgroup fits per CU. The scan is a latency-bound chain of small
matmuls with a serial dependency between chunks; at one workgroup per CU there
is no second workgroup to cover that latency, and it bleeds through into the
tile phase too.

**The conclusion generalizes past this kernel:** minimizing memory traffic and
maximizing occupancy are competing objectives, and when the consumer is
latency-bound rather than bandwidth-bound, occupancy wins. Paying for a tile
round trip to keep two workgroups resident is the cheaper side of that trade.

This is encoded in the spec rather than left as a comment:
`is_valid_scan_spec` checks `lds_bytes()` against `LDS_LIMIT //
min_occupancy`, not against the full budget. A scan spec that would fit the
hardware but not two-per-CU is **rejected**, because it would not be worth
running.

## 6. Where each path wins

Both paths ship, because their grids differ:

| Path | Tile grid | Scan grid |
| --- | --- | --- |
| Split (`build_kda_chunk_prep` + `build_kda_chunk_scan`) | one WG per chunk | one WG per (batch, head) |
| Fused (`build_kda_chunk_fused`) | one WG per (batch, head) | same WG |

The split path is the default and is the one that beats the reference at batch
scale. Its scan grid is `BH`, so it needs `BH` comfortably above the CU count.
Below that the scan is parallelism-starved and a reference implementation that
splits the `v` dimension to manufacture more workgroups will win instead — a
real limitation, and the honest place for it is here rather than hidden in a
shape sweep. Splitting `v` in this scan would require a second partitioning rule
for the same accumulator, which is how cross-wave reductions creep in; it has
not been implemented.

The fused path is retained because it is the right shape of answer for a
future arch with either a larger LDS budget or cheaper cross-chunk latency, and
because the bitwise agreement test makes it a strong check on the split path.

## 7. Reproducing

```bash
cd rocke/library/builders/gfx950/kda

python kda_chunk_prep.py                    # six tiles vs the float64 oracle
python kda_chunk_split.py                   # end to end, default path
python kda_chunk_fused.py                   # end to end, fused path
python kda_chunk_split.py --no-check --shapes 32x16x2048
```

The pytest lanes:

```bash
cd rocke/library
python -m pytest tests/test_kda_chunkwise_spec.py            # specs + compile, no GPU
python -m pytest tests/test_kda_chunkwise_gfx950_numeric.py -m gpu
```

Timing note: the benchmarks wrap launches in `time_launches`, which runs under
`no_fence()`. Timing fenced launches instead charges every iteration a host
stream sync (`LaunchConfig.fence` defaults to `True`), which is not how the
kernels run in a pipeline — the scan consumes the tiles on the same stream, so
FIFO order already covers the dependency. At short sequences that sync is a
large fraction of the measurement, and an early version of these benchmarks
understated the kernels because of it.

## 8. Aligned raw split path (stack-gap closure)

The production gap on equal-length C=32 shapes was traced to **host
preprocessing and layout packing**, not to the core recurrence math. The aligned
closure keeps the ten-shape contract fixed (`bf16` token-major I/O, `DK=DV=128`,
`C=32`, `Tseq % 32 == 0`, V-first fp32 final state) and adds a **default-off**
raw producer on the split path:

- `KdaChunkPrepSpec.raw_inputs` plus fused `q/k` L2 norm, `lower_bound *
  sigmoid(exp(A_log) * (g + dt_bias))`, and `sigmoid(beta)` at the load/commit
  seam. Prepared chunk-packed signatures stay byte-identical when raw mode is off.
- Raw beta has separate BF16 and FP32 compile-time prep ABIs. Both carry the
  `[B,T,H]` element strides and apply sigmoid in FP32; only the BF16 variant
  extends the load. This removes framework-side conversion and materialization
  while preserving direct FP32 input for callers that already own it.
- `KdaChunkScanSpec.value_splits in {1,2,4,8}` launches `(BH * value_splits)`
  workgroups with disjoint V/state slices, mirroring FlashKDA K2's independent
  `ceil(V/BW)` grid dimension.
- Token-major scan I/O removes the head-major pack from the hot path.

### Step 0 sweep space (methodology only)

Batch-compile the Cartesian product of legal `value_splits`, prep/scan tile
`block_size`, `scan_atom_m`, padding knobs, and `solve_block`, then correctness-
prune against the aligned float64 oracle (`kda_chunk_split.ref_aligned_raw`) and
the token-serial walk. Survivors are timed with identical event windows and at
least five median blocks via the in-repo benchmark scenario:

```bash
cd rocke/library
python -m benchmarks.gfx950.kda.benchmark_chunkwise --path split \
    --shapes 8x8x1024 8x16x2048 32x16x2048
```

The builder-local `bench()` helpers expose the same timing hooks for focused
experiments. Keep measured results outside the repository per compliance.

Before profiling, inspect HSACO resources and ISA for each survivor: VGPR/SGPR/
LDS occupancy, vector load/store widths, barrier/wait counts, shuffle reductions,
and transcendental growth on the raw producer. Profile one **underfilled**
`BH=12` shape and one filled shape to separate grid starvation, transform cost,
LDS stalls, and lost prefetch overlap. Revert levers that do not move the
bottleneck named by ATT.

The cutovers in `tuned_kda_chunk_scan_spec` were encoded only after that sweep
showed stable `BH/CU` regions; rerun it before carrying them to another
architecture. Raw monolithic fusion stays gated until the best raw split path
misses the end-to-end target on filled shapes.

## 9. ATT-driven scan software pipeline

WaveScope attributed the standalone scan's largest waits to the immediate
HBM-to-LDS tile copies and their rendezvous. Two load schedules move those
dependencies off the exposed path without allocating a second LDS tile set:

- V is issued before the state mirror and Z product, then consumed by the
  residual after those operations have covered its latency.
- Chunk zero is peeled. Each later chunk's materialized tiles are issued while
  its predecessor computes, held in registers, and committed only after the
  closing rendezvous retires all reads of the old LDS contents. The next
  iteration's existing state-publish rendezvous also publishes those writes.

The narrower SA16 schedule delays its tile issue until V retires because its
larger per-thread load burst otherwise pushes the older V requests back onto
the critical path. Single-wave and C16 schedules retain immediate staging:
their prefetched vectors either cross the VGPR occupancy boundary or duplicate
too much tail traffic.

The same trace also exposed a repeated decay load. Every accumulator slot owned
by one lane has the same state column for both supported scan atoms, so one
decay value now scales the whole fragment.

As with the LDS ballast experiment, validate the mechanism before accepting the
timing: inspect VGPRs and scratch for both schedules, then recapture ATT. A
successful pipeline must reduce VMEM-wait attribution without introducing
scratch or dropping the intended resident-wave count. The remaining bottleneck
after that transition is LDS operand readiness and barrier balance; padding
sweeps are shape-sensitive and should not be promoted from one trace alone.
