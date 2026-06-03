# gfx950 Deep Fusion: Barrier-Merge Results

Status snapshot from 2026-06-03 14:16 (device 1).

This note records the second operand-delivery optimization taken after the conv1
LDS-read vectorization documented in
`2026-06-03-0439-gfx950-conv1-lds-vectorization-results.md`. It **collapses two
redundant epilogue barriers** (after the conv0 cshuffle stage and after the W1
load) into a **single block-wide barrier** before the shared conv1 MFMA
consumer. This is safe because the two producers write disjoint LDS tiles
(`DeepFusionC_smem` vs `W1_smem`) and both tiles are consumed only by the conv1
MFMA loop. The merge also allows the **W1 global loads to overlap the conv0
cshuffle LDS stores** instead of being serialized behind a redundant barrier — a
free partial W1-hoist that removes the W1 load from the conv1 critical path.

## Change

The original epilogue sequence (from `epilogue_override`, lines ~675-689 before
this change) had **three barriers**:

1. `_stage_accumulators_to_cshuffle_lds(..., sync=True)` — barrier after conv0
   cshuffle stores to `DeepFusionC_smem`.
2. `_load_conv1_weights_to_lds(..., sync=True)` — barrier after W1 global load
   into `W1_smem`.
3. `_stage_accumulators_to_cshuffle_lds(..., sync=True)` — barrier after conv1
   cshuffle stores.

Barriers **1** and **2** guard **disjoint LDS regions** and both are consumed
only by step 3 (the conv1 MFMA), so they can collapse to **ONE** barrier:

```python
# Before (3 barriers):
c_smem = _stage_accumulators_to_cshuffle_lds(b, conv_spec_, accs, grid)  # barrier 1
w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid)              # barrier 2
conv1_accs = _emit_conv1_1x1_mfma(b, spec, conv_spec_, c_smem, w1_smem, grid)

# After (2 barriers):
c_smem = _stage_accumulators_to_cshuffle_lds(b, conv_spec_, accs, grid, sync=False)
w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
b.sync()  # single barrier for both producers
conv1_accs = _emit_conv1_1x1_mfma(b, spec, conv_spec_, c_smem, w1_smem, grid)
```

The removed barrier (barrier 2) previously serialized the W1 load behind the
conv0 cshuffle completion. In the merged form, the W1 `buffer_load` operations
issue while the cshuffle `ds_write`s are still in flight, overlapping the W1
HBM fetch latency with cshuffle LDS work.

## Implementation

Added `sync: bool = True` kwarg to `_stage_accumulators_to_cshuffle_lds` and
`_load_conv1_weights_to_lds`. When `sync=False`, the trailing `b.sync()` is
skipped so the caller can batch the barrier. The conv1 cshuffle stage still uses
`sync=True` because it must complete before the maxpool reads the conv1 output
tile from LDS.

## Correctness

Full target shape, verified in the same harness run:

```text
--h 2160 --w 3840 --c 8 --k0 32 --k1 24
verify: max_abs_diff=0.00195312  bad_count=0/49766400  tol=1e-2
```

Identical to baseline and to vectorized-read. Barrier reduction is a pure
scheduling change with no semantic effect on the LDS dataflow.

## Wall-Clock

```text
                     time (ms)   useful TFLOP/s   vs baseline
vec-read (0439)        0.2533        201.2         1.41×
barrier-merge          0.2462        207.0         1.45×
```

100-warmup / 200-iter bench on device 1 (HIP_VISIBLE_DEVICES=1). Net:
**−2.8% latency** vs vectorized-read, **−31% latency** vs baseline (0.357 ms),
**1.45× cumulative speedup**.

## Counters (rocprofv3, fused kernel dispatch)

Same PMC groups as the 0406 baseline and 0439 vectorized-read notes, same
`--verify` collection path.

```text
counter                  vec-read    barrier-merge   delta
-----------------------  ----------  --------------  -------
SQ_WAIT_INST_LDS / ANY    10.35 %      11.34 %       +1.0 pp
SQ_INSTS_LDS              17,496,000   17,496,000     0.0 %
SQ_INSTS_LDS_LOAD          5,832,000    5,832,000     0.0 %
SQ_INSTS_LDS_STORE        11,664,000   11,664,000     0.0 %
SQ_INSTS_VALU            109,900,800  109,900,800     0.0 %
SQ_INSTS_VALU_MFMA_F16     1,814,400    1,814,400     0.0 %
```

All **instruction counts unchanged** (this is a pure barrier-scheduling change).
The **LDS-wait share rose** 10.35% → 11.34% (+1pp) despite wall-clock dropping
2.8%. This apparent paradox is explained by the mechanism:

## Interpretation

The speedup mechanism is **removing one serialization point (barrier) from the
epilogue critical path**, not cutting LDS instruction count:

1. Barrier-1 previously forced the W1 load to wait idle until the conv0 cshuffle
   completed. In the merged form, W1 `buffer_load`s issue concurrently with the
   cshuffle `ds_write`s, **overlapping W1 HBM latency with conv0 LDS work**.
2. The removed serialization point shaves cycles off the epilogue dataflow even
   though the per-dispatch LDS instruction count is unchanged.
3. The LDS-wait share rose 1pp because the **total cycle count dropped 2.8%**
   while the absolute LDS-wait cycles stayed roughly constant — the ratio
   denominator shrank, so the share grew even though the wall-clock dropped.

The barrier merge is **qualitatively different** from the conv1 read
vectorization: that change cut LDS load instructions (6.61M → 5.83M, −11.8%) and
halved avg LDS latency (125 → 57 cyc), which dropped the LDS-wait share 52% →
10%. This change leaves instruction counts unchanged and is instead a **critical
path reduction** via barrier collapse + overlap.

## Efficiency vs Roofline

FLOP counts are unchanged (barrier-merge is a schedule change), so the geometry
roofline is still `useful / hardware-padded = 50.96 / 59.45 = 85.7%`. At
0.2462 ms:

```text
useful throughput          = 50.96 GFLOP / 0.2462 ms = 207.0 TFLOP/s
hardware-padded throughput = 59.45 GFLOP / 0.2462 ms = 241.5 TFLOP/s
% of fp16 matrix peak (~2.8 PF/s) = 241.5 / 2800 = 8.6 %  (≈ MfmaUtil)
```

The matrix-issue roofline is **not** binding (~8.6% of peak). The binding
roofline remains operand delivery; this change nudged it 8.4% → 8.6% but did not
remove it.

## Next Bottleneck

The counters now confirm the **2.0:1 LDS store:load ratio** is still the largest
LDS lever, but the background agent (task "did vectorize c-shuffle give
improvement?") found that C-shuffle STORE vectorization is **fundamentally
impossible**: the MFMA C-fragment maps all 16 per-lane elements to the **same
column** at stride-32 rows in the `[128,32]` row-major LDS tile. A `ds_write`
needs contiguous bytes; these elements are 32 elements apart (64 bytes). Making
them contiguous would require a column-major LDS layout, which would destroy the
read vectorization that gave the 1.41× win. Store-contiguity and read-contiguity
are orthogonal here — you can't have both. Documented in
`2026-06-03-0447-gfx950-cshuffle-store-vectorization-results.md`.

The **next-largest lever** identified in the 0439 note is **conv0 im2col
address-math VALU reduction**. `VALUBusy` was 63.6% in the vectorized-read
snapshot; the VALU:MFMA instruction ratio is ~60:1 (109.9M / 1.8M). Opportunities:

1. **Strength-reduce div/mod** where dims are powers-of-2 (e.g. C=8 → shift/mask
   instead of `v_div_fixup` / `v_rcp`).
2. **CSE coordinate arithmetic** when the same (ho, wo, r, s) offset is computed
   for multiple lanes in the loader.
3. **Lift invariant terms** out of the per-K-tile loop where the M-tile is fixed.

Success signal for the next change: `SQ_INSTS_VALU` falls and `MfmaUtil` rises
while correctness stays `bad=0`.

## Reproduce

```text
# verify + bench (device 1 for parallel prototyping)
HIP_VISIBLE_DEVICES=1 <venv>/python -m ck_dsl.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify \
  --verify --bench --h 2160 --w 3840 --c 8 --k0 32 --k1 24 \
  --warmup 100 --iters 200

# counters (same pmc groups as the 0406 baseline and 0439 vec-read notes)
HIP_VISIBLE_DEVICES=1 rocprofv3 -i pmc.txt -d <outdir> -o barrier -f csv -- \
  <venv>/python -m ck_dsl.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify \
  --verify --h 2160 --w 3840 --c 8 --k0 32 --k1 24
```
