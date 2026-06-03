# gfx950 Deep Fusion: conv1 LDS-Read Vectorization Results

Status snapshot from 2026-06-03 04:39.

This note records the first operand-delivery optimization taken after the
counter baseline in `2026-06-03-0406-gfx950-rocprof-baseline-counters.md`. It
vectorizes the conv1 1x1 MFMA operand reads out of C-shuffle LDS and drops the
statically dead K-tail mask. It is a **baseline-path win** (not opt-in): the
default `mem`, `async_dma=False`, no-cache config.

## Change

`_emit_conv1_1x1_mfma` previously read each MFMA A/B operand fragment from LDS
with a per-element scalar gather (`_masked_smem_frag_f16`): `frag_len`
single-element `ds_read_b16`s plus a per-element `col < K` `select` mask.

The fragment is a contiguous `frag_len = a_per_lane = b_per_lane = 8` column run
of a plain row-major LDS tile (`LdsLayout.cshuffle` and the conv1 weight tile are
both `k_pad=0`, no swizzle). For the target topology the tiling covers K exactly
(`k_chunks * tile_k = 2 * 16 = 32 = K`), so the mask is statically dead. The read
is now a single wide `ds_read_b128` per fragment with no mask.

The logic was promoted to a reusable helper
`helpers/mfma_gemm_inner.load_smem_frag_contiguous_f16(b, smem, row, col_base,
frag_len, *, needs_mask, valid_k)`: wide vector `ds_read` when
`needs_mask=False and frag_len in {2,4,8}`, else the masked scalar fallback. The
caller owns `needs_mask` because whether the fragment crosses the K tail depends
on `col_base` (a runtime value) vs K, not on `frag_len` alone. The kernel computes
`needs_mask = (k_chunks * tile_k != K)`.

No swizzle was added. The baseline counter showed `LdsBankConflict = 2.21%`
(minor); the prediction was that bank conflicts were not the bottleneck and that
swizzling would only add address-permute VALU. The post-change conflict number
(0.60%, below) confirms this.

## Correctness

Full target shape, verified in the same harness run:

```text
--h 2160 --w 3840 --c 8 --k0 32 --k1 24
verify: max_abs_diff=0.00195312  bad_count=0/49766400  tol=1e-2
```

Identical to baseline. Correctness held both for the inline change and after
extracting the shared helper.

## Wall-Clock

```text
                 time (ms)   useful TFLOP/s
baseline (best)    0.3567        142.9
vectorized         0.2533        201.2
```

100-warmup / 200-iter bench, repeated runs stable at 0.2547-0.2553 ms before the
helper extraction and 0.2533 ms after. Net: **-28% latency, +40% useful
throughput, 1.41x speedup.**

## Counters (rocprofv3, fused kernel dispatch)

Same PMC groups as the baseline note, same `--verify` collection path.

```text
metric                  baseline    vectorized   reading
----------------------  ----------  -----------  -----------------------------------
MfmaUtil                  6.21 %       8.40 %    matrix unit less starved
SQ_WAIT_INST_LDS / ANY   52.08 %      10.35 %    LDS wait share collapsed (headline)
LdsLatency              125.31 cyc    57.02 cyc   avg LDS access latency halved
LdsBankConflict           2.21 %       0.60 %    conflicts fell; no swizzle needed
VALUBusy                 47.67 %      63.58 %    VALU now denser -> next bottleneck
VALUUtilization         100.00 %     100.00 %    still non-divergent
SALUBusy                  7.76 %      10.50 %
MemUnitStalled           0.061 %      0.079 %    still ~0, not HBM-bound
```

Instruction counts:

```text
counter                  baseline      vectorized    delta
-----------------------  ------------  ------------  -------
SQ_INSTS_VALU            111,456,000   109,900,800   -1.4 %
SQ_INSTS_LDS              18,273,600    17,496,000    -4.3 %
SQ_INSTS_LDS_LOAD         6,609,600     5,832,000    -11.8 %
SQ_INSTS_LDS_STORE       11,664,000    11,664,000     0.0 % (untouched)
SQ_INSTS_VMEM             4,147,200     4,147,200     0.0 %
SQ_INSTS_VALU_MFMA_F16    1,814,400     1,814,400     0.0 % (same MFMA work)
```

Resources unchanged: VGPR 68, AGPR 0, SGPR 112, LDS 26,112 B/block, grid
8,294,400 work-items, SQ_WAVES 129,600.

`SQ_WAIT_INST_LDS / SQ_WAIT_ANY = 23,796,657 / 230,008,690 = 10.35 %`.

## Interpretation

The speedup mechanism is **eliminating serialized LDS-load waits**, not cutting
VALU count:

1. Replacing 8 dependent scalar `ds_read_b16`s (each ~125 cyc) with one
   `ds_read_b128` per fragment cut LDS load instructions 11.8% and **halved
   average LDS latency (125 -> 57 cyc)**.
2. That dropped the LDS-wait share of all wait from **52% to 10%** — the single
   biggest counter move.
3. `MfmaUtil` rose 6.2% -> 8.4%, matching the 8.3% forecast from the
   hardware-padded throughput / matrix-peak ratio (peak ~2.8 PF/s fp16, from
   `examples/gfx950/gemm_perf_square_warpspec/README.md`).
4. The dead-mask removal contributed only ~1.4% of the VALU count; the win is LDS
   latency/issue, not VALU.

## Efficiency vs Roofline

FLOP counts are unchanged (vectorization is a schedule change), so the geometry
"padding tax" roofline is fixed: `useful / hardware-padded = 50.96 / 59.45 =
85.7%` (from the roofline note). At 0.2533 ms:

```text
useful throughput          = 50.96 GFLOP / 0.2533 ms = 201.2 TFLOP/s
hardware-padded throughput = 59.45 GFLOP / 0.2533 ms = 234.7 TFLOP/s
% of fp16 matrix peak (~2.8 PF/s) = 234.7 / 2800 = 8.4 %  (== MfmaUtil)
```

The matrix-issue roofline is **not** binding (~8% of peak). The binding roofline
remains operand delivery; this change moved it closer but did not remove it.

## Next Bottleneck

The counters now point at two follow-ups, both flagged in the baseline note:

1. **Vectorize the C-shuffle stores.** With loads down, the LDS store:load ratio
   worsened to **2.0:1** (11,664,000 stores vs 5,832,000 loads). Stores are still
   `scalar_per_vector=1` in `_stage_accumulators_to_cshuffle_lds`. This is the
   next-largest LDS lever.
2. **Cut conv0 im2col address-math VALU.** `VALUBusy` rose to 63.6%; VALU is now
   the gating unit. The remaining cost is conv0 coordinate/address arithmetic.

Success signal for the next change is unchanged: `SQ_INSTS_LDS_STORE` falls and
`MfmaUtil` rises while correctness stays `bad=0`.

## Reproduce

```text
# verify + bench
<venv>/python -m ck_dsl.examples.gfx950.deep_fused_conv_pool_verify \
  --verify --bench --h 2160 --w 3840 --c 8 --k0 32 --k1 24 \
  --warmup 100 --iters 200

# counters (same pmc groups as the 0406 baseline note)
rocprofv3 -i pmc.txt -d <outdir> -o vec -f csv -- \
  <venv>/python -m ck_dsl.examples.gfx950.deep_fused_conv_pool_verify \
  --verify --h 2160 --w 3840 --c 8 --k0 32 --k1 24
```
