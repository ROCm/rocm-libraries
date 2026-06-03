# gfx950 Deep Fusion rocprof Baseline Counters

Status snapshot from 2026-06-03 04:06.

This note replaces the static-estimate-only bottleneck story with hardware
counters captured by `rocprofv3` 1.1.0 on the best-performing kernel: the
default `mem`, `async_dma=False`, no-cache baseline. It answers the open item
from the prior records ("save ISA/resource reports and counter-based profiling")
and confirms with hardware which resource actually limits the kernel.

## What Was Measured

Kernel: `instances/gfx950/deep_fused_conv_pool.py`, baseline config.

```text
pool_tile=4x8  tile_m=128  tile_n=32  tile_k=16
warp_m=2  warp_n=1  block_size=128  pipeline=mem  async_dma=False
cache_input_footprint=False  direct_conv0_from_input_cache=False
```

Full target shape, verified correct in the same run:

```text
conv0 input:  [1, 2160, 3840, 8]
conv0 out:    [1, 2160, 3840, 32]
conv1 out:    [1, 2160, 3840, 24]
pool out:     [1, 1080, 1920, 24]
verify: max_abs_diff=0.00195312  bad_count=0/49766400  tol=1e-2
```

Wall-clock (separate 100-warmup / 100-iter bench run):

```text
mean = 0.3567 ms   useful = 142.9 TFLOP/s
```

Counters were collected with `rocprofv3 -i <pmc.txt>` over the harness
`--verify` path (one blocking dispatch, minimal build/warmup noise). Counter
groups were split across passes so each fits a single hardware collection pass.

## Launch / Resource Facts

```text
grid work-items   = 8,294,400   (64,800 CTAs x 128 threads)
waves (SQ_WAVES)  = 129,600     (2 waves per CTA)
VGPR              = 68
AGPR              = 0
SGPR              = 112
LDS bytes/block   = 26,112
```

Note: the actual VGPR count (68) is lower than the earlier static occupancy
estimate (100). Trust the measured value.

## Headline Counters (fused kernel dispatch)

```text
metric                  value      reading
----------------------  ---------  -------------------------------------------
MfmaUtil                 6.21 %    matrix engine almost idle
VALUBusy                47.67 %    VALU pipe is the busy unit
VALUUtilization        100.00 %    no lane divergence; VALU work is "real"
SALUBusy                 7.76 %
LdsBankConflict          2.21 %    secondary, not catastrophic
LdsLatency             125.31 cyc  high average LDS access latency
MemUnitStalled           0.061 %   HBM / global path is NOT stalled
SQ_WAIT_INST_LDS / ANY  52.08 %    over half of all wait time is LDS waits
```

## Instruction Mix

```text
class                       total          per CTA    per wave
--------------------------  -------------  ---------  --------
SQ_INSTS_VALU               111,456,000      1,720       860
SQ_INSTS_SALU                18,144,000        280       140
SQ_INSTS_LDS                 18,273,600        282       141
  SQ_INSTS_LDS_LOAD           6,609,600        102        51
  SQ_INSTS_LDS_STORE         11,664,000        180        90
SQ_INSTS_VMEM                 4,147,200         64        32
SQ_INSTS_VALU_MFMA_F16        1,814,400         28        14
```

Derived:

```text
VALU : MFMA instruction ratio = 61.4 : 1
LDS  store : load ratio       = 1.76 : 1
```

MFMA FLOP cross-check: `1,814,400 x (2*32*32*16) = 59.45 GFLOP`, which matches
the hardware-padded MFMA estimate in the roofline note exactly. The 32x32x16
fp16 atom selection is confirmed dynamically, not just statically.

## Interpretation

The hardware counters confirm and sharpen the prior hypotheses:

1. **Not MFMA-bound.** `MfmaUtil = 6.2 %`. The matrix engine is idle most of the
   time. There is no point chasing more/larger MFMA atoms for this shape.

2. **Not HBM-bound.** `MemUnitStalled = 0.061 %` and only 4.1M VMEM instructions.
   This is the decisive hardware confirmation of why input-footprint caching
   regressed: there is essentially no global-memory stall to recover.

3. **VALU-bound by instruction volume.** VALU outnumbers MFMA 61:1 and VALUBusy
   is ~48%. With `VALUUtilization = 100%`, this is non-divergent scalar work:
   coordinate/address arithmetic, per-element `select` masking, and f32->f16
   conversions in the scalar LDS fragment paths
   (`_masked_smem_frag_f16`, the cshuffle `coord_fn`, the maxpool gather, and the
   conv1 inner loop). This is exactly the "scalar coordinate arithmetic" the
   GEMM/conv reviews flagged, now quantified.

4. **LDS-wait dominant, store-heavy.** LDS waits are 52% of all wait time at a
   125-cycle average latency, and LDS **stores** outnumber loads 1.76:1 (180 vs
   102 per CTA). The two `_stage_accumulators_to_cshuffle_lds` materializations
   with `scalar_per_vector=1` are the prime suspects for the store volume.

## Direct-Footprint Correctness Regression (side finding)

While attempting a counter contrast, the opt-in `direct_conv0_from_input_cache`
path was found to **fail correctness at the full target shape**:

```text
direct-footprint, full shape:
verify: max_abs_diff=0.636963  bad_count=258625/49766400  tol=1e-2
```

Prior records described this path as "numerically correct but slower." That only
held at the small bring-up shapes; at the full `2160x3840` shape it is now
incorrect (~0.5% of outputs wrong). The path remains opt-in and is not on the
baseline, but the "correct but slow" claim should not be relied on for the full
shape until the footprint gather is reworked.

## Implication for Next Work

The counters back the operand-delivery direction over any further async or
input-cache effort, in priority order:

1. **Vectorize scalar LDS fragment reads.** `_masked_smem_frag_f16` (conv1
   operands) reads contiguous `c0_smem` / `w1_smem` columns one f16 at a time;
   replace with vector `ds_read`. Directly attacks both the VALU count and LDS
   wait. On the baseline path, so it is a baseline win, not opt-in.
2. **Drop dead masking.** For `K=32` the `valid_k` per-element `select` is always
   true; the mask is pure VALU overhead on the baseline shape.
3. **Cut cshuffle store volume.** The two staging passes with
   `scalar_per_vector=1` drive the 1.76:1 store:load imbalance; vectorize the
   stores and/or reduce to one staging handoff between conv0/conv1/pool.
4. Re-measure these same counters after each change; the success signal is
   MfmaUtil rising while VALU instruction count and `SQ_WAIT_INST_LDS` fall.

## Reproduce

```text
pmc groups (one per line, each a single collection pass):
  MfmaUtil VALUBusy VALUUtilization SALUBusy
  LdsBankConflict LdsLatency MemUnitStalled
  SQ_WAIT_INST_LDS SQ_WAIT_ANY SQ_WAVES
  SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_INSTS_SALU
  SQ_INSTS_LDS_LOAD SQ_INSTS_LDS_STORE SQ_INSTS_VALU_MFMA_F16

rocprofv3 -i pmc.txt -d <outdir> -o baseline -f csv -- \
  <venv>/python -m ck_dsl.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify \
  --verify --h 2160 --w 3840 --c 8 --k0 32 --k1 24
```
