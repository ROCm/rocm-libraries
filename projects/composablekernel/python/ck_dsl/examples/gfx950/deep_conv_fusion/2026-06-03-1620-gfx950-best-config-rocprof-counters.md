# gfx950 Deep Fusion rocprof Counters — Best Config (4x4 / tk32)

Status snapshot from 2026-06-03 16:20.

This note re-captures `rocprofv3` 1.1.0 hardware counters on the **current best**
config, superseding the `2026-06-03-0406` baseline counters which profiled the
older `pool_tile=4x8 / tile_m=128 / tile_k=16` kernel. Captured via
`ck_dsl.examples.gfx950.deep_conv_fusion.profile_best_config` (single verified blocking dispatch),
same pmc groups as the prior note.

## Config Measured

```text
pool_tile=4x4  tile_m=64  tile_n=32  tile_k=32
warp_m=2  warp_n=1  block_size=128  pipeline=mem  async_dma=False
cache_input_footprint=False  direct_conv0_from_input_cache=False
verify: max_abs_diff=0.00195312  bad=0/49766400
grid = (1, 270, 480) = 129,600 CTAs x 128 threads
```

## Resource Facts

```text
metric            best 4x4/tk32     prior 4x8/tk16
----------------  ----------------  --------------
VGPR              44                68
AGPR              0                 0
SGPR              112               112
LDS bytes/block   17,920            26,112
```

Lower VGPR + lower LDS = higher achievable occupancy than the prior baseline.

## Headline Counters

```text
metric                  best 4x4/tk32   prior 4x8/tk16   reading
----------------------  --------------  ---------------  -----------------------
MfmaUtil                 11.1 %          6.2 %           matrix engine still idle
VALUBusy                 63.1 %          47.7 %          VALU pipe busier than ever
LdsBankConflict           0.62 %         2.21 %          improved
LdsLatency               66.9 cyc       125.3 cyc        halved
MemUnitStalled            1.19 %         0.06 %          HBM still not the limiter
SQ_WAIT_INST_LDS / ANY   21.1 %         52.1 %          LDS wait no longer dominates
```

## Instruction Mix (whole dispatch)

```text
class                    best 4x4/tk32   prior 4x8/tk16
-----------------------  --------------  --------------
SQ_INSTS_VALU            89,400,984      111,456,000
SQ_INSTS_SALU            15,088,723       18,144,000
SQ_INSTS_LDS             18,144,000       18,273,600
  SQ_INSTS_LDS_LOAD       7,257,600        6,609,600
  SQ_INSTS_LDS_STORE     10,886,400       11,664,000
SQ_INSTS_VMEM             3,466,814        4,147,200
SQ_INSTS_VALU_MFMA_F16    2,073,600        1,814,400
```

Derived:

```text
metric                  best 4x4/tk32   prior 4x8/tk16
----------------------  --------------  --------------
VALU : MFMA             43.1 : 1        61.4 : 1
LDS store : load        1.50 : 1        1.76 : 1
VALU per CTA            690             1,720
```

## The Counterintuitive Win

The best config executes **more** padded MFMA work yet runs faster:

```text
                        best 4x4/tk32   prior 4x8/tk16
hardware MFMA FLOPs     67.95 GFLOP     59.45 GFLOP
useful FLOPs            50.96 GFLOP     50.96 GFLOP
useful / hardware       75.0 %          85.7 %
wall clock              ~0.223 ms       ~0.357 ms
useful throughput       ~228 TFLOP/s    ~143 TFLOP/s
```

MFMA FLOP cross-check: `2,073,600 x (2*32*32*16) = 67.95 GFLOP`, matches the
`tile_m=64, tile_k=32` rectangular-tile estimate (conv0 K pad 72->96, 129,600
CTAs).

The kernel pays a *worse* padding tax (tk32 rounds K_gemm 72->96 vs tk16's
72->80, and there are 2x more CTAs) but still wins by a wide margin. The reason
is entirely on the overhead side: **total VALU dropped 20% (111M -> 89M), VALU
per CTA dropped 2.5x (1720 -> 690), LDS latency halved, and VGPR fell 68 -> 44.**
This is decisive confirmation the kernel is overhead-bound, not compute-bound:
adding MFMA work is free if it buys a reduction in VALU / LDS / occupancy
pressure.

## Current Bottleneck Read

Still VALU-bound, now harder so:

1. **VALUBusy rose to 63%** even as total VALU fell — the VALU pipe is now the
   near-exclusive limiter (MfmaUtil 11%, MemUnitStalled ~1%, LDS wait only 21%).
2. The big tk16->tk32 VALU win is already banked. What remains (89M VALU,
   43:1 vs MFMA) is the irreducible-looking coordinate/convert/select math plus
   the cshuffle staging stores.
3. **Store-side LDS still imbalanced 1.50:1** (10.9M stores). The two
   `_stage_accumulators_to_cshuffle_lds` passes with `scalar_per_vector=1` remain
   the prime store-volume suspect. Store *width* vectorization was already proven
   impossible (MFMA C-fragment geometry: same-column / different-row, stride-32),
   so the only remaining store lever is reducing the *number* of staging
   handoffs, not their width.

## Next Work, In Priority Order

1. **Attribute the residual 89M VALU by region.** The roofline note still lists
   ISA/disassembly as the missing evidence. Disassemble the best-config hsaco and
   bucket `v_*` ops (address math vs f32<->f16 convert vs `v_cndmask` select) to
   decide whether to chase the cshuffle, the maxpool gather, or the conv1 inner
   loop. Optimizing before this risks polishing the wrong region.
2. **Collapse the two cshuffle staging handoffs into one** (conv0->conv1 and
   conv1->pool). Biggest remaining structural lever on the baseline path; attacks
   the 1.50:1 store imbalance directly.
3. **Stage-isolation timing** (conv0-only, conv0+conv1, full) to confirm which
   stage owns the VALU before investing in (2).

## ISA Attribution — VALU Hotspots by Pipeline Stage

Disassembled the best-config hsaco (`llvm-objdump --mcpu=gfx950`) and segmented
by `s_barrier` boundaries, which align with the fused pipeline stages. Static
VALU counts (≈2x dynamic per CTA; the kernel has no deep hot loop — dynamic/static
≈ 690/343 ≈ 2.0 — so the once-per-CTA epilogue stages contribute as much VALU as
the K-loop):

```text
stage (ISA segment)                    VALU   notable ops          maps to
-------------------------------------  -----  -------------------  ---------------------------
seg0  conv0 load + coord prologue       78    magic-div im2col     A/B address math (once/CTA)
seg1-5 conv0 K-loop bodies            ~30ea   ds_read_b128 x4      conv0 load+MFMA (3 K-tiles)
seg6  conv0 ReLU + cshuffle stage       65    fmax=32, ds_w=17     acc_epilogue relu + staging
seg7  conv1 MFMA + ReLU + cshuffle      77    fmax=32, ds_w=16     conv1_epilogue relu + staging
seg8  maxpool gather + reduce + store   51    ds_r=12(u16), cvt=15  scalar pool gather
```

Key reads:

1. **All 64 `v_max_f32` are ReLU, not the pool.** seg6 (32) = conv0 `acc_epilogue`
   relu, seg7 (32) = conv1 `conv1_epilogue` relu, applied per owned accumulator
   element (c_per_lane=16 x 2 MFMAs). The maxpool's own reduction is only 6
   `v_max3_f32` (seg8). ReLU is the single largest VALU opcode in the kernel.

2. **The two cshuffle staging stores are scalar 16-bit** (seg6 `ds_write_b16`x..,
   seg7 16 stores) — the store-width lock confirmed at the ISA level
   (C-fragment geometry), while the operand *reads* are already vectorized
   (`ds_read_b128`).

3. **The maxpool gather is scalar** (seg8: 12 `ds_read_u16` n=1 reads + 15 `cvt`)
   — the un-vectorized window read flagged in the roofline note, now located.

4. **seg0 (78 VALU) is the im2col coordinate prologue** — already lowered to
   magic-division (`v_mul_hi_u32` + `v_mad_i32_i24` chains, constants 0x11111112
   etc. visible). Near codegen-optimal; little left to cut without compile-time
   shape folding, which magic-div already approximates.

### Cuttable VALU, evidence-ranked

```text
lever                                            targets        correctness
-----------------------------------------------  -------------  ---------------------------
A. Move conv1 ReLU AFTER the pool reduction       ~24 fmax      exact: relu(max)=max(relu)
   (relu pooled outputs, not conv1 accs;          (seg7->seg8   conv1 out is pool-only
    pool shrinks count 4x: 32 -> ~8)               net -24)      consumed
B. Vectorize the maxpool gather (n=1 -> wider)    12 ds_r_u16   exact
                                                  + some cvt
C. Collapse conv0/conv1 cshuffle handoffs (1)     ~16 ds_w      structural; needs care
D. conv0 ReLU                                     0             cannot move past conv1 MFMA;
                                                                 every elem needs activation
```

Lever A is the cleanest correctness-preserving baseline win: ReLU is monotonic
non-decreasing, so `relu(max(a,b,c,d)) = max(relu(a),relu(b),relu(c),relu(d))`,
and conv1 output is consumed only by the pool — so relu can move from the conv1
accumulator epilogue (32 elems/thread) to the pooled result (≈3/thread), a ~4x
cut on those fmax ops. Expect a few percent, not a step change (VALU is broad,
not concentrated in one removable hotspot).

### Lever A — implemented (2026-06-03 16:40)

Deferred the *entire* conv1 epilogue past the maxpool (bias/scale>=0/relu/clamp
all commute with max). `_epilogue_is_pool_deferrable` gates it; the conv1 MFMA
returns raw accs and `_emit_inline_maxpool_from_cshuffle` applies the epilogue to
each pooled fp32 result. conv1 output is pool-only consumed, so the result is
bit-stable (`max_abs_diff=0.00195312, bad=0/49766400`, unchanged).

```text
metric                   before         after          delta
-----------------------  -------------  -------------  ------
static v_max_f32         64             35             -29
static VALU              335            306            -8.7 %
wall clock (300-iter)    ~0.223 ms      ~0.219 ms      -1.8 %
useful throughput        ~228 TFLOP/s   ~233 TFLOP/s   +2.2 %
```

The −29 static `v_max_f32` is exactly the 32 per-element conv1 ReLU collapsing to
3 per-thread pooled ReLU. Matches the "few percent, not a step change"
prediction — VALU is broad. Remaining levers B (vectorize maxpool gather) and C
(collapse cshuffle handoffs) are still open.

### Lever B — implemented (2026-06-03 17:10)

Re-tiled `_emit_inline_maxpool_from_cshuffle` by `(window, k-block)` instead of
flat element index. The 2x2 maxpool corner rows depend only on the pooled window,
**not** on the channel `k`, so each thread now owns a contiguous `kvec`-wide run
of channels (kvec=4 for out_k=24, the largest width dividing out_k that keeps
≥half the block active). The window decode and the 4 corner-address computations
are computed once per kvec channels, and the per-channel scalar reads fold into
one wide read per corner. Bit-stable (`max_abs_diff=0.00195312, bad=0/49766400`).

The backend then did better than the source: the 4 corner `ds_read` (b64) merged
into **2 `ds_read2_b64`**, and the deferred ReLU `max(x,0)` fused into the
`v_max3_f32` reduction tree (`v_max3_f32 v2, v12, v7, 0`).

```text
seg8 (maxpool) metric    lever A        lever B        delta
-----------------------  -------------  -------------  ------
ds_read                  12x ds_read_u16  2x ds_read2_b64  16-bit -> 64-bit, fully vectorized
seg8 VALU                ~51            44             -7
wall clock (200-iter)    ~0.219 ms      ~0.218 ms      flat
useful throughput        ~233 TFLOP/s   ~234 TFLOP/s   flat
```

Wall-clock is flat: seg8 is a cold once-per-CTA stage and its reads were never
the limiter (MemUnitStalled ~1%, SQ_WAIT_INST_LDS 21%). The value is structural —
the un-vectorized gather flagged in the roofline note is closed, and the read
side of the kernel is now wide everywhere.

## Hotspot Map (lever-B binary, static per-thread, segmented by s_barrier)

```text
seg  VALU  MFMA  dsR  dsR-bits  dsW  dsW-bits  maps to
---  ----  ----  ---  --------  ---  --------  -----------------------------------
 0    78    0     0       0      3     384     conv0 im2col coord prologue (magic-div)
 1    29    1     4     512      0       0     conv0 K-tile 0 (ds_read_b128 x4)
 2    29    1     0       0      3     384     conv0 A-tile staging
 3     0    0     4     512      0       0     conv0 K-tile load
 4     4    1     0       0      3     384     conv0 K-tile
 5     3    2     4     512      0       0     conv0 K-loop tail
 6    63    1     0       0     17     384     conv0 ReLU (v_max_f32 x32) + cshuffle stage
 7    37    2     4     512     16     256     conv1 MFMA + cshuffle stage
 8    44    0     2     128      0       0     maxpool gather + reduce + store
TOT  287    8    18    2176     42    1792
```

Reads:

- **VALU hotspot = seg0 (78) + seg6 (63).** seg0 is the im2col coordinate prologue
  (v_add=15, v_lshrrev=11, v_mul_hi_u32=5, v_mad_i32_i24=5) — already magic-div
  lowered, near codegen-optimal. seg6's 32 `v_max_f32` are the conv0 ReLU, which
  **cannot** move (it feeds the conv1 MFMA — lever D, uncuttable). The two largest
  VALU blocks are both at/near their floor.
- **LDS-write hotspot = seg6 + seg7 = 33 of 42 ds_writes**, the two cshuffle
  staging handoffs (conv0->conv1 and conv1->pool). Mostly narrow `ds_write_b16`:
  store width is locked by the MFMA C-fragment geometry, so the only lever is
  **fewer handoffs, not wider ones** → this is exactly **lever C** (collapse the
  two handoffs into one). It also owns the staging VALU (32 `v_accvgpr_read_b32`
  + 16 `v_cvt_pk_f16_f32` across seg6/seg7).
- **LDS reads are healthy/wide everywhere.** conv0 K-loop reads are `ds_read_b128`
  (512 bits/segment); the maxpool gather is now `ds_read2_b64`. No scalar LDS
  reads remain.

**Verdict:** the two cuttable VALU hotspots (seg0 im2col, seg6 conv0 ReLU) are at
their codegen/algorithmic floor. The remaining structural lever is **C** — it is
simultaneously the LDS-write hotspot (33/42 writes) and a big chunk of the
staging VALU (~48 ops). Levers A and B are banked; C is the next move.

### Lever C (handoff #2 in-register) — implemented (2026-06-03 18:10)

Eliminated the **conv1->maxpool cshuffle handoff entirely** by keeping the conv1
output register-resident and reducing the pool intra-lane. The insight: with the
single 32x32 MFMA atom (`warp_n=1`, `mfmas_per_warp=1`), each lane's vec<16>
accumulator tiles a **4x4 conv-spatial block for one channel** (`channel =
lane%32`), which for a 2x2 stride-2 pool is exactly **2x2 = 4 pool windows whose
four corners all live in the same lane**. No cross-lane shuffle is needed — the
maxpool is purely `v_max_f32` over four `vec_extract` slots per window.

Slot decomposition (derived & ISA-verified from `_mfma_acc_32x32`):

```text
local_conv_h = warp_m_idx*4 + i//4     (warp_m=2 -> 8 conv rows = pool rows 0..3)
local_conv_w = (lane//32)*4 + i%4      (conv_tile_w=8 -> 8 conv cols)
channel      = lane%32
window (pho_l,pwo_l) -> slots: (0,0)={0,1,4,5} (0,1)={2,3,6,7}
                               (1,0)={8,9,12,13} (1,1)={10,11,14,15}
```

Gated by `_maxpool_is_intra_lane(spec, grid)` (32x32 atom, warp_m=2, warp_n=1,
single MFMA, 2x2 pool, conv_tile 8x8, tile_m=64); any other geometry falls back
to the LDS gather path. Bit-stable (`max_abs_diff=0.00195312, bad=0/49766400`).

```text
metric                   lever B        lever C        delta
-----------------------  -------------  -------------  ------
ds_write (whole kernel)  42             26             -16 (conv1 cshuffle stage)
ds_read  (whole kernel)  18             16             -2  (maxpool gather)
s_barrier                8              7              -1  (staging barrier)
wall clock (200-iter)    ~0.218 ms      ~0.184 ms      -15.6 %
useful throughput        ~234 TFLOP/s   ~277 TFLOP/s   +18 %
```

This is the step change the broad-VALU reads did *not* predict: removing the
handoff deletes not just the 16 `ds_write_b16` and the gather, but the **barrier
between conv1 and the pool**, so the once-per-CTA conv1 epilogue + pool now flow
without a block-wide stall. The 33/42 LDS-write hotspot is now just the conv0->
conv1 handoff (#1), which is an inherent M<->K0 transpose and not cheaply
eliminable. Levers A, B, C all banked.

## Reproduce

```text
single-dispatch target:
  ck_dsl/examples/gfx950/deep_conv_fusion/profile_best_config.py

HIP_VISIBLE_DEVICES=1 rocprofv3 -i pmc.txt -d <outdir> -o best -f csv -- \
  <venv>/python -m ck_dsl.examples.gfx950.deep_conv_fusion.profile_best_config

pmc groups: same as 2026-06-03-0406 note.
raw csv: .rocprofv3/best_4x4_tk32_20260603-161920/pmc_*/best_counter_collection.csv
```
