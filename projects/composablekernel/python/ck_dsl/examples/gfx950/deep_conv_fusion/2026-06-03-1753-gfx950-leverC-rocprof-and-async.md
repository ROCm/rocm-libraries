# gfx950 Deep Fusion — Lever-C rocprof Re-capture + Async Probe

Status snapshot 2026-06-03 17:53.

Re-captured `rocprofv3` 1.1.0 counters on the **lever-C** kernel (conv1->maxpool
handoff eliminated, register-resident pool) and probed whether async global
loads can push past the VALU wall. Same pmc groups and single-dispatch target
(`ck_dsl.examples.gfx950.deep_conv_fusion.profile_best_config`) as the `2026-06-03-1620` note, so
columns are directly comparable.

Config: `pool_tile=4x4 tile_m=64 tile_n=32 tile_k=32 warp=2x1 mem async=False`,
grid (1,270,480)=129,600 CTAs x 128 threads. verify `max_abs_diff=0.00195312`.

## Resource Facts

```text
metric            lever C        lever A (prev best)
----------------  -------------  -------------------
VGPR              40             44
AGPR              0              0
SGPR              112            112
LDS bytes/block   13,824         17,920
```

LDS dropped 4,096 B = exactly the removed conv1 cshuffle tile
(`tile_m*tile_n*2 = 64*32*2`). The second staging LDS allocation is gone.

## Headline Counters

```text
metric                  lever C        lever A        reading
----------------------  -------------  -------------  -----------------------
MfmaUtil                 14.1 %         11.1 %        same MFMA, less wall -> higher util
VALUBusy                 62.9 %         63.1 %        UNCHANGED -- still the limiter
LdsBankConflict           0.68 %         0.62 %       flat
LdsLatency               72.4 cyc       66.9 cyc      flat-ish
MemUnitStalled            0.18 %         1.19 %       HBM idle -- not memory bound
SQ_WAIT_INST_LDS / ANY   16.0 %         21.1 %        LDS wait dropped further
```

## Instruction Mix (whole dispatch)

```text
class                    lever C        lever A        delta
-----------------------  -------------  -------------  ------
SQ_INSTS_VALU            74,131,200     89,400,984     -17.1 %
SQ_INSTS_SALU            12,182,400     15,088,723     -19.3 %
SQ_INSTS_LDS             10,886,400     18,144,000     -40.0 %
  SQ_INSTS_LDS_LOAD       4,147,200      7,257,600     -42.9 %
  SQ_INSTS_LDS_STORE      6,739,200     10,886,400     -38.1 %
SQ_INSTS_VMEM             3,628,800      3,466,814     +4.7 %
SQ_INSTS_VALU_MFMA_F16    2,073,600      2,073,600     0
```

Derived:

```text
metric            lever C   lever A
----------------  --------  --------
VALU : MFMA       35.7 : 1  43.1 : 1
VALU per CTA      572       690
wall clock        0.184 ms  ~0.219 ms
useful throughput 277 TF/s  ~233 TF/s
```

## Is VALU still the bottleneck? Yes.

Lever C removed real work — VALU −17%, LDS −40%, VGPR 44->40, LDS −4 KB — yet
**VALUBusy is flat at ~63%**. The pipe occupancy did not move because we removed
work *and* wall-clock proportionally, so the limiter is unchanged: VALU is still
the near-exclusive bottleneck (MfmaUtil 14%, MemUnitStalled 0.18%, LDS wait 16%).
The residual 74M VALU is the irreducible region the ISA attribution already
located: the im2col coordinate prologue (seg0, magic-div, codegen-optimal) +
conv0 ReLU (seg6, immovable — feeds the conv1 MFMA) + the f32<->f16 converts.

## Async Global Loads — Probed, Does Not Help

Ran the best config with `async_dma=True` vs the `False` baseline:

```text
config                  verify              wall clock   useful TFLOP/s
----------------------  ------------------  -----------  --------------
4x4 tk32 mem  (sync)    PASS                0.184 ms     276.5
4x4 tk32 mem  (async)   FAIL (3.26 diff)    0.252 ms     202.2  (-27 %)
```

Two independent reasons async is a dead end here:

1. **Correctness:** the async path miscompiles for this fused conv0->cshuffle->
   conv1->pool dataflow (47M/49.8M elements wrong). It is not wired for the
   double-cshuffle epilogue, so it would need real work just to be correct.
2. **Even if correct, there is nothing to hide.** Async DMA pays off when the
   kernel is memory-latency bound — it overlaps global-load latency with
   compute. Here `MemUnitStalled = 0.18 %` and `SQ_WAIT_INST_LDS = 16 %`: the
   memory pipes are already idle. The kernel waits on the **VALU** pipe (63 %),
   which async does not touch. The measured −27 % is the async machinery's
   overhead with no latency to amortize against it.

**Conclusion:** async loads will not push this further. The kernel is
overhead/VALU-bound, not memory-bound. The only remaining levers are ones that
cut VALU itself, and the two largest VALU blocks (im2col coord, conv0 ReLU) are
already at their codegen/algorithmic floor — so further gains here are likely
small. The bigger structural question (handoff #1 conv0->conv1, an inherent
M<->K0 transpose) remains the only large untouched LDS-write region, but it is
not VALU and not cheaply eliminable.

## Reproduce

```text
HIP_VISIBLE_DEVICES=1 rocprofv3 -i pmc.txt -d <outdir> -o leverC -f csv -- \
  python3 -m ck_dsl.examples.gfx950.deep_conv_fusion.profile_best_config

async probe:
  run_config from ck_dsl.examples.gfx950.deep_conv_fusion.compare_pool_tile_configs
  make_spec(4,4,32,32,2,1, async_dma=True)

raw csv: .rocprofv3/best_4x4_tk32_leverC_20260603-174656/pmc_*/leverC_counter_collection.csv
```
