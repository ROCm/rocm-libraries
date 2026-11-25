# The tier-0 score

The cycles per main-loop trip, per wave, that tier 0 ranks candidates by. Code:
`search._tier0`, `perf.orderCore`, `perf.sideCost`.

```text
score = max(C_wmma, C_lds) + exposed_A + exposed_B + tdm + tdmExposed + tripOverhead
```

| Term | How it is computed |
|---|---|
| `C_wmma` | WMMAs per trip × `wmmaCycles` (8 for MXF8 16×16×128) |
| `C_lds` | ds_loads per trip × `dsCyclesPerLoad` (6) |
| `exposed_A`, `exposed_B` | the ds_load latency each side leaves exposed: [exposed-ds-latency.md](exposed-ds-latency.md) |
| `tdm` | tensor loads per trip (`perf.tensorLoads`) × `tdmIssueCycles` (40) |
| `tdmExposed` | `perf.tdmExposed`: the larger of the copies rule, `tdmLatency` (800) for a one-copy body, else `max(0, tdmLatency − max(C_wmma, C_lds))`, and the [legal-window](#the-legal-tensor-load-window) shortfall |
| `tripOverhead` | 17: loop branch, barrier signal/wait, waitcnts |

`C_wmma` and `C_lds` depend on the loop order only; the register choice changes the exposure,
the steady-body copies and the VGPRs.

- **ds_loads per trip** (`orderCore`): per side, resident tiles × ds_loads per tile, plus
  resident tiles / 2 for the MX scales.
- **copies** is the LCM of the two sides' copies; a side's copies are `VA / gcd(VA, positions)`,
  or 1 under CLR.

Ties go to fewer units loaded per trip (each pays the latency), then fewer copies, then fewer
VGPRs, then the canonical order.

## Worked example: the tuned MXF8 kernel

`LoopOrder` `KKNMNM`, one trip `[Ki 2, No 2, Mo 2, Ni 5, Mi 4]`:

- `C_wmma` = 160 WMMAs × 8 = 1280.
- `C_lds` = 162 loads × 6 = 972: A 16 tiles × 4 plus B 20 tiles × 4 is 144 loads, plus 8 + 10 = 18
  scale loads.
- `exposed_A` = `exposed_B` = 0 (A PLR 2 / VA 3, B PLR 1 / VA 2; see the exposure note).
- `tdm` = 2 tensor loads (A with MXSA, B with MXSB) × 40 = 80.
- `tdmExposed` = 0: the body has 3 copies, and one 1280-cycle trip covers the 800. The legal
  windows, 180 WMMAs for A and 200 for B less their 127- and 135-cycle read returns, cover the
  1267 cycles 576 rows take to land.

```text
score = max(1280, 972) + 0 + 0 + 80 + 0 + 17 = 1377
```

The trip is WMMA-bound, and the only register choices that change the score are the ones that
expose latency; A with PLR 1 / VA 2 exposes 560 and scores 1937.  A with CLR 1 / VA 4 exposes
nothing but makes a one-copy body, so it scores 1377 + 800 = 2177.

## The legal tensor-load window

In an LDS ring of `NumLdsBlk` buffers (`LDSBufferA/B` when set; `perf.ldsDepth`) the tensor
load of K chunk c overwrites chunk c − depth. `NestWalk.tdmWindow` finds both ends from the nest
walk and the side's PLR and ring:

- **The earliest legal issue:** once the last ds_load of chunk c − depth has returned and every
  wave has passed the barrier (write-after-read). The return is StinkyTofu's drain of the
  side's last unit (`perf.drainLatency`, its `computeDynamicDrainLatency`): the load's latency,
  one issue slot per wave (clamped to 1..4) for each load ahead in the 16-deep return queue, past
  the queue `waves / dsThroughput` a load, capped at `dsMaxDrain` (ds_load_b128: 56, 2 a cycle,
  255). Then `barrierSignalToWait` (11). The tuned kernel's A returns in 127 and B in 135.
- **The latest legal need:** the first ds_load of chunk c (read-after-write).
- **The latency between:** the hardware's issue-to-landing time, `tdmRowCycles` (2.2) per
  macro-tile row the trip's tensor loads move (`perf.tdmLandingLatency`; 1267 for 256 + 320
  rows). The copies rule keeps the calibrated `tdmLatency`, timed from GIR's load position.

PLR moves both ends alike, since it pulls a chunk's last read earlier as well as its first, so it
does not shorten the window. On the tuned nest A has 220 WMMAs at PLR 0, 4 and 8, and 180 at
PLR 2 in a three-buffer ring, whose refills wait for their slots. Each LDS buffer adds a trip.
Which trip issues the load (`PrefetchGlobalRead`) is the schedule's choice and does not bound
the window.

Positions are WMMAs, priced at the trip's cycles per position, `max(C_wmma, C_lds) / WMMAs`.
`perf.tdmWindowExposed` runs both operands' loads, issued and needed at those edges, through one
in-order clock. A load that lands during the other's wait costs nothing more. A side choice's
window counts toward pruning only where its shortfall exceeds the least the copies rule charges.

This is the most cover the parameters allow, not the cover the schedule gives. StinkyTofu can
only shorten it: in the MXF8 8×10 top 50 it took 14 kernels below the latency the parameters
cover (see [schedule-vs-model.md](schedule-vs-model.md)). `tdmExposed` charges the larger of
this and the copies rule.

### Measured on gfx1250 (2026-11-24)

**Landing latency follows rows, not bytes.** ATT of 13 kernels (release of a stalled
`s_wait_tensorcnt` minus the issue of the load it retires, steady loop, one card):

| Kernel | Rows a trip | Bytes a trip | Landing (median) |
|---|---|---|---|
| MXF8 / BF16 128×128 | 256 | 66 / 32 KB | 590–628 |
| BF16 256×256 | 512 | 64 KB | 985 |
| F8 K128 / MXF8 K256 256×320 | 576 | 72 / 148 KB | 1207–1337 |

Twice the bytes in the same rows land in the same time. One workgroup of 4 waves runs on each
CU in all of them, and 64 to 5120 workgroups (problem sizes 1024×1280 to 8192×10240) move it by
under 40 cycles, so it is not occupancy or contention across the GPU. A fit through the origin
is within 65 cycles of all but BF16 256×256 (−146).

**The bound holds where it charges.** Only F8 16×16×64 8×10 has a window shorter than its
landing: a two-buffer ring gives one 857-cycle trip against 1267, and tier 0 now charges its
fastest kernels about 400 (857 → 1259–1267). The hardware agrees: those kernels run 1093–1165
cycles a trip, and `s_wait_tensorcnt` plus the barrier waves queue behind it stall 370–456 a
trip. The variants the window does not charge stall about 3 cycles a trip on `tensorcnt`
(BF16 8×8, MXF8 4×4).

Against the measured kernels, with all register choices scored (no pruning):

| Variant | Changed | Largest increase | Best score | Top-50 overlap | Spearman before → after (pick) |
|---|---|---|---|---|---|
| MXF8 16×16×128 8×10 | 83836 of 112590 | +108 | 1377 | 15 | +0.27 → +0.26 (fastest both) |
| F8 16×16×128 8×10 | 129078 of 170317 | +108 | 1337 | 37 | not re-measured |
| MXF8 4×4 | 0 | 0 | 897 | 50 | −0.08 → −0.08 |
| BF16 4×4 | 0 | 0 | 857 | 50 | +0.92 → +0.92 |
| F8 16×16×64 8×10 | 737703 of 813778 | +1224 | 857 → 1127 | 9 | +0.53 → +0.36 (100% → 97%) |
| BF16 8×8 | 1068726 of 1210044 | +299 | 1081 | 50 | +0.80 → +0.81 |

(MXF8 8×10 against the 50 kernels of the ATT study; the rest against 20–32 kernels each,
measured twice.) On F8 K64 every candidate starves, and the 14 fastest score within 8 cycles of
each other (1259 or 1267) while measuring within 4%: the window is right about the stall but
its 8-cycle differences only reorder near-ties. The first version of the window, timed with
`tdmLatency` (800) and a 300-cycle read return, overcharged the 4×4 tiles, whose loads land in
about 600; it took BF16 4×4 from +0.92 to +0.17. The table in the next section predates the
window.

Not modelled: the MX scales a fused load (`TDMFuse`) carries, whose reads tier 0 does not
place; and split loads (`TDMSplitA/B`), which can only widen a region's window.

## Why a one-copy body pays the whole tensor-load latency

`StinkyWaitCntInsertionPass` sets the `s_wait_tensorcnt` depth on the scheduled loop.  A one-copy
steady body waits for `tensorcnt 0` every trip, 0.04 to 1.0 trips (median about 0.3) after the
load it retires issued; a multi-copy body keeps each load in flight 1.0 to 1.25 trips.  On 6
variants measured twice on one gfx1250 card (2026-11-19, 4096×5120×65536, 14-24 kernels each,
the faster of the two runs kept), one-copy kernels run at a median 70-79% of the variant's best
and multi-copy ones at 87-96%, on every 8×10 / 8×8 wave tile:

| Variant (wave tile, DepthU) | Tier 0 before | Tier 0 now | Tier 2 | Tier 3 |
|---|---|---|---|---|
| MXF8 16×16×128 8×10, 256 | +0.46 | +0.52 | +0.69 | +0.94 |
| F8 16×16×128 8×10, 256 | +0.82 | +0.81 | +0.78 | +0.98 |
| MXF8 16×16×128 4×4, 256 | +0.78 | +0.78 | +0.69 | +0.37 |
| F8 16×16×64 8×10, 128 | −0.04 | +0.34 | +0.13 | +0.68 |
| BF16 16×16×32 8×8, 64 | −0.31 | +0.62 | +0.62 | +0.82 |
| BF16 16×16×32 4×4, 64 | −0.56 | −0.63 | −0.66 | −0.18 |

(Spearman against GFLOPS within each variant.)  The 4×4 tiles' trips (256 cycles) are far
shorter than the load latency, so every body drains whatever its copies.  The term still
charges a one-copy body `tdmLatency` and a multi-copy one `tdmLatency − 256`, 800 against 544,
and on BF16 4×4 that is backwards: the one-copy kernels with both operands resident (CLR 1)
measured 0.97 PF on 2026-11-20, against 0.79 for every multi-copy kernel tied at the best
score.  They score 1201 against 997, 20% above the best, so a tuning set drawn near the best
score never holds them; tier 2 ranks them first (408–422 cycles against 832).  See
[selection.md](selection.md).  The term should not charge a short trip's one-copy body more
than its multi-copy body; that change is not made yet.  BF16 4×4's slowest kernels also hold all of A
and B's next-trip ds_loads in registers past v255 and switch `s_set_vgpr_msb` inside the WMMA
chain, which no tier models.  The 40-kernel MXF8 calibration set (all but one multi-copy)
stays at +0.51.
