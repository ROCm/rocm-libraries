# Scoreboard-driven filler placement (gfx1250 DAG)

Design for replacing the CDNA5 DAG's cycle hazard model with InsertWaitAlu's scoreboard. Status:
proposed. It extends [DAG placement of prefetches and wait-prone fillers](dag-wait-and-prefetch-placement.md).

## Rules

1. **The scoreboard is the only source of truth for waits.** InsertWaitAlu decides whether an
   `s_wait_alu` is inserted, so the DAG asks its scoreboard (`WaitAluTracker`) and keeps no wait
   model of its own: no cycle counts, no hide thresholds. The DAG only decides the final order.
2. **A VALU co-issues with WMMAs only if no wait is inserted.** If `query()` says it would wait,
   it does not issue in that slot. The smallest wait is taken only when nothing else can issue,
   WMMAs included.
3. **Placement levels are strict.** Barrier location > last ds > WMMA queue > WMMA tail >
   prefetch > ds pacer > fillers. Avoiding a wait never moves a node a higher level placed.
4. **Fillers move by swapping, not by delay.** A VALU or SALU only trades slots with another
   filler. A swap keeps the number of filler slots per window, so WMMAs and ds stay where they
   are. Delaying a filler alone pushes it into a later window, where it competes with that
   window's ds and fillers.

### The two cases

- **The consumer is fixed** (a prefetch, a ds the pacer placed, a VMEM or tensor_load): the
  producer moves earlier, swapping with an earlier filler.

  ```text
  before: vX(slot 3) ... vAddr(slot 9) -> prefetch(vAddr)   too close: wait
  after:  vAddr(slot 3) ... vX(slot 9) -> prefetch(vAddr)   no wait
  ```

- **The consumer is free to move** (another filler): it moves later, swapping with a later
  filler.

  ```text
  before: vP(slot 3) -> vC(slot 4, reads vP: wait) -> vY(slot 7)
  after:  vP(slot 3) -> vY(slot 4) -> vC(slot 7)            no wait
  ```

If no swap removes the wait, the weakest wait is taken.

### SALU sgpr to memory address

A SALU write of an sgpr that an SMEM, tensor_load or VMEM reads as an address stalls the wave if
the gap is under 8 cycles; the hardware interlocks, so nothing has to be inserted. It is placed by
the first case above: the SALU producer goes as early as it is ready. There is no threshold and no
cycle count, only "producer first".

## What the scoreboard counts

The unit is "N instructions of a class in between", never cycles (`InsertWaitAluPass.cpp`).

| Wait | Pair | Count | Hidden (no wait) when |
| --- | --- | --- | --- |
| `va_vdst` | VALU or WMMA writes a VGPR, a later non-VALU reads it | VA-pipe ops after the producer | 11–13 WMMAs (by form, `HWModel::WaitHide`) follow a plain VALU (CSMACC) or WMMA (XDL) producer with no f64/transcendental op since; or an earlier wait drained it |
| `vm_vsrc` | ds/VMEM reads a VGPR, a later VALU overwrites it | ops after the reader in the same FIFO (LDS or TEX) | 11 ops in that FIFO follow the reader, a bridge op hid it, or it drained |

Otherwise the wait is the count, capped at 14 (`va_vdst`) or 6 (`vm_vsrc`). A larger count is a
weaker wait. A VALU or WMMA consumer never gets a `va_vdst` (`computeWaitForInst`), so a VALU
feeding a WMMA operand is not a pair.

In the main loop of the gfx1250 perf kernels VALUs are sparse (0–1 per WMMA window) and most
loop waits are a `va_vdst` in front of a ds whose address a VALU just computed:

| Kernel | Loop WMMA / VALU / ds | `s_wait_alu` before ds / VALU / VMEM | `va_vdst` / `vm_vsrc` |
| --- | --- | --- | --- |
| maf mxf8 | 768 / 84 / 804 | 18 / 10 / 8 | 26 / 10 |
| maf mxf4 | 384 / 42 / 420 | 37 / 6 / 0 | 37 / 6 |
| medium mxf8 | 96 / 36 / 204 | 12 / 11 / 11 | 23 / 11 |
| maf bbs | 768 / 24 / 384 | 39 / 3 / 0 | 39 / 3 |

A plain VALU feeding a ds address needs about 12 WMMAs before the ds, so its producer has to
move whole windows earlier; reordering inside one window only weakens that wait.

## Mechanism

### Scoreboard API

`WaitAluTracker` (`InsertWaitAluPass.hpp`) answers three questions, all from the pass's own
scoreboard:

- `query(inst)`: the wait `inst` needs if it issues next.
- `untilFree(inst, op, formWmma)`: how many more `op`s (`Wmma`, `Lds`, `Tex`, `Flat`) must issue
  before `inst` no longer needs the wait that kind of op hides; -1 if none will. It counts ops
  that touch no VGPR on a copy of the scoreboard, so it is exact for the hides and an upper bound
  where an anchored flat op would drain sooner.
- `requiredGap(producer, consumer, op, formWmma)`: the same count for `consumer` right after
  `producer`, with nothing else in flight.

Both take the WMMA form the `va_vdst` hide counts from the WMMAs already issued, else from
`formWmma` (an XDL WMMA), so a BB's first region does not see every VALU pair as unhideable.

### Filler pacer

Fillers (VALU and SALU) get a plan in `onInitRegion`, like `DsSpreadPacer` plans ds:

1. **Producers of fixed consumers first.** Each is released by its consumer's planned window
   minus `requiredGap` WMMAs (VALU), or as soon as it is ready (SALU to memory address).
2. **The rest are spread evenly** over the remaining windows. The default is one per window (the
   current fill quota, `ceil(fillers / WMMAs)`); a window takes extra fillers to catch up when the
   region is behind the plan, or when a producer of a fixed consumer cannot be swapped in time.
3. **The pick fills a released slot with the best released filler:** no wait first, then the one
   `untilFree` says frees soonest, then plan order. A filler that would wait is swapped for the
   next released one; that is rule 4.

The pacer re-anchors each stage at the barrier that actually issued, so one late barrier does not
release a catch-up burst in the next stage (rule-conflicts F4).

It replaces the fill quota, `needWindow_` / critical fillers, and the strict-count hold in
`needsWaitAlu`.

### What goes away

- The cycle hazard model: the cycle lanes of `hazardGates_`, `getHazardWait`,
  `DAGNode::hazardDeadline`, the `cumCycles` pre-scan in `StinkyDAGSchedulerPass.cpp`, and the
  `promoteDuePrefetchHoist` / `promoteDueFillerHoist` rules. `ValuVgprToVmemAddr` is covered by
  the scoreboard; `SaluSgprToMemAddr` becomes a "producer first" tag in the pre-scan. The pipe-op
  rule `WmmaVgprSrcToDsWrite` stays.
- The fill quota, critical fillers and the strict-count hold, once the pacer covers them.

## Steps

Each step is one commit, checked with the GPU-less A/B on the 21 gfx1250 perf kernels: total and
strict `s_wait_alu`, `vm_vsrc(0)`, kernels changed, barrier positions.

0. `untilFree` and `requiredGap` on `WaitAluTracker`, with unit tests. No asm change.
1. Swap: in `pickFreeBest`, a filler that `query()` says would wait ranks below every one that
   would not. Measured: 1 kernel changed (medium mxf8), loop `s_wait_alu` 577 -> 575, no barrier
   moves. VALUs are sparse, so a slot rarely has two candidates to swap between.
2. Filler pacer, replacing the fill quota, critical fillers and the strict-count hold; producers
   of fixed consumers released `requiredGap` WMMAs ahead. This is where most loop waits (a
   `va_vdst` in front of a ds) can go.
3. Rule 2 as a hold, once the pacer gives held VALUs a later slot. Tried alone it made things
   worse (loop `s_wait_alu` 577 -> 634, strict ones 61 -> 78): held VALUs landed next to the ds
   that read them.
4. Remove the cycle hazard model; SALU to memory address becomes "producer first".

## Open

- `WaitAluTracker` starts each BB empty, where the pass seeds it from its predecessors, so
  loop-carried producers are not seen at the top of the loop body. Seed the tracker from the
  pass's BB entry state.
- When a candidate needs both `va_vdst` and `vm_vsrc`, how their two counts combine in the rank.
