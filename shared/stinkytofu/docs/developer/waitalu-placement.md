# Scoreboard-driven VALU placement (gfx1250 DAG)

Design for replacing the CDNA5 DAG's cycle hazard model with InsertWaitAlu's scoreboard. Status:
proposed. It extends [DAG placement of prefetches and wait-prone fillers](dag-wait-and-prefetch-placement.md).

## Rule

InsertWaitAlu decides whether an `s_wait_alu` is inserted, so its scoreboard (`WaitAluTracker`)
is the only source of truth. The DAG asks it, for a candidate at the current point, "if I place
this here, will the pass insert a wait?"

- No wait: place it.
- Wait: place a different VALU in this slot, and this one where it needs no wait (earlier or
  later, see below).
- No wait-free choice: place the one with the smallest wait.

The goal is fewer stalls. No wait is best; otherwise the weakest wait.

The DAG keeps no wait model of its own: no cycle counts, no hide thresholds. It only asks the
scoreboard and decides the final order. Every count below comes from `WaitAluTracker`.

The placement levels stay strictly ordered (barrier location, last ds, WMMA queue, WMMA tail,
prefetch, ds pacer, fillers). Avoiding a wait never moves a node a higher level placed: it moves
the VALUs around that node.

## What the scoreboard counts

The unit is "N instructions of a class in between", never cycles (`InsertWaitAluPass.cpp`).

| Wait | Pair | Count | Hidden (no wait) when |
| --- | --- | --- | --- |
| `va_vdst` | VALU or WMMA writes a VGPR, a later op reads it | VA-pipe ops after the producer | WMMA (XDL) or plain VALU (CSMACC) producer followed by `HWModel::WaitHide` matrix ops (11–13 by form, times the form's step) with no DPMACC/TRANS op since; or an earlier wait drained it |
| `vm_vsrc` | ds/VMEM reads a VGPR, a later VALU overwrites it | ops after the reader in the same FIFO (LDS or TEX) | 11 ops in that FIFO followed the reader (`vmVsrcLds`/`vmVsrcTex`), a bridge op hid it, or it drained |

Otherwise the wait is the count, capped at 14 (`va_vdst`) or 6 (`vm_vsrc`). A larger count is a
weaker wait: `va_vdst(5)` beats `va_vdst(1)`.

## Why reorder the VALU queue, not delay or advance a VALU

A VALU issues only into a VALU co-issue slot of the active WMMA window (`isValuPickable()`), and
only when no ds is ready and the fill quota allows (`findSmallestPickableNonWmma`). WMMAs and ds
set the slots; VALUs fill what is left. Delaying a VALU alone hands its slot to the next candidate,
the fill quota can then close the window, and the VALU lands in a later window where it competes
with that window's ds and VALUs (a short dependent WMMA gets `v_nop` padding instead). Advancing
one takes a slot from the next VALU in line. Choosing which ready VALU fills each slot keeps the
slot count and the WMMA/ds placement unchanged.

In the main loop of the gfx1250 perf kernels VALUs are sparse (0–1 per WMMA window) and most
loop waits are a `va_vdst` in front of a ds whose address a VALU just computed:

| Kernel | Loop WMMA / VALU / ds | `s_wait_alu` before ds / VALU / VMEM | `va_vdst` / `vm_vsrc` |
| --- | --- | --- | --- |
| maf mxf8 | 768 / 84 / 804 | 18 / 10 / 8 | 26 / 10 |
| maf mxf4 | 384 / 42 / 420 | 37 / 6 / 0 | 37 / 6 |
| medium mxf8 | 96 / 36 / 204 | 12 / 11 / 11 | 23 / 11 |
| maf bbs | 768 / 24 / 384 | 39 / 3 / 0 | 39 / 3 |

Those ds are placed by the pacer or the last-ds rule, so the fix is to issue the address VALU
earlier in the VALU order, not to move the ds.

## Mechanism

### Scoreboard API

`WaitAluTracker` (`InsertWaitAluPass.hpp`) answers three questions, all from the pass's own
scoreboard:

- `query(inst)`: the wait `inst` needs if it issues next.
- `untilFree(inst, op)`: how many more `op`s (`Wmma`, `Lds`, `Tex`, `Flat`) must issue before
  `inst` no longer needs the wait that kind of op hides; -1 if none will. It counts ops that touch
  no VGPR on a copy of the scoreboard, so it is exact for the hides and an upper bound where an
  anchored flat op would drain sooner.
- `requiredGap(producer, consumer, op, formWmma)`: the same count for `consumer` right after
  `producer` with nothing else in flight. The WMMA form is the one already issued, else
  `formWmma`'s.

### Pick: rank the VALU queue by the scoreboard

`pickFreeBest(valuQueue)` already chooses among every ready VALU. Rank by `waitAlu_->query()`
first: no wait, then the one `untilFree` says frees soonest, then the weakest wait, then the
existing keys (link urgency below, MSB bank,
elapse, id). A candidate that needs both counters is compared by the sum of its two counts.
`otherQueue` (SALU) ranks the same way where its picks carry a wait.

### Init: pre-scan the pairs and link them

In `onInitRegion`, walk the region's def-use chains once and ask `requiredGap` for each
producer-consumer pair; a non-zero answer is a pair that waits if the two end up too close. For each, record on the VALU side which way
it should move, as a per-node field set at init (not a DAG edge):

- **VALU feeds a placed consumer** (ds, prefetch, VMEM address or data, WMMA operand): the VALU
  should issue at least `requiredGap(..., Wmma)` WMMA windows before the consumer's planned
  window. A plain VALU feeding a ds address needs about 12, so this moves it whole windows earlier;
  reordering inside one window only weakens that wait. Store that window as the VALU's need-by window. It does not have to be exact; the
  pick ranks VALUs with an earlier need-by ahead of others that are equally wait-free.
- **VALU overwrites a ds/VMEM source** (`vm_vsrc`): the VALU should come after enough ops in that
  FIFO. Store the reader's planned position as a not-before hint; the pick's `query()` decides
  when it is actually clean.

The pick path reads these fields; no phase gets its own guard.

## Removed once the A/B shows fewer waits

- The cycle hazard model: the cycle lanes of `hazardGates_`, `getHazardWait`, `DAGNode::hazardDeadline`,
  the `cumCycles` pre-scan in `StinkyDAGSchedulerPass.cpp`, and the `promoteDuePrefetchHoist` /
  `promoteDueFillerHoist` rules. `SaluSgprToMemAddr` and `ValuVgprToVmemAddr` go with it; the
  pipe-op rule `WmmaVgprSrcToDsWrite` stays.
- The strict-count hold in `needsWaitAlu` (`WaitAluHoldStrictCount`, released two windows before
  a barrier wait).

## Steps

Each step is one commit, checked with the GPU-less A/B on the 21 gfx1250 perf kernels: total and
strict `s_wait_alu`, `vm_vsrc(0)`, kernels changed, barrier positions.

0. `untilFree` and `requiredGap` on `WaitAluTracker`, with unit tests. No asm change.
1. Rank `pickFreeBest(valuQueue)` by `query()` and `untilFree()`.
2. Pre-scan need-by windows for VALUs feeding placed consumers, and use them in the rank.
3. Not-before hints for `vm_vsrc` pairs.
4. Remove the cycle hazard model and the strict-count hold.

## Open

- `WaitAluTracker` starts each BB empty, where the pass seeds it from predecessors, so loop-carried
  producers are not seen at the top of the loop body.
- Whether a pair whose consumer is a WMMA (operand written by a VALU) should move the VALU or let
  the WMMA queue hide it.
