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

### What is removed

- Done: the cycle hazard model. The `hazardGates_` lanes, `getHazardWait`, `DAGNode::hazardFlags`
  and `hazardDeadline`, the deadline pre-scan in `StinkyDAGSchedulerPass.cpp` and the
  `promoteDuePrefetchHoist` / `promoteDueFillerHoist` rules are gone. In their place:
  - `promotePrefetchAddressValu` (Prefetch level): a filler feeding a prefetch issues once its
    need-by window (`needByWindow_`: the prefetch's planned window less `requiredGap` WMMAs)
    comes, so it does not lose its slot to a ds (the ds pacer is a lower level).
  - `unplacedMemWaitsOut`: a prefetch or VMEM load no level placed (no prefetch lead) waits while
    WMMAs can issue, until `untilFree` says its `va_vdst` is hidden.
  - SALU to memory address: the producer is placed first by the need-by rank; nothing holds the
    consumer (the hardware stalls).
  - The tracker runs whenever InsertWaitAlu does (`dagFeatures.waitAluScoreboard`, set from
    `EnableESM2`; `--wait-alu-scoreboard` in stinkytofu-opt). Without ESM2 (f64 kernels)
    there is no scoreboard and the hardware interlocks.
  The rule table (`HazardRules.hpp`) stays for `HazardGapAnalysisPass` and the pipe-op rule
  `WmmaVgprSrcToDsWrite`.
- Not yet: the fill quota, critical fillers (`needWindow_`, `kCriticalSlackWmmas`) and the
  strict-count hold.

## Steps

Each step is one commit, checked with the GPU-less A/B on the 21 gfx1250 perf kernels: total and
strict `s_wait_alu` (whole kernel and main loop), kernels changed, barrier positions, and SALU to
memory-address pairs under 8 cycles.

0. `untilFree` and `requiredGap` on `WaitAluTracker`, with unit tests. No asm change.
1. Swap: in `pickFreeBest`, a filler whose wait a later slot would drop (`untilFree` finite) ranks
   below every filler of its queue that needs none. Not demoted: a wait no move removes (an EXEC
   writer's `va_vdst(0)` drain), and a filler that feeds an op a higher level placed
   (`feedsPlacedOp`) or is the last input of real work. No asm change. A version without the
   placed-op exemption looked like -3 waits but added 2 strict ones; the same rank across VALU and
   SALU cost 4 waits in mab mxf8. Track strict waits, not only the total.
2. Producer first: `pickFreeBest` ranks free fillers by `needByWindow_`, earliest first. Loop
   `s_wait_alu` 577 -> 561, strict 61 -> 59. Moving the critical-filler trigger earlier by the gap
   too was worse: it pulled prefetch-address VALUs onto the WAR side of the prefetch still
   reading the register (`vm_vsrc(0)`).
3. Cycle hazard model removed (above). Loop `s_wait_alu` 561 -> 545 (577 on develop), strict
   59 -> 66 (61 on develop); whole kernels 26724 -> 26693, strict 16939 -> 16937. SALU to memory
   address under 8 cycles: 0 of 369 pairs before and after. No DAG barrier moves past its
   threshold; two barriers in medium mxf8 that already issue about 10 WMMAs late (their ds are
   late, rule-conflicts P1) issue one WMMA later still.
3b. Prefetch bridge over the back edge. Step 3's +5 strict loop waits were the prefetch-address
   VALUs: `PrefetchBridgeSubstitutionPass` declined any prefetch group that reaches an address
   overwrite around a back edge, so one address add moving past the loop's mid label lost the
   `flat_prefetch` anchor and each add took its own `vm_vsrc`. The gap over a back edge is now the
   fewest LDS ops along any CFG path. Loop `s_wait_alu` 545 -> 484, strict 66 -> 43 (develop 577 /
   61); whole kernels 26693 -> 26630, strict 16937 -> 16914. On b8-3 (interleaved reps) the PR
   alone was +0.49% slower on mxf4 medium; with this step +0.18%, mxf8 maf -0.13%, f8 maf
   finetune +0.14% (15 reps, within noise).
3c. Prefetch placement (PR #14366, below this stack). The cycle hazard model was also what placed
   each prefetch: develop fired it at its hazard deadline, about 5 windows after its planned slot.
   With the model gone the prefetch issues at the slot (`window - lead`), where it ran 1% behind
   develop on f8 k128k and 1.3% on mxfp8 medium. The pin (`promotePrefetchAtWindow`, after the
   window's fills) keeps prefetches where develop put them without the model.
3d. A negative need-by does not promote. A producer whose window is below 0 can no longer meet it;
   promoting it only took the slot of a ds the pacer had released. MAB kernels plan the whole
   prefetch group in window 3, so every address VALU hit this: f8 mab +1.18% -> +0.49%, nvf4 mab
   -1.11%, mxf8 mab -0.10%.
3e. Prefetch lead 40 -> 30 (`KernelWriter.py`). 40 was measured while the model added the delay.
   gfx1250 MAF sweep (5 reps, % vs develop, leads 25/30/35/40/45): the average is -0.10/-0.30/-0.16/
   +0.27/+0.86; 40 and above lose on every kernel but one, 30 is never worse than +0.02. A 64-window
   stage (4x8 tile) shows the same band: good 15-30, bad at 10 and from 35. The cliff moved by about
   5 windows when the stage halved, so the lead is a time (about 130-290 cycles before the next
   tensor_load), not a fraction of the stage.
3f. Single-stage prefetch hold. A prefetch group in a single-stage loop (lead below
   `kSingleStageLeadBelow`) is held while `needsMoreInstsBeforeIssue` says its address VALUs'
   `va_vdst` is not yet hidden, as develop's hazard gate did. A group not ready by the stage
   barrier then lands after its tensor_load. The negative need-by skip of 3d applies to
   single-stage loops only. `unplacedMemWaitsOut` holds a VMEM load or unpinned prefetch the same
   way, but only behind a WMMA that can issue; when the most-ready WMMA still has a source-data
   wait, Phase E lets the held op through Phase G instead of stalling.
3g. Both 3f patches are needed. Removing them (asm differs on the 5 MAB kernels only; medium and
   MAF are identical) costs on b8-3, % vs develop, pinned stack vs patches removed: bbs mab
   -0.02/+0.05, f8 mab -0.17/+1.24, mxf4 mab -0.33/+0.06, mxf8 mab -0.24/+0.35, nvf4 mab
   -1.18/+0.25. Final stack vs develop on all 21 perf yamls (8 interleaved reps): -1.20% to
   +0.42%; the worst are f8 medium +0.42% (sd 0.9), mxf4 maf +0.26% and mxf4 medium +0.24%.
4. Next: filler pacer (catch-up instead of the per-window quota), replacing the critical fillers
   and the strict-count hold; cross-queue order in its init plan.
5. Rule 2 as a hold, once the pacer gives held VALUs a later slot. Tried alone it made things
   worse (loop `s_wait_alu` 577 -> 634, strict 61 -> 78).

## Open

- `WaitAluTracker` starts each BB empty, where the pass seeds it from its predecessors, so
  loop-carried producers are not seen at the top of the loop body. Seed the tracker from the
  pass's BB entry state.
- When a candidate needs both `va_vdst` and `vm_vsrc`, how their two counts combine in the rank.
- Loop strict waits rose by 5 in step 3: traced and fixed in step 3b (medium mxf8 stays at 5).
- The prefetch lead is an empirical time, not derived. The MI400 shader guide gives the semantics
  (a SCOPE_SE prefetch fills GL2 only, returns nothing, and only VM_VSRC waits on it) but no
  latencies. A formula would need the TDM completion time and GL2 residency.
- The 5 left in medium mxf8 are the LDS read-address swap `v_xor` after the last ds that reads the
  old address (`vm_vsrc(0)`). Its need-by window (4-8) is far behind the region (WMMA 12-20), so
  holding it only delays the ds that need the new address. The cause is ds issuing late
  (rule-conflicts P1), not filler placement.
- Loop waits left after 3g, over the 21 yamls: 4 strict `va_vdst(0)` (three MAB kernels, the tail
  prefetches after the last WMMA of the iteration) and about 66 `vm_vsrc` at 0 or 1. Counting
  non-WMMA ops cannot remove a `va_vdst` wait, only weaken it: a CSMACC producer is hidden by
  matrix ops (`xdlHideCsmacc`), and other VALU followers only raise the count.
