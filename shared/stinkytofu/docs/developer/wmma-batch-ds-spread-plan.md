# WMMA batch refill and even ds_load spread: plan

> **Superseded.** This plan describes the earlier slot/section pacer (per-load slots, group timer, consumer deadlines, `DsSpreadToleranceCycles`, gap-CV metric). All of that was replaced by the per-stage even share per WMMA batch (`DsSpreadPacer.hpp`, options `DsBatchWmmas` and `DsCatchUpPace`); see the `DsSpread` row of `docs/user/scheduler-tuning-parameters.md`. Kept for history.

Status: phases 2-5 are implemented. The legacy ds pacing (cap, throttle, slot-first, hide-budget
prescan) is removed; the ds credit queue and `DsSpreadPacer` are the only ds pacing, and
`DsSpread` is on by default (its own commit, easy to revert). The perf A/B
(phase 7) still needs hardware. Scope: the gfx1250 CDNA5 DAG scheduler
(`src/transforms/asm/dag/CDNA5.hpp`).

## Why now

[#13704](https://github.com/ROCm/rocm-libraries/pull/13704) (`LDSSegmentBRotate`)
puts the two SIMD pairs on different LDS components. The ds-to-ds stall is gone,
so the scheduler no longer needs a stack of rules to dodge LDS port conflicts.
Today ds_load pacing is five overlapping mechanisms (cap, throttle, slot-first,
hide-budget ds distribution, knob heuristics). They were tuned for one WMMA per
window and bursty ds issue.

## Cost model that drives the design

| Event | Stall? | Efficiency loss? |
|---|---|---|
| ds_load issued while the ds queue (FIFO) is full | yes | **yes**, if no WMMA is running |
| WMMA issued while the WMMA queue is full | yes | **no**, the pipe is already busy |

So the WMMA queue is kept busy and the hardware drains it. The queue covers
small stalls and hides barrier stalls. WMMAs are over-issued on purpose. The ds
stream is spread so the ds FIFO never fills while the WMMA queue is thin.

Issue pattern, before and after:

```
before: wmma ds ds wmma ds ds wmma ds ds barrier
after:  wmma wmma ds ds wmma wmma wmma barrier
```

## Requirements

1. Batch WMMA already exists (`WmmaQueueDepth`, `WmmaQueueCoverCycles`).
2. Two WMMA issue modes: **fixed** (always N per refill, e.g. 4) and
   **dynamic** (the compiler decides how many).
3. Refill trigger is tunable: (a) the refill line, i.e. refill when the queue has
   only R cycles left; (b) the refill target, i.e. refill up to T cycles. The WMMA
   count is derived from the timeline and `wmma.latency`.
4. ds_load mechanism is simplified: reuse the queue system, spread ds_loads as
   evenly as possible over the whole main loop without breaking dependencies.
5. "How even" is computed automatically, with a manual override.
6. Prefetch system unchanged.
7. Evenly spread VALU/SALU fillers unchanged. The wait-alu pass (Darren's) is
   already used and is not touched.
8. `DsReadOrder` (all orders, including `AscendingCache`) unchanged.

## Design

### 1. WMMA refill policy

Replaces the one-at-a-time `preemptForQueue` rule.

- **Trigger:** refill starts when `queuedCoverCycles() < WmmaRefillBelowCycles`
  (R) and the queue has room.
- **Fixed mode** (`WmmaRefillMode=Fixed`, `WmmaBatchSize=N`): issue exactly N
  WMMAs back to back.
- **Dynamic mode** (`WmmaRefillMode=Dynamic`, `WmmaRefillToCycles=T`): issue
  until `cover >= T`. Count = `ceil((T - cover) / wmma.latency)`, from the live
  timeline.
- **A batch ends early** when the queue is full, no WMMA is ready, or a dependent
  WMMA needs coexec v_nops (the existing nop logic runs).
- **During a batch** Phase B skips the active-window, interleave and hide-budget
  blockers, as `preempt` does today.
- **Barriers:** the pre-scan sizes the batch before each barrier from the
  barrier's stall estimate, so the pipe is full when the barrier arrives.
- **Auto defaults** (R, T unset): T = full queue (`depth * L`), R = T / 2. Starting
  point, tune on hardware.

### 2. ds_load even spread (`DsSpreadPacer`)

New `src/transforms/asm/dag/DsSpreadPacer.hpp`. Replaces the cap, throttle,
slot-first and hide-budget ds logic.

- **Hard rule:** never issue a ds_load when the ds FIFO has no free credit while
  the WMMA queue is thin. Credits come from the existing `InFlightQueue`
  (`DsReadQueueDepth`).
- **Soft preference:** spread ds_loads evenly. A ready ds_load that fits a free
  credit issues. A load is delayed only to avoid a burst that would fill the
  FIFO. Evenness never idles the pipe by itself.
- **Timeline in cycles, not WMMA index, and adaptive:** at every decision the spread
  computes `gap = (WMMA cycles still to issue - ds return latency - tolerance) /
  ds_loads left` and issues the next ds_load once the clock is `gap` past the
  previous one. A load that was held (no credit, a dependency, a barrier) makes the
  next gaps shorter, so the stream catches up evenly and the last ds_load still
  returns before the region ends. A fixed plan (`k * regionTime / n`) was tried
  first: the real clock at region end is shorter than the plan, so the last loads
  were still held when the WMMAs ran out and drained as a burst of 15.
- **Per-load deadline:** a load whose first consumer WMMA needs it earlier has its
  own deadline (consumer window * L - lead) and issues then, bypassing the gap. No
  global slack: one tight consumer does not open the gate for the rest. The ds
  order lock is kept.
- **Never deadlocks:** the deadline lifts any hold, and the existing Phase G
  fallback stays.

### 3. How even

- **Metric:** coefficient of variation of ds_load issue gaps in the main loop,
  written to the debug log and asserted in tests.
- **Auto (-1):** no extra lead; the last ds_load issues its return latency before the
  region's last WMMA, and the per-load consumer deadlines are the only early release.
- **Manual:** `DsSpreadToleranceCycles=<cycles>`: extra lead on top of the return
  latency. Large = today's greedy behavior.

### 4. Unchanged

- Prefetch lead and staging.
- `EvenSpreadFillers` and its quota.
- Wait-alu pass and the co-issue scoreboard (`activeWindowSlots_`,
  `hazardGates_`, `regDataReadyCounters`). The new ds picker only reads them.
- `DsReadOrder` and `LockDsReadOrder`.

### 5. Legacy ds code: hard removal

Keeping the old rules for one release was considered and rejected. The old ds
pacing is touched 54 times in `CDNA5.hpp` and in about 20 other files. A selector
would put both paths in `findSmallestPickableNonWmma`, double the tests, and still
not reproduce old schedules once batch WMMA is on.

- Delete `DsIssueCap.hpp` and `DsIssueCapTest.cpp`.
- Delete the cap, throttle, transition, slot-first and hide-budget ds-distribution
  code.
- Reduce `SchedulingKnobHeuristics` to the cluster-barrier Rule 3 lead.
- Old module-option keys (`DsReadPerCap`, `DsReadPerWmma`, `DsIssueCapSpanCycles`,
  `DsIssueCapMode`, `DsReadThrottleLatency`, `DsReadThrottleTransition*`,
  `DsSlotFirst`) are still accepted but ignored, with a one-line warning.
- The A/B baseline comes from git history, not a runtime switch.

## Phases (one commit per phase)

1. **Baseline:** capture current output of the `dag_*.stir` filecheck tests and the
   gfx1250 yaml kernels, including `segment_b_rotate_gfx1250.yaml`, plus perf
   numbers.
2. **WMMA refill:** new options, refill state machine (`Types.hpp`, `CDNA5.hpp`,
   option plumbing). Unit tests in `CDNA5ReadyQueueTest`, filecheck for Fixed and
   Dynamic.
3. **`DsSpreadPacer`:** pre-scan, quota, credits, wired into
   `findSmallestPickableNonWmma`. Unit tests with synthetic windows.
4. **Evenness:** auto solver, manual knob, metric and log.
5. **Removal:** delete the legacy ds code and shrink the knob heuristics. Add the
   deprecation warning.
6. **Plumbing and docs:** CLI flags, `KernelWriter.py` `stinky_module_options`,
   yaml, rewrite `docs/user/scheduler-tuning-parameters.md`.
7. **Validation:** unit and filecheck suites, e2e yamls, perf A/B against the
   phase 1 baseline.

## Files touched

- `include/stinkytofu/core/Types.hpp` (new `DagFeatures` fields)
- `src/transforms/asm/dag/CDNA5.hpp`, new `DsSpreadPacer.hpp`, delete `DsIssueCap.hpp`
- `src/transforms/asm/dag/InFlightQueue.hpp` (drop throttle, keep credits)
- `include/.../WmmaHideBudgetAnalysis.hpp`, `src/analysis/asm/WmmaHideBudgetAnalysis.cpp`
- `include/.../SchedulingKnobHeuristics.hpp`, `src/.../SchedulingKnobHeuristics.cpp`
- `src/pipeline/backend/Gfx1250Backend.cpp`, `src/hardware/HWModel.cpp`,
  `src/conversion/rocisa/ToStinkyTofuUtils.cpp`, python bindings `Module.hpp`
- `tools/stinkytofu-opt/stinkytofu-opt.cpp`
- TensileLite: `GlobalParameters.py`, `ValidParameters.py`, `KernelWriter.py`
- Tests: `CDNA5ReadyQueueTest.cpp`, `DAGSchedulerPassTest.cpp`,
  `SchedulingKnobHeuristicsTest.cpp`, `tensor_load_dag_with_wmma.stir`, new tests
  for the pacer
- Docs: `docs/user/scheduler-tuning-parameters.md`,
  `docs/developer/dag-wait-and-prefetch-placement.md`

## Risks

- Soft spacing can still leave a short bubble when a ds_load is held and no other
  work is ready. Mitigation: hold only when other work can issue instead; the
  deadline and Phase G bound the rest.
- Every existing kernel's schedule changes, and hand-tuned ds values stop
  applying. Mitigated by the phase 1 baseline and the phase 7 perf A/B.
- Over-issuing WMMAs raises register pressure for accumulators and WAR gates on
  WMMA sources. The existing WAR gate is kept; check VGPR counts in phase 7.

## Open items

- Auto R and T defaults (T = depth * L, R = T / 2) need hardware tuning.
- Confirm the ds FIFO credit rule is the only hard ds rule (assumed yes).
