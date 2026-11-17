# Editing the DAG scheduler

Rules for changing the CDNA5 DAG scheduler (`CDNA5.hpp`, `DsSpreadPacer.hpp`,
`InFlightQueue.hpp`, `ReadyQueue.hpp`, `RegionDAG.*`, `SchedulingKnobHeuristics.cpp`).
Read this before you add a gate, a phase, a promotion rule, or a knob.

Most past scheduler bugs were not wrong rules. They were right rules that disagreed with each
other: one gate let a node through and a later one rejected it, or a low-priority rule
outvoted a high-priority one. Every rule below exists to stop that.

Verify names against the code before you rely on them. Line numbers drift.

## How a pick is made

`CDNA5ReadyQueue::pickOne()` runs these steps in order. Each step either returns a node or
falls through to the next.

| Step | What it does |
| --- | --- |
| `decidePromote()` | Pure decision. The first rule in `kRules` that fires sets `promotedPhase_` and `promotedNode_`. Mutates no queue. |
| Phase B | Issues the best WMMA if no veto holds it. Runs only when nothing is promoted. |
| Phase C / D | `NonWmmaFill`: issues the promoted fill node, else the smallest pickable non-WMMA (C inside the WMMA window, D outside). |
| Phase E | Forced WMMA: the most-ready WMMA, before barriers. |
| Barrier | The only place a barrier issues: the promoted one, or the drain case. |
| Phase G | Fallback: the least-blocked ready node, so the schedule always makes progress. |

`isPromote(p)` gates each phase: when nothing is promoted every phase runs; when something
is promoted only its phase runs, and the promoted node issues through that phase.

## The priority chain

When two rules want different things, the higher level wins. `PlacementLevel` in
`CDNA5.hpp` encodes this order, and `decidePromote()` asserts its rule table is sorted by it.

| Level | `PlacementLevel` | Goal |
| --- | --- | --- |
| L1 | `Barrier` | The barrier lands where it is planned. |
| L2 | `LastDs` | The last ds before a barrier issues in time for that barrier. |
| L3 | `WmmaQueue` | The WMMA queue stays busy: no window is wasted (the refill). |
| L4 | `WmmaTail` | Back-to-back WMMAs before a barrier cover its stall (`kBarrierTailWmmas`, `wmmaBarrierCover`). |
| L5 | `Prefetch` | Prefetches and global loads issue at their planned window. |
| L6 | `DsPacer` | ds loads are spread over the region (`DsSpreadPacer`). |
| L7 | `Filler` | SALU and VALU fill whatever slots are left. |

Consequences:

- A lower level never vetoes a higher one. The ds pacer (L6) must not delay the last ds
  before a barrier (L2). A filler (L7) must not take a prefetch's window (L5).
- Cover is measured, not counted. `wmma ds×8` does not cover a barrier; `wmma wmma ds×8`
  does. A rule that asks "is the barrier covered" uses `queuedCoverCycles()`, not the number
  of instructions picked.
- The WMMA tail (L4) exists to provide cover. VALU and SALU may co-issue inside it. Do not
  block them there.

## Rules

### 1. No spaghetti guards

The guard that admits a node into a phase and the phase that issues it must agree. If an
outer guard lets a WMMA through, nothing inside the phase may reject that WMMA for a reason
the outer guard did not check.

Wrong:

```cpp
// Outer guard: Phase B is entered because a WMMA "can go".
if (canIssueWmma(best)) {
    // Inner guard rejects it for a rule canIssueWmma() never saw.
    if (violatesTailRule(best)) { /* fall through, or pick something else */ }
    ...
}
```

Right: put the condition in the one predicate both places use.

- Eligibility of a candidate is decided by one shared function. Today those are
  `issuableNow()`, `findSmallestPickableNonWmma()`, `isValuPickable()`, and
  `isBarrierEligibleNow()`. If you need a new rejection, add it to the shared predicate so
  every gate that asks "can this go?" gets the same answer.
- If a guard asks "is there a pickable non-WMMA?", it must call the same function the fill
  phase picks with, not a look-alike.
- A rule that would only fire after another rule's guard has passed is a rule at a lower
  level. Rank it in the priority chain instead of nesting it.

### 2. Phases flow one way

If phase A issues nothing, control falls through to phase B, then phase C. A phase never
runs another phase's body.

Wrong:

```cpp
// Phase B
if (specialCondition) {
    return doPhaseA();  // jumps back to an earlier phase
}
```

Right: if phase A's work must win under a condition, express that condition where phase A
decides, or as a promotion rule. Then phase A issues it on its own turn, and the later
phases never need to know.

Existing debt, do not copy it: Phase E calls `pickFallback()` (Phase G) when
`heldMemOpPending()`. Fix it by moving that condition into the ordering, not by adding more
calls like it.

### 3. Forcing goes through `decidePromote()`

- A new forcing rule is one `promoteDue*()` / `promote*()` function plus one row in
  `kRules`, at its `PlacementLevel`. A rule between two levels is a new enumerator.
- Never add an early `return` at the top of `pickOne()` or inside a phase to force a node.
- `decidePromote()` stays a pure decision: it reads state and calls `promote()`. It never
  erases from a queue or advances time.
- A promoted node must be issuable now (`issuableNow()`). Do not promote a node that its
  phase would then refuse.

### 4. One model per concern

Each concern has one model. Query it; do not build a second copy beside it.

| Concern | Model |
| --- | --- |
| `s_wait_alu` waits and hazards | `WaitAluTracker` (`query`, `untilFree`, `requiredGap`, `commit`), the same scoreboard `InsertWaitAluPass` uses. |
| ds pacing | `DsSpreadPacer` and the ds credit queue (`dsReadInflight_`). |
| WMMA queue and cover | `outstandingWmmas()`, `wmmaQueueDepth()`, `queuedCoverCycles()`. |
| Barrier placement | The barrier thresholds (`barrierWmmaThresholds_`). |
| Prefetch placement | `prefetchEarliestWmma_` and `needByWindow_`. |

- The scoreboard counts instructions, not cycles. Do not convert its answer to cycles with
  an ad-hoc factor. If a rule needs cycles, that is a separate design discussion.
- A rule that disagrees with a model fixes the model, not a local exception.

### 5. Plans move with dependencies

Planned positions (barrier thresholds, prefetch windows, the last ds) are estimates made
before the DAG is scheduled. Every pick still respects the DAG: only nodes with in-degree 0
are ready.

- When a dependency delays a planned node, the plan moves with it. Do not hold other work
  to hit the original position.
- Do not add a rule that issues a node ahead of its producers, or that assumes a node is
  ready because the plan said so.

### 6. Refactors do not change behavior

- A refactor keeps the generated assembly identical. If it does not, either it is not a
  refactor, or you found a bug; report which.
- Do not add a feature, a knob, or a heuristic in a refactor PR.
- Knob defaults live in `SchedulingKnobHeuristics.cpp`, not in a phase.

## Before you send the change

Build and run the StinkyTofu ctest suite.

Read the related docs:

- [`docs/developer/dag-wait-and-prefetch-placement.md`](../../../../docs/developer/dag-wait-and-prefetch-placement.md)
- [`docs/developer/wmma-batch-ds-spread-plan.md`](../../../../docs/developer/wmma-batch-ds-spread-plan.md)
- [`docs/developer/waitalu-placement.md`](../../../../docs/developer/waitalu-placement.md)
- [`docs/developer/cluster-barrier.md`](../../../../docs/developer/cluster-barrier.md)
