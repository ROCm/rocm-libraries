# Recommendations — after `Solution` derivation slice 3a

New file in the `SolutionDerivation/` test dir per the add-only rule.

## Result for this target

The small derivation statics (`setGlobalReadVectorWidth`,
`checkAndAssignWaveSeparateGlobalRead`) are fully covered; `getMIOutputInfo` /
`isVgprForLocalReadPackingDoable` reach their gfx942-reachable branches; the two
giant `assign*` methods and the DTV/DTL predicates are covered on the happy path
+ early/reachable rejects. `Solution.py` rose 38.81% → 41.14% line (+52 lines).
No regression: full `-m unit` **1818 → 1844 passed** (+26), 201 skipped.

The exhaustive reject matrices (DTV/DTL deep branches, the ~1000-line
`assignDerivedParameters` body, the WMMA/EccHalf cap-gated arms) need a
dtype×ISA×MI config sweep — **slice 3b**, a multi-day increment (see
`resistance.md`).

## What worked (additions to the shared list)

- **Crafted minimal states for key-reading statics.** Functions that read only a
  handful of `state` keys are fully covered with tiny hand-built dicts + one
  mutation per reject branch — no fixtures, fully deterministic.
- **Seed predicates/derivation from a real `_state`.** A deep copy of a real
  solution's derived state drives the predicates and the `assign*` re-runs
  (reset the `Assigned*` flags) without hand-authoring a self-consistent kernel.
- **Branch on the rocisa build for backend-gated paths.** The SIA=4 test checks
  `rocisa.hasStinkyTofuBackend()`/`isSupportedByStinkyTofu` so it pins the right
  outcome regardless of how rocisa was built.

## Go / no-go on the next target

### Verdict: **GO — `Common/GlobalParameters.py`** (self-contained; defer Solution slice 3b)

| Candidate | Current | Why / why not | Effort |
|---|---|---|---|
| **`Common/GlobalParameters.py` (220 stmts, 90% line)** ✅ chosen | 90% | Self-contained; the remaining misses are `assignGlobalParameters` config-handling branches (L643-695, 266) + `setupRestoreClocks` (L720-723, 759-763, a `subprocess` GPU-clock path → monkeypatch). `restoreDefaultGlobalParameters`/`printCapabilitiesTable` are straightforward (the latter takes the real `isaInfoMap`). A clean ~1-day top-up to ≥95%. | ~1 day |
| `Solution.py` slice 3b (derivation config sweep) | 41% | The largest remaining block, but a multi-day dtype×ISA×MI config matrix (each reject needs a passing base + one mutation; WMMA/EccHalf arms need other ISAs). Best taken as a dedicated campaign after the cheap self-contained wins. | multi-day |

**Why GlobalParameters now:** it is a self-contained ~1-day top-up to ≥95% with
clear, enumerable remaining branches (config dispatch + one monkeypatched
subprocess path), versus the open-ended multi-day `Solution` derivation sweep.
Bank it, then return to slice 3b as a focused config-matrix effort if the
`Solution.py` derivation coverage is prioritised.

### Effort estimate

`GlobalParameters.py` ~1 day: `assignGlobalParameters` over a few config dicts +
the real `isaInfoMap`; `printCapabilitiesTable(isaInfoMap)` snapshot;
`restoreDefaultGlobalParameters` round-trip; `setupRestoreClocks` with
`subprocess.run` monkeypatched (as `LibraryIO.getCUCount` did). A grounded API
inventory + BEFORE baseline at kickoff is in the companion
`next-goal-globalparameters.md`.
