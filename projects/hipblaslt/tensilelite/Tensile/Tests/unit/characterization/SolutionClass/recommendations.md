# Recommendations — after `Solution` class slice 2

New file in the `SolutionClass/` test dir per the add-only rule.

## Result for this target

The targeted slice-2 **surface** symbols are fully covered: a real `Solution`
construction (normalised-state snapshot), the Mapping interface, identity /
hash / equality, and the simple statics (`getMIOutputInfo`,
`isDirectToVgprSupportDataType`, `getDivisorName`) + `getKernels`. The only
remaining targeted line is the dead `__ne__` NotImplemented arm (L5229).
Whole-file `Solution.py` moved 38.81% → 39.55% line (the derivation campaign is
slice 3). No regression: full `-m unit` **1800 → 1818 passed** (+18), 201
skipped unchanged.

## What worked (additions to the shared list)

- **Parse a real fixture into Solution objects instead of hand-authoring a
  config.** Reusing the LibraryIO cap fixtures + vendored logic fixture yields
  genuine `Solution`s, so the Mapping/identity/static surface is exercised
  against real state without constructing a self-consistent kernel by hand.
- **Snapshot schema + stable fields for big toolchain-derived state.** A 306-key
  state with cap-derived values is pinned by its sorted key set + a curated set
  of stable scalar fields — reproducible, reviewable.
- **Characterize pipeline-dependent accessors as their current error.**
  `getKernelBetaOlnyObjects`/`getKernelConversionObjects` raise on a freshly
  parsed Solution; pinned via `pytest.raises(AttributeError)`.

## Go / no-go on the next target

### Verdict: **GO — `Solution` slice 3: the parameter-derivation / reject machinery** (or `GlobalParameters.py` as a smaller intermediate win)

| Candidate | Why / why not | Effort |
|---|---|---|
| **`Solution` slice 3 — derivation statics** ✅ chosen | The bulk of the remaining `Solution.py` Missing lines: `assignProblemIndependentDerivedParameters`, `assignDerivedParameters`, `setGlobalReadVectorWidth`, `setGlobalLoadTileDimClassic`, `checkAndAssignWaveSeparateGlobalRead`, `isDirectToVgpr/LdsDoable`, `depthUIteration`, `_deriveAndValidateMXScaleLayoutAndTransport`. These are cap-gated and `reject()`-heavy: drive them with a **matrix of configs** (dtype × MI × transpose × the relevant flags) built on the vendored fixture's solution state, snapshotting the derived sub-state and the reject outcomes (`printSolutionRejectionReason=False`, `state["Valid"]`). Largest remaining coverage block. | multi-day (several test files; config matrix) |
| `Common/GlobalParameters.py` (220 stmts, 84.19%) | Self-contained env-coupled top-up (subprocess GPU clocks, `__version__`, mutable globals, `isaInfoMap`); a **good smaller next increment** before the big derivation push. | ~1–1.5 day |

**Why slice 3 (or GlobalParameters first):** slice 3 is where the remaining
`Solution.py` coverage lives, but it is a large, reject-branch-heavy effort
needing a config matrix and careful per-static state setup. If a shorter,
self-contained increment is preferred next, take `GlobalParameters.py`
(monkeypatch the subprocess/clock paths) and return to slice 3 afterward —
both are queued; `next-goal-solution-derivation.md` covers slice 3.

### Effort estimate

Slice 3 is multi-day: each derivation static has many cap-gated branches; expect
several test files and a reject-outcome matrix. Approach static-by-static,
seeding each from the vendored fixture's state and toggling one flag/dtype at a
time. A grounded API inventory + BEFORE baseline at kickoff is in the companion
`next-goal-solution-derivation.md`.
