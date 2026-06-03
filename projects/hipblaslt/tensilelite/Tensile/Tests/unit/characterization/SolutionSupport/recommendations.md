# Recommendations — after the `Solution.py` support slice

New file in the `SolutionSupport/` test dir per the add-only rule.

## Result for this target

The pure support slice (L165-439) went from largely-uncovered to **100% line
and branch** — the collector machinery, the arg dataclasses, and the index
helpers are fully pinned. Whole-file `Solution.py` moved 37.50% → 38.81% line
(the cap-coupled `Solution` class, ~3000 stmts, is out of slice). No regression:
full `-m unit` **1769 → 1800 passed** (+31), 201 skipped unchanged. Came in
under the ~0.5–1 day estimate.

## What worked (additions to the shared list)

- **importlib for shadowed submodules.** `SolutionStructs/__init__.py`
  re-exports the `Solution` class, shadowing the submodule, so the module must
  be loaded via `importlib.import_module(...)` — `import a.b.Solution as S`
  binds the class. Worth checking `__init__` re-exports before importing any
  `SolutionStructs` submodule.
- **Slice a megafile at the cap boundary.** `Solution.py`'s pure head (L165-439)
  was cleanly separable from the cap-coupled `Solution` class; characterizing it
  first banks 100% of a self-contained region and de-risks the campaign.
- **Reuse the collector-isolation pattern** (clear/capture/restore) for the
  module global shared with `ProblemType`.

## Go / no-go on the next target

### Verdict: **GO — `Solution.py` slice 2: `Solution` construction + Mapping/identity + the static `state`-helpers** (reusing the real `isaInfoMap`/`assembler` fixtures); `GlobalParameters.py` is the quicker alternative

The `Solution` class (L444-5230, ~3000 stmts) is the campaign's bulk. Approach
it construction-first:

| Candidate | Why / why not | Effort |
|---|---|---|
| **`Solution` slice 2 — construction happy-path + Mapping/identity + simple statics** ✅ chosen | `Solution.__init__(config, splitGSU, printSolutionRejectionReason, printIndexAssignmentInfo, assembler, isaInfoMap, srcName)` builds the whole solution. Reuse the LibraryIO/Validators conftest fixtures (`cxx_compiler` → `validateToolchain("amdclang++")`, `isa_info_map` → `makeIsaInfoMap(SUPPORTED_ISA, ...)`, `assembler`). Pin: a happy-path `Solution` from a small real config (normalised state snapshot, like the ProblemType suite), the `Mapping` dunders / `__hash__`/`__eq__`/`getSolutionNameFull` integration, and the simpler `@staticmethod` helpers that take `(state, isaInfoMap)` and aren't reject-heavy (`getMIOutputInfo`, `isDirectToVgprSupportDataType`, `getDivisorName`). Defer the deep cap-driven derivation (`assignDerivedParameters`, `setGlobalLoadTileDim*`, the `reject()`-heavy paths) to slice 3. | ~1.5–2.5 days |
| `Solution` slice 3 — parameter derivation / reject machinery | The reject-heavy assignment paths (hundreds of cap-gated branches). High effort, many states; needs a matrix of valid+invalid configs. After slice 2. | multi-day |
| `Common/GlobalParameters.py` (220 stmts, 84.19%) | Env-coupled (subprocess GPU clocks, `__version__`, mutable globals, `isaInfoMap`); a self-contained monkeypatch-heavy top-up. A good **quicker win** if a smaller next step is preferred over continuing the big `Solution` campaign. | ~1–1.5 day |

**Why slice 2 now:** it continues the `Solution.py` campaign with the highest-
value reachable surface (construction + the Mapping/name integration the rest of
the toolchain depends on) and reuses fixtures already built. If a shorter next
increment is wanted instead, take `GlobalParameters.py` (smaller, self-contained)
and return to the `Solution` class afterward.

### Effort estimate

Slice 2 ~1.5–2.5 days (cap fixtures + a normalised Solution-state snapshot +
the simpler statics). A grounded API inventory + BEFORE baseline at kickoff is
in the companion `next-goal-solution-class.md`.
