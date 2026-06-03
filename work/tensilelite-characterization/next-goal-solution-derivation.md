# Next goal — characterization tests for the `Solution` derivation (slice 3)

Follow-up to the completed `Solution` class slice 2. **Slice 3 of the
`Solution.py` campaign**: the parameter-derivation / reject machinery — the
largest remaining coverage block. Self-contained kickoff; assumes the
env/conventions of the ten completed targets under
`Tensile/Tests/unit/characterization/`.

> This is a LARGE, reject-branch-heavy slice. Expect multiple test files and a
> config matrix. Approach static-by-static. If a smaller next increment is
> preferred, `GlobalParameters.py` (see that module's row in the SolutionClass
> recommendations) is a self-contained alternative — but this file covers slice 3.

## GOAL

Add characterization tests for the `Solution` **derivation** statics, raising
their coverage as far as practical toward **≥95% line on the targeted statics**
(report file % + per-static slice view; some deeply cap-gated reject arms may
resist and are documented). syrupy; reuse `../survey.md`.

Achieved when:
- **BEFORE baseline captured FIRST** (`coverage-before.txt`, whole-file row).
- `target.md` listing the targeted statics + per-static coverage method.
- `Tensile/Tests/unit/characterization/SolutionDerivation/` with snapshots.
- Determinism: seed each static from the vendored fixture's solution **state**
  (a dict), toggle one dtype/flag at a time; `printSolutionRejectionReason=False`
  for quiet reject; snapshot the derived sub-state + reject outcome
  (`state["Valid"]`), not live objects / env-coupled values.
- `pytest --cov` shows the targeted statics ≥95% line (or document the
  unreachable cap-gated arms in resistance.md).
- **AFTER coverage** (`coverage-after.txt`) + delta in `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1818 passed / 201 skipped**; additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created (the remaining Solution.py surface, or
  `GlobalParameters.py` / the Toolchain group — grounded).

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "SolutionStructs/Solution.py|passed"
```

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**; new files only. **Commit atomically; never push.** Stay within
  `projects/hipblaslt/tensilelite`. BOUND: 180 turns / 180 min. AUTONOMY: don't
  ask; document blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work); rocisa
  baked. Install syrupy if the container was recreated.
- Run/snapshot/coverage via `docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite
  -w /work/projects/hipblaslt/tensilelite tl-char pytest ...`.
- **`--cov` takes a PATH (`Tensile`), never a dotted module** (rocisa SIGABRT).
- Snapshots written **inside the container** (root-owned); regenerate with
  `--snapshot-update` in-container.
- **Cap fixtures + real state:** copy `SolutionClass/conftest.py` — it builds
  `isa_info_map`/`assembler` and parses the vendored fixture into `Solution`
  objects. Use `solution._state` as the seed dict for the statics.
- **Import gotcha:** load the module via
  `importlib.import_module("Tensile.SolutionStructs.Solution")` for statics, and
  `from ...Solution import Solution` for the class.

## TARGET: `Solution` derivation statics

All are `@staticmethod` taking a mutable `state` dict (+ `isaInfoMap` and/or
`printRejectionReason`). They MUTATE `state` and call `reject()` on invalid
configs (`reject` sets `state["Valid"]=False` and returns True when
`printSolutionRejectionReason=False`).

| Static | Signature | Notes |
|---|---|---|
| `getMIOutputInfo` | `(state, isaInfoMap)` | already pinned in slice 2 — skip or extend with dtype variants |
| `assignProblemIndependentDerivedParameters` | `(state, printRejectionReason, isaInfoMap)` | problem-independent derivation; many branches |
| `setGlobalReadVectorWidth` | `(state, tc, totalVectors, grvw, printRejectionReason)` | GRVW assignment + reject |
| `setGlobalLoadTileDimClassic` | `(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp, depthU, printRejectionReason)` | tile-dim assignment |
| `checkAndAssignWaveSeparateGlobalRead` | `(state, tc, printRejectionReason)` | wave-separate GRead |
| `isVgprForLocalReadPackingDoable` | `(state, isaInfoMap)` | predicate |
| `isDirectToVgprSupportDataType` | `(state)` | pinned in slice 2 |
| `isDirectToVgprDoable` | `(state, tc, printRejectionReason, isaInfoMap)` | reject-heavy predicate |
| `isDirectToLdsDoable` | `(state, tc, isaInfoMap, printRejectionReason)` | reject-heavy predicate |
| `assignDerivedParameters` | `(state, ...)` | the big one — full derivation; many cap-gated rejects |
| `depthUIteration` | `(...)` | depthU search |
| `_deriveAndValidateMXScaleLayoutAndTransport` | `(state, asmCaps, archCaps, printRejectionReason)` | MX-scale layout (module-level fn, L57-162) |

### Determinism / approach

1. **Seed from real state.** `state = copy.deepcopy(solution._state)` from the
   vendored fixture; toggle one field (dtype / MI / transpose / GRVW) per test.
2. **Reject outcomes.** Call with `printRejectionReason=False`; snapshot
   `{returned, state["Valid"], <selected derived keys>}`. Avoid setting a valid
   `SolutionIndex` (so `reject` doesn't raise — see `Utilities.reject`).
3. **Static-by-static.** Start with the predicates (`isDirectToVgpr/LdsDoable`,
   `isVgprForLocalReadPackingDoable`) — smaller; then the setters
   (`setGlobalReadVectorWidth`, `setGlobalLoadTileDimClassic`); then
   `assignProblemIndependentDerivedParameters`; finally `assignDerivedParameters`
   (the largest — may need several configs and may leave documented reject arms).
4. **Snapshot derived sub-state**, never the whole state or env-coupled values.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/SolutionDerivation/`, split into a
  few files by static group; `pytestmark = pytest.mark.unit`. Copy the
  SolutionClass conftest fixtures. New `target.md`/`resistance.md`/
  `recommendations.md`/`coverage-before.txt`/`coverage-after.txt`; snapshots in
  `__snapshots__/`.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` + `target.md` + `conftest.py`.
2. The predicates (`isDirectToVgpr/LdsDoable`, `isVgprForLocalReadPackingDoable`).
3. The setters (`setGlobalReadVectorWidth`, `setGlobalLoadTileDimClassic`,
   `checkAndAssignWaveSeparateGlobalRead`).
4. `assignProblemIndependentDerivedParameters` + `depthUIteration` +
   `_deriveAndValidateMXScaleLayoutAndTransport`.
5. `assignDerivedParameters` (config matrix).
6. resistance.md + recommendations.md.
7. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
8. `next-goal-<target>.md` (grounded) + commit. Mark this checklist done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) BEFORE new tests.
- [ ] Targeted statics ≥95% line where reachable; unreachable cap-gated arms documented.
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Each targeted static has snapshot coverage of its main + reject paths.
- [ ] State seeded from the real fixture; snapshots normalised (no env-coupled values).
- [ ] resistance.md lists each resisting fn (expect: deeply cap-gated reject arms).
- [ ] recommendations.md: go/no-go on the next target (remaining Solution.py vs
      `GlobalParameters.py` vs the Toolchain group).
- [ ] full `-m unit` ≥ 1818 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
