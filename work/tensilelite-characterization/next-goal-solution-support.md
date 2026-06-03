# Next goal — characterization tests for the `Solution.py` support-functions slice

Follow-up to the completed `Utilities.py`+`LdsPadding.py` suite. The **first
slice** of the large `Solution.py` campaign: its pure module-level head. Self-
contained kickoff; assumes the env/conventions of the eight completed targets
under `Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO,DataType,ValidParameters,ProblemType,Naming,SolutionStructsUtils}/`.

## GOAL

Add a characterization suite pinning the **pure support functions** of
`Tensile/SolutionStructs/Solution.py` (L165-439) to **≥95% line coverage on the
targeted symbols**, using syrupy (reuse `../survey.md`). Document resisting fns
+ a go/no-go on the next slice.

Achieved when:
- **BEFORE baseline captured FIRST** — `Solution.py` whole-file coverage from
  the existing suite — `coverage-before.txt`.
- `target.md` listing the targeted symbols and reporting file % + slice %
  (whole-file % will be low; demonstrate via the Missing column that residual
  misses are out-of-slice).
- `Tensile/Tests/unit/characterization/SolutionSupport/` with syrupy snapshots
  driving every targeted symbol.
- Determinism: the collector functions mutate a module global — clear/capture/
  restore (as the `ProblemType` validate test did). The arg dataclasses need a
  `ProblemType` (now pinned) + small configs; `printExit`→`SystemExit` paths via
  `pytest.raises`.
- `pytest --cov` (path-mode + grep) shows the slice symbols ≥95% line.
- **AFTER coverage captured** (`coverage-after.txt`) + delta in `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1769 passed / 201 skipped**; additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created (the cap-coupled `Solution` class slice,
  grounded).

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "SolutionStructs/Solution.py|passed"
```

> Slice target: whole-file `Solution.py` is ~33% and will stay low (the
> cap-coupled `Solution` class is out of slice). Report the file % AND show the
> targeted L165-439 lines are covered (Missing column). Document in target.md.

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**; new files only. **Commit atomically; never push.** Stay within
  `projects/hipblaslt/tensilelite`. BOUND: 120 turns / 120 min. AUTONOMY: don't
  ask; document blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work); rocisa
  baked. Install syrupy if the container was recreated.
- Run/snapshot/coverage via `docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite
  -w /work/projects/hipblaslt/tensilelite tl-char pytest ...`.
- **`--cov` takes a PATH (`Tensile`), never a dotted module** (rocisa SIGABRT).
- Snapshots written **inside the container** (root-owned); regenerate with
  `--snapshot-update` in-container.

## TARGET: `Solution.py` support slice (L165-439)

The head of `Solution.py` is **pure** (no caps/toolchain); the cap-coupled
`Solution` class begins at L444 and is OUT of this slice. Note the module-level
imports pull in `Assembler`/toolchain at import time, but that already works in
the env (the module is imported by the existing suite).

### Targeted symbols

| Symbol | LOC | Tier / nature |
|---|---|---|
| `_getExpectedTypes(validParams)` + `_expectedParamTypes` | 165-187 | A — build name→{types} map from `validParameters`; snapshot a normalised view (sorted {name: sorted type-names}). |
| `_skipTypeCheck` | 193-201 | A — constant set; snapshot sorted. |
| `resetTypeMismatchCollector` / `getTypeMismatchCollector` / `mergeTypeMismatchCollector` | 211-245 | A/B — the shared collector; clear/seed/merge and snapshot the resulting state. |
| `validateParameterTypes(state, srcFile)` | 248-285 | B — clean vs `bool`-where-`int` mismatch; isolate the collector (clear/capture/restore); also `srcFile=""` branch. |
| `printTypeMismatchSummary(numFiles)` | 288-339 | B — empty (returns 0, optional clean message) vs populated (returns total; capture stdout via `capsys`, snapshot the summary lines). |
| `Fbs` (Enum) | 342-345 | A — snapshot members/values. |
| `FactorDimArgs(problemType, config)` | 350-366 | B — needs a `ProblemType`; `UseScaleAlphaVec`/`UseBias` gating; bad-dim `printWarning`; snapshot `factorDims`/`totalProblemSizes`. |
| `BiasTypeArgs(problemType, config)` | 368-388 | B — `UseBias` gating; dtype-not-in-list `printWarning`; empty→`printExit` (`SystemExit`); snapshot biasTypes. |
| `activationSetting` / `ActivationArgs(problemType, config)` | 394-423 | B — `ActivationType` none early return; enum parsing; `all`/`hipblaslt_all` vs specific; `printExit` paths. |
| `isPackedIndex(ks, index)` | 427-429 | A — `index in ProblemType["IndicesFree"]`. |
| `isExtractableIndex(ks, index, tc)` | 431-439 | A — A/B/x cases over `PackedC0/C1IndicesX`. |

### Determinism / fixture plan

1. **Collector isolation.** `_typeMismatchCollector` is module-global and shared
   with `ProblemType` validation; clear → run → capture delta → restore in a
   `finally` (verified-safe pattern from the `ProblemType` suite).
2. **ProblemType fixtures.** Build via `ProblemType({"DataType":0, ...}, False)`
   with the flags each dataclass reads (`UseScaleAlphaVec`/`UseBias`/
   `ActivationType`/`BiasDataTypeList`). Reuse the `ProblemType` suite's minimal-
   config approach.
3. **`isExtractableIndex`/`isPackedIndex`** read `ks["ProblemType"]["IndicesFree"]`
   and `ks["PackedC0/C1IndicesX"]` — drive with a plain dict `ks` (no full
   Solution needed).
4. **printExit/printWarning** → `printExit` is `sys.exit(-1)` (`SystemExit`);
   `printWarning` prints. Snapshot returns/state; pin `SystemExit` via
   `pytest.raises`; capture stdout with `capsys` where asserting messages.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/SolutionSupport/`, `pytestmark =
  pytest.mark.unit`. New `target.md`/`resistance.md`/`recommendations.md`/
  `coverage-before.txt`/`coverage-after.txt`; snapshots in `__snapshots__/`. A
  `conftest.py` with a `ProblemType` builder if it helps.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` + `target.md`.
2. `_getExpectedTypes`/`_skipTypeCheck`/collector machinery/`validateParameterTypes`/
   `printTypeMismatchSummary` + `Fbs`/`isPackedIndex`/`isExtractableIndex`.
3. The arg dataclasses (`FactorDimArgs`/`BiasTypeArgs`/`ActivationArgs`).
4. resistance.md + recommendations.md.
5. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
6. `next-goal-<target>.md` (the cap-coupled `Solution` class slice, grounded) +
   commit. Mark this checklist done.

## Definition of done checklist — COMPLETE

- [x] `coverage-before.txt` captured (baseline) BEFORE new tests.
      (whole-file 3272/2045 = 37.50% line; slice symbols largely uncovered.)
- [x] Slice symbols ≥95% line — **100% line+branch on L165-439** (AFTER Missing
      column's first entry is L473, inside the out-of-slice Solution class).
- [x] `coverage-after.txt` captured; before→after delta in target.md.
- [x] Every targeted Tier A/B symbol has snapshot coverage (collector machinery,
      validate/print, Fbs, the 3 arg dataclasses, index helpers).
- [x] Collector isolated (clear/capture/restore); `printExit` paths pinned via
      `pytest.raises(SystemExit)`.
- [x] resistance.md: nothing in-slice resisted; documents the cap-boundary
      scoping + the `importlib` submodule-shadowing gotcha.
- [x] recommendations.md: GO → Solution class slice 2 (construction + Mapping +
      simple statics, with isaInfoMap/assembler fixtures); GlobalParameters as
      the quicker alternative.
- [x] full `-m unit` = **1800 passed / 201 skipped**, no failures, additive only.
- [x] All work committed (atomic, no push); tree clean.
- [x] Next goal prompt `next-goal-solution-class.md` created and committed.
