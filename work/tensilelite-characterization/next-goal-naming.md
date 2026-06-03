# Next goal — characterization tests for `Tensile/SolutionStructs/Naming.py`

Follow-up to the completed `Problem.py` ProblemType slice. Same pattern, next
module. Self-contained kickoff; assumes the env and conventions of the six
completed targets under
`Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO,DataType,ValidParameters,ProblemType}/`.

## GOAL

Add a characterization-test suite pinning `Tensile/SolutionStructs/Naming.py`
with **≥95% line coverage on the module**, using syrupy (reuse
`../survey.md`). Document resisting functions + a go/no-go on the next target.

Achieved when:
- **BEFORE baseline captured FIRST** (`coverage-before.txt`).
- `target.md` (new file in the test dir, per add-only).
- `Tensile/Tests/unit/characterization/Naming/` with syrupy snapshots driving
  every reasonably-testable function (see inventory).
- Determinism handled: name strings are pure functions of the input state;
  snapshot the returned strings. Any solution state used is fixed/vendored.
- `pytest --cov` (path-mode + grep) reports **≥95% line** on the module.
- **AFTER coverage captured** (`coverage-after.txt`) + delta in `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1672 passed / 201 skipped** (post-ProblemType
  baseline); additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created, grounded in inspection.

## Coverage command (path-mode + grep, as every prior target)

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "SolutionStructs/Naming.py|passed"
```

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**; new files only. **Commit atomically; never push.** Stay within
  `projects/hipblaslt/tensilelite`. BOUND: 120 turns / 120 min. AUTONOMY: don't
  ask; document blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work);
  rocisa baked. Install syrupy if the container was recreated.
- Run/snapshot/coverage via `docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite
  -w /work/projects/hipblaslt/tensilelite tl-char pytest ...`.
- **`--cov` takes a PATH (`Tensile`), never a dotted module** (rocisa SIGABRT).
- Snapshots are written **inside the container** (root-owned); regenerate with
  `--snapshot-update` in-container rather than editing `.ambr` by hand.

## TARGET: `Tensile/SolutionStructs/Naming.py` (239 LOC)

**Clean imports** — `functools.lru_cache`, `Common.Constants.MAX_FILENAME_LENGTH`,
`Common.RequiredParameters.getRequiredParametersMin/Full`,
`.Problem.ProblemType`. No toolchain/GPU/subprocess. Turns a solution/kernel
`state` dict into name strings. `ProblemType` is already pinned (prior target),
so a `ProblemType` is cheap to build.

### Public API inventory (drive each; note the tier)

| Symbol | LOC | Tier / nature |
|---|---|---|
| `getParameterNameAbbreviation(name)` | 100-102 | A — uppercase-letters abbrev; `lru_cache`; snapshot across sample names. |
| `getPrimitiveParameterValueAbbreviation(key, value)` | 105-125 | A — per-type abbrev (str/bool/int≥0/int<0/ProblemType/float incl. the `p`-fraction path); parametrise every type branch. |
| `getParameterValueAbbreviation(key, value)` | 128-141 | A — ISA tuple, scalar, tuple, list, dict, and the new-object `raise`; parametrise all + a bad type for the raise. |
| `_getName(state, requiredParameters, splitGSU, ignoreInternalArgs)` | 144-205 | B — the core builder. Branches: `CustomKernelName` early return; `ignoreInternalArgs` GSU rewrite; MacroTile/DepthU block; MatrixInst vs ThreadTile; `UseCustomMainLoopSchedule`; SFA-skip; the required-parameter loop. Drive via a built solution state (below). |
| `getKeyNoInternalArgs(state, splitGSU)` | 47-99 | B — the other key builder (inspect for its own branch set; likely parallels `_getName`). |
| `shortenFileBase(splitGSU, kernel)` | 208-219 | A/B — returns base, or a sha256/base64-shortened name when `len > MAX_FILENAME_LENGTH`; drive both (a short kernel and one with enough params to exceed the limit). |
| `getKernelFileBase(splitGSU, kernel)` | 222-227 | A — CustomKernelName vs `shortenFileBase`. |
| `getKernelNameMin` / `getSolutionNameMin` / `getSolutionNameFull` | 230-239 | A — thin wrappers over `_getName` with Min/Full required-param sets; snapshot each. |

### Determinism / fixture plan

1. **Build a solution state, or vendor one.** `_getName` reads `state["ProblemType"]`
   (a `ProblemType` or a config dict — it will construct one if given a dict),
   `state["GlobalSplitU"]`, `state["UseCustomMainLoopSchedule"]`, and optional
   `MacroTile0/1`, `DepthU`, `MatrixInstM/N/B` plus the required-parameter keys
   (missing keys are skipped via `if key not in state: continue`). Two routes:
   (a) build a minimal state = `{"ProblemType": ProblemType({"DataType":0}, False),
   "GlobalSplitU": 1, "UseCustomMainLoopSchedule": False, ...}` and add
   MacroTile/MatrixInst keys to walk those branches; or (b) vendor a real
   solution state from the LibraryIO fixture
   (`LibraryIO/data/logic_gfx942_HSS_BH.yaml`, the solutions block) for a
   realistic full name. Prefer (a) for control, add (b) for a realistic check.
2. **`lru_cache`** on the abbrev helpers is snapshot-safe.
3. **`_getName` mutates then restores** `state["GlobalSplitU"]` and
   `state["ProblemType"]["GroupedGemm"]`; pass a fresh state per call (or assert
   the restore) so tests don't interfere.
4. **`shortenFileBase` long path** uses sha256+base64 of the *tail* — fully
   deterministic given a fixed kernel; snapshot the resulting string.
5. No paths/time/version/GPU — no normalisation needed beyond fixing the state.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/Naming/`, `pytestmark =
  pytest.mark.unit`. `target.md`/`resistance.md`/`recommendations.md`/
  `coverage-before.txt`/`coverage-after.txt` as new files; snapshots in
  `Naming/__snapshots__/`. A `conftest.py` with a solution-state builder if it
  helps (a `ProblemType`-backed state factory).

## Suggested commit sequence (atomic)

1. `coverage-before.txt` + `target.md`.
2. The pure abbrev helpers (`getParameterNameAbbreviation`,
   `getPrimitiveParameterValueAbbreviation`, `getParameterValueAbbreviation`
   incl. the raise).
3. `_getName` / `getKeyNoInternalArgs` over built solution states (Min/Full,
   custom-kernel, MacroTile/MatrixInst, ignoreInternalArgs both ways).
4. `shortenFileBase` (short + long), `getKernelFileBase`, the three name wrappers.
5. resistance.md + recommendations.md.
6. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
7. `next-goal-<target>.md` (grounded) + commit. Mark this checklist done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) and committed BEFORE new tests.
- [ ] `--cov=Tensile` + grep `Naming.py` row ≥95% line.
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Every Tier A/B function has snapshot coverage; Tier C documented.
- [ ] Raise / shorten-long / custom-kernel paths pinned.
- [ ] resistance.md lists each resisting fn.
- [ ] recommendations.md: go/no-go on the next target (a `Solution.py` slice vs
      `Utilities.py`+`LdsPadding.py` top-up vs `GlobalParameters.py`).
- [ ] full `-m unit` ≥ 1672 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
