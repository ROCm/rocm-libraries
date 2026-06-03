# Next goal — characterization tests for the `Problem.py` ProblemType slice

Follow-up to the completed `ValidParameters.py` suite. Same pattern, next
module — but **scoped to a slice** of a large file. This file is the
self-contained kickoff; it assumes the env and conventions already established
(see `prompt.md`, `env/README.md`, `HANDOFF.md`, and the five completed targets
under `Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO,DataType,ValidParameters}/`).

## GOAL

Add a characterization-test suite that pins current behaviour of a **slice** of
`Tensile/SolutionStructs/Problem.py` — the `ProblemType` type and its immediate
collaborators — with **≥95% line coverage on the *targeted symbols*** (NOT the
whole 1382-LOC file; see "Scope" — report slice coverage explicitly), using
syrupy (reuse `Tensile/Tests/unit/characterization/survey.md`, no new survey).
Document resisting functions + a go/no-go on the *next* target.

Achieved when:
- **Baseline (BEFORE) coverage captured FIRST** — `Problem.py` whole-file
  coverage from the *existing* suite — saved as `coverage-before.txt`.
- `target.md` for the ProblemType slice (new file in the test dir, per
  add-only) that **lists the targeted symbols** and reports coverage for the
  file overall AND a per-symbol view of the slice (so "≥95% on the slice" is
  auditable even though the file has out-of-slice code).
- `Tensile/Tests/unit/characterization/ProblemType/` exists with syrupy
  snapshots driving every reasonably-testable targeted symbol (see inventory).
- Determinism handled: `ProblemType` is a `Mapping` whose construction derives
  dtype fields via `DataType` and assigns indices — snapshot a **normalised**
  view (sorted items; dtype fields rendered as names/enums, not live objects),
  never the live object or any path/version.
- `pytest --cov` (path-mode + grep) reports the `Problem.py` row; the slice
  symbols reach **≥95% line** (verify via `--cov-report=term-missing` that the
  remaining misses are all outside the targeted symbols, and list them).
- **Final (AFTER) coverage captured** (`coverage-after.txt`) with before→after
  delta recorded in `target.md`.
- `resistance.md` (new file) lists every resisting fn + reason + workaround
  (expect real resistance here — see "Determinism / resistance").
- `recommendations.md` (new file): go/no-go on the *following* target (the rest
  of `Problem.py` / `Solution.py` core vs `Common/GlobalParameters.py`) +
  effort estimate.
- No regression: full `-m unit` still passes (current baseline after the
  ValidParameters suite = **1563 passed / 201 skipped**; this work only adds).
- **The work is committed** (atomic commits; tree clean; no push).
- **The next goal prompt is created**: `next-goal-<target>.md`, grounded in
  inspection of that module.

## Coverage reports (BEFORE / AFTER) — required deliverable

`--cov` on a single .py path-prefix does NOT match; use `--cov=Tensile` + grep
the row, as every prior target did.

```
# BEFORE (existing tests only) and AFTER (suite complete):
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "SolutionStructs/Problem.py|passed"
```

> Because this is a slice of a large file, the whole-file `Cover %` will be
> below 95% even when the slice is fully covered. Record the file % AND
> demonstrate slice coverage from the `Missing` column (every remaining missed
> line is outside the targeted symbols). Document this explicitly in target.md.

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**: only add new files. Never modify/delete any existing file.
  Keep `target.md`/`resistance.md`/`recommendations.md` as **new files in the
  per-target test dir**.
- **Commit frequently**, atomic commits. **Never push / no PRs.** Local only.
- rocm-libraries boundary: stay within `projects/hipblaslt/tensilelite`.
- Snapshot normalised structural views; no live objects / paths / versions.
- BOUND: 180 turns / 180 min. AUTONOMY: don't ask; document blockers, continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work).
  rocisa built (`LD_LIBRARY_PATH=/opt/rocm/lib`, baked).
- **Install syrupy if the container was recreated** (`pip install syrupy`).
- Run/snapshot/coverage via `docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite
  -w /work/projects/hipblaslt/tensilelite tl-char pytest ...` (full suite
  baseline+after each ~110s; suite alone seconds).
- **Coverage gotcha: pass `--cov` a PATH (`Tensile`), never a dotted module** —
  a dotted name re-imports rocisa (nanobind) → SIGABRT.
- The LibraryIO suite vendored a real logic fixture
  (`LibraryIO/data/logic_gfx942_HSS_BH.yaml`) whose `data[5]` is a ProblemType
  state block — reuse it (read-only) for a realistic ProblemType config.

## TARGET: `Tensile/SolutionStructs/Problem.py` — ProblemType slice

**Clean imports** — `rocisa.enum.DataTypeEnum`, `collections(.abc)`,
`Tensile.Activation.ActivationType`, `Common.fastdeepcopy`,
`Common.Constants.INDEX_CHARS`, `Common.DataType.DataType`,
`Common.Utilities` helpers. **No** `isaInfoMap`/`asmCaps`/`assembler`/
`subprocess`/toolchain anywhere in the file (verified). So the slice is
pure-Python over dicts + `DataType` (now fully pinned) + `ActivationType`.

### Scope (targeted symbols) + inventory

| Symbol | LOC area | Tier / nature |
|---|---|---|
| `getRealDataTypeA` / `getRealDataTypeB` | 377-406 | A — pure dtype mappers (mirror the LibraryIO pair); snapshot across dtype combos. |
| `_defaultProblemType` (+ `_expectedProblemTypeParamTypes`) | 408-…, 755-757 | A — registry dict; snapshot `sorted(keys())` + the derived expected-types map. |
| `problemTypeToEnum(problemType)` | 711-750 | A — in-place converts DataType-valued fields → enum `.value`; incl. the `DataTypeMetadata`/`MXSA`/`MXSB` present/absent branches (L738-750). Build a dict of `DataType` objects, run, snapshot the result. |
| `validateProblemTypeParameterTypes(state, srcFile)` | 760-799 | B — type checker writing into Solution's `_typeMismatchCollector` (imported inside the fn → circular-dep guard). Snapshot the collector delta for a clean state vs a `bool`-where-`int` mismatch; reset the collector between cases. |
| `ProblemType` (Mapping) | 802-1372 | B/C — `FromDefaultConfig`, `__init__`, `isGEMM`/`initGEMM`/`_checkIfSupportedGEMMType`, `assignDerivedParameters` (the heavy index-assignment path — likely partial resistance), the Mapping dunders (`keys`/`__len__`/`__iter__`/`__getitem__`/`__setitem__`/`get`), `__str__`/`__repr__`, `getAttributes`/`__hash__`/`__eq__`/`__ne__`. Construct from `_defaultProblemType` and from the vendored fixture's ProblemType block; snapshot a normalised Mapping view. |
| `ProblemSizeRange` | 39-171 | B — parses a sizes config against a ProblemType; snapshot `__str__` + state for a small config. |
| `Problem` / `ExactList` / `ExactDict` | 173-269 | B — exact-problem holders; `ExactList.convertLeadingDims` is pure-ish. Snapshot `__str__`/state for small inputs. |
| `ProblemSizesMock` / `ProblemSizesMockDummy` | 271-278 | A — trivial holders; construct + snapshot. |
| `ProblemSizes` | 279-376 | B — aggregates ranges/exacts; snapshot summary for a small config. |
| `getBiasDataTypeListDefault` | 1372-end | A — pure default list from a ProblemType. |

### Determinism / resistance plan

1. **Normalise the Mapping.** Snapshot `dict(sorted(pt.items()))` with dtype
   fields mapped to `DataType(...).toName()` / enum names — never the live
   `ProblemType`/`DataType` objects.
2. **`assignDerivedParameters` is the hard part.** Index assignment, GEMM-type
   support checks, and tensor/bias handling have many branches; some require a
   self-consistent operation/index config. Cover what a real config
   (default + vendored fixture) reaches; **document the rest in resistance.md**
   rather than forcing synthetic states (mirrors the LibraryIO approach).
3. **`validateProblemTypeParameterTypes` collector.** It mutates a module-level
   dict in `Solution`. Import + clear it around each case so snapshots are
   deterministic and the shared session is unaffected (verify full `-m unit`
   afterward, as the LibraryIO import-fallback tests did).
4. **`problemTypeToEnum` mutates in place** — pass a fresh dict each call.
5. **No env coupling** — no paths/time/version/GPU in this slice.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/ProblemType/`, files marked
  `pytestmark = pytest.mark.unit`. Collected by existing `testpaths` (no edit).
- `target.md`/`resistance.md`/`recommendations.md`/`coverage-before.txt`/
  `coverage-after.txt` as **new files in that dir**; snapshots in
  `ProblemType/__snapshots__/`.
- A `conftest.py` only if a shared ProblemType-building helper emerges (likely:
  a `_normalize(pt)` helper + a fixture that loads the vendored logic
  fixture's ProblemType block).

## Suggested commit sequence (atomic)

1. `coverage-before.txt` (baseline) + `target.md`.
2. Pure pieces: `getRealDataType{A,B}`, `_defaultProblemType` roster,
   `problemTypeToEnum`, `getBiasDataTypeListDefault`, the Mock holders.
3. `ProblemType` construction + Mapping dunders + equality/hash (default config
   + vendored fixture), normalised snapshots.
4. `validateProblemTypeParameterTypes` (clean + mismatch, collector reset).
5. `ProblemSizeRange`/`Problem`/`ExactList`/`ExactDict`/`ProblemSizes` on small
   configs.
6. resistance.md + recommendations.md.
7. Final coverage run + `coverage-after.txt` + no-regression confirmation.
8. Create `next-goal-<target>.md` (grounded) + commit. Mark this checklist done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) and committed BEFORE new tests.
- [ ] Slice symbols ≥95% line (file % will be lower — demonstrate via Missing
      column that residual misses are out-of-slice; document in target.md).
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Every targeted Tier A/B symbol has snapshot coverage; Tier C documented.
- [ ] `ProblemType` snapshots are normalised (no live objects/paths/versions).
- [ ] resistance.md lists each resisting fn (expect: deep `assignDerivedParameters`
      branches needing self-consistent op/index configs).
- [ ] recommendations.md: go/no-go on the next target (rest of Problem/Solution
      core vs `Common/GlobalParameters.py`).
- [ ] full `-m unit` ≥ 1563 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
