# Next goal — characterization tests for `Tensile/Common/DataType.py`

Follow-up to the completed `LibraryIO.py` suite. Same pattern, next module.
This file is the self-contained kickoff; it assumes the env and conventions
already established (see `prompt.md`, `env/README.md`, `HANDOFF.md`, and the
three completed targets under
`Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO}/`).

## GOAL

Add a characterization-test suite that pins current behaviour of
`Tensile/Common/DataType.py` with **≥95% line coverage on the module**, using
syrupy (reuse `Tensile/Tests/unit/characterization/survey.md`, no new survey),
integrated with the existing pytest/tox setup. Document resisting functions +
a go/no-go on the *next* target after this one.

Achieved when:
- **Baseline (BEFORE) coverage captured FIRST** — `DataType.py` coverage from
  the *existing* suite — saved as `coverage-before.txt` (command below).
- `target.md` for DataType (new file in the DataType test dir, per add-only).
- `Tensile/Tests/unit/characterization/DataType/` exists with syrupy snapshots
  driving every reasonably-testable public method (see API inventory).
- Determinism handled: outputs are pure functions of the dtype + arguments; no
  filesystem, no toolchain, no GPU. Snapshot returns directly.
- `pytest --cov` (path-mode + grep) reports **≥95% line** on the module.
- **Final (AFTER) coverage captured** (`coverage-after.txt`) with before→after
  delta recorded in `target.md`.
- `resistance.md` (new file) lists every resisting fn + reason + workaround.
- `recommendations.md` (new file): go/no-go on the *following* target
  (`Common/ValidParameters.py` vs a `SolutionStructs/Problem.py` ProblemType
  slice) + effort estimate.
- No regression: full `-m unit` still passes (current baseline after the
  LibraryIO suite = **1443 passed / 201 skipped**; this work only adds).
- **The work is committed** (atomic commits; tree clean; no push).
- **The next goal prompt is created**: `next-goal-<target>.md` for the target
  chosen in recommendations.md, grounded in inspection of that module.

## Coverage reports (BEFORE / AFTER) — required deliverable

`--cov` on a single .py path-prefix does NOT match (coverage warns
"module-not-imported / no data"); use `--cov=Tensile` + grep the row, exactly
as the LibraryIO target did (see its `coverage-before.txt`).

```
# BEFORE (existing tests only) and AFTER (suite complete):
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "DataType.py|passed"
```

> The Common/ tree has multiple `DataType`-named rows guarded by path; confirm
> you grep the `Tensile/Common/DataType.py` row specifically.

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**: only add new files. Never modify/delete any existing file.
  Anything needing an edit → new file or document as a limitation. Keep
  `target.md`/`resistance.md`/`recommendations.md` as **new files in the
  per-target test dir**, not edits to the shared ones.
- **Commit frequently**, atomic commits. **Never push / no PRs.** Local only.
- rocm-libraries boundary: stay within `projects/hipblaslt/tensilelite`.
- Snapshot pure returns; no nondeterministic blobs (there are none here).
- BOUND: 120 turns / 120 min. AUTONOMY: don't ask for confirmation; document
  blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work).
  rocisa built (`LD_LIBRARY_PATH=/opt/rocm/lib`, baked).
- **Install syrupy if the container was recreated** (`pip install syrupy`).
- Resume: see `HANDOFF.md`.
- **Coverage gotcha: pass `--cov` a PATH (`Tensile`), never a dotted module** —
  a dotted name re-imports rocisa (nanobind) → SIGABRT.
- Line coverage = `(Stmts - Miss)/Stmts` (report both line and blended).

## TARGET: `Tensile/Common/DataType.py` (1 file, 161 stmts)

A single `DataType` class wrapping a static `properties` table (one dict per
dtype: `enum`/`char`/`nameAbbrev`/`miOutTypeNameAbbrev`/`reg`/`hip`/`isComplex`),
plus a module-level `_populateLookupTable`. Almost entirely **pure** — no I/O,
no toolchain, no GPU. Import-only baseline is ~49% blended (161 stmts, 79
missing); the gap is exactly the predicate/converter methods, which snapshot
trivially. `rocisa.enum.DataTypeEnum` is the only dependency (already built).

### Public API inventory (drive each; note the tier)

| Symbol | LOC area | Tier / nature |
|---|---|---|
| `DataType.__init__` (enum / int / str / DataType / invalid) | 271-285 | A — 4 accepted input forms + the `RuntimeError` else; snapshot `.value` + raises |
| `toChar`/`toName`/`toNameAbbrev`/`toEnum` | 286-293 | A — table lookups; snapshot across all dtypes |
| `toDevice(language)` | 294-299 | A — `"HIP"` returns hip name; **else `assert 0`** → snapshot raises for non-HIP (resistance: the assert) |
| `zeroString(language, vectorWidth)` | 301-327 | A — string builder; parametrise vectorWidth 1 / >1 |
| `is*` predicates (~70 methods: isReal/isComplex/isHalf/isAnyFloat8/is8bitFloat/is6bitFloat/isFloat4/…A/…B variants) | 329-439 | A — pure booleans; **parametrise over all dtypes × all predicates** and snapshot the matrix |
| `isNone` | 438-439 | A — guard for the None/unknown dtype |
| `numRegisters`/`numBytes`/`MIOutputTypeNameAbbrev`/`flopsPerMac` | 441-448 | A — numeric/string table reads; snapshot across dtypes |
| `state`/`__str__`/`__repr__`/`getAttributes` | 450-458 | A — snapshot string/dict forms |
| `__hash__`/`__eq__`/`__lt__` | 460-473 | A — equality/ordering; snapshot pairwise over a few dtypes (note `__eq__` vs non-DataType) |
| `_populateLookupTable(properties, lookup)` | 477-490 | A/B — builds the str→index map; has a **duplicate-key guard** (L488-489) — snapshot the built table; the dup-raise may RESIST (table has no dups) → document |

### Determinism plan

1. **Enumerate dtypes from the table.** Iterate
   `DataType.properties` (or the `DataTypeEnum` members backing it) to build the
   parametrize list, so the suite tracks the table automatically. Snapshot a
   `{dtype_name: result}` dict per method (sorted keys) — one snapshot per
   method instead of one per dtype keeps the `.ambr` reviewable.
2. **Invalid inputs.** `__init__` with an unsupported type → `RuntimeError`;
   `toDevice("HLSL")` → `assert 0` (`AssertionError`). Snapshot via
   `pytest.raises`.
3. **`_populateLookupTable` dup guard.** The real table has no duplicate
   lookup keys, so the raise (L488-489) is likely unreachable → feed a tiny
   synthetic `properties` list with a dup to cover it, else document as
   resistance.
4. **No normalisation needed** — every output is a pure function of the dtype
   and arguments (no paths, versions, timestamps, or env coupling).

### Location & layout (consistent with prior targets)

- Suite: `Tensile/Tests/unit/characterization/DataType/`, files marked
  `pytestmark = pytest.mark.unit`. Collected by the existing
  `testpaths=Tensile/Tests` — no config edit.
- `target.md`, `resistance.md`, `recommendations.md`, `coverage-before.txt`,
  `coverage-after.txt` as **new files in that dir**; snapshots in
  `DataType/__snapshots__/` via `--snapshot-update`.
- No `conftest.py` needed (no toolchain/caps fixtures) unless a helper emerges.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` (baseline) + `target.md`.
2. Constructor (all input forms + invalid) + `to*` converters + `toDevice`
   (HIP + non-HIP raise) + `zeroString`.
3. The predicate matrix (all `is*` over all dtypes) — one snapshot per method.
4. Numeric reads (`numRegisters`/`numBytes`/`flopsPerMac`/`MIOutputTypeNameAbbrev`)
   + `state`/`__str__`/`__repr__`/`getAttributes`.
5. `__hash__`/`__eq__`/`__lt__` + `_populateLookupTable` (incl. synthetic dup).
6. resistance.md + recommendations.md.
7. Final coverage run + `coverage-after.txt` + no-regression confirmation.
8. Create `next-goal-<target>.md` for the target selected in
   recommendations.md (grounded in inspection) + commit. Mark this checklist
   done.

## Definition of done checklist — COMPLETE

- [x] `coverage-before.txt` captured (baseline) and committed BEFORE new tests.
      (161 stmts, 41 miss = 74.53% line; `--cov=Tensile` + grep fallback used.)
- [x] `--cov=Tensile` + grep `DataType.py` row ≥95% line — **100.00% line and
      branch** (161 stmts, 0 miss, 24 branch, 0 partial).
- [x] `coverage-after.txt` captured; before→after delta in target.md (+25.47
      pts line).
- [x] Every Tier A method has snapshot coverage (introspected predicate matrix
      + converters + numeric + state/dunder + lookup); no Tier B remained.
- [x] Invalid-input / assert paths pinned via `pytest.raises` (RuntimeError,
      KeyError, AssertionError, TypeError, both `_populateLookupTable` guards).
- [x] resistance.md: **nothing resisted** — documents why every defensive
      raise / asserted branch / table guard was reachable + out-of-scope
      `total_ordering` generated methods.
- [x] recommendations.md: GO → `Common/ValidParameters.py` next (pure table
      builders + validators), then a `Problem.py` ProblemType slice.
- [x] full `-m unit` = **1526 passed / 201 skipped**, no failures, additive only.
- [x] All work committed (atomic, no push); tree clean.
- [x] Next goal prompt `next-goal-validparameters.md` created and committed.
