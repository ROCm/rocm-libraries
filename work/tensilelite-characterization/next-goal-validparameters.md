# Next goal — characterization tests for `Tensile/Common/ValidParameters.py`

Follow-up to the completed `DataType.py` suite. Same pattern, next module.
This file is the self-contained kickoff; it assumes the env and conventions
already established (see `prompt.md`, `env/README.md`, `HANDOFF.md`, and the
four completed targets under
`Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO,DataType}/`).

## GOAL

Add a characterization-test suite that pins current behaviour of
`Tensile/Common/ValidParameters.py` with **≥95% line coverage on the module**,
using syrupy (reuse `Tensile/Tests/unit/characterization/survey.md`, no new
survey), integrated with the existing pytest/tox setup. Document resisting
functions + a go/no-go on the *next* target after this one.

Achieved when:
- **Baseline (BEFORE) coverage captured FIRST** — `ValidParameters.py` coverage
  from the *existing* suite — saved as `coverage-before.txt` (command below).
- `target.md` for ValidParameters (new file in the test dir, per add-only).
- `Tensile/Tests/unit/characterization/ValidParameters/` exists with syrupy
  snapshots driving every reasonably-testable public function (see inventory).
- Determinism handled: every output is a pure function of the inputs (no
  filesystem/toolchain/GPU). The `makeValid*` builders are `lru_cache`d and
  return large structures — snapshot whole where reviewable, else a normalised
  summary (counts + a sorted sample); the validators snapshot their raised
  messages via `pytest.raises`.
- `pytest --cov` (path-mode + grep) reports **≥95% line** on the module.
- **Final (AFTER) coverage captured** (`coverage-after.txt`) with before→after
  delta recorded in `target.md`.
- `resistance.md` (new file) lists every resisting fn + reason + workaround.
- `recommendations.md` (new file): go/no-go on the *following* target
  (a `SolutionStructs/Problem.py` ProblemType slice vs `Common/GlobalParameters.py`)
  + effort estimate.
- No regression: full `-m unit` still passes (current baseline after the
  DataType suite = **1526 passed / 201 skipped**; this work only adds).
- **The work is committed** (atomic commits; tree clean; no push).
- **The next goal prompt is created**: `next-goal-<target>.md` for the target
  chosen in recommendations.md, grounded in inspection of that module.

## Coverage reports (BEFORE / AFTER) — required deliverable

`--cov` on a single .py path-prefix does NOT match (coverage warns
"module-not-imported / no data"); use `--cov=Tensile` + grep the row, exactly
as the DataType/LibraryIO targets did.

```
# BEFORE (existing tests only) and AFTER (suite complete):
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "ValidParameters.py|passed"
```

> Confirm you grep the `Tensile/Common/ValidParameters.py` row specifically.

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**: only add new files. Never modify/delete any existing file.
  Anything needing an edit → new file or document as a limitation. Keep
  `target.md`/`resistance.md`/`recommendations.md` as **new files in the
  per-target test dir**, not edits to the shared ones.
- **Commit frequently**, atomic commits. **Never push / no PRs.** Local only.
- rocm-libraries boundary: stay within `projects/hipblaslt/tensilelite`.
- Snapshot pure returns / normalised summaries; no nondeterministic blobs
  (there are none here — the module is pure).
- BOUND: 150 turns / 150 min. AUTONOMY: don't ask for confirmation; document
  blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree at /work).
  rocisa built (`LD_LIBRARY_PATH=/opt/rocm/lib`, baked).
- **Install syrupy if the container was recreated** (`pip install syrupy`).
- Resume: see `HANDOFF.md`.
- Run/snapshot/coverage via `docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite
  -w /work/projects/hipblaslt/tensilelite tl-char pytest ...` (the DataType
  round used exactly this; baseline+after each take ~110s for the full suite,
  <15s for the suite alone).
- **Coverage gotcha: pass `--cov` a PATH (`Tensile`), never a dotted module** —
  a dotted name re-imports rocisa (nanobind) → SIGABRT.
- Line coverage = `(Stmts - Miss)/Stmts` (report both line and blended).

## TARGET: `Tensile/Common/ValidParameters.py` (1136 LOC)

**Pure and self-contained** — imports only `math`, `functools.lru_cache`,
`.Architectures.SUPPORTED_ISA`, `.Types.IsaVersion`. No toolchain, no
subprocess, no `asmCaps`, no GPU. Note the LOC is dominated by the giant
`validParameters` literal dict (one assignment statement spanning ~L228-1076,
plus inline comment diagrams), so the **statement** count is far below 1136 —
the import-time dict assignment is covered for free, and the real testable
surface is the `makeValid*` builders + the three validators.

### Public API inventory (drive each; note the tier)

| Symbol | LOC area | Tier / nature |
|---|---|---|
| `makeValidWorkGroups()` | 82-93 (`@lru_cache`) | A — builds the list of valid `[x,y,z]` workgroups; snapshot whole (small). |
| `makeValidWMMA()` | 94-96 | A — pure list builder; snapshot. |
| `makeValidSWMMAC()` | 97-99 | A — pure list builder; snapshot. |
| `makeValidMFMA()` | 100-174 (`@lru_cache`) | A — returns a dict of MI lists keyed by dtype combo (`HH`/`SS`/`BB`/`DD`/`B1k`/`XX`/`_format9`); large — snapshot a **normalised summary** (sorted keys + per-key len + a few sample rows) and/or the full dict if the `.ambr` stays reviewable. |
| `makeValidSMFMA()` | 175-206 (`@lru_cache`) | A — same shape (`HH`/`BB`/`4xi84xi8`/`_format9`); summary snapshot. |
| `makeValidMatrixInstructions()` | 207-226 (`@lru_cache`) | A — concatenates the above into the master MI list; snapshot len + head/tail + that it starts with `[[], [-1]]`. |
| `validParameters` (module dict) | 228-1076 | A — assigned at import (covered for free); add a snapshot of `sorted(validParameters.keys())` to pin the parameter roster, and spot-check a few values. |
| `checkSpaceFillAlgoIsValid(name, value)` | 1077-1090 | A — accept/reject; snapshot raises for invalid algo names. |
| `checkSpaceFillAlgoWGMIsValid(name, value)` | 1091-1108 | A — accept/reject; snapshot raises. |
| `checkParametersAreValid(param, validParams)` | 1109-1136 | A — the central validator. Branches: `ProblemSizes`/`InternalSupportParams` early returns; unknown-name `Exception`; out-of-range-value `Exception` (incl. the `>32`-combos message variant); the `SpaceFillingAlgo`/`SFCWGM` sub-validator dispatch; the `-1` (any-value) accept path. Parametrize all. |

### Determinism plan

1. **`lru_cache`.** The builders are cached; that is fine for snapshotting
   (same input → same output). No reset needed; just call and snapshot.
2. **Large structures.** For `makeValidMFMA`/`SMFMA`/`MatrixInstructions`,
   prefer a normalised summary (sorted keys, per-key counts, a deterministic
   small sample) so the `.ambr` is reviewable; snapshot the full structure only
   if it stays manageable. Document the choice in `target.md`.
3. **Validators.** Snapshot the raised exception **message** via
   `pytest.raises` for each reject branch; for accept paths snapshot the return
   (mostly `None`) and rely on "did not raise". Use `validParameters` itself as
   the `validParams` arg for realistic accept cases, and a tiny synthetic
   `validParams` for targeted reject branches (e.g. a key whose value list is
   `> 32` long to hit the truncated-message variant; a key set to `-1` for the
   any-value accept).
4. **No env coupling** — nothing reads paths/time/version/GPU.

### Location & layout (consistent with prior targets)

- Suite: `Tensile/Tests/unit/characterization/ValidParameters/`, files marked
  `pytestmark = pytest.mark.unit`. Collected by the existing
  `testpaths=Tensile/Tests` — no config edit.
- `target.md`, `resistance.md`, `recommendations.md`, `coverage-before.txt`,
  `coverage-after.txt` as **new files in that dir**; snapshots in
  `ValidParameters/__snapshots__/` via `--snapshot-update`.
- No `conftest.py` needed (no toolchain/caps fixtures) unless a helper emerges.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` (baseline) + `target.md`.
2. `makeValid*` builders (workgroups/WMMA/SWMMAC/MFMA/SMFMA/MatrixInstructions)
   + the `validParameters` key-roster snapshot.
3. The three validators (`checkParametersAreValid` accept/reject matrix +
   `checkSpaceFillAlgo*`).
4. resistance.md + recommendations.md.
5. Final coverage run + `coverage-after.txt` + no-regression confirmation.
6. Create `next-goal-<target>.md` for the target selected in
   recommendations.md (grounded in inspection) + commit. Mark this checklist
   done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) and committed BEFORE new tests.
- [ ] `--cov=Tensile` + grep `ValidParameters.py` row ≥95% line.
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Every Tier A function has snapshot coverage; any Tier B documented.
- [ ] Validator reject paths pinned via `pytest.raises` (incl. the `>32`-combos
      message variant and the `SpaceFillingAlgo`/`SFCWGM` sub-validators).
- [ ] resistance.md lists each resisting fn (expect few/none — pure module).
- [ ] recommendations.md: go/no-go on the next target (`Problem.py` ProblemType
      slice vs `Common/GlobalParameters.py`).
- [ ] full `-m unit` ≥ 1526 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
