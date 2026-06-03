# Next goal — characterization tests for `SolutionStructs/Utilities.py` + `LdsPadding.py`

Follow-up to the completed `Naming.py` suite. **One grouped target** covering
the two remaining pure `SolutionStructs` modules. Self-contained kickoff;
assumes the env/conventions of the seven completed targets under
`Tensile/Tests/unit/characterization/{Validators,TensileLogic,LibraryIO,DataType,ValidParameters,ProblemType,Naming}/`.

## GOAL

Add characterization suites pinning `Tensile/SolutionStructs/Utilities.py` and
`Tensile/SolutionStructs/LdsPadding.py`, each to **≥95% line coverage on the
module**, using syrupy (reuse `../survey.md`). Document resisting functions + a
go/no-go on the next target.

Achieved when:
- **BEFORE baselines captured FIRST** (one `coverage-before.txt`, both rows).
- `target.md` (new file in the test dir).
- `Tensile/Tests/unit/characterization/SolutionStructsUtils/` (and/or a second
  dir for LdsPadding — your call; one dir with two test files is fine) with
  syrupy snapshots driving every reasonably-testable function.
- Determinism: pure functions — snapshot returns. `reject` mutates `state` and
  prints; drive with `printSolutionRejectionReason=False` and snapshot the
  return + the `state["Valid"]` effect (as the `Validators` suite did).
- `pytest --cov` (path-mode + grep) reports **≥95% line** on **both** modules.
- **AFTER coverage captured** (`coverage-after.txt`, both rows) + delta in
  `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1713 passed / 201 skipped** (post-Naming
  baseline); additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created, grounded in inspection (the `Solution.py`
  multi-slice campaign is the expected next).

## Coverage command (path-mode + grep)

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit \
  | grep -E "SolutionStructs/(Utilities|LdsPadding).py|passed"
```

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

## TARGET 1: `SolutionStructs/Utilities.py` (49 stmts, baseline 46.84% line)

Imports only `sys`, `math`, `DataType`, `rocisa.enum`. Pure.

| Symbol | LOC | Tier / nature |
|---|---|---|
| `getMiInputType(kernel)` | 31-48 | A — 3 branches: `EnableF32XdlMathOp`+`UseF32XEmulation`→BFloat16; `EnableF32XdlMathOp`→`F32XdlMathOp`; neither→`DataType`. Needs a kernel dict with those flags + `ProblemType`. |
| `reject(state, printSolutionRejectionReason, *args)` | 50-82 | B — `NoReject`→False; reject (printReason=False)→`state["Valid"]=False`,True; printReason=True + `SolutionIndex==-1`→prints then True; printReason=True + valid `SolutionIndex`→`raise Exception`; `state=None`→None. Drive with printReason=False for the quiet paths; snapshot the raised message for the index path. |
| `pvar(state, field)` | 85-86 | A — `"field=value"`. |
| `roundupRatio(dividend, divisor)` | 88-89 | A — `ceil(dividend/divisor)`; parametrise exact + remainder. |
| `getRealDataTypeA` / `getRealDataTypeB` | 91-113 | A — `.value`-based mix-type mapping; parametrise over the 4 mix dtypes + a passthrough. |

## TARGET 2: `SolutionStructs/LdsPadding.py` (212 stmts, baseline 86.45% line)

Imports `functools.lru_cache`, `typing`. Pure numeric padding solvers.

| Symbol | LOC | Tier / nature |
|---|---|---|
| `get_fp4_mt_config(mt, key, miWaveTile, miWaveGroup)` | 211 | A — public selector over a computed config dict; parametrise mt/key/wave. |
| `get_fp8_mt_config(mt, key, miWaveTile, miWaveGroup)` | 225 | A — as above. |
| `get_fp16_mt_config(mt, key, miWaveGroup, ...)` | 316 | A — as above. |
| `get_fp32_mt_config(mt, key, vw, lrvw, ...)` | 392 | A — as above. |
| `get_mxs_mt_config(matrixInstK, mxBlock, vw, key)` | 411 | A — as above. |
| `_compute_*` / `_check` / `_search_padding` / `_b*_*` helpers | various | B — covered transitively by the public selectors; baseline already 86%, so the top-up is a handful of (mt, key, wave/vw) combinations that reach the currently-missed branches (L113, 147, 169, 243, 250, 301-314, 388, 405). |

### Determinism plan

- All outputs are pure ints/strings — snapshot directly; no normalisation.
- For `LdsPadding`, enumerate the valid `key`s for each `get_*_mt_config` (read
  the `_compute_*` return dict keys) and parametrise (mt, key) over a small
  representative grid; one snapshot per function as `{(mt,key,...): value}`.
- For `reject`, set `printSolutionRejectionReason=False` to avoid stdout/raises
  on the quiet paths; for the raise path build a state with a valid
  `SolutionIndex` and snapshot the exception message.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/SolutionStructsUtils/` with
  `test_utilities_char.py` + `test_ldspadding_char.py`, `pytestmark =
  pytest.mark.unit`. `target.md`/`resistance.md`/`recommendations.md`/
  `coverage-before.txt`/`coverage-after.txt` as new files; snapshots in the
  dir's `__snapshots__/`.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` (both rows) + `target.md`.
2. `test_utilities_char.py` (Utilities to ≥95%).
3. `test_ldspadding_char.py` (LdsPadding top-up to ≥95%).
4. resistance.md + recommendations.md.
5. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
6. `next-goal-<target>.md` (the `Solution.py` slice campaign, grounded) + commit.
   Mark this checklist done.

## Definition of done checklist — COMPLETE

- [x] `coverage-before.txt` captured (baseline, both rows) BEFORE new tests.
      (Utilities 49/23 = 53.06% line; LdsPadding 212/21 = 90.09% line.)
- [x] `--cov=Tensile` + grep: **both** ≥95% line — **Utilities 100%**,
      **LdsPadding 100% line**.
- [x] `coverage-after.txt` captured; before→after delta in target.md
      (+46.94 / +9.91 pts).
- [x] Every Tier A/B function has snapshot coverage; the defensive LdsPadding
      branches unreachable via the public selectors are characterized via direct
      private-helper tests (documented in resistance.md).
- [x] `reject` raise/quiet paths pinned; `getMiInputType` all 3 branches.
- [x] resistance.md: Utilities nothing resisted; LdsPadding 2 residual partial
      branches + the public-API-unreachable defensive code (direct-tested).
- [x] recommendations.md: GO → `Solution.py` support-functions slice next; then
      the cap-coupled `Solution` class slices; defer `GlobalParameters.py`.
- [x] full `-m unit` = **1769 passed / 201 skipped**, no failures, additive only.
- [x] All work committed (atomic, no push); tree clean.
- [x] Next goal prompt `next-goal-solution-support.md` created and committed.
